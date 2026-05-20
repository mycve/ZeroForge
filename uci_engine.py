#!/usr/bin/env python3
"""
ZeroForge UCI 引擎（高性能 + JAX 编译缓存）
"""

import os
import sys
import time
import gc
import argparse
import logging
import threading
from functools import partial

import numpy as np

# ==========================================================
# JAX 性能设置（必须在 import jax 前）
# ==========================================================

os.environ["JAX_COMPILATION_CACHE_DIR"] = os.path.expanduser(
    "~/.cache/zeroforge_jax"
)

os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "1"
os.environ["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] = "0"

# 避免 GPU 一次性占满显存
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

# CPU 模式
if "--cpu" in sys.argv:
    os.environ["JAX_PLATFORMS"] = "cpu"

# 抑制 JAX/第三方库启动日志，避免污染 UCI 输出、影响 GUI 解析
os.environ["JAX_LOGGING_LEVEL"] = "ERROR"
# 抑制 XLA C++ 层 glog 输出（如 cpu_aot_loader E0311 的 prefer-no-scatter 等）
os.environ.setdefault("GLOG_minloglevel", "3")  # 3=FATAL only，屏蔽 INFO/WARNING/ERROR
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

# 某些 XLA/C++ 日志会直接写到进程 stderr，绕过 Python logging。
# 为避免污染 UCI GUI 控制台和协议交互，这里在导入 JAX 前把 stderr 重定向到独立文件。
_BOOT_LOG_DIR = os.path.dirname(os.path.abspath(__file__))
_STDERR_LOG_FILE = os.path.join(_BOOT_LOG_DIR, "zeroforge_uci_stderr.log")
try:
    _stderr_fd = os.open(_STDERR_LOG_FILE, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    os.dup2(_stderr_fd, 2)
    os.close(_stderr_fd)
    sys.stderr = open(2, "w", buffering=1, encoding="utf-8", errors="replace", closefd=False)
except OSError:
    pass

# ==========================================================

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import mctx

from xiangqi.env import XiangqiEnv, NUM_OBSERVATION_CHANNELS
from xiangqi.actions import (
    move_to_action,
    action_to_move,
    move_to_uci,
    uci_to_move,
    ACTION_SPACE_SIZE,
    rotate_action,
)
from xiangqi.fen import parse_fen
from networks.alphazero import AlphaZeroNetwork

# 抑制第三方库日志，UCI 协议要求 stdout 仅输出协议行
for _name in ("jax", "jax._src", "orbax", "mctx", "absl"):
    _lg = logging.getLogger(_name)
    _lg.setLevel(logging.ERROR)

# ==========================================================
# 常量
# ==========================================================

STARTING_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"

_ROTATED_IDX = rotate_action(jnp.arange(ACTION_SPACE_SIZE))

MCTS_QTRANSFORM = partial(
    mctx.qtransform_completed_by_mix_value,
    value_scale=0.1,
)

DEFAULT_NUM_SIMULATIONS = 2048
DEFAULT_TOP_K = 32

# 日志写入磁盘（与 uci_engine.py 同目录），走法等信息由 send() 输出到 stdout
_LOG_DIR = os.path.dirname(os.path.abspath(__file__))
_LOG_FILE = os.path.join(_LOG_DIR, "zeroforge_uci.log")
logger = logging.getLogger("zeroforge_uci")
logger.setLevel(logging.DEBUG)
logger.handlers.clear()
logger.propagate = False
try:
    _fh = logging.FileHandler(_LOG_FILE, encoding="utf-8")
    _fh.setLevel(logging.DEBUG)
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_fh)
except OSError as e:
    _fh = logging.StreamHandler(sys.stderr)
    _fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_fh)
    logger.warning("无法创建日志文件 %s，改用 stderr: %s", _LOG_FILE, e)

# ==========================================================
# Engine
# ==========================================================


class Engine:
    def __init__(self, ckpt_dir, step=0):

        self.ckpt_dir = os.path.abspath(ckpt_dir)
        self.step = step

        self.env = XiangqiEnv()

        self.params = None
        self.net = None

        self._infer = None
        self._recurrent_fn = None

        self.rng = jax.random.PRNGKey(0)
        self._load_lock = threading.Lock()
        self._load_done = threading.Event()
        self._load_thread = None
        self._load_error = ""
        self._search_count = 0

    # ------------------------------------------------------

    def reset_runtime(self):

        with self._load_lock:
            self.params = None
            self.net = None
            self._infer = None
            self._recurrent_fn = None
            self._load_error = ""
            self._load_done.clear()
            self._load_thread = None

    # ------------------------------------------------------

    def configure(self, ckpt_dir=None, step=None):

        changed = False

        if ckpt_dir is not None:
            new_ckpt_dir = os.path.abspath(ckpt_dir)
            if new_ckpt_dir != self.ckpt_dir:
                self.ckpt_dir = new_ckpt_dir
                changed = True

        if step is not None and step != self.step:
            self.step = step
            changed = True

        if changed:
            logger.info("更新加载配置 ckpt_dir=%s step=%s", self.ckpt_dir, self.step)
            self.reset_runtime()

        return changed

    # ------------------------------------------------------

    def _infer_arch(self, params):

        channels = 128
        blocks = 0

        try:

            p = params.get("params", params)

            if "feature_embed" in p:
                channels = max(int(p["feature_embed"].shape[-1]) // 4, 1)
                return channels, 0

        except Exception:
            pass

        return channels, blocks

    # ------------------------------------------------------

    def _build_runtime(self, params):

        channels, blocks = self._infer_arch(params)

        self.net = AlphaZeroNetwork(
            action_space_size=ACTION_SPACE_SIZE,
            channels=channels,
            num_blocks=blocks,
        )

        self.params = params

        self._infer = jax.jit(
            lambda p, x: self.net.apply({"params": p}, x, train=False)
        )

        def recurrent_fn(params, rng_key, action, state):

            prev_player = state.current_player

            next_state = jax.vmap(self.env.step)(state, action)

            obs = jax.vmap(self.env.observe)(next_state)

            logits, value, *_ = self._infer(params, obs)

            logits = jnp.where(
                next_state.current_player[:, None] == 0,
                logits,
                logits[:, _ROTATED_IDX],
            )

            logits = logits - jnp.max(logits, axis=-1, keepdims=True)

            logits = jnp.where(
                next_state.legal_action_mask,
                logits,
                jnp.finfo(logits.dtype).min,
            )

            return (
                mctx.RecurrentFnOutput(
                    reward=next_state.rewards[
                        jnp.arange(next_state.rewards.shape[0]), prev_player
                    ],
                    discount=jnp.where(next_state.terminated, 0.0, -1.0),
                    prior_logits=logits,
                    value=value,
                ),
                next_state,
            )

        self._recurrent_fn = jax.jit(recurrent_fn)

    # ------------------------------------------------------

    def _warmup(self):

        dummy = jnp.zeros((1, NUM_OBSERVATION_CHANNELS, 10, 9), dtype=jnp.float32)
        self._infer(self.params, dummy)

        state = self.env.init(jax.random.PRNGKey(0))
        action = jnp.array([int(jnp.argmax(state.legal_action_mask))], dtype=jnp.int32)
        batched_state = jax.tree_util.tree_map(lambda x: x[None], state)
        self._recurrent_fn(self.params, jax.random.PRNGKey(1), action, batched_state)

    # ------------------------------------------------------

    def _load_impl(self):

        ckpt_manager = ocp.CheckpointManager(self.ckpt_dir)

        step = self.step or ckpt_manager.latest_step()

        if step is None:
            self._load_error = "未找到 checkpoint"
            return False

        try:
            restored = ckpt_manager.restore(step)
        except Exception:
            restored = ocp.StandardCheckpointer().restore(
                os.path.join(self.ckpt_dir, str(step))
            )

        params = None

        if isinstance(restored, dict):
            params = restored.get("params") or (
                restored.get("default") or {}
            ).get("params")

        if params is None:
            self._load_error = "checkpoint 无 params"
            return False

        self._build_runtime(params)
        self._warmup()
        logger.info("模型加载完成 step=%s", step)
        self._load_error = ""
        return True

    # ------------------------------------------------------

    def _load_worker(self):

        ok = False
        try:
            logger.info("后台加载线程启动")
            ok = self._load_impl()
        except Exception as e:
            self._load_error = str(e)
        finally:
            self._load_done.set()

    # ------------------------------------------------------

    def start_background_load(self):

        with self._load_lock:
            if self.params is not None:
                self._load_done.set()
                return
            if self._load_thread and self._load_thread.is_alive():
                return
            self._load_error = ""
            self._load_done.clear()
            self._load_thread = threading.Thread(
                target=self._load_worker,
                name="zeroforge-uci-loader",
                daemon=True,
            )
            self._load_thread.start()
            logger.info("已启动后台预加载")

    # ------------------------------------------------------

    def wait_until_ready(self, timeout=None):

        self.start_background_load()
        done = self._load_done.wait(timeout)
        return done and self.params is not None

    # ------------------------------------------------------

    def is_loading(self):
        return self._load_thread is not None and self._load_thread.is_alive()

    # ------------------------------------------------------

    def load(self):

        if self.params is not None:
            return True

        self.start_background_load()
        self._load_done.wait()
        return self.params is not None

    # ------------------------------------------------------

    def get_best_move(self, state, simulations, top_k):

        if self._infer is None:
            raise RuntimeError("engine not loaded")

        legal_mask = state.legal_action_mask

        legal_count = int(jnp.sum(legal_mask))

        if legal_count == 0:
            return None, 0.0, []

        # 仅有一个合法走法时直接返回，无需推理和 MCTS
        if legal_count == 1:
            action = int(jnp.argmax(legal_mask))
            from_sq, to_sq = action_to_move(action)
            move = move_to_uci(int(from_sq), int(to_sq))
            return move, 0.0, [{"uci": move, "weight": 1.0, "visits": 0, "cp": 0}]

        obs = self.env.observe(state)[None]

        logits, value, *_ = self._infer(self.params, obs)

        if state.current_player == 1:
            logits = logits[:, _ROTATED_IDX]

        logits = logits - jnp.max(logits, axis=-1, keepdims=True)

        logits = jnp.where(
            legal_mask,
            logits,
            jnp.finfo(logits.dtype).min,
        )

        self.rng, sk = jax.random.split(self.rng)

        root = mctx.RootFnOutput(
            prior_logits=logits,
            value=value,
            embedding=jax.tree_util.tree_map(
                lambda x: jnp.expand_dims(x, 0),
                state,
            ),
        )

        policy = mctx.gumbel_muzero_policy(
            params=self.params,
            rng_key=sk,
            root=root,
            recurrent_fn=self._recurrent_fn,
            num_simulations=simulations,
            max_num_considered_actions=top_k,
            invalid_actions=~legal_mask[None],
            qtransform=MCTS_QTRANSFORM,
            gumbel_scale=0.0,
        )

        weights = np.array(policy.action_weights[0])
        summary = policy.search_tree.summary()
        visit_counts = np.array(summary.visit_counts[0])
        qvalues = np.array(summary.qvalues[0])

        action = int(jnp.argmax(weights))

        from_sq, to_sq = action_to_move(action)

        move = move_to_uci(int(from_sq), int(to_sq))

        val = float(policy.search_tree.node_values[0, 0])

        # 构建 Top-N 候选（含 visits、qvalue，用于 UCI multipv）
        top_n = 32
        top_indices = np.argsort(weights)[::-1][:top_n]
        policy_info = []
        for idx in top_indices:
            w = float(weights[idx])
            if w <= 1e-6:
                break
            fs, ts = action_to_move(int(idx))
            visits = int(visit_counts[idx])
            qval = float(qvalues[idx])
            cp = int(qval * 1000)
            uci_move = move_to_uci(int(fs), int(ts))
            policy_info.append({
                "uci": uci_move,
                "weight": w,
                "visits": visits,
                "cp": cp,
            })

        del summary
        del policy
        del root

        self._search_count += 1
        if self._search_count % 16 == 0:
            gc.collect()

        return move, val, policy_info


# ==========================================================
# FEN
# ==========================================================


def state_to_fen(state) -> str:
    """将 XiangqiState 转为 FEN 字符串（用于日志）"""
    board = np.array(state.board)
    player = int(state.current_player)
    PIECE_TO_FEN = {
        1: "K", 2: "A", 3: "B", 4: "N", 5: "R", 6: "C", 7: "P",
        -1: "k", -2: "a", -3: "b", -4: "n", -5: "r", -6: "c", -7: "p",
    }
    rows = []
    for r in range(9, -1, -1):
        r_str, empty = "", 0
        for c in range(9):
            p = int(board[r, c])
            if p == 0:
                empty += 1
            else:
                if empty > 0:
                    r_str += str(empty)
                    empty = 0
                r_str += PIECE_TO_FEN.get(p, "?")
        if empty > 0:
            r_str += str(empty)
        rows.append(r_str)
    return "/".join(rows) + (" w" if player == 0 else " b")


def build_state(env, fen, moves):
    """从 FEN 和走法列表构建局面，非法走法直接跳过。"""
    board, player = parse_fen(fen)
    state = env.init_from_board(
        jnp.array(board, dtype=jnp.int8),
        jnp.int32(player),
    )
    for m in moves:
        m = (m or "").strip()
        if len(m) < 4:
            continue
        try:
            f, t = uci_to_move(m[:4])
            action = int(move_to_action(jnp.int32(f), jnp.int32(t)))
            if action < 0:
                continue
            state = env.step(state, jnp.int32(action))
            if state.terminated:
                break
        except (ValueError, IndexError, KeyError):
            continue
    return state


def parse_setoption(parts):
    """解析 UCI setoption，支持多词 option name/value。"""
    if "name" not in parts:
        return "", None
    name_idx = parts.index("name") + 1
    if "value" in parts:
        value_idx = parts.index("value")
        name = " ".join(parts[name_idx:value_idx]).strip()
        value = " ".join(parts[value_idx + 1 :]).strip()
    else:
        name = " ".join(parts[name_idx:]).strip()
        value = None
    return name, value


# ==========================================================
# UCI LOOP
# ==========================================================


def run_uci(engine, simulations, top_k):

    state = None
    last_fen = STARTING_FEN
    last_moves = []
    engine_move_count = 0  # 本局引擎已走次数（每输出一次 bestmove 计 1）
    opts = {
        "step": engine.step,
        "simulations": simulations,
        "top_k": top_k,
        "multipv": 1,
    }
    logger.info(
        "UCI 引擎启动 simulations=%s top_k=%s multipv=%s 日志文件: %s",
        simulations,
        top_k,
        opts["multipv"],
        _LOG_FILE,
    )

    def send(x):
        print(x, flush=True)

    for line in sys.stdin:
        cmd = ""
        try:
            raw_line = line.strip()
            if raw_line:
                logger.debug("收到命令: %s", raw_line)
            parts = line.strip().split()
            if not parts:
                continue
            cmd = parts[0]

            if cmd == "uci":
                send("id name ZeroForge")
                send("id author ZeroForge")
                send(f"option name Step type spin default {opts['step']} min 0 max 100000000")
                send(f"option name Simulations type spin default {opts['simulations']} min 16 max 4096")
                send(f"option name TopK type spin default {opts['top_k']} min 4 max 64")
                send(f"option name MultiPV type spin default 1 min 1 max 32")
                send("uciok")

            elif cmd == "setoption":
                name, val = parse_setoption(parts)
                if name == "Step" and val and val.isdigit():
                    opts["step"] = max(0, int(val))
                    engine.configure(step=opts["step"])
                elif name == "Simulations" and val and val.isdigit():
                    opts["simulations"] = max(16, min(4096, int(val)))
                elif name == "TopK" and val and val.isdigit():
                    opts["top_k"] = max(4, min(64, int(val)))
                elif name == "MultiPV" and val and val.isdigit():
                    opts["multipv"] = max(1, min(32, int(val)))
                elif name:
                    logger.info("收到未实现选项 %s=%s，忽略", name, val)
                if name in {"Step", "Simulations", "TopK", "MultiPV"}:
                    logger.info(
                        "设置选项 %s=%s -> step=%s simulations=%s top_k=%s multipv=%s",
                        name,
                        val,
                        opts["step"],
                        opts["simulations"],
                        opts["top_k"],
                        opts["multipv"],
                    )

            elif cmd == "isready":
                if engine.params is None:
                    logger.info("收到 isready，等待后台加载完成 loading=%s", engine.is_loading())
                    engine.wait_until_ready()
                send("readyok")

            elif cmd == "position":
                fen = STARTING_FEN
                moves = []
                if len(parts) < 2:
                    continue
                if parts[1] == "startpos":
                    if "moves" in parts:
                        i = parts.index("moves")
                        moves = parts[i + 1 :]
                elif parts[1] == "fen":
                    if "fen" not in parts:
                        continue
                    i = parts.index("fen") + 1
                    j = parts.index("moves") if "moves" in parts else len(parts)
                    fen = " ".join(parts[i:j])
                    if "moves" in parts:
                        moves = parts[j + 1 :]
                try:
                    state = build_state(engine.env, fen, moves)
                    last_fen = fen
                    last_moves = moves
                    half_moves = len(moves)
                    full_moves = half_moves // 2  # 1回合 = 红1着+黑1着
                    logger.info(
                        "局面更新 总着数=%s着(约%s回合) 着法=[%s]",
                        half_moves,
                        full_moves,
                        " ".join(moves[:20]) + (" ..." if len(moves) > 20 else ""),
                    )
                except (ValueError, KeyError):
                    state = build_state(engine.env, STARTING_FEN, [])
                    last_fen = STARTING_FEN
                    last_moves = []

            elif cmd == "go":
                depth = None
                if "depth" in parts:
                    try:
                        depth_idx = parts.index("depth")
                        depth = int(parts[depth_idx + 1])
                    except (ValueError, IndexError):
                        depth = None
                if not engine.wait_until_ready():
                    err = engine._load_error or "model load failed"
                    send(f"info string {err}")
                    send("bestmove 0000")
                    continue
                if state is None:
                    state = build_state(engine.env, STARTING_FEN, [])
                    last_fen = STARTING_FEN
                    last_moves = []
                half_moves = len(last_moves)
                full_moves = half_moves // 2
                sims, tk = opts["simulations"], opts["top_k"]
                legal_count = int(jnp.sum(state.legal_action_mask))
                current_fen = state_to_fen(state)
                logger.info(
                    "======== 开始搜索 ======== 本局引擎第%s次走棋 对局进度=%s着(约%s回合) depth=%s sims=%s top_k=%s legal=%s",
                    engine_move_count + 1,
                    half_moves,
                    full_moves,
                    depth,
                    sims,
                    tk,
                    legal_count,
                )
                logger.info("局面 FEN: %s", current_fen)
                logger.info("历史着法: %s", " ".join(last_moves) if last_moves else "(无)")
                t0 = time.perf_counter()
                move, val, policy_info = engine.get_best_move(state, sims, tk)
                elapsed_ms = int((time.perf_counter() - t0) * 1000)
                nps = (sims * 1000 // elapsed_ms) if elapsed_ms > 0 else 0
                if move:
                    engine_move_count += 1
                    cp = int(val * 1000)
                    logger.info(
                        "======== 搜索完成 ======== [本局引擎第%s步] bestmove=%s value=%.4f(cp=%s) elapsed_ms=%s nps=%s",
                        engine_move_count, move, val, cp, elapsed_ms, nps,
                    )
                    if policy_info:
                        top_str = " ".join(f"{p['uci']}({p['weight']:.3f})" for p in policy_info[:6])
                        logger.info("策略 Top-N: %s", top_str)
                    # UCI: 在 bestmove 前发送最终 info，multipv 模式下发送所有 k 条变例
                    multipv_k = min(opts["multipv"], len(policy_info)) if policy_info else 1
                    for i in range(multipv_k):
                        if policy_info and i < len(policy_info):
                            p = policy_info[i]
                            pv_cp, pv_move = p["cp"], p["uci"]
                        else:
                            pv_cp, pv_move = cp, move
                        send(
                            f"info depth 1 seldepth 1 multipv {i+1} score cp {pv_cp} "
                            f"nodes {sims} nps {nps} hashfull 0 tbhits 0 time {elapsed_ms} pv {pv_move}"
                        )
                    send(f"bestmove {move}")
                else:
                    send("bestmove 0000")

            elif cmd == "ucinewgame":
                state = None
                engine_move_count = 0
                logger.info("收到 ucinewgame，已重置内部局面")

            elif cmd == "stop":
                logger.info("收到 stop，但当前搜索为同步阻塞式，忽略")

            elif cmd == "quit":
                logger.info("收到 quit，退出 UCI 循环")
                break

        except Exception:
            if cmd == "go":
                try:
                    send("info string error")
                    send("bestmove 0000")
                except Exception:
                    pass


# ==========================================================
# MAIN
# ==========================================================


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--ckpt", default="checkpoints_nnue")
    parser.add_argument("--step", type=int, default=0)

    parser.add_argument("--simulations", type=int, default=DEFAULT_NUM_SIMULATIONS)
    parser.add_argument("--topk", type=int, default=DEFAULT_TOP_K)

    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()

    engine = Engine(args.ckpt, args.step)
    run_uci(engine, args.simulations, args.topk)

#!/usr/bin/env python3
"""
ZeroForge UCI 引擎（高性能 + JAX 编译缓存）
"""

import os
import sys
import time
import argparse
import logging
import threading
from functools import partial

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

DEFAULT_NUM_SIMULATIONS = 512
DEFAULT_TOP_K = 32
DEFAULT_THREADS = 1
DEFAULT_HASH_MB = 16

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

    # ------------------------------------------------------

    def _infer_arch(self, params):

        channels = 128
        blocks = 8

        try:

            p = params.get("params", params)

            if "input_proj" in p:
                channels = int(p["input_proj"]["kernel"].shape[-1])

            blocks = len([k for k in p.keys() if str(k).startswith("GraphBlock_")])

            if blocks <= 0:
                blocks = 8

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

            logits, value, _ = self._infer(params, obs)

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
            logger.error(self._load_error)
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
            logger.error(self._load_error)
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
            logger.exception("后台加载失败: %s", e)
        finally:
            if not ok and self.params is None and self._load_error:
                logger.error("模型未就绪: %s", self._load_error)
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
            return None, 0.0

        top_k = min(top_k, legal_count)

        obs = self.env.observe(state)[None]

        logits, value, _ = self._infer(self.params, obs)

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

        weights = policy.action_weights[0]

        action = int(jnp.argmax(weights))

        from_sq, to_sq = action_to_move(action)

        move = move_to_uci(int(from_sq), int(to_sq))

        val = float(policy.search_tree.node_values[0, 0])

        if state.current_player == 1:
            val = -val

        return move, val


# ==========================================================
# FEN
# ==========================================================


def build_state(env, fen, moves):
    """从 FEN 和走法列表构建局面，非法走法会跳过并记录"""
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
                logger.error("非法走法(动作-1): %s", m[:4])
                continue
            state = env.step(state, jnp.int32(action))
            if state.terminated:
                break
        except (ValueError, IndexError, KeyError) as e:
            logger.error("解析走法失败 %r: %s", m[:4], e)
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
    opts = {
        "simulations": simulations,
        "top_k": top_k,
        "threads": DEFAULT_THREADS,
        "hash": DEFAULT_HASH_MB,
    }
    logger.info("UCI 引擎启动 simulations=%s top_k=%s 日志文件: %s", simulations, top_k, _LOG_FILE)
    engine.start_background_load()

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
                engine.start_background_load()
                send("id name ZeroForge")
                send("id author ZeroForge")
                send(f"option name Simulations type spin default {opts['simulations']} min 16 max 4096")
                send(f"option name TopK type spin default {opts['top_k']} min 4 max 64")
                send(f"option name Threads type spin default {opts['threads']} min 1 max 1")
                send(f"option name Hash type spin default {opts['hash']} min 1 max 1024")
                send("option name Clear Hash type button")
                send("uciok")

            elif cmd == "setoption":
                name, val = parse_setoption(parts)
                if name == "Simulations" and val and val.isdigit():
                    opts["simulations"] = max(16, min(4096, int(val)))
                elif name == "TopK" and val and val.isdigit():
                    opts["top_k"] = max(4, min(64, int(val)))
                elif name == "Threads" and val and val.isdigit():
                    requested = int(val)
                    opts["threads"] = 1
                    logger.info("设置选项 Threads=%s -> 当前实现固定单线程", requested)
                elif name == "Hash" and val and val.isdigit():
                    opts["hash"] = max(1, min(1024, int(val)))
                    logger.info("设置选项 Hash=%sMB -> 当前实现无置换表，仅记录该值", opts["hash"])
                elif name == "Clear Hash":
                    logger.info("收到 Clear Hash -> 当前实现无置换表，忽略")
                elif name:
                    logger.info("收到未实现选项 %s=%s，忽略", name, val)
                if name in {"Simulations", "TopK"}:
                    logger.info(
                        "设置选项 %s=%s -> simulations=%s top_k=%s threads=%s hash=%s",
                        name,
                        val,
                        opts["simulations"],
                        opts["top_k"],
                        opts["threads"],
                        opts["hash"],
                    )

            elif cmd == "isready":
                if engine.params is None:
                    logger.info("收到 isready，等待后台加载完成 loading=%s", engine.is_loading())
                    if not engine.wait_until_ready():
                        logger.error("isready 等待结束但模型未就绪: %s", engine._load_error)
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
                    logger.info("局面更新 fen=%s moves=%s", fen, len(moves))
                except (ValueError, KeyError) as e:
                    logger.error("build_state 失败: %s", e)
                    state = build_state(engine.env, STARTING_FEN, [])

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
                sims, tk = opts["simulations"], opts["top_k"]
                logger.info("开始搜索 depth=%s simulations=%s top_k=%s", depth, sims, tk)
                t0 = time.perf_counter()
                move, val = engine.get_best_move(state, sims, tk)
                elapsed_ms = int((time.perf_counter() - t0) * 1000)
                if move:
                    cp = int(val * 1000)
                    logger.info("搜索完成 move=%s value=%.4f elapsed_ms=%s", move, val, elapsed_ms)
                    send(
                        f"info depth 1 seldepth 1 multipv 1 score cp {cp} "
                        f"nodes {sims} nps 0 hashfull 0 tbhits 0 time {elapsed_ms} pv {move}"
                    )
                    send(f"bestmove {move}")
                else:
                    logger.warning("搜索无合法着法 elapsed_ms=%s", elapsed_ms)
                    send("bestmove 0000")

            elif cmd == "ucinewgame":
                state = None
                logger.info("收到 ucinewgame，已重置内部局面")

            elif cmd == "stop":
                logger.info("收到 stop，但当前搜索为同步阻塞式，忽略")

            elif cmd == "quit":
                logger.info("收到 quit，退出 UCI 循环")
                break

        except Exception as e:
            logger.exception("UCI 命令处理异常 cmd=%s: %s", cmd, e)
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

    parser.add_argument("--ckpt", default="checkpoints")
    parser.add_argument("--step", type=int, default=0)

    parser.add_argument("--simulations", type=int, default=DEFAULT_NUM_SIMULATIONS)
    parser.add_argument("--topk", type=int, default=DEFAULT_TOP_K)

    parser.add_argument("--cpu", action="store_true")

    args = parser.parse_args()

    engine = Engine(args.ckpt, args.step)
    run_uci(engine, args.simulations, args.topk)

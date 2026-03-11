#!/usr/bin/env python3
"""
ZeroForge UCI 协议引擎
通过 stdin/stdout 与 GUI 通信，使用 UCI 命令（uci / uciok）
"""

import os
import sys
import argparse
import logging
from functools import partial

# 在 import jax 前设置
if "--cpu" in sys.argv:
    os.environ["JAX_PLATFORMS"] = "cpu"

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import mctx

from xiangqi.env import XiangqiEnv, XiangqiState, NUM_OBSERVATION_CHANNELS
from xiangqi.actions import (
    move_to_action, action_to_move, move_to_uci, uci_to_move,
    ACTION_SPACE_SIZE, rotate_action,
)
from xiangqi.fen import parse_fen

from networks.alphazero import AlphaZeroNetwork

# ============================================================================
# 常量
# ============================================================================

STARTING_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"
_ROTATED_IDX = rotate_action(jnp.arange(ACTION_SPACE_SIZE))
MCTS_QTRANSFORM = partial(
    mctx.qtransform_completed_by_mix_value,
    value_scale=0.1,
)
DEFAULT_NUM_SIMULATIONS = 128
DEFAULT_TOP_K = 16

# 日志输出到 stderr，避免污染 stdout（UCI 协议要求）
logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
    stream=sys.stderr,
)
logger = logging.getLogger("uci_engine")


# ============================================================================
# 模型加载与推理
# ============================================================================

class Engine:
    def __init__(self, ckpt_dir: str, step: int = 0):
        self.ckpt_dir = os.path.abspath(ckpt_dir)
        self.step = step
        self.params = None
        self.net = None
        self.env = XiangqiEnv()
        self._recurrent_fn = None

    def _infer_arch(self, params) -> tuple:
        """从参数推断 channels 和 num_blocks（与 gui/api ModelManager 一致）"""
        channels, blocks = 128, 8
        try:
            p = params.get("params", params) if isinstance(params, dict) else params
            if isinstance(p, dict):
                if "input_proj" in p:
                    channels = int(p["input_proj"]["kernel"].shape[-1])
                blocks = len([k for k in p.keys() if str(k).startswith("GraphBlock_")])
                if blocks <= 0:
                    blocks = 8
            return channels, blocks
        except Exception:
            return 128, 8

    def load(self) -> bool:
        """加载 checkpoint"""
        try:
            ckpt_manager = ocp.CheckpointManager(self.ckpt_dir)
            step = self.step or ckpt_manager.latest_step()
            if step is None:
                logger.error("未找到 checkpoint")
                return False

            try:
                restored = ckpt_manager.restore(step)
            except Exception:
                ckpt_path = os.path.join(self.ckpt_dir, str(step))
                restored = ocp.StandardCheckpointer().restore(ckpt_path)
            params = None
            if isinstance(restored, dict):
                params = restored.get("params") or (restored.get("default") or {}).get("params")
            if params is None:
                logger.error("Checkpoint 中无 params")
                return False

            channels, num_blocks = self._infer_arch(params)
            self.net = AlphaZeroNetwork(
                action_space_size=ACTION_SPACE_SIZE,
                channels=channels,
                num_blocks=num_blocks,
            )
            # 验证前向
            dummy = jnp.zeros((1, NUM_OBSERVATION_CHANNELS, 10, 9), dtype=jnp.float32)
            _ = self.net.apply({"params": params}, dummy, train=False)

            self.params = params
            self.step = step
            self._recurrent_fn = None
            return True
        except Exception as e:
            logger.error("加载失败: %s", e)
            return False

    def _recurrent_fn_impl(self, params, rng_key, action, state):
        prev_p = state.current_player
        ns = jax.vmap(self.env.step)(state, action)
        obs = jax.vmap(self.env.observe)(ns)
        l, v, _ = self.net.apply({"params": params}, obs, train=False)
        l = jnp.where(ns.current_player[:, None] == 0, l, l[:, _ROTATED_IDX])
        l = l - jnp.max(l, axis=-1, keepdims=True)
        l = jnp.where(ns.legal_action_mask, l, jnp.finfo(l.dtype).min)
        return mctx.RecurrentFnOutput(
            reward=ns.rewards[jnp.arange(ns.rewards.shape[0]), prev_p],
            discount=jnp.where(ns.terminated, 0.0, -1.0),
            prior_logits=l,
            value=v,
        ), ns

    def get_best_move(
        self,
        state: XiangqiState,
        num_simulations: int = DEFAULT_NUM_SIMULATIONS,
        top_k: int = DEFAULT_TOP_K,
    ) -> tuple[str | None, float]:
        """返回 (uci_move, value)，无合法着法时返回 (None, 0)"""
        if self.params is None or self.net is None:
            return None, 0.0

        legal_mask = np.array(state.legal_action_mask)
        legal_actions = np.where(legal_mask)[0]
        if len(legal_actions) == 0:
            return None, 0.0

        if len(legal_actions) == 1:
            action = int(legal_actions[0])
            from_sq, to_sq = action_to_move(jnp.int32(action))
            obs = self.env.observe(state)[None, ...]
            _, value, _ = self.net.apply({"params": self.params}, obs, train=False)
            val = float(value[0])
            if state.current_player == 1:
                val = -val
            return move_to_uci(int(from_sq), int(to_sq)), val

        if self._recurrent_fn is None:
            self._recurrent_fn = self._recurrent_fn_impl

        obs = self.env.observe(state)[None, ...]
        logits, value, _ = self.net.apply({"params": self.params}, obs, train=False)
        if state.current_player == 1:
            logits = logits[:, _ROTATED_IDX]
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)

        rng_key, sk = jax.random.split(jax.random.PRNGKey(0))
        root = mctx.RootFnOutput(
            prior_logits=logits,
            value=value,
            embedding=jax.tree.map(lambda x: jnp.expand_dims(x, 0), state),
        )
        policy_output = mctx.gumbel_muzero_policy(
            params=self.params,
            rng_key=sk,
            root=root,
            recurrent_fn=self._recurrent_fn,
            num_simulations=num_simulations,
            max_num_considered_actions=top_k,
            invalid_actions=~state.legal_action_mask[None, ...],
            qtransform=MCTS_QTRANSFORM,
            gumbel_scale=0.0,
        )
        improved = np.array(policy_output.action_weights[0])
        action = int(np.argmax(improved))
        from_sq, to_sq = action_to_move(action)
        uci_move = move_to_uci(int(from_sq), int(to_sq))
        search_val = float(policy_output.search_tree.node_values[0, 0])
        if state.current_player == 1:
            search_val = -search_val
        return uci_move, search_val


# ============================================================================
# 局面解析
# ============================================================================

def _board_from_fen(fen: str) -> tuple[np.ndarray, int]:
    """解析 FEN，返回 (board, current_player)"""
    board, player = parse_fen(fen)
    return board, player


def _build_state_from_fen_and_moves(env: XiangqiEnv, fen: str, moves: list[str]) -> XiangqiState | None:
    """从 FEN + moves 构建 XiangqiState"""
    try:
        board, player = _board_from_fen(fen)
        board = jnp.array(board, dtype=jnp.int8)
        state = env.init_from_board(board, jnp.int32(player))

        for uci_move in moves:
            uci_move = uci_move.strip()
            if len(uci_move) < 4:
                continue
            if len(uci_move) == 4:
                from_sq, to_sq = uci_to_move(uci_move)
            else:
                # 可能带 promotion 等后缀，忽略
                from_sq, to_sq = uci_to_move(uci_move[:4])
            action = move_to_action(jnp.int32(from_sq), jnp.int32(to_sq))
            if action < 0:
                logger.error("非法着法: %s", uci_move)
                return None
            state = env.step(state, action)
            if state.terminated:
                break

        return state
    except Exception as e:
        logger.error("解析局面失败: %s", e)
        return None


# ============================================================================
# UCI 主循环
# ============================================================================

def run_uci_loop(engine: Engine, num_simulations: int, top_k: int):
    """UCI 协议主循环"""
    state = None
    sent_id = False

    def send(msg: str):
        print(msg, flush=True)

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        cmd = parts[0].lower()

        if cmd == "uci":
            send("id name ZeroForge")
            send("id author ZeroForge")
            send("option name CheckpointDir type string default checkpoints")
            send("option name NumSimulations type spin default 128 min 1 max 512")
            send("option name TopK type spin default 32 min 1 max 64")
            send("uciok")
            sent_id = True

        elif cmd == "isready":
            if not engine.load():
                send("info string 模型加载失败")
            send("readyok")

        elif cmd == "setoption":
            # setoption name CheckpointDir value checkpoints
            if len(parts) >= 5 and parts[1].lower() == "name":
                if parts[2].lower() == "checkpointdir":
                    idx = parts.index("value")
                    if idx + 1 < len(parts):
                        engine.ckpt_dir = os.path.abspath(parts[idx + 1])
                elif parts[2].lower() == "numsimulations":
                    idx = parts.index("value")
                    if idx + 1 < len(parts):
                        num_simulations = max(1, min(512, int(parts[idx + 1])))
                elif parts[2].lower() == "topk":
                    idx = parts.index("value")
                    if idx + 1 < len(parts):
                        top_k = max(1, min(64, int(parts[idx + 1])))

        elif cmd == "position":
            # position [fen <fen> | startpos] [moves <m1> <m2> ...]
            fen = STARTING_FEN
            moves = []
            i = 1
            if i < len(parts) and parts[i].lower() == "fen":
                i += 1
                fen_parts = []
                while i < len(parts) and parts[i].lower() not in ("moves",):
                    fen_parts.append(parts[i])
                    i += 1
                fen = " ".join(fen_parts)
            elif i < len(parts) and parts[i].lower() == "startpos":
                i += 1
            if i < len(parts) and parts[i].lower() == "moves":
                moves = parts[i + 1:]
            state = _build_state_from_fen_and_moves(engine.env, fen, moves)
            if state is None:
                send("info string 局面解析失败")

        elif cmd == "go":
            if state is None:
                state = _build_state_from_fen_and_moves(engine.env, STARTING_FEN, [])
            if state is None:
                send("bestmove 0000")
                continue
            if state.terminated:
                send("bestmove 0000")
                continue
            # 若未加载模型则惰性加载（部分 GUI 不发送 isready）
            if engine.params is None and not engine.load():
                send("info string 模型加载失败，请先 isready 或检查 --ckpt")
                send("bestmove 0000")
                continue

            move, value = engine.get_best_move(state, num_simulations, top_k)
            if move:
                send(f"info score cp {int(value * 500):d}")
                send(f"bestmove {move}")
            else:
                send("bestmove 0000")

        elif cmd == "quit":
            break

        elif cmd == "stop":
            # 当前实现无中断，忽略
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZeroForge UCI 引擎")
    parser.add_argument("--ckpt", default="checkpoints", help="Checkpoint 目录")
    parser.add_argument("--step", type=int, default=0, help="Checkpoint 步数，0=最新")
    parser.add_argument("--cpu", action="store_true", help="强制 CPU")
    parser.add_argument("--simulations", type=int, default=DEFAULT_NUM_SIMULATIONS)
    parser.add_argument("--topk", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()

    engine = Engine(args.ckpt, args.step)
    run_uci_loop(engine, args.simulations, args.topk)

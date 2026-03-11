#!/usr/bin/env python3
"""
ZeroForge UCI 引擎（高性能 + JAX 编译缓存）
"""

import os
import sys
import argparse
import logging
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

# ==========================================================
# 常量
# ==========================================================

STARTING_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"

_ROTATED_IDX = rotate_action(jnp.arange(ACTION_SPACE_SIZE))

MCTS_QTRANSFORM = partial(
    mctx.qtransform_completed_by_mix_value,
    value_scale=0.1,
)

DEFAULT_NUM_SIMULATIONS = 64
DEFAULT_TOP_K = 16

logging.basicConfig(
    level=logging.WARNING,
    format="%(message)s",
    stream=sys.stderr,
)

logger = logging.getLogger("zeroforge_uci")

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

    def load(self):

        try:

            ckpt_manager = ocp.CheckpointManager(self.ckpt_dir)

            step = self.step or ckpt_manager.latest_step()

            if step is None:
                logger.error("未找到 checkpoint")
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
                logger.error("checkpoint 无 params")
                return False

            channels, blocks = self._infer_arch(params)

            self.net = AlphaZeroNetwork(
                action_space_size=ACTION_SPACE_SIZE,
                channels=channels,
                num_blocks=blocks,
            )

            self.params = params

            # --------------------------------------------------
            # JIT inference
            # --------------------------------------------------

            self._infer = jax.jit(
                lambda p, x: self.net.apply({"params": p}, x, train=False)
            )

            # --------------------------------------------------
            # recurrent_fn
            # --------------------------------------------------

            def recurrent_fn(params, rng_key, action, state):

                prev_player = state.current_player

                next_state = self.env.step(state, action)

                obs = self.env.observe(next_state)[None]

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

            # --------------------------------------------------
            # warmup（触发编译）
            # --------------------------------------------------

            dummy = jnp.zeros((1, NUM_OBSERVATION_CHANNELS, 10, 9))
            self._infer(self.params, dummy)

            logger.warning("模型加载完成 step=%s", step)

            return True

        except Exception as e:

            logger.error("加载失败 %s", e)

            return False

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

    board, player = parse_fen(fen)

    state = env.init_from_board(
        jnp.array(board, dtype=jnp.int8),
        jnp.int32(player),
    )

    for m in moves:

        f, t = uci_to_move(m[:4])

        action = move_to_action(jnp.int32(f), jnp.int32(t))

        state = env.step(state, action)

        if state.terminated:
            break

    return state


# ==========================================================
# UCI LOOP
# ==========================================================


def run_uci(engine, simulations, top_k):

    state = None

    def send(x):
        print(x, flush=True)

    for line in sys.stdin:

        parts = line.strip().split()

        if not parts:
            continue

        cmd = parts[0]

        if cmd == "uci":

            send("id name ZeroForge")
            send("id author ZeroForge")
            send("uciok")

        elif cmd == "isready":

            if engine.params is None:
                engine.load()

            send("readyok")

        elif cmd == "position":

            fen = STARTING_FEN
            moves = []

            if parts[1] == "startpos":

                if "moves" in parts:
                    i = parts.index("moves")
                    moves = parts[i + 1 :]

            elif parts[1] == "fen":

                i = parts.index("fen") + 1
                fen = " ".join(parts[i : i + 2])

                if "moves" in parts:
                    j = parts.index("moves")
                    moves = parts[j + 1 :]

            state = build_state(engine.env, fen, moves)

        elif cmd == "go":

            if engine.params is None:
                if not engine.load():
                    send("info string model load failed")
                    send("bestmove 0000")
                    continue

            if state is None:
                state = build_state(engine.env, STARTING_FEN, [])

            move, val = engine.get_best_move(state, simulations, top_k)

            if move:

                cp = int(val * 1000)

                send(f"info score cp {cp} pv {move}")
                send(f"bestmove {move}")

            else:

                send("bestmove 0000")

        elif cmd == "quit":

            break


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
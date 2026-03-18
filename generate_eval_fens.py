#!/usr/bin/env python3
"""
使用本地 Pikafish 自博弈生成固定评估 FEN 集。

思路：
- 从标准初始局面开始
- 前若干步在 MultiPV 候选中做受控随机，制造开局多样性
- 仅保留引擎评估接近平衡的局面，避免一边倒
"""

from __future__ import annotations

import os
import sys

if __package__ in (None, ""):
    _SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    _PROJECT_ROOT = os.path.dirname(_SCRIPT_DIR)
    if _SCRIPT_DIR in sys.path:
        sys.path.remove(_SCRIPT_DIR)
    sys.path.insert(0, _PROJECT_ROOT)

import argparse
import math
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from xiangqi.actions import move_to_action, uci_to_move
from xiangqi.env import XiangqiEnv


PIKAFISH_PATH = Path(__file__).resolve().parent.parent / "pikafish"

PIECE_TO_FEN = {
    1: "K",
    2: "A",
    3: "B",
    4: "N",
    5: "R",
    6: "C",
    7: "P",
    -1: "k",
    -2: "a",
    -3: "b",
    -4: "n",
    -5: "r",
    -6: "c",
    -7: "p",
}


def board_to_fen(board: np.ndarray, player: int) -> str:
    rows: list[str] = []
    for r in range(9, -1, -1):
        empty = 0
        row_str = ""
        for c in range(9):
            piece = int(board[r, c])
            if piece == 0:
                empty += 1
                continue
            if empty > 0:
                row_str += str(empty)
                empty = 0
            row_str += PIECE_TO_FEN[piece]
        if empty > 0:
            row_str += str(empty)
        rows.append(row_str)
    return "/".join(rows) + (" w" if player == 0 else " b")


@dataclass
class Candidate:
    move: str
    score: int
    multipv: int


class PikafishUCI:
    def __init__(self, path: Path, threads: int = 1, hash_mb: int = 32):
        self.path = path
        self.threads = threads
        self.hash_mb = hash_mb
        self.process: subprocess.Popen[str] | None = None

    def __enter__(self) -> "PikafishUCI":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def start(self) -> None:
        self.process = subprocess.Popen(
            [str(self.path)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self.send("uci")
        self._read_until("uciok")
        self.send(f"setoption name Threads value {self.threads}")
        self.send(f"setoption name Hash value {self.hash_mb}")
        self.send("setoption name MultiPV value 1")
        self.send("isready")
        self._read_until("readyok")

    def close(self) -> None:
        if self.process is None:
            return
        try:
            self.send("quit")
        except OSError:
            pass
        try:
            self.process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.wait(timeout=2)
        self.process = None

    def send(self, command: str) -> None:
        if self.process is None or self.process.stdin is None:
            raise RuntimeError("Pikafish 未启动")
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def _read_until(self, token: str) -> list[str]:
        if self.process is None or self.process.stdout is None:
            raise RuntimeError("Pikafish 未启动")
        lines: list[str] = []
        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError(f"Pikafish 意外退出，等待 {token!r} 失败")
            line = line.strip()
            lines.append(line)
            if line == token or line.startswith(token):
                return lines

    def analyze(self, fen: str, depth: int, multipv: int) -> tuple[list[Candidate], str]:
        self.send(f"setoption name MultiPV value {multipv}")
        self.send("isready")
        self._read_until("readyok")
        self.send(f"position fen {fen}")
        self.send(f"go depth {depth}")

        bestmove = None
        candidates: dict[int, Candidate] = {}
        if self.process is None or self.process.stdout is None:
            raise RuntimeError("Pikafish 未启动")

        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("Pikafish 分析过程中意外退出")
            line = line.strip()
            if line.startswith("info "):
                parsed = self._parse_info(line)
                if parsed is not None:
                    candidates[parsed.multipv] = parsed
            elif line.startswith("bestmove "):
                parts = line.split()
                bestmove = parts[1]
                break

        ordered = [candidates[idx] for idx in sorted(candidates)]
        if not ordered and bestmove:
            ordered = [Candidate(move=bestmove, score=0, multipv=1)]
        if not ordered or bestmove is None:
            raise RuntimeError(f"未能从 Pikafish 解析候选: {fen}")
        return ordered, bestmove

    @staticmethod
    def _parse_info(line: str) -> Candidate | None:
        parts = line.split()
        if "multipv" not in parts or "pv" not in parts or "score" not in parts:
            return None
        try:
            multipv = int(parts[parts.index("multipv") + 1])
            score_kind = parts[parts.index("score") + 1]
            score_raw = int(parts[parts.index("score") + 2])
            pv_move = parts[parts.index("pv") + 1]
        except (ValueError, IndexError):
            return None
        if score_kind == "cp":
            score = score_raw
        elif score_kind == "mate":
            score = 30000 - abs(score_raw) * 100 if score_raw > 0 else -30000 + abs(score_raw) * 100
        else:
            return None
        return Candidate(move=pv_move, score=score, multipv=multipv)


def choose_move(
    candidates: list[Candidate],
    ply: int,
    diversify_plies: int,
    cp_window: int,
    rng: random.Random,
) -> Candidate:
    best = candidates[0]
    if ply >= diversify_plies or len(candidates) == 1:
        return best

    eligible = [cand for cand in candidates if cand.score >= best.score - cp_window]
    if len(eligible) == 1:
        return eligible[0]

    weights = [math.exp(-(cand.score - best.score) / max(cp_window, 1)) for cand in eligible]
    total = sum(weights)
    pick = rng.random() * total
    acc = 0.0
    for cand, weight in zip(eligible, weights):
        acc += weight
        if acc >= pick:
            return cand
    return eligible[-1]


def generate_fens(args) -> list[str]:
    env = XiangqiEnv(max_steps=args.max_steps)
    seed_key = jax.random.PRNGKey(args.seed)
    base_state = env.init(seed_key)
    start_fen = board_to_fen(np.array(base_state.board), int(base_state.current_player))
    rng = random.Random(args.seed)
    collected: list[str] = []
    seen: set[str] = set()

    with PikafishUCI(PIKAFISH_PATH, threads=args.threads, hash_mb=args.hash_mb) as engine:
        for game_idx in range(args.max_games):
            state = env.init(seed_key)
            last_recorded_ply = -args.min_gap
            for ply in range(args.max_ply + 1):
                if bool(np.array(state.terminated)):
                    break
                fen = board_to_fen(np.array(state.board), int(state.current_player))
                candidates, _bestmove = engine.analyze(fen if ply > 0 else start_fen, args.depth, args.multipv)
                score = candidates[0].score
                if (
                    args.min_ply <= ply <= args.max_ply
                    and abs(score) <= args.balance_cp
                    and ply - last_recorded_ply >= args.min_gap
                    and fen not in seen
                ):
                    seen.add(fen)
                    collected.append(fen)
                    last_recorded_ply = ply
                    print(f"[collect] game={game_idx + 1} ply={ply} score={score:+d} fen={fen}")
                    if len(collected) >= args.num_fens:
                        return collected

                if bool(np.array(state.terminated)):
                    break
                chosen = choose_move(
                    candidates,
                    ply=ply,
                    diversify_plies=args.diversify_plies,
                    cp_window=args.diversify_cp_window,
                    rng=rng,
                )
                from_sq, to_sq = uci_to_move(chosen.move[:4])
                action = int(np.array(move_to_action(jnp.int32(from_sq), jnp.int32(to_sq))))
                if action < 0:
                    raise RuntimeError(f"非法 UCI 走法无法映射为动作: {chosen.move}")
                state = env.step(state, jnp.int32(action))

    return collected


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="用 Pikafish 生成固定评估 FEN 集")
    parser.add_argument("--output", type=Path, default=Path("training/eval_fens.txt"))
    parser.add_argument("--num-fens", type=int, default=24)
    parser.add_argument("--max-games", type=int, default=40)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--multipv", type=int, default=4)
    parser.add_argument("--min-ply", type=int, default=10)
    parser.add_argument("--max-ply", type=int, default=30)
    parser.add_argument("--min-gap", type=int, default=4)
    parser.add_argument("--balance-cp", type=int, default=80)
    parser.add_argument("--diversify-plies", type=int, default=12)
    parser.add_argument("--diversify-cp-window", type=int, default=50)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--hash-mb", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if not PIKAFISH_PATH.exists():
        raise FileNotFoundError(f"未找到 Pikafish: {PIKAFISH_PATH}")
    fens = generate_fens(args)
    if not fens:
        raise RuntimeError("未生成任何 FEN，请放宽 balance-cp 或增大 max-games")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        f.write("# 由 training/generate_eval_fens.py 自动生成\n")
        for fen in fens:
            f.write(fen + "\n")
    print(f"[done] wrote {len(fens)} FENs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
ZeroForge 评估脚本 —— 模型 vs Pikafish (UCI引擎) 对弈
用法示例:
    python eval_vs_pikafish.py --ckpt_dir checkpoints --depth 8 --games 50
    python eval_vs_pikafish.py --ckpt_dir checkpoints --step 200 --depth 10 --games 100 --simulations 128
"""

import os
import sys
import time
import queue
import argparse
import threading
import subprocess
from functools import partial
from pathlib import Path
from typing import Optional, Tuple, List

# 显存分配策略（需在 import jax 前设置）
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import orbax.checkpoint as ocp
import mctx

from xiangqi.env import XiangqiEnv, XiangqiState, NUM_OBSERVATION_CHANNELS
from xiangqi.rules import get_legal_moves_mask, BOARD_WIDTH, BOARD_HEIGHT
from xiangqi.actions import (
    move_to_action, action_to_move, move_to_uci, uci_to_move,
    ACTION_SPACE_SIZE, rotate_action,
)
from networks.alphazero import AlphaZeroNetwork

# ============================================================================
# 常量
# ============================================================================

STARTING_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"
_ROTATED_IDX = rotate_action(jnp.arange(ACTION_SPACE_SIZE))
DEFAULT_PIKAFISH_PATH = Path(__file__).parent / "pikafish"

# ============================================================================
# UCI 引擎封装
# ============================================================================

class UCIEngine:
    """UCI 引擎封装（Pikafish 等）"""

    def __init__(self, path: str):
        self.path = path
        self.process = None
        self.output_queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self.lock = threading.Lock()
        self.ready = False

    def start(self) -> bool:
        """启动引擎，等待 uciok"""
        if self.process and self.process.poll() is None:
            return True
        try:
            self.process = subprocess.Popen(
                [self.path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, bufsize=1,
            )
            self._stop_event.clear()
            threading.Thread(target=self._read_stdout, daemon=True).start()
            self.send("uci")
            start = time.time()
            while time.time() - start < 10:
                try:
                    line = self.output_queue.get(timeout=0.1)
                    if "uciok" in line:
                        self.ready = True
                        print(f"[UCI] 引擎启动成功: {self.path}")
                        return True
                except queue.Empty:
                    continue
            print("[UCI] 引擎启动超时")
            return False
        except Exception as e:
            print(f"[UCI] 启动失败: {e}")
            return False

    def _read_stdout(self):
        while not self._stop_event.is_set() and self.process and self.process.poll() is None:
            line = self.process.stdout.readline()
            if line:
                self.output_queue.put(line.strip())

    def send(self, cmd: str):
        if self.process and self.process.stdin:
            self.process.stdin.write(f"{cmd}\n")
            self.process.stdin.flush()

    def new_game(self):
        """新开局，清空 hash 表"""
        with self.lock:
            self.send("ucinewgame")
            self.send("isready")
            start = time.time()
            while time.time() - start < 5:
                try:
                    line = self.output_queue.get(timeout=0.1)
                    if "readyok" in line:
                        return True
                except queue.Empty:
                    continue
            return False

    def get_best_move(self, fen: str, depth: int = 10) -> Tuple[Optional[str], Optional[int]]:
        """获取最佳着法和评估分数"""
        with self.lock:
            # 清空队列中的旧输出
            while not self.output_queue.empty():
                self.output_queue.get()

            self.send(f"position fen {fen}")
            self.send(f"go depth {depth}")

            start_time = time.time()
            wait_seconds = max(10.0, depth * 2.0)  # 评估模式给更多时间
            last_score = None

            while time.time() - start_time < wait_seconds:
                try:
                    line = self.output_queue.get(timeout=0.1)
                    if "score cp" in line:
                        try:
                            parts = line.split("score cp")
                            if len(parts) > 1:
                                score_part = parts[1].strip().split()[0]
                                last_score = int(score_part)
                        except (ValueError, IndexError):
                            pass
                    elif "score mate" in line:
                        try:
                            parts = line.split("score mate")
                            if len(parts) > 1:
                                mate_in = int(parts[1].strip().split()[0])
                                last_score = (30000 - abs(mate_in) * 100) if mate_in > 0 else (-30000 + abs(mate_in) * 100)
                        except (ValueError, IndexError):
                            pass
                    if line.startswith("bestmove"):
                        return line.split()[1], last_score
                except queue.Empty:
                    continue
            print(f"[UCI] 引擎思考超时 (depth={depth}, waited={wait_seconds:.1f}s)")
            return None, None

    def stop(self):
        self._stop_event.set()
        if self.process:
            try:
                self.send("quit")
                self.process.wait(timeout=3)
            except Exception:
                self.process.terminate()


# ============================================================================
# 模型加载
# ============================================================================

def _infer_channels(params) -> Optional[int]:
    """从参数推断模型通道数"""
    try:
        if hasattr(params, "get") and params.get("Dense_0") is not None:
            return int(params["Dense_0"]["kernel"].shape[-1])
        if "Dense_0" in params:
            return int(params["Dense_0"]["kernel"].shape[-1])
    except Exception:
        pass
    try:
        for k in params.keys():
            if str(k).startswith("Dense_"):
                kernel = params[k]["kernel"]
                if kernel.ndim == 2:
                    return int(kernel.shape[-1])
    except Exception:
        return None
    return None


def _infer_num_blocks(params) -> int:
    """从参数推断 GraphBlock 数量"""
    try:
        return len([k for k in params.keys() if str(k).startswith("GraphBlock_")])
    except Exception:
        return 0


def load_model(ckpt_dir: str, step: int = 0):
    """
    加载检查点，返回 (params, net, step, channels, num_blocks)。
    失败则抛出异常。
    """
    ckpt_dir = os.path.abspath(ckpt_dir)
    if not os.path.exists(ckpt_dir):
        raise FileNotFoundError(f"检查点目录不存在: {ckpt_dir}")

    ckpt_manager = ocp.CheckpointManager(ckpt_dir)
    if step == 0:
        step = ckpt_manager.latest_step()
    if step is None:
        raise FileNotFoundError(f"在 {ckpt_dir} 中未找到任何检查点")

    # 恢复检查点
    restored = None
    try:
        restored = ckpt_manager.restore(step)
    except Exception:
        try:
            ckpt_path = os.path.join(ckpt_dir, str(step))
            restored = ocp.StandardCheckpointer().restore(ckpt_path)
        except Exception as e:
            raise RuntimeError(f"检查点恢复失败 (step={step}): {e}")

    # 提取 params
    params = None
    if isinstance(restored, dict) or hasattr(restored, "keys"):
        if "params" in restored:
            params = restored["params"]
        elif "default" in restored and isinstance(restored["default"], dict):
            params = restored["default"].get("params")
    if params is None:
        raise RuntimeError("检查点中未找到 params 字段")

    channels = _infer_channels(params)
    num_blocks = _infer_num_blocks(params)
    if not channels or num_blocks <= 0:
        raise RuntimeError(
            f"模型结构推断失败: channels={channels}, num_blocks={num_blocks}"
        )

    # 创建网络并验证
    net = AlphaZeroNetwork(
        action_space_size=ACTION_SPACE_SIZE,
        channels=channels,
        num_blocks=num_blocks,
    )
    dummy_obs = jnp.zeros(
        (1, NUM_OBSERVATION_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH), dtype=jnp.float32
    )
    try:
        _ = net.apply({"params": params}, dummy_obs, train=False)
    except Exception as e:
        raise RuntimeError(f"模型结构不兼容: {e}")

    print(f"[Model] 加载完成: step={step}, channels={channels}, blocks={num_blocks}")
    return params, net, step, channels, num_blocks


# ============================================================================
# FEN 转换
# ============================================================================

_FEN_TO_PIECE = {
    "K": 1, "A": 2, "B": 3, "N": 4, "R": 5, "C": 6, "P": 7,
    "k": -1, "a": -2, "b": -3, "n": -4, "r": -5, "c": -6, "p": -7,
}
_PIECE_TO_FEN = {v: k for k, v in _FEN_TO_PIECE.items()}


def parse_fen(fen: str) -> Tuple[np.ndarray, int]:
    """FEN -> (board, player)"""
    parts = fen.strip().split()
    board_str = parts[0]
    player = 0 if len(parts) < 2 or parts[1].lower() in ("w", "r") else 1
    board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=np.int8)
    rows = board_str.split("/")
    for r_idx, r_str in enumerate(rows):
        row = 9 - r_idx
        col = 0
        for char in r_str:
            if char.isdigit():
                col += int(char)
            elif char in _FEN_TO_PIECE:
                board[row, col] = _FEN_TO_PIECE[char]
                col += 1
    return board, player


def board_to_fen(board: np.ndarray, player: int) -> str:
    """(board, player) -> FEN"""
    rows = []
    for r in range(9, -1, -1):
        r_str, empty = "", 0
        for c in range(9):
            p = board[r, c]
            if p == 0:
                empty += 1
            else:
                if empty > 0:
                    r_str += str(empty)
                    empty = 0
                r_str += _PIECE_TO_FEN[int(p)]
        if empty > 0:
            r_str += str(empty)
        rows.append(r_str)
    return "/".join(rows) + (" w" if player == 0 else " b")


# ============================================================================
# 列出检查点
# ============================================================================

def list_checkpoints(ckpt_dir: str) -> List[int]:
    """列出目录下所有可用的检查点 step"""
    if not os.path.exists(ckpt_dir):
        return []
    steps = []
    for d in os.listdir(ckpt_dir):
        full = os.path.join(ckpt_dir, d)
        if os.path.isdir(full) and d.isdigit():
            steps.append(int(d))
    return sorted(steps)


# ============================================================================
# 对弈逻辑
# ============================================================================

class Evaluator:
    """模型 vs Pikafish 评估器"""

    def __init__(
        self,
        params,
        net: AlphaZeroNetwork,
        engine: UCIEngine,
        env: XiangqiEnv,
        num_simulations: int = 64,
        top_k: int = 16,
        pikafish_depth: int = 10,
        max_moves: int = 400,
    ):
        self.params = params
        self.net = net
        self.engine = engine
        self.env = env
        self.num_simulations = num_simulations
        self.top_k = top_k
        self.pikafish_depth = pikafish_depth
        self.max_moves = max_moves
        self.rng_key = jax.random.PRNGKey(int(time.time()))

        # MCTS 配置
        self._qtransform = partial(
            mctx.qtransform_completed_by_mix_value,
            value_scale=0.25,
        )
        # JIT 编译 MCTS 递归函数
        self._recurrent_fn = self._create_recurrent_fn()

    def _create_recurrent_fn(self):
        """创建 MCTS 递归函数"""
        net = self.net
        env = self.env
        rotated_idx = _ROTATED_IDX

        def recurrent_fn(params, rng_key, action, state):
            prev_p = state.current_player
            ns = jax.vmap(env.step)(state, action)
            obs = jax.vmap(env.observe)(ns)
            logits, value = net.apply({"params": params}, obs, train=False)
            logits = jnp.where(
                ns.current_player[:, None] == 0,
                logits,
                logits[:, rotated_idx],
            )
            logits = logits - jnp.max(logits, axis=-1, keepdims=True)
            logits = jnp.where(ns.legal_action_mask, logits, jnp.finfo(logits.dtype).min)
            reward = ns.rewards[jnp.arange(ns.rewards.shape[0]), prev_p]
            discount = jnp.where(ns.terminated, 0.0, -1.0)
            return mctx.RecurrentFnOutput(
                reward=reward, discount=discount,
                prior_logits=logits, value=value,
            ), ns

        return recurrent_fn

    def _get_model_action(self, jax_state: XiangqiState) -> Tuple[int, float]:
        """
        使用 MCTS 获取模型最佳着法。
        返回 (action_id, search_value)。
        """
        self.rng_key, sk = jax.random.split(self.rng_key)

        obs = self.env.observe(jax_state)[None, ...]
        logits, value = self.net.apply({"params": self.params}, obs, train=False)

        current_player = int(jax_state.current_player)
        if current_player == 1:
            logits = logits[:, _ROTATED_IDX]
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        logits = jnp.where(jax_state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)

        root = mctx.RootFnOutput(
            prior_logits=logits,
            value=value,
            embedding=jax.tree.map(lambda x: jnp.expand_dims(x, 0), jax_state),
        )

        policy_output = mctx.gumbel_muzero_policy(
            params=self.params,
            rng_key=sk,
            root=root,
            recurrent_fn=self._recurrent_fn,
            num_simulations=self.num_simulations,
            max_num_considered_actions=self.top_k,
            invalid_actions=~jax_state.legal_action_mask[None, ...],
            qtransform=self._qtransform,
            gumbel_scale=0.0,  # 评估关闭探索噪声
        )

        search_value = float(policy_output.search_tree.node_values[0, 0])
        if current_player == 1:
            search_value = -search_value  # 统一为红方视角

        weights = np.array(policy_output.action_weights[0])
        action = int(np.argmax(weights))
        return action, search_value

    def _get_pikafish_action(self, board: np.ndarray, player: int) -> Tuple[int, Optional[int]]:
        """
        获取 Pikafish 着法。
        返回 (action_id, score)。失败抛出异常。
        """
        fen = board_to_fen(board, player)
        bestmove, score = self.engine.get_best_move(fen, self.pikafish_depth)
        if not bestmove or bestmove in ("(none)", "0000"):
            raise RuntimeError(f"Pikafish 未能返回有效着法 (fen={fen})")

        from_sq, to_sq = uci_to_move(bestmove)
        action = int(move_to_action(from_sq, to_sq))
        if action < 0:
            raise RuntimeError(f"Pikafish 返回了非法着法: {bestmove} (fen={fen})")

        # 统一 score 为红方视角
        if player == 1 and score is not None:
            score = -score
        return action, score

    def play_one_game(self, model_is_red: bool, verbose: bool = False) -> dict:
        """
        下一局，返回结果字典。
        model_is_red: True 表示模型执红先手。
        """
        self.rng_key, sk = jax.random.split(self.rng_key)
        jax_state = self.env.init(sk)

        # Pikafish 新开局
        self.engine.new_game()

        move_count = 0
        move_log: List[str] = []
        game_start = time.time()

        while not bool(jax_state.terminated) and move_count < self.max_moves:
            current_player = int(jax_state.current_player)
            is_model_turn = (current_player == 0) == model_is_red

            board_np = np.array(jax_state.board)

            if is_model_turn:
                # 模型走棋
                action, search_value = self._get_model_action(jax_state)
                from_sq, to_sq = action_to_move(action)
                uci_str = move_to_uci(int(from_sq), int(to_sq))
                if verbose:
                    side = "红" if current_player == 0 else "黑"
                    print(f"  第{move_count+1}步 [{side}] Model: {uci_str} (V={search_value:+.3f})")
            else:
                # Pikafish 走棋
                try:
                    action, score = self._get_pikafish_action(board_np, current_player)
                except RuntimeError as e:
                    # Pikafish 返回错误 → 模型胜
                    print(f"  [WARNING] Pikafish 异常: {e}，判模型胜")
                    return {
                        "winner": "model",
                        "reason": f"pikafish_error: {e}",
                        "moves": move_count,
                        "model_color": "red" if model_is_red else "black",
                        "move_log": move_log,
                        "elapsed": time.time() - game_start,
                    }
                from_sq, to_sq = action_to_move(action)
                uci_str = move_to_uci(int(from_sq), int(to_sq))
                if verbose:
                    side = "红" if current_player == 0 else "黑"
                    score_str = f"(cp={score})" if score is not None else ""
                    print(f"  第{move_count+1}步 [{side}] Pikafish: {uci_str} {score_str}")

            # 检查着法合法性
            if not bool(jax_state.legal_action_mask[action]):
                who = "Model" if is_model_turn else "Pikafish"
                print(f"  [ERROR] {who} 返回了非法着法 action={action} uci={uci_str}，判对方胜")
                winner = "pikafish" if is_model_turn else "model"
                return {
                    "winner": winner,
                    "reason": f"illegal_move_by_{who.lower()}",
                    "moves": move_count,
                    "model_color": "red" if model_is_red else "black",
                    "move_log": move_log,
                    "elapsed": time.time() - game_start,
                }

            move_log.append(uci_str)
            jax_state = self.env.step(jax_state, action)
            move_count += 1

        # 判定结果
        elapsed = time.time() - game_start
        winner_code = int(jax_state.winner)  # -1=和, 0=红胜, 1=黑胜
        draw_reason = int(jax_state.draw_reason)

        if winner_code == -1:
            # 和棋
            result = "draw"
        elif (winner_code == 0 and model_is_red) or (winner_code == 1 and not model_is_red):
            result = "model"
        else:
            result = "pikafish"

        # 和棋/终局原因描述
        reason_map = {
            0: "未结束",
            1: "步数到限",
            2: "无吃子到限",
            3: "重复局面和棋",
            4: "长将判负",
            5: "无进攻子力",
            6: "长捉判负",
            7: "将捉交替判负",
            8: "将死/困毙",
        }
        reason = reason_map.get(draw_reason, f"unknown({draw_reason})")
        if move_count >= self.max_moves and not bool(jax_state.terminated):
            reason = "达到最大步数限制"
            result = "draw"

        return {
            "winner": result,
            "reason": reason,
            "moves": move_count,
            "model_color": "red" if model_is_red else "black",
            "move_log": move_log,
            "elapsed": elapsed,
        }

    def evaluate(self, num_games: int, verbose: bool = False) -> dict:
        """
        运行多局对弈评估。
        模型交替执红/黑，保证公平。
        """
        results = {"model_win": 0, "pikafish_win": 0, "draw": 0}
        # 按颜色细分
        red_results = {"model_win": 0, "pikafish_win": 0, "draw": 0}
        black_results = {"model_win": 0, "pikafish_win": 0, "draw": 0}

        total_moves = 0
        total_time = 0.0
        game_details: List[dict] = []

        print(f"\n{'='*60}")
        print(f"  模型 vs Pikafish 评估")
        print(f"  对局数: {num_games} (模型各执红、黑 {num_games//2} 局)")
        print(f"  MCTS 模拟次数: {self.num_simulations}")
        print(f"  Pikafish 深度: {self.pikafish_depth}")
        print(f"{'='*60}\n")

        for i in range(num_games):
            model_is_red = (i % 2 == 0)
            color_str = "红" if model_is_red else "黑"
            print(f"--- 第 {i+1}/{num_games} 局 (模型执{color_str}) ---")

            result = self.play_one_game(model_is_red=model_is_red, verbose=verbose)
            game_details.append(result)

            total_moves += result["moves"]
            total_time += result["elapsed"]

            # 统计
            if result["winner"] == "model":
                results["model_win"] += 1
                if model_is_red:
                    red_results["model_win"] += 1
                else:
                    black_results["model_win"] += 1
                emoji = "胜"
            elif result["winner"] == "pikafish":
                results["pikafish_win"] += 1
                if model_is_red:
                    red_results["pikafish_win"] += 1
                else:
                    black_results["pikafish_win"] += 1
                emoji = "负"
            else:
                results["draw"] += 1
                if model_is_red:
                    red_results["draw"] += 1
                else:
                    black_results["draw"] += 1
                emoji = "和"

            print(
                f"  结果: 模型{emoji} | 原因: {result['reason']} "
                f"| 步数: {result['moves']} | 耗时: {result['elapsed']:.1f}s"
            )

            # 打印中途汇总
            played = i + 1
            wr = results["model_win"] / played * 100
            print(
                f"  [进度] 胜{results['model_win']} "
                f"负{results['pikafish_win']} 和{results['draw']} "
                f"(胜率 {wr:.1f}%)\n"
            )

        # ===================== 最终汇总 =====================
        print(f"\n{'='*60}")
        print(f"  最终评估结果")
        print(f"{'='*60}")

        total = num_games
        mw, pw, dr = results["model_win"], results["pikafish_win"], results["draw"]
        # 胜率 = (胜 + 和*0.5) / 总局数 (ELO 计算用)
        win_rate = (mw + dr * 0.5) / total * 100 if total > 0 else 0
        pure_win_rate = mw / total * 100 if total > 0 else 0

        print(f"  总局数:       {total}")
        print(f"  模型胜:       {mw} ({mw/total*100:.1f}%)")
        print(f"  Pikafish胜:   {pw} ({pw/total*100:.1f}%)")
        print(f"  和棋:         {dr} ({dr/total*100:.1f}%)")
        print(f"  纯胜率:       {pure_win_rate:.1f}%")
        print(f"  得分率:       {win_rate:.1f}% (胜=1, 和=0.5)")

        # ELO 差估算 (基于 logistic 模型)
        if 0 < win_rate < 100:
            import math
            elo_diff = -400 * math.log10(100 / win_rate - 1)
            print(f"  预估 ELO 差:  {elo_diff:+.0f}")
        elif win_rate >= 100:
            print(f"  预估 ELO 差:  +inf (全胜)")
        else:
            print(f"  预估 ELO 差:  -inf (全负)")

        print(f"\n  --- 按颜色细分 ---")
        red_total = num_games // 2
        black_total = num_games - red_total
        if red_total > 0:
            print(
                f"  模型执红 ({red_total}局): "
                f"胜{red_results['model_win']} "
                f"负{red_results['pikafish_win']} "
                f"和{red_results['draw']}"
            )
        if black_total > 0:
            print(
                f"  模型执黑 ({black_total}局): "
                f"胜{black_results['model_win']} "
                f"负{black_results['pikafish_win']} "
                f"和{black_results['draw']}"
            )

        avg_moves = total_moves / total if total > 0 else 0
        print(f"\n  平均步数:     {avg_moves:.1f}")
        print(f"  总耗时:       {total_time:.1f}s")
        print(f"  平均每局:     {total_time/total:.1f}s" if total > 0 else "")
        print(f"{'='*60}\n")

        return {
            "results": results,
            "red_results": red_results,
            "black_results": black_results,
            "win_rate": win_rate,
            "game_details": game_details,
        }


# ============================================================================
# 批量评估多个检查点
# ============================================================================

def evaluate_checkpoints(args):
    """评估指定检查点目录下的一个或多个检查点"""

    # 初始化 Pikafish
    pikafish_path = args.pikafish or str(DEFAULT_PIKAFISH_PATH)
    if not os.path.exists(pikafish_path):
        print(f"[ERROR] 未找到 Pikafish: {pikafish_path}")
        print("  请通过 --pikafish 参数指定正确路径")
        sys.exit(1)

    engine = UCIEngine(pikafish_path)
    if not engine.start():
        print("[ERROR] Pikafish 引擎启动失败")
        sys.exit(1)

    # 初始化环境
    env = XiangqiEnv(
        max_steps=args.max_steps,
        max_no_capture_steps=120,
        repetition_threshold=5,
    )

    # 获取要评估的检查点列表
    if args.step > 0:
        steps_to_eval = [args.step]
    elif args.eval_all:
        steps_to_eval = list_checkpoints(args.ckpt_dir)
        if not steps_to_eval:
            print(f"[ERROR] {args.ckpt_dir} 中未找到任何检查点")
            sys.exit(1)
        print(f"[INFO] 找到 {len(steps_to_eval)} 个检查点: {steps_to_eval}")
    else:
        # 默认只评估最新
        steps_to_eval = [0]

    all_results = {}

    for step in steps_to_eval:
        step_label = f"step={step}" if step > 0 else "latest"
        print(f"\n{'#'*60}")
        print(f"  评估检查点: {step_label}")
        print(f"{'#'*60}")

        try:
            params, net, actual_step, channels, num_blocks = load_model(args.ckpt_dir, step)
        except Exception as e:
            print(f"[ERROR] 加载检查点失败 ({step_label}): {e}")
            continue

        evaluator = Evaluator(
            params=params,
            net=net,
            engine=engine,
            env=env,
            num_simulations=args.simulations,
            top_k=args.top_k,
            pikafish_depth=args.depth,
            max_moves=args.max_steps,
        )

        summary = evaluator.evaluate(num_games=args.games, verbose=args.verbose)
        all_results[actual_step] = summary

    # 多检查点汇总
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print(f"  多检查点汇总 (Pikafish depth={args.depth})")
        print(f"{'='*60}")
        print(f"  {'Step':>8}  {'胜':>4}  {'负':>4}  {'和':>4}  {'得分率':>8}")
        print(f"  {'-'*40}")
        for s in sorted(all_results.keys()):
            r = all_results[s]["results"]
            wr = all_results[s]["win_rate"]
            print(f"  {s:>8}  {r['model_win']:>4}  {r['pikafish_win']:>4}  {r['draw']:>4}  {wr:>7.1f}%")
        print(f"{'='*60}\n")

    # 清理
    engine.stop()
    return all_results


# ============================================================================
# 命令行入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ZeroForge 模型 vs Pikafish 对弈评估",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 评估最新检查点，Pikafish depth=8，下50局
  python eval_vs_pikafish.py --ckpt_dir checkpoints --depth 8 --games 50

  # 评估指定检查点
  python eval_vs_pikafish.py --ckpt_dir checkpoints --step 200 --depth 10 --games 100

  # 评估所有检查点
  python eval_vs_pikafish.py --ckpt_dir checkpoints --eval_all --depth 6 --games 20

  # 详细输出每步着法
  python eval_vs_pikafish.py --ckpt_dir checkpoints --depth 8 --games 10 --verbose
        """,
    )
    parser.add_argument("--ckpt_dir", type=str, default="checkpoints",
                        help="检查点目录 (默认: checkpoints)")
    parser.add_argument("--step", type=int, default=0,
                        help="指定检查点 step (默认: 0 = 最新)")
    parser.add_argument("--eval_all", action="store_true",
                        help="评估目录下所有检查点")
    parser.add_argument("--pikafish", type=str, default=None,
                        help="Pikafish 可执行文件路径 (默认: 项目目录下的 pikafish)")
    parser.add_argument("--depth", type=int, default=10,
                        help="Pikafish 搜索深度 (默认: 10)")
    parser.add_argument("--games", type=int, default=20,
                        help="对弈局数 (默认: 20，建议偶数以均衡红黑)")
    parser.add_argument("--simulations", type=int, default=64,
                        help="MCTS 模拟次数 (默认: 64)")
    parser.add_argument("--top_k", type=int, default=16,
                        help="MCTS 根节点候选数 (默认: 16)")
    parser.add_argument("--max_steps", type=int, default=400,
                        help="每局最大步数 (默认: 400)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="显示每步着法详情")

    args = parser.parse_args()

    # 确保局数为偶数（红黑均衡）
    if args.games % 2 != 0:
        args.games += 1
        print(f"[INFO] 对局数调整为偶数: {args.games}")

    evaluate_checkpoints(args)


if __name__ == "__main__":
    main()

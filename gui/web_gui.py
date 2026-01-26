"""
ZeroForge Web GUI - 现代化象棋对弈界面
支持人机、双 AI、UCI 引擎对弈，自适应移动端
"""

import os
import time
import subprocess
import threading
import queue
import gradio as gr
import numpy as np
from typing import Optional, Tuple, List
from dataclasses import dataclass
import traceback

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import mctx

from xiangqi.env import XiangqiEnv, XiangqiState
from xiangqi.rules import (
    get_legal_moves_mask, is_in_check, find_king,
    BOARD_WIDTH, BOARD_HEIGHT
)
from xiangqi.actions import (
    move_to_action, action_to_move, move_to_uci, uci_to_move,
    ACTION_SPACE_SIZE, rotate_action
)
from networks.alphazero import AlphaZeroNetwork

# ============================================================================
# 常量与配置
# ============================================================================

# 预计算旋转索引，避免每次推理重复计算 (JAX 性能优化)
_ROTATED_IDX = rotate_action(jnp.arange(ACTION_SPACE_SIZE))

CELL_SIZE = 60
BOARD_MARGIN = 40
PIECE_RADIUS = 26
SVG_WIDTH = BOARD_MARGIN * 2 + CELL_SIZE * (BOARD_WIDTH - 1)
SVG_HEIGHT = BOARD_MARGIN * 2 + CELL_SIZE * (BOARD_HEIGHT - 1)

COLOR_BG = "#F5DEB3"
COLOR_LINE = "#5D4037"
COLOR_RED = "#D32F2F"
COLOR_BLACK = "#212121"
COLOR_SELECTED = "#FFD600"
COLOR_LEGAL = "#4CAF50"
COLOR_LAST_MOVE = "#03A9F4"
COLOR_CHECK = "#F44336"

PIECE_NAMES = {
    1: ('帅', '将'), 2: ('仕', '士'), 3: ('相', '象'),
    4: ('马', '马'), 5: ('车', '车'), 6: ('炮', '炮'), 7: ('兵', '卒'),
}

STARTING_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"

# ============================================================================
# 工具函数
# ============================================================================

def list_checkpoints(ckpt_dir: str) -> List[int]:
    """列出目录下所有的 step 编号"""
    if not os.path.exists(ckpt_dir):
        return []
    steps = []
    for d in os.listdir(ckpt_dir):
        if os.path.isdir(os.path.join(ckpt_dir, d)) and d.isdigit():
            steps.append(int(d))
    return sorted(steps, reverse=True)

def parse_fen(fen: str) -> Tuple[np.ndarray, int]:
    parts = fen.strip().split()
    board_str = parts[0]
    player = 0 if len(parts) < 2 or parts[1].lower() in ['w', 'r'] else 1
    board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=np.int8)
    FEN_TO_PIECE = {'K':1,'A':2,'B':3,'N':4,'R':5,'C':6,'P':7,'k':-1,'a':-2,'b':-3,'n':-4,'r':-5,'c':-6,'p':-7}
    rows = board_str.split('/')
    for r_idx, r_str in enumerate(rows):
        row = 9 - r_idx
        col = 0
        for char in r_str:
            if char.isdigit(): col += int(char)
            elif char in FEN_TO_PIECE:
                board[row, col] = FEN_TO_PIECE[char]
                col += 1
    return board, player

def board_to_fen(board: np.ndarray, player: int) -> str:
    PIECE_TO_FEN = {1:'K',2:'A',3:'B',4:'N',5:'R',6:'C',7:'P',-1:'k',-2:'a',-3:'b',-4:'n',-5:'r',-6:'c',-7:'p'}
    rows = []
    for r in range(9, -1, -1):
        r_str, empty = "", 0
        for c in range(9):
            p = board[r, c]
            if p == 0: empty += 1
            else:
                if empty > 0: r_str += str(empty); empty = 0
                r_str += PIECE_TO_FEN[int(p)]
        if empty > 0: r_str += str(empty)
        rows.append(r_str)
    return "/".join(rows) + (" w" if player == 0 else " b")

# ============================================================================
# UCI 引擎支持
# ============================================================================

class UCIEngine:
    def __init__(self, path: str):
        self.path = path
        self.process = None
        self.output_queue = queue.Queue()
        self._stop_event = threading.Event()
        self.lock = threading.Lock()

    def start(self):
        try:
            self.process = subprocess.Popen(
                [self.path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, bufsize=1
            )
            self._stop_event.clear()
            threading.Thread(target=self._read_stdout, daemon=True).start()
            self.send("uci")
            return True
        except Exception as e:
            print(f"[UCI] 启动失败: {e}")
            return False

    def _read_stdout(self):
        while not self._stop_event.is_set() and self.process and self.process.poll() is None:
            line = self.process.stdout.readline()
            if line: self.output_queue.put(line.strip())

    def send(self, cmd: str):
        if self.process and self.process.stdin:
            self.process.stdin.write(f"{cmd}\n")
            self.process.stdin.flush()

    def get_best_move(self, fen: str, movetime: int = 1000, depth: Optional[int] = None) -> Tuple[Optional[str], Optional[int]]:
        """
        获取最佳着法和评估分数
        返回: (bestmove, score_cp) - score_cp 为厘兵分数，正值对当前走棋方有利
        """
        with self.lock:
            while not self.output_queue.empty(): self.output_queue.get()
            self.send(f"position fen {fen}")
            if depth is not None and depth > 0:
                self.send(f"go depth {depth}")
            else:
                self.send(f"go movetime {movetime}")
            start_time = time.time()
            wait_seconds = (movetime / 1000.0 + 2.0) if depth is None else max(2.0, depth * 0.5)
            last_score = None  # 记录最后一次的评估分数
            while time.time() - start_time < wait_seconds:
                try:
                    line = self.output_queue.get(timeout=0.1)
                    # 解析 info 行中的分数: "info depth X ... score cp YYY ..." 或 "score mate X"
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
                                # 将杀转换为大分数，正值表示己方能杀，负值表示被杀
                                last_score = 30000 - abs(mate_in) * 100 if mate_in > 0 else -30000 + abs(mate_in) * 100
                        except (ValueError, IndexError):
                            pass
                    if line.startswith("bestmove"):
                        return line.split()[1], last_score
                except queue.Empty: 
                    continue
        return None, None

    def stop(self):
        self._stop_event.set()
        if self.process: self.process.terminate()

# ============================================================================
# AI 模型管理
# ============================================================================

class ModelManager:
    def __init__(self):
        self.params = None
        self.net = None

    def _infer_channels(self, params) -> Optional[int]:
        try:
            conv0 = params.get("Conv_0") if hasattr(params, "get") else params["Conv_0"]
            kernel = conv0["kernel"]
            return int(kernel.shape[-1])
        except Exception:
            return None

    def _infer_num_blocks(self, params) -> int:
        try:
            keys = list(params.keys())
            return len([k for k in keys if str(k).startswith("ResBlock_")])
        except Exception:
            return 0

    def load(self, ckpt_dir: str, step: int):
        ckpt_dir = os.path.abspath(ckpt_dir)
        ckpt_manager = ocp.CheckpointManager(ckpt_dir)
        if step == 0:
            step = ckpt_manager.latest_step()
        if step is None:
            return False

        restored = None
        restore_err = None
        try:
            restored = ckpt_manager.restore(step)
        except Exception as e:
            restore_err = e

        if restored is None:
            try:
                ckpt_path = os.path.join(ckpt_dir, str(step))
                restored = ocp.StandardCheckpointer().restore(ckpt_path)
            except Exception as e:
                raise RuntimeError(f"Checkpoint 恢复失败: {restore_err or e}")

        params = None
        if isinstance(restored, dict) or hasattr(restored, "keys"):
            if "params" in restored:
                params = restored["params"]
            elif "default" in restored and isinstance(restored["default"], dict) and "params" in restored["default"]:
                params = restored["default"]["params"]

        if params is None:
            keys = list(restored.keys()) if hasattr(restored, "keys") else type(restored)
            raise RuntimeError(f"Checkpoint 不包含 params，keys={keys}")

        channels = self._infer_channels(params)
        num_blocks = self._infer_num_blocks(params)
        if not channels or num_blocks <= 0:
            keys = list(params.keys()) if hasattr(params, "keys") else type(params)
            raise RuntimeError(f"无法从参数推断网络结构，keys={keys}")

        self.net = AlphaZeroNetwork(
            action_space_size=ACTION_SPACE_SIZE,
            channels=channels,
            num_blocks=num_blocks,
        )
        self.params = params
        print(f"[AI] 模型加载完成: step={step}, channels={channels}, blocks={num_blocks}")
        return True

# ============================================================================
# 游戏状态
# ============================================================================

@dataclass
class GameState:
    board: np.ndarray
    current_player: int
    selected: Optional[Tuple[int, int]] = None
    legal_moves: List[Tuple[int, int]] = None
    last_move: Optional[Tuple[int, int, int, int]] = None
    is_check: bool = False
    king_pos: Optional[Tuple[int, int]] = None
    game_over: bool = False
    winner: int = -1
    step_count: int = 0
    history: List = None
    jax_state: Optional[XiangqiState] = None
    ai_value: float = 0.0           # ZeroForge AI 评估值 [-1, 1]，正值对红方有利
    uci_score: Optional[int] = None # UCI 引擎评估（厘兵），正值对当前走棋方有利
    last_move_player: int = 0       # 上一步是哪方走的（用于正确显示评估）
    last_move_uci: str = ""
    notice: str = ""
    replay_index: Optional[int] = None

    def __post_init__(self):
        self.legal_moves = self.legal_moves or []
        self.history = self.history or []

class ChessGame:
    def __init__(self):
        self.env = XiangqiEnv()
        self.state: Optional[GameState] = None
        self.model_mgr = ModelManager()
        self.uci_engine: Optional[UCIEngine] = None
        self._rng_key = jax.random.PRNGKey(int(time.time()))
        self.red_type = "Human"
        self.black_type = "ZeroForge AI"
        self.uci_movetime = 1000
        self.uci_depth = 3
        self.ai_delay = 1.0
        
        # 暂停状态
        self.paused = False
        
        # 缓存编译后的 MCTS recurrent_fn，避免每次推理重新编译
        self._mcts_recurrent_fn = self._create_mcts_recurrent_fn()
    
    def _create_mcts_recurrent_fn(self):
        """
        创建 MCTS 递归函数
        
        注意：此函数通过闭包引用 self.model_mgr，这样在加载新模型后
        会自动使用新的网络结构。JAX 会根据网络结构的变化决定是否重新编译。
        相比于每次推理都在函数内定义 recurrent_fn，这种方式可以：
        1. 同一网络结构下复用编译结果
        2. 网络结构变化时自动重新编译
        """
        env = self.env
        model_mgr = self.model_mgr  # 闭包引用，获取最新的 net
        
        def recurrent_fn(params, rng_key, action, state):
            prev_p = state.current_player
            ns = jax.vmap(env.step)(state, action)
            obs = jax.vmap(env.observe)(ns)
            # model_mgr.net 会在运行时获取，支持动态加载新模型
            l, v = model_mgr.net.apply({'params': params}, obs, train=False)
            l = jnp.where(ns.current_player[:, None] == 0, l, l[:, _ROTATED_IDX])
            l = l - jnp.max(l, axis=-1, keepdims=True)
            l = jnp.where(ns.legal_action_mask, l, jnp.finfo(l.dtype).min)
            return mctx.RecurrentFnOutput(
                reward=ns.rewards[jnp.arange(ns.rewards.shape[0]), prev_p], 
                discount=jnp.where(ns.terminated, 0.0, -1.0), 
                prior_logits=l, value=v
            ), ns
        
        return recurrent_fn

    def _build_replay_snapshots(self) -> List[dict]:
        """构建回放快照列表（每一步的局面）"""
        if not self.state or self.state.jax_state is None:
            return []

        snapshots = []
        for h in self.state.history:
            js = h.get("jax_state")
            if js is None:
                continue
            snapshots.append({
                "board": np.array(js.board),
                "current_player": int(js.current_player),
                "last_move": h.get("last_move"),
                "last_move_uci": h.get("last_move_uci", ""),
                "step_count": int(js.step_count),
                "game_over": bool(js.terminated),
                "winner": int(js.winner),
                "ai_value": float(h.get("ai_value", 0.0)),
                "uci_score": h.get("uci_score"),
                "last_move_player": h.get("last_move_player", 0),
            })

        js = self.state.jax_state
        snapshots.append({
            "board": np.array(js.board),
            "current_player": int(js.current_player),
            "last_move": self.state.last_move,
            "last_move_uci": self.state.last_move_uci,
            "step_count": int(js.step_count),
            "game_over": bool(js.terminated),
            "winner": int(js.winner),
            "ai_value": float(self.state.ai_value),
            "uci_score": self.state.uci_score,
            "last_move_player": self.state.last_move_player,
        })
        return snapshots

    def new_game(self, fen: str = STARTING_FEN):
        board, player = parse_fen(fen)
        self._rng_key, sk = jax.random.split(self._rng_key)
        jax_state = self.env.init(sk)
        jax_board = jnp.array(board, dtype=jnp.int8)
        jax_state = jax_state.replace(
            board=jax_board, current_player=jnp.int32(player),
            legal_action_mask=get_legal_moves_mask(jax_board, jnp.int32(player))
        )
        self.state = GameState(board=board, current_player=player, jax_state=jax_state)
        self._update_status()
        return self.state

    def _update_status(self):
        jb = jnp.array(self.state.board, dtype=jnp.int8)
        p = jnp.int32(self.state.current_player)
        self.state.is_check = bool(is_in_check(jb, p))
        if self.state.is_check:
            r, c = find_king(jb, p)
            self.state.king_pos = (int(r), int(c))
        else: self.state.king_pos = None

    def make_move(self, action: int):
        if self.state.game_over: return
        self.state.replay_index = None
        # 记录走这一步之前的状态
        self.state.history.append({
            'jax_state': self.state.jax_state, 'last_move': self.state.last_move,
            'last_move_uci': self.state.last_move_uci, 'ai_value': self.state.ai_value,
            'uci_score': self.state.uci_score, 'last_move_player': self.state.last_move_player
        })
        fs, ts = action_to_move(action)
        fr, fc, tr, tc = int(fs)//9, int(fs)%9, int(ts)//9, int(ts)%9
        # 记录是谁走的这一步
        self.state.last_move_player = self.state.current_player
        new_jax_state = self.env.step(self.state.jax_state, action)
        self.state.jax_state = new_jax_state
        self.state.board = np.array(new_jax_state.board)
        self.state.current_player = int(new_jax_state.current_player)
        self.state.last_move = (fr, fc, tr, tc)
        self.state.last_move_uci = move_to_uci(int(fs), int(ts))
        self.state.step_count += 1
        self.state.game_over = bool(new_jax_state.terminated)
        self.state.winner = int(new_jax_state.winner)
        self.state.selected = None
        self.state.legal_moves = []
        self._update_status()

    def undo(self):
        if self.state.history:
            h = self.state.history.pop()
            self.state.jax_state = h['jax_state']
            self.state.board = np.array(h['jax_state'].board)
            self.state.current_player = int(h['jax_state'].current_player)
            self.state.last_move = h['last_move']
            self.state.last_move_uci = h['last_move_uci']
            self.state.ai_value = h['ai_value']
            self.state.uci_score = h.get('uci_score')
            self.state.last_move_player = h.get('last_move_player', 0)
            self.state.step_count -= 1
            self.state.game_over = False
            self.state.selected = None
            self.state.legal_moves = []
            self.state.replay_index = None
            self._update_status()

    def fork_from_replay(self):
        """从回放位置分叉，截断后续历史，从该局面继续走棋"""
        replay_idx = self.state.replay_index
        if replay_idx is None:
            return
        
        history_len = len(self.state.history)
        
        if replay_idx == 0:
            # 回到初始局面
            fen = board_to_fen(np.array(self.state.history[0]['jax_state'].board) if self.state.history else self.state.board, 0)
            self.new_game(fen)
            print(f"[分叉] 从初始局面重新开始")
            self.state.notice = "从初始局面重新开始"
        elif replay_idx <= history_len:
            # history[i] 存的是执行第 i+1 步之前的状态
            # 要恢复到 replay_idx 对应的局面（即第 replay_idx 步走完后的状态）
            # 需要使用 history[replay_idx] 的 jax_state（如果存在）
            if replay_idx < history_len:
                # replay_idx 不是最后一步，需要恢复并截断
                h = self.state.history[replay_idx]
                self.state.jax_state = h['jax_state']
                self.state.board = np.array(h['jax_state'].board)
                self.state.current_player = int(h['jax_state'].current_player)
                self.state.last_move = h.get('last_move')
                self.state.last_move_uci = h.get('last_move_uci', '')
                self.state.ai_value = h.get('ai_value', 0.0)
                self.state.uci_score = h.get('uci_score')
                self.state.last_move_player = h.get('last_move_player', 0)
                self.state.step_count = replay_idx
                self.state.game_over = False
                # 截断历史
                self.state.history = self.state.history[:replay_idx]
                print(f"[分叉] 从第 {replay_idx} 步继续，截断 {history_len - replay_idx} 步历史")
                self.state.notice = f"从第 {replay_idx} 步分叉继续"
            # 如果 replay_idx == history_len，说明就是当前局面，不需要恢复
        
        self.state.replay_index = None
        self.state.selected = None
        self.state.legal_moves = []
        self._update_status()

    def get_ai_action(self) -> Optional[int]:
        if not self.model_mgr.params: return None
        obs = self.env.observe(self.state.jax_state)[None, ...]
        logits, value = self.model_mgr.net.apply({'params': self.model_mgr.params}, obs, train=False)
        
        # 使用模块级预计算的 _ROTATED_IDX，避免重复计算
        if self.state.current_player == 1: logits = logits[:, _ROTATED_IDX]
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        logits = jnp.where(self.state.jax_state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)

        self._rng_key, sk = jax.random.split(self._rng_key)
        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=jax.tree.map(lambda x: jnp.expand_dims(x, 0), self.state.jax_state))
        
        # 使用类级别的 recurrent_fn，避免每次调用重新编译
        policy_output = mctx.gumbel_muzero_policy(
            params=self.model_mgr.params, rng_key=sk, root=root, 
            recurrent_fn=self._mcts_recurrent_fn,
            num_simulations=256, max_num_considered_actions=32, 
            invalid_actions=(~self.state.jax_state.legal_action_mask)[None, ...])
        
        # 搜索后的根节点价值更准确
        # search_value 是当前走棋方视角的胜率，需要统一转换为红方视角
        search_value = float(policy_output.search_tree.node_values[0, 0])
        # 如果当前是黑方走棋，取负转换为红方视角
        if self.state.current_player == 1:
            search_value = -search_value
        self.state.ai_value = search_value
        
        # 输出 top-3 候选动作及其权重，方便调试臭棋
        weights = np.array(policy_output.action_weights[0])
        top_indices = np.argsort(weights)[-3:][::-1]
        print(f"[AI] step={self.state.step_count}, value={search_value:.3f}, top3: ", end="")
        for idx in top_indices:
            fs, ts = action_to_move(idx)
            uci = move_to_uci(int(fs), int(ts))
            print(f"{uci}({weights[idx]:.2f}) ", end="")
        print()
        
        return int(jnp.argmax(weights))

    def get_uci_action(self) -> Optional[int]:
        if not self.uci_engine: return None
        bm, score_cp = self.uci_engine.get_best_move(
            board_to_fen(self.state.board, self.state.current_player),
            self.uci_movetime,
            self.uci_depth
        )
        if not bm:
            return None
        if bm in ("(none)", "0000"):
            print(f"[UCI] bestmove 无效: {bm}")
            return None
        try:
            f, t = uci_to_move(bm)
        except Exception as e:
            print(f"[UCI] bestmove 解析失败: {bm}, err={e}")
            return None
        
        # 保存 UCI 评估分数（原始值，当前走棋方视角，正值对走棋方有利）
        if score_cp is not None:
            self.state.uci_score = score_cp
            player_name = "红方" if self.state.current_player == 0 else "黑方"
            print(f"[UCI] score={score_cp}cp ({player_name}视角)")
        
        # 验证动作有效性，move_to_action 返回 -1 表示无效
        action = int(move_to_action(f, t))
        if action < 0:
            print(f"[UCI] bestmove 对应的动作无效: {bm}, from={f}, to={t}")
            return None
        return action

# ============================================================================
# GUI 绘制
# ============================================================================

def render_svg(game: ChessGame) -> str:
    s = game.state
    svg = [f'<svg width="100%" height="100%" viewBox="0 0 {SVG_WIDTH} {SVG_HEIGHT}" xmlns="http://www.w3.org/2000/svg">']
    svg.append(f'<rect width="100%" height="100%" fill="{COLOR_BG}"/>')
    for i in range(9):
        x = BOARD_MARGIN + i * CELL_SIZE
        svg.append(f'<line x1="{x}" y1="{BOARD_MARGIN}" x2="{x}" y2="{BOARD_MARGIN+4*CELL_SIZE}" stroke="{COLOR_LINE}"/>')
        svg.append(f'<line x1="{x}" y1="{BOARD_MARGIN+5*CELL_SIZE}" x2="{x}" y2="{BOARD_MARGIN+9*CELL_SIZE}" stroke="{COLOR_LINE}"/>')
    for i in range(10):
        y = BOARD_MARGIN + i * CELL_SIZE
        svg.append(f'<line x1="{BOARD_MARGIN}" y1="{y}" x2="{BOARD_MARGIN+8*CELL_SIZE}" y2="{y}" stroke="{COLOR_LINE}"/>')
    for y_off in [0, 7*CELL_SIZE]:
        x1, x2, y1, y2 = BOARD_MARGIN+3*CELL_SIZE, BOARD_MARGIN+5*CELL_SIZE, BOARD_MARGIN+y_off, BOARD_MARGIN+y_off+2*CELL_SIZE
        svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{COLOR_LINE}"/><line x1="{x2}" y1="{y1}" x2="{x1}" y2="{y2}" stroke="{COLOR_LINE}"/>')
    if s.last_move:
        for r, c in [(s.last_move[0], s.last_move[1]), (s.last_move[2], s.last_move[3])]:
            x, y = BOARD_MARGIN + c*CELL_SIZE, BOARD_MARGIN + (9-r)*CELL_SIZE
            svg.append(f'<circle cx="{x}" cy="{y}" r="{PIECE_RADIUS+4}" fill="none" stroke="{COLOR_LAST_MOVE}" stroke-width="3"/>')
    if s.selected:
        x, y = BOARD_MARGIN + s.selected[1]*CELL_SIZE, BOARD_MARGIN + (9-s.selected[0])*CELL_SIZE
        svg.append(f'<circle cx="{x}" cy="{y}" r="{PIECE_RADIUS+4}" fill="none" stroke="{COLOR_SELECTED}" stroke-width="3"/>')
    for r, c in s.legal_moves:
        x, y = BOARD_MARGIN + c*CELL_SIZE, BOARD_MARGIN + (9-r)*CELL_SIZE
        svg.append(f'<circle cx="{x}" cy="{y}" r="6" fill="{COLOR_LEGAL}" opacity="0.6"/>')
    for r in range(10):
        for c in range(9):
            p = s.board[r, c]
            if p != 0:
                x, y = BOARD_MARGIN + c*CELL_SIZE, BOARD_MARGIN + (9-r)*CELL_SIZE
                color = COLOR_RED if p > 0 else COLOR_BLACK
                svg.append(f'<circle cx="{x}" cy="{y}" r="{PIECE_RADIUS}" fill="#FDF5E6" stroke="{color}" stroke-width="2"/>')
                svg.append(f'<text x="{x}" y="{y+8}" font-size="26" fill="{color}" text-anchor="middle" font-weight="bold">{PIECE_NAMES[abs(p)][0 if p > 0 else 1]}</text>')
    for r in range(10):
        for c in range(9):
            x, y = BOARD_MARGIN + c*CELL_SIZE, BOARD_MARGIN + (9-r)*CELL_SIZE
            svg.append(f'<rect x="{x-PIECE_RADIUS}" y="{y-PIECE_RADIUS}" width="{PIECE_RADIUS*2}" height="{PIECE_RADIUS*2}" fill="transparent" style="cursor:pointer" onclick="clickBoard({r},{c})"/>')
    svg.append('</svg>')
    return "".join(svg)

# ============================================================================
# Gradio UI
# ============================================================================

def create_ui():
    game = ChessGame()
    css = (
        ".board-col { max-width: 600px; margin: 0 auto; } "
        ".control-col { padding: 15px; } "
        "#hidden_ui { display: none; }"
    )
    with gr.Blocks(css=css, title="ZeroForge") as demo:
        gr.HTML("<h2 style='text-align: center;'>ZeroForge 象棋对弈</h2>")
        with gr.Row():
            with gr.Column(scale=3, elem_classes="board-col"):
                board_svg = gr.HTML()
                status_box = gr.Markdown()
                eval_box = gr.Markdown()
            with gr.Column(scale=2, elem_classes="control-col"):
                with gr.Tabs():
                    with gr.Tab("对弈"):
                        red_p = gr.Dropdown(["Human", "ZeroForge AI", "UCI Engine"], value="Human", label="红方")
                        black_p = gr.Dropdown(["Human", "ZeroForge AI", "UCI Engine"], value="ZeroForge AI", label="黑方")
                        new_btn = gr.Button("开始新局", variant="primary")
                        with gr.Row():
                            undo_btn = gr.Button("悔棋")
                            pause_btn = gr.Button("暂停", variant="secondary", visible=True)
                            continue_btn = gr.Button("继续", variant="primary", visible=False)
                        with gr.Row():
                            replay_prev = gr.Button("◀ 上一步")
                            replay_next = gr.Button("下一步 ▶")
                        replay_current = gr.Button("回到当前")
                        # 可点击的历史走法列表
                        replay_dropdown = gr.Dropdown(
                            choices=[], 
                            value=None, 
                            label="历史走法 (点击跳转)", 
                            interactive=True,
                            allow_custom_value=False
                        )
                    with gr.Tab("设置"):
                        ckpt_dir = gr.Textbox("checkpoints", label="AI 路径")
                        with gr.Row():
                            ckpt_dropdown = gr.Dropdown(choices=[], label="选择步数 (Step)", scale=2)
                            refresh_ckpt = gr.Button("🔄 刷新", scale=1)
                        load_ai = gr.Button("加载所选 AI 模型")
                        uci_path = gr.Textbox("./pikafish", label="UCI 路径")
                        uci_load = gr.Button("启动 UCI")
                        uci_depth = gr.Slider(1, 20, value=3, step=1, label="UCI 深度")
                        ai_delay = gr.Slider(0, 5, value=1, step=0.1, label="AI 延迟(秒)")
                    with gr.Tab("高级"):
                        fen_box = gr.Textbox(label="起始 FEN")
                        fen_current = gr.Textbox(label="当前 FEN", interactive=False)
                        apply_fen = gr.Button("应用 FEN")
        
        # 隐藏的点击触发器（保持 DOM 存在，JS 才能找到）
        with gr.Row(elem_id="hidden_ui", visible=True):
            click_r = gr.Textbox(elem_id="click_r")
            click_c = gr.Textbox(elem_id="click_c")
            click_btn = gr.Button("Click", elem_id="click_btn")

        def build_replay_choices(snapshots, replay_idx):
            """构建历史走法的下拉选项"""
            if not snapshots:
                return [], None
            choices = []
            total = len(snapshots) - 1
            for i, snap in enumerate(snapshots):
                move_uci = snap.get("last_move_uci") or ""
                if not move_uci or i == 0:
                    label = f"第{i}步: 初始局面"
                else:
                    # last_move_player 是走这一步的玩家
                    last_player = snap.get("last_move_player", 0)
                    player_name = "红" if last_player == 0 else "黑"
                    label = f"第{i}步: {move_uci} ({player_name})"
                choices.append(label)
            
            # 当前选中值
            current_idx = replay_idx if replay_idx is not None else (len(snapshots) - 1)
            current_value = choices[current_idx] if current_idx < len(choices) else None
            return choices, current_value

        def update():
            snapshots = game._build_replay_snapshots()
            replay_idx = game.state.replay_index
            if replay_idx is not None and snapshots:
                replay_idx = max(0, min(replay_idx, len(snapshots) - 1))
                snap = snapshots[replay_idx]
                board = snap["board"]
                current_player = snap["current_player"]
                last_move = snap["last_move"]
                last_move_uci = snap["last_move_uci"]
                ai_value = snap["ai_value"]
                uci_score = snap.get("uci_score")
                game_over = snap["game_over"]
                winner = snap["winner"]
                step_count = snap["step_count"]
            else:
                board = game.state.board
                current_player = game.state.current_player
                last_move = game.state.last_move
                last_move_uci = game.state.last_move_uci
                ai_value = game.state.ai_value
                uci_score = game.state.uci_score
                game_over = game.state.game_over
                winner = game.state.winner
                step_count = game.state.step_count

            p_name = "红方" if current_player == 0 else "黑方"
            status = f"### 当前: {p_name} | 第 {step_count} 步"
            if replay_idx is not None:
                status = f"### 回放: 第 {replay_idx} / {len(snapshots) - 1} 步\n\n" + status

            if game_over:
                res = "红胜" if winner == 0 else ("黑胜" if winner == 1 else "和棋")
                status = f"## 🎉 结束: {res}"
            else:
                is_check = bool(is_in_check(jnp.array(board, dtype=jnp.int8), jnp.int32(current_player)))
                if is_check:
                    status += " | ⚠️ **将军**"

            if game.state.notice:
                status += f"\n\n**提示**: {game.state.notice}"
                game.state.notice = ""
            
            # 评估信息构建
            eval_parts = []
            
            # ZeroForge AI 评估（ai_value 范围 [-1, 1]，已统一为红方视角，正值对红方有利）
            # 根据 AI 所属方显示对应胜率
            if game.model_mgr.params is not None:
                red_winrate = (ai_value + 1) / 2 * 100
                black_winrate = 100 - red_winrate
                # 判断 ZeroForge AI 是哪一方
                if game.red_type == "ZeroForge AI" and game.black_type != "ZeroForge AI":
                    eval_parts.append(f"ZeroForge(红): {red_winrate:.1f}%")
                elif game.black_type == "ZeroForge AI" and game.red_type != "ZeroForge AI":
                    eval_parts.append(f"ZeroForge(黑): {black_winrate:.1f}%")
                else:
                    # 双方都是 AI，同时显示红黑双方胜率
                    eval_parts.append(f"AI评估: 红{red_winrate:.1f}% / 黑{black_winrate:.1f}%")
            
            # UCI 引擎评估（原始厘兵分数，正值对当前走棋方有利）
            if uci_score is not None:
                # 判断 UCI 引擎是哪一方
                if game.red_type == "UCI Engine" and game.black_type != "UCI Engine":
                    uci_side = "红"
                elif game.black_type == "UCI Engine" and game.red_type != "UCI Engine":
                    uci_side = "黑"
                else:
                    uci_side = ""
                
                if abs(uci_score) >= 29000:
                    # 将杀局面
                    mate_in = (30000 - abs(uci_score)) // 100
                    uci_eval = f"M{mate_in}" if uci_score > 0 else f"-M{mate_in}"
                else:
                    uci_eval = f"{uci_score:+d}cp"
                eval_parts.append(f"UCI({uci_side}): {uci_eval}")
            
            # 上一着信息
            eval_parts.append(f"着法: {last_move_uci or '无'}")
            
            eval_str = " | ".join(eval_parts) if eval_parts else f"着法: {last_move_uci or '无'}"

            # 为回放渲染临时视图
            if replay_idx is not None:
                temp_game = ChessGame()
                temp_game.state = GameState(board=board, current_player=current_player)
                temp_game.state.last_move = last_move
                svg = render_svg(temp_game)
            else:
                svg = render_svg(game)
            
            # 构建历史走法下拉选项
            choices, current_choice = build_replay_choices(snapshots, replay_idx)
            
            # 暂停/继续按钮的可见性
            pause_visible = not game.paused
            continue_visible = game.paused
            
            return (
                svg, 
                status, 
                board_to_fen(board, current_player), 
                eval_str, 
                gr.update(choices=choices, value=current_choice),
                gr.update(visible=pause_visible),
                gr.update(visible=continue_visible)
            )

        def ai_step():
            if game.state.game_over:
                yield update()
                return

            # 防止递归爆栈：用循环并加安全上限
            max_auto_plies = 200
            plies = 0
            while not game.state.game_over:
                # 检查暂停状态
                if game.paused:
                    print(f"[AI] 已暂停 (step={game.state.step_count})")
                    yield update()
                    return
                
                t = game.red_type if game.state.current_player == 0 else game.black_type
                if t == "Human":
                    break

                if t == "ZeroForge AI":
                    if not game.model_mgr.params:
                        msg = "AI 未加载模型，无法走子，请先在设置中加载模型"
                        print(f"[AI] {msg} (player={game.state.current_player}, step={game.state.step_count})")
                        gr.Warning(msg)
                        break
                    a = game.get_ai_action()
                    if a is None:
                        raise RuntimeError(
                            "AI 未返回动作(模型已加载): "
                            f"player={game.state.current_player}, step={game.state.step_count}, "
                            f"last={game.state.last_move_uci}"
                        )
                    game.make_move(a)
                elif t == "UCI Engine":
                    if not game.uci_engine:
                        msg = "UCI 引擎未启动，无法走子，请先启动 UCI"
                        print(f"[UCI] {msg} (player={game.state.current_player}, step={game.state.step_count})")
                        gr.Warning(msg)
                        break
                    a = game.get_uci_action()
                    if a is None:
                        raise RuntimeError(
                            "UCI 未返回动作(引擎已启动): "
                            f"player={game.state.current_player}, step={game.state.step_count}, "
                            f"last={game.state.last_move_uci}"
                        )
                    game.make_move(a)
                else:
                    raise RuntimeError(f"未知玩家类型: {t}")

                # 实时渲染：每走一步就产出一次 UI
                yield update()
                if game.ai_delay > 0:
                    time.sleep(game.ai_delay)

                plies += 1
                if plies >= max_auto_plies:
                    raise RuntimeError(
                        "自动走子超过上限，可能出现死循环。"
                        f" player={game.state.current_player}, step={game.state.step_count}"
                    )
            yield update()

        def on_click(r, c):
            try:
                r, c = int(r), int(c)
                
                # 如果在回放模式，先恢复到该历史局面（分叉走棋）
                if game.state.replay_index is not None:
                    game.fork_from_replay()
                
                s = game.state
                p = s.board[r, c]
                own = (s.current_player == 0 and p > 0) or (s.current_player == 1 and p < 0)

                if s.selected:
                    a = move_to_action(s.selected[0]*9 + s.selected[1], r*9 + c)
                    if a != -1 and s.jax_state.legal_action_mask[a]:
                        game.make_move(int(a))
                        yield from ai_step()
                        return
                    elif own:
                        s.selected = (r, c)
                        s.legal_moves = []
                        mask = s.jax_state.legal_action_mask
                        for tr in range(10):
                            for tc in range(9):
                                if mask[move_to_action(r*9+c, tr*9+tc)]: s.legal_moves.append((tr, tc))
                    else:
                        s.selected, s.legal_moves = None, []
                elif own:
                    s.selected = (r, c)
                    s.legal_moves = []
                    mask = s.jax_state.legal_action_mask
                    for tr in range(10):
                        for tc in range(9):
                            if mask[move_to_action(r*9+c, tr*9+tc)]: s.legal_moves.append((tr, tc))
            except Exception as e:
                print(f"Click logic error: {e}")
                import traceback
                traceback.print_exc()
            yield update()

        def handle_load_ai(d, s):
            try:
                if not d or not os.path.isdir(d):
                    msg = f"AI 路径不存在: {d}"
                    print(f"[AI] {msg}")
                    gr.Warning(msg)
                    game.state.notice = msg
                    return gr.update(), update()

                if not s:
                    steps = list_checkpoints(d)
                    if not steps:
                        msg = f"未找到检查点: {d}"
                        print(f"[AI] {msg}")
                        gr.Warning(msg)
                        game.state.notice = msg
                        return gr.update(), update()
                    s = steps[0]
                else:
                    steps = list_checkpoints(d)
                    if steps and int(s) not in steps:
                        msg = f"检查点不存在: step={s}, dir={d}"
                        print(f"[AI] {msg}, steps={steps}")
                        gr.Warning(msg)
                        game.state.notice = msg
                        return gr.update(), update()

                print(f"[AI] 加载模型: dir={d}, step={s}, steps={steps}")
                success = game.model_mgr.load(d, int(s))
                if success:
                    msg = f"模型加载成功: step {s}"
                    gr.Info(msg)
                    game.state.notice = msg
                else:
                    msg = f"模型加载失败: step {s}"
                    print(f"[AI] {msg}")
                    gr.Warning(msg)
                    game.state.notice = msg
            except Exception as e:
                print(f"[AI] 加载异常: {e}")
                print(traceback.format_exc())
                gr.Error(f"加载异常: {str(e)}")
                game.state.notice = f"加载异常: {str(e)}"
            return gr.update(), update() # First update is for Info/Warning, not used

        def handle_refresh_ckpt(d):
            steps = list_checkpoints(d)
            if not steps:
                gr.Warning(f"目录 {d} 下未找到数字编号的检查点")
                return gr.update(choices=[], value=None)
            return gr.update(choices=[str(s) for s in steps], value=str(steps[0]))

        def handle_load_uci(p):
            try:
                if game.uci_engine:
                    game.uci_engine.stop()
                game.uci_engine = UCIEngine(p)
                if game.uci_engine.start():
                    gr.Info("UCI 引擎启动成功")
                else:
                    gr.Warning("UCI 引擎启动失败，请检查路径 (默认 ./pikafish)")
            except Exception as e:
                gr.Error(f"引擎异常: {str(e)}")
            return update()

        def handle_uci_depth(d):
            game.uci_depth = int(d)
            print(f"[UCI] 深度已设置为 {game.uci_depth}")
            return update()

        def handle_ai_delay(d):
            game.ai_delay = float(d)
            print(f"[AI] 延迟已设置为 {game.ai_delay} 秒")
            return update()

        def handle_red_type_change(t):
            """红方类型变化时实时更新"""
            game.red_type = t
            print(f"[设置] 红方类型: {t}")

        def handle_black_type_change(t):
            """黑方类型变化时实时更新"""
            game.black_type = t
            print(f"[设置] 黑方类型: {t}")

        def handle_init():
            # 初始化时自动刷新一次检查点列表
            steps = list_checkpoints("checkpoints")
            game.new_game()
            u = update()
            # u 包含 7 个元素: svg, status, fen, eval, replay_dropdown, pause_btn, continue_btn
            return u[0], u[1], u[2], u[3], u[4], u[5], u[6], gr.update(
                choices=[str(s) for s in steps],
                value=str(steps[0]) if steps else None
            )

        def handle_new_game(r, b, f):
            game.red_type = r
            game.black_type = b
            game.state.replay_index = None
            game.paused = False  # 新局重置暂停状态
            fen = f.strip() if isinstance(f, str) else ""
            try:
                game.new_game(fen if fen else STARTING_FEN)
            except Exception as e:
                print(f"[FEN] 新局 FEN 解析失败: {e}")
                gr.Error(f"FEN 解析失败: {str(e)}")
                game.new_game()
            yield from ai_step()

        def handle_undo():
            game.undo()
            return update()

        def handle_apply_fen(f):
            game.state.replay_index = None
            game.new_game(f)
            return update()

        def handle_replay_prev():
            snaps = game._build_replay_snapshots()
            if not snaps:
                return update()
            idx = game.state.replay_index
            if idx is None:
                idx = len(snaps) - 1
            game.state.replay_index = max(0, idx - 1)
            return update()

        def handle_replay_next():
            snaps = game._build_replay_snapshots()
            if not snaps:
                return update()
            idx = game.state.replay_index
            if idx is None:
                idx = len(snaps) - 1
            game.state.replay_index = min(len(snaps) - 1, idx + 1)
            return update()

        def handle_replay_current():
            game.state.replay_index = None
            return update()

        def handle_pause():
            """暂停：设置暂停状态（此事件会取消正在执行的 AI 走棋）"""
            game.paused = True
            print(f"[AI] 已暂停 (step={game.state.step_count})")
            return update()
        
        def handle_continue():
            """继续：取消暂停并继续 AI 走棋"""
            game.paused = False
            print(f"[AI] 继续走棋 (step={game.state.step_count})")
            yield from ai_step()

        def handle_replay_select(choice):
            """通过下拉列表选择跳转到某一步"""
            if not choice:
                return update()
            # 解析选中的步数，格式: "第N步: ..."
            try:
                step_str = choice.split(":")[0]  # "第N步"
                step_num = int(step_str.replace("第", "").replace("步", ""))
                # 避免重复设置相同的步数
                if game.state.replay_index == step_num:
                    return update()
                game.state.replay_index = step_num
                print(f"[回放] 跳转到第 {step_num} 步")
            except Exception as e:
                print(f"[回放] 解析失败: {choice}, err={e}")
            return update()

        # --- 事件绑定 ---
        ui_outputs = [board_svg, status_box, fen_current, eval_box, replay_dropdown, pause_btn, continue_btn]
        
        # 包含 AI 走棋的事件（需要能被取消）
        click_event = click_btn.click(on_click, [click_r, click_c], ui_outputs)
        
        # 红黑方类型变化时实时更新
        red_p.change(handle_red_type_change, [red_p])
        black_p.change(handle_black_type_change, [black_p])
        
        # 继续按钮：继续 AI 走棋
        continue_event = continue_btn.click(handle_continue, outputs=ui_outputs)
        
        # 新开局：取消所有正在执行的 AI 走棋事件，避免多个游戏同时运行
        new_game_event = new_btn.click(
            handle_new_game, 
            [red_p, black_p, fen_box], 
            ui_outputs,
            cancels=[click_event, continue_event]  # 取消旧游戏的事件
        )
        undo_btn.click(handle_undo, outputs=ui_outputs)
        
        # 暂停按钮：取消正在执行的 AI 走棋事件
        pause_btn.click(
            handle_pause, 
            outputs=ui_outputs,
            cancels=[click_event, new_game_event, continue_event]
        )
        
        replay_prev.click(handle_replay_prev, outputs=ui_outputs)
        replay_next.click(handle_replay_next, outputs=ui_outputs)
        replay_current.click(handle_replay_current, outputs=ui_outputs)
        replay_dropdown.change(handle_replay_select, [replay_dropdown], ui_outputs)
        
        refresh_ckpt.click(handle_refresh_ckpt, [ckpt_dir], [ckpt_dropdown])
        
        def handle_load_ai_final(d, s):
            _, u = handle_load_ai(d, s)
            return u
        load_ai.click(handle_load_ai_final, [ckpt_dir, ckpt_dropdown], ui_outputs)

        uci_load.click(handle_load_uci, [uci_path], ui_outputs)
        uci_depth.change(handle_uci_depth, [uci_depth], ui_outputs)
        ai_delay.change(handle_ai_delay, [ai_delay], ui_outputs)
        apply_fen.click(handle_apply_fen, [fen_box], ui_outputs)
        
        # 初始化
        demo.load(handle_init, outputs=ui_outputs + [ckpt_dropdown])
        
        # JS 点击逻辑增强
        js_code = """
        function() {
            window.clickBoard = function(r, c) {
                console.log("Board clicked:", r, c);
                const r_box = document.getElementById('click_r');
                const c_box = document.getElementById('click_c');
                const btn = document.getElementById('click_btn');
                const r_el = r_box ? r_box.querySelector('input, textarea') : null;
                const c_el = c_box ? c_box.querySelector('input, textarea') : null;
                
                if (r_el && c_el && btn) {
                    const setValue = (el, val) => {
                        el.value = val;
                        el.dispatchEvent(new Event('input', { bubbles: true }));
                        el.dispatchEvent(new Event('change', { bubbles: true }));
                    };
                    setValue(r_el, r.toString());
                    setValue(c_el, c.toString());
                    // 延迟触发按钮点击
                    setTimeout(() => {
                        btn.click();
                    }, 20);
                } else {
                    console.error("Required elements not found:", {r_box, c_box, r_el, c_el, btn});
                }
            };
        }
        """
        demo.load(None, None, js=js_code)
        demo.queue()
    return demo

def run_web_gui(share=False): create_ui().launch(share=share, server_name="0.0.0.0")
if __name__ == "__main__": run_web_gui()

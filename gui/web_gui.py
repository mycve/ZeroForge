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

    def get_best_move(self, fen: str, movetime: int = 1000, depth: Optional[int] = None) -> Tuple[Optional[str], Optional[float]]:
        """返回 (bestmove, score)，score 为当前方视角的评估值 (-1 到 +1)"""
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
                    # 解析 info 行中的评估分数
                    if line.startswith("info") and " score " in line:
                        last_score = self._parse_score(line)
                    if line.startswith("bestmove"):
                        return line.split()[1], last_score
                except queue.Empty: continue
        return None, None
    
    def _parse_score(self, info_line: str) -> Optional[float]:
        """从 UCI info 行解析评估分数，转换为 -1 到 +1 的值"""
        try:
            parts = info_line.split()
            if "score" not in parts:
                return None
            idx = parts.index("score")
            score_type = parts[idx + 1]
            score_val = int(parts[idx + 2])
            
            if score_type == "mate":
                # mate N 表示 N 步杀棋，正数是己方赢，负数是对方赢
                return 1.0 if score_val > 0 else -1.0
            elif score_type == "cp":
                # centipawn，使用 tanh 归一化到 -1 ~ +1
                # cp=400 约等于 0.76，cp=1000 约等于 0.96
                import math
                return math.tanh(score_val / 400.0)
        except Exception as e:
            print(f"[UCI] 评估解析失败: {info_line}, err={e}")
        return None

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
    ai_value: float = 0.0  # 保留兼容，但不再主要使用
    # 分别记录红方和黑方的评估（谁接管谁输出）
    red_eval: Optional[float] = None    # 红方AI/UCI的评估（红方视角胜率）
    black_eval: Optional[float] = None  # 黑方AI/UCI的评估（黑方视角胜率）
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
                "red_eval": h.get("red_eval"),
                "black_eval": h.get("black_eval"),
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
            "red_eval": self.state.red_eval,
            "black_eval": self.state.black_eval,
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
        self.state.history.append({
            'jax_state': self.state.jax_state, 'last_move': self.state.last_move,
            'last_move_uci': self.state.last_move_uci, 'ai_value': self.state.ai_value,
            'red_eval': self.state.red_eval, 'black_eval': self.state.black_eval
        })
        fs, ts = action_to_move(action)
        fr, fc, tr, tc = int(fs)//9, int(fs)%9, int(ts)//9, int(ts)%9
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
            self.state.red_eval = h.get('red_eval')
            self.state.black_eval = h.get('black_eval')
            self.state.step_count -= 1
            self.state.game_over = False
            self.state.selected = None
            self.state.legal_moves = []
            self.state.replay_index = None
            self._update_status()

    def get_ai_action(self) -> Optional[int]:
        if not self.model_mgr.params: return None
        obs = self.env.observe(self.state.jax_state)[None, ...]
        logits, value = self.model_mgr.net.apply({'params': self.model_mgr.params}, obs, train=False)
        
        _ROTATED_IDX = rotate_action(jnp.arange(ACTION_SPACE_SIZE))
        
        if self.state.current_player == 1: logits = logits[:, _ROTATED_IDX]
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        logits = jnp.where(self.state.jax_state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)
        
        def recurrent_fn(model, rng_key, action, state):
            prev_p = state.current_player
            ns = jax.vmap(self.env.step)(state, action)
            obs = jax.vmap(self.env.observe)(ns)
            l, v = self.model_mgr.net.apply({'params': self.model_mgr.params}, obs, train=False)
            l = jnp.where(ns.current_player[:, None] == 0, l, l[:, _ROTATED_IDX])
            l = l - jnp.max(l, axis=-1, keepdims=True)
            l = jnp.where(ns.legal_action_mask, l, jnp.finfo(l.dtype).min)
            return mctx.RecurrentFnOutput(reward=ns.rewards[jnp.arange(ns.rewards.shape[0]), prev_p], 
                                          discount=jnp.where(ns.terminated, 0.0, -1.0), prior_logits=l, value=v), ns

        self._rng_key, sk = jax.random.split(self._rng_key)
        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=jax.tree.map(lambda x: jnp.expand_dims(x, 0), self.state.jax_state))
        policy_output = mctx.gumbel_muzero_policy(params=None, rng_key=sk, root=root, recurrent_fn=recurrent_fn,
            num_simulations=256, max_num_considered_actions=16, invalid_actions=(~self.state.jax_state.legal_action_mask)[None, ...])
        
        # 模型输出的 value 是从当前玩家视角的评估
        current_player_eval = float(value[0])
        self.state.ai_value = current_player_eval  # 保留兼容
        
        # 根据当前玩家存储到对应字段（各方只显示自己的评估）
        if self.state.current_player == 0:
            self.state.red_eval = current_player_eval
            print(f"[AI] 红方评估: {current_player_eval:.4f} (胜率 {(current_player_eval+1)/2*100:.1f}%)")
        else:
            self.state.black_eval = current_player_eval
            print(f"[AI] 黑方评估: {current_player_eval:.4f} (胜率 {(current_player_eval+1)/2*100:.1f}%)")
        
        return int(jnp.argmax(policy_output.action_weights[0]))

    def get_uci_action(self) -> Optional[int]:
        if not self.uci_engine: return None
        bm, score = self.uci_engine.get_best_move(
            board_to_fen(self.state.board, self.state.current_player),
            self.uci_movetime,
            self.uci_depth
        )
        if not bm:
            return None
        if bm in ("(none)", "0000"):
            print(f"[UCI] bestmove 无效: {bm}")
            return None
        
        # UCI 返回的 score 是从当前方视角的评估
        if score is not None:
            if self.state.current_player == 0:
                self.state.red_eval = score
                print(f"[UCI] 红方评估: {score:.4f} (胜率 {(score+1)/2*100:.1f}%)")
            else:
                self.state.black_eval = score
                print(f"[UCI] 黑方评估: {score:.4f} (胜率 {(score+1)/2*100:.1f}%)")
        
        try:
            f, t = uci_to_move(bm)
        except Exception as e:
            print(f"[UCI] bestmove 解析失败: {bm}, err={e}")
            return None
        return int(move_to_action(f, t))

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
                        undo_btn = gr.Button("悔棋")
                        with gr.Row():
                            replay_prev = gr.Button("回放上一步")
                            replay_next = gr.Button("回放下一步")
                        replay_current = gr.Button("回到当前")
                        replay_list = gr.Markdown()
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

        def build_replay_markdown(snapshots, replay_idx):
            if not snapshots:
                return "暂无回放记录"
            lines = []
            total = len(snapshots) - 1
            for i, snap in enumerate(snapshots):
                move_uci = snap.get("last_move_uci") or "初始局面"
                tag = ">>" if replay_idx == i else "  "
                lines.append(f"{tag} 第{i}步/{total}: {move_uci}")
            return "\n".join(lines)

        def update():
            # 如果游戏状态未初始化，先初始化
            if game.state is None:
                game.new_game()
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
                red_eval = snap.get("red_eval")
                black_eval = snap.get("black_eval")
                game_over = snap["game_over"]
                winner = snap["winner"]
                step_count = snap["step_count"]
            else:
                board = game.state.board
                current_player = game.state.current_player
                last_move = game.state.last_move
                last_move_uci = game.state.last_move_uci
                ai_value = game.state.ai_value
                red_eval = game.state.red_eval
                black_eval = game.state.black_eval
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
            
            # 分别显示红方和黑方的评估（各方只显示自己的评估，可能是AI或UCI）
            eval_parts = []
            if red_eval is not None:
                red_winrate = (red_eval + 1) / 2 * 100
                eval_parts.append(f"**红方评估**: {red_winrate:.1f}%")
            else:
                eval_parts.append("**红方评估**: --")
            
            if black_eval is not None:
                black_winrate = (black_eval + 1) / 2 * 100
                eval_parts.append(f"**黑方评估**: {black_winrate:.1f}%")
            else:
                eval_parts.append("**黑方评估**: --")
            
            eval_str = " | ".join(eval_parts) + f" | 上一着: {last_move_uci or '无'}"

            # 为回放渲染临时视图
            if replay_idx is not None:
                temp_game = ChessGame()
                temp_game.state = GameState(board=board, current_player=current_player)
                temp_game.state.last_move = last_move
                svg = render_svg(temp_game)
            else:
                svg = render_svg(game)
            move_list_md = build_replay_markdown(snapshots, replay_idx)
            return svg, status, board_to_fen(board, current_player), eval_str, move_list_md

        def ai_step():
            if game.state.game_over:
                yield update()
                return

            # 防止递归爆栈：用循环并加安全上限
            max_auto_plies = 200
            plies = 0
            while not game.state.game_over:
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
                game.state.replay_index = None
                r, c = int(r), int(c)
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

        def handle_init():
            # 初始化时自动刷新一次检查点列表
            steps = list_checkpoints("checkpoints")
            game.new_game()
            u = update()
            return u[0], u[1], u[2], u[3], u[4], gr.update(
                choices=[str(s) for s in steps],
                value=str(steps[0]) if steps else None
            )

        def handle_new_game(r, b, f):
            game.red_type = r
            game.black_type = b
            game.state.replay_index = None
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

        # --- 事件绑定 ---
        ui_outputs = [board_svg, status_box, fen_current, eval_box, replay_list]
        
        click_btn.click(on_click, [click_r, click_c], ui_outputs)
        
        new_btn.click(handle_new_game, [red_p, black_p, fen_box], ui_outputs)
        undo_btn.click(handle_undo, outputs=ui_outputs)
        replay_prev.click(handle_replay_prev, outputs=ui_outputs)
        replay_next.click(handle_replay_next, outputs=ui_outputs)
        replay_current.click(handle_replay_current, outputs=ui_outputs)
        
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

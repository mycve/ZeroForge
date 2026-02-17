"""
ZeroForge Web API - FastAPI 后端
提供象棋对弈的 REST API，支持 ZeroForge AI 和 UCI 引擎
"""

import os
import sys
import time
import asyncio
import subprocess
import threading
import queue
import concurrent.futures
from functools import partial
from pathlib import Path
from typing import Optional, List, Tuple
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import mctx

from xiangqi.env import XiangqiEnv, XiangqiState, NUM_OBSERVATION_CHANNELS
from xiangqi.rules import get_legal_moves_mask, is_in_check, find_king, BOARD_WIDTH, BOARD_HEIGHT
from xiangqi.actions import (
    move_to_action, action_to_move, move_to_uci, uci_to_move,
    ACTION_SPACE_SIZE, rotate_action
)
from networks.alphazero import AlphaZeroNetwork

# ============================================================================
# 常量
# ============================================================================

STARTING_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"
_ROTATED_IDX = rotate_action(jnp.arange(ACTION_SPACE_SIZE))
MCTS_QTRANSFORM_VALUE_SCALE = 0.25
MCTS_GUMBEL_SCALE = 0.0  # 实战默认关闭探索噪声，最大化走子强度
MCTS_QTRANSFORM = partial(
    mctx.qtransform_completed_by_mix_value,
    value_scale=MCTS_QTRANSFORM_VALUE_SCALE,
)

# Pikafish 路径（程序目录下）
PIKAFISH_PATH = Path(__file__).parent.parent / "pikafish"

# ============================================================================
# UCI 引擎（全局单例）
# ============================================================================

class UCIEngine:
    """UCI 引擎封装，全局单例"""
    
    def __init__(self, path: str):
        self.path = path
        self.process = None
        self.output_queue = queue.Queue()
        self._stop_event = threading.Event()
        self.lock = threading.Lock()
        self.ready = False

    def start(self) -> bool:
        """启动引擎"""
        if self.process and self.process.poll() is None:
            return True  # 已经在运行
        try:
            self.process = subprocess.Popen(
                [self.path], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.PIPE, text=True, bufsize=1
            )
            self._stop_event.clear()
            threading.Thread(target=self._read_stdout, daemon=True).start()
            self.send("uci")
            # 等待 uciok
            start = time.time()
            while time.time() - start < 5:
                try:
                    line = self.output_queue.get(timeout=0.1)
                    if "uciok" in line:
                        self.ready = True
                        print(f"[UCI] 引擎启动成功: {self.path}")
                        return True
                except queue.Empty:
                    continue
            print(f"[UCI] 引擎启动超时")
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
            # 等待 readyok
            start = time.time()
            while time.time() - start < 2:
                try:
                    line = self.output_queue.get(timeout=0.1)
                    if "readyok" in line:
                        print("[UCI] Hash 已清空 (ucinewgame)")
                        return True
                except queue.Empty:
                    continue
            return False

    def get_best_move(self, fen: str, depth: int = 10) -> Tuple[Optional[str], Optional[int]]:
        """获取最佳着法和评估分数"""
        with self.lock:
            # 清空队列
            while not self.output_queue.empty():
                self.output_queue.get()
            
            self.send(f"position fen {fen}")
            self.send(f"go depth {depth}")
            
            start_time = time.time()
            wait_seconds = max(2.0, depth * 0.5)
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
        if self.process:
            self.process.terminate()


# 全局 UCI 引擎实例
uci_engine: Optional[UCIEngine] = None


def init_uci_engine():
    """初始化全局 UCI 引擎"""
    global uci_engine
    if PIKAFISH_PATH.exists():
        uci_engine = UCIEngine(str(PIKAFISH_PATH))
        if uci_engine.start():
            print(f"[UCI] Pikafish 全局引擎已启动")
        else:
            print(f"[UCI] Pikafish 启动失败，UCI 功能不可用")
            uci_engine = None
    else:
        print(f"[UCI] 未找到 Pikafish: {PIKAFISH_PATH}，UCI 功能不可用")


# ============================================================================
# AI 模型管理
# ============================================================================

class ModelManager:
    def __init__(self):
        self.params = None
        self.net = None
        self.step = 0
        self.channels = 0
        self.num_blocks = 0
        self.last_error = ""

    def _infer_channels(self, params) -> Optional[int]:
        # GNN 结构：首个 Dense_0 的输出维度即 channels
        try:
            if hasattr(params, "get") and params.get("Dense_0") is not None:
                dense0 = params.get("Dense_0")
                return int(dense0["kernel"].shape[-1])
            if "Dense_0" in params:
                dense0 = params["Dense_0"]
                return int(dense0["kernel"].shape[-1])
        except Exception:
            pass
        # 兜底：扫描 Dense_* 的 kernel 形状
        try:
            for k in params.keys():
                if str(k).startswith("Dense_"):
                    kernel = params[k]["kernel"]
                    if kernel.ndim == 2:
                        return int(kernel.shape[-1])
        except Exception:
            return None
        return None

    def _infer_num_blocks(self, params) -> int:
        try:
            return len([k for k in params.keys() if str(k).startswith("GraphBlock_")])
        except Exception:
            return 0

    def load(self, ckpt_dir: str, step: int) -> bool:
        self.last_error = ""
        ckpt_dir = os.path.abspath(ckpt_dir)
        ckpt_manager = ocp.CheckpointManager(ckpt_dir)
        if step == 0:
            step = ckpt_manager.latest_step()
        if step is None:
            self.last_error = f"未找到 checkpoint: {ckpt_dir}"
            return False

        restored = None
        try:
            restored = ckpt_manager.restore(step)
        except Exception:
            try:
                ckpt_path = os.path.join(ckpt_dir, str(step))
                restored = ocp.StandardCheckpointer().restore(ckpt_path)
            except Exception as e:
                self.last_error = f"Checkpoint 恢复失败: {e}"
                print(f"[AI] {self.last_error}")
                return False

        params = None
        if isinstance(restored, dict) or hasattr(restored, "keys"):
            if "params" in restored:
                params = restored["params"]
            elif "default" in restored and isinstance(restored["default"], dict):
                params = restored["default"].get("params")

        if params is None:
            self.last_error = "Checkpoint 中未找到 params"
            print(f"[AI] {self.last_error}")
            return False

        channels = self._infer_channels(params)
        num_blocks = self._infer_num_blocks(params)
        if not channels or num_blocks <= 0:
            self.last_error = (
                f"模型结构推断失败: channels={channels}, num_blocks={num_blocks} "
                "(可能是旧 CNN checkpoint 或参数格式不匹配)"
            )
            print(f"[AI] {self.last_error}")
            return False

        self.net = AlphaZeroNetwork(
            action_space_size=ACTION_SPACE_SIZE,
            channels=channels,
            num_blocks=num_blocks,
        )
        # 立即做一次前向验证，提前发现 checkpoint 与当前网络结构不兼容的问题
        try:
            dummy_obs = jnp.zeros(
                (1, NUM_OBSERVATION_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH), dtype=jnp.float32
            )
            _ = self.net.apply({'params': params}, dummy_obs, train=False)
        except Exception as e:
            self.net = None
            self.params = None
            self.last_error = f"Checkpoint 与当前网络结构不兼容: {e}"
            print(f"[AI] {self.last_error}")
            return False

        self.params = params
        self.step = step
        self.channels = channels
        self.num_blocks = num_blocks
        print(f"[AI] 模型加载完成: step={step}, channels={channels}, blocks={num_blocks}")
        return True


# 全局模型管理器
model_mgr = ModelManager()
env = XiangqiEnv()
rng_key = jax.random.PRNGKey(int(time.time()))


# ============================================================================
# 游戏状态
# ============================================================================

@dataclass
class GameState:
    board: np.ndarray
    current_player: int
    jax_state: Optional[XiangqiState] = None
    last_move: Optional[Tuple[int, int, int, int]] = None
    last_move_uci: str = ""
    game_over: bool = False
    winner: int = -1
    step_count: int = 0
    is_check: bool = False
    history: List = field(default_factory=list)  # 状态历史（用于悔棋和前进）
    move_history: List = field(default_factory=list)  # 着法历史（用于显示）
    ai_value: float = 0.0
    uci_score: Optional[int] = None
    current_history_index: int = -1  # 当前在历史中的位置，-1表示在最新状态


# 全局游戏状态
game_state: Optional[GameState] = None


# ============================================================================
# 工具函数
# ============================================================================

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
            if char.isdigit():
                col += int(char)
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
            if p == 0:
                empty += 1
            else:
                if empty > 0:
                    r_str += str(empty)
                    empty = 0
                r_str += PIECE_TO_FEN[int(p)]
        if empty > 0:
            r_str += str(empty)
        rows.append(r_str)
    return "/".join(rows) + (" w" if player == 0 else " b")


def list_checkpoints(ckpt_dir: str) -> List[int]:
    if not os.path.exists(ckpt_dir):
        return []
    steps = []
    for d in os.listdir(ckpt_dir):
        if os.path.isdir(os.path.join(ckpt_dir, d)) and d.isdigit():
            steps.append(int(d))
    return sorted(steps, reverse=True)


def get_legal_moves_list(jax_state: XiangqiState) -> List[dict]:
    """获取合法着法列表"""
    mask = np.array(jax_state.legal_action_mask)
    moves = []
    for action in np.where(mask)[0]:
        fs, ts = action_to_move(action)
        fs, ts = int(fs), int(ts)
        fr, fc = fs // 9, fs % 9
        tr, tc = ts // 9, ts % 9
        moves.append({
            "action": int(action),
            "from": [fr, fc],
            "to": [tr, tc],
            "uci": move_to_uci(fs, ts)
        })
    return moves


# ============================================================================
# MCTS 推理
# ============================================================================


def create_mcts_recurrent_fn():
    """创建 MCTS 递归函数"""
    def recurrent_fn(params, rng_key, action, state):
        prev_p = state.current_player
        ns = jax.vmap(env.step)(state, action)
        obs = jax.vmap(env.observe)(ns)
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


mcts_recurrent_fn = None


def get_ai_action(
    state: GameState, 
    num_simulations: int = 256, 
    top_k: int = 32,
    return_policy: bool = False,  # 是否返回完整策略分布
) -> Tuple[Optional[int], float, Optional[dict]]:
    """获取 AI 最佳着法（始终选择最大概率走法）
    
    Args:
        state: 当前游戏状态
        num_simulations: MCTS 模拟次数（越大越准，越慢）
        top_k: 每步考虑的最大着法数（越大越全面，越慢）
        return_policy: 是否返回完整策略分布
    
    Returns:
        (action, value, policy_info) - policy_info 包含策略分布信息，如果return_policy=False则为None
    """
    global rng_key, mcts_recurrent_fn
    
    if not model_mgr.params:
        return None, 0.0, None
    
    # 优化：只有一个合法走法时直接返回，不需要MCTS搜索
    legal_mask = np.array(state.jax_state.legal_action_mask)
    legal_actions = np.where(legal_mask)[0]
    if len(legal_actions) == 1:
        action = int(legal_actions[0])
        print(f"[AI] 只有一个合法走法，直接应用: action={action}")
        # 仍需获取value估计
        obs = env.observe(state.jax_state)[None, ...]
        _, value = model_mgr.net.apply({'params': model_mgr.params}, obs, train=False)
        search_value = float(value[0])
        if state.current_player == 1:
            search_value = -search_value
        
        policy_info = None
        if return_policy:
            policy_info = {
                "top_moves": [{
                    "action": action,
                    "uci": move_to_uci(*action_to_move(action)),
                    "weight": 1.0,
                    "prior": 1.0,
                }],
                "is_forced": True,
            }
        
        return action, search_value, policy_info
    
    if mcts_recurrent_fn is None:
        mcts_recurrent_fn = create_mcts_recurrent_fn()
    
    obs = env.observe(state.jax_state)[None, ...]
    logits, value = model_mgr.net.apply({'params': model_mgr.params}, obs, train=False)
    
    if state.current_player == 1:
        logits = logits[:, _ROTATED_IDX]
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    logits = jnp.where(state.jax_state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)

    invalid_mask = ~state.jax_state.legal_action_mask
    
    # 计算先验策略分布（用于policy_info）
    prior_probs = jax.nn.softmax(logits[0])

    rng_key, sk1 = jax.random.split(rng_key)
    root = mctx.RootFnOutput(
        prior_logits=logits, value=value,
        embedding=jax.tree.map(lambda x: jnp.expand_dims(x, 0), state.jax_state)
    )
    
    policy_output = mctx.gumbel_muzero_policy(
        params=model_mgr.params, rng_key=sk1, root=root,
        recurrent_fn=mcts_recurrent_fn,
        num_simulations=num_simulations, 
        max_num_considered_actions=top_k,
        invalid_actions=invalid_mask[None, ...],
        qtransform=MCTS_QTRANSFORM,
        gumbel_scale=MCTS_GUMBEL_SCALE,
    )
    
    search_value = float(policy_output.search_tree.node_values[0, 0])
    if state.current_player == 1:
        search_value = -search_value
    
    weights = np.array(policy_output.action_weights[0])
    
    # 始终选择最大概率走法（贪婪策略）
    action = int(np.argmax(weights))
    
    # 构建策略分布信息
    policy_info = None
    if return_policy:
        # 获取Top-K候选着法
        top_indices = np.argsort(weights)[::-1][:min(10, len(legal_actions))]  # 返回Top10
        top_moves = []
        for idx in top_indices:
            if weights[idx] > 1e-6:  # 过滤权重太小的
                fs, ts = action_to_move(int(idx))
                top_moves.append({
                    "action": int(idx),
                    "uci": move_to_uci(int(fs), int(ts)),
                    "weight": float(weights[idx]),
                    "prior": float(prior_probs[idx]),
                })
        
        policy_info = {
            "top_moves": top_moves,
            "is_forced": False,
        }
    
    return action, search_value, policy_info


# ============================================================================
# API 模型
# ============================================================================

class NewGameRequest(BaseModel):
    fen: str = STARTING_FEN


class MoveRequest(BaseModel):
    action: int


class LoadModelRequest(BaseModel):
    ckpt_dir: str = "checkpoints"
    step: int = 0


class AIThinkRequest(BaseModel):
    num_simulations: int = 256  # MCTS 模拟次数
    top_k: int = 32             # 考虑的最大着法数
    return_policy: bool = False  # 是否返回完整策略分布
    opening_random_moves: int = 0  # 开局前N步使用随机策略（0=不使用）
    opening_top_k: int = 5  # 开局随机时从Top-K候选中随机选择


class UCIThinkRequest(BaseModel):
    depth: int = 10


# ============================================================================
# FastAPI 应用
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理"""
    # 启动时初始化 UCI 引擎
    init_uci_engine()
    yield
    # 关闭时停止 UCI 引擎
    if uci_engine:
        uci_engine.stop()


app = FastAPI(title="ZeroForge Chess API", lifespan=lifespan)

# 静态文件服务
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def index():
    """返回前端页面"""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    raise HTTPException(status_code=404, detail="index.html not found")


@app.get("/api/status")
async def get_status():
    """获取当前游戏状态"""
    if game_state is None:
        return {"error": "No game in progress"}
    
    jb = jnp.array(game_state.board, dtype=jnp.int8)
    p = jnp.int32(game_state.current_player)
    is_check = bool(is_in_check(jb, p))
    king_pos = None
    if is_check:
        r, c = find_king(jb, p)
        king_pos = [int(r), int(c)]
    
    # 检查是否可以前进/后退
    # 后退：只要不在初始状态就可以后退
    can_undo = len(game_state.history) > 1 and (
        game_state.current_history_index == -1 or game_state.current_history_index > 0
    )
    # 前进：只有在历史中（不在最新状态）才能前进
    can_forward = game_state.current_history_index != -1
    
    return {
        "board": game_state.board.tolist(),
        "current_player": game_state.current_player,
        "fen": board_to_fen(game_state.board, game_state.current_player),
        "last_move": game_state.last_move,
        "last_move_uci": game_state.last_move_uci,
        "game_over": game_state.game_over,
        "winner": game_state.winner,
        "step_count": game_state.step_count,
        "is_check": is_check,
        "king_pos": king_pos,
        "legal_moves": get_legal_moves_list(game_state.jax_state),
        "ai_value": game_state.ai_value,
        "uci_score": game_state.uci_score,
        "ai_loaded": model_mgr.params is not None,
        "uci_ready": uci_engine is not None and uci_engine.ready,
        "history_length": len(game_state.history),
        "can_undo": can_undo,
        "can_forward": can_forward,
        "at_latest": game_state.current_history_index == -1,
    }


@app.post("/api/new_game")
async def new_game(req: NewGameRequest):
    """开始新游戏"""
    global game_state, rng_key
    
    # 清空 UCI 引擎 hash
    if uci_engine and uci_engine.ready:
        uci_engine.new_game()
    
    board, player = parse_fen(req.fen)
    rng_key, sk = jax.random.split(rng_key)
    jax_state = env.init(sk)
    jax_board = jnp.array(board, dtype=jnp.int8)
    jax_state = jax_state.replace(
        board=jax_board,
        current_player=jnp.int32(player),
        legal_action_mask=get_legal_moves_mask(jax_board, jnp.int32(player))
    )
    
    game_state = GameState(
        board=board,
        current_player=player,
        jax_state=jax_state,
        current_history_index=-1
    )
    
    # 保存初始状态到历史（第0步）
    game_state.history.append({
        'jax_state': jax_state,
        'last_move': None,
        'last_move_uci': '',
        'ai_value': 0.0,
        'uci_score': None,
        'board': board.copy(),
        'current_player': player,
        'game_over': False,
        'winner': -1,
    })
    
    return await get_status()


@app.post("/api/move")
async def make_move(req: MoveRequest):
    """执行着法"""
    global game_state
    
    if game_state is None:
        raise HTTPException(status_code=400, detail="No game in progress")
    if game_state.game_over:
        raise HTTPException(status_code=400, detail="Game is over")
    if not game_state.jax_state.legal_action_mask[req.action]:
        raise HTTPException(status_code=400, detail="Illegal move")
    
    # 如果不在最新状态，清除后续历史（创建新分支）
    if game_state.current_history_index != -1:
        # 清除当前位置之后的所有历史
        # current_history_index指向当前显示的状态在history中的位置
        # 例如：history有5个元素[0,1,2,3,4]，current_history_index=2，则保留[0,1,2]，删除[3,4]
        game_state.history = game_state.history[:game_state.current_history_index + 1]
        game_state.move_history = game_state.move_history[:game_state.current_history_index]  # move_history比history少一个（初始状态没有move）
        game_state.current_history_index = -1  # 回到最新状态
    
    # 执行着法
    fs, ts = action_to_move(req.action)
    fr, fc, tr, tc = int(fs)//9, int(fs)%9, int(ts)//9, int(ts)%9
    uci_move = move_to_uci(int(fs), int(ts))
    
    # 记录着法（走之前的玩家）
    player_before = game_state.current_player
    
    new_jax_state = env.step(game_state.jax_state, req.action)
    game_state.jax_state = new_jax_state
    game_state.board = np.array(new_jax_state.board)
    game_state.current_player = int(new_jax_state.current_player)
    game_state.last_move = (fr, fc, tr, tc)
    game_state.last_move_uci = uci_move
    game_state.step_count += 1
    game_state.game_over = bool(new_jax_state.terminated)
    game_state.winner = int(new_jax_state.winner)
    
    # 保存着法历史
    game_state.move_history.append({
        'step': game_state.step_count,
        'uci': uci_move,
        'player': '红' if player_before == 0 else '黑'
    })
    
    # 将走完后的状态保存到历史
    game_state.history.append({
        'jax_state': new_jax_state,
        'last_move': (fr, fc, tr, tc),
        'last_move_uci': uci_move,
        'ai_value': game_state.ai_value,
        'uci_score': game_state.uci_score,
        'board': game_state.board.copy(),
        'current_player': game_state.current_player,
        'game_over': game_state.game_over,
        'winner': game_state.winner,
    })
    
    return await get_status()


@app.post("/api/undo")
async def undo_move():
    """后退一步（不删除历史）"""
    global game_state
    
    if game_state is None:
        raise HTTPException(status_code=400, detail="No game in progress")
    if len(game_state.history) <= 1:  # 只有初始状态，无法后退
        raise HTTPException(status_code=400, detail="Already at the beginning")
    
    # 确定当前位置
    if game_state.current_history_index == -1:
        # 在最新状态，后退到倒数第二个
        game_state.current_history_index = len(game_state.history) - 2
    else:
        # 已经在历史中，继续后退
        if game_state.current_history_index <= 0:
            raise HTTPException(status_code=400, detail="Already at the beginning")
        game_state.current_history_index -= 1
    
    # 恢复到该状态
    h = game_state.history[game_state.current_history_index]
    game_state.jax_state = h['jax_state']
    game_state.board = h['board'].copy()
    game_state.current_player = h['current_player']
    game_state.last_move = h['last_move']
    game_state.last_move_uci = h['last_move_uci']
    game_state.ai_value = h['ai_value']
    game_state.uci_score = h.get('uci_score')
    game_state.game_over = h['game_over']
    game_state.winner = h['winner']
    game_state.step_count = game_state.current_history_index  # history[i]对应第i步
    
    return await get_status()


@app.post("/api/forward")
async def forward_move():
    """前进一步"""
    global game_state
    
    if game_state is None:
        raise HTTPException(status_code=400, detail="No game in progress")
    if game_state.current_history_index == -1:
        raise HTTPException(status_code=400, detail="Already at the latest position")
    
    # 前进一步
    game_state.current_history_index += 1
    
    # 检查是否到达最新状态
    if game_state.current_history_index >= len(game_state.history) - 1:
        # 到达最新状态
        game_state.current_history_index = -1
        # 恢复到history的最后一个元素
        h = game_state.history[-1]
    else:
        h = game_state.history[game_state.current_history_index]
    
    # 恢复到该状态
    game_state.jax_state = h['jax_state']
    game_state.board = h['board'].copy()
    game_state.current_player = h['current_player']
    game_state.last_move = h['last_move']
    game_state.last_move_uci = h['last_move_uci']
    game_state.ai_value = h['ai_value']
    game_state.uci_score = h.get('uci_score')
    game_state.game_over = h['game_over']
    game_state.winner = h['winner']
    
    if game_state.current_history_index == -1:
        game_state.step_count = len(game_state.history) - 1
    else:
        game_state.step_count = game_state.current_history_index
    
    return await get_status()


class GotoStepRequest(BaseModel):
    step: int  # 回退到的目标步数


@app.post("/api/goto_step")
async def goto_step(req: GotoStepRequest):
    """跳转到指定步数"""
    global game_state
    
    if game_state is None:
        raise HTTPException(status_code=400, detail="No game in progress")
    
    target_step = req.step
    max_step = len(game_state.history) - 1  # 最大步数
    
    if target_step < 0 or target_step > max_step:
        raise HTTPException(status_code=400, detail=f"Invalid step number (must be 0-{max_step})")
    
    # 如果目标步数是最后一步，设置为-1（最新状态）
    if target_step == max_step:
        game_state.current_history_index = -1
        h = game_state.history[-1]
    else:
        game_state.current_history_index = target_step
        h = game_state.history[target_step]
    
    # 恢复到该状态
    game_state.jax_state = h['jax_state']
    game_state.board = h['board'].copy()
    game_state.current_player = h['current_player']
    game_state.last_move = h['last_move']
    game_state.last_move_uci = h['last_move_uci']
    game_state.ai_value = h['ai_value']
    game_state.uci_score = h.get('uci_score')
    game_state.game_over = h['game_over']
    game_state.winner = h['winner']
    game_state.step_count = target_step
    
    return await get_status()


@app.get("/api/history")
async def get_history():
    """获取走棋历史记录"""
    if game_state is None:
        return {"history": [], "current_step": 0}
    
    return {
        "history": game_state.move_history,
        "current_step": game_state.step_count
    }


@app.post("/api/ai_think")
async def ai_think(req: AIThinkRequest):
    """AI 思考并返回最佳着法"""
    global rng_key
    
    if game_state is None:
        raise HTTPException(status_code=400, detail="No game in progress")
    if model_mgr.params is None:
        raise HTTPException(status_code=400, detail="AI model not loaded")
    if game_state.game_over:
        raise HTTPException(status_code=400, detail="Game is over")
    
    # 检查是否在开局随机阶段（优先级高于温度参数）
    use_opening_random = (
        req.opening_random_moves > 0 and 
        game_state.step_count < req.opening_random_moves
    )
    
    if use_opening_random:
        print(f"[AI] 第{game_state.step_count+1}步，开局随机模式（使用先验策略，无MCTS模拟）")
        
        # 检查是否只有一个合法走法
        legal_mask = np.array(game_state.jax_state.legal_action_mask)
        legal_actions = np.where(legal_mask)[0]
        
        if len(legal_actions) == 1:
            # 只有一步，直接返回
            action = int(legal_actions[0])
            print(f"[AI] 只有一个合法走法，直接应用: action={action}")
            obs = env.observe(game_state.jax_state)[None, ...]
            _, value = model_mgr.net.apply({'params': model_mgr.params}, obs, train=False)
            search_value = float(value[0])
            if game_state.current_player == 1:
                search_value = -search_value
            game_state.ai_value = search_value
            
            fs, ts = action_to_move(action)
            return {
                "action": action,
                "uci": move_to_uci(int(fs), int(ts)),
                "value": search_value,
                "opening_random": True,
                "is_forced": True,
            }
        
        # 获取模型先验策略（直接前向推理，不运行MCTS）
        obs = env.observe(game_state.jax_state)[None, ...]
        logits, value = model_mgr.net.apply({'params': model_mgr.params}, obs, train=False)
        
        # 处理黑方视角
        if game_state.current_player == 1:
            logits = logits[:, _ROTATED_IDX]
        
        # 计算先验概率
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        logits = jnp.where(game_state.jax_state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)
        prior_probs = jax.nn.softmax(logits[0])
        prior_probs = np.array(prior_probs)
        
        # 获取Top-K候选
        top_indices = np.argsort(prior_probs)[::-1][:min(req.opening_top_k, len(legal_actions))]
        top_moves = []
        for idx in top_indices:
            if prior_probs[idx] > 1e-6:
                fs, ts = action_to_move(int(idx))
                top_moves.append({
                    "action": int(idx),
                    "uci": move_to_uci(int(fs), int(ts)),
                    "prior": float(prior_probs[idx]),
                })
        
        # 从Top-K中随机选择
        if len(top_moves) > 0:
            rng_key, sk = jax.random.split(rng_key)
            chosen_idx = int(jax.random.randint(sk, (), 0, len(top_moves)))
            chosen_move = top_moves[chosen_idx]
            action = chosen_move["action"]
            print(f"[AI] 从Top-{len(top_moves)}候选中随机选择了第{chosen_idx+1}个: {chosen_move['uci']} (先验={chosen_move['prior']:.3f})")
        else:
            # 兜底：随机选择任意合法走法
            rng_key, sk = jax.random.split(rng_key)
            action = int(legal_actions[jax.random.randint(sk, (), 0, len(legal_actions))])
            print(f"[AI] 警告：未找到高概率走法，随机选择")
        
        # 计算value（当前行棋方视角）
        search_value = float(value[0])
        if game_state.current_player == 1:
            search_value = -search_value
        game_state.ai_value = search_value
        
        fs, ts = action_to_move(action)
        return {
            "action": action,
            "uci": move_to_uci(int(fs), int(ts)),
            "value": search_value,
            "opening_random": True,
            "is_forced": False,
        }
    else:
        # 正常模式：始终选择最大概率走法
        action, value, policy_info = get_ai_action(
            game_state, req.num_simulations, req.top_k,
            return_policy=req.return_policy
        )
        if action is None:
            raise HTTPException(status_code=500, detail="AI failed to find a move")
        
        game_state.ai_value = value
        
        fs, ts = action_to_move(action)
        result = {
            "action": action,
            "uci": move_to_uci(int(fs), int(ts)),
            "value": value,
            "opening_random": False,
        }
        
        if req.return_policy and policy_info:
            result["policy"] = policy_info
        
        return result


@app.post("/api/policy_analysis")
async def policy_analysis(req: AIThinkRequest):
    """获取当前局面的策略分析（不执行着法）"""
    if game_state is None:
        raise HTTPException(status_code=400, detail="No game in progress")
    if model_mgr.params is None:
        raise HTTPException(status_code=400, detail="AI model not loaded")
    if game_state.game_over:
        raise HTTPException(status_code=400, detail="Game is over")
    
    action, value, policy_info = get_ai_action(
        game_state, req.num_simulations, req.top_k, 
        return_policy=True
    )
    
    if action is None or policy_info is None:
        raise HTTPException(status_code=500, detail="AI failed to analyze position")
    
    fs, ts = action_to_move(action)
    return {
        "best_action": action,
        "best_uci": move_to_uci(int(fs), int(ts)),
        "value": value,
        "policy": policy_info,
    }


@app.post("/api/uci_think")
async def uci_think(req: UCIThinkRequest):
    """UCI 引擎思考并返回最佳着法"""
    if game_state is None:
        raise HTTPException(status_code=400, detail="No game in progress")
    if uci_engine is None or not uci_engine.ready:
        raise HTTPException(status_code=400, detail="UCI engine not ready")
    if game_state.game_over:
        raise HTTPException(status_code=400, detail="Game is over")
    
    fen = board_to_fen(game_state.board, game_state.current_player)
    bestmove, score = uci_engine.get_best_move(fen, req.depth)
    
    if not bestmove or bestmove in ("(none)", "0000"):
        raise HTTPException(status_code=500, detail="UCI engine failed to find a move")
    
    try:
        f, t = uci_to_move(bestmove)
        action = int(move_to_action(f, t))
        if action < 0:
            raise ValueError("Invalid action")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to parse UCI move: {bestmove}")
    
    # 统一为红方视角（UCI 返回的是当前行棋方视角）
    if game_state.current_player == 1 and score is not None:
        score = -score  # 黑方走时取反，变为红方视角
    game_state.uci_score = score
    
    return {
        "action": action,
        "uci": bestmove,
        "score": score,
    }


@app.get("/api/checkpoints")
async def get_checkpoints(ckpt_dir: str = "checkpoints"):
    """获取检查点列表"""
    steps = list_checkpoints(ckpt_dir)
    return {
        "steps": steps,
        "current_step": model_mgr.step if model_mgr.params else None,
    }


@app.post("/api/load_model")
async def load_model(req: LoadModelRequest):
    """加载 AI 模型"""
    global mcts_recurrent_fn
    
    # 在线程池中运行，避免 orbax 与 uvloop 的冲突
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        success = await loop.run_in_executor(pool, model_mgr.load, req.ckpt_dir, req.step)
    
    if success:
        mcts_recurrent_fn = None  # 重置 MCTS 函数，让它在下次推理时重新创建
        return {
            "success": True,
            "step": model_mgr.step,
            "channels": model_mgr.channels,
            "num_blocks": model_mgr.num_blocks,
        }
    else:
        detail = model_mgr.last_error or "Failed to load model"
        raise HTTPException(status_code=400, detail=detail)


@app.get("/api/model_info")
async def get_model_info():
    """获取当前模型信息"""
    if model_mgr.params is None:
        return {"loaded": False}
    return {
        "loaded": True,
        "step": model_mgr.step,
        "channels": model_mgr.channels,
        "num_blocks": model_mgr.num_blocks,
    }


# ============================================================================
# 启动入口
# ============================================================================

def run_server(host: str = "0.0.0.0", port: int = 7860):
    """启动服务器"""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()

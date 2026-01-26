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

from xiangqi.env import XiangqiEnv, XiangqiState
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

    def _infer_channels(self, params) -> Optional[int]:
        try:
            conv0 = params.get("Conv_0") if hasattr(params, "get") else params["Conv_0"]
            return int(conv0["kernel"].shape[-1])
        except Exception:
            return None

    def _infer_num_blocks(self, params) -> int:
        try:
            return len([k for k in params.keys() if str(k).startswith("ResBlock_")])
        except Exception:
            return 0

    def load(self, ckpt_dir: str, step: int) -> bool:
        ckpt_dir = os.path.abspath(ckpt_dir)
        ckpt_manager = ocp.CheckpointManager(ckpt_dir)
        if step == 0:
            step = ckpt_manager.latest_step()
        if step is None:
            return False

        restored = None
        try:
            restored = ckpt_manager.restore(step)
        except Exception:
            try:
                ckpt_path = os.path.join(ckpt_dir, str(step))
                restored = ocp.StandardCheckpointer().restore(ckpt_path)
            except Exception as e:
                print(f"[AI] Checkpoint 恢复失败: {e}")
                return False

        params = None
        if isinstance(restored, dict) or hasattr(restored, "keys"):
            if "params" in restored:
                params = restored["params"]
            elif "default" in restored and isinstance(restored["default"], dict):
                params = restored["default"].get("params")

        if params is None:
            return False

        channels = self._infer_channels(params)
        num_blocks = self._infer_num_blocks(params)
        if not channels or num_blocks <= 0:
            return False

        self.net = AlphaZeroNetwork(
            action_space_size=ACTION_SPACE_SIZE,
            channels=channels,
            num_blocks=num_blocks,
        )
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
    history: List = field(default_factory=list)  # 状态历史（用于悔棋）
    move_history: List = field(default_factory=list)  # 着法历史（用于显示）
    ai_value: float = 0.0
    uci_score: Optional[int] = None


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


def get_ai_action(state: GameState) -> Tuple[Optional[int], float]:
    """获取 AI 最佳着法"""
    global rng_key, mcts_recurrent_fn
    
    if not model_mgr.params:
        return None, 0.0
    
    if mcts_recurrent_fn is None:
        mcts_recurrent_fn = create_mcts_recurrent_fn()
    
    obs = env.observe(state.jax_state)[None, ...]
    logits, value = model_mgr.net.apply({'params': model_mgr.params}, obs, train=False)
    
    if state.current_player == 1:
        logits = logits[:, _ROTATED_IDX]
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    logits = jnp.where(state.jax_state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)

    invalid_mask = ~state.jax_state.legal_action_mask

    rng_key, sk = jax.random.split(rng_key)
    root = mctx.RootFnOutput(
        prior_logits=logits, value=value,
        embedding=jax.tree.map(lambda x: jnp.expand_dims(x, 0), state.jax_state)
    )
    
    policy_output = mctx.gumbel_muzero_policy(
        params=model_mgr.params, rng_key=sk, root=root,
        recurrent_fn=mcts_recurrent_fn,
        num_simulations=256, max_num_considered_actions=32,
        invalid_actions=invalid_mask[None, ...]
    )
    
    search_value = float(policy_output.search_tree.node_values[0, 0])
    if state.current_player == 1:
        search_value = -search_value
    
    weights = np.array(policy_output.action_weights[0])
    action = int(jnp.argmax(weights))
    
    return action, search_value


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
    pass  # 无需额外参数，规则已完善


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
        jax_state=jax_state
    )
    
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
    
    # 保存历史
    game_state.history.append({
        'jax_state': game_state.jax_state,
        'last_move': game_state.last_move,
        'last_move_uci': game_state.last_move_uci,
        'ai_value': game_state.ai_value,
        'uci_score': game_state.uci_score,
    })
    
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
    
    return await get_status()


@app.post("/api/undo")
async def undo_move():
    """悔棋"""
    global game_state
    
    if game_state is None:
        raise HTTPException(status_code=400, detail="No game in progress")
    if not game_state.history:
        raise HTTPException(status_code=400, detail="No moves to undo")
    
    h = game_state.history.pop()
    if game_state.move_history:
        game_state.move_history.pop()  # 同步移除着法历史
    game_state.jax_state = h['jax_state']
    game_state.board = np.array(h['jax_state'].board)
    game_state.current_player = int(h['jax_state'].current_player)
    game_state.last_move = h['last_move']
    game_state.last_move_uci = h['last_move_uci']
    game_state.ai_value = h['ai_value']
    game_state.uci_score = h.get('uci_score')
    game_state.step_count -= 1
    game_state.game_over = False
    
    return await get_status()


class GotoStepRequest(BaseModel):
    step: int  # 回退到的目标步数


@app.post("/api/goto_step")
async def goto_step(req: GotoStepRequest):
    """回退到指定步数"""
    global game_state
    
    if game_state is None:
        raise HTTPException(status_code=400, detail="No game in progress")
    
    target_step = req.step
    if target_step < 0:
        raise HTTPException(status_code=400, detail="Invalid step number")
    
    # 计算需要回退多少步
    current_step = game_state.step_count
    steps_to_undo = current_step - target_step
    
    if steps_to_undo <= 0:
        raise HTTPException(status_code=400, detail="Cannot go forward, only backward")
    if steps_to_undo > len(game_state.history):
        raise HTTPException(status_code=400, detail="Not enough history")
    
    # 连续回退
    for _ in range(steps_to_undo):
        h = game_state.history.pop()
        if game_state.move_history:
            game_state.move_history.pop()  # 同步移除着法历史
        game_state.jax_state = h['jax_state']
        game_state.board = np.array(h['jax_state'].board)
        game_state.current_player = int(h['jax_state'].current_player)
        game_state.last_move = h['last_move']
        game_state.last_move_uci = h['last_move_uci']
        game_state.ai_value = h['ai_value']
        game_state.uci_score = h.get('uci_score')
        game_state.step_count -= 1
    
    game_state.game_over = False
    
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
    if game_state is None:
        raise HTTPException(status_code=400, detail="No game in progress")
    if model_mgr.params is None:
        raise HTTPException(status_code=400, detail="AI model not loaded")
    if game_state.game_over:
        raise HTTPException(status_code=400, detail="Game is over")
    
    action, value = get_ai_action(game_state)
    if action is None:
        raise HTTPException(status_code=500, detail="AI failed to find a move")
    
    game_state.ai_value = value
    
    fs, ts = action_to_move(action)
    return {
        "action": action,
        "uci": move_to_uci(int(fs), int(ts)),
        "value": value,
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
        raise HTTPException(status_code=400, detail="Failed to load model")


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

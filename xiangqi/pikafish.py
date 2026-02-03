"""
Pikafish UCI 引擎接口
用于与 Pikafish 对弈训练
"""

import subprocess
import threading
import queue
import time
from typing import Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import jax.numpy as jnp
import numpy as np

from xiangqi.actions import (
    BOARD_HEIGHT, BOARD_WIDTH, NUM_SQUARES,
    move_to_uci, uci_to_move, move_to_action, action_to_move,
    R_KING, R_ADVISOR, R_BISHOP, R_KNIGHT, R_ROOK, R_CANNON, R_PAWN,
    B_KING, B_ADVISOR, B_BISHOP, B_KNIGHT, B_ROOK, B_CANNON, B_PAWN,
    EMPTY,
)

# ============================================================================
# FEN 转换工具
# ============================================================================

# 棋子到 FEN 字符的映射
_PIECE_TO_FEN = {
    R_KING: 'K', R_ADVISOR: 'A', R_BISHOP: 'B', R_KNIGHT: 'N',
    R_ROOK: 'R', R_CANNON: 'C', R_PAWN: 'P',
    B_KING: 'k', B_ADVISOR: 'a', B_BISHOP: 'b', B_KNIGHT: 'n',
    B_ROOK: 'r', B_CANNON: 'c', B_PAWN: 'p',
}

# FEN 字符到棋子的映射
_FEN_TO_PIECE = {v: k for k, v in _PIECE_TO_FEN.items()}


def board_to_fen(board: np.ndarray, current_player: int) -> str:
    """
    将棋盘状态转换为 FEN 字符串
    
    Args:
        board: (10, 9) 棋盘数组
        current_player: 当前玩家 (0=红, 1=黑)
        
    Returns:
        FEN 字符串
    """
    fen_rows = []
    
    # FEN 从黑方视角（第9行到第0行）
    for row in range(BOARD_HEIGHT - 1, -1, -1):
        fen_row = ""
        empty_count = 0
        
        for col in range(BOARD_WIDTH):
            piece = int(board[row, col])
            if piece == EMPTY:
                empty_count += 1
            else:
                if empty_count > 0:
                    fen_row += str(empty_count)
                    empty_count = 0
                fen_row += _PIECE_TO_FEN.get(piece, '?')
        
        if empty_count > 0:
            fen_row += str(empty_count)
        fen_rows.append(fen_row)
    
    # 组合 FEN
    position = "/".join(fen_rows)
    side = 'w' if current_player == 0 else 'b'  # 红方用 'w'，黑方用 'b'
    
    # 返回完整 FEN（Pikafish 格式）
    return f"{position} {side} - - 0 1"


def fen_to_board(fen: str) -> Tuple[np.ndarray, int]:
    """
    将 FEN 字符串转换为棋盘状态
    
    Args:
        fen: FEN 字符串
        
    Returns:
        (board, current_player)
    """
    parts = fen.split()
    position = parts[0]
    side = parts[1] if len(parts) > 1 else 'w'
    
    board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=np.int8)
    
    rows = position.split("/")
    for row_idx, fen_row in enumerate(rows):
        board_row = BOARD_HEIGHT - 1 - row_idx  # FEN 从上到下，棋盘从下到上
        col = 0
        for char in fen_row:
            if char.isdigit():
                col += int(char)
            elif char in _FEN_TO_PIECE:
                board[board_row, col] = _FEN_TO_PIECE[char]
                col += 1
    
    current_player = 0 if side == 'w' else 1
    return board, current_player


# ============================================================================
# Pikafish 引擎接口
# ============================================================================

class PikafishEngine:
    """
    Pikafish UCI 引擎接口
    
    支持：
    - 设置搜索深度
    - 获取最佳走法和评分
    - 多实例并行
    """
    
    def __init__(self, engine_path: str = "pikafish", engine_id: int = 0):
        """
        Args:
            engine_path: Pikafish 可执行文件路径
            engine_id: 引擎实例 ID（用于调试）
        """
        self.engine_path = engine_path
        self.engine_id = engine_id
        self.process: Optional[subprocess.Popen] = None
        self.depth = 1
        self._lock = threading.Lock()
        
    def start(self) -> bool:
        """启动引擎
        
        Returns:
            是否成功启动
        """
        try:
            self.process = subprocess.Popen(
                [self.engine_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            self._send("uci")
            if not self._wait_for("uciok", timeout=5.0):
                print(f"[Pikafish {self.engine_id}] UCI 握手超时")
                return False
            self._send("isready")
            if not self._wait_for("readyok", timeout=5.0):
                print(f"[Pikafish {self.engine_id}] 引擎就绪超时")
                return False
            return True
        except FileNotFoundError:
            print(f"[Pikafish {self.engine_id}] 找不到引擎: {self.engine_path}")
            return False
        except Exception as e:
            print(f"[Pikafish {self.engine_id}] 启动失败: {e}")
            return False
        
    def set_depth(self, depth: int):
        """设置搜索深度"""
        self.depth = max(1, depth)
        
    def new_game(self):
        """开始新游戏（重置引擎状态）"""
        with self._lock:
            self._send("ucinewgame")
            self._send("isready")
            self._wait_for("readyok", timeout=2.0)
        
    def get_best_move(self, fen: str) -> Tuple[Optional[str], int]:
        """
        获取最佳走法和评分
        
        Args:
            fen: 当前局面的 FEN 字符串
            
        Returns:
            (uci_move, score_cp) - UCI 走法（如 'a0a1'）和厘兵评分
            如果无合法走法，返回 (None, 0)
        """
        with self._lock:
            if self.process is None:
                return None, 0
                
            self._send(f"position fen {fen}")
            self._send(f"go depth {self.depth}")
            
            best_move = None
            score = 0
            
            while True:
                line = self._read_line(timeout=30.0)
                if line is None:
                    print(f"[Pikafish {self.engine_id}] 读取超时")
                    break
                    
                if line.startswith("info"):
                    # 解析评分
                    if "score cp" in line:
                        try:
                            parts = line.split()
                            cp_idx = parts.index("cp")
                            score = int(parts[cp_idx + 1])
                        except (ValueError, IndexError):
                            pass
                    elif "score mate" in line:
                        try:
                            parts = line.split()
                            mate_idx = parts.index("mate")
                            mate_in = int(parts[mate_idx + 1])
                            # 将杀转换为大分数
                            score = 10000 - abs(mate_in) if mate_in > 0 else -10000 + abs(mate_in)
                        except (ValueError, IndexError):
                            pass
                            
                elif line.startswith("bestmove"):
                    parts = line.split()
                    if len(parts) >= 2 and parts[1] != "(none)":
                        best_move = parts[1]
                    break
                    
            return best_move, score
    
    def get_score(self, fen: str) -> int:
        """
        仅获取当前局面评分（不返回走法）
        
        Args:
            fen: 当前局面的 FEN 字符串
            
        Returns:
            厘兵评分（从当前走子方视角）
        """
        _, score = self.get_best_move(fen)
        return score
        
    def quit(self):
        """关闭引擎"""
        with self._lock:
            if self.process:
                try:
                    self._send("quit")
                    self.process.wait(timeout=2.0)
                except:
                    self.process.kill()
                finally:
                    self.process = None
                    
    def _send(self, cmd: str):
        """发送命令到引擎"""
        if self.process and self.process.stdin:
            self.process.stdin.write(cmd + "\n")
            self.process.stdin.flush()
            
    def _read_line(self, timeout: float = 5.0) -> Optional[str]:
        """读取一行输出"""
        if self.process is None or self.process.stdout is None:
            return None
            
        # 使用简单的阻塞读取（UCI 协议保证会有响应）
        try:
            # 设置超时使用 select 或线程
            import select
            ready, _, _ = select.select([self.process.stdout], [], [], timeout)
            if ready:
                line = self.process.stdout.readline()
                return line.strip() if line else None
            return None
        except:
            # 回退到阻塞读取
            line = self.process.stdout.readline()
            return line.strip() if line else None
            
    def _wait_for(self, expected: str, timeout: float = 5.0) -> bool:
        """等待特定响应"""
        start = time.time()
        while time.time() - start < timeout:
            line = self._read_line(timeout=timeout)
            if line == expected:
                return True
            if line is None:
                break
        return False


# ============================================================================
# 引擎池（多实例管理）
# ============================================================================

class PikafishPool:
    """
    Pikafish 引擎池
    管理多个引擎实例，支持并行对弈
    """
    
    def __init__(self, engine_path: str = "pikafish", num_engines: int = 8):
        """
        Args:
            engine_path: Pikafish 可执行文件路径
            num_engines: 引擎实例数量
        """
        self.engine_path = engine_path
        self.num_engines = num_engines
        self.engines: list[PikafishEngine] = []
        self.available: queue.Queue = queue.Queue()
        self._depth = 1
        
    def start(self) -> int:
        """
        启动所有引擎
        
        Returns:
            成功启动的引擎数量
        """
        success_count = 0
        for i in range(self.num_engines):
            engine = PikafishEngine(self.engine_path, engine_id=i)
            if engine.start():
                engine.set_depth(self._depth)
                self.engines.append(engine)
                self.available.put(engine)
                success_count += 1
                
        print(f"[PikafishPool] 启动了 {success_count}/{self.num_engines} 个引擎")
        return success_count
        
    def set_depth(self, depth: int):
        """设置所有引擎的搜索深度"""
        self._depth = max(1, depth)
        for engine in self.engines:
            engine.set_depth(self._depth)
            
    def get_depth(self) -> int:
        """获取当前深度"""
        return self._depth
            
    def acquire(self, timeout: float = 30.0) -> Optional[PikafishEngine]:
        """获取一个可用引擎"""
        try:
            return self.available.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def release(self, engine: PikafishEngine):
        """释放引擎回池"""
        self.available.put(engine)
        
    def shutdown(self):
        """关闭所有引擎"""
        for engine in self.engines:
            engine.quit()
        self.engines.clear()
        # 清空队列
        while not self.available.empty():
            try:
                self.available.get_nowait()
            except queue.Empty:
                break


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    # 测试 FEN 转换
    from xiangqi.rules import get_initial_board
    
    board = get_initial_board()
    fen = board_to_fen(np.array(board), 0)
    print(f"初始局面 FEN: {fen}")
    
    # 测试引擎
    engine = PikafishEngine()
    if engine.start():
        print("引擎启动成功")
        
        # 测试获取走法
        engine.set_depth(5)
        move, score = engine.get_best_move(fen)
        print(f"最佳走法: {move}, 评分: {score} cp")
        
        engine.quit()
    else:
        print("引擎启动失败，请确保 pikafish 在 PATH 中")

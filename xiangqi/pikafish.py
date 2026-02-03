"""
Pikafish UCI 引擎接口
用于与 Pikafish 对弈训练
"""

import subprocess
import threading
import queue
import time
from typing import Optional, Tuple

import numpy as np

from xiangqi.actions import (
    BOARD_HEIGHT, BOARD_WIDTH,
    R_KING, R_ADVISOR, R_BISHOP, R_KNIGHT, R_ROOK, R_CANNON, R_PAWN,
    B_KING, B_ADVISOR, B_BISHOP, B_KNIGHT, B_ROOK, B_CANNON, B_PAWN,
    EMPTY,
)

# ============================================================================
# FEN 转换工具
# ============================================================================

_PIECE_TO_FEN = {
    R_KING: 'K', R_ADVISOR: 'A', R_BISHOP: 'B', R_KNIGHT: 'N',
    R_ROOK: 'R', R_CANNON: 'C', R_PAWN: 'P',
    B_KING: 'k', B_ADVISOR: 'a', B_BISHOP: 'b', B_KNIGHT: 'n',
    B_ROOK: 'r', B_CANNON: 'c', B_PAWN: 'p',
}

_FEN_TO_PIECE = {v: k for k, v in _PIECE_TO_FEN.items()}


def board_to_fen(board: np.ndarray, current_player: int) -> str:
    """将棋盘状态转换为 FEN 字符串"""
    rows = []
    for row in range(BOARD_HEIGHT - 1, -1, -1):
        row_str = ""
        empty_count = 0
        for col in range(BOARD_WIDTH):
            piece = int(board[row, col])
            if piece == EMPTY:
                empty_count += 1
            else:
                if empty_count > 0:
                    row_str += str(empty_count)
                    empty_count = 0
                row_str += _PIECE_TO_FEN.get(piece, '?')
        if empty_count > 0:
            row_str += str(empty_count)
        rows.append(row_str)
    
    position = "/".join(rows)
    side = 'w' if current_player == 0 else 'b'
    return f"{position} {side} - - 0 1"


# ============================================================================
# Pikafish 引擎接口（参考 webui 实现）
# ============================================================================

class PikafishEngine:
    """Pikafish UCI 引擎接口"""
    
    def __init__(self, engine_path: str = "./pikafish", engine_id: int = 0):
        self.engine_path = engine_path
        self.engine_id = engine_id
        self.process: Optional[subprocess.Popen] = None
        self.output_queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self.depth = 1
        self.ready = False
        
    def start(self) -> bool:
        """启动引擎"""
        if self.process and self.process.poll() is None:
            return True
            
        try:
            self.process = subprocess.Popen(
                [self.engine_path],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            self._stop_event.clear()
            
            # 启动读取线程
            threading.Thread(target=self._read_stdout, daemon=True).start()
            
            # UCI 握手
            self._send("uci")
            start = time.time()
            while time.time() - start < 10:  # 10 秒超时
                try:
                    line = self.output_queue.get(timeout=0.1)
                    if "uciok" in line:
                        self.ready = True
                        print(f"[Pikafish {self.engine_id}] 启动成功")
                        return True
                except queue.Empty:
                    continue
                    
            print(f"[Pikafish {self.engine_id}] UCI 握手超时")
            return False
            
        except FileNotFoundError:
            print(f"[Pikafish {self.engine_id}] 找不到引擎: {self.engine_path}")
            return False
        except Exception as e:
            print(f"[Pikafish {self.engine_id}] 启动失败: {e}")
            return False
    
    def _read_stdout(self):
        """后台线程读取引擎输出"""
        while not self._stop_event.is_set() and self.process and self.process.poll() is None:
            try:
                line = self.process.stdout.readline()
                if line:
                    self.output_queue.put(line.strip())
            except:
                break
        
    def _send(self, cmd: str):
        """发送命令"""
        if self.process and self.process.stdin:
            self.process.stdin.write(f"{cmd}\n")
            self.process.stdin.flush()
            
    def set_depth(self, depth: int):
        """设置搜索深度"""
        self.depth = max(1, depth)
        
    def new_game(self):
        """新开局"""
        with self._lock:
            # 清空队列
            while not self.output_queue.empty():
                try:
                    self.output_queue.get_nowait()
                except queue.Empty:
                    break
                    
            self._send("ucinewgame")
            self._send("isready")
            
            start = time.time()
            while time.time() - start < 2:
                try:
                    line = self.output_queue.get(timeout=0.1)
                    if "readyok" in line:
                        return True
                except queue.Empty:
                    continue
            return False
        
    def get_best_move(self, fen: str) -> Tuple[Optional[str], int]:
        """获取最佳走法和评分"""
        with self._lock:
            if self.process is None or not self.ready:
                return None, 0
            
            # 清空队列
            while not self.output_queue.empty():
                try:
                    self.output_queue.get_nowait()
                except queue.Empty:
                    break
                
            self._send(f"position fen {fen}")
            self._send(f"go depth {self.depth}")
            
            best_move = None
            score = 0
            wait_seconds = max(5.0, self.depth * 1.0)  # 根据深度调整等待时间
            start_time = time.time()
            
            while time.time() - start_time < wait_seconds:
                try:
                    line = self.output_queue.get(timeout=0.1)
                    
                    # 解析评分
                    if "score cp" in line:
                        try:
                            parts = line.split("score cp")
                            if len(parts) > 1:
                                score_part = parts[1].strip().split()[0]
                                score = int(score_part)
                        except (ValueError, IndexError):
                            pass
                    elif "score mate" in line:
                        try:
                            parts = line.split("score mate")
                            if len(parts) > 1:
                                mate_in = int(parts[1].strip().split()[0])
                                score = 30000 - abs(mate_in) * 100 if mate_in > 0 else -30000 + abs(mate_in) * 100
                        except (ValueError, IndexError):
                            pass
                    
                    # 找到最佳走法
                    if line.startswith("bestmove"):
                        parts = line.split()
                        if len(parts) >= 2 and parts[1] not in ("(none)", "0000"):
                            best_move = parts[1]
                        break
                        
                except queue.Empty:
                    continue
                    
            return best_move, score
    
    def quit(self):
        """关闭引擎"""
        self._stop_event.set()
        if self.process:
            try:
                self._send("quit")
                self.process.wait(timeout=2.0)
            except:
                self.process.kill()
            finally:
                self.process = None
                self.ready = False


# ============================================================================
# 引擎池
# ============================================================================

class PikafishPool:
    """Pikafish 引擎池，管理多个实例"""
    
    def __init__(self, engine_path: str = "./pikafish", num_engines: int = 8):
        self.engine_path = engine_path
        self.num_engines = num_engines
        self.engines: list[PikafishEngine] = []
        self.available: queue.Queue = queue.Queue()
        self._depth = 1
        
    def start(self) -> int:
        """启动所有引擎，返回成功数量"""
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
        """设置所有引擎的深度"""
        self._depth = max(1, depth)
        for engine in self.engines:
            engine.set_depth(self._depth)
            
    def get_depth(self) -> int:
        return self._depth
            
    def acquire(self, timeout: float = 30.0) -> Optional[PikafishEngine]:
        """获取一个可用引擎"""
        try:
            return self.available.get(timeout=timeout)
        except queue.Empty:
            return None
            
    def release(self, engine: PikafishEngine):
        """归还引擎"""
        self.available.put(engine)
        
    def shutdown(self):
        """关闭所有引擎"""
        for engine in self.engines:
            engine.quit()
        self.engines.clear()
        while not self.available.empty():
            try:
                self.available.get_nowait()
            except queue.Empty:
                break


# ============================================================================
# 测试
# ============================================================================

if __name__ == "__main__":
    from xiangqi.rules import get_initial_board
    
    board = get_initial_board()
    fen = board_to_fen(np.array(board), 0)
    print(f"初始局面 FEN: {fen}")
    
    engine = PikafishEngine()
    if engine.start():
        print("引擎启动成功")
        engine.set_depth(5)
        move, score = engine.get_best_move(fen)
        print(f"最佳走法: {move}, 评分: {score} cp")
        engine.quit()
    else:
        print("引擎启动失败")

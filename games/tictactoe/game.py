"""
TicTacToeGame - 井字棋游戏实现

实现 core.game.Game 接口，用于测试训练流程。

特点:
- 简单的 3x3 棋盘
- 完全可观测（无需历史状态）
- 适合快速验证框架功能
"""

import copy
import logging
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from core.game import Game, ObservationSpace, ActionSpace, GameState, GameMeta
from games import register_game

logger = logging.getLogger(__name__)


# ============================================================
# 常量定义
# ============================================================

BOARD_SIZE = 3
NUM_SQUARES = 9
NUM_CHANNELS = 3  # 己方/对方/当前玩家

# 所有获胜线（预计算）
WIN_LINES = [
    # 横向
    [0, 1, 2], [3, 4, 5], [6, 7, 8],
    # 纵向
    [0, 3, 6], [1, 4, 7], [2, 5, 8],
    # 对角线
    [0, 4, 8], [2, 4, 6],
]


# ============================================================
# TicTacToeGame 类
# ============================================================

@register_game("tictactoe")
class TicTacToeGame(Game):
    """井字棋游戏实现
    
    3x3 棋盘，两人轮流下棋，先连成三子者获胜。
    
    棋盘表示:
    - 0: 空位
    - 1: 玩家0 (X, 先手)
    - 2: 玩家1 (O, 后手)
    
    Attributes:
        config: 游戏配置
        board: 棋盘数组 (9,)
        _current_player: 当前玩家 (0 或 1)
        move_count: 已走步数
    
    Example:
        >>> game = TicTacToeGame()
        >>> obs = game.reset()
        >>> print(obs.shape)  # (3, 3, 3)
        >>> actions = game.legal_actions()
        >>> obs, reward, done, info = game.step(actions[0])
    """
    
    # 关联配置类
    from .config import TicTacToeConfig
    config_class = TicTacToeConfig
    
    # === 元数据方法 ===
    
    @classmethod
    def get_meta(cls) -> GameMeta:
        """返回游戏元数据"""
        return GameMeta(
            name="井字棋",
            description="经典的3x3井字棋游戏，两人轮流下棋，先连成三子者获胜",
            version="1.0.0",
            author="ZeroForge",
            tags=["board", "strategy", "simple", "2-player"],
            difficulty="easy",
            min_players=2,
            max_players=2,
        )
    
    @property
    def supported_render_modes(self) -> List[str]:
        """支持的渲染模式"""
        return ["text", "human", "json", "ascii"]
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """返回默认配置"""
        return {
            "max_game_length": 9,
        }
    
    def __init__(self, config: Optional["TicTacToeConfig"] = None):
        """初始化游戏
        
        Args:
            config: 游戏配置（可选）
        """
        if config is not None:
            self.config = config
        else:
            from .config import TicTacToeConfig
            self.config = TicTacToeConfig()
        
        # 定义空间
        self._observation_space = ObservationSpace(
            shape=(NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE),
            dtype=np.float32
        )
        self._action_space = ActionSpace(n=NUM_SQUARES)
        
        # 游戏状态
        self.board: np.ndarray = np.zeros(NUM_SQUARES, dtype=np.int8)
        self._current_player: int = 0
        self.move_count: int = 0
        self._winner: Optional[int] = None
        self._is_terminal: Optional[bool] = None
    
    # === 空间属性 ===
    
    @property
    def observation_space(self) -> ObservationSpace:
        return self._observation_space
    
    @property
    def action_space(self) -> ActionSpace:
        return self._action_space
    
    @property
    def num_players(self) -> int:
        return 2
    
    # === 核心方法 ===
    
    def reset(self) -> np.ndarray:
        """重置游戏到初始状态"""
        self.board = np.zeros(NUM_SQUARES, dtype=np.int8)
        self._current_player = 0
        self.move_count = 0
        self._winner = None
        self._is_terminal = None
        return self.get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """执行动作
        
        Args:
            action: 动作索引 (0-8)
        
        Returns:
            (observation, reward, done, info)
        
        Raises:
            ValueError: 如果动作非法
        """
        # 验证动作
        if action < 0 or action >= NUM_SQUARES:
            raise ValueError(f"动作索引超出范围: {action}，应在 [0, 8]")
        if self.board[action] != 0:
            raise ValueError(f"位置 {action} 已被占用")
        if self.is_terminal():
            raise ValueError("游戏已结束，无法执行动作")
        
        # 记录当前玩家
        current_player = self._current_player
        
        # 执行动作
        self.board[action] = current_player + 1  # 1 或 2
        self.move_count += 1
        
        # 重置缓存
        self._is_terminal = None
        self._winner = None
        
        # 检查游戏结束
        done = self.is_terminal()
        reward = 0.0
        info: Dict[str, Any] = {}
        
        if done:
            winner = self.get_winner()
            info["winner"] = winner
            
            if winner is not None:
                # 从执行动作的玩家视角计算奖励
                reward = 1.0 if winner == current_player else -1.0
                info["termination"] = "win"
            else:
                reward = 0.0  # 和棋
                info["termination"] = "draw"
            
            info["all_rewards"] = self.get_rewards()
        
        # 切换玩家
        self._current_player = 1 - current_player
        
        return self.get_observation(), reward, done, info
    
    def legal_actions(self) -> List[int]:
        """获取合法动作列表"""
        if self.is_terminal():
            return []
        return [i for i in range(NUM_SQUARES) if self.board[i] == 0]
    
    def current_player(self) -> int:
        """获取当前玩家 (0=X先手, 1=O后手)"""
        return self._current_player
    
    def clone(self) -> "TicTacToeGame":
        """深拷贝游戏状态"""
        cloned = TicTacToeGame(config=self.config)
        cloned.board = self.board.copy()
        cloned._current_player = self._current_player
        cloned.move_count = self.move_count
        cloned._winner = self._winner
        cloned._is_terminal = self._is_terminal
        return cloned
    
    # === 观测编码 ===
    
    def get_observation(self) -> np.ndarray:
        """获取当前观测
        
        3通道编码：
        - 通道0: 当前玩家的棋子位置
        - 通道1: 对手的棋子位置
        - 通道2: 当前玩家指示 (全0或全1)
        
        Returns:
            形状为 (3, 3, 3) 的 float32 数组
        """
        board_2d = self.board.reshape(BOARD_SIZE, BOARD_SIZE)
        
        # 当前玩家的棋子标记
        my_mark = self._current_player + 1  # 1 或 2
        opp_mark = 2 - self._current_player  # 2 或 1
        
        channel_me = (board_2d == my_mark).astype(np.float32)
        channel_opp = (board_2d == opp_mark).astype(np.float32)
        channel_player = np.full((BOARD_SIZE, BOARD_SIZE), self._current_player, dtype=np.float32)
        
        return np.stack([channel_me, channel_opp, channel_player], axis=0)
    
    def get_legal_actions_mask(self) -> np.ndarray:
        """获取合法动作掩码"""
        mask = np.zeros(NUM_SQUARES, dtype=np.float32)
        for action in self.legal_actions():
            mask[action] = 1.0
        return mask
    
    # === 状态查询 ===
    
    def is_terminal(self) -> bool:
        """检查游戏是否结束"""
        if self._is_terminal is not None:
            return self._is_terminal
        
        # 检查是否有人获胜
        winner = self._check_winner()
        if winner is not None:
            self._winner = winner
            self._is_terminal = True
            return True
        
        # 检查是否平局（棋盘已满）
        if np.all(self.board != 0):
            self._winner = None  # None 表示平局
            self._is_terminal = True
            return True
        
        self._is_terminal = False
        return False
    
    def _check_winner(self) -> Optional[int]:
        """检查是否有获胜者"""
        for line in WIN_LINES:
            vals = self.board[line]
            if vals[0] != 0 and vals[0] == vals[1] == vals[2]:
                return int(vals[0] - 1)  # 返回玩家编号 (0 或 1)
        return None
    
    def get_winner(self) -> Optional[int]:
        """获取获胜玩家 (0=X, 1=O, None=和棋或未结束)"""
        if not self.is_terminal():
            return None
        return self._winner
    
    def get_rewards(self) -> Dict[int, float]:
        """获取所有玩家的最终奖励"""
        winner = self.get_winner()
        if winner is None:
            return {0: 0.0, 1: 0.0}  # 和棋
        elif winner == 0:
            return {0: 1.0, 1: -1.0}
        else:
            return {0: -1.0, 1: 1.0}
    
    # === 可视化 ===
    
    def render(self, mode: str = "text") -> Any:
        """渲染棋盘
        
        Args:
            mode: 渲染模式
                - "human": 打印到控制台
                - "text" / "ascii": 返回文本字符串
                - "json": 返回通用 grid 格式（用于 Web 前端）
        
        Returns:
            根据 mode 返回不同格式的渲染数据
            
        Raises:
            ValueError: 如果 mode 不在 supported_render_modes 中
        """
        if mode not in self.supported_render_modes:
            raise ValueError(f"不支持的渲染模式: {mode}，支持: {self.supported_render_modes}")
        
        symbols = {0: "", 1: "X", 2: "O"}
        board_2d = self.board.reshape(BOARD_SIZE, BOARD_SIZE)
        
        # 文本格式
        lines = []
        lines.append("  0 1 2")
        for i in range(BOARD_SIZE):
            row_symbols = {0: ".", 1: "X", 2: "O"}
            row = [row_symbols[board_2d[i, j]] for j in range(BOARD_SIZE)]
            lines.append(f"{i} " + "|".join(row))
            if i < BOARD_SIZE - 1:
                lines.append("  -----")
        board_str = "\n".join(lines)
        
        if mode == "human":
            print(board_str)
            print(f"当前玩家: {'X' if self._current_player == 0 else 'O'}")
            return None
        
        elif mode == "text" or mode == "ascii":
            return {"type": "text", "text": board_str}
        
        elif mode == "json":
            # 通用 grid 格式（用于 Web 前端 GameBoard 组件）
            # 将棋盘转换为 cells: 0=空, "X"=玩家0, "O"=玩家1
            cells = []
            for i in range(BOARD_SIZE):
                row = []
                for j in range(BOARD_SIZE):
                    val = board_2d[i, j]
                    if val == 0:
                        row.append(None)  # 空位
                    elif val == 1:
                        row.append("X")  # 玩家0
                    else:
                        row.append("O")  # 玩家1
                cells.append(row)
            
            return {
                "type": "grid",
                "rows": BOARD_SIZE,
                "cols": BOARD_SIZE,
                "cells": cells,
                "labels": {
                    "col": ["0", "1", "2"],
                    "row": ["0", "1", "2"],
                },
                # 额外游戏信息
                "current_player": self._current_player,
                "current_symbol": "X" if self._current_player == 0 else "O",
                "move_count": self.move_count,
                "is_terminal": self.is_terminal(),
                "winner": self.get_winner(),
            }
        
        else:
            return {"type": "text", "text": board_str}
    
    def get_state_hash(self) -> int:
        """获取状态哈希"""
        return hash((self.board.tobytes(), self._current_player))
    
    # === 调试信息 ===
    
    def get_debug_info(self) -> Dict[str, Any]:
        """获取调试信息（用于 Web 界面）"""
        symbols = {0: ".", 1: "X", 2: "O"}
        board_2d = self.board.reshape(BOARD_SIZE, BOARD_SIZE)
        
        return {
            "board_ascii": self.render(mode="ascii"),
            "board_array": self.board.tolist(),
            "current_player_symbol": "X" if self._current_player == 0 else "O",
            "move_count": self.move_count,
            "is_terminal": self.is_terminal(),
            "winner": self.get_winner(),
            "winner_symbol": symbols.get((self.get_winner() or -1) + 1, None) if self.is_terminal() else None,
        }
    
    def __repr__(self) -> str:
        player_symbol = "X" if self._current_player == 0 else "O"
        return (
            f"TicTacToeGame(player={player_symbol}, "
            f"move_count={self.move_count}, "
            f"terminal={self.is_terminal()})"
        )


# ============================================================
# 辅助函数
# ============================================================

def action_to_position(action: int) -> Tuple[int, int]:
    """动作索引转行列坐标"""
    return action // BOARD_SIZE, action % BOARD_SIZE


def position_to_action(row: int, col: int) -> int:
    """行列坐标转动作索引"""
    return row * BOARD_SIZE + col


def action_to_string(action: int) -> str:
    """动作索引转可读字符串"""
    row, col = action_to_position(action)
    return f"({row},{col})"


# ============================================================
# 导出
# ============================================================

__all__ = [
    "TicTacToeGame",
    "BOARD_SIZE",
    "NUM_SQUARES",
    "WIN_LINES",
    "action_to_position",
    "position_to_action",
    "action_to_string",
]

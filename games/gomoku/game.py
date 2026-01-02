"""
GomokuGame - 五子棋游戏实现

实现 core.game.Game 接口，支持 9x9 和 15x15 棋盘。

特点:
- 支持可配置的棋盘大小
- 完全可观测（无需历史状态）
- 高效的胜负判定算法
"""

import copy
import logging
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from core.game import Game, ObservationSpace, ActionSpace, GameState, GameMeta
from games import register_game
from .config import (
    GomokuConfig, 
    Gomoku9x9Config, 
    Gomoku15x15Config,
    NUM_CHANNELS, 
    WIN_LENGTH,
)

logger = logging.getLogger(__name__)


# ============================================================
# 五子棋基类
# ============================================================

class GomokuGame(Game):
    """五子棋游戏实现
    
    NxN 棋盘，两人轮流下棋，先连成五子者获胜。
    
    棋盘表示:
    - 0: 空位
    - 1: 玩家0 (黑棋, 先手)
    - 2: 玩家1 (白棋, 后手)
    
    Attributes:
        config: 游戏配置
        board: 棋盘数组 (board_size * board_size,)
        _current_player: 当前玩家 (0 或 1)
        move_count: 已走步数
        last_move: 最后一步的位置
    
    Example:
        >>> game = GomokuGame(config=Gomoku9x9Config())
        >>> obs = game.reset()
        >>> print(obs.shape)  # (3, 9, 9)
        >>> actions = game.legal_actions()
        >>> obs, reward, done, info = game.step(actions[0])
    """
    
    # 关联配置类
    config_class = GomokuConfig
    
    # 检测方向：水平、垂直、主对角线、副对角线
    DIRECTIONS = [
        (0, 1),   # 水平 →
        (1, 0),   # 垂直 ↓
        (1, 1),   # 主对角线 ↘
        (1, -1),  # 副对角线 ↙
    ]
    
    @classmethod
    def get_meta(cls) -> GameMeta:
        """返回游戏元数据（子类应覆盖）"""
        return GameMeta(
            name="五子棋",
            description="经典的五子棋游戏，两人轮流下棋，先连成五子者获胜",
            version="1.0.0",
            author="ZeroForge",
            tags=["board", "strategy", "2-player"],
            difficulty="medium",
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
            "board_size": 15,
            "win_length": 5,
        }
    
    def __init__(self, config: Optional[GomokuConfig] = None):
        """初始化游戏
        
        Args:
            config: 游戏配置（可选）
        """
        if config is not None:
            self.config = config
        else:
            self.config = GomokuConfig()
        
        # 从配置获取参数
        self.board_size = self.config.board_size
        self.num_squares = self.config.num_squares
        self.win_length = self.config.win_length
        
        # 定义空间
        self._observation_space = ObservationSpace(
            shape=(NUM_CHANNELS, self.board_size, self.board_size),
            dtype=np.float32
        )
        self._action_space = ActionSpace(n=self.num_squares)
        
        # 游戏状态
        self.board: np.ndarray = np.zeros(self.num_squares, dtype=np.int8)
        self._current_player: int = 0
        self.move_count: int = 0
        self._winner: Optional[int] = None
        self._is_terminal: Optional[bool] = None
        self.last_move: Optional[int] = None
    
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
        self.board = np.zeros(self.num_squares, dtype=np.int8)
        self._current_player = 0
        self.move_count = 0
        self._winner = None
        self._is_terminal = None
        self.last_move = None
        return self.get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """执行动作
        
        Args:
            action: 动作索引 (0 到 board_size^2-1)
        
        Returns:
            (observation, reward, done, info)
        
        Raises:
            ValueError: 如果动作非法
        """
        # 验证动作
        if action < 0 or action >= self.num_squares:
            raise ValueError(f"动作索引超出范围: {action}，应在 [0, {self.num_squares - 1}]")
        if self.board[action] != 0:
            row, col = self._action_to_position(action)
            raise ValueError(f"位置 ({row}, {col}) 已被占用")
        if self.is_terminal():
            raise ValueError("游戏已结束，无法执行动作")
        
        # 记录当前玩家
        current_player = self._current_player
        
        # 执行动作
        self.board[action] = current_player + 1  # 1 或 2
        self.move_count += 1
        self.last_move = action
        
        # 重置缓存
        self._is_terminal = None
        self._winner = None
        
        # 检查游戏结束
        done = self.is_terminal()
        reward = 0.0
        info: Dict[str, Any] = {
            "last_move": action,
            "last_move_pos": self._action_to_position(action),
        }
        
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
        return [i for i in range(self.num_squares) if self.board[i] == 0]
    
    def current_player(self) -> int:
        """获取当前玩家 (0=黑棋先手, 1=白棋后手)"""
        return self._current_player
    
    def clone(self) -> "GomokuGame":
        """深拷贝游戏状态"""
        cloned = self.__class__(config=self.config)
        cloned.board = self.board.copy()
        cloned._current_player = self._current_player
        cloned.move_count = self.move_count
        cloned._winner = self._winner
        cloned._is_terminal = self._is_terminal
        cloned.last_move = self.last_move
        return cloned
    
    # === 观测编码 ===
    
    def get_observation(self) -> np.ndarray:
        """获取当前观测
        
        3通道编码：
        - 通道0: 当前玩家的棋子位置
        - 通道1: 对手的棋子位置
        - 通道2: 当前玩家指示 (全0或全1)
        
        Returns:
            形状为 (3, board_size, board_size) 的 float32 数组
        """
        board_2d = self.board.reshape(self.board_size, self.board_size)
        
        # 当前玩家的棋子标记
        my_mark = self._current_player + 1  # 1 或 2
        opp_mark = 2 - self._current_player  # 2 或 1
        
        channel_me = (board_2d == my_mark).astype(np.float32)
        channel_opp = (board_2d == opp_mark).astype(np.float32)
        channel_player = np.full((self.board_size, self.board_size), self._current_player, dtype=np.float32)
        
        return np.stack([channel_me, channel_opp, channel_player], axis=0)
    
    def get_legal_actions_mask(self) -> np.ndarray:
        """获取合法动作掩码"""
        mask = np.zeros(self.num_squares, dtype=np.float32)
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
        """检查是否有获胜者
        
        只需检查最后落子位置附近是否形成五连
        """
        if self.last_move is None:
            return None
        
        row, col = self._action_to_position(self.last_move)
        player_mark = self.board[self.last_move]
        
        if player_mark == 0:
            return None
        
        board_2d = self.board.reshape(self.board_size, self.board_size)
        
        for dr, dc in self.DIRECTIONS:
            count = 1  # 当前位置
            
            # 正方向计数
            r, c = row + dr, col + dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size:
                if board_2d[r, c] == player_mark:
                    count += 1
                    r += dr
                    c += dc
                else:
                    break
            
            # 反方向计数
            r, c = row - dr, col - dc
            while 0 <= r < self.board_size and 0 <= c < self.board_size:
                if board_2d[r, c] == player_mark:
                    count += 1
                    r -= dr
                    c -= dc
                else:
                    break
            
            if count >= self.win_length:
                return int(player_mark - 1)  # 返回玩家编号 (0 或 1)
        
        return None
    
    def get_winner(self) -> Optional[int]:
        """获取获胜玩家 (0=黑棋, 1=白棋, None=和棋或未结束)"""
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
    
    # === 辅助方法 ===
    
    def _action_to_position(self, action: int) -> Tuple[int, int]:
        """动作索引转行列坐标"""
        return action // self.board_size, action % self.board_size
    
    def _position_to_action(self, row: int, col: int) -> int:
        """行列坐标转动作索引"""
        return row * self.board_size + col
    
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
        
        board_2d = self.board.reshape(self.board_size, self.board_size)
        symbols = {0: ".", 1: "●", 2: "○"}
        
        # 生成列标签
        col_labels = [f"{i:2d}" for i in range(self.board_size)]
        
        # 文本格式
        lines = []
        lines.append("   " + " ".join(col_labels))
        for i in range(self.board_size):
            row = [symbols[board_2d[i, j]] for j in range(self.board_size)]
            # 高亮最后一步
            if self.last_move is not None:
                last_row, last_col = self._action_to_position(self.last_move)
                if i == last_row:
                    mark = board_2d[i, last_col]
                    row[last_col] = f"[{symbols[mark]}]" if mark != 0 else symbols[0]
            lines.append(f"{i:2d} " + "  ".join(row))
        board_str = "\n".join(lines)
        
        if mode == "human":
            print(board_str)
            print(f"当前玩家: {'●黑棋' if self._current_player == 0 else '○白棋'}")
            if self.last_move is not None:
                row, col = self._action_to_position(self.last_move)
                print(f"最后一步: ({row}, {col})")
            return None
        
        elif mode == "text" or mode == "ascii":
            return {"type": "text", "text": board_str}
        
        elif mode == "json":
            # 通用 grid 格式（用于 Web 前端 GameBoard 组件）
            cells = []
            for i in range(self.board_size):
                row = []
                for j in range(self.board_size):
                    val = board_2d[i, j]
                    if val == 0:
                        row.append(None)  # 空位
                    elif val == 1:
                        row.append("●")  # 黑棋
                    else:
                        row.append("○")  # 白棋
                cells.append(row)
            
            # 高亮最后一步 - 格式为 [[row, col], ...] 符合前端 GridRenderData 接口
            highlights: List[Tuple[int, int]] = []
            if self.last_move is not None:
                row, col = self._action_to_position(self.last_move)
                highlights.append((row, col))
            
            return {
                "type": "grid",
                "rows": self.board_size,
                "cols": self.board_size,
                "cells": cells,
                "labels": {
                    "col": [str(i) for i in range(self.board_size)],
                    "row": [str(i) for i in range(self.board_size)],
                },
                "highlights": highlights,
                # 额外游戏信息
                "current_player": self._current_player,
                "current_symbol": "●" if self._current_player == 0 else "○",
                "move_count": self.move_count,
                "is_terminal": self.is_terminal(),
                "winner": self.get_winner(),
                "last_move": self.last_move,
                "last_move_pos": self._action_to_position(self.last_move) if self.last_move else None,
            }
        
        else:
            return {"type": "text", "text": board_str}
    
    def get_state_hash(self) -> int:
        """获取状态哈希"""
        return hash((self.board.tobytes(), self._current_player))
    
    # === 调试信息 ===
    
    def get_debug_info(self) -> Dict[str, Any]:
        """获取调试信息（用于 Web 界面）"""
        symbols = {0: ".", 1: "●", 2: "○"}
        
        return {
            "board_ascii": self.render(mode="ascii"),
            "board_array": self.board.tolist(),
            "board_size": self.board_size,
            "current_player_symbol": "●" if self._current_player == 0 else "○",
            "move_count": self.move_count,
            "is_terminal": self.is_terminal(),
            "winner": self.get_winner(),
            "winner_symbol": symbols.get((self.get_winner() or -1) + 1, None) if self.is_terminal() else None,
            "last_move": self.last_move,
            "last_move_pos": self._action_to_position(self.last_move) if self.last_move else None,
        }
    
    def __repr__(self) -> str:
        player_symbol = "●" if self._current_player == 0 else "○"
        return (
            f"{self.__class__.__name__}(board_size={self.board_size}, "
            f"player={player_symbol}, "
            f"move_count={self.move_count}, "
            f"terminal={self.is_terminal()})"
        )


# ============================================================
# 9x9 五子棋（注册为独立游戏）
# ============================================================

@register_game("gomoku_9x9")
class Gomoku9x9Game(GomokuGame):
    """9x9 五子棋游戏
    
    较小的棋盘，适合快速训练和测试。
    """
    
    config_class = Gomoku9x9Config
    
    @classmethod
    def get_meta(cls) -> GameMeta:
        return GameMeta(
            name="五子棋 9×9",
            description="9×9 棋盘的五子棋，两人轮流下棋，先连成五子者获胜。较小棋盘适合快速训练。",
            version="1.0.0",
            author="ZeroForge",
            tags=["board", "strategy", "2-player", "gomoku"],
            difficulty="easy",
            min_players=2,
            max_players=2,
        )
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {
            "board_size": 9,
            "win_length": 5,
        }
    
    def __init__(self, config: Optional[Gomoku9x9Config] = None):
        if config is None:
            config = Gomoku9x9Config()
        super().__init__(config=config)


# ============================================================
# 15x15 五子棋（注册为独立游戏）
# ============================================================

@register_game("gomoku_15x15")
class Gomoku15x15Game(GomokuGame):
    """15x15 五子棋游戏
    
    标准棋盘大小，经典五子棋规则。
    """
    
    config_class = Gomoku15x15Config
    
    @classmethod
    def get_meta(cls) -> GameMeta:
        return GameMeta(
            name="五子棋 15×15",
            description="15×15 标准棋盘的五子棋，两人轮流下棋，先连成五子者获胜。",
            version="1.0.0",
            author="ZeroForge",
            tags=["board", "strategy", "2-player", "gomoku", "standard"],
            difficulty="medium",
            min_players=2,
            max_players=2,
        )
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        return {
            "board_size": 15,
            "win_length": 5,
        }
    
    def __init__(self, config: Optional[Gomoku15x15Config] = None):
        if config is None:
            config = Gomoku15x15Config()
        super().__init__(config=config)


# ============================================================
# 辅助函数
# ============================================================

def action_to_position(action: int, board_size: int) -> Tuple[int, int]:
    """动作索引转行列坐标"""
    return action // board_size, action % board_size


def position_to_action(row: int, col: int, board_size: int) -> int:
    """行列坐标转动作索引"""
    return row * board_size + col


def action_to_string(action: int, board_size: int) -> str:
    """动作索引转可读字符串"""
    row, col = action_to_position(action, board_size)
    return f"({row},{col})"


# ============================================================
# 导出
# ============================================================

__all__ = [
    "GomokuGame",
    "Gomoku9x9Game",
    "Gomoku15x15Game",
    "action_to_position",
    "position_to_action",
    "action_to_string",
]

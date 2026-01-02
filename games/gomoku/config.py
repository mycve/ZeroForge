"""
Gomoku 五子棋配置

支持 9x9 和 15x15 两种棋盘大小。
继承自 core.config.GameConfig 实现统一的配置管理。
"""

from dataclasses import dataclass
from typing import Optional

from core.config import GameConfig, DebugConfig as BaseDebugConfig


# ============================================================
# 棋盘常量
# ============================================================

NUM_PLAYERS = 2          # 玩家数量
NUM_CHANNELS = 3         # 观测通道数 (己方/对方/当前玩家)
WIN_LENGTH = 5           # 连续 5 子获胜


# ============================================================
# 游戏配置类（继承自 GameConfig）
# ============================================================

@dataclass
class GomokuConfig(GameConfig):
    """五子棋游戏配置
    
    继承自 GameConfig，支持不同棋盘大小。
    
    Attributes:
        board_size: 棋盘大小 (9 或 15)
        win_length: 连续多少子获胜（默认 5）
    
    Example:
        >>> config = GomokuConfig(board_size=9)
        >>> game = GomokuGame(config=config)
    """
    
    # 五子棋特有参数
    board_size: int = 15           # 棋盘尺寸（默认 15x15）
    win_length: int = WIN_LENGTH   # 连续几子获胜
    
    # 覆盖默认值
    history_steps: int = 1         # 五子棋状态完全可观测，不需要历史
    enable_augmentation: bool = False
    
    def __post_init__(self):
        """初始化后设置最大步数"""
        self.max_game_length = self.board_size * self.board_size
    
    @property
    def num_squares(self) -> int:
        """棋盘格子总数"""
        return self.board_size * self.board_size
    
    @property
    def observation_shape(self) -> tuple[int, int, int]:
        """观测张量形状 (C, H, W)"""
        return (NUM_CHANNELS, self.board_size, self.board_size)
    
    @property
    def action_size(self) -> int:
        """动作空间大小"""
        return self.num_squares
    
    def validate(self) -> None:
        """验证配置合法性"""
        super().validate()
        if self.board_size < 5:
            raise ValueError(f"棋盘大小至少为 5，当前: {self.board_size}")
        if self.win_length < 3 or self.win_length > self.board_size:
            raise ValueError(f"win_length 必须在 [3, {self.board_size}] 范围内")


@dataclass
class Gomoku9x9Config(GomokuConfig):
    """9x9 五子棋配置"""
    board_size: int = 9


@dataclass
class Gomoku15x15Config(GomokuConfig):
    """15x15 五子棋配置"""
    board_size: int = 15


# ============================================================
# 调试配置
# ============================================================

@dataclass  
class GomokuDebugConfig(BaseDebugConfig):
    """五子棋调试配置"""
    
    # 五子棋特有调试选项
    log_board_ascii: bool = True  # 是否记录 ASCII 棋盘
    highlight_last_move: bool = True  # 高亮显示最后一步


# ============================================================
# 默认配置实例
# ============================================================

DEFAULT_9x9_CONFIG = Gomoku9x9Config()
DEFAULT_15x15_CONFIG = Gomoku15x15Config()
DEFAULT_DEBUG_CONFIG = GomokuDebugConfig()


# ============================================================
# 导出
# ============================================================

__all__ = [
    "GomokuConfig",
    "Gomoku9x9Config",
    "Gomoku15x15Config",
    "GomokuDebugConfig",
    "NUM_PLAYERS",
    "NUM_CHANNELS",
    "WIN_LENGTH",
    "DEFAULT_9x9_CONFIG",
    "DEFAULT_15x15_CONFIG",
    "DEFAULT_DEBUG_CONFIG",
]

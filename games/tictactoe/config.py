"""
TicTacToe 游戏配置

定义井字棋相关的常量和配置参数。
继承自 core.config.GameConfig 实现统一的配置管理。
"""

from dataclasses import dataclass
from typing import Optional

from core.config import GameConfig, DebugConfig as BaseDebugConfig


# ============================================================
# 棋盘常量
# ============================================================

BOARD_SIZE = 3           # 棋盘尺寸 (3x3)
NUM_SQUARES = 9          # 总格子数
NUM_PLAYERS = 2          # 玩家数量
NUM_CHANNELS = 3         # 观测通道数 (己方/对方/当前玩家)


# ============================================================
# 游戏规则常量
# ============================================================

MAX_GAME_LENGTH = 9      # 最大步数（棋盘满了）
DEFAULT_HISTORY_STEPS = 1  # 井字棋不需要历史（状态完全可观测）


# ============================================================
# 观测空间
# ============================================================

def compute_input_channels(history_steps: int = DEFAULT_HISTORY_STEPS) -> int:
    """
    计算观测张量的通道数
    
    通道布局:
    - 己方棋子位置
    - 对方棋子位置
    - 当前玩家指示
    
    Args:
        history_steps: 历史步数（井字棋默认为1，不需要历史）
        
    Returns:
        通道数
    """
    return NUM_CHANNELS  # 固定3通道


# ============================================================
# 游戏配置类（继承自 GameConfig）
# ============================================================

@dataclass
class TicTacToeConfig(GameConfig):
    """井字棋游戏配置
    
    继承自 GameConfig，井字棋较简单，大部分使用默认值即可。
    
    Example:
        >>> config = TicTacToeConfig()
        >>> game = TicTacToeGame(config=config)
    """
    
    # 覆盖默认值
    max_game_length: int = MAX_GAME_LENGTH
    history_steps: int = DEFAULT_HISTORY_STEPS
    enable_augmentation: bool = False  # 井字棋对称性简单，暂不启用
    
    @property
    def observation_shape(self) -> tuple[int, int, int]:
        """观测张量形状 (C, H, W)"""
        return (NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE)
    
    @property
    def action_size(self) -> int:
        """动作空间大小"""
        return NUM_SQUARES
    
    def validate(self) -> None:
        """验证配置合法性"""
        super().validate()
        # 井字棋没有特殊验证需求


# ============================================================
# 调试配置
# ============================================================

@dataclass  
class TicTacToeDebugConfig(BaseDebugConfig):
    """井字棋调试配置"""
    
    # 井字棋特有调试选项
    log_board_ascii: bool = True  # 是否记录 ASCII 棋盘


# ============================================================
# 默认配置实例
# ============================================================

DEFAULT_CONFIG = TicTacToeConfig()
DEFAULT_DEBUG_CONFIG = TicTacToeDebugConfig()


# ============================================================
# 导出
# ============================================================

__all__ = [
    "TicTacToeConfig",
    "TicTacToeDebugConfig",
    "BOARD_SIZE",
    "NUM_SQUARES",
    "NUM_PLAYERS",
    "NUM_CHANNELS",
    "MAX_GAME_LENGTH",
    "DEFAULT_HISTORY_STEPS",
    "DEFAULT_CONFIG",
    "DEFAULT_DEBUG_CONFIG",
    "compute_input_channels",
]

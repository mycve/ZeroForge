"""
Chinese Chess 游戏配置

定义游戏相关的常量和配置参数。
继承自 core.config.GameConfig 实现统一的配置管理。
"""

from dataclasses import dataclass, field
from typing import Optional

from core.config import GameConfig, DebugConfig as BaseDebugConfig


# ============================================================
# 棋盘常量
# ============================================================

BOARD_HEIGHT = 10    # 棋盘高度
BOARD_WIDTH = 9      # 棋盘宽度
NUM_PIECE_TYPES = 14 # 棋子类型数量 (红方 7 + 黑方 7)
NUM_SQUARES = 90     # 棋盘格子数


# ============================================================
# 游戏规则常量
# ============================================================

MAX_GAME_LENGTH = 200        # 最大回合数
DEFAULT_HISTORY_STEPS = 4    # 默认历史步数（用于观测编码）


# ============================================================
# 动作空间
# ============================================================

# 动作空间大小约 2086（精简版，只包含合法走法模式）
# 包括: 直线走法(车/炮/将/兵) + 马的日字 + 象的田字 + 士的斜线


# ============================================================
# 观测空间
# ============================================================

def compute_input_channels(history_steps: int = DEFAULT_HISTORY_STEPS) -> int:
    """
    计算观测张量的通道数
    
    通道布局:
    - 14 种棋子 × 历史步数
    - 当前玩家通道
    - 回合数通道
    
    Args:
        history_steps: 历史步数
        
    Returns:
        通道数
    """
    return NUM_PIECE_TYPES * history_steps + 2


# ============================================================
# 游戏配置类（继承自 GameConfig）
# ============================================================

@dataclass
class ChineseChessConfig(GameConfig):
    """中国象棋游戏配置
    
    继承自 GameConfig，添加中国象棋特有的配置参数。
    
    Example:
        >>> config = ChineseChessConfig(max_game_length=300)
        >>> game = ChineseChessGame(config=config)
    """
    
    # 覆盖默认值
    max_game_length: int = MAX_GAME_LENGTH
    history_steps: int = DEFAULT_HISTORY_STEPS
    
    # 中国象棋特有配置
    initial_fen: Optional[str] = None  # 初始 FEN（None 表示标准开局）
    
    @property
    def observation_shape(self) -> tuple[int, int, int]:
        """观测张量形状 (C, H, W)"""
        channels = compute_input_channels(self.history_steps)
        return (channels, BOARD_HEIGHT, BOARD_WIDTH)
    
    def validate(self) -> None:
        """验证配置合法性"""
        super().validate()  # 调用父类验证
        
        # 中国象棋特定验证
        if self.initial_fen is not None:
            # TODO: 验证 FEN 格式
            pass


# ============================================================
# 调试配置（继承自 BaseDebugConfig）
# ============================================================

@dataclass  
class ChineseChessDebugConfig(BaseDebugConfig):
    """中国象棋调试配置
    
    继承自 core.config.DebugConfig，添加中国象棋特有的调试选项。
    """
    
    # 中国象棋特有调试选项
    log_fen: bool = True  # 是否记录 FEN
    log_moves_uci: bool = True  # 是否记录 UCI 格式走法


# ============================================================
# 默认配置实例
# ============================================================

DEFAULT_CONFIG = ChineseChessConfig()
DEFAULT_DEBUG_CONFIG = ChineseChessDebugConfig()


# ============================================================
# 兼容旧代码的别名
# ============================================================

DebugConfig = ChineseChessDebugConfig

"""
Gomoku 五子棋游戏模块

支持 9x9 和 15x15 两种棋盘大小。

使用方法:
    >>> from games.gomoku import Gomoku9x9Game, Gomoku15x15Game
    >>> 
    >>> # 9x9 棋盘
    >>> game = Gomoku9x9Game()
    >>> obs = game.reset()
    >>> 
    >>> # 15x15 棋盘
    >>> game = Gomoku15x15Game()
    >>> obs = game.reset()
    >>>
    >>> # 或者通过 make_game
    >>> from games import make_game
    >>> game = make_game("gomoku_9x9")
    >>> game = make_game("gomoku_15x15")
"""

from .game import (
    GomokuGame,
    Gomoku9x9Game,
    Gomoku15x15Game,
    action_to_position,
    position_to_action,
    action_to_string,
)
from .config import (
    GomokuConfig,
    Gomoku9x9Config,
    Gomoku15x15Config,
    GomokuDebugConfig,
    NUM_CHANNELS,
    WIN_LENGTH,
    DEFAULT_9x9_CONFIG,
    DEFAULT_15x15_CONFIG,
    DEFAULT_DEBUG_CONFIG,
)

__all__ = [
    # 游戏类
    "GomokuGame",
    "Gomoku9x9Game",
    "Gomoku15x15Game",
    # 配置类
    "GomokuConfig",
    "Gomoku9x9Config",
    "Gomoku15x15Config",
    "GomokuDebugConfig",
    # 常量
    "NUM_CHANNELS",
    "WIN_LENGTH",
    # 默认配置
    "DEFAULT_9x9_CONFIG",
    "DEFAULT_15x15_CONFIG",
    "DEFAULT_DEBUG_CONFIG",
    # 辅助函数
    "action_to_position",
    "position_to_action",
    "action_to_string",
]

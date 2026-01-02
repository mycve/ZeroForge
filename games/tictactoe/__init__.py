"""
TicTacToe - 井字棋游戏模块

用于测试和验证训练流程的简单游戏。

游戏目录结构:
├── __init__.py      # 模块入口 + 自动注册
├── game.py          # Game 接口实现
└── config.py        # 游戏配置（继承自 GameConfig）

使用方法:
    >>> from games.tictactoe import TicTacToeGame, TicTacToeConfig
    >>> 
    >>> # 使用默认配置
    >>> game = TicTacToeGame()
    >>> obs = game.reset()
    >>> actions = game.legal_actions()
    >>> obs, reward, done, info = game.step(actions[0])
    >>> 
    >>> # 渲染棋盘
    >>> game.render()
    >>>
    >>> # 或通过 make_game
    >>> from games import make_game
    >>> game = make_game("tictactoe")
"""

from .game import (
    TicTacToeGame,
    BOARD_SIZE,
    NUM_SQUARES,
    WIN_LINES,
    action_to_position,
    position_to_action,
    action_to_string,
)

from .config import (
    TicTacToeConfig,
    TicTacToeDebugConfig,
    # 常量
    NUM_CHANNELS,
    MAX_GAME_LENGTH,
    DEFAULT_HISTORY_STEPS,
    # 默认配置
    DEFAULT_CONFIG,
    DEFAULT_DEBUG_CONFIG,
)

__all__ = [
    # 游戏类
    "TicTacToeGame",
    # 配置类
    "TicTacToeConfig",
    "TicTacToeDebugConfig",
    # 辅助函数
    "action_to_position",
    "position_to_action",
    "action_to_string",
    # 常量
    "BOARD_SIZE",
    "NUM_SQUARES",
    "WIN_LINES",
    "NUM_CHANNELS",
    "MAX_GAME_LENGTH",
    "DEFAULT_HISTORY_STEPS",
    # 默认配置
    "DEFAULT_CONFIG",
    "DEFAULT_DEBUG_CONFIG",
]

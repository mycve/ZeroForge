"""
Chinese Chess - 中国象棋游戏模块

游戏目录结构:
├── __init__.py      # 模块入口 + 自动注册
├── game.py          # Game 接口实现
├── config.py        # 游戏配置（继承自 GameConfig）
└── cchess/          # 底层引擎
    ├── __init__.py  # 棋盘、走子、规则
    ├── engine.py    # 引擎常量
    └── svg.py       # SVG 渲染

使用方法:
    >>> from games.chinese_chess import ChineseChessGame, ChineseChessConfig
    >>> 
    >>> # 使用默认配置
    >>> game = ChineseChessGame()
    >>> obs = game.reset()
    >>> actions = game.legal_actions()
    >>> obs, reward, done, info = game.step(actions[0])
    >>> 
    >>> # 使用自定义配置
    >>> config = ChineseChessConfig(max_game_length=300)
    >>> game = ChineseChessGame(config=config)
    >>> 
    >>> # 或通过 from_config（推荐）
    >>> game = ChineseChessGame.from_config(max_game_length=300)
"""

from .game import (
    ChineseChessGame,
    get_action_space_size,
    action_to_uci,
    uci_to_action,
    augment_action_batch,
    augment_observation,
    augment_policy,
    TOTAL_ACTIONS,
    UCI_TO_IDX,
    IDX_TO_UCI,
)

from .config import (
    ChineseChessConfig,
    ChineseChessDebugConfig,
    # 常量
    BOARD_HEIGHT,
    BOARD_WIDTH,
    NUM_PIECE_TYPES,
    MAX_GAME_LENGTH,
    DEFAULT_HISTORY_STEPS,
    # 默认配置
    DEFAULT_CONFIG,
    DEFAULT_DEBUG_CONFIG,
)

__all__ = [
    # 游戏类
    "ChineseChessGame",
    # 配置类
    "ChineseChessConfig",
    "ChineseChessDebugConfig",
    # 辅助函数
    "get_action_space_size",
    "action_to_uci",
    "uci_to_action",
    "augment_action_batch",
    "augment_observation",
    "augment_policy",
    # 常量
    "TOTAL_ACTIONS",
    "UCI_TO_IDX",
    "IDX_TO_UCI",
    "BOARD_HEIGHT",
    "BOARD_WIDTH",
    "NUM_PIECE_TYPES",
    "MAX_GAME_LENGTH",
    "DEFAULT_HISTORY_STEPS",
    # 默认配置
    "DEFAULT_CONFIG",
    "DEFAULT_DEBUG_CONFIG",
]

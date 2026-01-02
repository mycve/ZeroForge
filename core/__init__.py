"""
Core Framework - 核心框架模块

包含:
- Game ABC: 游戏抽象基类
- Algorithm ABC: 算法抽象基类
- Config: 配置基类（GameConfig, MCTSConfig 等）
- MCTS: CPU 本地树 + GPU 批推理
- Env: nogil 多线程环境
- Logging: 结构化日志系统

架构设计:
- 开发者继承 Game + GameConfig 实现自定义游戏
- 利用 Python 3.13+ nogil 实现多核并行
- CPU 本地 MCTS 树支持节点复用
- GPU 批量叶子推理提高效率
"""

from .game import (
    Game,
    ObservationSpace,
    ActionSpace,
    GameState,
    PlayerType,
    GameMeta,
)
from .algorithm import Algorithm
from .config import (
    GameConfig,
    DebugConfig,
    MCTSConfig,
    BatcherConfig,
    ThreadedEnvConfig,
)
from .training_config import (
    TrainingConfig,
    get_best_device,
    resolve_device,
    get_config_schema,
)
from .logging import StructuredLogger, LogEvent, LogLevel, LogCategory
from .selfplay import ThreadedSelfPlay, EnvWorker, SelfPlayResult, SelfPlayStats
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer, SampleBatch
from .checkpoint import CheckpointManager, CheckpointInfo, get_checkpoint_manager

__all__ = [
    # Game
    "Game",
    "ObservationSpace", 
    "ActionSpace",
    "GameState",
    "PlayerType",
    "GameMeta",
    # Algorithm
    "Algorithm",
    # Config
    "GameConfig",
    "DebugConfig",
    "MCTSConfig",
    "BatcherConfig",
    "ThreadedEnvConfig",
    # Training Config
    "TrainingConfig",
    "get_best_device",
    "resolve_device",
    "get_config_schema",
    # Logging
    "StructuredLogger",
    "LogEvent",
    "LogLevel",
    "LogCategory",
    # Env
    "ThreadedSelfPlay",
    "EnvWorker",
    "SelfPlayResult",
    "SelfPlayStats",
    # Replay Buffer
    "ReplayBuffer",
    "PrioritizedReplayBuffer",
    "SampleBatch",
    # Checkpoint
    "CheckpointManager",
    "CheckpointInfo",
    "get_checkpoint_manager",
]


"""
Core Framework - 核心框架模块

包含:
- Game ABC: 游戏抽象基类
- Algorithm ABC: 算法抽象基类
- Config: 配置基类（GameConfig, MCTSConfig 等）
- MCTS: CPU 本地树 + GPU 批推理
- Trainer: 分布式训练器（支持 DDP）
- SelfPlay: 多线程异步自玩
- Logging: 结构化日志系统

架构设计:
- 开发者继承 Game + GameConfig 实现自定义游戏
- 多线程异步自玩 + 叶节点批量 GPU 推理
- 支持 DDP 多卡分布式训练
- CPU 本地 MCTS 树支持节点复用
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
from .trainer import DistributedTrainer, TrainerState, AsyncSelfPlayWorker

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
    # Trainer
    "DistributedTrainer",
    "TrainerState",
    "AsyncSelfPlayWorker",
    # Logging
    "StructuredLogger",
    "LogEvent",
    "LogLevel",
    "LogCategory",
    # SelfPlay
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


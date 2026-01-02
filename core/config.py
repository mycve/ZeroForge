"""
Config - 配置基类

定义游戏配置和调试配置的基类，所有游戏配置应继承 GameConfig。

设计原则:
1. 统一继承体系：所有游戏配置继承 GameConfig
2. 类型安全：使用 dataclass 确保类型检查
3. 可验证：提供 validate() 方法检查配置合法性
4. 可序列化：支持 to_dict() 和 from_dict() 便于保存/加载
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, Type, TypeVar
import logging

logger = logging.getLogger(__name__)

T = TypeVar('T', bound='GameConfig')


# ============================================================
# 游戏配置基类
# ============================================================

@dataclass
class GameConfig:
    """游戏配置基类
    
    所有游戏的配置类应继承此基类。提供统一的配置管理接口。
    
    Attributes:
        max_game_length: 最大游戏步数，超过后强制结束
        history_steps: 历史步数，用于观测编码
        enable_augmentation: 是否启用数据增强
    
    Example:
        >>> @dataclass
        ... class ChineseChessConfig(GameConfig):
        ...     board_width: int = 9
        ...     board_height: int = 10
        ...
        >>> config = ChineseChessConfig(max_game_length=200)
        >>> config.validate()
    """
    
    # 基础游戏参数
    max_game_length: int = 200
    history_steps: int = 4
    enable_augmentation: bool = True
    
    def validate(self) -> None:
        """验证配置合法性
        
        子类应覆盖此方法添加特定验证逻辑。
        
        Raises:
            ValueError: 如果配置不合法
        """
        if self.max_game_length < 1:
            raise ValueError(f"max_game_length 必须 >= 1，得到 {self.max_game_length}")
        if self.history_steps < 1:
            raise ValueError(f"history_steps 必须 >= 1，得到 {self.history_steps}")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典
        
        Returns:
            配置字典
        """
        return asdict(self)
    
    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """从字典创建配置
        
        Args:
            data: 配置字典
            
        Returns:
            配置实例
        """
        # 只使用该类定义的字段
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)
    
    def __post_init__(self):
        """dataclass 初始化后调用，执行验证"""
        self.validate()


# ============================================================
# 调试配置
# ============================================================

@dataclass
class DebugConfig:
    """调试配置
    
    用于控制调试信息的记录级别和输出。
    
    Attributes:
        log_steps: 是否记录每步详细信息
        log_mcts: 是否记录 MCTS 搜索过程
        log_network: 是否记录网络输入输出
        save_trajectories: 是否保存完整轨迹
        debug_dir: 调试数据保存目录
        max_history: 最大保存历史步数
    """
    
    # 日志控制
    log_steps: bool = False
    log_mcts: bool = False
    log_network: bool = False
    
    # 数据保存
    save_trajectories: bool = False
    debug_dir: str = "debug_logs"
    max_history: int = 1000
    
    # 性能分析
    profile_enabled: bool = False
    profile_interval: int = 100  # 每 N 步记录一次性能数据
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DebugConfig":
        """从字典创建"""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)


# ============================================================
# MCTS 配置
# ============================================================

@dataclass
class MCTSConfig:
    """MCTS 搜索配置
    
    控制 MCTS 搜索的各项参数。
    
    Attributes:
        num_simulations: 每次搜索的模拟次数
        c_puct: UCB 探索常数
        c_visit: 访问计数权重（用于 Gumbel MuZero）
        dirichlet_alpha: Dirichlet 噪声 alpha 参数
        dirichlet_epsilon: Dirichlet 噪声混合比例
        temperature_init: 初始温度
        temperature_final: 最终温度
        temperature_threshold: 温度变化的步数阈值
    """
    
    # 搜索参数
    num_simulations: int = 200
    c_puct: float = 1.5
    c_visit: float = 50.0
    
    # 探索噪声
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    
    # 温度调度
    temperature_init: float = 1.0
    temperature_final: float = 0.1
    temperature_threshold: int = 30
    
    # 树管理
    reuse_tree: bool = True  # 是否复用子树
    max_tree_depth: int = 50  # 最大树深度
    
    def get_temperature(self, move_count: int) -> float:
        """根据步数获取温度
        
        Args:
            move_count: 当前步数
            
        Returns:
            采样温度
        """
        if move_count < self.temperature_threshold:
            return self.temperature_init
        return self.temperature_final
    
    def validate(self) -> None:
        """验证配置合法性"""
        if self.num_simulations < 1:
            raise ValueError(f"num_simulations 必须 >= 1，得到 {self.num_simulations}")
        if self.c_puct <= 0:
            raise ValueError(f"c_puct 必须 > 0，得到 {self.c_puct}")
        if not (0 <= self.dirichlet_epsilon <= 1):
            raise ValueError(f"dirichlet_epsilon 必须在 [0, 1]，得到 {self.dirichlet_epsilon}")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MCTSConfig":
        """从字典创建"""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)
    
    def __post_init__(self):
        """初始化后验证"""
        self.validate()


# ============================================================
# 批推理配置
# ============================================================

@dataclass
class BatcherConfig:
    """批推理配置
    
    控制 GPU 批量推理的行为。
    
    Attributes:
        batch_size: 批大小，达到此数量触发推理
        timeout_ms: 超时时间（毫秒），超时后强制推理
        device: 推理设备
        use_amp: 是否使用自动混合精度
    """
    
    batch_size: int = 256
    timeout_ms: float = 5.0  # 5ms 超时
    device: str = "cuda"
    use_amp: bool = True
    
    def validate(self) -> None:
        """验证配置"""
        if self.batch_size < 1:
            raise ValueError(f"batch_size 必须 >= 1，得到 {self.batch_size}")
        if self.timeout_ms <= 0:
            raise ValueError(f"timeout_ms 必须 > 0，得到 {self.timeout_ms}")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "BatcherConfig":
        """从字典创建"""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)
    
    def __post_init__(self):
        """初始化后验证"""
        self.validate()


# ============================================================
# 多线程环境配置
# ============================================================

@dataclass
class ThreadedEnvConfig:
    """多线程持续自玩配置（旧版，用于 ThreadedSelfPlay）
    
    注意：当前 DistributedTrainer 使用 TrainingConfig 中的配置：
    - TrainingConfig.num_envs: 每 epoch 完成的游戏总数
    - TrainingConfig.concurrency: 同时并发运行的游戏数
    
    此配置类用于 ThreadedSelfPlay（持续自玩模式），其中：
    - num_envs: 持续运行的环境数量（每个环境一个线程）
    - games_per_env: 每个环境运行的游戏数（-1=无限）
    - use_nogil: 是否利用 nogil（Python 3.13+）
    """
    
    num_envs: int = 128  # 持续运行的环境/线程数量
    games_per_env: int = -1  # -1 表示无限
    use_nogil: bool = True
    
    # MCTS 配置
    mcts: MCTSConfig = field(default_factory=MCTSConfig)
    
    # 批推理配置
    batcher: BatcherConfig = field(default_factory=BatcherConfig)
    
    def validate(self) -> None:
        """验证配置"""
        if self.num_envs < 1:
            raise ValueError(f"num_envs 必须 >= 1，得到 {self.num_envs}")
        self.mcts.validate()
        self.batcher.validate()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "num_envs": self.num_envs,
            "games_per_env": self.games_per_env,
            "use_nogil": self.use_nogil,
            "mcts": self.mcts.to_dict(),
            "batcher": self.batcher.to_dict(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ThreadedEnvConfig":
        """从字典创建"""
        mcts_data = data.pop("mcts", {})
        batcher_data = data.pop("batcher", {})
        
        return cls(
            num_envs=data.get("num_envs", 128),
            games_per_env=data.get("games_per_env", -1),
            use_nogil=data.get("use_nogil", True),
            mcts=MCTSConfig.from_dict(mcts_data),
            batcher=BatcherConfig.from_dict(batcher_data),
        )
    
    def __post_init__(self):
        """初始化后验证"""
        self.validate()


# ============================================================
# 导出
# ============================================================

__all__ = [
    "GameConfig",
    "DebugConfig",
    "MCTSConfig",
    "BatcherConfig",
    "ThreadedEnvConfig",
]

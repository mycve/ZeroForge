"""
TrainingConfig - 训练配置

统一的训练配置类，用于 Web 界面和 CLI。

设计原则:
1. 设备自动检测: cuda > mps > cpu
2. 所有参数可通过 Web 界面配置
3. 支持序列化/反序列化
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
import logging

logger = logging.getLogger(__name__)


# ============================================================
# 设备检测
# ============================================================

def get_best_device() -> str:
    """自动检测最佳设备: cuda > mps > cpu"""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def resolve_device(device: str) -> str:
    """解析设备配置，auto 时自动检测"""
    if device == "auto":
        return get_best_device()
    return device


# ============================================================
# 训练配置
# ============================================================

@dataclass
class TrainingConfig:
    """训练配置
    
    Attributes:
        # 游戏和算法
        game_type: 游戏类型 (tictactoe, chinese_chess)
        algorithm: 算法类型 (alphazero, muzero)
        
        # 并发
        num_actors: 自玩线程数
        games_per_actor: 每个线程运行游戏数
        
        # 批处理
        batch_size: 训练批大小
        inference_batch_size: 推理批大小
        
        # 训练超参
        num_epochs: 训练轮数
        lr: 学习率
        weight_decay: 权重衰减
        
        # 网络架构
        num_channels: 网络通道数
        num_blocks: ResNet 块数
        
        # 回放缓冲区
        replay_buffer_size: 缓冲区容量
        min_buffer_size: 开始训练前最小样本数
        
        # MCTS
        num_simulations: 模拟次数
        c_puct: UCB 探索常数
        dirichlet_alpha: Dirichlet 噪声 alpha
        dirichlet_epsilon: 噪声混合比例
        
        # 检查点
        checkpoint_dir: 检查点目录
        save_interval: 保存间隔（轮）
        keep_checkpoints: 保留检查点数量
        
        # 系统
        device: 设备 (auto/cuda/mps/cpu)
        log_backends: 日志后端列表
    """
    
    # === 游戏和算法 ===
    game_type: str = "tictactoe"
    algorithm: str = "alphazero"
    
    # === 并发 ===
    num_actors: int = 4
    games_per_actor: int = 50
    
    # === 批处理 ===
    batch_size: int = 128
    inference_batch_size: int = 32
    
    # === 训练超参 ===
    num_epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    
    # === 网络架构 ===
    network_size: str = "auto"  # auto / small / medium / large
    num_channels: int = 64
    num_blocks: int = 4
    hidden_dim: int = 64  # 用于 SimpleAlphaZeroNetwork
    
    # === 回放缓冲区 ===
    replay_buffer_size: int = 50000
    min_buffer_size: int = 100  # 降低阈值，让训练更快开始
    
    # === MCTS ===
    num_simulations: int = 50
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature_threshold: int = 15
    
    # === 检查点 ===
    checkpoint_dir: str = "./checkpoints"
    save_interval: int = 10
    keep_checkpoints: int = 5
    
    # === 系统 ===
    device: str = "auto"
    log_backends: List[str] = field(default_factory=lambda: ["console"])
    log_dir: str = "./logs"
    
    def get_device(self) -> str:
        """获取实际使用的设备"""
        return resolve_device(self.device)
    
    def validate(self) -> None:
        """验证配置合法性"""
        if self.num_actors < 1:
            raise ValueError(f"num_actors 必须 >= 1，得到 {self.num_actors}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size 必须 >= 1，得到 {self.batch_size}")
        if self.num_epochs < 1:
            raise ValueError(f"num_epochs 必须 >= 1，得到 {self.num_epochs}")
        if self.lr <= 0:
            raise ValueError(f"lr 必须 > 0，得到 {self.lr}")
        if self.num_simulations < 1:
            raise ValueError(f"num_simulations 必须 >= 1，得到 {self.num_simulations}")
        if self.save_interval < 1:
            raise ValueError(f"save_interval 必须 >= 1，得到 {self.save_interval}")
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典（用于 JSON 序列化）"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        """从字典创建配置"""
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_data = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered_data)
    
    def __post_init__(self):
        """初始化后验证"""
        self.validate()


# ============================================================
# 配置分组（用于 Web 界面）
# ============================================================

CONFIG_GROUPS = {
    "game": {
        "label": "游戏设置",
        "fields": ["game_type", "algorithm"],
    },
    "training": {
        "label": "训练设置",
        "fields": ["num_epochs", "batch_size", "lr", "weight_decay"],
    },
    "network": {
        "label": "网络架构",
        "fields": ["network_size", "num_channels", "num_blocks", "hidden_dim"],
    },
    "mcts": {
        "label": "MCTS 设置",
        "fields": ["num_simulations", "c_puct", "dirichlet_alpha", "dirichlet_epsilon", "temperature_threshold"],
    },
    "selfplay": {
        "label": "自玩设置",
        "fields": ["num_actors", "games_per_actor", "inference_batch_size"],
    },
    "buffer": {
        "label": "回放缓冲区",
        "fields": ["replay_buffer_size", "min_buffer_size"],
    },
    "checkpoint": {
        "label": "检查点",
        "fields": ["checkpoint_dir", "save_interval", "keep_checkpoints"],
    },
    "system": {
        "label": "系统设置",
        "fields": ["device", "log_backends", "log_dir"],
    },
}


# 字段元数据（类型、范围、描述）
FIELD_METADATA = {
    "game_type": {"type": "select", "options": ["tictactoe", "chinese_chess"], "label": "游戏类型"},
    "algorithm": {"type": "select", "options": ["alphazero", "muzero"], "label": "算法"},
    "num_epochs": {"type": "int", "min": 1, "max": 100000, "label": "训练轮数"},
    "batch_size": {"type": "int", "min": 1, "max": 4096, "label": "批大小"},
    "lr": {"type": "float", "min": 1e-6, "max": 1.0, "step": 1e-4, "label": "学习率"},
    "weight_decay": {"type": "float", "min": 0, "max": 0.1, "step": 1e-5, "label": "权重衰减"},
    "network_size": {"type": "select", "options": ["auto", "small", "medium", "large"], "label": "网络大小", "description": "auto 自动选择，small 适合小游戏，large 适合复杂游戏"},
    "num_channels": {"type": "int", "min": 8, "max": 512, "label": "网络通道数", "description": "仅 medium/large 网络使用"},
    "num_blocks": {"type": "int", "min": 1, "max": 40, "label": "ResNet 块数", "description": "仅 medium/large 网络使用"},
    "hidden_dim": {"type": "int", "min": 16, "max": 512, "label": "隐藏层维度", "description": "仅 small 网络使用"},
    "num_simulations": {"type": "int", "min": 1, "max": 1600, "label": "MCTS 模拟次数"},
    "c_puct": {"type": "float", "min": 0.1, "max": 10.0, "step": 0.1, "label": "UCB 探索常数"},
    "dirichlet_alpha": {"type": "float", "min": 0.01, "max": 1.0, "step": 0.01, "label": "Dirichlet Alpha"},
    "dirichlet_epsilon": {"type": "float", "min": 0, "max": 1.0, "step": 0.05, "label": "噪声比例"},
    "temperature_threshold": {"type": "int", "min": 0, "max": 100, "label": "温度阈值步数"},
    "num_actors": {"type": "int", "min": 1, "max": 256, "label": "自玩线程数"},
    "games_per_actor": {"type": "int", "min": 1, "max": 1000, "label": "每线程游戏数"},
    "inference_batch_size": {"type": "int", "min": 1, "max": 512, "label": "推理批大小"},
    "replay_buffer_size": {"type": "int", "min": 1000, "max": 10000000, "label": "缓冲区大小"},
    "min_buffer_size": {"type": "int", "min": 10, "max": 100000, "label": "最小缓冲量", "description": "低于此值不开始训练"},
    "checkpoint_dir": {"type": "string", "label": "检查点目录"},
    "save_interval": {"type": "int", "min": 1, "max": 1000, "label": "保存间隔（轮）"},
    "keep_checkpoints": {"type": "int", "min": 1, "max": 100, "label": "保留数量"},
    "device": {"type": "select", "options": ["auto", "cuda", "mps", "cpu"], "label": "计算设备"},
    "log_backends": {"type": "multiselect", "options": ["console", "tensorboard", "wandb", "file"], "label": "日志后端"},
    "log_dir": {"type": "string", "label": "日志目录"},
}


def get_config_schema() -> Dict[str, Any]:
    """获取配置 schema（用于 Web 界面动态渲染）"""
    import copy
    # 返回深拷贝确保是纯 Python 字典，可被 JSON 序列化
    return {
        "groups": copy.deepcopy(CONFIG_GROUPS),
        "fields": copy.deepcopy(FIELD_METADATA),
    }


# ============================================================
# 导出
# ============================================================

__all__ = [
    "TrainingConfig",
    "get_best_device",
    "resolve_device",
    "CONFIG_GROUPS",
    "FIELD_METADATA",
    "get_config_schema",
]

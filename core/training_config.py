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
        inference_batch_size: 推理批大小（叶节点批量推理）
        inference_timeout_ms: 推理超时（毫秒）
        
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
        
        # 分布式训练
        use_ddp: 是否使用 DDP 多卡训练
        ddp_backend: DDP 后端 (nccl/gloo)
        
        # 系统
        device: 设备 (auto/cuda/mps/cpu)
        log_backends: 日志后端列表
    """
    
    # === 游戏和算法 ===
    game_type: str = "tictactoe"
    algorithm: str = "alphazero"
    
    # === 自玩配置 ===
    num_envs: int = 256                     # 每个 epoch 需要完成的游戏数量
    concurrency: int = 64                   # 同时并发运行的游戏数（推荐 64-128 以提高 GPU 利用率）
    train_batches_per_epoch: int = 10      # 每 epoch 训练批次数
    new_data_ratio: float = 0.8            # 新数据占比（80%新数据 + 20%经验池）
    
    # === 批处理 ===
    batch_size: int = 256              # 训练批大小
    inference_batch_size: int = 32     # 叶节点推理批大小（推荐 concurrency/2，GPU 利用率关键参数）
    inference_timeout_ms: float = 2.0  # 推理超时（毫秒），更短的超时减少 GPU 空闲
    
    # === 训练超参 ===
    num_epochs: int = 1000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0  # 梯度裁剪
    
    # === 网络架构 ===
    network_size: str = "auto"  # auto / small / medium / large
    num_channels: int = 64
    num_blocks: int = 4
    hidden_dim: int = 64  # 用于 SimpleAlphaZeroNetwork
    
    # === 回放缓冲区 ===
    replay_buffer_size: int = 100000
    min_buffer_size: int = 200  # 开始训练前最小样本数
    
    # === MCTS ===
    num_simulations: int = 50
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    temperature_threshold: int = 15
    
    # === Gumbel 搜索（用于 gumbel_muzero / gumbel_alphazero）===
    gumbel_max_actions: int = 8        # Gumbel Top-k 考虑的最大动作数
    # halving 轮数自动计算: ceil(log2(gumbel_max_actions))
    gumbel_scale: float = 1.0           # Gumbel 噪声缩放
    gumbel_c_visit: float = 50.0        # Q 值访问计数权重
    gumbel_discount: float = 0.997      # 折扣因子（Gymnasium 环境）
    
    # === 评估（新旧版本对弈）===
    eval_games: int = 5               # 每个 epoch 评估对弈局数
    eval_temperature: float = 0.5      # 评估时的动作采样温度
    
    # === 检查点 ===
    checkpoint_dir: str = "./checkpoints"
    save_interval: int = 10
    keep_checkpoints: int = 10
    
    # === 分布式训练 ===
    use_ddp: bool = False           # 是否使用 DDP
    ddp_backend: str = "nccl"       # DDP 后端: nccl (GPU) / gloo (CPU)
    
    # === 系统 ===
    device: str = "auto"
    log_backends: List[str] = field(default_factory=lambda: ["console"])
    log_dir: str = "./logs"
    
    def get_device(self) -> str:
        """获取实际使用的设备"""
        return resolve_device(self.device)
    
    def validate(self) -> None:
        """验证配置合法性"""
        if self.num_envs < 1:
            raise ValueError(f"num_envs 必须 >= 1，得到 {self.num_envs}")
        if self.concurrency < 1:
            raise ValueError(f"concurrency 必须 >= 1，得到 {self.concurrency}")
        
        # 批推理大小不能超过并发数（每个并发游戏同时只产生一个叶子节点）
        if self.inference_batch_size > self.concurrency:
            raise ValueError(
                f"inference_batch_size ({self.inference_batch_size}) 不能超过 concurrency ({self.concurrency})，"
                f"因为每个并发游戏同一时刻只产生一个叶子节点。"
                f"推荐设置为 concurrency 的一半（{self.concurrency // 2}）以形成流水线。"
            )
        if not 0.0 <= self.new_data_ratio <= 1.0:
            raise ValueError(f"new_data_ratio 必须在 [0, 1] 范围内，得到 {self.new_data_ratio}")
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
        "fields": ["num_epochs", "batch_size", "lr", "weight_decay", "grad_clip"],
    },
    "network": {
        "label": "网络架构",
        "fields": ["network_size", "num_channels", "num_blocks", "hidden_dim"],
    },
    "mcts": {
        "label": "MCTS 设置",
        "fields": ["num_simulations", "c_puct", "dirichlet_alpha", "dirichlet_epsilon", "temperature_threshold"],
    },
    "gumbel": {
        "label": "Gumbel 搜索",
        "description": "用于 gumbel_muzero / gumbel_alphazero 算法（Halving 轮数自动计算）",
        "fields": ["gumbel_max_actions", "gumbel_scale", "gumbel_c_visit", "gumbel_discount"],
    },
    "selfplay": {
        "label": "自玩设置",
        "fields": ["num_envs", "concurrency", "inference_batch_size", "inference_timeout_ms", 
                   "train_batches_per_epoch", "new_data_ratio"],
    },
    "buffer": {
        "label": "回放缓冲区",
        "fields": ["replay_buffer_size", "min_buffer_size"],
    },
    "eval": {
        "label": "评估设置",
        "fields": ["eval_games", "eval_temperature"],
    },
    "checkpoint": {
        "label": "检查点",
        "fields": ["checkpoint_dir", "save_interval", "keep_checkpoints"],
    },
    "distributed": {
        "label": "分布式训练",
        "fields": ["use_ddp", "ddp_backend"],
    },
    "system": {
        "label": "系统设置",
        "fields": ["device", "log_backends", "log_dir"],
    },
}


# 字段元数据（类型、范围、描述）
FIELD_METADATA = {
    "game_type": {"type": "select", "options": ["tictactoe", "chinese_chess"], "label": "游戏类型"},
    "algorithm": {"type": "select", "options": ["alphazero", "muzero", "gumbel_alphazero", "gumbel_muzero"], "label": "算法", "description": "gumbel 系列适用于 Gymnasium 等不支持克隆的环境"},
    "num_epochs": {"type": "int", "min": 1, "max": 100000, "label": "训练轮数"},
    "batch_size": {"type": "int", "min": 1, "max": 4096, "label": "训练批大小"},
    "lr": {"type": "float", "min": 1e-6, "max": 1.0, "step": 1e-4, "label": "学习率"},
    "weight_decay": {"type": "float", "min": 0, "max": 0.1, "step": 1e-5, "label": "权重衰减"},
    "grad_clip": {"type": "float", "min": 0.1, "max": 10.0, "step": 0.1, "label": "梯度裁剪"},
    "network_size": {"type": "select", "options": ["auto", "small", "medium", "large"], "label": "网络大小", "description": "auto 自动选择，small 适合小游戏，large 适合复杂游戏"},
    "num_channels": {"type": "int", "min": 8, "max": 512, "label": "网络通道数", "description": "仅 medium/large 网络使用"},
    "num_blocks": {"type": "int", "min": 1, "max": 40, "label": "ResNet 块数", "description": "仅 medium/large 网络使用"},
    "hidden_dim": {"type": "int", "min": 16, "max": 512, "label": "隐藏层维度", "description": "仅 small 网络使用"},
    "num_simulations": {"type": "int", "min": 1, "max": 1600, "label": "MCTS 模拟次数"},
    "c_puct": {"type": "float", "min": 0.1, "max": 10.0, "step": 0.1, "label": "UCB 探索常数"},
    "dirichlet_alpha": {"type": "float", "min": 0.01, "max": 1.0, "step": 0.01, "label": "Dirichlet Alpha"},
    "dirichlet_epsilon": {"type": "float", "min": 0, "max": 1.0, "step": 0.05, "label": "噪声比例"},
    "temperature_threshold": {"type": "int", "min": 0, "max": 100, "label": "温度阈值步数"},
    # Gumbel 搜索配置
    "gumbel_max_actions": {"type": "int", "min": 2, "max": 64, "label": "Top-k 动作数", "description": "Gumbel-Top-k 采样考虑的最大动作数（Halving 轮数 = ceil(log2(k)) 自动计算）"},
    "gumbel_scale": {"type": "float", "min": 0.1, "max": 5.0, "step": 0.1, "label": "Gumbel 缩放", "description": "Gumbel 噪声缩放因子"},
    "gumbel_c_visit": {"type": "float", "min": 1.0, "max": 200.0, "step": 1.0, "label": "访问权重", "description": "Q 值计算中的访问计数权重"},
    "gumbel_discount": {"type": "float", "min": 0.9, "max": 1.0, "step": 0.001, "label": "折扣因子", "description": "奖励折扣因子（Gymnasium 环境使用）"},
    "num_envs": {"type": "int", "min": 1, "max": 1000, "label": "游戏数量", "description": "每个 epoch 需要完成的游戏数量"},
    "concurrency": {"type": "int", "min": 1, "max": 512, "label": "并发数", "description": "同时并行运行的游戏数（GPU 利用率关键参数，推荐 64-128）"},
    "train_batches_per_epoch": {"type": "int", "min": 1, "max": 1000, "label": "每epoch训练批次", "description": "每个 epoch 从缓冲区采样训练的批次数"},
    "new_data_ratio": {"type": "float", "min": 0.0, "max": 1.0, "step": 0.1, "label": "新数据占比", "description": "训练时新数据占比（如 0.8 表示 80% 新数据 + 20% 经验池）"},
    "inference_batch_size": {"type": "int", "min": 1, "max": 512, "label": "推理批大小", "description": "叶节点批量推理（≤concurrency，推荐 concurrency/2 形成流水线）"},
    "inference_timeout_ms": {"type": "float", "min": 0.5, "max": 100.0, "step": 0.5, "label": "推理超时(ms)", "description": "超时后强制执行批推理（更小=GPU更忙但延迟更低）"},
    "replay_buffer_size": {"type": "int", "min": 1000, "max": 10000000, "label": "缓冲区大小"},
    "min_buffer_size": {"type": "int", "min": 10, "max": 100000, "label": "最小缓冲量", "description": "低于此值不开始训练"},
    "eval_games": {"type": "int", "min": 0, "max": 100, "label": "评估局数", "description": "每 epoch 新旧版本对弈局数（0=禁用）"},
    "eval_temperature": {"type": "float", "min": 0.1, "max": 2.0, "step": 0.1, "label": "评估温度", "description": "评估对弈时的动作采样温度"},
    "checkpoint_dir": {"type": "string", "label": "检查点目录"},
    "save_interval": {"type": "int", "min": 1, "max": 1000, "label": "保存间隔（轮）"},
    "keep_checkpoints": {"type": "int", "min": 1, "max": 100, "label": "保留数量"},
    "use_ddp": {"type": "bool", "label": "启用 DDP", "description": "分布式数据并行训练（多卡）"},
    "ddp_backend": {"type": "select", "options": ["nccl", "gloo"], "label": "DDP 后端", "description": "nccl 适用于 GPU，gloo 适用于 CPU"},
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

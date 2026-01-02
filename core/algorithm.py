"""
Algorithm ABC - 算法抽象基类

定义通用强化学习算法接口，支持:
- MuZero: 需要 dynamics network，学习环境模型
- AlphaZero: 不需要 dynamics network，使用真实环境

设计原则:
1. 算法与游戏解耦：通过 Game 接口获取环境信息
2. 网络与算法解耦：通过 create_network() 创建适配的网络
3. 搜索与算法解耦：通过 create_search() 创建搜索策略
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, Tuple, List, TYPE_CHECKING
import torch
import torch.nn as nn
import numpy as np
import logging

if TYPE_CHECKING:
    from .game import Game

logger = logging.getLogger(__name__)


# ============================================================
# 配置基类
# ============================================================

@dataclass
class AlgorithmConfig:
    """算法配置基类
    
    子类应继承并添加特定配置参数。
    """
    # 网络配置
    num_channels: int = 128
    num_blocks: int = 6
    
    # 搜索配置
    num_simulations: int = 200
    top_k: int = 8
    c_puct: float = 1.5
    
    # 训练配置
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    
    # 设备
    device: str = "cuda"


# ============================================================
# 轨迹数据结构
# ============================================================

@dataclass
class Trajectory:
    """游戏轨迹数据
    
    存储一局游戏的完整轨迹，用于训练。
    
    Attributes:
        observations: 观测序列 [T, *obs_shape]
        actions: 动作序列 [T]
        rewards: 奖励序列 [T]
        policies: 搜索策略序列 [T, action_space]
        values: 价值预测序列 [T]
        to_play: 玩家序列 [T]
    """
    observations: List[np.ndarray] = field(default_factory=list)
    actions: List[int] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    policies: List[Dict[int, float]] = field(default_factory=list)
    values: List[float] = field(default_factory=list)
    to_play: List[int] = field(default_factory=list)
    
    def __len__(self) -> int:
        return len(self.actions)
    
    def append(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        policy: Dict[int, float],
        value: float,
        to_play: int
    ):
        """添加一步数据"""
        self.observations.append(observation)
        self.actions.append(action)
        self.rewards.append(reward)
        self.policies.append(policy)
        self.values.append(value)
        self.to_play.append(to_play)


@dataclass
class TrainingTargets:
    """训练目标数据
    
    由算法的 compute_targets() 方法生成。
    
    Attributes:
        observations: 观测张量 [B, *obs_shape]
        actions: 动作张量 [B, unroll_steps]
        target_values: 目标价值 [B, unroll_steps+1]
        target_rewards: 目标奖励 [B, unroll_steps+1]
        target_policies: 目标策略 [B, unroll_steps+1, action_space]
        masks: 有效性掩码 [B, unroll_steps+1]
    """
    observations: torch.Tensor
    actions: torch.Tensor
    target_values: torch.Tensor
    target_rewards: torch.Tensor
    target_policies: torch.Tensor
    masks: torch.Tensor


# ============================================================
# 搜索接口
# ============================================================

class Search(ABC):
    """搜索算法抽象基类
    
    定义 MCTS 及其变体的通用接口。
    """
    
    @abstractmethod
    def run(
        self,
        game: "Game",
        network: nn.Module,
        temperature: float = 1.0
    ) -> Tuple[int, Dict[int, float], float]:
        """运行搜索
        
        Args:
            game: 当前游戏状态（会被克隆，不修改原状态）
            network: 神经网络
            temperature: 采样温度
        
        Returns:
            action: 选择的动作
            policy: 搜索后的策略分布 {action: prob}
            value: 根节点价值估计
        """
        ...
    
    @abstractmethod
    def run_batch(
        self,
        observations: torch.Tensor,
        legal_masks: torch.Tensor,
        temperature: float = 1.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """批量运行搜索（GPU 加速）
        
        Args:
            observations: 观测批次 [B, *obs_shape]
            legal_masks: 合法动作掩码 [B, action_space]
            temperature: 采样温度
        
        Returns:
            actions: 选择的动作 [B]
            policies: 策略分布 [B, action_space]
            values: 价值估计 [B]
        """
        ...


# ============================================================
# 算法抽象基类
# ============================================================

class Algorithm(ABC):
    """算法抽象基类
    
    定义 MuZero/AlphaZero 系列算法的通用接口。
    
    核心属性:
    - needs_dynamics: 是否需要 dynamics network
    - name: 算法名称
    
    核心方法:
    - create_network(): 创建神经网络
    - create_search(): 创建搜索策略
    - compute_targets(): 计算训练目标
    - compute_loss(): 计算损失函数
    
    Usage:
        >>> algo = MuZeroAlgorithm(config)
        >>> network = algo.create_network(game)
        >>> search = algo.create_search(network)
        >>> action, policy, value = search.run(game, network)
    """
    
    def __init__(self, config: AlgorithmConfig):
        self.config = config
    
    # === 核心属性 ===
    
    @property
    @abstractmethod
    def needs_dynamics(self) -> bool:
        """是否需要 dynamics network
        
        - MuZero: True (学习环境模型)
        - AlphaZero: False (使用真实环境)
        """
        ...
    
    @property
    @abstractmethod
    def name(self) -> str:
        """算法名称"""
        ...
    
    # === 网络创建 ===
    
    @abstractmethod
    def create_network(self, game: "Game") -> nn.Module:
        """创建神经网络
        
        Args:
            game: 游戏实例，用于获取观测/动作空间
        
        Returns:
            network: 适配该算法的神经网络
        
        Note:
            MuZero 返回带 dynamics 的网络
            AlphaZero 返回只有 representation + prediction 的网络
        """
        ...
    
    @abstractmethod
    def create_search(self, network: nn.Module) -> Search:
        """创建搜索策略
        
        Args:
            network: 神经网络
        
        Returns:
            search: 搜索策略实例
        """
        ...
    
    # === 训练相关 ===
    
    @abstractmethod
    def compute_targets(
        self,
        trajectory: Trajectory,
        start_idx: int,
        unroll_steps: int
    ) -> Dict[str, np.ndarray]:
        """计算训练目标
        
        从轨迹中提取训练样本。
        
        Args:
            trajectory: 游戏轨迹
            start_idx: 起始索引
            unroll_steps: 展开步数
        
        Returns:
            targets: 训练目标字典，包含:
                - 'observation': 初始观测
                - 'actions': 动作序列
                - 'target_values': 目标价值
                - 'target_rewards': 目标奖励
                - 'target_policies': 目标策略
                - 'masks': 有效性掩码
        """
        ...
    
    @abstractmethod
    def compute_loss(
        self,
        network: nn.Module,
        batch: TrainingTargets
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算损失函数
        
        Args:
            network: 神经网络
            batch: 训练批次数据
        
        Returns:
            loss: 总损失（标量张量）
            metrics: 损失分量字典 {'value_loss': ..., 'policy_loss': ..., ...}
        """
        ...
    
    # === 实用方法 ===
    
    def get_info(self) -> Dict[str, Any]:
        """获取算法信息"""
        return {
            "name": self.name,
            "needs_dynamics": self.needs_dynamics,
            "config": self.config.__dict__,
        }
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name}, dynamics={self.needs_dynamics})"


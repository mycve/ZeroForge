"""
AlphaZero Algorithm - AlphaZero 算法实现

AlphaZero 与 MuZero 的区别:
1. 不学习环境模型（无 dynamics network）
2. MCTS 使用真实游戏规则展开
3. 训练只有 value loss 和 policy loss

适用场景:
- 有完美游戏规则模拟器
- 不需要预测奖励
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple, TYPE_CHECKING
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from core.algorithm import Algorithm, AlgorithmConfig, Trajectory, TrainingTargets, Search
from algorithms import register_algorithm

if TYPE_CHECKING:
    from core.game import Game

logger = logging.getLogger(__name__)


# ============================================================
# AlphaZero 配置
# ============================================================

@dataclass
class AlphaZeroConfig(AlgorithmConfig):
    """AlphaZero 专属配置"""
    
    # 网络配置
    network_size: str = "auto"  # auto / small / medium / large
    num_channels: int = 128
    num_blocks: int = 6
    hidden_dim: int = 64  # 用于 SimpleAlphaZeroNetwork
    backbone: str = "convnext"
    
    # 搜索配置
    num_simulations: int = 800  # AlphaZero 通常使用更多模拟
    c_puct: float = 1.5
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.25
    
    # 训练配置
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    
    # 损失权重
    value_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0
    
    # 温度配置
    temperature_threshold: int = 30
    temperature_explore: float = 1.0
    temperature_exploit: float = 0.0
    
    # 设备
    device: str = "cuda"
    use_amp: bool = True
    
    def get_temperature(self, move_number: int) -> float:
        """根据步数获取温度"""
        if move_number < self.temperature_threshold:
            return self.temperature_explore
        return self.temperature_exploit
    
    def get_network_size(self, obs_shape: tuple, action_space: int) -> str:
        """自动确定网络大小
        
        根据观测空间和动作空间大小自动选择网络：
        - small: 观测 < 100 且动作 < 100（如井字棋）
        - medium: 中等规模
        - large: 大规模（如中国象棋）
        """
        if self.network_size != "auto":
            return self.network_size
        
        obs_size = 1
        for d in obs_shape:
            obs_size *= d
        
        # 根据问题规模选择
        if obs_size < 100 and action_space < 100:
            return "small"
        elif obs_size < 1000 and action_space < 500:
            return "medium"
        else:
            return "large"


# ============================================================
# AlphaZero 算法
# ============================================================

@register_algorithm("alphazero")
class AlphaZeroAlgorithm(Algorithm):
    """AlphaZero 算法实现
    
    核心组件:
    - Representation Network: 观测 -> 特征
    - Prediction Network: 特征 -> (策略, 价值)
    
    注意: 无 Dynamics Network，MCTS 使用真实游戏规则
    """
    
    def __init__(self, config: AlphaZeroConfig = None):
        if config is None:
            config = AlphaZeroConfig()
        super().__init__(config)
        self.config: AlphaZeroConfig = config
    
    @property
    def needs_dynamics(self) -> bool:
        return False  # AlphaZero 不需要 dynamics
    
    @property
    def name(self) -> str:
        return "AlphaZero"
    
    def create_network(self, game: "Game") -> nn.Module:
        """创建 AlphaZero 网络
        
        根据 network_size 配置选择网络：
        - small: SimpleAlphaZeroNetwork（MLP，适合小游戏）
        - medium/large: AlphaZeroNetwork（ConvNeXt，适合复杂游戏）
        """
        from .network import AlphaZeroNetwork, SimpleAlphaZeroNetwork
        
        obs_shape = game.observation_space.shape
        action_space = game.action_space.n
        
        # 确定网络大小
        network_size = self.config.get_network_size(obs_shape, action_space)
        
        if network_size == "small":
            # 轻量网络：适用于井字棋等小游戏
            logger.info(f"使用 SimpleAlphaZeroNetwork (hidden_dim={self.config.hidden_dim})")
            return SimpleAlphaZeroNetwork(
                obs_shape=obs_shape,
                action_space=action_space,
                hidden_dim=self.config.hidden_dim,
                num_layers=2,
            )
        elif network_size == "medium":
            # 中等网络
            logger.info(f"使用 AlphaZeroNetwork (channels=64, blocks=3)")
            return AlphaZeroNetwork(
                obs_shape=obs_shape,
                action_space=action_space,
                channels=64,
                num_blocks=3,
            )
        else:
            # 大型网络
            logger.info(f"使用 AlphaZeroNetwork (channels={self.config.num_channels}, "
                       f"blocks={self.config.num_blocks})")
            return AlphaZeroNetwork(
                obs_shape=obs_shape,
                action_space=action_space,
                channels=self.config.num_channels,
                num_blocks=self.config.num_blocks,
            )
    
    def create_search(self, game: "Game", network: nn.Module) -> "MCTSSearch":
        """创建搜索策略
        
        使用 core.mcts 的统一实现。
        
        Args:
            game: 游戏实例
            network: 神经网络
            
        Returns:
            MCTSSearch 实例
        """
        from core.mcts import MCTSSearch, LeafBatcher
        from core.config import MCTSConfig, BatcherConfig
        
        # 从算法配置创建 MCTS 配置
        mcts_config = MCTSConfig(
            num_simulations=self.config.num_simulations,
            c_puct=self.config.c_puct,
            dirichlet_alpha=self.config.dirichlet_alpha,
            dirichlet_epsilon=self.config.dirichlet_epsilon,
            temperature_init=self.config.temperature_explore,
            temperature_final=self.config.temperature_exploit,
            temperature_threshold=self.config.temperature_threshold,
        )
        
        # 创建 batcher
        batcher_config = BatcherConfig(device=self.config.device)
        batcher = LeafBatcher(network, batcher_config)
        
        return MCTSSearch(game, mcts_config, batcher=batcher, mode="alphazero")
    
    def compute_targets(
        self,
        trajectory: Trajectory,
        start_idx: int,
        unroll_steps: int = 0  # AlphaZero 不需要 unroll
    ) -> Dict[str, np.ndarray]:
        """计算训练目标
        
        AlphaZero 只需要单步目标：
        - 观测
        - 目标策略（MCTS 搜索结果）
        - 目标价值（游戏结果）
        """
        idx = start_idx
        
        obs = trajectory.observations[idx]
        
        # 目标价值：从当前玩家视角的最终结果
        target_value = self._compute_target_value(trajectory, idx)
        
        # 目标策略
        policy = trajectory.policies[idx]
        target_policy = self._policy_dict_to_array(policy)
        
        return {
            "observation": obs,
            "target_value": np.array([target_value], dtype=np.float32),
            "target_policy": target_policy,
        }
    
    def _compute_target_value(self, trajectory: Trajectory, idx: int) -> float:
        """计算目标价值"""
        final_reward = trajectory.rewards[-1]
        to_play = trajectory.to_play[idx]
        final_player = trajectory.to_play[-1]
        
        # 从当前玩家视角
        if to_play == final_player:
            return final_reward
        else:
            return -final_reward
    
    def _policy_dict_to_array(self, policy: Dict[int, float]) -> np.ndarray:
        """策略字典转数组"""
        arr = np.zeros(self._action_space_size, dtype=np.float32)
        for action, prob in policy.items():
            if 0 <= action < self._action_space_size:
                arr[action] = prob
        return arr
    
    @property
    def _action_space_size(self) -> int:
        return getattr(self, "_cached_action_space", 2086)
    
    def compute_loss(
        self,
        network: nn.Module,
        batch: TrainingTargets
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算损失函数
        
        AlphaZero 损失:
        - Value Loss: MSE(predicted_value, target_value)
        - Policy Loss: CrossEntropy(predicted_policy, target_policy)
        """
        batch_size = batch.observations.shape[0]
        
        # 前向传播
        policy_logits, value = network(batch.observations)
        
        # Value loss
        value_loss = F.mse_loss(
            value.squeeze(-1),
            batch.target_values[:, 0]  # 只有一步
        )
        
        # Policy loss
        log_probs = F.log_softmax(policy_logits, dim=-1)
        policy_loss = -(batch.target_policies[:, 0] * log_probs).sum(dim=-1).mean()
        
        # 总损失
        total_loss = (
            self.config.value_loss_weight * value_loss +
            self.config.policy_loss_weight * policy_loss
        )
        
        metrics = {
            "total_loss": total_loss.item(),
            "value_loss": value_loss.item(),
            "policy_loss": policy_loss.item(),
        }
        
        return total_loss, metrics


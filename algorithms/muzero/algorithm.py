"""
MuZero Algorithm - MuZero 算法实现

MuZero 的核心特点:
1. 学习环境模型（dynamics network）
2. 不需要环境规则，只需要观测和奖励
3. 适用于任意游戏

训练流程:
1. 自玩生成轨迹
2. 从轨迹采样训练数据
3. 计算 value/policy/reward loss
4. 更新网络参数
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, List, Optional, TYPE_CHECKING
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
# MuZero 配置
# ============================================================

@dataclass
class MuZeroConfig(AlgorithmConfig):
    """MuZero 专属配置"""
    
    # 网络配置
    num_channels: int = 128
    num_blocks: int = 6
    backbone: str = "convnext"
    
    # 搜索配置
    num_simulations: int = 200
    top_k: int = 8
    max_nodes: int = 400
    c_visit: float = 50.0
    c_scale: float = 1.5
    
    # 训练配置
    batch_size: int = 512
    lr: float = 1e-3
    weight_decay: float = 1e-4
    num_unroll_steps: int = 5
    
    # 损失权重
    value_loss_weight: float = 1.0
    policy_loss_weight: float = 1.0
    reward_loss_weight: float = 1.0
    
    # 温度配置
    temperature_threshold: int = 60
    temperature_explore: float = 1.0
    temperature_exploit: float = 0.1
    
    # 设备
    device: str = "cuda"
    use_amp: bool = True
    
    def get_temperature(self, move_number: int) -> float:
        """根据步数获取温度"""
        if move_number < self.temperature_threshold:
            return self.temperature_explore
        return self.temperature_exploit


# ============================================================
# MuZero 算法
# ============================================================

@register_algorithm("muzero")
class MuZeroAlgorithm(Algorithm):
    """MuZero 算法实现
    
    核心组件:
    - Representation Network: 观测 -> 隐藏状态
    - Dynamics Network: (隐藏状态, 动作) -> (下一状态, 奖励)
    - Prediction Network: 隐藏状态 -> (策略, 价值)
    """
    
    def __init__(self, config: MuZeroConfig = None):
        if config is None:
            config = MuZeroConfig()
        super().__init__(config)
        self.config: MuZeroConfig = config
    
    @property
    def needs_dynamics(self) -> bool:
        return True
    
    @property
    def name(self) -> str:
        return "MuZero"
    
    def create_network(self, game: "Game") -> nn.Module:
        """创建 MuZero 网络"""
        from .network import MuZeroNetwork
        
        obs_shape = game.observation_space.shape
        action_space = game.action_space.n
        
        return MuZeroNetwork(
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
            c_puct=self.config.c_scale,
            c_visit=self.config.c_visit,
            temperature_init=self.config.temperature_explore,
            temperature_final=self.config.temperature_exploit,
            temperature_threshold=self.config.temperature_threshold,
        )
        
        # 创建 batcher
        batcher_config = BatcherConfig(device=self.config.device)
        batcher = LeafBatcher(network, batcher_config)
        
        return MCTSSearch(game, mcts_config, batcher=batcher, mode="muzero")
    
    def compute_targets(
        self,
        trajectory: Trajectory,
        start_idx: int,
        unroll_steps: int
    ) -> Dict[str, np.ndarray]:
        """计算训练目标
        
        Args:
            trajectory: 游戏轨迹
            start_idx: 起始索引
            unroll_steps: 展开步数
        
        Returns:
            训练目标字典
        """
        T = len(trajectory)
        
        # 提取数据
        obs = trajectory.observations[start_idx]
        actions = []
        target_values = []
        target_rewards = []
        target_policies = []
        masks = []
        
        for k in range(unroll_steps + 1):
            idx = start_idx + k
            
            if idx < T:
                masks.append(1.0)
                
                # 动作（最后一步没有动作）
                if k < unroll_steps and idx < T:
                    actions.append(trajectory.actions[idx])
                
                # 目标价值（使用 Monte Carlo 回报或 bootstrap）
                target_values.append(self._compute_target_value(trajectory, idx))
                
                # 目标奖励
                target_rewards.append(trajectory.rewards[idx])
                
                # 目标策略
                policy = trajectory.policies[idx]
                target_policies.append(self._policy_dict_to_array(policy))
            else:
                masks.append(0.0)
                if k < unroll_steps:
                    actions.append(0)
                target_values.append(0.0)
                target_rewards.append(0.0)
                target_policies.append(np.zeros(self._action_space_size))
        
        return {
            "observation": obs,
            "actions": np.array(actions, dtype=np.int64),
            "target_values": np.array(target_values, dtype=np.float32),
            "target_rewards": np.array(target_rewards, dtype=np.float32),
            "target_policies": np.stack(target_policies).astype(np.float32),
            "masks": np.array(masks, dtype=np.float32),
        }
    
    def _compute_target_value(self, trajectory: Trajectory, idx: int) -> float:
        """计算目标价值（Monte Carlo）"""
        final_reward = trajectory.rewards[-1]
        to_play = trajectory.to_play[idx]
        # 从当前玩家视角计算价值
        return final_reward * (1 if to_play == trajectory.to_play[-1] else -1)
    
    def _policy_dict_to_array(self, policy: Dict[int, float]) -> np.ndarray:
        """策略字典转数组"""
        arr = np.zeros(self._action_space_size, dtype=np.float32)
        for action, prob in policy.items():
            if 0 <= action < self._action_space_size:
                arr[action] = prob
        return arr
    
    @property
    def _action_space_size(self) -> int:
        """动作空间大小（需要在创建网络时设置）"""
        return getattr(self, "_cached_action_space", 2086)
    
    def compute_loss(
        self,
        network: nn.Module,
        batch: TrainingTargets
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算损失函数
        
        损失组成:
        - Value Loss: MSE(predicted_value, target_value)
        - Policy Loss: CrossEntropy(predicted_policy, target_policy)
        - Reward Loss: MSE(predicted_reward, target_reward)
        """
        device = batch.observations.device
        batch_size = batch.observations.shape[0]
        
        # 初始推理
        hidden, policy_logits, value = network.initial_inference(batch.observations)
        
        # 初始 value loss
        value_loss = (batch.masks[:, 0] * F.mse_loss(
            value.squeeze(-1), batch.target_values[:, 0], reduction='none'
        )).sum()
        
        # 初始 policy loss
        log_probs = F.log_softmax(policy_logits, dim=-1)
        policy_loss = -(batch.masks[:, 0] * (batch.target_policies[:, 0] * log_probs).sum(dim=-1)).sum()
        
        reward_loss = torch.tensor(0.0, device=device)
        
        # Unroll
        current_hidden = hidden
        for k in range(self.config.num_unroll_steps):
            step_actions = batch.actions[:, k]
            current_hidden, reward, policy_logits, value = network.recurrent_inference(
                current_hidden, step_actions
            )
            
            step_mask = batch.masks[:, k + 1]
            
            # Reward loss
            reward_loss = reward_loss + (step_mask * F.mse_loss(
                reward.squeeze(-1), batch.target_rewards[:, k + 1], reduction='none'
            )).sum()
            
            # Value loss
            value_loss = value_loss + (step_mask * F.mse_loss(
                value.squeeze(-1), batch.target_values[:, k + 1], reduction='none'
            )).sum()
            
            # Policy loss
            log_probs = F.log_softmax(policy_logits, dim=-1)
            policy_loss = policy_loss - (step_mask * (batch.target_policies[:, k + 1] * log_probs).sum(dim=-1)).sum()
        
        # 总损失
        total_loss = (
            self.config.value_loss_weight * value_loss +
            self.config.policy_loss_weight * policy_loss +
            self.config.reward_loss_weight * reward_loss
        ) / batch_size
        
        metrics = {
            "total_loss": total_loss.item(),
            "value_loss": value_loss.item() / batch_size,
            "policy_loss": policy_loss.item() / batch_size,
            "reward_loss": reward_loss.item() / batch_size,
        }
        
        return total_loss, metrics


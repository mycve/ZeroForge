"""
Gumbel MuZero / AlphaZero 算法实现

基于论文: Policy improvement by planning with Gumbel (ICLR 2022)

核心改进:
1. 使用 Gumbel-Top-k 采样替代 UCB 选择
2. 使用 Sequential Halving 高效分配搜索预算
3. 不需要环境克隆（适用于 Gymnasium 等环境）

与标准 MuZero/AlphaZero 的区别:
- 搜索算法不同（Gumbel vs MCTS）
- 训练损失相同
- 网络结构相同
"""

from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, TYPE_CHECKING
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from core.algorithm import Algorithm, AlgorithmConfig, Trajectory, TrainingTargets
from algorithms import register_algorithm
from core.mcts import GumbelMCTSSearch, MCTSConfig
from .search import GumbelSearch, GumbelConfig

if TYPE_CHECKING:
    from core.game import Game

logger = logging.getLogger(__name__)


# ============================================================
# Gumbel MuZero 配置
# ============================================================

@dataclass
class GumbelMuZeroConfig(AlgorithmConfig):
    """Gumbel MuZero 配置"""
    
    # 网络配置
    num_channels: int = 128
    num_blocks: int = 6
    
    # 搜索配置（Gumbel 特有）
    num_simulations: int = 50           # 比标准 MCTS 少很多
    max_num_considered_actions: int = 16  # top-k 动作数
    gumbel_scale: float = 1.0           # Gumbel 噪声缩放
    # num_halving_rounds 由 max_num_considered_actions 自动计算: ceil(log2(k))
    
    # Q 值配置
    c_visit: float = 50.0
    c_scale: float = 1.0
    discount: float = 0.997
    
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
    temperature_threshold: int = 30
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
    
    def to_mcts_config(self) -> MCTSConfig:
        """转换为 MCTSConfig（用于 Gumbel MCTS）"""
        return MCTSConfig(
            num_simulations=self.num_simulations,
            c_puct=self.c_scale,
            c_visit=self.c_visit,
            temperature_init=self.temperature_explore,
            temperature_final=self.temperature_exploit,
            temperature_threshold=self.temperature_threshold,
        )


# ============================================================
# Gumbel AlphaZero 配置
# ============================================================

@dataclass
class GumbelAlphaZeroConfig(AlgorithmConfig):
    """Gumbel AlphaZero 配置"""
    
    # 网络配置
    network_size: str = "auto"
    num_channels: int = 128
    num_blocks: int = 6
    hidden_dim: int = 64
    
    # 搜索配置（Gumbel 特有）
    num_simulations: int = 50
    max_num_considered_actions: int = 16
    gumbel_scale: float = 1.0
    # num_halving_rounds 由 max_num_considered_actions 自动计算
    
    # Q 值配置
    c_visit: float = 50.0
    c_scale: float = 1.5
    discount: float = 1.0  # 棋类游戏通常不折扣
    
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
        if move_number < self.temperature_threshold:
            return self.temperature_explore
        return self.temperature_exploit
    
    def get_network_size(self, obs_shape: tuple, action_space: int) -> str:
        """自动确定网络大小"""
        if self.network_size != "auto":
            return self.network_size
        
        obs_size = 1
        for d in obs_shape:
            obs_size *= d
        
        if obs_size < 100 and action_space < 100:
            return "small"
        elif obs_size < 1000 and action_space < 500:
            return "medium"
        else:
            return "large"
    
    def to_mcts_config(self) -> MCTSConfig:
        """转换为 MCTSConfig（用于 Gumbel MCTS）"""
        return MCTSConfig(
            num_simulations=self.num_simulations,
            c_puct=self.c_scale,
            c_visit=self.c_visit,
            temperature_init=self.temperature_explore,
            temperature_final=self.temperature_exploit,
            temperature_threshold=self.temperature_threshold,
        )


# ============================================================
# Gumbel MuZero 算法
# ============================================================

@register_algorithm("gumbel_muzero")
class GumbelMuZeroAlgorithm(Algorithm):
    """Gumbel MuZero 算法
    
    使用 Gumbel-Top-k + Sequential Halving 替代传统 MCTS。
    不需要环境克隆，适用于 Gymnasium 等环境。
    
    核心组件:
    - Representation Network: 观测 -> 隐藏状态
    - Dynamics Network: (隐藏状态, 动作) -> (下一状态, 奖励)
    - Prediction Network: 隐藏状态 -> (策略, 价值)
    """
    
    def __init__(self, config: GumbelMuZeroConfig = None):
        if config is None:
            config = GumbelMuZeroConfig()
        super().__init__(config)
        self.config: GumbelMuZeroConfig = config
    
    @property
    def needs_dynamics(self) -> bool:
        return True  # MuZero 需要 dynamics
    
    @property
    def name(self) -> str:
        return "Gumbel MuZero"
    
    def create_network(self, game: "Game") -> nn.Module:
        """创建 MuZero 网络
        
        根据观测形状自动选择网络类型：
        - 3D 观测（图像）：使用 MuZeroNetwork（ConvNeXt）
        - 其他：使用 SimpleMuZeroNetwork（MLP）
        """
        from algorithms.muzero.network import MuZeroNetwork, SimpleMuZeroNetwork
        
        obs_shape = game.observation_space.shape
        action_space = game.action_space.n
        
        # 根据观测维度选择网络
        if len(obs_shape) == 3 and obs_shape[1] >= 4 and obs_shape[2] >= 4:
            # 3D 观测且足够大，使用卷积网络
            logger.info(f"使用 MuZeroNetwork (channels={self.config.num_channels})")
            return MuZeroNetwork(
                obs_shape=obs_shape,
                action_space=action_space,
                channels=self.config.num_channels,
                num_blocks=self.config.num_blocks,
            )
        else:
            # 低维观测，使用 MLP 网络
            logger.info(f"使用 SimpleMuZeroNetwork (hidden_dim=128)")
            return SimpleMuZeroNetwork(
                obs_shape=obs_shape,
                action_space=action_space,
                hidden_dim=128,
                num_layers=2,
            )
    
    def create_search(self, game: "Game", network: nn.Module):
        """创建 Gumbel 搜索
        
        MuZero 模式使用 dynamics network，无需 clone:
        - 支持 clone: 使用 GumbelMCTSSearch（官方实现）
        - 不支持 clone: 使用 GumbelSearch + dynamics（适用于 Gymnasium）
        """
        # 检测游戏是否支持 clone
        supports_clone = self._check_clone_support(game)
        
        if supports_clone:
            # 官方实现: Gumbel + MCTS 树搜索
            from core.mcts import LeafBatcher
            from core.config import BatcherConfig
            
            mcts_config = self.config.to_mcts_config()
            batcher_config = BatcherConfig(device=self.config.device)
            batcher = LeafBatcher(network, batcher_config)
            
            logger.info("Gumbel MuZero: 使用 GumbelMCTSSearch（官方实现）")
            return GumbelMCTSSearch(
                game=game,
                config=mcts_config,
                batcher=batcher,
                mode="muzero",
                max_considered_actions=self.config.max_num_considered_actions,
                gumbel_scale=self.config.gumbel_scale,
            )
        else:
            # 简化版: 使用 dynamics network（适用于 Gymnasium）
            gumbel_config = GumbelConfig(
                num_simulations=self.config.num_simulations,
                max_num_considered_actions=self.config.max_num_considered_actions,
                gumbel_scale=self.config.gumbel_scale,
                c_visit=self.config.c_visit,
                c_scale=self.config.c_scale,
                discount=self.config.discount,
                temperature=self.config.temperature_explore,
                device=self.config.device,
            )
            
            logger.info("Gumbel MuZero: 游戏不支持 clone，使用 GumbelSearch + dynamics")
            return GumbelSearch(
                network=network,
                config=gumbel_config,
                action_space=game.action_space.n,
                use_dynamics=True,  # 使用 dynamics network
            )
    
    def _check_clone_support(self, game: "Game") -> bool:
        """检测游戏是否支持 clone()"""
        try:
            cloned = game.clone()
            return cloned is not None
        except Exception:
            return False
    
    def compute_targets(
        self,
        trajectory: Trajectory,
        start_idx: int,
        unroll_steps: int
    ) -> Dict[str, np.ndarray]:
        """计算训练目标（与标准 MuZero 相同）"""
        T = len(trajectory)
        
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
                
                if k < unroll_steps and idx < T:
                    actions.append(trajectory.actions[idx])
                
                target_values.append(self._compute_target_value(trajectory, idx))
                target_rewards.append(trajectory.rewards[idx])
                
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
        """计算目标价值（使用 n-step return + bootstrap）"""
        # 简化版：使用 Monte Carlo 回报
        final_reward = trajectory.rewards[-1]
        to_play = trajectory.to_play[idx]
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
        return getattr(self, "_cached_action_space", 2086)
    
    def compute_loss(
        self,
        network: nn.Module,
        batch: TrainingTargets
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """计算损失函数（与标准 MuZero 相同）"""
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
            
            reward_loss = reward_loss + (step_mask * F.mse_loss(
                reward.squeeze(-1), batch.target_rewards[:, k + 1], reduction='none'
            )).sum()
            
            value_loss = value_loss + (step_mask * F.mse_loss(
                value.squeeze(-1), batch.target_values[:, k + 1], reduction='none'
            )).sum()
            
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


# ============================================================
# Gumbel AlphaZero 算法
# ============================================================

@register_algorithm("gumbel_alphazero")
class GumbelAlphaZeroAlgorithm(Algorithm):
    """Gumbel AlphaZero 算法
    
    使用 Gumbel-Top-k + Sequential Halving 替代传统 MCTS。
    需要真实游戏规则（但对 clone() 的依赖较少）。
    
    核心组件:
    - Representation Network: 观测 -> 特征
    - Prediction Network: 特征 -> (策略, 价值)
    """
    
    def __init__(self, config: GumbelAlphaZeroConfig = None):
        if config is None:
            config = GumbelAlphaZeroConfig()
        super().__init__(config)
        self.config: GumbelAlphaZeroConfig = config
    
    @property
    def needs_dynamics(self) -> bool:
        return False  # AlphaZero 不需要 dynamics
    
    @property
    def name(self) -> str:
        return "Gumbel AlphaZero"
    
    def create_network(self, game: "Game") -> nn.Module:
        """创建 AlphaZero 网络"""
        from algorithms.alphazero.network import AlphaZeroNetwork, SimpleAlphaZeroNetwork
        
        obs_shape = game.observation_space.shape
        action_space = game.action_space.n
        
        network_size = self.config.get_network_size(obs_shape, action_space)
        
        if network_size == "small":
            logger.info(f"使用 SimpleAlphaZeroNetwork (hidden_dim={self.config.hidden_dim})")
            return SimpleAlphaZeroNetwork(
                obs_shape=obs_shape,
                action_space=action_space,
                hidden_dim=self.config.hidden_dim,
                num_layers=2,
            )
        elif network_size == "medium":
            logger.info(f"使用 AlphaZeroNetwork (channels=64, blocks=3)")
            return AlphaZeroNetwork(
                obs_shape=obs_shape,
                action_space=action_space,
                channels=64,
                num_blocks=3,
            )
        else:
            logger.info(f"使用 AlphaZeroNetwork (channels={self.config.num_channels}, "
                       f"blocks={self.config.num_blocks})")
            return AlphaZeroNetwork(
                obs_shape=obs_shape,
                action_space=action_space,
                channels=self.config.num_channels,
                num_blocks=self.config.num_blocks,
            )
    
    def create_search(self, game: "Game", network: nn.Module):
        """创建 Gumbel 搜索
        
        自动检测游戏是否支持 clone():
        - 支持 clone: 使用 GumbelMCTSSearch（官方实现，更强）
        - 不支持 clone: 使用 GumbelSearch（简化版，用于 Gymnasium）
        """
        # 检测游戏是否支持 clone
        supports_clone = self._check_clone_support(game)
        
        if supports_clone:
            # 官方实现: Gumbel + MCTS 树搜索
            from core.mcts import LeafBatcher
            from core.config import BatcherConfig
            
            mcts_config = self.config.to_mcts_config()
            batcher_config = BatcherConfig(device=self.config.device)
            batcher = LeafBatcher(network, batcher_config)
            
            logger.info("Gumbel AlphaZero: 使用 GumbelMCTSSearch（官方实现）")
            return GumbelMCTSSearch(
                game=game,
                config=mcts_config,
                batcher=batcher,
                mode="alphazero",
                max_considered_actions=self.config.max_num_considered_actions,
                gumbel_scale=self.config.gumbel_scale,
            )
        else:
            # 简化版: 仅用网络评估（适用于 Gymnasium）
            gumbel_config = GumbelConfig(
                num_simulations=self.config.num_simulations,
                max_num_considered_actions=self.config.max_num_considered_actions,
                gumbel_scale=self.config.gumbel_scale,
                c_visit=self.config.c_visit,
                c_scale=self.config.c_scale,
                discount=self.config.discount,
                temperature=self.config.temperature_explore,
                device=self.config.device,
            )
            
            logger.warning("Gumbel AlphaZero: 游戏不支持 clone，使用 GumbelSearch（简化版）")
            return GumbelSearch(
                network=network,
                config=gumbel_config,
                action_space=game.action_space.n,
                use_dynamics=False,
            )
    
    def _check_clone_support(self, game: "Game") -> bool:
        """检测游戏是否支持 clone()"""
        try:
            cloned = game.clone()
            return cloned is not None
        except Exception:
            return False
    
    def compute_targets(
        self,
        trajectory: Trajectory,
        start_idx: int,
        unroll_steps: int = 0  # AlphaZero 不需要 unroll
    ) -> Dict[str, np.ndarray]:
        """计算训练目标（与标准 AlphaZero 相同）"""
        idx = start_idx
        
        obs = trajectory.observations[idx]
        target_value = self._compute_target_value(trajectory, idx)
        
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
        """计算损失函数（与标准 AlphaZero 相同）"""
        # 前向传播
        policy_logits, value = network(batch.observations)
        
        # Value loss
        value_loss = F.mse_loss(
            value.squeeze(-1),
            batch.target_values[:, 0]
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


# ============================================================
# 导出
# ============================================================

__all__ = [
    "GumbelMuZeroConfig",
    "GumbelAlphaZeroConfig",
    "GumbelMuZeroAlgorithm",
    "GumbelAlphaZeroAlgorithm",
]

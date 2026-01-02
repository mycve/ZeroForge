"""
ReplayBuffer - 经验回放缓冲区

存储自玩轨迹，支持采样训练数据。

特性:
- 优先经验回放（可选）
- 支持 MuZero 风格的 unroll 采样
- 线程安全
"""

import random
import threading
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

from .algorithm import Trajectory

logger = logging.getLogger(__name__)


# ============================================================
# 采样数据
# ============================================================

@dataclass
class SampleBatch:
    """训练采样批次
    
    Attributes:
        observations: 观测 [B, *obs_shape]
        actions: 动作序列 [B, unroll_steps]
        target_values: 目标价值 [B, unroll_steps+1]
        target_rewards: 目标奖励 [B, unroll_steps+1]
        target_policies: 目标策略 [B, unroll_steps+1, action_space]
        masks: 有效性掩码 [B, unroll_steps+1]
        weights: 重要性采样权重 [B]（优先回放用）
        indices: 样本索引（用于更新优先级）
    """
    observations: np.ndarray
    actions: np.ndarray
    target_values: np.ndarray
    target_rewards: np.ndarray
    target_policies: np.ndarray
    masks: np.ndarray
    weights: Optional[np.ndarray] = None
    indices: Optional[np.ndarray] = None


# ============================================================
# 经验回放缓冲区
# ============================================================

class ReplayBuffer:
    """经验回放缓冲区
    
    存储完整游戏轨迹，支持随机采样训练数据。
    
    Attributes:
        capacity: 最大存储轨迹数量
        trajectories: 轨迹列表
        
    Example:
        >>> buffer = ReplayBuffer(capacity=10000)
        >>> buffer.add(trajectory)
        >>> batch = buffer.sample(batch_size=256, unroll_steps=5)
    """
    
    def __init__(
        self,
        capacity: int = 10000,
        action_space_size: int = 9,  # 默认井字棋
    ):
        """初始化缓冲区
        
        Args:
            capacity: 最大轨迹数量
            action_space_size: 动作空间大小（用于策略数组）
        """
        self.capacity = capacity
        self.action_space_size = action_space_size
        
        self._trajectories: List[Trajectory] = []
        self._lock = threading.Lock()
        
        # 统计
        self._total_samples = 0
        self._total_trajectories_added = 0
    
    def add(self, trajectory: Trajectory) -> None:
        """添加轨迹
        
        Args:
            trajectory: 游戏轨迹
        """
        if len(trajectory) == 0:
            logger.warning("尝试添加空轨迹，已忽略")
            return
        
        with self._lock:
            self._trajectories.append(trajectory)
            self._total_trajectories_added += 1
            self._total_samples += len(trajectory)
            
            # 超出容量时移除旧轨迹
            while len(self._trajectories) > self.capacity:
                removed = self._trajectories.pop(0)
                self._total_samples -= len(removed)
    
    def add_batch(self, trajectories: List[Trajectory]) -> None:
        """批量添加轨迹
        
        Args:
            trajectories: 轨迹列表
        """
        for traj in trajectories:
            self.add(traj)
    
    def sample(
        self,
        batch_size: int,
        unroll_steps: int = 5,
    ) -> SampleBatch:
        """采样训练批次
        
        Args:
            batch_size: 批次大小
            unroll_steps: 展开步数（MuZero 用）
            
        Returns:
            SampleBatch: 训练数据
            
        Raises:
            RuntimeError: 缓冲区为空
        """
        with self._lock:
            if len(self._trajectories) == 0:
                raise RuntimeError("缓冲区为空，无法采样")
            
            observations = []
            actions_list = []
            target_values = []
            target_rewards = []
            target_policies = []
            masks = []
            
            for _ in range(batch_size):
                # 随机选择轨迹
                traj = random.choice(self._trajectories)
                
                # 随机选择起始位置
                max_start = max(0, len(traj) - 1)
                start_idx = random.randint(0, max_start)
                
                # 提取数据
                sample = self._extract_sample(traj, start_idx, unroll_steps)
                
                observations.append(sample["observation"])
                actions_list.append(sample["actions"])
                target_values.append(sample["target_values"])
                target_rewards.append(sample["target_rewards"])
                target_policies.append(sample["target_policies"])
                masks.append(sample["masks"])
            
            return SampleBatch(
                observations=np.stack(observations),
                actions=np.stack(actions_list),
                target_values=np.stack(target_values),
                target_rewards=np.stack(target_rewards),
                target_policies=np.stack(target_policies),
                masks=np.stack(masks),
            )
    
    def _extract_sample(
        self,
        trajectory: Trajectory,
        start_idx: int,
        unroll_steps: int,
    ) -> Dict[str, np.ndarray]:
        """从轨迹提取单个样本
        
        Args:
            trajectory: 轨迹
            start_idx: 起始索引
            unroll_steps: 展开步数
            
        Returns:
            样本字典
        """
        T = len(trajectory)
        
        # 观测
        obs = trajectory.observations[start_idx]
        
        # 构建展开数据
        actions = []
        target_values = []
        target_rewards = []
        target_policies = []
        masks = []
        
        for k in range(unroll_steps + 1):
            idx = start_idx + k
            
            if idx < T:
                masks.append(1.0)
                
                # 动作（最后一步不需要）
                if k < unroll_steps and idx < T:
                    actions.append(trajectory.actions[idx])
                elif k < unroll_steps:
                    actions.append(0)  # padding
                
                # 目标价值
                target_values.append(self._compute_target_value(trajectory, idx))
                
                # 目标奖励
                target_rewards.append(trajectory.rewards[idx])
                
                # 目标策略
                policy = trajectory.policies[idx]
                policy_array = self._policy_to_array(policy)
                target_policies.append(policy_array)
            else:
                masks.append(0.0)
                if k < unroll_steps:
                    actions.append(0)
                target_values.append(0.0)
                target_rewards.append(0.0)
                target_policies.append(np.zeros(self.action_space_size, dtype=np.float32))
        
        return {
            "observation": obs,
            "actions": np.array(actions, dtype=np.int64),
            "target_values": np.array(target_values, dtype=np.float32),
            "target_rewards": np.array(target_rewards, dtype=np.float32),
            "target_policies": np.stack(target_policies),
            "masks": np.array(masks, dtype=np.float32),
        }
    
    def _compute_target_value(self, trajectory: Trajectory, idx: int) -> float:
        """计算目标价值（蒙特卡洛回报）
        
        使用游戏最终结果作为目标价值，而非网络估计。
        对于零和博弈，从当前玩家视角计算。
        
        Args:
            trajectory: 游戏轨迹
            idx: 当前步骤索引
            
        Returns:
            目标价值（从当前玩家视角）
        """
        if idx >= len(trajectory):
            return 0.0
        
        # 获取当前步骤的玩家
        current_player = trajectory.to_play[idx] if idx < len(trajectory.to_play) else 0
        
        # 使用该步骤的奖励作为目标价值
        # rewards 已经在自玩结束时根据最终结果设置好了
        if idx < len(trajectory.rewards):
            return trajectory.rewards[idx]
        
        # 如果 rewards 未设置，尝试从最终结果推断
        if trajectory.rewards:
            final_reward = trajectory.rewards[-1]
            final_player = trajectory.to_play[-1] if trajectory.to_play else 0
            
            # 从当前玩家视角调整
            if current_player == final_player:
                return final_reward
            else:
                return -final_reward  # 零和博弈
        
        return 0.0
    
    def _policy_to_array(self, policy: Dict[int, float]) -> np.ndarray:
        """策略字典转数组"""
        arr = np.zeros(self.action_space_size, dtype=np.float32)
        for action, prob in policy.items():
            if 0 <= action < self.action_space_size:
                arr[action] = prob
        return arr
    
    def __len__(self) -> int:
        """当前轨迹数量"""
        return len(self._trajectories)
    
    @property
    def num_samples(self) -> int:
        """当前样本总数（所有轨迹的步数和）"""
        return self._total_samples
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        with self._lock:
            avg_length = (
                self._total_samples / len(self._trajectories)
                if self._trajectories else 0
            )
            return {
                "num_trajectories": len(self._trajectories),
                "total_samples": self._total_samples,
                "capacity": self.capacity,
                "avg_trajectory_length": avg_length,
                "total_added": self._total_trajectories_added,
            }
    
    def clear(self) -> None:
        """清空缓冲区"""
        with self._lock:
            self._trajectories.clear()
            self._total_samples = 0


# ============================================================
# 优先经验回放（可选）
# ============================================================

class PrioritizedReplayBuffer(ReplayBuffer):
    """优先经验回放缓冲区
    
    基于 TD-error 的优先采样。
    
    TODO: 实现 Sum Tree 优化采样效率
    """
    
    def __init__(
        self,
        capacity: int = 10000,
        action_space_size: int = 9,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 1e-5,
    ):
        super().__init__(capacity, action_space_size)
        
        self.alpha = alpha  # 优先级指数
        self.beta = beta    # 重要性采样指数
        self.beta_increment = beta_increment
        
        self._priorities: List[float] = []
        self._max_priority = 1.0
    
    def add(self, trajectory: Trajectory, priority: Optional[float] = None) -> None:
        """添加轨迹"""
        if len(trajectory) == 0:
            return
        
        with self._lock:
            self._trajectories.append(trajectory)
            
            # 设置优先级
            p = priority if priority is not None else self._max_priority
            self._priorities.append(p)
            
            self._total_trajectories_added += 1
            self._total_samples += len(trajectory)
            
            # 超出容量
            while len(self._trajectories) > self.capacity:
                self._trajectories.pop(0)
                self._priorities.pop(0)
    
    def sample(
        self,
        batch_size: int,
        unroll_steps: int = 5,
    ) -> SampleBatch:
        """优先采样"""
        with self._lock:
            if len(self._trajectories) == 0:
                raise RuntimeError("缓冲区为空")
            
            # 计算采样概率
            priorities = np.array(self._priorities) ** self.alpha
            probs = priorities / priorities.sum()
            
            # 采样索引
            indices = np.random.choice(
                len(self._trajectories),
                size=batch_size,
                p=probs,
                replace=True,
            )
            
            # 计算重要性采样权重
            weights = (len(self._trajectories) * probs[indices]) ** (-self.beta)
            weights = weights / weights.max()
            
            # 递增 beta
            self.beta = min(1.0, self.beta + self.beta_increment)
            
            # 提取数据
            observations = []
            actions_list = []
            target_values = []
            target_rewards = []
            target_policies = []
            masks = []
            
            for idx in indices:
                traj = self._trajectories[idx]
                max_start = max(0, len(traj) - 1)
                start_idx = random.randint(0, max_start)
                
                sample = self._extract_sample(traj, start_idx, unroll_steps)
                
                observations.append(sample["observation"])
                actions_list.append(sample["actions"])
                target_values.append(sample["target_values"])
                target_rewards.append(sample["target_rewards"])
                target_policies.append(sample["target_policies"])
                masks.append(sample["masks"])
            
            return SampleBatch(
                observations=np.stack(observations),
                actions=np.stack(actions_list),
                target_values=np.stack(target_values),
                target_rewards=np.stack(target_rewards),
                target_policies=np.stack(target_policies),
                masks=np.stack(masks),
                weights=weights.astype(np.float32),
                indices=indices,
            )
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """更新优先级
        
        Args:
            indices: 样本索引
            priorities: 新的优先级（通常是 TD-error）
        """
        with self._lock:
            for idx, p in zip(indices, priorities):
                if 0 <= idx < len(self._priorities):
                    self._priorities[idx] = float(p) + 1e-6  # 避免0
                    self._max_priority = max(self._max_priority, p)


# ============================================================
# 导出
# ============================================================

__all__ = [
    "SampleBatch",
    "ReplayBuffer",
    "PrioritizedReplayBuffer",
]

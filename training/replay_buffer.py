"""
Replay Buffer 模块
支持优先级采样和数据增强
"""

from __future__ import annotations
from typing import NamedTuple, Optional, Tuple, List
from dataclasses import dataclass, field
import numpy as np
import jax
import jax.numpy as jnp
from collections import deque
import threading
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# 数据结构
# ============================================================================

@dataclass
class Trajectory:
    """
    单局游戏的轨迹数据
    
    Attributes:
        observations: 观察序列 (T, C, H, W)
        actions: 动作序列 (T,)
        rewards: 奖励序列 (T,)
        policies: MCTS 策略序列 (T, A)
        values: 价值估计序列 (T,)
        to_plays: 每步的当前玩家 (T,)，0=红, 1=黑
        game_result: 游戏结果 (0=红胜, 1=黑胜, -1=平局)
        is_mirrored: 是否为镜像数据
    """
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    policies: np.ndarray
    values: np.ndarray
    to_plays: np.ndarray  # 每步的当前玩家
    game_result: int
    is_mirrored: bool = False
    
    def __len__(self):
        return len(self.actions)
    
    def get_target_values(self, discount: float = 0.997) -> np.ndarray:
        """
        计算每一步的目标价值 (基于游戏结果)
        
        对于棋类游戏，使用游戏最终结果作为所有步骤的目标价值。
        价值从当前玩家视角计算：
        - 当前玩家将要获胜: +1 * discount^remaining_steps
        - 当前玩家将要失败: -1 * discount^remaining_steps
        - 平局: 0
        """
        T = len(self.actions)
        target_values = np.zeros(T, dtype=np.float32)
        
        if self.game_result == -1:
            # 平局: 所有步骤价值为 0
            target_values[:] = 0.0
        else:
            # 有胜负
            # game_result: 0=红胜, 1=黑胜
            for t in range(T):
                # 直接使用存储的 to_play
                current_player = self.to_plays[t]
                
                # 剩余步数 (用于折扣)
                remaining_steps = T - t
                
                if current_player == self.game_result:
                    # 当前玩家最终获胜 -> 正价值
                    target_values[t] = discount ** remaining_steps
                else:
                    # 当前玩家最终失败 -> 负价值
                    target_values[t] = -(discount ** remaining_steps)
        
        return target_values


class SampleBatch(NamedTuple):
    """训练批次"""
    observations: jnp.ndarray      # (B, C, H, W)
    actions: jnp.ndarray           # (B, K) - K 步展开
    target_policies: jnp.ndarray   # (B, K, A)
    target_values: jnp.ndarray     # (B, K)
    target_rewards: jnp.ndarray    # (B, K)
    weights: jnp.ndarray           # (B,) - 重要性权重
    indices: jnp.ndarray           # (B,) - 用于更新优先级


# ============================================================================
# Sum Tree (用于优先级采样)
# ============================================================================

class SumTree:
    """
    Sum Tree 数据结构
    用于 O(log n) 的优先级采样
    """
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.data_pointer = 0
        self.n_entries = 0
    
    def _propagate(self, idx: int, change: float):
        """向上传播优先级变化"""
        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)
    
    def _retrieve(self, idx: int, s: float) -> int:
        """检索给定累积和对应的叶节点"""
        left = 2 * idx + 1
        right = left + 1
        
        if left >= len(self.tree):
            return idx
        
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])
    
    def total(self) -> float:
        """总优先级"""
        return self.tree[0]
    
    def add(self, priority: float) -> int:
        """添加新元素"""
        idx = self.data_pointer + self.capacity - 1
        
        self.update(idx, priority)
        
        self.data_pointer = (self.data_pointer + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
        
        return idx - self.capacity + 1  # 返回数据索引
    
    def update(self, tree_idx: int, priority: float):
        """更新优先级"""
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        self._propagate(tree_idx, change)
    
    def get(self, s: float) -> Tuple[int, float]:
        """获取给定累积和对应的数据索引和优先级"""
        idx = self._retrieve(0, s)
        data_idx = idx - self.capacity + 1
        return data_idx, self.tree[idx]
    
    def min_priority(self) -> float:
        """最小优先级 (用于计算重要性权重)"""
        return np.min(self.tree[-self.n_entries:])


# ============================================================================
# Replay Buffer
# ============================================================================

class ReplayBuffer:
    """
    基础 Replay Buffer
    使用均匀采样
    """
    
    def __init__(
        self,
        capacity: int = 100000,
        unroll_steps: int = 5,
        td_steps: int = 10,
    ):
        """
        Args:
            capacity: 最大存储的位置数
            unroll_steps: MuZero 展开步数
            td_steps: TD 目标计算步数
        """
        self.capacity = capacity
        self.unroll_steps = unroll_steps
        self.td_steps = td_steps
        
        # 存储
        self.trajectories: List[Trajectory] = []
        self.position_to_trajectory: List[Tuple[int, int]] = []  # (traj_idx, step_idx)
        
        self._lock = threading.Lock()
    
    def add(self, trajectory: Trajectory):
        """添加轨迹"""
        with self._lock:
            traj_idx = len(self.trajectories)
            self.trajectories.append(trajectory)
            
            # 添加位置索引
            for step_idx in range(len(trajectory)):
                if len(self.position_to_trajectory) >= self.capacity:
                    # 移除最老的位置
                    self.position_to_trajectory.pop(0)
                self.position_to_trajectory.append((traj_idx, step_idx))
    
    def sample(self, batch_size: int, key: jax.random.PRNGKey) -> SampleBatch:
        """均匀采样"""
        with self._lock:
            n = len(self.position_to_trajectory)
            if n == 0:
                raise ValueError("Buffer 为空，无法采样")
            
            # 随机选择位置
            indices = jax.random.randint(key, (batch_size,), 0, n)
            indices = np.array(indices)
            
            return self._make_batch(indices)
    
    def _make_batch(self, indices: np.ndarray) -> SampleBatch:
        """从位置索引创建批次"""
        batch_size = len(indices)
        
        # 获取第一个轨迹以确定形状
        first_traj = self.trajectories[self.position_to_trajectory[0][0]]
        obs_shape = first_traj.observations.shape[1:]  # (C, H, W)
        action_size = first_traj.policies.shape[1]
        
        # 预分配数组
        observations = np.zeros((batch_size,) + obs_shape, dtype=np.float32)
        actions = np.zeros((batch_size, self.unroll_steps), dtype=np.int32)
        target_policies = np.zeros((batch_size, self.unroll_steps, action_size), dtype=np.float32)
        target_values = np.zeros((batch_size, self.unroll_steps), dtype=np.float32)
        target_rewards = np.zeros((batch_size, self.unroll_steps), dtype=np.float32)
        
        for i, pos_idx in enumerate(indices):
            traj_idx, step_idx = self.position_to_trajectory[pos_idx]
            traj = self.trajectories[traj_idx]
            
            # 观察
            observations[i] = traj.observations[step_idx]
            
            # 展开 K 步
            traj_len = len(traj)
            game_values = traj.get_target_values()
            
            for k in range(self.unroll_steps):
                t = step_idx + k
                if t < traj_len:
                    actions[i, k] = traj.actions[t]
                    target_policies[i, k] = traj.policies[t]
                    target_values[i, k] = game_values[t]
                    target_rewards[i, k] = traj.rewards[t] if t > 0 else 0.0
                else:
                    # 超出轨迹，使用零填充
                    actions[i, k] = 0
                    target_policies[i, k] = np.ones(action_size) / action_size
                    target_values[i, k] = 0.0
                    target_rewards[i, k] = 0.0
        
        return SampleBatch(
            observations=jnp.array(observations),
            actions=jnp.array(actions),
            target_policies=jnp.array(target_policies),
            target_values=jnp.array(target_values),
            target_rewards=jnp.array(target_rewards),
            weights=jnp.ones(batch_size),
            indices=jnp.array(indices),
        )
    
    def __len__(self):
        return len(self.position_to_trajectory)
    
    def num_trajectories(self):
        return len(self.trajectories)


# ============================================================================
# Prioritized Replay Buffer
# ============================================================================

class PrioritizedReplayBuffer(ReplayBuffer):
    """
    优先级 Replay Buffer
    使用 TD 误差作为优先级
    """
    
    def __init__(
        self,
        capacity: int = 100000,
        unroll_steps: int = 5,
        td_steps: int = 10,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6,
    ):
        """
        Args:
            capacity: 最大存储位置数
            unroll_steps: 展开步数
            td_steps: TD 步数
            alpha: 优先级指数 (0 = 均匀, 1 = 完全优先)
            beta: 重要性采样指数
            beta_increment: beta 每步增量
            epsilon: 优先级下限
        """
        super().__init__(capacity, unroll_steps, td_steps)
        
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon
        
        self.sum_tree = SumTree(capacity)
        self.max_priority = 1.0
    
    def add(self, trajectory: Trajectory, priority: Optional[float] = None):
        """添加轨迹"""
        with self._lock:
            traj_idx = len(self.trajectories)
            self.trajectories.append(trajectory)
            
            # 为每个位置添加优先级
            for step_idx in range(len(trajectory)):
                if priority is None:
                    p = self.max_priority
                else:
                    p = priority
                
                p = (p + self.epsilon) ** self.alpha
                self.sum_tree.add(p)
                
                if len(self.position_to_trajectory) >= self.capacity:
                    self.position_to_trajectory.pop(0)
                self.position_to_trajectory.append((traj_idx, step_idx))
    
    def sample(self, batch_size: int, key: jax.random.PRNGKey) -> SampleBatch:
        """优先级采样"""
        with self._lock:
            n = self.sum_tree.n_entries
            if n == 0:
                raise ValueError("Buffer 为空，无法采样")
            
            indices = []
            priorities = []
            
            # 分层采样
            segment = self.sum_tree.total() / batch_size
            
            for i in range(batch_size):
                a = segment * i
                b = segment * (i + 1)
                
                # 在 [a, b) 范围内随机采样
                s = np.random.uniform(a, b)
                idx, priority = self.sum_tree.get(s)
                
                indices.append(idx)
                priorities.append(priority)
            
            indices = np.array(indices)
            priorities = np.array(priorities)
            
            # 计算重要性权重
            self.beta = min(1.0, self.beta + self.beta_increment)
            min_priority = self.sum_tree.min_priority()
            max_weight = (min_priority * n) ** (-self.beta)
            
            weights = (priorities * n) ** (-self.beta) / max_weight
            
            batch = self._make_batch(indices)
            
            return SampleBatch(
                observations=batch.observations,
                actions=batch.actions,
                target_policies=batch.target_policies,
                target_values=batch.target_values,
                target_rewards=batch.target_rewards,
                weights=jnp.array(weights, dtype=jnp.float32),
                indices=jnp.array(indices),
            )
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray):
        """更新优先级"""
        with self._lock:
            for idx, priority in zip(indices, priorities):
                p = (priority + self.epsilon) ** self.alpha
                tree_idx = idx + self.sum_tree.capacity - 1
                self.sum_tree.update(tree_idx, p)
                self.max_priority = max(self.max_priority, priority)


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    import jax
    import jax.numpy as jnp
    
    print("Replay Buffer 测试")
    print("=" * 50)
    
    # 创建测试轨迹
    def make_test_trajectory(length: int = 50):
        obs_shape = (240, 10, 9)
        action_size = 2086
        
        # 生成交替的 to_plays: 0, 1, 0, 1, ...
        to_plays = np.array([i % 2 for i in range(length)], dtype=np.int32)
        
        return Trajectory(
            observations=np.random.randn(length, *obs_shape).astype(np.float32),
            actions=np.random.randint(0, action_size, length),
            rewards=np.zeros(length, dtype=np.float32),
            policies=np.random.dirichlet(np.ones(action_size), length).astype(np.float32),
            values=np.random.uniform(-1, 1, length).astype(np.float32),
            to_plays=to_plays,
            game_result=0,
        )
    
    # 测试基础 Buffer
    buffer = ReplayBuffer(capacity=1000, unroll_steps=5)
    
    # 添加轨迹
    for _ in range(10):
        traj = make_test_trajectory(50)
        buffer.add(traj)
    
    print(f"Buffer 大小: {len(buffer)}")
    print(f"轨迹数量: {buffer.num_trajectories()}")
    
    # 采样
    key = jax.random.PRNGKey(42)
    batch = buffer.sample(32, key)
    
    print(f"\n采样批次:")
    print(f"  observations: {batch.observations.shape}")
    print(f"  actions: {batch.actions.shape}")
    print(f"  target_policies: {batch.target_policies.shape}")
    print(f"  target_values: {batch.target_values.shape}")
    print(f"  target_rewards: {batch.target_rewards.shape}")
    
    # 测试优先级 Buffer
    print("\n优先级 Buffer 测试:")
    p_buffer = PrioritizedReplayBuffer(capacity=1000, unroll_steps=5)
    
    for _ in range(10):
        traj = make_test_trajectory(50)
        p_buffer.add(traj)
    
    batch = p_buffer.sample(32, key)
    print(f"  weights 范围: [{batch.weights.min():.4f}, {batch.weights.max():.4f}]")
    
    # 更新优先级
    new_priorities = np.random.uniform(0.1, 2.0, 32)
    p_buffer.update_priorities(np.array(batch.indices), new_priorities)
    print("  优先级更新成功")

"""
输出头模块
策略头、价值头和奖励头
"""

from __future__ import annotations
from typing import Optional
import jax
import jax.numpy as jnp
import flax.linen as nn


class PolicyHead(nn.Module):
    """
    策略输出头
    
    将隐藏状态转换为动作概率分布 (logits)
    
    Attributes:
        action_space_size: 动作空间大小
        hidden_dim: 中间层维度
    """
    action_space_size: int
    hidden_dim: int = 256
    
    @nn.compact
    def __call__(self, x):
        """
        Args:
            x: 隐藏状态 (batch, channels, height, width)
            
        Returns:
            策略 logits (batch, action_space_size)
        """
        batch_size = x.shape[0]
        
        # 1x1 卷积降维
        x = nn.Conv(features=32, kernel_size=(1, 1), name='conv')(x)
        x = nn.gelu(x)
        
        # 展平
        x = x.reshape(batch_size, -1)
        
        # 全连接层
        x = nn.Dense(self.hidden_dim, name='fc1')(x)
        x = nn.gelu(x)
        x = nn.Dense(self.action_space_size, name='fc2')(x)
        
        return x


class ValueHead(nn.Module):
    """
    价值输出头
    
    将隐藏状态转换为状态价值
    
    Attributes:
        hidden_dim: 中间层维度
        support_size: 分类价值分布的支撑集大小 (MuZero 使用)
                      如果为 0，则输出标量价值
    """
    hidden_dim: int = 256
    support_size: int = 0  # 0 表示标量输出
    
    @nn.compact
    def __call__(self, x):
        """
        Args:
            x: 隐藏状态 (batch, channels, height, width)
            
        Returns:
            如果 support_size > 0: 价值分布 logits (batch, 2 * support_size + 1)
            否则: 标量价值 (batch,)
        """
        batch_size = x.shape[0]
        
        # 1x1 卷积降维
        x = nn.Conv(features=32, kernel_size=(1, 1), name='conv')(x)
        x = nn.gelu(x)
        
        # 展平
        x = x.reshape(batch_size, -1)
        
        # 全连接层
        x = nn.Dense(self.hidden_dim, name='fc1')(x)
        x = nn.gelu(x)
        
        if self.support_size > 0:
            # 分类价值分布
            num_bins = 2 * self.support_size + 1
            x = nn.Dense(num_bins, name='fc2')(x)
        else:
            # 标量价值
            x = nn.Dense(1, name='fc2')(x)
            x = jnp.tanh(x)  # 将价值限制在 [-1, 1]
            x = x.squeeze(-1)
        
        return x


class RewardHead(nn.Module):
    """
    奖励输出头
    
    预测执行动作后的即时奖励
    
    Attributes:
        hidden_dim: 中间层维度
        support_size: 分类奖励分布的支撑集大小
    """
    hidden_dim: int = 256
    support_size: int = 0
    
    @nn.compact
    def __call__(self, x):
        """
        Args:
            x: 隐藏状态 (batch, channels, height, width)
            
        Returns:
            如果 support_size > 0: 奖励分布 logits (batch, 2 * support_size + 1)
            否则: 标量奖励 (batch,)
        """
        batch_size = x.shape[0]
        
        # 1x1 卷积降维
        x = nn.Conv(features=32, kernel_size=(1, 1), name='conv')(x)
        x = nn.gelu(x)
        
        # 展平
        x = x.reshape(batch_size, -1)
        
        # 全连接层
        x = nn.Dense(self.hidden_dim, name='fc1')(x)
        x = nn.gelu(x)
        
        if self.support_size > 0:
            num_bins = 2 * self.support_size + 1
            x = nn.Dense(num_bins, name='fc2')(x)
        else:
            x = nn.Dense(1, name='fc2')(x)
            x = jnp.tanh(x)
            x = x.squeeze(-1)
        
        return x


# ============================================================================
# 辅助函数：分类价值转换
# ============================================================================

def scalar_to_support(x: jnp.ndarray, support_size: int) -> jnp.ndarray:
    """
    将标量值转换为分类分布 (soft label)
    
    用于训练分类价值/奖励头
    
    Args:
        x: 标量值 (batch,)
        support_size: 支撑集大小
        
    Returns:
        分类分布 (batch, 2 * support_size + 1)
    """
    # 支撑集: [-support_size, ..., 0, ..., support_size]
    support = jnp.arange(-support_size, support_size + 1, dtype=jnp.float32)
    
    # 缩放到支撑集范围
    x_clipped = jnp.clip(x, -support_size, support_size)
    
    # 计算相邻两个支撑点的权重 (线性插值)
    floor = jnp.floor(x_clipped).astype(jnp.int32)
    ceil = floor + 1
    
    # 权重
    ceil_weight = x_clipped - floor.astype(jnp.float32)
    floor_weight = 1.0 - ceil_weight
    
    # 构建分布
    batch_size = x.shape[0]
    num_bins = 2 * support_size + 1
    
    # 索引偏移
    floor_idx = floor + support_size
    ceil_idx = jnp.clip(ceil + support_size, 0, num_bins - 1)
    
    # 创建分布
    probs = jnp.zeros((batch_size, num_bins))
    batch_idx = jnp.arange(batch_size)
    
    probs = probs.at[batch_idx, floor_idx].add(floor_weight)
    probs = probs.at[batch_idx, ceil_idx].add(ceil_weight)
    
    return probs


def support_to_scalar(probs: jnp.ndarray, support_size: int) -> jnp.ndarray:
    """
    将分类分布转换为标量值 (期望)
    
    Args:
        probs: 分类分布 (batch, 2 * support_size + 1)
        support_size: 支撑集大小
        
    Returns:
        标量值 (batch,)
    """
    support = jnp.arange(-support_size, support_size + 1, dtype=jnp.float32)
    return jnp.sum(probs * support, axis=-1)


def logits_to_scalar(logits: jnp.ndarray, support_size: int) -> jnp.ndarray:
    """
    将分类分布 logits 转换为标量值
    
    Args:
        logits: 分类分布 logits (batch, 2 * support_size + 1)
        support_size: 支撑集大小
        
    Returns:
        标量值 (batch,)
    """
    probs = jax.nn.softmax(logits, axis=-1)
    return support_to_scalar(probs, support_size)


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    import jax
    import jax.numpy as jnp
    
    batch_size = 4
    channels = 256
    height, width = 10, 9
    action_space_size = 2086
    
    # 创建测试输入 (NCHW 格式)
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (batch_size, channels, height, width))
    
    # 测试策略头
    policy_head = PolicyHead(action_space_size=action_space_size)
    policy_params = policy_head.init(key, x)
    policy_logits = policy_head.apply(policy_params, x)
    print(f"策略头输出形状: {policy_logits.shape}")
    
    # 测试价值头 (标量)
    value_head = ValueHead(support_size=0)
    value_params = value_head.init(key, x)
    values = value_head.apply(value_params, x)
    print(f"价值头输出形状 (标量): {values.shape}")
    
    # 测试价值头 (分类)
    value_head_cat = ValueHead(support_size=50)
    value_params_cat = value_head_cat.init(key, x)
    value_logits = value_head_cat.apply(value_params_cat, x)
    print(f"价值头输出形状 (分类): {value_logits.shape}")
    
    # 测试分类转换
    target_values = jnp.array([0.5, -0.3, 0.8, -0.9])
    support = scalar_to_support(target_values * 50, support_size=50)
    recovered = support_to_scalar(support, support_size=50) / 50
    print(f"原始值: {target_values}")
    print(f"恢复值: {recovered}")

"""
AlphaZero 网络 - 轻量化高效架构
目标：快速推理 → 更多 MCTS 搜索 → 更强棋力
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
import optax


class ResBlock(nn.Module):
    """轻量残差块：Pre-LN + 3x3 Conv + ReLU"""
    channels: int
    dtype: jnp.dtype = jnp.float32
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        y = nn.LayerNorm(dtype=self.dtype)(x)
        y = nn.Conv(self.channels, (3, 3), padding='SAME', use_bias=False, dtype=self.dtype)(y)
        y = nn.relu(y)
        
        y = nn.LayerNorm(dtype=self.dtype)(y)
        y = nn.Conv(self.channels, (3, 3), padding='SAME', use_bias=False, dtype=self.dtype)(y)
        
        return x + y


class AlphaZeroNetwork(nn.Module):
    """AlphaZero 网络 - 轻量化架构
    
    设计原则：
    - 简单的残差块（无 SE、无混合卷积）
    - ReLU 激活（比 GELU 快）
    - 较少的通道数和层数
    - 推理速度优先 → 更多 MCTS 模拟
    """
    action_space_size: int = 2086
    channels: int = 64      # 轻量化：128 → 64
    num_blocks: int = 6     # 轻量化：8 → 6
    dtype: jnp.dtype = jnp.float32
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        # 强制将输入转换为指定 dtype
        x = x.astype(self.dtype)
        
        # NCHW -> NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))
        batch_size = x.shape[0]
        
        # === Stem: 3x3 卷积（5x5 对小棋盘意义不大）===
        x = nn.Conv(self.channels, (3, 3), padding='SAME', use_bias=False, dtype=self.dtype)(x)
        x = nn.relu(x)
        
        # === 主干：简洁残差塔 ===
        for _ in range(self.num_blocks):
            x = ResBlock(self.channels, dtype=self.dtype)(x, train)
        
        x = nn.LayerNorm(dtype=self.dtype)(x)
        
        # === 策略头（简化）===
        p = nn.Conv(32, (1, 1), dtype=self.dtype)(x)
        p = nn.relu(p)
        p = p.reshape((batch_size, -1))  # [B, 32*10*9] = [B, 2880]
        p = nn.Dense(self.action_space_size, dtype=self.dtype)(p)
        
        # === 价值头（简化）===
        v = jnp.mean(x, axis=(1, 2))  # 全局池化 [B, C]
        v = nn.Dense(128, dtype=self.dtype)(v)  # 256 → 128
        v = nn.relu(v)
        v = nn.Dense(1, dtype=self.dtype)(v)
        v = jnp.tanh(v).squeeze(-1)
        
        # 确保输出回退到 float32 以保证搜索精度
        return p.astype(jnp.float32), v.astype(jnp.float32)


def create_train_state(key, network, input_shape, learning_rate=2e-4):
    """创建训练状态 (无 BatchNorm，纯 LayerNorm 架构)"""
    x = jnp.zeros(input_shape)
    variables = network.init(key, x, train=True)
    
    tx = optax.adam(learning_rate)
    
    return TrainState.create(
        apply_fn=network.apply,
        params=variables['params'],
        tx=tx,
    )

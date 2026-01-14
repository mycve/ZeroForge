"""
AlphaZero 网络 - 现代高效架构
融合 KataGo + Lc0 + EfficientZero 最佳实践
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
import optax


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 块 (通道注意力)"""
    channels: int

    @nn.compact
    def __call__(self, x):
        w = jnp.mean(x, axis=(1, 2))  # [B, C]
        w = nn.Dense(self.channels // 4)(w)
        w = nn.relu(w)
        w = nn.Dense(self.channels)(w)
        w = nn.sigmoid(w)[:, None, None, :]
        return x * w


class MixedConvBlock(nn.Module):
    """混合卷积块：3x3 + 5x5 并行，捕捉多尺度特征
    
    对于象棋这种需要全局视野的游戏，5x5 可以更快捕捉长距离依赖
    """
    channels: int
    
    @nn.compact
    def __call__(self, x):
        # 3x3 分支 (局部特征)
        c3 = nn.Conv(self.channels // 2, (3, 3), padding='SAME', use_bias=False)(x)
        # 5x5 分支 (更大感受野)
        c5 = nn.Conv(self.channels // 2, (5, 5), padding='SAME', use_bias=False)(x)
        return jnp.concatenate([c3, c5], axis=-1)


class ResBlock(nn.Module):
    """残差块：Pre-LN + 混合卷积 + SE
    
    - Pre-LayerNorm: 训练更稳定
    - 混合卷积: 多尺度感受野
    - SE: 通道注意力
    """
    channels: int
    use_mixed_conv: bool = False  # 是否使用混合卷积
    use_se: bool = True
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        y = nn.LayerNorm()(x)
        
        if self.use_mixed_conv:
            y = MixedConvBlock(self.channels)(y)
        else:
            y = nn.Conv(self.channels, (3, 3), padding='SAME', use_bias=False)(y)
        y = nn.gelu(y)
        
        y = nn.LayerNorm()(y)
        y = nn.Conv(self.channels, (3, 3), padding='SAME', use_bias=False)(y)
        
        if self.use_se:
            y = SEBlock(self.channels)(y)
        
        return x + y


class AlphaZeroNetwork(nn.Module):
    """AlphaZero 网络 - 现代架构
    
    特性：
    1. Pre-LN ResNet: 训练更稳定
    2. 混合卷积 (3x3 + 5x5): 多尺度感受野，适合象棋全局依赖
    3. SE 注意力: 动态通道加权
    4. Global Pooling Policy Head: 融合全局信息
    """
    action_space_size: int = 2086
    channels: int = 128
    num_blocks: int = 10
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        # NCHW -> NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))
        batch_size = x.shape[0]
        
        # === Stem: 5x5 卷积快速扩大初始感受野 ===
        x = nn.Conv(self.channels, (5, 5), padding='SAME', use_bias=False)(x)
        x = nn.gelu(x)
        
        # === 主干：残差塔 ===
        for i in range(self.num_blocks):
            # 前 1/3 层用混合卷积扩大感受野，后面用普通 3x3 节省计算
            use_mixed = (i < self.num_blocks // 3)
            use_se = (i % 2 == 1)  # 隔层加 SE
            x = ResBlock(self.channels, use_mixed_conv=use_mixed, use_se=use_se)(x, train)
        
        x = nn.LayerNorm()(x)
        
        # === 策略头 (局部 + 全局) ===
        p_local = nn.Conv(32, (1, 1))(x)
        p_local = nn.gelu(p_local)
        p_local = p_local.reshape((batch_size, -1))
        
        p_global = jnp.mean(x, axis=(1, 2))
        p_global = nn.Dense(64)(p_global)
        p_global = nn.gelu(p_global)
        
        p = nn.Dense(self.action_space_size)(jnp.concatenate([p_local, p_global], axis=-1))
        
        # === 价值头 ===
        v = jnp.mean(x, axis=(1, 2))
        v = nn.Dense(256)(v)
        v = nn.gelu(v)
        v = nn.Dense(1)(v)
        v = jnp.tanh(v).squeeze(-1)
        
        return p, v


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

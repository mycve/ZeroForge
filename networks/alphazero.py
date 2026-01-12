"""
AlphaZero 网络
参考: https://github.com/zjjMaiMai/GumbelAlphaZero/blob/master/model.py
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.training.train_state import TrainState
import optax


class SEBlock(nn.Module):
    """Squeeze-and-Excitation 块 (通道注意力机制)
    这是现代顶级 AI (如 KataGo, Lc0) 的标准配置。
    它能让网络学会关注棋盘上的关键区域。
    """
    channels: int

    @nn.compact
    def __call__(self, x):
        # Squeeze: 全局平均池化
        w = jnp.mean(x, axis=(1, 2), keepdims=True)
        # Excitation: 两层全连接
        w = nn.Conv(self.channels // 4, (1, 1))(w)
        w = nn.relu(w)
        w = nn.Conv(self.channels, (1, 1))(w)
        w = nn.sigmoid(w)
        return x * w


class ResBlock(nn.Module):
    """增强残差块 (ResNet + SE)"""
    channels: int
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        y = nn.Conv(self.channels, (3, 3), padding='SAME', use_bias=False)(x)
        y = nn.BatchNorm(use_running_average=not train)(y)
        y = nn.relu(y)
        y = nn.Conv(self.channels, (3, 3), padding='SAME', use_bias=False)(y)
        y = nn.BatchNorm(use_running_average=not train)(y)
        # 加入 SE 注意力机制
        y = SEBlock(self.channels)(y)
        return nn.relu(x + y)


class AlphaZeroNetwork(nn.Module):
    """AlphaZero 网络"""
    action_space_size: int = 2086
    channels: int = 128
    num_blocks: int = 10
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        # NCHW -> NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))
        
        # Stem
        x = nn.Conv(self.channels, (3, 3), padding='SAME', use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = nn.relu(x)
        
        # 残差块
        for _ in range(self.num_blocks):
            x = ResBlock(self.channels)(x, train)
        
        # 策略头 (增强型：使用 3x3 卷积提取局部特征)
        p = nn.Conv(32, (3, 3), padding='SAME', use_bias=False)(x)
        p = nn.BatchNorm(use_running_average=not train)(p)
        p = nn.relu(p)
        p = p.reshape((p.shape[0], -1))
        p = nn.Dense(self.action_space_size)(p)
        
        # 价值头 (AlphaZero 标配通常是 1 通道)
        v = nn.Conv(1, (1, 1), use_bias=False)(x)
        v = nn.BatchNorm(use_running_average=not train)(v)
        v = nn.relu(v)
        v = v.reshape((v.shape[0], -1))  # 显式使用 tuple 形状
        v = nn.Dense(256)(v)
        v = nn.relu(v)
        v = nn.Dense(1)(v)
        v = jnp.tanh(v).squeeze(-1)
        
        return p, v


class TrainStateWithBN(TrainState):
    """训练状态（含 BatchNorm）"""
    batch_stats: dict


def create_train_state(key, network, input_shape, learning_rate=2e-4):
    """创建训练状态"""
    x = jnp.zeros(input_shape)
    variables = network.init(key, x, train=True)
    
    tx = optax.adam(learning_rate)
    
    return TrainStateWithBN.create(
        apply_fn=network.apply,
        params=variables['params'],
        tx=tx,
        batch_stats=variables['batch_stats'],
    )

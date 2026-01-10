"""
ConvNeXt 网络模块
现代化的卷积网络架构，用于棋盘状态编码
"""

from __future__ import annotations
from typing import Sequence, Optional
import jax
import jax.numpy as jnp
import flax.linen as nn
from functools import partial


class DropPath(nn.Module):
    """
    Stochastic Depth (Drop Path) 正则化
    在训练时随机丢弃整个残差分支
    """
    drop_prob: float = 0.0
    deterministic: Optional[bool] = None
    
    @nn.compact
    def __call__(self, x, deterministic: Optional[bool] = None):
        deterministic = nn.merge_param('deterministic', self.deterministic, deterministic)
        
        if deterministic or self.drop_prob == 0.0:
            return x
        
        keep_prob = 1 - self.drop_prob
        # 生成随机掩码
        rng = self.make_rng('dropout')
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = jax.random.bernoulli(rng, keep_prob, shape)
        
        return x * random_tensor / keep_prob


class ConvNeXtBlock(nn.Module):
    """
    ConvNeXt Block
    
    结构:
    1. Depthwise Conv (7x7)
    2. LayerNorm
    3. Linear (expand 4x)
    4. GELU
    5. Linear (reduce)
    6. Layer Scale
    7. Residual
    
    Attributes:
        dim: 特征维度
        layer_scale_init: Layer Scale 初始值
        drop_path: Drop Path 概率
    """
    dim: int
    layer_scale_init: float = 1e-6
    drop_path: float = 0.0
    
    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        """
        Args:
            x: 输入张量 (batch, height, width, channels)
            deterministic: 是否为推理模式
            
        Returns:
            输出张量，形状与输入相同
        """
        residual = x
        
        # Depthwise convolution (7x7)
        # 对于 10x9 的棋盘，使用 7x7 可以捕获大范围的空间关系
        x = nn.Conv(
            features=self.dim,
            kernel_size=(7, 7),
            padding='SAME',
            feature_group_count=self.dim,  # Depthwise
            name='dwconv'
        )(x)
        
        # LayerNorm (在 channel 维度)
        x = nn.LayerNorm(epsilon=1e-6, name='norm')(x)
        
        # Pointwise expansion (4x)
        x = nn.Dense(4 * self.dim, name='pwconv1')(x)
        
        # GELU 激活
        x = nn.gelu(x)
        
        # Pointwise reduction
        x = nn.Dense(self.dim, name='pwconv2')(x)
        
        # Layer Scale
        gamma = self.param(
            'gamma',
            nn.initializers.constant(self.layer_scale_init),
            (self.dim,)
        )
        x = gamma * x
        
        # Drop Path (训练时)
        x = DropPath(drop_prob=self.drop_path)(x, deterministic=deterministic)
        
        # Residual
        x = residual + x
        
        return x


class ConvNeXtStage(nn.Module):
    """
    ConvNeXt Stage (一组 blocks)
    
    Attributes:
        dim: 特征维度
        depth: block 数量
        layer_scale_init: Layer Scale 初始值
        drop_path_rates: 每个 block 的 drop path 概率
    """
    dim: int
    depth: int
    layer_scale_init: float = 1e-6
    drop_path_rates: Sequence[float] = ()
    
    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        drop_rates = self.drop_path_rates or [0.0] * self.depth
        
        for i in range(self.depth):
            x = ConvNeXtBlock(
                dim=self.dim,
                layer_scale_init=self.layer_scale_init,
                drop_path=drop_rates[i],
                name=f'block_{i}'
            )(x, deterministic=deterministic)
        
        return x


class ConvNeXtBackbone(nn.Module):
    """
    ConvNeXt 骨干网络
    
    针对中国象棋棋盘 (10x9) 优化:
    - 保持空间分辨率 (不下采样)
    - 使用较大的感受野捕获长距离依赖
    
    Attributes:
        in_channels: 输入通道数
        hidden_dim: 隐藏层维度
        depths: 每个 stage 的 block 数量
        layer_scale_init: Layer Scale 初始值
        drop_path_rate: 最大 drop path 概率
    """
    in_channels: int
    hidden_dim: int = 256
    depths: Sequence[int] = (8,)  # 单个 stage，8 个 blocks
    layer_scale_init: float = 1e-6
    drop_path_rate: float = 0.1
    
    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        """
        Args:
            x: 输入张量 (batch, channels, height, width) - 注意是 NCHW 格式
            deterministic: 是否为推理模式
            
        Returns:
            隐藏状态 (batch, hidden_dim, height, width)
        """
        # 转换为 NHWC 格式 (Flax 默认)
        x = jnp.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
        
        # Stem: 初始卷积层
        x = nn.Conv(
            features=self.hidden_dim,
            kernel_size=(3, 3),
            padding='SAME',
            name='stem_conv'
        )(x)
        x = nn.LayerNorm(epsilon=1e-6, name='stem_norm')(x)
        
        # 计算 drop path rates (线性增加)
        total_blocks = sum(self.depths)
        dp_rates = jnp.linspace(0, self.drop_path_rate, total_blocks)
        
        # Stages
        cur = 0
        for i, depth in enumerate(self.depths):
            stage_drop_rates = dp_rates[cur:cur + depth].tolist()
            x = ConvNeXtStage(
                dim=self.hidden_dim,
                depth=depth,
                layer_scale_init=self.layer_scale_init,
                drop_path_rates=stage_drop_rates,
                name=f'stage_{i}'
            )(x, deterministic=deterministic)
            cur += depth
        
        # 转换回 NCHW 格式
        x = jnp.transpose(x, (0, 3, 1, 2))  # NHWC -> NCHW
        
        return x


class ConvNeXtEncoder(nn.Module):
    """
    编码器：将观察转换为隐藏状态
    
    用于 MuZero 的 Representation Network
    """
    in_channels: int
    hidden_dim: int = 256
    num_blocks: int = 8
    
    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        return ConvNeXtBackbone(
            in_channels=self.in_channels,
            hidden_dim=self.hidden_dim,
            depths=(self.num_blocks,),
            name='backbone'
        )(x, deterministic=deterministic)


class ConvNeXtDecoder(nn.Module):
    """
    解码器：将隐藏状态转换为输出
    
    用于 MuZero 的 Dynamics 和 Prediction Networks
    """
    hidden_dim: int = 256
    num_blocks: int = 4
    
    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        return ConvNeXtBackbone(
            in_channels=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            depths=(self.num_blocks,),
            name='backbone'
        )(x, deterministic=deterministic)


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    import jax
    import jax.numpy as jnp
    
    # 测试参数
    batch_size = 4
    in_channels = 240  # 观察通道数
    hidden_dim = 256
    height, width = 10, 9  # 棋盘尺寸
    
    # 创建模型
    model = ConvNeXtBackbone(
        in_channels=in_channels,
        hidden_dim=hidden_dim,
        depths=(8,)
    )
    
    # 初始化
    key = jax.random.PRNGKey(42)
    x = jax.random.normal(key, (batch_size, in_channels, height, width))
    
    params = model.init(key, x)
    
    # 前向传播
    y = model.apply(params, x)
    
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")
    print(f"参数数量: {sum(p.size for p in jax.tree_util.tree_leaves(params))}")

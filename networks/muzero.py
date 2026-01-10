"""
MuZero 网络模块
包含 Representation、Dynamics 和 Prediction 三个网络
"""

from __future__ import annotations
from typing import NamedTuple, Optional, Tuple
import jax
import jax.numpy as jnp
import flax.linen as nn
from functools import partial

from networks.convnext import ConvNeXtBackbone, ConvNeXtBlock
from networks.heads import PolicyHead, ValueHead, RewardHead


# ============================================================================
# 网络输出类型
# ============================================================================

class NetworkOutput(NamedTuple):
    """网络输出"""
    hidden_state: jnp.ndarray  # (batch, hidden_dim, H, W)
    policy_logits: jnp.ndarray  # (batch, action_space_size)
    value: jnp.ndarray  # (batch,) 或 (batch, num_bins)


class DynamicsOutput(NamedTuple):
    """Dynamics 网络输出"""
    next_hidden_state: jnp.ndarray  # (batch, hidden_dim, H, W)
    reward: jnp.ndarray  # (batch,) 或 (batch, num_bins)


# ============================================================================
# Representation Network
# ============================================================================

class RepresentationNetwork(nn.Module):
    """
    表示网络
    将观察转换为隐藏状态
    
    观察 -> 隐藏状态
    
    Attributes:
        hidden_dim: 隐藏层维度
        num_blocks: ConvNeXt blocks 数量
    """
    hidden_dim: int = 256
    num_blocks: int = 8
    
    @nn.compact
    def __call__(self, observation: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """
        Args:
            observation: 观察张量 (batch, channels, height, width)
            deterministic: 是否为推理模式
            
        Returns:
            隐藏状态 (batch, hidden_dim, height, width)
        """
        in_channels = observation.shape[1]
        
        x = ConvNeXtBackbone(
            in_channels=in_channels,
            hidden_dim=self.hidden_dim,
            depths=(self.num_blocks,),
            name='backbone'
        )(observation, deterministic=deterministic)
        
        # 归一化隐藏状态 (有助于训练稳定性)
        x = _normalize_hidden_state(x)
        
        return x


# ============================================================================
# Dynamics Network
# ============================================================================

class DynamicsNetwork(nn.Module):
    """
    动态网络
    预测下一个隐藏状态和即时奖励
    
    (隐藏状态, 动作) -> (下一隐藏状态, 奖励)
    
    Attributes:
        hidden_dim: 隐藏层维度
        num_blocks: ConvNeXt blocks 数量
        action_space_size: 动作空间大小
        reward_support_size: 奖励分类分布支撑集大小
    """
    hidden_dim: int = 256
    num_blocks: int = 4
    action_space_size: int = 2086
    reward_support_size: int = 0  # 0 表示标量奖励
    
    @nn.compact
    def __call__(
        self,
        hidden_state: jnp.ndarray,
        action: jnp.ndarray,
        deterministic: bool = True
    ) -> DynamicsOutput:
        """
        Args:
            hidden_state: 当前隐藏状态 (batch, hidden_dim, height, width)
            action: 动作索引 (batch,)
            deterministic: 是否为推理模式
            
        Returns:
            DynamicsOutput (next_hidden_state, reward)
        """
        batch_size = hidden_state.shape[0]
        height, width = hidden_state.shape[2], hidden_state.shape[3]
        
        # 将动作编码为空间平面
        # 使用 one-hot 编码，然后通过全连接层映射到空间
        action_onehot = jax.nn.one_hot(action, self.action_space_size)
        
        # 动作嵌入
        action_embed = nn.Dense(self.hidden_dim, name='action_embed')(action_onehot)
        
        # 扩展到空间维度
        action_plane = action_embed[:, :, jnp.newaxis, jnp.newaxis]
        action_plane = jnp.broadcast_to(
            action_plane,
            (batch_size, self.hidden_dim, height, width)
        )
        
        # 连接隐藏状态和动作
        x = jnp.concatenate([hidden_state, action_plane], axis=1)
        
        # 转换为 NHWC
        x = jnp.transpose(x, (0, 2, 3, 1))
        
        # 降维 (2 * hidden_dim -> hidden_dim)
        x = nn.Conv(features=self.hidden_dim, kernel_size=(1, 1), name='reduce_conv')(x)
        x = nn.LayerNorm(epsilon=1e-6, name='reduce_norm')(x)
        
        # 转换回 NCHW
        x = jnp.transpose(x, (0, 3, 1, 2))
        
        # ConvNeXt blocks
        x = ConvNeXtBackbone(
            in_channels=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            depths=(self.num_blocks,),
            name='backbone'
        )(x, deterministic=deterministic)
        
        # 归一化
        next_hidden = _normalize_hidden_state(x)
        
        # 预测奖励
        reward = RewardHead(
            support_size=self.reward_support_size,
            name='reward_head'
        )(x)
        
        return DynamicsOutput(
            next_hidden_state=next_hidden,
            reward=reward
        )


# ============================================================================
# Prediction Network
# ============================================================================

class PredictionNetwork(nn.Module):
    """
    预测网络
    从隐藏状态预测策略和价值
    
    隐藏状态 -> (策略, 价值)
    
    Attributes:
        hidden_dim: 隐藏层维度
        num_blocks: ConvNeXt blocks 数量
        action_space_size: 动作空间大小
        value_support_size: 价值分类分布支撑集大小
    """
    hidden_dim: int = 256
    num_blocks: int = 4
    action_space_size: int = 2086
    value_support_size: int = 0  # 0 表示标量价值
    
    @nn.compact
    def __call__(
        self,
        hidden_state: jnp.ndarray,
        deterministic: bool = True
    ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """
        Args:
            hidden_state: 隐藏状态 (batch, hidden_dim, height, width)
            deterministic: 是否为推理模式
            
        Returns:
            (policy_logits, value)
            - policy_logits: (batch, action_space_size)
            - value: (batch,) 或 (batch, num_bins)
        """
        # ConvNeXt blocks
        x = ConvNeXtBackbone(
            in_channels=self.hidden_dim,
            hidden_dim=self.hidden_dim,
            depths=(self.num_blocks,),
            name='backbone'
        )(hidden_state, deterministic=deterministic)
        
        # 策略头
        policy_logits = PolicyHead(
            action_space_size=self.action_space_size,
            hidden_dim=self.hidden_dim,
            name='policy_head'
        )(x)
        
        # 价值头
        value = ValueHead(
            support_size=self.value_support_size,
            hidden_dim=self.hidden_dim,
            name='value_head'
        )(x)
        
        return policy_logits, value


# ============================================================================
# 完整 MuZero 网络
# ============================================================================

class MuZeroNetwork(nn.Module):
    """
    完整的 MuZero 网络
    
    包含:
    - Representation: 观察 -> 隐藏状态
    - Dynamics: (隐藏状态, 动作) -> (下一隐藏状态, 奖励)
    - Prediction: 隐藏状态 -> (策略, 价值)
    
    Attributes:
        observation_channels: 观察通道数
        hidden_dim: 隐藏层维度
        action_space_size: 动作空间大小
        repr_blocks: Representation 网络 blocks 数量
        dyn_blocks: Dynamics 网络 blocks 数量
        pred_blocks: Prediction 网络 blocks 数量
        value_support_size: 价值分类分布支撑集大小
        reward_support_size: 奖励分类分布支撑集大小
    """
    observation_channels: int = 240
    hidden_dim: int = 256
    action_space_size: int = 2086
    repr_blocks: int = 8
    dyn_blocks: int = 4
    pred_blocks: int = 4
    value_support_size: int = 0
    reward_support_size: int = 0
    
    def setup(self):
        self.representation = RepresentationNetwork(
            hidden_dim=self.hidden_dim,
            num_blocks=self.repr_blocks,
            name='representation'
        )
        
        self.dynamics = DynamicsNetwork(
            hidden_dim=self.hidden_dim,
            num_blocks=self.dyn_blocks,
            action_space_size=self.action_space_size,
            reward_support_size=self.reward_support_size,
            name='dynamics'
        )
        
        self.prediction = PredictionNetwork(
            hidden_dim=self.hidden_dim,
            num_blocks=self.pred_blocks,
            action_space_size=self.action_space_size,
            value_support_size=self.value_support_size,
            name='prediction'
        )
    
    def __call__(
        self,
        observation: jnp.ndarray,
        deterministic: bool = True
    ) -> NetworkOutput:
        """
        初始推理：观察 -> 隐藏状态 -> 策略/价值
        
        Args:
            observation: 观察张量 (batch, channels, height, width)
            deterministic: 是否为推理模式
            
        Returns:
            NetworkOutput
        """
        hidden_state = self.representation(observation, deterministic)
        policy_logits, value = self.prediction(hidden_state, deterministic)
        
        return NetworkOutput(
            hidden_state=hidden_state,
            policy_logits=policy_logits,
            value=value
        )
    
    def represent(
        self,
        observation: jnp.ndarray,
        deterministic: bool = True
    ) -> jnp.ndarray:
        """仅执行表示网络"""
        return self.representation(observation, deterministic)
    
    def recurrent_inference(
        self,
        hidden_state: jnp.ndarray,
        action: jnp.ndarray,
        deterministic: bool = True
    ) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        循环推理：(隐藏状态, 动作) -> (下一隐藏状态, 奖励, 策略, 价值)
        
        用于 MCTS 的树展开
        
        Args:
            hidden_state: 当前隐藏状态
            action: 动作索引
            deterministic: 是否为推理模式
            
        Returns:
            (next_hidden_state, reward, policy_logits, value)
        """
        dyn_output = self.dynamics(hidden_state, action, deterministic)
        policy_logits, value = self.prediction(dyn_output.next_hidden_state, deterministic)
        
        return (
            dyn_output.next_hidden_state,
            dyn_output.reward,
            policy_logits,
            value
        )


# ============================================================================
# 辅助函数
# ============================================================================

def _normalize_hidden_state(x: jnp.ndarray) -> jnp.ndarray:
    """
    归一化隐藏状态
    
    使用 min-max 缩放到 [0, 1] 范围
    有助于 Dynamics 网络的稳定训练
    """
    # 计算每个样本的最小/最大值
    batch_size = x.shape[0]
    x_flat = x.reshape(batch_size, -1)
    
    min_val = jnp.min(x_flat, axis=1, keepdims=True)
    max_val = jnp.max(x_flat, axis=1, keepdims=True)
    
    # 避免除零
    scale = jnp.maximum(max_val - min_val, 1e-8)
    
    x_normalized = (x_flat - min_val) / scale
    
    return x_normalized.reshape(x.shape)


def create_muzero_network(config: dict) -> MuZeroNetwork:
    """
    根据配置创建 MuZero 网络
    
    Args:
        config: 网络配置字典
        
    Returns:
        MuZeroNetwork 实例
    """
    return MuZeroNetwork(
        observation_channels=config.get('observation_channels', 240),
        hidden_dim=config.get('hidden_dim', 256),
        action_space_size=config.get('action_space_size', 2086),
        repr_blocks=config.get('repr_blocks', 8),
        dyn_blocks=config.get('dyn_blocks', 4),
        pred_blocks=config.get('pred_blocks', 4),
        value_support_size=config.get('value_support_size', 0),
        reward_support_size=config.get('reward_support_size', 0),
    )


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    import jax
    import jax.numpy as jnp
    
    # 测试配置
    batch_size = 4
    observation_channels = 240
    hidden_dim = 256
    action_space_size = 2086
    height, width = 10, 9
    
    # 创建网络
    network = MuZeroNetwork(
        observation_channels=observation_channels,
        hidden_dim=hidden_dim,
        action_space_size=action_space_size,
        repr_blocks=8,
        dyn_blocks=4,
        pred_blocks=4,
    )
    
    # 初始化
    key = jax.random.PRNGKey(42)
    observation = jax.random.normal(key, (batch_size, observation_channels, height, width))
    
    params = network.init(key, observation)
    
    # 初始推理
    output = network.apply(params, observation)
    print(f"隐藏状态形状: {output.hidden_state.shape}")
    print(f"策略 logits 形状: {output.policy_logits.shape}")
    print(f"价值形状: {output.value.shape}")
    
    # 循环推理
    action = jnp.array([0, 1, 2, 3])
    next_hidden, reward, policy, value = network.apply(
        params,
        output.hidden_state,
        action,
        method=network.recurrent_inference
    )
    print(f"\n循环推理:")
    print(f"下一隐藏状态形状: {next_hidden.shape}")
    print(f"奖励形状: {reward.shape}")
    print(f"策略形状: {policy.shape}")
    print(f"价值形状: {value.shape}")
    
    # 参数统计
    param_count = sum(p.size for p in jax.tree_util.tree_leaves(params))
    print(f"\n总参数数量: {param_count:,}")


# ============================================================================
# TrainState 创建
# ============================================================================

from flax.training.train_state import TrainState
import optax


def create_train_state(
    rng_key: jax.random.PRNGKey,
    network: MuZeroNetwork,
    input_shape: tuple,
    learning_rate: float = 3e-4,
) -> TrainState:
    """
    创建训练状态
    
    Args:
        rng_key: 随机数密钥
        network: MuZero 网络
        input_shape: 输入形状 (batch, channels, height, width)
        learning_rate: 学习率
        
    Returns:
        TrainState
    """
    # 初始化参数
    dummy_input = jnp.zeros(input_shape)
    
    # 需要初始化所有子网络 (representation + prediction + dynamics)
    k1, k2 = jax.random.split(rng_key)
    params = network.init(k1, dummy_input)
    
    # 初始化 dynamics (通过 recurrent_inference)
    hidden_dim = network.hidden_dim
    batch_size = input_shape[0]
    dummy_hidden = jnp.zeros((batch_size, hidden_dim, 10, 9))
    dummy_action = jnp.zeros((batch_size,), dtype=jnp.int32)
    dyn_params = network.init(k2, dummy_hidden, dummy_action, method=network.recurrent_inference)
    
    # 合并参数
    def merge_params(p1, p2):
        import copy
        result = copy.deepcopy(p1)
        def merge(d1, d2):
            for k, v in d2.items():
                if k not in d1:
                    d1[k] = v
                elif isinstance(v, dict):
                    merge(d1[k], v)
        merge(result, p2)
        return result
    
    params = merge_params(params, dyn_params)
    
    # 优化器
    optimizer = optax.adam(learning_rate)
    
    return TrainState.create(
        apply_fn=network.apply,
        params=params,
        tx=optimizer,
    )

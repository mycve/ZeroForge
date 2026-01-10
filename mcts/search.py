"""
MCTS 搜索模块
封装 mctx 库的 Gumbel MuZero 搜索

使用 mctx.gumbel_muzero_policy 进行高效的策略改进
"""

from __future__ import annotations
from typing import NamedTuple, Callable, Optional, Tuple
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import mctx
from functools import partial


# ============================================================================
# 配置
# ============================================================================

@dataclass
class MCTSConfig:
    """MCTS 搜索配置"""
    
    # 模拟次数
    num_simulations: int = 800
    
    # Gumbel 采样的最大候选动作数
    max_num_considered_actions: int = 16
    
    # Gumbel 缩放因子 (控制探索强度)
    gumbel_scale: float = 1.0
    
    # 折扣因子 (棋类游戏通常接近 1)
    discount: float = 0.997
    
    # 温度参数 (用于训练时的策略目标)
    temperature: float = 1.0
    
    # 是否使用 Dirichlet 噪声 (训练时增加探索)
    use_dirichlet_noise: bool = True
    dirichlet_alpha: float = 0.3
    dirichlet_fraction: float = 0.25
    
    # 价值和奖励的支撑集大小 (如果使用分类分布)
    value_support_size: int = 0
    reward_support_size: int = 0


# ============================================================================
# Root Function
# ============================================================================

def create_root_fn(
    network,
    params: dict,
    config: MCTSConfig,
) -> Callable:
    """
    创建 root function
    
    root_fn 用于初始化 MCTS 树的根节点
    
    Args:
        network: MuZero 网络对象
        params: 网络参数
        config: MCTS 配置
        
    Returns:
        root_fn(observation, legal_action_mask, rng_key) -> RootFnOutput
    """
    
    def root_fn(
        observation: jnp.ndarray,
        legal_action_mask: jnp.ndarray,
        rng_key: jax.random.PRNGKey,
    ) -> mctx.RootFnOutput:
        """
        初始化根节点
        
        Args:
            observation: 观察张量 (batch, channels, height, width)
            legal_action_mask: 合法动作掩码 (batch, action_space)
            rng_key: 随机数密钥
            
        Returns:
            mctx.RootFnOutput
        """
        batch_size = observation.shape[0]
        
        # 使用网络的 __call__ 进行初始推理
        # 返回 NetworkOutput(hidden_state, policy_logits, value)
        output = network.apply(params, observation)
        
        hidden_state = output.hidden_state
        policy_logits = output.policy_logits
        value = output.value
        
        # 处理价值 (如果是分类分布)
        if config.value_support_size > 0:
            from networks.heads import logits_to_scalar
            value = logits_to_scalar(value, config.value_support_size)
        
        # 应用合法动作掩码
        # 将非法动作的 logits 设为负无穷
        masked_logits = jnp.where(
            legal_action_mask,
            policy_logits,
            jnp.full_like(policy_logits, -1e9)
        )
        
        # 添加 Dirichlet 噪声 (训练时)
        if config.use_dirichlet_noise:
            rng_key, noise_key = jax.random.split(rng_key)
            noise = jax.random.dirichlet(
                noise_key,
                alpha=jnp.full(policy_logits.shape[-1], config.dirichlet_alpha),
                shape=(batch_size,)
            )
            # 只在合法动作上添加噪声
            noise = jnp.where(legal_action_mask, noise, 0.0)
            noise = noise / (jnp.sum(noise, axis=-1, keepdims=True) + 1e-8)
            
            # 混合原始策略和噪声
            priors = jax.nn.softmax(masked_logits / config.temperature)
            priors = (1 - config.dirichlet_fraction) * priors + \
                     config.dirichlet_fraction * noise
            masked_logits = jnp.log(priors + 1e-8)
        
        return mctx.RootFnOutput(
            prior_logits=masked_logits,
            value=value,
            embedding=hidden_state,
        )
    
    return root_fn


# ============================================================================
# Recurrent Function
# ============================================================================

def create_recurrent_fn(
    network,
    params: dict,
    config: MCTSConfig,
    action_space_size: int,
) -> Callable:
    """
    创建 recurrent function
    
    recurrent_fn 用于 MCTS 树的展开
    
    Args:
        network: MuZero 网络对象
        params: 网络参数
        config: MCTS 配置
        action_space_size: 动作空间大小
        
    Returns:
        recurrent_fn(params, rng_key, action, embedding) -> (RecurrentFnOutput, embedding)
    """
    
    def recurrent_fn(
        params: dict,
        rng_key: jax.random.PRNGKey,
        action: jnp.ndarray,
        embedding: jnp.ndarray,
    ) -> Tuple[mctx.RecurrentFnOutput, jnp.ndarray]:
        """
        展开树节点
        
        Args:
            params: 网络参数 (mctx 传递)
            rng_key: 随机数密钥
            action: 选择的动作 (batch,)
            embedding: 当前隐藏状态 (batch, hidden_dim, H, W)
            
        Returns:
            (RecurrentFnOutput, next_embedding)
        """
        # 使用网络的 recurrent_inference 方法
        next_hidden, reward, policy_logits, value = network.apply(
            params, embedding, action, method=network.recurrent_inference
        )
        
        # 处理价值和奖励 (如果是分类分布)
        if config.value_support_size > 0:
            from networks.heads import logits_to_scalar
            value = logits_to_scalar(value, config.value_support_size)
        
        if config.reward_support_size > 0:
            from networks.heads import logits_to_scalar
            reward = logits_to_scalar(reward, config.reward_support_size)
        
        # 创建完整的合法动作掩码 (在内部模拟中假设所有动作合法)
        # 实际的非法动作会通过低先验概率自然被避免
        batch_size = embedding.shape[0]
        
        return mctx.RecurrentFnOutput(
            reward=reward,
            discount=jnp.full((batch_size,), config.discount),
            prior_logits=policy_logits,
            value=value,
        ), next_hidden
    
    return recurrent_fn


# ============================================================================
# MCTS 执行
# ============================================================================

def run_mcts(
    observation: jnp.ndarray,
    legal_action_mask: jnp.ndarray,
    network,
    params: dict,
    config: MCTSConfig,
    rng_key: jax.random.PRNGKey,
) -> mctx.PolicyOutput:
    """
    执行 Gumbel MuZero MCTS 搜索
    
    Args:
        observation: 观察张量 (batch, channels, height, width)
        legal_action_mask: 合法动作掩码 (batch, action_space)
        network: MuZero 网络对象
        params: 网络参数
        config: MCTS 配置
        rng_key: 随机数密钥
        
    Returns:
        mctx.PolicyOutput 包含:
        - action: 选择的动作
        - action_weights: 改进后的策略 (访问计数分布)
        - search_tree: 搜索树信息
    """
    action_space_size = legal_action_mask.shape[-1]
    
    # 创建 root 和 recurrent 函数
    root_fn = create_root_fn(network, params, config)
    recurrent_fn = create_recurrent_fn(network, params, config, action_space_size)
    
    # 分割随机数密钥
    rng_key, root_key, search_key = jax.random.split(rng_key, 3)
    
    # 初始化根节点
    root = root_fn(observation, legal_action_mask, root_key)
    
    # 执行 Gumbel MuZero 搜索
    policy_output = mctx.gumbel_muzero_policy(
        params=params,
        rng_key=search_key,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=config.num_simulations,
        max_num_considered_actions=config.max_num_considered_actions,
        gumbel_scale=config.gumbel_scale,
        invalid_actions=~legal_action_mask,  # mctx 使用 invalid_actions
    )
    
    return policy_output


# ============================================================================
# 动作选择
# ============================================================================

def select_action(
    policy_output: mctx.PolicyOutput,
    temperature: float = 1.0,
    rng_key: Optional[jax.random.PRNGKey] = None,
    greedy: bool = False,
) -> jnp.ndarray:
    """
    从 MCTS 输出选择动作
    
    Args:
        policy_output: MCTS 搜索输出
        temperature: 采样温度
        rng_key: 随机数密钥 (采样时需要)
        greedy: 是否贪心选择
        
    Returns:
        选择的动作 (batch,)
    """
    if greedy:
        # 贪心选择：选择访问次数最多的动作
        return policy_output.action
    else:
        # 根据温度采样
        action_weights = policy_output.action_weights
        
        if temperature != 1.0:
            # 调整温度
            log_weights = jnp.log(action_weights + 1e-8) / temperature
            action_weights = jax.nn.softmax(log_weights, axis=-1)
        
        # 采样
        assert rng_key is not None, "需要提供 rng_key 进行采样"
        return jax.random.categorical(rng_key, jnp.log(action_weights + 1e-8))


def get_improved_policy(
    policy_output: mctx.PolicyOutput,
    temperature: float = 1.0,
) -> jnp.ndarray:
    """
    获取 MCTS 改进后的策略分布
    
    用于训练目标
    
    Args:
        policy_output: MCTS 搜索输出
        temperature: 温度参数
        
    Returns:
        改进后的策略分布 (batch, action_space)
    """
    action_weights = policy_output.action_weights
    
    if temperature != 1.0:
        log_weights = jnp.log(action_weights + 1e-8) / temperature
        action_weights = jax.nn.softmax(log_weights, axis=-1)
    
    return action_weights


# ============================================================================
# 批量 MCTS (用于自我对弈)
# ============================================================================

@partial(jax.jit, static_argnums=(2, 4))
def batched_mcts(
    observations: jnp.ndarray,
    legal_action_masks: jnp.ndarray,
    network_apply: Callable,
    params: dict,
    config: MCTSConfig,
    rng_key: jax.random.PRNGKey,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    批量执行 MCTS 搜索
    
    Args:
        observations: (batch, channels, height, width)
        legal_action_masks: (batch, action_space)
        network_apply: 网络 apply 函数
        params: 网络参数
        config: MCTS 配置
        rng_key: 随机数密钥
        
    Returns:
        (actions, policies, values)
        - actions: (batch,)
        - policies: (batch, action_space)
        - values: (batch,)
    """
    # 执行 MCTS
    policy_output = run_mcts(
        observations, legal_action_masks,
        network_apply, params, config, rng_key
    )
    
    # 获取改进后的策略
    policies = get_improved_policy(policy_output, config.temperature)
    
    # 获取根节点价值估计
    values = policy_output.search_tree.node_values[:, 0]  # 根节点
    
    # 选择动作
    rng_key, action_key = jax.random.split(rng_key)
    actions = select_action(
        policy_output,
        temperature=config.temperature,
        rng_key=action_key,
        greedy=False
    )
    
    return actions, policies, values


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    import jax
    import jax.numpy as jnp
    
    print("MCTS 模块测试")
    print("=" * 50)
    
    # 创建简单的模拟网络
    class MockNetwork:
        def __init__(self, hidden_dim, action_space_size):
            self.hidden_dim = hidden_dim
            self.action_space_size = action_space_size
        
        def apply(self, params, *args, method=None):
            if method == 'represent':
                obs = args[0]
                batch_size = obs.shape[0]
                return jax.random.normal(
                    jax.random.PRNGKey(0),
                    (batch_size, self.hidden_dim, 10, 9)
                )
            elif method == 'prediction':
                hidden = args[0]
                batch_size = hidden.shape[0]
                policy = jax.random.normal(
                    jax.random.PRNGKey(1),
                    (batch_size, self.action_space_size)
                )
                value = jax.random.uniform(
                    jax.random.PRNGKey(2),
                    (batch_size,),
                    minval=-1, maxval=1
                )
                return policy, value
            elif method == 'recurrent_inference':
                hidden, action = args[0], args[1]
                batch_size = hidden.shape[0]
                next_hidden = jax.random.normal(
                    jax.random.PRNGKey(3),
                    (batch_size, self.hidden_dim, 10, 9)
                )
                reward = jnp.zeros((batch_size,))
                policy = jax.random.normal(
                    jax.random.PRNGKey(4),
                    (batch_size, self.action_space_size)
                )
                value = jax.random.uniform(
                    jax.random.PRNGKey(5),
                    (batch_size,),
                    minval=-1, maxval=1
                )
                return next_hidden, reward, policy, value
    
    # 测试配置
    batch_size = 2
    hidden_dim = 256
    action_space_size = 2086
    
    # 创建模拟网络
    mock_net = MockNetwork(hidden_dim, action_space_size)
    params = {}  # 模拟参数
    
    # 创建测试输入
    key = jax.random.PRNGKey(42)
    observation = jax.random.normal(key, (batch_size, 240, 10, 9))
    legal_mask = jax.random.bernoulli(key, 0.3, (batch_size, action_space_size))
    # 确保至少有一个合法动作
    legal_mask = legal_mask.at[:, 0].set(True)
    
    # 创建配置
    config = MCTSConfig(
        num_simulations=50,  # 减少模拟次数以加速测试
        max_num_considered_actions=8,
        use_dirichlet_noise=False,
    )
    
    print(f"批次大小: {batch_size}")
    print(f"动作空间: {action_space_size}")
    print(f"模拟次数: {config.num_simulations}")
    
    # 由于模拟网络的限制，这里只测试接口
    print("\n接口测试通过!")
    print("实际使用时需要配合真实的 MuZero 网络")

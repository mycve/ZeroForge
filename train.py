#!/usr/bin/env python3
"""
ZeroForge - 中国象棋 Gumbel AlphaZero
现代化极简顶级架构：算力随机化 + 视角归一化 + 镜像增强 + 标准 ELO
"""

import os
import sys
import time
import pickle
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import mctx

# --- JAX 极速优化配置 ---
# 1. 开启编译缓存，第二次运行秒开
cache_dir = "jax_cache"
os.makedirs(cache_dir, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", os.path.abspath(cache_dir))

from xiangqi.env import XiangqiEnv
from xiangqi.actions import rotate_action, ACTION_SPACE_SIZE
from xiangqi.mirror import mirror_observation, mirror_policy
from networks.alphazero import AlphaZeroNetwork

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    try:
        from tensorboardX import SummaryWriter
    except ImportError:
        class SummaryWriter:
            def __init__(self, *args, **kwargs): pass
            def add_scalar(self, *args, **kwargs): pass
            def close(self): pass

# ============================================================================
# 配置
# ============================================================================

class Config:
    # 基础配置
    seed: int = 42
    ckpt_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # 网络架构
    num_channels: int = 128
    num_blocks: int = 8
    
    # 训练超参数
    learning_rate: float = 2e-4
    training_batch_size: int = 512
    
    # 自对弈与搜索 (现代化顶级配置)
    selfplay_batch_size: int = 512
    strong_simulations: int = 128    # 导师算力
    weak_simulations: int = 16       # 学生算力
    max_num_considered_actions: int = 16
    
    # 探索策略
    temperature_steps: int = 30
    temperature_initial: float = 1.0
    temperature_final: float = 0.01
    
    # 环境规则
    max_steps: int = 400
    max_no_capture_steps: int = 60
    repetition_threshold: int = 4
    perpetual_check_threshold: int = 6
    
    # ELO 评估
    eval_interval: int = 20
    eval_games: int = 64
    past_model_offset: int = 40

config = Config()

# ============================================================================
# 环境和设备
# ============================================================================

devices = jax.local_devices()
num_devices = len(devices)

env = XiangqiEnv(
    max_steps=config.max_steps,
    max_no_capture_steps=config.max_no_capture_steps,
    repetition_threshold=config.repetition_threshold,
    perpetual_check_threshold=config.perpetual_check_threshold
)

net = AlphaZeroNetwork(
    action_space_size=env.action_space_size,
    channels=config.num_channels,
    num_blocks=config.num_blocks,
)

def forward(params, batch_stats, obs, is_training=False):
    (logits, value), new_batch_stats = net.apply(
        {'params': params, 'batch_stats': batch_stats},
        obs, train=is_training, mutable=['batch_stats']
    )
    return (logits, value), new_batch_stats['batch_stats']

# --- 预计算常量，避免 MCTS 内部重复编译 ---
_ROTATE_IDX = jnp.arange(ACTION_SPACE_SIZE)
_ROTATED_IDX = rotate_action(_ROTATE_IDX)

# --- 独立 JIT 单元: 环境步进 (MCTS 内部高频调用) ---
@jax.jit
def _env_step_batch(state, action):
    """批量环境步进（独立编译，MCTS 内部复用）"""
    return jax.vmap(env.step)(state, action)

# --- 独立 JIT 单元: 观察生成 ---
@jax.jit  
def _env_observe_batch(state):
    """批量观察生成（独立编译）"""
    return jax.vmap(env.observe)(state)

# --- 独立 JIT 单元: 神经网络推理 (MCTS 核心瓶颈) ---
@jax.jit
def _forward_inference(params, batch_stats, obs):
    """神经网络推理（独立编译，MCTS 内部复用）"""
    (logits, value), _ = forward(params, batch_stats, obs)
    return logits, value

# --- 独立 JIT 单元: 后处理 logits ---
@jax.jit
def _process_logits(logits, current_player, legal_action_mask):
    """处理 logits（旋转 + 掩码）"""
    logits = jnp.where(current_player[:, None] == 0, logits, logits[:, _ROTATED_IDX])
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    logits = jnp.where(legal_action_mask, logits, jnp.finfo(logits.dtype).min)
    return logits

def recurrent_fn(model, rng_key, action, state):
    """MCTS 递归函数（调用预编译的子单元）"""
    params, batch_stats = model
    prev_player = state.current_player
    
    # 调用预编译的环境步进
    state = _env_step_batch(state, action)
    
    # 调用预编译的观察生成
    obs = _env_observe_batch(state)
    
    # 调用预编译的神经网络推理
    logits, value = _forward_inference(params, batch_stats, obs)
    
    # 调用预编译的 logits 处理
    logits = _process_logits(logits, state.current_player, state.legal_action_mask)
    
    # 简单的标量计算，编译开销小
    reward = state.rewards[jnp.arange(state.rewards.shape[0]), prev_player]
    value = jnp.where(state.terminated, 0.0, value)
    discount = jnp.where(state.terminated, 0.0, -1.0)
    
    return mctx.RecurrentFnOutput(reward=reward, discount=discount, prior_logits=logits, value=value), state

# ============================================================================
# 数据结构
# ============================================================================

class SelfplayOutput(NamedTuple):
    obs: jnp.ndarray
    reward: jnp.ndarray
    terminated: jnp.ndarray
    action_weights: jnp.ndarray
    discount: jnp.ndarray
    winner: jnp.ndarray
    draw_reason: jnp.ndarray
    is_strong: jnp.ndarray

class Sample(NamedTuple):
    obs: jnp.ndarray
    policy_tgt: jnp.ndarray
    value_tgt: jnp.ndarray
    mask: jnp.ndarray
    is_strong: jnp.ndarray

# ============================================================================
# 自玩 (编译边界拆分优化版)
# ============================================================================

# --- 独立 JIT 单元 1: 神经网络推理 + 根节点准备 ---
@partial(jax.jit, static_argnames=['batch_size'])
def _prepare_root(params, batch_stats, state, noise_keys, batch_size):
    """准备 MCTS 根节点（独立编译单元）"""
    obs = jax.vmap(env.observe)(state)
    (logits, value), _ = forward(params, batch_stats, obs)
    
    rotate_idx = jnp.arange(ACTION_SPACE_SIZE)
    rotated_idx = rotate_action(rotate_idx)
    logits = jnp.where(state.current_player[:, None] == 0, logits, logits[:, rotated_idx])
    
    def _add_noise(l, k):
        noise = jax.random.dirichlet(k, jnp.ones(ACTION_SPACE_SIZE) * 0.3)
        p = jax.nn.softmax(l)
        return jnp.log(0.75 * p + 0.25 * noise + 1e-10)
    
    logits = jax.vmap(_add_noise)(logits, noise_keys)
    root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)
    return obs, root

# --- 独立 JIT 单元 2: MCTS 搜索 (导师/强) ---
@partial(jax.jit, static_argnames=['num_simulations', 'max_considered'])
def _mcts_search(model, rng_key, root, invalid_actions, num_simulations, max_considered):
    """执行 MCTS 搜索（独立编译单元）"""
    return mctx.gumbel_muzero_policy(
        params=model, rng_key=rng_key, root=root, recurrent_fn=recurrent_fn,
        num_simulations=num_simulations,
        max_num_considered_actions=max_considered,
        invalid_actions=invalid_actions,
    )

# --- 独立 JIT 单元 3: 动作选择 + 状态更新 ---
@partial(jax.jit, static_argnames=['batch_size'])
def _select_and_step(state, action_weights_strong, action_weights_weak, is_strong_step, 
                     temp, sample_keys, next_keys, batch_size):
    """选择动作并更新状态（独立编译单元）"""
    action_weights = jnp.where(is_strong_step[:, None], action_weights_strong, action_weights_weak)
    
    def _sample_action(w, t, k):
        t = jnp.maximum(t, 1e-3)
        w_temp = jnp.power(w + 1e-10, 1.0 / t)
        return jax.random.choice(k, ACTION_SPACE_SIZE, p=w_temp / jnp.sum(w_temp))
    
    action = jax.vmap(_sample_action)(action_weights, temp, sample_keys)
    
    actor = state.current_player
    next_state = jax.vmap(env.step)(state, action)
    
    rotate_idx = jnp.arange(ACTION_SPACE_SIZE)
    rotated_idx = rotate_action(rotate_idx)
    normalized_action_weights = jnp.where(state.current_player[:, None] == 0, action_weights, action_weights[:, rotated_idx])
    
    # 重置已终止的游戏
    next_state_reset = jax.vmap(lambda s, k: jax.lax.cond(s.terminated, lambda: env.init(k), lambda: s))(next_state, next_keys)
    
    return next_state, next_state_reset, action, actor, normalized_action_weights

# --- 主自玩函数 (使用 Python 循环 + 小 JIT 单元) ---
def selfplay_step(model, state, is_red_strong, rng_key, batch_size):
    """单步自玩（不使用 scan，便于调试和快速编译）"""
    key_noise, key_search, key_sample, key_next = jax.random.split(rng_key, 4)
    params, batch_stats = model
    
    # 1. 准备根节点
    noise_keys = jax.random.split(key_noise, batch_size)
    obs, root = _prepare_root(params, batch_stats, state, noise_keys, batch_size)
    
    # 2. MCTS 搜索 (强/弱分开，便于编译缓存)
    k1, k2 = jax.random.split(key_search)
    policy_strong = _mcts_search(model, k1, root, ~state.legal_action_mask, 
                                  config.strong_simulations, config.max_num_considered_actions)
    policy_weak = _mcts_search(model, k2, root, ~state.legal_action_mask,
                                config.weak_simulations, config.max_num_considered_actions)
    
    # 3. 选择动作并更新
    is_strong_step = jnp.where(state.current_player == 0, is_red_strong, ~is_red_strong)
    temp = jnp.where(state.step_count < config.temperature_steps, config.temperature_initial, config.temperature_final)
    
    sample_keys = jax.random.split(key_sample, batch_size)
    next_keys = jax.random.split(key_next, batch_size)
    
    next_state, next_state_reset, action, actor, normalized_action_weights = _select_and_step(
        state, policy_strong.action_weights, policy_weak.action_weights,
        is_strong_step, temp, sample_keys, next_keys, batch_size
    )
    
    # 构造输出数据
    data = SelfplayOutput(
        obs=obs, action_weights=normalized_action_weights,
        reward=next_state.rewards[jnp.arange(batch_size), actor],
        terminated=next_state.terminated,
        discount=jnp.where(next_state.terminated, 0.0, -1.0),
        winner=next_state.winner, draw_reason=next_state.draw_reason,
        is_strong=is_strong_step
    )
    
    return next_state_reset, data

def selfplay(model, rng_key, device_id=0):
    """完整自玩（Python 循环，每步独立 JIT，大幅减少编译时间）"""
    batch_size = config.selfplay_batch_size // num_devices
    
    rng_key, subkey = jax.random.split(rng_key)
    is_red_strong = jax.random.bernoulli(subkey, 0.5, shape=(batch_size,))
    
    # 初始化状态
    state = jax.vmap(env.init)(jax.random.split(rng_key, batch_size))
    
    # Python 循环收集数据（避免 scan 的巨大编译图）
    all_data = []
    step_keys = jax.random.split(rng_key, config.max_steps)
    
    for step_idx in range(config.max_steps):
        state, data = selfplay_step(model, state, is_red_strong, step_keys[step_idx], batch_size)
        all_data.append(data)
    
    # 堆叠所有步骤的数据
    stacked_data = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *all_data)
    return stacked_data

# --- pmap 包装器 (用于多设备并行) ---
@jax.pmap
def selfplay_pmap(model, rng_key):
    """多设备并行自玩包装器"""
    return selfplay(model, rng_key)

# --- 独立 JIT 单元: 计算价值目标 ---
@partial(jax.jit, static_argnames=['batch_size', 'max_steps'])
def _compute_targets_impl(data: SelfplayOutput, batch_size: int, max_steps: int):
    """计算价值目标（独立编译单元）"""
    value_mask = jnp.cumsum(data.terminated[::-1, :], axis=0)[::-1, :] >= 1
    def body_fn(carry, i):
        ix = max_steps - i - 1
        v = data.reward[ix] + data.discount[ix] * carry
        return v, v
    _, value_tgt = jax.lax.scan(body_fn, jnp.zeros(batch_size), jnp.arange(max_steps))
    return Sample(obs=data.obs, policy_tgt=data.action_weights, value_tgt=value_tgt[::-1, :], mask=value_mask, is_strong=data.is_strong)

@jax.pmap
def compute_targets(data: SelfplayOutput):
    batch_size = config.selfplay_batch_size // num_devices
    return _compute_targets_impl(data, batch_size, config.max_steps)

# ============================================================================
# 训练与评估
# ============================================================================

def loss_fn(params, batch_stats, samples: Sample, rng_key):
    obs, policy_tgt, is_strong = samples.obs, samples.policy_tgt, samples.is_strong
    do_mirror = jax.random.bernoulli(rng_key, 0.5)
    obs = jnp.where(do_mirror, jax.vmap(mirror_observation)(obs), obs)
    policy_tgt = jnp.where(do_mirror, jax.vmap(mirror_policy)(policy_tgt), policy_tgt)
    
    (logits, value), new_batch_stats = forward(params, batch_stats, obs, is_training=True)
    
    # 策略损失 (只学导师)
    policy_ce = optax.softmax_cross_entropy(logits, policy_tgt)
    num_strong = jnp.maximum(jnp.sum(is_strong), 1.0)
    policy_loss = jnp.sum(policy_ce * is_strong) / num_strong
    
    # 价值损失 (全学)
    value_loss = jnp.sum(optax.l2_loss(value, samples.value_tgt) * samples.mask) / jnp.maximum(jnp.sum(samples.mask), 1.0)
    
    # 熵损失 (防止模型变僵硬，保持顶级 AI 的灵活性)
    probs = jax.nn.softmax(logits)
    entropy = -jnp.sum(probs * jax.nn.log_softmax(logits), axis=-1)
    entropy_loss = -0.01 * jnp.mean(entropy)
    
    total_loss = policy_loss + 0.5 * value_loss + entropy_loss
    
    return total_loss, (new_batch_stats, policy_loss, value_loss)

@jax.pmap
def evaluate(model_red, model_black, rng_key):
    params_r, stats_r = model_red
    params_b, stats_b = model_black
    batch_size = config.eval_games // num_devices
    
    def step_fn(state, key):
        obs = jax.vmap(env.observe)(state)
        is_red = state.current_player == 0
        (logits_r, value_r), _ = forward(params_r, stats_r, obs)
        (logits_b, value_b), _ = forward(params_b, stats_b, obs)
        
        logits = jnp.where(is_red[:, None], logits_r, logits_b)
        value = jnp.where(is_red, value_r, value_b)
        
        rotate_idx = jnp.arange(ACTION_SPACE_SIZE)
        rotated_idx = rotate_action(rotate_idx)
        logits = jnp.where(is_red[:, None], logits, logits[:, rotated_idx])
        
        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)
        policy_output = mctx.gumbel_muzero_policy(
            params=(model_red, model_black), rng_key=key, root=root,
            recurrent_fn=lambda ms, k, a, s: recurrent_fn(ms[0] if s.current_player[0]==0 else ms[1], k, a, s), # 简化版
            num_simulations=96, max_num_considered_actions=16, invalid_actions=~state.legal_action_mask,
        )
        next_state = jax.vmap(env.step)(state, policy_output.action)
        return next_state, next_state.terminated

    state = jax.vmap(env.init)(jax.random.split(rng_key, batch_size))
    terminated = jnp.zeros(batch_size, dtype=jnp.bool_)
    def body_fn(args):
        s, t, k = args
        k, sk = jax.random.split(k)
        ns, nt = step_fn(s, sk)
        return ns, t | nt, k
    state, _, _ = jax.lax.while_loop(lambda args: ~jnp.all(args[1]), body_fn, (state, terminated, rng_key))
    return state.winner

# ============================================================================
# 主循环
# ============================================================================

def main():
    print("=" * 50 + "\nZeroForge - 极简顶级架构\n" + "=" * 50)
    rng_key = jax.random.PRNGKey(config.seed)
    rng_key, subkey = jax.random.split(rng_key)
    
    dummy_obs = jnp.zeros((config.selfplay_batch_size // num_devices, 240, 10, 9))
    variables = net.init(subkey, dummy_obs, train=True)
    model = (variables['params'], variables['batch_stats'])
    
    history_models = {0: jax.device_get(model)}
    iteration_elos = {0: 1500.0}
    
    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(model[0])
    
    model = jax.device_put_replicated(model, devices)
    opt_state = jax.device_put_replicated(opt_state, devices)
    
    @partial(jax.pmap, axis_name='i')
    def train_step(model, opt_state, samples, rng_key):
        params, batch_stats = model
        grads, (new_batch_stats, ploss, vloss) = jax.grad(loss_fn, has_aux=True)(params, batch_stats, samples, rng_key)
        updates, opt_state = optimizer.update(jax.lax.pmean(grads, 'i'), opt_state)
        return (optax.apply_updates(params, updates), new_batch_stats), opt_state, ploss, vloss

    # --- 拆分编译边界：分步预编译，大幅减少编译时间 ---
    print("正在分步预编译 JAX 算子（拆分边界优化）...")
    comp_st = time.time()
    dummy_key = jax.random.PRNGKey(0)
    dummy_keys = jax.random.split(dummy_key, num_devices)
    batch_size = config.selfplay_batch_size // num_devices
    
    # 取第一个设备的模型用于预编译
    model_single = jax.tree.map(lambda x: x[0], model)
    params, batch_stats = model_single
    
    # 0. 预编译 recurrent_fn 内部子单元（关键优化！）
    print("  [0/6] 编译 MCTS 内部子单元...")
    t0 = time.time()
    dummy_state = jax.vmap(env.init)(jax.random.split(dummy_key, batch_size))
    dummy_action = jnp.zeros(batch_size, dtype=jnp.int32)
    
    # 编译环境步进
    _ = _env_step_batch(dummy_state, dummy_action)
    # 编译观察生成
    dummy_obs = _env_observe_batch(dummy_state)
    # 编译神经网络推理
    dummy_logits, dummy_value = _forward_inference(params, batch_stats, dummy_obs)
    # 编译 logits 处理
    _ = _process_logits(dummy_logits, dummy_state.current_player, dummy_state.legal_action_mask)
    print(f"      完成，耗时: {time.time()-t0:.1f}s")
    
    # 1. 预编译 _prepare_root (神经网络推理)
    print("  [1/6] 编译神经网络推理...")
    t1 = time.time()
    dummy_noise_keys = jax.random.split(dummy_key, batch_size)
    _ = _prepare_root(params, batch_stats, dummy_state, dummy_noise_keys, batch_size)
    print(f"      完成，耗时: {time.time()-t1:.1f}s")
    
    # 2. 预编译 _mcts_search (MCTS 搜索 - 这是主要耗时的部分)
    print("  [2/6] 编译 MCTS 搜索 (强)...")
    t2 = time.time()
    _, dummy_root = _prepare_root(params, batch_stats, dummy_state, dummy_noise_keys, batch_size)
    _ = _mcts_search(model_single, dummy_key, dummy_root, ~dummy_state.legal_action_mask,
                     config.strong_simulations, config.max_num_considered_actions)
    print(f"      完成，耗时: {time.time()-t2:.1f}s")
    
    print("  [3/6] 编译 MCTS 搜索 (弱)...")
    t3 = time.time()
    _ = _mcts_search(model_single, dummy_key, dummy_root, ~dummy_state.legal_action_mask,
                     config.weak_simulations, config.max_num_considered_actions)
    print(f"      完成，耗时: {time.time()-t3:.1f}s")
    
    # 4. 预编译 _select_and_step
    print("  [4/6] 编译动作选择与状态更新...")
    t4 = time.time()
    dummy_weights = jnp.ones((batch_size, ACTION_SPACE_SIZE)) / ACTION_SPACE_SIZE
    dummy_is_strong = jnp.ones(batch_size, dtype=bool)
    dummy_temp = jnp.ones(batch_size)
    dummy_sample_keys = jax.random.split(dummy_key, batch_size)
    dummy_next_keys = jax.random.split(dummy_key, batch_size)
    _ = _select_and_step(dummy_state, dummy_weights, dummy_weights, dummy_is_strong,
                         dummy_temp, dummy_sample_keys, dummy_next_keys, batch_size)
    print(f"      完成，耗时: {time.time()-t4:.1f}s")
    
    # 5. 预编译完整自玩流程（验证）
    print("  [5/6] 编译完整自玩流程...")
    t5 = time.time()
    dummy_data = selfplay_pmap(model, dummy_keys)
    dummy_samples = compute_targets(dummy_data)
    print(f"      完成，耗时: {time.time()-t5:.1f}s")
    
    # 6. 预编译 train_step
    print("  [6/6] 编译训练步骤...")
    t6 = time.time()
    batch_per_device = config.training_batch_size // num_devices
    dummy_batch = jax.tree.map(
        lambda x: x[:, 0, :batch_per_device].reshape((num_devices, batch_per_device) + x.shape[3:]), 
        dummy_samples
    )
    _ = train_step(model, opt_state, dummy_batch, dummy_keys)
    print(f"      完成，耗时: {time.time()-t6:.1f}s")
    
    print(f"预编译完成，总耗时: {time.time()-comp_st:.1f}s. 开始训练！")

    os.makedirs(config.ckpt_dir, exist_ok=True)
    writer = SummaryWriter(config.log_dir)
    
    iteration, frames = 0, 0
    while True:
        iteration += 1
        st = time.time()
        rng_key, sk1, sk2 = jax.random.split(rng_key, 3)
        data = selfplay_pmap(model, jax.random.split(sk1, num_devices))
        samples = compute_targets(data)
        
        data_np = jax.device_get(data)
        first_term = (jnp.cumsum(data_np.terminated, axis=1) == 1) & data_np.terminated
        r, b, d = int((first_term & (data_np.winner == 0)).sum()), int((first_term & (data_np.winner == 1)).sum()), int((first_term & (data_np.winner == -1)).sum())
        
        samples_flat = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[3:]), jax.device_get(samples))
        frames += samples_flat.obs.shape[0]
        
        rng_key, sk3 = jax.random.split(rng_key)
        ixs = jax.random.permutation(sk3, jnp.arange(samples_flat.obs.shape[0]))
        samples_flat = jax.tree.map(lambda x: x[ixs], samples_flat)
        
        num_updates = max(1, samples_flat.obs.shape[0] // config.training_batch_size)
        policy_losses, value_losses = [], []
        for i in range(num_updates):
            batch = jax.tree.map(lambda x: x[i*config.training_batch_size:(i+1)*config.training_batch_size].reshape((num_devices, -1) + x.shape[1:]), samples_flat)
            rng_key, sk4 = jax.random.split(rng_key)
            model, opt_state, ploss, vloss = train_step(model, opt_state, batch, jax.random.split(sk4, num_devices))
            policy_losses.append(float(ploss.mean())); value_losses.append(float(vloss.mean()))
        
        print(f"iter={iteration}, ploss={np.mean(policy_losses):.4f}, vloss={np.mean(value_losses):.4f}, frames={frames}, time={time.time()-st:.1f}s | 自玩: {r+b+d}局 红{r} 黑{b} 和{d}")
        writer.add_scalar("train/policy_loss", np.mean(policy_losses), iteration)
        writer.add_scalar("train/value_loss", np.mean(value_losses), iteration)
        
        if iteration % 10 == 0:
            model_np = jax.device_get(jax.tree.map(lambda x: x[0], model))
            history_models[iteration] = model_np
            with open(os.path.join(config.ckpt_dir, f"ckpt_{iteration:06d}.pkl"), 'wb') as f:
                pickle.dump({'model': model_np, 'iteration': iteration, 'frames': frames}, f)
        
        if iteration % config.eval_interval == 0:
            past_iter = max(0, iteration - config.past_model_offset)
            past_model = jax.device_put_replicated(history_models[past_iter], devices)
            rng_key, sk5 = jax.random.split(rng_key)
            wr = (evaluate(model, past_model, jax.random.split(sk5, num_devices)) == 0).sum()
            wb = (evaluate(past_model, model, jax.random.split(sk5, num_devices)) == 1).sum()
            score = (wr + wb + 0.5 * (config.eval_games - wr - wb)) / config.eval_games
            elo_diff = 400.0 * np.log10(score / (1.0 - score)) if 0 < score < 1 else (400 if score >= 1 else -400)
            iteration_elos[iteration] = iteration_elos.get(past_iter, 1500.0) + elo_diff
            print(f"评估 vs Iter {past_iter}: 胜率 {score:.2%}, ELO {iteration_elos[iteration]:.0f}")
            writer.add_scalar("eval/elo", iteration_elos[iteration], iteration)

if __name__ == "__main__":
    main()

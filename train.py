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
    num_channels: int = 256
    num_blocks: int = 12
    
    # 训练超参数
    learning_rate: float = 2e-4
    training_batch_size: int = 512
    
    # 自对弈与搜索 (现代化顶级配置)
    selfplay_batch_size: int = 512
    strong_simulations: int = 256    # 导师算力
    weak_simulations: int = 32       # 学生算力
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

def recurrent_fn(model, rng_key, action, state):
    params, batch_stats = model
    prev_player = state.current_player
    state = jax.vmap(env.step)(state, action)
    obs = jax.vmap(env.observe)(state)
    (logits, value), _ = forward(params, batch_stats, obs)
    
    rotate_idx = jnp.arange(ACTION_SPACE_SIZE)
    rotated_idx = rotate_action(rotate_idx)
    logits = jnp.where(state.current_player[:, None] == 0, logits, logits[:, rotated_idx])
    
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)
    
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
# 自玩
# ============================================================================

@partial(jax.pmap, axis_name='i')
def selfplay_step(model, state, key, is_red_strong):
    """单步自对弈算子 - 拆分编译边界的关键"""
    params, batch_stats = model
    batch_size = state.current_player.shape[0]
    
    key_noise, key_search, key_sample, key_next = jax.random.split(key, 4)
    obs = jax.vmap(env.observe)(state)
    (logits, value), _ = forward(params, batch_stats, obs)
    
    rotate_idx = jnp.arange(ACTION_SPACE_SIZE)
    rotated_idx = rotate_action(rotate_idx)
    logits = jnp.where(state.current_player[:, None] == 0, logits, logits[:, rotated_idx])
    
    def _add_noise(l, k):
        noise = jax.random.dirichlet(k, jnp.ones(ACTION_SPACE_SIZE) * 0.3)
        p = jax.nn.softmax(l)
        return jnp.log(0.75 * p + 0.25 * noise + 1e-10)
    
    noise_keys = jax.random.split(key_noise, batch_size)
    logits = jax.vmap(_add_noise)(logits, noise_keys)
    root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)
    
    is_strong_step = jnp.where(state.current_player == 0, is_red_strong, ~is_red_strong)
    
    # 使用 checkpoint 减少 recurrent_fn 的显存占用和编译图复杂度
    checkpointed_recurrent_fn = jax.checkpoint(recurrent_fn)
    
    k1, k2 = jax.random.split(key_search)
    policy_output_strong = mctx.gumbel_muzero_policy(
        params=model, rng_key=k1, root=root, recurrent_fn=checkpointed_recurrent_fn,
        num_simulations=config.strong_simulations,
        max_num_considered_actions=config.max_num_considered_actions,
        invalid_actions=~state.legal_action_mask,
    )
    policy_output_weak = mctx.gumbel_muzero_policy(
        params=model, rng_key=k2, root=root, recurrent_fn=checkpointed_recurrent_fn,
        num_simulations=config.weak_simulations,
        max_num_considered_actions=config.max_num_considered_actions,
        invalid_actions=~state.legal_action_mask,
    )
    
    action_weights = jnp.where(is_strong_step[:, None], policy_output_strong.action_weights, policy_output_weak.action_weights)
    
    temp = jnp.where(state.step_count < config.temperature_steps, config.temperature_initial, config.temperature_final)
    def _sample_action(w, t, k):
        t = jnp.maximum(t, 1e-3)
        w_temp = jnp.power(w + 1e-10, 1.0 / t)
        return jax.random.choice(k, ACTION_SPACE_SIZE, p=w_temp / jnp.sum(w_temp))
    
    sample_keys = jax.random.split(key_sample, batch_size)
    action = jax.vmap(_sample_action)(action_weights, temp, sample_keys)
    
    actor = state.current_player
    next_state = jax.vmap(env.step)(state, action)
    
    normalized_action_weights = jnp.where(state.current_player[:, None] == 0, action_weights, action_weights[:, rotated_idx])
    
    data = SelfplayOutput(
        obs=obs, action_weights=normalized_action_weights,
        reward=next_state.rewards[jnp.arange(batch_size), actor],
        terminated=next_state.terminated,
        discount=jnp.where(next_state.terminated, 0.0, -1.0),
        winner=next_state.winner, draw_reason=next_state.draw_reason,
        is_strong=is_strong_step
    )
    
    next_state_reset = jax.vmap(lambda s, k: jax.lax.cond(s.terminated, lambda: env.init(k), lambda: s))(next_state, jax.random.split(key_next, batch_size))
    return next_state_reset, data

def selfplay(model, rng_key):
    """
    自玩主循环 - 通过 Python 循环调用 pmap 算子，
    将“编译边界”限制在单步内，极大降低编译耗时。
    """
    batch_size_per_device = config.selfplay_batch_size // num_devices
    
    rng_key, subkey = jax.random.split(rng_key)
    is_red_strong = jax.random.bernoulli(subkey, 0.5, shape=(num_devices, batch_size_per_device))
    
    # 初始状态
    init_keys = jax.random.split(rng_key, num_devices * batch_size_per_device).reshape(num_devices, batch_size_per_device, -1)
    state = jax.pmap(jax.vmap(env.init))(init_keys)
    
    all_data = []
    step_keys = jax.random.split(rng_key, config.max_steps * num_devices).reshape(config.max_steps, num_devices, -1)
    
    print(f"开始自对弈迭代 ({config.max_steps} 步)...")
    for t in range(config.max_steps):
        state, data = selfplay_step(model, state, step_keys[t], is_red_strong)
        all_data.append(data)
    
    # 合并数据: 将 list of SelfplayOutput (each has shape [num_devices, batch_per_device, ...])
    # 转换为单个 SelfplayOutput (shape [num_devices, config.max_steps, batch_per_device, ...])
    # 注意：需要交换前两个维度，以便后续 pmap(compute_targets) 正常按设备分发
    def stack_and_transpose(*leaves):
        stacked = jnp.stack(leaves, axis=1) # [num_devices, max_steps, batch_per_device, ...]
        return stacked
    
    return jax.tree.map(stack_and_transpose, *all_data)

@jax.pmap
def compute_targets(data: SelfplayOutput):
    # data shape: [max_steps, batch_size_per_device, ...]
    batch_size = data.reward.shape[1]
    # ...
    value_mask = jnp.cumsum(data.terminated[::-1, :], axis=0)[::-1, :] >= 1
    def body_fn(carry, i):
        ix = config.max_steps - i - 1
        v = data.reward[ix] + data.discount[ix] * carry
        return v, v
    _, value_tgt = jax.lax.scan(body_fn, jnp.zeros(batch_size), jnp.arange(config.max_steps))
    return Sample(obs=data.obs, policy_tgt=data.action_weights, value_tgt=value_tgt[::-1, :], mask=value_mask, is_strong=data.is_strong)

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

@partial(jax.pmap, axis_name='i')
def evaluate_step(model_red, model_black, state, key):
    """单步评估算子"""
    params_r, stats_r = model_red
    params_b, stats_b = model_black
    
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
    
    # 评估时也使用 checkpoint
    checkpointed_recurrent_fn = jax.checkpoint(recurrent_fn)
    
    policy_output = mctx.gumbel_muzero_policy(
        params=(model_red, model_black), rng_key=key, root=root,
        recurrent_fn=lambda ms, k, a, s: checkpointed_recurrent_fn(ms[0] if s.current_player[0]==0 else ms[1], k, a, s),
        num_simulations=96, max_num_considered_actions=16, invalid_actions=~state.legal_action_mask,
    )
    next_state = jax.vmap(env.step)(state, policy_output.action)
    return next_state, next_state.terminated

def evaluate(model_red, model_black, rng_key):
    """
    评估主循环 - 同样通过拆分边界降低编译时间
    """
    batch_size_per_device = config.eval_games // num_devices
    init_keys = jax.random.split(rng_key, num_devices * batch_size_per_device).reshape(num_devices, batch_size_per_device, -1)
    state = jax.pmap(jax.vmap(env.init))(init_keys)
    
    terminated = jnp.zeros((num_devices, batch_size_per_device), dtype=jnp.bool_)
    
    # 使用 Python 循环替代 lax.while_loop
    step_count = 0
    while not jnp.all(terminated) and step_count < config.max_steps:
        rng_key, sk = jax.random.split(rng_key)
        step_keys = jax.random.split(sk, num_devices)
        state, step_terminated = evaluate_step(model_red, model_black, state, step_keys)
        terminated = terminated | step_terminated
        step_count += 1
        
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

    # --- 极简顶级优化：启动即并行编译所有核心算子 ---
    print("正在并行预编译 JAX Selfplay / Train 算子...")
    comp_st = time.time()
    dummy_key = jax.random.PRNGKey(0)
    dummy_keys = jax.random.split(dummy_key, num_devices)
    
    # 1. 编译 Selfplay Step (只需编译单步)
    batch_size_per_device = config.selfplay_batch_size // num_devices
    init_keys = jax.random.split(dummy_key, num_devices * batch_size_per_device).reshape(num_devices, batch_size_per_device, -1)
    dummy_state = jax.pmap(jax.vmap(env.init))(init_keys)
    dummy_is_strong = jnp.ones((num_devices, batch_size_per_device), dtype=jnp.bool_)
    
    _, dummy_step_data = selfplay_step(model, dummy_state, dummy_keys, dummy_is_strong)
    
    # 2. 编译 compute_targets (需要构造正确形状的 dummy 数据)
    # 构造一个 shape 为 [num_devices, max_steps, batch_size_per_device, ...] 的 dummy data
    def make_dummy_full(x):
        return jnp.broadcast_to(x[:, None], (num_devices, config.max_steps, batch_size_per_device) + x.shape[2:])
    dummy_full_data = jax.tree.map(make_dummy_full, dummy_step_data)
    dummy_samples = compute_targets(dummy_full_data)
    
    # 3. 编译 Train Step
    batch_per_device = config.training_batch_size // num_devices
    dummy_batch = jax.tree.map(
        lambda x: x[:, 0, :batch_per_device].reshape((num_devices, batch_per_device) + x.shape[3:]), 
        dummy_samples
    )
    _ = train_step(model, opt_state, dummy_batch, dummy_keys)
    
    print(f"预编译完成，耗时: {time.time()-comp_st:.1f}s. 开始训练！")

    os.makedirs(config.ckpt_dir, exist_ok=True)
    writer = SummaryWriter(config.log_dir)
    
    iteration, frames = 0, 0
    while True:
        iteration += 1
        st = time.time()
        rng_key, sk1, sk2 = jax.random.split(rng_key, 3)
        data = selfplay(model, jax.random.split(sk1, num_devices))
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

#!/usr/bin/env python3
"""
ZeroForge - 中国象棋 Gumbel AlphaZero
参考: https://github.com/sotetsuk/pgx/blob/main/examples/alphazero.py

用法: python train.py [seed=123] [num_simulations=64]
"""

import os
import sys
import time
import pickle
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import optax
import mctx

from xiangqi.env import XiangqiEnv
from networks.alphazero import AlphaZeroNetwork

# ============================================================================
# 配置
# ============================================================================

class Config:
    # 基础配置
    seed: int = 42
    ckpt_dir: str = "checkpoints"
    
    # 网络架构
    num_channels: int = 128
    num_blocks: int = 12
    
    # 训练超参数
    learning_rate: float = 2e-4
    training_batch_size: int = 512
    
    # 自对弈与搜索
    selfplay_batch_size: int = 512
    num_simulations: int = 96
    max_num_considered_actions: int = 16
    
    # 环境规则 (统一管理)
    max_steps: int = 400            # 总步数限制
    max_no_capture_steps: int = 60  # 无吃子步数限制 (强制进攻)
    repetition_threshold: int = 3   # 重复局面阈值
    perpetual_check_threshold: int = 6 # 连续将军阈值

config = Config()

# ============================================================================
# 环境和设备
# ============================================================================

devices = jax.local_devices()
num_devices = len(devices)

# 使用统一配置初始化环境
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

# ============================================================================
# 网络前向
# ============================================================================

def forward(params, batch_stats, obs, is_training=False):
    (logits, value), new_batch_stats = net.apply(
        {'params': params, 'batch_stats': batch_stats},
        obs, train=is_training, mutable=['batch_stats']
    )
    return (logits, value), new_batch_stats['batch_stats']

# ============================================================================
# MCTS
# ============================================================================

def recurrent_fn(model, rng_key, action, state):
    params, batch_stats = model
    current_player = state.current_player
    state = jax.vmap(env.step)(state, action)
    
    obs = jax.vmap(env.observe)(state)
    (logits, value), _ = forward(params, batch_stats, obs)
    
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)
    
    reward = state.rewards[jnp.arange(state.rewards.shape[0]), current_player]
    value = jnp.where(state.terminated, 0.0, value)
    discount = jnp.where(state.terminated, 0.0, -1.0)
    
    return mctx.RecurrentFnOutput(
        reward=reward, discount=discount, prior_logits=logits, value=value
    ), state

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

class Sample(NamedTuple):
    obs: jnp.ndarray
    policy_tgt: jnp.ndarray
    value_tgt: jnp.ndarray
    mask: jnp.ndarray

# ============================================================================
# 自玩
# ============================================================================

@jax.pmap
def selfplay(model, rng_key):
    params, batch_stats = model
    batch_size = config.selfplay_batch_size // num_devices
    
    def step_fn(state, key):
        key1, key2 = jax.random.split(key)
        obs = jax.vmap(env.observe)(state)
        
        (logits, value), _ = forward(params, batch_stats, obs)
        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)
        
        policy_output = mctx.gumbel_muzero_policy(
            params=model,
            rng_key=key1,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=config.num_simulations,
            max_num_considered_actions=config.max_num_considered_actions,
            invalid_actions=~state.legal_action_mask,
        )
        
        actor = state.current_player
        keys = jax.random.split(key2, batch_size)
        
        # 先执行一步动作，不立即 reset
        next_state = jax.vmap(env.step)(state, policy_output.action)
        
        # 记录数据
        data = SelfplayOutput(
            obs=obs,
            action_weights=policy_output.action_weights,
            reward=next_state.rewards[jnp.arange(batch_size), actor],
            terminated=next_state.terminated,
            discount=jnp.where(next_state.terminated, 0.0, -1.0),
            winner=next_state.winner,
        )
        
        # 检查是否需要 reset
        def _reset_fn(s, k):
            return jax.lax.cond(s.terminated, lambda: env.init(k), lambda: s)
        
        next_state_reset = jax.vmap(_reset_fn)(next_state, keys)
        
        return next_state_reset, data
    
    rng_key, subkey = jax.random.split(rng_key)
    keys = jax.random.split(subkey, batch_size)
    state = jax.vmap(env.init)(keys)
    
    _, data = jax.lax.scan(step_fn, state, jax.random.split(rng_key, config.max_steps))
    return data

@jax.pmap
def compute_targets(data: SelfplayOutput):
    batch_size = config.selfplay_batch_size // num_devices
    value_mask = jnp.cumsum(data.terminated[::-1, :], axis=0)[::-1, :] >= 1
    
    def body_fn(carry, i):
        ix = config.max_num_steps - i - 1
        v = data.reward[ix] + data.discount[ix] * carry
        return v, v
    
    _, value_tgt = jax.lax.scan(body_fn, jnp.zeros(batch_size), jnp.arange(config.max_num_steps))
    value_tgt = value_tgt[::-1, :]
    
    return Sample(obs=data.obs, policy_tgt=data.action_weights, value_tgt=value_tgt, mask=value_mask)

# ============================================================================
# 训练
# ============================================================================

def loss_fn(params, batch_stats, samples: Sample):
    (logits, value), new_batch_stats = forward(params, batch_stats, samples.obs, is_training=True)
    policy_loss = jnp.mean(optax.softmax_cross_entropy(logits, samples.policy_tgt))
    # 修复掩码：只在有效步数上计算损失 (mask 为 True 的地方)
    value_loss = jnp.mean(optax.l2_loss(value, samples.value_tgt) * samples.mask)
    return policy_loss + value_loss, (new_batch_stats, policy_loss, value_loss)

# ============================================================================
# 主函数
# ============================================================================

def main():
    print("=" * 50)
    print("ZeroForge - 中国象棋 Gumbel AlphaZero")
    print("=" * 50)
    print(f"设备: {num_devices}")
    print(f"网络: {config.num_channels}ch x {config.num_blocks}blocks")
    print(f"批大小: {config.selfplay_batch_size}, MCTS: {config.num_simulations} sims")
    
    rng_key = jax.random.PRNGKey(config.seed)
    rng_key, subkey = jax.random.split(rng_key)
    
    # 初始化模型
    batch_size = config.selfplay_batch_size // num_devices
    dummy_obs = jnp.zeros((batch_size, 240, 10, 9))
    variables = net.init(subkey, dummy_obs, train=True)
    model = (variables['params'], variables['batch_stats'])
    
    # 优化器
    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(model[0])
    
    # 复制到所有设备
    model = jax.device_put_replicated(model, devices)
    opt_state = jax.device_put_replicated(opt_state, devices)
    
    os.makedirs(config.ckpt_dir, exist_ok=True)
    
    iteration = 0
    frames = 0
    
    # pmap 训练函数
    @partial(jax.pmap, axis_name='i')
    def train_step(model, opt_state, samples):
        params, batch_stats = model
        grads, (new_batch_stats, ploss, vloss) = jax.grad(loss_fn, has_aux=True)(params, batch_stats, samples)
        grads = jax.lax.pmean(grads, axis_name='i')
        updates, opt_state = optimizer.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return (params, new_batch_stats), opt_state, ploss, vloss
    
    print("开始训练...")
    
    while True:
        iteration += 1
        st = time.time()
        
        # 自玩
        rng_key, subkey = jax.random.split(rng_key)
        keys = jax.random.split(subkey, num_devices)
        data = selfplay(model, keys)
        samples = compute_targets(data)
        
        # 统计（data 形状: num_devices, max_num_steps, batch_size）
        data_np = jax.device_get(data)
        term = data_np.terminated  # (D, T, B)
        winner = data_np.winner    # (D, T, B)
        # 沿时间轴找第一次结束
        first_term = (jnp.cumsum(term, axis=1) == 1) & term
        r = int((first_term & (winner == 0)).sum())
        b = int((first_term & (winner == 1)).sum())
        d = int((first_term & (winner == -1)).sum())
        
        # 处理样本
        samples_np = jax.device_get(samples)
        samples_flat = jax.tree.map(lambda x: x.reshape((-1,) + x.shape[3:]), samples_np)
        frames += samples_flat.obs.shape[0]
        
        # 打乱
        rng_key, subkey = jax.random.split(rng_key)
        ixs = jax.random.permutation(subkey, jnp.arange(samples_flat.obs.shape[0]))
        samples_flat = jax.tree.map(lambda x: x[ixs], samples_flat)
        
        # 分成 minibatch
        num_updates = max(1, samples_flat.obs.shape[0] // config.training_batch_size)
        minibatches = jax.tree.map(
            lambda x: x[:num_updates * config.training_batch_size].reshape((num_updates, num_devices, -1) + x.shape[1:]),
            samples_flat
        )
        
        # 训练
        policy_losses, value_losses = [], []
        for i in range(num_updates):
            batch = jax.tree.map(lambda x: x[i], minibatches)
            model, opt_state, ploss, vloss = train_step(model, opt_state, batch)
            policy_losses.append(float(ploss.mean()))
            value_losses.append(float(vloss.mean()))
        
        et = time.time()
        total = r + b + d
        
        print(
            f"iter={iteration}, "
            f"ploss={sum(policy_losses)/len(policy_losses):.4f}, "
            f"vloss={sum(value_losses)/len(value_losses):.4f}, "
            f"frames={frames}, "
            f"time={et-st:.1f}s | "
            f"自玩: {total}局 红{r} 黑{b} 和{d}"
        )
        
        # 保存
        if iteration % 10 == 0:
            model_save = jax.tree.map(lambda x: x[0], model)
            ckpt = {'model': jax.device_get(model_save), 'iteration': iteration, 'frames': frames}
            path = os.path.join(config.ckpt_dir, f"ckpt_{iteration:06d}.pkl")
            with open(path, 'wb') as f:
                pickle.dump(ckpt, f)
            print(f"已保存: {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
GPU 诊断脚本 - 分步测试各个组件
"""
import os
import time

print("=" * 60)
print("Step 1: 检查 JAX 配置")
print("=" * 60)

import jax
import jax.numpy as jnp

print(f"JAX 版本: {jax.__version__}")
print(f"默认后端: {jax.default_backend()}")
print(f"可用设备: {jax.devices()}")

if jax.default_backend() != 'gpu':
    print("\n⚠️ 警告: JAX 没有使用 GPU 后端!")
    print("请确保安装了 jax[cuda] 版本")
    print("pip install --upgrade 'jax[cuda12]'")

print("\n" + "=" * 60)
print("Step 2: 测试简单 GPU 计算")
print("=" * 60)

# 简单矩阵乘法测试
@jax.jit
def simple_matmul(x, y):
    return jnp.dot(x, y)

x = jnp.ones((1000, 1000))
y = jnp.ones((1000, 1000))

print("编译简单矩阵乘法...")
t0 = time.time()
result = simple_matmul(x, y)
result.block_until_ready()
print(f"首次运行 (含编译): {time.time() - t0:.2f}s")

t0 = time.time()
for _ in range(10):
    result = simple_matmul(x, y)
result.block_until_ready()
print(f"10 次运行: {time.time() - t0:.4f}s")
print("✅ 简单 GPU 计算正常")

print("\n" + "=" * 60)
print("Step 3: 测试环境初始化")
print("=" * 60)

from xiangqi.env import XiangqiEnv

env = XiangqiEnv()
print(f"动作空间大小: {env.action_space_size}")
print(f"观察形状: {env.observation_shape}")

key = jax.random.PRNGKey(42)

print("编译单局初始化...")
t0 = time.time()
state = env.init(key)
print(f"单局初始化: {time.time() - t0:.2f}s")

print("\n编译 vmap 初始化 (16 局)...")
v_init = jax.vmap(env.init)
t0 = time.time()
keys = jax.random.split(key, 16)
states = v_init(keys)
print(f"vmap 初始化: {time.time() - t0:.2f}s")
print("✅ 环境初始化正常")

print("\n" + "=" * 60)
print("Step 4: 测试环境 step")
print("=" * 60)

print("编译单局 step...")
t0 = time.time()
# 找一个合法动作
action = jnp.argmax(state.legal_action_mask)
new_state = env.step(state, action)
print(f"单局 step: {time.time() - t0:.2f}s")

print("\n编译 vmap step (16 局)...")
v_step = jax.vmap(env.step)
t0 = time.time()
actions = jnp.argmax(states.legal_action_mask, axis=-1)
new_states = v_step(states, actions)
print(f"vmap step: {time.time() - t0:.2f}s")
print("✅ 环境 step 正常")

print("\n" + "=" * 60)
print("Step 5: 测试网络前向传播")
print("=" * 60)

from networks.muzero import MuZeroNetwork, create_train_state

network = MuZeroNetwork(
    action_space_size=env.action_space_size,
    hidden_dim=128,  # 用较小的隐藏层
)

print("初始化网络...")
t0 = time.time()
train_state = create_train_state(
    key, network,
    input_shape=(16, 240, 10, 9),
    learning_rate=0.001,
)
print(f"网络初始化: {time.time() - t0:.2f}s")

print("\n编译网络前向传播...")
v_observe = jax.vmap(env.observe)
obs = v_observe(states)

@jax.jit
def forward(params, obs):
    return network.apply(params, obs)

t0 = time.time()
output = forward(train_state.params, obs)
print(f"前向传播编译: {time.time() - t0:.2f}s")
print(f"输出 policy_logits 形状: {output.policy_logits.shape}")
print(f"输出 value 形状: {output.value.shape}")
print("✅ 网络前向传播正常")

print("\n" + "=" * 60)
print("Step 6: 测试 MCTS (单步)")
print("=" * 60)

import mctx

def recurrent_fn(params, rng_key, action, embedding):
    next_state, reward, logits, value = network.apply(
        params, embedding, action.astype(jnp.int32),
        method=network.recurrent_inference
    )
    return mctx.RecurrentFnOutput(
        reward=reward,
        discount=jnp.ones_like(reward),
        prior_logits=logits,
        value=value,
    ), next_state

@jax.jit
def mcts_step(params, obs, legal_mask, key):
    output = network.apply(params, obs)
    root = mctx.RootFnOutput(
        prior_logits=output.policy_logits,
        value=output.value,
        embedding=output.hidden_state,
    )
    policy_output = mctx.gumbel_muzero_policy(
        params=params,
        rng_key=key,
        root=root,
        recurrent_fn=recurrent_fn,
        num_simulations=8,  # 很小的值
        invalid_actions=~legal_mask,
        max_num_considered_actions=8,
    )
    return policy_output

print("编译 MCTS (16 局并行, 8 次模拟)...")
t0 = time.time()
policy_out = mcts_step(
    train_state.params,
    obs,
    states.legal_action_mask,
    key,
)
policy_out.action.block_until_ready()
print(f"MCTS 编译: {time.time() - t0:.2f}s")
print(f"选择的动作: {policy_out.action[:5]}")
print("✅ MCTS 正常")

print("\n" + "=" * 60)
print("Step 7: 测试 lax.scan (多步)")
print("=" * 60)

@jax.jit
def multi_step(params, state, key):
    """执行 5 步自玩"""
    v_observe = jax.vmap(env.observe)
    v_step = jax.vmap(env.step)
    v_init = jax.vmap(env.init)
    
    def step_fn(state, key):
        obs = v_observe(state)
        output = network.apply(params, obs)
        
        # 简单策略：贪心选择
        logits = jnp.where(state.legal_action_mask, output.policy_logits, -1e9)
        action = jnp.argmax(logits, axis=-1)
        
        next_state = v_step(state, action)
        return next_state, None
    
    final_state, _ = jax.lax.scan(
        step_fn,
        state,
        jax.random.split(key, 5),  # 5 步
    )
    return final_state

print("编译 lax.scan (5 步, 16 局并行)...")
t0 = time.time()
final = multi_step(train_state.params, states, key)
final.board.block_until_ready()
print(f"lax.scan 编译: {time.time() - t0:.2f}s")
print("✅ lax.scan 正常")

print("\n" + "=" * 60)
print("🎉 所有测试通过!")
print("=" * 60)
print("\n如果这个脚本能正常运行完成，说明各个组件都没问题。")
print("问题可能在于 selfplay_fn 中嵌套了 MCTS + lax.scan，")
print("导致计算图过于复杂，编译时间过长。")
print("\n建议:")
print("1. 继续等待 train.py 的编译完成（可能需要 30-60 分钟）")
print("2. 或者考虑使用 AlphaZero 方式（真实环境模拟）而不是 MuZero（学习的动态模型）")

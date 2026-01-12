#!/usr/bin/env python3
"""
使用训练好的模型在 Web GUI 中对弈
用法: python play.py [checkpoint_path]
"""

import sys
import pickle
import jax
import jax.numpy as jnp
from gui.web_gui import run_web_gui
from networks.alphazero import AlphaZeroNetwork
from xiangqi.env import XiangqiEnv
from xiangqi.actions import ACTION_SPACE_SIZE, rotate_action
import mctx

def load_model(path, net):
    with open(path, 'rb') as f:
        ckpt = pickle.load(f)
    return ckpt['model']

def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/ckpt_000100.pkl"
    print(f"正在加载模型: {ckpt_path}")
    
    # 初始化环境和网络
    env = XiangqiEnv()
    net = AlphaZeroNetwork(
        action_space_size=ACTION_SPACE_SIZE,
        channels=128,
        num_blocks=12,
    )
    
    # 加载参数
    try:
        params, batch_stats = load_model(ckpt_path, net)
    except FileNotFoundError:
        print(f"错误: 找不到模型文件 {ckpt_path}")
        return

    def ai_callback(state):
        # 准备观察
        obs = env.observe(state)[None, ...] # 添加 batch 维度
        
        # 前向计算
        (logits, value), _ = net.apply(
            {'params': params, 'batch_stats': batch_stats},
            obs, train=False, mutable=['batch_stats']
        )
        
        # 视角修正 (如果是黑方，将网络输出的视角 logits 转回真实坐标)
        if state.current_player == 1:
            rotate_idx = jnp.arange(ACTION_SPACE_SIZE)
            rotated_idx = rotate_action(rotate_idx)
            logits = logits[:, rotated_idx]
        
        # 掩码
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)
        
        # MCTS 搜索逻辑
        def recurrent_fn(model, rng_key, action, state):
            prev_player = state.current_player
            state = jax.vmap(env.step)(state, action)
            obs = jax.vmap(env.observe)(state)
            (l, v), _ = net.apply(
                {'params': params, 'batch_stats': batch_stats},
                obs, train=False, mutable=['batch_stats']
            )
            
            # 视角修正
            rotate_idx = jnp.arange(ACTION_SPACE_SIZE)
            rotated_idx = rotate_action(rotate_idx)
            l = jnp.where(state.current_player[:, None] == 0, l, l[:, rotated_idx])
            
            l = l - jnp.max(l, axis=-1, keepdims=True)
            l = jnp.where(state.legal_action_mask, l, jnp.finfo(l.dtype).min)
            
            reward = state.rewards[jnp.arange(state.rewards.shape[0]), prev_player]
            discount = jnp.where(state.terminated, 0.0, -1.0)
            return mctx.RecurrentFnOutput(reward=reward, discount=discount, prior_logits=l, value=v), state

        # 运行 Gumbel MCTS
        key = jax.random.PRNGKey(42)
        # 为 chex dataclass 添加 batch 维度
        batched_state = jax.tree_map(lambda x: jnp.expand_dims(x, 0), state)
        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=batched_state)
        
        policy_output = mctx.gumbel_muzero_policy(
            params=None, # 不使用内部 model
            rng_key=key,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=256, # 实际下棋可以用更高的模拟次数
            max_num_considered_actions=16,
            invalid_actions=(~state.legal_action_mask)[None, ...],
        )
        
        return int(policy_output.action[0])

    print("启动 Web GUI...")
    run_web_gui(ai_callback=ai_callback)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
ZeroForge - 中国象棋 Gumbel AlphaZero
参考: https://github.com/zjjMaiMai/GumbelAlphaZero

用法: python train.py [config.yaml]
"""

import os
import sys
import yaml
import jax
import jax.numpy as jnp
import optax
import chex
from typing import NamedTuple
from flax.training.checkpoints import save_checkpoint, restore_checkpoint

from xiangqi.env import XiangqiEnv
from networks.alphazero import AlphaZeroNetwork, create_train_state
import mctx

# ============================================================================
# 加载配置
# ============================================================================

config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"
with open(config_path) as f:
    cfg = yaml.safe_load(f)

seed = cfg.get("seed", 42)
num_batchsize_train = cfg["training"]["batch_size"]
num_batchsize_selfplay = cfg["training"]["batch_size"]
num_step_selfplay = cfg["training"]["selfplay_steps"]
num_simulations = cfg["mcts"]["num_simulations"]
learning_rate = cfg["training"]["learning_rate"]
ckpt_dir = cfg["checkpoint"]["dir"]
ckpt_keep = cfg["checkpoint"]["keep"]

env = XiangqiEnv()

# 网络
net = AlphaZeroNetwork(
    action_space_size=env.action_space_size,
    channels=cfg["network"]["channels"],
    num_blocks=cfg["network"]["num_blocks"],
)


# ============================================================================
# MCTS recurrent_fn
# ============================================================================

def recurrent_fn(model, key, action, state):
    """AlphaZero 树扩展：使用真实环境"""
    player = state.current_player
    state = jax.vmap(env.step)(state, action)
    
    obs = jax.vmap(env.observe)(state)
    (logits, value), _ = model.apply_fn(
        {'params': model.params, 'batch_stats': model.batch_stats},
        obs, train=False, mutable=['batch_stats']
    )
    
    # 屏蔽非法动作
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)
    
    # 奖励：当前玩家视角
    reward = jax.vmap(lambda s, p: s.rewards[p])(state, player)
    
    return mctx.RecurrentFnOutput(
        reward=reward,
        discount=jnp.where(state.terminated, 0.0, -1.0),  # 零和博弈
        prior_logits=logits,
        value=jnp.where(state.terminated, 0.0, value),
    ), state


# ============================================================================
# 数据结构
# ============================================================================

class Trajectory(NamedTuple):
    obs: chex.Array
    prob: chex.Array
    reward: chex.Array
    discount: chex.Array
    terminated: chex.Array


class Sample(NamedTuple):
    obs: chex.Array
    prob: chex.Array
    value: chex.Array
    mask: chex.Array


# ============================================================================
# 自我对弈
# ============================================================================

def auto_reset(step_fn, init_fn):
    """游戏结束自动重置"""
    def wrapped(state, action, key):
        state = step_fn(state, action)
        state = jax.lax.cond(
            state.terminated,
            lambda: init_fn(key),
            lambda: state,
        )
        return state
    return wrapped


@jax.jit
def self_play(model, key):
    """自我对弈生成训练数据"""
    
    def body(state, key):
        k0, k1, k2 = jax.random.split(key, 3)
        
        obs = jax.vmap(env.observe)(state)
        (logits, value), _ = model.apply_fn(
            {'params': model.params, 'batch_stats': model.batch_stats},
            obs, train=False, mutable=['batch_stats']
        )
        
        root = mctx.RootFnOutput(
            prior_logits=logits,
            value=value,
            embedding=state,
        )
        
        policy = mctx.gumbel_muzero_policy(
            model,
            k0,
            root,
            recurrent_fn,
            num_simulations=num_simulations,
            invalid_actions=~state.legal_action_mask,
        )
        
        player = state.current_player
        state = jax.vmap(auto_reset(env.step, env.init))(
            state, policy.action, jax.random.split(k1, num_batchsize_selfplay)
        )
        
        return state, Trajectory(
            obs=obs,
            prob=policy.action_weights,
            reward=jax.vmap(lambda s, p: s.rewards[p])(state, player),
            discount=jnp.where(state.terminated, 0.0, -1.0),
            terminated=state.terminated,
        )
    
    k0, k1, k2 = jax.random.split(key, 3)
    state = jax.vmap(env.init)(jax.random.split(k0, num_batchsize_selfplay))
    _, traj = jax.lax.scan(body, state, jax.random.split(k1, num_step_selfplay))
    
    # 计算价值目标 (reverse scan)
    def compute_value(value, traj):
        value = traj.reward + traj.discount * value
        return value, value
    
    _, value = jax.lax.scan(
        compute_value,
        jnp.zeros_like(traj.reward[0]),
        traj,
        reverse=True,
    )
    
    # mask: 游戏结束后的步骤无效
    mask = jnp.flip(jnp.cumsum(jnp.flip(traj.terminated, 0), 0), 0) >= 1
    
    sample = Sample(obs=traj.obs, prob=traj.prob, value=value, mask=mask)
    
    # (T, B, ...) -> (T*B, ...)
    sample = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), sample)
    
    # 打乱并分成 minibatch
    ixs = jax.random.permutation(k2, sample.obs.shape[0])
    sample = jax.tree.map(lambda x: x[ixs], sample)
    sample = jax.tree.map(
        lambda x: x.reshape(-1, num_batchsize_train, *x.shape[1:]),
        sample
    )
    
    return sample


# ============================================================================
# 训练
# ============================================================================

@jax.jit
def train_step(model, batch):
    """单步训练"""
    
    def loss_fn(params):
        (logits, value), updates = model.apply_fn(
            {'params': params, 'batch_stats': model.batch_stats},
            batch.obs, train=True, mutable=['batch_stats']
        )
        
        policy_loss = optax.losses.softmax_cross_entropy(logits, batch.prob).mean()
        value_loss = (optax.losses.squared_error(value, batch.value) * ~batch.mask).mean()
        
        return policy_loss + value_loss, (updates['batch_stats'], policy_loss, value_loss)
    
    (loss, (batch_stats, ploss, vloss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(model.params)
    
    model = model.apply_gradients(grads=grads, batch_stats=batch_stats)
    
    return model, ploss, vloss


# ============================================================================
# 评估
# ============================================================================

@jax.jit
def eval_step(model, old_model, old_elo, key):
    """评估新模型 vs 旧模型"""
    num_eval = 100
    
    k0, k1, k2 = jax.random.split(key, 3)
    model_player = jax.random.randint(k0, (num_eval,), 0, 2)
    state = jax.vmap(env.init)(jax.random.split(k1, num_eval))
    
    def cond(val):
        s, k, r = val
        return ~s.terminated.all()
    
    def body(val):
        s, k, r = val
        k, k0, k1, k2 = jax.random.split(k, 4)
        
        obs = jax.vmap(env.observe)(s)
        (logits_new, _), _ = model.apply_fn(
            {'params': model.params, 'batch_stats': model.batch_stats},
            obs, train=False, mutable=['batch_stats']
        )
        (logits_old, _), _ = old_model.apply_fn(
            {'params': old_model.params, 'batch_stats': old_model.batch_stats},
            obs, train=False, mutable=['batch_stats']
        )
        
        logits = jnp.where(
            jnp.expand_dims(s.current_player == model_player, -1),
            logits_new, logits_old
        )
        logits = jnp.where(s.legal_action_mask, logits, jnp.finfo(logits.dtype).min)
        action = jnp.argmax(logits, axis=-1)
        
        s = jax.vmap(env.step)(s, action)
        r = r + s.rewards
        return s, k, r
    
    _, _, rewards = jax.lax.while_loop(cond, body, (state, k2, state.rewards))
    results = jax.vmap(lambda x, i: x[i])(rewards, model_player) * 0.5 + 0.5
    
    # 更新 ELO
    def update_elo(elo, result):
        expected = 1 / (1 + 10 ** ((old_elo - elo) / 400))
        elo = elo + 32 * (result - expected)
        return elo, elo
    
    elo, _ = jax.lax.scan(update_elo, old_elo, results)
    return elo


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("=" * 50)
    print("ZeroForge - 中国象棋 Gumbel AlphaZero")
    print("=" * 50)
    print(f"配置: {config_path}")
    print(f"JAX 后端: {jax.default_backend()}")
    print(f"设备数量: {jax.device_count()}")
    print(f"网络: channels={cfg['network']['channels']}, blocks={cfg['network']['num_blocks']}")
    print(f"MCTS: {num_simulations} simulations")
    
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    
    model = create_train_state(
        subkey, net,
        input_shape=(num_batchsize_train, 240, 10, 9),
        learning_rate=learning_rate,
    )
    elo = jnp.float32(1500)
    
    # 恢复检查点
    ckpt_path = os.path.abspath(ckpt_dir)
    os.makedirs(ckpt_path, exist_ok=True)
    
    restored = restore_checkpoint(ckpt_path, None)
    if restored is not None:
        model = model.replace(
            params=restored['params'],
            batch_stats=restored['batch_stats'],
            opt_state=restored['opt_state'],
            step=restored['step'],
        )
        elo = restored['elo']
        print(f"从检查点恢复: step={model.step}, elo={elo:.1f}")
    
    print("开始训练...")
    
    while True:
        key, k0, k1 = jax.random.split(key, 3)
        
        # 自我对弈
        data = self_play(model, k0)
        
        # 训练
        old_model = model
        for i in range(data.obs.shape[0]):
            batch = jax.tree.map(lambda x: x[i], data)
            model, ploss, vloss = train_step(model, batch)
        
        # 评估
        elo = eval_step(model, old_model, elo, k1)
        
        print(
            f"step={int(model.step)}, "
            f"ploss={float(ploss):.4f}, "
            f"vloss={float(vloss):.4f}, "
            f"elo={float(elo):.1f}"
        )
        
        # 保存检查点
        ckpt = {
            'params': model.params,
            'batch_stats': model.batch_stats,
            'opt_state': model.opt_state,
            'step': model.step,
            'elo': elo,
        }
        save_checkpoint(ckpt_path, ckpt, int(model.step), keep=ckpt_keep)


if __name__ == "__main__":
    main()

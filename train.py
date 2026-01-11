#!/usr/bin/env python3
"""
ZeroForge - 中国象棋 Gumbel AlphaZero
参考: https://github.com/zjjMaiMai/GumbelAlphaZero

用法: 
  python train.py           # 正常训练
  python train.py --debug   # 禁用编译调试
"""

import os
import sys

# 调试模式：禁用 JIT 编译
DEBUG_MODE = '--debug' in sys.argv
if DEBUG_MODE:
    os.environ['JAX_DISABLE_JIT'] = '1'
    os.environ['JAX_PLATFORMS'] = 'cpu'  # 强制 CPU，避免 GPU 编译

import yaml
import jax
import jax.numpy as jnp
import optax
import chex
from typing import NamedTuple
from functools import partial
from flax.training.checkpoints import save_checkpoint, restore_checkpoint

from xiangqi.env import XiangqiEnv
from networks.alphazero import AlphaZeroNetwork, create_train_state
import mctx

# ============================================================================
# 静态配置（改了会重新编译）
# ============================================================================

NUM_SIMULATIONS = 32        # MCTS 模拟次数
BATCH_SIZE = 128            # 每卡批大小
SELFPLAY_STEPS = 100        # 每轮自对弈步数
CHANNELS = 128              # 网络通道数
NUM_BLOCKS = 10             # 残差块数量

# ============================================================================
# 设备配置
# ============================================================================

NUM_DEVICES = 1 if DEBUG_MODE else jax.device_count()
TOTAL_BATCH = BATCH_SIZE * NUM_DEVICES

# ============================================================================
# 动态配置
# ============================================================================

config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"
with open(config_path) as f:
    cfg = yaml.safe_load(f)

seed = cfg.get("seed", 42)
learning_rate = cfg["training"].get("learning_rate", 2e-4)
ckpt_dir = cfg["checkpoint"].get("dir", "checkpoints")
ckpt_keep = cfg["checkpoint"].get("keep", 5)

env = XiangqiEnv()

net = AlphaZeroNetwork(
    action_space_size=env.action_space_size,
    channels=CHANNELS,
    num_blocks=NUM_BLOCKS,
)


# ============================================================================
# 数据结构
# ============================================================================

class Trajectory(NamedTuple):
    obs: chex.Array
    prob: chex.Array
    reward: chex.Array
    discount: chex.Array
    terminated: chex.Array
    winner: chex.Array


class Sample(NamedTuple):
    obs: chex.Array
    prob: chex.Array
    value: chex.Array
    mask: chex.Array


class SelfPlayStats(NamedTuple):
    red_wins: chex.Array
    black_wins: chex.Array
    draws: chex.Array
    avg_length: chex.Array


# ============================================================================
# 自我对弈（单卡）
# ============================================================================

def _self_play_single(model, key):
    """单卡自我对弈"""
    
    def recurrent_fn(model, key, action, state):
        player = state.current_player
        state = jax.vmap(env.step)(state, action)
        obs = jax.vmap(env.observe)(state)
        (logits, value), _ = model.apply_fn(
            {'params': model.params, 'batch_stats': model.batch_stats},
            obs, train=False, mutable=['batch_stats']
        )
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)
        reward = jax.vmap(lambda s, p: s.rewards[p])(state, player)
        return mctx.RecurrentFnOutput(
            reward=reward,
            discount=jnp.where(state.terminated, 0.0, -1.0),
            prior_logits=logits,
            value=jnp.where(state.terminated, 0.0, value),
        ), state
    
    def auto_reset(state, action, key):
        state = env.step(state, action)
        return jax.lax.cond(state.terminated, lambda: env.init(key), lambda: state)
    
    def body(state, key):
        k0, k1 = jax.random.split(key)
        obs = jax.vmap(env.observe)(state)
        (logits, value), _ = model.apply_fn(
            {'params': model.params, 'batch_stats': model.batch_stats},
            obs, train=False, mutable=['batch_stats']
        )
        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)
        policy = mctx.gumbel_muzero_policy(
            model, k0, root, recurrent_fn,
            num_simulations=NUM_SIMULATIONS,
            invalid_actions=~state.legal_action_mask,
        )
        player = state.current_player
        next_state = jax.vmap(auto_reset)(state, policy.action, jax.random.split(k1, BATCH_SIZE))
        return next_state, Trajectory(
            obs=obs,
            prob=policy.action_weights,
            reward=jax.vmap(lambda s, p: s.rewards[p])(next_state, player),
            discount=jnp.where(next_state.terminated, 0.0, -1.0),
            terminated=next_state.terminated,
            winner=next_state.winner,
        )
    
    k0, k1, k2 = jax.random.split(key, 3)
    state = jax.vmap(env.init)(jax.random.split(k0, BATCH_SIZE))
    _, traj = jax.lax.scan(body, state, jax.random.split(k1, SELFPLAY_STEPS))
    
    # 统计
    first_term = (jnp.cumsum(traj.terminated, axis=0) == 1) & traj.terminated
    stats = SelfPlayStats(
        red_wins=(first_term & (traj.winner == 0)).sum(),
        black_wins=(first_term & (traj.winner == 1)).sum(),
        draws=(first_term & (traj.winner == -1)).sum(),
        avg_length=jnp.where(first_term, jnp.arange(SELFPLAY_STEPS)[:, None] + 1, 0).sum() / (first_term.sum() + 1e-8),
    )
    
    # 计算价值目标
    def compute_value(value, traj):
        value = traj.reward + traj.discount * value
        return value, value
    _, value = jax.lax.scan(compute_value, jnp.zeros_like(traj.reward[0]), traj, reverse=True)
    
    mask = jnp.flip(jnp.cumsum(jnp.flip(traj.terminated, 0), 0), 0) >= 1
    sample = Sample(obs=traj.obs, prob=traj.prob, value=value, mask=mask)
    sample = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), sample)
    
    # 打乱
    ixs = jax.random.permutation(k2, sample.obs.shape[0])
    sample = jax.tree.map(lambda x: x[ixs], sample)
    sample = jax.tree.map(lambda x: x.reshape(-1, BATCH_SIZE, *x.shape[1:]), sample)
    
    return sample, stats


# ============================================================================
# 训练
# ============================================================================

def _train_step_single(model, batch):
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


if DEBUG_MODE:
    # 调试模式：不编译
    self_play = _self_play_single
    train_step = _train_step_single
else:
    # 正常模式：多卡并行
    @partial(jax.pmap, axis_name='devices')
    def self_play_pmap(model, key):
        sample, stats = _self_play_single(model, key)
        stats = SelfPlayStats(
            red_wins=jax.lax.psum(stats.red_wins, 'devices'),
            black_wins=jax.lax.psum(stats.black_wins, 'devices'),
            draws=jax.lax.psum(stats.draws, 'devices'),
            avg_length=jax.lax.pmean(stats.avg_length, 'devices'),
        )
        return sample, stats
    
    @partial(jax.pmap, axis_name='devices')
    def train_step_pmap(model, batch):
        model, ploss, vloss = _train_step_single(model, batch)
        grads_synced = jax.lax.pmean(model.params, 'devices')  # 同步参数
        return model, jax.lax.pmean(ploss, 'devices'), jax.lax.pmean(vloss, 'devices')
    
    self_play = self_play_pmap
    train_step = train_step_pmap


# ============================================================================
# 评估
# ============================================================================

@jax.jit
def eval_step(model, old_model, old_elo, key):
    num_eval = 100
    k0, k1, k2 = jax.random.split(key, 3)
    model_player = jax.random.randint(k0, (num_eval,), 0, 2)
    state = jax.vmap(env.init)(jax.random.split(k1, num_eval))
    
    def cond(val):
        return ~val[0].terminated.all()
    
    def body(val):
        s, k = val
        k, k0 = jax.random.split(k)
        obs = jax.vmap(env.observe)(s)
        (logits_new, _), _ = model.apply_fn(
            {'params': model.params, 'batch_stats': model.batch_stats},
            obs, train=False, mutable=['batch_stats']
        )
        (logits_old, _), _ = old_model.apply_fn(
            {'params': old_model.params, 'batch_stats': old_model.batch_stats},
            obs, train=False, mutable=['batch_stats']
        )
        logits = jnp.where(jnp.expand_dims(s.current_player == model_player, -1), logits_new, logits_old)
        logits = jnp.where(s.legal_action_mask, logits, jnp.finfo(logits.dtype).min)
        s = jax.vmap(env.step)(s, jnp.argmax(logits, axis=-1))
        return s, k
    
    final_state, _ = jax.lax.while_loop(cond, body, (state, k2))
    new_wins = (final_state.winner == model_player).astype(jnp.float32)
    draws = (final_state.winner == -1).astype(jnp.float32)
    win_rate = (new_wins + 0.5 * draws).mean()
    return old_elo + 400 * (win_rate - 0.5), win_rate


# ============================================================================
# 辅助函数
# ============================================================================

def replicate(x):
    return jax.device_put_replicated(x, jax.local_devices())

def unreplicate(x):
    return jax.tree.map(lambda a: a[0], x)


# ============================================================================
# 主函数
# ============================================================================

def main():
    print("=" * 50)
    print("ZeroForge - 中国象棋 Gumbel AlphaZero")
    print("=" * 50)
    print(f"模式: {'调试(无编译)' if DEBUG_MODE else '正常'}")
    print(f"JAX 后端: {jax.default_backend()}")
    print(f"设备数量: {NUM_DEVICES}")
    print(f"总批大小: {TOTAL_BATCH}")
    print(f"网络: {CHANNELS}ch x {NUM_BLOCKS}blocks")
    print(f"MCTS: {NUM_SIMULATIONS} sims")
    
    key = jax.random.PRNGKey(seed)
    key, subkey = jax.random.split(key)
    
    model = create_train_state(subkey, net, (BATCH_SIZE, 240, 10, 9), learning_rate)
    elo = jnp.float32(1500)
    
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
    
    if not DEBUG_MODE:
        model = replicate(model)
    
    print("开始训练...")
    
    while True:
        key, k0, k1 = jax.random.split(key, 3)
        
        if DEBUG_MODE:
            # 调试模式：单卡无编译
            data, stats = self_play(model, k0)
            n_batches = data.obs.shape[0]
            for i in range(n_batches):
                batch = jax.tree.map(lambda x: x[i], data)
                model, ploss, vloss = train_step(model, batch)
            model_single = model
            r, b, d = int(stats.red_wins), int(stats.black_wins), int(stats.draws)
            avg_len = float(stats.avg_length)
            ploss_val, vloss_val = float(ploss), float(vloss)
        else:
            # 正常模式：多卡并行
            keys = jax.random.split(k0, NUM_DEVICES)
            data, stats = self_play(model, keys)
            n_batches = data.obs.shape[1]
            for i in range(n_batches):
                batch = jax.tree.map(lambda x: x[:, i], data)
                model, ploss, vloss = train_step(model, batch)
            model_single = unreplicate(model)
            r, b, d = int(stats.red_wins[0]), int(stats.black_wins[0]), int(stats.draws[0])
            avg_len = float(stats.avg_length[0])
            ploss_val, vloss_val = float(ploss[0]), float(vloss[0])
        
        # 评估
        elo, win_rate = eval_step(model_single, model_single, elo, k1)
        total = r + b + d
        
        print(
            f"step={int(model_single.step)}, "
            f"ploss={ploss_val:.4f}, "
            f"vloss={vloss_val:.4f}, "
            f"elo={float(elo):.1f} | "
            f"自玩: {total}局 红{r} 黑{b} 和{d} 均长{avg_len:.0f}"
        )
        
        # 保存
        ckpt = {
            'params': model_single.params,
            'batch_stats': model_single.batch_stats,
            'opt_state': model_single.opt_state,
            'step': model_single.step,
            'elo': elo,
        }
        save_checkpoint(ckpt_path, ckpt, int(model_single.step), keep=ckpt_keep)


if __name__ == "__main__":
    main()

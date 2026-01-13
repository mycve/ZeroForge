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
    
    # 自对弈与搜索 (Gumbel 优势：低算力也能产生强信号)
    selfplay_batch_size: int = 512
    strong_simulations: int = 96
    weak_simulations: int = 32       # 提升学生算力，提供更有质量的辅助信号
    max_num_considered_actions: int = 16
    
    # 探索策略
    temperature_steps: int = 30
    temperature_initial: float = 1.0
    temperature_final: float = 0.01
    
    # 环境规则
    max_steps: int = 200
    max_no_capture_steps: int = 60
    repetition_threshold: int = 4
    perpetual_check_threshold: int = 6
    
    # ELO 评估 (优化：单链滑动窗口评估，确保 ELO 连续)
    eval_interval: int = 20
    eval_games: int = 64
    past_model_offset: int = 20      # 设置为与间隔相同，形成连续进化链

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

@jax.pmap
def selfplay(model, rng_key):
    """
    高性能自玩算子：重新使用 lax.scan 消除 Python 循环开销
    """
    params, batch_stats = model
    batch_size = config.selfplay_batch_size // num_devices
    
    rng_key, subkey = jax.random.split(rng_key)
    is_red_strong = jax.random.bernoulli(subkey, 0.5, shape=(batch_size,))
    
    def step_fn(state, key):
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
        
        k1, k2 = jax.random.split(key_search)
        policy_output_strong = mctx.gumbel_muzero_policy(
            params=model, rng_key=k1, root=root, recurrent_fn=recurrent_fn,
            num_simulations=config.strong_simulations,
            max_num_considered_actions=config.max_num_considered_actions,
            invalid_actions=~state.legal_action_mask,
        )
        policy_output_weak = mctx.gumbel_muzero_policy(
            params=model, rng_key=k2, root=root, recurrent_fn=recurrent_fn,
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

    state = jax.vmap(env.init)(jax.random.split(rng_key, batch_size))
    _, data = jax.lax.scan(step_fn, state, jax.random.split(rng_key, config.max_steps))
    return data

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
    
    # 策略损失 (导师 1.0 权重，学生 0.1 权重辅助训练)
    policy_ce = optax.softmax_cross_entropy(logits, policy_tgt)
    weights = jnp.where(is_strong, 1.0, 0.1)
    policy_loss = jnp.sum(policy_ce * weights) / jnp.maximum(jnp.sum(weights), 1.0)
    
    # 价值损失 (全学)
    value_loss = jnp.sum(optax.l2_loss(value, samples.value_tgt) * samples.mask) / jnp.maximum(jnp.sum(samples.mask), 1.0)
    
    # 熵损失 (保持一定的灵活性，但不宜过高)
    probs = jax.nn.softmax(logits)
    entropy = -jnp.sum(probs * jax.nn.log_softmax(logits), axis=-1)
    entropy_loss = -0.001 * jnp.mean(entropy)
    
    total_loss = policy_loss + 1.5 * value_loss + entropy_loss
    
    return total_loss, (new_batch_stats, policy_loss, value_loss)

@jax.pmap
def evaluate(model_red, model_black, rng_key):
    """高性能评估算子：重新使用 while_loop 跑满 GPU"""
    params_r, stats_r = model_red
    params_b, stats_b = model_black
    batch_size = config.eval_games // num_devices
    
    def evaluate_recurrent_fn(models, rng_key, action, state):
        m_red, m_black = models
        # 分别计算两个模型的输出
        out_red, next_state = recurrent_fn(m_red, rng_key, action, state)
        out_black, _ = recurrent_fn(m_black, rng_key, action, state)
        # 根据搜索模型索引进行合并
        use_red = state.search_model_index == 0
        out = jax.tree.map(
            lambda r, b: jnp.where(use_red[:, None] if r.ndim > 1 else use_red, r, b),
            out_red, out_black
        )
        return out, next_state
    
    def step_fn(state, key):
        # 在根节点确定当前搜索归属于哪个模型
        is_red = state.current_player == 0
        state = state.replace(search_model_index=jnp.where(is_red, 0, 1).astype(jnp.int32))
        
        obs = jax.vmap(env.observe)(state)
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
            recurrent_fn=evaluate_recurrent_fn,
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
        
    state, _, _ = jax.lax.while_loop(
        lambda args: (~jnp.all(args[1])) & (args[0].step_count[0] < config.max_steps), 
        body_fn, (state, terminated, rng_key)
    )
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
    
    # 1. 编译 Selfplay (现在是一个完整的并行算子)
    dummy_data = selfplay(model, dummy_keys)
    dummy_samples = compute_targets(dummy_data)
    
    # 2. 编译 Train Step
    batch_per_device = config.training_batch_size // num_devices
    dummy_batch = jax.tree.map(
        lambda x: x[:, 0, :batch_per_device].reshape((num_devices, batch_per_device) + x.shape[3:]), 
        dummy_samples
    )
    _ = train_step(model, opt_state, dummy_batch, dummy_keys)
    
    # 3. 编译 Evaluate
    _ = evaluate(model, model, dummy_keys)
    
    print(f"预编译完成，耗时: {time.time()-comp_st:.1f}s. 开始训练！")

    os.makedirs(config.ckpt_dir, exist_ok=True)
    writer = SummaryWriter(config.log_dir)
    
    iteration, frames = 0, 0
    start_time_total = time.time()
    
    while True:
        iteration += 1
        st = time.time()
        rng_key, sk1, sk2 = jax.random.split(rng_key, 3)
        data = selfplay(model, jax.random.split(sk1, num_devices))
        samples = compute_targets(data)
        
        data_np = jax.device_get(data)
        term = data_np.terminated
        winner = data_np.winner
        reasons = data_np.draw_reason
        
        # --- 增强版数据统计 ---
        first_term = (jnp.cumsum(term, axis=1) == 1) & term
        num_games = int(first_term.sum())
        
        # 1. 胜负平基础统计
        r_wins = int((first_term & (winner == 0)).sum())
        b_wins = int((first_term & (winner == 1)).sum())
        draws = int((first_term & (winner == -1)).sum())
        
        # 2. 和棋原因细分 (1=步数, 2=无吃子, 3=重复, 4=长将)
        d_max_steps = int((first_term & (reasons == 1)).sum())
        d_no_capture = int((first_term & (reasons == 2)).sum())
        d_repetition = int((first_term & (reasons == 3)).sum())
        d_perpetual = int((first_term & (reasons == 4)).sum())
        
        # 3. 对局长度
        # 找到每个对局结束时的步数
        game_lengths = jnp.where(term, jnp.arange(config.max_steps)[None, :, None], config.max_steps)
        final_lengths = jnp.min(game_lengths, axis=1)
        avg_length = float(jnp.mean(final_lengths))
        
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
        
        # --- 打印与日志 ---
        iter_time = time.time() - st
        fps = samples_flat.obs.shape[0] / iter_time
        
        print(f"iter={iteration:3d} | ploss={np.mean(policy_losses):.4f} vloss={np.mean(value_losses):.4f} | "
              f"len={avg_length:4.1f} fps={fps:4.0f} | "
              f"红{r_wins:3d} 黑{b_wins:3d} 和{draws:3d} (步{d_max_steps}/抓{d_no_capture}/复{d_repetition}/将{d_perpetual})")
        
        # TensorBoard 记录
        writer.add_scalar("train/policy_loss", np.mean(policy_losses), iteration)
        writer.add_scalar("train/value_loss", np.mean(value_losses), iteration)
        writer.add_scalar("stats/avg_game_length", avg_length, iteration)
        writer.add_scalar("stats/fps", fps, iteration)
        writer.add_scalar("stats/win_rate_red", r_wins / max(1, num_games), iteration)
        writer.add_scalar("stats/draw_rate", draws / max(1, num_games), iteration)
        
        if draws > 0:
            writer.add_scalar("draw_reasons/max_steps", d_max_steps / draws, iteration)
            writer.add_scalar("draw_reasons/no_capture", d_no_capture / draws, iteration)
            writer.add_scalar("draw_reasons/repetition", d_repetition / draws, iteration)
            writer.add_scalar("draw_reasons/perpetual_check", d_perpetual / draws, iteration)
        
        if iteration % 10 == 0:
            model_np = jax.device_get(jax.tree.map(lambda x: x[0], model))
            history_models[iteration] = model_np
            with open(os.path.join(config.ckpt_dir, f"ckpt_{iteration:06d}.pkl"), 'wb') as f:
                pickle.dump({'model': model_np, 'iteration': iteration, 'frames': frames}, f)
        
        if iteration % config.eval_interval == 0:
            past_iter = max(0, iteration - config.past_model_offset)
            past_model = jax.device_put_replicated(history_models[past_iter], devices)
            rng_key, sk5, sk6 = jax.random.split(rng_key, 3)
            # 恢复双边并行评估
            winners_r = evaluate(model, past_model, jax.random.split(sk5, num_devices))
            winners_b = evaluate(past_model, model, jax.random.split(sk6, num_devices))
            wr = (winners_r == 0).sum()
            wb = (winners_b == 1).sum()
            score = (wr + wb + 0.5 * (config.eval_games * 2 - wr - wb)) / (config.eval_games * 2)
            elo_diff = 400.0 * np.log10(score / (1.0 - score)) if 0 < score < 1 else (400 if score >= 1 else -400)
            iteration_elos[iteration] = iteration_elos.get(past_iter, 1500.0) + elo_diff
            print(f"评估 vs Iter {past_iter}: 胜率 {score:.2%}, ELO {iteration_elos[iteration]:.0f}")
            writer.add_scalar("eval/elo", iteration_elos[iteration], iteration)

if __name__ == "__main__":
    main()

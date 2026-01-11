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

from xiangqi.env import XiangqiEnv
from xiangqi.actions import rotate_action, ACTION_SPACE_SIZE
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
    
    # 探索策略 (新增加)
    temperature_steps: int = 30      # 前 N 步进行大温度采样
    temperature_initial: float = 1.0
    temperature_final: float = 0.01
    
    # 非对称自对弈 (破局核心)
    asymmetric_ratio: float = 0.5   # 50% 的对局使用非对称算力
    weak_simulations: int = 1       # 弱方模拟次数
    
    # 环境规则 (统一管理)
    max_steps: int = 400            # 总步数限制
    max_no_capture_steps: int = 60  # 无吃子步数限制 (强制进攻)
    repetition_threshold: int = 3   # 重复局面阈值
    perpetual_check_threshold: int = 6 # 连续将军阈值
    
    # 日志
    log_dir: str = "logs"
    
    # ELO 评估
    eval_interval: int = 20          # 每隔多少 iteration 评估一次
    eval_games: int = 64            # 评估对局数

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
    # 记录执行动作前的玩家，用于计算奖励
    prev_player = state.current_player
    
    # 执行动作
    state = jax.vmap(env.step)(state, action)
    
    # 获取观察（视角归一化已在 env.observe 中完成）
    obs = jax.vmap(env.observe)(state)
    (logits, value), _ = forward(params, batch_stats, obs)
    
    # --- 视角修正：将网络输出的视角 logits 转回真实坐标 ---
    # 如果执行完动作后的玩家是黑方 (1)，则网络输出的是基于旋转 180 度视角的 logits
    rotate_idx = jnp.arange(ACTION_SPACE_SIZE)
    rotated_idx = rotate_action(rotate_idx)
    
    logits = jnp.where(
        state.current_player[:, None] == 0,
        logits,
        logits[:, rotated_idx]
    )
    
    logits = logits - jnp.max(logits, axis=-1, keepdims=True)
    logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)
    
    # 奖励需要根据执行动作前的玩家来取
    reward = state.rewards[jnp.arange(state.rewards.shape[0]), prev_player]
    value = jnp.where(state.terminated, 0.0, value)
    discount = jnp.where(state.terminated, 0.0, -1.0)
    
    return mctx.RecurrentFnOutput(
        reward=reward, discount=discount, prior_logits=logits, value=value
    ), state

@jax.pmap
def evaluate(model_red, model_black, rng_key):
    params_r, stats_r = model_red
    params_b, stats_b = model_black
    batch_size = config.eval_games // num_devices
    
    def step_fn(state, key):
        key1, key2 = jax.random.split(key)
        obs = jax.vmap(env.observe)(state)
        
        # 根据当前玩家选择模型
        is_red = state.current_player == 0
        # 这里需要处理 batch，比较麻烦。简化方案：在 evaluate 中
        # 我们让 red 永远是 model_red, black 永远是 model_black
        
        # 获取红方视角的预测
        (logits_r, value_r), _ = forward(params_r, stats_r, obs)
        # 获取黑方视角的预测
        (logits_b, value_b), _ = forward(params_b, stats_b, obs)
        
        # 混合 logits 和 value
        logits = jnp.where(is_red[:, None], logits_r, logits_b)
        value = jnp.where(is_red, value_r, value_b)
        
        # 视角修正 (因为 env.observe 已经处理了视角，网络输出也是视角下的)
        rotate_idx = jnp.arange(ACTION_SPACE_SIZE)
        rotated_idx = rotate_action(rotate_idx)
        logits = jnp.where(is_red[:, None], logits, logits[:, rotated_idx])

        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)
        
        policy_output = mctx.gumbel_muzero_policy(
            params=(model_red, model_black), # 传个元组，recurrent_fn 里再分
            rng_key=key1,
            root=root,
            recurrent_fn=recurrent_fn_eval,
            num_simulations=config.num_simulations,
            max_num_considered_actions=config.max_num_considered_actions,
            invalid_actions=~state.legal_action_mask,
        )
        
        next_state = jax.vmap(env.step)(state, policy_output.action)
        return next_state, next_state.terminated
    
    def recurrent_fn_eval(models, rng_key, action, state):
        model_r, model_b = models
        prev_player = state.current_player
        state = jax.vmap(env.step)(state, action)
        obs = jax.vmap(env.observe)(state)
        
        is_red = state.current_player == 0
        (logits_r, value_r), _ = forward(model_r[0], model_r[1], obs)
        (logits_b, value_b), _ = forward(model_b[0], model_b[1], obs)
        
        logits = jnp.where(is_red[:, None], logits_r, logits_b)
        value = jnp.where(is_red, value_r, value_b)
        
        rotate_idx = jnp.arange(ACTION_SPACE_SIZE)
        rotated_idx = rotate_action(rotate_idx)
        logits = jnp.where(is_red[:, None], logits, logits[:, rotated_idx])
        
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)
        
        reward = state.rewards[jnp.arange(state.rewards.shape[0]), prev_player]
        value = jnp.where(state.terminated, 0.0, value)
        discount = jnp.where(state.terminated, 0.0, -1.0)
        
        return mctx.RecurrentFnOutput(
            reward=reward, discount=discount, prior_logits=logits, value=value
        ), state

    rng_key, subkey = jax.random.split(rng_key)
    keys = jax.random.split(subkey, batch_size)
    state = jax.vmap(env.init)(keys)
    
    # eval 不使用 scan 记录轨迹以节省显存，直接运行到结束
    def cond_fn(args):
        state, terminated, _ = args
        return ~jnp.all(terminated)
    
    def body_fn(args):
        state, terminated, key = args
        key, subkey = jax.random.split(key)
        new_state, new_term = step_fn(state, subkey)
        return new_state, terminated | new_term, key

    state, _, _ = jax.lax.while_loop(cond_fn, body_fn, (state, jnp.zeros(batch_size, dtype=jnp.bool_), rng_key))
    return state.winner

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
    power_mode: jnp.ndarray

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
        # 为 batch 里的每个环境分配独立的种子
        batch_keys = jax.random.split(key, batch_size)
        
        obs = jax.vmap(env.observe)(state)
        (logits, value), _ = forward(params, batch_stats, obs)
        
        # --- 视角修正 ---
        rotate_idx = jnp.arange(ACTION_SPACE_SIZE)
        rotated_idx = rotate_action(rotate_idx)
        logits = jnp.where(
            state.current_player[:, None] == 0,
            logits,
            logits[:, rotated_idx]
        )
        
        # 添加 Dirichlet 噪声
        def _add_noise(l, k):
            noise = jax.random.dirichlet(k, jnp.ones(ACTION_SPACE_SIZE) * 0.3)
            p = jax.nn.softmax(l)
            return jnp.log(0.75 * p + 0.25 * noise + 1e-10)
        
        noise_keys = jax.random.split(key, batch_size)
        logits = jax.vmap(_add_noise)(logits, noise_keys)

        # 准备 MCTS 根节点
        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)
        
        # 非对称算力判定
        is_weak_step = ((state.power_mode == 1) & (state.current_player == 1)) | \
                       ((state.power_mode == 2) & (state.current_player == 0))
        
        # 强/弱搜索使用不同的 batch_keys
        search_keys = jax.random.split(key, batch_size)
        
        policy_output_strong = mctx.gumbel_muzero_policy(
            params=model,
            rng_key=search_keys, # 关键：传入 batch 种子
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=config.num_simulations,
            max_num_considered_actions=config.max_num_considered_actions,
            invalid_actions=~state.legal_action_mask,
        )
        
        policy_output_weak = mctx.gumbel_muzero_policy(
            params=model,
            rng_key=search_keys,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=config.weak_simulations,
            max_num_considered_actions=config.max_num_considered_actions,
            invalid_actions=~state.legal_action_mask,
        )
        
        # 合并结果
        action_weights = jnp.where(is_weak_step[:, None], policy_output_weak.action_weights, policy_output_strong.action_weights)
        
        # --- 温度采样选择最终 Action ---
        # 这里的温度只影响动作选择，不干扰 MCTS 先验
        temp = jnp.where(state.step_count < config.temperature_steps, 
                         config.temperature_initial, 
                         config.temperature_final)
        
        def _sample_action(w, t, k):
            # 防止除零
            t = jnp.maximum(t, 1e-3)
            # 对 weights 进行温度缩放
            w_temp = jnp.power(w, 1.0 / t)
            w_temp = w_temp / jnp.sum(w_temp)
            return jax.random.choice(k, ACTION_SPACE_SIZE, p=w_temp)
        
        select_keys = jax.random.split(key, batch_size)
        action = jax.vmap(_sample_action)(action_weights, temp, select_keys)

        actor = state.current_player
        keys = jax.random.split(key, batch_size)
        
        # 执行一步动作
        next_state = jax.vmap(env.step)(state, action)
        
        # 记录数据
        normalized_action_weights = jnp.where(
            state.current_player[:, None] == 0,
            action_weights,
            action_weights[:, rotated_idx]
        )

        data = SelfplayOutput(
            obs=obs,
            action_weights=normalized_action_weights,
            reward=next_state.rewards[jnp.arange(batch_size), actor],
            terminated=next_state.terminated,
            discount=jnp.where(next_state.terminated, 0.0, -1.0),
            winner=next_state.winner,
            draw_reason=next_state.draw_reason,
            power_mode=state.power_mode,
        )
        
        # 检查是否需要 reset
        def _reset_fn(s, k):
            # reset 时重新随机算力模式
            new_key, _ = jax.random.split(k)
            new_mode_key, _ = jax.random.split(new_key)
            
            # 随机分配模式: 0 (正常), 1 (红强), 2 (黑强)
            r = jax.random.uniform(new_mode_key)
            new_mode = jnp.where(r < config.asymmetric_ratio / 2, 1, 
                                jnp.where(r < config.asymmetric_ratio, 2, 0))
            
            return jax.lax.cond(s.terminated, lambda: env.init(k, new_mode), lambda: s)
        
        next_state_reset = jax.vmap(_reset_fn)(next_state, keys)
        
        return next_state_reset, data
    
    rng_key, subkey = jax.random.split(rng_key)
    keys = jax.random.split(subkey, batch_size)
    
    # 初始算力模式分配
    mode_key, _ = jax.random.split(rng_key)
    r = jax.random.uniform(mode_key, (batch_size,))
    init_modes = jnp.where(r < config.asymmetric_ratio / 2, 1, 
                          jnp.where(r < config.asymmetric_ratio, 2, 0))
    
    state = jax.vmap(env.init)(keys, init_modes)
    
    _, data = jax.lax.scan(step_fn, state, jax.random.split(rng_key, config.max_steps))
    return data

@jax.pmap
def compute_targets(data: SelfplayOutput):
    batch_size = config.selfplay_batch_size // num_devices
    value_mask = jnp.cumsum(data.terminated[::-1, :], axis=0)[::-1, :] >= 1
    
    def body_fn(carry, i):
        ix = config.max_steps - i - 1
        v = data.reward[ix] + data.discount[ix] * carry
        return v, v
    
    _, value_tgt = jax.lax.scan(body_fn, jnp.zeros(batch_size), jnp.arange(config.max_steps))
    value_tgt = value_tgt[::-1, :]
    
    return Sample(obs=data.obs, policy_tgt=data.action_weights, value_tgt=value_tgt, mask=value_mask)

# ============================================================================
# 训练
# ============================================================================

def loss_fn(params, batch_stats, samples: Sample):
    (logits, value), new_batch_stats = forward(params, batch_stats, samples.obs, is_training=True)
    
    # 策略损失
    policy_loss = jnp.mean(optax.softmax_cross_entropy(logits, samples.policy_tgt))
    
    # 价值损失：修正稀释问题，只在有效步数上平均
    num_valid = jnp.maximum(jnp.sum(samples.mask), 1.0)
    value_loss = jnp.sum(optax.l2_loss(value, samples.value_tgt) * samples.mask) / num_valid
    
    # 总损失 (价值权重设为 0.5 稍微平衡一下，防止过早塌陷)
    total_loss = policy_loss + 0.5 * value_loss
    
    return total_loss, (new_batch_stats, policy_loss, value_loss)

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
    
    # 保存初始模型作为评估基准
    initial_model = jax.device_put_replicated(model, devices)
    
    # 优化器
    optimizer = optax.adam(config.learning_rate)
    opt_state = optimizer.init(model[0])
    
    # 复制到所有设备
    model = jax.device_put_replicated(model, devices)
    opt_state = jax.device_put_replicated(opt_state, devices)
    
    os.makedirs(config.ckpt_dir, exist_ok=True)
    os.makedirs(config.log_dir, exist_ok=True)
    writer = SummaryWriter(config.log_dir)
    
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
        
        # 统计（data 形状: num_devices, max_steps, batch_size）
        data_np = jax.device_get(data)
        term = data_np.terminated  # (D, T, B)
        winner = data_np.winner    # (D, T, B)
        p_mode = data_np.power_mode # (D, T, B)
        
        # 沿时间轴找第一次结束
        first_term = (jnp.cumsum(term, axis=1) == 1) & term
        r = int((first_term & (winner == 0)).sum())
        b = int((first_term & (winner == 1)).sum())
        d = int((first_term & (winner == -1)).sum())
        
        # 统计非对称局的胜率
        # 模式 1 (强红弱黑) 下红胜
        mode1_r_wins = int((first_term & (p_mode == 1) & (winner == 0)).sum())
        mode1_total = int((first_term & (p_mode == 1)).sum())
        # 模式 2 (弱红强黑) 下黑胜
        mode2_b_wins = int((first_term & (p_mode == 2) & (winner == 1)).sum())
        mode2_total = int((first_term & (p_mode == 2)).sum())
        
        # 细分和棋原因
        reason = data_np.draw_reason
        d_steps = int((first_term & (reason == 1)).sum())
        d_no_cap = int((first_term & (reason == 2)).sum())
        d_three = int((first_term & (reason == 3)).sum())
        d_perp = int((first_term & (reason == 4)).sum())
        
        # 计算平均对局长度
        game_lengths = jnp.argmax(first_term, axis=1)
        avg_steps = float(game_lengths[first_term.any(axis=1)].mean()) if first_term.any() else 0.0
        
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
        avg_ploss = sum(policy_losses)/len(policy_losses)
        avg_vloss = sum(value_losses)/len(value_losses)
        
        print(
            f"iter={iteration}, "
            f"ploss={avg_ploss:.4f}, "
            f"vloss={avg_vloss:.4f}, "
            f"frames={frames}, "
            f"time={et-st:.1f}s | "
            f"自玩: {total}局 红{r} 黑{b} 和{d}"
        )
        
        # 记录到 TensorBoard
        writer.add_scalar("train/policy_loss", avg_ploss, iteration)
        writer.add_scalar("train/value_loss", avg_vloss, iteration)
        writer.add_scalar("selfplay/red_wins", r, iteration)
        writer.add_scalar("selfplay/black_wins", b, iteration)
        writer.add_scalar("selfplay/draws", d, iteration)
        writer.add_scalar("selfplay/draw_max_steps", d_steps, iteration)
        writer.add_scalar("selfplay/draw_no_capture", d_no_cap, iteration)
        writer.add_scalar("selfplay/draw_threefold", d_three, iteration)
        writer.add_scalar("selfplay/draw_perpetual", d_perp, iteration)
        writer.add_scalar("selfplay/avg_steps", avg_steps, iteration)
        
        # 记录非对称局表现
        if mode1_total > 0:
            writer.add_scalar("asymmetric/strong_red_win_rate", mode1_r_wins / mode1_total, iteration)
        if mode2_total > 0:
            writer.add_scalar("asymmetric/strong_black_win_rate", mode2_b_wins / mode2_total, iteration)
        
        writer.add_scalar("stats/frames", frames, iteration)
        
        # 保存
        if iteration % 10 == 0:
            model_save = jax.tree.map(lambda x: x[0], model)
            ckpt = {'model': jax.device_get(model_save), 'iteration': iteration, 'frames': frames}
            path = os.path.join(config.ckpt_dir, f"ckpt_{iteration:06d}.pkl")
            with open(path, 'wb') as f:
                pickle.dump(ckpt, f)
            print(f"已保存: {path}")
            writer.flush()
            
        # 评估
        if iteration % config.eval_interval == 0:
            print(f"正在评估 (vs 随机基准)...", end="", flush=True)
            rng_key, subkey = jax.random.split(rng_key)
            eval_keys = jax.random.split(subkey, num_devices)
            
            # 1. 当前模型执红
            winners_r = evaluate(model, initial_model, eval_keys)
            # 2. 当前模型执黑
            winners_b = evaluate(initial_model, model, eval_keys)
            
            wr = int((winners_r == 0).sum())
            wb = int((winners_b == 1).sum())
            dr = int((winners_r == -1).sum())
            db = int((winners_b == -1).sum())
            total_wins = wr + wb
            total_draws = dr + db
            total_games = config.eval_games
            
            win_rate = total_wins / total_games
            print(f" 胜率: {win_rate:.2%}")
            writer.add_scalar("eval/win_rate_vs_random", win_rate, iteration)
            writer.add_scalar("eval/draw_rate_vs_random", total_draws / total_games, iteration)


if __name__ == "__main__":
    main()

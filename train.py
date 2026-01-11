#!/usr/bin/env python3
"""
ZeroForge - 中国象棋 Gumbel MuZero
多GPU训练脚本 - 编译缓存优化版

参考实现: https://github.com/zjjMaiMai/GumbelAlphaZero
"""

import os
import signal
import yaml

# ============================================================================
# 加载配置（必须在 import jax 之前，因为要设置环境变量）
# ============================================================================
import sys

def _load_config(path: str = "configs/default.yaml") -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)

# 支持命令行指定配置: python train.py configs/debug.yaml
_config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/default.yaml"
CONFIG = _load_config(_config_path)
print(f"加载配置: {_config_path}", flush=True)

import jax
import jax.numpy as jnp
import optax
import chex
import inspect
from functools import partial
from jax import lax
from typing import NamedTuple
from flax.training import checkpoints
from flax.training.train_state import TrainState
from tensorboardX import SummaryWriter
import time
from pathlib import Path

from xiangqi.env import XiangqiEnv
from networks.muzero import MuZeroNetwork, create_train_state
import mctx

# ============================================================================
# 全局常量（模块级别，编译时确定）
# ============================================================================

# 设备配置
NUM_DEVICES = jax.local_device_count()
GLOBAL_BATCH_SIZE = CONFIG["training"]["batch_size"]
GLOBAL_NUM_PARALLEL = CONFIG["self_play"]["num_parallel_games"]

# 检查配置
assert GLOBAL_BATCH_SIZE % NUM_DEVICES == 0, f"batch_size 必须能被 {NUM_DEVICES} 整除"
assert GLOBAL_NUM_PARALLEL % NUM_DEVICES == 0, f"num_parallel_games 必须能被 {NUM_DEVICES} 整除"

PER_DEVICE_BATCH = GLOBAL_BATCH_SIZE // NUM_DEVICES
PER_DEVICE_PARALLEL = GLOBAL_NUM_PARALLEL // NUM_DEVICES

# MCTS 配置
NUM_SIMULATIONS = CONFIG["mcts"]["num_simulations"]
MAX_ACTIONS = CONFIG["mcts"].get("max_num_considered_actions", 32)  # 增加到 32
DISCOUNT = CONFIG["mcts"].get("discount", 1.0)
# 探索参数：Dirichlet 噪声
ROOT_DIRICHLET_ALPHA = CONFIG["mcts"].get("root_dirichlet_alpha", 0.3)
ROOT_EXPLORATION_FRACTION = CONFIG["mcts"].get("root_exploration_fraction", 0.25)
MAX_STEPS = CONFIG["self_play"].get("max_steps", 200)
VALUE_LOSS_WEIGHT = CONFIG["training"].get("value_loss_weight", 1.0)

# 环境和网络（模块级别实例化）
ENV = XiangqiEnv()
_net_cfg = CONFIG["network"]
NETWORK = MuZeroNetwork(
    action_space_size=ENV.action_space_size,
    hidden_dim=_net_cfg.get("hidden_dim", 256),
    repr_blocks=_net_cfg.get("repr_blocks", 8),
    dyn_blocks=_net_cfg.get("dyn_blocks", 4),
    pred_blocks=_net_cfg.get("pred_blocks", 4),
    value_support_size=_net_cfg.get("value_support_size", 0),
    reward_support_size=_net_cfg.get("reward_support_size", 0),
)

# 全局退出标志
_SHOULD_EXIT = False

def _signal_handler(signum, frame):
    global _SHOULD_EXIT
    if _SHOULD_EXIT:
        print("\n强制退出...", flush=True)
        os._exit(1)
    else:
        _SHOULD_EXIT = True
        print("\n收到退出信号，当前迭代结束后保存退出...", flush=True)

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ============================================================================
# 数据结构
# ============================================================================

class Trajectory(NamedTuple):
    obs: chex.Array
    policy: chex.Array
    reward: chex.Array
    terminated: chex.Array
    player: chex.Array
    winner: chex.Array  # 新增：记录胜者


class Sample(NamedTuple):
    obs: chex.Array
    policy: chex.Array
    value: chex.Array
    mask: chex.Array


class SelfPlayStats(NamedTuple):
    """自对弈统计"""
    num_games: chex.Array      # 完成的游戏数
    red_wins: chex.Array       # 红方胜利数
    black_wins: chex.Array     # 黑方胜利数
    draws: chex.Array          # 和棋数
    avg_game_length: chex.Array  # 平均游戏长度
    valid_samples: chex.Array  # 有效样本数（非和棋）


# ============================================================================
# 自我对弈
# ============================================================================

def _recurrent_fn(params, rng_key, action, embedding):
    """MuZero 动态模型"""
    next_state, reward, logits, value = NETWORK.apply(
        params, embedding, action.astype(jnp.int32),
        method=NETWORK.recurrent_inference
    )
    return mctx.RecurrentFnOutput(
        reward=reward,
        discount=jnp.full_like(reward, DISCOUNT),
        prior_logits=logits,
        value=value,
    ), next_state


# 预先 vmap 环境函数
_v_init = jax.vmap(ENV.init)
_v_step = jax.vmap(ENV.step)
_v_observe = jax.vmap(ENV.observe)


# 温度采样参数
TEMPERATURE = CONFIG["mcts"].get("temperature", 1.0)  # 训练时温度（>1 更随机）
TEMP_THRESHOLD_STEPS = CONFIG["mcts"].get("temp_threshold_steps", 30)  # 前 N 步用高温度


def _selfplay_core(params, key):
    """单设备自我对弈（带温度采样）"""
    
    def selfplay_step(carry, rng_data):
        state, step_idx = carry  # step_idx: (B,) 每个游戏独立的步数计数器
        key = rng_data
        k_policy, k_sample, k_reset = jax.random.split(key, 3)
        
        obs = _v_observe(state)
        player = state.current_player
        
        output = NETWORK.apply(params, obs)
        root = mctx.RootFnOutput(
            prior_logits=output.policy_logits,
            value=output.value,
            embedding=output.hidden_state,
        )

        # Gumbel MuZero 策略
        policy_output = mctx.gumbel_muzero_policy(
            params=params,
            rng_key=k_policy,
            root=root,
            recurrent_fn=_recurrent_fn,
            num_simulations=NUM_SIMULATIONS,
            invalid_actions=~state.legal_action_mask,
            max_num_considered_actions=MAX_ACTIONS,
            gumbel_scale=1.0,
        )
        
        # === 温度采样：前 N 步用高温度增加探索，之后真正贪婪 ===
        weights = policy_output.action_weights  # (B, num_actions)
        
        # 每个游戏独立判断是否还在探索阶段
        use_temperature = step_idx < TEMP_THRESHOLD_STEPS  # (B,) bool
        
        # 对 log-probs 应用温度（仅用于 stochastic 采样）
        log_weights = jnp.log(weights + 1e-8)
        adjusted_logits = log_weights / TEMPERATURE  # 探索阶段用高温度
        
        # 确保非法动作不被选中
        adjusted_logits = jnp.where(state.legal_action_mask, adjusted_logits, -1e9)
        weights_masked = jnp.where(state.legal_action_mask, weights, -1e9)
        
        # 探索阶段：stochastic 采样；超过阈值：真正贪婪 (argmax)
        stochastic_action = jax.random.categorical(k_sample, adjusted_logits, axis=-1)  # (B,)
        greedy_action = jnp.argmax(weights_masked, axis=-1)  # (B,)
        sampled_action = jnp.where(use_temperature, stochastic_action, greedy_action)
        
        # 执行动作
        next_state = _v_step(state, sampled_action)
        reward = jax.vmap(lambda s, p: s.rewards[p])(next_state, player)

        reset_state = _v_init(jax.random.split(k_reset, PER_DEVICE_PARALLEL))
        term = next_state.terminated  # (B,)

        def _select(ns, rs):
            if ns.ndim == 0:
                return jnp.where(term, rs, ns)
            shape = (term.shape[0],) + (1,) * (ns.ndim - 1)
            return jnp.where(term.reshape(shape), rs, ns)

        state_after = jax.tree.map(_select, next_state, reset_state)
        
        # 每个游戏独立更新 step_idx：terminated 时重置为 0，否则 +1
        new_step_idx = jnp.where(term, 0, step_idx + 1)  # (B,)
        
        traj = Trajectory(
            obs=obs,
            policy=policy_output.action_weights,
            reward=reward,
            terminated=next_state.terminated,
            player=player,
            winner=next_state.winner,
        )
        return (state_after, new_step_idx), traj  # 返回 new_step_idx 而不是 step_idx+1
    
    k0, k1, k2 = jax.random.split(key, 3)
    state = _v_init(jax.random.split(k0, PER_DEVICE_PARALLEL))
    # step_idx 按游戏维度：每个并行游戏独立计数
    init_step_idx = jnp.zeros(PER_DEVICE_PARALLEL, dtype=jnp.int32)  # (B,)
    (_, _), traj = jax.lax.scan(selfplay_step, (state, init_step_idx), jax.random.split(k1, MAX_STEPS))
    
    # === 统计自对弈结果 ===
    # traj.terminated: (T, B), traj.winner: (T, B)
    # winner: 0=红胜, 1=黑胜, -1=和棋/未结束
    term_mask = traj.terminated.astype(jnp.float32)  # (T, B)
    
    # 每局游戏只统计第一次 terminated（游戏结束时刻）
    # 用 cumsum 找到每个并行游戏的第一次结束
    first_term = (jnp.cumsum(term_mask, axis=0) == 1) & (term_mask > 0)
    
    # 统计结果
    num_games = first_term.sum()
    red_wins = (first_term & (traj.winner == 0)).sum()
    black_wins = (first_term & (traj.winner == 1)).sum()
    draws = (first_term & (traj.winner == -1)).sum()
    
    # 计算平均游戏长度：找到每个游戏第一次 terminated 的位置
    # 为每个步骤标记索引
    step_indices = jnp.arange(MAX_STEPS)[:, None]  # (T, 1)
    # 每局游戏结束时的步数
    game_lengths = jnp.where(first_term, step_indices + 1, 0).sum(axis=0)  # (B,)
    # 只统计完成的游戏
    games_finished = first_term.any(axis=0)  # (B,)
    avg_game_length = jnp.where(
        games_finished.sum() > 0,
        game_lengths.sum() / (games_finished.sum() + 1e-8),
        jnp.float32(MAX_STEPS)
    )
    
    # 计算价值目标（reverse scan，从后往前传播）
    # 零和博弈：当前玩家价值 = reward - discount * 对手价值
    def compute_value(carry, t):
        term = t.terminated.astype(jnp.float32)  # (B,)
        
        # 1. 先 gate carry：episode 边界切断（terminated 时不传递后续 episode 的价值）
        gated_carry = carry * (1.0 - term)
        
        # 2. 计算当前步价值
        value = t.reward + DISCOUNT * gated_carry
        
        # 3. 翻转后传给前一步（对手视角的负价值）
        next_carry = -value
        
        return next_carry, value
    
    _, target_value = jax.lax.scan(compute_value, jnp.zeros(PER_DEVICE_PARALLEL), traj, reverse=True)
    
    # === 样本掩码 ===
    # 由于游戏会自动重置，所有样本都是有效的
    # 只有最后一步（如果游戏未结束）可能需要特殊处理
    # 简化起见：让所有样本都参与训练
    mask = jnp.zeros((MAX_STEPS, PER_DEVICE_PARALLEL), dtype=jnp.bool_)
    
    # 统计有效样本数
    valid_samples = (~mask).sum()  # 应该是 MAX_STEPS * PER_DEVICE_PARALLEL
    
    stats = SelfPlayStats(
        num_games=num_games,
        red_wins=red_wins,
        black_wins=black_wins,
        draws=draws,
        avg_game_length=avg_game_length,
        valid_samples=valid_samples,
    )
    
    sample = Sample(obs=traj.obs, policy=traj.policy, value=target_value, mask=mask)
    
    # (T, B, ...) -> (T*B, ...)
    sample = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), sample)
    
    # 打乱
    total = sample.obs.shape[0]
    perm = jax.random.permutation(k2, total)
    sample = jax.tree.map(lambda x: x[perm], sample)
    
    # 分成 minibatch: (n_batches, batch_size, ...)
    n_batches = total // PER_DEVICE_BATCH
    sample = jax.tree.map(
        lambda x: x[:n_batches * PER_DEVICE_BATCH].reshape(n_batches, PER_DEVICE_BATCH, *x.shape[1:]),
        sample
    )
    
    return sample, stats


@partial(jax.pmap, axis_name="devices")
def selfplay_pmap(params, key):
    """多设备并行自我对弈"""
    sample, stats = _selfplay_core(params, key)
    # 跨设备汇总统计
    stats = SelfPlayStats(
        num_games=jax.lax.psum(stats.num_games, axis_name="devices"),
        red_wins=jax.lax.psum(stats.red_wins, axis_name="devices"),
        black_wins=jax.lax.psum(stats.black_wins, axis_name="devices"),
        draws=jax.lax.psum(stats.draws, axis_name="devices"),
        avg_game_length=jax.lax.pmean(stats.avg_game_length, axis_name="devices"),
        valid_samples=jax.lax.psum(stats.valid_samples, axis_name="devices"),
    )
    return sample, stats


# ============================================================================
# 训练
# ============================================================================

class TrainMetrics(NamedTuple):
    """训练指标"""
    loss: chex.Array          # 总损失
    policy_loss: chex.Array   # 策略损失
    value_loss: chex.Array    # 价值损失
    policy_entropy: chex.Array  # 策略熵（越高表示预测越分散）
    policy_accuracy: chex.Array # 策略准确率（top-1）
    value_mae: chex.Array     # 价值平均绝对误差
    grad_norm: chex.Array     # 梯度范数


def _train_core(params, opt_state, batch):
    """训练核心逻辑"""
    def loss_fn(p):
        output = NETWORK.apply(p, batch.obs)
        
        # 策略损失
        policy_loss = optax.losses.softmax_cross_entropy(output.policy_logits, batch.policy)
        # 价值损失
        value_loss = optax.losses.squared_error(output.value, batch.value)
        
        # mask=True 表示无效
        valid = ~batch.mask
        mask_sum = valid.sum() + 1e-8
        policy_loss_mean = (policy_loss * valid).sum() / mask_sum
        value_loss_mean = (value_loss * valid).sum() / mask_sum
        
        total_loss = policy_loss_mean + VALUE_LOSS_WEIGHT * value_loss_mean
        
        # --- 统计指标 ---
        # 策略熵
        policy_probs = jax.nn.softmax(output.policy_logits, axis=-1)
        entropy = -jnp.sum(policy_probs * jnp.log(policy_probs + 1e-8), axis=-1)
        entropy_mean = (entropy * valid).sum() / mask_sum
        
        # 策略准确率
        pred_action = jnp.argmax(output.policy_logits, axis=-1)
        target_action = jnp.argmax(batch.policy, axis=-1)
        accuracy = (pred_action == target_action).astype(jnp.float32)
        accuracy_mean = (accuracy * valid).sum() / mask_sum
        
        # 价值平均绝对误差
        value_mae = jnp.abs(output.value - batch.value)
        value_mae_mean = (value_mae * valid).sum() / mask_sum
        
        aux = (policy_loss_mean, value_loss_mean, entropy_mean, accuracy_mean, value_mae_mean)
        return total_loss, aux
    
    (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    ploss, vloss, entropy, accuracy, value_mae = aux
    
    # 计算梯度范数
    grad_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree.leaves(grads)))
    
    return grads, TrainMetrics(
        loss=loss,
        policy_loss=ploss,
        value_loss=vloss,
        policy_entropy=entropy,
        policy_accuracy=accuracy,
        value_mae=value_mae,
        grad_norm=grad_norm,
    )


def _clip_grads_by_global_norm(grads, max_norm=1.0):
    """全局梯度范数裁剪（比逐元素裁剪更稳定）"""
    grad_norm = jnp.sqrt(sum(jnp.sum(g ** 2) for g in jax.tree.leaves(grads)))
    scale = jnp.minimum(1.0, max_norm / (grad_norm + 1e-8))
    return jax.tree.map(lambda g: g * scale, grads)


@partial(jax.pmap, axis_name="devices")
def train_step_pmap(state, batch):
    """多设备并行训练"""
    grads, metrics = _train_core(state.params, state.opt_state, batch)
    
    # 跨设备同步
    grads = jax.lax.pmean(grads, axis_name="devices")
    metrics = jax.tree.map(lambda x: jax.lax.pmean(x, axis_name="devices"), metrics)
    
    # 全局梯度范数裁剪（防止梯度爆炸）
    grads = _clip_grads_by_global_norm(grads, max_norm=1.0)
    new_state = state.apply_gradients(grads=grads)
    return new_state, metrics


# ============================================================================
# 评估
# ============================================================================

def _eval_core(new_params, old_params, key, num_games):
    """评估核心"""
    def play_step(carry, key):
        state, = carry
        obs = _v_observe(state)
        
        new_output = NETWORK.apply(new_params, obs)
        old_output = NETWORK.apply(old_params, obs)
        
        is_new_turn = (state.current_player == 0) == new_is_red
        logits = jnp.where(is_new_turn[:, None], new_output.policy_logits, old_output.policy_logits)
        logits = jnp.where(state.legal_action_mask, logits, -1e9)
        action = jnp.argmax(logits, axis=-1)
        
        next_state = _v_step(state, action)
        return (next_state,), None
    
    k0, k1, k2 = jax.random.split(key, 3)
    new_is_red = jax.random.bernoulli(k0, 0.5, (num_games,))
    state = _v_init(jax.random.split(k1, num_games))
    
    (final_state,), _ = jax.lax.scan(play_step, (state,), jax.random.split(k2, MAX_STEPS))
    
    new_wins = jnp.where(new_is_red, final_state.winner == 0, final_state.winner == 1).astype(jnp.float32)
    draws = (final_state.winner == -1).astype(jnp.float32)
    win_rate = (new_wins + 0.5 * draws).mean()
    
    return win_rate


NUM_EVAL_GAMES_PER_DEVICE = CONFIG["evaluation"].get("num_games", 100) // NUM_DEVICES

@partial(jax.pmap, axis_name="devices")
def eval_pmap(new_params, old_params, key):
    """多设备并行评估"""
    win_rate = _eval_core(new_params, old_params, key, NUM_EVAL_GAMES_PER_DEVICE)
    return jax.lax.pmean(win_rate, axis_name="devices")


# ============================================================================
# 辅助函数
# ============================================================================

def replicate(x):
    return jax.device_put_replicated(x, jax.local_devices()[:NUM_DEVICES])

def unreplicate(x):
    return jax.tree.map(lambda a: a[0], x)


# ============================================================================
# 检查点
# ============================================================================

class CheckpointManager:
    def __init__(self, checkpoint_dir: str, max_to_keep: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_to_keep = max_to_keep
        
    def save(self, state: TrainState, step: int, elo: float):
        ckpt = {"params": state.params, "opt_state": state.opt_state, "step": step, "elo": elo}
        checkpoints.save_checkpoint(str(self.checkpoint_dir), ckpt, step, keep=self.max_to_keep)
        print(f"检查点已保存: step={step}, elo={elo:.1f}", flush=True)
        
    def restore(self, state: TrainState) -> tuple:
        ckpt = checkpoints.restore_checkpoint(str(self.checkpoint_dir), None)
        if ckpt is None:
            return state, 0, 1500.0
        state = state.replace(params=ckpt["params"], opt_state=ckpt["opt_state"])
        return state, ckpt["step"], ckpt["elo"]


# ============================================================================
# 主函数
# ============================================================================

def main():
    global _SHOULD_EXIT
    
    print("=" * 60, flush=True)
    print("ZeroForge - 中国象棋 Gumbel MuZero", flush=True)
    print("=" * 60, flush=True)
    print(f"JAX 后端: {jax.default_backend()}", flush=True)
    print(f"设备数量: {NUM_DEVICES}", flush=True)
    print(f"编译缓存: {os.environ.get('JAX_COMPILATION_CACHE_DIR', '未启用')}", flush=True)
    print(f"每卡 batch: {PER_DEVICE_BATCH}, 每卡并行游戏: {PER_DEVICE_PARALLEL}", flush=True)
    print(f"MCTS 模拟: {NUM_SIMULATIONS}", flush=True)
    
    # 初始化状态
    key = jax.random.PRNGKey(CONFIG.get("seed", 42))
    key, init_key = jax.random.split(key)
    
    state = create_train_state(
        init_key, NETWORK,
        input_shape=(PER_DEVICE_BATCH, 240, 10, 9),
        learning_rate=CONFIG["training"]["learning_rate"],
    )
    
    # 检查点
    ckpt_manager = CheckpointManager(
        CONFIG["checkpoint"].get("checkpoint_dir", "checkpoints"),
        max_to_keep=CONFIG["checkpoint"].get("max_to_keep", 5),
    )
    state, step, elo = ckpt_manager.restore(state)
    
    if step > 0:
        print(f"从检查点恢复: step={step}, elo={elo:.1f}", flush=True)
    
    # TensorBoard
    writer = SummaryWriter(CONFIG["logging"].get("log_dir", "logs"))
    
    # 复制到所有设备
    state = replicate(state)
    old_params = unreplicate(state.params)
    
    # 配置
    save_interval = CONFIG["checkpoint"].get("save_interval", 1000)
    eval_interval = CONFIG["evaluation"].get("eval_interval", 5000)
    log_interval = CONFIG["logging"].get("console_interval", 100)
    num_training_steps = CONFIG["training"]["num_training_steps"]
    
    best_elo = elo
    
    print("开始训练...", flush=True)
    print("首次运行需要编译，后续运行会使用缓存...", flush=True)
    start_time = time.time()
    
    # 训练循环
    while step < num_training_steps and not _SHOULD_EXIT:
        iter_start = time.time()
        step_at_iter_start = step
        key, k0, k1 = jax.random.split(key, 3)
        
        device_keys = jax.random.split(k0, NUM_DEVICES)
        
        # 自我对弈
        data, sp_stats = selfplay_pmap(state.params, device_keys)
        n_batches = data.obs.shape[1]
        n_samples = NUM_DEVICES * n_batches * PER_DEVICE_BATCH
        
        # 训练
        for batch_idx in range(n_batches):
            batch = jax.tree.map(lambda x: x[:, batch_idx], data)
            state, metrics = train_step_pmap(state, batch)
        
        step += n_batches
        iter_time = time.time() - iter_start
        samples_per_sec = n_samples / iter_time
        
        # 日志
        def _crossed(prev, curr, interval):
            return interval > 0 and (prev // interval) != (curr // interval)
        
        if _crossed(step_at_iter_start, step, log_interval):
            # 从最后一个 batch 的指标取值（pmean 已同步，取第一个设备的值）
            m = {k: float(v[0]) for k, v in metrics._asdict().items()}
            # 自对弈统计（psum 已汇总，取第一个设备的值）
            sp = {k: float(v[0]) for k, v in sp_stats._asdict().items()}
            elapsed = time.time() - start_time
            
            # 计算胜率
            total_games = sp['num_games'] + 1e-8
            red_rate = sp['red_wins'] / total_games * 100
            black_rate = sp['black_wins'] / total_games * 100
            draw_rate = sp['draws'] / total_games * 100
            
            print(
                f"step={step}, loss={m['loss']:.4f}, ploss={m['policy_loss']:.4f}, vloss={m['value_loss']:.4f}, "
                f"acc={m['policy_accuracy']*100:.1f}%, ent={m['policy_entropy']:.2f}, grad={m['grad_norm']:.2f}, "
                f"samples={n_samples} ({samples_per_sec:.0f}/s), elapsed={elapsed/60:.1f}min",
                flush=True
            )
            valid_ratio = sp['valid_samples'] / (n_samples + 1e-8) * 100
            print(
                f"  自对弈: {int(sp['num_games'])}局, "
                f"红胜={red_rate:.1f}%, 黑胜={black_rate:.1f}%, 和棋={draw_rate:.1f}%, "
                f"平均步数={sp['avg_game_length']:.1f}, "
                f"有效样本={int(sp['valid_samples'])} ({valid_ratio:.1f}%)",
                flush=True
            )
            
            # TensorBoard 记录所有指标
            writer.add_scalar("loss/total", m['loss'], step)
            writer.add_scalar("loss/policy", m['policy_loss'], step)
            writer.add_scalar("loss/value", m['value_loss'], step)
            writer.add_scalar("train/policy_entropy", m['policy_entropy'], step)
            writer.add_scalar("train/policy_accuracy", m['policy_accuracy'], step)
            writer.add_scalar("train/value_mae", m['value_mae'], step)
            writer.add_scalar("train/grad_norm", m['grad_norm'], step)
            writer.add_scalar("perf/samples_per_sec", samples_per_sec, step)
            # 自对弈统计
            writer.add_scalar("selfplay/num_games", sp['num_games'], step)
            writer.add_scalar("selfplay/red_win_rate", red_rate / 100, step)
            writer.add_scalar("selfplay/black_win_rate", black_rate / 100, step)
            writer.add_scalar("selfplay/draw_rate", draw_rate / 100, step)
            writer.add_scalar("selfplay/avg_game_length", sp['avg_game_length'], step)
        
        # 评估
        if _crossed(step_at_iter_start, step, eval_interval):
            eval_keys = jax.random.split(k1, NUM_DEVICES)
            old_params_rep = replicate(old_params)
            
            win_rate = eval_pmap(state.params, old_params_rep, eval_keys)
            win_rate = float(win_rate[0])
            
            elo = elo + 32 * (win_rate - 0.5) * 100
            
            print(f"评估: elo={elo:.1f}, win_rate={win_rate:.2%}", flush=True)
            writer.add_scalar("eval/elo", elo, step)
            writer.add_scalar("eval/win_rate", win_rate, step)
            
            if elo > best_elo:
                best_elo = elo
                old_params = unreplicate(state.params)
                print(f"新最佳模型! elo={elo:.1f}", flush=True)
        
        # 保存
        if _crossed(step_at_iter_start, step, save_interval):
            ckpt_manager.save(unreplicate(state), step, elo)
    
    # 最终保存
    ckpt_manager.save(unreplicate(state), step, elo)
    writer.close()
    
    total_time = time.time() - start_time
    print(f"{'训练被中断' if _SHOULD_EXIT else '训练完成'}! 总时间: {total_time/3600:.1f}小时", flush=True)


if __name__ == "__main__":
    main()

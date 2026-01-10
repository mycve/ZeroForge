#!/usr/bin/env python3
"""
ZeroForge - 中国象棋 Gumbel MuZero
多GPU训练脚本

参考实现: https://github.com/zjjMaiMai/GumbelAlphaZero
"""

import os
import signal

# ============================================================================
# 启用 JAX 持久化编译缓存（必须在 import jax 之前设置）
# ============================================================================
os.environ["JAX_COMPILATION_CACHE_DIR"] = os.path.join(os.path.dirname(__file__), ".jax_cache")
os.environ["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"] = "0"  # 缓存所有编译结果
os.environ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"] = "0"  # 缓存所有编译结果

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
import yaml
from pathlib import Path

from xiangqi.env import XiangqiEnv
from networks.muzero import MuZeroNetwork, create_train_state
import mctx

# 全局退出标志
_SHOULD_EXIT = False

def _signal_handler(signum, frame):
    global _SHOULD_EXIT
    if _SHOULD_EXIT:
        print("\n强制退出...", flush=True)
        os._exit(1)
    else:
        _SHOULD_EXIT = True
        print("\n收到退出信号，将在当前迭代结束后保存并退出...", flush=True)

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


# ============================================================================
# 配置
# ============================================================================

def load_config(path: str = "configs/default.yaml") -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


# ============================================================================
# 数据结构
# ============================================================================

class Trajectory(NamedTuple):
    obs: chex.Array
    policy: chex.Array
    reward: chex.Array
    terminated: chex.Array
    player: chex.Array


class Sample(NamedTuple):
    obs: chex.Array
    policy: chex.Array
    value: chex.Array
    mask: chex.Array


# ============================================================================
# 全局变量（会在 main 中初始化）
# ============================================================================

_ENV = None
_NETWORK = None
_CONFIG = None
_NUM_DEVICES = None
_PER_DEVICE_BATCH = None
_PER_DEVICE_PARALLEL = None


def _init_globals(env, network, config, n_devices, per_device_batch, per_device_parallel):
    """初始化全局变量（必须在 pmap 函数定义之前调用）"""
    global _ENV, _NETWORK, _CONFIG, _NUM_DEVICES, _PER_DEVICE_BATCH, _PER_DEVICE_PARALLEL
    _ENV = env
    _NETWORK = network
    _CONFIG = config
    _NUM_DEVICES = n_devices
    _PER_DEVICE_BATCH = per_device_batch
    _PER_DEVICE_PARALLEL = per_device_parallel


# ============================================================================
# 自我对弈（单设备核心逻辑）
# ============================================================================

def _selfplay_single_device(params, key):
    """单设备自我对弈核心逻辑"""
    num_parallel = _PER_DEVICE_PARALLEL
    num_steps = _CONFIG["self_play"].get("max_steps", 200)
    num_simulations = _CONFIG["mcts"]["num_simulations"]
    max_actions = _CONFIG["mcts"].get("max_num_considered_actions", 16)
    discount = _CONFIG["mcts"].get("discount", 1.0)
    batch_size = _PER_DEVICE_BATCH
    
    # 检查 mctx 版本
    try:
        sig = inspect.signature(mctx.gumbel_muzero_policy)
        supports_max_actions = "max_num_considered_actions" in sig.parameters
    except:
        supports_max_actions = False
    
    def recurrent_fn(params, rng_key, action, embedding):
        next_state, reward, logits, value = _NETWORK.apply(
            params, embedding, action.astype(jnp.int32),
            method=_NETWORK.recurrent_inference
        )
        return mctx.RecurrentFnOutput(
            reward=reward,
            discount=jnp.full_like(reward, discount),
            prior_logits=logits,
            value=value,
        ), next_state
    
    v_init = jax.vmap(_ENV.init)
    v_step = jax.vmap(_ENV.step)
    v_observe = jax.vmap(_ENV.observe)
    
    def selfplay_step(state, key):
        k_policy, k_reset = jax.random.split(key)
        
        obs = v_observe(state)
        player = state.current_player
        
        output = _NETWORK.apply(params, obs)
        root = mctx.RootFnOutput(
            prior_logits=output.policy_logits,
            value=output.value,
            embedding=output.hidden_state,
        )

        if supports_max_actions:
            policy_output = mctx.gumbel_muzero_policy(
                params=params,
                rng_key=k_policy,
                root=root,
                recurrent_fn=recurrent_fn,
                num_simulations=num_simulations,
                invalid_actions=~state.legal_action_mask,
                max_num_considered_actions=max_actions,
            )
        else:
            policy_output = mctx.gumbel_muzero_policy(
                params=params,
                rng_key=k_policy,
                root=root,
                recurrent_fn=recurrent_fn,
                num_simulations=num_simulations,
                invalid_actions=~state.legal_action_mask,
            )
        
        next_state = v_step(state, policy_output.action)
        reward = jax.vmap(lambda s, p: s.rewards[p])(next_state, player)

        reset_state = v_init(jax.random.split(k_reset, num_parallel))
        term = next_state.terminated

        def _select(ns, rs):
            if ns.ndim == 0:
                return jnp.where(term, rs, ns)
            shape = (term.shape[0],) + (1,) * (ns.ndim - 1)
            return jnp.where(term.reshape(shape), rs, ns)

        state_after = jax.tree.map(_select, next_state, reset_state)
        
        traj = Trajectory(
            obs=obs,
            policy=policy_output.action_weights,
            reward=reward,
            terminated=next_state.terminated,
            player=player,
        )
        return state_after, traj
    
    k0, k1, k2 = jax.random.split(key, 3)
    state = v_init(jax.random.split(k0, num_parallel))
    _, traj = jax.lax.scan(selfplay_step, state, jax.random.split(k1, num_steps))
    
    # 计算价值目标
    def compute_value(carry, t):
        value = t.reward + discount * carry * (1 - t.terminated.astype(jnp.float32))
        return -value, value
    
    _, target_value = jax.lax.scan(compute_value, jnp.zeros(num_parallel), traj, reverse=True)
    
    # 掩码
    mask = jnp.flip(jnp.cumsum(jnp.flip(traj.terminated, 0), 0), 0) >= 1
    
    sample = Sample(
        obs=traj.obs,
        policy=traj.policy,
        value=target_value,
        mask=mask,
    )
    
    # (T, B, ...) -> (T*B, ...)
    sample = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), sample)
    
    # 打乱
    total = sample.obs.shape[0]
    perm = jax.random.permutation(k2, total)
    sample = jax.tree.map(lambda x: x[perm], sample)
    
    # 分成 minibatch: (n_batches, batch_size, ...)
    n_batches = total // batch_size
    sample = jax.tree.map(
        lambda x: x[:n_batches * batch_size].reshape(n_batches, batch_size, *x.shape[1:]),
        sample
    )
    
    return sample


@partial(jax.pmap, axis_name="devices")
def selfplay_pmap(params, key):
    """多设备并行自我对弈"""
    return _selfplay_single_device(params, key)


# ============================================================================
# 训练步骤
# ============================================================================

def _train_step_core(params, opt_state, batch):
    """训练核心逻辑"""
    value_loss_weight = _CONFIG["training"].get("value_loss_weight", 1.0)
    
    def loss_fn(p):
        output = _NETWORK.apply(p, batch.obs)
        
        policy_loss = optax.losses.softmax_cross_entropy(
            output.policy_logits, batch.policy
        )
        value_loss = optax.losses.squared_error(output.value, batch.value)
        
        # 用 mask 加权
        mask = ~batch.mask  # mask=True 表示无效，取反
        mask_sum = mask.sum() + 1e-8
        policy_loss = (policy_loss * mask).sum() / mask_sum
        value_loss = (value_loss * mask).sum() / mask_sum
        
        total_loss = policy_loss + value_loss_weight * value_loss
        return total_loss, (policy_loss, value_loss)
    
    (loss, (ploss, vloss)), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    return grads, loss, ploss, vloss


@partial(jax.pmap, axis_name="devices")
def train_step_pmap(state, batch):
    """多设备并行训练"""
    grads, loss, ploss, vloss = _train_step_core(state.params, state.opt_state, batch)
    
    # 跨设备同步梯度
    grads = jax.lax.pmean(grads, axis_name="devices")
    loss = jax.lax.pmean(loss, axis_name="devices")
    ploss = jax.lax.pmean(ploss, axis_name="devices")
    vloss = jax.lax.pmean(vloss, axis_name="devices")
    
    # 梯度裁剪
    grads = jax.tree.map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
    
    new_state = state.apply_gradients(grads=grads)
    return new_state, loss, ploss, vloss


# ============================================================================
# 评估
# ============================================================================

def _eval_core(new_params, old_params, key, num_games):
    """评估核心逻辑"""
    max_steps = _CONFIG["self_play"].get("max_steps", 200)
    
    v_init = jax.vmap(_ENV.init)
    v_step = jax.vmap(_ENV.step)
    v_observe = jax.vmap(_ENV.observe)
    
    k0, k1, k2 = jax.random.split(key, 3)
    new_is_red = jax.random.bernoulli(k0, 0.5, (num_games,))
    state = v_init(jax.random.split(k1, num_games))
    
    def play_step(carry, key):
        state, = carry
        obs = v_observe(state)
        
        new_output = _NETWORK.apply(new_params, obs)
        old_output = _NETWORK.apply(old_params, obs)
        
        is_new_turn = (state.current_player == 0) == new_is_red
        logits = jnp.where(
            is_new_turn[:, None],
            new_output.policy_logits,
            old_output.policy_logits,
        )
        
        logits = jnp.where(state.legal_action_mask, logits, -1e9)
        action = jnp.argmax(logits, axis=-1)
        
        next_state = v_step(state, action)
        return (next_state,), None
    
    (final_state,), _ = jax.lax.scan(play_step, (state,), jax.random.split(k2, max_steps))
    
    new_wins = jnp.where(
        new_is_red,
        final_state.winner == 0,
        final_state.winner == 1,
    ).astype(jnp.float32)
    
    draws = (final_state.winner == -1).astype(jnp.float32)
    win_rate = (new_wins + 0.5 * draws).mean()
    
    return win_rate


@partial(jax.pmap, axis_name="devices")
def eval_pmap(new_params, old_params, key):
    """多设备并行评估"""
    num_games = _CONFIG["evaluation"].get("num_games", 100) // _NUM_DEVICES
    win_rate = _eval_core(new_params, old_params, key, num_games)
    win_rate = jax.lax.pmean(win_rate, axis_name="devices")
    return win_rate


# ============================================================================
# 辅助函数
# ============================================================================

def replicate(x):
    """复制到所有设备"""
    return jax.device_put_replicated(x, jax.local_devices()[:_NUM_DEVICES])


def unreplicate(x):
    """从第一个设备取值"""
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
    
    config = load_config()
    
    # 设备配置
    devices = jax.local_devices()
    n_devices = len(devices)
    
    print("=" * 60, flush=True)
    print("ZeroForge - 中国象棋 Gumbel MuZero", flush=True)
    print("=" * 60, flush=True)
    print(f"JAX 后端: {jax.default_backend()}", flush=True)
    print(f"设备数量: {n_devices}", flush=True)
    print(f"编译缓存: {os.environ.get('JAX_COMPILATION_CACHE_DIR', '未启用')}", flush=True)
    
    # 检查配置
    global_batch = config["training"]["batch_size"]
    global_parallel = config["self_play"]["num_parallel_games"]
    
    if global_batch % n_devices != 0:
        raise ValueError(f"batch_size={global_batch} 必须能被 {n_devices} 整除")
    if global_parallel % n_devices != 0:
        raise ValueError(f"num_parallel_games={global_parallel} 必须能被 {n_devices} 整除")
    
    per_device_batch = global_batch // n_devices
    per_device_parallel = global_parallel // n_devices
    
    print(f"每卡 batch: {per_device_batch}, 每卡并行游戏: {per_device_parallel}", flush=True)
    print(f"MCTS 模拟: {config['mcts']['num_simulations']}", flush=True)
    
    # 初始化
    env = XiangqiEnv()
    network = MuZeroNetwork(
        action_space_size=env.action_space_size,
        hidden_dim=config["network"]["hidden_dim"],
    )
    
    # 初始化全局变量（必须在 pmap 函数调用之前）
    _init_globals(env, network, config, n_devices, per_device_batch, per_device_parallel)
    
    key = jax.random.PRNGKey(config.get("seed", 42))
    key, init_key = jax.random.split(key)
    
    state = create_train_state(
        init_key, network,
        input_shape=(per_device_batch, 240, 10, 9),
        learning_rate=config["training"]["learning_rate"],
    )
    
    # 检查点
    ckpt_manager = CheckpointManager(
        config["checkpoint"].get("checkpoint_dir", "checkpoints"),
        max_to_keep=config["checkpoint"].get("max_to_keep", 5),
    )
    state, step, elo = ckpt_manager.restore(state)
    
    if step > 0:
        print(f"从检查点恢复: step={step}, elo={elo:.1f}", flush=True)
    
    # TensorBoard
    writer = SummaryWriter(config["logging"].get("log_dir", "logs"))
    
    # 复制到所有设备
    state = replicate(state)
    old_params = unreplicate(state.params)
    
    # 配置
    save_interval = config["checkpoint"].get("save_interval", 1000)
    eval_interval = config["evaluation"].get("eval_interval", 5000)
    log_interval = config["logging"].get("console_interval", 100)
    num_training_steps = config["training"]["num_training_steps"]
    
    best_elo = elo
    
    print("开始训练...", flush=True)
    print("首次运行需要 JIT 编译，请耐心等待...", flush=True)
    start_time = time.time()
    
    # ====================================================================
    # 训练循环
    # ====================================================================
    while step < num_training_steps and not _SHOULD_EXIT:
        iter_start = time.time()
        step_at_iter_start = step
        key, k0, k1 = jax.random.split(key, 3)
        
        # 为每个设备生成随机数
        device_keys = jax.random.split(k0, n_devices)
        
        # ----- 多卡自我对弈 -----
        # data: (n_devices, n_batches, batch_size, ...)
        data = selfplay_pmap(state.params, device_keys)
        n_batches = data.obs.shape[1]
        n_samples = n_devices * n_batches * per_device_batch
        
        # ----- 多卡训练 (Python for 循环，每步 pmap) -----
        for batch_idx in range(n_batches):
            # 提取当前 batch: (n_devices, batch_size, ...)
            batch = jax.tree.map(lambda x: x[:, batch_idx], data)
            state, loss, ploss, vloss = train_step_pmap(state, batch)
        
        step += n_batches
        iter_time = time.time() - iter_start
        
        # ----- 日志 -----
        def _crossed(prev, curr, interval):
            return interval > 0 and (prev // interval) != (curr // interval)
        
        if _crossed(step_at_iter_start, step, log_interval):
            # 只在需要时同步
            loss_val = float(loss[0])
            ploss_val = float(ploss[0])
            vloss_val = float(vloss[0])
            elapsed = time.time() - start_time
            
            print(
                f"step={step}, loss={loss_val:.4f}, ploss={ploss_val:.4f}, vloss={vloss_val:.4f}, "
                f"samples={n_samples}, time={iter_time:.1f}s, elapsed={elapsed/60:.1f}min",
                flush=True
            )
            
            writer.add_scalar("loss/total", loss_val, step)
            writer.add_scalar("loss/policy", ploss_val, step)
            writer.add_scalar("loss/value", vloss_val, step)
        
        # ----- ELO 评估 -----
        if _crossed(step_at_iter_start, step, eval_interval):
            eval_keys = jax.random.split(k1, n_devices)
            current_params = state.params
            old_params_rep = replicate(old_params)
            
            win_rate = eval_pmap(current_params, old_params_rep, eval_keys)
            win_rate = float(win_rate[0])
            
            # 更新 ELO
            elo = elo + 32 * (win_rate - 0.5) * 100
            
            print(f"评估: elo={elo:.1f}, win_rate={win_rate:.2%}", flush=True)
            writer.add_scalar("eval/elo", elo, step)
            writer.add_scalar("eval/win_rate", win_rate, step)
            
            if elo > best_elo:
                best_elo = elo
                old_params = unreplicate(current_params)
                print(f"新最佳模型! elo={elo:.1f}", flush=True)
        
        # ----- 保存 -----
        if _crossed(step_at_iter_start, step, save_interval):
            ckpt_manager.save(unreplicate(state), step, elo)
    
    # 最终保存
    ckpt_manager.save(unreplicate(state), step, elo)
    writer.close()
    
    total_time = time.time() - start_time
    if _SHOULD_EXIT:
        print(f"训练被中断，已保存。总时间: {total_time/3600:.1f}小时", flush=True)
    else:
        print(f"训练完成! 总时间: {total_time/3600:.1f}小时", flush=True)


if __name__ == "__main__":
    main()

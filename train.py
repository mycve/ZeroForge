#!/usr/bin/env python3
"""
ZeroForge - 中国象棋 Gumbel MuZero
训练脚本：自我对弈 + 训练 + ELO 评估 + 断点继续 + TensorBoard

支持多卡数据并行 (pmap)
"""

import os
import signal
import jax
import jax.numpy as jnp
import optax
import chex
import inspect
from flax import jax_utils
from jax import lax
from typing import NamedTuple
from flax.training import checkpoints, train_state
from flax.training.train_state import TrainState
import orbax.checkpoint as ocp
from tensorboardX import SummaryWriter
import logging
import time
import yaml
from pathlib import Path

# 全局标志，用于优雅退出
_SHOULD_EXIT = False

def _signal_handler(signum, frame):
    """处理 Ctrl+C 信号"""
    global _SHOULD_EXIT
    if _SHOULD_EXIT:
        print("\n强制退出...", flush=True)
        os._exit(1)
    else:
        _SHOULD_EXIT = True
        print("\n收到退出信号，将在当前迭代结束后保存并退出...", flush=True)
        print("再按一次 Ctrl+C 强制退出", flush=True)

signal.signal(signal.SIGINT, _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)

from xiangqi.env import XiangqiEnv
from networks.muzero import MuZeroNetwork, create_train_state
import mctx

logging.basicConfig(
    level=logging.INFO, 
    format="[%(asctime)s] %(message)s",
    handlers=[logging.StreamHandler()]  # 强制输出到 stderr
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 确保日志立即输出
import sys
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
sys.stderr.reconfigure(line_buffering=True) if hasattr(sys.stderr, 'reconfigure') else None


# ============================================================================
# 配置
# ============================================================================

def load_config(path: str = "configs/default.yaml") -> dict:
    """加载配置"""
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


# ============================================================================
# 自我对弈 (为 pmap 设计，不加 @jax.jit)
# ============================================================================

def make_selfplay_fn(env, network, config: dict):
    """
    创建自我对弈函数（会被 pmap 包裹，不要加 @jax.jit）
    """
    
    num_parallel = config["self_play"]["num_parallel_games"]
    num_steps = config["self_play"].get("max_steps", 200)
    num_simulations = config["mcts"]["num_simulations"]
    max_actions = config["mcts"].get("max_num_considered_actions", 16)
    discount = config["mcts"].get("discount", 1.0)
    batch_size = config["training"]["batch_size"]

    # 兼容不同 mctx 版本
    try:
        _gumbel_sig = inspect.signature(mctx.gumbel_muzero_policy)
        supports_max_actions = "max_num_considered_actions" in _gumbel_sig.parameters
    except Exception as e:
        print(f"[警告] 无法检查 mctx.gumbel_muzero_policy 参数签名: {e}", flush=True)
        supports_max_actions = False
    
    def recurrent_fn(params, rng_key, action, embedding):
        """MuZero 动态模型"""
        next_state, reward, logits, value = network.apply(
            params, embedding, action.astype(jnp.int32),
            method=network.recurrent_inference
        )
        return mctx.RecurrentFnOutput(
            reward=reward,
            discount=jnp.full_like(reward, discount),
            prior_logits=logits,
            value=value,
        ), next_state
    
    v_init = jax.vmap(env.init)
    v_step = jax.vmap(env.step)
    v_observe = jax.vmap(env.observe)
    
    # 注意：不加 @jax.jit，因为会被 pmap 包裹
    def selfplay_fn(params, key):
        """单设备的自我对弈"""
        k0, k1, k2 = jax.random.split(key, 3)
        
        state = v_init(jax.random.split(k0, num_parallel))
        
        def selfplay_step(state, key):
            k_policy, k_reset = jax.random.split(key)
            
            obs = v_observe(state)
            player = state.current_player
            
            output = network.apply(params, obs)
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
        
        _, traj = jax.lax.scan(
            selfplay_step,
            state,
            jax.random.split(k1, num_steps),
        )
        
        def compute_value(carry, t):
            value = t.reward + discount * carry * (1 - t.terminated.astype(jnp.float32))
            return -value, value
        
        _, target_value = jax.lax.scan(
            compute_value,
            jnp.zeros(num_parallel),
            traj,
            reverse=True,
        )
        
        mask = jnp.cumsum(traj.terminated[::-1], axis=0)[::-1] == 0
        
        total_samples = num_steps * num_parallel
        sample = Sample(
            obs=traj.obs.reshape(total_samples, *traj.obs.shape[2:]),
            policy=traj.policy.reshape(total_samples, *traj.policy.shape[2:]),
            value=target_value.reshape(total_samples),
        )
        mask = mask.reshape(total_samples)
        
        perm = jax.random.permutation(k2, total_samples)
        sample = jax.tree.map(lambda x: x[perm], sample)
        mask = mask[perm]
        
        n_batches = total_samples // batch_size
        sample = jax.tree.map(
            lambda x: x[:n_batches * batch_size].reshape(n_batches, batch_size, *x.shape[1:]),
            sample
        )
        batch_mask = mask[:n_batches * batch_size].reshape(n_batches, batch_size)
        
        return sample, batch_mask
    
    return selfplay_fn


# ============================================================================
# 训练步骤
# ============================================================================

def make_train_step(network, config: dict):
    """创建训练步骤函数"""
    
    value_loss_weight = config["training"].get("value_loss_weight", 1.0)
    
    def train_step(state, batch, mask):
        """单步训练（会被 pmap 包裹）"""
        def loss_fn(params):
            output = network.apply(params, batch.obs)
            
            policy_loss_per_sample = optax.losses.softmax_cross_entropy(
                output.policy_logits, batch.policy
            )
            value_loss_per_sample = optax.losses.squared_error(
                output.value, batch.value
            )
            
            mask_sum = mask.sum() + 1e-8
            policy_loss = (policy_loss_per_sample * mask).sum() / mask_sum
            value_loss = (value_loss_per_sample * mask).sum() / mask_sum
            
            total_loss = policy_loss + value_loss_weight * value_loss
            return total_loss, (policy_loss, value_loss)
        
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (ploss, vloss)), grads = grad_fn(state.params)
        
        # 多卡：跨设备平均梯度
        grads = lax.pmean(grads, axis_name="devices")
        loss = lax.pmean(loss, axis_name="devices")
        ploss = lax.pmean(ploss, axis_name="devices")
        vloss = lax.pmean(vloss, axis_name="devices")

        grads = jax.tree.map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
        
        state = state.apply_gradients(grads=grads)
        return state, {"loss": loss, "policy_loss": ploss, "value_loss": vloss}
    
    return train_step


# ============================================================================
# ELO 评估
# ============================================================================

def make_eval_fn(env, network, config: dict):
    """创建评估函数"""
    
    num_games = config["evaluation"].get("num_games", 100)
    max_steps = config["self_play"].get("max_steps", 200)
    
    v_init = jax.vmap(env.init)
    v_step = jax.vmap(env.step)
    v_observe = jax.vmap(env.observe)
    
    @jax.jit
    def eval_fn(new_params, old_params, elo, key):
        k0, k1, k2 = jax.random.split(key, 3)
        
        new_is_red = jax.random.bernoulli(k0, 0.5, (num_games,))
        state = v_init(jax.random.split(k1, num_games))
        
        def play_step(carry, key):
            state, = carry
            obs = v_observe(state)
            
            new_output = network.apply(new_params, obs)
            old_output = network.apply(old_params, obs)
            
            is_new_turn = (state.current_player == 0) == new_is_red
            logits = jnp.where(
                is_new_turn[:, None],
                new_output.policy_logits,
                old_output.policy_logits,
            )
            
            logits = jnp.where(state.legal_action_mask, logits, -1e9)
            action = jnp.argmax(logits, axis=-1)
            
            next_state = v_step(state, action)
            return (next_state,), next_state.terminated
        
        (final_state,), _ = jax.lax.scan(
            play_step,
            (state,),
            jax.random.split(k2, max_steps),
        )
        
        new_wins = jnp.where(
            new_is_red,
            final_state.winner == 0,
            final_state.winner == 1,
        ).astype(jnp.float32)
        
        draws = (final_state.winner == -1).astype(jnp.float32)
        win_rate = (new_wins + 0.5 * draws).mean()
        
        expected = 1 / (1 + 10 ** ((elo - elo) / 400))
        new_elo = elo + 32 * (win_rate - expected) * num_games
        
        return new_elo, win_rate
    
    return eval_fn


# ============================================================================
# 检查点管理
# ============================================================================

class CheckpointManager:
    """检查点管理"""
    
    def __init__(self, checkpoint_dir: str, max_to_keep: int = 5):
        # 使用绝对路径
        self.checkpoint_dir = Path(checkpoint_dir).resolve()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_to_keep = max_to_keep
        
    def save(self, state: TrainState, step: int, elo: float):
        ckpt = {
            "params": state.params,
            "opt_state": state.opt_state,
            "step": step,
            "elo": elo,
        }
        checkpoints.save_checkpoint(
            str(self.checkpoint_dir),  # 已是绝对路径
            ckpt,
            step,
            keep=self.max_to_keep,
        )
        print(f"检查点已保存: step={step}, elo={elo:.1f}", flush=True)
        
    def restore(self, state: TrainState) -> tuple:
        ckpt = checkpoints.restore_checkpoint(str(self.checkpoint_dir), None)
        if ckpt is None:
            return state, 0, 1500.0
        
        state = state.replace(
            params=ckpt["params"],
            opt_state=ckpt["opt_state"],
        )
        return state, ckpt["step"], ckpt["elo"]


# ============================================================================
# 主函数
# ============================================================================

def main():
    """
    主训练函数 - 多卡数据并行版本
    """
    config = load_config()
    
    checkpoint_dir = config["checkpoint"].get("checkpoint_dir", "checkpoints")
    log_dir = config["logging"].get("log_dir", "logs")
    
    # ====================================================================
    # 多卡配置
    # ====================================================================
    devices = jax.local_devices()
    n_devices = len(devices)
    
    print("=" * 60, flush=True)
    print("ZeroForge - 中国象棋 Gumbel MuZero (多卡版)", flush=True)
    print("=" * 60, flush=True)
    print(f"JAX 后端: {jax.default_backend()}", flush=True)
    print(f"设备数量: {n_devices}", flush=True)
    print(f"设备列表: {devices}", flush=True)
    
    # 检查配置是否能被设备数整除
    global_batch_size = config["training"]["batch_size"]
    global_num_parallel = config["self_play"]["num_parallel_games"]
    
    if global_batch_size % n_devices != 0:
        raise ValueError(
            f"batch_size={global_batch_size} 必须能被设备数 {n_devices} 整除！"
            f"建议改为 {(global_batch_size // n_devices + 1) * n_devices}"
        )
    if global_num_parallel % n_devices != 0:
        raise ValueError(
            f"num_parallel_games={global_num_parallel} 必须能被设备数 {n_devices} 整除！"
            f"建议改为 {(global_num_parallel // n_devices + 1) * n_devices}"
        )
    
    per_device_batch = global_batch_size // n_devices
    per_device_parallel = global_num_parallel // n_devices
    
    print(f"全局 batch_size: {global_batch_size} (每卡 {per_device_batch})", flush=True)
    print(f"全局 num_parallel_games: {global_num_parallel} (每卡 {per_device_parallel})", flush=True)
    print(f"MCTS 模拟: {config['mcts']['num_simulations']}", flush=True)
    
    # 环境和网络
    env = XiangqiEnv()
    network = MuZeroNetwork(
        action_space_size=env.action_space_size,
        hidden_dim=config["network"]["hidden_dim"],
    )
    
    # 初始化
    key = jax.random.PRNGKey(config.get("seed", 42))
    key, init_key = jax.random.split(key)
    
    state = create_train_state(
        init_key, network,
        input_shape=(per_device_batch, 240, 10, 9),
        learning_rate=config["training"]["learning_rate"],
    )
    
    # 检查点
    ckpt_manager = CheckpointManager(
        checkpoint_dir,
        max_to_keep=config["checkpoint"].get("max_to_keep", 5),
    )
    state, step, elo = ckpt_manager.restore(state)
    
    if step > 0:
        print(f"从检查点恢复: step={step}, elo={elo:.1f}", flush=True)
    
    # TensorBoard
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard: tensorboard --logdir {log_dir}", flush=True)
    
    # ====================================================================
    # 创建每设备的配置和函数
    # ====================================================================
    config_per_device = dict(config)
    config_per_device["self_play"] = dict(config["self_play"])
    config_per_device["training"] = dict(config["training"])
    config_per_device["self_play"]["num_parallel_games"] = per_device_parallel
    config_per_device["training"]["batch_size"] = per_device_batch
    
    selfplay_fn = make_selfplay_fn(env, network, config_per_device)
    train_step = make_train_step(network, config_per_device)
    eval_fn = make_eval_fn(env, network, config)
    
    # ====================================================================
    # pmap 包装
    # ====================================================================
    # 自玩: 每卡独立运行，不需要跨卡通信
    p_selfplay = jax.pmap(selfplay_fn, axis_name="devices")
    
    # 训练: 用 lax.scan 处理所有 batch，避免 Python 循环导致的 GPU 停顿
    def train_epoch(state, data, data_mask):
        """处理一个 epoch 的所有 batch（在设备上执行）"""
        def train_one_batch(carry, batch_data):
            s, = carry
            batch, mask = batch_data
            new_state, metrics = train_step(s, batch, mask)
            return (new_state,), metrics
        
        # data: (n_batches, batch_size, ...)
        # 用 scan 遍历所有 batch
        (final_state,), all_metrics = lax.scan(
            train_one_batch,
            (state,),
            (data, data_mask),
        )
        
        # 计算平均 loss
        avg_metrics = jax.tree.map(lambda x: x.mean(), all_metrics)
        return final_state, avg_metrics
    
    # pmap: 每卡执行 train_epoch
    p_train_epoch = jax.pmap(train_epoch, axis_name="devices")
    
    # 复制参数到所有设备
    state = jax_utils.replicate(state)
    
    # 训练配置
    save_interval = config["checkpoint"].get("save_interval", 1000)
    eval_interval = config["evaluation"].get("eval_interval", 5000)
    log_interval = config["logging"].get("console_interval", 100)
    tb_interval = config["logging"].get("tensorboard_interval", 100)
    num_training_steps = config["training"]["num_training_steps"]
    
    old_params = jax_utils.unreplicate(state.params)
    best_elo = elo
    
    print("开始训练...", flush=True)
    print("首次运行需要 JIT 编译，可能需要几分钟...", flush=True)
    start_time = time.time()
    
    # ====================================================================
    # 训练循环
    # ====================================================================
    global _SHOULD_EXIT
    while step < num_training_steps and not _SHOULD_EXIT:
        iter_start = time.time()
        step_at_iter_start = step
        key, k0, k1 = jax.random.split(key, 3)
        
        # 为每张卡生成独立的随机数
        device_keys = jax.random.split(k0, n_devices)
        
        # ----- 多卡自我对弈 -----
        # data: (n_devices, n_batches, per_device_batch, ...)
        data, data_mask = p_selfplay(state.params, device_keys)
        n_batches = data.obs.shape[1]
        n_samples = n_devices * n_batches * per_device_batch
        
        # ----- 多卡训练 (用 lax.scan，全部在 GPU 上执行) -----
        state, metrics = p_train_epoch(state, data, data_mask)
        
        step += n_batches
        iter_time = time.time() - iter_start
        
        # 日志（只在需要时才同步 GPU->CPU）
        def _crossed(prev, curr, interval):
            return interval > 0 and (prev // interval) != (curr // interval)

        need_log = _crossed(step_at_iter_start, step, log_interval)
        need_tb = _crossed(step_at_iter_start, step, tb_interval)
        
        if need_log or need_tb:
            # 只在需要时才从 GPU 取值
            avg_loss = float(metrics["loss"][0])
            avg_ploss = float(metrics["policy_loss"][0])
            avg_vloss = float(metrics["value_loss"][0])
            
            if need_log:
                elapsed = time.time() - start_time
                samples_per_sec = n_samples / iter_time if iter_time > 0 else 0
                print(
                    f"step={step}, loss={avg_loss:.4f}, ploss={avg_ploss:.4f}, vloss={avg_vloss:.4f}, "
                    f"samples={n_samples} ({n_devices}卡×{n_batches}批×{per_device_batch}), "
                    f"time={iter_time:.1f}s ({samples_per_sec:.0f}/s), elapsed={elapsed/60:.1f}min",
                    flush=True
                )

            if need_tb:
                writer.add_scalar("loss/total", avg_loss, step)
                writer.add_scalar("loss/policy", avg_ploss, step)
                writer.add_scalar("loss/value", avg_vloss, step)
                writer.add_scalar("perf/samples_per_iter", n_samples, step)
                writer.add_scalar("perf/samples_per_sec", n_samples / iter_time, step)
        
        # ELO 评估 (单卡运行)
        if _crossed(step_at_iter_start, step, eval_interval):
            current_params = jax_utils.unreplicate(state.params)
            elo, win_rate = eval_fn(current_params, old_params, elo, k1)
            elo = float(elo)
            win_rate = float(win_rate)
            
            writer.add_scalar("eval/elo", elo, step)
            writer.add_scalar("eval/win_rate", win_rate, step)
            
            print(f"评估: elo={elo:.1f}, win_rate={win_rate:.2%}", flush=True)
            
            if elo > best_elo:
                best_elo = elo
                old_params = current_params
                print(f"新最佳模型! elo={elo:.1f}", flush=True)
        
        # 保存检查点
        if _crossed(step_at_iter_start, step, save_interval):
            ckpt_manager.save(jax_utils.unreplicate(state), step, elo)
    
    # 最终保存
    ckpt_manager.save(jax_utils.unreplicate(state), step, elo)
    writer.close()
    
    total_time = time.time() - start_time
    if _SHOULD_EXIT:
        print(f"训练被中断，已保存检查点。总时间: {total_time/3600:.1f}小时", flush=True)
    else:
        print(f"训练完成! 总时间: {total_time/3600:.1f}小时", flush=True)


if __name__ == "__main__":
    main()

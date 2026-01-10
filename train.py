#!/usr/bin/env python3
"""
ZeroForge - 中国象棋 Gumbel MuZero
训练脚本：自我对弈 + 训练 + ELO 评估 + 断点继续 + TensorBoard
"""

import os
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

from xiangqi.env import XiangqiEnv
from networks.muzero import MuZeroNetwork, create_train_state
import mctx

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
logger = logging.getLogger(__name__)


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
# 自我对弈 (全 JIT)
# ============================================================================

def make_selfplay_fn(env, network, config: dict):
    """创建 JIT 编译的自我对弈函数"""
    
    num_parallel = config["self_play"]["num_parallel_games"]
    num_steps = config["self_play"].get("max_steps", 200)
    num_simulations = config["mcts"]["num_simulations"]
    max_actions = config["mcts"].get("max_num_considered_actions", 16)
    discount = config["mcts"].get("discount", 1.0)
    batch_size = config["training"]["batch_size"]

    # 兼容不同 mctx 版本：有的版本支持 max_num_considered_actions，有的不支持
    try:
        _gumbel_sig = inspect.signature(mctx.gumbel_muzero_policy)
        supports_max_actions = "max_num_considered_actions" in _gumbel_sig.parameters
    except Exception as e:
        # 避免因 inspect 在某些环境异常导致训练直接挂掉
        logger.warning(f"无法检查 mctx.gumbel_muzero_policy 参数签名，将不传 max_num_considered_actions: {e}")
        supports_max_actions = False
    
    def recurrent_fn(params, rng_key, action, embedding):
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
    
    def selfplay_step(carry, key):
        state, params = carry
        k0, k1 = jax.random.split(key)
        
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
                rng_key=k0,
                root=root,
                recurrent_fn=recurrent_fn,
                num_simulations=num_simulations,
                invalid_actions=~state.legal_action_mask,
                max_num_considered_actions=max_actions,
            )
        else:
            policy_output = mctx.gumbel_muzero_policy(
                params=params,
                rng_key=k0,
                root=root,
                recurrent_fn=recurrent_fn,
                num_simulations=num_simulations,
                invalid_actions=~state.legal_action_mask,
            )
        
        next_state = v_step(state, policy_output.action)
        reward = jax.vmap(lambda s, p: s.rewards[p])(next_state, player)
        
        traj = Trajectory(
            obs=obs,
            policy=policy_output.action_weights,
            reward=reward,
            terminated=next_state.terminated,
            player=player,
        )
        return (next_state, params), traj
    
    @jax.jit
    def selfplay_fn(params, key):
        k0, k1, k2 = jax.random.split(key, 3)
        
        state = v_init(jax.random.split(k0, num_parallel))
        _, traj = jax.lax.scan(
            selfplay_step,
            (state, params),
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
        
        # 有效样本掩码 (游戏结束前的步骤)
        mask = jnp.cumsum(traj.terminated[::-1], axis=0)[::-1] == 0
        
        # 展平 (T, B, ...) -> (T*B, ...)
        total_samples = num_steps * num_parallel
        sample = Sample(
            obs=traj.obs.reshape(total_samples, *traj.obs.shape[2:]),
            policy=traj.policy.reshape(total_samples, *traj.policy.shape[2:]),
            value=target_value.reshape(total_samples),
        )
        mask = mask.reshape(total_samples)
        
        # 打乱顺序
        perm = jax.random.permutation(k2, total_samples)
        sample = jax.tree.map(lambda x: x[perm], sample)
        mask = mask[perm]
        
        # 分成 minibatch (固定大小)
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
    
    @jax.jit
    def train_step(state, batch, mask, *, axis_name: str | None = None):
        def loss_fn(params):
            output = network.apply(params, batch.obs)
            
            # 使用 mask 加权损失 (只计算有效样本)
            policy_loss_per_sample = optax.losses.softmax_cross_entropy(
                output.policy_logits, batch.policy
            )
            value_loss_per_sample = optax.losses.squared_error(
                output.value, batch.value
            )
            
            # 加权平均
            mask_sum = mask.sum() + 1e-8
            policy_loss = (policy_loss_per_sample * mask).sum() / mask_sum
            value_loss = (value_loss_per_sample * mask).sum() / mask_sum
            
            total_loss = policy_loss + value_loss_weight * value_loss
            return total_loss, (policy_loss, value_loss)
        
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (ploss, vloss)), grads = grad_fn(state.params)
        
        # 多卡：跨设备平均梯度/指标（数据并行）
        if axis_name is not None:
            grads = lax.pmean(grads, axis_name=axis_name)
            loss = lax.pmean(loss, axis_name=axis_name)
            ploss = lax.pmean(ploss, axis_name=axis_name)
            vloss = lax.pmean(vloss, axis_name=axis_name)

        # 梯度裁剪
        grads = jax.tree.map(lambda g: jnp.clip(g, -1.0, 1.0), grads)
        
        state = state.apply_gradients(grads=grads)
        return state, {"loss": loss, "policy_loss": ploss, "value_loss": vloss}
    
    return train_step


# ============================================================================
# ELO 评估
# ============================================================================

def make_eval_fn(env, network, config: dict):
    """创建评估函数 (新模型 vs 旧模型)"""
    
    num_games = config["evaluation"].get("num_games", 100)
    max_steps = config["self_play"].get("max_steps", 200)
    
    v_init = jax.vmap(env.init)
    v_step = jax.vmap(env.step)
    v_observe = jax.vmap(env.observe)
    
    @jax.jit
    def eval_fn(new_params, old_params, elo, key):
        k0, k1, k2 = jax.random.split(key, 3)
        
        # 随机分配新/旧模型的颜色
        new_is_red = jax.random.bernoulli(k0, 0.5, (num_games,))
        state = v_init(jax.random.split(k1, num_games))
        
        def play_step(carry, key):
            state, = carry
            obs = v_observe(state)
            
            # 新模型推理
            new_output = network.apply(new_params, obs)
            old_output = network.apply(old_params, obs)
            
            # 选择当前玩家对应的模型
            is_new_turn = (state.current_player == 0) == new_is_red
            logits = jnp.where(
                is_new_turn[:, None],
                new_output.policy_logits,
                old_output.policy_logits,
            )
            
            # 贪心选择
            logits = jnp.where(state.legal_action_mask, logits, -1e9)
            action = jnp.argmax(logits, axis=-1)
            
            next_state = v_step(state, action)
            return (next_state,), next_state.terminated
        
        (final_state,), _ = jax.lax.scan(
            play_step,
            (state,),
            jax.random.split(k2, max_steps),
        )
        
        # 计算新模型的胜率
        # winner: 0=红胜, 1=黑胜, -1=平局
        new_wins = jnp.where(
            new_is_red,
            final_state.winner == 0,  # 新模型是红，红胜则新胜
            final_state.winner == 1,  # 新模型是黑，黑胜则新胜
        ).astype(jnp.float32)
        
        draws = (final_state.winner == -1).astype(jnp.float32)
        win_rate = (new_wins + 0.5 * draws).mean()
        
        # 更新 ELO
        expected = 1 / (1 + 10 ** ((elo - elo) / 400))  # 对手ELO相同
        new_elo = elo + 32 * (win_rate - expected) * num_games
        
        return new_elo, win_rate
    
    return eval_fn


# ============================================================================
# 检查点管理
# ============================================================================

class CheckpointManager:
    """简单的检查点管理"""
    
    def __init__(self, checkpoint_dir: str, max_to_keep: int = 5):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_to_keep = max_to_keep
        
    def save(self, state: TrainState, step: int, elo: float):
        """保存检查点"""
        ckpt = {
            "params": state.params,
            "opt_state": state.opt_state,
            "step": step,
            "elo": elo,
        }
        path = self.checkpoint_dir / f"ckpt_{step}"
        checkpoints.save_checkpoint(
            str(self.checkpoint_dir),
            ckpt,
            step,
            keep=self.max_to_keep,
        )
        logger.info(f"检查点已保存: step={step}, elo={elo:.1f}")
        
    def restore(self, state: TrainState) -> tuple:
        """恢复检查点"""
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
    # 加载配置
    config = load_config()
    
    # 路径
    checkpoint_dir = config["checkpoint"].get("checkpoint_dir", "checkpoints")
    log_dir = config["logging"].get("log_dir", "logs")
    
    logger.info("=" * 60)
    logger.info("ZeroForge - 中国象棋 Gumbel MuZero")
    logger.info("=" * 60)
    logger.info(f"设备: {jax.devices()}")
    logger.info(f"并行游戏: {config['self_play']['num_parallel_games']}")
    logger.info(f"MCTS 模拟: {config['mcts']['num_simulations']}")
    
    # 环境和网络
    env = XiangqiEnv()
    network = MuZeroNetwork(
        action_space_size=env.action_space_size,
        hidden_dim=config["network"]["hidden_dim"],
    )
    
    # 初始化
    key = jax.random.PRNGKey(config.get("seed", 42))
    key, init_key = jax.random.split(key)
    
    # ----------------------------
    # 多卡配置（单机多卡数据并行）
    # ----------------------------
    devices = jax.local_devices()
    n_devices = len(devices)
    if n_devices <= 0:
        raise RuntimeError("未检测到任何 JAX 设备（local_devices 为空），无法训练。")

    requested_devices = config.get("training", {}).get("num_devices", None)
    if requested_devices is not None:
        if not isinstance(requested_devices, int) or requested_devices <= 0:
            raise ValueError(f"training.num_devices 必须是正整数，当前={requested_devices!r}")
        if requested_devices > n_devices:
            raise ValueError(
                f"training.num_devices={requested_devices} 超过可用设备数 n_devices={n_devices}。"
            )
        n_devices = requested_devices
        devices = devices[:n_devices]

    global_batch_size = int(config["training"]["batch_size"])
    global_num_parallel = int(config["self_play"]["num_parallel_games"])
    if global_batch_size % n_devices != 0:
        raise ValueError(
            f"training.batch_size={global_batch_size} 必须能被设备数 {n_devices} 整除，"
            f"否则无法做数据并行。建议改为 {n_devices} 的倍数。"
        )
    if global_num_parallel % n_devices != 0:
        raise ValueError(
            f"self_play.num_parallel_games={global_num_parallel} 必须能被设备数 {n_devices} 整除，"
            f"否则无法把自我对弈均匀分到每张卡。建议改为 {n_devices} 的倍数。"
        )

    per_device_batch_size = global_batch_size // n_devices
    per_device_num_parallel = global_num_parallel // n_devices

    logger.info(
        f"多卡: n_devices={n_devices}, per_device_batch_size={per_device_batch_size}, "
        f"per_device_num_parallel_games={per_device_num_parallel}"
    )

    state = create_train_state(
        init_key, network,
        # 初始化用 per-device batch 即可（不影响参数形状）
        input_shape=(per_device_batch_size, 240, 10, 9),
        learning_rate=config["training"]["learning_rate"],
    )
    
    # 检查点管理
    ckpt_manager = CheckpointManager(
        checkpoint_dir,
        max_to_keep=config["checkpoint"].get("max_to_keep", 5),
    )
    state, step, elo = ckpt_manager.restore(state)
    
    if step > 0:
        logger.info(f"从检查点恢复: step={step}, elo={elo:.1f}")
    
    # TensorBoard
    writer = SummaryWriter(log_dir)
    logger.info(f"TensorBoard: tensorboard --logdir {log_dir}")
    
    # 创建函数
    # 自我对弈/训练函数：按每设备的并行度与 batch 来构造，便于 pmap
    config_local = dict(config)
    config_local["self_play"] = dict(config["self_play"])
    config_local["training"] = dict(config["training"])
    config_local["self_play"]["num_parallel_games"] = per_device_num_parallel
    config_local["training"]["batch_size"] = per_device_batch_size

    selfplay_fn = make_selfplay_fn(env, network, config_local)
    train_step = make_train_step(network, config)
    eval_fn = make_eval_fn(env, network, config)

    # pmap 包装：自玩与训练都做数据并行
    p_selfplay_fn = jax.pmap(selfplay_fn, axis_name="devices", in_axes=(0, 0), out_axes=(0, 0))

    # 训练：把“每 batch 一次 dispatch”变成“设备内 scan”，显著提升 GPU 利用率
    @jax.jit
    def _train_epoch_single_device(s, data_batches, mask_batches):
        """
        单卡：对 (n_batches, per_device_batch, ...) 执行一轮训练，返回最后 state 和每步 metrics 序列
        注意：这个函数会被 pmap 包裹，内部会用 pmean 做跨卡平均。
        """
        def body_fn(carry, xs):
            b, m = xs
            new_state, metrics = train_step(carry, b, m, axis_name="devices")
            return new_state, metrics

        return lax.scan(body_fn, s, (data_batches, mask_batches))

    p_train_epoch = jax.pmap(
        _train_epoch_single_device,
        axis_name="devices",
        in_axes=(0, 0, 0),
        out_axes=(0, 0),
    )

    # 参数/优化器状态复制到每张卡
    state = jax_utils.replicate(state, devices=devices)
    
    # 训练配置
    save_interval = config["checkpoint"].get("save_interval", 1000)
    eval_interval = config["evaluation"].get("eval_interval", 5000)
    log_interval = config["logging"].get("console_interval", 100)
    tb_interval = config["logging"].get("tensorboard_interval", 100)
    num_steps = config["training"]["num_training_steps"]
    
    old_params = jax_utils.unreplicate(state.params)  # 用于 ELO 评估（单卡）
    best_elo = elo
    
    logger.info("开始训练...")
    start_time = time.time()
    
    # 训练循环
    while step < num_steps:
        iter_start = time.time()
        step_at_iter_start = step
        key, k0, k1 = jax.random.split(key, 3)
        # 每张卡一把 key，保证自玩随机性独立
        device_keys = jax.random.split(k0, n_devices)
        
        # 自我对弈
        data, data_mask = p_selfplay_fn(state.params, device_keys)
        # data: (n_devices, n_batches, per_device_batch, ...)
        n_batches = data.obs.shape[1]
        n_samples = int(n_devices) * int(n_batches) * int(per_device_batch_size)
        
        # 训练（设备内 scan 一次跑完 n_batches，减少 host 调度开销）
        state, metrics_seq = p_train_epoch(state, data, data_mask)
        # metrics_seq: (n_devices, n_batches) 的标量序列；已 pmean，取 [0] 即可
        metrics_mean = jax.tree.map(lambda x: jnp.mean(x[0], axis=0), metrics_seq)

        # step 按 batch 计数：一轮自玩会产生 n_batches 个更新
        step += int(n_batches)
        
        # 显式同步：确保 iter_time 统计的是“真实计算耗时”（包含 XLA 执行）
        metrics_host = jax.device_get(metrics_mean)
        avg_loss_host = float(metrics_host["loss"])
        iter_time = time.time() - iter_start
        
        # 控制台日志
        def _crossed_interval(prev_step: int, curr_step: int, interval: int) -> bool:
            """判断本轮内是否跨过了 interval 的边界（用于只输出一次日志/评估/保存）。"""
            return interval > 0 and (prev_step // interval) != (curr_step // interval)

        if _crossed_interval(step_at_iter_start, step, log_interval):
            elapsed = time.time() - start_time
            logger.info(
                f"step={step}, loss={avg_loss_host:.4f}, "
                f"samples={n_samples}, elo={elo:.1f}, "
                f"time={iter_time:.1f}s, elapsed={elapsed/60:.1f}min"
            )

        # TensorBoard（低频写入，避免 device↔host 同步抖动）
        if _crossed_interval(step_at_iter_start, step, tb_interval):
            writer.add_scalar("loss/total", float(metrics_host["loss"]), step)
            writer.add_scalar("loss/policy", float(metrics_host["policy_loss"]), step)
            writer.add_scalar("loss/value", float(metrics_host["value_loss"]), step)
        
        # ELO 评估
        if _crossed_interval(step_at_iter_start, step, eval_interval):
            current_params = jax_utils.unreplicate(state.params)
            elo, win_rate = eval_fn(current_params, old_params, elo, k1)
            elo = float(elo)
            win_rate = float(win_rate)
            
            writer.add_scalar("eval/elo", elo, step)
            writer.add_scalar("eval/win_rate", win_rate, step)
            
            logger.info(f"评估: elo={elo:.1f}, win_rate={win_rate:.2%}")
            
            if elo > best_elo:
                best_elo = elo
                old_params = current_params  # 更新对手
                logger.info(f"新最佳模型! elo={elo:.1f}")
        
        # 保存检查点
        if _crossed_interval(step_at_iter_start, step, save_interval):
            ckpt_manager.save(jax_utils.unreplicate(state), step, elo)
    
    # 最终保存
    ckpt_manager.save(jax_utils.unreplicate(state), step, elo)
    writer.close()
    
    total_time = time.time() - start_time
    logger.info(f"训练完成! 总时间: {total_time/3600:.1f}小时")


if __name__ == "__main__":
    main()

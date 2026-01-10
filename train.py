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
        
        mask = jnp.cumsum(traj.terminated[::-1], axis=0)[::-1] == 0
        
        sample = Sample(
            obs=traj.obs.reshape(-1, *traj.obs.shape[2:]),
            policy=traj.policy.reshape(-1, *traj.policy.shape[2:]),
            value=target_value.reshape(-1),
        )
        mask = mask.reshape(-1)
        
        n_valid = mask.sum()
        idx = jnp.where(mask, jnp.arange(mask.shape[0]), mask.shape[0])
        idx = jnp.sort(idx)[:n_valid]
        sample = jax.tree.map(lambda x: x[idx], sample)
        
        perm = jax.random.permutation(k2, n_valid)
        sample = jax.tree.map(lambda x: x[perm], sample)
        
        n_batches = n_valid // batch_size
        sample = jax.tree.map(
            lambda x: x[:n_batches * batch_size].reshape(n_batches, batch_size, *x.shape[1:]),
            sample
        )
        return sample
    
    return selfplay_fn


# ============================================================================
# 训练步骤
# ============================================================================

def make_train_step(network, config: dict):
    """创建训练步骤函数"""
    
    value_loss_weight = config["training"].get("value_loss_weight", 1.0)
    
    @jax.jit
    def train_step(state, batch):
        def loss_fn(params):
            output = network.apply(params, batch.obs)
            
            policy_loss = optax.losses.softmax_cross_entropy(
                output.policy_logits, batch.policy
            ).mean()
            
            value_loss = optax.losses.squared_error(
                output.value, batch.value
            ).mean()
            
            total_loss = policy_loss + value_loss_weight * value_loss
            return total_loss, (policy_loss, value_loss)
        
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (ploss, vloss)), grads = grad_fn(state.params)
        
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
    
    batch_size = config["training"]["batch_size"]
    state = create_train_state(
        init_key, network,
        input_shape=(batch_size, 240, 10, 9),
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
    selfplay_fn = make_selfplay_fn(env, network, config)
    train_step = make_train_step(network, config)
    eval_fn = make_eval_fn(env, network, config)
    
    # 训练配置
    save_interval = config["checkpoint"].get("save_interval", 1000)
    eval_interval = config["evaluation"].get("eval_interval", 5000)
    log_interval = config["logging"].get("console_interval", 100)
    num_steps = config["training"]["num_training_steps"]
    
    old_params = state.params  # 用于 ELO 评估
    best_elo = elo
    
    logger.info("开始训练...")
    start_time = time.time()
    
    # 训练循环
    while step < num_steps:
        iter_start = time.time()
        key, k0, k1 = jax.random.split(key, 3)
        
        # 自我对弈
        data = selfplay_fn(state.params, k0)
        n_samples = data.obs.shape[0] * data.obs.shape[1]
        
        # 训练
        total_loss = 0
        for i in range(data.obs.shape[0]):
            batch = jax.tree.map(lambda x: x[i], data)
            state, metrics = train_step(state, batch)
            total_loss += metrics["loss"]
            step += 1
            
            # 记录到 TensorBoard
            writer.add_scalar("loss/total", metrics["loss"], step)
            writer.add_scalar("loss/policy", metrics["policy_loss"], step)
            writer.add_scalar("loss/value", metrics["value_loss"], step)
        
        avg_loss = total_loss / data.obs.shape[0]
        iter_time = time.time() - iter_start
        
        # 控制台日志
        if step % log_interval < data.obs.shape[0]:
            elapsed = time.time() - start_time
            logger.info(
                f"step={step}, loss={avg_loss:.4f}, "
                f"samples={n_samples}, elo={elo:.1f}, "
                f"time={iter_time:.1f}s, elapsed={elapsed/60:.1f}min"
            )
        
        # ELO 评估
        if step % eval_interval < data.obs.shape[0]:
            elo, win_rate = eval_fn(state.params, old_params, elo, k1)
            elo = float(elo)
            win_rate = float(win_rate)
            
            writer.add_scalar("eval/elo", elo, step)
            writer.add_scalar("eval/win_rate", win_rate, step)
            
            logger.info(f"评估: elo={elo:.1f}, win_rate={win_rate:.2%}")
            
            if elo > best_elo:
                best_elo = elo
                old_params = state.params  # 更新对手
                logger.info(f"新最佳模型! elo={elo:.1f}")
        
        # 保存检查点
        if step % save_interval < data.obs.shape[0]:
            ckpt_manager.save(state, step, elo)
    
    # 最终保存
    ckpt_manager.save(state, step, elo)
    writer.close()
    
    total_time = time.time() - start_time
    logger.info(f"训练完成! 总时间: {total_time/3600:.1f}小时")


if __name__ == "__main__":
    main()

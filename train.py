#!/usr/bin/env python3
"""
ZeroForge - 中国象棋 Gumbel AlphaZero
现代化极简顶级架构：算力随机化 + 视角归一化 + 镜像增强 + 标准 ELO
"""

import os
import sys
import time
import json
import warnings
from functools import partial
from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
import optax
import mctx
import orbax.checkpoint as ocp

# --- JAX 编译缓存配置 ---
# 开启持久化编译缓存，二次启动秒开
cache_dir = os.path.abspath("jax_cache")
os.makedirs(cache_dir, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", cache_dir)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)  # 缓存所有编译结果
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)  # 不管编译时间都缓存

# --- XLA 编译加速建议 (无损方案) ---
# 1. 允许 XLA 融合更多算子
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_highest_priority_async_stream=true "
)
# 2. 强制使用 32 位哈希 (已在 env.py 实现)
# 3. 避免不需要的 64 位运算
jax.config.update("jax_enable_x64", False)

from xiangqi.env import XiangqiEnv
from xiangqi.actions import rotate_action, ACTION_SPACE_SIZE
from xiangqi.mirror import mirror_observation, mirror_policy
from networks.alphazero import AlphaZeroNetwork
from tensorboardX import SummaryWriter

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
    training_batch_size: int = 1024
    td_steps: int = 20   # 缩短 TD 步数，减少价值估计方差
    
    # 自对弈与搜索 (Gumbel 优势：低算力也能产生强信号)
    selfplay_batch_size: int = 1024
    num_simulations: int = 48           # 增加模拟次数，提升关键局面搜索质量
    top_k: int = 16
    
    # 经验回放配置
    replay_buffer_size: int = 500000
    sample_reuse_times: int = 3
    
    # 探索策略 (更保守的温度衰减，减少臭棋)
    temperature_steps: int = 60
    temperature_initial: float = 1.0
    temperature_final: float = 0.01
    
    # 环境规则
    max_steps: int = 200
    max_no_capture_steps: int = 60
    repetition_threshold: int = 3
    # perpetual_check_threshold 已废弃，现使用"重复局面+将军=长将判负"规则
    
    # ELO 评估
    eval_interval: int = 20
    eval_games: int = 100
    past_model_offset: int = 20
    
    # Checkpoint 配置
    ckpt_interval: int = 10         # 每 N 次迭代保存 checkpoint
    max_to_keep: int = 20            # 最多保留 N 个 checkpoint
    keep_period: int = 50           # 每 N 次迭代永久保留一个 checkpoint

config = Config()

# ============================================================================
# 环境和设备
# ============================================================================

devices = jax.local_devices()
num_devices = len(devices)

# 预计算旋转索引，避免在 JIT 循环内重复计算
_ROTATED_IDX = rotate_action(jnp.arange(ACTION_SPACE_SIZE))

def replicate_to_devices(pytree):
    """将 pytree 复制到所有设备"""
    # 使用 jax.device_put_replicated（虽然有 deprecation warning，但最可靠）
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return jax.device_put_replicated(pytree, devices)

env = XiangqiEnv(
    max_steps=config.max_steps,
    max_no_capture_steps=config.max_no_capture_steps,
    repetition_threshold=config.repetition_threshold,
)

net = AlphaZeroNetwork(
    action_space_size=env.action_space_size,
    channels=config.num_channels,
    num_blocks=config.num_blocks,
)

def forward(params, obs, is_training=False):
    """前向传播 (LayerNorm 架构，无需 batch_stats)"""
    logits, value = net.apply({'params': params}, obs, train=is_training)
    return logits, value

def recurrent_fn(params, rng_key, action, state):
    """MCTS 递归函数"""
    prev_player = state.current_player
    state = jax.vmap(env.step)(state, action)
    obs = jax.vmap(env.observe)(state)
    logits, value = forward(params, obs)
    
    logits = jnp.where(state.current_player[:, None] == 0, logits, logits[:, _ROTATED_IDX])
    
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
    root_value: jnp.ndarray  # MCTS 搜索后的根节点价值 (用于 TD bootstrap)

class Sample(NamedTuple):
    obs: jnp.ndarray
    policy_tgt: jnp.ndarray
    value_tgt: jnp.ndarray  # n-step TD 目标
    mask: jnp.ndarray

# ============================================================================
# 自玩
# ============================================================================

@jax.pmap
def selfplay(params, rng_key):
    """
    高性能自玩算子：
    - 使用 lax.scan 消除 Python 循环开销
    - Gumbel 算法无需 Dirichlet 噪声 (Gumbel 噪声已提供足够探索)
    - 统一策略（移除强弱差异，节省计算资源）
    """
    batch_size = config.selfplay_batch_size // num_devices
    
    def step_fn(state, key):
        key_search, key_sample, key_next = jax.random.split(key, 3)
        obs = jax.vmap(env.observe)(state)
        logits, value = forward(params, obs)
        
        logits = jnp.where(state.current_player[:, None] == 0, logits, logits[:, _ROTATED_IDX])
        
        # Gumbel AlphaZero 无需 Dirichlet 噪声，Gumbel 采样本身提供探索
        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)
        
        # 统一策略：只执行一次 MCTS 搜索
        policy_output = mctx.gumbel_muzero_policy(
            params=params, rng_key=key_search, root=root, recurrent_fn=recurrent_fn,
            num_simulations=config.num_simulations,
            max_num_considered_actions=config.top_k,
            invalid_actions=~state.legal_action_mask,
        )
        
        action_weights = policy_output.action_weights
        root_value = policy_output.search_tree.node_values[:, 0]
        
        temp = jnp.where(state.step_count < config.temperature_steps, 
                         config.temperature_initial, config.temperature_final)
        
        def _sample_action(w, t, k, legal_mask):
            """温度采样 (log 空间避免数值下溢)"""
            t = jnp.maximum(t, 1e-3)
            w_masked = jnp.where(legal_mask, w, 0.0)
            log_w = jnp.log(w_masked + 1e-10)
            log_w_temp = log_w / t
            log_w_temp = jnp.where(legal_mask, log_w_temp, -jnp.inf)
            log_w_temp = log_w_temp - jnp.max(log_w_temp)
            w_temp = jnp.exp(log_w_temp)
            w_temp = jnp.where(legal_mask, w_temp, 0.0)
            w_prob = w_temp / jnp.maximum(jnp.sum(w_temp), 1e-10)
            return jax.random.choice(k, ACTION_SPACE_SIZE, p=w_prob)
        
        sample_keys = jax.random.split(key_sample, batch_size)
        action = jax.vmap(_sample_action)(action_weights, temp, sample_keys, state.legal_action_mask)
        
        actor = state.current_player
        next_state = jax.vmap(env.step)(state, action)
        
        normalized_action_weights = jnp.where(state.current_player[:, None] == 0, 
                                              action_weights, action_weights[:, _ROTATED_IDX])
        
        data = SelfplayOutput(
            obs=obs, action_weights=normalized_action_weights,
            reward=next_state.rewards[jnp.arange(batch_size), actor],
            terminated=next_state.terminated,
            discount=jnp.where(next_state.terminated, 0.0, -1.0),
            winner=next_state.winner, draw_reason=next_state.draw_reason,
            root_value=root_value,
        )
        
        next_state_reset = jax.vmap(lambda s, k: jax.lax.cond(
            s.terminated, lambda: env.init(k), lambda: s
        ))(next_state, jax.random.split(key_next, batch_size))
        return next_state_reset, data

    state = jax.vmap(env.init)(jax.random.split(rng_key, batch_size))
    _, data = jax.lax.scan(step_fn, state, jax.random.split(rng_key, config.max_steps))
    return data

@jax.pmap
def compute_targets(data: SelfplayOutput):
    """向量化 n-step TD 目标计算 (消除 Python 循环)"""
    max_steps, batch_size = data.reward.shape[0], data.reward.shape[1]
    n = config.td_steps
    
    # 1. 严格对齐原始掩码逻辑
    value_mask = jnp.cumsum(data.terminated[::-1, :], axis=0)[::-1, :] >= 1
    
    # 2. 准备数据
    padded_reward = jnp.concatenate([data.reward, jnp.zeros((n, batch_size))], axis=0)
    padded_discount = jnp.concatenate([data.discount, jnp.zeros((n, batch_size))], axis=0)
    padded_root_v = jnp.concatenate([data.root_value, jnp.zeros((n, batch_size))], axis=0)
    
    # 使用 jax.lax.scan 向量化 TD 累加 (G_t = r_t + gamma_t * G_{t+1})
    # 我们从末尾向前扫描
    def scan_body(carry_v, t):
        # 当前步的奖励和折扣
        curr_r = padded_reward[t]
        curr_d = padded_discount[t]
        # 下一步的 G 值
        next_v = curr_r + curr_d * carry_v
        return next_v, next_v

    # n-step TD 递归计算: G_t = r_t + γ_t * G_{t+1}
    # 注：这里使用 Python 循环展开，因为：
    # 1. td_steps 通常较小 (10-20)，展开开销可接受
    # 2. JAX 切片要求静态索引，fori_loop 内的动态索引无法用于切片
    res_val = padded_root_v[n : max_steps + n]
    for i in reversed(range(n)):
        res_val = padded_reward[i : max_steps + i] + padded_discount[i : max_steps + i] * res_val
        
    return Sample(
        obs=data.obs, 
        policy_tgt=data.action_weights, 
        value_tgt=res_val,
        mask=value_mask,
    )

# ============================================================================
# 训练与评估
# ============================================================================

def loss_fn(params, samples: Sample, rng_key):
    """损失函数
    
    - 策略损失：所有样本都参与训练（统一策略，无强弱差异）
    - 价值损失：拟合 n-step TD 目标
    """
    # obs 以 uint8 传输（节省 4x 带宽），在 GPU 上转为 float32
    obs = samples.obs.astype(jnp.float32)
    policy_tgt = samples.policy_tgt
    
    # 随机镜像增强
    do_mirror = jax.random.bernoulli(rng_key, 0.5)
    obs = jnp.where(do_mirror, jax.vmap(mirror_observation)(obs), obs)
    policy_tgt = jnp.where(do_mirror, jax.vmap(mirror_policy)(policy_tgt), policy_tgt)
    
    logits, value = forward(params, obs, is_training=True)
    
    # 强制转换为 float32 进行损失计算，避免数值不稳定
    logits = logits.astype(jnp.float32)
    value = value.astype(jnp.float32)
    policy_tgt = policy_tgt.astype(jnp.float32)
    value_tgt = samples.value_tgt.astype(jnp.float32)
    
    # 策略损失（所有样本）
    policy_ce = optax.softmax_cross_entropy(logits, policy_tgt)
    policy_loss = jnp.sum(policy_ce * samples.mask) / jnp.maximum(jnp.sum(samples.mask), 1.0)
    
    # 价值损失 (n-step TD 目标)
    value_loss = jnp.sum(optax.l2_loss(value, value_tgt) * samples.mask) / jnp.maximum(jnp.sum(samples.mask), 1.0)
    
    total_loss = policy_loss + 1.0 * value_loss
    
    return total_loss, (policy_loss, value_loss)

@jax.pmap
def evaluate(params_red, params_black, rng_key):
    """高性能评估算子：双模型对战"""
    batch_size = config.eval_games // num_devices
    
    def evaluate_recurrent_fn(params_pair, rng_key, action, state):
        p_red, p_black = params_pair
        # 分别计算两个模型的输出
        out_red, next_state = recurrent_fn(p_red, rng_key, action, state)
        out_black, _ = recurrent_fn(p_black, rng_key, action, state)
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
        logits_r, value_r = forward(params_red, obs)
        logits_b, value_b = forward(params_black, obs)
        
        logits = jnp.where(is_red[:, None], logits_r, logits_b)
        value = jnp.where(is_red, value_r, value_b)
        
        logits = jnp.where(is_red[:, None], logits, logits[:, _ROTATED_IDX])
        
        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)
        
        k_search, k_sample = jax.random.split(key)
        policy_output = mctx.gumbel_muzero_policy(
            params=(params_red, params_black), rng_key=k_search, root=root,
            recurrent_fn=evaluate_recurrent_fn,
            num_simulations=config.num_simulations, max_num_considered_actions=config.top_k,
            invalid_actions=~state.legal_action_mask,
        )
        
        # 评估多样性：前 20 步高温度采样，后续几乎贪婪（减少臭棋）
        temp = jnp.where(state.step_count < 20, 0.5, 0.02)
        
        def _sample_action(w, t, k, legal_mask):
            """温度采样 (log 空间避免数值下溢)"""
            t = jnp.maximum(t, 1e-3)
            w_masked = jnp.where(legal_mask, w, 0.0)
            log_w = jnp.log(w_masked + 1e-10)
            log_w_temp = log_w / t
            log_w_temp = jnp.where(legal_mask, log_w_temp, -jnp.inf)
            log_w_temp = log_w_temp - jnp.max(log_w_temp)
            w_temp = jnp.exp(log_w_temp)
            w_temp = jnp.where(legal_mask, w_temp, 0.0)
            w_prob = w_temp / jnp.maximum(jnp.sum(w_temp), 1e-10)
            return jax.random.choice(k, ACTION_SPACE_SIZE, p=w_prob)
        
        sample_keys = jax.random.split(k_sample, batch_size)
        action = jax.vmap(_sample_action)(policy_output.action_weights, temp, sample_keys, state.legal_action_mask)
        
        next_state = jax.vmap(env.step)(state, action)
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
# 经验回放缓冲区
# ============================================================================

class ReplayBuffer:
    """纯 NumPy 环形缓冲区 - 零设备冲突，极简稳定"""
    
    def __init__(self, max_size: int, obs_shape: tuple, action_size: int):
        self.max_size = max_size
        
        # 全部用 NumPy 数组（CPU 内存）
        self.obs = np.zeros((max_size, *obs_shape), dtype=np.uint8)
        self.policy_tgt = np.zeros((max_size, action_size), dtype=np.float32)
        self.value_tgt = np.zeros((max_size,), dtype=np.float32)
        self.mask = np.zeros((max_size,), dtype=np.bool_)
        
        self.ptr = 0
        self.size = 0
        self.total_added = 0

    def add(self, samples: Sample):
        """存入样本（从 JAX 自动转 NumPy）"""
        # jax.device_get 会自动聚合 8 GPU 的分片数据并转成 NumPy
        samples_np = jax.device_get(samples)
        
        obs_flat = samples_np.obs.reshape(-1, *samples_np.obs.shape[3:]).astype(np.uint8)
        policy_flat = samples_np.policy_tgt.reshape(-1, *samples_np.policy_tgt.shape[3:])
        value_flat = samples_np.value_tgt.reshape(-1)
        mask_flat = samples_np.mask.reshape(-1)
        
        n_new = obs_flat.shape[0]
        indices = (np.arange(n_new) + self.ptr) % self.max_size
        
        # NumPy 原地更新，永远不会有设备冲突
        self.obs[indices] = obs_flat
        self.policy_tgt[indices] = policy_flat
        self.value_tgt[indices] = value_flat
        self.mask[indices] = mask_flat
        
        self.ptr = (self.ptr + n_new) % self.max_size
        self.size = min(self.size + n_new, self.max_size)
        self.total_added += n_new
    
    def sample(self, batch_size: int, rng_key) -> Sample:
        """采样后转回 JAX 数组
        
        优化：obs 保持 uint8 传输，减少 4x CPU→GPU 带宽
        在 GPU 上的 loss_fn 中再转为 float32
        """
        # NumPy 随机采样
        idx = np.random.randint(0, self.size, size=batch_size)
        
        # obs 保持 uint8 传输（节省 4x 带宽），在 GPU 上转换
        return Sample(
            obs=jnp.array(self.obs[idx], dtype=jnp.uint8),
            policy_tgt=jnp.array(self.policy_tgt[idx]),
            value_tgt=jnp.array(self.value_tgt[idx]),
            mask=jnp.array(self.mask[idx])
        )
    
    def cleanup(self):
        pass
    
    def stats(self):
        return {"size": self.size, "total_added": self.total_added, "ptr": self.ptr}

    def state_dict(self):
        return {
            "obs": self.obs, "policy_tgt": self.policy_tgt,
            "value_tgt": self.value_tgt, "mask": self.mask,
            "ptr": self.ptr, "size": self.size, "total_added": self.total_added
        }

    def load_state_dict(self, state):
        # 获取加载的数据
        loaded_obs = np.array(state["obs"])
        loaded_policy = np.array(state["policy_tgt"])
        loaded_value = np.array(state["value_tgt"])
        loaded_mask = np.array(state["mask"])
        loaded_size = int(state["size"])
        loaded_ptr = int(state["ptr"])
        
        # 如果加载的数据比当前缓冲区小，直接复制到前面
        old_max = loaded_obs.shape[0]
        actual_samples = min(loaded_size, old_max)
        
        if old_max <= self.max_size:
            # 旧数据能放下，直接复制
            self.obs[:old_max] = loaded_obs
            self.policy_tgt[:old_max] = loaded_policy
            self.value_tgt[:old_max] = loaded_value
            self.mask[:old_max] = loaded_mask
            self.size = actual_samples
            self.ptr = loaded_ptr % self.max_size
        else:
            # 旧数据比新缓冲区大，只保留最新的部分
            self.obs[:] = loaded_obs[:self.max_size]
            self.policy_tgt[:] = loaded_policy[:self.max_size]
            self.value_tgt[:] = loaded_value[:self.max_size]
            self.mask[:] = loaded_mask[:self.max_size]
            self.size = self.max_size
            self.ptr = 0
        
        self.total_added = int(state["total_added"])

# ============================================================================
# Checkpoint 管理 (使用 orbax 官方方案)
# ============================================================================

class TrainState(NamedTuple):
    """完整训练状态，用于断点续训"""
    params: dict                    # 模型参数
    opt_state: dict                 # 优化器状态
    iteration: int                  # 当前迭代次数
    frames: int                     # 总帧数
    rng_key: jnp.ndarray           # 随机数状态
    history_models: dict            # 历史模型 (用于 ELO 评估)
    iteration_elos: dict            # ELO 记录


def create_checkpoint_manager(ckpt_dir: str) -> ocp.CheckpointManager:
    """创建 orbax checkpoint manager
    
    使用官方推荐的 PyTreeCheckpointer，支持：
    - 自动版本管理
    - 异步保存
    - 最多保留 N 个 checkpoint
    """
    # orbax 要求绝对路径
    ckpt_dir = os.path.abspath(ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    options = ocp.CheckpointManagerOptions(
        max_to_keep=config.max_to_keep,
        keep_period=config.keep_period,
        save_interval_steps=1,  # 由外部控制保存间隔
    )
    return ocp.CheckpointManager(
        directory=ckpt_dir,
        options=options,
    )


def save_checkpoint(
    ckpt_manager: ocp.CheckpointManager,
    train_state: TrainState,
    replay_buffer: ReplayBuffer,
    step: int,
):
    """保存完整训练状态
    
    包含：模型参数、优化器状态、迭代计数、随机数状态、历史模型、ELO、回放缓冲区
    """
    # 从设备获取参数（只取第一个设备的副本）
    params_np = jax.device_get(jax.tree.map(lambda x: x[0], train_state.params))
    opt_state_np = jax.device_get(jax.tree.map(lambda x: x[0], train_state.opt_state))
    rng_key_np = jax.device_get(train_state.rng_key)
    
    # 构建完整状态字典
    state_dict = {
        "params": params_np,
        "opt_state": opt_state_np,
        "iteration": np.array(train_state.iteration),
        "frames": np.array(train_state.frames),
        "rng_key": rng_key_np,
        # history_models 和 iteration_elos 单独保存为 JSON (因为 key 是 int)
    }
    
    # 保存主状态
    ckpt_manager.save(step, args=ocp.args.StandardSave(state_dict))
    
    # 单独保存 metadata (history_models keys, elos) 和 replay buffer
    meta_dir = os.path.join(os.path.abspath(config.ckpt_dir), f"meta_{step}")
    os.makedirs(meta_dir, exist_ok=True)
    
    # 保存 ELO 和历史模型索引 (转换为 Python 原生类型，避免 JSON 序列化错误)
    with open(os.path.join(meta_dir, "metadata.json"), "w") as f:
        json.dump({
            "iteration_elos": {str(k): float(v) for k, v in train_state.iteration_elos.items()},
            "history_model_keys": [int(k) for k in train_state.history_models.keys()],
        }, f)
    
    # 保存历史模型参数
    for k, v in train_state.history_models.items():
        np.savez_compressed(os.path.join(meta_dir, f"history_{k}.npz"), 
                           **{f"arr_{i}": arr for i, arr in enumerate(jax.tree.leaves(v))})
    
    print(f"[Checkpoint] 已保存 step={step}")


def restore_checkpoint(
    ckpt_manager: ocp.CheckpointManager,
    params_template: dict,
    opt_state_template: dict,
    replay_buffer: ReplayBuffer,
) -> Optional[tuple]:
    """恢复训练状态
    
    Returns:
        None 如果没有 checkpoint
        (params, opt_state, iteration, frames, rng_key, history_models, iteration_elos) 如果成功恢复
    """
    latest_step = ckpt_manager.latest_step()
    if latest_step is None:
        print("[Checkpoint] 未找到已有 checkpoint，从头开始训练")
        return None
    
    print(f"[Checkpoint] 正在恢复 step={latest_step}...")
    
    # 构建恢复目标结构 (提供给 orbax 以正确恢复)
    restore_target = {
        "params": params_template,
        "opt_state": opt_state_template,
        "iteration": np.array(0),
        "frames": np.array(0),
        "rng_key": jax.random.PRNGKey(0),
    }
    
    # 恢复主状态
    restored = ckpt_manager.restore(latest_step, args=ocp.args.StandardRestore(restore_target))
    
    params = restored["params"]
    opt_state = restored["opt_state"]
    iteration = int(restored["iteration"])
    frames = int(restored["frames"])
    rng_key = restored["rng_key"]
    
    # 恢复 metadata
    meta_dir = os.path.join(os.path.abspath(config.ckpt_dir), f"meta_{latest_step}")
    
    history_models = {}
    iteration_elos = {}
    
    if os.path.exists(meta_dir):
        # 读取元数据
        with open(os.path.join(meta_dir, "metadata.json"), "r") as f:
            meta = json.load(f)
            iteration_elos = {int(k): v for k, v in meta["iteration_elos"].items()}
            history_model_keys = meta["history_model_keys"]
        
        # 恢复历史模型
        tree_struct = jax.tree.structure(params_template)
        for k in history_model_keys:
            npz_path = os.path.join(meta_dir, f"history_{k}.npz")
            if os.path.exists(npz_path):
                data = np.load(npz_path)
                leaves = [data[f"arr_{i}"] for i in range(len(data.files))]
                history_models[k] = jax.tree.unflatten(tree_struct, leaves)
        
    # 获取最近的 ELO（可能不是当前 iteration）
    if iteration_elos:
        latest_elo_iter = max(iteration_elos.keys())
        latest_elo = iteration_elos[latest_elo_iter]
        print(f"[Checkpoint] 恢复完成: iteration={iteration}, frames={frames}, ELO={latest_elo:.0f} (iter {latest_elo_iter})")
    else:
        print(f"[Checkpoint] 恢复完成: iteration={iteration}, frames={frames}")
    
    return params, opt_state, iteration, frames, rng_key, history_models, iteration_elos


# ============================================================================
# 主循环
# ============================================================================

def main():
    print("=" * 50 + "\nZeroForge - 现代高效架构\n" + "=" * 50)
    print("特性: Pre-LN ResNet + SE + Global Pooling + Gumbel + 经验回放 + 断点续训")
    
    # 创建必要目录
    os.makedirs(config.ckpt_dir, exist_ok=True)
    
    # 初始化经验回放缓冲区 (JAX 环形缓冲区)
    replay_buffer = ReplayBuffer(
        max_size=config.replay_buffer_size,
        obs_shape=(240, 10, 9),
        action_size=ACTION_SPACE_SIZE
    )
    
    # 初始化模型模板 (用于恢复时的结构参考)
    rng_key = jax.random.PRNGKey(config.seed)
    rng_key, subkey = jax.random.split(rng_key)
    dummy_obs = jnp.zeros((config.selfplay_batch_size // num_devices, 240, 10, 9))
    variables = net.init(subkey, dummy_obs, train=True)
    params_template = variables['params']
    
    optimizer = optax.adam(config.learning_rate)
    opt_state_template = optimizer.init(params_template)
    
    # === 创建 Checkpoint Manager 并尝试恢复 ===
    ckpt_manager = create_checkpoint_manager(config.ckpt_dir)
    restored = restore_checkpoint(ckpt_manager, params_template, opt_state_template, replay_buffer)
    
    if restored is not None:
        # 从 checkpoint 恢复
        params, opt_state, iteration, frames, rng_key, history_models, iteration_elos = restored
        print(f"[断点续训] 从 iteration={iteration} 继续训练")
    else:
        # 全新训练
        params = params_template
        opt_state = opt_state_template
        iteration = 0
        frames = 0
        history_models = {0: jax.device_get(params)}
        iteration_elos = {0: 1500.0}
    
    # 分发到所有设备
    params = replicate_to_devices(params)
    opt_state = replicate_to_devices(opt_state)
    
    @partial(jax.pmap, axis_name='i')
    def train_step(params, opt_state, samples, rng_key):
        grads, (ploss, vloss) = jax.grad(loss_fn, has_aux=True)(params, samples, rng_key)
        updates, opt_state = optimizer.update(jax.lax.pmean(grads, 'i'), opt_state)
        return optax.apply_updates(params, updates), opt_state, ploss, vloss

    writer = SummaryWriter(config.log_dir)
    start_time_total = time.time()
    
    print("开始训练！")
    
    while True:
        iteration += 1
        st = time.time()
        rng_key, sk1, sk2 = jax.random.split(rng_key, 3)
        # 分发随机数 key 到各设备，每个设备拿到一个独立的 (2,) key
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            selfplay_keys = jax.device_put_sharded(list(jax.random.split(sk1, num_devices)), devices)
        data = selfplay(params, selfplay_keys)
        samples = compute_targets(data)
        
        data_np = jax.device_get(data)
        term = data_np.terminated
        winner = data_np.winner
        reasons = data_np.draw_reason
        
        # --- 数据统计 ---
        first_term = (jnp.cumsum(term, axis=1) == 1) & term
        num_games = int(first_term.sum())
        
        # 1. 胜负平统计
        r_wins = int((first_term & (winner == 0)).sum())
        b_wins = int((first_term & (winner == 1)).sum())
        draws = int((first_term & (winner == -1)).sum())
        
        # 2. 和棋原因 (1=步数, 2=无吃子, 3=重复, 4=长将)
        d_max_steps = int((first_term & (reasons == 1)).sum())
        d_no_capture = int((first_term & (reasons == 2)).sum())
        d_repetition = int((first_term & (reasons == 3)).sum())
        d_perpetual = int((first_term & (reasons == 4)).sum())
        
        # 3. 对局长度
        game_lengths = jnp.where(term, jnp.arange(config.max_steps)[None, :, None], config.max_steps)
        final_lengths = jnp.min(game_lengths, axis=1)
        avg_length = float(jnp.mean(final_lengths))
        
        # --- 将新样本添加到经验回放缓冲区 ---
        replay_buffer.add(samples)
        new_frames = samples.obs.reshape(-1, *samples.obs.shape[3:]).shape[0]
        frames += new_frames
        
        # --- 从缓冲区采样训练 ---
        num_updates = (new_frames * config.sample_reuse_times) // config.training_batch_size
        num_updates = max(1, num_updates)
        
        policy_losses, value_losses = [], []
        # 预生成所有采样 key，减少 Python 循环开销
        sample_keys = jax.random.split(rng_key, num_updates + 1)
        rng_key = sample_keys[0]
        
        for i in range(num_updates):
            sk_sample, sk_train = jax.random.split(sample_keys[i+1])
            batch_flat = replay_buffer.sample(config.training_batch_size, sk_sample)
            # 重新 reshape 为 [num_devices, batch_per_device, ...] 用于 pmap
            batch = jax.tree.map(lambda x: x.reshape((num_devices, -1) + x.shape[1:]), batch_flat)
            # 分发训练 Key 到各设备
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                train_keys = jax.device_put_sharded(list(jax.random.split(sk_train, num_devices)), devices)
            params, opt_state, ploss, vloss = train_step(params, opt_state, batch, train_keys)
            policy_losses.append(float(ploss.mean())); value_losses.append(float(vloss.mean()))
        
        # --- 清理已训练足够次数的样本 ---
        replay_buffer.cleanup()
        
        # --- 打印与日志 ---
        iter_time = time.time() - st
        fps = new_frames / iter_time
        buf_stats = replay_buffer.stats()
        
        print(f"iter={iteration:3d} | ploss={np.mean(policy_losses):.4f} vloss={np.mean(value_losses):.4f} | "
              f"len={avg_length:4.1f} fps={fps:4.0f} buf={buf_stats['size']//1000}k train={num_updates} | "
              f"红{r_wins:3d} 黑{b_wins:3d} 和{draws:3d}")
        
        # TensorBoard 记录
        writer.add_scalar("train/policy_loss", np.mean(policy_losses), iteration)
        writer.add_scalar("train/value_loss", np.mean(value_losses), iteration)
        writer.add_scalar("stats/avg_game_length", avg_length, iteration)
        writer.add_scalar("stats/fps", fps, iteration)
        writer.add_scalar("replay/buffer_size", buf_stats['size'], iteration)
        
        # 胜负和统计 (数量)
        writer.add_scalar("games/red_wins", r_wins, iteration)
        writer.add_scalar("games/black_wins", b_wins, iteration)
        writer.add_scalar("games/draws", draws, iteration)
        writer.add_scalar("games/total", num_games, iteration)
        
        # 和棋原因 (数量)
        writer.add_scalar("draw_reasons/max_steps", d_max_steps, iteration)
        writer.add_scalar("draw_reasons/no_capture", d_no_capture, iteration)
        writer.add_scalar("draw_reasons/repetition", d_repetition, iteration)
        writer.add_scalar("draw_reasons/perpetual_check", d_perpetual, iteration)
        
        # === Checkpoint 保存 (使用 orbax 官方方案) ===
        if iteration % config.ckpt_interval == 0:
            # 更新历史模型
            params_np = jax.device_get(jax.tree.map(lambda x: x[0], params))
            history_models[iteration] = params_np
            
            # 保存完整训练状态
            train_state = TrainState(
                params=params,
                opt_state=opt_state,
                iteration=iteration,
                frames=frames,
                rng_key=rng_key,
                history_models=history_models,
                iteration_elos=iteration_elos,
            )
            save_checkpoint(ckpt_manager, train_state, replay_buffer, iteration)
        
        if iteration % config.eval_interval == 0:
            # 找到最近的可用历史模型
            past_iter = max(0, iteration - config.past_model_offset)
            available_iters = sorted(history_models.keys())
            past_iter = max([k for k in available_iters if k <= past_iter], default=0)
            
            if past_iter in history_models:
                past_params = replicate_to_devices(history_models[past_iter])
                rng_key, sk5, sk6 = jax.random.split(rng_key, 3)
                # 分发评估 Key 到各设备
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    eval_keys_r = jax.device_put_sharded(list(jax.random.split(sk5, num_devices)), devices)
                    eval_keys_b = jax.device_put_sharded(list(jax.random.split(sk6, num_devices)), devices)
                
                # 双边评估
                winners_r = evaluate(params, past_params, eval_keys_r)
                winners_b = evaluate(past_params, params, eval_keys_b)
                wr = (winners_r == 0).sum()
                wb = (winners_b == 1).sum()
                score = (wr + wb + 0.5 * (config.eval_games * 2 - wr - wb)) / (config.eval_games * 2)
                elo_diff = 400.0 * np.log10(score / (1.0 - score)) if 0 < score < 1 else (400 if score >= 1 else -400)
                iteration_elos[iteration] = iteration_elos.get(past_iter, 1500.0) + elo_diff
                print(f"评估 vs Iter {past_iter}: 胜率 {score:.2%}, ELO {iteration_elos[iteration]:.0f}")
                writer.add_scalar("eval/elo", iteration_elos[iteration], iteration)
            else:
                print(f"[警告] 无可用历史模型，跳过本次评估")

if __name__ == "__main__":
    main()

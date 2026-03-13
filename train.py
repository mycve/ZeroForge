#!/usr/bin/env python3
"""
ZeroForge - 中国象棋 Gumbel AlphaZero

架构：
- Gumbel-Top-k MCTS：搜索内探索，前 40 半步温度采样，后续 argmax
- 视角归一化：obs 始终以当前行棋方为视角；policy_tgt 与 obs 保持同一视角
- 镜像增强：30% 概率左右翻转 obs 与 policy_tgt
- 标准 ELO 评估
"""

import logging
import os
import sys
import time
import json
import warnings
from functools import partial
from typing import NamedTuple, Optional, List, Tuple

# 显存分配策略：减少碎片导致的大块申请失败（需在 import jax 前设置）
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

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

from xiangqi.env import XiangqiEnv, NUM_OBSERVATION_CHANNELS
from xiangqi.actions import rotate_action, ACTION_SPACE_SIZE, BOARD_HEIGHT, BOARD_WIDTH
from xiangqi.mirror import mirror_observation, mirror_policy
from networks.alphazero import AlphaZeroNetwork
from tensorboardX import SummaryWriter


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)
logger = logging.getLogger("zeroforge")
# 屏蔽第三方库的控制台输出（orbax、jax、absl 等）
for _name in ("jax", "jax._src", "orbax", "orbax.checkpoint", "mctx", "absl", "absl.logging"):
    logging.getLogger(_name).setLevel(logging.WARNING)

# ============================================================================
# 配置
# ============================================================================

class Config:
    # 基础配置
    seed: int = 42
    ckpt_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # 网络架构：3分支GNN（Local 8邻居+Row+Col，无Global）+ factorized policy head
    num_channels: int = 128   # 128 是当前稳妥默认；96 更快但上限略低
    num_blocks: int = 8       # 8 层是当前速度/强度折中；10 层更稳，6 层适合快实验
    # RTX 50 系上 BF16 通常具备接近 FP16 的速度，同时比 FP16 更稳
    network_dtype: str = "bfloat16"
    
    # 训练超参数
    learning_rate: float = 2e-4       # AdamW 起始 LR
    lr_warmup_steps: int = 2000       # 预热步数
    # LR 余弦退火：warmup 后平滑衰减到 min_ratio，无需手动调参
    lr_cosine_steps: int = 200000     # 余弦周期（opt steps）
    lr_min_ratio: float = 0.1        # 最低 LR = peak × 0.01 = 1e-5
    training_batch_size: int = 4096
    td_lambda: float = 0.75
    
    # 自对弈与搜索：Gumbel-Top-k，前期开局温度采样，后续 visit argmax
    selfplay_batch_size: int = 1024
    num_simulations: int = 96            # Gumbel 低模拟即可，快速生成对局更重要
    top_k: int = 24                       # 根节点候选数，Gumbel 无需高 top_k
    selfplay_temperature_steps: int = 40  # 前 40 半步用温度采样，后续直接 argmax
    selfplay_temperature: float = 1.2
    
    # 经验回放配置（纯均匀采样，AlphaZero 标准）
    replay_buffer_size: int = 1_000_000
    sample_reuse_times: int = 2         # 2 是当前推荐默认；1 更稳，3 以上更容易吃旧样本
    
    # 损失权重
    value_loss_weight: float = 1.0
    weight_decay: float = 1e-4
    qtransform_value_scale: float = 0.15   # 放大 Q 值差异，提升高收益分支被选概率
    selfplay_gumbel_scale: float = 1.5   # Gumbel 噪声强度（mctx 固定参数，无需动态调节）
    eval_gumbel_scale: float = 0.10         # 评估关闭 Gumbel 噪声，结果更稳定
    
    # 环境规则（符合象棋竞赛规则）
    max_steps: int = 300              # 总步数 400 步（200回合）判和
    max_no_capture_steps: int = 120   # 无吃子 120 步（60回合）判和，将军最多累计20回合
    repetition_threshold: int = 5     # 非将非捉重复局面 5 次判和
    # 长将/长捉规则已在 violation_rules.py 中实现
    
    # ELO 评估
    eval_interval: int = 20
    eval_games: int = 100
    past_model_offset: int = 20
    # Checkpoint 配置
    ckpt_interval: int = 10         # 每 N 次迭代保存 checkpoint
    max_to_keep: int = 20            # 最多保留 N 个 checkpoint
    keep_period: int = 50           # 每 N 次迭代永久保留一个 checkpoint

config = Config()
_QTRANSFORM = partial(
    mctx.qtransform_completed_by_mix_value,
    value_scale=config.qtransform_value_scale,
)

# ============================================================================
# 环境和设备
# ============================================================================

devices = jax.local_devices()
num_devices = len(devices)

def _align_to_device_multiple(value: int, field_name: str) -> int:
    """将批量参数对齐到设备数整数倍，避免 pmap 前 reshape 出错。"""
    if num_devices <= 0:
        raise RuntimeError("未检测到可用设备")
    if value <= 0:
        aligned = num_devices
    else:
        aligned = (value // num_devices) * num_devices
        if aligned == 0:
            aligned = num_devices
    if aligned != value:
        logger.info("[Config] %s 从 %s 自动调整为 %s (num_devices=%s)", field_name, value, aligned, num_devices)
    return aligned


def _shard_batch_for_devices(batch_flat):
    """将平铺 batch 安全切分到多设备，必要时裁掉尾部样本。"""
    first_leaf = jax.tree.leaves(batch_flat)[0]
    total = int(first_leaf.shape[0])
    if total < num_devices:
        raise ValueError(
            f"训练 batch 太小: {total} < num_devices({num_devices})，请增大 training_batch_size"
        )
    per_device = total // num_devices
    usable = per_device * num_devices
    if usable != total:
        logger.info("[Batch] 训练 batch %s 无法整除设备数 %s，自动裁剪为 %s", total, num_devices, usable)

    def _reshape(x):
        x = x[:usable]
        return x.reshape((num_devices, per_device) + x.shape[1:])

    return jax.tree.map(_reshape, batch_flat)


# 动态设备数下的批量自动对齐（避免 reshape/device_put_sharded 错误）
config.selfplay_batch_size = _align_to_device_multiple(config.selfplay_batch_size, "selfplay_batch_size")
config.training_batch_size = _align_to_device_multiple(config.training_batch_size, "training_batch_size")
config.eval_games = _align_to_device_multiple(config.eval_games, "eval_games")

if config.top_k < 2:
    raise ValueError(f"top_k 至少为 2，当前值: {config.top_k}")
if config.learning_rate <= 0:
    raise ValueError(f"learning_rate 必须 > 0，当前值: {config.learning_rate}")
if config.lr_warmup_steps < 0:
    raise ValueError(f"lr_warmup_steps 必须 >= 0，当前值: {config.lr_warmup_steps}")
# 预计算 180° 旋转索引：action i -> action rotate(i)，用于黑方视角与绝对坐标系互转
# 动作空间始终以红方（绝对）坐标系定义，黑方时 obs 翻转、logits 需旋转后与绝对目标对齐
_ROTATED_IDX = rotate_action(jnp.arange(ACTION_SPACE_SIZE))

def replicate_to_devices(pytree):
    """将 pytree 复制到所有设备"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return jax.device_put_replicated(pytree, devices)


def _format_duration(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:d}h{m:02d}m{s:02d}s"
    if m > 0:
        return f"{m:d}m{s:02d}s"
    return f"{s:d}s"



env = XiangqiEnv(
    max_steps=config.max_steps,
    max_no_capture_steps=config.max_no_capture_steps,
    repetition_threshold=config.repetition_threshold,
)

_DTYPE_MAP = {
    "float32": jnp.float32,
    "bfloat16": jnp.bfloat16,
}
if config.network_dtype not in _DTYPE_MAP:
    raise ValueError(f"非法网络 dtype: {config.network_dtype}, 仅支持 {list(_DTYPE_MAP.keys())}")

net = AlphaZeroNetwork(
    action_space_size=env.action_space_size,
    channels=config.num_channels,
    num_blocks=config.num_blocks,
    dtype=_DTYPE_MAP[config.network_dtype],
)

eval_net = AlphaZeroNetwork(
    action_space_size=env.action_space_size,
    channels=config.num_channels,
    num_blocks=config.num_blocks,
    dtype=jnp.float32,
)

def forward(params, obs, is_training=False, rng_key=None):
    """前向传播: 返回 (logits, value_scalar, wdl_logits)

    训练时需传入 rng_key 以支持 GraphBlock 内 dropout。
    """
    if is_training and rng_key is not None:
        logits, value, wdl_logits = net.apply(
            {'params': params}, obs, train=True, rngs={'dropout': rng_key}
        )
    else:
        logits, value, wdl_logits = net.apply({'params': params}, obs, train=is_training)
    return logits, value, wdl_logits


def eval_forward(params, obs):
    """评估前向传播：固定 float32，规避部分 GPU 上 BF16 Triton 编译问题。"""
    logits, value, _wdl = eval_net.apply({'params': params}, obs, train=False)
    return logits, value


def recurrent_fn(params, rng_key, action, state):
    """MCTS 递归函数（瓶颈在 forward 推理 ~85%+，env.step 占比极小）
    
    黑方时 obs 已翻转，网络输出为黑方坐标系；旋转到绝对坐标系供 MCTS 使用。
    """
    prev_player = state.current_player
    state = jax.vmap(env.step)(state, action)
    obs = jax.vmap(env.observe)(state)
    logits, value, _ = forward(params, obs)
    # 黑方视角：logits 旋转到绝对坐标系，与 legal_action_mask（绝对）一致
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
    """自对弈单步输出"""
    obs: jnp.ndarray              # 观察（红方=绝对视角，黑方=180° 翻转）
    reward: jnp.ndarray
    terminated: jnp.ndarray
    action_weights: jnp.ndarray   # 策略目标，与 obs 保持同一归一化视角
    discount: jnp.ndarray
    winner: jnp.ndarray           # 0=红胜 1=黑胜 -1=和棋
    draw_reason: jnp.ndarray
    root_value: jnp.ndarray       # MCTS 搜索后的标量价值估计
    root_wdl: jnp.ndarray        # 网络原始 WDL 概率 (B,3)，用于 WDL TD(λ) bootstrap
    root_visit_entropy: jnp.ndarray  # 根节点 visit 分布熵，用于监控探索程度

class Sample(NamedTuple):
    """训练样本（从 SelfplayOutput 经 compute_targets 得到）"""
    obs: jnp.ndarray
    policy_tgt: jnp.ndarray       # 策略目标，与 obs 保持同一归一化视角
    wdl_tgt: jnp.ndarray         # WDL TD(λ) 目标 (B,3)：[W, D, L] 概率分布
    mask: jnp.ndarray            # 有效步掩码（游戏结束前）

# ============================================================================
# 自玩
# ============================================================================


@partial(jax.pmap, static_broadcasted_argnums=(2,))
def selfplay(params, rng_key, batch_size):
    """
    高性能自玩算子：
    - lax.scan 消除 Python 循环开销
    - Gumbel-Top-k：探索在搜索内完成，无需 Dirichlet/温度/根节点采样，根节点直接 argmax
    """
    def step_fn(state, key):
        key_search, key_sample, key_reset = jax.random.split(key, 3)
        obs = jax.vmap(env.observe)(state)
        logits, value, wdl_logits = forward(params, obs)
        root_wdl = jax.nn.softmax(wdl_logits, axis=-1)  # (B, 3) WDL 概率
        # 黑方时 obs 已翻转，logits 旋转到绝对坐标系，供 MCTS 与 legal_action_mask 使用
        logits = jnp.where(state.current_player[:, None] == 0, logits, logits[:, _ROTATED_IDX])
        
        # MCTS 搜索
        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)
        policy_output = mctx.gumbel_muzero_policy(
            params=params, rng_key=key_search, root=root, recurrent_fn=recurrent_fn,
            num_simulations=config.num_simulations,
            max_num_considered_actions=config.top_k,
            invalid_actions=~state.legal_action_mask,
            qtransform=_QTRANSFORM,
            gumbel_scale=config.selfplay_gumbel_scale,
        )
        action_weights = policy_output.action_weights
        root_value = policy_output.search_tree.node_values[:, 0]
        
        # 仅有一个合法走法的局面：直接用唯一走法覆盖 MCTS 结果
        legal_counts = jnp.sum(state.legal_action_mask, axis=-1)
        only_one_move = legal_counts == 1
        action_idx = jnp.argmax(state.legal_action_mask, axis=-1)
        one_hot = jax.nn.one_hot(action_idx, ACTION_SPACE_SIZE, dtype=jnp.float32)
        action_weights = jnp.where(only_one_move[:, None], one_hot, action_weights)
        root_value = jnp.where(only_one_move, value, root_value)

        def _sample_action(w, legal_mask, sample_key):
            # 在 log 空间做温度缩放，避免小概率动作下溢。
            w_masked = jnp.where(legal_mask, w, 0.0)
            log_w = jnp.log(w_masked + 1e-10) / config.selfplay_temperature
            log_w = jnp.where(legal_mask, log_w, -jnp.inf)
            log_w = log_w - jnp.max(log_w)
            probs = jnp.exp(log_w)
            probs = jnp.where(legal_mask, probs, 0.0)
            probs = probs / jnp.maximum(jnp.sum(probs), 1e-10)
            return jax.random.choice(sample_key, ACTION_SPACE_SIZE, p=probs)

        # 前 40 半步增加根节点多样性，后续回到贪心走子。
        sample_keys = jax.random.split(key_sample, batch_size)
        sampled_action = jax.vmap(_sample_action)(action_weights, state.legal_action_mask, sample_keys)
        _MASK_VAL = -1e9  # 非法动作掩码，避免 -inf 带来的数值隐患
        action_weights_masked = jnp.where(state.legal_action_mask, action_weights, _MASK_VAL)
        greedy_action = jnp.argmax(action_weights_masked, axis=-1)
        use_temperature = state.step_count < config.selfplay_temperature_steps
        action = jnp.where(only_one_move, action_idx, jnp.where(use_temperature, sampled_action, greedy_action))
        
        actor = state.current_player
        
        # 执行动作
        next_state = jax.vmap(env.step)(state, action)
        
        # 恢复统一视角：策略目标与 obs 保持同一归一化视角。
        normalized_action_weights = jnp.where(
            state.current_player[:, None] == 0,
            action_weights,
            action_weights[:, _ROTATED_IDX],
        )
        
        # 根节点 visit 分布熵：-sum(p*log(p))，高熵=探索充分，低熵=决策集中
        w_masked = jnp.where(state.legal_action_mask, action_weights, 0.0)
        total = jnp.sum(w_masked, axis=-1, keepdims=True) + 1e-10
        p = jnp.where(state.legal_action_mask, w_masked / total, 0.0)
        root_visit_entropy = -jnp.sum(p * jnp.log(p + 1e-10), axis=-1)
        
        data = SelfplayOutput(
            obs=obs, action_weights=normalized_action_weights,
            reward=next_state.rewards[jnp.arange(batch_size), actor],
            terminated=next_state.terminated,
            discount=jnp.where(next_state.terminated, 0.0, -1.0),
            winner=next_state.winner, draw_reason=next_state.draw_reason,
            root_value=root_value,
            root_wdl=root_wdl,
            root_visit_entropy=root_visit_entropy,
        )
        
        # 结束后直接重置为标准初始局面
        next_state_reset = jax.vmap(lambda s, k: jax.lax.cond(
            s.terminated, lambda: env.init(k), lambda: s
        ))(next_state, jax.random.split(key_reset, batch_size))
        return next_state_reset, data

    key_init, key_scan = jax.random.split(rng_key, 2)
    state = jax.vmap(env.init)(jax.random.split(key_init, batch_size))
    _, data = jax.lax.scan(step_fn, state, jax.random.split(key_scan, config.max_steps))
    return data

@jax.pmap
def compute_targets(data: SelfplayOutput):
    """WDL TD(λ) 目标计算 - 在 [W, D, L] 三分量上独立做 TD(λ)
    
    与标量 TD(λ) 的区别：
    - bootstrap 用网络原始 WDL 概率（而非 MCTS 标量值）
    - 视角翻转用 W↔L 互换（对手的 [W,D,L] -> 我方的 [L,D,W]）
    - 游戏结果直接映射为 one-hot WDL（reward +1/0/-1 -> [W,D,L]）
    
    输出 Sample：policy_tgt 与 obs 使用同一归一化视角。
    """
    max_steps, batch_size = data.reward.shape[0], data.reward.shape[1]
    lam = config.td_lambda
    
    # 掩码：游戏结束前的步骤参与训练
    value_mask = jnp.cumsum(data.terminated[::-1, :], axis=0)[::-1, :] >= 1
    
    # s_{t+1} 的 WDL 预测，用于 bootstrap
    wdl_next = jnp.concatenate(
        [data.root_wdl[1:], jnp.zeros((1, batch_size, 3))], axis=0
    )
    
    # 游戏结果 → one-hot WDL（仅在 terminated 时有效）
    # reward: +1=赢, 0=和, -1=输（从 actor 视角）
    reward_wdl = jnp.stack([
        jnp.maximum(data.reward, 0.0),       # W
        1.0 - jnp.abs(data.reward),           # D
        jnp.maximum(-data.reward, 0.0),       # L
    ], axis=-1)  # (max_steps, batch_size, 3)
    
    def scan_fn(carry_wdl, inputs):
        """从后向前计算 WDL λ-return
        
        carry_wdl: G_{t+1} WDL，从 step t+1 的 actor 视角
        非终局：G_t = flip((1-λ) * WDL(s_{t+1}) + λ * G_{t+1})
        终局：G_t = reward_wdl
        """
        reward_wdl_t, terminated_t, wdl_next_t = inputs
        # 混合 bootstrap 和递推
        blended = (1.0 - lam) * wdl_next_t + lam * carry_wdl
        # 视角翻转：对手的 [W, D, L] → 我方的 [L, D, W]
        flipped = blended[:, ::-1]
        # 终局用游戏结果，非终局用翻转后的混合值
        g_wdl = jnp.where(terminated_t[:, None], reward_wdl_t, flipped)
        return g_wdl, g_wdl
    
    _, wdl_tgt_rev = jax.lax.scan(
        scan_fn,
        jnp.zeros((batch_size, 3)),
        (reward_wdl[::-1], data.terminated[::-1], wdl_next[::-1]),
    )
    wdl_tgt = wdl_tgt_rev[::-1]

    return Sample(
        obs=data.obs, 
        policy_tgt=data.action_weights, 
        wdl_tgt=wdl_tgt,
        mask=value_mask,
    )

# ============================================================================
# 自对弈统计（GPU 端计算，避免传输完整 data 到 CPU）
# ============================================================================

@jax.pmap
def compute_selfplay_stats(data: SelfplayOutput):
    """在 GPU 上计算自对弈统计标量，仅传回 ~100 字节而非 ~3GB 原始数据
    
    返回 (stats[12], avg_length, root_visit_entropy_mean) 均为 float32
    stats: [总局数, 红胜, 黑胜, 和棋, 步数到限, 无吃子到限, 重复和棋, 长将判负, ...]
    """
    term = data.terminated       # (max_steps, batch)
    winner = data.winner
    reasons = data.draw_reason
    max_steps = term.shape[0]
    
    first_term = (jnp.cumsum(term, axis=0) == 1) & term
    
    stats = jnp.stack([
        first_term.sum(),                            # 总对局数
        (first_term & (winner == 0)).sum(),          # 红胜
        (first_term & (winner == 1)).sum(),          # 黑胜
        (first_term & (winner == -1)).sum(),         # 和棋
        (first_term & (reasons == 1)).sum(),         # 步数到限
        (first_term & (reasons == 2)).sum(),         # 无吃子到限
        (first_term & (reasons == 3)).sum(),         # 重复局面和棋
        (first_term & (reasons == 4)).sum(),         # 长将判负
        (first_term & (reasons == 5)).sum(),         # 无进攻子力
        (first_term & (reasons == 6)).sum(),         # 长捉判负
        (first_term & (reasons == 7)).sum(),         # 将捉交替判负
        (first_term & (reasons == 8)).sum(),         # 将死/困毙
    ]).astype(jnp.float32)
    
    step_idx = jnp.arange(max_steps)[:, None]
    game_lengths = jnp.where(term, step_idx, max_steps).astype(jnp.float32)
    avg_length = jnp.mean(jnp.min(game_lengths, axis=0))
    
    root_visit_entropy_mean = jnp.mean(data.root_visit_entropy)
    
    return stats, avg_length, root_visit_entropy_mean

# ============================================================================
# 训练与评估
# ============================================================================

def loss_fn(params, samples: Sample, rng_key):
    """损失函数
    
    - 策略损失：交叉熵。policy_tgt 与 obs 处于同一归一化视角，可直接比较。
    - 价值损失：WDL 交叉熵（比标量 MSE 梯度信号更强，精确区分"和棋"与"不确定"）
    """
    obs = samples.obs.astype(jnp.float32)
    policy_tgt = samples.policy_tgt

    # 随机左右镜像增强（30% 概率）
    rng_mirror, rng_dropout = jax.random.split(rng_key, 2)
    do_mirror = jax.random.bernoulli(rng_mirror, 0.3)
    obs = jnp.where(do_mirror, jax.vmap(mirror_observation)(obs), obs)
    policy_tgt = jnp.where(do_mirror, jax.vmap(mirror_policy)(policy_tgt), policy_tgt)

    logits, _value, wdl_logits = forward(params, obs, is_training=True, rng_key=rng_dropout)
    
    logits = logits.astype(jnp.float32)
    wdl_logits = wdl_logits.astype(jnp.float32)
    policy_tgt = policy_tgt.astype(jnp.float32)
    wdl_tgt = samples.wdl_tgt.astype(jnp.float32)  # (B, 3) WDL 概率分布

    # 策略损失（交叉熵）
    policy_ce = optax.softmax_cross_entropy(logits, policy_tgt)
    policy_loss = jnp.sum(policy_ce * samples.mask) / jnp.maximum(jnp.sum(samples.mask), 1.0)
    
    # 价值损失：WDL 交叉熵（目标已由 compute_targets 正确计算）
    value_loss_per = -jnp.sum(wdl_tgt * jax.nn.log_softmax(wdl_logits, axis=-1), axis=-1)
    value_loss = jnp.sum(value_loss_per * samples.mask) / jnp.maximum(jnp.sum(samples.mask), 1.0)
    
    total_loss = policy_loss + config.value_loss_weight * value_loss
    return total_loss, (policy_loss, value_loss)

def _build_eval_initial_states(batch_size: int, rng_key) -> "XiangqiState":
    """构建评估用初始状态（纯标准初始局面）"""
    batch_per_device = batch_size // num_devices
    if batch_per_device * num_devices != batch_size:
        raise ValueError(f"batch_size {batch_size} 必须整除 num_devices {num_devices}")
    keys = jax.random.split(rng_key, batch_size)
    states_flat = jax.vmap(env.init)(keys)

    def _shard(x):
        return x.reshape(num_devices, batch_per_device, *x.shape[1:])

    return jax.tree.map(_shard, states_flat)


@jax.pmap
def evaluate(params_red, params_black, initial_states, rng_key):
    """高性能评估算子：双模型对战
    
    评估策略：
    - 全程贪婪走子，降低评估噪声，提升版本可比性
    - initial_states: 已按设备分片的初始状态
    """
    batch_size = initial_states.board.shape[0]  # shape[0]=batch, shape[1]=10(行)
    
    def recurrent_fn_eval(params, rng_key, action, state):
        prev_player = state.current_player
        state = jax.vmap(env.step)(state, action)
        obs = jax.vmap(env.observe)(state)
        logits, value = eval_forward(params, obs)
        # 黑方视角：logits 旋转到绝对坐标系
        logits = jnp.where(state.current_player[:, None] == 0, logits, logits[:, _ROTATED_IDX])
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)
        
        reward = state.rewards[jnp.arange(state.rewards.shape[0]), prev_player]
        value = jnp.where(state.terminated, 0.0, value)
        discount = jnp.where(state.terminated, 0.0, -1.0)
        
        return mctx.RecurrentFnOutput(
            reward=reward,
            discount=discount,
            prior_logits=logits,
            value=value,
        ), state

    def evaluate_recurrent_fn(params_pair, rng_key, action, state):
        p_red, p_black = params_pair
        # 分别计算两个模型的输出
        out_red, next_state = recurrent_fn_eval(p_red, rng_key, action, state)
        out_black, _ = recurrent_fn_eval(p_black, rng_key, action, state)
        # 根据搜索模型索引进行合并
        use_red = state.search_model_index == 0
        out = jax.tree.map(
            lambda r, b: jnp.where(use_red[:, None] if r.ndim > 1 else use_red, r, b),
            out_red, out_black
        )
        return out, next_state
    
    def step_fn(state, key):
        # 根节点：红方用 params_red，黑方用 params_black
        is_red = state.current_player == 0
        state = state.replace(search_model_index=jnp.where(is_red, 0, 1).astype(jnp.int32))
        
        obs = jax.vmap(env.observe)(state)
        logits_r, value_r = eval_forward(params_red, obs)
        logits_b, value_b = eval_forward(params_black, obs)
        
        logits = jnp.where(is_red[:, None], logits_r, logits_b)
        value = jnp.where(is_red, value_r, value_b)
        
        logits = jnp.where(is_red[:, None], logits, logits[:, _ROTATED_IDX])
        
        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)
        
        policy_output = mctx.gumbel_muzero_policy(
            params=(params_red, params_black), rng_key=key, root=root,
            recurrent_fn=evaluate_recurrent_fn,
            num_simulations=config.num_simulations, max_num_considered_actions=config.top_k,
            invalid_actions=~state.legal_action_mask,
            qtransform=_QTRANSFORM,
            gumbel_scale=config.eval_gumbel_scale,
        )
        
        masked_policy = jnp.where(state.legal_action_mask, policy_output.action_weights, -1.0)
        action = jnp.argmax(masked_policy, axis=-1).astype(jnp.int32)
        
        next_state = jax.vmap(env.step)(state, action)
        return next_state, next_state.terminated

    state = initial_states
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
    """纯 NumPy 环形缓冲区，均匀采样（AlphaZero 标准）
    
    存储：obs, policy_tgt(与 obs 同视角), wdl_tgt, mask
    """
    
    def __init__(self, max_size: int, obs_shape: tuple, action_size: int):
        self.max_size = max_size
        self.obs = np.zeros((max_size, *obs_shape), dtype=np.uint8)
        self.policy_tgt = np.zeros((max_size, action_size), dtype=np.float32)
        self.wdl_tgt = np.zeros((max_size, 3), dtype=np.float32)
        self.mask = np.zeros((max_size,), dtype=np.bool_)
        self.ptr = 0
        self.size = 0
        self.total_added = 0

    @staticmethod
    def _build_rng(rng_key):
        key_np = np.asarray(rng_key, dtype=np.uint32).reshape(-1)
        if key_np.size >= 2:
            seed = (np.uint64(key_np[0]) << np.uint64(32)) ^ np.uint64(key_np[1])
        elif key_np.size == 1:
            seed = np.uint64(key_np[0])
        else:
            seed = np.uint64(np.random.randint(0, np.iinfo(np.uint32).max, dtype=np.uint32))
        return np.random.default_rng(int(seed))

    def add(self, samples: Sample):
        """存入样本（JAX -> NumPy，含 obs/policy_tgt/wdl_tgt/mask）"""
        samples_np = jax.device_get(samples)
        
        obs_flat = samples_np.obs.reshape(-1, *samples_np.obs.shape[3:]).astype(np.uint8)
        policy_flat = samples_np.policy_tgt.reshape(-1, *samples_np.policy_tgt.shape[3:])
        wdl_flat = samples_np.wdl_tgt.reshape(-1, 3)
        mask_flat = samples_np.mask.reshape(-1)
        
        n_new = obs_flat.shape[0]
        indices = (np.arange(n_new) + self.ptr) % self.max_size
        
        self.obs[indices] = obs_flat
        self.policy_tgt[indices] = policy_flat
        self.wdl_tgt[indices] = wdl_flat
        self.mask[indices] = mask_flat
        self.ptr = (self.ptr + n_new) % self.max_size
        self.size = min(self.size + n_new, self.max_size)
        self.total_added += n_new
    
    def sample(self, batch_size: int, rng_key):
        """均匀采样，转回 JAX 数组
        
        obs 保持 uint8 以减小 CPU→GPU 带宽，loss_fn 内再转为 float32。
        """
        if self.size <= 0:
            raise ValueError("ReplayBuffer 为空，无法采样")

        rng = self._build_rng(rng_key)
        valid_size = self.size
        pool_mask = self.mask[:valid_size]
        pool_idx = np.flatnonzero(pool_mask)
        if pool_idx.size == 0:
            pool_idx = np.arange(valid_size, dtype=np.int64)

        idx = rng.choice(pool_idx, size=batch_size, replace=True)
        
        obs_batch = self.obs[idx]
        policy_batch = self.policy_tgt[idx]
        wdl_batch = self.wdl_tgt[idx]
        mask_batch = self.mask[idx]
        
        sample = Sample(
            obs=jnp.asarray(obs_batch, dtype=jnp.uint8),
            policy_tgt=jnp.asarray(policy_batch),
            wdl_tgt=jnp.asarray(wdl_batch),
            mask=jnp.asarray(mask_batch),
        )
        return sample, idx.astype(np.int64)

    def cleanup(self):
        pass
    
    def stats(self):
        return {"size": self.size, "total_added": self.total_added, "ptr": self.ptr}

    def state_dict(self):
        return {
            "obs": self.obs, "policy_tgt": self.policy_tgt,
            "wdl_tgt": self.wdl_tgt, "mask": self.mask,
            "ptr": self.ptr, "size": self.size, "total_added": self.total_added
        }

    def load_state_dict(self, state):
        loaded_obs = np.array(state["obs"])
        loaded_policy = np.array(state["policy_tgt"])
        loaded_wdl = np.array(state["wdl_tgt"])
        loaded_mask = np.array(state["mask"])
        loaded_size = int(state["size"])
        loaded_ptr = int(state["ptr"])
        old_max = loaded_obs.shape[0]
        actual_samples = min(loaded_size, old_max)
        if old_max <= self.max_size:
            self.obs[:old_max] = loaded_obs
            self.policy_tgt[:old_max] = loaded_policy
            self.wdl_tgt[:old_max] = loaded_wdl
            self.mask[:old_max] = loaded_mask
            self.size = actual_samples
            self.ptr = loaded_ptr % self.max_size
        else:
            self.obs[:] = loaded_obs[:self.max_size]
            self.policy_tgt[:] = loaded_policy[:self.max_size]
            self.wdl_tgt[:] = loaded_wdl[:self.max_size]
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
    iteration_elos: dict            # ELO 记录（用于选择评估对手）
    total_opt_steps: int            # 累计优化步数（用于 LR 调度）


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
    step: int,
) -> None:
    """保存完整训练状态
    
    包含：模型参数、优化器状态、迭代计数、随机数状态、ELO、total_opt_steps
    历史模型由 orbax 直接管理，评估时从 ckpt_manager.restore(ref_iter) 加载，无需单独保存。
    """
    # 从设备获取参数（只取第一个设备的副本）
    params_np = jax.device_get(jax.tree.map(lambda x: x[0], train_state.params))
    opt_state_np = jax.device_get(jax.tree.map(lambda x: x[0], train_state.opt_state))
    rng_key_np = jax.device_get(train_state.rng_key)
    
    state_dict = {
        "params": params_np,
        "opt_state": opt_state_np,
        "iteration": np.array(train_state.iteration),
        "frames": np.array(train_state.frames),
        "rng_key": rng_key_np,
    }
    ckpt_manager.save(step, args=ocp.args.StandardSave(state_dict))
    
    # 单文件 metadata.json：iteration_elos、total_opt_steps（仅保留 orbax 保留的 step 对应的 elo）
    kept_steps = set(ckpt_manager.all_steps(read=True))
    elos_to_save = {k: v for k, v in train_state.iteration_elos.items() if k in kept_steps}
    meta_path = os.path.join(os.path.abspath(config.ckpt_dir), "metadata.json")
    with open(meta_path, "w") as f:
        json.dump({
            "iteration_elos": {str(k): float(v) for k, v in elos_to_save.items()},
            "total_opt_steps": train_state.total_opt_steps,
        }, f)
    
    logger.info("[Checkpoint] 已保存 step=%s", step)


def _load_params_from_checkpoint(
    ckpt_manager: ocp.CheckpointManager,
    step: int,
    params_template: dict,
    opt_state_template: dict,
) -> dict:
    """从 checkpoint 加载指定 step 的 params（用于 ELO 评估的对手模型）
    
    必须传入完整 restore_target 结构，否则 orbax StandardRestore 会因结构不匹配而失败。
    """
    restore_target = {
        "params": params_template,
        "opt_state": opt_state_template,
        "iteration": np.array(0),
        "frames": np.array(0),
        "rng_key": jax.random.PRNGKey(0),
    }
    restored = ckpt_manager.restore(step, args=ocp.args.StandardRestore(restore_target))
    return restored["params"]


def restore_checkpoint(
    ckpt_manager: ocp.CheckpointManager,
    params_template: dict,
    opt_state_template: dict,
) -> Optional[tuple]:
    """恢复训练状态
    
    Returns:
        None 如果没有 checkpoint
        (params, opt_state, iteration, frames, rng_key, iteration_elos, total_opt_steps) 如果成功
    """
    latest_step = ckpt_manager.latest_step()
    if latest_step is None:
        logger.info("[Checkpoint] 未找到已有 checkpoint，从头开始训练")
        return None
    
    logger.info("[Checkpoint] 正在恢复 step=%s...", latest_step)
    
    restore_target = {
        "params": params_template,
        "opt_state": opt_state_template,
        "iteration": np.array(0),
        "frames": np.array(0),
        "rng_key": jax.random.PRNGKey(0),
    }
    restored = ckpt_manager.restore(latest_step, args=ocp.args.StandardRestore(restore_target))
    params = restored["params"]
    opt_state = restored["opt_state"]
    iteration = int(restored["iteration"])
    frames = int(restored["frames"])
    rng_key = restored["rng_key"]
    
    # 从 metadata.json 恢复（兼容旧版 meta_{step}/metadata.json）
    ckpt_root = os.path.abspath(config.ckpt_dir)
    meta_path = os.path.join(ckpt_root, "metadata.json")
    if not os.path.exists(meta_path):
        meta_path = os.path.join(ckpt_root, f"meta_{latest_step}", "metadata.json")
    iteration_elos = {}
    total_opt_steps = 0
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            meta = json.load(f)
            iteration_elos = {int(k): v for k, v in meta["iteration_elos"].items()}
            total_opt_steps = int(meta["total_opt_steps"])
    
    if total_opt_steps == 0:
        total_opt_steps = (frames * config.sample_reuse_times) // config.training_batch_size
    
    if iteration_elos:
        latest_elo_iter = max(iteration_elos.keys())
        latest_elo = iteration_elos[latest_elo_iter]
        logger.info(
            "[Checkpoint] 恢复完成: iteration=%s, frames=%s, opt_steps=%s, ELO=%.0f (iter %s)",
            iteration, frames, total_opt_steps, latest_elo, latest_elo_iter,
        )
    else:
        logger.info("[Checkpoint] 恢复完成: iteration=%s, frames=%s, opt_steps=%s", iteration, frames, total_opt_steps)
    
    return params, opt_state, iteration, frames, rng_key, iteration_elos, total_opt_steps


# ============================================================================
# 主循环
# ============================================================================

def main():
    logger.info("=" * 50)
    logger.info("ZeroForge - 现代高效架构")
    logger.info("特性: 3分支GNN + factorized policy head + 统一视角训练 + Gumbel-Top-k + TD(λ)")
    
    # 创建必要目录
    os.makedirs(config.ckpt_dir, exist_ok=True)
    
    # 初始化经验回放缓冲区 (JAX 环形缓冲区)
    replay_buffer = ReplayBuffer(
        max_size=config.replay_buffer_size,
        obs_shape=(NUM_OBSERVATION_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH),
        action_size=ACTION_SPACE_SIZE,
    )
    
    # 初始化模型模板 (用于恢复时的结构参考)
    rng_key = jax.random.PRNGKey(config.seed)
    rng_key, subkey = jax.random.split(rng_key)
    init_batch_per_device = 1
    dummy_obs = jnp.zeros((init_batch_per_device, NUM_OBSERVATION_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH))
    variables = net.init(subkey, dummy_obs, train=True)
    params_template = variables['params']
    
    def lr_schedule(step):
        """warmup + 余弦退火"""
        warmup_factor = jnp.minimum(step / max(config.lr_warmup_steps, 1), 1.0)
        decay_progress = jnp.maximum(step - config.lr_warmup_steps, 0.0)
        decay_progress = jnp.minimum(decay_progress / max(config.lr_cosine_steps, 1), 1.0)
        cosine_factor = config.lr_min_ratio + (1.0 - config.lr_min_ratio) * 0.5 * (
            1.0 + jnp.cos(jnp.pi * decay_progress)
        )
        return config.learning_rate * warmup_factor * cosine_factor
    
    final_lr = config.learning_rate * config.lr_min_ratio
    logger.info(
        "[LR] 余弦退火: %.1e → %.1e, 周期=%s steps(≈%s轮), warmup=%s steps",
        config.learning_rate,
        final_lr,
        config.lr_cosine_steps,
        config.lr_cosine_steps // 266,
        config.lr_warmup_steps,
    )
    logger.info("[TD(λ)] 固定 λ=%s", config.td_lambda)
    optimizer = optax.adamw(lr_schedule, weight_decay=config.weight_decay)
    opt_state_template = optimizer.init(params_template)
    
    # === 创建 Checkpoint Manager 并尝试恢复 ===
    ckpt_manager = create_checkpoint_manager(config.ckpt_dir)
    restored = restore_checkpoint(ckpt_manager, params_template, opt_state_template)
    
    if restored is not None:
        params, opt_state, iteration, frames, rng_key, iteration_elos, total_opt_steps = restored
        logger.info("[断点续训] 从 iteration=%s 继续训练", iteration)
    else:
        params = params_template
        opt_state = opt_state_template
        iteration = 0
        frames = 0
        iteration_elos = {0: 1500.0}
        total_opt_steps = 0

    params = replicate_to_devices(params)
    opt_state = replicate_to_devices(opt_state)
    
    @partial(jax.pmap, axis_name='i')
    def train_step(params, opt_state, samples, rng_key):
        grads, (ploss, vloss) = jax.grad(loss_fn, has_aux=True)(params, samples, rng_key)
        updates, opt_state = optimizer.update(jax.lax.pmean(grads, 'i'), opt_state, params)
        return optax.apply_updates(params, updates), opt_state, ploss, vloss

    # 根据关键超参自动生成日志子目录，便于 TensorBoard 对比实验
    run_name = (
        f"ch{config.num_channels}_b{config.num_blocks}"
        f"_sim{config.num_simulations}_k{config.top_k}"
        f"_lr{config.learning_rate:.0e}_bs{config.training_batch_size}"
        f"_td{config.td_lambda}_vw{config.value_loss_weight}"
        f"_sp{config.selfplay_batch_size}"
    )
    run_log_dir = os.path.join(config.log_dir, run_name)
    logger.info("[Log] TensorBoard 日志: %s", run_log_dir)
    writer = SummaryWriter(run_log_dir)
    start_time_total = time.time()
    
    logger.info("开始训练")
    
    # 恢复后若轮到评估（如崩溃在 ckpt 保存后、eval 前），补跑评估
    if restored is not None and iteration % config.eval_interval == 0 and iteration > 0:
        kept_steps = set(ckpt_manager.all_steps(read=True))
        available_iters = sorted(k for k in kept_steps if k < iteration)
        if available_iters:
            logger.info("[断点续训] 补跑 iteration=%s 的评估...", iteration)
            rated_iters = [k for k in available_iters if k in iteration_elos]
            ref_iter = max(rated_iters, key=lambda k: iteration_elos[k]) if rated_iters else available_iters[-1]
            ref_reason = "best_elo" if rated_iters else "latest_ckpt"
            past_params = replicate_to_devices(_load_params_from_checkpoint(
                ckpt_manager, ref_iter, params_template, opt_state_template
            ))
            rng_key, sk5, sk6, sk7 = jax.random.split(rng_key, 4)
            batch_per_eval = config.eval_games
            states = _build_eval_initial_states(batch_per_eval, sk7)
            states_r = states
            states_b = states
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                eval_keys_r = jax.device_put_sharded(list(jax.random.split(sk5, num_devices)), devices)
                eval_keys_b = jax.device_put_sharded(list(jax.random.split(sk6, num_devices)), devices)
                states_r = jax.device_put_sharded([jax.tree.map(lambda x: x[i], states_r) for i in range(num_devices)], devices)
                states_b = jax.device_put_sharded([jax.tree.map(lambda x: x[i], states_b) for i in range(num_devices)], devices)
            try:
                winners_r = evaluate(params, past_params, states_r, eval_keys_r)
                winners_b = evaluate(past_params, params, states_b, eval_keys_b)
            except Exception as e:
                logger.exception("[评估错误] 补跑评估失败: %s", e)
                raise RuntimeError("断点续训补跑评估异常") from e
            wins_red = int((winners_r == 0).sum())
            draws_red = int((winners_r == -1).sum())
            losses_red = int((winners_r == 1).sum())
            wins_black = int((winners_b == 1).sum())
            draws_black = int((winners_b == -1).sum())
            losses_black = int((winners_b == 0).sum())
            wins = wins_red + wins_black
            draws = draws_red + draws_black
            losses = losses_red + losses_black
            total_games = int(config.eval_games * 2)
            score = (wins + 0.5 * draws) / total_games
            decisive_games = wins + losses
            decisive_win_rate = wins / decisive_games if decisive_games > 0 else float("nan")
            elo_diff = 400.0 * np.log10(score / (1.0 - score)) if 0 < score < 1 else (400 if score >= 1 else -400)
            iteration_elos[iteration] = iteration_elos.get(ref_iter, 1500.0) + elo_diff
            decisive_text = f"{decisive_win_rate:.2%}" if decisive_games > 0 else "N/A"
            logger.info(
                "评估 vs Iter %s (%s): W/D/L %s/%s/%s | 得分率 %.2f%% | ELO %.0f",
                ref_iter,
                ref_reason,
                wins,
                draws,
                losses,
                score * 100.0,
                iteration_elos[iteration],
            )
            writer.add_scalar("eval/elo", iteration_elos[iteration], iteration)
            writer.add_scalar("eval/win_rate", wins / total_games, iteration)
            writer.add_scalar("eval/draw_rate", draws / total_games, iteration)
            writer.add_scalar("eval/loss_rate", losses / total_games, iteration)
            writer.add_scalar("eval/score", score, iteration)
            if decisive_games > 0:
                writer.add_scalar("eval/decisive_win_rate", decisive_win_rate, iteration)
    
    logger.info(
        "[Selfplay] batch_size=%s, gumbel_scale=%s, reuse=%s, temp_steps=%s, temp=%s",
        config.selfplay_batch_size,
        config.selfplay_gumbel_scale,
        config.sample_reuse_times,
        config.selfplay_temperature_steps,
        config.selfplay_temperature,
    )

    while True:
        iteration += 1
        st = time.time()

        # 自对弈（每轮单次调用，批大小由 selfplay_batch_size 决定）
        rng_key, sk_selfplay = jax.random.split(rng_key)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            selfplay_keys = jax.device_put_sharded(list(jax.random.split(sk_selfplay, num_devices)), devices)

        try:
            data = selfplay(params, selfplay_keys, config.selfplay_batch_size // num_devices)
        except jax.errors.JaxRuntimeError as e:
            msg = str(e).lower()
            if ("resource_exhausted" in msg) or ("out of memory" in msg):
                raise RuntimeError(
                    "[Selfplay OOM] 自对弈阶段显存不足。"
                    f"建议优先减小 selfplay_batch_size({config.selfplay_batch_size})；"
                    f"其次再调小 num_simulations({config.num_simulations}) / "
                    f"top_k({config.top_k}) / num_channels({config.num_channels}) / "
                    f"num_blocks({config.num_blocks})。"
                ) from e
            raise

        # GPU 上并行计算：TD(λ) 目标 + 统计标量（两个 pmap 共享 data，XLA 自动调度）
        samples = compute_targets(data)
        gpu_stats = compute_selfplay_stats(data)
        
        # data 的 GPU 引用在此之后可被释放，提前回收 ~3GB 显存给训练用
        del data
        
        replay_buffer.add(samples)
        new_frames = config.max_steps * config.selfplay_batch_size

        # 只传输统计标量（~100 字节），不再传输完整 data（~3GB）
        stats_dev, avg_len_dev, entropy_dev = gpu_stats
        stats_np = np.array(jax.device_get(stats_dev))
        stats = stats_np.sum(axis=0).astype(np.int64)   # 跨设备求和
        avg_length = float(jax.device_get(avg_len_dev).mean())
        root_visit_entropy = float(jax.device_get(entropy_dev).mean())

        (num_games, r_wins, b_wins, draws, d_max_steps, d_no_capture, d_repetition,
         d_perpetual, d_no_attackers, d_perpetual_chase, d_check_chase_alt, d_checkmate) = [int(x) for x in stats]
        
        # --- 将新样本添加到经验回放缓冲区 ---
        frames += new_frames
        
        # --- 从缓冲区采样训练 ---
        num_updates = (new_frames * config.sample_reuse_times) // config.training_batch_size
        num_updates = max(1, num_updates)
        
        # 预生成所有采样 key
        sample_keys = jax.random.split(rng_key, num_updates + 1)
        rng_key = sample_keys[0]
        
        def _prefetch_batch(key_idx):
            """CPU 采样 + 异步传输到 GPU（双缓冲预读取）"""
            batch_flat, _ = replay_buffer.sample(config.training_batch_size, sample_keys[key_idx])
            batch = _shard_batch_for_devices(batch_flat)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                keys = jax.device_put_sharded(
                    list(jax.random.split(sample_keys[key_idx], num_devices)), devices)
            return batch, keys

        ploss_acc, vloss_acc = None, None
        next_batch, next_keys = _prefetch_batch(1) if num_updates > 0 else (None, None)

        for i in range(num_updates):
            batch, train_keys = next_batch, next_keys
            
            try:
                params, opt_state, ploss, vloss = train_step(params, opt_state, batch, train_keys)
            except jax.errors.JaxRuntimeError as e:
                msg = str(e).lower()
                if ("resource_exhausted" in msg) or ("out of memory" in msg):
                    raise RuntimeError(
                        "训练显存不足（OOM）。建议进一步降低: "
                        f"training_batch_size({config.training_batch_size}), "
                        f"num_channels({config.num_channels}), num_blocks({config.num_blocks}), "
                        f"selfplay_batch_size({config.selfplay_batch_size})."
                    ) from e
                raise

            if i + 1 < num_updates:
                next_batch, next_keys = _prefetch_batch(i + 2)
            
            # 累积损失（保持在 GPU），只在最后同步
            if ploss_acc is None:
                ploss_acc, vloss_acc = ploss, vloss
            else:
                ploss_acc = ploss_acc + ploss
                vloss_acc = vloss_acc + vloss
        
        # 累计优化器步数
        total_opt_steps += num_updates
        
        # 一次性同步所有累积损失
        if num_updates > 0:
            policy_loss = float(ploss_acc.mean() / num_updates)
            value_loss = float(vloss_acc.mean() / num_updates)
        else:
            policy_loss = 0.0
            value_loss = 0.0
        
        # --- 清理已训练足够次数的样本 ---
        replay_buffer.cleanup()
        
        # --- 打印与日志 ---
        iter_time = time.time() - st
        fps = new_frames / max(iter_time, 1e-9)
        buf_stats = replay_buffer.stats()
        total_elapsed = time.time() - start_time_total
        logger.info(
            "iter=%3d | ploss=%.4f vloss=%.4f | len=%4.1f fps=%4.0f "
            "buf=%dk train=%d | ent=%.3f | 红%3d 黑%3d 和%3d | iter_t=%s total=%s",
            iteration,
            policy_loss,
            value_loss,
            avg_length,
            fps,
            buf_stats["size"] // 1000,
            num_updates,
            root_visit_entropy,
            r_wins,
            b_wins,
            draws,
            _format_duration(iter_time),
            _format_duration(total_elapsed),
        )
        base_lr = float(lr_schedule(total_opt_steps))
        
        # TensorBoard 记录
        writer.add_scalar("train/policy_loss", policy_loss, iteration)
        writer.add_scalar("train/value_loss", value_loss, iteration)
        writer.add_scalar("train/lr", base_lr, iteration)
        writer.add_scalar("stats/avg_game_length", avg_length, iteration)
        writer.add_scalar("stats/fps", fps, iteration)
        writer.add_scalar("stats/root_visit_entropy", root_visit_entropy, iteration)
        writer.add_scalar("replay/buffer_size", buf_stats['size'], iteration)
        
        # 胜负和统计 (数量)
        writer.add_scalar("games/red_wins", r_wins, iteration)
        writer.add_scalar("games/black_wins", b_wins, iteration)
        writer.add_scalar("games/draws", draws, iteration)
        writer.add_scalar("games/total", num_games, iteration)
        
        # 结束原因统计 (数量)
        writer.add_scalar("end_reasons/max_steps", d_max_steps, iteration)
        writer.add_scalar("end_reasons/no_capture", d_no_capture, iteration)
        writer.add_scalar("end_reasons/repetition_draw", d_repetition, iteration)
        writer.add_scalar("end_reasons/perpetual_check", d_perpetual, iteration)
        writer.add_scalar("end_reasons/no_attackers", d_no_attackers, iteration)
        writer.add_scalar("end_reasons/perpetual_chase", d_perpetual_chase, iteration)
        writer.add_scalar("end_reasons/check_chase_alt", d_check_chase_alt, iteration)
        writer.add_scalar("end_reasons/checkmate", d_checkmate, iteration)
        
        # === Checkpoint 保存 ===
        if iteration % config.ckpt_interval == 0:
            train_state = TrainState(
                params=params,
                opt_state=opt_state,
                iteration=iteration,
                frames=frames,
                rng_key=rng_key,
                iteration_elos=iteration_elos,
                total_opt_steps=total_opt_steps,
            )
            save_checkpoint(ckpt_manager, train_state, iteration)
            # 裁剪 iteration_elos，与 orbax 保留数量一致，避免内存膨胀
            kept_steps = set(ckpt_manager.all_steps(read=True))
            iteration_elos = {k: v for k, v in iteration_elos.items() if k in kept_steps}
        
        if iteration % config.eval_interval == 0:
            # 评估对手优先选择“历史最佳模型”（按 iteration_elos），
            # 若无可用 ELO 记录，则退化为最近历史 checkpoint。对手从 orbax checkpoint 加载。
            kept_steps = set(ckpt_manager.all_steps(read=True))
            available_iters = sorted(k for k in kept_steps if k < iteration)
            if available_iters:
                rated_iters = [k for k in available_iters if k in iteration_elos]
                if rated_iters:
                    ref_iter = max(rated_iters, key=lambda k: iteration_elos[k])
                    ref_reason = "best_elo"
                else:
                    ref_iter = available_iters[-1]
                    ref_reason = "latest_ckpt"

                past_params = replicate_to_devices(_load_params_from_checkpoint(
                    ckpt_manager, ref_iter, params_template, opt_state_template
                ))
                rng_key, sk5, sk6, sk7 = jax.random.split(rng_key, 4)
                batch_per_eval = config.eval_games
                states = _build_eval_initial_states(batch_per_eval, sk7)
                states_r = states
                states_b = states
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    eval_keys_r = jax.device_put_sharded(list(jax.random.split(sk5, num_devices)), devices)
                    eval_keys_b = jax.device_put_sharded(list(jax.random.split(sk6, num_devices)), devices)
                    states_r = jax.device_put_sharded(
                        [jax.tree.map(lambda x: x[i], states_r) for i in range(num_devices)], devices
                    )
                    states_b = jax.device_put_sharded(
                        [jax.tree.map(lambda x: x[i], states_b) for i in range(num_devices)], devices
                    )
                
                try:
                    winners_r = evaluate(params, past_params, states_r, eval_keys_r)
                    winners_b = evaluate(past_params, params, states_b, eval_keys_b)
                except Exception as e:
                    logger.error("[评估错误] 当前模型 dtype=%s, action_size=%s", config.network_dtype, ACTION_SPACE_SIZE)
                    logger.error("[评估错误] params dtype: %s", jax.tree.leaves(params)[0].dtype)
                    logger.error("[评估错误] past_params dtype: %s", jax.tree.leaves(past_params)[0].dtype)
                    raise RuntimeError("评估阶段发生异常，请检查 GPU/Triton 与模型参数兼容性") from e

                # 当前模型先手（红）与后手（黑）拆分统计，避免“和棋被误判为输”。
                wins_red = int((winners_r == 0).sum())
                draws_red = int((winners_r == -1).sum())
                losses_red = int((winners_r == 1).sum())
                wins_black = int((winners_b == 1).sum())
                draws_black = int((winners_b == -1).sum())
                losses_black = int((winners_b == 0).sum())

                wins = wins_red + wins_black
                draws = draws_red + draws_black
                losses = losses_red + losses_black
                total_games = int(config.eval_games * 2)

                win_rate = wins / total_games
                draw_rate = draws / total_games
                loss_rate = losses / total_games
                score = (wins + 0.5 * draws) / total_games
                decisive_games = wins + losses
                decisive_win_rate = wins / decisive_games if decisive_games > 0 else float("nan")

                elo_diff = 400.0 * np.log10(score / (1.0 - score)) if 0 < score < 1 else (400 if score >= 1 else -400)
                iteration_elos[iteration] = iteration_elos.get(ref_iter, 1500.0) + elo_diff
                decisive_text = f"{decisive_win_rate:.2%}" if decisive_games > 0 else "N/A"
                logger.info(
                    "评估 vs Iter %s (%s): W/D/L %s/%s/%s | 胜率 %.2f%% 和率 %.2f%% 负率 %.2f%% | "
                    "得分率 %.2f%% | 决胜局胜率 %s | ELO %.0f",
                    ref_iter,
                    ref_reason,
                    wins,
                    draws,
                    losses,
                    win_rate * 100.0,
                    draw_rate * 100.0,
                    loss_rate * 100.0,
                    score * 100.0,
                    decisive_text,
                    iteration_elos[iteration],
                )
                logger.info(
                    "  先手(当前红) W/D/L %s/%s/%s | 后手(当前黑) W/D/L %s/%s/%s",
                    wins_red,
                    draws_red,
                    losses_red,
                    wins_black,
                    draws_black,
                    losses_black,
                )
                writer.add_scalar("eval/elo", iteration_elos[iteration], iteration)
                writer.add_scalar("eval/win_rate", win_rate, iteration)
                writer.add_scalar("eval/draw_rate", draw_rate, iteration)
                writer.add_scalar("eval/loss_rate", loss_rate, iteration)
                writer.add_scalar("eval/score", score, iteration)
                if decisive_games > 0:
                    writer.add_scalar("eval/decisive_win_rate", decisive_win_rate, iteration)
            else:
                logger.warning("[警告] 无可用历史模型，跳过本次评估")

if __name__ == "__main__":
    main()

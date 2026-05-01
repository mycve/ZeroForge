#!/usr/bin/env python3
"""
ZeroForge - 中国象棋 Gumbel AlphaZero

架构：
- Gumbel-Top-k MCTS：模拟次数见 `Config.num_simulations`；温度退火采样（下限见 `selfplay_temperature_final`）
- 视角归一化：obs 始终以当前行棋方为视角；策略目标与 obs 保持同一视角
- 镜像增强：默认 30% 概率左右翻转 obs 与策略目标
- 价值：策略蒸馏 `action_weights`；价值为 MCTS 根 `root_value` 的 TD(λ) 标量目标，对网络 `value`（由 `value_logits` 得 W−L）做 MSE
- 标准 ELO 评估
"""

import argparse
import logging
import os
import sys
import time
import json
import warnings
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from functools import partial
from typing import NamedTuple, Optional, List, Tuple

# --- JAX 持久化编译缓存（必须在 import jax 之前设置）---
# 二次启动可复用 XLA 编译结果，显著缩短首步等待；目录默认可通过环境变量覆盖。
_JAX_CACHE_DIR = os.environ.get("JAX_COMPILATION_CACHE_DIR")
if not _JAX_CACHE_DIR:
    _JAX_CACHE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "jax_cache"))
os.makedirs(_JAX_CACHE_DIR, exist_ok=True)
os.environ["JAX_COMPILATION_CACHE_DIR"] = _JAX_CACHE_DIR
# 小编译也缓存；大条目不跳过（与 uci_engine 一致）
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "1")
os.environ.setdefault("JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES", "0")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import optax
import mctx
import orbax.checkpoint as ocp

from xiangqi.env import XiangqiEnv, NUM_OBSERVATION_CHANNELS
from xiangqi.actions import rotate_action, ACTION_SPACE_SIZE, BOARD_HEIGHT, BOARD_WIDTH
from xiangqi.fen import load_fens_from_file
from xiangqi.mirror import mirror_action, mirror_observation, mirror_board_swap_colors
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
    num_blocks: int = 10       # 8 层是当前速度/强度折中；10 层更稳，6 层适合快实验
    # RTX 50 系上 BF16 通常具备接近 FP16 的速度，同时比 FP16 更稳
    network_dtype: str = "bfloat16"
    
    # 训练超参数
    learning_rate: float = 2e-4       # AdamW peak LR
    lr_warmup_steps: int = 2000       # warmup steps
    lr_cosine_steps: int = 60000      # cosine decay steps
    lr_min_ratio: float = 0.05        # final LR = peak * 0.05
    training_batch_size: int = 1024 * 8
    td_lambda: float = 0.95              # λ 越大越信任终局结果，减少早期不准确 bootstrap 的偏差
    
    # 自对弈与搜索：Gumbel-Top-k，搜索质量优先
    selfplay_batch_size: int = 1024       # 减半 batch 换取更深搜索，每步数据质量 > 数据量
    num_simulations: int = 32            # 增大可提升 MCTS 质量（更耗算力）
    top_k: int = 16
    selfplay_temperature_steps: int = 24
    selfplay_temperature: float = 1.0      # 自对弈起始温度
    selfplay_temperature_final: float = 0.1

    # 经验回放配置（纯均匀采样，AlphaZero 标准）
    replay_buffer_size: int = 1_000_000    # 配合 batch_size=2048，约 4 轮填满
    sample_reuse_times: int = 1          # 数据产出减半，多学一遍弥补
    mirror_augmentation_prob: float = 0.3  # 左右镜像增强概率；0.3 更保守，避免过度改写原分布
    
    # 损失权重
    value_loss_weight: float = 1.0
    aux_remaining_loss_weight: float = 0.05
    aux_material_loss_weight: float = 0.05
    aux_occupancy_loss_weight: float = 0.05
    aux_future_horizon: int = 8
    weight_decay: float = 1e-4
    qtransform_value_scale: float = 0.10   # 放大 Q 值差异，提升高收益分支被选概率
    selfplay_gumbel_scale: float = 1.0   # Gumbel 噪声强度（mctx 固定参数，无需动态调节）
    eval_gumbel_scale: float = 0.0       # 评估关闭 Gumbel 扰动，提升结果稳定性与可比性
    
    # 环境规则（符合象棋竞赛规则）
    max_steps: int = 300              # 总步数 400 步（200回合）判和
    max_no_capture_steps: int = 120   # 无吃子 120 步（60回合）判和，将军最多累计20回合
    repetition_threshold: int = 5     # 非将非捉重复局面 5 次判和
    # 长将/长捉规则已在 violation_rules.py 中实现
    
    # ELO 评估
    eval_interval: int = 20
    eval_games: int = 96
    eval_fens_path: str = "eval_fens.txt"
    past_model_offset: int = 20
    # Checkpoint 配置
    ckpt_interval: int = 20         # 每 N 次迭代保存 checkpoint
    max_to_keep: int = 20            # 最多保留 N 个 checkpoint
    keep_period: int = 100           # 每 N 次迭代永久保留一个 checkpoint

def parse_args():
    parser = argparse.ArgumentParser(description="ZeroForge 训练入口")
    parser.add_argument(
        "--init-checkpoint",
        type=str,
        default=None,
        help="导入指定 checkpoint 作为基础模型开始训练；支持 step 编号或形如 checkpoints/100 的路径",
    )
    parser.add_argument("--learning-rate", type=float, default=None, help="覆盖 learning_rate")
    parser.add_argument("--td-lambda", type=float, default=None, help="覆盖 td_lambda")
    parser.add_argument("--training-batch-size", type=int, default=None, help="覆盖 training_batch_size")
    parser.add_argument("--selfplay-batch-size", type=int, default=None, help="覆盖 selfplay_batch_size")
    parser.add_argument("--replay-buffer-size", type=int, default=None, help="覆盖 replay_buffer_size")
    parser.add_argument("--sample-reuse-times", type=int, default=None, help="覆盖 sample_reuse_times")
    parser.add_argument("--num-simulations", type=int, default=None, help="覆盖 num_simulations")
    parser.add_argument("--top-k", type=int, default=None, help="覆盖搜索 max_num_considered_actions")
    parser.add_argument("--selfplay-temperature", type=float, default=None, help="覆盖 selfplay_temperature")
    parser.add_argument("--selfplay-temperature-steps", type=int, default=None, help="覆盖 selfplay_temperature_steps")
    parser.add_argument("--selfplay-temperature-final", type=float, default=None, help="覆盖 selfplay_temperature_final")
    parser.add_argument("--selfplay-gumbel-scale", type=float, default=None, help="覆盖 selfplay_gumbel_scale")
    parser.add_argument("--eval-gumbel-scale", type=float, default=None, help="覆盖 eval_gumbel_scale")
    parser.add_argument("--mirror-augmentation-prob", type=float, default=None, help="覆盖 mirror_augmentation_prob")
    return parser.parse_args()


def apply_cli_overrides(args):
    overrides = {
        "learning_rate": args.learning_rate,
        "td_lambda": args.td_lambda,
        "training_batch_size": args.training_batch_size,
        "selfplay_batch_size": args.selfplay_batch_size,
        "replay_buffer_size": args.replay_buffer_size,
        "sample_reuse_times": args.sample_reuse_times,
        "num_simulations": args.num_simulations,
        "top_k": args.top_k,
        "selfplay_temperature": args.selfplay_temperature,
        "selfplay_temperature_steps": args.selfplay_temperature_steps,
        "selfplay_temperature_final": args.selfplay_temperature_final,
        "selfplay_gumbel_scale": args.selfplay_gumbel_scale,
        "eval_gumbel_scale": args.eval_gumbel_scale,
        "mirror_augmentation_prob": args.mirror_augmentation_prob,
    }
    applied = {}
    for key, value in overrides.items():
        if value is not None:
            setattr(config, key, value)
            applied[key] = value
    return applied


config = Config()
CLI_ARGS = parse_args()
CLI_OVERRIDES = apply_cli_overrides(CLI_ARGS)
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
if config.selfplay_temperature <= 0 or config.selfplay_temperature_final <= 0:
    raise ValueError(
        "selfplay_temperature 与 selfplay_temperature_final 必须 > 0，"
        f"当前值: {config.selfplay_temperature}, {config.selfplay_temperature_final}"
    )
if config.selfplay_temperature_steps < 0:
    raise ValueError(
        f"selfplay_temperature_steps 必须 >= 0，当前值: {config.selfplay_temperature_steps}"
    )
if not 0.0 <= config.mirror_augmentation_prob <= 1.0:
    raise ValueError(
        "mirror_augmentation_prob 必须在 [0, 1] 内，"
        f"当前值: {config.mirror_augmentation_prob}"
    )
# 预计算 180° 旋转索引：action i -> action rotate(i)，用于黑方视角与绝对坐标系互转
# 动作空间始终以红方（绝对）坐标系定义，黑方时 obs 翻转、logits 需旋转后与绝对目标对齐
_ROTATED_IDX = rotate_action(jnp.arange(ACTION_SPACE_SIZE))


def _masked_normalize(x: jnp.ndarray, legal_mask: jnp.ndarray) -> jnp.ndarray:
    """对合法动作子集归一化，非法动作恒为 0。"""
    masked = jnp.where(legal_mask, x, 0.0)
    denom = jnp.maximum(jnp.sum(masked, axis=-1, keepdims=True), 1e-10)
    return jnp.where(legal_mask, masked / denom, 0.0)


def _opening_temperature(step_count: jnp.ndarray) -> jnp.ndarray:
    """全程温度退火：前期更散，达到下限后保持稳定小温度采样。"""
    if config.selfplay_temperature_steps <= 1:
        temp = config.selfplay_temperature_final
        return jnp.full_like(step_count, temp, dtype=jnp.float32)
    progress = jnp.clip(
        step_count.astype(jnp.float32) / float(config.selfplay_temperature_steps - 1),
        0.0,
        1.0,
    )
    temp = (
        config.selfplay_temperature
        + (config.selfplay_temperature_final - config.selfplay_temperature) * progress
    )
    return jnp.maximum(temp, config.selfplay_temperature_final).astype(jnp.float32)


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

_MATERIAL_VALUES = jnp.array(
    [0.0, 2.0, 2.0, 4.0, 9.0, 4.5, 1.0, 1.0, 4.5, 9.0, 4.0, 2.0, 2.0, 0.0],
    dtype=jnp.float32,
)
_REMAINING_PLY_BUCKETS = jnp.array([8, 16, 32, 64], dtype=jnp.int32)

def forward(params, obs, is_training=False, rng_key=None, return_aux=True):
    """前向传播；return_aux=False 时只返回自玩/评估所需的 policy_logits 和 value。

    value = softmax(value_logits) 的 p_W − p_L；训练时价值损失为 MSE(value, value_tgt)，梯度经 value 回传至 value_logits。
    训练时需传入 rng_key 以支持 GraphBlock 内 dropout。
    """
    if is_training and rng_key is not None:
        outputs = net.apply(
            {'params': params}, obs, train=True, return_aux=return_aux, rngs={'dropout': rng_key}
        )
    else:
        outputs = net.apply({'params': params}, obs, train=is_training, return_aux=return_aux)
    return outputs


def eval_forward(params, obs):
    """评估前向传播：固定 float32，规避部分 GPU 上 BF16 Triton 编译问题。"""
    logits, value = eval_net.apply({'params': params}, obs, train=False, return_aux=False)
    return logits, value


def recurrent_fn(params, rng_key, action, state):
    """MCTS 递归函数（瓶颈在 forward 推理 ~85%+，env.step 占比极小）
    
    黑方时 obs 已翻转，网络输出为黑方坐标系；旋转到绝对坐标系供 MCTS 使用。
    """
    prev_player = state.current_player
    state = jax.vmap(env.step)(state, action)
    obs = jax.vmap(env.observe)(state)
    logits, value = forward(params, obs, return_aux=False)
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
    policy_idx: jnp.ndarray       # 稀疏策略目标动作索引，与 obs 保持同一归一化视角
    policy_prob: jnp.ndarray      # 稀疏策略目标概率
    discount: jnp.ndarray
    winner: jnp.ndarray           # 0=红胜 1=黑胜 -1=和棋
    draw_reason: jnp.ndarray
    root_value: jnp.ndarray       # MCTS 根节点标量价值（与 mctx 搜索一致，供 TD(λ) 价值目标）
    root_visit_entropy: jnp.ndarray  # 根节点 visit 分布熵，用于监控探索程度
    step_count: jnp.ndarray       # 当前局面步数（半步），用于分段熵统计

class Sample(NamedTuple):
    """训练样本（从 SelfplayOutput 经 compute_targets 得到）"""
    obs: jnp.ndarray
    policy_idx: jnp.ndarray       # 稀疏策略目标的动作索引 (top-k)
    policy_prob: jnp.ndarray      # 稀疏策略目标的概率 (top-k)
    value_tgt: jnp.ndarray        # MCTS 根价值 TD(λ) 标量目标（与 Gumbel AlphaZero / mctx 一致）
    remaining_ply_bucket: jnp.ndarray
    future_material_delta: jnp.ndarray
    future_occupancy_change: jnp.ndarray
    mask: jnp.ndarray            # 有效步掩码（游戏结束前）

# ============================================================================
# 自玩
# ============================================================================


@partial(jax.pmap, static_broadcasted_argnums=(2,))
def selfplay(params, rng_key, batch_size):
    """
    高性能自玩算子：
    - lax.scan 消除 Python 循环开销
    - Gumbel-Top-k：探索在搜索内完成，并辅以全程根噪声 + 温度退火采样
    """
    def step_fn(state, key):
        key_mcts, key_sample, key_reset = jax.random.split(key, 3)
        obs = jax.vmap(env.observe)(state)
        logits, value = forward(params, obs, return_aux=False)
        # 黑方时 obs 已翻转，logits 旋转到绝对坐标系，供 MCTS 与 legal_action_mask 使用
        logits = jnp.where(state.current_player[:, None] == 0, logits, logits[:, _ROTATED_IDX])
        logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)
        
        # MCTS 搜索
        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)
        policy_output = mctx.gumbel_muzero_policy(
            params=params, rng_key=key_mcts, root=root, recurrent_fn=recurrent_fn,
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

        def _sample_action(w, legal_mask, temperature, sample_key):
            # 在 log 空间做温度缩放，避免小概率动作下溢。
            w_masked = _masked_normalize(w, legal_mask)
            log_w = jnp.log(w_masked + 1e-10) / temperature
            log_w = jnp.where(legal_mask, log_w, -jnp.inf)
            log_w = log_w - jnp.max(log_w)
            probs = jnp.exp(log_w)
            probs = _masked_normalize(probs, legal_mask)
            return jax.random.choice(sample_key, ACTION_SPACE_SIZE, p=probs)

        # 全程做温度退火采样；达到下限后维持小温度，避免中残局完全贪心。
        sample_keys = jax.random.split(key_sample, batch_size)
        sample_temperatures = _opening_temperature(state.step_count)
        sampled_action = jax.vmap(_sample_action)(
            action_weights, state.legal_action_mask, sample_temperatures, sample_keys
        )
        action = jnp.where(only_one_move, action_idx, sampled_action)
        
        actor = state.current_player
        
        # 执行动作
        next_state = jax.vmap(env.step)(state, action)
        
        # 恢复统一视角：策略目标与 obs 保持同一归一化视角。
        normalized_action_weights = jnp.where(
            state.current_player[:, None] == 0,
            action_weights,
            action_weights[:, _ROTATED_IDX],
        )
        policy_prob, policy_idx = jax.lax.top_k(normalized_action_weights, config.top_k)
        policy_denom = jnp.maximum(jnp.sum(policy_prob, axis=-1, keepdims=True), 1e-10)
        policy_prob = (policy_prob / policy_denom).astype(jnp.float32)
        
        # 根节点 visit 分布熵：-sum(p*log(p))，高熵=探索充分，低熵=决策集中
        p = _masked_normalize(action_weights, state.legal_action_mask)
        root_visit_entropy = -jnp.sum(p * jnp.log(p + 1e-10), axis=-1)
        
        data = SelfplayOutput(
            obs=obs,
            policy_idx=policy_idx.astype(jnp.int32),
            policy_prob=policy_prob,
            reward=next_state.rewards[jnp.arange(batch_size), actor],
            terminated=next_state.terminated,
            discount=jnp.where(next_state.terminated, 0.0, -1.0),
            winner=next_state.winner, draw_reason=next_state.draw_reason,
            root_value=root_value,
            root_visit_entropy=root_visit_entropy,
            step_count=state.step_count,
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

def _current_material(obs: jnp.ndarray) -> jnp.ndarray:
    current = obs[..., :14, :, :].astype(jnp.float32)
    piece_counts = jnp.sum(current, axis=(-1, -2))
    own = jnp.sum(piece_counts[..., :7] * _MATERIAL_VALUES[:7], axis=-1)
    opp = jnp.sum(piece_counts[..., 7:] * _MATERIAL_VALUES[7:], axis=-1)
    return (own - opp) / 45.0


def _current_occupancy(obs: jnp.ndarray) -> jnp.ndarray:
    return (jnp.sum(obs[..., :14, :, :], axis=-3) > 0).reshape(*obs.shape[:-3], BOARD_HEIGHT * BOARD_WIDTH)


@jax.pmap
def compute_targets(data: SelfplayOutput):
    """计算训练目标（与 mctx / Gumbel MuZero 推荐一致）。

    - policy_target：MCTS 的 action_weights（策略蒸馏），以 top-k 稀疏格式存储，减少回放搬运。
    - value_tgt：对 **MCTS 根标量 root_value** 做 TD(λ) 备份；损失为 MSE(value, value_tgt)，
      value 由 ValueHead 的 value_logits 经 softmax 得 W−L。
    """
    max_steps, batch_size = data.reward.shape[0], data.reward.shape[1]
    lam = config.td_lambda
    
    # 掩码：游戏结束前的步骤参与训练
    value_mask = jnp.cumsum(data.terminated[::-1, :], axis=0)[::-1, :] >= 1
    
    # s_{t+1} 的 MCTS 根价值，用于 bootstrap（与 Gumbel AlphaZero：价值信搜索根值一致）
    value_next = jnp.concatenate(
        [data.root_value[1:], jnp.zeros((1, batch_size), dtype=data.root_value.dtype)], axis=0
    )
    
    def scan_fn(carry_value, inputs):
        """从后向前：标量 TD(λ)（零和折扣 discount=-1，终局 discount=0）。"""
        reward_t, discount_t, value_next_t = inputs
        value_blended = (1.0 - lam) * value_next_t + lam * carry_value
        g_value = reward_t + discount_t * value_blended
        return g_value, g_value
    
    _, value_tgt_rev = jax.lax.scan(
        scan_fn,
        jnp.zeros((batch_size,), dtype=data.root_value.dtype),
        (
            data.reward[::-1],
            data.discount[::-1],
            value_next[::-1],
        ),
    )
    value_tgt = value_tgt_rev[::-1]
    first_term = (jnp.cumsum(data.terminated, axis=0) == 1) & data.terminated
    fallback_term = jnp.full((batch_size,), max_steps - 1, dtype=jnp.int32)
    term_idx = jnp.where(
        jnp.any(first_term, axis=0),
        jnp.argmax(first_term.astype(jnp.int32), axis=0),
        fallback_term,
    )
    step_idx = jnp.arange(max_steps, dtype=jnp.int32)[:, None]
    remaining_ply = jnp.maximum(term_idx[None, :] - step_idx + 1, 1)
    remaining_ply_bucket = jnp.sum(
        remaining_ply[..., None] > _REMAINING_PLY_BUCKETS[None, None, :],
        axis=-1,
    ).astype(jnp.int32)

    future_idx = jnp.minimum(step_idx + config.aux_future_horizon, term_idx[None, :])
    obs_by_batch = jnp.swapaxes(data.obs, 0, 1)
    idx_by_batch = jnp.swapaxes(future_idx, 0, 1)
    future_obs = jax.vmap(lambda obs_b, idx_b: obs_b[idx_b])(obs_by_batch, idx_by_batch)
    future_obs = jnp.swapaxes(future_obs, 0, 1)

    current_material = _current_material(data.obs)
    future_material = _current_material(future_obs)
    future_material_delta = jnp.clip(future_material - current_material, -1.0, 1.0)

    current_occupancy = _current_occupancy(data.obs)
    future_occupancy = _current_occupancy(future_obs)
    future_occupancy_change = (current_occupancy != future_occupancy).astype(jnp.float32)

    return Sample(
        obs=data.obs, 
        policy_idx=data.policy_idx.astype(jnp.int32),
        policy_prob=data.policy_prob.astype(jnp.float32),
        value_tgt=value_tgt,
        remaining_ply_bucket=remaining_ply_bucket,
        future_material_delta=future_material_delta,
        future_occupancy_change=future_occupancy_change,
        mask=value_mask,
    )

# ============================================================================
# 自对弈统计（GPU 端计算，避免传输完整 data 到 CPU）
# ============================================================================

@jax.pmap
def compute_selfplay_stats(data: SelfplayOutput):
    """在 GPU 上计算自对弈统计标量，仅传回 ~100 字节而非 ~3GB 原始数据
    
    返回 (stats[12], avg_length, entropy_all, entropy_opening, entropy_mid) 均为 float32
    stats: [总局数, 红胜, 黑胜, 和棋, ...]
    entropy_opening: 前 temperature_steps 半步的熵均值，用于监控开局坍缩
    entropy_mid: 后续局面的熵均值
    """
    term = data.terminated       # (max_steps, batch)
    winner = data.winner
    reasons = data.draw_reason
    max_steps = term.shape[0]
    temp_steps = config.selfplay_temperature_steps

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

    # 分段熵：开局（前 temp_steps 半步）vs 中残局，便于监控开局坍缩
    sc = data.step_count  # (max_steps, batch)
    ent = data.root_visit_entropy
    opening_mask = sc < temp_steps
    mid_mask = sc >= temp_steps
    n_open = jnp.maximum(jnp.sum(opening_mask.astype(jnp.float32)), 1.0)
    n_mid = jnp.maximum(jnp.sum(mid_mask.astype(jnp.float32)), 1.0)
    entropy_all = jnp.mean(ent)
    entropy_opening = jnp.sum(jnp.where(opening_mask, ent, 0.0)) / n_open
    entropy_mid = jnp.sum(jnp.where(mid_mask, ent, 0.0)) / n_mid

    return stats, avg_length, entropy_all, entropy_opening, entropy_mid

# ============================================================================
# 训练与评估
# ============================================================================

def loss_fn(params, samples: Sample, rng_key):
    """损失函数
    
    - 策略损失：稀疏交叉熵，拟合 mctx 的 top-k action_weights。
    - 价值损失：MSE(value, value_tgt)；value 由 value_logits 导出（p_W − p_L）。
    """
    obs = samples.obs.astype(jnp.float32)
    policy_idx = samples.policy_idx.astype(jnp.int32)
    policy_prob = samples.policy_prob.astype(jnp.float32)

    # 随机左右镜像增强（默认 30% 概率）
    rng_mirror, rng_dropout = jax.random.split(rng_key, 2)
    do_mirror = jax.random.bernoulli(rng_mirror, config.mirror_augmentation_prob)

    def _mirror_occupancy(occupancy):
        board = occupancy.reshape((occupancy.shape[0], BOARD_HEIGHT, BOARD_WIDTH))
        return jnp.flip(board, axis=2).reshape((occupancy.shape[0], BOARD_HEIGHT * BOARD_WIDTH))

    def _apply_mirror(args):
        obs_in, idx_in, prob_in, occ_in = args
        return jax.vmap(mirror_observation)(obs_in), mirror_action(idx_in), prob_in, _mirror_occupancy(occ_in)

    obs, policy_idx, policy_prob, future_occupancy_change = jax.lax.cond(
        do_mirror,
        _apply_mirror,
        lambda args: args,
        (obs, policy_idx, policy_prob, samples.future_occupancy_change.astype(jnp.float32)),
    )

    logits, value, _, remaining_logits, material_delta_pred, occupancy_logits = forward(
        params, obs, is_training=True, rng_key=rng_dropout
    )
    
    logits = logits.astype(jnp.float32)
    value = value.astype(jnp.float32)
    value_tgt = samples.value_tgt.astype(jnp.float32)
    remaining_ply_bucket = samples.remaining_ply_bucket.astype(jnp.int32)
    future_material_delta = samples.future_material_delta.astype(jnp.float32)
    future_occupancy_change = future_occupancy_change.astype(jnp.float32)

    # 稀疏策略损失：仅对非零目标动作做 gather，避免在回放中搬运完整 2086 维分布
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    sparse_log_probs = jnp.take_along_axis(log_probs, policy_idx, axis=-1)
    policy_ce = -jnp.sum(policy_prob * sparse_log_probs, axis=-1)
    policy_loss = jnp.sum(policy_ce * samples.mask) / jnp.maximum(jnp.sum(samples.mask), 1.0)

    # 价值损失：MSE，对 MCTS 根价值 TD 目标回归标量 value（经 value_logits 可导）
    err = value - value_tgt
    value_loss = jnp.sum(err * err * samples.mask) / jnp.maximum(jnp.sum(samples.mask), 1.0)

    mask = samples.mask.astype(jnp.float32)
    mask_denom = jnp.maximum(jnp.sum(mask), 1.0)
    remaining_ce = optax.softmax_cross_entropy_with_integer_labels(
        remaining_logits.astype(jnp.float32),
        remaining_ply_bucket,
    )
    remaining_loss = jnp.sum(remaining_ce * mask) / mask_denom

    material_err = material_delta_pred.astype(jnp.float32) - future_material_delta
    material_loss = jnp.sum(optax.huber_loss(material_err, delta=0.25) * mask) / mask_denom

    occupancy_bce = optax.sigmoid_binary_cross_entropy(
        occupancy_logits.astype(jnp.float32),
        future_occupancy_change,
    )
    occupancy_loss = jnp.sum(occupancy_bce * mask[:, None]) / jnp.maximum(
        jnp.sum(mask) * occupancy_bce.shape[-1],
        1.0,
    )

    aux_loss = (
        config.aux_remaining_loss_weight * remaining_loss
        + config.aux_material_loss_weight * material_loss
        + config.aux_occupancy_loss_weight * occupancy_loss
    )
    total_loss = policy_loss + config.value_loss_weight * value_loss + aux_loss
    return total_loss, (policy_loss, value_loss, remaining_loss, material_loss, occupancy_loss)


def _run_best_checkpoint_eval(
    current_params,
    iteration: int,
    rng_key,
    ckpt_manager: ocp.CheckpointManager,
    params_template: dict,
    opt_state_template: dict,
    iteration_elos: dict,
):
    """选择历史最强 checkpoint 作为单一对手，执行红黑交换评估。"""
    kept_steps = _get_kept_steps(config.ckpt_dir)
    available_iters = sorted(k for k in kept_steps if k < iteration)
    if not available_iters:
        return None, rng_key

    rated_iters = [k for k in available_iters if k in iteration_elos]
    eval_candidates = []
    if rated_iters:
        eval_candidates.extend(
            (k, "best_elo")
            for k in sorted(rated_iters, key=lambda step: (iteration_elos[step], step), reverse=True)
        )
    eval_candidates.extend(
        (k, "bootstrap_unrated_ckpt")
        for k in sorted((k for k in available_iters if k not in iteration_elos), reverse=True)
    )

    opponent_params = None
    ref_iter = None
    ref_reason = None
    failed_refs = []
    for candidate_iter, candidate_reason in eval_candidates:
        try:
            opponent_params = replicate_to_devices(
                _load_params_from_checkpoint(ckpt_manager, candidate_iter, params_template, opt_state_template)
            )
            ref_iter = candidate_iter
            ref_reason = candidate_reason
            break
        except Exception as e:
            failed_refs.append(candidate_iter)
            iteration_elos.pop(candidate_iter, None)
            logger.warning(
                "[Eval] 跳过无法恢复的 checkpoint step=%s: %s",
                candidate_iter,
                e,
            )

    if opponent_params is None:
        logger.warning("[Eval] 没有可用的历史 checkpoint，跳过本轮评估；失败 steps=%s", failed_refs)
        return None, rng_key

    rng_key, sk5, sk6 = jax.random.split(rng_key, 3)
    batch_per_eval = config.eval_games
    states_r, states_b = _build_eval_initial_states(batch_per_eval)
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

    winners_r = evaluate(current_params, opponent_params, states_r, eval_keys_r)
    winners_b = evaluate(opponent_params, current_params, states_b, eval_keys_b)

    winners_r_np = np.array(jax.device_get(winners_r))
    winners_b_np = np.array(jax.device_get(winners_b))
    wins_red = int((winners_r_np == 0).sum())
    draws_red = int((winners_r_np == -1).sum())
    losses_red = int((winners_r_np == 1).sum())
    wins_black = int((winners_b_np == 1).sum())
    draws_black = int((winners_b_np == -1).sum())
    losses_black = int((winners_b_np == 0).sum())

    wins = wins_red + wins_black
    draws = draws_red + draws_black
    losses = losses_red + losses_black
    total_games = int(config.eval_games * 2)
    score = (wins + 0.5 * draws) / total_games
    decisive_games = wins + losses
    decisive_win_rate = wins / decisive_games if decisive_games > 0 else float("nan")

    ref_elo = float(iteration_elos.get(ref_iter, 1500.0))
    elo_diff = 400.0 * np.log10(score / (1.0 - score)) if 0 < score < 1 else (400 if score >= 1 else -400)

    metrics = {
        "ref_iter": ref_iter,
        "ref_reason": ref_reason,
        "wins_red": wins_red,
        "draws_red": draws_red,
        "losses_red": losses_red,
        "wins_black": wins_black,
        "draws_black": draws_black,
        "losses_black": losses_black,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "total_games": total_games,
        "score": score,
        "decisive_win_rate": decisive_win_rate,
        "ref_elo": ref_elo,
        "elo": ref_elo + elo_diff,
    }
    return metrics, rng_key

@lru_cache(maxsize=None)
def _load_eval_fen_pool(eval_fens_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """加载固定评估 FEN 集，返回 (boards, players)。"""
    abs_path = os.path.abspath(eval_fens_path)
    fen_items = load_fens_from_file(abs_path)
    if not fen_items:
        raise ValueError(f"评估 FEN 文件为空: {abs_path}")

    boards = np.stack([board for board, _player in fen_items]).astype(np.int8)
    players = np.array([player for _board, player in fen_items], dtype=np.int32)
    logger.info("[Eval FEN] 已加载 %s 个评估局面: %s", len(fen_items), abs_path)
    return boards, players


def _build_eval_initial_states(batch_size: int):
    """基于固定评估 FEN 构建红黑交换评估局面。"""
    batch_per_device = batch_size // num_devices
    if batch_per_device * num_devices != batch_size:
        raise ValueError(f"batch_size {batch_size} 必须整除 num_devices {num_devices}")

    boards_np, players_np = _load_eval_fen_pool(config.eval_fens_path)
    if batch_size > len(boards_np):
        logger.info(
            "[Eval FEN] 评估对局数 %s 超过 FEN 数 %s，按固定顺序循环复用",
            batch_size,
            len(boards_np),
        )
    sample_idx = np.arange(batch_size) % len(boards_np)
    boards = jnp.asarray(boards_np[sample_idx], dtype=jnp.int8)
    players = jnp.asarray(players_np[sample_idx], dtype=jnp.int32)

    states_red_flat = jax.vmap(lambda b, p: env.init_from_board(b, p))(boards, players)
    swapped_boards = jax.vmap(mirror_board_swap_colors)(boards)
    swapped_players = jnp.int32(1) - players
    states_black_flat = jax.vmap(lambda b, p: env.init_from_board(b, p))(swapped_boards, swapped_players)

    def _shard(x):
        return x.reshape(num_devices, batch_per_device, *x.shape[1:])

    return jax.tree.map(_shard, states_red_flat), jax.tree.map(_shard, states_black_flat)


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
    
    仅存储有效样本：obs, 稀疏策略目标(与 obs 同视角), value_tgt(MCTS TD 标量), mask
    """
    
    def __init__(self, max_size: int, obs_shape: tuple, policy_target_size: int):
        self.max_size = max_size
        self.obs = np.zeros((max_size, *obs_shape), dtype=np.uint8)
        self.policy_idx = np.zeros((max_size, policy_target_size), dtype=np.uint16)
        self.policy_prob = np.zeros((max_size, policy_target_size), dtype=np.float16)
        self.value_tgt = np.zeros((max_size,), dtype=np.float32)
        self.remaining_ply_bucket = np.zeros((max_size,), dtype=np.uint8)
        self.future_material_delta = np.zeros((max_size,), dtype=np.float16)
        self.future_occupancy_change = np.zeros((max_size, BOARD_HEIGHT * BOARD_WIDTH), dtype=np.bool_)
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
        """存入样本（JAX -> NumPy），并在入库前裁掉终局后的无效步。"""
        samples_np = jax.device_get(samples)
        
        obs_flat = samples_np.obs.reshape(-1, *samples_np.obs.shape[3:]).astype(np.uint8)
        policy_idx_flat = samples_np.policy_idx.reshape(-1, samples_np.policy_idx.shape[-1]).astype(np.uint16)
        policy_prob_flat = samples_np.policy_prob.reshape(-1, samples_np.policy_prob.shape[-1]).astype(np.float16)
        value_flat = samples_np.value_tgt.reshape(-1)
        remaining_flat = samples_np.remaining_ply_bucket.reshape(-1).astype(np.uint8)
        material_flat = samples_np.future_material_delta.reshape(-1).astype(np.float16)
        occupancy_flat = samples_np.future_occupancy_change.reshape(
            -1, samples_np.future_occupancy_change.shape[-1]
        ).astype(np.bool_)
        mask_flat = samples_np.mask.reshape(-1)

        valid_mask = mask_flat.astype(np.bool_)
        if not np.any(valid_mask):
            return

        obs_flat = obs_flat[valid_mask]
        policy_idx_flat = policy_idx_flat[valid_mask]
        policy_prob_flat = policy_prob_flat[valid_mask]
        value_flat = value_flat[valid_mask]
        remaining_flat = remaining_flat[valid_mask]
        material_flat = material_flat[valid_mask]
        occupancy_flat = occupancy_flat[valid_mask]
        
        n_new = obs_flat.shape[0]
        indices = (np.arange(n_new) + self.ptr) % self.max_size
        
        self.obs[indices] = obs_flat
        self.policy_idx[indices] = policy_idx_flat
        self.policy_prob[indices] = policy_prob_flat
        self.value_tgt[indices] = value_flat
        self.remaining_ply_bucket[indices] = remaining_flat
        self.future_material_delta[indices] = material_flat
        self.future_occupancy_change[indices] = occupancy_flat
        self.mask[indices] = True
        self.ptr = (self.ptr + n_new) % self.max_size
        self.size = min(self.size + n_new, self.max_size)
        self.total_added += n_new
    
    def sample(self, batch_size: int, rng_key):
        """均匀采样，转回 JAX 数组
        
        obs 保持 uint8，策略目标保持稀疏格式，以减小 CPU→GPU 带宽。
        """
        if self.size <= 0:
            raise ValueError("ReplayBuffer 为空，无法采样")

        rng = self._build_rng(rng_key)
        valid_size = self.size
        idx = rng.integers(valid_size, size=batch_size)
        
        obs_batch = self.obs[idx]
        policy_idx_batch = self.policy_idx[idx]
        policy_prob_batch = self.policy_prob[idx]
        value_batch = self.value_tgt[idx]
        remaining_batch = self.remaining_ply_bucket[idx]
        material_batch = self.future_material_delta[idx]
        occupancy_batch = self.future_occupancy_change[idx]
        
        sample = Sample(
            obs=jnp.asarray(obs_batch, dtype=jnp.uint8),
            policy_idx=jnp.asarray(policy_idx_batch, dtype=jnp.int32),
            policy_prob=jnp.asarray(policy_prob_batch, dtype=jnp.float32),
            value_tgt=jnp.asarray(value_batch),
            remaining_ply_bucket=jnp.asarray(remaining_batch, dtype=jnp.int32),
            future_material_delta=jnp.asarray(material_batch, dtype=jnp.float32),
            future_occupancy_change=jnp.asarray(occupancy_batch, dtype=jnp.float32),
            mask=jnp.ones((batch_size,), dtype=jnp.bool_),
        )
        return sample, idx.astype(np.int64)

    def sample_sharded(self, batch_size: int, rng_key):
        """均匀采样后直接按设备分片上传，减少中间大数组搬运。"""
        if batch_size <= 0 or batch_size % num_devices != 0:
            raise ValueError(
                f"batch_size={batch_size} 必须是正数且能整除 num_devices({num_devices})"
            )
        if self.size <= 0:
            raise ValueError("ReplayBuffer 为空，无法采样")

        rng = self._build_rng(rng_key)
        idx = rng.integers(self.size, size=batch_size)
        per_device = batch_size // num_devices

        obs_batch = self.obs[idx].reshape((num_devices, per_device, *self.obs.shape[1:]))
        policy_idx_batch = self.policy_idx[idx].reshape((num_devices, per_device, self.policy_idx.shape[1]))
        policy_prob_batch = self.policy_prob[idx].reshape((num_devices, per_device, self.policy_prob.shape[1]))
        value_batch = self.value_tgt[idx].reshape((num_devices, per_device))
        remaining_batch = self.remaining_ply_bucket[idx].reshape((num_devices, per_device))
        material_batch = self.future_material_delta[idx].reshape((num_devices, per_device))
        occupancy_batch = self.future_occupancy_change[idx].reshape(
            (num_devices, per_device, BOARD_HEIGHT * BOARD_WIDTH)
        )
        mask_batch = np.ones((num_devices, per_device), dtype=np.bool_)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            sample = Sample(
                obs=jax.device_put_sharded([obs_batch[i] for i in range(num_devices)], devices),
                policy_idx=jax.device_put_sharded([policy_idx_batch[i] for i in range(num_devices)], devices),
                policy_prob=jax.device_put_sharded([policy_prob_batch[i] for i in range(num_devices)], devices),
                value_tgt=jax.device_put_sharded([value_batch[i] for i in range(num_devices)], devices),
                remaining_ply_bucket=jax.device_put_sharded([remaining_batch[i] for i in range(num_devices)], devices),
                future_material_delta=jax.device_put_sharded([material_batch[i] for i in range(num_devices)], devices),
                future_occupancy_change=jax.device_put_sharded([occupancy_batch[i] for i in range(num_devices)], devices),
                mask=jax.device_put_sharded([mask_batch[i] for i in range(num_devices)], devices),
            )
        return sample, idx.astype(np.int64)

    def cleanup(self):
        pass
    
    def stats(self):
        return {"size": self.size, "total_added": self.total_added, "ptr": self.ptr}

    def state_dict(self):
        return {
            "obs": self.obs,
            "policy_idx": self.policy_idx,
            "policy_prob": self.policy_prob,
            "value_tgt": self.value_tgt,
            "remaining_ply_bucket": self.remaining_ply_bucket,
            "future_material_delta": self.future_material_delta,
            "future_occupancy_change": self.future_occupancy_change,
            "mask": self.mask,
            "ptr": self.ptr, "size": self.size, "total_added": self.total_added
        }

    def load_state_dict(self, state):
        loaded_obs = np.array(state["obs"])
        loaded_value = np.array(state["value_tgt"])
        loaded_remaining = np.array(state["remaining_ply_bucket"], dtype=np.uint8)
        loaded_material = np.array(state["future_material_delta"], dtype=np.float16)
        loaded_occupancy = np.array(state["future_occupancy_change"], dtype=np.bool_)
        loaded_size = min(int(state["size"]), self.max_size, loaded_obs.shape[0])
        loaded_ptr = int(state["ptr"]) % max(loaded_size, 1)

        if "policy_idx" in state and "policy_prob" in state:
            loaded_policy_idx = np.array(state["policy_idx"], dtype=np.uint16)
            loaded_policy_prob = np.array(state["policy_prob"], dtype=np.float16)
        elif "policy_tgt" in state:
            loaded_policy_dense = np.array(state["policy_tgt"], dtype=np.float32)
            k = self.policy_idx.shape[1]
            part_idx = np.argpartition(loaded_policy_dense, -k, axis=-1)[:, -k:]
            part_prob = np.take_along_axis(loaded_policy_dense, part_idx, axis=-1)
            order = np.argsort(-part_prob, axis=-1)
            loaded_policy_idx = np.take_along_axis(part_idx, order, axis=-1).astype(np.uint16)
            loaded_policy_prob = np.take_along_axis(part_prob, order, axis=-1)
            denom = np.maximum(loaded_policy_prob.sum(axis=-1, keepdims=True), 1e-8)
            loaded_policy_prob = (loaded_policy_prob / denom).astype(np.float16)
        else:
            raise ValueError("checkpoint replay buffer 缺少 policy_idx/policy_prob 或旧版 policy_tgt")

        if loaded_obs.shape[0] < loaded_size:
            raise ValueError("checkpoint replay buffer 数据不完整")

        if loaded_size > 0:
            if state["mask"] is not None:
                loaded_mask = np.array(state["mask"][:loaded_size], dtype=np.bool_)
                if not np.all(loaded_mask):
                    raise ValueError("checkpoint replay buffer 含无效样本；当前版本不再兼容旧格式")

            if loaded_obs.shape[0] > self.max_size:
                start = (loaded_ptr - loaded_size) % loaded_obs.shape[0]
                idx = (np.arange(loaded_size) + start) % loaded_obs.shape[0]
                loaded_obs = loaded_obs[idx]
                loaded_policy_idx = loaded_policy_idx[idx]
                loaded_policy_prob = loaded_policy_prob[idx]
                loaded_value = loaded_value[idx]
                loaded_remaining = loaded_remaining[idx]
                loaded_material = loaded_material[idx]
                loaded_occupancy = loaded_occupancy[idx]
            else:
                loaded_obs = loaded_obs[:loaded_size]
                loaded_policy_idx = loaded_policy_idx[:loaded_size]
                loaded_policy_prob = loaded_policy_prob[:loaded_size]
                loaded_value = loaded_value[:loaded_size]
                loaded_remaining = loaded_remaining[:loaded_size]
                loaded_material = loaded_material[:loaded_size]
                loaded_occupancy = loaded_occupancy[:loaded_size]

            self.obs[:loaded_size] = loaded_obs
            self.policy_idx[:loaded_size] = loaded_policy_idx
            self.policy_prob[:loaded_size] = loaded_policy_prob
            self.value_tgt[:loaded_size] = loaded_value
            self.remaining_ply_bucket[:loaded_size] = loaded_remaining
            self.future_material_delta[:loaded_size] = loaded_material
            self.future_occupancy_change[:loaded_size] = loaded_occupancy

        if loaded_size < self.max_size:
            self.mask[loaded_size:] = False
        self.mask[:loaded_size] = True
        self.size = loaded_size
        self.ptr = loaded_size % self.max_size
        self.total_added = max(int(state["total_added"]), loaded_size)

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


def _get_kept_steps(ckpt_dir: str) -> set:
    """从磁盘直接扫描 checkpoint 目录，获取实际存在的 step 集合。
    避免 orbax all_steps(reload=True) 在异步删除/元数据缺失时报 FileNotFoundError。"""
    ckpt_root = os.path.abspath(ckpt_dir)
    if not os.path.isdir(ckpt_root):
        return set()
    steps = set()
    for name in os.listdir(ckpt_root):
        if name.isdigit():
            path = os.path.join(ckpt_root, name)
            if os.path.isdir(path):
                steps.add(int(name))
    return steps


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
    ckpt_manager.wait_until_finished()
    
    # 单文件 metadata.json：iteration_elos、total_opt_steps（仅保留 orbax 保留的 step 对应的 elo）
    kept_steps = _get_kept_steps(config.ckpt_dir)
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
    
    Orbax 需要完整训练状态树；这里按完整结构恢复，但只返回 params。
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


def _resolve_checkpoint_source(spec: str, default_ckpt_dir: str) -> Tuple[str, int]:
    """将 step 编号或 checkpoint 路径解析为 (ckpt_dir, step)。"""
    if spec is None:
        raise ValueError("checkpoint 来源不能为空")
    spec = spec.strip()
    if not spec:
        raise ValueError("checkpoint 来源不能为空字符串")

    if spec.isdigit():
        return os.path.abspath(default_ckpt_dir), int(spec)

    abs_path = os.path.abspath(spec)
    base = os.path.basename(abs_path)
    if not base.isdigit():
        raise ValueError(
            f"无法解析 checkpoint: {spec}。请使用 step 编号，或形如 checkpoints/100 的目录路径。"
        )
    if not os.path.isdir(abs_path):
        raise FileNotFoundError(f"checkpoint 目录不存在: {abs_path}")
    return os.path.dirname(abs_path), int(base)


def _load_init_params_from_source(
    checkpoint_spec: str,
    params_template: dict,
    opt_state_template: dict,
) -> Tuple[dict, str]:
    """从指定 checkpoint 导入模型参数，用作训练初始权重。"""
    ckpt_dir, step = _resolve_checkpoint_source(checkpoint_spec, config.ckpt_dir)
    ckpt_manager = create_checkpoint_manager(ckpt_dir)
    params = _load_params_from_checkpoint(ckpt_manager, step, params_template, opt_state_template)
    source_desc = os.path.join(os.path.abspath(ckpt_dir), str(step))
    return params, source_desc


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
    if CLI_OVERRIDES:
        logger.info("[CLI] 覆盖参数: %s", CLI_OVERRIDES)
    
    # 创建必要目录
    os.makedirs(config.ckpt_dir, exist_ok=True)
    
    # 初始化经验回放缓冲区 (NumPy 环形缓冲区)
    replay_buffer = ReplayBuffer(
        max_size=config.replay_buffer_size,
        obs_shape=(NUM_OBSERVATION_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH),
        policy_target_size=config.top_k,
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
    logger.info(
        "[TD(λ)] 固定 λ=%s | value_loss_weight=%.2f",
        config.td_lambda,
        config.value_loss_weight,
    )
    logger.info(
        "[Optimizer] AdamW(weight_decay=%.1e, train_batch=%s)",
        config.weight_decay,
        config.training_batch_size,
    )
    optimizer = optax.adamw(lr_schedule, weight_decay=config.weight_decay)
    opt_state_template = optimizer.init(params_template)
    
    # === 创建 Checkpoint Manager 并尝试恢复 ===
    ckpt_manager = create_checkpoint_manager(config.ckpt_dir)
    restored = restore_checkpoint(ckpt_manager, params_template, opt_state_template)
    
    if restored is not None:
        params, opt_state, iteration, frames, rng_key, iteration_elos, total_opt_steps = restored
        logger.info("[断点续训] 从 iteration=%s 继续训练", iteration)
        if CLI_ARGS.init_checkpoint:
            logger.info("[Init] 检测到已有训练断点，忽略 --init-checkpoint=%s", CLI_ARGS.init_checkpoint)
    else:
        if CLI_ARGS.init_checkpoint:
            try:
                params, init_source = _load_init_params_from_source(
                    CLI_ARGS.init_checkpoint, params_template, opt_state_template
                )
            except Exception as e:
                raise RuntimeError(
                    f"无法导入基础模型 {CLI_ARGS.init_checkpoint}，请确认 step 编号或 checkpoint 路径有效"
                ) from e
            logger.info("[Init] 已导入基础模型: %s；训练计数从 0 开始", init_source)
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
        grads, losses = jax.grad(loss_fn, has_aux=True)(params, samples, rng_key)
        grads = jax.lax.pmean(grads, 'i')
        updates, opt_state = optimizer.update(grads, opt_state, params)
        return (optax.apply_updates(params, updates), opt_state, *losses)

    # 根据关键超参自动生成日志子目录，便于 TensorBoard 对比实验
    run_name = (
        f"ch{config.num_channels}_b{config.num_blocks}"
        f"_sim{config.num_simulations}_k{config.top_k}"
        f"_lr{config.learning_rate:.0e}_bs{config.training_batch_size}"
        f"_td{config.td_lambda}_vw{config.value_loss_weight}"
        f"_sp{config.selfplay_batch_size}"
        f"_t{config.selfplay_temperature:.2f}-{config.selfplay_temperature_final:.2f}"
    )
    run_log_dir = os.path.join(config.log_dir, run_name)
    logger.info("[Log] TensorBoard 日志: %s", run_log_dir)
    writer = SummaryWriter(run_log_dir)
    start_time_total = time.time()
    
    logger.info("开始训练")
    
    # 恢复后若轮到评估（如崩溃在 ckpt 保存后、eval 前），补跑评估
    if restored is not None and iteration % config.eval_interval == 0 and iteration > 0:
        logger.info("[断点续训] 补跑 iteration=%s 的评估...", iteration)
        try:
            eval_metrics, rng_key = _run_best_checkpoint_eval(
                params, iteration, rng_key, ckpt_manager, params_template, opt_state_template, iteration_elos
            )
        except Exception as e:
            logger.exception("[评估错误] 补跑评估失败: %s", e)
            raise RuntimeError("断点续训补跑评估异常") from e
        if eval_metrics is not None:
            iteration_elos[iteration] = eval_metrics["elo"]
            decisive_text = (
                f'{eval_metrics["decisive_win_rate"]:.2%}'
                if np.isfinite(eval_metrics["decisive_win_rate"])
                else "N/A"
            )
            logger.info(
                "评估 vs Iter %s (%s): W/D/L %s/%s/%s | 得分率 %.2f%% | RefELO %.0f | ELO %.0f | 决胜胜率 %s",
                eval_metrics["ref_iter"],
                eval_metrics["ref_reason"],
                eval_metrics["wins"],
                eval_metrics["draws"],
                eval_metrics["losses"],
                eval_metrics["score"] * 100.0,
                eval_metrics["ref_elo"],
                eval_metrics["elo"],
                decisive_text,
            )
            writer.add_scalar("eval/elo", eval_metrics["elo"], iteration)
            writer.add_scalar("eval/win_rate", eval_metrics["wins"] / eval_metrics["total_games"], iteration)
            writer.add_scalar("eval/draw_rate", eval_metrics["draws"] / eval_metrics["total_games"], iteration)
            writer.add_scalar("eval/loss_rate", eval_metrics["losses"] / eval_metrics["total_games"], iteration)
            writer.add_scalar("eval/score", eval_metrics["score"], iteration)
            if np.isfinite(eval_metrics["decisive_win_rate"]):
                writer.add_scalar("eval/decisive_win_rate", eval_metrics["decisive_win_rate"], iteration)
    
    logger.info(
        "[Selfplay] batch_size=%s, gumbel_scale=%s, reuse=%s, temp_decay_steps=%s, temp=%.2f->%.2f(hold)",
        config.selfplay_batch_size,
        config.selfplay_gumbel_scale,
        config.sample_reuse_times,
        config.selfplay_temperature_steps,
        config.selfplay_temperature,
        config.selfplay_temperature_final,
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
        stats_dev, avg_len_dev, ent_all_dev, ent_open_dev, ent_mid_dev = gpu_stats
        stats_np = np.array(jax.device_get(stats_dev))
        stats = stats_np.sum(axis=0).astype(np.int64)   # 跨设备求和
        avg_length = float(jax.device_get(avg_len_dev).mean())
        root_visit_entropy = float(jax.device_get(ent_all_dev).mean())
        entropy_opening = float(jax.device_get(ent_open_dev).mean())
        entropy_mid = float(jax.device_get(ent_mid_dev).mean())

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
            """CPU 采样 + 直接按设备分片上传（双缓冲预读取）"""
            batch, _ = replay_buffer.sample_sharded(config.training_batch_size, sample_keys[key_idx])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                keys = jax.device_put_sharded(
                    list(jax.random.split(sample_keys[key_idx], num_devices)), devices)
            return batch, keys

        ploss_acc, vloss_acc = None, None
        rem_loss_acc, mat_loss_acc, occ_loss_acc = None, None, None
        with ThreadPoolExecutor(max_workers=1) as prefetch_executor:
            next_future = (
                prefetch_executor.submit(_prefetch_batch, 1) if num_updates > 0 else None
            )

            for i in range(num_updates):
                batch, train_keys = next_future.result()
                if i + 1 < num_updates:
                    next_future = prefetch_executor.submit(_prefetch_batch, i + 2)
                else:
                    next_future = None

                try:
                    params, opt_state, ploss, vloss, rem_loss, mat_loss, occ_loss = train_step(
                        params, opt_state, batch, train_keys
                    )
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

                # 累积损失（保持在 GPU），只在最后同步
                if ploss_acc is None:
                    ploss_acc, vloss_acc = ploss, vloss
                    rem_loss_acc, mat_loss_acc, occ_loss_acc = rem_loss, mat_loss, occ_loss
                else:
                    ploss_acc = ploss_acc + ploss
                    vloss_acc = vloss_acc + vloss
                    rem_loss_acc = rem_loss_acc + rem_loss
                    mat_loss_acc = mat_loss_acc + mat_loss
                    occ_loss_acc = occ_loss_acc + occ_loss
        
        # 累计优化器步数
        total_opt_steps += num_updates
        
        # 一次性同步所有累积损失
        if num_updates > 0:
            policy_loss = float(ploss_acc.mean() / num_updates)
            value_loss = float(vloss_acc.mean() / num_updates)
            remaining_loss = float(rem_loss_acc.mean() / num_updates)
            material_loss = float(mat_loss_acc.mean() / num_updates)
            occupancy_loss = float(occ_loss_acc.mean() / num_updates)
        else:
            policy_loss = 0.0
            value_loss = 0.0
            remaining_loss = 0.0
            material_loss = 0.0
            occupancy_loss = 0.0
        
        # --- 清理已训练足够次数的样本 ---
        replay_buffer.cleanup()
        
        # --- 打印与日志 ---
        iter_time = time.time() - st
        fps = new_frames / max(iter_time, 1e-9)
        buf_stats = replay_buffer.stats()
        total_elapsed = time.time() - start_time_total
        logger.info(
            "iter=%3d | ploss=%.4f vloss=%.4f aux=%.3f/%.3f/%.3f | len=%4.1f fps=%4.0f "
            "buf=%dk train=%d | ent=%.3f(开%.3f/中%.3f) | 红%3d 黑%3d 和%3d | iter_t=%s total=%s",
            iteration,
            policy_loss,
            value_loss,
            remaining_loss,
            material_loss,
            occupancy_loss,
            avg_length,
            fps,
            buf_stats["size"] // 1000,
            num_updates,
            root_visit_entropy,
            entropy_opening,
            entropy_mid,
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
        writer.add_scalar("train/aux_remaining_loss", remaining_loss, iteration)
        writer.add_scalar("train/aux_material_loss", material_loss, iteration)
        writer.add_scalar("train/aux_occupancy_loss", occupancy_loss, iteration)
        writer.add_scalar("train/lr", base_lr, iteration)
        writer.add_scalar("stats/avg_game_length", avg_length, iteration)
        writer.add_scalar("stats/fps", fps, iteration)
        writer.add_scalar("stats/root_visit_entropy", root_visit_entropy, iteration)
        writer.add_scalar("stats/entropy_opening", entropy_opening, iteration)
        writer.add_scalar("stats/entropy_mid", entropy_mid, iteration)
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
            kept_steps = _get_kept_steps(config.ckpt_dir)
            iteration_elos = {k: v for k, v in iteration_elos.items() if k in kept_steps}
        
        if iteration % config.eval_interval == 0:
            try:
                eval_metrics, rng_key = _run_best_checkpoint_eval(
                    params, iteration, rng_key, ckpt_manager, params_template, opt_state_template, iteration_elos
                )
            except Exception as e:
                logger.error("[评估错误] 当前模型 dtype=%s, action_size=%s", config.network_dtype, ACTION_SPACE_SIZE)
                logger.error("[评估错误] params dtype: %s", jax.tree.leaves(params)[0].dtype)
                raise RuntimeError("评估阶段发生异常，请检查 GPU/Triton 与模型参数兼容性") from e

            if eval_metrics is not None:
                iteration_elos[iteration] = eval_metrics["elo"]
                win_rate = eval_metrics["wins"] / eval_metrics["total_games"]
                draw_rate = eval_metrics["draws"] / eval_metrics["total_games"]
                loss_rate = eval_metrics["losses"] / eval_metrics["total_games"]
                score = eval_metrics["score"]
                decisive_games = eval_metrics["wins"] + eval_metrics["losses"]
                decisive_win_rate = eval_metrics["decisive_win_rate"]
                decisive_text = f"{decisive_win_rate:.2%}" if decisive_games > 0 else "N/A"
                logger.info(
                    "评估 vs Iter %s (%s): W/D/L %s/%s/%s | 胜率 %.2f%% 和率 %.2f%% 负率 %.2f%% | "
                    "得分率 %.2f%% | 决胜局胜率 %s | RefELO %.0f | ELO %.0f",
                    eval_metrics["ref_iter"],
                    eval_metrics["ref_reason"],
                    eval_metrics["wins"],
                    eval_metrics["draws"],
                    eval_metrics["losses"],
                    win_rate * 100.0,
                    draw_rate * 100.0,
                    loss_rate * 100.0,
                    score * 100.0,
                    decisive_text,
                    eval_metrics["ref_elo"],
                    iteration_elos[iteration],
                )
                logger.info(
                    "  先手(当前红) W/D/L %s/%s/%s | 后手(当前黑) W/D/L %s/%s/%s",
                    eval_metrics["wins_red"],
                    eval_metrics["draws_red"],
                    eval_metrics["losses_red"],
                    eval_metrics["wins_black"],
                    eval_metrics["draws_black"],
                    eval_metrics["losses_black"],
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


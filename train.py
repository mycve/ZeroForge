#!/usr/bin/env python3
"""
ZeroForge - 中国象棋 Gumbel AlphaZero
现代化极简顶级架构：算力随机化 + 视角归一化 + 镜像增强 + 标准 ELO
"""

import os
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
from xiangqi.fen import load_fens_from_file, parse_fen
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
    
    # 网络架构（4分支GNN：无合法走法计算，推理快 → 搜索质量高）
    num_channels: int = 128
    num_blocks: int = 10
    # RTX 50 系上 BF16 通常具备接近 FP16 的速度，同时比 FP16 更稳
    network_dtype: str = "bfloat16"
    
    # 训练超参数
    learning_rate: float = 2e-4       # AdamW 起始 LR
    lr_warmup_steps: int = 1000       # 预热步数
    # LR 余弦退火：warmup 后平滑衰减到 min_ratio，无需手动调参
    lr_cosine_steps: int = 200000     # 余弦周期（opt steps）
    lr_min_ratio: float = 0.1        # 最低 LR = peak × 0.01 = 1e-5
    training_batch_size: int = 4096
    td_lambda: float = 0.75          # 0.99 近似蒙特卡洛（方差极高），0.85 平衡偏差/方差
    
    # 自对弈与搜索 (Gumbel 优势：低算力也能产生强信号)
    selfplay_batch_size: int = 2048
    num_simulations: int = 24
    top_k: int = 4                      # 根节点候选数
    
    # 经验回放配置
    replay_buffer_size: int = 1_000_000
    sample_reuse_times: int = 1
    
    # 损失权重
    value_loss_weight: float = 0.25
    weight_decay: float = 1e-4
    qtransform_value_scale: float = 0.10   # 放大 Q 值差异，提升高收益分支被选概率
    selfplay_gumbel_scale: float = 1.2
    eval_gumbel_scale: float = 0.10         # 评估关闭 Gumbel 噪声，结果更稳定
    
    # 探索策略：三段式温度（开局/中局/残局）
    temperature_phase1_steps: int = 10    # 0-10 半步（~5回合）: 开局全探索
    temperature_phase2_steps: int = 60    # 10-60 半步（~30回合）: 中局适度探索
    temperature_phase1: float = 1.2
    temperature_phase2: float = 0.8
    temperature_final: float = 0.01
    
    # 环境规则（符合象棋竞赛规则）
    max_steps: int = 300              # 总步数 400 步（200回合）判和
    max_no_capture_steps: int = 120   # 无吃子 120 步（60回合）判和，将军最多累计20回合
    repetition_threshold: int = 5     # 非将非捉重复局面 5 次判和
    # 长将/长捉规则已在 violation_rules.py 中实现
    
    # ELO 评估
    eval_interval: int = 20
    eval_games: int = 100
    past_model_offset: int = 20
    eval_fen_file: Optional[str] = "openings_generated.txt"  # 若指定，从该文件批量加载 FEN 作为起始局面，先后手轮换
    
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
        print(f"[Config] {field_name} 从 {value} 自动调整为 {aligned} (num_devices={num_devices})")
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
        print(f"[Batch] 训练 batch {total} 无法整除设备数 {num_devices}，自动裁剪为 {usable}")

    def _reshape(x):
        x = x[:usable]
        return x.reshape((num_devices, per_device) + x.shape[1:])

    return jax.tree.map(_reshape, batch_flat)


# 动态设备数下的批量自动对齐（避免 reshape/device_put_sharded 错误）
config.selfplay_batch_size = _align_to_device_multiple(config.selfplay_batch_size, "selfplay_batch_size")
config.training_batch_size = _align_to_device_multiple(config.training_batch_size, "training_batch_size")
config.eval_games = _align_to_device_multiple(config.eval_games, "eval_games")

if config.learning_rate <= 0:
    raise ValueError(f"learning_rate 必须 > 0，当前值: {config.learning_rate}")
if config.lr_warmup_steps < 0:
    raise ValueError(f"lr_warmup_steps 必须 >= 0，当前值: {config.lr_warmup_steps}")
# 预计算旋转索引，避免在 JIT 循环内重复计算
_ROTATED_IDX = rotate_action(jnp.arange(ACTION_SPACE_SIZE))

def replicate_to_devices(pytree):
    """将 pytree 复制到所有设备"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return jax.device_put_replicated(pytree, devices)



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

def forward(params, obs, is_training=False):
    """前向传播: 返回 (logits, value_scalar, wdl_logits)"""
    logits, value, wdl_logits = net.apply({'params': params}, obs, train=is_training)
    return logits, value, wdl_logits


def eval_forward(params, obs):
    """评估前向传播：固定 float32，规避部分 GPU 上 BF16 Triton 编译问题。"""
    logits, value, _wdl = eval_net.apply({'params': params}, obs, train=False)
    return logits, value


def recurrent_fn(params, rng_key, action, state):
    """MCTS 递归函数（瓶颈在 forward 推理 ~85%+，env.step 占比极小）"""
    prev_player = state.current_player
    state = jax.vmap(env.step)(state, action)
    obs = jax.vmap(env.observe)(state)
    logits, value, _ = forward(params, obs)
    
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
    root_value: jnp.ndarray  # MCTS 搜索后的标量价值估计
    root_wdl: jnp.ndarray    # 网络原始 WDL 概率 (B,3)，用于 WDL TD(λ) bootstrap

class Sample(NamedTuple):
    obs: jnp.ndarray
    policy_tgt: jnp.ndarray
    wdl_tgt: jnp.ndarray   # WDL TD(λ) 目标 (B,3)：[W, D, L] 概率分布
    mask: jnp.ndarray

# ============================================================================
# 自玩
# ============================================================================


@partial(jax.pmap, static_broadcasted_argnums=(2,))
def selfplay(params, rng_key, batch_size):
    """
    高性能自玩算子：
    - 使用 lax.scan 消除 Python 循环开销
    - Gumbel 算法无需 Dirichlet 噪声 (Gumbel 噪声已提供足够探索)
    - 统一策略（移除强弱差异，节省计算资源）
    - 仅使用标准初始局面，保证训练目标一致性
    """
    def step_fn(state, key):
        key_search, key_sample, key_reset = jax.random.split(key, 3)
        obs = jax.vmap(env.observe)(state)
        logits, value, wdl_logits = forward(params, obs)
        root_wdl = jax.nn.softmax(wdl_logits, axis=-1)  # (B, 3) WDL 概率
        
        logits = jnp.where(state.current_player[:, None] == 0, logits, logits[:, _ROTATED_IDX])
        
        # MCTS 搜索（移除了全局 cond 分支，避免编译两套计算图的开销）
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
        
        temp = jnp.where(
            state.step_count < config.temperature_phase1_steps,
            config.temperature_phase1,
            jnp.where(
                state.step_count < config.temperature_phase2_steps,
                config.temperature_phase2,
                config.temperature_final,
            ),
        )
        
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
        
        # 执行动作
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
            root_wdl=root_wdl,
        )
        
        # 结束后直接重置为标准初始局面
        next_state_reset = jax.vmap(lambda s, k: jax.lax.cond(
            s.terminated, lambda: env.init(k), lambda: s
        ))(next_state, jax.random.split(key_reset, batch_size))
        return next_state_reset, data

    state = jax.vmap(env.init)(jax.random.split(rng_key, batch_size))
    _, data = jax.lax.scan(step_fn, state, jax.random.split(rng_key, config.max_steps))
    return data

@jax.pmap
def compute_targets(data: SelfplayOutput):
    """WDL TD(λ) 目标计算 - 在 [W, D, L] 三分量上独立做 TD(λ)
    
    与标量 TD(λ) 的区别：
    - bootstrap 用网络原始 WDL 概率（而非 MCTS 标量值）
    - 视角翻转用 W↔L 互换（而非标量取反）
    - 游戏结果直接映射为 one-hot WDL
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
    
    返回 (stats[12], avg_length) 均为 float32
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
    
    return stats, avg_length

# ============================================================================
# 训练与评估
# ============================================================================

def loss_fn(params, samples: Sample, rng_key):
    """损失函数
    
    - 策略损失：交叉熵
    - 价值损失：WDL 交叉熵（比标量 MSE 梯度信号更强，精确区分"和棋"与"不确定"）
    """
    obs = samples.obs.astype(jnp.float32)
    policy_tgt = samples.policy_tgt
    
    # 随机镜像增强
    do_mirror = jax.random.bernoulli(rng_key, 0.3)
    obs = jnp.where(do_mirror, jax.vmap(mirror_observation)(obs), obs)
    policy_tgt = jnp.where(do_mirror, jax.vmap(mirror_policy)(policy_tgt), policy_tgt)
    
    logits, _value, wdl_logits = forward(params, obs, is_training=True)
    
    logits = logits.astype(jnp.float32)
    wdl_logits = wdl_logits.astype(jnp.float32)
    policy_tgt = policy_tgt.astype(jnp.float32)
    wdl_tgt = samples.wdl_tgt.astype(jnp.float32)  # (B, 3) WDL 概率分布
    
    # 策略损失
    policy_ce = optax.softmax_cross_entropy(logits, policy_tgt)
    policy_loss = jnp.sum(policy_ce * samples.mask) / jnp.maximum(jnp.sum(samples.mask), 1.0)
    
    # 价值损失：WDL 交叉熵（目标已由 compute_targets 正确计算）
    value_loss_per = -jnp.sum(wdl_tgt * jax.nn.log_softmax(wdl_logits, axis=-1), axis=-1)
    value_loss = jnp.sum(value_loss_per * samples.mask) / jnp.maximum(jnp.sum(samples.mask), 1.0)
    
    total_loss = policy_loss + config.value_loss_weight * value_loss
    return total_loss, (policy_loss, value_loss)

def _build_eval_initial_states(
    fen_file: Optional[str],
    batch_size: int,
    rng_key,
    for_current_red: bool,
) -> "XiangqiState":
    """
    构建评估用初始状态，支持 FEN 批量导入与先后手轮换。
    
    Args:
        fen_file: FEN 文件路径，None 则用标准初始局面
        batch_size: 每侧局数（current=red 或 current=black 各 batch_size 局）
        rng_key: 随机 key（用于标准局面时的 init）
        for_current_red: True=当前模型执红，False=当前模型执黑
        
    Returns:
        已按设备分片的 XiangqiState，可直接传入 evaluate
    """
    batch_per_device = batch_size // num_devices
    if batch_per_device * num_devices != batch_size:
        raise ValueError(f"batch_size {batch_size} 必须整除 num_devices {num_devices}")
    
    if fen_file and os.path.exists(fen_file):
        # 从 FEN 文件加载，先后手轮换
        fens = load_fens_from_file(fen_file)
        if not fens:
            raise ValueError(f"FEN 文件为空: {fen_file}")
        if for_current_red:
            print(f"[评估] FEN 文件 {fen_file} 共 {len(fens)} 条局面，先后手轮换")
        # 先后手轮换：偶数索引→current=red，奇数索引→current=black
        boards_r, players_r = [], []
        boards_b, players_b = [], []
        for i, (board, player) in enumerate(fens):
            if i % 2 == 0:
                # 当前模型执红：需红方行棋，否则镜像
                if player == 0:
                    boards_r.append(board)
                    players_r.append(0)
                else:
                    b_mirror = np.array(-np.flip(board, axis=-1), dtype=np.int8)
                    boards_r.append(b_mirror)
                    players_r.append(0)
            else:
                # 当前模型执黑：需黑方行棋，否则镜像
                if player == 1:
                    boards_b.append(board)
                    players_b.append(1)
                else:
                    b_mirror = np.array(-np.flip(board, axis=-1), dtype=np.int8)
                    boards_b.append(b_mirror)
                    players_b.append(1)
        # 选对应侧并补齐到 batch_size
        if for_current_red:
            boards, players = boards_r, players_r
        else:
            boards, players = boards_b, players_b
        # 用标准局面补齐
        std_board, _ = parse_fen("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w")
        while len(boards) < batch_size:
            boards.append(std_board)
            players.append(0 if for_current_red else 1)
        boards = np.stack(boards[:batch_size], axis=0).astype(np.int8)
        players = np.array(players[:batch_size], dtype=np.int32)
        boards_jax = jnp.array(boards)
        players_jax = jnp.array(players)
        states_flat = jax.vmap(env.init_from_board)(boards_jax, players_jax)
    elif fen_file:
        print(f"[评估] FEN 文件不存在 {fen_file}，改用标准初始局面")
        keys = jax.random.split(rng_key, batch_size)
        states_flat = jax.vmap(env.init)(keys)
    else:
        # 标准初始局面
        keys = jax.random.split(rng_key, batch_size)
        states_flat = jax.vmap(env.init)(keys)
    
    # 按设备分片供 pmap 使用
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
        # 在根节点确定当前搜索归属于哪个模型
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
    """纯 NumPy 环形缓冲区 - 零设备冲突，极简稳定"""
    
    def __init__(self, max_size: int, obs_shape: tuple, action_size: int):
        self.max_size = max_size
        
        # 全部用 NumPy 数组（CPU 内存）
        self.obs = np.zeros((max_size, *obs_shape), dtype=np.uint8)
        self.policy_tgt = np.zeros((max_size, action_size), dtype=np.float32)
        self.wdl_tgt = np.zeros((max_size, 3), dtype=np.float32)  # WDL 概率分布
        self.mask = np.zeros((max_size,), dtype=np.bool_)
        
        self.ptr = 0
        self.size = 0
        self.total_added = 0

    def add(self, samples: Sample):
        """存入样本（从 JAX 自动转 NumPy）"""
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
    
    def sample(self, batch_size: int, rng_key) -> Sample:
        """采样后转回 JAX 数组
        
        优化：obs 保持 uint8 传输，减少 4x CPU→GPU 带宽
        在 GPU 上的 loss_fn 中再转为 float32
        """
        # 使用 JAX 随机数生成索引（保持计算图完整性）
        # 注意：这里仍需要用 NumPy，因为 buffer 本身在 CPU
        # 但我们可以减少 Python 逻辑
        idx = np.random.randint(0, self.size, size=batch_size)
        
        obs_batch = self.obs[idx]
        policy_batch = self.policy_tgt[idx]
        wdl_batch = self.wdl_tgt[idx]
        mask_batch = self.mask[idx]
        
        return Sample(
            obs=jnp.asarray(obs_batch, dtype=jnp.uint8),
            policy_tgt=jnp.asarray(policy_batch),
            wdl_tgt=jnp.asarray(wdl_batch),
            mask=jnp.asarray(mask_batch)
        )
    
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
        
        # 如果加载的数据比当前缓冲区小，直接复制到前面
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
    history_models: dict            # 历史模型 (用于 ELO 评估)
    iteration_elos: dict            # ELO 记录
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
):
    """保存完整训练状态
    
    包含：模型参数、优化器状态、迭代计数、随机数状态、历史模型、ELO
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
    }
    
    # 保存主状态
    ckpt_manager.save(step, args=ocp.args.StandardSave(state_dict))
    
    # 单独保存 metadata (history_models keys, elos)
    meta_dir = os.path.join(os.path.abspath(config.ckpt_dir), f"meta_{step}")
    os.makedirs(meta_dir, exist_ok=True)
    
    # 保存 ELO、历史模型索引、total_opt_steps
    with open(os.path.join(meta_dir, "metadata.json"), "w") as f:
        json.dump({
            "iteration_elos": {str(k): float(v) for k, v in train_state.iteration_elos.items()},
            "history_model_keys": [int(k) for k in train_state.history_models.keys()],
            "total_opt_steps": train_state.total_opt_steps,
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
) -> Optional[tuple]:
    """恢复训练状态
    
    Returns:
        None 如果没有 checkpoint
        (params, opt_state, iteration, frames, rng_key, history_models, iteration_elos, total_opt_steps) 如果成功恢复
    """
    latest_step = ckpt_manager.latest_step()
    if latest_step is None:
        print("[Checkpoint] 未找到已有 checkpoint，从头开始训练")
        return None
    
    print(f"[Checkpoint] 正在恢复 step={latest_step}...")
    
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
    
    # 恢复 metadata
    meta_dir = os.path.join(os.path.abspath(config.ckpt_dir), f"meta_{latest_step}")
    
    history_models = {}
    iteration_elos = {}
    
    total_opt_steps = 0
    if os.path.exists(meta_dir):
        with open(os.path.join(meta_dir, "metadata.json"), "r") as f:
            meta = json.load(f)
            iteration_elos = {int(k): v for k, v in meta["iteration_elos"].items()}
            history_model_keys = meta["history_model_keys"]
            total_opt_steps = int(meta.get("total_opt_steps", 0))
        
        tree_struct = jax.tree.structure(params_template)
        for k in history_model_keys:
            npz_path = os.path.join(meta_dir, f"history_{k}.npz")
            if os.path.exists(npz_path):
                data = np.load(npz_path)
                leaves = [data[f"arr_{i}"] for i in range(len(data.files))]
                history_models[k] = jax.tree.unflatten(tree_struct, leaves)
        
    if total_opt_steps == 0:
        total_opt_steps = (frames * config.sample_reuse_times) // config.training_batch_size
    
    if iteration_elos:
        latest_elo_iter = max(iteration_elos.keys())
        latest_elo = iteration_elos[latest_elo_iter]
        print(f"[Checkpoint] 恢复完成: iteration={iteration}, frames={frames}, opt_steps={total_opt_steps}, ELO={latest_elo:.0f} (iter {latest_elo_iter})")
    else:
        print(f"[Checkpoint] 恢复完成: iteration={iteration}, frames={frames}, opt_steps={total_opt_steps}")
    
    return params, opt_state, iteration, frames, rng_key, history_models, iteration_elos, total_opt_steps


# ============================================================================
# 主循环
# ============================================================================

def main():
    print("=" * 50 + "\nZeroForge - 现代高效架构\n" + "=" * 50)
    print("特性: 4分支GNN (Local+Row+Col+Global) + Gumbel + TD(λ) + 经验回放 + 断点续训")
    
    # 创建必要目录
    os.makedirs(config.ckpt_dir, exist_ok=True)
    
    # 初始化经验回放缓冲区 (JAX 环形缓冲区)
    replay_buffer = ReplayBuffer(
        max_size=config.replay_buffer_size,
        obs_shape=(NUM_OBSERVATION_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH),
        action_size=ACTION_SPACE_SIZE
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
    print(
        f"[LR] 余弦退火: {config.learning_rate:.1e} → {final_lr:.1e}, "
        f"周期={config.lr_cosine_steps}steps(≈{config.lr_cosine_steps // 266}轮), "
        f"warmup={config.lr_warmup_steps}steps"
    )
    print(f"[TD(λ)] 固定 λ={config.td_lambda}")
    optimizer = optax.adamw(lr_schedule, weight_decay=config.weight_decay)
    opt_state_template = optimizer.init(params_template)
    
    # === 创建 Checkpoint Manager 并尝试恢复 ===
    ckpt_manager = create_checkpoint_manager(config.ckpt_dir)
    restored = restore_checkpoint(ckpt_manager, params_template, opt_state_template)
    
    if restored is not None:
        params, opt_state, iteration, frames, rng_key, history_models, iteration_elos, total_opt_steps = restored
        print(f"[断点续训] 从 iteration={iteration} 继续训练")
    else:
        params = params_template
        opt_state = opt_state_template
        iteration = 0
        frames = 0
        history_models = {0: jax.device_get(params)}
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
    print(f"[Log] TensorBoard 日志: {run_log_dir}")
    writer = SummaryWriter(run_log_dir)
    start_time_total = time.time()
    
    print("开始训练！")
    
    # 恢复后若轮到评估（如崩溃在 ckpt 保存后、eval 前），补跑评估
    if restored is not None and iteration % config.eval_interval == 0 and iteration > 0:
        available_iters = sorted(k for k in history_models.keys() if k < iteration)
        if available_iters:
            print(f"[断点续训] 补跑 iteration={iteration} 的评估...")
            rated_iters = [k for k in available_iters if k in iteration_elos]
            ref_iter = max(rated_iters, key=lambda k: iteration_elos[k]) if rated_iters else available_iters[-1]
            ref_reason = "best_elo" if rated_iters else "latest_ckpt"
            past_params = replicate_to_devices(history_models[ref_iter])
            rng_key, sk5, sk6, sk7 = jax.random.split(rng_key, 4)
            batch_per_eval = config.eval_games
            states_r = _build_eval_initial_states(config.eval_fen_file, batch_per_eval, sk7, for_current_red=True)
            states_b = _build_eval_initial_states(config.eval_fen_file, batch_per_eval, sk7, for_current_red=False)
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
                print(f"[评估错误] 补跑评估失败: {e}")
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
            print(f"评估 vs Iter {ref_iter} ({ref_reason}): W/D/L {wins}/{draws}/{losses} | 得分率 {score:.2%} | ELO {iteration_elos[iteration]:.0f}")
            writer.add_scalar("eval/elo", iteration_elos[iteration], iteration)
            writer.add_scalar("eval/win_rate", wins / total_games, iteration)
            writer.add_scalar("eval/draw_rate", draws / total_games, iteration)
            writer.add_scalar("eval/loss_rate", losses / total_games, iteration)
            writer.add_scalar("eval/score", score, iteration)
            if decisive_games > 0:
                writer.add_scalar("eval/decisive_win_rate", decisive_win_rate, iteration)
    
    print(f"[Selfplay] batch_size={config.selfplay_batch_size}")

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
        stats_dev, avg_len_dev = gpu_stats
        stats_np = np.array(jax.device_get(stats_dev))
        stats = stats_np.sum(axis=0).astype(np.int64)   # 跨设备求和
        avg_length = float(jax.device_get(avg_len_dev).mean())

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
            batch_flat = replay_buffer.sample(config.training_batch_size, sample_keys[key_idx])
            batch = _shard_batch_for_devices(batch_flat)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", DeprecationWarning)
                keys = jax.device_put_sharded(
                    list(jax.random.split(sample_keys[key_idx], num_devices)), devices)
            return batch, keys
        
        # 双缓冲流水线：预读取第一个 batch，后续 batch 在 GPU 训练期间并行准备
        ploss_acc, vloss_acc = None, None
        next_batch, next_keys = _prefetch_batch(1) if num_updates > 0 else (None, None)
        
        for i in range(num_updates):
            batch, train_keys = next_batch, next_keys
            
            try:
                # 分派 GPU 训练（异步返回，不阻塞 Python）
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
            
            # GPU 训练期间，CPU 立即准备下一个 batch（流水线核心）
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
        
        print(f"iter={iteration:3d} | ploss={policy_loss:.4f} "
              f"vloss={value_loss:.4f} | "
              f"len={avg_length:4.1f} fps={fps:4.0f} buf={buf_stats['size']//1000}k train={num_updates} | "
              f"红{r_wins:3d} 黑{b_wins:3d} 和{draws:3d}")
        base_lr = float(lr_schedule(total_opt_steps))
        
        # TensorBoard 记录
        writer.add_scalar("train/policy_loss", policy_loss, iteration)
        writer.add_scalar("train/value_loss", value_loss, iteration)
        writer.add_scalar("train/lr", base_lr, iteration)
        writer.add_scalar("stats/avg_game_length", avg_length, iteration)
        writer.add_scalar("stats/fps", fps, iteration)
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
            params_np = jax.device_get(jax.tree.map(lambda x: x[0], params))
            history_models[iteration] = params_np
            
            train_state = TrainState(
                params=params,
                opt_state=opt_state,
                iteration=iteration,
                frames=frames,
                rng_key=rng_key,
                history_models=history_models,
                iteration_elos=iteration_elos,
                total_opt_steps=total_opt_steps,
            )
            save_checkpoint(ckpt_manager, train_state, iteration)
        
        if iteration % config.eval_interval == 0:
            # 评估对手优先选择“历史最佳模型”（按 iteration_elos），
            # 若无可用 ELO 记录，则退化为最近历史 checkpoint。
            available_iters = sorted(k for k in history_models.keys() if k < iteration)
            if available_iters:
                rated_iters = [k for k in available_iters if k in iteration_elos]
                if rated_iters:
                    ref_iter = max(rated_iters, key=lambda k: iteration_elos[k])
                    ref_reason = "best_elo"
                else:
                    ref_iter = available_iters[-1]
                    ref_reason = "latest_ckpt"

                past_params = replicate_to_devices(history_models[ref_iter])
                rng_key, sk5, sk6, sk7 = jax.random.split(rng_key, 4)
                batch_per_eval = config.eval_games
                # 构建初始状态（支持 FEN 批量导入 + 先后手轮换）
                states_r = _build_eval_initial_states(
                    config.eval_fen_file, batch_per_eval, sk7, for_current_red=True
                )
                states_b = _build_eval_initial_states(
                    config.eval_fen_file, batch_per_eval, sk7, for_current_red=False
                )
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
                    print(f"[评估错误] 当前模型 dtype={config.network_dtype}, action_size={ACTION_SPACE_SIZE}")
                    print(f"[评估错误] params dtype: {jax.tree.leaves(params)[0].dtype}")
                    print(f"[评估错误] past_params dtype: {jax.tree.leaves(past_params)[0].dtype}")
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
                print(
                    f"评估 vs Iter {ref_iter} ({ref_reason}): "
                    f"W/D/L {wins}/{draws}/{losses} | "
                    f"胜率 {win_rate:.2%} 和率 {draw_rate:.2%} 负率 {loss_rate:.2%} | "
                    f"得分率 {score:.2%} | 决胜局胜率 {decisive_text} | "
                    f"ELO {iteration_elos[iteration]:.0f}"
                )
                print(
                    f"  先手(当前红) W/D/L {wins_red}/{draws_red}/{losses_red} | "
                    f"后手(当前黑) W/D/L {wins_black}/{draws_black}/{losses_black}"
                )
                writer.add_scalar("eval/elo", iteration_elos[iteration], iteration)
                writer.add_scalar("eval/win_rate", win_rate, iteration)
                writer.add_scalar("eval/draw_rate", draw_rate, iteration)
                writer.add_scalar("eval/loss_rate", loss_rate, iteration)
                writer.add_scalar("eval/score", score, iteration)
                if decisive_games > 0:
                    writer.add_scalar("eval/decisive_win_rate", decisive_win_rate, iteration)
            else:
                print(f"[警告] 无可用历史模型，跳过本次评估")

if __name__ == "__main__":
    main()

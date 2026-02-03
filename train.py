#!/usr/bin/env python3
"""
ZeroForge - 中国象棋 Pikafish 对弈训练
模型 vs Pikafish + 深度自适应 + 评分差提前结束
"""

import os
import sys
import time
import json
import warnings
import argparse
import threading
from functools import partial
from typing import NamedTuple, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

# --- JAX 编译缓存配置 ---
cache_dir = os.path.abspath("jax_cache")
os.makedirs(cache_dir, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", cache_dir)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

# --- XLA 编译加速 ---
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_highest_priority_async_stream=true"
jax.config.update("jax_enable_x64", False)

from xiangqi.env import XiangqiEnv, NUM_OBSERVATION_CHANNELS
from xiangqi.actions import (
    rotate_action, ACTION_SPACE_SIZE, BOARD_HEIGHT, BOARD_WIDTH,
    action_to_move, move_to_action, uci_to_move, move_to_uci,
)
from xiangqi.mirror import mirror_observation, mirror_policy
from xiangqi.pikafish import PikafishPool, board_to_fen
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
    num_channels: int = 96
    num_blocks: int = 6
    
    # 训练超参数
    learning_rate: float = 2e-4
    training_batch_size: int = 512
    
    # 经验回放配置
    replay_buffer_size: int = 2000000
    sample_reuse_times: int = 4
    
    # 损失权重
    value_loss_weight: float = 1.5
    weight_decay: float = 1e-4
    
    # 探索策略
    temperature_steps: int = 40
    temperature_initial: float = 1.0
    temperature_final: float = 0.1
    
    # 环境规则
    max_steps: int = 200
    max_no_capture_steps: int = 120
    repetition_threshold: int = 5
    
    # Checkpoint 配置
    ckpt_interval: int = 10
    max_to_keep: int = 20
    keep_period: int = 50
    
    # ========== Pikafish 对弈配置 ==========
    pikafish_path: str = "pikafish"          # Pikafish 可执行文件路径
    pikafish_num_engines: int = 16           # Pikafish 引擎实例数量
    pikafish_initial_depth: int = 1          # 初始搜索深度
    pikafish_min_depth: int = 1              # 最小搜索深度
    pikafish_max_depth: int = 30             # 最大搜索深度
    
    # 深度自适应阈值
    pikafish_depth_up_winrate: float = 0.80  # 胜率超过此值，深度 +1
    pikafish_depth_down_winrate: float = 0.20  # 胜率低于此值，深度 -1
    pikafish_winrate_window: int = 5         # 计算胜率的滑动窗口
    
    # 评分差提前结束
    pikafish_resign_threshold: int = 350     # 评分差超过此值判定结束（厘兵）
    pikafish_resign_check_interval: int = 5  # 每 N 步检查一次
    
    # 模型执子配置
    pikafish_alternate_color: bool = True    # 交替执红执黑

config = Config()

# ============================================================================
# 环境和设备
# ============================================================================

devices = jax.local_devices()
num_devices = len(devices)

# 预计算旋转索引
_ROTATED_IDX = rotate_action(jnp.arange(ACTION_SPACE_SIZE))
_ROTATED_IDX_NP = np.array(_ROTATED_IDX)  # NumPy 版本用于索引

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

net = AlphaZeroNetwork(
    action_space_size=env.action_space_size,
    channels=config.num_channels,
    num_blocks=config.num_blocks,
    dtype=jnp.bfloat16,
)

def forward(params, obs, is_training=False):
    """前向传播"""
    logits, value = net.apply({'params': params}, obs, train=is_training)
    return logits, value

# ============================================================================
# 数据结构
# ============================================================================

class Sample(NamedTuple):
    obs: jnp.ndarray        # 观察 (uint8)
    policy_tgt: jnp.ndarray # 策略目标
    value_tgt: jnp.ndarray  # 价值目标（MC 目标：最终结果）
    mask: jnp.ndarray       # 掩码

# ============================================================================
# Pikafish 对弈统计
# ============================================================================

class PikafishStats:
    """Pikafish 对弈统计"""
    
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.games_played = 0
        self.model_wins = 0
        self.pikafish_wins = 0
        self.draws = 0
        self.total_steps = 0
        self.resign_count = 0
        
    def add_game(self, model_won: bool, pikafish_won: bool, steps: int, resigned: bool):
        self.games_played += 1
        self.total_steps += steps
        if resigned:
            self.resign_count += 1
        if model_won:
            self.model_wins += 1
        elif pikafish_won:
            self.pikafish_wins += 1
        else:
            self.draws += 1
            
    def get_winrate(self) -> float:
        if self.games_played == 0:
            return 0.5
        return (self.model_wins + 0.5 * self.draws) / self.games_played
        
    def get_avg_length(self) -> float:
        if self.games_played == 0:
            return 0.0
        return self.total_steps / self.games_played

# ============================================================================
# Pikafish 对弈
# ============================================================================

def play_single_game_vs_pikafish(
    params_np: dict,
    pikafish_engine,
    model_plays_red: bool,
    rng_key: jnp.ndarray,
) -> tuple:
    """
    与 Pikafish 进行单局对弈
    
    Returns:
        (obs_list, policy_list, final_value, model_won, pikafish_won, steps, resigned)
    """
    # 初始化环境
    state = env.init(rng_key)
    
    # 收集模型走的步骤
    obs_list = []
    action_weights_list = []
    
    resigned = False
    step = 0
    last_score = 0
    
    while not state.terminated and step < config.max_steps:
        current_player = int(state.current_player)
        is_model_turn = (current_player == 0) == model_plays_red
        
        if is_model_turn:
            # 模型走子
            obs = env.observe(state)
            obs_list.append(np.array(obs, dtype=np.uint8))
            
            # 网络推理
            logits, value = forward(params_np, obs[None, ...])
            logits = np.array(logits[0])
            
            # 视角归一化
            if current_player == 1:
                logits = logits[_ROTATED_IDX_NP]
            
            # 温度采样
            temp = config.temperature_initial if step < config.temperature_steps else config.temperature_final
            logits = logits - np.max(logits)
            legal_mask = np.array(state.legal_action_mask)
            logits = np.where(legal_mask, logits, -1e10)
            
            exp_logits = np.exp(logits / max(temp, 1e-3))
            probs = exp_logits / (np.sum(exp_logits) + 1e-10)
            
            rng_key, subkey = jax.random.split(rng_key)
            action = int(jax.random.choice(subkey, ACTION_SPACE_SIZE, p=probs))
            
            # 记录策略（归一化到红方视角）
            action_weights = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
            action_weights[action] = 1.0
            if current_player == 1:
                action_weights = action_weights[_ROTATED_IDX_NP]
            action_weights_list.append(action_weights)
            
        else:
            # Pikafish 走子
            board_np = np.array(state.board)
            fen = board_to_fen(board_np, current_player)
            
            uci_move, score = pikafish_engine.get_best_move(fen)
            last_score = score if current_player == 0 else -score  # 统一为红方视角
            
            if uci_move is None:
                break
                
            from_sq, to_sq = uci_to_move(uci_move)
            action = int(move_to_action(jnp.int32(from_sq), jnp.int32(to_sq)))
            
            if action < 0 or not state.legal_action_mask[action]:
                print(f"[警告] Pikafish 走法转换失败: {uci_move}")
                break
            
        # 执行动作
        state = env.step(state, action)
        step += 1
        
        # 检查评分差提前结束
        if (step % config.pikafish_resign_check_interval == 0 and 
            abs(last_score) >= config.pikafish_resign_threshold):
            resigned = True
            break
    
    # 确定胜负
    if state.terminated:
        winner = int(state.winner)
    elif resigned:
        winner = 0 if last_score > 0 else 1
    else:
        winner = -1
        
    model_won = (winner == 0) == model_plays_red if winner != -1 else False
    pikafish_won = (winner == 0) != model_plays_red if winner != -1 else False
    
    # MC 目标：最终结果
    if model_won:
        final_value = 1.0
    elif pikafish_won:
        final_value = -1.0
    else:
        final_value = 0.0
    
    return (obs_list, action_weights_list, final_value, 
            model_won, pikafish_won, step, resigned)


def play_vs_pikafish_batch(
    params,
    pikafish_pool,
    num_games: int,
    rng_key: jnp.ndarray,
) -> tuple:
    """批量与 Pikafish 对弈"""
    params_np = jax.device_get(jax.tree.map(lambda x: x[0], params))
    
    stats = PikafishStats()
    all_obs = []
    all_policy = []
    all_values = []
    
    game_keys = jax.random.split(rng_key, num_games)
    
    def play_game(game_idx):
        engine = pikafish_pool.acquire()
        if engine is None:
            return None
        try:
            engine.new_game()
            model_plays_red = (game_idx % 2 == 0) if config.pikafish_alternate_color else True
            return play_single_game_vs_pikafish(
                params_np, engine, model_plays_red, game_keys[game_idx]
            )
        finally:
            pikafish_pool.release(engine)
    
    # 并行对弈
    with ThreadPoolExecutor(max_workers=pikafish_pool.num_engines) as executor:
        futures = {executor.submit(play_game, i): i for i in range(num_games)}
        
        for future in as_completed(futures):
            result = future.result()
            if result is None:
                continue
                
            (obs_list, policy_list, final_value,
             model_won, pikafish_won, steps, resigned) = result
            
            if len(obs_list) == 0:
                continue
            
            # 收集样本（每个步骤都使用 MC 目标：最终结果）
            all_obs.extend(obs_list)
            all_policy.extend(policy_list)
            all_values.extend([final_value] * len(obs_list))
            
            stats.add_game(model_won, pikafish_won, steps, resigned)
    
    if len(all_obs) == 0:
        return None, stats
    
    # 构造 Sample
    samples = Sample(
        obs=jnp.array(np.array(all_obs, dtype=np.uint8)),
        policy_tgt=jnp.array(np.array(all_policy, dtype=np.float32)),
        value_tgt=jnp.array(np.array(all_values, dtype=np.float32)),
        mask=jnp.ones(len(all_obs), dtype=jnp.bool_),
    )
    
    return samples, stats


def adaptive_depth_update(current_depth: int, winrate_history: list) -> int:
    """根据胜率自适应调整深度"""
    window = config.pikafish_winrate_window
    if len(winrate_history) < window:
        return current_depth
        
    recent_winrate = sum(winrate_history[-window:]) / window
    
    new_depth = current_depth
    if recent_winrate >= config.pikafish_depth_up_winrate:
        new_depth = min(current_depth + 1, config.pikafish_max_depth)
        print(f"[深度调整] 胜率 {recent_winrate:.1%} >= {config.pikafish_depth_up_winrate:.0%}, "
              f"深度 {current_depth} → {new_depth}")
    elif recent_winrate <= config.pikafish_depth_down_winrate:
        new_depth = max(current_depth - 1, config.pikafish_min_depth)
        print(f"[深度调整] 胜率 {recent_winrate:.1%} <= {config.pikafish_depth_down_winrate:.0%}, "
              f"深度 {current_depth} → {new_depth}")
              
    return new_depth

# ============================================================================
# 训练
# ============================================================================

def loss_fn(params, samples: Sample, rng_key):
    """损失函数：策略 + 价值（MC 目标）"""
    obs = samples.obs.astype(jnp.float32)
    policy_tgt = samples.policy_tgt
    
    # 随机镜像增强
    do_mirror = jax.random.bernoulli(rng_key, 0.5)
    obs = jnp.where(do_mirror, jax.vmap(mirror_observation)(obs), obs)
    policy_tgt = jnp.where(do_mirror, jax.vmap(mirror_policy)(policy_tgt), policy_tgt)
    
    logits, value = forward(params, obs, is_training=True)
    
    logits = logits.astype(jnp.float32)
    value = value.astype(jnp.float32)
    policy_tgt = policy_tgt.astype(jnp.float32)
    value_tgt = samples.value_tgt.astype(jnp.float32)
    
    # 策略损失
    policy_loss = jnp.sum(optax.softmax_cross_entropy(logits, policy_tgt) * samples.mask) / jnp.maximum(jnp.sum(samples.mask), 1.0)
    
    # 价值损失（MC 目标）
    value_loss = jnp.sum(optax.l2_loss(value, value_tgt) * samples.mask) / jnp.maximum(jnp.sum(samples.mask), 1.0)
    
    total_loss = policy_loss + config.value_loss_weight * value_loss
    return total_loss, (policy_loss, value_loss)

# ============================================================================
# 经验回放缓冲区
# ============================================================================

class ReplayBuffer:
    """NumPy 环形缓冲区"""
    
    def __init__(self, max_size: int, obs_shape: tuple, action_size: int):
        self.max_size = max_size
        self.obs = np.zeros((max_size, *obs_shape), dtype=np.uint8)
        self.policy_tgt = np.zeros((max_size, action_size), dtype=np.float32)
        self.value_tgt = np.zeros((max_size,), dtype=np.float32)
        self.mask = np.zeros((max_size,), dtype=np.bool_)
        self.ptr = 0
        self.size = 0
        self.total_added = 0

    def add(self, samples: Sample):
        """添加样本"""
        obs_np = np.array(samples.obs, dtype=np.uint8)
        policy_np = np.array(samples.policy_tgt, dtype=np.float32)
        value_np = np.array(samples.value_tgt, dtype=np.float32)
        mask_np = np.array(samples.mask, dtype=np.bool_)
        
        n_new = obs_np.shape[0]
        indices = (np.arange(n_new) + self.ptr) % self.max_size
        
        self.obs[indices] = obs_np
        self.policy_tgt[indices] = policy_np
        self.value_tgt[indices] = value_np
        self.mask[indices] = mask_np
        
        self.ptr = (self.ptr + n_new) % self.max_size
        self.size = min(self.size + n_new, self.max_size)
        self.total_added += n_new
    
    def sample(self, batch_size: int, rng_key) -> Sample:
        """采样"""
        idx = np.random.randint(0, self.size, size=batch_size)
        return Sample(
            obs=jnp.asarray(self.obs[idx], dtype=jnp.uint8),
            policy_tgt=jnp.asarray(self.policy_tgt[idx]),
            value_tgt=jnp.asarray(self.value_tgt[idx]),
            mask=jnp.asarray(self.mask[idx])
        )
    
    def stats(self):
        return {"size": self.size, "total_added": self.total_added}

# ============================================================================
# Checkpoint 管理
# ============================================================================

class TrainState(NamedTuple):
    params: dict
    opt_state: dict
    iteration: int
    frames: int
    rng_key: jnp.ndarray
    pikafish_depth: int  # 当前 Pikafish 深度
    winrate_history: list  # 胜率历史


def create_checkpoint_manager(ckpt_dir: str) -> ocp.CheckpointManager:
    ckpt_dir = os.path.abspath(ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    options = ocp.CheckpointManagerOptions(
        max_to_keep=config.max_to_keep,
        keep_period=config.keep_period,
        save_interval_steps=1,
    )
    return ocp.CheckpointManager(directory=ckpt_dir, options=options)


def save_checkpoint(ckpt_manager: ocp.CheckpointManager, train_state: TrainState, step: int):
    """保存 checkpoint"""
    params_np = jax.device_get(jax.tree.map(lambda x: x[0], train_state.params))
    opt_state_np = jax.device_get(jax.tree.map(lambda x: x[0], train_state.opt_state))
    
    state_dict = {
        "params": params_np,
        "opt_state": opt_state_np,
        "iteration": np.array(train_state.iteration),
        "frames": np.array(train_state.frames),
        "rng_key": jax.device_get(train_state.rng_key),
        "pikafish_depth": np.array(train_state.pikafish_depth),
    }
    ckpt_manager.save(step, args=ocp.args.StandardSave(state_dict))
    
    # 保存胜率历史
    meta_dir = os.path.join(os.path.abspath(config.ckpt_dir), f"meta_{step}")
    os.makedirs(meta_dir, exist_ok=True)
    with open(os.path.join(meta_dir, "winrate_history.json"), "w") as f:
        json.dump(train_state.winrate_history, f)
    
    print(f"[Checkpoint] 已保存 step={step}, depth={train_state.pikafish_depth}")


def restore_checkpoint(ckpt_manager: ocp.CheckpointManager, params_template: dict, opt_state_template: dict) -> Optional[tuple]:
    """恢复 checkpoint"""
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
        "pikafish_depth": np.array(config.pikafish_initial_depth),
    }
    
    restored = ckpt_manager.restore(latest_step, args=ocp.args.StandardRestore(restore_target))
    
    params = restored["params"]
    opt_state = restored["opt_state"]
    iteration = int(restored["iteration"])
    frames = int(restored["frames"])
    rng_key = restored["rng_key"]
    pikafish_depth = int(restored["pikafish_depth"])
    
    # 恢复胜率历史
    winrate_history = []
    meta_dir = os.path.join(os.path.abspath(config.ckpt_dir), f"meta_{latest_step}")
    history_file = os.path.join(meta_dir, "winrate_history.json")
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            winrate_history = json.load(f)
    
    print(f"[Checkpoint] 恢复完成: iteration={iteration}, frames={frames}, depth={pikafish_depth}")
    return params, opt_state, iteration, frames, rng_key, pikafish_depth, winrate_history

# ============================================================================
# 主循环
# ============================================================================

def main():
    print("=" * 60)
    print("ZeroForge - Pikafish 对弈训练")
    print("=" * 60)
    print(f"Pikafish 路径: {config.pikafish_path}")
    print(f"引擎数量: {config.pikafish_num_engines}")
    print(f"初始深度: {config.pikafish_initial_depth}")
    print(f"深度范围: [{config.pikafish_min_depth}, {config.pikafish_max_depth}]")
    print(f"深度+1 阈值: 胜率 >= {config.pikafish_depth_up_winrate:.0%}")
    print(f"深度-1 阈值: 胜率 <= {config.pikafish_depth_down_winrate:.0%}")
    print(f"提前结束: 评分差 >= {config.pikafish_resign_threshold} cp")
    print("=" * 60)
    
    # 启动 Pikafish 引擎池
    pikafish_pool = PikafishPool(config.pikafish_path, config.pikafish_num_engines)
    success_count = pikafish_pool.start()
    
    if success_count == 0:
        print("[错误] 无法启动 Pikafish 引擎！")
        print("请确保 pikafish 在 PATH 中，或使用 --pikafish_path 指定路径")
        return
    
    # 创建目录
    os.makedirs(config.ckpt_dir, exist_ok=True)
    
    # 初始化回放缓冲区
    replay_buffer = ReplayBuffer(
        max_size=config.replay_buffer_size,
        obs_shape=(NUM_OBSERVATION_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH),
        action_size=ACTION_SPACE_SIZE
    )
    
    # 初始化模型
    rng_key = jax.random.PRNGKey(config.seed)
    rng_key, subkey = jax.random.split(rng_key)
    dummy_obs = jnp.zeros((1, NUM_OBSERVATION_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH))
    variables = net.init(subkey, dummy_obs, train=True)
    params_template = variables['params']
    
    optimizer = optax.adamw(config.learning_rate, weight_decay=config.weight_decay)
    opt_state_template = optimizer.init(params_template)
    
    # 尝试恢复
    ckpt_manager = create_checkpoint_manager(config.ckpt_dir)
    restored = restore_checkpoint(ckpt_manager, params_template, opt_state_template)
    
    if restored is not None:
        params, opt_state, iteration, frames, rng_key, current_depth, winrate_history = restored
        print(f"[断点续训] iteration={iteration}, depth={current_depth}")
    else:
        params = params_template
        opt_state = opt_state_template
        iteration = 0
        frames = 0
        current_depth = config.pikafish_initial_depth
        winrate_history = []
    
    # 设置深度
    pikafish_pool.set_depth(current_depth)
    
    # 分发到设备
    params = replicate_to_devices(params)
    opt_state = replicate_to_devices(opt_state)
    
    @partial(jax.pmap, axis_name='i')
    def train_step(params, opt_state, samples, rng_key):
        grads, (ploss, vloss) = jax.grad(loss_fn, has_aux=True)(params, samples, rng_key)
        updates, opt_state = optimizer.update(jax.lax.pmean(grads, 'i'), opt_state, params)
        return optax.apply_updates(params, updates), opt_state, ploss, vloss
    
    writer = SummaryWriter(config.log_dir)
    
    print(f"\n开始训练！每次迭代对弈 {config.pikafish_num_engines * 4} 局\n")
    
    try:
        while True:
            iteration += 1
            st = time.time()
            
            # 与 Pikafish 对弈
            rng_key, sk = jax.random.split(rng_key)
            num_games = config.pikafish_num_engines * 4
            
            samples, stats = play_vs_pikafish_batch(params, pikafish_pool, num_games, sk)
            
            if samples is None:
                print(f"[警告] 迭代 {iteration} 未收集到样本")
                continue
            
            # 更新胜率历史和深度
            winrate = stats.get_winrate()
            winrate_history.append(winrate)
            
            new_depth = adaptive_depth_update(current_depth, winrate_history)
            if new_depth != current_depth:
                current_depth = new_depth
                pikafish_pool.set_depth(current_depth)
            
            # 添加到缓冲区
            replay_buffer.add(samples)
            new_frames = len(samples.obs)
            frames += new_frames
            
            # 训练
            num_updates = max(1, min((new_frames * config.sample_reuse_times) // config.training_batch_size, 10))
            
            sample_keys = jax.random.split(rng_key, num_updates + 1)
            rng_key = sample_keys[0]
            
            ploss_acc, vloss_acc = None, None
            actual_updates = 0
            
            for i in range(num_updates):
                if replay_buffer.size < config.training_batch_size:
                    break
                    
                batch_flat = replay_buffer.sample(config.training_batch_size, sample_keys[i+1])
                batch = jax.tree.map(lambda x: x.reshape((num_devices, -1) + x.shape[1:]), batch_flat)
                
                train_key = jax.random.split(sample_keys[i+1], 1)[0]
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", DeprecationWarning)
                    train_keys = jax.device_put_sharded(list(jax.random.split(train_key, num_devices)), devices)
                
                params, opt_state, ploss, vloss = train_step(params, opt_state, batch, train_keys)
                
                if ploss_acc is None:
                    ploss_acc, vloss_acc = ploss, vloss
                else:
                    ploss_acc = ploss_acc + ploss
                    vloss_acc = vloss_acc + vloss
                actual_updates += 1
            
            # 计算损失
            if actual_updates > 0:
                avg_ploss = float(ploss_acc.mean() / actual_updates)
                avg_vloss = float(vloss_acc.mean() / actual_updates)
            else:
                avg_ploss, avg_vloss = 0.0, 0.0
            
            # 统计
            iter_time = time.time() - st
            fps = new_frames / iter_time
            buf_stats = replay_buffer.stats()
            avg_len = stats.get_avg_length()
            
            # 打印日志
            print(f"iter={iteration:4d} | depth={current_depth:2d} | "
                  f"胜率={winrate:.1%} (模型{stats.model_wins} 鱼{stats.pikafish_wins} 和{stats.draws}) | "
                  f"len={avg_len:.1f} resign={stats.resign_count} | "
                  f"ploss={avg_ploss:.4f} vloss={avg_vloss:.4f} | "
                  f"fps={fps:.0f} buf={buf_stats['size']//1000}k")
            
            # TensorBoard
            writer.add_scalar("pikafish/depth", current_depth, iteration)
            writer.add_scalar("pikafish/winrate", winrate, iteration)
            writer.add_scalar("pikafish/model_wins", stats.model_wins, iteration)
            writer.add_scalar("pikafish/pikafish_wins", stats.pikafish_wins, iteration)
            writer.add_scalar("pikafish/draws", stats.draws, iteration)
            writer.add_scalar("pikafish/resign_count", stats.resign_count, iteration)
            writer.add_scalar("pikafish/avg_length", avg_len, iteration)
            writer.add_scalar("train/policy_loss", avg_ploss, iteration)
            writer.add_scalar("train/value_loss", avg_vloss, iteration)
            writer.add_scalar("stats/fps", fps, iteration)
            writer.add_scalar("replay/buffer_size", buf_stats['size'], iteration)
            
            # 保存 checkpoint
            if iteration % config.ckpt_interval == 0:
                train_state = TrainState(
                    params=params,
                    opt_state=opt_state,
                    iteration=iteration,
                    frames=frames,
                    rng_key=rng_key,
                    pikafish_depth=current_depth,
                    winrate_history=winrate_history,
                )
                save_checkpoint(ckpt_manager, train_state, iteration)
                
    except KeyboardInterrupt:
        print("\n[中断] 正在关闭...")
    finally:
        pikafish_pool.shutdown()
        print("[完成] Pikafish 引擎已关闭")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZeroForge Pikafish 对弈训练")
    parser.add_argument("--pikafish_path", type=str, default="pikafish",
                        help="Pikafish 可执行文件路径")
    parser.add_argument("--engines", type=int, default=16,
                        help="Pikafish 引擎数量")
    parser.add_argument("--depth", type=int, default=1,
                        help="初始搜索深度")
    parser.add_argument("--resign", type=int, default=350,
                        help="评分差提前结束阈值（厘兵）")
    
    args = parser.parse_args()
    
    config.pikafish_path = args.pikafish_path
    config.pikafish_num_engines = args.engines
    config.pikafish_initial_depth = args.depth
    config.pikafish_resign_threshold = args.resign
    
    main()

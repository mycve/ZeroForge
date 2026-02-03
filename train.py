#!/usr/bin/env python3
"""
ZeroForge - 中国象棋 Pikafish 对弈训练
高效架构：GPU 批量推理 + Pikafish 进程池异步轮询
"""

import os
import time
import json
import warnings
import argparse
import threading
import queue
from functools import partial
from typing import NamedTuple, Optional, List, Dict
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

# --- JAX 配置 ---
cache_dir = os.path.abspath("jax_cache")
os.makedirs(cache_dir, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", cache_dir)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
os.environ["XLA_FLAGS"] = "--xla_gpu_enable_highest_priority_async_stream=true"
jax.config.update("jax_enable_x64", False)

from xiangqi.env import XiangqiEnv, XiangqiState, NUM_OBSERVATION_CHANNELS
from xiangqi.actions import (
    rotate_action, ACTION_SPACE_SIZE, BOARD_HEIGHT, BOARD_WIDTH,
    move_to_action, uci_to_move,
)
from xiangqi.mirror import mirror_observation, mirror_policy
from xiangqi.pikafish import PikafishPool, board_to_fen
from networks.alphazero import AlphaZeroNetwork
from tensorboardX import SummaryWriter

# ============================================================================
# 配置
# ============================================================================

class Config:
    seed: int = 42
    ckpt_dir: str = "checkpoints"
    log_dir: str = "logs"
    
    # 网络架构
    num_channels: int = 96
    num_blocks: int = 6
    
    # 训练
    learning_rate: float = 2e-4
    training_batch_size: int = 512
    replay_buffer_size: int = 2000000
    sample_reuse_times: int = 4
    value_loss_weight: float = 1.5
    weight_decay: float = 1e-4
    
    # 探索
    temperature_steps: int = 40
    temperature_initial: float = 1.0
    temperature_final: float = 0.1
    
    # 环境
    max_steps: int = 200
    
    # Checkpoint
    ckpt_interval: int = 10
    max_to_keep: int = 20
    keep_period: int = 50
    
    # ========== Pikafish 对弈配置 ==========
    pikafish_path: str = "./pikafish"
    pikafish_num_engines: int = 64           # 进程池大小（建议 = CPU 核心数）
    pikafish_concurrent_games: int = 256     # 同时进行的游戏数
    pikafish_initial_depth: int = 1
    pikafish_min_depth: int = 1
    pikafish_max_depth: int = 30
    
    # 深度自适应
    pikafish_depth_up_winrate: float = 0.80
    pikafish_depth_down_winrate: float = 0.20
    pikafish_winrate_window: int = 5
    
    # 评分差提前结束
    pikafish_resign_threshold: int = 350
    pikafish_resign_check_interval: int = 5
    
    # 交替执子
    pikafish_alternate_color: bool = True

config = Config()

# ============================================================================
# 环境和模型
# ============================================================================

devices = jax.local_devices()
num_devices = len(devices)

_ROTATED_IDX = rotate_action(jnp.arange(ACTION_SPACE_SIZE))
_ROTATED_IDX_NP = np.array(_ROTATED_IDX)

def replicate_to_devices(pytree):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return jax.device_put_replicated(pytree, devices)

env = XiangqiEnv(max_steps=config.max_steps)

net = AlphaZeroNetwork(
    action_space_size=env.action_space_size,
    channels=config.num_channels,
    num_blocks=config.num_blocks,
    dtype=jnp.bfloat16,
)

def forward(params, obs, is_training=False):
    logits, value = net.apply({'params': params}, obs, train=is_training)
    return logits, value

# JIT 编译的批量推理
@jax.jit
def batch_forward(params, obs_batch):
    """批量前向传播"""
    logits, values = net.apply({'params': params}, obs_batch, train=False)
    return logits, values

# ============================================================================
# 数据结构
# ============================================================================

class Sample(NamedTuple):
    obs: jnp.ndarray
    policy_tgt: jnp.ndarray
    value_tgt: jnp.ndarray
    mask: jnp.ndarray

@dataclass
class GameInstance:
    """单局游戏实例"""
    game_id: int
    state: XiangqiState
    model_plays_red: bool
    rng_key: jnp.ndarray
    
    # 收集的样本
    obs_list: List[np.ndarray] = field(default_factory=list)
    policy_list: List[np.ndarray] = field(default_factory=list)
    
    # 游戏状态
    step: int = 0
    last_score: int = 0
    finished: bool = False
    resigned: bool = False
    model_won: bool = False
    pikafish_won: bool = False
    
    # 当前等待状态
    waiting_for: str = "model"  # "model" 或 "pikafish"
    pending_fen: str = ""
    pending_engine_id: int = -1

class PikafishStats:
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.games_played = 0
        self.model_wins = 0
        self.pikafish_wins = 0
        self.draws = 0
        self.total_steps = 0
        self.resign_count = 0
        
    def add_game(self, model_won, pikafish_won, steps, resigned):
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
            
    def get_winrate(self):
        if self.games_played == 0:
            return 0.5
        return (self.model_wins + 0.5 * self.draws) / self.games_played
        
    def get_avg_length(self):
        if self.games_played == 0:
            return 0.0
        return self.total_steps / self.games_played

# ============================================================================
# 高效对弈引擎
# ============================================================================

class BatchGameEngine:
    """
    高效批量对弈引擎
    - GPU 批量推理模型走法
    - Pikafish 进程池异步处理
    """
    
    def __init__(self, params_np, pikafish_pool: PikafishPool):
        self.params_np = params_np
        self.pikafish_pool = pikafish_pool
        self.games: Dict[int, GameInstance] = {}
        self.stats = PikafishStats()
        self.collected_samples = []
        
        # Pikafish 请求/响应队列
        self.fish_request_queue = queue.Queue()
        self.fish_response_queue = queue.Queue()
        
        # 启动 Pikafish 工作线程
        self.fish_workers = []
        self.stop_event = threading.Event()
        for i in range(pikafish_pool.num_engines):
            t = threading.Thread(target=self._fish_worker, daemon=True)
            t.start()
            self.fish_workers.append(t)
    
    def _fish_worker(self):
        """Pikafish 工作线程：从请求队列取任务，结果放入响应队列"""
        while not self.stop_event.is_set():
            try:
                game_id, fen = self.fish_request_queue.get(timeout=0.1)
            except queue.Empty:
                continue
                
            engine = self.pikafish_pool.acquire(timeout=5.0)
            if engine is None:
                # 放回队列重试
                self.fish_request_queue.put((game_id, fen))
                continue
                
            try:
                uci_move, score = engine.get_best_move(fen)
                self.fish_response_queue.put((game_id, uci_move, score))
            finally:
                self.pikafish_pool.release(engine)
    
    def run_batch(self, num_games: int, rng_key) -> tuple:
        """
        运行一批对弈
        
        Returns:
            (samples, stats)
        """
        self.stats.reset()
        self.collected_samples = []
        self.games.clear()
        
        # 初始化所有游戏
        game_keys = jax.random.split(rng_key, num_games)
        for i in range(num_games):
            state = env.init(game_keys[i])
            model_plays_red = (i % 2 == 0) if config.pikafish_alternate_color else True
            
            game = GameInstance(
                game_id=i,
                state=state,
                model_plays_red=model_plays_red,
                rng_key=game_keys[i],
            )
            
            # 确定第一步谁走
            current_player = int(state.current_player)
            is_model_turn = (current_player == 0) == model_plays_red
            game.waiting_for = "model" if is_model_turn else "pikafish"
            
            if game.waiting_for == "pikafish":
                # 提交 Pikafish 请求
                fen = board_to_fen(np.array(state.board), current_player)
                game.pending_fen = fen
                self.fish_request_queue.put((i, fen))
            
            self.games[i] = game
        
        # 主循环：直到所有游戏结束
        while any(not g.finished for g in self.games.values()):
            # 1. 处理所有等待模型的游戏（GPU 批量推理）
            self._process_model_turns()
            
            # 2. 处理 Pikafish 响应
            self._process_fish_responses()
            
            # 小睡避免忙等
            time.sleep(0.001)
        
        # 收集样本
        return self._collect_samples(), self.stats
    
    def _process_model_turns(self):
        """批量处理所有等待模型的游戏"""
        # 收集所有等待模型的游戏
        model_games = [g for g in self.games.values() 
                       if not g.finished and g.waiting_for == "model"]
        
        if not model_games:
            return
        
        # 批量获取观察
        obs_list = []
        for game in model_games:
            obs = env.observe(game.state)
            obs_list.append(np.array(obs))
        
        obs_batch = np.stack(obs_list, axis=0)
        
        # GPU 批量推理
        logits_batch, values_batch = batch_forward(self.params_np, obs_batch)
        logits_batch = np.array(logits_batch)
        
        # 处理每个游戏
        for i, game in enumerate(model_games):
            current_player = int(game.state.current_player)
            logits = logits_batch[i]
            
            # 视角归一化
            if current_player == 1:
                logits = logits[_ROTATED_IDX_NP]
            
            # 温度采样
            temp = config.temperature_initial if game.step < config.temperature_steps else config.temperature_final
            logits = logits - np.max(logits)
            legal_mask = np.array(game.state.legal_action_mask)
            logits = np.where(legal_mask, logits, -1e10)
            
            exp_logits = np.exp(logits / max(temp, 1e-3))
            probs = exp_logits / (np.sum(exp_logits) + 1e-10)
            
            game.rng_key, subkey = jax.random.split(game.rng_key)
            action = int(jax.random.choice(subkey, ACTION_SPACE_SIZE, p=probs))
            
            # 记录样本
            game.obs_list.append(obs_list[i].astype(np.uint8))
            policy = np.zeros(ACTION_SPACE_SIZE, dtype=np.float32)
            policy[action] = 1.0
            if current_player == 1:
                policy = policy[_ROTATED_IDX_NP]
            game.policy_list.append(policy)
            
            # 执行动作
            game.state = env.step(game.state, action)
            game.step += 1
            
            # 检查游戏是否结束
            if self._check_game_end(game):
                continue
            
            # 切换到等待 Pikafish
            next_player = int(game.state.current_player)
            is_model_turn = (next_player == 0) == game.model_plays_red
            
            if is_model_turn:
                game.waiting_for = "model"
            else:
                game.waiting_for = "pikafish"
                fen = board_to_fen(np.array(game.state.board), next_player)
                game.pending_fen = fen
                self.fish_request_queue.put((game.game_id, fen))
    
    def _process_fish_responses(self):
        """处理 Pikafish 响应"""
        while True:
            try:
                game_id, uci_move, score = self.fish_response_queue.get_nowait()
            except queue.Empty:
                break
            
            game = self.games.get(game_id)
            if game is None or game.finished:
                continue
            
            current_player = int(game.state.current_player)
            game.last_score = score if current_player == 0 else -score
            
            if uci_move is None:
                # Pikafish 无合法走法
                self._finish_game(game)
                continue
            
            # 转换走法
            try:
                from_sq, to_sq = uci_to_move(uci_move)
                action = int(move_to_action(jnp.int32(from_sq), jnp.int32(to_sq)))
            except:
                self._finish_game(game)
                continue
            
            if action < 0 or not game.state.legal_action_mask[action]:
                self._finish_game(game)
                continue
            
            # 执行动作
            game.state = env.step(game.state, action)
            game.step += 1
            
            # 检查评分差提前结束
            if (game.step % config.pikafish_resign_check_interval == 0 and
                abs(game.last_score) >= config.pikafish_resign_threshold):
                game.resigned = True
                self._finish_game(game)
                continue
            
            # 检查游戏是否结束
            if self._check_game_end(game):
                continue
            
            # 切换到等待模型
            next_player = int(game.state.current_player)
            is_model_turn = (next_player == 0) == game.model_plays_red
            
            if is_model_turn:
                game.waiting_for = "model"
            else:
                game.waiting_for = "pikafish"
                fen = board_to_fen(np.array(game.state.board), next_player)
                game.pending_fen = fen
                self.fish_request_queue.put((game.game_id, fen))
    
    def _check_game_end(self, game: GameInstance) -> bool:
        """检查游戏是否结束"""
        if game.state.terminated or game.step >= config.max_steps:
            self._finish_game(game)
            return True
        return False
    
    def _finish_game(self, game: GameInstance):
        """结束游戏，确定胜负"""
        game.finished = True
        
        if game.state.terminated:
            winner = int(game.state.winner)
        elif game.resigned:
            winner = 0 if game.last_score > 0 else 1
        else:
            winner = -1
        
        game.model_won = (winner == 0) == game.model_plays_red if winner != -1 else False
        game.pikafish_won = (winner == 0) != game.model_plays_red if winner != -1 else False
        
        self.stats.add_game(game.model_won, game.pikafish_won, game.step, game.resigned)
    
    def _collect_samples(self) -> Optional[Sample]:
        """收集所有游戏的样本"""
        all_obs = []
        all_policy = []
        all_values = []
        
        for game in self.games.values():
            if len(game.obs_list) == 0:
                continue
            
            # MC 目标
            if game.model_won:
                value = 1.0
            elif game.pikafish_won:
                value = -1.0
            else:
                value = 0.0
            
            all_obs.extend(game.obs_list)
            all_policy.extend(game.policy_list)
            all_values.extend([value] * len(game.obs_list))
        
        if not all_obs:
            return None
        
        return Sample(
            obs=jnp.array(np.array(all_obs, dtype=np.uint8)),
            policy_tgt=jnp.array(np.array(all_policy, dtype=np.float32)),
            value_tgt=jnp.array(np.array(all_values, dtype=np.float32)),
            mask=jnp.ones(len(all_obs), dtype=jnp.bool_),
        )
    
    def shutdown(self):
        """关闭工作线程"""
        self.stop_event.set()
        for t in self.fish_workers:
            t.join(timeout=1.0)

# ============================================================================
# 训练
# ============================================================================

def loss_fn(params, samples: Sample, rng_key):
    obs = samples.obs.astype(jnp.float32)
    policy_tgt = samples.policy_tgt
    
    do_mirror = jax.random.bernoulli(rng_key, 0.5)
    obs = jnp.where(do_mirror, jax.vmap(mirror_observation)(obs), obs)
    policy_tgt = jnp.where(do_mirror, jax.vmap(mirror_policy)(policy_tgt), policy_tgt)
    
    logits, value = forward(params, obs, is_training=True)
    
    logits = logits.astype(jnp.float32)
    value = value.astype(jnp.float32)
    policy_tgt = policy_tgt.astype(jnp.float32)
    value_tgt = samples.value_tgt.astype(jnp.float32)
    
    policy_loss = jnp.sum(optax.softmax_cross_entropy(logits, policy_tgt) * samples.mask) / jnp.maximum(jnp.sum(samples.mask), 1.0)
    value_loss = jnp.sum(optax.l2_loss(value, value_tgt) * samples.mask) / jnp.maximum(jnp.sum(samples.mask), 1.0)
    
    total_loss = policy_loss + config.value_loss_weight * value_loss
    return total_loss, (policy_loss, value_loss)

# ============================================================================
# 经验回放
# ============================================================================

class ReplayBuffer:
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
# Checkpoint
# ============================================================================

class TrainState(NamedTuple):
    params: dict
    opt_state: dict
    iteration: int
    frames: int
    rng_key: jnp.ndarray
    pikafish_depth: int
    winrate_history: list

def create_checkpoint_manager(ckpt_dir: str):
    ckpt_dir = os.path.abspath(ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    options = ocp.CheckpointManagerOptions(
        max_to_keep=config.max_to_keep,
        keep_period=config.keep_period,
        save_interval_steps=1,
    )
    return ocp.CheckpointManager(directory=ckpt_dir, options=options)

def save_checkpoint(ckpt_manager, train_state: TrainState, step: int):
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
    
    meta_dir = os.path.join(os.path.abspath(config.ckpt_dir), f"meta_{step}")
    os.makedirs(meta_dir, exist_ok=True)
    with open(os.path.join(meta_dir, "winrate_history.json"), "w") as f:
        json.dump(train_state.winrate_history, f)
    
    print(f"[Checkpoint] 已保存 step={step}, depth={train_state.pikafish_depth}")

def restore_checkpoint(ckpt_manager, params_template, opt_state_template):
    latest_step = ckpt_manager.latest_step()
    if latest_step is None:
        print("[Checkpoint] 未找到 checkpoint，从头开始")
        return None
    
    print(f"[Checkpoint] 恢复 step={latest_step}...")
    
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
    
    winrate_history = []
    meta_dir = os.path.join(os.path.abspath(config.ckpt_dir), f"meta_{latest_step}")
    history_file = os.path.join(meta_dir, "winrate_history.json")
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            winrate_history = json.load(f)
    
    print(f"[Checkpoint] 恢复完成: iteration={iteration}, depth={pikafish_depth}")
    return params, opt_state, iteration, frames, rng_key, pikafish_depth, winrate_history

def adaptive_depth_update(current_depth: int, winrate_history: list) -> int:
    window = config.pikafish_winrate_window
    if len(winrate_history) < window:
        return current_depth
        
    recent_winrate = sum(winrate_history[-window:]) / window
    
    new_depth = current_depth
    if recent_winrate >= config.pikafish_depth_up_winrate:
        new_depth = min(current_depth + 1, config.pikafish_max_depth)
        print(f"[深度调整] 胜率 {recent_winrate:.1%} >= {config.pikafish_depth_up_winrate:.0%}, 深度 {current_depth} → {new_depth}")
    elif recent_winrate <= config.pikafish_depth_down_winrate:
        new_depth = max(current_depth - 1, config.pikafish_min_depth)
        print(f"[深度调整] 胜率 {recent_winrate:.1%} <= {config.pikafish_depth_down_winrate:.0%}, 深度 {current_depth} → {new_depth}")
              
    return new_depth

# ============================================================================
# 主循环
# ============================================================================

def main():
    print("=" * 70)
    print("ZeroForge - Pikafish 对弈训练（高效批量架构）")
    print("=" * 70)
    print(f"Pikafish 路径: {config.pikafish_path}")
    print(f"进程池大小: {config.pikafish_num_engines}")
    print(f"并发游戏数: {config.pikafish_concurrent_games}")
    print(f"初始深度: {config.pikafish_initial_depth}")
    print(f"深度范围: [{config.pikafish_min_depth}, {config.pikafish_max_depth}]")
    print(f"GPU 数量: {num_devices}")
    print("=" * 70)
    
    # 启动 Pikafish 进程池
    pikafish_pool = PikafishPool(config.pikafish_path, config.pikafish_num_engines)
    success_count = pikafish_pool.start()
    
    if success_count == 0:
        print("[错误] 无法启动 Pikafish 引擎！")
        return
    
    os.makedirs(config.ckpt_dir, exist_ok=True)
    
    # 初始化
    replay_buffer = ReplayBuffer(
        max_size=config.replay_buffer_size,
        obs_shape=(NUM_OBSERVATION_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH),
        action_size=ACTION_SPACE_SIZE
    )
    
    rng_key = jax.random.PRNGKey(config.seed)
    rng_key, subkey = jax.random.split(rng_key)
    dummy_obs = jnp.zeros((1, NUM_OBSERVATION_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH))
    variables = net.init(subkey, dummy_obs, train=True)
    params_template = variables['params']
    
    optimizer = optax.adamw(config.learning_rate, weight_decay=config.weight_decay)
    opt_state_template = optimizer.init(params_template)
    
    ckpt_manager = create_checkpoint_manager(config.ckpt_dir)
    restored = restore_checkpoint(ckpt_manager, params_template, opt_state_template)
    
    if restored is not None:
        params, opt_state, iteration, frames, rng_key, current_depth, winrate_history = restored
    else:
        params = params_template
        opt_state = opt_state_template
        iteration = 0
        frames = 0
        current_depth = config.pikafish_initial_depth
        winrate_history = []
    
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
    
    # 获取 CPU 参数用于推理
    params_np = jax.device_get(jax.tree.map(lambda x: x[0], params))
    
    # 创建批量对弈引擎
    game_engine = BatchGameEngine(params_np, pikafish_pool)
    
    print(f"\n开始训练！每轮 {config.pikafish_concurrent_games} 局并发对弈\n")
    
    try:
        while True:
            iteration += 1
            st = time.time()
            
            # 更新推理参数
            params_np = jax.device_get(jax.tree.map(lambda x: x[0], params))
            game_engine.params_np = params_np
            
            # 批量对弈
            rng_key, sk = jax.random.split(rng_key)
            samples, stats = game_engine.run_batch(config.pikafish_concurrent_games, sk)
            
            if samples is None:
                print(f"[警告] 迭代 {iteration} 无样本")
                continue
            
            # 更新胜率和深度
            winrate = stats.get_winrate()
            winrate_history.append(winrate)
            
            new_depth = adaptive_depth_update(current_depth, winrate_history)
            if new_depth != current_depth:
                current_depth = new_depth
                pikafish_pool.set_depth(current_depth)
            
            # 添加样本
            replay_buffer.add(samples)
            new_frames = len(samples.obs)
            frames += new_frames
            
            # 训练
            num_updates = max(1, min((new_frames * config.sample_reuse_times) // config.training_batch_size, 20))
            
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
            
            if actual_updates > 0:
                avg_ploss = float(ploss_acc.mean() / actual_updates)
                avg_vloss = float(vloss_acc.mean() / actual_updates)
            else:
                avg_ploss, avg_vloss = 0.0, 0.0
            
            iter_time = time.time() - st
            fps = new_frames / iter_time
            buf_stats = replay_buffer.stats()
            avg_len = stats.get_avg_length()
            
            print(f"iter={iteration:4d} | depth={current_depth:2d} | "
                  f"胜率={winrate:.1%} (模型{stats.model_wins} 鱼{stats.pikafish_wins} 和{stats.draws}) | "
                  f"len={avg_len:.1f} resign={stats.resign_count} | "
                  f"ploss={avg_ploss:.4f} vloss={avg_vloss:.4f} | "
                  f"fps={fps:.0f} buf={buf_stats['size']//1000}k")
            
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
        game_engine.shutdown()
        pikafish_pool.shutdown()
        print("[完成] 已关闭")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ZeroForge Pikafish 对弈训练")
    parser.add_argument("--pikafish_path", type=str, default="./pikafish")
    parser.add_argument("--engines", type=int, default=64, help="Pikafish 进程池大小")
    parser.add_argument("--games", type=int, default=256, help="并发游戏数")
    parser.add_argument("--depth", type=int, default=1, help="初始深度")
    parser.add_argument("--resign", type=int, default=350, help="提前结束阈值")
    
    args = parser.parse_args()
    
    config.pikafish_path = args.pikafish_path
    config.pikafish_num_engines = args.engines
    config.pikafish_concurrent_games = args.games
    config.pikafish_initial_depth = args.depth
    config.pikafish_resign_threshold = args.resign
    
    main()

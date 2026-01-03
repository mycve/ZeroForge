"""
Trainer - 周期训练器

训练流程（周期模式）:
┌─────────────────────────────────────────────────────────────────────────┐
│                           每个 Epoch                                    │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │               Phase 1: 自玩阶段（推理模式）                        │   │
│  │  - 加载推理模型到 GPU                                             │   │
│  │  - 并发运行 concurrency 个游戏                                    │   │
│  │  - 完成一个后自动启动下一个，直到 num_envs 个游戏全部完成          │   │
│  │  - 收集轨迹数据作为"新数据"                                       │   │
│  │  - 自玩完成后释放推理资源                                         │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              ↓                                          │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │               Phase 2: 训练阶段（训练模式）                        │   │
│  │  - 加载训练模型到 GPU                                             │   │
│  │  - 采样 80% 新数据 + 20% 经验池数据                               │   │
│  │  - 执行 train_batches_per_epoch 批次训练                          │   │
│  │  - 新数据存入经验池                                               │   │
│  │  - 训练完成后释放训练资源（可选）                                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              ↓                                          │
│                        保存检查点（可选）                               │
└─────────────────────────────────────────────────────────────────────────┘

优点:
- 内存管理清晰：训练和推理分离，避免同时占用大量显存
- 数据质量高：每轮训练使用最新策略生成的数据
- 可配置性强：可灵活调整游戏数量、并发数、新旧数据比例

============================================================
性能优化（自玩阶段）
============================================================

默认使用 SlotBatcher（无锁预分配槽位）替代 LeafBatcher：

【SlotBatcher 架构】
┌─────────────────────────────────────────────────────────────────────────┐
│   线程0 ── Slot[0] ─┐                                                   │
│   线程1 ── Slot[1] ─┼─► SlotBatcher(扫描) ─► GPU 推理 ─► 分发结果       │
│   线程N ── Slot[N] ─┘                                                   │
└─────────────────────────────────────────────────────────────────────────┘

特点:
- 无锁写入：每个线程直接写入自己的槽位
- 无锁扫描：Batcher 线程扫描所有槽位收集请求
- 动态槽位：游戏结束时槽位回收复用
- 适配 no-GIL：Python 3.13+ free-threaded 高并发

配置:
- use_slot_batcher: True（默认）使用无锁 SlotBatcher
                    False 回退到传统 LeafBatcher
- concurrency: 并发数 = SlotBatcher 槽位数

详细说明见: core/PERFORMANCE.md
"""

import os
import time
import logging
import threading
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable, Tuple
from pathlib import Path
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data import DataLoader, Dataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .training_config import TrainingConfig, resolve_device
from .replay_buffer import ReplayBuffer, SampleBatch
from .checkpoint import CheckpointManager
from .algorithm import Trajectory
from .config import MCTSConfig, BatcherConfig
from .mcts import LeafBatcher, LocalMCTSTree

logger = logging.getLogger(__name__)


def _to_python(obj):
    """将 numpy 类型转换为 Python 原生类型，用于 JSON 序列化"""
    import numpy as np
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    return obj


# ============================================================
# 训练状态
# ============================================================

@dataclass
class TrainerState:
    """训练器状态"""
    running: bool = False
    paused: bool = False
    epoch: int = 0
    step: int = 0
    total_games: int = 0
    
    # 损失
    loss: float = 0.0
    value_loss: float = 0.0
    policy_loss: float = 0.0
    reward_loss: float = 0.0  # MuZero 奖励预测损失
    
    # 自玩统计
    avg_game_length: float = 0.0
    games_per_second: float = 0.0
    steps_per_second: float = 0.0
    
    # 评估统计（新版本 vs 旧版本）
    eval_win_rate: float = 0.0      # 新版本胜率
    eval_games: int = 0             # 评估对弈局数
    elo_rating: float = 1500.0      # ELO 评分
    
    # 时间
    start_time: Optional[float] = None
    elapsed_time: float = 0.0
    
    # DDP
    rank: int = 0
    world_size: int = 1


# ============================================================
# 周期训练器
# ============================================================

class DistributedTrainer:
    """周期训练器
    
    训练流程：
    1. 自玩阶段：并发运行游戏，收集轨迹
    2. 训练阶段：80%新数据 + 20%经验池，更新网络
    3. 保存检查点
    
    内存管理：
    - 自玩时只保留推理模型
    - 训练时释放推理资源
    
    Example:
        >>> trainer = DistributedTrainer(config)
        >>> trainer.setup()
        >>> trainer.run()
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.state = TrainerState()
        
        # DDP 设置
        self._is_distributed = False
        self._local_rank = 0
        self._world_size = 1
        self._device = None
        
        # 组件（延迟初始化，按阶段加载）
        self._network = None
        self._optimizer = None
        self._batcher = None
        self._replay_buffer = None
        self._checkpoint_manager = None
        
        # 游戏工厂
        self._game_factory = None
        self._mcts_config = None
        
        # 评估：保存上一个 epoch 的网络权重
        self._prev_network_state = None
        
        # 控制
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._callbacks: List[Callable] = []
    
    # ========================================
    # 初始化
    # ========================================
    
    def setup(self) -> None:
        """初始化训练器（只初始化基础组件，网络延迟加载）"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch 未安装")
        
        # 检测分布式环境
        self._setup_distributed()
        
        # 创建网络（初始在 CPU）
        self._setup_network()
        
        # 创建优化器
        self._setup_optimizer()
        
        # 创建 ReplayBuffer
        self._setup_buffer()
        
        # 创建检查点管理器
        self._checkpoint_manager = CheckpointManager(
            base_dir=self.config.checkpoint_dir,
            keep_checkpoints=self.config.keep_checkpoints,
        )
        
        # 准备自玩配置（不在这里创建 workers）
        self._setup_selfplay_config()
        
        logger.info(f"训练器初始化完成: rank={self._local_rank}, world_size={self._world_size}, "
                   f"device={self._device}, num_envs={self.config.num_envs}, "
                   f"concurrency={self.config.concurrency}")
    
    def _setup_distributed(self) -> None:
        """设置分布式环境
        
        支持两种 DDP 启动方式：
        1. torchrun 启动：自动设置环境变量
        2. 手动配置：use_ddp=True，自动检测可用 GPU
        """
        # 方式 1：通过 torchrun 启动（环境变量已设置）
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            self._local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self._world_size = int(os.environ["WORLD_SIZE"])
            self._is_distributed = self._world_size > 1
            
            if self._is_distributed:
                dist.init_process_group(backend=self.config.ddp_backend)
                torch.cuda.set_device(self._local_rank)
                self._device = torch.device(f"cuda:{self._local_rank}")
                logger.info(f"DDP 初始化 (torchrun): rank={self._local_rank}, "
                           f"world_size={self._world_size}, backend={self.config.ddp_backend}")
            else:
                self._device = torch.device(resolve_device(self.config.device))
        
        # 方式 2：手动配置 use_ddp=True（单机多卡）
        elif self.config.use_ddp and torch.cuda.is_available():
            num_gpus = torch.cuda.device_count()
            if num_gpus > 1:
                # 单机多卡：主进程初始化
                logger.warning(f"检测到 {num_gpus} 个 GPU，但单进程无法使用 DDP。")
                logger.warning(f"请使用 torchrun 启动: torchrun --nproc_per_node={num_gpus} main.py train")
                logger.warning(f"当前将使用单卡训练 (GPU:0)")
            
            self._device = torch.device("cuda:0")
            self._local_rank = 0
            self._world_size = 1
            self._is_distributed = False
        
        else:
            # 单卡模式
            self._device = torch.device(resolve_device(self.config.device))
            self._local_rank = 0
            self._world_size = 1
            self._is_distributed = False
        
        self.state.rank = self._local_rank
        self.state.world_size = self._world_size
    
    def _setup_network(self) -> None:
        """创建网络"""
        from games import make_game
        from algorithms import make_algorithm
        
        # 创建游戏获取空间信息
        game = make_game(self.config.game_type)
        game.reset()
        
        # 创建算法和网络
        algo = make_algorithm(self.config.algorithm)
        self._network = algo.create_network(game).to(self._device)
        
        # DDP 包装
        if self._is_distributed:
            self._network = DDP(
                self._network,
                device_ids=[self._local_rank],
                output_device=self._local_rank,
            )
        
        param_count = sum(p.numel() for p in self._network.parameters())
        logger.info(f"网络参数量: {param_count:,}")
    
    def _setup_optimizer(self) -> None:
        """创建优化器"""
        self._optimizer = torch.optim.AdamW(
            self._network.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
    
    def _setup_buffer(self) -> None:
        """创建回放缓冲区"""
        from games import make_game
        
        game = make_game(self.config.game_type)
        action_space = game.action_space.n
        
        self._replay_buffer = ReplayBuffer(
            capacity=self.config.replay_buffer_size,
            action_space_size=action_space,
        )
    
    def _setup_selfplay_config(self) -> None:
        """准备自玩配置（不创建 workers，延迟到自玩阶段）"""
        from games import make_game
        
        # MCTS 配置
        self._mcts_config = MCTSConfig(
            num_simulations=self.config.num_simulations,
            c_puct=self.config.c_puct,
            dirichlet_alpha=self.config.dirichlet_alpha,
            dirichlet_epsilon=self.config.dirichlet_epsilon,
            temperature_threshold=self.config.temperature_threshold,
        )
        
        # 游戏工厂
        game_type = self.config.game_type
        def game_factory():
            return make_game(game_type)
        self._game_factory = game_factory
        
        logger.info(f"自玩配置准备完成: num_envs={self.config.num_envs}, "
                   f"concurrency={self.config.concurrency}, "
                   f"inference_batch_size={self.config.inference_batch_size}")
    
    # ========================================
    # 训练循环（周期模式）
    # ========================================
    
    def run(self) -> None:
        """运行训练循环（周期模式）
        
        每个 epoch:
        1. 自玩阶段：运行 num_envs 局游戏，并发数为 concurrency
        2. 训练阶段：80% 新数据 + 20% 经验池
        """
        self.state.running = True
        self.state.start_time = time.time()
        self._stop_event.clear()
        
        try:
            for epoch in range(self.config.num_epochs):
                if self._stop_event.is_set():
                    break
                
                # 暂停检查
                while self.state.paused and not self._stop_event.is_set():
                    time.sleep(0.1)
                
                if self._stop_event.is_set():
                    break
                
                epoch_start = time.time()
                
                # ========== Phase 1: 自玩阶段 ==========
                logger.info(f"[Epoch {epoch + 1}] 开始自玩阶段: {self.config.num_envs} 局游戏, 并发 {self.config.concurrency}")
                new_trajectories = self._run_selfplay_phase()
                
                if self._stop_event.is_set():
                    break
                
                selfplay_time = time.time() - epoch_start
                logger.info(f"[Epoch {epoch + 1}] 自玩完成: {len(new_trajectories)} 局, 耗时 {selfplay_time:.1f}s")
                
                # ========== Phase 2: 训练阶段 ==========
                train_start = time.time()
                train_metrics = self._run_training_phase(new_trajectories)
                train_time = time.time() - train_start
                
                # ========== Phase 3: 评估阶段（新版本 vs 旧版本）==========
                eval_metrics = {}
                if self.config.eval_games > 0 and self._prev_network_state is not None:
                    eval_start = time.time()
                    eval_metrics = self._run_eval_phase()
                    eval_time = time.time() - eval_start
                    logger.info(f"[Epoch {epoch + 1}] 评估完成: 胜率={eval_metrics.get('win_rate', 0):.1%}, "
                               f"ELO={eval_metrics.get('elo', 1500):.0f}, 耗时 {eval_time:.1f}s")
                
                # 保存当前网络状态作为下一轮的"旧版本"
                base_network = self._network.module if self._is_distributed else self._network
                self._prev_network_state = {k: v.cpu().clone() for k, v in base_network.state_dict().items()}
                
                # 更新状态
                self._update_state(epoch + 1, train_metrics, eval_metrics)
                
                # 保存检查点
                if (epoch + 1) % self.config.save_interval == 0:
                    self._save_checkpoint(epoch + 1)
                
                # 通知回调
                self._notify_callbacks()
                
                # 日志
                epoch_time = time.time() - epoch_start
                eval_info = f", 胜率:{self.state.eval_win_rate:.1%}" if eval_metrics else ""
                logger.info(
                    f"[Epoch {epoch + 1}/{self.config.num_epochs}] "
                    f"loss={train_metrics.get('total_loss', 0):.4f}, "
                    f"games={self.state.total_games}, "
                    f"buffer={len(self._replay_buffer)}{eval_info}, "
                    f"自玩:{selfplay_time:.1f}s 训练:{train_time:.1f}s 总计:{epoch_time:.1f}s"
                )
            
            # 保存最终检查点
            self._save_checkpoint(self.config.num_epochs, is_final=True)
            
        except Exception as e:
            logger.error(f"训练出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._cleanup()
    
    def _run_selfplay_phase(self) -> List[Trajectory]:
        """自玩阶段：运行 num_envs 局游戏，并发数为 concurrency
        
        Returns:
            新生成的轨迹列表
        """
        import gc
        from .mcts import SlotBatcher
        
        # 确保网络在正确设备上（推理模式）
        base_network = self._network.module if self._is_distributed else self._network
        base_network.to(self._device)
        base_network.eval()
        
        # 创建 Batcher 配置
        batcher_config = BatcherConfig(
            batch_size=self.config.inference_batch_size,
            timeout_ms=self.config.inference_timeout_ms,
            device=str(self._device),
        )
        
        trajectories: List[Trajectory] = []
        completed_games = 0
        total_games = self.config.num_envs
        concurrency = self.config.concurrency
        
        # 【优化】使用无锁 SlotBatcher（预分配槽位）
        use_slot_batcher = getattr(self.config, 'use_slot_batcher', True)
        
        if use_slot_batcher:
            batcher = SlotBatcher(base_network, batcher_config, num_slots=concurrency)
        else:
            batcher = LeafBatcher(base_network, batcher_config)
        batcher.start()
        
        # 线程安全的结果收集
        results_lock = threading.Lock()
        active_workers: List[threading.Thread] = []
        
        # 检查是否使用 Gumbel 搜索
        use_gumbel = self.config.algorithm in ("gumbel_alphazero", "gumbel_muzero")
        
        def run_one_game(worker_id: int, slot_id: int):
            """运行一局游戏
            
            Args:
                worker_id: 工作线程 ID（用于统计）
                slot_id: 槽位 ID（用于 SlotBatcher，循环复用）
            """
            nonlocal completed_games
            
            try:
                # 注册槽位（SlotBatcher）
                if use_slot_batcher:
                    batcher.register_slot(slot_id)
                
                game = self._game_factory()
                game.reset()
                
                trajectory = Trajectory()
                move_count = 0
                
                def evaluate_fn(obs: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, float]:
                    if use_slot_batcher:
                        return batcher.submit(slot_id, obs, mask)
                    else:
                        return batcher.submit(obs, mask, env_id=worker_id)
                
                # 根据算法类型选择搜索策略
                if use_gumbel:
                    from core.mcts import GumbelMCTSSearch
                    search = GumbelMCTSSearch(
                        game, 
                        self._mcts_config,
                        batcher=batcher,
                        mode="alphazero",
                        max_considered_actions=self.config.gumbel_max_actions,
                        gumbel_scale=self.config.gumbel_scale,
                    )
                else:
                    mcts_tree = LocalMCTSTree(game, self._mcts_config, mode="alphazero")
                
                while not game.is_terminal() and not self._stop_event.is_set():
                    current_player = game.current_player()
                    obs = game.get_observation()
                    
                    if use_gumbel:
                        # Gumbel MCTS 搜索（使用 Top-k + Sequential Halving）
                        action, policy_dict, root_value = search.run(
                            evaluate_fn=evaluate_fn,
                            num_simulations=self._mcts_config.num_simulations,
                        )
                    else:
                        # 标准 MCTS 搜索
                        add_noise = (move_count == 0)
                        action, policy_dict, root_value = mcts_tree.search(
                            evaluate_fn=evaluate_fn,
                            num_simulations=self._mcts_config.num_simulations,
                            add_noise=add_noise,
                        )
                    
                    trajectory.append(
                        observation=obs,
                        action=action,
                        reward=0.0,
                        policy=policy_dict,
                        value=root_value,
                        to_play=current_player,
                    )
                    
                    game.step(action)
                    move_count += 1
                    
                    if use_gumbel:
                        search.game = game
                        search.advance(action)
                    else:
                        mcts_tree.game = game
                        mcts_tree.advance(action)
                    
                    if move_count > 500:
                        logger.warning(f"Worker {worker_id}: 游戏超过 500 步，强制结束")
                        break
                
                # 设置奖励（MuZero 需要即时奖励，AlphaZero 需要终局奖励）
                winner = game.get_winner()
                num_steps = len(trajectory.to_play)
                
                # 中间步骤奖励为 0
                for i in range(num_steps - 1):
                    trajectory.rewards[i] = 0.0
                
                # 终局步骤奖励为游戏结果（从该步玩家视角）
                if num_steps > 0:
                    last_player = trajectory.to_play[-1]
                    if winner is None:
                        trajectory.rewards[-1] = 0.0  # 和棋
                    elif last_player == winner:
                        trajectory.rewards[-1] = 1.0  # 该玩家赢了
                    else:
                        trajectory.rewards[-1] = -1.0  # 该玩家输了
                
                with results_lock:
                    trajectories.append(trajectory)
                    completed_games += 1
                    
            except Exception as e:
                logger.error(f"Worker {worker_id} 自玩出错: {e}")
                import traceback
                traceback.print_exc()
            finally:
                # 注销槽位（SlotBatcher）
                if use_slot_batcher:
                    batcher.unregister_slot(slot_id)
        
        # 启动初始并发
        next_game_id = 0
        last_completed = 0
        last_update_time = time.time()
        phase_start_time = time.time()
        
        # 槽位分配：使用 worker_id % concurrency 复用槽位
        slot_assignments: Dict[int, int] = {}  # worker_id -> slot_id
        available_slots = list(range(concurrency))  # 可用槽位列表
        
        while completed_games < total_games and not self._stop_event.is_set():
            # 清理已完成的线程，回收槽位
            new_active = []
            for t, worker_id in active_workers:
                if t.is_alive():
                    new_active.append((t, worker_id))
                else:
                    # 线程结束，回收槽位
                    if worker_id in slot_assignments:
                        freed_slot = slot_assignments.pop(worker_id)
                        available_slots.append(freed_slot)
            active_workers = new_active
            
            # 启动新的游戏（保持并发数）
            while len(active_workers) < concurrency and next_game_id < total_games and available_slots:
                # 分配槽位
                slot_id = available_slots.pop(0)
                slot_assignments[next_game_id] = slot_id
                
                t = threading.Thread(
                    target=run_one_game, 
                    args=(next_game_id, slot_id), 
                    daemon=True
                )
                t.start()
                active_workers.append((t, next_game_id))
                next_game_id += 1
            
            # 实时更新状态（每 0.5 秒或有新完成时）
            current_time = time.time()
            if completed_games > last_completed or current_time - last_update_time >= 0.5:
                # 计算实时速度（最近一段时间）
                elapsed_since_start = current_time - phase_start_time
                if elapsed_since_start > 0:
                    realtime_speed = completed_games / elapsed_since_start
                else:
                    realtime_speed = 0
                
                with self._lock:
                    self.state.total_games += (completed_games - last_completed)
                    self.state.games_per_second = realtime_speed
                    # 计算预计剩余时间
                    remaining_games = total_games - completed_games
                    if realtime_speed > 0:
                        eta_seconds = remaining_games / realtime_speed
                    else:
                        eta_seconds = 0
                
                # 通知回调更新前端
                self._notify_callbacks()
                
                last_completed = completed_games
                last_update_time = current_time
            
            time.sleep(0.05)
        
        # 等待所有线程完成
        for t, _ in active_workers:
            t.join(timeout=30.0)
        
        # 停止 batcher，释放资源
        batcher.stop()
        
        # 更新统计（注意：total_games 已在循环中实时更新，这里只更新平均长度）
        with self._lock:
            # 确保最终计数正确（处理循环中可能遗漏的最后几局）
            final_diff = len(trajectories) - last_completed
            if final_diff > 0:
                self.state.total_games += final_diff
            
            if trajectories:
                total_length = sum(len(t) for t in trajectories)
                self.state.avg_game_length = total_length / len(trajectories)
        
        # 释放推理资源
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return trajectories
    
    def _run_training_phase(self, new_trajectories: List[Trajectory]) -> Dict[str, float]:
        """训练阶段：80% 新数据 + 20% 经验池
        
        根据算法类型选择训练方式：
        - AlphaZero: 直接前向传播，计算 policy + value loss
        - MuZero: 使用 dynamics unroll，计算 policy + value + reward loss
        
        Args:
            new_trajectories: 本轮自玩生成的新轨迹
            
        Returns:
            训练指标
        """
        from algorithms import make_algorithm
        
        # 确保网络在正确设备上（训练模式）
        base_network = self._network.module if self._is_distributed else self._network
        base_network.to(self._device)
        self._network.train()
        
        # 将新数据加入经验池
        self._replay_buffer.add_batch(new_trajectories)
        
        # 计算新数据量
        new_data_steps = sum(len(t) for t in new_trajectories)
        total_batch_size = self.config.batch_size
        
        # 如果经验池太小，等待更多数据
        if len(self._replay_buffer) < self.config.min_buffer_size:
            logger.debug(f"经验池太小 ({len(self._replay_buffer)}), 等待更多数据")
            return {"total_loss": 0, "value_loss": 0, "policy_loss": 0, "reward_loss": 0}
        
        # 获取算法实例
        algo = make_algorithm(self.config.algorithm)
        is_muzero = algo.needs_dynamics  # MuZero 需要 dynamics 网络
        
        # MuZero 使用展开步数，AlphaZero 不需要
        unroll_steps = getattr(algo.config, 'num_unroll_steps', 0) if is_muzero else 0
        
        total_loss = 0.0
        value_loss_sum = 0.0
        policy_loss_sum = 0.0
        reward_loss_sum = 0.0
        
        # 根据 replay_ratio 自动计算训练批次数
        # 这样每个新样本平均被训练 replay_ratio 次
        num_batches = max(1, int(new_data_steps * self.config.replay_ratio / total_batch_size))
        
        logger.debug(f"训练批次: {num_batches}, 新数据: {new_data_steps}, 批大小: {total_batch_size}")
        
        for batch_idx in range(num_batches):
            if self._stop_event.is_set():
                break
            
            # 从经验池采样
            batch = self._replay_buffer.sample(
                batch_size=total_batch_size,
                unroll_steps=unroll_steps,
            )
            
            if is_muzero:
                # MuZero 训练：使用算法的 compute_loss
                # 转换 batch 为 TrainingTargets 格式
                from algorithms.muzero.algorithm import TrainingTargets
                
                targets = TrainingTargets(
                    observations=torch.from_numpy(batch.observations).to(self._device),
                    actions=torch.from_numpy(batch.actions).to(self._device),
                    target_values=torch.from_numpy(batch.target_values).to(self._device),
                    target_rewards=torch.from_numpy(batch.target_rewards).to(self._device),
                    target_policies=torch.from_numpy(batch.target_policies).to(self._device),
                    masks=torch.from_numpy(batch.masks).to(self._device),
                )
                
                # 使用算法计算损失
                loss, metrics = algo.compute_loss(base_network, targets)
                
                value_loss_sum += metrics.get("value_loss", 0)
                policy_loss_sum += metrics.get("policy_loss", 0)
                reward_loss_sum += metrics.get("reward_loss", 0)
            else:
                # AlphaZero 训练：直接前向传播
                obs = torch.from_numpy(batch.observations).to(self._device)
                target_policy = torch.from_numpy(batch.target_policies[:, 0, :]).to(self._device)
                target_value = torch.from_numpy(batch.target_values[:, 0]).to(self._device)
                
                # 前向传播
                policy_logits, value = base_network(obs)
                
                # 计算损失
                policy_loss = F.cross_entropy(policy_logits, target_policy)
                value_loss = F.mse_loss(value.squeeze(-1), target_value)
                loss = policy_loss + value_loss
                
                value_loss_sum += value_loss.item()
                policy_loss_sum += policy_loss.item()
            
            # 反向传播
            self._optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._network.parameters(), 1.0)
            self._optimizer.step()
            
            total_loss += loss.item()
            
            with self._lock:
                self.state.step += 1
        
        return {
            "total_loss": total_loss / num_batches,
            "value_loss": value_loss_sum / num_batches,
            "policy_loss": policy_loss_sum / num_batches,
            "reward_loss": reward_loss_sum / num_batches,
        }
    
    def _run_eval_phase(self) -> Dict[str, float]:
        """评估阶段：新版本 vs 旧版本对弈
        
        Returns:
            评估指标: win_rate, games, elo
        """
        import copy
        
        if self._prev_network_state is None:
            return {}
        
        # 获取当前网络（新版本）
        new_network = self._network.module if self._is_distributed else self._network
        new_network.to(self._device)
        new_network.eval()
        
        # 创建旧版本网络副本
        from algorithms import make_algorithm
        algo = make_algorithm(self.config.algorithm)
        game_template = self._game_factory()
        old_network = algo.create_network(game_template).to(self._device)
        old_network.load_state_dict(self._prev_network_state)
        old_network.eval()
        
        # 对弈统计
        new_wins = 0
        old_wins = 0
        draws = 0
        
        temperature = self.config.eval_temperature
        num_games = self.config.eval_games
        
        def select_action_with_temperature(policy: np.ndarray, legal_actions: List[int], temp: float) -> int:
            """使用温度采样选择动作"""
            # 提取合法动作的概率
            legal_probs = np.array([policy[a] for a in legal_actions], dtype=np.float64)
            
            # 处理全零或负数情况
            legal_probs = np.maximum(legal_probs, 0)
            prob_sum = legal_probs.sum()
            
            if prob_sum < 1e-10:
                # 概率太小，均匀分布
                return np.random.choice(legal_actions)
            
            if temp < 0.01:
                # 温度接近0，选择最大概率
                return legal_actions[np.argmax(legal_probs)]
            
            # 应用温度
            legal_probs = np.power(legal_probs, 1.0 / temp)
            
            # 归一化（确保严格等于 1）
            prob_sum = legal_probs.sum()
            if prob_sum < 1e-10:
                return np.random.choice(legal_actions)
            legal_probs = legal_probs / prob_sum
            
            # 修正浮点误差，确保和为 1
            legal_probs[-1] = 1.0 - legal_probs[:-1].sum()
            legal_probs = np.clip(legal_probs, 0, 1)
            
            # 采样
            return np.random.choice(legal_actions, p=legal_probs)
        
        def get_network_action(network, game, temp: float) -> int:
            """使用网络选择动作"""
            obs = game.get_observation()
            legal_actions = game.legal_actions()
            
            with torch.no_grad():
                obs_t = torch.from_numpy(obs).unsqueeze(0).float().to(self._device)
                policy_logits, _ = network(obs_t)
                policy = torch.softmax(policy_logits, dim=-1)[0].cpu().numpy()
            
            return select_action_with_temperature(policy, legal_actions, temp)
        
        # 对弈（新版本先手一半，后手一半）
        for game_idx in range(num_games):
            if self._stop_event.is_set():
                break
            
            game = self._game_factory()
            game.reset()
            
            # 交替先后手
            new_is_player0 = (game_idx % 2 == 0)
            
            while not game.is_terminal():
                current_player = game.current_player()
                
                # 判断当前是新版本还是旧版本
                if (current_player == 0) == new_is_player0:
                    action = get_network_action(new_network, game, temperature)
                else:
                    action = get_network_action(old_network, game, temperature)
                
                game.step(action)
            
            # 统计结果
            winner = game.get_winner()
            if winner is None:
                draws += 1
            elif (winner == 0) == new_is_player0:
                new_wins += 1
            else:
                old_wins += 1
        
        # 计算胜率
        total_games = new_wins + old_wins + draws
        win_rate = new_wins / total_games if total_games > 0 else 0.5
        
        # 更新 ELO（简化版）
        # 预期胜率基于当前 ELO 差（新版本默认比旧版本高）
        expected_win_rate = 0.5  # 假设新旧版本 ELO 相同
        k_factor = 32  # ELO K 因子
        elo_change = k_factor * (win_rate - expected_win_rate)
        new_elo = self.state.elo_rating + elo_change
        
        logger.debug(f"评估结果: 新版本胜{new_wins} 旧版本胜{old_wins} 和{draws}, "
                    f"胜率={win_rate:.1%}, ELO变化={elo_change:+.1f}")
        
        # 清理旧网络
        del old_network
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return {
            "win_rate": win_rate,
            "games": total_games,
            "elo": new_elo,
            "new_wins": new_wins,
            "old_wins": old_wins,
            "draws": draws,
        }
    
    def _update_state(self, epoch: int, metrics: Dict[str, float], eval_metrics: Dict[str, float] = None) -> None:
        """更新训练状态"""
        with self._lock:
            self.state.epoch = epoch
            self.state.loss = metrics.get("total_loss", 0)
            self.state.value_loss = metrics.get("value_loss", 0)
            self.state.policy_loss = metrics.get("policy_loss", 0)
            self.state.reward_loss = metrics.get("reward_loss", 0)  # MuZero 奖励损失
            
            # 评估指标
            if eval_metrics:
                self.state.eval_win_rate = eval_metrics.get("win_rate", 0)
                self.state.eval_games = eval_metrics.get("games", 0)
                self.state.elo_rating = eval_metrics.get("elo", self.state.elo_rating)
            
            elapsed = time.time() - self.state.start_time
            self.state.elapsed_time = elapsed
            if elapsed > 0:
                self.state.games_per_second = self.state.total_games / elapsed
                self.state.steps_per_second = self.state.step / elapsed
    
    def _save_checkpoint(self, epoch: int, is_final: bool = False) -> None:
        """保存检查点"""
        if self._local_rank != 0:
            return
        
        if self._checkpoint_manager is None:
            return
        
        # 获取底层网络
        network = self._network.module if self._is_distributed else self._network
        
        metrics = {
            "epoch": epoch,
            "step": self.state.step,
            "loss": self.state.loss,
            "total_games": self.state.total_games,
        }
        
        try:
            path = self._checkpoint_manager.save(
                game_type=self.config.game_type,
                algorithm=self.config.algorithm,
                epoch=epoch,
                network=network,
                optimizer=self._optimizer,
                config=self.config.to_dict(),
                metrics=metrics,
            )
            logger.info(f"保存检查点: {path}")
        except Exception as e:
            logger.error(f"保存检查点失败: {e}")
    
    def _cleanup(self) -> None:
        """清理资源"""
        self.state.running = False
        
        # 清理 GPU 缓存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        if self._is_distributed:
            dist.destroy_process_group()
        
        logger.info("训练资源已清理")
    
    # ========================================
    # 控制接口
    # ========================================
    
    def pause(self) -> None:
        """暂停训练"""
        self.state.paused = True
        logger.info("训练已暂停")
    
    def resume(self) -> None:
        """恢复训练"""
        self.state.paused = False
        logger.info("训练已恢复")
    
    def stop(self) -> None:
        """停止训练"""
        self._stop_event.set()
        logger.info("训练停止中...")
    
    def add_callback(self, callback: Callable[[TrainerState], None]) -> None:
        """添加状态回调"""
        self._callbacks.append(callback)
    
    def _notify_callbacks(self) -> None:
        """通知回调"""
        for callback in self._callbacks:
            try:
                callback(self.state)
            except Exception as e:
                logger.error(f"回调执行失败: {e}")
    
    def get_state(self) -> Dict[str, Any]:
        """获取当前状态"""
        with self._lock:
            # 使用 _to_python 转换 numpy 类型，避免 JSON 序列化错误
            return _to_python({
                "running": self.state.running,
                "paused": self.state.paused,
                "epoch": self.state.epoch,
                "step": self.state.step,
                "total_games": self.state.total_games,
                "loss": self.state.loss,
                "value_loss": self.state.value_loss,
                "policy_loss": self.state.policy_loss,
                "reward_loss": self.state.reward_loss,  # MuZero 奖励损失
                "avg_game_length": self.state.avg_game_length,
                "games_per_second": self.state.games_per_second,
                "steps_per_second": self.state.steps_per_second,
                "elapsed_time": self.state.elapsed_time,
                "rank": self.state.rank,
                "world_size": self.state.world_size,
                "buffer_size": len(self._replay_buffer) if self._replay_buffer else 0,
                # 评估指标
                "eval_win_rate": self.state.eval_win_rate,
                "eval_games": self.state.eval_games,
                "elo_rating": self.state.elo_rating,
            })


# ============================================================
# 便捷函数
# ============================================================

def _ddp_worker(
    rank: int,
    world_size: int,
    config_dict: Dict[str, Any],
    master_port: int,
    callback_queue: Optional[Any] = None,
    command_queue: Optional[Any] = None,
) -> None:
    """DDP 工作进程
    
    Args:
        rank: 进程排名
        world_size: 总进程数
        config_dict: 配置字典
        master_port: 主进程端口
        callback_queue: 用于回调的队列（可选，子进程 -> 主进程）
        command_queue: 用于命令的队列（可选，主进程 -> 子进程）
    """
    import os
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    
    config = TrainingConfig.from_dict(config_dict)
    trainer = DistributedTrainer(config)
    
    # 如果有回调队列，添加状态回调
    if callback_queue is not None and rank == 0:
        def state_callback(state):
            try:
                callback_queue.put_nowait(("state", trainer.get_state()))
            except:
                pass
        trainer.add_callback(state_callback)
    
    # 先 setup
    trainer.setup()
    
    # rank 0 在 setup 之后启动命令监听线程
    if command_queue is not None and rank == 0:
        def command_listener():
            """监听主进程发来的命令"""
            while trainer.state.running:
                try:
                    cmd = command_queue.get(timeout=0.5)
                    if cmd == "save_checkpoint":
                        logger.info("[DDP rank 0] 收到保存检查点命令")
                        trainer._save_checkpoint(trainer.state.epoch, is_final=False)
                        if callback_queue:
                            callback_queue.put_nowait(("checkpoint_saved", trainer.state.epoch))
                    elif cmd == "stop":
                        logger.info("[DDP rank 0] 收到停止命令")
                        trainer.state.running = False
                        break
                except:
                    # 超时或队列空，继续
                    continue
        
        # setup 完成后启动命令监听线程
        cmd_thread = threading.Thread(target=command_listener, daemon=True)
        cmd_thread.start()
    
    trainer.run()
    
    if callback_queue is not None and rank == 0:
        callback_queue.put_nowait(("done", None))


def launch_distributed_training(
    config: TrainingConfig,
    num_gpus: int = 0,
    callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    blocking: bool = True,
) -> Optional[Dict[str, Any]]:
    """启动分布式训练（支持 Web 调用）
    
    使用 torch.multiprocessing.spawn 在代码中启动多进程 DDP 训练，
    不需要 torchrun 命令行。
    
    Args:
        config: 训练配置
        num_gpus: GPU 数量（0=自动检测）
        callback: 状态回调函数（仅 rank 0 调用）
        blocking: 是否阻塞等待训练完成
    
    Returns:
        字典包含:
        - command_queue: 命令队列（用于发送保存检查点等命令）
        - context: 进程上下文（blocking=False 时）
        阻塞模式返回 None
    
    Example:
        >>> config = TrainingConfig(game_type="chinese_chess", use_ddp=True)
        >>> 
        >>> # 阻塞模式（等待训练完成）
        >>> launch_distributed_training(config, num_gpus=4)
        >>> 
        >>> # 非阻塞模式（后台运行）
        >>> result = launch_distributed_training(config, num_gpus=4, blocking=False)
        >>> result["command_queue"].put("save_checkpoint")  # 发送保存命令
        >>> result["context"].join()  # 等待完成
    """
    import torch.multiprocessing as mp
    
    # 自动检测 GPU 数量
    if num_gpus <= 0:
        num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
    
    if num_gpus <= 1:
        # 单卡直接运行
        trainer = DistributedTrainer(config)
        if callback:
            trainer.add_callback(callback)
        trainer.setup()
        trainer.run()
        return None
    
    # 多卡使用 mp.spawn
    logger.info(f"启动 DDP 训练: {num_gpus} 个 GPU")
    
    # 找一个可用端口
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        master_port = s.getsockname()[1]
    
    # 创建回调队列（子进程 -> 主进程）
    callback_queue = mp.Queue() if callback else None
    
    # 创建命令队列（主进程 -> 子进程 rank 0）
    command_queue = mp.Queue()
    
    if callback:
        # 启动回调监听线程
        def callback_listener():
            while True:
                try:
                    msg_type, data = callback_queue.get(timeout=1.0)
                    if msg_type == "done":
                        break
                    elif msg_type == "state":
                        callback(data)
                    elif msg_type == "checkpoint_saved":
                        logger.info(f"[DDP] 检查点已保存: epoch {data}")
                except:
                    continue
        
        listener_thread = threading.Thread(target=callback_listener, daemon=True)
        listener_thread.start()
    
    config_dict = config.to_dict()
    
    if blocking:
        mp.spawn(
            _ddp_worker,
            args=(num_gpus, config_dict, master_port, callback_queue, command_queue),
            nprocs=num_gpus,
            join=True,
        )
        return None
    else:
        # 非阻塞模式
        ctx = mp.spawn(
            _ddp_worker,
            args=(num_gpus, config_dict, master_port, callback_queue, command_queue),
            nprocs=num_gpus,
            join=False,
        )
        return {
            "command_queue": command_queue,
            "context": ctx,
        }


# ============================================================
# 导出
# ============================================================

__all__ = [
    "TrainerState",
    "DistributedTrainer",
    "launch_distributed_training",
]


if __name__ == "__main__":
    # 被 torchrun 启动时执行
    import argparse
    import json
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    
    config_dict = json.loads(args.config)
    config = TrainingConfig.from_dict(config_dict)
    
    trainer = DistributedTrainer(config)
    trainer.setup()
    trainer.run()
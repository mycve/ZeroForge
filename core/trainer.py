"""
Trainer - 分布式训练器

支持:
- 单机单卡训练
- 单机多卡 DDP 训练
- 多线程异步自玩 + 批量 GPU 推理

架构:
┌─────────────────────────────────────────────────────────────────────────┐
│                          DistributedTrainer                             │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    推理线程 (LeafBatcher)                          │  │
│  │  - 收集多个 Env 提交的叶节点                                        │  │
│  │  - 达到 batch_size 或 timeout 后批量 GPU 推理                      │  │
│  │  - 分发结果给各 Env 线程                                           │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                              ↑ submit()                                 │
│         ┌────────────────────┼────────────────────┐                     │
│         ↓                    ↓                    ↓                     │
│  ┌────────────┐       ┌────────────┐       ┌────────────┐              │
│  │  EnvWorker │       │  EnvWorker │       │  EnvWorker │  ...         │
│  │  (线程 0)  │       │  (线程 1)  │       │  (线程 N)  │              │
│  │   MCTS     │       │   MCTS     │       │   MCTS     │              │
│  │   Game     │       │   Game     │       │   Game     │              │
│  └────────────┘       └────────────┘       └────────────┘              │
│                              │                                          │
│                              ↓                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                      ReplayBuffer                                  │  │
│  │  - 存储轨迹数据                                                     │  │
│  │  - 采样训练批次                                                     │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                              │                                          │
│                              ↓                                          │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                   训练循环 (可选 DDP)                               │  │
│  │  - 从 ReplayBuffer 采样                                            │  │
│  │  - 计算损失，更新网络                                               │  │
│  │  - 同步到推理网络                                                   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
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
from .config import MCTSConfig, BatcherConfig, ThreadedEnvConfig
from .mcts import LeafBatcher, LocalMCTSTree

logger = logging.getLogger(__name__)


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
    
    # 自玩统计
    avg_game_length: float = 0.0
    games_per_second: float = 0.0
    steps_per_second: float = 0.0
    
    # 时间
    start_time: Optional[float] = None
    elapsed_time: float = 0.0
    
    # DDP
    rank: int = 0
    world_size: int = 1


# ============================================================
# 异步自玩工作器
# ============================================================

class AsyncSelfPlayWorker:
    """异步自玩工作线程
    
    每个工作线程维护独立的:
    - 游戏实例
    - MCTS 树
    
    通过 LeafBatcher 提交叶节点到 GPU 批量推理。
    """
    
    def __init__(
        self,
        worker_id: int,
        game_factory: Callable,
        batcher: LeafBatcher,
        mcts_config: MCTSConfig,
        trajectory_queue: "ThreadSafeQueue",
    ):
        self.worker_id = worker_id
        self.game_factory = game_factory
        self.batcher = batcher
        self.mcts_config = mcts_config
        self.trajectory_queue = trajectory_queue
        
        # 状态
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._games_completed = 0
        self._total_steps = 0
    
    def start(self) -> None:
        """启动工作线程"""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
    
    def stop(self) -> None:
        """停止工作线程"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5.0)
            self._thread = None
    
    def _run_loop(self) -> None:
        """主循环：不断执行自玩"""
        logger.debug(f"Worker {self.worker_id} 启动")
        
        while self._running:
            try:
                trajectory = self._play_one_game()
                if trajectory and len(trajectory) > 0:
                    self.trajectory_queue.put(trajectory)
                    self._games_completed += 1
            except Exception as e:
                logger.error(f"Worker {self.worker_id} 自玩出错: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)
        
        logger.debug(f"Worker {self.worker_id} 停止")
    
    def _play_one_game(self) -> Optional[Trajectory]:
        """执行一局自玩"""
        game = self.game_factory()
        game.reset()
        
        # 创建 MCTS 树
        mcts_tree = LocalMCTSTree(game, self.mcts_config, mode="alphazero")
        
        trajectory = Trajectory()
        move_count = 0
        
        while not game.is_terminal() and self._running:
            current_player = game.current_player()
            obs = game.get_observation()
            
            # 使用 batcher 的评估函数
            def evaluate_fn(obs_arr: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, float]:
                return self.batcher.submit(obs_arr, mask, env_id=self.worker_id)
            
            # MCTS 搜索
            add_noise = (move_count == 0)
            action, policy_dict, root_value = mcts_tree.search(
                evaluate_fn=evaluate_fn,
                num_simulations=self.mcts_config.num_simulations,
                add_noise=add_noise,
            )
            
            # 记录轨迹
            trajectory.append(
                observation=obs,
                action=action,
                reward=0.0,
                policy=policy_dict,
                value=root_value,
                to_play=current_player,
            )
            
            # 执行动作
            game.step(action)
            move_count += 1
            self._total_steps += 1
            
            # 复用子树
            mcts_tree.game = game
            mcts_tree.advance(action)
            
            # 防止死循环
            if move_count > 500:
                logger.warning(f"Worker {self.worker_id}: 游戏超过 500 步，强制结束")
                break
        
        # 设置奖励
        winner = game.get_winner()
        for i, player in enumerate(trajectory.to_play):
            if winner is None:
                trajectory.rewards[i] = 0.0
            elif player == winner:
                trajectory.rewards[i] = 1.0
            else:
                trajectory.rewards[i] = -1.0
        
        return trajectory
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计"""
        return {
            "worker_id": self.worker_id,
            "games_completed": self._games_completed,
            "total_steps": self._total_steps,
            "running": self._running,
        }


# ============================================================
# 线程安全队列
# ============================================================

class ThreadSafeQueue:
    """线程安全的轨迹队列"""
    
    def __init__(self, maxsize: int = 1000):
        self._queue: List[Trajectory] = []
        self._lock = threading.Lock()
        self._maxsize = maxsize
    
    def put(self, item: Trajectory) -> None:
        with self._lock:
            self._queue.append(item)
            # 超出大小时丢弃旧的
            while len(self._queue) > self._maxsize:
                self._queue.pop(0)
    
    def get_all(self) -> List[Trajectory]:
        """获取所有轨迹并清空"""
        with self._lock:
            items = self._queue.copy()
            self._queue.clear()
            return items
    
    def __len__(self) -> int:
        with self._lock:
            return len(self._queue)


# ============================================================
# 分布式训练器
# ============================================================

class DistributedTrainer:
    """分布式训练器
    
    支持:
    - 单卡训练
    - DDP 多卡训练
    - 异步多线程自玩
    
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
        
        # 组件
        self._network = None
        self._optimizer = None
        self._batcher = None
        self._workers: List[AsyncSelfPlayWorker] = []
        self._trajectory_queue = ThreadSafeQueue()
        self._replay_buffer = None
        self._checkpoint_manager = None
        
        # 控制
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._callbacks: List[Callable] = []
    
    # ========================================
    # 初始化
    # ========================================
    
    def setup(self) -> None:
        """初始化训练器"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch 未安装")
        
        # 检测分布式环境
        self._setup_distributed()
        
        # 创建网络
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
        
        # 创建 Batcher 和 Workers（只在 rank 0）
        if self._local_rank == 0:
            self._setup_selfplay()
        
        logger.info(f"训练器初始化完成: rank={self._local_rank}, world_size={self._world_size}, "
                   f"device={self._device}")
    
    def _setup_distributed(self) -> None:
        """设置分布式环境"""
        # 检查环境变量
        if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
            self._local_rank = int(os.environ.get("LOCAL_RANK", 0))
            self._world_size = int(os.environ["WORLD_SIZE"])
            self._is_distributed = self._world_size > 1
            
            if self._is_distributed:
                dist.init_process_group(backend="nccl")
                torch.cuda.set_device(self._local_rank)
                self._device = torch.device(f"cuda:{self._local_rank}")
                logger.info(f"DDP 初始化: rank={self._local_rank}, world_size={self._world_size}")
            else:
                self._device = torch.device(resolve_device(self.config.device))
        else:
            # 单卡模式
            self._device = torch.device(resolve_device(self.config.device))
            self._local_rank = 0
            self._world_size = 1
        
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
    
    def _setup_selfplay(self) -> None:
        """设置自玩组件（只在 rank 0）"""
        from games import make_game
        
        # 创建 Batcher
        batcher_config = BatcherConfig(
            batch_size=self.config.inference_batch_size,
            timeout_ms=5.0,
            device=str(self._device),
        )
        
        # 获取底层网络（去掉 DDP 包装）
        base_network = self._network.module if self._is_distributed else self._network
        self._batcher = LeafBatcher(base_network, batcher_config)
        
        # 创建 MCTS 配置
        mcts_config = MCTSConfig(
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
        
        # 创建 Workers
        for i in range(self.config.num_envs):
            worker = AsyncSelfPlayWorker(
                worker_id=i,
                game_factory=game_factory,
                batcher=self._batcher,
                mcts_config=mcts_config,
                trajectory_queue=self._trajectory_queue,
            )
            self._workers.append(worker)
        
        logger.info(f"创建 {len(self._workers)} 个自玩工作线程")
    
    # ========================================
    # 训练循环
    # ========================================
    
    def run(self) -> None:
        """运行训练循环"""
        self.state.running = True
        self.state.start_time = time.time()
        self._stop_event.clear()
        
        try:
            # 启动自玩（只在 rank 0）
            if self._local_rank == 0:
                self._start_selfplay()
            
            # 训练循环
            for epoch in range(self.config.num_epochs):
                if self._stop_event.is_set():
                    break
                
                # 暂停检查
                while self.state.paused and not self._stop_event.is_set():
                    time.sleep(0.1)
                
                if self._stop_event.is_set():
                    break
                
                # 收集轨迹（只在 rank 0）
                if self._local_rank == 0:
                    self._collect_trajectories()
                
                # 等待足够数据
                if len(self._replay_buffer) < self.config.min_buffer_size:
                    if self._local_rank == 0:
                        logger.debug(f"等待数据: {len(self._replay_buffer)}/{self.config.min_buffer_size}")
                    time.sleep(0.5)
                    continue
                
                # 训练一个 epoch
                train_metrics = self._train_epoch()
                
                # 同步网络到 batcher（只在 rank 0）
                if self._local_rank == 0:
                    self._sync_network_to_batcher()
                
                # 更新状态
                self._update_state(epoch + 1, train_metrics)
                
                # 保存检查点
                if (epoch + 1) % self.config.save_interval == 0:
                    self._save_checkpoint(epoch + 1)
                
                # 通知回调
                self._notify_callbacks()
                
                # 日志
                if self._local_rank == 0:
                    logger.info(
                        f"Epoch {epoch + 1}/{self.config.num_epochs}: "
                        f"loss={train_metrics.get('total_loss', 0):.4f}, "
                        f"games={self.state.total_games}, "
                        f"buffer={len(self._replay_buffer)}"
                    )
            
            # 保存最终检查点
            self._save_checkpoint(self.config.num_epochs, is_final=True)
            
        except Exception as e:
            logger.error(f"训练出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self._cleanup()
    
    def _start_selfplay(self) -> None:
        """启动自玩"""
        if self._batcher:
            self._batcher.start()
        
        for worker in self._workers:
            worker.start()
        
        logger.info("自玩已启动")
    
    def _stop_selfplay(self) -> None:
        """停止自玩"""
        for worker in self._workers:
            worker.stop()
        
        if self._batcher:
            self._batcher.stop()
        
        logger.info("自玩已停止")
    
    def _collect_trajectories(self) -> None:
        """收集轨迹到 ReplayBuffer"""
        trajectories = self._trajectory_queue.get_all()
        if trajectories:
            self._replay_buffer.add_batch(trajectories)
            self.state.total_games += len(trajectories)
            
            # 计算平均长度
            total_length = sum(len(t) for t in trajectories)
            self.state.avg_game_length = total_length / len(trajectories)
    
    def _train_epoch(self) -> Dict[str, float]:
        """训练一个 epoch"""
        self._network.train()
        
        total_loss = 0.0
        value_loss_sum = 0.0
        policy_loss_sum = 0.0
        num_batches = max(1, self.config.train_batches_per_epoch)
        
        for _ in range(num_batches):
            if self._stop_event.is_set():
                break
            
            # 采样
            batch = self._replay_buffer.sample(
                batch_size=self.config.batch_size,
                unroll_steps=0,
            )
            
            # 转换为 tensor
            obs = torch.from_numpy(batch.observations).to(self._device)
            target_policy = torch.from_numpy(batch.target_policies[:, 0, :]).to(self._device)
            target_value = torch.from_numpy(batch.target_values[:, 0]).to(self._device)
            
            # 前向传播
            policy_logits, value = self._network(obs)
            
            # 计算损失
            policy_loss = F.cross_entropy(policy_logits, target_policy)
            value_loss = F.mse_loss(value.squeeze(-1), target_value)
            loss = policy_loss + value_loss
            
            # 反向传播
            self._optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._network.parameters(), 1.0)
            self._optimizer.step()
            
            total_loss += loss.item()
            value_loss_sum += value_loss.item()
            policy_loss_sum += policy_loss.item()
            
            with self._lock:
                self.state.step += 1
        
        return {
            "total_loss": total_loss / num_batches,
            "value_loss": value_loss_sum / num_batches,
            "policy_loss": policy_loss_sum / num_batches,
        }
    
    def _sync_network_to_batcher(self) -> None:
        """同步网络参数到 batcher"""
        # batcher 使用的是同一个网络引用，无需额外同步
        # 但如果使用了网络副本，需要在这里同步
        pass
    
    def _update_state(self, epoch: int, metrics: Dict[str, float]) -> None:
        """更新训练状态"""
        with self._lock:
            self.state.epoch = epoch
            self.state.loss = metrics.get("total_loss", 0)
            self.state.value_loss = metrics.get("value_loss", 0)
            self.state.policy_loss = metrics.get("policy_loss", 0)
            
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
        
        if self._local_rank == 0:
            self._stop_selfplay()
        
        if self._is_distributed:
            dist.destroy_process_group()
    
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
            return {
                "running": self.state.running,
                "paused": self.state.paused,
                "epoch": self.state.epoch,
                "step": self.state.step,
                "total_games": self.state.total_games,
                "loss": self.state.loss,
                "value_loss": self.state.value_loss,
                "policy_loss": self.state.policy_loss,
                "avg_game_length": self.state.avg_game_length,
                "games_per_second": self.state.games_per_second,
                "steps_per_second": self.state.steps_per_second,
                "elapsed_time": self.state.elapsed_time,
                "rank": self.state.rank,
                "world_size": self.state.world_size,
                "buffer_size": len(self._replay_buffer) if self._replay_buffer else 0,
            }


# ============================================================
# 便捷函数
# ============================================================

def launch_distributed_training(
    config: TrainingConfig,
    num_gpus: int = 1,
) -> None:
    """启动分布式训练
    
    Args:
        config: 训练配置
        num_gpus: GPU 数量
    
    Example:
        >>> config = TrainingConfig(game_type="chinese_chess")
        >>> launch_distributed_training(config, num_gpus=4)
    """
    if num_gpus > 1:
        # 使用 torchrun 启动
        import subprocess
        import sys
        
        cmd = [
            sys.executable, "-m", "torch.distributed.run",
            f"--nproc_per_node={num_gpus}",
            __file__,
            "--config", str(config.to_dict()),
        ]
        subprocess.run(cmd)
    else:
        # 单卡直接运行
        trainer = DistributedTrainer(config)
        trainer.setup()
        trainer.run()


# ============================================================
# 导出
# ============================================================

__all__ = [
    "TrainerState",
    "AsyncSelfPlayWorker",
    "ThreadSafeQueue",
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

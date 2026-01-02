"""
SelfPlay - 多线程自玩管理

提供高效的多线程自玩支持:
- ThreadedSelfPlay: 多线程自玩管理器
- EnvWorker: 单个环境工作线程（支持异步批推理）

架构:
┌─────────────────────────────────────────────────────────────────────┐
│                       ThreadedSelfPlay                              │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │               LeafBatcher (推理线程)                         │    │
│  │  - 收集各 Env 线程提交的叶节点                                │    │
│  │  - 批量 GPU 推理                                             │    │
│  │  - 分发结果                                                  │    │
│  └────────────────────────────────────────────────────────────┘    │
│                         ↑ submit()  ↓ result                        │
│       ┌─────────────────┼───────────┼─────────────────┐            │
│       ↓                 ↓           ↓                 ↓            │
│  ┌─────────┐      ┌─────────┐ ┌─────────┐      ┌─────────┐        │
│  │ Worker0 │      │ Worker1 │ │ Worker2 │ ...  │ WorkerN │        │
│  │ Game    │      │ Game    │ │ Game    │      │ Game    │        │
│  │ MCTS    │      │ MCTS    │ │ MCTS    │      │ MCTS    │        │
│  └─────────┘      └─────────┘ └─────────┘      └─────────┘        │
│       │                 │           │                 │            │
│       └─────────────────┴───────────┴─────────────────┘            │
│                              ↓                                      │
│                     轨迹数据 (Trajectory)                           │
└─────────────────────────────────────────────────────────────────────┘

设计原则:
- 每个线程维护独立的游戏实例和 MCTS 树
- MCTS 搜索遇到叶节点时，通过 batcher.submit() 提交到推理队列
- Batcher 收集满 batch_size 或超时后批量推理，分发结果
- 支持子树复用提升效率
"""

import threading
import time
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple
from queue import Queue
import numpy as np

from .config import ThreadedEnvConfig, MCTSConfig
from .mcts import MCTSSearch, LeafBatcher, LocalMCTSTree
from .algorithm import Trajectory

logger = logging.getLogger(__name__)


# ============================================================
# 自玩结果
# ============================================================

@dataclass
class SelfPlayResult:
    """单局自玩结果"""
    trajectory: Trajectory
    winner: Optional[int]
    game_length: int
    total_time: float


@dataclass
class SelfPlayStats:
    """自玩统计信息"""
    num_games: int = 0
    total_time: float = 0.0
    avg_game_length: float = 0.0
    win_stats: Dict[str, int] = field(default_factory=dict)
    games_per_second: float = 0.0


# ============================================================
# 环境工作线程
# ============================================================

class EnvWorker:
    """单个环境工作线程
    
    负责运行一个游戏实例，执行自玩。
    支持同步和异步两种模式。
    
    异步模式:
        MCTS 搜索时，遇到叶节点通过 batcher.submit() 提交到推理队列，
        等待批量推理完成后继续搜索。
    
    同步模式:
        直接调用 evaluate_fn 进行推理，适用于调试。
    
    Attributes:
        game_factory: 游戏创建函数
        config: MCTS 配置
        batcher: GPU 批推理器（异步模式）
        worker_id: 工作线程 ID
    """
    
    def __init__(
        self,
        game_factory: Callable,
        config: MCTSConfig,
        batcher: Optional[LeafBatcher] = None,
        worker_id: int = 0,
    ):
        self.game_factory = game_factory
        self.config = config
        self.batcher = batcher
        self.worker_id = worker_id
        
        # 创建游戏
        self.game = game_factory()
        
        # MCTS 树（使用本地树实现，支持异步评估）
        self.mcts_tree = LocalMCTSTree(self.game, config, mode="alphazero")
        
        # 统计
        self._games_completed = 0
        self._total_steps = 0
    
    def play_one_game_async(self) -> SelfPlayResult:
        """异步执行一局自玩（通过 batcher 批量推理）
        
        Returns:
            SelfPlayResult: 自玩结果
            
        Raises:
            RuntimeError: 如果 batcher 未设置
        """
        if self.batcher is None:
            raise RuntimeError("异步模式需要设置 batcher")
        
        start_time = time.time()
        
        # 重置
        self.game.reset()
        self.mcts_tree = LocalMCTSTree(self.game, self.config, mode="alphazero")
        
        trajectory = Trajectory()
        move_count = 0
        
        # 创建异步评估函数
        def evaluate_fn(obs: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, float]:
            """通过 batcher 提交并等待结果"""
            return self.batcher.submit(obs, mask, env_id=self.worker_id)
        
        while not self.game.is_terminal():
            obs = self.game.get_observation()
            current_player = self.game.current_player()
            
            # MCTS 搜索（内部会多次调用 evaluate_fn）
            add_noise = (move_count == 0)
            action, policy, value = self.mcts_tree.search(
                evaluate_fn=evaluate_fn,
                num_simulations=self.config.num_simulations,
                add_noise=add_noise,
            )
            
            # 记录轨迹
            trajectory.append(
                observation=obs,
                action=action,
                reward=0.0,
                policy=policy,
                value=value,
                to_play=current_player,
            )
            
            # 执行动作
            self.game.step(action)
            move_count += 1
            self._total_steps += 1
            
            # 复用子树
            self.mcts_tree.game = self.game
            self.mcts_tree.advance(action)
            
            # 检查最大步数
            if move_count >= self.config.max_tree_depth * 2:
                logger.warning(f"Worker {self.worker_id}: 达到最大步数限制 ({move_count})")
                break
        
        # 设置最终奖励
        winner = self.game.get_winner()
        for i, player in enumerate(trajectory.to_play):
            if winner is None:
                trajectory.rewards[i] = 0.0
            elif player == winner:
                trajectory.rewards[i] = 1.0
            else:
                trajectory.rewards[i] = -1.0
        
        elapsed = time.time() - start_time
        self._games_completed += 1
        
        return SelfPlayResult(
            trajectory=trajectory,
            winner=winner,
            game_length=move_count,
            total_time=elapsed,
        )
    
    def play_one_game(self) -> SelfPlayResult:
        """执行一局自玩（兼容旧接口，使用异步模式）"""
        return self.play_one_game_async()
    
    def play_one_game_sync(self, evaluate_fn: Callable) -> SelfPlayResult:
        """同步执行一局自玩（直接调用 evaluate_fn）
        
        Args:
            evaluate_fn: 评估函数 (obs, mask) -> (policy, value)
            
        Returns:
            SelfPlayResult: 自玩结果
        """
        start_time = time.time()
        
        # 重置
        self.game.reset()
        self.mcts_tree = LocalMCTSTree(self.game, self.config, mode="alphazero")
        
        trajectory = Trajectory()
        move_count = 0
        
        while not self.game.is_terminal():
            obs = self.game.get_observation()
            current_player = self.game.current_player()
            
            # 同步 MCTS 搜索
            add_noise = (move_count == 0)
            action, policy, value = self.mcts_tree.search(
                evaluate_fn=evaluate_fn,
                num_simulations=self.config.num_simulations,
                add_noise=add_noise,
            )
            
            trajectory.append(
                observation=obs,
                action=action,
                reward=0.0,
                policy=policy,
                value=value,
                to_play=current_player,
            )
            
            self.game.step(action)
            move_count += 1
            self._total_steps += 1
            
            # 复用子树
            self.mcts_tree.game = self.game
            self.mcts_tree.advance(action)
            
            if move_count >= self.config.max_tree_depth * 2:
                break
        
        # 设置最终奖励
        winner = self.game.get_winner()
        for i, player in enumerate(trajectory.to_play):
            if winner is None:
                trajectory.rewards[i] = 0.0
            elif player == winner:
                trajectory.rewards[i] = 1.0
            else:
                trajectory.rewards[i] = -1.0
        
        elapsed = time.time() - start_time
        self._games_completed += 1
        
        return SelfPlayResult(
            trajectory=trajectory,
            winner=winner,
            game_length=move_count,
            total_time=elapsed,
        )
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "worker_id": self.worker_id,
            "games_completed": self._games_completed,
            "total_steps": self._total_steps,
        }


# ============================================================
# 多线程自玩管理器
# ============================================================

class ThreadedSelfPlay:
    """多线程自玩管理器
    
    管理多个环境线程，并行执行自玩。
    
    Attributes:
        game_factory: 游戏创建函数
        network: 神经网络
        config: 多线程配置
        
    Example:
        >>> selfplay = ThreadedSelfPlay(
        ...     game_factory=lambda: make_game("tictactoe"),
        ...     network=network,
        ...     config=ThreadedEnvConfig(num_envs=8),
        ... )
        >>> selfplay.start()
        >>> trajectories = selfplay.collect(num_games=100)
        >>> selfplay.stop()
    """
    
    def __init__(
        self,
        game_factory: Callable,
        network: Any,  # nn.Module
        config: ThreadedEnvConfig,
    ):
        self.game_factory = game_factory
        self.network = network
        self.config = config
        
        # 创建 batcher
        self.batcher = LeafBatcher(network, config.batcher)
        
        # 工作线程
        self._workers: List[EnvWorker] = []
        self._threads: List[threading.Thread] = []
        self._running = False
        
        # 结果收集
        self._results_queue: Queue[SelfPlayResult] = Queue()
        self._lock = threading.Lock()
        
        # 统计
        self._total_games = 0
        self._total_time = 0.0
    
    def start(self) -> None:
        """启动所有工作线程"""
        if self._running:
            logger.warning("ThreadedSelfPlay 已在运行")
            return
        
        self._running = True
        
        # 启动 batcher
        self.batcher.start()
        
        # 创建工作线程
        for i in range(self.config.num_envs):
            worker = EnvWorker(
                game_factory=self.game_factory,
                config=self.config.mcts,
                batcher=self.batcher,
                worker_id=i,
            )
            self._workers.append(worker)
        
        logger.info(f"ThreadedSelfPlay 已启动: {self.config.num_envs} 个环境")
    
    def stop(self) -> None:
        """停止所有工作线程"""
        self._running = False
        
        # 等待所有线程结束
        for thread in self._threads:
            if thread.is_alive():
                thread.join(timeout=5.0)
        
        self._threads.clear()
        
        # 停止 batcher
        self.batcher.stop()
        
        logger.info("ThreadedSelfPlay 已停止")
    
    def collect(self, num_games: int) -> List[Trajectory]:
        """收集指定数量的自玩轨迹
        
        Args:
            num_games: 要收集的游戏数量
            
        Returns:
            轨迹列表
        """
        trajectories, _ = self.collect_with_stats(num_games)
        return trajectories
    
    def collect_with_stats(self, num_games: int) -> Tuple[List[Trajectory], SelfPlayStats]:
        """收集轨迹并返回统计信息
        
        Args:
            num_games: 要收集的游戏数量
            
        Returns:
            (trajectories, stats)
        """
        if not self._running:
            raise RuntimeError("ThreadedSelfPlay 未启动")
        
        start_time = time.time()
        trajectories: List[Trajectory] = []
        results: List[SelfPlayResult] = []
        
        # 简单实现：轮流使用 workers
        games_per_worker = num_games // len(self._workers) + 1
        
        # 为每个 worker 创建线程执行自玩
        def worker_task(worker: EnvWorker, num: int, results_list: List):
            for _ in range(num):
                if not self._running:
                    break
                try:
                    result = worker.play_one_game()
                    with self._lock:
                        results_list.append(result)
                except Exception as e:
                    logger.error(f"Worker {worker.worker_id} 自玩失败: {e}")
        
        threads = []
        results_list = []
        
        for worker in self._workers:
            t = threading.Thread(
                target=worker_task,
                args=(worker, games_per_worker, results_list)
            )
            threads.append(t)
            t.start()
        
        # 等待所有线程完成，或达到目标数量
        while len(results_list) < num_games and any(t.is_alive() for t in threads):
            time.sleep(0.1)
        
        # 停止多余的工作
        for t in threads:
            if t.is_alive():
                t.join(timeout=1.0)
        
        # 取前 num_games 个结果
        results = results_list[:num_games]
        trajectories = [r.trajectory for r in results]
        
        # 统计
        elapsed = time.time() - start_time
        win_stats: Dict[str, int] = {"player_0": 0, "player_1": 0, "draw": 0}
        total_length = 0
        
        for r in results:
            total_length += r.game_length
            if r.winner is None:
                win_stats["draw"] += 1
            elif r.winner == 0:
                win_stats["player_0"] += 1
            else:
                win_stats["player_1"] += 1
        
        stats = SelfPlayStats(
            num_games=len(results),
            total_time=elapsed,
            avg_game_length=total_length / len(results) if results else 0,
            win_stats=win_stats,
            games_per_second=len(results) / elapsed if elapsed > 0 else 0,
        )
        
        self._total_games += len(results)
        self._total_time += elapsed
        
        return trajectories, stats
    
    def collect_sync(
        self,
        num_games: int,
        evaluate_fn: Callable,
    ) -> Tuple[List[Trajectory], SelfPlayStats]:
        """同步收集（单线程，用于调试）
        
        Args:
            num_games: 游戏数量
            evaluate_fn: 评估函数
            
        Returns:
            (trajectories, stats)
        """
        start_time = time.time()
        
        # 使用单个 worker
        worker = EnvWorker(
            game_factory=self.game_factory,
            config=self.config.mcts,
            batcher=self.batcher,
            worker_id=0,
        )
        
        trajectories = []
        results = []
        
        for i in range(num_games):
            result = worker.play_one_game_sync(evaluate_fn)
            trajectories.append(result.trajectory)
            results.append(result)
            
            if (i + 1) % 10 == 0:
                logger.debug(f"同步自玩进度: {i + 1}/{num_games}")
        
        elapsed = time.time() - start_time
        
        win_stats = {"player_0": 0, "player_1": 0, "draw": 0}
        total_length = 0
        
        for r in results:
            total_length += r.game_length
            if r.winner is None:
                win_stats["draw"] += 1
            elif r.winner == 0:
                win_stats["player_0"] += 1
            else:
                win_stats["player_1"] += 1
        
        stats = SelfPlayStats(
            num_games=len(results),
            total_time=elapsed,
            avg_game_length=total_length / len(results) if results else 0,
            win_stats=win_stats,
            games_per_second=len(results) / elapsed if elapsed > 0 else 0,
        )
        
        return trajectories, stats
    
    def get_stats(self) -> Dict[str, Any]:
        """获取总体统计"""
        return {
            "total_games": self._total_games,
            "total_time": self._total_time,
            "num_envs": len(self._workers),
            "running": self._running,
            "batcher_stats": self.batcher.get_stats() if self.batcher else {},
        }


# ============================================================
# 导出
# ============================================================

__all__ = [
    "SelfPlayResult",
    "SelfPlayStats",
    "EnvWorker",
    "ThreadedSelfPlay",
]

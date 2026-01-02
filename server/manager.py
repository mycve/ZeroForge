"""
Managers - 服务管理器

提供训练、游戏、系统的状态管理和控制。
"""

import asyncio
import time
import logging
import uuid
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Callable, Set
from pathlib import Path
import json
import threading
import numpy as np

logger = logging.getLogger(__name__)


def _to_python(obj):
    """将 numpy 类型转换为 Python 原生类型，用于 JSON 序列化"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {_to_python(k): _to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    return obj


# ============================================================
# 训练管理器
# ============================================================

@dataclass
class TrainingStatus:
    """训练状态"""
    running: bool = False
    paused: bool = False
    step: int = 0
    epoch: int = 0
    total_games: int = 0
    
    # 最新指标
    loss: float = 0.0
    value_loss: float = 0.0
    policy_loss: float = 0.0
    reward_loss: float = 0.0
    
    # 自玩统计
    selfplay_games: int = 0
    avg_game_length: float = 0.0
    win_rate: Dict[str, float] = field(default_factory=dict)
    
    # 评估结果
    eval_win_rate: float = 0.0
    eval_elo: float = 0.0
    
    # 架构状态
    buffer_size: int = 0          # 回放缓冲区大小
    num_envs: int = 0             # 并行环境数
    
    # 时间
    start_time: Optional[float] = None
    elapsed_time: float = 0.0
    games_per_second: float = 0.0
    steps_per_second: float = 0.0


class TrainingManager:
    """训练管理器
    
    管理训练状态，支持暂停/恢复/停止。
    在后台线程中执行真实的神经网络训练。
    """
    
    def __init__(self):
        self.status = TrainingStatus()
        self._subscribers: Set[Callable] = set()
        self._config: Dict[str, Any] = {}
        self._lock = threading.RLock()
        self._training_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # 训练组件（运行时创建）
        self._network = None
        self._optimizer = None
        self._checkpoint_manager = None
        self._trainer = None  # DistributedTrainer 实例
        
        # 调试数据（用于前端展示）
        self._debug_data = {
            "selfplay": [],      # 最近的自玩调试信息
            "training": [],      # 最近的训练调试信息
            "trajectories": [],  # 最近的轨迹样本
            "mcts": [],          # 最近的 MCTS 搜索信息
        }
        self._debug_max_items = 100  # 每类保留最多条目
    
    def get_status(self) -> Dict[str, Any]:
        """获取训练状态"""
        with self._lock:
            # 使用 _to_python 转换 numpy 类型，避免 JSON 序列化错误
            return _to_python({
                "running": self.status.running,
                "paused": self.status.paused,
                "step": self.status.step,
                "epoch": self.status.epoch,
                "total_games": self.status.total_games,
                "loss": self.status.loss,
                "value_loss": self.status.value_loss,
                "policy_loss": self.status.policy_loss,
                "reward_loss": self.status.reward_loss,
                "selfplay_games": self.status.selfplay_games,
                "avg_game_length": self.status.avg_game_length,
                "win_rate": self.status.win_rate,
                "eval_win_rate": self.status.eval_win_rate,
                "eval_elo": self.status.eval_elo,
                "elapsed_time": self.status.elapsed_time,
                "games_per_second": self.status.games_per_second,
                "steps_per_second": self.status.steps_per_second,
                # 架构状态
                "buffer_size": self.status.buffer_size,
                "num_envs": self.status.num_envs,
            })
    
    def _add_debug(self, category: str, data: Dict[str, Any]):
        """添加调试数据"""
        with self._lock:
            import time
            data["timestamp"] = time.time()
            self._debug_data[category].append(data)
            # 限制数量
            if len(self._debug_data[category]) > self._debug_max_items:
                self._debug_data[category] = self._debug_data[category][-self._debug_max_items:]
    
    def get_debug_data(self, category: str = None, limit: int = 50) -> Dict[str, Any]:
        """获取调试数据
        
        Args:
            category: 类别 (selfplay/training/trajectories/mcts)，None 返回全部
            limit: 每类返回条数
        """
        with self._lock:
            if category:
                return {category: self._debug_data.get(category, [])[-limit:]}
            return {k: v[-limit:] for k, v in self._debug_data.items()}
    
    def clear_debug_data(self):
        """清空调试数据"""
        with self._lock:
            for k in self._debug_data:
                self._debug_data[k] = []
    
    def update_status(self, **kwargs):
        """更新训练状态"""
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self.status, key):
                    setattr(self.status, key, value)
            
            if self.status.start_time and self.status.running:
                elapsed = time.time() - self.status.start_time
                self.status.elapsed_time = elapsed
                if elapsed > 0:
                    self.status.games_per_second = self.status.total_games / elapsed
                    self.status.steps_per_second = self.status.step / elapsed
        
        self._notify_subscribers()
    
    def start(self, config: Dict[str, Any] = None):
        """开始训练"""
        with self._lock:
            if self.status.running:
                logger.warning("训练已在运行中")
                return
            
            self.status.running = True
            self.status.paused = False
            self.status.start_time = time.time()
            self.status.step = 0
            self.status.epoch = 0
            self.status.total_games = 0
            if config:
                self._config = config
        
        self._stop_event.clear()
        self._training_thread = threading.Thread(target=self._training_loop, daemon=True)
        self._training_thread.start()
        
        logger.info("训练已启动")
        self._notify_subscribers()
    
    def _training_loop(self):
        """训练主循环 - 使用 DistributedTrainer
        
        新架构特点:
        - 多环境并行自玩（num_envs 个环境，每个自动开线程）
        - 叶节点批量 GPU 推理（LeafBatcher）
        - 支持 DDP 多卡训练
        """
        try:
            from core.training_config import TrainingConfig
            from core.trainer import DistributedTrainer
            
            # 解析配置
            config = TrainingConfig.from_dict(self._config)
            logger.info(f"训练配置: game={config.game_type}, algo={config.algorithm}, "
                       f"envs={config.num_envs}, device={config.device}")
            
            # 创建分布式训练器
            trainer = DistributedTrainer(config)
            self._trainer = trainer
            
            # 添加状态回调
            def on_state_update(state):
                trainer_state = trainer.get_state()
                with self._lock:
                    self.status.epoch = state.epoch
                    self.status.step = state.step
                    self.status.total_games = state.total_games
                    self.status.selfplay_games = state.total_games
                    self.status.loss = state.loss
                    self.status.value_loss = state.value_loss
                    self.status.policy_loss = state.policy_loss
                    self.status.reward_loss = getattr(state, 'reward_loss', 0)  # MuZero 奖励损失
                    self.status.avg_game_length = state.avg_game_length
                    self.status.games_per_second = state.games_per_second
                    self.status.steps_per_second = state.steps_per_second
                    self.status.elapsed_time = state.elapsed_time
                    
                    # 评估指标（只有 eval_games > 0 才是真正评估过）
                    eval_games = trainer_state.get("eval_games", 0)
                    if eval_games > 0:
                        self.status.eval_win_rate = trainer_state.get("eval_win_rate", 0)
                        self.status.eval_elo = trainer_state.get("elo_rating", 1500)
                    
                    # 架构状态
                    self.status.buffer_size = trainer_state.get("buffer_size", 0)
                    self.status.num_envs = config.num_envs
                    
                    # 调试信息
                    self._add_debug("training", {
                        "epoch": state.epoch,
                        "step": state.step,
                        "total_games": state.total_games,
                        "loss": round(state.loss, 4) if state.loss else 0,
                        "value_loss": round(state.value_loss, 4) if state.value_loss else 0,
                        "policy_loss": round(state.policy_loss, 4) if state.policy_loss else 0,
                        "buffer_size": self.status.buffer_size,
                        "num_envs": self.status.num_envs,
                        "concurrency": config.concurrency,
                        "eval_win_rate": round(self.status.eval_win_rate, 3),
                        "eval_elo": round(self.status.eval_elo, 1),
                        "games_per_second": round(state.games_per_second, 2) if state.games_per_second else 0,
                        "steps_per_second": round(state.steps_per_second, 2) if state.steps_per_second else 0,
                    })
                
                self._notify_subscribers()
            
            trainer.add_callback(on_state_update)
            
            # 设置训练器
            trainer.setup()
            
            # 保存组件引用
            self._network = trainer._network
            self._optimizer = trainer._optimizer
            self._checkpoint_manager = trainer._checkpoint_manager
            
            # 运行训练
            trainer.run()
            
            logger.info("训练完成")
            
        except Exception as e:
            logger.error(f"训练出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            with self._lock:
                self.status.running = False
            self._notify_subscribers()
    
    def _training_loop_legacy(self):
        """旧版训练循环（保留用于对比测试）"""
        import torch
        import torch.nn.functional as F
        import numpy as np
        
        try:
            from core.training_config import TrainingConfig, resolve_device
            from core.replay_buffer import ReplayBuffer
            from core.checkpoint import CheckpointManager
            from games import make_game
            from algorithms import make_algorithm
            
            config = TrainingConfig.from_dict(self._config)
            device = resolve_device(config.device)
            
            game = make_game(config.game_type)
            game.reset()
            action_size = game.action_space.n
            
            algo = make_algorithm(config.algorithm)
            network = algo.create_network(game).to(device)
            
            optimizer = torch.optim.AdamW(
                network.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay,
            )
            
            replay_buffer = ReplayBuffer(
                capacity=config.replay_buffer_size,
                action_space_size=action_size,
            )
            
            self._checkpoint_manager = CheckpointManager(
                base_dir=config.checkpoint_dir,
                keep_checkpoints=config.keep_checkpoints,
            )
            
            self._network = network
            self._optimizer = optimizer
            
            for epoch in range(config.num_epochs):
                if self._stop_event.is_set():
                    break
                
                while self.status.paused and not self._stop_event.is_set():
                    time.sleep(0.1)
                
                if self._stop_event.is_set():
                    break
                
                # 自玩（旧版同步模式：每 epoch 执行固定数量游戏）
                num_games_sync = 10  # 旧版同步模式固定执行 10 局
                trajectories, selfplay_stats = self._run_selfplay(
                    game_type=config.game_type,
                    network=network,
                    device=device,
                    num_games=num_games_sync,
                    num_simulations=config.num_simulations,
                    c_puct=config.c_puct,
                    dirichlet_alpha=config.dirichlet_alpha,
                    dirichlet_epsilon=config.dirichlet_epsilon,
                    temperature_threshold=config.temperature_threshold,
                )
                
                replay_buffer.add_batch(trajectories)
                
                # 训练
                train_losses = {"total": 0, "value": 0, "policy": 0}
                
                if len(replay_buffer) >= config.min_buffer_size:
                    network.train()
                    num_batches = max(1, len(trajectories) * 5 // config.batch_size)
                    
                    for _ in range(num_batches):
                        if self._stop_event.is_set():
                            break
                        
                        batch = replay_buffer.sample(batch_size=config.batch_size, unroll_steps=0)
                        
                        obs = torch.from_numpy(batch.observations).to(device)
                        target_policy = torch.from_numpy(batch.target_policies[:, 0, :]).to(device)
                        target_value = torch.from_numpy(batch.target_values[:, 0]).to(device)
                        
                        policy_logits, value = network(obs)
                        
                        policy_loss = F.cross_entropy(policy_logits, target_policy)
                        value_loss = F.mse_loss(value.squeeze(), target_value)
                        total_loss = policy_loss + value_loss
                        
                        optimizer.zero_grad()
                        total_loss.backward()
                        torch.nn.utils.clip_grad_norm_(network.parameters(), config.grad_clip)
                        optimizer.step()
                        
                        train_losses["total"] += total_loss.item()
                        train_losses["value"] += value_loss.item()
                        train_losses["policy"] += policy_loss.item()
                        
                        with self._lock:
                            self.status.step += 1
                    
                    for k in train_losses:
                        train_losses[k] /= num_batches
                
                # 更新状态
                with self._lock:
                    self.status.epoch = epoch + 1
                    self.status.total_games += selfplay_stats["num_games"]
                    self.status.loss = train_losses["total"]
                    self.status.value_loss = train_losses["value"]
                    self.status.policy_loss = train_losses["policy"]
                
                self._notify_subscribers()
                
                if (epoch + 1) % config.save_interval == 0:
                    self._save_checkpoint(epoch + 1, config)
            
            self._save_checkpoint(config.num_epochs, config, is_final=True)
            
        except Exception as e:
            logger.error(f"训练出错: {e}")
            import traceback
            traceback.print_exc()
        finally:
            with self._lock:
                self.status.running = False
            self._notify_subscribers()
    
    def _run_selfplay(
        self,
        game_type: str,
        network,
        device: str,
        num_games: int,
        num_simulations: int,
        c_puct: float,
        dirichlet_alpha: float,
        dirichlet_epsilon: float,
        temperature_threshold: int,
    ) -> tuple:
        """运行自玩生成训练数据（使用完整 MCTS 搜索）
        
        核心改进：
        1. 使用 MCTS 搜索生成改进的策略（而非直接使用网络输出）
        2. 游戏结束后回传所有步骤的奖励
        3. 目标价值使用游戏最终结果
        """
        import torch
        import numpy as np
        from games import make_game, get_game_info
        from core.algorithm import Trajectory
        from core.game import PlayerType
        from core.mcts import LocalMCTSTree
        from core.config import MCTSConfig
        
        # 获取游戏信息
        game_info = get_game_info(game_type)
        player_type = game_info.get("player_type", "two_player")
        num_players = game_info.get("num_players", 2)
        
        trajectories = []
        win_counts = {i: 0 for i in range(num_players)}
        win_counts["draw"] = 0
        total_length = 0
        
        network.eval()
        
        # 创建网络评估函数（供 MCTS 使用）
        def evaluate_fn(obs: np.ndarray, mask: np.ndarray) -> tuple:
            """评估函数：返回策略和价值"""
            with torch.no_grad():
                obs_t = torch.from_numpy(obs).unsqueeze(0).float().to(device)
                policy_logits, value = network(obs_t)
                policy = torch.softmax(policy_logits, dim=-1)[0].cpu().numpy()
                
                # 应用合法动作掩码
                policy = policy * mask
                policy_sum = policy.sum()
                if policy_sum > 1e-8:
                    policy = policy / policy_sum
                else:
                    # 如果所有概率为 0，均匀分布
                    policy = mask / mask.sum()
                
                return policy, value[0].item()
        
        # 创建 MCTS 配置
        mcts_config = MCTSConfig(
            num_simulations=num_simulations,
            c_puct=c_puct,
            dirichlet_alpha=dirichlet_alpha,
            dirichlet_epsilon=dirichlet_epsilon,
            temperature_init=1.0,
            temperature_final=0.1,
            temperature_threshold=temperature_threshold,
        )
        
        # 调试：第一局详细输出
        debug_first_game = (num_games > 0)
        
        for game_idx in range(num_games):
            if self._stop_event.is_set():
                break
            
            game = make_game(game_type)
            game.reset()
            
            trajectory = Trajectory()
            move_count = 0
            
            # 创建 MCTS 树
            mcts_tree = LocalMCTSTree(game, mcts_config, mode="alphazero")
            
            # 调试：第一局输出游戏初始状态
            if debug_first_game and game_idx == 0:
                logger.debug(f"[调试] === 自玩第 1 局开始 ===")
                logger.debug(f"[调试] 游戏: {game_type}, MCTS模拟: {num_simulations}")
            
            while not game.is_terminal():
                current_player = game.current_player()
                obs = game.get_observation()
                legal_actions = game.legal_actions()
                
                # === 使用 MCTS 搜索 ===
                add_noise = (move_count == 0)  # 只在根节点添加噪声
                action, policy_dict, root_value = mcts_tree.search(
                    evaluate_fn=evaluate_fn,
                    num_simulations=num_simulations,
                    add_noise=add_noise,
                )
                
                # 调试：记录 MCTS 搜索结果（每局前3步）
                if game_idx < 3 and move_count < 5:
                    top_actions = sorted(policy_dict.items(), key=lambda x: -x[1])[:5]
                    self._add_debug("mcts", {
                        "game_idx": game_idx,
                        "step": move_count,
                        "player": current_player,
                        "legal_actions": len(legal_actions),
                        "selected_action": action,
                        "root_value": round(root_value, 4),
                        "root_visits": mcts_tree.root.visit_count,
                        "top_actions": [(a, round(p, 4)) for a, p in top_actions],
                    })
                
                # 日志输出
                if debug_first_game and game_idx == 0 and move_count < 3:
                    logger.debug(f"[调试] Step {move_count}: player={current_player}, "
                               f"legal_actions={len(legal_actions)}, "
                               f"selected={action}, value={root_value:.3f}")
                
                # 记录轨迹（使用 MCTS 搜索得到的策略）
                trajectory.append(
                    observation=obs,
                    action=action,
                    reward=0.0,  # 将在游戏结束后设置
                    policy=policy_dict,
                    value=root_value,  # MCTS 搜索得到的价值
                    to_play=current_player,
                )
                
                # 执行动作
                game.step(action)
                move_count += 1
                
                # 更新 MCTS 树（复用子树）
                mcts_tree.game = game
                mcts_tree.advance(action)
            
            # === 游戏结束：回传所有步骤的奖励 ===
            winner = game.get_winner()
            
            # 调试：第一局输出游戏结果
            if debug_first_game and game_idx == 0:
                logger.debug(f"[调试] 游戏结束: steps={move_count}, winner={winner}")
            
            # 设置每一步的奖励（基于最终游戏结果）
            for i, player in enumerate(trajectory.to_play):
                if winner is None:
                    trajectory.rewards[i] = 0.0  # 和棋
                elif player == winner:
                    trajectory.rewards[i] = 1.0  # 胜
                else:
                    trajectory.rewards[i] = -1.0  # 负
            
            # 调试：记录轨迹信息（前几局）
            if game_idx < 5:
                self._add_debug("trajectories", {
                    "game_idx": game_idx,
                    "length": len(trajectory),
                    "actions": trajectory.actions,
                    "rewards": trajectory.rewards,
                    "players": trajectory.to_play,
                    "winner": winner,
                    "values": [round(v, 3) for v in trajectory.values],
                })
                self._add_debug("selfplay", {
                    "game_idx": game_idx,
                    "length": move_count,
                    "winner": winner,
                })
            
            # 日志输出
            if debug_first_game and game_idx == 0:
                logger.debug(f"[调试] 轨迹长度: {len(trajectory)}")
                logger.debug(f"[调试] 动作序列: {trajectory.actions}")
                logger.debug(f"[调试] 奖励序列: {trajectory.rewards}")
            
            trajectories.append(trajectory)
            total_length += move_count
            
            if winner is None:
                win_counts["draw"] += 1
            elif winner in win_counts:
                win_counts[winner] += 1
            
            # 日志
            if (game_idx + 1) % max(1, num_games // 10) == 0:
                logger.debug(f"自玩进度: {game_idx + 1}/{num_games}, "
                           f"当前局长度: {move_count}, 胜者: {winner}")
        
        num_completed = len(trajectories)
        total_games = sum(win_counts.values())
        
        # 构建胜率统计
        win_rate = {"draw": win_counts["draw"] / total_games if total_games > 0 else 0}
        for i in range(num_players):
            win_rate[f"player_{i}"] = win_counts.get(i, 0) / total_games if total_games > 0 else 0
        
        stats = {
            "num_games": num_completed,
            "avg_length": total_length / num_completed if num_completed > 0 else 0,
            "win_rate": win_rate,
            "num_players": num_players,
            "player_type": player_type,
            "avg_simulations": num_simulations,
        }
        
        return trajectories, stats
    
    def _save_checkpoint(self, epoch: int, config, is_final: bool = False):
        """保存检查点"""
        if self._checkpoint_manager is None or self._network is None:
            return
        
        metrics = {
            "epoch": epoch,
            "step": self.status.step,
            "loss": self.status.loss,
            "eval_win_rate": self.status.eval_win_rate,
            "total_games": self.status.total_games,
        }
        
        try:
            path = self._checkpoint_manager.save(
                game_type=config.game_type,
                algorithm=config.algorithm,
                epoch=epoch,
                network=self._network,
                optimizer=self._optimizer,
                config=config.to_dict(),
                metrics=metrics,
            )
            logger.info(f"保存检查点: {path}")
        except Exception as e:
            logger.error(f"保存检查点失败: {e}")
    
    def save_checkpoint(self) -> Dict[str, Any]:
        """手动保存检查点"""
        if not self.status.running:
            return {"error": "训练未运行"}
        
        if self._checkpoint_manager is None or self._network is None:
            return {"error": "训练组件未初始化"}
        
        try:
            from core.training_config import TrainingConfig
            config = TrainingConfig.from_dict(self._config)
            self._save_checkpoint(self.status.epoch, config, is_final=False)
            return {"success": True, "epoch": self.status.epoch}
        except Exception as e:
            return {"error": str(e)}
    
    def pause(self):
        """暂停训练"""
        with self._lock:
            self.status.paused = True
        logger.info("训练已暂停")
        self._notify_subscribers()
    
    def resume(self):
        """恢复训练"""
        with self._lock:
            self.status.paused = False
        logger.info("训练已恢复")
        self._notify_subscribers()
    
    def stop(self):
        """停止训练"""
        self._stop_event.set()
        
        # 停止 DistributedTrainer
        if self._trainer is not None:
            self._trainer.stop()
        
        # 等待训练线程结束
        if self._training_thread and self._training_thread.is_alive():
            self._training_thread.join(timeout=5.0)
        
        with self._lock:
            self.status.running = False
            self.status.paused = False
            self._trainer = None
        logger.info("训练已停止")
        self._notify_subscribers()
    
    def subscribe(self, callback: Callable):
        """订阅状态更新"""
        self._subscribers.add(callback)
    
    def unsubscribe(self, callback: Callable):
        """取消订阅"""
        self._subscribers.discard(callback)
    
    def _notify_subscribers(self):
        """通知所有订阅者"""
        status = self.get_status()
        for callback in self._subscribers:
            try:
                callback(status)
            except Exception as e:
                logger.error(f"通知订阅者失败: {e}")
    
    def get_config(self) -> Dict[str, Any]:
        """获取当前配置"""
        return self._config.copy()
    
    def set_config(self, config: Dict[str, Any]):
        """设置配置"""
        self._config.update(config)


# ============================================================
# 游戏管理器（通用）
# ============================================================

@dataclass
class GameSession:
    """游戏会话（通用）"""
    id: str
    game_type: str
    num_players: int
    players: List[str]  # 通用玩家列表: ["human", "ai:muzero", ...]
    state: str = "waiting"  # waiting | playing | finished
    current_player: int = 0
    step_count: int = 0
    history: List[int] = field(default_factory=list)  # 动作历史
    result: Optional[Dict[str, Any]] = None  # {"winner": 0|1|None, "rewards": {...}}
    created_at: float = field(default_factory=time.time)
    
    # 内部游戏实例
    _game: Any = field(default=None, repr=False)


class GameManager:
    """游戏管理器（通用）
    
    管理对弈会话和调试会话，支持任意游戏类型。
    支持 AI 自动下棋。
    """
    
    def __init__(self):
        self._sessions: Dict[str, GameSession] = {}
        self._subscribers: Dict[str, Set[Callable]] = {}
        self._lock = threading.RLock()
        
        # AI 相关
        self._ai_threads: Dict[str, threading.Thread] = {}
        self._ai_networks: Dict[str, Any] = {}  # 缓存加载的网络
        
        # 调试会话存储（用于 Web 调试界面）
        self.debug_sessions: Dict[str, Dict[str, Any]] = {}
    
    def create_session(
        self,
        game_type: str,
        players: List[str] = None,
    ) -> str:
        """创建游戏会话（通用）"""
        import uuid
        from games import make_game, get_game_info
        
        session_id = str(uuid.uuid4())[:8]
        
        # 获取游戏信息
        try:
            game_info = get_game_info(game_type)
            num_players = game_info.get("num_players", 2)
        except Exception:
            num_players = 2
        
        # 默认玩家
        if players is None:
            players = ["human"] + ["ai:muzero"] * (num_players - 1)
        
        # 确保玩家数量正确
        while len(players) < num_players:
            players.append("ai:muzero")
        players = players[:num_players]
        
        with self._lock:
            self._sessions[session_id] = GameSession(
                id=session_id,
                game_type=game_type,
                num_players=num_players,
                players=players,
            )
            self._subscribers[session_id] = set()
        
        logger.info(f"创建游戏会话: {session_id} ({game_type})")
        return session_id
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话信息"""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return None
            # 使用 _to_python 转换 numpy 类型
            return _to_python({
                "id": session.id,
                "game_type": session.game_type,
                "num_players": session.num_players,
                "players": session.players,
                "state": session.state,
                "current_player": session.current_player,
                "step_count": session.step_count,
                "history": session.history,
                "result": session.result,
            })
    
    def update_session(self, session_id: str, **kwargs):
        """更新会话"""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                for key, value in kwargs.items():
                    if hasattr(session, key):
                        setattr(session, key, value)
        
        self._notify_session_subscribers(session_id)
    
    def do_action(self, session_id: str, action: int, is_ai: bool = False) -> Dict[str, Any]:
        """执行动作（通用）
        
        Args:
            session_id: 会话ID
            action: 动作索引
            is_ai: 是否是 AI 执行的动作（避免递归）
        """
        done = False
        
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return {"error": "会话不存在"}
            if session.state != "playing":
                return {"error": "游戏未开始或已结束"}
            if session._game is None:
                return {"error": "游戏实例未初始化"}
            
            game = session._game
            
            # 检查动作合法性
            if action not in game.legal_actions():
                return {"error": f"非法动作: {action}"}
            
            # 执行动作
            _, reward, done, info = game.step(action)
            
            # 更新会话
            session.history.append(action)
            session.step_count += 1
            session.current_player = game.current_player() if not done else -1
            
            if done:
                session.state = "finished"
                winner = game.get_winner()
                rewards = game.get_rewards()
                session.result = {
                    "winner": winner,
                    "rewards": rewards,
                }
        
        self._notify_session_subscribers(session_id)
        
        # 如果游戏未结束，检查是否需要 AI 下棋
        if not done and not is_ai:
            self._maybe_start_ai_turn(session_id)
        
        # 使用 _to_python 转换 numpy 类型
        return _to_python({"success": True, "action": action, "done": done})
    
    def start_game(self, session_id: str) -> Dict[str, Any]:
        """开始游戏"""
        from games import make_game
        
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return {"error": "会话不存在"}
            
            # 创建游戏实例
            try:
                game = make_game(session.game_type)
                game.reset()
                session._game = game
                session.state = "playing"
                session.current_player = game.current_player()
            except Exception as e:
                return {"error": f"创建游戏失败: {e}"}
        
        self._notify_session_subscribers(session_id)
        
        # 检查是否需要 AI 先手
        self._maybe_start_ai_turn(session_id)
        
        return {"success": True}
    
    def end_game(self, session_id: str, winner: Optional[int], rewards: Dict[int, float]):
        """结束游戏"""
        with self._lock:
            session = self._sessions.get(session_id)
            if session:
                session.state = "finished"
                session.result = {"winner": winner, "rewards": rewards}
        
        self._notify_session_subscribers(session_id)
    
    def _maybe_start_ai_turn(self, session_id: str):
        """检查是否需要启动 AI 下棋"""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or session.state != "playing":
                return
            
            current_player_idx = session.current_player
            if current_player_idx < 0 or current_player_idx >= len(session.players):
                return
            
            player_type = session.players[current_player_idx]
        
        # 检查是否是 AI 玩家
        if player_type == "human":
            return
        
        # 避免重复启动
        if session_id in self._ai_threads:
            thread = self._ai_threads[session_id]
            if thread.is_alive():
                return
        
        # 启动 AI 线程
        thread = threading.Thread(
            target=self._ai_move,
            args=(session_id, player_type),
            daemon=True,
        )
        self._ai_threads[session_id] = thread
        thread.start()
    
    def _ai_move(self, session_id: str, player_type: str):
        """AI 执行下棋动作"""
        import random
        import time
        
        # 短暂延迟，让界面更新
        time.sleep(0.2)
        
        # 获取游戏状态（在锁内）
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or session.state != "playing":
                self._cleanup_ai_thread(session_id)
                return
            game = session._game
            if game is None:
                self._cleanup_ai_thread(session_id)
                return
            
            legal_actions = game.legal_actions()
            if not legal_actions:
                self._cleanup_ai_thread(session_id)
                return
        
        action = None
        
        try:
            if player_type == "random":
                # 随机策略
                action = random.choice(legal_actions)
            
            elif player_type.startswith("checkpoint:"):
                # 使用检查点进行 MCTS
                checkpoint_path = player_type[len("checkpoint:"):]
                action = self._ai_move_with_checkpoint(session_id, checkpoint_path, legal_actions)
            
            elif player_type.startswith("ai:"):
                # 尝试加载最新检查点，否则使用随机
                algo = player_type[len("ai:"):]
                action = self._ai_move_with_algo(session_id, algo, legal_actions)
            
            else:
                # 默认随机
                action = random.choice(legal_actions)
        
        except Exception as e:
            logger.error(f"AI 下棋失败: {e}")
            action = random.choice(legal_actions) if legal_actions else None
        
        if action is not None:
            # 执行动作（标记为 AI 动作）
            result = self.do_action(session_id, action, is_ai=True)
            
            # 清理线程引用
            self._cleanup_ai_thread(session_id)
            
            # 如果成功且游戏未结束，触发下一个 AI
            if result.get("success") and not result.get("done"):
                self._maybe_start_ai_turn(session_id)
        else:
            self._cleanup_ai_thread(session_id)
    
    def _cleanup_ai_thread(self, session_id: str):
        """清理 AI 线程引用"""
        with self._lock:
            if session_id in self._ai_threads:
                del self._ai_threads[session_id]
    
    def _ai_move_with_checkpoint(
        self, 
        session_id: str, 
        checkpoint_path: str, 
        legal_actions: List[int],
        temperature: float = 0.5,  # 中等温度，增加随机性
    ) -> int:
        """使用检查点进行 AI 下棋
        
        Args:
            session_id: 会话 ID
            checkpoint_path: 检查点路径
            legal_actions: 合法动作列表
            temperature: 动作采样温度（0=贪婪，1=按概率采样，>1=更随机）
        """
        import torch
        import numpy as np
        import random
        
        try:
            from core.checkpoint import CheckpointManager
            from core.training_config import resolve_device
            from algorithms import make_algorithm
            from games import make_game
            
            # 加载检查点
            manager = CheckpointManager()
            checkpoint = manager.load(checkpoint_path)
            
            game_type = checkpoint.get("game_type")
            algo_type = checkpoint.get("algorithm")
            device = resolve_device("auto")
            
            # 获取或创建网络
            cache_key = checkpoint_path
            if cache_key not in self._ai_networks:
                algo = make_algorithm(algo_type)
                game_template = make_game(game_type)
                network = algo.create_network(game_template).to(device)
                network.load_state_dict(checkpoint["network_state_dict"])
                network.eval()
                self._ai_networks[cache_key] = network
            
            network = self._ai_networks[cache_key]
            
            # 获取当前观测
            with self._lock:
                session = self._sessions.get(session_id)
                if session is None:
                    return random.choice(legal_actions)
                game = session._game
                obs = game.get_observation()
            
            # 使用网络预测
            with torch.no_grad():
                obs_t = torch.from_numpy(obs).unsqueeze(0).float().to(device)
                policy_logits, value = network(obs_t)
                policy = torch.softmax(policy_logits, dim=-1)[0].cpu().numpy()
            
            # 使用温度采样选择动作（而不是贪婪选择）
            legal_probs = np.array([policy[a] for a in legal_actions])
            
            if temperature < 0.01:
                # 温度接近 0，贪婪选择
                action = legal_actions[np.argmax(legal_probs)]
            else:
                # 应用温度
                legal_probs = np.power(legal_probs + 1e-8, 1.0 / temperature)
                legal_probs = legal_probs / legal_probs.sum()
                # 按概率采样
                action = np.random.choice(legal_actions, p=legal_probs)
            
            return action
            
        except Exception as e:
            logger.error(f"检查点 AI 失败: {e}")
            return random.choice(legal_actions)
    
    def _ai_move_with_algo(self, session_id: str, algo: str, legal_actions: List[int]) -> int:
        """尝试使用指定算法的最新检查点"""
        import random
        
        try:
            from core.checkpoint import CheckpointManager
            
            with self._lock:
                session = self._sessions.get(session_id)
                if session is None:
                    return random.choice(legal_actions)
                game_type = session.game_type
            
            # 查找检查点
            manager = CheckpointManager()
            checkpoints = manager.list_checkpoints(game_type=game_type, algorithm=algo)
            
            if checkpoints:
                # 使用最新检查点
                best = checkpoints[0]  # 已按时间排序
                return self._ai_move_with_checkpoint(session_id, best.path, legal_actions)
            
        except Exception as e:
            logger.debug(f"未找到 {algo} 检查点: {e}")
        
        # 回退到随机
        return random.choice(legal_actions)
    
    def get_render(self, session_id: str, mode: str = "json") -> Dict[str, Any]:
        """获取游戏渲染数据"""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return {"error": "会话不存在"}
            if session._game is None:
                return {"error": "游戏未开始"}
            
            try:
                render_data = session._game.render(mode=mode)
                return _to_python({"render": render_data})
            except Exception as e:
                return {"error": f"渲染失败: {e}"}
    
    def get_legal_actions(self, session_id: str) -> Dict[str, Any]:
        """获取合法动作列表"""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None:
                return {"error": "会话不存在"}
            if session._game is None:
                return {"error": "游戏未开始"}
            
            try:
                actions = session._game.legal_actions()
                # 使用 _to_python 转换 numpy 类型
                return _to_python({"actions": actions})
            except Exception as e:
                return {"error": f"获取合法动作失败: {e}"}
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """列出所有会话"""
        with self._lock:
            return [self.get_session(sid) for sid in self._sessions]
    
    def delete_session(self, session_id: str) -> Dict[str, Any]:
        """删除会话"""
        with self._lock:
            if session_id not in self._sessions:
                return {"error": "会话不存在"}
            
            # 停止 AI 线程
            if session_id in self._ai_threads:
                del self._ai_threads[session_id]
            
            # 删除会话
            del self._sessions[session_id]
            
            # 删除订阅者
            if session_id in self._subscribers:
                del self._subscribers[session_id]
        
        logger.info(f"删除游戏会话: {session_id}")
        return {"success": True}
    
    def clear_all_sessions(self) -> Dict[str, Any]:
        """清空所有会话"""
        with self._lock:
            count = len(self._sessions)
            # 停止所有 AI 线程
            self._ai_threads.clear()
            # 清空会话
            self._sessions.clear()
            # 清空订阅者
            self._subscribers.clear()
        
        logger.info(f"清空所有游戏会话: 共 {count} 个")
        return {"success": True, "deleted_count": count}
    
    def subscribe(self, session_id: str, callback: Callable):
        """订阅会话更新"""
        with self._lock:
            if session_id in self._subscribers:
                self._subscribers[session_id].add(callback)
    
    def unsubscribe(self, session_id: str, callback: Callable):
        """取消订阅"""
        with self._lock:
            if session_id in self._subscribers:
                self._subscribers[session_id].discard(callback)
    
    def _notify_session_subscribers(self, session_id: str):
        """通知会话订阅者"""
        session_data = self.get_session(session_id)
        if session_data is None:
            return
        
        with self._lock:
            subscribers = self._subscribers.get(session_id, set()).copy()
        
        for callback in subscribers:
            try:
                callback(session_data)
            except Exception as e:
                logger.error(f"通知订阅者失败: {e}")


# ============================================================
# 系统管理器
# ============================================================

class SystemManager:
    """系统管理器
    
    管理全局配置和系统状态。
    使用 TrainingConfig 作为配置结构。
    """
    
    def __init__(self, config_path: str = "./config.json"):
        self.config_path = Path(config_path)
        self._config: Dict[str, Any] = self._load_config()
        self._lock = threading.RLock()  # 可重入锁
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置"""
        if self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    loaded = json.load(f)
                    # 合并默认配置和加载的配置
                    default = self._default_config()
                    default.update(loaded)
                    return default
            except Exception as e:
                logger.error(f"加载配置失败: {e}")
        
        return self._default_config()
    
    def _default_config(self) -> Dict[str, Any]:
        """默认配置 - 使用 TrainingConfig 结构"""
        from core.training_config import TrainingConfig
        return TrainingConfig().to_dict()
    
    def get_config(self, section: str = None) -> Dict[str, Any]:
        """获取配置
        
        Args:
            section: 配置分组名称，None 返回全部配置
        """
        with self._lock:
            if section:
                # 按配置组返回
                from core.training_config import CONFIG_GROUPS
                if section in CONFIG_GROUPS:
                    fields = CONFIG_GROUPS[section]["fields"]
                    return {k: self._config.get(k) for k in fields if k in self._config}
                return {}
            return self._config.copy()
    
    def set_config(self, section: str, values: Dict[str, Any]):
        """设置配置
        
        Args:
            section: 配置分组名称
            values: 要更新的配置值
        """
        with self._lock:
            # 直接更新扁平配置
            self._config.update(values)
            self._save_config()
    
    def update_config(self, values: Dict[str, Any]):
        """批量更新配置"""
        with self._lock:
            self._config.update(values)
            self._save_config()
    
    def _save_config(self):
        """保存配置"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self._config, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存配置失败: {e}")
    
    def get_training_config(self) -> "TrainingConfig":
        """获取训练配置对象"""
        from core.training_config import TrainingConfig
        return TrainingConfig.from_dict(self._config)
    
    def get_config_schema(self) -> Dict[str, Any]:
        """获取配置 schema（用于 Web 界面）"""
        from core.training_config import get_config_schema
        return get_config_schema()
    
    def get_system_info(self) -> Dict[str, Any]:
        """获取系统信息"""
        import platform
        from core.training_config import get_best_device
        
        info = {
            "platform": platform.system(),
            "python_version": platform.python_version(),
            "best_device": get_best_device(),
        }
        
        try:
            import torch
            info["torch_version"] = torch.__version__
            info["cuda_available"] = torch.cuda.is_available()
            info["mps_available"] = hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
            
            if torch.cuda.is_available():
                info["cuda_version"] = torch.version.cuda
                info["gpu_count"] = torch.cuda.device_count()
                info["gpu_name"] = torch.cuda.get_device_name(0)
                info["gpu_memory"] = torch.cuda.get_device_properties(0).total_memory // (1024 ** 3)
        except ImportError:
            info["torch_version"] = "未安装"
            info["cuda_available"] = False
            info["mps_available"] = False
        
        return info
    
    def list_games(self) -> List[Dict[str, Any]]:
        """列出可用游戏"""
        try:
            from games import list_games, get_game_info
            return [get_game_info(name) for name in list_games()]
        except Exception as e:
            logger.error(f"获取游戏列表失败: {e}")
            return []
    
    def list_algorithms(self) -> List[Dict[str, Any]]:
        """列出可用算法"""
        try:
            from algorithms import list_algorithms, ALGORITHM_REGISTRY
            result = []
            for name in list_algorithms():
                algo_cls = ALGORITHM_REGISTRY[name]
                # needs_dynamics 是 @property，需要实例化才能获取值
                try:
                    algo_instance = algo_cls({})
                    needs_dynamics = algo_instance.needs_dynamics
                except Exception:
                    needs_dynamics = False
                result.append({
                    "name": name,
                    "class": algo_cls.__name__,
                    "needs_dynamics": needs_dynamics,
                })
            return result
        except Exception as e:
            logger.error(f"获取算法列表失败: {e}")
            return []


# ============================================================
# 调试管理器 - 逐步调试训练流程
# ============================================================

@dataclass
class DebugSession:
    """调试会话"""
    session_id: str
    game_type: str
    algorithm: str
    device: str
    
    # 游戏实例
    game: Any = None
    # 网络
    network: Any = None
    # MCTS 树
    mcts_tree: Any = None
    # 当前轨迹
    trajectory: Any = None
    
    # 历史记录
    history: List[Dict[str, Any]] = field(default_factory=list)
    
    # 状态
    step: int = 0
    game_count: int = 0
    is_terminal: bool = False
    
    # 最后 MCTS 搜索结果（用于 step_game 使用）
    last_mcts_action: Optional[int] = None
    last_mcts_policy: Optional[Dict[int, float]] = None
    last_mcts_value: Optional[float] = None


class DebugManager:
    """调试管理器 - 支持逐步调试训练流程"""
    
    def __init__(self):
        self._sessions: Dict[str, DebugSession] = {}
        self._lock = threading.Lock()
    
    def create_session(
        self,
        game_type: str,
        algorithm: str = "alphazero",
        device: str = "cpu",
        checkpoint_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """创建调试会话
        
        Args:
            game_type: 游戏类型
            algorithm: 算法名称
            device: 计算设备
            checkpoint_path: 可选的检查点路径
        """
        import torch
        from games import make_game
        from algorithms import make_algorithm
        from core.replay_buffer import Trajectory
        from core.mcts import MCTSConfig, LocalMCTSTree
        
        session_id = str(uuid.uuid4())[:8]
        
        try:
            # 创建游戏
            game = make_game(game_type)
            game.reset()
            
            # 创建算法和网络
            algo = make_algorithm(algorithm)
            network = algo.create_network(game)
            
            # 加载检查点
            if checkpoint_path:
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
                # 兼容多种检查点格式
                state_dict = checkpoint.get("network_state_dict") or checkpoint.get("network") or checkpoint
                network.load_state_dict(state_dict)
                epoch_info = checkpoint.get("epoch", "?")
                logger.info(f"加载检查点: {checkpoint_path} (Epoch {epoch_info})")
            
            network = network.to(device)
            network.eval()
            
            # 创建 MCTS 配置和树
            mcts_config = MCTSConfig(
                num_simulations=50,
                c_puct=1.5,
                dirichlet_alpha=0.3,
                dirichlet_epsilon=0.25,
            )
            mcts_tree = LocalMCTSTree(game, mcts_config, mode="alphazero")
            
            # 创建轨迹
            trajectory = Trajectory()
            
            # 创建会话
            session = DebugSession(
                session_id=session_id,
                game_type=game_type,
                algorithm=algorithm,
                device=device,
                game=game,
                network=network,
                mcts_tree=mcts_tree,
                trajectory=trajectory,
            )
            
            with self._lock:
                self._sessions[session_id] = session
            
            logger.info(f"创建调试会话: {session_id}, game={game_type}, algo={algorithm}")
            
            return {
                "session_id": session_id,
                "game_type": game_type,
                "algorithm": algorithm,
                "device": device,
                "state": self._get_session_state(session),
            }
            
        except Exception as e:
            logger.error(f"创建调试会话失败: {e}")
            raise
    
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """获取会话状态"""
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return None
            return {
                "session_id": session_id,
                "game_type": session.game_type,
                "algorithm": session.algorithm,
                "step": session.step,
                "game_count": session.game_count,
                "is_terminal": session.is_terminal,
                "state": self._get_session_state(session),
                "history": session.history[-20:],  # 最近 20 条
            }
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """列出所有会话"""
        with self._lock:
            return [
                {
                    "session_id": s.session_id,
                    "game_type": s.game_type,
                    "algorithm": s.algorithm,
                    "step": s.step,
                    "game_count": s.game_count,
                    "is_terminal": s.is_terminal,
                }
                for s in self._sessions.values()
            ]
    
    def delete_session(self, session_id: str) -> bool:
        """删除会话"""
        with self._lock:
            if session_id in self._sessions:
                del self._sessions[session_id]
                logger.info(f"删除调试会话: {session_id}")
                return True
            return False
    
    def step_mcts(self, session_id: str, num_simulations: int = 1) -> Dict[str, Any]:
        """执行 MCTS 搜索步骤
        
        Args:
            session_id: 会话 ID
            num_simulations: 模拟次数
        
        Returns:
            MCTS 搜索结果详情
        """
        import torch
        
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                raise ValueError(f"会话不存在: {session_id}")
            
            if session.is_terminal:
                return {"error": "游戏已结束", "is_terminal": True}
            
            game = session.game
            network = session.network
            mcts_tree = session.mcts_tree
            device = session.device
            
            # 评估函数
            def evaluate_fn(obs, legal_mask):
                with torch.no_grad():
                    obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                    mask = np.array(legal_mask, dtype=np.float32)
                    
                    policy_logits, value = network(obs_t)
                    policy = torch.softmax(policy_logits, dim=-1)[0].cpu().numpy()
                    
                    policy = policy * mask
                    policy_sum = policy.sum()
                    if policy_sum > 1e-8:
                        policy = policy / policy_sum
                    else:
                        policy = mask / mask.sum()
                    
                    return policy, value[0].item()
            
            # 执行 MCTS 搜索
            _, policy_dict, root_value = mcts_tree.search(
                evaluate_fn=evaluate_fn,
                num_simulations=num_simulations,
                add_noise=(session.step == 0),
            )
            
            # 获取根节点详情
            import math
            root = mcts_tree.root
            
            # 调试模式：使用确定性选择（访问次数最多的动作）
            action = root.get_best_action()
            sqrt_parent_visits = math.sqrt(root.visit_count + 1)
            c_puct = mcts_tree.config.c_puct
            
            children_info = []
            for a, child in root.children.items():
                # 计算 UCB score
                q_value = -child.q_value if child.visit_count > 0 else 0
                prior_score = c_puct * child.prior * sqrt_parent_visits / (1 + child.visit_count)
                ucb = q_value + prior_score
                
                children_info.append({
                    "action": a,
                    "visit_count": child.visit_count,
                    "value": round(child.q_value, 4) if child.visit_count > 0 else 0,
                    "prior": round(child.prior, 4),
                    "ucb": round(ucb, 4),
                })
            children_info.sort(key=lambda x: -x["visit_count"])
            
            result = _to_python({
                "step": session.step,
                "current_player": game.current_player(),
                "legal_actions": game.legal_actions(),
                "selected_action": action,
                "root_value": round(root_value, 4),
                "root_visits": root.visit_count,
                "policy": {a: round(p, 4) for a, p in sorted(policy_dict.items(), key=lambda x: -x[1])[:10]},
                "children": children_info[:10],
                "game_render": game.render("text"),
            })
            
            # 保存 MCTS 搜索结果，供 step_game 使用
            session.last_mcts_action = action
            session.last_mcts_policy = policy_dict
            session.last_mcts_value = root_value
            
            # 记录历史
            session.history.append({
                "type": "mcts",
                "step": session.step,
                **result,
            })
            
            return result
    
    def step_game(self, session_id: str, action: Optional[int] = None) -> Dict[str, Any]:
        """执行一步游戏
        
        Args:
            session_id: 会话 ID
            action: 动作（None 则使用 MCTS 选择）
        
        Returns:
            执行结果
        """
        import torch
        
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                raise ValueError(f"会话不存在: {session_id}")
            
            if session.is_terminal:
                return {"error": "游戏已结束", "is_terminal": True}
            
            game = session.game
            mcts_tree = session.mcts_tree
            trajectory = session.trajectory
            
            current_player = game.current_player()
            obs = game.get_observation()
            legal_actions = game.legal_actions()
            
            # 如果没有指定动作
            if action is None:
                # 优先使用之前 MCTS 搜索的结果
                if session.last_mcts_action is not None:
                    action = session.last_mcts_action
                    policy_dict = session.last_mcts_policy or {action: 1.0}
                    root_value = session.last_mcts_value or 0.0
                    logger.debug(f"使用缓存的 MCTS 结果: action={action}")
                else:
                    # 否则执行新的 MCTS 搜索
                    network = session.network
                    device = session.device
                    
                    def evaluate_fn(obs, legal_mask):
                        with torch.no_grad():
                            obs_t = torch.from_numpy(obs).float().unsqueeze(0).to(device)
                            mask = np.array(legal_mask, dtype=np.float32)
                            
                            policy_logits, value = network(obs_t)
                            policy = torch.softmax(policy_logits, dim=-1)[0].cpu().numpy()
                            
                            policy = policy * mask
                            policy_sum = policy.sum()
                            if policy_sum > 1e-8:
                                policy = policy / policy_sum
                            else:
                                policy = mask / mask.sum()
                            
                            return policy, value[0].item()
                    
                    action, policy_dict, root_value = mcts_tree.search(
                        evaluate_fn=evaluate_fn,
                        num_simulations=50,
                        add_noise=(session.step == 0),
                    )
            else:
                if action not in legal_actions:
                    return {"error": f"非法动作: {action}", "legal_actions": legal_actions}
                policy_dict = {action: 1.0}
                root_value = 0.0
            
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
            session.step += 1
            
            # 更新 MCTS 树
            mcts_tree.game = game
            mcts_tree.advance(action)
            
            # 清除缓存的 MCTS 结果（下一步需要重新搜索）
            session.last_mcts_action = None
            session.last_mcts_policy = None
            session.last_mcts_value = None
            
            # 检查游戏是否结束
            is_terminal = game.is_terminal()
            session.is_terminal = is_terminal
            
            result = {
                "step": session.step,
                "action": action,
                "previous_player": current_player,
                "current_player": game.current_player() if not is_terminal else None,
                "is_terminal": is_terminal,
                "winner": game.get_winner() if is_terminal else None,
                "game_render": game.render("text"),
                "policy": {a: round(p, 4) for a, p in sorted(policy_dict.items(), key=lambda x: -x[1])[:5]},
                "value": round(root_value, 4),
            }
            
            # 如果游戏结束，更新轨迹奖励
            if is_terminal:
                winner = game.get_winner()
                for i, player in enumerate(trajectory.to_play):
                    if winner is None:
                        trajectory.rewards[i] = 0.0
                    elif player == winner:
                        trajectory.rewards[i] = 1.0
                    else:
                        trajectory.rewards[i] = -1.0
                
                result["trajectory_summary"] = {
                    "length": len(trajectory),
                    "actions": trajectory.actions,
                    "rewards": trajectory.rewards,
                    "values": [round(v, 3) for v in trajectory.values],
                }
            
            result = _to_python(result)
            
            # 记录历史
            session.history.append({
                "type": "step",
                **result,
            })
            
            return result
    
    def reset_game(self, session_id: str) -> Dict[str, Any]:
        """重置游戏
        
        Args:
            session_id: 会话 ID
        """
        from core.mcts import MCTSConfig, LocalMCTSTree
        from core.replay_buffer import Trajectory
        
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                raise ValueError(f"会话不存在: {session_id}")
            
            # 重置游戏
            session.game.reset()
            session.step = 0
            session.game_count += 1
            session.is_terminal = False
            
            # 重置 MCTS 树
            mcts_config = MCTSConfig(
                num_simulations=50,
                c_puct=1.5,
                dirichlet_alpha=0.3,
                dirichlet_epsilon=0.25,
            )
            session.mcts_tree = LocalMCTSTree(session.game, mcts_config, mode="alphazero")
            
            # 重置轨迹
            session.trajectory = Trajectory()
            
            # 清空历史
            session.history = []
            
            # 清除 MCTS 缓存
            session.last_mcts_action = None
            session.last_mcts_policy = None
            session.last_mcts_value = None
            
            logger.info(f"重置调试会话: {session_id}, game_count={session.game_count}")
            
            return {
                "session_id": session_id,
                "game_count": session.game_count,
                "state": self._get_session_state(session),
            }
    
    def run_full_game(self, session_id: str) -> Dict[str, Any]:
        """运行完整一局游戏
        
        Args:
            session_id: 会话 ID
        """
        results = []
        
        # 先重置
        self.reset_game(session_id)
        
        # 运行直到结束
        while True:
            result = self.step_game(session_id)
            results.append(result)
            
            if result.get("is_terminal") or result.get("error"):
                break
            
            if len(results) > 1000:  # 防止死循环
                break
        
        return {
            "steps": len(results),
            "final_result": results[-1] if results else None,
            "all_actions": [r.get("action") for r in results],
        }
    
    def _get_session_state(self, session: DebugSession) -> Dict[str, Any]:
        """获取会话的游戏状态"""
        game = session.game
        
        return _to_python({
            "current_player": game.current_player() if not session.is_terminal else None,
            "legal_actions": game.legal_actions() if not session.is_terminal else [],
            "is_terminal": session.is_terminal,
            "winner": game.get_winner() if session.is_terminal else None,
            "game_render": game.render("text"),
            "observation_shape": list(game.observation_space.shape),
            "action_space": game.action_space.n,
        })


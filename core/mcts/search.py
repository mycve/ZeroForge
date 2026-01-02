"""
MCTSSearch - MCTS 搜索控制器

整合 LocalMCTSTree 和 LeafBatcher，提供完整的搜索接口。

支持:
- 同步搜索（单线程，直接调用网络）
- 异步搜索（多线程，通过 batcher 批推理）
"""

from __future__ import annotations
from typing import Tuple, Dict, Optional, Any, Callable, TYPE_CHECKING
import numpy as np
import logging

from .node import MCTSNode
from .tree import LocalMCTSTree
from .batcher import LeafBatcher
from ..config import MCTSConfig

if TYPE_CHECKING:
    from core.game import Game

logger = logging.getLogger(__name__)


class MCTSSearch:
    """MCTS 搜索控制器
    
    封装 MCTS 搜索的完整流程，支持同步和异步两种模式。
    
    同步模式:
        直接调用 evaluate_fn 进行推理，适用于单线程场景。
        
    异步模式:
        通过 LeafBatcher 提交请求，适用于多线程 nogil 场景。
        
    Example (同步):
        >>> search = MCTSSearch(game, config)
        >>> action, policy, value = search.run(evaluate_fn)
        >>> game.step(action)
        >>> search.advance(action)
        
    Example (异步):
        >>> search = MCTSSearch(game, config, batcher=leaf_batcher)
        >>> action, policy, value = search.run_async()
        >>> game.step(action)
        >>> search.advance(action)
    """
    
    def __init__(
        self,
        game: Game,
        config: MCTSConfig,
        batcher: Optional[LeafBatcher] = None,
        mode: str = "alphazero",
    ):
        """初始化搜索控制器
        
        Args:
            game: 游戏实例
            config: MCTS 配置
            batcher: 可选的批推理器（用于异步模式）
            mode: 搜索模式 ('alphazero' | 'muzero')
        """
        self.game = game
        self.config = config
        self.batcher = batcher
        self.mode = mode
        
        # 创建本地树
        self.tree = LocalMCTSTree(game, config, mode)
        
        # 搜索状态
        self._move_count = 0
    
    # ========================================
    # 同步搜索
    # ========================================
    
    def run(
        self,
        evaluate_fn: Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, float]],
        num_simulations: Optional[int] = None,
        add_noise: bool = True,
    ) -> Tuple[int, Dict[int, float], float]:
        """执行同步 MCTS 搜索
        
        Args:
            evaluate_fn: 评估函数 (observation, mask) -> (policy, value)
            num_simulations: 模拟次数
            add_noise: 是否添加探索噪声
            
        Returns:
            (action, policy, value)
        """
        if num_simulations is None:
            num_simulations = self.config.num_simulations
        
        # 获取温度
        temperature = self.config.get_temperature(self._move_count)
        
        # 确保根节点已扩展
        if not self.tree.root.is_expanded:
            self._expand_root(evaluate_fn, add_noise)
            num_simulations -= 1
        
        # 搜索循环
        for _ in range(num_simulations):
            self._simulate_once(evaluate_fn)
        
        # 获取结果
        action = self.tree.get_action(temperature)
        policy = self.tree.get_policy(temperature)
        value = self.tree.get_root_value()
        
        return action, policy, value
    
    def _expand_root(
        self,
        evaluate_fn: Callable,
        add_noise: bool,
    ) -> None:
        """扩展根节点"""
        game_clone = self.game.clone()
        obs = game_clone.get_observation()
        mask = game_clone.get_legal_actions_mask()
        
        policy, value = evaluate_fn(obs, mask)
        
        self.tree.expand(
            self.tree.root,
            game_clone,
            policy,
            value,
            legal_actions=game_clone.legal_actions(),
        )
        self.tree.backup(self.tree.root, value)
        
        if add_noise:
            self.tree.add_exploration_noise()
    
    def _simulate_once(self, evaluate_fn: Callable) -> None:
        """执行一次模拟"""
        # 选择叶子节点
        node, game_clone = self.tree.select()
        
        if game_clone is None:
            raise RuntimeError("AlphaZero 模式需要 game_clone")
        
        # 检查终局
        if game_clone.is_terminal():
            value = self._get_terminal_value(game_clone)
            self.tree.backup(node, value)
            return
        
        # 评估叶子节点
        obs = game_clone.get_observation()
        mask = game_clone.get_legal_actions_mask()
        policy, value = evaluate_fn(obs, mask)
        
        # 扩展和回传
        self.tree.expand(
            node,
            game_clone,
            policy,
            value,
            legal_actions=game_clone.legal_actions(),
        )
        self.tree.backup(node, value)
    
    def _get_terminal_value(self, game: Game) -> float:
        """获取终局价值"""
        winner = game.get_winner()
        if winner is None:
            return 0.0  # 和棋
        
        # 从叶子节点当前玩家视角
        # 注意：终局时 current_player 可能不准确，使用 winner 判断
        # 这里假设调用时是从父节点的对手视角
        return 1.0 if winner == game.current_player() else -1.0
    
    # ========================================
    # 异步搜索
    # ========================================
    
    def run_async(
        self,
        num_simulations: Optional[int] = None,
        add_noise: bool = True,
    ) -> Tuple[int, Dict[int, float], float]:
        """执行异步 MCTS 搜索
        
        通过 LeafBatcher 进行批量推理。
        
        Args:
            num_simulations: 模拟次数
            add_noise: 是否添加探索噪声
            
        Returns:
            (action, policy, value)
        """
        if self.batcher is None:
            raise RuntimeError("异步搜索需要 LeafBatcher")
        
        if num_simulations is None:
            num_simulations = self.config.num_simulations
        
        temperature = self.config.get_temperature(self._move_count)
        
        # 确保根节点已扩展
        if not self.tree.root.is_expanded:
            self._expand_root_async(add_noise)
            num_simulations -= 1
        
        # 搜索循环
        for _ in range(num_simulations):
            self._simulate_once_async()
        
        # 获取结果
        action = self.tree.get_action(temperature)
        policy = self.tree.get_policy(temperature)
        value = self.tree.get_root_value()
        
        return action, policy, value
    
    def _expand_root_async(self, add_noise: bool) -> None:
        """异步扩展根节点"""
        game_clone = self.game.clone()
        obs = game_clone.get_observation()
        mask = game_clone.get_legal_actions_mask()
        
        # 通过 batcher 提交
        policy, value = self.batcher.submit(obs, mask)
        
        self.tree.expand(
            self.tree.root,
            game_clone,
            policy,
            value,
            legal_actions=game_clone.legal_actions(),
        )
        self.tree.backup(self.tree.root, value)
        
        if add_noise:
            self.tree.add_exploration_noise()
    
    def _simulate_once_async(self) -> None:
        """异步执行一次模拟"""
        node, game_clone = self.tree.select()
        
        if game_clone is None:
            raise RuntimeError("AlphaZero 模式需要 game_clone")
        
        # 检查终局
        if game_clone.is_terminal():
            value = self._get_terminal_value(game_clone)
            self.tree.backup(node, value)
            return
        
        # 通过 batcher 提交
        obs = game_clone.get_observation()
        mask = game_clone.get_legal_actions_mask()
        policy, value = self.batcher.submit(obs, mask)
        
        # 扩展和回传
        self.tree.expand(
            node,
            game_clone,
            policy,
            value,
            legal_actions=game_clone.legal_actions(),
        )
        self.tree.backup(node, value)
    
    # ========================================
    # 状态管理
    # ========================================
    
    def advance(self, action: int) -> bool:
        """执行动作并更新树
        
        Args:
            action: 执行的动作
            
        Returns:
            是否成功复用子树
        """
        self._move_count += 1
        return self.tree.advance(action)
    
    def reset(self) -> None:
        """重置搜索状态"""
        self.tree.reset()
        self._move_count = 0
    
    def sync_game(self, game: Game) -> None:
        """同步游戏状态
        
        当外部游戏状态改变时调用。
        
        Args:
            game: 新的游戏状态
        """
        self.game = game
        self.tree.game = game
    
    # ========================================
    # 统计
    # ========================================
    
    def get_stats(self) -> Dict[str, Any]:
        """获取搜索统计"""
        return {
            "move_count": self._move_count,
            "tree": self.tree.get_stats(),
            "mode": self.mode,
        }
    
    def print_tree(self, max_depth: int = 3) -> None:
        """打印树结构"""
        self.tree.print_tree(max_depth)
    
    def __repr__(self) -> str:
        return (
            f"MCTSSearch(mode={self.mode}, "
            f"move={self._move_count}, "
            f"root_visits={self.tree.root.visit_count})"
        )


# ============================================================
# 便捷函数
# ============================================================

def run_mcts_search(
    game: Game,
    evaluate_fn: Callable,
    num_simulations: int = 200,
    c_puct: float = 1.5,
    temperature: float = 1.0,
    add_noise: bool = True,
) -> Tuple[int, Dict[int, float], float]:
    """便捷的 MCTS 搜索函数
    
    一次性搜索，不保留树。
    
    Args:
        game: 游戏实例
        evaluate_fn: 评估函数
        num_simulations: 模拟次数
        c_puct: 探索常数
        temperature: 采样温度
        add_noise: 是否添加噪声
        
    Returns:
        (action, policy, value)
    """
    config = MCTSConfig(
        num_simulations=num_simulations,
        c_puct=c_puct,
        temperature_init=temperature,
        temperature_final=temperature,
    )
    
    search = MCTSSearch(game, config)
    return search.run(evaluate_fn, add_noise=add_noise)

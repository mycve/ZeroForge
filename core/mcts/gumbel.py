"""
Gumbel MCTS - 基于 Gumbel 的 MCTS 搜索

实现论文: "Policy improvement by planning with Gumbel" (ICLR 2022)

核心改进:
1. 根节点: 使用 Gumbel-Top-k 采样选择候选动作
2. 预算分配: 使用 Sequential Halving 分配模拟次数
3. 树搜索: 对每个候选动作执行标准 MCTS 树搜索
4. 策略改进: 基于搜索结果和 Completed Q-values

与标准 MCTS 的区别:
- 标准 MCTS: UCB 逐步探索所有动作
- Gumbel MCTS: 先选 k 个候选，再集中搜索
"""

from __future__ import annotations
import math
from typing import Tuple, Dict, Optional, List, Any, Callable, TYPE_CHECKING
import numpy as np
import logging

from .node import MCTSNode
from .tree import LocalMCTSTree
from .batcher import LeafBatcher
from ..config import MCTSConfig

if TYPE_CHECKING:
    from core.game import Game

logger = logging.getLogger(__name__)


class GumbelMCTSSearch:
    """Gumbel MCTS 搜索
    
    结合 Gumbel-Top-k 采样和 MCTS 树搜索。
    
    流程:
    1. 在根节点用 Gumbel 噪声选择 k 个候选动作
    2. 用 Sequential Halving 分配模拟预算
    3. 对每个候选动作执行 MCTS 树搜索
    4. 基于搜索结果计算改进策略
    
    Example:
        >>> search = GumbelMCTSSearch(game, config, batcher=batcher)
        >>> action, policy, value = search.run()
    """
    
    def __init__(
        self,
        game: Game,
        config: MCTSConfig,
        batcher: Optional[LeafBatcher] = None,
        mode: str = "alphazero",
        # Gumbel 特有参数
        max_considered_actions: int = 16,
        gumbel_scale: float = 1.0,
    ):
        """初始化 Gumbel MCTS 搜索
        
        Args:
            game: 游戏实例
            config: MCTS 配置
            batcher: 批推理器（可选）
            mode: 搜索模式 ('alphazero' | 'muzero')
            max_considered_actions: Gumbel Top-k 考虑的最大动作数
            gumbel_scale: Gumbel 噪声缩放
        """
        self.game = game
        self.config = config
        self.batcher = batcher
        self.mode = mode
        
        # Gumbel 参数
        self.max_considered_actions = max_considered_actions
        self.gumbel_scale = gumbel_scale
        
        # 自动计算 halving 轮数
        self.num_halving_rounds = max(1, math.ceil(math.log2(max_considered_actions)))
        
        # 创建 MCTS 树
        self.tree = LocalMCTSTree(game, config, mode)
        
        # 搜索状态
        self._move_count = 0
        
    # ========================================
    # 主搜索方法
    # ========================================
    
    def run(
        self,
        evaluate_fn: Optional[Callable] = None,
        num_simulations: Optional[int] = None,
        add_noise: bool = False,  # Gumbel 已经提供探索，不需要 Dirichlet
    ) -> Tuple[int, Dict[int, float], float]:
        """执行 Gumbel MCTS 搜索
        
        Args:
            evaluate_fn: 评估函数（同步模式）
            num_simulations: 总模拟次数
            add_noise: 是否添加 Dirichlet 噪声（Gumbel 模式通常不需要）
            
        Returns:
            (action, policy, root_value)
        """
        if num_simulations is None:
            num_simulations = self.config.num_simulations
        
        temperature = self.config.get_temperature(self._move_count)
        
        # Step 1: 扩展根节点，获取先验策略
        if not self.tree.root.is_expanded:
            self._expand_root(evaluate_fn)
        
        root_prior = self._get_root_prior()
        legal_actions = list(self.tree.root.children.keys())
        root_value = self.tree.root.q_value
        
        # Step 2: Gumbel-Top-k 采样选择候选动作
        k = min(self.max_considered_actions, len(legal_actions))
        selected_actions = self._gumbel_top_k(root_prior, legal_actions, k)
        
        logger.debug(f"Gumbel 选择 {k} 个候选动作: {selected_actions}")
        
        # Step 3: Sequential Halving 分配模拟预算并搜索
        q_values, visit_counts = self._sequential_halving_search(
            selected_actions, num_simulations, evaluate_fn
        )
        
        # Step 4: 计算改进策略（Completed Q-values）
        improved_policy = self._compute_improved_policy(
            root_prior, q_values, visit_counts, selected_actions, legal_actions
        )
        
        # Step 5: 选择动作
        action = self._select_action(improved_policy, temperature)
        
        return action, improved_policy, root_value
    
    # ========================================
    # Gumbel-Top-k 采样
    # ========================================
    
    def _gumbel_top_k(
        self,
        prior_probs: Dict[int, float],
        legal_actions: List[int],
        k: int,
    ) -> List[int]:
        """Gumbel-Top-k 采样
        
        使用 Gumbel 噪声选择 top-k 动作:
        g(a) = log(π(a)) + Gumbel(0, scale)
        选择 g(a) 最大的 k 个动作
        
        Args:
            prior_probs: 先验概率 {action: prob}
            legal_actions: 合法动作列表
            k: 选择的动作数
            
        Returns:
            选中的 k 个动作
        """
        scores = {}
        for a in legal_actions:
            log_prob = np.log(prior_probs.get(a, 1e-8) + 1e-8)
            gumbel_noise = np.random.gumbel(0, self.gumbel_scale)
            scores[a] = log_prob + gumbel_noise
        
        sorted_actions = sorted(scores.keys(), key=lambda a: scores[a], reverse=True)
        return sorted_actions[:k]
    
    # ========================================
    # Sequential Halving
    # ========================================
    
    def _sequential_halving_search(
        self,
        candidate_actions: List[int],
        total_simulations: int,
        evaluate_fn: Optional[Callable],
    ) -> Tuple[Dict[int, float], Dict[int, int]]:
        """Sequential Halving 搜索
        
        逐轮减半候选动作，将更多模拟预算分配给好动作。
        
        Args:
            candidate_actions: 候选动作列表
            total_simulations: 总模拟次数
            evaluate_fn: 评估函数
            
        Returns:
            (q_values, visit_counts): 每个动作的 Q 值和访问次数
        """
        remaining_actions = list(candidate_actions)
        q_values = {a: 0.0 for a in candidate_actions}
        visit_counts = {a: 0 for a in candidate_actions}
        
        # 计算每轮的模拟分配
        remaining_budget = total_simulations
        
        for round_idx in range(self.num_halving_rounds):
            if len(remaining_actions) <= 1:
                break
            
            n_actions = len(remaining_actions)
            rounds_left = self.num_halving_rounds - round_idx
            
            # 本轮每个动作的模拟次数
            sims_per_action = max(1, remaining_budget // (n_actions * rounds_left))
            
            # 对每个候选动作执行 MCTS 模拟
            for action in remaining_actions:
                for _ in range(sims_per_action):
                    # 从根节点的对应子节点开始搜索
                    self._simulate_from_action(action, evaluate_fn)
                    visit_counts[action] += 1
                    remaining_budget -= 1
                
                # 更新 Q 值
                child = self.tree.root.children.get(action)
                if child is not None and child.visit_count > 0:
                    q_values[action] = child.q_value
            
            # 减半：保留 Q 值最高的一半
            remaining_actions = sorted(
                remaining_actions,
                key=lambda a: q_values[a],
                reverse=True
            )[:max(1, n_actions // 2)]
            
            logger.debug(f"Round {round_idx + 1}: 保留 {len(remaining_actions)} 个动作")
        
        return q_values, visit_counts
    
    def _simulate_from_action(
        self,
        action: int,
        evaluate_fn: Optional[Callable],
    ) -> None:
        """从指定动作开始执行一次 MCTS 模拟
        
        Args:
            action: 起始动作
            evaluate_fn: 评估函数
        """
        # 获取或创建子节点
        if action not in self.tree.root.children:
            return
        
        child = self.tree.root.children[action]
        
        # 克隆游戏并执行起始动作
        if self.mode == "alphazero":
            game_clone = self.game.clone()
            game_clone.step(action)
            
            # 如果终局，直接回传
            if game_clone.is_terminal():
                value = self._get_terminal_value(game_clone)
                self.tree.backup(child, value)
                return
            
            # 从子节点继续选择
            node = child
            while node.is_expanded and len(node.children) > 0:
                next_action, node = node.select_child(self.config.c_puct)
                game_clone.step(next_action)
                
                if game_clone.is_terminal():
                    value = self._get_terminal_value(game_clone)
                    self.tree.backup(node, value)
                    return
            
            # 扩展叶子节点
            if not node.is_expanded:
                obs = game_clone.get_observation()
                mask = game_clone.get_legal_actions_mask()
                
                if evaluate_fn:
                    policy, value = evaluate_fn(obs, mask)
                elif self.batcher:
                    policy, value = self.batcher.submit(obs, mask)
                else:
                    raise RuntimeError("需要 evaluate_fn 或 batcher")
                
                self.tree.expand(
                    node, game_clone, policy, value,
                    legal_actions=game_clone.legal_actions()
                )
                self.tree.backup(node, value)
        else:
            # MuZero 模式：使用 dynamics network
            # TODO: 实现 MuZero 模式的模拟
            raise NotImplementedError("Gumbel MCTS 的 MuZero 模式尚未实现")
    
    def _get_terminal_value(self, game: Game) -> float:
        """获取终局价值"""
        winner = game.get_winner()
        if winner is None:
            return 0.0
        return 1.0 if winner == game.current_player() else -1.0
    
    # ========================================
    # 策略计算
    # ========================================
    
    def _compute_improved_policy(
        self,
        prior_probs: Dict[int, float],
        q_values: Dict[int, float],
        visit_counts: Dict[int, int],
        selected_actions: List[int],
        legal_actions: List[int],
    ) -> Dict[int, float]:
        """计算改进后的策略（Completed Q-values）
        
        对于搜索过的动作，使用搜索得到的 Q 值。
        对于未搜索的动作，使用先验估计。
        
        Args:
            prior_probs: 先验概率
            q_values: 搜索得到的 Q 值
            visit_counts: 访问次数
            selected_actions: 被选中搜索的动作
            legal_actions: 所有合法动作
            
        Returns:
            改进后的策略分布
        """
        # 计算 Completed Q-values
        completed_q = {}
        
        # 已搜索动作的平均 Q 值
        searched_q = [q_values[a] for a in selected_actions if visit_counts.get(a, 0) > 0]
        mean_q = np.mean(searched_q) if searched_q else 0.0
        
        for a in legal_actions:
            if a in q_values and visit_counts.get(a, 0) > 0:
                completed_q[a] = q_values[a]
            else:
                # 未搜索的动作，使用先验加权估计
                completed_q[a] = mean_q
        
        # 根据访问次数计算策略（类似标准 MCTS）
        total_visits = sum(visit_counts.get(a, 0) for a in selected_actions)
        
        if total_visits > 0:
            # 使用访问次数比例作为策略
            policy = {}
            for a in legal_actions:
                if a in selected_actions:
                    policy[a] = visit_counts.get(a, 0) / total_visits
                else:
                    policy[a] = 0.0
        else:
            # 回退到先验策略
            policy = prior_probs.copy()
        
        # 归一化
        total = sum(policy.values())
        if total > 0:
            policy = {a: p / total for a, p in policy.items()}
        
        return policy
    
    def _select_action(
        self,
        policy: Dict[int, float],
        temperature: float,
    ) -> int:
        """从策略中选择动作
        
        Args:
            policy: 策略分布
            temperature: 温度（0=贪婪，>0=采样）
            
        Returns:
            选择的动作
        """
        actions = list(policy.keys())
        probs = np.array([policy[a] for a in actions])
        
        if temperature == 0:
            return actions[np.argmax(probs)]
        else:
            probs = probs ** (1.0 / temperature)
            probs = probs / (probs.sum() + 1e-8)
            return np.random.choice(actions, p=probs)
    
    # ========================================
    # 辅助方法
    # ========================================
    
    def _expand_root(self, evaluate_fn: Optional[Callable]) -> None:
        """扩展根节点"""
        game_clone = self.game.clone()
        obs = game_clone.get_observation()
        mask = game_clone.get_legal_actions_mask()
        
        if evaluate_fn:
            policy, value = evaluate_fn(obs, mask)
        elif self.batcher:
            policy, value = self.batcher.submit(obs, mask)
        else:
            raise RuntimeError("需要 evaluate_fn 或 batcher")
        
        self.tree.expand(
            self.tree.root,
            game_clone,
            policy,
            value,
            legal_actions=game_clone.legal_actions(),
        )
        self.tree.backup(self.tree.root, value)
    
    def _get_root_prior(self) -> Dict[int, float]:
        """获取根节点的先验策略"""
        prior = {}
        for action, child in self.tree.root.children.items():
            prior[action] = child.prior
        return prior
    
    # ========================================
    # 状态管理
    # ========================================
    
    def advance(self, action: int) -> bool:
        """执行动作并更新树"""
        self._move_count += 1
        return self.tree.advance(action)
    
    def reset(self) -> None:
        """重置搜索状态"""
        self.tree.reset()
        self._move_count = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """获取搜索统计"""
        return {
            "move_count": self._move_count,
            "max_considered_actions": self.max_considered_actions,
            "num_halving_rounds": self.num_halving_rounds,
            "tree": self.tree.get_stats(),
        }


# ============================================================
# 导出
# ============================================================

__all__ = ["GumbelMCTSSearch"]

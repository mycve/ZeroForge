"""
LocalMCTSTree - CPU 本地 MCTS 树

管理 MCTS 搜索过程，支持:
- 节点选择（UCB）
- 节点扩展
- 价值回传
- 子树复用（节点重用）

设计原则:
- 树结构存储在 CPU 内存
- 每个 env 线程维护独立的树
- 叶子节点评估提交给 GPU 批推理
"""

from __future__ import annotations
from typing import Tuple, Optional, List, Dict, Any, TYPE_CHECKING
import numpy as np
import logging

from .node import MCTSNode
from ..config import MCTSConfig

if TYPE_CHECKING:
    from core.game import Game

logger = logging.getLogger(__name__)


class LocalMCTSTree:
    """CPU 本地 MCTS 树
    
    管理单个游戏的 MCTS 搜索树。设计支持:
    - AlphaZero 模式：使用真实环境（game.clone()）
    - MuZero 模式：使用学习的 dynamics
    
    Attributes:
        root: 根节点
        game: 游戏实例（用于 AlphaZero 模式）
        config: MCTS 配置
        mode: 搜索模式 ('alphazero' | 'muzero')
        
    Example:
        >>> game = make_game("chinese_chess")
        >>> tree = LocalMCTSTree(game, config, mode="alphazero")
        >>> 
        >>> # 搜索循环
        >>> for _ in range(num_simulations):
        ...     node, game_clone = tree.select()
        ...     policy, value = network(game_clone.get_observation())
        ...     tree.expand(node, game_clone, policy, value)
        ...     tree.backup(node, value)
        >>> 
        >>> # 选择动作
        >>> action = tree.get_action(temperature=1.0)
        >>> 
        >>> # 执行并复用
        >>> game.step(action)
        >>> tree.advance(action)
    """
    
    def __init__(
        self,
        game: Game,
        config: MCTSConfig,
        mode: str = "alphazero",
    ):
        """初始化 MCTS 树
        
        Args:
            game: 游戏实例
            config: MCTS 配置
            mode: 搜索模式 ('alphazero' | 'muzero')
        """
        self.game = game
        self.config = config
        self.mode = mode
        
        # 创建根节点
        self.root = MCTSNode()
        
        # 统计
        self._total_simulations = 0
        self._reuse_count = 0
    
    # ========================================
    # 核心搜索方法
    # ========================================
    
    def select(self) -> Tuple[MCTSNode, Optional[Game]]:
        """选择叶子节点
        
        从根节点开始，使用 UCB 策略向下选择，直到到达叶子节点。
        
        Returns:
            (leaf_node, game_clone):
                - leaf_node: 选中的叶子节点
                - game_clone: 到达该节点的游戏状态克隆（AlphaZero 模式）
                              或 None（MuZero 模式）
        """
        node = self.root
        game_clone = None
        
        if self.mode == "alphazero":
            # AlphaZero: 克隆游戏状态
            game_clone = self.game.clone()
        
        # 从根节点向下选择
        path_actions = []
        while node.is_expanded and len(node.children) > 0:
            action, node = node.select_child(self.config.c_puct)
            path_actions.append(action)
            
            if self.mode == "alphazero" and game_clone is not None:
                # 在克隆的游戏上执行动作
                game_clone.step(action)
        
        return node, game_clone
    
    def expand(
        self,
        node: MCTSNode,
        game_state: Optional[Game],
        policy: np.ndarray,
        value: float,
        legal_actions: Optional[List[int]] = None,
        hidden_state: Optional[np.ndarray] = None,
    ) -> None:
        """扩展叶子节点
        
        Args:
            node: 要扩展的叶子节点
            game_state: 游戏状态（AlphaZero 模式）
            policy: 策略分布（可以是完整策略向量或只有合法动作的）
            value: 节点价值
            legal_actions: 合法动作列表（如果为 None，从 game_state 获取）
            hidden_state: 隐藏状态（MuZero 模式）
        """
        if node.is_expanded:
            logger.debug("节点已扩展，跳过")
            return
        
        # 获取合法动作
        if legal_actions is None:
            if game_state is not None:
                legal_actions = game_state.legal_actions()
            else:
                raise ValueError("必须提供 legal_actions 或 game_state")
        
        # 如果没有合法动作（终局），不扩展
        if len(legal_actions) == 0:
            return
        
        # 扩展节点
        node.expand(
            legal_actions=legal_actions,
            priors=policy,
            game_state=game_state if self.mode == "alphazero" else None,
            hidden_state=hidden_state,
        )
    
    def backup(self, node: MCTSNode, value: float) -> None:
        """回传价值
        
        Args:
            node: 叶子节点
            value: 叶子节点价值（从叶子节点当前玩家视角）
        """
        node.backup(value)
        self._total_simulations += 1
    
    # ========================================
    # 动作选择
    # ========================================
    
    def get_action(self, temperature: float = 1.0) -> int:
        """根据搜索结果选择动作
        
        Args:
            temperature: 采样温度
            
        Returns:
            选择的动作
        """
        return self.root.select_action(temperature)
    
    def get_policy(self, temperature: float = 1.0) -> Dict[int, float]:
        """获取搜索后的策略分布
        
        Args:
            temperature: 采样温度
            
        Returns:
            策略分布 {action: probability}
        """
        return self.root.get_policy(temperature)
    
    def get_policy_array(self, action_space_size: int, temperature: float = 1.0) -> np.ndarray:
        """获取完整的策略数组
        
        Args:
            action_space_size: 动作空间大小
            temperature: 采样温度
            
        Returns:
            策略数组 [action_space_size]
        """
        policy = np.zeros(action_space_size, dtype=np.float32)
        policy_dict = self.get_policy(temperature)
        for action, prob in policy_dict.items():
            policy[action] = prob
        return policy
    
    def get_root_value(self) -> float:
        """获取根节点价值估计
        
        Returns:
            根节点 Q 值
        """
        return self.root.q_value
    
    # ========================================
    # 树管理
    # ========================================
    
    def advance(self, action: int) -> bool:
        """执行动作并更新树
        
        尝试复用子树。如果子树不存在，重置树。
        
        Args:
            action: 执行的动作
            
        Returns:
            是否成功复用子树
        """
        if self.config.reuse_tree and action in self.root.children:
            # 复用子树
            new_root = self.root.reuse_subtree(action)
            if new_root is not None:
                self.root = new_root
                self._reuse_count += 1
                logger.debug(f"复用子树成功: 动作={action}, 新根访问次数={self.root.visit_count}")
                return True
        
        # 无法复用，重置树
        self.reset()
        return False
    
    def reset(self) -> None:
        """重置树
        
        创建新的根节点，丢弃旧树。
        """
        self.root = MCTSNode()
        logger.debug("MCTS 树已重置")
    
    def add_exploration_noise(self) -> None:
        """向根节点添加探索噪声"""
        self.root.add_exploration_noise(
            dirichlet_alpha=self.config.dirichlet_alpha,
            epsilon=self.config.dirichlet_epsilon,
        )
    
    # ========================================
    # 完整搜索流程（便捷方法）
    # ========================================
    
    def search(
        self,
        evaluate_fn,
        num_simulations: Optional[int] = None,
        add_noise: bool = True,
    ) -> Tuple[int, Dict[int, float], float]:
        """执行完整的 MCTS 搜索
        
        这是一个便捷方法，封装了完整的搜索流程。
        
        Args:
            evaluate_fn: 叶子节点评估函数
                         签名: (observation, legal_mask) -> (policy, value)
            num_simulations: 模拟次数（默认使用配置）
            add_noise: 是否添加探索噪声
            
        Returns:
            (action, policy, value):
                - action: 选择的动作
                - policy: 策略分布
                - value: 根节点价值
        """
        if num_simulations is None:
            num_simulations = self.config.num_simulations
        
        # 确保根节点已扩展
        if not self.root.is_expanded:
            game_clone = self.game.clone()
            obs = game_clone.get_observation()
            mask = game_clone.get_legal_actions_mask()
            policy, value = evaluate_fn(obs, mask)
            
            self.expand(
                self.root,
                game_clone,
                policy,
                value,
                legal_actions=game_clone.legal_actions(),
            )
            self.backup(self.root, value)
            
            if add_noise:
                self.add_exploration_noise()
            
            num_simulations -= 1
        
        # 搜索循环
        for _ in range(num_simulations):
            node, game_clone = self.select()
            
            if game_clone is None:
                # MuZero 模式暂不支持
                raise NotImplementedError("MuZero 模式需要通过 LeafBatcher 使用")
            
            # 检查是否终局
            if game_clone.is_terminal():
                # 终局节点，直接使用游戏结果
                winner = game_clone.get_winner()
                if winner is None:
                    value = 0.0  # 和棋
                else:
                    # 从当前玩家视角
                    current_player = game_clone.current_player()
                    value = 1.0 if winner == current_player else -1.0
                
                self.backup(node, value)
                continue
            
            # 评估叶子节点
            obs = game_clone.get_observation()
            mask = game_clone.get_legal_actions_mask()
            policy, value = evaluate_fn(obs, mask)
            
            # 扩展和回传
            self.expand(
                node,
                game_clone,
                policy,
                value,
                legal_actions=game_clone.legal_actions(),
            )
            self.backup(node, value)
        
        # 获取结果
        temperature = self.config.get_temperature(0)  # TODO: 传入 move_count
        action = self.get_action(temperature)
        policy = self.get_policy(temperature)
        value = self.get_root_value()
        
        return action, policy, value
    
    # ========================================
    # 统计和调试
    # ========================================
    
    def get_stats(self) -> Dict[str, Any]:
        """获取树统计信息
        
        Returns:
            统计信息字典
        """
        return {
            "total_simulations": self._total_simulations,
            "reuse_count": self._reuse_count,
            "root_visits": self.root.visit_count,
            "root_value": self.root.q_value,
            "root_children": len(self.root.children),
            "mode": self.mode,
        }
    
    def print_tree(self, max_depth: int = 3) -> None:
        """打印树结构"""
        print(f"=== MCTS Tree (mode={self.mode}) ===")
        self.root.print_tree(max_depth)
    
    def __repr__(self) -> str:
        return (
            f"LocalMCTSTree(mode={self.mode}, "
            f"root_visits={self.root.visit_count}, "
            f"simulations={self._total_simulations})"
        )

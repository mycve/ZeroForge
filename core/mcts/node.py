"""
MCTSNode - MCTS 树节点（优化版）

CPU 端的 MCTS 节点数据结构，支持:
- 访问计数和价值累积
- 树结构（父节点、子节点）
- 子树复用（节点重用）
- AlphaZero 模式（存储 game clone）
- MuZero 模式（存储 hidden state）

============================================================
性能优化
============================================================

1. __slots__ 优化
   - 减少内存占用（无 __dict__）
   - 加速属性访问
   
2. expand() 使用 __new__ 跳过 __init__
   - 批量创建子节点时避免 __init__ 参数解析开销
   - 直接设置属性，减少 ~40% 节点创建时间

3. select_child() 纯 Python 实现
   - 避免 numpy.argmax 的调度开销（~0.88s）
   - 内联 q_value 计算，减少属性访问
   - 对于 MCTS 典型的子节点数量（<100），纯 Python 比 numpy 快

============================================================
API 说明
============================================================

核心方法:
- expand(legal_actions, priors, game_state, hidden_state)
  扩展节点，创建所有合法动作的子节点
  
- select_child(c_puct) -> (action, child)
  使用 UCB 公式选择最佳子节点
  
- backup(value)
  从当前节点向上回传价值（零和博弈自动取反）

- reuse_subtree(action) -> new_root
  子树复用，将子节点提升为新根节点

属性:
- q_value: 平均 Q 值 (value_sum / visit_count)
- is_expanded: 是否已扩展
- is_leaf: 是否为叶子节点
- depth: 节点深度

============================================================
使用示例
============================================================

>>> root = MCTSNode()
>>> root.expand(legal_actions=[0, 1, 2], priors=np.array([0.5, 0.3, 0.2]))
>>> action, child = root.select_child(c_puct=1.5)
>>> child.backup(value=0.8)
>>> root.print_tree(max_depth=2)
"""

from __future__ import annotations
from typing import Dict, Optional, Any, TYPE_CHECKING, List, Tuple
import numpy as np
import math
import logging

if TYPE_CHECKING:
    from core.game import Game

logger = logging.getLogger(__name__)


class MCTSNode:
    """MCTS 树节点（优化版，使用 __slots__）
    
    存储节点统计信息和树结构。设计支持 AlphaZero 和 MuZero 两种模式。
    
    Attributes:
        visit_count: 访问次数
        value_sum: 累计价值（从当前玩家视角）
        prior: 先验概率 P(s, a)
        parent: 父节点引用
        children: 子节点字典 {action: node}
        action: 到达此节点的动作（根节点为 -1）
        
        # AlphaZero 模式
        game_state: 游戏状态克隆（用于真实环境搜索）
        
        # MuZero 模式
        hidden_state: 隐藏状态张量（用于 dynamics 网络）
        reward: 到达此节点的即时奖励
        
    Example:
        >>> root = MCTSNode()
        >>> root.expand(legal_actions=[0, 1, 2], priors=[0.5, 0.3, 0.2])
        >>> child = root.children[0]
        >>> child.backup(value=0.8)
    """
    
    # 【优化】使用 __slots__ 减少内存开销和加速属性访问
    __slots__ = (
        'visit_count', 'value_sum', 'prior', 'parent', 'children', 
        'action', 'game_state', 'hidden_state', 'reward', '_expanded',
    )
    
    def __init__(
        self,
        visit_count: int = 0,
        value_sum: float = 0.0,
        prior: float = 0.0,
        parent: Optional[MCTSNode] = None,
        action: int = -1,
        game_state: Optional[Any] = None,
        hidden_state: Optional[np.ndarray] = None,
        reward: float = 0.0,
    ):
        # 保持完整签名以兼容旧代码，但优化了 expand 中的节点创建
        self.visit_count = visit_count
        self.value_sum = value_sum
        self.prior = prior
        self.parent = parent
        self.children: Dict[int, MCTSNode] = {}
        self.action = action
        self.game_state = game_state
        self.hidden_state = hidden_state
        self.reward = reward
        self._expanded = False
    
    # ========================================
    # 属性
    # ========================================
    
    @property
    def q_value(self) -> float:
        """平均 Q 值
        
        Returns:
            平均价值，未访问返回 0
        """
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count
    
    @property
    def is_expanded(self) -> bool:
        """是否已扩展"""
        return self._expanded
    
    @property
    def is_root(self) -> bool:
        """是否为根节点"""
        return self.parent is None
    
    @property
    def is_leaf(self) -> bool:
        """是否为叶子节点（未扩展或无子节点）"""
        return not self._expanded or len(self.children) == 0
    
    @property
    def depth(self) -> int:
        """节点深度（根节点为 0）"""
        d = 0
        node = self
        while node.parent is not None:
            d += 1
            node = node.parent
        return d
    
    # ========================================
    # 核心方法
    # ========================================
    
    def expand(
        self,
        legal_actions: List[int],
        priors: np.ndarray,
        game_state: Optional[Any] = None,
        hidden_state: Optional[np.ndarray] = None,
    ) -> None:
        """扩展节点
        
        创建所有合法动作的子节点。
        
        Args:
            legal_actions: 合法动作列表
            priors: 先验概率数组（与 legal_actions 对应或完整策略）
            game_state: 游戏状态（AlphaZero 模式）
            hidden_state: 隐藏状态（MuZero 模式）
        """
        if self._expanded:
            return  # 【优化】移除 logger 调用，减少开销
        
        # 存储状态
        self.game_state = game_state
        self.hidden_state = hidden_state
        
        # 【优化】预先判断 priors 类型，避免循环内判断
        children = self.children
        priors_len = len(priors)
        legal_len = len(legal_actions)
        use_index = (priors_len == legal_len)
        
        # 【优化】批量创建子节点
        if use_index:
            for i, action in enumerate(legal_actions):
                child = MCTSNode.__new__(MCTSNode)
                child.visit_count = 0
                child.value_sum = 0.0
                child.prior = float(priors[i])
                child.parent = self
                child.children = {}
                child.action = action
                child.game_state = None
                child.hidden_state = None
                child.reward = 0.0
                child._expanded = False
                children[action] = child
        else:
            for action in legal_actions:
                child = MCTSNode.__new__(MCTSNode)
                child.visit_count = 0
                child.value_sum = 0.0
                child.prior = float(priors[action])
                child.parent = self
                child.children = {}
                child.action = action
                child.game_state = None
                child.hidden_state = None
                child.reward = 0.0
                child._expanded = False
                children[action] = child
        
        self._expanded = True
    
    def select_child(self, c_puct: float = 1.5) -> Tuple[int, MCTSNode]:
        """使用 UCB 选择最佳子节点（优化版：纯 Python 循环，避免 numpy 调度开销）
        
        UCB = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
        
        Args:
            c_puct: 探索常数
            
        Returns:
            (action, child_node)
            
        Raises:
            ValueError: 如果节点未扩展或无子节点
        """
        if not self._expanded or len(self.children) == 0:
            raise ValueError("无法从未扩展或无子节点的节点选择")
        
        # 【优化】纯 Python 循环比 numpy 快（避免 argmax 调度开销 0.88s）
        sqrt_parent_visits = math.sqrt(self.visit_count + 1)
        best_action = -1
        best_score = -1e9  # 使用固定值比 float('-inf') 快
        best_child = None
        
        for action, child in self.children.items():
            # 内联 q_value 计算，避免函数调用
            if child.visit_count == 0:
                q_value = 0.0
            else:
                q_value = -child.value_sum / child.visit_count
            
            prior_score = c_puct * child.prior * sqrt_parent_visits / (1 + child.visit_count)
            ucb_score = q_value + prior_score
            
            if ucb_score > best_score:
                best_score = ucb_score
                best_action = action
                best_child = child
        
        return best_action, best_child
    
    def backup(self, value: float) -> None:
        """回传价值
        
        从当前节点向上更新所有祖先节点的统计信息。
        价值在每层取反（零和博弈）。
        
        Args:
            value: 叶子节点价值（从当前玩家视角）
        """
        node = self
        current_value = value
        
        while node is not None:
            node.visit_count += 1
            node.value_sum += current_value
            current_value = -current_value  # 零和博弈，价值取反
            node = node.parent
    
    def add_exploration_noise(
        self,
        dirichlet_alpha: float = 0.3,
        epsilon: float = 0.25,
    ) -> None:
        """向根节点添加探索噪声
        
        只在根节点调用，用于增加探索。
        
        Args:
            dirichlet_alpha: Dirichlet 分布 alpha 参数
            epsilon: 噪声混合比例
        """
        if not self._expanded:
            return
        
        actions = list(self.children.keys())
        noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
        
        for i, action in enumerate(actions):
            child = self.children[action]
            child.prior = (1 - epsilon) * child.prior + epsilon * noise[i]
    
    # ========================================
    # 子树复用
    # ========================================
    
    def reuse_subtree(self, action: int) -> Optional[MCTSNode]:
        """复用子树
        
        执行动作后，将对应子节点提升为新根节点。
        其他子树将被垃圾回收。
        
        Args:
            action: 执行的动作
            
        Returns:
            新的根节点，如果动作不存在返回 None
        """
        if action not in self.children:
            logger.debug(f"动作 {action} 不在子节点中，无法复用子树")
            return None
        
        new_root = self.children[action]
        new_root.parent = None  # 断开父引用，让旧节点被 GC
        
        # 清空其他子节点引用（加速 GC）
        self.children.clear()
        
        logger.debug(f"复用子树: 动作={action}, 新根访问次数={new_root.visit_count}")
        return new_root
    
    def detach(self) -> None:
        """从父节点分离
        
        用于复用时断开与旧树的连接。
        """
        if self.parent is not None:
            # 从父节点的 children 中移除自己
            if self.action in self.parent.children:
                del self.parent.children[self.action]
            self.parent = None
    
    # ========================================
    # 统计方法
    # ========================================
    
    def get_visit_distribution(self) -> Dict[int, int]:
        """获取子节点访问次数分布
        
        Returns:
            {action: visit_count} 字典
        """
        return {action: child.visit_count for action, child in self.children.items()}
    
    def get_policy(self, temperature: float = 1.0) -> Dict[int, float]:
        """根据访问次数计算策略分布
        
        Args:
            temperature: 采样温度，0 表示贪婪，1 表示按比例
            
        Returns:
            {action: probability} 字典
        """
        if len(self.children) == 0:
            return {}
        
        visits = np.array([child.visit_count for child in self.children.values()])
        actions = list(self.children.keys())
        
        if temperature == 0:
            # 贪婪选择
            probs = np.zeros_like(visits, dtype=np.float32)
            probs[np.argmax(visits)] = 1.0
        else:
            # 温度采样
            visits_temp = visits ** (1.0 / temperature)
            probs = visits_temp / visits_temp.sum()
        
        return {action: float(prob) for action, prob in zip(actions, probs)}
    
    def select_action(self, temperature: float = 1.0) -> int:
        """根据访问次数选择动作
        
        Args:
            temperature: 采样温度
            
        Returns:
            选择的动作
        """
        policy = self.get_policy(temperature)
        actions = list(policy.keys())
        probs = list(policy.values())
        return np.random.choice(actions, p=probs)
    
    def get_best_action(self) -> int:
        """获取最佳动作（访问次数最多）
        
        Returns:
            最佳动作
        """
        if len(self.children) == 0:
            raise ValueError("无子节点，无法获取最佳动作")
        return max(self.children.keys(), key=lambda a: self.children[a].visit_count)
    
    # ========================================
    # 调试方法
    # ========================================
    
    def get_stats(self) -> Dict[str, Any]:
        """获取节点统计信息
        
        Returns:
            统计信息字典
        """
        return {
            "visit_count": self.visit_count,
            "value_sum": self.value_sum,
            "q_value": self.q_value,
            "prior": self.prior,
            "action": self.action,
            "num_children": len(self.children),
            "is_expanded": self._expanded,
            "depth": self.depth,
            "has_game_state": self.game_state is not None,
            "has_hidden_state": self.hidden_state is not None,
        }
    
    def print_tree(self, max_depth: int = 3, indent: int = 0) -> None:
        """打印树结构（用于调试）
        
        Args:
            max_depth: 最大打印深度
            indent: 当前缩进级别
        """
        prefix = "  " * indent
        info = f"N={self.visit_count}, Q={self.q_value:.3f}, P={self.prior:.3f}"
        if self.action >= 0:
            print(f"{prefix}[{self.action}] {info}")
        else:
            print(f"{prefix}[ROOT] {info}")
        
        if indent < max_depth:
            # 按访问次数排序打印
            sorted_children = sorted(
                self.children.items(),
                key=lambda x: x[1].visit_count,
                reverse=True
            )
            for action, child in sorted_children[:5]:  # 只打印前 5 个
                child.print_tree(max_depth, indent + 1)
    
    def __repr__(self) -> str:
        return (
            f"MCTSNode(action={self.action}, N={self.visit_count}, "
            f"Q={self.q_value:.3f}, P={self.prior:.3f}, "
            f"children={len(self.children)})"
        )

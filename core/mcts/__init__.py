"""
MCTS - Monte Carlo Tree Search 模块

提供 CPU 本地 MCTS 树实现，支持:
- 节点复用（子树保留）
- GPU 批量叶子推理
- nogil 多线程并行
- Gumbel MCTS（根节点 Gumbel-Top-k + Sequential Halving）

核心组件:
- MCTSNode: 树节点数据结构
- LocalMCTSTree: 本地 MCTS 树
- LeafBatcher: GPU 批量推理收集器
- MCTSSearch: 标准 MCTS 搜索控制器
- GumbelMCTSSearch: Gumbel MCTS 搜索（论文实现）

使用示例:
    >>> from core.mcts import MCTSSearch, GumbelMCTSSearch
    >>> 
    >>> # 标准 MCTS
    >>> search = MCTSSearch(game, config)
    >>> action, policy, value = search.run(evaluate_fn)
    >>> 
    >>> # Gumbel MCTS（适用于 Gymnasium 等环境）
    >>> search = GumbelMCTSSearch(game, config, max_considered_actions=16)
    >>> action, policy, value = search.run(evaluate_fn)
"""

from .node import MCTSNode
from .tree import LocalMCTSTree
from .batcher import LeafBatcher, LeafRequest
from .search import MCTSSearch
from .gumbel import GumbelMCTSSearch
from ..config import MCTSConfig

__all__ = [
    "MCTSNode",
    "LocalMCTSTree",
    "LeafBatcher",
    "LeafRequest",
    "MCTSSearch",
    "GumbelMCTSSearch",  # Gumbel MCTS
    "MCTSConfig",
]

"""
MCTS - Monte Carlo Tree Search 模块

提供 CPU 本地 MCTS 树实现，支持:
- 节点复用（子树保留）
- GPU 批量叶子推理
- nogil 多线程并行

核心组件:
- MCTSNode: 树节点数据结构
- LocalMCTSTree: 本地 MCTS 树
- LeafBatcher: GPU 批量推理收集器
- MCTSSearch: 搜索控制器

使用示例:
    >>> from core.mcts import LocalMCTSTree, LeafBatcher
    >>> tree = LocalMCTSTree(game, config)
    >>> batcher = LeafBatcher(network, config)
    >>> 
    >>> # 搜索
    >>> for _ in range(num_simulations):
    ...     node, game_clone = tree.select()
    ...     policy, value = batcher.submit(game_clone.get_observation())
    ...     tree.expand(node, policy, value)
    ...     tree.backup(node, value)
"""

from .node import MCTSNode
from .tree import LocalMCTSTree
from .batcher import LeafBatcher, LeafRequest
from .search import MCTSSearch
from ..config import MCTSConfig

__all__ = [
    "MCTSNode",
    "LocalMCTSTree",
    "LeafBatcher",
    "LeafRequest",
    "MCTSSearch",
    "MCTSConfig",
]

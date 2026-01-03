"""
MCTS - Monte Carlo Tree Search 模块

提供 CPU 本地 MCTS 树实现，支持:
- 节点复用（子树保留）
- GPU 批量叶子推理
- nogil 多线程并行（Python 3.13+ free-threaded）
- Gumbel MCTS（根节点 Gumbel-Top-k + Sequential Halving）

============================================================
核心组件
============================================================

MCTSNode        - 树节点（__slots__ 优化，低内存开销）
LocalMCTSTree   - 本地 MCTS 树（单游戏搜索管理）
LeafBatcher     - GPU 批量推理（Queue 实现，有锁）
SlotBatcher     - GPU 批量推理（预分配槽位，无锁，推荐）
MCTSSearch      - 标准 MCTS 搜索控制器
GumbelMCTSSearch- Gumbel MCTS 搜索（论文实现）

============================================================
Batcher 选择指南
============================================================

| 场景                  | 推荐 Batcher    |
|-----------------------|-----------------|
| 固定并发数（训练）    | SlotBatcher     |
| 动态线程数            | LeafBatcher     |
| Python 3.13+ no-GIL   | SlotBatcher     |
| 传统 GIL Python       | 都可以          |

============================================================
使用示例
============================================================

1. 标准 MCTS 搜索:
    >>> from core.mcts import MCTSSearch
    >>> search = MCTSSearch(game, config)
    >>> action, policy, value = search.run(evaluate_fn)

2. Gumbel MCTS 搜索:
    >>> from core.mcts import GumbelMCTSSearch
    >>> search = GumbelMCTSSearch(game, config, max_considered_actions=16)
    >>> action, policy, value = search.run(evaluate_fn)

3. SlotBatcher（无锁高性能）:
    >>> from core.mcts import SlotBatcher
    >>> batcher = SlotBatcher(network, config, num_slots=64)
    >>> batcher.start()
    >>> # 每个线程使用固定槽位
    >>> policy, value = batcher.submit(slot_id, obs, mask)

============================================================
性能优化总结
============================================================

- MCTSNode: __slots__ + __new__ 创建，减少 40% 节点创建时间
- select_child: 纯 Python 循环，避免 numpy 调度开销
- SlotBatcher: 无锁预分配，消除 Queue 锁竞争
- LeafRequestPool: 线程本地对象池，避免 Event 创建开销
"""

from .node import MCTSNode
from .tree import LocalMCTSTree
from .batcher import LeafBatcher, LeafRequest, SlotBatcher
from .search import MCTSSearch
from .gumbel import GumbelMCTSSearch
from ..config import MCTSConfig

__all__ = [
    "MCTSNode",
    "LocalMCTSTree",
    "LeafBatcher",
    "LeafRequest",
    "SlotBatcher",  # 无锁预分配槽位 Batcher
    "MCTSSearch",
    "GumbelMCTSSearch",  # Gumbel MCTS
    "MCTSConfig",
]

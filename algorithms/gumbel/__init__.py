"""
Gumbel 算法模块

实现 Gumbel MuZero 和 Gumbel AlphaZero 算法。

核心特点:
1. 根节点使用 Gumbel-Top-k 采样选择候选动作
2. 使用 Sequential Halving 分配模拟预算
3. 使用 Completed Q-values 进行策略改进

参考论文:
- Policy improvement by planning with Gumbel (ICLR 2022)
- https://arxiv.org/abs/2104.06303

搜索实现（自动选择）:
- GumbelMCTSSearch: 官方实现，Gumbel + MCTS 树搜索（需要 clone）
- GumbelSearch: 简化版，仅用网络评估（用于 Gymnasium 等不支持 clone 的环境）

算法会自动检测游戏是否支持 clone()，选择合适的搜索实现。
"""

from .algorithm import GumbelMuZeroAlgorithm, GumbelAlphaZeroAlgorithm
from .search import GumbelSearch, GumbelConfig

__all__ = [
    "GumbelMuZeroAlgorithm",
    "GumbelAlphaZeroAlgorithm",
    "GumbelSearch",
    "GumbelConfig",
]

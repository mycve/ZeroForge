"""
Gumbel 算法模块

实现 Gumbel MuZero 和 Gumbel AlphaZero 算法。

核心特点:
1. 使用 Gumbel-Top-k 采样替代 UCB 选择
2. 使用 Sequential Halving 进行高效搜索
3. 不需要环境克隆（适用于 Gymnasium 等环境）
4. 使用 Completed Q-values 进行策略改进

参考论文:
- Policy improvement by planning with Gumbel (ICLR 2022)
- https://arxiv.org/abs/2104.06303
"""

from .algorithm import GumbelMuZeroAlgorithm, GumbelAlphaZeroAlgorithm
from .search import GumbelSearch

__all__ = [
    "GumbelMuZeroAlgorithm",
    "GumbelAlphaZeroAlgorithm",
    "GumbelSearch",
]

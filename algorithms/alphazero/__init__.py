"""
AlphaZero 算法实现

AlphaZero 特点:
- 使用真实环境进行搜索（无需学习环境模型）
- 只有 representation + prediction 网络
- MCTS 直接使用游戏规则展开节点

组件:
- AlphaZeroAlgorithm: 算法主类（网络创建 + 损失计算）
- AlphaZeroNetwork: 神经网络（无 dynamics）

搜索和自玩使用 core 模块:
- core.mcts.MCTSSearch: MCTS 搜索
- core.env.ThreadedSelfPlay: 多线程自玩
"""

from .algorithm import AlphaZeroAlgorithm, AlphaZeroConfig
from .network import AlphaZeroNetwork, SimpleAlphaZeroNetwork

__all__ = [
    "AlphaZeroAlgorithm",
    "AlphaZeroConfig",
    "AlphaZeroNetwork",
    "SimpleAlphaZeroNetwork",
]


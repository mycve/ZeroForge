"""
Algorithms - 算法实现模块

提供 MuZero 和 AlphaZero 系列算法实现。

算法列表:
- MuZero: 学习环境模型，需要 dynamics network
- AlphaZero: 使用真实环境，无需 dynamics network

使用方法:
    >>> from algorithms import make_algorithm
    >>> algo = make_algorithm("muzero", config)
    >>> network = algo.create_network(game)
"""

from typing import Dict, Type, Any, Optional
import logging

from core.algorithm import Algorithm, AlgorithmConfig

logger = logging.getLogger(__name__)


# ============================================================
# 算法注册表
# ============================================================

ALGORITHM_REGISTRY: Dict[str, Type[Algorithm]] = {}


def register_algorithm(name: str):
    """算法注册装饰器"""
    def decorator(cls: Type[Algorithm]) -> Type[Algorithm]:
        ALGORITHM_REGISTRY[name] = cls
        logger.debug(f"注册算法: {name} -> {cls.__name__}")
        return cls
    return decorator


def make_algorithm(name: str, config: Optional[AlgorithmConfig] = None, **kwargs) -> Algorithm:
    """创建算法实例
    
    Args:
        name: 算法名称 (alphazero, muzero)
        config: 配置对象（如果为 None，使用算法默认配置）
        **kwargs: 传递给配置的额外参数
    """
    if name not in ALGORITHM_REGISTRY:
        available = ", ".join(ALGORITHM_REGISTRY.keys())
        raise ValueError(f"未知算法: '{name}'。可用算法: {available or '(无)'}")
    
    algo_cls = ALGORITHM_REGISTRY[name]
    
    if config is None:
        # 使用算法自己的默认配置（不是基类 AlgorithmConfig）
        # 每个算法类在 __init__ 中会创建自己的默认配置
        return algo_cls(None)
    
    return algo_cls(config)


def list_algorithms() -> list:
    """列出所有已注册的算法"""
    return list(ALGORITHM_REGISTRY.keys())


# ============================================================
# 导入算法实现（触发注册）
# ============================================================

from .muzero import MuZeroAlgorithm
from .alphazero import AlphaZeroAlgorithm

__all__ = [
    "register_algorithm",
    "make_algorithm",
    "list_algorithms",
    "ALGORITHM_REGISTRY",
    "MuZeroAlgorithm",
    "AlphaZeroAlgorithm",
]


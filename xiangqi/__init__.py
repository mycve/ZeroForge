"""
中国象棋 JAX 环境模块
纯 JAX 实现，支持向量化和 JIT 编译
"""

from xiangqi.env import XiangqiEnv, XiangqiState
from xiangqi.actions import (
    ACTION_SPACE_SIZE,
    action_to_move,
    move_to_action,
    get_action_mask,
)
from xiangqi.rules import (
    is_valid_move,
    is_in_check,
    get_legal_moves,
    is_game_over,
)
from xiangqi.mirror import mirror_board, mirror_action, mirror_policy

__all__ = [
    "XiangqiEnv",
    "XiangqiState",
    "ACTION_SPACE_SIZE",
    "action_to_move",
    "move_to_action",
    "get_action_mask",
    "is_valid_move",
    "is_in_check",
    "get_legal_moves",
    "is_game_over",
    "mirror_board",
    "mirror_action",
    "mirror_policy",
]

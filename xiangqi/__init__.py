"""
中国象棋 JAX 环境模块
纯 JAX 实现，支持向量化和 JIT 编译

违规规则（符合正式比赛规则）：
- 长将：1子6回合、2子12回合、3子+18回合
- 长捉：6回合
- 将捉交替：1子12回合、多子18回合
- 重复局面5次判和（非将非捉）
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
from xiangqi.violation_rules import (
    is_chase_move,
    count_checking_pieces,
    check_violation,
    is_piece_pinned,
    has_real_protector,
)

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
    # 违规规则
    "is_chase_move",
    "count_checking_pieces",
    "check_violation",
    "is_piece_pinned",
    "has_real_protector",
]

"""
中国象棋违规规则检测模块 (纯 JAX 实现)

实现棋规中的"提示强制变招"规则：
- 长将检测（按子数：1子6回合、2子12回合、3子+18回合）
- 长捉检测（6回合限制）
- 将捉交替检测（1子12回合、多子18回合）
- 捉子判定（有根/无根、真根/假根、牵制）

基本术语：
- 将军：走子直接攻击对方"将"或"帅"
- 捉：走子攻击对方除将帅以外的任何无根子，并企图于下一步吃去
- 有根：被捉子如有另子保护，可反吃
- 真根：保护子在被吃后可以反吃
- 假根：保护子因受牵制不能反吃
- 牵制：一旦离位会被送将
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
from functools import partial
from xiangqi.actions import (
    BOARD_HEIGHT, BOARD_WIDTH, NUM_SQUARES, ACTION_SPACE_SIZE,
    EMPTY, R_KING, R_ADVISOR, R_BISHOP, R_KNIGHT, R_ROOK, R_CANNON, R_PAWN,
    B_KING, B_ADVISOR, B_BISHOP, B_KNIGHT, B_ROOK, B_CANNON, B_PAWN,
    action_to_move, move_to_action,
)
from xiangqi.rules import (
    find_king, is_in_check, is_in_check_at, apply_move, get_piece_type,
    is_own_piece, is_enemy_piece, has_crossed_river,
    _is_valid_rook_move, _is_valid_cannon_move, _is_valid_knight_move,
    _is_valid_pawn_move, _is_valid_king_move,
)

# ============================================================================
# 常量定义
# ============================================================================

# 长将回合限制（按参与将军的子数）
PERPETUAL_CHECK_LIMIT_1 = 6   # 1子长将6回合
PERPETUAL_CHECK_LIMIT_2 = 12  # 2子长将12回合
PERPETUAL_CHECK_LIMIT_3 = 18  # 3子+长将18回合

# 长捉回合限制
PERPETUAL_CHASE_LIMIT = 6     # 长捉6回合

# 将捉交替回合限制
CHECK_CHASE_ALT_LIMIT_1 = 12  # 1子将捉交替12回合
CHECK_CHASE_ALT_LIMIT_N = 18  # 多子将捉交替18回合

# 无吃子判和中将军的最大累计回合
MAX_CHECK_IN_NO_CAPTURE = 20  # 将军最多累计20回合（40步）

# 违规历史记录大小
VIOLATION_HISTORY_SIZE = 64


# ============================================================================
# 牵制检测
# ============================================================================

@jax.jit
def is_piece_pinned(board: jnp.ndarray, piece_row: jnp.ndarray, piece_col: jnp.ndarray, 
                    player: jnp.ndarray) -> jnp.ndarray:
    """
    检测指定棋子是否被牵制（离开后会导致己方被将军）
    
    Args:
        board: 棋盘状态
        piece_row, piece_col: 棋子位置
        player: 棋子所属方 (0=红, 1=黑)
        
    Returns:
        是否被牵制
    """
    # 找到己方将/帅位置
    king_row, king_col = find_king(board, player)
    
    # 模拟移除该棋子
    piece = board[piece_row, piece_col]
    temp_board = board.at[piece_row, piece_col].set(EMPTY)
    
    # 检查移除后是否被将军
    would_be_check = is_in_check_at(temp_board, player, king_row, king_col)
    
    # 只有己方棋子才可能被牵制
    is_own = is_own_piece(piece, player)
    
    return is_own & would_be_check


# ============================================================================
# 捉子检测核心
# ============================================================================

@jax.jit
def can_piece_attack(board: jnp.ndarray, attacker_row: jnp.ndarray, attacker_col: jnp.ndarray,
                     target_row: jnp.ndarray, target_col: jnp.ndarray, player: jnp.ndarray) -> jnp.ndarray:
    """
    检测指定棋子是否能攻击目标位置（仅检查走法，不检查是否送将）
    
    Args:
        board: 棋盘状态
        attacker_row, attacker_col: 攻击者位置
        target_row, target_col: 目标位置
        player: 攻击者所属方
        
    Returns:
        是否能攻击
    """
    piece = board[attacker_row, attacker_col]
    piece_type = get_piece_type(piece)
    
    # 车的攻击
    rook_attack = (piece_type == 5) & _is_valid_rook_move(
        board, attacker_row, attacker_col, target_row, target_col, player)
    
    # 炮的攻击（需要有炮架）
    cannon_attack = (piece_type == 6) & _is_valid_cannon_move(
        board, attacker_row, attacker_col, target_row, target_col, player)
    
    # 马的攻击
    knight_attack = (piece_type == 4) & _is_valid_knight_move(
        board, attacker_row, attacker_col, target_row, target_col, player)
    
    # 兵/卒的攻击
    pawn_attack = (piece_type == 7) & _is_valid_pawn_move(
        board, attacker_row, attacker_col, target_row, target_col, player)
    
    # 将/帅的攻击（在九宫内）
    king_attack = (piece_type == 1) & _is_valid_king_move(
        board, attacker_row, attacker_col, target_row, target_col, player)
    
    return rook_attack | cannon_attack | knight_attack | pawn_attack | king_attack


@jax.jit
def find_protectors(board: jnp.ndarray, target_row: jnp.ndarray, target_col: jnp.ndarray,
                    defender_player: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    找出所有能保护目标位置的己方棋子
    
    Args:
        board: 棋盘状态
        target_row, target_col: 被保护的位置
        defender_player: 防守方
        
    Returns:
        (protector_mask, protector_count) - 保护者掩码和数量
    """
    # 遍历所有格子，找能吃回目标位置的己方棋子
    def check_protector(sq):
        row = sq // BOARD_WIDTH
        col = sq % BOARD_WIDTH
        piece = board[row, col]
        
        # 必须是己方棋子且不是目标位置本身
        is_own = is_own_piece(piece, defender_player)
        is_not_target = (row != target_row) | (col != target_col)
        
        # 检查能否攻击到目标位置
        can_attack = can_piece_attack(board, row, col, target_row, target_col, defender_player)
        
        return is_own & is_not_target & can_attack
    
    squares = jnp.arange(NUM_SQUARES)
    protector_mask = jax.vmap(check_protector)(squares)
    protector_count = jnp.sum(protector_mask)
    
    return protector_mask, protector_count


@jax.jit
def is_protector_real(board: jnp.ndarray, protector_row: jnp.ndarray, protector_col: jnp.ndarray,
                      target_row: jnp.ndarray, target_col: jnp.ndarray,
                      attacker_row: jnp.ndarray, attacker_col: jnp.ndarray,
                      defender_player: jnp.ndarray) -> jnp.ndarray:
    """
    判断保护者是真根还是假根
    
    真根：保护子在被吃子被吃后能反吃攻击者
    假根：保护子因受牵制不能反吃
    
    Args:
        board: 棋盘状态
        protector_row, protector_col: 保护者位置
        target_row, target_col: 被保护棋子位置
        attacker_row, attacker_col: 攻击者位置
        defender_player: 防守方
        
    Returns:
        是否为真根
    """
    # 检查保护者是否被牵制
    is_pinned = is_piece_pinned(board, protector_row, protector_col, defender_player)
    
    # 模拟攻击者吃掉目标棋子后的局面
    attacker_piece = board[attacker_row, attacker_col]
    new_board = board.at[target_row, target_col].set(attacker_piece)
    new_board = new_board.at[attacker_row, attacker_col].set(EMPTY)
    
    # 检查保护者能否在新局面中吃回攻击者（现在在目标位置）
    can_recapture = can_piece_attack(new_board, protector_row, protector_col, 
                                      target_row, target_col, defender_player)
    
    # 检查吃回后是否会送将
    protector_piece = board[protector_row, protector_col]
    after_recapture = new_board.at[target_row, target_col].set(protector_piece)
    after_recapture = after_recapture.at[protector_row, protector_col].set(EMPTY)
    
    king_row, king_col = find_king(after_recapture, defender_player)
    would_be_check = is_in_check_at(after_recapture, defender_player, king_row, king_col)
    
    # 真根条件：不被牵制 且 能吃回 且 吃回后不送将
    return ~is_pinned & can_recapture & ~would_be_check


@jax.jit
def has_real_protector(board: jnp.ndarray, target_row: jnp.ndarray, target_col: jnp.ndarray,
                       attacker_row: jnp.ndarray, attacker_col: jnp.ndarray,
                       defender_player: jnp.ndarray) -> jnp.ndarray:
    """
    检查被捉子是否有真根保护
    
    Args:
        board: 棋盘状态
        target_row, target_col: 被捉子位置
        attacker_row, attacker_col: 攻击者位置（捉的一方）
        defender_player: 被捉子所属方
        
    Returns:
        是否有真根保护
    """
    protector_mask, protector_count = find_protectors(board, target_row, target_col, defender_player)
    
    # 如果没有保护者，直接返回False
    no_protector = protector_count == 0
    
    # 检查每个保护者是否为真根
    def check_real(sq):
        is_protector = protector_mask[sq]
        row = sq // BOARD_WIDTH
        col = sq % BOARD_WIDTH
        is_real = is_protector_real(board, row, col, target_row, target_col,
                                     attacker_row, attacker_col, defender_player)
        return is_protector & is_real
    
    squares = jnp.arange(NUM_SQUARES)
    real_protector_mask = jax.vmap(check_real)(squares)
    has_real = jnp.any(real_protector_mask)
    
    return jnp.where(no_protector, jnp.bool_(False), has_real)


# ============================================================================
# 捉子判定（完整规则）
# ============================================================================

@jax.jit
def is_chase_move(board_before: jnp.ndarray, board_after: jnp.ndarray,
                  from_sq: jnp.ndarray, to_sq: jnp.ndarray,
                  player: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    判断一步棋是否构成"捉"
    
    捉的定义：走子攻击对方除将帅以外的任何无根子，并企图于下一步吃去
    
    不算捉的情况：
    1. 分捉多子不算捉
    2. 捉未过河的卒（兵）不算捉
    3. 将卒（帅兵）捉其他子不算捉
    4. 受牵制的子捉其他子不算捉
    5. 捉真根子不算捉
    
    算捉的情况：
    1. 捉假根子算捉
    2. 捉过河卒（兵）算捉
    3. 马捉被绊脚马，算捉
    
    特殊情况：
    1. 除车卒(兵)外其他子捉车，无论车是否有根，算捉
    2. 捉同类子不算捉，但一方受牵制不能吃子时，算捉
    
    Args:
        board_before: 走子前的棋盘
        board_after: 走子后的棋盘
        from_sq: 起始格子
        to_sq: 目标格子
        player: 走子方
        
    Returns:
        (is_chase, chased_piece_sq, chased_piece_type) - 是否捉、被捉子位置、被捉子类型
    """
    from_row = from_sq // BOARD_WIDTH
    from_col = from_sq % BOARD_WIDTH
    to_row = to_sq // BOARD_WIDTH
    to_col = to_sq % BOARD_WIDTH
    
    moved_piece = board_before[from_row, from_col]
    moved_piece_type = get_piece_type(moved_piece)
    
    enemy_player = 1 - player
    
    # 规则：将/帅/兵/卒 捉其他子不算捉
    is_king_or_pawn = (moved_piece_type == 1) | (moved_piece_type == 7)
    
    # 规则：受牵制的子捉其他子不算捉
    is_attacker_pinned = is_piece_pinned(board_after, to_row, to_col, player)
    
    # 基本排除条件
    basic_exclude = is_king_or_pawn | is_attacker_pinned
    
    # 找出走子后能攻击到的所有敌方棋子
    def check_target(sq):
        row = sq // BOARD_WIDTH
        col = sq % BOARD_WIDTH
        target_piece = board_after[row, col]
        target_type = get_piece_type(target_piece)
        
        # 必须是敌方棋子
        is_enemy = is_enemy_piece(target_piece, player)
        
        # 不能是将/帅（捉将是将军，不是捉）
        is_not_king = target_type != 1
        
        # 检查能否攻击到
        can_attack = can_piece_attack(board_after, to_row, to_col, row, col, player)
        
        # 检查走子前是否已经能攻击（原本就能攻击的不算新捉）
        could_attack_before = can_piece_attack(board_before, from_row, from_col, row, col, player)
        
        # 新产生的攻击才算捉
        is_new_attack = can_attack & ~could_attack_before
        
        return is_enemy & is_not_king & is_new_attack
    
    squares = jnp.arange(NUM_SQUARES)
    new_threats = jax.vmap(check_target)(squares)
    threat_count = jnp.sum(new_threats)
    
    # 规则：分捉多子不算捉
    is_single_threat = threat_count == 1
    
    # 找出被捉子
    threatened_sq = jnp.argmax(new_threats)
    threatened_row = threatened_sq // BOARD_WIDTH
    threatened_col = threatened_sq % BOARD_WIDTH
    threatened_piece = board_after[threatened_row, threatened_col]
    threatened_type = get_piece_type(threatened_piece)
    
    # 规则：捉未过河的卒（兵）不算捉
    is_pawn = threatened_type == 7
    pawn_crossed = has_crossed_river(threatened_row, enemy_player)
    pawn_not_crossed = is_pawn & ~pawn_crossed
    
    # 规则：捉同类子不算捉（除非对方被牵制不能吃）
    is_same_type = moved_piece_type == threatened_type
    enemy_can_capture = can_piece_attack(board_after, threatened_row, threatened_col, to_row, to_col, enemy_player)
    enemy_pinned = is_piece_pinned(board_after, threatened_row, threatened_col, enemy_player)
    same_type_exclude = is_same_type & enemy_can_capture & ~enemy_pinned
    
    # 规则：除车卒(兵)外其他子捉车，无论车是否有根，算捉
    is_rook_target = threatened_type == 5
    attacker_is_rook_or_pawn = (moved_piece_type == 5) | (moved_piece_type == 7)
    special_rook_chase = is_rook_target & ~attacker_is_rook_or_pawn
    
    # 规则：捉真根子不算捉，捉假根子算捉
    has_real_root = has_real_protector(board_after, threatened_row, threatened_col,
                                        to_row, to_col, enemy_player)
    
    # 综合判断
    # 算捉条件：
    # 1. 不是基本排除（将/兵捉子、被牵制）
    # 2. 单捉（不是分捉多子）
    # 3. 不是捉未过河卒
    # 4. 不是捉同类子（除非对方被牵制）
    # 5. 以下满足其一：
    #    a. 特殊捉车（非车非卒捉车）
    #    b. 捉无根子或假根子
    
    is_valid_chase = (
        ~basic_exclude & 
        is_single_threat & 
        ~pawn_not_crossed &
        ~same_type_exclude &
        (threat_count > 0) &
        (special_rook_chase | ~has_real_root)
    )
    
    # 返回：是否捉、被捉子位置、被捉子类型
    return is_valid_chase, threatened_sq, threatened_type


# ============================================================================
# 参与将军的棋子数量检测
# ============================================================================

@jax.jit
def count_checking_pieces(board: jnp.ndarray, player: jnp.ndarray) -> jnp.ndarray:
    """
    计算当前有多少己方棋子在将军对方
    
    Args:
        board: 棋盘状态
        player: 将军方（刚走完的一方）
        
    Returns:
        参与将军的棋子数量
    """
    enemy_player = 1 - player
    king_row, king_col = find_king(board, enemy_player)
    
    # 检查每个己方棋子是否在攻击敌方将/帅
    def is_checking(sq):
        row = sq // BOARD_WIDTH
        col = sq % BOARD_WIDTH
        piece = board[row, col]
        
        # 必须是己方棋子
        is_own = is_own_piece(piece, player)
        
        # 检查是否能攻击到敌方将/帅
        can_attack_king = can_piece_attack(board, row, col, king_row, king_col, player)
        
        return is_own & can_attack_king
    
    squares = jnp.arange(NUM_SQUARES)
    checking_mask = jax.vmap(is_checking)(squares)
    
    return jnp.sum(checking_mask)


# ============================================================================
# 违规状态追踪
# ============================================================================

@jax.jit
def get_perpetual_check_limit(num_pieces: jnp.ndarray) -> jnp.ndarray:
    """
    根据参与将军的子数返回长将限制回合数
    
    Args:
        num_pieces: 参与将军的棋子数
        
    Returns:
        允许的最大回合数
    """
    return jnp.where(
        num_pieces <= 1, PERPETUAL_CHECK_LIMIT_1,
        jnp.where(num_pieces == 2, PERPETUAL_CHECK_LIMIT_2, PERPETUAL_CHECK_LIMIT_3)
    )


@jax.jit  
def get_check_chase_alt_limit(num_pieces: jnp.ndarray) -> jnp.ndarray:
    """
    根据参与将捉的子数返回交替限制回合数
    
    Args:
        num_pieces: 参与的棋子数
        
    Returns:
        允许的最大回合数
    """
    return jnp.where(num_pieces <= 1, CHECK_CHASE_ALT_LIMIT_1, CHECK_CHASE_ALT_LIMIT_N)


# ============================================================================
# 违规检测结果
# ============================================================================

@jax.jit
def check_violation(
    # 当前走子信息
    is_check: jnp.ndarray,           # 本步是否将军
    is_chase: jnp.ndarray,           # 本步是否捉
    checking_pieces: jnp.ndarray,    # 参与将军的子数
    is_capture: jnp.ndarray,         # 本步是否吃子
    
    # 红方违规计数
    red_check_count: jnp.ndarray,    # 红方连续将军回合
    red_chase_count: jnp.ndarray,    # 红方连续捉子回合
    red_alt_count: jnp.ndarray,      # 红方将捉交替回合
    red_max_check_pieces: jnp.ndarray,  # 红方长将中最大参与子数
    
    # 黑方违规计数
    black_check_count: jnp.ndarray,
    black_chase_count: jnp.ndarray,
    black_alt_count: jnp.ndarray,
    black_max_check_pieces: jnp.ndarray,
    
    # 当前玩家（刚走完的一方）
    current_player: jnp.ndarray,
) -> tuple:
    """
    检测违规并更新计数
    
    规则：
    - 长将：1子6回合、2子12回合、3子+18回合
    - 长捉：6回合
    - 将捉交替：1子12回合、多子18回合
    - 吃子后重新计算
    
    Returns:
        (
            violation_type,  # 0=无违规, 1=长将, 2=长捉, 3=将捉交替
            violator,        # 违规方 (0=红, 1=黑, -1=无)
            new_red_check_count, new_red_chase_count, new_red_alt_count, new_red_max_check_pieces,
            new_black_check_count, new_black_chase_count, new_black_alt_count, new_black_max_check_pieces,
        )
    """
    is_red = current_player == 0
    
    # 如果吃子，双方计数都重置
    reset_counts = is_capture
    
    # ========== 更新红方计数 ==========
    # 将军计数
    new_red_check = jnp.where(
        reset_counts, 0,
        jnp.where(is_red & is_check, red_check_count + 1, 
                  jnp.where(is_red & ~is_check, 0, red_check_count))
    )
    
    # 捉子计数
    new_red_chase = jnp.where(
        reset_counts, 0,
        jnp.where(is_red & is_chase & ~is_check, red_chase_count + 1,
                  jnp.where(is_red & ~is_chase, 0, red_chase_count))
    )
    
    # 将捉交替计数（将或捉都累加）
    new_red_alt = jnp.where(
        reset_counts, 0,
        jnp.where(is_red & (is_check | is_chase), red_alt_count + 1,
                  jnp.where(is_red & ~is_check & ~is_chase, 0, red_alt_count))
    )
    
    # 更新最大参与子数
    new_red_max_pieces = jnp.where(
        reset_counts | (is_red & ~is_check), 0,
        jnp.where(is_red & is_check, jnp.maximum(red_max_check_pieces, checking_pieces), red_max_check_pieces)
    )
    
    # ========== 更新黑方计数 ==========
    new_black_check = jnp.where(
        reset_counts, 0,
        jnp.where(~is_red & is_check, black_check_count + 1,
                  jnp.where(~is_red & ~is_check, 0, black_check_count))
    )
    
    new_black_chase = jnp.where(
        reset_counts, 0,
        jnp.where(~is_red & is_chase & ~is_check, black_chase_count + 1,
                  jnp.where(~is_red & ~is_chase, 0, black_chase_count))
    )
    
    new_black_alt = jnp.where(
        reset_counts, 0,
        jnp.where(~is_red & (is_check | is_chase), black_alt_count + 1,
                  jnp.where(~is_red & ~is_check & ~is_chase, 0, black_alt_count))
    )
    
    new_black_max_pieces = jnp.where(
        reset_counts | (~is_red & ~is_check), 0,
        jnp.where(~is_red & is_check, jnp.maximum(black_max_check_pieces, checking_pieces), black_max_check_pieces)
    )
    
    # ========== 检测违规 ==========
    
    # 红方长将违规
    red_check_limit = get_perpetual_check_limit(new_red_max_pieces)
    red_perpetual_check = new_red_check >= red_check_limit
    
    # 红方长捉违规
    red_perpetual_chase = new_red_chase >= PERPETUAL_CHASE_LIMIT
    
    # 红方将捉交替违规
    red_alt_limit = get_check_chase_alt_limit(new_red_max_pieces)
    # 只有将和捉交替出现才算（纯将或纯捉不算交替）
    red_has_both = (new_red_check > 0) | (new_red_chase > 0)  # 简化判断
    red_check_chase_alt = (new_red_alt >= red_alt_limit) & red_has_both
    
    # 黑方长将违规
    black_check_limit = get_perpetual_check_limit(new_black_max_pieces)
    black_perpetual_check = new_black_check >= black_check_limit
    
    # 黑方长捉违规
    black_perpetual_chase = new_black_chase >= PERPETUAL_CHASE_LIMIT
    
    # 黑方将捉交替违规
    black_alt_limit = get_check_chase_alt_limit(new_black_max_pieces)
    black_has_both = (new_black_check > 0) | (new_black_chase > 0)
    black_check_chase_alt = (new_black_alt >= black_alt_limit) & black_has_both
    
    # 确定违规类型和违规方
    # 优先级：先达到上限的一方变招
    # 这里简化处理：当前走子方如果违规则判负
    
    red_violation = red_perpetual_check | red_perpetual_chase | red_check_chase_alt
    black_violation = black_perpetual_check | black_perpetual_chase | black_check_chase_alt
    
    # 确定违规类型
    violation_type = jnp.where(
        red_perpetual_check | black_perpetual_check, 1,  # 长将
        jnp.where(
            red_perpetual_chase | black_perpetual_chase, 2,  # 长捉
            jnp.where(
                red_check_chase_alt | black_check_chase_alt, 3,  # 将捉交替
                0  # 无违规
            )
        )
    )
    
    # 确定违规方（当前走子方优先判负）
    violator = jnp.where(
        is_red & red_violation, 0,  # 红方违规
        jnp.where(
            ~is_red & black_violation, 1,  # 黑方违规
            jnp.int32(-1)  # 无违规
        )
    )
    
    return (
        violation_type, violator,
        new_red_check, new_red_chase, new_red_alt, new_red_max_pieces,
        new_black_check, new_black_chase, new_black_alt, new_black_max_pieces,
    )


# ============================================================================
# 无吃子将军计数（用于60回合和棋规则）
# ============================================================================

@jax.jit
def update_no_capture_check_count(
    no_capture_count: jnp.ndarray,
    check_in_no_capture: jnp.ndarray,
    is_capture: jnp.ndarray,
    is_check: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    更新无吃子步数和其中的将军计数
    
    规则：无吃子120步(60回合)判和，其中将军(双方)最多只累计20回合(40步)
    
    Args:
        no_capture_count: 当前无吃子步数
        check_in_no_capture: 无吃子期间的将军步数
        is_capture: 本步是否吃子
        is_check: 本步是否将军
        
    Returns:
        (new_no_capture_count, new_check_in_no_capture, is_draw)
    """
    # 吃子后重置
    new_no_capture = jnp.where(is_capture, 0, no_capture_count + 1)
    new_check_count = jnp.where(is_capture, 0, 
                                 jnp.where(is_check, check_in_no_capture + 1, check_in_no_capture))
    
    # 有效无吃子步数 = 总步数 - 超出20回合的将军步数
    # 将军最多累计20回合(40步)
    excess_check = jnp.maximum(0, new_check_count - MAX_CHECK_IN_NO_CAPTURE * 2)
    effective_no_capture = new_no_capture - excess_check
    
    # 判和条件：有效无吃子步数达到120步
    is_draw = effective_no_capture >= 120
    
    return new_no_capture, new_check_count, is_draw

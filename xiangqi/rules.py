"""
中国象棋规则验证模块 (纯 JAX 实现)
支持 JIT 编译和向量化操作
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
from functools import partial
from xiangqi.actions import (
    BOARD_HEIGHT, BOARD_WIDTH, NUM_SQUARES, ACTION_SPACE_SIZE,
    EMPTY, R_KING, R_ADVISOR, R_BISHOP, R_KNIGHT, R_ROOK, R_CANNON, R_PAWN,
    B_KING, B_ADVISOR, B_BISHOP, B_KNIGHT, B_ROOK, B_CANNON, B_PAWN,
    _ACTION_TO_FROM_SQ, _ACTION_TO_TO_SQ,
)

# ============================================================================
# 区域定义
# ============================================================================

# 红方九宫格 (row 0-2, col 3-5)
RED_PALACE_ROWS = jnp.array([0, 1, 2])
RED_PALACE_COLS = jnp.array([3, 4, 5])

# 黑方九宫格 (row 7-9, col 3-5)
BLACK_PALACE_ROWS = jnp.array([7, 8, 9])
BLACK_PALACE_COLS = jnp.array([3, 4, 5])

# 红方区域 (row 0-4)
RED_TERRITORY_MAX_ROW = 4

# 黑方区域 (row 5-9)
BLACK_TERRITORY_MIN_ROW = 5

# 河界
RIVER_ROW_RED = 4    # 红方最远可达的行 (过河前)
RIVER_ROW_BLACK = 5  # 黑方最远可达的行 (过河前)

# 象的合法位置 (预计算)
RED_BISHOP_POSITIONS = jnp.array([
    [0, 2], [0, 6],  # 第0行
    [2, 0], [2, 4], [2, 8],  # 第2行
    [4, 2], [4, 6],  # 第4行
])

BLACK_BISHOP_POSITIONS = jnp.array([
    [5, 2], [5, 6],  # 第5行
    [7, 0], [7, 4], [7, 8],  # 第7行
    [9, 2], [9, 6],  # 第9行
])


# ============================================================================
# 辅助函数
# ============================================================================

@jax.jit
def is_red_piece(piece: jnp.ndarray) -> jnp.ndarray:
    """检查是否为红方棋子"""
    return piece > 0


@jax.jit
def is_black_piece(piece: jnp.ndarray) -> jnp.ndarray:
    """检查是否为黑方棋子"""
    return piece < 0


@jax.jit
def is_own_piece(piece: jnp.ndarray, player: jnp.ndarray) -> jnp.ndarray:
    """检查是否为己方棋子 (player: 0=红, 1=黑)"""
    return jnp.where(player == 0, piece > 0, piece < 0)


@jax.jit
def is_enemy_piece(piece: jnp.ndarray, player: jnp.ndarray) -> jnp.ndarray:
    """检查是否为敌方棋子"""
    return jnp.where(player == 0, piece < 0, piece > 0)


@jax.jit
def get_piece_type(piece: jnp.ndarray) -> jnp.ndarray:
    """获取棋子类型 (绝对值)"""
    return jnp.abs(piece)


@jax.jit
def in_palace(row: jnp.ndarray, col: jnp.ndarray, player: jnp.ndarray) -> jnp.ndarray:
    """检查位置是否在九宫格内"""
    in_col = (col >= 3) & (col <= 5)
    in_red_palace = (row >= 0) & (row <= 2) & in_col
    in_black_palace = (row >= 7) & (row <= 9) & in_col
    return jnp.where(player == 0, in_red_palace, in_black_palace)


@jax.jit
def in_own_territory(row: jnp.ndarray, player: jnp.ndarray) -> jnp.ndarray:
    """检查位置是否在己方区域"""
    return jnp.where(player == 0, row <= RED_TERRITORY_MAX_ROW, row >= BLACK_TERRITORY_MIN_ROW)


@jax.jit
def has_crossed_river(row: jnp.ndarray, player: jnp.ndarray) -> jnp.ndarray:
    """检查是否已过河"""
    return jnp.where(player == 0, row > RED_TERRITORY_MAX_ROW, row < BLACK_TERRITORY_MIN_ROW)


# ============================================================================
# 单个棋子移动验证
# ============================================================================

@jax.jit
def _is_valid_king_move(
    board: jnp.ndarray,
    from_row: jnp.ndarray, from_col: jnp.ndarray,
    to_row: jnp.ndarray, to_col: jnp.ndarray,
    player: jnp.ndarray
) -> jnp.ndarray:
    """验证将/帅的移动"""
    # 必须在九宫格内
    in_palace_check = in_palace(to_row, to_col, player)
    
    # 只能走一格 (上下左右)
    dr = jnp.abs(to_row - from_row)
    dc = jnp.abs(to_col - from_col)
    one_step = ((dr == 1) & (dc == 0)) | ((dr == 0) & (dc == 1))
    
    return in_palace_check & one_step


@jax.jit
def _is_valid_advisor_move(
    board: jnp.ndarray,
    from_row: jnp.ndarray, from_col: jnp.ndarray,
    to_row: jnp.ndarray, to_col: jnp.ndarray,
    player: jnp.ndarray
) -> jnp.ndarray:
    """验证士的移动"""
    # 必须在九宫格内
    in_palace_check = in_palace(to_row, to_col, player)
    
    # 只能斜走一格
    dr = jnp.abs(to_row - from_row)
    dc = jnp.abs(to_col - from_col)
    diagonal_one = (dr == 1) & (dc == 1)
    
    return in_palace_check & diagonal_one


@jax.jit
def _is_valid_bishop_move(
    board: jnp.ndarray,
    from_row: jnp.ndarray, from_col: jnp.ndarray,
    to_row: jnp.ndarray, to_col: jnp.ndarray,
    player: jnp.ndarray
) -> jnp.ndarray:
    """验证象的移动"""
    # 必须在己方区域
    in_territory = in_own_territory(to_row, player)
    
    # 走田字 (斜走两格)
    dr = to_row - from_row
    dc = to_col - from_col
    is_field_move = (jnp.abs(dr) == 2) & (jnp.abs(dc) == 2)
    
    # 检查象眼 (中间位置不能有棋子)
    eye_row = from_row + dr // 2
    eye_col = from_col + dc // 2
    eye_blocked = board[eye_row, eye_col] != EMPTY
    
    return in_territory & is_field_move & ~eye_blocked


@jax.jit
def _is_valid_knight_move(
    board: jnp.ndarray,
    from_row: jnp.ndarray, from_col: jnp.ndarray,
    to_row: jnp.ndarray, to_col: jnp.ndarray,
    player: jnp.ndarray
) -> jnp.ndarray:
    """验证马的移动"""
    dr = to_row - from_row
    dc = to_col - from_col
    
    # 走日字
    is_knight_move = ((jnp.abs(dr) == 2) & (jnp.abs(dc) == 1)) | \
                     ((jnp.abs(dr) == 1) & (jnp.abs(dc) == 2))
    
    # 检查蹩马腿
    # 如果 |dr| == 2, 马腿在 (from_row + sign(dr), from_col)
    # 如果 |dc| == 2, 马腿在 (from_row, from_col + sign(dc))
    leg_row = jnp.where(jnp.abs(dr) == 2, from_row + jnp.sign(dr), from_row)
    leg_col = jnp.where(jnp.abs(dc) == 2, from_col + jnp.sign(dc), from_col)
    leg_blocked = board[leg_row, leg_col] != EMPTY
    
    return is_knight_move & ~leg_blocked


@jax.jit
def _is_valid_rook_move(
    board: jnp.ndarray,
    from_row: jnp.ndarray, from_col: jnp.ndarray,
    to_row: jnp.ndarray, to_col: jnp.ndarray,
    player: jnp.ndarray
) -> jnp.ndarray:
    """验证车的移动"""
    dr = to_row - from_row
    dc = to_col - from_col
    
    # 必须走直线
    is_straight = (dr == 0) | (dc == 0)
    is_same_position = (dr == 0) & (dc == 0)
    
    # 检查路径上没有其他棋子
    def check_path_clear():
        # 计算路径长度和方向
        steps = jnp.maximum(jnp.abs(dr), jnp.abs(dc))
        step_r = jnp.where(steps > 0, jnp.sign(dr), 0)
        step_c = jnp.where(steps > 0, jnp.sign(dc), 0)
        
        # 检查从1到steps-1的所有位置
        # 使用 vmap 替代 Python for 循环以减少编译图大小
        i = jnp.arange(1, 10)
        pos_in_path = i < steps
        check_row = from_row + step_r * i
        check_col = from_col + step_c * i
        
        # 安全索引：确保不会越界 (虽然 jnp.arange 1..10 且棋盘 10x9 很难越界，但习惯上应保持严谨)
        check_row = jnp.clip(check_row, 0, BOARD_HEIGHT - 1)
        check_col = jnp.clip(check_col, 0, BOARD_WIDTH - 1)
        
        pieces_at_pos = board[check_row, check_col]
        path_clear = jnp.all(~pos_in_path | (pieces_at_pos == EMPTY))
        
        return path_clear
    
    path_clear = check_path_clear()
    
    return is_straight & ~is_same_position & path_clear


@jax.jit
def _is_valid_cannon_move(
    board: jnp.ndarray,
    from_row: jnp.ndarray, from_col: jnp.ndarray,
    to_row: jnp.ndarray, to_col: jnp.ndarray,
    player: jnp.ndarray
) -> jnp.ndarray:
    """验证炮的移动"""
    dr = to_row - from_row
    dc = to_col - from_col
    
    # 必须走直线
    is_straight = (dr == 0) | (dc == 0)
    is_same_position = (dr == 0) & (dc == 0)
    
    # 目标位置的棋子
    target_piece = board[to_row, to_col]
    is_capture = target_piece != EMPTY
    
    # 计算路径上的棋子数量
    steps = jnp.maximum(jnp.abs(dr), jnp.abs(dc))
    step_r = jnp.where(steps > 0, jnp.sign(dr), 0)
    step_c = jnp.where(steps > 0, jnp.sign(dc), 0)
    
    # 使用 vmap 替代 Python for 循环
    i = jnp.arange(1, 10)
    pos_in_path = i < steps
    check_row = jnp.clip(from_row + step_r * i, 0, BOARD_HEIGHT - 1)
    check_col = jnp.clip(from_col + step_c * i, 0, BOARD_WIDTH - 1)
    
    pieces_at_pos = board[check_row, check_col]
    pieces_in_path = jnp.sum(jnp.where(pos_in_path & (pieces_at_pos != EMPTY), 1, 0))
    
    # 移动：路径上没有棋子，目标为空
    # 吃子：路径上恰好一个棋子（炮架）
    valid_move = is_straight & ~is_same_position & (
        (~is_capture & (pieces_in_path == 0)) |
        (is_capture & (pieces_in_path == 1))
    )
    
    return valid_move


@jax.jit
def _is_valid_pawn_move(
    board: jnp.ndarray,
    from_row: jnp.ndarray, from_col: jnp.ndarray,
    to_row: jnp.ndarray, to_col: jnp.ndarray,
    player: jnp.ndarray
) -> jnp.ndarray:
    """验证兵/卒的移动"""
    dr = to_row - from_row
    dc = to_col - from_col
    
    # 红方向上走 (row 增加), 黑方向下走 (row 减少)
    forward_dir = jnp.where(player == 0, 1, -1)
    
    # 是否过河
    crossed = has_crossed_river(from_row, player)
    
    # 前进一格
    forward_one = (dr == forward_dir) & (dc == 0)
    
    # 过河后可以左右移动
    sideways = crossed & (dr == 0) & (jnp.abs(dc) == 1)
    
    return forward_one | sideways


# ============================================================================
# 综合移动验证
# ============================================================================

@jax.jit
def is_valid_move_for_piece(
    board: jnp.ndarray,
    from_row: jnp.ndarray, from_col: jnp.ndarray,
    to_row: jnp.ndarray, to_col: jnp.ndarray,
    piece_type: jnp.ndarray,
    player: jnp.ndarray
) -> jnp.ndarray:
    """根据棋子类型验证移动"""
    # 使用 switch/case 风格的条件选择
    valid = jnp.where(
        piece_type == 1,  # KING
        _is_valid_king_move(board, from_row, from_col, to_row, to_col, player),
        jnp.where(
            piece_type == 2,  # ADVISOR
            _is_valid_advisor_move(board, from_row, from_col, to_row, to_col, player),
            jnp.where(
                piece_type == 3,  # BISHOP
                _is_valid_bishop_move(board, from_row, from_col, to_row, to_col, player),
                jnp.where(
                    piece_type == 4,  # KNIGHT
                    _is_valid_knight_move(board, from_row, from_col, to_row, to_col, player),
                    jnp.where(
                        piece_type == 5,  # ROOK
                        _is_valid_rook_move(board, from_row, from_col, to_row, to_col, player),
                        jnp.where(
                            piece_type == 6,  # CANNON
                            _is_valid_cannon_move(board, from_row, from_col, to_row, to_col, player),
                            jnp.where(
                                piece_type == 7,  # PAWN
                                _is_valid_pawn_move(board, from_row, from_col, to_row, to_col, player),
                                False  # 空或无效棋子
                            )
                        )
                    )
                )
            )
        )
    )
    return valid


@jax.jit
def is_valid_move(
    board: jnp.ndarray,
    from_sq: jnp.ndarray,
    to_sq: jnp.ndarray,
    player: jnp.ndarray
) -> jnp.ndarray:
    """
    验证一个移动是否合法（不考虑将军）
    
    Args:
        board: 棋盘状态 (10, 9)
        from_sq: 起始格子索引
        to_sq: 目标格子索引
        player: 当前玩家 (0=红, 1=黑)
        
    Returns:
        是否合法
    """
    from_row = from_sq // BOARD_WIDTH
    from_col = from_sq % BOARD_WIDTH
    to_row = to_sq // BOARD_WIDTH
    to_col = to_sq % BOARD_WIDTH
    
    # 边界检查
    valid_bounds = (from_row >= 0) & (from_row < BOARD_HEIGHT) & \
                   (from_col >= 0) & (from_col < BOARD_WIDTH) & \
                   (to_row >= 0) & (to_row < BOARD_HEIGHT) & \
                   (to_col >= 0) & (to_col < BOARD_WIDTH)
    
    # 获取起始位置的棋子
    piece = board[from_row, from_col]
    
    # 必须是己方棋子
    is_own = is_own_piece(piece, player)
    
    # 目标位置不能是己方棋子
    target_piece = board[to_row, to_col]
    target_not_own = ~is_own_piece(target_piece, player) | (target_piece == EMPTY)
    
    # 根据棋子类型验证移动
    piece_type = get_piece_type(piece)
    valid_piece_move = is_valid_move_for_piece(
        board, from_row, from_col, to_row, to_col, piece_type, player
    )
    
    return valid_bounds & is_own & target_not_own & valid_piece_move


# ============================================================================
# 将军检测
# ============================================================================

@jax.jit
def find_king(board: jnp.ndarray, player: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """找到指定玩家的将/帅位置"""
    king_piece = jnp.where(player == 0, R_KING, B_KING)
    king_mask = board == king_piece
    king_idx = jnp.argmax(king_mask.flatten())
    king_row = king_idx // BOARD_WIDTH
    king_col = king_idx % BOARD_WIDTH
    return king_row, king_col


@jax.jit
def is_in_check(board: jnp.ndarray, player: jnp.ndarray) -> jnp.ndarray:
    """
    以将/帅为中心的快速将军检测 (King-centric)
    由原来的 O(90) 降低为 O(1) 的固定检测，极大减少 XLA 图节点数量
    """
    king_row, king_col = find_king(board, player)
    
    # 敌方棋子类型
    is_red = player == 0
    E_KING = jnp.where(is_red, B_KING, R_KING)
    E_ROOK = jnp.where(is_red, B_ROOK, R_ROOK)
    E_CANNON = jnp.where(is_red, B_CANNON, R_CANNON)
    E_KNIGHT = jnp.where(is_red, B_KNIGHT, R_KNIGHT)
    E_PAWN = jnp.where(is_red, B_PAWN, R_PAWN)
    
    # 1. 探测四个方向的直线攻击 (车、炮、王见王)
    def scan_line(dr, dc):
        # 使用 arange 探测
        idx = jnp.arange(1, 10)
        rows = jnp.clip(king_row + dr * idx, 0, BOARD_HEIGHT - 1)
        cols = jnp.clip(king_col + dc * idx, 0, BOARD_WIDTH - 1)
        
        # 边界掩码
        in_bounds = (king_row + dr * idx >= 0) & (king_row + dr * idx < BOARD_HEIGHT) & \
                    (king_col + dc * idx >= 0) & (king_col + dc * idx < BOARD_WIDTH)
        
        pieces = jnp.where(in_bounds, board[rows, cols], EMPTY)
        
        # 找到第一个和第二个非空棋子
        is_piece = pieces != EMPTY
        first_idx = jnp.argmax(is_piece)
        first_exists = jnp.any(is_piece)
        first_piece = jnp.where(first_exists, pieces[first_idx], EMPTY)
        
        # 找第二个棋子
        mask_after_first = jnp.arange(9) > first_idx
        second_exists = jnp.any(is_piece & mask_after_first)
        second_idx = jnp.argmax(is_piece & mask_after_first)
        second_piece = jnp.where(second_exists, pieces[second_idx], EMPTY)
        
        # 将见将/车将军：第一个棋子是敌方王或车
        # 注意：王见王仅在同列(dc=0)有效
        is_king_face_to_face = (dc == 0) & (first_piece == E_KING)
        is_rook_check = first_piece == E_ROOK
        # 炮将军：第二个棋子是敌方炮
        is_cannon_check = second_piece == E_CANNON
        
        return is_king_face_to_face | is_rook_check | is_cannon_check

    check_h = scan_line(0, 1) | scan_line(0, -1)
    check_v = scan_line(1, 0) | scan_line(-1, 0)
    
    # 2. 探测马的攻击
    knight_deltas = jnp.array([
        [-2, -1], [-2, 1], [-1, -2], [-1, 2],
        [1, -2], [1, 2], [2, -1], [2, 1]
    ])
    # 马腿位置
    leg_deltas = jnp.array([
        [-1, 0], [-1, 0], [0, -1], [0, 1],
        [0, -1], [0, 1], [1, 0], [1, 0]
    ])
    
    def check_knight(i):
        dr, dc = knight_deltas[i]
        lr, lc = leg_deltas[i]
        tr, tc = king_row + dr, king_col + dc
        lr, lc = king_row + lr, king_col + lc
        
        in_bounds = (tr >= 0) & (tr < BOARD_HEIGHT) & (tc >= 0) & (tc < BOARD_WIDTH)
        leg_in_bounds = (lr >= 0) & (lr < BOARD_HEIGHT) & (lc >= 0) & (lc < BOARD_WIDTH)
        
        # 无蹩马腿且目标是敌方马
        return in_bounds & leg_in_bounds & (board[lr, lc] == EMPTY) & (board[tr, tc] == E_KNIGHT)

    check_knight_all = jnp.any(jax.vmap(check_knight)(jnp.arange(8)))
    
    # 3. 探测兵的攻击
    # 红方将(帅)在下方，会被黑方卒(向下走)攻击：dr = +1
    # 黑方将(将)在上方，会被红方兵(向上走)攻击：dr = -1
    pawn_dr = jnp.where(is_red, 1, -1)
    pawn_deltas = jnp.array([[pawn_dr, 0], [0, -1], [0, 1]])
    
    def check_pawn(i):
        dr, dc = pawn_deltas[i]
        tr, tc = king_row + dr, king_col + dc
        in_bounds = (tr >= 0) & (tr < BOARD_HEIGHT) & (tc >= 0) & (tc < BOARD_WIDTH)
        return in_bounds & (board[tr, tc] == E_PAWN)
        
    check_pawn_all = jnp.any(jax.vmap(check_pawn)(jnp.arange(3)))
    
    return check_h | check_v | check_knight_all | check_pawn_all


@jax.jit
def apply_move(board: jnp.ndarray, from_sq: jnp.ndarray, to_sq: jnp.ndarray) -> jnp.ndarray:
    """应用一个移动，返回新棋盘"""
    from_row = from_sq // BOARD_WIDTH
    from_col = from_sq % BOARD_WIDTH
    to_row = to_sq // BOARD_WIDTH
    to_col = to_sq % BOARD_WIDTH
    
    piece = board[from_row, from_col]
    new_board = board.at[from_row, from_col].set(EMPTY)
    new_board = new_board.at[to_row, to_col].set(piece)
    
    return new_board


@jax.jit
def is_legal_move(
    board: jnp.ndarray,
    from_sq: jnp.ndarray,
    to_sq: jnp.ndarray,
    player: jnp.ndarray
) -> jnp.ndarray:
    """
    验证一个移动是否完全合法（包括不能送将）
    
    Args:
        board: 棋盘状态
        from_sq: 起始格子
        to_sq: 目标格子
        player: 当前玩家
        
    Returns:
        是否合法
    """
    # 首先验证基本移动规则
    basic_valid = is_valid_move(board, from_sq, to_sq, player)
    
    # 模拟移动后检查是否被将军
    new_board = apply_move(board, from_sq, to_sq)
    still_in_check = is_in_check(new_board, player)
    
    return basic_valid & ~still_in_check


# ============================================================================
# 合法动作生成
# ============================================================================

def get_legal_moves_mask(board: jnp.ndarray, player: jnp.ndarray) -> jnp.ndarray:
    """
    获取所有合法动作的掩码
    
    Args:
        board: 棋盘状态 (10, 9)
        player: 当前玩家 (0=红, 1=黑)
        
    Returns:
        合法动作掩码 (ACTION_SPACE_SIZE,) bool 数组
    """
    # 对每个可能的动作检查是否合法
    def check_action(action_id):
        from_sq = _ACTION_TO_FROM_SQ[action_id]
        to_sq = _ACTION_TO_TO_SQ[action_id]
        return is_legal_move(board, from_sq, to_sq, player)
    
    # 向量化检查所有动作
    action_ids = jnp.arange(ACTION_SPACE_SIZE, dtype=jnp.int32)
    legal_mask = jax.vmap(check_action)(action_ids)
    
    return legal_mask


def get_legal_moves(board: jnp.ndarray, player: jnp.ndarray) -> list:
    """获取所有合法移动的列表 (非 JIT，用于调试)"""
    mask = get_legal_moves_mask(board, player)
    legal_actions = jnp.where(mask)[0]
    moves = []
    for action in legal_actions:
        from_sq = int(_ACTION_TO_FROM_SQ[action])
        to_sq = int(_ACTION_TO_TO_SQ[action])
        moves.append((from_sq, to_sq))
    return moves


# ============================================================================
# 游戏结束检测
# ============================================================================

@jax.jit
def is_game_over(board: jnp.ndarray, player: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    检查游戏是否结束
    
    Args:
        board: 棋盘状态
        player: 当前玩家
        
    Returns:
        (is_over, winner) - winner: 0=红, 1=黑, -1=平局
    """
    # 检查将/帅是否还在
    red_king_exists = jnp.any(board == R_KING)
    black_king_exists = jnp.any(board == B_KING)
    
    # 如果一方将/帅被吃
    red_wins = ~black_king_exists
    black_wins = ~red_king_exists
    
    # 检查是否有合法移动
    legal_mask = get_legal_moves_mask(board, player)
    has_legal_moves = jnp.any(legal_mask)
    
    # 无合法移动 = 被将死或被困毙
    no_moves = ~has_legal_moves
    
    # 游戏结束条件
    is_over = red_wins | black_wins | no_moves
    
    # 确定胜者
    winner = jnp.where(
        red_wins, 0,
        jnp.where(black_wins, 1, jnp.where(no_moves, 1 - player, -1))
    )
    
    return is_over, winner


# ============================================================================
# 初始棋盘
# ============================================================================

def get_initial_board() -> jnp.ndarray:
    """获取初始棋盘状态"""
    board = jnp.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=jnp.int8)
    
    # 红方 (下方, row 0-4)
    # 第0行: 车马象士帅士象马车
    board = board.at[0, 0].set(R_ROOK)
    board = board.at[0, 1].set(R_KNIGHT)
    board = board.at[0, 2].set(R_BISHOP)
    board = board.at[0, 3].set(R_ADVISOR)
    board = board.at[0, 4].set(R_KING)
    board = board.at[0, 5].set(R_ADVISOR)
    board = board.at[0, 6].set(R_BISHOP)
    board = board.at[0, 7].set(R_KNIGHT)
    board = board.at[0, 8].set(R_ROOK)
    # 第2行: 炮
    board = board.at[2, 1].set(R_CANNON)
    board = board.at[2, 7].set(R_CANNON)
    # 第3行: 兵
    board = board.at[3, 0].set(R_PAWN)
    board = board.at[3, 2].set(R_PAWN)
    board = board.at[3, 4].set(R_PAWN)
    board = board.at[3, 6].set(R_PAWN)
    board = board.at[3, 8].set(R_PAWN)
    
    # 黑方 (上方, row 5-9)
    # 第9行: 车马象士将士象马车
    board = board.at[9, 0].set(B_ROOK)
    board = board.at[9, 1].set(B_KNIGHT)
    board = board.at[9, 2].set(B_BISHOP)
    board = board.at[9, 3].set(B_ADVISOR)
    board = board.at[9, 4].set(B_KING)
    board = board.at[9, 5].set(B_ADVISOR)
    board = board.at[9, 6].set(B_BISHOP)
    board = board.at[9, 7].set(B_KNIGHT)
    board = board.at[9, 8].set(B_ROOK)
    # 第7行: 炮
    board = board.at[7, 1].set(B_CANNON)
    board = board.at[7, 7].set(B_CANNON)
    # 第6行: 卒
    board = board.at[6, 0].set(B_PAWN)
    board = board.at[6, 2].set(B_PAWN)
    board = board.at[6, 4].set(B_PAWN)
    board = board.at[6, 6].set(B_PAWN)
    board = board.at[6, 8].set(B_PAWN)
    
    return board

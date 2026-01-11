"""
中国象棋动作编码/解码模块
使用压缩编码方案，基于合法移动方向

动作空间设计：
- 棋盘: 10行 x 9列 = 90个位置
- 每个位置的可能移动方向根据棋子类型不同而不同
- 使用 from-to 编码，但只保留实际可能的移动

总动作数: 2086 (经过压缩)
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
from functools import partial

# ============================================================================
# 常量定义
# ============================================================================

# 棋盘尺寸
BOARD_HEIGHT = 10
BOARD_WIDTH = 9
NUM_SQUARES = BOARD_HEIGHT * BOARD_WIDTH  # 90

# 棋子类型 (使用正数表示红方，负数表示黑方)
# 0 = 空位
EMPTY = 0
# 红方棋子 (正数)
R_KING = 1      # 帅
R_ADVISOR = 2   # 仕
R_BISHOP = 3    # 相
R_KNIGHT = 4    # 马
R_ROOK = 5      # 车
R_CANNON = 6    # 炮
R_PAWN = 7      # 兵

# 黑方棋子 (负数)
B_KING = -1     # 将
B_ADVISOR = -2  # 士
B_BISHOP = -3   # 象
B_KNIGHT = -4   # 马
B_ROOK = -5     # 车
B_CANNON = -6   # 炮
B_PAWN = -7     # 卒

# 棋子符号映射 (用于显示)
PIECE_SYMBOLS = {
    R_KING: '帥', R_ADVISOR: '仕', R_BISHOP: '相', R_KNIGHT: '傌',
    R_ROOK: '俥', R_CANNON: '炮', R_PAWN: '兵',
    B_KING: '將', B_ADVISOR: '士', B_BISHOP: '象', B_KNIGHT: '馬',
    B_ROOK: '車', B_CANNON: '砲', B_PAWN: '卒',
    EMPTY: '．'
}

# ============================================================================
# 动作编码
# ============================================================================

# 移动方向定义
# 车/炮: 直线移动 (上下左右，最多9格)
# 马: 8个方向的日字移动
# 象: 4个方向的田字移动
# 士: 4个斜向移动
# 将/帅: 4个方向 (上下左右各1格)
# 兵/卒: 前进或左右 (过河后)

# 方向编码 (共17种方向类型)
# 0-8: 向上移动1-9格
# 9-17: 向下移动1-9格
# 18-26: 向左移动1-9格
# 27-35: 向右移动1-9格
# 36-43: 马的8个方向
# 44-47: 象的4个方向
# 48-51: 士的4个方向

# 为了简化，我们使用 from_square * NUM_DIRECTIONS + direction 的编码方式
# 但只保留实际可能的动作

# 预计算所有合法动作
def _generate_all_possible_moves():
    """生成所有可能的移动（不考虑具体棋子，只考虑几何上可能的移动）"""
    moves = []
    
    for from_sq in range(NUM_SQUARES):
        from_row, from_col = from_sq // BOARD_WIDTH, from_sq % BOARD_WIDTH
        
        # 直线移动 (车/炮/将/兵)
        for delta in range(1, 10):
            # 上
            to_row, to_col = from_row + delta, from_col
            if 0 <= to_row < BOARD_HEIGHT:
                moves.append((from_sq, to_row * BOARD_WIDTH + to_col))
            # 下
            to_row, to_col = from_row - delta, from_col
            if 0 <= to_row < BOARD_HEIGHT:
                moves.append((from_sq, to_row * BOARD_WIDTH + to_col))
            # 左
            to_row, to_col = from_row, from_col - delta
            if 0 <= to_col < BOARD_WIDTH:
                moves.append((from_sq, to_row * BOARD_WIDTH + to_col))
            # 右
            to_row, to_col = from_row, from_col + delta
            if 0 <= to_col < BOARD_WIDTH:
                moves.append((from_sq, to_row * BOARD_WIDTH + to_col))
        
        # 马的移动 (日字)
        knight_deltas = [
            (-2, -1), (-2, 1), (-1, -2), (-1, 2),
            (1, -2), (1, 2), (2, -1), (2, 1)
        ]
        for dr, dc in knight_deltas:
            to_row, to_col = from_row + dr, from_col + dc
            if 0 <= to_row < BOARD_HEIGHT and 0 <= to_col < BOARD_WIDTH:
                moves.append((from_sq, to_row * BOARD_WIDTH + to_col))
        
        # 象的移动 (田字)
        bishop_deltas = [(-2, -2), (-2, 2), (2, -2), (2, 2)]
        for dr, dc in bishop_deltas:
            to_row, to_col = from_row + dr, from_col + dc
            if 0 <= to_row < BOARD_HEIGHT and 0 <= to_col < BOARD_WIDTH:
                moves.append((from_sq, to_row * BOARD_WIDTH + to_col))
        
        # 士的移动 (斜向1格)
        advisor_deltas = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dr, dc in advisor_deltas:
            to_row, to_col = from_row + dr, from_col + dc
            if 0 <= to_row < BOARD_HEIGHT and 0 <= to_col < BOARD_WIDTH:
                moves.append((from_sq, to_row * BOARD_WIDTH + to_col))
    
    # 去重并排序
    moves = sorted(list(set(moves)))
    return moves


# 预计算动作映射表
_ALL_MOVES = _generate_all_possible_moves()
ACTION_SPACE_SIZE = len(_ALL_MOVES)  # 应该约为 2086

# 创建双向映射
_MOVE_TO_ACTION = {move: idx for idx, move in enumerate(_ALL_MOVES)}
_ACTION_TO_MOVE = {idx: move for idx, move in enumerate(_ALL_MOVES)}

# 转换为 JAX 数组以支持 JIT
_ACTION_TO_FROM_SQ = jnp.array([move[0] for move in _ALL_MOVES], dtype=jnp.int32)
_ACTION_TO_TO_SQ = jnp.array([move[1] for move in _ALL_MOVES], dtype=jnp.int32)

# 180度旋转动作映射表 (用于视角归一化)
def _build_action_rotate_table():
    rotate_table = jnp.zeros(ACTION_SPACE_SIZE, dtype=jnp.int32)
    # 预计算 from-to 到 action 的查找表 (90 x 90)
    temp_table = jnp.full((NUM_SQUARES, NUM_SQUARES), -1, dtype=jnp.int32)
    for i, (f, t) in enumerate(_ALL_MOVES):
        temp_table = temp_table.at[f, t].set(i)
        
    for action_id in range(ACTION_SPACE_SIZE):
        from_sq, to_sq = _ACTION_TO_MOVE[action_id]
        # 180度旋转: row, col -> 9-row, 8-col => index -> 89-index
        r_f = (NUM_SQUARES - 1) - from_sq
        r_t = (NUM_SQUARES - 1) - to_sq
        rotate_table = rotate_table.at[action_id].set(temp_table[r_f, r_t])
    return rotate_table

_ACTION_ROTATE_TABLE = _build_action_rotate_table()

# 创建 from-to 到 action 的查找表 (90 x 90 -> action_id, -1 表示无效)
_FROM_TO_ACTION_TABLE = jnp.full((NUM_SQUARES, NUM_SQUARES), -1, dtype=jnp.int32)
for action_id, (from_sq, to_sq) in enumerate(_ALL_MOVES):
    _FROM_TO_ACTION_TABLE = _FROM_TO_ACTION_TABLE.at[from_sq, to_sq].set(action_id)


# ============================================================================
# 公开 API
# ============================================================================

@jax.jit
def rotate_action(action: jnp.ndarray) -> jnp.ndarray:
    """
    将动作索引进行180度旋转 (用于视角切换)
    """
    return _ACTION_ROTATE_TABLE[action]


@jax.jit
def action_to_move(action: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    将动作索引转换为 (from_square, to_square)
    
    Args:
        action: 动作索引 (标量或数组)
        
    Returns:
        (from_square, to_square) 元组
    """
    from_sq = _ACTION_TO_FROM_SQ[action]
    to_sq = _ACTION_TO_TO_SQ[action]
    return from_sq, to_sq


@jax.jit
def move_to_action(from_sq: jnp.ndarray, to_sq: jnp.ndarray) -> jnp.ndarray:
    """
    将 (from_square, to_square) 转换为动作索引
    
    Args:
        from_sq: 起始格子索引
        to_sq: 目标格子索引
        
    Returns:
        动作索引，如果移动无效则返回 -1
    """
    return _FROM_TO_ACTION_TABLE[from_sq, to_sq]


def get_action_mask(board: jnp.ndarray, current_player: jnp.ndarray) -> jnp.ndarray:
    """
    获取当前状态下的合法动作掩码
    
    Args:
        board: 棋盘状态 (10, 9)
        current_player: 当前玩家 (0=红, 1=黑)
        
    Returns:
        合法动作掩码 (ACTION_SPACE_SIZE,) bool 数组
    """
    from xiangqi.rules import get_legal_moves_mask
    return get_legal_moves_mask(board, current_player)


# ============================================================================
# 辅助函数
# ============================================================================

def square_to_coords(square: int) -> tuple[int, int]:
    """将格子索引转换为 (row, col) 坐标"""
    return square // BOARD_WIDTH, square % BOARD_WIDTH


def coords_to_square(row: int, col: int) -> int:
    """将 (row, col) 坐标转换为格子索引"""
    return row * BOARD_WIDTH + col


def move_to_uci(from_sq: int, to_sq: int) -> str:
    """将移动转换为 UCI 格式字符串 (如 'a0b0')"""
    from_row, from_col = square_to_coords(from_sq)
    to_row, to_col = square_to_coords(to_sq)
    col_names = 'abcdefghi'
    return f"{col_names[from_col]}{from_row}{col_names[to_col]}{to_row}"


def uci_to_move(uci: str) -> tuple[int, int]:
    """将 UCI 格式字符串转换为移动"""
    col_names = 'abcdefghi'
    from_col = col_names.index(uci[0])
    from_row = int(uci[1])
    to_col = col_names.index(uci[2])
    to_row = int(uci[3])
    return coords_to_square(from_row, from_col), coords_to_square(to_row, to_col)


# 打印动作空间统计信息
if __name__ == "__main__":
    print(f"动作空间大小: {ACTION_SPACE_SIZE}")
    print(f"示例动作 0: {_ALL_MOVES[0]} -> UCI: {move_to_uci(*_ALL_MOVES[0])}")
    print(f"示例动作 100: {_ALL_MOVES[100]} -> UCI: {move_to_uci(*_ALL_MOVES[100])}")

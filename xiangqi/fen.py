"""
中国象棋 FEN 解析
支持批量从文件加载局面
"""

from __future__ import annotations
from typing import List, Tuple
import numpy as np

from xiangqi.actions import BOARD_HEIGHT, BOARD_WIDTH

FEN_TO_PIECE = {
    'K': 1, 'A': 2, 'B': 3, 'N': 4, 'R': 5, 'C': 6, 'P': 7,
    'k': -1, 'a': -2, 'b': -3, 'n': -4, 'r': -5, 'c': -6, 'p': -7,
}


def parse_fen(fen: str) -> Tuple[np.ndarray, int]:
    """
    解析单条 FEN 字符串
    
    Args:
        fen: 标准 FEN 格式，如 "rnbakabnr/9/1c5c1/... w"
        
    Returns:
        (board, player): board (10,9) int8, player 0=红 1=黑
        
    Raises:
        ValueError: FEN 格式错误
    """
    parts = fen.strip().split()
    if not parts:
        raise ValueError(f"空 FEN: {fen!r}")
    board_str = parts[0]
    player = 0 if len(parts) < 2 or parts[1].lower() in ('w', 'r') else 1
    
    board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=np.int8)
    rows = board_str.split('/')
    if len(rows) != 10:
        raise ValueError(f"FEN 行数应为 10，实际 {len(rows)}: {fen!r}")
    
    for r_idx, r_str in enumerate(rows):
        row = 9 - r_idx  # FEN 从上到下，我们 row 0 是红方底线
        col = 0
        for char in r_str:
            if char.isdigit():
                col += int(char)
            elif char in FEN_TO_PIECE:
                board[row, col] = FEN_TO_PIECE[char]
                col += 1
            else:
                raise ValueError(f"FEN 非法字符 '{char}' 于 {fen!r}")
        if col != 9:
            raise ValueError(f"FEN 第 {r_idx+1} 行列数应为 9，实际 {col}: {fen!r}")
    
    return board, player


def load_fens_from_file(path: str) -> List[Tuple[np.ndarray, int]]:
    """
    从文件批量加载 FEN，每行一条，空行和 # 开头行忽略
    
    Args:
        path: 文件路径
        
    Returns:
        [(board, player), ...]
        
    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 某行 FEN 解析失败
    """
    result = []
    with open(path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            try:
                board, player = parse_fen(line)
                result.append((board, player))
            except ValueError as e:
                raise ValueError(f"第 {line_num} 行 FEN 解析失败: {e}") from e
    return result

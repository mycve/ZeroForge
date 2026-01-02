"""
ChineseChessGame - 中国象棋游戏实现

实现 Game 接口，封装 cchess 库。

特点:
- 历史状态编码：支持多步历史（用于神经网络输入）
- 绝对坐标：红方在下，黑方在上（不翻转棋盘）
- 动作空间：约 2086 个有效动作（精简版）
"""

import copy
import logging
from typing import List, Tuple, Dict, Any, Optional
import numpy as np

from . import cchess  # 本地 cchess 引擎

from core.game import Game, ObservationSpace, ActionSpace, GameState, GameMeta
from games import register_game

logger = logging.getLogger(__name__)


# ============================================================
# 常量定义
# ============================================================

BOARD_HEIGHT = 10
BOARD_WIDTH = 9
NUM_PIECE_TYPES = 14  # 红方 7 种 + 黑方 7 种
MAX_GAME_LENGTH = 200
DEFAULT_HISTORY_STEPS = 4

# 输入通道数: 14棋子 * 历史步数 + 当前玩家 + 回合数
def _compute_input_channels(history_steps: int) -> int:
    return NUM_PIECE_TYPES * history_steps + 2


# ============================================================
# 动作空间构建
# ============================================================

def _build_action_index() -> Tuple[Dict[str, int], Dict[int, str]]:
    """构建精简的动作空间 (~2086个)
    
    只枚举中国象棋中实际可能的走法模式：
    1. 直线走法 (车/炮/将/兵): 横向 + 纵向
    2. 马的日字走法: 8个方向
    3. 象的田字走法: 4个对角方向
    4. 士的斜线走法: 九宫内4个方向
    """
    valid_moves = set()
    
    def sq_to_rc(sq: int) -> Tuple[int, int]:
        return sq // 9, sq % 9
    
    def rc_to_sq(r: int, c: int) -> int:
        return r * 9 + c
    
    def in_board(r: int, c: int) -> bool:
        return 0 <= r < 10 and 0 <= c < 9
    
    # 1. 直线走法 (车/炮/将/兵/帅) - 横向和纵向
    for from_sq in range(90):
        r, c = sq_to_rc(from_sq)
        # 横向
        for nc in range(9):
            if nc != c:
                valid_moves.add((from_sq, rc_to_sq(r, nc)))
        # 纵向
        for nr in range(10):
            if nr != r:
                valid_moves.add((from_sq, rc_to_sq(nr, c)))
    
    # 2. 马的日字走法 (8个方向)
    knight_moves = [
        (-2, -1), (-2, 1), (-1, -2), (-1, 2),
        (1, -2), (1, 2), (2, -1), (2, 1)
    ]
    for from_sq in range(90):
        r, c = sq_to_rc(from_sq)
        for dr, dc in knight_moves:
            nr, nc = r + dr, c + dc
            if in_board(nr, nc):
                valid_moves.add((from_sq, rc_to_sq(nr, nc)))
    
    # 3. 象的田字走法 (4个对角方向，步长2)
    bishop_moves = [(-2, -2), (-2, 2), (2, -2), (2, 2)]
    for from_sq in range(90):
        r, c = sq_to_rc(from_sq)
        for dr, dc in bishop_moves:
            nr, nc = r + dr, c + dc
            if in_board(nr, nc):
                valid_moves.add((from_sq, rc_to_sq(nr, nc)))
    
    # 4. 士的斜线走法 (4个对角方向，步长1)
    advisor_moves = [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    for from_sq in range(90):
        r, c = sq_to_rc(from_sq)
        for dr, dc in advisor_moves:
            nr, nc = r + dr, c + dc
            if in_board(nr, nc):
                valid_moves.add((from_sq, rc_to_sq(nr, nc)))
    
    # 构建索引映射
    uci_to_idx, idx_to_uci = {}, {}
    for idx, (from_sq, to_sq) in enumerate(sorted(valid_moves)):
        from_name = cchess.SQUARE_NAMES[from_sq]
        to_name = cchess.SQUARE_NAMES[to_sq]
        uci = from_name + to_name
        uci_to_idx[uci] = idx
        idx_to_uci[idx] = uci
    
    return uci_to_idx, idx_to_uci


# 全局动作索引（模块加载时初始化）
UCI_TO_IDX, IDX_TO_UCI = _build_action_index()
TOTAL_ACTIONS = len(UCI_TO_IDX)


# ============================================================
# 辅助函数
# ============================================================

def _flip_square(sq: int) -> int:
    """翻转棋盘坐标（180度旋转）"""
    return 89 - sq


def _flip_action(action_idx: int) -> int:
    """翻转动作索引（180度旋转）"""
    uci = IDX_TO_UCI[action_idx]
    from_sq = cchess.SQUARE_NAMES.index(uci[:2])
    to_sq = cchess.SQUARE_NAMES.index(uci[2:])
    new_uci = cchess.SQUARE_NAMES[_flip_square(from_sq)] + cchess.SQUARE_NAMES[_flip_square(to_sq)]
    return UCI_TO_IDX.get(new_uci, action_idx)


def _augment_action(action: int) -> int:
    """翻转动作坐标（左右镜像）"""
    uci = IDX_TO_UCI[action]
    from_col = ord(uci[0]) - ord('a')
    from_row = uci[1]
    to_col = ord(uci[2]) - ord('a')
    to_row = uci[3]
    
    new_from_col = chr(ord('a') + (8 - from_col))
    new_to_col = chr(ord('a') + (8 - to_col))
    new_uci = f"{new_from_col}{from_row}{new_to_col}{to_row}"
    
    return UCI_TO_IDX.get(new_uci, action)


# 预计算动作增强映射表
AUGMENT_ACTION_MAP = np.array([_augment_action(i) for i in range(TOTAL_ACTIONS)], dtype=np.int32)


# ============================================================
# ChineseChessGame 类
# ============================================================

@register_game("chinese_chess")
class ChineseChessGame(Game):
    """中国象棋游戏实现
    
    Attributes:
        config: 游戏配置
        board: cchess.Board 实例
        history: 棋盘历史列表（用于编码观测）
        history_steps: 历史步数
        move_count: 已走步数
    
    Example:
        >>> game = ChineseChessGame()
        >>> obs = game.reset()
        >>> print(obs.shape)  # (58, 10, 9)
        >>> actions = game.legal_actions()
        >>> obs, reward, done, info = game.step(actions[0])
        
        >>> # 使用配置创建
        >>> from games.chinese_chess.config import ChineseChessConfig
        >>> config = ChineseChessConfig(max_game_length=300)
        >>> game = ChineseChessGame(config=config)
    """
    
    # 关联配置类（用于 Game.from_config）
    from .config import ChineseChessConfig
    config_class = ChineseChessConfig
    
    # === 元数据方法 ===
    
    @classmethod
    def get_meta(cls) -> GameMeta:
        """返回游戏元数据"""
        return GameMeta(
            name="中国象棋",
            description="中国象棋，双方各16子，红方先行，将死对方为胜",
            version="1.0.0",
            author="ZeroForge",
            tags=["board", "strategy", "2-player", "chinese"],
            difficulty="hard",
            min_players=2,
            max_players=2,
        )
    
    @property
    def supported_render_modes(self) -> List[str]:
        """支持的渲染模式"""
        return ["text", "human", "ascii"]
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """返回默认配置"""
        return {
            "history_steps": DEFAULT_HISTORY_STEPS,
            "max_game_length": MAX_GAME_LENGTH,
        }
    
    def __init__(
        self, 
        history_steps: int = DEFAULT_HISTORY_STEPS,
        config: Optional["ChineseChessConfig"] = None,
    ):
        """初始化游戏
        
        Args:
            history_steps: 历史步数，用于状态编码（如果提供 config 则忽略）
            config: 游戏配置（推荐使用）
        """
        # 优先使用 config
        if config is not None:
            self.config = config
            self._history_steps = config.history_steps
        else:
            from .config import ChineseChessConfig
            self.config = ChineseChessConfig(history_steps=history_steps)
            self._history_steps = history_steps
        
        self._input_channels = _compute_input_channels(self._history_steps)
        
        # 定义空间
        self._observation_space = ObservationSpace(
            shape=(self._input_channels, BOARD_HEIGHT, BOARD_WIDTH),
            dtype=np.float32
        )
        self._action_space = ActionSpace(n=TOTAL_ACTIONS)
        
        # 游戏状态
        self.board: cchess.Board = cchess.Board()
        self.history: List[cchess.Board] = [self.board.copy()]
        self.move_count: int = 0
    
    # === 空间属性 ===
    
    @property
    def observation_space(self) -> ObservationSpace:
        return self._observation_space
    
    @property
    def action_space(self) -> ActionSpace:
        return self._action_space
    
    @property
    def num_players(self) -> int:
        return 2
    
    # === 核心方法 ===
    
    def reset(self) -> np.ndarray:
        """重置游戏到初始状态"""
        self.board = cchess.Board()
        self.history = [self.board.copy()]
        self.move_count = 0
        return self.get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """执行动作
        
        Args:
            action: 动作索引
        
        Returns:
            (observation, reward, done, info)
        
        Raises:
            ValueError: 如果动作非法
        """
        # 转换动作
        move = self._action_to_move(action)
        
        if move not in self.board.legal_moves:
            legal_actions = self.legal_actions()
            raise ValueError(
                f"非法动作! action={action}, move={move}, "
                f"fen={self.board.fen()}, legal_actions={legal_actions[:10]}..."
            )
        
        # 记录当前玩家
        current_player = self.current_player()
        
        # 执行动作
        self.board.push(move)
        self.move_count += 1
        
        # 更新历史
        self.history.append(self.board.copy())
        if len(self.history) > self._history_steps:
            self.history = self.history[-self._history_steps:]
        
        # 检查游戏结束
        done = self.is_terminal()
        reward = 0.0
        info: Dict[str, Any] = {}
        
        if done:
            winner = self.get_winner()
            info["winner"] = winner
            info["termination"] = self._get_termination_reason()
            
            if winner is not None:
                # 从执行动作的玩家视角计算奖励
                reward = 1.0 if winner == current_player else -1.0
            else:
                reward = 0.0  # 和棋
            
            info["all_rewards"] = self.get_rewards()
        
        # 检查超步
        if not done and self.move_count >= MAX_GAME_LENGTH:
            done = True
            info["termination"] = "timeout"
            info["winner"] = None
        
        return self.get_observation(), reward, done, info
    
    def legal_actions(self) -> List[int]:
        """获取合法动作列表"""
        actions = []
        for move in self.board.legal_moves:
            uci = cchess.SQUARE_NAMES[move.from_square] + cchess.SQUARE_NAMES[move.to_square]
            if uci in UCI_TO_IDX:
                actions.append(UCI_TO_IDX[uci])
        return actions
    
    def current_player(self) -> int:
        """获取当前玩家 (0=红方, 1=黑方)"""
        return 0 if self.board.turn == cchess.RED else 1
    
    def clone(self) -> "ChineseChessGame":
        """深拷贝游戏状态"""
        cloned = ChineseChessGame(history_steps=self._history_steps)
        cloned.board = self.board.copy()
        cloned.history = [b.copy() for b in self.history]
        cloned.move_count = self.move_count
        return cloned
    
    # === 观测编码 ===
    
    def get_observation(self) -> np.ndarray:
        """获取当前观测（编码棋盘历史）
        
        通道布局：
        - 红方棋子(0-6) + 黑方棋子(7-13) × 历史步数
        - 当前玩家通道（红=1.0，黑=0.0）
        - 回合数通道（归一化到 [0, 1]）
        """
        # 确保有足够的历史
        boards = self.history.copy()
        while len(boards) < self._history_steps:
            boards = [boards[0]] + boards
        boards = boards[-self._history_steps:]
        
        result = np.zeros(
            (self._input_channels, BOARD_HEIGHT, BOARD_WIDTH),
            dtype=np.float32
        )
        
        # 棋子类型映射
        piece_map_red = {
            cchess.KING: 0, cchess.ADVISOR: 1, cchess.BISHOP: 2,
            cchess.KNIGHT: 3, cchess.ROOK: 4, cchess.CANNON: 5, cchess.PAWN: 6
        }
        piece_map_black = {
            cchess.KING: 7, cchess.ADVISOR: 8, cchess.BISHOP: 9,
            cchess.KNIGHT: 10, cchess.ROOK: 11, cchess.CANNON: 12, cchess.PAWN: 13
        }
        
        # 编码每一步历史
        for t, board in enumerate(boards):
            for sq in cchess.SQUARES:
                piece = board.piece_at(sq)
                if piece is None:
                    continue
                
                if piece.color == cchess.RED:
                    channel = t * NUM_PIECE_TYPES + piece_map_red[piece.piece_type]
                else:
                    channel = t * NUM_PIECE_TYPES + piece_map_black[piece.piece_type]
                
                row, col = sq // 9, sq % 9
                result[channel, row, col] = 1.0
        
        # 当前玩家通道
        result[-2, :, :] = 1.0 if self.board.turn == cchess.RED else 0.0
        
        # 回合数通道
        result[-1, :, :] = min(self.move_count / MAX_GAME_LENGTH, 1.0)
        
        return result
    
    def get_legal_actions_mask(self) -> np.ndarray:
        """获取合法动作掩码"""
        mask = np.zeros(self.action_space.n, dtype=np.float32)
        for action in self.legal_actions():
            mask[action] = 1.0
        return mask
    
    # === 状态查询 ===
    
    def is_terminal(self) -> bool:
        """检查游戏是否结束"""
        if self.move_count >= MAX_GAME_LENGTH:
            return True
        return self.board.is_game_over()
    
    def get_winner(self) -> Optional[int]:
        """获取获胜玩家 (0=红方, 1=黑方, None=和棋)"""
        outcome = self.board.outcome()
        if outcome is None:
            return None
        if outcome.winner is None:
            return None
        return 0 if outcome.winner == cchess.RED else 1
    
    def get_rewards(self) -> Dict[int, float]:
        """获取所有玩家的最终奖励"""
        winner = self.get_winner()
        if winner is None:
            return {0: 0.0, 1: 0.0}
        elif winner == 0:
            return {0: 1.0, 1: -1.0}
        else:
            return {0: -1.0, 1: 1.0}
    
    def _get_termination_reason(self) -> str:
        """获取游戏终止原因"""
        outcome = self.board.outcome()
        if outcome is None:
            return "unknown"
        return outcome.termination.name.lower()
    
    # === 动作转换 ===
    
    def _action_to_move(self, action: int) -> cchess.Move:
        """动作索引转 cchess.Move"""
        uci = IDX_TO_UCI[action]
        from_sq = cchess.SQUARE_NAMES.index(uci[:2])
        to_sq = cchess.SQUARE_NAMES.index(uci[2:])
        return cchess.Move(from_sq, to_sq)
    
    def _move_to_action(self, move: cchess.Move) -> Optional[int]:
        """cchess.Move 转动作索引"""
        uci = cchess.SQUARE_NAMES[move.from_square] + cchess.SQUARE_NAMES[move.to_square]
        return UCI_TO_IDX.get(uci)
    
    # === 可视化 ===
    
    def render(self, mode: str = "text") -> Any:
        """渲染棋盘
        
        Args:
            mode: 渲染模式
                - "human": 打印到控制台
                - "text" / "ascii": 返回文本字符串
        
        Raises:
            ValueError: 如果 mode 不在 supported_render_modes 中
        """
        if mode not in self.supported_render_modes:
            raise ValueError(f"不支持的渲染模式: {mode}，支持: {self.supported_render_modes}")
        
        # 自定义渲染，使用全角字符确保对齐
        board_str = self._render_board_aligned()
        
        if mode == "human":
            print(board_str)
            return None
        elif mode in ("text", "ascii"):
            return {"type": "text", "text": board_str}
    
    def _render_board_aligned(self) -> str:
        """渲染简洁棋盘（全角字符对齐）"""
        # 棋子符号（全角）
        piece_symbols = {
            'R': '車', 'N': '馬', 'B': '相', 'A': '仕', 'K': '帥', 'C': '炮', 'P': '兵',
            'r': '车', 'n': '马', 'b': '象', 'a': '士', 'k': '将', 'c': '砲', 'p': '卒',
        }
        
        lines = []
        # 全角数字
        lines.append("　　９８７６５４３２１０")
        
        for row in range(9, -1, -1):
            # 行号用全角
            row_label = "９８７６５４３２１０"[9 - row]
            row_str = f"{row_label}　"
            for col in range(8, -1, -1):
                square = row * 9 + col
                piece = self.board.piece_at(square)
                if piece:
                    row_str += piece_symbols.get(piece.symbol(), '？')
                else:
                    row_str += "．"  # 全角圆点
            lines.append(row_str)
            
            if row == 5:
                lines.append("　　楚河　汉界　　")
        
        lines.append("　　９８７６５４３２１０")
        turn = "红" if self.board.turn == cchess.RED else "黑"
        lines.append(f"　　当前：{turn}方走棋")
        
        return "\n".join(lines)
    
    def get_state_hash(self) -> int:
        """获取状态哈希（用于检测重复局面）"""
        return hash((
            self.board.pawns, self.board.rooks, self.board.knights,
            self.board.bishops, self.board.advisors, self.board.kings,
            self.board.cannons, self.board.occupied_co[cchess.RED],
            self.board.occupied_co[cchess.BLACK], self.board.turn
        ))
    
    # === FEN 支持 ===
    
    def set_fen(self, fen: str) -> None:
        """从 FEN 字符串设置棋盘状态"""
        self.board = cchess.Board(fen)
        self.history = [self.board.copy()]
        self.move_count = 0
    
    def get_fen(self) -> str:
        """获取当前 FEN 字符串"""
        return self.board.fen()
    
    def __repr__(self) -> str:
        return (
            f"ChineseChessGame(fen='{self.board.fen()}', "
            f"move_count={self.move_count}, "
            f"current_player={'红' if self.current_player() == 0 else '黑'})"
        )


# ============================================================
# 导出辅助函数（供其他模块使用）
# ============================================================

def get_action_space_size() -> int:
    """获取动作空间大小"""
    return TOTAL_ACTIONS


def action_to_uci(action: int) -> str:
    """动作索引转 UCI 字符串"""
    return IDX_TO_UCI.get(action, "")


def uci_to_action(uci: str) -> Optional[int]:
    """UCI 字符串转动作索引"""
    return UCI_TO_IDX.get(uci)


def augment_action_batch(actions: np.ndarray) -> np.ndarray:
    """批量增强动作（左右镜像）"""
    return AUGMENT_ACTION_MAP[actions]


def augment_observation(obs: np.ndarray) -> np.ndarray:
    """增强观测（左右镜像）"""
    return obs[:, :, ::-1].copy()


def augment_policy(policy: np.ndarray) -> np.ndarray:
    """增强策略（左右镜像）"""
    result = np.zeros_like(policy)
    result[AUGMENT_ACTION_MAP] = policy
    return result


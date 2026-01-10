"""
中国象棋 Pygame 图形界面
直接使用 xiangqi 模块的规则实现，确保与训练一致
"""

import pygame
import sys
from typing import Optional, Tuple, List, Callable
from dataclasses import dataclass
import numpy as np

# ============================================================================
# 导入 xiangqi 规则模块 (确保使用同一套规则)
# ============================================================================

try:
    import jax
    import jax.numpy as jnp
    from xiangqi.env import XiangqiEnv, XiangqiState, POSITION_HISTORY_SIZE
    from xiangqi.rules import (
        get_legal_moves_mask, is_in_check, is_game_over, apply_move,
        get_initial_board, find_king,
        BOARD_HEIGHT, BOARD_WIDTH,
    )
    from xiangqi.actions import (
        ACTION_SPACE_SIZE, action_to_move, move_to_action,
        _ACTION_TO_FROM_SQ, _ACTION_TO_TO_SQ,
    )
    JAX_AVAILABLE = True
    print("[GUI] 使用 xiangqi 模块规则 (JAX)")
except ImportError as e:
    JAX_AVAILABLE = False
    print(f"[GUI] 警告: 无法导入 xiangqi 模块: {e}")
    print("[GUI] 请确保已安装 JAX: pip install jax")
    BOARD_HEIGHT, BOARD_WIDTH = 10, 9

# ============================================================================
# 配置常量
# ============================================================================

CELL_SIZE = 60
BOARD_MARGIN = 40

# 窗口尺寸
WINDOW_WIDTH = CELL_SIZE * (BOARD_WIDTH - 1) + BOARD_MARGIN * 2 + 250
WINDOW_HEIGHT = CELL_SIZE * (BOARD_HEIGHT - 1) + BOARD_MARGIN * 2

# 颜色定义
COLOR_BG = (239, 214, 181)
COLOR_LINE = (0, 0, 0)
COLOR_RED = (200, 30, 30)
COLOR_BLACK = (30, 30, 30)
COLOR_SELECTED = (50, 200, 50)
COLOR_LEGAL_MOVE = (100, 200, 100)
COLOR_LAST_MOVE = (255, 200, 50)
COLOR_CHECK = (255, 50, 50)
COLOR_INFO_BG = (245, 245, 245)
COLOR_BUTTON = (70, 130, 180)
COLOR_BUTTON_HOVER = (100, 160, 210)

# 棋子定义 - 中文名称 (红方, 黑方)
PIECE_NAMES_CN = {
    1: ('帅', '将'),  # 使用简体字
    2: ('仕', '士'),
    3: ('相', '象'),
    4: ('马', '马'),
    5: ('车', '车'),
    6: ('炮', '炮'),
    7: ('兵', '卒'),
}

# 英文备选名称 (用于字体不支持中文时)
PIECE_NAMES_EN = {
    1: ('K', 'k'),   # King
    2: ('A', 'a'),   # Advisor
    3: ('B', 'b'),   # Bishop/Elephant
    4: ('N', 'n'),   # Knight
    5: ('R', 'r'),   # Rook
    6: ('C', 'c'),   # Cannon
    7: ('P', 'p'),   # Pawn
}

# 默认使用中文
PIECE_NAMES = PIECE_NAMES_CN

# FEN 字符映射
FEN_PIECE_MAP = {
    'K': 1, 'A': 2, 'B': 3, 'N': 4, 'R': 5, 'C': 6, 'P': 7,
    'k': -1, 'a': -2, 'b': -3, 'n': -4, 'r': -5, 'c': -6, 'p': -7,
}

PIECE_TO_FEN = {v: k for k, v in FEN_PIECE_MAP.items()}

STARTING_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"


# ============================================================================
# FEN 解析
# ============================================================================

def parse_fen(fen: str) -> Tuple[np.ndarray, int]:
    """解析 FEN 字符串"""
    parts = fen.strip().split()
    board_fen = parts[0]
    player_fen = parts[1] if len(parts) > 1 else 'w'
    
    board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=np.int8)
    
    rows = board_fen.split('/')
    if len(rows) != 10:
        raise ValueError(f"FEN 应有 10 行，实际 {len(rows)} 行")
    
    for row_idx, row in enumerate(rows):
        col_idx = 0
        board_row = 9 - row_idx
        
        for char in row:
            if char.isdigit():
                col_idx += int(char)
            elif char in FEN_PIECE_MAP:
                if col_idx < BOARD_WIDTH:
                    board[board_row, col_idx] = FEN_PIECE_MAP[char]
                    col_idx += 1
            else:
                raise ValueError(f"无效 FEN 字符: {char}")
    
    current_player = 0 if player_fen.lower() in ('w', 'r') else 1
    return board, current_player


def board_to_fen(board: np.ndarray, current_player: int) -> str:
    """将棋盘转换为 FEN 字符串"""
    rows = []
    for row_idx in range(9, -1, -1):
        row_str = ""
        empty_count = 0
        for col_idx in range(BOARD_WIDTH):
            piece = board[row_idx, col_idx]
            if piece == 0:
                empty_count += 1
            else:
                if empty_count > 0:
                    row_str += str(empty_count)
                    empty_count = 0
                row_str += PIECE_TO_FEN.get(int(piece), '?')
        if empty_count > 0:
            row_str += str(empty_count)
        rows.append(row_str)
    
    player_char = 'w' if current_player == 0 else 'b'
    return '/'.join(rows) + ' ' + player_char


# ============================================================================
# 使用 xiangqi 模块的规则包装器
# ============================================================================

class XiangqiRules:
    """
    规则包装器 - 直接调用 xiangqi 模块的 JAX 实现
    确保 GUI 测试与训练使用完全相同的规则
    """
    
    def __init__(self):
        if not JAX_AVAILABLE:
            raise RuntimeError("JAX 未安装，无法使用 xiangqi 规则模块")
        
        self.env = XiangqiEnv()
        self._rng_key = jax.random.PRNGKey(42)
    
    def create_state_from_board(self, board: np.ndarray, player: int) -> XiangqiState:
        """从棋盘数组创建完整的 XiangqiState"""
        # 初始化一个空状态
        self._rng_key, init_key = jax.random.split(self._rng_key)
        state = self.env.init(init_key)
        
        # 替换棋盘和玩家
        jax_board = jnp.array(board, dtype=jnp.int8)
        
        # 计算合法动作
        legal_mask = get_legal_moves_mask(jax_board, jnp.int32(player))
        
        # 检查将军
        check = is_in_check(jax_board, jnp.int32(player))
        
        # 检查游戏结束
        game_over, winner = is_game_over(jax_board, jnp.int32(player))
        
        # 创建新状态
        from xiangqi.env import compute_position_hash
        pos_hash = compute_position_hash(jax_board, jnp.int32(player))
        position_hashes = jnp.zeros(POSITION_HISTORY_SIZE, dtype=jnp.int32)
        position_hashes = position_hashes.at[0].set(pos_hash)
        
        return XiangqiState(
            board=jax_board,
            history=jnp.zeros((16, BOARD_HEIGHT, BOARD_WIDTH), dtype=jnp.int8),
            current_player=jnp.int32(player),
            legal_action_mask=legal_mask,
            rewards=jnp.zeros(2, dtype=jnp.float32),
            terminated=game_over,
            step_count=jnp.int32(0),
            no_capture_count=jnp.int32(0),
            winner=winner,
            position_hashes=position_hashes,
            hash_count=jnp.int32(1),
            red_consecutive_checks=jnp.int32(0),
            black_consecutive_checks=jnp.int32(0),
        )
    
    def get_legal_moves(self, board: np.ndarray, row: int, col: int, player: int) -> List[Tuple[int, int]]:
        """获取指定棋子的合法走法"""
        jax_board = jnp.array(board, dtype=jnp.int8)
        legal_mask = get_legal_moves_mask(jax_board, jnp.int32(player))
        
        from_sq = row * BOARD_WIDTH + col
        moves = []
        
        # 遍历所有动作，找出从该位置出发的合法走法
        for action in range(ACTION_SPACE_SIZE):
            if legal_mask[action]:
                action_from = int(_ACTION_TO_FROM_SQ[action])
                action_to = int(_ACTION_TO_TO_SQ[action])
                if action_from == from_sq:
                    to_row = action_to // BOARD_WIDTH
                    to_col = action_to % BOARD_WIDTH
                    moves.append((to_row, to_col))
        
        return moves
    
    def is_legal_move(self, board: np.ndarray, from_row: int, from_col: int, 
                      to_row: int, to_col: int, player: int) -> bool:
        """检查走法是否合法"""
        jax_board = jnp.array(board, dtype=jnp.int8)
        legal_mask = get_legal_moves_mask(jax_board, jnp.int32(player))
        
        from_sq = from_row * BOARD_WIDTH + from_col
        to_sq = to_row * BOARD_WIDTH + to_col
        
        try:
            action = move_to_action(from_sq, to_sq)
            return bool(legal_mask[action])
        except:
            return False
    
    def make_move(self, state: XiangqiState, from_row: int, from_col: int,
                  to_row: int, to_col: int) -> XiangqiState:
        """执行走棋，返回新状态"""
        from_sq = from_row * BOARD_WIDTH + from_col
        to_sq = to_row * BOARD_WIDTH + to_col
        action = move_to_action(from_sq, to_sq)
        
        return self.env.step(state, jnp.int32(action))
    
    def is_in_check(self, board: np.ndarray, player: int) -> bool:
        """检查是否被将军"""
        jax_board = jnp.array(board, dtype=jnp.int8)
        return bool(is_in_check(jax_board, jnp.int32(player)))
    
    def check_game_over(self, board: np.ndarray, player: int) -> Tuple[bool, int]:
        """检查游戏是否结束"""
        jax_board = jnp.array(board, dtype=jnp.int8)
        game_over, winner = is_game_over(jax_board, jnp.int32(player))
        return bool(game_over), int(winner)
    
    def find_king(self, board: np.ndarray, player: int) -> Optional[Tuple[int, int]]:
        """找到将/帅位置"""
        jax_board = jnp.array(board, dtype=jnp.int8)
        king_sq = find_king(jax_board, jnp.int32(player))
        if king_sq < 0:
            return None
        row = int(king_sq) // BOARD_WIDTH
        col = int(king_sq) % BOARD_WIDTH
        return (row, col)


# ============================================================================
# GUI 类
# ============================================================================

class XiangqiGUI:
    """中国象棋图形界面 - 使用 xiangqi 模块规则"""
    
    # 类变量：是否支持中文
    _chinese_font_supported = None
    _font_name_used = None
    
    @classmethod
    def _load_chinese_font(cls, size: int) -> pygame.font.Font:
        """
        加载支持中文的字体 (跨平台)
        
        尝试顺序:
        1. Windows: SimHei, Microsoft YaHei
        2. macOS: PingFang SC, Heiti SC, STHeiti
        3. Linux: Noto Sans CJK, WenQuanYi
        4. 回退: 系统默认字体
        """
        import platform
        
        # 如果已知字体名称，直接使用
        if cls._font_name_used:
            return pygame.font.SysFont(cls._font_name_used, size)
        
        # 按优先级排列的中文字体列表
        font_candidates = []
        
        system = platform.system()
        if system == 'Windows':
            font_candidates = [
                'simhei', 'Microsoft YaHei', 'SimSun', 'FangSong', 'KaiTi'
            ]
        elif system == 'Darwin':  # macOS
            font_candidates = [
                'PingFang SC', 'Heiti SC', 'STHeiti', 'Songti SC', 
                'Hiragino Sans GB', 'Apple LiGothic'
            ]
        else:  # Linux
            font_candidates = [
                'Noto Sans CJK SC', 'Noto Sans CJK', 'WenQuanYi Micro Hei',
                'WenQuanYi Zen Hei', 'Droid Sans Fallback', 'AR PL UMing CN',
                'Noto Sans SC', 'Source Han Sans CN'
            ]
        
        # 尝试加载每个候选字体
        for font_name in font_candidates:
            try:
                font = pygame.font.SysFont(font_name, size)
                # 测试是否能渲染中文棋子字符
                test_surface = font.render('车马炮', True, (0, 0, 0))
                if test_surface.get_width() > 20:  # 成功渲染多个字符
                    print(f"[GUI] 使用中文字体: {font_name}")
                    cls._font_name_used = font_name
                    cls._chinese_font_supported = True
                    return font
            except:
                continue
        
        # 回退到默认字体
        print("[GUI] 警告: 未找到中文字体，使用英文棋子名称")
        cls._chinese_font_supported = False
        cls._font_name_used = None
        return pygame.font.Font(None, size + 10)  # 默认字体稍大一点
    
    def __init__(self, ai_callback: Optional[Callable] = None):
        if not JAX_AVAILABLE:
            raise RuntimeError(
                "JAX 未安装，无法运行 GUI。\n"
                "请安装: pip install jax jaxlib"
            )
        
        pygame.init()
        pygame.font.init()
        
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("中国象棋 - ZeroForge (使用 xiangqi 规则)")
        
        self.clock = pygame.time.Clock()
        self.running = True
        
        # 字体 - 跨平台中文字体支持
        self.font_large = self._load_chinese_font(36)
        self.font_medium = self._load_chinese_font(24)
        self.font_small = self._load_chinese_font(18)
        
        # 规则引擎 (使用 xiangqi 模块)
        self.rules = XiangqiRules()
        
        # 游戏状态
        self.state: Optional[XiangqiState] = None
        self.state_history: List[XiangqiState] = []
        
        # 交互状态
        self.selected_pos = None
        self.legal_moves = []
        self.last_move = None
        
        # AI 设置
        self.ai_callback = ai_callback
        self.ai_player = 1
        self.ai_thinking = False
        
        # 按钮
        self.buttons = {}
        self._create_buttons()
        
        # FEN 输入
        self.fen_input = ""
        self.fen_input_active = False
        
        # 消息
        self.message = ""
        self.message_time = 0
        
        # 缓存（避免每帧调用 JAX）
        self._is_check_cache = False
        self._king_pos_cache = None
        
        # 初始化
        self.reset_game()
    
    def _create_buttons(self):
        """创建按钮"""
        btn_x = CELL_SIZE * (BOARD_WIDTH - 1) + BOARD_MARGIN * 2 + 20
        btn_width = 100
        btn_height = 35
        
        buttons = [
            ("new_game", "新游戏", 50),
            ("undo", "悔棋", 100),
            ("switch_side", "换边", 150),
            ("import_fen", "导入FEN", 200),
            ("export_fen", "导出FEN", 250),
            ("ai_move", "AI走棋", 300),
        ]
        
        for name, label, y in buttons:
            self.buttons[name] = {
                'rect': pygame.Rect(btn_x, y, btn_width, btn_height),
                'label': label,
                'hover': False
            }
    
    def reset_game(self, fen: str = STARTING_FEN):
        """重置游戏"""
        try:
            board, player = parse_fen(fen)
            self.state = self.rules.create_state_from_board(board, player)
            self.state_history = [self.state]
            self.show_message("游戏开始 (使用 xiangqi 规则)")
        except Exception as e:
            self.show_message(f"FEN 解析错误: {e}")
            board, player = parse_fen(STARTING_FEN)
            self.state = self.rules.create_state_from_board(board, player)
            self.state_history = [self.state]
        
        self.selected_pos = None
        self.legal_moves = []
        self.last_move = None
        self.fen_input = ""
        self.fen_input_active = False
        
        # 更新缓存
        self._update_check_cache()
    
    def show_message(self, msg: str, duration: int = 3000):
        """显示消息"""
        self.message = msg
        self.message_time = pygame.time.get_ticks() + duration
    
    @property
    def board(self) -> np.ndarray:
        """获取当前棋盘"""
        return np.array(self.state.board)
    
    @property
    def current_player(self) -> int:
        """获取当前玩家"""
        return int(self.state.current_player)
    
    @property
    def is_game_over(self) -> bool:
        """是否游戏结束"""
        return bool(self.state.terminated)
    
    @property
    def winner(self) -> int:
        """获取胜者"""
        return int(self.state.winner)
    
    @property
    def is_check(self) -> bool:
        """是否被将军（使用缓存）"""
        return self._is_check_cache
    
    def _update_check_cache(self):
        """更新将军状态缓存"""
        self._is_check_cache = self.rules.is_in_check(self.board, self.current_player)
        if self._is_check_cache:
            self._king_pos_cache = self.rules.find_king(self.board, self.current_player)
        else:
            self._king_pos_cache = None
    
    @property
    def step_count(self) -> int:
        """步数"""
        return int(self.state.step_count)
    
    def board_to_screen(self, row: int, col: int) -> Tuple[int, int]:
        """棋盘坐标转屏幕坐标"""
        x = BOARD_MARGIN + col * CELL_SIZE
        y = BOARD_MARGIN + (BOARD_HEIGHT - 1 - row) * CELL_SIZE
        return x, y
    
    def screen_to_board(self, x: int, y: int) -> Optional[Tuple[int, int]]:
        """屏幕坐标转棋盘坐标"""
        col = round((x - BOARD_MARGIN) / CELL_SIZE)
        row = BOARD_HEIGHT - 1 - round((y - BOARD_MARGIN) / CELL_SIZE)
        
        if 0 <= row < BOARD_HEIGHT and 0 <= col < BOARD_WIDTH:
            return row, col
        return None
    
    def is_own_piece(self, row: int, col: int) -> bool:
        """检查是否为己方棋子"""
        piece = self.board[row, col]
        if self.current_player == 0:
            return piece > 0
        else:
            return piece < 0
    
    def draw_board(self):
        """绘制棋盘"""
        self.screen.fill(COLOR_BG)
        
        # 网格线
        for i in range(BOARD_WIDTH):
            x = BOARD_MARGIN + i * CELL_SIZE
            pygame.draw.line(self.screen, COLOR_LINE,
                           (x, BOARD_MARGIN),
                           (x, BOARD_MARGIN + 4 * CELL_SIZE), 2)
            pygame.draw.line(self.screen, COLOR_LINE,
                           (x, BOARD_MARGIN + 5 * CELL_SIZE),
                           (x, BOARD_MARGIN + 9 * CELL_SIZE), 2)
        
        for i in range(BOARD_HEIGHT):
            y = BOARD_MARGIN + i * CELL_SIZE
            width = 2 if i in [0, 4, 5, 9] else 1
            pygame.draw.line(self.screen, COLOR_LINE,
                           (BOARD_MARGIN, y),
                           (BOARD_MARGIN + 8 * CELL_SIZE, y), width)
        
        # 九宫格斜线
        # 红方
        pygame.draw.line(self.screen, COLOR_LINE,
                        (BOARD_MARGIN + 3 * CELL_SIZE, BOARD_MARGIN + 7 * CELL_SIZE),
                        (BOARD_MARGIN + 5 * CELL_SIZE, BOARD_MARGIN + 9 * CELL_SIZE), 1)
        pygame.draw.line(self.screen, COLOR_LINE,
                        (BOARD_MARGIN + 5 * CELL_SIZE, BOARD_MARGIN + 7 * CELL_SIZE),
                        (BOARD_MARGIN + 3 * CELL_SIZE, BOARD_MARGIN + 9 * CELL_SIZE), 1)
        # 黑方
        pygame.draw.line(self.screen, COLOR_LINE,
                        (BOARD_MARGIN + 3 * CELL_SIZE, BOARD_MARGIN),
                        (BOARD_MARGIN + 5 * CELL_SIZE, BOARD_MARGIN + 2 * CELL_SIZE), 1)
        pygame.draw.line(self.screen, COLOR_LINE,
                        (BOARD_MARGIN + 5 * CELL_SIZE, BOARD_MARGIN),
                        (BOARD_MARGIN + 3 * CELL_SIZE, BOARD_MARGIN + 2 * CELL_SIZE), 1)
        
        # 河界文字
        river_y = BOARD_MARGIN + 4.5 * CELL_SIZE
        text = self.font_medium.render("楚 河", True, COLOR_LINE)
        self.screen.blit(text, (BOARD_MARGIN + 0.5 * CELL_SIZE, river_y - 15))
        text = self.font_medium.render("漢 界", True, COLOR_LINE)
        self.screen.blit(text, (BOARD_MARGIN + 5.5 * CELL_SIZE, river_y - 15))
    
    def draw_highlights(self):
        """绘制高亮"""
        # 上一步移动
        if self.last_move:
            fr, fc, tr, tc = self.last_move
            for r, c in [(fr, fc), (tr, tc)]:
                x, y = self.board_to_screen(r, c)
                pygame.draw.circle(self.screen, COLOR_LAST_MOVE, (x, y), CELL_SIZE // 2 + 5, 3)
        
        # 选中的棋子
        if self.selected_pos:
            r, c = self.selected_pos
            x, y = self.board_to_screen(r, c)
            pygame.draw.circle(self.screen, COLOR_SELECTED, (x, y), CELL_SIZE // 2 + 5, 3)
        
        # 合法走法提示
        for r, c in self.legal_moves:
            x, y = self.board_to_screen(r, c)
            if self.board[r, c] == 0:
                pygame.draw.circle(self.screen, COLOR_LEGAL_MOVE, (x, y), 8)
            else:
                pygame.draw.circle(self.screen, COLOR_LEGAL_MOVE, (x, y), CELL_SIZE // 2 + 3, 3)
        
        # 将军警告 (使用缓存的 king 位置)
        if self.is_check and hasattr(self, '_king_pos_cache') and self._king_pos_cache:
            x, y = self.board_to_screen(*self._king_pos_cache)
            pygame.draw.circle(self.screen, COLOR_CHECK, (x, y), CELL_SIZE // 2 + 8, 4)
    
    def draw_pieces(self):
        """绘制棋子"""
        board = self.board
        
        # 根据字体支持选择棋子名称
        piece_names = PIECE_NAMES_CN if self._chinese_font_supported else PIECE_NAMES_EN
        
        for row in range(BOARD_HEIGHT):
            for col in range(BOARD_WIDTH):
                piece = board[row, col]
                if piece != 0:
                    x, y = self.board_to_screen(row, col)
                    
                    is_red = piece > 0
                    piece_color = COLOR_RED if is_red else COLOR_BLACK
                    bg_color = (255, 230, 200) if is_red else (230, 230, 230)
                    
                    pygame.draw.circle(self.screen, bg_color, (x, y), CELL_SIZE // 2 - 2)
                    pygame.draw.circle(self.screen, piece_color, (x, y), CELL_SIZE // 2 - 2, 2)
                    
                    piece_type = abs(piece)
                    if piece_type in piece_names:
                        name = piece_names[piece_type][0 if is_red else 1]
                        text = self.font_large.render(name, True, piece_color)
                        text_rect = text.get_rect(center=(x, y))
                        self.screen.blit(text, text_rect)
    
    def draw_info_panel(self):
        """绘制信息面板"""
        panel_x = CELL_SIZE * (BOARD_WIDTH - 1) + BOARD_MARGIN * 2 + 10
        panel_width = WINDOW_WIDTH - panel_x - 10
        
        pygame.draw.rect(self.screen, COLOR_INFO_BG,
                        (panel_x, 10, panel_width, WINDOW_HEIGHT - 20))
        pygame.draw.rect(self.screen, COLOR_LINE,
                        (panel_x, 10, panel_width, WINDOW_HEIGHT - 20), 1)
        
        # 标题
        title = self.font_medium.render("游戏信息", True, COLOR_LINE)
        self.screen.blit(title, (panel_x + 10, 15))
        
        # 当前玩家
        player_text = "红方走棋" if self.current_player == 0 else "黑方走棋"
        color = COLOR_RED if self.current_player == 0 else COLOR_BLACK
        text = self.font_small.render(player_text, True, color)
        self.screen.blit(text, (panel_x + 10, 350))
        
        # 步数
        text = self.font_small.render(f"回合: {self.step_count}", True, COLOR_LINE)
        self.screen.blit(text, (panel_x + 10, 380))
        
        # 无吃子步数
        no_cap = int(self.state.no_capture_count)
        text = self.font_small.render(f"无吃子: {no_cap}/120", True, COLOR_LINE)
        self.screen.blit(text, (panel_x + 10, 405))
        
        # 连续将军
        red_checks = int(self.state.red_consecutive_checks)
        black_checks = int(self.state.black_consecutive_checks)
        if red_checks > 0:
            text = self.font_small.render(f"红方连将: {red_checks}/6", True, COLOR_RED)
            self.screen.blit(text, (panel_x + 10, 430))
        if black_checks > 0:
            text = self.font_small.render(f"黑方连将: {black_checks}/6", True, COLOR_BLACK)
            self.screen.blit(text, (panel_x + 10, 455))
        
        # 将军状态
        if self.is_check:
            text = self.font_small.render("将军!", True, COLOR_CHECK)
            self.screen.blit(text, (panel_x + 80, 350))
        
        # 游戏结束
        if self.is_game_over:
            if self.winner == 0:
                result = "红方胜!"
                color = COLOR_RED
            elif self.winner == 1:
                result = "黑方胜!"
                color = COLOR_BLACK
            else:
                result = "和棋!"
                color = COLOR_LINE
            text = self.font_medium.render(result, True, color)
            self.screen.blit(text, (panel_x + 10, 480))
        
        # 绘制按钮
        mouse_pos = pygame.mouse.get_pos()
        for name, btn in self.buttons.items():
            rect = btn['rect']
            btn['hover'] = rect.collidepoint(mouse_pos)
            
            color = COLOR_BUTTON_HOVER if btn['hover'] else COLOR_BUTTON
            pygame.draw.rect(self.screen, color, rect, border_radius=5)
            pygame.draw.rect(self.screen, COLOR_LINE, rect, 1, border_radius=5)
            
            text = self.font_small.render(btn['label'], True, (255, 255, 255))
            text_rect = text.get_rect(center=rect.center)
            self.screen.blit(text, text_rect)
        
        # FEN 区域
        fen_y = 510
        text = self.font_small.render("FEN:", True, COLOR_LINE)
        self.screen.blit(text, (panel_x + 10, fen_y))
        
        fen_rect = pygame.Rect(panel_x + 10, fen_y + 25, panel_width - 20, 60)
        border_color = COLOR_SELECTED if self.fen_input_active else COLOR_LINE
        pygame.draw.rect(self.screen, (255, 255, 255), fen_rect)
        pygame.draw.rect(self.screen, border_color, fen_rect, 2)
        
        fen_text = self.fen_input if self.fen_input else board_to_fen(self.board, self.current_player)
        lines = [fen_text[i:i+20] for i in range(0, len(fen_text), 20)]
        for i, line in enumerate(lines[:3]):
            text = self.font_small.render(line, True, COLOR_LINE)
            self.screen.blit(text, (panel_x + 15, fen_y + 30 + i * 18))
        
        # 消息
        if self.message and pygame.time.get_ticks() < self.message_time:
            text = self.font_small.render(self.message, True, COLOR_RED)
            self.screen.blit(text, (panel_x + 10, WINDOW_HEIGHT - 50))
    
    def handle_click(self, pos: Tuple[int, int]):
        """处理点击"""
        x, y = pos
        
        # 检查按钮
        for name, btn in self.buttons.items():
            if btn['rect'].collidepoint(pos):
                self.handle_button_click(name)
                return
        
        # FEN 输入
        panel_x = CELL_SIZE * (BOARD_WIDTH - 1) + BOARD_MARGIN * 2 + 10
        fen_rect = pygame.Rect(panel_x + 10, 535, WINDOW_WIDTH - panel_x - 30, 60)
        if fen_rect.collidepoint(pos):
            self.fen_input_active = True
            return
        else:
            self.fen_input_active = False
        
        # 棋盘点击
        if self.is_game_over:
            return
        
        board_pos = self.screen_to_board(x, y)
        if board_pos is None:
            return
        
        row, col = board_pos
        
        if self.selected_pos:
            if (row, col) in self.legal_moves:
                self.make_move(self.selected_pos[0], self.selected_pos[1], row, col)
            elif self.is_own_piece(row, col):
                self.selected_pos = (row, col)
                self.legal_moves = self.rules.get_legal_moves(self.board, row, col, self.current_player)
            else:
                self.selected_pos = None
                self.legal_moves = []
        else:
            if self.is_own_piece(row, col):
                self.selected_pos = (row, col)
                self.legal_moves = self.rules.get_legal_moves(self.board, row, col, self.current_player)
    
    def make_move(self, from_row: int, from_col: int, to_row: int, to_col: int):
        """执行走棋 - 使用 xiangqi 规则"""
        # 保存历史
        self.state_history.append(self.state)

        # 使用 xiangqi 模块执行走棋
        self.state = self.rules.make_move(self.state, from_row, from_col, to_row, to_col)

        # 更新 UI 状态
        self.last_move = (from_row, from_col, to_row, to_col)
        self.selected_pos = None
        self.legal_moves = []
        
        # 更新缓存
        self._update_check_cache()
        
        # 检查结果
        if self.is_game_over:
            if self.winner == 0:
                self.show_message("红方胜!")
            elif self.winner == 1:
                self.show_message("黑方胜!")
            else:
                self.show_message("和棋!")
    
    def handle_button_click(self, name: str):
        """处理按钮"""
        if name == "new_game":
            self.reset_game()
        
        elif name == "undo":
            if len(self.state_history) > 1:
                self.state_history.pop()
                self.state = self.state_history[-1]
                self.last_move = None
                self.selected_pos = None
                self.legal_moves = []
                self._update_check_cache()  # 更新缓存
                self.show_message("悔棋成功")
            else:
                self.show_message("无法悔棋")
        
        elif name == "switch_side":
            self.ai_player = 1 - self.ai_player
            side = "红方" if self.ai_player == 0 else "黑方"
            self.show_message(f"AI 执{side}")
        
        elif name == "import_fen":
            if self.fen_input:
                self.reset_game(self.fen_input)
                self.show_message("FEN 导入成功")
            else:
                self.show_message("请先输入 FEN")
                self.fen_input_active = True
        
        elif name == "export_fen":
            fen = board_to_fen(self.board, self.current_player)
            self.fen_input = fen
            self.show_message("FEN 已生成")
        
        elif name == "ai_move":
            if self.ai_callback and not self.is_game_over:
                self.request_ai_move()
            else:
                self.show_message("AI 不可用")
    
    def request_ai_move(self):
        """请求 AI 走棋"""
        if self.ai_callback is None:
            return
        
        try:
            result = self.ai_callback(self.board.copy(), self.current_player)
            if result:
                (from_row, from_col), (to_row, to_col) = result
                if self.rules.is_legal_move(self.board, from_row, from_col, to_row, to_col, self.current_player):
                    self.make_move(from_row, from_col, to_row, to_col)
                    self.show_message("AI 走棋完成")
                else:
                    self.show_message("AI 走法非法!")
            else:
                self.show_message("AI 无法走棋")
        except Exception as e:
            self.show_message(f"AI 错误: {e}")
    
    def handle_key(self, event):
        """处理键盘"""
        if self.fen_input_active:
            if event.key == pygame.K_RETURN:
                self.handle_button_click("import_fen")
                self.fen_input_active = False
            elif event.key == pygame.K_BACKSPACE:
                self.fen_input = self.fen_input[:-1]
            elif event.key == pygame.K_ESCAPE:
                self.fen_input_active = False
            elif event.key == pygame.K_v and pygame.key.get_mods() & pygame.KMOD_CTRL:
                try:
                    pygame.scrap.init()
                    text = pygame.scrap.get(pygame.SCRAP_TEXT)
                    if text:
                        self.fen_input += text.decode().strip()
                except:
                    pass
            else:
                if event.unicode and event.unicode.isprintable():
                    self.fen_input += event.unicode
    
    def run(self):
        """主循环"""
        while self.running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.running = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:
                        self.handle_click(event.pos)
                elif event.type == pygame.KEYDOWN:
                    self.handle_key(event)
            
            # AI 自动走棋
            if (self.ai_callback and 
                not self.is_game_over and 
                self.current_player == self.ai_player and
                not self.ai_thinking):
                self.ai_thinking = True
                self.request_ai_move()
                self.ai_thinking = False
            
            # 绘制
            self.draw_board()
            self.draw_highlights()
            self.draw_pieces()
            self.draw_info_panel()
            
            pygame.display.flip()
            self.clock.tick(30)  # 棋盘游戏不需要高帧率
        
        pygame.quit()


def run_gui(ai_callback: Optional[Callable] = None, fen: Optional[str] = None):
    """启动 GUI"""
    gui = XiangqiGUI(ai_callback=ai_callback)
    if fen:
        gui.reset_game(fen)
    gui.run()


# ============================================================================
# 入口
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="中国象棋 GUI (使用 xiangqi 规则)")
    parser.add_argument("--fen", type=str, default=None, help="初始 FEN 字符串")
    args = parser.parse_args()
    
    print("=" * 60)
    print("中国象棋 GUI - ZeroForge")
    print("使用 xiangqi 模块规则，确保与训练一致")
    print("=" * 60)
    print("操作说明:")
    print("  - 点击棋子选中，再点击目标位置走棋")
    print("  - 绿色圆点: 可移动位置 (由 xiangqi.rules 计算)")
    print("  - 红色圈: 将军警告")
    print("  - 右侧显示: 连续将军计数、无吃子步数等规则状态")
    print("=" * 60)
    
    run_gui(fen=args.fen)

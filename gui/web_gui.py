"""
Gradio Web GUI for Chinese Chess (中国象棋)
使用 SVG 绘制棋盘，支持点击交互
"""

import gradio as gr
import numpy as np
from typing import Optional, Callable, Tuple, List
from dataclasses import dataclass
import json

# JAX 导入
try:
    import jax
    import jax.numpy as jnp
    from xiangqi.env import XiangqiEnv, XiangqiState
    from xiangqi.rules import (
        get_legal_moves_mask, is_in_check, find_king,
        BOARD_WIDTH, BOARD_HEIGHT
    )
    from xiangqi.actions import move_to_action, action_to_move
    JAX_AVAILABLE = True
except ImportError as e:
    print(f"[Web GUI] JAX 导入失败: {e}")
    JAX_AVAILABLE = False

# ============================================================================
# 常量定义
# ============================================================================

# 棋盘尺寸
CELL_SIZE = 60
BOARD_MARGIN = 40
PIECE_RADIUS = 26

# SVG 尺寸
SVG_WIDTH = BOARD_MARGIN * 2 + CELL_SIZE * (BOARD_WIDTH - 1)
SVG_HEIGHT = BOARD_MARGIN * 2 + CELL_SIZE * (BOARD_HEIGHT - 1)

# 颜色
COLOR_BG = "#DEB887"  # 棋盘背景
COLOR_LINE = "#8B4513"  # 线条
COLOR_RED = "#CC0000"  # 红方
COLOR_BLACK = "#000000"  # 黑方
COLOR_SELECTED = "#FFD700"  # 选中
COLOR_LEGAL = "#00FF00"  # 合法走法
COLOR_LAST_MOVE = "#87CEEB"  # 上一步
COLOR_CHECK = "#FF6347"  # 将军

# 棋子名称
PIECE_NAMES = {
    1: ('帅', '将'),
    2: ('仕', '士'),
    3: ('相', '象'),
    4: ('马', '马'),
    5: ('车', '车'),
    6: ('炮', '炮'),
    7: ('兵', '卒'),
}

# 初始 FEN
STARTING_FEN = "rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w"

# FEN 字符映射
FEN_TO_PIECE = {
    'K': 1, 'A': 2, 'B': 3, 'N': 4, 'R': 5, 'C': 6, 'P': 7,
    'k': -1, 'a': -2, 'b': -3, 'n': -4, 'r': -5, 'c': -6, 'p': -7,
}
PIECE_TO_FEN = {v: k for k, v in FEN_TO_PIECE.items()}


# ============================================================================
# FEN 解析
# ============================================================================

def parse_fen(fen: str) -> Tuple[np.ndarray, int]:
    """解析 FEN 字符串
    
    FEN 从上到下描述棋盘（黑方在上），但我们的坐标系：
    - row 0-4 是红方（屏幕下方）
    - row 5-9 是黑方（屏幕上方）
    所以 FEN 第一行对应 row 9，最后一行对应 row 0
    """
    parts = fen.strip().split()
    board_str = parts[0]
    player = 0 if len(parts) < 2 or parts[1].lower() in ['w', 'r'] else 1
    
    board = np.zeros((BOARD_HEIGHT, BOARD_WIDTH), dtype=np.int8)
    rows = board_str.split('/')
    
    for fen_row_idx, row_str in enumerate(rows):
        # FEN row 0 -> board row 9, FEN row 9 -> board row 0
        board_row = BOARD_HEIGHT - 1 - fen_row_idx
        col = 0
        for char in row_str:
            if char.isdigit():
                col += int(char)
            elif char in FEN_TO_PIECE:
                board[board_row, col] = FEN_TO_PIECE[char]
                col += 1
    
    return board, player


def board_to_fen(board: np.ndarray, player: int) -> str:
    """棋盘转 FEN（从 row 9 到 row 0）"""
    rows = []
    for row in range(BOARD_HEIGHT - 1, -1, -1):  # 从 row 9 到 row 0
        row_str = ""
        empty = 0
        for col in range(BOARD_WIDTH):
            piece = board[row, col]
            if piece == 0:
                empty += 1
            else:
                if empty > 0:
                    row_str += str(empty)
                    empty = 0
                row_str += PIECE_TO_FEN.get(int(piece), '?')
        if empty > 0:
            row_str += str(empty)
        rows.append(row_str)
    
    player_str = 'w' if player == 0 else 'b'
    return '/'.join(rows) + ' ' + player_str


# ============================================================================
# SVG 绘制
# ============================================================================

def render_board_svg(
    board: np.ndarray,
    current_player: int,
    selected: Optional[Tuple[int, int]] = None,
    legal_moves: List[Tuple[int, int]] = None,
    last_move: Optional[Tuple[int, int, int, int]] = None,
    is_check: bool = False,
    king_pos: Optional[Tuple[int, int]] = None,
) -> str:
    """渲染棋盘 SVG"""
    legal_moves = legal_moves or []
    
    svg_parts = []
    
    # SVG 头部
    svg_parts.append(f'''<svg width="{SVG_WIDTH}" height="{SVG_HEIGHT}" 
        xmlns="http://www.w3.org/2000/svg" 
        style="font-family: 'PingFang SC', 'Microsoft YaHei', 'SimHei', sans-serif;">''')
    
    # 背景
    svg_parts.append(f'<rect width="100%" height="100%" fill="{COLOR_BG}"/>')
    
    # 绘制网格线
    svg_parts.append(_draw_grid())
    
    # 绘制九宫格斜线
    svg_parts.append(_draw_palace())
    
    # 绘制河界
    svg_parts.append(_draw_river())
    
    # 高亮上一步
    if last_move:
        fr, fc, tr, tc = last_move
        for r, c in [(fr, fc), (tr, tc)]:
            x, y = _board_to_svg(r, c)
            svg_parts.append(f'<circle cx="{x}" cy="{y}" r="{PIECE_RADIUS + 5}" '
                           f'fill="none" stroke="{COLOR_LAST_MOVE}" stroke-width="3"/>')
    
    # 高亮选中的棋子
    if selected:
        r, c = selected
        x, y = _board_to_svg(r, c)
        svg_parts.append(f'<circle cx="{x}" cy="{y}" r="{PIECE_RADIUS + 5}" '
                       f'fill="none" stroke="{COLOR_SELECTED}" stroke-width="3"/>')
    
    # 显示合法走法
    for r, c in legal_moves:
        x, y = _board_to_svg(r, c)
        if board[r, c] == 0:
            svg_parts.append(f'<circle cx="{x}" cy="{y}" r="8" fill="{COLOR_LEGAL}" opacity="0.7"/>')
        else:
            svg_parts.append(f'<circle cx="{x}" cy="{y}" r="{PIECE_RADIUS + 3}" '
                           f'fill="none" stroke="{COLOR_LEGAL}" stroke-width="3" opacity="0.7"/>')
    
    # 将军警告
    if is_check and king_pos:
        r, c = king_pos
        x, y = _board_to_svg(r, c)
        svg_parts.append(f'<circle cx="{x}" cy="{y}" r="{PIECE_RADIUS + 8}" '
                       f'fill="none" stroke="{COLOR_CHECK}" stroke-width="4"/>')
    
    # 绘制棋子
    for row in range(BOARD_HEIGHT):
        for col in range(BOARD_WIDTH):
            piece = board[row, col]
            if piece != 0:
                svg_parts.append(_draw_piece(row, col, piece))
    
    # 绘制点击区域（透明，带 data 属性）
    for row in range(BOARD_HEIGHT):
        for col in range(BOARD_WIDTH):
            x, y = _board_to_svg(row, col)
            svg_parts.append(
                f'<circle cx="{x}" cy="{y}" r="{PIECE_RADIUS}" '
                f'fill="transparent" style="cursor:pointer" '
                f'class="click-area" data-row="{row}" data-col="{col}"/>'
            )
    
    svg_parts.append('</svg>')
    
    return '\n'.join(svg_parts)


def _board_to_svg(row: int, col: int) -> Tuple[int, int]:
    """棋盘坐标转 SVG 坐标
    
    棋盘坐标: row 0-4 是红方(下方), row 5-9 是黑方(上方)
    屏幕坐标: y=0 在上方
    所以需要翻转: row 9 -> y 最小, row 0 -> y 最大
    """
    x = BOARD_MARGIN + col * CELL_SIZE
    y = BOARD_MARGIN + (BOARD_HEIGHT - 1 - row) * CELL_SIZE  # 翻转 y 轴
    return x, y


def _draw_grid() -> str:
    """绘制网格"""
    lines = []
    
    # 竖线
    for i in range(BOARD_WIDTH):
        x = BOARD_MARGIN + i * CELL_SIZE
        # 上半部分
        lines.append(f'<line x1="{x}" y1="{BOARD_MARGIN}" '
                    f'x2="{x}" y2="{BOARD_MARGIN + 4 * CELL_SIZE}" '
                    f'stroke="{COLOR_LINE}" stroke-width="1"/>')
        # 下半部分
        lines.append(f'<line x1="{x}" y1="{BOARD_MARGIN + 5 * CELL_SIZE}" '
                    f'x2="{x}" y2="{BOARD_MARGIN + 9 * CELL_SIZE}" '
                    f'stroke="{COLOR_LINE}" stroke-width="1"/>')
    
    # 横线
    for i in range(BOARD_HEIGHT):
        y = BOARD_MARGIN + i * CELL_SIZE
        width = 2 if i in [0, 9] else 1
        lines.append(f'<line x1="{BOARD_MARGIN}" y1="{y}" '
                    f'x2="{BOARD_MARGIN + 8 * CELL_SIZE}" y2="{y}" '
                    f'stroke="{COLOR_LINE}" stroke-width="{width}"/>')
    
    # 边框
    lines.append(f'<rect x="{BOARD_MARGIN - 2}" y="{BOARD_MARGIN - 2}" '
                f'width="{CELL_SIZE * 8 + 4}" height="{CELL_SIZE * 9 + 4}" '
                f'fill="none" stroke="{COLOR_LINE}" stroke-width="3"/>')
    
    return '\n'.join(lines)


def _draw_palace() -> str:
    """绘制九宫格斜线"""
    lines = []
    
    # 上方九宫格（黑方）
    x1 = BOARD_MARGIN + 3 * CELL_SIZE
    x2 = BOARD_MARGIN + 5 * CELL_SIZE
    y1 = BOARD_MARGIN
    y2 = BOARD_MARGIN + 2 * CELL_SIZE
    lines.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{COLOR_LINE}" stroke-width="1"/>')
    lines.append(f'<line x1="{x2}" y1="{y1}" x2="{x1}" y2="{y2}" stroke="{COLOR_LINE}" stroke-width="1"/>')
    
    # 下方九宫格（红方）
    y1 = BOARD_MARGIN + 7 * CELL_SIZE
    y2 = BOARD_MARGIN + 9 * CELL_SIZE
    lines.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{COLOR_LINE}" stroke-width="1"/>')
    lines.append(f'<line x1="{x2}" y1="{y1}" x2="{x1}" y2="{y2}" stroke="{COLOR_LINE}" stroke-width="1"/>')
    
    return '\n'.join(lines)


def _draw_river() -> str:
    """绘制河界文字"""
    y = BOARD_MARGIN + 4.5 * CELL_SIZE
    return f'''
    <text x="{BOARD_MARGIN + 1 * CELL_SIZE}" y="{y + 8}" 
          font-size="20" fill="{COLOR_LINE}" text-anchor="middle">楚</text>
    <text x="{BOARD_MARGIN + 2 * CELL_SIZE}" y="{y + 8}" 
          font-size="20" fill="{COLOR_LINE}" text-anchor="middle">河</text>
    <text x="{BOARD_MARGIN + 6 * CELL_SIZE}" y="{y + 8}" 
          font-size="20" fill="{COLOR_LINE}" text-anchor="middle">汉</text>
    <text x="{BOARD_MARGIN + 7 * CELL_SIZE}" y="{y + 8}" 
          font-size="20" fill="{COLOR_LINE}" text-anchor="middle">界</text>
    '''


def _draw_piece(row: int, col: int, piece: int) -> str:
    """绘制单个棋子"""
    x, y = _board_to_svg(row, col)
    is_red = piece > 0
    piece_type = abs(piece)
    
    color = COLOR_RED if is_red else COLOR_BLACK
    bg_color = "#FFEEDD" if is_red else "#EEEEEE"
    name = PIECE_NAMES.get(piece_type, ('?', '?'))[0 if is_red else 1]
    
    return f'''
    <circle cx="{x}" cy="{y}" r="{PIECE_RADIUS}" fill="{bg_color}" 
            stroke="{color}" stroke-width="2"/>
    <text x="{x}" y="{y + 8}" font-size="28" fill="{color}" 
          text-anchor="middle" font-weight="bold">{name}</text>
    '''


# ============================================================================
# 游戏状态管理
# ============================================================================

@dataclass
class GameState:
    """游戏状态"""
    board: np.ndarray
    current_player: int
    selected: Optional[Tuple[int, int]] = None
    legal_moves: List[Tuple[int, int]] = None
    last_move: Optional[Tuple[int, int, int, int]] = None
    is_check: bool = False
    king_pos: Optional[Tuple[int, int]] = None
    game_over: bool = False
    winner: int = -1
    step_count: int = 0
    history: List = None
    jax_state: Optional[XiangqiState] = None
    
    def __post_init__(self):
        if self.legal_moves is None:
            self.legal_moves = []
        if self.history is None:
            self.history = []


class ChessGame:
    """象棋游戏逻辑"""
    
    def __init__(self):
        self.env = XiangqiEnv() if JAX_AVAILABLE else None
        self._rng_key = jax.random.PRNGKey(42) if JAX_AVAILABLE else None
        self.state: Optional[GameState] = None
        self.ai_callback: Optional[Callable] = None
        self.ai_player: int = 1  # AI 默认执黑
        
    def new_game(self, fen: str = STARTING_FEN) -> GameState:
        """开始新游戏"""
        board, player = parse_fen(fen)
        
        jax_state = None
        if JAX_AVAILABLE:
            jax_state = self._create_jax_state(board, player)
        
        self.state = GameState(
            board=board,
            current_player=player,
            jax_state=jax_state,
        )
        self._update_check_status()
        return self.state
    
    def _create_jax_state(self, board: np.ndarray, player: int) -> XiangqiState:
        """创建 JAX 状态"""
        self._rng_key, init_key = jax.random.split(self._rng_key)
        state = self.env.init(init_key)
        
        # 替换棋盘和玩家
        jax_board = jnp.array(board, dtype=jnp.int8)
        state = state.replace(
            board=jax_board,
            current_player=jnp.int32(player),
            legal_action_mask=get_legal_moves_mask(jax_board, jnp.int32(player)),
        )
        return state
    
    def _update_check_status(self):
        """更新将军状态"""
        if not JAX_AVAILABLE or self.state is None:
            return
        
        jax_board = jnp.array(self.state.board, dtype=jnp.int8)
        player = jnp.int32(self.state.current_player)
        
        self.state.is_check = bool(is_in_check(jax_board, player))
        if self.state.is_check:
            king_row, king_col = find_king(jax_board, player)
            self.state.king_pos = (int(king_row), int(king_col))
        else:
            self.state.king_pos = None
    
    def get_legal_moves(self, row: int, col: int) -> List[Tuple[int, int]]:
        """获取指定位置棋子的合法走法"""
        if not JAX_AVAILABLE or self.state is None:
            return []
        
        piece = self.state.board[row, col]
        if piece == 0:
            return []
        
        # 检查是否是当前玩家的棋子
        if (self.state.current_player == 0 and piece < 0) or \
           (self.state.current_player == 1 and piece > 0):
            return []
        
        moves = []
        from_sq = row * BOARD_WIDTH + col
        
        legal_mask = self.state.jax_state.legal_action_mask
        
        for to_row in range(BOARD_HEIGHT):
            for to_col in range(BOARD_WIDTH):
                to_sq = to_row * BOARD_WIDTH + to_col
                action = move_to_action(from_sq, to_sq)
                if legal_mask[action]:
                    moves.append((to_row, to_col))
        
        return moves
    
    def click(self, row: int, col: int) -> GameState:
        """处理点击"""
        if self.state is None or self.state.game_over:
            return self.state
        
        piece = self.state.board[row, col]
        is_own_piece = (self.state.current_player == 0 and piece > 0) or \
                       (self.state.current_player == 1 and piece < 0)
        
        if self.state.selected is None:
            # 没有选中棋子，尝试选择
            if is_own_piece:
                self.state.selected = (row, col)
                self.state.legal_moves = self.get_legal_moves(row, col)
        else:
            # 已有选中的棋子
            if (row, col) in self.state.legal_moves:
                # 合法走法，执行
                self._make_move(self.state.selected[0], self.state.selected[1], row, col)
            elif is_own_piece:
                # 选择另一个己方棋子
                self.state.selected = (row, col)
                self.state.legal_moves = self.get_legal_moves(row, col)
            else:
                # 取消选择
                self.state.selected = None
                self.state.legal_moves = []
        
        return self.state
    
    def _make_move(self, from_row: int, from_col: int, to_row: int, to_col: int):
        """执行走棋"""
        if not JAX_AVAILABLE:
            return
        
        # 保存历史
        self.state.history.append({
            'board': self.state.board.copy(),
            'player': self.state.current_player,
            'jax_state': self.state.jax_state,
        })
        
        # 执行走棋
        from_sq = from_row * BOARD_WIDTH + from_col
        to_sq = to_row * BOARD_WIDTH + to_col
        action = move_to_action(from_sq, to_sq)
        
        new_jax_state = self.env.step(self.state.jax_state, action)
        
        # 更新状态
        self.state.board = np.array(new_jax_state.board)
        self.state.current_player = int(new_jax_state.current_player)
        self.state.jax_state = new_jax_state
        self.state.last_move = (from_row, from_col, to_row, to_col)
        self.state.selected = None
        self.state.legal_moves = []
        self.state.step_count += 1
        self.state.game_over = bool(new_jax_state.terminated)
        self.state.winner = int(new_jax_state.winner)
        
        self._update_check_status()
    
    def undo(self) -> GameState:
        """悔棋"""
        if self.state is None or len(self.state.history) == 0:
            return self.state
        
        prev = self.state.history.pop()
        self.state.board = prev['board']
        self.state.current_player = prev['player']
        self.state.jax_state = prev['jax_state']
        self.state.selected = None
        self.state.legal_moves = []
        self.state.last_move = None
        self.state.step_count = max(0, self.state.step_count - 1)
        self.state.game_over = False
        self.state.winner = -1
        
        self._update_check_status()
        return self.state
    
    def ai_move(self) -> GameState:
        """AI 走棋"""
        if self.state is None or self.state.game_over:
            return self.state
        if self.ai_callback is None:
            return self.state
        
        # 调用 AI
        action = self.ai_callback(self.state.jax_state)
        if action is not None:
            from_sq, to_sq = action_to_move(action)
            from_row, from_col = from_sq // BOARD_WIDTH, from_sq % BOARD_WIDTH
            to_row, to_col = to_sq // BOARD_WIDTH, to_sq % BOARD_WIDTH
            self._make_move(from_row, from_col, to_row, to_col)
        
        return self.state


# ============================================================================
# Gradio 界面
# ============================================================================

def create_gui(ai_callback: Optional[Callable] = None):
    """创建 Gradio 界面"""
    
    game = ChessGame()
    game.ai_callback = ai_callback
    
    # 预热 JAX (第一次运行会编译，比较慢)
    print("[Web GUI] 预热 JAX JIT 编译 (首次启动较慢，请等待)...")
    print("[Web GUI] - 初始化环境...")
    game.new_game()
    print("[Web GUI] - 预编译走棋函数...")
    # 执行一次走棋来预编译 step 函数
    if game.state and game.state.jax_state:
        legal_actions = jnp.where(game.state.jax_state.legal_action_mask)[0]
        if len(legal_actions) > 0:
            test_action = int(legal_actions[0])
            _ = game.env.step(game.state.jax_state, test_action)
    print("[Web GUI] JAX 预热完成!")
    
    def render():
        """渲染当前状态"""
        if game.state is None:
            game.new_game()
        
        svg = render_board_svg(
            board=game.state.board,
            current_player=game.state.current_player,
            selected=game.state.selected,
            legal_moves=game.state.legal_moves,
            last_move=game.state.last_move,
            is_check=game.state.is_check,
            king_pos=game.state.king_pos,
        )
        
        # 状态信息
        player_name = "红方" if game.state.current_player == 0 else "黑方"
        status = f"当前: {player_name} | 步数: {game.state.step_count}"
        
        if game.state.game_over:
            if game.state.winner == 0:
                status = "🎉 游戏结束 - 红方胜！"
            elif game.state.winner == 1:
                status = "🎉 游戏结束 - 黑方胜！"
            else:
                status = "🤝 游戏结束 - 和棋"
        elif game.state.is_check:
            status += " | ⚠️ 将军！"
        
        fen = board_to_fen(game.state.board, game.state.current_player)
        
        return svg, status, fen
    
    def on_click(row: int, col: int):
        """处理点击"""
        game.click(row, col)
        return render()
    
    def new_game_click():
        """新游戏"""
        game.new_game()
        return render()
    
    def undo_click():
        """悔棋"""
        game.undo()
        return render()
    
    def ai_move_click():
        """AI 走棋"""
        game.ai_move()
        return render()
    
    def switch_side_click():
        """换边"""
        game.ai_player = 1 - game.ai_player
        side = "红方" if game.ai_player == 0 else "黑方"
        return f"AI 执{side}"
    
    def import_fen_click(fen: str):
        """导入 FEN"""
        try:
            game.new_game(fen)
            return render() + ("导入成功",)
        except Exception as e:
            svg, status, current_fen = render()
            return svg, status, current_fen, f"导入失败: {e}"
    
    # JavaScript 注入到全局 - 使用事件委托
    js_init = """
    function setupChessBoard() {
        document.addEventListener('click', function(e) {
            const target = e.target;
            if (target.classList && target.classList.contains('click-area')) {
                const row = target.getAttribute('data-row');
                const col = target.getAttribute('data-col');
                if (row !== null && col !== null) {
                    triggerMove(row, col);
                }
            }
        });
    }
    
    function triggerMove(row, col) {
        let rowInput = document.querySelector('#row-input textarea') 
                    || document.querySelector('#row-input input')
                    || document.querySelector('[id*="row-input"] textarea')
                    || document.querySelector('[id*="row-input"] input');
        let colInput = document.querySelector('#col-input textarea')
                    || document.querySelector('#col-input input')
                    || document.querySelector('[id*="col-input"] textarea')
                    || document.querySelector('[id*="col-input"] input');
        let clickBtn = document.querySelector('#click-handler')
                    || document.querySelector('[id*="click-handler"]')
                    || document.querySelector('button[id*="click-handler"]');
        
        if (rowInput && colInput && clickBtn) {
            const setter = rowInput.tagName === 'TEXTAREA' 
                ? Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set
                : Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
            
            setter.call(rowInput, row.toString());
            setter.call(colInput, col.toString());
            
            rowInput.dispatchEvent(new Event('input', { bubbles: true }));
            colInput.dispatchEvent(new Event('input', { bubbles: true }));
            
            setTimeout(() => clickBtn.click(), 30);
        }
    }
    
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', setupChessBoard);
    } else {
        setupChessBoard();
    }
    """
    
    # 创建界面
    with gr.Blocks(title="中国象棋 - ZeroForge", theme=gr.themes.Soft(), js=js_init) as demo:
        gr.Markdown("# 🎮 中国象棋 - ZeroForge AI")
        
        with gr.Row():
            with gr.Column(scale=2):
                board_html = gr.HTML(label="棋盘")
                status_text = gr.Textbox(label="状态", interactive=False)
            
            with gr.Column(scale=1):
                gr.Markdown("### 操作")
                
                with gr.Row():
                    new_game_btn = gr.Button("🆕 新游戏", variant="primary")
                    undo_btn = gr.Button("↩️ 悔棋")
                
                with gr.Row():
                    ai_move_btn = gr.Button("🤖 AI走棋", variant="secondary")
                    switch_btn = gr.Button("🔄 换边")
                
                switch_status = gr.Textbox(value="AI 执黑方", label="AI 设置", interactive=False)
                
                gr.Markdown("### FEN")
                fen_input = gr.Textbox(label="FEN 字符串", placeholder="输入 FEN...")
                
                with gr.Row():
                    import_btn = gr.Button("📥 导入")
                    # export 由 fen_output 自动显示
                
                fen_output = gr.Textbox(label="当前 FEN", interactive=False)
                import_status = gr.Textbox(label="", interactive=False, visible=False)
                
                gr.Markdown("### 说明")
                gr.Markdown("""
                - 点击棋子选择，再点击目标位置走棋
                - 绿色圆点表示合法走法
                - 黄色圈表示选中的棋子
                - 红色圈表示将军
                """)
        
        # 隐藏的输入用于接收点击 (用 CSS 隐藏，保证 DOM 存在)
        with gr.Row(elem_id="hidden-controls"):
            row_input = gr.Textbox(elem_id="row-input", value="", visible=True, 
                                   container=False, show_label=False)
            col_input = gr.Textbox(elem_id="col-input", value="", visible=True,
                                   container=False, show_label=False)
            click_btn = gr.Button("Click", elem_id="click-handler", visible=True)
        
        # CSS 隐藏这些元素
        gr.HTML("<style>#hidden-controls { display: none !important; }</style>")
        
        # 事件绑定
        def handle_board_click(row_str, col_str):
            try:
                row = int(row_str)
                col = int(col_str)
                return on_click(row, col)
            except:
                return render()
        
        click_btn.click(
            handle_board_click,
            inputs=[row_input, col_input],
            outputs=[board_html, status_text, fen_output]
        )
        
        new_game_btn.click(new_game_click, outputs=[board_html, status_text, fen_output])
        undo_btn.click(undo_click, outputs=[board_html, status_text, fen_output])
        ai_move_btn.click(ai_move_click, outputs=[board_html, status_text, fen_output])
        switch_btn.click(switch_side_click, outputs=[switch_status])
        import_btn.click(
            import_fen_click, 
            inputs=[fen_input],
            outputs=[board_html, status_text, fen_output, import_status]
        )
        
        # 初始化
        demo.load(render, outputs=[board_html, status_text, fen_output])
    
    return demo


def run_web_gui(ai_callback: Optional[Callable] = None, fen: Optional[str] = None, share: bool = False):
    """启动 Web GUI"""
    demo = create_gui(ai_callback)
    demo.launch(share=share, server_name="0.0.0.0")


# ============================================================================
# 入口
# ============================================================================

if __name__ == "__main__":
    run_web_gui()

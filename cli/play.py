"""
CLI 人机对弈模块
提供命令行界面与 AI 对弈
"""

from __future__ import annotations
from typing import Optional, Tuple
import sys
import time
import jax
import jax.numpy as jnp
import logging

from xiangqi.env import XiangqiEnv, XiangqiState
from xiangqi.actions import (
    ACTION_SPACE_SIZE, BOARD_HEIGHT, BOARD_WIDTH,
    move_to_action, action_to_move, uci_to_move, move_to_uci,
    coords_to_square, square_to_coords,
    PIECE_SYMBOLS, EMPTY,
)
from networks.muzero import MuZeroNetwork
from mcts.search import MCTSConfig, run_mcts, select_action
from training.checkpoint import load_params

logger = logging.getLogger(__name__)


# ============================================================================
# 棋盘显示
# ============================================================================

# Unicode 棋子符号 (更美观)
UNICODE_PIECES = {
    1: '帥', 2: '仕', 3: '相', 4: '傌', 5: '俥', 6: '炮', 7: '兵',
    -1: '將', -2: '士', -3: '象', -4: '馬', -5: '車', -6: '砲', -7: '卒',
    0: '．'
}

# ANSI 颜色代码
RED = '\033[91m'
BLACK = '\033[94m'
RESET = '\033[0m'
BOLD = '\033[1m'
GREEN = '\033[92m'
YELLOW = '\033[93m'


def colorize_piece(piece: int) -> str:
    """为棋子添加颜色"""
    symbol = UNICODE_PIECES.get(piece, '？')
    if piece > 0:
        return f"{RED}{symbol}{RESET}"
    elif piece < 0:
        return f"{BLACK}{symbol}{RESET}"
    return symbol


def print_board(
    state: XiangqiState,
    last_move: Optional[Tuple[int, int]] = None,
    highlight_squares: Optional[list] = None,
):
    """
    打印棋盘
    
    Args:
        state: 游戏状态
        last_move: 上一步移动 (from_sq, to_sq)
        highlight_squares: 要高亮的格子列表
    """
    board = state.board
    
    print()
    print(f"  ａ　ｂ　ｃ　ｄ　ｅ　ｆ　ｇ　ｈ　ｉ")
    print("  ─────────────────────────────────")
    
    for row in range(BOARD_HEIGHT - 1, -1, -1):
        line = f"{row}│"
        
        for col in range(BOARD_WIDTH):
            sq = row * BOARD_WIDTH + col
            piece = int(board[row, col])
            
            # 检查是否需要高亮
            highlight = False
            if last_move:
                if sq == last_move[0] or sq == last_move[1]:
                    highlight = True
            if highlight_squares and sq in highlight_squares:
                highlight = True
            
            if highlight:
                line += f"{YELLOW}[{colorize_piece(piece)}{YELLOW}]{RESET}"
            else:
                line += f" {colorize_piece(piece)} "
        
        line += f"│{row}"
        print(line)
        
        # 河界
        if row == 5:
            print("  │　　　楚　河　　　漢　界　　　│")
    
    print("  ─────────────────────────────────")
    print(f"  ａ　ｂ　ｃ　ｄ　ｅ　ｆ　ｇ　ｈ　ｉ")
    print()
    
    # 状态信息
    player = "红方" if state.current_player == 0 else "黑方"
    color = RED if state.current_player == 0 else BLACK
    print(f"当前: {color}{BOLD}{player}{RESET} | 步数: {state.step_count}")


# ============================================================================
# CLI 类
# ============================================================================

class ChessCLI:
    """
    命令行象棋界面
    
    支持:
    - 人机对弈
    - UCI 格式输入
    - 悔棋
    - 提示
    """
    
    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        network: Optional[MuZeroNetwork] = None,
        params: Optional[dict] = None,
        mcts_config: Optional[MCTSConfig] = None,
    ):
        """
        Args:
            checkpoint_path: 检查点路径
            network: 网络实例 (如果不从检查点加载)
            params: 网络参数 (如果不从检查点加载)
            mcts_config: MCTS 配置
        """
        self.env = XiangqiEnv()
        
        # 加载模型
        if checkpoint_path:
            self.params = load_params(checkpoint_path)
            self.network = MuZeroNetwork()  # 使用默认配置
        else:
            self.network = network
            self.params = params
        
        self.mcts_config = mcts_config or MCTSConfig(
            num_simulations=800,
            max_num_considered_actions=16,
            use_dirichlet_noise=False,  # 对弈时不需要探索噪声
        )
        
        # 游戏状态
        self.state = None
        self.history = []  # 用于悔棋
        self.last_move = None
        
        # 随机数
        self.rng_key = jax.random.PRNGKey(int(time.time()))
    
    def new_game(self, human_color: int = 0):
        """
        开始新游戏
        
        Args:
            human_color: 人类玩家颜色 (0=红, 1=黑)
        """
        self.rng_key, init_key = jax.random.split(self.rng_key)
        self.state = self.env.init(init_key)
        self.history = []
        self.last_move = None
        self.human_color = human_color
        
        print(f"\n{BOLD}=== 新游戏开始 ==={RESET}")
        print(f"您执: {RED if human_color == 0 else BLACK}{'红方' if human_color == 0 else '黑方'}{RESET}")
        print("输入 'help' 查看命令帮助\n")
    
    def get_ai_move(self) -> int:
        """AI 选择移动"""
        observation = self.env.observe(self.state)
        obs_batch = observation[jnp.newaxis, ...]
        legal_mask_batch = self.state.legal_action_mask[jnp.newaxis, ...]
        
        self.rng_key, search_key = jax.random.split(self.rng_key)
        
        print("AI 思考中...", end=" ", flush=True)
        start_time = time.time()
        
        # MCTS 搜索
        policy_output = run_mcts(
            observation=obs_batch,
            legal_action_mask=legal_mask_batch,
            network_apply=self.network.apply,
            params=self.params,
            config=self.mcts_config,
            rng_key=search_key,
        )
        
        # 选择最佳动作
        action = select_action(policy_output, greedy=True)
        action = int(action[0])
        
        elapsed = time.time() - start_time
        print(f"完成 ({elapsed:.1f}秒)")
        
        return action
    
    def parse_human_move(self, input_str: str) -> Optional[int]:
        """
        解析人类输入的移动
        
        支持格式:
        - UCI: a0a1, b2e2
        - 坐标: 0,0 0,1 (from_row,from_col to_row,to_col)
        
        Returns:
            动作索引，或 None (如果无效)
        """
        input_str = input_str.strip().lower()
        
        # 尝试 UCI 格式
        if len(input_str) == 4:
            try:
                from_sq, to_sq = uci_to_move(input_str)
                action = move_to_action(jnp.int32(from_sq), jnp.int32(to_sq))
                action = int(action)
                
                if action >= 0 and self.state.legal_action_mask[action]:
                    return action
                else:
                    print(f"{YELLOW}非法移动: {input_str}{RESET}")
                    return None
            except (ValueError, IndexError):
                pass
        
        # 尝试坐标格式
        parts = input_str.replace(',', ' ').split()
        if len(parts) == 4:
            try:
                from_row, from_col, to_row, to_col = map(int, parts)
                from_sq = coords_to_square(from_row, from_col)
                to_sq = coords_to_square(to_row, to_col)
                action = move_to_action(jnp.int32(from_sq), jnp.int32(to_sq))
                action = int(action)
                
                if action >= 0 and self.state.legal_action_mask[action]:
                    return action
                else:
                    print(f"{YELLOW}非法移动{RESET}")
                    return None
            except (ValueError, IndexError):
                pass
        
        print(f"{YELLOW}无法解析输入，请使用 UCI 格式 (如 a0a1) 或坐标格式 (如 0 0 0 1){RESET}")
        return None
    
    def make_move(self, action: int):
        """执行移动"""
        # 保存历史
        self.history.append((self.state, self.last_move))
        
        # 记录移动
        from_sq, to_sq = action_to_move(jnp.int32(action))
        self.last_move = (int(from_sq), int(to_sq))
        
        # 执行
        self.state = self.env.step(self.state, jnp.int32(action))
    
    def undo(self):
        """悔棋"""
        if len(self.history) >= 2:
            # 悔两步 (自己和对手)
            self.state, self.last_move = self.history.pop()
            self.state, self.last_move = self.history.pop()
            print(f"{GREEN}已悔棋{RESET}")
        elif len(self.history) == 1:
            self.state, self.last_move = self.history.pop()
            print(f"{GREEN}已悔棋{RESET}")
        else:
            print(f"{YELLOW}无法悔棋{RESET}")
    
    def show_hints(self):
        """显示提示 (合法移动)"""
        legal_actions = jnp.where(self.state.legal_action_mask)[0]
        
        print(f"\n{GREEN}合法移动 ({len(legal_actions)} 个):{RESET}")
        moves = []
        for action in legal_actions:
            from_sq, to_sq = action_to_move(action)
            uci = move_to_uci(int(from_sq), int(to_sq))
            moves.append(uci)
        
        # 每行显示 10 个
        for i in range(0, len(moves), 10):
            print("  " + "  ".join(moves[i:i+10]))
        print()
    
    def print_help(self):
        """打印帮助"""
        print(f"""
{BOLD}命令帮助:{RESET}
  {GREEN}移动{RESET}   - 使用 UCI 格式 (如 a0a1) 或坐标 (如 0 0 0 1)
  {GREEN}undo{RESET}   - 悔棋
  {GREEN}hint{RESET}   - 显示合法移动
  {GREEN}new{RESET}    - 开始新游戏
  {GREEN}quit{RESET}   - 退出

{BOLD}UCI 格式:{RESET}
  四个字符: 起始列+起始行+目标列+目标行
  列: a-i (从左到右)
  行: 0-9 (红方在下)
  例如: a0a1 表示把 a0 的棋子移动到 a1
""")
    
    def play(self):
        """主游戏循环"""
        self.new_game()
        
        while True:
            # 显示棋盘
            print_board(self.state, self.last_move)
            
            # 检查游戏是否结束
            if self.state.terminated:
                if self.state.winner == -1:
                    print(f"\n{BOLD}游戏结束: 平局!{RESET}")
                elif self.state.winner == self.human_color:
                    print(f"\n{GREEN}{BOLD}恭喜! 您获胜了!{RESET}")
                else:
                    print(f"\n{YELLOW}{BOLD}AI 获胜!{RESET}")
                
                # 询问是否再来一局
                response = input("\n再来一局? (y/n): ").strip().lower()
                if response == 'y':
                    self.new_game()
                    continue
                else:
                    break
            
            # 判断当前该谁走
            is_human_turn = (int(self.state.current_player) == self.human_color)
            
            if is_human_turn:
                # 人类回合
                prompt = f"{RED if self.human_color == 0 else BLACK}您的移动: {RESET}"
                user_input = input(prompt).strip().lower()
                
                # 处理命令
                if user_input == 'quit' or user_input == 'exit':
                    print("再见!")
                    break
                elif user_input == 'help':
                    self.print_help()
                    continue
                elif user_input == 'undo':
                    self.undo()
                    continue
                elif user_input == 'hint':
                    self.show_hints()
                    continue
                elif user_input == 'new':
                    color = input("选择颜色 (r=红, b=黑, 默认红): ").strip().lower()
                    self.new_game(human_color=1 if color == 'b' else 0)
                    continue
                
                # 解析移动
                action = self.parse_human_move(user_input)
                if action is not None:
                    self.make_move(action)
            else:
                # AI 回合
                if self.network is None or self.params is None:
                    # 如果没有模型，随机走
                    legal_actions = jnp.where(self.state.legal_action_mask)[0]
                    self.rng_key, choice_key = jax.random.split(self.rng_key)
                    idx = jax.random.randint(choice_key, (), 0, len(legal_actions))
                    action = int(legal_actions[idx])
                else:
                    action = self.get_ai_move()
                
                # 显示 AI 的移动
                from_sq, to_sq = action_to_move(jnp.int32(action))
                uci = move_to_uci(int(from_sq), int(to_sq))
                print(f"AI 移动: {GREEN}{uci}{RESET}")
                
                self.make_move(action)


# ============================================================================
# 便捷函数
# ============================================================================

def play_game(checkpoint_path: Optional[str] = None):
    """
    快速开始一局游戏
    
    Args:
        checkpoint_path: 检查点路径 (可选)
    """
    cli = ChessCLI(checkpoint_path=checkpoint_path)
    cli.play()


# ============================================================================
# 主入口
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="中国象棋 CLI")
    parser.add_argument("--checkpoint", type=str, help="模型检查点路径")
    parser.add_argument("--simulations", type=int, default=800, help="MCTS 模拟次数")
    
    args = parser.parse_args()
    
    # 配置
    mcts_config = MCTSConfig(
        num_simulations=args.simulations,
        use_dirichlet_noise=False,
    )
    
    # 启动游戏
    cli = ChessCLI(
        checkpoint_path=args.checkpoint,
        mcts_config=mcts_config,
    )
    cli.play()

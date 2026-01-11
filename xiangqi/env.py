"""
中国象棋 JAX 环境
兼容 pgx 风格的接口，支持向量化和 JIT 编译
"""

from __future__ import annotations
from typing import NamedTuple, Optional
import jax
import jax.numpy as jnp
import chex
from functools import partial

from xiangqi.actions import (
    ACTION_SPACE_SIZE, BOARD_HEIGHT, BOARD_WIDTH, NUM_SQUARES,
    action_to_move, move_to_action,
    EMPTY, R_KING, B_KING,
    PIECE_SYMBOLS,
)
from xiangqi.rules import (
    get_initial_board, apply_move, is_legal_move, is_game_over,
    get_legal_moves_mask, is_in_check, get_piece_type,
)

# ============================================================================
# 配置常量
# ============================================================================

# 历史步数
NUM_HISTORY_STEPS = 16

# 每步的通道数: 7种棋子 × 2方 = 14
CHANNELS_PER_STEP = 14

# 总观察通道数: (16历史 + 1当前) × 14棋子 + 1当前玩家 + 1步数 = 240
NUM_OBSERVATION_CHANNELS = (NUM_HISTORY_STEPS + 1) * CHANNELS_PER_STEP + 2

# 最大步数限制 (用于判断和棋)
MAX_STEPS = 200

# 无吃子步数限制 (用于判断和棋)
MAX_NO_CAPTURE_STEPS = 120

# 重复局面检测: 保存最近 N 步的局面哈希
POSITION_HISTORY_SIZE = 100

# 重复次数阈值 (三次重复判和)
# 训练时设高一点，避免模型学会"重复走棋"逃避胜负
REPETITION_THRESHOLD = 999  # 暂时禁用，强制分出胜负

# 长将检测: 连续将军次数阈值 (超过判负)
PERPETUAL_CHECK_THRESHOLD = 6


# ============================================================================
# 状态定义
# ============================================================================

@chex.dataclass
class XiangqiState:
    """中国象棋游戏状态"""
    
    # 当前棋盘 (10, 9) int8
    board: jnp.ndarray
    
    # 历史棋盘 (NUM_HISTORY_STEPS, 10, 9) int8
    history: jnp.ndarray
    
    # 当前玩家: 0=红方, 1=黑方
    current_player: jnp.int32
    
    # 合法动作掩码 (ACTION_SPACE_SIZE,) bool
    legal_action_mask: jnp.ndarray
    
    # 奖励 (2,) float32 - [红方奖励, 黑方奖励]
    rewards: jnp.ndarray
    
    # 是否结束
    terminated: jnp.bool_
    
    # 总步数
    step_count: jnp.int32
    
    # 无吃子步数 (用于和棋判断)
    no_capture_count: jnp.int32
    
    # 胜者: -1=未结束/平局, 0=红胜, 1=黑胜
    winner: jnp.int32
    
    # === 违规检测相关 ===
    
    # 局面哈希历史 (POSITION_HISTORY_SIZE,) int32 - 用于重复局面检测
    position_hashes: jnp.ndarray
    
    # 当前有效哈希数量
    hash_count: jnp.int32
    
    # 红方连续将军次数
    red_consecutive_checks: jnp.int32
    
    # 黑方连续将军次数
    black_consecutive_checks: jnp.int32


# ============================================================================
# 局面哈希 (用于重复局面检测)
# ============================================================================

# Zobrist 哈希: 预计算随机数表
# 形状: (棋子种类数+1, 棋盘格子数) = (15, 90)
# 棋子编码: -7 到 7 (0=空), 所以需要偏移
# 注意: 使用 int32 避免 JAX x64 模式问题，32 位哈希对于检测重复局面足够
_ZOBRIST_SEED = 42
_ZOBRIST_TABLE = jax.random.randint(
    jax.random.PRNGKey(_ZOBRIST_SEED),
    shape=(15, NUM_SQUARES),
    minval=jnp.iinfo(jnp.int32).min,
    maxval=jnp.iinfo(jnp.int32).max,
    dtype=jnp.int32
)

# 当前玩家的哈希值
_PLAYER_HASH = jax.random.randint(
    jax.random.PRNGKey(_ZOBRIST_SEED + 1),
    shape=(2,),
    minval=jnp.iinfo(jnp.int32).min,
    maxval=jnp.iinfo(jnp.int32).max,
    dtype=jnp.int32
)


@jax.jit
def compute_position_hash(board: jnp.ndarray, player: jnp.int32) -> jnp.int32:
    """
    计算局面的 Zobrist 哈希值
    
    Args:
        board: 棋盘状态 (10, 9)
        player: 当前玩家
        
    Returns:
        64位哈希值
    """
    flat_board = board.flatten()  # (90,)
    # 棋子值范围 [-7, 7], 转换为 [0, 14] 作为索引
    piece_indices = flat_board + 7
    
    # 获取每个位置的哈希值
    square_indices = jnp.arange(NUM_SQUARES)
    piece_hashes = _ZOBRIST_TABLE[piece_indices, square_indices]
    
    # XOR 所有哈希值
    board_hash = jnp.bitwise_xor.reduce(piece_hashes)
    
    # 加入玩家信息
    return board_hash ^ _PLAYER_HASH[player]


@jax.jit
def count_repetitions(position_hash: jnp.int32, 
                       position_hashes: jnp.ndarray, 
                       hash_count: jnp.int32) -> jnp.int32:
    """
    计算当前局面在历史中出现的次数
    
    Args:
        position_hash: 当前局面哈希
        position_hashes: 历史哈希数组
        hash_count: 有效哈希数量
        
    Returns:
        重复次数 (包括当前局面)
    """
    # 只比较有效的历史哈希
    valid_mask = jnp.arange(POSITION_HISTORY_SIZE) < hash_count
    matches = (position_hashes == position_hash) & valid_mask
    return jnp.sum(matches) + 1  # +1 包括当前局面


# ============================================================================
# 环境类
# ============================================================================

class XiangqiEnv:
    """
    中国象棋环境
    
    遵循 pgx 风格的接口:
    - init(key) -> State
    - step(state, action) -> State
    - observe(state) -> observation
    """
    
    def __init__(self):
        self.num_players = 2
        self.action_space_size = ACTION_SPACE_SIZE
        self.observation_shape = (NUM_OBSERVATION_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH)
    
    @partial(jax.jit, static_argnums=(0,))
    def init(self, key: jax.random.PRNGKey) -> XiangqiState:
        """
        初始化游戏状态
        
        Args:
            key: JAX 随机数密钥 (当前未使用，但保持接口兼容)
            
        Returns:
            初始游戏状态
        """
        board = get_initial_board()
        history = jnp.zeros((NUM_HISTORY_STEPS, BOARD_HEIGHT, BOARD_WIDTH), dtype=jnp.int8)
        current_player = jnp.int32(0)  # 红方先行
        legal_mask = get_legal_moves_mask(board, current_player)
        
        # 计算初始局面哈希
        init_hash = compute_position_hash(board, current_player)
        position_hashes = jnp.zeros(POSITION_HISTORY_SIZE, dtype=jnp.int32)
        position_hashes = position_hashes.at[0].set(init_hash)
        
        return XiangqiState(
            board=board,
            history=history,
            current_player=current_player,
            legal_action_mask=legal_mask,
            rewards=jnp.zeros(2, dtype=jnp.float32),
            terminated=jnp.bool_(False),
            step_count=jnp.int32(0),
            no_capture_count=jnp.int32(0),
            winner=jnp.int32(-1),
            # 违规检测
            position_hashes=position_hashes,
            hash_count=jnp.int32(1),
            red_consecutive_checks=jnp.int32(0),
            black_consecutive_checks=jnp.int32(0),
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: XiangqiState, action: jnp.ndarray) -> XiangqiState:
        """
        执行一步动作
        
        Args:
            state: 当前状态
            action: 动作索引
            
        Returns:
            新状态
        """
        # 如果游戏已结束，直接返回（安全保护）
        # 注意：自我对弈通常会做 auto-reset；但这里仍需要兜底，
        # 否则终局后 legal_action_mask 全 False，MCTS 可能出现全 invalid 的异常行为。
        def _do_step() -> XiangqiState:
            # 解码动作
            from_sq, to_sq = action_to_move(action)
            
            # 检查是否吃子
            to_row = to_sq // BOARD_WIDTH
            to_col = to_sq % BOARD_WIDTH
            captured_piece = state.board[to_row, to_col]
            is_capture = captured_piece != EMPTY
            
            # 更新历史
            # 将当前棋盘推入历史，移除最老的一步
            new_history = jnp.concatenate([
                state.board[jnp.newaxis, :, :],
                state.history[:-1, :, :]
            ], axis=0)
            
            # 执行移动
            new_board = apply_move(state.board, from_sq, to_sq)
            
            # 切换玩家
            new_player = 1 - state.current_player
            
            # 更新步数
            new_step_count = state.step_count + 1
            new_no_capture = jnp.where(is_capture, 0, state.no_capture_count + 1)
            
            # ========== 违规检测 ==========
            
            # 1. 计算新局面哈希
            new_hash = compute_position_hash(new_board, new_player)
            
            # 2. 更新哈希历史 (循环缓冲区)
            hash_idx = state.hash_count % POSITION_HISTORY_SIZE
            new_position_hashes = state.position_hashes.at[hash_idx].set(new_hash)
            new_hash_count = jnp.minimum(state.hash_count + 1, POSITION_HISTORY_SIZE)
            
            # 3. 检测重复局面
            repetitions = count_repetitions(new_hash, state.position_hashes, state.hash_count)
            is_threefold_repetition = repetitions >= REPETITION_THRESHOLD
            
            # 4. 检测将军状态 (对方是否被将军)
            is_check = is_in_check(new_board, new_player)
            
            # 5. 更新连续将军计数
            # 如果红方走棋后黑方被将军，红方连续将军+1
            # 如果红方走棋后黑方没被将军，红方连续将军清零
            new_red_checks = jnp.where(
                state.current_player == 0,  # 红方刚走
                jnp.where(is_check, state.red_consecutive_checks + 1, jnp.int32(0)),
                state.red_consecutive_checks
            )
            new_black_checks = jnp.where(
                state.current_player == 1,  # 黑方刚走
                jnp.where(is_check, state.black_consecutive_checks + 1, jnp.int32(0)),
                state.black_consecutive_checks
            )
            
            # 6. 检测长将 (连续将军超过阈值)
            red_perpetual_check = new_red_checks >= PERPETUAL_CHECK_THRESHOLD
            black_perpetual_check = new_black_checks >= PERPETUAL_CHECK_THRESHOLD
            
            # ========== 游戏结束判断 ==========
            
            # 检查基本游戏结束条件 (将死/困毙)
            game_over, winner = is_game_over(new_board, new_player)
            
            # 检查和棋条件: 步数限制、无吃子限制、三次重复
            is_draw = (new_step_count >= MAX_STEPS) | \
                      (new_no_capture >= MAX_NO_CAPTURE_STEPS) | \
                      is_threefold_repetition
            
            # 检查长将判负
            # 红方长将 -> 红方负 (黑方胜)
            # 黑方长将 -> 黑方负 (红方胜)
            perpetual_check_loss = red_perpetual_check | black_perpetual_check
            perpetual_winner = jnp.where(red_perpetual_check, 1, jnp.where(black_perpetual_check, 0, -1))
            
            # 综合判断
            game_over = game_over | is_draw | perpetual_check_loss
            
            # 确定最终胜者
            # 优先级: 将死 > 长将判负 > 和棋
            winner = jnp.where(
                winner != -1,
                winner,  # 已有胜者 (将死)
                jnp.where(
                    perpetual_check_loss,
                    perpetual_winner,  # 长将判负
                    jnp.where(is_draw, -1, -1)  # 和棋
                )
            )
            
            # 计算奖励
            # 胜: +1, 负: -1, 平: 0, 未结束: 0
            rewards = jnp.where(
                game_over,
                jnp.where(
                    winner == -1,
                    jnp.zeros(2),  # 平局
                    jnp.where(
                        winner == 0,
                        jnp.array([1.0, -1.0]),  # 红胜
                        jnp.array([-1.0, 1.0])   # 黑胜
                    )
                ),
                jnp.zeros(2)
            )
            
            # 获取新的合法动作
            new_legal_mask = jnp.where(
                game_over,
                jnp.zeros(ACTION_SPACE_SIZE, dtype=jnp.bool_),
                get_legal_moves_mask(new_board, new_player)
            )
            
            return XiangqiState(
                board=new_board,
                history=new_history,
                current_player=new_player,
                legal_action_mask=new_legal_mask,
                rewards=rewards,
                terminated=game_over,
                step_count=new_step_count,
                no_capture_count=new_no_capture,
                winner=winner,
                # 违规检测状态
                position_hashes=new_position_hashes,
                hash_count=new_hash_count,
                red_consecutive_checks=new_red_checks,
                black_consecutive_checks=new_black_checks,
            )
        
        return jax.lax.cond(state.terminated, lambda: state, _do_step)
    
    @partial(jax.jit, static_argnums=(0,))
    def observe(self, state: XiangqiState) -> jnp.ndarray:
        """
        将状态转换为观察张量
        
        Args:
            state: 游戏状态
            
        Returns:
            观察张量 (NUM_OBSERVATION_CHANNELS, 10, 9)
        """
        # 编码当前棋盘和历史
        def encode_board(board: jnp.ndarray) -> jnp.ndarray:
            """将棋盘编码为 14 通道的 one-hot 表示"""
            channels = []
            # 红方 7 种棋子
            for piece_type in range(1, 8):
                channels.append((board == piece_type).astype(jnp.float32))
            # 黑方 7 种棋子
            for piece_type in range(-7, 0):
                channels.append((board == piece_type).astype(jnp.float32))
            return jnp.stack(channels, axis=0)
        
        # 编码当前棋盘
        current_encoded = encode_board(state.board)
        
        # 编码历史棋盘
        history_encoded = jax.vmap(encode_board)(state.history)
        history_flat = history_encoded.reshape(-1, BOARD_HEIGHT, BOARD_WIDTH)
        
        # 合并
        board_planes = jnp.concatenate([current_encoded, history_flat], axis=0)
        
        # 添加元信息平面
        # 当前玩家 (全 0 或全 1)
        player_plane = jnp.full(
            (1, BOARD_HEIGHT, BOARD_WIDTH),
            state.current_player,
            dtype=jnp.float32
        )
        
        # 步数 (归一化到 0-1)
        step_plane = jnp.full(
            (1, BOARD_HEIGHT, BOARD_WIDTH),
            state.step_count / MAX_STEPS,
            dtype=jnp.float32
        )
        
        # 最终观察
        observation = jnp.concatenate([board_planes, player_plane, step_plane], axis=0)
        
        return observation
    
    def render(self, state: XiangqiState) -> str:
        """
        渲染棋盘为字符串 (用于调试和显示)
        
        Args:
            state: 游戏状态
            
        Returns:
            棋盘的字符串表示
        """
        board = state.board
        lines = []
        
        # 顶部坐标
        lines.append("  ａｂｃｄｅｆｇｈｉ")
        
        # 棋盘 (从上到下，即黑方视角)
        for row in range(BOARD_HEIGHT - 1, -1, -1):
            line = f"{row} "
            for col in range(BOARD_WIDTH):
                piece = int(board[row, col])
                line += PIECE_SYMBOLS.get(piece, '？')
            lines.append(line)
            
            # 河界
            if row == 5:
                lines.append("  ＝＝＝＝＝＝＝＝＝")
        
        # 底部坐标
        lines.append("  ａｂｃｄｅｆｇｈｉ")
        
        # 状态信息
        player_name = "红方" if state.current_player == 0 else "黑方"
        lines.append(f"\n当前: {player_name} | 步数: {state.step_count}")
        
        if state.terminated:
            if state.winner == 0:
                lines.append("游戏结束: 红方胜!")
            elif state.winner == 1:
                lines.append("游戏结束: 黑方胜!")
            else:
                lines.append("游戏结束: 平局!")
        
        return "\n".join(lines)


# ============================================================================
# 向量化环境
# ============================================================================

class VectorizedXiangqiEnv:
    """
    向量化的中国象棋环境
    支持批量处理多个游戏
    """
    
    def __init__(self, batch_size: int):
        self.batch_size = batch_size
        self.env = XiangqiEnv()
        
        # 向量化 init 和 step
        self.v_init = jax.vmap(self.env.init)
        self.v_step = jax.vmap(self.env.step)
        self.v_observe = jax.vmap(self.env.observe)
    
    @partial(jax.jit, static_argnums=(0,))
    def init(self, key: jax.random.PRNGKey) -> XiangqiState:
        """批量初始化"""
        keys = jax.random.split(key, self.batch_size)
        return self.v_init(keys)
    
    @partial(jax.jit, static_argnums=(0,))
    def step(self, state: XiangqiState, actions: jnp.ndarray) -> XiangqiState:
        """批量执行动作"""
        return self.v_step(state, actions)
    
    @partial(jax.jit, static_argnums=(0,))
    def observe(self, state: XiangqiState) -> jnp.ndarray:
        """批量获取观察"""
        return self.v_observe(state)


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    # 创建环境
    env = XiangqiEnv()
    
    # 初始化
    key = jax.random.PRNGKey(42)
    state = env.init(key)
    
    print("初始棋盘:")
    print(env.render(state))
    print(f"\n观察形状: {env.observe(state).shape}")
    print(f"合法动作数: {state.legal_action_mask.sum()}")
    
    # 测试一步移动 (炮二平五)
    # 红炮从 (2, 1) 移动到 (2, 4)
    from xiangqi.actions import move_to_action, coords_to_square
    from_sq = coords_to_square(2, 1)  # 红炮位置
    to_sq = coords_to_square(2, 4)    # 目标位置
    action = move_to_action(jnp.int32(from_sq), jnp.int32(to_sq))
    
    print(f"\n执行动作: 炮二平五 (action={action})")
    
    if action >= 0 and state.legal_action_mask[action]:
        state = env.step(state, action)
        print("\n移动后:")
        print(env.render(state))
    else:
        print("动作非法!")

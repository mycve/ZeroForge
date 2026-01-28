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
    R_ADVISOR, R_BISHOP, R_KNIGHT, R_ROOK, R_CANNON, R_PAWN,
    B_ADVISOR, B_BISHOP, B_KNIGHT, B_ROOK, B_CANNON, B_PAWN,
    PIECE_SYMBOLS,
)
from xiangqi.rules import (
    get_initial_board, apply_move, is_legal_move, is_game_over,
    get_legal_moves_mask, is_in_check, get_piece_type,
)
from xiangqi.violation_rules import (
    is_chase_move, count_checking_pieces, check_violation,
    update_no_capture_check_count, MAX_CHECK_IN_NO_CAPTURE,
)

# ============================================================================
# 配置常量
# ============================================================================

# 历史步数
NUM_HISTORY_STEPS = 8

# 每步的通道数: 7种棋子 × 2方 = 14
CHANNELS_PER_STEP = 14

# 总观察通道数: (8历史 + 1当前) × 14棋子 = 126
NUM_OBSERVATION_CHANNELS = (NUM_HISTORY_STEPS + 1) * CHANNELS_PER_STEP

# 重复局面检测: 保存最近 N 步的局面哈希
POSITION_HISTORY_SIZE = 256

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
    
    # 结束原因: 0=未结束, 1=步数到限, 2=无吃子到限, 3=重复局面和棋, 
    #          4=长将判负, 5=无进攻子力, 6=长捉判负, 7=将捉交替判负, 8=将死/困毙
    draw_reason: jnp.int32
    
    # === 重复局面检测 ===
    
    # 局面哈希历史 (POSITION_HISTORY_SIZE,) int32
    position_hashes: jnp.ndarray
    
    # 当前有效哈希数量
    hash_count: jnp.int32
    
    # === 违规检测相关（长将/长捉/将捉交替）===
    
    # 红方违规计数
    red_check_count: jnp.int32       # 红方连续将军回合数
    red_chase_count: jnp.int32       # 红方连续捉子回合数
    red_alt_count: jnp.int32         # 红方将捉交替回合数
    red_max_check_pieces: jnp.int32  # 红方长将中参与将军的最大子数
    
    # 黑方违规计数
    black_check_count: jnp.int32
    black_chase_count: jnp.int32
    black_alt_count: jnp.int32
    black_max_check_pieces: jnp.int32
    
    # 无吃子期间的将军步数（用于60回合判和中将军最多20回合的规则）
    check_in_no_capture: jnp.int32
    
    # === 搜索辅助 (不影响环境逻辑) ===
    # 标记当前搜索归属于哪个模型 (0=model_red/当前模型, 1=model_black/历史模型)
    # 仅在 evaluate 时使用，避免 TracerBoolConversionError
    search_model_index: jnp.int32 = 0


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
        hash_count: 有效哈希数量（可能大于 POSITION_HISTORY_SIZE）
        
    Returns:
        重复次数 (包括当前局面)
    """
    # 只比较有效的历史哈希，数量上限为缓冲区大小
    valid_count = jnp.minimum(hash_count, POSITION_HISTORY_SIZE)
    valid_mask = jnp.arange(POSITION_HISTORY_SIZE) < valid_count
    matches = (position_hashes == position_hash) & valid_mask
    return jnp.sum(matches) + 1  # +1 包括当前局面


@jax.jit
def has_attacking_pieces(board: jnp.ndarray, player: jnp.int32) -> jnp.bool_:
    """
    检查指定玩家是否有进攻子力（车、马、炮、兵/卒）
    
    进攻子力定义：车(5)、马(4)、炮(6)、兵/卒(7)
    帅/将(1)、仕/士(2)、相/象(3) 不算进攻子力
    
    Args:
        board: 棋盘状态 (10, 9)
        player: 玩家 (0=红方, 1=黑方)
        
    Returns:
        是否有进攻子力
    """
    # 红方棋子为正，黑方棋子为负
    # 进攻子力：车(±5)、马(±4)、炮(±6)、兵/卒(±7)
    if_red = player == 0
    
    # 红方进攻子力
    red_rook = jnp.sum(board == R_ROOK)      # 车 (5)
    red_knight = jnp.sum(board == R_KNIGHT)  # 马 (4)
    red_cannon = jnp.sum(board == R_CANNON)  # 炮 (6)
    red_pawn = jnp.sum(board == R_PAWN)      # 兵 (7)
    red_attacking = red_rook + red_knight + red_cannon + red_pawn
    
    # 黑方进攻子力
    black_rook = jnp.sum(board == B_ROOK)      # 车 (-5)
    black_knight = jnp.sum(board == B_KNIGHT)  # 马 (-4)
    black_cannon = jnp.sum(board == B_CANNON)  # 炮 (-6)
    black_pawn = jnp.sum(board == B_PAWN)      # 卒 (-7)
    black_attacking = black_rook + black_knight + black_cannon + black_pawn
    
    return jnp.where(if_red, red_attacking > 0, black_attacking > 0)


@jax.jit
def no_attacking_pieces(board: jnp.ndarray) -> jnp.bool_:
    """
    检查双方是否都没有进攻子力（自动判和条件）
    
    Args:
        board: 棋盘状态 (10, 9)
        
    Returns:
        True 如果双方都没有进攻子力
    """
    red_has = has_attacking_pieces(board, jnp.int32(0))
    black_has = has_attacking_pieces(board, jnp.int32(1))
    return ~red_has & ~black_has


# ============================================================================
# 环境类
# ============================================================================

class XiangqiEnv:
    """
    中国象棋环境
    """
    
    def __init__(
        self, 
        max_steps: int = 400,
        max_no_capture_steps: int = 120,
        repetition_threshold: int = 5,  # 非将非捉重复局面5次判和
    ):
        self.num_players = 2
        self.action_space_size = ACTION_SPACE_SIZE
        self.observation_shape = (NUM_OBSERVATION_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH)
        
        # 统一配置参数
        self.max_steps = max_steps
        self.max_no_capture_steps = max_no_capture_steps
        self.repetition_threshold = repetition_threshold
        # 长将/长捉规则参数（符合正式比赛规则）
        # 长将：1子6回合、2子12回合、3子+18回合
        # 长捉：6回合
        # 将捉交替：1子12回合、多子18回合
    
    @partial(jax.jit, static_argnums=(0,))
    def init(self, key: jax.random.PRNGKey) -> XiangqiState:
        """
        初始化游戏状态
        
        Args:
            key: JAX 随机数密钥
            
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
            draw_reason=jnp.int32(0),
            # 重复局面检测
            position_hashes=position_hashes,
            hash_count=jnp.int32(1),
            # 红方违规计数
            red_check_count=jnp.int32(0),
            red_chase_count=jnp.int32(0),
            red_alt_count=jnp.int32(0),
            red_max_check_pieces=jnp.int32(0),
            # 黑方违规计数
            black_check_count=jnp.int32(0),
            black_chase_count=jnp.int32(0),
            black_alt_count=jnp.int32(0),
            black_max_check_pieces=jnp.int32(0),
            # 无吃子期间将军计数
            check_in_no_capture=jnp.int32(0),
            # 搜索辅助
            search_model_index=jnp.int32(0),
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
            
            # ========== 重复局面检测 ==========
            
            new_hash = compute_position_hash(new_board, new_player)
            hash_idx = state.hash_count % POSITION_HISTORY_SIZE
            new_position_hashes = state.position_hashes.at[hash_idx].set(new_hash)
            new_hash_count = state.hash_count + 1
            
            repetitions = count_repetitions(new_hash, state.position_hashes, state.hash_count)
            is_fivefold_repetition = repetitions >= self.repetition_threshold
            
            # ========== 将军检测 ==========
            
            is_check = is_in_check(new_board, new_player)
            checking_pieces = jnp.where(is_check, count_checking_pieces(new_board, state.current_player), jnp.int32(0))
            
            # ========== 捉子检测 ==========
            
            is_chase, chased_sq, chased_type = is_chase_move(
                state.board, new_board, from_sq, to_sq, state.current_player
            )
            # 将军时不算捉（将优先于捉）
            is_chase = is_chase & ~is_check
            
            # ========== 违规检测（长将/长捉/将捉交替）==========
            
            (
                violation_type, violator,
                new_red_check, new_red_chase, new_red_alt, new_red_max_pieces,
                new_black_check, new_black_chase, new_black_alt, new_black_max_pieces,
            ) = check_violation(
                is_check=is_check,
                is_chase=is_chase,
                checking_pieces=checking_pieces,
                is_capture=is_capture,
                red_check_count=state.red_check_count,
                red_chase_count=state.red_chase_count,
                red_alt_count=state.red_alt_count,
                red_max_check_pieces=state.red_max_check_pieces,
                black_check_count=state.black_check_count,
                black_chase_count=state.black_chase_count,
                black_alt_count=state.black_alt_count,
                black_max_check_pieces=state.black_max_check_pieces,
                current_player=state.current_player,
            )
            
            # ========== 无吃子和棋（含将军特殊处理）==========
            
            new_no_capture, new_check_in_no_capture, is_no_capture_draw = update_no_capture_check_count(
                state.no_capture_count, state.check_in_no_capture, is_capture, is_check
            )
            
            # ========== 游戏结束判断 ==========
            
            # 基本结束条件 (将死/困毙)
            basic_game_over, basic_winner = is_game_over(new_board, new_player)
            
            # 和棋条件
            is_max_steps = new_step_count >= self.max_steps
            is_no_attackers = no_attacking_pieces(new_board)
            
            # ========== 重复局面处理（关键修复：考虑长将循环）==========
            # 
            # 问题场景：车炮连环将军
            # 1. 红方走车将军 → is_check=True, red_check_count=1
            # 2. 黑方应将（将上来）→ is_check=False, red_check_count=1（保持）
            # 3. 红方走车继续将军 → is_check=True, red_check_count=2
            # 4. 黑方应将（将下去）→ is_check=False, red_check_count=2（保持）
            # 
            # 当重复局面达到5次时：
            # - 如果任一方 check_count >= 2，说明处于长将循环中
            # - 此时应判长将方负，而不是和棋
            # 
            
            # 检查是否存在活跃的长将循环（任一方连续将军 >= 2 回合）
            has_active_perpetual_check = (new_red_check >= 2) | (new_black_check >= 2)
            
            # 重复局面 + 长将循环 = 长将判负
            is_repetition_check = is_fivefold_repetition & (is_check | has_active_perpetual_check)
            
            # 确定长将的违规方（进行将军的一方）
            perpetual_check_violator = jnp.where(
                new_red_check > new_black_check,
                jnp.int32(0),  # 红方在长将
                jnp.where(
                    new_black_check > new_red_check,
                    jnp.int32(1),  # 黑方在长将
                    state.current_player  # 相等时，当前走子方负
                )
            )
            
            # 重复+捉 = 长捉（排除长将循环）
            is_repetition_chase = is_fivefold_repetition & is_chase & ~is_check & ~has_active_perpetual_check
            
            # 普通重复（非将非捉且不在长将循环中）= 和棋
            is_normal_repetition = is_fivefold_repetition & ~is_check & ~is_chase & ~has_active_perpetual_check
            
            # 违规判负（来自 check_violation 的检测）
            has_violation = violation_type > 0
            violation_winner = jnp.where(violator == 0, 1, jnp.where(violator == 1, 0, -1))  # 违规方判负
            
            # 综合和棋判断（注意：长将循环中的重复不判和棋）
            is_draw = is_max_steps | is_no_capture_draw | is_normal_repetition | is_no_attackers
            
            # 记录结束原因
            # 0=未结束, 1=步数到限, 2=无吃子到限, 3=重复局面和棋, 
            # 4=长将判负, 5=无进攻子力, 6=长捉判负, 7=将捉交替判负, 8=将死/困毙
            new_draw_reason = jnp.where(
                basic_winner != -1, 8,  # 将死/困毙
                jnp.where(
                    has_violation, 
                    jnp.where(violation_type == 1, 4,  # 长将
                              jnp.where(violation_type == 2, 6,  # 长捉
                                        7)),  # 将捉交替
                    jnp.where(
                        is_repetition_check, 4,  # 重复局面+将军（备用规则）
                        jnp.where(
                            is_repetition_chase, 6,  # 重复局面+捉（备用规则）
                            jnp.where(
                                is_no_attackers, 5,
                                jnp.where(
                                    is_normal_repetition, 3,
                                    jnp.where(
                                        is_no_capture_draw, 2,
                                        jnp.where(is_max_steps, 1, 0)
                                    )
                                )
                            )
                        )
                    )
                )
            )
            
            # 综合游戏结束
            game_over = (basic_game_over | is_draw | has_violation | 
                        is_repetition_check | is_repetition_chase)
            
            # 确定最终胜者
            # 优先级: 将死/困毙 > 违规判负 > 重复局面判负 > 和棋
            
            # 长将重复：长将方判负
            perpetual_check_winner = jnp.where(perpetual_check_violator == 0, 1, 0)
            
            # 长捉重复：走子方判负
            repetition_chase_winner = jnp.where(state.current_player == 0, 1, 0)
            
            winner = jnp.where(
                basic_winner != -1, basic_winner,  # 将死/困毙
                jnp.where(
                    has_violation, violation_winner,  # 违规判负（包含长将/长捉/将捉交替）
                    jnp.where(
                        is_repetition_check, perpetual_check_winner,  # 重复+长将：长将方负
                        jnp.where(
                            is_repetition_chase, repetition_chase_winner,  # 重复+捉：走子方负
                            -1  # 和棋
                        )
                    )
                )
            )
            
            # ========== 计算奖励 ==========
            
            terminal_reward = jnp.where(
                game_over,
                jnp.where(
                    winner == -1,
                    jnp.zeros(2),  # 和棋
                    jnp.where(
                        winner == 0,
                        jnp.array([1.0, -1.0]),  # 红胜
                        jnp.array([-1.0, 1.0])   # 黑胜
                    )
                ),
                jnp.zeros(2)
            )
            rewards = terminal_reward
            
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
                draw_reason=jnp.where(game_over, new_draw_reason, jnp.int32(0)),
                # 重复局面检测
                position_hashes=new_position_hashes,
                hash_count=new_hash_count,
                # 红方违规计数
                red_check_count=new_red_check,
                red_chase_count=new_red_chase,
                red_alt_count=new_red_alt,
                red_max_check_pieces=new_red_max_pieces,
                # 黑方违规计数
                black_check_count=new_black_check,
                black_chase_count=new_black_chase,
                black_alt_count=new_black_alt,
                black_max_check_pieces=new_black_max_pieces,
                # 无吃子期间将军计数
                check_in_no_capture=new_check_in_no_capture,
                # 搜索辅助
                search_model_index=state.search_model_index,
            )
        
        return jax.lax.cond(state.terminated, lambda: state, _do_step)
    
    @partial(jax.jit, static_argnums=(0,))
    def observe(self, state: XiangqiState) -> jnp.ndarray:
        """
        将状态转换为观察张量 (视角归一化)
        始终让当前玩家在棋盘下方，且当前玩家棋子为正
        
        Args:
            state: 游戏状态
            
        Returns:
            观察张量 (NUM_OBSERVATION_CHANNELS, 10, 9)
        """
        # 1. 视角变换
        # 如果是黑方 (1)，则旋转 180 度并翻转棋子符号
        # 红方 (0) 保持不变
        is_black = state.current_player == 1
        
        board = jnp.where(
            is_black,
            -jnp.flip(state.board, axis=(0, 1)),
            state.board
        )
        
        history = jnp.where(
            is_black,
            -jnp.flip(state.history, axis=(1, 2)),
            state.history
        )
        
        # 编码当前棋盘和历史
        _PIECE_TYPES = jnp.array([1, 2, 3, 4, 5, 6, 7, -7, -6, -5, -4, -3, -2, -1])

        def encode_board(b: jnp.ndarray) -> jnp.ndarray:
            """将棋盘编码为 14 通道的 one-hot 表示 (向量化加速)"""
            # 形状: (14, 10, 9)
            return (b[..., None, :, :] == _PIECE_TYPES[:, None, None]).astype(jnp.float32)

        # 编码当前棋盘 (14, 10, 9)
        current_encoded = encode_board(board)
        
        # 编码历史棋盘 (NUM_HISTORY_STEPS, 14, 10, 9)
        history_encoded = jax.vmap(encode_board)(history)
        history_flat = history_encoded.reshape(-1, BOARD_HEIGHT, BOARD_WIDTH)
        
        # 最终观察: 仅包含棋盘和历史平面 (126, 10, 9)
        observation = jnp.concatenate([current_encoded, history_flat], axis=0)
        
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

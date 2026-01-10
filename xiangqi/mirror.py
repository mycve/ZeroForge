"""
中国象棋镜像数据增强模块
支持棋盘、动作和策略的左右镜像变换
"""

from __future__ import annotations
import jax
import jax.numpy as jnp
from functools import partial

from xiangqi.actions import (
    BOARD_WIDTH, BOARD_HEIGHT, NUM_SQUARES, ACTION_SPACE_SIZE,
    _ACTION_TO_FROM_SQ, _ACTION_TO_TO_SQ, _FROM_TO_ACTION_TABLE,
)


# ============================================================================
# 棋盘镜像
# ============================================================================

@jax.jit
def mirror_board(board: jnp.ndarray) -> jnp.ndarray:
    """
    左右镜像棋盘
    
    Args:
        board: 棋盘状态 (10, 9) 或 (batch, 10, 9)
        
    Returns:
        镜像后的棋盘
    """
    return jnp.flip(board, axis=-1)


@jax.jit
def mirror_history(history: jnp.ndarray) -> jnp.ndarray:
    """
    镜像历史棋盘
    
    Args:
        history: 历史状态 (num_steps, 10, 9)
        
    Returns:
        镜像后的历史
    """
    return jnp.flip(history, axis=-1)


@jax.jit
def mirror_observation(observation: jnp.ndarray) -> jnp.ndarray:
    """
    镜像观察张量
    
    Args:
        observation: 观察张量 (channels, 10, 9)
        
    Returns:
        镜像后的观察
    """
    return jnp.flip(observation, axis=-1)


# ============================================================================
# 动作镜像
# ============================================================================

def _build_action_mirror_table() -> jnp.ndarray:
    """
    构建动作镜像映射表
    
    将每个动作映射到其镜像动作的索引
    """
    mirror_table = jnp.zeros(ACTION_SPACE_SIZE, dtype=jnp.int32)
    
    for action_id in range(ACTION_SPACE_SIZE):
        from_sq = int(_ACTION_TO_FROM_SQ[action_id])
        to_sq = int(_ACTION_TO_TO_SQ[action_id])
        
        # 镜像格子
        from_row, from_col = from_sq // BOARD_WIDTH, from_sq % BOARD_WIDTH
        to_row, to_col = to_sq // BOARD_WIDTH, to_sq % BOARD_WIDTH
        
        mirror_from_col = BOARD_WIDTH - 1 - from_col
        mirror_to_col = BOARD_WIDTH - 1 - to_col
        
        mirror_from_sq = from_row * BOARD_WIDTH + mirror_from_col
        mirror_to_sq = to_row * BOARD_WIDTH + mirror_to_col
        
        # 查找镜像动作的索引
        mirror_action_id = int(_FROM_TO_ACTION_TABLE[mirror_from_sq, mirror_to_sq])
        
        if mirror_action_id < 0:
            # 这不应该发生，因为镜像动作应该也是有效的
            mirror_action_id = action_id  # 回退到原动作
        
        mirror_table = mirror_table.at[action_id].set(mirror_action_id)
    
    return mirror_table


# 预计算镜像映射表
_ACTION_MIRROR_TABLE = _build_action_mirror_table()


@jax.jit
def mirror_action(action: jnp.ndarray) -> jnp.ndarray:
    """
    将动作索引镜像
    
    Args:
        action: 动作索引 (标量或数组)
        
    Returns:
        镜像后的动作索引
    """
    return _ACTION_MIRROR_TABLE[action]


@jax.jit
def mirror_policy(policy: jnp.ndarray) -> jnp.ndarray:
    """
    镜像策略分布
    
    Args:
        policy: 策略分布 (ACTION_SPACE_SIZE,) 或 (batch, ACTION_SPACE_SIZE)
        
    Returns:
        镜像后的策略分布
    """
    # 根据镜像映射重新排列策略
    return policy[..., _ACTION_MIRROR_TABLE]


@jax.jit
def mirror_legal_mask(mask: jnp.ndarray) -> jnp.ndarray:
    """
    镜像合法动作掩码
    
    Args:
        mask: 合法动作掩码 (ACTION_SPACE_SIZE,)
        
    Returns:
        镜像后的掩码
    """
    return mask[_ACTION_MIRROR_TABLE]


# ============================================================================
# 完整状态镜像
# ============================================================================

def mirror_state(state):
    """
    镜像完整的游戏状态
    
    Args:
        state: XiangqiState 对象
        
    Returns:
        镜像后的状态
        
    注意:
        position_hashes 在镜像后会失效（因为镜像局面的哈希不同），
        但这在训练数据增强中不重要，因为镜像是在轨迹收集完成后进行的。
    """
    from xiangqi.env import XiangqiState
    
    return XiangqiState(
        board=mirror_board(state.board),
        history=mirror_history(state.history),
        current_player=state.current_player,
        legal_action_mask=mirror_legal_mask(state.legal_action_mask),
        rewards=state.rewards,
        terminated=state.terminated,
        step_count=state.step_count,
        no_capture_count=state.no_capture_count,
        winner=state.winner,
        # 违规检测状态 (镜像后保持不变，但哈希已失效)
        position_hashes=state.position_hashes,
        hash_count=state.hash_count,
        red_consecutive_checks=state.red_consecutive_checks,
        black_consecutive_checks=state.black_consecutive_checks,
    )


# ============================================================================
# 数据增强函数
# ============================================================================

@partial(jax.jit, static_argnums=())
def augment_with_probability(
    key: jax.random.PRNGKey,
    observation: jnp.ndarray,
    action: jnp.ndarray,
    policy: jnp.ndarray,
    prob: float = 0.5
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.bool_]:
    """
    以一定概率进行镜像增强
    
    Args:
        key: 随机数密钥
        observation: 观察张量
        action: 动作索引
        policy: 策略分布
        prob: 镜像概率
        
    Returns:
        (augmented_obs, augmented_action, augmented_policy, is_mirrored)
    """
    do_mirror = jax.random.uniform(key) < prob
    
    aug_obs = jnp.where(do_mirror, mirror_observation(observation), observation)
    aug_action = jnp.where(do_mirror, mirror_action(action), action)
    aug_policy = jnp.where(do_mirror, mirror_policy(policy), policy)
    
    return aug_obs, aug_action, aug_policy, do_mirror


def augment_trajectory(
    key: jax.random.PRNGKey,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    policies: jnp.ndarray,
    values: jnp.ndarray,
    rewards: jnp.ndarray,
    to_plays: jnp.ndarray,
    prob: float = 0.5
) -> tuple:
    """
    对整个轨迹进行镜像增强
    
    Args:
        key: 随机数密钥
        observations: (T, C, H, W)
        actions: (T,)
        policies: (T, A)
        values: (T,)
        rewards: (T,)
        to_plays: (T,) 每步的当前玩家
        prob: 镜像概率
        
    Returns:
        增强后的轨迹数据 (obs, actions, policies, values, rewards, to_plays, is_mirrored)
    """
    do_mirror = jax.random.uniform(key) < prob
    
    if do_mirror:
        # 镜像整个轨迹
        aug_obs = jax.vmap(mirror_observation)(observations)
        aug_actions = jax.vmap(mirror_action)(actions)
        aug_policies = jax.vmap(mirror_policy)(policies)
        # to_plays 镜像时保持不变（玩家身份不变）
        return aug_obs, aug_actions, aug_policies, values, rewards, to_plays, True
    else:
        return observations, actions, policies, values, rewards, to_plays, False


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    import jax
    import jax.numpy as jnp
    from xiangqi.env import XiangqiEnv
    
    # 创建环境和初始状态
    env = XiangqiEnv()
    key = jax.random.PRNGKey(42)
    state = env.init(key)
    
    print("原始棋盘:")
    print(env.render(state))
    
    # 镜像状态
    mirrored_state = mirror_state(state)
    print("\n镜像后棋盘:")
    print(env.render(mirrored_state))
    
    # 测试动作镜像
    # 炮二平五: (2,1) -> (2,4)
    # 镜像后: (2,7) -> (2,4) 即炮八平五
    from xiangqi.actions import coords_to_square, move_to_action, action_to_move
    
    from_sq = coords_to_square(2, 1)
    to_sq = coords_to_square(2, 4)
    action = move_to_action(jnp.int32(from_sq), jnp.int32(to_sq))
    
    print(f"\n原始动作: from=({2},{1}), to=({2},{4}), action={action}")
    
    mirror_act = mirror_action(action)
    m_from, m_to = action_to_move(mirror_act)
    m_from_row, m_from_col = int(m_from) // 9, int(m_from) % 9
    m_to_row, m_to_col = int(m_to) // 9, int(m_to) % 9
    
    print(f"镜像动作: from=({m_from_row},{m_from_col}), to=({m_to_row},{m_to_col}), action={mirror_act}")
    
    # 测试策略镜像
    policy = jnp.zeros(ACTION_SPACE_SIZE)
    policy = policy.at[int(action)].set(1.0)
    
    mirrored_policy = mirror_policy(policy)
    print(f"\n原始策略在 action {action} 处为 1.0")
    print(f"镜像策略在 action {mirror_act} 处为 {mirrored_policy[mirror_act]}")

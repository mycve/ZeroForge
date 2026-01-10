"""
并行自我对弈模块

同时运行多局游戏，利用批量 MCTS 推理，充分利用多 GPU
"""

from __future__ import annotations
from typing import List, Tuple, Optional
from dataclasses import dataclass
import numpy as np
import jax
import jax.numpy as jnp
import logging

from training.replay_buffer import Trajectory

logger = logging.getLogger(__name__)


@dataclass
class ParallelGames:
    """并行游戏状态"""
    states: List  # 游戏状态列表
    observations: List[List]  # 每局的观察历史
    actions: List[List]  # 每局的动作历史
    policies: List[List]  # 每局的策略历史
    values: List[List]  # 每局的价值历史
    rewards: List[List]  # 每局的奖励历史
    to_plays: List[List]  # 每局的玩家历史
    step_counts: List[int]  # 每局的步数
    active: List[bool]  # 是否还在进行


class ParallelSelfPlay:
    """
    并行自我对弈
    
    同时运行多局游戏，批量调用 MCTS，充分利用 GPU
    """
    
    def __init__(
        self,
        env,
        mcts_runner,
        num_parallel: int = 64,
        temp_threshold: int = 30,
        temp_high: float = 1.0,
        temp_low: float = 0.25,
    ):
        """
        初始化并行自我对弈
        
        Args:
            env: 游戏环境
            mcts_runner: MCTS 运行器
            num_parallel: 并行游戏数量
            temp_threshold: 温度退火阈值
            temp_high: 高温度
            temp_low: 低温度
        """
        self.env = env
        self.mcts_runner = mcts_runner
        self.num_parallel = num_parallel
        self.temp_threshold = temp_threshold
        self.temp_high = temp_high
        self.temp_low = temp_low
        
    def run_batch(
        self,
        params: dict,
        rng_key: jax.random.PRNGKey,
    ) -> List[Trajectory]:
        """
        运行一批并行游戏
        
        Args:
            params: 网络参数
            rng_key: 随机数密钥
            
        Returns:
            完成的轨迹列表
        """
        from mcts.search import get_improved_policy, select_action
        
        # 初始化所有游戏
        rng_key, *init_keys = jax.random.split(rng_key, self.num_parallel + 1)
        init_keys = jnp.array(init_keys)
        
        games = ParallelGames(
            states=[self.env.init(init_keys[i]) for i in range(self.num_parallel)],
            observations=[[] for _ in range(self.num_parallel)],
            actions=[[] for _ in range(self.num_parallel)],
            policies=[[] for _ in range(self.num_parallel)],
            values=[[] for _ in range(self.num_parallel)],
            rewards=[[] for _ in range(self.num_parallel)],
            to_plays=[[] for _ in range(self.num_parallel)],
            step_counts=[0] * self.num_parallel,
            active=[True] * self.num_parallel,
        )
        
        completed_trajectories = []
        max_steps = 500  # 最大步数限制
        
        while any(games.active):
            # 收集活跃游戏的索引
            active_indices = [i for i, a in enumerate(games.active) if a]
            if not active_indices:
                break
                
            batch_size = len(active_indices)
            
            # 批量收集观察和合法动作
            observations_batch = []
            legal_masks_batch = []
            
            for idx in active_indices:
                state = games.states[idx]
                obs = self.env.observe(state)
                observations_batch.append(obs)
                legal_masks_batch.append(state.legal_action_mask)
                
                # 记录观察和玩家
                games.observations[idx].append(np.array(obs))
                games.to_plays[idx].append(int(state.current_player))
            
            # 转换为批量张量
            obs_batch = jnp.stack(observations_batch)
            legal_batch = jnp.stack(legal_masks_batch)
            
            # 批量 MCTS 搜索
            rng_key, search_key = jax.random.split(rng_key)
            policy_output = self.mcts_runner.run(
                params=params,
                observation=obs_batch,
                legal_action_mask=legal_batch,
                rng_key=search_key,
            )
            
            # 处理每局游戏
            for batch_idx, game_idx in enumerate(active_indices):
                step = games.step_counts[game_idx]
                temp = self.temp_high if step < self.temp_threshold else self.temp_low
                
                # 记录策略和价值
                policy = get_improved_policy(policy_output, self.temp_high)
                games.policies[game_idx].append(np.array(policy[batch_idx]))
                
                root_value = float(policy_output.search_tree.node_values[batch_idx, 0])
                games.values[game_idx].append(root_value)
                
                # 选择动作
                rng_key, action_key = jax.random.split(rng_key)
                
                # 从批量输出中提取单个游戏的结果
                single_action_weights = policy_output.action_weights[batch_idx:batch_idx+1]
                
                # 根据温度采样
                if temp < 1.0:
                    # 低温度：贪心选择
                    action = int(jnp.argmax(single_action_weights[0]))
                else:
                    # 高温度：按概率采样
                    action = int(jax.random.categorical(action_key, jnp.log(single_action_weights[0] + 1e-8)))
                
                games.actions[game_idx].append(action)
                games.step_counts[game_idx] += 1
                
                # 执行动作
                new_state = self.env.step(games.states[game_idx], jnp.int32(action))
                games.states[game_idx] = new_state
                
                # 记录奖励
                reward = float(new_state.rewards[int(games.to_plays[game_idx][-1])])
                games.rewards[game_idx].append(reward)
                
                # 检查游戏是否结束
                if new_state.terminated or games.step_counts[game_idx] >= max_steps:
                    games.active[game_idx] = False
                    
                    # 创建轨迹
                    trajectory = Trajectory(
                        observations=np.array(games.observations[game_idx]),
                        actions=np.array(games.actions[game_idx], dtype=np.int32),
                        rewards=np.array(games.rewards[game_idx], dtype=np.float32),
                        policies=np.array(games.policies[game_idx]),
                        values=np.array(games.values[game_idx], dtype=np.float32),
                        to_plays=np.array(games.to_plays[game_idx], dtype=np.int32),
                        game_result=int(new_state.winner),
                    )
                    completed_trajectories.append(trajectory)
        
        return completed_trajectories


def run_parallel_selfplay(
    env,
    mcts_runner,
    params: dict,
    rng_key: jax.random.PRNGKey,
    num_parallel: int = 64,
    temp_threshold: int = 30,
    temp_high: float = 1.0,
    temp_low: float = 0.25,
) -> List[Trajectory]:
    """
    运行并行自我对弈
    
    Args:
        env: 游戏环境
        mcts_runner: MCTS 运行器
        params: 网络参数
        rng_key: 随机数密钥
        num_parallel: 并行游戏数量
        temp_threshold: 温度退火阈值
        temp_high: 高温度
        temp_low: 低温度
        
    Returns:
        完成的轨迹列表
    """
    selfplay = ParallelSelfPlay(
        env=env,
        mcts_runner=mcts_runner,
        num_parallel=num_parallel,
        temp_threshold=temp_threshold,
        temp_high=temp_high,
        temp_low=temp_low,
    )
    return selfplay.run_batch(params, rng_key)

"""
对弈场模块
用于自动进行模型间对弈评估
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, Callable, NamedTuple
from dataclasses import dataclass
import logging
import jax
import jax.numpy as jnp
from functools import partial
import time

from xiangqi.env import XiangqiEnv, XiangqiState
from xiangqi.actions import ACTION_SPACE_SIZE
from mcts.search import MCTSConfig, run_mcts, select_action
from evaluation.elo import ELOTracker, MatchResult

logger = logging.getLogger(__name__)


# ============================================================================
# 玩家定义
# ============================================================================

@dataclass
class Player:
    """玩家定义"""
    name: str
    network_apply: Callable
    params: Dict
    mcts_config: MCTSConfig
    
    def get_action(
        self,
        state: XiangqiState,
        observation: jnp.ndarray,
        rng_key: jax.random.PRNGKey,
        greedy: bool = True,
    ) -> jnp.ndarray:
        """
        获取动作
        
        Args:
            state: 当前游戏状态
            observation: 观察张量
            rng_key: 随机数密钥
            greedy: 是否贪心选择
            
        Returns:
            选择的动作
        """
        # 添加批次维度
        obs_batch = observation[jnp.newaxis, ...]
        legal_mask_batch = state.legal_action_mask[jnp.newaxis, ...]
        
        # MCTS 搜索
        policy_output = run_mcts(
            observation=obs_batch,
            legal_action_mask=legal_mask_batch,
            network_apply=self.network_apply,
            params=self.params,
            config=self.mcts_config,
            rng_key=rng_key,
        )
        
        # 选择动作
        action = select_action(policy_output, greedy=greedy)
        
        return action[0]  # 移除批次维度


# ============================================================================
# 对局结果
# ============================================================================

class GameResult(NamedTuple):
    """单局游戏结果"""
    winner: int  # 0=红胜, 1=黑胜, -1=平局
    moves: int   # 总步数
    duration: float  # 游戏时长 (秒)
    move_history: List[int]  # 移动历史


# ============================================================================
# 对弈场
# ============================================================================

class Arena:
    """
    对弈场
    
    用于:
    - 两个模型之间的对弈评估
    - 新模型 vs 最佳模型
    - ELO 评分计算
    """
    
    def __init__(
        self,
        elo_tracker: Optional[ELOTracker] = None,
    ):
        """
        Args:
            elo_tracker: ELO 追踪器
        """
        self.env = XiangqiEnv()
        self.elo_tracker = elo_tracker or ELOTracker()
    
    def play_game(
        self,
        player_red: Player,
        player_black: Player,
        rng_key: jax.random.PRNGKey,
        verbose: bool = False,
        max_moves: int = 200,
    ) -> GameResult:
        """
        进行一局游戏
        
        Args:
            player_red: 红方玩家
            player_black: 黑方玩家
            rng_key: 随机数密钥
            verbose: 是否打印过程
            max_moves: 最大步数
            
        Returns:
            游戏结果
        """
        start_time = time.time()
        move_history = []
        
        # 初始化游戏
        rng_key, init_key = jax.random.split(rng_key)
        state = self.env.init(init_key)
        
        players = [player_red, player_black]
        
        while not state.terminated and state.step_count < max_moves:
            # 当前玩家
            current_player = players[int(state.current_player)]
            
            # 获取观察
            observation = self.env.observe(state)
            
            # 选择动作
            rng_key, action_key = jax.random.split(rng_key)
            action = current_player.get_action(state, observation, action_key, greedy=True)
            action = int(action)
            
            move_history.append(action)
            
            # 执行动作
            state = self.env.step(state, jnp.int32(action))
            
            if verbose:
                print(self.env.render(state))
                print()
        
        # 确定胜者
        if state.terminated:
            winner = int(state.winner)
        else:
            # 超时判为平局
            winner = -1
        
        duration = time.time() - start_time
        
        return GameResult(
            winner=winner,
            moves=int(state.step_count),
            duration=duration,
            move_history=move_history,
        )
    
    def play_match(
        self,
        player_a: Player,
        player_b: Player,
        num_games: int = 100,
        rng_key: jax.random.PRNGKey = None,
        update_elo: bool = True,
        verbose: bool = False,
    ) -> Dict:
        """
        进行一场比赛 (多局)
        
        Args:
            player_a: 玩家 A
            player_b: 玩家 B
            num_games: 对局数 (会自动平衡先后手)
            rng_key: 随机数密钥
            update_elo: 是否更新 ELO
            verbose: 是否打印详细信息
            
        Returns:
            比赛结果统计
        """
        if rng_key is None:
            rng_key = jax.random.PRNGKey(int(time.time()))
        
        # 确保对局数为偶数 (平衡先后手)
        num_games = (num_games // 2) * 2
        
        wins_a = 0
        wins_b = 0
        draws = 0
        total_moves = 0
        total_duration = 0.0
        
        for i in range(num_games):
            rng_key, game_key = jax.random.split(rng_key)
            
            # 交替先后手
            if i % 2 == 0:
                player_red, player_black = player_a, player_b
                a_is_red = True
            else:
                player_red, player_black = player_b, player_a
                a_is_red = False
            
            # 进行游戏
            result = self.play_game(player_red, player_black, game_key, verbose=verbose)
            
            total_moves += result.moves
            total_duration += result.duration
            
            # 统计结果
            if result.winner == -1:
                draws += 1
            elif (result.winner == 0) == a_is_red:
                wins_a += 1
            else:
                wins_b += 1
            
            if verbose or (i + 1) % 10 == 0:
                logger.info(
                    f"Game {i+1}/{num_games}: "
                    f"{player_a.name} {wins_a} - {draws} - {wins_b} {player_b.name}"
                )
        
        # 更新 ELO
        if update_elo:
            self.elo_tracker.update_batch(
                player_a.name,
                player_b.name,
                wins_a,
                draws,
                wins_b,
            )
        
        # 计算胜率
        win_rate_a = (wins_a + 0.5 * draws) / num_games
        
        return {
            "player_a": player_a.name,
            "player_b": player_b.name,
            "wins_a": wins_a,
            "draws": draws,
            "wins_b": wins_b,
            "win_rate_a": win_rate_a,
            "avg_moves": total_moves / num_games,
            "avg_duration": total_duration / num_games,
            "elo_a": self.elo_tracker.get_rating(player_a.name),
            "elo_b": self.elo_tracker.get_rating(player_b.name),
        }
    
    def evaluate_checkpoint(
        self,
        new_player: Player,
        best_player: Player,
        num_games: int = 100,
        win_threshold: float = 0.55,
        rng_key: jax.random.PRNGKey = None,
    ) -> Tuple[bool, Dict]:
        """
        评估新检查点是否优于当前最佳
        
        Args:
            new_player: 新模型
            best_player: 当前最佳模型
            num_games: 对局数
            win_threshold: 胜率阈值 (超过此值认为新模型更强)
            rng_key: 随机数密钥
            
        Returns:
            (is_better, match_result)
        """
        logger.info(f"评估: {new_player.name} vs {best_player.name} ({num_games} 局)")
        
        result = self.play_match(
            new_player,
            best_player,
            num_games=num_games,
            rng_key=rng_key,
            update_elo=True,
        )
        
        is_better = result["win_rate_a"] >= win_threshold
        
        logger.info(
            f"结果: {new_player.name} 胜率 {result['win_rate_a']:.2%} "
            f"({'通过' if is_better else '未通过'})"
        )
        
        return is_better, result
    
    def round_robin(
        self,
        players: List[Player],
        games_per_pair: int = 10,
        rng_key: jax.random.PRNGKey = None,
    ) -> Dict:
        """
        循环赛
        
        每对玩家进行指定数量的对局
        
        Args:
            players: 玩家列表
            games_per_pair: 每对玩家的对局数
            rng_key: 随机数密钥
            
        Returns:
            比赛结果
        """
        if rng_key is None:
            rng_key = jax.random.PRNGKey(int(time.time()))
        
        results = []
        
        for i, player_a in enumerate(players):
            for player_b in players[i+1:]:
                rng_key, match_key = jax.random.split(rng_key)
                
                result = self.play_match(
                    player_a,
                    player_b,
                    num_games=games_per_pair,
                    rng_key=match_key,
                )
                results.append(result)
        
        # 打印排名
        self.elo_tracker.print_rankings()
        
        return {
            "matches": results,
            "rankings": self.elo_tracker.get_rankings(),
        }


# ============================================================================
# 随机玩家 (用于测试)
# ============================================================================

class RandomPlayer:
    """随机玩家 (用于测试)"""
    
    def __init__(self, name: str = "Random"):
        self.name = name
    
    def get_action(
        self,
        state: XiangqiState,
        observation: jnp.ndarray,
        rng_key: jax.random.PRNGKey,
        greedy: bool = True,
    ) -> jnp.ndarray:
        """随机选择合法动作"""
        legal_actions = jnp.where(state.legal_action_mask)[0]
        idx = jax.random.randint(rng_key, (), 0, len(legal_actions))
        return legal_actions[idx]


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    import jax
    
    print("对弈场测试")
    print("=" * 50)
    
    # 创建对弈场
    arena = Arena()
    
    # 创建两个随机玩家
    player1 = RandomPlayer(name="Random_v1")
    player2 = RandomPlayer(name="Random_v2")
    
    # 进行一局测试
    key = jax.random.PRNGKey(42)
    
    print("进行 10 局随机对弈...")
    
    wins_1 = 0
    wins_2 = 0
    draws = 0
    
    for i in range(10):
        key, game_key = jax.random.split(key)
        
        # 交替先后手
        if i % 2 == 0:
            red, black = player1, player2
            p1_is_red = True
        else:
            red, black = player2, player1
            p1_is_red = False
        
        result = arena.play_game(red, black, game_key, verbose=False, max_moves=100)
        
        if result.winner == -1:
            draws += 1
            result_str = "平局"
        elif (result.winner == 0) == p1_is_red:
            wins_1 += 1
            result_str = f"{player1.name} 胜"
        else:
            wins_2 += 1
            result_str = f"{player2.name} 胜"
        
        print(f"Game {i+1}: {result_str} (步数: {result.moves})")
    
    print(f"\n最终结果: {player1.name} {wins_1} - {draws} - {wins_2} {player2.name}")
    
    # 更新 ELO
    arena.elo_tracker.update_batch(player1.name, player2.name, wins_1, draws, wins_2)
    arena.elo_tracker.print_rankings()
    
    print("\n测试通过!")

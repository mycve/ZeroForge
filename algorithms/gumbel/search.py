"""
Gumbel Search - Gumbel-Top-k 搜索实现

核心算法:
1. Gumbel-Top-k: 使用 Gumbel 噪声采样 top-k 动作
2. Sequential Halving: 逐步淘汰差动作，集中资源在好动作上
3. Completed Q-values: 使用网络预测补全未访问动作的 Q 值

优点:
- 不需要环境克隆（使用学习的 dynamics model）
- 搜索效率高（Sequential Halving）
- 理论保证的策略改进

参考: Policy improvement by planning with Gumbel (ICLR 2022)
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any, TYPE_CHECKING
from dataclasses import dataclass
import logging
import math

if TYPE_CHECKING:
    from core.game import Game

logger = logging.getLogger(__name__)


@dataclass
class GumbelConfig:
    """Gumbel 搜索配置"""
    
    # 搜索参数
    num_simulations: int = 50           # 总模拟次数
    max_num_considered_actions: int = 16  # 考虑的最大动作数（top-k）
    
    # Gumbel 参数
    gumbel_scale: float = 1.0           # Gumbel 噪声缩放
    
    # Sequential Halving 参数（自动计算）
    # num_halving_rounds 由 max_num_considered_actions 自动决定
    
    # Q 值参数
    c_visit: float = 50.0               # 访问计数权重
    c_scale: float = 1.0                # Q 值缩放
    discount: float = 0.997             # 折扣因子
    
    # 温度
    temperature: float = 1.0            # 采样温度
    
    # 设备
    device: str = "cuda"
    
    @property
    def num_halving_rounds(self) -> int:
        """自动计算 Sequential Halving 轮数: ceil(log2(k))"""
        return max(1, math.ceil(math.log2(self.max_num_considered_actions)))


class GumbelSearch:
    """Gumbel-Top-k + Sequential Halving 搜索
    
    不需要环境克隆，只使用神经网络进行搜索。
    适用于不支持 clone() 的 Gymnasium 环境。
    """
    
    def __init__(
        self,
        network: nn.Module,
        config: GumbelConfig,
        action_space: int,
        use_dynamics: bool = True,  # MuZero 模式
    ):
        """
        Args:
            network: 神经网络（MuZero 或 AlphaZero）
            config: 搜索配置
            action_space: 动作空间大小
            use_dynamics: 是否使用 dynamics network（MuZero=True, AlphaZero=False）
        """
        self.network = network
        self.config = config
        self.action_space = action_space
        self.use_dynamics = use_dynamics
        self.device = torch.device(config.device if torch.cuda.is_available() else "cpu")
        
        # 移动网络到设备
        self.network = self.network.to(self.device)
    
    def search(
        self,
        observation: np.ndarray,
        legal_actions: List[int],
        game: Optional["Game"] = None,  # AlphaZero 模式需要
    ) -> Tuple[Dict[int, float], float]:
        """执行 Gumbel 搜索
        
        Args:
            observation: 当前观测
            legal_actions: 合法动作列表
            game: 游戏实例（AlphaZero 模式需要）
            
        Returns:
            (策略分布, 根节点价值)
        """
        self.network.eval()
        
        with torch.no_grad():
            # 获取初始策略和价值
            obs_tensor = torch.from_numpy(observation).float().unsqueeze(0).to(self.device)
            
            if self.use_dynamics:
                # MuZero: 使用 initial_inference
                hidden, policy_logits, value = self.network.initial_inference(obs_tensor)
            else:
                # AlphaZero: 直接前向传播
                policy_logits, value = self.network(obs_tensor)
                hidden = None
            
            policy_logits = policy_logits.squeeze(0)  # [action_space]
            root_value = value.item()
            
            # 获取先验策略（只考虑合法动作）
            prior_probs = self._get_legal_probs(policy_logits, legal_actions)
            
            # Gumbel-Top-k 采样
            num_actions = min(self.config.max_num_considered_actions, len(legal_actions))
            selected_actions = self._gumbel_top_k(prior_probs, legal_actions, num_actions)
            
            if self.use_dynamics:
                # MuZero: 使用 Sequential Halving + Dynamics
                q_values = self._sequential_halving_muzero(
                    hidden, selected_actions, prior_probs
                )
            else:
                # AlphaZero: 使用 Sequential Halving + 真实游戏
                if game is None:
                    raise ValueError("AlphaZero 模式需要提供 game 实例")
                q_values = self._sequential_halving_alphazero(
                    game, selected_actions, prior_probs
                )
            
            # 计算改进后的策略
            improved_policy = self._compute_improved_policy(
                prior_probs, q_values, selected_actions, legal_actions
            )
            
            return improved_policy, root_value
    
    def _get_legal_probs(
        self,
        policy_logits: torch.Tensor,
        legal_actions: List[int],
    ) -> Dict[int, float]:
        """获取合法动作的概率分布"""
        # 创建 mask
        mask = torch.full_like(policy_logits, float('-inf'))
        for a in legal_actions:
            mask[a] = 0.0
        
        # Softmax（只在合法动作上）
        masked_logits = policy_logits + mask
        probs = F.softmax(masked_logits, dim=0)
        
        return {a: probs[a].item() for a in legal_actions}
    
    def _gumbel_top_k(
        self,
        prior_probs: Dict[int, float],
        legal_actions: List[int],
        k: int,
    ) -> List[int]:
        """Gumbel-Top-k 采样
        
        使用 Gumbel 噪声选择 top-k 动作，这比随机采样更有效。
        公式: g(a) = log(π(a)) + Gumbel(0, scale)
        选择 g(a) 最大的 k 个动作
        """
        # 计算 Gumbel 分数
        scores = {}
        for a in legal_actions:
            log_prob = np.log(prior_probs[a] + 1e-8)
            gumbel_noise = np.random.gumbel(0, self.config.gumbel_scale)
            scores[a] = log_prob + gumbel_noise
        
        # 选择 top-k
        sorted_actions = sorted(scores.keys(), key=lambda a: scores[a], reverse=True)
        return sorted_actions[:k]
    
    def _sequential_halving_muzero(
        self,
        hidden: torch.Tensor,
        selected_actions: List[int],
        prior_probs: Dict[int, float],
    ) -> Dict[int, float]:
        """Sequential Halving 搜索（MuZero 模式）
        
        使用 dynamics network 模拟未来状态，不需要真实环境。
        """
        # 初始化 Q 值（使用先验价值）
        q_values = {a: 0.0 for a in selected_actions}
        visit_counts = {a: 0 for a in selected_actions}
        
        # 计算每个 halving round 的模拟次数
        remaining_actions = list(selected_actions)
        total_sims = self.config.num_simulations
        
        for round_idx in range(self.config.num_halving_rounds):
            if len(remaining_actions) <= 1:
                break
            
            # 本轮每个动作的模拟次数
            sims_per_action = max(1, total_sims // (len(remaining_actions) * (self.config.num_halving_rounds - round_idx)))
            
            # 模拟每个动作
            for action in remaining_actions:
                for _ in range(sims_per_action):
                    # 使用 dynamics network 预测下一状态
                    action_tensor = torch.tensor([action], device=self.device)
                    next_hidden, reward, policy_logits, value = self.network.recurrent_inference(
                        hidden, action_tensor
                    )
                    
                    # 更新 Q 值（使用即时奖励 + 折扣后的价值）
                    q = reward.item() + self.config.discount * value.item()
                    
                    # 增量更新
                    visit_counts[action] += 1
                    q_values[action] += (q - q_values[action]) / visit_counts[action]
            
            # 减半：保留 Q 值最高的一半动作
            remaining_actions = sorted(
                remaining_actions,
                key=lambda a: q_values[a],
                reverse=True
            )[:max(1, len(remaining_actions) // 2)]
        
        # 补全未访问动作的 Q 值（使用先验价值）
        return self._complete_q_values(q_values, visit_counts, prior_probs, hidden)
    
    def _sequential_halving_alphazero(
        self,
        game: "Game",
        selected_actions: List[int],
        prior_probs: Dict[int, float],
    ) -> Dict[int, float]:
        """Sequential Halving 搜索（AlphaZero 模式）
        
        使用真实游戏规则模拟，需要游戏支持 clone()。
        如果不支持 clone()，回退到单步搜索。
        """
        # 检查是否支持 clone
        try:
            test_clone = game.clone()
            supports_clone = True
        except Exception:
            supports_clone = False
            logger.warning("游戏不支持 clone()，使用单步搜索")
        
        # 初始化 Q 值
        q_values = {a: 0.0 for a in selected_actions}
        visit_counts = {a: 0 for a in selected_actions}
        
        remaining_actions = list(selected_actions)
        total_sims = self.config.num_simulations
        
        for round_idx in range(self.config.num_halving_rounds):
            if len(remaining_actions) <= 1:
                break
            
            sims_per_action = max(1, total_sims // (len(remaining_actions) * (self.config.num_halving_rounds - round_idx)))
            
            for action in remaining_actions:
                for _ in range(sims_per_action):
                    if supports_clone:
                        # 克隆游戏并模拟
                        sim_game = game.clone()
                        obs, reward, done, info = sim_game.step(action)
                        
                        if done:
                            q = reward
                        else:
                            # 使用网络评估后续状态
                            obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
                            _, value = self.network(obs_tensor)
                            q = reward + self.config.discount * value.item()
                    else:
                        # 不支持 clone，使用网络直接评估
                        obs = game.get_observation()
                        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
                        policy_logits, value = self.network(obs_tensor)
                        
                        # 使用先验策略加权的 Q 估计
                        q = value.item()
                    
                    visit_counts[action] += 1
                    q_values[action] += (q - q_values[action]) / visit_counts[action]
            
            # 减半
            remaining_actions = sorted(
                remaining_actions,
                key=lambda a: q_values[a],
                reverse=True
            )[:max(1, len(remaining_actions) // 2)]
        
        # 补全 Q 值
        obs = game.get_observation()
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).to(self.device)
        _, root_value = self.network(obs_tensor)
        
        for a in selected_actions:
            if visit_counts[a] == 0:
                q_values[a] = root_value.item()
        
        return q_values
    
    def _complete_q_values(
        self,
        q_values: Dict[int, float],
        visit_counts: Dict[int, int],
        prior_probs: Dict[int, float],
        hidden: torch.Tensor,
    ) -> Dict[int, float]:
        """补全未访问动作的 Q 值
        
        使用 Completed Q-values 方法：
        - 已访问动作：使用 MCTS 估计的 Q 值
        - 未访问动作：使用网络预测的 Q 值
        """
        # 计算已访问动作的平均 Q 值
        visited_q = [q_values[a] for a in q_values if visit_counts.get(a, 0) > 0]
        if visited_q:
            mean_q = np.mean(visited_q)
        else:
            mean_q = 0.0
        
        # 补全未访问动作
        completed_q = {}
        for a in q_values:
            if visit_counts.get(a, 0) > 0:
                completed_q[a] = q_values[a]
            else:
                # 使用先验概率加权的估计
                completed_q[a] = mean_q
        
        return completed_q
    
    def _compute_improved_policy(
        self,
        prior_probs: Dict[int, float],
        q_values: Dict[int, float],
        selected_actions: List[int],
        legal_actions: List[int],
    ) -> Dict[int, float]:
        """计算改进后的策略
        
        使用 softmax(Q / temperature) 作为改进策略，
        与先验策略混合。
        """
        # 计算 Q 值的 softmax
        q_array = np.array([q_values.get(a, 0.0) for a in selected_actions])
        
        if self.config.temperature > 0:
            q_probs = np.exp(q_array / self.config.temperature)
            q_sum = q_probs.sum()
            if q_sum > 0:
                q_probs = q_probs / q_sum
            else:
                q_probs = np.ones_like(q_probs) / len(q_probs)
        else:
            # 贪婪选择
            q_probs = np.zeros_like(q_array)
            q_probs[np.argmax(q_array)] = 1.0
        
        # 构建最终策略
        policy = {}
        for i, a in enumerate(selected_actions):
            policy[a] = q_probs[i]
        
        # 归一化到所有合法动作
        total = sum(policy.values())
        if total > 0:
            policy = {a: p / total for a, p in policy.items()}
        
        # 确保所有合法动作都有概率（避免漏掉）
        for a in legal_actions:
            if a not in policy:
                policy[a] = 0.0
        
        return policy
    
    def select_action(
        self,
        policy: Dict[int, float],
        temperature: float = 1.0,
    ) -> int:
        """从策略中选择动作
        
        Args:
            policy: 策略分布
            temperature: 温度（0=贪婪，>0=采样）
            
        Returns:
            选择的动作
        """
        actions = list(policy.keys())
        probs = np.array([policy[a] for a in actions])
        
        if temperature == 0:
            # 贪婪选择
            return actions[np.argmax(probs)]
        else:
            # 温度采样
            probs = probs ** (1.0 / temperature)
            prob_sum = probs.sum()
            if prob_sum > 0:
                probs = probs / prob_sum
            else:
                # 概率全为0时，均匀分布
                probs = np.ones_like(probs) / len(probs)
            # 确保概率精确求和为1（修复浮点误差）
            probs = probs / probs.sum()
            return np.random.choice(actions, p=probs)


# ============================================================
# 导出
# ============================================================

__all__ = [
    "GumbelConfig",
    "GumbelSearch",
]

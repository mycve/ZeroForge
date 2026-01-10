"""
ELO 评分系统
用于跟踪和比较不同检查点的强度
"""

from __future__ import annotations
from typing import Dict, List, Tuple, Optional, NamedTuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import math
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


# ============================================================================
# ELO 评分
# ============================================================================

@dataclass
class ELORating:
    """单个玩家的 ELO 评分"""
    rating: float = 1500.0
    games_played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    peak_rating: float = 1500.0
    last_updated: str = ""
    
    @property
    def win_rate(self) -> float:
        """胜率"""
        if self.games_played == 0:
            return 0.0
        return (self.wins + 0.5 * self.draws) / self.games_played
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "rating": self.rating,
            "games_played": self.games_played,
            "wins": self.wins,
            "draws": self.draws,
            "losses": self.losses,
            "peak_rating": self.peak_rating,
            "last_updated": self.last_updated,
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'ELORating':
        """从字典恢复"""
        return cls(**d)


class MatchResult(NamedTuple):
    """对局结果"""
    player_a: str  # 玩家 A 标识
    player_b: str  # 玩家 B 标识
    result: float  # 结果: 1.0=A胜, 0.5=平, 0.0=B胜
    timestamp: str = ""


# ============================================================================
# ELO 追踪器
# ============================================================================

class ELOTracker:
    """
    ELO 评分追踪器
    
    使用标准 ELO 算法:
    - 预期得分: E = 1 / (1 + 10^((R_opponent - R_player) / 400))
    - 新评分: R_new = R_old + K * (S - E)
    
    其中 K 因子随对局数减小，新玩家 K 值较大以快速定位
    """
    
    def __init__(
        self,
        k_factor: float = 32.0,
        k_factor_min: float = 16.0,
        k_decay_games: int = 30,
        initial_rating: float = 1500.0,
        save_path: Optional[str] = None,
    ):
        """
        Args:
            k_factor: 初始 K 因子
            k_factor_min: 最小 K 因子
            k_decay_games: K 因子衰减到最小值所需的对局数
            initial_rating: 初始评分
            save_path: 保存路径
        """
        self.k_factor = k_factor
        self.k_factor_min = k_factor_min
        self.k_decay_games = k_decay_games
        self.initial_rating = initial_rating
        self.save_path = Path(save_path) if save_path else None
        
        # 评分字典
        self.ratings: Dict[str, ELORating] = {}
        
        # 对局历史
        self.match_history: List[MatchResult] = []
        
        # 尝试加载
        if self.save_path and self.save_path.exists():
            self.load()
    
    def get_k_factor(self, player_id: str) -> float:
        """获取玩家的 K 因子"""
        if player_id not in self.ratings:
            return self.k_factor
        
        games = self.ratings[player_id].games_played
        if games >= self.k_decay_games:
            return self.k_factor_min
        
        # 线性衰减
        return self.k_factor - (self.k_factor - self.k_factor_min) * games / self.k_decay_games
    
    def get_expected_score(self, rating_a: float, rating_b: float) -> float:
        """计算玩家 A 的预期得分"""
        return 1.0 / (1.0 + math.pow(10.0, (rating_b - rating_a) / 400.0))
    
    def update(
        self,
        player_a: str,
        player_b: str,
        result: float,
    ) -> Tuple[float, float]:
        """
        更新 ELO 评分
        
        Args:
            player_a: 玩家 A 标识
            player_b: 玩家 B 标识
            result: 结果 (1.0=A胜, 0.5=平, 0.0=B胜)
            
        Returns:
            (A 的新评分, B 的新评分)
        """
        # 确保玩家存在
        for player_id in [player_a, player_b]:
            if player_id not in self.ratings:
                self.ratings[player_id] = ELORating(rating=self.initial_rating)
        
        rating_a = self.ratings[player_a].rating
        rating_b = self.ratings[player_b].rating
        
        # 预期得分
        expected_a = self.get_expected_score(rating_a, rating_b)
        expected_b = 1.0 - expected_a
        
        # K 因子
        k_a = self.get_k_factor(player_a)
        k_b = self.get_k_factor(player_b)
        
        # 新评分
        new_rating_a = rating_a + k_a * (result - expected_a)
        new_rating_b = rating_b + k_b * ((1.0 - result) - expected_b)
        
        # 更新 A
        self.ratings[player_a].rating = new_rating_a
        self.ratings[player_a].games_played += 1
        self.ratings[player_a].peak_rating = max(
            self.ratings[player_a].peak_rating,
            new_rating_a
        )
        self.ratings[player_a].last_updated = datetime.now().isoformat()
        
        if result == 1.0:
            self.ratings[player_a].wins += 1
        elif result == 0.5:
            self.ratings[player_a].draws += 1
        else:
            self.ratings[player_a].losses += 1
        
        # 更新 B
        self.ratings[player_b].rating = new_rating_b
        self.ratings[player_b].games_played += 1
        self.ratings[player_b].peak_rating = max(
            self.ratings[player_b].peak_rating,
            new_rating_b
        )
        self.ratings[player_b].last_updated = datetime.now().isoformat()
        
        if result == 0.0:
            self.ratings[player_b].wins += 1
        elif result == 0.5:
            self.ratings[player_b].draws += 1
        else:
            self.ratings[player_b].losses += 1
        
        # 记录对局
        match = MatchResult(
            player_a=player_a,
            player_b=player_b,
            result=result,
            timestamp=datetime.now().isoformat(),
        )
        self.match_history.append(match)
        
        # 自动保存
        if self.save_path:
            self.save()
        
        return new_rating_a, new_rating_b
    
    def update_batch(
        self,
        player_a: str,
        player_b: str,
        wins_a: int,
        draws: int,
        wins_b: int,
    ) -> Tuple[float, float]:
        """
        批量更新评分（多局比赛）
        
        Args:
            player_a: 玩家 A
            player_b: 玩家 B
            wins_a: A 获胜局数
            draws: 平局数
            wins_b: B 获胜局数
            
        Returns:
            (A 的新评分, B 的新评分)
        """
        rating_a, rating_b = None, None
        
        # A 获胜的局
        for _ in range(wins_a):
            rating_a, rating_b = self.update(player_a, player_b, 1.0)
        
        # 平局
        for _ in range(draws):
            rating_a, rating_b = self.update(player_a, player_b, 0.5)
        
        # B 获胜的局
        for _ in range(wins_b):
            rating_a, rating_b = self.update(player_a, player_b, 0.0)
        
        return rating_a or self.initial_rating, rating_b or self.initial_rating
    
    def get_rating(self, player_id: str) -> float:
        """获取玩家评分"""
        if player_id not in self.ratings:
            return self.initial_rating
        return self.ratings[player_id].rating
    
    def get_player_info(self, player_id: str) -> ELORating:
        """获取玩家详细信息"""
        if player_id not in self.ratings:
            return ELORating(rating=self.initial_rating)
        return self.ratings[player_id]
    
    def get_rankings(self) -> List[Tuple[str, float]]:
        """获取排名列表"""
        rankings = [
            (player_id, rating.rating)
            for player_id, rating in self.ratings.items()
        ]
        rankings.sort(key=lambda x: x[1], reverse=True)
        return rankings
    
    def get_best_player(self) -> Tuple[str, float]:
        """获取最强玩家"""
        rankings = self.get_rankings()
        if not rankings:
            return ("", self.initial_rating)
        return rankings[0]
    
    def save(self):
        """保存到文件"""
        if self.save_path is None:
            return
        
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            "ratings": {k: v.to_dict() for k, v in self.ratings.items()},
            "match_history": [
                {"player_a": m.player_a, "player_b": m.player_b, 
                 "result": m.result, "timestamp": m.timestamp}
                for m in self.match_history[-1000:]  # 只保留最近 1000 局
            ],
            "config": {
                "k_factor": self.k_factor,
                "k_factor_min": self.k_factor_min,
                "k_decay_games": self.k_decay_games,
                "initial_rating": self.initial_rating,
            },
        }
        
        with open(self.save_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def load(self):
        """从文件加载"""
        if self.save_path is None or not self.save_path.exists():
            return
        
        with open(self.save_path, "r") as f:
            data = json.load(f)
        
        self.ratings = {
            k: ELORating.from_dict(v) for k, v in data.get("ratings", {}).items()
        }
        
        self.match_history = [
            MatchResult(**m) for m in data.get("match_history", [])
        ]
        
        logger.info(f"加载了 {len(self.ratings)} 个玩家的评分数据")
    
    def print_rankings(self, top_n: int = 10):
        """打印排名"""
        rankings = self.get_rankings()[:top_n]
        
        print("\n" + "=" * 60)
        print(f"{'排名':<6}{'玩家':<20}{'ELO':<10}{'对局':<8}{'胜率':<10}")
        print("=" * 60)
        
        for i, (player_id, rating) in enumerate(rankings, 1):
            info = self.ratings[player_id]
            print(
                f"{i:<6}{player_id:<20}{rating:<10.1f}"
                f"{info.games_played:<8}{info.win_rate:<10.2%}"
            )
        
        print("=" * 60)


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    import tempfile
    
    print("ELO 评分系统测试")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = Path(tmpdir) / "elo.json"
        
        # 创建追踪器
        tracker = ELOTracker(save_path=str(save_path))
        
        # 模拟对局
        # 玩家 A 连续击败 B
        for _ in range(5):
            tracker.update("model_v1", "model_v0", 1.0)
        
        # 玩家 C 击败 A
        for _ in range(3):
            tracker.update("model_v2", "model_v1", 1.0)
        
        # 一些平局
        tracker.update("model_v2", "model_v0", 0.5)
        tracker.update("model_v1", "model_v0", 0.5)
        
        # 打印排名
        tracker.print_rankings()
        
        # 测试保存和加载
        tracker.save()
        
        new_tracker = ELOTracker(save_path=str(save_path))
        print(f"\n加载后的排名:")
        new_tracker.print_rankings()
        
        # 预期得分测试
        elo_diff = 200
        expected = tracker.get_expected_score(1600, 1400)
        print(f"\nELO 差距 {elo_diff}: 预期得分 = {expected:.3f}")
        
        print("\n测试通过!")

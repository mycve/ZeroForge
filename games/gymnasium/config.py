"""
Gymnasium 游戏配置

定义 Gymnasium 环境的配置参数。
"""

from dataclasses import dataclass, field
from typing import Optional, Tuple, List

from core.config import GameConfig


# ============================================================
# Gymnasium 环境配置
# ============================================================

@dataclass
class GymnasiumConfig(GameConfig):
    """Gymnasium 环境配置
    
    Attributes:
        env_id: Gymnasium 环境 ID，如 "ALE/Breakout-v5"
        frame_stack: 帧堆叠数量（用于提供时序信息）
        frame_skip: 跳帧数量（动作重复执行次数）
        grayscale: 是否转换为灰度图
        resize: 图像缩放大小，None 表示不缩放
        normalize_obs: 是否归一化观测到 [0, 1]
        clip_rewards: 是否裁剪奖励到 [-1, 1]
        terminal_on_life_loss: 生命丢失时是否视为终止（Atari）
        discrete_bins: 连续动作空间离散化 bins 数量
        render_mode: 渲染模式（"rgb_array" 或 None）
    """
    
    # 环境标识
    env_id: str = "CartPole-v1"
    
    # 观测预处理
    frame_stack: int = 1              # 默认不堆叠
    frame_skip: int = 1               # 默认不跳帧
    grayscale: bool = False           # 默认保持彩色
    resize: Optional[Tuple[int, int]] = None  # 默认不缩放
    normalize_obs: bool = True        # 归一化观测
    
    # 奖励处理
    clip_rewards: bool = False        # 默认不裁剪
    reward_scale: float = 1.0         # 奖励缩放因子
    
    # Atari 特定
    terminal_on_life_loss: bool = False  # 生命丢失是否终止
    noop_max: int = 0                    # 开始时随机 NOOP 动作数
    
    # 动作空间
    discrete_bins: int = 11           # 连续动作离散化 bins
    
    # 渲染
    render_mode: Optional[str] = "rgb_array"
    
    # 覆盖基类
    max_game_length: int = 10000      # 默认最大步数
    history_steps: int = 1            # 历史步数
    
    def __post_init__(self):
        """后处理配置"""
        # Atari 游戏默认配置
        if self.env_id.startswith("ALE/") or "Atari" in self.env_id:
            if self.frame_stack == 1:
                self.frame_stack = 4      # Atari 默认 4 帧堆叠
            if self.frame_skip == 1:
                self.frame_skip = 4       # Atari 默认跳 4 帧
            if not self.grayscale:
                self.grayscale = True     # Atari 默认灰度
            if self.resize is None:
                self.resize = (84, 84)    # Atari 默认 84x84
            if not self.clip_rewards:
                self.clip_rewards = True  # Atari 默认裁剪奖励
    
    @property
    def observation_shape(self) -> Tuple[int, ...]:
        """观测张量形状"""
        if self.resize:
            h, w = self.resize
        else:
            # 默认形状（实际由环境决定）
            h, w = 84, 84
        
        if self.grayscale:
            return (self.frame_stack, h, w)
        else:
            return (self.frame_stack, h, w, 3)
    
    def validate(self) -> None:
        """验证配置合法性"""
        super().validate()
        
        if self.frame_stack < 1:
            raise ValueError(f"frame_stack 必须 >= 1，当前: {self.frame_stack}")
        if self.frame_skip < 1:
            raise ValueError(f"frame_skip 必须 >= 1，当前: {self.frame_skip}")
        if self.discrete_bins < 2:
            raise ValueError(f"discrete_bins 必须 >= 2，当前: {self.discrete_bins}")


@dataclass
class AtariConfig(GymnasiumConfig):
    """Atari 游戏专用配置
    
    预设了 Atari 游戏的推荐参数。
    """
    
    # Atari 默认配置
    frame_stack: int = 4
    frame_skip: int = 4
    grayscale: bool = True
    resize: Optional[Tuple[int, int]] = field(default_factory=lambda: (84, 84))
    normalize_obs: bool = True
    clip_rewards: bool = True
    terminal_on_life_loss: bool = True
    noop_max: int = 30
    max_game_length: int = 108000  # 30 分钟 @ 60fps / 4 skip


@dataclass
class ClassicControlConfig(GymnasiumConfig):
    """经典控制环境配置
    
    用于 CartPole、LunarLander 等简单环境。
    """
    
    # 经典控制默认配置
    frame_stack: int = 1
    frame_skip: int = 1
    grayscale: bool = False
    resize: Optional[Tuple[int, int]] = None
    normalize_obs: bool = True
    clip_rewards: bool = False
    max_game_length: int = 1000


@dataclass
class MuJoCoConfig(GymnasiumConfig):
    """MuJoCo 物理仿真配置
    
    用于 Ant、HalfCheetah 等连续控制环境。
    """
    
    # MuJoCo 默认配置
    frame_stack: int = 1
    frame_skip: int = 1
    grayscale: bool = False
    resize: Optional[Tuple[int, int]] = None
    normalize_obs: bool = True
    clip_rewards: bool = False
    discrete_bins: int = 11  # 连续动作离散化
    max_game_length: int = 1000


# ============================================================
# 预设配置
# ============================================================

# Atari 游戏列表
ATARI_GAMES: List[str] = [
    "ALE/Breakout-v5",
    "ALE/Pong-v5",
    "ALE/SpaceInvaders-v5",
    "ALE/Qbert-v5",
    "ALE/Seaquest-v5",
    "ALE/BeamRider-v5",
    "ALE/Enduro-v5",
    "ALE/Asteroids-v5",
    "ALE/MsPacman-v5",
    "ALE/Freeway-v5",
]

# 经典控制游戏列表
CLASSIC_CONTROL_GAMES: List[str] = [
    "CartPole-v1",
    "LunarLander-v3",
    "MountainCar-v0",
    "Acrobot-v1",
    "Pendulum-v1",
]

# MuJoCo 游戏列表
MUJOCO_GAMES: List[str] = [
    "Ant-v5",
    "HalfCheetah-v5",
    "Hopper-v5",
    "Walker2d-v5",
    "Humanoid-v5",
    "Swimmer-v5",
    "Reacher-v5",
    "InvertedPendulum-v5",
]


# ============================================================
# 导出
# ============================================================

__all__ = [
    "GymnasiumConfig",
    "AtariConfig",
    "ClassicControlConfig",
    "MuJoCoConfig",
    "ATARI_GAMES",
    "CLASSIC_CONTROL_GAMES",
    "MUJOCO_GAMES",
]

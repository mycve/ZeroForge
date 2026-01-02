"""
Game ABC - 游戏抽象基类

定义通用游戏接口，支持:
- 单人游戏 (Atari, 2048, 推箱子等)
- 双人对弈游戏 (中国象棋, 围棋, 国际象棋等)
- 多人游戏 (扑克, 麻将等)

设计原则:
1. 最小化接口：只定义必要的抽象方法
2. 零拷贝友好：观测返回 numpy 数组，避免不必要的转换
3. 可扩展：通过 info dict 返回游戏特定信息
4. 统一配置：通过 config_class 关联配置类

开发者使用方式:
    >>> from core.game import Game
    >>> from core.config import GameConfig
    >>> from dataclasses import dataclass
    >>> 
    >>> @dataclass
    ... class MyGameConfig(GameConfig):
    ...     board_size: int = 8
    >>> 
    >>> class MyGame(Game):
    ...     config_class = MyGameConfig
    ...     
    ...     def __init__(self, config: MyGameConfig = None):
    ...         self.config = config or MyGameConfig()
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Tuple, Optional, Any, Dict, TypeVar, Generic, Type, TYPE_CHECKING
import numpy as np
import logging

if TYPE_CHECKING:
    from .config import GameConfig

logger = logging.getLogger(__name__)


# ============================================================
# 空间定义
# ============================================================

@dataclass(frozen=True)
class ObservationSpace:
    """观测空间定义
    
    Attributes:
        shape: 观测张量形状，例如 (58, 10, 9) 表示中国象棋
        dtype: 数据类型，默认 float32
        low: 最小值（可选，用于归一化）
        high: 最大值（可选，用于归一化）
    
    Examples:
        >>> obs_space = ObservationSpace(shape=(58, 10, 9))
        >>> obs_space.num_elements
        5220
    """
    shape: Tuple[int, ...]
    dtype: np.dtype = field(default_factory=lambda: np.float32)
    low: Optional[float] = None
    high: Optional[float] = None
    
    @property
    def num_elements(self) -> int:
        """观测张量元素总数"""
        result = 1
        for dim in self.shape:
            result *= dim
        return result
    
    def sample(self) -> np.ndarray:
        """随机采样一个观测（用于测试）"""
        if self.low is not None and self.high is not None:
            return np.random.uniform(self.low, self.high, self.shape).astype(self.dtype)
        return np.random.randn(*self.shape).astype(self.dtype)


@dataclass(frozen=True)
class ActionSpace:
    """动作空间定义（离散动作）
    
    Attributes:
        n: 动作总数
        action_names: 动作名称映射（可选，用于可视化）
    
    Examples:
        >>> action_space = ActionSpace(n=2086)
        >>> action_space.sample()
        1234
    """
    n: int
    action_names: Optional[Dict[int, str]] = None
    
    def sample(self) -> int:
        """随机采样一个动作"""
        return np.random.randint(0, self.n)
    
    def contains(self, action: int) -> bool:
        """检查动作是否在有效范围内"""
        return 0 <= action < self.n


# ============================================================
# 玩家类型
# ============================================================

class PlayerType(Enum):
    """玩家类型枚举
    
    用于标识游戏的玩家模式，影响训练策略和 UI 展示。
    """
    SINGLE = "single"           # 单人游戏（如 2048、Atari）
    TWO_PLAYER = "two_player"   # 双人对弈（如象棋、围棋）
    MULTI_PLAYER = "multi"      # 多人游戏（如扑克、麻将）
    COOPERATIVE = "coop"        # 合作模式（多人合作）


# ============================================================
# 游戏元数据
# ============================================================

@dataclass
class GameMeta:
    """游戏元数据
    
    用于游戏注册、展示和自动扫描。
    开发者必须为每个游戏实现 get_meta() 方法返回此对象。
    
    Attributes:
        name: 显示名称（如 "井字棋"）
        description: 游戏描述
        version: 版本号
        author: 作者
        tags: 标签列表（如 ["board", "strategy"]）
        difficulty: 难度（easy/medium/hard）
        min_players: 最小玩家数
        max_players: 最大玩家数
    """
    name: str
    description: str = ""
    version: str = "1.0.0"
    author: str = ""
    tags: List[str] = field(default_factory=list)
    difficulty: str = "medium"  # easy / medium / hard
    min_players: int = 1
    max_players: int = 2


# ============================================================
# 游戏状态
# ============================================================

@dataclass
class GameState:
    """游戏状态快照
    
    用于保存和恢复游戏状态，支持 MCTS 树搜索中的状态克隆。
    
    Attributes:
        observation: 当前观测
        legal_actions: 合法动作列表
        current_player: 当前玩家 ID (0-indexed)
        done: 游戏是否结束
        winner: 获胜玩家 ID，None 表示和棋或未结束
        rewards: 各玩家累计奖励
        info: 游戏特定信息
    """
    observation: np.ndarray
    legal_actions: List[int]
    current_player: int
    done: bool = False
    winner: Optional[int] = None
    rewards: Dict[int, float] = field(default_factory=dict)
    info: Dict[str, Any] = field(default_factory=dict)


# ============================================================
# 游戏抽象基类
# ============================================================

T = TypeVar('T', bound='Game')


class Game(ABC):
    """游戏抽象基类
    
    所有游戏必须实现此接口。设计遵循 OpenAI Gym 风格，
    但针对 MCTS/AlphaZero/MuZero 场景进行了优化。
    
    核心方法:
    - reset(): 重置游戏，返回初始观测
    - step(action): 执行动作，返回 (obs, reward, done, info)
    - legal_actions(): 返回当前合法动作列表
    - clone(): 克隆游戏状态（用于 MCTS）
    
    属性:
    - observation_space: 观测空间
    - action_space: 动作空间
    - num_players: 玩家数量
    - player_type: 玩家类型（单人/双人/多人）
    
    类属性:
    - config_class: 关联的配置类（可选，继承自 GameConfig）
    
    Example:
        >>> game = ChineseChessGame()
        >>> obs = game.reset()
        >>> while not game.is_terminal():
        ...     action = game.legal_actions()[0]
        ...     obs, reward, done, info = game.step(action)
        
        >>> # 使用配置创建
        >>> game = ChineseChessGame.from_config({"max_game_length": 300})
    """
    
    # === 配置类关联（可选，子类可覆盖）===
    config_class: Optional[Type["GameConfig"]] = None
    
    # === 必须由子类定义的类属性 ===
    
    @property
    @abstractmethod
    def observation_space(self) -> ObservationSpace:
        """观测空间定义"""
        ...
    
    @property
    @abstractmethod
    def action_space(self) -> ActionSpace:
        """动作空间定义"""
        ...
    
    @property
    @abstractmethod
    def num_players(self) -> int:
        """玩家数量: 1=单人, 2=双人对弈, N=多人"""
        ...
    
    @property
    def player_type(self) -> PlayerType:
        """玩家类型（根据 num_players 自动推断，子类可覆盖）"""
        if self.num_players == 1:
            return PlayerType.SINGLE
        elif self.num_players == 2:
            return PlayerType.TWO_PLAYER
        else:
            return PlayerType.MULTI_PLAYER
    
    # === 元数据方法（必须实现）===
    
    @classmethod
    @abstractmethod
    def get_meta(cls) -> GameMeta:
        """返回游戏元数据（框架自动扫描用）
        
        开发者必须实现此方法，返回 GameMeta 对象。
        
        Example:
            @classmethod
            def get_meta(cls) -> GameMeta:
                return GameMeta(
                    name="我的游戏",
                    description="游戏描述",
                    tags=["board", "strategy"],
                )
        """
        ...
    
    @property
    @abstractmethod
    def supported_render_modes(self) -> List[str]:
        """返回支持的渲染模式列表
        
        必须至少支持 'text' 或 'human'（ASCII 文本输出）。
        
        可选模式:
        - 'text' / 'human': ASCII 文本（必须至少有一个）
        - 'json': JSON 可序列化数据（推荐，用于 Web UI）
        - 'svg': SVG 矢量图
        - 'rgb_array': RGB 图像数组
        
        Example:
            @property
            def supported_render_modes(self) -> List[str]:
                return ['text', 'json', 'svg']
        """
        ...
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """返回推荐的默认配置
        
        子类可覆盖此方法提供默认配置。
        用于 Web 界面展示和自动配置。
        """
        return {}
    
    # === 游戏属性（辅助）===
    
    @property
    def is_zero_sum(self) -> bool:
        """是否是零和博弈（双人对弈默认 True）"""
        return self.player_type == PlayerType.TWO_PLAYER
    
    @property
    def supports_self_play(self) -> bool:
        """是否支持自玩训练"""
        return self.player_type in (PlayerType.TWO_PLAYER, PlayerType.MULTI_PLAYER)
    
    # === 配置相关方法 ===
    
    @classmethod
    def from_config(cls: Type[T], config: Optional[Dict[str, Any]] = None, **kwargs) -> T:
        """从配置创建游戏实例
        
        这是推荐的创建游戏实例的方式，支持:
        1. 从字典创建配置
        2. 从关键字参数创建配置
        3. 验证配置合法性
        
        Args:
            config: 配置字典（可选）
            **kwargs: 额外的配置参数
            
        Returns:
            游戏实例
            
        Example:
            >>> game = ChineseChessGame.from_config({"max_game_length": 300})
            >>> game = ChineseChessGame.from_config(max_game_length=300)
        """
        # 合并配置
        final_config = {}
        if config:
            final_config.update(config)
        final_config.update(kwargs)
        
        # 如果有配置类，创建配置对象
        if cls.config_class is not None:
            from .config import GameConfig
            config_obj = cls.config_class.from_dict(final_config)
            return cls(config=config_obj)
        
        # 没有配置类，直接传递参数
        return cls(**final_config)
    
    @classmethod
    def get_config_class(cls) -> Optional[Type["GameConfig"]]:
        """获取关联的配置类
        
        Returns:
            配置类，如果没有关联则返回 None
        """
        return cls.config_class
    
    def get_config(self) -> Optional["GameConfig"]:
        """获取当前配置
        
        Returns:
            配置对象，如果没有配置则返回 None
        """
        return getattr(self, 'config', None)
    
    # === 核心方法 ===
    
    @abstractmethod
    def reset(self) -> np.ndarray:
        """重置游戏到初始状态
        
        Returns:
            observation: 初始观测，形状为 observation_space.shape
        
        Raises:
            RuntimeError: 如果重置失败
        """
        ...
    
    @abstractmethod
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """执行一个动作
        
        Args:
            action: 动作索引，必须在 legal_actions() 中
        
        Returns:
            observation: 新的观测
            reward: 当前玩家的即时奖励
            done: 游戏是否结束
            info: 额外信息字典，可包含:
                - 'winner': 获胜玩家 ID
                - 'termination': 终止原因
                - 'all_rewards': 所有玩家的奖励 dict
        
        Raises:
            ValueError: 如果动作非法
            RuntimeError: 如果执行动作失败
        """
        ...
    
    @abstractmethod
    def legal_actions(self) -> List[int]:
        """获取当前状态下的合法动作列表
        
        Returns:
            actions: 合法动作索引列表，可能为空（游戏结束时）
        
        Note:
            返回的列表应该是新创建的，调用者可以安全修改
        """
        ...
    
    @abstractmethod
    def current_player(self) -> int:
        """获取当前玩家 ID
        
        Returns:
            player_id: 玩家 ID (0-indexed)
            对于单人游戏，始终返回 0
            对于双人对弈，返回 0 或 1
        """
        ...
    
    @abstractmethod
    def clone(self: T) -> T:
        """深拷贝当前游戏状态
        
        用于 MCTS 搜索中的状态保存和恢复。
        
        Returns:
            game: 独立的游戏副本，修改不影响原游戏
        
        Note:
            实现必须确保完全独立，包括内部状态、历史记录等
        """
        ...
    
    # === 观测相关方法 ===
    
    @abstractmethod
    def get_observation(self) -> np.ndarray:
        """获取当前观测
        
        Returns:
            observation: 当前状态的观测张量
        
        Note:
            对于双人对弈游戏，观测应该从当前玩家视角编码
        """
        ...
    
    def get_legal_actions_mask(self) -> np.ndarray:
        """获取合法动作掩码
        
        Returns:
            mask: 形状为 (action_space.n,) 的布尔数组
                  mask[i] = True 表示动作 i 合法
        
        Note:
            默认实现基于 legal_actions()，子类可以覆盖以优化性能
        """
        mask = np.zeros(self.action_space.n, dtype=np.float32)
        for action in self.legal_actions():
            mask[action] = 1.0
        return mask
    
    # === 状态查询方法 ===
    
    def is_terminal(self) -> bool:
        """检查游戏是否结束
        
        默认实现：检查是否无合法动作
        """
        return len(self.legal_actions()) == 0
    
    def get_winner(self) -> Optional[int]:
        """获取获胜玩家
        
        Returns:
            winner: 获胜玩家 ID，None 表示和棋或游戏未结束
        
        Note:
            只在 is_terminal() 为 True 时调用才有意义
        """
        return None
    
    def get_rewards(self) -> Dict[int, float]:
        """获取所有玩家的最终奖励
        
        Returns:
            rewards: {player_id: reward} 字典
        
        Note:
            对于零和博弈，奖励之和应为 0
        """
        return {}
    
    # === 可视化方法（必须实现）===
    
    @abstractmethod
    def render(self, mode: str = "text") -> Any:
        """渲染当前游戏状态（必须实现）
        
        开发者必须实现此方法，至少支持 'text' 模式的 ASCII 输出。
        
        Args:
            mode: 渲染模式，必须是 supported_render_modes 中的一个
                - "text" / "human": ASCII 文本输出（必须支持）
                - "json": JSON 可序列化数据（推荐，用于 Web UI）
                - "svg": SVG 矢量图字符串
                - "rgb_array": RGB 图像数组 (H, W, 3)
        
        Returns:
            渲染结果，类型取决于 mode:
            - text/human: str 或 {"type": "text", "text": str}
            - json: dict (见下方 GridRenderData 格式)
            - svg: str 或 {"type": "svg", "svg": str}
            - rgb_array: np.ndarray
        
        Raises:
            ValueError: 如果 mode 不在 supported_render_modes 中
        
        JSON 模式返回格式（GridRenderData）:
            当 type="grid" 时，用于网格类游戏（棋盘游戏等）:
            {
                "type": "grid",
                "rows": int,                    # 行数
                "cols": int,                    # 列数
                "cells": list[list[any]],       # 二维数组，每个元素是格子内容
                "cell_colors": list[list[str]], # 可选，格子颜色
                "highlights": list[tuple[int, int]],  # ⚠️ 高亮格子坐标，格式为 [(row, col), ...]
                "labels": {                     # 可选，坐标标签
                    "row": list[str],
                    "col": list[str],
                },
                ... # 其他游戏特定字段
            }
            
            ⚠️ 注意: highlights 必须是元组列表 [(row, col), ...]，
            不能是对象列表 [{"row": r, "col": c}, ...]！
            前端会使用解构语法 ([r, c]) 来读取坐标。
        
        Example:
            def render(self, mode: str = "text") -> Any:
                if mode not in self.supported_render_modes:
                    raise ValueError(f"不支持的渲染模式: {mode}")
                if mode == "text":
                    return self._render_text()
                elif mode == "json":
                    # 正确的 highlights 格式
                    highlights = [(row, col)]  # ✓ 元组列表
                    # 错误: highlights = [{"row": row, "col": col}]  # ✗
                    return {
                        "type": "grid",
                        "rows": self.board_size,
                        "cols": self.board_size,
                        "cells": cells,
                        "highlights": highlights,
                    }
        """
        ...
    
    def get_state_hash(self) -> int:
        """获取当前状态的哈希值
        
        用于检测重复状态（如三次重复局面和棋）
        
        Returns:
            hash: 状态哈希值
        """
        return hash(self.get_observation().tobytes())
    
    # === 状态保存/恢复 ===
    
    def get_state(self) -> GameState:
        """获取当前完整游戏状态
        
        Returns:
            state: 可序列化的游戏状态快照
        """
        return GameState(
            observation=self.get_observation(),
            legal_actions=self.legal_actions(),
            current_player=self.current_player(),
            done=self.is_terminal(),
            winner=self.get_winner(),
            rewards=self.get_rewards(),
        )
    
    def set_state(self, state: GameState) -> None:
        """从状态快照恢复游戏
        
        Args:
            state: 之前通过 get_state() 获取的状态
        
        Note:
            默认抛出 NotImplementedError，子类按需实现
        """
        raise NotImplementedError("set_state() not implemented for this game")
    
    # === 实用方法 ===
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"obs_space={self.observation_space.shape}, "
            f"action_space={self.action_space.n}, "
            f"players={self.num_players})"
        )
    
    def validate(self) -> bool:
        """验证游戏实现是否正确
        
        执行基本的健全性检查，用于调试。
        
        Returns:
            valid: 是否通过所有检查
        
        Raises:
            AssertionError: 如果检查失败
        """
        # 重置游戏
        obs = self.reset()
        
        # 检查观测形状
        assert obs.shape == self.observation_space.shape, \
            f"观测形状不匹配: {obs.shape} != {self.observation_space.shape}"
        
        # 检查观测类型
        assert obs.dtype == self.observation_space.dtype, \
            f"观测类型不匹配: {obs.dtype} != {self.observation_space.dtype}"
        
        # 检查合法动作
        legal = self.legal_actions()
        assert len(legal) > 0, "初始状态应该有合法动作"
        assert all(0 <= a < self.action_space.n for a in legal), \
            f"非法动作索引: {[a for a in legal if not (0 <= a < self.action_space.n)]}"
        
        # 检查玩家
        player = self.current_player()
        assert 0 <= player < self.num_players, \
            f"玩家 ID 越界: {player} >= {self.num_players}"
        
        # 检查克隆
        cloned = self.clone()
        assert np.array_equal(self.get_observation(), cloned.get_observation()), \
            "克隆后观测不一致"
        
        # 执行一步并检查
        action = legal[0]
        obs, reward, done, info = self.step(action)
        
        assert obs.shape == self.observation_space.shape, \
            f"step 后观测形状不匹配: {obs.shape}"
        assert isinstance(reward, (int, float)), \
            f"奖励类型错误: {type(reward)}"
        assert isinstance(done, bool), \
            f"done 类型错误: {type(done)}"
        assert isinstance(info, dict), \
            f"info 类型错误: {type(info)}"
        
        logger.info(f"✓ 游戏 {self.__class__.__name__} 验证通过")
        return True


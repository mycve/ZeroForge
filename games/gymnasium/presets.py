"""
Gymnasium 预设游戏

提供常用游戏的便捷包装，可直接通过 make_game() 创建。

使用方法:
    >>> from games import make_game
    >>> game = make_game("atari_breakout")
    >>> game = make_game("gym_cartpole")
    >>> game = make_game("gym_lunarlander")
"""

import logging
from typing import Dict, Any, Type, Optional

from core.game import GameMeta

logger = logging.getLogger(__name__)


# ============================================================
# 依赖检测（实际尝试创建环境来验证）
# ============================================================

def _check_atari_available() -> bool:
    """检测 Atari/ALE 是否可用（实际创建环境验证）"""
    try:
        import gymnasium
        # 实际尝试创建环境
        env = gymnasium.make('ALE/Breakout-v5')
        env.close()
        return True
    except Exception:
        return False

def _check_box2d_available() -> bool:
    """检测 Box2D 是否可用（LunarLander 等需要）"""
    try:
        import gymnasium
        # 实际尝试创建环境
        env = gymnasium.make('LunarLander-v3')
        env.close()
        return True
    except Exception:
        return False

def _check_mujoco_available() -> bool:
    """检测 MuJoCo 是否可用"""
    try:
        import gymnasium
        # 实际尝试创建环境
        env = gymnasium.make('Ant-v5')
        env.close()
        return True
    except Exception:
        return False

# 缓存检测结果
_ATARI_AVAILABLE: Optional[bool] = None
_BOX2D_AVAILABLE: Optional[bool] = None
_MUJOCO_AVAILABLE: Optional[bool] = None

def is_atari_available() -> bool:
    global _ATARI_AVAILABLE
    if _ATARI_AVAILABLE is None:
        _ATARI_AVAILABLE = _check_atari_available()
    return _ATARI_AVAILABLE

def is_box2d_available() -> bool:
    global _BOX2D_AVAILABLE
    if _BOX2D_AVAILABLE is None:
        _BOX2D_AVAILABLE = _check_box2d_available()
    return _BOX2D_AVAILABLE

def is_mujoco_available() -> bool:
    global _MUJOCO_AVAILABLE
    if _MUJOCO_AVAILABLE is None:
        _MUJOCO_AVAILABLE = _check_mujoco_available()
    return _MUJOCO_AVAILABLE


# ============================================================
# 预设游戏工厂
# ============================================================

def create_atari_game(env_id: str, name: str, description: str):
    """创建 Atari 游戏类
    
    Args:
        env_id: Gymnasium 环境 ID
        name: 游戏显示名称
        description: 游戏描述
        
    Returns:
        游戏类
    """
    from .wrapper import GymnasiumWrapper
    from .config import AtariConfig
    
    class AtariGame(GymnasiumWrapper):
        """Atari 游戏"""
        
        config_class = AtariConfig
        _env_id_preset = env_id
        _name_preset = name
        _description_preset = description
        
        @classmethod
        def get_meta(cls) -> GameMeta:
            return GameMeta(
                name=cls._name_preset,
                description=cls._description_preset,
                version="1.0.0",
                author="ZeroForge",
                tags=["atari", "arcade", "single-player"],
                difficulty="medium",
                min_players=1,
                max_players=1,
            )
        
        def __init__(self, config: Optional[AtariConfig] = None):
            if config is None:
                config = AtariConfig(env_id=self._env_id_preset)
            super().__init__(config=config)
    
    # 设置类名
    game_name = env_id.split("/")[-1].replace("-v5", "").replace("-", "")
    AtariGame.__name__ = f"Atari{game_name}Game"
    AtariGame.__qualname__ = f"Atari{game_name}Game"
    
    return AtariGame


def create_classic_control_game(env_id: str, name: str, description: str):
    """创建经典控制游戏类"""
    from .wrapper import GymnasiumWrapper
    from .config import ClassicControlConfig
    
    class ClassicControlGame(GymnasiumWrapper):
        """经典控制游戏"""
        
        config_class = ClassicControlConfig
        _env_id_preset = env_id
        _name_preset = name
        _description_preset = description
        
        @classmethod
        def get_meta(cls) -> GameMeta:
            return GameMeta(
                name=cls._name_preset,
                description=cls._description_preset,
                version="1.0.0",
                author="ZeroForge",
                tags=["classic-control", "simple", "single-player"],
                difficulty="easy",
                min_players=1,
                max_players=1,
            )
        
        def __init__(self, config: Optional[ClassicControlConfig] = None):
            if config is None:
                config = ClassicControlConfig(env_id=self._env_id_preset)
            super().__init__(config=config)
    
    # 设置类名
    game_name = env_id.split("-")[0].replace("-", "")
    ClassicControlGame.__name__ = f"{game_name}Game"
    ClassicControlGame.__qualname__ = f"{game_name}Game"
    
    return ClassicControlGame


def create_mujoco_game(env_id: str, name: str, description: str):
    """创建 MuJoCo 游戏类"""
    from .wrapper import GymnasiumWrapper
    from .config import MuJoCoConfig
    
    class MuJoCoGame(GymnasiumWrapper):
        """MuJoCo 物理仿真游戏"""
        
        config_class = MuJoCoConfig
        _env_id_preset = env_id
        _name_preset = name
        _description_preset = description
        
        @classmethod
        def get_meta(cls) -> GameMeta:
            return GameMeta(
                name=cls._name_preset,
                description=cls._description_preset,
                version="1.0.0",
                author="ZeroForge",
                tags=["mujoco", "physics", "continuous-control", "single-player"],
                difficulty="hard",
                min_players=1,
                max_players=1,
            )
        
        def __init__(self, config: Optional[MuJoCoConfig] = None):
            if config is None:
                config = MuJoCoConfig(env_id=self._env_id_preset)
            super().__init__(config=config)
    
    # 设置类名
    game_name = env_id.split("-")[0].replace("-", "")
    MuJoCoGame.__name__ = f"MuJoCo{game_name}Game"
    MuJoCoGame.__qualname__ = f"MuJoCo{game_name}Game"
    
    return MuJoCoGame


# ============================================================
# 预设游戏定义
# ============================================================

ATARI_PRESETS: Dict[str, Dict[str, str]] = {
    "atari_breakout": {
        "env_id": "ALE/Breakout-v5",
        "name": "Breakout (Atari)",
        "description": "经典打砖块游戏，控制挡板反弹球击碎上方砖块",
    },
    "atari_pong": {
        "env_id": "ALE/Pong-v5",
        "name": "Pong (Atari)",
        "description": "经典乒乓球游戏，控制挡板与电脑对打",
    },
    "atari_spaceinvaders": {
        "env_id": "ALE/SpaceInvaders-v5",
        "name": "Space Invaders (Atari)",
        "description": "经典太空侵略者，射击下降的外星人",
    },
    "atari_qbert": {
        "env_id": "ALE/Qbert-v5",
        "name": "Q*bert (Atari)",
        "description": "控制 Q*bert 在金字塔上跳跃改变方块颜色",
    },
    "atari_seaquest": {
        "env_id": "ALE/Seaquest-v5",
        "name": "Seaquest (Atari)",
        "description": "驾驶潜艇射击敌人并营救潜水员",
    },
    "atari_beamrider": {
        "env_id": "ALE/BeamRider-v5",
        "name": "Beam Rider (Atari)",
        "description": "在光束上移动射击外星飞船",
    },
    "atari_enduro": {
        "env_id": "ALE/Enduro-v5",
        "name": "Enduro (Atari)",
        "description": "耐力赛车游戏，在变化的天气中超越其他车辆",
    },
    "atari_asteroids": {
        "env_id": "ALE/Asteroids-v5",
        "name": "Asteroids (Atari)",
        "description": "驾驶飞船射击并躲避小行星",
    },
    "atari_mspacman": {
        "env_id": "ALE/MsPacman-v5",
        "name": "Ms. Pac-Man (Atari)",
        "description": "控制吃豆人吃掉所有豆子并躲避鬼魂",
    },
    "atari_freeway": {
        "env_id": "ALE/Freeway-v5",
        "name": "Freeway (Atari)",
        "description": "控制小鸡穿越繁忙的高速公路",
    },
}

CLASSIC_CONTROL_PRESETS: Dict[str, Dict[str, str]] = {
    "gym_cartpole": {
        "env_id": "CartPole-v1",
        "name": "CartPole",
        "description": "平衡倒立摆，通过移动小车保持杆子直立",
    },
    "gym_lunarlander": {
        "env_id": "LunarLander-v3",
        "name": "Lunar Lander",
        "description": "控制月球着陆器安全降落到指定区域",
    },
    "gym_mountaincar": {
        "env_id": "MountainCar-v0",
        "name": "Mountain Car",
        "description": "驾驶小车通过来回摆动爬上山顶",
    },
    "gym_acrobot": {
        "env_id": "Acrobot-v1",
        "name": "Acrobot",
        "description": "控制双摆机器人摆动到目标高度",
    },
    "gym_pendulum": {
        "env_id": "Pendulum-v1",
        "name": "Pendulum",
        "description": "控制力矩使倒立摆保持平衡（连续控制）",
    },
}

MUJOCO_PRESETS: Dict[str, Dict[str, str]] = {
    "mujoco_ant": {
        "env_id": "Ant-v5",
        "name": "Ant (MuJoCo)",
        "description": "控制四足蚂蚁机器人行走",
    },
    "mujoco_halfcheetah": {
        "env_id": "HalfCheetah-v5",
        "name": "HalfCheetah (MuJoCo)",
        "description": "控制二维猎豹机器人奔跑",
    },
    "mujoco_hopper": {
        "env_id": "Hopper-v5",
        "name": "Hopper (MuJoCo)",
        "description": "控制单腿机器人跳跃前进",
    },
    "mujoco_walker2d": {
        "env_id": "Walker2d-v5",
        "name": "Walker2d (MuJoCo)",
        "description": "控制双足机器人行走",
    },
    "mujoco_humanoid": {
        "env_id": "Humanoid-v5",
        "name": "Humanoid (MuJoCo)",
        "description": "控制人形机器人行走（高维度挑战）",
    },
    "mujoco_swimmer": {
        "env_id": "Swimmer-v5",
        "name": "Swimmer (MuJoCo)",
        "description": "控制三节蛇形机器人游泳",
    },
}


# ============================================================
# 注册函数
# ============================================================

def register_preset_games():
    """注册所有预设游戏
    
    调用此函数后，可以通过 make_game("atari_breakout") 等方式创建游戏。
    只注册依赖已安装的游戏，未安装依赖的游戏静默跳过。
    """
    from games import register_game
    
    registered = []
    
    # 注册 Atari 游戏（需要 ale-py）
    if is_atari_available():
        for game_key, preset in ATARI_PRESETS.items():
            try:
                game_class = create_atari_game(
                    env_id=preset["env_id"],
                    name=preset["name"],
                    description=preset["description"],
                )
                register_game(game_key, validate=False)(game_class)
                registered.append(game_key)
            except Exception as e:
                logger.debug(f"注册 {game_key} 失败: {e}")
    
    # 注册经典控制游戏
    for game_key, preset in CLASSIC_CONTROL_PRESETS.items():
        # LunarLander 需要 Box2D
        if game_key == "gym_lunarlander" and not is_box2d_available():
            continue
        try:
            game_class = create_classic_control_game(
                env_id=preset["env_id"],
                name=preset["name"],
                description=preset["description"],
            )
            register_game(game_key, validate=False)(game_class)
            registered.append(game_key)
        except Exception as e:
            logger.debug(f"注册 {game_key} 失败: {e}")
    
    # 注册 MuJoCo 游戏（需要 mujoco）
    if is_mujoco_available():
        for game_key, preset in MUJOCO_PRESETS.items():
            try:
                game_class = create_mujoco_game(
                    env_id=preset["env_id"],
                    name=preset["name"],
                    description=preset["description"],
                )
                register_game(game_key, validate=False)(game_class)
                registered.append(game_key)
            except Exception as e:
                logger.debug(f"注册 {game_key} 失败: {e}")
    
    if registered:
        logger.debug(f"已注册 {len(registered)} 个 Gymnasium 预设游戏")
    
    return registered


def get_all_preset_names() -> Dict[str, list]:
    """获取所有预设游戏名称"""
    return {
        "atari": list(ATARI_PRESETS.keys()),
        "classic_control": list(CLASSIC_CONTROL_PRESETS.keys()),
        "mujoco": list(MUJOCO_PRESETS.keys()),
    }


# ============================================================
# 导出
# ============================================================

__all__ = [
    # 工厂函数
    "create_atari_game",
    "create_classic_control_game",
    "create_mujoco_game",
    # 注册
    "register_preset_games",
    "get_all_preset_names",
    # 预设列表
    "ATARI_PRESETS",
    "CLASSIC_CONTROL_PRESETS",
    "MUJOCO_PRESETS",
    # 依赖检测
    "is_atari_available",
    "is_box2d_available",
    "is_mujoco_available",
]

"""
Gymnasium 游戏模块

将 Gymnasium 环境（Atari、经典控制、MuJoCo 等）适配为 ZeroForge Game 接口。

依赖安装:
    # 基础依赖
    pip install gymnasium
    
    # Atari 游戏
    pip install gymnasium[atari] ale-py
    
    # MuJoCo 物理仿真
    pip install gymnasium[mujoco]

使用方法:
    # 方式1: 使用预设游戏
    >>> from games import make_game
    >>> game = make_game("atari_breakout")
    >>> game = make_game("gym_cartpole")
    >>> game = make_game("mujoco_ant")
    
    # 方式2: 自定义配置
    >>> from games.gymnasium import GymnasiumWrapper
    >>> game = GymnasiumWrapper("ALE/Breakout-v5")
    
    # 方式3: 使用配置类
    >>> from games.gymnasium import GymnasiumWrapper, AtariConfig
    >>> config = AtariConfig(
    ...     env_id="ALE/Breakout-v5",
    ...     frame_stack=4,
    ...     frame_skip=4,
    ... )
    >>> game = GymnasiumWrapper(config=config)

算法选择:
    - 支持 clone() 的环境: AlphaZero, MCTS
    - 不支持 clone() 的环境: Gumbel MuZero, Gumbel AlphaZero, Gumbel EfficientZero
    
    检查方式:
    >>> game = make_game("atari_breakout")
    >>> print(game.supports_mcts)       # 是否支持传统 MCTS
    >>> print(game.recommended_algorithm)  # 推荐算法
"""

import logging

logger = logging.getLogger(__name__)

# ============================================================
# 依赖检测
# ============================================================

GYMNASIUM_AVAILABLE = False
ATARI_AVAILABLE = False
MUJOCO_AVAILABLE = False

try:
    import gymnasium as gym
    GYMNASIUM_AVAILABLE = True
    
    # 检测 Atari
    try:
        import ale_py
        ATARI_AVAILABLE = True
    except ImportError:
        pass
    
    # 检测 MuJoCo
    try:
        import mujoco
        MUJOCO_AVAILABLE = True
    except ImportError:
        pass
        
except ImportError:
    logger.debug("gymnasium 未安装，Gymnasium 游戏不可用")


def check_gymnasium():
    """检查 gymnasium 是否可用，不可用则抛出异常"""
    if not GYMNASIUM_AVAILABLE:
        raise ImportError(
            "需要安装 gymnasium:\n"
            "  pip install gymnasium\n\n"
            "Atari 游戏:\n"
            "  pip install gymnasium[atari] ale-py\n\n"
            "MuJoCo 物理仿真:\n"
            "  pip install gymnasium[mujoco]"
        )


def check_atari():
    """检查 Atari 依赖是否可用"""
    check_gymnasium()
    if not ATARI_AVAILABLE:
        raise ImportError(
            "需要安装 Atari 依赖:\n"
            "  pip install gymnasium[atari] ale-py"
        )


def check_mujoco():
    """检查 MuJoCo 依赖是否可用"""
    check_gymnasium()
    if not MUJOCO_AVAILABLE:
        raise ImportError(
            "需要安装 MuJoCo 依赖:\n"
            "  pip install gymnasium[mujoco]"
        )


# ============================================================
# 条件导入
# ============================================================

if GYMNASIUM_AVAILABLE:
    from .wrapper import GymnasiumWrapper
    from .config import (
        GymnasiumConfig,
        AtariConfig,
        ClassicControlConfig,
        MuJoCoConfig,
        ATARI_GAMES,
        CLASSIC_CONTROL_GAMES,
        MUJOCO_GAMES,
    )
    from .utils import (
        FrameStack,
        preprocess_atari_observation,
        get_env_info,
    )
    from .presets import (
        register_preset_games,
        get_all_preset_names,
        ATARI_PRESETS,
        CLASSIC_CONTROL_PRESETS,
        MUJOCO_PRESETS,
    )
    
    # 自动注册预设游戏
    try:
        register_preset_games()
    except Exception as e:
        logger.debug(f"自动注册预设游戏失败: {e}")

else:
    # 提供占位符，导入时不报错但使用时报错
    def GymnasiumWrapper(*args, **kwargs):
        check_gymnasium()
    
    GymnasiumConfig = None
    AtariConfig = None
    ClassicControlConfig = None
    MuJoCoConfig = None


# ============================================================
# 导出
# ============================================================

__all__ = [
    # 依赖检测
    "GYMNASIUM_AVAILABLE",
    "ATARI_AVAILABLE",
    "MUJOCO_AVAILABLE",
    "check_gymnasium",
    "check_atari",
    "check_mujoco",
    # 核心类
    "GymnasiumWrapper",
    # 配置类
    "GymnasiumConfig",
    "AtariConfig",
    "ClassicControlConfig",
    "MuJoCoConfig",
]

# 条件导出
if GYMNASIUM_AVAILABLE:
    __all__.extend([
        "FrameStack",
        "preprocess_atari_observation",
        "get_env_info",
        "register_preset_games",
        "get_all_preset_names",
        "ATARI_GAMES",
        "CLASSIC_CONTROL_GAMES",
        "MUJOCO_GAMES",
        "ATARI_PRESETS",
        "CLASSIC_CONTROL_PRESETS",
        "MUJOCO_PRESETS",
    ])

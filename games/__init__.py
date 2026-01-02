"""
Games - 游戏实现模块

提供游戏注册系统和内置游戏实现。

目录结构:
├── __init__.py          # 注册系统 + 统一入口
└── <game_name>/         # 每个游戏独立目录
    ├── __init__.py      # 模块入口
    ├── game.py          # Game 接口实现
    ├── config.py        # 游戏配置
    └── <dependencies>/  # 游戏依赖（引擎等）

框架要求:
每个游戏必须实现以下方法:
- get_meta(): 返回 GameMeta 元数据
- supported_render_modes: 返回支持的渲染模式（至少 'text' 或 'human'）
- render(mode): 渲染游戏状态

渲染格式规范 (render mode="json"):
    对于网格类游戏，render(mode="json") 返回 GridRenderData 格式:
    {
        "type": "grid",
        "rows": int,
        "cols": int,
        "cells": list[list[any]],
        "highlights": list[tuple[int, int]],  # ⚠️ 必须是元组列表 [(row, col), ...]
        ...
    }
    
    ⚠️ 重要: highlights 必须是元组列表，不能是对象列表！
    
    正确: highlights = [(4, 4), (5, 3)]
    错误: highlights = [{"row": 4, "col": 4}]  # 会导致前端报错！

使用方法:
    >>> from games import make_game, list_games
    >>> game = make_game("tictactoe")
    >>> obs = game.reset()
    >>> action = game.legal_actions()[0]
    >>> obs, reward, done, info = game.step(action)
"""

from typing import Dict, Type, Any, List, Optional
import logging

from core.game import Game, GameMeta

logger = logging.getLogger(__name__)


# ============================================================
# 游戏注册表
# ============================================================

GAME_REGISTRY: Dict[str, Type[Game]] = {}
GAME_META_CACHE: Dict[str, GameMeta] = {}


class GameValidationError(Exception):
    """游戏验证错误"""
    pass


def _validate_game_class(cls: Type[Game], name: str) -> None:
    """验证游戏类是否实现了必须的方法
    
    Raises:
        GameValidationError: 如果验证失败
    """
    errors = []
    
    # 1. 检查 get_meta 方法
    if not hasattr(cls, 'get_meta') or not callable(getattr(cls, 'get_meta')):
        errors.append("缺少 get_meta() 类方法")
    else:
        try:
            meta = cls.get_meta()
            if not isinstance(meta, GameMeta):
                errors.append(f"get_meta() 必须返回 GameMeta 对象，得到 {type(meta)}")
        except Exception as e:
            errors.append(f"get_meta() 调用失败: {e}")
    
    # 2. 检查 supported_render_modes 属性（需要实例化来检查）
    # 这是 @property @abstractmethod，所以必须由子类实现
    
    # 3. 尝试实例化并检查渲染模式
    try:
        game = cls()
        render_modes = game.supported_render_modes
        if not render_modes:
            errors.append("supported_render_modes 不能为空")
        elif not any(m in render_modes for m in ['text', 'human']):
            errors.append("supported_render_modes 必须至少包含 'text' 或 'human'")
        
        # 检查 render 方法是否可用
        for mode in render_modes:
            try:
                result = game.render(mode=mode)
                if result is None:
                    logger.warning(f"游戏 '{name}' 的 render({mode}) 返回 None")
            except NotImplementedError:
                errors.append(f"render(mode='{mode}') 未实现")
            except Exception as e:
                errors.append(f"render(mode='{mode}') 失败: {e}")
                
    except TypeError as e:
        # 可能构造函数需要参数
        logger.warning(f"游戏 '{name}' 实例化检查跳过: {e}")
    except Exception as e:
        errors.append(f"实例化检查失败: {e}")
    
    if errors:
        error_msg = f"游戏 '{name}' ({cls.__name__}) 验证失败:\n" + "\n".join(f"  - {e}" for e in errors)
        raise GameValidationError(error_msg)


def register_game(name: str, validate: bool = True):
    """游戏注册装饰器
    
    Args:
        name: 游戏注册名称
        validate: 是否验证游戏类（默认 True）
    
    Usage:
        @register_game("tictactoe")
        class TicTacToeGame(Game):
            @classmethod
            def get_meta(cls) -> GameMeta:
                return GameMeta(name="井字棋", description="...")
            
            @property
            def supported_render_modes(self) -> List[str]:
                return ['text', 'json']
    """
    def decorator(cls: Type[Game]) -> Type[Game]:
        if name in GAME_REGISTRY:
            logger.warning(f"游戏 '{name}' 已注册，将被覆盖")
        
        # 验证游戏类
        if validate:
            try:
                _validate_game_class(cls, name)
                logger.debug(f"游戏 '{name}' 验证通过")
            except GameValidationError as e:
                logger.error(str(e))
                # 仍然注册，但发出警告
                logger.warning(f"游戏 '{name}' 验证失败，但仍然注册")
        
        GAME_REGISTRY[name] = cls
        
        # 缓存元数据
        try:
            GAME_META_CACHE[name] = cls.get_meta()
        except Exception:
            pass
        
        logger.debug(f"注册游戏: {name} -> {cls.__name__}")
        return cls
    return decorator


def make_game(name: str, **kwargs) -> Game:
    """创建游戏实例
    
    Args:
        name: 游戏名称
        **kwargs: 传递给游戏构造函数的参数
    
    Returns:
        Game 实例
    
    Raises:
        ValueError: 如果游戏未注册
    
    Example:
        >>> game = make_game("chinese_chess")
        >>> game = make_game("chinese_chess", history_steps=4)
    """
    if name not in GAME_REGISTRY:
        available = ", ".join(GAME_REGISTRY.keys())
        raise ValueError(
            f"未知游戏: '{name}'。可用游戏: {available or '(无)'}"
        )
    
    game_cls = GAME_REGISTRY[name]
    return game_cls(**kwargs)


def list_games() -> List[str]:
    """列出所有已注册的游戏"""
    return list(GAME_REGISTRY.keys())


def get_game_class(name: str) -> Optional[Type[Game]]:
    """获取游戏类（不实例化）"""
    return GAME_REGISTRY.get(name)


def get_game_info(name: str) -> Dict[str, Any]:
    """获取游戏信息
    
    返回游戏的元数据，包括观测空间、动作空间等。
    用于 Web 界面展示和 AI 理解。
    """
    if name not in GAME_REGISTRY:
        raise ValueError(f"未知游戏: '{name}'")
    
    game_cls = GAME_REGISTRY[name]
    
    # 创建临时实例获取空间信息
    try:
        game = game_cls()
        
        # 获取元数据
        meta = GAME_META_CACHE.get(name) or game_cls.get_meta()
        
        # 获取支持的渲染模式
        render_modes = game.supported_render_modes
        
        info = {
            "name": name,
            "class": game_cls.__name__,
            "module": game_cls.__module__,
            # 元数据
            "display_name": meta.name,
            "description": meta.description,
            "version": meta.version,
            "author": meta.author,
            "tags": meta.tags,
            "difficulty": meta.difficulty,
            # 空间信息
            "observation_space": {
                "shape": list(game.observation_space.shape),
                "dtype": str(game.observation_space.dtype),
            },
            "action_space": {
                "n": game.action_space.n,
            },
            # 玩家信息
            "num_players": game.num_players,
            "min_players": meta.min_players,
            "max_players": meta.max_players,
            "player_type": game.player_type.value,
            "is_zero_sum": game.is_zero_sum,
            "supports_self_play": game.supports_self_play,
            # 渲染
            "render_modes": render_modes,
            # 默认配置
            "default_config": game_cls.get_default_config(),
        }
        return info
    except Exception as e:
        logger.error(f"获取游戏 '{name}' 信息失败: {e}")
        return {
            "name": name,
            "class": game_cls.__name__,
            "num_players": 2,
            "render_modes": ["text"],
            "error": str(e),
        }


def get_game_debug_info(game: Game) -> Dict[str, Any]:
    """获取游戏调试信息
    
    用于 Web 调试界面，返回当前游戏状态的详细信息。
    
    Args:
        game: 游戏实例
        
    Returns:
        调试信息字典，包含:
        - state: 当前状态
        - legal_actions: 合法动作列表
        - observation: 观测数据
        - history: 历史记录
    """
    state = game.get_state()
    
    debug_info = {
        "current_player": state.current_player,
        "is_terminal": state.done,
        "winner": state.winner,
        "legal_actions_count": len(state.legal_actions),
        "legal_actions": state.legal_actions[:20],  # 最多返回 20 个
        "observation_shape": state.observation.shape,
        "observation_dtype": str(state.observation.dtype),
        "rewards": state.rewards,
        "info": state.info,
    }
    
    # 如果游戏有特定的调试方法，调用它
    if hasattr(game, 'get_debug_info'):
        debug_info.update(game.get_debug_info())
    
    return debug_info


# ============================================================
# 导入内置游戏（触发注册）
# ============================================================

from .chinese_chess import ChineseChessGame
from .tictactoe import TicTacToeGame
from .gomoku import Gomoku9x9Game, Gomoku15x15Game

__all__ = [
    # 注册系统
    "register_game",
    "make_game",
    "list_games",
    "get_game_class",
    "get_game_info",
    "get_game_debug_info",
    "GAME_REGISTRY",
    # 内置游戏
    "ChineseChessGame",
    "TicTacToeGame",
    "Gomoku9x9Game",
    "Gomoku15x15Game",
]

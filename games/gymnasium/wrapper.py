"""
GymnasiumWrapper - 将 Gymnasium 环境适配为 ZeroForge Game 接口

支持:
- Atari 游戏 (ALE)
- 经典控制 (CartPole, LunarLander 等)
- MuJoCo 物理仿真
- 其他 Gymnasium 兼容环境

特点:
- 自动处理观测预处理（帧堆叠、灰度化、缩放等）
- 连续动作空间自动离散化
- 支持 Gumbel 系列算法（不依赖 clone）
- 可选的 MCTS 支持（需要环境可深拷贝）
"""

import copy
import logging
import base64
from io import BytesIO
from typing import List, Tuple, Dict, Any, Optional, Callable
import numpy as np

from core.game import Game, ObservationSpace, ActionSpace, GameState, GameMeta, PlayerType

logger = logging.getLogger(__name__)


class GymnasiumWrapper(Game):
    """Gymnasium 环境适配器
    
    将 Gymnasium 环境包装为符合 ZeroForge Game 接口的对象。
    
    Attributes:
        env: 底层 Gymnasium 环境
        config: 环境配置
        frame_stacker: 帧堆叠器
        action_decoder: 动作解码器（离散 -> 实际动作）
        
    Example:
        >>> from games.gymnasium import GymnasiumWrapper
        >>> game = GymnasiumWrapper("CartPole-v1")
        >>> obs = game.reset()
        >>> action = game.legal_actions()[0]
        >>> obs, reward, done, info = game.step(action)
        
        >>> # Atari 游戏
        >>> game = GymnasiumWrapper("ALE/Breakout-v5")
    """
    
    # 关联配置类
    from .config import GymnasiumConfig
    config_class = GymnasiumConfig
    
    def __init__(
        self,
        env_id: Optional[str] = None,
        config: Optional["GymnasiumConfig"] = None,
        env: Optional[Any] = None,  # 直接传入 Gymnasium 环境
    ):
        """初始化适配器
        
        Args:
            env_id: Gymnasium 环境 ID（如 "CartPole-v1"）
            config: 环境配置
            env: 直接传入的 Gymnasium 环境实例
        """
        # 检查依赖
        try:
            import gymnasium as gym
        except ImportError:
            raise ImportError(
                "需要安装 gymnasium:\n"
                "  pip install gymnasium\n"
                "Atari 游戏:\n"
                "  pip install gymnasium[atari] ale-py\n"
                "MuJoCo:\n"
                "  pip install gymnasium[mujoco]"
            )
        
        # 处理配置
        if config is not None:
            self.config = config
            env_id = config.env_id
        else:
            from .config import GymnasiumConfig
            self.config = GymnasiumConfig(env_id=env_id or "CartPole-v1")
        
        # 创建环境
        if env is not None:
            self.env = env
            self._env_id = getattr(env, 'spec', None)
            if self._env_id:
                self._env_id = self._env_id.id
            else:
                self._env_id = str(type(env).__name__)
        else:
            self._env_id = self.config.env_id
            self.env = gym.make(
                self._env_id,
                render_mode=self.config.render_mode,
            )
        
        # 导入工具
        from .utils import (
            get_discrete_action_size,
            create_action_decoder,
            FrameStack,
            check_env_clonable,
        )
        
        # 动作空间处理
        self._action_size = get_discrete_action_size(
            self.env.action_space,
            bins=self.config.discrete_bins,
        )
        self._action_decoder = create_action_decoder(
            self.env.action_space,
            bins=self.config.discrete_bins,
        )
        
        # 判断是否是连续动作空间
        self._is_continuous = isinstance(self.env.action_space, gym.spaces.Box)
        
        # 观测空间处理
        self._setup_observation_space()
        
        # 帧堆叠
        self._frame_stacker = FrameStack(self.config.frame_stack) if self.config.frame_stack > 1 else None
        
        # MCTS 支持检测
        self._supports_clone = check_env_clonable(self.env)
        if not self._supports_clone:
            logger.info(f"环境 '{self._env_id}' 不支持深拷贝，将使用 Gumbel 算法")
        
        # 游戏状态
        self._current_obs: Optional[np.ndarray] = None
        self._done: bool = False
        self._total_reward: float = 0.0
        self._step_count: int = 0
        self._last_raw_obs: Optional[np.ndarray] = None  # 用于渲染
    
    def _setup_observation_space(self):
        """设置观测空间"""
        import gymnasium as gym
        
        obs_space = self.env.observation_space
        
        if isinstance(obs_space, gym.spaces.Box):
            # 图像观测
            if len(obs_space.shape) >= 2:
                if self.config.resize:
                    h, w = self.config.resize
                else:
                    h, w = obs_space.shape[:2]
                
                if self.config.grayscale and len(obs_space.shape) == 3:
                    # 灰度图像
                    shape = (self.config.frame_stack, h, w)
                else:
                    # 彩色图像或低维状态
                    if len(obs_space.shape) == 3:
                        shape = (self.config.frame_stack, h, w, obs_space.shape[2])
                    else:
                        shape = (self.config.frame_stack, h, w)
            else:
                # 低维状态向量
                shape = (self.config.frame_stack, *obs_space.shape)
            
            self._observation_space = ObservationSpace(
                shape=shape,
                dtype=np.float32,
                low=0.0 if self.config.normalize_obs else float(obs_space.low.min()),
                high=1.0 if self.config.normalize_obs else float(obs_space.high.max()),
            )
        
        elif isinstance(obs_space, gym.spaces.Discrete):
            # 离散观测（one-hot 编码）
            self._observation_space = ObservationSpace(
                shape=(self.config.frame_stack, obs_space.n),
                dtype=np.float32,
            )
        
        else:
            raise ValueError(f"不支持的观测空间类型: {type(obs_space)}")
        
        self._action_space = ActionSpace(n=self._action_size)
    
    # === 元数据方法 ===
    
    @classmethod
    def get_meta(cls) -> GameMeta:
        """返回游戏元数据"""
        return GameMeta(
            name="Gymnasium 环境",
            description="通用 Gymnasium 环境适配器，支持 Atari、经典控制、MuJoCo 等",
            version="1.0.0",
            author="ZeroForge",
            tags=["gymnasium", "rl", "single-player"],
            difficulty="medium",
            min_players=1,
            max_players=1,
        )
    
    @property
    def supported_render_modes(self) -> List[str]:
        """支持的渲染模式"""
        return ["text", "human", "json", "rgb_array"]
    
    @classmethod
    def get_default_config(cls) -> Dict[str, Any]:
        """返回默认配置"""
        return {
            "env_id": "CartPole-v1",
            "frame_stack": 1,
            "frame_skip": 1,
        }
    
    # === 空间属性 ===
    
    @property
    def observation_space(self) -> ObservationSpace:
        return self._observation_space
    
    @property
    def action_space(self) -> ActionSpace:
        return self._action_space
    
    @property
    def num_players(self) -> int:
        return 1  # Gymnasium 环境都是单人
    
    @property
    def player_type(self) -> PlayerType:
        return PlayerType.SINGLE
    
    # === 核心方法 ===
    
    def reset(self) -> np.ndarray:
        """重置游戏到初始状态"""
        raw_obs, info = self.env.reset()
        
        # NOOP 开始（Atari）
        if self.config.noop_max > 0:
            noop_steps = np.random.randint(1, self.config.noop_max + 1)
            for _ in range(noop_steps):
                raw_obs, _, terminated, truncated, _ = self.env.step(0)
                if terminated or truncated:
                    raw_obs, info = self.env.reset()
        
        # 预处理观测
        processed = self._preprocess_observation(raw_obs)
        
        # 帧堆叠重置
        if self._frame_stacker:
            self._current_obs = self._frame_stacker.reset(processed)
        else:
            self._current_obs = processed[np.newaxis, ...]  # 添加 frame 维度
        
        # 重置状态
        self._done = False
        self._total_reward = 0.0
        self._step_count = 0
        self._last_raw_obs = raw_obs
        
        return self._current_obs.astype(np.float32)
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict[str, Any]]:
        """执行动作
        
        Args:
            action: 离散动作索引
            
        Returns:
            (observation, reward, done, info)
        """
        if self._done:
            raise ValueError("游戏已结束，请先调用 reset()")
        
        # 解码动作
        actual_action = self._action_decoder(action)
        
        # 执行动作（可能跳帧）
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        
        for _ in range(self.config.frame_skip):
            raw_obs, reward, terminated, truncated, step_info = self.env.step(actual_action)
            total_reward += reward
            info.update(step_info)
            
            if terminated or truncated:
                break
        
        # Atari 生命丢失处理
        if self.config.terminal_on_life_loss and 'lives' in info:
            if hasattr(self, '_last_lives') and info['lives'] < self._last_lives:
                terminated = True
            self._last_lives = info['lives']
        
        # 预处理观测
        processed = self._preprocess_observation(raw_obs)
        
        # 帧堆叠
        if self._frame_stacker:
            self._current_obs = self._frame_stacker.push(processed)
        else:
            self._current_obs = processed[np.newaxis, ...]
        
        # 奖励处理
        if self.config.clip_rewards:
            reward = np.clip(total_reward, -1.0, 1.0)
        else:
            reward = total_reward * self.config.reward_scale
        
        # 更新状态
        self._done = terminated or truncated
        self._total_reward += total_reward
        self._step_count += 1
        self._last_raw_obs = raw_obs
        
        # 构建 info
        info.update({
            "raw_reward": total_reward,
            "total_reward": self._total_reward,
            "step_count": self._step_count,
            "terminated": terminated,
            "truncated": truncated,
        })
        
        if self._done:
            info["episode_reward"] = self._total_reward
            info["episode_length"] = self._step_count
        
        return self._current_obs.astype(np.float32), float(reward), self._done, info
    
    def _preprocess_observation(self, obs: np.ndarray) -> np.ndarray:
        """预处理观测"""
        import gymnasium as gym
        
        # 如果是离散观测，转为 one-hot
        if isinstance(self.env.observation_space, gym.spaces.Discrete):
            one_hot = np.zeros(self.env.observation_space.n, dtype=np.float32)
            one_hot[obs] = 1.0
            return one_hot
        
        processed = obs.astype(np.float32)
        
        # 图像预处理
        if len(obs.shape) >= 2:
            from .utils import preprocess_atari_observation
            processed = preprocess_atari_observation(
                processed,
                grayscale=self.config.grayscale,
                resize=self.config.resize,
                normalize=self.config.normalize_obs,
            )
        elif self.config.normalize_obs:
            # 低维状态归一化（处理无限边界的情况）
            low = self.env.observation_space.low
            high = self.env.observation_space.high
            
            # 检查是否有有限边界
            finite_low = np.isfinite(low)
            finite_high = np.isfinite(high)
            finite_mask = finite_low & finite_high
            
            if np.any(finite_mask):
                # 只归一化有限边界的维度
                range_vals = high - low
                range_vals[~finite_mask] = 1.0  # 避免除以无限值
                range_vals[range_vals == 0] = 1.0  # 避免除以零
                
                processed_norm = processed.copy()
                processed_norm[finite_mask] = (processed[finite_mask] - low[finite_mask]) / range_vals[finite_mask]
                processed = processed_norm
            # 如果没有有限边界，保持原值
        
        return processed
    
    def legal_actions(self) -> List[int]:
        """获取合法动作列表"""
        if self._done:
            return []
        return list(range(self._action_size))
    
    def current_player(self) -> int:
        """获取当前玩家（单人游戏始终返回 0）"""
        return 0
    
    def clone(self) -> "GymnasiumWrapper":
        """深拷贝游戏状态
        
        注意：部分环境不支持深拷贝，此时会抛出异常。
        对于不支持 clone 的环境，请使用 Gumbel 系列算法。
        """
        if not self._supports_clone:
            raise RuntimeError(
                f"环境 '{self._env_id}' 不支持深拷贝，无法用于传统 MCTS。\n"
                f"建议使用 Gumbel MuZero/AlphaZero/EfficientZero 算法，"
                f"这些算法不依赖环境克隆。"
            )
        
        try:
            cloned = GymnasiumWrapper.__new__(GymnasiumWrapper)
            cloned.config = self.config
            cloned._env_id = self._env_id
            cloned.env = copy.deepcopy(self.env)
            cloned._action_size = self._action_size
            cloned._action_decoder = self._action_decoder
            cloned._is_continuous = self._is_continuous
            cloned._observation_space = self._observation_space
            cloned._action_space = self._action_space
            cloned._frame_stacker = copy.deepcopy(self._frame_stacker) if self._frame_stacker else None
            cloned._supports_clone = self._supports_clone
            cloned._current_obs = self._current_obs.copy() if self._current_obs is not None else None
            cloned._done = self._done
            cloned._total_reward = self._total_reward
            cloned._step_count = self._step_count
            cloned._last_raw_obs = self._last_raw_obs.copy() if self._last_raw_obs is not None else None
            if hasattr(self, '_last_lives'):
                cloned._last_lives = self._last_lives
            return cloned
        except Exception as e:
            raise RuntimeError(f"环境克隆失败: {e}")
    
    # === 观测相关方法 ===
    
    def get_observation(self) -> np.ndarray:
        """获取当前观测"""
        if self._current_obs is None:
            return self.reset()
        return self._current_obs.astype(np.float32)
    
    # === 状态查询 ===
    
    def is_terminal(self) -> bool:
        """检查游戏是否结束"""
        return self._done
    
    def get_winner(self) -> Optional[int]:
        """获取获胜玩家（单人游戏返回 None）"""
        return None
    
    def get_rewards(self) -> Dict[int, float]:
        """获取所有玩家的最终奖励"""
        return {0: float(self._total_reward)}
    
    # === 可视化 ===
    
    def render(self, mode: str = "text") -> Any:
        """渲染当前状态"""
        if mode not in self.supported_render_modes:
            raise ValueError(f"不支持的渲染模式: {mode}，支持: {self.supported_render_modes}")
        
        if mode == "rgb_array":
            return self._last_raw_obs
        
        elif mode == "human":
            print(f"环境: {self._env_id}")
            print(f"步数: {self._step_count}")
            print(f"累计奖励: {self._total_reward:.2f}")
            print(f"是否结束: {self._done}")
            return None
        
        elif mode == "text":
            return {
                "type": "text",
                "text": (
                    f"环境: {self._env_id}\n"
                    f"步数: {self._step_count}\n"
                    f"累计奖励: {self._total_reward:.2f}\n"
                    f"是否结束: {self._done}"
                ),
            }
        
        elif mode == "json":
            # 确保所有值都是 JSON 可序列化的（转换 numpy 类型）
            result = {
                "type": "image",
                "env_id": self._env_id,
                "step_count": int(self._step_count),
                "total_reward": float(self._total_reward),
                "is_terminal": bool(self._done),
                "observation_shape": [int(x) for x in self._current_obs.shape] if self._current_obs is not None else None,
            }
            
            # 获取渲染图像
            render_img = None
            render_error = None
            
            # 优先使用环境的 render() 方法获取可视化图像
            if self.config.render_mode == "rgb_array":
                try:
                    render_img = self.env.render()
                    if render_img is not None:
                        logger.debug(f"env.render() 成功: shape={render_img.shape}")
                except Exception as e:
                    render_error = str(e)
                    logger.warning(f"env.render() 失败: {e}")
            
            # 如果环境渲染失败，尝试使用原始观测（仅适用于图像观测）
            if render_img is None and self._last_raw_obs is not None and len(self._last_raw_obs.shape) >= 2:
                render_img = self._last_raw_obs
                logger.debug(f"使用原始观测作为图像: shape={render_img.shape}")
            
            # 编码图像
            if render_img is not None and len(render_img.shape) >= 2:
                try:
                    from PIL import Image
                    img = render_img
                    # 处理灰度图像
                    if len(img.shape) == 2:
                        img = np.stack([img, img, img], axis=-1)
                    # 确保是 uint8
                    if img.dtype != np.uint8:
                        if img.max() <= 1.0:
                            img = (img * 255).astype(np.uint8)
                        else:
                            img = img.astype(np.uint8)
                    pil_img = Image.fromarray(img)
                    buffer = BytesIO()
                    pil_img.save(buffer, format='PNG')
                    result["image_base64"] = base64.b64encode(buffer.getvalue()).decode('utf-8')
                    result["image_width"] = pil_img.width
                    result["image_height"] = pil_img.height
                except ImportError as e:
                    logger.warning(f"PIL 不可用: {e}")
                except Exception as e:
                    logger.warning(f"图像编码失败: {e}")
            else:
                # 记录为什么没有图像
                if render_error:
                    result["render_error"] = render_error
                elif self.config.render_mode != "rgb_array":
                    result["render_error"] = f"render_mode 不是 rgb_array: {self.config.render_mode}"
            
            return result
        
        return None
    
    # === Gumbel 算法支持 ===
    
    @property
    def supports_mcts(self) -> bool:
        """是否支持传统 MCTS（需要 clone 能力）"""
        return self._supports_clone
    
    @property
    def supports_gumbel(self) -> bool:
        """是否支持 Gumbel 算法（总是支持）"""
        return True
    
    @property
    def recommended_algorithm(self) -> str:
        """推荐的算法类型"""
        if self._supports_clone:
            return "alphazero"  # 支持 clone，可用传统 MCTS
        else:
            return "gumbel_muzero"  # 不支持 clone，推荐 Gumbel
    
    # === 调试信息 ===
    
    def get_debug_info(self) -> Dict[str, Any]:
        """获取调试信息（确保所有值都是 JSON 可序列化的）"""
        return {
            "env_id": self._env_id,
            "step_count": int(self._step_count),
            "total_reward": float(self._total_reward),
            "is_terminal": bool(self._done),
            "action_size": int(self._action_size),
            "is_continuous": bool(self._is_continuous),
            "supports_clone": bool(self._supports_clone),
            "frame_stack": int(self.config.frame_stack),
            "frame_skip": int(self.config.frame_skip),
            "recommended_algorithm": self.recommended_algorithm,
        }
    
    def __repr__(self) -> str:
        return (
            f"GymnasiumWrapper(env_id='{self._env_id}', "
            f"step={self._step_count}, "
            f"reward={self._total_reward:.2f}, "
            f"done={self._done})"
        )
    
    def close(self):
        """关闭环境"""
        if hasattr(self, 'env') and self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass  # 忽略关闭时的错误
    
    def __del__(self):
        """析构时关闭环境"""
        try:
            self.close()
        except Exception:
            pass  # Python 关闭时可能会报错，忽略


# ============================================================
# 导出
# ============================================================

__all__ = [
    "GymnasiumWrapper",
]

"""
Gymnasium 工具函数

提供动作空间离散化、观测预处理等工具。
"""

import numpy as np
from typing import Tuple, Callable, Optional, Any
import logging

logger = logging.getLogger(__name__)


# ============================================================
# 动作空间离散化
# ============================================================

def discretize_continuous_action(
    action_idx: int,
    low: np.ndarray,
    high: np.ndarray,
    bins: int = 11,
) -> np.ndarray:
    """将离散动作索引转换为连续动作
    
    Args:
        action_idx: 离散动作索引
        low: 动作空间下界
        high: 动作空间上界
        bins: 每个维度的离散化 bins 数量
        
    Returns:
        连续动作值数组
    """
    dim = len(low)
    
    # 解码多维动作索引
    indices = []
    remaining = action_idx
    for _ in range(dim):
        indices.append(remaining % bins)
        remaining //= bins
    
    # 转换为连续值
    continuous = np.zeros(dim, dtype=np.float32)
    for i in range(dim):
        # 将 [0, bins-1] 映射到 [low, high]
        ratio = indices[i] / (bins - 1) if bins > 1 else 0.5
        continuous[i] = low[i] + ratio * (high[i] - low[i])
    
    return continuous


def get_discrete_action_size(action_space: Any, bins: int = 11) -> int:
    """计算离散化后的动作空间大小
    
    Args:
        action_space: Gymnasium 动作空间
        bins: 每个维度的离散化 bins 数量
        
    Returns:
        离散动作总数
    """
    try:
        import gymnasium as gym
    except ImportError:
        raise ImportError("需要安装 gymnasium: pip install gymnasium")
    
    if isinstance(action_space, gym.spaces.Discrete):
        return action_space.n
    elif isinstance(action_space, gym.spaces.Box):
        dim = int(np.prod(action_space.shape))
        return bins ** dim
    elif isinstance(action_space, gym.spaces.MultiDiscrete):
        return int(np.prod(action_space.nvec))
    else:
        raise ValueError(f"不支持的动作空间类型: {type(action_space)}")


def create_action_decoder(
    action_space: Any,
    bins: int = 11,
) -> Callable[[int], Any]:
    """创建动作解码器：离散索引 -> 实际动作
    
    Args:
        action_space: Gymnasium 动作空间
        bins: 连续空间离散化 bins 数量
        
    Returns:
        解码函数
    """
    try:
        import gymnasium as gym
    except ImportError:
        raise ImportError("需要安装 gymnasium: pip install gymnasium")
    
    if isinstance(action_space, gym.spaces.Discrete):
        # 离散空间，直接返回
        return lambda a: int(a)
    
    elif isinstance(action_space, gym.spaces.Box):
        # 连续空间，离散化
        low = action_space.low.flatten()
        high = action_space.high.flatten()
        return lambda a: discretize_continuous_action(a, low, high, bins).reshape(action_space.shape)
    
    elif isinstance(action_space, gym.spaces.MultiDiscrete):
        # 多离散空间
        nvec = action_space.nvec
        def decode(action_idx: int) -> np.ndarray:
            actions = []
            remaining = action_idx
            for n in nvec:
                actions.append(remaining % n)
                remaining //= n
            return np.array(actions, dtype=np.int64)
        return decode
    
    else:
        raise ValueError(f"不支持的动作空间类型: {type(action_space)}")


# ============================================================
# 观测预处理
# ============================================================

def preprocess_atari_observation(
    obs: np.ndarray,
    grayscale: bool = True,
    resize: Optional[Tuple[int, int]] = (84, 84),
    normalize: bool = True,
) -> np.ndarray:
    """预处理 Atari 观测
    
    Args:
        obs: 原始观测 (H, W, C) 格式
        grayscale: 是否转换为灰度
        resize: 调整大小，None 表示不调整
        normalize: 是否归一化到 [0, 1]
        
    Returns:
        预处理后的观测
    """
    processed = obs.astype(np.float32)
    
    # 灰度化
    if grayscale and len(obs.shape) == 3 and obs.shape[-1] == 3:
        # RGB to grayscale: 0.299*R + 0.587*G + 0.114*B
        processed = 0.299 * processed[..., 0] + 0.587 * processed[..., 1] + 0.114 * processed[..., 2]
    
    # 调整大小
    if resize is not None and processed.shape[:2] != resize:
        processed = resize_image(processed, resize)
    
    # 归一化
    if normalize:
        processed = processed / 255.0
    
    return processed


def resize_image(img: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """简单的图像缩放（双线性插值）
    
    Args:
        img: 输入图像
        size: 目标大小 (height, width)
        
    Returns:
        缩放后的图像
    """
    try:
        # 尝试使用 PIL（如果可用）
        from PIL import Image
        if len(img.shape) == 2:
            pil_img = Image.fromarray(img.astype(np.uint8), mode='L')
        else:
            pil_img = Image.fromarray(img.astype(np.uint8))
        pil_img = pil_img.resize((size[1], size[0]), Image.BILINEAR)
        return np.array(pil_img, dtype=np.float32)
    except ImportError:
        pass
    
    try:
        # 尝试使用 cv2（如果可用）
        import cv2
        return cv2.resize(img, (size[1], size[0]), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    except ImportError:
        pass
    
    # 简单的最近邻插值（fallback）
    h, w = img.shape[:2]
    new_h, new_w = size
    
    row_ratio = h / new_h
    col_ratio = w / new_w
    
    row_idx = (np.arange(new_h) * row_ratio).astype(int)
    col_idx = (np.arange(new_w) * col_ratio).astype(int)
    
    row_idx = np.clip(row_idx, 0, h - 1)
    col_idx = np.clip(col_idx, 0, w - 1)
    
    if len(img.shape) == 2:
        return img[row_idx][:, col_idx].astype(np.float32)
    else:
        return img[row_idx][:, col_idx, :].astype(np.float32)


class FrameStack:
    """帧堆叠器
    
    将最近 N 帧堆叠为单个观测，用于提供时序信息。
    
    Attributes:
        num_frames: 堆叠帧数
        frames: 帧缓冲区
    """
    
    def __init__(self, num_frames: int = 4):
        """初始化帧堆叠器
        
        Args:
            num_frames: 堆叠帧数
        """
        self.num_frames = num_frames
        self.frames: list = []
    
    def reset(self, initial_frame: np.ndarray) -> np.ndarray:
        """重置并用初始帧填充
        
        Args:
            initial_frame: 初始帧
            
        Returns:
            堆叠后的观测
        """
        self.frames = [initial_frame.copy() for _ in range(self.num_frames)]
        return self.get()
    
    def push(self, frame: np.ndarray) -> np.ndarray:
        """添加新帧
        
        Args:
            frame: 新帧
            
        Returns:
            堆叠后的观测
        """
        self.frames.pop(0)
        self.frames.append(frame.copy())
        return self.get()
    
    def get(self) -> np.ndarray:
        """获取堆叠观测
        
        Returns:
            堆叠后的观测，形状为 (num_frames, H, W) 或 (num_frames, H, W, C)
        """
        return np.stack(self.frames, axis=0)


# ============================================================
# 环境工具
# ============================================================

def check_env_clonable(env: Any) -> bool:
    """检查环境是否支持深拷贝
    
    Args:
        env: Gymnasium 环境
        
    Returns:
        是否支持深拷贝
    """
    import copy
    try:
        cloned = copy.deepcopy(env)
        # 验证克隆是否有效
        if hasattr(cloned, 'reset'):
            return True
        return False
    except Exception as e:
        logger.debug(f"环境不支持深拷贝: {e}")
        return False


def get_env_info(env: Any) -> dict:
    """获取环境信息
    
    Args:
        env: Gymnasium 环境
        
    Returns:
        环境信息字典
    """
    try:
        import gymnasium as gym
    except ImportError:
        return {"error": "gymnasium not installed"}
    
    info = {
        "observation_space": str(env.observation_space),
        "action_space": str(env.action_space),
        "reward_range": getattr(env, 'reward_range', (-float('inf'), float('inf'))),
        "spec": str(getattr(env, 'spec', None)),
    }
    
    # 动作空间类型
    if isinstance(env.action_space, gym.spaces.Discrete):
        info["action_type"] = "discrete"
        info["action_size"] = env.action_space.n
    elif isinstance(env.action_space, gym.spaces.Box):
        info["action_type"] = "continuous"
        info["action_shape"] = env.action_space.shape
        info["action_low"] = env.action_space.low.tolist()
        info["action_high"] = env.action_space.high.tolist()
    elif isinstance(env.action_space, gym.spaces.MultiDiscrete):
        info["action_type"] = "multi_discrete"
        info["action_nvec"] = env.action_space.nvec.tolist()
    
    # 观测空间类型
    if isinstance(env.observation_space, gym.spaces.Box):
        info["obs_type"] = "box"
        info["obs_shape"] = env.observation_space.shape
        info["obs_dtype"] = str(env.observation_space.dtype)
    elif isinstance(env.observation_space, gym.spaces.Discrete):
        info["obs_type"] = "discrete"
        info["obs_n"] = env.observation_space.n
    
    return info


# ============================================================
# 导出
# ============================================================

__all__ = [
    "discretize_continuous_action",
    "get_discrete_action_size",
    "create_action_decoder",
    "preprocess_atari_observation",
    "resize_image",
    "FrameStack",
    "check_env_clonable",
    "get_env_info",
]

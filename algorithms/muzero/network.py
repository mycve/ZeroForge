"""
MuZero Network - MuZero 神经网络

包含三个核心网络:
- RepresentationNetwork: 观测 -> 隐藏状态
- DynamicsNetwork: (隐藏状态, 动作) -> (下一状态, 奖励)
- PredictionNetwork: 隐藏状态 -> (策略, 价值)

使用 ConvNeXt V2 作为骨干网络。
"""

import math
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# ConvNeXt V2 组件
# ============================================================

class GRN(nn.Module):
    """Global Response Normalization"""
    def __init__(self, dim: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, dim, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, dim, 1, 1))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gx = torch.norm(x, p=2, dim=(2, 3), keepdim=True)
        nx = gx / (gx.mean(dim=1, keepdim=True) + 1e-6)
        return x * (self.gamma * nx + self.beta) + x


class ConvNeXtV2Block(nn.Module):
    """ConvNeXt V2 Block"""
    def __init__(self, dim: int, expansion: int = 4):
        super().__init__()
        hidden_dim = dim * expansion
        
        self.dwconv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)
        self.norm = nn.LayerNorm(dim)
        self.pwconv1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.grn = GRN(hidden_dim)
        self.pwconv2 = nn.Linear(hidden_dim, dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = x.permute(0, 3, 1, 2)
        x = self.grn(x)
        x = x.permute(0, 2, 3, 1)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)
        return residual + x


# ============================================================
# MuZero 网络组件
# ============================================================

class RepresentationNetwork(nn.Module):
    """表示网络: 观测 -> 隐藏状态"""
    
    def __init__(
        self,
        input_channels: int,
        channels: int,
        num_blocks: int,
        board_height: int,
        board_width: int
    ):
        super().__init__()
        self.board_height = board_height
        self.board_width = board_width
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, channels, 3, padding=1),
            nn.LayerNorm([channels, board_height, board_width])
        )
        
        # ConvNeXt V2 blocks
        self.blocks = nn.ModuleList([
            ConvNeXtV2Block(channels) for _ in range(num_blocks)
        ])
        
        # 输出归一化
        self.norm = nn.LayerNorm([channels, board_height, board_width])
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.stem(obs)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)


class DynamicsNetwork(nn.Module):
    """动态网络: (隐藏状态, 动作) -> (下一状态, 奖励)"""
    
    def __init__(
        self,
        channels: int,
        num_blocks: int,
        action_space: int,
        board_height: int,
        board_width: int
    ):
        super().__init__()
        self.board_height = board_height
        self.board_width = board_width
        
        self.action_embed = nn.Embedding(action_space, channels)
        
        # 融合层
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.LayerNorm([channels, board_height, board_width])
        )
        
        # ConvNeXt V2 blocks
        self.blocks = nn.ModuleList([
            ConvNeXtV2Block(channels) for _ in range(num_blocks)
        ])
        
        self.norm = nn.LayerNorm([channels, board_height, board_width])
        
        # 奖励头
        self.reward_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels * board_height * board_width, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )
    
    def forward(self, hidden: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = hidden.size(0)
        
        # 动作嵌入 -> 空间平面
        action_emb = self.action_embed(action).view(batch_size, -1, 1, 1)
        action_plane = action_emb.expand(-1, -1, self.board_height, self.board_width)
        
        # 融合
        x = torch.cat([hidden, action_plane], dim=1)
        x = self.fuse(x)
        
        # ConvNeXt V2 blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        reward = self.reward_head(x)
        
        return x, reward


class PredictionNetwork(nn.Module):
    """预测网络: 隐藏状态 -> (策略, 价值)"""
    
    def __init__(self, channels: int, action_space: int, board_height: int, board_width: int):
        super().__init__()
        spatial_size = board_height * board_width
        
        # 策略头
        self.policy_conv = nn.Conv2d(channels, 32, 1)
        self.policy_bn = nn.BatchNorm2d(32)
        self.policy_fc = nn.Linear(32 * spatial_size, action_space)
        
        # 价值头
        self.value_conv = nn.Conv2d(channels, 32, 1)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * spatial_size, 256)
        self.value_fc2 = nn.Linear(256, 1)
    
    def forward(self, hidden: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        hidden = hidden.contiguous()
        
        # 策略
        policy = F.relu(self.policy_bn(self.policy_conv(hidden)))
        policy = policy.reshape(policy.size(0), -1)
        policy = self.policy_fc(policy)
        
        # 价值
        value = F.relu(self.value_bn(self.value_conv(hidden)))
        value = value.reshape(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value


# ============================================================
# MuZero 完整网络
# ============================================================

class MuZeroNetwork(nn.Module):
    """MuZero 完整网络
    
    Args:
        obs_shape: 观测形状 (C, H, W)
        action_space: 动作空间大小
        channels: 隐藏通道数
        num_blocks: ConvNeXt 块数量
    """
    
    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        action_space: int,
        channels: int = 128,
        num_blocks: int = 6
    ):
        super().__init__()
        
        assert len(obs_shape) == 3, f"观测形状必须是 (C, H, W)，得到 {obs_shape}"
        
        input_channels, board_height, board_width = obs_shape
        
        self.obs_shape = obs_shape
        self.action_space = action_space
        self.channels = channels
        self.board_height = board_height
        self.board_width = board_width
        
        # 三个核心网络
        self.representation = RepresentationNetwork(
            input_channels, channels, num_blocks, board_height, board_width
        )
        self.dynamics = DynamicsNetwork(
            channels, num_blocks, action_space, board_height, board_width
        )
        self.prediction = PredictionNetwork(
            channels, action_space, board_height, board_width
        )
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播: 观测 -> (策略 logits, 价值)
        
        兼容 AlphaZero 风格的调用，用于训练循环。
        """
        hidden = self.representation(obs)
        policy, value = self.prediction(hidden)
        return policy, value
    
    def initial_inference(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """初始推理: 观测 -> (隐藏状态, 策略, 价值)"""
        hidden = self.representation(obs)
        policy, value = self.prediction(hidden)
        return hidden, policy, value
    
    def recurrent_inference(
        self, hidden: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """循环推理: (隐藏状态, 动作) -> (下一状态, 奖励, 策略, 价值)"""
        next_hidden, reward = self.dynamics(hidden, action)
        policy, value = self.prediction(next_hidden)
        return next_hidden, reward, policy, value
    
    def get_info(self) -> dict:
        """获取网络信息"""
        return {
            "obs_shape": self.obs_shape,
            "action_space": self.action_space,
            "channels": self.channels,
            "parameters": sum(p.numel() for p in self.parameters()),
        }


# ============================================================
# 简单 MuZero 网络（用于低维状态观测，如 Gymnasium）
# ============================================================

class SimpleMuZeroNetwork(nn.Module):
    """简单 MuZero 网络（MLP）
    
    用于低维状态观测（如 CartPole 的 4 维状态）。
    不使用卷积，适合 Gymnasium 等环境。
    
    Args:
        obs_shape: 观测形状（可以是任意维度，会被展平）
        action_space: 动作空间大小
        hidden_dim: 隐藏层维度
        num_layers: 隐藏层数量
    """
    
    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        action_space: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
    ):
        super().__init__()
        
        # 计算输入维度
        obs_size = 1
        for d in obs_shape:
            obs_size *= d
        
        self.obs_shape = obs_shape
        self.obs_size = obs_size
        self.action_space = action_space
        self.hidden_dim = hidden_dim
        
        # Representation Network: obs -> hidden
        repr_layers = [nn.Linear(obs_size, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            repr_layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.representation = nn.Sequential(*repr_layers)
        
        # Dynamics Network: (hidden, action) -> (next_hidden, reward)
        self.dynamics_hidden = nn.Sequential(
            nn.Linear(hidden_dim + action_space, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.dynamics_reward = nn.Linear(hidden_dim, 1)
        
        # Prediction Network: hidden -> (policy, value)
        self.prediction_policy = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_space),
        )
        self.prediction_value = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh(),
        )
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播: 观测 -> (策略 logits, 价值)"""
        obs_flat = obs.view(obs.shape[0], -1)
        hidden = self.representation(obs_flat)
        policy = self.prediction_policy(hidden)
        value = self.prediction_value(hidden)
        return policy, value
    
    def initial_inference(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """初始推理: 观测 -> (隐藏状态, 策略, 价值)"""
        obs_flat = obs.view(obs.shape[0], -1)
        hidden = self.representation(obs_flat)
        policy = self.prediction_policy(hidden)
        value = self.prediction_value(hidden)
        return hidden, policy, value
    
    def recurrent_inference(
        self, hidden: torch.Tensor, action: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """循环推理: (隐藏状态, 动作) -> (下一状态, 奖励, 策略, 价值)"""
        batch_size = hidden.shape[0]
        
        # One-hot 编码动作
        action_one_hot = torch.zeros(batch_size, self.action_space, device=hidden.device)
        action_one_hot.scatter_(1, action.unsqueeze(-1), 1)
        
        # Dynamics
        dynamics_input = torch.cat([hidden, action_one_hot], dim=-1)
        next_hidden = self.dynamics_hidden(dynamics_input)
        reward = self.dynamics_reward(next_hidden)
        
        # Prediction
        policy = self.prediction_policy(next_hidden)
        value = self.prediction_value(next_hidden)
        
        return next_hidden, reward, policy, value
    
    def get_info(self) -> dict:
        """获取网络信息"""
        return {
            "obs_shape": self.obs_shape,
            "action_space": self.action_space,
            "hidden_dim": self.hidden_dim,
            "parameters": sum(p.numel() for p in self.parameters()),
        }


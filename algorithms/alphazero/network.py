"""
AlphaZero Network - AlphaZero 神经网络

与 MuZero 的区别:
- 无 DynamicsNetwork
- 只有 representation + prediction

提供多种网络大小:
- SimpleAlphaZeroNetwork: 轻量网络，适用于井字棋等小游戏（~1K 参数）
- AlphaZeroNetwork: 标准网络，使用 ConvNeXt V2（~100K+ 参数）
"""

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# 简单网络（适用于小游戏）
# ============================================================

class SimpleAlphaZeroNetwork(nn.Module):
    """简单 AlphaZero 网络
    
    适用于井字棋等小游戏，使用全连接网络而非卷积。
    参数量极少（~1K-10K），训练快速。
    
    Args:
        obs_shape: 观测形状 (C, H, W)
        action_space: 动作空间大小
        hidden_dim: 隐藏层维度
        num_layers: 隐藏层数量
    """
    
    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        action_space: int,
        hidden_dim: int = 64,
        num_layers: int = 2
    ):
        super().__init__()
        
        self.obs_shape = obs_shape
        self.action_space = action_space
        self.hidden_dim = hidden_dim
        
        # 计算输入维度
        input_dim = 1
        for d in obs_shape:
            input_dim *= d
        self.input_dim = input_dim
        
        # 共享表示层
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        self.shared = nn.Sequential(*layers)
        
        # 策略头
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_space)
        )
        
        # 价值头
        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()
        )
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播
        
        Args:
            obs: 观测张量 [B, C, H, W] 或 [B, input_dim]
        
        Returns:
            policy_logits: 策略 logits [B, action_space]
            value: 价值预测 [B, 1]
        """
        # 展平输入
        x = obs.reshape(obs.size(0), -1)
        
        # 共享表示
        hidden = self.shared(x)
        
        # 策略和价值
        policy_logits = self.policy_head(hidden)
        value = self.value_head(hidden)
        
        return policy_logits, value
    
    def get_info(self) -> dict:
        """获取网络信息"""
        return {
            "obs_shape": self.obs_shape,
            "action_space": self.action_space,
            "hidden_dim": self.hidden_dim,
            "parameters": sum(p.numel() for p in self.parameters()),
            "type": "SimpleAlphaZero (MLP)",
        }


# ============================================================
# ConvNeXt V2 组件（复用）
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
# AlphaZero 网络
# ============================================================

class AlphaZeroNetwork(nn.Module):
    """AlphaZero 神经网络
    
    只有 representation + prediction，无 dynamics。
    
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
        
        # Representation Network (Stem + Blocks)
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, channels, 3, padding=1),
            nn.LayerNorm([channels, board_height, board_width])
        )
        
        self.blocks = nn.ModuleList([
            ConvNeXtV2Block(channels) for _ in range(num_blocks)
        ])
        
        self.norm = nn.LayerNorm([channels, board_height, board_width])
        
        # Prediction Network
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
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """前向传播
        
        Args:
            obs: 观测张量 [B, C, H, W]
        
        Returns:
            policy_logits: 策略 logits [B, action_space]
            value: 价值预测 [B, 1]
        """
        # Representation
        x = self.stem(obs)
        for block in self.blocks:
            x = block(x)
        hidden = self.norm(x)
        
        # Prediction
        hidden = hidden.contiguous()
        
        # 策略
        policy = F.relu(self.policy_bn(self.policy_conv(hidden)))
        policy = policy.reshape(policy.size(0), -1)
        policy_logits = self.policy_fc(policy)
        
        # 价值
        value = F.relu(self.value_bn(self.value_conv(hidden)))
        value = value.reshape(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy_logits, value
    
    def get_features(self, obs: torch.Tensor) -> torch.Tensor:
        """获取特征（用于 MCTS）"""
        x = self.stem(obs)
        for block in self.blocks:
            x = block(x)
        return self.norm(x)
    
    def predict(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """从特征预测策略和价值"""
        hidden = features.contiguous()
        
        policy = F.relu(self.policy_bn(self.policy_conv(hidden)))
        policy = policy.reshape(policy.size(0), -1)
        policy_logits = self.policy_fc(policy)
        
        value = F.relu(self.value_bn(self.value_conv(hidden)))
        value = value.reshape(value.size(0), -1)
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))
        
        return policy_logits, value
    
    def get_info(self) -> dict:
        """获取网络信息"""
        return {
            "obs_shape": self.obs_shape,
            "action_space": self.action_space,
            "channels": self.channels,
            "parameters": sum(p.numel() for p in self.parameters()),
            "type": "AlphaZero (no dynamics)",
        }


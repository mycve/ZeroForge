"""
AlphaZero 网络 - 格子图 GNN 架构
目标：用关系建模增强棋子规律与细节变化的学习能力
"""

import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from xiangqi.actions import BOARD_HEIGHT, BOARD_WIDTH, ACTION_SPACE_SIZE


def _build_grid_neighbors(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    """构建格子图的 8 邻接边 (含对角线)"""
    neighbors = []
    for r in range(height):
        for c in range(width):
            ns = []
            for dr, dc in [
                (-1, 0), (1, 0), (0, -1), (0, 1),
                (-1, -1), (-1, 1), (1, -1), (1, 1),
            ]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width:
                    ns.append(nr * width + nc)
            neighbors.append(ns)
    max_deg = max(len(ns) for ns in neighbors)
    neighbor_idx = np.full((height * width, max_deg), -1, dtype=np.int32)
    neighbor_mask = np.zeros((height * width, max_deg), dtype=np.float32)
    for i, ns in enumerate(neighbors):
        neighbor_idx[i, :len(ns)] = ns
        neighbor_mask[i, :len(ns)] = 1.0
    return neighbor_idx, neighbor_mask


class GraphBlock(nn.Module):
    """轻量图消息传递块：聚合邻居特征 + 残差 MLP"""
    hidden_dim: int
    dtype: jnp.dtype = jnp.float32
    
    @nn.compact
    def __call__(self, h, neighbor_idx, neighbor_mask, train: bool = True):
        # h: (B, N, F)
        idx = jnp.where(neighbor_idx < 0, 0, neighbor_idx)
        neigh = jnp.take(h, idx, axis=1)  # (B, N, M, F)
        mask = neighbor_mask[None, :, :, None]  # (1, N, M, 1)
        neigh_sum = jnp.sum(neigh * mask, axis=2)
        denom = jnp.maximum(jnp.sum(mask, axis=2), 1.0)
        neigh_mean = neigh_sum / denom
        
        x = jnp.concatenate([h, neigh_mean], axis=-1)
        x = nn.LayerNorm(dtype=self.dtype)(x)
        x = nn.Dense(self.hidden_dim, dtype=self.dtype)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, dtype=self.dtype)(x)
        return h + x


class AlphaZeroNetwork(nn.Module):
    """格子图 GNN 网络
    
    设计原则：
    - 用格子图编码局面关系（四邻接）
    - 轻量消息传递 + 残差 MLP
    - 全局池化输出价值
    """
    action_space_size: int = ACTION_SPACE_SIZE
    channels: int = 96
    num_blocks: int = 6
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        neighbor_idx, neighbor_mask = _build_grid_neighbors(BOARD_HEIGHT, BOARD_WIDTH)
        self.neighbor_idx = jnp.array(neighbor_idx, dtype=jnp.int32)
        self.neighbor_mask = jnp.array(neighbor_mask, dtype=jnp.float32)
    
    @nn.compact
    def __call__(self, x, train: bool = True):
        if x.ndim != 4:
            raise ValueError(f"GNN 输入必须为 4D 张量 (B,C,H,W), 实际 ndim={x.ndim}")
        if x.shape[2] != BOARD_HEIGHT or x.shape[3] != BOARD_WIDTH:
            raise ValueError(
                f"GNN 输入棋盘尺寸不匹配，期望=({BOARD_HEIGHT},{BOARD_WIDTH})，"
                f"实际=({x.shape[2]},{x.shape[3]})"
            )
        
        x = x.astype(self.dtype)
        x = jnp.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
        batch_size = x.shape[0]
        
        # 节点特征：每个格子一个向量
        h = x.reshape((batch_size, BOARD_HEIGHT * BOARD_WIDTH, x.shape[-1]))
        h = nn.Dense(self.channels, dtype=self.dtype)(h)
        h = nn.relu(h)
        
        for _ in range(self.num_blocks):
            h = GraphBlock(self.channels, dtype=self.dtype)(h, self.neighbor_idx, self.neighbor_mask, train)
        
        # 策略头：节点特征 -> 全局动作 logits
        p = nn.Dense(32, dtype=self.dtype)(h)
        p = nn.relu(p)
        p = p.reshape((batch_size, -1))  # [B, 90*32] = [B, 2880]
        p = nn.Dense(self.action_space_size, dtype=self.dtype)(p)
        
        # 价值头：全局池化
        v = jnp.mean(h, axis=1)
        v = nn.Dense(128, dtype=self.dtype)(v)
        v = nn.relu(v)
        v = nn.Dense(1, dtype=self.dtype)(v)
        v = jnp.tanh(v).squeeze(-1)
        
        return p.astype(jnp.float32), v.astype(jnp.float32)

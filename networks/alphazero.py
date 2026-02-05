"""
AlphaZero 网络 - 多边类型 GNN 架构
目标：用多种边类型显式编码象棋规则，增强棋子关系建模能力

边类型：
1. 相邻边（8邻接）- 局部特征
2. 同行边 - 车/炮横向攻击线
3. 同列边 - 车/炮纵向攻击线  
4. 马步边 - 马的攻击范围（日字）
5. 象步边 - 象的攻击范围（田字对角）
"""

import numpy as np
import jax.numpy as jnp
import flax.linen as nn
from xiangqi.actions import BOARD_HEIGHT, BOARD_WIDTH, ACTION_SPACE_SIZE


def _build_edge_index(height: int, width: int, edge_type: str) -> tuple[np.ndarray, np.ndarray]:
    """构建指定类型的边索引
    
    Args:
        height: 棋盘高度
        width: 棋盘宽度
        edge_type: 边类型 'adjacent' | 'row' | 'col' | 'knight' | 'elephant'
    
    Returns:
        neighbor_idx: [N, max_degree] 邻居索引
        neighbor_mask: [N, max_degree] 有效掩码
    """
    N = height * width
    neighbors = [[] for _ in range(N)]
    
    for r in range(height):
        for c in range(width):
            idx = r * width + c
            
            if edge_type == 'adjacent':
                # 8邻接（含对角线）
                for dr, dc in [(-1,0), (1,0), (0,-1), (0,1), (-1,-1), (-1,1), (1,-1), (1,1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        neighbors[idx].append(nr * width + nc)
                        
            elif edge_type == 'row':
                # 同行（车/炮横向），排除自身和相邻
                for nc in range(width):
                    if abs(nc - c) > 1:  # 排除相邻的，已在 adjacent 中
                        neighbors[idx].append(r * width + nc)
                        
            elif edge_type == 'col':
                # 同列（车/炮纵向），排除自身和相邻
                for nr in range(height):
                    if abs(nr - r) > 1:  # 排除相邻的
                        neighbors[idx].append(nr * width + c)
                        
            elif edge_type == 'knight':
                # 马步（日字）
                for dr, dc in [(-2,-1), (-2,1), (-1,-2), (-1,2), (1,-2), (1,2), (2,-1), (2,1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        neighbors[idx].append(nr * width + nc)
                        
            elif edge_type == 'elephant':
                # 象步（田字对角）
                for dr, dc in [(-2,-2), (-2,2), (2,-2), (2,2)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < height and 0 <= nc < width:
                        neighbors[idx].append(nr * width + nc)
    
    # 转换为固定大小数组
    max_deg = max(len(ns) for ns in neighbors) if any(neighbors) else 1
    max_deg = max(max_deg, 1)  # 至少为 1
    neighbor_idx = np.full((N, max_deg), -1, dtype=np.int32)
    neighbor_mask = np.zeros((N, max_deg), dtype=np.float32)
    
    for i, ns in enumerate(neighbors):
        neighbor_idx[i, :len(ns)] = ns
        neighbor_mask[i, :len(ns)] = 1.0
    
    return neighbor_idx, neighbor_mask


def _build_all_edges(height: int, width: int) -> dict:
    """构建所有边类型"""
    edge_types = ['adjacent', 'row', 'col', 'knight', 'elephant']
    edges = {}
    for etype in edge_types:
        idx, mask = _build_edge_index(height, width, etype)
        edges[etype] = {'idx': idx, 'mask': mask}
    return edges


class MultiEdgeGraphBlock(nn.Module):
    """多边类型图消息传递块
    
    每种边类型有独立的消息变换，最后聚合所有边类型的消息。
    这样模型可以学习不同类型关系的不同重要性。
    """
    hidden_dim: int
    num_edge_types: int = 5  # adjacent, row, col, knight, elephant
    dtype: jnp.dtype = jnp.float32
    
    @nn.compact
    def __call__(self, h, edge_indices, edge_masks, train: bool = True):
        """
        Args:
            h: [B, N, F] 节点特征
            edge_indices: list of [N, max_deg] 每种边类型的邻居索引
            edge_masks: list of [N, max_deg] 每种边类型的有效掩码
        """
        B, N, F = h.shape
        
        # 对每种边类型分别计算消息
        all_messages = []
        for i, (neighbor_idx, neighbor_mask) in enumerate(zip(edge_indices, edge_masks)):
            # 聚合该边类型的邻居特征
            idx = jnp.where(neighbor_idx < 0, 0, neighbor_idx)
            neigh = jnp.take(h, idx, axis=1)  # (B, N, M, F)
            mask = neighbor_mask[None, :, :, None]  # (1, N, M, 1)
            neigh_sum = jnp.sum(neigh * mask, axis=2)
            denom = jnp.maximum(jnp.sum(mask, axis=2), 1.0)
            neigh_mean = neigh_sum / denom  # (B, N, F)
            
            # 每种边类型有独立的线性变换
            msg = nn.Dense(self.hidden_dim, dtype=self.dtype, name=f'edge_proj_{i}')(neigh_mean)
            all_messages.append(msg)
        
        # 聚合所有边类型的消息（求和）
        aggregated = sum(all_messages)
        
        # 残差 MLP
        x = jnp.concatenate([h, aggregated], axis=-1)
        x = nn.LayerNorm(dtype=self.dtype)(x)
        x = nn.Dense(self.hidden_dim, dtype=self.dtype)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, dtype=self.dtype)(x)
        
        return h + x


class AlphaZeroNetwork(nn.Module):
    """多边类型 GNN 网络
    
    设计原则：
    - 多种边类型显式编码象棋规则
    - 相邻边：局部特征
    - 行列边：车/炮攻击线
    - 马步边：马的攻击范围
    - 象步边：象的攻击范围
    - 全局池化输出价值（分布式：64分位数）
    """
    action_space_size: int = ACTION_SPACE_SIZE
    channels: int = 96
    num_blocks: int = 6
    num_quantiles: int = 64  # 分布式价值：分位数数量
    dtype: jnp.dtype = jnp.float32
    
    def setup(self):
        # 构建所有边类型
        edges = _build_all_edges(BOARD_HEIGHT, BOARD_WIDTH)
        edge_types = ['adjacent', 'row', 'col', 'knight', 'elephant']
        
        self.edge_indices = [jnp.array(edges[et]['idx'], dtype=jnp.int32) for et in edge_types]
        self.edge_masks = [jnp.array(edges[et]['mask'], dtype=jnp.float32) for et in edge_types]
        self.num_edge_types = len(edge_types)
    
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
        
        # 多边类型消息传递
        for _ in range(self.num_blocks):
            h = MultiEdgeGraphBlock(
                self.channels, 
                num_edge_types=self.num_edge_types,
                dtype=self.dtype
            )(h, self.edge_indices, self.edge_masks, train)
        
        # 策略头：节点特征 -> 全局动作 logits
        p = nn.Dense(32, dtype=self.dtype)(h)
        p = nn.relu(p)
        p = p.reshape((batch_size, -1))  # [B, 90*32] = [B, 2880]
        p = nn.Dense(self.action_space_size, dtype=self.dtype)(p)
        
        # 价值头：全局池化 -> 分布式价值（64分位数）
        v = jnp.mean(h, axis=1)
        v = nn.Dense(128, dtype=self.dtype)(v)
        v = nn.relu(v)
        v_quantiles = nn.Dense(self.num_quantiles, dtype=self.dtype)(v)
        v_quantiles = jnp.tanh(v_quantiles)  # [-1, 1] 范围
        
        return p.astype(jnp.float32), v_quantiles.astype(jnp.float32)

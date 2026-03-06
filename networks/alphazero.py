"""
AlphaZero network - 精简 4 分支 GNN 架构

分支设计（全部基于静态棋盘拓扑，无动态规则计算）:
- Local: 8 方向邻居注意力（马腿/象眼等短距交互）
- Row: 同行分组注意力（车/炮行攻击，9 节点一组 × 10 行）
- Col: 同列分组注意力（车/炮列攻击，10 节点一组 × 9 列）
- Global: 全局注意力（长距离战术模式，90×90）

关键设计决策:
- 不在网络内计算合法走法/攻击/将军掩码，让网络从原始棋子位置自行学习
- Row/Col 使用分组注意力（reshape 分组）而非 N×N 全量掩码，效率提升 ~10x
- Factorized Policy Head (from/to pair scoring) + Attention-Pooled Value Head
"""

import math
import numpy as np
import jax.numpy as jnp
import flax.linen as nn

from xiangqi.actions import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    ACTION_SPACE_SIZE,
    _ACTION_TO_FROM_SQ,
    _ACTION_TO_TO_SQ,
)


# ============================================================================
# 静态棋盘拓扑
# ============================================================================

def _build_grid_neighbors(height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构建 8 邻居图拓扑（上下左右 + 4 对角）"""
    directions = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    ]
    num_nodes = height * width
    max_deg = len(directions)
    neighbor_idx = np.full((num_nodes, max_deg), -1, dtype=np.int32)
    neighbor_mask = np.zeros((num_nodes, max_deg), dtype=np.float32)
    neighbor_dir = np.zeros((num_nodes, max_deg), dtype=np.int32)

    for r in range(height):
        for c in range(width):
            node = r * width + c
            for dir_id, (dr, dc) in enumerate(directions):
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width:
                    neighbor_idx[node, dir_id] = nr * width + nc
                    neighbor_mask[node, dir_id] = 1.0
                    neighbor_dir[node, dir_id] = dir_id
    return neighbor_idx, neighbor_mask, neighbor_dir


# ============================================================================
# 基础模块
# ============================================================================

class SwiGLU(nn.Module):
    """SwiGLU feed-forward: silu(xW1) * xW2"""
    hidden_dim: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        uv = nn.Dense(2 * self.hidden_dim, dtype=self.dtype)(x)
        u, v = jnp.split(uv, 2, axis=-1)
        return nn.silu(u) * v


# ============================================================================
# 4 分支 Graph Attention Block
# ============================================================================

class GraphBlock(nn.Module):
    """4-branch graph attention + gated FFN residual block

    4 个注意力分支各有明确分工:
    - local: 8 方向邻居，捕捉相邻格交互（马腿、象眼、兵的攻击范围）
    - row: 同行 9 节点分组注意力，捕捉行方向交互（车/炮横向攻击）
    - col: 同列 10 节点分组注意力，捕捉列方向交互（车/炮纵向攻击）
    - global: 90 节点全局注意力，捕捉长距离战术关联（可选）

    use_global=False 时跳过 global 分支，节省 ~12% FLOPs/block。
    gate 使用 sigmoid（各分支独立控制），避免 softmax 导致的分支坍缩。
    """
    hidden_dim: int
    use_global: bool = True
    num_rows: int = BOARD_HEIGHT   # 10
    num_cols: int = BOARD_WIDTH    # 9
    mlp_ratio: float = 2.0
    layer_scale_init: float = 1e-2
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        h: jnp.ndarray,
        neighbor_idx: jnp.ndarray,
        neighbor_mask: jnp.ndarray,
        neighbor_dir: jnp.ndarray,
        train: bool = True,
    ) -> jnp.ndarray:
        del train

        D = self.hidden_dim
        scale = 1.0 / math.sqrt(float(D))

        x = nn.LayerNorm(dtype=self.dtype, name="ln_attn")(h)
        q = nn.Dense(D, use_bias=False, dtype=self.dtype, name="q_proj")(x)
        k = nn.Dense(D, use_bias=False, dtype=self.dtype, name="k_proj")(x)
        v = nn.Dense(D, use_bias=False, dtype=self.dtype, name="v_proj")(x)

        B = q.shape[0]

        # ── Branch 1: Local 8-directional attention ──
        idx = jnp.where(neighbor_idx < 0, 0, neighbor_idx)
        k_neigh = jnp.take(k, idx, axis=1)               # (B, N, 8, D)
        v_neigh = jnp.take(v, idx, axis=1)               # (B, N, 8, D)
        dir_embed = nn.Embed(
            num_embeddings=8, features=D, dtype=self.dtype, name="dir_embed"
        )(jnp.clip(neighbor_dir, 0, 7))[None, :, :, :]   # (1, N, 8, D)

        local_logits = jnp.sum(q[:, :, None, :] * (k_neigh + dir_embed), axis=-1) * scale
        local_logits = jnp.where(
            neighbor_mask[None, :, :] > 0,
            local_logits,
            jnp.finfo(local_logits.dtype).min,
        )
        local_attn = nn.softmax(local_logits, axis=-1)
        local_agg = jnp.sum(local_attn[..., None] * (v_neigh + dir_embed), axis=2)
        local_out = nn.Dense(D, dtype=self.dtype, name="local_out")(local_agg)

        # ── Branch 2: Row grouped attention (10 行 x 9 节点) ──
        qr = q.reshape(B, self.num_rows, self.num_cols, D)
        kr = k.reshape(B, self.num_rows, self.num_cols, D)
        vr = v.reshape(B, self.num_rows, self.num_cols, D)
        row_scores = jnp.einsum("brnd,brmd->brnm", qr, kr) * scale
        row_attn = nn.softmax(row_scores, axis=-1)
        row_agg = jnp.einsum("brnm,brmd->brnd", row_attn, vr).reshape(B, -1, D)
        row_out = nn.Dense(D, dtype=self.dtype, name="row_out")(row_agg)

        # ── Branch 3: Col grouped attention (9 列 x 10 节点) ──
        qc = q.reshape(B, self.num_rows, self.num_cols, D).transpose(0, 2, 1, 3)
        kc = k.reshape(B, self.num_rows, self.num_cols, D).transpose(0, 2, 1, 3)
        vc = v.reshape(B, self.num_rows, self.num_cols, D).transpose(0, 2, 1, 3)
        col_scores = jnp.einsum("bcnd,bcmd->bcnm", qc, kc) * scale
        col_attn = nn.softmax(col_scores, axis=-1)
        col_agg = jnp.einsum("bcnm,bcmd->bcnd", col_attn, vc)
        col_agg = col_agg.transpose(0, 2, 1, 3).reshape(B, -1, D)
        col_out = nn.Dense(D, dtype=self.dtype, name="col_out")(col_agg)

        # ── Branch 4: Global attention (90 x 90, 可选) ──
        if self.use_global:
            global_scores = jnp.einsum("bnd,bmd->bnm", q, k) * scale
            global_attn = nn.softmax(global_scores, axis=-1)
            global_agg = jnp.einsum("bnm,bmd->bnd", global_attn, v)
            global_out = nn.Dense(D, dtype=self.dtype, name="global_out")(global_agg)

            branches = jnp.stack([local_out, row_out, col_out, global_out], axis=2)
            num_branches = 4
        else:
            branches = jnp.stack([local_out, row_out, col_out], axis=2)
            num_branches = 3

        # ── Gated mix (sigmoid: 各分支独立控制，防止坍缩) ──
        gate_logits = nn.Dense(num_branches, dtype=self.dtype, name="rel_gate")(x)
        gates = nn.sigmoid(gate_logits)
        agg = jnp.sum(branches * gates[..., None], axis=2)
        agg = nn.Dense(D, dtype=self.dtype, name="mix_out")(agg)

        # ── Residual + layer scale ──
        gamma1 = self.param(
            "gamma1", nn.initializers.constant(self.layer_scale_init), (D,)
        )
        h = h + gamma1[None, None, :] * agg

        # ── FFN ──
        x = nn.LayerNorm(dtype=self.dtype, name="ln_ffn")(h)
        ffn_dim = max(int(D * self.mlp_ratio), D)
        x = SwiGLU(hidden_dim=ffn_dim, dtype=self.dtype, name="ffn_in")(x)
        x = nn.Dense(D, dtype=self.dtype, name="ffn_out")(x)
        gamma2 = self.param(
            "gamma2", nn.initializers.constant(self.layer_scale_init), (D,)
        )
        return h + gamma2[None, None, :] * x


# ============================================================================
# Policy Head
# ============================================================================

class PolicyHead(nn.Module):
    """Factorized from/to policy head

    将动作分解为 (from_square, to_square) 对，用点积评分:
    score(a) = base_factorized(a) + correction_delta(a) + global

    correction 分支为小型残差打分器，补偿纯 from/to 因子化的表达上限。
    """
    action_space_size: int
    model_dim: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, h: jnp.ndarray, from_idx: jnp.ndarray, to_idx: jnp.ndarray) -> jnp.ndarray:
        x = nn.LayerNorm(dtype=self.dtype)(h)
        proj_dim = max(self.model_dim // 2, 64)
        q_from = nn.Dense(proj_dim, use_bias=False, dtype=self.dtype, name="from_proj")(x)
        k_to = nn.Dense(proj_dim, use_bias=False, dtype=self.dtype, name="to_proj")(x)

        from_bias = nn.Dense(1, dtype=self.dtype, name="from_bias")(x).squeeze(-1)
        to_bias = nn.Dense(1, dtype=self.dtype, name="to_bias")(x).squeeze(-1)

        pair_scores = jnp.einsum("bnd,bmd->bnm", q_from, k_to) * (1.0 / math.sqrt(float(proj_dim)))
        logits = (
            pair_scores[:, from_idx, to_idx]
            + from_bias[:, from_idx]
            + to_bias[:, to_idx]
        )

        # Lightweight correction branch: delta(a) from projected (from,to) node features.
        corr_dim = max(self.model_dim // 16, 8)
        corr_from = nn.Dense(corr_dim, use_bias=False, dtype=self.dtype, name="corr_from_proj")(x)
        corr_to = nn.Dense(corr_dim, use_bias=False, dtype=self.dtype, name="corr_to_proj")(x)
        corr_from_act = corr_from[:, from_idx, :]
        corr_to_act = corr_to[:, to_idx, :]
        corr_feat = jnp.concatenate(
            [corr_from_act * corr_to_act, jnp.abs(corr_from_act - corr_to_act)], axis=-1
        )
        corr_hidden = nn.Dense(corr_dim, dtype=self.dtype, name="corr_fc1")(corr_feat)
        corr_hidden = nn.silu(corr_hidden)
        corr_delta = nn.Dense(1, dtype=self.dtype, name="corr_fc2")(corr_hidden).squeeze(-1)
        corr_scale = self.param("corr_scale", nn.initializers.constant(0.0), ())
        logits = logits + corr_scale * corr_delta

        global_ctx = jnp.mean(x, axis=1)
        logits = logits + nn.Dense(self.action_space_size, dtype=self.dtype, name="global_proj")(global_ctx)
        return logits


# ============================================================================
# Value Head
# ============================================================================

class ValueHead(nn.Module):
    """WDL (Win/Draw/Loss) value head

    三路池化（attention + mean + max）→ MLP → 3-class softmax
    输出: (value_scalar, wdl_logits)
      - value_scalar: W - L ∈ [-1, 1]，供 MCTS 使用
      - wdl_logits: (B, 3) 原始 logits，供交叉熵 loss 使用
    """
    model_dim: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, h: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        x = nn.LayerNorm(dtype=self.dtype)(h)
        pool_logits = nn.Dense(1, dtype=self.dtype, name="pool_logits")(x).squeeze(-1)
        pool_weights = nn.softmax(pool_logits, axis=1)
        pooled = jnp.sum(pool_weights[..., None] * x, axis=1)
        mean_pooled = jnp.mean(x, axis=1)
        max_pooled = jnp.max(x, axis=1)

        fused = jnp.concatenate([pooled, mean_pooled, max_pooled], axis=-1)
        fused = nn.Dense(self.model_dim * 2, dtype=self.dtype, name="fc1")(fused)
        fused = nn.silu(fused)
        fused = nn.Dense(self.model_dim, dtype=self.dtype, name="fc2")(fused)
        fused = nn.silu(fused)

        wdl_logits = nn.Dense(3, dtype=self.dtype, name="wdl_out")(fused)  # (B, 3)
        wdl_probs = nn.softmax(wdl_logits, axis=-1)
        value = wdl_probs[:, 0] - wdl_probs[:, 2]  # W - L
        return value, wdl_logits


# ============================================================================
# 主网络
# ============================================================================

class AlphaZeroNetwork(nn.Module):
    """精简 4 分支 GNN AlphaZero 网络

    输入: (B, C, H, W) 观察张量（126 通道 = 9 帧 × 14 棋子类型）
    输出: (policy_logits, value, wdl_logits)
        - policy_logits: (B, ACTION_SPACE_SIZE)
        - value: (B,) in [-1, 1]  (= W - L，供 MCTS 使用)
        - wdl_logits: (B, 3) 原始 logits (供训练 cross-entropy loss)
    """
    action_space_size: int = ACTION_SPACE_SIZE
    channels: int = 128
    num_blocks: int = 8
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        neighbor_idx, neighbor_mask, neighbor_dir = _build_grid_neighbors(BOARD_HEIGHT, BOARD_WIDTH)
        self.neighbor_idx = jnp.array(neighbor_idx, dtype=jnp.int32)
        self.neighbor_mask = jnp.array(neighbor_mask, dtype=jnp.float32)
        self.neighbor_dir = jnp.array(neighbor_dir, dtype=jnp.int32)
        self.action_from_idx = jnp.array(_ACTION_TO_FROM_SQ, dtype=jnp.int32)
        self.action_to_idx = jnp.array(_ACTION_TO_TO_SQ, dtype=jnp.int32)

    @nn.compact
    def __call__(self, x: jnp.ndarray, train: bool = True) -> tuple[jnp.ndarray, jnp.ndarray]:
        if x.ndim != 4:
            raise ValueError(f"输入必须是 4D (B,C,H,W), 实际 ndim={x.ndim}")
        if x.shape[2] != BOARD_HEIGHT or x.shape[3] != BOARD_WIDTH:
            raise ValueError(
                f"棋盘尺寸不匹配, 期望 ({BOARD_HEIGHT},{BOARD_WIDTH}), "
                f"实际 ({x.shape[2]},{x.shape[3]})"
            )

        x = x.astype(self.dtype)
        x = jnp.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC, (B, H, W, 126)
        batch_size = x.shape[0]
        num_nodes = BOARD_HEIGHT * BOARD_WIDTH

        # 输入因子化编码: piece(role+side) embedding + frame embedding
        # 目标: 让同类棋子共享统计结构，提高跨局面泛化与样本效率
        num_frames = 9
        channels_per_frame = 14
        if x.shape[-1] != num_frames * channels_per_frame:
            raise ValueError(
                f"通道数不匹配，期望 {num_frames * channels_per_frame}，实际 {x.shape[-1]}"
            )

        piece_embed_dim = max(self.channels // 8, 16)
        role_embed = self.param(
            "piece_role_embed",
            nn.initializers.normal(stddev=0.02),
            (7, piece_embed_dim),
        )
        side_embed = self.param(
            "piece_side_embed",
            nn.initializers.normal(stddev=0.02),
            (2, piece_embed_dim),
        )
        piece_channel_bias = self.param(
            "piece_channel_bias",
            nn.initializers.zeros,
            (channels_per_frame, piece_embed_dim),
        )
        # Must match env.observe channel order:
        # [R_KING..R_PAWN, B_PAWN..B_KING]
        # Use local constants + jnp.take to avoid tracer->numpy conversion in nested JAX loops.
        piece_role_idx = jnp.asarray(
            [0, 1, 2, 3, 4, 5, 6, 6, 5, 4, 3, 2, 1, 0], dtype=jnp.int32
        )
        piece_side_idx = jnp.asarray(
            [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=jnp.int32
        )
        piece_embed_table = (
            jnp.take(role_embed, piece_role_idx, axis=0)
            + jnp.take(side_embed, piece_side_idx, axis=0)
            + piece_channel_bias
        ).astype(self.dtype)  # (14, Dp)

        x = x.reshape((batch_size, BOARD_HEIGHT, BOARD_WIDTH, num_frames, channels_per_frame))

        # 历史帧时间权重：9 帧各一个可学习标量，初始化为指数衰减
        # 帧顺序: [当前, t-1, t-2, ..., t-8]，越近权重越大
        frame_weight_raw = self.param(
            "frame_time_weight",
            lambda key, shape: jnp.linspace(0.0, -1.5, num_frames),
            (num_frames,),
        )
        frame_weight = nn.softplus(frame_weight_raw).astype(self.dtype)
        frame_embed = self.param(
            "frame_embed",
            nn.initializers.normal(stddev=0.02),
            (num_frames, piece_embed_dim),
        ).astype(self.dtype)

        # (B, H, W, F, 14) x (14, Dp) -> (B, H, W, F, Dp)
        piece_feat = jnp.einsum("bhwfp,pd->bhwfd", x, piece_embed_table)
        piece_presence = jnp.sum(x, axis=-1, keepdims=True)  # 0/1 occupancy per frame
        piece_feat = piece_feat + piece_presence * frame_embed[None, None, None, :, :]
        piece_feat = piece_feat * frame_weight[None, None, None, :, None]

        # 保留时间维度再投影，避免直接求和造成时序信息丢失
        h = piece_feat.reshape((batch_size, num_nodes, num_frames * piece_embed_dim))
        h = nn.Dense(self.channels, dtype=self.dtype, name="input_proj")(h)

        pos_embed = self.param(
            "pos_embed",
            nn.initializers.normal(stddev=0.02),
            (num_nodes, self.channels),
        )
        h = h + pos_embed[None, :, :]
        h = nn.silu(h)

        half = self.num_blocks // 2
        for i in range(self.num_blocks):
            h = GraphBlock(
                hidden_dim=self.channels,
                use_global=(i >= half),
                dtype=self.dtype,
            )(
                h,
                self.neighbor_idx,
                self.neighbor_mask,
                self.neighbor_dir,
                train=train,
            )

        policy_logits = PolicyHead(
            action_space_size=self.action_space_size,
            model_dim=self.channels,
            dtype=self.dtype,
        )(h, self.action_from_idx, self.action_to_idx)

        value, wdl_logits = ValueHead(model_dim=self.channels, dtype=self.dtype)(h)
        return (
            policy_logits.astype(jnp.float32),
            value.astype(jnp.float32),
            wdl_logits.astype(jnp.float32),
        )

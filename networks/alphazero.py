"""
AlphaZero network - 精简 3 分支 GNN 架构 + 全局动作先验

分支设计（全部基于静态棋盘拓扑，无动态规则计算）:
- Local: 8 方向邻居（上下左右+4对角），负责短程交互
- Row: 同行分组注意力（车/炮横向），并对空位做 occupancy mask
- Col: 同列分组注意力（车/炮纵向），并对空位做 occupancy mask

关键设计决策:
- 主干去掉 Global attention，Row/Col 多跳覆盖大部分长距离关系
- Policy head 保留 global action bias，用局面级动作先验补足 from/to 分解表达力
- Local 仅 8 邻居，节省计算，自对弈更快
- 不在网络内计算合法走法，让网络从原始棋子位置自行学习
- Factorized Policy Head + global action bias + 轻量 pair/span correction + Attention-Pooled Value Head
"""

import math
import numpy as np
import jax.numpy as jnp
import flax.linen as nn

from xiangqi.actions import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    ACTION_SPACE_SIZE,
    _ACTION_SPAN_EMBED_IDX,
    _ACTION_TO_FROM_SQ,
    _ACTION_TO_TO_SQ,
)
from xiangqi.env import BOARD_OBSERVATION_CHANNELS, NUM_RULE_STATE_CHANNELS


# ============================================================================
# 静态棋盘拓扑
# ============================================================================

def _build_grid_neighbors(height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """构建 8 邻居图拓扑：上下左右 + 4 对角

    马/象走法由 Row+Col 多跳传播学习，不显式建模。
    """
    num_nodes = height * width
    direct_dirs = [
        (-1, 0), (1, 0), (0, -1), (0, 1),
        (-1, -1), (-1, 1), (1, -1), (1, 1),
    ]
    max_deg = len(direct_dirs)  # 8
    neighbor_idx = np.full((num_nodes, max_deg), -1, dtype=np.int32)
    neighbor_mask = np.zeros((num_nodes, max_deg), dtype=np.float32)
    neighbor_dir = np.zeros((num_nodes, max_deg), dtype=np.int32)

    for r in range(height):
        for c in range(width):
            node = r * width + c
            for dir_id, (dr, dc) in enumerate(direct_dirs):
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width:
                    neighbor_idx[node, dir_id] = nr * width + nc
                    neighbor_mask[node, dir_id] = 1.0
                    neighbor_dir[node, dir_id] = dir_id
    return neighbor_idx, neighbor_mask, neighbor_dir


def _build_region_ids(height: int, width: int) -> np.ndarray:
    """构建棋盘区域 ID：九宫、河界、半场

    区域定义（中国象棋）:
    - 0: 红九宫 (r in [0,1,2], c in [3,4,5])
    - 1: 黑九宫 (r in [7,8,9], c in [3,4,5])
    - 2: 河界 (r in [4,5]，楚河汉界)
    - 3: 红方半场 (r<=3 且非红九宫)
    - 4: 黑方半场 (r>=6 且非黑九宫)
    """
    num_nodes = height * width
    region_id = np.zeros(num_nodes, dtype=np.int32)
    for r in range(height):
        for c in range(width):
            node = r * width + c
            if r <= 2 and 3 <= c <= 5:
                region_id[node] = 0  # 红九宫
            elif r >= 7 and 3 <= c <= 5:
                region_id[node] = 1  # 黑九宫
            elif r in (4, 5):
                region_id[node] = 2  # 河界
            elif r <= 3:
                region_id[node] = 3  # 红方半场
            else:
                region_id[node] = 4  # 黑方半场
    return region_id


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
# 3 分支 Graph Attention Block（无 Global）
# ============================================================================

class GraphBlock(nn.Module):
    """3-branch graph attention + gated FFN residual block

    3 个注意力分支:
    - local: 8 邻居（上下左右+4对角），兵/士/将等短距
    - row: 同行 9 节点分组注意力（车/炮横向）
    - col: 同列 10 节点分组注意力（车/炮纵向）

    gate 使用 sigmoid，FFN 含 dropout（仅训练时生效）。
    """
    hidden_dim: int
    num_rows: int = BOARD_HEIGHT   # 10
    num_cols: int = BOARD_WIDTH    # 9
    mlp_ratio: float = 2.0
    layer_scale_init: float = 1e-2
    dropout_rate: float = 0.05
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        h: jnp.ndarray,
        neighbor_idx: jnp.ndarray,
        neighbor_mask: jnp.ndarray,
        neighbor_dir: jnp.ndarray,
        node_occupancy: jnp.ndarray,
        train: bool = True,
    ) -> jnp.ndarray:

        D = self.hidden_dim
        scale = 1.0 / math.sqrt(float(D))
        max_deg = neighbor_idx.shape[1]  # 8

        x = nn.LayerNorm(dtype=self.dtype, name="ln_attn")(h)
        q = nn.Dense(D, use_bias=False, dtype=self.dtype, name="q_proj")(x)
        k = nn.Dense(D, use_bias=False, dtype=self.dtype, name="k_proj")(x)
        v = nn.Dense(D, use_bias=False, dtype=self.dtype, name="v_proj")(x)

        B = q.shape[0]

        # ── Branch 1: Local attention (8 邻居) ──
        idx = jnp.where(neighbor_idx < 0, 0, neighbor_idx)
        k_neigh = jnp.take(k, idx, axis=1)               # (B, N, 8, D)
        v_neigh = jnp.take(v, idx, axis=1)               # (B, N, 8, D)
        dir_embed = nn.Embed(
            num_embeddings=max_deg, features=D, dtype=self.dtype, name="dir_embed"
        )(jnp.clip(neighbor_dir, 0, max_deg - 1))[None, :, :, :]   # (1, N, 20, D)

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
        row_occ = node_occupancy.reshape(B, self.num_rows, self.num_cols).astype(jnp.bool_)
        row_self = jnp.eye(self.num_cols, dtype=jnp.bool_)[None, None, :, :]
        row_key_mask = row_occ[:, :, None, :] | row_self
        row_scores = jnp.einsum("brnd,brmd->brnm", qr, kr) * scale
        row_scores = jnp.where(
            row_key_mask,
            row_scores,
            jnp.finfo(row_scores.dtype).min,
        )
        row_attn = nn.softmax(row_scores, axis=-1)
        row_agg = jnp.einsum("brnm,brmd->brnd", row_attn, vr).reshape(B, -1, D)
        row_out = nn.Dense(D, dtype=self.dtype, name="row_out")(row_agg)

        # ── Branch 3: Col grouped attention (9 列 x 10 节点) ──
        qc = q.reshape(B, self.num_rows, self.num_cols, D).transpose(0, 2, 1, 3)
        kc = k.reshape(B, self.num_rows, self.num_cols, D).transpose(0, 2, 1, 3)
        vc = v.reshape(B, self.num_rows, self.num_cols, D).transpose(0, 2, 1, 3)
        col_occ = row_occ.transpose(0, 2, 1)
        col_self = jnp.eye(self.num_rows, dtype=jnp.bool_)[None, None, :, :]
        col_key_mask = col_occ[:, :, None, :] | col_self
        col_scores = jnp.einsum("bcnd,bcmd->bcnm", qc, kc) * scale
        col_scores = jnp.where(
            col_key_mask,
            col_scores,
            jnp.finfo(col_scores.dtype).min,
        )
        col_attn = nn.softmax(col_scores, axis=-1)
        col_agg = jnp.einsum("bcnm,bcmd->bcnd", col_attn, vc)
        col_agg = col_agg.transpose(0, 2, 1, 3).reshape(B, -1, D)
        col_out = nn.Dense(D, dtype=self.dtype, name="col_out")(col_agg)

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

        # ── FFN（含 dropout 提升泛化）──
        x = nn.LayerNorm(dtype=self.dtype, name="ln_ffn")(h)
        ffn_dim = max(int(D * self.mlp_ratio), D)
        x = SwiGLU(hidden_dim=ffn_dim, dtype=self.dtype, name="ffn_in")(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
        x = nn.Dense(D, dtype=self.dtype, name="ffn_out")(x)
        gamma2 = self.param(
            "gamma2", nn.initializers.constant(self.layer_scale_init), (D,)
        )
        return h + gamma2[None, None, :] * x


# ============================================================================
# Policy Head
# ============================================================================

class PolicyHead(nn.Module):
    """Factorized from/to policy head.

    将动作分解为 (from_square, to_square) 对，用点积评分:
    score(a) = base_factorized(a) + correction_delta(a) + global + grid_span_ctx(a)
    其中 grid_span_ctx 为「跨度」嵌入（max(|Δr|,|Δc|)）与全局上下文的点积，显式感知格子距离。
    global_proj 保留为局面级动作先验；grid_span_ctx 作为可学习的轻量补充。
    """
    action_space_size: int
    model_dim: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        h: jnp.ndarray,
        from_idx: jnp.ndarray,
        to_idx: jnp.ndarray,
        action_span_embed_idx: jnp.ndarray,
    ) -> jnp.ndarray:
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

        # Small pair-specific correction branch to improve tactical ranking.
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

        # 格子跨度感知：9 档 (跨度 1..9) 与局面级向量点积，按动作索引 gather
        # caae0ef 强版本中的全局动作偏置。它给每个 action 一个局面级自由度，
        # 对开局偏好和长程战术先验很重要；低秩 from/to 分解不能完全替代。
        global_ctx = jnp.mean(x, axis=1)
        logits = logits + nn.Dense(self.action_space_size, dtype=self.dtype, name="global_proj")(global_ctx)

        span_dim = max(self.model_dim // 8, 16)
        grid_ctx = nn.Dense(span_dim, dtype=self.dtype, name="grid_span_ctx_proj")(
            jnp.mean(x, axis=1)
        )
        span_embed = nn.Embed(
            num_embeddings=max(BOARD_HEIGHT, BOARD_WIDTH) - 1,
            features=span_dim,
            dtype=self.dtype,
            name="grid_span_embed",
        )(action_span_embed_idx)
        span_scale = self.param(
            "grid_span_scale",
            nn.initializers.constant(0.0),
            (),
        )
        logits = logits + span_scale * jnp.einsum("bd,ad->ba", grid_ctx, span_embed)
        return logits


# ============================================================================
# Value Head
# ============================================================================

class ValueHead(nn.Module):
    """标量价值头：Win / Draw / Loss 三个 logits 参数化局面，value = p_W − p_L（∈ [-1,1]）。

    四路池化（attention + mean + max + std）→ MLP → Dense(3)。
    训练只对 **标量 value** 与 **value_tgt**（MCTS 根价值 TD(λ)）做 MSE，梯度回传至三 logits；**不做**三分类交叉熵。
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
        std_pooled = jnp.std(x, axis=1)

        fused = jnp.concatenate([pooled, mean_pooled, max_pooled, std_pooled], axis=-1)
        fused = nn.Dense(self.model_dim * 2, dtype=self.dtype, name="fc1")(fused)
        fused = nn.silu(fused)
        fused = nn.Dense(self.model_dim, dtype=self.dtype, name="fc2")(fused)
        fused = nn.silu(fused)

        value_logits = nn.Dense(3, dtype=self.dtype, name="value_logits")(fused)  # (B, 3)
        probs = nn.softmax(value_logits, axis=-1)
        value = probs[:, 0] - probs[:, 2]  # W - L
        return value, value_logits


# ============================================================================
# Auxiliary Heads
# ============================================================================

class AuxiliaryHeads(nn.Module):
    """Training-only auxiliary predictions from the shared board trunk."""
    model_dim: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, h: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        x = nn.LayerNorm(dtype=self.dtype)(h)
        pooled = jnp.concatenate([jnp.mean(x, axis=1), jnp.max(x, axis=1)], axis=-1)
        pooled = nn.Dense(self.model_dim, dtype=self.dtype, name="aux_fc1")(pooled)
        pooled = nn.silu(pooled)

        remaining_logits = nn.Dense(5, dtype=self.dtype, name="remaining_ply_logits")(pooled)
        material_delta = nn.Dense(1, dtype=self.dtype, name="future_material_delta")(pooled).squeeze(-1)

        node_hidden = nn.Dense(max(self.model_dim // 2, 32), dtype=self.dtype, name="occupancy_fc1")(x)
        node_hidden = nn.silu(node_hidden)
        occupancy_logits = nn.Dense(1, dtype=self.dtype, name="future_occupancy_change")(node_hidden).squeeze(-1)
        return remaining_logits, material_delta, occupancy_logits


# ============================================================================
# 主网络
# ============================================================================

class AlphaZeroNetwork(nn.Module):
    """精简 3 分支 GNN AlphaZero 网络（Local 8 邻居 + Row + Col，无 Global）

    输入: (B, C, H, W) 观察张量（126 棋盘通道 + 10 规则状态通道）
    输出: (policy_logits, value, value_logits)
        - policy_logits: (B, ACTION_SPACE_SIZE)
        - value: (B,) in [-1, 1]，由 value_logits 经 softmax 得 p_W、p_L 后 value = p_W − p_L
        - value_logits: (B, 3)；价值损失为 MSE(value, value_tgt)，不单独对 logits 做 CE
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
        self.region_id = jnp.array(_build_region_ids(BOARD_HEIGHT, BOARD_WIDTH), dtype=jnp.int32)
        self.action_from_idx = jnp.array(_ACTION_TO_FROM_SQ, dtype=jnp.int32)
        self.action_to_idx = jnp.array(_ACTION_TO_TO_SQ, dtype=jnp.int32)
        self.action_span_embed_idx = jnp.array(_ACTION_SPAN_EMBED_IDX, dtype=jnp.int32)

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, train: bool = True, return_aux: bool = True
    ) -> tuple[jnp.ndarray, ...]:
        """返回 policy/value；return_aux=True 时附带训练用辅助头。"""
        if x.ndim != 4:
            raise ValueError(f"输入必须是 4D (B,C,H,W), 实际 ndim={x.ndim}")
        if x.shape[2] != BOARD_HEIGHT or x.shape[3] != BOARD_WIDTH:
            raise ValueError(
                f"棋盘尺寸不匹配, 期望 ({BOARD_HEIGHT},{BOARD_WIDTH}), "
                f"实际 ({x.shape[2]},{x.shape[3]})"
            )

        x = x.astype(self.dtype)
        x = jnp.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
        batch_size = x.shape[0]
        num_nodes = BOARD_HEIGHT * BOARD_WIDTH

        # 输入因子化编码: piece(role+side) embedding + frame embedding
        # 目标: 让同类棋子共享统计结构，提高跨局面泛化与样本效率
        num_frames = 9
        channels_per_frame = 14
        expected_channels = BOARD_OBSERVATION_CHANNELS + NUM_RULE_STATE_CHANNELS
        if x.shape[-1] != expected_channels:
            raise ValueError(
                f"通道数不匹配，期望 {expected_channels}，实际 {x.shape[-1]}"
            )
        board_x = x[..., :BOARD_OBSERVATION_CHANNELS]
        rule_x = x[..., BOARD_OBSERVATION_CHANNELS:]

        piece_embed_dim = max(self.channels // 4, 24)  # 增大以更好区分棋子类型
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

        board_x = board_x.reshape((batch_size, BOARD_HEIGHT, BOARD_WIDTH, num_frames, channels_per_frame))
        node_occupancy = (jnp.sum(board_x[:, :, :, 0, :], axis=-1) > 0).reshape(batch_size, num_nodes)

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
        piece_feat = jnp.einsum("bhwfp,pd->bhwfd", board_x, piece_embed_table)
        piece_presence = jnp.sum(board_x, axis=-1, keepdims=True)  # 0/1 occupancy per frame
        piece_feat = piece_feat + piece_presence * frame_embed[None, None, None, :, :]
        piece_feat = piece_feat * frame_weight[None, None, None, :, None]

        # 保留时间维度再投影，避免直接求和造成时序信息丢失
        h = piece_feat.reshape((batch_size, num_nodes, num_frames * piece_embed_dim))
        h = nn.Dense(self.channels, dtype=self.dtype, name="input_proj")(h)
        rule_vec = jnp.mean(rule_x, axis=(1, 2))
        rule_feat = nn.Dense(self.channels, dtype=self.dtype, name="rule_state_proj")(rule_vec)
        h = h + rule_feat[:, None, :]

        pos_embed = self.param(
            "pos_embed",
            nn.initializers.normal(stddev=0.02),
            (num_nodes, self.channels),
        )
        region_embed = self.param(
            "region_embed",
            nn.initializers.normal(stddev=0.02),
            (5, self.channels),  # 红九宫/黑九宫/河界/红半场/黑半场
        )
        # Use jnp.take instead of advanced indexing to avoid tracer->numpy conversion
        # under JIT/fori_loop (e.g. mctx search recurrent_fn).
        region_bias = jnp.take(region_embed, self.region_id, axis=0)[None, :, :]
        h = h + pos_embed[None, :, :] + region_bias
        h = nn.silu(h)

        for i in range(self.num_blocks):
            h = GraphBlock(
                hidden_dim=self.channels,
                dropout_rate=0.05,
                dtype=self.dtype,
            )(
                h,
                self.neighbor_idx,
                self.neighbor_mask,
                self.neighbor_dir,
                node_occupancy,
                train=train,
            )

        policy_logits = PolicyHead(
            action_space_size=self.action_space_size,
            model_dim=self.channels,
            dtype=self.dtype,
        )(h, self.action_from_idx, self.action_to_idx, self.action_span_embed_idx)

        value, value_logits = ValueHead(model_dim=self.channels, dtype=self.dtype)(h)
        if not return_aux:
            return (
                policy_logits.astype(jnp.float32),
                value.astype(jnp.float32),
            )

        remaining_logits, material_delta, occupancy_logits = AuxiliaryHeads(
            model_dim=self.channels,
            dtype=self.dtype,
        )(h)
        return (
            policy_logits.astype(jnp.float32),
            value.astype(jnp.float32),
            value_logits.astype(jnp.float32),
            remaining_logits.astype(jnp.float32),
            material_delta.astype(jnp.float32),
            occupancy_logits.astype(jnp.float32),
        )

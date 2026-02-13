"""
AlphaZero network - upgraded graph architecture.

Key upgrades:
- Edge-aware graph attention blocks with directional embeddings
- Factorized policy head (from/to pair scoring) aligned to action encoding
- Attention-pooled value head for better global state estimation
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


def _build_grid_neighbors(height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build 8-neighbor graph with directional ids for each edge slot."""
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


class SwiGLU(nn.Module):
    """SwiGLU feed-forward unit."""
    hidden_dim: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        uv = nn.Dense(2 * self.hidden_dim, dtype=self.dtype)(x)
        u, v = jnp.split(uv, 2, axis=-1)
        return nn.silu(u) * v


class GraphBlock(nn.Module):
    """Edge-aware graph attention + gated feed-forward residual block."""
    hidden_dim: int
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
        del train  # reserved for future dropout/stochastic-depth extension

        idx = jnp.where(neighbor_idx < 0, 0, neighbor_idx)
        valid_mask = neighbor_mask > 0.0

        # Neighborhood attention with directional embeddings.
        x = nn.LayerNorm(dtype=self.dtype, name="ln_attn")(h)
        q = nn.Dense(self.hidden_dim, use_bias=False, dtype=self.dtype, name="q_proj")(x)
        k = nn.Dense(self.hidden_dim, use_bias=False, dtype=self.dtype, name="k_proj")(x)
        v = nn.Dense(self.hidden_dim, use_bias=False, dtype=self.dtype, name="v_proj")(x)

        k_neigh = jnp.take(k, idx, axis=1)  # (B, N, M, D)
        v_neigh = jnp.take(v, idx, axis=1)  # (B, N, M, D)
        dir_embed = nn.Embed(
            num_embeddings=8, features=self.hidden_dim, dtype=self.dtype, name="dir_embed"
        )(jnp.clip(neighbor_dir, 0, 7))  # (N, M, D)
        dir_embed = dir_embed[None, :, :, :]  # (1, N, M, D)

        scale = 1.0 / math.sqrt(float(self.hidden_dim))
        attn_logits = jnp.sum(q[:, :, None, :] * (k_neigh + dir_embed), axis=-1) * scale
        neg_inf = jnp.finfo(attn_logits.dtype).min
        attn_logits = jnp.where(valid_mask[None, :, :], attn_logits, neg_inf)
        attn = nn.softmax(attn_logits, axis=-1)

        agg = jnp.sum(attn[..., None] * (v_neigh + dir_embed), axis=2)
        attn_out = nn.Dense(self.hidden_dim, dtype=self.dtype, name="attn_out")(agg)
        gamma1 = self.param(
            "gamma1", nn.initializers.constant(self.layer_scale_init), (self.hidden_dim,)
        )
        h = h + gamma1[None, None, :] * attn_out

        # Gated feed-forward.
        x = nn.LayerNorm(dtype=self.dtype, name="ln_ffn")(h)
        ffn_dim = max(int(self.hidden_dim * self.mlp_ratio), self.hidden_dim)
        x = SwiGLU(hidden_dim=ffn_dim, dtype=self.dtype, name="ffn_in")(x)
        x = nn.Dense(self.hidden_dim, dtype=self.dtype, name="ffn_out")(x)
        gamma2 = self.param(
            "gamma2", nn.initializers.constant(self.layer_scale_init), (self.hidden_dim,)
        )
        return h + gamma2[None, None, :] * x


class PolicyHead(nn.Module):
    """Factorized from/to policy head mapped onto compressed action space."""
    action_space_size: int
    model_dim: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, h: jnp.ndarray, from_idx: jnp.ndarray, to_idx: jnp.ndarray) -> jnp.ndarray:
        x = nn.LayerNorm(dtype=self.dtype)(h)
        proj_dim = max(self.model_dim // 2, 64)
        q_from = nn.Dense(proj_dim, use_bias=False, dtype=self.dtype, name="from_proj")(x)
        k_to = nn.Dense(proj_dim, use_bias=False, dtype=self.dtype, name="to_proj")(x)

        from_bias = nn.Dense(1, dtype=self.dtype, name="from_bias")(x).squeeze(-1)  # (B, N)
        to_bias = nn.Dense(1, dtype=self.dtype, name="to_bias")(x).squeeze(-1)      # (B, N)

        pair_scores = jnp.einsum("bnd,bmd->bnm", q_from, k_to) * (1.0 / math.sqrt(float(proj_dim)))

        logits = (
            pair_scores[:, from_idx, to_idx]
            + from_bias[:, from_idx]
            + to_bias[:, to_idx]
        )

        # Add global prior correction.
        global_ctx = jnp.mean(x, axis=1)
        logits = logits + nn.Dense(self.action_space_size, dtype=self.dtype, name="global_proj")(global_ctx)
        return logits


class ValueHead(nn.Module):
    """Attention pooled value head."""
    model_dim: int
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, h: jnp.ndarray) -> jnp.ndarray:
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
        value = nn.Dense(1, dtype=self.dtype, name="out")(fused)
        return jnp.tanh(value).squeeze(-1)


class AlphaZeroNetwork(nn.Module):
    """Upgraded graph AlphaZero network."""
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
            raise ValueError(f"GNN input must be 4D (B,C,H,W), got ndim={x.ndim}")
        if x.shape[2] != BOARD_HEIGHT or x.shape[3] != BOARD_WIDTH:
            raise ValueError(
                f"GNN board size mismatch, expected ({BOARD_HEIGHT},{BOARD_WIDTH}), "
                f"got ({x.shape[2]},{x.shape[3]})"
            )

        x = x.astype(self.dtype)
        x = jnp.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
        batch_size = x.shape[0]
        num_nodes = BOARD_HEIGHT * BOARD_WIDTH

        h = x.reshape((batch_size, num_nodes, x.shape[-1]))
        h = nn.Dense(self.channels, dtype=self.dtype)(h)  # keep Dense_0 key for ckpt tooling

        pos_embed = self.param(
            "pos_embed",
            nn.initializers.normal(stddev=0.02),
            (num_nodes, self.channels),
        )
        h = h + pos_embed[None, :, :]
        h = nn.silu(h)

        for _ in range(self.num_blocks):
            h = GraphBlock(hidden_dim=self.channels, dtype=self.dtype)(
                h, self.neighbor_idx, self.neighbor_mask, self.neighbor_dir, train=train
            )

        policy_logits = PolicyHead(
            action_space_size=self.action_space_size,
            model_dim=self.channels,
            dtype=self.dtype,
        )(h, self.action_from_idx, self.action_to_idx)

        value = ValueHead(model_dim=self.channels, dtype=self.dtype)(h)

        return policy_logits.astype(jnp.float32), value.astype(jnp.float32)

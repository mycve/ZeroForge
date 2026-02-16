"""
AlphaZero network - upgraded graph architecture.

Key upgrades:
- Multi-relation graph attention (local + rank + file + global)
- Factorized policy head (from/to pair scoring) aligned to action encoding
- Attention pooled scalar value head
"""

import math
import numpy as np
import jax
import jax.numpy as jnp
import flax.linen as nn
from xiangqi.rules import get_legal_moves_mask

from xiangqi.actions import (
    BOARD_HEIGHT,
    BOARD_WIDTH,
    ACTION_SPACE_SIZE,
    _ACTION_TO_FROM_SQ,
    _ACTION_TO_TO_SQ,
    rotate_action,
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


def _build_rank_file_masks(height: int, width: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build same-rank/same-file/global relation masks."""
    num_nodes = height * width
    row_mask = np.zeros((num_nodes, num_nodes), dtype=np.bool_)
    col_mask = np.zeros((num_nodes, num_nodes), dtype=np.bool_)
    global_mask = np.ones((num_nodes, num_nodes), dtype=np.bool_)

    for i in range(num_nodes):
        ri, ci = divmod(i, width)
        for j in range(num_nodes):
            if i == j:
                continue
            rj, cj = divmod(j, width)
            if ri == rj:
                row_mask[i, j] = True
            if ci == cj:
                col_mask[i, j] = True
    return row_mask, col_mask, global_mask


_PIECE_ORDER = jnp.array([1, 2, 3, 4, 5, 6, 7, -7, -6, -5, -4, -3, -2, -1], dtype=jnp.int8)


def _decode_board_from_obs(obs_nchw: jnp.ndarray) -> jnp.ndarray:
    """Decode current board (B,H,W int8) from the first 14 piece planes."""
    current_planes = obs_nchw[:, :14, :, :]
    piece_idx = jnp.argmax(current_planes, axis=1)
    has_piece = jnp.sum(current_planes, axis=1) > 0.5
    decoded = _PIECE_ORDER[piece_idx]
    return jnp.where(has_piece, decoded, jnp.int8(0)).astype(jnp.int8)


def _actions_to_relation_mask(
    action_mask: jnp.ndarray,
    from_idx: jnp.ndarray,
    to_idx: jnp.ndarray,
    num_nodes: int,
) -> jnp.ndarray:
    """Convert action mask (B,A) to directed relation mask (B,N,N)."""
    rel = jnp.zeros((action_mask.shape[0], num_nodes, num_nodes), dtype=jnp.bool_)
    rel = rel.at[:, from_idx, to_idx].set(action_mask)
    return rel


def _map_actions_from_rotated_view(action_mask_rot: jnp.ndarray, rotate_idx: jnp.ndarray) -> jnp.ndarray:
    """Map action mask from rotated-opponent view back to current-player view."""
    out = jnp.zeros_like(action_mask_rot)
    out = out.at[:, rotate_idx].set(action_mask_rot)
    return out


def _king_square(board: jnp.ndarray, king_piece: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Get king square index and validity for each board in batch."""
    flat = board.reshape((board.shape[0], -1))
    is_king = flat == jnp.int8(king_piece)
    valid = jnp.any(is_king, axis=1)
    sq = jnp.argmax(is_king, axis=1).astype(jnp.int32)
    return sq, valid


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
    """Multi-relation graph attention + gated feed-forward residual block."""
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
        row_mask: jnp.ndarray,
        col_mask: jnp.ndarray,
        global_mask: jnp.ndarray,
        own_legal_mask: jnp.ndarray,
        own_attack_mask: jnp.ndarray,
        opp_legal_mask: jnp.ndarray,
        opp_attack_mask: jnp.ndarray,
        own_check_mask: jnp.ndarray,
        opp_check_mask: jnp.ndarray,
        train: bool = True,
    ) -> jnp.ndarray:
        del train  # reserved for future dropout/stochastic-depth extension

        idx = jnp.where(neighbor_idx < 0, 0, neighbor_idx)
        local_valid_mask = neighbor_mask > 0.0

        x = nn.LayerNorm(dtype=self.dtype, name="ln_attn")(h)
        q = nn.Dense(self.hidden_dim, use_bias=False, dtype=self.dtype, name="q_proj")(x)
        k = nn.Dense(self.hidden_dim, use_bias=False, dtype=self.dtype, name="k_proj")(x)
        v = nn.Dense(self.hidden_dim, use_bias=False, dtype=self.dtype, name="v_proj")(x)

        # Local directional relation.
        k_neigh = jnp.take(k, idx, axis=1)  # (B, N, M, D)
        v_neigh = jnp.take(v, idx, axis=1)  # (B, N, M, D)
        dir_embed = nn.Embed(
            num_embeddings=8, features=self.hidden_dim, dtype=self.dtype, name="dir_embed"
        )(jnp.clip(neighbor_dir, 0, 7))
        dir_embed = dir_embed[None, :, :, :]

        scale = 1.0 / math.sqrt(float(self.hidden_dim))
        local_logits = jnp.sum(q[:, :, None, :] * (k_neigh + dir_embed), axis=-1) * scale
        local_logits = jnp.where(
            local_valid_mask[None, :, :],
            local_logits,
            jnp.finfo(local_logits.dtype).min,
        )
        local_attn = nn.softmax(local_logits, axis=-1)
        local_agg = jnp.sum(local_attn[..., None] * (v_neigh + dir_embed), axis=2)
        local_out = nn.Dense(self.hidden_dim, dtype=self.dtype, name="local_out")(local_agg)

        def relation_agg(mask: jnp.ndarray, name: str) -> jnp.ndarray:
            logits = jnp.einsum("bnd,bmd->bnm", q, k) * scale
            bias = self.param(f"{name}_bias", nn.initializers.zeros, (1,))
            logits = logits + bias[0]

            rel_mask = mask if mask.ndim == 3 else mask[None, :, :]
            # Avoid rows with all-invalid entries causing NaNs in softmax.
            no_valid = jnp.sum(rel_mask, axis=-1, keepdims=True) == 0
            eye = jnp.eye(rel_mask.shape[-1], dtype=jnp.bool_)[None, :, :]
            rel_mask = rel_mask | (no_valid & eye)

            logits = jnp.where(rel_mask, logits, jnp.finfo(logits.dtype).min)
            attn = nn.softmax(logits, axis=-1)
            agg = jnp.einsum("bnm,bmd->bnd", attn, v)
            return nn.Dense(self.hidden_dim, dtype=self.dtype, name=f"{name}_out")(agg)

        row_out = relation_agg(row_mask, "row")
        col_out = relation_agg(col_mask, "col")
        global_out = relation_agg(global_mask, "global")
        own_legal_out = relation_agg(own_legal_mask, "own_legal")
        own_attack_out = relation_agg(own_attack_mask, "own_attack")
        opp_legal_out = relation_agg(opp_legal_mask, "opp_legal")
        opp_attack_out = relation_agg(opp_attack_mask, "opp_attack")
        own_check_out = relation_agg(own_check_mask, "own_check")
        opp_check_out = relation_agg(opp_check_mask, "opp_check")

        branches = jnp.stack(
            [
                local_out,
                row_out,
                col_out,
                global_out,
                own_legal_out,
                own_attack_out,
                opp_legal_out,
                opp_attack_out,
                own_check_out,
                opp_check_out,
            ],
            axis=2,
        )  # (B,N,10,D)
        gate_logits = nn.Dense(10, dtype=self.dtype, name="rel_gate")(x)
        rel_gates = nn.softmax(gate_logits, axis=-1)
        agg = jnp.sum(branches * rel_gates[..., None], axis=2)
        agg = nn.Dense(self.hidden_dim, dtype=self.dtype, name="mix_out")(agg)

        gamma1 = self.param(
            "gamma1", nn.initializers.constant(self.layer_scale_init), (self.hidden_dim,)
        )
        h = h + gamma1[None, None, :] * agg

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

        from_bias = nn.Dense(1, dtype=self.dtype, name="from_bias")(x).squeeze(-1)
        to_bias = nn.Dense(1, dtype=self.dtype, name="to_bias")(x).squeeze(-1)

        pair_scores = jnp.einsum("bnd,bmd->bnm", q_from, k_to) * (1.0 / math.sqrt(float(proj_dim)))
        logits = (
            pair_scores[:, from_idx, to_idx]
            + from_bias[:, from_idx]
            + to_bias[:, to_idx]
        )

        global_ctx = jnp.mean(x, axis=1)
        logits = logits + nn.Dense(self.action_space_size, dtype=self.dtype, name="global_proj")(global_ctx)
        return logits


class ValueHead(nn.Module):
    """Attention pooled scalar value head."""
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

        value = jnp.tanh(nn.Dense(1, dtype=self.dtype, name="value_out")(fused).squeeze(-1))
        return value


class AlphaZeroNetwork(nn.Module):
    """Upgraded graph AlphaZero network."""
    action_space_size: int = ACTION_SPACE_SIZE
    channels: int = 128
    num_blocks: int = 8
    dtype: jnp.dtype = jnp.float32

    def setup(self):
        neighbor_idx, neighbor_mask, neighbor_dir = _build_grid_neighbors(BOARD_HEIGHT, BOARD_WIDTH)
        row_mask, col_mask, global_mask = _build_rank_file_masks(BOARD_HEIGHT, BOARD_WIDTH)
        self.neighbor_idx = jnp.array(neighbor_idx, dtype=jnp.int32)
        self.neighbor_mask = jnp.array(neighbor_mask, dtype=jnp.float32)
        self.neighbor_dir = jnp.array(neighbor_dir, dtype=jnp.int32)
        self.row_mask = jnp.array(row_mask, dtype=jnp.bool_)
        self.col_mask = jnp.array(col_mask, dtype=jnp.bool_)
        self.global_mask = jnp.array(global_mask, dtype=jnp.bool_)
        self.action_from_idx = jnp.array(_ACTION_TO_FROM_SQ, dtype=jnp.int32)
        self.action_to_idx = jnp.array(_ACTION_TO_TO_SQ, dtype=jnp.int32)
        self.rotate_idx = rotate_action(jnp.arange(ACTION_SPACE_SIZE))

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
        board = _decode_board_from_obs(x)
        x = jnp.transpose(x, (0, 2, 3, 1))  # NCHW -> NHWC
        batch_size = x.shape[0]
        num_nodes = BOARD_HEIGHT * BOARD_WIDTH

        # Build dynamic relation masks from tactical actions (both sides).
        players = jnp.zeros((batch_size,), dtype=jnp.int32)
        own_legal_actions = jax.vmap(get_legal_moves_mask)(board, players)  # (B, A)
        board_flat = board.reshape((batch_size, num_nodes))
        own_attack_actions = own_legal_actions & (board_flat[:, self.action_to_idx] < 0)

        # Opponent perspective: rotate+flip signs so opponent becomes current player.
        opp_board = -jnp.flip(board, axis=(1, 2))
        opp_legal_rot = jax.vmap(get_legal_moves_mask)(opp_board, players)
        opp_board_flat = opp_board.reshape((batch_size, num_nodes))
        opp_attack_rot = opp_legal_rot & (opp_board_flat[:, self.action_to_idx] < 0)
        opp_legal_actions = _map_actions_from_rotated_view(opp_legal_rot, self.rotate_idx)
        opp_attack_actions = _map_actions_from_rotated_view(opp_attack_rot, self.rotate_idx)

        own_king_sq, own_king_valid = _king_square(board, 1)
        opp_king_sq, opp_king_valid = _king_square(board, -1)
        own_check_actions = (
            own_legal_actions
            & (self.action_to_idx[None, :] == opp_king_sq[:, None])
            & opp_king_valid[:, None]
        )
        opp_check_actions = (
            opp_legal_actions
            & (self.action_to_idx[None, :] == own_king_sq[:, None])
            & own_king_valid[:, None]
        )

        own_legal_rel_mask = _actions_to_relation_mask(
            own_legal_actions, self.action_from_idx, self.action_to_idx, num_nodes
        )
        own_attack_rel_mask = _actions_to_relation_mask(
            own_attack_actions, self.action_from_idx, self.action_to_idx, num_nodes
        )
        opp_legal_rel_mask = _actions_to_relation_mask(
            opp_legal_actions, self.action_from_idx, self.action_to_idx, num_nodes
        )
        opp_attack_rel_mask = _actions_to_relation_mask(
            opp_attack_actions, self.action_from_idx, self.action_to_idx, num_nodes
        )
        own_check_rel_mask = _actions_to_relation_mask(
            own_check_actions, self.action_from_idx, self.action_to_idx, num_nodes
        )
        opp_check_rel_mask = _actions_to_relation_mask(
            opp_check_actions, self.action_from_idx, self.action_to_idx, num_nodes
        )

        h = x.reshape((batch_size, num_nodes, x.shape[-1]))
        h = nn.Dense(self.channels, dtype=self.dtype)(h)  # keep Dense_0 for tooling compatibility

        pos_embed = self.param(
            "pos_embed",
            nn.initializers.normal(stddev=0.02),
            (num_nodes, self.channels),
        )
        h = h + pos_embed[None, :, :]
        h = nn.silu(h)

        for _ in range(self.num_blocks):
            h = GraphBlock(hidden_dim=self.channels, dtype=self.dtype)(
                h,
                self.neighbor_idx,
                self.neighbor_mask,
                self.neighbor_dir,
                self.row_mask,
                self.col_mask,
                self.global_mask,
                own_legal_rel_mask,
                own_attack_rel_mask,
                opp_legal_rel_mask,
                opp_attack_rel_mask,
                own_check_rel_mask,
                opp_check_rel_mask,
                train=train,
            )

        policy_logits = PolicyHead(
            action_space_size=self.action_space_size,
            model_dim=self.channels,
            dtype=self.dtype,
        )(h, self.action_from_idx, self.action_to_idx)

        value = ValueHead(model_dim=self.channels, dtype=self.dtype)(h)
        return policy_logits.astype(jnp.float32), value.astype(jnp.float32)

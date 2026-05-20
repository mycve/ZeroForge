"""
Pure NNUE-style sparse feature network for ZeroForge.

The public class name stays `AlphaZeroNetwork` so the rest of the training,
UCI and GUI code can keep using the existing import path on this branch.
"""

from __future__ import annotations

import jax.numpy as jnp
import flax.linen as nn

from xiangqi.actions import ACTION_SPACE_SIZE, BOARD_HEIGHT, BOARD_WIDTH
from xiangqi.env import (
    BOARD_OBSERVATION_CHANNELS,
    CHANNELS_PER_STEP,
    NUM_HISTORY_STEPS,
    NUM_RULE_STATE_CHANNELS,
)


NUM_FRAMES = NUM_HISTORY_STEPS + 1
NUM_SQUARES = BOARD_HEIGHT * BOARD_WIDTH
NUM_PIECE_FEATURES = NUM_FRAMES * CHANNELS_PER_STEP * NUM_SQUARES


def _clipped_relu(x: jnp.ndarray) -> jnp.ndarray:
    """NNUE-style bounded activation."""
    return jnp.clip(x, 0.0, 1.0)


class AlphaZeroNetwork(nn.Module):
    """Sparse piece-square NNUE network.

    Input remains the existing `(B, C, H, W)` observation tensor:
    - 9 board frames x 14 piece channels are converted to sparse
      `(frame, piece, square)` feature ids.
    - Rule-state planes are pooled into 10 scalar bits and projected into the
      accumulator.

    Output contract is unchanged: `(policy_logits, value, value_logits)`.
    """

    action_space_size: int = ACTION_SPACE_SIZE
    channels: int = 128
    num_blocks: int = 0
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(
        self, x: jnp.ndarray, train: bool = True
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        del train
        if x.ndim != 4:
            raise ValueError(f"输入必须是 4D (B,C,H,W), 实际 ndim={x.ndim}")
        if x.shape[2] != BOARD_HEIGHT or x.shape[3] != BOARD_WIDTH:
            raise ValueError(
                f"棋盘尺寸不匹配, 期望 ({BOARD_HEIGHT},{BOARD_WIDTH}), "
                f"实际 ({x.shape[2]},{x.shape[3]})"
            )

        expected_channels = BOARD_OBSERVATION_CHANNELS + NUM_RULE_STATE_CHANNELS
        if x.shape[1] != expected_channels:
            raise ValueError(f"通道数不匹配，期望 {expected_channels}，实际 {x.shape[1]}")
        if self.channels <= 0:
            raise ValueError(f"channels 必须 > 0，实际 {self.channels}")

        batch_size = x.shape[0]
        accumulator_dim = self.channels
        hidden_dim = max(self.channels, 64)
        bottleneck_dim = max(self.channels, 64)

        x = x.astype(self.dtype)
        board_x = x[:, :BOARD_OBSERVATION_CHANNELS, :, :]
        rule_x = x[:, BOARD_OBSERVATION_CHANNELS:, :, :]

        board_x = board_x.reshape(
            batch_size, NUM_FRAMES, CHANNELS_PER_STEP, BOARD_HEIGHT, BOARD_WIDTH
        )
        board_x = board_x.reshape(batch_size, NUM_FRAMES, CHANNELS_PER_STEP, NUM_SQUARES)
        board_x = jnp.transpose(board_x, (0, 1, 3, 2))  # (B, F, S, 14)

        occupancy = jnp.sum(board_x, axis=-1) > 0
        piece_idx = jnp.argmax(board_x, axis=-1).astype(jnp.int32)

        frame_base = (
            jnp.arange(NUM_FRAMES, dtype=jnp.int32)[:, None]
            * CHANNELS_PER_STEP
            * NUM_SQUARES
        )
        square_idx = jnp.arange(NUM_SQUARES, dtype=jnp.int32)[None, :]
        feature_ids = frame_base + piece_idx * NUM_SQUARES + square_idx
        feature_ids = jnp.where(occupancy, feature_ids, 0)

        feature_embed = self.param(
            "feature_embed",
            nn.initializers.normal(stddev=0.02),
            (NUM_PIECE_FEATURES, accumulator_dim),
        ).astype(self.dtype)
        active = occupancy.astype(self.dtype)[..., None]
        piece_acc = jnp.sum(jnp.take(feature_embed, feature_ids, axis=0) * active, axis=(1, 2))

        rule_bits = jnp.mean(rule_x, axis=(2, 3))
        rule_acc = nn.Dense(accumulator_dim, dtype=self.dtype, name="rule_proj")(rule_bits)

        h = piece_acc + rule_acc
        h = _clipped_relu(h)

        h = nn.Dense(hidden_dim, dtype=self.dtype, name="fc1")(h)
        h = _clipped_relu(h)
        h = nn.Dense(bottleneck_dim, dtype=self.dtype, name="fc2")(h)
        h = _clipped_relu(h)

        policy_logits = nn.Dense(
            self.action_space_size,
            dtype=self.dtype,
            name="policy",
        )(h)

        value_hidden = nn.Dense(bottleneck_dim, dtype=self.dtype, name="value_fc")(h)
        value_hidden = _clipped_relu(value_hidden)
        value_logits = nn.Dense(3, dtype=self.dtype, name="value_logits")(value_hidden)
        probs = nn.softmax(value_logits.astype(jnp.float32), axis=-1)
        value = probs[:, 0] - probs[:, 2]

        return (
            policy_logits.astype(jnp.float32),
            value.astype(jnp.float32),
            value_logits.astype(jnp.float32),
        )

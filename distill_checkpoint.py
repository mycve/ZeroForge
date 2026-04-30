"""Export teacher labels from a ZeroForge checkpoint.

This script restores a saved AlphaZero checkpoint and turns it into a compact
single-file distillation dataset that can be used by another model:

    obs:          uint8   [N, 126, 10, 9]
    policy_idx:   uint16  [N, top_k]
    policy_prob:  float16 [N, top_k]
    value_tgt:    float32 [N]
    value_logits: float32 [N, 3]
    legal_mask:   packed bool bits [N, ceil(ACTION_SPACE_SIZE / 8)]

The current training checkpoint format stores model/optimizer state, not the
in-memory replay buffer. For that reason this script distills the checkpoint by
rolling out the teacher policy and labeling visited positions with teacher
outputs.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp

from networks.alphazero import AlphaZeroNetwork
from xiangqi.actions import ACTION_SPACE_SIZE, BOARD_HEIGHT, BOARD_WIDTH, rotate_action
from xiangqi.env import NUM_OBSERVATION_CHANNELS, XiangqiEnv
from xiangqi.fen import load_fens_from_file


DTYPE_MAP = {
    "float32": jnp.float32,
    "bfloat16": jnp.bfloat16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Distill a ZeroForge checkpoint into one NPZ training file."
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="checkpoints",
        help="Checkpoint root directory, or a concrete step directory such as checkpoints/100.",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=None,
        help="Checkpoint step to restore. Defaults to the latest step in checkpoint-dir.",
    )
    parser.add_argument(
        "--output-dir",
        default="distill_data",
        help="Directory for the NPZ file and metadata.",
    )
    parser.add_argument("--num-positions", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument(
        "--output-file",
        default=None,
        help="Single NPZ output path. Defaults to <output-dir>/distill_dataset.npz.",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=5.0,
        help="Seconds between progress prints. Use 0 to print every batch.",
    )
    parser.add_argument("--top-k", type=int, default=32)
    parser.add_argument(
        "--policy-temperature",
        type=float,
        default=1.0,
        help="Temperature used for exported policy soft labels.",
    )
    parser.add_argument(
        "--rollout-temperature",
        type=float,
        default=0.75,
        help="Temperature used to sample teacher rollouts. Use 0 for greedy rollout.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fen-file",
        default="eval_fens.txt",
        help="Optional FEN pool used as rollout starts. Empty string disables it.",
    )
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--max-no-capture-steps", type=int, default=120)
    parser.add_argument("--repetition-threshold", type=int, default=5)
    parser.add_argument("--num-channels", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=10)
    parser.add_argument(
        "--network-dtype",
        choices=sorted(DTYPE_MAP),
        default="bfloat16",
        help="Network compute dtype used to build the checkpoint parameter tree.",
    )
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lr-warmup-steps", type=int, default=2000)
    parser.add_argument("--lr-cosine-steps", type=int, default=120_000)
    parser.add_argument("--lr-min-ratio", type=float, default=0.05)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--keep-dense-policy",
        action="store_true",
        help="Also save dense teacher policy probabilities. This is much larger.",
    )
    return parser.parse_args()


def resolve_checkpoint(checkpoint_dir: str, step: int | None) -> tuple[Path, int]:
    path = Path(checkpoint_dir).expanduser().resolve()
    if path.name.isdigit() and path.is_dir():
        inferred_step = int(path.name)
        if step is not None and step != inferred_step:
            raise ValueError(
                f"Checkpoint path points to step {inferred_step}, but --step={step} was given."
            )
        return path.parent, inferred_step
    if not path.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {path}")

    manager = ocp.CheckpointManager(directory=str(path))
    latest_step = manager.latest_step()
    if step is None:
        if latest_step is None:
            numeric_steps = sorted(
                int(p.name) for p in path.iterdir() if p.is_dir() and p.name.isdigit()
            )
            if not numeric_steps:
                raise FileNotFoundError(f"No checkpoint steps found in {path}")
            step = numeric_steps[-1]
        else:
            step = int(latest_step)
    return path, int(step)


def build_templates(args: argparse.Namespace) -> tuple[AlphaZeroNetwork, dict[str, Any], Any]:
    net = AlphaZeroNetwork(
        action_space_size=ACTION_SPACE_SIZE,
        channels=args.num_channels,
        num_blocks=args.num_blocks,
        dtype=DTYPE_MAP[args.network_dtype],
    )
    dummy_obs = jnp.zeros(
        (1, NUM_OBSERVATION_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH),
        dtype=jnp.float32,
    )
    params_template = net.init(jax.random.PRNGKey(args.seed), dummy_obs, train=False)["params"]

    def lr_schedule(opt_step: jnp.ndarray) -> jnp.ndarray:
        warmup = jnp.minimum(opt_step / max(args.lr_warmup_steps, 1), 1.0)
        decay_step = jnp.maximum(opt_step - args.lr_warmup_steps, 0.0)
        decay_progress = jnp.minimum(decay_step / max(args.lr_cosine_steps, 1), 1.0)
        cosine = args.lr_min_ratio + (1.0 - args.lr_min_ratio) * 0.5 * (
            1.0 + jnp.cos(jnp.pi * decay_progress)
        )
        return args.learning_rate * warmup * cosine

    optimizer = optax.adamw(lr_schedule, weight_decay=args.weight_decay)
    opt_state_template = optimizer.init(params_template)
    return net, params_template, opt_state_template


def restore_params(
    ckpt_root: Path,
    step: int,
    params_template: dict[str, Any],
    opt_state_template: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    manager = ocp.CheckpointManager(directory=str(ckpt_root))
    restore_target = {
        "params": params_template,
        "opt_state": opt_state_template,
        "iteration": np.array(0),
        "frames": np.array(0),
        "rng_key": jax.random.PRNGKey(0),
    }
    restored = manager.restore(step, args=ocp.args.StandardRestore(restore_target))
    info = {
        "iteration": int(restored.get("iteration", step)),
        "frames": int(restored.get("frames", 0)),
    }
    return restored["params"], info


def load_fen_pool(fen_file: str) -> tuple[np.ndarray | None, np.ndarray | None]:
    if not fen_file:
        return None, None
    path = Path(fen_file).expanduser()
    if not path.exists():
        return None, None
    items = load_fens_from_file(str(path))
    if not items:
        return None, None
    boards = np.stack([board for board, _ in items]).astype(np.int8)
    players = np.asarray([player for _, player in items], dtype=np.int32)
    return boards, players


def make_distiller(
    args: argparse.Namespace,
    net: AlphaZeroNetwork,
    env: XiangqiEnv,
    fen_boards: np.ndarray | None,
    fen_players: np.ndarray | None,
):
    rotated_idx = rotate_action(jnp.arange(ACTION_SPACE_SIZE))
    policy_temperature = float(args.policy_temperature)
    rollout_temperature = float(args.rollout_temperature)
    top_k = int(args.top_k)

    if top_k <= 0 or top_k > ACTION_SPACE_SIZE:
        raise ValueError(f"top-k must be in [1, {ACTION_SPACE_SIZE}], got {top_k}")
    if policy_temperature <= 0:
        raise ValueError("--policy-temperature must be > 0")

    use_fens = fen_boards is not None and fen_players is not None
    if use_fens:
        fen_boards_jax = jnp.asarray(fen_boards)
        fen_players_jax = jnp.asarray(fen_players)
        fen_count = int(fen_boards.shape[0])
    else:
        fen_boards_jax = None
        fen_players_jax = None
        fen_count = 0

    def _init_states(key: jnp.ndarray, batch_size: int):
        keys = jax.random.split(key, batch_size)
        if use_fens:
            idx = jax.random.randint(key, (batch_size,), 0, fen_count)
            return jax.vmap(env.init_from_board)(fen_boards_jax[idx], fen_players_jax[idx])
        return jax.vmap(env.init)(keys)

    init_states = jax.jit(_init_states, static_argnums=(1,))

    @jax.jit
    def collect_batch(params: dict[str, Any], state: Any, key: jnp.ndarray):
        obs = jax.vmap(env.observe)(state)
        logits, value, value_logits, *_ = net.apply({"params": params}, obs, train=False)

        legal_norm = jnp.where(
            state.current_player[:, None] == 0,
            state.legal_action_mask,
            state.legal_action_mask[:, rotated_idx],
        )
        label_logits = logits / policy_temperature
        label_logits = jnp.where(legal_norm, label_logits, jnp.finfo(label_logits.dtype).min)
        policy = jax.nn.softmax(label_logits, axis=-1)
        policy_prob, policy_idx = jax.lax.top_k(policy, top_k)
        policy_prob = policy_prob / jnp.maximum(
            jnp.sum(policy_prob, axis=-1, keepdims=True),
            1e-10,
        )

        abs_logits = jnp.where(
            state.current_player[:, None] == 0,
            logits,
            logits[:, rotated_idx],
        )
        if rollout_temperature > 0.0:
            rollout_logits = abs_logits / rollout_temperature
            rollout_logits = jnp.where(
                state.legal_action_mask,
                rollout_logits,
                jnp.finfo(rollout_logits.dtype).min,
            )
            rollout_policy = jax.nn.softmax(rollout_logits, axis=-1)
            sample_keys = jax.random.split(key, obs.shape[0])

            def sample_action(probs, sample_key):
                return jax.random.choice(sample_key, ACTION_SPACE_SIZE, p=probs)

            actions = jax.vmap(sample_action)(rollout_policy, sample_keys)
        else:
            rollout_logits = jnp.where(
                state.legal_action_mask,
                abs_logits,
                jnp.finfo(abs_logits.dtype).min,
            )
            actions = jnp.argmax(rollout_logits, axis=-1).astype(jnp.int32)

        next_state = jax.vmap(env.step)(state, actions)
        fresh_state = _init_states(jax.random.fold_in(key, 2), obs.shape[0])
        next_state = jax.vmap(
            lambda new_state, reset_state: jax.lax.cond(
                new_state.terminated,
                lambda: reset_state,
                lambda: new_state,
            )
        )(next_state, fresh_state)

        return next_state, {
            "obs": obs.astype(jnp.uint8),
            "policy_idx": policy_idx.astype(jnp.uint16),
            "policy_prob": policy_prob.astype(jnp.float16),
            "value_tgt": value.astype(jnp.float32),
            "value_logits": value_logits.astype(jnp.float32),
            "legal_mask": legal_norm,
            "dense_policy": policy.astype(jnp.float16),
        }

    return init_states, collect_batch


def trim_batch(batch: dict[str, np.ndarray], n: int) -> dict[str, np.ndarray]:
    return {key: value[:n] for key, value in batch.items()}


def write_dataset(
    output_file: Path,
    rows: list[dict[str, np.ndarray]],
    keep_dense_policy: bool,
) -> tuple[Path, int]:
    merged: dict[str, np.ndarray] = {}
    for key in rows[0]:
        if key == "dense_policy" and not keep_dense_policy:
            continue
        values = [row[key] for row in rows]
        if key == "legal_mask":
            packed = np.packbits(np.concatenate(values, axis=0), axis=-1)
            merged[key] = packed.astype(np.uint8)
        else:
            merged[key] = np.concatenate(values, axis=0)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_file, **merged)
    return output_file, int(merged["obs"].shape[0])


def main() -> None:
    args = parse_args()
    if args.num_positions <= 0:
        raise ValueError("--num-positions must be > 0")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be > 0")

    ckpt_root, step = resolve_checkpoint(args.checkpoint_dir, args.step)
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = (
        Path(args.output_file).expanduser().resolve()
        if args.output_file
        else output_dir / "distill_dataset.npz"
    )

    env = XiangqiEnv(
        max_steps=args.max_steps,
        max_no_capture_steps=args.max_no_capture_steps,
        repetition_threshold=args.repetition_threshold,
    )
    net, params_template, opt_state_template = build_templates(args)
    params, ckpt_info = restore_params(ckpt_root, step, params_template, opt_state_template)

    fen_boards, fen_players = load_fen_pool(args.fen_file)
    init_states, collect_batch = make_distiller(args, net, env, fen_boards, fen_players)

    key = jax.random.PRNGKey(args.seed)
    key, init_key = jax.random.split(key)
    state = init_states(init_key, args.batch_size)

    all_rows: list[dict[str, np.ndarray]] = []
    total = 0
    start_time = time.perf_counter()
    last_progress_time = start_time

    def print_progress(force: bool = False) -> None:
        nonlocal last_progress_time
        now = time.perf_counter()
        if not force and args.progress_interval > 0:
            if now - last_progress_time < args.progress_interval:
                return

        elapsed = max(now - start_time, 1e-9)
        rows_per_sec = total / elapsed
        remaining = max(args.num_positions - total, 0)
        eta_seconds = remaining / rows_per_sec if rows_per_sec > 0 else math.inf
        eta = "unknown" if math.isinf(eta_seconds) else format_duration(eta_seconds)
        pct = 100.0 * total / max(args.num_positions, 1)
        print(
            "[distill] progress "
            f"{total}/{args.num_positions} ({pct:.1f}%) "
            f"speed={rows_per_sec:.1f} rows/s "
            f"eta={eta} buffered={total}"
        )
        last_progress_time = now

    while total < args.num_positions:
        key, step_key = jax.random.split(key)
        state, batch = collect_batch(params, state, step_key)
        batch_np = jax.device_get(batch)

        remaining = args.num_positions - total
        current_n = min(args.batch_size, remaining)
        batch_np = trim_batch(batch_np, current_n)
        all_rows.append(batch_np)
        total += current_n
        print_progress(force=args.progress_interval == 0)

    print_progress(force=True)
    dataset_path, dataset_rows = write_dataset(output_file, all_rows, args.keep_dense_policy)
    print(f"[distill] wrote {dataset_path} rows={dataset_rows}")

    metadata = {
        "format": "zeroforge_distill_npz_v1",
        "checkpoint_root": str(ckpt_root),
        "checkpoint_step": step,
        "checkpoint_info": ckpt_info,
        "num_positions": total,
        "file": str(dataset_path),
        "rows": dataset_rows,
        "obs_shape": [NUM_OBSERVATION_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH],
        "action_space_size": ACTION_SPACE_SIZE,
        "top_k": args.top_k,
        "policy_temperature": args.policy_temperature,
        "rollout_temperature": args.rollout_temperature,
        "legal_mask_packed_bits": True,
        "value_target": "teacher_value_current_player",
        "fen_file": str(Path(args.fen_file).resolve()) if args.fen_file else None,
        "fen_count": 0 if fen_boards is None else int(fen_boards.shape[0]),
        "network": {
            "num_channels": args.num_channels,
            "num_blocks": args.num_blocks,
            "network_dtype": args.network_dtype,
        },
    }
    with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(
        "[distill] done "
        f"rows={total} output={dataset_path}"
    )


def format_duration(seconds: float) -> str:
    seconds = max(int(seconds), 0)
    hours, rem = divmod(seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m{seconds:02d}s"
    if minutes:
        return f"{minutes}m{seconds:02d}s"
    return f"{seconds}s"


if __name__ == "__main__":
    main()

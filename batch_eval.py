#!/usr/bin/env python3
"""
ZeroForge 批量检查点性能分析脚本

功能：
- 批量加载不同 run（不同 ckpt_dir）下的多个 checkpoint
- 支持同一 run 内不同 step、跨 run 的 checkpoint 互相对战
- 按网络结构分组：仅结构相同（channels、num_blocks）的 checkpoint 可对战
- 输出：胜/和/负、得分率、ELO 差、可选 CSV/JSON 报告

用法：
  python batch_eval.py --runs runs.json [--output report.csv]
  python batch_eval.py --runs "checkpoints_v1,checkpoints_v2" --steps "10,20,50" [--games 100]

runs.json 示例：
  {
    "runs": [
      {"name": "lr2e4", "ckpt_dir": "checkpoints_lr2e4", "steps": [10, 20, 50]},
      {"name": "lr1e4", "ckpt_dir": "checkpoints_lr1e4", "steps": "all"}
    ],
    "eval_games": 100,
    "fen_file": "openings_generated.txt"
  }
"""

import os
import sys
import json
import argparse
import logging
import warnings
from functools import partial
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass, field

# 需在 import jax 前设置
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.numpy as jnp
import numpy as np
import mctx
import orbax.checkpoint as ocp

from xiangqi.env import XiangqiEnv, XiangqiState, NUM_OBSERVATION_CHANNELS
from xiangqi.actions import rotate_action, ACTION_SPACE_SIZE, BOARD_HEIGHT, BOARD_WIDTH
from xiangqi.fen import load_fens_from_file, parse_fen
from networks.alphazero import AlphaZeroNetwork

# ============================================================================
# 日志
# ============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ============================================================================
# 配置
# ============================================================================

@dataclass
class EvalConfig:
    """批量评估配置（与 train.py 保持一致）"""
    eval_games: int = 100
    num_simulations: int = 16
    top_k: int = 4
    eval_gumbel_scale: float = 0.10
    qtransform_value_scale: float = 0.30
    max_steps: int = 300
    max_no_capture_steps: int = 120
    repetition_threshold: int = 5
    fen_file: Optional[str] = "openings_generated.txt"

# ============================================================================
# Checkpoint 加载与结构推断
# ============================================================================

def _infer_channels(params) -> Optional[int]:
    """从 params 推断 channels"""
    try:
        if hasattr(params, "get") and params.get("Dense_0") is not None:
            return int(params["Dense_0"]["kernel"].shape[-1])
        if "Dense_0" in params:
            return int(params["Dense_0"]["kernel"].shape[-1])
        for k in params.keys():
            if str(k).startswith("Dense_"):
                kernel = params[k]["kernel"]
                if kernel.ndim == 2:
                    return int(kernel.shape[-1])
    except Exception:
        pass
    return None

def _infer_num_blocks(params) -> int:
    """从 params 推断 num_blocks"""
    try:
        return len([k for k in params.keys() if str(k).startswith("GraphBlock_")])
    except Exception:
        return 0

def load_checkpoint(ckpt_dir: str, step: int) -> Tuple[dict, int, int]:
    """
    加载 checkpoint 并推断网络结构。
    
    Returns:
        (params, channels, num_blocks)
    
    Raises:
        FileNotFoundError: checkpoint 不存在
        ValueError: 无法推断结构
    """
    ckpt_dir = os.path.abspath(ckpt_dir)
    if not os.path.isdir(ckpt_dir):
        raise FileNotFoundError(f"Checkpoint 目录不存在: {ckpt_dir}")
    
    ckpt_manager = ocp.CheckpointManager(ckpt_dir)
    if step == 0 or step is None:
        step = ckpt_manager.latest_step()
    if step is None:
        raise FileNotFoundError(f"未找到 checkpoint: {ckpt_dir}")
    
    try:
        restored = ckpt_manager.restore(step)
    except Exception:
        ckpt_path = os.path.join(ckpt_dir, str(step))
        restored = ocp.StandardCheckpointer().restore(ckpt_path)
    
    params = None
    if isinstance(restored, dict) or hasattr(restored, "keys"):
        if "params" in restored:
            params = restored["params"]
        elif "default" in restored and isinstance(restored["default"], dict):
            params = restored["default"].get("params")
    
    if params is None:
        raise ValueError(f"Checkpoint 中未找到 params: {ckpt_dir}/{step}")
    
    channels = _infer_channels(params)
    num_blocks = _infer_num_blocks(params)
    if not channels or num_blocks <= 0:
        raise ValueError(
            f"无法推断网络结构: channels={channels}, num_blocks={num_blocks} "
            f"(可能是旧格式 checkpoint)"
        )
    
    return params, channels, num_blocks

def list_checkpoints(ckpt_dir: str) -> List[int]:
    """列出目录下所有 checkpoint step"""
    if not os.path.exists(ckpt_dir):
        return []
    steps = []
    for d in os.listdir(ckpt_dir):
        path = os.path.join(ckpt_dir, d)
        if os.path.isdir(path) and d.isdigit():
            steps.append(int(d))
    return sorted(steps, reverse=True)

# ============================================================================
# 评估逻辑（复用 train 的核心流程）
# ============================================================================

def _build_eval_states(
    fen_file: Optional[str],
    batch_size: int,
    rng_key,
    env: XiangqiEnv,
    for_red: bool,
) -> XiangqiState:
    """构建评估用初始状态（与 train._build_eval_initial_states 逻辑一致）"""
    devices = jax.local_devices()
    num_devices = len(devices)
    batch_per_device = batch_size // num_devices
    if batch_per_device * num_devices != batch_size:
        raise ValueError(f"batch_size {batch_size} 必须整除 num_devices {num_devices}")
    
    if fen_file and os.path.exists(fen_file):
        fens = load_fens_from_file(fen_file)
        if not fens:
            raise ValueError(f"FEN 文件为空: {fen_file}")
        boards_r, players_r = [], []
        boards_b, players_b = [], []
        for i, (board, player) in enumerate(fens):
            if i % 2 == 0:
                if player == 0:
                    boards_r.append(board)
                    players_r.append(0)
                else:
                    b_mirror = np.array(-np.flip(board, axis=-1), dtype=np.int8)
                    boards_r.append(b_mirror)
                    players_r.append(0)
            else:
                if player == 1:
                    boards_b.append(board)
                    players_b.append(1)
                else:
                    b_mirror = np.array(-np.flip(board, axis=-1), dtype=np.int8)
                    boards_b.append(b_mirror)
                    players_b.append(1)
        boards, players = (boards_r, players_r) if for_red else (boards_b, players_b)
        std_board, _ = parse_fen("rnbakabnr/9/1c5c1/p1p1p1p1p/9/9/P1P1P1P1P/1C5C1/9/RNBAKABNR w")
        while len(boards) < batch_size:
            boards.append(std_board)
            players.append(0 if for_red else 1)
        boards = np.stack(boards[:batch_size], axis=0).astype(np.int8)
        players = np.array(players[:batch_size], dtype=np.int32)
        states_flat = jax.vmap(env.init_from_board)(jnp.array(boards), jnp.array(players))
    else:
        keys = jax.random.split(rng_key, batch_size)
        states_flat = jax.vmap(env.init)(keys)
    
    def _shard(x):
        return x.reshape(num_devices, batch_per_device, *x.shape[1:])
    
    return jax.tree.map(_shard, states_flat)

def run_evaluation(
    params_a: dict,
    params_b: dict,
    net,
    env: XiangqiEnv,
    config: EvalConfig,
    rng_key,
) -> Tuple[int, int, int]:
    """
    运行 A vs B 对战（A 执红、B 执黑 与 A 执黑、B 执红 各一半）。
    
    Returns:
        (wins_a, draws, wins_b) 从 A 的视角
    """
    devices = jax.local_devices()
    num_devices = len(devices)
    # 每侧局数需整除设备数（与 train.py 一致）
    half_games = config.eval_games // 2
    batch_per_eval = max(num_devices, (half_games // num_devices) * num_devices)
    if batch_per_eval != half_games:
        log.debug("eval_games 已对齐: %d -> 每侧 %d 局", config.eval_games, batch_per_eval)
    
    _ROTATED_IDX = rotate_action(jnp.arange(ACTION_SPACE_SIZE))
    _QTRANSFORM = partial(
        mctx.qtransform_completed_by_mix_value,
        value_scale=config.qtransform_value_scale,
    )
    
    def eval_forward(params, obs):
        logits, value, _ = net.apply({'params': params}, obs, train=False)
        return logits.astype(jnp.float32), value.astype(jnp.float32)
    
    @jax.pmap
    def evaluate(params_red, params_black, initial_states, rng_key):
        batch_size = initial_states.board.shape[0]
        
        def recurrent_fn_eval(params, rng_key, action, state):
            prev_player = state.current_player
            state = jax.vmap(env.step)(state, action)
            obs = jax.vmap(env.observe)(state)
            logits, value = eval_forward(params, obs)
            logits = jnp.where(state.current_player[:, None] == 0, logits, logits[:, _ROTATED_IDX])
            logits = logits - jnp.max(logits, axis=-1, keepdims=True)
            logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)
            reward = state.rewards[jnp.arange(state.rewards.shape[0]), prev_player]
            value = jnp.where(state.terminated, 0.0, value)
            discount = jnp.where(state.terminated, 0.0, -1.0)
            return mctx.RecurrentFnOutput(reward=reward, discount=discount, prior_logits=logits, value=value), state
        
        def evaluate_recurrent_fn(params_pair, rng_key, action, state):
            p_red, p_black = params_pair
            out_red, next_state = recurrent_fn_eval(p_red, rng_key, action, state)
            out_black, _ = recurrent_fn_eval(p_black, rng_key, action, state)
            use_red = state.search_model_index == 0
            out = jax.tree.map(
                lambda r, b: jnp.where(use_red[:, None] if r.ndim > 1 else use_red, r, b),
                out_red, out_black
            )
            return out, next_state
        
        def step_fn(state, key):
            is_red = state.current_player == 0
            state = state.replace(search_model_index=jnp.where(is_red, 0, 1).astype(jnp.int32))
            obs = jax.vmap(env.observe)(state)
            logits_r, value_r = eval_forward(params_red, obs)
            logits_b, value_b = eval_forward(params_black, obs)
            logits = jnp.where(is_red[:, None], logits_r, logits_b)
            value = jnp.where(is_red, value_r, value_b)
            logits = jnp.where(is_red[:, None], logits, logits[:, _ROTATED_IDX])
            root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)
            policy_output = mctx.gumbel_muzero_policy(
                params=(params_red, params_black), rng_key=key, root=root,
                recurrent_fn=evaluate_recurrent_fn,
                num_simulations=config.num_simulations,
                max_num_considered_actions=config.top_k,
                invalid_actions=~state.legal_action_mask,
                qtransform=_QTRANSFORM,
                gumbel_scale=config.eval_gumbel_scale,
            )
            masked_policy = jnp.where(state.legal_action_mask, policy_output.action_weights, -1.0)
            action = jnp.argmax(masked_policy, axis=-1).astype(jnp.int32)
            next_state = jax.vmap(env.step)(state, action)
            return next_state, next_state.terminated
        
        state = initial_states
        terminated = jnp.zeros(batch_size, dtype=jnp.bool_)
        
        def body_fn(args):
            s, t, k = args
            k, sk = jax.random.split(k)
            ns, nt = step_fn(s, sk)
            return ns, t | nt, k
        
        state, _, _ = jax.lax.while_loop(
            lambda args: (~jnp.all(args[1])) & (args[0].step_count[0] < config.max_steps),
            body_fn, (state, terminated, rng_key)
        )
        return state.winner
    
    # 复制到多设备
    def replicate(pytree):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            return jax.device_put_replicated(pytree, devices)
    
    params_a_dev = replicate(params_a)
    params_b_dev = replicate(params_b)
    
    rng_key, sk1, sk2, sk3 = jax.random.split(rng_key, 4)
    states_r = _build_eval_states(config.fen_file, batch_per_eval, sk3, env, for_red=True)
    states_b = _build_eval_states(config.fen_file, batch_per_eval, sk3, env, for_red=False)
    
    states_r = jax.device_put_sharded(
        [jax.tree.map(lambda x: x[i], states_r) for i in range(num_devices)], devices
    )
    states_b = jax.device_put_sharded(
        [jax.tree.map(lambda x: x[i], states_b) for i in range(num_devices)], devices
    )
    eval_keys_r = jax.device_put_sharded(list(jax.random.split(sk1, num_devices)), devices)
    eval_keys_b = jax.device_put_sharded(list(jax.random.split(sk2, num_devices)), devices)
    
    # A 执红 vs B 执黑
    winners_r = evaluate(params_a_dev, params_b_dev, states_r, eval_keys_r)
    # B 执红 vs A 执黑
    winners_b = evaluate(params_b_dev, params_a_dev, states_b, eval_keys_b)
    
    # 从 A 视角统计：winners_r 中 0=A 赢 -1=和 1=B 赢；winners_b 中 1=A 赢 -1=和 0=B 赢
    wins_a = int((winners_r == 0).sum()) + int((winners_b == 1).sum())
    draws = int((winners_r == -1).sum()) + int((winners_b == -1).sum())
    wins_b = int((winners_r == 1).sum()) + int((winners_b == 0).sum())
    
    return wins_a, draws, wins_b

# ============================================================================
# 批量分析
# ============================================================================

@dataclass
class CheckpointRef:
    """检查点引用"""
    run_name: str
    ckpt_dir: str
    step: int
    channels: int = 0
    num_blocks: int = 0
    params: Optional[dict] = None
    
    def structure_key(self) -> Tuple[int, int]:
        return (self.channels, self.num_blocks)
    
    def label(self) -> str:
        return f"{self.run_name}/step{self.step}"

def discover_checkpoints(
    runs_config: List[Dict[str, Any]],
) -> Tuple[Dict[Tuple[int, int], List[CheckpointRef]], List[str]]:
    """
    发现所有 checkpoint 并按结构分组。
    
    Returns:
        (structure_to_refs, errors)
    """
    structure_to_refs: Dict[Tuple[int, int], List[CheckpointRef]] = {}
    errors: List[str] = []
    
    for run in runs_config:
        name = run.get("name", run.get("ckpt_dir", "unknown"))
        ckpt_dir = run["ckpt_dir"]
        steps_spec = run.get("steps", "all")
        
        if isinstance(steps_spec, list):
            steps = [int(s) for s in steps_spec]
        elif steps_spec == "all" or steps_spec == "latest":
            steps = list_checkpoints(ckpt_dir)
            if steps_spec == "latest" and steps:
                steps = [steps[0]]
            if not steps:
                errors.append(f"run {name}: 未找到 checkpoint")
                continue
        else:
            errors.append(f"run {name}: 无效的 steps 配置 {steps_spec}")
            continue
        
        for step in steps:
            try:
                params, channels, num_blocks = load_checkpoint(ckpt_dir, step)
                ref = CheckpointRef(
                    run_name=name,
                    ckpt_dir=ckpt_dir,
                    step=step,
                    channels=channels,
                    num_blocks=num_blocks,
                    params=params,
                )
                key = ref.structure_key()
                if key not in structure_to_refs:
                    structure_to_refs[key] = []
                structure_to_refs[key].append(ref)
            except Exception as e:
                errors.append(f"{name}/step{step}: {e}")
                log.warning("加载失败 %s/step%d: %s", name, step, e)
    
    return structure_to_refs, errors

def compute_elo_diff(score: float) -> float:
    """根据得分率计算 ELO 差（相对基准）"""
    if score <= 0:
        return -400.0
    if score >= 1:
        return 400.0
    return 400.0 * np.log10(score / (1.0 - score))

def run_batch_analysis(
    runs_config: List[Dict[str, Any]],
    eval_config: EvalConfig,
    mode: str = "matrix",  # "matrix" | "vs_baseline" | "round_robin"
    baseline: Optional[Tuple[str, int]] = None,  # (run_name, step) for vs_baseline
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """
    执行批量分析。
    
    Returns:
        结果列表，每项包含 {model_a, model_b, wins_a, draws, wins_b, score_a, elo_diff}
    """
    structure_to_refs, errors = discover_checkpoints(runs_config)
    if errors:
        for e in errors:
            log.error(e)
        if not structure_to_refs:
            raise RuntimeError("无有效 checkpoint 可分析")
    
    results = []
    rng = jax.random.PRNGKey(seed)
    
    for (channels, num_blocks), refs in structure_to_refs.items():
        log.info("结构 (%d, %d): %d 个 checkpoint", channels, num_blocks, len(refs))
        
        # 创建网络与环境
        net = AlphaZeroNetwork(
            action_space_size=ACTION_SPACE_SIZE,
            channels=channels,
            num_blocks=num_blocks,
            dtype=jnp.float32,
        )
        env = XiangqiEnv(
            max_steps=eval_config.max_steps,
            max_no_capture_steps=eval_config.max_no_capture_steps,
            repetition_threshold=eval_config.repetition_threshold,
        )
        
        # 构建对战对
        pairs = []
        if mode == "matrix":
            for i, ref_a in enumerate(refs):
                for ref_b in refs[i+1:]:  # 避免重复与自对战
                    pairs.append((ref_a, ref_b))
            if len(refs) < 2:
                log.warning("结构 (%d, %d) 仅 %d 个 checkpoint，matrix 模式无对战对", channels, num_blocks, len(refs))
                continue
        elif mode == "vs_baseline" and baseline:
            base_run, base_step = baseline
            base_ref = next((r for r in refs if r.run_name == base_run and r.step == base_step), None)
            if base_ref is None:
                log.warning("未找到基准 %s/step%d，跳过该结构", base_run, base_step)
                continue
            for ref in refs:
                if ref != base_ref:
                    pairs.append((ref, base_ref))
        else:
            # round_robin: 每对一次
            for i, ref_a in enumerate(refs):
                for ref_b in refs:
                    if ref_a != ref_b:
                        pairs.append((ref_a, ref_b))
        
        for ref_a, ref_b in pairs:
            if ref_a.params is None:
                ref_a.params, _, _ = load_checkpoint(ref_a.ckpt_dir, ref_a.step)
            if ref_b.params is None:
                ref_b.params, _, _ = load_checkpoint(ref_b.ckpt_dir, ref_b.step)
            
            rng, key = jax.random.split(rng)
            try:
                wins_a, draws, wins_b = run_evaluation(
                    ref_a.params, ref_b.params, net, env, eval_config, key
                )
            except Exception as e:
                log.exception("评估 %s vs %s 失败: %s", ref_a.label(), ref_b.label(), e)
                raise
            
            total = wins_a + draws + wins_b
            score_a = (wins_a + 0.5 * draws) / total if total > 0 else 0.5
            elo_diff = compute_elo_diff(score_a)
            
            results.append({
                "model_a": ref_a.label(),
                "model_b": ref_b.label(),
                "wins_a": wins_a,
                "draws": draws,
                "wins_b": wins_b,
                "total": total,
                "score_a": score_a,
                "elo_diff": elo_diff,  # A 相对 B 的 ELO 差
                "structure": f"{channels}x{num_blocks}",
            })
            
            log.info(
                "%s vs %s: W/D/L %d/%d/%d | 得分率 %.2f%% | ELO差 %.0f",
                ref_a.label(), ref_b.label(), wins_a, draws, wins_b,
                score_a * 100, elo_diff
            )
    
    return results

# ============================================================================
# 主入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ZeroForge 批量检查点性能分析",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--runs",
        required=True,
        help="runs 配置：JSON 文件路径，或逗号分隔的 ckpt_dir 列表（如 ckpt1,ckpt2）",
    )
    parser.add_argument(
        "--steps",
        default=None,
        help="当 --runs 为目录列表时，指定 step（逗号分隔，或 all/latest）",
    )
    parser.add_argument(
        "--games",
        type=int,
        default=100,
        help="每对评估局数（默认 100）",
    )
    parser.add_argument(
        "--fen",
        default=None,
        help="FEN 局面文件路径（默认 openings_generated.txt）",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="输出 CSV 或 JSON 报告路径",
    )
    parser.add_argument(
        "--mode",
        choices=["matrix", "vs_baseline", "round_robin"],
        default="matrix",
        help="matrix=同结构内两两对战(不含自对战); vs_baseline=所有 vs 指定基准; round_robin=全配对",
    )
    parser.add_argument(
        "--baseline",
        default=None,
        help="vs_baseline 模式下的基准，格式 run_name:step",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机种子",
    )
    args = parser.parse_args()
    
    # 解析 runs 配置
    if args.runs.endswith(".json"):
        with open(args.runs, "r", encoding="utf-8") as f:
            cfg = json.load(f)
        runs_config = cfg.get("runs", cfg)
        if isinstance(runs_config, dict):
            runs_config = [runs_config]
        eval_games = cfg.get("eval_games", args.games)
        fen_file = cfg.get("fen_file", args.fen or "openings_generated.txt")
    else:
        dirs = [d.strip() for d in args.runs.split(",") if d.strip()]
        steps = args.steps or "all"
        if steps.lower() in ("all", "latest"):
            steps = steps.lower()
        else:
            steps = [int(s) for s in steps.split(",")]
        runs_config = [
            {"name": os.path.basename(os.path.abspath(d)), "ckpt_dir": d, "steps": steps}
            for d in dirs
        ]
        eval_games = args.games
        fen_file = args.fen or "openings_generated.txt"
    
    eval_config = EvalConfig(
        eval_games=eval_games,
        fen_file=fen_file if fen_file and os.path.exists(fen_file) else None,
    )
    
    baseline_tuple = None
    if args.mode == "vs_baseline" and args.baseline:
        parts = args.baseline.split(":")
        if len(parts) != 2:
            log.error("--baseline 格式应为 run_name:step")
            sys.exit(1)
        baseline_tuple = (parts[0].strip(), int(parts[1]))
    
    log.info("开始批量分析: %d 个 run, 每对 %d 局", len(runs_config), eval_config.eval_games)
    
    results = run_batch_analysis(
        runs_config,
        eval_config,
        mode=args.mode,
        baseline=baseline_tuple,
        seed=args.seed,
    )
    
    if args.output:
        ext = os.path.splitext(args.output)[1].lower()
        if ext == ".json":
            with open(args.output, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        else:
            import csv
            with open(args.output, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=["model_a", "model_b", "wins_a", "draws", "wins_b", "score_a", "elo_diff", "structure"])
                w.writeheader()
                for r in results:
                    w.writerow({k: r.get(k, "") for k in w.fieldnames})
        log.info("报告已保存: %s", args.output)
    
    # 打印汇总
    print("\n" + "=" * 60)
    print("批量分析汇总")
    print("=" * 60)
    for r in results:
        print(f"  {r['model_a']} vs {r['model_b']}: 得分率 {r['score_a']:.2%} | ELO差 {r['elo_diff']:.0f}")
    print("=" * 60)

if __name__ == "__main__":
    main()

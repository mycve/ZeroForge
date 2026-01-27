#!/usr/bin/env python3
"""
批量评估检查点模型胜率脚本

功能：
1. 自动扫描 checkpoints 目录下所有可用检查点
2. 支持指定检查点列表进行对战评估
3. 多局对战计算胜率矩阵
4. 计算并展示 ELO 评分

用法：
    # 评估所有检查点（两两对战）
    python evaluate_checkpoints.py

    # 评估指定检查点
    python evaluate_checkpoints.py --checkpoints 100 200 300

    # 指定对战局数和模拟次数
    python evaluate_checkpoints.py --games 50 --simulations 64
    
    # 只对战相邻检查点（节省时间）
    python evaluate_checkpoints.py --adjacent-only
"""

import os
import sys
import json
import argparse
import warnings
from itertools import combinations
from typing import Dict, List, Tuple, Optional

import jax
import jax.numpy as jnp
import numpy as np
import mctx
import orbax.checkpoint as ocp

# --- JAX 配置 ---
cache_dir = os.path.abspath("jax_cache")
os.makedirs(cache_dir, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", cache_dir)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_enable_x64", False)

from xiangqi.env import XiangqiEnv, NUM_OBSERVATION_CHANNELS
from xiangqi.actions import rotate_action, ACTION_SPACE_SIZE, BOARD_HEIGHT, BOARD_WIDTH
from networks.alphazero import AlphaZeroNetwork

# ============================================================================
# 全局配置
# ============================================================================

class EvalConfig:
    """评估配置"""
    ckpt_dir: str = "checkpoints"
    
    # 网络架构 (必须与训练时一致)
    num_channels: int = 128
    num_blocks: int = 8
    
    # 评估参数
    eval_games: int = 100          # 每对模型对战局数（双边，实际 200 局）
    num_simulations: int = 128     # MCTS 模拟次数
    top_k: int = 32                # Gumbel 采样 top-k
    
    # 环境规则
    max_steps: int = 200
    max_no_capture_steps: int = 120
    repetition_threshold: int = 5


# ============================================================================
# 设备和环境初始化
# ============================================================================

devices = jax.local_devices()
num_devices = len(devices)

# 预计算旋转索引
_ROTATED_IDX = rotate_action(jnp.arange(ACTION_SPACE_SIZE))

def replicate_to_devices(pytree):
    """将 pytree 复制到所有设备"""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        return jax.device_put_replicated(pytree, devices)


# ============================================================================
# 模型加载
# ============================================================================

def create_network_and_env(config: EvalConfig):
    """创建网络和环境"""
    env = XiangqiEnv(
        max_steps=config.max_steps,
        max_no_capture_steps=config.max_no_capture_steps,
        repetition_threshold=config.repetition_threshold,
    )
    
    net = AlphaZeroNetwork(
        action_space_size=env.action_space_size,
        channels=config.num_channels,
        num_blocks=config.num_blocks,
        dtype=jnp.bfloat16,
    )
    
    return net, env


def load_checkpoint(ckpt_dir: str, step: int, params_template: dict) -> dict:
    """加载指定步数的检查点参数
    
    只恢复模型参数 params，跳过优化器状态等其他字段
    """
    ckpt_dir = os.path.abspath(ckpt_dir)
    
    # 创建临时 CheckpointManager 用于加载
    options = ocp.CheckpointManagerOptions(max_to_keep=1, save_interval_steps=1)
    ckpt_manager = ocp.CheckpointManager(directory=ckpt_dir, options=options)
    
    # 检查步数是否存在
    all_steps = ckpt_manager.all_steps()
    if step not in all_steps:
        raise ValueError(f"检查点 step={step} 不存在，可用: {sorted(all_steps)}")
    
    # 只恢复 params，跳过 opt_state 等其他字段
    # 使用 ocp.args.Composite 选择性恢复特定字段
    restore_args = ocp.args.Composite(
        params=ocp.args.StandardRestore(params_template),
    )
    
    restored = ckpt_manager.restore(step, args=restore_args)
    return restored["params"]


def get_available_checkpoints(ckpt_dir: str) -> List[int]:
    """获取所有可用的检查点步数"""
    ckpt_dir = os.path.abspath(ckpt_dir)
    
    if not os.path.exists(ckpt_dir):
        return []
    
    options = ocp.CheckpointManagerOptions(max_to_keep=100, save_interval_steps=1)
    ckpt_manager = ocp.CheckpointManager(directory=ckpt_dir, options=options)
    
    return sorted(ckpt_manager.all_steps())


# ============================================================================
# 评估函数
# ============================================================================

def create_evaluate_fn(net, env, config: EvalConfig):
    """创建评估函数"""
    
    def forward(params, obs, is_training=False):
        """前向传播"""
        logits, value = net.apply({'params': params}, obs, train=is_training)
        return logits, value
    
    def recurrent_fn(params, rng_key, action, state):
        """MCTS 递归函数"""
        prev_player = state.current_player
        state = jax.vmap(env.step)(state, action)
        obs = jax.vmap(env.observe)(state)
        logits, value = forward(params, obs)
        
        logits = jnp.where(state.current_player[:, None] == 0, logits, logits[:, _ROTATED_IDX])
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)
        
        reward = state.rewards[jnp.arange(state.rewards.shape[0]), prev_player]
        value = jnp.where(state.terminated, 0.0, value)
        discount = jnp.where(state.terminated, 0.0, -1.0)
        
        return mctx.RecurrentFnOutput(reward=reward, discount=discount, prior_logits=logits, value=value), state
    
    @jax.pmap
    def evaluate(params_red, params_black, rng_key):
        """评估对战：红方用 params_red，黑方用 params_black"""
        batch_size = config.eval_games // num_devices
        
        def evaluate_recurrent_fn(params_pair, rng_key, action, state):
            p_red, p_black = params_pair
            out_red, next_state = recurrent_fn(p_red, rng_key, action, state)
            out_black, _ = recurrent_fn(p_black, rng_key, action, state)
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
            logits_r, value_r = forward(params_red, obs)
            logits_b, value_b = forward(params_black, obs)
            
            logits = jnp.where(is_red[:, None], logits_r, logits_b)
            value = jnp.where(is_red, value_r, value_b)
            logits = jnp.where(is_red[:, None], logits, logits[:, _ROTATED_IDX])
            
            root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=state)
            
            k_search, k_sample = jax.random.split(key)
            policy_output = mctx.gumbel_muzero_policy(
                params=(params_red, params_black), rng_key=k_search, root=root,
                recurrent_fn=evaluate_recurrent_fn,
                num_simulations=config.num_simulations, 
                max_num_considered_actions=config.top_k,
                invalid_actions=~state.legal_action_mask,
            )
            
            # 评估时使用较低温度，减少随机性
            temp = jnp.where(
                state.step_count < 6, 0.8,
                jnp.where(state.step_count < 20, 0.4, 0.05)
            )
            
            def _sample_action(w, t, k, legal_mask):
                t = jnp.maximum(t, 1e-3)
                w_masked = jnp.where(legal_mask, w, 0.0)
                log_w = jnp.log(w_masked + 1e-10)
                log_w_temp = log_w / t
                log_w_temp = jnp.where(legal_mask, log_w_temp, -jnp.inf)
                log_w_temp = log_w_temp - jnp.max(log_w_temp)
                w_temp = jnp.exp(log_w_temp)
                w_temp = jnp.where(legal_mask, w_temp, 0.0)
                w_prob = w_temp / jnp.maximum(jnp.sum(w_temp), 1e-10)
                return jax.random.choice(k, ACTION_SPACE_SIZE, p=w_prob)
            
            sample_keys = jax.random.split(k_sample, batch_size)
            action = jax.vmap(_sample_action)(policy_output.action_weights, temp, sample_keys, state.legal_action_mask)
            
            next_state = jax.vmap(env.step)(state, action)
            return next_state, (next_state.terminated, next_state.winner, next_state.draw_reason)
        
        state = jax.vmap(env.init)(jax.random.split(rng_key, batch_size))
        terminated = jnp.zeros(batch_size, dtype=jnp.bool_)
        
        def body_fn(args):
            s, t, k = args
            k, sk = jax.random.split(k)
            ns, (nt, _, _) = step_fn(s, sk)
            return ns, t | nt, k
        
        state, _, _ = jax.lax.while_loop(
            lambda args: (~jnp.all(args[1])) & (args[0].step_count[0] < config.max_steps),
            body_fn, (state, terminated, rng_key)
        )
        
        return state.winner, state.draw_reason
    
    return evaluate


def run_evaluation(
    params_a: dict, 
    params_b: dict, 
    evaluate_fn,
    config: EvalConfig,
    rng_key,
) -> Dict:
    """运行双边评估
    
    返回：
        结果字典，包含：
        - a_as_red_wins: A 执红时的胜场
        - a_as_black_wins: A 执黑时的胜场
        - b_as_red_wins: B 执红时的胜场
        - b_as_black_wins: B 执黑时的胜场
        - draws: 和棋场数
        - total_games: 总局数
        - a_win_rate: A 的总胜率
    """
    rng_key, sk1, sk2 = jax.random.split(rng_key, 3)
    
    # 分发参数到设备
    params_a_rep = replicate_to_devices(params_a)
    params_b_rep = replicate_to_devices(params_b)
    
    # 分发随机数 Key
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        eval_keys_1 = jax.device_put_sharded(list(jax.random.split(sk1, num_devices)), devices)
        eval_keys_2 = jax.device_put_sharded(list(jax.random.split(sk2, num_devices)), devices)
    
    # A 执红 vs B 执黑
    winners_1, reasons_1 = evaluate_fn(params_a_rep, params_b_rep, eval_keys_1)
    winners_1 = jax.device_get(winners_1).flatten()
    reasons_1 = jax.device_get(reasons_1).flatten()
    
    # B 执红 vs A 执黑
    winners_2, reasons_2 = evaluate_fn(params_b_rep, params_a_rep, eval_keys_2)
    winners_2 = jax.device_get(winners_2).flatten()
    reasons_2 = jax.device_get(reasons_2).flatten()
    
    # 统计结果
    # A 执红胜 (winner=0 表示红胜)
    a_as_red_wins = int((winners_1 == 0).sum())
    # A 执黑胜 (winner=1 表示黑胜)
    a_as_black_wins = int((winners_2 == 1).sum())
    # B 执红胜
    b_as_red_wins = int((winners_2 == 0).sum())
    # B 执黑胜
    b_as_black_wins = int((winners_1 == 1).sum())
    # 和棋
    draws = int((winners_1 == -1).sum()) + int((winners_2 == -1).sum())
    
    total_games = len(winners_1) + len(winners_2)
    a_total_wins = a_as_red_wins + a_as_black_wins
    b_total_wins = b_as_red_wins + b_as_black_wins
    
    # 胜率计算（胜=1分，和=0.5分）
    a_score = a_total_wins + 0.5 * draws
    a_win_rate = a_score / total_games
    
    # 结束原因统计
    all_reasons = np.concatenate([reasons_1, reasons_2])
    reason_counts = {
        'checkmate': int((all_reasons == 8).sum()),
        'max_steps': int((all_reasons == 1).sum()),
        'no_capture': int((all_reasons == 2).sum()),
        'repetition': int((all_reasons == 3).sum()),
        'perpetual_check': int((all_reasons == 4).sum()),
        'no_attackers': int((all_reasons == 5).sum()),
        'perpetual_chase': int((all_reasons == 6).sum()),
        'check_chase_alt': int((all_reasons == 7).sum()),
    }
    
    return {
        'a_as_red_wins': a_as_red_wins,
        'a_as_black_wins': a_as_black_wins,
        'b_as_red_wins': b_as_red_wins,
        'b_as_black_wins': b_as_black_wins,
        'a_total_wins': a_total_wins,
        'b_total_wins': b_total_wins,
        'draws': draws,
        'total_games': total_games,
        'a_win_rate': a_win_rate,
        'reason_counts': reason_counts,
    }


def calculate_elo(win_rate: float, base_elo: float = 1500.0) -> float:
    """根据胜率计算 ELO 差值"""
    if win_rate <= 0:
        return base_elo - 400
    elif win_rate >= 1:
        return base_elo + 400
    else:
        return base_elo + 400 * np.log10(win_rate / (1 - win_rate))


def calculate_elo_ratings(results: Dict[Tuple[int, int], Dict], checkpoints: List[int]) -> Dict[int, float]:
    """根据对战结果计算所有模型的 ELO 评分
    
    使用简单的逐步 ELO 计算方法
    """
    if not checkpoints:
        return {}
    
    # 初始化第一个模型为基准 1500
    elo_ratings = {checkpoints[0]: 1500.0}
    
    for i in range(1, len(checkpoints)):
        ckpt = checkpoints[i]
        
        # 找与之前所有模型的对战结果，计算平均 ELO
        elo_estimates = []
        for prev_ckpt in checkpoints[:i]:
            key = (prev_ckpt, ckpt)
            if key in results:
                # 在这个对战中，prev_ckpt 是 A，ckpt 是 B
                win_rate_b = 1.0 - results[key]['a_win_rate']
                elo_diff = 400 * np.log10(win_rate_b / (1 - win_rate_b)) if 0 < win_rate_b < 1 else (400 if win_rate_b >= 1 else -400)
                elo_estimates.append(elo_ratings[prev_ckpt] + elo_diff)
        
        if elo_estimates:
            elo_ratings[ckpt] = np.mean(elo_estimates)
        else:
            elo_ratings[ckpt] = 1500.0
    
    return elo_ratings


# ============================================================================
# 主函数
# ============================================================================

def print_results_table(results: Dict[Tuple[int, int], Dict], checkpoints: List[int]):
    """打印胜率矩阵表格"""
    print("\n" + "=" * 80)
    print("胜率矩阵 (行 vs 列)")
    print("=" * 80)
    
    # 表头
    header = "         |" + "|".join([f" {ckpt:6d} " for ckpt in checkpoints]) + "|"
    print(header)
    print("-" * len(header))
    
    # 每行
    for i, ckpt_a in enumerate(checkpoints):
        row = f" {ckpt_a:6d} |"
        for j, ckpt_b in enumerate(checkpoints):
            if i == j:
                row += "   --   |"
            else:
                key = (min(ckpt_a, ckpt_b), max(ckpt_a, ckpt_b))
                if key in results:
                    res = results[key]
                    # 如果 ckpt_a < ckpt_b, 则 ckpt_a 是 A
                    if ckpt_a < ckpt_b:
                        win_rate = res['a_win_rate']
                    else:
                        win_rate = 1.0 - res['a_win_rate']
                    row += f" {win_rate*100:5.1f}% |"
                else:
                    row += "   ?    |"
        print(row)
    
    print("=" * 80)


def print_elo_table(elo_ratings: Dict[int, float]):
    """打印 ELO 评分表"""
    print("\n" + "=" * 50)
    print("ELO 评分")
    print("=" * 50)
    
    # 按 ELO 排序
    sorted_elos = sorted(elo_ratings.items(), key=lambda x: -x[1])
    
    print(f"{'排名':^6} | {'检查点':^10} | {'ELO':^10} | {'相对基准':^10}")
    print("-" * 50)
    
    base_elo = sorted_elos[-1][1] if sorted_elos else 1500.0
    for rank, (ckpt, elo) in enumerate(sorted_elos, 1):
        diff = elo - base_elo
        print(f"{rank:^6} | {ckpt:^10} | {elo:^10.1f} | {diff:+10.1f}")
    
    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(description="批量评估检查点模型胜率")
    parser.add_argument('--ckpt-dir', type=str, default='checkpoints',
                        help='检查点目录路径 (默认: checkpoints)')
    parser.add_argument('--checkpoints', type=int, nargs='+', default=None,
                        help='指定要评估的检查点列表，不指定则评估所有')
    parser.add_argument('--games', type=int, default=100,
                        help='每对模型对战局数 (实际双边共 2*games 局) (默认: 100)')
    parser.add_argument('--simulations', type=int, default=128,
                        help='MCTS 模拟次数 (默认: 128)')
    parser.add_argument('--adjacent-only', action='store_true',
                        help='只评估相邻检查点之间的对战 (节省时间)')
    parser.add_argument('--output', type=str, default=None,
                        help='结果输出 JSON 文件路径')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子 (默认: 42)')
    
    args = parser.parse_args()
    
    # 创建配置
    config = EvalConfig()
    config.ckpt_dir = args.ckpt_dir
    config.eval_games = args.games
    config.num_simulations = args.simulations
    
    print("=" * 60)
    print("ZeroForge 检查点批量评估")
    print("=" * 60)
    print(f"检查点目录: {config.ckpt_dir}")
    print(f"每对对战局数: {config.eval_games} × 2 = {config.eval_games * 2}")
    print(f"MCTS 模拟次数: {config.num_simulations}")
    print(f"设备数量: {num_devices}")
    print()
    
    # 获取可用检查点
    available_ckpts = get_available_checkpoints(config.ckpt_dir)
    if not available_ckpts:
        print(f"错误: 在 {config.ckpt_dir} 中未找到任何检查点")
        sys.exit(1)
    
    print(f"可用检查点: {available_ckpts}")
    
    # 确定要评估的检查点
    if args.checkpoints:
        checkpoints = sorted([c for c in args.checkpoints if c in available_ckpts])
        if len(checkpoints) != len(args.checkpoints):
            missing = set(args.checkpoints) - set(checkpoints)
            print(f"警告: 以下检查点不存在，已跳过: {missing}")
    else:
        checkpoints = available_ckpts
    
    if len(checkpoints) < 2:
        print("错误: 需要至少 2 个检查点才能进行对战评估")
        sys.exit(1)
    
    print(f"评估检查点: {checkpoints}")
    
    # 确定对战组合
    if args.adjacent_only:
        matchups = [(checkpoints[i], checkpoints[i+1]) for i in range(len(checkpoints)-1)]
    else:
        matchups = list(combinations(checkpoints, 2))
    
    print(f"对战组合数: {len(matchups)}")
    print()
    
    # 创建网络和环境
    net, env = create_network_and_env(config)
    
    # 初始化模板参数
    rng_key = jax.random.PRNGKey(args.seed)
    rng_key, subkey = jax.random.split(rng_key)
    dummy_obs = jnp.zeros((1, NUM_OBSERVATION_CHANNELS, BOARD_HEIGHT, BOARD_WIDTH))
    variables = net.init(subkey, dummy_obs, train=False)
    params_template = variables['params']
    
    # 创建评估函数
    evaluate_fn = create_evaluate_fn(net, env, config)
    
    # 预加载所有需要的检查点参数
    print("加载检查点参数...")
    params_cache = {}
    for ckpt in checkpoints:
        print(f"  加载检查点 {ckpt}...")
        params_cache[ckpt] = load_checkpoint(config.ckpt_dir, ckpt, params_template)
    print("加载完成!\n")
    
    # 运行所有对战
    results = {}
    for idx, (ckpt_a, ckpt_b) in enumerate(matchups, 1):
        print(f"[{idx}/{len(matchups)}] 评估: 检查点 {ckpt_a} vs {ckpt_b}")
        
        rng_key, subkey = jax.random.split(rng_key)
        result = run_evaluation(
            params_cache[ckpt_a],
            params_cache[ckpt_b],
            evaluate_fn,
            config,
            subkey,
        )
        
        results[(ckpt_a, ckpt_b)] = result
        
        # 打印单次结果
        print(f"  {ckpt_a} 胜率: {result['a_win_rate']*100:.1f}%")
        print(f"  ({ckpt_a}执红胜:{result['a_as_red_wins']}, {ckpt_a}执黑胜:{result['a_as_black_wins']}, "
              f"{ckpt_b}执红胜:{result['b_as_red_wins']}, {ckpt_b}执黑胜:{result['b_as_black_wins']}, "
              f"和:{result['draws']})")
        print(f"  结束原因: 将死{result['reason_counts']['checkmate']}, "
              f"步数限{result['reason_counts']['max_steps']}, "
              f"无吃子{result['reason_counts']['no_capture']}, "
              f"重复{result['reason_counts']['repetition']}")
        print()
    
    # 打印汇总结果
    print_results_table(results, checkpoints)
    
    # 计算 ELO
    elo_ratings = calculate_elo_ratings(results, checkpoints)
    print_elo_table(elo_ratings)
    
    # 保存结果
    if args.output:
        output_data = {
            'checkpoints': checkpoints,
            'matchups': [[a, b] for a, b in matchups],
            'results': {f"{a}_{b}": {
                **{k: v for k, v in r.items() if k != 'reason_counts'},
                'reason_counts': r['reason_counts']
            } for (a, b), r in results.items()},
            'elo_ratings': elo_ratings,
            'config': {
                'games': config.eval_games,
                'simulations': config.num_simulations,
                'seed': args.seed,
            }
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\n结果已保存到: {args.output}")
    
    print("\n评估完成!")


if __name__ == "__main__":
    main()

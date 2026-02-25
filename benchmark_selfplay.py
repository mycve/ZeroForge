#!/usr/bin/env python3
"""
ZeroForge 自玩性能基准测试
用法: uv run python benchmark_selfplay.py

测试内容:
1. 正确性验证: step_for_search 与 env.step 结果一致性
2. 性能基准: env.step vs step_for_search 吞吐量对比
3. 编译时间: XLA 编译图大小差异
"""
import os
import time

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

import jax
import jax.numpy as jnp
import numpy as np

from xiangqi.env import XiangqiEnv

BATCH = 256
NUM_ITERS = 300
NUM_RUNS = 3


def correctness_check(env):
    """验证 step_for_search 与 env.step 在短序列（无违规/和棋触发）中结果一致
    
    step_for_search 故意跳过违规检测/重复局面/无吃子计数等，
    因此只验证前 15 步内的核心字段（此阶段不可能触发这些规则）。
    """
    print("\n--- 正确性验证（前 15 步无违规场景）---")
    batch = 64
    keys = jax.random.split(jax.random.PRNGKey(42), batch)
    states = jax.vmap(env.init)(keys)

    step_fn = jax.jit(jax.vmap(env.step))
    search_fn = jax.jit(jax.vmap(env.step_for_search))

    s_full = states
    s_search = states
    all_ok = True

    for i in range(15):
        actions = jnp.argmax(s_full.legal_action_mask, axis=-1).astype(jnp.int32)
        s_full = step_fn(s_full, actions)
        s_search = search_fn(s_search, actions)

        checks = {
            "board": bool(jnp.all(s_full.board == s_search.board)),
            "player": bool(jnp.all(s_full.current_player == s_search.current_player)),
            "legal_mask": bool(jnp.all(s_full.legal_action_mask == s_search.legal_action_mask)),
            "terminated": bool(jnp.all(s_full.terminated == s_search.terminated)),
            "rewards": bool(jnp.allclose(s_full.rewards, s_search.rewards, atol=1e-6)),
        }
        for name, ok in checks.items():
            if not ok:
                print(f"  Step {i+1}: {name} 不匹配!")
                all_ok = False

    if all_ok:
        print(f"  通过! 15 步 × {batch} 局 × 5 项字段 = {15*batch*5} 项检查全部一致")
    return all_ok


def benchmark_step(env, step_fn, name, states):
    """通用 step 基准测试（多轮取最优）"""
    actions = jnp.argmax(states.legal_action_mask, axis=-1).astype(jnp.int32)

    # 编译预热 + 测编译时间
    t_compile_start = time.time()
    out = step_fn(states, actions)
    jax.block_until_ready(out)
    compile_time = time.time() - t_compile_start
    print(f"  {name} 编译: {compile_time:.3f}s")

    # 多轮测量
    best_elapsed = float('inf')
    all_sps = []

    for run in range(NUM_RUNS):
        s = states
        t0 = time.time()
        for _ in range(NUM_ITERS):
            a = jnp.argmax(s.legal_action_mask, axis=-1).astype(jnp.int32)
            s = step_fn(s, a)
        jax.block_until_ready(s)
        elapsed = time.time() - t0

        total = BATCH * NUM_ITERS
        sps = total / elapsed
        all_sps.append(sps)
        best_elapsed = min(best_elapsed, elapsed)
        print(f"    Run {run + 1}: {elapsed:.3f}s, {sps:.0f} steps/s")

    avg_sps = np.mean(all_sps)
    best_sps = BATCH * NUM_ITERS / best_elapsed
    print(f"  {name} 最佳: {best_sps:.0f} steps/s, 平均: {avg_sps:.0f} steps/s")
    return best_elapsed, best_sps, compile_time


def main():
    print("=" * 60)
    print("ZeroForge 自玩性能基准测试")
    print("=" * 60)
    print(f"JAX: {jax.__version__}")
    print(f"设备: {jax.devices()}")
    print(f"配置: batch={BATCH}, iters={NUM_ITERS}, runs={NUM_RUNS}")

    env = XiangqiEnv(max_steps=300, max_no_capture_steps=120, repetition_threshold=5)

    keys = jax.random.split(jax.random.PRNGKey(0), BATCH)
    states = jax.vmap(env.init)(keys)
    jax.block_until_ready(states)

    # 正确性验证
    correct = correctness_check(env)
    if not correct:
        print("\n正确性验证未通过! 终止测试。")
        return

    # --- 性能测试 ---
    print(f"\n--- 性能基准: {BATCH} batch × {NUM_ITERS} iters × {NUM_RUNS} runs ---")

    print("\n[env.step]")
    step_fn = jax.jit(jax.vmap(env.step))
    elapsed_step, sps_step, compile_step = benchmark_step(env, step_fn, "env.step", states)

    print("\n[step_for_search]")
    search_fn = jax.jit(jax.vmap(env.step_for_search))
    elapsed_search, sps_search, compile_search = benchmark_step(env, search_fn, "step_for_search", states)

    # --- 汇总 ---
    print(f"\n{'=' * 60}")
    print(f"{'指标':<25} {'env.step':>12} {'step_for_search':>16} {'提升':>8}")
    print(f"{'-' * 60}")
    print(f"{'编译时间':.<25} {compile_step:>11.3f}s {compile_search:>15.3f}s {compile_step/compile_search:>7.2f}x")
    print(f"{'最佳吞吐':.<25} {sps_step:>10.0f}/s {sps_search:>14.0f}/s {sps_search/sps_step:>7.2f}x")
    print(f"{'最佳耗时':.<25} {elapsed_step:>11.3f}s {elapsed_search:>15.3f}s {elapsed_step/elapsed_search:>7.2f}x")
    print(f"{'=' * 60}")

    # 优化前基线对比
    baseline_sps = 4241      # 原始 env.step (action_space=2550, 双重 get_legal_moves_mask)
    baseline_search = 4727   # 上一轮 step_for_search (action_space=2550)
    print(f"\n[对比全部优化前基线 (action_space=2550)]")
    print(f"  原始 env.step:     {baseline_sps} steps/s")
    print(f"  上轮 search(2550): {baseline_search} steps/s")
    print(f"  当前 env.step:     {sps_step:.0f} steps/s ({(sps_step/baseline_sps - 1)*100:+.1f}%)")
    print(f"  当前 search(2086): {sps_search:.0f} steps/s ({(sps_search/baseline_sps - 1)*100:+.1f}%)")

    print("\n测试完成！")


if __name__ == "__main__":
    main()

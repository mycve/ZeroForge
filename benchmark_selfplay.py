#!/usr/bin/env python3
"""
ZeroForge A/B 性能基准测试
用法: uv run python benchmark_selfplay.py

公平对比 step_for_search 的两种实现：
  A: 近似掩码 (get_basic_valid_mask, 跳过送将检测)
  B: 精确掩码 (get_legal_moves_mask, 含送将检测)

为避免编译缓存干扰，A 先编译运行（无缓存优势），B 后运行。
"""
import os
import time
import gc

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("TF_GPU_ALLOCATOR", "cuda_malloc_async")

import jax
import jax.numpy as jnp
import numpy as np

from xiangqi.env import XiangqiEnv

BATCH = 256
NUM_ITERS = 300
NUM_RUNS = 3


def benchmark_one(fn, states, name):
    """单函数基准: 编译 + 多轮运行"""
    actions = jnp.argmax(states.legal_action_mask, axis=-1).astype(jnp.int32)

    # 编译 (冷启动)
    t0 = time.time()
    out = fn(states, actions)
    jax.block_until_ready(out)
    compile_ms = (time.time() - t0) * 1000
    print(f"  [{name}] 编译: {compile_ms:.0f}ms")

    # 多轮测量
    all_sps = []
    for run in range(NUM_RUNS):
        s = states
        t0 = time.time()
        for _ in range(NUM_ITERS):
            a = jnp.argmax(s.legal_action_mask, axis=-1).astype(jnp.int32)
            s = fn(s, a)
        jax.block_until_ready(s)
        elapsed = time.time() - t0
        sps = BATCH * NUM_ITERS / elapsed
        all_sps.append(sps)
        print(f"    Run {run+1}: {elapsed:.3f}s, {sps:.0f} steps/s")

    best = max(all_sps)
    avg = np.mean(all_sps)
    print(f"  [{name}] best={best:.0f}, avg={avg:.0f} steps/s")
    return best, avg, compile_ms


def main():
    print("=" * 60)
    print("ZeroForge A/B 公平基准测试")
    print("=" * 60)
    print(f"JAX: {jax.__version__}, devices: {jax.devices()}")
    print(f"batch={BATCH}, iters={NUM_ITERS}, runs={NUM_RUNS}")
    print(f"total steps per run: {BATCH * NUM_ITERS}")

    env = XiangqiEnv(max_steps=300, max_no_capture_steps=120, repetition_threshold=5)
    keys = jax.random.split(jax.random.PRNGKey(0), BATCH)
    states = jax.vmap(env.init)(keys)
    jax.block_until_ready(states)

    # ================================================================
    # A: 近似掩码版本 (get_basic_valid_mask) — 先编译，无缓存优势
    # ================================================================
    print("\n--- A: step_for_search (近似掩码, 跳过送将检测) ---")
    fn_approx = jax.jit(jax.vmap(env.step_for_search))
    best_a, avg_a, comp_a = benchmark_one(fn_approx, states, "approx")

    # 强制清理，减少内存/缓存交叉影响
    gc.collect()

    # ================================================================
    # B: 精确掩码版本 (get_legal_moves_mask) — 后编译
    # ================================================================
    print("\n--- B: step_for_search_exact (精确掩码, 含送将检测) ---")
    fn_exact = jax.jit(jax.vmap(env.step_for_search_exact))
    best_b, avg_b, comp_b = benchmark_one(fn_exact, states, "exact")

    # ================================================================
    # 汇总
    # ================================================================
    print(f"\n{'=' * 60}")
    print(f"{'':>30} {'approx':>10} {'exact':>10} {'ratio':>8}")
    print(f"{'-' * 60}")
    print(f"{'compile (ms)':>30} {comp_a:>10.0f} {comp_b:>10.0f} {comp_b/max(comp_a,1):>7.2f}x")
    print(f"{'best sps':>30} {best_a:>10.0f} {best_b:>10.0f} {best_a/max(best_b,1):>7.2f}x")
    print(f"{'avg sps':>30} {avg_a:>10.0f} {avg_b:>10.0f} {avg_a/max(avg_b,1):>7.2f}x")
    print(f"{'=' * 60}")

    if best_a > best_b:
        pct = (best_a / best_b - 1) * 100
        print(f"\n=> 近似掩码比精确掩码快 {pct:.1f}% (best sps)")
    else:
        pct = (best_b / best_a - 1) * 100
        print(f"\n=> 精确掩码比近似掩码快 {pct:.1f}% (best sps)")

    print("\n测试完成!")


if __name__ == "__main__":
    main()

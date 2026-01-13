#!/usr/bin/env python3
"""
使用训练好的模型在 Web GUI 中对弈
用法: python play.py [checkpoint_path]
"""

import sys
import pickle
import jax
import jax.numpy as jnp
from train import config  # 直接从主训练文件导入配置
from gui.web_gui import run_web_gui
from networks.alphazero import AlphaZeroNetwork
from xiangqi.env import XiangqiEnv
from xiangqi.actions import ACTION_SPACE_SIZE, rotate_action, action_to_move, move_to_uci
import mctx
import numpy as np


def load_model(path, net):
    with open(path, 'rb') as f:
        ckpt = pickle.load(f)
    return ckpt['model']

def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/ckpt_000100.pkl"
    print(f"正在加载模型: {ckpt_path}")
    
    # 初始化环境和网络 (自动使用 train.py 中的配置)
    env = XiangqiEnv()
    net = AlphaZeroNetwork(
        action_space_size=ACTION_SPACE_SIZE,
        channels=config.num_channels,
        num_blocks=config.num_blocks,
    )
    
    # 加载参数
    try:
        params, batch_stats = load_model(ckpt_path, net)
    except FileNotFoundError:
        print(f"错误: 找不到模型文件 {ckpt_path}")
        return

    def ai_callback(state):
        # 准备观察
        obs = env.observe(state)[None, ...] # 添加 batch 维度
        
        # 前向计算 (获取初始胜率评估)
        (logits, value), _ = net.apply(
            {'params': params, 'batch_stats': batch_stats},
            obs, train=False, mutable=['batch_stats']
        )
        
        # 视角修正 (如果是黑方，将网络输出的视角 logits 转回真实坐标)
        if state.current_player == 1:
            rotate_idx = jnp.arange(ACTION_SPACE_SIZE)
            rotated_idx = rotate_action(rotate_idx)
            logits = logits[:, rotated_idx]
        
        # 掩码
        logits = logits - jnp.max(logits, axis=-1, keepdims=True)
        logits = jnp.where(state.legal_action_mask, logits, jnp.finfo(logits.dtype).min)
        
        # MCTS 搜索逻辑
        def recurrent_fn(model, rng_key, action, state):
            prev_player = state.current_player
            state = jax.vmap(env.step)(state, action)
            obs = jax.vmap(env.observe)(state)
            (l, v), _ = net.apply(
                {'params': params, 'batch_stats': batch_stats},
                obs, train=False, mutable=['batch_stats']
            )
            
            # 视角修正
            rotate_idx = jnp.arange(ACTION_SPACE_SIZE)
            rotated_idx = rotate_action(rotate_idx)
            l = jnp.where(state.current_player[:, None] == 0, l, l[:, rotated_idx])
            
            l = l - jnp.max(l, axis=-1, keepdims=True)
            l = jnp.where(state.legal_action_mask, l, jnp.finfo(l.dtype).min)
            
            reward = state.rewards[jnp.arange(state.rewards.shape[0]), prev_player]
            discount = jnp.where(state.terminated, 0.0, -1.0)
            return mctx.RecurrentFnOutput(reward=reward, discount=discount, prior_logits=l, value=v), state

        # 运行 Gumbel MCTS
        key = jax.random.PRNGKey(42)
        # 为 chex dataclass 添加 batch 维度
        batched_state = jax.tree.map(lambda x: jnp.expand_dims(x, 0), state)
        root = mctx.RootFnOutput(prior_logits=logits, value=value, embedding=batched_state)
        
        policy_output = mctx.gumbel_muzero_policy(
            params=None, # 不使用内部 model
            rng_key=key,
            root=root,
            recurrent_fn=recurrent_fn,
            num_simulations=256, # 实际下棋可以用更高的模拟次数
            max_num_considered_actions=16,
            invalid_actions=(~state.legal_action_mask)[None, ...],
        )
        
        # --- 对弈多样性优化：前 10 步使用适中温度采样 ---
        # 10 步 = 5 手棋，足以产生不同的开局分支
        temp = jnp.where(state.step_count < 10, 0.5, 0.01)
        
        def _sample_action(w, t, k, legal_mask):
            """
            温度采样，确保只从合法动作中选择
            
            问题：当温度很低时，power(x, 1/t) 会导致数值下溢
            解决：先应用掩码，再用 log 空间计算避免下溢
            """
            t = jnp.maximum(t, 1e-3)
            
            # 用 legal_mask 清零非法动作的权重
            w_masked = jnp.where(legal_mask, w, 0.0)
            
            # 使用 log 空间避免数值下溢: (w + eps)^(1/t) = exp(log(w + eps) / t)
            log_w = jnp.log(w_masked + 1e-10)
            log_w_temp = log_w / t
            
            # 减去最大值避免 exp 溢出 (log-sum-exp trick)
            log_w_temp = jnp.where(legal_mask, log_w_temp, -jnp.inf)
            log_w_temp = log_w_temp - jnp.max(log_w_temp)
            
            w_temp = jnp.exp(log_w_temp)
            w_temp = jnp.where(legal_mask, w_temp, 0.0)  # 再次确保非法动作为 0
            
            w_sum = jnp.sum(w_temp)
            w_prob = w_temp / jnp.maximum(w_sum, 1e-10)
            
            return jax.random.choice(k, ACTION_SPACE_SIZE, p=w_prob)
        
        # 使用 rng_key 进行采样
        sample_key = jax.random.PRNGKey(np.random.randint(0, 10000))
        action = _sample_action(policy_output.action_weights[0], temp, sample_key, state.legal_action_mask)
        action_int = int(action)
        
        # 额外验证：检查动作起点是否是己方棋子
        from_sq_check, to_sq_check = action_to_move(action_int)
        from_row_check = int(from_sq_check) // 9
        from_col_check = int(from_sq_check) % 9
        piece_check = int(state.board[from_row_check, from_col_check])
        player_check = int(state.current_player)
        
        is_own_piece = (player_check == 0 and piece_check > 0) or (player_check == 1 and piece_check < 0)
        
        if piece_check == 0 or not is_own_piece:
            uci_check = move_to_uci(int(from_sq_check), int(to_sq_check))
            player_name = "红方" if player_check == 0 else "黑方"
            print(f"\n{'='*60}")
            print(f"[严重错误] AI 选择的动作起点异常！")
            print(f"{'='*60}")
            print(f"当前玩家: {player_name} ({player_check})")
            print(f"选择动作: {action_int}, UCI: {uci_check}")
            print(f"起点 ({from_row_check},{from_col_check}) 棋子: {piece_check}")
            print(f"是己方棋子: {is_own_piece}")
            print(f"动作在 legal_mask 中: {bool(state.legal_action_mask[action_int])}")
            print(f"合法动作总数: {int(state.legal_action_mask.sum())}")
            print(f"\n当前棋盘 (jax_state.board):")
            print(np.array(state.board))
            print(f"{'='*60}\n")
            
            raise RuntimeError(
                f"AI选择的动作起点异常! 动作={action_int}, UCI={uci_check}, "
                f"玩家={player_name}, 起点棋子={piece_check}"
            )
        
        # 严格检查：如果选到非法动作，必须抛出异常
        if not state.legal_action_mask[action_int]:
            from_sq, to_sq = action_to_move(action_int)
            from_row, from_col = int(from_sq) // 9, int(from_sq) % 9
            to_row, to_col = int(to_sq) // 9, int(to_sq) % 9
            uci = move_to_uci(int(from_sq), int(to_sq))
            piece_at_from = int(state.board[from_row, from_col])
            
            weights = np.array(policy_output.action_weights[0])
            legal_mask_np = np.array(state.legal_action_mask)
            total_legal_weight = np.sum(weights * legal_mask_np)
            
            # 打印详细诊断信息
            player_name = "红方" if state.current_player == 0 else "黑方"
            print(f"\n{'='*60}")
            print(f"[严重错误] AI 选择了非法动作！")
            print(f"{'='*60}")
            print(f"当前玩家: {player_name}, 步数: {int(state.step_count)}")
            print(f"选择动作: {action_int}, UCI: {uci}")
            print(f"起点 ({from_row},{from_col}) 棋子: {piece_at_from}")
            print(f"终点 ({to_row},{to_col})")
            print(f"合法动作总数: {int(state.legal_action_mask.sum())}")
            print(f"MCTS合法动作总权重: {total_legal_weight:.6f}")
            print(f"MCTS所有权重总和: {np.sum(weights):.6f}")
            
            # 打印前10个合法动作及其权重
            legal_indices = np.where(legal_mask_np)[0][:10]
            print(f"\n前10个合法动作:")
            for idx in legal_indices:
                f, t = action_to_move(int(idx))
                print(f"  {move_to_uci(int(f), int(t))} (动作 {idx}): 权重={weights[idx]:.6f}")
            
            # 打印棋盘状态
            print(f"\n当前棋盘:")
            print(state.board)
            print(f"{'='*60}\n")
            
            raise RuntimeError(
                f"MCTS选择了非法动作! 动作={action_int}, UCI={uci}, "
                f"玩家={player_name}, 合法权重总和={total_legal_weight:.6f}"
            )
        
        # 最终搜索后的 Value 往往比初始 Value 更准
        # value 是当前玩家视角的评估，转换为红方视角
        raw_value = float(value[0])
        # 如果当前是黑方，反转 value 得到红方视角
        red_value = -raw_value if state.current_player == 1 else raw_value
        
        return action_int, red_value

    print("启动 Web GUI...")
    run_web_gui(ai_callback=ai_callback)

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
ZeroForge - 中国象棋 Gumbel MuZero AI
主入口点

用法:
    python main.py train                    # 开始训练
    python main.py train --resume           # 从检查点继续训练
    python main.py play                     # 人机对弈
    python main.py play --checkpoint PATH   # 使用指定检查点对弈
    python main.py eval                     # 评估模型
"""

from __future__ import annotations
import argparse
import logging
import sys
from pathlib import Path
from typing import Optional
import yaml

import jax
import jax.numpy as jnp

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ============================================================================
# 配置加载
# ============================================================================

def load_config(config_path: str = "configs/default.yaml") -> dict:
    """加载配置文件"""
    config_path = Path(config_path)
    if not config_path.exists():
        logger.warning(f"配置文件不存在: {config_path}, 使用默认配置")
        return {}
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    return config


# ============================================================================
# 训练命令
# ============================================================================

def cmd_train(args):
    """训练命令"""
    from xiangqi.env import XiangqiEnv, NUM_OBSERVATION_CHANNELS
    from xiangqi.actions import ACTION_SPACE_SIZE
    from xiangqi.mirror import augment_trajectory
    from networks.muzero import MuZeroNetwork
    from mcts.search import MCTSConfig, batched_mcts
    from training.trainer import MuZeroTrainer, TrainingConfig
    from training.replay_buffer import Trajectory, PrioritizedReplayBuffer
    from training.logging import TrainingLogger, setup_logging
    from training.checkpoint import CheckpointManager
    from evaluation.arena import Arena, Player
    from evaluation.elo import ELOTracker
    import numpy as np
    import time
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置日志
    setup_logging(log_level="INFO")
    
    logger.info("=" * 60)
    logger.info("ZeroForge - 中国象棋 Gumbel MuZero 训练")
    logger.info("=" * 60)
    logger.info(f"设备: {jax.devices()}")
    logger.info(f"配置: {args.config}")
    
    # 创建环境
    env = XiangqiEnv()
    logger.info(f"观察空间: {env.observation_shape}")
    logger.info(f"动作空间: {env.action_space_size}")
    
    # 创建网络
    net_config = config.get("network", {})
    network = MuZeroNetwork(
        observation_channels=net_config.get("observation_channels", NUM_OBSERVATION_CHANNELS),
        hidden_dim=net_config.get("hidden_dim", 256),
        action_space_size=net_config.get("action_space_size", ACTION_SPACE_SIZE),
        repr_blocks=net_config.get("repr_blocks", 8),
        dyn_blocks=net_config.get("dyn_blocks", 4),
        pred_blocks=net_config.get("pred_blocks", 4),
        value_support_size=net_config.get("value_support_size", 0),
        reward_support_size=net_config.get("reward_support_size", 0),
    )
    
    # 训练配置
    train_config = config.get("training", {})
    training_config = TrainingConfig(
        seed=config.get("seed", 42),
        num_training_steps=train_config.get("num_training_steps", 1000000),
        batch_size=train_config.get("batch_size", 256),
        learning_rate=train_config.get("learning_rate", 2e-4),
        lr_warmup_steps=train_config.get("lr_warmup_steps", 1000),
        lr_decay_steps=train_config.get("lr_decay_steps", 100000),
        weight_decay=train_config.get("weight_decay", 1e-4),
        unroll_steps=train_config.get("unroll_steps", 5),
        td_steps=train_config.get("td_steps", 10),
        policy_loss_weight=train_config.get("policy_loss_weight", 1.0),
        value_loss_weight=train_config.get("value_loss_weight", 0.25),
        reward_loss_weight=train_config.get("reward_loss_weight", 1.0),
        replay_buffer_size=config.get("replay_buffer", {}).get("capacity", 100000),
        min_replay_size=config.get("replay_buffer", {}).get("min_size", 1000),
        use_ema=train_config.get("use_ema", True),
        ema_decay=train_config.get("ema_decay", 0.999),
    )
    
    # MCTS 配置
    mcts_cfg = config.get("mcts", {})
    mcts_config = MCTSConfig(
        num_simulations=mcts_cfg.get("num_simulations", 100),
        max_num_considered_actions=mcts_cfg.get("max_num_considered_actions", 16),
        gumbel_scale=mcts_cfg.get("gumbel_scale", 1.0),
        discount=mcts_cfg.get("discount", 1.0),
        temperature=mcts_cfg.get("temperature_high", 1.0),  # 默认高温度
        use_dirichlet_noise=True,
        dirichlet_alpha=mcts_cfg.get("dirichlet_alpha", 0.3),
        dirichlet_fraction=mcts_cfg.get("dirichlet_fraction", 0.25),
    )
    
    # 温度退火配置
    temp_threshold = mcts_cfg.get("temperature_threshold", 30)
    temp_high = mcts_cfg.get("temperature_high", 1.0)
    temp_low = mcts_cfg.get("temperature_low", 0.25)
    
    # 创建目录
    checkpoint_dir = config.get("checkpoint", {}).get("checkpoint_dir", "checkpoints")
    log_dir = config.get("logging", {}).get("log_dir", "logs")
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # 创建训练器
    trainer = MuZeroTrainer(network, training_config, checkpoint_dir)
    
    # 创建日志器
    experiment_name = config.get("experiment_name", "xiangqi_muzero")
    train_logger = TrainingLogger(
        log_dir=log_dir,
        experiment_name=experiment_name,
        console_interval=config.get("logging", {}).get("console_interval", 100),
        tensorboard_interval=config.get("logging", {}).get("tensorboard_interval", 10),
    )
    
    # 创建 ELO 追踪器
    elo_tracker = ELOTracker(save_path=f"{checkpoint_dir}/elo.json")
    
    # 初始化训练状态
    rng_key = jax.random.PRNGKey(config.get("seed", 42))
    sample_obs = jnp.zeros((1, NUM_OBSERVATION_CHANNELS, 10, 9))
    state = trainer.init_state(rng_key, sample_obs)
    
    logger.info(f"训练起始步数: {state.step}")
    
    # 自我对弈函数
    def run_self_play_game(params, rng_key):
        """运行一局自我对弈"""
        observations = []
        actions = []
        policies = []
        values = []
        rewards = []
        to_plays = []  # 记录每步的当前玩家
        
        # 初始化游戏
        rng_key, init_key = jax.random.split(rng_key)
        game_state = env.init(init_key)
        
        step_in_game = 0
        while not game_state.terminated:
            # 记录当前玩家
            to_plays.append(int(game_state.current_player))
            
            # 温度退火：前 N 步高温度，之后低温度
            current_temp = temp_high if step_in_game < temp_threshold else temp_low
            
            # 获取观察
            obs = env.observe(game_state)
            observations.append(np.array(obs))
            
            # MCTS 搜索
            obs_batch = obs[jnp.newaxis, ...]
            legal_mask_batch = game_state.legal_action_mask[jnp.newaxis, ...]
            
            rng_key, search_key = jax.random.split(rng_key)
            
            from mcts.search import run_mcts, get_improved_policy, select_action
            policy_output = run_mcts(
                observation=obs_batch,
                legal_action_mask=legal_mask_batch,
                network_apply=network.apply,
                params=params,
                config=mcts_config,
                rng_key=search_key,
            )
            
            # 记录策略和价值（训练目标用高温度）
            policy = get_improved_policy(policy_output, temp_high)
            policies.append(np.array(policy[0]))
            
            root_value = float(policy_output.search_tree.node_values[0, 0])
            values.append(root_value)
            
            # 选择动作（使用退火后的温度）
            rng_key, action_key = jax.random.split(rng_key)
            action = select_action(policy_output, temperature=current_temp, rng_key=action_key)
            action = int(action[0])
            actions.append(action)
            
            step_in_game += 1
            
            # 执行动作
            game_state = env.step(game_state, jnp.int32(action))
            
            # 记录奖励
            rewards.append(float(game_state.rewards[0]))  # 红方视角
        
        # 创建轨迹
        trajectory = Trajectory(
            observations=np.array(observations),
            actions=np.array(actions),
            rewards=np.array(rewards),
            policies=np.array(policies),
            values=np.array(values),
            to_plays=np.array(to_plays, dtype=np.int32),
            game_result=int(game_state.winner),
        )
        
        return trajectory
    
    # 主训练循环
    logger.info("开始训练...")
    
    save_interval = config.get("checkpoint", {}).get("save_interval", 1000)
    eval_interval = config.get("evaluation", {}).get("eval_interval", 5000)
    
    best_elo = 1500.0
    
    try:
        while state.step < training_config.num_training_steps:
            # 自我对弈生成数据
            if len(trainer.replay_buffer) < training_config.min_replay_size or \
               state.step % 10 == 0:  # 每10步补充数据
                rng_key, play_key = jax.random.split(rng_key)
                trajectory = run_self_play_game(state.params, play_key)
                trainer.add_trajectory(trajectory)
                
                # 数据增强
                if config.get("self_play", {}).get("use_mirror_augmentation", True):
                    rng_key, aug_key = jax.random.split(rng_key)
                    if jax.random.uniform(aug_key) < 0.5:
                        from xiangqi.mirror import mirror_observation, mirror_action, mirror_policy
                        mirrored_traj = Trajectory(
                            observations=np.array([np.array(mirror_observation(jnp.array(o))) for o in trajectory.observations]),
                            actions=np.array([int(mirror_action(jnp.int32(a))) for a in trajectory.actions]),
                            rewards=trajectory.rewards,
                            policies=np.array([np.array(mirror_policy(jnp.array(p))) for p in trajectory.policies]),
                            values=trajectory.values,
                            to_plays=trajectory.to_plays,  # to_plays 镜像不变
                            game_result=trajectory.game_result,
                            is_mirrored=True,
                        )
                        trainer.add_trajectory(mirrored_traj)
            
            # 训练步骤
            rng_key, train_key = jax.random.split(rng_key)
            state, metrics = trainer.train_step(state, train_key)
            
            if metrics:
                train_logger.log_training(state.step, metrics)
                
                # 记录 buffer 状态
                train_logger.log_buffer(
                    state.step,
                    len(trainer.replay_buffer),
                    trainer.replay_buffer.num_trajectories(),
                )
            
            # 保存检查点
            if state.step > 0 and state.step % save_interval == 0:
                current_elo = elo_tracker.get_rating(f"step_{state.step}")
                is_best = current_elo > best_elo
                if is_best:
                    best_elo = current_elo
                
                trainer.save_checkpoint(
                    state,
                    elo_ratings={f"step_{state.step}": current_elo},
                    is_best=is_best,
                )
            
            # 评估
            if state.step > 0 and state.step % eval_interval == 0:
                logger.info(f"开始评估 (step={state.step})...")
                # 这里可以添加与历史最佳版本的对弈评估
                # 简化起见，只记录当前 ELO
                elo_tracker.ratings[f"step_{state.step}"] = elo_tracker.ratings.get(
                    f"step_{state.step}",
                    type(elo_tracker.ratings.get("step_0", ELOTracker().get_player_info("")))(
                        rating=1500 + state.step * 0.001
                    )
                )
                
                current_elo = elo_tracker.get_rating(f"step_{state.step}")
                train_logger.log_eval(
                    state.step,
                    elo=current_elo,
                    win_rate=0.5,  # 占位
                    games_played=0,
                )
    
    except KeyboardInterrupt:
        logger.info("训练被用户中断")
    
    finally:
        # 保存最终检查点
        trainer.save_checkpoint(state)
        train_logger.close()
        logger.info("训练完成!")


# ============================================================================
# 对弈命令
# ============================================================================

def cmd_play(args):
    """人机对弈命令 (CLI 模式)"""
    from cli.play import ChessCLI
    from mcts.search import MCTSConfig
    
    mcts_config = MCTSConfig(
        num_simulations=args.simulations,
        use_dirichlet_noise=False,
    )
    
    cli = ChessCLI(
        checkpoint_path=args.checkpoint,
        mcts_config=mcts_config,
    )
    cli.play()


# ============================================================================
# Web GUI 命令
# ============================================================================

def cmd_web(args):
    """Web 图形界面 (Gradio)"""
    from gui.web_gui import run_web_gui
    import numpy as np
    
    ai_callback = None
    
    if args.checkpoint:
        logger.info(f"加载检查点: {args.checkpoint}")
        
        # 加载配置和模型
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        
        import jax
        import jax.numpy as jnp
        from networks.muzero import MuZeroNetwork
        from mcts.search import gumbel_muzero_policy
        from training.checkpoint import CheckpointManager
        from xiangqi.env import XiangqiEnv
        
        # 初始化
        env = XiangqiEnv()
        network = MuZeroNetwork(
            action_dim=env.num_actions,
            **config.get('network', {})
        )
        
        # 加载权重
        ckpt_manager = CheckpointManager(args.checkpoint)
        train_state = ckpt_manager.restore_latest(network)
        
        if train_state is None:
            logger.warning("未找到检查点，使用随机初始化")
            rng = jax.random.PRNGKey(0)
            params = network.init(rng, jnp.zeros((1, 10, 9, 119)))
        else:
            params = train_state.params
            logger.info("检查点加载成功")
        
        # 创建 AI 回调
        def ai_callback(state):
            obs = jnp.expand_dims(state.observation, 0)
            root = network.apply(params, obs, method=network.initial_inference)
            
            policy_output = gumbel_muzero_policy(
                params=params,
                rng_key=jax.random.PRNGKey(np.random.randint(0, 2**31)),
                root=root,
                recurrent_fn=lambda p, k, a, e: network.apply(p, e, a, method=network.recurrent_inference),
                num_simulations=args.simulations,
                invalid_actions=~state.legal_action_mask,
                max_num_considered_actions=16,
            )
            
            action = int(policy_output.action[0])
            return action
    
    logger.info("启动 Web 图形界面...")
    logger.info("浏览器访问: http://localhost:7860")
    if args.share:
        logger.info("将创建公网分享链接")
    
    run_web_gui(ai_callback=ai_callback, share=args.share)


# ============================================================================
# 评估命令
# ============================================================================

def cmd_eval(args):
    """评估命令"""
    from evaluation.arena import Arena
    from evaluation.elo import ELOTracker
    
    logger.info("评估功能")
    logger.info("使用: python main.py eval --checkpoint1 PATH1 --checkpoint2 PATH2")
    
    # 这里可以添加两个检查点之间的对弈评估
    # 简化起见，只打印帮助信息


# ============================================================================
# 主入口
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="ZeroForge - 中国象棋 Gumbel MuZero AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
    python main.py train                         # 开始训练
    python main.py train --config my_config.yaml # 使用自定义配置
    python main.py play                          # CLI 人机对弈
    python main.py gui                           # 图形界面 (推荐)
    python main.py gui --checkpoint ckpt/        # 使用训练的模型对弈
    python main.py gui --fen "FEN字符串"          # 导入指定局面
""",
    )
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # 训练命令
    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="配置文件路径"
    )
    train_parser.add_argument(
        "--resume", action="store_true",
        help="从检查点继续训练"
    )
    
    # CLI 对弈命令
    play_parser = subparsers.add_parser("play", help="CLI 人机对弈")
    play_parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="模型检查点路径"
    )
    play_parser.add_argument(
        "--simulations", type=int, default=200,
        help="MCTS 模拟次数 (Gumbel MuZero 推荐 50-200)"
    )
    
    # Web GUI 命令 (推荐)
    web_parser = subparsers.add_parser("web", help="Web 图形界面 (Gradio)")
    web_parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="模型检查点路径 (不提供则为双人模式)"
    )
    web_parser.add_argument(
        "--config", type=str, default="configs/default.yaml",
        help="配置文件路径"
    )
    web_parser.add_argument(
        "--simulations", type=int, default=200,
        help="MCTS 模拟次数 (Gumbel MuZero 推荐 50-200)"
    )
    web_parser.add_argument(
        "--share", action="store_true",
        help="创建公网分享链接"
    )
    
    # 评估命令
    eval_parser = subparsers.add_parser("eval", help="评估模型")
    eval_parser.add_argument(
        "--checkpoint1", type=str,
        help="模型1检查点路径"
    )
    eval_parser.add_argument(
        "--checkpoint2", type=str,
        help="模型2检查点路径"
    )
    eval_parser.add_argument(
        "--games", type=int, default=100,
        help="对局数"
    )
    
    args = parser.parse_args()
    
    if args.command == "train":
        cmd_train(args)
    elif args.command == "play":
        cmd_play(args)
    elif args.command == "web":
        cmd_web(args)
    elif args.command == "eval":
        cmd_eval(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

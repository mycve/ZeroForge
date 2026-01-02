#!/usr/bin/env python3
"""
RL Framework - 通用强化学习框架入口

支持:
- MuZero / AlphaZero 算法
- 任意离散动作空间游戏
- Web 控制台
- 分布式训练

Usage:
    # 启动 Web 服务
    python main.py serve
    
    # 启动训练
    python main.py train --game chinese_chess --algorithm muzero
    
    # 评估模型
    python main.py eval --checkpoint path/to/checkpoint.pt
"""

import argparse
import logging
import sys
from pathlib import Path

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger(__name__)


def cmd_serve(args):
    """启动 Web 服务"""
    import uvicorn
    from server import create_app
    
    app = create_app()
    
    logger.info(f"启动 Web 服务: http://{args.host}:{args.port}")
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info"
    )


def cmd_train(args):
    """启动训练"""
    import torch
    from games import make_game
    from algorithms import make_algorithm
    from core.logging import StructuredLogger, set_logger
    
    # 初始化日志
    log_backends = args.log_backends.split(',') if args.log_backends else ['console', 'tensorboard']
    logger_instance = StructuredLogger(
        backends=log_backends,
        log_dir=args.log_dir,
        project_name="rl-framework",
        run_name=args.run_name,
    )
    set_logger(logger_instance)
    
    # 创建游戏
    game = make_game(args.game)
    logger.info(f"游戏: {game}")
    
    # 创建算法
    from algorithms.muzero.algorithm import MuZeroConfig
    from algorithms.alphazero.algorithm import AlphaZeroConfig
    
    if args.algorithm == 'muzero':
        config = MuZeroConfig(
            num_channels=args.channels,
            num_blocks=args.blocks,
            num_simulations=args.simulations,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
        )
    else:
        config = AlphaZeroConfig(
            num_channels=args.channels,
            num_blocks=args.blocks,
            num_simulations=args.simulations,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
        )
    
    algo = make_algorithm(args.algorithm, config)
    logger.info(f"算法: {algo}")
    
    # 创建网络
    network = algo.create_network(game)
    network = network.to(args.device)
    logger.info(f"网络参数量: {sum(p.numel() for p in network.parameters()):,}")
    
    # 创建自玩
    from core.selfplay import ThreadedSelfPlay
    from core.config import MCTSConfig, BatcherConfig, ThreadedEnvConfig
    
    env_config = ThreadedEnvConfig(
        num_envs=args.num_envs,
        mcts=MCTSConfig(num_simulations=args.simulations),
        batcher=BatcherConfig(device=args.device),
    )
    
    selfplay = ThreadedSelfPlay(
        game_factory=lambda: make_game(args.game),
        network=network,
        config=env_config,
    )
    
    # 简单训练循环
    logger.info("开始训练...")
    
    import torch.optim as optim
    optimizer = optim.AdamW(network.parameters(), lr=args.lr, weight_decay=1e-4)
    
    selfplay.start()
    
    from core.replay_buffer import ReplayBuffer
    
    # 创建回放缓冲区
    replay_buffer = ReplayBuffer(
        capacity=10000,
        action_space_size=game.action_space.n,
    )
    
    for epoch in range(args.epochs):
        # 自玩生成数据
        trajectories, stats = selfplay.collect_with_stats(num_games=args.games_per_epoch)
        
        # 添加到回放缓冲区
        replay_buffer.add_batch(trajectories)
        
        avg_len = stats.avg_game_length
        logger.info(f"Epoch {epoch+1}: 生成 {stats.num_games} 局, 平均长度 {avg_len:.1f}, 缓冲区 {len(replay_buffer)}")
        
        # 训练（如果缓冲区足够）
        if len(replay_buffer) >= args.batch_size:
            network.train()
            
            for _ in range(10):  # 每轮训练10步
                batch = replay_buffer.sample(
                    batch_size=args.batch_size,
                    unroll_steps=5 if args.algorithm == 'muzero' else 0,
                )
                
                # 转换为 tensor
                obs_t = torch.from_numpy(batch.observations).to(args.device)
                
                # 计算损失
                loss, metrics = algo.compute_loss(network, batch)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            
            logger.info(f"  训练损失: {metrics.get('total_loss', 0):.4f}")
        
        # 记录统计
        stats_dict = {
            "num_games": stats.num_games,
            "avg_game_length": stats.avg_game_length,
            "games_per_second": stats.games_per_second,
        }
        stats_dict.update(stats.win_stats)
        logger_instance.log_selfplay(stats.num_games, stats_dict, step=epoch)
    
    selfplay.stop()
    logger.info("训练完成")
    logger_instance.close()


def cmd_eval(args):
    """评估模型"""
    import torch
    from games import make_game
    from algorithms import make_algorithm
    
    if not Path(args.checkpoint).exists():
        logger.error(f"检查点不存在: {args.checkpoint}")
        return
    
    # 加载检查点
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    # 创建游戏和算法
    game = make_game(args.game)
    algo = make_algorithm(checkpoint.get('algorithm', 'muzero'))
    
    # 创建网络
    network = algo.create_network(game)
    network.load_state_dict(checkpoint['network'])
    network = network.to(args.device)
    network.eval()
    
    # 创建搜索
    from core.mcts import MCTSSearch
    from core.config import MCTSConfig
    
    mcts_config = MCTSConfig(num_simulations=200, temperature_init=0.0, temperature_final=0.0)
    search = MCTSSearch(game, mcts_config)
    
    # 创建评估函数
    def evaluate_fn(obs, mask):
        import torch
        obs_t = torch.from_numpy(obs).unsqueeze(0).to(args.device)
        mask_t = torch.from_numpy(mask).unsqueeze(0).to(args.device)
        with torch.no_grad():
            policy_logits, value = network(obs_t)
            policy = torch.softmax(policy_logits.masked_fill(~mask_t.bool(), -1e9), dim=-1)
        return policy[0].cpu().numpy(), value[0].item()
    
    logger.info(f"评估模型: {args.checkpoint}")
    logger.info(f"游戏: {game}")
    
    # 运行评估
    wins, losses, draws = 0, 0, 0
    
    for i in range(args.num_games):
        game.reset()
        search.reset()
        
        while not game.is_terminal():
            action, _, _ = search.run(evaluate_fn, add_noise=False)
            game.step(action)
            search.advance(action)
        
        winner = game.get_winner()
        if winner == 0:
            wins += 1
        elif winner == 1:
            losses += 1
        else:
            draws += 1
        
        if (i + 1) % 10 == 0:
            logger.info(f"进度: {i+1}/{args.num_games} | 胜: {wins} 负: {losses} 和: {draws}")
    
    win_rate = wins / args.num_games
    logger.info(f"评估完成: 胜率 {win_rate:.1%} ({wins}胜 {losses}负 {draws}和)")


def main():
    parser = argparse.ArgumentParser(
        description='RL Framework - 通用强化学习框架',
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # serve 命令
    serve_parser = subparsers.add_parser('serve', help='启动 Web 服务')
    serve_parser.add_argument('--host', default='127.0.0.1', help='监听地址')
    serve_parser.add_argument('--port', type=int, default=8000, help='监听端口')
    
    # train 命令
    train_parser = subparsers.add_parser('train', help='启动训练')
    train_parser.add_argument('--game', default='tictactoe', help='游戏名称 (tictactoe/chinese_chess)')
    train_parser.add_argument('--algorithm', default='alphazero', choices=['muzero', 'alphazero'], help='算法')
    train_parser.add_argument('--device', default='auto', help='设备 (auto/cuda/mps/cpu)')
    train_parser.add_argument('--network-size', default='auto', choices=['auto', 'small', 'medium', 'large'], help='网络大小')
    train_parser.add_argument('--channels', type=int, default=64, help='网络通道数 (medium/large)')
    train_parser.add_argument('--blocks', type=int, default=4, help='网络块数 (medium/large)')
    train_parser.add_argument('--simulations', type=int, default=50, help='MCTS 模拟次数')
    train_parser.add_argument('--batch-size', type=int, default=128, help='批大小')
    train_parser.add_argument('--lr', type=float, default=1e-3, help='学习率')
    train_parser.add_argument('--epochs', type=int, default=100, help='训练轮数')
    train_parser.add_argument('--games-per-epoch', type=int, default=50, help='每轮自玩局数')
    train_parser.add_argument('--num-envs', type=int, default=4, help='并行环境数')
    train_parser.add_argument('--log-dir', default='./logs', help='日志目录')
    train_parser.add_argument('--log-backends', default='console', help='日志后端 (console/tensorboard)')
    train_parser.add_argument('--run-name', default=None, help='运行名称')
    
    # eval 命令
    eval_parser = subparsers.add_parser('eval', help='评估模型')
    eval_parser.add_argument('--checkpoint', required=True, help='检查点路径')
    eval_parser.add_argument('--game', default='chinese_chess', help='游戏名称')
    eval_parser.add_argument('--device', default='cpu', help='设备')
    eval_parser.add_argument('--num-games', type=int, default=100, help='评估局数')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return
    
    if args.command == 'serve':
        cmd_serve(args)
    elif args.command == 'train':
        cmd_train(args)
    elif args.command == 'eval':
        cmd_eval(args)


if __name__ == '__main__':
    main()


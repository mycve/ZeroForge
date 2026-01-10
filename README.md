# ZeroForge - 中国象棋 Gumbel MuZero AI

基于 JAX + mctx 的中国象棋 AI，使用 Gumbel MuZero 算法自我对弈进化。

## 特性

- **Gumbel MuZero**: 高效的 MCTS 搜索，100 次模拟即可达到好效果
- **全 JAX JIT**: 自我对弈和训练完全 JIT 编译，GPU 高利用率
- **批量并行**: 512 局游戏同时进行，充分利用 GPU
- **完整象棋规则**: 将军、将死、长将、三次重复等
- **ELO 评估**: 自动评估模型强度
- **TensorBoard**: 实时监控训练进度
- **断点继续**: 自动保存和恢复检查点

## 快速开始

```bash
# 安装依赖
pip install -e .

# 开始训练
python train.py

# 查看训练日志
tensorboard --logdir logs
```

## 项目结构

```
ZeroForge/
├── train.py           # 训练入口
├── configs/
│   └── default.yaml   # 配置文件
├── networks/          # 神经网络
│   ├── muzero.py      # MuZero 网络
│   ├── convnext.py    # ConvNeXt 骨干
│   └── heads.py       # 输出头
├── xiangqi/           # 象棋环境
│   ├── env.py         # JAX 环境
│   ├── rules.py       # 规则
│   ├── actions.py     # 动作编码
│   └── mirror.py      # 数据增强
└── gui/
    └── web_gui.py     # Gradio 人机对弈
```

## 配置

编辑 `configs/default.yaml`:

```yaml
# 并行游戏数 (根据显存调整)
self_play:
  num_parallel_games: 512

# MCTS 模拟次数
mcts:
  num_simulations: 100

# 训练
training:
  batch_size: 512
  learning_rate: 0.0003
```

## 训练输出

```
[2026-01-11] ZeroForge - 中国象棋 Gumbel MuZero
[2026-01-11] 设备: [CudaDevice(id=0), ...]
[2026-01-11] TensorBoard: tensorboard --logdir logs
[2026-01-11] 开始训练...
[2026-01-11] step=512, loss=2.34, samples=51200, elo=1500.0
[2026-01-11] 评估: elo=1523.5, win_rate=54.00%
[2026-01-11] 检查点已保存: step=1000, elo=1523.5
```

## 技术细节

### 算法
- **Gumbel MuZero**: 使用 Gumbel-Top-k 技巧高效采样动作
- **mctx**: Google DeepMind 的 MCTS 库

### 网络架构
- **Representation**: ConvNeXt 编码器
- **Dynamics**: 预测下一状态
- **Prediction**: 策略和价值头

### 观察空间
- 形状: `(240, 10, 9)`
- 16 步历史 + 当前局面
- 每步 14 个平面 (7 种棋子 × 2 方)

### 动作空间
- 大小: 2550
- 编码: 起点 (90) × 终点 (90) 的子集

## 依赖

- JAX + CUDA
- Flax
- mctx
- Optax
- TensorBoard

## License

MIT

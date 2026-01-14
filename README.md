# ZeroForge - 中国象棋 Gumbel AlphaZero

基于 **JAX** + **mctx** 实现的现代化中国象棋 AI，采用 Gumbel AlphaZero 算法，从零开始自我进化。

## 核心特性

- **Gumbel AlphaZero**: DeepMind 的 Gumbel-Top-k 策略改善算法，低模拟次数（128 次）即可产生强训练信号
- **经验回放**: 样本平均复用 4 次，提高数据利用效率
- **断点续训**: 基于 orbax-checkpoint 的完整状态保存，支持无差别恢复
- **n-step TD**: MuZero 风格的 n-step 时序差分目标，平衡方差与偏差
- **视角归一化**: 始终以当前玩家为中心观察，简化网络学习
- **镜像增强**: 训练时自动左右镜像变换，数据利用率翻倍
- **极速 JAX 优化**:
  - 全向量化规则，消除 Python 循环
  - `jax.pmap` 多设备分发 + `jax.lax.scan` 循环优化
  - 编译缓存，二次启动秒开

## 快速开始

### 1. 安装依赖

```bash
pip install -e .
```

或手动安装：
```bash
pip install jax[cuda12] flax optax mctx orbax-checkpoint tensorboardX gradio
```

### 2. 开始训练

```bash
python train.py
```

训练会自动：
- 检测并恢复已有 checkpoint（断点续训）
- 保存 checkpoint 到 `checkpoints/` 目录
- 输出日志到 TensorBoard

### 3. 监控训练

```bash
tensorboard --logdir logs
```

### 4. 人机对弈

```bash
# 自动加载最新模型
python play.py

# 或指定 checkpoint
python play.py checkpoints/100
```

## 训练输出说明

```
iter= 10 | ploss=2.81 vloss=0.15 | len=153 fps=516 buf=358k train=800 | 红403 黑195 和426
```

| 指标 | 说明 |
|------|------|
| `ploss` | 策略损失（交叉熵） |
| `vloss` | 价值损失（L2） |
| `len` | 平均对局步数 |
| `fps` | 每秒采样帧数 |
| `buf` | 经验回放缓冲区大小 |
| `train` | 本次迭代训练批次数 |
| `红/黑/和` | 本次迭代的胜负和统计 |

## Checkpoint 管理

### 目录结构

```
checkpoints/
├── 10/              # orbax checkpoint (模型参数、优化器状态)
├── 20/
├── meta_10/         # 额外元数据
│   ├── metadata.json       # ELO 记录
│   ├── history_*.npz       # 历史模型（用于 ELO 评估）
│   └── replay_buffer.npz   # 经验回放缓冲区
└── meta_20/
```

### 迁移到其他机器

```bash
# 只需对弈
scp -r remote:checkpoints/100 ./checkpoints/

# 断点续训
scp -r remote:checkpoints/100 ./checkpoints/
scp -r remote:checkpoints/meta_100 ./checkpoints/
```

## 默认配置

| 参数 | 值 | 说明 |
|------|-----|------|
| `num_channels` | 128 | 网络通道数 |
| `num_blocks` | 8 | 残差块数量 |
| `num_simulations` | 128 | MCTS 模拟次数 |
| `selfplay_batch_size` | 1024 | 自对弈并行数 |
| `training_batch_size` | 1024 | 训练批大小 |
| `replay_buffer_size` | 500000 | 回放缓冲区容量 |
| `sample_reuse_times` | 4 | 样本复用次数 |
| `td_steps` | 10 | n-step TD 步数 |

## 项目结构

```
ZeroForge/
├── train.py           # 训练入口
├── play.py            # 人机对弈
├── networks/
│   └── alphazero.py   # AlphaZero 网络 (ResNet + SE + LayerNorm)
├── xiangqi/
│   ├── env.py         # 环境状态管理
│   ├── rules.py       # 向量化规则校验
│   ├── actions.py     # 动作编码
│   └── mirror.py      # 镜像增强
├── gui/
│   └── web_gui.py     # Web 对弈界面
├── checkpoints/       # 模型存档
└── logs/              # TensorBoard 日志
```

## License

MIT License

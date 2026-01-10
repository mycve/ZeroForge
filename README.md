# ZeroForge 🀄

> **中国象棋 Gumbel MuZero AI** - 基于 JAX/Flax 的强化学习训练框架

## 特性

- 🚀 **Gumbel MuZero** - 最先进算法，仅需 50-200 次模拟（传统需 800）
- ⚡ **JAX 加速** - 纯 JAX 实现，支持 JIT 编译和多 GPU 数据并行
- 🧠 **ConvNeXt 网络** - 现代卷积神经网络架构，16 步历史状态输入
- 🎮 **完整规则** - 纯 JAX 实现的中国象棋引擎，支持长将、重复局面检测
- 🌐 **Web 界面** - Gradio Web GUI，支持人机对弈、FEN 导入测试
- 📊 **训练监控** - TensorBoard 集成、ELO 评分、检查点管理

## 安装

```bash
# 克隆仓库
git clone https://github.com/mycve/zeroforge.git
cd zeroforge

# 安装依赖 (GPU 版本)
pip install -e .

# 或 CPU 版本
pip install -e ".[cpu]"
```

## 快速开始

### 训练模型

```bash
# 使用默认配置训练
uv run python main.py train

# 使用自定义配置
uv run python main.py train --config configs/default.yaml

# 从检查点继续训练
uv run python main.py train --resume
```

### Web 界面对弈 (推荐)

```bash
# 双人模式 - 打开 http://localhost:7860
python main.py web

# 使用训练好的模型对弈
python main.py web --checkpoint checkpoints/

# 分享到公网 (生成临时链接)
python main.py web --share

# 调整 AI 思考深度
python main.py web --checkpoint checkpoints/ --simulations 400
```

### CLI 对弈

```bash
python main.py play --checkpoint checkpoints/
```

## 项目结构

```
ZeroForge/
├── main.py                 # 主入口
├── configs/
│   └── default.yaml        # 训练配置
├── xiangqi/                # 中国象棋引擎 (纯 JAX)
│   ├── env.py              # 游戏环境
│   ├── rules.py            # 规则实现
│   ├── actions.py          # 动作空间
│   └── mirror.py           # 数据增强
├── networks/               # 神经网络
│   ├── muzero.py           # MuZero 网络
│   ├── convnext.py         # ConvNeXt 骨干
│   └── heads.py            # 输出头
├── mcts/                   # 蒙特卡洛树搜索
│   └── search.py           # Gumbel MCTS
├── training/               # 训练模块
│   ├── trainer.py          # 训练器
│   ├── replay_buffer.py    # 经验回放
│   ├── checkpoint.py       # 检查点
│   └── logging.py          # 日志
├── evaluation/             # 评估模块
│   ├── arena.py            # 对弈竞技场
│   └── elo.py              # ELO 评分
├── gui/                    # Web 界面
│   └── web_gui.py          # Gradio GUI
└── cli/                    # 命令行界面
    └── play.py             # CLI 对弈
```

## 技术细节

### Gumbel MuZero 优势

| 特性 | AlphaZero/MuZero | Gumbel MuZero |
|------|------------------|---------------|
| MCTS 模拟次数 | 800 | **50-200** |
| 策略改进 | 访问计数 | Sequential Halving |
| 探索策略 | UCB | Gumbel-Top-k |

### 观察空间

- **形状**: `(240, 10, 9)`
- **内容**: 
  - 当前棋盘 + 16 步历史 (每步 14 通道 = 7 棋子类型 × 2 颜色)
  - 当前玩家通道
  - 步数通道

### 动作空间

- **大小**: 2086 个离散动作
- **编码**: 压缩的 (起点, 终点) 对，仅包含合法移动模式

### 网络架构

```
观察 (240, 10, 9)
    │
    ▼
┌─────────────────┐
│ Representation  │  ConvNeXt (12 blocks)
│    Network      │  → 隐藏状态 (384, 10, 9)
└─────────────────┘
    │
    ├─────────────────────┐
    ▼                     ▼
┌─────────────┐    ┌─────────────┐
│  Dynamics   │    │ Prediction  │
│   Network   │    │   Network   │
│ (6 blocks)  │    │ (6 blocks)  │
└─────────────┘    └─────────────┘
    │                     │
    ▼                     ├───────┬───────┐
 下一状态              策略    价值    奖励
```

### 规则实现

- ✅ 所有棋子移动规则（将、士、象、马、车、炮、兵）
- ✅ 蹩马腿、塞象眼
- ✅ 将帅对面
- ✅ 将军检测
- ✅ 将死/困毙判定
- ✅ 重复局面检测 (Zobrist 哈希，三次重复判和)
- ✅ 长将检测 (连续将军 6 次判负)
- ✅ 和棋规则 (200 步/120 步无吃子)

## 配置说明

默认配置针对 **8×GPU (32GB) + 128核 CPU** 优化:

```yaml
# 网络配置
network:
  hidden_dim: 384           # 隐藏层维度
  repr_blocks: 12           # 表示网络深度
  dyn_blocks: 6             # 动态网络深度
  pred_blocks: 6            # 预测网络深度

# MCTS 配置
mcts:
  num_simulations: 100      # Gumbel MuZero 不需要太多
  discount: 1.0             # 棋类游戏用 1.0
  temperature_threshold: 30 # 前 30 步高温度探索
  temperature_high: 1.0     # 探索温度
  temperature_low: 0.25     # 利用温度

# 训练配置
training:
  batch_size: 512           # 每 GPU，8 GPU 总共 4096
  learning_rate: 0.003      # 大 batch 需要更高 LR
  value_loss_weight: 1.0    # 棋类游戏 value 重要
```

<details>
<summary>小规模配置 (单 GPU)</summary>

```yaml
network:
  hidden_dim: 256
  repr_blocks: 8
  dyn_blocks: 4
  pred_blocks: 4

training:
  batch_size: 256
  learning_rate: 0.0002

self_play:
  num_parallel_games: 32
```

</details>

## 依赖

- Python >= 3.12
- JAX >= 0.4.30 (支持 CUDA 12)
- Flax >= 0.8.0
- mctx >= 0.0.5
- Gradio >= 4.0.0

## 参考

- [Gumbel MuZero (Danihelka et al., 2022)](https://arxiv.org/abs/2104.06303) - Policy improvement by planning with Gumbel
- [MuZero (Schrittwieser et al., 2020)](https://arxiv.org/abs/1911.08265) - Mastering Atari, Go, Chess and Shogi
- [AlphaZero (Silver et al., 2018)](https://arxiv.org/abs/1712.01815) - Mastering Chess and Shogi
- [mctx - JAX MCTS Library](https://github.com/google-deepmind/mctx)

## License

MIT License

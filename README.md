# ZeroForge - 中国象棋 Gumbel AlphaZero

ZeroForge 是一个基于 **JAX** + **mctx** 实现的现代化、极简主义中国象棋 AI。它采用 Gumbel AlphaZero 算法，通过自我对弈（Self-play）实现从零开始的自我进化。

## 核心特性

- **Gumbel AlphaZero**: 采用 DeepMind 的 Gumbel-Top-k 策略改善算法，在极低模拟次数（如 64-96 次）下即可产生强力的训练信号。
- **现代化架构**:
    - **算力随机化 (Compute Randomization)**: 自对象中采用“导师（强算力）- 学生（弱算力）”博弈模式，学生对局以低权重参与训练，增强模型鲁棒性。
    - **视角归一化 (Perspective Normalization)**: 始终以当前玩家为中心进行观察，简化网络学习难度。
    - **镜像增强 (Mirror Augmentation)**: 训练时自动进行左右镜像变换，数据利用率翻倍。
- **极速 JAX 优化**:
    - **全向量化规则**: 象棋规则（将军、将死、重复局面等）完全使用 JAX 算子重写，消除 Python 循环，极大减小计算图规模。
    - **并行预编译**: 启动时并行预热 Selfplay 和 Train 算子，解决 JAX 编译耗时问题。
    - **高 GPU 利用率**: 通过 `jax.pmap` 多设备分发和 `jax.lax.scan` 循环优化，确保 GPU 持续满载运行。
- **顶级网络架构**: 
    - **ResNet + SE**: 经典的残差网络结构，集成 **Squeeze-and-Excitation (SE)** 通道注意力机制（KataGo/Lc0 同款配置）。
- **全方位监控**: 
    - 实时统计胜负、四种细分和棋原因（步数超限、无吃子、三次重复、长将）。
    - 监控平均对局长度、FPS（采样吞吐量）及标准 ELO 等级分演变。

## 快速开始

### 1. 安装依赖
确保已安装 JAX（建议配合 CUDA 使用）：
```bash
pip install -U jax jaxlib flax mctx optax chex
```

### 2. 开始训练
直接运行主脚本即可启动：
```bash
python train.py
```

### 3. 可视化监控
启动 TensorBoard 查看详细训练指标：
```bash
tensorboard --logdir logs
```

## 技术规格

- **观察空间**: `(240, 10, 9)`。包含 16 步历史轨迹，视角归一化后的棋子位置及步数信息。
- **动作空间**: `2086`。经过压缩编码的中国象棋所有几何合法动作空间。
- **默认配置**:
    - **网络**: 256 通道 x 12 个 SE 残差块。
    - **搜索**: 导师 96 次模拟 / 学生 32 次模拟。
    - **批大小**: 512 (Selfplay) / 512 (Training)。

## 项目结构

```text
ZeroForge/
├── train.py           # 训练入口：包含 Selfplay、Train、Evaluate 核心循环
├── networks/          # 神经网络
│   └── alphazero.py   # AlphaZero 网络 (ResNet + SE)
├── xiangqi/           # 象棋环境 (高性能 JAX 实现)
│   ├── env.py         # 状态管理与视角归一化
│   ├── rules.py       # 纯向量化规则校验
│   ├── actions.py     # 动作压缩编码与解码
│   └── mirror.py      # 左右镜像数据增强
└── gui/
    └── web_gui.py     # 基于 Web 的人机对弈界面
```

## 训练指标说明

在控制台输出或 TensorBoard 中，您可以监控：
- **ploss / vloss**: 策略和价值损失。
- **len**: 平均对局步数。
- **fps**: 每秒生成的采样帧数。
- **和棋统计**: 区分“步数上限(步)”、“无吃子(抓)”、“三次重复(复)”和“长将(将)”。
- **ELO**: 与过去模型对局评估出的强度等级。

## License

[MIT License](LICENSE)

# ZeroForge - 中国象棋 Gumbel AlphaZero

基于 **JAX** + **mctx** 实现的现代化中国象棋 AI，采用 Gumbel AlphaZero 算法，从零开始自我进化。

## 核心特性

- **Gumbel AlphaZero**: Gumbel-Top-k MCTS，自对弈按温度退火采样；评估关闭 Gumbel 扰动以保证可比性
- **3 分支 GNN 网络**: Local 8 邻居 + Row + Col 注意力，配合 factorized policy head、全局动作先验与跨度上下文
- **规则状态平面**: 将重复、无吃子、长将/长捉压力等规则信息作为额外观察通道输入网络
- **经验回放**: 样本可复用，提高数据利用效率
- **断点续训**: 基于 orbax-checkpoint + lz4 经验池快照的完整状态保存，支持无差别恢复
- **TD(λ) 价值目标**: 对 MCTS 根标量 `root_value` 做时序差分备份，与 `value`（由 `value_logits` 得 W−L）做 MSE
- **视角归一化**: 始终以当前玩家为中心观察，简化网络学习
- **镜像增强**: 训练时自动左右镜像变换，数据利用率翻倍
- **BF16 训练路径**: 默认使用 `bfloat16` 网络前向，评估路径固定 float32 以规避部分 GPU 编译问题
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
pip install jax[cuda12] flax optax mctx orbax-checkpoint tensorboardX lz4 fastapi "uvicorn[standard]"
```

### 2. 开始训练

```bash
python train.py
```

训练会自动：
- 检测并恢复已有 checkpoint（断点续训）
- 保存 checkpoint 到 `checkpoints/` 目录
- 输出日志到 TensorBoard

也可以导入一个指定模型作为基础模型，再从 `iteration=0` 开始自玩训练：

```bash
# 使用当前 checkpoints 目录下的 step
python train.py --init-checkpoint 100

# 或直接指定 checkpoint 目录
python train.py --init-checkpoint checkpoints/100

# 导入强模型后，用更保守的参数继续训练
python train.py --init-checkpoint checkpoints/100 \
  --learning-rate 5e-5 \
  --sample-reuse-times 1 \
  --selfplay-temperature-steps 30 \
  --selfplay-gumbel-scale 0.5
```

说明：
- 如果当前 `checkpoints/` 里已经存在可恢复的训练断点，仍然会优先断点续训
- `--init-checkpoint` 只在“没有现成训练断点”时生效
- 改变 `num_channels`、`num_blocks` 或策略头结构后，旧 checkpoint 通常不能直接续训；请使用新的 checkpoint 目录重新开始

常用覆盖参数：

```bash
python train.py \
  --num-channels 128 \
  --num-simulations 40 \
  --training-batch-size 4096 \
  --td-lambda 0.75
```

宽模型或 CUDA allocator 不稳定时，可先启用更保守的释放策略定位问题：

```bash
ZEROFORGE_STABLE_CUDA_ALLOCATOR=1 python train.py --selfplay-batch-size 512
```

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
iter= 10 | ploss=2.81 vloss=0.15 | len=153 fps=516 buf=358k train=300 | ent=1.92(开2.04/中1.66) | 红403 黑195 和426
```

| 指标 | 说明 |
|------|------|
| `ploss` | 完整策略蒸馏损失（对 MCTS `action_weights` 全动作分布的交叉熵） |
| `vloss` | 价值损失（对 `value` 与 `value_tgt` 的均方误差 MSE） |
| `len` | 平均对局步数 |
| `fps` | 每秒采样帧数 |
| `buf` | 经验回放缓冲区大小 |
| `train` | 本次迭代训练批次数 |
| `ent` | 根 visit 分布熵，括号内为开局/中后局分段统计 |
| `红/黑/和` | 本次迭代的胜负和统计 |

## Checkpoint 管理

### 目录结构

```
checkpoints/
├── 10/              # orbax checkpoint (模型参数、优化器状态)
├── 20/
├── metadata.json    # ELO 记录、total_opt_steps（单文件，评估对手从 orbax 加载）
└── replay_buffer.lz4 # lz4 压缩经验池快照，用于无损断点续训
```

### 迁移到其他机器

```bash
# 只需对弈
scp -r remote:checkpoints/100 ./checkpoints/

# 断点续训
scp -r remote:checkpoints/100 ./checkpoints/
scp remote:checkpoints/metadata.json ./checkpoints/
scp remote:checkpoints/replay_buffer.lz4 ./checkpoints/
```

## 默认配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_channels` | 128 | GNN 通道数；当前强度基线优先保证搜索和训练质量 |
| `num_blocks` | 10 | GraphBlock 数量 |
| `network_dtype` | `bfloat16` | 训练网络 dtype；评估固定 float32 |
| `num_simulations` | 40 | MCTS 模拟次数 |
| `selfplay_batch_size` | 1024 | 自对弈并行数 |
| `training_batch_size` | 4096 | 训练批大小 |
| `learning_rate` | 2e-4 | AdamW peak LR |
| `lr_warmup_steps` | 2000 | 学习率 warmup 步数 |
| `lr_cosine_steps` | 100000 | 余弦退火周期（优化步） |
| `lr_min_ratio` | 0.2 | 最低学习率比例 |
| `replay_buffer_size` | 2000000 | 回放缓冲区容量 |
| `sample_reuse_times` | 3 | 样本复用次数 |
| `td_lambda` | 0.75 | TD(λ) 系数 |
| `selfplay_temperature` | 1.0 | 自对弈初始采样温度 |
| `selfplay_temperature_final` | 0.25 | 退火后的最低采样温度 |
| `selfplay_temperature_steps` | 60 | 温度退火半步数 |
| `top_k` | 8 | Gumbel 根节点候选动作数 |

## 项目结构

```
ZeroForge/
├── train.py           # 训练入口
├── play.py            # 人机对弈
├── networks/
│   └── alphazero.py   # 3 分支 GNN AlphaZero 网络
├── xiangqi/
│   ├── env.py         # 环境状态管理
│   ├── rules.py       # 向量化规则校验
│   ├── violation_rules.py # 长将/长捉等判负规则
│   ├── actions.py     # 动作编码
│   ├── fen.py         # FEN 解析与评估局面加载
│   └── mirror.py      # 镜像增强
├── gui/
│   ├── api.py         # FastAPI 对弈接口
│   ├── web_gui.py     # Web 对弈服务入口
│   └── static/        # 前端静态资源
├── uci_engine.py      # UCI 引擎入口
├── generate_eval_fens.py # 生成固定评估局面
├── eval_fens.txt      # 固定评估 FEN 池
├── checkpoints/       # 模型存档
└── logs/              # TensorBoard 日志
```

## License

MIT License

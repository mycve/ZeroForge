# ZeroForge

ZeroForge 是一个现代化的 AlphaZero/MuZero 训练框架，提供完整的自博弈强化学习流程，支持多线程异步自玩、DDP 多卡训练、Web 可视化界面和游戏对弈功能。

## 特性

- 🎮 **游戏插件化** - 通过 `Game` 接口快速接入新游戏，自动验证和注册
- 🧠 **算法模块化** - 支持 AlphaZero、MuZero，可扩展
- 🔍 **完整 MCTS** - GPU 批推理 + CPU 本地树搜索
- 🚀 **高效并行** - 多线程异步自玩 + 叶节点批量推理
- 🖥️ **分布式训练** - 支持 DDP 多卡训练
- 🌐 **Web 界面** - React 前端 + FastAPI 后端，实时训练监控
- ⚙️ **智能配置** - 自动设备检测（CUDA > MPS > CPU），网络大小自适应
- 💾 **检查点管理** - 自动保存、加载模型，支持手动保存
- 🎯 **游戏对弈** - 人机对弈、AI vs AI、随机玩家

## 系统架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          DistributedTrainer                             │
│                                                                         │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                 num_envs 个环境 (各自开线程持续自玩)              │   │
│   │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐               │   │
│   │  │  Env 0  │ │  Env 1  │ │  Env 2  │ │  ...N   │               │   │
│   │  │  Game   │ │  Game   │ │  Game   │ │  Game   │ ← 每个 env    │   │
│   │  │  MCTS   │ │  MCTS   │ │  MCTS   │ │  MCTS   │   持续自玩    │   │
│   │  └────┬────┘ └────┬────┘ └────┬────┘ └────┬────┘               │   │
│   │       │           │           │           │                     │   │
│   │       └───────────┴─────┬─────┴───────────┘                     │   │
│   │                         ↓ submit(obs, mask)                     │   │
│   └─────────────────────────┼───────────────────────────────────────┘   │
│                             ↓                                           │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                  GPU 推理池 (LeafBatcher)                        │   │
│   │  - 收集叶节点请求，达到 batch_size 或 timeout 后批量推理         │   │
│   │  - 分发 (policy, value) 结果给各 env                            │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                             │                                           │
│                             ↓ 游戏结束产生轨迹                          │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                      ReplayBuffer                                │   │
│   │  - 存储完整游戏轨迹                                              │   │
│   └─────────────────────────────────────────────────────────────────┘   │
│                             │                                           │
│         当 buffer 达到 min_buffer_size 时，暂停推理，切换到训练模式     │
│                             ↓                                           │
│   ┌─────────────────────────────────────────────────────────────────┐   │
│   │                   训练阶段 (可选 DDP 多卡)                        │   │
│   │  - 从 ReplayBuffer 采样 train_batches_per_epoch 批次             │   │
│   │  - 计算损失，更新网络                                            │   │
│   │  - 完成后继续推理模式                                            │   │
│   └─────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────────┘
```

### 核心组件

| 组件 | 文件 | 说明 |
|------|------|------|
| `DistributedTrainer` | `core/trainer.py` | 分布式训练器，支持 DDP 多卡 |
| `LeafBatcher` | `core/mcts/batcher.py` | GPU 推理池，收集叶节点批量推理 |
| `AsyncSelfPlayWorker` | `core/trainer.py` | 持续运行的自玩工作线程 |
| `LocalMCTSTree` | `core/mcts/tree.py` | 本地 MCTS 树实现 |
| `ReplayBuffer` | `core/replay_buffer.py` | 经验回放缓冲区 |

### 工作流程

```
1. num_envs 个环境持续运行，各自执行 MCTS 搜索
2. MCTS 遇到叶节点 → batcher.submit(obs, mask) → 线程阻塞等待
3. LeafBatcher 收集请求 → 达到 batch_size 或 timeout → 批量 GPU 推理
4. 推理完成 → 分发结果 → 各 env 继续 MCTS 搜索
5. 游戏结束 → 轨迹放入 ReplayBuffer → 开始新游戏
6. Buffer 达到阈值 → 暂停推理 → 训练 train_batches_per_epoch 批次 → 继续推理
```

## 快速开始

### 安装（使用 uv）

```bash
# 安装 uv（如果未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 克隆仓库
git clone https://github.com/your-repo/ZeroForge.git
cd ZeroForge

# 同步依赖（自动创建虚拟环境）
uv sync
```

### 启动 Web 服务

```bash
# 启动后端 API 服务
uv run python main.py server

# 或指定端口
uv run uvicorn server.api:app --reload --port 8000
```

启动前端（另开终端）：

```bash
cd web
npm install
npm run dev
```

访问：
- 前端界面: http://localhost:3000
- API 文档: http://localhost:8000/docs

### 后端Ui功能截图
![训练流程](assets/train_game.png)
![debug流程](assets/debug.png)
![debug mcts访问分布流程](assets/debug_mcts.png)
![debug mcts访问分布流程](assets/cchess.png)

### 命令行训练

```bash
# 训练井字棋（简单游戏，快速验证，默认配置）
uv run python main.py train

# 训练井字棋（指定参数）
uv run python main.py train --game tictactoe --algorithm alphazero --epochs 100

# 训练中国象棋（复杂游戏，使用大网络）
uv run python main.py train --game chinese_chess --algorithm alphazero --network-size large --epochs 1000

# 多卡 DDP 训练
torchrun --nproc_per_node=4 main.py train --game chinese_chess --use-ddp
```

## 项目结构

```
ZeroForge/
├── core/                       # 核心框架
│   ├── game.py                 # 游戏抽象基类 (Game, GameState, GameMeta)
│   ├── algorithm.py            # 算法抽象基类 (Algorithm, Trajectory)
│   ├── config.py               # 配置类 (MCTSConfig, BatcherConfig)
│   ├── training_config.py      # 训练配置 (TrainingConfig)
│   ├── trainer.py              # 分布式训练器 (DistributedTrainer) ⭐
│   ├── selfplay.py             # 多线程自玩 (ThreadedSelfPlay, EnvWorker)
│   ├── mcts/                   # MCTS 实现
│   │   ├── node.py             # 树节点 (MCTSNode)
│   │   ├── tree.py             # 本地树 (LocalMCTSTree)
│   │   ├── search.py           # 搜索控制器 (MCTSSearch)
│   │   └── batcher.py          # GPU 批推理 (LeafBatcher) ⭐
│   ├── replay_buffer.py        # 经验回放缓冲区
│   ├── checkpoint.py           # 检查点管理
│   └── logging.py              # 结构化日志
│
├── games/                      # 游戏实现
│   ├── __init__.py             # 注册系统 + 验证
│   ├── tictactoe/              # 井字棋
│   │   ├── game.py             # 游戏逻辑
│   │   └── config.py           # 游戏配置
│   └── chinese_chess/          # 中国象棋
│       ├── game.py             # 游戏逻辑
│       ├── config.py           # 游戏配置
│       └── cchess/             # 象棋引擎
│
├── algorithms/                 # 算法实现
│   ├── __init__.py             # 注册系统
│   ├── alphazero/              # AlphaZero
│   │   ├── algorithm.py        # 算法逻辑
│   │   └── network.py          # 神经网络
│   └── muzero/                 # MuZero
│       ├── algorithm.py        # 算法逻辑
│       └── network.py          # 神经网络
│
├── server/                     # Web 后端
│   ├── api.py                  # FastAPI 路由
│   └── manager.py              # 训练/游戏管理器
│
├── web/                        # React 前端
│   └── src/
│       ├── pages/              # 页面组件
│       └── components/         # 通用组件
│
└── main.py                     # CLI 入口
```

## 配置说明

训练配置 `TrainingConfig`：

```python
from core.training_config import TrainingConfig

config = TrainingConfig(
    # === 游戏和算法 ===
    game_type="chinese_chess",      # 游戏类型
    algorithm="alphazero",          # 算法 (alphazero/muzero)
    
    # === 网络 ===
    network_size="auto",            # auto/small/medium/large
    num_channels=128,               # 网络通道数（medium/large）
    num_blocks=6,                   # ResNet 块数（medium/large）
    
    # === 并发（自玩）===
    num_envs=8,                     # 并行环境数（每个 env 自动开线程）
    train_batches_per_epoch=10,     # 每 epoch 训练批次数
    
    # === 批处理 ===
    batch_size=256,                 # 训练批大小
    inference_batch_size=64,        # 叶节点推理批大小（越大 GPU 利用率越高）
    inference_timeout_ms=5.0,       # 推理超时（ms）
    
    # === 训练超参 ===
    num_epochs=100,                 # 训练轮数
    lr=1e-3,                        # 学习率
    weight_decay=1e-4,              # 权重衰减
    grad_clip=1.0,                  # 梯度裁剪
    
    # === MCTS ===
    num_simulations=100,            # 每步模拟次数
    c_puct=1.5,                     # UCB 探索常数
    dirichlet_alpha=0.3,            # Dirichlet 噪声
    dirichlet_epsilon=0.25,         # 噪声比例
    
    # === 回放缓冲区 ===
    replay_buffer_size=100000,      # 缓冲区容量
    min_buffer_size=500,            # 最小样本数
    
    # === 分布式训练 ===
    use_ddp=False,                  # 是否启用 DDP
    ddp_backend="nccl",             # DDP 后端 (nccl/gloo)
    
    # === 系统 ===
    device="auto",                  # auto/cuda/mps/cpu
)
```

### 关键参数说明

| 参数 | 说明 | 推荐值 |
|------|------|--------|
| `num_envs` | 并行环境数（决定并发度） | CPU 核心数 |
| `inference_batch_size` | GPU 推理池批大小（**≤ num_envs**） | **num_envs / 2** |
| `inference_timeout_ms` | 推理超时（ms） | 5-10ms |
| `train_batches_per_epoch` | 每 epoch 训练批次数 | 10-50 |
| `num_simulations` | MCTS 模拟次数 | 100-800 |
| `min_buffer_size` | 开始训练的最小样本数 | 500-1000 |

> ⚠️ **注意**: 
> - `inference_batch_size` 不能超过 `num_envs`（每个 env 同时只产生一个叶子节点）
> - **推荐设置为 `num_envs / 2`**：形成流水线，当 GPU 推理一批时，其他 env 产生下一批，GPU 始终忙碌

### 网络大小自动选择

- `small`: 井字棋等小游戏（MLP，~1K 参数）
- `medium`: 中等规模游戏（ConvNeXt 3层）
- `large`: 中国象棋等复杂游戏（ConvNeXt 6层，~100K+ 参数）

## DDP 多卡训练

```bash
# 单机 4 卡
torchrun --nproc_per_node=4 main.py train \
    --game chinese_chess \
    --algorithm alphazero \
    --use-ddp \
    --epochs 1000

# 或使用 Python API
from core.trainer import DistributedTrainer
from core.training_config import TrainingConfig

config = TrainingConfig(
    game_type="chinese_chess",
    use_ddp=True,
    ddp_backend="nccl",
)

trainer = DistributedTrainer(config)
trainer.setup()
trainer.run()
```

### DDP 架构

```
┌────────────────────────────────────────────────────────────────────┐
│                        Rank 0 (主进程)                             │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │                    自玩 + 推理                                │ │
│  │  LeafBatcher ← EnvWorker × N                                 │ │
│  │       ↓                                                       │ │
│  │  ReplayBuffer                                                 │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                              ↓                                     │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │                    训练 (DDP)                                 │ │
│  │  Network (device:0) ←→ AllReduce ←→ Network (device:1-3)    │ │
│  └──────────────────────────────────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────┘
                              ↓ 梯度同步
┌────────────────────────────────────────────────────────────────────┐
│  Rank 1-3: 只参与训练梯度同步，不执行自玩                           │
└────────────────────────────────────────────────────────────────────┘
```

## 添加新游戏

1. 创建游戏目录 `games/mygame/`
2. 实现 `Game` 接口：

```python
from core.game import Game, GameState, GameMeta, ObservationSpace, ActionSpace
from games import register_game

@register_game("mygame")
class MyGame(Game):
    """我的游戏实现"""
    
    # === 必须实现的类方法 ===
    
    @classmethod
    def get_meta(cls) -> GameMeta:
        """返回游戏元数据"""
        return GameMeta(
            name="我的游戏",
            description="游戏描述",
            tags=["board", "strategy"],
            min_players=2,
            max_players=2,
        )
    
    # === 必须实现的属性 ===
    
    @property
    def observation_space(self) -> ObservationSpace:
        return ObservationSpace(shape=(C, H, W))
    
    @property
    def action_space(self) -> ActionSpace:
        return ActionSpace(n=100)
    
    @property
    def num_players(self) -> int:
        return 2
    
    @property
    def supported_render_modes(self) -> list:
        return ["text", "json"]  # 至少支持 text
    
    # === 必须实现的方法 ===
    
    def reset(self) -> np.ndarray:
        """重置游戏，返回初始观测"""
        ...
    
    def step(self, action: int) -> tuple:
        """执行动作，返回 (obs, reward, done, info)"""
        ...
    
    def legal_actions(self) -> list:
        """返回合法动作列表"""
        ...
    
    def clone(self) -> "MyGame":
        """克隆游戏状态（用于 MCTS）"""
        ...
    
    def render(self, mode: str = "text") -> Any:
        """渲染游戏状态"""
        if mode not in self.supported_render_modes:
            raise ValueError(f"不支持: {mode}")
        if mode == "text":
            return {"type": "text", "text": "..."}
        elif mode == "json":
            return {"type": "grid", "cells": [...]}
```

## Web API

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/training/start` | POST | 启动训练 |
| `/api/training/stop` | POST | 停止训练 |
| `/api/training/status` | GET | 获取训练状态 |
| `/api/training/save` | POST | 手动保存检查点 |
| `/api/games` | GET | 列出所有游戏 |
| `/api/games/{id}/start` | POST | 开始对弈 |
| `/api/games/{id}/action` | POST | 执行动作 |
| `/api/checkpoints` | GET | 列出检查点 |
| `/api/config/schema` | GET | 获取配置 schema |
| `/ws/training` | WebSocket | 实时训练状态 |

## 技术栈

- **深度学习**: PyTorch 2.x + DDP
- **并行**: Python threading + GPU 批推理
- **Web 后端**: FastAPI + WebSocket
- **Web 前端**: React + Vite + Zustand
- **配置管理**: dataclasses

## 路线图

- [x] 核心框架（Game, Algorithm, MCTS）
- [x] AlphaZero 实现
- [x] MuZero 实现
- [x] 井字棋游戏
- [x] 中国象棋游戏
- [x] Web 训练界面
- [x] 游戏对弈功能
- [x] 检查点管理
- [x] 网络大小自适应
- [x] 多线程异步自玩
- [x] 叶节点批量推理
- [x] DDP 分布式训练
- [ ] 更多游戏（围棋、国际象棋）
- [ ] 混合精度训练 (AMP)
- [ ] 模型量化推理

## License

MIT License

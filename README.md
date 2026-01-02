# ZeroForge

ZeroForge 是一个现代化的 AlphaZero/MuZero 训练框架，提供完整的自博弈强化学习流程，包括 Web 可视化界面、实时监控和游戏对弈功能。

## 特性

- 🎮 **游戏插件化** - 通过 `Game` 接口快速接入新游戏，自动验证和注册
- 🧠 **算法模块化** - 支持 AlphaZero、MuZero，可扩展
- 🔍 **完整 MCTS** - GPU 批推理 + CPU 本地树搜索
- 🌐 **Web 界面** - React 前端 + FastAPI 后端，实时训练监控
- ⚙️ **智能配置** - 自动设备检测（CUDA > MPS > CPU），网络大小自适应
- 💾 **检查点管理** - 自动保存、加载模型，支持手动保存
- 🎯 **游戏对弈** - 人机对弈、AI vs AI、随机玩家

## 快速开始

### 安装（使用 uv

```bash
# 安装 uv（如果未安装）
curl -LsSf https://astral.sh/uv/install.sh | sh

# 克隆仓库
git clone https://github.com/your-repo/ZeroForge.git
cd ZeroForge

# 同步依赖（自动创建虚拟环境）python3.14
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

### 后端部分debug game截图
![debug流程](assets/debug.png)
![debug mcts访问分布流程](assets/debug_mcts.png)
![debug mcts访问分布流程](assets/cchess.png)

### 命令行训练

```bash
# 训练井字棋（简单游戏，快速验证，默认配置）
uv run python main.py train

# 训练井字棋（指定参数）
uv run python main.py train --game tictactoe --algorithm alphazero --epochs 100

# 训练中国象棋（复杂游戏，使用 MuZero）
uv run python main.py train --game chinese_chess --algorithm muzero --network-size large --epochs 1000
```

## 项目结构

```
ZeroForge/
├── core/                       # 核心框架
│   ├── game.py                 # 游戏抽象基类 (Game, GameState, GameMeta)
│   ├── algorithm.py            # 算法抽象基类 (Algorithm, Trajectory)
│   ├── config.py               # 配置类 (MCTSConfig, BatcherConfig)
│   ├── training_config.py      # 训练配置 (TrainingConfig)
│   ├── mcts/                   # MCTS 实现
│   │   ├── node.py             # 树节点 (MCTSNode)
│   │   ├── tree.py             # 本地树 (LocalMCTSTree)
│   │   ├── search.py           # 搜索控制器 (MCTSSearch)
│   │   └── batcher.py          # GPU 批推理 (LeafBatcher)
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

## 配置说明

训练配置 `TrainingConfig`：

```python
from core.training_config import TrainingConfig

config = TrainingConfig(
    # 游戏和算法
    game_type="tictactoe",      # 游戏类型
    algorithm="alphazero",      # 算法 (alphazero/muzero)
    
    # 网络
    network_size="auto",        # auto/small/medium/large
    
    # 训练
    num_epochs=100,             # 训练轮数
    batch_size=128,             # 批大小
    lr=1e-3,                    # 学习率
    
    # MCTS
    num_simulations=50,         # 每步模拟次数
    c_puct=1.5,                 # UCB 探索常数
    
    # 自玩
    games_per_actor=50,         # 每轮自玩局数
    
    # 系统
    device="auto",              # auto/cuda/mps/cpu
)
```

网络大小自动选择：
- `small`: 井字棋等小游戏（MLP，~1K 参数）
- `medium`: 中等规模游戏（ConvNeXt 3层）
- `large`: 中国象棋等复杂游戏（ConvNeXt 6层，~100K+ 参数）

## 技术栈

- **深度学习**: PyTorch 2.x
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
- [ ] 分布式训练
- [ ] 更多游戏（围棋、国际象棋）

## License

MIT License

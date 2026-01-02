# ZeroForge

ZeroForge 是一个面向 MuZero、AlphaZero 及其演进算法的现代化训练框架。它以**流程化训练、Web 配置、可视化与深度调试**为核心，将自博弈强化学习从零散脚本提升为可观测、可组合、可演化的系统工程。

## 特性

- 🎮 **游戏插件化** - 通过简单的接口定义，快速接入新游戏
- 🧠 **算法模块化** - 支持 AlphaZero、MuZero（开发中）等算法
- 🌐 **Web API** - FastAPI 后端，支持训练控制、状态监控
- 📊 **实验管理** - 对比不同配置的训练效果
- 🔧 **配置驱动** - Pydantic 验证，类型安全的配置管理

## 快速开始

### 安装

```bash
# 克隆仓库
git clone https://github.com/zeroforge/zeroforge.git
cd zeroforge

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/macOS
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 或使用 pip 安装（开发模式）
pip install -e ".[all]"
```

### 训练井字棋 AI

```bash
# 运行示例训练脚本
python examples/train_tictactoe.py
```

输出示例：
```
============================================================
ZeroForge - 井字棋训练示例
============================================================
配置:
  - 迭代次数: 20
  - 每迭代自博弈: 50 局
  - 每迭代训练: 50 步
  - MCTS 模拟: 50 次/步
============================================================

游戏: tictactoe
  - 动作空间: 9
  - 观测形状: (3, 3, 3)

网络参数量: 25,481

==================== 迭代 1/20 ====================
[自博弈] 进行 50 局...
  玩家0胜: 24, 玩家1胜: 18, 平局: 8
  ...
```

### 启动 Web API

```bash
# 安装 web 依赖
pip install fastapi uvicorn

# 启动服务
uvicorn zeroforge.web.app:app --reload --port 8000
```

访问 http://localhost:8000/docs 查看 API 文档。

## 项目结构

```
zeroforge/
├── core/                       # 核心抽象层
│   ├── game.py                 # 游戏基类 (GameState, Game)
│   ├── network.py              # 神经网络基类
│   ├── algorithm.py            # 算法基类
│   └── config.py               # 配置管理 (Pydantic)
│
├── games/                      # 游戏插件目录
│   ├── registry.py             # 游戏注册表
│   └── tictactoe/              # 井字棋实现
│       ├── game.py             # 游戏逻辑
│       └── network.py          # 专用网络结构
│
├── algorithms/                 # 算法实现
│   ├── mcts.py                 # MCTS 核心
│   └── alphazero/              # AlphaZero
│       ├── self_play.py        # 自博弈
│       ├── trainer.py          # 训练器
│       └── evaluator.py        # 评估器
│
├── storage/                    # 数据存储
│   ├── replay_buffer.py        # 经验回放
│   └── checkpoint.py           # 模型存档
│
├── web/                        # Web 后端 (FastAPI)
│   ├── app.py                  # FastAPI 入口
│   └── api/                    # REST API
│
└── utils/                      # 工具类
    └── logger.py               # 日志系统
```

## 添加新游戏

1. 在 `zeroforge/games/` 下创建新目录
2. 实现 `GameState` 和 `Game` 子类
3. 使用 `@register_game` 装饰器注册

```python
from zeroforge.core.game import Game, GameState
from zeroforge.games.registry import register_game

class MyGameState(GameState):
    def get_legal_actions(self) -> list[int]:
        ...
    
    def apply_action(self, action: int) -> "MyGameState":
        ...
    
    # ... 其他必要方法

@register_game("mygame")
class MyGame(Game):
    @property
    def name(self) -> str:
        return "mygame"
    
    @property
    def action_size(self) -> int:
        return 100  # 你的动作空间大小
    
    # ... 其他必要方法
```

## API 端点

| 端点 | 方法 | 描述 |
|------|------|------|
| `/api/training/start` | POST | 启动训练 |
| `/api/training/stop` | POST | 停止训练 |
| `/api/training/status` | GET | 获取训练状态 |
| `/api/games/` | GET | 列出所有游戏 |
| `/api/games/{name}` | GET | 获取游戏详情 |
| `/api/experiments/` | GET/POST | 实验管理 |

## 配置示例

```python
from zeroforge.core.config import TrainingConfig, MCTSConfig

config = TrainingConfig(
    game_name="tictactoe",
    algorithm="alphazero",
    device="cuda",  # 或 "cpu"
    num_iterations=100,
    games_per_iteration=100,
    training_steps_per_iteration=100,
    batch_size=256,
    mcts=MCTSConfig(
        num_simulations=800,
        c_puct=1.5,
    ),
)
```

## 开发

```bash
# 安装开发依赖
pip install -e ".[dev]"

# 运行测试
pytest

# 代码检查
ruff check zeroforge/

# 类型检查
mypy zeroforge/
```

## 技术栈

- **深度学习**: PyTorch 2.x
- **Web 框架**: FastAPI
- **配置管理**: Pydantic v2
- **日志**: loguru

## 路线图

- [x] 核心框架设计
- [x] AlphaZero 实现
- [x] 井字棋游戏
- [x] Web API
- [ ] MuZero 实现
- [ ] 中国象棋游戏
- [ ] 分布式训练
- [ ] Web 可视化界面

## License

MIT License

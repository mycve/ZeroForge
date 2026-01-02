# ZeroForge API 接口说明

> 本文档供 AI 学习理解项目结构与接口设计，同时也作为开发者参考手册。

## 目录

- [核心概念](#核心概念)
- [游戏模块 API](#游戏模块-api) ⭐ 新增
- [Gymnasium 游戏支持](#gymnasium-游戏支持) ⭐ 新增
- [游戏调试 API](#游戏调试-api) ⭐ 新增
- [训练器 API](#训练器-api)
- [算法 API](#算法-api)
- [环境 API](#环境-api)
- [MCTS API](#mcts-api)
- [网络模型 API](#网络模型-api)
- [经验回放 API](#经验回放-api)
- [Web API](#web-api)
- [CLI 命令](#cli-命令)

---

## 核心概念

### 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│                        ZeroForge 架构                            │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   Web UI    │  │    CLI      │  │  Python API │   用户接口   │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘              │
│         │                │                │                      │
│         └────────────────┼────────────────┘                      │
│                          ▼                                       │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    ZeroForgeTrainer                        │  │
│  │  - 统一训练入口                                            │  │
│  │  - 配置管理                                                │  │
│  │  - 生命周期控制                                            │  │
│  └───────────────────────────────────────────────────────────┘  │
│                          │                                       │
│         ┌────────────────┼────────────────┐                      │
│         ▼                ▼                ▼                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │  Algorithm  │  │ Environment │  │ReplayBuffer │   核心组件   │
│  │  (MuZero)   │  │  (Gym-like) │  │ (优先采样)  │              │
│  └──────┬──────┘  └─────────────┘  └─────────────┘              │
│         │                                                        │
│         ▼                                                        │
│  ┌─────────────┐  ┌─────────────┐                               │
│  │    MCTS     │◄─│   Network   │   搜索与推理                  │
│  │ (树搜索)    │  │ (神经网络)  │                               │
│  └─────────────┘  └─────────────┘                               │
└─────────────────────────────────────────────────────────────────┘
```

### 核心数据流（周期模式）

```
┌─────────────────────────────────────────────────────────────────────┐
│                          每个 Epoch                                  │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                   Phase 1: 自玩阶段                           │   │
│  │                                                               │   │
│  │  ┌────────┐    action    ┌────────┐   trajectory  ┌────────┐ │   │
│  │  │  MCTS  │─────────────►│  Env   │──────────────►│ 新数据 │ │   │
│  │  │ Search │              │  Step  │               │  缓冲  │ │   │
│  │  └───▲────┘              └────────┘               └───┬────┘ │   │
│  │      │ policy, value                                  │      │   │
│  │  ┌───┴────┐                                           │      │   │
│  │  │ Leaf   │  ← 批量 GPU 推理                          │      │   │
│  │  │Batcher │                                           │      │   │
│  │  └────────┘                                           │      │   │
│  └───────────────────────────────────────────────────────│──────┘   │
│                                                          ↓          │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                   Phase 2: 训练阶段                           │   │
│  │                                                               │   │
│  │  ┌────────────┐   80%    ┌──────────┐                        │   │
│  │  │   新数据   │─────────►│          │                        │   │
│  │  └────────────┘          │  混合    │    ┌──────────┐        │   │
│  │  ┌────────────┐   20%    │  采样    │───►│ 训练循环 │        │   │
│  │  │  经验池    │─────────►│          │    └──────────┘        │   │
│  │  └────────────┘          └──────────┘                        │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │                   Phase 3: 评估阶段                           │   │
│  │                                                               │   │
│  │  ┌────────────┐                      ┌─────────────┐         │   │
│  │  │ 新版本模型 │  vs  旧版本模型  ───►│ 胜率 + ELO  │         │   │
│  │  └────────────┘                      └─────────────┘         │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 游戏模块 API

### 游戏目录结构

每个游戏作为独立模块，放置在 `games/<game_name>/` 目录下：

```
games/
├── __init__.py              # 注册系统 + 统一入口
│
└── chinese_chess/           # 中国象棋（示例）
    ├── __init__.py          # 模块入口，导出 + 自动注册
    ├── game.py              # Game 接口实现
    ├── config.py            # 游戏配置（常量、参数）
    │
    └── cchess/              # 游戏引擎（依赖）
        ├── __init__.py      # 棋盘、走子、规则逻辑
        ├── engine.py        # 引擎常量（如有需要）
        └── svg.py           # SVG 渲染
```

### 游戏渲染格式规范

游戏的 `render(mode="json")` 方法返回的数据用于 Web 前端展示。对于网格类游戏（棋盘游戏），必须遵循 `GridRenderData` 格式：

```python
# GridRenderData 格式
{
    "type": "grid",                    # 必须为 "grid"
    "rows": int,                       # 行数
    "cols": int,                       # 列数
    "cells": list[list[any]],          # 二维数组，每个元素是格子内容（None/字符串/数字）
    "cell_colors": list[list[str]],    # 可选，格子颜色
    "highlights": list[tuple[int, int]],  # ⚠️ 高亮格子，格式为 [(row, col), ...]
    "labels": {                        # 可选，坐标标签
        "row": list[str],
        "col": list[str],
    },
    # ... 其他游戏特定字段
}
```

#### ⚠️ 重要注意事项

**`highlights` 字段格式要求**：

```python
# ✅ 正确格式 - 元组列表
highlights = [(row, col), ...]
# 例如: [(4, 4), (5, 3)]

# ❌ 错误格式 - 对象列表（会导致前端报错！）
highlights = [{"row": row, "col": col}, ...]
```

前端使用解构语法 `([r, c])` 读取坐标，因此 `highlights` **必须**是可迭代的二元组列表。Python 元组在 JSON 序列化后会变成数组 `[[row, col], ...]`，这是前端期望的格式。

**完整示例**：

```python
def render(self, mode: str = "json") -> dict:
    if mode == "json":
        # 构建高亮列表 - 使用元组格式
        highlights: List[Tuple[int, int]] = []
        if self.last_move is not None:
            row, col = self._action_to_position(self.last_move)
            highlights.append((row, col))  # ✅ 元组
        
        return {
            "type": "grid",
            "rows": self.board_size,
            "cols": self.board_size,
            "cells": cells,
            "highlights": highlights,  # JSON 序列化后变成 [[row, col], ...]
            "labels": {
                "row": [str(i) for i in range(self.board_size)],
                "col": [str(i) for i in range(self.board_size)],
            },
        }
```

---

### 添加新游戏指南

1. **创建游戏目录**:

```bash
mkdir -p games/my_game
```

2. **实现 Game 接口** (`games/my_game/game.py`):

```python
from core.game import Game, ObservationSpace, ActionSpace
from games import register_game

@register_game("my_game")
class MyGame(Game):
    """自定义游戏实现"""
    
    @property
    def observation_space(self) -> ObservationSpace:
        return ObservationSpace(shape=(C, H, W), dtype=np.float32)
    
    @property
    def action_space(self) -> ActionSpace:
        return ActionSpace(n=NUM_ACTIONS)
    
    @property
    def num_players(self) -> int:
        return 2  # 双人对弈
    
    def reset(self) -> np.ndarray:
        """重置游戏"""
        ...
    
    def step(self, action: int) -> tuple[np.ndarray, float, bool, dict]:
        """执行动作"""
        ...
    
    def legal_actions(self) -> list[int]:
        """获取合法动作"""
        ...
    
    def clone(self) -> "MyGame":
        """深拷贝（用于 MCTS）"""
        ...
```

3. **创建配置文件** (`games/my_game/config.py`):

```python
from dataclasses import dataclass

@dataclass
class MyGameConfig:
    board_size: int = 8
    max_game_length: int = 100
    history_steps: int = 4
```

4. **在模块入口注册** (`games/my_game/__init__.py`):

```python
from .game import MyGame
from .config import MyGameConfig

__all__ = ["MyGame", "MyGameConfig"]
```

5. **在 games/__init__.py 中导入**:

```python
from .my_game import MyGame
```

### 游戏注册系统

```python
from games import register_game, make_game, list_games, get_game_info

# 注册装饰器
@register_game("my_game")
class MyGame(Game):
    ...

# 创建游戏实例
game = make_game("chinese_chess")
game = make_game("chinese_chess", history_steps=8)

# 列出所有游戏
print(list_games())  # ['chinese_chess', 'my_game', ...]

# 获取游戏信息
info = get_game_info("chinese_chess")
print(info)
# {
#     'name': 'chinese_chess',
#     'class': 'ChineseChessGame',
#     'observation_space': {'shape': (58, 10, 9), 'dtype': 'float32'},
#     'action_space': {'n': 2086},
#     'num_players': 2,
#     'player_type': 'TWO_PLAYER'
# }
```

---

## Gymnasium 游戏支持

ZeroForge 支持所有 Gymnasium 兼容环境，包括 Atari、经典控制、MuJoCo 等。

### 依赖安装

```bash
# 基础 Gymnasium（经典控制游戏）
pip install gymnasium

# Atari 游戏
pip install gymnasium[atari] ale-py

# MuJoCo 物理仿真
pip install gymnasium[mujoco]

# 一次性安装所有
pip install gymnasium[atari,mujoco] ale-py
```

### 使用方式

#### 方式1: 预设游戏（推荐）

```python
from games import make_game, list_games

# 查看所有可用游戏
print(list_games())
# ['chinese_chess', 'tictactoe', 'gomoku_9x9', 'gomoku_15x15',
#  'atari_breakout', 'atari_pong', 'gym_cartpole', 'gym_lunarlander', ...]

# 创建 Atari 游戏
game = make_game("atari_breakout")
obs = game.reset()
print(obs.shape)  # (4, 84, 84) - 4帧灰度图

# 创建经典控制游戏
game = make_game("gym_cartpole")
obs = game.reset()
action = game.legal_actions()[0]
obs, reward, done, info = game.step(action)
```

#### 方式2: 自定义配置

```python
from games.gymnasium import GymnasiumWrapper, AtariConfig

# 使用配置类
config = AtariConfig(
    env_id="ALE/Breakout-v5",
    frame_stack=4,        # 帧堆叠
    frame_skip=4,         # 跳帧
    grayscale=True,       # 灰度化
    resize=(84, 84),      # 缩放
    clip_rewards=True,    # 奖励裁剪
)
game = GymnasiumWrapper(config=config)

# 或直接指定环境 ID
game = GymnasiumWrapper("CartPole-v1")
game = GymnasiumWrapper("LunarLander-v3")
```

#### 方式3: 任意 Gymnasium 环境

```python
import gymnasium as gym
from games.gymnasium import GymnasiumWrapper

# 包装任意 Gymnasium 环境
env = gym.make("MyCustomEnv-v0")
game = GymnasiumWrapper(env=env)
```

### 预设游戏列表

#### Atari 游戏

| 游戏名称 | 环境 ID | 描述 |
|---------|---------|------|
| `atari_breakout` | ALE/Breakout-v5 | 打砖块 |
| `atari_pong` | ALE/Pong-v5 | 乒乓球 |
| `atari_spaceinvaders` | ALE/SpaceInvaders-v5 | 太空侵略者 |
| `atari_qbert` | ALE/Qbert-v5 | Q*bert |
| `atari_mspacman` | ALE/MsPacman-v5 | 吃豆人 |
| `atari_asteroids` | ALE/Asteroids-v5 | 小行星 |
| `atari_seaquest` | ALE/Seaquest-v5 | 深海探险 |
| `atari_beamrider` | ALE/BeamRider-v5 | 光束骑士 |
| `atari_enduro` | ALE/Enduro-v5 | 耐力赛车 |
| `atari_freeway` | ALE/Freeway-v5 | 穿越高速公路 |

#### 经典控制游戏

| 游戏名称 | 环境 ID | 描述 |
|---------|---------|------|
| `gym_cartpole` | CartPole-v1 | 平衡倒立摆 |
| `gym_lunarlander` | LunarLander-v3 | 月球着陆器 |
| `gym_mountaincar` | MountainCar-v0 | 爬山车 |
| `gym_acrobot` | Acrobot-v1 | 双摆机器人 |
| `gym_pendulum` | Pendulum-v1 | 倒立摆（连续控制） |

#### MuJoCo 物理仿真

| 游戏名称 | 环境 ID | 描述 |
|---------|---------|------|
| `mujoco_ant` | Ant-v5 | 四足蚂蚁 |
| `mujoco_halfcheetah` | HalfCheetah-v5 | 猎豹 |
| `mujoco_hopper` | Hopper-v5 | 单腿跳跃 |
| `mujoco_walker2d` | Walker2d-v5 | 双足行走 |
| `mujoco_humanoid` | Humanoid-v5 | 人形机器人 |
| `mujoco_swimmer` | Swimmer-v5 | 游泳者 |

### 算法选择

Gymnasium 环境的特殊性在于部分环境不支持深拷贝（clone），这会影响传统 MCTS 的使用。

```python
game = make_game("atari_breakout")

# 检查是否支持传统 MCTS
print(game.supports_mcts)  # True/False

# 获取推荐算法
print(game.recommended_algorithm)  # "alphazero" 或 "gumbel_muzero"
```

#### 算法对比

| 算法 | 需要 clone | 适用场景 |
|------|-----------|---------|
| AlphaZero | 是 | 棋类游戏、支持 clone 的环境 |
| MCTS | 是 | 同上 |
| **Gumbel MuZero** | **否** | 所有环境，特别是不支持 clone 的 |
| **Gumbel AlphaZero** | **否** | 同上 |
| **Gumbel EfficientZero** | **否** | 同上，更高效的样本利用 |

Gumbel 系列算法使用 Gumbel-Top-k 技巧进行策略改进，不需要完整的 MCTS 树搜索，因此可以在不支持 clone 的环境中使用。

### 配置参数

```python
from games.gymnasium import GymnasiumConfig

config = GymnasiumConfig(
    # 环境
    env_id="CartPole-v1",       # Gymnasium 环境 ID
    
    # 观测预处理
    frame_stack=4,              # 帧堆叠数量
    frame_skip=4,               # 跳帧数量
    grayscale=True,             # 灰度化（图像环境）
    resize=(84, 84),            # 缩放大小
    normalize_obs=True,         # 归一化到 [0, 1]
    
    # 奖励处理
    clip_rewards=True,          # 裁剪奖励到 [-1, 1]
    reward_scale=1.0,           # 奖励缩放
    
    # Atari 特定
    terminal_on_life_loss=True, # 生命丢失视为终止
    noop_max=30,                # 开始时随机 NOOP
    
    # 连续动作空间
    discrete_bins=11,           # 离散化 bins 数量
)
```

### 连续动作空间处理

对于 MuJoCo 等连续控制环境，ZeroForge 自动将连续动作空间离散化：

```python
# 原始: Box(-1, 1, shape=(3,))
# 离散化: Discrete(11^3 = 1331)

game = make_game("mujoco_ant")
print(game.action_space.n)  # 1331 (11^3, 3维动作空间)
```

每个维度离散化为 `discrete_bins` 个动作（默认 11），映射为均匀分布的值。

---

## 游戏调试 API

### 调试数据结构

Web 调试界面可以观察游戏整个生命周期的数据：

```python
@dataclass
class GameDebugState:
    """游戏调试状态快照"""
    
    # 基础状态
    step_index: int              # 当前步数
    current_player: int          # 当前玩家
    is_terminal: bool            # 是否结束
    winner: int | None           # 获胜者
    
    # 动作信息
    last_action: int | None      # 上一步动作
    legal_actions: list[int]     # 当前合法动作
    
    # 观测数据
    observation: np.ndarray      # 观测张量
    raw_state: dict              # 原始状态（游戏特定）
    
    # MCTS 数据（如启用）
    mcts_policy: np.ndarray | None    # MCTS 策略分布
    mcts_value: float | None          # MCTS 根节点价值
    mcts_visit_counts: dict | None    # 各动作访问次数
    
    # 网络输出（如启用）
    network_policy: np.ndarray | None # 网络原始策略
    network_value: float | None       # 网络原始价值
    
    # 时间戳
    timestamp: float             # 状态时间戳


@dataclass
class GameDebugHistory:
    """游戏调试历史"""
    
    game_id: str                 # 游戏 ID
    game_type: str               # 游戏类型
    states: list[GameDebugState] # 状态历史
    config: dict                 # 游戏配置
    
    def to_json(self) -> dict:
        """序列化为 JSON"""
        ...
    
    @classmethod
    def from_json(cls, data: dict) -> "GameDebugHistory":
        """从 JSON 反序列化"""
        ...
```

### REST API 调试端点

| 方法 | 端点 | 描述 |
|------|------|------|
| POST | `/api/debug/game/create` | 创建调试游戏会话 |
| GET | `/api/debug/game/{id}/state` | 获取当前状态 |
| POST | `/api/debug/game/{id}/step` | 执行一步 |
| POST | `/api/debug/game/{id}/reset` | 重置游戏 |
| GET | `/api/debug/game/{id}/history` | 获取完整历史 |
| POST | `/api/debug/game/{id}/goto/{step}` | 跳转到指定步 |
| GET | `/api/debug/game/{id}/observation` | 获取观测数据 |
| GET | `/api/debug/game/{id}/legal_actions` | 获取合法动作 |
| POST | `/api/debug/game/{id}/mcts` | 执行 MCTS 搜索 |

### 调试端点示例

```python
# 创建调试会话
POST /api/debug/game/create
Content-Type: application/json

{
    "game_type": "chinese_chess",
    "config": {
        "history_steps": 4,
        "enable_mcts_log": true,
        "enable_network_log": true
    }
}

# 响应
{
    "game_id": "debug_20260102_143052",
    "game_type": "chinese_chess",
    "state": {
        "step_index": 0,
        "current_player": 0,
        "legal_actions_count": 44,
        "is_terminal": false
    }
}
```

```python
# 获取详细状态
GET /api/debug/game/debug_20260102_143052/state?include_observation=true

# 响应
{
    "step_index": 5,
    "current_player": 1,
    "is_terminal": false,
    "winner": null,
    "last_action": 1234,
    "legal_actions": [100, 101, 102, ...],
    "observation": {
        "shape": [58, 10, 9],
        "dtype": "float32",
        "data_base64": "..."  // Base64 编码的观测数据
    },
    "raw_state": {
        "fen": "rnbakabnr/9/1c5c1/...",
        "move_count": 5
    }
}
```

```python
# 执行一步并获取调试信息
POST /api/debug/game/debug_20260102_143052/step
Content-Type: application/json

{
    "action": 1234,
    "run_mcts": true,
    "mcts_simulations": 50
}

# 响应
{
    "step_index": 6,
    "reward": 0.0,
    "done": false,
    "mcts_result": {
        "policy": [0.01, 0.02, ...],
        "value": 0.15,
        "visit_counts": {"100": 12, "101": 8, ...},
        "selected_action": 100
    }
}
```

### WebSocket 实时调试

```python
# 连接调试 WebSocket
ws://localhost:8000/ws/debug/game/{game_id}

# 客户端发送命令
{
    "type": "step",
    "action": 1234
}

{
    "type": "reset"
}

{
    "type": "mcts",
    "simulations": 100
}

# 服务端推送状态
{
    "type": "state_update",
    "data": {
        "step_index": 6,
        "current_player": 1,
        "observation_preview": [...],  // 简化预览
        "legal_actions": [100, 101, ...]
    }
}
```

### 中国象棋调试扩展

中国象棋游戏提供额外的调试方法：

```python
class ChineseChessGame(Game):
    """中国象棋游戏"""
    
    def get_debug_info(self) -> dict:
        """获取调试信息（特定于中国象棋）"""
        return {
            "fen": self.board.fen(),
            "move_count": self.move_count,
            "is_check": self.board.is_check(),
            "last_move_uci": self._last_move_uci(),
            "piece_counts": self._get_piece_counts(),
            "svg": self.render(mode="svg"),
        }
    
    def render(self, mode: str = "human") -> Any:
        """
        渲染棋盘
        
        Args:
            mode: 
                - "human": 打印到控制台
                - "svg": 返回 SVG 字符串
                - "json": 返回 JSON 状态
        """
        ...
```

### 调试配置

```python
from games.chinese_chess.config import DebugConfig

debug_config = DebugConfig(
    log_steps=True,           # 记录每步详细信息
    log_mcts_tree=True,       # 记录 MCTS 搜索树
    log_network_output=True,  # 记录网络输出
    debug_dir="debug_logs",   # 调试数据保存路径
    max_history=1000,         # 最大保存历史步数
)
```

---

## 训练器 API

### `ZeroForgeTrainer`

统一的训练入口类，管理整个训练生命周期。

```python
class ZeroForgeTrainer:
    """
    ZeroForge 核心训练器
    
    负责协调算法、环境、回放缓冲区等组件，
    执行完整的自博弈强化学习训练流程。
    """
    
    def __init__(
        self,
        algorithm: BaseAlgorithm,
        env: BaseEnv,
        config: dict | TrainerConfig | None = None,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
    ) -> None:
        """
        初始化训练器
        
        Args:
            algorithm: 算法实例 (MuZero, AlphaZero 等)
            env: 环境实例，需符合 Gymnasium 接口
            config: 训练配置，支持字典或 TrainerConfig 对象
            checkpoint_dir: 模型检查点保存目录
            log_dir: 日志输出目录
            
        Raises:
            ValueError: 配置参数无效时抛出
            RuntimeError: 算法与环境不兼容时抛出
        """
        ...
    
    def train(
        self,
        num_iterations: int = 1000,
        save_interval: int = 100,
        eval_interval: int = 50,
        callbacks: list[Callback] | None = None,
    ) -> TrainingResult:
        """
        执行训练循环
        
        Args:
            num_iterations: 总训练迭代次数
            save_interval: 检查点保存间隔
            eval_interval: 评估间隔
            callbacks: 回调函数列表
            
        Returns:
            TrainingResult: 包含训练统计信息的结果对象
            
        Raises:
            TrainingInterrupted: 训练被中断时抛出
            ResourceExhausted: GPU 内存不足等资源问题
        """
        ...
    
    def evaluate(
        self,
        num_episodes: int = 100,
        render: bool = False,
    ) -> EvalResult:
        """
        评估当前模型
        
        Args:
            num_episodes: 评估的回合数
            render: 是否渲染环境
            
        Returns:
            EvalResult: 评估结果，包含平均奖励、胜率等
        """
        ...
    
    def save(self, path: str) -> None:
        """保存完整训练状态（模型 + 优化器 + 回放缓冲区）"""
        ...
    
    def load(self, path: str) -> None:
        """加载训练状态"""
        ...
    
    @property
    def metrics(self) -> dict[str, float]:
        """获取当前训练指标"""
        ...
```

### `TrainerConfig`

训练配置数据类，使用 Pydantic 进行验证。

```python
from pydantic import BaseModel, Field

class TrainerConfig(BaseModel):
    """训练器配置"""
    
    # 基础训练参数
    batch_size: int = Field(default=256, ge=1, description="训练批次大小")
    learning_rate: float = Field(default=1e-3, gt=0, description="学习率")
    weight_decay: float = Field(default=1e-4, ge=0, description="权重衰减")
    
    # Self-Play 参数
    num_actors: int = Field(default=4, ge=1, description="并行 Actor 数量")
    games_per_iteration: int = Field(default=100, ge=1, description="每轮迭代的对局数")
    
    # MCTS 参数
    num_simulations: int = Field(default=50, ge=1, description="MCTS 模拟次数")
    temperature: float = Field(default=1.0, ge=0, description="动作选择温度")
    
    # 回放缓冲区参数
    replay_buffer_size: int = Field(default=100000, ge=1000, description="回放缓冲区大小")
    priority_alpha: float = Field(default=0.6, ge=0, le=1, description="优先级指数")
    
    # 硬件配置
    device: str = Field(default="cuda", description="训练设备 (cuda/cpu)")
    num_gpus: int = Field(default=1, ge=0, description="使用的 GPU 数量")
    
    class Config:
        extra = "forbid"  # 禁止未知字段
```

---

## 算法 API

### `BaseAlgorithm`

所有算法的抽象基类。

```python
from abc import ABC, abstractmethod
from typing import Any
import torch

class BaseAlgorithm(ABC):
    """
    算法基类
    
    定义了 ZeroForge 支持的算法必须实现的接口。
    所有算法（MuZero, AlphaZero 等）都继承此类。
    """
    
    @abstractmethod
    def select_action(
        self,
        state: torch.Tensor,
        legal_actions: list[int] | None = None,
        temperature: float = 1.0,
    ) -> tuple[int, dict[str, Any]]:
        """
        根据当前状态选择动作
        
        Args:
            state: 当前状态张量
            legal_actions: 合法动作列表，None 表示所有动作合法
            temperature: 探索温度，0 表示贪婪选择
            
        Returns:
            action: 选择的动作索引
            info: 额外信息（MCTS 统计、策略分布等）
        """
        ...
    
    @abstractmethod
    def train_step(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        """
        执行一步训练
        
        Args:
            batch: 训练批次，包含:
                - observations: 观测序列
                - actions: 动作序列
                - rewards: 奖励序列
                - target_policies: 目标策略
                - target_values: 目标价值
                
        Returns:
            losses: 各项损失值字典
                - total_loss: 总损失
                - policy_loss: 策略损失
                - value_loss: 价值损失
                - reward_loss: 奖励预测损失 (MuZero)
        """
        ...
    
    @abstractmethod
    def get_network(self) -> torch.nn.Module:
        """获取底层神经网络模型"""
        ...
    
    @property
    @abstractmethod
    def name(self) -> str:
        """算法名称"""
        ...
```

### `MuZero`

MuZero 算法实现。

```python
class MuZero(BaseAlgorithm):
    """
    MuZero 算法实现
    
    MuZero 通过学习环境动态模型来进行规划，
    无需访问真实的游戏规则或模拟器。
    
    核心组件:
        - Representation: h(o) -> s  观测编码为隐状态
        - Dynamics: g(s, a) -> r, s' 状态转移与奖励预测
        - Prediction: f(s) -> p, v   策略与价值预测
    """
    
    def __init__(
        self,
        observation_shape: tuple[int, ...],
        action_space_size: int,
        hidden_state_size: int = 256,
        num_residual_blocks: int = 16,
        num_simulations: int = 50,
        discount: float = 0.997,
        support_size: int = 300,  # 价值支持集大小
    ) -> None:
        """
        初始化 MuZero
        
        Args:
            observation_shape: 观测空间形状
            action_space_size: 动作空间大小
            hidden_state_size: 隐状态维度
            num_residual_blocks: 残差块数量
            num_simulations: MCTS 模拟次数
            discount: 折扣因子
            support_size: 分类价值的支持集大小
        """
        ...
    
    def initial_inference(
        self,
        observation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        初始推理（根节点）
        
        Args:
            observation: 原始观测
            
        Returns:
            hidden_state: 编码后的隐状态
            policy_logits: 策略 logits
            value: 预测价值
        """
        ...
    
    def recurrent_inference(
        self,
        hidden_state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        递归推理（扩展节点）
        
        Args:
            hidden_state: 当前隐状态
            action: 执行的动作
            
        Returns:
            next_hidden_state: 下一隐状态
            reward: 预测奖励
            policy_logits: 策略 logits
            value: 预测价值
        """
        ...
```

### `AlphaZero`

AlphaZero 算法实现。

```python
class AlphaZero(BaseAlgorithm):
    """
    AlphaZero 算法实现
    
    AlphaZero 使用已知的游戏规则进行 MCTS 搜索，
    神经网络仅预测策略和价值。
    
    与 MuZero 的区别:
        - 需要环境提供完美模拟器（step 函数）
        - 无需学习动态模型
        - 训练更简单，但泛化性较弱
    """
    
    def __init__(
        self,
        observation_shape: tuple[int, ...],
        action_space_size: int,
        num_residual_blocks: int = 19,
        num_filters: int = 256,
        num_simulations: int = 800,
    ) -> None:
        """
        初始化 AlphaZero
        
        Args:
            observation_shape: 观测空间形状
            action_space_size: 动作空间大小
            num_residual_blocks: 残差块数量
            num_filters: 卷积滤波器数量
            num_simulations: MCTS 模拟次数
        """
        ...
```

---

## 环境 API

### `BaseEnv`

环境抽象基类，兼容 Gymnasium 接口。

```python
from abc import ABC, abstractmethod
from typing import Any
import numpy as np

class BaseEnv(ABC):
    """
    环境基类
    
    遵循 Gymnasium 接口规范，同时添加 ZeroForge 所需的额外方法。
    """
    
    @abstractmethod
    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        重置环境
        
        Args:
            seed: 随机种子
            options: 额外选项
            
        Returns:
            observation: 初始观测
            info: 环境信息
        """
        ...
    
    @abstractmethod
    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        执行动作
        
        Args:
            action: 动作索引
            
        Returns:
            observation: 新观测
            reward: 即时奖励
            terminated: 是否终止（游戏结束）
            truncated: 是否截断（超时等）
            info: 额外信息
        """
        ...
    
    @abstractmethod
    def get_legal_actions(self) -> list[int]:
        """
        获取当前状态下的合法动作
        
        Returns:
            合法动作索引列表
        """
        ...
    
    @abstractmethod
    def clone(self) -> "BaseEnv":
        """
        深拷贝环境（用于 MCTS 模拟）
        
        Returns:
            环境的深拷贝
        """
        ...
    
    @property
    @abstractmethod
    def observation_shape(self) -> tuple[int, ...]:
        """观测空间形状"""
        ...
    
    @property
    @abstractmethod
    def action_space_size(self) -> int:
        """动作空间大小"""
        ...
    
    @property
    def current_player(self) -> int:
        """当前玩家（多人游戏），默认返回 0"""
        return 0
```

### 内置环境

```python
# 经典控制
from zeroforge.envs import CartPoleEnv, MountainCarEnv

# 棋类游戏
from zeroforge.envs import TicTacToeEnv, Connect4Env, GomokuEnv

# Atari 游戏（需要 ale-py）
from zeroforge.envs import AtariEnv

# 自定义环境包装
from zeroforge.envs import GymWrapper

# 使用示例
env = GymWrapper("LunarLander-v2")
```

---

## MCTS API

### `MCTS`

蒙特卡洛树搜索实现。

```python
class MCTS:
    """
    蒙特卡洛树搜索
    
    支持 MuZero 和 AlphaZero 两种模式:
        - MuZero 模式: 使用学习的动态模型进行模拟
        - AlphaZero 模式: 使用真实环境进行模拟
    """
    
    def __init__(
        self,
        algorithm: BaseAlgorithm,
        num_simulations: int = 50,
        c_puct: float = 1.25,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.25,
        discount: float = 0.997,
    ) -> None:
        """
        初始化 MCTS
        
        Args:
            algorithm: 算法实例
            num_simulations: 模拟次数
            c_puct: PUCT 探索常数
            dirichlet_alpha: Dirichlet 噪声 alpha 参数
            dirichlet_epsilon: Dirichlet 噪声混合比例
            discount: 折扣因子
        """
        ...
    
    def search(
        self,
        root_state: torch.Tensor | BaseEnv,
        legal_actions: list[int] | None = None,
        add_noise: bool = True,
    ) -> MCTSResult:
        """
        执行 MCTS 搜索
        
        Args:
            root_state: 根状态（张量或环境实例）
            legal_actions: 合法动作列表
            add_noise: 是否在根节点添加探索噪声
            
        Returns:
            MCTSResult: 搜索结果
        """
        ...
    
    def get_action_policy(
        self,
        temperature: float = 1.0,
    ) -> np.ndarray:
        """
        根据访问次数获取动作概率分布
        
        Args:
            temperature: 温度参数，0 表示贪婪
            
        Returns:
            policy: 动作概率分布
        """
        ...


class MCTSResult:
    """MCTS 搜索结果"""
    
    root_value: float          # 根节点价值估计
    visit_counts: np.ndarray   # 各动作访问次数
    action_probs: np.ndarray   # 动作概率分布
    selected_action: int       # 选择的动作
    search_tree: MCTSNode | None  # 搜索树（调试用）


class MCTSNode:
    """MCTS 树节点"""
    
    visit_count: int           # 访问次数
    value_sum: float           # 累积价值
    prior: float               # 先验概率
    children: dict[int, "MCTSNode"]  # 子节点
    hidden_state: torch.Tensor | None  # 隐状态 (MuZero)
    
    @property
    def value(self) -> float:
        """平均价值 Q(s, a)"""
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0
```

---

## 周期训练架构 ⭐ 更新

### 架构概览

采用"自玩 → 训练 → 评估"的周期模式，清晰分离推理和训练阶段。

```
┌─────────────────────────────────────────────────────────────────────────┐
│                              每个 Epoch                                  │
├─────────────────────────────────────────────────────────────────────────┤
│  Phase 1: 自玩阶段                                                       │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │                    GPU InferenceBatcher                            │  │
│  │              收集叶子请求 → 批量推理 → 返回结果                     │  │
│  └───────────────────────────────────────────────────────────────────┘  │
│                              ↑↓                                          │
│        ┌─────────────────────┼─────────────────────┐                    │
│        ↓                     ↓                     ↓                    │
│  ┌───────────┐    ┌───────────┐    ┌───────────┐                        │
│  │  Game 1   │    │  Game 2   │    │  Game N   │  ← 最多 concurrency 个 │
│  │   MCTS    │    │   MCTS    │    │   MCTS    │    并发运行            │
│  └───────────┘    └───────────┘    └───────────┘                        │
│                                                                          │
│  完成 num_envs 局后 → 收集轨迹 → 释放推理资源                            │
├─────────────────────────────────────────────────────────────────────────┤
│  Phase 2: 训练阶段                                                       │
│  - 采样 80% 新数据 + 20% 经验池                                          │
│  - 执行 train_batches_per_epoch 批次训练                                 │
├─────────────────────────────────────────────────────────────────────────┤
│  Phase 3: 评估阶段（Epoch 2+ 可选）                                       │
│  - 新版本 vs 旧版本对弈 eval_games 局                                    │
│  - 计算胜率、更新 ELO 评分                                               │
└─────────────────────────────────────────────────────────────────────────┘
```

**核心特点**:
- **内存管理清晰**：训练和推理分离，避免同时占用大量显存
- **并发控制**：`num_envs` 控制总游戏数，`concurrency` 控制并发数
- **数据混合**：80% 新数据 + 20% 经验池，提高样本效率
- **自动评估**：新旧版本对弈，实时 ELO 追踪

### 配置类

```python
from core.config import MCTSConfig, BatcherConfig
from core.training_config import TrainingConfig

# MCTS 配置
mcts_config = MCTSConfig(
    num_simulations=50,     # 每步模拟次数
    c_puct=1.5,             # UCB 探索常数
    reuse_tree=True,        # 是否复用子树
    dirichlet_alpha=0.3,    # 探索噪声
)

# 批推理配置
batcher_config = BatcherConfig(
    batch_size=8,           # 批大小（≤ concurrency）
    timeout_ms=10.0,        # 超时触发
    device="cuda",
)

# 完整训练配置
config = TrainingConfig(
    # 自玩
    num_envs=256,           # 每 epoch 完成的游戏数
    concurrency=16,         # 并发游戏数
    new_data_ratio=0.8,     # 新数据占比
    
    # 评估
    eval_games=5,           # 评估对弈局数
    eval_temperature=0.5,   # 评估采样温度
)
```

### CPU 本地 MCTS 树

```python
from core.mcts import LocalMCTSTree, MCTSNode

# 创建树
tree = LocalMCTSTree(game, mcts_config, mode="alphazero")

# 搜索循环
for _ in range(num_simulations):
    # 选择叶子节点（返回 game clone）
    node, game_clone = tree.select()
    
    # 评估（通过 batcher 或直接网络调用）
    policy, value = evaluate(game_clone.get_observation())
    
    # 扩展节点
    tree.expand(node, game_clone, policy, value,
                legal_actions=game_clone.legal_actions())
    
    # 回传价值
    tree.backup(node, value)

# 选择动作
action = tree.get_action(temperature=1.0)
policy = tree.get_policy(temperature=1.0)

# 执行并复用子树（关键优化！）
game.step(action)
tree.advance(action)  # 复用子树
```

### MCTSNode 数据结构

```python
@dataclass
class MCTSNode:
    """CPU 端 MCTS 节点"""
    
    # 统计
    visit_count: int = 0
    value_sum: float = 0.0
    prior: float = 0.0
    
    # 树结构
    parent: Optional[MCTSNode] = None
    children: Dict[int, MCTSNode] = field(default_factory=dict)
    action: int = -1
    
    # AlphaZero: 存储 game clone
    game_state: Optional[Game] = None
    
    # MuZero: 存储 hidden state
    hidden_state: Optional[np.ndarray] = None
    reward: float = 0.0
    
    @property
    def q_value(self) -> float:
        """平均 Q 值"""
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0
    
    def reuse_subtree(self, action: int) -> Optional[MCTSNode]:
        """复用子树，返回新根节点"""
        if action in self.children:
            child = self.children[action]
            child.parent = None  # 断开父引用，让旧节点被 GC
            return child
        return None
```

### GPU 批推理器

```python
from core.mcts import LeafBatcher

# 创建 batcher
batcher = LeafBatcher(network, batcher_config)
batcher.start()  # 启动 GPU 线程

# 在 env 线程中使用（阻塞等待）
policy, value = batcher.submit(
    observation=game.get_observation(),
    legal_mask=game.get_legal_actions_mask(),
)

# 停止
batcher.stop()
```

**触发推理条件（满足任一）**:
1. 收集到 `batch_size` 个请求
2. 等待超过 `timeout_ms` 毫秒

### 周期训练使用

```python
from core.trainer import DistributedTrainer
from core.training_config import TrainingConfig

# 配置周期训练
config = TrainingConfig(
    game_type="chinese_chess",
    algorithm="alphazero",
    
    # 自玩配置
    num_envs=256,           # 每个 epoch 完成 256 局游戏
    concurrency=16,         # 同时并发 16 个游戏
    new_data_ratio=0.8,     # 80% 新数据 + 20% 经验池
    
    # 训练配置
    train_batches_per_epoch=10,
    batch_size=256,
    
    # 评估配置
    eval_games=5,           # 每 epoch 评估 5 局
    eval_temperature=0.5,
)

# 创建训练器
trainer = DistributedTrainer(config)
trainer.setup()

# 运行训练（周期模式自动执行：自玩 → 训练 → 评估）
trainer.run()

# 训练过程中可以获取实时状态
state = trainer.get_state()
print(f"Epoch: {state['epoch']}")
print(f"总游戏数: {state['total_games']}")
print(f"速度: {state['games_per_second']:.2f} games/s")
print(f"评估胜率: {state['eval_win_rate']:.1%}")
print(f"ELO: {state['elo_rating']:.0f}")
```

### 开发者使用方式

```python
from dataclasses import dataclass
from core.game import Game
from core.config import GameConfig
from games import register_game

# 1. 定义游戏配置（继承 GameConfig）
@dataclass
class MyGameConfig(GameConfig):
    board_size: int = 8
    some_option: bool = True

# 2. 实现游戏（关联配置类）
@register_game("my_game")
class MyGame(Game):
    config_class = MyGameConfig  # 关联配置类
    
    def __init__(self, config: MyGameConfig = None):
        self.config = config or MyGameConfig()
        # ... 初始化 ...
    
    # 实现抽象方法...

# 3. 使用
from games import make_game

# 默认配置
game = make_game("my_game")

# 自定义配置
game = make_game("my_game", board_size=10)

# 或使用 from_config
game = MyGame.from_config(board_size=10, some_option=False)
```

---

## 网络模型 API

### 网络组件

```python
import torch
import torch.nn as nn

class RepresentationNetwork(nn.Module):
    """
    表示网络 h(o) -> s
    
    将原始观测编码为隐状态表示。
    """
    
    def __init__(
        self,
        observation_shape: tuple[int, ...],
        hidden_state_size: int,
        num_blocks: int = 16,
    ) -> None:
        ...
    
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        """
        Args:
            observation: [B, *obs_shape] 观测张量
            
        Returns:
            hidden_state: [B, hidden_state_size] 隐状态
        """
        ...


class DynamicsNetwork(nn.Module):
    """
    动态网络 g(s, a) -> r, s'
    
    预测状态转移和即时奖励。
    """
    
    def __init__(
        self,
        hidden_state_size: int,
        action_space_size: int,
        num_blocks: int = 16,
        support_size: int = 300,
    ) -> None:
        ...
    
    def forward(
        self,
        hidden_state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_state: [B, hidden_state_size] 当前隐状态
            action: [B] 或 [B, 1] 动作索引
            
        Returns:
            next_hidden_state: [B, hidden_state_size] 下一隐状态
            reward: [B] 预测奖励
        """
        ...


class PredictionNetwork(nn.Module):
    """
    预测网络 f(s) -> p, v
    
    预测策略和价值。
    """
    
    def __init__(
        self,
        hidden_state_size: int,
        action_space_size: int,
        support_size: int = 300,
    ) -> None:
        ...
    
    def forward(
        self,
        hidden_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_state: [B, hidden_state_size] 隐状态
            
        Returns:
            policy_logits: [B, action_space_size] 策略 logits
            value: [B] 价值估计
        """
        ...
```

---

## 经验回放 API

### `PrioritizedReplayBuffer`

优先经验回放缓冲区。

```python
class PrioritizedReplayBuffer:
    """
    优先经验回放缓冲区
    
    基于 TD-error 的优先采样，支持:
        - Sum Tree 高效采样
        - 重要性采样权重
        - 批量更新优先级
    """
    
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 1e-5,
    ) -> None:
        """
        Args:
            capacity: 缓冲区容量
            alpha: 优先级指数 (0 = 均匀采样, 1 = 完全优先)
            beta: 重要性采样指数
            beta_increment: beta 每步增量
        """
        ...
    
    def add(
        self,
        game_history: GameHistory,
        priority: float | None = None,
    ) -> None:
        """
        添加完整对局
        
        Args:
            game_history: 对局历史数据
            priority: 初始优先级，None 使用最大优先级
        """
        ...
    
    def sample(
        self,
        batch_size: int,
        unroll_steps: int = 5,
    ) -> tuple[dict[str, torch.Tensor], np.ndarray, np.ndarray]:
        """
        采样训练批次
        
        Args:
            batch_size: 批次大小
            unroll_steps: 展开步数（MuZero）
            
        Returns:
            batch: 训练数据字典
            indices: 样本索引（用于更新优先级）
            weights: 重要性采样权重
        """
        ...
    
    def update_priorities(
        self,
        indices: np.ndarray,
        priorities: np.ndarray,
    ) -> None:
        """更新样本优先级"""
        ...
    
    def __len__(self) -> int:
        """当前存储的对局数"""
        ...


class GameHistory:
    """
    单局游戏历史
    
    存储完整的对局轨迹用于训练。
    """
    
    observations: list[np.ndarray]   # 观测序列
    actions: list[int]               # 动作序列
    rewards: list[float]             # 奖励序列
    policies: list[np.ndarray]       # MCTS 策略序列
    root_values: list[float]         # MCTS 根价值序列
    
    def store_transition(
        self,
        observation: np.ndarray,
        action: int,
        reward: float,
        policy: np.ndarray,
        root_value: float,
    ) -> None:
        """存储一步转移"""
        ...
    
    def compute_target_values(
        self,
        discount: float = 0.997,
        bootstrap_steps: int = 10,
    ) -> list[float]:
        """计算 n-step 目标价值"""
        ...
```

---

## Web API

### REST API 端点

ZeroForge 提供 RESTful API 用于 Web 界面和外部集成。

```
基础 URL: http://localhost:8000/api/v1
```

#### 训练控制

| 方法 | 端点 | 描述 |
|------|------|------|
| POST | `/training/start` | 启动训练 |
| POST | `/training/stop` | 停止训练 |
| POST | `/training/pause` | 暂停训练 |
| POST | `/training/resume` | 恢复训练 |
| GET  | `/training/status` | 获取训练状态 |

```python
# 请求示例: 启动训练
POST /api/v1/training/start
Content-Type: application/json

{
    "algorithm": "muzero",
    "env": "CartPole-v1",
    "config": {
        "num_simulations": 50,
        "batch_size": 256,
        "learning_rate": 0.001
    }
}

# 响应
{
    "status": "started",
    "run_id": "run_20260102_143052",
    "message": "Training started successfully"
}
```

#### 模型管理

| 方法 | 端点 | 描述 |
|------|------|------|
| GET  | `/models` | 列出所有模型 |
| GET  | `/models/{id}` | 获取模型详情 |
| POST | `/models/{id}/load` | 加载模型 |
| DELETE | `/models/{id}` | 删除模型 |

#### 指标查询

| 方法 | 端点 | 描述 |
|------|------|------|
| GET  | `/metrics` | 获取当前指标 |
| GET  | `/metrics/history` | 获取历史指标 |
| WS   | `/metrics/stream` | 实时指标流（WebSocket） |

```python
# WebSocket 连接示例
import websockets
import asyncio

async def stream_metrics():
    async with websockets.connect("ws://localhost:8000/api/v1/metrics/stream") as ws:
        async for message in ws:
            metrics = json.loads(message)
            print(f"Step: {metrics['step']}, Loss: {metrics['loss']:.4f}")

asyncio.run(stream_metrics())
```

#### 配置管理

| 方法 | 端点 | 描述 |
|------|------|------|
| GET  | `/config` | 获取当前配置 |
| PUT  | `/config` | 更新配置 |
| GET  | `/config/schema` | 获取配置 JSON Schema |

### FastAPI 应用结构

```python
from fastapi import FastAPI, WebSocket
from pydantic import BaseModel

app = FastAPI(
    title="ZeroForge API",
    version="0.1.0",
    description="ZeroForge 训练框架 API",
)

class TrainingRequest(BaseModel):
    """训练请求模型"""
    algorithm: str
    env: str
    config: dict | None = None

class TrainingStatus(BaseModel):
    """训练状态响应"""
    status: str  # running, paused, stopped, completed
    run_id: str
    current_iteration: int
    total_iterations: int
    metrics: dict[str, float]

@app.post("/api/v1/training/start")
async def start_training(request: TrainingRequest) -> dict:
    """启动训练任务"""
    ...

@app.websocket("/api/v1/metrics/stream")
async def metrics_stream(websocket: WebSocket):
    """实时指标流"""
    await websocket.accept()
    while True:
        metrics = await get_current_metrics()
        await websocket.send_json(metrics)
        await asyncio.sleep(1)
```

---

## CLI 命令

### 命令行接口

```bash
# 帮助信息
zeroforge --help

# 训练命令
zeroforge train [OPTIONS]
    --config, -c PATH        配置文件路径
    --algorithm, -a TEXT     算法名称 (muzero/alphazero)
    --env, -e TEXT           环境名称
    --iterations, -n INT     训练迭代次数
    --checkpoint PATH        继续训练的检查点
    --device TEXT            设备 (cuda/cpu)
    --seed INT               随机种子

# 评估命令
zeroforge eval [OPTIONS]
    --checkpoint, -c PATH    模型检查点路径 [必需]
    --episodes, -n INT       评估回合数 (默认: 100)
    --render                 渲染环境
    --export PATH            导出评估结果

# Web 服务
zeroforge serve [OPTIONS]
    --host TEXT              服务地址 (默认: 0.0.0.0)
    --port, -p INT           服务端口 (默认: 8000)
    --reload                 开发模式热重载

# 模型导出
zeroforge export [OPTIONS]
    --checkpoint PATH        检查点路径 [必需]
    --output PATH            输出路径
    --format TEXT            格式 (onnx/torchscript)

# 对弈演示
zeroforge play [OPTIONS]
    --checkpoint PATH        模型路径
    --env TEXT               环境名称
    --human                  人机对弈模式
```

### 命令使用示例

```bash
# 使用默认配置训练 MuZero 玩 CartPole
zeroforge train -a muzero -e CartPole-v1 -n 1000

# 使用配置文件训练
zeroforge train -c configs/muzero_atari.yaml

# 从检查点继续训练
zeroforge train -c config.yaml --checkpoint checkpoints/iter_500.pt

# 评估模型
zeroforge eval -c checkpoints/best.pt -n 100 --render

# 启动 Web 界面
zeroforge serve --port 8080

# 导出为 ONNX
zeroforge export --checkpoint model.pt --output model.onnx --format onnx

# 人机对弈
zeroforge play --checkpoint model.pt --env TicTacToe --human
```

---

## 回调与扩展

### 回调系统

```python
from abc import ABC, abstractmethod

class Callback(ABC):
    """训练回调基类"""
    
    def on_training_start(self, trainer: ZeroForgeTrainer) -> None:
        """训练开始时调用"""
        pass
    
    def on_training_end(self, trainer: ZeroForgeTrainer) -> None:
        """训练结束时调用"""
        pass
    
    def on_iteration_start(self, trainer: ZeroForgeTrainer, iteration: int) -> None:
        """每轮迭代开始时调用"""
        pass
    
    def on_iteration_end(
        self,
        trainer: ZeroForgeTrainer,
        iteration: int,
        metrics: dict[str, float],
    ) -> None:
        """每轮迭代结束时调用"""
        pass
    
    def on_game_end(
        self,
        trainer: ZeroForgeTrainer,
        game_history: GameHistory,
    ) -> None:
        """每局游戏结束时调用"""
        pass


# 内置回调
class TensorBoardCallback(Callback):
    """TensorBoard 日志回调"""
    ...

class WandBCallback(Callback):
    """Weights & Biases 日志回调"""
    ...

class CheckpointCallback(Callback):
    """模型检查点回调"""
    ...

class EarlyStoppingCallback(Callback):
    """早停回调"""
    ...
```

---

## 错误处理

### 异常类型

```python
class ZeroForgeError(Exception):
    """ZeroForge 基础异常"""
    pass

class ConfigurationError(ZeroForgeError):
    """配置错误"""
    pass

class TrainingError(ZeroForgeError):
    """训练过程错误"""
    pass

class ModelError(ZeroForgeError):
    """模型相关错误"""
    pass

class EnvironmentError(ZeroForgeError):
    """环境相关错误"""
    pass
```

### 错误处理示例

```python
from zeroforge.exceptions import ConfigurationError, TrainingError

try:
    trainer = ZeroForgeTrainer(algorithm, env, config)
    trainer.train(num_iterations=1000)
except ConfigurationError as e:
    logger.error(f"配置错误: {e}")
    raise
except TrainingError as e:
    logger.error(f"训练错误: {e}")
    # 尝试保存当前状态
    trainer.save("emergency_checkpoint.pt")
    raise
```

---

## 附录

### 类型定义

```python
from typing import TypeAlias
import numpy as np
import torch

# 观测类型
Observation: TypeAlias = np.ndarray | torch.Tensor

# 动作类型
Action: TypeAlias = int | np.ndarray

# 配置类型
Config: TypeAlias = dict[str, any] | TrainerConfig

# 指标类型
Metrics: TypeAlias = dict[str, float]
```

### 配置文件示例

```yaml
# config.yaml - MuZero CartPole 配置示例

algorithm:
  name: muzero
  num_simulations: 50
  discount: 0.997
  
network:
  hidden_state_size: 256
  num_residual_blocks: 8
  
training:
  batch_size: 256
  learning_rate: 0.001
  weight_decay: 0.0001
  num_iterations: 1000
  
self_play:
  num_actors: 4
  games_per_iteration: 100
  temperature_schedule:
    - [0, 1.0]      # 迭代 0: 温度 1.0
    - [500, 0.5]    # 迭代 500: 温度 0.5
    - [800, 0.25]   # 迭代 800: 温度 0.25
    
replay_buffer:
  size: 100000
  priority_alpha: 0.6
  priority_beta: 0.4
  
environment:
  name: CartPole-v1
  
logging:
  tensorboard: true
  wandb: false
  log_interval: 10
```

"""
Server - Web 服务模块

提供 FastAPI 后端和 WebSocket 实时推送。

功能:
- REST API: 配置管理、游戏控制
- WebSocket: 训练状态、对弈实时推送
- 静态文件: Web 前端资源
"""

from .api import create_app, get_app
from .manager import TrainingManager, GameManager, SystemManager

__all__ = [
    "create_app",
    "get_app",
    "TrainingManager",
    "GameManager", 
    "SystemManager",
]


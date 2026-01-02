"""
API - FastAPI 后端实现

提供 REST API 和 WebSocket 接口。

路由:
- /api/config: 配置管理
- /api/training: 训练控制
- /api/games: 游戏管理
- /api/system: 系统信息
- /ws/training: 训练状态推送
- /ws/game/{game_id}: 对弈实时推送
"""

import asyncio
import json
import logging
from typing import Dict, Any, Optional, List, Set
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

from .manager import TrainingManager, GameManager, SystemManager

logger = logging.getLogger(__name__)


# ============================================================
# Pydantic 模型
# ============================================================

class ConfigUpdate(BaseModel):
    """配置更新请求"""
    section: str
    values: Dict[str, Any]


class TrainingCommand(BaseModel):
    """训练命令"""
    action: str  # start | pause | resume | stop


class GameCreate(BaseModel):
    """创建游戏请求（通用）"""
    game_type: str = "tictactoe"
    players: List[str] = ["human", "ai:muzero"]  # 通用玩家列表


class ActionRequest(BaseModel):
    """执行动作请求（通用）"""
    action: int  # 动作索引


class DebugSessionCreate(BaseModel):
    """创建调试会话请求"""
    game_type: str = "tictactoe"
    algorithm: str = "alphazero"
    device: str = "cpu"
    checkpoint_path: Optional[str] = None


class DebugStepRequest(BaseModel):
    """调试步骤请求"""
    action: Optional[int] = None  # 指定动作，None 则使用 MCTS
    num_simulations: int = 50  # MCTS 模拟次数


# ============================================================
# 全局管理器
# ============================================================

training_manager: Optional[TrainingManager] = None
game_manager: Optional[GameManager] = None
system_manager: Optional[SystemManager] = None
debug_manager: Optional["DebugManager"] = None


# ============================================================
# WebSocket 连接管理
# ============================================================

class ConnectionManager:
    """WebSocket 连接管理器"""
    
    def __init__(self):
        self.training_connections: Set[WebSocket] = set()
        self.game_connections: Dict[str, Set[WebSocket]] = {}
    
    async def connect_training(self, websocket: WebSocket):
        await websocket.accept()
        self.training_connections.add(websocket)
        logger.info(f"训练 WebSocket 连接: {len(self.training_connections)} 个活跃连接")
    
    def disconnect_training(self, websocket: WebSocket):
        self.training_connections.discard(websocket)
        logger.info(f"训练 WebSocket 断开: {len(self.training_connections)} 个活跃连接")
    
    async def connect_game(self, game_id: str, websocket: WebSocket):
        await websocket.accept()
        if game_id not in self.game_connections:
            self.game_connections[game_id] = set()
        self.game_connections[game_id].add(websocket)
    
    def disconnect_game(self, game_id: str, websocket: WebSocket):
        if game_id in self.game_connections:
            self.game_connections[game_id].discard(websocket)
    
    async def broadcast_training(self, data: Dict[str, Any]):
        """广播训练状态"""
        message = json.dumps(data)
        disconnected = set()
        
        for ws in self.training_connections:
            try:
                await ws.send_text(message)
            except Exception:
                disconnected.add(ws)
        
        self.training_connections -= disconnected
    
    async def broadcast_game(self, game_id: str, data: Dict[str, Any]):
        """广播游戏状态"""
        if game_id not in self.game_connections:
            return
        
        message = json.dumps(data)
        disconnected = set()
        
        for ws in self.game_connections[game_id]:
            try:
                await ws.send_text(message)
            except Exception:
                disconnected.add(ws)
        
        self.game_connections[game_id] -= disconnected


ws_manager = ConnectionManager()


# ============================================================
# 应用工厂
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期"""
    global training_manager, game_manager, system_manager, debug_manager
    
    from .manager import DebugManager
    
    # 启动
    training_manager = TrainingManager()
    game_manager = GameManager()
    system_manager = SystemManager()
    debug_manager = DebugManager()
    
    # 获取主事件循环（用于线程安全调度）
    main_loop = asyncio.get_running_loop()
    
    # 订阅训练状态更新（线程安全）
    def on_training_update(status):
        try:
            # 从后台线程安全地调度到主事件循环
            asyncio.run_coroutine_threadsafe(
                ws_manager.broadcast_training({
                    "type": "training_status",
                    "data": status
                }),
                main_loop
            )
        except Exception as e:
            logger.debug(f"广播训练状态失败: {e}")
    
    training_manager.subscribe(on_training_update)
    
    logger.info("服务启动完成")
    
    yield
    
    # 关闭
    training_manager.unsubscribe(on_training_update)
    logger.info("服务关闭")


def create_app() -> FastAPI:
    """创建 FastAPI 应用"""
    app = FastAPI(
        title="RL Framework API",
        description="强化学习框架 API",
        version="1.0.0",
        lifespan=lifespan,
    )
    
    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # 注册路由
    _register_routes(app)
    
    return app


def _register_routes(app: FastAPI):
    """注册路由"""
    
    # === 系统 API ===
    
    @app.get("/api/system/info")
    async def get_system_info():
        """获取系统信息"""
        return system_manager.get_system_info()
    
    @app.get("/api/system/games")
    async def list_games():
        """列出可用游戏"""
        return system_manager.list_games()
    
    @app.get("/api/system/algorithms")
    async def list_algorithms():
        """列出可用算法"""
        return system_manager.list_algorithms()
    
    # === 配置 API ===
    
    @app.get("/api/config")
    async def get_config(section: Optional[str] = Query(None)):
        """获取配置"""
        return system_manager.get_config(section)
    
    @app.get("/api/config/schema")
    async def get_config_schema():
        """获取配置 schema（用于动态渲染表单）"""
        return system_manager.get_config_schema()
    
    @app.post("/api/config")
    async def update_config(req: ConfigUpdate):
        """更新配置"""
        system_manager.set_config(req.section, req.values)
        return {"success": True}
    
    @app.put("/api/config")
    async def update_config_batch(values: Dict[str, Any]):
        """批量更新配置"""
        # 验证配置合法性
        try:
            from core.training_config import TrainingConfig
            # 合并当前配置和新值进行验证
            current = system_manager.get_config()
            merged = {**current, **values}
            TrainingConfig.from_dict(merged)  # 触发验证
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        
        system_manager.update_config(values)
        return {"success": True}
    
    # === 训练 API ===
    
    @app.get("/api/training/status")
    async def get_training_status():
        """获取训练状态"""
        return training_manager.get_status()
    
    @app.get("/api/training/debug")
    async def get_training_debug(
        category: Optional[str] = Query(None, description="类别: selfplay/training/trajectories/mcts"),
        limit: int = Query(50, description="返回条数")
    ):
        """获取训练调试数据
        
        返回训练过程中的详细调试信息，包括：
        - selfplay: 自玩游戏信息
        - training: 训练批次信息
        - trajectories: 轨迹数据
        - mcts: MCTS 搜索详情
        """
        return training_manager.get_debug_data(category, limit)
    
    @app.delete("/api/training/debug")
    async def clear_training_debug():
        """清空调试数据"""
        training_manager.clear_debug_data()
        return {"success": True}
    
    @app.post("/api/training/command")
    async def training_command(cmd: TrainingCommand):
        """训练控制命令"""
        if cmd.action == "start":
            config = system_manager.get_config()
            training_manager.start(config)
        elif cmd.action == "pause":
            training_manager.pause()
        elif cmd.action == "resume":
            training_manager.resume()
        elif cmd.action == "stop":
            training_manager.stop()
        else:
            raise HTTPException(status_code=400, detail=f"未知命令: {cmd.action}")
        
        return {"success": True, "action": cmd.action}
    
    @app.post("/api/training/save")
    async def save_checkpoint():
        """手动保存检查点"""
        result = training_manager.save_checkpoint()
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    
    # === 检查点 API ===
    
    @app.get("/api/checkpoints")
    async def list_checkpoints(
        game_type: Optional[str] = Query(None),
        algorithm: Optional[str] = Query(None),
    ):
        """列出检查点"""
        from core.checkpoint import get_checkpoint_manager
        manager = get_checkpoint_manager(system_manager.get_config().get("checkpoint_dir", "./checkpoints"))
        checkpoints = manager.list_checkpoints(game_type, algorithm)
        return {"checkpoints": [cp.to_dict() for cp in checkpoints]}
    
    @app.get("/api/checkpoints/{game_type}/{algorithm}")
    async def list_checkpoints_for_game(game_type: str, algorithm: str):
        """列出指定游戏/算法的检查点"""
        from core.checkpoint import get_checkpoint_manager
        manager = get_checkpoint_manager(system_manager.get_config().get("checkpoint_dir", "./checkpoints"))
        checkpoints = manager.list_checkpoints(game_type, algorithm)
        return {"checkpoints": [cp.to_dict() for cp in checkpoints]}
    
    @app.delete("/api/checkpoints")
    async def delete_checkpoint(path: str = Query(...)):
        """删除检查点"""
        from core.checkpoint import get_checkpoint_manager
        manager = get_checkpoint_manager(system_manager.get_config().get("checkpoint_dir", "./checkpoints"))
        success = manager.delete_checkpoint(path)
        if not success:
            raise HTTPException(status_code=404, detail="检查点不存在或删除失败")
        return {"success": True}
    
    # === 调试 API ===
    
    @app.get("/api/debug/sessions")
    async def list_debug_sessions():
        """列出所有调试会话"""
        return {"sessions": debug_manager.list_sessions()}
    
    @app.post("/api/debug/sessions")
    async def create_debug_session(req: DebugSessionCreate):
        """创建调试会话"""
        try:
            result = debug_manager.create_session(
                game_type=req.game_type,
                algorithm=req.algorithm,
                device=req.device,
                checkpoint_path=req.checkpoint_path,
            )
            return result
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.get("/api/debug/sessions/{session_id}")
    async def get_debug_session(session_id: str):
        """获取调试会话状态"""
        result = debug_manager.get_session(session_id)
        if result is None:
            raise HTTPException(status_code=404, detail="会话不存在")
        return result
    
    @app.delete("/api/debug/sessions/{session_id}")
    async def delete_debug_session(session_id: str):
        """删除调试会话"""
        success = debug_manager.delete_session(session_id)
        if not success:
            raise HTTPException(status_code=404, detail="会话不存在")
        return {"success": True}
    
    @app.post("/api/debug/sessions/{session_id}/step")
    async def debug_step_game(session_id: str, req: DebugStepRequest = None):
        """执行一步游戏（使用 MCTS 或指定动作）"""
        try:
            action = req.action if req else None
            result = debug_manager.step_game(session_id, action)
            return result
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.post("/api/debug/sessions/{session_id}/mcts")
    async def debug_step_mcts(session_id: str, num_simulations: int = Query(50)):
        """执行 MCTS 搜索（不执行动作，只查看搜索结果）"""
        try:
            result = debug_manager.step_mcts(session_id, num_simulations)
            return result
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.post("/api/debug/sessions/{session_id}/reset")
    async def debug_reset_game(session_id: str):
        """重置游戏"""
        try:
            result = debug_manager.reset_game(session_id)
            return result
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.post("/api/debug/sessions/{session_id}/run")
    async def debug_run_full_game(session_id: str):
        """运行完整一局游戏"""
        try:
            result = debug_manager.run_full_game(session_id)
            return result
        except ValueError as e:
            raise HTTPException(status_code=404, detail=str(e))
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    # === 游戏 API ===
    
    @app.get("/api/games")
    async def list_game_sessions():
        """列出游戏会话"""
        return game_manager.list_sessions()
    
    @app.post("/api/games")
    async def create_game(req: GameCreate):
        """创建游戏会话（通用）"""
        session_id = game_manager.create_session(
            game_type=req.game_type,
            players=req.players,
        )
        return {"session_id": session_id}
    
    @app.get("/api/games/{game_id}")
    async def get_game(game_id: str):
        """获取游戏状态"""
        session = game_manager.get_session(game_id)
        if session is None:
            raise HTTPException(status_code=404, detail="游戏不存在")
        return session
    
    @app.post("/api/games/{game_id}/start")
    async def start_game(game_id: str):
        """开始游戏"""
        result = game_manager.start_game(game_id)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    
    @app.post("/api/games/{game_id}/action")
    async def do_action(game_id: str, req: ActionRequest):
        """执行动作（通用）"""
        result = game_manager.do_action(game_id, req.action)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        
        # 广播更新
        session = game_manager.get_session(game_id)
        await ws_manager.broadcast_game(game_id, {
            "type": "game_update",
            "data": session
        })
        
        return result
    
    @app.delete("/api/games/{game_id}")
    async def delete_game(game_id: str):
        """删除游戏会话"""
        result = game_manager.delete_session(game_id)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    
    @app.delete("/api/games")
    async def clear_all_games():
        """清空所有游戏会话"""
        result = game_manager.clear_all_sessions()
        return result
    
    @app.get("/api/games/{game_id}/render")
    async def get_game_render(game_id: str, mode: str = "json"):
        """获取游戏渲染数据"""
        result = game_manager.get_render(game_id, mode)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    
    @app.get("/api/games/{game_id}/legal_actions")
    async def get_legal_actions(game_id: str):
        """获取合法动作列表"""
        result = game_manager.get_legal_actions(game_id)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    
    # === WebSocket ===
    
    @app.websocket("/ws/training")
    async def websocket_training(websocket: WebSocket):
        """训练状态 WebSocket
        
        改进：
        1. 服务端主动发送心跳，避免连接超时断开
        2. 使用 asyncio.wait_for 添加接收超时
        3. 异常处理更健壮
        """
        await ws_manager.connect_training(websocket)
        
        # 发送当前状态
        try:
            await websocket.send_json({
                "type": "training_status",
                "data": training_manager.get_status()
            })
        except Exception as e:
            logger.error(f"发送初始状态失败: {e}")
            ws_manager.disconnect_training(websocket)
            return
        
        # 心跳间隔（秒）
        heartbeat_interval = 15
        last_heartbeat = asyncio.get_event_loop().time()
        
        try:
            while True:
                try:
                    # 带超时的接收消息，超时后发送心跳
                    data = await asyncio.wait_for(
                        websocket.receive_text(),
                        timeout=heartbeat_interval
                    )
                    msg = json.loads(data)
                    
                    if msg.get("type") == "ping":
                        await websocket.send_json({"type": "pong"})
                    elif msg.get("type") == "command":
                        # 处理命令
                        action = msg.get("action")
                        if action == "start":
                            config = system_manager.get_config()
                            training_manager.start(config)
                        elif action == "pause":
                            training_manager.pause()
                        elif action == "resume":
                            training_manager.resume()
                        elif action == "stop":
                            training_manager.stop()
                        # 立即返回新状态
                        await websocket.send_json({
                            "type": "training_status",
                            "data": training_manager.get_status()
                        })
                        
                except asyncio.TimeoutError:
                    # 超时 - 发送服务端心跳保持连接
                    current_time = asyncio.get_event_loop().time()
                    if current_time - last_heartbeat >= heartbeat_interval:
                        try:
                            await websocket.send_json({
                                "type": "heartbeat",
                                "timestamp": current_time
                            })
                            last_heartbeat = current_time
                        except Exception:
                            # 发送失败说明连接已断开
                            break
                            
        except WebSocketDisconnect:
            logger.debug("训练 WebSocket 客户端主动断开")
        except Exception as e:
            logger.warning(f"训练 WebSocket 异常: {e}")
        finally:
            ws_manager.disconnect_training(websocket)
    
    @app.websocket("/ws/game/{game_id}")
    async def websocket_game(websocket: WebSocket, game_id: str):
        """游戏状态 WebSocket"""
        session = game_manager.get_session(game_id)
        if session is None:
            await websocket.close(code=4004, reason="游戏不存在")
            return
        
        await ws_manager.connect_game(game_id, websocket)
        
        # 发送当前状态
        await websocket.send_json({
            "type": "game_state",
            "data": session
        })
        
        # 订阅游戏更新
        async def on_game_update(data):
            await ws_manager.broadcast_game(game_id, {
                "type": "game_update",
                "data": data
            })
        
        try:
            while True:
                data = await websocket.receive_text()
                msg = json.loads(data)
                
                if msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                elif msg.get("type") == "move":
                    move = msg.get("move")
                    if move:
                        game_manager.make_move(game_id, move)
        except WebSocketDisconnect:
            ws_manager.disconnect_game(game_id, websocket)
    
    # === 调试 API ===
    
    @app.post("/api/debug/game/create")
    async def create_debug_game(game_type: str = "chinese_chess", config: Optional[Dict[str, Any]] = None):
        """创建调试游戏会话"""
        from games import make_game, get_game_info
        import time
        
        # 生成调试会话 ID
        game_id = f"debug_{int(time.time() * 1000)}"
        
        try:
            # 创建游戏实例
            game = make_game(game_type, **(config or {}))
            game.reset()
            
            # 存储到调试会话
            game_manager.debug_sessions[game_id] = {
                "game": game,
                "game_type": game_type,
                "config": config or {},
                "history": [],
                "created_at": time.time(),
            }
            
            return {
                "game_id": game_id,
                "game_type": game_type,
                "state": _get_debug_state(game, 0),
            }
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
    
    @app.get("/api/debug/game/{game_id}/state")
    async def get_debug_state(
        game_id: str, 
        include_observation: bool = False,
        include_legal_actions: bool = True
    ):
        """获取调试游戏状态"""
        session = game_manager.debug_sessions.get(game_id)
        if not session:
            raise HTTPException(status_code=404, detail="调试会话不存在")
        
        game = session["game"]
        step_index = len(session["history"])
        
        state = _get_debug_state(game, step_index)
        
        if include_observation:
            import base64
            obs = game.get_observation()
            state["observation"] = {
                "shape": list(obs.shape),
                "dtype": str(obs.dtype),
                "data_base64": base64.b64encode(obs.tobytes()).decode(),
            }
        
        if include_legal_actions:
            state["legal_actions"] = game.legal_actions()
        
        # 添加游戏特定调试信息
        if hasattr(game, 'get_debug_info'):
            state["game_specific"] = game.get_debug_info()
        
        return state
    
    @app.post("/api/debug/game/{game_id}/step")
    async def debug_step(game_id: str, action: int, run_mcts: bool = False, mcts_simulations: int = 50):
        """执行调试游戏一步"""
        session = game_manager.debug_sessions.get(game_id)
        if not session:
            raise HTTPException(status_code=404, detail="调试会话不存在")
        
        game = session["game"]
        
        # 验证动作合法性
        if action not in game.legal_actions():
            raise HTTPException(status_code=400, detail=f"非法动作: {action}")
        
        # 记录历史
        step_record = {
            "step_index": len(session["history"]),
            "action": action,
            "player": game.current_player(),
            "legal_actions_before": game.legal_actions(),
        }
        
        # 执行动作
        obs, reward, done, info = game.step(action)
        
        # 更新记录
        step_record.update({
            "reward": reward,
            "done": done,
            "info": info,
        })
        session["history"].append(step_record)
        
        result = {
            "step_index": len(session["history"]),
            "reward": reward,
            "done": done,
            "info": info,
            "current_player": game.current_player() if not done else None,
            "legal_actions_count": len(game.legal_actions()) if not done else 0,
        }
        
        return result
    
    @app.post("/api/debug/game/{game_id}/reset")
    async def reset_debug_game(game_id: str):
        """重置调试游戏"""
        session = game_manager.debug_sessions.get(game_id)
        if not session:
            raise HTTPException(status_code=404, detail="调试会话不存在")
        
        game = session["game"]
        game.reset()
        session["history"] = []
        
        return {
            "game_id": game_id,
            "state": _get_debug_state(game, 0),
        }
    
    @app.get("/api/debug/game/{game_id}/history")
    async def get_debug_history(game_id: str):
        """获取调试游戏历史"""
        session = game_manager.debug_sessions.get(game_id)
        if not session:
            raise HTTPException(status_code=404, detail="调试会话不存在")
        
        return {
            "game_id": game_id,
            "game_type": session["game_type"],
            "config": session["config"],
            "history": session["history"],
            "total_steps": len(session["history"]),
        }
    
    @app.get("/api/debug/game/{game_id}/render")
    async def render_debug_game(game_id: str, mode: str = "json"):
        """渲染调试游戏"""
        session = game_manager.debug_sessions.get(game_id)
        if not session:
            raise HTTPException(status_code=404, detail="调试会话不存在")
        
        game = session["game"]
        
        if hasattr(game, 'render'):
            result = game.render(mode=mode)
            if mode == "svg":
                from fastapi.responses import Response
                return Response(content=result, media_type="image/svg+xml")
            return {"render": result}
        
        return {"error": "游戏不支持渲染"}
    
    @app.delete("/api/debug/game/{game_id}")
    async def delete_debug_game(game_id: str):
        """删除调试会话"""
        if game_id in game_manager.debug_sessions:
            del game_manager.debug_sessions[game_id]
            return {"success": True}
        raise HTTPException(status_code=404, detail="调试会话不存在")
    
    @app.get("/api/debug/games")
    async def list_debug_games():
        """列出所有调试会话"""
        sessions = []
        for gid, session in game_manager.debug_sessions.items():
            sessions.append({
                "game_id": gid,
                "game_type": session["game_type"],
                "steps": len(session["history"]),
                "created_at": session["created_at"],
            })
        return {"sessions": sessions}
    
    # === 调试 WebSocket ===
    
    @app.websocket("/ws/debug/game/{game_id}")
    async def websocket_debug_game(websocket: WebSocket, game_id: str):
        """调试游戏 WebSocket"""
        session = game_manager.debug_sessions.get(game_id)
        if not session:
            await websocket.close(code=4004, reason="调试会话不存在")
            return
        
        await websocket.accept()
        game = session["game"]
        
        # 发送当前状态
        await websocket.send_json({
            "type": "state",
            "data": _get_debug_state(game, len(session["history"]))
        })
        
        try:
            while True:
                data = await websocket.receive_text()
                msg = json.loads(data)
                
                if msg.get("type") == "ping":
                    await websocket.send_json({"type": "pong"})
                
                elif msg.get("type") == "step":
                    action = msg.get("action")
                    if action is not None and action in game.legal_actions():
                        obs, reward, done, info = game.step(action)
                        session["history"].append({
                            "action": action,
                            "reward": reward,
                            "done": done,
                        })
                        await websocket.send_json({
                            "type": "step_result",
                            "data": {
                                "step_index": len(session["history"]),
                                "reward": reward,
                                "done": done,
                                "state": _get_debug_state(game, len(session["history"]))
                            }
                        })
                    else:
                        await websocket.send_json({
                            "type": "error",
                            "message": f"非法动作: {action}"
                        })
                
                elif msg.get("type") == "reset":
                    game.reset()
                    session["history"] = []
                    await websocket.send_json({
                        "type": "reset",
                        "data": _get_debug_state(game, 0)
                    })
                
                elif msg.get("type") == "get_legal_actions":
                    await websocket.send_json({
                        "type": "legal_actions",
                        "data": game.legal_actions()
                    })
        
        except WebSocketDisconnect:
            pass


def _to_python(obj):
    """将 numpy 类型转换为 Python 原生类型"""
    import numpy as np
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_to_python(v) for v in obj]
    return obj


def _get_debug_state(game, step_index: int) -> Dict[str, Any]:
    """获取调试状态快照"""
    # 使用 _to_python 转换 numpy 类型
    return _to_python({
        "step_index": step_index,
        "current_player": game.current_player(),
        "is_terminal": game.is_terminal(),
        "winner": game.get_winner(),
        "legal_actions_count": len(game.legal_actions()),
    })


# ============================================================
# 单例
# ============================================================

_app: Optional[FastAPI] = None


def get_app() -> FastAPI:
    """获取应用单例"""
    global _app
    if _app is None:
        _app = create_app()
    return _app


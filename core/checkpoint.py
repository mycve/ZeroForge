"""
CheckpointManager - 检查点管理

提供模型检查点的保存、加载和管理功能。

特性:
- 自动保存最佳模型
- 保留最近 N 个检查点
- 支持恢复训练
"""

import os
import json
import logging
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime

logger = logging.getLogger(__name__)


# ============================================================
# 检查点信息
# ============================================================

@dataclass
class CheckpointInfo:
    """检查点信息"""
    path: str
    game_type: str
    algorithm: str
    epoch: int
    step: int
    loss: float
    eval_win_rate: float
    created_at: str
    is_best: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CheckpointInfo":
        return cls(**data)


# ============================================================
# 检查点管理器
# ============================================================

class CheckpointManager:
    """检查点管理器
    
    管理模型检查点的保存、加载和清理。
    
    目录结构:
        checkpoints/
        ├── tictactoe_alphazero/
        │   ├── epoch_10.pt
        │   ├── epoch_20.pt
        │   ├── best.pt -> epoch_20.pt
        │   └── meta.json
        └── chinese_chess_muzero/
            └── ...
    
    Attributes:
        base_dir: 检查点根目录
        keep_checkpoints: 保留的检查点数量
    
    Example:
        >>> manager = CheckpointManager("./checkpoints")
        >>> manager.save(epoch=10, network=net, optimizer=opt, config=cfg, metrics=m)
        >>> checkpoint = manager.load_latest("tictactoe", "alphazero")
    """
    
    def __init__(self, base_dir: str = "./checkpoints", keep_checkpoints: int = 5):
        self.base_dir = Path(base_dir)
        self.keep_checkpoints = keep_checkpoints
        
        # 确保目录存在
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"检查点目录: {self.base_dir.absolute()}")
    
    def _get_run_dir(self, game_type: str, algorithm: str) -> Path:
        """获取特定游戏/算法的检查点目录"""
        run_dir = self.base_dir / f"{game_type}_{algorithm}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir
    
    def _get_meta_path(self, game_type: str, algorithm: str) -> Path:
        """获取元数据文件路径"""
        return self._get_run_dir(game_type, algorithm) / "meta.json"
    
    def _load_meta(self, game_type: str, algorithm: str) -> Dict[str, Any]:
        """加载元数据"""
        meta_path = self._get_meta_path(game_type, algorithm)
        if meta_path.exists():
            try:
                with open(meta_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"加载元数据失败: {e}")
        return {"checkpoints": [], "best_epoch": None, "best_metric": -float("inf")}
    
    def _save_meta(self, game_type: str, algorithm: str, meta: Dict[str, Any]):
        """保存元数据"""
        meta_path = self._get_meta_path(game_type, algorithm)
        try:
            with open(meta_path, 'w') as f:
                json.dump(meta, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"保存元数据失败: {e}")
    
    def save(
        self,
        game_type: str,
        algorithm: str,
        epoch: int,
        network: Any,
        optimizer: Any,
        config: Dict[str, Any],
        metrics: Dict[str, float],
        is_manual: bool = False,
    ) -> str:
        """保存检查点
        
        Args:
            game_type: 游戏类型
            algorithm: 算法名称
            epoch: 训练轮数
            network: 神经网络 (nn.Module)
            optimizer: 优化器
            config: 训练配置
            metrics: 训练指标
            is_manual: 是否手动保存
            
        Returns:
            保存的检查点路径
        """
        import torch
        
        run_dir = self._get_run_dir(game_type, algorithm)
        checkpoint_name = f"epoch_{epoch}.pt"
        checkpoint_path = run_dir / checkpoint_name
        
        # 构建检查点数据
        checkpoint_data = {
            "epoch": epoch,
            "game_type": game_type,
            "algorithm": algorithm,
            "network_state_dict": network.state_dict() if network else None,
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
            "config": config,
            "metrics": metrics,
            "created_at": datetime.now().isoformat(),
        }
        
        # 保存检查点
        torch.save(checkpoint_data, checkpoint_path)
        logger.info(f"保存检查点: {checkpoint_path}")
        
        # 更新元数据
        meta = self._load_meta(game_type, algorithm)
        
        checkpoint_info = CheckpointInfo(
            path=str(checkpoint_path),
            game_type=game_type,
            algorithm=algorithm,
            epoch=epoch,
            step=metrics.get("step", 0),
            loss=metrics.get("loss", 0.0),
            eval_win_rate=metrics.get("eval_win_rate", 0.0),
            created_at=checkpoint_data["created_at"],
        )
        
        meta["checkpoints"].append(checkpoint_info.to_dict())
        
        # 更新最佳模型
        eval_win_rate = metrics.get("eval_win_rate", 0.0)
        if eval_win_rate > meta.get("best_metric", -float("inf")):
            meta["best_epoch"] = epoch
            meta["best_metric"] = eval_win_rate
            
            # 创建 best.pt 符号链接或复制
            best_path = run_dir / "best.pt"
            if best_path.exists():
                best_path.unlink()
            shutil.copy(checkpoint_path, best_path)
            logger.info(f"更新最佳模型: epoch {epoch}, win_rate={eval_win_rate:.2%}")
        
        self._save_meta(game_type, algorithm, meta)
        
        # 清理旧检查点（保留最近 N 个 + best）
        if not is_manual:
            self._cleanup_old_checkpoints(game_type, algorithm)
        
        return str(checkpoint_path)
    
    def _cleanup_old_checkpoints(self, game_type: str, algorithm: str):
        """清理旧检查点，保留最近 N 个"""
        meta = self._load_meta(game_type, algorithm)
        checkpoints = meta.get("checkpoints", [])
        
        if len(checkpoints) <= self.keep_checkpoints:
            return
        
        # 按 epoch 排序，保留最新的
        checkpoints_sorted = sorted(checkpoints, key=lambda x: x["epoch"], reverse=True)
        to_keep = checkpoints_sorted[:self.keep_checkpoints]
        to_remove = checkpoints_sorted[self.keep_checkpoints:]
        
        # 删除旧检查点
        for cp in to_remove:
            cp_path = Path(cp["path"])
            if cp_path.exists():
                try:
                    cp_path.unlink()
                    logger.debug(f"删除旧检查点: {cp_path}")
                except Exception as e:
                    logger.warning(f"删除检查点失败: {e}")
        
        # 更新元数据
        meta["checkpoints"] = to_keep
        self._save_meta(game_type, algorithm, meta)
    
    def load(self, path: str) -> Dict[str, Any]:
        """加载检查点
        
        Args:
            path: 检查点路径
            
        Returns:
            检查点数据字典
        """
        import torch
        
        checkpoint_path = Path(path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"检查点不存在: {path}")
        
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        logger.info(f"加载检查点: {path}, epoch={checkpoint.get('epoch')}")
        return checkpoint
    
    def load_latest(self, game_type: str, algorithm: str) -> Optional[Dict[str, Any]]:
        """加载最新检查点"""
        meta = self._load_meta(game_type, algorithm)
        checkpoints = meta.get("checkpoints", [])
        
        if not checkpoints:
            return None
        
        # 找最新的
        latest = max(checkpoints, key=lambda x: x["epoch"])
        return self.load(latest["path"])
    
    def load_best(self, game_type: str, algorithm: str) -> Optional[Dict[str, Any]]:
        """加载最佳检查点"""
        run_dir = self._get_run_dir(game_type, algorithm)
        best_path = run_dir / "best.pt"
        
        if best_path.exists():
            return self.load(str(best_path))
        return None
    
    def list_checkpoints(
        self,
        game_type: Optional[str] = None,
        algorithm: Optional[str] = None,
    ) -> List[CheckpointInfo]:
        """列出检查点
        
        Args:
            game_type: 可选，过滤游戏类型
            algorithm: 可选，过滤算法
            
        Returns:
            检查点信息列表
        """
        results = []
        
        # 遍历所有运行目录
        if not self.base_dir.exists():
            return results
        
        for run_dir in self.base_dir.iterdir():
            if not run_dir.is_dir():
                continue
            
            # 直接读取 meta.json 获取信息（避免解析目录名的问题）
            meta_path = run_dir / "meta.json"
            if not meta_path.exists():
                continue
            
            try:
                with open(meta_path, 'r') as f:
                    meta = json.load(f)
            except Exception:
                continue
            
            # 从检查点数据中获取 game_type 和 algorithm
            checkpoints = meta.get("checkpoints", [])
            if not checkpoints:
                continue
            
            # 从第一个检查点获取元信息
            first_cp = checkpoints[0]
            gt = first_cp.get("game_type", "")
            algo = first_cp.get("algorithm", "")
            
            # 过滤
            if game_type and gt != game_type:
                continue
            if algorithm and algo != algorithm:
                continue
            
            best_epoch = meta.get("best_epoch")
            
            for cp in checkpoints:
                info = CheckpointInfo.from_dict(cp)
                info.is_best = (info.epoch == best_epoch)
                results.append(info)
        
        # 按时间倒序
        results.sort(key=lambda x: x.created_at, reverse=True)
        return results
    
    def get_checkpoint_info(self, path: str) -> Optional[CheckpointInfo]:
        """获取检查点信息"""
        all_checkpoints = self.list_checkpoints()
        for cp in all_checkpoints:
            if cp.path == path:
                return cp
        return None
    
    def delete_checkpoint(self, path: str) -> bool:
        """删除检查点"""
        try:
            checkpoint_path = Path(path)
            if checkpoint_path.exists():
                checkpoint_path.unlink()
                logger.info(f"删除检查点: {path}")
                return True
        except Exception as e:
            logger.error(f"删除检查点失败: {e}")
        return False


# ============================================================
# 全局单例
# ============================================================

_checkpoint_manager: Optional[CheckpointManager] = None


def get_checkpoint_manager(base_dir: str = "./checkpoints", keep_checkpoints: int = 5) -> CheckpointManager:
    """获取检查点管理器单例"""
    global _checkpoint_manager
    if _checkpoint_manager is None:
        _checkpoint_manager = CheckpointManager(base_dir, keep_checkpoints)
    return _checkpoint_manager


# ============================================================
# 导出
# ============================================================

__all__ = [
    "CheckpointInfo",
    "CheckpointManager",
    "get_checkpoint_manager",
]

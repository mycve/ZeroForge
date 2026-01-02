"""
Structured Logger - 结构化日志系统

支持多后端日志输出:
- Console: 控制台输出（彩色、格式化）
- TensorBoard: TensorBoard 可视化
- WandB: Weights & Biases 云端记录
- JSON File: JSON 文件持久化

设计原则:
1. 结构化：所有日志都是带时间戳和分类的事件
2. 可扩展：轻松添加新的后端
3. 异步安全：支持多进程/多线程环境
"""

import os
import sys
import json
import time
import logging
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum, auto
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
from queue import Queue
import numpy as np


# ============================================================
# 日志级别和分类
# ============================================================

class LogLevel(Enum):
    """日志级别"""
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()


class LogCategory(Enum):
    """日志分类"""
    SYSTEM = "system"           # 系统事件（启动、关闭等）
    TRAINING = "training"       # 训练指标
    SELFPLAY = "selfplay"       # 自玩统计
    EVALUATION = "evaluation"   # 评估结果
    CHECKPOINT = "checkpoint"   # 检查点保存/加载
    CONFIG = "config"           # 配置变更
    PERFORMANCE = "performance" # 性能指标


# ============================================================
# 日志事件
# ============================================================

@dataclass
class LogEvent:
    """结构化日志事件
    
    Attributes:
        timestamp: Unix 时间戳
        level: 日志级别
        category: 日志分类
        message: 日志消息
        metrics: 数值指标字典
        tags: 标签字典（用于过滤）
        step: 全局步数（可选）
        epoch: 轮次（可选）
    """
    timestamp: float
    level: LogLevel
    category: LogCategory
    message: str
    metrics: Dict[str, Any] = field(default_factory=dict)
    tags: Dict[str, str] = field(default_factory=dict)
    step: Optional[int] = None
    epoch: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为可序列化的字典"""
        return {
            "timestamp": self.timestamp,
            "datetime": datetime.fromtimestamp(self.timestamp).isoformat(),
            "level": self.level.name,
            "category": self.category.value,
            "message": self.message,
            "metrics": self._serialize_metrics(self.metrics),
            "tags": self.tags,
            "step": self.step,
            "epoch": self.epoch,
        }
    
    def _serialize_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """序列化指标（处理 numpy 类型）"""
        result = {}
        for key, value in metrics.items():
            if isinstance(value, np.ndarray):
                result[key] = value.tolist()
            elif isinstance(value, (np.floating, np.integer)):
                result[key] = float(value)
            elif isinstance(value, dict):
                result[key] = self._serialize_metrics(value)
            else:
                result[key] = value
        return result


# ============================================================
# 后端抽象基类
# ============================================================

class LogBackend(ABC):
    """日志后端抽象基类"""
    
    @abstractmethod
    def write(self, event: LogEvent) -> None:
        """写入日志事件"""
        ...
    
    @abstractmethod
    def flush(self) -> None:
        """刷新缓冲区"""
        ...
    
    @abstractmethod
    def close(self) -> None:
        """关闭后端"""
        ...


# ============================================================
# Console 后端
# ============================================================

class ConsoleBackend(LogBackend):
    """控制台日志后端
    
    支持彩色输出和格式化。
    """
    
    # ANSI 颜色码
    COLORS = {
        LogLevel.DEBUG: "\033[36m",      # Cyan
        LogLevel.INFO: "\033[32m",       # Green
        LogLevel.WARNING: "\033[33m",    # Yellow
        LogLevel.ERROR: "\033[31m",      # Red
        LogLevel.CRITICAL: "\033[35m",   # Magenta
    }
    RESET = "\033[0m"
    BOLD = "\033[1m"
    
    def __init__(self, colorize: bool = True, min_level: LogLevel = LogLevel.INFO):
        self.colorize = colorize and sys.stdout.isatty()
        self.min_level = min_level
        self._lock = threading.Lock()
    
    def write(self, event: LogEvent) -> None:
        if event.level.value < self.min_level.value:
            return
        
        with self._lock:
            line = self._format_event(event)
            print(line, flush=True)
    
    def _format_event(self, event: LogEvent) -> str:
        """格式化日志事件"""
        ts = datetime.fromtimestamp(event.timestamp).strftime("%H:%M:%S")
        level = event.level.name[:4]
        category = event.category.value[:8].ljust(8)
        
        # 构建消息
        parts = [f"[{ts}]", f"[{level}]", f"[{category}]", event.message]
        
        # 添加指标
        if event.metrics:
            metrics_str = " | ".join(
                f"{k}={self._format_value(v)}" 
                for k, v in event.metrics.items()
            )
            parts.append(f"| {metrics_str}")
        
        # 添加步数
        if event.step is not None:
            parts.insert(3, f"[step={event.step}]")
        
        line = " ".join(parts)
        
        # 添加颜色
        if self.colorize:
            color = self.COLORS.get(event.level, "")
            line = f"{color}{line}{self.RESET}"
        
        return line
    
    def _format_value(self, value: Any) -> str:
        """格式化单个值"""
        if isinstance(value, float):
            if abs(value) < 0.001 or abs(value) > 1000:
                return f"{value:.2e}"
            return f"{value:.4f}"
        elif isinstance(value, int):
            return f"{value:,}"
        elif isinstance(value, (list, np.ndarray)):
            return f"[{len(value)} items]"
        return str(value)
    
    def flush(self) -> None:
        sys.stdout.flush()
    
    def close(self) -> None:
        self.flush()


# ============================================================
# TensorBoard 后端
# ============================================================

class TensorBoardBackend(LogBackend):
    """TensorBoard 日志后端"""
    
    def __init__(self, log_dir: str):
        try:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir=log_dir)
            self.enabled = True
        except ImportError:
            logging.warning("TensorBoard not available, disabling backend")
            self.enabled = False
            self.writer = None
    
    def write(self, event: LogEvent) -> None:
        if not self.enabled or not event.metrics:
            return
        
        step = event.step or 0
        prefix = f"{event.category.value}/"
        
        for key, value in event.metrics.items():
            if isinstance(value, (int, float, np.number)):
                self.writer.add_scalar(f"{prefix}{key}", value, step)
            elif isinstance(value, dict):
                # 嵌套字典
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, (int, float, np.number)):
                        self.writer.add_scalar(f"{prefix}{key}/{sub_key}", sub_value, step)
    
    def flush(self) -> None:
        if self.enabled:
            self.writer.flush()
    
    def close(self) -> None:
        if self.enabled:
            self.writer.close()


# ============================================================
# WandB 后端
# ============================================================

class WandbBackend(LogBackend):
    """Weights & Biases 日志后端"""
    
    def __init__(
        self,
        project: str,
        name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        try:
            import wandb
            self.wandb = wandb
            self.run = wandb.init(
                project=project,
                name=name,
                config=config,
                reinit=True
            )
            self.enabled = True
        except ImportError:
            logging.warning("wandb not available, disabling backend")
            self.enabled = False
    
    def write(self, event: LogEvent) -> None:
        if not self.enabled or not event.metrics:
            return
        
        # 添加前缀
        log_dict = {
            f"{event.category.value}/{key}": value
            for key, value in event.metrics.items()
            if isinstance(value, (int, float, np.number))
        }
        
        if log_dict:
            self.wandb.log(log_dict, step=event.step)
    
    def flush(self) -> None:
        pass  # wandb 自动刷新
    
    def close(self) -> None:
        if self.enabled:
            self.wandb.finish()


# ============================================================
# JSON File 后端
# ============================================================

class JSONFileBackend(LogBackend):
    """JSON 文件日志后端
    
    每行一个 JSON 对象（JSONL 格式）。
    """
    
    def __init__(self, file_path: str, max_buffer_size: int = 100):
        self.file_path = Path(file_path)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.max_buffer_size = max_buffer_size
        self._buffer: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        self._file = open(self.file_path, "a", encoding="utf-8")
    
    def write(self, event: LogEvent) -> None:
        with self._lock:
            self._buffer.append(event.to_dict())
            
            if len(self._buffer) >= self.max_buffer_size:
                self._flush_buffer()
    
    def _flush_buffer(self) -> None:
        """刷新缓冲区到文件"""
        for item in self._buffer:
            self._file.write(json.dumps(item, ensure_ascii=False) + "\n")
        self._file.flush()
        self._buffer.clear()
    
    def flush(self) -> None:
        with self._lock:
            self._flush_buffer()
    
    def close(self) -> None:
        self.flush()
        self._file.close()


# ============================================================
# 结构化日志器
# ============================================================

class StructuredLogger:
    """结构化日志器
    
    统一管理多个日志后端，提供便捷的日志记录接口。
    
    Usage:
        >>> logger = StructuredLogger(["console", "tensorboard"])
        >>> logger.info("training", "Starting training", metrics={"lr": 0.001})
        >>> logger.log_metrics("selfplay", {"games": 100, "win_rate": 0.65}, step=1000)
    """
    
    def __init__(
        self,
        backends: Optional[List[str]] = None,
        log_dir: str = "./logs",
        project_name: str = "rl-framework",
        run_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """初始化日志器
        
        Args:
            backends: 后端列表 ["console", "tensorboard", "wandb", "file"]
            log_dir: 日志目录
            project_name: 项目名称（用于 wandb）
            run_name: 运行名称
            config: 配置字典
        """
        self.backends: List[LogBackend] = []
        self._global_step = 0
        self._epoch = 0
        
        backends = backends or ["console"]
        
        # 生成运行名称
        if run_name is None:
            run_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        run_dir = os.path.join(log_dir, run_name)
        
        # 初始化后端
        for backend_name in backends:
            backend_name = backend_name.lower()
            
            if backend_name == "console":
                self.backends.append(ConsoleBackend())
            elif backend_name == "tensorboard":
                self.backends.append(TensorBoardBackend(run_dir))
            elif backend_name == "wandb":
                self.backends.append(WandbBackend(
                    project=project_name,
                    name=run_name,
                    config=config
                ))
            elif backend_name == "file":
                self.backends.append(JSONFileBackend(
                    os.path.join(run_dir, "events.jsonl")
                ))
            else:
                logging.warning(f"Unknown backend: {backend_name}")
    
    def set_step(self, step: int) -> None:
        """设置全局步数"""
        self._global_step = step
    
    def set_epoch(self, epoch: int) -> None:
        """设置当前轮次"""
        self._epoch = epoch
    
    def log(
        self,
        category: Union[str, LogCategory],
        message: str,
        level: LogLevel = LogLevel.INFO,
        metrics: Optional[Dict[str, Any]] = None,
        tags: Optional[Dict[str, str]] = None,
        step: Optional[int] = None
    ) -> None:
        """记录日志事件
        
        Args:
            category: 日志分类
            message: 日志消息
            level: 日志级别
            metrics: 数值指标
            tags: 标签
            step: 步数（默认使用全局步数）
        """
        if isinstance(category, str):
            try:
                category = LogCategory(category)
            except ValueError:
                category = LogCategory.SYSTEM
        
        event = LogEvent(
            timestamp=time.time(),
            level=level,
            category=category,
            message=message,
            metrics=metrics or {},
            tags=tags or {},
            step=step if step is not None else self._global_step,
            epoch=self._epoch,
        )
        
        for backend in self.backends:
            try:
                backend.write(event)
            except Exception as e:
                logging.error(f"Backend write failed: {e}")
    
    # === 便捷方法 ===
    
    def debug(self, category: str, message: str, **kwargs) -> None:
        self.log(category, message, LogLevel.DEBUG, **kwargs)
    
    def info(self, category: str, message: str, **kwargs) -> None:
        self.log(category, message, LogLevel.INFO, **kwargs)
    
    def warning(self, category: str, message: str, **kwargs) -> None:
        self.log(category, message, LogLevel.WARNING, **kwargs)
    
    def error(self, category: str, message: str, **kwargs) -> None:
        self.log(category, message, LogLevel.ERROR, **kwargs)
    
    def log_metrics(
        self,
        category: str,
        metrics: Dict[str, Any],
        step: Optional[int] = None,
        message: str = ""
    ) -> None:
        """记录指标（简化接口）"""
        self.log(category, message, metrics=metrics, step=step)
    
    def log_training(
        self,
        loss: float,
        metrics: Optional[Dict[str, float]] = None,
        step: Optional[int] = None
    ) -> None:
        """记录训练指标"""
        all_metrics = {"loss": loss}
        if metrics:
            all_metrics.update(metrics)
        self.log("training", f"Loss: {loss:.4f}", metrics=all_metrics, step=step)
    
    def log_selfplay(
        self,
        games: int,
        metrics: Optional[Dict[str, Any]] = None,
        step: Optional[int] = None
    ) -> None:
        """记录自玩统计"""
        all_metrics = {"games": games}
        if metrics:
            all_metrics.update(metrics)
        self.log("selfplay", f"Completed {games} games", metrics=all_metrics, step=step)
    
    def log_evaluation(
        self,
        win_rate: float,
        metrics: Optional[Dict[str, Any]] = None,
        step: Optional[int] = None
    ) -> None:
        """记录评估结果"""
        all_metrics = {"win_rate": win_rate}
        if metrics:
            all_metrics.update(metrics)
        self.log("evaluation", f"Win rate: {win_rate:.1%}", metrics=all_metrics, step=step)
    
    def flush(self) -> None:
        """刷新所有后端"""
        for backend in self.backends:
            try:
                backend.flush()
            except Exception as e:
                logging.error(f"Backend flush failed: {e}")
    
    def close(self) -> None:
        """关闭所有后端"""
        for backend in self.backends:
            try:
                backend.close()
            except Exception as e:
                logging.error(f"Backend close failed: {e}")
    
    def __enter__(self) -> "StructuredLogger":
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()


# ============================================================
# 全局日志器实例
# ============================================================

_global_logger: Optional[StructuredLogger] = None


def get_logger() -> StructuredLogger:
    """获取全局日志器"""
    global _global_logger
    if _global_logger is None:
        _global_logger = StructuredLogger(["console"])
    return _global_logger


def set_logger(logger: StructuredLogger) -> None:
    """设置全局日志器"""
    global _global_logger
    _global_logger = logger


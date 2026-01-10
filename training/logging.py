"""
训练日志和监控模块
支持 TensorBoard 和控制台输出
"""

from __future__ import annotations
from typing import Dict, Any, Optional
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime
import json
import logging
import sys
import time
from collections import deque

from tensorboard.summary import Writer
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# 指标追踪器
# ============================================================================

class MetricTracker:
    """
    指标追踪器
    支持移动平均和统计计算
    """
    
    def __init__(self, window_size: int = 100):
        """
        Args:
            window_size: 移动平均窗口大小
        """
        self.window_size = window_size
        self.values: Dict[str, deque] = {}
        self.total_values: Dict[str, list] = {}
    
    def update(self, name: str, value: float):
        """更新指标"""
        if name not in self.values:
            self.values[name] = deque(maxlen=self.window_size)
            self.total_values[name] = []
        
        self.values[name].append(value)
        self.total_values[name].append(value)
    
    def update_dict(self, metrics: Dict[str, float]):
        """批量更新指标"""
        for name, value in metrics.items():
            self.update(name, value)
    
    def get_mean(self, name: str) -> float:
        """获取移动平均"""
        if name not in self.values or len(self.values[name]) == 0:
            return 0.0
        return np.mean(list(self.values[name]))
    
    def get_std(self, name: str) -> float:
        """获取移动标准差"""
        if name not in self.values or len(self.values[name]) < 2:
            return 0.0
        return np.std(list(self.values[name]))
    
    def get_all_means(self) -> Dict[str, float]:
        """获取所有指标的移动平均"""
        return {name: self.get_mean(name) for name in self.values}
    
    def get_total_mean(self, name: str) -> float:
        """获取全局平均"""
        if name not in self.total_values or len(self.total_values[name]) == 0:
            return 0.0
        return np.mean(self.total_values[name])
    
    def reset(self, name: Optional[str] = None):
        """重置指标"""
        if name is None:
            self.values.clear()
            self.total_values.clear()
        else:
            if name in self.values:
                self.values[name].clear()
            if name in self.total_values:
                self.total_values[name].clear()


# ============================================================================
# TensorBoard 日志器
# ============================================================================

class TrainingLogger:
    """
    训练日志器
    
    功能:
    - TensorBoard 日志
    - 控制台进度显示
    - JSON 日志保存
    """
    
    def __init__(
        self,
        log_dir: str,
        experiment_name: Optional[str] = None,
        console_interval: int = 100,
        tensorboard_interval: int = 10,
    ):
        """
        Args:
            log_dir: 日志目录
            experiment_name: 实验名称
            console_interval: 控制台输出间隔 (步数)
            tensorboard_interval: TensorBoard 日志间隔
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 实验名称
        if experiment_name is None:
            experiment_name = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_name = experiment_name
        
        # TensorBoard
        tb_dir = self.log_dir / "tensorboard" / experiment_name
        self.writer = Writer(str(tb_dir))
        
        # 间隔
        self.console_interval = console_interval
        self.tensorboard_interval = tensorboard_interval
        
        # 指标追踪器
        self.metrics = MetricTracker()
        
        # 时间追踪
        self.start_time = time.time()
        self.step_times = deque(maxlen=100)
        self.last_step_time = self.start_time
        
        # JSON 日志
        self.json_log_path = self.log_dir / f"{experiment_name}_log.jsonl"
        
        logger.info(f"日志目录: {self.log_dir}")
        logger.info(f"TensorBoard: tensorboard --logdir {tb_dir}")
    
    def log_training(
        self,
        step: int,
        metrics: Dict[str, float],
        extra_info: Optional[Dict[str, Any]] = None,
    ):
        """
        记录训练指标
        
        Args:
            step: 当前步数
            metrics: 指标字典
            extra_info: 额外信息
        """
        # 更新追踪器
        self.metrics.update_dict(metrics)
        
        # 记录步骤时间
        current_time = time.time()
        step_time = current_time - self.last_step_time
        self.step_times.append(step_time)
        self.last_step_time = current_time
        
        # TensorBoard 日志
        if step % self.tensorboard_interval == 0:
            for name, value in metrics.items():
                self.writer.add_scalar(f"train/{name}", value, step)
            
            # 性能指标
            steps_per_sec = 1.0 / np.mean(list(self.step_times)) if self.step_times else 0
            self.writer.add_scalar("perf/steps_per_sec", steps_per_sec, step)
        
        # 控制台输出
        if step % self.console_interval == 0:
            self._print_progress(step, metrics)
        
        # JSON 日志
        self._write_json_log(step, metrics, extra_info)
    
    def log_eval(
        self,
        step: int,
        elo: float,
        win_rate: float,
        games_played: int,
    ):
        """记录评估结果"""
        self.writer.add_scalar("eval/elo", elo, step)
        self.writer.add_scalar("eval/win_rate", win_rate, step)
        self.writer.add_scalar("eval/games_played", games_played, step)
        
        logger.info(f"[Eval] Step {step}: ELO={elo:.1f}, WinRate={win_rate:.2%}, Games={games_played}")
    
    def log_mcts(
        self,
        step: int,
        avg_depth: float,
        avg_simulations: float,
        root_value: float,
    ):
        """记录 MCTS 统计"""
        self.writer.add_scalar("mcts/avg_depth", avg_depth, step)
        self.writer.add_scalar("mcts/avg_simulations", avg_simulations, step)
        self.writer.add_scalar("mcts/root_value", root_value, step)
    
    def log_buffer(
        self,
        step: int,
        buffer_size: int,
        num_trajectories: int,
    ):
        """记录 Replay Buffer 状态"""
        self.writer.add_scalar("buffer/size", buffer_size, step)
        self.writer.add_scalar("buffer/trajectories", num_trajectories, step)
    
    def log_histogram(self, name: str, values, step: int):
        """记录直方图"""
        self.writer.add_histogram(name, values, step)
    
    def log_image(self, name: str, image, step: int):
        """记录图像"""
        self.writer.add_image(name, image, step)
    
    def _print_progress(self, step: int, metrics: Dict[str, float]):
        """打印训练进度"""
        elapsed = time.time() - self.start_time
        steps_per_sec = step / elapsed if elapsed > 0 else 0
        
        # 格式化指标
        metric_str = " | ".join([
            f"{name}={self.metrics.get_mean(name):.4f}"
            for name in ['total_loss', 'policy_loss', 'value_loss']
            if name in metrics
        ])
        
        print(
            f"\r[Step {step:>7}] {metric_str} | "
            f"Speed: {steps_per_sec:.1f} steps/s | "
            f"Elapsed: {self._format_time(elapsed)}",
            end="",
            flush=True,
        )
        
        # 每1000步换行
        if step % 1000 == 0:
            print()
    
    def _write_json_log(
        self,
        step: int,
        metrics: Dict[str, float],
        extra_info: Optional[Dict[str, Any]] = None,
    ):
        """写入 JSON 日志"""
        log_entry = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "metrics": metrics,
        }
        
        if extra_info:
            log_entry["extra"] = extra_info
        
        with open(self.json_log_path, "a") as f:
            f.write(json.dumps(log_entry) + "\n")
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """格式化时间"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    
    def close(self):
        """关闭日志器"""
        self.writer.close()
        print()  # 换行
        logger.info(f"训练完成. 总时间: {self._format_time(time.time() - self.start_time)}")


# ============================================================================
# 进度条
# ============================================================================

class ProgressBar:
    """简单的进度条"""
    
    def __init__(self, total: int, desc: str = "", width: int = 50):
        self.total = total
        self.desc = desc
        self.width = width
        self.current = 0
        self.start_time = time.time()
    
    def update(self, n: int = 1):
        """更新进度"""
        self.current += n
        self._display()
    
    def set(self, n: int):
        """设置当前进度"""
        self.current = n
        self._display()
    
    def _display(self):
        """显示进度条"""
        progress = self.current / self.total
        filled = int(self.width * progress)
        bar = "█" * filled + "░" * (self.width - filled)
        
        elapsed = time.time() - self.start_time
        eta = elapsed / progress * (1 - progress) if progress > 0 else 0
        
        print(
            f"\r{self.desc}: |{bar}| {self.current}/{self.total} "
            f"[{self._format_time(elapsed)}<{self._format_time(eta)}]",
            end="",
            flush=True,
        )
        
        if self.current >= self.total:
            print()
    
    @staticmethod
    def _format_time(seconds: float) -> str:
        """格式化时间"""
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"


# ============================================================================
# 设置日志
# ============================================================================

def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
):
    """
    设置全局日志
    
    Args:
        log_level: 日志级别
        log_file: 日志文件路径 (可选)
    """
    # 格式
    formatter = logging.Formatter(
        "[%(asctime)s] %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    
    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    # 根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))
    root_logger.addHandler(console_handler)
    
    # 文件处理器
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)
    
    # 减少第三方库日志
    logging.getLogger("jax").setLevel(logging.WARNING)
    logging.getLogger("absl").setLevel(logging.WARNING)


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    import tempfile
    import numpy as np
    
    print("日志模块测试")
    print("=" * 50)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建日志器
        train_logger = TrainingLogger(
            log_dir=tmpdir,
            experiment_name="test_run",
            console_interval=10,
        )
        
        # 模拟训练
        for step in range(1, 51):
            metrics = {
                "total_loss": np.random.uniform(0.5, 1.5),
                "policy_loss": np.random.uniform(0.2, 0.8),
                "value_loss": np.random.uniform(0.1, 0.5),
                "reward_loss": np.random.uniform(0.05, 0.2),
                "grad_norm": np.random.uniform(0.1, 2.0),
            }
            
            train_logger.log_training(step, metrics)
            
            # 模拟评估
            if step % 25 == 0:
                train_logger.log_eval(
                    step=step,
                    elo=1500 + step * 2,
                    win_rate=0.5 + step * 0.005,
                    games_played=100,
                )
        
        train_logger.close()
        
        print(f"\n日志已保存到: {tmpdir}")
        print("测试通过!")

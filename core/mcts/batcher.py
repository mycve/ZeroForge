"""
LeafBatcher - GPU 批量叶子推理

收集多个 env 线程提交的叶子节点请求，批量推理后返回结果。

设计原则:
- 线程安全：支持多线程并发提交
- 高效批处理：达到数量或超时触发推理
- 低延迟：使用条件变量实现高效等待

架构:
┌─────────────────────────────────────────────────────────┐
│                    LeafBatcher                           │
│   - 请求队列（线程安全）                                 │
│   - 批量收集逻辑                                         │
│   - GPU 推理执行                                         │
│   - 结果分发                                             │
└─────────────────────────────────────────────────────────┘
                          ↑↓
        ┌─────────────────┼─────────────────┐
        ↓                 ↓                 ↓
   Env线程1           Env线程2          Env线程N
   submit()           submit()          submit()
   wait...            wait...           wait...
"""

from __future__ import annotations
import threading
import time
from dataclasses import dataclass, field
from typing import Tuple, List, Optional, Callable, Any
from queue import Queue, Empty
import numpy as np
import logging

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from ..config import BatcherConfig

logger = logging.getLogger(__name__)


@dataclass
class LeafRequest:
    """叶子节点推理请求
    
    由 env 线程创建，提交到 batcher 队列。
    
    Attributes:
        observation: 观测数据 [C, H, W]
        legal_mask: 合法动作掩码 [action_space]
        env_id: 环境 ID（用于调试）
        timestamp: 提交时间戳
        event: 完成事件（用于等待结果）
        result: 推理结果 (policy, value)
    """
    observation: np.ndarray
    legal_mask: np.ndarray
    env_id: int = 0
    timestamp: float = field(default_factory=time.time)
    
    # 同步机制
    event: threading.Event = field(default_factory=threading.Event)
    
    # 结果
    result: Optional[Tuple[np.ndarray, float]] = None
    error: Optional[Exception] = None


class LeafBatcher:
    """GPU 批量叶子推理收集器
    
    收集多个线程的推理请求，批量执行后分发结果。
    
    触发推理的条件（满足任一）:
    1. 收集到 batch_size 个请求
    2. 等待超过 timeout_ms 毫秒
    
    Example:
        >>> # 创建 batcher
        >>> batcher = LeafBatcher(network, config)
        >>> batcher.start()
        >>> 
        >>> # 在 env 线程中使用
        >>> def env_thread():
        ...     policy, value = batcher.submit(observation, legal_mask)
        ...     # 使用结果...
        >>> 
        >>> # 停止
        >>> batcher.stop()
    """
    
    def __init__(
        self,
        network: Any,  # nn.Module，避免强制依赖 torch
        config: BatcherConfig,
        network_fn: Optional[Callable] = None,
    ):
        """初始化 batcher
        
        Args:
            network: 神经网络（需要有 forward 或 initial_inference 方法）
            config: 批推理配置
            network_fn: 可选的自定义推理函数
                        签名: (observations, masks) -> (policies, values)
        """
        self.network = network
        self.config = config
        self.network_fn = network_fn
        
        # 请求队列
        self._queue: Queue[LeafRequest] = Queue()
        
        # 控制
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # 统计
        self._total_requests = 0
        self._total_batches = 0
        self._total_inference_time = 0.0
        
        # 设备
        self._device = config.device
        
        logger.info(f"LeafBatcher 初始化: batch_size={config.batch_size}, "
                   f"timeout={config.timeout_ms}ms, device={config.device}")
    
    # ========================================
    # 公共接口
    # ========================================
    
    def submit(
        self,
        observation: np.ndarray,
        legal_mask: np.ndarray,
        env_id: int = 0,
        timeout: Optional[float] = None,
    ) -> Tuple[np.ndarray, float]:
        """提交推理请求并等待结果
        
        此方法会阻塞直到推理完成。
        
        Args:
            observation: 观测数据 [C, H, W]
            legal_mask: 合法动作掩码 [action_space]
            env_id: 环境 ID（用于调试）
            timeout: 等待超时（秒），None 表示无限等待
            
        Returns:
            (policy, value):
                - policy: 策略分布 [action_space]
                - value: 价值估计
                
        Raises:
            RuntimeError: 如果 batcher 未运行
            TimeoutError: 如果等待超时
            Exception: 如果推理过程出错
        """
        if not self._running:
            raise RuntimeError("LeafBatcher 未运行，请先调用 start()")
        
        # 创建请求
        request = LeafRequest(
            observation=observation,
            legal_mask=legal_mask,
            env_id=env_id,
        )
        
        # 提交到队列
        self._queue.put(request)
        
        # 等待结果
        if not request.event.wait(timeout=timeout):
            raise TimeoutError(f"等待推理结果超时 (env_id={env_id})")
        
        # 检查错误
        if request.error is not None:
            raise request.error
        
        return request.result
    
    def start(self) -> None:
        """启动 batcher 线程"""
        if self._running:
            logger.warning("LeafBatcher 已在运行")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("LeafBatcher 已启动")
    
    def stop(self, timeout: float = 5.0) -> None:
        """停止 batcher 线程
        
        Args:
            timeout: 等待线程结束的超时时间
        """
        if not self._running:
            return
        
        self._running = False
        
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            if self._thread.is_alive():
                logger.warning("LeafBatcher 线程未能在超时内结束")
            self._thread = None
        
        logger.info("LeafBatcher 已停止")
    
    # ========================================
    # 内部方法
    # ========================================
    
    def _run_loop(self) -> None:
        """主循环：收集请求 -> 批推理 -> 分发结果"""
        logger.debug("LeafBatcher 主循环开始")
        
        while self._running:
            try:
                # 收集批次
                batch = self._collect_batch()
                
                if not batch:
                    continue
                
                # 执行推理
                start_time = time.time()
                try:
                    results = self._batch_inference(batch)
                except Exception as e:
                    logger.error(f"批推理失败: {e}")
                    # 将错误传递给所有请求
                    for request in batch:
                        request.error = e
                        request.event.set()
                    continue
                
                inference_time = time.time() - start_time
                
                # 分发结果
                self._dispatch_results(batch, results)
                
                # 更新统计
                with self._lock:
                    self._total_batches += 1
                    self._total_requests += len(batch)
                    self._total_inference_time += inference_time
                
                logger.debug(f"批推理完成: batch_size={len(batch)}, "
                           f"time={inference_time*1000:.1f}ms")
                
            except Exception as e:
                logger.error(f"LeafBatcher 主循环异常: {e}")
        
        logger.debug("LeafBatcher 主循环结束")
    
    def _collect_batch(self) -> List[LeafRequest]:
        """收集请求批次
        
        满足以下任一条件触发返回:
        1. 收集到 batch_size 个请求
        2. 等待超过 timeout_ms
        
        Returns:
            请求列表
        """
        batch: List[LeafRequest] = []
        timeout_sec = self.config.timeout_ms / 1000.0
        deadline = time.time() + timeout_sec
        
        while len(batch) < self.config.batch_size:
            remaining = deadline - time.time()
            
            if remaining <= 0:
                # 超时，返回已收集的
                break
            
            try:
                request = self._queue.get(timeout=min(remaining, 0.001))
                batch.append(request)
            except Empty:
                # 队列空，检查是否已有数据
                if batch:
                    break
                # 没有数据，继续等待
                continue
        
        return batch
    
    def _batch_inference(
        self,
        batch: List[LeafRequest],
    ) -> List[Tuple[np.ndarray, float]]:
        """执行批量推理
        
        Args:
            batch: 请求列表
            
        Returns:
            结果列表 [(policy, value), ...]
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch 未安装，无法执行 GPU 推理")
        
        # 堆叠输入
        observations = np.stack([r.observation for r in batch])
        masks = np.stack([r.legal_mask for r in batch])
        
        # 转换为 tensor
        obs_tensor = torch.from_numpy(observations).to(self._device)
        mask_tensor = torch.from_numpy(masks).to(self._device)
        
        # 推理
        with torch.no_grad():
            if self.network_fn is not None:
                # 使用自定义推理函数
                policies, values = self.network_fn(obs_tensor, mask_tensor)
            else:
                # 尝试调用网络的标准方法
                policies, values = self._call_network(obs_tensor, mask_tensor)
        
        # 转换回 numpy
        policies_np = policies.cpu().numpy()
        values_np = values.cpu().numpy().flatten()
        
        # 构建结果列表
        results = []
        for i in range(len(batch)):
            results.append((policies_np[i], float(values_np[i])))
        
        return results
    
    def _call_network(
        self,
        observations: "torch.Tensor",
        masks: "torch.Tensor",
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """调用网络
        
        尝试不同的网络接口。
        
        Args:
            observations: 观测张量 [B, C, H, W]
            masks: 掩码张量 [B, action_space]
            
        Returns:
            (policies, values): 策略和价值张量
        """
        import torch.nn.functional as F
        
        # 解包网络（处理 DDP/compile）
        net = self.network
        while hasattr(net, "module"):
            net = net.module
        while hasattr(net, "_orig_mod"):
            net = net._orig_mod
        
        # 尝试 initial_inference（MuZero 风格）
        if hasattr(net, "initial_inference"):
            hidden, logits, values = net.initial_inference(observations)
            # 应用掩码并 softmax
            masked_logits = logits.masked_fill(~masks.bool(), -1e9)
            policies = F.softmax(masked_logits, dim=-1)
            return policies, values.squeeze(-1)
        
        # 尝试 forward（AlphaZero 风格）
        if hasattr(net, "forward"):
            # 假设返回 (policy_logits, value)
            output = net(observations)
            if isinstance(output, tuple) and len(output) == 2:
                logits, values = output
                masked_logits = logits.masked_fill(~masks.bool(), -1e9)
                policies = F.softmax(masked_logits, dim=-1)
                return policies, values.squeeze(-1)
        
        raise ValueError("网络没有 initial_inference 或兼容的 forward 方法")
    
    def _dispatch_results(
        self,
        batch: List[LeafRequest],
        results: List[Tuple[np.ndarray, float]],
    ) -> None:
        """分发推理结果
        
        Args:
            batch: 请求列表
            results: 结果列表
        """
        for request, result in zip(batch, results):
            request.result = result
            request.event.set()  # 唤醒等待的线程
    
    # ========================================
    # 统计
    # ========================================
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        with self._lock:
            avg_batch_size = (
                self._total_requests / self._total_batches 
                if self._total_batches > 0 else 0
            )
            avg_inference_time = (
                self._total_inference_time / self._total_batches * 1000
                if self._total_batches > 0 else 0
            )
            
            return {
                "total_requests": self._total_requests,
                "total_batches": self._total_batches,
                "avg_batch_size": avg_batch_size,
                "avg_inference_time_ms": avg_inference_time,
                "queue_size": self._queue.qsize(),
                "running": self._running,
            }
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return (
            f"LeafBatcher(running={stats['running']}, "
            f"batches={stats['total_batches']}, "
            f"requests={stats['total_requests']})"
        )


# ============================================================
# MuZero 专用 Batcher
# ============================================================

@dataclass
class MuZeroLeafRequest:
    """MuZero 叶子节点请求
    
    除了观测，还包含 hidden_state 和 action 用于 dynamics 推理。
    """
    # 输入
    hidden_state: np.ndarray  # 父节点隐藏状态
    action: int  # 到达此节点的动作
    env_id: int = 0
    timestamp: float = field(default_factory=time.time)
    
    # 同步
    event: threading.Event = field(default_factory=threading.Event)
    
    # 结果
    result: Optional[Tuple[np.ndarray, np.ndarray, float, float]] = None
    # (next_hidden, policy, value, reward)
    error: Optional[Exception] = None


class MuZeroLeafBatcher:
    """MuZero 专用批推理器
    
    支持 recurrent_inference（dynamics + prediction）。
    """
    
    def __init__(
        self,
        network: Any,
        config: BatcherConfig,
    ):
        self.network = network
        self.config = config
        
        self._queue: Queue[MuZeroLeafRequest] = Queue()
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # 统计
        self._total_requests = 0
        self._total_batches = 0
    
    def submit(
        self,
        hidden_state: np.ndarray,
        action: int,
        env_id: int = 0,
        timeout: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray, float, float]:
        """提交 MuZero 推理请求
        
        Args:
            hidden_state: 父节点隐藏状态
            action: 动作
            env_id: 环境 ID
            timeout: 超时时间
            
        Returns:
            (next_hidden, policy, value, reward)
        """
        if not self._running:
            raise RuntimeError("MuZeroLeafBatcher 未运行")
        
        request = MuZeroLeafRequest(
            hidden_state=hidden_state,
            action=action,
            env_id=env_id,
        )
        
        self._queue.put(request)
        
        if not request.event.wait(timeout=timeout):
            raise TimeoutError(f"等待 MuZero 推理超时 (env_id={env_id})")
        
        if request.error is not None:
            raise request.error
        
        return request.result
    
    def start(self) -> None:
        """启动"""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
    
    def stop(self, timeout: float = 5.0) -> None:
        """停止"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None
    
    def _run_loop(self) -> None:
        """主循环"""
        while self._running:
            batch = self._collect_batch()
            if not batch:
                continue
            
            try:
                results = self._batch_inference(batch)
                self._dispatch_results(batch, results)
                
                with self._lock:
                    self._total_batches += 1
                    self._total_requests += len(batch)
            except Exception as e:
                logger.error(f"MuZero 批推理失败: {e}")
                for request in batch:
                    request.error = e
                    request.event.set()
    
    def _collect_batch(self) -> List[MuZeroLeafRequest]:
        """收集批次"""
        batch = []
        timeout_sec = self.config.timeout_ms / 1000.0
        deadline = time.time() + timeout_sec
        
        while len(batch) < self.config.batch_size:
            remaining = deadline - time.time()
            if remaining <= 0:
                break
            
            try:
                request = self._queue.get(timeout=min(remaining, 0.001))
                batch.append(request)
            except Empty:
                if batch:
                    break
                continue
        
        return batch
    
    def _batch_inference(
        self,
        batch: List[MuZeroLeafRequest],
    ) -> List[Tuple[np.ndarray, np.ndarray, float, float]]:
        """批量推理"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch 未安装")
        
        import torch
        import torch.nn.functional as F
        
        # 堆叠输入
        hidden_states = np.stack([r.hidden_state for r in batch])
        actions = np.array([r.action for r in batch])
        
        hidden_tensor = torch.from_numpy(hidden_states).to(self.config.device)
        action_tensor = torch.from_numpy(actions).long().to(self.config.device)
        
        # 解包网络
        net = self.network
        while hasattr(net, "module"):
            net = net.module
        while hasattr(net, "_orig_mod"):
            net = net._orig_mod
        
        with torch.no_grad():
            next_hidden, reward, logits, value = net.recurrent_inference(
                hidden_tensor, action_tensor
            )
        
        # 转换
        next_hidden_np = next_hidden.cpu().numpy()
        policies_np = F.softmax(logits, dim=-1).cpu().numpy()
        values_np = value.cpu().numpy().flatten()
        rewards_np = reward.cpu().numpy().flatten()
        
        results = []
        for i in range(len(batch)):
            results.append((
                next_hidden_np[i],
                policies_np[i],
                float(values_np[i]),
                float(rewards_np[i]),
            ))
        
        return results
    
    def _dispatch_results(
        self,
        batch: List[MuZeroLeafRequest],
        results: List[Tuple[np.ndarray, np.ndarray, float, float]],
    ) -> None:
        """分发结果"""
        for request, result in zip(batch, results):
            request.result = result
            request.event.set()
    
    def get_stats(self) -> dict:
        """统计信息"""
        with self._lock:
            return {
                "total_requests": self._total_requests,
                "total_batches": self._total_batches,
                "queue_size": self._queue.qsize(),
                "running": self._running,
            }

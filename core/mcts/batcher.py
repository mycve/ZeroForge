"""
LeafBatcher - GPU 批量叶子推理

收集多个 env 线程提交的叶子节点请求，批量推理后返回结果。

提供两种实现：
1. LeafBatcher：基于 Queue 的传统实现（有锁）
2. SlotBatcher：基于预分配槽位的无锁实现（推荐）

============================================================
性能对比
============================================================

| 特性           | LeafBatcher      | SlotBatcher      |
|----------------|------------------|------------------|
| 提交请求       | Queue.put (有锁) | 直接写入 (无锁)  |
| 收集批次       | Queue.get (有锁) | 扫描数组 (无锁)  |
| 对象创建       | 每次创建 Request | 预分配复用       |
| 适用场景       | 动态线程数       | 固定并发数       |
| no-GIL 性能    | 锁竞争严重       | 无锁，高并发     |

============================================================
使用指南
============================================================

1. LeafBatcher（传统方式）:
   >>> batcher = LeafBatcher(network, config)
   >>> batcher.start()
   >>> policy, value = batcher.submit(obs, mask, env_id=0)

2. SlotBatcher（推荐，无锁高性能）:
   >>> batcher = SlotBatcher(network, config, num_slots=64)
   >>> batcher.start()
   >>> batcher.register_slot(slot_id)  # 线程开始时注册
   >>> policy, value = batcher.submit(slot_id, obs, mask)
   >>> batcher.unregister_slot(slot_id)  # 线程结束时注销

============================================================
架构图
============================================================

【LeafBatcher - 有锁队列】
┌─────────────────────────────────────────────────────────┐
│                    LeafBatcher                           │
│   - Queue（线程安全，有锁）                              │
│   - 对象池（线程本地，无锁）                             │
└─────────────────────────────────────────────────────────┘
                          ↑↓
        ┌─────────────────┼─────────────────┐
        ↓                 ↓                 ↓
   Env线程1           Env线程2          Env线程N
   submit()           submit()          submit()

【SlotBatcher - 无锁槽位】
┌─────────────────────────────────────────────────────────┐
│                    SlotBatcher                           │
│   - 预分配槽位数组（无锁）                               │
│   - 原子状态标记                                         │
│   - 活跃槽位跟踪                                         │
└─────────────────────────────────────────────────────────┘
                          ↑↓
        ┌─────────────────┼─────────────────┐
        ↓                 ↓                 ↓
    Slot[0]           Slot[1]          Slot[N-1]
   线程直接写入      线程直接写入      线程直接写入
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


class LeafRequest:
    """叶子节点推理请求（优化版，使用 __slots__ 减少内存开销）
    
    由 env 线程创建，提交到 batcher 队列。
    
    Attributes:
        observation: 观测数据 [C, H, W]
        legal_mask: 合法动作掩码 [action_space]
        env_id: 环境 ID（用于调试）
        event: 完成事件（用于等待结果）
        result: 推理结果 (policy, value)
    """
    __slots__ = ('observation', 'legal_mask', 'env_id', 'event', 'result', 'error')
    
    def __init__(
        self,
        observation: np.ndarray = None,
        legal_mask: np.ndarray = None,
        env_id: int = 0,
        event: threading.Event = None,
    ):
        self.observation = observation
        self.legal_mask = legal_mask
        self.env_id = env_id
        self.event = event if event is not None else threading.Event()
        self.result: Optional[Tuple[np.ndarray, float]] = None
        self.error: Optional[Exception] = None
    
    def reset(self, observation: np.ndarray, legal_mask: np.ndarray, env_id: int = 0):
        """重置请求以便复用"""
        self.observation = observation
        self.legal_mask = legal_mask
        self.env_id = env_id
        self.result = None
        self.error = None
        self.event.clear()


class LeafRequestPool:
    """LeafRequest 对象池（优化版：线程本地存储，无锁竞争）"""
    
    def __init__(self, initial_size: int = 512):
        # 【优化】使用线程本地存储，每个线程有自己的池，避免锁竞争
        self._local = threading.local()
        self._initial_size = initial_size
    
    def _get_local_pool(self) -> List[LeafRequest]:
        """获取当前线程的本地池"""
        if not hasattr(self._local, 'pool'):
            # 每个线程首次访问时创建本地池
            self._local.pool = [LeafRequest() for _ in range(32)]
        return self._local.pool
    
    def acquire(self, observation: np.ndarray, legal_mask: np.ndarray, env_id: int = 0) -> LeafRequest:
        """获取一个请求对象（无锁）"""
        pool = self._get_local_pool()
        if pool:
            req = pool.pop()
            req.reset(observation, legal_mask, env_id)
            return req
        # 池空了，创建新的
        return LeafRequest(observation, legal_mask, env_id)
    
    def release(self, request: LeafRequest) -> None:
        """归还请求对象（无锁）"""
        pool = self._get_local_pool()
        # 限制每个线程的池大小
        if len(pool) < 64:
            pool.append(request)


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
        
        # 【优化】对象池，避免频繁创建 LeafRequest 和 threading.Event
        self._request_pool = LeafRequestPool(initial_size=config.batch_size * 4)
        
        # 【优化】预分配批量推理数组（延迟初始化，第一次推理时确定形状）
        self._obs_buffer: Optional[np.ndarray] = None
        self._mask_buffer: Optional[np.ndarray] = None
        self._obs_shape: Optional[Tuple[int, ...]] = None
        self._mask_shape: Optional[Tuple[int, ...]] = None
        
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
        
        # 【优化】从对象池获取请求，避免创建新 Event
        request = self._request_pool.acquire(observation, legal_mask, env_id)
        
        # 提交到队列
        self._queue.put(request)
        
        # 等待结果
        if not request.event.wait(timeout=timeout):
            self._request_pool.release(request)  # 超时也要归还
            raise TimeoutError(f"等待推理结果超时 (env_id={env_id})")
        
        # 检查错误
        if request.error is not None:
            error = request.error
            self._request_pool.release(request)
            raise error
        
        result = request.result
        # 【优化】归还请求对象到池
        self._request_pool.release(request)
        
        return result
    
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
        
        优化策略：
        1. 先一次性取出队列中所有可用请求（不等待）
        2. 如果不够 batch_size，等待更多请求到来
        3. 达到 batch_size 或超时后返回
        
        Returns:
            请求列表
        """
        batch: List[LeafRequest] = []
        timeout_sec = self.config.timeout_ms / 1000.0
        deadline = time.time() + timeout_sec
        
        # 第一步：立即取出队列中所有可用请求（非阻塞）
        while len(batch) < self.config.batch_size:
            try:
                request = self._queue.get_nowait()
                batch.append(request)
            except Empty:
                break
        
        # 如果已经收集够了，直接返回
        if len(batch) >= self.config.batch_size:
            return batch
        
        # 第二步：如果还不够，等待更多请求
        while len(batch) < self.config.batch_size:
            remaining = deadline - time.time()
            
            if remaining <= 0:
                # 超时，返回已收集的（即使不够 batch_size）
                break
            
            try:
                # 等待下一个请求，但不要等太久
                request = self._queue.get(timeout=min(remaining, 0.005))
                batch.append(request)
                
                # 取到一个后，再尝试非阻塞取更多
                while len(batch) < self.config.batch_size:
                    try:
                        request = self._queue.get_nowait()
                        batch.append(request)
                    except Empty:
                        break
                        
            except Empty:
                # 队列空，如果已有数据就返回（不再等待）
                if batch:
                    break
                # 完全没数据，继续等待
                continue
        
        return batch
    
    def _batch_inference(
        self,
        batch: List[LeafRequest],
    ) -> List[Tuple[np.ndarray, float]]:
        """执行批量推理（优化版：使用预分配缓冲区）
        
        Args:
            batch: 请求列表
            
        Returns:
            结果列表 [(policy, value), ...]
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch 未安装，无法执行 GPU 推理")
        
        batch_size = len(batch)
        
        # 【优化】延迟初始化预分配缓冲区
        if self._obs_buffer is None:
            self._obs_shape = batch[0].observation.shape
            self._mask_shape = batch[0].legal_mask.shape
            # 预分配最大 batch_size 的缓冲区
            max_batch = self.config.batch_size
            self._obs_buffer = np.empty((max_batch,) + self._obs_shape, dtype=np.float32)
            self._mask_buffer = np.empty((max_batch,) + self._mask_shape, dtype=np.float32)
        
        # 【优化】直接拷贝到预分配缓冲区，避免 np.stack 创建新数组
        for i, req in enumerate(batch):
            self._obs_buffer[i] = req.observation
            self._mask_buffer[i] = req.legal_mask
        
        # 使用缓冲区的 view（只取当前 batch_size 部分）
        observations = self._obs_buffer[:batch_size]
        masks = self._mask_buffer[:batch_size]
        
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
        results = [(policies_np[i], float(values_np[i])) for i in range(batch_size)]
        
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


# ============================================================
# 无锁预分配槽位 Batcher（高性能版）
# ============================================================

class SlotBatcher:
    """无锁预分配槽位 Batcher
    
    使用预分配槽位替代队列，完全无锁设计。
    适用于已知并发数量的场景。
    
    设计特点:
    - 预分配固定数量槽位，每个线程使用固定槽位
    - 使用原子标记替代锁
    - 支持动态活跃线程数（游戏结束时减少）
    - 超时机制避免等待已结束的线程
    
    架构:
    ┌─────────────────────────────────────────────────────────┐
    │                    SlotBatcher                           │
    │   - 预分配槽位数组（无锁）                               │
    │   - 原子状态标记                                         │
    │   - GPU 推理执行                                         │
    └─────────────────────────────────────────────────────────┘
                          ↑↓
        ┌─────────────────┼─────────────────┐
        ↓                 ↓                 ↓
    Slot[0]           Slot[1]          Slot[N-1]
    线程0直接写        线程1直接写       线程N-1直接写
    
    Example:
        >>> batcher = SlotBatcher(network, config, num_slots=64)
        >>> batcher.start()
        >>> 
        >>> # 在 env 线程中使用（需要传入槽位 ID）
        >>> def env_thread(slot_id):
        ...     policy, value = batcher.submit(slot_id, observation, legal_mask)
        >>> 
        >>> batcher.stop()
    """
    
    # 槽位状态常量
    SLOT_EMPTY = 0      # 空闲
    SLOT_READY = 1      # 请求已提交，等待处理
    SLOT_DONE = 2       # 结果已就绪
    
    def __init__(
        self,
        network: Any,
        config: BatcherConfig,
        num_slots: int,
        network_fn: Optional[Callable] = None,
    ):
        """初始化 SlotBatcher
        
        Args:
            network: 神经网络
            config: 批推理配置
            num_slots: 槽位数量（= 最大并发数）
            network_fn: 可选的自定义推理函数
        """
        self.network = network
        self.config = config
        self.num_slots = num_slots
        self.network_fn = network_fn
        self._device = config.device
        
        # 预分配槽位数组
        self._slot_states = [self.SLOT_EMPTY] * num_slots  # 状态标记（原子读写）
        self._slot_obs: List[Optional[np.ndarray]] = [None] * num_slots
        self._slot_mask: List[Optional[np.ndarray]] = [None] * num_slots
        self._slot_results: List[Optional[Tuple[np.ndarray, float]]] = [None] * num_slots
        self._slot_events = [threading.Event() for _ in range(num_slots)]
        
        # 活跃槽位跟踪
        self._active_slots = set(range(num_slots))  # 当前活跃的槽位
        self._active_lock = threading.Lock()  # 仅用于注册/注销
        
        # 预分配批量推理缓冲区
        self._obs_buffer: Optional[np.ndarray] = None
        self._mask_buffer: Optional[np.ndarray] = None
        self._obs_shape: Optional[Tuple[int, ...]] = None
        self._mask_shape: Optional[Tuple[int, ...]] = None
        
        # 控制
        self._running = False
        self._thread: Optional[threading.Thread] = None
        
        # 统计（无锁，允许轻微不准确）
        self._total_requests = 0
        self._total_batches = 0
        
        logger.info(f"SlotBatcher 初始化: num_slots={num_slots}, "
                   f"batch_size={config.batch_size}, device={config.device}")
    
    def register_slot(self, slot_id: int) -> None:
        """注册槽位为活跃状态（线程开始时调用）"""
        with self._active_lock:
            self._active_slots.add(slot_id)
    
    def unregister_slot(self, slot_id: int) -> None:
        """注销槽位（线程结束时调用）"""
        with self._active_lock:
            self._active_slots.discard(slot_id)
    
    def submit(
        self,
        slot_id: int,
        observation: np.ndarray,
        legal_mask: np.ndarray,
        timeout: Optional[float] = None,
    ) -> Tuple[np.ndarray, float]:
        """提交推理请求（无锁写入）
        
        Args:
            slot_id: 槽位 ID（每个线程使用固定槽位）
            observation: 观测数据
            legal_mask: 合法动作掩码
            timeout: 等待超时
            
        Returns:
            (policy, value)
        """
        if not self._running:
            raise RuntimeError("SlotBatcher 未运行")
        
        # 直接写入预分配槽位（无锁）
        self._slot_obs[slot_id] = observation
        self._slot_mask[slot_id] = legal_mask
        self._slot_events[slot_id].clear()
        self._slot_states[slot_id] = self.SLOT_READY  # 原子写入，标记就绪
        
        # 等待结果
        if not self._slot_events[slot_id].wait(timeout=timeout):
            self._slot_states[slot_id] = self.SLOT_EMPTY  # 超时，重置
            raise TimeoutError(f"槽位 {slot_id} 等待超时")
        
        # 获取结果
        result = self._slot_results[slot_id]
        self._slot_states[slot_id] = self.SLOT_EMPTY  # 重置为空闲
        
        return result
    
    def start(self) -> None:
        """启动 batcher 线程"""
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("SlotBatcher 已启动")
    
    def stop(self, timeout: float = 5.0) -> None:
        """停止 batcher"""
        self._running = False
        if self._thread:
            self._thread.join(timeout=timeout)
            self._thread = None
        logger.info("SlotBatcher 已停止")
    
    def _run_loop(self) -> None:
        """主循环：扫描槽位 -> 批推理 -> 分发结果"""
        while self._running:
            try:
                # 收集就绪的请求
                batch_slots, batch_obs, batch_mask = self._collect_batch()
                
                if not batch_slots:
                    continue
                
                # 执行推理
                results = self._batch_inference(batch_obs, batch_mask)
                
                # 分发结果
                self._dispatch_results(batch_slots, results)
                
                # 更新统计
                self._total_batches += 1
                self._total_requests += len(batch_slots)
                
            except Exception as e:
                logger.error(f"SlotBatcher 主循环异常: {e}")
    
    def _collect_batch(self) -> Tuple[List[int], List[np.ndarray], List[np.ndarray]]:
        """收集就绪的请求（无锁扫描）
        
        策略：
        1. 扫描所有槽位，收集 READY 状态的请求
        2. 达到 batch_size 或超时后返回
        3. 如果活跃槽位全部就绪，立即返回（不等待）
        """
        batch_slots: List[int] = []
        batch_obs: List[np.ndarray] = []
        batch_mask: List[np.ndarray] = []
        
        timeout_sec = self.config.timeout_ms / 1000.0
        deadline = time.time() + timeout_sec
        
        while len(batch_slots) < self.config.batch_size:
            # 扫描所有槽位（无锁）
            for slot_id in range(self.num_slots):
                if self._slot_states[slot_id] == self.SLOT_READY:
                    batch_slots.append(slot_id)
                    batch_obs.append(self._slot_obs[slot_id])
                    batch_mask.append(self._slot_mask[slot_id])
                    
                    if len(batch_slots) >= self.config.batch_size:
                        break
            
            # 已收集到足够请求
            if len(batch_slots) >= self.config.batch_size:
                break
            
            # 检查是否所有活跃槽位都已就绪（处理剩余环境不足的情况）
            with self._active_lock:
                active_count = len(self._active_slots)
            
            if active_count > 0 and len(batch_slots) >= active_count:
                # 所有活跃槽位都已提交，立即处理
                break
            
            # 超时检查
            if time.time() >= deadline:
                break
            
            # 如果没有就绪请求，短暂休眠避免忙等
            if not batch_slots:
                time.sleep(0.0005)  # 0.5ms
        
        return batch_slots, batch_obs, batch_mask
    
    def _batch_inference(
        self,
        observations: List[np.ndarray],
        masks: List[np.ndarray],
    ) -> List[Tuple[np.ndarray, float]]:
        """执行批量推理"""
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch 未安装")
        
        batch_size = len(observations)
        
        # 延迟初始化缓冲区
        if self._obs_buffer is None:
            self._obs_shape = observations[0].shape
            self._mask_shape = masks[0].shape
            max_batch = self.config.batch_size
            self._obs_buffer = np.empty((max_batch,) + self._obs_shape, dtype=np.float32)
            self._mask_buffer = np.empty((max_batch,) + self._mask_shape, dtype=np.float32)
        
        # 填充缓冲区
        for i, (obs, mask) in enumerate(zip(observations, masks)):
            self._obs_buffer[i] = obs
            self._mask_buffer[i] = mask
        
        # 转换为 tensor
        obs_tensor = torch.from_numpy(self._obs_buffer[:batch_size]).to(self._device)
        mask_tensor = torch.from_numpy(self._mask_buffer[:batch_size]).to(self._device)
        
        # 推理
        with torch.no_grad():
            if self.network_fn is not None:
                policies, values = self.network_fn(obs_tensor, mask_tensor)
            else:
                policies, values = self._call_network(obs_tensor, mask_tensor)
        
        # 转换回 numpy
        policies_np = policies.cpu().numpy()
        values_np = values.cpu().numpy().flatten()
        
        return [(policies_np[i], float(values_np[i])) for i in range(batch_size)]
    
    def _call_network(
        self,
        observations: "torch.Tensor",
        masks: "torch.Tensor",
    ) -> Tuple["torch.Tensor", "torch.Tensor"]:
        """调用网络"""
        import torch.nn.functional as F
        
        net = self.network
        while hasattr(net, "module"):
            net = net.module
        while hasattr(net, "_orig_mod"):
            net = net._orig_mod
        
        if hasattr(net, "initial_inference"):
            hidden, logits, values = net.initial_inference(observations)
            masked_logits = logits.masked_fill(~masks.bool(), -1e9)
            policies = F.softmax(masked_logits, dim=-1)
            return policies, values.squeeze(-1)
        
        if hasattr(net, "forward"):
            output = net(observations)
            if isinstance(output, tuple) and len(output) == 2:
                logits, values = output
                masked_logits = logits.masked_fill(~masks.bool(), -1e9)
                policies = F.softmax(masked_logits, dim=-1)
                return policies, values.squeeze(-1)
        
        raise ValueError("网络没有 initial_inference 或兼容的 forward 方法")
    
    def _dispatch_results(
        self,
        batch_slots: List[int],
        results: List[Tuple[np.ndarray, float]],
    ) -> None:
        """分发结果到对应槽位"""
        for slot_id, result in zip(batch_slots, results):
            self._slot_results[slot_id] = result
            self._slot_states[slot_id] = self.SLOT_DONE
            self._slot_events[slot_id].set()  # 唤醒等待线程
    
    def get_stats(self) -> dict:
        """获取统计信息"""
        with self._active_lock:
            active_count = len(self._active_slots)
        
        ready_count = sum(1 for s in self._slot_states if s == self.SLOT_READY)
        
        return {
            "total_requests": self._total_requests,
            "total_batches": self._total_batches,
            "num_slots": self.num_slots,
            "active_slots": active_count,
            "ready_slots": ready_count,
            "running": self._running,
        }

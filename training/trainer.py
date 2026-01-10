"""
MuZero 训练器模块
支持多 GPU 数据并行训练
"""

from __future__ import annotations
from typing import NamedTuple, Optional, Dict, Any, Callable, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import time
import logging
import jax
import jax.numpy as jnp
import optax
from functools import partial

from networks.muzero import MuZeroNetwork
from networks.heads import scalar_to_support, logits_to_scalar
from training.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer, SampleBatch
from training.checkpoint import CheckpointManager, CheckpointState

logger = logging.getLogger(__name__)


# ============================================================================
# 训练配置
# ============================================================================

@dataclass
class TrainingConfig:
    """训练配置"""
    
    # 基本参数
    seed: int = 42
    num_training_steps: int = 1000000
    
    # 批次和学习率
    batch_size: int = 256
    learning_rate: float = 2e-4
    lr_warmup_steps: int = 1000
    lr_decay_steps: int = 100000
    weight_decay: float = 1e-4
    
    # MuZero 展开
    unroll_steps: int = 5
    td_steps: int = 10
    
    # 损失权重
    policy_loss_weight: float = 1.0
    value_loss_weight: float = 0.25
    reward_loss_weight: float = 1.0
    
    # Replay Buffer
    replay_buffer_size: int = 100000
    min_replay_size: int = 1000
    
    # 日志和检查点
    log_interval: int = 100
    checkpoint_interval: int = 1000
    eval_interval: int = 5000
    
    # 分布式训练
    num_actors: int = 4
    
    # 价值和奖励分布 (0 = 标量)
    value_support_size: int = 0
    reward_support_size: int = 0
    
    # EMA (指数移动平均)
    use_ema: bool = True
    ema_decay: float = 0.999


# ============================================================================
# 训练状态
# ============================================================================

class TrainingState(NamedTuple):
    """训练状态"""
    params: Dict
    opt_state: Any
    step: int
    rng_key: jax.random.PRNGKey
    ema_params: Optional[Dict] = None


# ============================================================================
# 损失函数
# ============================================================================

def compute_loss(
    params: Dict,
    network: MuZeroNetwork,
    batch: SampleBatch,
    config: TrainingConfig,
    deterministic: bool = False,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    计算 MuZero 损失
    
    Args:
        params: 网络参数
        network: MuZero 网络
        batch: 训练批次
        config: 训练配置
        deterministic: 是否为推理模式
        
    Returns:
        (total_loss, loss_dict)
    """
    batch_size = batch.observations.shape[0]
    
    # 初始推理: 观察 -> 隐藏状态 -> 策略/价值
    output = network.apply(params, batch.observations, deterministic=deterministic)
    hidden_state = output.hidden_state
    
    # 初始策略损失
    policy_loss = _cross_entropy_loss(
        output.policy_logits,
        batch.target_policies[:, 0],
        batch.weights
    )
    
    # 初始价值损失
    if config.value_support_size > 0:
        target_support = scalar_to_support(batch.target_values[:, 0], config.value_support_size)
        value_loss = _cross_entropy_loss(output.value, target_support, batch.weights)
    else:
        value_loss = _mse_loss(output.value, batch.target_values[:, 0], batch.weights)
    
    # 奖励损失 (初始步没有奖励)
    reward_loss = jnp.array(0.0)
    
    # 展开 K 步
    gradient_scale = 1.0 / config.unroll_steps
    
    for k in range(1, config.unroll_steps):
        # 动态网络: (隐藏状态, 动作) -> (下一隐藏状态, 奖励)
        actions = batch.actions[:, k - 1]
        next_hidden, reward, policy_logits, value = network.apply(
            params, hidden_state, actions,
            method=network.recurrent_inference,
            deterministic=deterministic
        )
        
        # 缩放梯度 (MuZero 技巧)
        hidden_state = _scale_gradient(next_hidden, 0.5)
        
        # 策略损失
        policy_loss += gradient_scale * _cross_entropy_loss(
            policy_logits,
            batch.target_policies[:, k],
            batch.weights
        )
        
        # 价值损失
        if config.value_support_size > 0:
            target_support = scalar_to_support(batch.target_values[:, k], config.value_support_size)
            value_loss += gradient_scale * _cross_entropy_loss(value, target_support, batch.weights)
        else:
            value_loss += gradient_scale * _mse_loss(value, batch.target_values[:, k], batch.weights)
        
        # 奖励损失
        if config.reward_support_size > 0:
            target_support = scalar_to_support(batch.target_rewards[:, k], config.reward_support_size)
            reward_loss += gradient_scale * _cross_entropy_loss(reward, target_support, batch.weights)
        else:
            reward_loss += gradient_scale * _mse_loss(reward, batch.target_rewards[:, k], batch.weights)
    
    # 总损失
    total_loss = (
        config.policy_loss_weight * policy_loss +
        config.value_loss_weight * value_loss +
        config.reward_loss_weight * reward_loss
    )
    
    loss_dict = {
        'total_loss': total_loss,
        'policy_loss': policy_loss,
        'value_loss': value_loss,
        'reward_loss': reward_loss,
    }
    
    return total_loss, loss_dict


def _cross_entropy_loss(
    logits: jnp.ndarray,
    targets: jnp.ndarray,
    weights: jnp.ndarray,
) -> jnp.ndarray:
    """加权交叉熵损失"""
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    loss = -jnp.sum(targets * log_probs, axis=-1)
    return jnp.mean(loss * weights)


def _mse_loss(
    predictions: jnp.ndarray,
    targets: jnp.ndarray,
    weights: jnp.ndarray,
) -> jnp.ndarray:
    """加权 MSE 损失"""
    loss = jnp.square(predictions - targets)
    return jnp.mean(loss * weights)


def _scale_gradient(x: jnp.ndarray, scale: float) -> jnp.ndarray:
    """缩放梯度 (前向传播不变)"""
    return x * scale + jax.lax.stop_gradient(x) * (1 - scale)


# ============================================================================
# 训练器
# ============================================================================

class MuZeroTrainer:
    """
    MuZero 训练器
    
    支持:
    - 多 GPU 数据并行
    - 混合精度训练 (可选)
    - EMA 参数平均
    - 断点续训
    """
    
    def __init__(
        self,
        network: MuZeroNetwork,
        config: TrainingConfig,
        checkpoint_dir: str,
    ):
        """
        Args:
            network: MuZero 网络
            config: 训练配置
            checkpoint_dir: 检查点目录
        """
        self.network = network
        self.config = config
        
        # 设备
        self.devices = jax.devices()
        self.num_devices = len(self.devices)
        logger.info(f"可用设备: {self.num_devices} ({[d.platform for d in self.devices]})")
        
        # 检查点管理器
        self.checkpoint_manager = CheckpointManager(checkpoint_dir)
        
        # 优化器
        self.optimizer = self._create_optimizer()
        
        # Replay Buffer
        self.replay_buffer = PrioritizedReplayBuffer(
            capacity=config.replay_buffer_size,
            unroll_steps=config.unroll_steps,
            td_steps=config.td_steps,
        )
        
        # 编译训练步骤
        self._compile_train_step()
    
    def _create_optimizer(self) -> optax.GradientTransformation:
        """创建优化器"""
        # 学习率调度
        warmup_fn = optax.linear_schedule(
            init_value=0.0,
            end_value=self.config.learning_rate,
            transition_steps=self.config.lr_warmup_steps,
        )
        
        decay_fn = optax.cosine_decay_schedule(
            init_value=self.config.learning_rate,
            decay_steps=self.config.lr_decay_steps,
            alpha=0.1,
        )
        
        schedule = optax.join_schedules(
            schedules=[warmup_fn, decay_fn],
            boundaries=[self.config.lr_warmup_steps],
        )
        
        # 优化器链
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=schedule, weight_decay=self.config.weight_decay),
        )
        
        return optimizer
    
    def _compile_train_step(self):
        """编译训练步骤 (支持多 GPU)"""
        
        def train_step(
            state: TrainingState,
            batch: SampleBatch,
        ) -> Tuple[TrainingState, Dict[str, jnp.ndarray]]:
            """单步训练"""
            
            def loss_fn(params):
                return compute_loss(params, self.network, batch, self.config)
            
            # 计算梯度
            (loss, loss_dict), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            
            # 多设备梯度平均
            if self.num_devices > 1:
                grads = jax.lax.pmean(grads, axis_name='devices')
                loss_dict = jax.tree_util.tree_map(
                    lambda x: jax.lax.pmean(x, axis_name='devices'),
                    loss_dict
                )
            
            # 更新参数
            updates, new_opt_state = self.optimizer.update(grads, state.opt_state, state.params)
            new_params = optax.apply_updates(state.params, updates)
            
            # EMA 更新
            if self.config.use_ema and state.ema_params is not None:
                new_ema_params = jax.tree_util.tree_map(
                    lambda ema, p: self.config.ema_decay * ema + (1 - self.config.ema_decay) * p,
                    state.ema_params,
                    new_params,
                )
            else:
                new_ema_params = state.ema_params
            
            # 新状态
            new_state = TrainingState(
                params=new_params,
                opt_state=new_opt_state,
                step=state.step + 1,
                rng_key=state.rng_key,
                ema_params=new_ema_params,
            )
            
            # 添加梯度范数到日志
            grad_norm = optax.global_norm(grads)
            loss_dict['grad_norm'] = grad_norm
            
            return new_state, loss_dict
        
        if self.num_devices > 1:
            self._train_step = jax.pmap(train_step, axis_name='devices')
        else:
            self._train_step = jax.jit(train_step)
    
    def init_state(
        self,
        rng_key: jax.random.PRNGKey,
        sample_observation: jnp.ndarray,
    ) -> TrainingState:
        """
        初始化训练状态
        
        Args:
            rng_key: 随机数密钥
            sample_observation: 示例观察 (用于初始化网络)
            
        Returns:
            初始训练状态
        """
        # 尝试恢复检查点
        restored = self.checkpoint_manager.restore()
        
        if restored is not None:
            logger.info(f"从检查点恢复: step={restored.step}")
            return TrainingState(
                params=restored.params,
                opt_state=restored.opt_state,
                step=restored.step,
                rng_key=restored.rng_key,
                ema_params=restored.params if self.config.use_ema else None,
            )
        
        # 初始化网络 (需要初始化所有子网络的参数)
        init_key, rng_key = jax.random.split(rng_key)
        
        # 使用 __call__ 初始化 representation 和 prediction
        params = self.network.init(init_key, sample_observation)
        
        # 额外初始化 dynamics 网络 (通过 recurrent_inference)
        # 创建假的隐藏状态和动作
        hidden_dim = self.network.hidden_dim
        batch_size = sample_observation.shape[0]
        dummy_hidden = jnp.zeros((batch_size, hidden_dim, 10, 9))
        dummy_action = jnp.zeros((batch_size,), dtype=jnp.int32)
        
        init_key2, rng_key = jax.random.split(rng_key)
        dyn_params = self.network.init(
            init_key2, dummy_hidden, dummy_action, 
            method=self.network.recurrent_inference
        )
        
        # 合并参数 (dynamics 参数会在 dyn_params 中)
        params = _merge_params(params, dyn_params)
        
        # 初始化优化器
        opt_state = self.optimizer.init(params)
        
        # EMA 参数
        ema_params = params if self.config.use_ema else None
        
        return TrainingState(
            params=params,
            opt_state=opt_state,
            step=0,
            rng_key=rng_key,
            ema_params=ema_params,
        )
    
    def train_step(
        self,
        state: TrainingState,
        rng_key: jax.random.PRNGKey,
    ) -> Tuple[TrainingState, Dict[str, float]]:
        """
        执行单步训练
        
        Args:
            state: 当前训练状态
            rng_key: 随机数密钥
            
        Returns:
            (new_state, metrics)
        """
        # 检查 buffer 是否有足够数据
        if len(self.replay_buffer) < self.config.min_replay_size:
            return state, {}
        
        # 采样批次
        batch = self.replay_buffer.sample(self.config.batch_size, rng_key)
        
        # 多设备分发
        if self.num_devices > 1:
            batch = _shard_batch(batch, self.num_devices)
            state = _replicate_state(state, self.num_devices)
        
        # 训练步骤
        new_state, loss_dict = self._train_step(state, batch)
        
        # 收集结果
        if self.num_devices > 1:
            new_state = _unreplicate_state(new_state)
            loss_dict = jax.tree_util.tree_map(lambda x: x[0], loss_dict)
        
        # 转换为 Python 标量
        metrics = {k: float(v) for k, v in loss_dict.items()}
        
        return new_state, metrics
    
    def save_checkpoint(
        self,
        state: TrainingState,
        elo_ratings: Optional[Dict[str, float]] = None,
        training_stats: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
    ):
        """保存检查点"""
        self.checkpoint_manager.save(
            step=state.step,
            params=state.params,
            opt_state=state.opt_state,
            rng_key=state.rng_key,
            elo_ratings=elo_ratings,
            training_stats=training_stats,
            is_best=is_best,
        )
    
    def close(self):
        """关闭训练器，等待所有异步操作完成"""
        self.checkpoint_manager.close()
    
    def add_trajectory(self, trajectory):
        """添加轨迹到 replay buffer"""
        self.replay_buffer.add(trajectory)


# ============================================================================
# 辅助函数
# ============================================================================

def _merge_params(params1: dict, params2: dict) -> dict:
    """
    深度合并两个参数字典
    
    用于合并不同方法初始化的参数
    """
    import copy
    result = copy.deepcopy(params1)
    
    def merge_dict(d1, d2):
        for key, value in d2.items():
            if key in d1:
                if isinstance(value, dict) and isinstance(d1[key], dict):
                    merge_dict(d1[key], value)
                # 如果 key 已存在且不是 dict，保留 d1 的值
            else:
                d1[key] = value
    
    merge_dict(result, params2)
    return result


def _shard_batch(batch: SampleBatch, num_devices: int) -> SampleBatch:
    """将批次分片到多个设备"""
    def shard(x):
        return x.reshape(num_devices, -1, *x.shape[1:])
    
    return SampleBatch(
        observations=shard(batch.observations),
        actions=shard(batch.actions),
        target_policies=shard(batch.target_policies),
        target_values=shard(batch.target_values),
        target_rewards=shard(batch.target_rewards),
        weights=shard(batch.weights),
        indices=shard(batch.indices),
    )


def _replicate_state(state: TrainingState, num_devices: int) -> TrainingState:
    """复制状态到多个设备"""
    return jax.tree_util.tree_map(
        lambda x: jnp.broadcast_to(x, (num_devices,) + x.shape),
        state
    )


def _unreplicate_state(state: TrainingState) -> TrainingState:
    """从多设备状态提取单个副本"""
    return jax.tree_util.tree_map(lambda x: x[0], state)


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    import tempfile
    import jax
    import jax.numpy as jnp
    from networks.muzero import MuZeroNetwork
    from training.replay_buffer import Trajectory
    import numpy as np
    
    print("MuZero Trainer 测试")
    print("=" * 50)
    
    # 配置
    config = TrainingConfig(
        batch_size=8,
        unroll_steps=3,
        min_replay_size=10,
    )
    
    # 创建网络
    network = MuZeroNetwork(
        observation_channels=240,
        hidden_dim=64,  # 小网络用于测试
        action_space_size=2086,
        repr_blocks=2,
        dyn_blocks=1,
        pred_blocks=1,
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # 创建训练器
        trainer = MuZeroTrainer(network, config, tmpdir)
        
        # 添加测试轨迹
        for _ in range(20):
            # 交替的 to_plays: 0, 1, 0, 1, ...
            to_plays = np.array([i % 2 for i in range(30)], dtype=np.int32)
            traj = Trajectory(
                observations=np.random.randn(30, 240, 10, 9).astype(np.float32),
                actions=np.random.randint(0, 2086, 30),
                rewards=np.zeros(30, dtype=np.float32),
                policies=np.random.dirichlet(np.ones(2086), 30).astype(np.float32),
                values=np.random.uniform(-1, 1, 30).astype(np.float32),
                to_plays=to_plays,
                game_result=0,
            )
            trainer.add_trajectory(traj)
        
        print(f"Buffer 大小: {len(trainer.replay_buffer)}")
        
        # 初始化状态
        key = jax.random.PRNGKey(42)
        sample_obs = jnp.zeros((1, 240, 10, 9))
        state = trainer.init_state(key, sample_obs)
        
        print(f"初始步数: {state.step}")
        
        # 训练几步
        for i in range(5):
            key, step_key = jax.random.split(key)
            state, metrics = trainer.train_step(state, step_key)
            if metrics:
                print(f"Step {state.step}: loss={metrics.get('total_loss', 0):.4f}")
        
        # 保存检查点
        trainer.save_checkpoint(state)
        
        print("\n测试通过!")

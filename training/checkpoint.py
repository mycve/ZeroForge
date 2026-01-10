"""
检查点管理模块
支持断点续训、模型保存和加载
"""

from __future__ import annotations
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import json
import logging
import shutil
import jax
import jax.numpy as jnp
import orbax.checkpoint as ocp
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================================
# 检查点状态
# ============================================================================

@dataclass
class CheckpointState:
    """检查点状态"""
    step: int
    params: Dict
    opt_state: Any
    rng_key: jax.random.PRNGKey
    elo_ratings: Dict[str, float]
    training_stats: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        """转换为可序列化的字典"""
        return {
            'step': self.step,
            'params': self.params,
            'opt_state': self.opt_state,
            'rng_key': np.array(self.rng_key),
            'elo_ratings': self.elo_ratings,
            'training_stats': self.training_stats,
        }
    
    @classmethod
    def from_dict(cls, d: Dict) -> 'CheckpointState':
        """从字典恢复"""
        return cls(
            step=d['step'],
            params=d['params'],
            opt_state=d['opt_state'],
            rng_key=jnp.array(d['rng_key']),
            elo_ratings=d.get('elo_ratings', {}),
            training_stats=d.get('training_stats', {}),
        )


# ============================================================================
# 检查点管理器
# ============================================================================

class CheckpointManager:
    """
    检查点管理器
    
    功能:
    - 保存和恢复训练状态
    - 管理多个检查点版本
    - 自动清理旧检查点
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        max_to_keep: int = 5,
        keep_period: int = 10000,
    ):
        """
        Args:
            checkpoint_dir: 检查点目录
            max_to_keep: 最多保留的检查点数量
            keep_period: 永久保留的检查点间隔 (步数)
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_to_keep = max_to_keep
        self.keep_period = keep_period
        
        # Orbax 检查点管理器
        self.options = ocp.CheckpointManagerOptions(
            max_to_keep=max_to_keep,
            save_interval_steps=1,
        )
        
        self.manager = ocp.CheckpointManager(
            self.checkpoint_dir,
            options=self.options,
        )
        
        # 元数据文件
        self.metadata_file = self.checkpoint_dir / 'metadata.json'
        self._load_metadata()
    
    def _load_metadata(self):
        """加载元数据"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {
                'latest_step': 0,
                'best_step': 0,
                'best_elo': 1500.0,
                'permanent_checkpoints': [],
            }
    
    def _save_metadata(self):
        """保存元数据"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def save(
        self,
        step: int,
        params: Dict,
        opt_state: Any,
        rng_key: jax.random.PRNGKey,
        elo_ratings: Optional[Dict[str, float]] = None,
        training_stats: Optional[Dict[str, Any]] = None,
        is_best: bool = False,
    ):
        """
        保存检查点
        
        Args:
            step: 当前步数
            params: 模型参数
            opt_state: 优化器状态
            rng_key: 随机数密钥
            elo_ratings: ELO 评分历史
            training_stats: 训练统计信息
            is_best: 是否为最佳模型
        """
        state = CheckpointState(
            step=step,
            params=params,
            opt_state=opt_state,
            rng_key=rng_key,
            elo_ratings=elo_ratings or {},
            training_stats=training_stats or {},
        )
        
        # 使用 Orbax 保存
        save_args = ocp.args.StandardSave(state.to_dict())
        self.manager.save(step, args=save_args)
        
        # 更新元数据
        self.metadata['latest_step'] = step
        
        if is_best:
            self.metadata['best_step'] = step
            if elo_ratings:
                latest_elo = list(elo_ratings.values())[-1] if elo_ratings else 1500.0
                self.metadata['best_elo'] = latest_elo
        
        # 检查是否需要永久保留
        if step % self.keep_period == 0 and step > 0:
            if step not in self.metadata['permanent_checkpoints']:
                self.metadata['permanent_checkpoints'].append(step)
        
        self._save_metadata()
        
        logger.info(f"检查点已保存: step={step}, is_best={is_best}")
    
    def restore(
        self,
        step: Optional[int] = None,
        restore_best: bool = False,
    ) -> Optional[CheckpointState]:
        """
        恢复检查点
        
        Args:
            step: 指定步数，None 表示最新
            restore_best: 是否恢复最佳模型
            
        Returns:
            CheckpointState 或 None (如果没有检查点)
        """
        if restore_best:
            step = self.metadata.get('best_step', 0)
        elif step is None:
            step = self.metadata.get('latest_step', 0)
        
        if step == 0:
            logger.warning("没有可用的检查点")
            return None
        
        try:
            restored = self.manager.restore(step)
            state = CheckpointState.from_dict(restored)
            logger.info(f"检查点已恢复: step={state.step}")
            return state
        except Exception as e:
            logger.error(f"恢复检查点失败: {e}")
            return None
    
    def get_latest_step(self) -> int:
        """获取最新步数"""
        return self.metadata.get('latest_step', 0)
    
    def get_best_step(self) -> int:
        """获取最佳模型步数"""
        return self.metadata.get('best_step', 0)
    
    def list_checkpoints(self) -> list:
        """列出所有检查点"""
        return sorted(self.manager.all_steps())
    
    def cleanup_old_checkpoints(self):
        """清理旧检查点 (保留永久检查点)"""
        all_steps = self.list_checkpoints()
        permanent = set(self.metadata.get('permanent_checkpoints', []))
        
        # 保留: 最新的 max_to_keep 个 + 永久检查点 + 最佳
        best_step = self.metadata.get('best_step', 0)
        keep_steps = set(all_steps[-self.max_to_keep:])
        keep_steps.update(permanent)
        keep_steps.add(best_step)
        
        for step in all_steps:
            if step not in keep_steps:
                try:
                    step_dir = self.checkpoint_dir / str(step)
                    if step_dir.exists():
                        shutil.rmtree(step_dir)
                        logger.info(f"已删除旧检查点: step={step}")
                except Exception as e:
                    logger.warning(f"删除检查点失败 step={step}: {e}")


# ============================================================================
# 简单的参数保存/加载 (用于导出模型)
# ============================================================================

def save_params(params: Dict, path: str):
    """
    保存模型参数 (仅参数，不含优化器状态)
    
    Args:
        params: 模型参数
        path: 保存路径
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpointer = ocp.PyTreeCheckpointer()
    checkpointer.save(path, params)
    
    logger.info(f"参数已保存到: {path}")


def load_params(path: str) -> Dict:
    """
    加载模型参数
    
    Args:
        path: 参数文件路径
        
    Returns:
        模型参数
    """
    path = Path(path)
    
    checkpointer = ocp.PyTreeCheckpointer()
    params = checkpointer.restore(path)
    
    logger.info(f"参数已加载: {path}")
    return params


# ============================================================================
# 测试代码
# ============================================================================

if __name__ == "__main__":
    import tempfile
    import jax
    import jax.numpy as jnp
    
    print("检查点管理器测试")
    print("=" * 50)
    
    # 创建临时目录
    with tempfile.TemporaryDirectory() as tmpdir:
        manager = CheckpointManager(tmpdir, max_to_keep=3)
        
        # 创建模拟参数
        key = jax.random.PRNGKey(42)
        params = {
            'layer1': jax.random.normal(key, (100, 100)),
            'layer2': jax.random.normal(key, (100, 10)),
        }
        opt_state = {'step': 0, 'momentum': None}
        
        # 保存多个检查点
        for step in [100, 200, 300, 400, 500]:
            manager.save(
                step=step,
                params=params,
                opt_state=opt_state,
                rng_key=key,
                elo_ratings={f'model_{step}': 1500 + step * 0.1},
                is_best=(step == 400),
            )
        
        print(f"所有检查点: {manager.list_checkpoints()}")
        print(f"最新步数: {manager.get_latest_step()}")
        print(f"最佳步数: {manager.get_best_step()}")
        
        # 恢复最新
        state = manager.restore()
        print(f"\n恢复最新: step={state.step}")
        
        # 恢复最佳
        state = manager.restore(restore_best=True)
        print(f"恢复最佳: step={state.step}")
        
        # 恢复指定
        state = manager.restore(step=200)
        print(f"恢复指定: step={state.step}")
        
        print("\n测试通过!")

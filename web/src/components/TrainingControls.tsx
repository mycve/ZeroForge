import { useState } from 'react'
import { motion } from 'framer-motion'
import { Play, Pause, Square, RefreshCw, Save } from 'lucide-react'
import { clsx } from 'clsx'
import { useAppStore } from '../store'
import { sendTrainingCommand, saveCheckpoint } from '../utils/api'
import { useToast } from './Toast'

export function TrainingControls() {
  const { trainingStatus } = useAppStore()
  const { running, paused } = trainingStatus
  const [saving, setSaving] = useState(false)
  const toast = useToast()

  const handleStart = async () => {
    try {
      await sendTrainingCommand('start')
      toast.success('训练已启动')
    } catch (e) {
      const msg = e instanceof Error ? e.message : '未知错误'
      toast.error('启动失败', msg)
    }
  }

  const handlePause = async () => {
    try {
      await sendTrainingCommand('pause')
      toast.info('训练已暂停')
    } catch (e) {
      const msg = e instanceof Error ? e.message : '未知错误'
      toast.error('暂停失败', msg)
    }
  }

  const handleResume = async () => {
    try {
      await sendTrainingCommand('resume')
      toast.success('训练已恢复')
    } catch (e) {
      const msg = e instanceof Error ? e.message : '未知错误'
      toast.error('恢复失败', msg)
    }
  }

  const handleStop = async () => {
    try {
      await sendTrainingCommand('stop')
      toast.info('训练已停止')
    } catch (e) {
      const msg = e instanceof Error ? e.message : '未知错误'
      toast.error('停止失败', msg)
    }
  }
  
  const handleSaveCheckpoint = async () => {
    if (saving) return
    setSaving(true)
    try {
      const result = await saveCheckpoint()
      if (result.success) {
        toast.success('检查点已保存', `Epoch ${result.epoch}`)
      } else {
        toast.error('保存失败', result.error)
      }
    } catch (e) {
      const msg = e instanceof Error ? e.message : '未知错误'
      toast.error('保存检查点失败', msg)
    } finally {
      setSaving(false)
    }
  }

  return (
    <div className="bg-bg-card border border-bg-elevated rounded-xl p-6">
      <h3 className="text-lg font-semibold font-display mb-4">训练控制</h3>
      
      {/* 状态指示 */}
      <div className="flex items-center gap-3 mb-6">
        <div
          className={clsx(
            'w-3 h-3 rounded-full',
            running && !paused && 'bg-accent animate-pulse',
            running && paused && 'bg-warning',
            !running && 'bg-gray-500'
          )}
        />
        <span className="text-sm text-gray-400">
          {running && !paused && '训练中...'}
          {running && paused && '已暂停'}
          {!running && '已停止'}
        </span>
      </div>

      {/* 控制按钮 */}
      <div className="grid grid-cols-2 gap-3">
        {!running ? (
          <motion.button
            whileHover={{ scale: 1.02 }}
            whileTap={{ scale: 0.98 }}
            onClick={handleStart}
            className="col-span-2 flex items-center justify-center gap-2 px-4 py-3 rounded-lg bg-accent text-black font-semibold hover:bg-accent-light transition-colors"
          >
            <Play className="w-5 h-5" />
            开始训练
          </motion.button>
        ) : (
          <>
            {paused ? (
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={handleResume}
                className="flex items-center justify-center gap-2 px-4 py-3 rounded-lg bg-accent/10 text-accent border border-accent/30 font-medium hover:bg-accent/20 transition-colors"
              >
                <RefreshCw className="w-4 h-4" />
                恢复
              </motion.button>
            ) : (
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={handlePause}
                className="flex items-center justify-center gap-2 px-4 py-3 rounded-lg bg-warning/10 text-warning border border-warning/30 font-medium hover:bg-warning/20 transition-colors"
              >
                <Pause className="w-4 h-4" />
                暂停
              </motion.button>
            )}
            
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={handleStop}
              className="flex items-center justify-center gap-2 px-4 py-3 rounded-lg bg-error/10 text-error border border-error/30 font-medium hover:bg-error/20 transition-colors"
            >
              <Square className="w-4 h-4" />
              停止
            </motion.button>
            
            {/* 保存检查点按钮 */}
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={handleSaveCheckpoint}
              disabled={saving}
              className="col-span-2 flex items-center justify-center gap-2 px-4 py-3 rounded-lg bg-blue-500/10 text-blue-400 border border-blue-500/30 font-medium hover:bg-blue-500/20 transition-colors disabled:opacity-50"
            >
              {saving ? (
                <RefreshCw className="w-4 h-4 animate-spin" />
              ) : (
                <Save className="w-4 h-4" />
              )}
              保存检查点
            </motion.button>
          </>
        )}
      </div>

      {/* 训练进度 */}
      {running && (
        <div className="mt-6 space-y-3">
          <div className="flex justify-between text-sm">
            <span className="text-gray-400">训练步数</span>
            <span className="text-white font-mono">{trainingStatus.step.toLocaleString()}</span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-gray-400">已完成对局</span>
            <span className="text-white font-mono">{trainingStatus.total_games.toLocaleString()}</span>
          </div>
          <div className="flex justify-between text-sm">
            <span className="text-gray-400">训练时长</span>
            <span className="text-white font-mono">
              {formatDuration(trainingStatus.elapsed_time)}
            </span>
          </div>
        </div>
      )}
    </div>
  )
}

function formatDuration(seconds: number): string {
  if (seconds < 60) return `${Math.floor(seconds)}s`
  if (seconds < 3600) {
    const mins = Math.floor(seconds / 60)
    const secs = Math.floor(seconds % 60)
    return `${mins}m ${secs}s`
  }
  const hours = Math.floor(seconds / 3600)
  const mins = Math.floor((seconds % 3600) / 60)
  return `${hours}h ${mins}m`
}


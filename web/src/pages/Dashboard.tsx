import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { 
  Zap, 
  Gamepad2, 
  TrendingUp, 
  Clock, 
  Cpu, 
  Trophy,
  Activity,
  BarChart3
} from 'lucide-react'
import { StatCard } from '../components/StatCard'
import { LossChart } from '../components/LossChart'
import { TrainingControls } from '../components/TrainingControls'
import { useAppStore } from '../store'
import { useTrainingWebSocket } from '../hooks/useWebSocket'
import { getSystemInfo } from '../utils/api'

export default function Dashboard() {
  const { trainingStatus, systemInfo, setSystemInfo } = useAppStore()
  const [loading, setLoading] = useState(true)
  
  // 连接 WebSocket
  useTrainingWebSocket()

  // 获取系统信息
  useEffect(() => {
    async function fetchSystemInfo() {
      try {
        const info = await getSystemInfo()
        setSystemInfo(info)
      } catch (e) {
        console.error('获取系统信息失败:', e)
      } finally {
        setLoading(false)
      }
    }
    fetchSystemInfo()
  }, [setSystemInfo])

  const containerVariants = {
    hidden: { opacity: 0 },
    visible: {
      opacity: 1,
      transition: { staggerChildren: 0.1 }
    }
  }

  return (
    <div className="p-8">
      {/* 页面标题 */}
      <motion.div 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <h1 className="text-3xl font-bold font-display">训练仪表盘</h1>
        <p className="text-gray-400 mt-2">实时监控训练状态和性能指标</p>
      </motion.div>

      {/* 统计卡片 */}
      <motion.div
        variants={containerVariants}
        initial="hidden"
        animate="visible"
        className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6 mb-8"
      >
        <StatCard
          title="训练步数"
          value={trainingStatus.step}
          subtitle={`${trainingStatus.steps_per_second.toFixed(1)} steps/s`}
          icon={Zap}
          color="accent"
        />
        
        <StatCard
          title="完成对局"
          value={trainingStatus.total_games}
          subtitle={`${trainingStatus.games_per_second.toFixed(2)} games/s`}
          icon={Gamepad2}
          color="purple"
        />
        
        <StatCard
          title="当前损失"
          value={trainingStatus.loss.toFixed(4)}
          subtitle="总损失"
          icon={TrendingUp}
          color="warning"
        />
        
        <StatCard
          title="评估胜率"
          value={trainingStatus.eval_elo > 0 
            ? `${(trainingStatus.eval_win_rate * 100).toFixed(1)}%` 
            : '—'}
          subtitle={trainingStatus.eval_elo > 0 
            ? `ELO: ${trainingStatus.eval_elo.toFixed(0)}` 
            : 'Epoch 2+ 开始评估'}
          icon={Trophy}
          color="accent"
        />
      </motion.div>

      {/* 主内容区域 */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* 损失图表 */}
        <div className="lg:col-span-2">
          <LossChart height={320} />
        </div>

        {/* 训练控制 */}
        <div>
          <TrainingControls />
        </div>
      </div>

      {/* 详细指标 */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="mt-8 grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6"
      >
        {/* 损失详情 */}
        <div className="bg-bg-card border border-bg-elevated rounded-xl p-6">
          <div className="flex items-center gap-3 mb-4">
            <BarChart3 className="w-5 h-5 text-accent" />
            <h3 className="text-lg font-semibold font-display">损失详情</h3>
          </div>
          <div className="space-y-4">
            <MetricRow label="价值损失" value={trainingStatus.value_loss.toFixed(4)} color="text-blue-400" />
            <MetricRow label="策略损失" value={trainingStatus.policy_loss.toFixed(4)} color="text-purple-400" />
            <MetricRow label="奖励损失" value={trainingStatus.reward_loss.toFixed(4)} color="text-orange-400" />
          </div>
        </div>

        {/* 自玩统计 */}
        <div className="bg-bg-card border border-bg-elevated rounded-xl p-6">
          <div className="flex items-center gap-3 mb-4">
            <Activity className="w-5 h-5 text-accent" />
            <h3 className="text-lg font-semibold font-display">自玩统计</h3>
          </div>
          <div className="space-y-4">
            <MetricRow label="已完成对局" value={trainingStatus.selfplay_games.toString()} />
            <MetricRow label="平均步数" value={trainingStatus.avg_game_length.toFixed(1)} />
            <MetricRow 
              label="环境数量" 
              value={trainingStatus.num_envs?.toString() || '0'}
              color="text-accent"
            />
            <MetricRow 
              label="缓冲区大小" 
              value={trainingStatus.buffer_size?.toLocaleString() || '0'}
              color="text-green-400"
            />
            <MetricRow 
              label="玩家1胜率" 
              value={`${((trainingStatus.win_rate?.player_0 || 0) * 100).toFixed(1)}%`}
              color="text-blue-400"
            />
          </div>
        </div>

        {/* 系统信息 */}
        <div className="bg-bg-card border border-bg-elevated rounded-xl p-6">
          <div className="flex items-center gap-3 mb-4">
            <Cpu className="w-5 h-5 text-accent" />
            <h3 className="text-lg font-semibold font-display">系统信息</h3>
          </div>
          {loading ? (
            <div className="text-gray-500">加载中...</div>
          ) : systemInfo ? (
            <div className="space-y-4">
              <MetricRow label="平台" value={systemInfo.platform} />
              <MetricRow label="PyTorch" value={systemInfo.torch_version} />
              {systemInfo.cuda_available && (
                <>
                  <MetricRow label="GPU" value={systemInfo.gpu_name?.slice(0, 20) || 'N/A'} />
                  <MetricRow label="显存" value={`${systemInfo.gpu_memory}GB`} />
                </>
              )}
            </div>
          ) : (
            <div className="text-gray-500">无法获取系统信息</div>
          )}
        </div>
      </motion.div>
    </div>
  )
}

// 指标行组件
function MetricRow({ 
  label, 
  value, 
  color = 'text-white' 
}: { 
  label: string
  value: string
  color?: string 
}) {
  return (
    <div className="flex justify-between items-center">
      <span className="text-gray-400 text-sm">{label}</span>
      <span className={`font-mono text-sm ${color}`}>{value}</span>
    </div>
  )
}


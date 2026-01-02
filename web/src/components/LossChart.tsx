import { useMemo } from 'react'
import { 
  LineChart, 
  Line, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  Legend 
} from 'recharts'
import { useAppStore } from '../store'

interface LossChartProps {
  height?: number
}

export function LossChart({ height = 300 }: LossChartProps) {
  const { lossHistory, trainingStatus } = useAppStore()

  // 构建图表数据
  const chartData = lossHistory.length > 0 
    ? lossHistory 
    : [{ epoch: 0, loss: 0 }]
  
  // 计算 Y 轴范围：从 0 开始，最大值取数据最大值的 1.1 倍（留出空间）
  const yDomain = useMemo(() => {
    if (lossHistory.length === 0) return [0, 1]
    const maxLoss = Math.max(...lossHistory.map(d => d.loss))
    const minLoss = Math.min(...lossHistory.map(d => d.loss))
    // 如果最大值很小，设置一个最小范围
    const upperBound = Math.max(maxLoss * 1.1, 0.1)
    return [0, upperBound]
  }, [lossHistory])

  return (
    <div className="bg-bg-card border border-bg-elevated rounded-xl p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold font-display">训练损失</h3>
        <div className="flex items-center gap-4 text-sm">
          <span className="text-gray-400">
            当前: <span className="text-accent">{trainingStatus.loss.toFixed(4)}</span>
          </span>
        </div>
      </div>
      
      <ResponsiveContainer width="100%" height={height}>
        <LineChart data={chartData} margin={{ top: 5, right: 20, bottom: 5, left: 0 }}>
          <CartesianGrid 
            strokeDasharray="3 3" 
            stroke="rgba(255,255,255,0.05)" 
            vertical={false}
          />
          <XAxis 
            dataKey="epoch" 
            stroke="#666"
            tick={{ fill: '#666', fontSize: 12 }}
            tickFormatter={(value) => `E${value}`}
          />
          <YAxis 
            stroke="#666"
            tick={{ fill: '#666', fontSize: 12 }}
            tickFormatter={(value) => value.toFixed(2)}
            domain={yDomain}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: '#1a1a24',
              border: '1px solid #2a2a3a',
              borderRadius: '8px',
              padding: '12px',
            }}
            labelStyle={{ color: '#888' }}
            itemStyle={{ color: '#00d4aa' }}
            formatter={(value: number) => [value.toFixed(4), '损失']}
            labelFormatter={(label) => `Epoch ${label}`}
          />
          <Legend 
            wrapperStyle={{ paddingTop: '20px' }}
          />
          <Line
            type="monotone"
            dataKey="loss"
            name="总损失"
            stroke="#00d4aa"
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, fill: '#00d4aa' }}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}


import { useMemo } from 'react'
import ReactECharts from 'echarts-for-react'
import { useAppStore } from '../store'

interface LossChartProps {
  height?: number
}

/**
 * 训练损失图表
 * 
 * 交互方式：
 * - 鼠标滚轮：缩放 X 轴
 * - 鼠标拖拽：平移 X 轴
 * - 双击：重置缩放
 */
export function LossChart({ height = 300 }: LossChartProps) {
  const { lossHistory, trainingStatus } = useAppStore()

  // 构建图表配置
  const option = useMemo(() => {
    const data = lossHistory.length > 0 
      ? lossHistory.map(d => [d.epoch, d.loss])
      : [[0, 0]]
    
    return {
      // 背景透明
      backgroundColor: 'transparent',
      
      // 提示框
      tooltip: {
        trigger: 'axis',
        backgroundColor: '#1a1a24',
        borderColor: '#2a2a3a',
        borderWidth: 1,
        textStyle: {
          color: '#888',
        },
        formatter: (params: any) => {
          const p = params[0]
          return `<div style="padding: 4px 8px;">
            <div style="color: #888; margin-bottom: 4px;">Epoch ${p.data[0]}</div>
            <div style="color: #00d4aa; font-weight: bold;">损失: ${p.data[1].toFixed(4)}</div>
          </div>`
        },
      },
      
      // 图例（移到右上角避免重叠）
      legend: {
        show: true,
        top: 0,
        right: 0,
        textStyle: {
          color: '#666',
        },
      },
      
      // 网格
      grid: {
        left: 50,
        right: 20,
        top: 30,
        bottom: 30,
      },
      
      // X 轴
      xAxis: {
        type: 'value',
        axisLine: {
          lineStyle: {
            color: '#333',
          },
        },
        axisLabel: {
          color: '#666',
          formatter: (value: number) => `E${Math.round(value)}`,
        },
        splitLine: {
          show: false,
        },
      },
      
      // Y 轴
      yAxis: {
        type: 'value',
        axisLine: {
          lineStyle: {
            color: '#333',
          },
        },
        axisLabel: {
          color: '#666',
          formatter: (value: number) => value.toFixed(2),
        },
        splitLine: {
          lineStyle: {
            color: 'rgba(255,255,255,0.05)',
            type: 'dashed',
          },
        },
        min: 0,
      },
      
      // 数据缩放 - 鼠标滚轮缩放 X 轴
      dataZoom: [
        {
          type: 'inside',      // 内置缩放（鼠标滚轮 + 拖拽）
          xAxisIndex: 0,
          filterMode: 'none',
          zoomOnMouseWheel: true,  // 滚轮缩放
          moveOnMouseMove: true,   // 按住鼠标移动平移
          moveOnMouseWheel: false, // 禁止滚轮平移
        },
      ],
      
      // 数据系列
      series: [
        {
          name: '总损失',
          type: 'line',
          smooth: true,
          showSymbol: false,
          lineStyle: {
            color: '#00d4aa',
            width: 2,
          },
          areaStyle: {
            color: {
              type: 'linear',
              x: 0,
              y: 0,
              x2: 0,
              y2: 1,
              colorStops: [
                { offset: 0, color: 'rgba(0, 212, 170, 0.3)' },
                { offset: 1, color: 'rgba(0, 212, 170, 0)' },
              ],
            },
          },
          emphasis: {
            focus: 'series',
          },
          data: data,
        },
      ],
    }
  }, [lossHistory])

  return (
    <div className="bg-bg-card border border-bg-elevated rounded-xl p-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold font-display">训练损失</h3>
        <div className="flex items-center gap-4 text-sm">
          <span className="text-gray-500 text-xs">滚轮缩放 | 拖拽平移</span>
          <span className="text-gray-400">
            当前: <span className="text-accent">{trainingStatus.loss.toFixed(4)}</span>
          </span>
        </div>
      </div>
      
      <ReactECharts
        option={option}
        style={{ height: height }}
        opts={{ renderer: 'canvas' }}
        notMerge={true}
      />
    </div>
  )
}

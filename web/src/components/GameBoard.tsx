/**
 * GameBoard - 通用游戏渲染组件
 * 
 * 支持多种渲染模式:
 * - grid: 网格类游戏（井字棋、围棋、五子棋等）
 * - svg: 后端返回 SVG（中国象棋等）
 * - text: 文本渲染
 * - image: 图像渲染（Gymnasium/Atari 游戏等）
 */

import { useState, useCallback, useEffect, useMemo } from 'react'
import { motion } from 'framer-motion'

// ============================================================
// 类型定义
// ============================================================

// 网格渲染数据
interface GridRenderData {
  type: 'grid'
  rows: number
  cols: number
  cells: (string | number | null)[][]  // 每个格子的值
  cell_colors?: (string | null)[][]    // 格子颜色
  highlights?: [number, number][]       // 高亮的格子
  labels?: {
    row?: string[]
    col?: string[]
  }
}

// 图像渲染数据（Gymnasium/Atari 游戏）
interface ImageRenderData {
  type: 'image'
  image_base64?: string      // Base64 编码的图像
  image_width?: number       // 图像宽度
  image_height?: number      // 图像高度
  env_id?: string            // 环境 ID
  step_count?: number        // 步数
  total_reward?: number      // 累计奖励
  is_terminal?: boolean      // 是否结束
}

// 通用渲染数据
type RenderData = GridRenderData | ImageRenderData | { type: 'svg'; svg: string } | { type: 'text'; text: string } | unknown

interface GameBoardProps {
  renderData?: RenderData
  legalActions?: number[]
  actionToPosition?: (action: number) => [number, number] | null  // 动作索引 -> 棋盘坐标
  positionToAction?: (row: number, col: number) => number | null  // 棋盘坐标 -> 动作索引
  onAction?: (action: number) => void
  interactive?: boolean
  gameType?: string  // 游戏类型，用于特殊处理
  className?: string
}

// ============================================================
// 网格游戏渲染器
// ============================================================

interface GridRendererProps {
  data: GridRenderData
  legalActions?: number[]
  positionToAction?: (row: number, col: number) => number | null
  onAction?: (action: number) => void
  interactive?: boolean
  selectedCell: [number, number] | null
  onSelectCell: (cell: [number, number] | null) => void
}

function GridRenderer({
  data,
  legalActions = [],
  positionToAction,
  onAction,
  interactive = true,
  selectedCell,
  onSelectCell,
}: GridRendererProps) {
  const { rows, cols, cells, cell_colors, highlights, labels } = data
  
  // 计算格子大小
  const cellSize = Math.min(50, 400 / Math.max(rows, cols))
  const boardWidth = cols * cellSize
  const boardHeight = rows * cellSize
  
  // 判断位置是否合法
  const isLegalPosition = useCallback((row: number, col: number): boolean => {
    if (!positionToAction) return false
    const action = positionToAction(row, col)
    return action !== null && legalActions.includes(action)
  }, [positionToAction, legalActions])
  
  // 处理点击
  const handleClick = useCallback((row: number, col: number) => {
    if (!interactive || !onAction || !positionToAction) return
    
    const action = positionToAction(row, col)
    if (action !== null && legalActions.includes(action)) {
      onAction(action)
      onSelectCell(null)
    }
  }, [interactive, onAction, positionToAction, legalActions, onSelectCell])
  
  // 获取格子显示内容
  const getCellDisplay = (value: string | number | null): string => {
    if (value === null || value === 0 || value === '') return ''
    if (typeof value === 'number') {
      // 常见映射: 1 -> X/黑, 2 -> O/白, -1 -> X
      if (value === 1) return '●'
      if (value === 2 || value === -1) return '○'
      return String(value)
    }
    return String(value)
  }
  
  // 获取格子颜色
  const getCellColor = (value: string | number | null, customColor?: string | null): string => {
    if (customColor) return customColor
    if (value === 1) return '#1a1a1a'  // 黑子
    if (value === 2 || value === -1) return '#ffffff'  // 白子
    return 'transparent'
  }
  
  return (
    <div className="relative inline-block">
      <svg 
        width={boardWidth + 40} 
        height={boardHeight + 40}
        className="select-none"
      >
        {/* 背景 */}
        <rect 
          x="20" y="20" 
          width={boardWidth} 
          height={boardHeight}
          fill="#DEB887"
          stroke="#8B4513"
          strokeWidth="2"
          rx="4"
        />
        
        {/* 网格线 */}
        {Array.from({ length: rows + 1 }).map((_, i) => (
          <line
            key={`h${i}`}
            x1={20}
            y1={20 + i * cellSize}
            x2={20 + boardWidth}
            y2={20 + i * cellSize}
            stroke="#8B4513"
            strokeWidth="1"
            opacity="0.5"
          />
        ))}
        {Array.from({ length: cols + 1 }).map((_, i) => (
          <line
            key={`v${i}`}
            x1={20 + i * cellSize}
            y1={20}
            x2={20 + i * cellSize}
            y2={20 + boardHeight}
            stroke="#8B4513"
            strokeWidth="1"
            opacity="0.5"
          />
        ))}
        
        {/* 格子内容 */}
        {cells.map((row, rowIdx) =>
          row.map((cell, colIdx) => {
            const cx = 20 + colIdx * cellSize + cellSize / 2
            const cy = 20 + rowIdx * cellSize + cellSize / 2
            const isLegal = isLegalPosition(rowIdx, colIdx)
            const isSelected = selectedCell?.[0] === rowIdx && selectedCell?.[1] === colIdx
            const isHighlighted = highlights?.some(([r, c]) => r === rowIdx && c === colIdx)
            const customColor = cell_colors?.[rowIdx]?.[colIdx]
            
            return (
              <g key={`${rowIdx}-${colIdx}`}>
                {/* 高亮背景 */}
                {isHighlighted && (
                  <rect
                    x={20 + colIdx * cellSize + 2}
                    y={20 + rowIdx * cellSize + 2}
                    width={cellSize - 4}
                    height={cellSize - 4}
                    fill="rgba(0, 212, 170, 0.2)"
                    rx="2"
                  />
                )}
                
                {/* 合法移动提示 */}
                {interactive && isLegal && !cell && (
                  <circle
                    cx={cx}
                    cy={cy}
                    r={cellSize / 6}
                    fill="rgba(0, 212, 170, 0.4)"
                    className="animate-pulse"
                  />
                )}
                
                {/* 选中指示 */}
                {isSelected && (
                  <rect
                    x={20 + colIdx * cellSize + 1}
                    y={20 + rowIdx * cellSize + 1}
                    width={cellSize - 2}
                    height={cellSize - 2}
                    fill="none"
                    stroke="#00d4aa"
                    strokeWidth="3"
                    rx="4"
                  />
                )}
                
                {/* 棋子 */}
                {cell !== null && cell !== 0 && cell !== '' && (
                  <motion.circle
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    cx={cx}
                    cy={cy}
                    r={cellSize / 2.5}
                    fill={getCellColor(cell, customColor)}
                    stroke={cell === 1 ? '#333' : '#999'}
                    strokeWidth="2"
                    filter="url(#shadow)"
                  />
                )}
                
                {/* 文字标签（如果有） */}
                {typeof cell === 'string' && cell.length === 1 && (
                  <text
                    x={cx}
                    y={cy + 5}
                    textAnchor="middle"
                    fill={cell === 'X' ? '#e74c3c' : cell === 'O' ? '#3498db' : '#333'}
                    fontSize={cellSize / 2}
                    fontWeight="bold"
                    fontFamily="sans-serif"
                  >
                    {cell}
                  </text>
                )}
                
                {/* 可点击区域 */}
                {interactive && (
                  <rect
                    x={20 + colIdx * cellSize}
                    y={20 + rowIdx * cellSize}
                    width={cellSize}
                    height={cellSize}
                    fill="transparent"
                    cursor={isLegal ? 'pointer' : 'default'}
                    onClick={() => handleClick(rowIdx, colIdx)}
                  />
                )}
              </g>
            )
          })
        )}
        
        {/* 坐标标签 */}
        {labels?.col && labels.col.map((label, i) => (
          <text
            key={`col${i}`}
            x={20 + i * cellSize + cellSize / 2}
            y={boardHeight + 35}
            textAnchor="middle"
            fill="#666"
            fontSize="12"
          >
            {label}
          </text>
        ))}
        {labels?.row && labels.row.map((label, i) => (
          <text
            key={`row${i}`}
            x={8}
            y={20 + i * cellSize + cellSize / 2 + 4}
            textAnchor="middle"
            fill="#666"
            fontSize="12"
          >
            {label}
          </text>
        ))}
        
        {/* 阴影滤镜 */}
        <defs>
          <filter id="shadow" x="-50%" y="-50%" width="200%" height="200%">
            <feDropShadow dx="1" dy="1" stdDeviation="1" floodOpacity="0.3" />
          </filter>
        </defs>
      </svg>
    </div>
  )
}

// ============================================================
// SVG 渲染器
// ============================================================

function SVGRenderer({ svg, className }: { svg: string; className?: string }) {
  return (
    <div 
      className={className}
      dangerouslySetInnerHTML={{ __html: svg }}
    />
  )
}

// ============================================================
// 文本渲染器
// ============================================================

function TextRenderer({ text, className }: { text: string; className?: string }) {
  return (
    <pre 
      className={`text-sm bg-bg-elevated p-4 rounded-lg overflow-auto whitespace-pre ${className || ''}`}
      style={{ fontFamily: '"Noto Sans Mono CJK SC", "Source Han Mono", "Sarasa Mono SC", "Microsoft YaHei", "PingFang SC", monospace' }}
    >
      {text}
    </pre>
  )
}

// ============================================================
// 图像渲染器（Gymnasium/Atari 游戏）
// ============================================================

function ImageRenderer({ data, className }: { data: ImageRenderData; className?: string }) {
  const { image_base64, image_width, image_height, env_id, step_count, total_reward, is_terminal } = data
  
  // 调试：检查数据
  if (!image_base64) {
    console.warn('[ImageRenderer] 没有图像数据:', { 
      env_id, 
      step_count, 
      image_width, 
      image_height, 
      render_error: (data as Record<string, unknown>).render_error,
      dataKeys: Object.keys(data) 
    })
  }
  
  return (
    <div className={`flex flex-col items-center ${className || ''}`}>
      {/* 游戏图像 */}
      {image_base64 ? (
        <div className="relative bg-black rounded-lg overflow-hidden">
          <img
            src={`data:image/png;base64,${image_base64}`}
            alt={env_id || 'Game'}
            width={image_width || 400}
            height={image_height || 400}
            className="block"
            style={{ 
              imageRendering: 'auto',  // 使用默认渲染，更清晰
              maxWidth: '100%',
              maxHeight: '500px',
              objectFit: 'contain',
            }}
          />
          {is_terminal && (
            <div className="absolute inset-0 flex items-center justify-center bg-black/50">
              <span className="text-2xl font-bold text-white">游戏结束</span>
            </div>
          )}
        </div>
      ) : (
        <div className="w-96 h-64 bg-bg-elevated rounded-lg flex flex-col items-center justify-center text-gray-500 border border-gray-700 p-4">
          <span>暂无图像</span>
          {(data as Record<string, unknown>).render_error && (
            <span className="text-xs text-red-400 mt-2 text-center">
              {String((data as Record<string, unknown>).render_error)}
            </span>
          )}
        </div>
      )}
      
      {/* 游戏信息 */}
      <div className="mt-4 grid grid-cols-3 gap-4 text-sm">
        {env_id && (
          <div className="text-center">
            <div className="text-gray-500">环境</div>
            <div className="font-mono text-accent">{env_id}</div>
          </div>
        )}
        {step_count !== undefined && (
          <div className="text-center">
            <div className="text-gray-500">步数</div>
            <div className="font-mono">{step_count}</div>
          </div>
        )}
        {total_reward !== undefined && (
          <div className="text-center">
            <div className="text-gray-500">奖励</div>
            <div className={`font-mono ${total_reward >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {total_reward.toFixed(1)}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

// ============================================================
// 主组件
// ============================================================

export function GameBoard({
  renderData,
  legalActions = [],
  actionToPosition,
  positionToAction,
  onAction,
  interactive = true,
  gameType,
  className = '',
}: GameBoardProps) {
  const [selectedCell, setSelectedCell] = useState<[number, number] | null>(null)
  
  // 清除选中当 legalActions 变化时
  useEffect(() => {
    setSelectedCell(null)
  }, [legalActions])
  
  // 默认位置转动作（适用于简单网格游戏）
  const defaultPositionToAction = useMemo(() => {
    if (positionToAction) return positionToAction
    
    // 对于网格游戏，默认 action = row * cols + col
    if (renderData && typeof renderData === 'object' && 'type' in renderData) {
      const data = renderData as { type: string; cols?: number }
      if (data.type === 'grid' && data.cols) {
        return (row: number, col: number) => row * data.cols! + col
      }
    }
    return undefined
  }, [renderData, positionToAction])
  
  // 渲染内容
  if (!renderData) {
    return (
      <div className={`flex items-center justify-center h-64 bg-bg-elevated rounded-xl ${className}`}>
        <span className="text-gray-500">等待游戏数据...</span>
      </div>
    )
  }
  
  // 类型判断
  if (typeof renderData === 'object' && 'type' in renderData) {
    const data = renderData as { type: string }
    
    if (data.type === 'grid') {
      return (
        <div className={`flex justify-center ${className}`}>
          <GridRenderer
            data={renderData as GridRenderData}
            legalActions={legalActions}
            positionToAction={defaultPositionToAction}
            onAction={onAction}
            interactive={interactive}
            selectedCell={selectedCell}
            onSelectCell={setSelectedCell}
          />
        </div>
      )
    }
    
    if (data.type === 'svg' && 'svg' in renderData) {
      return <SVGRenderer svg={(renderData as { svg: string }).svg} className={className} />
    }
    
    if (data.type === 'text' && 'text' in renderData) {
      return <TextRenderer text={(renderData as { text: string }).text} className={className} />
    }
    
    if (data.type === 'image') {
      return (
        <div className={`flex justify-center ${className}`}>
          <ImageRenderer data={renderData as ImageRenderData} />
        </div>
      )
    }
  }
  
  // 未知格式，显示 JSON（用于调试）
  const typeInfo = typeof renderData === 'object' && renderData !== null && 'type' in renderData 
    ? (renderData as {type: string}).type 
    : '无type字段'
  
  return (
    <div className={`bg-bg-elevated rounded-xl p-4 ${className}`}>
      <div className="text-sm text-gray-400 mb-2">
        未识别的渲染格式 (type: {typeInfo})
      </div>
      <pre className="font-mono text-xs overflow-auto max-h-64">
        {JSON.stringify(renderData, null, 2)}
      </pre>
    </div>
  )
}

// ============================================================
// 导出辅助组件
// ============================================================

export { GridRenderer, SVGRenderer, TextRenderer, ImageRenderer }
export type { GridRenderData, ImageRenderData, RenderData, GameBoardProps }

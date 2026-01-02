/**
 * GameBoard - 通用游戏渲染组件
 * 
 * 支持多种渲染模式:
 * - grid: 网格类游戏（井字棋、围棋等）
 * - svg: 后端返回 SVG
 * - text: 文本渲染
 * - custom: 自定义渲染（中国象棋等）
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

// 通用渲染数据
type RenderData = GridRenderData | { type: 'svg'; svg: string } | { type: 'text'; text: string } | unknown

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
  }
  
  // 未知格式，显示 JSON
  return (
    <div className={`bg-bg-elevated rounded-xl p-4 ${className}`}>
      <div className="text-sm text-gray-400 mb-2">原始数据:</div>
      <pre className="font-mono text-xs overflow-auto max-h-64">
        {JSON.stringify(renderData, null, 2)}
      </pre>
    </div>
  )
}

// ============================================================
// 导出辅助组件
// ============================================================

export { GridRenderer, SVGRenderer, TextRenderer }
export type { GridRenderData, RenderData, GameBoardProps }

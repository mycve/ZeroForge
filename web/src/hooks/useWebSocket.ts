import { useEffect, useRef, useCallback } from 'react'
import { useAppStore } from '../store'

export function useTrainingWebSocket() {
  const wsRef = useRef<WebSocket | null>(null)
  const reconnectTimeoutRef = useRef<number | null>(null)
  const { setTrainingStatus, setConnected, addLossHistory } = useAppStore()

  const connect = useCallback(() => {
    // 清理之前的连接
    if (wsRef.current) {
      wsRef.current.close()
    }

    const ws = new WebSocket(`ws://${window.location.host}/ws/training`)

    ws.onopen = () => {
      console.log('训练 WebSocket 已连接')
      setConnected(true)
    }

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data)
        
        if (message.type === 'training_status') {
          const status = message.data
          setTrainingStatus(status)
          
          // 添加到历史
          if (status.step > 0 && status.loss > 0) {
            addLossHistory(status.step, status.loss)
          }
        }
      } catch (e) {
        console.error('解析 WebSocket 消息失败:', e)
      }
    }

    ws.onclose = () => {
      console.log('训练 WebSocket 已断开')
      setConnected(false)
      
      // 自动重连
      reconnectTimeoutRef.current = window.setTimeout(() => {
        console.log('尝试重新连接...')
        connect()
      }, 3000)
    }

    ws.onerror = (error) => {
      console.error('WebSocket 错误:', error)
    }

    wsRef.current = ws
  }, [setTrainingStatus, setConnected, addLossHistory])

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
    }
    if (wsRef.current) {
      wsRef.current.close()
    }
  }, [])

  const sendCommand = useCallback((action: string) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'command', action }))
    }
  }, [])

  useEffect(() => {
    connect()
    return () => disconnect()
  }, [connect, disconnect])

  return { sendCommand }
}

export function useGameWebSocket(gameId: string | null) {
  const wsRef = useRef<WebSocket | null>(null)
  const { setCurrentSession } = useAppStore()

  const connect = useCallback((id: string) => {
    if (wsRef.current) {
      wsRef.current.close()
    }

    const ws = new WebSocket(`ws://${window.location.host}/ws/game/${id}`)

    ws.onopen = () => {
      console.log(`游戏 ${id} WebSocket 已连接`)
    }

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data)
        
        if (message.type === 'game_state' || message.type === 'game_update') {
          setCurrentSession(message.data)
        }
      } catch (e) {
        console.error('解析游戏 WebSocket 消息失败:', e)
      }
    }

    ws.onclose = () => {
      console.log(`游戏 ${id} WebSocket 已断开`)
    }

    wsRef.current = ws
  }, [setCurrentSession])

  const disconnect = useCallback(() => {
    if (wsRef.current) {
      wsRef.current.close()
    }
  }, [])

  const sendMove = useCallback((move: string) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type: 'move', move }))
    }
  }, [])

  useEffect(() => {
    if (gameId) {
      connect(gameId)
    }
    return () => disconnect()
  }, [gameId, connect, disconnect])

  return { sendMove }
}


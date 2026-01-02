import { useEffect, useCallback } from 'react'
import { useAppStore } from '../store'

// ============================================================
// 全局 WebSocket 单例管理器
// ============================================================

class TrainingWebSocketManager {
  private ws: WebSocket | null = null
  private reconnectTimeout: number | null = null
  private heartbeatInterval: number | null = null
  private reconnectAttempts = 0
  private isConnecting = false

  // 获取 WebSocket URL
  private getWsUrl(): string {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const isDev = window.location.port === '5173' || window.location.port === '5174'
    const host = isDev ? `${window.location.hostname}:8000` : window.location.host
    return `${protocol}//${host}/ws/training`
  }

  // 启动心跳
  private startHeartbeat(): void {
    this.stopHeartbeat()
    this.heartbeatInterval = window.setInterval(() => {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({ type: 'ping' }))
      }
    }, 10000)
  }

  // 停止心跳
  private stopHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval)
      this.heartbeatInterval = null
    }
  }

  // 连接
  connect(): void {
    // 已连接或正在连接则跳过
    if (this.isConnecting || (this.ws && this.ws.readyState === WebSocket.OPEN)) {
      return
    }
    
    this.isConnecting = true
    
    // 清理旧连接
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }

    const wsUrl = this.getWsUrl()
    console.log(`[WS] 连接: ${wsUrl}`)
    
    try {
      const ws = new WebSocket(wsUrl)
      const store = useAppStore.getState()

      ws.onopen = () => {
        console.log('[WS] 已连接')
        this.isConnecting = false
        store.setConnected(true)
        this.reconnectAttempts = 0
        this.startHeartbeat()
      }

      ws.onmessage = (event) => {
        try {
          const message = JSON.parse(event.data)
          const store = useAppStore.getState()
          
          if (message.type === 'training_status') {
            store.setTrainingStatus(message.data)
            
            if (message.data.epoch > 0 && message.data.loss > 0) {
              store.addLossHistory(message.data.epoch, message.data.loss)
            }
          }
        } catch (e) {
          console.error('[WS] 解析消息失败:', e)
        }
      }

      ws.onclose = (event) => {
        console.log(`[WS] 断开: code=${event.code}`)
        this.isConnecting = false
        this.ws = null
        useAppStore.getState().setConnected(false)
        this.stopHeartbeat()
        
        // 自动重连
        const baseDelay = 2000
        const maxDelay = 30000
        const delay = Math.min(baseDelay * Math.pow(2, this.reconnectAttempts), maxDelay)
        this.reconnectAttempts++
        
        console.log(`[WS] ${delay / 1000}s 后重连 (第 ${this.reconnectAttempts} 次)`)
        this.reconnectTimeout = window.setTimeout(() => this.connect(), delay)
      }

      ws.onerror = (error) => {
        console.error('[WS] 错误:', error)
        this.isConnecting = false
      }

      this.ws = ws
    } catch (e) {
      console.error('[WS] 创建失败:', e)
      this.isConnecting = false
    }
  }

  // 断开
  disconnect(): void {
    this.stopHeartbeat()
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout)
      this.reconnectTimeout = null
    }
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }
    this.isConnecting = false
  }

  // 发送命令
  sendCommand(action: string): void {
    if (this.ws && this.ws.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({ type: 'command', action }))
    } else {
      console.warn(`[WS] 无法发送命令 ${action}: 未连接`)
    }
  }

  // 是否已连接
  isConnected(): boolean {
    return this.ws !== null && this.ws.readyState === WebSocket.OPEN
  }
}

// 全局单例
const wsManager = new TrainingWebSocketManager()

// 应用启动时自动连接
if (typeof window !== 'undefined') {
  wsManager.connect()
}

// ============================================================
// React Hook（只是对单例的包装）
// ============================================================

export function useTrainingWebSocket() {
  // 确保连接（如果意外断开）
  useEffect(() => {
    if (!wsManager.isConnected()) {
      wsManager.connect()
    }
  }, [])

  const sendCommand = useCallback((action: string) => {
    wsManager.sendCommand(action)
  }, [])

  return { sendCommand }
}

// ============================================================
// 游戏 WebSocket（保持原样，每个游戏一个连接）
// ============================================================

export function useGameWebSocket(gameId: string | null) {
  const { setCurrentSession } = useAppStore()

  useEffect(() => {
    if (!gameId) return

    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:'
    const isDev = window.location.port === '5173' || window.location.port === '5174'
    const host = isDev ? `${window.location.hostname}:8000` : window.location.host
    const ws = new WebSocket(`${protocol}//${host}/ws/game/${gameId}`)

    ws.onopen = () => {
      console.log(`[游戏WS] ${gameId} 已连接`)
    }

    ws.onmessage = (event) => {
      try {
        const message = JSON.parse(event.data)
        if (message.type === 'game_state' || message.type === 'game_update') {
          setCurrentSession(message.data)
        }
      } catch (e) {
        console.error('[游戏WS] 解析消息失败:', e)
      }
    }

    ws.onclose = () => {
      console.log(`[游戏WS] ${gameId} 断开`)
    }

    return () => {
      ws.close()
    }
  }, [gameId, setCurrentSession])

  const sendMove = useCallback((move: string) => {
    // 这个需要重构，暂时保持简单
    console.warn('sendMove 未实现')
  }, [])

  return { sendMove }
}

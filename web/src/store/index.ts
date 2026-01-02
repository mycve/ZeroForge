import { create } from 'zustand'
import type { TrainingStatus, GameSession, SystemInfo, GameInfo } from '../utils/api'

// ============================================================
// 类型重导出
// ============================================================

export type { TrainingStatus, GameSession, SystemInfo, GameInfo }

// ============================================================
// 应用配置
// ============================================================

export interface AppConfig {
  game: {
    type: string
    max_game_length: number
  }
  algorithm: {
    type: string
    num_simulations: number
    c_visit: number
  }
  training: {
    batch_size: number
    lr: number
    num_epochs: number
  }
  system: {
    device: string
    num_envs: number
    log_backends: string[]
  }
}

// ============================================================
// Store
// ============================================================

interface AppState {
  // 训练状态
  trainingStatus: TrainingStatus
  setTrainingStatus: (status: TrainingStatus) => void
  
  // 游戏会话（通用）
  gameSessions: GameSession[]
  currentSession: GameSession | null
  setGameSessions: (sessions: GameSession[]) => void
  setCurrentSession: (session: GameSession | null) => void
  
  // 可用游戏列表
  availableGames: GameInfo[]
  setAvailableGames: (games: GameInfo[]) => void
  
  // 系统信息
  systemInfo: SystemInfo | null
  setSystemInfo: (info: SystemInfo) => void
  
  // 配置
  config: AppConfig | null
  setConfig: (config: AppConfig) => void
  
  // 连接状态
  connected: boolean
  setConnected: (connected: boolean) => void
  
  // 历史数据（用于图表）
  lossHistory: { step: number; loss: number }[]
  addLossHistory: (step: number, loss: number) => void
  clearLossHistory: () => void
}

const defaultTrainingStatus: TrainingStatus = {
    running: false,
    paused: false,
    step: 0,
    epoch: 0,
    total_games: 0,
    loss: 0,
    value_loss: 0,
    policy_loss: 0,
    reward_loss: 0,
    selfplay_games: 0,
    avg_game_length: 0,
    win_rate: {},
    eval_win_rate: 0,
    eval_elo: 0,
    elapsed_time: 0,
    games_per_second: 0,
    steps_per_second: 0,
    buffer_size: 0,
    num_envs: 0,
}

export const useAppStore = create<AppState>((set) => ({
  // 训练状态
  trainingStatus: defaultTrainingStatus,
  setTrainingStatus: (status) => set({ trainingStatus: status }),
  
  // 游戏会话
  gameSessions: [],
  currentSession: null,
  setGameSessions: (sessions) => set({ gameSessions: sessions }),
  setCurrentSession: (session) => set({ currentSession: session }),
  
  // 可用游戏
  availableGames: [],
  setAvailableGames: (games) => set({ availableGames: games }),
  
  // 系统信息
  systemInfo: null,
  setSystemInfo: (info) => set({ systemInfo: info }),
  
  // 配置
  config: null,
  setConfig: (config) => set({ config }),
  
  // 连接状态
  connected: false,
  setConnected: (connected) => set({ connected }),
  
  // 历史数据
  lossHistory: [],
  addLossHistory: (step, loss) =>
    set((state) => ({
      lossHistory: [...state.lossHistory.slice(-100), { step, loss }],
    })),
  clearLossHistory: () => set({ lossHistory: [] }),
}))


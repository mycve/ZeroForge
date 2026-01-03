const API_BASE = '/api'

// ============================================================
// 通用请求函数
// ============================================================

async function request<T>(
  endpoint: string,
  options?: RequestInit
): Promise<T> {
  const response = await fetch(`${API_BASE}${endpoint}`, {
    headers: {
      'Content-Type': 'application/json',
    },
    ...options,
  })

  if (!response.ok) {
    const error = await response.json().catch(() => ({ detail: '请求失败' }))
    throw new Error(error.detail || `HTTP ${response.status}`)
  }

  return response.json()
}

// ============================================================
// 系统 API
// ============================================================

export interface SystemInfo {
  platform: string
  python_version: string
  torch_version: string
  cuda_available: boolean
  cuda_version?: string
  gpu_count?: number
  gpu_name?: string
  gpu_memory?: number
}

export interface GameInfo {
  name: string
  class: string
  observation_space?: {
    shape: number[]
    dtype: string
  }
  action_space?: {
    n: number
  }
  num_players: number
  render_modes: string[]  // 支持的渲染模式: "text", "json", "svg"
}

export interface AlgorithmInfo {
  name: string
  class: string
  needs_dynamics: boolean
}

export async function getSystemInfo(): Promise<SystemInfo> {
  return request('/system/info')
}

export async function listGames(): Promise<GameInfo[]> {
  return request('/system/games')
}

export async function listAlgorithms(): Promise<AlgorithmInfo[]> {
  return request('/system/algorithms')
}

// ============================================================
// 配置 API
// ============================================================

export interface ConfigFieldMeta {
  type: 'int' | 'float' | 'string' | 'bool' | 'select' | 'multiselect'
  label: string
  description?: string
  min?: number
  max?: number
  step?: number
  options?: string[]
}

export interface ConfigGroup {
  label: string
  fields: string[]
}

export interface ConfigSchema {
  groups: Record<string, ConfigGroup>
  fields: Record<string, ConfigFieldMeta>
}

export async function getConfig(section?: string): Promise<Record<string, unknown>> {
  const query = section ? `?section=${section}` : ''
  return request(`/config${query}`)
}

export async function getConfigSchema(): Promise<ConfigSchema> {
  return request('/config/schema')
}

export async function updateConfig(
  section: string,
  values: Record<string, unknown>
): Promise<{ success: boolean }> {
  return request('/config', {
    method: 'POST',
    body: JSON.stringify({ section, values }),
  })
}

export async function updateConfigBatch(
  values: Record<string, unknown>
): Promise<{ success: boolean }> {
  return request('/config', {
    method: 'PUT',
    body: JSON.stringify(values),
  })
}

// ============================================================
// 训练 API
// ============================================================

export interface TrainingStatus {
  running: boolean
  paused: boolean
  step: number
  epoch: number
  total_games: number
  loss: number
  value_loss: number
  policy_loss: number
  reward_loss: number
  selfplay_games: number
  avg_game_length: number
  win_rate: Record<string, number>  // 通用: "player_0", "player_1", "draw"
  eval_win_rate: number
  eval_elo: number
  elapsed_time: number
  games_per_second: number
  steps_per_second: number
  // 架构状态
  buffer_size: number      // 回放缓冲区大小
  num_envs: number         // 并行环境数
  concurrency: number      // 并发数
  // 任务配置信息
  game_type: string        // 游戏类型
  algorithm: string        // 算法
  num_epochs: number       // 总 epoch 数
  batch_size: number       // 批大小
  lr: number               // 学习率
  num_simulations: number  // MCTS 模拟次数
  use_ddp: boolean         // 是否使用 DDP
}

export async function getTrainingStatus(): Promise<TrainingStatus> {
  return request('/training/status')
}

export async function sendTrainingCommand(
  action: 'start' | 'pause' | 'resume' | 'stop'
): Promise<{ success: boolean; action: string }> {
  return request('/training/command', {
    method: 'POST',
    body: JSON.stringify({ action }),
  })
}

// ============================================================
// 调试 API
// ============================================================

export interface DebugSession {
  session_id: string
  game_type: string
  algorithm: string
  step: number
  game_count: number
  is_terminal: boolean
}

export interface DebugSessionState {
  current_player: number | null
  legal_actions: number[]
  is_terminal: boolean
  winner: number | null
  game_render: string
  observation_shape: number[]
  action_space: number
}

export interface DebugSessionDetail {
  session_id: string
  game_type: string
  algorithm: string
  step: number
  game_count: number
  is_terminal: boolean
  state: DebugSessionState
  history: Array<{ type: string; step: number; [key: string]: unknown }>
}

export interface CreateDebugSessionRequest {
  game_type: string
  algorithm: string
  device: string
  checkpoint_path?: string
}

export async function listDebugSessions(): Promise<{ sessions: DebugSession[] }> {
  return request('/debug/sessions')
}

export async function createDebugSession(
  req: CreateDebugSessionRequest
): Promise<{ session_id: string; state: DebugSessionState }> {
  return request('/debug/sessions', {
    method: 'POST',
    body: JSON.stringify(req),
  })
}

export async function getDebugSession(sessionId: string): Promise<DebugSessionDetail> {
  return request(`/debug/sessions/${sessionId}`)
}

export async function deleteDebugSession(sessionId: string): Promise<{ success: boolean }> {
  return request(`/debug/sessions/${sessionId}`, { method: 'DELETE' })
}

export async function debugStepGame(
  sessionId: string,
  action?: number
): Promise<unknown> {
  return request(`/debug/sessions/${sessionId}/step`, {
    method: 'POST',
    body: JSON.stringify({ action }),
  })
}

export async function debugStepMcts(
  sessionId: string,
  numSimulations: number = 50
): Promise<unknown> {
  return request(`/debug/sessions/${sessionId}/mcts?num_simulations=${numSimulations}`, {
    method: 'POST',
  })
}

export async function debugResetGame(sessionId: string): Promise<unknown> {
  return request(`/debug/sessions/${sessionId}/reset`, { method: 'POST' })
}

export async function debugRunFullGame(sessionId: string): Promise<unknown> {
  return request(`/debug/sessions/${sessionId}/run`, { method: 'POST' })
}

// === 训练过程调试数据（已有训练时收集） ===

export interface TrainingDebugData {
  mcts: Array<{
    timestamp: number
    game_idx: number
    step: number
    player: number
    legal_actions: number
    selected_action: number
    root_value: number
    root_visits: number
    top_actions: [number, number][]
  }>
  trajectories: Array<{
    timestamp: number
    game_idx: number
    length: number
    actions: number[]
    rewards: number[]
    players: number[]
    winner: number | null
    values: number[]
  }>
  training: Array<{
    timestamp: number
    epoch: number
    num_batches: number
    batch_size: number
    buffer_size: number
    total_loss: number
    policy_loss: number
    value_loss: number
    target_value_range: [number, number] | null
    pred_value_range: [number, number] | null
  }>
  selfplay: Array<{
    timestamp: number
    game_idx: number
    length: number
    winner: number | null
  }>
}

export async function getTrainingDebug(
  category?: string,
  limit: number = 50
): Promise<TrainingDebugData> {
  const params = new URLSearchParams()
  if (category) params.set('category', category)
  params.set('limit', limit.toString())
  return request(`/training/debug?${params}`)
}

export async function clearTrainingDebug(): Promise<{ success: boolean }> {
  return request('/training/debug', { method: 'DELETE' })
}

export async function saveCheckpoint(): Promise<{ success: boolean; epoch?: number; error?: string }> {
  return request('/training/save', { method: 'POST' })
}

// ============================================================
// 检查点 API
// ============================================================

export interface CheckpointInfo {
  path: string
  game_type: string
  algorithm: string
  epoch: number
  step: number
  loss: number
  eval_win_rate: number
  created_at: string
  is_best: boolean
}

export async function listCheckpoints(
  gameType?: string,
  algorithm?: string
): Promise<{ checkpoints: CheckpointInfo[] }> {
  const params = new URLSearchParams()
  if (gameType) params.set('game_type', gameType)
  if (algorithm) params.set('algorithm', algorithm)
  const query = params.toString() ? `?${params.toString()}` : ''
  return request(`/checkpoints${query}`)
}

export async function deleteCheckpoint(path: string): Promise<{ success: boolean }> {
  return request(`/checkpoints?path=${encodeURIComponent(path)}`, { method: 'DELETE' })
}

// ============================================================
// 游戏会话 API（通用）
// ============================================================

export interface GameSession {
  id: string
  game_type: string
  num_players: number
  players: string[]           // 通用玩家列表: ["human", "ai:muzero"]
  state: 'waiting' | 'playing' | 'finished'
  current_player: number      // 当前玩家索引
  step_count: number          // 当前步数
  history: number[]           // 动作历史（动作索引）
  result: {                   // 对局结果
    winner: number | null     // 获胜玩家索引，null=平局
    rewards: Record<number, number>
  } | null
  render_data?: unknown       // 游戏特定渲染数据（由 render API 返回）
}

export async function listGameSessions(): Promise<GameSession[]> {
  return request('/games')
}

export async function createGameSession(
  game_type: string,
  players: string[] = ['human', 'ai:muzero']
): Promise<{ session_id: string }> {
  return request('/games', {
    method: 'POST',
    body: JSON.stringify({ game_type, players }),
  })
}

export async function getGameSession(gameId: string): Promise<GameSession> {
  return request(`/games/${gameId}`)
}

export async function deleteGameSession(gameId: string): Promise<{ success: boolean }> {
  return request(`/games/${gameId}`, { method: 'DELETE' })
}

export async function clearAllGameSessions(): Promise<{ success: boolean; deleted_count: number }> {
  return request('/games', { method: 'DELETE' })
}

export async function startGame(gameId: string): Promise<{ success: boolean }> {
  return request(`/games/${gameId}/start`, { method: 'POST' })
}

export async function doAction(
  gameId: string,
  action: number
): Promise<{ success: boolean; action: number; done: boolean }> {
  return request(`/games/${gameId}/action`, {
    method: 'POST',
    body: JSON.stringify({ action }),
  })
}

export async function getGameRender(
  gameId: string,
  mode: 'text' | 'json' | 'svg' = 'json'
): Promise<{ render: unknown }> {
  return request(`/games/${gameId}/render?mode=${mode}`)
}

export async function getGameLegalActions(
  gameId: string
): Promise<{ actions: number[]; action_names?: string[] }> {
  return request(`/games/${gameId}/legal_actions`)
}

// ============================================================
// 调试 API
// ============================================================

export interface DebugGameState {
  step_index: number
  current_player: number
  is_terminal: boolean
  winner: number | null
  legal_actions: number[]
  render_data?: unknown
}

export async function createDebugGame(
  game_type: string,
  config?: Record<string, unknown>
): Promise<{ game_id: string; game_type: string; state: DebugGameState }> {
  return request('/debug/game/create', {
    method: 'POST',
    body: JSON.stringify({ game_type, config }),
  })
}

export async function getDebugState(
  gameId: string,
  includeRender: boolean = true
): Promise<DebugGameState> {
  return request(`/debug/game/${gameId}/state?include_render=${includeRender}`)
}

export async function debugStep(
  gameId: string,
  action: number
): Promise<{ step_index: number; reward: number; done: boolean; state: DebugGameState }> {
  return request(`/debug/game/${gameId}/step?action=${action}`, { method: 'POST' })
}

export async function resetDebugGame(gameId: string): Promise<{ state: DebugGameState }> {
  return request(`/debug/game/${gameId}/reset`, { method: 'POST' })
}


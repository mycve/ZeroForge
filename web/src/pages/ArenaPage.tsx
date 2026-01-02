import { useState, useCallback, useEffect, useMemo } from 'react'
import { motion } from 'framer-motion'
import { Swords, Play, RotateCcw, User, Cpu, Gamepad2, Brain } from 'lucide-react'
import { clsx } from 'clsx'
import { GameBoard } from '../components/GameBoard'
import { 
  createGameSession, startGame, doAction, 
  listGames, getGameSession, getGameLegalActions, getGameRender,
  listCheckpoints
} from '../utils/api'
import type { GameInfo, GameSession, CheckpointInfo } from '../utils/api'
import { useGameWebSocket } from '../hooks/useWebSocket'
import { useAppStore } from '../store'

// 基础玩家选项
const basePlayerOptions = [
  { value: 'human', label: '人类玩家', icon: User },
  { value: 'random', label: '随机策略', icon: Cpu },
  { value: 'ai:alphazero', label: 'AlphaZero (最新)', icon: Brain },
  { value: 'ai:muzero', label: 'MuZero (最新)', icon: Brain },
]

export default function ArenaPage() {
  // 游戏选择
  const [availableGames, setAvailableGames] = useState<GameInfo[]>([])
  const [selectedGame, setSelectedGame] = useState<string>('tictactoe')
  
  // 检查点
  const [checkpoints, setCheckpoints] = useState<CheckpointInfo[]>([])
  
  // 玩家设置（通用）
  const [players, setPlayers] = useState<string[]>(['human', 'random'])
  
  // 对局状态
  const [gameId, setGameId] = useState<string | null>(null)
  const [gameState, setGameState] = useState<'idle' | 'playing' | 'finished'>('idle')
  const [session, setSession] = useState<GameSession | null>(null)
  const [renderData, setRenderData] = useState<unknown>(null)
  const [legalActions, setLegalActions] = useState<number[]>([])
  
  const { sendMove } = useGameWebSocket(gameId)

  // 加载可用游戏和检查点
  useEffect(() => {
    async function loadData() {
      try {
        const [games, cpData] = await Promise.all([
          listGames(),
          listCheckpoints().catch(() => ({ checkpoints: [] })),
        ])
        setAvailableGames(games)
        setCheckpoints(cpData.checkpoints || [])
        if (games.length > 0 && !games.find(g => g.name === selectedGame)) {
          setSelectedGame(games[0].name)
        }
      } catch (e) {
        console.error('加载数据失败:', e)
      }
    }
    loadData()
  }, [])
  
  // 动态生成玩家选项（包含检查点）
  const playerOptions = useMemo(() => {
    const options = [...basePlayerOptions]
    
    // 添加当前游戏的检查点
    const gameCheckpoints = checkpoints.filter(cp => cp.game_type === selectedGame)
    gameCheckpoints.forEach(cp => {
      options.push({
        value: `checkpoint:${cp.path}`,
        label: `${cp.algorithm} Epoch ${cp.epoch}${cp.is_best ? ' (最佳)' : ''}`,
        icon: Brain,
      })
    })
    
    return options
  }, [checkpoints, selectedGame])

  // 获取当前游戏玩家数
  const currentGameInfo = availableGames.find(g => g.name === selectedGame)
  const numPlayers = currentGameInfo?.num_players || 2

  // 确保玩家数量匹配
  useEffect(() => {
    if (players.length !== numPlayers) {
      const newPlayers = [...players]
      while (newPlayers.length < numPlayers) {
        newPlayers.push('random')
      }
      setPlayers(newPlayers.slice(0, numPlayers))
    }
  }, [numPlayers])
  
  // 游戏切换时重新加载检查点
  useEffect(() => {
    async function loadCheckpoints() {
      try {
        const data = await listCheckpoints(selectedGame)
        setCheckpoints(data.checkpoints || [])
      } catch (e) {
        console.error('加载检查点失败:', e)
      }
    }
    loadCheckpoints()
  }, [selectedGame])
  
  // 游戏进行中时轮询刷新状态（用于 AI vs AI 对弈）
  useEffect(() => {
    if (!gameId || gameState !== 'playing') return
    
    const interval = setInterval(async () => {
      try {
        const [sess, legal, render] = await Promise.all([
          getGameSession(gameId),
          getGameLegalActions(gameId).catch(() => ({ actions: [] })),
          getGameRender(gameId, 'json').catch(() => ({ render: null })),
        ])
        setSession(sess)
        setLegalActions(legal.actions)
        setRenderData(render.render)
        
        if (sess.state === 'finished') {
          setGameState('finished')
        }
      } catch (e) {
        console.error('轮询状态失败:', e)
      }
    }, 500) // 每 500ms 刷新一次
    
    return () => clearInterval(interval)
  }, [gameId, gameState])

  // 创建并开始对局
  const handleStartGame = async () => {
    try {
      const { session_id } = await createGameSession(selectedGame, players)
      setGameId(session_id)
      await startGame(session_id)
      setGameState('playing')
      
      // 加载初始状态
      await refreshGameState(session_id)
    } catch (e) {
      console.error('创建对局失败:', e)
      alert('创建对局失败')
    }
  }

  // 刷新游戏状态
  const refreshGameState = async (gid: string) => {
    try {
      const [sess, legal, render] = await Promise.all([
        getGameSession(gid),
        getGameLegalActions(gid).catch(() => ({ actions: [] })),
        getGameRender(gid, 'json').catch(() => ({ render: null })),
      ])
      setSession(sess)
      setLegalActions(legal.actions)
      setRenderData(render.render)
      
      if (sess.state === 'finished') {
        setGameState('finished')
      }
    } catch (e) {
      console.error('刷新状态失败:', e)
    }
  }

  // 重置
  const handleReset = () => {
    setGameId(null)
    setGameState('idle')
    setSession(null)
    setRenderData(null)
    setLegalActions([])
  }

  // 执行动作
  const handleAction = useCallback(async (action: number) => {
    if (!gameId || gameState !== 'playing') return
    
    try {
      const result = await doAction(gameId, action)
      await refreshGameState(gameId)
      
      if (result.done) {
        setGameState('finished')
      }
    } catch (e) {
      console.error('执行动作失败:', e)
    }
  }, [gameId, gameState])

  // 判断当前玩家是否是人类
  const isHumanTurn = session && players[session.current_player] === 'human'

  return (
    <div className="p-8">
      {/* 页面标题 */}
      <motion.div 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <h1 className="text-3xl font-bold font-display">对弈竞技场</h1>
        <p className="text-gray-400 mt-2">与 AI 对弈或观看 AI 之间的对决</p>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* 对弈设置 */}
        <div className="lg:col-span-1">
          <div className="bg-bg-card border border-bg-elevated rounded-xl p-6">
            <div className="flex items-center gap-3 mb-6">
              <Swords className="w-5 h-5 text-accent" />
              <h3 className="text-lg font-semibold font-display">对弈设置</h3>
            </div>

            {gameState === 'idle' ? (
              <>
                {/* 游戏选择 */}
                <div className="mb-6">
                  <label className="block text-sm text-gray-400 mb-3">
                    <Gamepad2 className="w-4 h-4 inline mr-2" />
                    选择游戏
                  </label>
                  <select
                    value={selectedGame}
                    onChange={e => setSelectedGame(e.target.value)}
                    className="w-full px-4 py-3 rounded-lg bg-bg-elevated border border-bg-hover text-white focus:border-accent focus:outline-none"
                  >
                    {availableGames.map(game => (
                      <option key={game.name} value={game.name}>
                        {game.name} ({game.num_players}人)
                      </option>
                    ))}
                  </select>
                </div>

                {/* 玩家选择 */}
                {players.map((player, idx) => (
                  <div key={idx} className="mb-6">
                    <label className="block text-sm text-gray-400 mb-3">
                      玩家 {idx + 1}
                    </label>
                    <select
                      value={player}
                      onChange={(e) => {
                        const newPlayers = [...players]
                        newPlayers[idx] = e.target.value
                        setPlayers(newPlayers)
                      }}
                      className={clsx(
                        'w-full px-4 py-3 rounded-lg border focus:outline-none transition-all',
                        idx === 0
                          ? 'bg-blue-900/20 border-blue-500/50 text-blue-400 focus:border-blue-400'
                          : 'bg-purple-900/20 border-purple-500/50 text-purple-400 focus:border-purple-400'
                      )}
                    >
                      <optgroup label="基础选项">
                        {basePlayerOptions.map(opt => (
                          <option key={opt.value} value={opt.value}>
                            {opt.label}
                          </option>
                        ))}
                      </optgroup>
                      {checkpoints.filter(cp => cp.game_type === selectedGame).length > 0 && (
                        <optgroup label="模型检查点">
                          {checkpoints
                            .filter(cp => cp.game_type === selectedGame)
                            .map(cp => (
                              <option key={cp.path} value={`checkpoint:${cp.path}`}>
                                {cp.algorithm} Epoch {cp.epoch}{cp.is_best ? ' (最佳)' : ''} - Loss: {cp.loss.toFixed(4)}
                              </option>
                            ))}
                        </optgroup>
                      )}
                    </select>
                  </div>
                ))}

                {/* 开始按钮 */}
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={handleStartGame}
                  className="w-full flex items-center justify-center gap-2 px-6 py-4 rounded-lg bg-accent text-black font-semibold hover:bg-accent-light transition-colors"
                >
                  <Play className="w-5 h-5" />
                  开始对弈
                </motion.button>
              </>
            ) : (
              <>
                {/* 对局进行中 */}
                <div className="mb-4 p-3 bg-bg-elevated rounded-lg text-center">
                  <span className="text-sm text-gray-400">游戏: </span>
                  <span className="text-accent font-medium">{selectedGame}</span>
                  </div>
                  
                <div className="space-y-3 mb-6">
                  {players.map((player, idx) => (
                    <div 
                      key={idx} 
                      className={clsx(
                        'p-4 rounded-lg transition-all',
                        session?.current_player === idx 
                          ? 'bg-accent/10 border border-accent/30' 
                          : 'bg-bg-elevated'
                      )}
                    >
                      <div className="text-sm text-gray-400 mb-1">
                        玩家 {idx + 1}
                        {session?.current_player === idx && (
                          <span className="ml-2 text-accent">● 当前</span>
                        )}
                      </div>
                      <div className={`font-medium ${idx === 0 ? 'text-blue-400' : 'text-purple-400'}`}>
                        {playerOptions.find(p => p.value === player)?.label}
                      </div>
                    </div>
                  ))}
                </div>

                <div className="space-y-3">
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-400">对局 ID</span>
                    <span className="font-mono text-accent">{gameId?.slice(0, 8)}</span>
                  </div>
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-400">状态</span>
                    <span className={clsx(
                      'px-2 py-1 rounded text-xs',
                      gameState === 'playing' && 'bg-accent/20 text-accent',
                      gameState === 'finished' && 'bg-gray-500/20 text-gray-400'
                    )}>
                      {gameState === 'playing' ? '进行中' : '已结束'}
                    </span>
                  </div>
                  <div className="flex items-center justify-between text-sm">
                    <span className="text-gray-400">步数</span>
                    <span className="font-mono">{session?.step_count || 0}</span>
                  </div>
                </div>

                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={handleReset}
                  className="w-full flex items-center justify-center gap-2 px-6 py-3 mt-6 rounded-lg border border-gray-600 text-gray-400 hover:bg-bg-hover transition-colors"
                >
                  <RotateCcw className="w-4 h-4" />
                  重新开始
                </motion.button>
              </>
            )}
          </div>
        </div>

        {/* 游戏画面 */}
        <div className="lg:col-span-2">
          <div className="bg-bg-card border border-bg-elevated rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold font-display">游戏画面</h3>
              {gameState === 'playing' && isHumanTurn && (
                <div className="text-sm text-accent animate-pulse">轮到你操作...</div>
              )}
            </div>
            
            <GameBoard 
              renderData={renderData}
              legalActions={legalActions}
              onAction={handleAction}
              interactive={gameState === 'playing' && isHumanTurn}
              gameType={selectedGame}
            />

            {/* 动作历史 */}
            {session && session.history.length > 0 && (
              <div className="mt-6">
                <div className="text-sm text-gray-400 mb-2">动作历史</div>
                <div className="bg-bg-elevated rounded-lg p-4 max-h-32 overflow-y-auto">
                  <div className="flex flex-wrap gap-2">
                    {session.history.map((action, i) => (
                      <span 
                        key={i}
                        className={`px-2 py-1 rounded text-xs font-mono ${
                          i % numPlayers === 0 
                            ? 'bg-blue-900/30 text-blue-400' 
                            : 'bg-purple-900/30 text-purple-400'
                        }`}
                      >
                        {i + 1}. #{action}
                      </span>
                    ))}
                  </div>
                </div>
              </div>
            )}

            {/* 对局结果 */}
            {session?.result && (
              <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                className="mt-6 p-6 bg-accent/10 border border-accent/30 rounded-lg text-center"
              >
                <div className="text-2xl font-bold font-display text-accent mb-2">
                  对局结束
                </div>
                <div className="text-gray-400">
                  {session.result.winner === null 
                    ? '平局' 
                    : `玩家 ${session.result.winner + 1} 获胜! 🎉`}
                </div>
              </motion.div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}


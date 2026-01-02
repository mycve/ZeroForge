import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { Play, Eye, Clock, Users, Gamepad2, Trash2 } from 'lucide-react'
import { GameBoard } from '../components/GameBoard'
import { listGameSessions, getGameSession, getGameRender, deleteGameSession } from '../utils/api'
import type { GameSession } from '../utils/api'

export default function GamesPage() {
  const [sessions, setSessions] = useState<GameSession[]>([])
  const [selectedSession, setSelectedSession] = useState<GameSession | null>(null)
  const [renderData, setRenderData] = useState<unknown>(null)
  const [loading, setLoading] = useState(true)

  // 加载会话列表
  useEffect(() => {
    async function loadSessions() {
      try {
        const data = await listGameSessions()
        setSessions(data)
      } catch (e) {
        console.error('加载会话列表失败:', e)
      } finally {
        setLoading(false)
      }
    }
    loadSessions()
    
    // 定时刷新
    const interval = setInterval(loadSessions, 5000)
    return () => clearInterval(interval)
  }, [])

  // 选中会话
  const handleSelectSession = async (id: string) => {
    try {
      const session = await getGameSession(id)
      setSelectedSession(session)
      
      // 获取渲染数据
      try {
        const { render } = await getGameRender(id, 'json')
        setRenderData(render)
      } catch {
        // 如果没有渲染接口，使用 session 自带的 render_data
        setRenderData(session.render_data)
      }
    } catch (e) {
      console.error('获取会话详情失败:', e)
    }
  }
  
  // 删除会话
  const handleDeleteSession = async (id: string, e: React.MouseEvent) => {
    e.stopPropagation() // 防止触发选中
    if (!confirm('确定要删除这个对局吗？')) return
    
    try {
      await deleteGameSession(id)
      setSessions(prev => prev.filter(s => s.id !== id))
      if (selectedSession?.id === id) {
        setSelectedSession(null)
        setRenderData(null)
      }
    } catch (err) {
      console.error('删除会话失败:', err)
    }
  }

  // 获取玩家显示名称
  const getPlayerName = (player: string, index: number): string => {
    if (player === 'human') return `玩家 ${index + 1}`
    if (player.startsWith('ai:')) return player.replace('ai:', 'AI-')
    return player
  }

  // 获取状态标签
  const getStateLabel = (state: string): string => {
    switch (state) {
      case 'playing': return '进行中'
      case 'finished': return '已结束'
      case 'waiting': return '等待中'
      default: return state
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
        <h1 className="text-3xl font-bold font-display">游戏可视化</h1>
        <p className="text-gray-400 mt-2">查看和观战正在进行的对局</p>
      </motion.div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
        {/* 会话列表 */}
        <div className="lg:col-span-1">
          <div className="bg-bg-card border border-bg-elevated rounded-xl p-6">
            <h3 className="text-lg font-semibold font-display mb-4">对局列表</h3>
            
            {loading ? (
              <div className="text-gray-500 py-8 text-center">加载中...</div>
            ) : sessions.length === 0 ? (
              <div className="text-gray-500 py-8 text-center">
                暂无进行中的对局
              </div>
            ) : (
              <div className="space-y-3">
                {sessions.map(session => (
                  <motion.div
                    key={session.id}
                    whileHover={{ scale: 1.02 }}
                    onClick={() => handleSelectSession(session.id)}
                    className={`p-4 rounded-lg border cursor-pointer transition-all ${
                      selectedSession?.id === session.id
                        ? 'bg-accent/10 border-accent/30'
                        : 'bg-bg-elevated border-bg-hover hover:border-accent/20'
                    }`}
                  >
                    <div className="flex items-center justify-between mb-2">
                      <div className="flex items-center gap-2">
                        <Gamepad2 className="w-4 h-4 text-accent" />
                        <span className="font-mono text-sm text-accent">{session.game_type}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className={`text-xs px-2 py-1 rounded ${
                          session.state === 'playing' 
                            ? 'bg-accent/20 text-accent' 
                            : session.state === 'finished'
                            ? 'bg-gray-500/20 text-gray-400'
                            : 'bg-warning/20 text-warning'
                        }`}>
                          {getStateLabel(session.state)}
                        </span>
                        <button
                          onClick={(e) => handleDeleteSession(session.id, e)}
                          className="p-1 rounded hover:bg-error/20 text-gray-500 hover:text-error transition-colors"
                          title="删除对局"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                    
                    <div className="flex items-center gap-2 text-sm text-gray-400">
                      <Users className="w-4 h-4" />
                      <span>
                        {session.players.map((p, i) => getPlayerName(p, i)).join(' vs ')}
                      </span>
                    </div>
                    
                    <div className="flex items-center gap-2 text-sm text-gray-500 mt-1">
                      <Clock className="w-4 h-4" />
                      <span>{session.step_count} 步</span>
                    </div>
                  </motion.div>
                ))}
              </div>
            )}
          </div>
        </div>

        {/* 游戏画面 */}
        <div className="lg:col-span-2">
          <div className="bg-bg-card border border-bg-elevated rounded-xl p-6">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold font-display">游戏画面</h3>
              {selectedSession && (
                <div className="flex items-center gap-2 text-sm text-gray-400">
                  <Eye className="w-4 h-4" />
                  观战模式
                </div>
              )}
            </div>
            
            {selectedSession ? (
              <>
                <GameBoard 
                  renderData={renderData}
                  interactive={false}
                  gameType={selectedSession.game_type}
                />
                
                {/* 对局信息 */}
                <div className="mt-6 grid grid-cols-2 gap-4">
                  {selectedSession.players.map((player, i) => (
                    <div key={i} className="p-4 bg-bg-elevated rounded-lg">
                      <div className="text-sm text-gray-400 mb-1">
                        玩家 {i + 1}
                        {selectedSession.current_player === i && (
                          <span className="ml-2 text-accent">● 当前</span>
                        )}
                      </div>
                      <div className={`font-medium ${i === 0 ? 'text-blue-400' : 'text-purple-400'}`}>
                        {getPlayerName(player, i)}
                  </div>
                  </div>
                  ))}
                </div>
                
                {/* 动作历史 */}
                <div className="mt-4">
                  <div className="text-sm text-gray-400 mb-2">动作历史</div>
                  <div className="bg-bg-elevated rounded-lg p-4 max-h-40 overflow-y-auto">
                    {selectedSession.history.length === 0 ? (
                      <span className="text-gray-500">暂无动作</span>
                    ) : (
                      <div className="flex flex-wrap gap-2">
                        {selectedSession.history.map((action, i) => (
                          <span 
                            key={i}
                            className={`px-2 py-1 rounded text-xs font-mono ${
                              i % 2 === 0 ? 'bg-blue-900/30 text-blue-400' : 'bg-purple-900/30 text-purple-400'
                            }`}
                          >
                            {i + 1}. #{action}
                          </span>
                        ))}
                      </div>
                    )}
                  </div>
                </div>
                
                {/* 对局结果 */}
                {selectedSession.result && (
                  <motion.div
                    initial={{ opacity: 0, scale: 0.9 }}
                    animate={{ opacity: 1, scale: 1 }}
                    className="mt-6 p-4 bg-accent/10 border border-accent/30 rounded-lg text-center"
                  >
                    <div className="text-xl font-bold font-display text-accent mb-1">
                      对局结束
                    </div>
                    <div className="text-gray-400">
                      {selectedSession.result.winner === null 
                        ? '平局' 
                        : `玩家 ${selectedSession.result.winner + 1} 获胜! 🎉`}
                    </div>
                  </motion.div>
                )}
              </>
            ) : (
              <div className="flex flex-col items-center justify-center py-20 text-gray-500">
                <Play className="w-12 h-12 mb-4 opacity-30" />
                <p>选择一局对局开始观战</p>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}


import { useState, useEffect, useCallback } from 'react';
import {
  createDebugSession,
  getDebugSession,
  listDebugSessions,
  deleteDebugSession,
  debugStepGame,
  debugStepMcts,
  debugResetGame,
  debugRunFullGame,
  listGames,
  listAlgorithms,
  listCheckpoints,
  type DebugSessionDetail,
} from '../utils/api';
import { motion } from 'framer-motion';
import { Bug, Play, RotateCcw, Zap, Search, Trash2 } from 'lucide-react';

interface DebugSession {
  session_id: string;
  game_type: string;
  algorithm: string;
  step: number;
  game_count: number;
  is_terminal: boolean;
}

interface SessionState {
  current_player: number | null;
  legal_actions: number[];
  is_terminal: boolean;
  winner: number | null;
  game_render: string | { type: string; text: string };
  observation_shape: number[];
  action_space: number;
}

// 解析 render 结果，统一返回字符串
function parseRender(render: string | { type: string; text: string } | null | undefined): string {
  if (!render) return '加载中...';
  if (typeof render === 'string') return render;
  if (typeof render === 'object' && 'text' in render) return render.text;
  return JSON.stringify(render);
}

interface MCTSResult {
  step: number;
  current_player: number;
  legal_actions: number[];
  selected_action: number;
  root_value: number;
  root_visits: number;
  policy: Record<number, number>;
  children: Array<{
    action: number;
    visit_count: number;
    value: number;
    prior: number;
    ucb: number;
  }>;
  game_render: string | { type: string; text: string };
}

interface StepResult {
  step: number;
  action: number;
  previous_player: number;
  current_player: number | null;
  is_terminal: boolean;
  winner: number | null;
  game_render: string | { type: string; text: string };
  policy: Record<number, number>;
  value: number;
  trajectory_summary?: {
    length: number;
    actions: number[];
    rewards: number[];
    values: number[];
  };
}

interface HistoryItem {
  type: 'step' | 'mcts';
  step: number;
  [key: string]: any;
}

export default function DebugPage() {
  // 游戏和算法列表
  const [games, setGames] = useState<Array<{ name: string; display_name?: string }>>([]);
  const [algorithms, setAlgorithms] = useState<Array<{ name: string }>>([]);
  const [checkpoints, setCheckpoints] = useState<Array<{ path: string; epoch: number }>>([]);
  
  // 创建会话表单
  const [selectedGame, setSelectedGame] = useState('tictactoe');
  const [selectedAlgo, setSelectedAlgo] = useState('alphazero');
  const [selectedCheckpoint, setSelectedCheckpoint] = useState<string>('');
  const [device, setDevice] = useState('cpu');
  
  // 当前会话
  const [sessions, setSessions] = useState<DebugSession[]>([]);
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);
  const [sessionState, setSessionState] = useState<SessionState | null>(null);
  const [history, setHistory] = useState<HistoryItem[]>([]);
  
  // MCTS 结果
  const [mctsResult, setMctsResult] = useState<MCTSResult | null>(null);
  const [lastStepResult, setLastStepResult] = useState<StepResult | null>(null);
  
  // 状态
  const [loading, setLoading] = useState(false);
  const [numSimulations, setNumSimulations] = useState(50);

  // 加载游戏和算法列表
  useEffect(() => {
    const loadMeta = async () => {
      try {
        const [gamesData, algosData] = await Promise.all([
          listGames(),
          listAlgorithms(),
        ]);
        setGames(gamesData);
        setAlgorithms(algosData);
      } catch (e) {
        console.error('加载元数据失败:', e);
      }
    };
    loadMeta();
  }, []);

  // 加载检查点列表
  useEffect(() => {
    const loadCheckpoints = async () => {
      try {
        const data = await listCheckpoints(selectedGame, selectedAlgo);
        setCheckpoints(data.checkpoints || []);
      } catch (e) {
        console.error('加载检查点失败:', e);
      }
    };
    if (selectedGame && selectedAlgo) {
      loadCheckpoints();
    }
  }, [selectedGame, selectedAlgo]);

  // 加载会话列表
  const loadSessions = useCallback(async () => {
    try {
      const data = await listDebugSessions();
      setSessions(data.sessions || []);
    } catch (e) {
      console.error('加载会话列表失败:', e);
    }
  }, []);

  useEffect(() => {
    loadSessions();
  }, [loadSessions]);

  // 加载活跃会话状态
  const loadActiveSession = useCallback(async () => {
    if (!activeSessionId) return;
    try {
      const data: DebugSessionDetail = await getDebugSession(activeSessionId);
      setSessionState(data.state);
      setHistory((data.history || []) as HistoryItem[]);
    } catch (e) {
      console.error('加载会话状态失败:', e);
    }
  }, [activeSessionId]);

  useEffect(() => {
    loadActiveSession();
  }, [loadActiveSession]);

  // 创建会话
  const handleCreateSession = async () => {
    setLoading(true);
    try {
      const data = await createDebugSession({
        game_type: selectedGame,
        algorithm: selectedAlgo,
        device,
        checkpoint_path: selectedCheckpoint || undefined,
      });
      await loadSessions();
      setActiveSessionId(data.session_id);
      setSessionState(data.state);
      setHistory([]);
      setMctsResult(null);
      setLastStepResult(null);
    } catch (e: any) {
      alert(`创建失败: ${e.message}`);
    } finally {
      setLoading(false);
    }
  };

  // 删除会话
  const handleDeleteSession = async (sessionId: string) => {
    try {
      await deleteDebugSession(sessionId);
      await loadSessions();
      if (activeSessionId === sessionId) {
        setActiveSessionId(null);
        setSessionState(null);
        setHistory([]);
        setMctsResult(null);
        setLastStepResult(null);
      }
    } catch (e: any) {
      alert(`删除失败: ${e.message}`);
    }
  };

  // 执行 MCTS
  const handleStepMcts = async () => {
    if (!activeSessionId) return;
    setLoading(true);
    try {
      const result = await debugStepMcts(activeSessionId, numSimulations);
      setMctsResult(result as MCTSResult);
      await loadActiveSession();
    } catch (e: any) {
      alert(`MCTS 失败: ${e.message}`);
    } finally {
      setLoading(false);
    }
  };

  // 执行一步（使用 MCTS 选择或指定动作）
  const handleStepGame = async (action?: number) => {
    if (!activeSessionId) return;
    setLoading(true);
    try {
      const result = await debugStepGame(activeSessionId, action);
      setLastStepResult(result as StepResult);
      setMctsResult(null);
      await loadActiveSession();
    } catch (e: any) {
      alert(`执行失败: ${e.message}`);
    } finally {
      setLoading(false);
    }
  };

  // 重置游戏
  const handleResetGame = async () => {
    if (!activeSessionId) return;
    setLoading(true);
    try {
      await debugResetGame(activeSessionId);
      setMctsResult(null);
      setLastStepResult(null);
      await loadActiveSession();
    } catch (e: any) {
      alert(`重置失败: ${e.message}`);
    } finally {
      setLoading(false);
    }
  };

  // 运行完整游戏
  const handleRunFullGame = async () => {
    if (!activeSessionId) return;
    setLoading(true);
    try {
      const result = await debugRunFullGame(activeSessionId) as { 
        steps: number; 
        final_result: StepResult; 
        all_actions: number[] 
      };
      setLastStepResult(result.final_result);
      setMctsResult(null);
      await loadActiveSession();
    } catch (e: any) {
      alert(`运行失败: ${e.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="p-8">
      {/* 页面标题 */}
      <motion.div 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <h1 className="text-3xl font-bold font-display flex items-center gap-3">
          <Bug className="w-8 h-8 text-accent" />
          训练调试
        </h1>
        <p className="text-gray-400 mt-2">逐步调试自玩流程、MCTS 搜索</p>
      </motion.div>

      <div className="grid grid-cols-12 gap-6">
        {/* 左侧：会话管理 */}
        <div className="col-span-3 space-y-4">
          <motion.div 
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            className="bg-bg-card border border-bg-elevated rounded-xl p-4"
          >
            <h3 className="font-bold mb-4 text-accent">创建调试会话</h3>
            
            <div className="space-y-4">
              <div>
                <label className="block text-sm text-gray-400 mb-1">游戏</label>
                <select
                  value={selectedGame}
                  onChange={(e) => setSelectedGame(e.target.value)}
                  className="w-full px-3 py-2 bg-bg-elevated border border-bg-hover rounded-lg text-white focus:border-accent focus:outline-none"
                >
                  {games.map((g) => (
                    <option key={g.name} value={g.name}>
                      {g.display_name || g.name}
                    </option>
                  ))}
                </select>
              </div>
              
              <div>
                <label className="block text-sm text-gray-400 mb-1">算法</label>
                <select
                  value={selectedAlgo}
                  onChange={(e) => setSelectedAlgo(e.target.value)}
                  className="w-full px-3 py-2 bg-bg-elevated border border-bg-hover rounded-lg text-white focus:border-accent focus:outline-none"
                >
                  {algorithms.map((a) => (
                    <option key={a.name} value={a.name}>{a.name}</option>
                  ))}
                </select>
              </div>
              
              <div>
                <label className="block text-sm text-gray-400 mb-1">检查点（可选）</label>
                <select
                  value={selectedCheckpoint}
                  onChange={(e) => setSelectedCheckpoint(e.target.value)}
                  className="w-full px-3 py-2 bg-bg-elevated border border-bg-hover rounded-lg text-white focus:border-accent focus:outline-none"
                >
                  <option value="">随机初始化</option>
                  {checkpoints.map((cp) => (
                    <option key={cp.path} value={cp.path}>
                      Epoch {cp.epoch}
                    </option>
                  ))}
                </select>
              </div>
              
              <div>
                <label className="block text-sm text-gray-400 mb-1">设备</label>
                <select
                  value={device}
                  onChange={(e) => setDevice(e.target.value)}
                  className="w-full px-3 py-2 bg-bg-elevated border border-bg-hover rounded-lg text-white focus:border-accent focus:outline-none"
                >
                  <option value="cpu">CPU</option>
                  <option value="cuda">CUDA</option>
                  <option value="mps">MPS</option>
                </select>
              </div>
              
              <button
                onClick={handleCreateSession}
                disabled={loading}
                className="w-full py-2.5 bg-accent text-black font-semibold rounded-lg hover:bg-accent-light transition-colors disabled:opacity-50"
              >
                创建会话
              </button>
            </div>
          </motion.div>

          {/* 会话列表 */}
          <motion.div 
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 }}
            className="bg-bg-card border border-bg-elevated rounded-xl p-4"
          >
            <h3 className="font-bold mb-4 text-accent">活跃会话</h3>
            {sessions.length === 0 ? (
              <p className="text-gray-500 text-sm text-center py-4">暂无会话</p>
            ) : (
              <div className="space-y-2">
                {sessions.map((s) => (
                  <div
                    key={s.session_id}
                    className={`p-3 rounded-lg cursor-pointer transition-all ${
                      activeSessionId === s.session_id
                        ? 'bg-accent/20 border border-accent'
                        : 'bg-bg-elevated hover:bg-bg-hover border border-transparent'
                    }`}
                    onClick={() => setActiveSessionId(s.session_id)}
                  >
                    <div className="flex justify-between items-start">
                      <div>
                        <div className="font-mono text-sm text-accent">{s.session_id}</div>
                        <div className="text-xs text-gray-400 mt-1">
                          {s.game_type} / {s.algorithm}
                        </div>
                        <div className="text-xs text-gray-500">
                          步数: {s.step} | 局数: {s.game_count}
                        </div>
                      </div>
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          handleDeleteSession(s.session_id);
                        }}
                        className="text-error hover:text-red-400 p-1"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </motion.div>
        </div>

        {/* 中间：游戏状态和控制 */}
        <div className="col-span-5 space-y-4">
          {!activeSessionId ? (
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="bg-bg-card border border-bg-elevated rounded-xl p-12 text-center"
            >
              <Bug className="w-16 h-16 text-gray-600 mx-auto mb-4" />
              <p className="text-gray-400">请选择或创建一个调试会话</p>
            </motion.div>
          ) : (
            <>
              {/* 游戏渲染 */}
              <motion.div 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="bg-bg-card border border-bg-elevated rounded-xl p-4"
              >
                <h3 className="font-bold mb-3 text-accent">游戏状态</h3>
                <pre className="font-mono text-sm bg-bg p-4 rounded-lg overflow-auto whitespace-pre text-accent border border-bg-elevated">
                  {parseRender(sessionState?.game_render)}
                </pre>
                <div className="mt-3 grid grid-cols-3 gap-4 text-sm">
                  <div className="bg-bg-elevated rounded-lg p-2 text-center">
                    <span className="text-gray-400 text-xs">当前玩家</span>
                    <div className="font-bold text-white">
                      {sessionState?.is_terminal ? '结束' : `P${sessionState?.current_player}`}
                    </div>
                  </div>
                  <div className="bg-bg-elevated rounded-lg p-2 text-center">
                    <span className="text-gray-400 text-xs">合法动作</span>
                    <div className="font-bold text-white">{sessionState?.legal_actions?.length || 0}</div>
                  </div>
                  <div className="bg-bg-elevated rounded-lg p-2 text-center">
                    <span className="text-gray-400 text-xs">胜者</span>
                    <div className={`font-bold ${
                      sessionState?.winner === null && sessionState?.is_terminal
                        ? 'text-gray-400'
                        : sessionState?.winner === 0
                        ? 'text-accent'
                        : 'text-error'
                    }`}>
                      {sessionState?.is_terminal
                        ? sessionState?.winner === null
                          ? '平局'
                          : `P${sessionState?.winner}`
                        : '-'}
                    </div>
                  </div>
                </div>
              </motion.div>

              {/* 控制面板 */}
              <motion.div 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="bg-bg-card border border-bg-elevated rounded-xl p-4"
              >
                <h3 className="font-bold mb-3 text-accent">控制面板</h3>
                
                <div className="flex items-center gap-4 mb-4">
                  <div className="flex items-center gap-2">
                    <label className="text-sm text-gray-400">MCTS 模拟次数:</label>
                    <input
                      type="number"
                      value={numSimulations}
                      onChange={(e) => setNumSimulations(parseInt(e.target.value) || 50)}
                      className="w-20 px-2 py-1 bg-bg-elevated border border-bg-hover rounded text-white focus:border-accent focus:outline-none"
                      min={1}
                      max={1000}
                    />
                  </div>
                </div>
                
                <div className="flex gap-2 flex-wrap">
                  <button
                    onClick={handleStepMcts}
                    disabled={loading || sessionState?.is_terminal}
                    className="flex items-center gap-2 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-500 transition-colors disabled:opacity-50"
                  >
                    <Search className="w-4 h-4" />
                    MCTS 搜索
                  </button>
                  <button
                    onClick={() => handleStepGame()}
                    disabled={loading || sessionState?.is_terminal}
                    className="flex items-center gap-2 px-4 py-2 bg-accent text-black font-semibold rounded-lg hover:bg-accent-light transition-colors disabled:opacity-50"
                  >
                    <Play className="w-4 h-4" />
                    执行一步
                  </button>
                  <button
                    onClick={handleRunFullGame}
                    disabled={loading || sessionState?.is_terminal}
                    className="flex items-center gap-2 px-4 py-2 bg-warning text-black font-semibold rounded-lg hover:bg-yellow-400 transition-colors disabled:opacity-50"
                  >
                    <Zap className="w-4 h-4" />
                    完整运行
                  </button>
                  <button
                    onClick={handleResetGame}
                    disabled={loading}
                    className="flex items-center gap-2 px-4 py-2 bg-bg-elevated text-white rounded-lg hover:bg-bg-hover transition-colors disabled:opacity-50"
                  >
                    <RotateCcw className="w-4 h-4" />
                    重置
                  </button>
                </div>

                {/* 手动选择动作 */}
                {sessionState && !sessionState.is_terminal && sessionState.legal_actions.length > 0 && (
                  <div className="mt-4">
                    <label className="block text-sm text-gray-400 mb-2">手动选择动作:</label>
                    <div className="flex gap-1 flex-wrap">
                      {sessionState.legal_actions.slice(0, 20).map((action) => (
                        <button
                          key={action}
                          onClick={() => handleStepGame(action)}
                          disabled={loading}
                          className="px-2 py-1 text-xs bg-bg-elevated text-white rounded hover:bg-accent hover:text-black transition-colors"
                        >
                          {action}
                        </button>
                      ))}
                      {sessionState.legal_actions.length > 20 && (
                        <span className="text-xs text-gray-500 self-center">
                          +{sessionState.legal_actions.length - 20} 更多
                        </span>
                      )}
                    </div>
                  </div>
                )}
              </motion.div>

              {/* 最后执行结果 */}
              {lastStepResult && (
                <motion.div 
                  initial={{ opacity: 0, y: 20 }}
                  animate={{ opacity: 1, y: 0 }}
                  className="bg-bg-card border border-bg-elevated rounded-xl p-4"
                >
                  <h3 className="font-bold mb-3 text-accent">执行结果</h3>
                  <div className="grid grid-cols-2 gap-4 text-sm">
                    <div className="bg-bg-elevated rounded-lg p-2">
                      <span className="text-gray-400">动作:</span>
                      <span className="ml-2 font-mono text-white">{lastStepResult.action}</span>
                    </div>
                    <div className="bg-bg-elevated rounded-lg p-2">
                      <span className="text-gray-400">价值:</span>
                      <span className={`ml-2 font-mono ${lastStepResult.value > 0 ? 'text-accent' : lastStepResult.value < 0 ? 'text-error' : 'text-white'}`}>
                        {lastStepResult.value}
                      </span>
                    </div>
                  </div>
                  {lastStepResult.trajectory_summary && (
                    <div className="mt-3 p-3 bg-warning/10 border border-warning/30 rounded-lg text-sm">
                      <div className="font-bold text-warning mb-2">游戏结束 - 轨迹摘要</div>
                      <div className="space-y-1 text-gray-300">
                        <div><span className="text-gray-400">长度:</span> {lastStepResult.trajectory_summary.length}</div>
                        <div><span className="text-gray-400">动作:</span> <span className="font-mono">[{lastStepResult.trajectory_summary.actions.join(', ')}]</span></div>
                        <div><span className="text-gray-400">奖励:</span> <span className="font-mono">[{lastStepResult.trajectory_summary.rewards.join(', ')}]</span></div>
                        <div><span className="text-gray-400">价值:</span> <span className="font-mono">[{lastStepResult.trajectory_summary.values.join(', ')}]</span></div>
                      </div>
                    </div>
                  )}
                </motion.div>
              )}
            </>
          )}
        </div>

        {/* 右侧：MCTS 详情和历史 */}
        <div className="col-span-4 space-y-4">
          {/* MCTS 搜索结果 */}
          {mctsResult && (
            <motion.div 
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className="bg-bg-card border border-bg-elevated rounded-xl p-4"
            >
              <h3 className="font-bold mb-3 text-purple-400">MCTS 搜索结果</h3>
              <div className="grid grid-cols-2 gap-2 text-sm mb-3">
                <div className="bg-bg-elevated rounded-lg p-2">
                  <span className="text-gray-400 text-xs">根节点访问</span>
                  <div className="font-mono text-white">{mctsResult.root_visits}</div>
                </div>
                <div className="bg-bg-elevated rounded-lg p-2">
                  <span className="text-gray-400 text-xs">根节点价值</span>
                  <div className={`font-mono ${mctsResult.root_value > 0 ? 'text-accent' : mctsResult.root_value < 0 ? 'text-error' : 'text-white'}`}>
                    {mctsResult.root_value}
                  </div>
                </div>
                <div className="bg-bg-elevated rounded-lg p-2">
                  <span className="text-gray-400 text-xs">推荐动作</span>
                  <div className="font-mono font-bold text-accent">{mctsResult.selected_action}</div>
                </div>
                <div className="bg-bg-elevated rounded-lg p-2">
                  <span className="text-gray-400 text-xs">合法动作数</span>
                  <div className="font-mono text-white">{mctsResult.legal_actions.length}</div>
                </div>
              </div>
              
              <div className="mb-3">
                <div className="text-sm text-gray-400 mb-2">策略分布 (Top 10):</div>
                <div className="flex gap-1 flex-wrap">
                  {Object.entries(mctsResult.policy).map(([action, prob]) => (
                    <span
                      key={action}
                      className={`px-2 py-1 rounded text-xs font-mono ${
                        parseInt(action) === mctsResult.selected_action
                          ? 'bg-accent text-black'
                          : 'bg-bg-elevated text-gray-300'
                      }`}
                    >
                      {action}: {(prob * 100).toFixed(1)}%
                    </span>
                  ))}
                </div>
              </div>

              <div>
                <div className="text-sm text-gray-400 mb-2">子节点详情:</div>
                <div className="max-h-48 overflow-auto">
                  <table className="w-full text-xs">
                    <thead className="sticky top-0 bg-bg-elevated">
                      <tr className="text-gray-400">
                        <th className="px-2 py-1 text-left">动作</th>
                        <th className="px-2 py-1 text-right">访问</th>
                        <th className="px-2 py-1 text-right">价值</th>
                        <th className="px-2 py-1 text-right">先验</th>
                        <th className="px-2 py-1 text-right">UCB</th>
                      </tr>
                    </thead>
                    <tbody>
                      {mctsResult.children.map((child) => (
                        <tr
                          key={child.action}
                          className={child.action === mctsResult.selected_action ? 'bg-accent/20' : 'hover:bg-bg-hover'}
                        >
                          <td className="px-2 py-1 font-mono text-white">{child.action}</td>
                          <td className="px-2 py-1 text-right text-gray-300">{child.visit_count}</td>
                          <td className={`px-2 py-1 text-right ${child.value > 0 ? 'text-accent' : child.value < 0 ? 'text-error' : 'text-gray-300'}`}>
                            {child.value.toFixed(3)}
                          </td>
                          <td className="px-2 py-1 text-right text-gray-300">{child.prior.toFixed(3)}</td>
                          <td className="px-2 py-1 text-right text-gray-300">{child.ucb.toFixed(3)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </motion.div>
          )}

          {/* 历史记录 */}
          <motion.div 
            initial={{ opacity: 0, x: 20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 }}
            className="bg-bg-card border border-bg-elevated rounded-xl p-4"
          >
            <h3 className="font-bold mb-3 text-accent">历史记录</h3>
            {history.length === 0 ? (
              <p className="text-gray-500 text-sm text-center py-4">暂无历史</p>
            ) : (
              <div className="max-h-96 overflow-auto space-y-2">
                {history.slice().reverse().map((item, idx) => (
                  <div
                    key={idx}
                    className={`p-2 rounded-lg text-xs border-l-2 ${
                      item.type === 'mcts'
                        ? 'bg-purple-900/20 border-purple-500'
                        : 'bg-accent/10 border-accent'
                    }`}
                  >
                    <div className="flex justify-between text-gray-300">
                      <span className="font-bold">
                        {item.type === 'mcts' ? 'MCTS' : '步骤'} #{item.step}
                      </span>
                      {item.type === 'step' && (
                        <span>动作: <span className="text-white">{item.action}</span></span>
                      )}
                    </div>
                    {item.type === 'step' && item.is_terminal && (
                      <div className="text-warning mt-1">
                        游戏结束 - {item.winner === null ? '平局' : `P${item.winner} 胜`}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </motion.div>
        </div>
      </div>
    </div>
  );
}

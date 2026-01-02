import { useEffect, useState } from 'react'
import { motion } from 'framer-motion'
import { Save, RefreshCw, Settings, Cpu, Brain, Gamepad2, Database, Zap, FolderOpen, Server } from 'lucide-react'
import { clsx } from 'clsx'
import { getConfig, updateConfigBatch, getConfigSchema, listGames, listAlgorithms } from '../utils/api'
import type { GameInfo, AlgorithmInfo, ConfigSchema, ConfigFieldMeta } from '../utils/api'
import { useToast } from '../components/Toast'

// 标签页定义
const tabs = [
  { id: 'game', label: '游戏', icon: Gamepad2 },
  { id: 'training', label: '训练', icon: Brain },
  { id: 'network', label: '网络', icon: Zap },
  { id: 'mcts', label: 'MCTS', icon: Settings },
  { id: 'gumbel', label: 'Gumbel', icon: Zap },  // Gumbel 搜索配置
  { id: 'selfplay', label: '自玩', icon: Cpu },
  { id: 'buffer', label: '缓冲区', icon: Database },
  { id: 'checkpoint', label: '检查点', icon: FolderOpen },
  { id: 'distributed', label: '分布式', icon: Server },
  { id: 'system', label: '系统', icon: Cpu },
]

export default function ConfigPage() {
  const [config, setConfig] = useState<Record<string, unknown> | null>(null)
  const [schema, setSchema] = useState<ConfigSchema | null>(null)
  const [games, setGames] = useState<GameInfo[]>([])
  const [algorithms, setAlgorithms] = useState<AlgorithmInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [activeTab, setActiveTab] = useState('game')

  // 加载数据
  useEffect(() => {
    async function loadData() {
      try {
        const [configData, schemaData, gamesData, algorithmsData] = await Promise.all([
          getConfig(),
          getConfigSchema().catch(() => null),
          listGames(),
          listAlgorithms(),
        ])
        setConfig(configData)
        setSchema(schemaData)
        setGames(gamesData)
        setAlgorithms(algorithmsData)
      } catch (e) {
        console.error('加载配置失败:', e)
      } finally {
        setLoading(false)
      }
    }
    loadData()
  }, [])

  // Toast 通知
  const toast = useToast()
  
  // 保存配置
  const handleSave = async () => {
    if (!config) return
    setSaving(true)
    try {
      await updateConfigBatch(config)
      toast.success('保存成功', '配置已更新')
    } catch (e) {
      console.error('保存配置失败:', e)
      const message = e instanceof Error ? e.message : '未知错误'
      toast.error('保存失败', message)
    } finally {
      setSaving(false)
    }
  }

  // 更新配置值
  const updateValue = (key: string, value: unknown) => {
    setConfig(prev => prev ? { ...prev, [key]: value } : prev)
  }

  // 渲染单个字段
  const renderField = (key: string, meta?: ConfigFieldMeta) => {
    if (!config) return null
    
    const value = config[key]
    const label = meta?.label || key
    const type = meta?.type || 'string'
    
    // 特殊处理游戏类型选择
    if (key === 'game_type') {
      return (
        <div key={key}>
          <label className="block text-sm text-gray-400 mb-2">{label}</label>
          <select
            value={value as string || 'tictactoe'}
            onChange={e => updateValue(key, e.target.value)}
            className="w-full px-4 py-3 rounded-lg bg-bg-elevated border border-bg-hover text-white focus:border-accent focus:outline-none"
          >
            {games.map(game => (
              <option key={game.name} value={game.name}>{game.name}</option>
            ))}
          </select>
        </div>
      )
    }
    
    // 特殊处理算法选择
    if (key === 'algorithm') {
      return (
        <div key={key}>
          <label className="block text-sm text-gray-400 mb-2">{label}</label>
          <select
            value={value as string || 'alphazero'}
            onChange={e => updateValue(key, e.target.value)}
            className="w-full px-4 py-3 rounded-lg bg-bg-elevated border border-bg-hover text-white focus:border-accent focus:outline-none"
          >
            {algorithms.map(algo => (
              <option key={algo.name} value={algo.name}>{algo.name}</option>
            ))}
          </select>
        </div>
      )
    }
    
    // 选择框
    if (type === 'select' && meta?.options) {
      return (
        <div key={key}>
          <label className="block text-sm text-gray-400 mb-2">{label}</label>
          <select
            value={value as string || meta.options[0]}
            onChange={e => updateValue(key, e.target.value)}
            className="w-full px-4 py-3 rounded-lg bg-bg-elevated border border-bg-hover text-white focus:border-accent focus:outline-none"
          >
            {meta.options.map(opt => (
              <option key={opt} value={opt}>{opt}</option>
            ))}
          </select>
        </div>
      )
    }
    
    // 多选框
    if (type === 'multiselect' && meta?.options) {
      const currentValues = (value as string[]) || []
      return (
        <div key={key}>
          <label className="block text-sm text-gray-400 mb-2">{label}</label>
          <div className="flex flex-wrap gap-3">
            {meta.options.map(opt => (
              <label key={opt} className="flex items-center gap-2 cursor-pointer">
                <input
                  type="checkbox"
                  checked={currentValues.includes(opt)}
                  onChange={e => {
                    const newValues = e.target.checked
                      ? [...currentValues, opt]
                      : currentValues.filter(v => v !== opt)
                    updateValue(key, newValues)
                  }}
                  className="w-4 h-4 rounded border-bg-hover bg-bg-elevated text-accent focus:ring-accent"
                />
                <span className="text-sm">{opt}</span>
              </label>
            ))}
          </div>
          {meta?.description && (
            <p className="text-xs text-gray-500 mt-1">{meta.description}</p>
          )}
        </div>
      )
    }
    
    // 布尔开关
    if (type === 'bool') {
      return (
        <div key={key}>
          <label className="flex items-center gap-3 cursor-pointer">
            <div className="relative">
              <input
                type="checkbox"
                checked={value as boolean || false}
                onChange={e => updateValue(key, e.target.checked)}
                className="sr-only peer"
              />
              <div className="w-11 h-6 bg-bg-elevated rounded-full peer peer-checked:bg-accent transition-colors border border-bg-hover"></div>
              <div className="absolute left-1 top-1 w-4 h-4 bg-gray-400 rounded-full transition-transform peer-checked:translate-x-5 peer-checked:bg-white"></div>
            </div>
            <span className="text-sm text-white">{label}</span>
          </label>
          {meta?.description && (
            <p className="text-xs text-gray-500 mt-1 ml-14">{meta.description}</p>
          )}
        </div>
      )
    }
    
    // 整数输入
    if (type === 'int') {
      return (
        <div key={key}>
          <label className="block text-sm text-gray-400 mb-2">{label}</label>
          <input
            type="number"
            value={value as number || 0}
            min={meta?.min}
            max={meta?.max}
            onChange={e => updateValue(key, parseInt(e.target.value) || 0)}
            className="w-full px-4 py-3 rounded-lg bg-bg-elevated border border-bg-hover text-white focus:border-accent focus:outline-none"
          />
          {meta?.description && (
            <p className="text-xs text-gray-500 mt-1">{meta.description}</p>
          )}
        </div>
      )
    }
    
    // 浮点数输入
    if (type === 'float') {
      return (
        <div key={key}>
          <label className="block text-sm text-gray-400 mb-2">{label}</label>
          <input
            type="number"
            step={meta?.step || 0.001}
            value={value as number || 0}
            min={meta?.min}
            max={meta?.max}
            onChange={e => updateValue(key, parseFloat(e.target.value) || 0)}
            className="w-full px-4 py-3 rounded-lg bg-bg-elevated border border-bg-hover text-white focus:border-accent focus:outline-none"
          />
          {meta?.description && (
            <p className="text-xs text-gray-500 mt-1">{meta.description}</p>
          )}
        </div>
      )
    }
    
    // 字符串输入（默认）
    return (
      <div key={key}>
        <label className="block text-sm text-gray-400 mb-2">{label}</label>
        <input
          type="text"
          value={value as string || ''}
          onChange={e => updateValue(key, e.target.value)}
          className="w-full px-4 py-3 rounded-lg bg-bg-elevated border border-bg-hover text-white focus:border-accent focus:outline-none"
        />
      </div>
    )
  }

  // 渲染配置组
  const renderGroup = (groupId: string) => {
    if (!config || !schema) return null
    
    const group = schema.groups[groupId]
    if (!group) return null
    
    const fields = group.fields.filter(f => f in config)
    if (fields.length === 0) return null
    
    return (
      <div className="space-y-6">
        <h3 className="text-lg font-semibold font-display">{group.label}</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {fields.map(field => renderField(field, schema.fields[field]))}
        </div>
      </div>
    )
  }

  if (loading) {
    return (
      <div className="p-8 flex items-center justify-center h-screen">
        <RefreshCw className="w-8 h-8 text-accent animate-spin" />
      </div>
    )
  }

  return (
    <div className="p-8">
      {/* 页面标题 */}
      <motion.div 
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8 flex items-center justify-between"
      >
        <div>
          <h1 className="text-3xl font-bold font-display">训练配置</h1>
          <p className="text-gray-400 mt-2">配置游戏、算法、网络和训练参数</p>
        </div>
        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={handleSave}
          disabled={saving}
          className="flex items-center gap-2 px-6 py-3 rounded-lg bg-accent text-black font-semibold hover:bg-accent-light transition-colors disabled:opacity-50"
        >
          {saving ? (
            <RefreshCw className="w-5 h-5 animate-spin" />
          ) : (
            <Save className="w-5 h-5" />
          )}
          保存配置
        </motion.button>
      </motion.div>

      {/* 标签页 */}
      <div className="flex flex-wrap gap-2 mb-6 border-b border-bg-elevated pb-4">
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={clsx(
              'flex items-center gap-2 px-4 py-2 rounded-lg transition-colors',
              activeTab === tab.id
                ? 'bg-accent/10 text-accent border border-accent/30'
                : 'text-gray-400 hover:bg-bg-hover hover:text-white'
            )}
          >
            <tab.icon className="w-4 h-4" />
            {tab.label}
          </button>
        ))}
      </div>

      {/* 配置内容 */}
      <motion.div
        key={activeTab}
        initial={{ opacity: 0, x: 20 }}
        animate={{ opacity: 1, x: 0 }}
        className="bg-bg-card border border-bg-elevated rounded-xl p-6"
      >
        {schema ? (
          renderGroup(activeTab)
        ) : (
          <div className="text-gray-400">加载配置 schema 失败，使用原始编辑模式</div>
        )}
        
        {/* 可用游戏信息（仅在游戏标签显示） */}
        {activeTab === 'game' && (
          <div className="mt-6 p-4 bg-bg-elevated rounded-lg">
            <h4 className="text-sm font-medium text-gray-400 mb-3">可用游戏</h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {games.map(game => (
                <div key={game.name} className="p-3 bg-bg-card rounded-lg border border-bg-hover">
                  <div className="font-medium">{game.name}</div>
                  <div className="text-xs text-gray-500 mt-1">
                    {game.observation_space && `观测: ${JSON.stringify(game.observation_space.shape)}`}
                    {game.action_space && ` | 动作: ${game.action_space.n}`}
                    {` | 玩家: ${game.num_players}`}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </motion.div>
    </div>
  )
}

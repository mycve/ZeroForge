import { Routes, Route, NavLink } from 'react-router-dom'
import { motion } from 'framer-motion'
import { 
  LayoutDashboard, 
  Settings, 
  Gamepad2, 
  Swords,
  Activity,
  Server,
  Bug
} from 'lucide-react'
import { clsx } from 'clsx'
import Dashboard from './pages/Dashboard'
import ConfigPage from './pages/ConfigPage'
import GamesPage from './pages/GamesPage'
import ArenaPage from './pages/ArenaPage'
import DebugPage from './pages/DebugPage'
import { ToastProvider } from './components/Toast'

// 导航项
const navItems = [
  { path: '/', icon: LayoutDashboard, label: '仪表盘' },
  { path: '/config', icon: Settings, label: '配置' },
  { path: '/games', icon: Gamepad2, label: '游戏可视化' },
  { path: '/arena', icon: Swords, label: '对弈竞技场' },
  { path: '/debug', icon: Bug, label: '训练调试' },
]

function App() {
  return (
    <ToastProvider>
    <div className="min-h-screen bg-bg chess-grid">
      {/* 侧边栏 */}
      <aside className="fixed left-0 top-0 h-screen w-64 bg-bg-card border-r border-bg-elevated z-50">
        {/* Logo */}
        <div className="h-16 flex items-center px-6 border-b border-bg-elevated">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-accent/20 flex items-center justify-center">
              <Activity className="w-5 h-5 text-accent" />
            </div>
            <span className="font-display font-semibold text-lg">RL Framework</span>
          </div>
        </div>

        {/* 导航 */}
        <nav className="p-4 space-y-2">
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) =>
                clsx(
                  'flex items-center gap-3 px-4 py-3 rounded-lg transition-all',
                  isActive
                    ? 'bg-accent/10 text-accent border border-accent/30'
                    : 'text-gray-400 hover:bg-bg-hover hover:text-white'
                )
              }
            >
              <item.icon className="w-5 h-5" />
              <span className="font-medium">{item.label}</span>
            </NavLink>
          ))}
        </nav>

        {/* 底部状态 */}
        <div className="absolute bottom-0 left-0 right-0 p-4 border-t border-bg-elevated">
          <div className="flex items-center gap-3 px-4 py-3 bg-bg-elevated rounded-lg">
            <Server className="w-5 h-5 text-accent" />
            <div>
              <div className="text-sm font-medium">后端连接</div>
              <div className="text-xs text-accent">● 已连接</div>
            </div>
          </div>
        </div>
      </aside>

      {/* 主内容 */}
      <main className="ml-64 min-h-screen">
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.3 }}
        >
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/config" element={<ConfigPage />} />
            <Route path="/games" element={<GamesPage />} />
            <Route path="/arena" element={<ArenaPage />} />
            <Route path="/debug" element={<DebugPage />} />
          </Routes>
        </motion.div>
      </main>
    </div>
    </ToastProvider>
  )
}

export default App


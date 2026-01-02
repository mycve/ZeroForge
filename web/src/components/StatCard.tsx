import { motion } from 'framer-motion'
import { LucideIcon } from 'lucide-react'
import { clsx } from 'clsx'

interface StatCardProps {
  title: string
  value: string | number
  subtitle?: string
  icon: LucideIcon
  color?: 'accent' | 'warning' | 'error' | 'purple'
  trend?: 'up' | 'down' | 'neutral'
  trendValue?: string
}

const colorMap = {
  accent: {
    bg: 'bg-accent/10',
    text: 'text-accent',
    border: 'border-accent/30',
    glow: 'shadow-accent/20',
  },
  warning: {
    bg: 'bg-warning/10',
    text: 'text-warning',
    border: 'border-warning/30',
    glow: 'shadow-warning/20',
  },
  error: {
    bg: 'bg-error/10',
    text: 'text-error',
    border: 'border-error/30',
    glow: 'shadow-error/20',
  },
  purple: {
    bg: 'bg-purple-500/10',
    text: 'text-purple-400',
    border: 'border-purple-500/30',
    glow: 'shadow-purple-500/20',
  },
}

export function StatCard({
  title,
  value,
  subtitle,
  icon: Icon,
  color = 'accent',
  trend,
  trendValue,
}: StatCardProps) {
  const colors = colorMap[color]

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      whileHover={{ scale: 1.02 }}
      className={clsx(
        'p-6 rounded-xl bg-bg-card border card-hover',
        colors.border,
        `hover:shadow-lg hover:${colors.glow}`
      )}
    >
      <div className="flex items-start justify-between">
        <div className="space-y-2">
          <p className="text-sm text-gray-400 font-medium">{title}</p>
          <p className={clsx('text-3xl font-bold font-display', colors.text)}>
            {typeof value === 'number' ? value.toLocaleString() : value}
          </p>
          {subtitle && (
            <p className="text-xs text-gray-500">{subtitle}</p>
          )}
          {trend && trendValue && (
            <div className="flex items-center gap-1 text-xs">
              <span
                className={clsx(
                  trend === 'up' && 'text-accent',
                  trend === 'down' && 'text-error',
                  trend === 'neutral' && 'text-gray-400'
                )}
              >
                {trend === 'up' ? '↑' : trend === 'down' ? '↓' : '→'}
              </span>
              <span className="text-gray-400">{trendValue}</span>
            </div>
          )}
        </div>
        <div className={clsx('p-3 rounded-lg', colors.bg)}>
          <Icon className={clsx('w-6 h-6', colors.text)} />
        </div>
      </div>
    </motion.div>
  )
}


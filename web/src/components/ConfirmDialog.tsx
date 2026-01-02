import { motion, AnimatePresence } from 'framer-motion'
import { AlertTriangle, X } from 'lucide-react'

interface ConfirmDialogProps {
  isOpen: boolean
  title: string
  message: string
  confirmText?: string
  cancelText?: string
  onConfirm: () => void
  onCancel: () => void
  type?: 'danger' | 'warning' | 'info'
}

const typeStyles = {
  danger: {
    icon: 'text-red-400',
    button: 'bg-red-500 hover:bg-red-600',
  },
  warning: {
    icon: 'text-yellow-400',
    button: 'bg-yellow-500 hover:bg-yellow-600',
  },
  info: {
    icon: 'text-blue-400',
    button: 'bg-blue-500 hover:bg-blue-600',
  },
}

export function ConfirmDialog({
  isOpen,
  title,
  message,
  confirmText = '确定',
  cancelText = '取消',
  onConfirm,
  onCancel,
  type = 'danger',
}: ConfirmDialogProps) {
  const styles = typeStyles[type]

  return (
    <AnimatePresence>
      {isOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center">
          {/* 遮罩层 */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onCancel}
            className="absolute inset-0 bg-black/60 backdrop-blur-sm"
          />
          
          {/* 对话框 */}
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 20 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 20 }}
            className="relative w-full max-w-md mx-4"
          >
            <div className="bg-bg-card border border-bg-elevated rounded-xl shadow-2xl overflow-hidden">
              {/* 头部 */}
              <div className="flex items-center justify-between p-4 border-b border-bg-elevated">
                <div className="flex items-center gap-3">
                  <AlertTriangle className={`w-5 h-5 ${styles.icon}`} />
                  <h3 className="text-lg font-semibold">{title}</h3>
                </div>
                <button
                  onClick={onCancel}
                  className="p-1 rounded-lg hover:bg-bg-hover transition-colors text-gray-400 hover:text-white"
                >
                  <X className="w-5 h-5" />
                </button>
              </div>
              
              {/* 内容 */}
              <div className="p-4">
                <p className="text-gray-300">{message}</p>
              </div>
              
              {/* 按钮 */}
              <div className="flex justify-end gap-3 p-4 bg-bg-elevated/50">
                <button
                  onClick={onCancel}
                  className="px-4 py-2 rounded-lg bg-bg-hover text-gray-300 hover:bg-bg-elevated hover:text-white transition-colors"
                >
                  {cancelText}
                </button>
                <button
                  onClick={onConfirm}
                  className={`px-4 py-2 rounded-lg text-white font-medium transition-colors ${styles.button}`}
                >
                  {confirmText}
                </button>
              </div>
            </div>
          </motion.div>
        </div>
      )}
    </AnimatePresence>
  )
}

// Hook 用于简化使用
import { useState, useCallback } from 'react'

export function useConfirmDialog() {
  const [isOpen, setIsOpen] = useState(false)
  const [config, setConfig] = useState<{
    title: string
    message: string
    onConfirm: () => void
    type?: 'danger' | 'warning' | 'info'
  } | null>(null)

  const confirm = useCallback((options: {
    title: string
    message: string
    type?: 'danger' | 'warning' | 'info'
  }): Promise<boolean> => {
    return new Promise((resolve) => {
      setConfig({
        ...options,
        onConfirm: () => {
          setIsOpen(false)
          resolve(true)
        },
      })
      setIsOpen(true)
    })
  }, [])

  const handleCancel = useCallback(() => {
    setIsOpen(false)
  }, [])

  const DialogComponent = config ? (
    <ConfirmDialog
      isOpen={isOpen}
      title={config.title}
      message={config.message}
      type={config.type}
      onConfirm={config.onConfirm}
      onCancel={handleCancel}
    />
  ) : null

  return { confirm, DialogComponent }
}

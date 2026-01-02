/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        // 自定义深色主题
        bg: {
          DEFAULT: '#0a0a0f',
          card: '#12121a',
          elevated: '#1a1a24',
          hover: '#242432',
        },
        accent: {
          DEFAULT: '#00d4aa',
          light: '#00ffcc',
          dark: '#00a080',
        },
        warning: '#ffc107',
        error: '#ff4757',
        success: '#00d4aa',
      },
      fontFamily: {
        sans: ['JetBrains Mono', 'SF Mono', 'Fira Code', 'monospace'],
        display: ['Space Grotesk', 'Inter', 'sans-serif'],
      },
      animation: {
        'pulse-slow': 'pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
      },
      keyframes: {
        glow: {
          '0%': { boxShadow: '0 0 5px #00d4aa, 0 0 10px #00d4aa' },
          '100%': { boxShadow: '0 0 20px #00d4aa, 0 0 30px #00d4aa' },
        },
      },
    },
  },
  plugins: [],
}


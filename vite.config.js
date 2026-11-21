import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: process.env.VITE_BASE || "/kopernik/",
  server: {
    host: '127.0.0.1',
    port: 5173,
    strictPort: true,
    open: false,
    proxy: {
      '/api': 'http://127.0.0.1:8000',
      '/output': 'http://127.0.0.1:8000',
    },
    watch: {
      ignored: ['**/.env']
    }
  }
})
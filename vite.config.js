import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  base: "/kopernik/",
  server: {
    host: '127.0.0.1',
    port: 5173,
    strictPort: true,
    open: false,
    watch: {
      ignored: ['**/.env']
    }
  }
})
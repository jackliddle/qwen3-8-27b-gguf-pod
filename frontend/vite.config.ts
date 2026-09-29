import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// Dev: `MP_DEV=1 API_KEY=dev uvicorn app.main:app` in supervisor/, then `npm run dev`.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': 'http://localhost:8000',
      '/v1': 'http://localhost:8000',
    },
  },
})

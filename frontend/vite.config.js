import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { VitePWA } from 'vite-plugin-pwa'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    vue(),
    VitePWA({
      registerType: 'autoUpdate',
      includeAssets: ['favicon.ico'],
      manifest: {
        name: 'พยากรณ์ PM2.5 - ภาคอีสาน',
        short_name: 'PM2.5 ภาคอีสาน',
        description: 'พยากรณ์ PM2.5 ล่วงหน้า 24 ชั่วโมง 20 จังหวัดภาคอีสาน ด้วย Dynamic Routing (LightGBM/XGBoost)',
        lang: 'th',
        start_url: '/',
        display: 'standalone',
        background_color: '#c9d9e6',
        theme_color: '#6b7f96',
        icons: [
          { src: 'pwa-192x192.png', sizes: '192x192', type: 'image/png' },
          { src: 'pwa-512x512.png', sizes: '512x512', type: 'image/png' },
          { src: 'pwa-512x512.png', sizes: '512x512', type: 'image/png', purpose: 'maskable' },
        ],
      },
      workbox: {
        // ไม่ cache /api/* แบบ offline-first เพราะข้อมูลพยากรณ์ต้องเป็นข้อมูลสดเสมอ
        // ปล่อยให้ fetch ตรงไป network ทุกครั้ง ตัว service worker ดูแลแค่ static assets (JS/CSS/HTML shell)
        navigateFallbackDenylist: [/^\/api\//],
      },
    }),
  ],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url)),
    },
  },
  server: {
    proxy: {
      '/api': 'http://localhost:5000',
    },
  },
})

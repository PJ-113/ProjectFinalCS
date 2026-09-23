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
        runtimeCaching: [
          {
            // แผนที่ (/map) โหลด tile จาก OpenStreetMap — เป็นภาพนิ่งไม่เปลี่ยนบ่อย cache ยาวได้ปลอดภัย
            // ต่างจาก /api/* ด้านบนที่ต้องสดเสมอ, ลด request ซ้ำและช่วยให้ใช้ offline ได้บางส่วน
            urlPattern: /^https:\/\/[abc]\.tile\.openstreetmap\.org\/.*/i,
            handler: "CacheFirst",
            options: {
              cacheName: "osm-tiles-cache",
              expiration: {
                maxEntries: 200,
                maxAgeSeconds: 30 * 24 * 60 * 60,
              },
              cacheableResponse: { statuses: [0, 200] },
            },
          },
        ],
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

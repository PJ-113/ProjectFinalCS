import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { VitePWA } from 'vite-plugin-pwa'

// https://vite.dev/config/
export default defineConfig({
  plugins: [
    vue(),
    VitePWA({
      // injectManifest (ไม่ใช่ generateSW เดิม) เพราะต้องมี service worker ของเราเอง (src/sw.js) ที่รับ
      // push/notificationclick event ได้ — generateSW ไม่เปิดช่องให้เขียน event listener เองแบบนี้
      // (dry-run ยืนยันแล้วว่า precache manifest ออกมาเหมือนเดิมทุกไฟล์ ดู CHANGELOG ตอน Web Push)
      strategies: 'injectManifest',
      srcDir: 'src',
      filename: 'sw.js',
      injectManifest: {
        // ไฟล์ขนาดใหญ่ (เช่น model-adjacent bundle, isaan boundary/mask JSON) เกิน default 2MB ของ
        // injectManifest ถ้าไม่ขยายจะ build fail แบบเงียบๆตอนรายการไฟล์ใหญ่กว่า limit
        maximumFileSizeToCacheInBytes: 5 * 1024 * 1024,
      },
      registerType: 'autoUpdate',
      includeAssets: ['favicon.ico'],
      manifest: {
        name: 'DustCast24 - พยากรณ์ PM2.5 ภาคอีสาน',
        short_name: 'DustCast24',
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
      // หมายเหตุ: ไม่มี key `workbox` อีกแล้ว (เป็น option เฉพาะ generateSW) — OSM tile caching +
      // /api navigateFallback denylist ที่เคยอยู่ตรงนี้ ย้ายไปเขียนเองใน src/sw.js แทน
      // (เคยเปิด devOptions.enabled ไว้เผื่อทดสอบ sw.js ผ่าน `npm run dev` แต่ไม่เคยได้ใช้จริง — ตอนทดสอบ
      // Web Push ทั้งหมดทำผ่าน `npm run build` + Flask serve ตรงๆ เท่านั้น พอเปิดจริงกลับทำให้ ServiceWorker
      // registration พังใน dev mode ปกติ เอาออกเพราะไม่คุ้มและไม่จำเป็น)
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

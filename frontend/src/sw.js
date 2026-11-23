// Custom service worker (injectManifest mode — ดู vite.config.js) แทนที่ generateSW เดิม
// เหตุผลที่ต้องย้ายมาเขียนเอง: generateSW ไม่เปิดช่องให้เพิ่ม event listener เอง (push/notificationclick)
// ทุกอย่างที่ workbox: {...} เคยทำให้ฟรีใน generateSW ต้องประกอบเองที่นี่ทั้งหมด

import { precacheAndRoute, createHandlerBoundToURL } from 'workbox-precaching'
import { registerRoute, NavigationRoute } from 'workbox-routing'
import { CacheFirst } from 'workbox-strategies'
import { ExpirationPlugin } from 'workbox-expiration'
import { CacheableResponsePlugin } from 'workbox-cacheable-response'
import { clientsClaim } from 'workbox-core'

// registerType: 'autoUpdate' เดิมได้พฤติกรรม "เปิดใช้ service worker ใหม่ทันที" มาฟรีจาก generateSW —
// ต้องสั่งเองสองบรรทัดนี้ถึงจะได้พฤติกรรมเดิม
self.skipWaiting()
clientsClaim()

// precache manifest ถูกแทนที่ตรงนี้ตอน build โดย vite-plugin-pwa (self.__WB_MANIFEST)
precacheAndRoute(self.__WB_MANIFEST)

// /api/* ต้องเป็นข้อมูลสดเสมอ ไม่ cache แบบ offline-first — ปล่อยให้ fetch ตรงไป network ทุกครั้ง
// service worker ดูแลแค่ static assets (JS/CSS/HTML shell) เท่านั้น (เดิมคือ navigateFallbackDenylist)
registerRoute(
  new NavigationRoute(createHandlerBoundToURL('/index.html'), {
    denylist: [/^\/api\//],
  })
)

// แผนที่ (/map) โหลด tile จาก OpenStreetMap — เป็นภาพนิ่งไม่เปลี่ยนบ่อย cache ยาวได้ปลอดภัย
// ต่างจาก /api/* ด้านบนที่ต้องสดเสมอ, ลด request ซ้ำและช่วยให้ใช้ offline ได้บางส่วน
registerRoute(
  /^https:\/\/[abc]\.tile\.openstreetmap\.org\/.*/i,
  new CacheFirst({
    cacheName: 'osm-tiles-cache',
    plugins: [
      new ExpirationPlugin({ maxEntries: 200, maxAgeSeconds: 30 * 24 * 60 * 60 }),
      new CacheableResponsePlugin({ statuses: [0, 200] }),
    ],
  })
)

// ===== Web Push =====
// payload ถูกส่งมาจาก app.py:send_push() เป็น JSON {title, body, tag, url} — เลียนข้อความจาก
// frontend/src/notifications.js builder ตัวเดียวกันเป๊ะ (ดู CHANGELOG ตอน Web Push)
self.addEventListener('push', (event) => {
  const data = event.data ? event.data.json() : {}
  event.waitUntil(
    self.registration.showNotification(data.title || 'พยากรณ์ PM2.5', {
      body: data.body || '',
      icon: '/pwa-192x192.png',
      badge: '/pwa-192x192.png',
      tag: data.tag, // ชื่อเงื่อนไข (threshold/transition/imminent/trend/silence) — กันซ้อนถ้าเผลอส่งซ้ำ
      data: { url: data.url || '/notifications' },
    })
  )
})

// คลิกแจ้งเตือน -> โฟกัสแท็บที่เปิดอยู่ถ้ามี ไม่งั้นเปิดแท็บใหม่ไปหน้า /notifications
self.addEventListener('notificationclick', (event) => {
  event.notification.close()
  const url = event.notification.data?.url || '/notifications'
  event.waitUntil(
    self.clients.matchAll({ type: 'window', includeUncontrolled: true }).then((clientList) => {
      for (const client of clientList) {
        if (client.url.includes(self.location.origin) && 'focus' in client) {
          return client.navigate(url).then(() => client.focus())
        }
      }
      return self.clients.openWindow(url)
    })
  )
})

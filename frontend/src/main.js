import './assets/main.css'
import './assets/auth-theme.css'

import { createApp } from 'vue'
import App from './App.vue'
import router from './router'
import { initTheme } from './theme'
import { initLocale } from './i18n'

initTheme()
initLocale()

// reload อัตโนมัติเมื่อ service worker เวอร์ชันใหม่เข้าควบคุมหน้า (sw.js มี skipWaiting()+clientsClaim()
// อยู่แล้ว แต่ถ้าไม่ reload แท็บที่เปิดค้างไว้จะยังรัน JS เวอร์ชันเก่าต่อ ทำให้ asset/manifest ไม่ตรงกับ
// ไฟล์ที่ build ใหม่ล่าสุด แล้วขึ้นค้าง/ข้อมูลไม่ขึ้น) — กันปัญหาต้องเข้า DevTools ไป unregister เองทุกครั้ง
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.addEventListener('controllerchange', () => {
    window.location.reload()
  })
}

const app = createApp(App)

app.use(router)

app.mount('#app')

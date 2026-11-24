import { onMounted, ref, toValue } from 'vue'
import { subscribeToPush, unsubscribeFromPush, getExistingPushSubscription } from './notifications'
import { t } from './i18n'

// สถานะ Web Push ใช้ร่วมกันระหว่าง PushPrompt.vue (การ์ดในหน้าแจ้งเตือน) และสวิตช์ในหน้าตั้งค่า
// (SettingsView.vue) — แยกออกมาที่เดียวกันไม่ให้ logic ตรวจ iOS/permission/subscription เขียนซ้ำสองที่
// status: 'checking' | 'unsupported' | 'denied' | 'ios-not-installed' | 'subscribable' | 'subscribed'
// pushedProvince: จังหวัดที่ subscription ของเครื่องนี้ล็อกไว้จริงในตาราง push_subscriptions (ถามจาก
// /api/push/status) — คนละค่ากับจังหวัดที่กำลังดูอยู่ (push ไม่เปลี่ยนตามจังหวัดที่ดู ดู CHANGELOG 269)

async function fetchServerProvince(endpoint) {
  const res = await fetch('/api/push/status', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ endpoint }),
  })
  if (!res.ok) throw new Error(`Server error (${res.status})`)
  return res.json() // { subscribed, province }
}

function isStandalone() {
  // iOS Safari ต้องติดตั้งเป็น PWA ก่อนถึงจะรับ push ได้จริง (เหมือน InstallPrompt.vue)
  return window.matchMedia('(display-mode: standalone)').matches || window.navigator.standalone === true
}

function isIOS() {
  return /iphone|ipad|ipod/i.test(navigator.userAgent)
}

export function usePushStatus(province) {
  const status = ref('checking')
  const busy = ref(false)
  const errorMsg = ref('')
  const pushedProvince = ref(null)

  async function refreshStatus() {
    // เช็ค iOS-ไม่ได้ติดตั้งก่อนเช็คว่า API มีจริงไหม — บน iOS Safari ที่ไม่ได้เปิดจากไอคอนหน้าจอโฮม
    // Notification/PushManager จะไม่มีอยู่ใน window เลย ถ้าเช็ค unsupported ก่อนจะไม่ได้โชว์ hint "ติดตั้งแอปก่อน"
    if (isIOS() && !isStandalone()) {
      status.value = 'ios-not-installed'
      return
    }
    if (!('Notification' in window) || !('serviceWorker' in navigator) || !('PushManager' in window)) {
      status.value = 'unsupported'
      return
    }
    if (Notification.permission === 'denied') {
      status.value = 'denied'
      return
    }
    const existing = await getExistingPushSubscription().catch(() => null)
    if (!existing) {
      pushedProvince.value = null
      status.value = 'subscribable'
      return
    }
    try {
      const server = await fetchServerProvince(existing.endpoint)
      // เบราว์เซอร์ยังมี subscription แต่ฝั่ง server ไม่มีแถวแล้ว (เช่นถูกลบตอนส่งแล้วได้ 404/410) —
      // ถือว่ายังไม่ได้เปิด กดเปิดใหม่จะ upsert แถวกลับมาเอง
      pushedProvince.value = server.subscribed ? server.province : null
      status.value = server.subscribed ? 'subscribed' : 'subscribable'
    } catch {
      // ถาม server ไม่ได้ (offline ฯลฯ) — ยังถือว่าเปิดอยู่ตามเบราว์เซอร์ แค่ไม่รู้จังหวัด
      pushedProvince.value = null
      status.value = 'subscribed'
    }
  }

  async function enable() {
    errorMsg.value = ''
    busy.value = true
    try {
      const permission = await Notification.requestPermission()
      if (permission !== 'granted') {
        status.value = permission === 'denied' ? 'denied' : 'subscribable'
        return
      }
      const p = toValue(province)
      await subscribeToPush(p)
      pushedProvince.value = p
      status.value = 'subscribed'
    } catch {
      errorMsg.value = t('เปิดการแจ้งเตือนไม่สำเร็จ ลองใหม่อีกครั้ง')
    } finally {
      busy.value = false
    }
  }

  async function disable() {
    errorMsg.value = ''
    busy.value = true
    try {
      await unsubscribeFromPush()
      pushedProvince.value = null
      status.value = 'subscribable'
    } catch {
      errorMsg.value = t('ปิดการแจ้งเตือนไม่สำเร็จ ลองใหม่อีกครั้ง')
    } finally {
      busy.value = false
    }
  }

  // เปลี่ยนจังหวัดที่แจ้งเตือน — ใช้ subscription เดิมของเบราว์เซอร์ (subscribeToPush ไม่สร้างใหม่ถ้ามีอยู่แล้ว)
  // แล้ว POST /api/push/subscribe ซึ่ง upsert ทับแถวเดิมด้วยจังหวัดใหม่ (+ รีเซ็ต dedup) ไม่ต้องขอ permission ซ้ำ
  async function changeProvince(newProvince) {
    if (!newProvince || newProvince === pushedProvince.value) return
    errorMsg.value = ''
    busy.value = true
    try {
      await subscribeToPush(newProvince)
      pushedProvince.value = newProvince
    } catch {
      errorMsg.value = t('เปลี่ยนจังหวัดไม่สำเร็จ ลองใหม่อีกครั้ง')
    } finally {
      busy.value = false
    }
  }

  onMounted(refreshStatus)

  return { status, busy, errorMsg, pushedProvince, enable, disable, changeProvince, refreshStatus }
}

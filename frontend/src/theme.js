import { reactive } from 'vue'

// ธีมสว่าง/มืด — เลือกได้ 3 แบบในหน้าตั้งค่า: 'light' | 'dark' | 'system'
// เลือกสว่าง/มืด = เก็บค่าไว้ใน localStorage, เลือก "ตามระบบ" = ลบค่าที่เก็บทิ้ง แล้วตามธีมของเครื่อง
// (prefers-color-scheme) ต่อไป รวมถึงตอนเครื่องสลับโหมดกลางคัน ธีมของทุกหน้าอ่านจาก attribute data-theme บน <html>
const STORAGE_KEY = 'pm25_theme'
const THEME_COLOR = { light: '#f6f4ef', dark: '#0e1013' } // meta theme-color = สี --ground ของแต่ละธีม

function readStored() {
  try {
    const v = localStorage.getItem(STORAGE_KEY)
    return v === 'light' || v === 'dark' ? v : null
  } catch {
    return null // private mode / ปิด storage — ใช้ธีมเครื่องแทน ไม่ให้แอปพัง
  }
}

function systemTheme() {
  return window.matchMedia?.('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
}

const stored = readStored()

export const themeState = reactive({
  mode: stored || systemTheme(), // 'light' | 'dark' (ค่าที่ใช้อยู่จริง)
  preference: stored || 'system', // 'light' | 'dark' | 'system' (ที่ผู้ใช้เลือกในหน้าตั้งค่า)
})

function apply(mode) {
  const root = document.documentElement
  root.setAttribute('data-theme', mode)
  root.style.colorScheme = mode // ให้ scrollbar/ช่องกรอกของเบราว์เซอร์เปลี่ยนตามด้วย
  document.querySelector('meta[name="theme-color"]')?.setAttribute('content', THEME_COLOR[mode])
}

export function initTheme() {
  apply(themeState.mode)
  // เลือก "ตามระบบ" อยู่ -> ตามเครื่องต่อไปเวลาเครื่องสลับโหมดกลางคัน
  window.matchMedia?.('(prefers-color-scheme: dark)').addEventListener?.('change', () => {
    if (themeState.preference !== 'system') return
    themeState.mode = systemTheme()
    apply(themeState.mode)
  })
}

export function setTheme(preference) {
  if (preference !== 'light' && preference !== 'dark' && preference !== 'system') return
  themeState.preference = preference
  themeState.mode = preference === 'system' ? systemTheme() : preference
  try {
    if (preference === 'system') localStorage.removeItem(STORAGE_KEY)
    else localStorage.setItem(STORAGE_KEY, preference)
  } catch {
    // เก็บไม่ได้ก็ใช้ได้เฉพาะเซสชันนี้
  }
  apply(themeState.mode)
}

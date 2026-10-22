import { reactive } from 'vue'

// สลับธีมสว่าง/มืดด้วยปุ่ม — เก็บค่าที่เลือกไว้ใน localStorage ถ้ายังไม่เคยเลือกจะตามธีมของเครื่อง
// (prefers-color-scheme) และตามต่อเมื่อเครื่องเปลี่ยน ธีมของทุกหน้าอ่านจาก attribute data-theme บน <html>
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
  explicit: stored !== null, // true = ผู้ใช้เลือกเอง ไม่ตามธีมเครื่องอีก
})

function apply(mode) {
  const root = document.documentElement
  root.setAttribute('data-theme', mode)
  root.style.colorScheme = mode // ให้ scrollbar/ช่องกรอกของเบราว์เซอร์เปลี่ยนตามด้วย
  document.querySelector('meta[name="theme-color"]')?.setAttribute('content', THEME_COLOR[mode])
}

export function initTheme() {
  apply(themeState.mode)
  // ยังไม่เคยเลือกเอง -> ตามเครื่องต่อไปเวลาเครื่องสลับโหมดกลางคัน
  window.matchMedia?.('(prefers-color-scheme: dark)').addEventListener?.('change', () => {
    if (themeState.explicit) return
    themeState.mode = systemTheme()
    apply(themeState.mode)
  })
}

export function toggleTheme() {
  themeState.mode = themeState.mode === 'dark' ? 'light' : 'dark'
  themeState.explicit = true
  try {
    localStorage.setItem(STORAGE_KEY, themeState.mode)
  } catch {
    // เก็บไม่ได้ก็สลับได้เฉพาะเซสชันนี้
  }
  apply(themeState.mode)
}

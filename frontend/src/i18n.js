import { reactive } from 'vue'
import en from './locales/en.js'

// ระบบเปลี่ยนภาษา (ไทย/English) แบบเบา ไม่ใช้ vue-i18n
// ใช้ "ข้อความภาษาไทยเป็นคีย์" (แบบ gettext): t('ตั้งค่า') คืนค่าเดิมเมื่อเป็นภาษาไทย และคืนคำแปลจาก
// locales/en.js เมื่อเป็นภาษาอังกฤษ — เลือกแบบนี้เพราะค่าข้อมูลหลายตัวในแอปเป็นภาษาไทยอยู่แล้ว
// (ระดับคุณภาพอากาศ "ดีมาก"/"ดี"/..., กลุ่มเสี่ยง "ทั่วไป"/..., ความหนักกิจกรรม "เบา"/...) และใช้เป็นคีย์ทั้ง
// ใน frontend และค่าที่ backend ส่งมา จึงแปลตอนแสดงผลด้วย t(value) ได้เลยโดยไม่ต้องเปลี่ยนค่าข้อมูลเอง
// ข้อความที่มีตัวแปรใช้ {ชื่อ} เช่น t('อีก {h} ชม.', { h: 3 })
// ขอบเขต: แปลเฉพาะ UI — ข้อความที่ backend สร้าง (สรุปรายวันจาก Gemini, ปัจจัย SHAP, Web Push) ยังเป็นไทย
// หน้า /compare (admin) ไม่ได้แปล
const STORAGE_KEY = 'pm25_locale'
export const LOCALES = ['th', 'en']

function readStored() {
  try {
    const v = localStorage.getItem(STORAGE_KEY)
    return LOCALES.includes(v) ? v : null
  } catch {
    return null
  }
}

export const localeState = reactive({
  locale: readStored() || 'th',
})

const _warned = new Set()

export function t(th, params) {
  let s = th
  if (localeState.locale === 'en' && th != null) {
    const hit = en[th]
    if (hit != null) s = hit
    else if (import.meta.env.DEV && typeof th === 'string' && /[฀-๿]/.test(th) && !_warned.has(th)) {
      _warned.add(th)
      console.warn('[i18n] missing en:', th)
    }
  }
  if (params && typeof s === 'string') {
    s = s.replace(/\{(\w+)\}/g, (m, k) => (params[k] != null ? params[k] : m))
  }
  return s
}

export function isEn() {
  return localeState.locale === 'en'
}

// เวลาแบบนาฬิกา: ไทยต่อท้าย " น." ("14:00 น."), อังกฤษไม่ต่อ ("14:00")
export function clock(hhmm) {
  if (hhmm == null || hhmm === '') return ''
  return localeState.locale === 'en' ? String(hhmm) : `${hhmm} น.`
}

function applyLang() {
  document.documentElement.setAttribute('lang', localeState.locale)
}

export function initLocale() {
  applyLang()
}

export function setLocale(locale) {
  if (!LOCALES.includes(locale)) return
  localeState.locale = locale
  try {
    localStorage.setItem(STORAGE_KEY, locale)
  } catch {
    // เก็บไม่ได้ก็ใช้ได้เฉพาะเซสชันนี้
  }
  applyLang()
}

// ชื่อจังหวัดสำหรับแสดงผล — registry มี province_en เป็น slug (เช่น "khonkaen") ไม่ใช่ชื่อที่อ่านง่าย
// จึงมีตารางชื่อภาษาอังกฤษแยกไว้ที่นี่ (ครบ 20 จังหวัดใน data/province_station_registry.json)
const PROVINCE_NAME_EN = {
  khonkaen: 'Khon Kaen',
  loei: 'Loei',
  nongkhai: 'Nong Khai',
  udonthani: 'Udon Thani',
  nakhonratchasima: 'Nakhon Ratchasima',
  ubonratchathani: 'Ubon Ratchathani',
  buriram: 'Buriram',
  surin: 'Surin',
  sisaket: 'Sisaket',
  sakonnakhon: 'Sakon Nakhon',
  kalasin: 'Kalasin',
  mukdahan: 'Mukdahan',
  yasothon: 'Yasothon',
  chaiyaphum: 'Chaiyaphum',
  amnatcharoen: 'Amnat Charoen',
  nongbualamphu: 'Nong Bua Lamphu',
  buengkan: 'Bueng Kan',
  nakhonphanom: 'Nakhon Phanom',
  roiet: 'Roi Et',
  mahasarakham: 'Maha Sarakham',
}

// รับ object จังหวัดจาก /api/provinces ({province_th, province_en, ...}) หรือ undefined
export function provinceName(p) {
  if (!p) return ''
  if (localeState.locale === 'en') return PROVINCE_NAME_EN[p.province_en] || p.province_en
  return p.province_th
}

// ใช้ตอนค้นหาจังหวัด — ค้นได้ทั้งชื่อไทย ชื่ออังกฤษ และ slug ไม่ว่าจะเลือกภาษาไหนอยู่
export function provinceSearchText(p) {
  return `${p.province_th} ${PROVINCE_NAME_EN[p.province_en] || ''} ${p.province_en}`.toLowerCase()
}

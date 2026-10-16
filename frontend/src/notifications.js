import { reactive } from 'vue'
import { provinceState } from './provinces'

// สถานะแจ้งเตือนที่ใช้ร่วมกันระหว่าง AppHeader (กระดิ่ง) และ ForecastView (การ์ด/แบนเนอร์เตือน)
// pattern เดียวกับ auth.js (reactive singleton + dedup ผ่าน promise cache) — โปรเจกต์นี้ไม่ใช้ Pinia
export const notificationState = reactive({
  status: 'idle', // 'idle' | 'loading' | 'ready' | 'error'
  highPollutionRisk: false,
  riskGroup: 'ทั่วไป',
  riskHorizons: [],
  fallbackHorizons: [],
  currentPm25: null,
  currentStatus: '',
})

const WARNING_ADVICE_BY_RISK_GROUP = {
  ทั่วไป: 'โปรดระมัดระวังเพิ่มเติม',
  'เด็ก-ผู้สูงอายุ': 'กลุ่มเด็ก/ผู้สูงอายุควรหลีกเลี่ยงกิจกรรมกลางแจ้งเป็นเวลานานและระมัดระวังเป็นพิเศษ',
  มีโรคประจำตัว: 'ผู้มีโรคประจำตัวทางเดินหายใจควรหลีกเลี่ยงกิจกรรมกลางแจ้งและเตรียมอุปกรณ์ป้องกัน',
}

// ใช้ทั้งใน .warning-box ของ ForecastView และในกระดิ่งแจ้งเตือนของ AppHeader — นิยามที่เดียว
// กันข้อความสองที่ไม่ตรงกัน
export function buildWarningMessage(riskGroup) {
  const advice = WARNING_ADVICE_BY_RISK_GROUP[riskGroup] || WARNING_ADVICE_BY_RISK_GROUP['ทั่วไป']
  return `ช่วงนี้ PM2.5 อยู่ในระดับสูง ค่าพยากรณ์อาจต่ำกว่าความจริงเล็กน้อยในช่วงมลพิษสูง ${advice}`
}

let _fetchPromise = null

// /api/forecast ทำงานหนัก (เรียก API ภายนอก + inference 3 โมเดล + บางครั้งเรียก Gemini) —
// AppHeader mount อยู่ทุกหน้า ส่วน ForecastView mount เฉพาะหน้า "/" ถ้าไม่ dedup ที่นี่ หน้า "/"
// จะยิง /api/forecast 2 ครั้งซ้อนกัน (จากทั้งกระดิ่งและตัวหน้าเอง)
export async function fetchForecast() {
  if (_fetchPromise) return _fetchPromise
  notificationState.status = 'loading'
  _fetchPromise = (async () => {
    const res = await fetch(`/api/forecast?province=${provinceState.selected}`)
    if (!res.ok) {
      const body = await res.json().catch(() => ({}))
      const err = new Error(body.error || `Server error (${res.status})`)
      err.dataUnavailable = Boolean(body.data_unavailable) // 503: สถานีของจังหวัดนี้ไม่มีข้อมูลเลย (ไม่ใช่ server พัง)
      throw err
    }
    const data = await res.json()
    notificationState.highPollutionRisk = data.high_pollution_risk
    notificationState.riskGroup = data.risk_group || 'ทั่วไป'
    notificationState.riskHorizons = data.risk_horizons || []
    notificationState.fallbackHorizons = data.fallback_triggered_horizons || []
    notificationState.currentPm25 = data.current?.pm25 ?? null
    notificationState.currentStatus = data.current?.status ?? ''
    notificationState.status = 'ready'
    return data
  })()
  try {
    return await _fetchPromise
  } catch (err) {
    notificationState.status = 'error'
    throw err
  } finally {
    _fetchPromise = null
  }
}

// เบากว่า fetchForecast() มาก — ไม่ผ่าน feature engineering/model inference/log ใดๆ แค่ค่า PM2.5
// ล่าสุดของจังหวัดเดียว ใช้สำหรับ poll รีเฟรชการ์ด "ปัจจุบัน" เป็นระยะโดยไม่ยิง /api/forecast ซ้ำถี่ๆ
// ไม่ throw และไม่แตะ notificationState (ตัวแปรร่วมกับกระดิ่ง/แบนเนอร์เตือน) — คืน null เงียบๆ เมื่อพลาด
// ให้ caller (ForecastView) ตัดสินใจเองว่าจะคงค่าเดิมไว้หรือทำอะไรต่อ
export async function fetchCurrentPm25(province) {
  try {
    const res = await fetch(`/api/current_pm25?province=${province}`)
    if (!res.ok) return null
    return await res.json()
  } catch {
    return null
  }
}

// เหมือน fetchForecast() แต่สำหรับ "จังหวัดหลัก" ของ user (home_province) แยกต่างหาก — ใช้
// /api/notifications/status แทน /api/forecast เพราะถูกกว่ามาก (อ่านจาก notification_log/
// prediction_log ล่าสุด ไม่เรียก live API/inference ซ้ำ) เหมาะกับกระดิ่งที่แค่ต้องรู้สถานะเสี่ยง
// ไม่ต้องมีค่าพยากรณ์ 24 ชม.เต็มแบบหน้า "/" คนละฟังก์ชันกับ fetchForecast() เพราะจังหวัดอาจไม่ตรงกับ
// provinceState.selected ที่หน้า "/" ใช้แสดงผล
export async function fetchHomeProvinceStatus(province) {
  notificationState.status = 'loading'
  try {
    const res = await fetch(`/api/notifications/status?province=${province}`)
    if (!res.ok) {
      const body = await res.json().catch(() => ({}))
      throw new Error(body.error || `Server error (${res.status})`)
    }
    const data = await res.json()
    notificationState.highPollutionRisk = data.high_pollution_risk
    notificationState.riskGroup = data.risk_group || 'ทั่วไป'
    notificationState.riskHorizons = data.risk_horizons || []
    notificationState.fallbackHorizons = data.fallback_horizons || []
    notificationState.currentPm25 = data.current_pm25
    notificationState.currentStatus = data.current_status || ''
    notificationState.status = 'ready'
    return data
  } catch (err) {
    notificationState.status = 'error'
    throw err
  }
}

// ประวัติการแจ้งเตือน — endpoint แยกจาก /api/forecast ไม่ต้อง prewarm ตอน mount เหมือน fetchForecast()
// (กระดิ่งไม่จำเป็นต้องมีประวัติพร้อมใช้ทันทีเหมือน badge จุดแดง) ดึงตอนเปิด dropdown/เข้าหน้า
// /notifications เท่านั้น
export const notificationHistory = reactive({
  status: 'idle', // 'idle' | 'loading' | 'ready' | 'error'
  items: [],
})

// "YYYY-MM-DD HH:MM" (เวลาไทยจาก backend, recorded_at_th) -> Date local ล้วนๆ ไม่ยุ่ง timezone
// เพิ่มเติม เพราะแค่บวกชั่วโมงแล้ว format กลับเป็น string เองทั้งหมด
function parseThaiDateTime(str) {
  const [datePart, timePart] = str.split(' ')
  const [y, m, d] = datePart.split('-').map(Number)
  const [hh, mm] = timePart.split(':').map(Number)
  return new Date(y, m - 1, d, hh, mm)
}

const THAI_MONTHS_ABBR = [
  'ม.ค.', 'ก.พ.', 'มี.ค.', 'เม.ย.', 'พ.ค.', 'มิ.ย.',
  'ก.ค.', 'ส.ค.', 'ก.ย.', 'ต.ค.', 'พ.ย.', 'ธ.ค.',
]

function formatHM(date) {
  const hh = String(date.getHours()).padStart(2, '0')
  const mm = String(date.getMinutes()).padStart(2, '0')
  return `${hh}:${mm} น.`
}

// หัวการ์ดประวัติ: "2026-09-23 21:00" (ดิบจาก backend) -> "วันนี้ 21:00 น." / "เมื่อวาน ..." / "23 ก.ย. ..."
// อ่านง่ายกว่า ISO-ish string เดิม โดยเทียบกับวันที่ปัจจุบันของเบราว์เซอร์ผู้ใช้
export function formatCardTime(str) {
  const d = parseThaiDateTime(str)
  const now = new Date()
  const yesterday = new Date(now)
  yesterday.setDate(now.getDate() - 1)
  if (d.toDateString() === now.toDateString()) return `วันนี้ ${formatHM(d)}`
  if (d.toDateString() === yesterday.toDateString()) return `เมื่อวาน ${formatHM(d)}`
  return `${d.getDate()} ${THAI_MONTHS_ABBR[d.getMonth()]} ${formatHM(d)}`
}

// จัดกลุ่ม horizon ที่ติดกันเป็นช่วงเดียว (เช่น [1,2,3,7,8] -> [[1,3],[7,8]]) กัน list ยาวเป็นพรืด
function groupConsecutive(horizons) {
  const groups = []
  let start = horizons[0]
  let prev = horizons[0]
  for (let i = 1; i < horizons.length; i++) {
    if (horizons[i] === prev + 1) {
      prev = horizons[i]
    } else {
      groups.push([start, prev])
      start = horizons[i]
      prev = horizons[i]
    }
  }
  groups.push([start, prev])
  return groups
}

// ติดป้าย "พรุ่งนี้" แค่ตอนวันเปลี่ยนจริง แค่จุดเดียวต่อช่วง (ไม่ใช่ทุกจุดเวลาเหมือนเดิม) —
// horizon สูงสุด 24 ข้ามได้แค่ 1 วัน ไม่ต้องรองรับ "มะรืนนี้"
function formatRange(base, start, end) {
  const startDate = new Date(base.getTime() + start * 3600000)
  const startLabel = startDate.toDateString() === base.toDateString() ? '' : 'พรุ่งนี้ '
  if (start === end) return `${startLabel}${formatHM(startDate)}`
  const endDate = new Date(base.getTime() + end * 3600000)
  const endLabel = endDate.toDateString() === startDate.toDateString() ? '' : 'พรุ่งนี้ '
  return `${startLabel}${formatHM(startDate)} – ${endLabel}${formatHM(endDate)}`
}

// สรุป 1 รายการประวัติเป็น { headline, ranges, note } ให้ template render เป็นหัวข้อ + chip แยกใบ
// (ไม่ใช่ string ประโยคยาวคั่นจุลภาคเหมือนเดิม — อ่านยากเมื่อมีหลายช่วง) ใช้ร่วมกันทั้ง dropdown ใน
// AppHeader และหน้า /notifications เต็ม กันข้อความไม่ตรงกันเหมือนที่ทำกับ buildWarningMessage
export function summarizeHistoryItem(item) {
  let headline = ''
  let ranges = []
  if (item.risk_horizons.length === 24) {
    headline = 'คาดว่า PM2.5 จะอยู่ในระดับสูงตลอด 24 ชั่วโมงข้างหน้า'
  } else if (item.risk_horizons.length > 0) {
    headline = `มี ${item.risk_horizons.length} ช่วงเวลาที่ควรระวังในอีก 24 ชม. ข้างหน้า`
    const base = parseThaiDateTime(item.recorded_at_th)
    ranges = groupConsecutive(item.risk_horizons).map(([start, end]) => formatRange(base, start, end))
  } else {
    headline = `PM2.5 ปัจจุบันสูง (${item.latest_pm25} µg/m³)`
  }
  const note = item.fallback_horizons.length > 0 ? 'บางช่วงข้อมูลไม่ครบ ใช้ค่าพยากรณ์ชั่วคราวแทน' : null
  return { headline, ranges, note }
}

// เปิดไว้ชั่วคราวเพื่อโชว์อาจารย์ (ผู้ใช้สั่ง 2026-10-04) — ของจริงตอนนี้ว่างเปล่าเพราะ PM2.5 ยังไม่เคย
// ถึงเกณฑ์แจ้งเตือนในข้อมูลที่มี (ตรวจ DB แล้ว ไม่ใช่บั๊ก) ตั้ง false เพื่อกลับไปดึงข้อมูลจริงจาก API
// เหมือนเดิม — กระทบทั้ง dropdown กระดิ่งใน AppHeader.vue และหน้า /notifications เพราะทั้งคู่อ่านจาก
// notificationHistory.items ตัวเดียวกัน
const SAMPLE_NOTIFICATION_HISTORY = true

function hoursAgo(h) {
  return new Date(Date.now() - h * 3600000)
}
function daysAgo(d, hour) {
  const dt = new Date()
  dt.setDate(dt.getDate() - d)
  dt.setHours(hour, 0, 0, 0)
  return dt
}
function fmtThaiDateTime(date) {
  const p = (n) => String(n).padStart(2, '0')
  return `${date.getFullYear()}-${p(date.getMonth() + 1)}-${p(date.getDate())} ${p(date.getHours())}:${p(date.getMinutes())}`
}

// รูปแบบเดียวกับที่ /api/notifications ส่งจริง (recorded_at_th/latest_pm25/risk_horizons/
// fallback_horizons) ครอบคลุมทุกเคสที่ summarizeHistoryItem() แตกกิ่งจริง (เสี่ยงตลอด 24 ชม.,
// เสี่ยงบางช่วง, ฝุ่นสูง ณ ตอนนั้น, ข้อมูลไม่ครบ/ใช้ค่าพยากรณ์ชั่วคราว)
function sampleHistoryItems() {
  return [
    { recorded_at_th: fmtThaiDateTime(hoursAgo(2)), latest_pm25: 45.2, risk_horizons: Array.from({ length: 24 }, (_, i) => i + 1), fallback_horizons: [] },
    { recorded_at_th: fmtThaiDateTime(hoursAgo(5)), latest_pm25: 38.1, risk_horizons: [13, 14, 15, 16, 20, 21], fallback_horizons: [] },
    { recorded_at_th: fmtThaiDateTime(daysAgo(1, 14)), latest_pm25: 52.0, risk_horizons: [], fallback_horizons: [] },
    { recorded_at_th: fmtThaiDateTime(daysAgo(3, 7)), latest_pm25: 40.5, risk_horizons: [1, 2, 3], fallback_horizons: [1, 2, 3] },
  ]
}

// province: ไม่ส่งมา = ใช้ provinceState.selected เหมือนเดิม (guest/ยังไม่ตั้งจังหวัดหลัก) — ส่งมา
// เพื่อให้ตรงกับจังหวัดที่กระดิ่ง (badge สถานะ) กำลังเฝ้าอยู่จริง เวลามี home_province (AppHeader.vue)
export async function fetchNotificationHistory(limit = 5, province = provinceState.selected) {
  if (SAMPLE_NOTIFICATION_HISTORY) {
    notificationHistory.items = sampleHistoryItems().slice(0, limit)
    notificationHistory.status = 'ready'
    return
  }
  notificationHistory.status = 'loading'
  try {
    const res = await fetch(`/api/notifications?limit=${limit}&province=${province}`)
    if (!res.ok) {
      const body = await res.json().catch(() => ({}))
      throw new Error(body.error || `Server error (${res.status})`)
    }
    notificationHistory.items = await res.json()
    notificationHistory.status = 'ready'
  } catch (err) {
    notificationHistory.status = 'error'
    throw err
  }
}

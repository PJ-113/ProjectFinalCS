import { reactive } from 'vue'
import { provinceState } from './provinces'

// สถานะแจ้งเตือนที่ใช้ร่วมกันระหว่าง AppHeader (กระดิ่ง) และ ForecastView (การ์ด/แบนเนอร์เตือน)
export const notificationState = reactive({
  status: 'idle', // 'idle' | 'loading' | 'ready' | 'error'
  highPollutionRisk: false,
  riskGroup: 'ทั่วไป',
  riskHorizons: [],
  fallbackHorizons: [],
  currentPm25: null,
  currentStatus: '',
  // ข้อ 2 "ใกล้ถึงช่วงเสี่ยงแล้ว"
  riskImminent: false,
  minRiskHorizon: null,
  minRiskHorizonPm25: null,
  minRiskHorizonDisplayTime: null,
  // ข้อ 3 "แนวโน้มฝุ่นจะดีขึ้น"
  trendDirection: null,
  trendPeriod: null,
  // ข้อ 4 "สถานีเงียบนานผิดปกติ" — ตั้งเฉพาะใน fetchHomeProvinceStatus() เท่านั้น (ดูคอมเมนต์ที่นั่น)
  dataStale: null,
  staleHours: null,
})

const WARNING_ADVICE_BY_RISK_GROUP = {
  ทั่วไป: 'โปรดระมัดระวังเพิ่มเติม',
  'เด็ก-ผู้สูงอายุ': 'กลุ่มเด็ก/ผู้สูงอายุควรหลีกเลี่ยงกิจกรรมกลางแจ้งเป็นเวลานานและระมัดระวังเป็นพิเศษ',
  มีโรคประจำตัว: 'ผู้มีโรคประจำตัวทางเดินหายใจควรหลีกเลี่ยงกิจกรรมกลางแจ้งและเตรียมอุปกรณ์ป้องกัน',
}

// ใช้ทั้งใน .warning-box ของ ForecastView และกระดิ่งแจ้งเตือนของ AppHeader
export function buildWarningMessage(riskGroup) {
  const advice = WARNING_ADVICE_BY_RISK_GROUP[riskGroup] || WARNING_ADVICE_BY_RISK_GROUP['ทั่วไป']
  return `ช่วงนี้ PM2.5 อยู่ในระดับสูง ค่าพยากรณ์อาจต่ำกว่าความจริงเล็กน้อยในช่วงมลพิษสูง ${advice}`
}

// ข้อ 1 "ระดับเปลี่ยนสถานะ" — transition: { from_level, to_level, from_pm25, to_pm25, direction, levels_moved }
// จาก backend (risk_group-independent จึงไม่ต้องมี dict ต่อ risk_group แบบ buildWarningMessage)
// โทน "สรุปนำหน้า" — ขึ้นต้นด้วยทิศทาง/ขนาดการเปลี่ยนเป็นภาษาพูดก่อน (แย่ลง/ดีขึ้น, เล็กน้อย/มาก ตาม
// levels_moved) ตามด้วย "จาก X เป็น Y" แบบสั้น — เก็บค่า PM2.5 แค่ตัวล่าสุด (to_pm25) ไม่ใส่ from_pm25 ด้วย
// เพราะข้อความยาวเกินไปจนตัดบรรทัดดูไม่สวย (ผู้ใช้เลือกตัดค่าก่อนหน้าออกแทนการเปลี่ยนวิธีตัดบรรทัด)
export function buildLevelTransitionMessage(transition) {
  const directionWord = transition.direction === 'worse' ? 'แย่ลง' : 'ดีขึ้น'
  const magnitude = transition.levels_moved >= 2 ? 'มาก' : 'เล็กน้อย'
  return `อากาศ${directionWord}${magnitude} จาก "${transition.from_level}" เป็น "${transition.to_level}" (${transition.to_pm25} µg/m³)`
}

// ข้อ 2 "ใกล้ถึงช่วงเสี่ยงแล้ว" — โทนกระชับ ตรงประเด็น ไม่มีคำแนะนำ/อีโมจิ (ตัดสินใจแล้ว ต่างจาก 3 แบบอื่น)
// item: ต้องมี min_risk_horizon, min_risk_horizon_pm25, min_risk_horizon_display_time (backend คำนวณเวลาให้
// ตรงกับที่กราฟพยากรณ์หลักโชว์ไว้แล้ว ไม่ต้องคำนวณ horizon->เวลาซ้ำฝั่ง frontend) ใช้ได้ทั้งจาก notificationState
// (fetchForecast()/fetchHomeProvinceStatus()) และจาก item หนึ่งใน notificationHistory.items เพราะมี field ชุดนี้
// เหมือนกันทุกแหล่ง
export function buildImminentRiskMessage(item) {
  return `ฝุ่นจะเกินเกณฑ์ ${item.min_risk_horizon_display_time} น. (อีก ${item.min_risk_horizon} ชม.) คาดการณ์ ${item.min_risk_horizon_pm25} µg/m³`
}

// ข้อ 3 "แนวโน้มฝุ่นจะดีขึ้น" (ข่าวดี) — เรียกเฉพาะตอน trend_direction === "down" เท่านั้น
export function buildImprovingTrendMessage(trendPeriod) {
  return `แนวโน้มช่วง${trendPeriod}ฝุ่นจะลดลง`
}

// ข้อ 4 "สถานีเงียบนานผิดปกติ"
export function buildStationSilenceMessage(staleHours) {
  return `ไม่มีข้อมูลสถานีใหม่มา ${Math.round(staleHours)} ชั่วโมงแล้ว`
}

let _fetchPromise = null

// /api/forecast ทำงานหนัก (เรียก API ภายนอก + inference 3 โมเดล + บางครั้งเรียก Gemini) —
// dedup ด้วย promise cache กันการยิงซ้ำซ้อนตอน AppHeader และ ForecastView mount พร้อมกัน
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
    notificationState.riskImminent = data.risk_imminent || false
    notificationState.minRiskHorizon = data.min_risk_horizon ?? null
    notificationState.minRiskHorizonPm25 = data.min_risk_horizon_pm25 ?? null
    notificationState.minRiskHorizonDisplayTime = data.min_risk_horizon_display_time ?? null
    notificationState.trendDirection = data.trend_direction ?? null
    notificationState.trendPeriod = data.trend_period ?? null
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

// ดึงค่า PM2.5 ล่าสุดของจังหวัดเดียวเท่านั้น (ไม่ผ่าน feature engineering/model inference/log)
// ใช้สำหรับ poll รีเฟรชการ์ด "ปัจจุบัน" เป็นระยะ ไม่ throw และไม่แตะ notificationState
// คืน null เงียบๆ เมื่อพลาด ให้ caller (ForecastView) ตัดสินใจเองว่าจะทำอะไรต่อ
export async function fetchCurrentPm25(province) {
  try {
    const res = await fetch(`/api/current_pm25?province=${province}`)
    if (!res.ok) return null
    return await res.json()
  } catch {
    return null
  }
}

// เหมือน fetchForecast() แต่สำหรับ "จังหวัดหลัก" ของ user (home_province) ใช้
// /api/notifications/status ซึ่งอ่านจาก notification_log/prediction_log ล่าสุด
// ไม่เรียก live API/inference ซ้ำ และไม่คืนค่าพยากรณ์ 24 ชม.เต็มแบบหน้า "/"
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
    notificationState.riskImminent = data.risk_imminent || false
    notificationState.minRiskHorizon = data.min_risk_horizon ?? null
    notificationState.minRiskHorizonPm25 = data.min_risk_horizon_pm25 ?? null
    notificationState.minRiskHorizonDisplayTime = data.min_risk_horizon_display_time ?? null
    notificationState.trendDirection = data.trend_direction ?? null
    notificationState.trendPeriod = data.trend_period ?? null
    // ข้อ 4 "สถานีเงียบนานผิดปกติ" — ตั้งเฉพาะที่นี่เท่านั้น ไม่ใช่ใน fetchForecast() เพราะ /api/forecast เป็น
    // live call ที่ไม่คืนค่าเงียบแบบนี้อยู่แล้ว (มี data_quality.target_stale ของตัวเองแยกต่างหากใน DashboardMap.vue)
    notificationState.dataStale = data.data_stale ?? null
    notificationState.staleHours = data.stale_hours ?? null
    notificationState.status = 'ready'
    return data
  } catch (err) {
    notificationState.status = 'error'
    throw err
  }
}

// ประวัติการแจ้งเตือน — endpoint แยกจาก /api/forecast ดึงตอนเปิด dropdown/เข้าหน้า /notifications เท่านั้น
export const notificationHistory = reactive({
  status: 'idle', // 'idle' | 'loading' | 'ready' | 'error'
  items: [],
})

// "YYYY-MM-DD HH:MM" (เวลาไทยจาก backend, recorded_at_th) -> Date local ล้วนๆ ไม่ยุ่ง timezone
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
// เทียบกับวันที่ปัจจุบันของเบราว์เซอร์ผู้ใช้
export function formatCardTime(str) {
  const d = parseThaiDateTime(str)
  const now = new Date()
  const yesterday = new Date(now)
  yesterday.setDate(now.getDate() - 1)
  if (d.toDateString() === now.toDateString()) return `วันนี้ ${formatHM(d)}`
  if (d.toDateString() === yesterday.toDateString()) return `เมื่อวาน ${formatHM(d)}`
  return `${d.getDate()} ${THAI_MONTHS_ABBR[d.getMonth()]} ${formatHM(d)}`
}

// จัดกลุ่ม horizon ที่ติดกันเป็นช่วงเดียว (เช่น [1,2,3,7,8] -> [[1,3],[7,8]])
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

// horizon (1-24) -> เวลานาฬิกาจริง เทียบจาก base (เวลาที่บันทึก snapshot) — แยกออกมาจาก formatRange() เดิม
// เพื่อใช้ร่วมกับ buildImminentRiskMessage() ที่ต้องการแค่เวลาจุดเดียว ไม่ใช่ช่วง "start – end"
function horizonToClockTime(base, h) {
  return new Date(base.getTime() + h * 3600000)
}

// ติดป้าย "พรุ่งนี้" แค่ตอนวันเปลี่ยนจริง แค่จุดเดียวต่อช่วง
// horizon สูงสุด 24 ข้ามได้แค่ 1 วัน ไม่ต้องรองรับ "มะรืนนี้"
function formatRange(base, start, end) {
  const startDate = horizonToClockTime(base, start)
  const startLabel = startDate.toDateString() === base.toDateString() ? '' : 'พรุ่งนี้ '
  if (start === end) return `${startLabel}${formatHM(startDate)}`
  const endDate = horizonToClockTime(base, end)
  const endLabel = endDate.toDateString() === startDate.toDateString() ? '' : 'พรุ่งนี้ '
  return `${startLabel}${formatHM(startDate)} – ${endLabel}${formatHM(endDate)}`
}

// สรุป 1 รายการประวัติเป็น { headline, ranges, note } ให้ template render เป็นหัวข้อ + chip แยกใบ
// ใช้ร่วมกันทั้ง dropdown ใน AppHeader และหน้า /notifications เต็ม
export function summarizeHistoryItem(item) {
  let headline = ''
  let ranges = []
  // เดิมแถวที่เข้ามาถึงนี่ได้มีแค่ 2 เหตุผล (เสี่ยงตอนนี้/เสี่ยงบางช่วง) เดาได้จาก risk_horizons อย่างเดียว —
  // ตอนนี้ api_notifications() ส่งแถวที่ "ระดับเปลี่ยน" หรือ "แนวโน้มดีขึ้น" มาด้วยแม้ไม่เสี่ยงเลยก็ตาม ต้องเช็ค
  // current_at_risk ตรงๆ แทนการเดาจาก risk_horizons ว่างหรือไม่ว่าง
  if (item.risk_horizons.length === 24) {
    headline = 'คาดว่า PM2.5 จะอยู่ในระดับสูงตลอด 24 ชั่วโมงข้างหน้า'
  } else if (item.risk_horizons.length > 0) {
    headline = `ควรระวัง ${item.risk_horizons.length} ช่วงเวลาในอีก 24 ชม. ข้างหน้า`
    const base = parseThaiDateTime(item.recorded_at_th)
    ranges = groupConsecutive(item.risk_horizons).map(([start, end]) => formatRange(base, start, end))
  } else if (item.current_at_risk) {
    headline = `PM2.5 ปัจจุบันสูง (${item.latest_pm25} µg/m³)`
  } else if (item.transition) {
    headline = buildLevelTransitionMessage(item.transition)
  } else if (item.trend_direction === 'down') {
    headline = buildImprovingTrendMessage(item.trend_period)
  } else {
    headline = `PM2.5 ${item.latest_pm25} µg/m³ ` // เผื่อไว้ ปกติไม่ควรมาถึงบรรทัดนี้เพราะ backend กรองมาแล้ว
  }
  const note = item.fallback_horizons.length > 0 ? 'บางช่วงข้อมูลไม่ครบ ใช้ค่าพยากรณ์ชั่วคราวแทน' : null
  // ข้อ 1/3 อาจเกิดร่วมกับเหตุผลหลักอื่นได้พร้อมกันในแถวเดียว (เช่น เสี่ยงจริง + ระดับเปลี่ยนพร้อมกัน) — ถ้า
  // headline หลักไม่ได้ใช้ข้อความนี้ไปแล้ว ให้โชว์เป็น field เสริมแยกต่างหาก ไม่ทับ/ไม่ซ้ำ headline
  const transitionNote = item.transition && headline !== buildLevelTransitionMessage(item.transition)
    ? buildLevelTransitionMessage(item.transition) : null
  const goodNews = item.trend_direction === 'down' && headline !== buildImprovingTrendMessage(item.trend_period)
    ? buildImprovingTrendMessage(item.trend_period) : null
  return { headline, ranges, note, transitionNote, goodNews }
}

// true = ใช้ข้อมูลตัวอย่างแทนประวัติจริงจาก /api/notifications
// กระทบทั้ง dropdown กระดิ่งใน AppHeader.vue และหน้า /notifications เพราะทั้งคู่อ่านจาก
// notificationHistory.items ตัวเดียวกัน
const SAMPLE_NOTIFICATION_HISTORY = false

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

// province: ไม่ส่งมา = ใช้ provinceState.selected (guest/ยังไม่ตั้งจังหวัดหลัก) ส่งมาเพื่อให้ตรงกับ
// จังหวัดที่กระดิ่ง (badge สถานะ) กำลังเฝ้าอยู่จริง เวลามี home_province (AppHeader.vue)
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

// ===== Web Push =====

// VAPID public key เป็น base64url — PushManager.subscribe() ต้องการ Uint8Array ไม่ใช่ string ตรงๆ
function urlBase64ToUint8Array(base64String) {
  const padding = '='.repeat((4 - (base64String.length % 4)) % 4)
  const base64 = (base64String + padding).replace(/-/g, '+').replace(/_/g, '/')
  const rawData = window.atob(base64)
  return Uint8Array.from([...rawData].map((c) => c.charCodeAt(0)))
}

// รอ service worker active แบบมี timeout — `serviceWorker.ready` เฉยๆค้างตลอดกาลถ้าไม่มี service worker
// ถูก register เลย (เช่น `npm run dev` ที่ไม่เปิด devOptions) ไม่งั้นปุ่ม "เปิด/ปิดการแจ้งเตือน" จะหมุนค้างไม่จบ
function waitForServiceWorker(timeoutMs = 5000) {
  return Promise.race([
    navigator.serviceWorker.ready,
    new Promise((_, reject) => setTimeout(() => reject(new Error('service worker ยังไม่พร้อม')), timeoutMs)),
  ])
}

// สมัครรับ push ของ province นี้ — คืน subscription ที่สมัครสำเร็จ หรือ throw ถ้าพลาด (ให้ caller
// จัดการ error ของตัวเอง เช่น โชว์ข้อความใน PushPrompt.vue)
export async function subscribeToPush(province) {
  const registration = await waitForServiceWorker()
  let subscription = await registration.pushManager.getSubscription()
  if (!subscription) {
    const res = await fetch('/api/push/vapid-public-key')
    const { public_key } = await res.json()
    subscription = await registration.pushManager.subscribe({
      userVisibleOnly: true,
      applicationServerKey: urlBase64ToUint8Array(public_key),
    })
  }
  const res = await fetch('/api/push/subscribe', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ ...subscription.toJSON(), province }),
  })
  if (!res.ok) {
    const body = await res.json().catch(() => ({}))
    throw new Error(body.error || `Server error (${res.status})`)
  }
  return subscription
}

// ยกเลิกรับ push — ยกเลิกทั้งฝั่ง browser (pushManager.unsubscribe) และฝั่ง backend (ลบแถว DB)
export async function unsubscribeFromPush() {
  const registration = await waitForServiceWorker()
  const subscription = await registration.pushManager.getSubscription()
  if (!subscription) return
  const endpoint = subscription.endpoint
  await subscription.unsubscribe()
  await fetch('/api/push/unsubscribe', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ endpoint }),
  })
}

// เช็คว่ามี subscription อยู่แล้วหรือยัง — ใช้ตอน mount ของ PushPrompt.vue เพื่อตัดสินใจว่าจะโชว์ปุ่ม
// "เปิดการแจ้งเตือน" หรือ "ปิดการแจ้งเตือน"
//
// ใช้ getRegistration() ไม่ใช่ `serviceWorker.ready` — `.ready` resolve ก็ต่อเมื่อมี service worker
// active อยู่จริงเท่านั้น ถ้าไม่เคย register เลย (เช่นตอนรันด้วย `npm run dev` ซึ่งไม่เปิด devOptions ของ
// vite-plugin-pwa — ดู CHANGELOG ตอนเอา devOptions ออก) promise จะค้างไปตลอดกาล ไม่ reject ด้วย ทำให้
// PushPrompt.vue ค้างที่ status 'checking' และการ์ดทั้งใบไม่โผล่ขึ้นมาเลย getRegistration() resolve ทันที
// (คืน undefined ถ้ายังไม่มี) จึงไม่ค้างแบบนั้น
export async function getExistingPushSubscription() {
  if (!('serviceWorker' in navigator)) return null
  const registration = await navigator.serviceWorker.getRegistration()
  if (!registration) return null
  return registration.pushManager.getSubscription()
}

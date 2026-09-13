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
      throw new Error(body.error || `Server error (${res.status})`)
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

// ประวัติการแจ้งเตือน — endpoint แยกจาก /api/forecast ไม่ต้อง prewarm ตอน mount เหมือน fetchForecast()
// (กระดิ่งไม่จำเป็นต้องมีประวัติพร้อมใช้ทันทีเหมือน badge จุดแดง) ดึงตอนเปิด dropdown/เข้าหน้า
// /notifications เท่านั้น
export const notificationHistory = reactive({
  status: 'idle', // 'idle' | 'loading' | 'ready' | 'error'
  items: [],
})

// สรุป 1 รายการประวัติเป็นข้อความสั้นๆ อ่านง่าย — ใช้ร่วมกันทั้ง dropdown ใน AppHeader และหน้า
// /notifications เต็ม กันข้อความไม่ตรงกันเหมือนที่ทำกับ buildWarningMessage
export function summarizeHistoryItem(item) {
  const parts = []
  if (item.risk_horizons.length > 0) {
    parts.push(`มี ${item.risk_horizons.length} ช่วงเวลาที่พยากรณ์ PM2.5 สูง (t+${item.risk_horizons.join(', t+')})`)
  }
  if (item.fallback_horizons.length > 0) {
    parts.push(`ข้อมูลไม่สมบูรณ์ ใช้ค่าพยากรณ์แบบคงที่ชั่วคราวที่ t+${item.fallback_horizons.join(', t+')}`)
  }
  if (parts.length === 0) {
    parts.push(`PM2.5 ปัจจุบันสูง (${item.latest_pm25} µg/m³)`)
  }
  return parts.join(' • ')
}

export async function fetchNotificationHistory(limit = 5) {
  notificationHistory.status = 'loading'
  try {
    const res = await fetch(`/api/notifications?limit=${limit}&province=${provinceState.selected}`)
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

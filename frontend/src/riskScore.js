import { reactive } from 'vue'

// Personalized Risk Score — เก็บ activity ที่ผู้ใช้เลือกไว้ล่าสุดผ่าน localStorage
// offset/breakpoint ของสูตรคำนวณมาจาก risk_score_config ที่ /api/forecast ส่งมา
export const ACTIVITY_LEVELS = ['เบา', 'ปานกลาง', 'หนัก']

const STORAGE_KEY = 'pm25_activity_intensity'

function loadStoredActivity() {
  const stored = localStorage.getItem(STORAGE_KEY)
  return ACTIVITY_LEVELS.includes(stored) ? stored : 'ปานกลาง'
}

export const riskScoreState = reactive({
  activity: loadStoredActivity(),
})

export function setActivity(level) {
  if (!ACTIVITY_LEVELS.includes(level)) return
  riskScoreState.activity = level
  localStorage.setItem(STORAGE_KEY, level)
}

// ระดับเดียวกับ PM25_LEVELS ใน app.py (มาตรฐาน คพ. 2566) — ต้อง match string ให้ตรงเป๊ะกับที่
// _classify_pm25_level() คืนกลับมา
const LEVEL_BOUNDARIES = [
  [15, 'ดีมาก'],
  [25, 'ดี'],
  [37.5, 'ปานกลาง'],
  [75, 'เริ่มมีผลกระทบต่อสุขภาพ'],
  [Infinity, 'มีผลกระทบต่อสุขภาพ'],
]

function classifyLevel(value) {
  for (const [threshold, name] of LEVEL_BOUNDARIES) {
    if (value <= threshold) return name
  }
  return LEVEL_BOUNDARIES[LEVEL_BOUNDARIES.length - 1][1]
}

// mirror ของ compute_risk_score() ใน app.py — ใช้สูตร linear interpolation เดียวกัน
// offset/breakpoints มาจาก config ที่ backend ส่งมา
export function computeRiskScore(pm25, config, activity) {
  if (pm25 == null || !config) return { score: null, level: '', effective_pm25: null }
  const activityOffset = config.activity_offsets[activity] ?? 0
  const effective = pm25 - config.risk_group_offset - activityOffset
  const breakpoints = config.breakpoints
  const cap = breakpoints[breakpoints.length - 1][0]
  const clamped = Math.max(0, Math.min(effective, cap))

  let score = breakpoints[breakpoints.length - 1][1]
  for (let i = 0; i < breakpoints.length - 1; i++) {
    const [x0, y0] = breakpoints[i]
    const [x1, y1] = breakpoints[i + 1]
    if (clamped <= x1) {
      score = y0 + ((y1 - y0) * (clamped - x0)) / (x1 - x0)
      break
    }
  }

  return {
    score: Math.round(score),
    level: classifyLevel(effective),
    effective_pm25: Math.round(effective * 10) / 10,
  }
}

// คำแนะนำสั้นๆ ต่อท้ายคะแนน ตามระดับความเสี่ยง
const RISK_SCORE_ADVICE = {
  ดีมาก: 'ความเสี่ยงต่ำมาก ทำกิจกรรมนี้ได้ตามปกติ',
  ดี: 'ความเสี่ยงต่ำ ทำกิจกรรมนี้ได้ตามปกติ',
  ปานกลาง: 'ความเสี่ยงปานกลาง ทำได้แต่ควรสังเกตอาการตนเอง',
  เริ่มมีผลกระทบต่อสุขภาพ: 'ความเสี่ยงเริ่มสูง ควรลดความหนักหรือระยะเวลาของกิจกรรมนี้',
  มีผลกระทบต่อสุขภาพ: 'ความเสี่ยงสูง ควรเลี่ยงกิจกรรมนี้ในช่วงนี้',
}

export function riskScoreAdvice(level) {
  return RISK_SCORE_ADVICE[level] || ''
}

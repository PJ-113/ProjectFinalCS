// ระดับ/สี PM2.5 ตรงกับ PM25_LEVELS ใน app.py (มาตรฐาน คพ. 2566) — โมดูลกลางเดียวที่ ForecastView.vue
// และ MapView.vue ใช้ร่วมกัน ต้อง match string สถานะให้ตรงเป๊ะกับที่ pm25_status() คืนกลับมา

export const PM25_LEVELS = [
  { max: 15, status: "ดีมาก", cssVar: "--excellent", statusClass: "status-excellent", hex: "#3b82f6" },
  { max: 25, status: "ดี", cssVar: "--good", statusClass: "status-good", hex: "#22c55e" },
  { max: 37.5, status: "ปานกลาง", cssVar: "--moderate", statusClass: "status-moderate", hex: "#eab308" },
  { max: 75, status: "เริ่มมีผลกระทบต่อสุขภาพ", cssVar: "--unhealthy", statusClass: "status-unhealthy", hex: "#f97316" },
  { max: Infinity, status: "มีผลกระทบต่อสุขภาพ", cssVar: "--bad", statusClass: "status-bad", hex: "#ef4444" },
];

function levelForPm25(pm25) {
  return PM25_LEVELS.find((l) => pm25 <= l.max) || PM25_LEVELS[PM25_LEVELS.length - 1];
}

export function levelIndex(pm25) {
  const i = PM25_LEVELS.findIndex((l) => pm25 <= l.max);
  return i === -1 ? PM25_LEVELS.length - 1 : i;
}

// ตำแหน่ง (0-100%) ของ marker บนแถบสเกล 5 ระดับเท่ากันช่อง — ตำแหน่งในช่องคือสัดส่วนภายในช่วงของระดับนั้น
// ระดับสุดท้าย (>75) ไม่มีเพดาน ใช้ 75-125 เป็นช่วงแสดงผลเท่านั้น (ค่าเกินนั้นชิดขอบขวาสุด)
const SCALE_LOWS = [0, 15, 25, 37.5, 75];
const SCALE_HIGHS = [15, 25, 37.5, 75, 125];
export function scaleMarkerPct(pm25) {
  if (pm25 == null || Number.isNaN(Number(pm25))) return 0;
  const i = levelIndex(pm25);
  const frac = Math.min(Math.max((pm25 - SCALE_LOWS[i]) / (SCALE_HIGHS[i] - SCALE_LOWS[i]), 0), 1);
  return ((i + frac) / PM25_LEVELS.length) * 100;
}

// คำแนะนำสั้นๆ ใต้ค่า PM2.5 ปัจจุบัน ตามระดับ status ที่ backend ส่งมา
const ADVICE_BY_STATUS = {
  ดีมาก: "อากาศดีมาก ทำกิจกรรมกลางแจ้งได้ตามปกติ",
  ดี: "อากาศดี ทำกิจกรรมกลางแจ้งได้ตามปกติ",
  ปานกลาง: "อากาศพอใช้ กลุ่มเสี่ยงควรลดกิจกรรมกลางแจ้งที่ใช้แรงมากเป็นเวลานาน",
  เริ่มมีผลกระทบต่อสุขภาพ: "ควรลดกิจกรรมกลางแจ้ง กลุ่มเสี่ยงควรสวมหน้ากากป้องกันฝุ่นเมื่อออกนอกอาคาร",
  มีผลกระทบต่อสุขภาพ: "หลีกเลี่ยงกิจกรรมกลางแจ้ง และสวมหน้ากาก N95 หากจำเป็นต้องออกนอกอาคาร",
};
export function adviceForStatus(status) {
  return ADVICE_BY_STATUS[status] || "";
}

// numeric pm25 -> ชื่อ CSS custom property (--excellent/--good/--moderate/--unhealthy/--bad)
export function severityVar(pm25) {
  return levelForPm25(pm25).cssVar;
}

// Thai status string (เช่น current.status จาก /api/forecast) -> CSS class name
export function statusClass(status) {
  const level = PM25_LEVELS.find((l) => l.status === status);
  return level ? level.statusClass : "status-bad";
}

// numeric pm25 -> hex สีตรง ใช้ตอนต้องการ literal color (เช่น Leaflet circleMarker fillColor)
export function severityHex(pm25) {
  return levelForPm25(pm25).hex;
}

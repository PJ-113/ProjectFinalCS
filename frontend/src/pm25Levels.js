// ระดับ/สี PM2.5 ตรงกับ PM25_LEVELS ใน app.py (มาตรฐาน คพ. 2566) — โมดูลกลางเดียว ไม่ให้ค่า/สี/label
// กระจายซ้ำในหลายไฟล์ (ก่อนหน้านี้มีสำเนาเดียวอยู่ใน ForecastView.vue เท่านั้น, MapView.vue ใช้ชุด
// เดียวกัน) ต้อง match string สถานะให้ตรงเป๊ะกับที่ pm25_status() คืนกลับมา ไม่งั้นจะ fall through
// ไปโชว์สี/ระดับผิด (เคยเป็นบั๊กมาก่อนตอนยังมีแค่ 3 ระดับ)

const PM25_LEVELS = [
  { max: 15, status: "ดีมาก", cssVar: "--excellent", statusClass: "status-excellent", hex: "#3b82f6" },
  { max: 25, status: "ดี", cssVar: "--good", statusClass: "status-good", hex: "#22c55e" },
  { max: 37.5, status: "ปานกลาง", cssVar: "--moderate", statusClass: "status-moderate", hex: "#eab308" },
  { max: 75, status: "เริ่มมีผลกระทบต่อสุขภาพ", cssVar: "--unhealthy", statusClass: "status-unhealthy", hex: "#f97316" },
  { max: Infinity, status: "มีผลกระทบต่อสุขภาพ", cssVar: "--bad", statusClass: "status-bad", hex: "#ef4444" },
];

function levelForPm25(pm25) {
  return PM25_LEVELS.find((l) => pm25 <= l.max) || PM25_LEVELS[PM25_LEVELS.length - 1];
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
// แทนการพึ่ง CSS var() ซึ่งไม่ใช่ทุก renderer จะ resolve ให้แน่นอน
export function severityHex(pm25) {
  return levelForPm25(pm25).hex;
}

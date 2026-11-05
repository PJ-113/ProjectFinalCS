<script setup>
import { ref, computed, onMounted, onUnmounted, watch, nextTick } from "vue";
import {
  Laugh, Smile, Meh, Frown, Angry, Thermometer, Droplet, Wind,
  MapPin, ChevronDown, TriangleAlert, Info, Sparkles,
  Sun, CloudSun, Cloud, CloudFog, CloudRain, CloudSnow, CloudLightning,
  TrendingUp, TrendingDown,
} from "@lucide/vue";
import { fetchForecast, fetchCurrentPm25, buildWarningMessage, formatCardTime } from "../notifications";
import { riskScoreState, setActivity, ACTIVITY_LEVELS, computeRiskScore, riskScoreAdvice } from "../riskScore";
import { provinceState, selectProvince } from "../provinces";
import { statusClass, severityVar, PM25_LEVELS, scaleMarkerPct, adviceForStatus } from "../pm25Levels";
import ProvincePickerSheet from "../components/ProvincePickerSheet.vue";

// SHOW_EXTRA_CARDS = false ซ่อนการ์ดทั้งหมด, SAMPLE_EXTRA_CARDS_DATA = true ใช้ข้อมูลตัวอย่างแทน API
const SHOW_EXTRA_CARDS = true;
const SAMPLE_EXTRA_CARDS_DATA = false;
// SHOW_UPDATED_LABEL = true โชว์ป้าย "อัปเดต ... น." บนการ์ด PM2.5 ตอนนี้ (เคยปิดไว้ก่อนนำเสนอ 2026-10-20
// เปิดกลับคืนตามคำขอผู้ใช้ 2026-11-05 — ให้เห็นชัดว่า "ตอนนี้" อ้างอิงเวลาไหนจริงๆ เพราะ Air4Thai เองก็มี
// ความหน่วงรายงานเป็นครั้งคราว ดู CHANGELOG (191))
const SHOW_UPDATED_LABEL = true;
// SHOW_CHART_LEGEND = false ซ่อน legend (จุด PM2.5/เส้นเกณฑ์ 37.5) + คำอธิบายใต้กราฟพยากรณ์อากาศรายชั่วโมง
const SHOW_CHART_LEGEND = false;

const SAMPLE_BEST_TIMES = [
  { time: "05:00", display_time: "05:00", pm25: 11 },
  { time: "04:00", display_time: "04:00", pm25: 19 },
  { time: "18:00", display_time: "18:00", pm25: 27 },
];
// ค่าตัวอย่าง คำนวณจากสูตรจริงใน app.py (compute_risk_score())
const SAMPLE_RISK_TABLE = {
  เบา: {
    score: 45, level: "ปานกลาง", advice: "ความเสี่ยงปานกลาง ทำได้แต่ควรสังเกตอาการตนเอง",
    times: [{ time: "05:00", display_time: "05:00", riskScore: 13 }, { time: "04:00", display_time: "04:00", riskScore: 17 }, { time: "06:00", display_time: "06:00", riskScore: 22 }],
  },
  ปานกลาง: {
    score: 53, level: "ปานกลาง", advice: "ความเสี่ยงปานกลาง ทำได้แต่ควรสังเกตอาการตนเอง",
    times: [{ time: "05:00", display_time: "05:00", riskScore: 20 }, { time: "04:00", display_time: "04:00", riskScore: 26 }, { time: "06:00", display_time: "06:00", riskScore: 32 }],
  },
  หนัก: {
    score: 60, level: "เริ่มมีผลกระทบต่อสุขภาพ", advice: "ความเสี่ยงเริ่มสูง ควรลดความหนักหรือระยะเวลาของกิจกรรมนี้",
    times: [{ time: "05:00", display_time: "05:00", riskScore: 30 }, { time: "04:00", display_time: "04:00", riskScore: 36 }, { time: "06:00", display_time: "06:00", riskScore: 42 }],
  },
};
const SAMPLE_DAILY = {
  avg: 38.4,
  text: "วันนี้ฝุ่น PM2.5 ในภาพรวมอยู่ในระดับปานกลาง ช่วงเช้ามืดอากาศดีที่สุด เหมาะสำหรับออกกำลังกายกลางแจ้ง " +
    "ส่วนช่วงบ่ายถึงเย็นค่าฝุ่นมีแนวโน้มสูงขึ้นเล็กน้อย กลุ่มเสี่ยงควรเลี่ยงกิจกรรมหนักในช่วงดังกล่าว",
};

const state = ref("loading"); // "loading" | "error" | "content"
const errorMessage = ref("ไม่สามารถโหลดข้อมูลได้ ลองใหม่อีกครั้ง");

const current = ref({ pm25: "-", datetime_th: "-", status: "" });
const currentWeather = ref({ temp_c: null, humidity_pct: null, wind_speed_kmh: null });
const dailySummary = ref({ daily_avg_pm25: null, explanation: "" });
const forecast = ref([]);
// เส้นเวลา 49 จุด (อดีต 24 ชม. + ตอนนี้ + พยากรณ์ 24 ชม.) จาก data.hourly_timeline — ใช้กับกราฟ "พยากรณ์อากาศ"
const hourlyTimeline = ref([]);
const chartEl = ref(null);
const weatherScrollEl = ref(null);
const highPollutionRisk = ref(false);
const riskGroup = ref("ทั่วไป");
const fallbackHorizons = ref([]);
// สถานะข้อมูลที่ใช้ทำพยากรณ์ (data_quality จาก /api/forecast): target_stale = reading ล่าสุดของจังหวัดนี้เก่ากว่า 6 ชม.
const dataQuality = ref({ target_stale: false, target_age_hours: 0 });
const provinceInfo = ref({ th: "", en: "", is_zero_shot: false });
const riskScoreConfig = ref(null);

const pickerOpen = ref(false);
// index ของแท่งที่แตะในกราฟ (null = ยังไม่เลือก)
const selectedIdx = ref(null);

// statusClass/severityVar/PM25_LEVELS อยู่ใน pm25Levels.js (โมดูลกลาง ใช้ร่วมกับ MapView.vue)

// ชื่อคลาสตั้ง --hero-bg/--hero-ink ของการ์ดหลัก (tint ต่อระดับ) — ดูบล็อก .aqi-* ใน <style>
function bodyClassFor(status) {
  if (status === "ดีมาก") return "aqi-excellent";
  if (status === "ดี") return "aqi-good";
  if (status === "ปานกลาง") return "aqi-moderate";
  if (status === "เริ่มมีผลกระทบต่อสุขภาพ") return "aqi-unhealthy";
  return "aqi-bad";
}

// ไอคอนหน้าคน (ชุดเดียวกับ Lucide ทั้งการ์ด) ไล่ระดับความรุนแรงจากยิ้มกว้างไปหน้าบึ้ง
function moodIconFor(status) {
  if (status === "ดีมาก") return Laugh;
  if (status === "ดี") return Smile;
  if (status === "ปานกลาง") return Meh;
  if (status === "เริ่มมีผลกระทบต่อสุขภาพ") return Frown;
  return Angry;
}

function barColor(pm25) {
  return `var(${severityVar(pm25)})`;
}

// ไอคอนสภาพอากาศจากรหัส WMO ที่ Open-Meteo ส่งมา (hourly_timeline[].weather_code) — ใช้กับการ์ดรายชั่วโมง
function weatherIconFor(code) {
  if (code == null) return Cloud;
  if (code === 0) return Sun;
  if (code === 1 || code === 2) return CloudSun;
  if (code === 3) return Cloud;
  if (code === 45 || code === 48) return CloudFog;
  if ([51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82].includes(code)) return CloudRain;
  if ([71, 73, 75, 77, 85, 86].includes(code)) return CloudSnow;
  if ([95, 96, 99].includes(code)) return CloudLightning;
  return Cloud;
}

// ความกว้างต่อ 1 ชั่วโมงในกราฟ/แถวการ์ด (px) — ใช้ค่าเดียวกันทั้งคู่ให้คอลัมน์ตรงกัน
const WEATHER_POINT_GAP = 64;
const WEATHER_PADDING = { top: 14, right: 16, bottom: 26, left: 16 };

function drawWeatherChart(pts) {
  const svg = chartEl.value;
  if (!svg) return;
  svg.innerHTML = "";
  const padding = WEATHER_PADDING;
  const width = padding.left + padding.right + pts.length * WEATHER_POINT_GAP;
  const height = 130;
  svg.setAttribute("width", width);
  svg.setAttribute("height", height);
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  if (!pts.length) return;

  const plotH = height - padding.top - padding.bottom;
  const xFor = (i) => padding.left + i * WEATHER_POINT_GAP + WEATHER_POINT_GAP / 2;

  // ซูมแกน Y ให้พอดีกับช่วงค่าจริงที่เกิดขึ้น (ไม่ตรึง 0-37.5 เหมือนเดิม) — วันที่ฝุ่นน้อยและแกว่งอยู่แค่
  // 15-20 µg/m³ เดิมเส้นจะแบนราบเพราะถูกบีบให้อยู่ในสเกล 0-37.5 เต็มช่วง ทั้งที่ความแกว่งจริงมีความหมาย
  function computeScale(key) {
    const vals = pts.map((p) => p[key]).filter((v) => v != null);
    if (!vals.length) return { lo: 0, hi: 1, fn: () => padding.top + plotH / 2 };
    const minV = Math.min(...vals);
    const maxV = Math.max(...vals);
    const span = maxV - minV;
    const pad = span > 0 ? span * 0.2 : Math.max(1, maxV * 0.1); // กันเส้นชิดขอบบน/ล่างเกินไป, กันหาร 0 ตอนค่าแบนสนิท
    const lo = Math.max(0, minV - pad);
    const hi = maxV + pad;
    const range = hi - lo || 1;
    return { lo, hi, minV, maxV, fn: (v) => padding.top + plotH - ((v - lo) / range) * plotH };
  }
  const pmScale = computeScale("pm25");
  const yPm = pmScale.fn;

  const ns = "http://www.w3.org/2000/svg";

  // แรเงาช่วงอดีต (ก่อน "ตอนนี้") ให้แยกจากช่วงพยากรณ์ได้ด้วยตาทันที ไม่ต้องกดดูทีละจุด
  const nowIdx = pts.findIndex((p) => p.is_now);
  if (nowIdx >= 0) {
    const nowX = xFor(nowIdx);
    const pastBand = document.createElementNS(ns, "rect");
    pastBand.setAttribute("x", padding.left);
    pastBand.setAttribute("y", padding.top);
    pastBand.setAttribute("width", Math.max(0, nowX - padding.left));
    pastBand.setAttribute("height", plotH);
    pastBand.setAttribute("class", "past-band");
    svg.appendChild(pastBand);

    // ป้าย "ข้อมูลย้อนหลัง" 2 จุด — จุดแรกชิดขอบซ้ายสุดของกราฟ (เห็นตอนเลื่อนไปสุดอดีต) จุดที่สองชิดเส้น
    // "ตอนนี้" (เห็นตอนเลื่อนมาดูรอยต่ออดีต/ปัจจุบัน ซึ่งเป็นตำแหน่งที่ค่าเริ่มต้นเลื่อนมาแสดงอยู่แล้ว) —
    // เดิมมีแค่จุดเดียวที่ขอบซ้ายสุด ถ้าช่วงอดีตยาว (เลื่อนมาไกลจากขอบซ้าย) จะมองไม่เห็นป้ายเลย
    const pastLabel = document.createElementNS(ns, "text");
    pastLabel.setAttribute("x", padding.left + 6);
    pastLabel.setAttribute("y", padding.top + 12);
    pastLabel.setAttribute("class", "past-label");
    pastLabel.textContent = "ข้อมูลย้อนหลัง";
    svg.appendChild(pastLabel);

    if (nowX - padding.left > WEATHER_POINT_GAP * 2) { // เว้นระยะพอสมควรจากป้ายแรกถึงไม่ซ้อนทับกัน
      const pastLabelNearNow = document.createElementNS(ns, "text");
      pastLabelNearNow.setAttribute("x", nowX - 6);
      pastLabelNearNow.setAttribute("y", padding.top + 12);
      pastLabelNearNow.setAttribute("text-anchor", "end");
      pastLabelNearNow.setAttribute("class", "past-label");
      pastLabelNearNow.textContent = "ข้อมูลย้อนหลัง";
      svg.appendChild(pastLabelNearNow);
    }

    const line = document.createElementNS(ns, "line");
    line.setAttribute("x1", nowX); line.setAttribute("x2", nowX);
    line.setAttribute("y1", padding.top); line.setAttribute("y2", height - padding.bottom);
    line.setAttribute("class", "now-line");
    svg.appendChild(line);
  }

  // เส้นเกณฑ์ 37.5 µg/m³ — โชว์เฉพาะตอนอยู่ในช่วงที่มองเห็นจริง (วันฝุ่นน้อยไม่ต้องลากเส้นที่ไกลลิบไม่เกี่ยวข้อง)
  if (37.5 >= pmScale.lo && 37.5 <= pmScale.hi) {
    const thrLine = document.createElementNS(ns, "line");
    thrLine.setAttribute("x1", padding.left); thrLine.setAttribute("x2", width - padding.right);
    thrLine.setAttribute("y1", yPm(37.5)); thrLine.setAttribute("y2", yPm(37.5));
    thrLine.setAttribute("class", "thr-grid-line");
    svg.appendChild(thrLine);
  }

  const labelStep = 2;
  pts.forEach((p, i) => {
    if (i % labelStep !== 0 && !p.is_now) return;
    const label = document.createElementNS(ns, "text");
    label.setAttribute("x", xFor(i));
    label.setAttribute("y", height - padding.bottom + 16);
    label.setAttribute("text-anchor", "middle");
    label.setAttribute("class", "axis-label");
    label.textContent = p.is_now ? "ตอนนี้" : p.time;
    svg.appendChild(label);
  });

  // เส้นโค้งมน (Catmull-Rom แปลงเป็น cubic Bezier) แทนเส้นตรงต่อจุด — ช่วงที่ค่าแกว่งน้อย (เช่น 15-20 µg/m³)
  // เส้นตรงต่อกันจะดูสะดุด/ตายด้าน เส้นโค้งทำให้ดูเนียนและมีมิติขึ้นแม้ความต่างจริงจะน้อย
  function smoothPathD(runPoints) {
    if (runPoints.length < 2) return "";
    let d = `M${runPoints[0].x},${runPoints[0].y} `;
    for (let i = 0; i < runPoints.length - 1; i++) {
      const p0 = runPoints[i - 1] || runPoints[i];
      const p1 = runPoints[i];
      const p2 = runPoints[i + 1];
      const p3 = runPoints[i + 2] || p2;
      const c1x = p1.x + (p2.x - p0.x) / 6, c1y = p1.y + (p2.y - p0.y) / 6;
      const c2x = p2.x - (p3.x - p1.x) / 6, c2y = p2.y - (p3.y - p1.y) / 6;
      d += `C${c1x},${c1y} ${c2x},${c2y} ${p2.x},${p2.y} `;
    }
    return d.trim();
  }

  // แบ่งจุดเป็นช่วงต่อเนื่อง (คั่นตรงที่ไม่มีค่า) เพราะอดีต/ปัจจุบัน/อนาคตบางจุดอาจไม่มีข้อมูล (null)
  function toRuns(key, yFn) {
    const runs = []; let current = [];
    pts.forEach((p, i) => {
      const v = p[key];
      if (v == null) { if (current.length) runs.push(current); current = []; return; }
      current.push({ x: xFor(i), y: yFn(v) });
    });
    if (current.length) runs.push(current);
    return runs;
  }

  function drawPath(key, yFn, color, gradientId) {
    const runs = toRuns(key, yFn);
    const baseY = padding.top + plotH; // เส้นฐานล่างสุดของพื้นที่กราฟ สำหรับปิดรูปแรเงาใต้เส้น

    // พื้นที่ไล่สีใต้เส้น (เริ่มทึบที่เส้น จางหายไปที่ฐาน) — เพิ่มน้ำหนักให้กราฟดูมีเนื้อหา ไม่โล่งจนดูว่างเปล่า
    const gradient = document.createElementNS(ns, "linearGradient");
    gradient.setAttribute("id", gradientId);
    gradient.setAttribute("x1", "0"); gradient.setAttribute("y1", "0");
    gradient.setAttribute("x2", "0"); gradient.setAttribute("y2", "1");
    const stop1 = document.createElementNS(ns, "stop");
    stop1.setAttribute("offset", "0%"); stop1.setAttribute("stop-color", color); stop1.setAttribute("stop-opacity", "0.28");
    const stop2 = document.createElementNS(ns, "stop");
    stop2.setAttribute("offset", "100%"); stop2.setAttribute("stop-color", color); stop2.setAttribute("stop-opacity", "0");
    gradient.appendChild(stop1); gradient.appendChild(stop2);
    let defs = svg.querySelector("defs");
    if (!defs) { defs = document.createElementNS(ns, "defs"); svg.insertBefore(defs, svg.firstChild); }
    defs.appendChild(gradient);

    runs.forEach((run) => {
      if (run.length < 2) return;
      const lineD = smoothPathD(run);
      const areaD = `${lineD} L${run[run.length - 1].x},${baseY} L${run[0].x},${baseY} Z`;
      const area = document.createElementNS(ns, "path");
      area.setAttribute("d", areaD);
      area.setAttribute("fill", `url(#${gradientId})`);
      area.setAttribute("stroke", "none");
      svg.appendChild(area);

      const line = document.createElementNS(ns, "path");
      line.setAttribute("d", lineD);
      line.setAttribute("fill", "none");
      line.setAttribute("stroke", color);
      line.setAttribute("stroke-width", "2");
      line.setAttribute("stroke-linecap", "round");
      line.setAttribute("stroke-linejoin", "round");
      svg.appendChild(line);
    });

    // จุดกลมบนเส้นทุกชั่วโมงที่มีค่า (ไม่ใช่แค่เส้นเปล่า) — เน้นจุด "ตอนนี้"/ที่เลือกให้ใหญ่กว่าจุดอื่น
    pts.forEach((p, i) => {
      const v = p[key];
      if (v == null) return;
      const circle = document.createElementNS(ns, "circle");
      circle.setAttribute("cx", xFor(i));
      circle.setAttribute("cy", yFn(v));
      circle.setAttribute("r", p.is_now || i === selectedIdx.value ? "4" : "2.5");
      circle.setAttribute("fill", color);
      svg.appendChild(circle);
    });
  }
  drawPath("pm25", yPm, "var(--w-pm25)", "pm25-area-gradient");
}

// เลื่อนแถวการ์ด/กราฟให้ "ตอนนี้" อยู่ชิดขอบซ้าย (พฤติกรรมเดิมก่อนเพิ่มอดีต — อดีตเลื่อนย้อนดูเพิ่มเอง)
function scrollToNow() {
  const el = weatherScrollEl.value;
  const nowIdx = hourlyTimeline.value.findIndex((p) => p.is_now);
  if (!el || nowIdx < 0) return;
  el.scrollLeft = Math.max(0, nowIdx * WEATHER_POINT_GAP - 8);
}

const pageClass = computed(() => bodyClassFor(current.value.status));
const dailyAvgDisplay = computed(() => {
  if (SAMPLE_EXTRA_CARDS_DATA) return SAMPLE_DAILY.avg;
  return dailySummary.value.daily_avg_pm25 !== null && dailySummary.value.daily_avg_pm25 !== undefined
    ? dailySummary.value.daily_avg_pm25
    : "-";
});
const dailyTextDisplay = computed(() =>
  SAMPLE_EXTRA_CARDS_DATA ? SAMPLE_DAILY.text : dailySummary.value.explanation || "กำลังเตรียมข้อมูลสำหรับวันนี้"
);
// horizon (1..24) -> เวลานาฬิกาของจุดนั้นใน forecast[] — ใช้ display_time (ตรง convention เดียวกับเว็บ
// OpenAQ เอง, ดู OPENAQ_DISPLAY_LABEL_SHIFT_H ใน app.py) ไม่ใช่ time ดิบที่ backend ใช้คำนวณภายใน
const fallbackTimesText = computed(() =>
  fallbackHorizons.value.map((h) => forecast.value[h - 1]?.display_time).filter(Boolean).join(", ")
);

// ข้อความเตือนช่วงมลพิษสูง (ปรับตาม risk_group) อยู่ใน notifications.js (ใช้ร่วมกับ AppHeader.vue)
const warningMessage = computed(() => buildWarningMessage(riskGroup.value));

const markerPct = computed(() => scaleMarkerPct(current.value.pm25));
const currentAdvice = computed(() => adviceForStatus(current.value.status));
const updatedLabel = computed(() =>
  current.value.datetime_th && current.value.datetime_th !== "-" ? formatCardTime(current.value.datetime_th) : ""
);
const activeLevelIndex = computed(() => PM25_LEVELS.findIndex((l) => l.status === current.value.status));

// ช่วงเวลาที่อากาศดีที่สุด (ค่าเดียว ไม่ใช่ top-3) กรองเฉพาะ 06:00-20:00 ก่อน sort, fallback ไป forecast
// ทั้งหมดถ้าว่าง
const topBestTime = computed(() => {
  if (SAMPLE_EXTRA_CARDS_DATA) return SAMPLE_BEST_TIMES[0] ?? null;
  const activeHours = forecast.value.filter((f) => f.display_hour_thai >= 6 && f.display_hour_thai <= 20);
  const pool = activeHours.length > 0 ? activeHours : forecast.value;
  return [...pool].sort((a, b) => a.pm25 - b.pm25)[0] ?? null;
});

// Personalized Risk Score (proposal) — ผสาน pm25 ปัจจุบัน + risk_group + activity (เลือกสดหน้าเว็บ)
// เป็นคะแนนเดียว 0-100 สูตร/offset ทั้งหมดมาจาก riskScoreConfig ที่ backend ส่งมา
const currentRiskScore = computed(() => {
  if (SAMPLE_EXTRA_CARDS_DATA) {
    const row = SAMPLE_RISK_TABLE[riskScoreState.activity] ?? SAMPLE_RISK_TABLE["ปานกลาง"];
    return { score: row.score, level: row.level, effective_pm25: null };
  }
  return computeRiskScore(current.value.pm25, riskScoreConfig.value, riskScoreState.activity);
});
const currentRiskAdvice = computed(() =>
  SAMPLE_EXTRA_CARDS_DATA
    ? (SAMPLE_RISK_TABLE[riskScoreState.activity] ?? SAMPLE_RISK_TABLE["ปานกลาง"]).advice
    : riskScoreAdvice(currentRiskScore.value.level)
);

// เหมือน topBestTimes แต่ personalize ตามกิจกรรมที่เลือก (sort ด้วยคะแนนความเสี่ยงแทน pm25 ดิบ)
const bestActivityTimes = computed(() => {
  if (SAMPLE_EXTRA_CARDS_DATA) return (SAMPLE_RISK_TABLE[riskScoreState.activity] ?? SAMPLE_RISK_TABLE["ปานกลาง"]).times;
  if (!riskScoreConfig.value) return [];
  const activeHours = forecast.value.filter((f) => f.display_hour_thai >= 6 && f.display_hour_thai <= 20);
  const pool = activeHours.length > 0 ? activeHours : forecast.value;
  return [...pool]
    .map((f) => ({ ...f, riskScore: computeRiskScore(f.pm25, riskScoreConfig.value, riskScoreState.activity).score }))
    .sort((a, b) => a.riskScore - b.riskScore)
    .slice(0, 3);
});

const selectedPoint = computed(() => (selectedIdx.value == null ? null : hourlyTimeline.value[selectedIdx.value] || null));

// อธิบาย "ทำไมโมเดลพยากรณ์ชั่วโมงนี้ถึงเป็นค่านี้" ด้วย SHAP (ตอบคอมเมนต์กรรมการ: ต้องมีอิทธิพลจาก SHAP
// กำกับคู่กับ Actual vs Predicted) — คำนวณเฉพาะชั่วโมงพยากรณ์ที่ผู้ใช้แตะเลือกจริง ไม่ใช่ทั้ง 24 ชม. (ต้นทุน
// คำนวณ SHAP สูง เหตุผลเดียวกับที่ NLG รายวันอธิบายแค่ horizon ตัวแทน ดู REP_HORIZON_FOR_SHAP ใน app.py)
const shapLoading = ref(false);
const shapResult = ref(null); // { model_used, horizon, factors } หรือ null = ยังไม่ได้โหลด/ไม่เกี่ยวข้อง
const shapExpanded = ref(false); // false = โชว์แค่ SHAP_LIST_COLLAPSED_N ตัวแรก, true = โชว์ทั้งหมดที่ backend ส่งมา
const shapPanelOpen = ref(false); // false = โชว์แค่ปุ่ม "ดูปัจจัยที่มีผล", true = กดแล้วค่อยแสดงรายการจริง
const SHAP_LIST_COLLAPSED_N = 3;

async function loadShapFactors(province, horizon) {
  shapLoading.value = true;
  shapResult.value = null;
  try {
    const res = await fetch(`/api/forecast/shap?province=${province}&horizon=${horizon}`);
    shapResult.value = res.ok ? await res.json() : null;
  } catch {
    shapResult.value = null;
  } finally {
    shapLoading.value = false;
  }
}

const shapFactorsShown = computed(() => {
  const factors = shapResult.value?.factors;
  if (!factors) return [];
  return shapExpanded.value ? factors : factors.slice(0, SHAP_LIST_COLLAPSED_N);
});

watch(selectedPoint, (point) => {
  shapExpanded.value = false;
  shapPanelOpen.value = false;
  if (point && point.is_forecast && !point.is_gap_fill && point.horizon != null) {
    loadShapFactors(provinceState.selected, point.horizon);
  } else {
    shapLoading.value = false;
    shapResult.value = null;
  }
});

// แนวโน้มของชั่วโมงที่เลือก เทียบกับ "ตอนนี้" — ใช้ pm25 ที่ backend คำนวณมาแล้วตรงๆ (คือ residual ของโมเดล
// pm25_t+h - pm25_now นั่นเอง ไม่ต้องคำนวณซ้ำฝั่ง frontend) ไม่โชว์ถ้าใกล้ 0 เกินไปจนดูเป็น noise — ไม่โชว์
// สำหรับจุด "ช่องว่าง" (is_gap_fill) ด้วย เพราะผู้ใช้มองว่าควรปฏิบัติเหมือนข้อมูลย้อนหลัง (ไม่ใช่พยากรณ์ที่
// ควรเชื่อถือรายละเอียดเท่าพยากรณ์จริง)
const trendInfo = computed(() => {
  if (!selectedPoint.value?.is_forecast || selectedPoint.value?.is_gap_fill || current.value.pm25 === "-") return null;
  const diff = selectedPoint.value.pm25 - current.value.pm25;
  if (Math.abs(diff) < 0.5) return { label: "ใกล้เคียงกับค่าตอนนี้", diff, kind: "flat" };
  return { label: diff > 0 ? "ฝุ่นมีแนวโน้มสูงขึ้น" : "ฝุ่นมีแนวโน้มลดลง", diff, kind: diff > 0 ? "up" : "down" };
});


async function loadForecast() {
  state.value = "loading";
  selectedIdx.value = null;
  try {
    const data = await fetchForecast();
    current.value = data.current;
    currentWeather.value = data.current_weather || {};
    dailySummary.value = data.daily_summary || {};
    forecast.value = data.forecast;
    hourlyTimeline.value = data.hourly_timeline || [];
    highPollutionRisk.value = data.high_pollution_risk;
    riskGroup.value = data.risk_group || "ทั่วไป";
    fallbackHorizons.value = data.fallback_triggered_horizons || [];
    dataQuality.value = data.data_quality || { target_stale: false, target_age_hours: 0 };
    provinceInfo.value = data.province || { th: "", en: "", is_zero_shot: false };
    riskScoreConfig.value = data.risk_score_config || null;
    state.value = "content";
    await nextTick();
    drawWeatherChart(hourlyTimeline.value);
    scrollToNow();
  } catch (err) {
    errorMessage.value = err.dataUnavailable ? err.message : "ไม่สามารถโหลดข้อมูลได้ ลองใหม่อีกครั้ง";
    state.value = "error";
    console.error("โหลดข้อมูลพยากรณ์ล้มเหลว:", err);
  }
}

// สลับจังหวัดจาก ProvincePickerSheet -> provinceState.selected เปลี่ยน -> โหลดพยากรณ์ใหม่
watch(() => provinceState.selected, loadForecast);

// วาดกราฟใหม่ตอนเลือก/ยกเลิกเลือกการ์ด ให้จุดที่เลือกอยู่ขยายใหญ่ขึ้นตามจริง (ไม่ใช่แค่ครั้งแรกตอนโหลดข้อมูล)
watch(selectedIdx, () => drawWeatherChart(hourlyTimeline.value));

onMounted(loadForecast);

// รีเฟรชแค่การ์ด "ปัจจุบัน" เป็นระยะผ่าน /api/current_pm25 (แยกจาก loadForecast() ทั้งก้อน) อ่าน
// provinceState.selected สดทุก tick จึงตามจังหวัดที่เลือกอยู่เสมอโดยไม่ต้อง clear/ตั้ง interval ใหม่
const CURRENT_PM25_POLL_MS = 2 * 60 * 1000;
let currentPm25IntervalId = null;

async function refreshCurrentPm25() {
  const province = provinceState.selected;
  const data = await fetchCurrentPm25(province);
  // เงียบๆ ข้ามถ้า fetch พลาด (data=null) หรือผู้ใช้สลับจังหวัดไปแล้วระหว่างรอ response (กัน race เขียนทับ
  // ด้วยค่าของจังหวัดเก่า)
  if (data && provinceState.selected === province) {
    current.value = { pm25: data.pm25, datetime_th: data.datetime_th, status: data.status };
  }
}

onMounted(() => {
  currentPm25IntervalId = setInterval(refreshCurrentPm25, CURRENT_PM25_POLL_MS);
});
onUnmounted(() => {
  if (currentPm25IntervalId) clearInterval(currentPm25IntervalId);
});

defineExpose({ loadForecast });
</script>

<template>
  <div class="forecast-page" :class="pageClass">
    <div class="wrap">
      <div class="top-row">
        <button type="button" class="province-chip" aria-haspopup="dialog" @click="pickerOpen = true">
          <MapPin :size="18" :stroke-width="2" aria-hidden="true" />
          <span class="province-chip-name">{{ provinceInfo.th || "..." }}</span>
          <span v-if="provinceInfo.is_zero_shot" class="zs-badge">zero-shot</span>
          <ChevronDown :size="16" :stroke-width="2" aria-hidden="true" />
        </button>
        <span class="page-title">พยากรณ์ PM2.5</span>
      </div>

      <!-- Loading skeleton -->
      <div v-if="state === 'loading'" class="stack">
        <div class="skeleton skeleton-hero"></div>
        <div class="card"><div class="skeleton skeleton-line"></div><div class="skeleton skeleton-line" style="width: 60%"></div></div>
        <div class="card"><div class="skeleton skeleton-chart"></div></div>
      </div>

      <!-- Error state -->
      <div v-else-if="state === 'error'" class="card error-box">
        <p>{{ errorMessage }}</p>
        <button type="button" class="retry-btn" @click="loadForecast">ลองอีกครั้ง</button>
      </div>

      <!-- Content -->
      <div v-else class="layout">
        <div class="col-primary">
          <div v-if="highPollutionRisk" class="banner banner-warn" role="alert">
            <TriangleAlert :size="22" :stroke-width="2" aria-hidden="true" />
            <div>
              <div class="banner-title">ช่วงนี้ฝุ่นอยู่ในระดับสูง</div>
              <div class="banner-text">{{ warningMessage }}</div>
            </div>
          </div>

          <div v-if="dataQuality.target_stale" class="banner banner-warn" role="alert">
            <TriangleAlert :size="22" :stroke-width="2" aria-hidden="true" />
            <div class="banner-text">
              ข้อมูลล่าสุดของสถานีตรวจวัดในจังหวัดนี้เก่ากว่า {{ Math.round(dataQuality.target_age_hours) }} ชั่วโมง
              (สถานีอาจขัดข้อง) ค่าที่แสดงอาจไม่ตรงกับสถานการณ์ปัจจุบัน
            </div>
          </div>

          <div v-if="dataQuality.input_imputed" class="banner banner-info">
          <Info :size="22" :stroke-width="2" aria-hidden="true" />
          <div class="banner-text">
            สถานีตรวจวัดในจังหวัดนี้ขาดการรายงานประมาณ {{ dataQuality.input_imputed.k_hours }} ชั่วโมง
            ค่าที่แสดงเป็นค่าประมาณจากจังหวัดข้างเคียง (ค่าที่วัดได้ล่าสุด {{ dataQuality.input_imputed.last_observed_pm25 }} µg/m³
            เมื่อ {{ formatCardTime(dataQuality.input_imputed.last_observed_datetime_th) }}) ช่วงความเชื่อมั่นจึงกว้างกว่าปกติ
          </div>
        </div>

        <div v-if="dataQuality.series_source === 'stale_cache' || dataQuality.weather_source === 'climatology'" class="banner banner-info">
            <Info :size="22" :stroke-width="2" aria-hidden="true" />
            <div class="banner-text">
              <template v-if="dataQuality.series_source === 'stale_cache'">
                ระบบเชื่อมต่อแหล่งข้อมูลฝุ่นไม่ได้ชั่วคราว จึงใช้ข้อมูลที่บันทึกไว้ล่าสุด (เมื่อประมาณ {{ Math.max(1, Math.round(dataQuality.series_age_hours)) }} ชั่วโมงก่อน)
              </template>
              <template v-if="dataQuality.weather_source === 'climatology'">
                ระบบดึงสภาพอากาศล่าสุดไม่ได้ จึงใช้ค่าเฉลี่ยตามฤดูกาลของจังหวัดนี้แทน ความแม่นยำอาจลดลงเล็กน้อย
              </template>
            </div>
          </div>

        <div v-if="fallbackHorizons.length > 0" class="banner banner-info">
            <Info :size="22" :stroke-width="2" aria-hidden="true" />
            <div class="banner-text">
              ข้อมูลบางช่วงไม่ครบ ระบบจึงใช้ค่าพยากรณ์ชั่วคราว (ความเชื่อมั่นต่ำ)
              <template v-if="fallbackTimesText"> ที่เวลา {{ fallbackTimesText }} น.</template>
            </div>
          </div>

          <!-- Hero: ค่า PM2.5 ปัจจุบัน -->
          <section class="hero" aria-label="PM2.5 ตอนนี้">
            <div class="hero-head">
              <span class="hero-label">PM2.5 ตอนนี้<span v-if="dataQuality.input_imputed" class="estimate-tag">ค่าประมาณ</span></span>
              <span v-if="SHOW_UPDATED_LABEL && updatedLabel" class="hero-updated">อัปเดต {{ updatedLabel }}</span>
            </div>
            <div class="pm-value-row">
              <span class="pm-value">{{ current.pm25 }}</span>
              <span class="pm-unit">µg/m³</span>
            </div>
            <div class="level-chip" :class="statusClass(current.status)">
              <component :is="moodIconFor(current.status)" :size="28" :stroke-width="2" class="level-icon" />
              <span class="level-text">{{ current.status }}</span>
            </div>
            <div class="scale" aria-hidden="true">
              <div class="scale-track">
                <span
                  v-for="(l, i) in PM25_LEVELS"
                  :key="l.status"
                  class="scale-seg"
                  :class="{ 'scale-seg-active': i === activeLevelIndex }"
                  :style="{ background: `var(${l.cssVar})` }"
                ></span>
                <span class="scale-marker" :style="{ left: markerPct + '%' }"></span>
              </div>
              <div class="scale-ends"><span>ดีมาก</span><span>มีผลกระทบ</span></div>
            </div>
            <p class="hero-advice">{{ currentAdvice }}</p>
            <p class="hero-note">ค่ารายชั่วโมงล่าสุด ไม่ใช่ค่าเฉลี่ย 24 ชม. แบบ AQI ทางการ</p>

            <div v-if="currentWeather.temp_c != null" class="stat-tiles">
              <div class="stat-tile">
                <Thermometer :size="20" :stroke-width="2" aria-hidden="true" />
                <div class="stat-tile-value">{{ currentWeather.temp_c }}°C</div>
                <div class="stat-tile-label">อุณหภูมิ</div>
              </div>
              <div class="stat-tile">
                <Droplet :size="20" :stroke-width="2" aria-hidden="true" />
                <div class="stat-tile-value">{{ currentWeather.humidity_pct }}%</div>
                <div class="stat-tile-label">ความชื้น</div>
              </div>
              <div class="stat-tile">
                <Wind :size="20" :stroke-width="2" aria-hidden="true" />
                <div class="stat-tile-value">{{ currentWeather.wind_speed_kmh }}</div>
                <div class="stat-tile-label">กม./ชม. · ลม</div>
              </div>
            </div>
          </section>

          <!-- ช่วงเวลาที่ฝุ่นต่ำที่สุด -->
          <section v-if="SHOW_EXTRA_CARDS" class="card">
            <div class="card-head">
              <h2>ช่วงเวลาที่ฝุ่นต่ำที่สุด</h2>
              <span class="card-head-sub">24 ชม. ข้างหน้า</span>
            </div>
            <div v-if="topBestTime" class="best-single">
              <span class="best-single-time">{{ topBestTime.display_time }} น.</span>
              <span class="best-pm">
                <span class="dot" :style="{ background: barColor(topBestTime.pm25) }"></span>{{ topBestTime.pm25 }}
                <span class="best-unit">µg/m³</span>
              </span>
            </div>
          </section>
        </div>

        <div class="col-secondary">
          <!-- พยากรณ์อากาศรายชั่วโมง: การ์ด + กราฟเส้น เลื่อนดูได้ทั้งอดีต 24 ชม. และอนาคต 24 ชม. -->
          <section class="card chart-card">
            <div class="card-head chart-head">
              <div>
                <h2>พยากรณ์รายชั่วโมง</h2>
              </div>
            </div>
            <div class="weather-scroll" ref="weatherScrollEl">
              <div class="weather-inner" :style="{ width: (hourlyTimeline.length * WEATHER_POINT_GAP + 32) + 'px' }">
                <div class="hour-cards" role="group" aria-label="สภาพอากาศรายชั่วโมง อดีตและพยากรณ์">
                  <button
                    v-for="(point, i) in hourlyTimeline"
                    :key="point.time + '-' + i"
                    type="button"
                    class="hour-card"
                    :class="{
                      selected: selectedIdx === i,
                      'hour-card-now': point.is_now,
                      'hour-card-past': !point.is_forecast && !point.is_now,
                      'hour-card-gap': point.is_gap_fill,
                    }"
                    :aria-pressed="selectedIdx === i"
                    :aria-label="`${point.is_now ? 'ตอนนี้' : point.time} น. ${point.is_gap_fill ? '(ประมาณการ ข้อมูลจริงยังไม่มา) ' : (!point.is_forecast && !point.is_now ? '(ข้อมูลย้อนหลัง) ' : '')}PM2.5 ${point.pm25 ?? '-'}`"
                    @click="selectedIdx = selectedIdx === i ? null : i"
                  >
                    <span class="hour-card-time">{{ point.is_now ? "ตอนนี้" : point.time }}</span>
                    <component :is="weatherIconFor(point.weather_code)" :size="20" :stroke-width="1.8" class="hour-card-icon" />
                    <span class="hour-card-pm25">{{ point.pm25 ?? "-" }}</span>
                  </button>
                </div>
                <svg ref="chartEl" class="weather-chart-svg"></svg>
              </div>
            </div>
            <div class="chart-foot">
              <div v-if="SHOW_CHART_LEGEND" class="weather-legend">
                <span><span class="dot" style="background: var(--w-pm25)"></span>PM2.5</span>
                <span><span class="thr-swatch"></span>เกณฑ์ 37.5 µg/m³</span>
              </div>
              <span class="scroll-hint">ข้อมูลย้อนหลัง ← ตอนนี้ → พยากรณ์</span>
            </div>
            <p v-if="SHOW_CHART_LEGEND" class="chart-caption">
              พื้นที่แรเงา = ช่วงเวลาที่ผ่านไปแล้ว (ค่าจริง ไม่ใช่พยากรณ์) · เส้นประตั้งตรงกลาง = เวลาปัจจุบัน
              ("ตอนนี้") · จุดบนเส้นที่ใหญ่กว่าจุดอื่น = ชั่วโมงที่กำลังดูอยู่
            </p>
            <div class="bar-detail" aria-live="polite">
              <template v-if="selectedPoint">
                <div class="bar-detail-time">
                  {{ selectedPoint.is_now ? `ตอนนี้ (${String(selectedPoint.hour_thai).padStart(2, "0")}:00 น.)` : selectedPoint.time + " น." }}
                  <span v-if="!selectedPoint.is_forecast && !selectedPoint.is_now" class="hist-tag">ข้อมูลย้อนหลัง</span>
                  <span v-else-if="selectedPoint.is_gap_fill" class="hist-tag">ประมาณการ (ข้อมูลจริงยังไม่มา)</span>
                </div>
                <div class="bar-detail-main">
                  <span class="bar-detail-value">{{ selectedPoint.pm25 ?? "-" }}</span>
                  <span class="best-unit">µg/m³</span>
                </div>
                <div v-if="selectedPoint.is_forecast && !selectedPoint.is_gap_fill && selectedPoint.pm25_lo != null" class="bar-detail-meta">
                  คาดว่าค่าฝุ่น PM2.5 จะอยู่ระหว่าง {{ selectedPoint.pm25_lo }}–{{ selectedPoint.pm25_hi }} µg/m³
                </div>
                <div v-if="trendInfo" class="trend-row" :class="`trend-${trendInfo.kind}`">
                  <component
                    v-if="trendInfo.kind !== 'flat'"
                    :is="trendInfo.kind === 'up' ? TrendingUp : TrendingDown"
                    :size="16" :stroke-width="2.5"
                  />
                  <span>
                    {{ trendInfo.label }}
                    <template v-if="trendInfo.kind !== 'flat'">({{ trendInfo.diff > 0 ? "+" : "" }}{{ trendInfo.diff.toFixed(1) }} µg/m³ จากตอนนี้)</template>
                  </span>
                </div>
                <div v-if="selectedPoint.is_forecast && !selectedPoint.is_gap_fill" class="shap-block">
                  <button
                    type="button"
                    class="shap-toggle shap-reveal-btn"
                    :aria-expanded="shapPanelOpen"
                    @click="shapPanelOpen = !shapPanelOpen"
                  >
                    <ChevronDown :size="14" :stroke-width="2.5" :class="{ 'shap-reveal-icon-open': shapPanelOpen }" />
                    {{ shapPanelOpen ? "ซ่อนปัจจัยที่มีผล" : "ดูปัจจัยที่มีผลต่อค่าพยากรณ์นี้" }}
                  </button>
                  <template v-if="shapPanelOpen">
                    <div v-if="shapLoading" class="shap-loading">กำลังวิเคราะห์ปัจจัย...</div>
                    <div v-else-if="shapResult?.model_used === 'naive_fallback'" class="shap-loading">
                      ชั่วโมงนี้ใช้ค่าคงที่ล่าสุด (Persistence) ไม่มีปัจจัยให้วิเคราะห์
                    </div>
                    <template v-else-if="shapResult?.factors?.length">
                      <div class="shap-label">ปัจจัยที่มีอิทธิพลต่อค่าพยากรณ์นี้มากที่สุด</div>
                      <ul class="shap-list">
                        <li v-for="f in shapFactorsShown" :key="f.feature" class="shap-item">
                          <component
                            :is="f.direction === 'up' ? TrendingUp : TrendingDown"
                            :size="14" :stroke-width="2.5"
                            :class="f.direction === 'up' ? 'shap-up' : 'shap-down'"
                          />
                          <span class="shap-item-name">{{ f.narrative_th }}</span>
                          <span class="shap-item-val" :class="f.direction === 'up' ? 'shap-up' : 'shap-down'">
                            {{ f.shap_value > 0 ? "+" : "" }}{{ f.shap_value.toFixed(2) }} µg/m³
                          </span>
                        </li>
                      </ul>
                      <button
                        v-if="shapResult.factors.length > SHAP_LIST_COLLAPSED_N"
                        type="button"
                        class="shap-toggle"
                        @click="shapExpanded = !shapExpanded"
                      >
                        {{ shapExpanded ? "แสดงน้อยลง" : `ดูเพิ่มเติม (${shapResult.factors.length - SHAP_LIST_COLLAPSED_N})` }}
                      </button>
                    </template>
                  </template>
                </div>
              </template>
              <div v-else class="bar-detail-empty">แตะการ์ดเพื่อดูรายละเอียดของชั่วโมงนั้น</div>
            </div>
          </section>

          <!-- ความเสี่ยงตามกิจกรรมของคุณ -->
          <section v-if="SHOW_EXTRA_CARDS" class="card">
            <h2>ความเสี่ยงตามกิจกรรมของคุณ</h2>
            <p class="card-caption">เลือกความหนักของกิจกรรมกลางแจ้งที่วางแผนจะทำ</p>
            <div class="segmented" role="group" aria-label="ความหนักของกิจกรรม">
              <button
                v-for="level in ACTIVITY_LEVELS"
                :key="level"
                type="button"
                class="segmented-btn"
                :class="{ active: riskScoreState.activity === level }"
                :aria-pressed="riskScoreState.activity === level"
                @click="setActivity(level)"
              >{{ level }}</button>
            </div>
            <div class="risk-row">
              <span class="risk-score">
                <span class="risk-score-value">{{ currentRiskScore.score ?? "-" }}</span>
                <span class="risk-score-unit">/100</span>
              </span>
              <span class="risk-tag">
                <span class="risk-tag-label">ระดับความเสี่ยงของคุณ</span>
                <span class="risk-chip" :class="statusClass(currentRiskScore.level)">
                  <span class="dot"></span>{{ currentRiskScore.level || "-" }}
                </span>
              </span>
            </div>
            <p class="risk-advice">{{ currentRiskAdvice }}</p>
            <p class="disclaimer-text">คำแนะนำนี้เป็นข้อมูลประกอบการวางแผนกิจกรรม ไม่ใช่คำวินิจฉัยหรือคำแนะนำทางการแพทย์ ผู้มีโรคประจำตัวควรปฏิบัติตามคำแนะนำของบุคลากรทางการแพทย์</p>
            <template v-if="bestActivityTimes.length">
              <div class="risk-sublabel">ช่วงเวลาที่เหมาะกับกิจกรรมนี้มากที่สุด</div>
              <ol class="best-list">
                <li v-for="(point, i) in bestActivityTimes" :key="point.display_time" class="best-row">
                  <span class="rank-badge" :class="`rank-${i + 1}`">{{ i + 1 }}</span>
                  <span class="best-time">{{ point.display_time }} น.</span>
                  <span class="best-pm">{{ point.riskScore }} <span class="best-unit">คะแนน</span></span>
                </li>
              </ol>
            </template>
          </section>

          <!-- สรุปวันนี้ -->
          <section v-if="SHOW_EXTRA_CARDS" class="card">
            <div class="summary-eyebrow">
              <Sparkles :size="18" :stroke-width="2" aria-hidden="true" />
              <span>สรุปวันนี้ · อัปเดต 06:00 น.</span>
            </div>
            <div class="summary-value">
              {{ dailyAvgDisplay }}<span class="summary-unit"> µg/m³ ค่าเฉลี่ยพยากรณ์ 24 ชม.</span>
            </div>
            <p class="summary-text">{{ dailyTextDisplay }}</p>
            <p class="disclaimer-text">คำแนะนำนี้เป็นข้อมูลประกอบการวางแผนกิจกรรม ไม่ใช่คำวินิจฉัยหรือคำแนะนำทางการแพทย์ ผู้มีโรคประจำตัวควรปฏิบัติตามคำแนะนำของบุคลากรทางการแพทย์</p>
          </section>
        </div>
      </div>
    </div>

    <ProvincePickerSheet
      :open="pickerOpen"
      :selected="provinceState.selected"
      @select="selectProvince"
      @close="pickerOpen = false"
    />
  </div>
</template>

<style scoped>
.forecast-page {
  /* โทเคนกลาง (ground/card/ink/muted/line/track/accent/warn/info) มาจาก :root ใน assets/main.css
     เหลือแค่โทเคนที่หน้านี้ใช้เฉพาะตัวเอง (สีระดับฝุ่น/โมเดล/การ์ดหลัก) */
  --chip-bg: rgba(255, 255, 255, 0.72);

  --excellent: #3b82f6;
  --good: #22c55e;
  --moderate: #eab308;
  --unhealthy: #f97316;
  --bad: #ef4444;

  /* สีใน "พยากรณ์อากาศรายชั่วโมง" — w-pm25 คือเส้นกราฟ, w-icon แค่สีไอคอนสภาพอากาศบนการ์ด */
  --w-icon: #3b82f6;
  --w-pm25: #f97316;

  --hero-bg: #faefc1;
  --hero-ink: #4f3b00;

  min-height: 100vh;
  background: var(--ground);
  color: var(--ink);
  padding: var(--app-top-offset) 16px calc(var(--app-bottom-offset) + 24px);
}

  :root[data-theme="dark"] .forecast-page {
    --chip-bg: rgba(255, 255, 255, 0.09);
  }



.forecast-page.aqi-excellent { --hero-bg: #e3edfd; --hero-ink: #1e3a8a; }
.forecast-page.aqi-good      { --hero-bg: #dff4e6; --hero-ink: #14532d; }
.forecast-page.aqi-moderate  { --hero-bg: #faefc1; --hero-ink: #4f3b00; }
.forecast-page.aqi-unhealthy { --hero-bg: #fce4ce; --hero-ink: #6e2a00; }
.forecast-page.aqi-bad       { --hero-bg: #fadada; --hero-ink: #7f1d1d; }

  :root[data-theme="dark"] .forecast-page.aqi-excellent { --hero-bg: #14233f; --hero-ink: #c3d8ff; }
  :root[data-theme="dark"] .forecast-page.aqi-good { --hero-bg: #12291b; --hero-ink: #bdebcb; }
  :root[data-theme="dark"] .forecast-page.aqi-moderate { --hero-bg: #2b2410; --hero-ink: #f5de8a; }
  :root[data-theme="dark"] .forecast-page.aqi-unhealthy { --hero-bg: #33200f; --hero-ink: #ffc999; }
  :root[data-theme="dark"] .forecast-page.aqi-bad { --hero-bg: #351516; --hero-ink: #ffb8b8; }


.wrap { max-width: 1080px; margin: 0 auto; }

.top-row {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 12px;
  margin-bottom: 16px;
}
.page-title { color: var(--muted); font-weight: 600; font-size: 0.95rem; }

@media (max-width: 720px) { .page-title { display: none; } }

.province-chip {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  min-height: 44px;
  padding: 0 14px 0 12px;
  border-radius: 22px;
  border: 1px solid var(--line);
  background: var(--card);
  color: var(--ink);
  font-family: inherit;
  font-size: 1rem;
  font-weight: 600;
  cursor: pointer;
}
.province-chip:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.zs-badge {
  font-size: 0.7rem;
  font-weight: 600;
  padding: 2px 8px;
  border-radius: 999px;
  background: var(--track);
  color: var(--muted);
}

.stack { display: flex; flex-direction: column; gap: 16px; }
.layout { display: flex; flex-direction: column; gap: 16px; }
.col-primary, .col-secondary { display: flex; flex-direction: column; gap: 16px; min-width: 0; }

@media (min-width: 960px) {
  .layout { display: grid; grid-template-columns: 420px minmax(0, 1fr); gap: 20px; align-items: start; }
}

/* banners */
.banner {
  display: flex;
  gap: 12px;
  padding: 14px 16px;
  border-radius: 16px;
  border: 1px solid;
  line-height: 1.55;
}
.banner > svg { flex-shrink: 0; margin-top: 1px; }
.banner-warn { background: var(--warn-bg); border-color: var(--warn-line); color: var(--warn-ink); }
.banner-info { background: var(--info-bg); border-color: var(--info-line); color: var(--info-ink); }
.banner-title { font-weight: 600; font-size: 0.95rem; line-height: 1.4; }
.banner-text { font-size: 0.9rem; }

/* hero */
.hero {
  background: var(--hero-bg);
  color: var(--hero-ink);
  border-radius: 32px;
  padding: 24px 22px 22px;
  display: flex;
  flex-direction: column;
  gap: 16px;
  transition: background 0.4s ease, color 0.4s ease;
}
.hero-head { display: flex; justify-content: space-between; align-items: center; font-size: 0.9rem; }
.hero-label { font-weight: 600; }
.hero-updated { opacity: 0.85; }
.estimate-tag {
  margin-left: 8px;
  padding: 2px 10px;
  border-radius: 999px;
  background: var(--chip-bg);
  font-size: 0.75rem;
  font-weight: 600;
}
.pm-value-row { display: flex; align-items: flex-end; gap: 8px; line-height: 1; }
.pm-value {
  font-family: var(--font-display);
  font-weight: 600;
  font-size: 6rem;
  letter-spacing: -0.02em;
}
.pm-unit { font-size: 1rem; padding-bottom: 14px; opacity: 0.85; }
.level-chip {
  display: inline-flex;
  align-items: center;
  align-self: flex-start;
  gap: 10px;
  padding: 8px 16px 8px 10px;
  border-radius: 24px;
  background: var(--chip-bg);
  color: var(--hero-ink);
}
.level-icon { flex-shrink: 0; }
.status-excellent .level-icon { color: var(--excellent); }
.status-good .level-icon { color: var(--good); }
.status-moderate .level-icon { color: var(--moderate); }
.status-unhealthy .level-icon { color: var(--unhealthy); }
.status-bad .level-icon { color: var(--bad); }
.level-text { font-size: 1.05rem; font-weight: 600; }

.scale-track { position: relative; display: flex; gap: 4px; padding: 4px 0; }
.scale-seg { flex: 1; height: 8px; border-radius: 4px; opacity: 0.35; }
.scale-seg-active { opacity: 1; }
.scale-marker {
  position: absolute;
  top: 0;
  width: 16px;
  height: 16px;
  margin-left: -8px;
  border-radius: 50%;
  background: var(--card);
  border: 3px solid var(--hero-ink);
  transition: left 0.5s ease;
}
.scale-ends { display: flex; justify-content: space-between; font-size: 0.75rem; opacity: 0.85; margin-top: 4px; }

.hero-advice { margin: 0; font-size: 0.95rem; line-height: 1.55; }
.hero-note { margin: -6px 0 0; font-size: 0.75rem; opacity: 0.75; }

.stat-tiles { display: flex; gap: 8px; padding-top: 4px; }
.stat-tile {
  flex: 1;
  display: flex;
  flex-direction: column;
  gap: 4px;
  padding: 12px;
  border-radius: 16px;
  background: var(--chip-bg);
}
.stat-tile-value { font-size: 1.1rem; font-weight: 600; }
.stat-tile-label { font-size: 0.75rem; opacity: 0.85; }

/* การ์ดทั่วไป */
.card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 24px;
  padding: 20px;
}
.card h2 { margin: 0; font-size: 1.05rem; font-weight: 600; }
.card-head { display: flex; justify-content: space-between; align-items: baseline; gap: 8px; padding-bottom: 8px; }
.card-head-sub { font-size: 0.8rem; color: var(--muted); }
.card-caption { margin: 4px 0 12px; font-size: 0.85rem; color: var(--muted); line-height: 1.5; }
.dot { display: inline-block; width: 10px; height: 10px; border-radius: 50%; margin-right: 6px; flex-shrink: 0; }

/* รายการอันดับ (ช่วงที่ควรออก / เหมาะกับกิจกรรม) */
.best-list { list-style: none; margin: 0; padding: 0; }
.best-row {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px 0;
  border-top: 1px solid var(--line);
}
.rank-badge {
  width: 28px;
  height: 28px;
  border-radius: 50%;
  display: inline-flex;
  align-items: center;
  justify-content: center;
  font-size: 0.85rem;
  font-weight: 700;
  flex-shrink: 0;
}
.rank-1 { background: #e9b949; color: #3b2a00; }
.rank-2 { background: #c9cdd3; color: #22262c; }
.rank-3 { background: #d9a377; color: #3b1f08; }
.best-time { flex: 1; font-size: 1.05rem; font-weight: 600; }
.best-pm { display: inline-flex; align-items: center; font-size: 0.95rem; }
.best-unit { color: var(--muted); font-size: 0.8rem; margin-left: 4px; }
.best-single {
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 12px 0 2px;
  margin-top: 4px;
  border-top: 1px solid var(--line);
}
.best-single-time { font-size: 1.3rem; font-weight: 700; }

/* กราฟพยากรณ์อากาศ (การ์ด + เส้น) */
.chart-head { padding-bottom: 12px; }
.chart-card { padding-left: 0; padding-right: 0; }
.chart-card .card-head, .chart-card .chart-foot, .chart-card .chart-caption, .chart-card .bar-detail { padding-left: 20px; padding-right: 20px; }

.weather-scroll {
  overflow-x: auto;
  -webkit-overflow-scrolling: touch;
  padding-bottom: 4px;
}
.weather-inner { position: relative; }

.hour-cards { display: flex; padding: 0 16px; }
.hour-card {
  flex: 0 0 64px;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  padding: 8px 0;
  border: 0;
  border-radius: 12px;
  background: transparent;
  color: var(--ink);
  font-family: inherit;
  cursor: pointer;
}
.hour-card.selected { background: var(--track); }
.hour-card-now { font-weight: 700; }
.hour-card-past { opacity: 0.55; }
.hour-card-past.selected { opacity: 0.85; }
.hour-card-gap { opacity: 0.78; }
.hour-card-gap.selected { opacity: 0.95; }
.hour-card:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
.hour-card-time { font-size: 0.72rem; color: var(--muted); }
.hour-card-icon { color: var(--w-icon); }
.hour-card-pm25 { font-size: 1rem; font-weight: 600; }

.weather-chart-svg { display: block; margin-top: 4px; }
.weather-chart-svg :deep(.axis-label) { font-size: 9px; fill: var(--muted); }
.weather-chart-svg :deep(.now-line) { stroke: var(--line); stroke-width: 1.5; stroke-dasharray: 3 3; }
.weather-chart-svg :deep(.thr-grid-line) { stroke: var(--w-pm25); stroke-width: 1; opacity: 0.25; stroke-dasharray: 2 3; }
.weather-chart-svg :deep(.past-band) { fill: var(--track); opacity: 0.55; }
.weather-chart-svg :deep(.past-label) { font-size: 8px; fill: var(--muted); }

.chart-foot {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
  padding-top: 12px;
  font-size: 0.75rem;
  color: var(--muted);
}
.weather-legend { display: flex; gap: 12px; }
.weather-legend span { display: inline-flex; align-items: center; }
.thr-swatch { width: 16px; margin-right: 6px; border-top: 1.5px dashed var(--w-pm25); }
.chart-caption { margin: 6px 0 0; font-size: 0.72rem; color: var(--muted); line-height: 1.5; }

.bar-detail {
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid var(--line);
  min-height: 92px;
}
.bar-detail-time { display: flex; align-items: center; gap: 8px; font-size: 0.8rem; font-weight: 600; color: var(--muted); }
.hist-tag {
  padding: 1px 8px;
  border-radius: 999px;
  background: var(--track);
  color: var(--muted);
  font-size: 0.68rem;
  font-weight: 600;
}
.bar-detail-main { display: flex; align-items: baseline; gap: 6px; margin: 2px 0 6px; }
.bar-detail-value { font-family: var(--font-display); font-weight: 600; font-size: 1.8rem; line-height: 1.1; }
.bar-detail-meta { display: flex; align-items: center; font-size: 0.85rem; color: var(--muted); line-height: 1.6; }
.bar-detail-empty { font-size: 0.9rem; color: var(--muted); padding-top: 8px; }

.trend-row { display: flex; align-items: center; gap: 6px; font-size: 0.85rem; margin: 2px 0 0; }
.trend-up { color: var(--bad); }
.trend-down { color: var(--good); }
.trend-flat { color: var(--muted); }

.shap-block { margin-top: 10px; padding-top: 10px; border-top: 1px dashed var(--line); }
.shap-loading { font-size: 0.82rem; color: var(--muted); }
.shap-label { font-size: 0.72rem; font-weight: 600; color: var(--muted); margin-bottom: 8px; }
.shap-up { color: var(--bad); flex-shrink: 0; }
.shap-down { color: var(--good); flex-shrink: 0; }

.shap-list { list-style: none; margin: 0; padding: 0; display: flex; flex-direction: column; gap: 7px; }
.shap-item { display: flex; align-items: center; gap: 6px; font-size: 0.85rem; }
.shap-item-name { flex: 1; min-width: 0; }
.shap-item-val { font-size: 0.78rem; font-variant-numeric: tabular-nums; flex-shrink: 0; }
.shap-toggle {
  margin-top: 8px;
  padding: 0;
  border: 0;
  background: transparent;
  color: var(--accent);
  font-family: inherit;
  font-size: 0.78rem;
  font-weight: 600;
  cursor: pointer;
}
.shap-reveal-btn { display: flex; align-items: center; gap: 4px; margin-top: 0; }
.shap-reveal-btn :deep(svg) { transition: transform 0.15s ease; }
.shap-reveal-btn :deep(.shap-reveal-icon-open) { transform: rotate(180deg); }
.shap-toggle:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

/* ความเสี่ยงส่วนบุคคล */
.segmented {
  display: flex;
  gap: 4px;
  padding: 4px;
  border-radius: 16px;
  background: var(--track);
  margin-bottom: 16px;
}
.segmented-btn {
  flex: 1;
  min-height: 44px;
  border: 0;
  border-radius: 12px;
  background: transparent;
  color: var(--muted);
  font-family: inherit;
  font-size: 0.95rem;
  font-weight: 600;
  cursor: pointer;
}
.segmented-btn.active { background: var(--ink); color: var(--card); }
.segmented-btn:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
.risk-row { display: flex; align-items: center; justify-content: space-between; gap: 12px; }
.risk-score { display: inline-flex; align-items: baseline; gap: 4px; }
.risk-score-value { font-family: var(--font-display); font-weight: 600; font-size: 2.75rem; line-height: 1; }
.risk-score-unit { color: var(--muted); font-size: 0.95rem; }
.risk-tag { display: flex; flex-direction: column; align-items: flex-end; gap: 4px; }
.risk-tag-label { font-size: 0.7rem; font-weight: 600; color: var(--muted); }
.risk-chip {
  display: inline-flex;
  align-items: center;
  padding: 6px 14px;
  border-radius: 20px;
  background: var(--track);
  font-size: 0.9rem;
  font-weight: 600;
  white-space: nowrap;
}
.risk-chip .dot { margin-right: 8px; }
.risk-chip.status-excellent .dot { background: var(--excellent); }
.risk-chip.status-good .dot { background: var(--good); }
.risk-chip.status-moderate .dot { background: var(--moderate); }
.risk-chip.status-unhealthy .dot { background: var(--unhealthy); }
.risk-chip.status-bad .dot { background: var(--bad); }
.risk-advice { margin: 12px 0 0; font-size: 0.92rem; line-height: 1.55; color: var(--muted); }
.disclaimer-text { margin: 8px 0 0; font-size: 0.78rem; line-height: 1.5; color: var(--muted); opacity: 0.85; }
.risk-sublabel { margin: 16px 0 4px; font-size: 0.8rem; font-weight: 600; color: var(--muted); }

/* สรุปวันนี้ */
.summary-eyebrow { display: flex; align-items: center; gap: 8px; color: var(--muted); font-size: 0.8rem; font-weight: 500; }
.summary-value { font-family: var(--font-display); font-weight: 600; font-size: 2rem; line-height: 1.2; margin: 8px 0; }
.summary-unit { font-family: var(--font-sans); font-weight: 400; font-size: 0.85rem; color: var(--muted); }
.summary-text { margin: 0; font-size: 0.95rem; line-height: 1.6; }

/* loading skeleton */
.skeleton {
  background: linear-gradient(90deg, var(--line) 25%, transparent 37%, var(--line) 63%);
  background-size: 400% 100%;
  animation: shimmer 1.4s ease infinite;
  border-radius: 8px;
}
@keyframes shimmer {
  0% { background-position: 100% 50%; }
  100% { background-position: 0 50%; }
}
.skeleton-hero { height: 340px; border-radius: 32px; }
.skeleton-line { width: 220px; height: 16px; margin: 6px 0; }
.skeleton-chart { width: 100%; height: 200px; }

/* error state */
.error-box { text-align: center; padding: 32px 20px; }
.error-box p { color: var(--muted); margin: 0 0 16px; }
.retry-btn {
  min-height: 44px;
  background: var(--accent);
  color: #fff;
  border: none;
  padding: 0 24px;
  border-radius: 22px;
  font-family: inherit;
  font-size: 0.95rem;
  font-weight: 600;
  cursor: pointer;
}
.retry-btn:hover { opacity: 0.9; }

@media (prefers-reduced-motion: reduce) {
  .hero, .scale-marker, .bar, .bar-range-cap { transition: none; }
  .skeleton { animation: none; }
}
</style>

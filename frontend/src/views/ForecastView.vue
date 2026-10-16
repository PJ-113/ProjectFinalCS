<script setup>
import { ref, computed, onMounted, onUnmounted, watch } from "vue";
import {
  Laugh, Smile, Meh, Frown, Angry, Thermometer, Droplet, Wind,
  MapPin, ChevronDown, TriangleAlert, Info, Sparkles,
} from "@lucide/vue";
import { fetchForecast, fetchCurrentPm25, buildWarningMessage, formatCardTime } from "../notifications";
import { riskScoreState, setActivity, ACTIVITY_LEVELS, computeRiskScore, riskScoreAdvice } from "../riskScore";
import { provinceState, selectProvince } from "../provinces";
import { statusClass, severityVar, PM25_LEVELS, scaleMarkerPct, adviceForStatus } from "../pm25Levels";
import ProvincePickerSheet from "../components/ProvincePickerSheet.vue";

// 3 การ์ดนี้กลับไปใช้ข้อมูลจริงจาก API แล้ว (ผู้ใช้สั่ง — เดิมเคยเปิดด้วยข้อมูลตัวอย่างคงที่ชั่วคราวเพื่อ
// ถ่ายภาพ/โชว์เฉยๆ ตั้งแต่ 2026-10-02) ต้องการกลับไปโชว์ sample อีก: ตั้ง SAMPLE_EXTRA_CARDS_DATA = true
// ต้องการซ่อนการ์ดทั้งหมด: ตั้ง SHOW_EXTRA_CARDS = false (โค้ด/logic ของจริงยังอยู่ครบไม่ถูกลบ)
const SHOW_EXTRA_CARDS = true;
const SAMPLE_EXTRA_CARDS_DATA = false;

const SAMPLE_BEST_TIMES = [
  { time: "05:00", pm25: 11 },
  { time: "04:00", pm25: 19 },
  { time: "18:00", pm25: 27 },
];
// ค่าตัวอย่างคำนวณจากสูตรจริง (compute_risk_score() ใน app.py) เอง ไม่ได้เดา — สมมติ PM2.5 ปัจจุบัน
// 28 µg/m³, activity_offset (เบา=0/ปานกลาง=-5/หนัก=-10) แล้ว interpolate ข้าม RISK_SCORE_BREAKPOINTS
// คะแนนยิ่งสูง = ยิ่งเสี่ยง (ไม่ใช่กลับกัน) จึงต้องเพิ่มขึ้นตามความหนักกิจกรรม ของเดิมคำนวณผิดทิศทาง
// (เคยลดลงจากเบา→หนัก) แก้แล้วตามนี้
const SAMPLE_RISK_TABLE = {
  เบา: {
    score: 45, level: "ปานกลาง", advice: "ความเสี่ยงปานกลาง ทำได้แต่ควรสังเกตอาการตนเอง",
    times: [{ time: "05:00", riskScore: 13 }, { time: "04:00", riskScore: 17 }, { time: "06:00", riskScore: 22 }],
  },
  ปานกลาง: {
    score: 53, level: "ปานกลาง", advice: "ความเสี่ยงปานกลาง ทำได้แต่ควรสังเกตอาการตนเอง",
    times: [{ time: "05:00", riskScore: 20 }, { time: "04:00", riskScore: 26 }, { time: "06:00", riskScore: 32 }],
  },
  หนัก: {
    score: 60, level: "เริ่มมีผลกระทบต่อสุขภาพ", advice: "ความเสี่ยงเริ่มสูง ควรลดความหนักหรือระยะเวลาของกิจกรรมนี้",
    times: [{ time: "05:00", riskScore: 30 }, { time: "04:00", riskScore: 36 }, { time: "06:00", riskScore: 42 }],
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
const highPollutionRisk = ref(false);
const riskGroup = ref("ทั่วไป");
const fallbackHorizons = ref([]);
// สถานะข้อมูลที่ใช้ทำพยากรณ์ (data_quality จาก /api/forecast): target_stale = reading ล่าสุดของจังหวัดนี้เก่ากว่า 6 ชม.
const dataQuality = ref({ target_stale: false, target_age_hours: 0 });
const provinceInfo = ref({ th: "", en: "", is_zero_shot: false });
const riskScoreConfig = ref(null);

const pickerOpen = ref(false);
// index ของแท่งที่แตะในกราฟ (null = ยังไม่เลือก) — แทน title= tooltip เดิมที่ใช้ไม่ได้บนหน้าจอทัช
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
// ช่วงความเชื่อมั่น (ขอบบน P95 conformal) วาดเป็นส่วนต่อขยายโปร่งแสงของแท่งเดียวกัน (สีเดียวกับ
// .bar เอง ที่ ~28% opacity) อยู่ชั้นล่างของแท่งทึบ ครอบคลุม 0→pm25_hi เต็ม แท่งทึบบังส่วนล่างไว้
// เหลือให้เห็นแค่ส่วนที่ "เกิน" จุดพยากรณ์ขึ้นไปถึงขอบบนจริงๆ
function barRangeColor(pm25) {
  return `color-mix(in srgb, var(${severityVar(pm25)}) 28%, transparent)`;
}

// ตำแหน่ง % จากขอบล่างของ bar-track — ใช้ maxPm เดียวกับความสูงแท่งหลัก ให้อยู่สเกลเดียวกันเป๊ะ
function pctOf(value) {
  return (value / maxPm.value) * 100;
}

const pageClass = computed(() => bodyClassFor(current.value.status));
// รวม pm25_hi (ขอบบนช่วงความเชื่อมั่น) เข้าไปตอนหาค่าสูงสุดด้วย ไม่งั้นส่วนต่อขยายจะโผล่พ้นกราฟตอน
// ช่วงกว้าง (horizon ไกลๆ) — 37.5 คือเพดานต่ำสุดเพื่อให้เส้นเกณฑ์อยู่ในกราฟเสมอ
const maxPm = computed(() =>
  Math.max(...forecast.value.flatMap((f) => [f.pm25, f.pm25_hi ?? f.pm25]), 37.5)
);
const dailyAvgDisplay = computed(() => {
  if (SAMPLE_EXTRA_CARDS_DATA) return SAMPLE_DAILY.avg;
  return dailySummary.value.daily_avg_pm25 !== null && dailySummary.value.daily_avg_pm25 !== undefined
    ? dailySummary.value.daily_avg_pm25
    : "-";
});
const dailyTextDisplay = computed(() =>
  SAMPLE_EXTRA_CARDS_DATA ? SAMPLE_DAILY.text : dailySummary.value.explanation || "กำลังเตรียมข้อมูลสำหรับวันนี้"
);
// horizon (1..24) -> เวลานาฬิกาของจุดนั้นใน forecast[] (ผู้ใช้ไม่ต้องแปล "t+N" เอง)
const fallbackTimesText = computed(() =>
  fallbackHorizons.value.map((h) => forecast.value[h - 1]?.time).filter(Boolean).join(", ")
);

// ข้อความเตือนช่วงมลพิษสูง (ปรับตาม risk_group) อยู่ใน notifications.js เพราะกระดิ่งแจ้งเตือนใน
// AppHeader.vue ต้องใช้ข้อความเดียวกันเป๊ะ ไม่ให้สองที่ไม่ตรงกัน
const warningMessage = computed(() => buildWarningMessage(riskGroup.value));

const markerPct = computed(() => scaleMarkerPct(current.value.pm25));
const currentAdvice = computed(() => adviceForStatus(current.value.status));
const updatedLabel = computed(() =>
  current.value.datetime_th && current.value.datetime_th !== "-" ? formatCardTime(current.value.datetime_th) : ""
);
const activeLevelIndex = computed(() => PM25_LEVELS.findIndex((l) => l.status === current.value.status));

// ช่วงเวลาที่อากาศดีที่สุด (ค่าเดียว ไม่ใช่ top-3 — ตัดสินใจ 2026-10-13 เพราะใกล้วันนำเสนอ ไม่อยากปรับ
// เยอะตอนนี้ จะพิจารณารวมกับการ์ด bestActivityTimes อีกทีใน phase ถัดไป ดู future-todo-list) คำนวณสด
// จาก forecast[] กรองเฉพาะ 06:00-20:00 (ช่วงที่ใช้งานจริงได้) ก่อน sort เหมือน best_time ฝั่ง backend —
// fallback ไป forecast ทั้งหมดถ้าว่าง
const topBestTime = computed(() => {
  if (SAMPLE_EXTRA_CARDS_DATA) return SAMPLE_BEST_TIMES[0] ?? null;
  const activeHours = forecast.value.filter((f) => f.hour_thai >= 6 && f.hour_thai <= 20);
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
  const activeHours = forecast.value.filter((f) => f.hour_thai >= 6 && f.hour_thai <= 20);
  const pool = activeHours.length > 0 ? activeHours : forecast.value;
  return [...pool]
    .map((f) => ({ ...f, riskScore: computeRiskScore(f.pm25, riskScoreConfig.value, riskScoreState.activity).score }))
    .sort((a, b) => a.riskScore - b.riskScore)
    .slice(0, 3);
});

const selectedPoint = computed(() => (selectedIdx.value == null ? null : forecast.value[selectedIdx.value] || null));

async function loadForecast() {
  state.value = "loading";
  selectedIdx.value = null;
  try {
    const data = await fetchForecast();
    current.value = data.current;
    currentWeather.value = data.current_weather || {};
    dailySummary.value = data.daily_summary || {};
    forecast.value = data.forecast;
    highPollutionRisk.value = data.high_pollution_risk;
    riskGroup.value = data.risk_group || "ทั่วไป";
    fallbackHorizons.value = data.fallback_triggered_horizons || [];
    dataQuality.value = data.data_quality || { target_stale: false, target_age_hours: 0 };
    provinceInfo.value = data.province || { th: "", en: "", is_zero_shot: false };
    riskScoreConfig.value = data.risk_score_config || null;
    state.value = "content";
  } catch (err) {
    errorMessage.value = err.dataUnavailable ? err.message : "ไม่สามารถโหลดข้อมูลได้ ลองใหม่อีกครั้ง";
    state.value = "error";
    console.error("โหลดข้อมูลพยากรณ์ล้มเหลว:", err);
  }
}

// สลับจังหวัดจาก ProvincePickerSheet -> provinceState.selected เปลี่ยน -> โหลดพยากรณ์ใหม่
watch(() => provinceState.selected, loadForecast);

onMounted(loadForecast);

// รีเฟรชแค่การ์ด "ปัจจุบัน" เป็นระยะ แยกจาก loadForecast() ทั้งก้อน เพราะ /api/forecast หนัก (inference
// 2 โมเดล + บางครั้ง Gemini) ใช้ /api/current_pm25 ที่เบากว่ามากแทน — อ่าน provinceState.selected สด
// ทุกครั้งที่ tick จึงตามจังหวัดที่เลือกอยู่เสมอโดยไม่ต้อง clear/ตั้ง interval ใหม่ตอนสลับจังหวัดเอง
const CURRENT_PM25_POLL_MS = 2 * 60 * 1000; // สั้นกว่า cache TTL ฝั่ง backend (5 นาที) พอเห็นค่าขยับ แต่ไม่ถี่เกิน
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

        <div v-if="dataQuality.series_source === 'stale_cache' || dataQuality.series_source === 'dustboy' || dataQuality.weather_source === 'climatology'" class="banner banner-info">
            <Info :size="22" :stroke-width="2" aria-hidden="true" />
            <div class="banner-text">
              <template v-if="dataQuality.series_source === 'dustboy'">
                สถานีตรวจวัดหลักของจังหวัดนี้ขัดข้อง ระบบใช้ข้อมูลจากเครือข่ายเซนเซอร์ทางเลือก (Dustboy) เป็นการชั่วคราว ความแม่นยำอาจต่ำกว่าปกติ
              </template>
              <template v-else-if="dataQuality.series_source === 'stale_cache'">
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
              <span v-if="updatedLabel" class="hero-updated">อัปเดต {{ updatedLabel }}</span>
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
              <span class="best-single-time">{{ topBestTime.time }} น.</span>
              <span class="best-pm">
                <span class="dot" :style="{ background: barColor(topBestTime.pm25) }"></span>{{ topBestTime.pm25 }}
                <span class="best-unit">µg/m³</span>
              </span>
            </div>
          </section>
        </div>

        <div class="col-secondary">
          <!-- กราฟรายชั่วโมง -->
          <section class="card chart-card">
            <div class="card-head chart-head">
              <div>
                <h2>พยากรณ์รายชั่วโมง</h2>
                <!-- <p class="chart-sub">ส่วนจางเหนือแท่งคือขอบบนของช่วงความเชื่อมั่น ~90%</p> -->
              </div>
            </div>
            <div class="bars" role="group" aria-label="พยากรณ์ PM2.5 รายชั่วโมง 24 ชั่วโมงข้างหน้า">
              <button
                v-for="(point, i) in forecast"
                :key="point.time"
                type="button"
                class="bar-col"
                :class="{ selected: selectedIdx === i }"
                :aria-pressed="selectedIdx === i"
                :aria-label="`${point.time} น. ${point.pm25} ไมโครกรัมต่อลูกบาศก์เมตร`"
                @click="selectedIdx = selectedIdx === i ? null : i"
              >
                <span class="bar-value">{{ Math.round(point.pm25) }}</span>
                <span class="bar-track">
                  <span class="thr-line" :style="{ bottom: pctOf(37.5) + '%' }"></span>
                  <span
                    v-if="point.pm25_hi != null"
                    class="bar-range-cap"
                    :style="{ height: pctOf(point.pm25_hi) + '%', background: barRangeColor(point.pm25) }"
                  ></span>
                  <span class="bar" :style="{ height: pctOf(point.pm25) + '%', background: barColor(point.pm25) }"></span>
                </span>
                <span class="bar-label">{{ point.time }}</span>
                <span v-if="point.model_used === 'naive_fallback'" class="fallback-mark" aria-hidden="true"></span>
              </button>
            </div>
            <div class="chart-foot">
              <span class="thr-legend"><span class="thr-swatch"></span>37.5 µg/m³ เกณฑ์เริ่มมีผลต่อสุขภาพ</span>
              <span class="scroll-hint">เลื่อนดู →</span>
            </div>
            <div class="bar-detail" aria-live="polite">
              <template v-if="selectedPoint">
                <div class="bar-detail-time">{{ selectedPoint.time }} น.</div>
                <div class="bar-detail-main">
                  <span class="bar-detail-value">{{ selectedPoint.pm25 }}</span>
                  <span class="best-unit">µg/m³</span>
                </div>
                <div class="bar-detail-meta">
                  คาดว่าค่าฝุ่น PM2.5 จะอยู่ระหว่าง {{ selectedPoint.pm25_lo }}–{{ selectedPoint.pm25_hi }} µg/m³
                </div>
              </template>
              <div v-else class="bar-detail-empty">แตะแท่งเพื่อดูรายละเอียดของชั่วโมงนั้น</div>
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
                <li v-for="(point, i) in bestActivityTimes" :key="point.time" class="best-row">
                  <span class="rank-badge" :class="`rank-${i + 1}`">{{ i + 1 }}</span>
                  <span class="best-time">{{ point.time }} น.</span>
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
  /* โทเคนกลางของหน้า (ground/card/ink/muted/line/track/accent/warn/info) มาจาก :root ใน
     assets/main.css ที่โหลดทั้งแอปอยู่แล้ว (เดิมประกาศซ้ำไว้ในนี้ด้วย — เอาออกเพื่อกันสองที่ไม่ตรงกัน)
     เหลือแค่โทเคนที่หน้านี้ใช้เฉพาะตัวเอง (สีระดับฝุ่น/โมเดล/การ์ดหลัก) */
  --chip-bg: rgba(255, 255, 255, 0.72);

  --excellent: #3b82f6;
  --good: #22c55e;
  --moderate: #eab308;
  --unhealthy: #f97316;
  --bad: #ef4444;

  --model-fallback: #ec4899;

  --hero-bg: #faefc1;
  --hero-ink: #4f3b00;

  min-height: 100vh;
  background: var(--ground);
  color: var(--ink);
  padding: var(--app-top-offset) 16px calc(var(--app-bottom-offset) + 24px);
}
/* ธีมมืด (สลับด้วย data-theme บน <html> — ดู theme.js): โทเคนกลาง (ground/card/ink/muted/line/track/
   accent/warn/info) สลับค่าอัตโนมัติจาก main.css แล้ว เหลือแค่โทเคนเฉพาะหน้านี้ */
  :root[data-theme="dark"] .forecast-page {
    --chip-bg: rgba(255, 255, 255, 0.09);

    --model-fallback: #f472b6;
  }


/* tint การ์ดหลักตามระดับ (ตัวหนังสือเข้มบน tint อ่อน / อ่อนบน tint เข้มใน dark) ผ่าน AA ทุกระดับ */
.forecast-page.aqi-excellent { --hero-bg: #e3edfd; --hero-ink: #1e3a8a; }
.forecast-page.aqi-good      { --hero-bg: #dff4e6; --hero-ink: #14532d; }
.forecast-page.aqi-moderate  { --hero-bg: #faefc1; --hero-ink: #4f3b00; }
.forecast-page.aqi-unhealthy { --hero-bg: #fce4ce; --hero-ink: #6e2a00; }
.forecast-page.aqi-bad       { --hero-bg: #fadada; --hero-ink: #7f1d1d; }
/* ธีมมืด (สลับด้วย data-theme บน <html> — ดู theme.js) */
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
/* มือถือ: header ด้านบนมี brand อยู่แล้ว ไม่ต้องซ้ำหัวข้อหน้าอีก */
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
/* เดสก์ท็อป: ซ้าย = ค่าปัจจุบัน+ช่วงที่ควรออก / ขวา = กราฟ+ความเสี่ยง+สรุป */
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

/* กราฟ */
.chart-head { padding-bottom: 12px; }
.chart-sub { margin: 2px 0 0; font-size: 0.8rem; color: var(--muted); line-height: 1.5; }
.chart-card { padding-left: 0; padding-right: 0; }
.chart-card .card-head, .chart-card .chart-foot, .chart-card .bar-detail { padding-left: 20px; padding-right: 20px; }
.bars {
  display: flex;
  align-items: flex-end;
  overflow-x: auto;
  scroll-snap-type: x proximity;
  padding: 0 12px 4px;
  -webkit-overflow-scrolling: touch;
}
.bar-col {
  flex: 0 0 38px;
  scroll-snap-align: start;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 6px;
  padding: 4px 0;
  border: 0;
  border-radius: 10px;
  background: transparent;
  color: var(--ink);
  font-family: inherit;
  cursor: pointer;
  position: relative;
}
.bar-col.selected { background: var(--track); }
.bar-col:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
.bar-value { font-size: 0.75rem; font-weight: 600; }
.bar-track {
  position: relative;
  width: 100%;
  height: 150px;
  display: flex;
  align-items: flex-end;
  justify-content: center;
}
.bar, .bar-range-cap {
  position: absolute;
  left: 50%;
  width: 22px;
  margin-left: -11px;
  bottom: 0;
  border-radius: 8px 8px 4px 4px;
  transition: height 0.6s cubic-bezier(0.4, 0, 0.2, 1);
}
.bar-range-cap { pointer-events: none; }
.thr-line {
  position: absolute;
  left: 0;
  right: 0;
  border-top: 1.5px dashed var(--muted);
  opacity: 0.6;
  pointer-events: none;
}
.bar-label { font-size: 0.7rem; color: var(--muted); }
.fallback-mark {
  position: absolute;
  top: 2px;
  right: 6px;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--model-fallback);
}

.chart-foot {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
  padding-top: 12px;
  font-size: 0.75rem;
  color: var(--muted);
}
.thr-legend { display: inline-flex; align-items: center; gap: 6px; }
.thr-swatch { width: 18px; border-top: 1.5px dashed var(--muted); }

.bar-detail {
  margin-top: 12px;
  padding-top: 12px;
  border-top: 1px solid var(--line);
  min-height: 92px;
}
.bar-detail-time { font-size: 0.8rem; font-weight: 600; color: var(--muted); }
.bar-detail-main { display: flex; align-items: baseline; gap: 6px; margin: 2px 0 6px; }
.bar-detail-value { font-family: var(--font-display); font-weight: 600; font-size: 1.8rem; line-height: 1.1; }
.bar-detail-meta { display: flex; align-items: center; font-size: 0.85rem; color: var(--muted); line-height: 1.6; }
.bar-detail-empty { font-size: 0.9rem; color: var(--muted); padding-top: 8px; }

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

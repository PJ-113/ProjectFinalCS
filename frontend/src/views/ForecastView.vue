<script setup>
import { ref, computed, onMounted, watch } from "vue";
import { Laugh, Smile, Meh, Frown, Angry, Thermometer, Droplet, Wind } from "@lucide/vue";
import { fetchForecast, buildWarningMessage } from "../notifications";
import { riskScoreState, setActivity, ACTIVITY_LEVELS, computeRiskScore, riskScoreAdvice } from "../riskScore";
import { provinceState } from "../provinces";

// ชั่วคราว: ปิดการ์ดสรุปวันนี้ (NLG)/ช่วงเวลาดีที่สุด/คะแนนความเสี่ยงส่วนบุคคลไว้ก่อน ตอนตั้งค่า
// รันบนเครื่องใหม่ เหลือแค่ "การพยากรณ์ + ค่าฝุ่นปัจจุบัน" ตามที่ขอ — เปลี่ยนกลับเป็น true ทีหลังได้
const SHOW_EXTRA_CARDS = false;

const state = ref("loading"); // "loading" | "error" | "content"
const errorMessage = ref("ไม่สามารถโหลดข้อมูลได้ ลองใหม่อีกครั้ง");

const current = ref({ pm25: "-", datetime_th: "-", status: "" });
const currentWeather = ref({ temp_c: null, humidity_pct: null, wind_speed_kmh: null });
const bestTime = ref({ time: "-", pm25: "-" });
const dailySummary = ref({ daily_avg_pm25: null, explanation: "" });
const forecast = ref([]);
const highPollutionRisk = ref(false);
const riskGroup = ref("ทั่วไป");
const fallbackHorizons = ref([]);
const provinceInfo = ref({ th: "", en: "", is_zero_shot: false });
const riskScoreConfig = ref(null);

// ระดับตรงกับ PM25_LEVELS ใน app.py (มาตรฐาน คพ. 2566): ดีมาก/ดี/ปานกลาง/
// เริ่มมีผลกระทบต่อสุขภาพ/มีผลกระทบต่อสุขภาพ — ต้อง match string ให้ตรงเป๊ะกับที่ pm25_status()
// คืนกลับมา ไม่งั้นจะ fall through ไปโชว์สี/icon ผิดระดับ (เคยเป็นบั๊กมาก่อน ตอนยังมีแค่ 3 ระดับ)
function statusClass(status) {
  if (status === "ดีมาก") return "status-excellent";
  if (status === "ดี") return "status-good";
  if (status === "ปานกลาง") return "status-moderate";
  if (status === "เริ่มมีผลกระทบต่อสุขภาพ") return "status-unhealthy";
  return "status-bad";
}

function bodyClassFor(status) {
  if (status === "ดีมาก") return "aqi-excellent";
  if (status === "ดี") return "aqi-good";
  if (status === "ปานกลาง") return "aqi-moderate";
  if (status === "เริ่มมีผลกระทบต่อสุขภาพ") return "aqi-unhealthy";
  return "aqi-bad";
}

// ไอคอนหน้าคน (ชุดเดียวกับ Lucide ทั้งการ์ด แทน emoji เดิม) ไล่ระดับความรุนแรงจากยิ้มกว้าง
// ไปหน้าบึ้ง ให้เข้าชุดกับ thermometer/droplet/wind ด้านล่างการ์ด
function moodIconFor(status) {
  if (status === "ดีมาก") return Laugh;
  if (status === "ดี") return Smile;
  if (status === "ปานกลาง") return Meh;
  if (status === "เริ่มมีผลกระทบต่อสุขภาพ") return Frown;
  return Angry;
}

// threshold เดียวกับ PM25_LEVELS ใน app.py (15/25/37.5/75) — ไล่สี (เข้ม->อ่อน) ของสีระดับเดียวกัน
// เอง ต่อแท่ง ไม่ใช่ไล่สีฟ้าเดียวทุกแท่งแบบ mockup เพราะสีระดับ PM2.5 เป็นข้อมูลจริงที่ต้องคงไว้
function severityVar(pm25) {
  return pm25 <= 15 ? "--excellent"
    : pm25 <= 25 ? "--good"
    : pm25 <= 37.5 ? "--moderate"
    : pm25 <= 75 ? "--unhealthy"
    : "--bad";
}
function barColor(pm25) {
  const v = severityVar(pm25);
  return `linear-gradient(180deg, var(${v}) 0%, color-mix(in srgb, var(${v}) 55%, white) 100%)`;
}
// ช่วงความเชื่อมั่น (ขอบบน P95 conformal) วาดเป็นส่วนต่อขยายโปร่งแสงของแท่งเดียวกัน (สีเดียวกับ
// .bar เอง ที่ ~28% opacity) แทน tick-cap/dot-line เดิมที่ลอยแยกจากแท่งเหมือนหนวดแมลง — ส่วนขยายนี้
// อยู่ชั้นล่าง (z-index ต่ำกว่า .bar) ครอบคลุม 0→pm25_hi เต็ม แต่ถูกแท่งทึบด้านบนบังส่วนล่างไว้หมด
// เหลือให้เห็นแค่ส่วนที่ "เกิน" จุดพยากรณ์ขึ้นไปถึงขอบบนจริงๆ — ไม่ต้องคำนวณตำแหน่งซ้อนเอง
function barRangeColor(pm25) {
  const v = severityVar(pm25);
  return `color-mix(in srgb, var(${v}) 28%, transparent)`;
}

// ตำแหน่ง % จากขอบล่างของ bar-track สำหรับช่วงความเชื่อมั่น (P05-P95 หลัง conformal) — ใช้ maxPm
// เดียวกับที่คำนวณความสูงแท่งหลัก ให้อยู่สเกลเดียวกันเป๊ะ
function pctOf(value) {
  return (value / maxPm.value) * 100;
}

function modelColor(modelUsed) {
  if (modelUsed === "xgboost_pooled") return "var(--model-xgboost)";
  if (modelUsed === "naive_fallback") return "var(--model-fallback)";
  return "var(--model-lightgbm)";
}

const pageClass = computed(() => bodyClassFor(current.value.status));
// รวม pm25_hi (ขอบบนช่วงความเชื่อมั่น) เข้าไปตอนหาค่าสูงสุดด้วย ไม่งั้นเส้น error bar จะโผล่พ้น
// กราฟไปตอนช่วงกว้าง (horizon ไกลๆ) — ไม่ใช่แค่ point.pm25 เหมือนเดิม
const maxPm = computed(() =>
  Math.max(...forecast.value.flatMap((f) => [f.pm25, f.pm25_hi ?? f.pm25]), 37.5)
);
const dailyAvgDisplay = computed(() =>
  dailySummary.value.daily_avg_pm25 !== null && dailySummary.value.daily_avg_pm25 !== undefined
    ? dailySummary.value.daily_avg_pm25
    : "-"
);
const dailyTextDisplay = computed(() => dailySummary.value.explanation || "กำลังเตรียมข้อมูลสำหรับวันนี้");
const fallbackHorizonsText = computed(() => fallbackHorizons.value.map((h) => `t+${h}`).join(", "));

// ข้อความเตือนช่วงมลพิษสูง (ปรับตาม risk_group) ย้ายไป notifications.js แล้ว เพราะกระดิ่งแจ้งเตือน
// ใน AppHeader.vue ต้องใช้ข้อความเดียวกันเป๊ะ ไม่ให้สองที่ไม่ตรงกัน
const warningMessage = computed(() => buildWarningMessage(riskGroup.value));

// 3 ช่วงเวลาที่อากาศดีที่สุด เรียงจากดีสุดไปน้อยสุด — คำนวณสดจาก forecast[] ที่มีอยู่แล้ว
// กรองเฉพาะ 06:00-20:00 (ช่วงที่ใช้งานจริงได้) ก่อน sort เหมือน best_time ฝั่ง backend
// (app.py get_forecast()) กัน suggest ช่วงดึกที่ไม่มีประโยชน์ในทางปฏิบัติ — fallback ไป forecast
// ทั้งหมดถ้า active hours ว่างเปล่าจริงๆ
const topBestTimes = computed(() => {
  const activeHours = forecast.value.filter((f) => f.hour_thai >= 6 && f.hour_thai <= 20);
  const pool = activeHours.length > 0 ? activeHours : forecast.value;
  return [...pool].sort((a, b) => a.pm25 - b.pm25).slice(0, 3);
});

// Personalized Risk Score (proposal) — ผสาน pm25 ปัจจุบัน + risk_group (จาก session) + activity
// (เลือกสดหน้าเว็บ, ดู riskScoreState ใน riskScore.js) เป็นคะแนนเดียว 0-100 สูตร/ตัวเลข offset
// ทั้งหมดมาจาก riskScoreConfig ที่ backend ส่งมา (app.py compute_risk_score()) ที่นี่แค่เรียก mirror
// function เดียวกัน ไม่มี business constant ซ้ำ
const currentRiskScore = computed(() =>
  computeRiskScore(current.value.pm25, riskScoreConfig.value, riskScoreState.activity)
);
const currentRiskAdvice = computed(() => riskScoreAdvice(currentRiskScore.value.level));

// เหมือน topBestTimes แต่ personalize ตามกิจกรรมที่เลือก (sort ด้วยคะแนนความเสี่ยงแทน pm25 ดิบ) —
// กรอง 06:00-20:00 เดียวกันเสมอ
const bestActivityTimes = computed(() => {
  if (!riskScoreConfig.value) return [];
  const activeHours = forecast.value.filter((f) => f.hour_thai >= 6 && f.hour_thai <= 20);
  const pool = activeHours.length > 0 ? activeHours : forecast.value;
  return [...pool]
    .map((f) => ({ ...f, riskScore: computeRiskScore(f.pm25, riskScoreConfig.value, riskScoreState.activity).score }))
    .sort((a, b) => a.riskScore - b.riskScore)
    .slice(0, 3);
});

async function loadForecast() {
  state.value = "loading";
  try {
    const data = await fetchForecast();
    current.value = data.current;
    currentWeather.value = data.current_weather || {};
    bestTime.value = data.best_time;
    dailySummary.value = data.daily_summary || {};
    forecast.value = data.forecast;
    highPollutionRisk.value = data.high_pollution_risk;
    riskGroup.value = data.risk_group || "ทั่วไป";
    fallbackHorizons.value = data.fallback_triggered_horizons || [];
    provinceInfo.value = data.province || { th: "", en: "", is_zero_shot: false };
    riskScoreConfig.value = data.risk_score_config || null;
    state.value = "content";
  } catch (err) {
    errorMessage.value = "ไม่สามารถโหลดข้อมูลได้ ลองใหม่อีกครั้ง";
    state.value = "error";
    console.error("โหลดข้อมูลพยากรณ์ล้มเหลว:", err);
  }
}

// สลับจังหวัดจาก dropdown ใน AppHeader.vue -> provinceState.selected เปลี่ยน -> โหลดพยากรณ์ใหม่
watch(() => provinceState.selected, loadForecast);

onMounted(loadForecast);
defineExpose({ loadForecast });
</script>

<template>
  <div class="forecast-page" :class="pageClass">
    <div class="haze-decor"><span></span><span></span><span></span><span></span></div>

    <div class="wrap">
      <h1><span class="title-main">พยากรณ์ PM2.5</span><span class="title-sub"> - จังหวัด{{ provinceInfo.th || "..." }}{{ provinceInfo.is_zero_shot ? " (zero-shot)" : "" }}</span></h1>
      <!--<div class="subtitle">
        <span class="live-dot"></span>
        ข้อมูลจาก OpenAQ + Open-Meteo, พยากรณ์ล่วงหน้า 24 ชั่วโมงด้วยระบบ Dynamic Routing
        (เลือกจาก LightGBM / XGBoost pooled model อัตโนมัติตาม horizon) ·
        <router-link to="/compare">ดูความแม่นยำจริง →</router-link>
      </div> -->

      <!-- Loading skeleton -->
      <div v-if="state === 'loading'">
        <div class="card">
          <div class="current-row">
            <div class="skeleton skeleton-value"></div>
            <div class="skeleton skeleton-badge"></div>
          </div>
          <div class="skeleton skeleton-line"></div>
        </div>
        <div class="card">
          <div class="skeleton skeleton-line" style="width: 220px"></div>
        </div>
        <div class="card">
          <div class="skeleton skeleton-chart"></div>
        </div>
      </div>

      <!-- Error state -->
      <div v-else-if="state === 'error'" class="card error-box">
        <p>{{ errorMessage }}</p>
        <button class="retry-btn" @click="loadForecast">ลองอีกครั้ง</button>
      </div>

      <!-- Content -->
      <div v-else>
        <div v-if="highPollutionRisk" class="warning-box">
          ⚠️ {{ warningMessage }}
        </div>

        <div v-if="fallbackHorizons.length > 0" class="info-box">
          ℹ️ ข้อมูล input ไม่สมบูรณ์ในบางช่วงเวลา ระบบ Circuit Breaker จึงใช้ค่าพยากรณ์แบบคงที่ (ความเชื่อมั่นต่ำ) ชั่วคราวสำหรับ {{ fallbackHorizonsText }}
        </div>

        <div class="dashboard-grid">
          <div class="grid-col-main card">
            <div class="current-info">
              <div class="current-row">
                <div class="mood-badge" :class="statusClass(current.status)">
                  <component :is="moodIconFor(current.status)" :size="30" :stroke-width="2" />
                </div>
                <div class="pm-main">
                  <div class="label-muted">PM2.5 ล่าสุดรายชั่วโมง (ไม่ใช่ค่าเฉลี่ย 24 ชม. แบบ AQI ทางการ)</div>
                  <div class="pm-value-row">
                    <span class="pm-value">{{ current.pm25 }}</span>
                    <span class="pm-unit">µg/m³</span>
                  </div>
                </div>
                <span class="status-chip" :class="statusClass(current.status)">{{ current.status }}</span>
              </div>
              <div class="updated-at">อัปเดตล่าสุด: {{ current.datetime_th }}</div>
            </div>

            <div v-if="currentWeather.temp_c != null" class="stat-tiles">
              <div class="stat-tile">
                <Thermometer class="stat-tile-icon icon-temp" :size="20" :stroke-width="2" />
                <div class="stat-tile-value">{{ currentWeather.temp_c }}°C</div>
                <div class="stat-tile-label eyebrow">อุณหภูมิ</div>
              </div>
              <div class="stat-tile">
                <Droplet class="stat-tile-icon icon-humidity" :size="20" :stroke-width="2" />
                <div class="stat-tile-value">{{ currentWeather.humidity_pct }}%</div>
                <div class="stat-tile-label eyebrow">ความชื้น</div>
              </div>
              <div class="stat-tile">
                <Wind class="stat-tile-icon icon-wind" :size="20" :stroke-width="2" />
                <div class="stat-tile-value">{{ currentWeather.wind_speed_kmh }} km/h</div>
                <div class="stat-tile-label eyebrow">ความเร็วลม</div>
              </div>
            </div>
          </div>

          <div v-if="SHOW_EXTRA_CARDS" class="grid-col-side">
            <div class="card">
              <div class="label-muted">📋 สรุปวันนี้ (ค่าเฉลี่ยพยากรณ์ 24 ชม., อัปเดตตอน 06:00 น.)</div>
              <div class="daily-summary-value">
                {{ dailyAvgDisplay }}<span class="daily-summary-unit"> µg/m³</span>
              </div>
              <div class="daily-summary-text">{{ dailyTextDisplay }}</div>
            </div>

            <div class="card">
              <div class="label-muted">🌤️ ช่วงเวลาที่อากาศดีที่สุดในอีก 24 ชม.</div>
              <div class="best-times-list">
                <div class="best-time-row" v-for="(point, i) in topBestTimes" :key="point.time">
                  <span class="rank-badge" :class="`rank-${i + 1}`">{{ i + 1 }}</span>
                  <span class="best-time-time">{{ point.time }}</span>
                  <span class="best-time-pm">{{ point.pm25 }} µg/m³</span>
                </div>
              </div>
            </div>

            <div class="card">
              <div class="label-muted">🎯 คะแนนความเสี่ยงส่วนบุคคล</div>
              <div class="label-muted activity-select-caption">เลือกความหนักของกิจกรรมกลางแจ้งที่วางแผนจะทำ</div>
              <div class="activity-selector">
                <button
                  v-for="level in ACTIVITY_LEVELS"
                  :key="level"
                  type="button"
                  class="activity-btn"
                  :class="{ active: riskScoreState.activity === level }"
                  @click="setActivity(level)"
                >{{ level }}</button>
              </div>
              <div class="label-muted risk-score-caption">คะแนนความเสี่ยงสำหรับกิจกรรมนี้ (0 = เสี่ยงน้อยที่สุด, 100 = เสี่ยงมากที่สุด)</div>
              <div class="risk-score-row">
                <span class="risk-score-value-wrap">
                  <span class="risk-score-value" :class="statusClass(currentRiskScore.level)">{{ currentRiskScore.score }}</span>
                  <span class="risk-score-unit">/100</span>
                </span>
                <span class="risk-level-tag">
                  <span class="risk-level-label">ความเสี่ยงของคุณ</span>
                  <span class="status-chip" :class="statusClass(currentRiskScore.level)">{{ currentRiskScore.level }}</span>
                </span>
              </div>
              <div class="daily-summary-text risk-score-advice">{{ currentRiskAdvice }}</div>
              <div class="label-muted risk-score-sublabel">ช่วงเวลาที่เหมาะกับกิจกรรมนี้มากที่สุด</div>
              <div class="best-times-list">
                <div class="best-time-row" v-for="(point, i) in bestActivityTimes" :key="point.time">
                  <span class="rank-badge" :class="`rank-${i + 1}`">{{ i + 1 }}</span>
                  <span class="best-time-time">{{ point.time }}</span>
                  <span class="best-time-pm">{{ point.riskScore }} คะแนน</span>
                </div>
              </div>
            </div>
          </div>
        </div>

        <div class="card">
          <div class="chart-header">
            <div class="chart-title"> พยากรณ์ PM2.5 รายชั่วโมง (24 ชม. ข้างหน้า) <span class="chart-subtitle">ส่วนจางด้านบนคือขอบบนของช่วงความเชื่อมั่น ~90%</span></div>
            <div class="chart-legend">
              <span><span class="dot" style="background: var(--model-lightgbm)"></span>LightGBM</span>
              <span><span class="dot" style="background: var(--model-xgboost)"></span>XGBoost</span>
              <span><span class="dot" style="background: var(--model-fallback)"></span>Fallback (ความเชื่อมั่นต่ำ)</span>
            </div>
          </div>
          <div class="bars">
            <div class="bar-col" v-for="point in forecast" :key="point.time">
              <div class="bar-value">{{ point.pm25 }}</div>
              <div class="bar-track">
                <!-- ช่วงความเชื่อมั่น ~90% (P05-P95 หลัง conformal calibration) วาดเป็นส่วนขยายโปร่งแสง
                     ของแท่งเดียวกัน สูงถึง pm25_hi เต็ม แต่แท่งทึบด้านหน้า (.bar) บังส่วนล่างไว้หมด
                     เหลือให้เห็นแค่ส่วนเกินจากจุดพยากรณ์ขึ้นไปถึงขอบบนจริง (ไม่แสดง pm25_lo แยก
                     เพราะอยู่ใต้แท่งทึบเสมอ ไม่มีอะไรให้เห็นเพิ่ม — ค่าเต็มยังอยู่ใน tooltip) ซ่อนไปเอง
                     ถ้า pm25_hi ไม่มีมาใน response (backward compatible) -->
                <div
                  v-if="point.pm25_hi != null"
                  class="bar-range-cap"
                  :style="{ height: pctOf(point.pm25_hi) + '%', background: barRangeColor(point.pm25) }"
                ></div>
                <div
                  class="bar"
                  :style="{ height: pctOf(point.pm25) + '%', background: barColor(point.pm25) }"
                  :title="`${point.time}: ${point.pm25} µg/m³ (ช่วง ${point.pm25_lo}-${point.pm25_hi}) (${point.model_used})`"
                ></div>
              </div>
              <div class="model-dot" :style="{ background: modelColor(point.model_used) }" :title="point.model_used"></div>
              <div class="bar-label">{{ point.time }}</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.forecast-page {
  --sky-top: #6b7f96;
  --sky-mid: #9db4c9;
  --sky-bottom: #c9d9e6;
  --page-text: #0f172a;
  --page-text-muted: rgba(15, 23, 42, 0.7);

  --card-bg: rgba(255, 255, 255, 0.65);
  --card-border: rgba(255, 255, 255, 0.4);
  --text: #0f172a;
  --muted: #64748b;
  --border: rgba(255, 255, 255, 0.3);

  --excellent: #3b82f6;
  --good: #22c55e;
  --moderate: #eab308;
  --unhealthy: #f97316;
  --bad: #ef4444;
  --accent: #3b82f6;

  --model-lightgbm: #6366f1;
  --model-xgboost: #10b981;
  --model-fallback: #ec4899;

  min-height: 100vh;
  background: linear-gradient(180deg, var(--sky-top) 0%, var(--sky-mid) 55%, var(--sky-bottom) 100%);
  transition: background 1.2s ease;
  color: var(--text);
  padding: 100px 16px 48px;
  position: relative;
  overflow-x: hidden;
}
@media (prefers-color-scheme: dark) {
  .forecast-page {
    --card-bg: rgba(22, 25, 30, 0.55);
    --card-border: rgba(255, 255, 255, 0.1);
    --text: #f8fafc;
    --muted: #94a3b8;
    --border: rgba(255, 255, 255, 0.1);

    --model-lightgbm: #818cf8;
    --model-xgboost: #34d399;
    --model-fallback: #f472b6;
  }
}

.forecast-page.aqi-excellent {
  --sky-top: #1d68b3; --sky-mid: #5fb3ec; --sky-bottom: #eaf7ff;
  --page-text: #0a2c47; --page-text-muted: rgba(10, 44, 71, 0.7);
}
.forecast-page.aqi-good {
  --sky-top: #2f7fd1; --sky-mid: #79c2f0; --sky-bottom: #e3f4fd;
  --page-text: #0d2b45; --page-text-muted: rgba(13, 43, 69, 0.7);
}
.forecast-page.aqi-moderate {
  --sky-top: #d9b52b; --sky-mid: #eed685; --sky-bottom: #faf3d5;
  --page-text: #453a0a; --page-text-muted: rgba(69, 58, 10, 0.7);
}
.forecast-page.aqi-unhealthy {
  --sky-top: #d98d2b; --sky-mid: #eab766; --sky-bottom: #f7e6c4;
  --page-text: #4a2f0a; --page-text-muted: rgba(74, 47, 10, 0.7);
}
.forecast-page.aqi-bad {
  --sky-top: #4a3220; --sky-mid: #7a5238; --sky-bottom: #241812;
  --page-text: #f5e9de; --page-text-muted: rgba(245, 233, 222, 0.75);
}

/* ก้อนหมอก/ไอฝุ่นลอยเบาๆ เป็นฉากหลัง */
.haze-decor {
  position: fixed;
  inset: 0;
  pointer-events: none;
  overflow: hidden;
  z-index: 0;
}
.haze-decor span {
  position: absolute;
  border-radius: 50%;
  filter: blur(30px);
  background: rgba(255, 255, 255, 0.35);
  animation: float 20s ease-in-out infinite;
}
.haze-decor span:nth-child(1) { width: 220px; height: 220px; top: 8%;  left: -60px; animation-duration: 22s; }
.haze-decor span:nth-child(2) { width: 160px; height: 160px; top: 30%; right: -40px; animation-duration: 26s; animation-delay: -4s; }
.haze-decor span:nth-child(3) { width: 260px; height: 260px; top: 65%; left: 10%; animation-duration: 30s; animation-delay: -10s; }
.haze-decor span:nth-child(4) { width: 140px; height: 140px; top: 80%; right: 15%; animation-duration: 18s; animation-delay: -6s; }
@keyframes float {
  0%, 100% { transform: translate(0, 0); }
  50% { transform: translate(24px, -18px); }
}

.wrap { max-width: 1180px; margin: 0 auto; position: relative; z-index: 1; }

h1 {
  font-size: 1.9rem;
  font-weight: 800;
  margin: 0 0 4px;
}
.title-main { color: var(--page-text); }
.title-sub { color: var(--page-text-muted); font-weight: 600; font-size: 1.1rem; }
@media (max-width: 600px) {
  h1 { font-size: 1.4rem; }
  .title-sub { font-size: 0.85rem; display: block; }
}

.subtitle {
  color: var(--page-text-muted);
  font-size: 0.85rem;
  margin-bottom: 20px;
  display: flex;
  align-items: center;
  flex-wrap: wrap;
  gap: 6px;
}
.subtitle :deep(a) { color: var(--accent); text-decoration: none; font-weight: 600; }
.subtitle :deep(a):hover { text-decoration: underline; }
.live-dot {
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: var(--good);
  flex-shrink: 0;
  animation: pulse-dot 2s ease-in-out infinite;
}
@keyframes pulse-dot {
  0%, 100% { opacity: 1; box-shadow: 0 0 0 0 rgba(22, 163, 74, 0.4); }
  50% { opacity: 0.7; box-shadow: 0 0 0 5px rgba(22, 163, 74, 0); }
}

/* Hero (การ์ด PM2.5 ปัจจุบัน เต็มความกว้าง) ต่อด้วยแถวการ์ดรอง 3 ใบเท่ากัน — แทนเลย์เอาต์ 2 คอลัมน์เดิม
   (หลักซ้าย 1.6fr / รองขวา 1fr ซ้อนกัน 3 การ์ด) ที่ทำให้คอลัมน์รองสูงเกินคอลัมน์หลักไปมากหลังเพิ่ม
   การ์ด Personalized Risk Score เป็นใบที่ 3 (CHANGELOG 66) — เลย์เอาต์ใหม่นี้ทุกคอลัมน์สูงตามเนื้อหา
   ตัวเองเสมอ ไม่มีทางเกิดปัญหาสูงไม่เท่ากันแบบเดิมได้อีก */
.dashboard-grid {
  display: flex;
  flex-direction: column;
  gap: 20px;
  margin-bottom: 16px;
}
.dashboard-grid .card { margin-bottom: 0; }
.grid-col-main {
  display: flex;
  align-items: center;
  gap: 32px;
  flex-wrap: wrap;
}
.grid-col-side {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 20px;
  align-items: stretch;
}
.grid-col-side > .card {
  height: 100%;
}
.grid-col-side > .card:nth-child(1) {
  grid-column: 1;
  grid-row: 1;
}
.grid-col-side > .card:nth-child(2) {
  grid-column: 1;
  grid-row: 2;
}
.grid-col-side > .card:nth-child(3) {
  grid-column: 2;
  grid-row: 1 / span 2;
}
@media (max-width: 880px) {
  .grid-col-side { grid-template-columns: 1fr; }
  .grid-col-side > .card:nth-child(1),
  .grid-col-side > .card:nth-child(2),
  .grid-col-side > .card:nth-child(3) {
    grid-column: auto;
    grid-row: auto;
  }
  .grid-col-main { flex-direction: column; align-items: stretch; }
}

.card {
  background: var(--card-bg);
  border: 1px solid var(--card-border);
  backdrop-filter: blur(24px);
  -webkit-backdrop-filter: blur(24px);
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.08);
  border-radius: 24px;
  padding: 24px;
  margin-bottom: 20px;
  transition: box-shadow 0.3s ease, transform 0.3s ease;
}
.card:hover {
  box-shadow: 0 16px 50px rgba(0, 0, 0, 0.12), 0 0 0 1px color-mix(in srgb, var(--accent) 35%, transparent);
  border-color: color-mix(in srgb, var(--accent) 45%, var(--card-border));
  transform: translateY(-2px);
}


.current-info {
  flex: 1;
  display: flex;
  flex-direction: column;
  justify-content: center;
}
.current-row {
  display: flex;
  align-items: center;
  gap: 18px;
  flex-wrap: wrap;
}
.mood-badge {
  width: 60px;
  height: 60px;
  border-radius: 50%;
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: color-mix(in srgb, currentColor 16%, transparent);
  border: 1px solid color-mix(in srgb, currentColor 32%, transparent);
  transition: background 0.4s ease, border-color 0.4s ease;
}
.pm-main { min-width: 0; }
.label-muted {
  font-size: 0.8rem;
  font-weight: 600;
  color: var(--muted);
  margin-bottom: 4px;
}
.eyebrow {
  text-transform: uppercase;
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.06em;
}
.pm-value-row {
  display: flex;
  align-items: baseline;
  gap: 4px;
}
.pm-value {
  font-size: 3.6rem;
  font-weight: 800;
  line-height: 1;
  letter-spacing: -0.02em;
}
.pm-unit {
  font-size: 1rem;
  font-weight: 500;
  color: var(--muted);
}
.status-chip {
  display: inline-flex;
  align-items: center;
  white-space: nowrap;
  padding: 7px 16px;
  border-radius: 999px;
  font-size: 0.85rem;
  font-weight: 700;
  background: color-mix(in srgb, currentColor 20%, var(--card-bg));
  border: 1px solid color-mix(in srgb, currentColor 38%, transparent);
  margin-left: auto;
}
.status-excellent { color: var(--excellent); }
.status-good { color: var(--good); }
.status-moderate { color: var(--moderate); }
.status-unhealthy { color: var(--unhealthy); }
.status-bad { color: var(--bad); }

.updated-at {
  color: var(--muted);
  font-size: 0.8rem;
  margin-top: 8px;
}

.stat-tiles {
  display: flex;
  flex-shrink: 0;
  padding-left: 28px;
  border-left: 1px solid var(--border);
}
@media (max-width: 880px) {
  .stat-tiles {
    padding-left: 0;
    border-left: none;
    margin-top: 16px;
    padding-top: 16px;
    border-top: 1px solid var(--border);
    width: 100%;
  }
}
.stat-tile {
  flex: 1;
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 4px;
  padding: 0 16px;
  text-align: center;
}
.stat-tile:not(:first-child) {
  border-left: 1px solid var(--border);
}
.stat-tile-icon { flex-shrink: 0; }
.icon-temp { color: #ef4444; }
.icon-humidity { color: #3b82f6; }
.icon-wind { color: #22c55e; }
.stat-tile-value { font-size: 1.1rem; font-weight: 700; margin-top: 2px; }
.stat-tile-label { color: var(--muted); font-size: 0.72rem; margin-top: 2px; }

@media (max-width: 480px) {
  .pm-value { font-size: 2.6rem; }
  .mood-badge { width: 48px; height: 48px; }
  .status-chip { padding: 5px 12px; font-size: 0.78rem; margin-left: 0; }
  .stat-tile-value { font-size: 0.95rem; }
  .stat-tile-label { font-size: 0.62rem; }
}

.daily-summary-value {
  font-size: 1.6rem;
  font-weight: 700;
  margin-bottom: 8px;
}
.daily-summary-unit { font-size: 0.95rem; font-weight: 400; color: var(--muted); }
.daily-summary-text {
  font-size: 0.92rem;
  line-height: 1.6;
  color: var(--text);
}

.best-times-list {
  display: flex;
  gap: 8px;
  flex-wrap: nowrap;
}
.best-time-row {
  flex: 1 1 0;
  min-width: 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  gap: 4px;
  font-size: 0.95rem;
  text-align: center;
}
.best-time-time,
.best-time-pm {
  white-space: nowrap;
  overflow: hidden;
  text-overflow: ellipsis;
  max-width: 100%;
}
.rank-badge {
  display: inline-flex;
  align-items: center;
  justify-content: center;
  width: 22px;
  height: 22px;
  border-radius: 50%;
  font-size: 0.75rem;
  font-weight: 700;
  flex-shrink: 0;
}
.rank-badge.rank-1 { background: var(--accent); color: #fff; }
.rank-badge.rank-2,
.rank-badge.rank-3 { background: transparent; color: var(--text); border: 1.5px solid var(--border); }
.best-time-time { font-weight: 600; }
.best-time-pm { color: var(--muted); font-size: 0.85rem; }

.activity-selector {
  display: flex;
  gap: 8px;
  margin-bottom: 14px;
}
.activity-btn {
  flex: 1;
  padding: 8px 0;
  border-radius: 10px;
  border: 1px solid var(--border);
  background: transparent;
  color: var(--text);
  font-size: 0.82rem;
  font-weight: 600;
  cursor: pointer;
  transition: background 0.2s ease, border-color 0.2s ease, color 0.2s ease;
}
.activity-btn:hover:not(.active) {
  border-color: var(--accent);
  color: var(--accent);
}
.activity-btn.active {
  background: var(--accent);
  border-color: var(--accent);
  color: #fff;
}
.activity-select-caption {
  margin-bottom: 8px;
}
.risk-score-caption {
  line-height: 1.4;
}
.risk-score-row {
  display: flex;
  align-items: center;
  gap: 12px;
}
.risk-score-value-wrap {
  display: flex;
  align-items: baseline;
  gap: 3px;
}
.risk-score-value {
  font-size: 2.4rem;
  font-weight: 800;
  line-height: 1;
}
.risk-score-unit {
  font-size: 0.9rem;
  font-weight: 500;
  color: var(--muted);
}
/* label เล็กๆ นอก chip (ไม่ยัดเข้าไปในคำเดียวกับ chip กันข้อความล้น/ตัดคำแปลกๆ บนจอเล็ก) แยกให้ชัดว่า
   ระดับนี้คือความเสี่ยงเฉพาะบุคคล ไม่ใช่ระดับ PM2.5 ดิบแบบการ์ดปัจจุบันด้านบน (ใช้คำ/สีชุดเดียวกัน
   เพราะ reuse statusClass() เดิม แต่ความหมายคนละแกน อาจดูเหมือนขัดแย้งกันถ้าไม่มี label นี้กำกับ) */
.risk-level-tag {
  display: flex;
  flex-direction: column;
  align-items: flex-end;
  gap: 3px;
  margin-left: auto;
}
.risk-level-label {
  font-size: 0.68rem;
  font-weight: 600;
  color: var(--muted);
  white-space: nowrap;
}
.risk-score-advice {
  margin-top: 8px;
  margin-bottom: 4px;
}
.risk-score-sublabel {
  margin-top: 16px;
}

.chart-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  flex-wrap: wrap;
  gap: 10px;
  margin-bottom: 14px;
}
.chart-title {
  font-size: 0.95rem;
  font-weight: 600;
  margin-bottom: 0;
}
.chart-subtitle {
  font-size: 0.75rem;
  font-weight: 500;
  color: var(--muted);
  margin-left: 4px;
}

.bars {
  display: flex;
  align-items: flex-end;
  gap: 6px;
  height: 220px;
  overflow-x: auto;
  padding-bottom: 4px;
}
.bar-col {
  flex: 1 0 20px;
  display: flex;
  flex-direction: column;
  align-items: center;
  height: 100%;
  min-width: 20px;
}
.bar-value {
  font-size: 0.65rem;
  font-weight: 600;
  color: var(--text);
  margin-bottom: 4px;
  white-space: nowrap;
}
.bar-track {
  flex: 1 1 auto;
  width: 100%;
  display: flex;
  align-items: flex-end;
  justify-content: center;
  position: relative;
}
.bar-range-cap {
  position: absolute;
  left: 0;
  right: 0;
  bottom: 0;
  border-radius: 8px 8px 0 0;
  pointer-events: none;
  z-index: 1;
  transition: height 0.6s cubic-bezier(0.4, 0, 0.2, 1);
}
.bar {
  position: relative;
  z-index: 2;
  width: 100%;
  border-radius: 8px 8px 0 0;
  min-height: 2px;
  transition: height 0.6s cubic-bezier(0.4, 0, 0.2, 1);
}
.bar:hover {
  opacity: 0.8;
}
.bar-label {
  font-size: 0.6rem;
  color: var(--muted);
  margin-top: 6px;
  writing-mode: vertical-rl;
  transform: rotate(180deg);
  white-space: nowrap;
}
.model-dot {
  width: 6px;
  height: 6px;
  border-radius: 50%;
  margin-top: 6px;
  flex-shrink: 0;
}

.chart-legend {
  display: flex;
  gap: 14px;
  flex-wrap: wrap;
  font-size: 0.72rem;
  color: var(--muted);
}
.chart-legend span { display: inline-flex; align-items: center; gap: 5px; }
.chart-legend .dot { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }

.info-box {
  background: rgba(37, 99, 235, 0.12);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(37, 99, 235, 0.35);
  color: var(--text);
  border-radius: 14px;
  padding: 12px 16px;
  margin-bottom: 16px;
  font-size: 0.85rem;
  line-height: 1.5;
}

/* loading skeleton */
.skeleton {
  background: linear-gradient(90deg, var(--border) 25%, transparent 37%, var(--border) 63%);
  background-size: 400% 100%;
  animation: shimmer 1.4s ease infinite;
  border-radius: 8px;
}
@keyframes shimmer {
  0% { background-position: 100% 50%; }
  100% { background-position: 0 50%; }
}
.skeleton-value { width: 120px; height: 48px; }
.skeleton-badge { width: 90px; height: 28px; border-radius: 999px; }
.skeleton-line { width: 180px; height: 14px; margin-top: 8px; }
.skeleton-chart { width: 100%; height: 160px; }

/* error state */
.error-box { text-align: center; padding: 32px 20px; }
.error-box p { color: var(--muted); margin-bottom: 16px; }
.retry-btn {
  background: var(--accent);
  color: #fff;
  border: none;
  padding: 10px 22px;
  border-radius: 8px;
  font-size: 0.9rem;
  cursor: pointer;
}
.retry-btn:hover { opacity: 0.9; }

.warning-box {
  background: rgba(220, 38, 38, 0.16);
  backdrop-filter: blur(10px);
  border: 1px solid rgba(220, 38, 38, 0.45);
  color: var(--text);
  border-radius: 14px;
  padding: 12px 16px;
  margin-bottom: 16px;
  font-size: 0.85rem;
  line-height: 1.5;
}
</style>

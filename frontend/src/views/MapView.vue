<script setup>
import { ref, onMounted, onUnmounted } from "vue";
import { useRouter } from "vue-router";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { severityHex } from "../pm25Levels";
import { selectProvince } from "../provinces";
import isaanMaskGeoJson from "../assets/isaan_provinces_mask.json";
import isaanBoundaryGeoJson from "../assets/isaan_provinces_boundary.json";

// ใช้ L.circleMarker ทั้งหมด ไม่ใช้ L.marker/ไอคอนหมุดปกติ — เลี่ยงปัญหารู้จักกันดีที่ Leaflet
// default icon path พังใต้ Vite bundler (คำนวณ path รูปจาก script tag ที่ bundler ไม่ได้ทำงานแบบนั้น)

// ขอบเขตจริงตามเส้นแบ่ง 20 จังหวัดอีสาน (GADM level-1, ผ่านทาง jeepkd/thailand.json gist ซึ่งรวม
// บึงกาฬที่ apisit/thailand.json ต้นทางเดิมไม่มี — ใบอนุญาตของ GADM ให้ใช้เพื่อการศึกษา/ไม่ใช่เชิง
// พาณิชย์ได้ฟรี ตรงกับงานวิทยานิพนธ์นี้) ตัดปัดทศนิยมเหลือ 4 ตำแหน่ง (~11 ม.) พอสำหรับแค่ mask ตกแต่ง
// ไม่ใช่ข้อมูลคำนวณ isaan_provinces_mask.json = สี่เหลี่ยมคลุมทั้งโลก + รู 20 จังหวัด (fillRule
// evenodd ของ Leaflet ทำให้พื้นที่ในรูโปร่งใส) isaan_provinces_boundary.json = ขอบเขต 20 จังหวัดล้วนๆ
// ใช้วาดเส้นเน้นทับอีกที ทั้งสองไฟล์สร้างจากสคริปต์ครั้งเดียว ไม่ผูก build step ถาวร
const router = useRouter();

const state = ref("loading"); // "loading" | "error" | "content"
const errorMessage = ref("ไม่สามารถโหลดข้อมูลได้ ลองใหม่อีกครั้ง");
const mapEl = ref(null);
let mapInstance = null;
const markerByProvince = new Map(); // province_en -> L.CircleMarker (คงไว้ข้ามการรีเฟรช ไม่ลบสร้างใหม่ทุกครั้ง
// กัน popup ที่เปิดอยู่โดนปิดตอนกด +/- ข้างใน — ดู createOrUpdateMarkers())

const ISAAN_CENTER = [15.8, 103.5];
const ISAAN_ZOOM = 7;
// bbox จริงของ 20 จังหวัด (lat 14.12-18.45, lon 100.83-105.64) บวก padding ~0.6 องศา กันการ pan/zoom
// ออกไปเห็นภาคอื่น/ประเทศเพื่อนบ้านเด่นเท่าอีสาน
const ISAAN_MAX_BOUNDS = [
  [13.5, 100.2],
  [19.05, 106.3],
];
const NO_DATA_COLOR = "#94a3b8";
const MIN_HORIZON = 0;
const MAX_HORIZON = 24;

// ปุ่ม +/- ใน popup ปรับได้ทีละ 1 ชม. เต็มช่วง 0-24 (เดิมเคยเป็นแถวปุ่ม preset 5 ปุ่มเหนือแผนที่ — user
// feedback 2026-09-23 ให้ย้ายเข้า popup และเลือกได้ทุกชั่วโมงไม่ใช่แค่ 6/12/18/24 — ยังใช้ horizon
// เดียวกันทั้งหน้า ไม่ใช่แยกต่อจังหวัด เพื่อให้ยังเห็นภาพรวมทั้ง 20 จังหวัด ณ เวลาเดียวกันได้เหมือนเดิม)
const selectedHorizon = ref(0);

function goToProvince(provinceEn) {
  selectProvince(provinceEn);
  router.push("/");
}

function stepHorizon(direction) {
  const next = selectedHorizon.value + direction;
  if (next < MIN_HORIZON || next > MAX_HORIZON) return;
  selectedHorizon.value = next;
  loadOverview();
}

async function loadOverview() {
  if (markerByProvince.size === 0) state.value = "loading"; // ครั้งแรกเท่านั้น ไม่งั้นกด +/- แล้วแผนที่กระพริบ
  try {
    const res = await fetch(`/api/provinces/overview?horizon=${selectedHorizon.value}`);
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.error || `Server error (${res.status})`);
    }
    const data = await res.json();
    state.value = "content";
    createOrUpdateMarkers(data);
  } catch (err) {
    errorMessage.value = "ไม่สามารถโหลดข้อมูลได้ ลองใหม่อีกครั้ง";
    state.value = "error";
    console.error("โหลดภาพรวมแผนที่ล้มเหลว:", err);
  }
}

function popupHtmlFor(p) {
  const horizon = selectedHorizon.value;
  const isForecast = horizon > 0;

  const valueText = isForecast
    ? (p.forecast_pm25 != null ? `${p.forecast_pm25} µg/m³` : "ไม่มีข้อมูล")
    : (p.current_pm25 != null ? `${p.current_pm25} µg/m³` : "ไม่มีข้อมูล");
  const statusText = isForecast ? (p.forecast_status || "-") : (p.current_status || "-");
  const timeLabel = isForecast
    ? `พยากรณ์เวลา ${p.forecast_target_time_th || "-"}`
    : `อัปเดตล่าสุด ${p.current_recorded_at_th || "-"}`;
  const fallbackNote =
    isForecast && p.forecast_model_used === "naive_fallback"
      ? `<div class="map-popup-fallback-note">(ใช้ค่าคงที่ชั่วคราว เนื่องจากข้อมูลนำเข้าขาดหาย)</div>`
      : "";

  const nowLabel = horizon === 0 ? `<div class="map-popup-now-hint">ตอนนี้</div>` : "";
  const minusDisabled = horizon === MIN_HORIZON ? "disabled" : "";
  const plusDisabled = horizon === MAX_HORIZON ? "disabled" : "";

  return `
    <div class="map-popup">
      <div class="map-popup-title">${p.province_th}${p.is_zero_shot ? " (zero-shot)" : ""}</div>
      <div class="map-popup-pm25">${valueText}</div>
      <div class="map-popup-status">${statusText}</div>
      <div class="map-popup-time">${timeLabel}</div>
      ${fallbackNote}
      ${nowLabel}
      <div class="map-popup-stepper">
        <button type="button" class="map-popup-step map-popup-minus" ${minusDisabled}>−</button>
        <input type="number" class="map-popup-horizon-input" min="${MIN_HORIZON}" max="${MAX_HORIZON}" value="${horizon}" />
        <span class="map-popup-step-unit">ชม. ข้างหน้า</span>
        <button type="button" class="map-popup-step map-popup-plus" ${plusDisabled}>+</button>
      </div>
      <button type="button" class="map-popup-btn">ดูรายละเอียด →</button>
    </div>
  `;
}


// ผูก event handler ให้ปุ่มใน popup — ใช้ .onclick = แทน addEventListener เพราะ setPopupContent()
// (เรียกตอนรีเฟรช popup ที่เปิดค้างอยู่) แทนที่ innerHTML ทั้งก้อน แต่ elements เดิมที่เคย
// addEventListener ไว้จะหลุดไปกับ innerHTML เก่า ต้อง bind ใหม่ทุกครั้งที่เนื้อหาถูกแทนที่ — .onclick =
// เขียนทับตัวเก่าเองในตัว ไม่มี listener ซ้อนสะสม
function bindPopupButtons(marker, provinceEn) {
  const el = marker.getPopup()?.getElement();
  if (!el) return;
  const detailBtn = el.querySelector(".map-popup-btn");
  if (detailBtn) detailBtn.onclick = () => goToProvince(provinceEn);
  const minusBtn = el.querySelector(".map-popup-minus");
  if (minusBtn) minusBtn.onclick = () => stepHorizon(-1);
  const plusBtn = el.querySelector(".map-popup-plus");
  if (plusBtn) plusBtn.onclick = () => stepHorizon(1);

  const input = el.querySelector(".map-popup-horizon-input");
  if (input) {
    const commitInput = () => {
      let next = parseInt(input.value, 10);
      if (Number.isNaN(next)) next = selectedHorizon.value;
      next = Math.min(MAX_HORIZON, Math.max(MIN_HORIZON, next));
      input.value = next; // สะท้อนค่าหลัง clamp ให้เห็นทันที ไม่ต้องรอรอบ fetch ถัดไป
      if (next !== selectedHorizon.value) {
        selectedHorizon.value = next;
        loadOverview();
      }
    };
    input.onchange = commitInput;
    input.onkeydown = (e) => {
      if (e.key === "Enter") input.blur(); // blur -> trigger change -> commitInput
    };
  }
}

// สร้าง marker ครั้งแรก, ครั้งต่อไป (กด +/- ในบางจังหวัด) แค่อัปเดตสี+เนื้อหา popup ของ marker เดิม
// ไม่ลบสร้างใหม่ทั้งหมด — กัน popup ที่เปิดอยู่ตอนกด +/- โดนปิดไปเอง
function createOrUpdateMarkers(provinces) {
  if (!mapInstance) return;
  const horizon = selectedHorizon.value;

  provinces.forEach((p) => {
    const valueForColor = horizon > 0 ? p.forecast_pm25 : p.current_pm25;
    const color = valueForColor != null ? severityHex(valueForColor) : NO_DATA_COLOR;
    let marker = markerByProvince.get(p.province_en);

    if (!marker) {
      marker = L.circleMarker([p.lat, p.lon], {
        radius: 11,
        fillColor: color,
        color: "#ffffff",
        weight: 2,
        fillOpacity: 0.9,
      }).addTo(mapInstance);
      marker.bindPopup(popupHtmlFor(p));
      marker.on("popupopen", () => bindPopupButtons(marker, p.province_en));
      markerByProvince.set(p.province_en, marker);
      return;
    }

    marker.setStyle({ fillColor: color });
    marker.setPopupContent(popupHtmlFor(p));
    if (marker.isPopupOpen()) bindPopupButtons(marker, p.province_en);
  });
}

onMounted(() => {
  mapInstance = L.map(mapEl.value, {
    minZoom: ISAAN_ZOOM, // ห้ามซูมออกเลย (คงมุมมองเริ่มต้นเป็นระดับที่แคบที่สุดที่อนุญาต)
    maxBounds: ISAAN_MAX_BOUNDS,
    maxBoundsViscosity: 1.0, // กัน pan หลุดกรอบแบบยางยืด ให้รู้สึกเหมือนขอบแข็งจริงๆ
  }).setView(ISAAN_CENTER, ISAAN_ZOOM);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
    attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
    maxZoom: 18,
  }).addTo(mapInstance);

  // มืดคลุมพื้นที่นอกอีสาน (interactive: false กันบัง click/hover ของ tile ใต้มาส์ก) + เส้นขอบเน้น
  // เขตอีสานทับอีกชั้น ให้รู้สึก "โฟกัส" ชัดเจนกว่าแค่ล็อก bounds เฉยๆ
  L.geoJSON(isaanMaskGeoJson, {
    interactive: false,
    style: { fillColor: "#0f172a", fillOpacity: 0.45, stroke: false },
  }).addTo(mapInstance);
  L.geoJSON(isaanBoundaryGeoJson, {
    interactive: false,
    style: { color: "#3b82f6", weight: 1.5, fill: false, dashArray: "4,3", opacity: 0.7 },
  }).addTo(mapInstance);

  loadOverview();
});

onUnmounted(() => {
  if (mapInstance) {
    mapInstance.remove();
    mapInstance = null;
  }
  markerByProvince.clear();
});
</script>

<template>
  <div class="map-page">
    <div class="haze-decor"><span></span><span></span><span></span></div>

    <div class="wrap">
      <h1>ภาพรวมคุณภาพอากาศ 20 จังหวัดภาคอีสาน</h1>
      <div class="subtitle">แตะหมุดแล้วกด +/− ในป๊อปอัปเพื่อดูค่าพยากรณ์ล่วงหน้า</div>

      <div v-if="state === 'error'" class="card error-box">
        <p>{{ errorMessage }}</p>
        <button class="retry-btn" @click="loadOverview">ลองอีกครั้ง</button>
      </div>

      <div class="card map-card">
        <div ref="mapEl" class="leaflet-container-el"></div>
        <div v-if="state === 'loading'" class="map-loading-overlay">กำลังโหลดข้อมูล...</div>
      </div>

      <div class="card legend-card">
        <div class="legend-title">ระดับคุณภาพอากาศ (PM2.5 เฉลี่ยรายชั่วโมง, µg/m³)</div>
        <div class="legend-list">
          <span class="legend-item"><span class="legend-dot" style="background: var(--excellent)"></span>ดีมาก (0-15)</span>
          <span class="legend-item"><span class="legend-dot" style="background: var(--good)"></span>ดี (15-25)</span>
          <span class="legend-item"><span class="legend-dot" style="background: var(--moderate)"></span>ปานกลาง (25-37.5)</span>
          <span class="legend-item"><span class="legend-dot" style="background: var(--unhealthy)"></span>เริ่มมีผลกระทบต่อสุขภาพ (37.5-75)</span>
          <span class="legend-item"><span class="legend-dot" style="background: var(--bad)"></span>มีผลกระทบต่อสุขภาพ (75+)</span>
          <span class="legend-item"><span class="legend-dot" style="background: #94a3b8"></span>ไม่มีข้อมูล</span>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.map-page {
  --sky-top: #4a7fae;
  --sky-mid: #8fb4d1;
  --sky-bottom: #dce8f0;
  --page-text: #0f172a;
  --page-text-muted: rgba(15, 23, 42, 0.7);

  --card-bg: rgba(255, 255, 255, 0.65);
  --card-border: rgba(255, 255, 255, 0.4);
  --text: #0f172a;
  --muted: #64748b;
  --border: rgba(255, 255, 255, 0.3);
  --accent: #3b82f6;

  --excellent: #3b82f6;
  --good: #22c55e;
  --moderate: #eab308;
  --unhealthy: #f97316;
  --bad: #ef4444;

  min-height: 100vh;
  background: linear-gradient(180deg, var(--sky-top) 0%, var(--sky-mid) 55%, var(--sky-bottom) 100%);
  color: var(--text);
  padding: 100px 16px 48px;
  position: relative;
  overflow-x: hidden;
}
@media (prefers-color-scheme: dark) {
  .map-page {
    --card-bg: rgba(22, 25, 30, 0.55);
    --card-border: rgba(255, 255, 255, 0.1);
    --text: #f8fafc;
    --muted: #94a3b8;
    --border: rgba(255, 255, 255, 0.1);
  }
}

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
  background: rgba(255, 255, 255, 0.3);
  animation: float 24s ease-in-out infinite;
}
.haze-decor span:nth-child(1) { width: 220px; height: 220px; top: 6%;  left: -60px; animation-duration: 22s; }
.haze-decor span:nth-child(2) { width: 160px; height: 160px; top: 40%; right: -40px; animation-duration: 26s; animation-delay: -4s; }
.haze-decor span:nth-child(3) { width: 240px; height: 240px; top: 78%; left: 15%; animation-duration: 30s; animation-delay: -10s; }
@keyframes float {
  0%, 100% { transform: translate(0, 0); }
  50% { transform: translate(24px, -18px); }
}

.wrap { max-width: 860px; margin: 0 auto; position: relative; z-index: 1; }

h1 { font-size: 1.3rem; margin: 0 0 4px; color: var(--page-text); }
.subtitle { color: var(--page-text-muted); font-size: 0.85rem; margin-bottom: 20px; }

.card {
  background: var(--card-bg);
  border: 1px solid var(--card-border);
  backdrop-filter: blur(24px);
  -webkit-backdrop-filter: blur(24px);
  box-shadow: 0 10px 40px rgba(0, 0, 0, 0.08);
  border-radius: 24px;
  padding: 24px;
  margin-bottom: 20px;
}

.map-card { padding: 0; overflow: hidden; position: relative; }
.leaflet-container-el { width: 100%; height: 60vh; min-height: 420px; border-radius: 24px; }
.map-loading-overlay {
  position: absolute;
  inset: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: rgba(255, 255, 255, 0.5);
  font-weight: 600;
  color: var(--text);
}

.legend-title { font-weight: 700; margin-bottom: 10px; }
.legend-list { display: flex; flex-wrap: wrap; gap: 14px; font-size: 0.85rem; color: var(--text); }
.legend-item { display: inline-flex; align-items: center; gap: 6px; }
.legend-dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }

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

/* popup DOM ถูก Leaflet inject นอกขอบเขต scoped style ของ Vue จึงต้องใช้ :deep() */
:deep(.map-popup) { font-size: 0.85rem; line-height: 1.5; min-width: 190px; }
:deep(.map-popup-title) { font-weight: 700; margin-bottom: 6px; }
:deep(.map-popup-pm25) { font-size: 1.1rem; font-weight: 800; }
:deep(.map-popup-status) { color: #334155; }
:deep(.map-popup-time) { color: #64748b; font-size: 0.75rem; margin: 4px 0 8px; }
:deep(.map-popup-fallback-note) { color: #b45309; font-size: 0.72rem; margin-bottom: 6px; }
:deep(.map-popup-stepper) {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  margin: 8px 0;
  padding: 6px 0;
  border-top: 1px solid #e2e8f0;
  border-bottom: 1px solid #e2e8f0;
}
:deep(.map-popup-step) {
  border: none;
  background: #3b82f6;
  color: #fff;
  width: 22px;
  height: 22px;
  border-radius: 50%;
  font-size: 0.9rem;
  line-height: 1;
  cursor: pointer;
  flex-shrink: 0;
}
:deep(.map-popup-step:disabled) { background: #cbd5e1; cursor: not-allowed; }
:deep(.map-popup-step:not(:disabled):hover) { opacity: 0.85; }
:deep(.map-popup-horizon-input) {
  width: 40px;
  text-align: center;
  font-size: 0.85rem;
  font-weight: 700;
  border: 1px solid #cbd5e1;
  border-radius: 6px;
  padding: 3px 2px;
  -moz-appearance: textfield;
}
:deep(.map-popup-step-unit) { font-size: 0.75rem; color: #64748b; }
:deep(.map-popup-now-hint) { text-align: center; font-size: 0.72rem; color: #64748b; margin-top: 2px; }
:deep(.map-popup-btn) {
  border: none;
  background: #3b82f6;
  color: #fff;
  padding: 6px 12px;
  border-radius: 8px;
  font-size: 0.8rem;
  cursor: pointer;
  width: 100%;
}
:deep(.map-popup-btn:hover) { opacity: 0.9; }

@media (max-width: 600px) {
  /* header กลายเป็น 2 แถวคงที่บนจอแคบ (ดู AppHeader.vue) ต้องเผื่อ padding-top เพิ่มไม่งั้นเนื้อหา
     ด้านบนโดนบัง */
  .map-page { padding-top: 148px; }
  .leaflet-container-el { height: 50vh; min-height: 320px; }
  :deep(.map-popup-step) { width: 28px; height: 28px; } /* เป้าแตะใหญ่ขึ้นสำหรับนิ้ว */
}
</style>

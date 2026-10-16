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
// bbox จริงของ 20 จังหวัด (lat 14.12-18.45, lon 100.83-105.64) บวก padding กันการ pan/zoom ออกไปเห็นภาคอื่น/
// ประเทศเพื่อนบ้านเด่นเท่าอีสาน ขอบเหนือ (lat) เผื่อเพิ่มเป็นพิเศษ (~2.85 องศา แทนที่จะเป็น ~0.6 องศาเท่าขอบอื่น)
// เพราะ maxBoundsViscosity: 1.0 ด้านล่างล็อกขอบแข็ง ถ้าจังหวัดติดขอบ (เช่น เลย/หนองคาย/บึงกาฬ/นครพนม) ถูกกด
// popup จะพยายาม autoPan ขึ้นให้พ้นขอบจอ แต่ขยับไม่ได้เพราะชนขอบเขตเดิม ทำให้ popup โดนตัดบนจริง (user เจอ 2026-10-02,
// ขยายรอบแรกเป็น 20.6 ยังไม่พอ ขยายเพิ่มอีกรอบเป็น 21.3)
const ISAAN_MAX_BOUNDS = [
  [13.5, 100.2],
  [21.3, 106.3],
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

// badge วงกลมมีตัวเลข PM2.5 อยู่ข้างใน (แทน L.circleMarker เดิมที่เป็นจุดสีเปล่าๆ ไม่มีที่ใส่ตัวหนังสือ)
// ปัดเป็นจำนวนเต็มให้พอดีช่องวงกลมเล็กๆ ค่าทศนิยมเต็มยังดูได้จาก popup/tooltip ตามปกติ
function markerIcon(value, color) {
  const label = value != null ? Math.round(value) : "-";
  return L.divIcon({
    html: `<div class="map-marker-badge" style="background:${color}">${label}</div>`,
    className: "map-marker-badge-wrap",
    iconSize: [30, 30],
    iconAnchor: [15, 15],
  });
}

// กล่อง hover เล็กๆ บอก PM2.5 แยกจาก popup ตอนคลิก (คนละกลไกของ Leaflet ใช้ร่วมกันได้ไม่ชนกัน)
function tooltipHtmlFor(p) {
  const horizon = selectedHorizon.value;
  const pm = horizon > 0 ? p.forecast_pm25 : p.current_pm25;
  return `
    <div class="map-hover-title">${p.province_th}</div>
    <div class="map-hover-row">PM2.5: ${pm != null ? pm + " µg/m³" : "-"}</div>
  `;
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
      marker = L.marker([p.lat, p.lon], { icon: markerIcon(valueForColor, color) }).addTo(mapInstance);
      marker.bindPopup(popupHtmlFor(p));
      marker.bindTooltip(tooltipHtmlFor(p), { direction: "top", offset: [0, -16], className: "map-hover-tooltip" });
      marker.on("popupopen", () => bindPopupButtons(marker, p.province_en));
      markerByProvince.set(p.province_en, marker);
      return;
    }

    // L.marker+divIcon ไม่มี setStyle() แบบ circleMarker เดิม ต้อง setIcon() ใหม่ทั้งก้อนแทน
    marker.setIcon(markerIcon(valueForColor, color));
    marker.setPopupContent(popupHtmlFor(p));
    marker.setTooltipContent(tooltipHtmlFor(p));
    if (marker.isPopupOpen()) bindPopupButtons(marker, p.province_en);
  });
}

onMounted(() => {
  mapInstance = L.map(mapEl.value, {
    minZoom: ISAAN_ZOOM, // ห้ามซูมออกเลย (คงมุมมองเริ่มต้นเป็นระดับที่แคบที่สุดที่อนุญาต)
    maxBounds: ISAAN_MAX_BOUNDS,
    maxBoundsViscosity: 1.0, // กัน pan หลุดกรอบแบบยางยืด ให้รู้สึกเหมือนขอบแข็งจริงๆ
    zoomControl: false, // ปิด default (มุมซ้ายบน) แล้วสร้างใหม่มุมขวาบนด้านล่างแทน
  }).setView(ISAAN_CENTER, ISAAN_ZOOM);
  L.control.zoom({ position: "topright" }).addTo(mapInstance);
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
    <div class="wrap">
      <h1>ภาพรวมคุณภาพอากาศ</h1>
      <p class="subtitle">20 จังหวัดภาคอีสาน · แตะหมุดแล้วกด +/− ในป๊อปอัปเพื่อดูค่าพยากรณ์ล่วงหน้า</p>

      <div v-if="state === 'error'" class="card error-box" role="alert">
        <p>{{ errorMessage }}</p>
        <button type="button" class="retry-btn" @click="loadOverview">ลองอีกครั้ง</button>
      </div>

      <div class="card map-card">
        <div ref="mapEl" class="leaflet-container-el"></div>
        <div v-if="state === 'loading'" class="map-loading-overlay">กำลังโหลดข้อมูล...</div>
      </div>

      <section class="card legend-card">
        <h2 class="legend-title">ระดับคุณภาพอากาศ</h2>
        <p class="legend-sub">PM2.5 รายชั่วโมง (µg/m³)</p>
        <ul class="legend-list">
          <li class="legend-item"><span class="legend-dot" style="background: var(--excellent)"></span><span class="legend-name">ดีมาก</span><span class="legend-range">0–15</span></li>
          <li class="legend-item"><span class="legend-dot" style="background: var(--good)"></span><span class="legend-name">ดี</span><span class="legend-range">15–25</span></li>
          <li class="legend-item"><span class="legend-dot" style="background: var(--moderate)"></span><span class="legend-name">ปานกลาง</span><span class="legend-range">25–37.5</span></li>
          <li class="legend-item"><span class="legend-dot" style="background: var(--unhealthy)"></span><span class="legend-name">เริ่มมีผลกระทบต่อสุขภาพ</span><span class="legend-range">37.5–75</span></li>
          <li class="legend-item"><span class="legend-dot" style="background: var(--bad)"></span><span class="legend-name">มีผลกระทบต่อสุขภาพ</span><span class="legend-range">75+</span></li>
          <li class="legend-item"><span class="legend-dot" style="background: var(--nodata)"></span><span class="legend-name">ไม่มีข้อมูล</span></li>
        </ul>
      </section>
    </div>
  </div>
</template>

<style scoped>
/* โทเคนกลาง (ground/card/ink/muted/line/track/accent) มาจาก :root ใน assets/main.css ที่โหลดทั้งแอป
   อยู่แล้ว (เดิมประกาศซ้ำไว้ในนี้ด้วย — เอาออกเพื่อกันสองที่ไม่ตรงกัน) เหลือแค่โทเคนสีระดับฝุ่นที่หน้านี้ใช้เฉพาะตัวเอง */
.map-page {
  --excellent: #3b82f6;
  --good: #22c55e;
  --moderate: #eab308;
  --unhealthy: #f97316;
  --bad: #ef4444;
  --nodata: #94a3b8;

  min-height: 100vh;
  background: var(--ground);
  color: var(--ink);
  padding: var(--app-top-offset) 16px calc(var(--app-bottom-offset) + 24px);
}


.wrap { max-width: 860px; margin: 0 auto; }

h1 { font-size: 1.5rem; font-weight: 700; margin: 0 0 4px; }
.subtitle { color: var(--muted); font-size: 0.9rem; line-height: 1.5; margin: 0 0 16px; }

.card {
  background: var(--card);
  border: 1px solid var(--line);
  border-radius: 24px;
  padding: 20px;
  margin-bottom: 16px;
}

/* แผนที่ (พื้นภาพ + ปุ่ม zoom + popup + attribution) ใช้ธีมสว่างเสมอ ไม่ตามโหมดมืด — ตั้งใจทับค่าจาก
   main.css ด้วยโทเคนสว่างตรงนี้ (ไม่ใช่ของซ้ำที่ลบได้แบบ .map-page ด้านบน) ทำให้ลูกทั้งหมด (รวม DOM ของ
   Leaflet/popup) รับค่าสว่างไปด้วย ส่วนการ์ด legend และพื้นหน้ายังตามธีมปกติ */
.map-card {
  --card: #ffffff;
  --ink: #16181d;
  --muted: #565c66;
  --line: #e6e1d6;
  --track: #ece8de;
  --accent: #2563eb;
  padding: 0;
  overflow: hidden;
  position: relative;
  color: var(--ink);
}
.leaflet-container-el { width: 100%; height: 60vh; min-height: 420px; border-radius: 24px; background: var(--track); }
.map-loading-overlay {
  position: absolute;
  inset: 0;
  z-index: 500;
  display: flex;
  align-items: center;
  justify-content: center;
  background: color-mix(in srgb, var(--card) 70%, transparent);
  font-weight: 600;
  color: var(--ink);
}

.legend-title { margin: 0; font-size: 1.05rem; font-weight: 600; }
.legend-sub { margin: 2px 0 12px; font-size: 0.8rem; color: var(--muted); }
.legend-list { list-style: none; margin: 0; padding: 0; display: grid; grid-template-columns: 1fr; }
.legend-item {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 0;
  border-top: 1px solid var(--line);
  font-size: 0.95rem;
}
.legend-name { flex: 1; }
.legend-range { color: var(--muted); font-size: 0.85rem; }
.legend-dot { width: 12px; height: 12px; border-radius: 50%; flex-shrink: 0; }
@media (min-width: 721px) {
  .legend-list { grid-template-columns: 1fr 1fr; column-gap: 32px; }
}

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

/* ---- ปรับ Leaflet ให้เข้าธีม (DOM ของ Leaflet อยู่ใต้ .map-page จึงรับตัวแปรสีได้ แต่อยู่นอกขอบเขต
   scoped ของ Vue จึงต้องใช้ :deep()) ---- */
:deep(.leaflet-container) { font-family: inherit; }
:deep(.leaflet-bar) { border: 1px solid var(--line); box-shadow: none; border-radius: 14px; overflow: hidden; }
:deep(.leaflet-bar a) {
  width: 40px;
  height: 40px;
  line-height: 40px;
  background: var(--card);
  color: var(--ink);
  border-bottom: 1px solid var(--line);
}
:deep(.leaflet-bar a:last-child) { border-bottom: 0; }
:deep(.leaflet-bar a:hover) { background: var(--track); }
:deep(.leaflet-bar a.leaflet-disabled) { color: var(--muted); opacity: 0.5; background: var(--card); }
:deep(.leaflet-control-attribution) {
  background: color-mix(in srgb, var(--card) 80%, transparent);
  color: var(--muted);
  font-size: 0.65rem;
}
:deep(.leaflet-control-attribution a) { color: var(--accent); }

/* หมุดวงกลมมีตัวเลข (L.divIcon) — wrapper ต้องล้าง style ของ Leaflet default icon (พื้นหลัง/เงา/ขอบ) ทิ้งก่อน
   ให้ div ลูกข้างในคุมหน้าตาเองทั้งหมด */
:deep(.map-marker-badge-wrap) { background: transparent; border: none; }
:deep(.map-marker-badge) {
  width: 30px;
  height: 30px;
  border-radius: 50%;
  border: 2px solid #ffffff;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.3);
  display: flex;
  align-items: center;
  justify-content: center;
  color: #ffffff;
  font-weight: 600;
  font-size: 0.65rem;
  font-family: inherit; /* ฟอนต์ตัวเนื้อหาปกติ ไม่ใช่ --font-display (ฟอนต์หัวข้อตัวหนา ดูหนัก/ใหญ่เกินไปในวงเล็กๆ) */
  letter-spacing: -0.2px;
  line-height: 1;
  padding-top: 1px; /* line-height:1 เหลือที่ว่างใต้ตัวเลขมากกว่าบน (ฟอนต์ไม่มี descender) เลยดูลอยขึ้นบน ขยับลงชดเชย */
}

/* กล่อง hover (bindTooltip) แยกจาก popup ตอนคลิก — ใช้โทเคนสีเดียวกับ .map-card */
:deep(.map-hover-tooltip) {
  background: var(--card);
  color: var(--ink);
  border: 1px solid var(--line);
  border-radius: 10px;
  padding: 6px 10px;
  box-shadow: 0 6px 18px rgba(0, 0, 0, 0.18);
  opacity: 1;
}
:deep(.map-hover-tooltip)::before { border-top-color: var(--card); }
:deep(.map-hover-title) { font-weight: 700; font-size: 0.8rem; margin-bottom: 2px; }
:deep(.map-hover-row) { font-size: 0.75rem; color: var(--muted); white-space: nowrap; }

:deep(.leaflet-popup-content-wrapper) {
  background: var(--card);
  color: var(--ink);
  border-radius: 18px;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.25);
}
:deep(.leaflet-popup-tip) { background: var(--card); }
:deep(.leaflet-popup-content) { margin: 14px 16px; }
:deep(.leaflet-popup-close-button) { color: var(--muted) !important; width: 32px !important; height: 32px !important; font-size: 22px !important; }

/* popup */
:deep(.map-popup) { font-size: 0.9rem; line-height: 1.5; min-width: 200px; }
:deep(.map-popup-title) { font-weight: 700; font-size: 1rem; margin-bottom: 6px; padding-right: 16px; }
:deep(.map-popup-pm25) { font-family: var(--font-display); font-size: 1.5rem; font-weight: 600; line-height: 1.2; }
:deep(.map-popup-status) { color: var(--ink); font-weight: 500; }
:deep(.map-popup-time) { color: var(--muted); font-size: 0.78rem; margin: 4px 0 8px; }
:deep(.map-popup-fallback-note) { color: #b45309; font-size: 0.75rem; margin-bottom: 6px; }
:deep(.map-popup-stepper) {
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 10px;
  margin: 8px 0;
  padding: 8px 0;
  border-top: 1px solid var(--line);
  border-bottom: 1px solid var(--line);
}
:deep(.map-popup-step) {
  border: none;
  background: var(--accent);
  color: #fff;
  width: 32px;
  height: 32px;
  border-radius: 50%;
  font-size: 1.1rem;
  line-height: 1;
  cursor: pointer;
  flex-shrink: 0;
}
:deep(.map-popup-step:disabled) { background: var(--track); color: var(--muted); cursor: not-allowed; }
:deep(.map-popup-step:not(:disabled):hover) { opacity: 0.85; }
:deep(.map-popup-horizon-input) {
  width: 46px;
  text-align: center;
  font-family: inherit;
  font-size: 1rem; /* >=16px กัน iOS zoom เข้าช่องพิมพ์ */
  font-weight: 700;
  color: var(--ink);
  background: var(--track);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 4px 2px;
  -moz-appearance: textfield;
}
:deep(.map-popup-step-unit) { font-size: 0.78rem; color: var(--muted); }
:deep(.map-popup-now-hint) { text-align: center; font-size: 0.75rem; color: var(--muted); margin-top: 2px; }
:deep(.map-popup-btn) {
  border: none;
  background: var(--ink);
  color: var(--card);
  min-height: 40px;
  padding: 0 14px;
  border-radius: 20px;
  font-family: inherit;
  font-size: 0.9rem;
  font-weight: 600;
  cursor: pointer;
  width: 100%;
}
:deep(.map-popup-btn:hover) { opacity: 0.9; }

@media (max-width: 600px) {
  .leaflet-container-el { height: 55vh; min-height: 360px; }
  :deep(.map-popup-step) { width: 40px; height: 40px; } /* เป้าแตะ ≥40px สำหรับนิ้ว */
}
</style>

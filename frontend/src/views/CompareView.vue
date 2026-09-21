<script setup>
import { ref, computed, onMounted, nextTick, watch } from "vue";
import { CircleCheck, CircleAlert, CircleX, ChevronDown } from "@lucide/vue";
import { provinceState } from "../provinces";

const horizon = ref(1);
const model = ref("all");
const showAdvanced = ref(false);

const state = ref("loading"); // "loading" | "error" | "content"
const errorMessage = ref("ไม่สามารถโหลดข้อมูลได้ ลองใหม่อีกครั้ง");

const summary = ref({
  mae: null, bias: null, n_matched: 0, n_total: 0, n_backfill: 0, n_used: 0, basis: null,
  coverage_pct: null, n_coverage: 0,
});
const points = ref([]);
const currentHorizonLabel = ref(1);

const overviewLoaded = ref(false);
const overviewData = ref(null);
const chartEl = ref(null);

// สีดิบสำหรับใช้ใน SVG fill attribute โดยตรง (ไม่ผ่าน CSS var เพื่อความชัวร์ข้าม browser) — ใช้แค่ใน
// model-breakdown chip ของส่วน "รายละเอียดทางเทคนิค" เท่านั้น กราฟหลักไม่ encode สีตามโมเดลแล้ว (ดู
// drawChart() — ผู้ใช้ตัดสินใจแล้วว่าให้ซ่อนชื่อ/สีโมเดลไว้ในส่วนขั้นสูงเท่านั้น)
function modelHex(modelUsed) {
  if (String(modelUsed).startsWith("xgboost")) return "#1baf7a";
  if (modelUsed === "naive_fallback") return "#eda100";
  return "#2a78d6";
}

function qualityForMae(mae) {
  if (mae <= 1.5) return { label: "แม่นยำมาก", cls: "good" };
  if (mae <= 3) return { label: "แม่นยำดี", cls: "good" };
  if (mae <= 5) return { label: "แม่นยำปานกลาง", cls: "moderate" };
  return { label: "คลาดเคลื่อนค่อนข้างมาก", cls: "bad" };
}

function biasPhrase(bias) {
  if (bias > 0.5) {
    return `และมีแนวโน้ม<strong>ทายค่าต่ำกว่าความเป็นจริง</strong> โดยเฉลี่ยประมาณ ${bias.toFixed(2)} µg/m³`;
  }
  if (bias < -0.5) {
    return `และมีแนวโน้ม<strong>ทายค่าสูงกว่าความเป็นจริง</strong> โดยเฉลี่ยประมาณ ${Math.abs(bias).toFixed(2)} µg/m³`;
  }
  return `และ<strong>ไม่มีอคติชัดเจน</strong> (ไม่ได้ทายสูงหรือต่ำกว่าจริงอย่างเป็นระบบ)`;
}

// coverage: % ของค่าจริงที่ตกอยู่ในช่วงความเชื่อมั่น 90% (P05-P95 conformal) — เป้าหมายตามงานวิจัยคือ
// ~90% พอดี (ALPHA=0.10 ใน scripts/train_multiprovince_models.py) ไม่ใช่ "ยิ่งสูงยิ่งดี" เหมือน MAE
// เพราะช่วงกว้างเกินไปก็ทำให้ coverage สูงเกิน 90% ได้เหมือนกัน (แต่แปลว่าช่วงกว้างเกินความจำเป็น
// ไม่มีประโยชน์กับผู้ใช้) จึงให้ข้อความสามทาง: ใกล้เป้า / ต่ำกว่าเป้า / สูงกว่าเป้า
function coveragePhrase(pct, n) {
  if (pct == null) return "";
  const diff = pct - 90;
  let verdict;
  if (Math.abs(diff) <= 5) verdict = "ใกล้เคียงเป้าหมาย ~90% ตามที่ออกแบบไว้";
  else if (diff < 0) verdict = "ต่ำกว่าเป้าหมาย ~90% — ช่วงที่แสดงอาจแคบไปสำหรับสภาพจริงตอนนี้";
  else verdict = "สูงกว่าเป้าหมาย ~90% — ช่วงที่แสดงกว้างกว่าที่จำเป็น (ระวังไม่ได้)";
  return `ช่วงความเชื่อมั่น 90% ที่แสดงบนกราฟครอบคลุมค่าจริงได้จริง <strong>${pct.toFixed(1)}%</strong> ของคู่ที่วัดได้ (n=${n}) ${verdict}`;
}

function barColorForMae(mae) {
  const q = qualityForMae(mae);
  if (q.cls === "good") return "var(--good)";
  if (q.cls === "moderate") return "var(--moderate)";
  return "var(--bad)";
}

// ไอคอนหัวข้อการ์ด "คำตอบหลัก" — คนละชุดกับไอคอนหน้ายิ้มของ ForecastView โดยตั้งใจ (ที่นั่นสื่อ
// สถานะคุณภาพอากาศ ที่นี่สื่อ "ความน่าเชื่อถือของค่าพยากรณ์" คนละเรื่องกัน ใช้ปนกันจะสับสน)
function qualityIcon(cls) {
  if (cls === "good") return CircleCheck;
  if (cls === "moderate") return CircleAlert;
  return CircleX;
}

const summaryQuality = computed(() => (summary.value.n_matched > 0 ? qualityForMae(summary.value.mae) : null));

const summaryHtml = computed(() => {
  const { mae, bias, n_matched, n_total, basis, n_used } = summary.value;
  if (n_matched === 0) {
    return n_total === 0
      ? `ยังไม่เคยพยากรณ์ล่วงหน้า <strong>${currentHorizonLabel.value} ชั่วโมง</strong> เลยตั้งแต่เริ่มเก็บข้อมูล — ลองเปิดหน้าแรกเพื่อให้ระบบพยากรณ์ก่อน แล้วค่อยกลับมาดูที่นี่`
      : `มีการพยากรณ์ล่วงหน้า <strong>${currentHorizonLabel.value} ชั่วโมง</strong> ไว้แล้ว ${n_total} ครั้ง แต่ยังไม่ถึงเวลาของครั้งไหนเลย จึงยังไม่มีค่าจริงมาเทียบ — เปิดหน้านี้ใหม่อีกครั้งหลังผ่านไปสักพัก`;
  }
  return (
    `จากพยากรณ์ล่วงหน้า <strong>${currentHorizonLabel.value} ชั่วโมง</strong> ที่มีค่าจริงมาเทียบได้แล้ว <strong>${n_matched}</strong> ครั้ง ` +
    `(จากทั้งหมด ${n_total} ครั้งที่เคยพยากรณ์ไว้) ค่าพยากรณ์คลาดเคลื่อนจากความเป็นจริงโดยเฉลี่ย <strong>±${mae.toFixed(2)} µg/m³</strong> ` +
    biasPhrase(bias) +
    (basis === "mixed"
      ? ` <span class="basis-note">(ยังมีข้อมูล live ไม่พอ เลขนี้รวม backfill ชั่วคราวอยู่ n=${n_used})</span>`
      : basis === "live"
      ? ` <span class="basis-note">(อิงจาก live data ล้วนๆ n=${n_used})</span>`
      : "")
  );
});

const summaryNoteText = computed(() =>
  summary.value.n_matched > 0 && currentHorizonLabel.value >= 12
    ? "หมายเหตุ: การพยากรณ์ล่วงหน้านานขึ้นมักคลาดเคลื่อนมากขึ้นเป็นธรรมชาติ ไม่ได้แปลว่าโมเดลแย่ลงเสมอไป"
    : ""
);

const summaryCoverageHtml = computed(() =>
  coveragePhrase(summary.value.coverage_pct, summary.value.n_coverage)
);

const emptyMsgVisible = computed(() => points.value.filter((p) => p.actual != null).length === 0);

async function loadComparison() {
  state.value = "loading";
  try {
    const res = await fetch(`/api/comparison?horizon=${horizon.value}&model=${model.value}&province=${provinceState.selected}`);
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.error || `Server error (${res.status})`);
    }
    const data = await res.json();
    summary.value = data.summary;
    points.value = data.points;
    currentHorizonLabel.value = data.horizon;
    state.value = "content";
    await nextTick();
    drawChart(points.value);
  } catch (err) {
    errorMessage.value = "ไม่สามารถโหลดข้อมูลได้ ลองใหม่อีกครั้ง";
    state.value = "error";
    console.error("โหลดข้อมูลเปรียบเทียบล้มเหลว:", err);
  }
}

function drawChart(pts) {
  const svg = chartEl.value;
  if (!svg) return;
  svg.innerHTML = "";

  const padding = { top: 12, right: 16, bottom: 34, left: 40 };
  const pointGap = 26;
  const width = Math.max(560, padding.left + padding.right + pts.length * pointGap);
  const height = 260;
  svg.setAttribute("width", width);
  svg.setAttribute("height", height);
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

  if (pts.length === 0) return;

  const values = [];
  pts.forEach((p) => {
    values.push(p.predicted);
    if (p.actual != null) values.push(p.actual);
    if (p.pm25_lo != null) values.push(p.pm25_lo);
    if (p.pm25_hi != null) values.push(p.pm25_hi);
  });
  const minV = Math.min(...values, 0);
  const maxV = Math.max(...values, 37.5);
  const yRange = maxV - minV || 1;

  const plotW = width - padding.left - padding.right;
  const plotH = height - padding.top - padding.bottom;

  const xFor = (i) => padding.left + (pts.length === 1 ? plotW / 2 : (i / (pts.length - 1)) * plotW);
  const yFor = (v) => padding.top + plotH - ((v - minV) / yRange) * plotH;

  const ns = "http://www.w3.org/2000/svg";

  for (let i = 0; i <= 4; i++) {
    const v = minV + (yRange * i) / 4;
    const y = yFor(v);
    const line = document.createElementNS(ns, "line");
    line.setAttribute("x1", padding.left);
    line.setAttribute("x2", width - padding.right);
    line.setAttribute("y1", y);
    line.setAttribute("y2", y);
    line.setAttribute("class", "grid-line");
    svg.appendChild(line);

    const label = document.createElementNS(ns, "text");
    label.setAttribute("x", padding.left - 6);
    label.setAttribute("y", y + 3);
    label.setAttribute("text-anchor", "end");
    label.setAttribute("class", "axis-label");
    label.textContent = v.toFixed(0);
    svg.appendChild(label);
  }

  const labelStep = Math.max(1, Math.ceil(pts.length / 10));
  pts.forEach((p, i) => {
    if (i % labelStep !== 0 && i !== pts.length - 1) return;
    const label = document.createElementNS(ns, "text");
    label.setAttribute("x", xFor(i));
    label.setAttribute("y", height - padding.bottom + 16);
    label.setAttribute("text-anchor", "middle");
    label.setAttribute("class", "axis-label");
    label.textContent = p.time;
    svg.appendChild(label);
  });

  // ช่วงความเชื่อมั่น 90% (P05-P95 conformal) เป็น area wash สีเดียวกับเส้นพยากรณ์ ความเข้ม ~10%
  // ตาม mark spec ของ dataviz skill (area fill = series hue ที่ ~10% opacity, ไม่ใช่บล็อกทึบ) วาด
  // แยกเป็นช่วงต่อเนื่อง (run) เพราะบางจุดอาจไม่มี pm25_lo/pm25_hi (แถวเก่าก่อน CHANGELOG 81 หรือยัง
  // ไม่ backfill) ข้ามจุดที่ไม่มีค่าแทนที่จะทำให้ทั้งแถบขาดหาย
  function drawBand() {
    let run = [];
    const runs = [];
    pts.forEach((p, i) => {
      if (p.pm25_lo != null && p.pm25_hi != null) {
        run.push(i);
      } else if (run.length) {
        runs.push(run);
        run = [];
      }
    });
    if (run.length) runs.push(run);

    runs.forEach((r) => {
      if (r.length < 2) return;
      let d = "";
      r.forEach((i, idx) => {
        d += (idx === 0 ? "M" : "L") + xFor(i) + "," + yFor(pts[i].pm25_hi) + " ";
      });
      for (let k = r.length - 1; k >= 0; k--) {
        d += "L" + xFor(r[k]) + "," + yFor(pts[r[k]].pm25_lo) + " ";
      }
      d += "Z";
      const band = document.createElementNS(ns, "path");
      band.setAttribute("d", d.trim());
      band.setAttribute("fill", "rgba(37, 99, 235, 0.12)");
      band.setAttribute("stroke", "none");
      svg.appendChild(band);
    });
  }
  drawBand();

  function drawPath(key, color) {
    let d = "";
    let drawing = false;
    pts.forEach((p, i) => {
      const v = p[key];
      if (v == null) { drawing = false; return; }
      const x = xFor(i), y = yFor(v);
      d += (drawing ? "L" : "M") + x + "," + y + " ";
      drawing = true;
    });
    const path = document.createElementNS(ns, "path");
    path.setAttribute("d", d.trim());
    path.setAttribute("fill", "none");
    path.setAttribute("stroke", color);
    path.setAttribute("stroke-width", "2");
    svg.appendChild(path);
  }

  drawPath("predicted", "#2563eb");
  drawPath("actual", "#16a34a");

  // จุดค่าจริง: วงกลมทึบสีเขียวเดิม (ground truth ไม่ขึ้นกับโมเดล)
  pts.forEach((p, i) => {
    if (p.actual == null) return;
    const circle = document.createElementNS(ns, "circle");
    circle.setAttribute("cx", xFor(i));
    circle.setAttribute("cy", yFor(p.actual));
    circle.setAttribute("r", "2.5");
    circle.setAttribute("fill", "#16a34a");
    circle.appendChild(Object.assign(document.createElementNS(ns, "title"), { textContent: `${p.time}: actual ${p.actual}` }));
    svg.appendChild(circle);
  });

  // จุดค่าพยากรณ์: สีเดียวเสมอ (ไม่ encode โมเดล/backfill-live ด้วยสี/รูปทรงบนกราฟหลักแล้ว ตาม
  // การตัดสินใจ "ซ่อนรายละเอียดโมเดลไว้ในส่วนขั้นสูง") — รายละเอียดโมเดล/แหล่งข้อมูลยังอยู่ใน
  // tooltip (hover ดูได้) ไม่ได้ทิ้งข้อมูลไป แค่ไม่บังคับให้ทุกคนต้องแปลรหัสสีบนหน้าจอ
  pts.forEach((p, i) => {
    const circle = document.createElementNS(ns, "circle");
    circle.setAttribute("cx", xFor(i));
    circle.setAttribute("cy", yFor(p.predicted));
    circle.setAttribute("r", "3");
    circle.setAttribute("fill", "#2563eb");
    const rangeText = p.pm25_lo != null && p.pm25_hi != null ? `, ช่วง ${p.pm25_lo}-${p.pm25_hi}` : "";
    circle.appendChild(Object.assign(document.createElementNS(ns, "title"), {
      textContent: `${p.time}: predicted ${p.predicted}${rangeText} (${p.model_used}, ${p.source})`,
    }));
    svg.appendChild(circle);
  });
}

async function loadOverview() {
  try {
    const res = await fetch(`/api/comparison/overview?province=${provinceState.selected}`);
    if (!res.ok) throw new Error(`Server error (${res.status})`);
    overviewData.value = await res.json();
    overviewLoaded.value = true;
  } catch (err) {
    overviewLoaded.value = false;
    console.error("โหลดภาพรวมความแม่นยำล้มเหลว:", err);
  }
}

const overviewQuality = computed(() =>
  overviewData.value && overviewData.value.n_matched > 0 ? qualityForMae(overviewData.value.overall_mae) : null
);

const overviewHtml = computed(() => {
  const d = overviewData.value;
  if (!d) return "โหลดภาพรวมความแม่นยำไม่สำเร็จ";
  if (d.n_matched === 0) {
    return `ยังไม่มีข้อมูลค่าจริงมาเทียบเลยสักคู่ (มีการพยากรณ์ไว้ทั้งหมด ${d.n_total} รายการ) — เปิดหน้านี้ใหม่อีกครั้งหลังผ่านไปสักพัก`;
  }
  return (
    `จากข้อมูลที่เทียบได้ทั้งหมด <strong>${d.n_matched}</strong> คู่ (จาก ${d.n_total} ครั้งที่เคยพยากรณ์ไว้ ทุกช่วงเวลารวมกัน) ` +
    `ความคลาดเคลื่อนเฉลี่ยรวมอยู่ที่ <strong>±${d.overall_mae.toFixed(2)} µg/m³</strong> ` +
    biasPhrase(d.overall_bias) +
    (d.overall_basis === "mixed"
      ? ` <span class="basis-note">(ยังมีข้อมูล live ไม่พอ เลขนี้รวม backfill ชั่วคราวอยู่ n=${d.overall_n_used})</span>`
      : d.overall_basis === "live"
      ? ` <span class="basis-note">(อิงจาก live data ล้วนๆ n=${d.overall_n_used})</span>`
      : "")
  );
});

const overviewCoverageHtml = computed(() => {
  const d = overviewData.value;
  if (!d) return "";
  return coveragePhrase(d.overall_coverage_pct, d.overall_n_coverage);
});

const overviewNoteHtml = computed(() => {
  const d = overviewData.value;
  if (!d || d.n_matched === 0) return "";
  if (d.best_horizon && d.worst_horizon && d.best_horizon.horizon !== d.worst_horizon.horizon) {
    return (
      `แม่นยำที่สุดที่ <strong>t+${d.best_horizon.horizon}</strong> (±${d.best_horizon.mae}) ` +
      `คลาดเคลื่อนมากสุดที่ <strong>t+${d.worst_horizon.horizon}</strong> (±${d.worst_horizon.mae}) — ` +
      `เป็นไปตามธรรมชาติที่ยิ่งพยากรณ์ไกลยิ่งคลาดเคลื่อนมากขึ้น ไม่ได้แปลว่าโมเดลแย่ลง`
    );
  }
  return "";
});

const overviewBars = computed(() => {
  const byHorizon = overviewData.value?.by_horizon || [];
  const maxMae = Math.max(...byHorizon.map((b) => b.mae ?? 0), 1);
  return byHorizon.map((b) => ({
    ...b,
    heightPct: b.mae != null ? (b.mae / maxMae) * 100 : null,
    color: b.mae != null ? barColorForMae(b.mae) : null,
    label: b.horizon % 4 === 1 || b.horizon === 24 ? b.horizon : "",
  }));
});

// รุ่นเก่า (ไม่มี suffix) ยังอยู่ในแถวเก่าของ prediction_log — v2 คือโมเดลรุ่นล่าสุด (21 feature, CHANGELOG 141)
const modelLabel = {
  lightgbm_pooled_v3: "LightGBM (v3 ข้อมูลแก้แล้ว)", xgboost_pooled_v3: "XGBoost (v3 ข้อมูลแก้แล้ว)",
  lightgbm_pooled_v2: "LightGBM (v2 ข้อมูลป้ายเวลาผิด)", xgboost_pooled_v2: "XGBoost (v2 ข้อมูลป้ายเวลาผิด)",
  lightgbm_pooled: "LightGBM (รุ่นก่อน)", xgboost_pooled: "XGBoost (รุ่นก่อน)", naive_fallback: "Fallback",
};
const modelBreakdown = computed(() => {
  const byModel = overviewData.value?.by_model || [];
  return byModel.map((m) => ({
    ...m,
    label: modelLabel[m.model_used] || m.model_used,
    color: modelHex(m.model_used),
    line1: `MAE ±${m.mae} µg/m³ (n=${m.n}${m.basis === "mixed" ? ", รวม backfill" : m.basis === "live" ? ", live ล้วน" : ""})`,
    line2: `backfill ${m.n_backfill} / live ${m.n_live}`,
    // naive_fallback ไม่มี coverage (ช่วงกว้าง 0 เสมอ ไม่ใช่ conformal interval จริง — ดู backend)
    line3: m.coverage_pct != null ? `coverage ${m.coverage_pct}% (n=${m.n_coverage})` : m.model_used === "naive_fallback" ? "ไม่มีช่วงความเชื่อมั่น (persistence)" : null,
  }));
});

onMounted(() => {
  loadOverview();
  loadComparison();
});

// สลับจังหวัดจาก dropdown บน header ขณะเปิดหน้านี้ค้างไว้ — ต้องรีโหลดทั้งภาพรวมและกราฟรายละเอียด
// ตาม pattern เดียวกับ ForecastView.vue (ไม่งั้นข้อมูลจะค้างเป็นของจังหวัดเดิมจนกว่าจะรีเฟรชหน้าเอง)
watch(() => provinceState.selected, () => {
  loadOverview();
  loadComparison();
});
</script>

<template>
  <div class="compare-page">
    <div class="haze-decor"><span></span><span></span><span></span></div>

    <div class="wrap">
      <div class="top-row">
        <div>
          <h1>เปรียบเทียบผลพยากรณ์กับค่าจริง</h1>
          <div class="subtitle">ดูว่าที่ผ่านมาระบบพยากรณ์ PM2.5 แม่นยำแค่ไหน เทียบกับค่าที่วัดได้จริง</div>
        </div>
        <router-link class="back-link" to="/">← กลับหน้าแรก</router-link>
      </div>

      <!-- การ์ดคำตอบหลัก — สิ่งแรกที่ควรเห็น: "พยากรณ์แม่นยำแค่ไหน" แบบภาพรวม ไม่ต้องอ่านศัพท์สถิติ -->
      <div class="card hero-card">
        <div class="hero-row">
          <div class="hero-icon" :class="overviewQuality ? 'quality-' + overviewQuality.cls : ''">
            <component :is="qualityIcon(overviewQuality?.cls)" :size="36" :stroke-width="2" v-if="overviewQuality" />
          </div>
          <div class="hero-text">
            <div class="hero-label" :class="overviewQuality ? 'quality-text-' + overviewQuality.cls : ''">
              {{ overviewQuality ? overviewQuality.label : "กำลังประเมิน..." }}
            </div>
            <div class="hero-detail" v-html="overviewLoaded ? overviewHtml : 'กำลังโหลดภาพรวม...'"></div>
            <div v-if="overviewCoverageHtml" class="summary-note" v-html="overviewCoverageHtml"></div>
          </div>
        </div>
      </div>

      <!-- ความแม่นยำตามระยะเวลาพยากรณ์ล่วงหน้า -->
      <div class="card">
        <div class="section-title">ความแม่นยำ ตามระยะเวลาที่พยากรณ์ล่วงหน้า</div>
        <div class="bar-legend">
          <span><span class="dot" style="background: var(--good)"></span>แม่นยำ</span>
          <span><span class="dot" style="background: var(--moderate)"></span>ปานกลาง</span>
          <span><span class="dot" style="background: var(--bad)"></span>คลาดเคลื่อนมาก</span>
        </div>
        <div class="mini-bars">
          <div class="mini-bar-col" v-for="b in overviewBars" :key="b.horizon">
            <div class="mini-bar-track">
              <div v-if="b.mae != null" class="mini-bar" :style="{ height: b.heightPct + '%', background: b.color }" :title="`t+${b.horizon}: คลาดเคลื่อนเฉลี่ย ${b.mae} µg/m³ (n=${b.n})`"></div>
              <div v-else class="mini-bar-placeholder" :title="`t+${b.horizon}: ยังไม่มีข้อมูล`"></div>
            </div>
            <div class="mini-bar-label">{{ b.label }}</div>
          </div>
        </div>
        <div v-if="overviewNoteHtml" class="summary-note" v-html="overviewNoteHtml"></div>
      </div>

      <!-- ตัวอย่างค่าพยากรณ์เทียบค่าจริงต่อชั่วโมง -->
      <div class="card">
        <div class="controls">
          <label for="horizonSelect">ดูตัวอย่างค่าพยากรณ์ล่วงหน้า:</label>
          <select id="horizonSelect" v-model.number="horizon" @change="loadComparison">
            <option v-for="h in 24" :key="h" :value="h">t+{{ h }} ชม.</option>
          </select>
        </div>

        <div v-if="state === 'loading'" class="skeleton skeleton-chart" style="margin-top: 16px"></div>

        <div v-else-if="state === 'error'" class="error-box">
          <p>{{ errorMessage }}</p>
          <button class="retry-btn" @click="loadComparison">ลองอีกครั้ง</button>
        </div>

        <div v-else>
          <div class="detail-badge-row">
            <span v-if="summaryQuality" class="quality-badge" :class="'quality-' + summaryQuality.cls">{{ summaryQuality.label }}</span>
            <span class="detail-text" v-html="summaryHtml"></span>
          </div>
          <div v-if="summaryNoteText" class="summary-note">{{ summaryNoteText }}</div>
          <div v-if="summaryCoverageHtml" class="summary-note" v-html="summaryCoverageHtml"></div>

          <div class="legend">
            <span><span class="dot" style="background: var(--predicted)"></span>ค่าพยากรณ์</span>
            <span><span class="dot" style="background: var(--actual)"></span>ค่าจริง</span>
            <span><span class="swatch-band"></span>ช่วงความเชื่อมั่น 90%</span>
          </div>
          <div class="chart-wrap">
            <svg ref="chartEl"></svg>
          </div>
          <div v-if="emptyMsgVisible" class="empty-msg">
            ยังไม่มีข้อมูลค่าจริงให้เปรียบเทียบสำหรับช่วงเวลานี้ — ต้องรอจนถึงเวลาที่พยากรณ์ไว้ก่อน แล้วเปิดหน้านี้ใหม่อีกครั้ง
          </div>
        </div>
      </div>

      <!-- รายละเอียดทางเทคนิค — ซ่อนไว้เป็นค่าเริ่มต้น เพื่อไม่ให้ศัพท์สถิติ/ชื่อโมเดลรบกวนภาพรวมด้านบน -->
      <button type="button" class="advanced-toggle" @click="showAdvanced = !showAdvanced">
        <ChevronDown :size="16" :stroke-width="2" :class="{ 'toggle-icon-open': showAdvanced }" />
        {{ showAdvanced ? "ซ่อนรายละเอียดทางเทคนิค" : "ดูรายละเอียดทางเทคนิค" }}
      </button>

      <div class="card" v-if="showAdvanced">
        <div class="section-title">ตัวเลขดิบของช่วงเวลาที่เลือก (t+{{ currentHorizonLabel }} ชม.)</div>
        <div class="stats-row">
          <div class="stat">
            <div class="stat-value">{{ summary.mae ?? "-" }}</div>
            <div class="stat-label">MAE — ค่าคลาดเคลื่อนเฉลี่ย (µg/m³)</div>
          </div>
          <div class="stat">
            <div class="stat-value">{{ summary.bias != null ? (summary.bias > 0 ? "+" : "") + summary.bias : "-" }}</div>
            <div class="stat-label">Bias — ค่าเอนเอียงเฉลี่ย (+ = ทายต่ำกว่าจริง)</div>
          </div>
          <div class="stat">
            <div class="stat-value">{{ summary.n_matched }} / {{ summary.n_total }}</div>
            <div class="stat-label">คู่ที่เทียบได้ / ทั้งหมด</div>
          </div>
          <div class="stat">
            <div class="stat-value">{{ summary.n_backfill }}</div>
            <div class="stat-label">เป็น backfill (ทดสอบย้อนหลัง ไม่ใช่พยากรณ์สด)</div>
          </div>
          <div class="stat">
            <div class="stat-value">{{ summary.coverage_pct != null ? summary.coverage_pct + "%" : "-" }}</div>
            <div class="stat-label">Coverage — ค่าจริงตกในช่วง 90% กี่ % (เป้า ~90%, n={{ summary.n_coverage }})</div>
          </div>
        </div>

        <div class="section-title" style="margin-top: 20px">แยกตามโมเดลที่ระบบ Dynamic Routing เลือกใช้ (ภาพรวมทุก horizon)</div>
        <div class="controls" style="margin-bottom: 4px">
          <label for="modelSelect">กรองกราฟด้านบนเฉพาะโมเดล:</label>
          <select id="modelSelect" v-model="model" @change="loadComparison">
            <option value="all">ทุกโมเดล</option>
            <option value="lightgbm_pooled_v3">LightGBM (v3 ข้อมูลแก้แล้ว)</option>
            <option value="xgboost_pooled_v3">XGBoost (v3 ข้อมูลแก้แล้ว)</option>
            <option value="lightgbm_pooled_v2">LightGBM (v2 ข้อมูลป้ายเวลาผิด)</option>
            <option value="xgboost_pooled_v2">XGBoost (v2 ข้อมูลป้ายเวลาผิด)</option>
            <option value="lightgbm_pooled">LightGBM (รุ่นก่อน)</option>
            <option value="xgboost_pooled">XGBoost (รุ่นก่อน)</option>
            <option value="naive_fallback">Fallback</option>
          </select>
        </div>
        <div class="model-breakdown">
          <div class="model-chip" v-for="m in modelBreakdown" :key="m.model_used" :style="{ borderLeftColor: m.color }">
            <div class="name">{{ m.label }}</div>
            <div class="stat-line">{{ m.line1 }}</div>
            <div class="stat-line">{{ m.line2 }}</div>
            <div v-if="m.line3" class="stat-line">{{ m.line3 }}</div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.compare-page {
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
  --good: #22c55e;
  --moderate: #f97316;
  --bad: #ef4444;
  --accent: #3b82f6;
  --predicted: #3b82f6;
  --actual: #22c55e;
  --select-bg: rgba(255, 255, 255, 0.6);

  --model-lightgbm: #6366f1;
  --model-xgboost: #10b981;
  --model-fallback: #ec4899;

  min-height: 100vh;
  background: linear-gradient(180deg, var(--sky-top) 0%, var(--sky-mid) 55%, var(--sky-bottom) 100%);
  color: var(--text);
  padding: 100px 16px 48px;
  position: relative;
  overflow-x: hidden;
}
@media (prefers-color-scheme: dark) {
  .compare-page {
    --card-bg: rgba(22, 25, 30, 0.55);
    --card-border: rgba(255, 255, 255, 0.1);
    --text: #f8fafc;
    --muted: #94a3b8;
    --border: rgba(255, 255, 255, 0.1);
    --select-bg: rgba(0, 0, 0, 0.35);
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

.top-row {
  display: flex;
  justify-content: space-between;
  align-items: flex-start;
  gap: 12px;
  flex-wrap: wrap;
}
h1 { font-size: 1.3rem; margin: 0 0 4px; color: var(--page-text); }
.subtitle { color: var(--page-text-muted); font-size: 0.85rem; margin-bottom: 20px; }
.back-link {
  color: var(--page-text);
  text-decoration: none;
  font-size: 0.85rem;
  white-space: nowrap;
  opacity: 0.85;
}
.back-link:hover { text-decoration: underline; }

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
  box-shadow: 0 16px 50px rgba(0, 0, 0, 0.12);
  transform: translateY(-2px);
}

.hero-card { padding: 28px 24px; }
.hero-row {
  display: flex;
  align-items: center;
  gap: 18px;
}
.hero-icon {
  width: 60px;
  height: 60px;
  border-radius: 50%;
  flex-shrink: 0;
  display: flex;
  align-items: center;
  justify-content: center;
  background: color-mix(in srgb, currentColor 16%, transparent);
  border: 1px solid color-mix(in srgb, currentColor 32%, transparent);
}
.hero-text { min-width: 0; }
.hero-label {
  font-size: 1.7rem;
  font-weight: 800;
  line-height: 1.2;
  margin-bottom: 6px;
}
.hero-detail {
  font-size: 0.92rem;
  line-height: 1.6;
  color: var(--text);
}
.quality-good { color: var(--good); }
.quality-moderate { color: var(--moderate); }
.quality-bad { color: var(--bad); }
.quality-text-good { color: var(--good); }
.quality-text-moderate { color: var(--moderate); }
.quality-text-bad { color: var(--bad); }

@media (max-width: 480px) {
  .hero-row { gap: 12px; }
  .hero-icon { width: 48px; height: 48px; }
  .hero-label { font-size: 1.3rem; }
}

.section-title {
  font-size: 0.95rem;
  font-weight: 600;
  margin-bottom: 10px;
}

.bar-legend {
  display: flex;
  gap: 14px;
  flex-wrap: wrap;
  font-size: 0.78rem;
  color: var(--muted);
  margin-bottom: 4px;
}
.bar-legend span { display: inline-flex; align-items: center; gap: 5px; }
.bar-legend .dot { width: 9px; height: 9px; border-radius: 50%; display: inline-block; }

.controls {
  display: flex;
  align-items: center;
  gap: 10px;
  font-size: 0.9rem;
  flex-wrap: wrap;
}
select {
  background: var(--select-bg);
  color: var(--text);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 8px 12px;
  font-size: 0.95rem;
  transition: all 0.2s ease;
}
select:focus {
  outline: none;
  border-color: var(--accent);
  box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.2);
}

.quality-badge {
  display: inline-block;
  padding: 5px 12px;
  border-radius: 999px;
  font-size: 0.85rem;
  font-weight: 600;
  color: #fff;
  white-space: nowrap;
  flex-shrink: 0;
}
.quality-badge.quality-good { background: var(--good); color: #fff; }
.quality-badge.quality-moderate { background: var(--moderate); color: #fff; }
.quality-badge.quality-bad { background: var(--bad); color: #fff; }

.detail-badge-row {
  display: flex;
  align-items: baseline;
  gap: 10px;
  flex-wrap: wrap;
  margin-top: 14px;
  margin-bottom: 4px;
}
.detail-text {
  font-size: 0.9rem;
  line-height: 1.6;
  color: var(--text);
}

.summary-note {
  color: var(--muted);
  font-size: 0.8rem;
  margin-top: 8px;
}
.hero-detail :deep(.basis-note),
.detail-text :deep(.basis-note),
.summary-note :deep(.basis-note) {
  color: var(--muted);
  font-size: 0.85em;
}

.stats-row {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
}
.stat {
  flex: 1 1 140px;
  text-align: center;
}
.stat-value { font-size: 1.6rem; font-weight: 700; }
.stat-label { color: var(--muted); font-size: 0.75rem; margin-top: 2px; }

.mini-bars {
  display: flex;
  align-items: flex-end;
  gap: 3px;
  height: 90px;
  margin-top: 10px;
}
.mini-bar-col {
  flex: 1 1 0;
  display: flex;
  flex-direction: column;
  align-items: center;
  height: 100%;
  min-width: 8px;
}
.mini-bar-track {
  flex: 1 1 auto;
  width: 100%;
  display: flex;
  align-items: flex-end;
  justify-content: center;
}
.mini-bar {
  width: 70%;
  border-radius: 2px 2px 0 0;
  min-height: 2px;
}
.mini-bar-placeholder {
  width: 40%;
  height: 4px;
  border-radius: 2px;
  background: var(--border);
}
.mini-bar-label {
  font-size: 0.55rem;
  color: var(--muted);
  margin-top: 4px;
  height: 12px;
}

.legend {
  display: flex;
  gap: 16px;
  flex-wrap: wrap;
  font-size: 0.8rem;
  color: var(--muted);
  margin: 12px 0 10px;
}
.legend span { display: inline-flex; align-items: center; gap: 6px; }
.legend .dot { width: 10px; height: 10px; border-radius: 50%; display: inline-block; }
.legend .swatch-band {
  width: 16px;
  height: 10px;
  border-radius: 3px;
  display: inline-block;
  background: rgba(37, 99, 235, 0.12);
}

.advanced-toggle {
  display: flex;
  align-items: center;
  gap: 6px;
  margin: 0 auto 20px;
  padding: 8px 16px;
  background: transparent;
  border: 1px dashed var(--border);
  border-radius: 999px;
  color: var(--muted);
  font-size: 0.85rem;
  cursor: pointer;
  transition: all 0.2s ease;
}
.advanced-toggle:hover {
  color: var(--accent);
  border-color: var(--accent);
}
.advanced-toggle :deep(svg) {
  transition: transform 0.2s ease;
}
.advanced-toggle :deep(.toggle-icon-open) {
  transform: rotate(180deg);
}

.model-breakdown {
  display: flex;
  gap: 10px;
  flex-wrap: wrap;
}
.model-chip {
  flex: 1 1 140px;
  border-radius: 12px;
  padding: 10px 12px;
  background: rgba(127, 127, 127, 0.08);
  border-left: 4px solid var(--muted);
}
.model-chip .name {
  font-size: 0.8rem;
  font-weight: 600;
  margin-bottom: 4px;
}
.model-chip .stat-line {
  font-size: 0.75rem;
  color: var(--muted);
  line-height: 1.5;
}

.chart-wrap { overflow-x: auto; }
svg { display: block; }
.chart-wrap :deep(.axis-label) { font-size: 9px; fill: var(--muted); }
.chart-wrap :deep(.grid-line) { stroke: var(--border); stroke-width: 1; }

.empty-msg {
  color: var(--muted);
  font-size: 0.85rem;
  text-align: center;
  padding: 24px;
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
.skeleton-chart { width: 100%; height: 260px; }

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
</style>

const horizonSelect = document.getElementById("horizonSelect");
for (let h = 1; h <= 24; h++) {
  const opt = document.createElement("option");
  opt.value = h;
  opt.textContent = `t+${h} ชม.`;
  horizonSelect.appendChild(opt);
}
horizonSelect.addEventListener("change", loadComparison);

const modelSelect = document.getElementById("modelSelect");
modelSelect.addEventListener("change", loadComparison);

function modelColor(modelUsed) {
  if (modelUsed === "xgboost") return "var(--model-xgboost)";
  if (modelUsed === "prophet") return "var(--model-prophet)";
  return "var(--model-lightgbm)";
}

// สีดิบสำหรับใช้ใน SVG fill attribute โดยตรง (ไม่ผ่าน CSS var เพื่อความชัวร์ข้าม browser)
function modelHex(modelUsed) {
  if (modelUsed === "xgboost") return "#1baf7a";
  if (modelUsed === "prophet") return "#eda100";
  return "#2a78d6";
}

function showState(state) {
  document.getElementById("loadingState").classList.toggle("hidden", state !== "loading");
  document.getElementById("errorState").classList.toggle("hidden", state !== "error");
  document.getElementById("content").classList.toggle("hidden", state !== "content");
}

async function loadComparison() {
  showState("loading");
  try {
    const horizon = horizonSelect.value;
    const model = modelSelect.value;
    const res = await fetch(`/api/comparison?horizon=${horizon}&model=${model}`);
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.error || `Server error (${res.status})`);
    }
    const data = await res.json();
    renderComparison(data);
    showState("content");
  } catch (err) {
    document.getElementById("errorMessage").textContent =
      "ไม่สามารถโหลดข้อมูลได้ ลองใหม่อีกครั้ง";
    showState("error");
    console.error("โหลดข้อมูลเปรียบเทียบล้มเหลว:", err);
  }
}

function renderComparison(data) {
  document.getElementById("statMae").textContent = data.summary.mae ?? "-";
  document.getElementById("statBias").textContent =
    data.summary.bias != null ? (data.summary.bias > 0 ? "+" : "") + data.summary.bias : "-";
  document.getElementById("statMatched").textContent = `${data.summary.n_matched} / ${data.summary.n_total}`;
  document.getElementById("statBackfill").textContent = data.summary.n_backfill;

  renderSummarySentence(data);

  const matchedPoints = data.points.filter(p => p.actual != null);
  document.getElementById("emptyMsg").classList.toggle("hidden", matchedPoints.length > 0);

  drawChart(data.points);
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

function renderSummarySentence(data) {
  const { mae, bias, n_matched, n_total, basis, n_used } = data.summary;
  const badge = document.getElementById("qualityBadge");
  const text = document.getElementById("summaryText");
  const note = document.getElementById("summaryNote");

  if (n_matched === 0) {
    badge.classList.add("hidden");
    note.classList.add("hidden");
    text.innerHTML = n_total === 0
      ? `ยังไม่เคยพยากรณ์ล่วงหน้า <strong>${data.horizon} ชั่วโมง</strong> เลยตั้งแต่เริ่มเก็บข้อมูล — ลองเปิดหน้าแรกเพื่อให้ระบบพยากรณ์ก่อน แล้วค่อยกลับมาดูที่นี่`
      : `มีการพยากรณ์ล่วงหน้า <strong>${data.horizon} ชั่วโมง</strong> ไว้แล้ว ${n_total} ครั้ง แต่ยังไม่ถึงเวลาของครั้งไหนเลย จึงยังไม่มีค่าจริงมาเทียบ — เปิดหน้านี้ใหม่อีกครั้งหลังผ่านไปสักพัก`;
    return;
  }

  const q = qualityForMae(mae);
  badge.textContent = q.label;
  badge.className = "quality-badge quality-" + q.cls;
  badge.classList.remove("hidden");

  text.innerHTML =
    `จากพยากรณ์ล่วงหน้า <strong>${data.horizon} ชั่วโมง</strong> ที่มีค่าจริงมาเทียบได้แล้ว <strong>${n_matched}</strong> ครั้ง ` +
    `(จากทั้งหมด ${n_total} ครั้งที่เคยพยากรณ์ไว้) ค่าพยากรณ์คลาดเคลื่อนจากความเป็นจริงโดยเฉลี่ย <strong>±${mae.toFixed(2)} µg/m³</strong> ` +
    biasPhrase(bias) +
    (basis === "mixed"
      ? ` <span class="basis-note">(ยังมีข้อมูล live ไม่พอ เลขนี้รวม backfill ชั่วคราวอยู่ n=${n_used})</span>`
      : basis === "live"
      ? ` <span class="basis-note">(อิงจาก live data ล้วนๆ n=${n_used})</span>`
      : "");

  if (data.horizon >= 12) {
    note.textContent = "หมายเหตุ: การพยากรณ์ล่วงหน้านานขึ้นมักคลาดเคลื่อนมากขึ้นเป็นธรรมชาติ ไม่ได้แปลว่าโมเดลแย่ลงเสมอไป";
    note.classList.remove("hidden");
  } else {
    note.classList.add("hidden");
  }
}

function drawChart(points) {
  const svg = document.getElementById("chart");
  svg.innerHTML = "";

  const padding = { top: 12, right: 16, bottom: 34, left: 40 };
  const pointGap = 26;
  const width = Math.max(560, padding.left + padding.right + points.length * pointGap);
  const height = 260;
  svg.setAttribute("width", width);
  svg.setAttribute("height", height);
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);

  if (points.length === 0) return;

  const values = [];
  points.forEach(p => {
    values.push(p.predicted);
    if (p.actual != null) values.push(p.actual);
  });
  const minV = Math.min(...values, 0);
  const maxV = Math.max(...values, 37.5);
  const yRange = maxV - minV || 1;

  const plotW = width - padding.left - padding.right;
  const plotH = height - padding.top - padding.bottom;

  const xFor = i => padding.left + (points.length === 1 ? plotW / 2 : (i / (points.length - 1)) * plotW);
  const yFor = v => padding.top + plotH - ((v - minV) / yRange) * plotH;

  const ns = "http://www.w3.org/2000/svg";

  // gridlines + y labels (4 bands)
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

  // x-axis time labels (show at most ~10, evenly spaced)
  const labelStep = Math.max(1, Math.ceil(points.length / 10));
  points.forEach((p, i) => {
    if (i % labelStep !== 0 && i !== points.length - 1) return;
    const label = document.createElementNS(ns, "text");
    label.setAttribute("x", xFor(i));
    label.setAttribute("y", height - padding.bottom + 16);
    label.setAttribute("text-anchor", "middle");
    label.setAttribute("class", "axis-label");
    label.textContent = p.time;
    svg.appendChild(label);
  });

  function drawPath(key, color) {
    let d = "";
    let drawing = false;
    points.forEach((p, i) => {
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
  points.forEach((p, i) => {
    if (p.actual == null) return;
    const circle = document.createElementNS(ns, "circle");
    circle.setAttribute("cx", xFor(i));
    circle.setAttribute("cy", yFor(p.actual));
    circle.setAttribute("r", "2.5");
    circle.setAttribute("fill", "#16a34a");
    circle.appendChild(Object.assign(document.createElementNS(ns, "title"), { textContent: `${p.time}: actual ${p.actual}` }));
    svg.appendChild(circle);
  });

  // จุดค่าพยากรณ์: สีตามโมเดลที่ใช้ (LightGBM/XGBoost/Prophet), วงกลมกลวง = backfill, ทึบ = live
  points.forEach((p, i) => {
    const circle = document.createElementNS(ns, "circle");
    circle.setAttribute("cx", xFor(i));
    circle.setAttribute("cy", yFor(p.predicted));
    circle.setAttribute("r", "3");
    const color = modelHex(p.model_used);
    if (p.source === "backfill") {
      circle.setAttribute("fill", "none");
      circle.setAttribute("stroke", color);
      circle.setAttribute("stroke-width", "1.5");
    } else {
      circle.setAttribute("fill", color);
    }
    circle.appendChild(Object.assign(document.createElementNS(ns, "title"), {
      textContent: `${p.time}: predicted ${p.predicted} (${p.model_used}, ${p.source})`,
    }));
    svg.appendChild(circle);
  });
}

function barColorForMae(mae) {
  const q = qualityForMae(mae);
  if (q.cls === "good") return "var(--good)";
  if (q.cls === "moderate") return "var(--moderate)";
  return "var(--bad)";
}

async function loadOverview() {
  try {
    const res = await fetch("/api/comparison/overview");
    if (!res.ok) throw new Error(`Server error (${res.status})`);
    renderOverview(await res.json());
  } catch (err) {
    document.getElementById("overviewText").textContent = "โหลดภาพรวมความแม่นยำไม่สำเร็จ";
    console.error("โหลดภาพรวมความแม่นยำล้มเหลว:", err);
  }
}

function renderOverview(data) {
  const badge = document.getElementById("overviewBadge");
  const text = document.getElementById("overviewText");
  const note = document.getElementById("overviewNote");

  if (data.n_matched === 0) {
    badge.classList.add("hidden");
    note.classList.add("hidden");
    text.innerHTML = `ยังไม่มีข้อมูลค่าจริงมาเทียบเลยสักคู่ (มีการพยากรณ์ไว้ทั้งหมด ${data.n_total} รายการ) — เปิดหน้านี้ใหม่อีกครั้งหลังผ่านไปสักพัก`;
    renderOverviewBars(data.by_horizon);
    renderModelBreakdown(data.by_model || []);
    return;
  }

  const q = qualityForMae(data.overall_mae);
  badge.textContent = q.label;
  badge.className = "quality-badge quality-" + q.cls;
  badge.classList.remove("hidden");

  text.innerHTML =
    `จากข้อมูลที่เทียบได้ทั้งหมด <strong>${data.n_matched}</strong> คู่ (จาก ${data.n_total} ครั้งที่เคยพยากรณ์ไว้ ทุก horizon รวมกัน) ` +
    `ความคลาดเคลื่อนเฉลี่ยรวมอยู่ที่ <strong>±${data.overall_mae.toFixed(2)} µg/m³</strong> ` +
    biasPhrase(data.overall_bias) +
    (data.overall_basis === "mixed"
      ? ` <span class="basis-note">(ยังมีข้อมูล live ไม่พอ เลขนี้รวม backfill ชั่วคราวอยู่ n=${data.overall_n_used})</span>`
      : data.overall_basis === "live"
      ? ` <span class="basis-note">(อิงจาก live data ล้วนๆ n=${data.overall_n_used})</span>`
      : "");

  if (data.best_horizon && data.worst_horizon && data.best_horizon.horizon !== data.worst_horizon.horizon) {
    note.innerHTML =
      `แม่นยำที่สุดที่ <strong>t+${data.best_horizon.horizon}</strong> (±${data.best_horizon.mae}) ` +
      `คลาดเคลื่อนมากสุดที่ <strong>t+${data.worst_horizon.horizon}</strong> (±${data.worst_horizon.mae}) — ` +
      `เป็นไปตามธรรมชาติที่ยิ่งพยากรณ์ไกลยิ่งคลาดเคลื่อนมากขึ้น ไม่ได้แปลว่าโมเดลแย่ลง`;
    note.classList.remove("hidden");
  } else {
    note.classList.add("hidden");
  }

  renderOverviewBars(data.by_horizon);
  renderModelBreakdown(data.by_model || []);
}

function renderModelBreakdown(byModel) {
  const el = document.getElementById("modelBreakdown");
  el.innerHTML = "";
  if (byModel.length === 0) return;

  const modelLabel = { lightgbm: "LightGBM", xgboost: "XGBoost", prophet: "Prophet" };

  byModel.forEach(m => {
    const chip = document.createElement("div");
    chip.className = "model-chip";
    chip.style.borderLeftColor = modelHex(m.model_used);

    const name = document.createElement("div");
    name.className = "name";
    name.textContent = modelLabel[m.model_used] || m.model_used;

    const stat1 = document.createElement("div");
    stat1.className = "stat-line";
    stat1.textContent = `MAE ±${m.mae} µg/m³ (n=${m.n}${m.basis === "mixed" ? ", รวม backfill" : m.basis === "live" ? ", live ล้วน" : ""})`;

    const stat2 = document.createElement("div");
    stat2.className = "stat-line";
    stat2.textContent = `backfill ${m.n_backfill} / live ${m.n_live}`;

    chip.appendChild(name);
    chip.appendChild(stat1);
    chip.appendChild(stat2);
    el.appendChild(chip);
  });
}

function renderOverviewBars(byHorizon) {
  const el = document.getElementById("overviewBars");
  el.innerHTML = "";
  const maxMae = Math.max(...byHorizon.map(b => b.mae ?? 0), 1);

  byHorizon.forEach(b => {
    const col = document.createElement("div");
    col.className = "mini-bar-col";

    const track = document.createElement("div");
    track.className = "mini-bar-track";

    if (b.mae != null) {
      const bar = document.createElement("div");
      bar.className = "mini-bar";
      bar.style.height = `${(b.mae / maxMae) * 100}%`;
      bar.style.background = barColorForMae(b.mae);
      bar.title = `t+${b.horizon}: MAE ${b.mae} µg/m³ (n=${b.n})`;
      track.appendChild(bar);
    } else {
      const placeholder = document.createElement("div");
      placeholder.className = "mini-bar-placeholder";
      placeholder.title = `t+${b.horizon}: ยังไม่มีข้อมูล`;
      track.appendChild(placeholder);
    }

    const label = document.createElement("div");
    label.className = "mini-bar-label";
    label.textContent = (b.horizon % 4 === 1 || b.horizon === 24) ? b.horizon : "";

    col.appendChild(track);
    col.appendChild(label);
    el.appendChild(col);
  });
}

loadOverview();
loadComparison();

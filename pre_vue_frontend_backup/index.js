const API_URL = "/api/forecast";

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

function iconFor(status) {
  if (status === "ดีมาก") return "☀️";
  if (status === "ดี") return "🙂";
  if (status === "ปานกลาง") return "🌤️";
  if (status === "เริ่มมีผลกระทบต่อสุขภาพ") return "😷";
  return "☠️";
}

// threshold เดียวกับ PM25_LEVELS ใน app.py (15/25/37.5/75)
function barColor(pm25) {
  if (pm25 <= 15) return "var(--excellent)";
  if (pm25 <= 25) return "var(--good)";
  if (pm25 <= 37.5) return "var(--moderate)";
  if (pm25 <= 75) return "var(--unhealthy)";
  return "var(--bad)";
}

function modelColor(modelUsed) {
  if (modelUsed === "xgboost") return "var(--model-xgboost)";
  if (modelUsed === "prophet") return "var(--model-prophet)";
  return "var(--model-lightgbm)";
}

function showState(state) {
  document.getElementById("loadingState").classList.toggle("hidden", state !== "loading");
  document.getElementById("errorState").classList.toggle("hidden", state !== "error");
  document.getElementById("content").classList.toggle("hidden", state !== "content");
}

async function loadForecast() {
  showState("loading");
  try {
    const res = await fetch(API_URL);
    if (!res.ok) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.error || `Server error (${res.status})`);
    }
    const data = await res.json();
    renderForecast(data);
    showState("content");
  } catch (err) {
    document.getElementById("errorMessage").textContent =
      "ไม่สามารถโหลดข้อมูลได้ ลองใหม่อีกครั้ง";
    showState("error");
    console.error("โหลดข้อมูลพยากรณ์ล้มเหลว:", err);
  }
}

function renderForecast(data) {
  document.body.className = bodyClassFor(data.current.status);
  document.getElementById("weatherIcon").textContent = iconFor(data.current.status);

  document.getElementById("currentPm").textContent = data.current.pm25;
  document.getElementById("updatedAt").textContent = data.current.datetime_th;

  document.getElementById("biasWarning").classList.toggle("hidden", !data.high_pollution_risk);

  const prophetHorizons = data.prophet_triggered_horizons || [];
  document.getElementById("circuitBreakerNote").classList.toggle("hidden", prophetHorizons.length === 0);
  if (prophetHorizons.length > 0) {
    document.getElementById("prophetHorizonsText").textContent =
      prophetHorizons.map(h => `t+${h}`).join(", ");
  }

  const badge = document.getElementById("statusBadge");
  badge.textContent = data.current.status;
  badge.className = "status-badge " + statusClass(data.current.status);

  document.getElementById("bestTime").textContent = data.best_time.time;
  document.getElementById("bestPm").textContent = data.best_time.pm25;

  const dailySummary = data.daily_summary || {};
  document.getElementById("dailyAvgPm").textContent =
    dailySummary.daily_avg_pm25 !== null && dailySummary.daily_avg_pm25 !== undefined
      ? dailySummary.daily_avg_pm25 : "-";
  document.getElementById("dailySummaryText").textContent =
    dailySummary.explanation || "กำลังเตรียมข้อมูลสำหรับวันนี้";

  const maxPm = Math.max(...data.forecast.map(f => f.pm25), 37.5);
  const barsEl = document.getElementById("bars");
  barsEl.innerHTML = "";
  data.forecast.forEach(point => {
    const col = document.createElement("div");
    col.className = "bar-col";

    const value = document.createElement("div");
    value.className = "bar-value";
    value.textContent = point.pm25;

    const track = document.createElement("div");
    track.className = "bar-track";

    const bar = document.createElement("div");
    bar.className = "bar";
    bar.style.height = `${(point.pm25 / maxPm) * 100}%`;
    bar.style.background = barColor(point.pm25);
    bar.title = `${point.time}: ${point.pm25} µg/m³ (${point.model_used})`;

    const dot = document.createElement("div");
    dot.className = "model-dot";
    dot.style.background = modelColor(point.model_used);
    dot.title = point.model_used;

    const label = document.createElement("div");
    label.className = "bar-label";
    label.textContent = point.time;

    track.appendChild(bar);
    col.appendChild(value);
    col.appendChild(track);
    col.appendChild(dot);
    col.appendChild(label);
    barsEl.appendChild(col);
  });
}

loadForecast();

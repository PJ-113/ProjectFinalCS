"""
วัด latency จริงของ Gemini API (gemini-3.1-flash-lite) สำหรับ prompt แบบ NLG รายวัน
เรียก _gemini_client.models.generate_content(...) ตรงๆ โดยไม่ผ่าน timeout wrapper ของ _call_gemini()
เพื่อวัดการกระจาย (รวม tail) ของเวลาตอบจริง แล้วเสนอค่า GEMINI_TIMEOUT_SECONDS ที่เหมาะสม
"""
import importlib.util
import os
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_PATH = os.path.join(_ROOT, "app.py")
N_CALLS = 30
# เว้นจังหวะระหว่างคำขอ กัน rate limit (429)
SLEEP_BETWEEN_S = 4.5
OUT_CSV_PATH = os.path.join(_ROOT, "models", f"gemini_latency_measurement_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv")


def load_app():
    spec = importlib.util.spec_from_file_location("app_under_test", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


print("loading app.py...")
app = load_app()
assert app._gemini_client is not None, "Gemini client ไม่พร้อมใช้งาน (เช็ค GEMINI_KEY ใน .env)"

sys.path.insert(0, os.path.join(_ROOT, "scripts"))
import shap_analysis_clean as sac

test = sac._load_test_split()
valid = test.dropna(subset=["pm25_t+1"])
# สุ่มแถวจาก test + สลับ daily_avg_pm25/status หลายระดับ ให้ prompt หลากหลายใกล้เคียงของจริง
rng = np.random.default_rng(42)
sample_idx = rng.choice(len(valid), size=min(N_CALLS, len(valid)), replace=False)
daily_avg_choices = [5.0, 15.0, 30.0, 45.0, 60.0]

rows = []
print(f"เริ่มยิง {N_CALLS} ครั้ง เว้นจังหวะ {SLEEP_BETWEEN_S}s/ครั้ง (ไม่ผ่าน timeout wrapper)...")
for i, idx in enumerate(sample_idx):
    feature_row = valid[sac.FEATURE_COLS].iloc[[idx]]
    daily_avg = daily_avg_choices[i % len(daily_avg_choices)]
    status = app.pm25_status(daily_avg)
    model_used = app.MODEL_LGB if i % 2 == 0 else app.MODEL_XGB
    shap_pairs = app.get_shap_explanation(model_used, app.REP_HORIZON_FOR_SHAP, feature_row)
    prompt = app._build_daily_nlg_prompt(daily_avg, status, shap_pairs, "ทั่วไป")

    t0 = time.monotonic()
    error = None
    try:
        response = app._gemini_client.models.generate_content(
            model=app.GEMINI_MODEL,
            contents=prompt,
            config=app.genai_types.GenerateContentConfig(system_instruction=app._NLG_SYSTEM_PROMPT),
        )
        _ = (response.text or "").strip()
    except Exception as e:
        error = f"{type(e).__name__}: {e}"
    latency_s = time.monotonic() - t0

    rows.append({"call_index": i, "daily_avg_pm25": daily_avg, "model_used": model_used,
                 "latency_s": round(latency_s, 3), "error": error})
    status_str = "ERROR" if error else "ok"
    print(f"  [{i+1}/{len(sample_idx)}] latency={latency_s:.2f}s ({status_str})" + (f" {error}" if error else ""))

    if i < len(sample_idx) - 1:
        time.sleep(SLEEP_BETWEEN_S)

df = pd.DataFrame(rows)
df.to_csv(OUT_CSV_PATH, index=False, encoding="utf-8-sig")
print(f"\nบันทึกผลที่: {OUT_CSV_PATH}")

ok = df[df["error"].isna()]
print(f"\nสำเร็จ {len(ok)}/{len(df)} ครั้ง")
if len(ok) > 0:
    lat = ok["latency_s"]
    print(f"min={lat.min():.2f}s  mean={lat.mean():.2f}s  p50={lat.quantile(.50):.2f}s  "
          f"p90={lat.quantile(.90):.2f}s  p95={lat.quantile(.95):.2f}s  p99={lat.quantile(.99):.2f}s  max={lat.max():.2f}s")
    import math
    suggested = max(5, math.ceil(lat.quantile(.99)) + 1)
    print(f"\nข้อเสนอ GEMINI_TIMEOUT_SECONDS ใหม่ (ceil(p99)+1, ไม่ต่ำกว่า 5 เดิม) = {suggested}")
if (~df["error"].isna()).any():
    print("\nคำขอที่ error (ไม่ใช่ timeout เพราะไม่ได้ครอบ timeout — เป็น error อื่นของ SDK/เครือข่ายจริง):")
    print(df[~df["error"].isna()][["call_index", "error"]].to_string(index=False))

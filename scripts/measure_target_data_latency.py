"""
วัด "ความหน่วงปกติ" ของ reading ล่าสุดของ OpenAQ ต่อจังหวัด ณ ตอนนี้ (snapshot เดียวต่อการรัน) เพื่อตั้ง IMPUTATION_MIN_K ใน app.py
k = ชั่วโมงเต็มที่ reading ล่าสุดตามหลัง "ชั่วโมงที่เพิ่งจบ" (now-1h) — ตรงกับนิยามใน impute_target_series
รันหลายครั้งต่างเวลา/วัน (เช่น 3-5 ครั้ง) แล้วดูการกระจาย: MIN_K ควรสูงกว่า k ปกติของจังหวัดส่วนใหญ่ (เช่น >= p95) ไม่งั้นทุกคำขอจะเข้าสู่ขั้นเติมค่า
ต่อท้ายผลลงไฟล์ models/target_data_latency_log.csv (สะสมข้ามการรัน) ห้าม log คีย์/URL — ดึงผ่านฟังก์ชันของ app.py ซึ่งไม่พิมพ์ URL
รัน: python scripts/measure_target_data_latency.py
"""
import importlib.util
import os

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
spec = importlib.util.spec_from_file_location("app_mod", os.path.join(_ROOT, "app.py"))
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)

LOG = os.path.join(_ROOT, "models", "target_data_latency_log.csv")


def main():
    now = pd.Timestamp.now(tz="UTC")
    anchor = now.floor("h") - pd.Timedelta(hours=1)
    rows = []
    for p in app.PROVINCE_REGISTRY:
        slug = p["province_en"]
        try:
            offset = int(p.get("openaq_timestamp_offset_hours", 0)) if app.USE_TIMESTAMP_OFFSET else 0
            df = app.fetch_recent_pm25_openaq(p["sensor_id_pm25"], hours_back=48, offset_hours=offset)
            if df is None or df.empty:
                rows.append({"measured_utc": now, "province_en": slug, "latest_utc": None, "k": None})
                continue
            latest = df["datetime_utc"].max()
            rows.append({"measured_utc": now, "province_en": slug, "latest_utc": latest,
                         "k": int((anchor - latest) / pd.Timedelta(hours=1))})
        except Exception as e:
            rows.append({"measured_utc": now, "province_en": slug, "latest_utc": None, "k": None, "error": type(e).__name__})
    out = pd.DataFrame(rows)
    out.to_csv(LOG, mode="a", header=not os.path.exists(LOG), index=False)
    print(f"snapshot {now:%Y-%m-%d %H:%M} UTC (anchor = {anchor:%H:%M})")
    print(out[["province_en", "k"]].to_string(index=False))
    print("\nการกระจาย k:", out["k"].value_counts(dropna=False).sort_index().to_dict())
    allk = pd.read_csv(LOG)["k"].dropna()
    print(f"สะสมทั้งหมด {len(allk)} ค่า: p50={allk.quantile(.5):.0f} p90={allk.quantile(.9):.0f} p95={allk.quantile(.95):.0f} max={allk.max():.0f}")


if __name__ == "__main__":
    main()

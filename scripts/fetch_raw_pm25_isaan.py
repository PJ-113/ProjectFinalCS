"""
ดึง PM2.5 ดิบ (ไม่ interpolate/mask) ของ 20 จังหวัดอีสานจาก OpenAQ แล้วเก็บเป็นไฟล์ — ใช้เพื่อรู้ว่าชั่วโมงไหน
"วัดได้จริง" (observed mask) และการกระจายความยาวช่องว่างจริง (dataset เทรนเติมช่องว่าง <=48 ชม.ไปแล้ว แยกไม่ออก)

ใช้ fetch_pm25() ของ build_raw_dataset_multiprovince.py ช่วงเวลาเดียวกับตอนสร้าง dataset เทรน
ผลลัพธ์: data/raw_pm25_isaan_<ts>.csv (province_en, datetime_utc, pm25 เฉพาะชั่วโมงที่มีค่าจริง) + log ต่อจังหวัด
ตรวจ parity: ค่าดิบต้องตรงกับ pm25 ใน dataset เทรนล่าสุด (>=99%) ไม่งั้นหยุด
"""
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
OUT_PATH = os.path.join(_ROOT, "data", f"raw_pm25_isaan_{TIMESTAMP}.csv")
LOG_PATH = os.path.join(_ROOT, "data", f"raw_pm25_isaan_log_{TIMESTAMP}.csv")
DATASET_PATH = os.path.join(
    _ROOT, "data", "multiprovince_pm25_features_raw_spatial_k1_firehotspot_nbtrend_20260919_174805.csv"
)
PARITY_MIN = 0.99


def load_builder():
    spec = importlib.util.spec_from_file_location(
        "build_mp", os.path.join(_ROOT, "scripts", "build_raw_dataset_multiprovince.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    bm = load_builder()
    with open(bm.REGISTRY_PATH, encoding="utf-8") as f:
        registry = json.load(f)

    frames, log_rows = [], []
    for p in registry["provinces"]:
        offset = int(p.get("openaq_timestamp_offset_hours", 0))
        print(f"{p['province_en']} sensor={p['sensor_id_pm25']} offset={offset:+d}h ...", flush=True)
        try:
            df = bm.fetch_pm25(p["sensor_id_pm25"], offset)
        except Exception as e:  # ไม่ print URL/คีย์: แสดงเฉพาะชนิด error
            print(f"  !! ดึงไม่สำเร็จ: {type(e).__name__}")
            log_rows.append({"province_en": p["province_en"], "status": f"error:{type(e).__name__}", "hours": 0})
            continue
        df = df[(df["datetime_utc"] >= bm.WINDOW_START) & (df["datetime_utc"] < bm.WINDOW_END)]
        df.insert(0, "province_en", p["province_en"])
        frames.append(df[["province_en", "datetime_utc", "pm25"]])
        log_rows.append({"province_en": p["province_en"], "status": "ok", "hours": len(df),
                         "first": df["datetime_utc"].min(), "last": df["datetime_utc"].max()})
        print(f"  {len(df)} ชั่วโมงที่มีค่าจริง", flush=True)

    if not frames:
        sys.exit("ไม่มีจังหวัดไหนดึงสำเร็จเลย")
    raw = pd.concat(frames, ignore_index=True)

    # --- parity กับ dataset เทรน ---
    ds = pd.read_csv(DATASET_PATH, usecols=["province_en", "datetime_utc", "pm25"])
    ds["datetime_utc"] = pd.to_datetime(ds["datetime_utc"], utc=True)
    m = raw.merge(ds, on=["province_en", "datetime_utc"], how="inner", suffixes=("_raw", "_ds"))
    parity = {}
    for prov, g in m.groupby("province_en"):
        parity[prov] = float((np.abs(g["pm25_raw"] - g["pm25_ds"]) < 1e-6).mean())
    overall = float((np.abs(m["pm25_raw"] - m["pm25_ds"]) < 1e-6).mean())
    print(f"\nparity ค่าดิบ vs dataset เทรน (ชั่วโมงที่ซ้อนกัน {len(m)} แถว): {overall:.4f}")
    for prov, v in sorted(parity.items(), key=lambda kv: kv[1]):
        print(f"  {prov:18s} {v:.4f}")
    for row in log_rows:
        row["parity_vs_dataset"] = parity.get(row["province_en"])

    pd.DataFrame(log_rows).to_csv(LOG_PATH, index=False)
    if overall < PARITY_MIN or min(parity.values()) < PARITY_MIN:
        sys.exit(f"parity ต่ำกว่า {PARITY_MIN} — ตรวจป้ายเวลา/สถานีก่อนใช้ข้อมูลนี้ (ยังไม่บันทึกไฟล์ผลลัพธ์)")

    raw.sort_values(["province_en", "datetime_utc"]).to_csv(OUT_PATH, index=False)
    print(f"\nบันทึก {len(raw)} แถว -> {OUT_PATH}\nlog -> {LOG_PATH}")


if __name__ == "__main__":
    main()

"""
แปลงข้อมูล FIRMS hotspot events (จาก scripts/fetch_firms_hotspots_isaan.py, entry 123) เป็น
feature ต่อจังหวัด/ต่อชั่วโมง: จำนวนจุดความร้อนในรัศมี RADIUS_KM ย้อนหลัง WINDOW_HOURS ชั่วโมง —
ยังไม่ merge เข้า pipeline หลัก (build_raw_dataset_multiprovince.py) เพราะยังอยู่ขั้น validate
แค่สร้างไฟล์ทดสอบแยกไว้ก่อนตามธรรมเนียมโปรเจกต์ (เทียบ scripts/augment_raw_dataset_with_spatial_lag.py
ที่ทำแบบเดียวกันตอน validate spatial-lag feature ครั้งแรก, CHANGELOG 94)

RADIUS_KM=100 (ไกลกว่า 3km ที่ใช้กับ amenity_density มาก เพราะควันจากการเผาไร่ลอยไปได้ไกลกว่า
สิ่งปลูกสร้างเยอะ) WINDOW_HOURS=24 (ค่าเริ่มต้น เผื่อไว้ทดสอบ 48h เพิ่มถ้า 24h ไม่พอ) — ทั้งสองค่า
เป็น hyperparameter ที่ validate_fire_hotspot_feature.py จะทดสอบเทียบกันต่อ

วิธีคำนวณให้เร็ว (ข้อมูล hotspot อาจมีหลักแสนแถว x training row หลักแสนแถว ทำ nested loop ตรงๆ
ไม่ไหว): กรองด้วย haversine แบบ vectorized เอาเฉพาะ hotspot ในรัศมีของแต่ละจังหวัดก่อน (ทำครั้ง
เดียวต่อจังหวัด) แล้วใช้ np.searchsorted บน timestamp ที่ sort แล้วนับจำนวนในหน้าต่างเวลาแบบ
binary search (O(log n) ต่อแถว แทนที่จะวนลูปตรงๆ)
"""
import glob
import json
import os

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_ROOT, "data")
REGISTRY_PATH = os.path.join(DATA_DIR, "province_station_registry.json")
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

RADIUS_KM = 100
WINDOW_HOURS = 24
FEATURE_COL = f"fire_hotspot_count_{RADIUS_KM}km_{WINDOW_HOURS}h"


def latest_file(pattern):
    matches = sorted(glob.glob(os.path.join(DATA_DIR, pattern)))
    if not matches:
        raise FileNotFoundError(f"ไม่พบไฟล์ตรงกับ {pattern} ใน {DATA_DIR}")
    return matches[-1]


def haversine_km_vec(lat1, lon1, lat2_arr, lon2_arr):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2_arr)
    dphi = np.radians(lat2_arr - lat1)
    dlambda = np.radians(lon2_arr - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def compute_counts_for_province(province_lat, province_lon, hotspot_lat, hotspot_lon, hotspot_dt,
                                 target_timestamps, radius_km, window_hours):
    dist = haversine_km_vec(province_lat, province_lon, hotspot_lat, hotspot_lon)
    nearby_dt = np.sort(hotspot_dt[dist <= radius_km])
    if len(nearby_dt) == 0:
        return np.zeros(len(target_timestamps), dtype=int)
    window = np.timedelta64(window_hours, "h")
    idx_hi = np.searchsorted(nearby_dt, target_timestamps, side="right")
    idx_lo = np.searchsorted(nearby_dt, target_timestamps - window, side="right")
    return (idx_hi - idx_lo).astype(int), len(nearby_dt)


# override ได้จากภายนอก (scripts/rebuild_dataset_clean.py ส่ง dataset ที่แก้ป้ายเวลาแล้วเข้ามา)
K1_DATA_PATH = os.path.join(DATA_DIR, "multiprovince_pm25_features_raw_spatial_k1_20260917_001104.csv")
OUT_PATH = os.path.join(DATA_DIR, f"multiprovince_pm25_features_raw_spatial_k1_with_firehotspot_{TIMESTAMP}.csv")


def main():
    hotspot_path = latest_file("firms_hotspots_isaan_*.csv")
    print(f"โหลด hotspot events: {hotspot_path}")
    hotspots = pd.read_csv(hotspot_path)
    hotspots["acq_datetime_utc"] = pd.to_datetime(hotspots["acq_datetime_utc"], utc=True)
    print(f"hotspot events ทั้งหมด: {len(hotspots)} แถว ({hotspots['acq_datetime_utc'].min()} ถึง "
          f"{hotspots['acq_datetime_utc'].max()})")

    with open(REGISTRY_PATH, encoding="utf-8") as f:
        registry = json.load(f)
    provinces = registry["provinces"]

    k1_data_path = K1_DATA_PATH
    print(f"\nโหลด training dataset: {k1_data_path}")
    df = pd.read_csv(k1_data_path)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)

    hotspot_lat = hotspots["latitude"].values
    hotspot_lon = hotspots["longitude"].values
    hotspot_dt = hotspots["acq_datetime_utc"].values

    df[FEATURE_COL] = 0
    for p in provinces:
        province_en = p["province_en"]
        mask_rows = df["province_en"] == province_en
        if mask_rows.sum() == 0:
            continue
        target_ts = df.loc[mask_rows, "datetime_utc"].values
        counts, n_nearby = compute_counts_for_province(
            p["lat"], p["lon"], hotspot_lat, hotspot_lon, hotspot_dt, target_ts, RADIUS_KM, WINDOW_HOURS)
        df.loc[mask_rows, FEATURE_COL] = counts
        print(f"  {province_en}: hotspot ในรัศมี {RADIUS_KM}km = {n_nearby} จุด (ทั้งช่วงเวลา), "
              f"{FEATURE_COL} เฉลี่ย={counts.mean():.2f} max={counts.max()}")

    out_path = OUT_PATH
    df.to_csv(out_path, index=False)
    print(f"\nบันทึกที่: {out_path}")

    print("\n" + "=" * 90)
    print(f"ตรวจสอบความสมเหตุสมผล: {FEATURE_COL} เฉลี่ยต่อเดือน (ควรสูงช่วง ก.พ.-เม.ย.)")
    print("=" * 90)
    df["month"] = df["datetime_utc"].dt.month
    print(df.groupby("month")[FEATURE_COL].agg(["mean", "max", "count"]).to_string())

    print(f"\ncorrelation({FEATURE_COL}, pm25) รวมทุกจังหวัด (ตรวจสอบทิศทางคร่าวๆ ก่อน validate จริง):")
    print(f"  r = {df[FEATURE_COL].corr(df['pm25']):.3f}")


if __name__ == "__main__":
    main()

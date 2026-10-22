"""
เทียบข้อมูล PM2.5 ดิบจาก OpenAQ (sensor 1304249) กับข้อมูลที่ผ่าน interpolation แล้วใน
data/khonkaen_pm25_features_final.csv ช่วงเวลาเดียวกับ test set เพื่อแยกสมมติฐาน:
  (A) อากาศนิ่งจริงช่วงนั้น (raw ก็นิ่งพอๆ กับ clean)
  (B) interpolation ลบความผันผวนออกไป (raw ผันผวนกว่า clean มาก)
"""
import time

import pandas as pd
import requests

SENSOR_ID = 1304249  # pm25, location 7747 (Hydrogeological Group Water Resources Regional Office4 khonkaen)
TEST_START = "2025-05-16T04:00:00Z"
TEST_END = "2026-07-01T02:00:00Z"
import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(_ROOT, r"data\khonkaen_pm25_features_final.csv")

with open(os.path.join(_ROOT, r".env")) as f:
    OPENAQ_KEY = next(line.strip().split("=", 1)[1] for line in f if line.startswith("OPENAQ_KEY="))

HEADERS = {"X-API-Key": OPENAQ_KEY}


def fetch_month_chunk(dt_from, dt_to):
    params = {"datetime_from": dt_from.isoformat(), "datetime_to": dt_to.isoformat(), "limit": 1000}
    r = requests.get(f"https://api.openaq.org/v3/sensors/{SENSOR_ID}/measurements",
                      headers=HEADERS, params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("results", [])


def fetch_all_raw(start_str, end_str):
    start = pd.Timestamp(start_str)
    end = pd.Timestamp(end_str)
    all_rows = []
    cur = start
    while cur < end:
        chunk_end = min(cur + pd.Timedelta(days=30), end)
        rows = fetch_month_chunk(cur, chunk_end)
        all_rows.extend(rows)
        print(f"  fetched {cur.date()} -> {chunk_end.date()}: {len(rows)} rows")
        cur = chunk_end
        time.sleep(0.5)
    return all_rows


print("=" * 90)
print("ดึงข้อมูลดิบจาก OpenAQ (sensor 1304249) ช่วง test set...")
print("=" * 90)
raw_rows = fetch_all_raw(TEST_START, TEST_END)

raw_df = pd.DataFrame([{
    "datetime_utc": r["period"]["datetimeFrom"]["utc"],
    "pm25": r["value"],
} for r in raw_rows])
raw_df["datetime_utc"] = pd.to_datetime(raw_df["datetime_utc"])
raw_df = raw_df.dropna(subset=["pm25"]).drop_duplicates(subset=["datetime_utc"]).sort_values("datetime_utc").reset_index(drop=True)

print(f"\nrows ดึงมาได้ทั้งหมด (raw, ไม่ซ้ำ): {len(raw_df)}")
print(f"ช่วงเวลาจริงที่ได้: {raw_df['datetime_utc'].min()} -> {raw_df['datetime_utc'].max()}")

# ===== เทียบ max hourly change: raw vs clean(CSV) =====
raw_change = raw_df["pm25"].diff().abs()

csv_df = pd.read_csv(CSV_PATH)
csv_df["datetime_utc"] = pd.to_datetime(csv_df["datetime_utc"])
csv_df = csv_df.sort_values("datetime_utc").reset_index(drop=True)
split_idx = int(len(csv_df) * 0.8)
csv_test = csv_df.iloc[split_idx:].copy()
csv_change = csv_test["pm25"].diff().abs()

print("\n" + "=" * 90)
print("เทียบ hourly change: RAW (OpenAQ) vs CLEAN/INTERPOLATED (CSV) ช่วงเวลาเดียวกัน")
print("=" * 90)
print(f"{'':20} {'RAW (OpenAQ)':>15} {'CLEAN (CSV)':>15}")
print(f"{'n rows':20} {len(raw_df):>15} {len(csv_test):>15}")
print(f"{'mean |change|':20} {raw_change.mean():>15.3f} {csv_change.mean():>15.3f}")
print(f"{'max |change|':20} {raw_change.max():>15.3f} {csv_change.max():>15.3f}")
print(f"{'count >3':20} {(raw_change>3).sum():>15} {(csv_change>3).sum():>15}")
print(f"{'count >5':20} {(raw_change>5).sum():>15} {(csv_change>5).sum():>15}")
print(f"{'count >10':20} {(raw_change>10).sum():>15} {(csv_change>10).sum():>15}")

# ===== % ของช่วงเวลานี้ที่เป็นค่า interpolate (raw ไม่มีข้อมูลจริง) =====
expected_hours = pd.date_range(raw_df["datetime_utc"].min(), raw_df["datetime_utc"].max(), freq="h", tz="UTC")
n_expected = len(expected_hours)
n_actual_raw = raw_df["datetime_utc"].nunique()
n_missing = n_expected - n_actual_raw
pct_missing = n_missing / n_expected * 100

print("\n" + "=" * 90)
print("สัดส่วนชั่วโมงที่ไม่มีข้อมูลจริงจาก raw source (ต้องถูก interpolate ถ้าจะเติมให้ครบ)")
print("=" * 90)
print(f"ชั่วโมงที่ควรมีทั้งหมด (ถ้าครบทุกชม.): {n_expected}")
print(f"ชั่วโมงที่มีข้อมูลจริงจาก OpenAQ raw : {n_actual_raw}")
print(f"ชั่วโมงที่ขาด (ต้อง interpolate)     : {n_missing} ({pct_missing:.2f}%)")

print("\n" + "=" * 90)
print("SUMMARY (Task 1)")
print("=" * 90)
if raw_change.max() > csv_change.max() * 2:
    verdict = "B: interpolation ลบความผันผวนออกไปจริง"
else:
    verdict = "A: อากาศนิ่งจริงช่วงนี้ ไม่เกี่ยวกับ interpolation มากนัก"
print(f"max change: raw={raw_change.max():.1f} vs clean={csv_change.max():.1f}")
print(f"-> สรุปเบื้องต้น: {verdict}")
print(f"% ข้อมูลที่ขาดหาย (ต้อง interpolate): {pct_missing:.2f}%")

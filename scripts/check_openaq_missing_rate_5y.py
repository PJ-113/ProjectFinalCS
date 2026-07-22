"""
เช็ค missing rate ของข้อมูลดิบจาก OpenAQ (sensor 1304249) ตลอด 5 ปี (2021-04-29 ถึง
2026-07-02) แยกรายปี เพื่อดูว่า missing rate สูงคงที่ตลอด หรือแย่ลงเฉพาะช่วงหลัง
"""
import time

import pandas as pd
import requests

SENSOR_ID = 1304249
FULL_START = "2021-04-29T18:00:00Z"
FULL_END = "2026-07-02T00:00:00Z"

import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
with open(os.path.join(_ROOT, r".env")) as f:
    OPENAQ_KEY = next(line.strip().split("=", 1)[1] for line in f if line.startswith("OPENAQ_KEY="))
HEADERS = {"X-API-Key": OPENAQ_KEY}


def fetch_chunk(dt_from, dt_to):
    params = {"datetime_from": dt_from.isoformat(), "datetime_to": dt_to.isoformat(), "limit": 1000}
    r = requests.get(f"https://api.openaq.org/v3/sensors/{SENSOR_ID}/measurements",
                      headers=HEADERS, params=params, timeout=30)
    r.raise_for_status()
    return r.json().get("results", [])


def fetch_all(start_str, end_str):
    start = pd.Timestamp(start_str)
    end = pd.Timestamp(end_str)
    all_rows = []
    cur = start
    while cur < end:
        chunk_end = min(cur + pd.Timedelta(days=30), end)
        rows = fetch_chunk(cur, chunk_end)
        all_rows.extend(rows)
        cur = chunk_end
        time.sleep(0.4)
    return all_rows


print("ดึงข้อมูลดิบ OpenAQ ทั้ง 5 ปี (อาจใช้เวลาสักครู่)...")
raw_rows = fetch_all(FULL_START, FULL_END)

raw_df = pd.DataFrame([{
    "datetime_utc": r["period"]["datetimeFrom"]["utc"],
    "pm25": r["value"],
} for r in raw_rows])
raw_df["datetime_utc"] = pd.to_datetime(raw_df["datetime_utc"])
raw_df = raw_df.dropna(subset=["pm25"]).drop_duplicates(subset=["datetime_utc"]).sort_values("datetime_utc").reset_index(drop=True)

print(f"rows ดึงมาได้ทั้งหมด (raw, ไม่ซ้ำ): {len(raw_df)}")
print(f"ช่วงเวลาจริง: {raw_df['datetime_utc'].min()} -> {raw_df['datetime_utc'].max()}")

raw_df["year"] = raw_df["datetime_utc"].dt.year
raw_df["quarter"] = raw_df["datetime_utc"].dt.to_period("Q")

full_range = pd.date_range(raw_df["datetime_utc"].min(), raw_df["datetime_utc"].max(), freq="h", tz="UTC")
full_df = pd.DataFrame({"datetime_utc": full_range})
full_df["year"] = full_df["datetime_utc"].dt.year
full_df["quarter"] = full_df["datetime_utc"].dt.to_period("Q")

print("\n" + "=" * 70)
print("Missing rate รายปี")
print("=" * 70)
print(f"{'year':>6} {'expected_hrs':>13} {'actual_hrs':>11} {'missing_pct':>12}")
for year, grp in full_df.groupby("year"):
    expected = len(grp)
    actual = raw_df[raw_df["year"] == year].shape[0]
    missing_pct = (expected - actual) / expected * 100
    print(f"{year:>6} {expected:>13} {actual:>11} {missing_pct:>11.2f}%")

print("\n" + "=" * 70)
print("Missing rate รายไตรมาส")
print("=" * 70)
print(f"{'quarter':>10} {'expected_hrs':>13} {'actual_hrs':>11} {'missing_pct':>12}")
for q, grp in full_df.groupby("quarter"):
    expected = len(grp)
    actual = raw_df[raw_df["quarter"] == q].shape[0]
    missing_pct = (expected - actual) / expected * 100
    print(f"{str(q):>10} {expected:>13} {actual:>11} {missing_pct:>11.2f}%")

overall_expected = len(full_df)
overall_actual = len(raw_df)
print(f"\noverall missing rate (5y): {(overall_expected - overall_actual) / overall_expected * 100:.2f}%")

raw_df.to_csv(os.path.join(_ROOT, r"scripts\_openaq_raw_5y_cache.csv"), index=False)
print("cached raw data to scripts/_openaq_raw_5y_cache.csv (สำหรับใช้ต่อ ไม่ต้องดึงซ้ำ)")

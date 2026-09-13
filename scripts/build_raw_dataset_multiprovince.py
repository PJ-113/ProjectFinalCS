"""
Phase 1 (แผนขยาย 17+3 จังหวัด): สร้าง raw feature dataset หลายจังหวัด ต่อยอดจาก pattern
เดิมของ scripts/build_raw_dataset.py (สถานีเดียวขอนแก่น) แต่วนลูปทุกสถานีใน
data/province_station_registry.json (ต้องรัน scripts/province_registry.py ก่อน)

Pipeline ต่อจังหวัด (เหมือนเดิมทุกจุด แค่ทำซ้ำต่อสถานี):
1. ดึง pm25 รายชั่วโมงจาก OpenAQ v3 /v3/sensors/{id}/measurements ตรงๆ (ไม่ใช้ cache เดิม
   เพราะเป็นคนละสถานี) ช่วง 2024-01-01 ถึง 2026-07-02
2. ดึง weather รายชั่วโมงจาก Open-Meteo archive API ที่ lat/lon ของสถานีจริง (ไม่ใช่จุดค้นหา)
   รวม wind_direction_10m/surface_pressure ที่เพิ่มเข้ามาใหม่ (ของเดิมมีแค่ temp/humidity/
   wind_speed/precipitation)
3. gap <=48h -> interpolate, gap >48h -> mask ทิ้ง (สูตรเดียวกับสคริปต์เดิมเป๊ะ)
4. คำนวณ feature แบบ vectorized: weather(4) + cyclical เวลา(5) + pm25 lag/rolling(9) +
   wind_dir_sin/cos + pressure(3) + lat/lon(2) = 23 features, target 24 horizon
5. รวมทุกจังหวัดเป็น CSV เดียว, self-check ด้วย np.allclose แบบเดิม, log แถว/gap ต่อจังหวัด
"""
import functools
import json
import math
import os
import sys
import time

print = functools.partial(print, flush=True)

import numpy as np
import pandas as pd
import requests

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(_ROOT, ".env")
REGISTRY_PATH = os.path.join(_ROOT, "data", "province_station_registry.json")
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
OUT_PATH = os.path.join(_ROOT, "data", f"multiprovince_pm25_features_raw_{TIMESTAMP}.csv")
LOG_PATH = os.path.join(_ROOT, "data", f"multiprovince_build_log_{TIMESTAMP}.csv")

WINDOW_START = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
WINDOW_END = pd.Timestamp("2026-07-02 00:00:00", tz="UTC")
GAP_CUTOFF_HOURS = 48
REQUEST_SLEEP_S = 0.4

WEATHER_FEATURE_COLS = ["temp", "humidity", "wind_speed", "precipitation"]
TIME_FEATURE_COLS = ["hour_sin", "hour_cos", "month_sin", "month_cos", "dayofweek"]
PM25_FEATURE_COLS = [
    "pm25_lag_1", "pm25_lag_3", "pm25_lag_6", "pm25_lag_12", "pm25_lag_24",
    "pm25_roll_mean_6", "pm25_roll_std_6", "pm25_roll_mean_24", "pm25_roll_std_24",
]
ENV_FEATURE_COLS = ["wind_dir_sin", "wind_dir_cos", "pressure"]
STATION_FEATURE_COLS = ["lat", "lon"]
FEATURE_COLS = WEATHER_FEATURE_COLS + TIME_FEATURE_COLS + PM25_FEATURE_COLS + ENV_FEATURE_COLS + STATION_FEATURE_COLS
TARGET_COLS = [f"pm25_t+{h}" for h in range(1, 25)]
ID_COLS = ["province_th", "province_en", "is_zero_shot", "datetime_utc"]
assert len(FEATURE_COLS) == 23, f"expected 23 features, got {len(FEATURE_COLS)}"


def _load_env_key(name):
    with open(ENV_PATH, encoding="utf-8") as f:
        for line in f:
            if line.startswith(f"{name}="):
                return line.strip().split("=", 1)[1]
    raise RuntimeError(f"{name} not found in .env")


OPENAQ_KEY = _load_env_key("OPENAQ_KEY")
HEADERS = {"X-API-Key": OPENAQ_KEY}


def fetch_pm25(sensor_id):
    def fetch_chunk(dt_from, dt_to):
        params = {"datetime_from": dt_from.isoformat(), "datetime_to": dt_to.isoformat(), "limit": 1000}
        for attempt in range(6):
            try:
                r = requests.get(f"https://api.openaq.org/v3/sensors/{sensor_id}/measurements",
                                  headers=HEADERS, params=params, timeout=30)
            except requests.exceptions.RequestException as e:
                print(f"    !! network error (attempt {attempt+1}): {e}, retry ใน {2 ** attempt}s")
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                print(f"    !! HTTP {r.status_code} (attempt {attempt+1}) ที่ {dt_from.date()}..{dt_to.date()}, retry ใน {2 ** attempt}s")
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r.json().get("results", [])
        print(f"    !! ยอมแพ้ chunk {dt_from.date()}..{dt_to.date()} หลัง retry 6 ครั้ง ข้ามช่วงนี้ (จะเป็น gap ที่ interpolate/mask ทีหลัง)")
        return []

    all_rows = []
    cur = WINDOW_START
    while cur < WINDOW_END:
        chunk_end = min(cur + pd.Timedelta(days=30), WINDOW_END)
        all_rows.extend(fetch_chunk(cur, chunk_end))
        cur = chunk_end
        time.sleep(REQUEST_SLEEP_S)

    if not all_rows:
        return pd.DataFrame(columns=["datetime_utc", "pm25"])
    df = pd.DataFrame([{
        "datetime_utc": r["period"]["datetimeFrom"]["utc"],
        "pm25": r["value"],
    } for r in all_rows])
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df = df.dropna(subset=["pm25"]).drop_duplicates(subset=["datetime_utc"]).sort_values("datetime_utc").reset_index(drop=True)
    return df


def fetch_weather(lat, lon):
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": WINDOW_START.strftime("%Y-%m-%d"), "end_date": WINDOW_END.strftime("%Y-%m-%d"),
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,wind_direction_10m,surface_pressure",
        "timezone": "UTC",
    }
    for attempt in range(6):
        try:
            r = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=60)
        except requests.exceptions.RequestException as e:
            print(f"    !! network error ดึง weather (attempt {attempt+1}): {e}, retry ใน {2 ** attempt}s")
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 429 or r.status_code >= 500:
            print(f"    !! HTTP {r.status_code} ดึง weather (attempt {attempt+1}), retry ใน {2 ** attempt}s")
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        break
    else:
        raise RuntimeError("ดึง weather ไม่สำเร็จหลัง retry 6 ครั้ง")
    h = r.json()["hourly"]
    df = pd.DataFrame({
        "datetime_utc": pd.to_datetime(h["time"]).tz_localize("UTC"),
        "temp": h["temperature_2m"],
        "humidity": h["relative_humidity_2m"],
        "wind_speed": h["wind_speed_10m"],
        "precipitation": h["precipitation"],
        "wind_dir_deg": h["wind_direction_10m"],
        "pressure": h["surface_pressure"],
    })
    return df


def build_one_station(province, pm25_raw, weather):
    full_grid = pd.date_range(WINDOW_START, WINDOW_END, freq="h", tz="UTC")
    grid = pd.DataFrame({"datetime_utc": full_grid})
    grid = grid.merge(pm25_raw[["datetime_utc", "pm25"]], on="datetime_utc", how="left")
    grid = grid.merge(weather, on="datetime_utc", how="left")
    grid = grid.sort_values("datetime_utc").reset_index(drop=True)

    n_weather_missing = grid["temp"].isna().sum()
    if n_weather_missing:
        grid[["temp", "humidity", "wind_speed", "precipitation", "wind_dir_deg", "pressure"]] = \
            grid[["temp", "humidity", "wind_speed", "precipitation", "wind_dir_deg", "pressure"]].interpolate(
                method="linear", limit_direction="both")

    grid["pm25_missing"] = grid["pm25"].isna()
    grid["gap_group"] = (grid["pm25_missing"] != grid["pm25_missing"].shift()).cumsum()
    gap_run_len = grid[grid["pm25_missing"]].groupby("gap_group").size()
    long_gap_groups = gap_run_len[gap_run_len > GAP_CUTOFF_HOURS].index
    grid["is_long_gap"] = grid["pm25_missing"] & grid["gap_group"].isin(long_gap_groups)

    n_short_gap = int(grid["pm25_missing"].sum() - grid["is_long_gap"].sum())
    n_long_gap = int(grid["is_long_gap"].sum())

    grid["pm25_filled"] = grid["pm25"].interpolate(method="linear", limit_direction="both")
    if grid["pm25_filled"].isna().any():
        # ทั้งช่วงไม่มีข้อมูลเลยสักจุด (สถานีนี้ไม่มีข้อมูลในช่วงที่ต้องการเลย) -> คืน None ให้ caller
        return None, n_short_gap, n_long_gap, len(grid)

    grid["hour"] = grid["datetime_utc"].dt.hour
    grid["month"] = grid["datetime_utc"].dt.month
    grid["dayofweek"] = grid["datetime_utc"].dt.dayofweek
    grid["hour_sin"] = np.sin(2 * np.pi * grid["hour"] / 24)
    grid["hour_cos"] = np.cos(2 * np.pi * grid["hour"] / 24)
    grid["month_sin"] = np.sin(2 * np.pi * grid["month"] / 12)
    grid["month_cos"] = np.cos(2 * np.pi * grid["month"] / 12)
    grid["wind_dir_sin"] = np.sin(2 * np.pi * grid["wind_dir_deg"] / 360)
    grid["wind_dir_cos"] = np.cos(2 * np.pi * grid["wind_dir_deg"] / 360)

    s = grid["pm25_filled"]
    grid["pm25_lag_1"] = s.shift(1)
    grid["pm25_lag_3"] = s.shift(3)
    grid["pm25_lag_6"] = s.shift(6)
    grid["pm25_lag_12"] = s.shift(12)
    grid["pm25_lag_24"] = s.shift(24)
    grid["pm25_roll_mean_6"] = s.rolling(6).mean()
    grid["pm25_roll_std_6"] = s.rolling(6).std()
    grid["pm25_roll_mean_24"] = s.rolling(24).mean()
    grid["pm25_roll_std_24"] = s.rolling(24).std()

    long_gap_int = grid["is_long_gap"].astype(int)
    touches_long_gap = long_gap_int.rolling(25, min_periods=1).max().astype(bool)
    grid.loc[touches_long_gap, PM25_FEATURE_COLS] = np.nan

    pm25_by_time = grid.set_index("datetime_utc")["pm25_filled"]
    is_long_gap_by_time = grid.set_index("datetime_utc")["is_long_gap"]
    for h in range(1, 25):
        target_time = grid["datetime_utc"] + pd.Timedelta(hours=h)
        values = target_time.map(pm25_by_time)
        invalid = target_time.map(is_long_gap_by_time).fillna(True).astype(bool)
        grid[f"pm25_t+{h}"] = values.where(~invalid)

    grid["lat"] = province["lat"]
    grid["lon"] = province["lon"]
    grid["province_th"] = province["province_th"]
    grid["province_en"] = province["province_en"]
    grid["is_zero_shot"] = province["is_zero_shot"]

    before_drop = len(grid)
    grid_final = grid.dropna(subset=["pm25_lag_24", "pm25_roll_mean_24", "pm25_roll_std_24"]).copy()
    grid_final = grid_final.reset_index(drop=True)

    grid_final = grid_final.drop(columns=["pm25"]).rename(columns={"pm25_filled": "pm25"})
    out_cols = ID_COLS + ["pm25"] + FEATURE_COLS + TARGET_COLS
    grid_final = grid_final[out_cols]

    assert grid_final[FEATURE_COLS].isna().sum().sum() == 0, \
        f"{province['province_en']}: ยังมี NaN หลงเหลือใน feature columns"

    return grid_final, n_short_gap, n_long_gap, before_drop


def self_check(df):
    """ยืนยัน feature ที่คำนวณเองตรงกับสูตร lag/rolling ที่ควรจะเป็น สุ่มเช็คสัก 1 จังหวัด

    หมายเหตุ: แถวในไฟล์สุดท้ายถูกตัดทิ้งไปแล้ว (dropna แถวที่ lookback ไม่พอ/แตะ long-gap)
    ดังนั้นแถวติดกันใน df ไม่ได้แปลว่าติดกันจริงในเวลา (อาจมี gap ที่ถูกตัดออกคั่นกลาง) ต้อง
    กรองเฉพาะคู่แถวที่ datetime_utc ห่างกันพอดี 1 ชม.ก่อนเทียบ shift(1) ไม่งั้นจะ false-positive
    ตรง edge ของ gap ที่ถูกตัดออก"""
    sample_province = df["province_en"].iloc[0]
    sub = df[df["province_en"] == sample_province].sort_values("datetime_utc").reset_index(drop=True)
    if len(sub) < 30:
        return
    s = sub["pm25"]
    is_consecutive_hour = sub["datetime_utc"].diff() == pd.Timedelta(hours=1)
    lag1_check = is_consecutive_hour & sub["pm25_lag_1"].notna()
    n_mismatch = (~np.isclose(sub.loc[lag1_check, "pm25_lag_1"], s.shift(1)[lag1_check], rtol=1e-6)).sum()
    if n_mismatch:
        print(f"!! self-check ({sample_province}) เจอ pm25_lag_1 ไม่ตรงสูตร {n_mismatch}/{lag1_check.sum()} แถว (ที่ติดกันจริง 1 ชม.) ควรตรวจสอบเพิ่ม")
    else:
        print(f"self-check ({sample_province}) ผ่าน: pm25_lag_1 ตรงกับสูตรอ้างอิง ({lag1_check.sum()} แถวที่ติดกันจริง 1 ชม.)")


def main():
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        registry = json.load(f)

    all_frames = []
    log_rows = []

    for province in registry["provinces"]:
        tag = " [zero-shot]" if province["is_zero_shot"] else ""
        print(f"\n{province['province_th']} ({province['province_en']}){tag} "
              f"sensor={province['sensor_id_pm25']}")

        try:
            print("  ดึง pm25 จาก OpenAQ...")
            pm25_raw = fetch_pm25(province["sensor_id_pm25"])
            print(f"    {len(pm25_raw)} ชม.ที่มีข้อมูลจริง")

            if pm25_raw.empty:
                print("  !! ไม่มีข้อมูล pm25 เลยในช่วงที่ต้องการ ข้ามจังหวัดนี้")
                log_rows.append({"province_en": province["province_en"], "status": "no_data",
                                  "raw_hours": 0, "short_gap_hours": None, "long_gap_hours": None, "final_rows": 0})
                continue

            print("  ดึง weather จาก Open-Meteo archive...")
            weather = fetch_weather(province["lat"], province["lon"])
            time.sleep(REQUEST_SLEEP_S)

            result, n_short, n_long, n_before = build_one_station(province, pm25_raw, weather)
            if result is None:
                print("  !! interpolate ไม่สำเร็จ (มี gap ที่ขอบข้อมูลทั้งช่วง) ข้ามจังหวัดนี้")
                log_rows.append({"province_en": province["province_en"], "status": "interpolate_failed",
                                  "raw_hours": len(pm25_raw), "short_gap_hours": n_short, "long_gap_hours": n_long,
                                  "final_rows": 0})
                continue

            print(f"    gap <= {GAP_CUTOFF_HOURS}h (impute): {n_short}h, gap > {GAP_CUTOFF_HOURS}h (mask): {n_long}h")
            print(f"    แถวสุดท้ายหลังตัด lookback ไม่พอ: {n_before} -> {len(result)}")
            all_frames.append(result)
            log_rows.append({"province_en": province["province_en"], "status": "ok",
                              "raw_hours": len(pm25_raw), "short_gap_hours": n_short, "long_gap_hours": n_long,
                              "final_rows": len(result)})
        except Exception as e:
            print(f"  !! error ไม่คาดคิดระหว่างประมวลผลจังหวัดนี้: {e!r} — ข้ามจังหวัดนี้ ไปต่อจังหวัดถัดไป")
            log_rows.append({"province_en": province["province_en"], "status": f"error: {e!r}",
                              "raw_hours": None, "short_gap_hours": None, "long_gap_hours": None, "final_rows": 0})

    if not all_frames:
        raise RuntimeError("ไม่มีจังหวัดไหนดึงข้อมูลสำเร็จเลย")

    combined = pd.concat(all_frames, ignore_index=True)
    self_check(combined)

    combined.to_csv(OUT_PATH, index=False)
    pd.DataFrame(log_rows).to_csv(LOG_PATH, index=False)

    print(f"\n{'='*70}")
    print(f"บันทึก dataset แล้วที่: {OUT_PATH}")
    print(f"บันทึก log แล้วที่: {LOG_PATH}")
    print(f"จำนวนแถวรวมทั้งหมด: {len(combined)}")

    train_rows = combined[~combined["is_zero_shot"]]
    zeroshot_rows = combined[combined["is_zero_shot"]]
    print(f"แถวของ 17 จังหวัดเทรน: {len(train_rows)} (เป้าหมายอ้างอิงจากผู้ใช้: 323,347)")
    print(f"แถวของ 3 จังหวัด zero-shot: {len(zeroshot_rows)}")
    print("\nแถวต่อจังหวัด:")
    print(combined.groupby(["province_th", "is_zero_shot"]).size().to_string())


if __name__ == "__main__":
    main()

"""
สร้าง dataset จากข้อมูลดิบ (ไม่ผ่าน interpolation ที่ลบความผันผวนออก) สำหรับช่วง
2024-01-01 ถึง 2026-07-02

Pipeline:
1. โหลด pm25 ดิบจาก OpenAQ (cache เดิม, สถานีเดียวกับที่ใช้เทรนโมเดล)
2. ดึง weather ย้อนหลังจาก Open-Meteo archive API (reanalysis, ไม่มี gap)
3. gap <=48h -> interpolate (เก็บไว้ใช้เทรนได้), gap >48h -> mask ทิ้ง (ห้ามใช้)
4. คำนวณ feature (lag/rolling/cyclical) แบบ vectorized บน series ที่ interpolate เต็มแล้ว
   แล้ว mask (set NaN) ทุกจุดที่ lookback window แตะ >48h-gap หรือ target ตกอยู่ใน >48h-gap
   เพื่อไม่ให้มีการ "แต่งข้อมูล" ปนเข้ามาในค่าที่ใช้เทรนจริง
5. Save เป็น CSV ใหม่ (ไม่ทับไฟล์เดิม) + sanity check
"""
import pickle

import numpy as np
import pandas as pd
import requests

import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPENAQ_CACHE_PATH = os.path.join(_ROOT, r"scripts\_openaq_raw_5y_cache.csv")
XGB_MODEL_PATH_FOR_LOOKUP = os.path.join(_ROOT, r"models\xgb_pm25_multistep_20260712_015251.pkl")  # แค่ยืมพิกัดสถานี
STATION_LAT = 16.44538333
STATION_LON = 102.83525
WINDOW_START = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
WINDOW_END = pd.Timestamp("2026-07-02 00:00:00", tz="UTC")
GAP_CUTOFF_HOURS = 48
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
OUT_PATH = os.path.join(_ROOT, rf"data\khonkaen_pm25_features_raw_{TIMESTAMP}.csv")

FEATURE_COLS = [
    "temp", "humidity", "wind_speed", "precipitation",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "dayofweek",
    "pm25_lag_1", "pm25_lag_3", "pm25_lag_6", "pm25_lag_12", "pm25_lag_24",
    "pm25_roll_mean_6", "pm25_roll_std_6", "pm25_roll_mean_24", "pm25_roll_std_24",
]
TARGET_COLS = [f"pm25_t+{h}" for h in range(1, 25)]


def load_pm25():
    raw = pd.read_csv(OPENAQ_CACHE_PATH)
    raw["datetime_utc"] = pd.to_datetime(raw["datetime_utc"])
    raw = raw[(raw["datetime_utc"] >= WINDOW_START) & (raw["datetime_utc"] <= WINDOW_END)]
    raw = raw.drop_duplicates(subset=["datetime_utc"]).sort_values("datetime_utc").reset_index(drop=True)
    return raw


def fetch_weather():
    params = {
        "latitude": STATION_LAT, "longitude": STATION_LON,
        "start_date": WINDOW_START.strftime("%Y-%m-%d"), "end_date": WINDOW_END.strftime("%Y-%m-%d"),
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
        "timezone": "UTC",
    }
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=60)
    r.raise_for_status()
    h = r.json()["hourly"]
    df = pd.DataFrame({
        "datetime_utc": pd.to_datetime(h["time"]).tz_localize("UTC"),
        "temp": h["temperature_2m"],
        "humidity": h["relative_humidity_2m"],
        "wind_speed": h["wind_speed_10m"],
        "precipitation": h["precipitation"],
    })
    return df


print("โหลด pm25 ดิบจาก OpenAQ cache...")
pm25_raw = load_pm25()
print(f"  {len(pm25_raw)} ชม.ที่มีข้อมูลจริง ({pm25_raw['datetime_utc'].min()} -> {pm25_raw['datetime_utc'].max()})")

print("ดึง weather ย้อนหลังจาก Open-Meteo archive...")
weather = fetch_weather()
print(f"  {len(weather)} ชม.")

# ===== สร้าง grid เต็ม + merge =====
full_grid = pd.date_range(WINDOW_START, WINDOW_END, freq="h", tz="UTC")
grid = pd.DataFrame({"datetime_utc": full_grid})
grid = grid.merge(pm25_raw[["datetime_utc", "pm25"]], on="datetime_utc", how="left")
grid = grid.merge(weather, on="datetime_utc", how="left")
grid = grid.sort_values("datetime_utc").reset_index(drop=True)

assert grid["temp"].isna().sum() == 0, "weather มี gap ไม่คาดคิด (ควรครบทุกชม.)"

# ===== หา gap run ของ pm25 แล้วแบ่ง <=48h (impute) vs >48h (mask ทิ้ง) =====
grid["pm25_missing"] = grid["pm25"].isna()
grid["gap_group"] = (grid["pm25_missing"] != grid["pm25_missing"].shift()).cumsum()
gap_run_len = grid[grid["pm25_missing"]].groupby("gap_group").size()
long_gap_groups = gap_run_len[gap_run_len > GAP_CUTOFF_HOURS].index
grid["is_long_gap"] = grid["pm25_missing"] & grid["gap_group"].isin(long_gap_groups)

n_short_gap = grid["pm25_missing"].sum() - grid["is_long_gap"].sum()
n_long_gap = grid["is_long_gap"].sum()
print(f"\nชม.ที่ขาด <= {GAP_CUTOFF_HOURS}h (impute): {n_short_gap}")
print(f"ชม.ที่ขาด >  {GAP_CUTOFF_HOURS}h (mask ทิ้ง): {n_long_gap}")

# interpolate pm25 ทุก gap (รวม long gap ด้วย) แค่เพื่อให้ shift/rolling ทำงานได้ต่อเนื่อง
# ค่าที่ interpolate ข้าม long-gap จะถูก mask ทิ้งอีกทีด้านล่าง ไม่ถูกใช้เทรนจริง
grid["pm25_filled"] = grid["pm25"].interpolate(method="linear", limit_direction="both")
assert grid["pm25_filled"].isna().sum() == 0, "ยังมี NaN หลง interpolate (gap อยู่ที่ขอบข้อมูล?)"

# ===== feature เวลา (cyclical) =====
grid["hour"] = grid["datetime_utc"].dt.hour
grid["month"] = grid["datetime_utc"].dt.month
grid["dayofweek"] = grid["datetime_utc"].dt.dayofweek
grid["hour_sin"] = np.sin(2 * np.pi * grid["hour"] / 24)
grid["hour_cos"] = np.cos(2 * np.pi * grid["hour"] / 24)
grid["month_sin"] = np.sin(2 * np.pi * grid["month"] / 12)
grid["month_cos"] = np.cos(2 * np.pi * grid["month"] / 12)

# ===== lag / rolling (สูตรเดียวกับ build_feature_row()/build_feature_row_at()) =====
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

# ===== mask: แถวไหน lookback (0..24 ชม.ก่อนหน้า) แตะ long-gap ให้ feature เป็น NaN ทั้งหมด =====
long_gap_int = grid["is_long_gap"].astype(int)
touches_long_gap = long_gap_int.rolling(25, min_periods=1).max().astype(bool)  # รวมตัวเองถึงย้อนไป 24 ชม.
pm25_feature_cols = ["pm25_lag_1", "pm25_lag_3", "pm25_lag_6", "pm25_lag_12", "pm25_lag_24",
                     "pm25_roll_mean_6", "pm25_roll_std_6", "pm25_roll_mean_24", "pm25_roll_std_24"]
grid.loc[touches_long_gap, pm25_feature_cols] = np.nan

# ===== target: pm25_t+h ผ่าน timestamp lookup ตรงๆ (ไม่ใช้ row-shift) =====
pm25_by_time = grid.set_index("datetime_utc")["pm25_filled"]
is_long_gap_by_time = grid.set_index("datetime_utc")["is_long_gap"]
for h in range(1, 25):
    target_time = grid["datetime_utc"] + pd.Timedelta(hours=h)
    values = target_time.map(pm25_by_time)
    invalid = target_time.map(is_long_gap_by_time).fillna(True)  # นอกช่วงข้อมูล = ถือว่า invalid
    grid[f"pm25_t+{h}"] = values.where(~invalid)

# ===== ตัดแถวที่ feature หลักไม่ครบ (lag_24/roll_24 เป็น NaN) ทิ้ง =====
before_drop = len(grid)
grid_final = grid.dropna(subset=["pm25_lag_24", "pm25_roll_mean_24", "pm25_roll_std_24"]).copy()
grid_final = grid_final.reset_index(drop=True)
print(f"\nตัดแถวที่ feature ไม่ครบ (lookback ไม่พอ/แตะ long gap): {before_drop} -> {len(grid_final)}")

# drop คอลัมน์ "pm25" ดิบก่อน rename pm25_filled -> pm25 (ไม่ให้มี 2 คอลัมน์ชื่อซ้ำ)
grid_final = grid_final.drop(columns=["pm25"]).rename(columns={"pm25_filled": "pm25"})
out_cols = ["datetime_utc", "pm25"] + FEATURE_COLS + TARGET_COLS
grid_final = grid_final[out_cols]

# ===== sanity check: ทุก feature หลักต้องไม่มี NaN, target อนุญาตให้มี NaN บ้าง (ตาม horizon) =====
assert grid_final[FEATURE_COLS].isna().sum().sum() == 0, "ยังมี NaN หลงเหลือใน feature columns"

grid_final.to_csv(OUT_PATH, index=False)
print(f"\nบันทึกแล้วที่: {OUT_PATH}")
print(f"จำนวนแถวสุดท้าย: {len(grid_final)}")
print("\nจำนวนแถวที่ target ไม่เป็น NaN ต่อ horizon:")
for h in [1, 6, 12, 18, 24]:
    print(f"  t+{h:2d}: {grid_final[f'pm25_t+{h}'].notna().sum()}")

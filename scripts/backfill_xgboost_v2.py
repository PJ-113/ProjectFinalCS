"""
Backfill XGBoost เข้า predictions.db (เวอร์ชันใหม่) ย้อนหลัง เพื่อแก้ cold-start deadlock ของ
Dynamic Routing หลังจาก reset predictions.db พร้อมโมเดลชุดใหม่ที่เทรนบนข้อมูลดิบ (ดู
scripts/retrain_weighted_v2.py, scripts/train_xgb_prophet_v2.py)

Logic เดียวกับ backfill_xgboost.py เป๊ะ ต่างแค่:
1. DATA_PATH/XGB_MODEL_PATH ชี้ไปข้อมูล/โมเดลชุดใหม่
2. ข้อมูลดิบมี long-gap ที่ถูกตัดทิ้ง (ต่างจาก CSV เก่าที่ไม่มี gap เลย) จึงต้องกรอง eligible
   points ให้แน่ใจว่าหน้าต่าง [idx-24, idx+24] ไม่ตกขอบไฟล์ (index ไม่ต่อเนื่องเป็นชั่วโมงเป๊ะ
   เหมือนเดิม เพราะแถวที่ถูกตัดทิ้งหายไปจาก DataFrame เลย ไม่ใช่แค่ NaN) ใช้ diff ของ
   datetime_utc เช็คแทน (เหมือนเดิม แต่สำคัญกว่าเดิมเพราะมี gap จริง ไม่ใช่แค่ทฤษฎี)
3. เขียนทั้ง actual_pm25 และ actual_pm25_calibrated = actual_pm25 (identity) เพราะ actual มา
   จาก CSV ที่เป็นสเกล OpenAQ อยู่แล้ว (ดู migrate_calibrated_actuals() ใน app.py)
"""
import pickle
import random
import sqlite3
import sys

import numpy as np
import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")

import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(_ROOT, r"data\khonkaen_pm25_features_raw_20260712_191249.csv")
XGB_MODEL_PATH = os.path.join(_ROOT, r"models\xgb_pm25_multistep_20260712_191350.pkl")
DB_PATH = os.path.join(_ROOT, r"backups\predictions.db")  # sqlite เดิม ก่อน migrate ไป Postgres (ดู CHANGELOG (30))
N_POINTS = 30
RANDOM_SEED = 42

df = pd.read_csv(DATA_PATH)
df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
df = df.sort_values("datetime_utc").reset_index(drop=True)

with open(XGB_MODEL_PATH, "rb") as f:
    xgb_package = pickle.load(f)
XGB_MODELS = xgb_package["models"]
FEATURE_COLS = xgb_package["feature_cols"]

# train/test split เดียวกับตอนเทรน XGBoost (scripts/train_xgb_prophet_v2.py) — XGBoost ไม่เคย
# เห็น target ของแถวใน test set เลยตอนเทรน จึงเป็นการประเมินแบบ held-out ที่ยุติธรรม
split_idx = int(len(df) * 0.8)
test_start, test_end = split_idx, len(df) - 1
print(f"test set: index {test_start}..{test_end} ({df.loc[test_start, 'datetime_utc']} -> {df.loc[test_end, 'datetime_utc']})")

# หาช่วง gap ของข้อมูล (ชม.) เพื่อกรองเฉพาะจุดที่หน้าต่าง [idx-24, idx+24] ต่อเนื่องไม่มีรู
# ข้อมูลดิบมีแถวที่ถูกตัดทิ้งจริง (ไม่ใช่แค่ NaN) จึง diff ของ datetime_utc ระหว่างแถวติดกัน
# ต้องเท่ากับ 1 ชม.เป๊ะทุกคู่ในหน้าต่างนี้ ถ้าไม่ใช่แปลว่ามี gap ซ่อนอยู่ในช่วง lag/rolling/target
gaps = df["datetime_utc"].diff().dt.total_seconds().div(3600)

eligible = []
for idx in range(test_start, test_end + 1):
    if idx - 24 < 0 or idx + 24 > test_end:
        continue
    window_gaps = gaps.iloc[idx - 24: idx + 25]
    if (window_gaps.dropna() != 1.0).any():
        continue
    if df.loc[idx, [f"pm25_t+{h}" for h in range(1, 25)]].isna().any():
        continue
    eligible.append(idx)

print(f"eligible candidates (clean 48h window, no gaps): {len(eligible)} / {test_end - test_start + 1}")

random.seed(RANDOM_SEED)
chosen = sorted(random.sample(eligible, min(N_POINTS, len(eligible))))
chosen_times = df.loc[chosen, "datetime_utc"]
print(f"chosen {len(chosen)} points, spread: {chosen_times.min()} -> {chosen_times.max()}")


def build_feature_row_at(idx):
    """
    สร้าง feature row ที่จุดเวลา df.loc[idx] โดยใช้เฉพาะข้อมูล datetime_utc <= จุดนั้น
    (logic เดียวกับ build_feature_row() ใน app.py)
    """
    backtest_time = df.loc[idx, "datetime_utc"]

    hist = df.loc[:idx]
    assert hist["datetime_utc"].max() <= backtest_time, "leak: หลุดข้อมูลอนาคตเข้ามาใน history slice"
    pm25_series = hist.set_index("datetime_utc")["pm25"]

    row = df.loc[idx]
    features = {
        "temp": row["temp"],
        "humidity": row["humidity"],
        "wind_speed": row["wind_speed"],
        "precipitation": row["precipitation"],
        "hour_sin": row["hour_sin"],
        "hour_cos": row["hour_cos"],
        "month_sin": row["month_sin"],
        "month_cos": row["month_cos"],
        "dayofweek": row["dayofweek"],
        "pm25_lag_1": pm25_series.iloc[-2],
        "pm25_lag_3": pm25_series.iloc[-4],
        "pm25_lag_6": pm25_series.iloc[-7],
        "pm25_lag_12": pm25_series.iloc[-13],
        "pm25_lag_24": pm25_series.iloc[-25],
        "pm25_roll_mean_6": pm25_series.iloc[-6:].mean(),
        "pm25_roll_std_6": pm25_series.iloc[-6:].std(),
        "pm25_roll_mean_24": pm25_series.iloc[-24:].mean(),
        "pm25_roll_std_24": pm25_series.iloc[-24:].std(),
    }

    # assert ว่า timestamp ของข้อมูลดิบที่ใช้คำนวณ feature ทั้งหมด <= backtest_time จริง
    max_lookback_rows = 25
    used_slice = pm25_series.iloc[-max_lookback_rows:]
    assert used_slice.index.max() <= backtest_time, "leak: lag/rolling ใช้ข้อมูลหลัง backtest_time"

    X_now = pd.DataFrame([features])[FEATURE_COLS]

    # sanity check: เทียบกับ feature ที่ precompute ไว้แล้วในไฟล์ CSV ต้องตรงกันเป๊ะ
    precomputed = df.loc[idx, FEATURE_COLS].astype(float).values
    reconstructed = X_now.iloc[0].astype(float).values
    assert np.allclose(precomputed, reconstructed, atol=1e-6, equal_nan=True), \
        f"reconstructed feature mismatch precomputed CSV at idx={idx}"

    return X_now, backtest_time


def cross_check_target_alignment(idx, h):
    """เช็คว่า pm25_t+{h} ที่ precompute ไว้ตรงกับค่า pm25 จริงจาก timestamp lookup
    (ไม่ใช่ row-shift เพราะข้อมูลดิบมี gap ที่ตัดแถวทิ้งไปจริง แถว idx+h อาจไม่ใช่ h ชม.ถัดไป)"""
    target_col_value = df.loc[idx, f"pm25_t+{h}"]
    backtest_time = df.loc[idx, "datetime_utc"]
    target_time = backtest_time + pd.Timedelta(hours=h)
    pm25_by_time = df.set_index("datetime_utc")["pm25"]
    row_lookup_value = pm25_by_time.get(target_time)
    assert row_lookup_value is not None, f"target time {target_time} ไม่มีในข้อมูล (idx={idx}, h={h})"
    assert np.isclose(target_col_value, row_lookup_value, atol=1e-6), \
        f"target column misaligned with wall-clock at idx={idx}, h={h}"
    return target_col_value


log_rows = []
for idx in chosen:
    X_now, backtest_time = build_feature_row_at(idx)

    for h in range(1, 25):
        actual = cross_check_target_alignment(idx, h)
        pred = round(float(XGB_MODELS[h].predict(X_now)[0]), 1)
        target_time_utc = backtest_time + pd.Timedelta(hours=h)

        log_rows.append({
            "predicted_at_utc": backtest_time.isoformat(),
            "target_time_utc": target_time_utc.isoformat(),
            "horizon": h,
            "predicted_pm25": pred,
            "actual_pm25": round(float(actual), 1),
        })

print(f"\nbuilt {len(log_rows)} backfill rows ({len(chosen)} points x 24 horizons) — all asserts passed")

conn = sqlite3.connect(DB_PATH)
before_count = conn.execute("SELECT COUNT(*) FROM prediction_log").fetchone()[0]

# actual_pm25_calibrated = actual_pm25 ตรงๆ (identity) เพราะ actual มาจาก CSV เทรนซึ่งเป็นสเกล
# OpenAQ อยู่แล้ว ต่างจากแถว live ที่ต้องผ่าน a*x+b (ดู migrate_calibrated_actuals() ใน app.py)
conn.executemany(
    """INSERT OR IGNORE INTO prediction_log
       (predicted_at_utc, target_time_utc, horizon, predicted_pm25, actual_pm25,
        actual_pm25_calibrated, model_used, source)
       VALUES (?, ?, ?, ?, ?, ?, 'xgboost', 'backfill')""",
    [(r["predicted_at_utc"], r["target_time_utc"], r["horizon"], r["predicted_pm25"],
      r["actual_pm25"], r["actual_pm25"])
     for r in log_rows],
)
conn.commit()

after_count = conn.execute("SELECT COUNT(*) FROM prediction_log").fetchone()[0]
print(f"predictions.db row count: {before_count} -> {after_count} (+{after_count - before_count})")
conn.close()

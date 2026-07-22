"""
Backfill ข้อมูล xgboost เข้า predictions.db ย้อนหลัง เพื่อแก้ cold-start deadlock ของ
Dynamic Routing (select_model() ต้องการ >=5 คู่ matched ต่อ horizon ก่อนจะเทียบ R2 ได้จริง
แต่รอสะสมจาก live ใช้เวลาหลายวัน)

วิธีการ:
1. สุ่ม 30 จุดเวลาจาก test set ของ data/khonkaen_pm25_features_final.csv (แถวที่ XGBoost
   ไม่เคยเห็นตอนเทรน เพราะ train/test split แบบ time-based 80/20 เดียวกับตอนเทรนโมเดล)
   กระจายทั่ว test set ไม่ติดกันเป็นช่วงเดียว เพื่อลด autocorrelation bias
2. สำหรับแต่ละจุด สร้าง feature row จากข้อมูลที่ "รู้แล้ว ณ เวลานั้น" เท่านั้น (เหมือน
   build_feature_row() ตอน inference จริง) — assert ว่าไม่มีการดึงข้อมูลอนาคตมาคำนวณ lag/rolling
3. Predict ด้วย XGBoost ทุก horizon (1-24) แล้ว insert ลง predictions.db พร้อม actual_pm25 จริง
   ที่มีอยู่แล้วใน dataset (ไม่ต้องรอ backfill_actuals())
4. Mark source='backfill' แยกจาก source='live' ของเดิม
"""
import pickle
import random
import sqlite3

import numpy as np
import pandas as pd

import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(_ROOT, r"data\khonkaen_pm25_features_final.csv")
XGB_MODEL_PATH = os.path.join(_ROOT, r"models\xgb_pm25_multistep_20260712_015251.pkl")
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

# train/test split เดียวกับตอนเทรน XGBoost (scripts/train_xgb_prophet.py) — XGBoost ไม่เคย
# เห็น target ของแถวใน test set เลยตอนเทรน จึงเป็นการประเมินแบบ held-out ที่ยุติธรรม
split_idx = int(len(df) * 0.8)
test_start, test_end = split_idx, len(df) - 1
print(f"test set: index {test_start}..{test_end} ({df.loc[test_start, 'datetime_utc']} -> {df.loc[test_end, 'datetime_utc']})")

# หาช่วง gap ของข้อมูล (ชม.) เพื่อกรองเฉพาะจุดที่หน้าต่าง [idx-24, idx+24] ต่อเนื่องไม่มีรู
# (ถ้ามี gap อยู่ในช่วง lag/rolling หรือช่วง target ล่วงหน้า ค่าที่ได้จะไม่ตรงกับเวลาจริงที่ต้องการ)
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
    (logic เดียวกับ build_feature_row() ใน app.py หลังแก้ rolling-window bug)
    """
    backtest_time = df.loc[idx, "datetime_utc"]

    # ตัดเอาเฉพาะแถวที่ <= backtest_time เท่านั้น (จำลองว่า "ตอนนี้" รู้แค่อดีต เหมือน live inference)
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
    # (ไม่มี lag/rolling ไหนหลุดไปอ้างอิงแถวที่ index > idx)
    max_lookback_rows = 25  # pm25_lag_24 มองย้อนไป 25 แถว (iloc[-25])
    used_slice = pm25_series.iloc[-max_lookback_rows:]
    assert used_slice.index.max() <= backtest_time, "leak: lag/rolling ใช้ข้อมูลหลัง backtest_time"

    X_now = pd.DataFrame([features])[FEATURE_COLS]

    # sanity check เพิ่ม: เทียบกับ feature ที่ precompute ไว้แล้วในไฟล์ CSV ต้องตรงกันเป๊ะ
    # (ยืนยันว่า reconstruction ไม่มีบั๊ก เหมือนตอนแก้ rolling-window bug ก่อนหน้านี้)
    precomputed = df.loc[idx, FEATURE_COLS].astype(float).values
    reconstructed = X_now.iloc[0].astype(float).values
    assert np.allclose(precomputed, reconstructed, atol=1e-6, equal_nan=True), \
        f"reconstructed feature mismatch precomputed CSV at idx={idx}"

    return X_now, backtest_time


def cross_check_target_alignment(idx, h):
    """เช็คว่า pm25_t+{h} ที่ precompute ไว้ตรงกับค่า pm25 จริงที่แถว idx+h (ห่างกัน h ชม.จริง)"""
    target_col_value = df.loc[idx, f"pm25_t+{h}"]
    row_shift_value = df.loc[idx + h, "pm25"]
    assert np.isclose(target_col_value, row_shift_value, atol=1e-6), \
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

# actual_pm25_calibrated = actual_pm25 ตรงๆ (ไม่ผ่าน calibrate_pm25 ของ app.py) เพราะ actual
# ที่นี่มาจาก CSV เทรนซึ่งเป็นสเกล OpenAQ อยู่แล้ว ต่างจากแถว live ที่ backfill_actuals() เติม
# ค่าดิบ Air4Thai แล้วต้องแปลงด้วย a*x+b ก่อน — ดูคอมเมนต์ที่ migrate_calibrated_actuals() ใน app.py
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

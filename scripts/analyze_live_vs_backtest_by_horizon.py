"""
TASK 2: เทียบ live MAE กับ backtest MAE ของ LightGBM (weighted, deployed) แบบเทียบ
horizon ต่อ horizon เป๊ะๆ (ไม่ใช่เทียบข้ามช่วง 0.445-4.25 แบบกว้างๆ) ใช้ test_mae_by_horizon
ที่บันทึกไว้ในตัว pkl ของโมเดลที่ deploy อยู่จริงเป็น baseline (ค่าเดียวกับที่มาจาก
eval ตอนเทรน ตรงกับที่ขอให้ใช้ lgb_results_summary/eval CSV)
"""
import pickle
import sqlite3

import pandas as pd

import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(_ROOT, r"models\lgb_pm25_multistep_weighted_20260709_004955.pkl")
DB_PATH = os.path.join(_ROOT, r"backups\predictions.db")  # sqlite เดิม ก่อน migrate ไป Postgres (ดู CHANGELOG (30))

with open(MODEL_PATH, "rb") as f:
    model_package = pickle.load(f)
backtest_mae = model_package["test_mae_by_horizon"]

conn = sqlite3.connect(DB_PATH)
rows = conn.execute("""
    SELECT horizon, predicted_pm25, actual_pm25
    FROM prediction_log
    WHERE model_used='lightgbm' AND source='live' AND actual_pm25 IS NOT NULL
""").fetchall()
conn.close()

df = pd.DataFrame(rows, columns=["horizon", "predicted_pm25", "actual_pm25"])
df["error"] = (df["predicted_pm25"] - df["actual_pm25"]).abs()

g = df.groupby("horizon")["error"].agg(["count", "mean"]).reset_index()
g.columns = ["horizon", "n_live", "live_mae"]
g["backtest_mae"] = g["horizon"].map(backtest_mae)
g["ratio"] = g["live_mae"] / g["backtest_mae"]

print(g.to_string(index=False))
print()
print(f"avg ratio (mean across horizons): {g['ratio'].mean():.2f}")
print(f"overall live MAE (pooled): {df['error'].mean():.2f}")
print(f"overall backtest MAE (mean across horizons): {pd.Series(backtest_mae).mean():.3f}")
print()
print("horizons with highest ratio:")
print(g.sort_values('ratio', ascending=False).head(5).to_string(index=False))

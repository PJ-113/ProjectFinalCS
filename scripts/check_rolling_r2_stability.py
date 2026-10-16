"""
ตรวจสอบด้วยข้อมูลจริงว่า Rolling R2 (N<=24, ขั้นต่ำ 5 คู่) ไม่เสถียรเมื่อจำนวนตัวอย่างน้อยจริงหรือไม่
ใช้ prediction_log จริง (ทุก (model, horizon, province)) จำลองค่า R2 ที่ select_model() จะเห็น ณ แต่ละ
จุดเวลา (sliding window ท้ายสุด 24 คู่ ขั้นต่ำ 5) แล้วเทียบการกระจาย/ความผันผวนตามขนาด n
"""
import os

import numpy as np
import pandas as pd
import psycopg2
from sklearn.metrics import r2_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_env():
    env = {}
    with open(os.path.join(_ROOT, ".env"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


env = load_env()
conn = psycopg2.connect(host=env["POSTGRES_HOST"], port=env["POSTGRES_PORT"], dbname=env["POSTGRES_DB"],
                         user=env["POSTGRES_USER"], password=env["POSTGRES_PASSWORD"])
cur = conn.cursor()
cur.execute(
    """SELECT model_used, horizon, province, target_time_utc, predicted_pm25, actual_pm25
       FROM prediction_log
       WHERE model_used IN ('lightgbm_pooled_v3', 'xgboost_pooled_v3') AND actual_pm25 IS NOT NULL
       ORDER BY model_used, horizon, province, target_time_utc"""
)
rows = cur.fetchall()
cur.close()
conn.close()

df = pd.DataFrame(rows, columns=["model_used", "horizon", "province", "target_time_utc", "predicted", "actual"])
print(f"แถวทั้งหมด (v3, มี actual): {len(df)}")

records = []  # n, r2, group key, position
flip_records = []  # ต่อ (horizon, province, timestamp index): r2_xgb, r2_lgb, n_xgb, n_lgb

for (model, horizon, province), g in df.groupby(["model_used", "horizon", "province"]):
    g = g.sort_values("target_time_utc").reset_index(drop=True)
    pred = g["predicted"].to_numpy()
    actual = g["actual"].to_numpy()
    n_total = len(g)
    for i in range(4, n_total):  # i=4 -> 5 คู่ (index 0..4)
        lo = max(0, i - 23)
        window_pred = pred[lo:i + 1]
        window_actual = actual[lo:i + 1]
        n = len(window_pred)
        if np.allclose(window_actual, window_actual[0]):
            r2 = np.nan  # ไม่สามารถนิยาม R2 ได้ (variance=0) - สูตร sklearn จะคืนค่าพิเศษ/0
        else:
            r2 = r2_score(window_actual, window_pred)
        records.append({"model": model, "horizon": horizon, "province": province, "n": n, "r2": r2})

rec_df = pd.DataFrame(records)
rec_df["n_bucket"] = pd.cut(rec_df["n"], bins=[4, 9, 14, 19, 24], labels=["5-9", "10-14", "15-19", "20-24"])

print("\n" + "=" * 70)
print("กระจาย R2 ตามขนาด n (ยิ่ง n น้อย ควรเห็น std/range กว้างกว่าถ้าไม่เสถียรจริง)")
print("=" * 70)
summary = rec_df.groupby("n_bucket")["r2"].agg(["count", "mean", "std", "min", "max"])
print(summary)

print("\n" + "=" * 70)
print("สัดส่วนที่ R2 ติดลบรุนแรง (< -1) ตามขนาด n")
print("=" * 70)
rec_df["severe_negative"] = rec_df["r2"] < -1
print(rec_df.groupby("n_bucket")["severe_negative"].mean().round(3))

# ความผันผวนแบบ step-to-step: ภายในกลุ่มเดียวกัน (model,horizon,province) เรียงตามลำดับเวลา
# ดูขนาดการเปลี่ยนแปลงของ r2 จากจุดหนึ่งไปจุดถัดไป แยกตามว่า n ของจุดนั้นอยู่ช่วงไหน
print("\n" + "=" * 70)
print("ขนาดการเปลี่ยนแปลง |delta R2| ระหว่างจุดเวลาติดกัน แยกตาม n ของจุดปัจจุบัน")
print("=" * 70)
deltas = []
for (model, horizon, province), g in df.groupby(["model_used", "horizon", "province"]):
    g = g.sort_values("target_time_utc").reset_index(drop=True)
    pred = g["predicted"].to_numpy()
    actual = g["actual"].to_numpy()
    n_total = len(g)
    prev_r2 = None
    for i in range(4, n_total):
        lo = max(0, i - 23)
        window_pred = pred[lo:i + 1]
        window_actual = actual[lo:i + 1]
        n = len(window_pred)
        if np.allclose(window_actual, window_actual[0]):
            r2 = np.nan
        else:
            r2 = r2_score(window_actual, window_pred)
        if prev_r2 is not None and not np.isnan(r2) and not np.isnan(prev_r2):
            deltas.append({"n": n, "abs_delta": abs(r2 - prev_r2)})
        prev_r2 = r2

delta_df = pd.DataFrame(deltas)
delta_df["n_bucket"] = pd.cut(delta_df["n"], bins=[4, 9, 14, 19, 24], labels=["5-9", "10-14", "15-19", "20-24"])
print(delta_df.groupby("n_bucket")["abs_delta"].agg(["count", "mean", "median", "std"]))

# เทียบ "ผู้ชนะ" (LGB vs XGB) ที่ select_model() จะเลือก ถ้าคำนวณด้วย n เต็ม (<=24) เทียบกับ
# ถ้าหยุดคำนวณที่ n=5 (ขั้นต่ำสุดที่ระบบยอมรับ) ว่าผลตรงกันแค่ไหน
print("\n" + "=" * 70)
print("อัตราที่ผลการเลือกโมเดล (LGB vs XGB) จาก n=5 ไม่ตรงกับผลจาก n=24 (ที่จุดเวลาเดียวกัน)")
print("=" * 70)
mismatch = 0
total_compared = 0
for (horizon, province), g_all in df.groupby(["horizon", "province"]):
    for model in ["lightgbm_pooled_v3", "xgboost_pooled_v3"]:
        pass
    g_xgb = g_all[g_all.model_used == "xgboost_pooled_v3"].sort_values("target_time_utc").reset_index(drop=True)
    g_lgb = g_all[g_all.model_used == "lightgbm_pooled_v3"].sort_values("target_time_utc").reset_index(drop=True)
    n_common = min(len(g_xgb), len(g_lgb))
    if n_common < 24:
        continue
    for i in range(23, n_common):
        xw_pred = g_xgb["predicted"].to_numpy()[i - 23:i + 1]
        xw_act = g_xgb["actual"].to_numpy()[i - 23:i + 1]
        lw_pred = g_lgb["predicted"].to_numpy()[i - 23:i + 1]
        lw_act = g_lgb["actual"].to_numpy()[i - 23:i + 1]
        if np.allclose(xw_act, xw_act[0]) or np.allclose(lw_act, lw_act[0]):
            continue
        r2_x_full = r2_score(xw_act, xw_pred)
        r2_l_full = r2_score(lw_act, lw_pred)
        r2_x_5 = r2_score(xw_act[-5:], xw_pred[-5:]) if not np.allclose(xw_act[-5:], xw_act[-5]) else None
        r2_l_5 = r2_score(lw_act[-5:], lw_pred[-5:]) if not np.allclose(lw_act[-5:], lw_act[-5]) else None
        if r2_x_5 is None or r2_l_5 is None:
            continue
        winner_full = "xgb" if r2_x_full > r2_l_full else "lgb"
        winner_5 = "xgb" if r2_x_5 > r2_l_5 else "lgb"
        total_compared += 1
        if winner_full != winner_5:
            mismatch += 1

if total_compared:
    print(f"เทียบได้ {total_compared} จุดเวลา, ไม่ตรงกัน {mismatch} ครั้ง ({mismatch/total_compared*100:.1f}%)")
else:
    print("ไม่มีจุดเวลาที่เทียบได้ (ข้อมูลยังไม่ถึง 24 คู่ในบางกลุ่ม)")

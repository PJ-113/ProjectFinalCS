"""
วิเคราะห์ว่า LightGBM (weighted model, ที่ deploy อยู่จริง) พลาดหนักตอน PM2.5
เปลี่ยนแปลงฉับพลันหรือไม่ โดยใช้ backtest จาก test set เดิม (ไม่ต้องรอ live data)

สำคัญ: พบว่า test set (แถวสุดท้าย 20% ตามเวลา) มี shock event น้อยมาก (|change|>3 มีแค่ 2
จุด, >5 มีแค่ 1 จุด จาก 7582 แถว) ขณะที่ train set มีเยอะกว่ามาก (106/108 ของทั้งหมดที่
threshold>3) การเปรียบเทียบ MAE แบบ shock-vs-normal ที่ขอมาจึงทำบน test set อย่างเดียว
ไม่ได้อย่างมีนัยสำคัญทางสถิติ — สคริปต์นี้เลยแบ่งเป็น 2 ส่วน: (1) test set จริง (n น้อย,
รายงานตรงๆ ว่าน้อยแค่ไหน) (2) train set แบบ supplementary (กำกับชัดว่าเป็น in-sample,
ไม่ใช่ held-out validation, มีแนวโน้ม underestimate ความคลาดเคลื่อนจริงเพราะโมเดลเคยเห็น
ข้อมูลนี้ตอนเทรนแล้ว) ส่วน precipitation/Scavenging Effect ใช้ full dataset ได้เพราะเป็น
คำถามเกี่ยวกับข้อมูลสภาพอากาศ ไม่ใช่ performance ของโมเดล จึงไม่มีปัญหาเรื่อง train/test leakage
"""
import pickle

import numpy as np
import pandas as pd

import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(_ROOT, r"data\khonkaen_pm25_features_final.csv")
MODEL_PATH = os.path.join(_ROOT, r"models\lgb_pm25_multistep_weighted_20260709_004955.pkl")
HORIZONS = [1, 2, 3, 4]
THRESHOLD = 3  # ปรับลงจาก 10/7 ที่ขอมา เพราะ test set ไม่มีจุดเข้าเกณฑ์เลยแม้ที่ 7

df = pd.read_csv(DATA_PATH)
df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
df = df.sort_values("datetime_utc").reset_index(drop=True)
df["pm25_change_1h"] = df["pm25"] - df["pm25_lag_1"]

with open(MODEL_PATH, "rb") as f:
    model_package = pickle.load(f)

MODELS = model_package["models"]
FEATURE_COLS = model_package["feature_cols"]
BASELINE_MAE = model_package["test_mae_by_horizon"]

split_idx = int(len(df) * 0.8)
train = df.iloc[:split_idx].copy()
test = df.iloc[split_idx:].copy()

print("=" * 100)
print("STEP 1-2: หา shock event ใน test set ตามที่ขอ (threshold 10 -> 7 -> ต่ำกว่านั้น)")
print("=" * 100)
for t in [10, 7, 5, 3, 2]:
    n = (test["pm25_change_1h"].abs() > t).sum()
    print(f"  test set, threshold > {t}: พบ {n} จุด")

n_test_shock = (test["pm25_change_1h"].abs() > THRESHOLD).sum()
n_train_shock = (train["pm25_change_1h"].abs() > THRESHOLD).sum()
print(f"\n>>> ที่ threshold > {THRESHOLD}: test set มีแค่ {n_test_shock} จุด, train set มี {n_train_shock} จุด")
print(">>> สรุป: test set (แถวสุดท้าย 20% ตามเวลา, พ.ค.2025-ก.ค.2026) เป็นช่วงที่ PM2.5 นิ่งผิดปกติ")
print(">>> เทียบกับ train set — ไม่พอจะสรุปทางสถิติจาก test set อย่างเดียวได้ ดังนั้นแบ่งวิเคราะห์เป็น 2 ส่วน\n")


def compute_mae_for_mask(data, mask, horizon):
    idx = data.index[mask]
    if len(idx) == 0:
        return None, 0
    X = data.loc[idx, FEATURE_COLS]
    y_true = data.loc[idx, f"pm25_t+{horizon}"]
    y_pred = MODELS[horizon].predict(X)
    errors = np.abs(y_true.values - y_pred)
    return float(errors.mean()), len(idx)


def run_mae_comparison(data, label):
    change = data["pm25_change_1h"]
    shock_mask = change.abs() > THRESHOLD
    drop_mask = shock_mask & (change < 0)
    surge_mask = shock_mask & (change > 0)

    print(f"--- {label}: shock={shock_mask.sum()} (ร่วง={drop_mask.sum()}, พุ่ง={surge_mask.sum()}) ---")
    print(f"{'horizon':>7} {'baseline':>10} {'shock_mae':>10} {'x':>6} {'drop_mae':>10} {'x':>6} {'surge_mae':>10} {'x':>6}")

    rows = []
    for h in HORIZONS:
        baseline = BASELINE_MAE[h]
        shock_mae, _ = compute_mae_for_mask(data, shock_mask, h)
        drop_mae, _ = compute_mae_for_mask(data, drop_mask, h)
        surge_mae, _ = compute_mae_for_mask(data, surge_mask, h)

        r_all = shock_mae / baseline if shock_mae else None
        r_drop = drop_mae / baseline if drop_mae else None
        r_surge = surge_mae / baseline if surge_mae else None
        rows.append({"horizon": h, "shock_mae": shock_mae, "r_all": r_all, "drop_mae": drop_mae,
                      "r_drop": r_drop, "surge_mae": surge_mae, "r_surge": r_surge})

        def fmt(v, r):
            return (f"{v:>10.3f} {r:>5.1f}x" if v is not None else f"{'--':>10} {'--':>6}")

        print(f"{h:>7} {baseline:>10.3f} " + fmt(shock_mae, r_all) + " " + fmt(drop_mae, r_drop) + " " + fmt(surge_mae, r_surge))

    return rows, shock_mask, drop_mask, surge_mask


print("=" * 100)
print("STEP 3-4: MAE shock vs baseline, แยกร่วง/พุ่ง")
print("=" * 100)
print("\n[A] TEST SET จริง (held-out, ถูกต้องตามหลักวิชาการ แต่ n น้อยมาก — ตีความอย่างระวัง)")
test_rows, test_shock_mask, test_drop_mask, test_surge_mask = run_mae_comparison(test, "TEST")

print("\n[B] TRAIN SET (supplementary เท่านั้น — โมเดลเคยเห็นข้อมูลนี้ตอนเทรน ไม่ใช่ held-out")
print("    ผลจากส่วนนี้มีแนวโน้ม UNDERESTIMATE ความคลาดเคลื่อนจริง เพราะโมเดล fit ข้อมูลนี้ไปแล้ว)")
train_rows, train_shock_mask, train_drop_mask, train_surge_mask = run_mae_comparison(train, "TRAIN")

print("\n" + "=" * 100)
print("STEP 5: Precipitation vs Scavenging Effect (ใช้ full dataset — ไม่ติดปัญหา train/test leakage)")
print("=" * 100)

full_change = df["pm25_change_1h"]
full_shock_mask = full_change.abs() > THRESHOLD
full_drop_mask = full_shock_mask & (full_change < 0)
full_surge_mask = full_shock_mask & (full_change > 0)

baseline_precip_mean = df["precipitation"].mean()
baseline_precip_wet_pct = (df["precipitation"] > 0).mean() * 100
drop_precip_mean = df.loc[full_drop_mask, "precipitation"].mean()
drop_precip_wet_pct = (df.loc[full_drop_mask, "precipitation"] > 0).mean() * 100
surge_precip_mean = df.loc[full_surge_mask, "precipitation"].mean()
surge_precip_wet_pct = (df.loc[full_surge_mask, "precipitation"] > 0).mean() * 100

print(f"ทั้ง dataset (n={len(df)})   : precip เฉลี่ย {baseline_precip_mean:.4f} mm | มีฝนตก {baseline_precip_wet_pct:.2f}%")
print(f"จุดร่วงฮวบ (n={full_drop_mask.sum()})     : precip เฉลี่ย {drop_precip_mean:.4f} mm | มีฝนตก {drop_precip_wet_pct:.2f}%")
print(f"จุดพุ่งฮวบ (n={full_surge_mask.sum()})     : precip เฉลี่ย {surge_precip_mean:.4f} mm | มีฝนตก {surge_precip_wet_pct:.2f}%")

ratio = drop_precip_wet_pct / baseline_precip_wet_pct if baseline_precip_wet_pct > 0 else float("nan")
print(f"\n>>> โอกาสมีฝนตกตอนร่วงฮวบ สูงกว่าค่าเฉลี่ยทั่วไป {ratio:.1f} เท่า")

print("\n" + "=" * 100)
print("SUMMARY")
print("=" * 100)
print(f"1) test set มี shock event (>{THRESHOLD}) แค่ {n_test_shock} จุด เทียบ train set {n_train_shock} จุด")
print(f"   -> test set (ช่วงเวลาที่ใช้ประเมินโมเดลทั้งหมดที่ผ่านมา รวม MAE=2.729 ที่รายงานไปก่อนหน้า)")
print(f"   เป็นช่วงที่ PM2.5 นิ่งผิดปกติเทียบกับข้อมูลทั้งหมด — backtest MAE ที่เคยรายงานจึงมีโอกาส")
print(f"   ประเมินความแม่นยำโมเดล 'ดีเกินจริง' เพราะแทบไม่เคยเจอสถานการณ์ผันผวนเลยตอนทดสอบ")
if test_rows[0]["r_all"]:
    avg_r_test = np.mean([r["r_all"] for r in test_rows if r["r_all"]])
    print(f"2) [held-out, n={n_test_shock} เท่านั้น] MAE ตอน shock สูงกว่าปกติเฉลี่ย {avg_r_test:.1f} เท่า (ตัวเลขนี้ไม่นิ่งเพราะ n น้อยมาก)")
avg_r_train = np.mean([r["r_all"] for r in train_rows if r["r_all"]])
avg_r_train_drop = np.mean([r["r_drop"] for r in train_rows if r["r_drop"]])
avg_r_train_surge = np.mean([r["r_surge"] for r in train_rows if r["r_surge"]])
print(f"3) [supplementary, in-sample, n={n_train_shock}] MAE ตอน shock สูงกว่าปกติเฉลี่ย {avg_r_train:.1f} เท่า")
print(f"   แยกทิศทาง: ร่วง {avg_r_train_drop:.1f}x, พุ่ง {avg_r_train_surge:.1f}x")
print(f"4) Scavenging Effect: จุดร่วงฮวบมีโอกาสฝนตกสูงกว่าค่าเฉลี่ยทั่วไป {ratio:.1f} เท่า")

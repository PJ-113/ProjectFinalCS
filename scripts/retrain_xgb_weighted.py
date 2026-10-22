"""
เทรน weighted XGBoost ด้วย sample weighting (แก้ bias ประเมินค่าต่ำในช่วงมลพิษสูง)
feature_cols/target_cols/station_id โหลดจาก pkl ของ XGBoost ที่ deploy อยู่ตอนนี้
baseline (unweighted) คำนวณสดจากโมเดล deploy จริง บน test set เดียวกับที่ใช้ evaluate โมเดลใหม่
"""
import pickle
import sys
from datetime import datetime as dt

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(_ROOT, r"data\khonkaen_pm25_features_raw_20260712_191249.csv")
DEPLOYED_XGB_PKL_PATH = os.path.join(_ROOT, r"models\xgb_pm25_multistep_20260712_191350.pkl")
MODEL_DIR = os.path.join(_ROOT, r"models")
HIGH_THRESHOLD = 37.5
WEIGHT_GRID = [1.0, 3.0, 5.0, 10.0]
# horizon ตัวแทนที่ SHAP feature-importance ของ LightGBM กับ XGBoost ต่างกันมากสุด
# (วัดจาก scripts/shap_divergence_by_horizon.py)
REP_HORIZONS = [1, 21, 24]

with open(DEPLOYED_XGB_PKL_PATH, "rb") as f:
    deployed_pkg = pickle.load(f)
feature_cols = deployed_pkg["feature_cols"]
target_cols = deployed_pkg["target_cols"]
station_id = deployed_pkg["station_id"]
deployed_models = deployed_pkg["models"]

df_final = pd.read_csv(DATA_PATH)
df_final["datetime_utc"] = pd.to_datetime(df_final["datetime_utc"])

split_idx = int(len(df_final) * 0.8)
train = df_final.iloc[:split_idx]
test = df_final.iloc[split_idx:]

# nested 85/15 split ของ train สำหรับหา weight (ไม่แตะ test set จริง)
tune_split_idx = int(len(train) * 0.85)
train_tune = train.iloc[:tune_split_idx]
valid_tune = train.iloc[tune_split_idx:]

print(f"train     : {len(train)} แถว ({train['datetime_utc'].min()} -> {train['datetime_utc'].max()})")
print(f"test      : {len(test)} แถว ({test['datetime_utc'].min()} -> {test['datetime_utc'].max()})")
print(f"train_tune: {len(train_tune)} แถว, valid_tune: {len(valid_tune)} แถว\n")


def evaluate(y_true, y_pred, threshold=HIGH_THRESHOLD):
    mae_all = mean_absolute_error(y_true, y_pred)
    mask_high = y_true > threshold
    n_high = int(mask_high.sum())
    if n_high > 1:
        mae_high = mean_absolute_error(y_true[mask_high], y_pred[mask_high])
        bias_high = float((y_true[mask_high] - y_pred[mask_high]).mean())
    else:
        mae_high = bias_high = float("nan")
    return mae_all, mae_high, bias_high, n_high


print("=== Step 1: เลือก weight ที่เหมาะสมด้วย train_tune/valid_tune (ไม่แตะ test set จริง) ===\n")
print(f"{'horizon':>7} {'weight':>7} {'MAE_all':>9} {'MAE_high':>9} {'bias_high':>10}")

grid_results = []
for h in REP_HORIZONS:
    target = f"pm25_t+{h}"
    tt_valid = train_tune.dropna(subset=[target])
    vt_valid = valid_tune.dropna(subset=[target])
    X_tt, y_tt = tt_valid[feature_cols], tt_valid[target]
    X_vt, y_vt = vt_valid[feature_cols], vt_valid[target]

    for w in WEIGHT_GRID:
        sample_weight = np.where(y_tt > HIGH_THRESHOLD, w, 1.0)
        model = xgb.XGBRegressor(n_estimators=200, random_state=42, verbosity=0)
        model.fit(X_tt, y_tt, sample_weight=sample_weight)
        y_pred = model.predict(X_vt)
        mae_all, mae_high, bias_high, n_high = evaluate(y_vt.values, y_pred)
        grid_results.append({"horizon": h, "weight": w, "MAE_all": mae_all, "MAE_high": mae_high, "bias_high": bias_high})
        print(f"{h:7d} {w:7.1f} {mae_all:9.3f} {mae_high:9.3f} {bias_high:10.3f}")

grid_df = pd.DataFrame(grid_results)
print()

# เลือก weight ต่อ horizon ตัวแทน: |bias_high| น้อยสุด โดย MAE_all ต้องไม่แย่ลงเกิน 8% เทียบ weight=1.0
chosen_weights = {}
for h in REP_HORIZONS:
    sub = grid_df[grid_df["horizon"] == h].reset_index(drop=True)
    baseline_mae = sub.loc[sub["weight"] == 1.0, "MAE_all"].values[0]
    sub["mae_increase_pct"] = (sub["MAE_all"] - baseline_mae) / baseline_mae * 100
    ok = sub[sub["mae_increase_pct"] <= 8.0]
    best = ok.loc[ok["bias_high"].abs().idxmin()]
    chosen_weights[h] = best["weight"]
    print(f"t+{h}: เลือก weight={best['weight']:.1f} (MAE_all +{best['mae_increase_pct']:.1f}%, bias_high={best['bias_high']:.3f})")

# interpolate weight ไปทั้ง 24 horizon จาก 3 จุดตัวแทน (linear interpolation)
xs = REP_HORIZONS
ys = [chosen_weights[h] for h in xs]
final_weight_by_horizon = {h: float(np.interp(h, xs, ys)) for h in range(1, 25)}

print("\nWeight ที่ใช้จริงทุก horizon (interpolate จาก t+1, t+12, t+24):")
print({h: round(w, 2) for h, w in final_weight_by_horizon.items()})

print("\n=== Step 2: เทรนโมเดลสุดท้ายด้วย sample weight บน train เต็ม แล้ว evaluate บน test set จริง ===")
print("=== (เทียบกับโมเดล unweighted ที่ deploy อยู่ตอนนี้ บน test set เดียวกัน) ===\n")

weighted_models = {}
results_final = []
for h in range(1, 25):
    target = f"pm25_t+{h}"
    w = final_weight_by_horizon[h]

    # แต่ละ horizon มี NaN ต่างกัน (target ตกใน long-gap หรือเกินขอบข้อมูล) ตัดเฉพาะแถวที่ใช้ได้
    train_valid = train.dropna(subset=[target])
    test_valid = test.dropna(subset=[target])
    X_tr, y_tr = train_valid[feature_cols], train_valid[target]
    X_te, y_te = test_valid[feature_cols], test_valid[target]

    sample_weight = np.where(y_tr > HIGH_THRESHOLD, w, 1.0)

    model = xgb.XGBRegressor(n_estimators=200, random_state=42, verbosity=0)
    model.fit(X_tr, y_tr, sample_weight=sample_weight)

    y_pred = model.predict(X_te)
    mae_all, mae_high, bias_high, n_high = evaluate(y_te.values, y_pred)
    rmse_all = mean_squared_error(y_te, y_pred) ** 0.5
    r2_all = r2_score(y_te, y_pred)

    # baseline: โมเดล unweighted ที่ deploy อยู่จริงตอนนี้ ประเมินบน test_valid ชุดเดียวกัน
    y_pred_base = deployed_models[h].predict(X_te)
    mae_all_b, mae_high_b, bias_high_b, _ = evaluate(y_te.values, y_pred_base)

    weighted_models[h] = model
    results_final.append({
        "horizon": h, "weight": w,
        "MAE_all_baseline": mae_all_b, "MAE_all_weighted": mae_all,
        "MAE_high_baseline": mae_high_b, "MAE_high_weighted": mae_high,
        "bias_high_baseline": bias_high_b, "bias_high_weighted": bias_high,
        "RMSE_all": rmse_all, "R2_all": r2_all, "n_high": n_high,
        "n_train": len(train_valid), "n_test": len(test_valid),
    })
    print(f"t+{h:2d} (w={w:4.1f}) -> MAE_all: {mae_all_b:.3f}->{mae_all:.3f} | "
          f"MAE_high: {mae_high_b:.3f}->{mae_high:.3f} | bias_high: {bias_high_b:+.3f}->{bias_high:+.3f} "
          f"(n_train={len(train_valid)}, n_test={len(test_valid)})")

comparison = pd.DataFrame(results_final)

print("\n=== เปรียบเทียบ baseline (deploy อยู่จริง, unweighted) vs weighted (โมเดลใหม่) ===")
print(comparison[["horizon", "MAE_all_baseline", "MAE_all_weighted",
                   "MAE_high_baseline", "MAE_high_weighted",
                   "bias_high_baseline", "bias_high_weighted"]].to_string(index=False))

print(f"\nMAE_all เฉลี่ย: baseline={comparison['MAE_all_baseline'].mean():.3f} -> weighted={comparison['MAE_all_weighted'].mean():.3f}")
print(f"MAE_high เฉลี่ย: baseline={comparison['MAE_high_baseline'].mean():.3f} -> weighted={comparison['MAE_high_weighted'].mean():.3f}")
print(f"bias_high เฉลี่ย: baseline={comparison['bias_high_baseline'].mean():.3f} -> weighted={comparison['bias_high_weighted'].mean():.3f}")

timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
model_package = {
    "models": weighted_models,
    "feature_cols": feature_cols,
    "target_cols": target_cols,
    "trained_at": timestamp,
    "station_id": station_id,
    "hyperparameters": ("default (n_estimators=200); sample_weight up-weights training rows where "
                         "actual pm25 > 37.5 to correct high-pollution under-prediction bias; "
                         f"trained on raw (non-interpolated) data {DATA_PATH}"),
    "weight_by_horizon": final_weight_by_horizon,
    "test_mae_by_horizon": comparison.set_index("horizon")["MAE_all_weighted"].to_dict(),
    "test_r2_by_horizon": comparison.set_index("horizon")["R2_all"].to_dict(),
    "test_mae_high_by_horizon": comparison.set_index("horizon")["MAE_high_weighted"].to_dict(),
    "test_bias_high_by_horizon": comparison.set_index("horizon")["bias_high_weighted"].to_dict(),
}
model_path = rf"{MODEL_DIR}\xgb_pm25_multistep_weighted_{timestamp}.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model_package, f)

eval_csv_path = rf"{MODEL_DIR}\eval_xgb_weighted_vs_baseline_{timestamp}.csv"
comparison.drop(columns=["n_train", "n_test"], errors="ignore").to_csv(eval_csv_path, index=False)

print(f"\nบันทึกโมเดลใหม่ที่: {model_path}")
print(f"บันทึก eval CSV ที่: {eval_csv_path}")

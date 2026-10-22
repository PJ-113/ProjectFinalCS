"""
เทรน unweighted baseline LightGBM บน PM2.5 ดิบของ Khon Kaen (24 horizon แยกโมเดล)
วัด MAE/RMSE/R2 ทั้งหมดและเฉพาะช่วงมลพิษสูง (pm25 > 37.5)
"""
import os
import pickle
from datetime import datetime as dt

import lightgbm as lgb
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(_ROOT, r"data\khonkaen_pm25_features_raw_20260712_142359.csv")
MODEL_DIR = os.path.join(_ROOT, r"models")
HIGH_THRESHOLD = 37.5

feature_cols = [
    "temp", "humidity", "wind_speed", "precipitation",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "dayofweek",
    "pm25_lag_1", "pm25_lag_3", "pm25_lag_6", "pm25_lag_12", "pm25_lag_24",
    "pm25_roll_mean_6", "pm25_roll_std_6", "pm25_roll_mean_24", "pm25_roll_std_24"
]
target_cols = [f"pm25_t+{h}" for h in range(1, 25)]

df_final = pd.read_csv(DATA_PATH)
df_final["datetime_utc"] = pd.to_datetime(df_final["datetime_utc"])

split_idx = int(len(df_final) * 0.8)
train = df_final.iloc[:split_idx]
test = df_final.iloc[split_idx:]

X_train = train[feature_cols]
X_test = test[feature_cols]

print(f"train: {len(train)} แถว ({train['datetime_utc'].min()} -> {train['datetime_utc'].max()})")
print(f"test : {len(test)} แถว ({test['datetime_utc'].min()} -> {test['datetime_utc'].max()})\n")

baseline_models = {}
rows = []
for h in range(1, 25):
    target = f"pm25_t+{h}"

    # แต่ละ horizon มี NaN ต่างกัน (target ตกใน long-gap หรือเกินขอบข้อมูล) ตัดเฉพาะแถวที่ใช้ได้
    train_valid = train.dropna(subset=[target])
    test_valid = test.dropna(subset=[target])

    model = lgb.LGBMRegressor(n_estimators=200, random_state=42, verbose=-1)
    model.fit(train_valid[feature_cols], train_valid[target])

    y_true = test_valid[target]
    y_pred = model.predict(test_valid[feature_cols])

    mae_all = mean_absolute_error(y_true, y_pred)
    rmse_all = mean_squared_error(y_true, y_pred) ** 0.5
    r2_all = r2_score(y_true, y_pred)

    mask_high = y_true > HIGH_THRESHOLD
    n_high = int(mask_high.sum())
    if n_high > 1:
        mae_high = mean_absolute_error(y_true[mask_high], y_pred[mask_high])
        rmse_high = mean_squared_error(y_true[mask_high], y_pred[mask_high]) ** 0.5
        bias_high = float((y_true[mask_high] - y_pred[mask_high]).mean())
    else:
        mae_high = rmse_high = bias_high = float("nan")

    baseline_models[h] = model
    rows.append({
        "horizon": h, "MAE_all": mae_all, "RMSE_all": rmse_all, "R2_all": r2_all,
        "n_high": n_high, "MAE_high": mae_high, "RMSE_high": rmse_high, "bias_high": bias_high,
        "n_train": len(train_valid), "n_test": len(test_valid),
    })
    print(f"t+{h:2d} -> MAE_all: {mae_all:.3f} | R2_all: {r2_all:.3f} | n_high: {n_high} | "
          f"MAE_high: {mae_high:.3f} | bias_high: {bias_high:+.3f} (n_train={len(train_valid)}, n_test={len(test_valid)})")

results = pd.DataFrame(rows)
print("\n=== สรุป ===")
print(f"MAE_all เฉลี่ยทุก horizon: {results['MAE_all'].mean():.3f}")
print(f"MAE_high เฉลี่ยทุก horizon: {results['MAE_high'].mean():.3f}")
print(f"bias_high เฉลี่ยทุก horizon: {results['bias_high'].mean():+.3f}")

timestamp = dt.now().strftime("%Y%m%d_%H%M%S")
model_package = {
    "models": baseline_models,
    "feature_cols": feature_cols,
    "target_cols": target_cols,
    "trained_at": timestamp,
    "station_id": 7747,
    "hyperparameters": "default (n_estimators=200), no sample weighting — baseline for new raw (non-interpolated) dataset",
    "test_mae_by_horizon": results.set_index("horizon")["MAE_all"].to_dict(),
    "test_r2_by_horizon": results.set_index("horizon")["R2_all"].to_dict(),
    "test_mae_high_by_horizon": results.set_index("horizon")["MAE_high"].to_dict(),
    "test_bias_high_by_horizon": results.set_index("horizon")["bias_high"].to_dict(),
}
model_path = os.path.join(MODEL_DIR, f"lgb_pm25_multistep_baseline_{timestamp}.pkl")
with open(model_path, "wb") as f:
    pickle.dump(model_package, f)

eval_csv_path = os.path.join(MODEL_DIR, "eval_high_pollution_by_horizon_v2.csv")
results.drop(columns=["n_train", "n_test"]).to_csv(eval_csv_path, index=False)

print(f"\nบันทึกโมเดล baseline ที่: {model_path}")
print(f"บันทึกผล eval ที่: {eval_csv_path}")

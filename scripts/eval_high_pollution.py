import pickle

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(_ROOT, r"data\khonkaen_pm25_features_final.csv")
MODEL_PATH = os.path.join(_ROOT, r"models\lgb_pm25_multistep_20260708_153732.pkl")
HIGH_THRESHOLD = 37.5

df_final = pd.read_csv(DATA_PATH)
df_final["datetime_utc"] = pd.to_datetime(df_final["datetime_utc"])

with open(MODEL_PATH, "rb") as f:
    package = pickle.load(f)

feature_cols = package["feature_cols"]
models = package["models"]

# same chronological 80/20 split used during training
split_idx = int(len(df_final) * 0.8)
train = df_final.iloc[:split_idx]
test = df_final.iloc[split_idx:]

X_test = test[feature_cols]

print(f"Test period: {test['datetime_utc'].min()} -> {test['datetime_utc'].max()}")
print(f"Test rows: {len(test)}\n")

rows = []
for h in range(1, 25):
    target = f"pm25_t+{h}"
    y_true = test[target]
    y_pred = models[h].predict(X_test)

    # overall
    mae_all = mean_absolute_error(y_true, y_pred)
    rmse_all = mean_squared_error(y_true, y_pred) ** 0.5
    r2_all = r2_score(y_true, y_pred)

    # high-pollution subset only (actual value > threshold)
    mask_high = y_true > HIGH_THRESHOLD
    n_high = mask_high.sum()
    if n_high > 1:
        mae_high = mean_absolute_error(y_true[mask_high], y_pred[mask_high])
        rmse_high = mean_squared_error(y_true[mask_high], y_pred[mask_high]) ** 0.5
        # signed bias: positive = under-prediction (model predicts lower than actual)
        bias_high = (y_true[mask_high] - y_pred[mask_high]).mean()
    else:
        mae_high = rmse_high = bias_high = float("nan")

    rows.append({
        "horizon": h,
        "MAE_all": mae_all,
        "RMSE_all": rmse_all,
        "R2_all": r2_all,
        "n_high": n_high,
        "MAE_high": mae_high,
        "RMSE_high": rmse_high,
        "bias_high": bias_high,
    })

results = pd.DataFrame(rows)
pd.set_option("display.width", 140)
pd.set_option("display.float_format", lambda x: f"{x:.3f}")
print(results.to_string(index=False))

print("\n=== สรุป ===")
print(f"จำนวนแถวทั้งหมดใน test set ที่ actual PM2.5 > {HIGH_THRESHOLD}: {(test[[f'pm25_t+{h}' for h in range(1,25)]] > HIGH_THRESHOLD).any(axis=1).sum()} (นับซ้ำได้หลาย horizon ต่อแถว)")
print(f"MAE เฉลี่ยทั้งหมด (all horizons, overall): {results['MAE_all'].mean():.3f}")
print(f"MAE เฉลี่ยทั้งหมด (all horizons, เฉพาะช่วงสูง >{HIGH_THRESHOLD}): {results['MAE_high'].mean():.3f}")
print(f"Bias เฉลี่ยช่วงสูง (บวก = โมเดลทายต่ำกว่าจริง): {results['bias_high'].mean():.3f}")

results.to_csv(os.path.join(_ROOT, r"models\eval_high_pollution_by_horizon.csv"), index=False)
print("\nบันทึกผลละเอียดที่ models/eval_high_pollution_by_horizon.csv")

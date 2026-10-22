"""
คำนวณ bias_high/mae_high ต่อ horizon ของ XGBoost ที่ deploy จริง บน test set เดิม
(inference ซ้ำ ไม่ retrain) สำหรับใช้ประกอบ EFFECTIVE_THRESHOLD_BY_HORIZON_BY_MODEL ใน app.py
"""
import pickle
import pandas as pd
from sklearn.metrics import mean_absolute_error

import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(_ROOT, r"data\khonkaen_pm25_features_raw_20260712_191249.csv")
XGB_PATH = os.path.join(_ROOT, r"models\xgb_pm25_multistep_20260712_191350.pkl")
HIGH_THRESHOLD = 37.5

with open(XGB_PATH, "rb") as f:
    xgb_pkg = pickle.load(f)
FEATURE_COLS = xgb_pkg["feature_cols"]

df = pd.read_csv(DATA_PATH)
df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
split_idx = int(len(df) * 0.8)
test = df.iloc[split_idx:]

rows = []
for h in range(1, 25):
    target = f"pm25_t+{h}"
    test_valid = test.dropna(subset=[target])
    model = xgb_pkg["models"][h]
    y_true = test_valid[target].values
    y_pred = model.predict(test_valid[FEATURE_COLS])

    mask_high = y_true > HIGH_THRESHOLD
    n_high = int(mask_high.sum())
    if n_high > 0:
        bias_high = float((y_true[mask_high] - y_pred[mask_high]).mean())
        mae_high = float(mean_absolute_error(y_true[mask_high], y_pred[mask_high]))
    else:
        bias_high = float("nan")
        mae_high = float("nan")

    rows.append({"horizon": h, "bias_high": bias_high, "mae_high": mae_high, "n_high": n_high})

df_out = pd.DataFrame(rows)
out_path = os.path.join(_ROOT, r"models\xgb_bias_high_by_horizon_20260712_191350.csv")
df_out.to_csv(out_path, index=False)
print(df_out.to_string(index=False))
print(f"\nsaved: {out_path}")

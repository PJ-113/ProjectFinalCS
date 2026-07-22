"""
Backtest ผลของโมเดลที่ deploy อยู่จริง (app.py's MODEL_PATH/XGB_MODEL_PATH) เฉพาะช่วงฤดูฝุ่น
(ธ.ค.-มี.ค.) ที่ตกอยู่ใน chronological test split เดิม (80/20 เหมือนตอนเทรน) — ข้อมูลส่วนนี้เป็น
out-of-sample จริง (โมเดลไม่เคยเห็นตอนเทรน) ไม่ใช่แค่กรองจากข้อมูลทั้งหมดเฉยๆ ซึ่งจะเสี่ยง data
leakage ถ้าดันไปรวมแถวที่อยู่ใน train split ด้วย

อ่านอย่างเดียว ไม่แตะ predictions.db/Postgres, ไม่ต้องเปิด Flask/frontend เลย (ดู CLAUDE.md
เรื่อง "Data source mismatch" — เทียบใน sqlite/CSV scale เดียวกับตอนเทรน ไม่ต้องผ่าน
calibrate_pm25() เพราะไม่ได้ดึงจาก Air4Thai)
"""
import pickle
import sys

sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(_ROOT, r"data\khonkaen_pm25_features_raw_20260712_191249.csv")
LGB_MODEL_PATH = os.path.join(_ROOT, r"models\lgb_pm25_multistep_weighted_20260712_191056.pkl")
XGB_MODEL_PATH = os.path.join(_ROOT, r"models\xgb_pm25_multistep_20260712_191350.pkl")
HIGH_THRESHOLD = 37.5
HAZE_MONTHS = {12, 1, 2, 3}

df = pd.read_csv(DATA_PATH)
df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])

with open(LGB_MODEL_PATH, "rb") as f:
    lgb_package = pickle.load(f)
with open(XGB_MODEL_PATH, "rb") as f:
    xgb_package = pickle.load(f)

feature_cols = lgb_package["feature_cols"]
assert feature_cols == xgb_package["feature_cols"], "feature_cols mismatch between LightGBM/XGBoost packages"
lgb_models = lgb_package["models"]
xgb_models = xgb_package["models"]

# chronological 80/20 split เดียวกับตอนเทรน (ดู CLAUDE.md: train/test split convention)
split_idx = int(len(df) * 0.8)
test = df.iloc[split_idx:]
haze_test = test[test["datetime_utc"].dt.month.isin(HAZE_MONTHS)]

print(f"Test split ทั้งหมด: {test['datetime_utc'].min()} -> {test['datetime_utc'].max()} ({len(test)} แถว)")
print(f"เฉพาะฤดูฝุ่น (ธ.ค.-มี.ค.) ภายใน test split: "
      f"{haze_test['datetime_utc'].min()} -> {haze_test['datetime_utc'].max()} ({len(haze_test)} แถว)")
print()


def eval_model(models, X_all, label):
    rows = []
    for h in range(1, 25):
        target = f"pm25_t+{h}"
        sub = haze_test.dropna(subset=[target])  # แต่ละ horizon มีจำนวนแถวใช้ได้ไม่เท่ากันใกล้ gap (ดู CLAUDE.md)
        if len(sub) == 0:
            rows.append({"horizon": h, "n": 0, "MAE": float("nan"), "bias": float("nan"),
                         "n_high": 0, "MAE_high": float("nan"), "bias_high": float("nan")})
            continue

        X = sub[feature_cols]
        y_true = sub[target]
        y_pred = models[h].predict(X)

        mae = mean_absolute_error(y_true, y_pred)
        bias = (y_true - y_pred).mean()  # บวก = ทายต่ำกว่าจริง

        mask_high = y_true > HIGH_THRESHOLD
        n_high = int(mask_high.sum())
        if n_high > 1:
            mae_high = mean_absolute_error(y_true[mask_high], y_pred[mask_high])
            bias_high = (y_true[mask_high] - y_pred[mask_high]).mean()
        else:
            mae_high = bias_high = float("nan")

        rows.append({
            "horizon": h, "n": len(sub), "MAE": mae, "bias": bias,
            "n_high": n_high, "MAE_high": mae_high, "bias_high": bias_high,
        })

    results = pd.DataFrame(rows)
    print(f"=== {label} (เฉพาะฤดูฝุ่นใน test split) ===")
    pd.set_option("display.width", 140)
    pd.set_option("display.float_format", lambda x: f"{x:.3f}")
    print(results.to_string(index=False))
    print(f"MAE เฉลี่ยทุก horizon: {results['MAE'].mean():.3f}")
    print(f"MAE เฉลี่ยเฉพาะช่วงสูง (>{HIGH_THRESHOLD}): {results['MAE_high'].mean():.3f}")
    print(f"Bias เฉลี่ยช่วงสูง (บวก = ทายต่ำกว่าจริง): {results['bias_high'].mean():.3f}")
    print()
    return results


lgb_results = eval_model(lgb_models, haze_test, "LightGBM (weighted, deployed)")
xgb_results = eval_model(xgb_models, haze_test, "XGBoost (deployed)")

out_path = os.path.join(_ROOT, r"models\eval_haze_season_backtest.csv")
combined = pd.concat([lgb_results.assign(model="lightgbm"), xgb_results.assign(model="xgboost")], ignore_index=True)
combined.to_csv(out_path, index=False)
print(f"บันทึกผลละเอียดที่ {out_path}")

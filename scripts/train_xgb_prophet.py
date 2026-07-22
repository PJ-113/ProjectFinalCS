"""
เทรน XGBoost (24 horizon แบบ direct multi-step) และ Prophet ให้พร้อมใช้งานแบบเดียวกับ
LightGBM ที่ deploy อยู่ (models/lgb_pm25_multistep_weighted_....pkl) สำหรับระบบ
Circuit Breaker + Dynamic Routing

feature_cols / target_cols / station_id โหลดมาจาก metadata ของ pkl LightGBM ที่มีอยู่แล้ว
เพื่อให้ตรงกันเป๊ะ ไม่ hardcode ซ้ำ ส่วน hyperparameter และ train/test split ใช้ตาม
train_models.ipynb cell 7bcd234a (XGBoost) และ 15ec5c41/57a04954/46d0ea61 (Prophet)
เพื่อให้ผลตรงกับตอนเปรียบเทียบโมเดลครั้งแรก (XGBoost MAE ~2.728, Prophet MAE ~7.586)
"""
import os
import pickle
import time
from datetime import datetime as dt

import pandas as pd
import xgboost as xgb
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(_ROOT, r"data\khonkaen_pm25_features_final.csv")
LGB_PKL_PATH = os.path.join(_ROOT, r"models\lgb_pm25_multistep_weighted_20260709_004955.pkl")
MODEL_DIR = os.path.join(_ROOT, r"models")

# โหลด feature_cols/target_cols/station_id จาก metadata ของ LightGBM ที่ deploy อยู่ ให้ตรงกันเป๊ะ
with open(LGB_PKL_PATH, "rb") as f:
    lgb_package = pickle.load(f)
FEATURE_COLS = lgb_package["feature_cols"]
TARGET_COLS = lgb_package["target_cols"]
STATION_ID = lgb_package["station_id"]

df_final = pd.read_csv(DATA_PATH)
df_final["datetime_utc"] = pd.to_datetime(df_final["datetime_utc"])

# split แบบ time-based 80/20 เดียวกับตอนเปรียบเทียบโมเดลครั้งแรก
split_idx = int(len(df_final) * 0.8)
train = df_final.iloc[:split_idx]
test = df_final.iloc[split_idx:]

X_train, y_train = train[FEATURE_COLS], train[TARGET_COLS]
X_test, y_test = test[FEATURE_COLS], test[TARGET_COLS]

timestamp = dt.now().strftime("%Y%m%d_%H%M%S")


def train_xgboost():
    xgb_models = {}
    results = []

    start_total = time.time()
    for h in range(1, 25):
        target = f"pm25_t+{h}"
        model = xgb.XGBRegressor(n_estimators=200, random_state=42, verbosity=0)
        model.fit(X_train, y_train[target])

        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test[target], y_pred)
        rmse = mean_squared_error(y_test[target], y_pred) ** 0.5
        r2 = r2_score(y_test[target], y_pred)

        xgb_models[h] = model
        results.append({"horizon": h, "MAE": mae, "RMSE": rmse, "R2": r2})
        print(f"[xgb] t+{h:2d} -> MAE: {mae:.3f} | RMSE: {rmse:.3f} | R2: {r2:.3f}")

    print(f"[xgb] total train time: {time.time() - start_total:.2f}s")

    df_results = pd.DataFrame(results)

    package = {
        "models": xgb_models,
        "feature_cols": FEATURE_COLS,
        "target_cols": TARGET_COLS,
        "trained_at": timestamp,
        "station_id": STATION_ID,
        "hyperparameters": "default (n_estimators=200), matches original 3-model comparison",
        "test_mae_by_horizon": df_results.set_index("horizon")["MAE"].to_dict(),
        "test_r2_by_horizon": df_results.set_index("horizon")["R2"].to_dict(),
    }

    model_path = os.path.join(MODEL_DIR, f"xgb_pm25_multistep_{timestamp}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(package, f)

    print(f"[xgb] saved: {model_path}")
    print(f"[xgb] overall MAE (mean across 24 horizons): {df_results['MAE'].mean():.3f}")
    return model_path, df_results


def train_prophet():
    df_prophet = df_final[["datetime_utc", "pm25", "temp", "humidity", "wind_speed", "precipitation"]].copy()
    df_prophet.columns = ["ds", "y", "temp", "humidity", "wind_speed", "precipitation"]
    df_prophet["ds"] = df_prophet["ds"].dt.tz_localize(None)

    train_prophet_df = df_prophet.iloc[:split_idx]
    test_prophet_df = df_prophet.iloc[split_idx:]

    start = time.time()
    model = Prophet()
    model.add_regressor("temp")
    model.add_regressor("humidity")
    model.add_regressor("wind_speed")
    model.add_regressor("precipitation")
    model.fit(train_prophet_df)
    print(f"[prophet] train time: {time.time() - start:.2f}s")

    future = test_prophet_df[["ds", "temp", "humidity", "wind_speed", "precipitation"]].copy()
    forecast = model.predict(future)

    y_true = test_prophet_df["y"].values
    y_pred = forecast["yhat"].values

    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    r2 = r2_score(y_true, y_pred)
    print(f"[prophet] MAE: {mae:.3f} | RMSE: {rmse:.3f} | R2: {r2:.3f}")

    package = {
        "model": model,
        "feature_cols": ["temp", "humidity", "wind_speed", "precipitation"],
        "target_cols": ["pm25"],
        "trained_at": timestamp,
        "station_id": STATION_ID,
        "hyperparameters": "default Prophet params + 4 extra regressors (temp/humidity/wind_speed/precipitation)",
        "test_mae": mae,
        "test_rmse": rmse,
        "test_r2": r2,
    }

    model_path = os.path.join(MODEL_DIR, f"prophet_pm25_{timestamp}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(package, f)

    print(f"[prophet] saved: {model_path}")
    return model_path, mae, rmse, r2


def verify_prediction(xgb_path, prophet_path):
    print("\n" + "=" * 80)
    print("VERIFICATION: predict 1 point from each reloaded model vs actual (test set row 0)")
    print("=" * 80)

    row = X_test.iloc[[0]]
    actual_t1 = y_test["pm25_t+1"].iloc[0]

    with open(xgb_path, "rb") as f:
        xgb_pkg = pickle.load(f)
    xgb_pred_t1 = xgb_pkg["models"][1].predict(row)[0]
    print(f"XGBoost  t+1 predicted: {xgb_pred_t1:.2f} | actual: {actual_t1:.2f}")

    with open(prophet_path, "rb") as f:
        prophet_pkg = pickle.load(f)
    df_prophet = df_final[["datetime_utc", "pm25", "temp", "humidity", "wind_speed", "precipitation"]].copy()
    df_prophet.columns = ["ds", "y", "temp", "humidity", "wind_speed", "precipitation"]
    df_prophet["ds"] = df_prophet["ds"].dt.tz_localize(None)
    test_prophet_df = df_prophet.iloc[split_idx:]
    one_row_future = test_prophet_df[["ds", "temp", "humidity", "wind_speed", "precipitation"]].iloc[[0]]
    prophet_pred = prophet_pkg["model"].predict(one_row_future)["yhat"].iloc[0]
    prophet_actual = test_prophet_df["y"].iloc[0]
    print(f"Prophet  predicted: {prophet_pred:.2f} | actual: {prophet_actual:.2f}")


if __name__ == "__main__":
    print(f"feature_cols ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    print(f"target_cols ({len(TARGET_COLS)}): {TARGET_COLS[0]} .. {TARGET_COLS[-1]}")
    print(f"station_id: {STATION_ID}")
    print(f"train rows: {len(train)} | test rows: {len(test)}\n")

    xgb_path, xgb_results = train_xgboost()
    print()
    prophet_path, prophet_mae, prophet_rmse, prophet_r2 = train_prophet()

    verify_prediction(xgb_path, prophet_path)

    print("\n" + "=" * 80)
    print("SUMMARY vs original comparison (train_models.ipynb)")
    print("=" * 80)
    print(f"XGBoost  MAE mean (24 horizons): {xgb_results['MAE'].mean():.3f}  (original: ~2.728)")
    print(f"Prophet  MAE:                    {prophet_mae:.3f}  (original: ~7.586)")
    print(f"\nXGBoost model:  {xgb_path}")
    print(f"Prophet model:  {prophet_path}")

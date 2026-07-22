import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(_ROOT, r"data\khonkaen_pm25_features_final.csv")
OUT_CSV = os.path.join(_ROOT, r"models\r2_comparison_3models_by_horizon.csv")

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

X_train, y_train = train[feature_cols], train[target_cols]
X_test, y_test = test[feature_cols], test[target_cols]

rows = []
for h in range(1, 25):
    target = f"pm25_t+{h}"

    m_lgb = lgb.LGBMRegressor(n_estimators=200, random_state=42, verbose=-1)
    m_lgb.fit(X_train, y_train[target])
    r2_lgb = r2_score(y_test[target], m_lgb.predict(X_test))

    m_xgb = xgb.XGBRegressor(n_estimators=200, random_state=42, verbosity=0)
    m_xgb.fit(X_train, y_train[target])
    r2_xgb = r2_score(y_test[target], m_xgb.predict(X_test))

    rows.append({"horizon": h, "R2_lgb": r2_lgb, "R2_xgb": r2_xgb})
    print(f"t+{h:2d} -> R2_lgb={r2_lgb:.4f}  R2_xgb={r2_xgb:.4f}")

# --- Prophet: single-timestep nowcast R2 (not per-horizon, different methodology) ---
from prophet import Prophet

df_prophet = df_final[["datetime_utc", "pm25", "temp", "humidity", "wind_speed", "precipitation"]].copy()
df_prophet.columns = ["ds", "y", "temp", "humidity", "wind_speed", "precipitation"]
df_prophet["ds"] = df_prophet["ds"].dt.tz_localize(None)
train_prophet = df_prophet.iloc[:split_idx]
test_prophet = df_prophet.iloc[split_idx:]

model_prophet = Prophet()
for col in ["temp", "humidity", "wind_speed", "precipitation"]:
    model_prophet.add_regressor(col)
model_prophet.fit(train_prophet)

future = test_prophet[["ds", "temp", "humidity", "wind_speed", "precipitation"]].copy()
forecast = model_prophet.predict(future)
r2_prophet = r2_score(test_prophet["y"].values, forecast["yhat"].values)
print(f"\nProphet (single-timestep nowcast) R2 = {r2_prophet:.4f}")

df_out = pd.DataFrame(rows)
df_out["R2_prophet_flat"] = r2_prophet
df_out.to_csv(OUT_CSV, index=False)
print(f"\nSaved: {OUT_CSV}")

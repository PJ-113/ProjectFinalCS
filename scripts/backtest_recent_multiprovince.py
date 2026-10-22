"""
Backtest โมเดลที่เทรนไว้ (train_multiprovince_models.py) กับข้อมูลจริงหลังช่วงเทรน
(2026-07-02 เป็นต้นไป) ดึงข้อมูลสดจาก OpenAQ/Open-Meteo เอง (ไม่แตะ app.py/DB)
วัด MAE/R2/coverage ต่อ horizon ต่อจังหวัด เก็บผลลง CSV แยกต่างหาก
"""
import functools
import json
import os
import pickle
import time

import numpy as np
import pandas as pd
import requests

print = functools.partial(print, flush=True)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(_ROOT, ".env")
REGISTRY_PATH = os.path.join(_ROOT, "data", "province_station_registry.json")
MODEL_DIR = os.path.join(_ROOT, "models")

LGB_MODEL_PATH = os.path.join(MODEL_DIR, "lgb_pm25_multiprovince_residual_bagged_quantile_20260913_014901.pkl")
XGB_MODEL_PATH = os.path.join(MODEL_DIR, "xgb_pm25_multiprovince_residual_bagged_quantile_20260913_014901.pkl")
CONFORMAL_LGB_PATH = os.path.join(MODEL_DIR, "conformal_calibration_lgb_20260913_014901.csv")
CONFORMAL_XGB_PATH = os.path.join(MODEL_DIR, "conformal_calibration_xgb_20260913_014901.csv")

TRAIN_WINDOW_END = pd.Timestamp("2026-07-02 00:00:00", tz="UTC")  # ท้ายช่วงเทรนเดิม
FETCH_START = TRAIN_WINDOW_END - pd.Timedelta(days=5)  # buffer ให้ lag_24/roll_24 มีข้อมูลย้อนหลังพอ
FETCH_END = pd.Timestamp.utcnow().floor("h") - pd.Timedelta(hours=1)  # เผื่อข้อมูลล่าสุดยังไม่ finalize
GAP_CUTOFF_HOURS = 48
REQUEST_SLEEP_S = 0.4
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

FEATURE_COLS = [
    "temp", "humidity", "wind_speed", "precipitation",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "dayofweek",
    "pm25_lag_1", "pm25_lag_3", "pm25_lag_6", "pm25_lag_12", "pm25_lag_24",
    "pm25_roll_mean_6", "pm25_roll_std_6", "pm25_roll_mean_24", "pm25_roll_std_24",
    "wind_dir_sin", "wind_dir_cos", "pressure", "lat", "lon",
]
PM25_FEATURE_COLS = [
    "pm25_lag_1", "pm25_lag_3", "pm25_lag_6", "pm25_lag_12", "pm25_lag_24",
    "pm25_roll_mean_6", "pm25_roll_std_6", "pm25_roll_mean_24", "pm25_roll_std_24",
]


def _load_env_key(name):
    with open(ENV_PATH, encoding="utf-8") as f:
        for line in f:
            if line.startswith(f"{name}="):
                return line.strip().split("=", 1)[1]
    raise RuntimeError(f"{name} not found in .env")


OPENAQ_KEY = _load_env_key("OPENAQ_KEY")
HEADERS = {"X-API-Key": OPENAQ_KEY}


def fetch_pm25(sensor_id):
    def fetch_chunk(dt_from, dt_to):
        params = {"datetime_from": dt_from.isoformat(), "datetime_to": dt_to.isoformat(), "limit": 1000}
        for attempt in range(6):
            try:
                r = requests.get(f"https://api.openaq.org/v3/sensors/{sensor_id}/measurements",
                                  headers=HEADERS, params=params, timeout=30)
            except requests.exceptions.RequestException:
                time.sleep(2 ** attempt)
                continue
            if r.status_code == 429 or r.status_code >= 500:
                time.sleep(2 ** attempt)
                continue
            r.raise_for_status()
            return r.json().get("results", [])
        print(f"    !! ยอมแพ้ chunk {dt_from}..{dt_to}")
        return []

    all_rows = []
    cur = FETCH_START
    while cur < FETCH_END:
        chunk_end = min(cur + pd.Timedelta(days=30), FETCH_END)
        all_rows.extend(fetch_chunk(cur, chunk_end))
        cur = chunk_end
        time.sleep(REQUEST_SLEEP_S)

    if not all_rows:
        return pd.DataFrame(columns=["datetime_utc", "pm25"])
    df = pd.DataFrame([{"datetime_utc": r["period"]["datetimeFrom"]["utc"], "pm25": r["value"]} for r in all_rows])
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    return df.dropna(subset=["pm25"]).drop_duplicates(subset=["datetime_utc"]).sort_values("datetime_utc").reset_index(drop=True)


def fetch_weather(lat, lon):
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": FETCH_START.strftime("%Y-%m-%d"), "end_date": FETCH_END.strftime("%Y-%m-%d"),
        "hourly": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,wind_direction_10m,surface_pressure",
        "timezone": "UTC",
    }
    for attempt in range(6):
        r = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=60)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        break
    h = r.json()["hourly"]
    return pd.DataFrame({
        "datetime_utc": pd.to_datetime(h["time"]).tz_localize("UTC"),
        "temp": h["temperature_2m"], "humidity": h["relative_humidity_2m"],
        "wind_speed": h["wind_speed_10m"], "precipitation": h["precipitation"],
        "wind_dir_deg": h["wind_direction_10m"], "pressure": h["surface_pressure"],
    })


def build_station_features(province, pm25_raw, weather):
    """เหมือน build_one_station ใน build_raw_dataset_multiprovince.py แบบย่อ"""
    full_grid = pd.date_range(FETCH_START, FETCH_END, freq="h", tz="UTC")
    grid = pd.DataFrame({"datetime_utc": full_grid})
    grid = grid.merge(pm25_raw[["datetime_utc", "pm25"]], on="datetime_utc", how="left")
    grid = grid.merge(weather, on="datetime_utc", how="left")
    grid = grid.sort_values("datetime_utc").reset_index(drop=True)

    weather_cols = ["temp", "humidity", "wind_speed", "precipitation", "wind_dir_deg", "pressure"]
    grid[weather_cols] = grid[weather_cols].interpolate(method="linear", limit_direction="both")

    grid["pm25_missing"] = grid["pm25"].isna()
    grid["gap_group"] = (grid["pm25_missing"] != grid["pm25_missing"].shift()).cumsum()
    gap_run_len = grid[grid["pm25_missing"]].groupby("gap_group").size()
    long_gap_groups = gap_run_len[gap_run_len > GAP_CUTOFF_HOURS].index
    grid["is_long_gap"] = grid["pm25_missing"] & grid["gap_group"].isin(long_gap_groups)

    grid["pm25_filled"] = grid["pm25"].interpolate(method="linear", limit_direction="both")
    if grid["pm25_filled"].isna().any():
        return None

    grid["hour"] = grid["datetime_utc"].dt.hour
    grid["month"] = grid["datetime_utc"].dt.month
    grid["dayofweek"] = grid["datetime_utc"].dt.dayofweek
    grid["hour_sin"] = np.sin(2 * np.pi * grid["hour"] / 24)
    grid["hour_cos"] = np.cos(2 * np.pi * grid["hour"] / 24)
    grid["month_sin"] = np.sin(2 * np.pi * grid["month"] / 12)
    grid["month_cos"] = np.cos(2 * np.pi * grid["month"] / 12)
    grid["wind_dir_sin"] = np.sin(2 * np.pi * grid["wind_dir_deg"] / 360)
    grid["wind_dir_cos"] = np.cos(2 * np.pi * grid["wind_dir_deg"] / 360)

    s = grid["pm25_filled"]
    grid["pm25_lag_1"] = s.shift(1)
    grid["pm25_lag_3"] = s.shift(3)
    grid["pm25_lag_6"] = s.shift(6)
    grid["pm25_lag_12"] = s.shift(12)
    grid["pm25_lag_24"] = s.shift(24)
    grid["pm25_roll_mean_6"] = s.rolling(6).mean()
    grid["pm25_roll_std_6"] = s.rolling(6).std()
    grid["pm25_roll_mean_24"] = s.rolling(24).mean()
    grid["pm25_roll_std_24"] = s.rolling(24).std()

    long_gap_int = grid["is_long_gap"].astype(int)
    touches_long_gap = long_gap_int.rolling(25, min_periods=1).max().astype(bool)
    grid.loc[touches_long_gap, PM25_FEATURE_COLS] = np.nan

    pm25_by_time = grid.set_index("datetime_utc")["pm25_filled"]
    is_long_gap_by_time = grid.set_index("datetime_utc")["is_long_gap"]
    for h in range(1, 25):
        target_time = grid["datetime_utc"] + pd.Timedelta(hours=h)
        values = target_time.map(pm25_by_time)
        invalid = target_time.map(is_long_gap_by_time).fillna(True).astype(bool)
        grid[f"pm25_t+{h}"] = values.where(~invalid)

    grid["lat"] = province["lat"]
    grid["lon"] = province["lon"]
    grid["province_th"] = province["province_th"]
    grid["province_en"] = province["province_en"]
    grid["is_zero_shot"] = province["is_zero_shot"]

    grid = grid.drop(columns=["pm25"]).rename(columns={"pm25_filled": "pm25"})
    # เอาเฉพาะช่วงหลังจากช่วงเทรนเดิม — buffer ก่อนหน้านั้นใช้แค่คำนวณ lag/rolling
    grid = grid[grid["datetime_utc"] >= TRAIN_WINDOW_END].reset_index(drop=True)
    grid = grid.dropna(subset=PM25_FEATURE_COLS).reset_index(drop=True)
    return grid


def predict_with_conformal(model_pkg, conformal_df, X, current_pm25):
    """คืน p50/lo/hi (ก่อน conformal) และ lo_adj/hi_adj (หลัง conformal) ต่อ horizon"""
    out = {}
    q_hat_by_h = conformal_df.set_index("horizon")["q_hat"].to_dict()
    for h in range(1, 25):
        models_by_q = model_pkg["models"][h]
        p50 = models_by_q[0.5].predict(X) + current_pm25
        lo = models_by_q[0.05].predict(X) + current_pm25
        hi = models_by_q[0.95].predict(X) + current_pm25
        q_hat = q_hat_by_h[h]
        out[h] = {"p50": p50, "lo": lo, "hi": hi, "lo_adj": lo - q_hat, "hi_adj": hi + q_hat}
    return out


def main():
    print(f"ช่วง backtest (ข้อมูลจริงหลังช่วงเทรน): {TRAIN_WINDOW_END} -> {FETCH_END}")
    print(f"(ดึงข้อมูลตั้งแต่ {FETCH_START} เพื่อให้ lag_24/roll_24 มีข้อมูลย้อนหลังพอ)")

    with open(REGISTRY_PATH, encoding="utf-8") as f:
        registry = json.load(f)
    with open(LGB_MODEL_PATH, "rb") as f:
        lgb_pkg = pickle.load(f)
    with open(XGB_MODEL_PATH, "rb") as f:
        xgb_pkg = pickle.load(f)
    conformal_lgb = pd.read_csv(CONFORMAL_LGB_PATH)
    conformal_xgb = pd.read_csv(CONFORMAL_XGB_PATH)

    assert lgb_pkg["feature_cols"] == FEATURE_COLS, "feature_cols ไม่ตรงกับตอนเทรน ต้องตรวจสอบ"

    all_rows = []
    for province in registry["provinces"]:
        tag = " [zero-shot]" if province["is_zero_shot"] else ""
        print(f"\n{province['province_th']} ({province['province_en']}){tag}")
        try:
            pm25_raw = fetch_pm25(province["sensor_id_pm25"])
            print(f"  pm25: {len(pm25_raw)} ชม.")
            if pm25_raw.empty:
                print("  !! ไม่มีข้อมูล ข้าม")
                continue
            weather = fetch_weather(province["lat"], province["lon"])
            time.sleep(REQUEST_SLEEP_S)
            grid = build_station_features(province, pm25_raw, weather)
            if grid is None or len(grid) == 0:
                print("  !! ไม่มีแถวที่ใช้ประเมินได้หลังตัด lookback/gap ข้าม")
                continue
            print(f"  แถวที่ใช้ backtest ได้: {len(grid)}")

            X = grid[FEATURE_COLS]
            current_pm25 = grid["pm25"].values
            lgb_pred = predict_with_conformal(lgb_pkg, conformal_lgb, X, current_pm25)
            xgb_pred = predict_with_conformal(xgb_pkg, conformal_xgb, X, current_pm25)

            for h in range(1, 25):
                target_col = f"pm25_t+{h}"
                valid = grid[target_col].notna()
                if valid.sum() == 0:
                    continue
                y_true = grid.loc[valid, target_col].values
                row = {
                    "province_th": province["province_th"], "province_en": province["province_en"],
                    "is_zero_shot": province["is_zero_shot"], "horizon": h, "n": int(valid.sum()),
                }
                for tag_m, pred in (("lgb", lgb_pred), ("xgb", xgb_pred)):
                    p50 = pred[h]["p50"][valid.values]
                    lo_adj, hi_adj = pred[h]["lo_adj"][valid.values], pred[h]["hi_adj"][valid.values]
                    row[f"{tag_m}_mae"] = float(np.mean(np.abs(y_true - p50)))
                    ss_res = np.sum((y_true - p50) ** 2)
                    ss_tot = np.sum((y_true - y_true.mean()) ** 2)
                    row[f"{tag_m}_r2"] = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
                    row[f"{tag_m}_coverage"] = float(np.mean((y_true >= lo_adj) & (y_true <= hi_adj)))
                all_rows.append(row)
        except Exception as e:
            print(f"  !! error: {e!r} ข้ามจังหวัดนี้")

    if not all_rows:
        raise RuntimeError("ไม่มีจังหวัดไหน backtest สำเร็จเลย")

    result_df = pd.DataFrame(all_rows)
    out_path = os.path.join(MODEL_DIR, f"backtest_recent_by_horizon_{TIMESTAMP}.csv")
    result_df.to_csv(out_path, index=False)

    def summarize(df, label):
        print(f"\n=== {label} (n จังหวัด={df['province_en'].nunique()}) ===")
        for tag_m in ("lgb", "xgb"):
            w = df["n"]
            mae = np.average(df[f"{tag_m}_mae"], weights=w)
            r2 = np.average(df[f"{tag_m}_r2"], weights=w)
            cov = np.average(df[f"{tag_m}_coverage"], weights=w)
            print(f"  {tag_m.upper()}: MAE={mae:.3f}  R2={r2:.3f}  coverage={cov:.1%} (เป้าหมาย 90%)")

    summarize(result_df[~result_df["is_zero_shot"]], "17 จังหวัดเทรน")
    summarize(result_df[result_df["is_zero_shot"]], "3 จังหวัด zero-shot")
    summarize(result_df, "รวมทั้งหมด (20 จังหวัด)")

    print("\nรายจังหวัด (เฉลี่ยทุก horizon, LightGBM):")
    per_prov = result_df.groupby(["province_th", "is_zero_shot"]).apply(
        lambda g: pd.Series({"mae": np.average(g["lgb_mae"], weights=g["n"]),
                              "r2": np.average(g["lgb_r2"], weights=g["n"]),
                              "coverage": np.average(g["lgb_coverage"], weights=g["n"])}),
        include_groups=False,
    )
    print(per_prov.to_string())

    print(f"\nบันทึกรายละเอียดต่อ horizon ต่อจังหวัดที่: {out_path}")


if __name__ == "__main__":
    main()

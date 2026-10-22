"""
กวาด grid radius x window ของ fire hotspot feature (fire_hotspot_count_{radius}km_{window}h)
เทียบกับ baseline (ไม่มี fire hotspot) ด้วย LightGBM/XGBoost
วัด MAE รวม/ช่วงฤดูฝุ่น/>p90 ที่ 5 horizon ตัวแทน (t+1,6,12,18,24)
"""
import glob
import importlib.util
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_ROOT, "data")
MODEL_DIR = os.path.join(_ROOT, "models")
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
REPRESENTATIVE_HORIZONS = [1, 6, 12, 18, 24]
BURNING_MONTHS = {2, 3, 4}

RADIUS_GRID = [50, 100, 150]
WINDOW_GRID = [24, 48]


def latest_file(pattern):
    matches = sorted(glob.glob(os.path.join(DATA_DIR, pattern)))
    if not matches:
        raise FileNotFoundError(f"ไม่พบไฟล์ตรงกับ {pattern} ใน {DATA_DIR}")
    return matches[-1]


def load_train_module():
    spec = importlib.util.spec_from_file_location(
        "train_multiprovince_models", os.path.join(_ROOT, "scripts", "train_multiprovince_models.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def haversine_km_vec(lat1, lon1, lat2_arr, lon2_arr):
    r = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2_arr)
    dphi = np.radians(lat2_arr - lat1)
    dlambda = np.radians(lon2_arr - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlambda / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def compute_all_configs(df, provinces_meta, hotspot_lat, hotspot_lon, hotspot_dt):
    """คำนวณทุก (radius, window) config คืนชื่อคอลัมน์ทั้งหมดที่เพิ่มเข้า df"""
    feature_cols = []
    for radius_km in RADIUS_GRID:
        # กรองด้วยรัศมีครั้งเดียวต่อจังหวัดต่อ radius แล้ว reuse ข้าม window
        nearby_by_province = {}
        for province_en, meta in provinces_meta.items():
            dist = haversine_km_vec(meta["lat"], meta["lon"], hotspot_lat, hotspot_lon)
            nearby_by_province[province_en] = np.sort(hotspot_dt[dist <= radius_km])

        for window_hours in WINDOW_GRID:
            col = f"fire_hotspot_count_{radius_km}km_{window_hours}h"
            feature_cols.append(col)
            df[col] = 0
            window = np.timedelta64(window_hours, "h")
            for province_en in provinces_meta:
                mask_rows = df["province_en"] == province_en
                if mask_rows.sum() == 0:
                    continue
                nearby_dt = nearby_by_province[province_en]
                target_ts = df.loc[mask_rows, "datetime_utc"].values
                if len(nearby_dt) == 0:
                    continue
                idx_hi = np.searchsorted(nearby_dt, target_ts, side="right")
                idx_lo = np.searchsorted(nearby_dt, target_ts - window, side="right")
                df.loc[mask_rows, col] = (idx_hi - idx_lo).astype(int)
            print(f"  คำนวณ {col} เสร็จ (เฉลี่ย={df[col].mean():.2f}, corr กับ pm25={df[col].corr(df['pm25']):.3f})")
    return feature_cols


def fit_p50(X, y, kind, train_mod):
    if kind == "lgb":
        import lightgbm as lgb
        return lgb.LGBMRegressor(objective="quantile", alpha=0.5, n_estimators=train_mod.N_ESTIMATORS,
                                  random_state=42, verbose=-1, **train_mod.BAGGING_PARAMS_LGB).fit(X, y)
    import xgboost as xgb
    return xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, tree_method="hist",
                             n_estimators=train_mod.N_ESTIMATORS, random_state=42, verbosity=0,
                             **train_mod.BAGGING_PARAMS_XGB).fit(X, y)


def eval_variant(train_df, test_df, feature_cols, kind, variant_name, train_mod):
    rows = []
    for h in REPRESENTATIVE_HORIZONS:
        target_col = f"pm25_t+{h}"
        tr = train_df.dropna(subset=[target_col])
        te = test_df.dropna(subset=[target_col])
        X_tr, y_tr_resid = tr[feature_cols], tr[target_col] - tr["pm25"]
        X_te, current_te = te[feature_cols], te["pm25"].values
        y_te_actual = te[target_col].values

        model = fit_p50(X_tr, y_tr_resid, kind, train_mod)
        pred = model.predict(X_te) + current_te
        mae_overall = mean_absolute_error(y_te_actual, pred)

        month = pd.to_datetime(te["datetime_utc"]).dt.month
        is_burning = month.isin(BURNING_MONTHS).values
        mae_burning = mean_absolute_error(y_te_actual[is_burning], pred[is_burning]) if is_burning.sum() else np.nan

        bins = [-0.01, np.percentile(y_te_actual, 50), np.percentile(y_te_actual, 75),
                np.percentile(y_te_actual, 90), y_te_actual.max() + 0.01]
        sev_bin = pd.cut(y_te_actual, bins=bins, labels=["<=p50", "p50-p75", "p75-p90", ">p90"])
        mae_p90 = mean_absolute_error(y_te_actual[sev_bin == ">p90"], pred[sev_bin == ">p90"]) \
            if (sev_bin == ">p90").sum() else np.nan

        rows.append({"variant": variant_name, "model": kind, "horizon": h, "mae": mae_overall,
                      "mae_burning": mae_burning, "mae_gt_p90": mae_p90, "n": len(te)})
    print(f"  [{variant_name}/{kind}] เสร็จ — MAE เฉลี่ย={np.mean([r['mae'] for r in rows]):.3f}  "
          f"MAE>p90 เฉลี่ย={np.mean([r['mae_gt_p90'] for r in rows]):.3f}")
    return rows


def main():
    k1_data_path = os.path.join(DATA_DIR, "multiprovince_pm25_features_raw_spatial_k1_20260917_001104.csv")
    hotspot_path = latest_file("firms_hotspots_isaan_*.csv")
    print(f"โหลด training dataset: {k1_data_path}")
    print(f"โหลด hotspot events: {hotspot_path}")

    df_all = pd.read_csv(k1_data_path)
    df_all["datetime_utc"] = pd.to_datetime(df_all["datetime_utc"], utc=True)
    hotspots = pd.read_csv(hotspot_path)
    hotspots["acq_datetime_utc"] = pd.to_datetime(hotspots["acq_datetime_utc"], utc=True)

    import json
    with open(os.path.join(DATA_DIR, "province_station_registry.json"), encoding="utf-8") as f:
        registry = json.load(f)
    provinces_meta = {p["province_en"]: p for p in registry["provinces"]}

    print("\nคำนวณ feature ทุก config (radius x window):")
    feature_cols = compute_all_configs(
        df_all, provinces_meta, hotspots["latitude"].values, hotspots["longitude"].values,
        hotspots["acq_datetime_utc"].values,
    )

    df = df_all[~df_all["is_zero_shot"]].reset_index(drop=True)
    train_mod = load_train_module()
    train_df, _, test_df = train_mod.split_train_calib_test(df)
    print(f"\ntrain={len(train_df)}  test={len(test_df)}")

    baseline_features = train_mod.FEATURE_COLS
    all_rows = []
    for kind in ("lgb", "xgb"):
        print(f"\n=== {kind.upper()} — baseline ===")
        all_rows += eval_variant(train_df, test_df, baseline_features, kind, "baseline", train_mod)
        for col in feature_cols:
            print(f"\n=== {kind.upper()} — {col} ===")
            all_rows += eval_variant(train_df, test_df, baseline_features + [col], kind, col, train_mod)

    result_df = pd.DataFrame(all_rows)
    out_path = os.path.join(MODEL_DIR, f"fire_hotspot_hyperparam_tuning_{TIMESTAMP}.csv")
    result_df.to_csv(out_path, index=False)
    print(f"\nบันทึกผลดิบที่: {out_path}")

    print("\n" + "=" * 100)
    print("สรุป: MAE เฉลี่ยข้าม 5 horizon ตัวแทน ต่อ config (delta% เทียบ baseline)")
    print("=" * 100)
    for metric in ("mae", "mae_burning", "mae_gt_p90"):
        summary = result_df.groupby(["model", "variant"])[metric].mean().unstack("model")
        for kind in ("lgb", "xgb"):
            baseline_val = summary.loc["baseline", kind]
            summary[f"{kind}_delta_pct"] = (summary[kind] - baseline_val) / baseline_val * 100
        print(f"\n--- {metric} ---")
        print(summary.round(3).to_string())


if __name__ == "__main__":
    main()

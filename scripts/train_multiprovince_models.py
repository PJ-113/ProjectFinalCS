import os
import pickle
from datetime import datetime as dt

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(
    _ROOT, "data", "multiprovince_pm25_features_raw_spatial_k1_firehotspot_nbtrend_20260919_174805.csv"
)
MODEL_DIR = os.path.join(_ROOT, "models")
TIMESTAMP = dt.now().strftime("%Y%m%d_%H%M%S")

QUANTILES = [0.05, 0.5, 0.95]
ALPHA = 0.10
N_ESTIMATORS = 300
TUNED_PARAMS_LGB = dict(learning_rate=0.05, num_leaves=63, min_child_samples=50, reg_lambda=1.0, n_estimators=400)
TUNED_PARAMS_XGB = dict(learning_rate=0.03, max_depth=8, min_child_weight=20, reg_lambda=10.0, n_estimators=800)
BAGGING_PARAMS_LGB = dict(subsample=0.8, subsample_freq=1, colsample_bytree=0.8)
BAGGING_PARAMS_XGB = dict(subsample=0.8, colsample_bytree=0.8)

FEATURE_COLS = [
    "temp", "humidity", "wind_speed", "precipitation",
    "pm25_lag_1", "pm25_lag_3", "pm25_lag_6", "pm25_lag_12", "pm25_lag_24",
    "pm25_roll_mean_6", "pm25_roll_std_6", "pm25_roll_mean_24", "pm25_roll_std_24",
    "wind_dir_sin", "wind_dir_cos", "pressure", "lat", "lon",
    "fire_hotspot_count_100km_24h",
    "nb_trend_12h",
]
assert len(FEATURE_COLS) == 20
TARGET_COLS = [f"pm25_t+{h}" for h in range(1, 25)]


def split_train_calib_test(df, fracs=(0.65, 0.15, 0.20)):
    trains, calibs, tests = [], [], []
    for _, g in df.groupby("province_en", sort=False):
        g = g.sort_values("datetime_utc").reset_index(drop=True)
        n = len(g)
        n_train = int(n * fracs[0])
        n_calib = int(n * fracs[1])
        trains.append(g.iloc[:n_train])
        calibs.append(g.iloc[n_train:n_train + n_calib])
        tests.append(g.iloc[n_train + n_calib:])
    return (pd.concat(trains, ignore_index=True),
            pd.concat(calibs, ignore_index=True),
            pd.concat(tests, ignore_index=True))


def fit_lgb_quantile(X, y, alpha):
    model = lgb.LGBMRegressor(
        objective="quantile", alpha=alpha,
        random_state=42, verbose=-1,
        **BAGGING_PARAMS_LGB, **TUNED_PARAMS_LGB,
    )
    model.fit(X, y)
    return model


def fit_xgb_quantile(X, y, alpha):
    model = xgb.XGBRegressor(
        objective="reg:quantileerror", quantile_alpha=alpha,
        tree_method="hist",
        random_state=42, verbosity=0,
        **BAGGING_PARAMS_XGB, **TUNED_PARAMS_XGB,
    )
    model.fit(X, y)
    return model


def conformalize(lo, hi, y_true, alpha=ALPHA):
    scores = np.maximum(lo - y_true, y_true - hi)
    n = len(scores)
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, q_level))


def coverage(lo, hi, y_true):
    return float(((y_true >= lo) & (y_true <= hi)).mean())


def main():
    print(f"โหลดข้อมูล: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    df = df[~df["is_zero_shot"]].reset_index(drop=True)
    provinces_used = sorted(df["province_en"].unique().tolist())
    print(f"จังหวัดที่ใช้เทรน ({len(provinces_used)}): {provinces_used}")
    print(f"แถวรวม (17 จังหวัด, ไม่รวม zero-shot): {len(df)}")

    train_df, calib_df, test_df = split_train_calib_test(df)
    print(f"train={len(train_df)}  calibration={len(calib_df)}  test={len(test_df)}")

    results = {"lgb": {"models": {}, "eval": []}, "xgb": {"models": {}, "eval": []}}
    conformal_rows = {"lgb": [], "xgb": []}

    for h in range(1, 25):
        target_col = f"pm25_t+{h}"
        train_h = train_df.dropna(subset=[target_col])
        calib_h = calib_df.dropna(subset=[target_col])
        test_h = test_df.dropna(subset=[target_col])

        X_tr = train_h[FEATURE_COLS]
        y_tr_resid = train_h[target_col] - train_h["pm25"]

        X_calib, calib_current = calib_h[FEATURE_COLS], calib_h["pm25"].values
        y_calib_actual = calib_h[target_col].values

        X_test, test_current = test_h[FEATURE_COLS], test_h["pm25"].values
        y_test_actual = test_h[target_col].values

        lgb_q, xgb_q = {}, {}
        for q in QUANTILES:
            lgb_q[q] = fit_lgb_quantile(X_tr, y_tr_resid, q)
            xgb_q[q] = fit_xgb_quantile(X_tr, y_tr_resid, q)

        for tag, models_by_q in (("lgb", lgb_q), ("xgb", xgb_q)):
            results[tag]["models"][h] = models_by_q

            p50_test = models_by_q[0.5].predict(X_test) + test_current
            mae_p50 = mean_absolute_error(y_test_actual, p50_test)
            r2_p50 = r2_score(y_test_actual, p50_test)

            lo_calib = models_by_q[0.05].predict(X_calib) + calib_current
            hi_calib = models_by_q[0.95].predict(X_calib) + calib_current
            q_hat = conformalize(lo_calib, hi_calib, y_calib_actual)

            lo_test = models_by_q[0.05].predict(X_test) + test_current
            hi_test = models_by_q[0.95].predict(X_test) + test_current
            cov_before = coverage(lo_test, hi_test, y_test_actual)
            cov_after = coverage(lo_test - q_hat, hi_test + q_hat, y_test_actual)

            results[tag]["eval"].append({
                "horizon": h, "mae_p50": mae_p50, "r2_p50": r2_p50,
                "coverage_before": cov_before, "coverage_after": cov_after, "q_hat": q_hat,
                "n_train": len(train_h), "n_calib": len(calib_h), "n_test": len(test_h),
            })
            conformal_rows[tag].append({"horizon": h, "q_hat": q_hat,
                                         "coverage_before": cov_before, "coverage_after": cov_after})

        print(f"t+{h:2d}  lgb: MAE={results['lgb']['eval'][-1]['mae_p50']:.3f} "
              f"cov {results['lgb']['eval'][-1]['coverage_before']:.1%}->{results['lgb']['eval'][-1]['coverage_after']:.1%}"
              f"  |  xgb: MAE={results['xgb']['eval'][-1]['mae_p50']:.3f} "
              f"cov {results['xgb']['eval'][-1]['coverage_before']:.1%}->{results['xgb']['eval'][-1]['coverage_after']:.1%}")

    summary_rows = []
    for tag in ("lgb", "xgb"):
        eval_df = pd.DataFrame(results[tag]["eval"])
        mean_mae = eval_df["mae_p50"].mean()
        mean_r2 = eval_df["r2_p50"].mean()
        mean_cov_before = eval_df["coverage_before"].mean()
        mean_cov_after = eval_df["coverage_after"].mean()
        summary_rows.append({"model": tag, "mae_p50_mean_24h": mean_mae, "r2_p50_mean_24h": mean_r2,
                              "coverage_before_mean": mean_cov_before, "coverage_after_mean": mean_cov_after})
        print(f"\n=== {tag.upper()} สรุป 24h ===")
        print(f"MAE(P50) เฉลี่ย: {mean_mae:.3f}")
        print(f"R2(P50) เฉลี่ย: {mean_r2:.3f}")
        print(f"Coverage: {mean_cov_before:.1%} -> {mean_cov_after:.1%} (เป้าหมาย 90%)")

        model_package = {
            "models": results[tag]["models"], "feature_cols": FEATURE_COLS, "target_cols": TARGET_COLS,
            "quantiles": QUANTILES, "trained_at": TIMESTAMP, "provinces": provinces_used,
            "technique": "residual forecasting (target=pm25_t+h - pm25_now) + per-tree bagging "
                         "(subsample/colsample) + quantile regression (P05/P50/P95) + conformal calibration",
            "conformal_alpha": ALPHA,
            "eval_by_horizon": results[tag]["eval"],
        }
        model_path = os.path.join(MODEL_DIR, f"{tag}_pm25_multiprovince_residual_bagged_quantile_{TIMESTAMP}.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model_package, f)
        print(f"บันทึกโมเดลที่: {model_path}")

        pd.DataFrame(conformal_rows[tag]).to_csv(
            os.path.join(MODEL_DIR, f"conformal_calibration_{tag}_{TIMESTAMP}.csv"), index=False)

    pd.DataFrame(summary_rows).to_csv(
        os.path.join(MODEL_DIR, f"multiprovince_eval_summary_{TIMESTAMP}.csv"), index=False)

    print("\n" + "=" * 70)
    print("เทียบกับตัวเลขที่ผู้ใช้รายงานไว้:")
    print("  LightGBM: MAE(P50)=2.912, coverage 86.7%->89.5%")
    print("  XGBoost : MAE(P50)=2.993, coverage 83.1%->88.9%")


if __name__ == "__main__":
    main()

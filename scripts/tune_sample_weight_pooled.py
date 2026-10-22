"""
Sample weighting สำหรับแถวฝุ่นสูง (pm25_t+h > HIGH_THRESHOLD): binary w in {1,2,3,5} และ continuous 1+y/37.5
วัด MAE รวม/ฤดูฝุ่น/>p90 บน calib split + bias (pred-actual) ที่ <=p50 และ >p90
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

import importlib.util

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(_ROOT, "models")
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
HORIZONS = [1, 6, 12, 18, 24]
BURNING_MONTHS = {2, 3, 4}
HIGH_THRESHOLD = 37.5

BEST = {
    "lgb": {"learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 50, "reg_lambda": 1.0, "n_estimators": 400},
    "xgb": {"learning_rate": 0.03, "max_depth": 8, "min_child_weight": 20, "reg_lambda": 10.0, "n_estimators": 800},
}
VARIANTS = ["w1", "w2", "w3", "w5", "cont"]


def load_train_module():
    spec = importlib.util.spec_from_file_location(
        "train_multiprovince_models", os.path.join(_ROOT, "scripts", "train_multiprovince_models.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def make_weights(y_target, variant):
    if variant == "cont":
        return 1.0 + y_target / HIGH_THRESHOLD
    w = float(variant[1:])
    return np.where(y_target > HIGH_THRESHOLD, w, 1.0)


def fit_p50(X, y, kind, params, weights, train_mod):
    if kind == "lgb":
        import lightgbm as lgb
        return lgb.LGBMRegressor(objective="quantile", alpha=0.5, random_state=42, verbose=-1,
                                  **train_mod.BAGGING_PARAMS_LGB, **params).fit(X, y, sample_weight=weights)
    import xgboost as xgb
    return xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, tree_method="hist",
                             random_state=42, verbosity=0, **train_mod.BAGGING_PARAMS_XGB,
                             **params).fit(X, y, sample_weight=weights)


def main():
    train_mod = load_train_module()
    df = pd.read_csv(train_mod.DATA_PATH)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    df = df[~df["is_zero_shot"]].reset_index(drop=True)
    train_df, calib_df, _ = train_mod.split_train_calib_test(df)
    feats = train_mod.FEATURE_COLS
    print(f"train={len(train_df)} calib={len(calib_df)}  (selection บน calib เท่านั้น)")

    rows = []
    for kind in ("lgb", "xgb"):
        for variant in VARIANTS:
            for h in HORIZONS:
                tcol = f"pm25_t+{h}"
                tr = train_df.dropna(subset=[tcol])
                ev = calib_df.dropna(subset=[tcol])
                w = make_weights(tr[tcol].values, variant)
                model = fit_p50(tr[feats], tr[tcol] - tr["pm25"], kind, BEST[kind], w, train_mod)
                pred = model.predict(ev[feats]) + ev["pm25"].values
                y = ev[tcol].values
                burning = pd.to_datetime(ev["datetime_utc"]).dt.month.isin(BURNING_MONTHS).values
                hi, lo = y > np.percentile(y, 90), y <= np.percentile(y, 50)
                rows.append({"model": kind, "variant": variant, "horizon": h,
                             "mae": mean_absolute_error(y, pred),
                             "mae_burning": mean_absolute_error(y[burning], pred[burning]),
                             "mae_gt_p90": mean_absolute_error(y[hi], pred[hi]),
                             "bias_gt_p90": float((pred[hi] - y[hi]).mean()),
                             "bias_le_p50": float((pred[lo] - y[lo]).mean())})
            r = pd.DataFrame([x for x in rows if x["model"] == kind and x["variant"] == variant])
            print(f"  [{kind}/{variant}] MAE={r['mae'].mean():.3f} burning={r['mae_burning'].mean():.3f} "
                  f">p90={r['mae_gt_p90'].mean():.3f} bias>p90={r['bias_gt_p90'].mean():+.2f} "
                  f"bias<=p50={r['bias_le_p50'].mean():+.2f}", flush=True)

    out = pd.DataFrame(rows)
    out_path = os.path.join(MODEL_DIR, f"tune_sample_weight_pooled_{TIMESTAMP}.csv")
    out.to_csv(out_path, index=False)
    print(f"\nบันทึกผลที่: {out_path}")
    print("\n" + "=" * 90 + "\nสรุป (delta% เทียบ w1 = ไม่ถ่วงน้ำหนัก)\n" + "=" * 90)
    for kind in ("lgb", "xgb"):
        g = out[out["model"] == kind].groupby("variant")[["mae", "mae_burning", "mae_gt_p90", "bias_gt_p90", "bias_le_p50"]].mean()
        base = g.loc["w1"]
        for v in VARIANTS:
            d = {k: (g.loc[v, k] - base[k]) / base[k] * 100 for k in ("mae", "mae_burning", "mae_gt_p90")}
            print(f"  {kind} {v:5s} MAE {d['mae']:+.2f}%  burning {d['mae_burning']:+.2f}%  >p90 {d['mae_gt_p90']:+.2f}%  "
                  f"bias>p90={g.loc[v, 'bias_gt_p90']:+.2f} bias<=p50={g.loc[v, 'bias_le_p50']:+.2f}")


if __name__ == "__main__":
    main()

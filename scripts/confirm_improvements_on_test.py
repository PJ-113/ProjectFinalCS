"""
Phase D ขั้นยืนยัน — แตะ test split ครั้งเดียวเพื่อยืนยันผลรวมของสิ่งที่ผ่านการเลือกบน calib (A1 hyperparameter + nb_trend_12h)
เทียบ 3 variant (P50, 5 horizon ตัวแทน, LGB+XGB), วัด 4 มุม (รวม/ปกติ/ฤดูฝุ่น/>p90):
  ref     = โมเดลปัจจุบัน (20 feature, default hyperparameter)
  tuned   = 20 feature + hyperparameter จูนแล้ว (แยกผลของ A1)
  tuned+nb= 21 feature (+nb_trend_12h) + hyperparameter จูนแล้ว (candidate สุดท้าย)
"""
import importlib.util
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(_ROOT, "models")
DATA = os.path.join(_ROOT, "data", "multiprovince_pm25_features_raw_spatial_k1_firehotspot_nbtrend_20260919_135009.csv")
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
HORIZONS = [1, 6, 12, 18, 24]
BURNING = {2, 3, 4}
TUNED = {
    "lgb": {"learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 50, "reg_lambda": 1.0, "n_estimators": 400},
    "xgb": {"learning_rate": 0.03, "max_depth": 8, "min_child_weight": 20, "reg_lambda": 10.0, "n_estimators": 800},
}
DEFAULT = {
    "lgb": {"n_estimators": 300},
    "xgb": {"n_estimators": 300},
}


def fit_p50(X, y, kind, params, tm):
    if kind == "lgb":
        import lightgbm as lgb
        return lgb.LGBMRegressor(objective="quantile", alpha=0.5, random_state=42, verbose=-1,
                                  **tm.BAGGING_PARAMS_LGB, **params).fit(X, y)
    import xgboost as xgb
    return xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, tree_method="hist",
                             random_state=42, verbosity=0, **tm.BAGGING_PARAMS_XGB, **params).fit(X, y)


def main():
    spec = importlib.util.spec_from_file_location(
        "tm", os.path.join(_ROOT, "scripts", "train_multiprovince_models.py"))
    tm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tm)
    df = pd.read_csv(DATA)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    df = df[~df["is_zero_shot"]].reset_index(drop=True)
    train_df, _, test_df = tm.split_train_calib_test(df)
    base = tm.FEATURE_COLS
    variants = {
        "ref": (base, DEFAULT),
        "tuned": (base, TUNED),
        "tuned+nb": (base + ["nb_trend_12h"], TUNED),
    }
    rows = []
    for kind in ("lgb", "xgb"):
        for name, (feats, params) in variants.items():
            for h in HORIZONS:
                tcol = f"pm25_t+{h}"
                tr, te = train_df.dropna(subset=[tcol]), test_df.dropna(subset=[tcol])
                model = fit_p50(tr[feats], tr[tcol] - tr["pm25"], kind, params[kind], tm)
                pred = model.predict(te[feats]) + te["pm25"].values
                y = te[tcol].values
                burning = pd.to_datetime(te["datetime_utc"]).dt.month.isin(BURNING).values
                hi = y > np.percentile(y, 90)
                rows.append({"model": kind, "variant": name, "horizon": h,
                             "mae": mean_absolute_error(y, pred),
                             "mae_normal": mean_absolute_error(y[~burning], pred[~burning]),
                             "mae_burning": mean_absolute_error(y[burning], pred[burning]),
                             "mae_gt_p90": mean_absolute_error(y[hi], pred[hi])})
            print(f"  [{kind}/{name}] เสร็จ", flush=True)
    res = pd.DataFrame(rows)
    out_path = os.path.join(MODEL_DIR, f"confirm_improvements_on_test_{TIMESTAMP}.csv")
    res.to_csv(out_path, index=False)
    print(f"\nบันทึกผลที่: {out_path}")
    print("\n" + "=" * 90 + "\nTEST SET: MAE เฉลี่ย 5 horizon (delta% เทียบ ref)\n" + "=" * 90)
    for kind in ("lgb", "xgb"):
        g = res[res["model"] == kind].groupby("variant")[["mae", "mae_normal", "mae_burning", "mae_gt_p90"]].mean()
        print(f"\n{kind.upper()}")
        for v in ("ref", "tuned", "tuned+nb"):
            d = ((g.loc[v] - g.loc["ref"]) / g.loc["ref"] * 100)
            print(f"  {v:9s} MAE={g.loc[v,'mae']:.3f} ({d['mae']:+.2f}%)  ปกติ={g.loc[v,'mae_normal']:.3f} ({d['mae_normal']:+.2f}%)  "
                  f"ฤดูฝุ่น={g.loc[v,'mae_burning']:.3f} ({d['mae_burning']:+.2f}%)  >p90={g.loc[v,'mae_gt_p90']:.3f} ({d['mae_gt_p90']:+.2f}%)")
    print("\nต่อ horizon (tuned+nb เทียบ ref, MAE):")
    for kind in ("lgb", "xgb"):
        p = res[res["model"] == kind].pivot(index="horizon", columns="variant", values="mae")
        p["d%"] = (p["tuned+nb"] - p["ref"]) / p["ref"] * 100
        print(f"\n{kind.upper()}\n{p.round(3).to_string()}")


if __name__ == "__main__":
    main()

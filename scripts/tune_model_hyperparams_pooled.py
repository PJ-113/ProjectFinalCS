"""
Phase A1 — hyperparameter tuning ของ pooled LGB/XGB (P50 only) ที่ไม่เคยทำมาก่อน
(fit_lgb_quantile/fit_xgb_quantile ใน train_multiprovince_models.py ใช้แค่ default + n_estimators=300 + bagging 0.8)

วิธี: random search N_CONFIGS config/โมเดล เทรนบน train แล้ว "วัดบน calib split" (ไม่แตะ test เพื่อกัน overfit
จากการลองหลาย config) ที่ SEARCH_HORIZONS แล้วนำ top-K ไปยืนยันที่ CONFIRM_HORIZONS
วัด 4 มุมเหมือน tune_fire_hotspot_hyperparams.py (รวม / ฤดูฝุ่น ก.พ.-เม.ย. / >p90) — baseline = config ปัจจุบันของ train_multiprovince_models.py
"""
import importlib.util
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(_ROOT, "models")
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
SEARCH_HORIZONS = [1, 12, 24]
CONFIRM_HORIZONS = [1, 6, 12, 18, 24]
BURNING_MONTHS = {2, 3, 4}
N_CONFIGS = 20
TOP_K = 3
RNG_SEED = 2026

LGB_SPACE = {
    "learning_rate": [0.03, 0.05, 0.1],
    "num_leaves": [15, 31, 63],
    "min_child_samples": [20, 50, 100],
    "reg_lambda": [0.0, 1.0, 10.0],
    "n_estimators": [200, 400, 800],
}
XGB_SPACE = {
    "learning_rate": [0.03, 0.05, 0.1, 0.3],
    "max_depth": [4, 6, 8],
    "min_child_weight": [1, 5, 20],
    "reg_lambda": [1.0, 10.0, 30.0],
    "n_estimators": [200, 400, 800],
}
BASELINES = {
    "lgb": {"learning_rate": 0.1, "num_leaves": 31, "min_child_samples": 20, "reg_lambda": 0.0, "n_estimators": 300},
    "xgb": {"learning_rate": 0.3, "max_depth": 6, "min_child_weight": 1, "reg_lambda": 1.0, "n_estimators": 300},
}


def load_train_module():
    spec = importlib.util.spec_from_file_location(
        "train_multiprovince_models", os.path.join(_ROOT, "scripts", "train_multiprovince_models.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fit_p50(X, y, kind, params, train_mod):
    if kind == "lgb":
        import lightgbm as lgb
        return lgb.LGBMRegressor(objective="quantile", alpha=0.5, random_state=42, verbose=-1,
                                  **train_mod.BAGGING_PARAMS_LGB, **params).fit(X, y)
    import xgboost as xgb
    return xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, tree_method="hist",
                             random_state=42, verbosity=0, **train_mod.BAGGING_PARAMS_XGB, **params).fit(X, y)


def eval_config(train_df, eval_df, features, kind, params, horizons, train_mod):
    rows = []
    for h in horizons:
        target_col = f"pm25_t+{h}"
        tr = train_df.dropna(subset=[target_col])
        ev = eval_df.dropna(subset=[target_col])
        model = fit_p50(tr[features], tr[target_col] - tr["pm25"], kind, params, train_mod)
        pred = model.predict(ev[features]) + ev["pm25"].values
        y = ev[target_col].values
        is_burning = pd.to_datetime(ev["datetime_utc"]).dt.month.isin(BURNING_MONTHS).values
        p90 = np.percentile(y, 90)
        hi = y > p90
        rows.append({"horizon": h, "mae": mean_absolute_error(y, pred),
                     "mae_burning": mean_absolute_error(y[is_burning], pred[is_burning]) if is_burning.sum() else np.nan,
                     "mae_gt_p90": mean_absolute_error(y[hi], pred[hi]) if hi.sum() else np.nan})
    r = pd.DataFrame(rows)
    return {"mae": r["mae"].mean(), "mae_burning": r["mae_burning"].mean(), "mae_gt_p90": r["mae_gt_p90"].mean()}


def sample_configs(space, n, rng):
    seen, out = set(), []
    while len(out) < n:
        cfg = {k: v[rng.integers(len(v))] for k, v in space.items()}
        key = tuple(sorted(cfg.items()))
        if key not in seen:
            seen.add(key)
            out.append(cfg)
    return out


def main():
    train_mod = load_train_module()
    df = pd.read_csv(train_mod.DATA_PATH)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    df = df[~df["is_zero_shot"]].reset_index(drop=True)
    train_df, calib_df, _ = train_mod.split_train_calib_test(df)
    features = train_mod.FEATURE_COLS
    print(f"train={len(train_df)}  calib={len(calib_df)}  (selection บน calib เท่านั้น ไม่แตะ test)")

    rng = np.random.default_rng(RNG_SEED)
    all_rows = []
    for kind, space in (("lgb", LGB_SPACE), ("xgb", XGB_SPACE)):
        configs = [("baseline", BASELINES[kind])] + [(f"cfg{i:02d}", c) for i, c in enumerate(sample_configs(space, N_CONFIGS, rng))]
        stage1 = []
        for name, params in configs:
            m = eval_config(train_df, calib_df, features, kind, params, SEARCH_HORIZONS, train_mod)
            print(f"  [{kind}/{name}] {params} -> MAE={m['mae']:.3f} >p90={m['mae_gt_p90']:.3f}", flush=True)
            stage1.append((name, params, m))
            all_rows.append({"model": kind, "stage": "search", "config": name, **params, **m})

        base_mae = stage1[0][2]["mae"]
        ranked = sorted(stage1[1:], key=lambda t: t[2]["mae"])[:TOP_K]
        print(f"\n--- {kind.upper()} ยืนยัน baseline + top-{TOP_K} ที่ {CONFIRM_HORIZONS} (baseline search MAE={base_mae:.3f}) ---")
        for name, params, _ in [stage1[0]] + ranked:
            m = eval_config(train_df, calib_df, features, kind, params, CONFIRM_HORIZONS, train_mod)
            print(f"  [{kind}/{name}] confirm MAE={m['mae']:.3f} burning={m['mae_burning']:.3f} >p90={m['mae_gt_p90']:.3f}", flush=True)
            all_rows.append({"model": kind, "stage": "confirm", "config": name, **params, **m})

    out = pd.DataFrame(all_rows)
    out_path = os.path.join(MODEL_DIR, f"tune_model_hyperparams_pooled_{TIMESTAMP}.csv")
    out.to_csv(out_path, index=False)
    print(f"\nบันทึกผลที่: {out_path}")

    print("\n" + "=" * 90 + "\nสรุป confirm stage (delta% เทียบ baseline ของแต่ละโมเดล)\n" + "=" * 90)
    for kind in ("lgb", "xgb"):
        c = out[(out["model"] == kind) & (out["stage"] == "confirm")].reset_index(drop=True)
        base = c[c["config"] == "baseline"].iloc[0]
        for _, r in c.iterrows():
            d = {k: (r[k] - base[k]) / base[k] * 100 for k in ("mae", "mae_burning", "mae_gt_p90")}
            print(f"  {kind} {r['config']:9s} MAE={r['mae']:.3f} ({d['mae']:+.2f}%)  burning ({d['mae_burning']:+.2f}%)  "
                  f">p90 ({d['mae_gt_p90']:+.2f}%)")


if __name__ == "__main__":
    main()

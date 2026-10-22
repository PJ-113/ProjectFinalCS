"""
Ablation study: ตัด feature ทีละกลุ่ม (leave-one-group-out) วัดผลต่อ MAE/R² (P50 only)
ที่ horizon ตัวแทน 5 จุด (t+1, t+6, t+12, t+18, t+24)
"""
import importlib.util
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = None  # ตั้งใน main()
MODEL_DIR = os.path.join(_ROOT, "models")

N_ESTIMATORS = 300
BAGGING_PARAMS_LGB = dict(subsample=0.8, subsample_freq=1, colsample_bytree=0.8)
BAGGING_PARAMS_XGB = dict(subsample=0.8, colsample_bytree=0.8)
REPRESENTATIVE_HORIZONS = [1, 6, 12, 18, 24]

FEATURE_GROUPS = {
    "pm25_lag_rolling": ["pm25_lag_1", "pm25_lag_3", "pm25_lag_6", "pm25_lag_12", "pm25_lag_24",
                          "pm25_roll_mean_6", "pm25_roll_std_6", "pm25_roll_mean_24", "pm25_roll_std_24"],
    "weather": ["temp", "humidity", "wind_speed", "precipitation"],
    "wind_pressure": ["wind_dir_sin", "wind_dir_cos", "pressure"],
    "lat_lon": ["lat", "lon"],
    "spatial_lag": ["nb_trend_12h"],
    "fire_hotspot": ["fire_hotspot_count_100km_24h"],
}

HYPOTHESES = {
    "pm25_lag_rolling": "กลุ่มสำคัญที่สุด (autocorrelation ของมลพิษ) — ตัดออกควร MAE แย่ลงมากที่สุด",
    "weather": "ช่วยเสริมพอสมควร โดยเฉพาะ precipitation (scavenging effect)",
    "wind_pressure": "feature ใหม่ล่าสุด ช่วยเรื่อง advection/inversion แต่ผลควรเล็กกว่า weather/lag",
    "spatial_lag": "กลุ่มสำคัญสูง (ข้อมูลจังหวัดข้างเคียง + แนวโน้ม) — ตัดออกควรแย่ลงชัด",
    "fire_hotspot": "ช่วยเฉพาะจุดช่วงเผา ผลรวมควรเล็ก",
    "lat_lon": "จำเป็นสำหรับ zero-shot generalization แต่ผลต่อ MAE ใน-training อาจน้อยที่สุด",
}


def load_train_module():
    spec = importlib.util.spec_from_file_location(
        "train_multiprovince_models", os.path.join(_ROOT, "scripts", "train_multiprovince_models.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fit_p50(X, y, kind):
    if kind == "lgb":
        model = lgb.LGBMRegressor(objective="quantile", alpha=0.5, random_state=42, verbose=-1, **BAGGING_PARAMS_LGB, **TUNED_LGB)
    else:
        model = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, tree_method="hist",
                                  random_state=42, verbosity=0, **BAGGING_PARAMS_XGB, **TUNED_XGB)
    model.fit(X, y)
    return model


def eval_variant(train_df, test_df, feature_cols, kind, variant_name):
    rows = []
    for h in REPRESENTATIVE_HORIZONS:
        target_col = f"pm25_t+{h}"
        tr = train_df.dropna(subset=[target_col])
        te = test_df.dropna(subset=[target_col])
        X_tr, y_tr_resid = tr[feature_cols], tr[target_col] - tr["pm25"]
        X_te, current_te = te[feature_cols], te["pm25"].values
        y_te_actual = te[target_col].values

        model = fit_p50(X_tr, y_tr_resid, kind)
        pred = model.predict(X_te) + current_te
        mae = mean_absolute_error(y_te_actual, pred)
        r2 = r2_score(y_te_actual, pred)
        rows.append({"variant": variant_name, "model": kind, "horizon": h, "mae": mae, "r2": r2, "n": len(te)})
        print(f"  [{variant_name}/{kind}] t+{h:2d}  MAE={mae:.3f}  R2={r2:.3f}  (n_features={len(feature_cols)})")
    return rows


def main():
    global TUNED_LGB, TUNED_XGB
    train_mod = load_train_module()
    TUNED_LGB, TUNED_XGB = train_mod.TUNED_PARAMS_LGB, train_mod.TUNED_PARAMS_XGB
    data_path = train_mod.DATA_PATH
    print(f"โหลดข้อมูล: {data_path}")
    df = pd.read_csv(data_path)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    df = df[~df["is_zero_shot"]].reset_index(drop=True)

    train_df, _, test_df = train_mod.split_train_calib_test(df)
    print(f"train={len(train_df)}  test={len(test_df)}  (split เดียวกับ task A / CHANGELOG 75)")

    all_features = train_mod.FEATURE_COLS
    assert sum(len(v) for v in FEATURE_GROUPS.values()) == len(all_features), "feature group ไม่ครบ"

    print("\nสมมติฐานก่อนทดสอบ:")
    for g, hyp in HYPOTHESES.items():
        print(f"  - {g}: {hyp}")

    all_rows = []
    for kind in ("lgb", "xgb"):
        print(f"\n=== {kind.upper()} — baseline (feature ครบทุกตัว) ===")
        all_rows += eval_variant(train_df, test_df, all_features, kind, "baseline")

        for group_name, group_cols in FEATURE_GROUPS.items():
            reduced = [c for c in all_features if c not in group_cols]
            print(f"\n=== {kind.upper()} — leave-out: {group_name} (เหลือ {len(reduced)} feature) ===")
            all_rows += eval_variant(train_df, test_df, reduced, kind, f"without_{group_name}")

    result_df = pd.DataFrame(all_rows)
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(MODEL_DIR, f"ablation_feature_groups_clean_{timestamp}.csv")
    result_df.to_csv(out_path, index=False)
    print(f"\nบันทึกผลดิบแล้วที่: {out_path}")

    print("\n" + "=" * 90)
    print("สรุป: MAE เฉลี่ยข้าม 5 horizon ตัวแทน ต่อ variant (เทียบ delta จาก baseline)")
    print("=" * 90)
    summary = result_df.groupby(["model", "variant"])["mae"].mean().reset_index()
    for kind in ("lgb", "xgb"):
        sub = summary[summary["model"] == kind].set_index("variant")
        baseline_mae = sub.loc["baseline", "mae"]
        print(f"\n{kind.upper()}  (baseline MAE = {baseline_mae:.3f})")
        for variant in sub.index:
            if variant == "baseline":
                continue
            mae = sub.loc[variant, "mae"]
            delta_pct = (mae - baseline_mae) / baseline_mae * 100
            group = variant.replace("without_", "")
            print(f"  ตัด {group:20s} -> MAE={mae:.3f}  (delta {delta_pct:+.1f}%)  "
                  f"[สมมติฐาน: {HYPOTHESES[group]}]")


if __name__ == "__main__":
    main()

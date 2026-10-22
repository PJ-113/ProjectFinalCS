"""
Validate elevation_m เป็น model feature: เทียบ baseline (19 feature ปัจจุบัน) vs
+elevation_m (20 feature) ด้วย MAE/R² (P50-only) ที่ horizon ตัวแทน 5 จุด
"""
import importlib.util
import json
import os

import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_PATH = os.path.join(_ROOT, "data", "province_station_registry.json")
MODEL_DIR = os.path.join(_ROOT, "models")
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
REPRESENTATIVE_HORIZONS = [1, 6, 12, 18, 24]


def load_train_module():
    spec = importlib.util.spec_from_file_location(
        "train_multiprovince_models", os.path.join(_ROOT, "scripts", "train_multiprovince_models.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


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
        mae = mean_absolute_error(y_te_actual, pred)
        r2 = r2_score(y_te_actual, pred)
        rows.append({"variant": variant_name, "model": kind, "horizon": h, "mae": mae, "r2": r2, "n": len(te)})
        print(f"  [{variant_name}/{kind}] t+{h:2d}  MAE={mae:.3f}  R2={r2:.3f}  (n_features={len(feature_cols)})")
    return rows


def main():
    train_mod = load_train_module()

    with open(REGISTRY_PATH, encoding="utf-8") as f:
        registry = json.load(f)
    elevation_by_province = {p["province_en"]: p["elevation_m"] for p in registry["provinces"]}

    print(f"โหลดข้อมูล: {train_mod.DATA_PATH}")
    df = pd.read_csv(train_mod.DATA_PATH)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    df = df[~df["is_zero_shot"]].reset_index(drop=True)
    df["elevation_m"] = df["province_en"].map(elevation_by_province)
    assert df["elevation_m"].isna().sum() == 0, "มีแถวที่ map elevation ไม่ได้"

    train_df, _, test_df = train_mod.split_train_calib_test(df)
    print(f"train={len(train_df)}  test={len(test_df)}")

    baseline_features = train_mod.FEATURE_COLS  # 19 feature ปัจจุบัน
    extended_features = baseline_features + ["elevation_m"]  # 20 feature

    all_rows = []
    for kind in ("lgb", "xgb"):
        print(f"\n=== {kind.upper()} — baseline (19 feature, ไม่มี elevation) ===")
        all_rows += eval_variant(train_df, test_df, baseline_features, kind, "baseline", train_mod)
        print(f"\n=== {kind.upper()} — with_elevation (20 feature) ===")
        all_rows += eval_variant(train_df, test_df, extended_features, kind, "with_elevation", train_mod)

    result_df = pd.DataFrame(all_rows)
    out_path = os.path.join(MODEL_DIR, f"elevation_feature_validation_{TIMESTAMP}.csv")
    result_df.to_csv(out_path, index=False)
    print(f"\nบันทึกผลดิบที่: {out_path}")

    print("\n" + "=" * 80)
    print("สรุป: MAE เฉลี่ยข้าม 5 horizon ตัวแทน")
    print("=" * 80)
    summary = result_df.groupby(["model", "variant"])["mae"].mean().reset_index()
    verdicts = {}
    for kind in ("lgb", "xgb"):
        sub = summary[summary["model"] == kind].set_index("variant")
        baseline_mae = sub.loc["baseline", "mae"]
        with_mae = sub.loc["with_elevation", "mae"]
        delta_pct = (with_mae - baseline_mae) / baseline_mae * 100
        verdicts[kind] = delta_pct
        print(f"\n{kind.upper()}: baseline={baseline_mae:.3f}  with_elevation={with_mae:.3f}  delta={delta_pct:+.2f}%")

    print("\n" + "=" * 80)
    if all(v < -0.3 for v in verdicts.values()):
        print("สรุป: MAE ดีขึ้นทั้ง 2 โมเดล -> คุ้มที่จะ sync feature formula 4 จุด + retrain เต็มรูปแบบ")
    elif all(v > 0.3 for v in verdicts.values()):
        print("สรุป: MAE แย่ลงทั้ง 2 โมเดล -> ไม่แนะนำเพิ่ม elevation เป็น feature")
    else:
        print("สรุป: ผลไม่ชัดเจน/ไม่ตรงกันระหว่าง 2 โมเดล (หรือผลเล็กน้อยมาก) -> ไม่คุ้มลงทุน retrain เต็ม "
              "สำหรับ improvement ขนาดนี้")


if __name__ == "__main__":
    main()

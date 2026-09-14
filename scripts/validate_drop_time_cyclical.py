"""
ตรวจซ้ำ finding จาก task D (time_cyclical ตัดออกแล้ว MAE ดีขึ้น) บนโมเดล 24-feature ที่มี
pm25_neighbor_avg แล้ว (task F) — D ทดสอบตอนยังไม่มี spatial-lag feature, interaction ระหว่าง
feature อาจเปลี่ยนไปได้ ต้องยืนยันซ้ำก่อน retrain รอบสุดท้ายจริง

วิธีเดียวกับ ablation_feature_groups.py: P50-only, representative horizons, reuse
split_train_calib_test() จาก train_multiprovince_models.py (ตอนนี้ FEATURE_COLS/DATA_PATH ของ
ไฟล์นั้นเป็นเวอร์ชัน 24-feature/spatial แล้ว)
"""
import importlib.util
import os

import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(_ROOT, "models")
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
REPRESENTATIVE_HORIZONS = [1, 6, 12, 18, 24]
TIME_CYCLICAL_COLS = ["hour_sin", "hour_cos", "month_sin", "month_cos", "dayofweek"]


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
    print(f"โหลดข้อมูล: {train_mod.DATA_PATH}")
    df = pd.read_csv(train_mod.DATA_PATH)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    df = df[~df["is_zero_shot"]].reset_index(drop=True)

    train_df, _, test_df = train_mod.split_train_calib_test(df)
    print(f"train={len(train_df)}  test={len(test_df)}")

    full_features = train_mod.FEATURE_COLS  # 24 feature (มี pm25_neighbor_avg แล้ว)
    reduced_features = [c for c in full_features if c not in TIME_CYCLICAL_COLS]  # 19 feature
    print(f"full_features ({len(full_features)}): {full_features}")
    print(f"reduced_features ({len(reduced_features)}, ตัด time_cyclical): {reduced_features}")

    all_rows = []
    for kind in ("lgb", "xgb"):
        print(f"\n=== {kind.upper()} — baseline (24 feature, มี time_cyclical) ===")
        all_rows += eval_variant(train_df, test_df, full_features, kind, "baseline_with_time_cyclical", train_mod)
        print(f"\n=== {kind.upper()} — without_time_cyclical (19 feature) ===")
        all_rows += eval_variant(train_df, test_df, reduced_features, kind, "without_time_cyclical", train_mod)

    result_df = pd.DataFrame(all_rows)
    out_path = os.path.join(MODEL_DIR, f"validate_drop_time_cyclical_{TIMESTAMP}.csv")
    result_df.to_csv(out_path, index=False)
    print(f"\nบันทึกผลดิบที่: {out_path}")

    print("\n" + "=" * 80)
    print("สรุป: MAE เฉลี่ยข้าม 5 horizon ตัวแทน")
    print("=" * 80)
    summary = result_df.groupby(["model", "variant"])["mae"].mean().reset_index()
    verdicts = {}
    for kind in ("lgb", "xgb"):
        sub = summary[summary["model"] == kind].set_index("variant")
        baseline_mae = sub.loc["baseline_with_time_cyclical", "mae"]
        without_mae = sub.loc["without_time_cyclical", "mae"]
        delta_pct = (without_mae - baseline_mae) / baseline_mae * 100
        verdicts[kind] = delta_pct
        print(f"\n{kind.upper()}: with_time_cyclical={baseline_mae:.3f}  without={without_mae:.3f}  delta={delta_pct:+.2f}%")

    print("\n" + "=" * 80)
    if all(v < -0.3 for v in verdicts.values()):
        print("สรุป: ยืนยันซ้ำ - ตัด time_cyclical ออกช่วยทั้ง 2 โมเดล -> ควร retrain รอบสุดท้ายแบบตัดออกจริง")
    elif all(v > 0.3 for v in verdicts.values()):
        print("สรุป: ขัดกับผล D เดิม! หลัง F เปลี่ยน interaction ของ feature แล้ว time_cyclical กลับช่วย -> ไม่ควรตัดออก")
    else:
        print("สรุป: ผลไม่ตรงกันระหว่าง 2 โมเดล/ไม่ชัดเจน -> ควรพิจารณาเพิ่มเติมก่อนตัดสินใจ")


if __name__ == "__main__":
    main()

"""
ตรวจสอบ*สาเหตุ*ที่การตัด time_cyclical (hour/month/dayofweek cyclical features) ออกช่วยให้ MAE
ดีขึ้น (ยืนยันซ้ำแล้ว 2 รอบ: CHANGELOG 90, 96) — ที่ผ่านมามีแค่*สมมติฐาน*ว่า "ซ้ำซ้อนกับ
pm25_lag/rolling ซึ่งมี autocorrelation แฝง diurnal/seasonal pattern อยู่แล้วในตัว" แต่ไม่เคย
ตรวจสอบจริง งานนี้ทำ 2 ขั้น:

ขั้น A (ไม่ retrain): correlation matrix ระหว่าง time_cyclical (hour_sin/cos, month_sin/cos,
dayofweek) กับกลุ่ม pm25_lag_rolling (9 ตัว) — คอลัมน์เหล่านี้ยังอยู่ในไฟล์ raw dataset ปัจจุบัน
(ไม่ได้ถูกลบออกจากไฟล์ แค่ไม่ได้อยู่ใน FEATURE_COLS ที่ใช้เทรนจริงแล้ว) ไม่ต้องคำนวณใหม่

ขั้น B (retrain เบา ๆ): add-one-back-in ablation — ตรงข้ามกับ ablation_individual_features.py
(leave-one-out) ทดสอบใส่ time_cyclical กลับเข้าไปทีละกลุ่มย่อยเชิงแนวคิด (hour cyclical pair,
month cyclical pair, dayofweek เดี่ยว) บน baseline 19-feature ปัจจุบัน เพื่อหาว่าตัวไหนเป็น
ตัวการจริง ๆ แทนที่จะมองเป็นก้อนเดียวแบบ ablation_feature_groups.py (CHANGELOG 90) ซึ่งเป็น
ปัญหา aggregation เดียวกับที่ entry 107 แก้ให้กลุ่ม pm25_lag_rolling ไปแล้ว (ตัดทั้งกลุ่ม +9%
แต่ตัดทีละตัว +0.34% เฉลี่ย — ผลกลุ่มไม่เท่ากับผลรวมของทีละตัวเสมอไป)
"""
import importlib.util
import os

import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(_ROOT, "models")
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
REPRESENTATIVE_HORIZONS = [1, 6, 12, 18, 24]

TIME_CYCLICAL_SUBGROUPS = {
    "with_hour_cyclical": ["hour_sin", "hour_cos"],
    "with_month_cyclical": ["month_sin", "month_cos"],
    "with_dayofweek": ["dayofweek"],
}
LAG_ROLLING_GROUP = [
    "pm25_lag_1", "pm25_lag_3", "pm25_lag_6", "pm25_lag_12", "pm25_lag_24",
    "pm25_roll_mean_6", "pm25_roll_std_6", "pm25_roll_mean_24", "pm25_roll_std_24",
]


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
    return rows


def main():
    train_mod = load_train_module()
    print(f"โหลดข้อมูล: {train_mod.DATA_PATH}")
    df = pd.read_csv(train_mod.DATA_PATH)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    df = df[~df["is_zero_shot"]].reset_index(drop=True)

    train_df, _, test_df = train_mod.split_train_calib_test(df)
    print(f"train={len(train_df)}  test={len(test_df)}")

    # ---------- ขั้น A: correlation matrix (ไม่ retrain) ----------
    print("\n" + "=" * 90)
    print("ขั้น A: correlation ระหว่าง time_cyclical กับกลุ่ม pm25_lag_rolling (บน train split)")
    print("=" * 90)
    time_cyclical_cols = ["hour_sin", "hour_cos", "month_sin", "month_cos", "dayofweek"]
    corr = train_df[time_cyclical_cols + LAG_ROLLING_GROUP].corr()
    corr_block = corr.loc[time_cyclical_cols, LAG_ROLLING_GROUP]
    print(corr_block.round(3))
    print("\n|correlation| เฉลี่ยต่อ time_cyclical feature (ข้ามกลุ่ม pm25_lag_rolling ทั้ง 9 ตัว):")
    print(corr_block.abs().mean(axis=1).round(3))
    corr_out_path = os.path.join(MODEL_DIR, f"time_cyclical_correlation_{TIMESTAMP}.csv")
    corr_block.to_csv(corr_out_path)
    print(f"\nบันทึก correlation matrix ที่: {corr_out_path}")

    # ---------- ขั้น B: add-one-back-in ablation ----------
    baseline_features = train_mod.FEATURE_COLS
    print("\n" + "=" * 90)
    print(f"ขั้น B: add-one-back-in — baseline ({len(baseline_features)} feature ปัจจุบัน) "
          f"+ ใส่ time_cyclical กลับทีละกลุ่มย่อย")
    print(f"จำนวน variant: {1 + len(TIME_CYCLICAL_SUBGROUPS)} x 2 โมเดล x {len(REPRESENTATIVE_HORIZONS)} horizon = "
          f"{(1 + len(TIME_CYCLICAL_SUBGROUPS)) * 2 * len(REPRESENTATIVE_HORIZONS)} fit")
    print("=" * 90)

    all_rows = []
    for kind in ("lgb", "xgb"):
        print(f"\n=== {kind.upper()} — baseline ({len(baseline_features)} feature, ไม่มี time_cyclical) ===")
        all_rows += eval_variant(train_df, test_df, baseline_features, kind, "baseline", train_mod)

        for variant_name, extra_cols in TIME_CYCLICAL_SUBGROUPS.items():
            feats = baseline_features + extra_cols
            print(f"  {kind}: {variant_name} (+{extra_cols}, {len(feats)} feature รวม)...")
            all_rows += eval_variant(train_df, test_df, feats, kind, variant_name, train_mod)

    result_df = pd.DataFrame(all_rows)
    out_path = os.path.join(MODEL_DIR, f"time_cyclical_investigation_{TIMESTAMP}.csv")
    result_df.to_csv(out_path, index=False)
    print(f"\nบันทึกผลดิบที่: {out_path}")

    print("\n" + "=" * 90)
    print("สรุป: MAE เฉลี่ยข้าม 5 horizon ตัวแทน ต่อ variant (เทียบ baseline)")
    print("=" * 90)
    summary = result_df.groupby(["model", "variant"])["mae"].mean().reset_index()
    summary_rows = []
    for kind in ("lgb", "xgb"):
        sub = summary[summary["model"] == kind].set_index("variant")
        baseline_mae = sub.loc["baseline", "mae"]
        print(f"\n{kind.upper()}  (baseline MAE = {baseline_mae:.3f})")
        for variant_name in TIME_CYCLICAL_SUBGROUPS:
            mae = sub.loc[variant_name, "mae"]
            delta_pct = (mae - baseline_mae) / baseline_mae * 100
            print(f"  {variant_name:22s} -> MAE={mae:.3f}  (delta {delta_pct:+.2f}%)")
            summary_rows.append({"model": kind, "variant": variant_name, "baseline_mae": baseline_mae,
                                  "mae_with": mae, "delta_pct": delta_pct})

    summary_df = pd.DataFrame(summary_rows)
    summary_out_path = os.path.join(MODEL_DIR, f"time_cyclical_investigation_summary_{TIMESTAMP}.csv")
    summary_df.to_csv(summary_out_path, index=False)
    print(f"\nบันทึกตารางสรุปที่: {summary_out_path}")


if __name__ == "__main__":
    main()

"""
Ablation study แบบทีละ feature เดี่ยว (leave-one-out) วัดผลต่อ MAE/R² (P50 only)
ของทุก feature ที่ horizon ตัวแทน 5 จุด (t+1, t+6, t+12, t+18, t+24)

หมายเหตุ: บาง feature มีความสัมพันธ์กันสูง (multicollinearity) เช่น pm25_lag/roll ทั้งกลุ่ม,
wind_dir_sin/cos, lat/lon — ผลตัดทีละตัวจึงควรดูคู่กับผล ablation ทีละกลุ่ม
"""
import importlib.util
import os

import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(_ROOT, "models")
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
REPRESENTATIVE_HORIZONS = [1, 6, 12, 18, 24]

CORRELATED_GROUP = {
    "pm25_lag_1": "pm25_lag_rolling", "pm25_lag_3": "pm25_lag_rolling", "pm25_lag_6": "pm25_lag_rolling",
    "pm25_lag_12": "pm25_lag_rolling", "pm25_lag_24": "pm25_lag_rolling",
    "pm25_roll_mean_6": "pm25_lag_rolling", "pm25_roll_std_6": "pm25_lag_rolling",
    "pm25_roll_mean_24": "pm25_lag_rolling", "pm25_roll_std_24": "pm25_lag_rolling",
    "wind_dir_sin": "wind_dir_pair", "wind_dir_cos": "wind_dir_pair",
    "lat": "lat_lon_pair", "lon": "lat_lon_pair",
}


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
        return lgb.LGBMRegressor(objective="quantile", alpha=0.5, random_state=42, verbose=-1, **train_mod.BAGGING_PARAMS_LGB, **train_mod.TUNED_PARAMS_LGB).fit(X, y)
    import xgboost as xgb
    return xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, tree_method="hist",
                             random_state=42, verbosity=0,
                             **train_mod.BAGGING_PARAMS_XGB, **train_mod.TUNED_PARAMS_XGB).fit(X, y)


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

    all_features = train_mod.FEATURE_COLS
    print(f"feature ทั้งหมด ({len(all_features)}): {all_features}")
    print(f"จำนวน variant: {len(all_features) + 1} (baseline + leave-one-out ทีละตัว) x 2 โมเดล x "
          f"{len(REPRESENTATIVE_HORIZONS)} horizon = {(len(all_features) + 1) * 2 * len(REPRESENTATIVE_HORIZONS)} fit\n")

    all_rows = []
    for kind in ("lgb", "xgb"):
        print(f"=== {kind.upper()} — baseline (feature ครบ {len(all_features)} ตัว) ===")
        all_rows += eval_variant(train_df, test_df, all_features, kind, "baseline", train_mod)

        for feat in all_features:
            reduced = [c for c in all_features if c != feat]
            print(f"  {kind}: leave-out {feat} ({len(reduced)} feature เหลือ)...")
            all_rows += eval_variant(train_df, test_df, reduced, kind, f"without_{feat}", train_mod)

    result_df = pd.DataFrame(all_rows)
    out_path = os.path.join(MODEL_DIR, f"ablation_individual_features_{TIMESTAMP}.csv")
    result_df.to_csv(out_path, index=False)
    print(f"\nบันทึกผลดิบที่: {out_path}")

    print("\n" + "=" * 100)
    print("สรุป: MAE เฉลี่ยข้าม 5 horizon ตัวแทน ต่อ feature ที่ตัดออก (เรียงผลกระทบมากไปน้อย)")
    print("=" * 100)
    summary = result_df.groupby(["model", "variant"])["mae"].mean().reset_index()

    summary_rows = []
    for kind in ("lgb", "xgb"):
        sub = summary[summary["model"] == kind].set_index("variant")
        baseline_mae = sub.loc["baseline", "mae"]
        for feat in all_features:
            mae = sub.loc[f"without_{feat}", "mae"]
            delta_pct = (mae - baseline_mae) / baseline_mae * 100
            summary_rows.append({
                "model": kind, "feature": feat, "correlated_group": CORRELATED_GROUP.get(feat, "independent"),
                "baseline_mae": baseline_mae, "mae_without": mae, "delta_pct": delta_pct,
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_out_path = os.path.join(MODEL_DIR, f"ablation_individual_summary_{TIMESTAMP}.csv")
    summary_df.to_csv(summary_out_path, index=False)

    for kind in ("lgb", "xgb"):
        print(f"\n{kind.upper()}  (baseline MAE = {summary_df[summary_df.model==kind]['baseline_mae'].iloc[0]:.3f})")
        sub = summary_df[summary_df["model"] == kind].sort_values("delta_pct", ascending=False)
        for _, row in sub.iterrows():
            flag = f"  [กลุ่ม: {row['correlated_group']}]" if row["correlated_group"] != "independent" else ""
            print(f"  ตัด {row['feature']:20s} -> MAE={row['mae_without']:.3f}  (delta {row['delta_pct']:+.2f}%){flag}")

    print("\n" + "=" * 100)
    print("ตรวจสอบ multicollinearity hypothesis: เทียบผลรวม 'ทีละตัว' ของกลุ่ม pm25_lag_rolling "
          "กับผล 'ทั้งกลุ่ม' จาก ablation_feature_groups.py (CHANGELOG 90, +8-9% ตอนตัดทั้งกลุ่ม)")
    print("=" * 100)
    for kind in ("lgb", "xgb"):
        sub = summary_df[(summary_df["model"] == kind) & (summary_df["correlated_group"] == "pm25_lag_rolling")]
        print(f"  {kind}: delta เฉลี่ยทีละตัว (9 ตัวในกลุ่ม pm25_lag_rolling) = {sub['delta_pct'].mean():+.2f}% "
              f"(เทียบกับตัดทั้งกลุ่มทีเดียว = {'+9.1%' if kind == 'lgb' else '+8.0%'})")

    print(f"\nบันทึกตารางสรุปที่: {summary_out_path}")


if __name__ == "__main__":
    main()

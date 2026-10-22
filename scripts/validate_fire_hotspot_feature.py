"""
Validate feature fire_hotspot_count_100km_24h (จำนวนจุดความร้อนดาวเทียมในรัศมี 100km ย้อนหลัง 24 ชม.):
เทียบ baseline (19 feature) vs +fire_hotspot_count_100km_24h (20 feature) ด้วย MAE/R² (P50-only)
ที่ 5 horizon ตัวแทน แยกตามรวม/ปกติ/ฤดูฝุ่น/severity bin (>p90)
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
FEATURE_COL = "fire_hotspot_count_100km_24h"
BURNING_MONTHS = {2, 3, 4}


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
        r2 = r2_score(y_te_actual, pred)

        month = pd.to_datetime(te["datetime_utc"]).dt.month
        is_burning = month.isin(BURNING_MONTHS).values
        mae_normal = mean_absolute_error(y_te_actual[~is_burning], pred[~is_burning]) if (~is_burning).sum() else np.nan
        mae_burning = mean_absolute_error(y_te_actual[is_burning], pred[is_burning]) if is_burning.sum() else np.nan

        bins = [-0.01, np.percentile(y_te_actual, 50), np.percentile(y_te_actual, 75),
                np.percentile(y_te_actual, 90), y_te_actual.max() + 0.01]
        sev_bin = pd.cut(y_te_actual, bins=bins, labels=["<=p50", "p50-p75", "p75-p90", ">p90"])
        mae_p90 = mean_absolute_error(y_te_actual[sev_bin == ">p90"], pred[sev_bin == ">p90"]) \
            if (sev_bin == ">p90").sum() else np.nan

        rows.append({
            "variant": variant_name, "model": kind, "horizon": h, "mae": mae_overall, "r2": r2, "n": len(te),
            "mae_normal": mae_normal, "mae_burning": mae_burning, "mae_gt_p90": mae_p90,
        })
        print(f"  [{variant_name}/{kind}] t+{h:2d}  MAE={mae_overall:.3f} (ปกติ={mae_normal:.3f} "
              f"ฤดูฝุ่น={mae_burning:.3f} >p90={mae_p90:.3f})")
    return rows


def main():
    feature_data_path = latest_file("multiprovince_pm25_features_raw_spatial_k1_with_firehotspot_*.csv")
    print(f"โหลดข้อมูล: {feature_data_path}")
    df_all = pd.read_csv(feature_data_path)
    df_all["datetime_utc"] = pd.to_datetime(df_all["datetime_utc"])
    df = df_all[~df_all["is_zero_shot"]].reset_index(drop=True)

    train_mod = load_train_module()
    train_df, _, test_df = train_mod.split_train_calib_test(df)
    print(f"train={len(train_df)}  test={len(test_df)}")
    print(f"{FEATURE_COL} สถิติบน test: mean={test_df[FEATURE_COL].mean():.2f} "
          f"max={test_df[FEATURE_COL].max()} nonzero_pct={(test_df[FEATURE_COL] > 0).mean() * 100:.1f}%")

    baseline_features = train_mod.FEATURE_COLS
    extended_features = baseline_features + [FEATURE_COL]

    all_rows = []
    for kind in ("lgb", "xgb"):
        print(f"\n=== {kind.upper()} — baseline ({len(baseline_features)} feature, ไม่มี fire hotspot) ===")
        all_rows += eval_variant(train_df, test_df, baseline_features, kind, "baseline", train_mod)
        print(f"\n=== {kind.upper()} — with_fire_hotspot ({len(extended_features)} feature, +{FEATURE_COL}) ===")
        all_rows += eval_variant(train_df, test_df, extended_features, kind, "with_fire_hotspot", train_mod)

    result_df = pd.DataFrame(all_rows)
    out_path = os.path.join(MODEL_DIR, f"fire_hotspot_feature_validation_{TIMESTAMP}.csv")
    result_df.to_csv(out_path, index=False)
    print(f"\nบันทึกผลดิบที่: {out_path}")

    print("\n" + "=" * 90)
    print("สรุป: MAE เฉลี่ยข้าม 5 horizon ตัวแทน — ภาพรวม vs แยกฤดู/severity")
    print("=" * 90)
    for metric in ("mae", "mae_normal", "mae_burning", "mae_gt_p90"):
        summary = result_df.groupby(["model", "variant"])[metric].mean().unstack("variant")
        summary["delta_pct"] = (summary["with_fire_hotspot"] - summary["baseline"]) / summary["baseline"] * 100
        print(f"\n--- {metric} ---")
        print(summary.round(3).to_string())

    print("\n" + "=" * 90)
    verdicts = {}
    for kind in ("lgb", "xgb"):
        sub = result_df[result_df["model"] == kind].groupby("variant")["mae"].mean()
        delta = (sub["with_fire_hotspot"] - sub["baseline"]) / sub["baseline"] * 100
        verdicts[kind] = delta
    if all(v < -0.5 for v in verdicts.values()):
        print("สรุป: MAE รวมดีขึ้นทั้ง 2 โมเดล (>0.5%) -> คุ้มที่จะลงทุนเต็มรูปแบบ (sync feature formula "
              "4 จุด + retrain ทั้งชุด)")
    elif all(v > 0.5 for v in verdicts.values()):
        print("สรุป: MAE รวมแย่ลงทั้ง 2 โมเดล -> ไม่แนะนำ fire hotspot feature แบบนี้ไม่ช่วย MAE รวม "
              "(ตรวจ mae_burning/mae_gt_p90 ด้านบนเพิ่มเติม เผื่อช่วยเฉพาะจุดแม้ MAE รวมไม่ดีขึ้น)")
    else:
        print("สรุป: ผลไม่ตรงกันระหว่าง 2 โมเดล — ตรวจ mae_burning/mae_gt_p90 ประกอบก่อนตัดสินใจ")


if __name__ == "__main__":
    main()

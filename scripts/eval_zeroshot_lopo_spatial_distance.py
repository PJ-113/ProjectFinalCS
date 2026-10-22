"""
Leave-one-province-out (LOPO) ข้ามทั้ง 17 จังหวัดที่เทรนได้จริง: แต่ละรอบตัด 1 จังหวัดออกจาก
training pool, เทรนบน 16 จังหวัดที่เหลือ, ประเมินบน test split ของจังหวัดที่ตัดออก แล้ววัด
correlation + p-value ระหว่างระยะทางไปจังหวัดเทรนที่ใกล้ที่สุดกับ MAE (ที่ REPRESENTATIVE_HORIZONS)
3 จังหวัด zero-shot จริง (นครพนม/ร้อยเอ็ด/มหาสารคาม) ไม่เข้าร่วมทั้ง pool และ held-out เลย
"""
import importlib.util
import json
import os

import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_PATH = os.path.join(_ROOT, "data", "province_station_registry.json")
MODEL_DIR = os.path.join(_ROOT, "models")
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
REPRESENTATIVE_HORIZONS = [1, 6, 12, 18, 24]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, "scripts", filename))
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


def main():
    train_mod = load_module("train_multiprovince_models", "train_multiprovince_models.py")
    reg_mod = load_module("province_registry", "province_registry.py")
    haversine_km = reg_mod.haversine_km

    with open(REGISTRY_PATH, encoding="utf-8") as f:
        registry = json.load(f)
    provinces_meta = {p["province_en"]: p for p in registry["provinces"]}

    print(f"โหลดข้อมูล: {train_mod.DATA_PATH}")
    df = pd.read_csv(train_mod.DATA_PATH)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    df = df[~df["is_zero_shot"]].reset_index(drop=True)

    trainable_provinces = sorted(df["province_en"].unique().tolist())
    print(f"จังหวัดที่เทรนได้จริง (LOPO pool): {len(trainable_provinces)} จังหวัด")
    print(f"งบประมาณ: {len(trainable_provinces)} fold x 2 โมเดล x {len(REPRESENTATIVE_HORIZONS)} horizon = "
          f"{len(trainable_provinces) * 2 * len(REPRESENTATIVE_HORIZONS)} fit\n")

    per_fold_rows = []
    for i, held_out in enumerate(trainable_provinces, 1):
        pool_df = df[df["province_en"] != held_out].reset_index(drop=True)
        held_out_df = df[df["province_en"] == held_out].reset_index(drop=True)

        train_pool, calib_pool, _ = train_mod.split_train_calib_test(pool_df)
        train_pool = pd.concat([train_pool, calib_pool], ignore_index=True)
        _, _, held_out_test = train_mod.split_train_calib_test(held_out_df)

        remaining = [p for p in trainable_provinces if p != held_out]
        p_meta = provinces_meta[held_out]
        dists = {tp: haversine_km(p_meta["lat"], p_meta["lon"], provinces_meta[tp]["lat"], provinces_meta[tp]["lon"])
                 for tp in remaining}
        nearest_province = min(dists, key=dists.get)
        nearest_km = dists[nearest_province]

        print(f"[{i}/{len(trainable_provinces)}] held-out={held_out}  "
              f"train_pool={len(train_pool)}  held_out_test={len(held_out_test)}  "
              f"nearest={nearest_province} ({nearest_km:.1f}km)")

        for kind in ("lgb", "xgb"):
            for h in REPRESENTATIVE_HORIZONS:
                target_col = f"pm25_t+{h}"
                tr = train_pool.dropna(subset=[target_col])
                te = held_out_test.dropna(subset=[target_col])
                if te.empty:
                    continue
                X_tr, y_tr_resid = tr[train_mod.FEATURE_COLS], tr[target_col] - tr["pm25"]
                X_te, current_te = te[train_mod.FEATURE_COLS], te["pm25"].values
                y_te_actual = te[target_col].values

                model = fit_p50(X_tr, y_tr_resid, kind, train_mod)
                pred = model.predict(X_te) + current_te
                per_fold_rows.append({
                    "held_out_province": held_out, "model": kind, "horizon": h,
                    "mae": mean_absolute_error(y_te_actual, pred), "r2": r2_score(y_te_actual, pred),
                    "n": len(te), "nearest_trained_province": nearest_province,
                    "distance_to_nearest_trained_km": nearest_km,
                })

    result_df = pd.DataFrame(per_fold_rows)
    out_path = os.path.join(MODEL_DIR, f"zeroshot_lopo_distance_per_horizon_{TIMESTAMP}.csv")
    result_df.to_csv(out_path, index=False)
    print(f"\nบันทึก per-fold per-horizon ที่: {out_path}")

    summary = result_df.groupby(["model", "held_out_province"]).agg(
        mae=("mae", "mean"), distance_to_nearest_trained_km=("distance_to_nearest_trained_km", "first"),
        nearest_trained_province=("nearest_trained_province", "first"),
    ).reset_index()

    print("\n" + "=" * 90)
    print(f"สรุป: MAE เฉลี่ย {len(REPRESENTATIVE_HORIZONS)} horizon ตัวแทน ต่อจังหวัด LOPO (n=17) "
          f"เทียบระยะทางไปจังหวัดเทรนที่ใกล้ที่สุด")
    print("=" * 90)
    for kind in ("lgb", "xgb"):
        sub = summary[summary["model"] == kind].sort_values("distance_to_nearest_trained_km")
        print(f"\n{kind.upper()}:")
        print(sub[["held_out_province", "nearest_trained_province", "distance_to_nearest_trained_km", "mae"]]
              .to_string(index=False))
        corr, p_value = pearsonr(sub["distance_to_nearest_trained_km"], sub["mae"])
        print(f"  correlation(distance, mae) = {corr:.3f}  p-value = {p_value:.4f}  (n=17)")
        if p_value < 0.05:
            direction = "ตรงสมมติฐาน (ยิ่งไกลยิ่ง MAE สูง)" if corr > 0 else "ขัดสมมติฐาน (ยิ่งไกลยิ่ง MAE ต่ำ)"
            print(f"  -> มีนัยสำคัญทางสถิติ (p<0.05), ทิศทาง: {direction}")
        else:
            print("  -> ไม่มีนัยสำคัญทางสถิติที่ n=17 นี้ (p>=0.05)")

    summary_path = os.path.join(MODEL_DIR, f"zeroshot_lopo_distance_summary_{TIMESTAMP}.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nบันทึกตารางสรุปที่: {summary_path}")


if __name__ == "__main__":
    main()

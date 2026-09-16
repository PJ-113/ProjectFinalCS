"""
Tune จำนวนจังหวัดเพื่อนบ้าน (k) ใน pm25_neighbor_avg (spatial-lag feature) — ปัจจุบันใช้ k=3
คงที่มาตั้งแต่ validate ครั้งแรก (CHANGELOG 94, scripts/validate_spatial_lag_feature.py) ไม่เคย
ลองค่าอื่นเทียบกันจริง งานนี้กวาด k หลายค่าเพื่อดูว่า k=3 เป็นค่าที่ดีที่สุดจริงไหม

ไม่ต้องเรียก API ใหม่เลย — คอลัมน์ pm25 ดิบของทุกจังหวัด (รวม zero-shot) มีอยู่แล้วใน
DATA_PATH (train_multiprovince_models.DATA_PATH) พอสำหรับ pivot คำนวณเพื่อนบ้านที่ k ใดๆ

สมมติฐานตั้งไว้ล่วงหน้า: MAE ควรดีขึ้นตาม k จนถึงจุดหนึ่งแล้วแย่ลง (เพื่อนบ้านไกลเกินไปเริ่มเป็น
noise ไม่ใช่สัญญาณ spatial correlation จริง ตาม task B ที่ยืนยันว่า correlation ลดลงตามระยะทาง)
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
K_GRID = [1, 2, 3, 5, 7, 10, 15, 19]
CURRENT_K = 3


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, "scripts", filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_neighbor_avg_feature(df_all_provinces, registry_provinces, haversine_km, k):
    """คัดลอกจาก validate_spatial_lag_feature.py — คืน DataFrame (index=datetime_utc,
    columns=province_en) ของค่าเฉลี่ย pm25 ปัจจุบันของ k จังหวัดที่ใกล้ที่สุด ณ เวลาเดียวกัน"""
    pivot = df_all_provinces.pivot_table(index="datetime_utc", columns="province_en", values="pm25")
    all_provinces = pivot.columns.tolist()

    neighbors = {}
    for p in all_provinces:
        p_meta = registry_provinces[p]
        dists = {
            other: haversine_km(p_meta["lat"], p_meta["lon"], registry_provinces[other]["lat"], registry_provinces[other]["lon"])
            for other in all_provinces if other != p
        }
        nearest = sorted(dists, key=dists.get)[:k]
        neighbors[p] = nearest

    result = {p: pivot[neighbors[p]].mean(axis=1) for p in all_provinces}
    return pd.DataFrame(result)


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
        print(f"  [{variant_name}/{kind}] t+{h:2d}  MAE={mae:.3f}  R2={r2:.3f}")
    return rows


def main():
    train_mod = load_module("train_multiprovince_models", "train_multiprovince_models.py")
    reg_mod = load_module("province_registry", "province_registry.py")
    haversine_km = reg_mod.haversine_km

    with open(REGISTRY_PATH, encoding="utf-8") as f:
        registry = json.load(f)
    provinces_meta = {p["province_en"]: p for p in registry["provinces"]}

    print(f"โหลดข้อมูล: {train_mod.DATA_PATH}")
    df_all = pd.read_csv(train_mod.DATA_PATH)
    df_all["datetime_utc"] = pd.to_datetime(df_all["datetime_utc"])

    df = df_all[~df_all["is_zero_shot"]].reset_index(drop=True)  # 17 จังหวัดเทรน (เหมือนทุกงานก่อนหน้า)

    baseline_features = [c for c in train_mod.FEATURE_COLS if c != "pm25_neighbor_avg"]
    print(f"baseline feature (ไม่มี spatial-lag, {len(baseline_features)} ตัว): {baseline_features}")

    train_df, _, test_df = train_mod.split_train_calib_test(df)
    print(f"train={len(train_df)}  test={len(test_df)}")

    all_rows = []
    print(f"\n=== baseline (ไม่มี pm25_neighbor_avg เลย) ===")
    for kind in ("lgb", "xgb"):
        all_rows += eval_variant(train_df, test_df, baseline_features, kind, "baseline", train_mod)

    for k in K_GRID:
        print(f"\n=== k={k}: คำนวณ pm25_neighbor_avg_k{k} ===")
        neighbor_avg_wide = build_neighbor_avg_feature(df_all, provinces_meta, haversine_km, k)
        col_name = f"pm25_neighbor_avg_k{k}"
        neighbor_avg_long = (
            neighbor_avg_wide.stack(future_stack=True).rename(col_name).reset_index()
            .rename(columns={"level_1": "province_en"})
        )
        df_k = df.merge(neighbor_avg_long, on=["datetime_utc", "province_en"], how="left")
        n_missing = df_k[col_name].isna().sum()
        df_k = df_k.dropna(subset=[col_name]).reset_index(drop=True)
        print(f"  แถวที่ไม่มีเพื่อนบ้าน (ตัดทิ้ง): {n_missing}/{len(df) + n_missing}  เหลือ={len(df_k)}")

        train_k, _, test_k = train_mod.split_train_calib_test(df_k)
        extended_features = baseline_features + [col_name]

        for kind in ("lgb", "xgb"):
            rows = eval_variant(train_k, test_k, extended_features, kind, f"k={k}", train_mod)
            all_rows += rows

    result_df = pd.DataFrame(all_rows)
    out_path = os.path.join(MODEL_DIR, f"spatial_lag_k_tuning_{TIMESTAMP}.csv")
    result_df.to_csv(out_path, index=False)
    print(f"\nบันทึกผลดิบที่: {out_path}")

    print("\n" + "=" * 80)
    print("สรุป: MAE เฉลี่ยข้าม 5 horizon ตัวแทน ต่อค่า k (เรียงจากดีสุด)")
    print("=" * 80)
    summary = result_df.groupby(["model", "variant"])["mae"].mean().reset_index()

    summary_rows = []
    for kind in ("lgb", "xgb"):
        sub = summary[summary["model"] == kind].set_index("variant")
        baseline_mae = sub.loc["baseline", "mae"]
        print(f"\n{kind.upper()}  (baseline MAE ไม่มี spatial-lag = {baseline_mae:.3f})")
        rows_for_kind = []
        for k in K_GRID:
            mae_k = sub.loc[f"k={k}", "mae"]
            delta_pct = (mae_k - baseline_mae) / baseline_mae * 100
            rows_for_kind.append({"model": kind, "k": k, "baseline_mae": baseline_mae,
                                   "mae": mae_k, "delta_vs_baseline_pct": delta_pct})
        rows_for_kind.sort(key=lambda r: r["mae"])
        for r in rows_for_kind:
            marker = "  <-- k ปัจจุบัน (deployed candidate)" if r["k"] == CURRENT_K else ""
            print(f"  k={r['k']:2d}  MAE={r['mae']:.3f}  (delta vs no-spatial-lag: {r['delta_vs_baseline_pct']:+.2f}%){marker}")
        summary_rows.extend(rows_for_kind)

    summary_df = pd.DataFrame(summary_rows)
    summary_out_path = os.path.join(MODEL_DIR, f"spatial_lag_k_tuning_summary_{TIMESTAMP}.csv")
    summary_df.to_csv(summary_out_path, index=False)

    print("\n" + "=" * 80)
    print(f"เทียบ k={CURRENT_K} (ค่าที่ deploy candidate ปัจจุบันใช้) กับ k ที่ดีที่สุดที่เจอ")
    print("=" * 80)
    for kind in ("lgb", "xgb"):
        sub = summary_df[summary_df["model"] == kind]
        current_mae = sub[sub["k"] == CURRENT_K]["mae"].iloc[0]
        best_row = sub.loc[sub["mae"].idxmin()]
        if best_row["k"] == CURRENT_K:
            print(f"  {kind}: k={CURRENT_K} เป็นค่าที่ดีที่สุดในกริดนี้อยู่แล้ว ไม่ต้องเปลี่ยน")
        else:
            improvement_pct = (current_mae - best_row["mae"]) / current_mae * 100
            print(f"  {kind}: k={int(best_row['k'])} ดีกว่า k={CURRENT_K} อยู่ {improvement_pct:+.2f}% "
                  f"(MAE {current_mae:.3f} -> {best_row['mae']:.3f})")

    print(f"\nบันทึกตารางสรุปที่: {summary_out_path}")


if __name__ == "__main__":
    main()

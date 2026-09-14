"""
Task F (feedback อาจารย์ ข้อ 1) — ก่อนลงทุนแก้ feature formula ที่ซ้ำกัน 4 จุด + retrain เต็มระบบ
(งานใหญ่สุดในแผน) ทำ scaled-down validation ก่อน: เพิ่ม spatial-lag feature เดียว
("pm25_neighbor_avg" = ค่าเฉลี่ย pm25 ปัจจุบันของ K จังหวัดที่ใกล้ที่สุด ณ เวลาเดียวกัน) แล้ววัดผล
แบบเดียวกับ ablation study (task D) — ถ้าช่วยจริงค่อยลงทุนงานใหญ่ ถ้าไม่ช่วยก็ไม่เสียเวลา

เหตุผลที่ทำแบบนี้: task B ยืนยันชัดเจนว่า PM2.5 มี spatial correlation จริง (r=-0.644, n=190 คู่)
แต่ task C (n=3 จุด) ไม่ยืนยันในระดับโมเดลจริง — สัญญาณผสม จึงควร validate ราคาถูกก่อนลงทุนงานใหญ่

Neighbor pm25 ใช้ค่า ณ เวลาเดียวกับแถวนั้น (contemporaneous, ไม่ใช่อนาคต) — ไม่ leak เพราะใน
production จริง scheduled_forecast_job() ดึงข้อมูลทุกจังหวัดพร้อมกันอยู่แล้วทุกชั่วโมง (ดู app.py)
จึงมีค่า pm25 ปัจจุบันของจังหวัดอื่นให้ใช้จริงตอน predict ไม่ใช่ข้อมูลที่รู้ทีหลัง
"""
import importlib.util
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(_ROOT, "data", "multiprovince_pm25_features_raw_20260913_012629.csv")
REGISTRY_PATH = os.path.join(_ROOT, "data", "province_station_registry.json")
MODEL_DIR = os.path.join(_ROOT, "models")
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
K_NEAREST = 3
REPRESENTATIVE_HORIZONS = [1, 6, 12, 18, 24]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, "scripts", filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_neighbor_avg_feature(df_all_provinces, registry_provinces, haversine_km, k):
    """คืน Series (index = (province_en, datetime_utc)) ของค่าเฉลี่ย pm25 ปัจจุบันของ k จังหวัด
    ที่ใกล้ที่สุด ณ เวลาเดียวกัน (ใช้ข้อมูลจากทั้ง 20 จังหวัด รวม zero-shot เพราะ production จริง
    มีข้อมูลจังหวัดเหล่านี้ให้ใช้เป็น context ด้วยเช่นกัน)"""
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
        print(f"  {p}: {k} จังหวัดใกล้สุด = {nearest}")

    result = {}
    for p in all_provinces:
        result[p] = pivot[neighbors[p]].mean(axis=1)
    neighbor_avg_wide = pd.DataFrame(result)  # index=datetime_utc, columns=province_en
    return neighbor_avg_wide


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

    print(f"โหลดข้อมูล: {DATA_PATH}")
    df_all = pd.read_csv(DATA_PATH)
    df_all["datetime_utc"] = pd.to_datetime(df_all["datetime_utc"])

    print(f"\nคำนวณ neighbor average feature (k={K_NEAREST}) จากทั้ง 20 จังหวัด:")
    neighbor_avg_wide = build_neighbor_avg_feature(df_all, provinces_meta, haversine_km, K_NEAREST)

    df = df_all[~df_all["is_zero_shot"]].reset_index(drop=True)  # 17 จังหวัดเทรน (เหมือน task A/D)

    neighbor_avg_long = (
        neighbor_avg_wide.stack(future_stack=True).rename("pm25_neighbor_avg").reset_index()
        .rename(columns={"level_1": "province_en"})
    )
    df = df.merge(neighbor_avg_long, on=["datetime_utc", "province_en"], how="left")
    n_missing = df["pm25_neighbor_avg"].isna().sum()
    print(f"\nจำนวนแถวที่ neighbor_avg เป็น NaN (ไม่มีข้อมูลเพื่อนบ้าน ณ เวลานั้น): {n_missing}/{len(df)}")

    df = df.dropna(subset=["pm25_neighbor_avg"]).reset_index(drop=True)
    print(f"แถวที่เหลือหลังตัด NaN: {len(df)}")

    train_df, _, test_df = train_mod.split_train_calib_test(df)
    print(f"train={len(train_df)}  test={len(test_df)}")

    baseline_features = train_mod.FEATURE_COLS
    extended_features = baseline_features + ["pm25_neighbor_avg"]

    all_rows = []
    for kind in ("lgb", "xgb"):
        print(f"\n=== {kind.upper()} — baseline (23 feature, ไม่มี spatial-lag) ===")
        all_rows += eval_variant(train_df, test_df, baseline_features, kind, "baseline", train_mod)
        print(f"\n=== {kind.upper()} — with_spatial_lag (24 feature, +pm25_neighbor_avg) ===")
        all_rows += eval_variant(train_df, test_df, extended_features, kind, "with_spatial_lag", train_mod)

    result_df = pd.DataFrame(all_rows)
    out_path = os.path.join(MODEL_DIR, f"spatial_lag_validation_{TIMESTAMP}.csv")
    result_df.to_csv(out_path, index=False)
    print(f"\nบันทึกผลดิบที่: {out_path}")

    print("\n" + "=" * 80)
    print("สรุป: MAE เฉลี่ยข้าม 5 horizon ตัวแทน (baseline vs with_spatial_lag)")
    print("=" * 80)
    summary = result_df.groupby(["model", "variant"])["mae"].mean().reset_index()
    verdicts = {}
    for kind in ("lgb", "xgb"):
        sub = summary[summary["model"] == kind].set_index("variant")
        baseline_mae = sub.loc["baseline", "mae"]
        with_mae = sub.loc["with_spatial_lag", "mae"]
        delta_pct = (with_mae - baseline_mae) / baseline_mae * 100
        verdicts[kind] = delta_pct
        print(f"\n{kind.upper()}: baseline={baseline_mae:.3f}  with_spatial_lag={with_mae:.3f}  delta={delta_pct:+.2f}%")

    print("\n" + "=" * 80)
    if all(v < -0.5 for v in verdicts.values()):
        print("สรุป: MAE ดีขึ้นทั้ง 2 โมเดล (>0.5%) -> คุ้มที่จะลงทุนทำ F เต็มรูปแบบ "
              "(แก้ feature formula 4 จุด + retrain ทั้งชุด)")
    elif all(v > 0.5 for v in verdicts.values()):
        print("สรุป: MAE แย่ลงทั้ง 2 โมเดล -> ไม่แนะนำทำ F เต็มรูปแบบ spatial-lag feature "
              "แบบง่ายนี้ไม่ช่วย")
    else:
        print("สรุป: ผลไม่ชัดเจน/ไม่ตรงกันระหว่าง 2 โมเดล -> ยังสรุปไม่ได้ชัด ควรพิจารณาปัจจัยอื่น "
              "ก่อนตัดสินใจลงทุน F เต็มรูปแบบ")


if __name__ == "__main__":
    main()

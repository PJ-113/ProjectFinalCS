"""
ประเมินโมเดล +nb_trend+tuned (_20260919_135655, CHANGELOG 127) กับ **3 จังหวัด zero-shot จริง**
(นครพนม/ร้อยเอ็ด/มหาสารคาม, is_zero_shot=True) — ต่อจาก entry 119 ที่เช็คโมเดล k=1 เดิมแล้วพบว่า
coverage ต่ำกว่าจังหวัดเทรนชัดเจน (72-87%) โดยเฉพาะนครพนม (74.1-75.4% เดิม) ที่ entry 124 พบว่า
เป็นจังหวัด zero-shot ที่มีค่าเฉลี่ย fire_hotspot สูงสุด (49.56) — เช็คว่า feature ใหม่ที่ validate
แล้วว่าช่วย MAE ที่ severity สูงสุด (entry 125) จะช่วยจังหวัดนี้ในทางปฏิบัติจริงได้ไหม

dataset ที่ใช้ (`multiprovince_pm25_features_raw_spatial_k1_with_firehotspot_20260918_155102.csv`)
มี fire_hotspot_count_100km_24h คำนวณไว้ครบทั้ง 20 จังหวัดอยู่แล้ว (รวม zero-shot, จาก entry 124)
ไม่ต้องคำนวณใหม่

ประเมิน 2 ช่วงเวลา: (1) จำกัดช่วงเดียวกับ test split ทางการของ 17 จังหวัดเทรน (เทียบตรงกับ entry 119
ได้) (2) เต็มประวัติที่มี (ไว้เป็นข้อมูลเสริม)
"""
import importlib.util
import json
import os
import pickle

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(_ROOT, "models")
REGISTRY_PATH = os.path.join(_ROOT, "data", "province_station_registry.json")
MODEL_TAG = "20260919_135655"
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, "scripts", filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_model_pkg(tag):
    path = os.path.join(MODEL_DIR, f"{tag}_pm25_multiprovince_residual_bagged_quantile_{MODEL_TAG}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_baseline_qhat(tag):
    path = os.path.join(MODEL_DIR, f"conformal_calibration_{tag}_{MODEL_TAG}.csv")
    df = pd.read_csv(path)
    return {int(r.horizon): float(r.q_hat) for r in df.itertuples()}


def predict_all_horizons(pkg, df_split, feature_cols, q_hat_by_h):
    out_rows = []
    for h in range(1, 25):
        target_col = f"pm25_t+{h}"
        sub = df_split.dropna(subset=[target_col])
        if sub.empty:
            continue
        X = sub[feature_cols]
        current = sub["pm25"].values
        models_by_q = pkg["models"][h]

        p50 = models_by_q[0.5].predict(X) + current
        lo = models_by_q[0.05].predict(X) + current
        hi = models_by_q[0.95].predict(X) + current
        q_hat = q_hat_by_h[h]
        lo_adj, hi_adj = lo - q_hat, hi + q_hat

        actual = sub[target_col].values
        out_rows.append(pd.DataFrame({
            "province_en": sub["province_en"].values, "datetime_utc": sub["datetime_utc"].values,
            "horizon": h, "predicted": p50, "actual": actual, "pm25_lo": lo_adj, "pm25_hi": hi_adj,
        }))
    return pd.concat(out_rows, ignore_index=True)


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

    trained_df = df[~df["is_zero_shot"]].reset_index(drop=True)
    _, _, official_test_df = train_mod.split_train_calib_test(trained_df)
    test_start, test_end = official_test_df["datetime_utc"].min(), official_test_df["datetime_utc"].max()
    print(f"ช่วงเวลา test split ทางการ (17 จังหวัดเทรน): {test_start} ถึง {test_end}")

    zero_shot_df = df[df["is_zero_shot"]].reset_index(drop=True)
    zero_shot_provinces = sorted(zero_shot_df["province_en"].unique().tolist())
    print(f"จังหวัด zero-shot จริง: {zero_shot_provinces}")
    for p in zero_shot_provinces:
        sub = zero_shot_df[zero_shot_df["province_en"] == p]
        print(f"  {p}: {len(sub)} แถว ({sub['datetime_utc'].min()} ถึง {sub['datetime_utc'].max()})")

    trained_provinces = sorted(trained_df["province_en"].unique().tolist())
    dist_rows = []
    for p in zero_shot_provinces:
        p_meta = provinces_meta[p]
        dists = {tp: haversine_km(p_meta["lat"], p_meta["lon"], provinces_meta[tp]["lat"], provinces_meta[tp]["lon"])
                 for tp in trained_provinces}
        nearest = min(dists, key=dists.get)
        dist_rows.append({"province_en": p, "nearest_trained_province": nearest,
                           "distance_to_nearest_trained_km": dists[nearest]})
    dist_df = pd.DataFrame(dist_rows)
    print("\nระยะทางจากแต่ละจังหวัด zero-shot จริง ไปจังหวัดเทรนที่ใกล้ที่สุด:")
    print(dist_df.to_string(index=False))

    zero_shot_restricted = zero_shot_df[
        (zero_shot_df["datetime_utc"] >= test_start) & (zero_shot_df["datetime_utc"] <= test_end)
    ].reset_index(drop=True)
    print(f"\nจำนวนแถว zero-shot ในช่วง test split ทางการ: {len(zero_shot_restricted)} "
          f"(เต็มประวัติ: {len(zero_shot_df)})")

    all_rows = []
    for tag in ("lgb", "xgb"):
        pkg = load_model_pkg(tag)
        baseline_qhat = load_baseline_qhat(tag)
        fcols = pkg["feature_cols"]

        for window_name, window_df in (("test_period_only", zero_shot_restricted), ("full_history", zero_shot_df)):
            if window_df.empty:
                continue
            res = predict_all_horizons(pkg, window_df, fcols, baseline_qhat)
            res["abs_error"] = (res["predicted"] - res["actual"]).abs()
            res["covered"] = (res["actual"] >= res["pm25_lo"]) & (res["actual"] <= res["pm25_hi"])
            res["model"] = tag
            res["window"] = window_name
            all_rows.append(res)

    result_df = pd.concat(all_rows, ignore_index=True)
    out_path = os.path.join(MODEL_DIR, f"zeroshot_real_provinces_per_horizon_{TIMESTAMP}.csv")
    result_df.to_csv(out_path, index=False)
    print(f"\nบันทึกผลดิบที่: {out_path}")

    print("\n" + "=" * 100)
    print("สรุป: MAE/coverage เฉลี่ยข้ามทุก horizon ต่อจังหวัด zero-shot จริง")
    print("=" * 100)
    summary = result_df.groupby(["model", "window", "province_en"]).agg(
        mae=("abs_error", "mean"), coverage_pct=("covered", lambda s: round(s.mean() * 100, 1)),
        n=("abs_error", "count"),
    ).reset_index()
    summary = summary.merge(dist_df, on="province_en")

    for tag in ("lgb", "xgb"):
        for window_name in ("test_period_only", "full_history"):
            sub = summary[(summary["model"] == tag) & (summary["window"] == window_name)] \
                .sort_values("distance_to_nearest_trained_km")
            if sub.empty:
                continue
            print(f"\n{tag.upper()} — {window_name}:")
            print(sub[["province_en", "nearest_trained_province", "distance_to_nearest_trained_km",
                        "mae", "coverage_pct", "n"]].to_string(index=False))

    summary_path = os.path.join(MODEL_DIR, f"zeroshot_real_provinces_summary_{TIMESTAMP}.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nบันทึกตารางสรุปที่: {summary_path}")

    print("\n" + "=" * 100)
    print("เทียบกับช่วง LOPO simulation (entry 114, n=17 จังหวัดเทรน, test_period เดียวกัน, MAE เฉลี่ย "
          "5 horizon ตัวแทน) — จังหวัด zero-shot จริงตกอยู่ในช่วงเดียวกันไหม")
    print("=" * 100)
    lopo_path = os.path.join(MODEL_DIR, "zeroshot_lopo_distance_summary_20260918_132858.csv")
    if os.path.exists(lopo_path):
        lopo_df = pd.read_csv(lopo_path)
        for tag in ("lgb", "xgb"):
            lopo_mae_range = lopo_df[lopo_df["model"] == tag]["mae"].agg(["min", "max", "mean"])
            print(f"  {tag}: LOPO MAE range = [{lopo_mae_range['min']:.2f}, {lopo_mae_range['max']:.2f}] "
                  f"(เฉลี่ย {lopo_mae_range['mean']:.2f})")
    else:
        print("  ไม่พบไฟล์ LOPO summary (entry 114) ข้ามการเทียบนี้")


if __name__ == "__main__":
    main()

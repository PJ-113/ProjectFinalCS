"""
วัด Pearson correlation ระหว่าง LOPO (leave-one-province-out) MAE กับตัวแปรอธิบายต่อจังหวัด:
distance_to_nearest_trained_km, missing_ratio (สัดส่วน NaN ใน FEATURE_COLS), pm25_mean/std,
elevation_m, amenity_density_3km, n_rows
"""
import importlib.util
import json
import glob
import os

import pandas as pd
from scipy.stats import pearsonr

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(_ROOT, "models")
REGISTRY_PATH = os.path.join(_ROOT, "data", "province_station_registry.json")
LOPO_SUMMARY_PATH = sorted(glob.glob(os.path.join(MODEL_DIR, "zeroshot_lopo_distance_summary_*.csv")))[-1]  # ไฟล์ LOPO ล่าสุด
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")


def load_train_module():
    spec = importlib.util.spec_from_file_location(
        "train_multiprovince_models", os.path.join(_ROOT, "scripts", "train_multiprovince_models.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    train_mod = load_train_module()
    print(f"โหลดข้อมูล: {train_mod.DATA_PATH}")
    df = pd.read_csv(train_mod.DATA_PATH)
    trained_df = df[~df["is_zero_shot"]].reset_index(drop=True)

    with open(REGISTRY_PATH, encoding="utf-8") as f:
        registry = json.load(f)
    provinces_meta = {p["province_en"]: p for p in registry["provinces"]}

    print(f"โหลดผล LOPO (entry 114): {LOPO_SUMMARY_PATH}")
    lopo_df = pd.read_csv(LOPO_SUMMARY_PATH)

    feature_cols = train_mod.FEATURE_COLS
    confounder_rows = []
    for province in sorted(trained_df["province_en"].unique().tolist()):
        sub = trained_df[trained_df["province_en"] == province]
        missing_ratio = sub[feature_cols].isna().mean(axis=1).mean()
        meta = provinces_meta[province]
        confounder_rows.append({
            "province_en": province,
            "missing_ratio": missing_ratio,
            "pm25_mean": sub["pm25"].mean(),
            "pm25_std": sub["pm25"].std(),
            "elevation_m": meta.get("elevation_m"),
            "amenity_density_3km": meta.get("amenity_density_3km"),
            "n_rows": len(sub),
        })
    confounder_df = pd.DataFrame(confounder_rows)
    print("\nตัวแปรอธิบายเพิ่มเติมต่อจังหวัด:")
    print(confounder_df.to_string(index=False))

    merged = lopo_df.merge(confounder_df, left_on="held_out_province", right_on="province_en")
    out_path = os.path.join(MODEL_DIR, f"zeroshot_lopo_confounders_merged_{TIMESTAMP}.csv")
    merged.to_csv(out_path, index=False)
    print(f"\nบันทึกตารางรวมที่: {out_path}")

    candidate_vars = ["distance_to_nearest_trained_km", "missing_ratio", "pm25_mean", "pm25_std",
                       "elevation_m", "amenity_density_3km", "n_rows"]

    print("\n" + "=" * 100)
    print("สรุป: correlation(ตัวแปร, LOPO MAE) เรียงตาม |r| มากไปน้อย ต่อโมเดล (n=17, เทียบกับ distance "
          "เดิมจาก entry 114 ที่ r=0.28-0.30, p=0.24-0.29)")
    print("=" * 100)
    corr_rows = []
    for tag in ("lgb", "xgb"):
        sub = merged[merged["model"] == tag]
        print(f"\n{tag.upper()}:")
        for var in candidate_vars:
            r, p = pearsonr(sub[var], sub["mae"])
            corr_rows.append({"model": tag, "variable": var, "r": r, "p_value": p})
        var_results = [row for row in corr_rows if row["model"] == tag]
        var_results.sort(key=lambda r: -abs(r["r"]))
        for row in var_results:
            sig = " <- significant (p<0.05)" if row["p_value"] < 0.05 else ""
            print(f"  {row['variable']:28s} r={row['r']:+.3f}  p={row['p_value']:.4f}{sig}")

    corr_df = pd.DataFrame(corr_rows)
    corr_path = os.path.join(MODEL_DIR, f"zeroshot_lopo_confounders_correlation_{TIMESTAMP}.csv")
    corr_df.to_csv(corr_path, index=False)
    print(f"\nบันทึกตาราง correlation ที่: {corr_path}")


if __name__ == "__main__":
    main()

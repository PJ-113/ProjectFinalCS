"""
รีบิลด์คอลัมน์ pm25_neighbor_avg ของ dataset เดิมใหม่ด้วย k=1 โดยเรียก
add_spatial_lag_feature(k=1) จาก build_raw_dataset_multiprovince.py ตรงๆ
"""
import importlib.util
import json
import os

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_PATH = os.path.join(_ROOT, "data", "multiprovince_pm25_features_raw_spatial_20260914_203924.csv")
REGISTRY_PATH = os.path.join(_ROOT, "data", "province_station_registry.json")
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
OUT_PATH = os.path.join(_ROOT, "data", f"multiprovince_pm25_features_raw_spatial_k1_{TIMESTAMP}.csv")
K_NEW = 1


def load_build_module():
    spec = importlib.util.spec_from_file_location(
        "build_raw_dataset_multiprovince", os.path.join(_ROOT, "scripts", "build_raw_dataset_multiprovince.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    build_mod = load_build_module()

    print(f"โหลดข้อมูล: {SOURCE_PATH}")
    combined = pd.read_csv(SOURCE_PATH)
    combined["datetime_utc"] = pd.to_datetime(combined["datetime_utc"])
    print(f"แถวรวม (ก่อนคำนวณ neighbor ใหม่): {len(combined)}")

    combined = combined.drop(columns=["pm25_neighbor_avg"])

    with open(REGISTRY_PATH, encoding="utf-8") as f:
        registry = json.load(f)
    registry_provinces = {p["province_en"]: p for p in registry["provinces"]}

    print(f"\nคำนวณ pm25_neighbor_avg ใหม่ด้วย k={K_NEW}...")
    combined = build_mod.add_spatial_lag_feature(combined, registry_provinces, k=K_NEW)

    n_missing = combined["pm25_neighbor_avg"].isna().sum()
    print(f"แถวที่ไม่มีเพื่อนบ้านให้ข้อมูล ณ เวลานั้น (ตัดทิ้ง): {n_missing}/{len(combined)}")
    combined = combined.dropna(subset=["pm25_neighbor_avg"]).reset_index(drop=True)

    combined = combined[build_mod.ID_COLS + ["pm25"] + build_mod.FEATURE_COLS + build_mod.TARGET_COLS]
    combined.to_csv(OUT_PATH, index=False)

    print(f"\nบันทึก dataset ใหม่ (k={K_NEW}) ที่: {OUT_PATH}")
    print(f"แถวรวมสุดท้าย: {len(combined)}")
    train_rows = combined[~combined["is_zero_shot"]]
    print(f"แถวของ 17 จังหวัดเทรน: {len(train_rows)}")


if __name__ == "__main__":
    main()

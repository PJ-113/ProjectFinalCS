"""
Task F — เพิ่ม pm25_neighbor_avg เข้าไปในชุดข้อมูลเทรนที่มีอยู่แล้ว โดยไม่ต้อง fetch ใหม่จาก
OpenAQ/Open-Meteo (feature นี้คำนวณได้จาก pm25 ที่มีอยู่แล้วในไฟล์ raw ล้วนๆ ไม่ใช่ข้อมูลภายนอกใหม่)
— reuse add_spatial_lag_feature() จาก build_raw_dataset_multiprovince.py ตรงๆ (ซึ่งแก้ให้รองรับ
feature นี้แล้ว จะได้สอดคล้องกันถ้ามีการ rebuild เต็มจาก scratch ในอนาคต)

เหตุผลที่แยกเป็นสคริปต์ต่างหาก ไม่ได้ rerun build_raw_dataset_multiprovince.py ทั้งไฟล์: การ fetch
จาก OpenAQ/Open-Meteo ของ 20 จังหวัด x ~2.5 ปีข้อมูล ใช้เวลานานมาก (chunked+retry ต่อจังหวัด) และ
ข้อมูล pm25 ที่มีอยู่แล้วในไฟล์ raw ปัจจุบันก็เพียงพอคำนวณ feature นี้ได้ครบถ้วนอยู่แล้ว
"""
import importlib.util
import json
import os

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE_CSV = os.path.join(_ROOT, "data", "multiprovince_pm25_features_raw_20260913_012629.csv")
REGISTRY_PATH = os.path.join(_ROOT, "data", "province_station_registry.json")
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
OUT_PATH = os.path.join(_ROOT, "data", f"multiprovince_pm25_features_raw_spatial_{TIMESTAMP}.csv")


def main():
    spec = importlib.util.spec_from_file_location(
        "build_raw_dataset_multiprovince", os.path.join(_ROOT, "scripts", "build_raw_dataset_multiprovince.py")
    )
    build_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build_mod)

    with open(REGISTRY_PATH, encoding="utf-8") as f:
        registry = json.load(f)
    registry_provinces = {p["province_en"]: p for p in registry["provinces"]}

    print(f"โหลด: {SOURCE_CSV}")
    combined = pd.read_csv(SOURCE_CSV)
    combined["datetime_utc"] = pd.to_datetime(combined["datetime_utc"])
    print(f"แถวเดิม: {len(combined)}")

    print("คำนวณ pm25_neighbor_avg...")
    combined = build_mod.add_spatial_lag_feature(combined, registry_provinces)
    n_missing = combined["pm25_neighbor_avg"].isna().sum()
    print(f"แถวที่ไม่มีเพื่อนบ้านให้ข้อมูล ณ เวลานั้น (ตัดทิ้ง): {n_missing}/{len(combined)}")
    combined = combined.dropna(subset=["pm25_neighbor_avg"]).reset_index(drop=True)

    id_cols = ["province_th", "province_en", "is_zero_shot", "datetime_utc"]
    target_cols = [f"pm25_t+{h}" for h in range(1, 25)]
    combined = combined[id_cols + ["pm25"] + build_mod.FEATURE_COLS + target_cols]

    combined.to_csv(OUT_PATH, index=False)
    print(f"\nบันทึกแล้วที่: {OUT_PATH}")
    print(f"แถวสุดท้าย: {len(combined)}")
    print(f"columns: {list(combined.columns)}")


if __name__ == "__main__":
    main()

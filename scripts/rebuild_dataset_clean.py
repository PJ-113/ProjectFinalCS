"""
สร้าง training dataset ใหม่ทั้งสาย: เรียกสคริปต์ 4 ตัวต่อกัน ส่ง path ผลลัพธ์ของขั้นก่อนหน้าเป็น input ของขั้นถัดไป
  1) build_raw_dataset_multiprovince.py        ดึง OpenAQ + Open-Meteo, lag/rolling/target, neighbor k=3
  2) augment_raw_dataset_with_spatial_lag_k1.py pm25_neighbor_avg ด้วย k=1
  3) compute_fire_hotspot_feature.py            fire_hotspot_count_100km_24h
  4) compute_neighbor_trend_feature.py          nb_trend_12h
เขียนไฟล์ผลลัพธ์ใหม่ทุกขั้น (ไม่ทับของเดิม) + manifest data/rebuild_clean_manifest_{ts}.json
"""
import importlib.util
import json
import os
import time

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_ROOT, "data")
OLD_DATASET = os.path.join(DATA_DIR, "multiprovince_pm25_features_raw_spatial_k1_firehotspot_nbtrend_20260919_135009.csv")
TS = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")


def load(fname):
    spec = importlib.util.spec_from_file_location(fname.replace(".py", ""), os.path.join(_ROOT, "scripts", fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def banner(text):
    print("\n" + "#" * 100 + f"\n# {text}\n" + "#" * 100, flush=True)


def main():
    t0 = time.time()
    with open(os.path.join(DATA_DIR, "province_station_registry.json"), encoding="utf-8") as f:
        reg = json.load(f)["provinces"]
    offsets = {p["province_en"]: p.get("openaq_timestamp_offset_hours") for p in reg}
    assert all(v is not None for v in offsets.values()), "registry ยังไม่มี openaq_timestamp_offset_hours — รัน annotate_registry_timestamp_offset.py ก่อน"
    print("offset ป้ายเวลาต่อจังหวัด (ชม.):", {k: v for k, v in offsets.items() if v})

    banner("1/4 build_raw_dataset_multiprovince (ดึง OpenAQ + weather ใหม่ทั้งหมด)")
    m1 = load("build_raw_dataset_multiprovince.py")
    m1.main()
    raw_path = m1.OUT_PATH

    banner("2/4 pm25_neighbor_avg k=1")
    m2 = load("augment_raw_dataset_with_spatial_lag_k1.py")
    m2.SOURCE_PATH = raw_path
    m2.main()
    k1_path = m2.OUT_PATH

    banner("3/4 fire_hotspot_count_100km_24h")
    m3 = load("compute_fire_hotspot_feature.py")
    m3.K1_DATA_PATH = k1_path
    m3.main()
    fire_path = m3.OUT_PATH

    banner("4/4 nb_trend_12h")
    m4 = load("compute_neighbor_trend_feature.py")
    m4.SOURCE = fire_path
    m4.main()
    final_path = m4.OUT

    manifest = {"created": TS, "offsets": offsets, "raw_k3": raw_path, "k1": k1_path, "with_fire": fire_path, "final": final_path,
                "old_dataset": OLD_DATASET}
    man_path = os.path.join(DATA_DIR, f"rebuild_clean_manifest_{TS}.json")
    with open(man_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    banner("สรุป: จำนวนแถวต่อจังหวัด เทียบ dataset เดิม")
    new = pd.read_csv(final_path, usecols=["province_en", "is_zero_shot", "datetime_utc"])
    old = pd.read_csv(OLD_DATASET, usecols=["province_en", "datetime_utc"])
    cmp_df = pd.DataFrame({"ใหม่": new.groupby("province_en").size(), "เดิม": old.groupby("province_en").size()})
    cmp_df["offset"] = pd.Series(offsets)
    cmp_df["ต่าง"] = cmp_df["ใหม่"] - cmp_df["เดิม"]
    print(cmp_df.to_string())
    print(f"\nรวม ใหม่ {len(new)} แถว / เดิม {len(old)} แถว; ใช้เวลา {(time.time() - t0) / 60:.1f} นาที")
    print(f"manifest: {man_path}\ndataset สุดท้าย: {final_path}")


if __name__ == "__main__":
    main()

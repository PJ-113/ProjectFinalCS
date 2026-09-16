"""
เวอร์ชันแก้ไขของ fetch_province_building_density.py — building footprint count เจอปัญหา OSM
bulk-import artifact ชัดเจน (มหาสารคาม: อาคาร 2,125 หลังใน 1กม. tag "roof" ล้วน 100%, มีแค่
0.05% ที่มี tag รายละเอียด = สัญญาณ import อัตโนมัติ ไม่ใช่ mapping โดยชุมชนจริง — ดู
diagnose_building_count_anomaly.py) เปลี่ยนมานับ **amenity + shop POI** แทน เพราะต้องมีคน tag
ธุรกิจ/สถานที่จริงทีละจุด ไม่ใช่สิ่งที่ import อัตโนมัติได้ง่าย — ตรวจสอบแล้ว (diagnose_amenity_
density.py) ให้ลำดับเมืองที่สมเหตุสมผล (ขอนแก่น/โคราช สูงสุดจริง ตรงกับขนาดเมืองจริง)
"""
import json
import os
import time

import requests

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_PATH = os.path.join(_ROOT, "data", "province_station_registry.json")
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
RADIUS_M = 3000
REQUEST_SLEEP_S = 6
HEADERS = {"User-Agent": "pm25-forecast-capstone-project (educational use, 20 one-off queries)"}


def save_registry(registry):
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)


def query_amenity_count(lat, lon):
    query = f"""
    [out:json][timeout:40];
    (
      node["amenity"](around:{RADIUS_M},{lat},{lon});
      way["amenity"](around:{RADIUS_M},{lat},{lon});
      node["shop"](around:{RADIUS_M},{lat},{lon});
      way["shop"](around:{RADIUS_M},{lat},{lon});
    );
    out count;
    """
    for attempt in range(6):
        try:
            r = requests.post(OVERPASS_URL, data={"data": query}, headers=HEADERS, timeout=50)
        except requests.exceptions.RequestException as e:
            print(f"    !! network error (attempt {attempt+1}): {e}, retry ใน 10s")
            time.sleep(10)
            continue
        if r.status_code in (429, 406, 502, 503, 504):
            print(f"    !! HTTP {r.status_code} (attempt {attempt+1}), รอ 30s ตาม usage policy")
            time.sleep(30)
            continue
        r.raise_for_status()
        data = r.json()
        count_tag = data["elements"][0]["tags"]
        return int(count_tag["total"])
    raise RuntimeError("ดึง amenity count ไม่สำเร็จหลัง retry")


def main():
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        registry = json.load(f)

    provinces = registry["provinces"]
    print(f"เรียก Overpass API (amenity+shop) สำหรับ {len(provinces)} จังหวัด (รัศมี {RADIUS_M}m)...")

    n_skipped = sum(1 for p in provinces if "amenity_density_3km" in p)
    if n_skipped:
        print(f"ข้าม {n_skipped} จังหวัดที่มีข้อมูลอยู่แล้ว (resume จากรอบก่อน)")

    for i, p in enumerate(provinces):
        if "amenity_density_3km" in p:
            continue
        count = query_amenity_count(p["lat"], p["lon"])
        p["amenity_density_3km"] = count
        print(f"  [{i+1}/{len(provinces)}] {p['province_th']} ({p['province_en']}): {count} amenity/shop")
        save_registry(registry)
        if i < len(provinces) - 1:
            time.sleep(REQUEST_SLEEP_S)

    counts = [p["amenity_density_3km"] for p in provinces]
    print(f"\nบันทึกกลับเข้า: {REGISTRY_PATH}")
    print(f"amenity_density_3km ต่ำสุด: {min(counts)}, สูงสุด: {max(counts)}, เฉลี่ย: {sum(counts)/len(counts):.0f}")

    ranked = sorted(provinces, key=lambda p: p["amenity_density_3km"], reverse=True)
    print("\nอันดับความหนาแน่นเมือง (มากไปน้อย):")
    for p in ranked:
        print(f"  {p['province_th']:12s} {p['amenity_density_3km']:6d}")


if __name__ == "__main__":
    main()

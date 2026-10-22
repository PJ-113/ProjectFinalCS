"""
ทดสอบว่า amenity/shop POI count เป็นตัวชี้วัดความหนาแน่นเมืองที่น่าเชื่อถือกว่า building count ไหม
ตรวจ 2 อย่าง: (1) จำนวนรวมที่รัศมี 3กม. เทียบว่าลำดับเมืองใหญ่/เล็กสมเหตุสมผลไหม
(2) tag breakdown ที่รัศมี 1กม. เช็คความหลากหลายของ tag
"""
import json
import os
import time
from collections import Counter

import requests

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_PATH = os.path.join(_ROOT, "data", "province_station_registry.json")
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
HEADERS = {"User-Agent": "pm25-forecast-capstone-project (data quality check, 10 queries)"}

SUSPECT_PROVINCES = ["mahasarakham", "mukdahan", "sisaket"]
REFERENCE_PROVINCES = ["nakhonratchasima", "khonkaen"]


def overpass(query):
    for attempt in range(6):
        try:
            r = requests.post(OVERPASS_URL, data={"data": query}, headers=HEADERS, timeout=50)
        except requests.exceptions.RequestException as e:
            print(f"    !! network error (attempt {attempt+1}): {e}, retry ใน 10s")
            time.sleep(10)
            continue
        if r.status_code in (429, 406, 502, 503, 504):
            print(f"    !! HTTP {r.status_code} (attempt {attempt+1}), รอ 30s")
            time.sleep(30)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError("ดึงข้อมูลไม่สำเร็จหลัง retry")


def count_at_radius(lat, lon, radius_m):
    query = f"""
    [out:json][timeout:40];
    (
      node["amenity"](around:{radius_m},{lat},{lon});
      way["amenity"](around:{radius_m},{lat},{lon});
      node["shop"](around:{radius_m},{lat},{lon});
      way["shop"](around:{radius_m},{lat},{lon});
    );
    out count;
    """
    data = overpass(query)
    return int(data["elements"][0]["tags"]["total"])


def tag_breakdown_at_radius(lat, lon, radius_m):
    query = f"""
    [out:json][timeout:40];
    (
      node["amenity"](around:{radius_m},{lat},{lon});
      node["shop"](around:{radius_m},{lat},{lon});
    );
    out tags;
    """
    data = overpass(query)
    elements = data["elements"]
    vals = Counter()
    for e in elements:
        tags = e.get("tags", {})
        vals[tags.get("amenity") or tags.get("shop") or "(?)"] += 1
    named = sum(1 for e in elements if "name" in e.get("tags", {}))
    return len(elements), vals, named


def main():
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        registry = json.load(f)
    provinces = {p["province_en"]: p for p in registry["provinces"]}
    targets = SUSPECT_PROVINCES + REFERENCE_PROVINCES

    print("=== จำนวน amenity+shop รวม ที่รัศมี 3km (รัศมีเดียวกับ feature จริง) ===\n")
    counts_3km = {}
    for i, slug in enumerate(targets):
        p = provinces[slug]
        c = count_at_radius(p["lat"], p["lon"], 3000)
        counts_3km[slug] = c
        tag = "[สงสัย]" if slug in SUSPECT_PROVINCES else "[อ้างอิง เมืองใหญ่จริง]"
        print(f"  {p['province_th']:12s} ({slug}) {tag}: {c}")
        if i < len(targets) - 1:
            time.sleep(3)

    print("\n=== Tag breakdown ที่รัศมี 1km (เทียบเคียงจุดที่เคยเจอปัญหา building) ===\n")
    for i, slug in enumerate(targets):
        p = provinces[slug]
        n, vals, named = tag_breakdown_at_radius(p["lat"], p["lon"], 1000)
        tag = "[สงสัย]" if slug in SUSPECT_PROVINCES else "[อ้างอิง เมืองใหญ่จริง]"
        print(f"{p['province_th']} ({slug}) {tag}")
        print(f"  จำนวน: {n}, มีชื่อสถานที่จริง (name=*): {named}/{n} ({named/n*100 if n else 0:.0f}%)")
        print(f"  breakdown: {dict(vals.most_common(8))}\n")
        if i < len(targets) - 1:
            time.sleep(3)

    print("=== สรุปลำดับ (3km) ===")
    for slug, c in sorted(counts_3km.items(), key=lambda kv: -kv[1]):
        print(f"  {provinces[slug]['province_th']:12s} {c}")


if __name__ == "__main__":
    main()

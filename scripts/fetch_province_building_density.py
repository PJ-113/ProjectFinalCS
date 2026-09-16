"""
ดึงความหนาแน่นของอาคาร (จำนวนอาคารในรัศมี 3 กม. รอบพิกัดสถานีจริง) เข้า
data/province_station_registry.json — ตอบ feedback อาจารย์เรื่องเขตเมือง vs เขตนอกเมือง
(ประเด็นสุดท้ายที่เหลือจาก spatial correlation feedback)

ใช้ OpenStreetMap Overpass API (https://overpass-api.de/api/interpreter) — ฟรี ไม่ต้อง API key
สำหรับ casual use (<10,000 query/วัน, เราใช้แค่ 20) ใส่ User-Agent ตาม usage policy + sleep
ระหว่าง request กันโดน rate limit ของ free server (ไม่ยิง parallel requests)

นับที่พิกัดสถานีจริง (ไม่ใช่ค่าเฉลี่ยทั้งจังหวัด) เพราะสถานีอาจอยู่ใจกลางเมืองหรือชานเมืองก็ได้
"""
import json
import os
import time

import requests

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_PATH = os.path.join(_ROOT, "data", "province_station_registry.json")
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
RADIUS_M = 3000
REQUEST_SLEEP_S = 6  # free public server เจอ 429 ถ้ายิงถี่เกินไป — เว้นช่วงให้กว้างขึ้น
HEADERS = {"User-Agent": "pm25-forecast-capstone-project (educational use, 20 one-off queries)"}


def save_registry(registry):
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)


def query_building_count(lat, lon):
    query = f"""
    [out:json][timeout:25];
    (
      way["building"](around:{RADIUS_M},{lat},{lon});
      relation["building"](around:{RADIUS_M},{lat},{lon});
    );
    out count;
    """
    for attempt in range(6):
        try:
            r = requests.post(OVERPASS_URL, data={"data": query}, headers=HEADERS, timeout=40)
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
    raise RuntimeError("ดึง building count ไม่สำเร็จหลัง retry")


def main():
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        registry = json.load(f)

    provinces = registry["provinces"]
    print(f"เรียก Overpass API สำหรับ {len(provinces)} จังหวัด (รัศมี {RADIUS_M}m ต่อจุด)...")

    n_skipped = sum(1 for p in provinces if "building_count_3km" in p)
    if n_skipped:
        print(f"ข้าม {n_skipped} จังหวัดที่มีข้อมูลอยู่แล้ว (resume จากรอบก่อน)")

    for i, p in enumerate(provinces):
        if "building_count_3km" in p:
            continue
        count = query_building_count(p["lat"], p["lon"])
        p["building_count_3km"] = count
        print(f"  [{i+1}/{len(provinces)}] {p['province_th']} ({p['province_en']}): {count} อาคาร")
        save_registry(registry)  # checkpoint ทันทีต่อจังหวัด กันพัง Overpass rate-limit แล้วเสียของเดิม
        if i < len(provinces) - 1:
            time.sleep(REQUEST_SLEEP_S)

    counts = [p["building_count_3km"] for p in provinces]
    print(f"\nบันทึกกลับเข้า: {REGISTRY_PATH}")
    print(f"building_count_3km ต่ำสุด: {min(counts)}, สูงสุด: {max(counts)}, เฉลี่ย: {sum(counts)/len(counts):.0f}")

    ranked = sorted(provinces, key=lambda p: p["building_count_3km"], reverse=True)
    print("\nอันดับความหนาแน่นเมือง (มากไปน้อย):")
    for p in ranked:
        print(f"  {p['province_th']:12s} {p['building_count_3km']:6d}")


if __name__ == "__main__":
    main()

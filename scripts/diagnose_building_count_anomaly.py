"""
ตรวจสอบคุณภาพข้อมูล building_count_3km: ดึง building tag ของอาคารในรัศมี 1 กม. ที่จุด "ผิดปกติ"
(มหาสารคาม, มุกดาหาร, ศรีสะเกษ) เทียบกับจุด "อ้างอิง" (นครราชสีมา, ขอนแก่น) และดู breakdown
ของประเภท tag กับ spacing ระหว่างอาคาร เพื่อแยก bulk/cadastral import จากความหนาแน่นเมืองจริง
"""
import json
import os
import time
from collections import Counter

import requests

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_PATH = os.path.join(_ROOT, "data", "province_station_registry.json")
OVERPASS_URL = "https://overpass-api.de/api/interpreter"
HEADERS = {"User-Agent": "pm25-forecast-capstone-project (data quality check, 5 queries)"}
CHECK_RADIUS_M = 1000

SUSPECT_PROVINCES = ["mahasarakham", "mukdahan", "sisaket"]
REFERENCE_PROVINCES = ["nakhonratchasima", "khonkaen"]


def query_building_tags(lat, lon, radius_m):
    query = f"""
    [out:json][timeout:40];
    way["building"](around:{radius_m},{lat},{lon});
    out tags center;
    """
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
        return r.json()["elements"]
    raise RuntimeError("ดึงข้อมูลไม่สำเร็จหลัง retry")


def nearest_neighbor_spacing_stats(elements):
    """ระยะห่างแบบ approx (องศา * ~111km) ไปยังอาคารที่ใกล้ที่สุด ต่ออาคาร — ถ้า spacing
    สม่ำเสมอผิดปกติ (std ต่ำมากเทียบ mean) = สัญญาณของ grid/import ไม่ใช่เมืองที่โตแบบธรรมชาติ"""
    pts = [(e["center"]["lat"], e["center"]["lon"]) for e in elements if "center" in e]
    if len(pts) < 10:
        return None, None
    import math
    def dist_m(a, b):
        return math.hypot((a[0]-b[0])*111000, (a[1]-b[1])*111000*math.cos(math.radians(a[0])))
    # sample สูงสุด 800 จุดกันช้าเกินไป (O(n^2))
    sample = pts[:800]
    nn = []
    for i, p in enumerate(sample):
        best = min(dist_m(p, q) for j, q in enumerate(sample) if j != i)
        nn.append(best)
    mean_nn = sum(nn) / len(nn)
    var = sum((x - mean_nn) ** 2 for x in nn) / len(nn)
    return mean_nn, var ** 0.5


def main():
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        registry = json.load(f)
    provinces = {p["province_en"]: p for p in registry["provinces"]}

    targets = SUSPECT_PROVINCES + REFERENCE_PROVINCES
    print(f"ตรวจสอบ building tag breakdown ที่รัศมี {CHECK_RADIUS_M}m (เล็กกว่าตอน fetch หลัก "
          f"เพื่อกันข้อมูลหนักเกิน) สำหรับ {len(targets)} จังหวัด\n")

    for i, slug in enumerate(targets):
        p = provinces[slug]
        tag_label = "[จุดที่สงสัย]" if slug in SUSPECT_PROVINCES else "[จุดอ้างอิง เมืองใหญ่จริง]"
        print(f"{'='*80}\n{p['province_th']} ({slug}) {tag_label}  lat={p['lat']}, lon={p['lon']}\n{'='*80}")

        elements = query_building_tags(p["lat"], p["lon"], CHECK_RADIUS_M)
        print(f"  จำนวนอาคารในรัศมี {CHECK_RADIUS_M}m: {len(elements)}")

        building_vals = Counter(e.get("tags", {}).get("building", "(no tag)") for e in elements)
        print(f"  breakdown ค่า building=*: {dict(building_vals.most_common(8))}")

        other_tag_counts = Counter()
        for e in elements:
            other_tag_counts.update(k for k in e.get("tags", {}).keys() if k != "building")
        print(f"  จำนวนอาคารที่มี tag อื่นนอกจาก building=* ประกอบ (เช่น name/addr): "
              f"{sum(1 for e in elements if len(e.get('tags', {})) > 1)}/{len(elements)}")

        mean_nn, std_nn = nearest_neighbor_spacing_stats(elements)
        if mean_nn is not None:
            print(f"  ระยะห่างเฉลี่ยไปอาคารใกล้สุด: mean={mean_nn:.1f}m, std={std_nn:.1f}m "
                  f"(std/mean={std_nn/mean_nn:.2f} — ยิ่งต่ำ = spacing สม่ำเสมอผิดธรรมชาติ ยิ่งสูง = กระจายตามธรรมชาติ)")
        print()

        if i < len(targets) - 1:
            time.sleep(3)


if __name__ == "__main__":
    main()

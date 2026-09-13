"""
Phase 0 (แผนขยาย 17+3 จังหวัด): หา OpenAQ location/sensor ID ของสถานีวัด PM2.5
อ้างอิง (reference-grade เท่าที่มี) ใน 20 จังหวัดภาคอีสาน แล้วบันทึกเป็น registry
กลาง (JSON) ให้ data-collection script และ training script ในเฟสถัดไปใช้ร่วมกัน

ห้ามเดา location/sensor ID เอง — ทุกค่า query จาก OpenAQ v3 API จริง
(coordinates + radius search รอบเมืองแต่ละจังหวัด, radius สูงสุดที่ v3 อนุญาตคือ 25km)
แล้วเลือกสถานีที่วัด pm25 ที่อยู่ใกล้จุดค้นหาที่สุด — ขอนแก่นใช้ location 7747/sensor
1304249 ตรงตามที่ CLAUDE.md ระบุไว้แล้ว (ยืนยันซ้ำด้วย query จริง ไม่ hardcode ข้าม)

จังหวัดที่ไม่เจอสถานีในรัศมี 25km หรือเจอมากกว่า 1 ตัวเลือกที่ก้ำกึ่งกัน จะถูก print
ออกมาให้ตรวจสอบเอง ไม่เลือกให้อัตโนมัติแบบเดา
"""
import json
import os
import time

import pandas as pd
import requests

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(_ROOT, ".env")
OUT_PATH = os.path.join(_ROOT, "data", "province_station_registry.json")

RADIUS_M = 25000  # ค่าสูงสุดที่ OpenAQ v3 /v3/locations อนุญาต
REQUEST_SLEEP_S = 0.5
PREFERRED_PROVIDER = "Air4Thai"  # = เครือข่าย PCD ทางการของไทย ยืนยันแล้วว่าขอนแก่น 7747 ก็เป็น provider นี้
WINDOW_START = pd.Timestamp("2024-01-01T00:00:00Z")
WINDOW_END = pd.Timestamp("2026-07-02T00:00:00Z")


def _load_env_key(name):
    with open(ENV_PATH, encoding="utf-8") as f:
        for line in f:
            if line.startswith(f"{name}="):
                return line.strip().split("=", 1)[1]
    raise RuntimeError(f"{name} not found in .env")


OPENAQ_KEY = _load_env_key("OPENAQ_KEY")
HEADERS = {"X-API-Key": OPENAQ_KEY}
BASE = "https://api.openaq.org/v3"

# จุดค้นหา = พิกัดเมือง/ศาลากลางจังหวัด (โดยประมาณ) ใช้เป็นศูนย์กลางรัศมีค้นหาเท่านั้น
# lat/lon จริงของสถานีที่ deploy จะมาจากผลลัพธ์ API ไม่ใช่ค่าพวกนี้
PROVINCES = [
    # --- 17 จังหวัดเทรน ---
    {"th": "ขอนแก่น", "en": "khonkaen", "lat": 16.44538333, "lon": 102.83525, "zero_shot": False,
     "known_location_id": 7747},  # ยืนยันจาก CLAUDE.md แล้ว ไม่ต้อง radius-search
    {"th": "เลย", "en": "loei", "lat": 17.4860, "lon": 101.7223, "zero_shot": False},
    {"th": "หนองคาย", "en": "nongkhai", "lat": 17.8783, "lon": 102.7420, "zero_shot": False},
    {"th": "อุดรธานี", "en": "udonthani", "lat": 17.4156, "lon": 102.7859, "zero_shot": False},
    {"th": "นครราชสีมา", "en": "nakhonratchasima", "lat": 14.9799, "lon": 102.0978, "zero_shot": False},
    {"th": "อุบลราชธานี", "en": "ubonratchathani", "lat": 15.2286, "lon": 104.8564, "zero_shot": False},
    {"th": "บุรีรัมย์", "en": "buriram", "lat": 14.9930, "lon": 103.1029, "zero_shot": False},
    {"th": "สุรินทร์", "en": "surin", "lat": 14.8828, "lon": 103.4936, "zero_shot": False},
    {"th": "ศรีสะเกษ", "en": "sisaket", "lat": 15.1186, "lon": 104.3220, "zero_shot": False},
    {"th": "สกลนคร", "en": "sakonnakhon", "lat": 17.1546, "lon": 104.1468, "zero_shot": False},
    {"th": "กาฬสินธุ์", "en": "kalasin", "lat": 16.4314, "lon": 103.5060, "zero_shot": False},
    {"th": "มุกดาหาร", "en": "mukdahan", "lat": 16.5453, "lon": 104.7233, "zero_shot": False},
    {"th": "ยโสธร", "en": "yasothon", "lat": 15.7922, "lon": 104.1451, "zero_shot": False},
    {"th": "ชัยภูมิ", "en": "chaiyaphum", "lat": 15.8068, "lon": 102.0316, "zero_shot": False},
    {"th": "อำนาจเจริญ", "en": "amnatcharoen", "lat": 15.8656, "lon": 104.6259, "zero_shot": False},
    {"th": "หนองบัวลำภู", "en": "nongbualamphu", "lat": 17.2216, "lon": 102.4260, "zero_shot": False},
    {"th": "บึงกาฬ", "en": "buengkan", "lat": 18.3609, "lon": 103.6466, "zero_shot": False},
    # --- 3 จังหวัด zero-shot (ให้บริการได้ แต่ไม่ได้อยู่ในชุดเทรน) ---
    {"th": "นครพนม", "en": "nakhonphanom", "lat": 17.4107, "lon": 104.7791, "zero_shot": True},
    {"th": "ร้อยเอ็ด", "en": "roiet", "lat": 16.0538, "lon": 103.6520, "zero_shot": True},
    {"th": "มหาสารคาม", "en": "mahasarakham", "lat": 16.1852, "lon": 103.3010, "zero_shot": True,
     "expected_low_cost_sensor": True},  # ผู้ใช้ระบุว่าเป็นเซนเซอร์ AirGradient ไม่ใช่ reference-grade
]


def _get(url, params=None, max_retries=5):
    for attempt in range(max_retries):
        r = requests.get(url, headers=HEADERS, params=params, timeout=30)
        if r.status_code == 429:
            wait = 2 ** attempt
            print(f"    rate limited, retry in {wait}s...")
            time.sleep(wait)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"gave up after {max_retries} retries: {url}")


def get_pm25_parameter_id():
    """หา parameter id ของ pm25 จาก API จริง ไม่ hardcode"""
    data = _get(f"{BASE}/parameters", params={"limit": 400})
    for p in data.get("results", []):
        if p.get("name", "").lower() == "pm25":
            return p["id"]
    raise RuntimeError("ไม่เจอ parameter 'pm25' ใน /v3/parameters response")


def find_pm25_sensor_in_location(loc):
    """คืน sensor id + parameter display name ของ pm25 ใน location detail, หรือ None ถ้าไม่มี

    บาง location (เช่น Air4Thai หลายจังหวัด) มี pm25 sensor ซ้ำ 2 ตัว: ตัวเก่า (id เลขน้อย
    เช่น 1304xxx) ที่ยังมีข้อมูลจริงใช้งานได้ กับตัวใหม่ (id เลขเยอะ เช่น 1631xxxx) ที่พบว่า
    /v3/sensors/{id} คืน 500 Internal Server Error ทุกครั้ง (ยืนยันด้วย query จริงหลายรอบ ไม่ใช่
    ปัญหาชั่วคราว) จึงเลือก sensor id ที่น้อยที่สุดเสมอ (สม่ำเสมอกับทุกจังหวัดที่ query สำเร็จ
    ปกติอยู่แล้ว ซึ่งล้วนมี sensor id หลักเดียวเลขน้อย)"""
    pm25_sensor_ids = [s["id"] for s in loc.get("sensors", []) if s.get("parameter", {}).get("name", "").lower() == "pm25"]
    if not pm25_sensor_ids:
        return None, None
    return min(pm25_sensor_ids), "pm25"


def haversine_km(lat1, lon1, lat2, lon2):
    import math
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(a ** 0.5)


def coverage_overlap_days(datetime_first, datetime_last):
    """วันที่ location มีข้อมูลจริง ซ้อนทับกับช่วง target window (2024-01-01..2026-07-02) กี่วัน"""
    if not datetime_first or not datetime_last:
        return 0.0
    first = pd.Timestamp(datetime_first["utc"])
    last = pd.Timestamp(datetime_last["utc"])
    start = max(first, WINDOW_START)
    end = min(last, WINDOW_END)
    return max(0.0, (end - start).total_seconds() / 86400.0)


def resolve_known_location(location_id, province):
    detail = _get(f"{BASE}/locations/{location_id}")
    loc = detail["results"][0]
    sensor_id, _ = find_pm25_sensor_in_location(loc)
    coords = loc.get("coordinates", {})
    return {
        "location_id": loc["id"],
        "location_name": loc.get("name"),
        "sensor_id_pm25": sensor_id,
        "lat": coords.get("latitude"),
        "lon": coords.get("longitude"),
        "provider": (loc.get("provider") or {}).get("name"),
        "distance_km_from_search_point": round(
            haversine_km(province["lat"], province["lon"], coords.get("latitude"), coords.get("longitude")), 2
        ) if coords.get("latitude") is not None else None,
        "coverage_overlap_days": round(coverage_overlap_days(loc.get("datetimeFirst"), loc.get("datetimeLast")), 1),
    }


def search_province(province, pm25_param_id):
    params = {
        "coordinates": f"{province['lat']},{province['lon']}",
        "radius": RADIUS_M,
        "parameters_id": pm25_param_id,
        "limit": 50,
    }
    data = _get(f"{BASE}/locations", params=params)
    results = data.get("results", [])
    candidates = []
    for loc in results:
        sensor_id, _ = find_pm25_sensor_in_location(loc)
        if sensor_id is None:
            continue
        coords = loc.get("coordinates", {})
        dist = haversine_km(province["lat"], province["lon"], coords.get("latitude"), coords.get("longitude"))
        candidates.append({
            "location_id": loc["id"],
            "location_name": loc.get("name"),
            "sensor_id_pm25": sensor_id,
            "lat": coords.get("latitude"),
            "lon": coords.get("longitude"),
            "provider": (loc.get("provider") or {}).get("name"),
            "distance_km_from_search_point": round(dist, 2),
        })
    # เรียงก่อนด้วยระยะทาง แค่เพื่อจำกัดจำนวนที่ต้องยิง detail call เพิ่ม (เอาแค่ผู้เข้าชิงที่สมเหตุสมผล)
    candidates.sort(key=lambda c: c["distance_km_from_search_point"])
    top = candidates[:6]
    for c in top:
        time.sleep(REQUEST_SLEEP_S)
        detail = _get(f"{BASE}/locations/{c['location_id']}")["results"][0]
        c["coverage_overlap_days"] = round(
            coverage_overlap_days(detail.get("datetimeFirst"), detail.get("datetimeLast")), 1
        )
    # จัดอันดับจริง: ให้น้ำหนัก provider ที่ตรงเครือข่ายทางการ (Air4Thai) ก่อน แล้วค่อยดู
    # ความครอบคลุมช่วงเวลาที่ต้องใช้เทรน (คุมด้วยข้อมูลจริง ไม่ใช่เดา) แล้วค่อยดูระยะทางเป็นตัวตัดสินสุดท้าย
    top.sort(key=lambda c: (
        0 if c["provider"] == PREFERRED_PROVIDER else 1,
        -c["coverage_overlap_days"],
        c["distance_km_from_search_point"],
    ))
    return top


def main():
    print("หา parameter id ของ pm25 จาก OpenAQ API...")
    pm25_param_id = get_pm25_parameter_id()
    print(f"  pm25 parameter_id = {pm25_param_id}")

    registry = []
    needs_review = []

    for province in PROVINCES:
        print(f"\n{province['th']} ({province['en']})" + (" [zero-shot]" if province["zero_shot"] else ""))
        time.sleep(REQUEST_SLEEP_S)

        if "known_location_id" in province:
            resolved = resolve_known_location(province["known_location_id"], province)
            print(f"  ใช้ known_location_id={province['known_location_id']} (ยืนยันจาก CLAUDE.md): "
                  f"{resolved['location_name']} sensor={resolved['sensor_id_pm25']} "
                  f"dist={resolved['distance_km_from_search_point']}km")
            entry = {
                "province_th": province["th"], "province_en": province["en"],
                "is_zero_shot": province["zero_shot"], **resolved,
                "auto_selected": False, "n_candidates": 1,
            }
            registry.append(entry)
            continue

        candidates = search_province(province, pm25_param_id)
        print(f"  เจอ {len(candidates)} สถานีที่วัด pm25 ในรัศมี {RADIUS_M/1000:.0f}km (เรียงตาม provider={PREFERRED_PROVIDER} ก่อน แล้วดู coverage ช่วงเวลา แล้วค่อยระยะทาง)")
        for c in candidates:
            print(f"    - id={c['location_id']} sensor={c['sensor_id_pm25']} "
                  f"name={c['location_name']!r} provider={c['provider']!r} "
                  f"dist={c['distance_km_from_search_point']}km coverage={c['coverage_overlap_days']}d")

        if not candidates:
            print("  !! ไม่เจอสถานีเลย ต้องตรวจสอบเอง (ขยายรัศมีค้นหาด้วยมือ หรือหา location เพิ่มเติม)")
            needs_review.append({"province_th": province["th"], "province_en": province["en"], "reason": "no_candidates"})
            continue

        best = candidates[0]
        # ต้องตรวจสอบเอง 3 กรณี: (1) ไม่ใช่ provider ทางการ Air4Thai เลย (2) coverage ของช่วงเวลาที่ต้องใช้เทรนต่ำมาก
        # (3) มีตัวเลือก Air4Thai อื่นที่ coverage ใกล้เคียงกันมาก (คลุมเครือจริง ไม่ใช่แค่ระยะทางใกล้กัน)
        reasons = []
        if best["provider"] != PREFERRED_PROVIDER:
            reasons.append(f"provider ที่เลือกไม่ใช่ {PREFERRED_PROVIDER} (เป็น {best['provider']!r})")
        window_days = (WINDOW_END - WINDOW_START).total_seconds() / 86400.0
        if best["coverage_overlap_days"] < window_days * 0.5:
            reasons.append(f"coverage ของช่วง 2024-01-01..2026-07-02 ต่ำ ({best['coverage_overlap_days']}/{window_days:.0f} วัน)")
        runner_up = next((c for c in candidates[1:] if c["provider"] == PREFERRED_PROVIDER), None)
        if runner_up and abs(runner_up["coverage_overlap_days"] - best["coverage_overlap_days"]) < 30:
            reasons.append(f"มีสถานี {PREFERRED_PROVIDER} อื่น (id={runner_up['location_id']}) coverage ใกล้เคียงกันมาก")

        entry = {
            "province_th": province["th"], "province_en": province["en"],
            "is_zero_shot": province["zero_shot"],
            "location_id": best["location_id"], "location_name": best["location_name"],
            "sensor_id_pm25": best["sensor_id_pm25"], "lat": best["lat"], "lon": best["lon"],
            "provider": best["provider"], "distance_km_from_search_point": best["distance_km_from_search_point"],
            "coverage_overlap_days": best["coverage_overlap_days"],
            "auto_selected": True, "n_candidates": len(candidates),
        }
        registry.append(entry)
        if reasons:
            print(f"  !! ต้องตรวจสอบ: {'; '.join(reasons)}")
            needs_review.append({"province_th": province["th"], "province_en": province["en"],
                                  "reasons": reasons, "candidates": candidates[:4]})

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump({"pm25_parameter_id": pm25_param_id, "provinces": registry, "needs_review": needs_review},
                   f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}")
    print(f"บันทึกแล้วที่: {OUT_PATH}")
    print(f"จังหวัดที่ resolve ได้: {len(registry)}/{len(PROVINCES)}")
    if needs_review:
        print(f"\n!! จังหวัดที่ต้องตรวจสอบเพิ่ม ({len(needs_review)}):")
        for item in needs_review:
            print(f"  - {item['province_th']}: {item['reason']}")
    else:
        print("ไม่มีจังหวัดที่ต้องตรวจสอบเพิ่ม")

    khonkaen = next(p for p in registry if p["province_en"] == "khonkaen")
    assert khonkaen["location_id"] == 7747, "ขอนแก่นต้องตรงกับ location 7747 ตาม CLAUDE.md"
    assert khonkaen["sensor_id_pm25"] == 1304249, "ขอนแก่นต้องตรงกับ sensor 1304249 ตาม CLAUDE.md"
    print("\nsanity check ผ่าน: ขอนแก่นตรงกับ location 7747 / sensor 1304249 ตามที่ CLAUDE.md ระบุไว้")


if __name__ == "__main__":
    main()

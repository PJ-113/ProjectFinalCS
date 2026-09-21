"""
ตรวจสถานะสถานีของจังหวัดเลยใน OpenAQ (อ่านอย่างเดียว ไม่แก้ registry): ทำไมถึงไม่มีข้อมูลตั้งแต่ ~15 ก.ย. 2026
  (1) location ที่ลงทะเบียนไว้: datetimeFirst/datetimeLast และรายการ sensor ตามที่ OpenAQ รายงาน
  (2) reading ล่าสุดจริงของ sensor pm25 ที่ใช้ (จาก /measurements และ /hours ย้อนหลัง 30 วัน)
  (3) สถานีอื่นที่มี pm25 ในรัศมี 25 กม. จากพิกัดสถานีเดิม (ค่าสูงสุดที่ OpenAQ อนุญาต) พร้อมอายุ reading ล่าสุด — ดูว่ามีตัวสำรองไหม
ไม่พิมพ์คีย์ (อ่านจาก .env เฉพาะตอนสร้าง header)
"""
import json
import os
import time

import numpy as np
import pandas as pd
import requests

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = "https://api.openaq.org/v3"


def load_env_key(name):
    with open(os.path.join(_ROOT, ".env"), encoding="utf-8") as f:
        for line in f:
            if line.startswith(f"{name}="):
                return line.strip().split("=", 1)[1]
    raise RuntimeError(f"{name} not found in .env")


HEADERS = {"X-API-Key": load_env_key("OPENAQ_KEY")}


def get(url, params=None, retries=5):
    for attempt in range(retries):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=30)
        except requests.exceptions.RequestException:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        if r.status_code != 200:
            return {"_status": r.status_code, "results": []}
        return r.json()
    return {"_status": "failed", "results": []}


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def age_h(ts):
    if not ts:
        return None
    return round((pd.Timestamp.now(tz="UTC") - pd.Timestamp(ts)).total_seconds() / 3600, 1)


def last_reading(sensor_id, endpoint):
    end = pd.Timestamp.now(tz="UTC")
    params = {"datetime_from": (end - pd.Timedelta(days=30)).isoformat(), "datetime_to": end.isoformat(), "limit": 1000}
    data = get(f"{BASE}/sensors/{sensor_id}/{endpoint}", params)
    res = data.get("results", [])
    if not res:
        return None, data.get("_status", 200), 0
    key = "period" if endpoint == "hours" else "period"
    times = []
    for x in res:
        p = x.get(key, {}).get("datetimeTo", {}).get("utc") or x.get(key, {}).get("datetimeFrom", {}).get("utc")
        if p:
            times.append(pd.Timestamp(p))
    return (max(times) if times else None), data.get("_status", 200), len(res)


def main():
    reg = json.load(open(os.path.join(_ROOT, "data", "province_station_registry.json"), encoding="utf-8"))["provinces"]
    loei = next(p for p in reg if p["province_en"] == "loei")
    loc_id, sensor_id = loei["location_id"], loei["sensor_id_pm25"]
    print(f"เวลาตอนนี้ (UTC): {pd.Timestamp.now(tz='UTC')}")
    print(f"สถานีที่ลงทะเบียน: location {loc_id} ({loei['location_name']}) sensor {sensor_id} provider {loei['provider']}\n")

    print("=== (1) location ตามที่ OpenAQ รายงาน ===")
    d = get(f"{BASE}/locations/{loc_id}")
    res = d.get("results", [])
    if res:
        loc = res[0]
        first, last = (loc.get("datetimeFirst") or {}).get("utc"), (loc.get("datetimeLast") or {}).get("utc")
        print(f"  datetimeFirst={first}  datetimeLast={last}  (อายุ {age_h(last)} ชม.)  isMonitor={loc.get('isMonitor')}  "
              f"provider={(loc.get('provider') or {}).get('name')}")
        for s in loc.get("sensors", []):
            print(f"  sensor {s.get('id')} {s.get('parameter', {}).get('name')}")
    else:
        print(f"  ไม่พบ location (status={d.get('_status')})")

    print("\n=== (2) reading ล่าสุดจริงของ sensor ที่ใช้ (ย้อนหลัง 30 วัน) ===")
    for ep in ("measurements", "hours"):
        t, status, n = last_reading(sensor_id, ep)
        print(f"  /{ep}: {n} แถว  ล่าสุด={t}  อายุ={age_h(t)} ชม.  (status={status})")
    latest = get(f"{BASE}/sensors/{sensor_id}")
    lr = (latest.get("results") or [{}])[0]
    print(f"  /sensors/{sensor_id}: datetimeLast={(lr.get('datetimeLast') or {}).get('utc')}  latest={lr.get('latest')}")

    print("\n=== (3) สถานี pm25 อื่นในรัศมี 25 กม. จากสถานีเดิม ===")
    pid = next(p["id"] for p in get(f"{BASE}/parameters", {"limit": 400})["results"] if p.get("name", "").lower() == "pm25")
    cand = get(f"{BASE}/locations", {"coordinates": f"{loei['lat']},{loei['lon']}", "radius": 25000, "parameters_id": pid, "limit": 50})
    rows = []
    for loc in cand.get("results", []):
        c = loc.get("coordinates", {})
        last = (loc.get("datetimeLast") or {}).get("utc")
        rows.append({"location_id": loc["id"], "name": loc.get("name"), "provider": (loc.get("provider") or {}).get("name"),
                     "km": round(haversine_km(loei["lat"], loei["lon"], c.get("latitude"), c.get("longitude")), 1),
                     "datetimeLast": last, "age_h": age_h(last)})
    if rows:
        print(pd.DataFrame(rows).sort_values("km").to_string(index=False))
    else:
        print("  ไม่พบสถานี pm25 ในรัศมี 25 กม.")


if __name__ == "__main__":
    main()

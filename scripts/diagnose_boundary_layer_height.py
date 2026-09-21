"""
Phase C0 — เช็คว่า Open-Meteo คืน boundary_layer_height (และตัวแปรใกล้เคียง) ครบช่วงข้อมูลเทรนหรือไม่
ก่อนลงทุนดึงทุกจังหวัด (ธรรมเนียม "diagnose ก่อน fetch เต็ม" เหมือน diagnose_firms_hotspot_access.py)
เช็คทั้ง archive API (สำหรับเทรน) และ forecast API (สำหรับ live inference ในอนาคต) — ไม่ต้องใช้ API key
"""
import requests

KHONKAEN = (16.4322, 102.8236)
VARS = "boundary_layer_height,cape,cloud_cover,shortwave_radiation"


def check(name, url, params):
    r = requests.get(url, params=params, timeout=60)
    print(f"\n[{name}] HTTP {r.status_code}")
    if r.status_code != 200:
        print(r.text[:300])
        return
    h = r.json()["hourly"]
    n = len(h["time"])
    print(f"  ช่วง {h['time'][0]} -> {h['time'][-1]}  ({n} ชั่วโมง)")
    for v in VARS.split(","):
        vals = h.get(v)
        if vals is None:
            print(f"  {v}: ไม่มีในผลลัพธ์")
            continue
        valid = [x for x in vals if x is not None]
        print(f"  {v}: non-null {len(valid)}/{n}"
              + (f"  min={min(valid):.1f} max={max(valid):.1f}" if valid else ""))


if __name__ == "__main__":
    lat, lon = KHONKAEN
    # ช่วงเต็มครั้งเดียวคืน body ว่าง → แบ่งเป็นรายปี
    for start, end in (("2024-01-02", "2024-12-31"), ("2025-01-01", "2025-12-31"), ("2026-01-01", "2026-07-02")):
        check(f"archive {start}..{end}", "https://archive-api.open-meteo.com/v1/archive",
              {"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
               "hourly": VARS, "timezone": "UTC"})
    check("forecast (live, past 2d + next 2d)", "https://api.open-meteo.com/v1/forecast",
          {"latitude": lat, "longitude": lon, "past_days": 2, "forecast_days": 2,
           "hourly": VARS, "timezone": "UTC"})

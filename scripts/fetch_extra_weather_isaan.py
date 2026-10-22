"""
ดึงตัวแปร weather เพิ่ม (boundary_layer_height, shortwave_radiation, cloud_cover) จาก Open-Meteo archive
ทุก 20 จังหวัด ตามช่วงข้อมูลเทรน แบ่งดึงรายปี พร้อม retry/backoff
output: data/extra_weather_isaan_{timestamp}.csv (province_en, datetime_utc, 3 คอลัมน์)
"""
import json
import os
import time

import pandas as pd
import requests

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_ROOT, "data")
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
VARS = ["boundary_layer_height", "shortwave_radiation", "cloud_cover"]
CHUNKS = [("2024-01-02", "2024-12-31"), ("2025-01-01", "2025-12-31"), ("2026-01-01", "2026-07-02")]
REQUEST_SLEEP_S = 1.0


def fetch_chunk(lat, lon, start, end):
    params = {"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
              "hourly": ",".join(VARS), "timezone": "UTC"}
    for attempt in range(6):
        try:
            r = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params, timeout=60)
            if r.status_code == 200 and r.text.strip():
                h = r.json()["hourly"]
                df = pd.DataFrame({"datetime_utc": pd.to_datetime(h["time"]).tz_localize("UTC"),
                                   **{v: h[v] for v in VARS}})
                return df
            print(f"    !! HTTP {r.status_code} / body {len(r.text)} bytes (attempt {attempt+1}), retry ใน {2 ** attempt}s")
        except requests.exceptions.RequestException as e:
            print(f"    !! network error (attempt {attempt+1}): {e}, retry ใน {2 ** attempt}s")
        time.sleep(2 ** attempt)
    raise RuntimeError(f"ดึง {start}..{end} ไม่สำเร็จหลัง retry 6 ครั้ง")


def main():
    with open(os.path.join(DATA_DIR, "province_station_registry.json"), encoding="utf-8") as f:
        registry = json.load(f)
    frames = []
    for p in registry["provinces"]:
        parts = []
        for start, end in CHUNKS:
            parts.append(fetch_chunk(p["lat"], p["lon"], start, end))
            time.sleep(REQUEST_SLEEP_S)
        d = pd.concat(parts, ignore_index=True)
        d.insert(0, "province_en", p["province_en"])
        print(f"  {p['province_en']:14s} {len(d)} แถว  blh non-null {d['boundary_layer_height'].notna().mean():.0%}")
        frames.append(d)
    out = pd.concat(frames, ignore_index=True)
    out_path = os.path.join(DATA_DIR, f"extra_weather_isaan_{TIMESTAMP}.csv")
    out.to_csv(out_path, index=False)
    print(f"\nบันทึกที่: {out_path}  ({len(out)} แถว)")


if __name__ == "__main__":
    main()

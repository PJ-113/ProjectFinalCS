"""
ส่งออกตาราง climatology สภาพอากาศ (จังหวัด x เดือน x ชั่วโมง UTC) ให้ app.py ใช้เป็นทางสุดท้ายเมื่อดึง Open-Meteo ไม่ได้
และไม่มีค่า cache (ใช้ 7 feature สภาพอากาศเดียวกับที่โมเดลใช้)

ดึงสภาพอากาศครบทุกชั่วโมงของ 20 จังหวัดจาก Open-Meteo archive (ช่วงเดียวกับตอนเทรน) แล้วเฉลี่ยต่อ (เดือน, ชั่วโมง UTC)
ผลลัพธ์: models/weather_climatology_<timestamp>.json แล้วแก้ WEATHER_CLIMATOLOGY_PATH ใน app.py
รัน: python scripts/export_weather_climatology.py
"""
import importlib.util
import json
import math
import os
import time

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
WEATHER = ["temp", "humidity", "wind_speed", "precipitation", "wind_dir_sin", "wind_dir_cos", "pressure"]


def main():
    spec = importlib.util.spec_from_file_location("bd", os.path.join(_ROOT, "scripts", "build_raw_dataset_multiprovince.py"))
    bd = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bd)
    reg = json.load(open(os.path.join(_ROOT, "data", "province_station_registry.json"), encoding="utf-8"))["provinces"]

    out, n_rows = {}, 0
    for p in reg:
        slug = p["province_en"]
        w = bd.fetch_weather(p["lat"], p["lon"])  # คอลัมน์ wind_dir_deg เป็นองศา
        w = w.dropna(subset=["temp", "humidity", "wind_speed", "precipitation", "wind_dir_deg", "pressure"]).copy()
        rad = w["wind_dir_deg"] * 3.141592653589793 / 180.0
        w["wind_dir_sin"], w["wind_dir_cos"] = rad.map(math.sin), rad.map(math.cos)
        w["month"], w["hour"] = w["datetime_utc"].dt.month, w["datetime_utc"].dt.hour
        g = w.groupby(["month", "hour"])[WEATHER].mean()
        assert len(g) == 12 * 24, f"{slug}: climatology ไม่ครบ 288 ช่อง ({len(g)})"
        out[slug] = [[[round(float(g.loc[(m, h), c]), 4) for c in WEATHER] for h in range(24)] for m in range(1, 13)]
        n_rows += len(w)
        print(f"  {slug}: {len(w)} ชั่วโมง", flush=True)
        time.sleep(bd.REQUEST_SLEEP_S)

    payload = {
        "created": pd.Timestamp.now().isoformat(timespec="seconds"),
        "source": "open-meteo archive (พิกัดสถานี)",
        "window": f"{bd.WINDOW_START:%Y-%m-%d}..{bd.WINDOW_END:%Y-%m-%d}",
        "n_rows": int(n_rows),
        "features": WEATHER,
        "layout": "table[province][month-1][hour_utc] = [feature values in `features` order]",
        "table": out,
    }
    path = os.path.join(_ROOT, "models", f"weather_climatology_{TIMESTAMP}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, separators=(",", ":"))
    print(f"บันทึก {path} ({os.path.getsize(path) / 1024:.0f} KB) จังหวัด {len(out)} แถวต้นทาง {n_rows}")


if __name__ == "__main__":
    main()

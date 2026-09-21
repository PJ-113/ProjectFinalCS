"""
E3 (แผน deploy feasibility) — วัดความพร้อมของข้อมูล OpenAQ "ตอนนี้จริงๆ" สำหรับ feature เพื่อนบ้าน (pm25_neighbor_avg, nb_trend_12h)
ดึง pm25 ย้อนหลัง 72 ชม. ของทั้ง 20 จังหวัด (ตรรกะ fetch เหมือน fetch_recent_pm25_openaq ใน app.py แต่ไม่ import app.py
เพราะ import แล้วจะต่อ DB/โหลดโมเดล — สคริปต์นี้ต้องอ่านอย่างเดียว) แล้ววัดต่อจังหวัด:
  - อายุของ reading ล่าสุด (ชม. เทียบตอนนี้)
  - เพื่อนบ้านที่ใกล้สุด (k=1 จากทั้ง 20 จังหวัด ตามที่ add_spatial_lag_feature ทำ) และความต่างของเวลา reading ล่าสุดระหว่างคู่
  - เพื่อนบ้านมีค่า "จริง" (ไม่ใช่ interpolate) ณ ชั่วโมงล่าสุดของจังหวัดเป้าหมายหรือไม่ / ที่ t-12h หรือไม่
  - ชั่วโมงที่ขาดใน 12 ชม.ล่าสุดของทั้งคู่
ไม่พิมพ์คีย์ (อ่านจาก .env เฉพาะตอนสร้าง header)
"""
import json
import os
import time

import numpy as np
import pandas as pd
import requests

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(_ROOT, "models")
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")


def load_env_key(name):
    with open(os.path.join(_ROOT, ".env"), encoding="utf-8") as f:
        for line in f:
            if line.startswith(f"{name}="):
                return line.strip().split("=", 1)[1]
    raise RuntimeError(f"{name} not found in .env")


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def fetch_recent(sensor_id, headers, hours_back=72):
    end = pd.Timestamp.now(tz="UTC")
    start = end - pd.Timedelta(hours=hours_back)
    params = {"datetime_from": start.isoformat(), "datetime_to": end.isoformat(), "limit": 1000}
    t0 = time.time()
    for attempt in range(5):
        try:
            r = requests.get(f"https://api.openaq.org/v3/sensors/{sensor_id}/measurements", headers=headers,
                             params=params, timeout=15)
        except requests.exceptions.RequestException:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        break
    else:
        return None, time.time() - t0
    res = r.json().get("results", [])
    df = pd.DataFrame([{"datetime_utc": x["period"]["datetimeFrom"]["utc"], "pm25": x["value"]} for x in res])
    if df.empty:
        return df, time.time() - t0
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df = df.dropna(subset=["pm25"]).drop_duplicates(subset=["datetime_utc"]).sort_values("datetime_utc")
    return df.set_index("datetime_utc")["pm25"], time.time() - t0


def main():
    headers = {"X-API-Key": load_env_key("OPENAQ_KEY")}
    with open(os.path.join(_ROOT, "data", "province_station_registry.json"), encoding="utf-8") as f:
        provs = json.load(f)["provinces"]
    now = pd.Timestamp.now(tz="UTC")
    print(f"เวลาตอนนี้ (UTC): {now}")

    series, req_time = {}, {}
    for p in provs:
        s, dt_req = fetch_recent(p["sensor_id_pm25"], headers)
        series[p["province_en"]] = s
        req_time[p["province_en"]] = dt_req
        time.sleep(0.3)

    rows = []
    for p in provs:
        en = p["province_en"]
        s = series[en]
        others = [q for q in provs if q["province_en"] != en]
        near = min(others, key=lambda q: haversine_km(p["lat"], p["lon"], q["lat"], q["lon"]))
        nen = near["province_en"]
        dist = haversine_km(p["lat"], p["lon"], near["lat"], near["lon"])
        row = {"province": en, "nearest": nen, "dist_km": round(dist, 1), "req_sec": round(req_time[en], 2),
               "n_readings_72h": 0 if s is None or len(s) == 0 else len(s)}
        ns = series[nen]
        if s is None or len(s) == 0:
            rows.append({**row, "status": "target_no_data"})
            continue
        t_latest = s.index.max()
        row["latest_age_h"] = round((now - t_latest).total_seconds() / 3600, 1)
        if ns is None or len(ns) == 0:
            rows.append({**row, "status": "neighbor_no_data"})
            continue
        n_latest = ns.index.max()
        row["neighbor_latest_age_h"] = round((now - n_latest).total_seconds() / 3600, 1)
        row["latest_gap_pair_h"] = round(abs((t_latest - n_latest).total_seconds()) / 3600, 1)
        row["neighbor_has_real_at_t"] = bool(t_latest in ns.index)
        row["neighbor_has_real_at_t-12h"] = bool((t_latest - pd.Timedelta(hours=12)) in ns.index)
        grid = pd.date_range(t_latest - pd.Timedelta(hours=12), t_latest, freq="h", tz="UTC")
        row["target_missing_last12h"] = int(len(grid) - s.reindex(grid).notna().sum())
        row["neighbor_missing_last12h"] = int(len(grid) - ns.reindex(grid).notna().sum())
        row["status"] = "ok"
        rows.append(row)

    res = pd.DataFrame(rows)
    out = os.path.join(MODEL_DIR, f"check_live_data_availability_{TIMESTAMP}.csv")
    res.to_csv(out, index=False)
    pd.set_option("display.width", 250)
    print(res.to_string(index=False))

    ok = res[res["status"] == "ok"]
    print("\n" + "=" * 90 + "\nสรุป\n" + "=" * 90)
    print(f"จังหวัดที่มีข้อมูลทั้งคู่: {len(ok)}/{len(res)}  (ไม่มีข้อมูลเลย: {(res['status'] != 'ok').sum()})")
    if len(ok):
        print(f"อายุ reading ล่าสุดของจังหวัด: มัธยฐาน {ok['latest_age_h'].median():.1f} ชม., สูงสุด {ok['latest_age_h'].max():.1f} ชม.")
        print(f"ความต่างเวลา reading ล่าสุดระหว่างจังหวัดกับเพื่อนบ้าน: มัธยฐาน {ok['latest_gap_pair_h'].median():.1f} ชม., สูงสุด {ok['latest_gap_pair_h'].max():.1f} ชม.")
        print(f"เพื่อนบ้านมีค่าจริง ณ เวลาล่าสุดของจังหวัด: {ok['neighbor_has_real_at_t'].mean():.0%}")
        print(f"เพื่อนบ้านมีค่าจริงที่ t-12h: {ok['neighbor_has_real_at_t-12h'].mean():.0%}")
        print(f"ชั่วโมงที่ขาดใน 12 ชม.ล่าสุด (จังหวัด/เพื่อนบ้าน) มัธยฐาน: {ok['target_missing_last12h'].median():.0f} / {ok['neighbor_missing_last12h'].median():.0f}")
        print(f"เวลาต่อ request มัธยฐาน {res['req_sec'].median():.2f} วิ (รวม 20 จังหวัด {res['req_sec'].sum():.1f} วิ)")


if __name__ == "__main__":
    main()

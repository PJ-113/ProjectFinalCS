"""
ตรวจ "ป้ายเวลาเลื่อน" ของ OpenAQ /measurements เทียบกับ /hours (ค้นพบตอนหาสาเหตุที่ 9 จังหวัดดูเหมือนข้อมูลเก่า 25 ชม. — CHANGELOG 144)
สำหรับแต่ละสถานีในทะเบียน (20 จังหวัด) หา shift k (ชม.) ที่ทำให้ค่าจาก /hours(t) ตรงกับ /measurements(t-k) มากที่สุด
  - ช่วงล่าสุด 400 ชม. (สถานะปัจจุบัน) และช่วงย้อนหลังในประวัติข้อมูลเทรน (2024-03, 2025-03, 2026-03) เพื่อดูว่าคงที่ตลอดไหม
ไม่พิมพ์คีย์ อ่านอย่างเดียว
"""
import json
import os
import time

import numpy as np
import pandas as pd
import requests

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
WINDOWS = [("ล่าสุด", None), ("2024-03", ("2024-03-01", "2024-03-20")), ("2025-03", ("2025-03-01", "2025-03-20")),
           ("2026-03", ("2026-03-01", "2026-03-20"))]


def load_key():
    with open(os.path.join(_ROOT, ".env"), encoding="utf-8") as f:
        for line in f:
            if line.startswith("OPENAQ_KEY="):
                return line.strip().split("=", 1)[1]


def fetch(kind, sid, start, end, headers):
    params = {"datetime_from": start.isoformat(), "datetime_to": end.isoformat(), "limit": 1000}
    for a in range(5):
        try:
            r = requests.get(f"https://api.openaq.org/v3/sensors/{sid}/{kind}", headers=headers, params=params, timeout=30)
        except requests.exceptions.RequestException:
            time.sleep(2 ** a)
            continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2 ** a)
            continue
        if r.status_code != 200:
            return pd.Series(dtype=float)
        return pd.Series({pd.Timestamp(x["period"]["datetimeFrom"]["utc"]): x["value"] for x in r.json()["results"]
                          if x["value"] is not None}).sort_index()
    return pd.Series(dtype=float)


def best_shift(m, h):
    """k ที่ทำให้ h(t) == m(t-k) มากที่สุด (k>0 = ป้ายเวลาของ /measurements 'เร็วกว่าความจริง' k ชม. หรือ 'ช้ากว่า' ตามที่รายงาน)"""
    best = (None, 0, np.inf)
    for k in range(-30, 31):
        ms = m.copy()
        ms.index = ms.index + pd.Timedelta(hours=k)
        cm = h.index.intersection(ms.index)
        if len(cm) < 40:
            continue
        d = (h.loc[cm] - ms.loc[cm]).abs().mean()
        if d < best[2]:
            best = (k, len(cm), d)
    return best


def main():
    headers = {"X-API-Key": load_key()}
    provs = json.load(open(os.path.join(_ROOT, "data", "province_station_registry.json"), encoding="utf-8"))["provinces"]
    now = pd.Timestamp.now(tz="UTC")
    rows = []
    for p in provs:
        row = {"province": p["province_en"], "sensor": p["sensor_id_pm25"]}
        for label, win in WINDOWS:
            if win is None:
                start, end = now - pd.Timedelta(hours=400), now
            else:
                start, end = pd.Timestamp(win[0], tz="UTC"), pd.Timestamp(win[1], tz="UTC")
            m = fetch("measurements", p["sensor_id_pm25"], start, end, headers)
            h = fetch("hours", p["sensor_id_pm25"], start, end, headers)
            k, n, d = best_shift(m, h) if len(m) and len(h) else (None, 0, np.nan)
            row[label] = "ไม่มีข้อมูล" if k is None else f"{k:+d}h (n={n}, diff={d:.3f})"
            time.sleep(0.3)
        rows.append(row)
        print(f"  {row['province']:16s} " + " | ".join(f"{lbl}: {row[lbl]}" for lbl, _ in WINDOWS), flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(_ROOT, "models", f"check_openaq_timestamp_shift_{TIMESTAMP}.csv"), index=False)


if __name__ == "__main__":
    main()

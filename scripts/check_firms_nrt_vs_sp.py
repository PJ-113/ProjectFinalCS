"""
เทียบจำนวนจุดความร้อนจาก FIRMS แบบ near-real-time (NRT, ที่ live ใช้) กับแบบ archive (SP, ที่ใช้เทรน)
ดึงทั้งสอง source ในหน้าต่างวันเดียวกัน (bbox เดียวกับตอนเทรน) แล้วเทียบ:
  (ก) แหล่งไหนมีข้อมูลในช่วงไหน
  (ข) ช่วงที่ทั้งสองมีข้อมูล: จำนวนจุด/วัน ต่อจังหวัด (รัศมี 100 กม.) — อัตราส่วน, Spearman/Pearson
ไม่พิมพ์ FIRMS_KEY (ไม่พิมพ์ URL)
"""
import json
import os
import time
from io import StringIO

import numpy as np
import pandas as pd
import requests

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(_ROOT, "models")
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
AREA = "100.5,13.5,106.5,19.0"
SP, NRT = "VIIRS_NOAA20_SP", "VIIRS_NOAA20_NRT"
RADIUS_KM = 100
# หน้าต่างละ 5 วัน — เลือกให้ครอบทั้งช่วงที่ SP ควรมี (<= 2026-06-30) และช่วงใกล้ปัจจุบัน
WINDOWS = ["2026-03-10", "2026-04-05", "2026-05-01", "2026-06-01", "2026-06-20", "2026-07-20", "2026-08-25", "2026-09-10"]


def load_key():
    with open(os.path.join(_ROOT, ".env"), encoding="utf-8") as f:
        for line in f:
            if line.startswith("FIRMS_KEY="):
                return line.strip().split("=", 1)[1]
    raise RuntimeError("FIRMS_KEY not found in .env")


def fetch(key, source, date_str):
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/{source}/{AREA}/5/{date_str}"
    for attempt in range(4):
        try:
            r = requests.get(url, timeout=60)
        except requests.exceptions.RequestException:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 200:
            text = r.text.strip()
            if not text or text.lower().startswith("invalid") or "latitude" not in text.split("\n")[0]:
                return pd.DataFrame()
            return pd.read_csv(StringIO(text))
        if r.status_code in (429, 500, 502, 503):
            time.sleep(2 ** attempt)
            continue
        return pd.DataFrame()  # 4xx: ไม่มีข้อมูล/ไม่รองรับ (ไม่พิมพ์ URL เพราะมีคีย์)
    return pd.DataFrame()


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def daily_counts(df, provs):
    """จำนวนจุดต่อวัน (UTC) ต่อจังหวัด ในรัศมี RADIUS_KM"""
    out = {}
    d = df["acq_date"].values
    for p in provs:
        near = haversine_km(p["lat"], p["lon"], df["latitude"].values, df["longitude"].values) <= RADIUS_KM
        out[p["province_en"]] = pd.Series(d[near]).value_counts()
    return pd.DataFrame(out).fillna(0).astype(int)


def main():
    key = load_key()
    with open(os.path.join(_ROOT, "data", "province_station_registry.json"), encoding="utf-8") as f:
        provs = json.load(f)["provinces"]

    rows, pair_rows = [], []
    for w in WINDOWS:
        sp, nrt = fetch(key, SP, w), fetch(key, NRT, w)
        time.sleep(1.0)
        rows.append({"window_start": w, "sp_rows": len(sp), "nrt_rows": len(nrt),
                     "sp_dates": f"{sp['acq_date'].min()}..{sp['acq_date'].max()}" if len(sp) else "-",
                     "nrt_dates": f"{nrt['acq_date'].min()}..{nrt['acq_date'].max()}" if len(nrt) else "-"})
        print(f"  {w}: SP={len(sp)} จุด, NRT={len(nrt)} จุด", flush=True)
        if len(sp) and len(nrt):
            c_sp, c_nrt = daily_counts(sp, provs), daily_counts(nrt, provs)
            days = sorted(set(c_sp.index) & set(c_nrt.index))
            for prov in c_sp.columns:
                for day in days:
                    pair_rows.append({"window": w, "province": prov, "date": day,
                                      "sp": int(c_sp.loc[day, prov]), "nrt": int(c_nrt.loc[day, prov])})

    avail = pd.DataFrame(rows)
    avail.to_csv(os.path.join(MODEL_DIR, f"check_firms_nrt_vs_sp_availability_{TIMESTAMP}.csv"), index=False)
    pd.set_option("display.width", 200)
    print("\nแหล่งข้อมูลที่มีในแต่ละหน้าต่าง:\n" + avail.to_string(index=False))

    if not pair_rows:
        print("\nไม่มีหน้าต่างที่ทั้ง SP และ NRT มีข้อมูลพร้อมกัน -> เทียบตรงไม่ได้ (ใช้ผลความไวจาก E4 แทน)")
        return
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(os.path.join(MODEL_DIR, f"check_firms_nrt_vs_sp_daily_{TIMESTAMP}.csv"), index=False)
    tot = pairs.groupby("province")[["sp", "nrt"]].sum()
    tot["ratio_nrt_over_sp"] = tot["nrt"] / tot["sp"].replace(0, np.nan)
    print("\nรวมทุกวันที่ทับกัน ต่อจังหวัด (รัศมี 100 กม.):\n" + tot.round(3).to_string())
    active = pairs[(pairs["sp"] + pairs["nrt"]) > 0]
    print(f"\nจังหวัด-วันที่เทียบได้: {len(pairs)} (มีจุดไฟ {len(active)})")
    print(f"อัตราส่วนรวม NRT/SP = {pairs['nrt'].sum() / max(pairs['sp'].sum(), 1):.3f}")
    print(f"Spearman (จังหวัด-วัน) = {pairs[['sp', 'nrt']].corr(method='spearman').iloc[0, 1]:.3f}")
    print(f"Pearson (จังหวัด-วัน) = {pairs[['sp', 'nrt']].corr().iloc[0, 1]:.3f}")
    print(f"มัธยฐานอัตราส่วนต่อจังหวัด = {tot['ratio_nrt_over_sp'].median():.3f}")


if __name__ == "__main__":
    main()

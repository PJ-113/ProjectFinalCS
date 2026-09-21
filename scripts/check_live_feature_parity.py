"""
E2 (แผน deploy feasibility) — ตรวจ "train/serve skew" ของ 3 feature ใหม่: สูตรแบบ live (ใช้เฉพาะข้อมูลที่มีถึงเวลา t เหมือน
build_feature_row ใน app.py: ดึงย้อนหลัง 72 ชม., reindex กริด 48 ชม., interpolate linear limit=48) ให้ค่าตรงกับแถวใน dataset เทรนไหม
  - pm25_neighbor_avg : pm25 ของจังหวัดที่ใกล้สุด (k=1 จากทั้ง 20 จังหวัด) ณ เวลา t
  - nb_trend_12h      : neighbor(t) - neighbor(t-12h)
  - fire_hotspot_count_100km_24h : จำนวนจุดใน (t-24h, t] รัศมี 100 กม. (ใช้ไฟล์ SP เดิม => ทดสอบเฉพาะ "สูตร" ไม่ใช่ตัวแหล่ง; ต่างแหล่งดู E1/E4)
จุดที่ต่างกันโดยธรรมชาติ: dataset เทรน interpolate ช่องว่างโดยเห็น "อนาคต" ของช่วงนั้นด้วย แต่ live เห็นแค่อดีต (ค้างค่าท้ายสาย)
=> วัดว่าต่างกันบ่อยแค่ไหน/มากแค่ไหน (โดยเฉพาะเมื่อเพื่อนบ้านไม่มี reading จริง ณ เวลา t)
ดึง OpenAQ รอบเดียวต่อจังหวัดครอบช่วงทดสอบ แล้ว slice ทีละ t แบบ 72 ชม. (ไม่ import app.py; ไม่พิมพ์คีย์)
"""
import glob
import json
import os
import time

import numpy as np
import pandas as pd
import requests

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_ROOT, "data")
MODEL_DIR = os.path.join(_ROOT, "models")
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
WIN_START, WIN_END = pd.Timestamp("2026-06-10", tz="UTC"), pd.Timestamp("2026-07-01", tz="UTC")
N_PER_PROVINCE = 25
RADIUS_KM = 100
ATOL = 1e-3
SEED = 11


def load_key(name):
    with open(os.path.join(_ROOT, ".env"), encoding="utf-8") as f:
        for line in f:
            if line.startswith(f"{name}="):
                return line.strip().split("=", 1)[1]
    raise RuntimeError(f"{name} not found in .env")


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def fetch_range(sensor_id, headers, start, end):
    params = {"datetime_from": start.isoformat(), "datetime_to": end.isoformat(), "limit": 1000}
    for attempt in range(5):
        try:
            r = requests.get(f"https://api.openaq.org/v3/sensors/{sensor_id}/measurements", headers=headers,
                             params=params, timeout=30)
        except requests.exceptions.RequestException:
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        res = r.json().get("results", [])
        if not res:
            return pd.Series(dtype=float)
        df = pd.DataFrame([{"t": x["period"]["datetimeFrom"]["utc"], "pm25": x["value"]} for x in res])
        df["t"] = pd.to_datetime(df["t"], utc=True)
        return df.dropna(subset=["pm25"]).drop_duplicates("t").sort_values("t").set_index("t")["pm25"]
    raise RuntimeError(f"fetch failed sensor {sensor_id}")


def live_value(series, t, hours_before=(0, 12)):
    """เลียนแบบ build_feature_row: ใช้ข้อมูล <= t ย้อนหลัง 72 ชม., กริด 48 ชม., interpolate limit=48 both -> คืนค่า ณ t และ t-12h
    (คืน NaN ถ้าซีรีส์ว่าง) และบอกด้วยว่ามี reading จริง ณ t หรือไม่"""
    hist = series[(series.index <= t) & (series.index >= t - pd.Timedelta(hours=72))]
    if hist.empty:
        return [np.nan] * len(hours_before), False
    grid = pd.date_range(t - pd.Timedelta(hours=48), t, freq="h", tz="UTC")
    filled = hist.reindex(grid).interpolate(method="linear", limit=48, limit_direction="both")
    return [filled.get(t - pd.Timedelta(hours=h), np.nan) for h in hours_before], bool(t in hist.index)


def main():
    headers = {"X-API-Key": load_key("OPENAQ_KEY")}
    with open(os.path.join(DATA_DIR, "province_station_registry.json"), encoding="utf-8") as f:
        provs = {p["province_en"]: p for p in json.load(f)["provinces"]}
    ds = pd.read_csv(os.path.join(DATA_DIR, "multiprovince_pm25_features_raw_spatial_k1_firehotspot_nbtrend_20260919_135009.csv"),
                     usecols=["province_en", "is_zero_shot", "datetime_utc", "pm25", "pm25_neighbor_avg", "nb_trend_12h",
                              "fire_hotspot_count_100km_24h"])
    ds["datetime_utc"] = pd.to_datetime(ds["datetime_utc"], utc=True)
    hot = pd.read_csv(sorted(glob.glob(os.path.join(DATA_DIR, "firms_hotspots_isaan_*.csv")))[-1])
    hot["t"] = pd.to_datetime(hot["acq_datetime_utc"], utc=True)
    hot = hot.sort_values("t")

    print("ดึง OpenAQ ช่วง 2026-06-07..2026-07-01 ของทั้ง 20 จังหวัด (1 request ต่อจังหวัด)...")
    series = {}
    for en, p in provs.items():
        series[en] = fetch_range(p["sensor_id_pm25"], headers, WIN_START - pd.Timedelta(days=3), WIN_END + pd.Timedelta(days=1))
        time.sleep(0.4)
    print("  จำนวน reading: " + ", ".join(f"{k}={len(v)}" for k, v in series.items()))

    # เพื่อนบ้านที่ใกล้สุด (k=1) จากทั้ง 20 จังหวัด ตามที่ add_spatial_lag_feature ทำ
    nearest = {}
    for en, p in provs.items():
        others = [q for e2, q in provs.items() if e2 != en]
        nearest[en] = min(others, key=lambda q: haversine_km(p["lat"], p["lon"], q["lat"], q["lon"]))["province_en"]

    rng = np.random.default_rng(SEED)
    rows = []
    for en in provs:
        if provs[en].get("is_zero_shot"):
            continue
        tgt, nb_en = series[en], nearest[en]
        cand = ds[(ds["province_en"] == en) & (ds["datetime_utc"] >= WIN_START) & (ds["datetime_utc"] <= WIN_END)]
        cand = cand[cand["datetime_utc"].isin(tgt.index)]  # ต้องมี reading จริงของเป้าหมาย ณ t (ไม่งั้น live จะมี latest_time เป็นชั่วโมงก่อนหน้า)
        if len(cand) == 0:
            continue
        pick = cand.sample(min(N_PER_PROVINCE, len(cand)), random_state=int(rng.integers(1e9)))
        near_mask_cache = None
        for _, r in pick.iterrows():
            t = r["datetime_utc"]
            (nb_t, nb_12), nb_real_t = live_value(series[nb_en], t)
            trend = nb_t - nb_12 if not (np.isnan(nb_t) or np.isnan(nb_12)) else np.nan
            p = provs[en]
            if near_mask_cache is None:
                near_mask_cache = haversine_km(p["lat"], p["lon"], hot["latitude"].values, hot["longitude"].values) <= RADIUS_KM
            ht = hot["t"][near_mask_cache]
            fire = int(((ht > t - pd.Timedelta(hours=24)) & (ht <= t)).sum())
            rows.append({"province": en, "neighbor": nb_en, "t": t, "nb_real_at_t": nb_real_t,
                         "nb_ds": r["pm25_neighbor_avg"], "nb_live": nb_t,
                         "trend_ds": r["nb_trend_12h"], "trend_live": trend,
                         "fire_ds": r["fire_hotspot_count_100km_24h"], "fire_live": fire})

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(MODEL_DIR, f"check_live_feature_parity_{TIMESTAMP}.csv"), index=False)
    print(f"\nจุดเปรียบเทียบทั้งหมด: {len(res)} (จาก {res['province'].nunique()} จังหวัด)")

    def summarize(name, a, b, mask=None):
        m = res.index if mask is None else res.index[mask]
        x, y = res.loc[m, a].astype(float), res.loc[m, b].astype(float)
        both = x.notna() & y.notna()
        eq = (np.abs(x - y) <= ATOL) & both
        print(f"  {name:38s} n={len(m):4d}  ตรง(|diff|<={ATOL}): {eq.sum() / max(len(m), 1):6.1%}  "
              f"live เป็น NaN: {y.isna().sum():3d}  |diff| เฉลี่ย(ที่เทียบได้): {np.abs(x - y)[both].mean():.3f}  "
              f"มัธยฐาน: {np.abs(x - y)[both].median():.3f}  สูงสุด: {np.abs(x - y)[both].max():.3f}")

    print("\n=== ภาพรวมทุกจุด ===")
    summarize("pm25_neighbor_avg", "nb_ds", "nb_live")
    summarize("nb_trend_12h", "trend_ds", "trend_live")
    summarize("fire_hotspot_count_100km_24h", "fire_ds", "fire_live")

    print("\n=== แยกตาม: เพื่อนบ้านมี reading จริง ณ เวลา t หรือไม่ ===")
    for label, mask in (("มี reading จริง ณ t", res["nb_real_at_t"]), ("ไม่มี (ต้อง interpolate/ค้างค่า)", ~res["nb_real_at_t"])):
        print(f" [{label}] n={mask.sum()}")
        if mask.sum():
            summarize("  pm25_neighbor_avg", "nb_ds", "nb_live", mask.values)
            summarize("  nb_trend_12h", "trend_ds", "trend_live", mask.values)


if __name__ == "__main__":
    main()

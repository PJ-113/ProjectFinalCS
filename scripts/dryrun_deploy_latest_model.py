"""
E5 (แผน deploy feasibility) — dry-run pipeline แบบ get_forecast() ด้วยโมเดล _20260919_135655 "โดยไม่แตะ app.py/DB/NLG"
ดึง OpenAQ ล่าสุดของทั้ง 20 จังหวัด "รอบเดียว" แล้วใช้ร่วมกัน (จังหวัดเป้าหมาย + เพื่อนบ้าน: เท่ากับจำนวน request ที่ scheduled_forecast_job
ใช้อยู่แล้ว ไม่เพิ่มภาระ OpenAQ), Open-Meteo current ต่อจังหวัด, FIRMS NRT 1 request (ครอบ bbox ทั้งอีสาน) ใช้ร่วมทุกจังหวัด
นโยบาย feature เพื่อนบ้าน (เลือกจาก E4):
  - เพื่อนบ้านอันดับ 1 "สด" (มี reading จริงภายใน FRESH_H ชม. ของเวลา reading ล่าสุดของจังหวัดเป้าหมาย) -> ใช้ตามสูตรเทรนเป๊ะ
  - ไม่สด -> เฉลี่ยเพื่อนบ้านสดที่ใกล้สุดสูงสุด 3 อันดับ, nb_trend_12h คำนวณจากค่าเฉลี่ยชุดเดียวกัน (E4: เสียหาย ~+2.5%)
  - ไม่มีเพื่อนบ้านสดเลย -> flag naive_fallback (ห้ามส่ง NaN เข้าโมเดล: E4 พบ +32%/+65% แย่กว่า persistence)
ตรวจ: feature ครบ/ไม่ NaN, อยู่ในช่วง 1-99 percentile ของ dataset เทรน, ผลพยากรณ์สมเหตุสมผล, เทียบกับโมเดลที่ app.py ใช้ตอนนี้
(_20260913_014901) ผ่าน pkl ตรงๆ, เวลาต่อจังหวัด, จำนวน API call — ไม่พิมพ์คีย์
"""
import importlib.util
import json
import os
import pickle
import time
from io import StringIO

import numpy as np
import pandas as pd
import requests

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR, MODEL_DIR = os.path.join(_ROOT, "data"), os.path.join(_ROOT, "models")
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
NEW_TAG, OLD_TAG = "20260919_135655", "20260913_014901"
FRESH_H = 3
TEST_PROVINCES = ["khonkaen", "nakhonphanom", "buriram", "udonthani", "sakonnakhon", "surin"]
AREA = "100.5,13.5,106.5,19.0"
API_CALLS = {"openaq": 0, "openmeteo": 0, "firms": 0}


def load_key(name):
    with open(os.path.join(_ROOT, ".env"), encoding="utf-8") as f:
        for line in f:
            if line.startswith(f"{name}="):
                return line.strip().split("=", 1)[1]
    raise RuntimeError(f"{name} not found in .env")


def load_tm():
    spec = importlib.util.spec_from_file_location("tm", os.path.join(_ROOT, "scripts", "train_multiprovince_models.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def fetch_openaq(sensor_id, headers, hours_back=72):
    end = pd.Timestamp.now(tz="UTC")
    params = {"datetime_from": (end - pd.Timedelta(hours=hours_back)).isoformat(), "datetime_to": end.isoformat(), "limit": 1000}
    for attempt in range(5):
        API_CALLS["openaq"] += 1
        try:
            r = requests.get(f"https://api.openaq.org/v3/sensors/{sensor_id}/measurements", headers=headers, params=params, timeout=15)
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
    raise RuntimeError("OpenAQ fetch failed")


def fetch_weather(lat, lon):
    API_CALLS["openmeteo"] += 1
    r = requests.get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": lat, "longitude": lon, "timezone": "UTC",
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,wind_direction_10m,surface_pressure"}, timeout=15)
    r.raise_for_status()
    return r.json()["current"]


def fetch_firms_nrt(key):
    API_CALLS["firms"] += 1
    r = requests.get(f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{key}/VIIRS_NOAA20_NRT/{AREA}/2", timeout=60)
    if r.status_code != 200 or "latitude" not in r.text.split("\n")[0]:
        return pd.DataFrame(columns=["latitude", "longitude", "t"])
    df = pd.read_csv(StringIO(r.text))
    df["t"] = pd.to_datetime(df["acq_date"] + " " + df["acq_time"].astype(str).str.zfill(4), format="%Y-%m-%d %H%M", utc=True)
    return df


def filled_grid(series, t):
    hist = series[(series.index <= t) & (series.index >= t - pd.Timedelta(hours=72))]
    grid = pd.date_range(t - pd.Timedelta(hours=48), t, freq="h", tz="UTC")
    return hist.reindex(grid).interpolate(method="linear", limit=48, limit_direction="both"), hist


def build_features(prov, provs, series, weather, hotspots, feats_needed):
    """สร้าง feature ตามสูตรของ app.build_feature_row (target) + 3 feature ใหม่ด้วยนโยบายเพื่อนบ้านข้างต้น"""
    s = series[prov]
    if s is None or len(s) == 0:
        return None, {"status": "no_target_data"}
    t = s.index.max()
    grid, _ = filled_grid(s, t)
    lag = lambda h: grid.get(t - pd.Timedelta(hours=h), np.nan)
    w6, w24 = grid.loc[t - pd.Timedelta(hours=5):t], grid.loc[t - pd.Timedelta(hours=23):t]
    wd = weather["wind_direction_10m"]
    f = {"temp": weather["temperature_2m"], "humidity": weather["relative_humidity_2m"], "wind_speed": weather["wind_speed_10m"],
         "precipitation": weather["precipitation"], "hour_sin": np.sin(2 * np.pi * t.hour / 24), "hour_cos": np.cos(2 * np.pi * t.hour / 24),
         "month_sin": np.sin(2 * np.pi * t.month / 12), "month_cos": np.cos(2 * np.pi * t.month / 12), "dayofweek": t.dayofweek,
         "pm25_lag_1": lag(1), "pm25_lag_3": lag(3), "pm25_lag_6": lag(6), "pm25_lag_12": lag(12), "pm25_lag_24": lag(24),
         "pm25_roll_mean_6": w6.mean(), "pm25_roll_std_6": w6.std(), "pm25_roll_mean_24": w24.mean(), "pm25_roll_std_24": w24.std(),
         "wind_dir_sin": np.sin(2 * np.pi * wd / 360), "wind_dir_cos": np.cos(2 * np.pi * wd / 360),
         "pressure": weather["surface_pressure"], "lat": provs[prov]["lat"], "lon": provs[prov]["lon"]}

    ranked = sorted([q for q in provs if q != prov], key=lambda q: haversine_km(provs[prov]["lat"], provs[prov]["lon"], provs[q]["lat"], provs[q]["lon"]))
    fresh = []
    for q in ranked:
        sq = series[q]
        if sq is not None and len(sq) and (t - sq.index.max()) <= pd.Timedelta(hours=FRESH_H) and sq.index.max() <= t + pd.Timedelta(hours=FRESH_H):
            fresh.append(q)
    info = {"status": "ok", "t": t, "target_age_h": round((pd.Timestamp.now(tz="UTC") - t).total_seconds() / 3600, 1),
            "nearest": ranked[0], "nb_policy": None, "nb_used": None}
    if not fresh:
        info["status"] = "naive_fallback(no fresh neighbor)"
        return None, info
    if ranked[0] == fresh[0]:
        used = [ranked[0]]
        info["nb_policy"] = "k=1 exact"
    else:
        used = fresh[:3]
        info["nb_policy"] = f"avg of {len(used)} nearest fresh"
    info["nb_used"] = ",".join(used)
    vals, vals12 = [], []
    for q in used:
        g, _ = filled_grid(series[q], t)
        vals.append(g.get(t, np.nan))
        vals12.append(g.get(t - pd.Timedelta(hours=12), np.nan))
    f["pm25_neighbor_avg"] = float(np.nanmean(vals))
    f["nb_trend_12h"] = float(np.nanmean(vals) - np.nanmean(vals12))
    near = haversine_km(provs[prov]["lat"], provs[prov]["lon"], hotspots["latitude"].values, hotspots["longitude"].values) <= 100
    ht = hotspots["t"][near]
    f["fire_hotspot_count_100km_24h"] = int(((ht > t - pd.Timedelta(hours=24)) & (ht <= t)).sum())
    return pd.DataFrame([f])[feats_needed], info | {"latest_pm25": float(s.loc[t])}


def predict_all(pkg, X, latest, qhat=None):
    out = {}
    for h in range(1, 25):
        mq = pkg["models"][h]
        p50 = float(mq[0.5].predict(X)[0]) + latest
        row = {"p50": p50}
        if qhat is not None:
            row["lo"] = max(0.0, float(mq[0.05].predict(X)[0]) + latest - qhat[h])
            row["hi"] = float(mq[0.95].predict(X)[0]) + latest + qhat[h]
        out[h] = row
    return out


def main():
    hdr = {"X-API-Key": load_key("OPENAQ_KEY")}
    firms_key = load_key("FIRMS_KEY")
    provs = {p["province_en"]: p for p in json.load(open(os.path.join(DATA_DIR, "province_station_registry.json"), encoding="utf-8"))["provinces"]}
    tm = load_tm()
    new = {k: pickle.load(open(os.path.join(MODEL_DIR, f"{k}_pm25_multiprovince_residual_bagged_quantile_{NEW_TAG}.pkl"), "rb")) for k in ("lgb", "xgb")}
    old = pickle.load(open(os.path.join(MODEL_DIR, f"lgb_pm25_multiprovince_residual_bagged_quantile_{OLD_TAG}.pkl"), "rb"))
    qhat = {k: dict(pd.read_csv(os.path.join(MODEL_DIR, f"conformal_calibration_{k}_{NEW_TAG}.csv"))[["horizon", "q_hat"]].astype(float).values) for k in ("lgb", "xgb")}
    qhat = {k: {int(h): v for h, v in d.items()} for k, d in qhat.items()}
    assert new["lgb"]["feature_cols"] == tm.FEATURE_COLS

    ds = pd.read_csv(tm.DATA_PATH, usecols=tm.FEATURE_COLS + ["is_zero_shot"])
    ds = ds[~ds["is_zero_shot"]]
    lo, hi = ds[tm.FEATURE_COLS].quantile(0.01), ds[tm.FEATURE_COLS].quantile(0.99)

    t0 = time.time()
    series = {en: fetch_openaq(p["sensor_id_pm25"], hdr) for en, p in provs.items()}
    t_openaq = time.time() - t0
    t0 = time.time()
    hotspots = fetch_firms_nrt(firms_key)
    t_firms = time.time() - t0
    print(f"OpenAQ 20 จังหวัด: {t_openaq:.1f} วิ | FIRMS NRT 1 request: {t_firms:.1f} วิ ({len(hotspots)} จุดใน 2 วันล่าสุด)")

    pd.set_option("display.width", 250)
    rows = []
    for prov in TEST_PROVINCES:
        t0 = time.time()
        w = fetch_weather(provs[prov]["lat"], provs[prov]["lon"])
        X, info = build_features(prov, provs, series, w, hotspots, tm.FEATURE_COLS)
        print(f"\n=== {prov} === {info}")
        if X is None:
            rows.append({"province": prov, **{k: v for k, v in info.items() if k in ("status",)}})
            continue
        latest = info["latest_pm25"]
        nan_cols = X.columns[X.isna().any()].tolist()
        oor = [c for c in tm.FEATURE_COLS if not (lo[c] <= X[c].iloc[0] <= hi[c])]
        pl, px = predict_all(new["lgb"], X, latest, qhat["lgb"]), predict_all(new["xgb"], X, latest, qhat["xgb"])
        X_old = X.reindex(columns=old["feature_cols"])
        po = predict_all(old, X_old, latest)
        print(f"  ค่าปัจจุบัน pm25={latest:.1f}  NaN feature: {nan_cols or 'ไม่มี'}  นอกช่วง p1-p99 เทรน: {oor or 'ไม่มี'}")
        print(f"  neighbor_avg={X['pm25_neighbor_avg'].iloc[0]:.1f} trend12h={X['nb_trend_12h'].iloc[0]:+.1f} fire24h={X['fire_hotspot_count_100km_24h'].iloc[0]}")
        for h in (1, 6, 12, 24):
            print(f"  t+{h:2d}: ใหม่ LGB {pl[h]['p50']:.1f} [{pl[h]['lo']:.1f}, {pl[h]['hi']:.1f}] | ใหม่ XGB {px[h]['p50']:.1f} | โมเดลเดิม(app) {po[h]['p50']:.1f}")
        sane = all(0 <= pl[h]["p50"] < 500 and pl[h]["lo"] <= pl[h]["hi"] for h in pl)
        rows.append({"province": prov, "status": info["status"], "nb_policy": info["nb_policy"], "nb_used": info["nb_used"],
                     "target_age_h": info["target_age_h"], "latest_pm25": latest, "nan_features": ",".join(nan_cols),
                     "features_out_of_range": ",".join(oor), "sane_predictions": sane,
                     "lgb_t24": pl[24]["p50"], "xgb_t24": px[24]["p50"], "old_t24": po[24]["p50"], "sec": round(time.time() - t0, 2)})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(MODEL_DIR, f"dryrun_deploy_latest_model_{TIMESTAMP}.csv"), index=False)
    print("\n" + "=" * 100 + "\nสรุป dry-run\n" + "=" * 100)
    print(res.to_string(index=False))
    print(f"\nAPI call รวมรอบนี้: {API_CALLS}  (OpenAQ ต่อ 20 จังหวัดเท่ากับที่ scheduled_forecast_job ใช้อยู่ = ไม่เพิ่ม; เพิ่มเฉพาะ FIRMS 1 request/ชม.)")


if __name__ == "__main__":
    main()

"""
Fit a linear calibration mapping Air4Thai's live pm25 readings onto the OpenAQ scale
that all 3 models are trained on.

Root cause (confirmed earlier this investigation): app.py's fetch_recent_pm25() pulls
LIVE inference data from Air4Thai (station 46t), but every model is trained on OpenAQ
(location 7747 / sensor 1304249) data. Same physical station (identical coordinates),
but the two sources disagree substantially (correlation ~0.54, mean abs diff ~3-5 ug/m3),
which explains why live MAE is much worse than backtest MAE.

Fix: calibrate Air4Thai readings onto the OpenAQ scale at inference time, so the model
sees inputs that resemble what it was trained on.
"""
import pickle
import sys
from datetime import timedelta

sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
import requests
import urllib3
from sklearn.linear_model import LinearRegression

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

STATION_ID_AIR4THAI = "46t"
OPENAQ_CACHE_PATH = "scripts/_openaq_raw_5y_cache.csv"
OUT_PATH = "models/air4thai_openaq_calibration.pkl"

# Air4Thai's getHistoryData.php has a hard ~2-month retention limit (confirmed via direct
# testing earlier this session). Fetch the longest window it will actually serve.
FETCH_DAYS_BACK = 75


def fetch_air4thai_history():
    edate = pd.Timestamp.now()
    sdate = edate - timedelta(days=FETCH_DAYS_BACK)

    url = "https://air4thai.com/forweb/getHistoryData.php"
    params = {
        "stationID": STATION_ID_AIR4THAI,
        "param": "PM25",
        "type": "hr",
        "sdate": sdate.strftime("%Y-%m-%d"),
        "edate": edate.strftime("%Y-%m-%d"),
        "stime": "00",
        "etime": "23",
    }
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    r = requests.get(url, params=params, headers=headers, timeout=30, verify=False)
    r.raise_for_status()
    records = r.json()["stations"][0]["data"]

    df = pd.DataFrame(records).rename(columns={"DATETIMEDATA": "datetime_utc", "PM25": "air4thai_pm25"})
    df["datetime_utc"] = (
        pd.to_datetime(df["datetime_utc"])
        .dt.tz_localize("Asia/Bangkok")
        .dt.tz_convert("UTC")
    )
    df["air4thai_pm25"] = pd.to_numeric(df["air4thai_pm25"], errors="coerce")
    df = df.dropna(subset=["air4thai_pm25"]).sort_values("datetime_utc").reset_index(drop=True)
    return df


def main():
    print(f"Fetching Air4Thai history (station {STATION_ID_AIR4THAI}, last {FETCH_DAYS_BACK} days)...")
    air4thai_df = fetch_air4thai_history()
    print(f"  Air4Thai rows: {len(air4thai_df)}  range: {air4thai_df['datetime_utc'].min()} -> {air4thai_df['datetime_utc'].max()}")

    openaq_df = pd.read_csv(OPENAQ_CACHE_PATH)
    openaq_df["datetime_utc"] = pd.to_datetime(openaq_df["datetime_utc"], utc=True)
    openaq_df = openaq_df.rename(columns={"pm25": "openaq_pm25"})[["datetime_utc", "openaq_pm25"]]
    print(f"  OpenAQ cache rows: {len(openaq_df)}  range: {openaq_df['datetime_utc'].min()} -> {openaq_df['datetime_utc'].max()}")

    merged = pd.merge(air4thai_df, openaq_df, on="datetime_utc", how="inner").dropna(
        subset=["air4thai_pm25", "openaq_pm25"]
    )
    print(f"  Overlap (exact-hour match, both non-null): n={len(merged)}")
    print(f"  Overlap range: {merged['datetime_utc'].min()} -> {merged['datetime_utc'].max()}")

    X = merged[["air4thai_pm25"]].values
    y = merged["openaq_pm25"].values

    reg = LinearRegression().fit(X, y)
    a = float(reg.coef_[0])
    b = float(reg.intercept_)
    r2 = float(reg.score(X, y))
    n = len(merged)

    print("\n=== Calibration fit: openaq_equiv = a * air4thai + b ===")
    print(f"  a (slope)     = {a:.6f}")
    print(f"  b (intercept) = {b:.6f}")
    print(f"  R^2           = {r2:.4f}")
    print(f"  n             = {n}")

    calibration = {
        "a": a,
        "b": b,
        "r2": r2,
        "n": n,
        "fitted_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "source_a": "air4thai_station_46t",
        "source_b": "openaq_sensor_1304249",
        "overlap_start": merged["datetime_utc"].min().isoformat(),
        "overlap_end": merged["datetime_utc"].max().isoformat(),
    }
    with open(OUT_PATH, "wb") as f:
        pickle.dump(calibration, f)
    print(f"\nSaved calibration to {OUT_PATH}")

    # example before/after
    sample = merged.sample(min(5, len(merged)), random_state=42).sort_values("datetime_utc")
    print("\n=== ตัวอย่าง before/after calibration ===")
    for _, row in sample.iterrows():
        raw = row["air4thai_pm25"]
        calibrated = a * raw + b
        print(f"  {row['datetime_utc']}  air4thai_raw={raw:.1f}  ->  calibrated={calibrated:.1f}  (openaq_actual={row['openaq_pm25']:.1f})")


if __name__ == "__main__":
    main()

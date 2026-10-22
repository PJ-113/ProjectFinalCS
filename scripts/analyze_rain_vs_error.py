"""
วัดความสัมพันธ์ระหว่างฝนตก (precipitation จาก Open-Meteo) กับความคลาดเคลื่อนของโมเดล
(abs_error จาก prediction_log) ต่อจังหวัด ด้วย correlation และเทียบ mean abs_error
ระหว่างช่วงฝนตก/ไม่ตก
"""
import os
import sys
import numpy as np
import pandas as pd
import requests

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

import importlib.util
spec = importlib.util.spec_from_file_location("app_under_test", os.path.join(_ROOT, "app.py"))
app = importlib.util.module_from_spec(spec)
spec.loader.exec_module(app)


def load_prediction_errors():
    conn = app.get_db_connection()
    df = pd.read_sql(
        """SELECT province, horizon, target_time_utc, predicted_pm25, actual_pm25, model_used
           FROM prediction_log
           WHERE actual_pm25 IS NOT NULL AND model_used != 'naive_fallback'""",
        conn,
    )
    conn.close()
    df["target_time_utc"] = pd.to_datetime(df["target_time_utc"], utc=True)
    df["abs_error"] = (df["predicted_pm25"] - df["actual_pm25"]).abs()
    df["hour_key"] = df["target_time_utc"].dt.floor("h")
    return df


def fetch_precip_for_province(lat, lon, start_date, end_date):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "hourly": "precipitation",
        "start_date": start_date, "end_date": end_date,
        "timezone": "UTC",
    }
    r = requests.get(url, params=params, timeout=20)
    r.raise_for_status()
    h = r.json()["hourly"]
    s = pd.Series(h["precipitation"], index=pd.to_datetime(h["time"], utc=True))
    return s


def main():
    df = load_prediction_errors()
    print(f"prediction_log rows with actual (excl. naive_fallback): {len(df)}")
    if df.empty:
        print("ไม่มีข้อมูลพอวิเคราะห์")
        return

    start_date = df["hour_key"].min().strftime("%Y-%m-%d")
    end_date = df["hour_key"].max().strftime("%Y-%m-%d")
    print(f"date range: {start_date} to {end_date}")

    provinces = sorted(df["province"].unique())
    print(f"provinces: {provinces}")

    precip_frames = []
    for slug in provinces:
        meta = app.PROVINCES_BY_SLUG.get(slug)
        if meta is None:
            print(f"  [skip] {slug}: not in registry")
            continue
        try:
            s = fetch_precip_for_province(meta["lat"], meta["lon"], start_date, end_date)
        except Exception as e:
            print(f"  [skip] {slug}: fetch failed ({e})")
            continue
        frame = s.rename("precipitation").reset_index().rename(columns={"index": "hour_key"})
        frame["province"] = slug
        precip_frames.append(frame)
        print(f"  {slug}: fetched {len(frame)} hourly precip rows")

    precip_df = pd.concat(precip_frames, ignore_index=True)

    merged = df.merge(precip_df, on=["province", "hour_key"], how="inner")
    print(f"\nmatched rows (error + precipitation): {len(merged)} / {len(df)}")

    if merged.empty:
        print("join ไม่ติดเลย (เวลาไม่ตรง?) ตรวจสอบ hour_key ทั้งสองฝั่ง")
        return

    corr = merged["precipitation"].corr(merged["abs_error"])
    print(f"\nPearson correlation(precipitation, abs_error) = {corr:.4f}")

    merged["is_raining"] = merged["precipitation"] > 0.1
    grp = merged.groupby("is_raining")["abs_error"].agg(["mean", "median", "count"])
    print("\nabs_error by rain/no-rain (threshold 0.1mm):")
    print(grp)

    bins = [-0.01, 0, 1, 5, 100]
    labels = ["0mm", "0-1mm", "1-5mm", ">5mm"]
    merged["precip_bin"] = pd.cut(merged["precipitation"], bins=bins, labels=labels)
    grp2 = merged.groupby("precip_bin", observed=True)["abs_error"].agg(["mean", "median", "count"])
    print("\nabs_error by precipitation bin:")
    print(grp2)

    by_h = merged.groupby("is_raining")["abs_error"].mean()
    if True in by_h.index and False in by_h.index and by_h[False] > 0:
        pct_increase = (by_h[True] - by_h[False]) / by_h[False] * 100
        print(f"\nขณะฝนตก abs_error เฉลี่ยสูงกว่าขณะไม่ตกฝน {pct_increase:.1f}%")


if __name__ == "__main__":
    main()

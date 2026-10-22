"""
วัด Pearson correlation ดิบระหว่างความชื้น (humidity) กับ PM2.5 รายจังหวัด แยกหน้าหนาว
(ธ.ค.-ก.พ.) กับนอกหน้าหนาว และแบบ pooled ทุกจังหวัด
"""
import os

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(_ROOT, "data", "multiprovince_pm25_features_raw_spatial_k1_firehotspot_nbtrend_20260919_174805.csv")
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")


def main():
    df = pd.read_csv(DATA_PATH, usecols=["province_en", "datetime_utc", "pm25", "humidity"])
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    df = df.dropna(subset=["pm25", "humidity"])
    month = df["datetime_utc"].dt.tz_convert("Asia/Bangkok").dt.month if df["datetime_utc"].dt.tz is not None \
        else (df["datetime_utc"] + pd.Timedelta(hours=7)).dt.month
    df["winter"] = month.isin([12, 1, 2])
    rows = []
    for prov, g in df.groupby("province_en"):
        rows.append({
            "province_en": prov, "n": len(g),
            "r_all": g["pm25"].corr(g["humidity"]),
            "r_winter": g.loc[g["winter"], "pm25"].corr(g.loc[g["winter"], "humidity"]),
            "r_non_winter": g.loc[~g["winter"], "pm25"].corr(g.loc[~g["winter"], "humidity"]),
        })
    out = pd.DataFrame(rows)
    print(out.round(3).to_string(index=False))
    for c in ["r_all", "r_winter", "r_non_winter"]:
        print(f"{c}: min={out[c].min():.3f}  median={out[c].median():.3f}  max={out[c].max():.3f}")
    pooled = {"all": df["pm25"].corr(df["humidity"]),
              "winter": df.loc[df["winter"], "pm25"].corr(df.loc[df["winter"], "humidity"]),
              "non_winter": df.loc[~df["winter"], "pm25"].corr(df.loc[~df["winter"], "humidity"])}
    print("pooled:", {k: round(v, 3) for k, v in pooled.items()})
    path = os.path.join(_ROOT, "models", f"humidity_raw_correlation_clean_{TIMESTAMP}.csv")
    out.to_csv(path, index=False)
    print("บันทึกที่:", path)


if __name__ == "__main__":
    main()

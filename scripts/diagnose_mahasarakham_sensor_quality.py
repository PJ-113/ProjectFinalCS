import os

import numpy as np
import pandas as pd

# CHANGELOG (75) ตั้งข้อสงสัยไว้ว่าสถานีมหาสารคามที่ registry เลือก (Pramuan Suk Health Park,
# coverage_overlap_days 544.9/913 — ต่ำกว่าจังหวัดอื่นที่เกือบเต็ม) อาจเป็นเซนเซอร์ AirGradient
# ราคาถูก ไม่ใช่ reference-grade ตามที่ผู้ใช้เคยรู้มา — ไม่เคยถูกตรวจสอบซ้ำเลย สคริปต์นี้เช็คคร่าวๆ
# ก่อนใช้ข้อมูลจังหวัดนี้เทรนจริงจังในการทดลอง all-20-province
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(
    _ROOT, "data", "multiprovince_pm25_features_raw_spatial_k1_firehotspot_nbtrend_20260919_174805.csv"
)

TARGET = "mahasarakham"
NEIGHBORS = ["khonkaen", "roiet"]  # ติดกับมหาสารคามโดยตรง ใช้เทียบลักษณะข้อมูล


def summarize(series):
    s = series.dropna()
    diffs = s.diff().dropna()
    repeat_ratio = float((diffs == 0).mean()) if len(diffs) else float("nan")
    # เซนเซอร์ราคาถูกมักปัดเศษหยาบกว่า (เช่น จำนวนเต็ม) ดูว่าเศษทศนิยมกระจายแบบไหน
    frac_is_integer = float(np.isclose(s % 1, 0).mean())
    return {
        "n": len(s),
        "pct_missing": float(series.isna().mean() * 100),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "min": float(s.min()),
        "max": float(s.max()),
        "pct_consecutive_repeat": repeat_ratio * 100,
        "pct_integer_values": frac_is_integer * 100,
        "autocorr_lag1": float(s.autocorr(lag=1)),
    }


def main():
    print(f"โหลดข้อมูล: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, usecols=["province_en", "datetime_utc", "pm25"])
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])

    provinces = [TARGET] + NEIGHBORS
    rows = []
    for p in provinces:
        g = df[df["province_en"] == p].sort_values("datetime_utc")
        stats = summarize(g["pm25"])
        stats["province_en"] = p
        rows.append(stats)

    out = pd.DataFrame(rows).set_index("province_en")
    cols = ["n", "pct_missing", "mean", "std", "min", "max",
            "pct_consecutive_repeat", "pct_integer_values", "autocorr_lag1"]
    print("\n=== เทียบสถิติพื้นฐาน PM2.5 ดิบ: มหาสารคาม vs จังหวัดข้างเคียง ===")
    print(out[cols].round(3).to_string())

    ks_target = out.loc[TARGET]
    for n in NEIGHBORS:
        ks_n = out.loc[n]
        print(f"\n--- {TARGET} เทียบ {n} ---")
        print(f"  ค่าซ้ำติดกัน (flatlining): {ks_target['pct_consecutive_repeat']:.1f}% vs {ks_n['pct_consecutive_repeat']:.1f}%")
        print(f"  สัดส่วนค่าเป็นจำนวนเต็ม: {ks_target['pct_integer_values']:.1f}% vs {ks_n['pct_integer_values']:.1f}%")
        print(f"  std: {ks_target['std']:.2f} vs {ks_n['std']:.2f}")

    print("\nหมายเหตุ: การเช็คนี้เป็นการดูลักษณะข้อมูลคร่าวๆ (flatlining/ปัดเศษ/ความผันผวน) "
          "ไม่ใช่การยืนยัน hardware โดยตรง ใช้เป็นข้อมูลประกอบการตีความผลการทดลองเท่านั้น "
          "ไม่ใช่เกณฑ์ตัดสินให้หยุด/ไม่หยุดการทดลอง")


if __name__ == "__main__":
    main()

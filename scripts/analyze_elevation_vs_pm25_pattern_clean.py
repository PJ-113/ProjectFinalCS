"""
ตอบคำถามอาจารย์ตรงๆ: "ภาคอีสานที่เป็นพื้นที่สูงมักมีความรุนแรงของฝุ่นไม่ต่อเนื่อง" จริงไหม
(ขอบเขต: เทียบภายใน 20 จังหวัดอีสานเอง ไม่เทียบกับภาคเหนือเพราะไม่มีข้อมูล)

สมมติฐาน: จังหวัดที่ elevation สูงกว่า ควรมี PM2.5 ผันผวน/ไม่ต่อเนื่องมากกว่า (std สูงกว่า, การ
เปลี่ยนแปลงชั่วโมงต่อชั่วโมงแรงกว่า) — วัดด้วย 2 metric: (1) std ของ pm25 รายจังหวัด (2) ค่าเฉลี่ย
|diff(pm25)| ระหว่างชั่วโมงติดกัน (volatility เชิงเวลา)
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(_ROOT, "data", "multiprovince_pm25_features_raw_spatial_k1_firehotspot_nbtrend_20260919_174805.csv")
REGISTRY_PATH = os.path.join(_ROOT, "data", "province_station_registry.json")
FIGURES_DIR = os.path.join(_ROOT, "figures")
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")


def main():
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        registry = json.load(f)
    elevation_by_province = {p["province_en"]: p["elevation_m"] for p in registry["provinces"]}
    assert all("elevation_m" in p for p in registry["provinces"]), \
        "บาง province ไม่มี elevation_m — รัน scripts/fetch_province_elevation.py ก่อน"

    print(f"โหลดข้อมูล: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, usecols=["province_en", "datetime_utc", "pm25"])
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    df = df.sort_values(["province_en", "datetime_utc"])

    rows = []
    for province, g in df.groupby("province_en"):
        g = g.sort_values("datetime_utc")
        is_consecutive = g["datetime_utc"].diff() == pd.Timedelta(hours=1)
        hourly_diff = g["pm25"].diff()[is_consecutive].abs()
        rows.append({
            "province_en": province,
            "elevation_m": elevation_by_province[province],
            "pm25_mean": g["pm25"].mean(),
            "pm25_std": g["pm25"].std(),
            "mean_abs_hourly_diff": hourly_diff.mean(),
            "n": len(g),
        })
    stats_df = pd.DataFrame(rows).sort_values("elevation_m").reset_index(drop=True)
    print("\nสถิติต่อจังหวัด (เรียงตาม elevation):")
    print(stats_df.to_string(index=False))

    out_csv = os.path.join(_ROOT, "models", f"elevation_vs_pm25_pattern_{TIMESTAMP}.csv")
    stats_df.to_csv(out_csv, index=False)
    print(f"\nบันทึกที่: {out_csv}")

    corr_std = stats_df["elevation_m"].corr(stats_df["pm25_std"])
    corr_diff = stats_df["elevation_m"].corr(stats_df["mean_abs_hourly_diff"])
    print(f"\nPearson correlation(elevation_m, pm25_std) = {corr_std:.4f}")
    print(f"Pearson correlation(elevation_m, mean_abs_hourly_diff) = {corr_diff:.4f}")

    print("\n" + "=" * 80)
    for name, corr in [("pm25_std (ความผันผวนโดยรวม)", corr_std),
                        ("mean_abs_hourly_diff (ความไม่ต่อเนื่องชั่วโมงต่อชั่วโมง)", corr_diff)]:
        if corr > 0.2:
            verdict = "ตรงสมมติฐาน (ที่สูงกว่า ผันผวนมากกว่าจริง)"
        elif corr < -0.2:
            verdict = "ขัดสมมติฐาน (ที่สูงกว่า กลับผันผวนน้อยกว่า)"
        else:
            verdict = "ไม่มีความสัมพันธ์ชัดเจน (|r|<0.2)"
        print(f"{name}: r={corr:.3f} -> {verdict}")
    print("=" * 80)
    print(f"หมายเหตุ: elevation ของ 20 จังหวัดอีสานอยู่ในช่วงแคบ ({stats_df['elevation_m'].min():.0f}-"
          f"{stats_df['elevation_m'].max():.0f}m) เพราะเป็นที่ราบสูงโคราชทั้งหมด ไม่ใช่ภูเขาสูงแบบภาคเหนือ "
          f"— ถ้า correlation อ่อน อาจเป็นเพราะ range ของตัวแปรเองแคบเกินจะเห็นผลชัด ไม่ใช่ว่าไม่มีผลจริง")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, ycol, ylabel in [(axes[0], "pm25_std", "PM2.5 std (volatility)"),
                              (axes[1], "mean_abs_hourly_diff", "mean |hour-to-hour diff|")]:
        ax.scatter(stats_df["elevation_m"], stats_df[ycol], s=40)
        for _, row in stats_df.iterrows():
            ax.annotate(row["province_en"], (row["elevation_m"], row[ycol]), fontsize=6,
                        xytext=(3, 3), textcoords="offset points")
        z = np.polyfit(stats_df["elevation_m"], stats_df[ycol], 1)
        trend_x = np.linspace(stats_df["elevation_m"].min(), stats_df["elevation_m"].max(), 50)
        ax.plot(trend_x, np.polyval(z, trend_x), color="red", linewidth=1.5)
        ax.set_xlabel("Elevation (m)")
        ax.set_ylabel(ylabel)
    axes[0].set_title(f"Elevation vs PM2.5 std (r={corr_std:.3f})")
    axes[1].set_title(f"Elevation vs hourly volatility (r={corr_diff:.3f})")
    fig.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, f"elevation_vs_pm25_pattern_{TIMESTAMP}.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"\nบันทึกกราฟที่: {fig_path}")


if __name__ == "__main__":
    main()

"""
วัดความสัมพันธ์ระหว่าง amenity/shop density (รัศมี 3km) กับ pattern ของ PM2.5 รายจังหวัด
(pm25_mean, pm25_std, ความผันผวนรายชั่วโมง) ด้วย Pearson correlation
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
    density_by_province = {p["province_en"]: p["amenity_density_3km"] for p in registry["provinces"]}
    assert all("amenity_density_3km" in p for p in registry["provinces"]), \
        "บาง province ไม่มี amenity_density_3km — รัน scripts/fetch_province_amenity_density.py ก่อน"

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
            "amenity_density_3km": density_by_province[province],
            "pm25_mean": g["pm25"].mean(),
            "pm25_std": g["pm25"].std(),
            "mean_abs_hourly_diff": hourly_diff.mean(),
            "n": len(g),
        })
    stats_df = pd.DataFrame(rows).sort_values("amenity_density_3km").reset_index(drop=True)
    print("\nสถิติต่อจังหวัด (เรียงตาม amenity_density_3km):")
    print(stats_df.to_string(index=False))

    out_csv = os.path.join(_ROOT, "models", f"amenity_density_vs_pm25_pattern_{TIMESTAMP}.csv")
    stats_df.to_csv(out_csv, index=False)
    print(f"\nบันทึกที่: {out_csv}")

    corr_mean = stats_df["amenity_density_3km"].corr(stats_df["pm25_mean"])
    corr_std = stats_df["amenity_density_3km"].corr(stats_df["pm25_std"])
    corr_diff = stats_df["amenity_density_3km"].corr(stats_df["mean_abs_hourly_diff"])
    print(f"\nPearson correlation(amenity_density, pm25_mean) = {corr_mean:.4f}")
    print(f"Pearson correlation(amenity_density, pm25_std) = {corr_std:.4f}")
    print(f"Pearson correlation(amenity_density, mean_abs_hourly_diff) = {corr_diff:.4f}")

    print("\n" + "=" * 80)
    for name, corr in [("pm25_mean (ระดับฝุ่นเฉลี่ย)", corr_mean),
                        ("pm25_std (ความผันผวนโดยรวม)", corr_std),
                        ("mean_abs_hourly_diff (ความไม่ต่อเนื่องชั่วโมงต่อชั่วโมง)", corr_diff)]:
        if abs(corr) > 0.3:
            direction = "เขตเมืองหนาแน่นกว่า -> ค่าสูงกว่า" if corr > 0 else "เขตเมืองหนาแน่นกว่า -> ค่าต่ำกว่า"
            print(f"{name}: r={corr:.3f} -> มีความสัมพันธ์ชัดเจนพอสมควร ({direction})")
        else:
            print(f"{name}: r={corr:.3f} -> ไม่มีความสัมพันธ์ชัดเจน (|r|<=0.3)")
    print("=" * 80)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, ycol, ylabel in [(axes[0], "pm25_mean", "PM2.5 mean"),
                              (axes[1], "pm25_std", "PM2.5 std (volatility)"),
                              (axes[2], "mean_abs_hourly_diff", "mean |hour-to-hour diff|")]:
        ax.scatter(stats_df["amenity_density_3km"], stats_df[ycol], s=40)
        for _, row in stats_df.iterrows():
            ax.annotate(row["province_en"], (row["amenity_density_3km"], row[ycol]), fontsize=6,
                        xytext=(3, 3), textcoords="offset points")
        z = np.polyfit(stats_df["amenity_density_3km"], stats_df[ycol], 1)
        trend_x = np.linspace(stats_df["amenity_density_3km"].min(), stats_df["amenity_density_3km"].max(), 50)
        ax.plot(trend_x, np.polyval(z, trend_x), color="red", linewidth=1.5)
        ax.set_xlabel("Amenity/shop density (3km radius)")
        ax.set_ylabel(ylabel)
    axes[0].set_title(f"Density vs PM2.5 mean (r={corr_mean:.3f})")
    axes[1].set_title(f"Density vs PM2.5 std (r={corr_std:.3f})")
    axes[2].set_title(f"Density vs hourly volatility (r={corr_diff:.3f})")
    fig.tight_layout()
    fig_path = os.path.join(FIGURES_DIR, f"amenity_density_vs_pm25_pattern_{TIMESTAMP}.png")
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"\nบันทึกกราฟที่: {fig_path}")


if __name__ == "__main__":
    main()

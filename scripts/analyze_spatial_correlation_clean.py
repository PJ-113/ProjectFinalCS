"""
วัดความสัมพันธ์ระหว่างระยะทาง (haversine) กับ correlation ของ PM2.5 ระหว่างคู่จังหวัด
(Pearson correlation ของ pairwise PM2.5 correlation matrix vs distance matrix)
"""
import importlib.util
import itertools
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


def load_haversine():
    spec = importlib.util.spec_from_file_location(
        "province_registry", os.path.join(_ROOT, "scripts", "province_registry.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.haversine_km


def main():
    haversine_km = load_haversine()

    print(f"โหลด registry: {REGISTRY_PATH}")
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        registry = json.load(f)
    provinces = {p["province_en"]: p for p in registry["provinces"]}
    print(f"จังหวัดทั้งหมด: {len(provinces)}")

    print(f"โหลดข้อมูล PM2.5: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, usecols=["province_en", "datetime_utc", "pm25"])
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])

    pivot = df.pivot_table(index="datetime_utc", columns="province_en", values="pm25")
    print(f"pivot matrix: {pivot.shape[0]} timestamps x {pivot.shape[1]} จังหวัด")

    corr_matrix = pivot.corr(method="pearson", min_periods=100)
    print("\nคำนวณ correlation matrix เสร็จ (pairwise, min_periods=100 ชั่วโมง overlap)")

    provinces_ordered = sorted(corr_matrix.columns.tolist())
    n = len(provinces_ordered)
    dist_matrix = pd.DataFrame(index=provinces_ordered, columns=provinces_ordered, dtype=float)
    for a, b in itertools.product(provinces_ordered, repeat=2):
        pa, pb = provinces[a], provinces[b]
        dist_matrix.loc[a, b] = haversine_km(pa["lat"], pa["lon"], pb["lat"], pb["lon"])

    pairs = []
    for a, b in itertools.combinations(provinces_ordered, 2):
        c = corr_matrix.loc[a, b]
        d = dist_matrix.loc[a, b]
        if pd.notna(c):
            pairs.append({"province_a": a, "province_b": b, "distance_km": d, "correlation": c})
    pairs_df = pd.DataFrame(pairs).sort_values("distance_km").reset_index(drop=True)
    print(f"\nคู่จังหวัดที่มีข้อมูลพอ (min 100 ชม. overlap): {len(pairs_df)}/{n * (n - 1) // 2}")

    overall_corr = pairs_df["distance_km"].corr(pairs_df["correlation"])
    print(f"\nPearson correlation(distance_km, pm25_correlation) ระหว่างคู่จังหวัด = {overall_corr:.4f}")
    if overall_corr < -0.1:
        print("-> ตรงสมมติฐาน: ยิ่งไกลกัน correlation ยิ่งต่ำ (สนับสนุน spatial-lag feature ใน task F)")
    elif overall_corr > 0.1:
        print("-> ขัดสมมติฐาน (correlation กลับสูงขึ้นตามระยะทาง — ผิดปกติ ควรตรวจข้อมูลเพิ่ม)")
    else:
        print("-> ไม่มีความสัมพันธ์ชัดเจนระหว่างระยะทางกับ correlation (อ่อนแอ) — อาจไม่คุ้มลงทุน task F")

    print("\n10 คู่จังหวัดที่ใกล้กันที่สุด:")
    print(pairs_df.head(10)[["province_a", "province_b", "distance_km", "correlation"]].to_string(index=False))
    print("\n10 คู่จังหวัดที่ไกลกันที่สุด:")
    print(pairs_df.tail(10)[["province_a", "province_b", "distance_km", "correlation"]].to_string(index=False))

    csv_path = os.path.join(_ROOT, "models", f"spatial_correlation_pairs_clean_{TIMESTAMP}.csv")
    pairs_df.to_csv(csv_path, index=False)
    print(f"\nบันทึกตารางคู่จังหวัดที่: {csv_path}")

    os.makedirs(FIGURES_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr_matrix.loc[provinces_ordered, provinces_ordered], cmap="RdYlBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(n)); ax.set_xticklabels(provinces_ordered, rotation=90, fontsize=7)
    ax.set_yticks(range(n)); ax.set_yticklabels(provinces_ordered, fontsize=7)
    ax.set_title("PM2.5 correlation between provinces (Pearson, pairwise hourly)")
    fig.colorbar(im, ax=ax, label="correlation")
    fig.tight_layout()
    heatmap_path = os.path.join(FIGURES_DIR, f"spatial_correlation_heatmap_{TIMESTAMP}.png")
    fig.savefig(heatmap_path, dpi=150)
    plt.close(fig)
    print(f"บันทึก heatmap ที่: {heatmap_path}")

    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.scatter(pairs_df["distance_km"], pairs_df["correlation"], alpha=0.5, s=20)
    z = np.polyfit(pairs_df["distance_km"], pairs_df["correlation"], 1)
    trend_x = np.linspace(pairs_df["distance_km"].min(), pairs_df["distance_km"].max(), 50)
    ax2.plot(trend_x, np.polyval(z, trend_x), color="red", linewidth=2, label=f"trend (slope={z[0]:.5f}/km)")
    ax2.set_xlabel("Distance between provinces (km, haversine)")
    ax2.set_ylabel("PM2.5 correlation (Pearson)")
    ax2.set_title(f"Correlation vs distance (r={overall_corr:.3f}, n={len(pairs_df)} province pairs)")
    ax2.legend()
    fig2.tight_layout()
    scatter_path = os.path.join(FIGURES_DIR, f"spatial_correlation_vs_distance_{TIMESTAMP}.png")
    fig2.savefig(scatter_path, dpi=150)
    plt.close(fig2)
    print(f"บันทึก scatter plot ที่: {scatter_path}")


if __name__ == "__main__":
    main()

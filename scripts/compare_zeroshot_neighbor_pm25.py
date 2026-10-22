"""
เทียบค่า PM2.5 จริง (ไม่ใช่ค่าพยากรณ์) ของแต่ละจังหวัด zero-shot (3 จังหวัด) กับจังหวัดเพื่อนบ้านที่ใกล้สุด
ที่ใช้เทรนจริง (จับคู่ด้วย haversine): มหาสารคาม<->กาฬสินธุ์, ร้อยเอ็ด<->กาฬสินธุ์, นครพนม<->สกลนคร
วัด correlation ของค่า PM2.5 จริงทั้งคู่ (ทั้งช่วงข้อมูลและช่วงฤดูฝุ่น ม.ค.-มี.ค. 2569) และพล็อตกราฟเทียบ
"""
import importlib.util
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

# ใช้ฟอนต์ Tahoma เพื่อรองรับข้อความไทยในกราฟ
matplotlib.rcParams["font.family"] = "Tahoma"

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(_ROOT, "data", "multiprovince_pm25_features_raw_spatial_k1_firehotspot_nbtrend_20260919_174805.csv")
REGISTRY_PATH = os.path.join(_ROOT, "data", "province_station_registry.json")
FIGURES_DIR = os.path.join(_ROOT, "figures")
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

# ช่วงตัวอย่าง: ฤดูฝุ่น ม.ค.-มี.ค. 2569
WINDOW_START = "2026-01-01"
WINDOW_END = "2026-03-31"

# สี: น้ำเงิน = จังหวัด zero-shot, ส้ม = เพื่อนบ้านที่เทรนจริง (คงที่ทั้ง 3 กราฟ)
COLOR_ZEROSHOT = "#2a78d6"
COLOR_NEIGHBOR = "#eb6834"


def load_haversine():
    spec = importlib.util.spec_from_file_location("province_registry", os.path.join(_ROOT, "scripts", "province_registry.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.haversine_km


def find_nearest_trained(zs, trained, haversine_km):
    return min(trained, key=lambda t: haversine_km(zs["lat"], zs["lon"], t["lat"], t["lon"]))


def main():
    haversine_km = load_haversine()

    with open(REGISTRY_PATH, encoding="utf-8") as f:
        registry = json.load(f)
    provinces = registry["provinces"]
    zero_shot = [p for p in provinces if p["is_zero_shot"]]
    trained = [p for p in provinces if not p["is_zero_shot"]]

    print(f"โหลดข้อมูล PM2.5: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH, usecols=["province_en", "datetime_utc", "pm25"])
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])

    os.makedirs(FIGURES_DIR, exist_ok=True)
    summary_rows = []

    for zs in zero_shot:
        nb = find_nearest_trained(zs, trained, haversine_km)
        dist_km = haversine_km(zs["lat"], zs["lon"], nb["lat"], nb["lon"])
        print(f"\n{zs['province_th']} (zero-shot) <-> {nb['province_th']} (เทรนจริง)  ระยะ {dist_km:.1f} กม.")

        s_zs = df[df["province_en"] == zs["province_en"]].set_index("datetime_utc")["pm25"]
        s_nb = df[df["province_en"] == nb["province_en"]].set_index("datetime_utc")["pm25"]
        aligned = pd.concat([s_zs.rename("zs"), s_nb.rename("nb")], axis=1).dropna()
        corr_full = aligned["zs"].corr(aligned["nb"])
        print(f"  correlation (ข้อมูลทั้งหมดที่ overlap กัน, n={len(aligned)} ชม.) = {corr_full:.3f}")

        window = aligned.loc[WINDOW_START:WINDOW_END]
        corr_window = window["zs"].corr(window["nb"])
        print(f"  correlation (เฉพาะช่วง {WINDOW_START}..{WINDOW_END}, n={len(window)} ชม.) = {corr_window:.3f}")

        summary_rows.append({
            "zero_shot_province": zs["province_th"], "nearest_trained_province": nb["province_th"],
            "distance_km": round(dist_km, 1),
            "correlation_full": round(float(corr_full), 3), "n_full": len(aligned),
            "correlation_window": round(float(corr_window), 3), "n_window": len(window),
        })

        fig, ax = plt.subplots(figsize=(10, 4.5))
        ax.plot(window.index, window["zs"], color=COLOR_ZEROSHOT, linewidth=1.5,
                label=f"{zs['province_th']} (zero-shot)")
        ax.plot(window.index, window["nb"], color=COLOR_NEIGHBOR, linewidth=1.5,
                label=f"{nb['province_th']} (เทรนจริง, ห่าง {dist_km:.0f} กม.)")
        ax.set_ylabel("PM2.5 จริง (µg/m³)")
        ax.set_title(
            f"{zs['province_th']} vs {nb['province_th']}: ค่า PM2.5 จริงรายชั่วโมง "
            f"({WINDOW_START} ถึง {WINDOW_END}, r={corr_window:.2f})"
        )
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
        ax.grid(True, axis="y", linewidth=0.5, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(loc="upper right", frameon=False)
        fig.tight_layout()

        fig_path = os.path.join(FIGURES_DIR, f"zeroshot_vs_neighbor_{zs['province_en']}_{TIMESTAMP}.png")
        fig.savefig(fig_path, dpi=150)
        plt.close(fig)
        print(f"  บันทึกกราฟที่: {fig_path}")

    print("\n" + "=" * 70)
    print("สรุปทั้ง 3 คู่ (สำหรับใส่เล่ม):")
    print("=" * 70)
    summary_df = pd.DataFrame(summary_rows)
    print(summary_df.to_string(index=False))

    csv_path = os.path.join(_ROOT, "models", f"zeroshot_vs_neighbor_correlation_{TIMESTAMP}.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\nบันทึกตารางสรุปที่: {csv_path}")


if __name__ == "__main__":
    main()

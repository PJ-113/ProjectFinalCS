"""
วัดขนาดช่องว่างจริงระหว่างข้อมูลอุตุนิยมวิทยาที่ใช้ฝึกสอนโมเดล (Open-Meteo Archive API —
reanalysis, สร้างหลังเกิดเหตุการณ์จริง) กับข้อมูลที่ระบบใช้งานจริงได้ (Open-Meteo Forecast API —
nowcast จากโมเดลพยากรณ์อากาศ มีความคลาดเคลื่อนในตัวเอง) — พบจากการตรวจ CLAUDE review (Perplexity)
ว่า build_raw_dataset_multiprovince.py ใช้ archive-api ส่วน app.py ใช้ api.open-meteo.com/v1/forecast
คนละ endpoint กัน ไม่เคยวัดผลกระทบมาก่อนในโปรเจกต์นี้

ข้อจำกัดของวิธีวัด: ดึง "current weather สด" ย้อนหลังจริงไม่ได้ (Forecast API ให้แค่ค่าปัจจุบัน ณ
ตอนเรียก ไม่มี endpoint ย้อนอดีต) จึงใช้ **Historical Forecast API**
(historical-forecast-api.open-meteo.com) แทน ตามที่ Open-Meteo แนะนำเองสำหรับ backtesting ระบบที่
ใช้ Forecast API จริง — เก็บค่าที่โมเดลพยากรณ์อากาศของ Open-Meteo เคยทำนายไว้จริงในอดีต (ไม่ใช่ reanalysis)
ใกล้เคียงกับคุณภาพข้อมูลที่ระบบสดได้รับมากกว่า Archive API

วิธีทำ: สุ่ม 3 ช่วงเวลา ~10 วัน กระจายทั่วปี (คลุมทั้งฤดูฝุ่น/ฝน/ปกติ) x 17 จังหวัดที่ใช้ฝึกโมเดล
ดึงทั้ง 2 แหล่งในช่วงเดียวกัน เทียบ MAE/correlation ต่อตัวแปร (temp, humidity, wind_speed,
precipitation, pressure, wind_direction) แล้ววัดผลกระทบจริงต่อการพยากรณ์: แทนค่าอุตุนิยมวิทยาใน
feature row ของ test set ด้วยค่าจาก historical-forecast แทน archive แล้วเทียบ MAE การพยากรณ์ PM2.5
ของโมเดล v3 ที่ deploy จริง (ไม่ retrain) ว่าต่างจาก MAE บน feature archive (ของเดิมที่ใช้ประเมินในบทที่ 4) แค่ไหน

รัน: python scripts/experiment_weather_train_serve_skew.py
"""
import importlib.util
import json
import os
import time

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import mean_absolute_error

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(_ROOT, "models")
REGISTRY_PATH = os.path.join(_ROOT, "data", "province_station_registry.json")
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

HOURLY_VARS = "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,wind_direction_10m,surface_pressure"
VAR_MAP = {
    "temperature_2m": "temp", "relative_humidity_2m": "humidity", "wind_speed_10m": "wind_speed",
    "precipitation": "precipitation", "wind_direction_10m": "wind_dir_deg", "surface_pressure": "pressure",
}

# 3 ช่วง ~12 วัน กระจายฤดูกาล ภายในขอบเขตข้อมูลฝึกสอนจริง (2024-01-01 .. 2026-07-02)
WINDOWS = [
    ("2024-03-01", "2024-03-12"),   # ฤดูฝุ่น (ก.พ.-เม.ย.)
    ("2024-08-01", "2024-08-12"),   # ฤดูฝน
    ("2025-11-15", "2025-11-26"),   # ฤดูหนาว/ปกติ
]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, "scripts", filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fetch(base_url, lat, lon, start, end, label):
    params = {"latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
              "hourly": HOURLY_VARS, "timezone": "UTC"}
    for attempt in range(4):
        try:
            r = requests.get(base_url, params=params, timeout=30)
        except requests.exceptions.RequestException as e:
            print(f"    !! network error ({label}, attempt {attempt+1}): {e}")
            time.sleep(2 ** attempt)
            continue
        if r.status_code == 429 or r.status_code >= 500:
            print(f"    !! HTTP {r.status_code} ({label}, attempt {attempt+1})")
            time.sleep(2 ** attempt)
            continue
        r.raise_for_status()
        h = r.json()["hourly"]
        df = pd.DataFrame({"datetime_utc": pd.to_datetime(h["time"]).tz_localize("UTC")})
        for raw, col in VAR_MAP.items():
            df[col] = h[raw]
        return df
    print(f"    !! ยอมแพ้ ({label}) หลัง retry")
    return None


def main():
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        registry = json.load(f)
    provinces = [p for p in registry["provinces"] if not p.get("is_zero_shot", False)]
    print(f"จังหวัดที่ใช้ฝึกโมเดล (ไม่รวม zero-shot): {len(provinces)}")

    rows = []
    for wi, (start, end) in enumerate(WINDOWS):
        print(f"\nช่วงที่ {wi+1}: {start} .. {end}")
        for p in provinces:
            arc = fetch("https://archive-api.open-meteo.com/v1/archive", p["lat"], p["lon"], start, end, "archive")
            time.sleep(0.3)
            hfc = fetch("https://historical-forecast-api.open-meteo.com/v1/forecast", p["lat"], p["lon"], start, end, "hist-forecast")
            time.sleep(0.3)
            if arc is None or hfc is None:
                print(f"  {p['province_en']}: ข้ามเพราะดึงไม่สำเร็จ")
                continue
            merged = arc.merge(hfc, on="datetime_utc", suffixes=("_archive", "_forecast"))
            merged["province_en"] = p["province_en"]
            merged["window"] = wi
            rows.append(merged)
            print(f"  {p['province_en']}: {len(merged)} ชั่วโมง")

    full = pd.concat(rows, ignore_index=True)
    out_path = os.path.join(MODEL_DIR, f"weather_train_serve_skew_raw_{TIMESTAMP}.csv")
    full.to_csv(out_path, index=False)
    print(f"\nบันทึกข้อมูลดิบที่: {out_path} ({len(full)} แถว)")

    print("\n" + "=" * 80)
    print("ความต่างระหว่าง Archive API (ใช้ฝึกสอน) vs Historical Forecast API (ใกล้เคียงของจริง) ต่อตัวแปร:")
    print("=" * 80)
    summary_rows = []
    for var in ["temp", "humidity", "wind_speed", "precipitation", "pressure", "wind_dir_deg"]:
        a = full[f"{var}_archive"]
        f_ = full[f"{var}_forecast"]
        valid = a.notna() & f_.notna()
        mae = mean_absolute_error(a[valid], f_[valid])
        bias = (f_[valid] - a[valid]).mean()
        corr = a[valid].corr(f_[valid])
        std_a = a[valid].std()
        summary_rows.append({"variable": var, "n": int(valid.sum()), "mae": mae, "bias": bias,
                              "corr": corr, "std_archive": std_a, "mae_pct_of_std": mae / std_a * 100 if std_a else np.nan})
    summary = pd.DataFrame(summary_rows)
    print(summary.round(4).to_string(index=False))
    summary_path = os.path.join(MODEL_DIR, f"weather_train_serve_skew_summary_{TIMESTAMP}.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nบันทึกสรุปที่: {summary_path}")

    # ===== ส่วนที่ 2: ผลกระทบต่อ MAE การพยากรณ์ PM2.5 จริง (ไม่ retrain ใช้โมเดล v3 deploy จริง) =====
    print("\n" + "=" * 80)
    print("ส่วนที่ 2: ผลกระทบต่อ MAE การพยากรณ์ PM2.5 จริง")
    print("=" * 80)

    ens_mod = load_module("eval_ensemble_vs_routing_clean", "eval_ensemble_vs_routing_clean.py")
    bt_mod = load_module("analyze_burning_season_backtest_clean", "analyze_burning_season_backtest_clean.py")
    train_mod = bt_mod.load_train_module()

    print(f"โหลด feature dataset หลัก: {train_mod.DATA_PATH}")
    df_all = pd.read_csv(train_mod.DATA_PATH)
    df_all["datetime_utc"] = pd.to_datetime(df_all["datetime_utc"], utc=True).dt.tz_localize(None)

    full["datetime_utc_naive"] = full["datetime_utc"].dt.tz_localize(None)
    weather_lookup = full.set_index(["province_en", "datetime_utc_naive"])

    # เฉพาะแถวที่ตรงกับช่วงเวลา/จังหวัดที่ดึงมาเทียบได้ และอยู่ใน test split ของ 17 จังหวัดฝึกสอน
    df_trained = df_all[~df_all["is_zero_shot"]].reset_index(drop=True)
    _, _, test_df = train_mod.split_train_calib_test(df_trained)
    idx = test_df.set_index(["province_en", "datetime_utc"]).index
    overlap_mask = idx.isin(weather_lookup.index)
    test_overlap = test_df[overlap_mask].copy()
    print(f"แถว test set ที่ตรงกับช่วงเวลาที่สุ่มมาเทียบได้: {len(test_overlap)} จาก {len(test_df)}")

    if len(test_overlap) < 20:
        print("overlap น้อยเกินไป (test set เป็น 20% ท้ายสุดตามเวลา อาจไม่ตรงกับช่วงที่สุ่มมา) ข้ามส่วนที่ 2")
        return

    # สร้างสำเนา feature row โดยแทนค่าตัวแปรอุตุนิยมวิทยาด้วยค่าจาก historical-forecast แทน archive เดิม
    test_forecast_quality = test_overlap.copy()
    for var in ["temp", "humidity", "wind_speed", "precipitation", "pressure"]:
        vals = [weather_lookup.loc[(r.province_en, r.datetime_utc), f"{var}_forecast"]
                if (r.province_en, r.datetime_utc) in weather_lookup.index else np.nan
                for r in test_overlap.itertuples()]
        test_forecast_quality[var] = vals
    wd_vals = [weather_lookup.loc[(r.province_en, r.datetime_utc), "wind_dir_deg_forecast"]
               if (r.province_en, r.datetime_utc) in weather_lookup.index else np.nan
               for r in test_overlap.itertuples()]
    test_forecast_quality["wind_dir_sin"] = np.sin(2 * np.pi * np.array(wd_vals) / 360)
    test_forecast_quality["wind_dir_cos"] = np.cos(2 * np.pi * np.array(wd_vals) / 360)

    pkgs = {tag: bt_mod.load_model_pkg(tag) for tag in ("lgb", "xgb")}
    for tag in ("lgb", "xgb"):
        preds_archive = ens_mod.predict_raw_quantiles(pkgs[tag], test_overlap, pkgs[tag]["feature_cols"])
        preds_forecast = ens_mod.predict_raw_quantiles(pkgs[tag], test_forecast_quality, pkgs[tag]["feature_cols"])
        merged_cmp = preds_archive.merge(preds_forecast, on=["province_en", "datetime_utc", "horizon"],
                                          suffixes=("_archive", "_forecast"))
        mae_archive = mean_absolute_error(merged_cmp["actual_archive"], merged_cmp["p50_archive"])
        mae_forecast = mean_absolute_error(merged_cmp["actual_forecast"], merged_cmp["p50_forecast"])
        pct = (mae_forecast - mae_archive) / mae_archive * 100
        print(f"\n{tag.upper()}: MAE ด้วย weather แบบ Archive (เหมือนประเมินในบทที่ 4) = {mae_archive:.4f}")
        print(f"{tag.upper()}: MAE ด้วย weather แบบ Historical Forecast (ใกล้เคียงของจริง) = {mae_forecast:.4f}  ({pct:+.2f}%)")


if __name__ == "__main__":
    main()

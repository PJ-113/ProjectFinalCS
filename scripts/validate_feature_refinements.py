"""
Phase B + C — ทดสอบ feature ใหม่/ปรับปรุงทีละตัว (baseline 20 feature + 1 candidate) ด้วย hyperparameter ที่จูนแล้วจาก A1
(lgb cfg06, xgb cfg02) เพื่อให้ delta% วัดเทียบโมเดลที่ดีที่สุดตอนนี้ ไม่ใช่ค่า default
Selection บน calib split เท่านั้น (ไม่แตะ test), P50-only, SCREEN_HORIZONS=[1,12,24]

candidate:
  B1 (FIRMS, ใช้ frp + ทิศลม): fire_frp_sum_100km_24h, fire_upwind_count_100km_24h, fire_upwind_frp_100km_24h
      (upwind = จุดไฟอยู่ในทิศที่ลมพัดมา ±60° ณ ชั่วโมงนั้น; wind_dir แบบ meteorological "ลมมาจาก")
      หมายเหตุ: log1p(count) ไม่ทดสอบ เพราะ tree-based invariant ต่อ monotonic transform
      หมายเหตุ: ไม่แยกในประเทศ/ข้ามพรมแดน เพราะไม่มี boundary data ในโปรเจกต์ (ทำแบบประมาณด้วย lon/lat จะไม่น่าเชื่อถือ)
  B2 (spatial-lag): nb_upwind_avg (เพื่อนบ้าน <=200km ถ่วง cos(มุมลม)+/ระยะ), nb_trend_3h (neighbor_avg(t)-(t-3h)),
      nb_max_150km (max pm25 ของจังหวัดภายใน 150km)
  B3 (dry-spell): dry_hours (ชั่วโมงตั้งแต่ฝนตกล่าสุด precip>0.1, cap 336), precip_sum_24h, precip_sum_72h
  C  (weather ใหม่): blh (boundary_layer_height), shortwave_radiation, cloud_cover (จาก fetch_extra_weather_isaan.py;
      blh ปี 2024 ขาด ~50% → ปล่อย NaN ให้ tree จัดการเอง)
ทุก feature ใช้ข้อมูล <= เวลา t เท่านั้น (ไม่ leak อนาคต) ยกเว้น neighbor/weather ณ เวลา t เอง (เหมือน pm25_neighbor_avg เดิม)
บันทึก dataset ที่มีคอลัมน์ candidate ทั้งหมดเป็นไฟล์ใหม่ เพื่อใช้ต่อใน Phase D
"""
import glob
import importlib.util
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_ROOT, "data")
MODEL_DIR = os.path.join(_ROOT, "models")
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
SCREEN_HORIZONS = [1, 12, 24]
BURNING_MONTHS = {2, 3, 4}
RADIUS_KM = 100
WINDOW_H = 24
UPWIND_HALF_ANGLE = 60.0

BEST = {
    "lgb": {"learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 50, "reg_lambda": 1.0, "n_estimators": 400},
    "xgb": {"learning_rate": 0.03, "max_depth": 8, "min_child_weight": 20, "reg_lambda": 10.0, "n_estimators": 800},
}
CANDIDATES = [
    "fire_frp_sum_100km_24h", "fire_upwind_count_100km_24h", "fire_upwind_frp_100km_24h",
    "nb_upwind_avg", "nb_trend_3h", "nb_max_150km",
    "dry_hours", "precip_sum_24h", "precip_sum_72h",
    "blh", "shortwave_radiation", "cloud_cover",
]


def latest_file(pattern):
    m = sorted(glob.glob(os.path.join(DATA_DIR, pattern)))
    if not m:
        raise FileNotFoundError(pattern)
    return m[-1]


def load_train_module():
    spec = importlib.util.spec_from_file_location(
        "train_multiprovince_models", os.path.join(_ROOT, "scripts", "train_multiprovince_models.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


def bearing_deg(lat1, lon1, lat2, lon2):
    """ทิศ (0-360, 0=เหนือ) จากจุดแรกไปจุดที่สอง"""
    p1, p2, dl = np.radians(lat1), np.radians(lat2), np.radians(lon2 - lon1)
    x = np.sin(dl) * np.cos(p2)
    y = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360


def angle_diff_deg(a, b):
    return np.abs((a - b + 180) % 360 - 180)


def add_fire_features(df, registry, hotspots):
    h_lat, h_lon = hotspots["latitude"].values, hotspots["longitude"].values
    h_dt, h_frp = hotspots["acq_datetime_utc"].values, hotspots["frp"].fillna(0).values
    for c in ("fire_frp_sum_100km_24h", "fire_upwind_count_100km_24h", "fire_upwind_frp_100km_24h"):
        df[c] = 0.0
    window = np.timedelta64(WINDOW_H, "h")
    for prov, meta in registry.items():
        rows = np.where(df["province_en"].values == prov)[0]
        if len(rows) == 0:
            continue
        near = haversine_km(meta["lat"], meta["lon"], h_lat, h_lon) <= RADIUS_KM
        order = np.argsort(h_dt[near])
        n_dt, n_frp = h_dt[near][order], h_frp[near][order]
        n_brg = bearing_deg(meta["lat"], meta["lon"], h_lat[near], h_lon[near])[order]
        ts = df["datetime_utc"].values[rows].astype("datetime64[ns]")
        wind_from = (np.degrees(np.arctan2(df["wind_dir_sin"].values[rows], df["wind_dir_cos"].values[rows])) + 360) % 360
        lo = np.searchsorted(n_dt, ts - window, side="right")
        hi = np.searchsorted(n_dt, ts, side="right")
        csum = np.concatenate([[0.0], np.cumsum(n_frp)])
        df.loc[df.index[rows], "fire_frp_sum_100km_24h"] = csum[hi] - csum[lo]
        up_cnt, up_frp = np.zeros(len(rows)), np.zeros(len(rows))
        for i in range(len(rows)):
            if hi[i] > lo[i]:
                m = angle_diff_deg(n_brg[lo[i]:hi[i]], wind_from[i]) <= UPWIND_HALF_ANGLE
                up_cnt[i] = m.sum()
                up_frp[i] = n_frp[lo[i]:hi[i]][m].sum()
        df.loc[df.index[rows], "fire_upwind_count_100km_24h"] = up_cnt
        df.loc[df.index[rows], "fire_upwind_frp_100km_24h"] = up_frp
        print(f"  fire {prov}: mean count(upwind)={up_cnt.mean():.2f}")
    return df


def add_neighbor_features(df, registry):
    pivot = df.pivot_table(index="datetime_utc", columns="province_en", values="pm25")
    provs = pivot.columns.tolist()
    wind_from_all = (np.degrees(np.arctan2(df["wind_dir_sin"].values, df["wind_dir_cos"].values)) + 360) % 360
    df["_wind_from"] = wind_from_all
    df["nb_upwind_avg"] = np.nan
    df["nb_max_150km"] = np.nan
    for p in provs:
        m = registry[p]
        others = [o for o in provs if o != p]
        d = np.array([haversine_km(m["lat"], m["lon"], registry[o]["lat"], registry[o]["lon"]) for o in others])
        b = np.array([bearing_deg(m["lat"], m["lon"], registry[o]["lat"], registry[o]["lon"]) for o in others])
        rows = np.where(df["province_en"].values == p)[0]
        ts = df["datetime_utc"].values[rows]
        vals = pivot.reindex(df["datetime_utc"].iloc[rows])[others].values  # (n, others)
        wind = df["_wind_from"].values[rows][:, None]
        # ถ่วง: cos ของมุมระหว่าง "ทิศที่ลมมา" กับ "ทิศไปหาเพื่อนบ้าน" (เฉพาะบวก) / ระยะ, เพื่อนบ้านภายใน 200km
        w = np.clip(np.cos(np.radians(angle_diff_deg(b[None, :], wind))), 0, None) / d[None, :]
        w = w * (d[None, :] <= 200)
        valid = ~np.isnan(vals)
        w = w * valid
        num = np.nansum(vals * w, axis=1)
        den = w.sum(axis=1)
        df.loc[df.index[rows], "nb_upwind_avg"] = np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)
        near = (d <= 150)[None, :] & valid
        masked = np.where(near, vals, -np.inf)
        mx = masked.max(axis=1)
        df.loc[df.index[rows], "nb_max_150km"] = np.where(np.isfinite(mx), mx, np.nan)
    df = df.drop(columns=["_wind_from"])
    # trend ของ pm25_neighbor_avg: (t) - (t-3h) ต่อจังหวัด แบบ time-indexed
    lag = df[["province_en", "datetime_utc", "pm25_neighbor_avg"]].copy()
    lag["datetime_utc"] = lag["datetime_utc"] + pd.Timedelta(hours=3)
    lag = lag.rename(columns={"pm25_neighbor_avg": "_nb_lag3"})
    df = df.merge(lag, on=["province_en", "datetime_utc"], how="left")
    df["nb_trend_3h"] = df["pm25_neighbor_avg"] - df["_nb_lag3"]
    return df.drop(columns=["_nb_lag3"])


def add_dry_spell_features(df):
    out = []
    for prov, g in df.groupby("province_en", sort=False):
        s = g.set_index("datetime_utc")["precipitation"]
        full = s.reindex(pd.date_range(s.index.min(), s.index.max(), freq="h"))
        rain_time = full.index.to_series().where(full.fillna(0) > 0.1)
        last_rain = rain_time.ffill()
        dry_h = (full.index.to_series() - last_rain).dt.total_seconds() / 3600
        dry_h = dry_h.fillna(336).clip(upper=336)  # ไม่เคยเห็นฝนเลยในหน้าต่าง -> cap
        p24 = full.rolling("24h", min_periods=1).sum()
        p72 = full.rolling("72h", min_periods=1).sum()
        f = pd.DataFrame({"dry_hours": dry_h, "precip_sum_24h": p24, "precip_sum_72h": p72})
        f = f.reindex(g["datetime_utc"]).reset_index()
        f["province_en"] = prov
        out.append(f)
    feats = pd.concat(out, ignore_index=True)
    return df.merge(feats, on=["province_en", "datetime_utc"], how="left")


def fit_p50(X, y, kind, train_mod):
    if kind == "lgb":
        import lightgbm as lgb
        return lgb.LGBMRegressor(objective="quantile", alpha=0.5, random_state=42, verbose=-1,
                                  **train_mod.BAGGING_PARAMS_LGB, **BEST["lgb"]).fit(X, y)
    import xgboost as xgb
    return xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, tree_method="hist",
                             random_state=42, verbosity=0, **train_mod.BAGGING_PARAMS_XGB, **BEST["xgb"]).fit(X, y)


def eval_variant(train_df, eval_df, feats, kind, name, train_mod):
    rows = []
    for h in SCREEN_HORIZONS:
        tcol = f"pm25_t+{h}"
        tr, ev = train_df.dropna(subset=[tcol]), eval_df.dropna(subset=[tcol])
        model = fit_p50(tr[feats], tr[tcol] - tr["pm25"], kind, train_mod)
        pred = model.predict(ev[feats]) + ev["pm25"].values
        y = ev[tcol].values
        burning = pd.to_datetime(ev["datetime_utc"]).dt.month.isin(BURNING_MONTHS).values
        hi = y > np.percentile(y, 90)
        rows.append({"variant": name, "model": kind, "horizon": h, "mae": mean_absolute_error(y, pred),
                     "mae_burning": mean_absolute_error(y[burning], pred[burning]),
                     "mae_gt_p90": mean_absolute_error(y[hi], pred[hi])})
    print(f"  [{name}/{kind}] MAE={np.mean([r['mae'] for r in rows]):.3f} "
          f">p90={np.mean([r['mae_gt_p90'] for r in rows]):.3f}", flush=True)
    return rows


def main():
    train_mod = load_train_module()
    df = pd.read_csv(train_mod.DATA_PATH)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    with open(os.path.join(DATA_DIR, "province_station_registry.json"), encoding="utf-8") as f:
        registry = {p["province_en"]: p for p in json.load(f)["provinces"]}
    hotspots = pd.read_csv(latest_file("firms_hotspots_isaan_*.csv"))
    hotspots["acq_datetime_utc"] = pd.to_datetime(hotspots["acq_datetime_utc"], utc=True).dt.tz_localize(None)
    df["datetime_utc"] = df["datetime_utc"]
    print(f"แถวรวม {len(df)}  |  คำนวณ feature ใหม่...")

    # fire ใช้เวลา naive-UTC เทียบกับ hotspot naive-UTC
    tmp = df.copy()
    tmp["datetime_utc"] = tmp["datetime_utc"].dt.tz_localize(None)
    tmp = add_fire_features(tmp, registry, hotspots)
    for c in ("fire_frp_sum_100km_24h", "fire_upwind_count_100km_24h", "fire_upwind_frp_100km_24h"):
        df[c] = tmp[c].values

    df = add_neighbor_features(df, registry)
    df = add_dry_spell_features(df)

    extra = pd.read_csv(latest_file("extra_weather_isaan_*.csv"))
    extra["datetime_utc"] = pd.to_datetime(extra["datetime_utc"], utc=True)
    extra = extra.rename(columns={"boundary_layer_height": "blh"})
    df = df.merge(extra[["province_en", "datetime_utc", "blh", "shortwave_radiation", "cloud_cover"]],
                  on=["province_en", "datetime_utc"], how="left")

    print("\nsanity (ค่าเฉลี่ยรายเดือนของ fire_frp_sum ควรสูงช่วง ก.พ.-เม.ย.):")
    print(df.groupby(df["datetime_utc"].dt.month)["fire_frp_sum_100km_24h"].mean().round(1).to_string())
    print("\nNaN ต่อคอลัมน์ candidate:")
    print(df[CANDIDATES].isna().mean().round(3).to_string())
    print("\ncorr กับ pm25:")
    print(df[CANDIDATES].corrwith(df["pm25"]).round(3).to_string())

    aug_path = os.path.join(DATA_DIR, f"multiprovince_pm25_features_refined_candidates_{TIMESTAMP}.csv")
    df.to_csv(aug_path, index=False)
    print(f"\nบันทึก dataset + candidate ที่: {aug_path}")

    trained = df[~df["is_zero_shot"]].reset_index(drop=True)
    train_df, calib_df, _ = train_mod.split_train_calib_test(trained)
    base = train_mod.FEATURE_COLS
    all_rows = []
    for kind in ("lgb", "xgb"):
        print(f"\n=== {kind.upper()} (evaluate บน calib, hyperparameter จูนแล้ว) ===")
        all_rows += eval_variant(train_df, calib_df, base, kind, "baseline", train_mod)
        for c in CANDIDATES:
            all_rows += eval_variant(train_df, calib_df, base + [c], kind, c, train_mod)

    res = pd.DataFrame(all_rows)
    out_path = os.path.join(MODEL_DIR, f"validate_feature_refinements_{TIMESTAMP}.csv")
    res.to_csv(out_path, index=False)
    print(f"\nบันทึกผลดิบที่: {out_path}")
    print("\n" + "=" * 100 + "\nสรุป delta% เทียบ baseline (ติดลบ = ดีขึ้น)\n" + "=" * 100)
    for metric in ("mae", "mae_burning", "mae_gt_p90"):
        t = res.groupby(["model", "variant"])[metric].mean().unstack("model")
        for k in ("lgb", "xgb"):
            t[f"{k}_d%"] = (t[k] - t.loc["baseline", k]) / t.loc["baseline", k] * 100
        print(f"\n--- {metric} ---")
        print(t.round(3).to_string())


if __name__ == "__main__":
    main()

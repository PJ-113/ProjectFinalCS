"""
Validate feature set บน dataset ที่แก้ timestamp offset แล้ว: baseline = 18 feature (ตัด
pm25_neighbor_avg/fire/nb_trend_12h) แล้วเพิ่มทีละตัว/ชุด:
  +nb_k1 / +nb_k2 / +nb_k3 / +nb_k5 (ค่าเฉลี่ย pm25 ของ k จังหวัดใกล้สุด ณ เวลาเดียวกัน)
  +fire, +nb_k1+nb_trend, +nb_k1+nb_trend+fire (ชุดปัจจุบัน), +time_cyclical
เทรนบน train, เลือกบน calib (ไม่แตะ test), P50, horizon [1,12,24], LGB+XGB
วัด MAE รวม/ปกติ/ฤดูฝุ่น (ก.พ.-เม.ย.)/>p90 ต่อ variant เทียบ baseline
"""
import glob
import importlib.util
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR, MODEL_DIR = os.path.join(_ROOT, "data"), os.path.join(_ROOT, "models")
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
HORIZONS = [1, 12, 24]
BURNING = {2, 3, 4}
TIME_COLS = ["hour_sin", "hour_cos", "month_sin", "month_cos", "dayofweek"]
K_LIST = [2, 3, 5]  # k=1 มีในคอลัมน์ pm25_neighbor_avg ของ dataset อยู่แล้ว


def load_tm():
    spec = importlib.util.spec_from_file_location("tm", os.path.join(_ROOT, "scripts", "train_multiprovince_models.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def haversine_km(a, b, reg):
    p1, p2 = np.radians(reg[a]["lat"]), np.radians(reg[b]["lat"])
    x = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(reg[b]["lon"] - reg[a]["lon"]) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(x))


def add_knn_avg(df, reg, k, col):
    """สูตรเดียวกับ build_raw_dataset_multiprovince.add_spatial_lag_feature: เฉลี่ย pm25 ของ k จังหวัดใกล้สุดจากทั้ง 20 ณ เวลาเดียวกัน"""
    pv = df.pivot_table(index="datetime_utc", columns="province_en", values="pm25")
    out = {}
    for p in pv.columns:
        near = sorted((q for q in pv.columns if q != p), key=lambda q: haversine_km(p, q, reg))[:k]
        out[p] = pv[near].mean(axis=1)
    long = pd.DataFrame(out).stack(future_stack=True).rename(col).reset_index().rename(columns={"level_1": "province_en"})
    return df.merge(long, on=["datetime_utc", "province_en"], how="left")


def fit_p50(kind, X, y, tm):
    if kind == "lgb":
        import lightgbm as lgb
        return lgb.LGBMRegressor(objective="quantile", alpha=0.5, random_state=42, verbose=-1, **tm.BAGGING_PARAMS_LGB, **tm.TUNED_PARAMS_LGB).fit(X, y)
    import xgboost as xgb
    return xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, tree_method="hist", random_state=42, verbosity=0,
                            **tm.BAGGING_PARAMS_XGB, **tm.TUNED_PARAMS_XGB).fit(X, y)


def main():
    tm = load_tm()
    man = json.load(open(sorted(glob.glob(os.path.join(DATA_DIR, "rebuild_clean_manifest_*.json")))[-1], encoding="utf-8"))
    print("dataset สะอาด:", man["final"])
    df = pd.read_csv(man["final"])
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    reg = {p["province_en"]: p for p in json.load(open(os.path.join(DATA_DIR, "province_station_registry.json"), encoding="utf-8"))["provinces"]}
    for k in K_LIST:
        df = add_knn_avg(df, reg, k, f"nb_k{k}")
        print(f"  คำนวณ nb_k{k} เสร็จ (NaN {df[f'nb_k{k}'].isna().mean():.2%})")
    # ตรวจ: สูตรเดียวกันต้องได้ k=1 ตรงกับคอลัมน์ pm25_neighbor_avg ของ dataset
    chk = add_knn_avg(df[["datetime_utc", "province_en", "pm25"]], reg, 1, "nb_chk")
    ok = (chk["nb_chk"].values - df["pm25_neighbor_avg"].values)
    print(f"  ตรวจสูตร k=1: ตรงกับ pm25_neighbor_avg ของ dataset {(np.abs(np.nan_to_num(ok)) < 1e-6).mean():.1%} ของแถว")

    nb1, trend, fire = "pm25_neighbor_avg", "nb_trend_12h", "fire_hotspot_count_100km_24h"
    base = [c for c in tm.FEATURE_COLS if c not in (nb1, trend, fire)]
    variants = {
        "baseline (18)": base,
        "+nb_k1": base + [nb1], "+nb_k2": base + ["nb_k2"], "+nb_k3": base + ["nb_k3"], "+nb_k5": base + ["nb_k5"],
        "+fire": base + [fire],
        "+nb_k1+trend": base + [nb1, trend],
        "+nb_k1+trend+fire (ชุดปัจจุบัน 21)": base + [nb1, trend, fire],
        "+time_cyclical": base + TIME_COLS,
    }
    cand_cols = sorted({c for v in variants.values() for c in v})
    d = df[~df["is_zero_shot"]].dropna(subset=cand_cols).reset_index(drop=True)
    train_df, calib_df, _ = tm.split_train_calib_test(d)
    print(f"แถวที่ใช้ (ทุก variant ชุดเดียวกัน): train={len(train_df)} calib={len(calib_df)}  (เลือกบน calib เท่านั้น)")

    rows = []
    for kind in ("lgb", "xgb"):
        for name, feats in variants.items():
            for h in HORIZONS:
                t = f"pm25_t+{h}"
                tr, ev = train_df.dropna(subset=[t]), calib_df.dropna(subset=[t])
                m = fit_p50(kind, tr[feats], tr[t] - tr["pm25"], tm)
                pred = m.predict(ev[feats]) + ev["pm25"].values
                y = ev[t].values
                burning = ev["datetime_utc"].dt.month.isin(BURNING).values
                hi = y > np.percentile(y, 90)
                rows.append({"model": kind, "variant": name, "horizon": h, "mae": mean_absolute_error(y, pred),
                             "mae_normal": mean_absolute_error(y[~burning], pred[~burning]),
                             "mae_burning": mean_absolute_error(y[burning], pred[burning]), "mae_gt_p90": mean_absolute_error(y[hi], pred[hi])})
            r = pd.DataFrame([x for x in rows if x["model"] == kind and x["variant"] == name])
            print(f"  [{kind}/{name}] MAE={r['mae'].mean():.3f} >p90={r['mae_gt_p90'].mean():.3f}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(MODEL_DIR, f"validate_features_clean_{TIMESTAMP}.csv"), index=False)

    pd.set_option("display.width", 220)
    for metric in ("mae", "mae_burning", "mae_gt_p90"):
        g = res.groupby(["model", "variant"])[metric].mean().unstack("model")
        for k in ("lgb", "xgb"):
            g[f"{k}_d%"] = (g[k] - g.loc["baseline (18)", k]) / g.loc["baseline (18)", k] * 100
        print(f"\n--- {metric} (delta% เทียบ baseline 18 feature; ติดลบ = ดีกว่า) ---\n{g.round(3).to_string()}")
    print("\n--- MAE รวม แยก horizon (delta% เทียบ baseline) ---")
    for kind in ("lgb", "xgb"):
        pt = res[res["model"] == kind].pivot(index="variant", columns="horizon", values="mae")
        base_row = pt.loc["baseline (18)"]
        print(f"\n{kind.upper()}\n{((pt - base_row) / base_row * 100).round(2).to_string()}")


if __name__ == "__main__":
    main()

"""
ช่องว่างที่เหลือของนโยบายสำรอง (CHANGELOG 141): check_model_sensitivity_live_degradation.py วัดแค่ MAE — ช่วงความเชื่อมั่น 90% ที่ live ใช้
(P05/P95 + q_hat ต่อ horizon เหมือน app.py) ยังครอบคลุมค่าจริงเมื่อ feature เพื่อนบ้านมาจากนโยบายสำรองไหม?
วัดบน test set เดียวกับตอนเทรน, 5 horizon ตัวแทน, ทั้ง LGB/XGB, q_hat จาก conformal_calibration_*_20260919_135655.csv (ตัวเดียวกับที่ deploy):
  baseline | avg 3 nearest (นโยบายที่ใช้เมื่ออันดับ 1 ไม่สด) | stale 24h hold | nb=mean of others
รายงาน coverage รวม / ฤดูฝุ่น (ก.พ.-เม.ย.) / >p90 และความกว้างช่วง — ไม่แก้ app.py
"""
import importlib.util
import json
import os
import pickle

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(_ROOT, "models")
TAG = "20260919_135655"
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
HORIZONS = [1, 6, 12, 18, 24]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    tm = load("tm", os.path.join(_ROOT, "scripts", "train_multiprovince_models.py"))
    e4 = load("e4", os.path.join(_ROOT, "scripts", "check_model_sensitivity_live_degradation.py"))
    df = pd.read_csv(tm.DATA_PATH)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    _, _, test_df = tm.split_train_calib_test(df[~df["is_zero_shot"]].reset_index(drop=True))
    test_df = test_df.reset_index(drop=True)
    feats = tm.FEATURE_COLS

    # ตัวแปรสมมติสถานการณ์ live (สร้างเหมือน E4)
    pv = df.pivot_table(index="datetime_utc", columns="province_en", values="pm25")
    reg = {q["province_en"]: q for q in json.load(open(os.path.join(_ROOT, "data", "province_station_registry.json"), encoding="utf-8"))["provinces"]}
    def hav(a, b):
        p1, p2 = np.radians(reg[a]["lat"]), np.radians(reg[b]["lat"])
        x = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(reg[b]["lon"] - reg[a]["lon"]) / 2) ** 2
        return 2 * 6371.0 * np.arcsin(np.sqrt(x))
    rank = {a: sorted([b for b in reg if b != a], key=lambda b: hav(a, b)) for a in reg}
    n = len(test_df)
    times = test_df["datetime_utc"]
    avg3, avg3_12 = np.full(n, np.nan), np.full(n, np.nan)
    for prov, g in test_df.groupby("province_en"):
        ix = g.index.values
        t0, t12 = times.loc[ix], times.loc[ix] - pd.Timedelta(hours=12)
        avg3[ix] = np.nanmean([pv[r].reindex(t0).values for r in rank[prov][:3]], axis=0)
        avg3_12[ix] = np.nanmean([pv[r].reindex(t12).values for r in rank[prov][:3]], axis=0)
    others_mean = (pv.sum(axis=1).reindex(times).values - np.nan_to_num(np.array([pv.reindex(times).values[i, list(pv.columns).index(p)] for i, p in enumerate(test_df["province_en"])]))) / (pv.notna().sum(axis=1).reindex(times).values - 1)
    fill = lambda a: np.where(np.isnan(a), others_mean, a)
    nb_t = test_df["pm25_neighbor_avg"].values
    nb_t12 = nb_t - test_df["nb_trend_12h"].values
    stale24 = e4.shifted(df, "pm25_neighbor_avg", 24, test_df)
    variants = {
        "baseline (เพื่อนบ้านอันดับ 1 สด)": {},
        "avg 3 nearest (นโยบายสำรอง)": {"pm25_neighbor_avg": fill(avg3), "nb_trend_12h": np.where(np.isnan(avg3_12), 0.0, fill(avg3) - avg3_12)},
        "stale 24h hold": {"pm25_neighbor_avg": np.where(np.isnan(stale24), nb_t, stale24), "nb_trend_12h": 0.0},
        "mean of others (trend=0)": {"pm25_neighbor_avg": others_mean, "nb_trend_12h": 0.0},
    }

    pk = {t: pickle.load(open(os.path.join(MODEL_DIR, f"{t}_pm25_multiprovince_residual_bagged_quantile_{TAG}.pkl"), "rb")) for t in ("lgb", "xgb")}
    qhat = {t: pd.read_csv(os.path.join(MODEL_DIR, f"conformal_calibration_{t}_{TAG}.csv")).set_index("horizon")["q_hat"].to_dict() for t in ("lgb", "xgb")}

    rows = []
    for h in HORIZONS:
        tcol = f"pm25_t+{h}"
        te = test_df.dropna(subset=[tcol])
        y = te[tcol].values
        burning = te["datetime_utc"].dt.month.isin({2, 3, 4}).values
        hi90 = y > np.percentile(y, 90)
        for kind in ("lgb", "xgb"):
            mq = pk[kind]["models"][h]
            for name, over in variants.items():
                X = te[feats].copy()
                for col, val in over.items():
                    X[col] = val[te.index] if isinstance(val, np.ndarray) else val
                cur = te["pm25"].values
                lo = mq[0.05].predict(X) + cur - qhat[kind][h]
                hi = mq[0.95].predict(X) + cur + qhat[kind][h]
                inside = (y >= np.maximum(lo, 0)) & (y <= hi)
                rows.append({"model": kind, "variant": name, "horizon": h, "cov": inside.mean() * 100,
                             "cov_burning": inside[burning].mean() * 100, "cov_gt_p90": inside[hi90].mean() * 100,
                             "width": (hi - np.maximum(lo, 0)).mean()})
        print(f"  t+{h} เสร็จ", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(MODEL_DIR, f"check_fallback_interval_coverage_{TIMESTAMP}.csv"), index=False)
    g = res.groupby(["model", "variant"])[["cov", "cov_burning", "cov_gt_p90", "width"]].mean()
    print("\n" + "=" * 96 + "\ncoverage ของช่วง 90% ที่ deploy (q_hat เดียวต่อ horizon) เฉลี่ย 5 horizon — เป้า 90%\n" + "=" * 96)
    for kind in ("lgb", "xgb"):
        print(f"\n{kind.upper()}")
        print(g.loc[kind].round(1).to_string())


if __name__ == "__main__":
    main()

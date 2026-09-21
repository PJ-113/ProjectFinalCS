"""
E4 (แผน deploy feasibility) — โมเดล _20260919_135655 ทนต่อข้อมูล live ที่ไม่สมบูรณ์แค่ไหน
จำลองสถานการณ์ live บน test set เดียวกับตอนเทรน (P50 เท่านั้น, ไม่ต้องเทรนใหม่ แค่ predict ด้วย pkl เดิม) แล้ววัด MAE ที่เสียหาย:
  - pm25_neighbor_avg เก่ากว่าจริง k=1/2/3 ชม. (สถานีเพื่อนบ้านรายงานช้า; live ค้างค่าท้ายสาย = neighbor(t-k)
    ส่วน nb_trend_12h ปรับให้สอดคล้อง = neighbor(t-k) - neighbor(t-12))
  - pm25_neighbor_avg = NaN / = pm25 ของจังหวัดตัวเอง / = ค่าเฉลี่ยของจังหวัดอื่นทั้งหมด ณ เวลาเดียวกัน (แทนเมื่อไม่มีเพื่อนบ้าน)
  - nb_trend_12h = NaN / 0
  - fire_hotspot_count x0.5/x0.8/x1.2 (จำลอง NRT != SP) และ = 0 (FIRMS ใช้ไม่ได้)
เทียบกับ naive persistence (พยากรณ์ = ค่าปัจจุบัน) ที่เป็นเพดานของ fallback ของ app.py
baseline ต้องได้ MAE ตรงกับ confirm_improvements_on_test.py (LGB 2.538 / XGB 2.542 ที่ 5 horizon)
"""
import importlib.util
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(_ROOT, "models")
TAG = "20260919_135655"
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
HORIZONS = [1, 6, 12, 18, 24]
BURNING = {2, 3, 4}


def load_tm():
    spec = importlib.util.spec_from_file_location("tm", os.path.join(_ROOT, "scripts", "train_multiprovince_models.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def shifted(df_all, col, hours, keys):
    """ค่า col ของ (province, t - hours) แบบ time-indexed จาก df_all (ทุกแถวทุกจังหวัด) จัดเรียงตาม keys"""
    lag = df_all[["province_en", "datetime_utc", col]].copy()
    lag["datetime_utc"] = lag["datetime_utc"] + pd.Timedelta(hours=hours)
    lag = lag.rename(columns={col: "_v"})
    out = keys[["province_en", "datetime_utc"]].merge(lag, on=["province_en", "datetime_utc"], how="left")
    assert len(out) == len(keys)
    return out["_v"].values


def main():
    tm = load_tm()
    df = pd.read_csv(tm.DATA_PATH)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    trained = df[~df["is_zero_shot"]].reset_index(drop=True)
    _, _, test_df = tm.split_train_calib_test(trained)
    test_df = test_df.reset_index(drop=True)
    feats = tm.FEATURE_COLS

    # ค่าเฉลี่ย pm25 ของ "จังหวัดอื่นทั้งหมด" ณ เวลาเดียวกัน (ทั้ง 20 จังหวัดใน df)
    pv = df.pivot_table(index="datetime_utc", columns="province_en", values="pm25")
    tot, cnt = pv.sum(axis=1), pv.notna().sum(axis=1)
    self_v = pv.reindex(test_df["datetime_utc"]).values
    prov_idx = {p: i for i, p in enumerate(pv.columns)}
    own = np.array([self_v[i, prov_idx[p]] for i, p in enumerate(test_df["province_en"])])
    others_mean = (tot.reindex(test_df["datetime_utc"]).values - np.nan_to_num(own)) / \
                  (cnt.reindex(test_df["datetime_utc"]).values - (~np.isnan(own)).astype(int))

    variants = {"baseline": {}}
    for k in (1, 2, 3):
        nb_t = test_df["pm25_neighbor_avg"].values
        nb_tk = shifted(df, "pm25_neighbor_avg", k, test_df)
        nb_t12 = nb_t - test_df["nb_trend_12h"].values  # = neighbor(t-12)
        variants[f"nb_stale_{k}h"] = {"pm25_neighbor_avg": nb_tk, "nb_trend_12h": nb_tk - nb_t12}
    # stale ยาว (ตรงกับที่ E3 พบ: เพื่อนบ้านค้างค่าเก่า ~23-25 ชม.): live "ค้างค่าท้ายสาย" (hold) => avg = neighbor(t-k),
    # trend = neighbor(t-k) - neighbor(t-12) เมื่อ k<12; เมื่อ k>=12 ทั้งสองจุดเป็นค่าที่ค้างเดียวกัน => trend = 0
    nb_t12_all = test_df["pm25_neighbor_avg"].values - test_df["nb_trend_12h"].values
    for k in (6, 12, 24):
        nb_tk = shifted(df, "pm25_neighbor_avg", k, test_df)
        variants[f"nb_stale_{k}h (hold)"] = {"pm25_neighbor_avg": nb_tk,
                                              "nb_trend_12h": (nb_tk - nb_t12_all) if k < 12 else 0.0}
    # นโยบายสำรองเมื่อเพื่อนบ้านอันดับ 1 ค้าง: ใช้เพื่อนบ้านอันดับ 2 / เฉลี่ยอันดับ 1-3 (ระยะทางจาก registry, ทั้ง 20 จังหวัด)
    import json
    reg = {q["province_en"]: q for q in json.load(open(os.path.join(_ROOT, "data", "province_station_registry.json"), encoding="utf-8"))["provinces"]}
    def hav(a, b):
        p1, p2 = np.radians(reg[a]["lat"]), np.radians(reg[b]["lat"])
        x = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(reg[b]["lon"] - reg[a]["lon"]) / 2) ** 2
        return 2 * 6371.0 * np.arcsin(np.sqrt(x))
    rank = {a: sorted([b for b in reg if b != a], key=lambda b: hav(a, b)) for a in reg}
    n = len(test_df)
    second, second12 = np.full(n, np.nan), np.full(n, np.nan)
    avg3, avg3_12 = np.full(n, np.nan), np.full(n, np.nan)
    times = test_df["datetime_utc"]
    for prov, g in test_df.groupby("province_en"):
        ix = g.index.values
        t0, t12 = times.loc[ix], times.loc[ix] - pd.Timedelta(hours=12)
        v = lambda name, tt: pv[name].reindex(tt).values
        second[ix], second12[ix] = v(rank[prov][1], t0), v(rank[prov][1], t12)
        avg3[ix] = np.nanmean([v(r, t0) for r in rank[prov][:3]], axis=0)
        avg3_12[ix] = np.nanmean([v(r, t12) for r in rank[prov][:3]], axis=0)
    fill = lambda a: np.where(np.isnan(a), others_mean, a)
    variants["nb=2nd nearest (trend from it)"] = {"pm25_neighbor_avg": fill(second),
                                                   "nb_trend_12h": np.where(np.isnan(second12), 0.0, fill(second) - second12)}
    variants["nb=avg 3 nearest (trend from it)"] = {"pm25_neighbor_avg": fill(avg3),
                                                     "nb_trend_12h": np.where(np.isnan(avg3_12), 0.0, fill(avg3) - avg3_12)}
    variants["nb_NaN(+trend NaN)"] = {"pm25_neighbor_avg": np.nan, "nb_trend_12h": np.nan}
    variants["nb=own pm25 (trend=0)"] = {"pm25_neighbor_avg": test_df["pm25"].values, "nb_trend_12h": 0.0}
    variants["nb=mean of others (trend=0)"] = {"pm25_neighbor_avg": others_mean, "nb_trend_12h": 0.0}
    variants["trend NaN only"] = {"nb_trend_12h": np.nan}
    variants["trend 0 only"] = {"nb_trend_12h": 0.0}
    for f in (0.5, 0.8, 1.2):
        variants[f"fire x{f}"] = {"fire_hotspot_count_100km_24h": test_df["fire_hotspot_count_100km_24h"].values * f}
    variants["fire = 0"] = {"fire_hotspot_count_100km_24h": 0.0}

    pk = {t: pickle.load(open(os.path.join(MODEL_DIR, f"{t}_pm25_multiprovince_residual_bagged_quantile_{TAG}.pkl"), "rb"))
          for t in ("lgb", "xgb")}
    assert pk["lgb"]["feature_cols"] == feats, "feature_cols ของ pkl ไม่ตรงกับ train_multiprovince_models.FEATURE_COLS"

    rows = []
    for h in HORIZONS:
        tcol = f"pm25_t+{h}"
        te = test_df.dropna(subset=[tcol])
        y = te[tcol].values
        burning = te["datetime_utc"].dt.month.isin(BURNING).values
        hi = y > np.percentile(y, 90)
        naive = te["pm25"].values
        rows.append({"model": "naive_persistence", "variant": "naive", "horizon": h,
                     "mae": mean_absolute_error(y, naive), "mae_burning": mean_absolute_error(y[burning], naive[burning]),
                     "mae_gt_p90": mean_absolute_error(y[hi], naive[hi])})
        for kind in ("lgb", "xgb"):
            model = pk[kind]["models"][h][0.5]
            for name, over in variants.items():
                X = te[feats].copy()
                idx = te.index
                for col, val in over.items():
                    X[col] = val[idx] if isinstance(val, np.ndarray) else val
                pred = model.predict(X) + te["pm25"].values
                rows.append({"model": kind, "variant": name, "horizon": h, "mae": mean_absolute_error(y, pred),
                             "mae_burning": mean_absolute_error(y[burning], pred[burning]),
                             "mae_gt_p90": mean_absolute_error(y[hi], pred[hi])})
        print(f"  horizon t+{h} เสร็จ", flush=True)

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(MODEL_DIR, f"check_model_sensitivity_live_degradation_{TIMESTAMP}.csv"), index=False)
    g = res.groupby(["model", "variant"])[["mae", "mae_burning", "mae_gt_p90"]].mean()
    print("\n" + "=" * 100 + "\nMAE เฉลี่ย 5 horizon และ delta% เทียบ baseline ของโมเดลเดียวกัน\n" + "=" * 100)
    for kind in ("lgb", "xgb"):
        b = g.loc[(kind, "baseline")]
        print(f"\n{kind.upper()}  (baseline MAE={b['mae']:.3f} burning={b['mae_burning']:.3f} >p90={b['mae_gt_p90']:.3f})")
        for name in variants:
            r = g.loc[(kind, name)]
            d = (r - b) / b * 100
            print(f"  {name:32s} MAE={r['mae']:.3f} ({d['mae']:+.2f}%)  burning ({d['mae_burning']:+.2f}%)  >p90 ({d['mae_gt_p90']:+.2f}%)")
    n = g.loc[("naive_persistence", "naive")]
    print(f"\nnaive persistence: MAE={n['mae']:.3f} burning={n['mae_burning']:.3f} >p90={n['mae_gt_p90']:.3f}")


if __name__ == "__main__":
    main()

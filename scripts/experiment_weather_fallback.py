"""
ทดลองทางสำรองเมื่อดึงสภาพอากาศจาก Open-Meteo ไม่ได้ (คำถาม: ใช้ค่าเก่าได้นานแค่ไหน / ใช้ climatology แทนได้ไหม / ยังดีกว่า Persistence ไหม)

ใช้โมเดล v3 (_20260919_203455) ตัวเดิม ไม่เทรนใหม่ predict P50 แล้วเปลี่ยนเฉพาะ 7 feature สภาพอากาศ
(temp, humidity, wind_speed, precipitation, wind_dir_sin, wind_dir_cos, pressure) ตามแต่ละสถานการณ์ แล้ววัด MAE ที่เสียไป
  - stale_k (k=1,2,3,6,12,24): ค่าสภาพอากาศเก่ากว่าจริง k ชม. ของจังหวัดเดียวกัน (= ใช้ค่าที่ cache ไว้ตอนดึงสำเร็จครั้งสุดท้าย)
  - clim_month_hour: ค่าเฉลี่ย (จังหวัด x เดือน x ชั่วโมง UTC) คำนวณจากชุด train เท่านั้น (ไม่รั่ว calib/test)
    ตกไปใช้ (จังหวัด x เดือน) แล้ว (จังหวัด) ถ้าช่องนั้นว่าง
  - แถวที่ใช้เทียบ = แถวที่ทุกสถานการณ์คำนวณได้ (ชุดเดียวกัน ไม่มี selection ต่างกันระหว่างสถานการณ์)

เกณฑ์ตัดสิน (ตั้งไว้ "ก่อน" รันครั้งแรก; ตัดสินจาก calib split ยืนยันบน test split ครั้งเดียว; ค่าเฉลี่ย 5 horizon [1,6,12,18,24]
ของแต่ละโมเดล ทั้ง LGB และ XGB ต้องผ่านพร้อมกัน):
  A. ใช้ค่าเก่า k ชม. ได้ (จึงตั้ง WEATHER_CACHE_MAX_AGE = k ที่มากที่สุดที่ผ่าน) ถ้า MAE รวมเสียไม่เกิน +1.0%,
     ช่วงฤดูฝุ่น (เดือน 2-4) ไม่เกิน +1.5%, และ >p90 ไม่เกิน +1.5% เทียบ baseline (สภาพอากาศจริง) ของโมเดลเดียวกัน
  B. ใช้ climatology เป็นทางสุดท้ายได้ ถ้า MAE รวมของ climatology "ต่ำกว่า Persistence" ทุก horizon ทั้ง 5 ตัว (รวมทั้งช่วงฤดูฝุ่น)
     — ตัวเทียบจริงคือ naive_fallback ที่ระบบจะตกไปใช้ ไม่ใช่ baseline
  ยืนยันบน test: k ที่เลือกจาก calib ต้องยังผ่านเกณฑ์ A บน test ด้วย ถ้าไม่ผ่านให้ลด k ลงจนผ่าน (รายงานตามจริง ไม่ปรับเกณฑ์)
รัน: python scripts/experiment_weather_fallback.py
"""
import importlib.util
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(_ROOT, "models")
TAG = "20260919_203455"
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
HORIZONS = [1, 6, 12, 18, 24]
BURNING = {2, 3, 4}
WEATHER = ["temp", "humidity", "wind_speed", "precipitation", "wind_dir_sin", "wind_dir_cos", "pressure"]
STALE_K = [1, 2, 3, 6, 12, 24]
TOL = {"mae": 1.0, "mae_burning": 1.5, "mae_gt_p90": 1.5}  # % (เกณฑ์ A)


def load_tm():
    spec = importlib.util.spec_from_file_location("tm", os.path.join(_ROOT, "scripts", "train_multiprovince_models.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def stale_weather(df_all, hours, keys):
    """สภาพอากาศของ (จังหวัด, t-hours) แบบ time-indexed คืน DataFrame เรียงตาม keys"""
    lag = df_all[["province_en", "datetime_utc"] + WEATHER].copy()
    lag["datetime_utc"] = lag["datetime_utc"] + pd.Timedelta(hours=hours)
    out = keys[["province_en", "datetime_utc"]].merge(lag, on=["province_en", "datetime_utc"], how="left")
    assert len(out) == len(keys)
    return out[WEATHER].reset_index(drop=True)


def climatology(train_df, keys):
    """ค่าเฉลี่ยจาก train ตาม (จังหวัด, เดือน, ชั่วโมง UTC) -> (จังหวัด, เดือน) -> (จังหวัด)"""
    t = train_df[["province_en", "datetime_utc"] + WEATHER].copy()
    t["month"], t["hour"] = t["datetime_utc"].dt.month, t["datetime_utc"].dt.hour
    k = keys[["province_en", "datetime_utc"]].copy()
    k["month"], k["hour"] = k["datetime_utc"].dt.month, k["datetime_utc"].dt.hour
    out = k.merge(t.groupby(["province_en", "month", "hour"])[WEATHER].mean().reset_index(), on=["province_en", "month", "hour"], how="left")
    m2 = k.merge(t.groupby(["province_en", "month"])[WEATHER].mean().reset_index(), on=["province_en", "month"], how="left")
    m3 = k.merge(t.groupby("province_en")[WEATHER].mean().reset_index(), on="province_en", how="left")
    for c in WEATHER:
        out[c] = out[c].fillna(m2[c]).fillna(m3[c])
    return out[WEATHER].reset_index(drop=True)


def evaluate(split_name, part, df_all, train_df, pk, feats):
    part = part.reset_index(drop=True)
    variants = {"baseline": part[WEATHER].reset_index(drop=True)}
    for k in STALE_K:
        variants[f"stale_{k}h"] = stale_weather(df_all, k, part)
    variants["clim_month_hour"] = climatology(train_df, part)
    ok = np.ones(len(part), dtype=bool)
    for v in variants.values():
        ok &= v.notna().all(axis=1).values
    part = part[ok].reset_index(drop=True)
    variants = {n: v[ok].reset_index(drop=True) for n, v in variants.items()}
    print(f"[{split_name}] แถวที่ใช้เทียบ (ทุกสถานการณ์คำนวณได้) = {len(part)}", flush=True)

    rows = []
    for h in HORIZONS:
        tcol = f"pm25_t+{h}"
        m = part[tcol].notna().values
        te = part[m].reset_index(drop=True)
        y = te[tcol].values
        burning = te["datetime_utc"].dt.month.isin(BURNING).values
        hi = y > np.percentile(y, 90)
        naive = te["pm25"].values
        rows.append({"split": split_name, "model": "naive_persistence", "variant": "naive", "horizon": h,
                     "mae": mean_absolute_error(y, naive), "mae_burning": mean_absolute_error(y[burning], naive[burning]),
                     "mae_gt_p90": mean_absolute_error(y[hi], naive[hi])})
        for kind in ("lgb", "xgb"):
            model = pk[kind]["models"][h][0.5]
            for name, w in variants.items():
                X = te[feats].copy()
                for c in WEATHER:
                    X[c] = w[c].values[m]
                pred = model.predict(X) + te["pm25"].values
                rows.append({"split": split_name, "model": kind, "variant": name, "horizon": h,
                             "mae": mean_absolute_error(y, pred), "mae_burning": mean_absolute_error(y[burning], pred[burning]),
                             "mae_gt_p90": mean_absolute_error(y[hi], pred[hi])})
        print(f"  [{split_name}] horizon t+{h} เสร็จ", flush=True)
    return pd.DataFrame(rows)


def summarize(res, split_name):
    r = res[res["split"] == split_name]
    g = r.groupby(["model", "variant"])[["mae", "mae_burning", "mae_gt_p90"]].mean()
    print(f"\n{'=' * 100}\n[{split_name}] MAE เฉลี่ย 5 horizon และ delta% เทียบ baseline ของโมเดลเดียวกัน\n{'=' * 100}")
    verdict = {}
    for kind in ("lgb", "xgb"):
        b = g.loc[(kind, "baseline")]
        print(f"\n{kind.upper()}  baseline MAE={b['mae']:.3f} burning={b['mae_burning']:.3f} >p90={b['mae_gt_p90']:.3f}")
        for name in [f"stale_{k}h" for k in STALE_K] + ["clim_month_hour"]:
            x = g.loc[(kind, name)]
            d = (x - b) / b * 100
            passed = all(d[c] <= TOL[c] for c in TOL)
            verdict[(kind, name)] = passed
            print(f"  {name:18s} MAE={x['mae']:.3f} ({d['mae']:+.2f}%)  burning ({d['mae_burning']:+.2f}%)  >p90 ({d['mae_gt_p90']:+.2f}%)  "
                  f"{'ผ่าน A' if passed else 'ไม่ผ่าน A'}")
    n = g.loc[("naive_persistence", "naive")]
    print(f"\nnaive persistence (เฉลี่ย 5 horizon): MAE={n['mae']:.3f} burning={n['mae_burning']:.3f} >p90={n['mae_gt_p90']:.3f}")
    # เกณฑ์ B: climatology ต่ำกว่า Persistence ทุก horizon (รวม + ฤดูฝุ่น)
    beats = {}
    for kind in ("lgb", "xgb"):
        c = r[(r["model"] == kind) & (r["variant"] == "clim_month_hour")].set_index("horizon")
        nv = r[r["model"] == "naive_persistence"].set_index("horizon")
        beats[kind] = bool(((c["mae"] < nv["mae"]).all()) and ((c["mae_burning"] < nv["mae_burning"]).all()))
        per_h = " ".join(f"t+{h}:{(1 - c.loc[h, 'mae'] / nv.loc[h, 'mae']) * 100:+.1f}%" for h in HORIZONS)
        print(f"  เกณฑ์ B {kind}: clim ดีกว่า Persistence (MAE ที่ลดลง) {per_h} -> {'ผ่าน' if beats[kind] else 'ไม่ผ่าน'}")
    return verdict, beats


def main():
    tm = load_tm()
    df = pd.read_csv(tm.DATA_PATH)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    trained = df[~df["is_zero_shot"]].reset_index(drop=True)
    train_df, calib_df, test_df = tm.split_train_calib_test(trained)
    feats = tm.FEATURE_COLS
    for c in WEATHER:
        assert c in feats, c
    pk = {t: pickle.load(open(os.path.join(MODEL_DIR, f"{t}_pm25_multiprovince_residual_bagged_quantile_{TAG}.pkl"), "rb"))
          for t in ("lgb", "xgb")}
    assert pk["lgb"]["feature_cols"] == feats

    res = pd.concat([evaluate("calib", calib_df, df, train_df, pk, feats),
                     evaluate("test", test_df, df, train_df, pk, feats)], ignore_index=True)
    res.to_csv(os.path.join(MODEL_DIR, f"experiment_weather_fallback_{TIMESTAMP}.csv"), index=False)

    v_cal, b_cal = summarize(res, "calib")
    v_te, b_te = summarize(res, "test")

    print("\n" + "=" * 100 + "\nสรุปตามเกณฑ์ที่ตั้งไว้ก่อน\n" + "=" * 100)
    chosen = None
    for k in sorted(STALE_K, reverse=True):
        if all(v_cal[(m, f"stale_{k}h")] for m in ("lgb", "xgb")):
            chosen = k
            break
    print(f"ค่าเก่าที่ยอมรับได้มากที่สุดตาม calib (A): k = {chosen}")
    if chosen is not None:
        final = None
        for k in [x for x in sorted(STALE_K, reverse=True) if x <= chosen]:
            if all(v_te[(m, f"stale_{k}h")] for m in ("lgb", "xgb")):
                final = k
                break
        print(f"ยืนยันบน test: k = {final}  (WEATHER_CACHE_MAX_AGE ที่แนะนำ = {final} ชม.)")
    print(f"climatology เป็นทางสุดท้าย (B) calib: {b_cal}  test: {b_te}")
    print(f"\nผลดิบ: models/experiment_weather_fallback_{TIMESTAMP}.csv")


if __name__ == "__main__":
    main()

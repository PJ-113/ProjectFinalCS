"""
วัดว่าเพื่อนบ้าน "ไม่สด" ไกลแค่ไหนยัง MAE ดีกว่า naive persistence (บนโมเดล v3 + ข้อมูลสะอาด)
เพื่อหาค่า NEIGHBOR_STALE_MAX_HOURS สำหรับ tier ใหม่ใน compute_neighbor_features()

เกณฑ์ตัดสิน (ตั้งก่อนรัน): ค่าความไม่สด (ชม.) มากที่สุดที่ยัง MAE ดีกว่า naive_persistence อย่างน้อย 10%
ทั้ง LGB และ XGB = NEIGHBOR_STALE_MAX_HOURS ที่แนะนำ
"""
import importlib.util
import os
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(_ROOT, "models")
TAG = "20260919_203455"  # v3
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
HORIZONS = [1, 6, 12, 18, 24]
BURNING = {2, 3, 4}
PASS_MARGIN_PCT = 10.0  # ต้องดีกว่า naive_persistence อย่างน้อยเท่านี้ %


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
    print(f"dataset: {tm.DATA_PATH}")
    df = pd.read_csv(tm.DATA_PATH)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    trained = df[~df["is_zero_shot"]].reset_index(drop=True)
    _, _, test_df = tm.split_train_calib_test(trained)
    test_df = test_df.reset_index(drop=True)
    feats = tm.FEATURE_COLS

    # v3 ไม่ใช้ pm25_neighbor_avg เป็น feature โดยตรง มีแค่ nb_trend_12h ที่กระทบ live ถ้าเพื่อนบ้านไม่สด —
    # เพื่อนบ้านไม่สด k ชม. = เห็นข้อมูลล่าสุดแค่ถึง t-k จึงคำนวณ trend ได้แค่ nb_trend_12h ที่ประเมิน ณ เวลา t-k
    variants = {"baseline": {}}
    for k in (1, 2, 3, 4, 5, 6, 12, 24, 48):
        variants[f"nb_stale_{k}h"] = {"nb_trend_12h": shifted(df, "nb_trend_12h", k, test_df)}
    variants["nb_NaN(trend NaN)"] = {"nb_trend_12h": np.nan}

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
    res.to_csv(os.path.join(MODEL_DIR, f"check_neighbor_staleness_sensitivity_clean_{TIMESTAMP}.csv"), index=False)
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

    print("\n" + "=" * 100 + f"\nเกณฑ์ตัดสิน: ดีกว่า naive_persistence อย่างน้อย {PASS_MARGIN_PCT}% ทั้ง LGB และ XGB\n" + "=" * 100)
    stale_variants = [v for v in variants if v.startswith("nb_stale_")]

    def hours_of(name):
        core = name.replace("nb_stale_", "").replace(" (hold)", "").rstrip("h")
        return float(core)

    passed_hours = []
    for name in sorted(stale_variants, key=hours_of):
        h_val = hours_of(name)
        ok_both = True
        for kind in ("lgb", "xgb"):
            r, nv = g.loc[(kind, name)], n
            improve_vs_persistence = (nv["mae"] - r["mae"]) / nv["mae"] * 100
            ok = improve_vs_persistence >= PASS_MARGIN_PCT
            ok_both = ok_both and ok
            print(f"  {name:22s} {kind.upper()}: MAE={r['mae']:.3f} ดีกว่า persistence {improve_vs_persistence:+.2f}% "
                  f"(เกณฑ์ >= {PASS_MARGIN_PCT}%) -> {'ผ่าน' if ok else 'ไม่ผ่าน'}")
        passed_hours.append((h_val, ok_both))

    best_hours = 0.0
    for h_val, ok_both in passed_hours:  # เรียงจากน้อยไปมาก หาค่ามากสุดที่ผ่านต่อเนื่อง
        if ok_both:
            best_hours = h_val
        else:
            break
    print(f"\n=> NEIGHBOR_STALE_MAX_HOURS ที่แนะนำ = {best_hours:.0f} ชม. (ค่ามากสุดที่ยังผ่านเกณฑ์ทั้ง LGB/XGB ต่อเนื่องจาก 1 ชม.)")


if __name__ == "__main__":
    main()

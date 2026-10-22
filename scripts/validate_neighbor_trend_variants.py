"""
ทดสอบ nb_trend หลายหน้าต่าง (1h/3h/6h/12h) และ combo กับ precip_sum_72h บน calib split เท่านั้น
วัด MAE/MAE ฤดูฝุ่น/MAE >p90 เทียบ baseline ที่ horizon [1,12,24]
"""
import glob
import importlib.util
import os

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_ROOT, "data")
MODEL_DIR = os.path.join(_ROOT, "models")
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")


def load_module(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, "scripts", fname))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def add_trend(df, hours):
    lag = df[["province_en", "datetime_utc", "pm25_neighbor_avg"]].copy()
    lag["datetime_utc"] = lag["datetime_utc"] + pd.Timedelta(hours=hours)
    lag = lag.rename(columns={"pm25_neighbor_avg": "_lag"})
    n0 = len(df)
    df = df.merge(lag, on=["province_en", "datetime_utc"], how="left")
    assert len(df) == n0, "merge ทำให้แถวซ้ำ"
    df[f"nb_trend_{hours}h"] = df["pm25_neighbor_avg"] - df["_lag"]
    return df.drop(columns=["_lag"])


def main():
    train_mod = load_module("train_multiprovince_models", "train_multiprovince_models.py")
    val = load_module("validate_feature_refinements", "validate_feature_refinements.py")
    path = sorted(glob.glob(os.path.join(DATA_DIR, "multiprovince_pm25_features_refined_candidates_*.csv")))[-1]
    print(f"โหลด {path}")
    df = pd.read_csv(path)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    for hrs in (1, 6, 12):
        df = add_trend(df, hrs)
    # ตรวจ trend 3h เดิมกับ recompute (ธรรมเนียม np.allclose)
    chk = add_trend(df[["province_en", "datetime_utc", "pm25_neighbor_avg"]].copy(), 3)["nb_trend_3h"].values
    assert np.allclose(chk, df["nb_trend_3h"].values, equal_nan=True), "nb_trend_3h ไม่ตรงกับ recompute"
    print("nb_trend_3h ตรงกับ recompute (allclose)")

    trained = df[~df["is_zero_shot"]].reset_index(drop=True)
    train_df, calib_df, _ = train_mod.split_train_calib_test(trained)
    base = train_mod.FEATURE_COLS
    variants = {
        "baseline": base,
        "nb_trend_1h": base + ["nb_trend_1h"],
        "nb_trend_3h": base + ["nb_trend_3h"],
        "nb_trend_6h": base + ["nb_trend_6h"],
        "nb_trend_12h": base + ["nb_trend_12h"],
        "trend_3h+6h": base + ["nb_trend_3h", "nb_trend_6h"],
        "trend_3h+precip72": base + ["nb_trend_3h", "precip_sum_72h"],
    }
    rows = []
    for kind in ("lgb", "xgb"):
        print(f"\n=== {kind.upper()} ===")
        for name, feats in variants.items():
            rows += val.eval_variant(train_df, calib_df, feats, kind, name, train_mod)
    res = pd.DataFrame(rows)
    out_path = os.path.join(MODEL_DIR, f"validate_neighbor_trend_variants_{TIMESTAMP}.csv")
    res.to_csv(out_path, index=False)
    print(f"\nบันทึกผลที่: {out_path}")
    print("\n" + "=" * 90 + "\nสรุป delta% เทียบ baseline\n" + "=" * 90)
    for metric in ("mae", "mae_burning", "mae_gt_p90"):
        t = res.groupby(["model", "variant"])[metric].mean().unstack("model")
        for k in ("lgb", "xgb"):
            t[f"{k}_d%"] = (t[k] - t.loc["baseline", k]) / t.loc["baseline", k] * 100
        print(f"\n--- {metric} ---")
        print(t.round(3).to_string())


if __name__ == "__main__":
    main()

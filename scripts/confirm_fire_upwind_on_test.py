"""
ยืนยันบน test split (ครั้งเดียว) สำหรับ feature ไฟทิศต้นลม (fire_upwind_250km): เทียบ base20
(โมเดล v3 ปัจจุบัน) กับ base20 + fire_upwind_250km (P50, 5 horizon ตัวแทน, LGB+XGB)
วัด MAE รวม/ปกติ/ฤดูฝุ่น ก.พ.-เม.ย./>p90 เกณฑ์ผ่าน: ทั้ง LGB และ XGB ต้องไม่แย่ลงทั้ง MAE รวมและ MAE >p90
"""
import glob
import importlib.util
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR, MODEL_DIR = os.path.join(_ROOT, "data"), os.path.join(_ROOT, "models")
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
HORIZONS = [1, 6, 12, 18, 24]
BURNING = {2, 3, 4}
CAND = "fire_upwind_250km"


def fit_p50(kind, X, y, tm):
    if kind == "lgb":
        import lightgbm as lgb
        return lgb.LGBMRegressor(objective="quantile", alpha=0.5, random_state=42, verbose=-1, **tm.BAGGING_PARAMS_LGB, **tm.TUNED_PARAMS_LGB).fit(X, y)
    import xgboost as xgb
    return xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, tree_method="hist", random_state=42, verbosity=0,
                            **tm.BAGGING_PARAMS_XGB, **tm.TUNED_PARAMS_XGB).fit(X, y)


def main():
    spec = importlib.util.spec_from_file_location("tm", os.path.join(_ROOT, "scripts", "train_multiprovince_models.py"))
    tm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tm)
    df = pd.read_csv(tm.DATA_PATH)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    cand_path = sorted(glob.glob(os.path.join(DATA_DIR, "fire_wide_candidate_features_*.csv")))[-1]
    cf = pd.read_csv(cand_path, usecols=["province_en", "datetime_utc", CAND])
    cf["datetime_utc"] = pd.to_datetime(cf["datetime_utc"])
    df = df.merge(cf, on=["province_en", "datetime_utc"], how="left")
    assert df[CAND].notna().all(), "feature ตัวเลือกมีค่าว่างหลัง merge"
    print("dataset:", tm.DATA_PATH, "\nfeature ตัวเลือก:", cand_path)
    df = df[~df["is_zero_shot"]].reset_index(drop=True)
    train_df, _, test_df = tm.split_train_calib_test(df)
    print(f"train={len(train_df)} test={len(test_df)}  test ช่วง {test_df['datetime_utc'].min()} .. {test_df['datetime_utc'].max()}")
    base = list(tm.FEATURE_COLS)
    variants = {"base20": base, "base20+upwind": base + [CAND]}

    rows = []
    for kind in ("lgb", "xgb"):
        for name, feats in variants.items():
            for h in HORIZONS:
                t = f"pm25_t+{h}"
                tr, te = train_df.dropna(subset=[t]), test_df.dropna(subset=[t])
                m = fit_p50(kind, tr[feats], tr[t] - tr["pm25"], tm)
                pred = m.predict(te[feats]) + te["pm25"].values
                y = te[t].values
                burning = te["datetime_utc"].dt.month.isin(BURNING).values
                hi = y > np.percentile(y, 90)
                rows.append({"model": kind, "variant": name, "horizon": h, "mae": mean_absolute_error(y, pred),
                             "mae_normal": mean_absolute_error(y[~burning], pred[~burning]),
                             "mae_burning": mean_absolute_error(y[burning], pred[burning]) if burning.any() else np.nan,
                             "mae_gt_p90": mean_absolute_error(y[hi], pred[hi]), "n": len(te)})
            r = pd.DataFrame([x for x in rows if x["model"] == kind and x["variant"] == name])
            print(f"  [{kind}/{name}] MAE={r['mae'].mean():.3f} ปกติ={r['mae_normal'].mean():.3f} "
                  f"ฤดูฝุ่น={r['mae_burning'].mean():.3f} >p90={r['mae_gt_p90'].mean():.3f}", flush=True)
    res = pd.DataFrame(rows)
    path = os.path.join(MODEL_DIR, f"confirm_fire_upwind_on_test_{TIMESTAMP}.csv")
    res.to_csv(path, index=False)

    print("\n" + "=" * 100 + "\nการตัดสินตามเกณฑ์ที่ตั้งไว้ก่อนรัน (delta% เทียบ base20; ติดลบ = ดีกว่า)\n" + "=" * 100)
    oks = []
    for kind in ("lgb", "xgb"):
        s = res[res["model"] == kind].groupby("variant")[["mae", "mae_normal", "mae_burning", "mae_gt_p90"]].mean()
        d = (s.loc["base20+upwind"] - s.loc["base20"]) / s.loc["base20"] * 100
        print(f"  {kind.upper()}: รวม {d['mae']:+.2f}%  ปกติ {d['mae_normal']:+.2f}%  ฤดูฝุ่น {d['mae_burning']:+.2f}%  >p90 {d['mae_gt_p90']:+.2f}%")
        oks.append(d["mae"] <= 0 and d["mae_gt_p90"] <= 0)
    print(f"\nผล: {'ผ่าน' if all(oks) else 'ไม่ผ่าน'} (LGB={oks[0]}, XGB={oks[1]})\nผลดิบ: {path}")
    per_h = res.pivot_table(index=["model", "horizon"], columns="variant", values="mae_gt_p90")
    print("\nMAE >p90 ต่อ horizon:\n", per_h.round(3).to_string())


if __name__ == "__main__":
    main()

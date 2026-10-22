"""
ยืนยันบน test split (ครั้งเดียว) เทียบ 3 variant (P50, 5 horizon ตัวแทน, LGB+XGB)
วัด MAE รวม/ปกติ/ฤดูฝุ่น ก.พ.-เม.ย./>p90:
  18-default : 18 feature (ไม่มี neighbor/fire/nb_trend) + hyperparameter default
  18-tuned   : 18 feature + hyperparameter จูนแล้ว
  21-tuned   : 21 feature + hyperparameter จูนแล้ว (โมเดล v3)
เกณฑ์ผ่าน: 21-tuned ต้องไม่แย่กว่า 18-tuned ที่ MAE รวม ทั้ง LGB และ XGB
"""
import importlib.util
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(_ROOT, "models")
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
HORIZONS = [1, 6, 12, 18, 24]
BURNING = {2, 3, 4}


def fit_p50(kind, X, y, params, tm):
    if kind == "lgb":
        import lightgbm as lgb
        return lgb.LGBMRegressor(objective="quantile", alpha=0.5, random_state=42, verbose=-1, **tm.BAGGING_PARAMS_LGB, **params).fit(X, y)
    import xgboost as xgb
    return xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, tree_method="hist", random_state=42, verbosity=0,
                            **tm.BAGGING_PARAMS_XGB, **params).fit(X, y)


def main():
    spec = importlib.util.spec_from_file_location("tm", os.path.join(_ROOT, "scripts", "train_multiprovince_models.py"))
    tm = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tm)
    print("dataset:", tm.DATA_PATH)
    df = pd.read_csv(tm.DATA_PATH)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    df = df[~df["is_zero_shot"]].reset_index(drop=True)
    train_df, _, test_df = tm.split_train_calib_test(df)
    print(f"train={len(train_df)} test={len(test_df)}")
    nb1, trend, fire = "pm25_neighbor_avg", "nb_trend_12h", "fire_hotspot_count_100km_24h"
    f21 = tm.FEATURE_COLS
    f18 = [c for c in f21 if c not in (nb1, trend, fire)]
    tuned = {"lgb": tm.TUNED_PARAMS_LGB, "xgb": tm.TUNED_PARAMS_XGB}
    default = {"lgb": {"n_estimators": 300}, "xgb": {"n_estimators": 300}}
    variants = {"18-default": (f18, default), "18-tuned": (f18, tuned), "21-tuned": (f21, tuned)}

    rows = []
    for kind in ("lgb", "xgb"):
        for name, (feats, params) in variants.items():
            for h in HORIZONS:
                t = f"pm25_t+{h}"
                tr, te = train_df.dropna(subset=[t]), test_df.dropna(subset=[t])
                m = fit_p50(kind, tr[feats], tr[t] - tr["pm25"], params[kind], tm)
                pred = m.predict(te[feats]) + te["pm25"].values
                y = te[t].values
                burning = te["datetime_utc"].dt.month.isin(BURNING).values
                hi = y > np.percentile(y, 90)
                rows.append({"model": kind, "variant": name, "horizon": h, "mae": mean_absolute_error(y, pred),
                             "mae_normal": mean_absolute_error(y[~burning], pred[~burning]),
                             "mae_burning": mean_absolute_error(y[burning], pred[burning]), "mae_gt_p90": mean_absolute_error(y[hi], pred[hi])})
            print(f"  [{kind}/{name}] เสร็จ", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(MODEL_DIR, f"confirm_v3_on_test_{TIMESTAMP}.csv"), index=False)

    print("\n" + "=" * 100 + "\nTEST SET: MAE เฉลี่ย 5 horizon (delta% เทียบ 18-tuned)\n" + "=" * 100)
    verdict = {}
    for kind in ("lgb", "xgb"):
        g = res[res["model"] == kind].groupby("variant")[["mae", "mae_normal", "mae_burning", "mae_gt_p90"]].mean()
        print(f"\n{kind.upper()}")
        for v in ("18-default", "18-tuned", "21-tuned"):
            d = (g.loc[v] - g.loc["18-tuned"]) / g.loc["18-tuned"] * 100
            print(f"  {v:11s} MAE={g.loc[v, 'mae']:.3f} ({d['mae']:+.2f}%)  ปกติ={g.loc[v, 'mae_normal']:.3f} ({d['mae_normal']:+.2f}%)  "
                  f"ฤดูฝุ่น={g.loc[v, 'mae_burning']:.3f} ({d['mae_burning']:+.2f}%)  >p90={g.loc[v, 'mae_gt_p90']:.3f} ({d['mae_gt_p90']:+.2f}%)")
        verdict[kind] = (g.loc["21-tuned", "mae"] - g.loc["18-tuned", "mae"]) / g.loc["18-tuned", "mae"] * 100
    print("\nต่อ horizon (MAE):")
    for kind in ("lgb", "xgb"):
        p = res[res["model"] == kind].pivot(index="horizon", columns="variant", values="mae")
        p["21 vs 18-tuned %"] = (p["21-tuned"] - p["18-tuned"]) / p["18-tuned"] * 100
        print(f"\n{kind.upper()}\n{p.round(3).to_string()}")
    ok = all(v <= 0.0 for v in verdict.values())
    print("\n" + "=" * 100)
    print(f"ผลตามเกณฑ์ที่ตั้งไว้ก่อน (21-tuned ต้องไม่แย่กว่า 18-tuned ที่ MAE รวมทั้ง 2 โมเดล): {'ผ่าน' if ok else 'ไม่ผ่าน'} "
          f"(LGB {verdict['lgb']:+.2f}%, XGB {verdict['xgb']:+.2f}%)")


if __name__ == "__main__":
    main()

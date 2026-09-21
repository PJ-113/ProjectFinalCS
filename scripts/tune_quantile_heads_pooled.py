"""
ทดลอง 2 ไอเดียค้างจาก entry 134/136 เพื่อแก้ coverage <90% (โดยเฉพาะ >p90/ฤดูฝุ่น) ของโมเดล _20260919_135655:
  (1) sample weight เฉพาะหัว P95 (แถวที่ target>37.5 ถ่วง w=2/3/5) — P50 ไม่แตะ (entry 134 พบว่าถ่วง P50 แล้ว MAE รวมแย่ลง)
  (2) hyperparameter แยกสำหรับหัว P05/P95 (โมเดลปัจจุบันใช้ค่าที่จูนจาก P50 ทั้งสามหัว) — random search แนว regularize มากขึ้น
วิธีวัด (ไม่แตะ test): เทรนบน train, แบ่ง calib เป็น 2 ท่อนตามเวลาต่อจังหวัด: calibA ใช้คำนวณ q_hat (conformal, single q_hat),
calibB วัด coverage/ความกว้างช่วง — วัด coverage รวม / >p90 / ฤดูฝุ่น (ก.พ.-เม.ย.) + ความกว้างช่วงเฉลี่ย (ต้องไม่กว้างขึ้นเกินจำเป็น)
horizon [1,12,24]; base = ค่าปัจจุบัน (TUNED_PARAMS ของ train_multiprovince_models.py)
"""
import importlib.util
import os

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(_ROOT, "models")
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
HORIZONS = [1, 12, 24]
BURNING = {2, 3, 4}
HIGH = 37.5
N_RANDOM = 6
SEED = 77

SPACES = {
    "lgb": {"learning_rate": [0.03, 0.05], "num_leaves": [15, 31], "min_child_samples": [100, 300],
            "reg_lambda": [10.0, 30.0], "n_estimators": [300, 600]},
    "xgb": {"learning_rate": [0.03, 0.05], "max_depth": [4, 6], "min_child_weight": [20, 50],
            "reg_lambda": [10.0, 30.0, 100.0], "n_estimators": [400, 800]},
}


def load_tm():
    spec = importlib.util.spec_from_file_location("tm", os.path.join(_ROOT, "scripts", "train_multiprovince_models.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fit_q(X, y, kind, alpha, params, weights, tm):
    if kind == "lgb":
        import lightgbm as lgb
        return lgb.LGBMRegressor(objective="quantile", alpha=alpha, random_state=42, verbose=-1,
                                  **tm.BAGGING_PARAMS_LGB, **params).fit(X, y, sample_weight=weights)
    import xgboost as xgb
    return xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=alpha, tree_method="hist",
                             random_state=42, verbosity=0, **tm.BAGGING_PARAMS_XGB, **params).fit(X, y, sample_weight=weights)


def split_calib(calib):
    a, b = [], []
    for _, g in calib.groupby("province_en", sort=False):
        g = g.sort_values("datetime_utc")
        k = len(g) // 2
        a.append(g.iloc[:k])
        b.append(g.iloc[k:])
    return pd.concat(a, ignore_index=True), pd.concat(b, ignore_index=True)


def main():
    tm = load_tm()
    df = pd.read_csv(tm.DATA_PATH)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    df = df[~df["is_zero_shot"]].reset_index(drop=True)
    train_df, calib_df, _ = tm.split_train_calib_test(df)
    calA, calB = split_calib(calib_df)
    feats = tm.FEATURE_COLS
    print(f"train={len(train_df)} calibA={len(calA)} calibB={len(calB)} (ไม่แตะ test)")

    rng = np.random.default_rng(SEED)
    base = {"lgb": tm.TUNED_PARAMS_LGB, "xgb": tm.TUNED_PARAMS_XGB}
    rows = []
    for kind in ("lgb", "xgb"):
        variants = [("base", base[kind], 1.0), ("wP95=2", base[kind], 2.0), ("wP95=3", base[kind], 3.0),
                    ("wP95=5", base[kind], 5.0)]
        seen = set()
        while len(seen) < N_RANDOM:
            cfg = {k: v[rng.integers(len(v))] for k, v in SPACES[kind].items()}
            key = tuple(sorted(cfg.items()))
            if key not in seen:
                seen.add(key)
                variants.append((f"reg{len(seen)}", cfg, 1.0))
        for name, params, w_hi in variants:
            for h in HORIZONS:
                tcol = f"pm25_t+{h}"
                tr = train_df.dropna(subset=[tcol])
                y_res = tr[tcol] - tr["pm25"]
                w95 = np.where(tr[tcol].values > HIGH, w_hi, 1.0) if w_hi != 1.0 else None
                m05 = fit_q(tr[feats], y_res, kind, 0.05, params, None, tm)
                m95 = fit_q(tr[feats], y_res, kind, 0.95, params, w95, tm)
                out = {}
                for tag, d in (("A", calA), ("B", calB)):
                    d = d.dropna(subset=[tcol])
                    out[tag] = (d, m05.predict(d[feats]) + d["pm25"].values, m95.predict(d[feats]) + d["pm25"].values,
                                d[tcol].values)
                dA, loA, hiA, yA = out["A"]
                q_hat = tm.conformalize(loA, hiA, yA)
                dB, loB, hiB, yB = out["B"]
                lo, hi = loB - q_hat, hiB + q_hat
                inside = (yB >= lo) & (yB <= hi)
                burning = pd.to_datetime(dB["datetime_utc"]).dt.month.isin(BURNING).values
                gt90 = yB > np.percentile(yB, 90)
                rows.append({"model": kind, "variant": name, "horizon": h, "q_hat": q_hat,
                             "cov_all": inside.mean() * 100, "cov_burning": inside[burning].mean() * 100,
                             "cov_gt_p90": inside[gt90].mean() * 100, "width": (hi - lo).mean(), **{f"p_{k}": v for k, v in params.items()}, "w_hi": w_hi})
            r = pd.DataFrame([x for x in rows if x["model"] == kind and x["variant"] == name])
            print(f"  [{kind}/{name}] cov={r['cov_all'].mean():.1f} burning={r['cov_burning'].mean():.1f} "
                  f">p90={r['cov_gt_p90'].mean():.1f} width={r['width'].mean():.2f} q_hat={r['q_hat'].mean():.2f}", flush=True)

    res = pd.DataFrame(rows)
    out_path = os.path.join(MODEL_DIR, f"tune_quantile_heads_pooled_{TIMESTAMP}.csv")
    res.to_csv(out_path, index=False)
    print(f"\nบันทึกผลที่: {out_path}")
    print("\n" + "=" * 90 + "\nสรุปเฉลี่ยข้าม horizon (calibB, single q_hat จาก calibA; เป้า coverage 90%)\n" + "=" * 90)
    for kind in ("lgb", "xgb"):
        g = res[res["model"] == kind].groupby("variant")[["cov_all", "cov_burning", "cov_gt_p90", "width"]].mean().round(2)
        print(f"\n{kind.upper()}\n{g.to_string()}")


if __name__ == "__main__":
    main()

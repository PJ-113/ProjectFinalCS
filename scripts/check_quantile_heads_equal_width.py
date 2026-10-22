"""
เทียบ quantile-head variant ต่างๆ ที่ความกว้างช่วงเท่ากัน (ขยาย base ด้วย q_hat เพิ่ม) วัด coverage
และ Winkler interval score; calibA คำนวณ q_hat, calibB วัดผล (ไม่แตะ test); horizon [1,12,24]
"""
import importlib.util
import os

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
HORIZONS = [1, 12, 24]
ALPHA = 0.10


def load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, "scripts", fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def winkler(lo, hi, y, alpha=ALPHA):
    return np.mean((hi - lo) + (2 / alpha) * np.maximum(lo - y, 0) + (2 / alpha) * np.maximum(y - hi, 0))


def main():
    tm = load("tm", "train_multiprovince_models.py")
    tq = load("tq", "tune_quantile_heads_pooled.py")
    df = pd.read_csv(tm.DATA_PATH)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    df = df[~df["is_zero_shot"]].reset_index(drop=True)
    train_df, calib_df, _ = tm.split_train_calib_test(df)
    calA, calB = tq.split_calib(calib_df)
    feats = tm.FEATURE_COLS
    cfgs = {
        "lgb": {"base": tm.TUNED_PARAMS_LGB,
                "reg3": dict(learning_rate=0.03, num_leaves=15, min_child_samples=100, reg_lambda=30.0, n_estimators=300)},
        "xgb": {"base": tm.TUNED_PARAMS_XGB,
                "reg2": dict(learning_rate=0.03, max_depth=4, min_child_weight=50, reg_lambda=10.0, n_estimators=400),
                "reg3": dict(learning_rate=0.03, max_depth=4, min_child_weight=20, reg_lambda=30.0, n_estimators=400)},
    }
    rows = []
    for kind, variants in cfgs.items():
        for h in HORIZONS:
            tcol = f"pm25_t+{h}"
            tr = train_df.dropna(subset=[tcol])
            y_res = tr[tcol] - tr["pm25"]
            preds = {}
            for name, params in variants.items():
                m05 = tq.fit_q(tr[feats], y_res, kind, 0.05, params, None, tm)
                m95 = tq.fit_q(tr[feats], y_res, kind, 0.95, params, None, tm)
                pr = {}
                for tag, d in (("A", calA), ("B", calB)):
                    d = d.dropna(subset=[tcol])
                    pr[tag] = (m05.predict(d[feats]) + d["pm25"].values, m95.predict(d[feats]) + d["pm25"].values,
                               d[tcol].values)
                preds[name] = pr
            def stats(lo, hi, y):
                inside = (y >= lo) & (y <= hi)
                gt90 = y > np.percentile(y, 90)
                return inside.mean() * 100, inside[gt90].mean() * 100, (hi - lo).mean(), winkler(lo, hi, y)
            qh = {n: tm.conformalize(*preds[n]["A"]) for n in preds}
            base_lo, base_hi, yB = preds["base"]["B"]
            for name in variants:
                lo, hi, y = preds[name]["B"]
                c, c90, w, ws = stats(lo - qh[name], hi + qh[name], y)
                rows.append({"model": kind, "variant": name, "kind": "own_qhat", "horizon": h, "cov": c, "cov_gt_p90": c90, "width": w, "winkler": ws})
                if name != "base":
                    # base ขยายให้กว้างเท่า variant
                    extra = (w - (base_hi - base_lo + 2 * qh["base"]).mean()) / 2
                    c, c90, w2, ws = stats(base_lo - qh["base"] - extra, base_hi + qh["base"] + extra, yB)
                    rows.append({"model": kind, "variant": f"base_widened_to_{name}", "kind": "equal_width", "horizon": h,
                                 "cov": c, "cov_gt_p90": c90, "width": w2, "winkler": ws})
            print(f"  {kind} h={h} เสร็จ", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(_ROOT, "models", f"check_quantile_heads_equal_width_{TIMESTAMP}.csv"), index=False)
    print(res.groupby(["model", "variant"])[["cov", "cov_gt_p90", "width", "winkler"]].mean().round(2).to_string())


if __name__ == "__main__":
    main()

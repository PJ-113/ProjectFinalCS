"""
คำนวณ conformal recalibration ของช่วงความเชื่อมั่น 90% แยกตามระยะเงียบ k และ horizon h
เมื่อ input ถูกเติมค่าด้วยวิธี H-lite

ต่อ (kind, k, h): เติมค่าด้วย H-lite บน calib แล้วหา q_imp จากคะแนนความไม่สอดคล้อง s = max(lo-y, y-hi)
(split conformal / CQR) จากนั้นวัด coverage และความกว้างช่วงบน test เทียบกับช่วงเดิม (q_hat เดิม)
แยกรายงานฤดูฝุ่น/ช่วงปกติ
ผลลัพธ์: models/imputation_interval_recalibration_<ts>.csv และตาราง q_imp ต่อ (kind,k,h) ใน
models/imputation_interval_qhat_<ts>.json
รัน: python scripts/experiment_imputation_interval_recalibration.py [--quick]
"""
import glob
import importlib.util
import json
import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("dn", os.path.join(_ROOT, "scripts", "experiment_imputation_downstream.py"))
dn = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dn)
ea = dn.ea

TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
QUICK = ea.QUICK
ALPHA = 0.10
N_CAL = 30 if QUICK else 200      # ต่อจังหวัดต่อ k (calib)
N_TEST = 25 if QUICK else 150     # ต่อจังหวัดต่อ k (test)
BURNING = ea.BURNING


def conformal_q(scores, alpha=ALPHA):
    n = len(scores)
    level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return float(np.quantile(scores, level))


def build_inputs(P, tr, ds_ix, ds, models, feats, p, t, k):
    """คืน dict วิธี H: (X_mod, base_mod) + ค่าจริงต่อ h; และตัวช่วย (months)"""
    rows_t = dn.ds_rows(P, ds_ix, p, t)
    Xds = ds.iloc[rows_t][feats].reset_index(drop=True)
    Ws = dn.impute_window(P, tr, {}, ds_ix, models, p, t, k, only_h=True)
    f, b = dn.feats_from_window(Ws["I_HL"])
    X = Xds.copy()
    for c, v in f.items():
        X[c] = v
    return X, b


def main():
    rng = np.random.default_rng(ea.SEED)
    P = ea.load_panels(need_full_dataset=True)
    ds, tm = P["ds"], P["tm"]
    feats = tm.FEATURE_COLS
    ds_frame = ea.build_ds_index(P)

    class _Ix:
        pass
    ds_ix = _Ix()
    ds_ix.index = ds_frame.index
    ds_ix.attrs = {"ix_frame": ds_frame}
    models = ea.load_v3_models()
    q_orig = {kind: pd.read_csv(os.path.join(ea.MODEL_DIR, f"conformal_calibration_{kind}_{ea.MODEL_TAG}.csv")).set_index("horizon")["q_hat"].to_dict()
              for kind in ("lgb", "xgb")}
    tuned_files = sorted(f for f in glob.glob(os.path.join(ea.MODEL_DIR, "imputation_tuned_params_2*.json")) if ("_quick" in f) == QUICK)
    tuned = json.load(open(tuned_files[-1], encoding="utf-8"))
    assert "switch_j_lite" in tuned["tune"], "ไฟล์พารามิเตอร์ที่จูนไม่มีจุดสลับ H-lite — รัน experiment_imputation_accuracy.py ใหม่"
    print("ใช้พารามิเตอร์:", os.path.basename(tuned_files[-1]), {k: v for k, v in tuned["tune"].items() if k != "switch_mae_calib"}, flush=True)
    tr = ea.fit_trailing(P, np.random.default_rng(ea.SEED), 20000 if QUICK else 60000, tuned["tune"], fast=True)

    rows, qtable = [], {"lgb": {}, "xgb": {}}
    for k in dn.KS_B:
        # ---- calib: หา q_imp ----
        pc, tc = dn.sample_issue(P, ds_ix, k, N_CAL, rng, split="calib")
        Xc, bc = build_inputs(P, tr, ds_ix, ds, models, feats, pc, tc, k)
        # ---- test: วัดผล (จุดเวลาชุดอื่น) ----
        pt, tt = dn.sample_issue(P, ds_ix, k, N_TEST, rng, split="test")
        Xt, bt = build_inputs(P, tr, ds_ix, ds, models, feats, pt, tt, k)
        months = P["month"][tt]
        for kind in ("lgb", "xgb"):
            qtable[kind].setdefault(str(k), {})
            for h in dn.HS:
                yc = P["obs"][tc + h, pc]
                lo_c = dn.model_predict(models, kind, h, Xc, bc, 0.05)
                hi_c = dn.model_predict(models, kind, h, Xc, bc, 0.95)
                okc = ~np.isnan(yc)
                q_imp = conformal_q(np.maximum(lo_c[okc] - yc[okc], yc[okc] - hi_c[okc]))
                qtable[kind][str(k)][str(h)] = q_imp

                y = P["obs"][tt + h, pt]
                lo = dn.model_predict(models, kind, h, Xt, bt, 0.05)
                hi = dn.model_predict(models, kind, h, Xt, bt, 0.95)
                ok = ~np.isnan(y)
                for label, q in (("orig", q_orig[kind][h]), ("recal", q_imp)):
                    inside = ((y >= lo - q) & (y <= hi + q))[ok]
                    burn = np.isin(months, list(BURNING))[ok]
                    rows.append({"kind": kind, "k": k, "h": h, "interval": label, "q": q,
                                 "coverage": float(inside.mean()),
                                 "coverage_burning": float(inside[burn].mean()) if burn.any() else np.nan,
                                 "coverage_non_burning": float(inside[~burn].mean()) if (~burn).any() else np.nan,
                                 "mean_width": float((hi - lo + 2 * q)[ok].mean()),
                                 "n_test": int(ok.sum()), "n_calib": int(okc.sum())})
        print(f"  k={k:2d} เสร็จ (calib n={len(pc)}, test n={len(pt)})", flush=True)

    suffix = "_quick" if QUICK else ""
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(ea.MODEL_DIR, f"imputation_interval_recalibration_{TIMESTAMP}{suffix}.csv"), index=False)
    with open(os.path.join(ea.MODEL_DIR, f"imputation_interval_qhat_{TIMESTAMP}{suffix}.json"), "w", encoding="utf-8") as f:
        json.dump({"alpha": ALPHA, "method": "H-lite", "note": "q_imp ต่อ (kind,k,h) จาก calib; ใช้ q ของ k ที่ถัดไปที่มากกว่าเมื่อ k จริงไม่ตรงค่าที่ทดสอบ",
                   "q_hat_imputed": qtable}, f, indent=1)

    for kind in ("lgb", "xgb"):
        print(f"\n=== {kind.upper()} coverage บน test (เป้า 90%): เดิม -> หลังปรับ ===")
        piv = res[res.kind == kind].pivot_table(index=["k", "interval"], columns="h", values="coverage").round(3)
        print(piv.to_string())
        w = res[res.kind == kind].pivot_table(index="k", columns="interval", values="mean_width").round(2)
        print("ความกว้างเฉลี่ยของช่วง (µg/m³):\n" + w.to_string())
    r = res[res.interval == "recal"]
    print("\nเกณฑ์: coverage หลังปรับอยู่ใน 88–92% ทุก (kind,k,h):",
          bool(((r.coverage >= 0.88) & (r.coverage <= 0.92)).all()),
          f"(ต่ำสุด {r.coverage.min():.3f}, สูงสุด {r.coverage.max():.3f})")
    print("ฤดูฝุ่น/ปกติ อยู่ในเป้า ±3 จุดจาก 90%:",
          bool(((r.coverage_burning - 0.9).abs() <= 0.03).all()), bool(((r.coverage_non_burning - 0.9).abs() <= 0.03).all()))


if __name__ == "__main__":
    main()

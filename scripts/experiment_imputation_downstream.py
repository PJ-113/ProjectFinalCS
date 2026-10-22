"""
วัดว่าเติมค่าแล้วพยากรณ์ต่อ ดีกว่าใช้ Persistence/โมเดลสำรองหรือไม่

สถานการณ์ S-B (สถานีเงียบ k ชม. ก่อนเวลาพยากรณ์ t; k=1,3,6,12,24,48): เทียบ P0_persistence,
P3_stale_origin (โมเดลหลักจาก origin t0), C_complete (ข้อมูลครบ, เพดานบน), และ I_M0..I_M9
(เติมค่าช่องว่างด้วยวิธี M* จาก experiment_imputation_accuracy.py แล้วสร้าง feature ใหม่ป้อนโมเดลหลัก)
ที่ horizon h = 1,6,12,18,24
สถานการณ์ S-C (ช่องว่างภายในหน้าต่าง lag/rolling ขณะที่ค่า ณ t มี; L=3,6,12,24): M1/M2/M3i/M4i เทียบ C_complete
วัด coverage ของช่วงความเชื่อมั่น 90% (conformal) เมื่อ input ถูกเติม
รายงานส่วนต่าง MAE เป็น % เทียบ P0/P3 พร้อม block-bootstrap 95% CI (บล็อก = จังหวัด x สัปดาห์)
รัน: python scripts/experiment_imputation_downstream.py [--quick]
"""
import glob
import json
import os
import sys

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "scripts"))
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location("ea", os.path.join(_ROOT, "scripts", "experiment_imputation_accuracy.py"))
ea = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ea)

TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
QUICK = ea.QUICK
MODEL_DIR = ea.MODEL_DIR
KS_B = [1, 3, 6, 12, 24, 48]
LS_C = [3, 6, 12, 24]
HS = [1, 6, 12, 18, 24]
BURNING = ea.BURNING
N_PER_PROV = 25 if QUICK else 150       # ต่อจังหวัดต่อ k
N_BOOT = 100 if QUICK else 500
IMPUTED = ["I_M0", "I_M3", "I_M4", "I_M5", "I_M6", "I_M7", "I_M8", "I_M9", "I_H", "I_HL"]
LAG_COLS = {"pm25_lag_1": 23, "pm25_lag_3": 21, "pm25_lag_6": 18, "pm25_lag_12": 12, "pm25_lag_24": 0}


def latest(pattern):
    files = sorted(f for f in glob.glob(os.path.join(MODEL_DIR, pattern)) if ("_quick" in f) == QUICK)
    if not files:
        raise SystemExit(f"ไม่พบ {pattern} — รัน experiment_imputation_accuracy.py ก่อน")
    return files[-1]


def feats_from_window(W):
    """W (n,25): ซีรีส์ pm25 ที่ t-24..t (index 24 = t) -> ค่าของ 9 feature pm25 + ฐาน residual ตามสูตรเดียวกับ build_raw_dataset_multiprovince"""
    r6, r24 = W[:, 19:25], W[:, 1:25]
    out = {c: W[:, i] for c, i in LAG_COLS.items()}
    out.update({"pm25_roll_mean_6": r6.mean(1), "pm25_roll_std_6": r6.std(1, ddof=1),
                "pm25_roll_mean_24": r24.mean(1), "pm25_roll_std_24": r24.std(1, ddof=1)})
    return out, W[:, 24]


def model_predict(models, kind, h, X, base, q=0.5):
    return models[kind]["models"][h][q].predict(X) + base


def sample_issue(P, ds_ix, k, n_per_prov, rng, block_L=None, split="test"):
    """เวลา t (ในช่วง test) ที่ ds มีแถว (p,t), วัดได้จริงที่ t และ t0 = t-k (หรือ t-L-1 สำหรับ S-C) และซีรีส์ filled ครบในหน้าต่างส่วนที่ไม่ถูกเติม"""
    gap = k if block_L is None else block_L + 1
    ps, ts = [], []
    keys_prov = np.array(P["provs"])
    for p in range(len(P["provs"])):
        lo, hi = P["bounds"][p][split]
        obs_ok = ~np.isnan(P["obs"][:, p])
        nanf = np.cumsum(np.isnan(P["filled"][:, p]))
        cand = np.arange(max(lo, 30 + gap), hi - max(HS))
        cand = cand[obs_ok[cand] & obs_ok[cand - gap]]
        if len(cand) == 0:
            continue
        a = np.maximum(cand - 24, 0) if gap < 24 else cand - gap
        b = cand - gap
        bad = (nanf[b] - np.where(a > 0, nanf[np.maximum(a - 1, 0)], 0)) > 0
        cand = cand[~bad]
        has_row = ds_ix.index.get_indexer(pd.MultiIndex.from_arrays([np.full(len(cand), keys_prov[p]), P["grid"][cand]])) >= 0
        cand = cand[has_row]
        if len(cand) == 0:
            continue
        pick = rng.choice(cand, size=min(n_per_prov, len(cand)), replace=False)
        ps.append(np.full(len(pick), p)); ts.append(pick)
    return np.concatenate(ps), np.concatenate(ts)


def ds_rows(P, ds_ix, p, t):
    keys = pd.MultiIndex.from_arrays([np.array(P["provs"])[p], P["grid"][t]])
    return ds_ix.index.get_indexer(keys)


def impute_window(P, tr, blend, ds_ix, models, p, t, k, only_h=False):
    """W (n,25) = filled ที่ t-24..t โดยตำแหน่งที่อยู่หลัง t0=t-k ถูกแทนด้วยค่าที่เติมของแต่ละวิธี -> dict {วิธี: W}"""
    n = len(p)
    t0 = t - k
    base_W = P["filled"][t[:, None] + np.arange(-24, 1)[None, :], p[:, None]]
    Ws = {m: base_W.copy() for m in IMPUTED}
    for j in range(max(k - 24, 1), k + 1):
        col = 24 + (j - k)
        jj = np.full(n, j)
        F = ea.make_feats(P, p, t0, jj)
        if only_h:  # เฉพาะวิธี H-lite (ใช้ใน experiment_imputation_interval_recalibration.py)
            out = ea.impute_trailing(P, tr, F, ("HL",))
        else:
            out = ea.impute_trailing(P, tr, F, ("M0", "M3", "M4", "M5", "M6", "M7", "H", "HL"))
            m8 = np.full(n, np.nan)
            if j <= 24:
                m8 = ea.m8_model_from_origin(P, ds_ix.attrs["ix_frame"], models, F, "lgb")
            out["M8"] = ea.clip0(np.where(np.isnan(m8), out["M3"], m8))
            jb = next((x for x in ea.K_LIST if x >= j), ea.K_LIST[-1])
            w = blend.get(jb)
            cands = np.column_stack([out[c] for c in ("M3", "M4", "M5", "M6")])
            out["M9"] = ea.clip0(cands @ np.array(w)) if w is not None else out["M3"]
        for name, v in out.items():
            Ws["I_" + name][:, col] = v
    return Ws


def block_ids(P, p, t):
    week = (P["grid"][t].isocalendar().week.values.astype(int) + 100 * P["grid"][t].year.values)
    return p * 100000 + week


def bootstrap_delta(err_m, err_ref, blocks, rng, n_boot):
    """delta% = (mean|e_m| - mean|e_ref|) / mean|e_ref| * 100 ; CI แบบ block bootstrap"""
    ub = np.unique(blocks)
    idx = {b: np.where(blocks == b)[0] for b in ub}
    sm = np.array([err_m[idx[b]].sum() for b in ub]); sr = np.array([err_ref[idx[b]].sum() for b in ub])
    cnt = np.array([len(idx[b]) for b in ub])
    point = (err_m.mean() - err_ref.mean()) / err_ref.mean() * 100
    res = np.empty(n_boot)
    for i in range(n_boot):
        s = rng.integers(0, len(ub), len(ub))
        res[i] = (sm[s].sum() - sr[s].sum()) / sr[s].sum() * 100
    return float(point), float(np.percentile(res, 2.5)), float(np.percentile(res, 97.5))


def main():
    rng = np.random.default_rng(ea.SEED)
    P = ea.load_panels(need_full_dataset=True)
    ds, tm = P["ds"], P["tm"]
    feats = tm.FEATURE_COLS
    ds_frame = ea.build_ds_index(P)

    class _Ix:  # ห่อ index + เก็บ frame ให้ m8_model_from_origin ใช้
        pass
    ds_ix = _Ix()
    ds_ix.index = ds_frame.index
    ds_ix.attrs = {"ix_frame": ds_frame}
    models = ea.load_v3_models()
    q_hat = {kind: pd.read_csv(os.path.join(MODEL_DIR, f"conformal_calibration_{kind}_{ea.MODEL_TAG}.csv")).set_index("horizon")["q_hat"].to_dict()
             for kind in ("lgb", "xgb")}
    tuned = json.load(open(latest("imputation_tuned_params_2*.json"), encoding="utf-8"))
    blend = {int(k): v for k, v in tuned["blend"].items()}
    print("พารามิเตอร์ที่จูนไว้:", tuned["tune"], flush=True)
    tr = ea.fit_trailing(P, np.random.default_rng(ea.SEED), 20000 if QUICK else 60000, tuned["tune"])

    trained_idx = [i for i, p in enumerate(P["provs"]) if not P["zero_shot"][p]]
    p90 = float(np.nanpercentile(P["obs"][:, trained_idx], 90))
    agg_rows, cov_rows, pair_rows = [], [], []
    pooled = {}  # (kind, k, method) -> dict(err arrays, months, y, blocks) รวมทุก h สำหรับ bootstrap

    for k in KS_B:
        p, t = sample_issue(P, ds_ix, k, N_PER_PROV, rng)
        n = len(p)
        t0 = t - k
        rows_t = ds_rows(P, ds_ix, p, t)
        Xds = ds.iloc[rows_t][feats].reset_index(drop=True)
        base_ds = ds.iloc[rows_t]["pm25"].values
        # parity: สร้าง feature จาก window ที่ไม่เติมค่า ต้องตรง dataset (ตรวจก่อนใช้จริง)
        W0 = P["filled"][t[:, None] + np.arange(-24, 1)[None, :], p[:, None]]
        f0, b0 = feats_from_window(W0)
        for c, v in f0.items():
            # เผื่อ tolerance 1e-4 เฉพาะ std (ต่างจาก np.std เล็กน้อย)
            tol = 1e-4 if "_std_" in c else 1e-6
            assert np.nanmax(np.abs(v - Xds[c].values)) < tol, f"parity ล้มเหลวที่ {c}"
        assert np.nanmax(np.abs(b0 - base_ds)) < 1e-6
        Ws = impute_window(P, tr, blend, ds_ix, models, p, t, k)
        Xmod, basemod = {}, {}
        for m, W in Ws.items():
            f, b = feats_from_window(W)
            X = Xds.copy()
            for c, v in f.items():
                X[c] = v
            Xmod[m], basemod[m] = X, b
        x_t0 = P["obs"][t0, p]
        blocks = block_ids(P, p, t)
        months_t = P["month"][t]
        print(f"S-B k={k:2d}: n={n}", flush=True)
        for kind in ("lgb", "xgb"):
            for h in HS:
                y = P["obs"][t + h, p]
                preds = {"P0_persistence": x_t0.copy(), "C_complete": model_predict(models, kind, h, Xds, base_ds)}
                if k + h <= 24:
                    r0 = ds_rows(P, ds_ix, p, t0)
                    ok0 = r0 >= 0
                    v = np.full(n, np.nan)
                    if ok0.any():
                        sub = ds.iloc[r0[ok0]]
                        v[ok0] = model_predict(models, kind, k + h, sub[feats], sub["pm25"].values)
                    preds["P3_stale_origin"] = v
                for m in IMPUTED:
                    preds[m] = np.maximum(model_predict(models, kind, h, Xmod[m], basemod[m]), 0)
                # coverage (ช่วง 90% หลัง conformal) ของ C_complete และวิธีเติมค่า
                for m in ["C_complete"] + IMPUTED:
                    X = Xds if m == "C_complete" else Xmod[m]
                    b = base_ds if m == "C_complete" else basemod[m]
                    lo = model_predict(models, kind, h, X, b, 0.05) - q_hat[kind][h]
                    hi = model_predict(models, kind, h, X, b, 0.95) + q_hat[kind][h]
                    okc = ~np.isnan(y)
                    cov_rows.append({"scenario": "S-B", "kind": kind, "k": k, "h": h, "method": m,
                                     "coverage": float(((y[okc] >= lo[okc]) & (y[okc] <= hi[okc])).mean()),
                                     "mean_width": float((hi - lo)[okc].mean()), "n": int(okc.sum())})
                # common rows: y วัดได้จริง + ทุกวิธีที่มีผลต้องไม่ NaN
                mat = np.column_stack(list(preds.values()))
                okm = ~np.isnan(y) & ~np.isnan(mat).any(axis=1)
                if okm.sum() < 20:
                    continue
                for name, v in preds.items():
                    e = np.abs(v[okm] - y[okm])
                    for grp, sel in (("all", np.ones(okm.sum(), bool)), ("burning", np.isin(months_t[okm], list(BURNING))),
                                     ("non_burning", ~np.isin(months_t[okm], list(BURNING))), ("gt_p90", y[okm] > p90)):
                        if sel.sum() == 0:
                            continue
                        agg_rows.append({"scenario": "S-B", "kind": kind, "k": k, "h": h, "method": name, "group": grp,
                                         "mae": float(e[sel].mean()), "rmse": float(np.sqrt(((v[okm] - y[okm])[sel] ** 2).mean())),
                                         "n": int(sel.sum())})
                    key = (kind, k, name)
                    d = pooled.setdefault(key, {"e": [], "months": [], "y": [], "blocks": [], "h": []})
                    d["e"].append(e); d["months"].append(months_t[okm]); d["y"].append(y[okm])
                    d["blocks"].append(blocks[okm]); d["h"].append(np.full(okm.sum(), h))
        print(f"    เสร็จ k={k}", flush=True)

    # ---------- paired vs P0 / P3 (pooled ข้าม h) ----------
    for (kind, k, name), d in pooled.items():
        if name in ("P0_persistence",):
            continue
        for ref in ("P0_persistence", "P3_stale_origin"):
            if ref == name or (kind, k, ref) not in pooled:
                continue
            dr = pooled[(kind, k, ref)]
            # ต้อง align แถว: เก็บตามลำดับ (h) เดียวกัน แต่ความยาวต่างกันได้ถ้าเงื่อนไข common-row ต่างกัน -> จับคู่เฉพาะเมื่อยาวเท่ากัน
            e_m, e_r = np.concatenate(d["e"]), np.concatenate(dr["e"])
            if len(e_m) != len(e_r):
                continue
            months, y, blocks = np.concatenate(d["months"]), np.concatenate(d["y"]), np.concatenate(d["blocks"])
            for grp, sel in (("all", np.ones(len(e_m), bool)), ("burning", np.isin(months, list(BURNING))),
                             ("non_burning", ~np.isin(months, list(BURNING))), ("gt_p90", y > p90)):
                if sel.sum() < 30:
                    continue
                pt, lo, hi = bootstrap_delta(e_m[sel], e_r[sel], blocks[sel], rng, N_BOOT)
                pair_rows.append({"kind": kind, "k": k, "method": name, "ref": ref, "group": grp,
                                  "mae_method": float(e_m[sel].mean()), "mae_ref": float(e_r[sel].mean()),
                                  "delta_pct": pt, "ci_lo": lo, "ci_hi": hi, "n": int(sel.sum())})

    # ---------- S-C: ช่องว่างภายในหน้าต่าง lag/rolling ขณะที่ค่า ณ t มี ----------
    for L in LS_C:
        p, t = sample_issue(P, ds_ix, 0, N_PER_PROV, rng, block_L=L)
        n = len(p)
        t0 = t - L - 1
        rows_t = ds_rows(P, ds_ix, p, t)
        Xds = ds.iloc[rows_t][feats].reset_index(drop=True)
        base_ds = ds.iloc[rows_t]["pm25"].values
        base_W = P["filled"][t[:, None] + np.arange(-24, 1)[None, :], p[:, None]]
        Wm = {m: base_W.copy() for m in ("M1", "M2", "M3i", "M4i")}
        for j in range(1, L + 1):
            col = 24 + (j - (L + 1))  # tau = t0 + j = t - (L+1-j)
            if col < 0:
                continue
            F = ea.make_feats(P, p, t0, np.full(n, j), L)
            imp = {"M1": ea.m1_linear(F), "M2": ea.m2_pchip(P, F), "M3i": ea.m3i_diurnal(F),
                   "M4i": np.where(np.isnan(ea.m4i_spatial(F)), F["lin"], ea.m4i_spatial(F))}
            for m, v in imp.items():
                Wm[m][:, col] = ea.clip0(v)
        Xmod, basemod = {}, {}
        for m, W in Wm.items():
            f, b = feats_from_window(W)
            X = Xds.copy()
            for c, v in f.items():
                X[c] = v
            Xmod[m], basemod[m] = X, b
        months_t = P["month"][t]
        for kind in ("lgb", "xgb"):
            for h in HS:
                y = P["obs"][t + h, p]
                preds = {"C_complete": model_predict(models, kind, h, Xds, base_ds)}
                for m in Wm:
                    preds[m] = np.maximum(model_predict(models, kind, h, Xmod[m], basemod[m]), 0)
                okm = ~np.isnan(y)
                for name, v in preds.items():
                    e = np.abs(v[okm] - y[okm])
                    for grp, sel in (("all", np.ones(okm.sum(), bool)), ("burning", np.isin(months_t[okm], list(BURNING))),
                                     ("non_burning", ~np.isin(months_t[okm], list(BURNING))), ("gt_p90", y[okm] > p90)):
                        if sel.sum():
                            agg_rows.append({"scenario": "S-C", "kind": kind, "k": L, "h": h, "method": name, "group": grp,
                                             "mae": float(e[sel].mean()), "rmse": float(np.sqrt(((v[okm] - y[okm])[sel] ** 2).mean())),
                                             "n": int(sel.sum())})
        print(f"S-C L={L:2d}: n={n}", flush=True)

    suffix = "_quick" if QUICK else ""
    agg = pd.DataFrame(agg_rows)
    agg.to_csv(os.path.join(MODEL_DIR, f"imputation_downstream_{TIMESTAMP}{suffix}.csv"), index=False)
    pair = pd.DataFrame(pair_rows)
    pair.to_csv(os.path.join(MODEL_DIR, f"imputation_downstream_paired_{TIMESTAMP}{suffix}.csv"), index=False)
    cov = pd.DataFrame(cov_rows)
    cov.to_csv(os.path.join(MODEL_DIR, f"imputation_coverage_{TIMESTAMP}{suffix}.csv"), index=False)

    # ---------- สรุปผล ----------
    print("\n" + "=" * 100 + "\nMAE เฉลี่ย 5 horizon (LGB) แยกตาม k — S-B, ทุกช่วงเวลา\n" + "=" * 100)
    t1 = agg[(agg.scenario == "S-B") & (agg.kind == "lgb") & (agg.group == "all")].groupby(["method", "k"])["mae"].mean().unstack().round(3)
    print(t1.to_string())
    if len(pair):
        print("\nส่วนต่าง % เทียบ P0_persistence (ลบ = ดีกว่า) LGB, ทุกช่วงเวลา, [CI 95%]")
        s = pair[(pair.kind == "lgb") & (pair.ref == "P0_persistence") & (pair.group == "all")]
        for _, r in s.sort_values(["k", "delta_pct"]).iterrows():
            print(f"  k={int(r.k):2d} {r.method:15s} {r.delta_pct:+6.2f}%  [{r.ci_lo:+6.2f}, {r.ci_hi:+6.2f}]")
    print("\ncoverage เฉลี่ย (LGB) ต่อ k (เป้า ~90%)")
    print(cov[cov.kind == "lgb"].groupby(["method", "k"])["coverage"].mean().unstack().round(3).to_string())
    print("\nบันทึกไฟล์ผลลัพธ์ใน models/ (imputation_downstream_*, imputation_downstream_paired_*, imputation_coverage_*)")


if __name__ == "__main__":
    main()

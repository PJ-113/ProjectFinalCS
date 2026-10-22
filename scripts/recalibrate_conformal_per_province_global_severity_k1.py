"""
Mondrian conformal calibration: severity group จาก threshold สัมบูรณ์รวมทุกจังหวัด (เหมือน flat_3grp)
แต่คำนวณ q_hat แยกต่อจังหวัดภายในแต่ละ severity group (province_within_global_severity)
เทียบ coverage ของ 4 วิธี: single_qhat, flat_3grp, province_severity (relative percentile),
province_within_global_severity
"""
import importlib.util
import os
import pickle

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(_ROOT, "models")
MODEL_TAG = "20260917_001147"
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
THRESHOLDS_PCTL = [85, 95]  # 3 กลุ่ม: rest / p85-p95 / >p95
MIN_GROUP_SIZE = 20


def load_train_module():
    spec = importlib.util.spec_from_file_location(
        "train_multiprovince_models", os.path.join(_ROOT, "scripts", "train_multiprovince_models.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_model_pkg(tag):
    path = os.path.join(MODEL_DIR, f"{tag}_pm25_multiprovince_residual_bagged_quantile_{MODEL_TAG}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_baseline_qhat(tag):
    path = os.path.join(MODEL_DIR, f"conformal_calibration_{tag}_{MODEL_TAG}.csv")
    df = pd.read_csv(path)
    return {int(r.horizon): float(r.q_hat) for r in df.itertuples()}


def coverage(lo, hi, y_true):
    return float(((y_true >= lo) & (y_true <= hi)).mean())


def assign_group(p50_values, thresholds):
    group = np.zeros(len(p50_values), dtype=int)
    for t in thresholds:
        group += (p50_values >= t).astype(int)
    return group


def quantile_qhat(scores, n):
    q_level = min(np.ceil((n + 1) * 0.9) / n, 1.0)
    return float(np.quantile(scores, q_level))


def compute_flat_group_qhat(p50_calib, lo_calib, hi_calib, y_calib, thresholds_pctl):
    """severity-only, รวมทุกจังหวัด"""
    thresholds_val = [np.percentile(p50_calib, p) for p in thresholds_pctl]
    group_calib = assign_group(p50_calib, thresholds_val)
    q_hats = {}
    for g in range(len(thresholds_pctl) + 1):
        mask = group_calib == g
        n = int(mask.sum())
        q_hats[g] = quantile_qhat(np.maximum(lo_calib[mask] - y_calib[mask], y_calib[mask] - hi_calib[mask]), n) \
            if n >= MIN_GROUP_SIZE else None
    return thresholds_val, q_hats


def compute_province_relative_severity_qhat(provinces_calib, p50_calib, lo_calib, hi_calib, y_calib, thresholds_pctl):
    """percentile เทียบจังหวัดตัวเอง"""
    thresholds_by_province, qhat_by_province_group, pooled_qhat_by_province = {}, {}, {}
    for province in np.unique(provinces_calib):
        mask_p = provinces_calib == province
        p50_p, lo_p, hi_p, y_p = p50_calib[mask_p], lo_calib[mask_p], hi_calib[mask_p], y_calib[mask_p]
        n_p = len(y_p)
        pooled_qhat_by_province[province] = (
            quantile_qhat(np.maximum(lo_p - y_p, y_p - hi_p), n_p) if n_p >= MIN_GROUP_SIZE else None)
        thresholds_val = [np.percentile(p50_p, p) for p in thresholds_pctl]
        thresholds_by_province[province] = thresholds_val
        group_p = assign_group(p50_p, thresholds_val)
        for g in range(len(thresholds_pctl) + 1):
            mask_g = group_p == g
            n = int(mask_g.sum())
            qhat_by_province_group[(province, g)] = (
                quantile_qhat(np.maximum(lo_p[mask_g] - y_p[mask_g], y_p[mask_g] - hi_p[mask_g]), n)
                if n >= MIN_GROUP_SIZE else None)
    return thresholds_by_province, qhat_by_province_group, pooled_qhat_by_province


def apply_province_relative(provinces_test, p50_test, thresholds_by_province, qhat_by_province_group,
                             pooled_qhat_by_province, baseline_qhat_h):
    q_hat_per_row = np.full(len(provinces_test), baseline_qhat_h, dtype=float)
    for province in np.unique(provinces_test):
        mask_p = provinces_test == province
        thresholds_val = thresholds_by_province.get(province)
        if thresholds_val is None:
            continue
        group_p = assign_group(p50_test[mask_p], thresholds_val)
        pooled_fallback = pooled_qhat_by_province.get(province)
        idx_p = np.where(mask_p)[0]
        for g in np.unique(group_p):
            qh = qhat_by_province_group.get((province, int(g)))
            if qh is None:
                qh = pooled_fallback if pooled_fallback is not None else baseline_qhat_h
            q_hat_per_row[idx_p[group_p == g]] = qh
    return q_hat_per_row


def compute_province_within_global_severity_qhat(provinces_calib, p50_calib, lo_calib, hi_calib, y_calib,
                                                   global_thresholds_val):
    """severity group มาจาก threshold สัมบูรณ์รวมทุกจังหวัด (เหมือน flat_3grp)
    แต่คำนวณ q_hat แยกต่อจังหวัดภายในแต่ละ group"""
    group_calib_global = assign_group(p50_calib, global_thresholds_val)
    qhat_by_province_group, pooled_qhat_by_province = {}, {}
    for province in np.unique(provinces_calib):
        mask_p = provinces_calib == province
        lo_p, hi_p, y_p = lo_calib[mask_p], hi_calib[mask_p], y_calib[mask_p]
        n_p = len(y_p)
        pooled_qhat_by_province[province] = (
            quantile_qhat(np.maximum(lo_p - y_p, y_p - hi_p), n_p) if n_p >= MIN_GROUP_SIZE else None)
        group_p_global = group_calib_global[mask_p]
        for g in range(len(global_thresholds_val) + 1):
            mask_g = group_p_global == g
            n = int(mask_g.sum())
            qhat_by_province_group[(province, g)] = (
                quantile_qhat(np.maximum(lo_p[mask_g] - y_p[mask_g], y_p[mask_g] - hi_p[mask_g]), n)
                if n >= MIN_GROUP_SIZE else None)
    return qhat_by_province_group, pooled_qhat_by_province


def apply_province_within_global(provinces_test, p50_test, global_thresholds_val, qhat_by_province_group,
                                  pooled_qhat_by_province, baseline_qhat_h):
    group_test_global = assign_group(p50_test, global_thresholds_val)
    q_hat_per_row = np.full(len(provinces_test), baseline_qhat_h, dtype=float)
    for province in np.unique(provinces_test):
        mask_p = provinces_test == province
        pooled_fallback = pooled_qhat_by_province.get(province)
        idx_p = np.where(mask_p)[0]
        for g in np.unique(group_test_global[mask_p]):
            qh = qhat_by_province_group.get((province, int(g)))
            if qh is None:
                qh = pooled_fallback if pooled_fallback is not None else baseline_qhat_h
            q_hat_per_row[idx_p[group_test_global[idx_p] == g]] = qh
    return q_hat_per_row


def main():
    train_mod = load_train_module()
    print(f"โหลดข้อมูล: {train_mod.DATA_PATH}")
    df = pd.read_csv(train_mod.DATA_PATH)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    df = df[~df["is_zero_shot"]].reset_index(drop=True)

    train_df, calib_df, test_df = train_mod.split_train_calib_test(df)
    print(f"train={len(train_df)}  calib={len(calib_df)}  test={len(test_df)}  (โมเดล k=1 tag={MODEL_TAG})")

    all_rows = []
    fallback_rows = []

    for tag in ("lgb", "xgb"):
        pkg = load_model_pkg(tag)
        baseline_qhat = load_baseline_qhat(tag)
        print(f"\n{'='*90}\n{tag.upper()}\n{'='*90}")

        for h in range(1, 25):
            target_col = f"pm25_t+{h}"
            calib_h = calib_df.dropna(subset=[target_col])
            test_h = test_df.dropna(subset=[target_col])
            models_by_q = pkg["models"][h]
            fcols = pkg["feature_cols"]

            calib_current = calib_h["pm25"].values
            p50_calib = models_by_q[0.5].predict(calib_h[fcols]) + calib_current
            lo_calib = models_by_q[0.05].predict(calib_h[fcols]) + calib_current
            hi_calib = models_by_q[0.95].predict(calib_h[fcols]) + calib_current
            y_calib = calib_h[target_col].values
            provinces_calib = calib_h["province_en"].values

            test_current = test_h["pm25"].values
            p50_test = models_by_q[0.5].predict(test_h[fcols]) + test_current
            lo_test = models_by_q[0.05].predict(test_h[fcols]) + test_current
            hi_test = models_by_q[0.95].predict(test_h[fcols]) + test_current
            y_test = test_h[target_col].values
            provinces_test = test_h["province_en"].values

            bins = [-0.01, np.percentile(y_test, 50), np.percentile(y_test, 75),
                    np.percentile(y_test, 90), y_test.max() + 0.01]
            labels = ["<=p50", "p50-p75", "p75-p90", ">p90"]
            severity_bin = pd.cut(y_test, bins=bins, labels=labels)

            configs_lo_hi = {"single_qhat": (lo_test - baseline_qhat[h], hi_test + baseline_qhat[h])}

            global_thresholds_val, flat_qhats = compute_flat_group_qhat(
                p50_calib, lo_calib, hi_calib, y_calib, THRESHOLDS_PCTL)
            group_test_flat = assign_group(p50_test, global_thresholds_val)
            q_hat_flat = np.array([flat_qhats[g] if flat_qhats[g] is not None else baseline_qhat[h]
                                    for g in group_test_flat])
            configs_lo_hi["flat_3grp"] = (lo_test - q_hat_flat, hi_test + q_hat_flat)

            thr_by_prov, qhat_rel, pooled_rel = compute_province_relative_severity_qhat(
                provinces_calib, p50_calib, lo_calib, hi_calib, y_calib, THRESHOLDS_PCTL)
            q_hat_rel = apply_province_relative(
                provinces_test, p50_test, thr_by_prov, qhat_rel, pooled_rel, baseline_qhat[h])
            configs_lo_hi["province_severity"] = (lo_test - q_hat_rel, hi_test + q_hat_rel)

            qhat_glob, pooled_glob = compute_province_within_global_severity_qhat(
                provinces_calib, p50_calib, lo_calib, hi_calib, y_calib, global_thresholds_val)
            q_hat_glob = apply_province_within_global(
                provinces_test, p50_test, global_thresholds_val, qhat_glob, pooled_glob, baseline_qhat[h])
            configs_lo_hi["province_within_global_severity"] = (lo_test - q_hat_glob, hi_test + q_hat_glob)

            if h == 1:
                n_fallback = sum(1 for v in qhat_glob.values() if v is None)
                n_fallback_global = sum(1 for v in pooled_glob.values() if v is None)
                fallback_rows.append({"model": tag, "n_province_group_cells": len(qhat_glob),
                                       "n_fallback_to_province_pooled": n_fallback,
                                       "n_fallback_to_global": n_fallback_global})

            for config_name, (lo_c, hi_c) in configs_lo_hi.items():
                for label in labels:
                    mask_b = np.asarray(severity_bin == label)
                    if mask_b.sum() == 0:
                        continue
                    all_rows.append({
                        "model": tag, "horizon": h, "config": config_name, "severity_bin": label,
                        "n": int(mask_b.sum()), "coverage": coverage(lo_c[mask_b], hi_c[mask_b], y_test[mask_b]),
                    })
                all_rows.append({
                    "model": tag, "horizon": h, "config": config_name, "severity_bin": "ALL",
                    "n": int(len(y_test)), "coverage": coverage(lo_c, hi_c, y_test),
                })

        print(f"  {tag}: ประเมินครบ 24 horizon x 4 config")

    print("\nจำนวน (province, severity) cell ที่ต้อง fallback (province_within_global_severity, ที่ horizon=1):")
    print(pd.DataFrame(fallback_rows).to_string(index=False))

    result_df = pd.DataFrame(all_rows)
    out_path = os.path.join(MODEL_DIR, f"recalibrate_province_global_severity_k1_{TIMESTAMP}.csv")
    result_df.to_csv(out_path, index=False)
    print(f"\nบันทึกผลดิบที่: {out_path}")

    print("\n" + "=" * 100)
    print("สรุป: coverage เฉลี่ยข้าม 24 horizon ต่อ severity bin x config (target = 90%) [โมเดล k=1]")
    print("=" * 100)
    summary = result_df.groupby(["model", "severity_bin", "config"])["coverage"].mean().unstack("config") * 100
    summary = summary[["single_qhat", "flat_3grp", "province_severity", "province_within_global_severity"]].round(1)
    for tag in ("lgb", "xgb"):
        print(f"\n--- {tag.upper()} ---")
        print(summary.loc[tag])


if __name__ == "__main__":
    main()

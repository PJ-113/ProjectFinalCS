"""
ต่อยอดจาก entry 120 (confounder analysis บนผล LOPO): พบว่า `pm25_std` (ความผันผวนของ PM2.5 ใน
จังหวัดนั้นเอง) อธิบาย MAE ได้ดีกว่าระยะทางหรือ predicted-severity มาก (r=0.85-0.87, p<0.0001) —
งานนี้ลอง Mondrian calibration แบบจัดกลุ่มด้วย **pm25_std ของจังหวัด** (ค่าคงที่ต่อจังหวัด ไม่ใช่
per-row แบบ severity เดิม) แทน predicted P50 severity (entry 115) หรือ (province, severity)
ร่วมกัน (entry 117-118) — ข้อดีสำคัญ: กลุ่มนี้ประกอบด้วยหลายจังหวัดรวมกัน (tertile ของ 17
จังหวัด ~5-6 จังหวัด/กลุ่ม) ทำให้ calib sample ต่อกลุ่มใหญ่กว่ามาก (~13,000+ แถว/กลุ่ม) ไม่เจอ
ปัญหาความแปรปรวนของ q_hat จากกลุ่มเล็กแบบ entry 117-118

ทดสอบ 2 มุมพร้อมกันในสคริปต์เดียว (ไม่มีการ retrain โมเดลใดๆ ใช้ pkg `_20260917_001147` ที่เทรน
ไว้แล้วอย่างเดียว):
  (A) แก้ปัญหา coverage >p90 เดิม (entry 111-118) บน test split ทางการของ 17 จังหวัดเทรน —
      เทียบกับ single_qhat และ flat_3grp (ดีที่สุดเดิมจาก entry 115)
  (B) แก้ปัญหา coverage ต่ำของ 3 จังหวัด zero-shot จริง (entry 119) — คำนวณ pm25_std ของแต่ละ
      จังหวัด zero-shot เอง แล้วจัดเข้ากลุ่ม volatility เดียวกับที่คำนวณจาก 17 จังหวัดเทรน (ใช้
      bin edge เดียวกัน) แล้วใช้ q_hat ของกลุ่มนั้นแทน single_qhat เดิม — เทียบ coverage บนช่วง
      test_period_only เดียวกับ entry 119
"""
import importlib.util
import json
import os
import pickle

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(_ROOT, "models")
REGISTRY_PATH = os.path.join(_ROOT, "data", "province_station_registry.json")
MODEL_TAG = "20260917_001147"
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
N_GROUPS = 3  # tertile ของ pm25_std ต่อจังหวัด (rest/mid/high volatility)
MIN_GROUP_SIZE = 20


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, "scripts", filename))
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


def quantile_qhat(scores, n):
    q_level = min(np.ceil((n + 1) * 0.9) / n, 1.0)
    return float(np.quantile(scores, q_level))


def compute_flat_group_qhat(p50_calib, lo_calib, hi_calib, y_calib, thresholds_pctl):
    """severity-only, รวมทุกจังหวัด (entry 115, ดีที่สุดเดิม — เก็บไว้เทียบในตารางเดียวกัน)"""
    thresholds_val = [np.percentile(p50_calib, p) for p in thresholds_pctl]
    group_calib = np.zeros(len(p50_calib), dtype=int)
    for t in thresholds_val:
        group_calib += (p50_calib >= t).astype(int)
    q_hats = {}
    for g in range(len(thresholds_pctl) + 1):
        mask = group_calib == g
        n = int(mask.sum())
        q_hats[g] = quantile_qhat(np.maximum(lo_calib[mask] - y_calib[mask], y_calib[mask] - hi_calib[mask]), n) \
            if n >= MIN_GROUP_SIZE else None
    return thresholds_val, q_hats


def assign_flat_group(p50_values, thresholds):
    group = np.zeros(len(p50_values), dtype=int)
    for t in thresholds:
        group += (p50_values >= t).astype(int)
    return group


def qhat_by_volatility_group(lo_calib, hi_calib, y_calib, provinces_calib, province_to_group, n_groups,
                              baseline_qhat_h):
    groups_calib = pd.Series(provinces_calib).map(province_to_group).values
    q_hats = {}
    for g in range(n_groups):
        mask = groups_calib == g
        n = int(mask.sum())
        q_hats[g] = quantile_qhat(np.maximum(lo_calib[mask] - y_calib[mask], y_calib[mask] - hi_calib[mask]), n) \
            if n >= MIN_GROUP_SIZE else None
    return q_hats


def apply_volatility_group(provinces_arr, province_to_group, q_hats, baseline_qhat_h):
    groups = pd.Series(provinces_arr).map(province_to_group).values
    return np.array([q_hats[g] if (g in q_hats and q_hats[g] is not None) else baseline_qhat_h for g in groups])


def main():
    train_mod = load_module("train_multiprovince_models", "train_multiprovince_models.py")
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        registry = json.load(f)
    provinces_meta = {p["province_en"]: p for p in registry["provinces"]}

    print(f"โหลดข้อมูล: {train_mod.DATA_PATH}")
    df = pd.read_csv(train_mod.DATA_PATH)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    trained_df = df[~df["is_zero_shot"]].reset_index(drop=True)
    zero_shot_df = df[df["is_zero_shot"]].reset_index(drop=True)

    train_df, calib_df, test_df = train_mod.split_train_calib_test(trained_df)
    test_start, test_end = test_df["datetime_utc"].min(), test_df["datetime_utc"].max()
    print(f"train={len(train_df)}  calib={len(calib_df)}  test={len(test_df)}  (โมเดล k=1 tag={MODEL_TAG})")

    # ---------- จัดกลุ่ม volatility ของ 17 จังหวัดเทรน ด้วย pm25_std (tertile) ----------
    province_std = trained_df.groupby("province_en")["pm25"].std().sort_values()
    labels, bin_edges = pd.qcut(province_std, N_GROUPS, labels=list(range(N_GROUPS)), retbins=True)
    province_to_group = {p: int(g) for p, g in zip(province_std.index, labels)}
    print(f"\nจัดกลุ่ม volatility (tertile ของ pm25_std, bin edges={bin_edges.round(2).tolist()}):")
    for g in range(N_GROUPS):
        members = [p for p, gg in province_to_group.items() if gg == g]
        print(f"  กลุ่ม {g} (pm25_std {'ต่ำ' if g == 0 else 'กลาง' if g == 1 else 'สูง'}): {members}")

    zero_shot_provinces = sorted(zero_shot_df["province_en"].unique().tolist())
    zero_shot_std = zero_shot_df.groupby("province_en")["pm25"].std()
    zero_shot_to_group = {}
    for p in zero_shot_provinces:
        std_val = zero_shot_std[p]
        g = int(np.clip(np.digitize(std_val, bin_edges[1:-1]), 0, N_GROUPS - 1))
        zero_shot_to_group[p] = g
    print("\nจังหวัด zero-shot จริง จัดเข้ากลุ่ม volatility เดียวกัน (เทียบ bin edge จาก 17 จังหวัดเทรน):")
    for p in zero_shot_provinces:
        print(f"  {p}: pm25_std={zero_shot_std[p]:.2f} -> กลุ่ม {zero_shot_to_group[p]}")

    zero_shot_restricted = zero_shot_df[
        (zero_shot_df["datetime_utc"] >= test_start) & (zero_shot_df["datetime_utc"] <= test_end)
    ].reset_index(drop=True)

    all_rows_trained, all_rows_zeroshot = [], []

    for tag in ("lgb", "xgb"):
        pkg = load_model_pkg(tag)
        baseline_qhat = load_baseline_qhat(tag)
        fcols = pkg["feature_cols"]
        print(f"\n{'='*90}\n{tag.upper()}\n{'='*90}")

        for h in range(1, 25):
            target_col = f"pm25_t+{h}"
            calib_h = calib_df.dropna(subset=[target_col])
            test_h = test_df.dropna(subset=[target_col])
            models_by_q = pkg["models"][h]

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
            labels_sev = ["<=p50", "p50-p75", "p75-p90", ">p90"]
            severity_bin = pd.cut(y_test, bins=bins, labels=labels_sev)

            configs_lo_hi = {"single_qhat": (lo_test - baseline_qhat[h], hi_test + baseline_qhat[h])}

            _, flat_qhats = compute_flat_group_qhat(p50_calib, lo_calib, hi_calib, y_calib, [85, 95])
            thresholds_flat = [np.percentile(p50_calib, p) for p in [85, 95]]
            group_test_flat = assign_flat_group(p50_test, thresholds_flat)
            q_hat_flat = np.array([flat_qhats[g] if flat_qhats[g] is not None else baseline_qhat[h]
                                    for g in group_test_flat])
            configs_lo_hi["flat_3grp"] = (lo_test - q_hat_flat, hi_test + q_hat_flat)

            vol_qhats = qhat_by_volatility_group(lo_calib, hi_calib, y_calib, provinces_calib, province_to_group,
                                                  N_GROUPS, baseline_qhat[h])
            q_hat_vol = apply_volatility_group(provinces_test, province_to_group, vol_qhats, baseline_qhat[h])
            configs_lo_hi["volatility_grp"] = (lo_test - q_hat_vol, hi_test + q_hat_vol)

            for config_name, (lo_c, hi_c) in configs_lo_hi.items():
                for label in labels_sev:
                    mask_b = np.asarray(severity_bin == label)
                    if mask_b.sum() == 0:
                        continue
                    all_rows_trained.append({
                        "model": tag, "horizon": h, "config": config_name, "severity_bin": label,
                        "n": int(mask_b.sum()), "coverage": coverage(lo_c[mask_b], hi_c[mask_b], y_test[mask_b]),
                    })
                all_rows_trained.append({
                    "model": tag, "horizon": h, "config": config_name, "severity_bin": "ALL",
                    "n": int(len(y_test)), "coverage": coverage(lo_c, hi_c, y_test),
                })

            # ---------- (B) จังหวัด zero-shot จริง ----------
            zs_h = zero_shot_restricted.dropna(subset=[target_col])
            if not zs_h.empty:
                zs_current = zs_h["pm25"].values
                lo_zs = models_by_q[0.05].predict(zs_h[fcols]) + zs_current
                hi_zs = models_by_q[0.95].predict(zs_h[fcols]) + zs_current
                y_zs = zs_h[target_col].values
                provinces_zs = zs_h["province_en"].values

                lo_zs_single, hi_zs_single = lo_zs - baseline_qhat[h], hi_zs + baseline_qhat[h]
                q_hat_zs_vol = apply_volatility_group(provinces_zs, zero_shot_to_group, vol_qhats, baseline_qhat[h])
                lo_zs_vol, hi_zs_vol = lo_zs - q_hat_zs_vol, hi_zs + q_hat_zs_vol

                for province in np.unique(provinces_zs):
                    mask_p = provinces_zs == province
                    all_rows_zeroshot.append({
                        "model": tag, "horizon": h, "province_en": province, "config": "single_qhat",
                        "n": int(mask_p.sum()), "coverage": coverage(lo_zs_single[mask_p], hi_zs_single[mask_p], y_zs[mask_p]),
                    })
                    all_rows_zeroshot.append({
                        "model": tag, "horizon": h, "province_en": province, "config": "volatility_grp",
                        "n": int(mask_p.sum()), "coverage": coverage(lo_zs_vol[mask_p], hi_zs_vol[mask_p], y_zs[mask_p]),
                    })

        print(f"  {tag}: ประเมินครบ 24 horizon x 3 config (single_qhat / flat_3grp / volatility_grp) "
              f"บนจังหวัดเทรน + zero-shot จริง")

    # ---------- (A) สรุปผลบนจังหวัดเทรน ----------
    result_trained = pd.DataFrame(all_rows_trained)
    out_path_a = os.path.join(MODEL_DIR, f"recalibrate_volatility_k1_trained_{TIMESTAMP}.csv")
    result_trained.to_csv(out_path_a, index=False)

    print("\n" + "=" * 100)
    print("(A) สรุป coverage บนจังหวัดเทรน — เทียบ single_qhat / flat_3grp (115) / volatility_grp (ใหม่)")
    print("=" * 100)
    summary_a = result_trained.groupby(["model", "severity_bin", "config"])["coverage"].mean().unstack("config") * 100
    summary_a = summary_a[["single_qhat", "flat_3grp", "volatility_grp"]].round(1)
    for tag in ("lgb", "xgb"):
        print(f"\n--- {tag.upper()} ---")
        print(summary_a.loc[tag].to_string())

    # ---------- (B) สรุปผลบนจังหวัด zero-shot จริง ----------
    result_zs = pd.DataFrame(all_rows_zeroshot)
    out_path_b = os.path.join(MODEL_DIR, f"recalibrate_volatility_k1_zeroshot_{TIMESTAMP}.csv")
    result_zs.to_csv(out_path_b, index=False)

    print("\n" + "=" * 100)
    print("(B) สรุป coverage บนจังหวัด zero-shot จริง — เทียบ single_qhat (baseline entry 119) / volatility_grp (ใหม่)")
    print("=" * 100)
    summary_b = result_zs.groupby(["model", "province_en", "config"])["coverage"].mean().unstack("config") * 100
    summary_b = summary_b.round(1)
    for tag in ("lgb", "xgb"):
        print(f"\n--- {tag.upper()} ---")
        print(summary_b.loc[tag].to_string())

    print(f"\nบันทึกผล (A) ที่: {out_path_a}")
    print(f"บันทึกผล (B) ที่: {out_path_b}")


if __name__ == "__main__":
    main()

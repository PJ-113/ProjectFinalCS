"""
ทำ group-conditional (Mondrian) conformal calibration ซ้ำบน**โมเดล +nb_trend+tuned**
(_20260919_135655, 20 feature: k=1 + fire_hotspot_count_100km_24h, CHANGELOG 127) — เช็คว่า
config ที่ดีที่สุดเดิม (flat_3grp, entry 115) ช่วยแก้จุดที่ entry 128 พบว่า XGB coverage ที่ >p90
แย่ลง (80.8%->79.5%) ตอนใช้ single q_hat ได้ไหม — ตรรกะเดียวกับ recalibrate_conformal_configs_k1.py
เป๊ะ เปลี่ยนแค่ MODEL_TAG ให้ตรงกับโมเดลใหม่

predict P05/P50/P95 ต่อ horizon ต่อโมเดลแค่ครั้งเดียว แล้วเอาไปใช้ซ้ำกับทุก config (ไม่ต้อง
predict ใหม่ทุกรอบ) — จัดกลุ่มด้วย predicted P50 เท่านั้น (ไม่ใช่ actual) ตามหลัก conformal
prediction เดิม — ไม่มีการ retrain โมเดลใดๆ ในงานนี้ (ใช้ pkg ที่เทรนไว้แล้วอย่างเดียว)
"""
import importlib.util
import os
import pickle

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(_ROOT, "models")
MODEL_TAG = "20260919_135655"
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

# แต่ละ config = list ของ upper-percentile cutpoints (ascending) นิยามขอบเขตกลุ่ม
# เช่น [90] -> 2 กลุ่ม: normal(<p90), high(>=p90)
#      [85, 95] -> 3 กลุ่ม: normal(<p85), high(p85-p95), extreme(>=p95)
CONFIGS = {
    "2grp_top10": [90],
    "2grp_top5": [95],
    "3grp_top5_15": [85, 95],
}
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
    """คืน group index (0=normal, 1=..., len(thresholds)=สุดท้าย/รุนแรงสุด) ตาม threshold ที่ให้มา
    (ค่า threshold เป็น µg/m³ จริง ไม่ใช่ percentile — คำนวณจาก calib ก่อนแล้วส่งเข้ามา)"""
    group = np.zeros(len(p50_values), dtype=int)
    for t in thresholds:
        group += (p50_values >= t).astype(int)
    return group


def compute_group_qhats(p50_calib, lo_calib, hi_calib, y_calib, thresholds_pctl):
    thresholds_val = [np.percentile(p50_calib, p) for p in thresholds_pctl]
    group_calib = assign_group(p50_calib, thresholds_val)
    n_groups = len(thresholds_pctl) + 1
    q_hats = {}
    sizes = {}
    for g in range(n_groups):
        mask = group_calib == g
        n = int(mask.sum())
        sizes[g] = n
        if n < MIN_GROUP_SIZE:
            q_hats[g] = None
            continue
        scores = np.maximum(lo_calib[mask] - y_calib[mask], y_calib[mask] - hi_calib[mask])
        q_level = min(np.ceil((n + 1) * 0.9) / n, 1.0)
        q_hats[g] = float(np.quantile(scores, q_level))
    return thresholds_val, q_hats, sizes


def main():
    train_mod = load_train_module()
    print(f"โหลดข้อมูล: {train_mod.DATA_PATH}")
    df = pd.read_csv(train_mod.DATA_PATH)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    df = df[~df["is_zero_shot"]].reset_index(drop=True)

    train_df, calib_df, test_df = train_mod.split_train_calib_test(df)
    print(f"train={len(train_df)}  calib={len(calib_df)}  test={len(test_df)}  (โมเดล k=1 tag={MODEL_TAG})")

    all_rows = []
    size_rows = []

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

            test_current = test_h["pm25"].values
            p50_test = models_by_q[0.5].predict(test_h[fcols]) + test_current
            lo_test = models_by_q[0.05].predict(test_h[fcols]) + test_current
            hi_test = models_by_q[0.95].predict(test_h[fcols]) + test_current
            y_test = test_h[target_col].values

            bins = [-0.01, np.percentile(y_test, 50), np.percentile(y_test, 75),
                    np.percentile(y_test, 90), y_test.max() + 0.01]
            labels = ["<=p50", "p50-p75", "p75-p90", ">p90"]
            severity_bin = pd.cut(y_test, bins=bins, labels=labels)

            lo_single = lo_test - baseline_qhat[h]
            hi_single = hi_test + baseline_qhat[h]

            configs_lo_hi = {"single_qhat": (lo_single, hi_single)}
            for config_name, thresholds_pctl in CONFIGS.items():
                thresholds_val, q_hats, sizes = compute_group_qhats(
                    p50_calib, lo_calib, hi_calib, y_calib, thresholds_pctl)
                if h == 1:
                    for g, n in sizes.items():
                        size_rows.append({"model": tag, "config": config_name, "group": g, "n_calib": n})
                group_test = assign_group(p50_test, thresholds_val)
                q_hat_per_row = np.array([q_hats[g] if q_hats[g] is not None else baseline_qhat[h]
                                           for g in group_test])
                configs_lo_hi[config_name] = (lo_test - q_hat_per_row, hi_test + q_hat_per_row)

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

        print(f"  {tag}: ประเมินครบ 24 horizon x {len(CONFIGS)+1} config")

    result_df = pd.DataFrame(all_rows)
    out_path = os.path.join(MODEL_DIR, f"recalibrate_configs_k1_{TIMESTAMP}.csv")
    result_df.to_csv(out_path, index=False)
    print(f"\nบันทึกผลดิบที่: {out_path}")

    size_df = pd.DataFrame(size_rows)
    print("\nขนาดกลุ่ม calibration ต่อ config (ที่ horizon=1):")
    print(size_df.to_string(index=False))

    print("\n" + "=" * 100)
    print("สรุป: coverage เฉลี่ยข้าม 24 horizon ต่อ severity bin x config (target = 90%) [โมเดล k=1]")
    print("=" * 100)
    summary = result_df.groupby(["model", "severity_bin", "config"])["coverage"].mean().unstack("config") * 100
    config_order = ["single_qhat"] + list(CONFIGS.keys())
    summary = summary[config_order].round(1)
    for tag in ("lgb", "xgb"):
        print(f"\n--- {tag.upper()} ---")
        print(summary.loc[tag])


if __name__ == "__main__":
    main()

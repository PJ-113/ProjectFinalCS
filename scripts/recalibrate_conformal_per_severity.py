"""
แก้ปัญหา coverage ตกช่วงฝุ่นสูง (task A, CHANGELOG 89) ด้วย group-conditional (Mondrian) conformal
calibration — แทนที่จะใช้ q_hat ค่าเดียวรวมทุกอย่างต่อ horizon (วิธีเดิมใน train_multiprovince_models.py),
แบ่ง calibration split เป็น 2 กลุ่มตาม predicted P50 ("high" = top 10%, "normal" = ที่เหลือ)
แล้วคำนวณ q_hat แยกต่อกลุ่ม — จัดกลุ่มด้วยค่าที่ทำนายไว้ล่วงหน้าเท่านั้น (ไม่ใช่ค่าจริง) เพื่อไม่ให้
ผิดหลัก conformal prediction (grouping ต้องมาจากข้อมูลที่รู้ก่อนเห็นผลจริง)

ใช้กับโมเดลใหม่ล่าสุด (_20260914_211416, 19 feature, spatial-lag+ตัด time_cyclical) ไม่ retrain
quantile model ใหม่ แค่เปลี่ยนวิธีคำนวณ q_hat ตอน calibration — ยังไม่แก้ app.py ในรอบนี้
"""
import importlib.util
import os
import pickle

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(_ROOT, "models")
MODEL_TAG = "20260914_211416"
HIGH_GROUP_PERCENTILE = 90  # top 10% ของ predicted P50 -> กลุ่ม "high" (ตรงจุดที่ task A พบปัญหา)
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")


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


def main():
    train_mod = load_train_module()
    print(f"โหลดข้อมูล: {train_mod.DATA_PATH}")
    df = pd.read_csv(train_mod.DATA_PATH)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    df = df[~df["is_zero_shot"]].reset_index(drop=True)

    train_df, calib_df, test_df = train_mod.split_train_calib_test(df)
    print(f"train={len(train_df)}  calib={len(calib_df)}  test={len(test_df)}")

    all_rows = []
    group_size_rows = []

    for tag in ("lgb", "xgb"):
        pkg = load_model_pkg(tag)
        baseline_qhat = load_baseline_qhat(tag)
        print(f"\n{'='*90}\n{tag.upper()}\n{'='*90}")

        for h in range(1, 25):
            target_col = f"pm25_t+{h}"
            calib_h = calib_df.dropna(subset=[target_col])
            test_h = test_df.dropna(subset=[target_col])
            models_by_q = pkg["models"][h]

            # --- calibration: จัดกลุ่มด้วย predicted P50 ของ calib เอง ---
            calib_current = calib_h["pm25"].values
            p50_calib = models_by_q[0.5].predict(calib_h[pkg["feature_cols"]]) + calib_current
            threshold = np.percentile(p50_calib, HIGH_GROUP_PERCENTILE)

            lo_calib = models_by_q[0.05].predict(calib_h[pkg["feature_cols"]]) + calib_current
            hi_calib = models_by_q[0.95].predict(calib_h[pkg["feature_cols"]]) + calib_current
            y_calib = calib_h[target_col].values

            is_high_calib = p50_calib >= threshold
            q_hats = {}
            for group_name, mask in (("normal", ~is_high_calib), ("high", is_high_calib)):
                n = mask.sum()
                group_size_rows.append({"model": tag, "horizon": h, "group": group_name, "n_calib": int(n)})
                if n < 20:
                    q_hats[group_name] = None
                    continue
                scores = np.maximum(lo_calib[mask] - y_calib[mask], y_calib[mask] - hi_calib[mask])
                q_level = min(np.ceil((n + 1) * 0.9) / n, 1.0)
                q_hats[group_name] = float(np.quantile(scores, q_level))

            # --- ประเมินบน test: จัดกลุ่มด้วย threshold ที่ได้จาก calib เท่านั้น ---
            test_current = test_h["pm25"].values
            p50_test = models_by_q[0.5].predict(test_h[pkg["feature_cols"]]) + test_current
            lo_test = models_by_q[0.05].predict(test_h[pkg["feature_cols"]]) + test_current
            hi_test = models_by_q[0.95].predict(test_h[pkg["feature_cols"]]) + test_current
            y_test = test_h[target_col].values
            is_high_test = p50_test >= threshold

            q_hat_per_row = np.where(is_high_test, q_hats["high"] or 0.0, q_hats["normal"] or 0.0)
            lo_group = lo_test - q_hat_per_row
            hi_group = hi_test + q_hat_per_row

            q_hat_single = baseline_qhat[h]
            lo_single = lo_test - q_hat_single
            hi_single = hi_test + q_hat_single

            # severity bin ของค่าจริง (เหมือน task A) เพื่อเทียบ >p90 โดยตรง
            bins = [-0.01, np.percentile(y_test, 50), np.percentile(y_test, 75),
                    np.percentile(y_test, 90), y_test.max() + 0.01]
            labels = ["<=p50", "p50-p75", "p75-p90", ">p90"]
            severity_bin = pd.cut(y_test, bins=bins, labels=labels)

            for label in labels:
                mask_b = np.asarray(severity_bin == label)
                if mask_b.sum() == 0:
                    continue
                all_rows.append({
                    "model": tag, "horizon": h, "severity_bin": label, "n": int(mask_b.sum()),
                    "coverage_single_qhat": coverage(lo_single[mask_b], hi_single[mask_b], y_test[mask_b]),
                    "coverage_group_conditional": coverage(lo_group[mask_b], hi_group[mask_b], y_test[mask_b]),
                    "q_hat_single": q_hat_single, "q_hat_normal": q_hats["normal"], "q_hat_high": q_hats["high"],
                    "threshold_p50": float(threshold),
                })
            all_rows.append({
                "model": tag, "horizon": h, "severity_bin": "ALL", "n": int(len(y_test)),
                "coverage_single_qhat": coverage(lo_single, hi_single, y_test),
                "coverage_group_conditional": coverage(lo_group, hi_group, y_test),
                "q_hat_single": q_hat_single, "q_hat_normal": q_hats["normal"], "q_hat_high": q_hats["high"],
                "threshold_p50": float(threshold),
            })

        print(f"  {tag}: ประเมินครบ 24 horizon")

    result_df = pd.DataFrame(all_rows)
    out_path = os.path.join(MODEL_DIR, f"recalibrate_per_severity_{TIMESTAMP}.csv")
    result_df.to_csv(out_path, index=False)
    print(f"\nบันทึกผลดิบที่: {out_path}")

    group_size_df = pd.DataFrame(group_size_rows)
    print("\nขนาดกลุ่ม calibration (ตรวจว่าไม่เล็กเกินไป):")
    print(group_size_df.groupby(["model", "group"])["n_calib"].agg(["mean", "min", "max"]).round(1))

    print("\n" + "=" * 90)
    print("สรุป: coverage เฉลี่ยข้าม 24 horizon ต่อ severity bin (single q_hat vs group-conditional)")
    print("=" * 90)
    summary = result_df.groupby(["model", "severity_bin"])[["coverage_single_qhat", "coverage_group_conditional"]].mean()
    summary = (summary * 100).round(1)
    print(summary)


if __name__ == "__main__":
    main()

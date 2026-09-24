"""
ส่งออกค่าสัมประสิทธิ์ของวิธีเติมข้อมูล H-lite (Ridge-lite + จุดสลับ + ตาราง q(k,h) ของช่วงความเชื่อมั่นเมื่อ input ถูกเติมค่า) เป็น JSON เล็กๆ
ให้ app.py อ่านตอนเริ่ม (IMPUTATION_CONFIG_PATH) — ไม่ต้องพึ่ง sklearn/pickle ตอนใช้งานจริง

อ่าน: models/imputation_tuned_params_*.json (จูนบน calib โดย experiment_imputation_accuracy.py) และ
      models/imputation_interval_qhat_*.json (recalibration โดย experiment_imputation_interval_recalibration.py, method = H-lite)
fit Ridge-lite ซ้ำด้วย seed/จำนวนแถว/พารามิเตอร์ชุดเดียวกับตอนทดลอง (ผลต้องเหมือนกันเป๊ะ — ตรวจด้วยการเทียบค่าที่เติมกับ impute_trailing "HL")
เขียน: models/imputation_h_lite_<ts>.json  (ไฟล์เล็ก เก็บใน git ได้; ห้ามมีคีย์/ความลับ)
รัน: python scripts/export_imputation_h_lite.py
"""
import glob
import importlib.util
import json
import os

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("ea", os.path.join(_ROOT, "scripts", "experiment_imputation_accuracy.py"))
ea = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ea)

TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
FEATURES = ["x0", "nb_now", "dnb", "nb_0", "j", "log1p_j", "hour_sin", "hour_cos"]  # ต้องตรง ea.matrix_trailing_lite และ app.IMPUTATION_FEATURES


def latest(pattern):
    files = sorted(f for f in glob.glob(os.path.join(ea.MODEL_DIR, pattern)) if "_quick" not in f)
    if not files:
        raise SystemExit(f"ไม่พบ {pattern}")
    return files[-1]


def apply_json_formula(cfg, F):
    """สูตรเดียวกับที่ app.py ใช้ (numpy ล้วน) เพื่อตรวจว่าตรงกับ ea.impute_trailing 'HL'"""
    mean, scale, coef = (np.array(cfg[k]) for k in ("scaler_mean", "scaler_scale", "coef"))
    X = np.column_stack([F["x0"], F["nb_now"], F["dnb"], F["nb_0"], F["j"], np.log1p(F["j"]), F["hs"], F["hc"]])
    ridge = F["x0"] + cfg["intercept"] + ((X - mean) / scale) @ coef
    idw = F["x0"] + F["dnb"]
    v = np.where(F["j"] <= cfg["switch_j"], idw, ridge)
    v = np.where(F["has_nb"], v, F["x0"])  # ไม่มีเพื่อนบ้านที่ใช้ได้ -> Persistence
    return np.maximum(v, 0.0)


def main():
    tuned_path, qhat_path = latest("imputation_tuned_params_2*.json"), latest("imputation_interval_qhat_2*.json")
    tuned = json.load(open(tuned_path, encoding="utf-8"))["tune"]
    qhat = json.load(open(qhat_path, encoding="utf-8"))
    assert qhat.get("method") == "H-lite", f"{os.path.basename(qhat_path)} ไม่ใช่ตาราง q ของ H-lite — รัน recalibration ใหม่"
    for key in ("switch_j_lite", "ridge_alpha_lite"):
        assert key in tuned, f"ไฟล์พารามิเตอร์ไม่มี {key} — รัน experiment_imputation_accuracy.py ใหม่"

    P = ea.load_panels(need_full_dataset=False)
    model = ea.fit_trailing(P, np.random.default_rng(ea.SEED), 60000, tuned, fast=True)
    rl = model["fitted"]["ridge_lite"]
    cfg = {
        "method": "H-lite",
        "created": pd.Timestamp.now().isoformat(),
        "source_tuned_params": os.path.basename(tuned_path),
        "source_qhat": os.path.basename(qhat_path),
        "features": FEATURES,
        "scaler_mean": [float(x) for x in rl["sc"].mean_],
        "scaler_scale": [float(x) for x in rl["sc"].scale_],
        "coef": [float(x) for x in rl["model"].coef_],
        "intercept": float(rl["model"].intercept_),
        "ridge_alpha": float(tuned["ridge_alpha_lite"]),
        "switch_j": int(tuned["switch_j_lite"]),
        "n_neighbors_pool": ea.N_NEIGHBORS,
        "max_valid_neighbors": 3,
        "neighbor_weight": "1 / max(distance_km, 1)^2",
        "target": "delta = pm25(t0+j) - pm25(t0); IDW ส่วนต่างเมื่อ j <= switch_j ไม่งั้น Ridge; ไม่มีเพื่อนบ้าน -> Persistence",
        "q_hat_imputed": qhat["q_hat_imputed"],
        "q_hat_note": qhat["note"],
    }

    # ---- ตรวจสูตร JSON = วิธี HL ในสคริปต์ทดลอง (บนตัวอย่าง test จริง) ----
    rng = np.random.default_rng(0)
    for j in (1, 3, 6, 7, 12, 24, 48):
        p_, t_ = ea.sample_trailing(P, "test", j, 30, rng)
        F = ea.make_feats(P, p_, t_, np.full(len(p_), j))
        ref = ea.impute_trailing(P, model, F, ("HL",))["HL"]
        got = apply_json_formula(cfg, F)
        assert np.allclose(ref, got, atol=1e-9), f"สูตร JSON ไม่ตรง HL ที่ j={j}: max diff {np.abs(ref - got).max()}"
    print("ตรวจแล้ว: สูตรจาก JSON ตรงกับ impute_trailing('HL') ที่ j = 1,3,6,7,12,24,48")

    out = os.path.join(ea.MODEL_DIR, f"imputation_h_lite_{TIMESTAMP}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=1)
    print(f"บันทึก {out}\n  switch_j={cfg['switch_j']}  alpha={cfg['ridge_alpha']}  coef={np.round(cfg['coef'], 3).tolist()}")


if __name__ == "__main__":
    main()

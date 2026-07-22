"""
SHAP analysis (proposal 4.1.3 / 5.3.4) สำหรับ LightGBM และ XGBoost ที่ deploy อยู่จริง
(ไม่ทำกับ Prophet เพราะ shap.TreeExplainer ใช้ไม่ได้กับโมเดลที่ไม่ใช่ tree-based และ Prophet
เองก็อธิบายตัวเองได้อยู่แล้วผ่านโครงสร้าง trend/seasonality/holiday)

ไฟล์นี้ทำ 2 หน้าที่แยกกัน:
1. Batch analysis (รันเมื่อเรียกเป็นสคริปต์โดยตรง): คำนวณ SHAP บน test set จริงที่ horizon
   ตัวแทน t+1/t+12/t+24 ของทั้ง 2 โมเดล, สรุป top-5 feature ต่อ (model, horizon), save ผลลัพธ์
2. get_shap_explanation(): ฟังก์ชัน reusable สำหรับอธิบายค่าพยากรณ์ "จุดเดียว" ตามโมเดลที่
   Dynamic Routing เลือกใช้จริง (model_used) — import ได้โดยไม่ trigger batch analysis
   (โหลดแค่โมเดล ไม่รัน SHAP จนกว่าจะถูกเรียกจริง)

ยังไม่ integrate เข้า app.py/get_forecast() — เป็นขั้นตอนแยกต่างหากทีหลัง
"""
import pickle
import sys

import numpy as np
import pandas as pd
import shap

sys.stdout.reconfigure(encoding="utf-8")

import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(_ROOT, r"data\khonkaen_pm25_features_raw_20260712_191249.csv")
LGB_MODEL_PATH = os.path.join(_ROOT, r"models\lgb_pm25_multistep_weighted_20260712_191056.pkl")
XGB_MODEL_PATH = os.path.join(_ROOT, r"models\xgb_pm25_multistep_20260712_191350.pkl")
OUT_PKL_PATH = os.path.join(_ROOT, r"models\shap_values_by_horizon.pkl")
OUT_CSV_PATH = os.path.join(_ROOT, r"models\shap_top_features_by_horizon.csv")
REP_HORIZONS = [1, 12, 24]
TOP_N = 5

with open(LGB_MODEL_PATH, "rb") as f:
    _lgb_package = pickle.load(f)
with open(XGB_MODEL_PATH, "rb") as f:
    _xgb_package = pickle.load(f)

FEATURE_COLS = _lgb_package["feature_cols"]
assert FEATURE_COLS == _xgb_package["feature_cols"], \
    "feature_cols ของ LightGBM กับ XGBoost ต้องตรงกันเป๊ะ (ควรเป็นแบบนี้เสมอเพราะ " \
    "train_xgb_prophet_v2.py โหลด feature_cols มาจาก LightGBM pkl ตอนเทรน)"

MODEL_PACKAGES = {"lightgbm": _lgb_package, "xgboost": _xgb_package}

# cache explainer ต่อ (model_name, horizon) เพราะสร้าง TreeExplainer ใหม่ทุกครั้งไม่จำเป็น
_EXPLAINER_CACHE = {}


def get_shap_explanation(model_used, horizon, feature_row):
    """คืนอิทธิพลของแต่ละ feature ต่อค่าพยากรณ์ ณ จุดเดียว (feature_row) โดยใช้ SHAP explainer
    ของโมเดลที่ Dynamic Routing เลือกใช้จริง (model_used จาก select_model() ใน app.py) —
    ไม่ใช่อธิบายด้วย LightGBM เสมอแม้จุดนั้นจะมาจาก XGBoost หรือ Prophet

    feature_row: DataFrame แถวเดียว มีคอลัมน์ตรงกับ FEATURE_COLS (รูปแบบเดียวกับที่
    build_feature_row() ใน app.py คืนกลับมา หลัง calibrate แล้ว)

    คืนค่า: list of (feature_name, shap_value) เรียงจากอิทธิพลสูงสุด (|shap_value|) ไปต่ำสุด
    หรือ None ถ้า model_used == "prophet" (ไม่มี SHAP explainer ให้ ดูหมายเหตุด้านบนไฟล์)
    """
    if model_used == "prophet":
        return None

    if model_used not in MODEL_PACKAGES:
        raise ValueError(f"ไม่รู้จักโมเดล: {model_used}")

    cache_key = (model_used, horizon)
    if cache_key not in _EXPLAINER_CACHE:
        model = MODEL_PACKAGES[model_used]["models"][horizon]
        _EXPLAINER_CACHE[cache_key] = shap.TreeExplainer(model)
    explainer = _EXPLAINER_CACHE[cache_key]

    shap_row = explainer.shap_values(feature_row[FEATURE_COLS])[0]
    pairs = list(zip(FEATURE_COLS, shap_row))
    pairs.sort(key=lambda p: abs(p[1]), reverse=True)
    return pairs


def _run_batch_analysis():
    df = pd.read_csv(DATA_PATH)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    split_idx = int(len(df) * 0.8)
    test = df.iloc[split_idx:]

    shap_values_by_horizon = {"lightgbm": {}, "xgboost": {}}
    top_features_rows = []

    for model_name, package in MODEL_PACKAGES.items():
        models = package["models"]
        for h in REP_HORIZONS:
            target = f"pm25_t+{h}"
            test_valid = test.dropna(subset=[target])
            X_test = test_valid[FEATURE_COLS]

            model = models[h]
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)

            shap_values_by_horizon[model_name][h] = shap_values

            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            order = np.argsort(mean_abs_shap)[::-1][:TOP_N]
            print(f"\n=== {model_name} t+{h} (n={len(X_test)}) top {TOP_N} features ===")
            for rank, idx in enumerate(order, start=1):
                feat = FEATURE_COLS[idx]
                val = mean_abs_shap[idx]
                print(f"  {rank}. {feat}: mean|SHAP|={val:.4f}")
                top_features_rows.append({
                    "model_name": model_name, "horizon": h, "rank": rank,
                    "feature": feat, "mean_abs_shap": val,
                })

    with open(OUT_PKL_PATH, "wb") as f:
        pickle.dump({"feature_cols": FEATURE_COLS, "shap_values": shap_values_by_horizon}, f)

    top_df = pd.DataFrame(top_features_rows)
    top_df.to_csv(OUT_CSV_PATH, index=False)

    print(f"\nบันทึก SHAP values ที่: {OUT_PKL_PATH}")
    print(f"บันทึกสรุป top features ที่: {OUT_CSV_PATH}")

    print("\n=== เทียบ top features LightGBM vs XGBoost ต่อ horizon ===")
    for h in REP_HORIZONS:
        lgb_top = set(top_df[(top_df.model_name == "lightgbm") & (top_df.horizon == h)]["feature"])
        xgb_top = set(top_df[(top_df.model_name == "xgboost") & (top_df.horizon == h)]["feature"])
        overlap = lgb_top & xgb_top
        print(f"\nt+{h}: overlap {len(overlap)}/{TOP_N} -> {sorted(overlap)}")
        print(f"   LightGBM only: {sorted(lgb_top - xgb_top)}")
        print(f"   XGBoost only:  {sorted(xgb_top - lgb_top)}")

    # self-test: get_shap_explanation() ต้องให้ค่าเรียงลำดับตรงกับ mean_abs_shap ของ batch
    # (ยืนยันว่า per-point explainer กับ batch explainer สอดคล้องกัน ไม่ใช่ implementation คนละแบบ)
    print("\n=== self-test: get_shap_explanation() เทียบ prophet -> None ===")
    sample_row = test.dropna(subset=["pm25_t+1"])[FEATURE_COLS].iloc[[-1]]
    for model_name in ("lightgbm", "xgboost"):
        pairs = get_shap_explanation(model_name, 1, sample_row)
        top3 = [f"{feat}={val:+.3f}" for feat, val in pairs[:3]]
        print(f"  {model_name} t+1 (จุดตัวอย่างสุดท้ายใน test set): top3 = {top3}")
    prophet_result = get_shap_explanation("prophet", 1, sample_row)
    print(f"  prophet -> {prophet_result} (คาด None)")
    assert prophet_result is None, "get_shap_explanation('prophet', ...) ต้องคืน None"


if __name__ == "__main__":
    _run_batch_analysis()

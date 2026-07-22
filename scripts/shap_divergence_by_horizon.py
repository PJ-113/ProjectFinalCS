"""
หา horizon ที่ SHAP feature importance ของ LightGBM กับ XGBoost (โมเดล deploy จริงทั้งคู่) ต่างกัน
ชัดเจนที่สุด — ใช้เลือก representative horizon สำหรับ weight grid search ของ
scripts/retrain_xgb_weighted.py แทนการ copy t+1/t+12/t+24 จาก LightGBM ตรงๆ (จุดที่ LightGBM
เลือกไว้สะท้อนพฤติกรรมของ LightGBM เอง ไม่ได้การันตีว่าเป็นจุดที่มีความหมายสำหรับ XGBoost ด้วย)

วิธีวัดความต่าง: คำนวณ mean|SHAP| ต่อ feature ต่อโมเดลต่อ horizon (ทั้ง 24 horizon ไม่ใช่แค่ 3
จุดแบบ shap_analysis.py) normalize เป็นสัดส่วนรวม 1 ต่อ horizon แล้ววัด L1 distance ระหว่าง
สัดส่วนของสองโมเดล — ยิ่งสูงยิ่งแปลว่าสองโมเดลให้น้ำหนัก feature ต่างกันมากที่ horizon นั้น
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
XGB_MODEL_PATH = os.path.join(_ROOT, r"models\xgb_pm25_multistep_20260712_191350.pkl")  # baseline unweighted ตัวปัจจุบัน

with open(LGB_MODEL_PATH, "rb") as f:
    lgb_pkg = pickle.load(f)
with open(XGB_MODEL_PATH, "rb") as f:
    xgb_pkg = pickle.load(f)

FEATURE_COLS = lgb_pkg["feature_cols"]
assert FEATURE_COLS == xgb_pkg["feature_cols"]

df = pd.read_csv(DATA_PATH)
df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
split_idx = int(len(df) * 0.8)
test = df.iloc[split_idx:]

rows = []
top_features_by_h = {}
for h in range(1, 25):
    target = f"pm25_t+{h}"
    test_valid = test.dropna(subset=[target])
    X_test = test_valid[FEATURE_COLS]

    lgb_shap = shap.TreeExplainer(lgb_pkg["models"][h]).shap_values(X_test)
    xgb_shap = shap.TreeExplainer(xgb_pkg["models"][h]).shap_values(X_test)

    lgb_importance = np.abs(lgb_shap).mean(axis=0)
    xgb_importance = np.abs(xgb_shap).mean(axis=0)

    # normalize เป็นสัดส่วนรวม 1 ต่อโมเดลต่อ horizon กันมาตราส่วนรวม (magnitude) ของ SHAP ต่างกัน
    # ระหว่างโมเดลมากวน — สนใจแค่ "สัดส่วนน้ำหนักที่ให้แต่ละ feature" ว่าต่างกันแค่ไหน
    lgb_prop = lgb_importance / lgb_importance.sum()
    xgb_prop = xgb_importance / xgb_importance.sum()

    l1_dist = float(np.abs(lgb_prop - xgb_prop).sum())  # ช่วง 0 (เหมือนกันเป๊ะ) ถึง 2 (ไม่ overlap เลย)

    lgb_top_feat = FEATURE_COLS[int(np.argmax(lgb_prop))]
    xgb_top_feat = FEATURE_COLS[int(np.argmax(xgb_prop))]
    top_features_by_h[h] = (lgb_top_feat, xgb_top_feat)

    rows.append({"horizon": h, "l1_dist": l1_dist, "lgb_top_feature": lgb_top_feat, "xgb_top_feature": xgb_top_feat})
    print(f"t+{h:2d}: L1 dist={l1_dist:.3f}  LightGBM top={lgb_top_feat:<20s} XGBoost top={xgb_top_feat}")

out_df = pd.DataFrame(rows).sort_values("l1_dist", ascending=False)
print("\n=== เรียงจาก horizon ที่ SHAP importance ของสองโมเดลต่างกันมากสุด ===")
print(out_df.to_string(index=False))

top3 = out_df.head(3)["horizon"].tolist()
print(f"\nHorizon ที่ต่างกันมากสุด 3 อันดับแรก: {sorted(top3)}")

out_path = os.path.join(_ROOT, r"models\shap_divergence_by_horizon.csv")
out_df.to_csv(out_path, index=False)
print(f"บันทึกที่: {out_path}")

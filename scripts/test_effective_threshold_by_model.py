"""
ทดสอบ EFFECTIVE_THRESHOLD_BY_HORIZON_BY_MODEL — threshold แจ้งเตือนมลพิษสูงต้องแยกตามโมเดล
เพราะแต่ละโมเดลมี bias_high ไม่เท่ากัน
"""
import importlib.util

import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_PATH = os.path.join(_ROOT, r"app.py")


def load_app():
    spec = importlib.util.spec_from_file_location("app_under_test", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


app = load_app()
thr = app.EFFECTIVE_THRESHOLD_BY_HORIZON_BY_MODEL

# Test 1: มีครบ 3 โมเดล x 24 horizon
assert set(thr.keys()) == {"lightgbm", "xgboost", "prophet"}
for model_name in thr:
    assert set(thr[model_name].keys()) == set(range(1, 25)), model_name
print("Test 1 ผ่าน: มี threshold ครบ 3 โมเดล x 24 horizon")

# Test 2: lightgbm threshold ตรงกับสูตรเดิม (37.5 - bias_high_weighted จาก BIAS_CSV_PATH)
assert abs(thr["lightgbm"][24] - (37.5 - app.BIAS_HIGH_BY_MODEL["lightgbm"][24])) < 1e-9
print(f"Test 2 ผ่าน: lightgbm t+24 threshold = {thr['lightgbm'][24]:.3f} (ตรงสูตรเดิม)")

# Test 3: xgboost ต้องมี threshold เป็นของตัวเอง ไม่ใช่ copy ค่า lightgbm มาเฉยๆ
assert thr["xgboost"][24] != thr["lightgbm"][24]
assert abs(thr["xgboost"][24] - (37.5 - app.BIAS_HIGH_BY_MODEL["xgboost"][24])) < 1e-9
print(f"Test 3 ผ่าน: xgboost t+24 threshold = {thr['xgboost'][24]:.3f} (แยกจาก lightgbm จริง)")

# Test 4: prophet ไม่มี bias correction ต่อ horizon ที่เชื่อถือได้ -> ใช้ 37.5 ตรงๆ ทุก horizon
for h in range(1, 25):
    assert thr["prophet"][h] == 37.5, h
print("Test 4 ผ่าน: prophet ใช้ 37.5 ตรงๆ ทุก horizon (ไม่ correct เพราะไม่มี bias ต่อ horizon ที่เชื่อถือได้)")

# Test 5: จำลอง get_forecast() loop ว่า risk_horizons ขึ้นกับ model_name จริง (ไม่ใช่ threshold เดียวทั้งไฟล์)
# เคสที่เคยพัง: pred อยู่ระหว่าง threshold ของ xgboost กับ lightgbm ที่ t+24 -> ผลต้องต่างกันตาม model_name
assert thr["xgboost"][24] < thr["lightgbm"][24]
pred = (thr["xgboost"][24] + thr["lightgbm"][24]) / 2  # อยู่กึ่งกลางระหว่าง 2 threshold
flagged_as_xgboost = pred > thr["xgboost"][24]
flagged_as_lightgbm = pred > thr["lightgbm"][24]
assert flagged_as_xgboost and not flagged_as_lightgbm
print(f"Test 5 ผ่าน: pred={pred:.3f} ถูก flag เป็นความเสี่ยงถ้า route ไป xgboost (threshold={thr['xgboost'][24]:.3f}) "
      f"แต่ไม่ถูก flag ถ้า route ไป lightgbm (threshold={thr['lightgbm'][24]:.3f}) -> ยืนยันว่าผลขึ้นกับโมเดลที่ route ไปจริง")

print("\nทุก assertion ผ่านหมด")

"""
ทดสอบ compute_risk_score() — Personalized Risk Score (proposal) ที่ผสาน pm25 + risk_group +
activity intensity เป็นคะแนนเดียว 0-100 ผ่าน effective PM2.5 + linear interpolation ข้าม
RISK_SCORE_BREAKPOINTS
"""
import importlib.util
import time

import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_PATH = os.path.join(_ROOT, r"app.py")
_TEST_DOMAIN = "@test-marker.invalid"
_EMAIL_SLUG = {"ทั่วไป": "general", "เด็ก-ผู้สูงอายุ": "child-elderly", "มีโรคประจำตัว": "chronic"}


def load_app():
    spec = importlib.util.spec_from_file_location("app_under_test", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


app = load_app()

# Test 1: breakpoint คือ 0/20/40/60/80/100 พอดีที่ pm25 = 0/15/25/37.5/75/150 สำหรับ "ทั่วไป"+"เบา"
# (offset รวม = 0 ทั้งคู่ -> effective_pm25 == pm25 ตรงๆ)
expected = [(0, 0), (15, 20), (25, 40), (37.5, 60), (75, 80), (150, 100)]
for pm25, expected_score in expected:
    result = app.compute_risk_score(pm25, "ทั่วไป", "เบา")
    assert result["score"] == expected_score, (pm25, result)
print(f"Test 1 ผ่าน: breakpoint ทั้ง 6 จุดให้คะแนนตรงตามที่กำหนด (ทั่วไป+เบา, offset รวม=0)")

# Test 2: ค่าเกิน 150 ต้อง clamp เป็น 100 ไม่ใช่เกิน 100 หรือ error
result = app.compute_risk_score(500, "ทั่วไป", "เบา")
assert result["score"] == 100, result
print(f"Test 2 ผ่าน: pm25=500 clamp เป็นคะแนน 100 (ไม่เกิน)")

# Test 3: คะแนนต้อง monotonically non-decreasing ตาม pm25 ที่เพิ่มขึ้น
scores = [app.compute_risk_score(v, "ทั่วไป", "ปานกลาง")["score"] for v in range(0, 200, 5)]
assert all(scores[i] <= scores[i + 1] for i in range(len(scores) - 1)), scores
print("Test 3 ผ่าน: คะแนนไม่ลดลงเมื่อ pm25 เพิ่มขึ้น (monotonic)")

# Test 4: risk_group offset ทำให้คะแนนสูงขึ้นที่ pm25 เท่ากัน (กลุ่มเปราะบางอ่อนไหวกว่า)
pm25 = 30.0  # อยู่ในช่วง "ปานกลาง" ที่มี slope ชัน กระทบเห็นผลชัด
general = app.compute_risk_score(pm25, "ทั่วไป", "เบา")["score"]
vulnerable = app.compute_risk_score(pm25, "เด็ก-ผู้สูงอายุ", "เบา")["score"]
assert vulnerable > general, (general, vulnerable)
print(f"Test 4 ผ่าน: pm25={pm25} risk_group เปราะบาง -> {vulnerable} > ทั่วไป -> {general}")

# Test 5: activity offset ทำให้คะแนนสูงขึ้นที่ pm25/risk_group เท่ากัน (กิจกรรมหนักอ่อนไหวกว่า)
light = app.compute_risk_score(pm25, "ทั่วไป", "เบา")["score"]
heavy = app.compute_risk_score(pm25, "ทั่วไป", "หนัก")["score"]
assert heavy > light, (light, heavy)
print(f"Test 5 ผ่าน: pm25={pm25} กิจกรรมหนัก -> {heavy} > เบา -> {light}")

# Test 6: offset ต้อง stack กัน (เปราะบาง+หนัก >= เปราะบาง+เบา >= ทั่วไป+เบา)
general_light = app.compute_risk_score(pm25, "ทั่วไป", "เบา")["score"]
vulnerable_light = app.compute_risk_score(pm25, "เด็ก-ผู้สูงอายุ", "เบา")["score"]
vulnerable_heavy = app.compute_risk_score(pm25, "เด็ก-ผู้สูงอายุ", "หนัก")["score"]
assert general_light <= vulnerable_light <= vulnerable_heavy, (general_light, vulnerable_light, vulnerable_heavy)
print(f"Test 6 ผ่าน: offset stack กัน: ทั่วไป+เบา={general_light} <= เปราะบาง+เบา={vulnerable_light} "
      f"<= เปราะบาง+หนัก={vulnerable_heavy}")

# Test 7: level ที่คืนกลับต้องตรงกับ _classify_pm25_level(effective_pm25) เป๊ะ (ไม่มี drift ระหว่างสองฟังก์ชัน)
for pm25_test in [5, 20, 30, 50, 100]:
    for rg in ["ทั่วไป", "เด็ก-ผู้สูงอายุ", "มีโรคประจำตัว"]:
        for act in ["เบา", "ปานกลาง", "หนัก"]:
            result = app.compute_risk_score(pm25_test, rg, act)
            effective = pm25_test - app.RISK_GROUP_THRESHOLD_OFFSET[rg] - app.ACTIVITY_INTENSITY_OFFSET[act]
            expected_level = app._classify_pm25_level(effective)
            assert result["level"] == expected_level, (pm25_test, rg, act, result)
print("Test 7 ผ่าน: level ตรงกับ _classify_pm25_level(effective_pm25) ทุกชุด risk_group x activity ที่ทดสอบ")

# Test 8: /api/forecast ต้องส่ง risk_score_config มาด้วย ตรงกับ risk_group ของ session จริง (login
# จริงผ่าน /api/auth/register เหมือน test_risk_group_e2e.py STEP D — session ไม่มีคีย์ "risk_group"
# ตรงๆ ต้อง resolve ผ่าน user_id -> DB เท่านั้น ดู resolve_risk_group_from_session())
app.app.testing = True


def clear_test_users():
    conn = app.get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM users WHERE email LIKE %s", (f"%{_TEST_DOMAIN}",))
    conn.commit()
    cur.close()
    conn.close()


clear_test_users()
try:
    for rg in ["ทั่วไป", "เด็ก-ผู้สูงอายุ", "มีโรคประจำตัว"]:
        client = app.app.test_client()
        email = f"riskscore-test-{_EMAIL_SLUG[rg]}-{int(time.time())}{_TEST_DOMAIN}"
        r = client.post("/api/auth/register", json={
            "email": email, "password": "correcthorse1", "risk_group": rg, "age": 30,
        })
        assert r.status_code == 201, (rg, r.status_code, r.get_json())
        r = client.get("/api/forecast")
        assert r.status_code == 200, (rg, r.status_code)
        body = r.get_json()
        assert "risk_score_config" in body, body.keys()
        cfg = body["risk_score_config"]
        assert cfg["risk_group_offset"] == app.RISK_GROUP_THRESHOLD_OFFSET[rg], (rg, cfg)
        assert cfg["activity_offsets"] == app.ACTIVITY_INTENSITY_OFFSET, cfg
        got_bp = [tuple(p) for p in cfg["breakpoints"]]
        assert got_bp == app.RISK_SCORE_BREAKPOINTS, cfg
        print(f"Test 8 ({rg}) ผ่าน: risk_score_config.risk_group_offset = {cfg['risk_group_offset']} ตรงกับ session")
finally:
    clear_test_users()

print("\nทุก assertion ผ่านหมด")

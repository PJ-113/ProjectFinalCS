"""
ทดสอบ end-to-end การผูก risk_group เข้ากับ NLG รายวันและ alert threshold

ครอบคลุม:
  - Threshold: effective_threshold_for() ลด threshold ให้กลุ่มเปราะบางถูกต้องตาม
    RISK_GROUP_THRESHOLD_OFFSET และ get_forecast(risk_group) ใช้ค่านี้จริงตอนคำนวณ risk_horizons/
    high_pollution_risk
  - NLG: _build_daily_nlg_prompt() ใส่โทนที่ถูกต้องตาม risk_group จริง (ตรวจจาก prompt ที่ส่งเข้า
    _call_gemini ไม่ใช่ข้อความตอบกลับ เพราะข้อความจริงจาก Gemini ไม่ deterministic พอจะ assert ได้)
    รวมถึง daily_nlg_log คีย์ composite (date_thai, risk_group) แยกกันจริงต่อกลุ่ม (สูงสุด 3
    เรียก Gemini/วัน ไม่ใช่ 24 แบบที่เคย deferred ไว้)
  - Guest (ไม่ login) = "ทั่วไป" เสมอ ทั้งฝั่ง get_forecast() default arg และฝั่ง /api/forecast จริง
    ผ่าน session (ไม่บังคับ login)

Gemini/เวลาถูก mock (เหมือน test_daily_nlg.py) กันเรียกจริงซ้ำหลายครั้ง/รัน และกัน daily_nlg_log
ชนข้อมูลจริง — ใช้ date_thai ปี 2199 (marker เดียวกับ test_daily_nlg.py) ผู้ใช้ทดสอบใช้ email ลงท้าย
@test-marker.invalid (marker เดียวกับ test_auth_routes.py) ลบทิ้งทั้งก่อน/หลังใน finally

หมายเหตุ: STEP B/D เรียก get_forecast()/​/api/forecast จริง (ไม่ mock fetch_recent_pm25/
fetch_current_weather) ตาม convention เดิมของโปรเจกต์ที่ไม่ mock external boundary — ใช้ข้อมูลสดจริง
ดังนั้น risk_horizons อาจว่างเปล่าทั้ง 3 กลุ่มได้ถ้าสภาพอากาศตอนรันสะอาดมาก (ไม่ใช่บั๊ก) — ความถูกต้อง
ของกลไก threshold offset เองถูกพิสูจน์แบบ structural/unit ใน STEP A แยกต่างหาก ไม่ต้องพึ่งข้อมูลสด
"""
import importlib.util
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_PATH = os.path.join(_ROOT, r"app.py")
_TEST_YEAR = "2199"
_TEST_DOMAIN = "@test-marker.invalid"
RISK_GROUPS = ["ทั่วไป", "เด็ก-ผู้สูงอายุ", "มีโรคประจำตัว"]
_EMAIL_SLUG = {"ทั่วไป": "general", "เด็ก-ผู้สูงอายุ": "child-elderly", "มีโรคประจำตัว": "chronic"}


def load_app():
    spec = importlib.util.spec_from_file_location("app_under_test", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clear_test_rows(app):
    conn = app.get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM daily_nlg_log WHERE date_thai LIKE %s", (f"{_TEST_YEAR}-%",))
    cur.execute("DELETE FROM users WHERE email LIKE %s", (f"%{_TEST_DOMAIN}",))
    conn.commit()
    cur.close()
    conn.close()


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    return condition


def main():
    app = load_app()
    app.app.testing = True
    all_ok = True

    clear_test_rows(app)

    captured_prompts = []

    def _fake_call_gemini(prompt):
        captured_prompts.append(prompt)
        return "วันนี้คุณภาพอากาศอยู่ในเกณฑ์ดี สามารถทำกิจกรรมกลางแจ้งได้ตามปกติ"

    app._call_gemini = _fake_call_gemini
    # 05:00 UTC = 12:00 น. ไทย (>= 6, ยังไม่มีแถวของวันนี้เลยสำหรับ marker year 2199)
    app.get_thai_now = lambda: pd.Timestamp(f"{_TEST_YEAR}-07-20T05:00:00", tz="UTC").tz_convert("Asia/Bangkok")

    try:
        print("=" * 80)
        print("STEP A: effective_threshold_for() offset ถูกต้อง (structural, ไม่พึ่งข้อมูลสด)")
        print("=" * 80)
        for model_name in ["lightgbm", "xgboost", "prophet"]:
            for h in [1, 12, 24]:
                base = app.EFFECTIVE_THRESHOLD_BY_HORIZON_BY_MODEL[model_name][h]
                general = app.effective_threshold_for(model_name, h, "ทั่วไป")
                child_elderly = app.effective_threshold_for(model_name, h, "เด็ก-ผู้สูงอายุ")
                chronic = app.effective_threshold_for(model_name, h, "มีโรคประจำตัว")
                all_ok &= check(f"{model_name} h={h}: ทั่วไป == base ({base:.2f})", general == base)
                all_ok &= check(f"{model_name} h={h}: เด็ก-ผู้สูงอายุ == base-10", child_elderly == base - 10.0)
                all_ok &= check(f"{model_name} h={h}: มีโรคประจำตัว == base-10", chronic == base - 10.0)
        unknown = app.effective_threshold_for("lightgbm", 1, "ไม่มีกลุ่มนี้")
        all_ok &= check("risk_group ไม่รู้จัก -> fallback offset=0 (เท่า ทั่วไป)",
                         unknown == app.EFFECTIVE_THRESHOLD_BY_HORIZON_BY_MODEL["lightgbm"][1])

        print("\n" + "=" * 80)
        print("STEP B: get_forecast(risk_group) จริง 3 กลุ่ม + guest default (เรียกติดกัน ข้อมูลสดชุดเดียวกัน)")
        print("=" * 80)
        results = {rg: app.get_forecast(rg) for rg in RISK_GROUPS}
        guest_result = app.get_forecast()  # ไม่ส่ง risk_group เลย = simulate guest

        all_ok &= check("guest (ไม่ส่ง risk_group) -> response risk_group == 'ทั่วไป'",
                         guest_result["risk_group"] == "ทั่วไป")
        all_ok &= check("guest high_pollution_risk == ทั่วไป high_pollution_risk",
                         guest_result["high_pollution_risk"] == results["ทั่วไป"]["high_pollution_risk"])
        all_ok &= check("guest risk_horizons == ทั่วไป risk_horizons",
                         guest_result["risk_horizons"] == results["ทั่วไป"]["risk_horizons"])

        for rg in RISK_GROUPS:
            all_ok &= check(f"{rg}: response risk_group field ตรง", results[rg]["risk_group"] == rg)

        # threshold ต่ำกว่า -> กลุ่มเปราะบางต้อง trigger ได้ง่ายกว่าหรือเท่ากันเสมอ (superset) ไม่ว่าข้อมูลสดวันนี้จะเป็นอย่างไร
        general_set = set(results["ทั่วไป"]["risk_horizons"])
        for rg in ["เด็ก-ผู้สูงอายุ", "มีโรคประจำตัว"]:
            rg_set = set(results[rg]["risk_horizons"])
            all_ok &= check(f"{rg}: risk_horizons เป็น superset ของ ทั่วไป (threshold ต่ำกว่าเสมอ)",
                             general_set.issubset(rg_set))

        print("\nตารางเทียบ 3 กลุ่ม (ข้อมูลสดจริง ณ เวลารัน):")
        print(f"{'risk_group':<18}{'high_pollution_risk':<22}{'n_risk_horizons':<18}{'risk_horizons'}")
        for rg in RISK_GROUPS:
            r = results[rg]
            print(f"{rg:<18}{str(r['high_pollution_risk']):<22}{len(r['risk_horizons']):<18}{r['risk_horizons']}")

        print("\n" + "=" * 80)
        print("STEP C: _build_daily_nlg_prompt() ใส่โทนถูกต้องตาม risk_group (ตรวจ prompt ที่ capture จาก STEP B)")
        print("=" * 80)
        all_ok &= check("เรียก Gemini (stub) ครบ 3 ครั้งจาก STEP B (1 ครั้ง/กลุ่ม, ไม่เกิน 3 ตามที่ออกแบบ)",
                         len(captured_prompts) == 3)
        for rg, prompt in zip(RISK_GROUPS, captured_prompts):
            expected_tone = app._USER_GROUP_TONE[rg]
            all_ok &= check(f"{rg}: prompt มีข้อความโทนที่ถูกต้อง", expected_tone in prompt)
            other_tones = [t for g, t in app._USER_GROUP_TONE.items() if g != rg]
            all_ok &= check(f"{rg}: prompt ไม่มีโทนของกลุ่มอื่นปน", all(t not in prompt for t in other_tones))

        captured_prompts.clear()
        app.get_forecast("ทั่วไป")  # เรียกซ้ำ risk_group เดิม วันเดียวกัน (mock เวลาเดิม)
        all_ok &= check("เรียกซ้ำ risk_group เดิมในวันเดียวกัน ไม่เรียก Gemini ซ้ำ (cache ผ่าน daily_nlg_log)",
                         len(captured_prompts) == 0)

        for rg in RISK_GROUPS:
            row = app.get_daily_nlg_row(f"{_TEST_YEAR}-07-20", rg)
            all_ok &= check(f"{rg}: daily_nlg_log มีแถวของกลุ่มนี้โดยเฉพาะ (คีย์ composite ถูกต้อง)", row is not None)

        print("\n" + "=" * 80)
        print("STEP D: /api/forecast ผ่าน HTTP จริง — login เป็นแต่ละ risk_group แล้วเทียบ session-resolution")
        print("=" * 80)
        for rg in RISK_GROUPS:
            client = app.app.test_client()
            email = f"riskgroup-test-{_EMAIL_SLUG[rg]}-{int(time.time())}{_TEST_DOMAIN}"
            r = client.post("/api/auth/register", json={
                "email": email, "password": "correcthorse1", "risk_group": rg, "age": 30,
            })
            all_ok &= check(f"{rg}: register -> 201", r.status_code == 201)
            r = client.get("/api/forecast")
            all_ok &= check(f"{rg}: GET /api/forecast -> 200", r.status_code == 200)
            body = r.get_json()
            all_ok &= check(f"{rg}: /api/forecast resolves session -> risk_group ตรง", body.get("risk_group") == rg)

        guest_client = app.app.test_client()  # ไม่ login เลย
        r = guest_client.get("/api/forecast")
        all_ok &= check("guest (ไม่ login) -> 200", r.status_code == 200)
        all_ok &= check("guest (ไม่ login) -> risk_group == 'ทั่วไป'", r.get_json().get("risk_group") == "ทั่วไป")
    finally:
        clear_test_rows(app)

    print("\n" + ("ALL TESTS PASSED" if all_ok else "SOME TESTS FAILED"))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

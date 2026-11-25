"""ทดสอบคำเตือนฝุ่นสูงแบบรวม risk_alert_for() + risk_alert_text_th() + ข้อความ push (app.py, CHANGELOG 272)
ไม่แตะ DB ไม่เรียก API ภายนอก — รัน: python tests/manual/test_risk_alert.py
"""
import os
import sys
import importlib.util

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APP_PATH = os.path.join(_ROOT, "app.py")
results = []


def check(name, cond):
    results.append(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")


def load_app():
    os.chdir(_ROOT)
    spec = importlib.util.spec_from_file_location("app_under_test", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    app = load_app()
    t = lambda h: f"{(16 + h) % 24:02d}:00"  # เวลาสมมติ: horizon h -> 16+h นาฬิกา

    check("ไม่เสี่ยงทั้งตอนนี้และ 24 ชม. -> None", app.risk_alert_for(False, [], t) is None)

    a = app.risk_alert_for(True, [5, 6], t)
    check("เกินตอนนี้ แต่ชั่วโมงหน้าไม่เกิน -> now, until None", a["now"] and a["until_horizon"] is None)
    title, sub = app.risk_alert_text_th(a, "ทั่วไป")
    check("  ข้อความ: 'คาดว่าจะลดลงภายในชั่วโมงหน้า'", sub == "คาดว่าจะลดลงภายในชั่วโมงหน้า")

    a = app.risk_alert_for(True, [1, 2, 3, 7, 8], t)
    check("เกินตอนนี้ + h1-3 ติดกัน -> until = 3 (ไม่นับช่วงหลังที่ขาดตอน)", a["until_horizon"] == 3 and a["until_display_time"] == "19:00")
    title, sub = app.risk_alert_text_th(a, "มีโรคประจำตัว")
    check("  หัวข้อ: 'ผู้มีโรคประจำตัว ควรระวังฝุ่นตอนนี้'", title == "ผู้มีโรคประจำตัว ควรระวังฝุ่นตอนนี้")
    check("  บรรทัดรอง: 'ไปจนถึงประมาณ 19:00 น.'", sub == "คาดว่าจะสูงแบบนี้ไปจนถึงประมาณ 19:00 น.")

    a = app.risk_alert_for(False, [2, 3, 4, 10], t)
    check("ยังไม่เกิน จะเกิน h2 ถึง h4 -> start 2, until 4", (a["now"], a["start_horizon"], a["until_horizon"]) == (False, 2, 4))
    title, _ = app.risk_alert_text_th(a, "เด็ก-ผู้สูงอายุ")
    check("  หัวข้อ: 'เด็กและผู้สูงอายุ ควรระวังฝุ่นตั้งแต่ 18:00 น. (อีก 2 ชม.)'",
          title == "เด็กและผู้สูงอายุ ควรระวังฝุ่นตั้งแต่ 18:00 น. (อีก 2 ชม.)")

    a = app.risk_alert_for(True, list(range(1, 25)), t)
    check("เกินตลอด 24 ชม. -> through_window_end", a["through_window_end"])
    check("  ข้อความ: 'ตลอด 24 ชม. ข้างหน้า'", app.risk_alert_text_th(a, "ทั่วไป")[1] == "คาดว่าจะสูงแบบนี้ตลอด 24 ชม. ข้างหน้า")

    _, body = app._push_payload_for("threshold", "มีโรคประจำตัว", None, None, None, None, 30.0, None, None,
                                    app.risk_alert_for(True, [1, 2], t))
    check("push threshold ใช้ข้อความเดียวกับแบนเนอร์ (ไม่มีหมายเหตุ/เกณฑ์)",
          body == "ผู้มีโรคประจำตัว ควรระวังฝุ่นตอนนี้ — คาดว่าจะสูงแบบนี้ไปจนถึงประมาณ 18:00 น."
          and "เกณฑ์" not in body and "ต่ำกว่าความจริง" not in body)
    _, body = app._push_payload_for("threshold", "ทั่วไป", None, None, None, None, 40.0, None, None)
    check("push threshold ไม่ส่ง risk_alert (ผู้เรียกเก่า) -> ไม่พัง", body == "ทุกคน ควรระวังฝุ่น")
    _, body = app._push_payload_for("imminent", "ทั่วไป", None, 1, 38.2, "18:00", 30.0, None, None)
    check("push imminent ไม่ใช้คำว่า 'เกณฑ์'", "เกณฑ์" not in body and "18:00 น." in body)

    # CHANGELOG 275: push imminent ครั้งเดียวต่อช่วง และไม่ส่งถ้าฝุ่นเกินอยู่แล้ว
    iv = app._push_imminent_value
    check("ช่วงเสี่ยงยังไกล -> ไม่ส่ง imminent", iv(False, False) is None)
    check("อีก <= 2 ชม. จะเกิน -> imminent 'active'", iv(False, True) == "active")
    check("ค่าเดียวกันไม่ว่าจะอีก 2 หรือ 1 ชม. -> dedup ไม่ส่งซ้ำ", iv(False, True) == iv(False, True))
    check("เกินอยู่แล้วตอนนี้ -> ไม่ส่ง imminent (threshold ครอบคลุม)", iv(True, True) is None)

    print()
    print("ALL TESTS PASSED" if all(results) else f"SOME TESTS FAILED ({results.count(False)}/{len(results)})")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()

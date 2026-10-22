"""
ทดสอบ validate_health_message() (NLG Safety Check) — ข้อความที่ Gemini สร้างต้องตรงกับระดับ
PM2.5 จริงจาก pm25_status(), ไม่เกินจริง/ไม่ประเมินต่ำไป และกัน false-positive จาก keyword
ที่เป็น substring ของระดับอื่น
"""
import importlib.util
import math

import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_PATH = os.path.join(_ROOT, r"app.py")


def load_app():
    spec = importlib.util.spec_from_file_location("app_under_test", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


app = load_app()

print("=== Test 1: ข้อความตรงตามระดับ (pm25=30 -> ปานกลาง, ข้อความบอกปานกลาง) ===")
is_valid, level, msg = app.validate_health_message(
    "วันนี้คุณภาพอากาศปานกลาง กลุ่มเสี่ยงควรระมัดระวัง", 30.0
)
print(f"  is_valid={is_valid}, level={level!r}, msg={msg!r}")
assert is_valid is True, "ข้อความตรงตามระดับควรผ่าน (is_valid=True)"
assert level == "ปานกลาง"
assert msg == "วันนี้คุณภาพอากาศปานกลาง กลุ่มเสี่ยงควรระมัดระวัง", "ข้อความที่ผ่านต้องส่งข้อความเดิม ไม่ใช่ template"

print("\n=== Test 2: ข้อความเกินจริง (pm25=30 -> ปานกลาง, แต่ข้อความบอกอันตรายถึงชีวิต) ===")
is_valid, level, msg = app.validate_health_message(
    "อากาศอันตรายถึงชีวิต ห้ามออกจากบ้านเด็ดขาด", 30.0
)
print(f"  is_valid={is_valid}, level={level!r}, msg={msg!r}")
assert is_valid is False, "ข้อความเกินจริงต้องไม่ผ่าน"
assert level == "ปานกลาง", "level ที่คืนต้องเป็นระดับจริงจาก pm25 ไม่ใช่ระดับที่ข้อความอ้าง"
assert msg == app._fallback_message("ปานกลาง"), "ต้องได้ template ของระดับจริง (ปานกลาง) ไม่ใช่ข้อความเดิม"

print("\n=== Test 3: ข้อความประเมินต่ำไป (pm25=95 -> มีผลกระทบต่อสุขภาพ, แต่ข้อความบอกอากาศดี) ===")
is_valid, level, msg = app.validate_health_message(
    "วันนี้อากาศดีมาก ออกไปวิ่งได้สบายๆ", 95.0
)
print(f"  is_valid={is_valid}, level={level!r}, msg={msg!r}")
assert is_valid is False, "ข้อความประเมินต่ำไปต้องไม่ผ่าน (อันตรายกว่าเพราะทำให้ผู้ใช้ชะล่าใจ)"
assert level == "มีผลกระทบต่อสุขภาพ"
assert msg == app._fallback_message("มีผลกระทบต่อสุขภาพ")

print("\n=== Test 4: predicted_pm25 เป็น NaN ===")
is_valid, level, msg = app.validate_health_message("อากาศดีมาก", float("nan"))
print(f"  is_valid={is_valid}, level={level!r}, msg={msg!r}")
assert is_valid is False, "pm25 เป็น NaN ต้องไม่ผ่าน"
assert level is None, "ประเมินระดับไม่ได้เมื่อ pm25 เป็น NaN"
assert msg == app._fallback_message(None)

print("\n=== Test 5: ข้อความว่างเปล่า ===")
is_valid, level, msg = app.validate_health_message("", 30.0)
print(f"  is_valid={is_valid}, level={level!r}, msg={msg!r}")
assert is_valid is False, "ข้อความว่างเปล่าต้องไม่ผ่าน"
assert level == "ปานกลาง", "ยังประเมินระดับจาก pm25 ได้ปกติแม้ข้อความว่าง"
assert msg == app._fallback_message("ปานกลาง")

print("\n=== ทดสอบเพิ่มเติม: กัน false-positive จาก keyword ที่เป็น substring ของกันเอง ===")
print("--- (a) 'เริ่มมีผลกระทบต่อสุขภาพ' มี 'มีผลกระทบต่อสุขภาพ' เป็น substring อยู่ในตัว ---")
is_valid, level, msg = app.validate_health_message(
    "วันนี้คุณภาพอากาศเริ่มมีผลกระทบต่อสุขภาพ ควรลดกิจกรรมกลางแจ้ง", 60.0
)
print(f"  is_valid={is_valid}, level={level!r}, msg={msg!r}")
assert level == "เริ่มมีผลกระทบต่อสุขภาพ"
assert is_valid is True, "ข้อความที่ตรงระดับจริงเป๊ะต้องผ่าน แม้ชื่อระดับถัดไปจะเป็น substring อยู่ในตัวก็ตาม"

print("--- (b) 'อากาศดีมาก' มี 'อากาศดี' เป็น substring อยู่ในตัว ---")
is_valid, level, msg = app.validate_health_message("วันนี้อากาศดีมาก", 10.0)
print(f"  is_valid={is_valid}, level={level!r}, msg={msg!r}")
assert level == "ดีมาก"
assert is_valid is True, "ข้อความที่ตรงระดับจริงเป๊ะต้องผ่าน แม้จะมี keyword ของระดับข้างเคียงซ้อนอยู่เป็น substring"

print("\n=== ทดสอบเพิ่มเติม: ข้อความไม่มี keyword ระดับไหนเลย (fail-safe) ===")
is_valid, level, msg = app.validate_health_message("วันนี้อากาศเป็นยังไงนะ", 30.0)
print(f"  is_valid={is_valid}, level={level!r}, msg={msg!r}")
assert is_valid is False, "ข้อความที่ไม่มี keyword ระดับไหนเลยต้อง fail-safe (ยืนยันความสอดคล้องไม่ได้)"

print("\n=== ทดสอบเพิ่มเติม: ข้อความจริงจาก Gemini ที่เคยตกเพราะ keyword matching เข้มงวดเกินไป ===")
print("--- (c) pm25=20 -> 'ดี', Gemini เขียน 'อากาศจะอยู่ในเกณฑ์ดี' (ไม่มี 'อากาศดี' ตรงตัว) ---")
is_valid, level, msg = app.validate_health_message(
    "ในอีก 12 ชั่วโมงข้างหน้า อากาศจะอยู่ในเกณฑ์ดีและฝุ่นละอองมีแนวโน้มลดลงจากสภาพความชื้นที่ช่วยให้ฝุ่น"
    "กระจายตัวได้ดีครับ คุณสามารถทำกิจกรรมกลางแจ้งได้ตามปกติ แต่อย่าลืมติดตามสถานการณ์อย่างต่อเนื่อง"
    "เพราะค่าฝุ่นอาจเปลี่ยนแปลงได้ตามสภาพอากาศในแต่ละช่วงเวลาครับ",
    20.0,
)
print(f"  is_valid={is_valid}, level={level!r}, msg={msg!r}")
assert level == "ดี"
assert is_valid is True, "ข้อความตรงระดับจริง (ดี) ต้องผ่าน แม้ไม่มี 'อากาศดี' ตรงตัวอักษรเป๊ะ (มีคำแทรก 'จะอยู่ในเกณฑ์' คั่นอยู่)"

print("--- (d) pm25=30 -> 'ปานกลาง', Gemini เขียน 'หลีกเลี่ยงกิจกรรมกลางแจ้ง' (คำเตือนทั่วไป ไม่ใช่ระดับสูงสุด) ---")
is_valid, level, msg = app.validate_health_message(
    "คาดการณ์ว่าในวันพรุ่งนี้คุณภาพอากาศจะอยู่ในระดับปานกลาง ซึ่งได้รับอานิสงส์จากความชื้นที่ช่วยลด"
    "ปริมาณฝุ่นลงได้บ้าง อย่างไรก็ตาม เด็กและผู้สูงอายุควรระมัดระวังสุขภาพเป็นพิเศษและหลีกเลี่ยงกิจกรรม"
    "กลางแจ้งหากเริ่มรู้สึกระคายเคืองตาหรือทางเดินหายใจครับ",
    30.0,
)
print(f"  is_valid={is_valid}, level={level!r}, msg={msg!r}")
assert level == "ปานกลาง"
assert is_valid is True, ("ข้อความตรงระดับจริง (ปานกลาง) ต้องผ่าน แม้จะมี 'หลีกเลี่ยงกิจกรรมกลางแจ้ง'/'ระคายเคือง' "
                          "ปนอยู่ (เป็นคำเตือนทั่วไปที่ใช้ได้ตั้งแต่ปานกลางขึ้นไป ไม่ใช่ keyword เฉพาะระดับสูงสุด)")

print("--- (e) pm25=50 -> 'เริ่มมีผลกระทบต่อสุขภาพ', Gemini เขียน 'ส่งผลกระทบต่อสุขภาพ' + 'หลีกเลี่ยงการทำกิจกรรมกลางแจ้ง' ---")
is_valid, level, msg = app.validate_health_message(
    "แนวโน้มค่าฝุ่นในชั่วโมงข้างหน้าเริ่มลดลงเล็กน้อย แต่อยู่ในระดับที่อาจส่งผลกระทบต่อสุขภาพได้ครับ "
    "ผู้ที่มีโรคประจำตัวเกี่ยวกับทางเดินหายใจควรหลีกเลี่ยงการทำกิจกรรมกลางแจ้งและระมัดระวังสุขภาพเป็นพิเศษครับ",
    50.0,
)
print(f"  is_valid={is_valid}, level={level!r}, msg={msg!r}")
assert level == "เริ่มมีผลกระทบต่อสุขภาพ"
assert is_valid is True, ("ข้อความตรงระดับจริง ต้องผ่าน แม้จะพูด 'ส่งผลกระทบต่อสุขภาพ' (คำละคำจาก keyword เดิม) "
                          "และ 'หลีกเลี่ยงการทำกิจกรรมกลางแจ้ง' (มีคำแทรก 'การทำ' คั่นอยู่)")

print("--- (f) pm25=10 -> 'ดีมาก', คำแทรกยาวๆ ระหว่าง 'อากาศ' กับ 'ดี' ต้องไม่ทำให้ flexible match ของระดับ 'ดี' แย่ง 'ดีมาก' ไปทั้งที่ 'ดีมาก' ตรงตัวอักษรเป๊ะอยู่แล้ว ---")
is_valid, level, msg = app.validate_health_message(
    "ในอีก 1 ชั่วโมงข้างหน้า คุณภาพอากาศอยู่ในเกณฑ์ดีมากครับ เนื่องจากแนวโน้มปริมาณฝุ่นสะสมลดลง"
    "ต่อเนื่องจากช่วงที่ผ่านมา คุณสามารถทำกิจกรรมกลางแจ้งได้ตามปกติโดยไม่ต้องกังวลเรื่องสุขภาพครับ",
    10.0,
)
print(f"  is_valid={is_valid}, level={level!r}, msg={msg!r}")
assert level == "ดีมาก"
assert is_valid is True, ("ข้อความตรงระดับจริง (ดีมาก) ต้องผ่าน — literal keyword 'ดีมาก' ต้องมีสิทธิ์เหนือกว่า "
                          "flexible match ('อากาศ'...'ดี') ที่บังเอิญ span ยาวกว่าเพราะมีคำแทรกยาวๆ คั่นอยู่ก่อนหน้า")

print("\n=== ทั้งหมดผ่าน ===")

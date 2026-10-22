"""
ทดสอบ NLG รายวัน (generate_daily_nlg_explanation, maybe_generate_daily_nlg, get_today_daily_nlg,
get_thai_now) — เน้นเคสข้ามวัน UTC/เวลาไทย (06:00 น. ไทย = 23:00 UTC ของวันก่อนหน้า)

เขียนแถวทดสอบลง daily_nlg_log จริงด้วย date_thai ปี 2199 แล้วลบทิ้งก่อน/หลังรัน
"""
import importlib.util
import os
import sys

sys.stdout.reconfigure(encoding="utf-8")

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_PATH = os.path.join(_ROOT, r"app.py")
DATA_PATH = os.path.join(_ROOT, r"data\khonkaen_pm25_features_raw_20260712_191249.csv")
_TEST_YEAR = "2199"  # ทุกแถวทดสอบใช้ date_thai ขึ้นต้นด้วยปีนี้เสมอ ไม่มีทางชนข้อมูลจริง


def load_app():
    spec = importlib.util.spec_from_file_location("app_under_test", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clear_test_rows(app):
    conn = app.get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM daily_nlg_log WHERE date_thai LIKE %s", (f"{_TEST_YEAR}-%",))
    conn.commit()
    cur.close()
    conn.close()


app = load_app()
clear_test_rows(app)

_original_call_gemini = app._call_gemini
_gemini_call_count = {"n": 0}


def _fake_call_gemini(prompt):
    _gemini_call_count["n"] += 1
    return "วันนี้คุณภาพอากาศอยู่ในเกณฑ์ดี สามารถทำกิจกรรมกลางแจ้งได้ตามปกติ"


def make_fake_forecast(pm25_values):
    return [{"time": f"{h:02d}:00", "pm25": v, "model_used": "lightgbm"} for h, v in enumerate(pm25_values, start=1)]


def make_fake_feature_row():
    return pd.DataFrame([{c: 1.0 for c in app.FEATURE_COLS}])


app._call_gemini = _fake_call_gemini
fake_forecast = make_fake_forecast([20.0] * 24)
fake_feature_row = make_fake_feature_row()

print("=" * 80)
print(f"TEST 1: ข้าม UTC เที่ยงคืน — 23:00 UTC {_TEST_YEAR}-07-14 = 06:00 น. ไทย {_TEST_YEAR}-07-15 (ต้องบันทึก date_thai={_TEST_YEAR}-07-15)")
print("=" * 80)
app.get_thai_now = lambda: pd.Timestamp(f"{_TEST_YEAR}-07-14T23:00:00", tz="UTC").tz_convert("Asia/Bangkok")
_gemini_call_count["n"] = 0
app.maybe_generate_daily_nlg(fake_forecast, fake_feature_row)

row_correct = app.get_daily_nlg_row(f"{_TEST_YEAR}-07-15")
row_wrong = app.get_daily_nlg_row(f"{_TEST_YEAR}-07-14")
print(f"  แถว date_thai={_TEST_YEAR}-07-15: {row_correct}")
print(f"  แถว date_thai={_TEST_YEAR}-07-14 (ต้องไม่มี): {row_wrong}")
assert row_correct is not None, f"ควรมีแถวของ {_TEST_YEAR}-07-15 (เวลาไทยหลังแปลง) ไม่ใช่ {_TEST_YEAR}-07-14"
assert row_wrong is None, f"ห้ามมีแถว {_TEST_YEAR}-07-14 — ถ้ามีแปลว่าใช้วันที่ฝั่ง UTC ตรงๆ ผิดพลาด"
assert _gemini_call_count["n"] == 1, f"ควรเรียก Gemini 1 ครั้ง แต่เรียก {_gemini_call_count['n']} ครั้ง"
print("  PASS")

print("\n" + "=" * 80)
print("TEST 2: เรียกซ้ำในชั่วโมงเดียวกัน (23:45 UTC วันเดิม) — ต้องไม่เรียก Gemini ซ้ำ")
print("=" * 80)
app.get_thai_now = lambda: pd.Timestamp(f"{_TEST_YEAR}-07-14T23:45:00", tz="UTC").tz_convert("Asia/Bangkok")
_gemini_call_count["n"] = 0
app.maybe_generate_daily_nlg(fake_forecast, fake_feature_row)
print(f"  จำนวนครั้งที่เรียก Gemini: {_gemini_call_count['n']}")
assert _gemini_call_count["n"] == 0, f"date_thai={_TEST_YEAR}-07-15 มีแถวอยู่แล้ว ไม่ควรเรียก Gemini ซ้ำ"
print("  PASS")

print("\n" + "=" * 80)
print(f"TEST 3: ข้ามไปวันถัดไป — 23:00 UTC {_TEST_YEAR}-07-15 = 06:00 น. ไทย {_TEST_YEAR}-07-16 (ต้อง trigger ใหม่)")
print("=" * 80)
app.get_thai_now = lambda: pd.Timestamp(f"{_TEST_YEAR}-07-15T23:00:00", tz="UTC").tz_convert("Asia/Bangkok")
_gemini_call_count["n"] = 0
app.maybe_generate_daily_nlg(make_fake_forecast([15.0] * 24), fake_feature_row)
row_new_day = app.get_daily_nlg_row(f"{_TEST_YEAR}-07-16")
print(f"  แถว date_thai={_TEST_YEAR}-07-16: {row_new_day}")
assert row_new_day is not None
assert _gemini_call_count["n"] == 1
print("  PASS")

print("\n" + "=" * 80)
print("TEST 4: daily_avg_pm25 คำนวณถูกต้องจาก forecast[] จริง")
print("=" * 80)
values = [10.0, 20.0, 30.0] + [15.0] * 21
expected_avg = round(sum(values) / len(values), 1)
daily_avg_pm25, explanation_text, used_fallback = app.generate_daily_nlg_explanation(
    make_fake_forecast(values), fake_feature_row
)
print(f"  daily_avg_pm25 คำนวณได้: {daily_avg_pm25}, คาดหวัง: {expected_avg}")
assert daily_avg_pm25 == expected_avg, f"expected {expected_avg}, got {daily_avg_pm25}"
print("  PASS")

print("\n" + "=" * 80)
print("TEST 5: edge case — 22:59 UTC (ยังไม่ถึง 06:00 น. ไทย, hour<6) ต้องไม่ trigger")
print("*** พฤติกรรมเปลี่ยนจากเดิม ***: เดิมเงื่อนไขคือ hour == 6 พอดี ตอนนี้เปลี่ยนเป็น hour >= 6")
print("(trigger ได้ตั้งแต่ 06:00 น. เป็นต้นไปจนกว่าจะมีแถวของวันนั้น ไม่ใช่แค่ชั่วโมง 6 เป๊ะ)")
print("=" * 80)
# 22:59 UTC 16 = 05:59 น. ไทย 17 -> hour=5 ยังไม่ถึง 6 ต้องไม่ trigger
app.get_thai_now = lambda: pd.Timestamp(f"{_TEST_YEAR}-07-16T22:59:00", tz="UTC").tz_convert("Asia/Bangkok")
_gemini_call_count["n"] = 0
app.maybe_generate_daily_nlg(make_fake_forecast([15.0] * 24), fake_feature_row)
print(f"  22:59 UTC (05:59 น. ไทย, hour=5): เรียก Gemini {_gemini_call_count['n']} ครั้ง (คาดหวัง 0, ไม่เปลี่ยนจากเดิม)")
assert _gemini_call_count["n"] == 0, "05:59 น. ไทย (hour=5) ยังไม่ถึง 06:00 น. ต้องไม่ trigger"
assert app.get_daily_nlg_row(f"{_TEST_YEAR}-07-17") is None

# 00:30 UTC 17 = 07:30 น. ไทย 17 -> hour=7, ยังไม่มีแถวของวันนั้น ต้อง trigger
app.get_thai_now = lambda: pd.Timestamp(f"{_TEST_YEAR}-07-17T00:30:00", tz="UTC").tz_convert("Asia/Bangkok")
_gemini_call_count["n"] = 0
app.maybe_generate_daily_nlg(make_fake_forecast([15.0] * 24), fake_feature_row)
print(f"  00:30 UTC วันถัดไป (07:30 น. ไทย, hour=7): เรียก Gemini {_gemini_call_count['n']} ครั้ง "
      f"(คาดหวัง 1 — *** เปลี่ยนจากเดิมที่คาดหวัง 0 *** เพราะตอนนี้ hour>=6 แล้ว)")
assert _gemini_call_count["n"] == 1, "07:30 น. ไทย (hour=7 >= 6) และยังไม่มีแถวของวันนี้ ต้อง trigger ตามเงื่อนไขใหม่"
assert app.get_daily_nlg_row(f"{_TEST_YEAR}-07-17") is not None, "ต้องมีแถวของวันนี้หลัง trigger ที่ hour=7"
print("  PASS")

print("\n" + "=" * 80)
print("TEST 8 (ใหม่): mock 09:00 น. ไทย วันที่ยังไม่มี daily_nlg_log เลย — ต้อง trigger ทันที")
print("(ก่อนหน้านี้จะไม่ trigger เพราะ hour=9 != 6 พอดี ต้องรอข้ามวันไปเลย)")
print("=" * 80)
# 02:00 UTC 20 = 09:00 น. ไทย 20
app.get_thai_now = lambda: pd.Timestamp(f"{_TEST_YEAR}-07-20T02:00:00", tz="UTC").tz_convert("Asia/Bangkok")
_gemini_call_count["n"] = 0
app.maybe_generate_daily_nlg(make_fake_forecast([15.0] * 24), fake_feature_row)
print(f"  09:00 น. ไทย (hour=9): เรียก Gemini {_gemini_call_count['n']} ครั้ง (คาดหวัง 1)")
assert _gemini_call_count["n"] == 1, "09:00 น. ไทย (hour=9 >= 6) ควร trigger ทันทีถ้ายังไม่มีแถวของวันนี้"
assert app.get_daily_nlg_row(f"{_TEST_YEAR}-07-20") is not None
print("  PASS")

print("\n" + "=" * 80)
print("TEST 9 (ใหม่): mock 23:00 น. ไทย วันที่ยังไม่มี daily_nlg_log เลยทั้งวัน — ต้องยัง trigger ได้")
print("(ดีกว่าพลาดไปทั้งวัน แม้จะช้ามากก็ตาม)")
print("=" * 80)
# 16:00 UTC 21 = 23:00 น. ไทย 21
app.get_thai_now = lambda: pd.Timestamp(f"{_TEST_YEAR}-07-21T16:00:00", tz="UTC").tz_convert("Asia/Bangkok")
_gemini_call_count["n"] = 0
app.maybe_generate_daily_nlg(make_fake_forecast([15.0] * 24), fake_feature_row)
print(f"  23:00 น. ไทย (hour=23): เรียก Gemini {_gemini_call_count['n']} ครั้ง (คาดหวัง 1)")
assert _gemini_call_count["n"] == 1, "23:00 น. ไทย (hour=23 >= 6) ยังควร trigger ได้ถ้ายังไม่มีแถวของวันนี้เลย"
assert app.get_daily_nlg_row(f"{_TEST_YEAR}-07-21") is not None
print("  PASS")

print("\n" + "=" * 80)
print("TEST 6: get_today_daily_nlg() คืน placeholder ถ้ายังไม่มีแถวของวันนั้นเลย")
print("=" * 80)
app.get_thai_now = lambda: pd.Timestamp(f"{_TEST_YEAR}-08-01T10:00:00", tz="UTC").tz_convert("Asia/Bangkok")
result = app.get_today_daily_nlg()
print(f"  ผลลัพธ์: {result}")
assert result["daily_avg_pm25"] is None
assert result["explanation"] == "กำลังเตรียมข้อมูลสำหรับวันนี้"
print("  PASS")

# คืนค่า _call_gemini ตัวจริงกลับมา สำหรับเคสสุดท้ายที่เรียก Gemini จริง
app._call_gemini = _original_call_gemini

print("\n" + "=" * 80)
print("TEST 7 (เรียก Gemini จริง ไม่ mock): ตัวอย่างข้อความ NLG รายวันจริงจากค่าเฉลี่ยจริง")
print("=" * 80)
df = pd.read_csv(DATA_PATH)
real_feature_row = df.dropna(subset=app.FEATURE_COLS)[app.FEATURE_COLS].iloc[[-1]]
real_forecast = make_fake_forecast([12.0, 14.0, 16.0, 18.0, 20.0, 22.0, 24.0, 23.0,
                                     21.0, 19.0, 17.0, 16.0, 15.0, 14.0, 16.0, 18.0,
                                     20.0, 22.0, 24.0, 23.0, 20.0, 17.0, 14.0, 12.0])
daily_avg_pm25, explanation_text, used_fallback = app.generate_daily_nlg_explanation(
    real_forecast, real_feature_row
)
print(f"  daily_avg_pm25 = {daily_avg_pm25} ({app.pm25_status(daily_avg_pm25)})")
print(f"  ข้อความ NLG รายวันจริงจาก Gemini: {explanation_text!r}")
print(f"  ใช้ fallback template หรือไม่: {used_fallback}")
is_valid_check = used_fallback or app.validate_health_message(explanation_text, daily_avg_pm25)[0]
assert is_valid_check, "ข้อความสุดท้ายต้องผ่าน safety check หรือเป็น fallback template เท่านั้น"
print("  PASS")

print("\n=== ทั้งหมดผ่าน ===")

clear_test_rows(app)

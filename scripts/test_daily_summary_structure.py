"""
ทดสอบส่วนที่แก้ของ "สรุปวันนี้" (daily NLG) รอบนี้: classify_daily_trend(), การใช้ _factor_narrative_th ใน
_format_shap_factor_lines(), _mentions_contradicting_trend(), และ _mentions_generic_caution_below() ที่ขยายให้
ครอบคลุมทุก risk_group — ไม่แก้/ไม่รัน scripts/test_daily_nlg.py เดิม (stale signature, pre-existing failure ตาม
CLAUDE.md ไม่อยู่ใน scope ของงานนี้)

Manual assert + print ตาม convention เดียวกับ scripts/test_nlg_safety_check.py
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


def make_forecast(pm25_by_hour):
    """pm25_by_hour: dict {display_hour_thai: pm25}. สร้าง forecast list 24 จุดจากนี้"""
    return [
        {"display_hour_thai": h, "hour_thai": h, "time": f"{h:02d}:00", "pm25": pm25, "model_used": "lightgbm_pooled_v3"}
        for h, pm25 in pm25_by_hour.items()
    ]


all_ok = True


def check(label, condition):
    global all_ok
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    if not condition:
        all_ok = False
    return condition


print("=== classify_daily_trend() ===")

print("--- ทรงตัวจริง (ทุกชั่วโมง pm25 เท่ากันเป๊ะ) ---")
flat_forecast = make_forecast({h: 20.0 for h in range(24)})
direction, period = app.classify_daily_trend(flat_forecast)
check("direction == 'flat'", direction == "flat")
check("period is None", period is None)

print("--- สูงขึ้นชัดเจนช่วงเช้า (เช้า=40, บ่าย/เย็น=15 -> baseline ต่ำกว่าช่วงเช้ามาก) ---")
morning_up = {}
for h in range(24):
    if h in app.DAILY_TREND_PERIODS[0][1]:  # เช้า 6-11
        morning_up[h] = 40.0
    else:
        morning_up[h] = 15.0
direction, period = app.classify_daily_trend(make_forecast(morning_up))
check("direction == 'up'", direction == "up")
check("period == 'เช้า'", period == "เช้า")

print("--- ลดลงชัดเจนช่วงเย็น-กลางคืน (เย็นกลางคืน=5, อื่นๆ=25) ---")
evening_down = {}
for h in range(24):
    if h in app.DAILY_TREND_PERIODS[2][1]:  # เย็น-กลางคืน 18-23,0-5
        evening_down[h] = 5.0
    else:
        evening_down[h] = 25.0
direction, period = app.classify_daily_trend(make_forecast(evening_down))
check("direction == 'down'", direction == "down")
check("period == 'เย็น-กลางคืน'", period == "เย็น-กลางคืน")

print("--- edge case: มีข้อมูลแค่ 1 bucket (เช้าอย่างเดียว 6 จุด) ---")
only_morning = make_forecast({h: 20.0 + h for h in range(6, 12)})
direction, period = app.classify_daily_trend(only_morning)
check("ไม่ error เมื่อมีแค่ 1 bucket (คืนค่าอะไรมาก็ได้ ไม่ crash)", direction in ("flat", "up", "down"))

print("\n=== _format_shap_factor_lines ใช้ _factor_narrative_th (ไม่ใช่แค่ชื่อ feature เปล่าๆ) ===")
fake_shap_pairs = [("wind_speed", -1.2), ("fire_hotspot_count_100km_24h", 0.8)]
lines = app._format_shap_factor_lines(fake_shap_pairs)
print(f"  output:\n{lines}")
check("มีเนื้อหาเหตุผลจริงจาก _FACTOR_NARRATIVE_TH (ไม่ใช่แค่คำว่า 'ดันค่าฝุ่น')",
      "ดันค่าฝุ่น" not in lines and ("ลม" in lines or "ไฟป่า" in lines or "ความร้อน" in lines))

print("\n=== _mentions_contradicting_trend() ===")
check("direction='up' + 'มีแนวโน้มลดลงต่อเนื่อง' (คำว่าแนวโน้มอยู่ติดกับคำขัดแย้ง) -> ขัดแย้ง (True)",
      app._mentions_contradicting_trend("วันนี้มีแนวโน้มลดลงต่อเนื่องตลอดวัน", ("up", "เช้า")) is True)
check("direction='up' + 'มีแนวโน้มสูงขึ้นในช่วงเช้า' -> ไม่ขัดแย้ง (False)",
      app._mentions_contradicting_trend("วันนี้มีแนวโน้มสูงขึ้นในช่วงเช้า", ("up", "เช้า")) is False)
check("direction='down' + 'มีแนวโน้มเพิ่มขึ้น' -> ขัดแย้ง (True)",
      app._mentions_contradicting_trend("ฝุ่นมีแนวโน้มเพิ่มขึ้น", ("down", "เย็น-กลางคืน")) is True)
check("direction='flat' + 'แนวโน้มลดลงเล็กน้อย' -> ไม่ขัดแย้ง (False, จงใจปล่อยผ่านตอน flat)",
      app._mentions_contradicting_trend("แนวโน้มลดลงเล็กน้อย", ("flat", None)) is False)
check("ไม่พูดเรื่องแนวโน้มเลย -> ไม่ขัดแย้ง (False)",
      app._mentions_contradicting_trend("วันนี้อากาศดี ทำกิจกรรมได้ตามปกติ", ("up", "เช้า")) is False)
check("direction='up' + 'ลดลง' อยู่ไกลจากคำว่า 'แนวโน้ม' เกินไป (พูดถึงปัจจัยย่อย ไม่ใช่แนวโน้มรวม) -> ไม่ขัดแย้ง (False)",
      app._mentions_contradicting_trend(
          "วันนี้มีแนวโน้มสูงขึ้นชัดเจนในช่วงเย็น เนื่องจากแม้ฝุ่นจากพื้นที่ใกล้เคียงจะลดลงบ้างแล้วก็ตาม",
          ("up", "เย็น-กลางคืน")) is False)

print("\n=== validate_health_message() กับ trend ที่ขัดแย้งจริง (end-to-end ผ่าน public function) ===")
ok, level, msg = app.validate_health_message(
    "วันนี้คุณภาพอากาศอยู่ในเกณฑ์ดี มีแนวโน้มลดลงต่อเนื่องตลอดวัน", 20.0, "ทั่วไป", ("up", "เช้า")
)
check("ระดับถูกต้องแต่แนวโน้มขัดแย้ง -> ต้อง reject (is_valid=False)", ok is False)
check("ยัง classify ระดับจริงได้ถูกต้อง (level='ดี')", level == "ดี")

ok, level, msg = app.validate_health_message(
    "วันนี้คุณภาพอากาศอยู่ในเกณฑ์ดี มีแนวโน้มสูงขึ้นในช่วงเช้า", 20.0, "ทั่วไป", ("up", "เช้า")
)
check("ระดับถูกต้องและแนวโน้มตรงกัน -> ต้องผ่าน (is_valid=True)", ok is True)

print("\n=== _mentions_generic_caution_below() ตอนนี้ต้องเช็คทุก risk_group (ไม่ใช่แค่ 'ทั่วไป') ===")
for rg in ("ทั่วไป", "เด็ก-ผู้สูงอายุ", "มีโรคประจำตัว"):
    ok, level, msg = app.validate_health_message(
        "วันนี้คุณภาพอากาศอยู่ในเกณฑ์ดีมาก แต่ควรหลีกเลี่ยงกิจกรรมกลางแจ้งเพราะระคายเคือง", 10.0, rg
    )
    check(f"risk_group='{rg}': คำเตือนเกินจริงบนวันอากาศดีมาก -> ต้อง reject เหมือนกันทุกกลุ่ม", ok is False)

print("\n=== _mentioned_levels(): literal match ที่ไหนก็ได้ต้องกัน flexible match ของระดับอื่นไม่ให้ซ้อน ===")
msg_double_mention = (
    "คุณภาพอากาศวันนี้อยู่ในเกณฑ์ดีมากและมีแนวโน้มทรงตัวตลอด 24 ชั่วโมงข้างหน้า "
    "โดยสภาพอากาศที่ดีนี้ได้รับอิทธิพลจากปริมาณฝุ่นจากจังหวัดข้างเคียงที่มีแนวโน้มลดลง "
    "แม้จะมีฝุ่นสะสมต่อเนื่องมาบ้างก็ตาม ขอให้ทุกท่านทำกิจกรรมกลางแจ้งได้ตามปกติอย่างมั่นใจครับ"
)
mentioned = app._mentioned_levels(msg_double_mention)
check("มี literal 'ดีมาก' + flexible 'อากาศ...ดี' (คนละตำแหน่ง ไม่ทับกัน) -> ต้องเหลือแค่ 'ดีมาก' ตัวเดียว",
      mentioned == ["ดีมาก"])
ok, level, _ = app.validate_health_message(msg_double_mention, 11.5, "มีโรคประจำตัว")
check("ข้อความจริงที่เคยโดน false-reject (เจอตอนทดสอบกับ Gemini จริง) -> ต้องผ่าน (is_valid=True)", ok is True)

print("\n=== PM25_LEVELS keyword ใหม่สำหรับ 'ดี' ไม่ชนกับระดับอื่น (กัน false-positive) ===")
ok, level, msg = app.validate_health_message("วันนี้อากาศโปร่ง น่าพอใจมาก ออกไปทำกิจกรรมได้ตามปกติ", 20.0)
check("'อากาศโปร่ง'/'น่าพอใจ' จับเป็นระดับ 'ดี' ได้ และผ่าน (ไม่หลุดไปชนคำว่า 'มีผลกระทบต่อสุขภาพ')",
      level == "ดี" and ok is True)

print("\n" + "=" * 60)
if all_ok:
    print("ทั้งหมดผ่าน")
else:
    print("มีบางเคส FAIL — ดูรายละเอียดด้านบน")
    raise SystemExit(1)

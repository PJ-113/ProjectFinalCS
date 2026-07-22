"""
ทดสอบ find_best_time() — Best Time Window Finder ต้องกรองเฉพาะ 06:00-20:00 เวลาไทย (ช่วงที่ใช้งาน
จริงได้) ก่อนหาช่วงที่ PM2.5 ต่ำสุด แทนที่จะหาจากทั้ง 24 ชั่วโมงรวมช่วงดึกที่ไม่มีประโยชน์ในทางปฏิบัติ
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


def make_point(hour_thai, pm25):
    return {"time": f"{hour_thai:02d}:00", "hour_thai": hour_thai, "pm25": pm25, "model_used": "lightgbm"}


# Test 1: pm25 ต่ำสุดจริงอยู่ตอนตี 3 (นอกช่วง active hours) แต่ในช่วง 06:00-20:00 ต่ำสุดคือ 14:00
# -> find_best_time() ต้องเลือก 14:00 ไม่ใช่ตี 3
forecast = [make_point(3, 5.0), make_point(14, 15.0), make_point(20, 18.0), make_point(23, 6.0)]
# เติมจุดอื่นๆ ให้ครบ active-hour ช่วง (ค่าสูงกว่า 15.0 หมด กัน false pass จากมีจุดเดียว)
for h in [6, 8, 10, 12, 18]:
    forecast.append(make_point(h, 25.0))
best = app.find_best_time(forecast)
assert best["hour_thai"] == 14, best
assert best["pm25"] == 15.0, best
print(f"Test 1 ผ่าน: best_time={best['time']} (pm25={best['pm25']}) ไม่ใช่ตี 3 แม้ pm25 ตี 3 จะต่ำกว่า")

# Test 2: edge case — ทุกจุดใน 06:00-20:00 ค่าสูงหมด (>50) แต่ตี 3 ต่ำกว่ามาก
# -> ต้อง fallback ไปแนะนำช่วงกลางวันที่ดีที่สุดเท่าที่มี (55.0 ตอน 12:00) ไม่ใช่กระโดดไปตี 3
forecast2 = [make_point(3, 5.0), make_point(6, 80.0), make_point(12, 55.0), make_point(20, 90.0)]
best2 = app.find_best_time(forecast2)
assert best2["hour_thai"] == 12, best2
assert best2["pm25"] == 55.0, best2
print(f"Test 2 ผ่าน: best_time={best2['time']} (pm25={best2['pm25']}) ยังอยู่ในช่วงกลางวัน แม้ค่าสูงกว่าตี 3 มาก "
      f"(fallback ไป forecast ทั้งหมดใช้เฉพาะกรณี active_hours ว่างเปล่าจริงๆ)")

# Test 3: active_hours ว่างเปล่าจริง (ทุกจุดอยู่นอก 06:00-20:00) -> ต้อง fallback ไป forecast ทั้งหมด
forecast3 = [make_point(0, 10.0), make_point(3, 5.0), make_point(23, 8.0)]
best3 = app.find_best_time(forecast3)
assert best3["hour_thai"] == 3, best3
print(f"Test 3 ผ่าน: active_hours ว่างเปล่าจริง -> fallback ไป forecast ทั้งหมด (best_time={best3['time']})")

# Test 4: ขอบเขต 06:00 และ 20:00 ต้องถูกนับรวม (inclusive ทั้งสองขอบ)
forecast4 = [make_point(5, 1.0), make_point(6, 2.0), make_point(20, 3.0), make_point(21, 0.5)]
best4 = app.find_best_time(forecast4)
assert best4["hour_thai"] == 6, best4  # 05:00 (นอกช่วง) ถูกตัด, เหลือ 06:00/20:00 -> 06:00 ต่ำสุด
print(f"Test 4 ผ่าน: ขอบเขต 06:00/20:00 รวมอยู่ใน active hours (best_time={best4['time']})")

print("\nทุก assertion ผ่านหมด")

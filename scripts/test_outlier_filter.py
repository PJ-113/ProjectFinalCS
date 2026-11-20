"""
ทดสอบการกรองค่า PM2.5/สภาพอากาศผิดปกติจากเซนเซอร์ก่อนเข้ากระบวนการพยากรณ์ (_filter_pm25_outliers, _check_weather
ที่ขยายแล้ว, fetch_current_weather_batch ที่แก้บั๊ก all-or-nothing) — assert ธรรมดา ไม่มี pytest, ไม่แตะ DB จริง,
ไม่ยิง network จริง (สร้างข้อมูลสังเคราะห์/monkeypatch เอง)

  1. _filter_pm25_outliers(): hard bound (ติดลบ/>1500), isolated spike-revert (เจอจริงแบบ khonkaen
     2024-06-14/2024-08-28: ~18->~55->~17), เหตุการณ์จริงที่ค่อยๆ เปลี่ยนแปลงต่อเนื่อง (จำลองแบบ buengkan 226.4)
     ต้องไม่ถูกกรอง, แถวล่าสุด (anchor) ไม่ถูกแตะโดย spike-revert check, USE_OUTLIER_FILTER=False -> parity เดิม
  2. _check_weather() ที่ขยายแล้ว: แต่ละ field ผิดปกติถูก reject ทีละตัว, ค่าปกติผ่าน, toggle-off คืน behavior เดิม
  3. fetch_current_weather_batch(): ยืนยันว่ายังคง all-or-nothing เหมือนเดิม (ค่าผิดปกติจังหวัดเดียวทำให้ raise
     ทั้ง batch) — ตั้งใจไม่เปลี่ยน behavior นี้ เพราะ scripts/test_input_fallback.py:160/165 มี test คุมไว้อยู่แล้ว
     ว่าเป็น design ที่จงใจ (prefetch สำเร็จ = ครบทุกจังหวัด, ไม่สำเร็จ = ไม่มีใครเข้า cache เลย) ไม่ใช่บั๊ก
  4. integration: fetch_recent_pm25_openaq()/_hours() ผ่าน mock ของ _openaq_get คืนค่าที่มี outlier ฝังอยู่ ->
     ยืนยันว่า outlier หายไปจาก df ที่คืนกลับมา

รัน: python scripts/test_outlier_filter.py
"""
import importlib.util
import os
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_PATH = os.path.join(_ROOT, "app.py")


def load_app():
    spec = importlib.util.spec_from_file_location("app_under_test", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


app = load_app()

all_ok = True


def check(label, condition):
    global all_ok
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    if not condition:
        all_ok = False
    return condition


def make_series(hourly_values, start="2026-01-01T00:00:00Z"):
    idx = pd.date_range(start=start, periods=len(hourly_values), freq="h", tz="UTC")
    return pd.DataFrame({"datetime_utc": idx, "pm25": [float(v) for v in hourly_values]})


print("=== [1] _filter_pm25_outliers(): hard bound ===")
df = make_series([20, 20, -5, 20, 20, 9999, 20])
out = app._filter_pm25_outliers(df)
check("ค่าติดลบถูกตั้งเป็น NaN", pd.isna(out.loc[2, "pm25"]))
check("ค่า 9999 (เกิน PM25_HARD_MAX) ถูกตั้งเป็น NaN", pd.isna(out.loc[5, "pm25"]))
check("ค่าปกติไม่ถูกแตะ", out.loc[0, "pm25"] == 20 and out.loc[6, "pm25"] == 20)

print("\n=== [1] isolated spike-revert — เจอจริงจากข้อมูล 20 จังหวัด (khonkaen 2024-06-14/2024-08-28) ===")
df = make_series([19, 18, 17.9, 57.6, 16.8, 17, 18])  # ...17.9 -> 57.6 -> 16.8...
out = app._filter_pm25_outliers(df)
check("จุดพุ่งเดี่ยวที่กลับคืนทันที (18->55-> 17 แบบที่เจอจริง) ถูกกรอง", pd.isna(out.loc[3, "pm25"]))
check("เพื่อนบ้านของจุดที่กรอง (17.9, 16.8) ไม่ถูกแตะ",
      out.loc[2, "pm25"] == 17.9 and out.loc[4, "pm25"] == 16.8)

print("\n=== [1] เหตุการณ์จริงที่ค่อยๆ เปลี่ยนแปลงต่อเนื่อง (จำลองแบบ buengkan 226.4, ไม่ revert) ต้องไม่ถูกกรอง ===")
# ไล่ขึ้นต่อเนื่องจาก 30 -> 226.4 -> ค่อยๆ ลดลง ตลอด 10 ชม. (ไม่มีจุดไหน "กลับคืน" ในชั่วโมงถัดไปทันที)
rising_falling = [30, 60, 110, 170, 226.4, 200, 160, 120, 80, 50]
df = make_series(rising_falling)
out = app._filter_pm25_outliers(df)
check("ไม่มีจุดกลางไหนถูกกรองเลย (เหตุการณ์จริงต้องรอดทั้งหมด)",
      out["pm25"].iloc[1:-1].equals(df["pm25"].iloc[1:-1]))

print("\n=== [1] แถวล่าสุด (anchor) ไม่ถูกแตะโดย spike-revert check (ไม่มีแถวถัดไปให้ยืนยัน) ===")
df = make_series([18, 18, 18, 55])  # แถวสุดท้ายพุ่งขึ้นแต่ยังไม่เกิน anchor-jump (8x) และ floor (150)
out = app._filter_pm25_outliers(df)
check("แถวล่าสุดที่พุ่งขึ้นแต่ไม่สุดโต่งพอ (ไม่เกิน floor=150) ไม่ถูกกรอง", out["pm25"].iloc[-1] == 55)

print("\n=== [1] anchor one-sided jump: สุดโต่งจากฐานปกติจริงๆ ถึงถูกกรอง ===")
df = make_series([18, 18, 18, 2000])  # เกิน floor (150) และเกิน 8x ของ prev (18*8=144) พร้อมกัน, prev < floor
out = app._filter_pm25_outliers(df)
check("แถวล่าสุดที่กระโดดสุดโต่งจากฐานปกติ (18->2000) ถูกกรอง", pd.isna(out["pm25"].iloc[-1]))

df = make_series([18, 18, 18, 1400])  # ยังไม่เกิน PM25_HARD_MAX (1500) แต่ anchor-jump เงื่อนไขควรจับได้
out = app._filter_pm25_outliers(df)
check("แถวล่าสุดกระโดดสุดโต่งแต่ยังไม่เกิน hard bound -> Tier 3 ยังจับได้", pd.isna(out["pm25"].iloc[-1]))

print("\n=== [1] prev เองก็สูงอยู่แล้ว (ไม่ใช่ฐานปกติ) -> ไม่ถือเป็น anchor jump ===")
df = make_series([18, 18, 160, 1300])  # prev=160 สูงกว่า floor แล้ว -> การไต่ขึ้นต่อถือเป็นเหตุการณ์ต่อเนื่อง
out = app._filter_pm25_outliers(df)
check("prev ที่สูงอยู่แล้ว (>= floor) -> แถวล่าสุดไม่ถูกกรอง แม้กระโดดสูง", out["pm25"].iloc[-1] == 1300)

print("\n=== [1] USE_OUTLIER_FILTER=False -> ฟังก์ชันเรียกใน fetch ไม่ถูกเรียกเลย (parity) ===")
_orig = app.USE_OUTLIER_FILTER
app.USE_OUTLIER_FILTER = False
try:
    _orig_openaq_get = app._openaq_get
    resp = MagicMock()
    resp.json.return_value = {"results": [
        {"period": {"datetimeFrom": {"utc": "2026-01-01T00:00:00Z"}}, "value": 20},
        {"period": {"datetimeFrom": {"utc": "2026-01-01T01:00:00Z"}}, "value": -5},  # จะไม่ถูกกรองเพราะปิด flag
    ]}
    app._openaq_get = MagicMock(return_value=resp)
    df_out = app.fetch_recent_pm25_openaq(sensor_id=1, hours_back=72)
    check("USE_OUTLIER_FILTER=False -> ค่าติดลบยังหลุดผ่าน dropna เดิม (ไม่ใช่ NaN จึงไม่ถูก dropna ทิ้ง)",
          (df_out["pm25"] == -5).any())
finally:
    app.USE_OUTLIER_FILTER = _orig
    app._openaq_get = _orig_openaq_get

print("\n=== [2] _check_weather() ที่ขยายแล้ว ===")
base_w = {"temperature_2m": 28.0, "relative_humidity_2m": 60.0, "wind_speed_10m": 10.0,
          "precipitation": 0.0, "wind_direction_10m": 90.0, "surface_pressure": 1005.0}


def check_raises(label, w):
    try:
        app._check_weather(w)
        check(label, False)
    except ValueError:
        check(label, True)


check("ค่าปกติทั้งหมดต้องผ่าน (ไม่ raise)", app._check_weather(dict(base_w)) is None)
check_raises("humidity เกิน 100 -> ต้อง reject", {**base_w, "relative_humidity_2m": 150})
check_raises("wind_speed ติดลบ -> ต้อง reject", {**base_w, "wind_speed_10m": -5})
check_raises("precipitation ติดลบ -> ต้อง reject", {**base_w, "precipitation": -1})
check_raises("surface_pressure นอกช่วง -> ต้อง reject", {**base_w, "surface_pressure": 300})
check_raises("temperature นอกช่วง (sentinel 9999) -> ต้อง reject", {**base_w, "temperature_2m": 9999})

_orig = app.USE_OUTLIER_FILTER
app.USE_OUTLIER_FILTER = False
try:
    check("USE_OUTLIER_FILTER=False -> ค่าผิดปกติไม่ถูกเช็คอีก (เฉพาะ key หายที่ยัง raise)",
          app._check_weather({**base_w, "temperature_2m": 9999}) is None)
finally:
    app.USE_OUTLIER_FILTER = _orig

print("\n=== [3] fetch_current_weather_batch(): ยังคง all-or-nothing เหมือนเดิม (ตั้งใจไม่เปลี่ยน) ===")
provinces = [{"province_en": "a", "lat": 16.0, "lon": 102.0}, {"province_en": "b", "lat": 17.0, "lon": 103.0}]
_orig_meteo_get = app._open_meteo_get
app._open_meteo_get = MagicMock(return_value=[
    {"current": dict(base_w)},
    {"current": {**base_w, "temperature_2m": 9999}},  # จังหวัด b ค่าผิดปกติ (sentinel)
])
try:
    raised = False
    try:
        app.fetch_current_weather_batch(provinces)
    except ValueError:
        raised = True
    check("ค่าผิดปกติจังหวัดเดียวทำให้ทั้ง batch raise เหมือนเดิม (ไม่ใช่ข้ามแค่จังหวัดนั้น)", raised)
finally:
    app._open_meteo_get = _orig_meteo_get

print("\n=== [4] integration: fetch_recent_pm25_openaq() กรอง outlier จริงผ่าน mock _openaq_get ===")
_orig_openaq_get = app._openaq_get
resp = MagicMock()
resp.json.return_value = {"results": [
    {"period": {"datetimeFrom": {"utc": "2026-01-01T00:00:00Z"}}, "value": 18},
    {"period": {"datetimeFrom": {"utc": "2026-01-01T01:00:00Z"}}, "value": 18},
    {"period": {"datetimeFrom": {"utc": "2026-01-01T02:00:00Z"}}, "value": 17.9},
    {"period": {"datetimeFrom": {"utc": "2026-01-01T03:00:00Z"}}, "value": 57.6},  # outlier
    {"period": {"datetimeFrom": {"utc": "2026-01-01T04:00:00Z"}}, "value": 16.8},
    {"period": {"datetimeFrom": {"utc": "2026-01-01T05:00:00Z"}}, "value": 17},
]}
app._openaq_get = MagicMock(return_value=resp)
try:
    df_out = app.fetch_recent_pm25_openaq(sensor_id=1, hours_back=72)
    check("outlier (57.6) ไม่อยู่ใน df ที่คืนกลับมาแล้ว (ถูก dropna ทิ้งไปหลังกรอง)",
          not (df_out["pm25"] == 57.6).any())
    check("เหลือ 5 แถว (6 แถว - 1 ที่ถูกกรอง)", len(df_out) == 5)
finally:
    app._openaq_get = _orig_openaq_get

print("\n" + "=" * 60)
if all_ok:
    print("ทั้งหมดผ่าน")
else:
    print("มีบางเคส FAIL — ดูรายละเอียดด้านบน")
    raise SystemExit(1)

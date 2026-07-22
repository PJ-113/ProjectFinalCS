"""
เช็ค missing rate ของข้อมูลดิบจาก Air4Thai (station 46t, getHistoryData.php) เทียบกับ
OpenAQ raw (sensor 1304249) ในช่วงเวลาเดียวกัน เพื่อดูว่า Air4Thai ครบกว่าจริงไหม

พบว่า Air4Thai มี hard retention limit ~2 เดือน (ขอย้อนไปไกลกว่านั้นได้ 0 แถวเสมอ)
จึงเทียบเฉพาะช่วงที่ Air4Thai คืนข้อมูลมาได้จริงเท่านั้น
"""
import pandas as pd
import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OPENAQ_CACHE_PATH = os.path.join(_ROOT, r"scripts\_openaq_raw_5y_cache.csv")

url = "https://air4thai.com/forweb/getHistoryData.php"
headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

# ทดสอบ retention limit ก่อน: ขอช่วงเก่ากว่า ~2 เดือนดูว่าได้ข้อมูลไหม
print("ทดสอบ retention limit ของ Air4Thai:")
for sdate, edate in [("2025-05-16", "2025-06-16"), ("2025-09-01", "2025-10-01"), ("2026-01-01", "2026-02-01")]:
    params = {"stationID": "46t", "param": "PM25", "type": "hr", "sdate": sdate, "edate": edate, "stime": "00", "etime": "23"}
    r = requests.get(url, params=params, headers=headers, timeout=30, verify=False)
    n = len(r.json()["stations"][0]["data"])
    print(f"  {sdate} -> {edate}: {n} แถว")

# ดึงช่วงล่าสุดที่มีข้อมูลจริง
params = {"stationID": "46t", "param": "PM25", "type": "hr", "sdate": "2025-05-16", "edate": "2026-07-01", "stime": "00", "etime": "23"}
r = requests.get(url, params=params, headers=headers, timeout=30, verify=False)
recs = r.json()["stations"][0]["data"]
df = pd.DataFrame(recs).rename(columns={"DATETIMEDATA": "datetime_utc", "PM25": "pm25"})
df["datetime_utc"] = pd.to_datetime(df["datetime_utc"]).dt.tz_localize("Asia/Bangkok").dt.tz_convert("UTC")

start, end = df["datetime_utc"].min(), df["datetime_utc"].max()
n_expected = len(pd.date_range(start, end, freq="h", tz="UTC"))
n_actual = df["pm25"].notna().sum()
air4thai_missing_pct = (n_expected - n_actual) / n_expected * 100

print(f"\nAir4Thai จริงคืนข้อมูลได้แค่ช่วง: {start} -> {end}")
print(f"Air4Thai missing rate ในช่วงนี้: {air4thai_missing_pct:.2f}% ({n_actual}/{n_expected})")

# เทียบกับ OpenAQ ในช่วงเดียวกันเป๊ะ (ใช้ cache ที่ดึงไว้จาก check_openaq_missing_rate_5y.py)
openaq_raw = pd.read_csv(OPENAQ_CACHE_PATH)
openaq_raw["datetime_utc"] = pd.to_datetime(openaq_raw["datetime_utc"])
openaq_window = openaq_raw[(openaq_raw["datetime_utc"] >= start) & (openaq_raw["datetime_utc"] <= end)]
openaq_missing_pct = (n_expected - len(openaq_window)) / n_expected * 100

print(f"OpenAQ missing rate ในช่วงเดียวกัน: {openaq_missing_pct:.2f}% ({len(openaq_window)}/{n_expected})")
print(f"\n>>> Air4Thai ครบกว่า OpenAQ ~{openaq_missing_pct / air4thai_missing_pct:.1f} เท่า ในช่วงที่ Air4Thai มีข้อมูล")
print(">>> แต่ Air4Thai มี retention window แค่ ~2 เดือน ใช้แทน OpenAQ สำหรับ retrain ระยะยาวไม่ได้")

"""
เก็บ archive ข้อมูล Air4Thai (station 46t) สะสมไว้ในไฟล์เดียวก่อนที่ retention ~2 เดือนของ
getHistoryData.php จะเลื่อนผ่านไป ทุกครั้งที่รันดึงข้อมูลย้อนหลัง FETCH_DAYS_BACK วันเต็ม
แล้ว merge + dedupe กับ archive เดิม
"""
import sys
from datetime import timedelta

import pandas as pd
import requests
import urllib3

sys.stdout.reconfigure(encoding="utf-8")
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

STATION_ID_AIR4THAI = "46t"
FETCH_DAYS_BACK = 58
import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARCHIVE_PATH = os.path.join(_ROOT, r"data\air4thai_archive.csv")
STALE_GAP_WARNING_DAYS = 55


def fetch_air4thai_history(days_back):
    edate = pd.Timestamp.now()
    sdate = edate - timedelta(days=days_back)

    url = "https://air4thai.com/forweb/getHistoryData.php"
    params = {
        "stationID": STATION_ID_AIR4THAI,
        "param": "PM25",
        "type": "hr",
        "sdate": sdate.strftime("%Y-%m-%d"),
        "edate": edate.strftime("%Y-%m-%d"),
        "stime": "00",
        "etime": "23",
    }
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}

    r = requests.get(url, params=params, headers=headers, timeout=30, verify=False)
    r.raise_for_status()
    records = r.json()["stations"][0]["data"]

    df = pd.DataFrame(records).rename(columns={"DATETIMEDATA": "datetime_utc", "PM25": "pm25"})
    df["datetime_utc"] = (
        pd.to_datetime(df["datetime_utc"])
        .dt.tz_localize("Asia/Bangkok")
        .dt.tz_convert("UTC")
    )
    df["pm25"] = pd.to_numeric(df["pm25"], errors="coerce")
    df = df.dropna(subset=["pm25"]).sort_values("datetime_utc").reset_index(drop=True)
    return df[["datetime_utc", "pm25"]]


def main():
    print(f"ดึง Air4Thai ย้อนหลัง {FETCH_DAYS_BACK} วัน (station {STATION_ID_AIR4THAI})...")
    fetched = fetch_air4thai_history(FETCH_DAYS_BACK)
    print(f"  ดึงได้ {len(fetched)} ชม. ({fetched['datetime_utc'].min()} -> {fetched['datetime_utc'].max()})")

    try:
        existing = pd.read_csv(ARCHIVE_PATH)
        existing["datetime_utc"] = pd.to_datetime(existing["datetime_utc"], utc=True)
        print(f"archive เดิม: {len(existing)} ชม. ({existing['datetime_utc'].min()} -> {existing['datetime_utc'].max()})")

        gap_days = (fetched["datetime_utc"].min() - existing["datetime_utc"].max()).total_seconds() / 86400
        if gap_days > 1:
            print(f"⚠️  ห่างจากครั้งก่อน {gap_days:.1f} วัน ({existing['datetime_utc'].max()} -> {fetched['datetime_utc'].min()})")
            if gap_days > STALE_GAP_WARNING_DAYS - FETCH_DAYS_BACK:
                print(f"⚠️⚠️  ช่องว่างนี้อาจเกินหน้าต่างที่ fetch ครั้งนี้ครอบคลุมได้ ({FETCH_DAYS_BACK} วัน) — ข้อมูลบางส่วนอาจหลุดไปแล้วถาวร")
    except FileNotFoundError:
        existing = None
        print("ยังไม่มี archive เดิม สร้างใหม่")

    combined = pd.concat([existing, fetched], ignore_index=True) if existing is not None else fetched
    combined = combined.drop_duplicates(subset=["datetime_utc"], keep="last").sort_values("datetime_utc").reset_index(drop=True)

    n_new = len(combined) - (len(existing) if existing is not None else 0)
    combined.to_csv(ARCHIVE_PATH, index=False)

    print(f"\nบันทึกแล้วที่: {ARCHIVE_PATH}")
    print(f"รวมทั้งหมด: {len(combined)} ชม. ({combined['datetime_utc'].min()} -> {combined['datetime_utc'].max()})")
    print(f"เพิ่มใหม่รอบนี้: {n_new} ชม.")


if __name__ == "__main__":
    main()

"""
ทดสอบการเข้าถึง NASA FIRMS API ก่อนดึงข้อมูลเต็ม: เช็ค MAP_KEY ใช้งานได้, ดึงข้อมูล NRT 5 วันล่าสุด
ครอบคลุมอีสาน+กันชนลาว, โครงสร้างข้อมูลที่ได้สมเหตุสมผล, ดึงข้อมูลย้อนหลังได้ และยืนยัน day_range สูงสุด
"""
import os

import pandas as pd
import requests

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(_ROOT, ".env")


def _load_env_key(name):
    with open(ENV_PATH, encoding="utf-8") as f:
        for line in f:
            if line.startswith(f"{name}="):
                return line.strip().split("=", 1)[1]
    raise RuntimeError(f"{name} not found in .env")


FIRMS_KEY = _load_env_key("FIRMS_KEY")

# west, south, east, north — ครอบคลุม 20 จังหวัดอีสาน + กันชนลาว/กัมพูชา
AREA = "100.5,13.5,106.5,19.0"
SOURCE_NRT = "VIIRS_NOAA20_NRT"
SOURCE_ARCHIVE = "VIIRS_NOAA20_SP"  # ใช้สำหรับข้อมูลเก่า


def fetch(source, day_range, date=None):
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{FIRMS_KEY}/{source}/{AREA}/{day_range}"
    if date:
        url += f"/{date}"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.text


def main():
    print(f"Bounding box (west,south,east,north): {AREA}")

    print("\n--- ทดสอบ 1: ข้อมูล NRT 5 วันล่าสุด (max day_range ต่อ request = 5) ---")
    text = fetch(SOURCE_NRT, 5)
    print(f"Response ยาว {len(text)} bytes, {len(text.splitlines())} บรรทัด (รวม header)")
    print("5 บรรทัดแรก:")
    for line in text.splitlines()[:5]:
        print(f"  {line}")

    if "Invalid" in text or "Error" in text[:200]:
        print("\n!!! ดูเหมือน MAP_KEY หรือ request มีปัญหา ตรวจสอบ response เต็ม:")
        print(text[:1000])
        return

    from io import StringIO
    df = pd.read_csv(StringIO(text))
    print(f"\nparse เป็น DataFrame ได้: {df.shape}, columns={list(df.columns)}")
    if not df.empty:
        print(df.head().to_string(index=False))
        print(f"\nช่วงวันที่ในผลลัพธ์: {df['acq_date'].min()} ถึง {df['acq_date'].max()}")

    print("\n--- ทดสอบ 2: ข้อมูลย้อนหลังไกลๆ (ต้นปี 2024, ใช้ SOURCE archive) ---")
    text_old = fetch(SOURCE_ARCHIVE, 5, date="2024-02-15")
    print(f"Response ยาว {len(text_old)} bytes, {len(text_old.splitlines())} บรรทัด")
    if "Invalid" in text_old or "Error" in text_old[:200]:
        print("!!! archive query มีปัญหา:")
        print(text_old[:1000])
    else:
        df_old = pd.read_csv(StringIO(text_old))
        print(f"parse ได้: {df_old.shape}")
        if not df_old.empty:
            print(f"ช่วงวันที่: {df_old['acq_date'].min()} ถึง {df_old['acq_date'].max()}")
            print(df_old.head(3).to_string(index=False))

    print("\n--- ทดสอบ 3: ยืนยัน limit จริง (day_range=6 ควร error ถ้า max=5) ---")
    try:
        text_wide = fetch(SOURCE_ARCHIVE, 6, date="2024-02-15")
        if "Invalid" in text_wide or "Error" in text_wide[:200]:
            print(f"day_range=6 ถูกปฏิเสธ (ยืนยัน max=5): {text_wide[:300]}")
        else:
            df_wide = pd.read_csv(StringIO(text_wide))
            print(f"day_range=6 ใช้ได้จริง (ไม่ตรงกับที่คาด): {df_wide.shape}")
    except requests.HTTPError as e:
        print(f"day_range=6 error (ยืนยัน max=5): {e}")


if __name__ == "__main__":
    main()

"""
ดึงข้อมูลจุดความร้อน (fire hotspot) จาก NASA FIRMS ย้อนหลังเต็มช่วงที่ training dataset มี (2024-01-02 ถึง 2026-07-02)

วน request ทีละ 5 วัน ด้วย source archive (_SP) ก่อนเสมอ ถ้า response ว่างเปล่าให้ fallback ไป
near-real-time (_NRT) แทนสำหรับ chunk นั้น บันทึกว่าแต่ละ chunk ใช้ source ไหนไว้ตรวจสอบย้อนหลังได้
Bounding box ครอบคลุมอีสาน + กันชนฝั่งลาว/กัมพูชา
"""
import os
import time
from io import StringIO

import pandas as pd
import requests

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENV_PATH = os.path.join(_ROOT, ".env")
DATA_DIR = os.path.join(_ROOT, "data")
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

START_DATE = pd.Timestamp("2024-01-02")
END_DATE = pd.Timestamp("2026-07-02")
DAY_RANGE = 5  # max ต่อ request
AREA = "100.5,13.5,106.5,19.0"
SOURCE_SP = "VIIRS_NOAA20_SP"
SOURCE_NRT = "VIIRS_NOAA20_NRT"
REQUEST_SLEEP_S = 1.5


def _load_env_key(name):
    with open(ENV_PATH, encoding="utf-8") as f:
        for line in f:
            if line.startswith(f"{name}="):
                return line.strip().split("=", 1)[1]
    raise RuntimeError(f"{name} not found in .env")


FIRMS_KEY = _load_env_key("FIRMS_KEY")


def fetch_chunk(source, date_str, day_range=DAY_RANGE):
    url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{FIRMS_KEY}/{source}/{AREA}/{day_range}/{date_str}"
    for attempt in range(4):
        try:
            resp = requests.get(url, timeout=30)
            if resp.status_code == 200:
                return resp.text
            if resp.status_code in (429, 502, 503, 504):
                time.sleep(15)
                continue
            resp.raise_for_status()
        except requests.exceptions.RequestException:
            time.sleep(10)
    raise RuntimeError(f"fetch_chunk failed after retries: {source} {date_str}")


def parse_hotspots(text):
    if not text.strip() or "Invalid" in text[:200] or "Error" in text[:200]:
        return None
    df = pd.read_csv(StringIO(text))
    return df if not df.empty else pd.DataFrame(columns=df.columns)


def main():
    print(f"ช่วงที่ดึง: {START_DATE.date()} ถึง {END_DATE.date()}  bounding box={AREA}")
    dates = pd.date_range(START_DATE, END_DATE, freq=f"{DAY_RANGE}D")
    n_chunks = len(dates)
    print(f"จำนวน chunk: {n_chunks} (day_range={DAY_RANGE} วัน/chunk)")

    all_frames = []
    source_log = []
    for i, chunk_start in enumerate(dates, 1):
        date_str = chunk_start.strftime("%Y-%m-%d")
        text_sp = fetch_chunk(SOURCE_SP, date_str)
        df_sp = parse_hotspots(text_sp)

        if df_sp is not None and not df_sp.empty:
            df_sp["_source_used"] = SOURCE_SP
            all_frames.append(df_sp)
            source_log.append({"chunk_start": date_str, "source": SOURCE_SP, "n_rows": len(df_sp)})
        else:
            time.sleep(REQUEST_SLEEP_S)
            text_nrt = fetch_chunk(SOURCE_NRT, date_str)
            df_nrt = parse_hotspots(text_nrt)
            if df_nrt is not None and not df_nrt.empty:
                df_nrt["_source_used"] = SOURCE_NRT
                all_frames.append(df_nrt)
                source_log.append({"chunk_start": date_str, "source": SOURCE_NRT, "n_rows": len(df_nrt)})
            else:
                source_log.append({"chunk_start": date_str, "source": "none", "n_rows": 0})

        if i % 20 == 0 or i == n_chunks:
            print(f"  [{i}/{n_chunks}] {date_str} — สะสม {sum(f['n_rows'] for f in source_log)} แถว")
        time.sleep(REQUEST_SLEEP_S)

    combined = pd.concat(all_frames, ignore_index=True) if all_frames else pd.DataFrame()
    print(f"\nรวมทั้งหมด: {len(combined)} แถว hotspot")

    combined["acq_datetime_utc"] = pd.to_datetime(
        combined["acq_date"] + " " + combined["acq_time"].astype(str).str.zfill(4),
        format="%Y-%m-%d %H%M", utc=True,
    )

    out_path = os.path.join(DATA_DIR, f"firms_hotspots_isaan_{TIMESTAMP}.csv")
    combined.to_csv(out_path, index=False)
    print(f"บันทึกที่: {out_path}")

    log_df = pd.DataFrame(source_log)
    log_path = os.path.join(DATA_DIR, f"firms_hotspots_fetch_log_{TIMESTAMP}.csv")
    log_df.to_csv(log_path, index=False)
    print(f"บันทึก fetch log ที่: {log_path}")
    print(f"\nสรุป source ที่ใช้: {log_df['source'].value_counts().to_dict()}")
    print(f"chunk ที่ไม่มีข้อมูลเลย (source=none): {(log_df['source'] == 'none').sum()} / {n_chunks}")

    combined["month"] = combined["acq_datetime_utc"].dt.month
    print("\nจำนวน hotspot ต่อเดือน (สังเกตความแตกต่างฤดูฝุ่น ก.พ.-เม.ย. คร่าวๆ):")
    print(combined.groupby("month").size().to_string())


if __name__ == "__main__":
    main()

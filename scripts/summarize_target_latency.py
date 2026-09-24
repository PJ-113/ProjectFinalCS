"""
สรุปความหน่วงปกติของ reading ล่าสุด (observed_lag_k) เพื่อตัดสินค่า IMPUTATION_MIN_K ใน app.py

อ่าน 2 แหล่ง (ถ้ามี):
  models/target_latency_scheduler_log.csv  — app.py บันทึกให้อัตโนมัติทุกครั้งที่งานรายชั่วโมงรัน (log_target_latency)
  models/target_data_latency_log.csv       — snapshot จาก scripts/measure_target_data_latency.py (รันมือ; เรียก /measurements ตรงๆ จึงนับเป็น live)
k ปกติ = 0 (reading ของชั่วโมงที่เพิ่งจบมาแล้ว) k>=1 = ช้ากว่าปกติ (ความหน่วงของ OpenAQ หรือสถานีเงียบ)
หลักตัดสิน (ตั้งไว้ก่อนดูข้อมูล): MIN_K ควรสูงกว่า k ของ "การรายงานปกติ" — ใช้ p99 ของ k เมื่อตัดจังหวัด/ช่วงที่เงียบยาว (k>=6 ถือว่าเป็นเหตุการณ์
สถานีเงียบ ไม่ใช่ความหน่วงปกติ) แล้ว MIN_K = p99 + 1 ; แนะนำอย่างน้อย 2; ต้องมีตัวอย่างอย่างน้อย 200 ค่า และครอบคลุมหลายนาทีของชั่วโมง

series_source (CHANGELOG 173): log ของ scheduler มีคอลัมน์ที่บอกว่า reading มาจากทางไหน — คำแนะนำคิดจากแถว `live` (= /measurements ทางหลัก) เท่านั้น
เพราะ hours_endpoint ช้ากว่า /measurements ~1 ชม. และ stale_cache มี k โตตามอายุ cache (ทั้งคู่ไม่ใช่ความหน่วงปกติของ OpenAQ) แถวเก่าที่ไม่มีคอลัมน์นี้
(เขียนก่อนมีทางสำรอง) นับเป็น live ; แถวที่ไม่ใช่ live รายงานแยกไว้ท้ายผลลัพธ์
รัน: python scripts/summarize_target_latency.py
"""
import os

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCHED = os.path.join(_ROOT, "models", "target_latency_scheduler_log.csv")
SNAP = os.path.join(_ROOT, "models", "target_data_latency_log.csv")
SCHED_COLS = ["measured_utc", "province_en", "observed_lag_k", "target_age_hours", "imputed", "series_source"]
MIN_SAMPLES = 200
NORMAL_MAX_K = 5  # k >= 6 ถือเป็นสถานีเงียบ ไม่นับเป็นความหน่วงปกติ


def load():
    frames = []
    if os.path.exists(SCHED):
        # header=0 + names=: ทนทั้งไฟล์ที่ย้าย header เป็น 6 คอลัมน์แล้วและไฟล์เก่า 5 คอลัมน์ ; แถวที่ขาดฟิลด์ท้าย (legacy) ได้ NaN -> live
        d = pd.read_csv(SCHED, header=0, names=SCHED_COLS)
        d["origin"] = "scheduler"
        d["series_source"] = d["series_source"].fillna("live")
        frames.append(d[["measured_utc", "province_en", "observed_lag_k", "origin", "series_source"]].rename(columns={"observed_lag_k": "k"}))
    if os.path.exists(SNAP):
        d = pd.read_csv(SNAP)
        d["origin"] = "snapshot"
        d["series_source"] = "live"
        frames.append(d[["measured_utc", "province_en", "k", "origin", "series_source"]])
    if not frames:
        raise SystemExit("ยังไม่มีข้อมูล — รัน app.py ให้งานรายชั่วโมงทำงานสักพัก หรือรัน scripts/measure_target_data_latency.py")
    df = pd.concat(frames, ignore_index=True).dropna(subset=["k"])
    df["measured_utc"] = pd.to_datetime(df["measured_utc"], utc=True, format="mixed")
    df["minute"] = df["measured_utc"].dt.minute
    return df


def summarize(all_rows):
    """พิมพ์สรุปและคืน dict {n_live, n_non_live, p99, recommended, conclusive} — คำแนะนำคิดจากแถว live เท่านั้น"""
    non_live = all_rows[all_rows["series_source"] != "live"]
    df = all_rows[all_rows["series_source"] == "live"].copy()
    n = len(df)
    if n == 0:
        raise SystemExit("ไม่มีแถว live เลย (ทุกแถวมาจากทางสำรอง) — สรุปความหน่วงปกติของ /measurements ไม่ได้")
    span = f"{df['measured_utc'].min():%Y-%m-%d %H:%M} .. {df['measured_utc'].max():%Y-%m-%d %H:%M} UTC"
    print(f"ตัวอย่าง live {n} ค่า ({df['origin'].value_counts().to_dict()}), ช่วงเวลา {span}, จังหวัด {df['province_en'].nunique()}")
    print("\nการกระจาย k ทั้งหมด (สัดส่วน):")
    print(df["k"].astype(int).value_counts(normalize=True).sort_index().round(4).to_string())

    normal = df[df["k"] <= NORMAL_MAX_K]
    p99 = float(normal["k"].quantile(0.99))
    print(f"\nk ที่ถือเป็นความหน่วงปกติ (k <= {NORMAL_MAX_K}): p50={normal['k'].quantile(.5):.0f} p95={normal['k'].quantile(.95):.0f} p99={p99:.0f}")
    print(f"สัดส่วน k >= 1: {(df['k'] >= 1).mean():.3%}  k >= 2: {(df['k'] >= 2).mean():.3%}  k >= 3: {(df['k'] >= 3).mean():.3%}  k >= 6: {(df['k'] >= 6).mean():.3%}")

    df["minute_bin"] = pd.cut(df["minute"], [-1, 14, 29, 44, 59], labels=["00-14", "15-29", "30-44", "45-59"])
    print("\nสัดส่วน k >= 1 ตามนาทีของชั่วโมงที่วัด (ความหน่วงมักสูงในนาทีต้นชั่วโมง):")
    print(df.groupby("minute_bin", observed=True)["k"].agg(n="size", share_k_ge_1=lambda s: (s >= 1).mean()).round(4).to_string())

    prov = df.groupby("province_en")["k"].agg(n="size", p_ge_1=lambda s: (s >= 1).mean(), p_ge_2=lambda s: (s >= 2).mean(), p_ge_6=lambda s: (s >= 6).mean())
    print("\nต่อจังหวัด (เรียงตามสัดส่วน k>=2):")
    print(prov.sort_values("p_ge_2", ascending=False).round(3).head(10).to_string())

    rec = max(2, int(p99) + 1)
    bins_covered = df["minute_bin"].nunique()
    print(f"\nข้อเสนอ IMPUTATION_MIN_K = {rec}  (= max(2, p99 ของ k ปกติ + 1))")
    conclusive = not (n < MIN_SAMPLES or bins_covered < 3)
    if not conclusive:
        print(f"** ยังสรุปไม่ได้: ตัวอย่าง {n} (<{MIN_SAMPLES}) หรือครอบคลุมนาทีของชั่วโมงแค่ {bins_covered}/4 ช่วง — เก็บข้อมูลต่อ **")

    print("\n" + "-" * 80)
    if non_live.empty:
        print("แถวจากทางสำรอง (hours_endpoint / stale_cache): ไม่มี — ข้อมูลทั้งหมดมาจาก /measurements")
    else:
        print(f"** มีแถวจากทางสำรอง {len(non_live)} แถว — ตัดออกจากคำแนะนำข้างบนแล้ว (ไม่ใช่ความหน่วงปกติของ /measurements) **")
        for src, g in non_live.groupby("series_source"):
            print(f"  {src}: {len(g)} แถว, การกระจาย k = {g['k'].astype(int).value_counts().sort_index().to_dict()}")
    return {"n_live": n, "n_non_live": int(len(non_live)), "p99": p99, "recommended": rec, "conclusive": conclusive}


def main():
    return summarize(load())


if __name__ == "__main__":
    main()

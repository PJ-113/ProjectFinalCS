"""
Phase A (feedback อาจารย์ 2026-09-30, ข้อ 2): ตรวจสมมติฐานว่าแนวโน้มฝุ่นเพื่อนบ้านช่วง "กลางคืน" (18:00-06:00 ไทย, ความชื้น/น้ำค้าง
ทำให้ฝุ่นไม่ลอยตัว) กับช่วง "กลางวัน" (06:00-18:00 ไทย, แดด/การพาความร้อนทำให้ฝุ่นลอยตัวเจือจาง) มีผลต่อการพยากรณ์ตอนเช้า
(07:00-09:00, ใช้ตัดสินใจเดินทาง/ไปโรงเรียน/ใส่หน้ากากจริง) ต่างกันไหม — ก่อนสร้าง feature จริงและ retrain (Phase B)

นิยาม (ไม่มีในโค้ดเดิม กำหนดขึ้นใหม่สำหรับงานนี้): กลางวัน/กลางคืนใช้เวลานาฬิกาคงที่ (ไทยใกล้เส้นศูนย์สูตร พระอาทิตย์ขึ้น/ตกใกล้ 06:00/18:00
ตลอดปี ไม่ต้องคำนวณดาราศาสตร์) "เช้า" = 06:00-09:00 ไทย เพื่อไม่ให้ scan ทุก (t,h) ที่ target ตกช่วงเช้า (ซับซ้อนเกินจำเป็นสำหรับ
diagnostic ราคาถูก) จึงยึดจุดอ้างอิงเดียวที่ตรงกับ use case จริงที่สุด: **t = 06:00 น. พอดี** (จุดธรรมชาติของ "พยากรณ์เช้านี้") แล้วดู
h=1,2,3 (เป้าหมาย 07:00/08:00/09:00) ข้อดี: nb_trend_12h ที่มีอยู่แล้วในข้อมูล (= neighbor_avg(t)-neighbor_avg(t-12h)) ที่แถว t=06:00
คือ "แนวโน้มกลางคืนล่าสุด" (18:00 เมื่อวาน -> 06:00 วันนี้) พอดีเป๊ะ ไม่ต้องสร้าง feature ใหม่ ส่วน "แนวโน้มกลางวันเมื่อวาน" (06:00->18:00
เมื่อวาน) = ค่า nb_trend_12h ของแถวเดียวกันแต่เมื่อ 12 ชม.ก่อน (18:00 เมื่อวาน) — แค่ shift-merge ค่าคอลัมน์เดิม

เกณฑ์ตัดสิน (ตั้งก่อนรัน): ไปต่อ Phase B ถ้า |r(กลางคืน) - r(กลางวัน)| >= 0.05 กับ pm25_t+1/2/3 (เฉลี่ย 3 horizon) และทั้งสองค่ามี
p<0.05 (ทั้งรวมทุกฤดูหรือเฉพาะฤดูฝุ่นอย่างใดอย่างหนึ่งก็พอ) ไม่ผ่าน = หยุด รายงานผลลบตามจริง ไม่ไป Phase B
รัน: python scripts/experiment_diurnal_nb_trend_hypothesis.py
"""
import glob
import json
import os

import numpy as np
import pandas as pd
from scipy import stats

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR, MODEL_DIR = os.path.join(_ROOT, "data"), os.path.join(_ROOT, "models")
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
DAWN_HOUR = 6            # t = 06:00 ไทย พอดี (จุดอ้างอิง "พยากรณ์เช้านี้")
MORNING_HORIZONS = [1, 2, 3]  # -> เป้าหมาย 07:00/08:00/09:00
BURNING_MONTHS = {2, 3, 4}
R_DIFF_THRESHOLD = 0.05
P_THRESHOLD = 0.05


def main():
    man = json.load(open(sorted(glob.glob(os.path.join(DATA_DIR, "rebuild_clean_manifest_*.json")))[-1], encoding="utf-8"))
    df = pd.read_csv(man["final"], usecols=["province_en", "datetime_utc", "is_zero_shot", "pm25_neighbor_avg", "nb_trend_12h"]
                      + [f"pm25_t+{h}" for h in MORNING_HORIZONS])
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df = df[~df["is_zero_shot"]].reset_index(drop=True)
    print(f"dataset สะอาด: {man['final']}\nแถว 17 จังหวัดเทรน: {len(df)}")

    local_hour = df["datetime_utc"].dt.tz_convert("Asia/Bangkok").dt.hour
    dawn = df[local_hour == DAWN_HOUR].copy()
    print(f"แถวที่ t=06:00 ไทยพอดี: {len(dawn)} ({dawn['province_en'].nunique()} จังหวัด)")

    # แนวโน้มกลางคืนล่าสุด (18:00 เมื่อวาน -> 06:00 วันนี้) = nb_trend_12h ของแถวนี้เป๊ะ (นิยามเดิมของ compute_neighbor_trend_feature.py)
    dawn = dawn.rename(columns={"nb_trend_12h": "night_trend"})

    # แนวโน้มกลางวันเมื่อวาน (06:00 -> 18:00 เมื่อวาน) = nb_trend_12h ของแถวเดียวกันเมื่อ 12 ชม.ก่อน (18:00 เมื่อวาน) — time-indexed lookup
    lag = df[["province_en", "datetime_utc", "nb_trend_12h"]].copy()
    lag["datetime_utc"] = lag["datetime_utc"] + pd.Timedelta(hours=12)
    lag = lag.rename(columns={"nb_trend_12h": "day_trend"})
    n0 = len(dawn)
    dawn = dawn.merge(lag, on=["province_en", "datetime_utc"], how="left")
    assert len(dawn) == n0, "merge ทำให้จำนวนแถวเปลี่ยน"

    # ตรวจซ้ำอิสระผ่าน pivot ของ pm25_neighbor_avg (เหมือน compute_neighbor_trend_feature.py) — คนละวิธีคำนวณ (pivot+reindex
    # แทน merge) เพื่อจับบั๊กที่อาจซ่อนอยู่ในวิธีหลักไม่ให้หลุดรอด
    piv = df.pivot_table(index="datetime_utc", columns="province_en", values="pm25_neighbor_avg")
    night_ref = (piv - piv.shift(freq=pd.Timedelta(hours=12))).stack().rename("ref_night").reset_index()
    day_ref = ((piv.shift(freq=pd.Timedelta(hours=12)) - piv.shift(freq=pd.Timedelta(hours=24)))
               .stack().rename("ref_day").reset_index())
    chk = dawn.merge(night_ref, on=["datetime_utc", "province_en"], how="left").merge(day_ref, on=["datetime_utc", "province_en"], how="left")
    ok_night = np.allclose(chk["night_trend"].fillna(-999).values, chk["ref_night"].fillna(-999).values)
    ok_day = np.allclose(chk["day_trend"].fillna(-999).values, chk["ref_day"].fillna(-999).values)
    assert ok_night and ok_day, f"recompute ไม่ตรง (night={ok_night}, day={ok_day})"
    print("night_trend/day_trend ตรงกับ recompute ผ่าน pivot (allclose)")

    dawn = dawn.dropna(subset=["night_trend", "day_trend"])
    print(f"แถวที่มีทั้ง night_trend และ day_trend ครบ: {len(dawn)}")
    burning = dawn["datetime_utc"].dt.month.isin(BURNING_MONTHS).values

    def corr_report(mask, label):
        sub = dawn[mask]
        if len(sub) < 30:
            print(f"  [{label}] ข้ามเช็ค (n={len(sub)} น้อยเกินไป)")
            return None
        results = {}
        for feat in ("night_trend", "day_trend", "night_trend_minus_day_trend"):
            vals = sub["night_trend"] - sub["day_trend"] if feat == "night_trend_minus_day_trend" else sub[feat]
            rs, ps = [], []
            for h in MORNING_HORIZONS:
                y = sub[f"pm25_t+{h}"]
                m = y.notna() & vals.notna()
                r, p = stats.pearsonr(vals[m], y[m])
                rs.append(r)
                ps.append(p)
            results[feat] = (float(np.mean(rs)), float(np.max(ps)))  # max p = อนุรักษ์นิยม (ต้อง significant ทั้ง 3 horizon)
            print(f"  [{label}] {feat}: r เฉลี่ย (h=1,2,3)={np.mean(rs):+.4f}  p สูงสุด={np.max(ps):.4g}  n={m.sum()}")
        return results

    print("\n=== รวมทุกฤดู ===")
    r_all = corr_report(np.ones(len(dawn), dtype=bool), "รวมทุกฤดู")
    print("\n=== เฉพาะฤดูฝุ่น (ก.พ.-เม.ย.) ===")
    r_burn = corr_report(burning, "ฤดูฝุ่น")
    print("\n=== นอกฤดูฝุ่น ===")
    r_nonburn = corr_report(~burning, "นอกฤดูฝุ่น")

    out = dawn[["province_en", "datetime_utc", "night_trend", "day_trend"] + [f"pm25_t+{h}" for h in MORNING_HORIZONS]]
    out_path = os.path.join(MODEL_DIR, f"experiment_diurnal_nb_trend_hypothesis_{TIMESTAMP}.csv")
    out.to_csv(out_path, index=False)
    print(f"\nข้อมูลดิบ: {out_path}")

    print("\n" + "=" * 90 + "\nสรุปตามเกณฑ์ที่ตั้งไว้ก่อนรัน\n" + "=" * 90)
    passed = False
    for label, res in (("รวมทุกฤดู", r_all), ("ฤดูฝุ่น", r_burn), ("นอกฤดูฝุ่น", r_nonburn)):
        if res is None:
            continue
        r_night, p_night = res["night_trend"]
        r_day, p_day = res["day_trend"]
        diff = abs(r_night - r_day)
        sig = p_night < P_THRESHOLD and p_day < P_THRESHOLD
        this_pass = diff >= R_DIFF_THRESHOLD and sig
        passed = passed or this_pass
        print(f"[{label}] |r(กลางคืน)-r(กลางวัน)| = {diff:.4f} (เกณฑ์ >= {R_DIFF_THRESHOLD})  "
              f"significant ทั้งคู่ (p<{P_THRESHOLD}): {sig}  -> {'ผ่าน' if this_pass else 'ไม่ผ่าน'}")
    print(f"\nสรุป: {'ผ่านอย่างน้อย 1 มุมมอง -> ไปต่อ Phase B (สร้าง feature จริง + time-blocked CV)' if passed else 'ไม่ผ่านทุกมุมมอง -> หยุดที่นี่ รายงานผลลบ'}")


if __name__ == "__main__":
    main()

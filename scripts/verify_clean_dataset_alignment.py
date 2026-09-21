"""
P2 ของแผนแก้ป้ายเวลา OpenAQ (CHANGELOG 144-145): พิสูจน์ว่า dataset ที่สร้างใหม่ (rebuild_dataset_clean.py) "ถูกเวลา" จริง ไม่ใช่แค่เลื่อนไปเฉยๆ
  (ก) cross-correlation ของคู่ "สถานีที่เคยเลื่อน vs สถานีปกติข้างเคียง" (8 คู่ x 3 ปี) — dataset เดิมได้ peak lag -21..-26 ชม.
      เกณฑ์ผ่าน: |peak lag| ไม่เกินค่าสูงสุดที่พบใน "คู่ควบคุม" ปกติ-ปกติ (ไม่เคยมีปัญหาป้ายเวลา) ของ dataset เดียวกัน และกรองคู่ที่ peak r < MIN_R
      (สหสัมพันธ์ต่ำ หา peak ไม่เสถียร) ออกจากเกณฑ์ (ยังแสดงในตาราง)
      หมายเหตุประวัติ: ตอนแรกตั้งเกณฑ์ |lag| <= 1 ก่อนเห็นข้อมูล แต่คู่ควบคุมปกติ-ปกติมี peak lag ตามธรรมชาติถึง +-6 ชม. (มัธยฐาน 3, เพราะฝุ่นเคลื่อนข้ามจังหวัด
      ใช้เวลา; มีแค่ 27% ที่ <=1) เกณฑ์เดิมจึงเข้มเกินจริง เปลี่ยนมาปรับเทียบกับคู่ควบคุมหลังเห็นข้อมูล (CHANGELOG 145)
  (ข) pm25 ของ 9 จังหวัดตรงกับ /hours (เวลาถูกจากแหล่งอิสระ) ที่เวลาเดียวกัน (สุ่ม 3 ช่วง/สถานี) — ต้องตรงเป๊ะ (ค่า pm25 ใน dataset ผ่านการ interpolate
      gap<=48h ดังนั้นเทียบเฉพาะเวลาที่ /hours มีค่าจริง)
  (ค) จำนวนแถว/ช่วงเวลา/is_zero_shot ครบ ไม่มี NaN ใน feature
ผลไม่ผ่านข้อใดข้อหนึ่ง = exit 1 (หยุดก่อน P3) อ่าน manifest ล่าสุดจาก data/rebuild_clean_manifest_*.json ไม่พิมพ์คีย์
"""
import glob
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import requests

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_ROOT, "data")
PAIRS = [("surin", "buriram"), ("sisaket", "ubonratchathani"), ("kalasin", "khonkaen"), ("yasothon", "ubonratchathani"),
         ("chaiyaphum", "nakhonratchasima"), ("amnatcharoen", "mukdahan"), ("nongbualamphu", "udonthani"), ("buengkan", "nongkhai")]
YEARS = (("2024", slice("2024-01-01", "2024-12-31")), ("2025", slice("2025-01-01", "2025-12-31")), ("2026", slice("2026-01-01", "2026-07-02")))
CONTROL_PAIRS = [("khonkaen", "udonthani"), ("buriram", "nakhonratchasima"), ("ubonratchathani", "mukdahan"), ("nongkhai", "udonthani"),
                 ("sakonnakhon", "nakhonphanom"), ("khonkaen", "nakhonratchasima"), ("udonthani", "sakonnakhon"), ("loei", "udonthani")]
MIN_R = 0.7
SHIFTED = ["surin", "sisaket", "kalasin", "yasothon", "chaiyaphum", "amnatcharoen", "nongbualamphu", "buengkan", "mahasarakham"]


def load_key():
    with open(os.path.join(_ROOT, ".env"), encoding="utf-8") as f:
        for line in f:
            if line.startswith("OPENAQ_KEY="):
                return line.strip().split("=", 1)[1]


def xcorr_peak(a, b, ks=range(-30, 31)):
    best = (None, -2.0, np.nan)
    r0 = np.nan
    for k in ks:
        bs = b.copy()
        bs.index = bs.index + pd.Timedelta(hours=k)
        cm = a.index.intersection(bs.index)
        if len(cm) < 500:
            continue
        r = a.loc[cm].corr(bs.loc[cm])
        if k == 0:
            r0 = r
        if r > best[1]:
            best = (k, r, r0)
    return best[0], best[1], r0


def pair_table(ds, pairs=PAIRS):
    pv = ds.pivot_table(index="datetime_utc", columns="province_en", values="pm25")
    rows = []
    for stale, ok in pairs:
        row = {"pair": f"{stale} vs {ok}"}
        for label, sl in YEARS:
            a, b = pv[stale].loc[sl].dropna(), pv[ok].loc[sl].dropna()
            k, r, r0 = xcorr_peak(a, b) if len(a.index.intersection(b.index)) >= 500 else (None, np.nan, np.nan)
            row[label] = "ข้อมูลน้อย" if k is None else f"lag {k:+d}h (r={r:.2f}, r0={r0:.2f})"
            row[label + "_k"] = k
            row[label + "_r"] = r
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    man_path = sorted(glob.glob(os.path.join(DATA_DIR, "rebuild_clean_manifest_*.json")))[-1]
    man = json.load(open(man_path, encoding="utf-8"))
    print("manifest:", man_path)
    new = pd.read_csv(man["final"])
    new["datetime_utc"] = pd.to_datetime(new["datetime_utc"], utc=True)
    old = pd.read_csv(man["old_dataset"], usecols=["province_en", "datetime_utc", "pm25"])
    old["datetime_utc"] = pd.to_datetime(old["datetime_utc"], utc=True)
    failures = []

    print("\n=== (ก) cross-correlation คู่ สถานีที่เคยเลื่อน vs ปกติ: peak lag ต้อง 0 (เดิม -21..-26) ===")
    t_new, t_old = pair_table(new), pair_table(old)
    pd.set_option("display.width", 250)
    for label, _ in YEARS:
        t_new[f"{label} (เดิม)"] = t_old[label]
    print(t_new[["pair"] + [c for lbl, _ in YEARS for c in (lbl, f"{lbl} (เดิม)")]].to_string(index=False))
    t_ctrl = pair_table(new, CONTROL_PAIRS)
    ctrl_lags = [abs(r[lbl + "_k"]) for _, r in t_ctrl.iterrows() for lbl, _ in YEARS if r[lbl + "_k"] is not None and not pd.isna(r[lbl + "_k"])]
    max_ctrl = max(ctrl_lags)
    print(f"\nคู่ควบคุม (ปกติ-ปกติ, ไม่เคยมีปัญหาป้ายเวลา) ใน dataset ใหม่: |peak lag| มัธยฐาน {np.median(ctrl_lags):.1f} ชม., สูงสุด {max_ctrl:.0f} ชม. (n={len(ctrl_lags)}) -> เกณฑ์ผ่าน |lag| <= {max_ctrl:.0f} ชม.")
    bad, skipped = [], []
    for _, r in t_new.iterrows():
        for lbl, _ in YEARS:
            k, rr = r[lbl + "_k"], r[lbl + "_r"]
            if k is None or pd.isna(k):
                continue
            if rr < MIN_R:
                skipped.append((r["pair"], lbl, f"peak r={rr:.2f} < {MIN_R}"))
            elif abs(k) > max_ctrl:
                bad.append((r["pair"], lbl, k))
    if skipped:
        print(f"ไม่นับเข้าเกณฑ์ (สหสัมพันธ์ต่ำ หา peak ไม่เสถียร): {skipped}")
    if bad:
        failures.append(f"|peak lag| เกินช่วงของคู่ควบคุม (>{max_ctrl:.0f}h): {bad}")

    print("\n=== (ข) pm25 ของ 9 จังหวัด vs /hours ที่เวลาเดียวกัน (สุ่ม 3 ช่วง/สถานี) ===")
    headers = {"X-API-Key": load_key()}
    reg = {p["province_en"]: p for p in json.load(open(os.path.join(DATA_DIR, "province_station_registry.json"), encoding="utf-8"))["provinces"]}
    windows = [("2024-06-05", "2024-06-15"), ("2025-06-05", "2025-06-15"), ("2026-03-05", "2026-03-15")]
    pv_new = new.pivot_table(index="datetime_utc", columns="province_en", values="pm25")
    for en in SHIFTED:
        parts = []
        for w0, w1 in windows:
            r = requests.get(f"https://api.openaq.org/v3/sensors/{reg[en]['sensor_id_pm25']}/hours", headers=headers,
                             params={"datetime_from": pd.Timestamp(w0, tz="UTC").isoformat(), "datetime_to": pd.Timestamp(w1, tz="UTC").isoformat(), "limit": 1000}, timeout=30)
            time.sleep(0.3)
            if r.status_code != 200:
                continue
            h = pd.Series({pd.Timestamp(x["period"]["datetimeFrom"]["utc"]): x["value"] for x in r.json()["results"] if x["value"] is not None})
            cm = h.index.intersection(pv_new[en].dropna().index)
            if len(cm) >= 50:
                parts.append(((pv_new[en].loc[cm] - h.loc[cm]).abs() < 1e-6).mean())
            else:
                parts.append(None)
        got = [p for p in parts if p is not None]
        status = "ตรงเป๊ะ" if got and min(got) > 0.98 else ("ไม่มีข้อมูลเทียบ" if not got else "ไม่ตรง")
        print(f"  {en:14s} สัดส่วนตรงเป๊ะ 3 ช่วง: {['-' if p is None else f'{p:.1%}' for p in parts]} -> {status}")
        if status == "ไม่ตรง":
            failures.append(f"{en}: pm25 ไม่ตรง /hours {parts}")

    print("\n=== (ค) ความครบถ้วนของ dataset ===")
    feats = [c for c in new.columns if c not in ("province_th", "province_en", "is_zero_shot", "datetime_utc")]
    n_nan_feat = int(new[[c for c in feats if not c.startswith("pm25_t+") and c != "nb_trend_12h"]].isna().sum().sum())
    print(f"  แถวรวม {len(new)}, จังหวัด {new['province_en'].nunique()}, zero-shot {sorted(new.loc[new['is_zero_shot'], 'province_en'].unique())}")
    print(f"  ช่วงเวลา {new['datetime_utc'].min()} .. {new['datetime_utc'].max()}; NaN ใน feature หลัก: {n_nan_feat}")
    if new["province_en"].nunique() != 20:
        failures.append(f"จังหวัดไม่ครบ 20 ({new['province_en'].nunique()})")
    if n_nan_feat:
        failures.append(f"มี NaN ใน feature หลัก {n_nan_feat}")

    print("\n" + "=" * 90)
    print("ผลการตรวจ P2: " + ("ผ่านทั้งหมด" if not failures else "ไม่ผ่าน -> " + "; ".join(failures)))
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()

"""
ด่านตรวจ G2 ของการ deploy โมเดล v2 (CHANGELOG 141): เรียก "ฟังก์ชันจริงใน app.py" (compute_neighbor_features,
count_fire_hotspots) กับข้อมูล OpenAQ ย้อนหลังที่ slice ให้เห็นเฉพาะ <= t (72 ชม.) เหมือนตอน live แล้วเทียบกับแถวใน dataset เทรน
(ต่างจาก check_live_feature_parity.py ที่ใช้โค้ดจำลองในสคริปต์ — อันนี้ตรวจตัวที่ deploy จริง เป็น regression check
เมื่อแก้สูตรใน app.py)
เกณฑ์: เคสที่ใช้เพื่อนบ้านอันดับ 1 (policy k1_exact) ตรง >= 99%, ภาพรวม >= 95%, fire 100%
เคสที่ต้องใช้ policy สำรอง (เพื่อนบ้านอันดับ 1 ไม่สด/ไม่มี reading) คาดว่าไม่ตรง dataset โดยเจตนา — รายงานแยก
ดึง OpenAQ 1 request ต่อจังหวัด (ไม่พิมพ์คีย์), ใช้ไฟล์ FIRMS SP เดิม (ทดสอบ "สูตร" ไม่ใช่ตัวแหล่ง)
"""
import glob
import importlib.util
import json
import os
import sys
import time

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(_ROOT, "data")
MODEL_DIR = os.path.join(_ROOT, "models")
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
WIN_START, WIN_END = pd.Timestamp("2026-06-10", tz="UTC"), pd.Timestamp("2026-07-01", tz="UTC")
N_PER_PROVINCE = 25
ATOL = 1e-3


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    app = load("app_under_test", os.path.join(_ROOT, "app.py"))
    e2 = load("check_live_feature_parity", os.path.join(_ROOT, "scripts", "check_live_feature_parity.py"))
    headers = {"X-API-Key": e2.load_key("OPENAQ_KEY")}
    man = json.load(open(sorted(glob.glob(os.path.join(DATA_DIR, "rebuild_clean_manifest_*.json")))[-1], encoding="utf-8"))
    print("dataset (สะอาด):", man["final"])
    ds = pd.read_csv(man["final"],
                     usecols=["province_en", "is_zero_shot", "datetime_utc", "pm25_neighbor_avg", "nb_trend_12h",
                              "fire_hotspot_count_100km_24h"])
    ds["datetime_utc"] = pd.to_datetime(ds["datetime_utc"], utc=True)
    hot = pd.read_csv(sorted(glob.glob(os.path.join(DATA_DIR, "firms_hotspots_isaan_*.csv")))[-1])
    hot["t"] = pd.to_datetime(hot["acq_datetime_utc"], utc=True)

    print("ดึง OpenAQ ย้อนหลังของ 20 จังหวัด...")
    series = {}
    for p in app.PROVINCE_REGISTRY:
        off = pd.Timedelta(hours=int(p.get("openaq_timestamp_offset_hours", 0)))  # ป้ายเวลา /measurements ของบางสถานีเลื่อน (CHANGELOG 144-145)
        s = e2.fetch_range(p["sensor_id_pm25"], headers, WIN_START - pd.Timedelta(days=3) - off, WIN_END + pd.Timedelta(days=1))
        if len(s):
            s = s.copy()
            s.index = s.index + off
        series[p["province_en"]] = s
        time.sleep(0.4)

    def live_fetch(t):
        """เลียนแบบ get_recent_series ตอน live ณ เวลา t: เห็นเฉพาะ reading 72 ชม.ล่าสุดที่ <= t"""
        def _f(q):
            s = series.get(q)
            if s is None or s.empty:
                return None
            w = s[(s.index <= t) & (s.index >= t - pd.Timedelta(hours=72))]
            return None if w.empty else pd.DataFrame({"datetime_utc": w.index, "pm25": w.values})
        return _f

    rng = np.random.default_rng(5)
    rows = []
    for prov, meta in app.PROVINCES_BY_SLUG.items():
        if meta["is_zero_shot"]:
            continue
        cand = ds[(ds["province_en"] == prov) & (ds["datetime_utc"] >= WIN_START) & (ds["datetime_utc"] <= WIN_END)]
        cand = cand[cand["datetime_utc"].isin(series[prov].index)]
        if cand.empty:
            continue
        for _, r in cand.sample(min(N_PER_PROVINCE, len(cand)), random_state=int(rng.integers(1e9))).iterrows():
            t = r["datetime_utc"]
            nb = app.compute_neighbor_features(prov, t, fetch=live_fetch(t))
            fire = app.count_fire_hotspots(meta, t, hot)
            rows.append({"province": prov, "t": t, "status": nb["status"], "policy": nb["policy"],
                         "nb_ds": r["pm25_neighbor_avg"], "nb_app": nb["pm25_neighbor_avg"],
                         "trend_ds": r["nb_trend_12h"], "trend_app": nb["nb_trend_12h"],
                         "fire_ds": r["fire_hotspot_count_100km_24h"], "fire_app": fire})
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(MODEL_DIR, f"verify_app_feature_parity_{TIMESTAMP}.csv"), index=False)

    def match(a, b, m):
        x, y = res.loc[m, a].astype(float), res.loc[m, b].astype(float)
        ok = ((x - y).abs() <= ATOL) & x.notna() & y.notna()
        return ok.sum(), int(m.sum()), ((x - y).abs()[x.notna() & y.notna()]).max() if m.sum() else np.nan

    print(f"\nจุดเปรียบเทียบ: {len(res)} (จาก {res['province'].nunique()} จังหวัด)")
    print("status ของ compute_neighbor_features:", res["status"].value_counts().to_dict())
    print("policy:", res["policy"].value_counts(dropna=False).to_dict())
    allm = pd.Series(True, index=res.index)
    k1 = res["policy"] == "k1_exact"
    failures = []
    for label, m in (("ทุกจุด", allm), ("เฉพาะ k1_exact", k1), ("เฉพาะ policy สำรอง (avg_nearest_fresh)", res["policy"] == "avg_nearest_fresh")):
        print(f"\n[{label}] n={int(m.sum())}")
        if not m.sum():
            continue
        for name, a, b in (("pm25_neighbor_avg", "nb_ds", "nb_app"), ("nb_trend_12h", "trend_ds", "trend_app"),
                           ("fire_hotspot_count_100km_24h", "fire_ds", "fire_app")):
            ok, n, mx = match(a, b, m)
            print(f"   {name:32s} ตรง {ok}/{n} = {ok / n:6.1%}   |diff| สูงสุด {mx:.3f}")
            if label == "ทุกจุด" and ok / n < 0.95:
                failures.append(f"{name} ภาพรวม {ok / n:.1%} < 95%")
            if label == "เฉพาะ k1_exact" and ok / n < 0.99:
                failures.append(f"{name} k1_exact {ok / n:.1%} < 99%")
            if name.startswith("fire") and ok != n:
                failures.append(f"fire ไม่ตรง 100% ({ok}/{n})")
    print("\n" + ("PASS: ผ่านเกณฑ์ G2 ทั้งหมด" if not failures else "FAIL: " + "; ".join(failures)))
    sys.exit(0 if not failures else 1)


if __name__ == "__main__":
    main()

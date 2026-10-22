"""
วัดป้ายเวลาที่เลื่อนกี่ชั่วโมงของแต่ละสถานี OpenAQ (เทียบ /hours กับ /measurements บน 2 ช่วงเวลา)
แล้วเขียนเป็น field `openaq_timestamp_offset_hours` ต่อจังหวัดใน data/province_station_registry.json
สถานีที่วัดไม่ได้/ไม่ตรงกันในสองช่วง จะไม่เขียนและรายงานเป็น "ต้องตรวจ" (exit code 1)
สำรองไฟล์เดิมเป็น data/province_station_registry_backup_{ts}.json ก่อนเขียนทุกครั้ง
"""
import importlib.util
import json
import os
import shutil
import sys
import time

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_PATH = os.path.join(_ROOT, "data", "province_station_registry.json")
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
FIELD = "openaq_timestamp_offset_hours"
DIFF_TOL = 0.01
MIN_POINTS = 100


def load_module(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, "scripts", fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    sh = load_module("shift_check", "check_openaq_timestamp_shift.py")
    headers = {"X-API-Key": sh.load_key()}
    with open(REGISTRY_PATH, encoding="utf-8") as f:
        registry = json.load(f)
    now = pd.Timestamp.now(tz="UTC")
    windows = [("ล่าสุด", now - pd.Timedelta(hours=400), now),
               ("2025-03", pd.Timestamp("2025-03-01", tz="UTC"), pd.Timestamp("2025-03-20", tz="UTC"))]

    results, problems = {}, []
    for p in registry["provinces"]:
        en, sid = p["province_en"], p["sensor_id_pm25"]
        found = []
        for label, start, end in windows:
            m = sh.fetch("measurements", sid, start, end, headers)
            h = sh.fetch("hours", sid, start, end, headers)
            time.sleep(0.3)
            if len(m) < MIN_POINTS or len(h) < MIN_POINTS:
                found.append((label, None, 0, float("nan")))
                continue
            k, n, d = sh.best_shift(m, h)
            found.append((label, k, n, d))
        ks = {k for _, k, _, _ in found}
        diffs_ok = all(d < DIFF_TOL for _, k, _, d in found if k is not None)
        desc = " | ".join(f"{label}: {'ไม่มีข้อมูล' if k is None else f'{k:+d}h (n={n}, diff={d:.4f})'}" for label, k, n, d in found)
        if len(ks) == 1 and None not in ks and diffs_ok:
            results[en] = int(next(iter(ks)))
            print(f"  {en:16s} offset = {results[en]:>2d}   [{desc}]")
        else:
            problems.append(en)
            print(f"  {en:16s} !! ต้องตรวจ (ไม่ตรงกัน/ข้อมูลไม่พอ) [{desc}]")

    backup = os.path.join(_ROOT, "data", f"province_station_registry_backup_{TIMESTAMP}.json")
    shutil.copy2(REGISTRY_PATH, backup)
    for p in registry["provinces"]:
        if p["province_en"] in results:
            p[FIELD] = results[p["province_en"]]
    registry["timestamp_offset_measured_at_utc"] = now.isoformat()
    registry["needs_timestamp_offset_review"] = problems
    with open(REGISTRY_PATH, "w", encoding="utf-8") as f:
        json.dump(registry, f, ensure_ascii=False, indent=2)
        f.write("\n")
    shifted = sorted(en for en, k in results.items() if k != 0)
    print(f"\nสำรอง registry เดิมที่: {backup}")
    print(f"เขียน field {FIELD} แล้ว {len(results)}/{len(registry['provinces'])} จังหวัด; offset != 0: {len(shifted)} จังหวัด -> {shifted}")
    if problems:
        print(f"ต้องตรวจ: {problems}")
        sys.exit(1)


if __name__ == "__main__":
    main()

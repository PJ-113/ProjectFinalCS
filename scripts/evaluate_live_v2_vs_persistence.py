"""
เทียบโมเดลที่ deploy อยู่บนข้อมูลสด กับ persistence (พยากรณ์ = ค่าปัจจุบันตอนพยากรณ์) โดยอ่าน
matched pairs จาก prediction_log/notification_log ที่มี actual_pm25 แล้ว (อ่านอย่างเดียว)
วัด MAE/bias/coverage ต่อช่วง horizon, ต่อโมเดล, ต่อจังหวัด ที่ horizon 12-24 ชม.

เกณฑ์ตัดสิน: n < MIN_PAIRS_FAR -> "ข้อมูลยังไม่พอ"; improve_pct >= IMPROVE_PASS (10%) -> PASS;
improve_pct > 0 แต่ < 10% -> WARN; improve_pct <= 0 -> FAIL
"""
import os

import numpy as np
import pandas as pd
import psycopg2

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SINCE_UTC = "2026-09-19"  # วันที่ deploy v3 — กรองเฉพาะแถว *_v3
MIN_PAIRS_FAR = 300
IMPROVE_PASS = 0.10
FAR_HORIZONS = (12, 24)
HORIZON_BUCKETS = [("t+1", 1, 1), ("t+2..6", 2, 6), ("t+7..12", 7, 12), ("t+13..18", 13, 18), ("t+19..24", 19, 24)]


def load_env():
    env = {}
    with open(os.path.join(_ROOT, ".env"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def fetch_rows():
    env = load_env()
    conn = psycopg2.connect(host=env["POSTGRES_HOST"], port=env["POSTGRES_PORT"], dbname=env["POSTGRES_DB"],
                            user=env["POSTGRES_USER"], password=env["POSTGRES_PASSWORD"])
    cur = conn.cursor()
    cur.execute(
        """SELECT p.province, p.horizon, p.model_used, p.predicted_at_utc, p.target_time_utc, p.predicted_pm25,
                  p.actual_pm25, p.pm25_lo, p.pm25_hi, n.latest_pm25
           FROM prediction_log p
           LEFT JOIN notification_log n ON n.recorded_at_utc = p.predicted_at_utc AND n.province = p.province
           WHERE p.model_used LIKE '%%_v3' AND p.predicted_at_utc >= %s""",
        (SINCE_UTC,),
    )
    cols = ["province", "horizon", "model_used", "predicted_at_utc", "target_time_utc", "predicted_pm25", "actual_pm25",
            "pm25_lo", "pm25_hi", "latest_pm25"]
    df = pd.DataFrame(cur.fetchall(), columns=cols)
    cur.close()
    conn.close()
    return df


def summarize(sub):
    """MAE ของ v2 เทียบ persistence บนแถวเดียวกัน + bias + coverage ของช่วงความเชื่อมั่น"""
    if len(sub) == 0:
        return {"n": 0, "mae_v2": np.nan, "mae_persistence": np.nan, "improve_pct": np.nan, "bias_v2": np.nan,
                "coverage_pct": np.nan, "width": np.nan}
    err_v2 = (sub["actual_pm25"] - sub["predicted_pm25"]).abs()
    err_ps = (sub["actual_pm25"] - sub["latest_pm25"]).abs()
    has_int = sub["pm25_lo"].notna() & sub["pm25_hi"].notna()
    inside = ((sub["actual_pm25"] >= sub["pm25_lo"]) & (sub["actual_pm25"] <= sub["pm25_hi"]))[has_int]
    mae_v2, mae_ps = err_v2.mean(), err_ps.mean()
    return {"n": len(sub), "mae_v2": mae_v2, "mae_persistence": mae_ps,
            "improve_pct": (mae_ps - mae_v2) / mae_ps * 100 if mae_ps > 0 else np.nan,
            "bias_v2": (sub["predicted_pm25"] - sub["actual_pm25"]).mean(),
            "coverage_pct": inside.mean() * 100 if len(inside) else np.nan,
            "width": (sub.loc[has_int, "pm25_hi"] - sub.loc[has_int, "pm25_lo"]).mean() if has_int.any() else np.nan}


def evaluate(df):
    """df: แถวจาก fetch_rows() -> dict(matched, pending, no_persistence, by_bucket, by_model_far, by_province_far, verdict, verdict_detail)"""
    pending = int(df["actual_pm25"].isna().sum())
    with_actual = df[df["actual_pm25"].notna()]
    no_persistence = int(with_actual["latest_pm25"].isna().sum())
    matched = with_actual[with_actual["latest_pm25"].notna()].copy()

    rows = []
    for label, lo, hi in HORIZON_BUCKETS:
        rows.append({"horizon": label, **summarize(matched[(matched["horizon"] >= lo) & (matched["horizon"] <= hi)])})
    by_bucket = pd.DataFrame(rows)

    far = matched[(matched["horizon"] >= FAR_HORIZONS[0]) & (matched["horizon"] <= FAR_HORIZONS[1])]
    by_model_far = pd.DataFrame(
        [{"model": "ทุกโมเดล v2", **summarize(far)}] +
        [{"model": m, **summarize(far[far["model_used"] == m])} for m in sorted(far["model_used"].unique())]
    )
    by_province_far = pd.DataFrame([{"province": p, **summarize(g)} for p, g in far.groupby("province")])
    if len(by_province_far):
        by_province_far = by_province_far.sort_values("improve_pct")

    s = summarize(far)
    if s["n"] < MIN_PAIRS_FAR:
        verdict = "ข้อมูลยังไม่พอ"
        detail = (f"มีคู่ที่จับคู่ได้ที่ horizon {FAR_HORIZONS[0]}-{FAR_HORIZONS[1]} ชม. เพียง {s['n']} คู่ (ต้อง >= {MIN_PAIRS_FAR}) "
                  f"— รอให้ระบบทำงานต่อ (พยากรณ์ล่วงหน้า 24 ชม. ต้องรอครบ 1 วันถึงจะได้ค่าจริง)")
    elif s["improve_pct"] >= IMPROVE_PASS * 100:
        verdict = "PASS"
        detail = f"v2 ดีกว่า persistence {s['improve_pct']:.1f}% (>= {IMPROVE_PASS:.0%}) ที่ horizon {FAR_HORIZONS[0]}-{FAR_HORIZONS[1]} ชม."
    elif s["improve_pct"] > 0:
        verdict = "WARN"
        detail = f"v2 ดีกว่า persistence เพียง {s['improve_pct']:.1f}% (< {IMPROVE_PASS:.0%}) — ต่ำกว่าที่คาด (ผลย้อนหลัง ~16%) ควรดูรายจังหวัด"
    else:
        verdict = "FAIL"
        detail = (f"v2 แย่กว่า persistence ({s['improve_pct']:.1f}%) — สงสัยปัญหาที่ pipeline ข้อมูลสด (เพื่อนบ้าน/จุดความร้อน/ข้อมูลเก่า) "
                  f"ก่อนสรุปว่าโมเดลมีปัญหา ให้ดู data_quality ของ /api/forecast และตารางรายจังหวัดด้านบน")
    return {"matched": matched, "pending": pending, "no_persistence": no_persistence, "by_bucket": by_bucket,
            "by_model_far": by_model_far, "by_province_far": by_province_far, "verdict": verdict, "verdict_detail": detail,
            "far_summary": s}


def _fmt(t):
    return t.round(3).to_string(index=False)


def main():
    df = fetch_rows()
    print(f"แถว v2 ใน prediction_log ตั้งแต่ {SINCE_UTC}: {len(df)} (ชื่อโมเดล: {sorted(df['model_used'].unique().tolist())})")
    if df.empty:
        print("ยังไม่มีแถวของ v2 — รัน python app.py ให้ scheduler พยากรณ์ก่อน")
        return
    r = evaluate(df)
    print(f"ได้ค่าจริงแล้ว {len(df) - r['pending']} แถว | รอค่าจริง {r['pending']} แถว | ไม่มี persistence (ไม่มี notification_log) {r['no_persistence']} แถว")
    print(f"คู่ที่ใช้เทียบ (matched): {len(r['matched'])} คู่ ใน {r['matched']['province'].nunique() if len(r['matched']) else 0} จังหวัด, "
          f"ช่วงเวลาเป้าหมาย {r['matched']['target_time_utc'].min() if len(r['matched']) else '-'} .. {r['matched']['target_time_utc'].max() if len(r['matched']) else '-'}")
    print("\n=== MAE ของ v2 เทียบ persistence ตามช่วง horizon (improve_pct บวก = v2 ดีกว่า) ===")
    print(_fmt(r["by_bucket"]))
    print(f"\n=== horizon {FAR_HORIZONS[0]}-{FAR_HORIZONS[1]} ชม. แยกโมเดล ===")
    print(_fmt(r["by_model_far"]))
    if len(r["by_province_far"]):
        print(f"\n=== horizon {FAR_HORIZONS[0]}-{FAR_HORIZONS[1]} ชม. แยกจังหวัด (เรียงจาก improve_pct น้อยไปมาก) ===")
        print(_fmt(r["by_province_far"]))
    print("\n" + "=" * 90)
    print(f"ผลตัดสิน: {r['verdict']}")
    print(r["verdict_detail"])
    print("หมายเหตุ: coverage เป้าหมาย 90% (ผลย้อนหลัง ~88-89%); ช่วงข้อมูลที่มีอาจเป็นฤดูฝุ่นต่ำ ยังไม่ใช่การทดสอบตอนฝุ่นสูง")


if __name__ == "__main__":
    main()

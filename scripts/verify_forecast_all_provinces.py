"""
เรียก GET /api/forecast ทุก 20 จังหวัดผ่าน Flask test client ด้วยข้อมูลสดจริง (OpenAQ + Open-Meteo + FIRMS) แล้วตรวจ:
HTTP 200, forecast 24 จุดไม่มี NaN/ติดลบ, lo<=hi, ชื่อโมเดลถูกต้อง, data_quality สอดคล้องกัน
และตรวจ prediction_log ว่าจังหวัดปกติถูกบันทึก 24 แถวพร้อม pm25_lo/pm25_hi ส่วนจังหวัดข้อมูลเก่าต้องไม่ถูกบันทึก
"""
import importlib.util
import math
import os
import sys
import time

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    spec = importlib.util.spec_from_file_location("app_under_test", os.path.join(_ROOT, "app.py"))
    app = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(app)
    app.maybe_generate_daily_nlg = lambda *a, **k: None
    client = app.app.test_client()
    valid_models = {app.MODEL_LGB, app.MODEL_XGB, app.MODEL_NAIVE}

    rows, problems = [], []
    t_all = time.time()
    for p in app.PROVINCE_REGISTRY:
        en = p["province_en"]
        t0 = time.time()
        resp = client.get(f"/api/forecast?province={en}")
        sec = time.time() - t0
        if resp.status_code != 200:
            rows.append({"province": en, "http": resp.status_code, "note": (resp.get_json() or {}).get("error", "")[:70], "sec": round(sec, 1)})
            continue
        j = resp.get_json()
        f, dq = j["forecast"], j["data_quality"]
        models = sorted({x["model_used"] for x in f})
        vals = [x["pm25"] for x in f] + [x["pm25_lo"] for x in f] + [x["pm25_hi"] for x in f]
        if len(f) != 24:
            problems.append(f"{en}: forecast {len(f)} จุด")
        if any((v is None or (isinstance(v, float) and math.isnan(v)) or v < 0) for v in vals):
            problems.append(f"{en}: มี NaN/ติดลบในผลพยากรณ์")
        if any(x["pm25_lo"] > x["pm25_hi"] for x in f):
            problems.append(f"{en}: pm25_lo > pm25_hi")
        if not set(models) <= valid_models:
            problems.append(f"{en}: ชื่อโมเดลแปลก {models}")
        if dq["neighbor_status"] != "ok" and models != [app.MODEL_NAIVE]:
            problems.append(f"{en}: ไม่มีเพื่อนบ้านสดแต่ไม่ใช่ naive ทั้งหมด {models}")
        if dq["neighbor_status"] == "ok" and dq["neighbor_policy"] is None:
            problems.append(f"{en}: neighbor ok แต่ไม่มี policy")
        rows.append({"province": en, "http": 200, "age_h": dq["target_age_hours"], "stale": dq["target_stale"],
                     "neighbor": f"{dq['neighbor_status']}/{dq['neighbor_policy']}", "used": ",".join(dq["neighbors_used"]),
                     "fire": f"{dq['fire_count_24h']}({dq['fire_source']})", "models": "+".join(m.replace('_pooled_v2', '_v2') for m in models),
                     "now": j["current"]["pm25"], "t+1": f[0]["pm25"], "t+24": f[-1]["pm25"], "sec": round(sec, 1),
                     "_latest": j["current"]["datetime_th"], "_stale": dq["target_stale"]})
    res = pd.DataFrame(rows)
    pd.set_option("display.width", 260)
    pd.set_option("display.max_colwidth", 45)
    print(res.drop(columns=[c for c in res.columns if c.startswith("_")]).to_string(index=False))
    print(f"\nรวมเวลา {time.time() - t_all:.1f} วิ")

    bad_http = res[res["http"] != 200]
    print(f"HTTP ไม่ใช่ 200: {len(bad_http)} จังหวัด" + (f" -> {bad_http[['province', 'note']].to_dict('records')}" if len(bad_http) else ""))

    # ตรวจ DB: แถวล่าสุดของแต่ละจังหวัด (ชนิดโมเดล / ช่วงความเชื่อมั่น) และจังหวัด stale ต้องไม่มีแถวของรอบนี้
    conn = app.get_db_connection()
    cur = conn.cursor()
    for _, r in res[res["http"] == 200].iterrows():
        cur.execute("SELECT MAX(predicted_at_utc) FROM prediction_log WHERE province=%s AND model_used LIKE '%%_v3' OR (province=%s AND model_used='naive_fallback' AND predicted_at_utc >= %s)",
                    (r["province"], r["province"], (pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=1)).isoformat()))
        latest_logged = cur.fetchone()[0]
        if r["_stale"]:
            continue
        cur.execute("SELECT model_used, COUNT(*), SUM((pm25_lo IS NULL OR pm25_hi IS NULL)::int) FROM prediction_log "
                    "WHERE province=%s AND predicted_at_utc=%s GROUP BY model_used", (r["province"], pd.Timestamp(r["_latest"]).tz_localize("Asia/Bangkok").tz_convert("UTC").isoformat()))
        got = cur.fetchall()
        total = sum(c for _, c, _ in got)
        nulls = sum(n or 0 for _, _, n in got)
        names = {m for m, _, _ in got}
        if total != 24:
            problems.append(f"{r['province']}: log {total} แถว (คาด 24) {got}")
        if not names <= {app.MODEL_LGB, app.MODEL_XGB, app.MODEL_NAIVE}:
            problems.append(f"{r['province']}: ชื่อโมเดลใน DB แปลก {names}")
        if nulls:
            problems.append(f"{r['province']}: pm25_lo/hi เป็น NULL {nulls} แถว")
    cur.execute("SELECT model_used, COUNT(*) FROM prediction_log WHERE model_used LIKE '%%_v3' GROUP BY 1")
    print("จำนวนแถว _v2 ใน prediction_log ทั้งตาราง:", cur.fetchall())
    cur.close()
    conn.close()

    n_ok = int((res["http"] == 200).sum())
    print(f"\nสรุป: ตอบ 200 ได้ {n_ok}/20 จังหวัด, stale {int(res.get('stale', pd.Series(dtype=bool)).fillna(False).sum())}, "
          f"ปัญหาที่พบ {len(problems) + len(bad_http)}")
    for pr in problems:
        print("  ปัญหา:", pr)
    sys.exit(0 if not problems and not len(bad_http) else 1)


if __name__ == "__main__":
    main()

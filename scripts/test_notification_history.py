"""
ทดสอบ /api/notifications (ประวัติการแจ้งเตือน, CHANGELOG ถัดจาก 57)

เขียนแถวทดสอบลง notification_log + prediction_log จริงใน Postgres โดยใช้ predicted_at_utc /
recorded_at_utc ปี 2100 (marker เดียวกับ test_dynamic_routing.py) แล้วลบทิ้งทั้งก่อน/หลังใน finally

ครอบคลุม:
- risk_horizons คำนวณสดตาม risk_group ของผู้เรียก ไม่ใช่ค่าที่ freeze ไว้ตอนบันทึก — ชั่วโมงที่
  predicted_pm25 เกิน threshold ของกลุ่มเปราะบาง (base-10) แต่ไม่เกินของ "ทั่วไป" (base) ต้องปรากฏใน
  ประวัติของกลุ่มเปราะบางเท่านั้น
- ชั่วโมงที่เงียบสนิท (ทุกอย่างต่ำกว่า threshold ทุกกลุ่ม) ต้องไม่ปรากฏในประวัติของใครเลย
- ชั่วโมงที่เกิน threshold ของ "ทั่วไป" ด้วย ต้องปรากฏในประวัติทุกกลุ่ม
- prophet_horizons สะท้อน model_used='prophet' ถูกต้อง แม้ horizon นั้นเองจะไม่ใช่ตัวที่ทำให้
  high_pollution_risk เป็นจริง (เป็นแค่ metadata แนบไปกับแถวที่ผ่านเกณฑ์รวมแล้ว)
"""
import importlib.util

import pandas as pd

import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_PATH = os.path.join(_ROOT, r"app.py")
_TEST_MARKER = "2100-"


def load_app():
    spec = importlib.util.spec_from_file_location("app_under_test", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clear_test_rows(app):
    conn = app.get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM prediction_log WHERE predicted_at_utc LIKE %s", (f"{_TEST_MARKER}%",))
    cur.execute("DELETE FROM notification_log WHERE recorded_at_utc LIKE %s", (f"{_TEST_MARKER}%",))
    conn.commit()
    cur.close()
    conn.close()


def insert_snapshot(app, recorded_at_utc, latest_pm25, horizon_rows):
    """horizon_rows: list of (horizon, predicted_pm25, model_used)"""
    conn = app.get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO notification_log (recorded_at_utc, latest_pm25) VALUES (%s, %s)",
        (recorded_at_utc, latest_pm25),
    )
    target = pd.Timestamp(recorded_at_utc) + pd.Timedelta(hours=1)
    cur.executemany(
        """INSERT INTO prediction_log
           (predicted_at_utc, target_time_utc, horizon, predicted_pm25, model_used, source)
           VALUES (%s, %s, %s, %s, %s, 'live')""",
        [(recorded_at_utc, target.isoformat(), h, pred, model) for h, pred, model in horizon_rows],
    )
    conn.commit()
    cur.close()
    conn.close()


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    return condition


def main():
    app = load_app()
    app.app.testing = True
    all_ok = True

    clear_test_rows(app)
    try:
        base_h1 = app.EFFECTIVE_THRESHOLD_BY_HORIZON_BY_MODEL["lightgbm"][1]

        # แถว 1: ชายขอบ — เกิน threshold กลุ่มเปราะบาง (base-10) แต่ไม่เกินของทั่วไป (base)
        # แถมมี horizon อื่นที่ใช้ prophet (ค่าต่ำ ไม่เสี่ยง) ไว้เช็ค prophet_horizons metadata
        borderline_pm25 = base_h1 - 5.0
        insert_snapshot(
            app, f"{_TEST_MARKER}01-01T00:00:00+00:00", 15.0,
            [(1, borderline_pm25, "lightgbm"), (2, 10.0, "prophet")],
        )

        # แถว 2: เงียบสนิท ทุกอย่างต่ำกว่า threshold ทุกกลุ่ม
        insert_snapshot(
            app, f"{_TEST_MARKER}01-02T00:00:00+00:00", 10.0,
            [(1, 10.0, "lightgbm"), (2, 12.0, "lightgbm")],
        )

        # แถว 3: เกิน threshold ของทั่วไปด้วย (base+5) ต้องปรากฏทุกกลุ่ม
        above_general_pm25 = base_h1 + 5.0
        insert_snapshot(
            app, f"{_TEST_MARKER}01-03T00:00:00+00:00", 20.0,
            [(1, above_general_pm25, "lightgbm")],
        )

        clients = {}
        for rg in ["ทั่วไป", "เด็ก-ผู้สูงอายุ", "มีโรคประจำตัว"]:
            client = app.app.test_client()
            client.post("/api/auth/register", json={
                "email": f"notif-hist-test-{rg}@test-marker.invalid", "password": "correcthorse1",
                "risk_group": rg, "age": 30,
            })
            clients[rg] = client
        guest_client = app.app.test_client()

        results = {}
        for rg, client in clients.items():
            r = client.get("/api/notifications?limit=100")
            all_ok &= check(f"{rg}: GET /api/notifications -> 200", r.status_code == 200)
            body = r.get_json()
            results[rg] = {row["recorded_at_th"]: row for row in body if row["recorded_at_th"].startswith("2100")}

        r = guest_client.get("/api/notifications?limit=100")
        guest_results = {row["recorded_at_th"]: row for row in r.get_json() if row["recorded_at_th"].startswith("2100")}

        # กรอง key เฉพาะ marker rows (แปลง UTC -> ไทยแล้วดูแค่ prefix วันที่)
        def keys_for(rg_results):
            return set(rg_results.keys())

        n_general = len(keys_for(results["ทั่วไป"]))
        n_child = len(keys_for(results["เด็ก-ผู้สูงอายุ"]))
        n_chronic = len(keys_for(results["มีโรคประจำตัว"]))
        n_guest = len(keys_for(guest_results))

        print(f"จำนวนแถว marker ที่ปรากฏ: ทั่วไป={n_general}, เด็ก-ผู้สูงอายุ={n_child}, มีโรคประจำตัว={n_chronic}, guest={n_guest}")

        all_ok &= check("ทั่วไป: เห็นแค่แถว 3 (เกิน threshold จริง) ไม่เห็นแถวชายขอบ/เงียบ", n_general == 1)
        all_ok &= check("เด็ก-ผู้สูงอายุ: เห็น 2 แถว (ชายขอบ + แถว 3) ไม่เห็นแถวเงียบ", n_child == 2)
        all_ok &= check("มีโรคประจำตัว: เห็น 2 แถว เหมือนเด็ก-ผู้สูงอายุ", n_chronic == 2)
        all_ok &= check("guest เห็นเท่ากับ ทั่วไป เป๊ะ", n_guest == n_general)

        # เช็ค prophet_horizons ของแถวชายขอบ (ต้องมี horizon 2 อยู่ แม้ predicted_pm25 ต่ำ)
        borderline_row = next(iter(results["เด็ก-ผู้สูงอายุ"].values()), None)
        # หาแถวที่มี latest_pm25 == 15.0 (แถวชายขอบ ไม่ใช่แถว 3 ที่ latest_pm25=20.0)
        borderline_row = next((row for row in results["เด็ก-ผู้สูงอายุ"].values() if row["latest_pm25"] == 15.0), None)
        all_ok &= check("แถวชายขอบมีอยู่จริงในผลของกลุ่มเปราะบาง", borderline_row is not None)
        if borderline_row:
            all_ok &= check("แถวชายขอบ: risk_horizons == [1]", borderline_row["risk_horizons"] == [1])
            all_ok &= check("แถวชายขอบ: prophet_horizons == [2] (metadata แม้ horizon 2 เองไม่เสี่ยง)",
                             borderline_row["prophet_horizons"] == [2])

        # เช็คว่าแถวเงียบ (latest_pm25=10.0 อยู่ที่ 2100-01-02) ไม่ปรากฏในผลของใครเลย
        for rg, rows in {**results, "guest": guest_results}.items():
            has_quiet = any(row["latest_pm25"] == 10.0 and "01-02" in row["recorded_at_th"] for row in rows.values())
            all_ok &= check(f"{rg}: ไม่เห็นแถวเงียบ (2100-01-02)", not has_quiet)

        # เช็ค limit param
        r = clients["ทั่วไป"].get("/api/notifications?limit=1")
        all_ok &= check("limit=1 คืนแค่ 1 แถว", len(r.get_json()) <= 1)
    finally:
        clear_test_rows(app)
        conn = app.get_db_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM users WHERE email LIKE %s", ("notif-hist-test-%@test-marker.invalid",))
        conn.commit()
        cur.close()
        conn.close()

    print("\n" + ("ALL TESTS PASSED" if all_ok else "SOME TESTS FAILED"))
    import sys
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

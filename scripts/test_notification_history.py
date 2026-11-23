"""
ทดสอบ /api/notifications (ประวัติการแจ้งเตือน)

เขียนแถวทดสอบลง notification_log + prediction_log จริงด้วย predicted_at_utc /
recorded_at_utc ปี 2100 แล้วลบทิ้งทั้งก่อน/หลังใน finally

ครอบคลุม:
- risk_horizons คำนวณสดตาม risk_group ของผู้เรียก ไม่ใช่ค่าที่ freeze ไว้ตอนบันทึก — ชั่วโมงที่
  predicted_pm25 เกิน threshold ของกลุ่มเปราะบาง (base-10) แต่ไม่เกินของ "ทั่วไป" (base) ต้องปรากฏใน
  ประวัติของกลุ่มเปราะบางเท่านั้น
- ชั่วโมงที่เงียบสนิท (ทุกอย่างต่ำกว่า threshold ทุกกลุ่ม) ต้องไม่ปรากฏในประวัติของใครเลย
- ชั่วโมงที่เกิน threshold ของ "ทั่วไป" ด้วย ต้องปรากฏในประวัติทุกกลุ่ม
- fallback_horizons สะท้อน model_used==MODEL_NAIVE ถูกต้อง แม้ horizon นั้นเองจะไม่ใช่ตัวที่ทำให้
  high_pollution_risk เป็นจริง (เป็นแค่ metadata แนบไปกับแถวที่ผ่านเกณฑ์รวมแล้ว)

หมายเหตุ (แก้ 2026-11-22): ไฟล์นี้เคยอ้างอิง EFFECTIVE_THRESHOLD_BY_HORIZON_BY_MODEL/prophet_horizons ซึ่งเป็น
กลไกของสถาปัตยกรรมเก่า (Khon Kaen เดี่ยว ก่อน pooled v3) ที่ถูกถอดออกไปแล้ว — ยืนยันว่ารันไม่ผ่านจริงก่อนแก้
(AttributeError) แก้ให้ใช้ effective_threshold_for()/fallback_horizons/MODEL_NAIVE ตามสถาปัตยกรรมปัจจุบันแทน
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


def insert_snapshot(app, recorded_at_utc, latest_pm25, horizon_rows, trend_direction=None, trend_period=None):
    """horizon_rows: list of (horizon, predicted_pm25, model_used)"""
    conn = app.get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO notification_log (recorded_at_utc, latest_pm25, trend_direction, trend_period) "
        "VALUES (%s, %s, %s, %s)",
        (recorded_at_utc, latest_pm25, trend_direction, trend_period),
    )
    target = pd.Timestamp(recorded_at_utc) + pd.Timedelta(hours=1)
    if horizon_rows:
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


def test_notification_history_basic(app):
    all_ok = True

    clear_test_rows(app)
    try:
        base_h1 = app.effective_threshold_for("ทั่วไป")

        # แถว 1: ชายขอบ — เกิน threshold กลุ่มเปราะบาง (base-10) แต่ไม่เกินของทั่วไป (base)
        # แถมมี horizon อื่นที่ใช้ MODEL_NAIVE (ค่าต่ำ ไม่เสี่ยง) ไว้เช็ค fallback_horizons metadata
        borderline_pm25 = base_h1 - 5.0
        insert_snapshot(
            app, f"{_TEST_MARKER}01-01T00:00:00+00:00", 15.0,
            [(1, borderline_pm25, app.MODEL_LGB), (2, 10.0, app.MODEL_NAIVE)],
        )

        # แถว 2: เงียบสนิท ทุกอย่างต่ำกว่า threshold ทุกกลุ่ม
        insert_snapshot(
            app, f"{_TEST_MARKER}01-02T00:00:00+00:00", 10.0,
            [(1, 10.0, app.MODEL_LGB), (2, 12.0, app.MODEL_LGB)],
        )

        # แถว 3: เกิน threshold ของทั่วไปด้วย (base+5) ต้องปรากฏทุกกลุ่ม
        above_general_pm25 = base_h1 + 5.0
        insert_snapshot(
            app, f"{_TEST_MARKER}01-03T00:00:00+00:00", 20.0,
            [(1, above_general_pm25, app.MODEL_LGB)],
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

        # เช็ค fallback_horizons ของแถวชายขอบ (ต้องมี horizon 2 อยู่ แม้ predicted_pm25 ต่ำ)
        # หาแถวที่มี latest_pm25 == 15.0 (แถวชายขอบ ไม่ใช่แถว 3 ที่ latest_pm25=20.0)
        borderline_row = next((row for row in results["เด็ก-ผู้สูงอายุ"].values() if row["latest_pm25"] == 15.0), None)
        all_ok &= check("แถวชายขอบมีอยู่จริงในผลของกลุ่มเปราะบาง", borderline_row is not None)
        if borderline_row:
            all_ok &= check("แถวชายขอบ: risk_horizons == [1]", borderline_row["risk_horizons"] == [1])
            all_ok &= check("แถวชายขอบ: fallback_horizons == [2] (metadata แม้ horizon 2 เองไม่เสี่ยง)",
                             borderline_row["fallback_horizons"] == [2])

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

    return all_ok


def test_level_transition(app):
    """ข้อ 1 'ระดับเปลี่ยนสถานะ' — ใช้ province=khonkaen (ค่า default ของ notification_log) แยก date
    prefix เป็นของตัวเอง (2100-02-xx) ไม่ให้ปนกับ test_notification_history_basic (2100-01-xx) เพราะ
    _level_transition_for() เทียบกับแถวก่อนหน้าจริงๆ ถ้าแถวของเทสอื่นมาคั่นกลางจะเพี้ยน"""
    all_ok = True
    clear_test_rows(app)
    try:
        client = app.app.test_client()
        client.post("/api/auth/register", json={
            "email": "notif-transition-test@test-marker.invalid", "password": "correcthorse1",
            "risk_group": "ทั่วไป", "age": 30,
        })

        # ชุด A: แกว่งไปมา (30->40->30->40 รายชั่วโมง) ต้องขึ้นแค่ครั้งแรก ครั้งหลังถูก debounce
        for i, pm25 in enumerate([30.0, 40.0, 30.0, 40.0]):
            insert_snapshot(app, f"{_TEST_MARKER}02-01T0{i}:00:00+00:00", pm25, [])

        r = client.get("/api/notifications?limit=100")
        all_ok &= check("level_transition ชุดแกว่ง: GET -> 200", r.status_code == 200)
        rows_a = {row["recorded_at_th"]: row for row in r.get_json() if row["recorded_at_th"].startswith("2100-02-01")}

        row_01 = rows_a.get("2100-02-01 08:00")  # UTC 01:00 -> ไทย +7
        row_02 = rows_a.get("2100-02-01 09:00")  # UTC 02:00 (30 กลับมาอีกรอบ)
        row_03 = rows_a.get("2100-02-01 10:00")  # UTC 03:00 (40 กลับมาอีกรอบ)
        all_ok &= check("ชม.ที่ 2 (30->40 ครั้งแรก): transition ขึ้นจริง และ direction=worse",
                         bool(row_01) and row_01.get("transition") is not None
                         and row_01["transition"]["direction"] == "worse")
        # pm25=30 ไม่เสี่ยง (threshold ทั่วไป=37.5) และ transition ถูก debounce พอดี -> ไม่มีอะไรน่าสนใจเลย
        # แถวนี้จึงไม่ถูกส่งกลับมาใน response เลย (ไม่ใช่ส่งมาพร้อม transition=None)
        all_ok &= check("ชม.ที่ 3 (40->30 แกว่งกลับ, ไม่เสี่ยง): ถูก debounce จนไม่มีอะไรน่าสนใจ ไม่โผล่ใน response เลย",
                         row_02 is None)
        all_ok &= check("ชม.ที่ 4 (30->40 แกว่งกลับอีกรอบ): ถูก debounce เช่นกัน",
                         row_03 is not None and row_03.get("transition") is None)

        clear_test_rows(app)

        # ชุด B: ไต่ระดับต่อเนื่องจริง (30->40->80) ต้องขึ้นทุกครั้ง ไม่ถูก debounce เพราะแต่ละระดับใหม่จริง
        for i, pm25 in enumerate([30.0, 40.0, 80.0]):
            insert_snapshot(app, f"{_TEST_MARKER}02-02T0{i}:00:00+00:00", pm25, [])

        r = client.get("/api/notifications?limit=100")
        rows_b = {row["recorded_at_th"]: row for row in r.get_json() if row["recorded_at_th"].startswith("2100-02-02")}
        row_b1 = rows_b.get("2100-02-02 08:00")  # UTC 01:00 (30->40)
        row_b2 = rows_b.get("2100-02-02 09:00")  # UTC 02:00 (40->80)
        all_ok &= check("ไต่ระดับจริง ชม.ที่ 2 (30->40): ขึ้นปกติ ไม่ถูก debounce",
                         bool(row_b1) and row_b1.get("transition") is not None)
        all_ok &= check("ไต่ระดับจริง ชม.ที่ 3 (40->80): ขึ้นปกติ ไม่ถูก debounce",
                         bool(row_b2) and row_b2.get("transition") is not None
                         and row_b2["transition"]["levels_moved"] >= 1)
    finally:
        clear_test_rows(app)
        conn = app.get_db_connection()
        cur = conn.cursor()
        cur.execute("DELETE FROM users WHERE email = %s", ("notif-transition-test@test-marker.invalid",))
        conn.commit()
        cur.close()
        conn.close()
    return all_ok


def test_imminent_risk(app):
    """ข้อ 2 'ใกล้ถึงช่วงเสี่ยงแล้ว' — horizon=1 (<=IMMINENT_HORIZON_H) ต้อง risk_imminent=True,
    horizon=10 (>IMMINENT_HORIZON_H) ต้อง risk_imminent=False แม้ risk_horizons ไม่ว่างก็ตาม"""
    all_ok = True
    clear_test_rows(app)
    try:
        client = app.app.test_client()
        base = app.effective_threshold_for("ทั่วไป")

        insert_snapshot(app, f"{_TEST_MARKER}03-01T00:00:00+00:00", 20.0, [(1, base + 10.0, app.MODEL_LGB)])
        insert_snapshot(app, f"{_TEST_MARKER}03-02T00:00:00+00:00", 20.0, [(10, base + 10.0, app.MODEL_LGB)])

        r = client.get("/api/notifications?limit=100")
        rows = {row["recorded_at_th"]: row for row in r.get_json() if row["recorded_at_th"].startswith("2100-03")}
        row_near = rows.get("2100-03-01 07:00")
        row_far = rows.get("2100-03-02 07:00")

        all_ok &= check("horizon=1 ถึงเพดาน: risk_imminent=True", bool(row_near) and row_near["risk_imminent"] is True)
        all_ok &= check("horizon=1: min_risk_horizon_pm25 ตรงกับค่าที่ใส่",
                         bool(row_near) and row_near["min_risk_horizon_pm25"] == base + 10.0)
        # recorded_at_utc=2100-03-01T00:00 UTC, horizon=1, OPENAQ_DISPLAY_LABEL_SHIFT_H=1 -> UTC 02:00 -> ไทย 09:00
        all_ok &= check("horizon=1: min_risk_horizon_display_time = 09:00 (บวก OPENAQ_DISPLAY_LABEL_SHIFT_H ด้วย)",
                         bool(row_near) and row_near["min_risk_horizon_display_time"] == "09:00")
        all_ok &= check("horizon=10 ถึงเพดาน: risk_imminent=False (แม้ risk_horizons ไม่ว่าง)",
                         bool(row_far) and row_far["risk_imminent"] is False and row_far["risk_horizons"] == [10])
    finally:
        clear_test_rows(app)
    return all_ok


def test_improving_trend(app):
    """ข้อ 3 'แนวโน้มฝุ่นจะดีขึ้น' — trend_direction/trend_period ต้องส่งผ่านมาถึง response ตรงๆ
    และแถว NULL (จำลองข้อมูลก่อน migration) ต้องไม่ error"""
    all_ok = True
    clear_test_rows(app)
    try:
        client = app.app.test_client()
        insert_snapshot(app, f"{_TEST_MARKER}04-01T00:00:00+00:00", 20.0, [],
                         trend_direction="down", trend_period="เย็น-กลางคืน")
        insert_snapshot(app, f"{_TEST_MARKER}04-02T00:00:00+00:00", 20.0, [])  # ไม่มี trend (NULL)

        r = client.get("/api/notifications?limit=100")
        all_ok &= check("improving_trend: GET -> 200", r.status_code == 200)
        rows = {row["recorded_at_th"]: row for row in r.get_json() if row["recorded_at_th"].startswith("2100-04")}
        row_trend = rows.get("2100-04-01 07:00")
        row_null = rows.get("2100-04-02 07:00")

        all_ok &= check("แถวมี trend: trend_direction='down', trend_period ตรงกับที่ใส่",
                         bool(row_trend) and row_trend["trend_direction"] == "down"
                         and row_trend["trend_period"] == "เย็น-กลางคืน")
        # แถว NULL ไม่ได้อยู่ในเงื่อนไข "น่าสนใจ" ข้อไหนเลย (ไม่เสี่ยง/ไม่เปลี่ยนระดับ/ไม่ใช่เทรนด์ดีขึ้น) จึงไม่
        # ควรโผล่ใน /api/notifications เลย — เช็คว่า endpoint ไม่ error ตอนเจอ NULL (ไม่ใช่เช็คว่าต้องโผล่)
        all_ok &= check("แถว NULL trend ไม่ทำให้ endpoint error (ไม่ error = ผ่านแล้ว)", True)
        if row_null is not None:
            all_ok &= check("แถว NULL trend: trend_direction เป็น None ไม่ error", row_null["trend_direction"] is None)
    finally:
        clear_test_rows(app)
    return all_ok


def test_station_silence(app):
    """ข้อ 4 'สถานีเงียบนานผิดปกติ' — เทส _station_silence_for() ตรงๆ แบบ unit test (ไม่ผ่าน DB/HTTP)
    เพราะเทียบกับเวลา 'ตอนนี้' จริง ใช้ marker ปี 2100 (อนาคต) ไม่ได้เพราะจะได้ค่าห่างติดลบ ไม่ใช่ค่าบวกเหมือน
    ข้อมูลเก่าจริง — ส่ง now_utc เข้าไปเองแทนเพื่อควบคุมผลให้แน่นอน ไม่พึ่งพาข้อมูลจริงในฐานข้อมูลที่คุมไม่ได้"""
    all_ok = True
    now = pd.Timestamp("2026-01-01T12:00:00+00:00")
    stale_hours, data_stale = app._station_silence_for((now - pd.Timedelta(hours=8)).isoformat(), now_utc=now)
    all_ok &= check("เงียบ 8 ชม. (> TARGET_STALE_HOURS=6): data_stale=True", data_stale is True)
    all_ok &= check("เงียบ 8 ชม.: stale_hours คำนวณถูกต้อง (~8.0)", abs(stale_hours - 8.0) < 0.01)

    fresh_hours, fresh_stale = app._station_silence_for((now - pd.Timedelta(hours=1)).isoformat(), now_utc=now)
    all_ok &= check("เงียบแค่ 1 ชม. (< TARGET_STALE_HOURS=6): data_stale=False", fresh_stale is False)
    return all_ok


def main():
    app = load_app()
    app.app.testing = True

    all_ok = True
    all_ok &= test_notification_history_basic(app)
    all_ok &= test_level_transition(app)
    all_ok &= test_imminent_risk(app)
    all_ok &= test_improving_trend(app)
    all_ok &= test_station_silence(app)

    print("\n" + ("ALL TESTS PASSED" if all_ok else "SOME TESTS FAILED"))
    import sys
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

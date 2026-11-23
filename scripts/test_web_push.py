"""
ทดสอบ dispatch_push_for_province() (Web Push Phase 3) — ไม่ส่ง push จริง (monkeypatch app.send_push เป็น
stub ที่แค่บันทึกว่าถูกเรียกด้วย args อะไร คืน True เสมอ) เพื่อตรวจ logic การตัดสินใจ/dedup ล้วนๆ โดยไม่ต้อง
พึ่ง VAPID key จริง/เครือข่ายจริง

เขียนแถวทดสอบลง notification_log/prediction_log/push_subscriptions จริงด้วย marker ปี 2100 (province ปลอม
ที่ไม่ชนกับ 20 จังหวัดจริง) แล้วลบทิ้งทั้งก่อน/หลังใน finally — ตาม convention เดียวกับ
scripts/test_notification_history.py

ครอบคลุม:
- เงื่อนไข "threshold" (เกินเพดาน) ยิง push ครั้งแรกที่เข้าเกณฑ์ แล้วไม่ยิงซ้ำอีกตราบใดที่ยังเกินอยู่ต่อเนื่อง
  (dedup sentinel "active" ไม่ใช่ค่า pm25 ดิบที่แกว่งได้)
- เงื่อนไข "transition" ยิง 1 ครั้งตอนเปลี่ยนระดับจริง แล้วเคลียร์ตัวเองในรอบถัดไป (ไม่ค้าง)
- เงื่อนไข "silence" ใช้ now_utc override ทดสอบได้ (เลี่ยงปัญหา marker ปี 2100 อยู่ในอนาคตเทียบเวลาจริง)
- subscription ของผู้ใช้ risk_group ต่างกัน (ผ่าน user_id) เห็นเงื่อนไข "threshold"/"imminent" ต่างกันได้
  ตามที่ risk_group ของตัวเอง ไม่ใช่ตามจังหวัด
- push_dedup_state ถูกเขียนกลับ DB จริงถูกต้องหลัง dispatch แต่ละรอบ
"""
import importlib.util
import os

import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_PATH = os.path.join(_ROOT, "app.py")
_TEST_MARKER = "2100-"
_TEST_PROVINCE = "pytest_fake_province"  # ไม่ใช่จังหวัดจริง กันชนกับข้อมูล/scheduler จริง


def load_app():
    spec = importlib.util.spec_from_file_location("app_under_test", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class FakeSendPush:
    """stub แทน app.send_push — บันทึกทุกครั้งที่ถูกเรียก คืน True เสมอ (จำลองส่งสำเร็จ) ไม่แตะเครือข่ายจริง"""
    def __init__(self):
        self.calls = []

    def __call__(self, endpoint, p256dh, auth, title, body, tag, url="/notifications"):
        self.calls.append({"endpoint": endpoint, "title": title, "body": body, "tag": tag})
        return True


def clear_test_rows(app):
    conn = app.get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM prediction_log WHERE predicted_at_utc LIKE %s", (f"{_TEST_MARKER}%",))
    cur.execute("DELETE FROM notification_log WHERE province = %s", (_TEST_PROVINCE,))
    cur.execute("DELETE FROM push_subscriptions WHERE province = %s", (_TEST_PROVINCE,))
    cur.execute("DELETE FROM users WHERE email LIKE %s", (f"{_TEST_MARKER}%",))
    conn.commit()
    cur.close()
    conn.close()


def insert_snapshot(app, recorded_at_utc, latest_pm25, horizon_rows=None, trend_direction=None, trend_period=None):
    conn = app.get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO notification_log (recorded_at_utc, latest_pm25, province, trend_direction, trend_period) "
        "VALUES (%s, %s, %s, %s, %s)",
        (recorded_at_utc, latest_pm25, _TEST_PROVINCE, trend_direction, trend_period),
    )
    if horizon_rows:
        target = pd.Timestamp(recorded_at_utc) + pd.Timedelta(hours=1)
        cur.executemany(
            """INSERT INTO prediction_log
               (predicted_at_utc, target_time_utc, horizon, predicted_pm25, model_used, source, province)
               VALUES (%s, %s, %s, %s, %s, 'live', %s)""",
            [(recorded_at_utc, target.isoformat(), h, pred, model, _TEST_PROVINCE) for h, pred, model in horizon_rows],
        )
    conn.commit()
    cur.close()
    conn.close()


def insert_subscription(app, endpoint, user_id=None):
    conn = app.get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO push_subscriptions (endpoint, p256dh, auth, user_id, province, created_at, push_dedup_state)
           VALUES (%s, 'fake_p256dh', 'fake_auth', %s, %s, %s, '{}'::jsonb)
           ON CONFLICT (endpoint) DO UPDATE SET push_dedup_state = '{}'::jsonb""",
        (endpoint, user_id, _TEST_PROVINCE, pd.Timestamp.now(tz="UTC").isoformat()),
    )
    conn.commit()
    cur.close()
    conn.close()


def get_dedup_state(app, endpoint):
    conn = app.get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT push_dedup_state FROM push_subscriptions WHERE endpoint = %s", (endpoint,))
    row = cur.fetchone()
    cur.close()
    conn.close()
    return row[0] if row else None


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {label}")
    return condition


def tag_count(fake_send, tag):
    return sum(1 for c in fake_send.calls if c["tag"] == tag)


def test_threshold_dedup(app):
    """เข้าเกณฑ์เกินเพดานต่อเนื่อง 3 ชม. ติดกัน — ต้อง push 'threshold' แค่ครั้งแรก ไม่ยิงซ้ำ
    (เช็คด้วย tag_count ไม่ใช่ len(calls) ดิบ เพราะค่า PM2.5 ที่ขยับข้ามเพดาน 37.5 มักข้าม pm25_status band
    ไปด้วยพร้อมกัน ทำให้ 'transition' อาจ push ร่วมด้วยถูกต้องตามจริง ไม่ใช่บั๊ก)"""
    all_ok = True
    clear_test_rows(app)
    fake_send = FakeSendPush()
    app.send_push = fake_send
    try:
        endpoint = "https://test.invalid/threshold-sub"
        insert_subscription(app, endpoint)

        base = app.effective_threshold_for("ทั่วไป")
        risky_pm25 = base + 10.0

        # ชม.ที่ 1: เพิ่งเริ่มเกินเพดาน — ควร push
        insert_snapshot(app, f"{_TEST_MARKER}03-01T00:00:00+00:00", risky_pm25)
        app.dispatch_push_for_province(_TEST_PROVINCE)
        all_ok &= check("ชม.ที่ 1 (เริ่มเกินเพดาน): push 'threshold' 1 ครั้ง", tag_count(fake_send, "threshold") == 1)
        state = get_dedup_state(app, endpoint)
        all_ok &= check("ชม.ที่ 1: dedup_state['threshold'] == 'active'", state.get("threshold") == "active")

        # ชม.ที่ 2: ยังเกินเพดานต่อเนื่อง (ค่าขยับนิดเดียว ไม่ข้าม pm25_status band) — ต้องไม่ push ซ้ำ
        insert_snapshot(app, f"{_TEST_MARKER}03-01T01:00:00+00:00", risky_pm25 + 0.3)
        app.dispatch_push_for_province(_TEST_PROVINCE)
        all_ok &= check("ชม.ที่ 2 (ยังเกินต่อเนื่อง): ไม่ push 'threshold' ซ้ำ", tag_count(fake_send, "threshold") == 1)

        # ชม.ที่ 3: กลับมาปกติแล้ว — เคลียร์ state ไม่ push 'threshold' เพิ่ม (จะมี 'transition' ร่วมด้วยได้
        # เพราะข้าม pm25_status band จริง — ไม่เช็คตรงนี้ เป็นพฤติกรรมถูกต้องคนละเงื่อนไข)
        insert_snapshot(app, f"{_TEST_MARKER}03-01T02:00:00+00:00", 10.0)
        app.dispatch_push_for_province(_TEST_PROVINCE)
        all_ok &= check("ชม.ที่ 3 (กลับปกติ): ไม่ push 'threshold' เพิ่ม", tag_count(fake_send, "threshold") == 1)
        state = get_dedup_state(app, endpoint)
        all_ok &= check("ชม.ที่ 3: dedup_state['threshold'] กลับเป็น None", state.get("threshold") is None)

        # ชม.ที่ 4: เกินเพดานอีกรอบ (เหตุการณ์ใหม่) — ต้อง push 'threshold' อีกครั้ง
        insert_snapshot(app, f"{_TEST_MARKER}03-01T03:00:00+00:00", risky_pm25)
        app.dispatch_push_for_province(_TEST_PROVINCE)
        all_ok &= check("ชม.ที่ 4 (เกินเพดานรอบใหม่): push 'threshold' อีกครั้ง (รวม 2 ครั้ง)", tag_count(fake_send, "threshold") == 2)
    finally:
        clear_test_rows(app)
    return all_ok


def test_transition_fires_once(app):
    """ระดับเปลี่ยนสถานะ — push แค่ชั่วโมงที่เปลี่ยนจริง แล้วเคลียร์ตัวเองชั่วโมงถัดไป"""
    all_ok = True
    clear_test_rows(app)
    fake_send = FakeSendPush()
    app.send_push = fake_send
    try:
        endpoint = "https://test.invalid/transition-sub"
        insert_subscription(app, endpoint)

        insert_snapshot(app, f"{_TEST_MARKER}04-01T00:00:00+00:00", 10.0)  # ดีมาก
        app.dispatch_push_for_province(_TEST_PROVINCE)
        all_ok &= check("ชม.ที่ 1 (แถวแรก ไม่มีแถวก่อนหน้าเทียบ): ไม่ push", len(fake_send.calls) == 0)

        insert_snapshot(app, f"{_TEST_MARKER}04-01T01:00:00+00:00", 20.0)  # ดี — ระดับเปลี่ยนจริง
        app.dispatch_push_for_province(_TEST_PROVINCE)
        all_ok &= check("ชม.ที่ 2 (ระดับเปลี่ยนจริง): push 1 ครั้ง", len(fake_send.calls) == 1)
        all_ok &= check("ชม.ที่ 2: tag เป็น 'transition'", fake_send.calls[-1]["tag"] == "transition" if fake_send.calls else False)

        insert_snapshot(app, f"{_TEST_MARKER}04-01T02:00:00+00:00", 20.5)  # ระดับเดิม (ดี) ไม่เปลี่ยน
        app.dispatch_push_for_province(_TEST_PROVINCE)
        all_ok &= check("ชม.ที่ 3 (ระดับเดิม): ไม่ push เพิ่ม (ยังคง 1 ครั้ง)", len(fake_send.calls) == 1)
    finally:
        clear_test_rows(app)
    return all_ok


def test_silence_with_now_utc_override(app):
    """สถานีเงียบ — ใช้ now_utc override เลี่ยงปัญหา marker ปี 2100 อยู่ในอนาคตเทียบเวลาจริง"""
    all_ok = True
    clear_test_rows(app)
    fake_send = FakeSendPush()
    app.send_push = fake_send
    try:
        endpoint = "https://test.invalid/silence-sub"
        insert_subscription(app, endpoint)

        recorded_at = f"{_TEST_MARKER}05-01T00:00:00+00:00"
        insert_snapshot(app, recorded_at, 10.0)

        fresh_now = pd.Timestamp(f"{_TEST_MARKER}05-01T01:00:00+00:00")  # ห่าง 1 ชม. — ไม่เงียบ
        app.dispatch_push_for_province(_TEST_PROVINCE, now_utc=fresh_now)
        all_ok &= check("เพิ่งมีข้อมูล 1 ชม.ก่อน: ไม่ push silence", len(fake_send.calls) == 0)

        stale_now = pd.Timestamp(f"{_TEST_MARKER}05-01T09:00:00+00:00")  # ห่าง 9 ชม. > TARGET_STALE_HOURS
        app.dispatch_push_for_province(_TEST_PROVINCE, now_utc=stale_now)
        all_ok &= check("ห่าง 9 ชม. (เกิน TARGET_STALE_HOURS): push silence 1 ครั้ง", len(fake_send.calls) == 1)
        all_ok &= check("tag เป็น 'silence'", fake_send.calls[-1]["tag"] == "silence" if fake_send.calls else False)

        app.dispatch_push_for_province(_TEST_PROVINCE, now_utc=stale_now + pd.Timedelta(hours=1))
        all_ok &= check("เงียบต่อเนื่อง: ไม่ push ซ้ำ", len(fake_send.calls) == 1)
    finally:
        clear_test_rows(app)
    return all_ok


def test_risk_group_per_subscription(app):
    """2 subscription คนละ risk_group ในจังหวัดเดียวกัน — ต้องเห็นเงื่อนไข threshold ต่างกันตาม risk_group
    ของตัวเอง ไม่ใช่ตามจังหวัดร่วม"""
    all_ok = True
    clear_test_rows(app)
    fake_send = FakeSendPush()
    app.send_push = fake_send
    try:
        user_id = app.create_user(
            f"{_TEST_MARKER}vulnerable@test.invalid", "x", "เด็ก-ผู้สูงอายุ", 70, None,
        )
        endpoint_general = "https://test.invalid/general-sub"
        endpoint_vulnerable = "https://test.invalid/vulnerable-sub"
        insert_subscription(app, endpoint_general, user_id=None)  # guest = ทั่วไป
        insert_subscription(app, endpoint_vulnerable, user_id=user_id)

        base_general = app.effective_threshold_for("ทั่วไป")
        base_vulnerable = app.effective_threshold_for("เด็ก-ผู้สูงอายุ")
        # ค่าอยู่ระหว่างสองเกณฑ์ — เกินของกลุ่มเปราะบางเท่านั้น ไม่เกินของทั่วไป
        borderline_pm25 = (base_general + base_vulnerable) / 2
        assert base_vulnerable < borderline_pm25 < base_general, "สมมติฐานค่าชายขอบผิด ปรับ offset ในการทดสอบ"

        insert_snapshot(app, f"{_TEST_MARKER}06-01T00:00:00+00:00", borderline_pm25)
        app.dispatch_push_for_province(_TEST_PROVINCE)

        tags_by_endpoint = {}
        for call in fake_send.calls:
            tags_by_endpoint.setdefault(call["endpoint"], []).append(call["tag"])

        all_ok &= check(
            "guest (ทั่วไป) ไม่ push (ยังไม่เกินเกณฑ์ตัวเอง)",
            "threshold" not in tags_by_endpoint.get(endpoint_general, []),
        )
        all_ok &= check(
            "ผู้ใช้กลุ่มเปราะบาง push threshold (เกินเกณฑ์ตัวเองแล้ว)",
            "threshold" in tags_by_endpoint.get(endpoint_vulnerable, []),
        )
    finally:
        clear_test_rows(app)
    return all_ok


def main():
    app = load_app()
    original_send_push = app.send_push
    try:
        results = [
            test_threshold_dedup(app),
            test_transition_fires_once(app),
            test_silence_with_now_utc_override(app),
            test_risk_group_per_subscription(app),
        ]
    finally:
        app.send_push = original_send_push
        clear_test_rows(app)

    all_ok = all(results)
    print("\nALL TESTS PASSED" if all_ok else "\nSOME TESTS FAILED")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

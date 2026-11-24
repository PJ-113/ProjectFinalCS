"""
จำลองการส่ง Web Push จริง 1 เงื่อนไข (หรือครบทั้ง 5) ไปยัง subscription จริงที่สมัครไว้แล้ว — ไม่แตะ
notification_log/prediction_log เลย (เรียก app.send_push() ตรงๆ ด้วย payload ตัวอย่าง ผ่าน
app._push_payload_for() ตัวเดียวกับที่ dispatch_push_for_province() ใช้จริงตอน production) เหมาะไว้ทดสอบว่า
push ไปถึงอุปกรณ์จริงไหม (โดยเฉพาะ iOS ที่ต้องติดตั้ง PWA ลงหน้าจอโฮมก่อนถึงจะรับ push ได้)

ก่อนรัน: ต้องสมัครรับการแจ้งเตือนจริงก่อน 1 ครั้ง — เปิดเว็บ (บน iOS ต้องเพิ่มไปยังหน้าจอโฮมแล้วเปิดจากไอคอนนั้น
ไม่ใช่เปิดจาก Safari ตรงๆ) ไปหน้า /notifications กดปุ่ม "เปิด" แล้วอนุญาต permission ก่อน

Usage:
    python scripts/simulate_push_notification.py <condition> [province]
    python scripts/simulate_push_notification.py all [province]

    condition: threshold | transition | imminent | trend | silence | all
    province: ไม่ใส่ = ส่งให้ subscription ทุกแถวทุกจังหวัด, ใส่ = กรองเฉพาะจังหวัดนั้น (เช่น khonkaen)
"""
import importlib.util
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_PATH = os.path.join(_ROOT, "app.py")

CONDITIONS = ["threshold", "transition", "imminent", "trend", "silence"]

# ค่าตัวอย่างสมมติต่อเงื่อนไข — ไม่ใช่ข้อมูลจริง แค่ให้ _push_payload_for() ประกอบข้อความได้
_EXAMPLE_TRANSITION = {
    "from_level": "ปานกลาง", "to_level": "เริ่มมีผลกระทบต่อสุขภาพ",
    "from_pm25": 35.0, "to_pm25": 45.0, "direction": "worse", "levels_moved": 1,
}
_EXAMPLE_MIN_RISK_HORIZON = 2
_EXAMPLE_MIN_RISK_HORIZON_PM25 = 45.0
_EXAMPLE_MIN_RISK_HORIZON_DISPLAY_TIME = "15:00"
_EXAMPLE_LATEST_PM25 = 45.0
_EXAMPLE_TREND_PERIOD = "บ่าย"
_EXAMPLE_STALE_HOURS = 8.0
_EXAMPLE_RISK_ALERT = {  # ตัวอย่าง risk_alert_for(): เกินอยู่แล้วตอนนี้ ไปถึง 22:00 น. (CHANGELOG 272)
    "now": True, "start_horizon": None, "start_display_time": None,
    "until_horizon": 7, "until_display_time": "22:00", "through_window_end": False,
}


def load_app():
    spec = importlib.util.spec_from_file_location("app_under_test", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_example_payload(app, condition, risk_group):
    return app._push_payload_for(
        condition, risk_group, _EXAMPLE_TRANSITION,
        _EXAMPLE_MIN_RISK_HORIZON, _EXAMPLE_MIN_RISK_HORIZON_PM25, _EXAMPLE_MIN_RISK_HORIZON_DISPLAY_TIME,
        _EXAMPLE_LATEST_PM25, _EXAMPLE_TREND_PERIOD, _EXAMPLE_STALE_HOURS, _EXAMPLE_RISK_ALERT,
    )


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in CONDITIONS + ["all"]:
        print(__doc__)
        sys.exit(1)
    condition_arg = sys.argv[1]
    province = sys.argv[2] if len(sys.argv) > 2 else None
    conditions_to_send = CONDITIONS if condition_arg == "all" else [condition_arg]

    app = load_app()

    if province is not None and province not in app.PROVINCES_BY_SLUG:
        print(f"ไม่รู้จักจังหวัด '{province}' — ดู province_en ที่ถูกต้องได้จาก data/province_station_registry.json")
        sys.exit(1)

    conn = app.get_db_connection()
    cur = conn.cursor()
    if province:
        cur.execute("SELECT endpoint, p256dh, auth, user_id, province FROM push_subscriptions WHERE province = %s", (province,))
    else:
        cur.execute("SELECT endpoint, p256dh, auth, user_id, province FROM push_subscriptions")
    subscriptions = cur.fetchall()
    cur.close()
    conn.close()

    if not subscriptions:
        where = f" จังหวัด {province}" if province else ""
        print(f"ไม่พบ subscription เลย{where} — ต้องสมัครรับการแจ้งเตือนก่อน:")
        print("1. เปิดเว็บ (iOS: ต้อง 'เพิ่มไปยังหน้าจอโฮม' ก่อน แล้วเปิดจากไอคอนนั้น ไม่ใช่เปิดจาก Safari ตรงๆ)")
        print("2. ไปหน้า /notifications")
        print("3. กดปุ่ม 'เปิด' แล้วอนุญาต permission เมื่อเบราว์เซอร์ถาม")
        sys.exit(1)

    print(f"พบ {len(subscriptions)} subscription — จะส่ง {len(conditions_to_send)} เงื่อนไข: {conditions_to_send}\n")

    for endpoint, p256dh, auth, user_id, sub_province in subscriptions:
        risk_group = "ทั่วไป"
        if user_id is not None:
            user = app.get_user_by_id(user_id)
            if user is not None:
                risk_group = user["risk_group"]

        print(f"--- {endpoint[:60]}... (จังหวัด {sub_province}, risk_group={risk_group}) ---")
        for condition in conditions_to_send:
            title, body = build_example_payload(app, condition, risk_group)
            ok = app.send_push(endpoint, p256dh, auth, title, body, condition)
            status = "ส่งสำเร็จ" if ok else "ส่งไม่สำเร็จ (ดู log ด้านบนถ้ามี — subscription อาจหมดอายุแล้วถูกลบไปแล้ว)"
            print(f"  [{condition}] {status}")
            print(f"    title: {title}")
            print(f"    body:  {body}")

    print("\nเสร็จแล้ว — ถ้าโทรศัพท์ไม่ขึ้นแจ้งเตือนเลย เช็คตามนี้: "
          "(1) iOS ต้องเปิดจากไอคอนที่เพิ่มไว้หน้าจอโฮมเท่านั้น ไม่ใช่ Safari ตรงๆ "
          "(2) เช็คว่า permission ไม่ได้ถูกปิดไว้ใน Settings ของเครื่อง "
          "(3) เช็คว่า 'ส่งสำเร็จ' จริง ไม่ใช่ 'ส่งไม่สำเร็จ'")


if __name__ == "__main__":
    main()

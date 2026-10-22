"""
ทดสอบ auth routes (register/login/me/profile/change-password/logout) ใน app.py
เขียนแถวทดสอบลง users จริงใน Postgres ด้วยอีเมลลงท้าย "@test-marker.invalid" แล้วลบทิ้งก่อน/หลังรัน
"""
import importlib.util
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_PATH = os.path.join(_ROOT, r"app.py")
_TEST_DOMAIN = "@test-marker.invalid"


def load_app():
    spec = importlib.util.spec_from_file_location("app_under_test", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clear_test_rows(app):
    conn = app.get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM users WHERE email LIKE %s", (f"%{_TEST_DOMAIN}",))
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
    email = f"auth-test-{int(time.time())}{_TEST_DOMAIN}"
    all_ok = True

    clear_test_rows(app)
    try:
        client = app.app.test_client()

        # 1. register
        r = client.post("/api/auth/register", json={
            "email": email, "password": "correcthorse1", "risk_group": "เด็ก-ผู้สูงอายุ", "age": 30,
        })
        all_ok &= check("register -> 201", r.status_code == 201)
        body = r.get_json()
        all_ok &= check("register response has no password_hash", "password_hash" not in body)
        all_ok &= check("register response email matches", body.get("email") == email)

        # 2. duplicate register -> 409
        r = client.post("/api/auth/register", json={"email": email, "password": "correcthorse1"})
        all_ok &= check("duplicate register -> 409", r.status_code == 409)

        # 3. wrong password login -> 401
        client.post("/api/auth/logout")
        r = client.post("/api/auth/login", json={"email": email, "password": "wrongpassword"})
        all_ok &= check("wrong password login -> 401", r.status_code == 401)

        # 4. correct login -> 200
        r = client.post("/api/auth/login", json={"email": email, "password": "correcthorse1"})
        all_ok &= check("correct login -> 200", r.status_code == 200)

        # 5. GET /me matches
        r = client.get("/api/auth/me")
        me = r.get_json()
        all_ok &= check("GET /me -> 200", r.status_code == 200)
        all_ok &= check("GET /me risk_group matches", me.get("risk_group") == "เด็ก-ผู้สูงอายุ")

        # 6. PUT /profile updates risk_group/age
        r = client.put("/api/auth/profile", json={"risk_group": "มีโรคประจำตัว", "age": 45})
        all_ok &= check("PUT /profile -> 200", r.status_code == 200)
        r = client.get("/api/auth/me")
        me = r.get_json()
        all_ok &= check("profile update persisted (risk_group)", me.get("risk_group") == "มีโรคประจำตัว")
        all_ok &= check("profile update persisted (age)", me.get("age") == 45)

        # 7. change-password wrong current -> 400
        r = client.post("/api/auth/change-password", json={
            "current_password": "wrongone", "new_password": "newpassword2",
        })
        all_ok &= check("change-password wrong current -> 400", r.status_code == 400)

        # 8. change-password correct -> 200
        r = client.post("/api/auth/change-password", json={
            "current_password": "correcthorse1", "new_password": "newpassword2",
        })
        all_ok &= check("change-password correct -> 200", r.status_code == 200)

        # 9. logout then login with NEW password proves hash rotated
        client.post("/api/auth/logout")
        r = client.post("/api/auth/login", json={"email": email, "password": "newpassword2"})
        all_ok &= check("login with new password -> 200 (hash rotated)", r.status_code == 200)

        # 10. logout then GET /me -> 401
        client.post("/api/auth/logout")
        r = client.get("/api/auth/me")
        all_ok &= check("GET /me after logout -> 401", r.status_code == 401)

        # 11. fresh client (no cookie) -> 401
        fresh_client = app.app.test_client()
        r = fresh_client.get("/api/auth/me")
        all_ok &= check("GET /me with no session -> 401", r.status_code == 401)
    finally:
        clear_test_rows(app)

    print("\n" + ("ALL TESTS PASSED" if all_ok else "SOME TESTS FAILED"))
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()

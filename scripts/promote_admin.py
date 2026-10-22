"""
ตั้งค่า role='admin' ให้ user ที่มีอยู่แล้ว ผ่าน UPDATE ตรงๆ ใน DB

Usage: python scripts/promote_admin.py user@example.com
"""
import os
import sys

import psycopg2

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load_env():
    env = {}
    with open(os.path.join(_ROOT, ".env"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/promote_admin.py <email>")
        sys.exit(1)
    email = sys.argv[1].strip().lower()

    env = load_env()
    conn = psycopg2.connect(host=env["POSTGRES_HOST"], port=env["POSTGRES_PORT"], dbname=env["POSTGRES_DB"],
                             user=env["POSTGRES_USER"], password=env["POSTGRES_PASSWORD"])
    cur = conn.cursor()
    cur.execute("SELECT id, role FROM users WHERE email = %s", (email,))
    row = cur.fetchone()
    if row is None:
        print(f"ไม่พบผู้ใช้ที่อีเมล {email} — ต้องสมัครสมาชิกผ่านหน้าเว็บก่อน แล้วค่อยรันสคริปต์นี้")
        cur.close()
        conn.close()
        sys.exit(1)

    user_id, current_role = row
    if current_role == "admin":
        print(f"{email} เป็น admin อยู่แล้ว (user id {user_id}) ไม่ต้องทำอะไรเพิ่ม")
        cur.close()
        conn.close()
        return

    cur.execute("UPDATE users SET role = 'admin' WHERE id = %s", (user_id,))
    conn.commit()
    cur.close()
    conn.close()
    print(f"ตั้ง {email} (user id {user_id}) เป็น admin แล้ว — role เช็คสดจาก DB ทุกครั้งที่ /api/auth/me "
          "ถูกเรียก (ไม่แคชไว้ใน session cookie) มีผลทันทีที่ backend แต่ฝั่ง frontend (authState.user) "
          "ยัง cache ค่าเก่าอยู่จนกว่าจะโหลดหน้าเว็บใหม่ — ให้ผู้ใช้รีเฟรชหน้า (ไม่ต้อง logout ก็ได้)")


if __name__ == "__main__":
    main()

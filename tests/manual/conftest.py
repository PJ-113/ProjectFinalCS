# tests/manual/ เป็นสคริปต์ทดสอบแบบรันตรง (python tests/manual/test_xxx.py) ที่แตะ DB/เครือข่ายจริง
# ไม่ใช่ pytest — กันไม่ให้ `pytest tests/` เก็บไฟล์ในโฟลเดอร์นี้ไปรัน (CI รันแค่ tests/unit/)
collect_ignore_glob = ["*.py"]

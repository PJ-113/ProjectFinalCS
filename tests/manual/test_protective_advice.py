"""ทดสอบตารางกฎการ์ด "คำแนะนำวันนี้" (PROTECTIVE_ADVICE_RULES / protective_advice_config ใน app.py, CHANGELOG 270)
ไม่แตะ DB ไม่เรียก API ภายนอก — ตรวจโครงสร้างตารางล้วนๆ
รัน: python tests/manual/test_protective_advice.py
"""
import os
import sys
import importlib.util

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
APP_PATH = os.path.join(_ROOT, "app.py")

results = []


def check(name, cond):
    results.append(cond)
    print(f"[{'PASS' if cond else 'FAIL'}] {name}")


def load_app():
    os.chdir(_ROOT)  # app.py อ่าน .env / models ด้วย path สัมพัทธ์
    spec = importlib.util.spec_from_file_location("app_under_test", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    app = load_app()
    rules = app.PROTECTIVE_ADVICE_RULES
    items = set(app.PROTECTIVE_ADVICE_ITEMS)
    levels = [name for _, name, _ in app.PM25_LEVELS]

    check("กลุ่มเสี่ยงในตาราง = VALID_RISK_GROUPS พอดี", set(rules) == set(app.VALID_RISK_GROUPS))
    for group in app.VALID_RISK_GROUPS:
        g = rules.get(group, {})
        check(f"{group}: มีครบทุกระดับ ({len(levels)} ระดับ) ไม่มีระดับแปลกปลอม", set(g) == set(levels))
        for level in levels:
            ids = g.get(level, [])
            check(f"{group}/{level}: id ทุกตัวอยู่ใน PROTECTIVE_ADVICE_ITEMS", set(ids) <= items)
            check(f"{group}/{level}: ไม่มี id ซ้ำ และไม่เกิน 4 ข้อ", len(ids) == len(set(ids)) and len(ids) <= 4)
            check(f"{group}/{level}: ไม่ใช้ทั้ง mask และ n95 พร้อมกัน", not ({"mask", "n95"} <= set(ids)))
        check(f"{group}: ระดับดีมาก = ปกติ (ไม่มีข้อควรระวัง)", g.get("ดีมาก") == [])
        # ระดับฝุ่นสูงขึ้น ต้องไม่ได้คำแนะนำน้อยลง
        counts = [len(g.get(level, [])) for level in levels]
        check(f"{group}: จำนวนข้อไม่ลดลงเมื่อฝุ่นแย่ลง {counts}", all(a <= b for a, b in zip(counts, counts[1:])))

    check("ไม่มี id เกี่ยวกับออกแรง/เวลา (อยู่การ์ดกิจกรรมที่เดียว)",
          not any(word in i for i in items for word in ("exert", "activity", "exercise", "time")))

    cfg = app.protective_advice_config("มีโรคประจำตัว")
    check("protective_advice_config คืนตารางของกลุ่มนั้น", cfg == rules["มีโรคประจำตัว"])
    cfg["ดี"].append("mask")
    check("แก้ค่าที่คืนไปแล้วไม่กระทบตารางต้นฉบับ (คืนสำเนา)", rules["มีโรคประจำตัว"]["ดี"] == ["medicine"])
    check("กลุ่มที่ไม่รู้จัก -> ใช้กลุ่มทั่วไป", app.protective_advice_config("ไม่มีกลุ่มนี้") == rules["ทั่วไป"])

    print()
    print("ALL TESTS PASSED" if all(results) else f"SOME TESTS FAILED ({results.count(False)}/{len(results)})")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()

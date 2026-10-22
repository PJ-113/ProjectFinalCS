"""
ทดสอบตรรกะของ evaluate_live_v2_vs_persistence.evaluate() ด้วยข้อมูลสังเคราะห์ (ไม่ต่อ DB)
ที่คำนวณค่า MAE/coverage/verdict ที่คาดหวังไว้ด้วยมือ
"""
import importlib.util
import os

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load():
    spec = importlib.util.spec_from_file_location("evalmod", os.path.join(_ROOT, "scripts", "evaluate_live_v2_vs_persistence.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def make_rows(n, horizon, v2_err, ps_err, model="lightgbm_pooled_v2", province="khonkaen", inside=True, actual=20.0):
    """n แถว: actual คงที่, v2 ทำนายพลาด v2_err, persistence พลาด ps_err, ช่วงกว้าง 10 (ครอบ actual ถ้า inside)"""
    pred = actual + v2_err
    lo, hi = (actual - 5, actual + 5) if inside else (actual + 6, actual + 16)
    return pd.DataFrame({"province": province, "horizon": horizon, "model_used": model, "predicted_at_utc": "2026-09-19T07:00:00+00:00",
                         "target_time_utc": "2026-09-20T07:00:00+00:00", "predicted_pm25": pred, "actual_pm25": actual,
                         "pm25_lo": lo, "pm25_hi": hi, "latest_pm25": actual + ps_err}, index=range(n))


def main():
    m = load()

    # 1) v2 ผิดพลาด 1, persistence ผิดพลาด 2 -> ดีกว่า 50% ที่ horizon ไกล -> PASS; coverage 100%, width 10
    df = pd.concat([make_rows(200, 12, 1.0, 2.0), make_rows(200, 24, 1.0, 2.0, model="xgboost_pooled_v2")], ignore_index=True)
    r = m.evaluate(df)
    s = r["far_summary"]
    assert s["n"] == 400 and abs(s["mae_v2"] - 1.0) < 1e-9 and abs(s["mae_persistence"] - 2.0) < 1e-9
    assert abs(s["improve_pct"] - 50.0) < 1e-9 and s["coverage_pct"] == 100.0 and abs(s["width"] - 10.0) < 1e-9
    assert abs(s["bias_v2"] - 1.0) < 1e-9, "bias = pred - actual"
    assert r["verdict"] == "PASS", r["verdict"]
    assert set(r["by_model_far"]["model"]) == {"ทุกโมเดล v2", "lightgbm_pooled_v2", "xgboost_pooled_v2"}
    print("ผ่าน 1: PASS, ค่า MAE/improve/coverage/width/bias ตรงที่คำนวณด้วยมือ")

    # 2) ดีขึ้นเพียง 5% -> WARN
    r = m.evaluate(make_rows(400, 18, 1.9, 2.0))
    assert r["verdict"] == "WARN" and abs(r["far_summary"]["improve_pct"] - 5.0) < 1e-9
    print("ผ่าน 2: WARN เมื่อดีกว่า persistence < 10%")

    # 3) v2 แย่กว่า -> FAIL
    r = m.evaluate(make_rows(400, 24, 3.0, 2.0))
    assert r["verdict"] == "FAIL" and r["far_summary"]["improve_pct"] < 0
    print("ผ่าน 3: FAIL เมื่อ v2 แย่กว่า persistence")

    # 4) ข้อมูลไม่พอ (< MIN_PAIRS_FAR) -> ไม่ตัดสิน แม้ผลจะดี
    r = m.evaluate(make_rows(m.MIN_PAIRS_FAR - 1, 12, 0.1, 5.0))
    assert r["verdict"] == "ข้อมูลยังไม่พอ"
    print("ผ่าน 4: ข้อมูลไม่พอ -> ไม่ตัดสิน")

    # 5) horizon ใกล้ (t+1) ไม่นับในเกณฑ์ตัดสิน: t+1 ดีมากแต่ far แย่ -> FAIL
    df = pd.concat([make_rows(500, 1, 0.0, 9.0), make_rows(400, 12, 3.0, 2.0)], ignore_index=True)
    r = m.evaluate(df)
    assert r["verdict"] == "FAIL", "เกณฑ์ต้องดูเฉพาะ horizon 12-24"
    b = r["by_bucket"].set_index("horizon")
    assert b.loc["t+1", "n"] == 500 and b.loc["t+1", "improve_pct"] == 100.0
    print("ผ่าน 5: เกณฑ์ใช้เฉพาะ horizon ไกล; ตาราง bucket แยกถูก")

    # 6) แถวที่ยังไม่มีค่าจริง / ไม่มี persistence ต้องไม่ปนใน matched แต่นับแยก
    df = pd.concat([make_rows(350, 12, 1.0, 2.0),
                    make_rows(10, 12, 1.0, 2.0).assign(actual_pm25=np.nan),
                    make_rows(7, 12, 1.0, 2.0).assign(latest_pm25=np.nan)], ignore_index=True)
    r = m.evaluate(df)
    assert len(r["matched"]) == 350 and r["pending"] == 10 and r["no_persistence"] == 7
    print("ผ่าน 6: pending/ไม่มี persistence ถูกแยกออกจากคู่ที่ใช้เทียบ")

    # 7) coverage: ช่วงไม่ครอบ actual -> 0%; แถวที่ไม่มีช่วง (NULL) ไม่ถูกนับใน coverage
    df = pd.concat([make_rows(300, 12, 1.0, 2.0, inside=False),
                    make_rows(100, 12, 1.0, 2.0).assign(pm25_lo=np.nan, pm25_hi=np.nan)], ignore_index=True)
    s = m.evaluate(df)["far_summary"]
    assert s["coverage_pct"] == 0.0 and s["n"] == 400
    print("ผ่าน 7: coverage นับเฉพาะแถวที่มีช่วง")

    # 8) ไม่มีข้อมูลเลย -> ไม่ crash
    empty = make_rows(1, 12, 1.0, 2.0).iloc[0:0]
    r = m.evaluate(empty)
    assert r["verdict"] == "ข้อมูลยังไม่พอ" and len(r["matched"]) == 0
    print("ผ่าน 8: ไม่มีข้อมูล -> ไม่ crash")

    print("\nทุกเคสผ่าน (test_evaluate_live_v2_vs_persistence)")


if __name__ == "__main__":
    main()

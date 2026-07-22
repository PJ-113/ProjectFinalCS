"""
ทดสอบ Circuit Breaker + Dynamic Routing (select_model, compute_missing_ratio, compute_rolling_r2)

Test 1-2 เขียนแถวทดสอบลง prediction_log จริงใน Postgres แต่ใช้ predicted_at_utc ปี 2100
(ไม่มีทางชนกับข้อมูลจริง) แล้วลบทิ้งเองทั้งก่อน/หลังรัน (idempotent, safe to re-run) แทนการ
สร้าง temp sqlite db แยกแบบเดิม เพราะ app.py ย้ายไป Postgres แล้วไม่มี DB_PATH ให้ monkeypatch
เป็นไฟล์แยกอีกต่อไป (ดู CHANGELOG: migrate predictions.db -> PostgreSQL)
"""
import importlib.util

import numpy as np
import pandas as pd

import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_PATH = os.path.join(_ROOT, r"app.py")
_TEST_MARKER = "2100-"  # predicted_at_utc ของแถวทดสอบทั้งหมดขึ้นต้นด้วยปีนี้เสมอ


def load_app():
    spec = importlib.util.spec_from_file_location("app_under_test", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_feature_row(app, missing_count=0):
    row = {c: 1.0 for c in app.FEATURE_COLS}
    for c in app.FEATURE_COLS[:missing_count]:
        row[c] = np.nan
    return pd.DataFrame([row])


def clear_test_rows(app):
    conn = app.get_db_connection()
    cur = conn.cursor()
    cur.execute("DELETE FROM prediction_log WHERE predicted_at_utc LIKE %s", (f"{_TEST_MARKER}%",))
    conn.commit()
    cur.close()
    conn.close()


def insert_rows(app, model_name, horizon, actual_values, predicted_values, start_hour=0):
    conn = app.get_db_connection()
    cur = conn.cursor()
    base = pd.Timestamp(f"{_TEST_MARKER}01-01T00:00:00+00:00")
    rows = []
    for i, (actual, pred) in enumerate(zip(actual_values, predicted_values)):
        t = base + pd.Timedelta(hours=start_hour + i)
        # เขียน actual ตรงๆ ลง actual_pm25_calibrated เลย (ไม่ผ่าน calibrate_pm25) เพราะ test นี้
        # ทดสอบ logic เปรียบเทียบ R2 ของ compute_rolling_r2() ตรงๆ ไม่ได้ทดสอบ calibration pipeline
        rows.append((t.isoformat(), t.isoformat(), horizon, float(pred), float(actual), float(actual), model_name))
    cur.executemany(
        """INSERT INTO prediction_log
           (predicted_at_utc, target_time_utc, horizon, predicted_pm25, actual_pm25, actual_pm25_calibrated, model_used)
           VALUES (%s, %s, %s, %s, %s, %s, %s)""",
        rows,
    )
    conn.commit()
    cur.close()
    conn.close()


def test_1_circuit_breaker(app):
    print("=" * 90)
    print("TEST 1: Circuit Breaker (missing_ratio > SIGMA_CRITICAL -> prophet)")
    print("=" * 90)

    n_features = len(app.FEATURE_COLS)
    n_missing = int(n_features * 0.35)  # > 30%
    row = make_feature_row(app, missing_count=n_missing)
    ratio = app.compute_missing_ratio(row)
    print(f"n_features={n_features}, n_missing={n_missing}, missing_ratio={ratio:.3f} (SIGMA_CRITICAL={app.SIGMA_CRITICAL})")

    result = app.select_model(row, horizon=1)
    print(f"select_model() -> {result}")

    passed = ratio > app.SIGMA_CRITICAL and result == "prophet"
    print("RESULT:", "PASS" if passed else "FAIL")
    return passed


def test_2_rolling_r2(app):
    print("\n" + "=" * 90)
    print("TEST 2: Rolling R2 -> select_model() picks the better model, both directions")
    print("=" * 90)

    clear_test_rows(app)
    normal_row = make_feature_row(app, missing_count=0)

    # กรณี A: xgboost ดีกว่า (error น้อย) ที่ horizon=5
    actual = [20 + 8 * np.sin(i / 4) for i in range(30)]
    xgb_pred_good = [a + np.random.uniform(-0.3, 0.3) for a in actual]
    lgb_pred_bad = [a + 8 + np.random.uniform(-1, 1) for a in actual]
    insert_rows(app, "xgboost", horizon=5, actual_values=actual, predicted_values=xgb_pred_good, start_hour=0)
    insert_rows(app, "lightgbm", horizon=5, actual_values=actual, predicted_values=lgb_pred_bad, start_hour=1000)

    r2_xgb_a = app.compute_rolling_r2("xgboost", 5)
    r2_lgb_a = app.compute_rolling_r2("lightgbm", 5)
    result_a = app.select_model(normal_row, horizon=5)
    print(f"[case A] r2_xgb={r2_xgb_a:.3f} r2_lgb={r2_lgb_a:.3f} -> select_model()={result_a} (expected: xgboost)")

    # กรณี B: สลับกัน lightgbm ดีกว่า ที่ horizon=6
    lgb_pred_good = [a + np.random.uniform(-0.3, 0.3) for a in actual]
    xgb_pred_bad = [a + 8 + np.random.uniform(-1, 1) for a in actual]
    insert_rows(app, "lightgbm", horizon=6, actual_values=actual, predicted_values=lgb_pred_good, start_hour=0)
    insert_rows(app, "xgboost", horizon=6, actual_values=actual, predicted_values=xgb_pred_bad, start_hour=1000)

    r2_xgb_b = app.compute_rolling_r2("xgboost", 6)
    r2_lgb_b = app.compute_rolling_r2("lightgbm", 6)
    result_b = app.select_model(normal_row, horizon=6)
    print(f"[case B] r2_xgb={r2_xgb_b:.3f} r2_lgb={r2_lgb_b:.3f} -> select_model()={result_b} (expected: lightgbm)")

    clear_test_rows(app)

    passed = (result_a == "xgboost") and (result_b == "lightgbm")
    print("RESULT:", "PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    app = load_app()
    try:
        r1 = test_1_circuit_breaker(app)
        r2 = test_2_rolling_r2(app)
    finally:
        clear_test_rows(app)  # กันแถวทดสอบตกค้างแม้ assertion จะพังกลางคัน

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"Test 1 (circuit breaker):  {'PASS' if r1 else 'FAIL'}")
    print(f"Test 2 (rolling R2 route): {'PASS' if r2 else 'FAIL'}")

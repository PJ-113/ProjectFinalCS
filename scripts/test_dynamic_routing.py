"""
ทดสอบ Circuit Breaker + Dynamic Routing (select_model, compute_missing_ratio, compute_rolling_r2)
ต่อจังหวัด (province-scoped pooled model) — ครอบคลุม naive_fallback, circuit breaker,
rolling R2 routing, cold-start exploration (ทั้งสองโมเดล และฝั่งเดียว), force_naive

เขียนแถวทดสอบลง prediction_log จริงด้วย predicted_at_utc ปี 2100 แล้วลบทิ้งก่อน/หลังรัน
"""
import importlib.util

import numpy as np
import pandas as pd

import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_PATH = os.path.join(_ROOT, r"app.py")
_TEST_MARKER = "2100-"  # predicted_at_utc ของแถวทดสอบทั้งหมดขึ้นต้นด้วยปีนี้เสมอ
_TEST_PROVINCE = "khonkaen"


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
    cur.execute(
        "DELETE FROM prediction_log WHERE predicted_at_utc LIKE %s AND province = %s",
        (f"{_TEST_MARKER}%", _TEST_PROVINCE),
    )
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
        # actual_pm25_calibrated เก็บค่าเดียวกับ actual_pm25 (ไม่มี calibration)
        rows.append((t.isoformat(), t.isoformat(), horizon, float(pred), float(actual), float(actual),
                     model_name, _TEST_PROVINCE))
    cur.executemany(
        """INSERT INTO prediction_log
           (predicted_at_utc, target_time_utc, horizon, predicted_pm25, actual_pm25, actual_pm25_calibrated,
            model_used, province)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s)""",
        rows,
    )
    conn.commit()
    cur.close()
    conn.close()


def test_1_circuit_breaker(app):
    print("=" * 90)
    print("TEST 1: Circuit Breaker (missing_ratio > SIGMA_CRITICAL -> naive_fallback)")
    print("=" * 90)

    n_features = len(app.FEATURE_COLS)
    n_missing = int(n_features * 0.35)  # > 30%
    row = make_feature_row(app, missing_count=n_missing)
    ratio = app.compute_missing_ratio(row)
    print(f"n_features={n_features}, n_missing={n_missing}, missing_ratio={ratio:.3f} (SIGMA_CRITICAL={app.SIGMA_CRITICAL})")

    result = app.select_model(row, horizon=1, province=_TEST_PROVINCE, r2_lookup=app.fetch_rolling_r2_lookup(_TEST_PROVINCE))
    print(f"select_model() -> {result}")

    passed = ratio > app.SIGMA_CRITICAL and result == app.MODEL_NAIVE
    print("RESULT:", "PASS" if passed else "FAIL")
    return passed


def test_2_rolling_r2(app):
    print("\n" + "=" * 90)
    print("TEST 2: Rolling R2 -> select_model() picks the better model, both directions")
    print("=" * 90)

    clear_test_rows(app)
    normal_row = make_feature_row(app, missing_count=0)

    # กรณี A: xgboost_pooled ดีกว่า (error น้อย) ที่ horizon=5
    actual = [20 + 8 * np.sin(i / 4) for i in range(30)]
    xgb_pred_good = [a + np.random.uniform(-0.3, 0.3) for a in actual]
    lgb_pred_bad = [a + 8 + np.random.uniform(-1, 1) for a in actual]
    insert_rows(app, app.MODEL_XGB, horizon=5, actual_values=actual, predicted_values=xgb_pred_good, start_hour=0)
    insert_rows(app, app.MODEL_LGB, horizon=5, actual_values=actual, predicted_values=lgb_pred_bad, start_hour=1000)

    r2_xgb_a = app.compute_rolling_r2(app.MODEL_XGB, 5, _TEST_PROVINCE)
    r2_lgb_a = app.compute_rolling_r2(app.MODEL_LGB, 5, _TEST_PROVINCE)
    r2_lookup = app.fetch_rolling_r2_lookup(_TEST_PROVINCE)
    result_a = app.select_model(normal_row, horizon=5, province=_TEST_PROVINCE, r2_lookup=r2_lookup)
    print(f"[case A] r2_xgb={r2_xgb_a:.3f} r2_lgb={r2_lgb_a:.3f} -> select_model()={result_a} (expected: xgboost_pooled)")

    # กรณี B: สลับกัน lightgbm_pooled ดีกว่า ที่ horizon=6
    lgb_pred_good = [a + np.random.uniform(-0.3, 0.3) for a in actual]
    xgb_pred_bad = [a + 8 + np.random.uniform(-1, 1) for a in actual]
    insert_rows(app, app.MODEL_LGB, horizon=6, actual_values=actual, predicted_values=lgb_pred_good, start_hour=0)
    insert_rows(app, app.MODEL_XGB, horizon=6, actual_values=actual, predicted_values=xgb_pred_bad, start_hour=1000)

    r2_xgb_b = app.compute_rolling_r2(app.MODEL_XGB, 6, _TEST_PROVINCE)
    r2_lgb_b = app.compute_rolling_r2(app.MODEL_LGB, 6, _TEST_PROVINCE)
    result_b = app.select_model(normal_row, horizon=6, province=_TEST_PROVINCE, r2_lookup=r2_lookup)
    print(f"[case B] r2_xgb={r2_xgb_b:.3f} r2_lgb={r2_lgb_b:.3f} -> select_model()={result_b} (expected: lightgbm_pooled)")

    clear_test_rows(app)

    passed = (result_a == app.MODEL_XGB) and (result_b == app.MODEL_LGB)
    print("RESULT:", "PASS" if passed else "FAIL")
    return passed


def test_3_cold_start_both_none(app):
    print("\n" + "=" * 90)
    print("TEST 3: Cold start - both models have no data -> random 50/50 (not always the same model)")
    print("=" * 90)

    clear_test_rows(app)
    normal_row = make_feature_row(app, missing_count=0)

    horizon = 9  # horizon ที่ไม่มีข้อมูลทดสอบใดๆ เลย
    r2_lookup = app.fetch_rolling_r2_lookup(_TEST_PROVINCE)
    results = {app.select_model(normal_row, horizon=horizon, province=_TEST_PROVINCE, r2_lookup=r2_lookup) for _ in range(60)}
    print(f"outcomes seen across 60 draws: {results}")

    passed = results == {app.MODEL_XGB, app.MODEL_LGB}
    print("RESULT:", "PASS" if passed else "FAIL")
    return passed


def test_4_single_none_explores(app):
    print("\n" + "=" * 90)
    print("TEST 4: Single-model cold start -> explores the missing one sometimes (both directions)")
    print("=" * 90)
    print("(regression guard for the bug fixed in CHANGELOG (20): checking only one side of")
    print(" None used to starve the other model of live data forever)")

    clear_test_rows(app)
    normal_row = make_feature_row(app, missing_count=0)

    # ฝั่ง A: มีแค่ xgboost_pooled ข้อมูลพอ (lightgbm_pooled ไม่มีข้อมูลเลย) -> ต้อง explore ไป
    # lightgbm_pooled บ้าง (ไม่ fallback ไป xgboost_pooled 100%)
    actual = [20 + 5 * np.sin(i / 3) for i in range(30)]
    xgb_pred = [a + np.random.uniform(-0.3, 0.3) for a in actual]
    insert_rows(app, app.MODEL_XGB, horizon=11, actual_values=actual, predicted_values=xgb_pred, start_hour=0)
    r2_lookup_a = app.fetch_rolling_r2_lookup(_TEST_PROVINCE)
    results_a = {app.select_model(normal_row, horizon=11, province=_TEST_PROVINCE, r2_lookup=r2_lookup_a) for _ in range(80)}
    print(f"[A: only xgboost_pooled has data] outcomes across 80 draws: {results_a}")

    clear_test_rows(app)

    # ฝั่ง B: สลับกัน มีแค่ lightgbm_pooled ข้อมูลพอ -> ต้อง explore ไป xgboost_pooled บ้าง
    lgb_pred = [a + np.random.uniform(-0.3, 0.3) for a in actual]
    insert_rows(app, app.MODEL_LGB, horizon=12, actual_values=actual, predicted_values=lgb_pred, start_hour=0)
    r2_lookup_b = app.fetch_rolling_r2_lookup(_TEST_PROVINCE)
    results_b = {app.select_model(normal_row, horizon=12, province=_TEST_PROVINCE, r2_lookup=r2_lookup_b) for _ in range(80)}
    print(f"[B: only lightgbm_pooled has data] outcomes across 80 draws: {results_b}")

    clear_test_rows(app)

    passed = (results_a == {app.MODEL_XGB, app.MODEL_LGB}) and (results_b == {app.MODEL_XGB, app.MODEL_LGB})
    print("RESULT:", "PASS" if passed else "FAIL")
    return passed


def test_5_force_naive(app):
    """force_naive ต้องได้ naive_fallback ทุก horizon แม้ feature อื่นครบและ rolling R2 ดี
    และ NaN เพียง 2 จาก 21 feature ไม่ trigger circuit breaker เอง"""
    print("\n" + "=" * 90)
    print("TEST 5: force_naive -> naive_fallback ทุก horizon (ไม่อ่าน rolling R2)")
    print("=" * 90)
    normal_row = make_feature_row(app, missing_count=0)
    lookup = {(app.MODEL_LGB, h): 0.95 for h in range(1, 25)}
    lookup.update({(app.MODEL_XGB, h): 0.90 for h in range(1, 25)})
    all_naive = all(app.select_model(normal_row, h, _TEST_PROVINCE, lookup, force_naive=True) == app.MODEL_NAIVE
                    for h in range(1, 25))
    not_forced = all(app.select_model(normal_row, h, _TEST_PROVINCE, lookup) == app.MODEL_LGB for h in range(1, 25))
    two_missing = normal_row.copy()
    nb_cols = [c for c in ("pm25_neighbor_avg", "nb_trend_12h") if c in app.FEATURE_COLS]
    two_missing[nb_cols] = np.nan
    ratio = app.compute_missing_ratio(two_missing)
    print(f"force_naive ทุก horizon -> naive: {all_naive}; ไม่ force -> เลือกตาม R2: {not_forced}; missing_ratio(2 NaN)={ratio:.3f}")
    passed = all_naive and not_forced and ratio < app.SIGMA_CRITICAL
    print("RESULT:", "PASS" if passed else "FAIL")
    return passed


if __name__ == "__main__":
    app = load_app()
    try:
        r1 = test_1_circuit_breaker(app)
        r2 = test_2_rolling_r2(app)
        r3 = test_3_cold_start_both_none(app)
        r4 = test_4_single_none_explores(app)
        r5 = test_5_force_naive(app)
    finally:
        clear_test_rows(app)  # กันแถวทดสอบตกค้างแม้ assertion จะพังกลางคัน

    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)
    print(f"Test 1 (circuit breaker):        {'PASS' if r1 else 'FAIL'}")
    print(f"Test 2 (rolling R2 route):       {'PASS' if r2 else 'FAIL'}")
    print(f"Test 3 (cold start, both None):  {'PASS' if r3 else 'FAIL'}")
    print(f"Test 4 (cold start, single None):{'PASS' if r4 else 'FAIL'}")
    print(f"Test 5 (force_naive):            {'PASS' if r5 else 'FAIL'}")

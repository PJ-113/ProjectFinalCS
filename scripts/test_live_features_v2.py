"""
ทดสอบ feature live ของโมเดล v2 (CHANGELOG 141): compute_neighbor_features (นโยบายเพื่อนบ้าน), count_fire_hotspots,
get_fire_hotspots (ไม่ล่ม/ไม่รั่วคีย์), get_recent_series (cache), select_model(force_naive), build_feature_row (21 feature)

ใช้ข้อมูลสังเคราะห์ + ฉีดฟังก์ชัน fetch เอง: ไม่เรียก API ภายนอก และไม่เขียนแถวลง prediction_log/notification_log
(โหลด app.py ผ่าน importlib จะรัน init_db() ที่เป็น CREATE TABLE IF NOT EXISTS เหมือนทุก test เดิมในโปรเจกต์)
รัน: python scripts/test_live_features_v2.py
"""
import contextlib
import importlib.util
import io
import os

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_PATH = os.path.join(_ROOT, "app.py")
T = pd.Timestamp("2026-06-20 07:00:00", tz="UTC")


def load_app():
    spec = importlib.util.spec_from_file_location("app_under_test", APP_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_df(end, hours=72, base=10.0, slope=0.5):
    """pm25 รายชั่วโมงถึง `end` : ค่า = base + slope * (ชั่วโมงนับจากจุดเริ่ม) -> ค่า ณ เวลาใดๆ คำนวณมือได้"""
    times = pd.date_range(end - pd.Timedelta(hours=hours - 1), end, freq="h", tz="UTC")
    return pd.DataFrame({"datetime_utc": times, "pm25": [base + slope * i for i in range(len(times))]})


def value_at(df, t):
    return float(df.set_index("datetime_utc")["pm25"].loc[t])


def main():
    app = load_app()
    prov = "khonkaen"
    rank = app.NEIGHBOR_RANKING[prov]
    assert len(rank) == 19 and prov not in rank, "ranking ต้องมี 19 จังหวัดอื่นครบ (รวม zero-shot)"
    print(f"เพื่อนบ้านอันดับ 1-4 ของ {prov}: {rank[:4]}")

    # --- T1: อันดับ 1 สด -> k1_exact ตามสูตรเทรน ---
    dfs = {q: make_df(T, base=20.0 + i) for i, q in enumerate(rank)}
    calls = []
    def fetch_all_fresh(q):
        calls.append(q)
        return dfs[q]
    r = app.compute_neighbor_features(prov, T, fetch=fetch_all_fresh)
    d1 = dfs[rank[0]]
    assert r["status"] == "ok" and r["policy"] == "k1_exact" and r["neighbors_used"] == [rank[0]]
    assert abs(r["pm25_neighbor_avg"] - value_at(d1, T)) < 1e-9
    assert abs(r["nb_trend_12h"] - (value_at(d1, T) - value_at(d1, T - pd.Timedelta(hours=12)))) < 1e-9
    assert calls == [rank[0]], "อันดับ 1 สดแล้วต้องไม่ดึงเพื่อนบ้านเพิ่ม (ประหยัด request)"
    print("T1 ผ่าน: อันดับ 1 สด -> k1_exact, ค่า/trend ตรงสูตร, ดึง 1 request")

    # --- T2: อันดับ 1 ค้าง 20 ชม. -> เฉลี่ย 3 อันดับสดถัดไป ---
    dfs2 = {q: make_df(T, base=30.0 + 2 * i) for i, q in enumerate(rank)}
    dfs2[rank[0]] = make_df(T - pd.Timedelta(hours=20), base=99.0)
    r = app.compute_neighbor_features(prov, T, fetch=lambda q: dfs2[q])
    used = rank[1:4]
    assert r["status"] == "ok" and r["policy"] == "avg_nearest_fresh" and r["neighbors_used"] == used
    exp_avg = np.mean([value_at(dfs2[q], T) for q in used])
    exp_ago = np.mean([value_at(dfs2[q], T - pd.Timedelta(hours=12)) for q in used])
    assert abs(r["pm25_neighbor_avg"] - exp_avg) < 1e-9 and abs(r["nb_trend_12h"] - (exp_avg - exp_ago)) < 1e-9
    assert r["pm25_neighbor_avg"] < 90, "ต้องไม่ใช้ค่าของอันดับ 1 ที่ค้าง (99+)"
    print("T2 ผ่าน: อันดับ 1 ค้าง -> เฉลี่ย 3 จังหวัดสด, trend คิดจากค่าเฉลี่ยชุดเดียวกัน")

    # --- T3: อันดับ 1 ดึงไม่ได้ (None) + ที่เหลือสดเพียงบางส่วน ---
    dfs3 = {q: (make_df(T, base=40.0) if i in (2, 5) else None) for i, q in enumerate(rank)}
    r = app.compute_neighbor_features(prov, T, fetch=lambda q: dfs3[q])
    assert r["status"] == "ok" and r["neighbors_used"] == [rank[2], rank[5]] and r["policy"] == "avg_nearest_fresh"
    print("T3 ผ่าน: เพื่อนบ้านบางตัวไม่มีข้อมูล -> ใช้เท่าที่สด (2 จังหวัด)")

    # --- T4: ไม่มีสดเลย -> no_fresh_neighbor และจำกัดจำนวน request ---
    calls4 = []
    def fetch_none(q):
        calls4.append(q)
        return make_df(T - pd.Timedelta(hours=24)) if len(calls4) % 2 else None
    r = app.compute_neighbor_features(prov, T, fetch=fetch_none)
    assert r["status"] == "no_fresh_neighbor" and r["policy"] is None and r["neighbors_used"] == []
    assert np.isnan(r["pm25_neighbor_avg"]) and np.isnan(r["nb_trend_12h"])
    assert len(calls4) == app.NEIGHBOR_MAX_FETCH, f"ต้องดึงไม่เกิน {app.NEIGHBOR_MAX_FETCH} จังหวัด (ดึง {len(calls4)})"
    print("T4 ผ่าน: ไม่มีสดเลย -> no_fresh_neighbor, ดึงไม่เกิน NEIGHBOR_MAX_FETCH")

    # --- T5: เส้นแบ่งความสด (3 ชม.) ---
    for lag_h, expected in ((3.0, True), (3.5, False), (0.0, True)):
        d = make_df(T - pd.Timedelta(hours=lag_h) if lag_h == int(lag_h) else T - pd.Timedelta(hours=lag_h), base=5.0)
        assert app._neighbor_is_fresh(d, T) is expected, f"lag {lag_h}h ควร fresh={expected}"
    assert app._neighbor_is_fresh(make_df(T + pd.Timedelta(hours=2)), T) is True
    assert app._neighbor_is_fresh(make_df(T + pd.Timedelta(hours=5)), T) is False
    assert app._neighbor_is_fresh(None, T) is False and app._neighbor_is_fresh(pd.DataFrame(), T) is False
    print("T5 ผ่าน: เส้นแบ่งความสด 3 ชม. ทั้งเก่ากว่า/ใหม่กว่า และ None/ว่าง")

    # --- T6: เพื่อนบ้านมีข้อมูลแค่ 5 ชม.ล่าสุด -> avg ได้, trend เติมค่าเป็น 0.0 หรือค่าที่ backfill ได้ (ไม่ NaN) ---
    short = make_df(T, hours=5, base=12.0)
    r = app.compute_neighbor_features(prov, T, fetch=lambda q: short)
    assert r["status"] == "ok" and not np.isnan(r["pm25_neighbor_avg"]) and not np.isnan(r["nb_trend_12h"])
    print(f"T6 ผ่าน: ประวัติสั้น -> ไม่ NaN (trend={r['nb_trend_12h']:.2f})")

    # --- T7: count_fire_hotspots ---
    meta = app.PROVINCES_BY_SLUG[prov]
    lat, lon = meta["lat"], meta["lon"]
    far_lat = lat + 1.5  # ~166 กม. นอกรัศมี 100
    hs = pd.DataFrame({
        "latitude": [lat, lat + 0.3, lat, lat, far_lat, lat, lat],
        "longitude": [lon, lon, lon, lon, lon, lon, lon],
        "t": [T, T - pd.Timedelta(hours=12), T - pd.Timedelta(hours=24),  # ขอบซ้าย (t-24h) ไม่นับ
              T + pd.Timedelta(minutes=1),  # หลัง t ไม่นับ
              T - pd.Timedelta(hours=1),  # นอกรัศมี ไม่นับ
              T - pd.Timedelta(hours=23, minutes=59), T - pd.Timedelta(hours=30)],
    })
    assert app.count_fire_hotspots(meta, T, hs) == 3, "นับเฉพาะ (t-24h, t] ในรัศมี 100 กม. -> 3 จุด"
    assert app.count_fire_hotspots(meta, T, None) == 0 and app.count_fire_hotspots(meta, T, hs.iloc[0:0]) == 0
    print("T7 ผ่าน: นับจุดความร้อนถูกทั้งขอบเวลา/รัศมี/ไม่มีข้อมูล")

    # --- T8: get_fire_hotspots ไม่ล่มและไม่รั่วคีย์ ---
    app.FIRMS_KEY = None
    assert app.get_fire_hotspots() == (None, "no_key")
    app.FIRMS_KEY = "SECRETKEY123"
    app._FIRE_CACHE.update({"fetched": None, "df": None})
    real_get = app.requests.get
    def boom(url, **kw):
        raise RuntimeError(f"connection failed: {url}")  # เลียนแบบ requests ที่ใส่ URL (มีคีย์) ในข้อความ error
    app.requests.get = boom
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        res = app.get_fire_hotspots()
    assert res == (None, "unavailable")
    assert "SECRETKEY123" not in buf.getvalue(), "ห้ามพิมพ์คีย์/URL ลง log"
    # cache เก่ายังใช้ได้เมื่อดึงไม่สำเร็จ
    import time as _t
    app._FIRE_CACHE.update({"fetched": _t.monotonic() - app.FIRE_CACHE_TTL_S - 10, "df": hs})
    with contextlib.redirect_stdout(io.StringIO()):
        res = app.get_fire_hotspots()
    assert res[1] == "stale_cache" and res[0] is hs
    # ตอบกลับไม่ใช่ CSV (เช่น invalid key) -> unavailable/stale ไม่ raise
    class Resp:
        status_code = 200
        text = "Invalid MAP_KEY."
    app.requests.get = lambda url, **kw: Resp()
    app._FIRE_CACHE.update({"fetched": None, "df": None})
    with contextlib.redirect_stdout(io.StringIO()):
        assert app.get_fire_hotspots() == (None, "unavailable")
    # ตอบกลับ CSV ปกติ -> parse ได้
    class RespOk:
        status_code = 200
        text = "latitude,longitude,acq_date,acq_time\n16.4,102.8,2026-06-20,551\n"
    app.requests.get = lambda url, **kw: RespOk()
    df_live, src = app.get_fire_hotspots()
    assert src == "live" and len(df_live) == 1 and df_live["t"].iloc[0] == pd.Timestamp("2026-06-20 05:51", tz="UTC")
    assert app.get_fire_hotspots()[1] == "cache"
    app.requests.get, app.FIRMS_KEY = real_get, None
    app._FIRE_CACHE.update({"fetched": None, "df": None})
    print("T8 ผ่าน: FIRMS ไม่มีคีย์/ล้มเหลว/ตอบผิดรูป -> ไม่ล่ม ไม่รั่วคีย์, cache/stale_cache/live ทำงาน")

    # --- T9: get_recent_series (cache + strict) ---
    fetch_calls = []
    real_fetch = app.fetch_recent_pm25_openaq
    def fake_fetch(sensor_id, hours_back=72, offset_hours=0):
        fetch_calls.append(sensor_id)
        return make_df(T)
    app.fetch_recent_pm25_openaq = fake_fetch
    real_fb = app.USE_INPUT_FALLBACK
    real_dustboy = app.USE_DUSTBOY_LOEI
    app.USE_INPUT_FALLBACK = False  # T9 = ความหมายเดิม (strict ไม่ใช้ cache เก่า, ไม่ยิง /hours, ไม่เขียนไฟล์ cache); ทางสำรองทดสอบใน test_input_fallback.py
    app.USE_DUSTBOY_LOEI = False  # "loei" ใช้เป็นจังหวัดตัวอย่างทั่วไปในเทสต์นี้เท่านั้น ไม่เกี่ยวกับ Dustboy จริง (ดู test_dustboy_fallback.py)
    app._SERIES_CACHE.clear()
    app.get_recent_series("loei")
    app.get_recent_series("loei")
    assert len(fetch_calls) == 1, "ครั้งที่ 2 ต้องใช้ cache"
    app.get_recent_series("loei", strict=True, max_age_s=0)
    assert len(fetch_calls) == 2, "max_age_s=0 ต้องดึงใหม่"
    def failing_fetch(sensor_id, hours_back=72, offset_hours=0):
        raise RuntimeError("OpenAQ down")
    app.fetch_recent_pm25_openaq = failing_fetch
    try:
        app.get_recent_series("loei", strict=True, max_age_s=0)
        raise AssertionError("strict ต้อง raise")
    except RuntimeError:
        pass
    assert app.get_recent_series("loei", max_age_s=0) is not None, "เพื่อนบ้าน: ดึงไม่ได้ต้องใช้ cache เก่า"
    app._SERIES_CACHE.clear()
    assert app.get_recent_series("loei", max_age_s=0) is None, "ไม่มี cache + ดึงไม่ได้ -> None (ไม่ raise)"
    app.fetch_recent_pm25_openaq = real_fetch
    app.USE_INPUT_FALLBACK = real_fb
    app.USE_DUSTBOY_LOEI = real_dustboy
    app._SERIES_CACHE.clear()
    print("T9 ผ่าน: cache/strict/fallback ของ get_recent_series")

    # --- T10: select_model force_naive ---
    row = pd.DataFrame([{c: 1.0 for c in app.FEATURE_COLS}])
    lookup = {(app.MODEL_LGB, 1): 0.9, (app.MODEL_XGB, 1): 0.5}
    assert app.select_model(row, 1, prov, lookup, force_naive=True) == app.MODEL_NAIVE
    assert app.select_model(row, 1, prov, lookup, force_naive=False) == app.MODEL_LGB
    assert app.select_model(row, 1, prov, lookup) == app.MODEL_LGB, "default ต้องไม่ force"
    nb_cols = [c for c in ("pm25_neighbor_avg", "nb_trend_12h") if c in app.FEATURE_COLS]
    two_nan = row.copy()
    two_nan[nb_cols] = np.nan
    assert app.compute_missing_ratio(two_nan) < app.SIGMA_CRITICAL, "NaN เฉพาะ feature เพื่อนบ้านไม่ trigger circuit breaker เอง (เหตุผลที่ต้องมี force_naive)"
    assert app.select_model(two_nan, 1, prov, lookup) != app.MODEL_NAIVE
    print("T10 ผ่าน: force_naive บังคับ naive; NaN เพื่อนบ้าน 2/21 ไม่ trigger เอง (ยืนยันว่าต้องใช้ flag)")

    # --- T11: build_feature_row + โมเดลจริง ---
    assert "nb_trend_12h" in app.FEATURE_COLS and "fire_hotspot_count_100km_24h" in app.FEATURE_COLS
    assert len(app.FEATURE_COLS) == (21 if "pm25_neighbor_avg" in app.FEATURE_COLS else 20)
    weather = {"temperature_2m": 30.0, "relative_humidity_2m": 60.0, "wind_speed_10m": 8.0, "precipitation": 0.0,
               "wind_direction_10m": 90.0, "surface_pressure": 1008.0}
    df_t = make_df(T, base=15.0)
    nb = app.compute_neighbor_features(prov, T, fetch=lambda q: dfs[q])
    X, latest_time, latest_pm25 = app.build_feature_row(df_t, weather, meta, nb, 7)
    assert list(X.columns) == app.FEATURE_COLS and X.shape == (1, len(app.FEATURE_COLS))
    assert not X.isna().any().any(), "ต้องไม่มี NaN เมื่อข้อมูลครบ"
    assert X["fire_hotspot_count_100km_24h"].iloc[0] == 7
    if "pm25_neighbor_avg" in X.columns:
        assert abs(X["pm25_neighbor_avg"].iloc[0] - nb["pm25_neighbor_avg"]) < 1e-12
    assert abs(X["nb_trend_12h"].iloc[0] - nb["nb_trend_12h"]) < 1e-12
    assert latest_time == T and abs(latest_pm25 - value_at(df_t, T)) < 1e-9
    for h in (1, 12, 24):
        for model_map in (app.MODELS, app.XGB_MODELS):
            p = model_map[h][0.5].predict(X)[0]
            assert np.isfinite(p), f"โมเดล h={h} ให้ค่าไม่จำกัด"
    X_bad, _, _ = app.build_feature_row(df_t, weather, meta, app.compute_neighbor_features(prov, T, fetch=lambda q: None), 0)
    assert X_bad[nb_cols].isna().all(axis=None), "ไม่มีเพื่อนบ้าน -> NaN (ต้อง force_naive ที่ get_forecast)"
    print(f"T11 ผ่าน: build_feature_row ได้ {len(app.FEATURE_COLS)} feature ตรง FEATURE_COLS, โมเดลทั้ง 2 ทำนายได้")

    # --- T13: offset ป้ายเวลา OpenAQ (CHANGELOG 149) ---
    class FakeResp:
        status_code = 200

        def __init__(self, labels, values):
            self._res = [{"period": {"datetimeFrom": {"utc": t.strftime("%Y-%m-%dT%H:%M:%SZ")}}, "value": v} for t, v in zip(labels, values)]

        def raise_for_status(self):
            pass

        def json(self):
            return {"results": self._res}

    now_h = pd.Timestamp.now(tz="UTC").floor("h")
    captured = {}

    def make_get(labels, values):
        def _get(url, headers=None, params=None, timeout=None):
            captured["params"] = params
            return FakeResp(labels, values)
        return _get

    real_get_ = app.requests.get
    try:
        # (ก) สถานีที่ป้ายเวลาเลื่อน 23 ชม.: label ล่าสุด = now-25h -> หลังบวก offset ต้องได้ ~now-2h, ค่าเดิม, และขอ label ย้อนหลังเพิ่ม offset
        labels = list(pd.date_range(now_h - pd.Timedelta(hours=96), now_h - pd.Timedelta(hours=25), freq="h", tz="UTC"))
        vals = [float(i) for i in range(len(labels))]
        app.requests.get = make_get(labels, vals)
        df_o = app.fetch_recent_pm25_openaq(999, offset_hours=23)
        assert df_o["datetime_utc"].max() == labels[-1] + pd.Timedelta(hours=23) == now_h - pd.Timedelta(hours=2)
        assert df_o["pm25"].tolist() == vals and len(df_o) == len(labels)
        sent_from = pd.Timestamp(captured["params"]["datetime_from"])
        assert abs((pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=72 + 23)) - sent_from) < pd.Timedelta(minutes=5), "ต้องขอ label ย้อนหลังเพิ่ม offset"
        assert pd.Timestamp(captured["params"]["datetime_to"]) >= now_h, "ห้ามตัดปลายช่วงที่ now-offset (ไม่งั้น guard จับไม่ได้)"
        # (ข) OpenAQ แก้ป้ายเวลาแล้ว (label ล่าสุด = now-1h): offset จะล้ำอนาคต -> ต้องทิ้ง offset ไม่ให้ข้อมูลถูกเลื่อน
        labels_fixed = list(pd.date_range(now_h - pd.Timedelta(hours=72), now_h - pd.Timedelta(hours=1), freq="h", tz="UTC"))
        app.requests.get = make_get(labels_fixed, [1.0] * len(labels_fixed))
        buf13 = io.StringIO()
        with contextlib.redirect_stdout(buf13):
            df_f = app.fetch_recent_pm25_openaq(999, offset_hours=23)
        assert df_f["datetime_utc"].max() == labels_fixed[-1], "guard ต้องทิ้ง offset เมื่อข้อมูลล้ำอนาคต"
        assert "ทิ้ง offset" in buf13.getvalue()
        # (ค) offset 0 = พฤติกรรมเดิม (ไม่เลื่อน ขอช่วง 72 ชม.)
        app.requests.get = make_get(labels_fixed, [1.0] * len(labels_fixed))
        df_z = app.fetch_recent_pm25_openaq(999)
        assert df_z["datetime_utc"].max() == labels_fixed[-1]
        assert abs((pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=72)) - pd.Timestamp(captured["params"]["datetime_from"])) < pd.Timedelta(minutes=5)
    finally:
        app.requests.get = real_get_
    # (ง) get_recent_series ส่ง offset จาก registry เฉพาะเมื่อเปิด USE_TIMESTAMP_OFFSET
    offsets_seen = []
    real_fetch2, real_switch, real_fb2 = app.fetch_recent_pm25_openaq, app.USE_TIMESTAMP_OFFSET, app.USE_INPUT_FALLBACK
    app.USE_INPUT_FALLBACK = False  # ไม่เขียนไฟล์ cache จริงด้วยข้อมูลสังเคราะห์
    app.fetch_recent_pm25_openaq = lambda sensor_id, hours_back=72, offset_hours=0: (offsets_seen.append(offset_hours) or make_df(now_h))
    try:
        for switch, prov_, expected in ((False, "surin", 0), (True, "surin", 23), (True, "khonkaen", 0)):
            app.USE_TIMESTAMP_OFFSET = switch
            app._SERIES_CACHE.clear()
            app.get_recent_series(prov_)
            assert offsets_seen[-1] == expected, f"USE_TIMESTAMP_OFFSET={switch} {prov_}: คาด {expected} ได้ {offsets_seen[-1]}"
    finally:
        app.fetch_recent_pm25_openaq, app.USE_TIMESTAMP_OFFSET, app.USE_INPUT_FALLBACK = real_fetch2, real_switch, real_fb2
        app._SERIES_CACHE.clear()
    print("T13 ผ่าน: offset เลื่อนป้ายเวลาถูก, ขอ label ย้อนหลังเพิ่ม, guard ทิ้ง offset เมื่อ OpenAQ แก้แล้ว, offset=0 คงเดิม, registry->offset ตามสวิตช์")

    # --- T12: get_forecast() ตลอดเส้นทาง (ข้อมูลสังเคราะห์, ปิดเครือข่าย/DB/NLG) ---
    now_h = pd.Timestamp.now(tz="UTC").floor("h")
    logged = {"pred": [], "notif": []}
    orig = {n: getattr(app, n) for n in (
        "get_recent_series", "fetch_current_weather_openmeteo", "compute_neighbor_features", "get_fire_hotspots",
        "fetch_rolling_r2_lookup", "log_predictions", "log_notification_snapshot", "backfill_actuals",
        "maybe_generate_daily_nlg", "get_today_daily_nlg")}
    app.fetch_current_weather_openmeteo = lambda lat, lon: weather
    app.get_fire_hotspots = lambda: (None, "unavailable")
    app.fetch_rolling_r2_lookup = lambda province: {}
    app.log_predictions = lambda at, rows, province: logged["pred"].append((province, len(rows), {r["model_used"] for r in rows}))
    app.log_notification_snapshot = lambda at, pm, province: logged["notif"].append(province)
    app.backfill_actuals = lambda df, province: None
    app.maybe_generate_daily_nlg = lambda *a, **k: logged.setdefault("nlg", []).append(a[-1])
    app.get_today_daily_nlg = lambda risk_group, province: {}
    ok_neighbor = {"status": "ok", "policy": "k1_exact", "neighbors_used": [rank[0]], "pm25_neighbor_avg": 12.0, "nb_trend_12h": 1.0}
    none_neighbor = {"status": "no_fresh_neighbor", "policy": None, "neighbors_used": [], "pm25_neighbor_avg": np.nan, "nb_trend_12h": np.nan}
    try:
        # (ก) ปกติ: เพื่อนบ้านสด + FIRMS ใช้ไม่ได้ -> โมเดล v2 ทุก horizon, fire=0 พร้อม flag, ถูก log
        app.get_recent_series = lambda prov_, strict=False, max_age_s=0, info=None: make_df(now_h - pd.Timedelta(hours=1), base=15.0)
        app.compute_neighbor_features = lambda prov_, t, fetch=None: ok_neighbor
        res = app.get_forecast(prov)
        models = {x["model_used"] for x in res["forecast"]}
        assert models <= {app.MODEL_LGB, app.MODEL_XGB} and len(res["forecast"]) == 24
        assert res["data_quality"]["fire_source"] == "unavailable" and res["data_quality"]["fire_count_24h"] == 0
        assert res["data_quality"]["target_stale"] is False and logged["pred"] and logged["notif"] == [prov]
        print("T12ก ผ่าน: ปกติ -> โมเดล v2, fire=0 พร้อม flag, บันทึก log")

        # (ข) ไม่มีเพื่อนบ้านสดเลย -> naive_fallback ครบ 24 horizon, ค่า = ค่าปัจจุบัน, ไม่มี NaN, ยังบันทึก log ด้วยชื่อ naive
        logged["pred"].clear(); logged["notif"].clear()
        app.compute_neighbor_features = lambda prov_, t, fetch=None: none_neighbor
        res = app.get_forecast(prov)
        f = res["forecast"]
        cur = res["current"]["pm25"]
        assert {x["model_used"] for x in f} == {app.MODEL_NAIVE}, "ต้อง naive_fallback ทุก horizon"
        assert all(abs(x["pm25"] - cur) < 0.06 for x in f), "naive = ค่าปัจจุบันคงที่"
        assert all(x["pm25_lo"] == x["pm25_hi"] == x["pm25"] for x in f), "naive ไม่มีช่วงความเชื่อมั่น"
        assert not any(np.isnan(x["pm25"]) for x in f)
        assert res["fallback_triggered_horizons"] == list(range(1, 25))
        assert res["data_quality"]["neighbor_status"] == "no_fresh_neighbor"
        assert logged["pred"] == [(prov, 24, {app.MODEL_NAIVE})] and logged["notif"] == [prov]
        print("T12ข ผ่าน: ไม่มีเพื่อนบ้านสด -> naive ครบ 24 horizon ตลอด get_forecast, ค่า=ปัจจุบัน, บันทึกชื่อ naive")

        # (ค) ข้อมูลเป้าหมายเก่า 10 ชม. -> ตอบได้แต่ไม่ log/ไม่ทำ NLG, flag stale
        logged["pred"].clear(); logged["notif"].clear(); logged.pop("nlg", None)
        app.get_recent_series = lambda prov_, strict=False, max_age_s=0, info=None: make_df(now_h - pd.Timedelta(hours=10), base=15.0)
        app.compute_neighbor_features = lambda prov_, t, fetch=None: ok_neighbor
        res = app.get_forecast(prov)
        assert res["data_quality"]["target_stale"] is True and res["data_quality"]["target_age_hours"] >= 9.9
        assert not logged["pred"] and not logged["notif"] and "nlg" not in logged, "ข้อมูลเก่าห้าม log/ห้ามทำ NLG"
        assert len(res["forecast"]) == 24
        print("T12ค ผ่าน: ข้อมูลเก่า -> ตอบพร้อม flag stale แต่ไม่ log/ไม่ทำ NLG")

        # (ง) ไม่มีข้อมูลเลย -> DataUnavailableError (route จะตอบ 503)
        app.get_recent_series = lambda prov_, strict=False, max_age_s=0, info=None: None
        try:
            app.get_forecast(prov)
            raise AssertionError("ต้อง raise DataUnavailableError")
        except app.DataUnavailableError:
            pass
        print("T12ง ผ่าน: ไม่มีข้อมูลเลย -> DataUnavailableError")
        # (จ) โมเดลที่ไม่ใช้ neighbor/fire: ต้องไม่ดึง/คำนวณ และไม่ force_naive แม้เพื่อนบ้านไม่มีข้อมูล
        app.get_recent_series = lambda prov_, strict=False, max_age_s=0, info=None: make_df(now_h - pd.Timedelta(hours=1), base=15.0)
        called = []
        app.compute_neighbor_features = lambda prov_, t, fetch=None: called.append("nb") or none_neighbor
        app.get_fire_hotspots = lambda: called.append("fire") or (None, "unavailable")
        real_uses = app.USES_NEIGHBOR, app.USES_FIRE
        app.USES_NEIGHBOR, app.USES_FIRE = False, False
        try:
            res = app.get_forecast(prov)
        finally:
            app.USES_NEIGHBOR, app.USES_FIRE = real_uses
        assert called == [], f"ไม่ควรเรียก neighbor/fire เมื่อโมเดลไม่ใช้: {called}"
        assert res["data_quality"]["neighbor_status"] == "not_used" and res["data_quality"]["fire_source"] == "not_used"
        assert {x["model_used"] for x in res["forecast"]} <= {app.MODEL_LGB, app.MODEL_XGB}, "ไม่ใช้ neighbor -> ต้องไม่ force_naive"
        print("T12จ ผ่าน: โมเดลไม่ใช้ neighbor/fire -> ไม่ดึงข้อมูล ไม่ force_naive")
    finally:
        for n, fn in orig.items():
            setattr(app, n, fn)

    print("\nทุกเคสผ่าน (test_live_features_v2)")


if __name__ == "__main__":
    main()

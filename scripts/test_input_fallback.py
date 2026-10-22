"""
ทดสอบทางสำรองเมื่อดึง OpenAQ / Open-Meteo ไม่ได้ (app.py) — assert ธรรมดา ไม่ยิงเครือข่ายจริง ไม่แตะ DB จริง
ไม่เขียนไฟล์ cache จริง (SERIES_DISK_CACHE_DIR ชี้ไปโฟลเดอร์ชั่วคราว)
  1. สภาพอากาศ: ค่าที่ cache ไว้ใช้ซ้ำใน TTL / ดึงสดสำเร็จ / พลาด -> ค่าเก่า <=2 ชม. / เกิน -> climatology / ไม่มี climatology -> raise /
     ค่า null จาก API = ดึงไม่สำเร็จ / USE_INPUT_FALLBACK=False = ดึงสดอย่างเดียวเหมือนเดิม
  2. climatology: ค่าตรงตารางที่ส่งออก, wind sin/cos เข้าโมเดลเป็นค่าเฉลี่ยตรงๆ, ครบ 20 จังหวัด x 12 เดือน x 24 ชั่วโมง
  3. batch Open-Meteo: จับคู่ตามลำดับจังหวัด, จำนวนไม่ตรง/ค่าไม่ครบ -> raise
  4. circuit breaker: เปิดหลังล้มเหลวต่อเนื่อง, ช่วงพักไม่ยิง request, สำเร็จแล้วรีเซ็ต
  5. OpenAQ: /measurements พลาด -> /hours (ไม่บวก offset) ; ทั้งคู่พลาด -> จังหวัดเป้าหมายใช้ cache <=6 ชม. (source=stale_cache) เกินหรือไม่มี -> raise;
     เพื่อนบ้านเหมือนเดิม; ไฟล์ cache ข้ามการรีสตาร์ต
  6. get_forecast: weather=climatology / series=stale_cache -> data_quality บอกแหล่ง + ไม่ log/NLG ; ค่าสำรองที่ผ่านเกณฑ์ (weather stale_cache) -> log ตามปกติ
รัน: python scripts/test_input_fallback.py
"""
import importlib.util
import json
import os
import tempfile
import time

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


app = load("app_mod", "app.py")
TMP = tempfile.mkdtemp(prefix="fallback_test_")
app.SERIES_DISK_CACHE_DIR = TMP  # ห้ามเขียนทับ runtime_cache จริงด้วยข้อมูลสังเคราะห์
assert app.USE_INPUT_FALLBACK and app.WEATHER_CLIM is not None, "ต้องมีไฟล์ climatology ตาม WEATHER_CLIMATOLOGY_PATH"

W_OK = {"temperature_2m": 30.0, "relative_humidity_2m": 55, "wind_speed_10m": 8.0, "precipitation": 0.0,
        "wind_direction_10m": 90, "surface_pressure": 1005.0}
META = app.PROVINCES_BY_SLUG["khonkaen"]


def reset_weather():
    app._WEATHER_CACHE.clear()
    app._WEATHER_BREAKER._fails, app._WEATHER_BREAKER._open_until = 0, 0.0


def reset_openaq():
    app._SERIES_CACHE.clear()
    app._OPENAQ_BREAKER._fails, app._OPENAQ_BREAKER._open_until = 0, 0.0


def raises(fn, exc=Exception):
    try:
        fn()
    except exc:
        return True
    return False


def test_weather_chain():
    real = app.fetch_current_weather_openmeteo
    calls = []
    try:
        reset_weather()
        app.fetch_current_weather_openmeteo = lambda lat, lon: (calls.append(1), W_OK)[1]
        w, info = app.get_current_weather(META)
        assert w == W_OK and info == {"source": "live", "age_hours": 0.0} and len(calls) == 1
        w, info = app.get_current_weather(META)  # TTL ภายใน 10 นาที -> ไม่ยิงซ้ำ
        assert info["source"] == "live" and len(calls) == 1, "ต้องใช้ค่าที่ดึงไว้ซ้ำใน TTL"

        def down(lat, lon):
            raise RuntimeError("Open-Meteo down")
        app.fetch_current_weather_openmeteo = down
        # cache หมด TTL แต่ยังไม่เกิน 2 ชม. -> stale_cache
        app._WEATHER_CACHE["khonkaen"] = (time.monotonic() - 3000, W_OK)
        w, info = app.get_current_weather(META)
        assert info["source"] == "stale_cache" and w == W_OK and 0.8 < info["age_hours"] < 0.9, info
        # เกิน 2 ชม. -> climatology ตรงเดือน/ชั่วโมงของ `at`
        app._WEATHER_CACHE["khonkaen"] = (time.monotonic() - (app.WEATHER_CACHE_MAX_AGE_H * 3600 + 60), W_OK)
        at = pd.Timestamp("2026-03-15 06:00", tz="UTC")
        w, info = app.get_current_weather(META, at=at)
        row = dict(zip(app.WEATHER_CLIM["features"], app.WEATHER_CLIM["table"]["khonkaen"][2][6]))
        assert info["source"] == "climatology" and info["age_hours"] is None
        assert w["temperature_2m"] == row["temp"] and w["relative_humidity_2m"] == row["humidity"]
        assert w["_wind_dir_sin"] == row["wind_dir_sin"] and w["surface_pressure"] == row["pressure"]
        # ไม่มี cache เลย -> climatology ; ไม่มี climatology ด้วย -> raise
        reset_weather()
        assert app.get_current_weather(META, at=at)[1]["source"] == "climatology"
        saved, app.WEATHER_CLIM = app.WEATHER_CLIM, None
        reset_weather()
        assert raises(lambda: app.get_current_weather(META, at=at), RuntimeError), "ไม่มีทั้ง cache และ climatology ต้อง raise"
        app.WEATHER_CLIM = saved
        # ค่า null จาก API = ดึงไม่สำเร็จ ไม่ส่งเข้าโมเดล
        reset_weather()
        app.fetch_current_weather_openmeteo = lambda lat, lon: {**W_OK, "surface_pressure": None}
        assert app.get_current_weather(META, at=at)[1]["source"] == "climatology"
        assert "khonkaen" not in app._WEATHER_CACHE, "ค่าไม่ครบห้ามเข้า cache"
        # สวิตช์ปิด = ดึงสดอย่างเดียว พลาดคือ raise
        app.USE_INPUT_FALLBACK = False
        reset_weather()
        app.fetch_current_weather_openmeteo = down
        assert raises(lambda: app.get_current_weather(META, at=at), RuntimeError)
        app.USE_INPUT_FALLBACK = True
    finally:
        app.fetch_current_weather_openmeteo = real
        app.USE_INPUT_FALLBACK = True
        reset_weather()
    print("ok: สภาพอากาศ (live / TTL / stale_cache / climatology / raise / null / สวิตช์)")


def test_climatology_table_and_features():
    tbl = app.WEATHER_CLIM["table"]
    assert set(tbl) == {p["province_en"] for p in app.PROVINCE_REGISTRY}
    for prov, months in tbl.items():
        assert len(months) == 12 and all(len(h) == 24 and all(len(v) == 7 and np.all(np.isfinite(v)) for v in h) for h in months), prov
    # ค่าที่เข้าโมเดล: wind sin/cos = ค่าเฉลี่ยในตาราง (ไม่ใช่ sin/cos ของทิศที่แปลงกลับ)
    at = pd.Timestamp("2026-04-10 03:00", tz="UTC")
    clim = app.weather_climatology("khonkaen", at)
    idx = pd.date_range(end=at, periods=72, freq="h", tz="UTC")
    df = pd.DataFrame({"datetime_utc": idx, "pm25": 30.0 + np.arange(72) % 5})
    X, _, _ = app.build_feature_row(df, clim, META, {"pm25_neighbor_avg": 20.0, "nb_trend_12h": 1.0}, 3)
    row = dict(zip(app.WEATHER_CLIM["features"], tbl["khonkaen"][3][3]))
    assert X["wind_dir_sin"].iloc[0] == row["wind_dir_sin"] and X["wind_dir_cos"].iloc[0] == row["wind_dir_cos"]
    assert X["temp"].iloc[0] == row["temp"] and X["pressure"].iloc[0] == row["pressure"]
    # ค่าจริงจาก Open-Meteo ยังคิดจากองศาเหมือนเดิม
    X2, _, _ = app.build_feature_row(df, W_OK, META, {"pm25_neighbor_avg": 20.0, "nb_trend_12h": 1.0}, 3)
    assert abs(X2["wind_dir_sin"].iloc[0] - np.sin(np.pi / 2)) < 1e-12 and abs(X2["wind_dir_cos"].iloc[0] - np.cos(np.pi / 2)) < 1e-12
    print("ok: ตาราง climatology ครบ + build_feature_row ใช้ค่าเฉลี่ย sin/cos ตรงๆ")


def test_batch():
    real_get = app.requests.get

    class R:
        def __init__(self, payload, status=200):
            self._p, self.status_code = payload, status
        def json(self):
            return self._p
        def raise_for_status(self):
            if self.status_code >= 400:
                raise RuntimeError("http")

    provs = app.PROVINCE_REGISTRY
    try:
        good = [{"current": {**W_OK, "temperature_2m": float(i)}} for i in range(len(provs))]
        seen = {}
        def fake_get(url, params=None, timeout=None, headers=None):
            seen.update(params)
            return R(good)
        app.requests.get = fake_get
        out = app.fetch_current_weather_batch(provs)
        assert [out[p["province_en"]]["temperature_2m"] for p in provs] == list(map(float, range(len(provs)))), "ต้องจับคู่ตามลำดับ"
        assert seen["latitude"].count(",") == len(provs) - 1 and seen["longitude"].count(",") == len(provs) - 1
        app.requests.get = lambda *a, **k: R(good[:-1])
        assert raises(lambda: app.fetch_current_weather_batch(provs), ValueError), "จำนวนไม่ตรงต้อง raise"
        bad = [dict(g) for g in good]
        bad[3] = {"current": {**W_OK, "wind_speed_10m": None}}
        app.requests.get = lambda *a, **k: R(bad)
        assert raises(lambda: app.fetch_current_weather_batch(provs), ValueError), "ค่า null ต้อง raise"
        # prefetch: สำเร็จ -> ทุกจังหวัดเข้า cache ; ล้มเหลว -> ไม่ raise
        reset_weather()
        app.requests.get = lambda *a, **k: R(good)
        app.prefetch_weather_all()
        assert set(app._WEATHER_CACHE) == {p["province_en"] for p in provs}
        reset_weather()
        app.requests.get = lambda *a, **k: R([], status=400)
        app.prefetch_weather_all()
        assert not app._WEATHER_CACHE
    finally:
        app.requests.get = real_get
        reset_weather()
    print("ok: batch Open-Meteo (ลำดับ / จำนวน / null / prefetch)")


def test_breaker():
    b = app._CircuitBreaker("t", threshold=3, cooldown_s=0.2)
    assert b.allow()
    b.failure(); b.failure()
    assert b.allow()
    b.success()
    b.failure(); b.failure()
    assert b.allow(), "สำเร็จแล้วต้องรีเซ็ตตัวนับ"
    b.failure()
    assert not b.allow(), "ล้มเหลวต่อเนื่องครบ threshold ต้องเปิดวงจร"
    time.sleep(0.25)
    assert b.allow(), "หมด cooldown ต้องลองใหม่"
    # ตอนวงจรเปิด weather ไม่ยิง request
    real = app.fetch_current_weather_openmeteo
    calls = []
    try:
        reset_weather()
        app._WEATHER_BREAKER._open_until = time.monotonic() + 30
        app.fetch_current_weather_openmeteo = lambda lat, lon: (calls.append(1), W_OK)[1]
        assert app.get_current_weather(META, at=pd.Timestamp("2026-03-15 06:00", tz="UTC"))[1]["source"] == "climatology" and not calls
    finally:
        app.fetch_current_weather_openmeteo = real
        reset_weather()
    print("ok: circuit breaker")


def synth(end, hours=72, base=30.0):
    idx = pd.date_range(end=end, periods=hours, freq="h", tz="UTC")
    return pd.DataFrame({"datetime_utc": idx, "pm25": base + np.arange(hours) % 7})


def test_openaq_chain():
    real_m, real_h = app.fetch_recent_pm25_openaq, app.fetch_recent_pm25_openaq_hours
    now_h = pd.Timestamp.now(tz="UTC").floor("h")
    df_live, df_hours = synth(now_h, base=30.0), synth(now_h, base=99.0)  # base ต่างกันเพื่อรู้ว่าอ่านจากแหล่งไหน
    seen = {"m": [], "h": []}

    def down(*a, **k):
        raise RuntimeError("down")
    try:
        # 1) /measurements ปกติ
        reset_openaq()
        app.fetch_recent_pm25_openaq = lambda sid, hours_back=72, offset_hours=0: (seen["m"].append(offset_hours), df_live)[1]
        app.fetch_recent_pm25_openaq_hours = lambda sid, hours_back=72: (seen["h"].append(1), df_hours)[1]
        info = {}
        out = app.get_recent_series("surin", strict=True, max_age_s=0, info=info)
        assert out is df_live and info["source"] == "live" and seen["m"] == [23] and not seen["h"], "surin ต้องส่ง offset 23"
        assert os.path.exists(app._series_disk_path("surin")), "ต้องเขียนไฟล์ cache"
        # 2) /measurements พลาด -> /hours (ไม่มี offset)
        reset_openaq()
        app.fetch_recent_pm25_openaq = down
        info = {}
        out = app.get_recent_series("surin", strict=True, max_age_s=0, info=info)
        assert out is df_hours and info["source"] == "hours_endpoint" and seen["h"] == [1]
        # 3) ทั้งคู่พลาด + มี cache <=6 ชม. -> stale_cache (จังหวัดเป้าหมาย)
        reset_openaq()
        app.fetch_recent_pm25_openaq_hours = down
        app._SERIES_CACHE["khonkaen"] = (time.monotonic() - 2 * 3600, df_live)
        info = {}
        out = app.get_recent_series("khonkaen", strict=True, max_age_s=0, info=info)
        assert out is df_live and info["source"] == "stale_cache" and 1.9 < info["age_hours"] < 2.1
        # cache เกิน 6 ชม. / ไม่มี cache -> raise
        app._SERIES_CACHE["khonkaen"] = (time.monotonic() - 7 * 3600, df_live)
        assert raises(lambda: app.get_recent_series("khonkaen", strict=True, max_age_s=0), RuntimeError)
        reset_openaq()
        assert raises(lambda: app.get_recent_series("khonkaen", strict=True, max_age_s=0), RuntimeError)
        # cache เก็บ None (ไม่มีข้อมูลเลย) -> strict ไม่คืน None แบบ stale
        app._SERIES_CACHE["khonkaen"] = (time.monotonic() - 3600, None)
        assert raises(lambda: app.get_recent_series("khonkaen", strict=True, max_age_s=0), RuntimeError)
        # 4) เพื่อนบ้าน (non-strict): เหมือนเดิม — cache <=6 ชม. ใช้ได้, ไม่มี -> None
        reset_openaq()
        app._SERIES_CACHE["udonthani"] = (time.monotonic() - 3 * 3600, df_live)
        assert app.get_recent_series("udonthani", max_age_s=0) is df_live
        reset_openaq()
        assert app.get_recent_series("udonthani", max_age_s=0) is None
        # 5) สวิตช์ปิด: strict ไม่ใช้ cache เก่า และไม่ลอง /hours
        app.USE_INPUT_FALLBACK = False
        seen["h"].clear()
        app._SERIES_CACHE["khonkaen"] = (time.monotonic() - 3600, df_live)
        assert raises(lambda: app.get_recent_series("khonkaen", strict=True, max_age_s=0), RuntimeError) and not seen["h"]
        app.USE_INPUT_FALLBACK = True
        # 6) breaker: /measurements + /hours พลาดต่อเนื่อง -> เปิดวงจร -> ไม่ยิงเลย
        reset_openaq()
        n = {"m": 0}
        def counting_down(*a, **k):
            n["m"] += 1
            raise RuntimeError("down")
        app.fetch_recent_pm25_openaq = counting_down
        for _ in range(app._OPENAQ_BREAKER.threshold):
            assert app.get_recent_series("udonthani", max_age_s=0) is None
        before = n["m"]
        assert app.get_recent_series("udonthani", max_age_s=0) is None and n["m"] == before, "วงจรเปิดแล้วต้องไม่ยิง request"
        # 7) ไฟล์ cache ข้ามการรีสตาร์ต (ไฟล์ล่าสุดที่เขียน = ผลของข้อ 2 คือ /hours) : ล้างหน่วยความจำแล้วโหลดจากไฟล์
        reset_openaq()
        app._load_series_disk_cache()
        assert "surin" in app._SERIES_CACHE
        age = time.monotonic() - app._SERIES_CACHE["surin"][0]
        assert 0 <= age < 3600, age
        pd.testing.assert_frame_equal(app._SERIES_CACHE["surin"][1].reset_index(drop=True), df_hours.reset_index(drop=True), check_freq=False)
        # ไฟล์เก่าเกิน 6 ชม. ไม่โหลด
        old = time.time() - 7 * 3600
        os.utime(app._series_disk_path("surin"), (old, old))
        reset_openaq()
        app._load_series_disk_cache()
        assert "surin" not in app._SERIES_CACHE
    finally:
        app.fetch_recent_pm25_openaq, app.fetch_recent_pm25_openaq_hours = real_m, real_h
        app.USE_INPUT_FALLBACK = True
        reset_openaq()
    print("ok: OpenAQ (/hours สำรอง / stale_cache <=6 ชม. / เพื่อนบ้านเดิม / สวิตช์ / breaker / ไฟล์ cache)")


def test_hours_parse():
    """fetch_recent_pm25_openaq_hours: parse ผลจาก /hours (mock response) — ไม่บวก offset, ตัด null/ซ้ำ, เรียงเวลา"""
    real_get = app.requests.get

    class R:
        status_code = 200
        def __init__(self, results):
            self._r = results
        def json(self):
            return {"results": self._r}
        def raise_for_status(self):
            pass

    def item(ts, v):
        return {"period": {"datetimeFrom": {"utc": ts}}, "value": v}
    try:
        app.requests.get = lambda url, headers=None, params=None, timeout=None: (
            seen_urls.append(url) or R([item("2026-09-24T02:00:00Z", 5.0), item("2026-09-24T01:00:00Z", 4.0),
                                        item("2026-09-24T01:00:00Z", 4.0), item("2026-09-24T03:00:00Z", None)]))
        seen_urls = []
        df = app.fetch_recent_pm25_openaq_hours(123)
        assert seen_urls[0].endswith("/v3/sensors/123/hours")
        assert list(df["pm25"]) == [4.0, 5.0] and df["datetime_utc"].iloc[0] == pd.Timestamp("2026-09-24 01:00", tz="UTC")
        app.requests.get = lambda *a, **k: R([])
        assert app.fetch_recent_pm25_openaq_hours(123).empty
    finally:
        app.requests.get = real_get
    print("ok: parse /hours")


def test_get_forecast_integration():
    calls = {"log": 0, "notif": 0, "nlg": 0}
    now = pd.Timestamp.now(tz="UTC")
    anchor = now.floor("h") - pd.Timedelta(hours=1)
    prov = "khonkaen"
    names = ("get_recent_series", "fetch_current_weather_openmeteo", "get_fire_hotspots", "log_predictions", "log_notification_snapshot",
             "maybe_generate_daily_nlg", "backfill_actuals", "fetch_rolling_r2_lookup", "get_today_daily_nlg")
    originals = {n: getattr(app, n) for n in names}

    def series(p, strict=False, max_age_s=0, info=None):
        if info is not None:
            info.update(series_info)
        return synth(anchor, 90) if p == prov else synth(anchor, 100, base=26 + (hash(p) % 7))
    series_info = {"source": "live", "age_hours": 0.0}
    app.get_recent_series = series
    app.get_fire_hotspots = lambda: (None, "no_key")
    app.log_predictions = lambda *a, **kw: calls.__setitem__("log", calls["log"] + 1)
    app.log_notification_snapshot = lambda *a, **kw: calls.__setitem__("notif", calls["notif"] + 1)
    app.maybe_generate_daily_nlg = lambda *a, **kw: calls.__setitem__("nlg", calls["nlg"] + 1)
    app.backfill_actuals = lambda df, province: None
    app.fetch_rolling_r2_lookup = lambda province, N=24: {}
    app.get_today_daily_nlg = lambda risk_group, province: {"daily_avg_pm25": None, "explanation": ""}

    def down(lat, lon):
        raise RuntimeError("Open-Meteo down")
    try:
        # (ก) ปกติ: log ตามเดิม, แหล่ง = live
        reset_weather()
        app.fetch_current_weather_openmeteo = lambda lat, lon: W_OK
        res = app.get_forecast(prov)
        dq = res["data_quality"]
        assert dq["weather_source"] == "live" and dq["series_source"] == "live"
        assert calls["log"] == 1 and calls["notif"] == 1 and calls["nlg"] == 1
        # (ข) weather ดึงไม่ได้ + cache <=2 ชม. -> ยัง log (ผ่านเกณฑ์ที่ตั้งไว้)
        calls.update({"log": 0, "notif": 0, "nlg": 0})
        reset_weather()
        app._WEATHER_CACHE[prov] = (time.monotonic() - 3000, W_OK)
        app.fetch_current_weather_openmeteo = down
        res = app.get_forecast(prov)
        assert res["data_quality"]["weather_source"] == "stale_cache" and len(res["forecast"]) == 24
        assert calls["log"] == 1 and calls["nlg"] == 1
        # (ค) weather ดึงไม่ได้ + ไม่มี cache -> climatology, ตอบได้ครบ 24 ชม., ไม่ log/NLG
        calls.update({"log": 0, "notif": 0, "nlg": 0})
        reset_weather()
        res = app.get_forecast(prov)
        dq = res["data_quality"]
        assert dq["weather_source"] == "climatology" and dq["weather_age_hours"] is None and len(res["forecast"]) == 24
        assert calls["log"] == 0 and calls["notif"] == 0 and calls["nlg"] == 0, "climatology ห้าม log/NLG"
        assert all(np.isfinite(f["pm25"]) for f in res["forecast"])
        # (ง) ซีรีส์ pm25 มาจาก cache เก่า -> ไม่ log/NLG, data_quality บอกอายุ
        calls.update({"log": 0, "notif": 0, "nlg": 0})
        reset_weather()
        app.fetch_current_weather_openmeteo = lambda lat, lon: W_OK
        series_info.update({"source": "stale_cache", "age_hours": 1.5})
        res = app.get_forecast(prov)
        dq = res["data_quality"]
        assert dq["series_source"] == "stale_cache" and dq["series_age_hours"] == 1.5
        assert calls["log"] == 0 and calls["notif"] == 0 and calls["nlg"] == 0, "ซีรีส์จาก cache เก่าห้าม log/NLG"
        # (จ) /hours (ข้อมูลเหมือนของจริง) -> log ตามปกติ
        calls.update({"log": 0, "notif": 0, "nlg": 0})
        series_info.update({"source": "hours_endpoint", "age_hours": 0.0})
        res = app.get_forecast(prov)
        assert res["data_quality"]["series_source"] == "hours_endpoint" and calls["log"] == 1
    finally:
        for n, f in originals.items():
            setattr(app, n, f)
        reset_weather()
    print("ok: get_forecast (live / weather cache / climatology / series cache เก่า / hours_endpoint)")


def test_latency_log():
    """log_target_latency: คอลัมน์ series_source, ย้าย header ไฟล์เก่า (5 -> 6 คอลัมน์) ครั้งเดียวโดยแถวเก่าคงเดิม, ไม่ raise เมื่อเขียนไม่ได้"""
    real_path = app.TARGET_LATENCY_LOG_PATH
    dq = {"observed_lag_k": 0, "target_age_hours": 1.2, "input_imputed": None, "series_source": "live"}
    header = app.TARGET_LATENCY_LOG_HEADER
    try:
        # ไฟล์ใหม่
        new = os.path.join(TMP, "lat_new.csv")
        app.TARGET_LATENCY_LOG_PATH = new
        app.log_target_latency("khonkaen", dq)
        app.log_target_latency("surin", {**dq, "observed_lag_k": 1, "series_source": "hours_endpoint"})
        app.log_target_latency("loei", {"observed_lag_k": 0, "target_age_hours": 1.0})  # ไม่มี series_source -> live
        d = pd.read_csv(new)
        assert list(d.columns) == header.split(",") and d["series_source"].tolist() == ["live", "hours_endpoint", "live"]
        assert d["observed_lag_k"].tolist() == [0, 1, 0]
        # ไฟล์เก่า header 5 คอลัมน์
        old = os.path.join(TMP, "lat_old.csv")
        with open(old, "w", encoding="utf-8", newline="") as f:
            f.write(app.TARGET_LATENCY_LOG_HEADER_OLD + "\n2026-09-24T13:58:31+00:00,khonkaen,0,2.0,0\n2026-09-24T14:00:01+00:00,kalasin,1,2.0,0\n")
        app.TARGET_LATENCY_LOG_PATH = old
        app.log_target_latency("udonthani", {**dq, "observed_lag_k": 2, "series_source": "stale_cache"})
        d = pd.read_csv(old)
        assert list(d.columns) == header.split(",") and len(d) == 3
        assert d["observed_lag_k"].tolist() == [0, 1, 2] and d["province_en"].tolist() == ["khonkaen", "kalasin", "udonthani"]
        assert d["series_source"].isna().tolist() == [True, True, False] and d["series_source"].iloc[2] == "stale_cache"
        app.log_target_latency("udonthani", dq)  # ครั้งที่สอง: ไม่ย้ายซ้ำ ไม่ซ้ำแถว
        assert len(pd.read_csv(old)) == 4 and open(old, encoding="utf-8").readline().strip() == header
        # header ที่ไม่รู้จัก: ไม่แตะบรรทัดแรก (ไม่ทำลายไฟล์ที่ไม่ใช่ของเรา) และไม่ raise
        odd = os.path.join(TMP, "lat_odd.csv")
        with open(odd, "w", encoding="utf-8", newline="") as f:
            f.write("a,b,c\n1,2,3\n")
        app.TARGET_LATENCY_LOG_PATH = odd
        app.log_target_latency("khonkaen", dq)
        assert open(odd, encoding="utf-8").readline().strip() == "a,b,c"
        # เขียนไม่ได้ -> ไม่ raise
        app.TARGET_LATENCY_LOG_PATH = os.path.join(TMP, "no_such_dir", "x.csv")
        app.log_target_latency("khonkaen", dq)
    finally:
        app.TARGET_LATENCY_LOG_PATH = real_path
    print("ok: log ความหน่วง (series_source / ย้าย header ไฟล์เก่า / header แปลก / เขียนไม่ได้)")


def test_summarize_filters_non_live():
    """summarize_target_latency: คำแนะนำ MIN_K คิดจากแถว live เท่านั้น (แถวสำรองไม่ดึง p99 ขึ้น), แถว legacy (ไม่มีคอลัมน์ที่ 6) นับเป็น live"""
    import contextlib
    import io
    summ = load("summ", "scripts/summarize_target_latency.py")
    sched = os.path.join(TMP, "sched.csv")
    base = pd.Timestamp("2026-09-24 00:00", tz="UTC")
    lines = [app.TARGET_LATENCY_LOG_HEADER]
    for i in range(300):  # live: กระจายนาที, k=1 เฉพาะนาที 0-4 (ความหน่วงต้นชั่วโมง) ที่เหลือ 0 ; 100 แถวแรกเป็น legacy (ไม่มีฟิลด์ที่ 6)
        ts = base + pd.Timedelta(minutes=11 * i)
        k = 1 if ts.minute < 5 else 0
        row = f"{ts.isoformat(timespec='seconds')},khonkaen,{k},1.0,0"
        lines.append(row if i < 100 else row + ",live")
    for i in range(20):  # hours_endpoint: ช้ากว่า ~1 ชม. -> k สูงกว่า (3) ถ้าไม่กรองจะดึง p99 ขึ้น
        lines.append(f"{(base + pd.Timedelta(minutes=13 * i)).isoformat(timespec='seconds')},surin,3,2.0,0,hours_endpoint")
    for i in range(5):  # stale_cache: k โตตามอายุ cache
        lines.append(f"{(base + pd.Timedelta(minutes=17 * i)).isoformat(timespec='seconds')},kalasin,9,8.0,0,stale_cache")
    with open(sched, "w", encoding="utf-8", newline="") as f:
        f.write("\n".join(lines) + "\n")
    real = (summ.SCHED, summ.SNAP)
    summ.SCHED, summ.SNAP = sched, os.path.join(TMP, "no_snapshot.csv")
    try:
        allrows = summ.load()
        naive = max(2, int(allrows[allrows["k"] <= summ.NORMAL_MAX_K]["k"].quantile(0.99)) + 1)  # ถ้าไม่กรอง
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            res = summ.main()
    finally:
        summ.SCHED, summ.SNAP = real
    assert naive > 2, "ตัวอย่างต้องทำให้การไม่กรองให้คำแนะนำสูงขึ้น (ไม่งั้นเทสต์ไม่มีความหมาย)"
    assert res["n_live"] == 300 and res["n_non_live"] == 25, res
    assert res["recommended"] == 2 and res["conclusive"] is True, res
    out = buf.getvalue()
    assert "hours_endpoint: 20 แถว" in out and "stale_cache: 5 แถว" in out and "ตัดออกจากคำแนะนำ" in out
    print(f"ok: summarize_target_latency (คำแนะนำจาก live = {res['recommended']}, ถ้าไม่กรองจะได้ {naive})")


if __name__ == "__main__":
    test_latency_log()
    test_summarize_filters_non_live()
    test_weather_chain()
    test_climatology_table_and_features()
    test_batch()
    test_breaker()
    test_hours_parse()
    test_openaq_chain()
    test_get_forecast_integration()
    print("\nผ่านทุกการทดสอบ")

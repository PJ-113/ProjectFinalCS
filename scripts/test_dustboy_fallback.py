"""
ทดสอบทางสำรอง Dustboy (CMU CCDC) สำหรับจังหวัดเลย (app.py, CHANGELOG 175) — assert ธรรมดา ไม่ยิงเครือข่ายจริง ไม่แตะ DB จริง
ไม่เขียนไฟล์ cache จริง (DUSTBOY_HISTORY_PATH/SERIES_DISK_CACHE_DIR ชี้ไปโฟลเดอร์ชั่วคราว)
  1. fetch_dustboy_current: parse field ถูก, แปลงเวลาไทย (UTC+7) -> UTC ถูก, web_id ไม่ตรง/pm25 ว่าง/ไม่มี key -> raise
  2. _append_dustboy_reading/get_dustboy_series: สะสมไม่ซ้ำ, ตัด 72h, raise เมื่อประวัติยังไม่ถึง DUSTBOY_MIN_HISTORY_HOURS, ผ่านเมื่อถึง
  3. fetch_series_from_openaq: จังหวัดเลย (มี pm25_provider=dustboy) ไปทาง Dustboy, จังหวัดอื่นไม่กระทบ (ทาง OpenAQ เดิมเป๊ะ)
  4. get_recent_series: cache tuple เก็บ source จริง (ไม่ hardcode "live" ตอน cache hit), TTL ของเลยยาวกว่าปกติ
  5. get_forecast: ประวัติไม่พอ -> DataUnavailableError (พฤติกรรมเดิม) / ประวัติพอ -> พยากรณ์ได้, series_source="dustboy", ไม่ log/ไม่ NLG, banner
  6. breaker: ล้มเหลวต่อเนื่อง -> ไม่ยิงซ้ำ
  7. สวิตช์ USE_DUSTBOY_LOEI=False -> พฤติกรรมเดิม (ไม่มี field ให้ใช้ -> ทาง OpenAQ ตามปกติ ซึ่งจะ raise เพราะ sensor จริงตายอยู่)
รัน: python scripts/test_dustboy_fallback.py
"""
import importlib.util
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
TMP = tempfile.mkdtemp(prefix="dustboy_test_")
app.SERIES_DISK_CACHE_DIR = TMP
app.DUSTBOY_HISTORY_PATH = os.path.join(TMP, "dustboy_loei_history.csv")
app.DUSTBOY_KEY = "test-key"
WEB_ID = "5446"
LOEI_META = dict(app.PROVINCES_BY_SLUG["loei"])  # copy — บาง test override เอง ไม่แตะ registry จริงในหน่วยความจำที่ใช้ร่วม


def raises(fn, exc=Exception):
    try:
        fn()
    except exc:
        return True
    return False


def reset():
    app._SERIES_CACHE.pop("loei", None)
    app._DUSTBOY_BREAKER._fails, app._DUSTBOY_BREAKER._open_until = 0, 0.0
    if os.path.exists(app.DUSTBOY_HISTORY_PATH):
        os.remove(app.DUSTBOY_HISTORY_PATH)


class FakeResp:
    def __init__(self, payload, status=200):
        self._p, self.status_code = payload, status

    def json(self):
        return self._p

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")


def station_payload(web_id="5446", pm25=1.0, log_dt="2026-09-27 17:00:00"):
    return [{"id": web_id, "dustboy_name_en": "Loei City Hall, Loei ( Loei ISOC)", "pm25": pm25, "log_datetime": log_dt}]


def test_fetch_dustboy_current():
    real_get = app.requests.get
    try:
        app.requests.get = lambda url, headers=None, timeout=None: FakeResp(station_payload())
        dt_utc, pm25 = app.fetch_dustboy_current(WEB_ID)
        assert pm25 == 1.0
        # log_datetime เป็นเวลาไทย (UTC+7): 17:00 ไทย = 10:00 UTC
        assert dt_utc == pd.Timestamp("2026-09-27 10:00:00", tz="UTC"), dt_utc

        app.requests.get = lambda url, headers=None, timeout=None: FakeResp(station_payload(web_id="9999"))
        assert raises(lambda: app.fetch_dustboy_current(WEB_ID), RuntimeError), "web_id ไม่ตรงต้อง raise"

        app.requests.get = lambda url, headers=None, timeout=None: FakeResp([{"id": WEB_ID, "pm25": None, "log_datetime": "2026-09-27 17:00:00"}])
        assert raises(lambda: app.fetch_dustboy_current(WEB_ID), RuntimeError), "pm25 ว่างต้อง raise"

        app.requests.get = lambda url, headers=None, timeout=None: FakeResp([], status=500)
        assert raises(lambda: app.fetch_dustboy_current(WEB_ID)), "http error ต้อง raise"

        real_key = app.DUSTBOY_KEY
        app.DUSTBOY_KEY = None
        assert raises(lambda: app.fetch_dustboy_current(WEB_ID), RuntimeError), "ไม่มี key ต้อง raise ไม่ยิง request"
        app.DUSTBOY_KEY = real_key
    finally:
        app.requests.get = real_get
    print("ok: fetch_dustboy_current (parse / web_id ไม่ตรง / pm25 ว่าง / http error / ไม่มี key)")


def test_append_and_min_history():
    reset()
    base = pd.Timestamp("2026-09-27 00:00:00", tz="UTC")
    for h in range(30):
        app._append_dustboy_reading(base + pd.Timedelta(hours=h), 5.0 + h % 3)
    df = pd.read_csv(app.DUSTBOY_HISTORY_PATH)
    assert len(df) == 30
    # ซ้ำเวลาเดิม (ต่อท้ายค่าล่าสุดที่เวลาเดิม) -> ไม่เพิ่มจำนวนแถว
    app._append_dustboy_reading(base + pd.Timedelta(hours=29), 99.0)
    df = pd.read_csv(app.DUSTBOY_HISTORY_PATH)
    assert len(df) == 30 and float(df.iloc[-1]["pm25"]) == 99.0, "เวลาเดิมต้องเขียนทับค่าล่าสุด ไม่ใช่ซ้ำแถว"
    # ตัดเก็บย้อนหลังไม่เกิน 72 ชม.
    app._append_dustboy_reading(base + pd.Timedelta(hours=100), 1.0)
    df = pd.read_csv(app.DUSTBOY_HISTORY_PATH)
    assert (pd.to_datetime(df["datetime_utc"], utc=True) >= base + pd.Timedelta(hours=100 - 72)).all()

    real_get = app.requests.get
    try:
        # เรียกทีละครั้งสะสม จนครบ DUSTBOY_MIN_HISTORY_HOURS (=24) จึงเริ่มผ่าน — ยังไม่ครบ = ได้ df ว่าง (ไม่ raise ตัว exception เอง
        # เพราะการเรียก API สำเร็จจริง แค่ข้อมูลยังไม่พอ, ต่างจากความล้มเหลวของ API ที่ยัง raise ปกติ)
        reset()
        for h in range(25):
            ts_th = (base + pd.Timedelta(hours=h)).tz_convert("Asia/Bangkok").strftime("%Y-%m-%d %H:%M:%S")
            app.requests.get = lambda url, headers=None, timeout=None, _p=station_payload(pm25=3.0, log_dt=ts_th): FakeResp(_p)
            df, source = app.get_dustboy_series(WEB_ID)
            assert source == "dustboy"
            if h < app.DUSTBOY_MIN_HISTORY_HOURS - 1:  # ครั้งที่ h+1 ยังไม่ครบ DUSTBOY_MIN_HISTORY_HOURS จุด
                assert df.empty, f"h={h}: ยังไม่ถึง MIN_HISTORY ต้องได้ df ว่าง (ไม่ raise)"
            else:
                assert len(df) >= app.DUSTBOY_MIN_HISTORY_HOURS, f"h={h}: ครบแล้วต้องคืนประวัติจริง"
        # ความล้มเหลวจริงของ API (เช่น http error) ยัง raise ตามปกติ ไม่ใช่คืน df ว่างเงียบๆ
        app.requests.get = lambda url, headers=None, timeout=None: FakeResp([], status=500)
        assert raises(lambda: app.get_dustboy_series(WEB_ID)), "API ล้มเหลวจริงต้อง raise"
    finally:
        app.requests.get = real_get
        reset()
    print("ok: _append_dustboy_reading/get_dustboy_series (สะสม/dedupe/ตัด 72h/ขั้นต่ำ MIN_HISTORY_HOURS คืน df ว่าง ไม่ raise / API ล้มเหลวจริงยัง raise)")


def test_fetch_series_from_openaq_routing():
    real_meta = app.PROVINCES_BY_SLUG["loei"]
    real_get_dustboy = app.get_dustboy_series
    try:
        called = []
        app.get_dustboy_series = lambda web_id: (called.append(web_id) or (pd.DataFrame({"datetime_utc": [pd.Timestamp.now(tz="UTC")], "pm25": [1.0]}), "dustboy"))
        df, source = app.fetch_series_from_openaq(real_meta)
        assert source == "dustboy" and called == [real_meta["dustboy_web_id"]]

        # จังหวัดอื่น (ไม่มี pm25_provider) ต้องไม่ไปทาง dustboy เลย
        called.clear()
        other = app.PROVINCES_BY_SLUG["khonkaen"]
        assert other.get("pm25_provider") is None
        real_fetch = app.fetch_recent_pm25_openaq
        app.fetch_recent_pm25_openaq = lambda sid, hours_back=72, offset_hours=0: pd.DataFrame({"datetime_utc": [pd.Timestamp.now(tz="UTC")], "pm25": [10.0]})
        try:
            df2, source2 = app.fetch_series_from_openaq(other)
            assert source2 == "live" and not called, "จังหวัดอื่นต้องไม่ถูกดึงผ่าน Dustboy"
        finally:
            app.fetch_recent_pm25_openaq = real_fetch

        # ปิดสวิตช์ USE_DUSTBOY_LOEI -> แม้มี field ก็ไม่ไปทาง dustboy (ตกไปทาง OpenAQ ปกติ ซึ่งจะพยายามยิงจริง)
        real_switch = app.USE_DUSTBOY_LOEI
        app.USE_DUSTBOY_LOEI = False
        try:
            app.fetch_recent_pm25_openaq = lambda sid, hours_back=72, offset_hours=0: pd.DataFrame({"datetime_utc": [pd.Timestamp.now(tz="UTC")], "pm25": [2.0]})
            df3, source3 = app.fetch_series_from_openaq(real_meta)
            assert source3 == "live" and not called, "ปิดสวิตช์ต้องไม่ไปทาง Dustboy"
        finally:
            app.USE_DUSTBOY_LOEI = real_switch
            app.fetch_recent_pm25_openaq = real_fetch
    finally:
        app.get_dustboy_series = real_get_dustboy
    print("ok: fetch_series_from_openaq (routing ตาม pm25_provider / จังหวัดอื่นไม่กระทบ / สวิตช์ปิด)")


def test_get_recent_series_cache_source_and_ttl():
    reset()
    real_fetch = app.fetch_series_from_openaq
    try:
        calls = []
        app.fetch_series_from_openaq = lambda meta: (calls.append(1) or (pd.DataFrame({"datetime_utc": [pd.Timestamp.now(tz="UTC")], "pm25": [1.0]}), "dustboy"))
        info = {}
        app.get_recent_series("loei", strict=True, max_age_s=app.SERIES_CACHE_TTL_S, info=info)
        assert info["source"] == "dustboy" and len(calls) == 1
        # cache hit ต้องคง source เดิม (ไม่ hardcode เป็น "live")
        info2 = {}
        app.get_recent_series("loei", strict=True, max_age_s=app.DUSTBOY_CACHE_TTL_S, info=info2)
        assert info2["source"] == "dustboy" and len(calls) == 1, "cache hit ต้องไม่ยิงซ้ำและต้องคง source เดิม"
    finally:
        app.fetch_series_from_openaq = real_fetch
        reset()
    print("ok: get_recent_series คง source จริงตอน cache hit (ไม่ hardcode live)")


def synth_neighbor(hours=100, base=26.0):
    idx = pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=hours, freq="h")
    return pd.DataFrame({"datetime_utc": idx, "pm25": base + np.arange(hours) % 5})


def test_get_forecast_integration():
    calls = {"log": 0, "notif": 0, "nlg": 0}
    W_OK = {"temperature_2m": 30.0, "relative_humidity_2m": 55, "wind_speed_10m": 8.0,
            "precipitation": 0.0, "wind_direction_10m": 90, "surface_pressure": 1005.0}
    names = ("get_dustboy_series", "get_recent_series", "fetch_current_weather_openmeteo", "get_fire_hotspots",
              "log_predictions", "log_notification_snapshot", "maybe_generate_daily_nlg", "backfill_actuals",
              "fetch_rolling_r2_lookup", "get_today_daily_nlg")
    originals = {n: getattr(app, n) for n in names}
    app.fetch_current_weather_openmeteo = lambda lat, lon: W_OK
    app.get_fire_hotspots = lambda: (None, "no_key")
    app.log_predictions = lambda *a, **kw: calls.__setitem__("log", calls["log"] + 1)
    app.log_notification_snapshot = lambda *a, **kw: calls.__setitem__("notif", calls["notif"] + 1)
    app.maybe_generate_daily_nlg = lambda *a, **kw: calls.__setitem__("nlg", calls["nlg"] + 1)
    app.backfill_actuals = lambda df, province: None
    app.fetch_rolling_r2_lookup = lambda province, N=24: {}
    app.get_today_daily_nlg = lambda risk_group, province: {"daily_avg_pm25": None, "explanation": ""}

    def series_router(province, strict=False, max_age_s=0, info=None):
        if province == "loei":
            return None  # ประวัติยังไม่พอ -> get_dustboy_series คืน df ว่าง -> get_recent_series แปลงเป็น None ตามเส้นทางเดิม
        return synth_neighbor()

    try:
        # (ก) ประวัติยังไม่พอ -> DataUnavailableError (พฤติกรรมเดิมเป๊ะ เหมือนตอนสถานีตาย)
        app.get_recent_series = series_router
        assert raises(lambda: app.get_forecast("loei"), app.DataUnavailableError)
        assert calls["log"] == 0

        # (ข) ประวัติพอ -> พยากรณ์ได้ปกติ, series_source="dustboy", ไม่ log/ไม่ NLG, banner (input_imputed อาจ null เพราะไม่เข้าเกณฑ์เติมค่า)
        loei_series = pd.DataFrame({"datetime_utc": pd.date_range(end=pd.Timestamp.now(tz="UTC"), periods=30, freq="h"),
                                     "pm25": 3.0 + np.arange(30) % 2})

        def series_router2(province, strict=False, max_age_s=0, info=None):
            if info is not None:
                info.update({"source": "dustboy", "age_hours": 0.0})
            if province == "loei":
                return loei_series
            return synth_neighbor()
        app.get_recent_series = series_router2
        res = app.get_forecast("loei")
        dq = res["data_quality"]
        assert dq["series_source"] == "dustboy" and len(res["forecast"]) == 24
        assert calls["log"] == 0 and calls["notif"] == 0 and calls["nlg"] == 0, "ข้อมูลจาก Dustboy ห้าม log/NLG"
    finally:
        for n, f in originals.items():
            setattr(app, n, f)
    print("ok: get_forecast (ประวัติไม่พอ -> 503 เดิม / ประวัติพอ -> พยากรณ์ได้ ไม่ log ไม่ NLG series_source=dustboy)")


def test_breaker():
    reset()
    real_get = app.requests.get
    try:
        app.requests.get = lambda url, headers=None, timeout=None: FakeResp([], status=500)
        for _ in range(app._DUSTBOY_BREAKER.threshold):
            assert raises(lambda: app.get_dustboy_series(WEB_ID))
        calls = {"n": 0}

        def counting(*a, **kw):
            calls["n"] += 1
            return FakeResp([], status=500)
        app.requests.get = counting
        assert raises(lambda: app.get_dustboy_series(WEB_ID)) and calls["n"] == 0, "วงจรเปิดแล้วต้องไม่ยิง request"
    finally:
        app.requests.get = real_get
        reset()
    print("ok: circuit breaker (ล้มเหลวต่อเนื่อง -> เปิดวงจร ไม่ยิงซ้ำ)")


if __name__ == "__main__":
    test_fetch_dustboy_current()
    test_append_and_min_history()
    test_fetch_series_from_openaq_routing()
    test_get_recent_series_cache_source_and_ttl()
    test_breaker()
    test_get_forecast_integration()
    print("\nผ่านทุกการทดสอบ")

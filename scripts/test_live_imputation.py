"""
ทดสอบการนำ H-lite เข้า app.py (impute_target_series / imputed_q_hat / get_forecast) — assert ธรรมดา ไม่มี pytest, ไม่แตะ DB จริง
(ฟังก์ชันที่เขียน DB/เรียก API ถูก monkeypatch เป็นตัวนับ/ค่าปลอม; ข้อมูลอนุกรมสร้างสังเคราะห์สัมพัทธ์กับเวลาปัจจุบัน)
  1. parity: ค่าที่ app เติม = ค่าที่วิธี HL ในสคริปต์ทดลอง (experiment_imputation_accuracy) ให้ ที่ทุก j (ความต่าง <1e-6) บนข้อมูลจริง
  2. เงื่อนไข: k < MIN_K, k > MAX_K, ไม่มีเพื่อนบ้าน ณ ปัจจุบัน, ฟีเจอร์ปิด -> ไม่เติม (df เดิมเป๊ะ)
  3. imputed_q_hat: ใช้ k ถัดไปที่มากกว่า, interpolate ตาม h, ตรงตารางที่จุดทดสอบ
  4. get_forecast: เมื่อเติม -> data_quality.input_imputed, พยากรณ์เริ่มที่เวลาปัจจุบัน, ไม่ log/ไม่ NLG, backfill ได้เฉพาะข้อมูลจริง, ช่วงกว้างกว่าเดิม;
     เมื่อไม่เติม -> log ตามเดิมและ input_imputed = None
รัน: python scripts/test_live_imputation.py   (ต้องมีไฟล์ค่าสัมประสิทธิ์ตาม IMPUTATION_CONFIG_PATH ใน app.py)
"""
import glob
import importlib.util
import os

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


app = load("app_mod", "app.py")
assert app.IMPUTATION_CFG is not None, "IMPUTATION_CFG ว่าง — ไม่พบ/อ่านไฟล์ค่าสัมประสิทธิ์ไม่ได้ (IMPUTATION_CONFIG_PATH)"
CFG = app.IMPUTATION_CFG


def synth_series(province_hours, end, base=30.0, phase=0.0):
    idx = pd.date_range(end=end, periods=province_hours, freq="h", tz="UTC")
    vals = base + 10 * np.sin(2 * np.pi * (np.arange(province_hours) + phase) / 24)
    return pd.DataFrame({"datetime_utc": idx, "pm25": vals})


def test_parity_with_experiment():
    ea = load("ea", "scripts/experiment_imputation_accuracy.py")
    P = ea.load_panels(need_full_dataset=False)
    tuned_files = sorted(f for f in glob.glob(os.path.join(ea.MODEL_DIR, "imputation_tuned_params_2*.json")) if "_quick" not in f)
    import json
    tune = json.load(open(tuned_files[-1], encoding="utf-8"))["tune"]
    model = ea.fit_trailing(P, np.random.default_rng(ea.SEED), 60000, tune, fast=True)
    rng = np.random.default_rng(3)
    checked = 0
    for pname in ("khonkaen", "udonthani", "nakhonratchasima", "sakonnakhon"):
        p = P["provs"].index(pname)
        lo, hi = P["bounds"][p]["test"]
        k = 12
        valid = ~np.isnan(P["obs"][:, p])
        cand = [t for t in range(lo + 100, hi - k - 2) if valid[t]]
        for t0 in rng.choice(cand, size=3, replace=False):
            tau_end = t0 + k
            anchor = P["grid"][tau_end]
            # ซีรีส์เป้าหมายจนถึง t0 (ข้อมูลจริง); เพื่อนบ้านจนถึง anchor (ข้อมูลจริง) — mock get_recent_series
            def fetch(q, _P=P, _anchor=anchor):
                qi = _P["provs"].index(q)
                s = pd.Series(_P["obs"][:, qi], index=_P["grid"]).dropna()
                s = s[s.index <= _anchor]
                return pd.DataFrame({"datetime_utc": s.index, "pm25": s.values})
            s0 = pd.Series(P["obs"][:, p], index=P["grid"]).dropna()
            s0 = s0[s0.index <= P["grid"][t0]]
            df = pd.DataFrame({"datetime_utc": s0.index, "pm25": s0.values})
            now = anchor + pd.Timedelta(hours=1, minutes=17)
            aug, info = app.impute_target_series(pname, df, now=now, fetch=fetch)
            if info is None:
                continue
            got = aug.iloc[len(df):]["pm25"].values
            assert len(got) == k and info["k_hours"] == k
            js = np.arange(1, k + 1)
            F = ea.make_feats(P, np.full(k, p), np.full(k, t0), js)
            ref = ea.impute_trailing(P, model, F, ("HL",))["HL"]
            assert np.allclose(got, ref, atol=1e-6), f"{pname} t0={P['grid'][t0]} max diff {np.abs(got - ref).max()}"
            checked += 1
    assert checked >= 6, f"ตรวจได้น้อยเกินไป ({checked})"
    print(f"ok: parity กับ HL ในสคริปต์ทดลอง ({checked} กรณี, k=12 ทุก j)")


def test_conditions():
    now = pd.Timestamp.now(tz="UTC")
    anchor = now.floor("h") - pd.Timedelta(hours=1)
    prov = "khonkaen"
    nb = app.NEIGHBOR_RANKING[prov][:5]
    fetch_all = lambda q: synth_series(100, anchor, base=25 + hash(q) % 5)

    def df_for(k):
        return synth_series(80, anchor - pd.Timedelta(hours=k))
    # k = 0/1 (< MIN_K) -> ไม่เติม
    for k in range(0, app.IMPUTATION_MIN_K):
        df = df_for(k)
        aug, info = app.impute_target_series(prov, df, now=now, fetch=fetch_all)
        assert info is None and aug is df
    # k เกิน MAX_K -> ไม่เติม
    df = df_for(app.IMPUTATION_MAX_K + 1)
    aug, info = app.impute_target_series(prov, df, now=now, fetch=fetch_all)
    assert info is None and aug is df
    # k ปกติ -> เติม k แถว ต่อเนื่องรายชั่วโมง จบที่ anchor และไม่ติดลบ
    for k in (app.IMPUTATION_MIN_K, 5, 24, app.IMPUTATION_MAX_K):
        df = df_for(k)
        aug, info = app.impute_target_series(prov, df, now=now, fetch=fetch_all)
        assert info is not None and info["k_hours"] == k and len(aug) == len(df) + k
        assert aug["datetime_utc"].iloc[-1] == anchor and (aug["pm25"].iloc[len(df):] >= 0).all()
        assert (aug["datetime_utc"].diff().dropna().iloc[len(df) - 1:] == pd.Timedelta(hours=1)).all()
    # ไม่มีเพื่อนบ้านเลย -> ไม่เติม
    df = df_for(6)
    aug, info = app.impute_target_series(prov, df, now=now, fetch=lambda q: None)
    assert info is None and aug is df
    # เพื่อนบ้านมีข้อมูลแต่หยุดก่อน anchor (ไม่มีค่า ณ ปัจจุบัน) -> ไม่เติม
    stale_nb = lambda q: synth_series(100, anchor - pd.Timedelta(hours=10))
    aug, info = app.impute_target_series(prov, df_for(6), now=now, fetch=stale_nb)
    assert info is None
    # ฟีเจอร์ปิด -> ผ่านตรง
    saved = app.IMPUTATION_CFG
    app.IMPUTATION_CFG = None
    try:
        df = df_for(8)
        aug, info = app.impute_target_series(prov, df, now=now, fetch=fetch_all)
        assert info is None and aug is df
    finally:
        app.IMPUTATION_CFG = saved
    print("ok: เงื่อนไขการเติม/ไม่เติม")


def test_q_hat_lookup():
    tab = CFG["_q"][app.MODEL_LGB]
    ks, hs = sorted(tab), sorted(tab[sorted(tab)[0]])
    for k in ks:
        for h in hs:
            assert abs(app.imputed_q_hat(app.MODEL_LGB, k, h) - tab[k][h]) < 1e-12
    assert abs(app.imputed_q_hat(app.MODEL_LGB, 4, 6) - tab[6][6]) < 1e-12          # k ไม่ตรง -> k ถัดไปที่มากกว่า
    assert abs(app.imputed_q_hat(app.MODEL_LGB, 60, 6) - tab[ks[-1]][6]) < 1e-12    # เกินตาราง -> k สูงสุด
    lo, hi = tab[3][6], tab[3][12]
    assert min(lo, hi) - 1e-12 <= app.imputed_q_hat(app.MODEL_LGB, 3, 9) <= max(lo, hi) + 1e-12  # interpolate ตาม h
    assert app.imputed_q_hat(app.MODEL_XGB, 3, 1) > 0
    print("ok: imputed_q_hat")


def test_get_forecast_integration():
    calls = {"log": 0, "notif": 0, "nlg": 0, "backfill": []}
    now = pd.Timestamp.now(tz="UTC")
    anchor = now.floor("h") - pd.Timedelta(hours=1)
    prov = "khonkaen"

    def series_for(k):
        def _get(p, strict=False, max_age_s=0, info=None):
            if p == prov:
                return synth_series(90, anchor - pd.Timedelta(hours=k))
            return synth_series(100, anchor, base=26 + (hash(p) % 7))
        return _get

    originals = {n: getattr(app, n) for n in ("get_recent_series", "fetch_current_weather_openmeteo", "get_fire_hotspots", "log_predictions",
                                               "log_notification_snapshot", "maybe_generate_daily_nlg", "backfill_actuals",
                                               "fetch_rolling_r2_lookup", "get_today_daily_nlg")}
    app.fetch_current_weather_openmeteo = lambda lat, lon: {"temperature_2m": 30.0, "relative_humidity_2m": 55, "wind_speed_10m": 8.0,
                                                          "precipitation": 0.0, "wind_direction_10m": 90, "surface_pressure": 1005.0}
    app.get_fire_hotspots = lambda: (None, "no_key")
    app.log_predictions = lambda *a, **kw: calls.__setitem__("log", calls["log"] + 1)
    app.log_notification_snapshot = lambda *a, **kw: calls.__setitem__("notif", calls["notif"] + 1)
    app.maybe_generate_daily_nlg = lambda *a, **kw: calls.__setitem__("nlg", calls["nlg"] + 1)
    app.backfill_actuals = lambda df, province: calls["backfill"].append(df["datetime_utc"].max())
    app.fetch_rolling_r2_lookup = lambda province, N=24: {}
    app.get_today_daily_nlg = lambda risk_group, province: {"daily_avg_pm25": None, "explanation": ""}
    try:
        # --- เติมค่า (k=8) ---
        k = 8
        app.get_recent_series = series_for(k)
        res = app.get_forecast(prov)
        ii = res["data_quality"]["input_imputed"]
        assert ii is not None and ii["k_hours"] == k and ii["method"] == "H-lite"
        assert res["data_quality"]["observed_lag_k"] == k, res["data_quality"]["observed_lag_k"]
        assert res["data_quality"]["target_stale"] is False
        assert len(res["forecast"]) == 24
        first_time = (anchor + pd.Timedelta(hours=1)).tz_convert("Asia/Bangkok").strftime("%H:%M")
        assert res["forecast"][0]["time"] == first_time, (res["forecast"][0]["time"], first_time)
        assert calls["log"] == 0 and calls["notif"] == 0 and calls["nlg"] == 0, "ห้าม log/NLG เมื่อ input ถูกเติมค่า"
        assert calls["backfill"][-1] == anchor - pd.Timedelta(hours=k), "backfill ต้องได้ข้อมูลจริงล้วน (ไม่มีแถวที่เติม)"
        assert abs(res["current"]["pm25"] - ii["estimated_pm25"]) < 0.06
        # ช่วงต้องกว้างกว่าที่ใช้ q เดิม
        for f in res["forecast"]:
            if f["model_used"] != app.MODEL_NAIVE:
                assert f["pm25_hi"] >= f["pm25"] >= 0
        # --- ไม่เติม (k=1: ความหน่วงปกติ) ---
        calls.update({"log": 0, "notif": 0, "nlg": 0, "backfill": []})
        app.get_recent_series = series_for(1)
        res2 = app.get_forecast(prov)
        assert res2["data_quality"]["input_imputed"] is None and res2["data_quality"]["observed_lag_k"] == 1
        assert calls["log"] == 1 and calls["notif"] == 1 and calls["nlg"] == 1
        assert calls["backfill"][-1] == anchor - pd.Timedelta(hours=1)
        # --- k เกิน MAX_K: พฤติกรรมเดิม (stale, ไม่ log, ไม่ input_imputed) ---
        calls.update({"log": 0, "notif": 0, "nlg": 0, "backfill": []})
        app.get_recent_series = series_for(app.IMPUTATION_MAX_K + 5)
        res3 = app.get_forecast(prov)
        assert res3["data_quality"]["input_imputed"] is None and res3["data_quality"]["target_stale"] is True
        assert calls["log"] == 0 and calls["nlg"] == 0
    finally:
        for n, f in originals.items():
            setattr(app, n, f)
    print("ok: get_forecast (เติมค่า / ไม่เติม / เกิน MAX_K)")


if __name__ == "__main__":
    test_q_hat_lookup()
    test_conditions()
    test_get_forecast_integration()
    test_parity_with_experiment()
    print("\nผ่านทุกการทดสอบ")

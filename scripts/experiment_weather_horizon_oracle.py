"""
ทดสอบ "เพดานผลลัพธ์" ของการใช้สภาพอากาศ ณ เวลาเป้าหมาย (t+h) แทนสภาพอากาศปัจจุบัน (t) ที่ใช้ซ้ำทุก horizon

ที่มา: โมเดล pooled ใช้ 7 feature สภาพอากาศ (temp/humidity/wind_speed/precipitation/wind_dir_sin/cos/pressure) เป็นค่า ณ เวลาปัจจุบัน
เดียวกันไม่ว่าจะทำนาย t+1 หรือ t+24 — ข้อจำกัดนี้เขียนไว้ตั้งแต่ยุค Prophet ("ยังไม่มี weather forecast รายชั่วโมงจริง") ไม่เคยถูกแก้
การทดลองนี้ตอบคำถาม "ถ้าพยากรณ์อากาศแม่นสมบูรณ์แบบ (= ใช้ค่าจริงที่เกิดขึ้น ณ เวลาเป้าหมาย) จะช่วยได้แค่ไหน" — เป็นเพดานบนสุด ไม่ใช่สิ่งที่
deploy ได้จริง (live ต้องใช้พยากรณ์จาก Open-Meteo ซึ่งแม่นน้อยกว่านี้เสมอ) ถ้าเพดานยังช่วยน้อย แปลว่าไม่คุ้มสร้างระบบดึงพยากรณ์จริงต่อ

วิธี: dataset เทรน (`train_multiprovince_models.DATA_PATH`) มีสภาพอากาศเป็นค่ารายชั่วโมงต่อแถวอยู่แล้ว (มาจาก fetch_weather() merge
เข้ากริดรายชั่วโมงเต็มช่วง) จึง "สภาพอากาศที่เวลา t+h" ของแถว t = ค่าที่แถว t+h ของจังหวัดเดียวกัน (time-shift ธรรมดา ไม่ต้องดึงข้อมูลใหม่)
oracle variant: แทนที่ 7 คอลัมน์สภาพอากาศด้วยค่า ณ t+h ทั้งตอนเทรนและตอนประเมิน (fit ใหม่ต่อ horizon เหมือน baseline)
ใช้ hyperparameter ที่จูนแล้ว (BEST, เดียวกับ scripts/validate_feature_refinements.py) ไม่ retune เพื่อแยกผลของ feature ล้วนๆ

เกณฑ์ตัดสิน (ตั้งก่อนรัน): บน calib, oracle ต้องลด MAE เฉลี่ยที่ h in {18,24} >= 2% เทียบ baseline ทั้ง LGB และ XGB พร้อมกัน
จึงไปยืนยันบน test (เกณฑ์เดียวกัน ไม่ผ่อน) ไม่ผ่าน calib -> หยุดที่ calib รายงานผลลบตามจริง ไม่แตะ test
รัน: python scripts/experiment_weather_horizon_oracle.py
"""
import importlib.util
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(_ROOT, "models")
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
HORIZONS = [1, 6, 12, 18, 24]
LONG_HORIZONS = [18, 24]
BURNING_MONTHS = {2, 3, 4}
WEATHER_COLS = ["temp", "humidity", "wind_speed", "precipitation", "wind_dir_sin", "wind_dir_cos", "pressure"]
PASS_THRESHOLD_PCT = 2.0  # ลด MAE เฉลี่ยที่ LONG_HORIZONS อย่างน้อยเท่านี้ % ทั้ง 2 โมเดล

BEST = {  # เดียวกับ scripts/validate_feature_refinements.py (hyperparameter ที่จูนแล้วของโมเดล production)
    "lgb": {"learning_rate": 0.05, "num_leaves": 63, "min_child_samples": 50, "reg_lambda": 1.0, "n_estimators": 400},
    "xgb": {"learning_rate": 0.03, "max_depth": 8, "min_child_weight": 20, "reg_lambda": 10.0, "n_estimators": 800},
}


def load_train_module():
    spec = importlib.util.spec_from_file_location("tm", os.path.join(_ROOT, "scripts", "train_multiprovince_models.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def weather_at_target(df_all, h, keys):
    """ค่า WEATHER_COLS ของ (province_en, datetime_utc + h ชม.) แบบ time-indexed — คืน DataFrame เรียงตาม keys
    (แถวปลายช่วงเวลาของแต่ละจังหวัดที่ไม่มี t+h ให้ดู -> NaN ตัด dropna ทีหลัง เหมือนเป้าหมาย pm25_t+h เอง)"""
    lead = df_all[["province_en", "datetime_utc"] + WEATHER_COLS].copy()
    lead["datetime_utc"] = lead["datetime_utc"] - pd.Timedelta(hours=h)  # แถวนี้ห่างไปข้างหน้า h ชม. คือค่าที่ต้องการของเวลา key
    out = keys[["province_en", "datetime_utc"]].merge(lead, on=["province_en", "datetime_utc"], how="left")
    assert len(out) == len(keys)
    return out[WEATHER_COLS].reset_index(drop=True)


def fit_p50(X, y, kind, train_mod):
    if kind == "lgb":
        import lightgbm as lgb
        return lgb.LGBMRegressor(objective="quantile", alpha=0.5, random_state=42, verbose=-1,
                                  **train_mod.BAGGING_PARAMS_LGB, **BEST["lgb"]).fit(X, y)
    import xgboost as xgb
    return xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, tree_method="hist",
                             random_state=42, verbosity=0, **train_mod.BAGGING_PARAMS_XGB, **BEST["xgb"]).fit(X, y)


def eval_horizon(train_df, eval_df, df_all, feats, h, kind, variant, train_mod):
    tcol = f"pm25_t+{h}"
    tr, ev = train_df.dropna(subset=[tcol]).copy(), eval_df.dropna(subset=[tcol]).copy()
    if variant == "oracle":
        tr_w, ev_w = weather_at_target(df_all, h, tr), weather_at_target(df_all, h, ev)
        ok_tr, ok_ev = tr_w.notna().all(axis=1).values, ev_w.notna().all(axis=1).values
        tr, ev = tr[ok_tr].reset_index(drop=True), ev[ok_ev].reset_index(drop=True)
        tr_w, ev_w = tr_w[ok_tr].reset_index(drop=True), ev_w[ok_ev].reset_index(drop=True)
        Xtr, Xev = tr[feats].copy(), ev[feats].copy()
        for c in WEATHER_COLS:
            Xtr[c], Xev[c] = tr_w[c].values, ev_w[c].values
    else:
        Xtr, Xev = tr[feats], ev[feats]
    model = fit_p50(Xtr, tr[tcol] - tr["pm25"], kind, train_mod)
    pred = model.predict(Xev) + ev["pm25"].values
    y = ev[tcol].values
    burning = ev["datetime_utc"].dt.month.isin(BURNING_MONTHS).values
    hi = y > np.percentile(y, 90)
    return {"variant": variant, "model": kind, "horizon": h, "n": len(ev),
            "mae": mean_absolute_error(y, pred), "mae_burning": mean_absolute_error(y[burning], pred[burning]),
            "mae_gt_p90": mean_absolute_error(y[hi], pred[hi])}


def run_split(split_name, train_df, eval_df, df_all, feats, horizons, train_mod):
    rows = []
    for kind in ("lgb", "xgb"):
        for h in horizons:
            for variant in ("baseline", "oracle"):
                r = eval_horizon(train_df, eval_df, df_all, feats, h, kind, variant, train_mod)
                r["split"] = split_name
                rows.append(r)
                print(f"  [{split_name}/{kind}/h={h:>2}/{variant:8s}] n={r['n']:>6} MAE={r['mae']:.3f}", flush=True)
    return pd.DataFrame(rows)


def check_pass(res, split_name):
    """คืน (passed: bool, detail: dict) — ผ่านเมื่อ oracle ลด MAE เฉลี่ยที่ LONG_HORIZONS >= PASS_THRESHOLD_PCT% ทั้ง lgb และ xgb"""
    r = res[(res["split"] == split_name) & (res["horizon"].isin(LONG_HORIZONS))]
    detail = {}
    passed = True
    for kind in ("lgb", "xgb"):
        b = r[(r["model"] == kind) & (r["variant"] == "baseline")]["mae"].mean()
        o = r[(r["model"] == kind) & (r["variant"] == "oracle")]["mae"].mean()
        d = (b - o) / b * 100  # บวก = oracle ดีขึ้น
        detail[kind] = {"baseline_mae": b, "oracle_mae": o, "improve_pct": d}
        passed &= d >= PASS_THRESHOLD_PCT
    return passed, detail


def print_full_horizon_curve(res, split_name):
    """เส้นโค้ง MAE ทุก horizon (เชิงพรรณนา ไม่ใช้ตัดสิน) — ดูว่าช่องว่าง oracle/baseline ขยายตาม horizon ไหม"""
    r = res[res["split"] == split_name]
    t = r.pivot_table(index=["model", "horizon"], columns="variant", values="mae")
    t["improve_pct"] = (t["baseline"] - t["oracle"]) / t["baseline"] * 100
    print(f"\n[{split_name}] เส้นโค้ง MAE ทุก horizon ที่ทดสอบ (improve_pct บวก = oracle ดีกว่า):")
    print(t.round(3).to_string())


def main():
    train_mod = load_train_module()
    df = pd.read_csv(train_mod.DATA_PATH)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    for c in WEATHER_COLS:
        assert c in train_mod.FEATURE_COLS, c
    trained = df[~df["is_zero_shot"]].reset_index(drop=True)
    train_df, calib_df, test_df = train_mod.split_train_calib_test(trained)
    feats = train_mod.FEATURE_COLS

    print("=== calib (ตัดสินก่อน) ===")
    res_calib = run_split("calib", train_df, calib_df, df, feats, HORIZONS, train_mod)
    print_full_horizon_curve(res_calib, "calib")
    passed, detail = check_pass(res_calib, "calib")
    print(f"\nเกณฑ์หลัก (ลด MAE >= {PASS_THRESHOLD_PCT}% ที่ h in {LONG_HORIZONS} ทั้ง 2 โมเดล) บน calib:")
    for kind, d in detail.items():
        print(f"  {kind}: baseline={d['baseline_mae']:.3f} oracle={d['oracle_mae']:.3f} ลดลง {d['improve_pct']:+.2f}%")
    print(f"  -> {'ผ่าน ไปทดสอบยืนยันบน test' if passed else 'ไม่ผ่าน หยุดที่นี่ (ไม่แตะ test)'}")

    all_res = [res_calib]
    if passed:
        print("\n=== test (ยืนยันครั้งเดียว) ===")
        res_test = run_split("test", train_df, test_df, df, feats, HORIZONS, train_mod)
        print_full_horizon_curve(res_test, "test")
        passed_test, detail_test = check_pass(res_test, "test")
        print(f"\nยืนยันบน test (เกณฑ์เดียวกัน ไม่ผ่อน):")
        for kind, d in detail_test.items():
            print(f"  {kind}: baseline={d['baseline_mae']:.3f} oracle={d['oracle_mae']:.3f} ลดลง {d['improve_pct']:+.2f}%")
        print(f"  -> {'ผ่านยืนยัน — เพดานสูงพอที่จะพิจารณาสร้างระบบดึงพยากรณ์จริง' if passed_test else 'ไม่ผ่านยืนยันบน test — ไม่แนะนำให้ลงทุนต่อ'}")
        all_res.append(res_test)
    else:
        print("\n(ข้าม test ตามเกณฑ์ที่ตั้งไว้ล่วงหน้า — calib ไม่ผ่าน)")

    out = pd.concat(all_res, ignore_index=True)
    path = os.path.join(MODEL_DIR, f"experiment_weather_horizon_oracle_{TIMESTAMP}.csv")
    out.to_csv(path, index=False)
    print(f"\nผลดิบ: {path}")


if __name__ == "__main__":
    main()

"""
SHAP analysis สำหรับโมเดล pooled (LightGBM/XGBoost) — อธิบายค่าพยากรณ์ P50 เฉพาะ
โมเดล pooled ทายค่า **residual** (pm25_t+h - pm25_now) ไม่ใช่ค่าจริง — SHAP values ที่ได้จึง
อธิบายอิทธิพลต่อ "ส่วนต่าง" ไม่ใช่ค่า PM2.5 สัมบูรณ์

ไฟล์นี้ทำ 3 หน้าที่แยกกัน:
1. Batch analysis (รันเมื่อเรียกเป็นสคริปต์โดยตรง): คำนวณ SHAP บน test set ที่ horizon ตัวแทน
   t+1/t+12/t+24 ของทั้ง 2 โมเดล สรุป top-5 feature ต่อ (model, horizon)
2. Meteorological plausibility check: ตั้งสมมติฐานทิศทางไว้ก่อนต่อ feature อุตุนิยมวิทยาหลัก
   (precipitation, pressure, wind_speed) แล้วเช็คว่า SHAP value จริงไปทิศทางเดียวกับสมมติฐานไหม
3. get_shap_explanation(): ฟังก์ชัน reusable สำหรับอธิบายค่าพยากรณ์ "จุดเดียว" ตามโมเดลที่
   Dynamic Routing เลือกใช้จริง (model_used ใน app.py: "lightgbm_pooled_v3"/"xgboost_pooled_v3"/
   "naive_fallback") — import ได้โดยไม่ trigger batch analysis
"""
import importlib.util
import os
import pickle
import sys

import numpy as np
import pandas as pd
import shap

sys.stdout.reconfigure(encoding="utf-8")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(_ROOT, "data", "multiprovince_pm25_features_raw_spatial_k1_firehotspot_nbtrend_20260919_174805.csv")
LGB_MODEL_PATH = os.path.join(_ROOT, "models", "lgb_pm25_multiprovince_residual_bagged_quantile_20260919_203455.pkl")
XGB_MODEL_PATH = os.path.join(_ROOT, "models", "xgb_pm25_multiprovince_residual_bagged_quantile_20260919_203455.pkl")
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
OUT_PKL_PATH = os.path.join(_ROOT, "models", f"shap_values_by_horizon_{TIMESTAMP}.pkl")
OUT_CSV_PATH = os.path.join(_ROOT, "models", f"shap_top_features_by_horizon_{TIMESTAMP}.csv")
REP_HORIZONS = [1, 12, 24]
TOP_N = 5
EXPLAIN_QUANTILE = 0.5  # อธิบายเฉพาะ P50 (จุดพยากรณ์หลัก)
SUBSAMPLE_STRIDE = 3  # ใช้ทุกๆ 3 แถวของ test เพื่อคุม runtime ของ TreeSHAP

# สมมติฐานทิศทางเชิงอุตุนิยมวิทยา —
# "+"=ค่า feature สูง ควรทำให้ residual (และ pm25) ทายสูงขึ้น, "-"=ตรงข้าม
METEOROLOGICAL_HYPOTHESES = {
    "precipitation": ("-", "ฝนตกชะล้าง PM2.5 ออกจากอากาศ (scavenging effect) — ฝนมาก ควรทายฝุ่นลดลง"),
    "pressure": ("+", "ความกดอากาศสูงมักสัมพันธ์กับ temperature inversion ที่กดฝุ่นไว้ใกล้พื้น — ควรทายฝุ่นสูงขึ้น"),
    "wind_speed": ("-", "ลมแรงช่วยกระจาย/เจือจางฝุ่น (dispersion) — ควรทายฝุ่นลดลง"),
    "nb_trend_12h": ("+", "PM2.5 จังหวัดข้างเคียงกำลังเพิ่มขึ้น (แนวโน้ม 12 ชม.) — กลุ่มฝุ่นกำลังเคลื่อนเข้า ควรทายฝุ่นสูงขึ้น"),
    "fire_hotspot_count_100km_24h": ("+", "จุดความร้อน (การเผา) ในรัศมี 100 กม. ย้อนหลัง 24 ชม. มาก — ควรทายฝุ่นสูงขึ้น"),
    "humidity": ("+", "ความชื้นสูง (โดยเฉพาะอากาศนิ่ง ไม่ใช่ฝนตก) ทำให้ฝุ่นดูดความชื้นขยายตัว/หนักขึ้น "
                       "(hygroscopic growth) และมักเกิดร่วมกับ temperature inversion ตอนเช้าหน้าหนาว — "
                       "ควรทายฝุ่นสูงขึ้น (เป็น feature ที่อาจารย์ระบุเจาะจงเรื่องหน้าหนาว จึงเช็คแยกฤดูเพิ่มด้านล่าง)"),
}
WINTER_MONTHS = {11, 12, 1, 2}  # หน้าหนาวไทย (พ.ย.-ก.พ.)
WINTER_STRATIFIED_FEATURES = ["humidity"]

with open(LGB_MODEL_PATH, "rb") as f:
    _lgb_package = pickle.load(f)
with open(XGB_MODEL_PATH, "rb") as f:
    _xgb_package = pickle.load(f)

FEATURE_COLS = _lgb_package["feature_cols"]
assert FEATURE_COLS == _xgb_package["feature_cols"], \
    "feature_cols ของ LightGBM กับ XGBoost ต้องตรงกันเป๊ะ (เทรนจาก FEATURE_COLS เดียวกันใน train_multiprovince_models.py)"

MODEL_PACKAGES = {"lightgbm_pooled_v3": _lgb_package, "xgboost_pooled_v3": _xgb_package}

# cache explainer ต่อ (model_name, horizon) เพราะสร้าง TreeExplainer ใหม่ทุกครั้งไม่จำเป็น
_EXPLAINER_CACHE = {}


def _load_test_split():
    """reuse split_train_calib_test() จาก train_multiprovince_models.py เพื่อได้ test rows
    ชุดเดียวกับที่ใช้ training/evaluation อื่นๆ"""
    spec = importlib.util.spec_from_file_location(
        "train_multiprovince_models", os.path.join(_ROOT, "scripts", "train_multiprovince_models.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    df = pd.read_csv(DATA_PATH)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    df = df[~df["is_zero_shot"]].reset_index(drop=True)
    _, _, test_df = mod.split_train_calib_test(df)
    return test_df.iloc[::SUBSAMPLE_STRIDE].reset_index(drop=True)


def get_shap_explanation(model_used, horizon, feature_row):
    """คืนอิทธิพลของแต่ละ feature ต่อค่าพยากรณ์ P50 ณ จุดเดียว (feature_row) โดยใช้ SHAP explainer
    ของโมเดลที่ Dynamic Routing เลือกใช้จริง (model_used จาก select_model() ใน app.py:
    "lightgbm_pooled_v3"/"xgboost_pooled_v3"/"naive_fallback")

    feature_row: DataFrame แถวเดียว มีคอลัมน์ตรงกับ FEATURE_COLS (รูปแบบเดียวกับที่
    build_feature_row() ใน app.py คืนกลับมา)

    คืนค่า: list of (feature_name, shap_value) เรียงจากอิทธิพลสูงสุด (|shap_value|) ไปต่ำสุด
    (shap_value อธิบายผลต่อ residual ไม่ใช่ pm25 สัมบูรณ์) หรือ None ถ้า model_used ==
    "naive_fallback" (ไม่มีโมเดล quantile ให้อธิบาย เป็นแค่ persistence)
    """
    if model_used == "naive_fallback":
        return None

    if model_used not in MODEL_PACKAGES:
        raise ValueError(f"ไม่รู้จักโมเดล: {model_used}")

    cache_key = (model_used, horizon)
    if cache_key not in _EXPLAINER_CACHE:
        model = MODEL_PACKAGES[model_used]["models"][horizon][EXPLAIN_QUANTILE]
        _EXPLAINER_CACHE[cache_key] = shap.TreeExplainer(model)
    explainer = _EXPLAINER_CACHE[cache_key]

    shap_row = explainer.shap_values(feature_row[FEATURE_COLS])[0]
    pairs = list(zip(FEATURE_COLS, shap_row))
    pairs.sort(key=lambda p: abs(p[1]), reverse=True)
    return pairs


def _meteorological_plausibility_check(model_name, h, X_test, shap_values, datetime_series=None):
    """เช็คว่าทิศทางที่โมเดลเรียนรู้จริง (correlation ระหว่างค่า feature กับ SHAP value ของมัน)
    ตรงกับสมมติฐานเชิงฟิสิกส์ที่ตั้งไว้ล่วงหน้าไหม + เช็คแยกหน้าหนาวสำหรับ feature ที่ระบุใน
    WINTER_STRATIFIED_FEATURES"""
    print(f"\n  -- meteorological plausibility check ({model_name} t+{h}) --")
    for feat, (expected_sign, reason) in METEOROLOGICAL_HYPOTHESES.items():
        idx = FEATURE_COLS.index(feat)
        feat_values = X_test[feat].values
        feat_shap = shap_values[:, idx]
        if np.nanstd(feat_values) == 0:
            print(f"    {feat}: ข้ามเช็ค (ค่าคงที่ในชุดทดสอบนี้)")
            continue
        valid = ~np.isnan(feat_values)  # nb_trend_12h มี NaN ~0.4% (12 ชม.แรกของแต่ละจังหวัด) — corrcoef ต้องตัดแถวที่ไม่มีค่า
        corr = np.corrcoef(feat_values[valid], feat_shap[valid])[0, 1]
        observed_sign = "+" if corr > 0 else "-"
        verdict = "ตรงสมมติฐาน" if observed_sign == expected_sign else "ขัดสมมติฐาน"
        print(f"    {feat}: corr(value, shap)={corr:+.3f} (คาด {expected_sign}, ได้ {observed_sign}) -> {verdict}")
        print(f"      เหตุผล: {reason}")

        if feat in WINTER_STRATIFIED_FEATURES and datetime_series is not None:
            months = pd.to_datetime(datetime_series).dt.month.values
            is_winter = np.isin(months, list(WINTER_MONTHS))
            for label, mask in (("หน้าหนาว (พ.ย.-ก.พ.)", is_winter), ("นอกหน้าหนาว", ~is_winter)):
                if mask.sum() < 30 or np.std(feat_values[mask]) == 0:
                    print(f"      [{label}] ข้ามเช็ค (ข้อมูลน้อยเกินไปหรือค่าคงที่, n={mask.sum()})")
                    continue
                corr_season = np.corrcoef(feat_values[mask], feat_shap[mask])[0, 1]
                sign_season = "+" if corr_season > 0 else "-"
                verdict_season = "ตรงสมมติฐาน" if sign_season == expected_sign else "ขัดสมมติฐาน"
                print(f"      [{label}, n={mask.sum()}] corr={corr_season:+.3f} -> {verdict_season}")


def _run_batch_analysis():
    test = _load_test_split()
    print(f"test rows (split เดียวกับ task A/D): {len(test)}")

    shap_values_by_horizon = {"lightgbm_pooled_v3": {}, "xgboost_pooled_v3": {}}
    top_features_rows = []

    for model_name, package in MODEL_PACKAGES.items():
        models = package["models"]
        for h in REP_HORIZONS:
            target = f"pm25_t+{h}"
            test_valid = test.dropna(subset=[target])
            X_test = test_valid[FEATURE_COLS]

            model = models[h][EXPLAIN_QUANTILE]
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_test)

            shap_values_by_horizon[model_name][h] = shap_values

            mean_abs_shap = np.abs(shap_values).mean(axis=0)
            order = np.argsort(mean_abs_shap)[::-1][:TOP_N]
            print(f"\n=== {model_name} t+{h} (n={len(X_test)}, อธิบาย residual pm25_t+{h}-pm25_now, P50) top {TOP_N} features ===")
            for rank, idx in enumerate(order, start=1):
                feat = FEATURE_COLS[idx]
                val = mean_abs_shap[idx]
                print(f"  {rank}. {feat}: mean|SHAP|={val:.4f}")
                top_features_rows.append({
                    "model_name": model_name, "horizon": h, "rank": rank,
                    "feature": feat, "mean_abs_shap": val,
                })

            _meteorological_plausibility_check(model_name, h, X_test, shap_values, test_valid["datetime_utc"])

    with open(OUT_PKL_PATH, "wb") as f:
        pickle.dump({"feature_cols": FEATURE_COLS, "shap_values": shap_values_by_horizon}, f)

    top_df = pd.DataFrame(top_features_rows)
    top_df.to_csv(OUT_CSV_PATH, index=False)

    print(f"\nบันทึก SHAP values ที่: {OUT_PKL_PATH}")
    print(f"บันทึกสรุป top features ที่: {OUT_CSV_PATH}")

    print("\n=== เทียบ top features LightGBM vs XGBoost ต่อ horizon ===")
    for h in REP_HORIZONS:
        lgb_top = set(top_df[(top_df.model_name == "lightgbm_pooled_v3") & (top_df.horizon == h)]["feature"])
        xgb_top = set(top_df[(top_df.model_name == "xgboost_pooled_v3") & (top_df.horizon == h)]["feature"])
        overlap = lgb_top & xgb_top
        print(f"\nt+{h}: overlap {len(overlap)}/{TOP_N} -> {sorted(overlap)}")
        print(f"   LightGBM only: {sorted(lgb_top - xgb_top)}")
        print(f"   XGBoost only:  {sorted(xgb_top - lgb_top)}")

    # self-test: get_shap_explanation() ต้องให้ค่าเรียงลำดับตรงกับ mean_abs_shap ของ batch
    print("\n=== self-test: get_shap_explanation() เทียบ naive_fallback -> None ===")
    sample_row = test.dropna(subset=["pm25_t+1"])[FEATURE_COLS].iloc[[-1]]
    for model_name in ("lightgbm_pooled_v3", "xgboost_pooled_v3"):
        pairs = get_shap_explanation(model_name, 1, sample_row)
        top3 = [f"{feat}={val:+.3f}" for feat, val in pairs[:3]]
        print(f"  {model_name} t+1 (จุดตัวอย่างสุดท้ายใน test set): top3 = {top3}")
    fallback_result = get_shap_explanation("naive_fallback", 1, sample_row)
    print(f"  naive_fallback -> {fallback_result} (คาด None)")
    assert fallback_result is None, "get_shap_explanation('naive_fallback', ...) ต้องคืน None"


if __name__ == "__main__":
    _run_batch_analysis()

"""
Phase 2b — รายละเอียดวิธีทดลอง zero-shot transfer ดู CHANGELOG (75) ข้อ 5

สคริปต์นี้ใช้ P50-only (ไม่มี quantile P05/P95 + conformal เหมือน train_multiprovince_models.py
เพราะการทดลองนี้วัดแค่ MAE/R² ของ zero-shot transfer) แต่ยังคง residual forecasting + bagging
(subsample/colsample) เหมือนโมเดล deploy จริง
"""
import os

import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, r2_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(_ROOT, "data", "multiprovince_pm25_features_raw_20260913_012629.csv")
MODEL_DIR = os.path.join(_ROOT, "models")

HELD_OUT = ["udonthani", "ubonratchathani", "chaiyaphum"]
N_ESTIMATORS = 300
BAGGING_PARAMS_LGB = dict(subsample=0.8, subsample_freq=1, colsample_bytree=0.8)
BAGGING_PARAMS_XGB = dict(subsample=0.8, colsample_bytree=0.8)

FEATURE_COLS = [
    "temp", "humidity", "wind_speed", "precipitation",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "dayofweek",
    "pm25_lag_1", "pm25_lag_3", "pm25_lag_6", "pm25_lag_12", "pm25_lag_24",
    "pm25_roll_mean_6", "pm25_roll_std_6", "pm25_roll_mean_24", "pm25_roll_std_24",
    "wind_dir_sin", "wind_dir_cos", "pressure", "lat", "lon",
]


def split_train_test(df, train_frac=0.65 + 0.15):
    """เหมือน split_train_calib_test ใน train_multiprovince_models.py แต่รวม calibration เข้ากับ
    train เป็นก้อนเดียว (การทดลองนี้ไม่ต้องใช้ calibration split) ให้ test portion (20% ท้าย)
    ตรงกันเป๊ะกับของโมเดล deploy จริง เพราะ split ต่อจังหวัดแบบ chronological เหมือนกัน"""
    trains, tests = [], []
    for _, g in df.groupby("province_en", sort=False):
        g = g.sort_values("datetime_utc").reset_index(drop=True)
        n_train = int(len(g) * train_frac)
        trains.append(g.iloc[:n_train])
        tests.append(g.iloc[n_train:])
    return pd.concat(trains, ignore_index=True), pd.concat(tests, ignore_index=True)


def fit_p50(X, y, kind):
    if kind == "lgb":
        model = lgb.LGBMRegressor(objective="quantile", alpha=0.5, n_estimators=N_ESTIMATORS,
                                   random_state=42, verbose=-1, **BAGGING_PARAMS_LGB)
    else:
        model = xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, tree_method="hist",
                                  n_estimators=N_ESTIMATORS, random_state=42, verbosity=0, **BAGGING_PARAMS_XGB)
    model.fit(X, y)
    return model


def train_and_eval(train_df, eval_df, kind, label):
    per_horizon = []
    for h in range(1, 25):
        target_col = f"pm25_t+{h}"
        tr = train_df.dropna(subset=[target_col])
        ev = eval_df.dropna(subset=[target_col])
        if len(ev) == 0:
            continue
        X_tr, y_tr_resid = tr[FEATURE_COLS], tr[target_col] - tr["pm25"]
        X_ev, current_ev = ev[FEATURE_COLS], ev["pm25"].values
        y_ev_actual = ev[target_col].values

        model = fit_p50(X_tr, y_tr_resid, kind)
        pred = model.predict(X_ev) + current_ev
        mae = mean_absolute_error(y_ev_actual, pred)
        r2 = r2_score(y_ev_actual, pred)
        per_horizon.append({"horizon": h, "mae": mae, "r2": r2, "n_eval": len(ev)})
    df_h = pd.DataFrame(per_horizon)
    print(f"  [{label}/{kind}] MAE เฉลี่ย 24h = {df_h['mae'].mean():.3f}  R2 เฉลี่ย 24h = {df_h['r2'].mean():.3f}")
    return df_h


def main():
    print(f"โหลดข้อมูล: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    df = df[~df["is_zero_shot"]].reset_index(drop=True)  # ตัด 3 จังหวัด zero-shot จริงออกจาก pool เสมอ

    in_training_pool = df  # 17 จังหวัด รวม held-out
    zero_shot_pool = df[~df["province_en"].isin(HELD_OUT)].reset_index(drop=True)  # 14 จังหวัด ไม่รวม held-out เลย

    print(f"in_training_pool: {in_training_pool['province_en'].nunique()} จังหวัด, {len(in_training_pool)} แถว")
    print(f"zero_shot_pool  : {zero_shot_pool['province_en'].nunique()} จังหวัด, {len(zero_shot_pool)} แถว")

    train_in, test_in = split_train_test(in_training_pool)
    train_zs, test_zs_pool = split_train_test(zero_shot_pool)

    # test rows ของ 3 จังหวัด held-out ต้องเหมือนกันทั้งสองฝั่ง (split ต่อจังหวัดแบบ chronological
    # ไม่ขึ้นกับว่าจังหวัดอื่นถูกรวม/ตัดออก) -> ยืนยันด้วย assert ก่อนประเมินจริง
    held_out_test_in = test_in[test_in["province_en"].isin(HELD_OUT)].reset_index(drop=True)
    held_out_only = df[df["province_en"].isin(HELD_OUT)].reset_index(drop=True)
    _, held_out_test_direct = split_train_test(held_out_only)
    assert len(held_out_test_in) == len(held_out_test_direct), \
        "test rows ของจังหวัด held-out ควรเท่ากันไม่ว่าจะ split จาก pool ไหน (per-province chronological split)"

    print(f"\ntest rows ของ 3 จังหวัด held-out: {len(held_out_test_in)}")

    results = {}
    for kind in ("lgb", "xgb"):
        print(f"\n=== {kind.upper()} ===")
        results[(kind, "in_training")] = train_and_eval(train_in, held_out_test_in, kind, "in_training")
        results[(kind, "zero_shot")] = train_and_eval(train_zs, held_out_test_direct, kind, "zero_shot")

    summary = []
    for kind in ("lgb", "xgb"):
        for label in ("in_training", "zero_shot"):
            d = results[(kind, label)]
            summary.append({"model": kind, "scenario": label,
                             "mae_mean_24h": d["mae"].mean(), "r2_mean_24h": d["r2"].mean()})
    summary_df = pd.DataFrame(summary)

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(MODEL_DIR, f"zeroshot_eval_{timestamp}.csv")
    summary_df.to_csv(out_path, index=False)

    print("\n" + "=" * 70)
    print(summary_df.to_string(index=False))
    print(f"\nบันทึกแล้วที่: {out_path}")
    print("\nเทียบกับตัวเลขที่ผู้ใช้รายงานไว้ (การทดลองต้นฉบับ):")
    print("  in-training MAE=3.278 (ไม่ระบุโมเดล)")
    print("  zero-shot   MAE=3.375, R2=0.912 (ไม่ระบุโมเดล)")


if __name__ == "__main__":
    main()

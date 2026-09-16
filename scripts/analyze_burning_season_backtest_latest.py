"""
ทำ task A (backtest ฤดูฝุ่นสูง ก.พ.-เม.ย. เทียบช่วงปกติ) ซ้ำบน**โมเดลล่าสุด**
(_20260914_211416, 19 feature: spatial-lag + ตัด time_cyclical) — ของเดิม
(analyze_burning_season_backtest.py, CHANGELOG 89) รันบนโมเดลเก่าที่ deploy อยู่ตอนนั้น
(_20260913_014901) เท่านั้น ยังไม่เคยเช็คว่าโมเดลใหม่ที่มีแนวโน้มจะ deploy จริง มีปัญหาเดียวกัน
หรือดีขึ้นแค่ไหน — ตรรกะเดียวกันเป๊ะ เปลี่ยนแค่ MODEL_TAG/DATA_PATH ให้ตรงกับโมเดลใหม่
"""
import importlib.util
import os
import pickle

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(_ROOT, "models")
MODEL_TAG = "20260914_211416"

BURNING_MONTHS = {2, 3, 4}  # ก.พ.-เม.ย. — ฤดูเผาไร่/หมอกควันภาคอีสาน


def load_train_module():
    spec = importlib.util.spec_from_file_location(
        "train_multiprovince_models", os.path.join(_ROOT, "scripts", "train_multiprovince_models.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_model_pkg(tag):
    path = os.path.join(MODEL_DIR, f"{tag}_pm25_multiprovince_residual_bagged_quantile_{MODEL_TAG}.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_all_horizons(pkg, test_df, feature_cols):
    """คืน DataFrame ยาว (1 แถวต่อ 1 (row, horizon)) พร้อม predicted/actual/lo/hi"""
    q_hat_by_h = {row["horizon"]: row["q_hat"] for row in pkg["eval_by_horizon"]}
    out_rows = []
    for h in range(1, 25):
        target_col = f"pm25_t+{h}"
        sub = test_df.dropna(subset=[target_col])
        if sub.empty:
            continue
        X = sub[feature_cols]
        current = sub["pm25"].values
        models_by_q = pkg["models"][h]

        p50 = models_by_q[0.5].predict(X) + current
        lo = models_by_q[0.05].predict(X) + current
        hi = models_by_q[0.95].predict(X) + current
        q_hat = q_hat_by_h[h]
        lo_adj, hi_adj = lo - q_hat, hi + q_hat

        actual = sub[target_col].values
        frame = pd.DataFrame({
            "province_en": sub["province_en"].values,
            "datetime_utc": sub["datetime_utc"].values,
            "horizon": h,
            "predicted": p50,
            "actual": actual,
            "pm25_lo": lo_adj,
            "pm25_hi": hi_adj,
        })
        out_rows.append(frame)
    return pd.concat(out_rows, ignore_index=True)


def main():
    train_mod = load_train_module()
    print(f"โหลดข้อมูล: {train_mod.DATA_PATH}")
    df = pd.read_csv(train_mod.DATA_PATH)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    df = df[~df["is_zero_shot"]].reset_index(drop=True)

    _, _, test_df = train_mod.split_train_calib_test(df)
    print(f"test rows (17 จังหวัดเทรน, split เดียวกับตอนเทรนโมเดลนี้): {len(test_df)}")
    print(f"ช่วงเวลาของ test split: {test_df['datetime_utc'].min()} ถึง {test_df['datetime_utc'].max()}")

    results = {}
    for tag in ("lgb", "xgb"):
        pkg = load_model_pkg(tag)
        results[tag] = predict_all_horizons(pkg, test_df, pkg["feature_cols"])
        print(f"{tag}: predict เสร็จ {len(results[tag])} แถว (province x horizon x test-row)")

    for tag, res in results.items():
        res = res.copy()
        res["abs_error"] = (res["predicted"] - res["actual"]).abs()
        res["covered"] = (res["actual"] >= res["pm25_lo"]) & (res["actual"] <= res["pm25_hi"])
        res["month"] = pd.to_datetime(res["datetime_utc"]).dt.month
        res["is_burning_season"] = res["month"].isin(BURNING_MONTHS)

        print("\n" + "=" * 78)
        print(f"{tag.upper()} — ฤดูฝุ่น (ก.พ.-เม.ย.) vs ช่วงปกติ [โมเดลล่าสุด {MODEL_TAG}]")
        print("=" * 78)
        grp = res.groupby("is_burning_season")["abs_error"].agg(["mean", "median", "count"])
        cov = res.groupby("is_burning_season")["covered"].mean()
        grp["coverage_pct"] = (cov * 100).round(1)
        print(grp.rename(index={False: "ปกติ", True: "ฤดูฝุ่น (ก.พ.-เม.ย.)"}))

        if False in grp.index and True in grp.index and grp.loc[False, "mean"] > 0:
            pct = (grp.loc[True, "mean"] - grp.loc[False, "mean"]) / grp.loc[False, "mean"] * 100
            print(f"MAE ช่วงฤดูฝุ่นสูงกว่าช่วงปกติ {pct:.1f}%")

        bins = [-0.01, res["actual"].quantile(0.5), res["actual"].quantile(0.75),
                res["actual"].quantile(0.9), res["actual"].max() + 0.01]
        labels = ["<=p50", "p50-p75", "p75-p90", ">p90"]
        res["severity_bin"] = pd.cut(res["actual"], bins=bins, labels=labels)
        print("\nMAE/coverage ตาม severity bin ของ actual PM2.5:")
        grp2 = res.groupby("severity_bin", observed=True)["abs_error"].agg(["mean", "median", "count"])
        cov2 = res.groupby("severity_bin", observed=True)["covered"].mean()
        grp2["coverage_pct"] = (cov2 * 100).round(1)
        print(grp2)

        h24 = res[res["horizon"] == 24]
        if not h24.empty:
            g24 = h24.groupby("is_burning_season")["abs_error"].mean()
            print(f"\nMAE เฉพาะ t+24 — ปกติ={g24.get(False, float('nan')):.3f}  "
                  f"ฤดูฝุ่น={g24.get(True, float('nan')):.3f}")


if __name__ == "__main__":
    main()

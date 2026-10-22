"""
คำนวณ Accuracy = 100% - MAPE% (MAPE = Mean Absolute Percentage Error) บน test split เดียวกับที่
รายงานผลทางการ แยกตามช่วงปกติ/ฤดูฝุ่น และต่อ horizon ตัวแทน

MAPE ไม่เสถียรเมื่อค่าจริงใกล้ 0 จึงรายงานทั้งแบบรวมทุกแถว และแบบตัดแถวที่ actual < 5 ug/m3 ออก
"""
import importlib.util
import os

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOW_ACTUAL_THRESHOLD = 5.0  # ug/m3 — ต่ำกว่านี้ %error ไม่เสถียร


def load_backtest_module():
    spec = importlib.util.spec_from_file_location(
        "analyze_burning_season_backtest_clean",
        os.path.join(_ROOT, "scripts", "analyze_burning_season_backtest_clean.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def mape_accuracy(actual, predicted):
    ape = (actual - predicted).abs() / actual.abs() * 100
    mape = ape.mean()
    return mape, 100 - mape


def main():
    bt = load_backtest_module()
    train_mod = bt.load_train_module()
    print(f"โหลดข้อมูล: {train_mod.DATA_PATH}")
    df = pd.read_csv(train_mod.DATA_PATH)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    df = df[~df["is_zero_shot"]].reset_index(drop=True)

    _, _, test_df = train_mod.split_train_calib_test(df)
    print(f"test rows: {len(test_df)}\n")

    results = {}
    for tag in ("lgb", "xgb"):
        pkg = bt.load_model_pkg(tag)
        results[tag] = bt.predict_all_horizons(pkg, test_df, pkg["feature_cols"])

    for tag, res in results.items():
        res = res.copy()
        res["month"] = pd.to_datetime(res["datetime_utc"]).dt.month
        res["is_burning_season"] = res["month"].isin({2, 3, 4})

        print("=" * 78)
        print(f"{tag.upper()} — Accuracy (= 100% - MAPE%)")
        print("=" * 78)

        mape_all, acc_all = mape_accuracy(res["actual"], res["predicted"])
        n_low = (res["actual"] < LOW_ACTUAL_THRESHOLD).sum()
        res_clean = res[res["actual"] >= LOW_ACTUAL_THRESHOLD]
        mape_clean, acc_clean = mape_accuracy(res_clean["actual"], res_clean["predicted"])

        print(f"รวมทุกแถว (n={len(res)}): MAPE={mape_all:.2f}%  -> Accuracy={acc_all:.2f}%")
        print(f"ตัดแถว actual<{LOW_ACTUAL_THRESHOLD} ug/m3 ออก (n={n_low} แถวที่ตัด, เหลือ {len(res_clean)}): "
              f"MAPE={mape_clean:.2f}%  -> Accuracy={acc_clean:.2f}%")

        print("\nAccuracy แยกตามช่วง:")
        for label, mask in [("ช่วงปกติ", ~res_clean["is_burning_season"]),
                             ("ฤดูฝุ่น (ก.พ.-เม.ย.)", res_clean["is_burning_season"])]:
            m, a = mape_accuracy(res_clean.loc[mask, "actual"], res_clean.loc[mask, "predicted"])
            print(f"  {label}: MAPE={m:.2f}%  Accuracy={a:.2f}%  (n={mask.sum()})")

        print("\nAccuracy ต่อ horizon ตัวแทน:")
        for h in (1, 6, 12, 18, 24):
            sub = res_clean[res_clean["horizon"] == h]
            m, a = mape_accuracy(sub["actual"], sub["predicted"])
            print(f"  t+{h:2d}: MAPE={m:.2f}%  Accuracy={a:.2f}%")
        print()


if __name__ == "__main__":
    main()

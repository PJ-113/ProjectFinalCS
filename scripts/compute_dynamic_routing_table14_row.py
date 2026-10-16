"""
เพิ่มแถว "Dynamic Routing" เข้าตารางที่ 14 ในเล่ม (ปัจจุบันมีแค่ LightGBM/XGBoost/Persistence)
ให้ scope ตรงกับตารางที่ 14 เป๊ะ: 17 จังหวัดที่ใช้ฝึกโมเดล (ไม่รวม 3 zero-shot), เฉลี่ยทั้ง 24
horizon ด้วย MAE/RMSE/R² — วิธีเฉลี่ยตามที่ train_multiprovince_models.py ใช้จริง (คำนวณ
MAE/RMSE/R² แยกต่อ horizon ก่อน แล้วค่อยเฉลี่ย 24 ค่านั้นรวมกัน ไม่ใช่ pool แถวทั้งหมดคำนวณทีเดียว)
เพื่อให้ตัวเลข LightGBM/XGBoost ที่ได้จากสคริปต์นี้ตรงกับ models/multiprovince_eval_rmse_
20260919_203455.csv (ต้นทางจริงของตารางที่ 14) เป็นการ sanity check methodology ไปในตัว

ใช้ simulate_province() ตัวเดียวกับ backtest_dynamic_routing_offline.py (select_model() ของจริง,
3 seed) ไม่ต้องรันใหม่จากศูนย์ แค่ import ฟังก์ชันมาใช้ซ้ำ แล้วเปลี่ยนการ aggregate: เดิมสคริปต์นั้น
กรองแค่ 5 horizon ตัวแทน (REPORT_HORIZONS) และไม่แยก zero-shot ออกตอนสรุปผลต่อ horizon — สคริปต์นี้
กรองเฉพาะ 17 จังหวัดฝึกสอน (is_zero_shot == False) และใช้ทั้ง 24 horizon ให้ตรง scope ตารางที่ 14

รัน: python scripts/compute_dynamic_routing_table14_row.py
"""
import importlib.util
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(_ROOT, "models")
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
SEEDS = [0, 1, 2]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, "scripts", filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    bt = load_module("backtest_dynamic_routing_offline", "backtest_dynamic_routing_offline.py")

    ens_mod = bt.load_module("eval_ensemble_vs_routing_clean", "eval_ensemble_vs_routing_clean.py")
    bt_mod = bt.load_module("analyze_burning_season_backtest_clean", "analyze_burning_season_backtest_clean.py")
    train_mod = bt_mod.load_train_module()

    print(f"โหลดข้อมูล: {train_mod.DATA_PATH}")
    df = pd.read_csv(train_mod.DATA_PATH)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True).dt.tz_localize(None)

    df_trained = df[~df["is_zero_shot"]].reset_index(drop=True)
    _, _, test_df = train_mod.split_train_calib_test(df_trained)
    df_zeroshot = df[df["is_zero_shot"]].reset_index(drop=True)
    all_test_df = pd.concat([test_df, df_zeroshot], ignore_index=True)
    print(f"held-out test rows: {len(test_df)} (17 จังหวัดเทรน) + {len(df_zeroshot)} (zero-shot) = {len(all_test_df)}")

    print("predict ดิบ (P50) บน test ทั้งหมด จาก pkl v3 ที่ deploy จริง (ไม่ retrain)...")
    merged = bt.build_merged_predictions(ens_mod, bt_mod, all_test_df)

    print("โหลด app.py (select_model/compute_missing_ratio ของจริง) ผ่าน importlib...")
    app = bt.load_app()
    model_lgb, model_xgb, model_naive = app.MODEL_LGB, app.MODEL_XGB, app.MODEL_NAIVE

    bt.MIN_PAIRS = app.MIN_R2_PAIRS
    print(f"MIN_PAIRS = app.MIN_R2_PAIRS = {bt.MIN_PAIRS}")

    provinces = sorted(merged["province_en"].unique())
    print(f"\nรัน simulation (ทุก horizon, ไม่กรองแค่ 5 ตัวแทน): {len(provinces)} จังหวัด x {len(SEEDS)} seed ...")
    sim_frames = []
    for seed in SEEDS:
        for province in provinces:
            df_prov = merged[merged["province_en"] == province]
            sim_frames.append(bt.simulate_province(province, df_prov, app, model_lgb, model_xgb, model_naive, seed))
        print(f"  seed={seed} เสร็จ")
    sim_df = pd.concat(sim_frames, ignore_index=True)

    # กรองเฉพาะ 17 จังหวัดฝึกสอน ให้ scope ตรงตารางที่ 14
    sim17 = sim_df[~sim_df["is_zero_shot"]].reset_index(drop=True)
    print(f"\nsim rows ทั้งหมด = {len(sim_df)}, เฉพาะ 17 จังหวัดฝึกสอน = {len(sim17)}")

    # เฉลี่ยต่อ horizon ก่อน (เหมือน train_multiprovince_models.py) แล้วค่อยเฉลี่ย 24 horizon รวมกัน
    # คำนวณแยกต่อ seed ก่อน แล้วเฉลี่ยข้าม seed ด้วย (ระบบมีสุ่ม cold-start/epsilon-greedy)
    per_seed_horizon = []
    for seed in SEEDS:
        g_seed = sim17[sim17["seed"] == seed]
        for h in range(1, 25):
            gh = g_seed[g_seed["horizon"] == h]
            if len(gh) < 2:
                continue
            mae = mean_absolute_error(gh["actual"], gh["predicted"])
            rmse = mean_squared_error(gh["actual"], gh["predicted"]) ** 0.5
            r2 = r2_score(gh["actual"], gh["predicted"])
            per_seed_horizon.append({"seed": seed, "horizon": h, "mae": mae, "rmse": rmse, "r2": r2, "n": len(gh)})
    psh = pd.DataFrame(per_seed_horizon)

    # เฉลี่ย 24 horizon รวมกัน ต่อ seed ก่อน (ให้ทุก horizon น้ำหนักเท่ากัน เหมือนตัวเลขต้นทางตารางที่ 14)
    per_seed_mean = psh.groupby("seed")[["mae", "rmse", "r2"]].mean()
    print("\nต่อ seed (เฉลี่ยทั้ง 24 horizon, 17 จังหวัดฝึกสอน):")
    print(per_seed_mean.round(4).to_string())

    final_mean = per_seed_mean.mean()
    final_std = per_seed_mean.std()

    print("\n" + "=" * 70)
    print("แถว Dynamic Routing สำหรับตารางที่ 14 (เฉลี่ยทั้ง 3 seed):")
    print("=" * 70)
    print(f"MAE  = {final_mean['mae']:.3f}  (std ข้าม seed = {final_std['mae']:.4f})")
    print(f"RMSE = {final_mean['rmse']:.3f}  (std ข้าม seed = {final_std['rmse']:.4f})")
    print(f"R2   = {final_mean['r2']:.3f}  (std ข้าม seed = {final_std['r2']:.4f})")

    print("\nเทียบกับตัวเลขต้นทาง LightGBM/XGBoost ของตารางที่ 14 (models/multiprovince_eval_rmse_20260919_203455.csv):")
    print("  LightGBM: MAE=2.780 RMSE=4.318 R2=0.939")
    print("  XGBoost : MAE=2.778 RMSE=4.319 R2=0.938")
    print("  (Dynamic Routing ควรอยู่ระหว่างสองค่านี้ หรือใกล้เคียงมาก เพราะเลือกจากสองโมเดลนี้เท่านั้น)")

    out_path = os.path.join(MODEL_DIR, f"dynamic_routing_table14_row_{TIMESTAMP}.csv")
    psh.to_csv(out_path, index=False)
    print(f"\nบันทึกผลดิบ (ต่อ seed ต่อ horizon) ที่: {out_path}")


if __name__ == "__main__":
    main()

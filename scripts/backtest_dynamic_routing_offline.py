"""
Backtest Dynamic Routing (select_model() จาก app.py) แบบ online ทีละชั่วโมงตามลำดับเวลา บน
held-out test set ของโมเดล v3 จำลองแบบ partial-feedback: prediction_log เก็บผลเฉพาะโมเดลที่ถูก
เลือกในแต่ละครั้ง ไม่ใช่ log ทั้ง lgb และ xgb ทุกครั้ง

ต่อจังหวัด เดินเวลาไปข้างหน้าทีละ anchor time t:
  1. resolve prediction เก่าที่ target_time <= t เข้า rolling window (maxlen=24) ของ (model, horizon)
  2. คำนวณ r2_lookup จาก rolling window ปัจจุบัน แล้วเรียก app.select_model() ของจริง
  3. บันทึก prediction ของโมเดลที่ถูกเลือกไว้ resolve ที่ t+h ชั่วโมง

รันซ้ำ 3 seed (select_model() มีการสุ่มจาก cold-start/epsilon-greedy) รายงานเฉลี่ย ± ส่วนเบี่ยงเบน
มาตรฐาน วัด MAE/R2 เทียบ lgb_only/xgb_only/oracle_routing/persistence ต่อ horizon และต่อจังหวัด
"""
import heapq
import importlib.util
import os
import random

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(_ROOT, "models")
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
SEEDS = [0, 1, 2]
REPORT_HORIZONS = [1, 6, 12, 18, 24]
ROLLING_N = 24
# ต้องตรงกับ app.MIN_R2_PAIRS (set ใน main() หลังโหลด app.py จริง)
MIN_PAIRS = None


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, "scripts", filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_app():
    spec = importlib.util.spec_from_file_location("app_under_test", os.path.join(_ROOT, "app.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def build_merged_predictions(ens_mod, bt_mod, df_test_all):
    """p50 ของ lgb/xgb (ดิบ ไม่ conformal) + actual + pm25 (persistence "ตอนนี้") ต่อ (province,
    anchor time, horizon)"""
    pkgs = {tag: bt_mod.load_model_pkg(tag) for tag in ("lgb", "xgb")}
    preds = {tag: ens_mod.predict_raw_quantiles(pkgs[tag], df_test_all, pkgs[tag]["feature_cols"])
             for tag in ("lgb", "xgb")}
    merged = ens_mod.merged_frame(preds["lgb"], preds["xgb"])
    now_lookup = (df_test_all[["province_en", "datetime_utc", "pm25", "is_zero_shot"]]
                  .drop_duplicates(["province_en", "datetime_utc"]))
    merged = merged.merge(now_lookup, on=["province_en", "datetime_utc"], how="left")
    return merged


def rolling_r2(window):
    if len(window) < MIN_PAIRS:
        return None
    y_pred = [p for p, _ in window]
    y_true = [a for _, a in window]
    return r2_score(y_true, y_pred)


def simulate_province(province, df_prov, app, model_lgb, model_xgb, model_naive, seed):
    random.seed(f"{seed}-{province}")
    grouped = dict(tuple(df_prov.groupby("datetime_utc")))
    anchor_times = sorted(grouped.keys())

    windows = {}   # (model, horizon) -> list [(pred, actual), ...] maxlen=ROLLING_N
    pending = []   # heap of (resolve_time, horizon, model, pred, actual)
    dummy_row = pd.DataFrame([{c: 0.0 for c in app.FEATURE_COLS}])  # no NaN -> circuit breaker never fires

    # r2_cache: resolve ใหม่เฉพาะคู่ (model, horizon) ที่เพิ่งมีผลเข้ามาในรอบนี้
    r2_cache = {}
    out = []
    for t in anchor_times:
        while pending and pending[0][0] <= t:
            resolve_time, h, model, pred_val, actual_val = heapq.heappop(pending)
            w = windows.setdefault((model, h), [])
            w.append((pred_val, actual_val))
            if len(w) > ROLLING_N:
                w.pop(0)
            r2_cache[(model, h)] = rolling_r2(w)

        rows_t = grouped[t]
        horizons_here = rows_t["horizon"].unique()
        r2_lookup = {}
        for h in horizons_here:
            r2_lookup[(model_lgb, h)] = r2_cache.get((model_lgb, h))
            r2_lookup[(model_xgb, h)] = r2_cache.get((model_xgb, h))

        for row in rows_t.itertuples():
            h = int(row.horizon)
            model_name = app.select_model(dummy_row, h, province, r2_lookup, force_naive=False)
            if model_name == model_lgb:
                pred = row.p50_lgb
            elif model_name == model_xgb:
                pred = row.p50_xgb
            else:
                pred = row.pm25  # naive_fallback = persistence

            actual_val = float(row.actual)
            resolve_time = t + pd.Timedelta(hours=h)
            heapq.heappush(pending, (resolve_time, h, model_name, float(pred), actual_val))
            out.append({
                "province": province, "seed": seed, "anchor_time": t, "horizon": h,
                "model_chosen": model_name, "predicted": float(pred), "actual": actual_val,
                "persistence_pred": float(row.pm25), "is_zero_shot": bool(row.is_zero_shot),
            })
    return pd.DataFrame(out)


def compute_by_province(sim_df, model_lgb, model_xgb, model_naive):
    rows = []
    for (province, seed), g in sim_df.groupby(["province", "seed"]):
        g24 = g[g["horizon"] == 24]
        mae_h24 = (g24["predicted"] - g24["actual"]).abs().mean()
        r2_h24 = r2_score(g24["actual"], g24["predicted"]) if len(g24) >= 5 else np.nan
        mae_overall = (g["predicted"] - g["actual"]).abs().mean()
        persist_mae_h24 = (g24["persistence_pred"] - g24["actual"]).abs().mean()
        persist_mae_overall = (g["persistence_pred"] - g["actual"]).abs().mean()
        pct_h24 = (persist_mae_h24 - mae_h24) / persist_mae_h24 * 100 if persist_mae_h24 else np.nan
        pct_overall = (persist_mae_overall - mae_overall) / persist_mae_overall * 100 if persist_mae_overall else np.nan
        n = len(g24)
        rows.append({
            "province": province, "seed": seed, "is_zero_shot": bool(g["is_zero_shot"].iloc[0]),
            "n_test_rows_h24": n,
            "mae_h24": mae_h24, "r2_h24": r2_h24, "mae_overall": mae_overall,
            "persistence_mae_h24": persist_mae_h24, "persistence_mae_overall": persist_mae_overall,
            "pct_reduction_vs_persistence_h24": pct_h24, "pct_reduction_vs_persistence_overall": pct_overall,
            "model_share_lgb_h24": (g24["model_chosen"] == model_lgb).mean() if n else np.nan,
            "model_share_xgb_h24": (g24["model_chosen"] == model_xgb).mean() if n else np.nan,
            "model_share_naive_h24": (g24["model_chosen"] == model_naive).mean() if n else np.nan,
        })
    per_seed = pd.DataFrame(rows)

    agg_cols = ["mae_h24", "r2_h24", "mae_overall", "persistence_mae_h24", "persistence_mae_overall",
                "pct_reduction_vs_persistence_h24", "pct_reduction_vs_persistence_overall",
                "model_share_lgb_h24", "model_share_xgb_h24", "model_share_naive_h24"]
    summary = per_seed.groupby(["province", "is_zero_shot"])[agg_cols].agg(["mean", "std"])
    summary.columns = [f"{c}_{stat}" for c, stat in summary.columns]
    summary = summary.reset_index()
    n_rows = per_seed.groupby("province")["n_test_rows_h24"].first().rename("n_test_rows_h24")
    summary = summary.merge(n_rows, on="province")
    return summary


def compute_by_horizon(sim_df, merged, model_lgb, model_xgb):
    """เฉลี่ยทุกจังหวัดรวมกัน ต่อ horizon ตัวแทน — เทียบ Dynamic Routing (จากการจำลอง) กับ
    lgb_only/xgb_only/oracle_routing"""
    rows = []
    for h in REPORT_HORIZONS:
        sim_h = sim_df[sim_df["horizon"] == h]
        dr_mae_by_seed = sim_h.groupby("seed").apply(lambda g: (g["predicted"] - g["actual"]).abs().mean())

        pool_h = merged[merged["horizon"] == h]
        lgb_mae = (pool_h["p50_lgb"] - pool_h["actual"]).abs().mean()
        xgb_mae = (pool_h["p50_xgb"] - pool_h["actual"]).abs().mean()
        oracle_mae = min(lgb_mae, xgb_mae)  # perfect-hindsight per-horizon upper bound
        persist_mae = (pool_h["pm25"] - pool_h["actual"]).abs().mean()

        rows.append({
            "horizon": h,
            "dynamic_routing_mae_mean": dr_mae_by_seed.mean(), "dynamic_routing_mae_std": dr_mae_by_seed.std(),
            "lgb_only_mae": lgb_mae, "xgb_only_mae": xgb_mae,
            "oracle_routing_mae": oracle_mae, "persistence_mae": persist_mae,
        })
    return pd.DataFrame(rows)


def main():
    ens_mod = load_module("eval_ensemble_vs_routing_clean", "eval_ensemble_vs_routing_clean.py")
    bt_mod = load_module("analyze_burning_season_backtest_clean", "analyze_burning_season_backtest_clean.py")
    train_mod = bt_mod.load_train_module()

    print(f"โหลดข้อมูล: {train_mod.DATA_PATH}")
    df = pd.read_csv(train_mod.DATA_PATH)
    # normalize เป็น tz-naive ให้แน่นอน
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True).dt.tz_localize(None)

    df_trained = df[~df["is_zero_shot"]].reset_index(drop=True)
    _, _, test_df = train_mod.split_train_calib_test(df_trained)
    df_zeroshot = df[df["is_zero_shot"]].reset_index(drop=True)
    all_test_df = pd.concat([test_df, df_zeroshot], ignore_index=True)
    print(f"held-out test rows: {len(test_df)} (17 จังหวัดเทรน) + {len(df_zeroshot)} (zero-shot) = {len(all_test_df)}")

    print("predict ดิบ (P50) บน test ทั้งหมด จาก pkl v3 ที่ deploy จริง (ไม่ retrain)...")
    merged = build_merged_predictions(ens_mod, bt_mod, all_test_df)
    print(f"merged rows (province x anchor_time x horizon) = {len(merged)}")

    print("โหลด app.py (select_model/compute_missing_ratio ของจริง) ผ่าน importlib...")
    app = load_app()
    model_lgb, model_xgb, model_naive = app.MODEL_LGB, app.MODEL_XGB, app.MODEL_NAIVE
    print(f"MODEL_LGB={model_lgb}  MODEL_XGB={model_xgb}  MODEL_NAIVE={model_naive}")

    global MIN_PAIRS
    MIN_PAIRS = app.MIN_R2_PAIRS
    print(f"MIN_PAIRS = app.MIN_R2_PAIRS = {MIN_PAIRS}")

    provinces = sorted(merged["province_en"].unique())
    print(f"\nรัน simulation: {len(provinces)} จังหวัด x {len(SEEDS)} seed ...")
    sim_frames = []
    for seed in SEEDS:
        for province in provinces:
            df_prov = merged[merged["province_en"] == province]
            sim_frames.append(simulate_province(province, df_prov, app, model_lgb, model_xgb, model_naive, seed))
        print(f"  seed={seed} เสร็จ")
    sim_df = pd.concat(sim_frames, ignore_index=True)

    by_province = compute_by_province(sim_df, model_lgb, model_xgb, model_naive)
    by_horizon = compute_by_horizon(sim_df, merged, model_lgb, model_xgb)

    province_path = os.path.join(MODEL_DIR, f"dynamic_routing_offline_backtest_by_province_{TIMESTAMP}.csv")
    horizon_path = os.path.join(MODEL_DIR, f"dynamic_routing_offline_backtest_by_horizon_{TIMESTAMP}.csv")
    by_province.to_csv(province_path, index=False)
    by_horizon.to_csv(horizon_path, index=False)

    print("\n" + "=" * 90)
    print("ตารางต่อจังหวัด (MAE/R2 ที่ h=24, MAE รวมทุก horizon, %ลดจาก persistence):")
    print("=" * 90)
    show_cols = ["province", "is_zero_shot", "n_test_rows_h24", "mae_h24_mean", "mae_h24_std",
                 "r2_h24_mean", "mae_overall_mean", "pct_reduction_vs_persistence_h24_mean"]
    print(by_province[show_cols].round(3).to_string(index=False))

    print("\n" + "=" * 90)
    print("ตารางต่อ horizon (รวมทุกจังหวัด) — Dynamic Routing (จำลอง, ไม่ใช่ oracle) เทียบ lgb_only/xgb_only/oracle:")
    print("=" * 90)
    print(by_horizon.round(3).to_string(index=False))

    print("\nsanity check: oracle_routing_mae <= dynamic_routing_mae_mean <= max(lgb_only,xgb_only) ทุก horizon ควรเป็นจริง")
    for _, r in by_horizon.iterrows():
        best_single = min(r["lgb_only_mae"], r["xgb_only_mae"])
        worst_single = max(r["lgb_only_mae"], r["xgb_only_mae"])
        ok = r["oracle_routing_mae"] - 1e-9 <= r["dynamic_routing_mae_mean"] <= worst_single + 1e-9
        flag = "OK" if ok else "!! ผิดปกติ ตรวจ simulation !!"
        print(f"  h={int(r['horizon']):>2}  oracle={r['oracle_routing_mae']:.3f}  "
              f"dynamic_routing={r['dynamic_routing_mae_mean']:.3f}  best_single={best_single:.3f}  "
              f"worst_single={worst_single:.3f}  [{flag}]")

    print(f"\nบันทึกผลที่:\n  {province_path}\n  {horizon_path}")


if __name__ == "__main__":
    main()

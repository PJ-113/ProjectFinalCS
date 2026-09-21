"""
เทียบ ensemble (รวมพยากรณ์ LGB+XGB) กับ Dynamic Routing (เลือกโมเดลเดียวต่อ horizon/province
ตาม rolling R2 แบบ epsilon-greedy, select_model() ใน app.py) — ใช้โมเดล v3 ข้อมูลสะอาด ล่าสุด
(_20260919_203455, 20 feature, CHANGELOG 127) แทนโมเดล k=1 ก่อนหน้า (_20260917_001147,
19 feature) ที่ eval_ensemble_vs_routing_k1.py ใช้ — ไม่ต้อง retrain ใหม่ เพราะเป็นแค่ eval-time
ensembling บน prediction ที่ทำได้จาก pkg ที่เทรนไว้แล้ว ตรรกะเดียวกันเป๊ะ เปลี่ยนแค่ module
ที่ import (analyze_burning_season_backtest_clean แทน _k1) ทำให้ MODEL_TAG เปลี่ยนตามไปด้วย

**ข้อจำกัดสำคัญที่ต้องอ่านคู่กับผลลัพธ์เสมอ**: Dynamic Routing ตัวจริงเป็นระบบ online ที่ปรับ
ตาม rolling window (24 คู่ predict/actual ล่าสุด) ของผลจริงแยกตาม province และมี cold-start
exploration (EXPLORATION_RATE) — ไม่มีข้อมูลนี้ในการ evaluate แบบออฟไลน์บน train/calib/test split
เดียว ตัวเลข "oracle_routing" ในผลลัพธ์นี้คือ**เพดานบนแบบ perfect-hindsight**เท่านั้น (รู้ผลลัพธ์
จริงล่วงหน้าแล้วเลือกโมเดลที่ดีที่สุดต่อ horizon) — ไม่ใช่สิ่งที่ Dynamic Routing ตัวจริงทำได้ เพราะ
ตัวจริงต้องเรียนรู้จากอดีตที่ lag และมี noise จาก exploration ผลจริงของ Dynamic Routing แบบ live
น่าจะอยู่ระหว่าง lgb_only/xgb_only กับ oracle_routing ไม่ใช่เท่ากับ oracle_routing

ค่าถ่วงน้ำหนัก (weighted_avg) เลือกจาก calibration split (ไม่ใช่ test) เพื่อไม่ tune ทับ test set
เดียวกับที่ใช้รายงานผล — สอดคล้องกับหลักการ train/calib/test แยกกันเข้มงวดที่ใช้มาทั้งโปรเจกต์
"""
import importlib.util
import os

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(_ROOT, "models")
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
WEIGHT_GRID = [0.3, 0.4, 0.5, 0.6, 0.7]  # น้ำหนักของ lgb; xgb = 1-w
BURNING_MONTHS = {2, 3, 4}


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, "scripts", filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def predict_raw_quantiles(pkg, df_split, feature_cols):
    """เหมือน analyze_burning_season_backtest_clean.predict_all_horizons() แต่คืนค่า
    P05/P50/P95 แบบ**ดิบ ก่อน conformal adjustment** (เพื่อให้เอาไปรวม/ถ่วงน้ำหนักแล้ว
    conformal-calibrate ใหม่ทีเดียวตอน ensemble ไม่ใช่ adjust ซ้ำสองรอบ)"""
    out_rows = []
    for h in range(1, 25):
        target_col = f"pm25_t+{h}"
        sub = df_split.dropna(subset=[target_col])
        if sub.empty:
            continue
        X = sub[feature_cols]
        current = sub["pm25"].values
        models_by_q = pkg["models"][h]

        p05 = models_by_q[0.05].predict(X) + current
        p50 = models_by_q[0.5].predict(X) + current
        p95 = models_by_q[0.95].predict(X) + current
        actual = sub[target_col].values
        out_rows.append(pd.DataFrame({
            "province_en": sub["province_en"].values,
            "datetime_utc": sub["datetime_utc"].values,
            "horizon": h, "actual": actual, "p05": p05, "p50": p50, "p95": p95,
        }))
    return pd.concat(out_rows, ignore_index=True)


def merged_frame(lgb_pred, xgb_pred):
    m = lgb_pred.merge(xgb_pred, on=["province_en", "datetime_utc", "horizon", "actual"],
                        suffixes=("_lgb", "_xgb"))
    return m


def mae_by_horizon(pred, actual, horizon):
    return pd.DataFrame({"horizon": horizon, "abs_error": (pred - actual).abs()}) \
        .groupby("horizon")["abs_error"].mean()


def main():
    bt = load_module("analyze_burning_season_backtest_clean", "analyze_burning_season_backtest_clean.py")
    train_mod = bt.load_train_module()
    print(f"โหลดข้อมูล: {train_mod.DATA_PATH}")
    df = pd.read_csv(train_mod.DATA_PATH)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    df = df[~df["is_zero_shot"]].reset_index(drop=True)

    _, calib_df, test_df = train_mod.split_train_calib_test(df)
    print(f"calib={len(calib_df)}  test={len(test_df)}  (โมเดล v3 ข้อมูลสะอาด tag={bt.MODEL_TAG})")

    pkgs = {tag: bt.load_model_pkg(tag) for tag in ("lgb", "xgb")}

    print("\npredict ดิบ (P05/P50/P95, ยังไม่ conformal-adjust) บน calib และ test...")
    calib_preds = {tag: predict_raw_quantiles(pkgs[tag], calib_df, pkgs[tag]["feature_cols"]) for tag in ("lgb", "xgb")}
    test_preds = {tag: predict_raw_quantiles(pkgs[tag], test_df, pkgs[tag]["feature_cols"]) for tag in ("lgb", "xgb")}

    calib_m = merged_frame(calib_preds["lgb"], calib_preds["xgb"])
    test_m = merged_frame(test_preds["lgb"], test_preds["xgb"])
    print(f"calib merged rows={len(calib_m)}  test merged rows={len(test_m)}")

    # ---------- ระดับ P50: เทียบ MAE ของ variant ต่างๆ ----------
    print("\n" + "=" * 90)
    print("ส่วนที่ 1: MAE (P50) ต่อ variant — คำนวณบน test set เดียวกับที่รายงานผลทางการ")
    print("=" * 90)

    mae_per_horizon = {}
    mae_per_horizon["lgb_only"] = mae_by_horizon(test_m["p50_lgb"], test_m["actual"], test_m["horizon"])
    mae_per_horizon["xgb_only"] = mae_by_horizon(test_m["p50_xgb"], test_m["actual"], test_m["horizon"])
    mae_per_horizon["simple_avg"] = mae_by_horizon(
        (test_m["p50_lgb"] + test_m["p50_xgb"]) / 2, test_m["actual"], test_m["horizon"])

    # oracle_routing: ต่อ horizon เลือกโมเดลที่ MAE ต่ำกว่าจริงบน test (เพดานบนแบบ hindsight เท่านั้น)
    oracle_choice = {}
    for h in mae_per_horizon["lgb_only"].index:
        oracle_choice[h] = "lgb" if mae_per_horizon["lgb_only"][h] <= mae_per_horizon["xgb_only"][h] else "xgb"
    oracle_pred = np.where(test_m["horizon"].map(oracle_choice) == "lgb", test_m["p50_lgb"], test_m["p50_xgb"])
    mae_per_horizon["oracle_routing"] = mae_by_horizon(pd.Series(oracle_pred), test_m["actual"], test_m["horizon"])

    # weighted_avg: หาน้ำหนัก w ที่ดีที่สุดจาก CALIB (ไม่ใช่ test) แล้วค่อยประเมินผลจริงบน test
    calib_mae_by_w = {}
    for w in WEIGHT_GRID:
        pred_w = w * calib_m["p50_lgb"] + (1 - w) * calib_m["p50_xgb"]
        calib_mae_by_w[w] = (pred_w - calib_m["actual"]).abs().mean()
    best_w = min(calib_mae_by_w, key=calib_mae_by_w.get)
    print(f"\nเลือกน้ำหนัก w จาก calib split: {calib_mae_by_w}")
    print(f"  -> w ที่ดีที่สุดบน calib = {best_w} (lgb weight)")

    test_pred_best_w = best_w * test_m["p50_lgb"] + (1 - best_w) * test_m["p50_xgb"]
    mae_per_horizon[f"weighted_avg(w={best_w})"] = mae_by_horizon(test_pred_best_w, test_m["actual"], test_m["horizon"])

    mae_table = pd.DataFrame(mae_per_horizon)
    mae_table.to_csv(os.path.join(MODEL_DIR, f"ensemble_vs_routing_mae_by_horizon_{TIMESTAMP}.csv"))

    print("\nMAE เฉลี่ยรวมทุก horizon ต่อ variant:")
    overall = mae_table.mean().sort_values()
    for variant, mae in overall.items():
        print(f"  {variant:28s} MAE={mae:.4f}")

    print("\nMAE ที่ horizon ตัวแทน (t+1/6/12/18/24):")
    print(mae_table.loc[[1, 6, 12, 18, 24]].round(3))

    lgb_only_mae = overall["lgb_only"]
    best_single = min(overall["lgb_only"], overall["xgb_only"])
    for variant in ("simple_avg", f"weighted_avg(w={best_w})", "oracle_routing"):
        delta = (overall[variant] - best_single) / best_single * 100
        print(f"\n{variant}: {'ดีกว่า' if delta < 0 else 'แย่กว่า'} best-single-model อยู่ {delta:+.2f}%")

    # ---------- ระดับ P05/P95: conformal-calibrate ensemble ใหม่ แล้วเทียบ coverage ----------
    print("\n" + "=" * 90)
    print("ส่วนที่ 2: Coverage หลัง conformal calibration — รวม vs เฉพาะฤดูฝุ่น (ก.พ.-เม.ย.)")
    print("=" * 90)

    def build_ensemble_bounds(frame, weight):
        lo = weight * frame["p05_lgb"] + (1 - weight) * frame["p05_xgb"]
        hi = weight * frame["p95_lgb"] + (1 - weight) * frame["p95_xgb"]
        return lo, hi

    coverage_rows = []
    for label, weight in (("simple_avg (w=0.5)", 0.5), (f"weighted_avg (w={best_w})", best_w)):
        lo_calib, hi_calib = build_ensemble_bounds(calib_m, weight)
        lo_test, hi_test = build_ensemble_bounds(test_m, weight)

        q_hat = train_mod.conformalize(lo_calib.values, hi_calib.values, calib_m["actual"].values, alpha=train_mod.ALPHA)
        lo_adj = lo_test - q_hat
        hi_adj = hi_test + q_hat

        test_m_tmp = test_m.copy()
        test_m_tmp["month"] = pd.to_datetime(test_m_tmp["datetime_utc"]).dt.month
        test_m_tmp["is_burning"] = test_m_tmp["month"].isin(BURNING_MONTHS)
        test_m_tmp["covered"] = (test_m_tmp["actual"] >= lo_adj) & (test_m_tmp["actual"] <= hi_adj)

        cov_all = test_m_tmp["covered"].mean()
        cov_burning = test_m_tmp.loc[test_m_tmp["is_burning"], "covered"].mean()
        cov_normal = test_m_tmp.loc[~test_m_tmp["is_burning"], "covered"].mean()
        coverage_rows.append({"variant": label, "q_hat": q_hat, "coverage_all": cov_all,
                               "coverage_normal": cov_normal, "coverage_burning": cov_burning})
        print(f"\n{label}: q_hat={q_hat:.3f}")
        print(f"  coverage รวม   = {cov_all:.1%}  (เป้าหมาย 90%)")
        print(f"  coverage ปกติ  = {cov_normal:.1%}")
        print(f"  coverage ฤดุฝุ่น = {cov_burning:.1%}")

    print("\nเทียบกับ lgb/xgb เดี่ยว (จาก pkg เดิม, coverage_after เฉลี่ยข้าม horizon บน calib/test เดิม):")
    for tag in ("lgb", "xgb"):
        eval_rows = pkgs[tag]["eval_by_horizon"]
        cov_mean = np.mean([r["coverage_after"] for r in eval_rows])
        print(f"  {tag}_only: coverage_after เฉลี่ย (ข้าม horizon, ไม่แยกฤดู) = {cov_mean:.1%}")
    print("  (อ้างอิงเพิ่ม: CHANGELOG 128 พบ coverage เฉพาะช่วงฝุ่นวิกฤต >p90 บนโมเดล v3 ข้อมูลสะอาด นี้ "
          "= LGB 85.3% / XGB 79.5% เทียบ k=1 เดิม LGB 84.9% / XGB 80.8%)")

    coverage_df = pd.DataFrame(coverage_rows)
    coverage_df.to_csv(os.path.join(MODEL_DIR, f"ensemble_vs_routing_coverage_{TIMESTAMP}.csv"), index=False)
    print(f"\nบันทึกผล MAE ที่: models/ensemble_vs_routing_mae_by_horizon_{TIMESTAMP}.csv")
    print(f"บันทึกผล coverage ที่: models/ensemble_vs_routing_coverage_{TIMESTAMP}.csv")


if __name__ == "__main__":
    main()

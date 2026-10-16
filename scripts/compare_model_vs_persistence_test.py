"""
เปรียบเทียบโมเดล v3 (_20260919_203455) กับ persistence (=ค่า pm25 ณ เวลาพยากรณ์ คงที่ทุก horizon)
บนชุด test set จริง (chronological holdout, ไม่ใช่ live 2 วัน) แบ่งผลตาม horizon bucket / ฤดู / severity
เพื่อตอบคำถาม "โมเดลดีกว่า persistence แค่ไหนในแต่ละส่วน" อย่างครบและมีนัยสำคัญทางสถิติ (n ใหญ่)

reuse โครงจาก analyze_burning_season_backtest_clean.py (โหลดโมเดล, split, predict_all_horizons)
"""
import importlib.util
import os
import pickle

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_ROOT)
MODEL_DIR = os.path.join(_ROOT, "models")
MODEL_TAG = "20260919_203455"
BURNING_MONTHS = {2, 3, 4}
HORIZON_BUCKETS = [("t+1", 1, 1), ("t+2..6", 2, 6), ("t+7..12", 7, 12), ("t+13..18", 13, 18), ("t+19..24", 19, 24)]


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
    out_rows = []
    for h in range(1, 25):
        target_col = f"pm25_t+{h}"
        sub = test_df.dropna(subset=[target_col])
        if sub.empty:
            continue
        X = sub[feature_cols]
        current = sub["pm25"].values
        p50 = pkg["models"][h][0.5].predict(X) + current
        actual = sub[target_col].values
        frame = pd.DataFrame({
            "province_en": sub["province_en"].values,
            "datetime_utc": sub["datetime_utc"].values,
            "horizon": h,
            "predicted": p50,
            "persistence": current,
            "actual": actual,
        })
        out_rows.append(frame)
    return pd.concat(out_rows, ignore_index=True)


def summarize(res, group_col):
    g = res.groupby(group_col, observed=True)
    out = pd.DataFrame({
        "n": g.size(),
        "mae_model": g.apply(lambda d: (d.predicted - d.actual).abs().mean()),
        "mae_persist": g.apply(lambda d: (d.persistence - d.actual).abs().mean()),
    })
    out["improve_pct"] = (out.mae_persist - out.mae_model) / out.mae_persist * 100
    return out.round(3)


def main():
    train_mod = load_train_module()
    df = pd.read_csv(train_mod.DATA_PATH)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    df = df[~df["is_zero_shot"]].reset_index(drop=True)
    _, _, test_df = train_mod.split_train_calib_test(df)
    print(f"test rows: {len(test_df)}  ช่วง {test_df.datetime_utc.min()} - {test_df.datetime_utc.max()}")

    for tag in ("lgb", "xgb"):
        pkg = load_model_pkg(tag)
        res = predict_all_horizons(pkg, test_df, pkg["feature_cols"])
        res["month"] = pd.to_datetime(res["datetime_utc"]).dt.month
        res["is_burning"] = res["month"].isin(BURNING_MONTHS)
        res["hbucket"] = pd.cut(
            res["horizon"], bins=[0, 1, 6, 12, 18, 24],
            labels=["t+1", "t+2..6", "t+7..12", "t+13..18", "t+19..24"],
        )
        bins = [-0.01, res["actual"].quantile(0.5), res["actual"].quantile(0.75), res["actual"].quantile(0.9), res["actual"].max() + 0.01]
        res["severity"] = pd.cut(res["actual"], bins=bins, labels=["<=p50", "p50-p75", "p75-p90", ">p90"])

        print("\n" + "=" * 90)
        print(f"{tag.upper()} — โมเดล vs persistence บน test set จริง ({len(res)} แถว, n horizon x province x เวลา)")
        print("=" * 90)

        print("\n[ภาพรวม]")
        print(summarize(res, res.apply(lambda r: "รวม", axis=1)))

        print("\n[ตาม horizon]")
        print(summarize(res, "hbucket"))

        print("\n[ตามฤดู]")
        print(summarize(res.assign(season=res.is_burning.map({True: "ฤดูฝุ่น", False: "ปกติ"})), "season"))

        print("\n[ตาม severity ของค่าจริง]")
        print(summarize(res, "severity"))

        print("\n[ตามฤดู x horizon]")
        piv = res.assign(season=res.is_burning.map({True: "ฤดูฝุ่น", False: "ปกติ"}))
        print(summarize(piv, ["season", "hbucket"]))

        out_path = os.path.join(MODEL_DIR, f"compare_vs_persistence_{tag}_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv")
        res.to_csv(out_path, index=False)
        print(f"\nบันทึกผลดิบที่: {out_path}")


if __name__ == "__main__":
    main()

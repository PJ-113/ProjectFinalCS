"""
ตรวจก่อนตัด pm25_neighbor_avg ออกจากชุด v3 (CHANGELOG 149): ablation บน test (CHANGELOG 148) บอกว่าตัดแล้ว MAE ไม่เปลี่ยน (-0.04%/-0.11%) แต่การตัดสินจาก test
ไม่ใช่วิธีเลือก feature ที่ถูก จึงยืนยันด้วย rolling-origin CV เดียวกับ validate_features_timeblocked_cv.py (4 fold ครอบคลุมฤดูฝุ่น) ก่อนตัดจริง
เทียบ: 21 feature (ชุดปัจจุบัน) vs 20 feature (ตัด pm25_neighbor_avg, เก็บ nb_trend_12h + fire) ; horizon [1,12,24], LGB+XGB, hyperparameter ชุดเดียวกัน

เกณฑ์ (ตั้งไว้ก่อนรัน — non-inferiority): "ตัดได้" ต่อเมื่อทั้ง LGB และ XGB
  (ก) macro MAE ของ 20 ไม่แย่กว่า 21 เกิน +0.2%
  (ข) MAE เฉลี่ยของ fold ฤดูฝุ่น (A, C) ไม่แย่กว่าเกิน +0.5%
  (ค) MAE ที่ >p90 ของ fold ฤดูฝุ่นไม่แย่กว่าเกิน +1.0%
"""
import glob
import importlib.util
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR, MODEL_DIR = os.path.join(_ROOT, "data"), os.path.join(_ROOT, "models")
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
TOL_MACRO, TOL_BURN, TOL_BURN_P90 = 0.2, 0.5, 1.0


def load(name, fname):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, "scripts", fname))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    tm = load("tm", "train_multiprovince_models.py")
    cv = load("cv", "validate_features_timeblocked_cv.py")
    man = json.load(open(sorted(glob.glob(os.path.join(DATA_DIR, "rebuild_clean_manifest_*.json")))[-1], encoding="utf-8"))
    df = pd.read_csv(man["final"])
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    f21 = tm.FEATURE_COLS
    f20 = [c for c in f21 if c != "pm25_neighbor_avg"]
    variants = {"21": f21, "20 (ตัด neighbor_avg)": f20}
    d = df[~df["is_zero_shot"]].dropna(subset=f21).reset_index(drop=True)
    print(f"แถวที่ใช้ {len(d)}; 21 feature vs 20 feature (ตัด pm25_neighbor_avg)")

    rows = []
    for fname, v0, v1 in cv.FOLDS:
        vs, ve = pd.Timestamp(v0, tz="UTC"), pd.Timestamp(v1, tz="UTC") + pd.Timedelta(hours=23)
        tr_all = d[d["datetime_utc"] <= vs - pd.Timedelta(hours=24)]
        va_all = d[(d["datetime_utc"] >= vs) & (d["datetime_utc"] <= ve)]
        print(f"\n=== fold {fname}: train {len(tr_all)}, validate {len(va_all)} ===", flush=True)
        for kind in ("lgb", "xgb"):
            for vname, feats in variants.items():
                for h in cv.HORIZONS:
                    t = f"pm25_t+{h}"
                    tr, va = tr_all.dropna(subset=[t]), va_all.dropna(subset=[t])
                    m = cv.fit_p50(kind, tr[feats], tr[t] - tr["pm25"], tm)
                    pred = m.predict(va[feats]) + va["pm25"].values
                    y = va[t].values
                    hi = y > np.percentile(y, 90)
                    rows.append({"fold": fname, "model": kind, "variant": vname, "horizon": h, "mae": mean_absolute_error(y, pred),
                                 "mae_gt_p90": mean_absolute_error(y[hi], pred[hi])})
                r = pd.DataFrame([x for x in rows if x["fold"] == fname and x["model"] == kind and x["variant"] == vname])
                print(f"  [{kind}/{vname}] MAE={r['mae'].mean():.3f} >p90={r['mae_gt_p90'].mean():.3f}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(MODEL_DIR, f"validate_drop_neighbor_avg_cv_{TIMESTAMP}.csv"), index=False)
    fm = res.groupby(["fold", "model", "variant"])[["mae", "mae_gt_p90"]].mean().reset_index()
    pd.set_option("display.width", 200)
    for kind in ("lgb", "xgb"):
        print(f"\n{kind.upper()} — MAE ต่อ fold\n{fm[fm['model'] == kind].pivot(index='variant', columns='fold', values='mae').round(3).to_string()}")

    print("\n" + "=" * 100 + "\nการตัดสินตามเกณฑ์ที่ตั้งไว้ก่อนรัน (delta% ของ 20 เทียบ 21; บวก = แย่กว่า)\n" + "=" * 100)
    ok_all = True
    for kind in ("lgb", "xgb"):
        s = fm[fm["model"] == kind]
        a, b = s[s["variant"] == "21"].set_index("fold"), s[s["variant"].str.startswith("20")].set_index("fold")
        macro = (b["mae"].mean() - a["mae"].mean()) / a["mae"].mean() * 100
        burn = (b.loc[cv.BURNING_FOLDS, "mae"].mean() - a.loc[cv.BURNING_FOLDS, "mae"].mean()) / a.loc[cv.BURNING_FOLDS, "mae"].mean() * 100
        p90 = (b.loc[cv.BURNING_FOLDS, "mae_gt_p90"].mean() - a.loc[cv.BURNING_FOLDS, "mae_gt_p90"].mean()) / a.loc[cv.BURNING_FOLDS, "mae_gt_p90"].mean() * 100
        ok = macro <= TOL_MACRO and burn <= TOL_BURN and p90 <= TOL_BURN_P90
        ok_all &= ok
        print(f"  {kind.upper()}: macro {macro:+.2f}% (เกณฑ์ <= +{TOL_MACRO}%)  ฤดูฝุ่น {burn:+.2f}% (<= +{TOL_BURN}%)  >p90 ฤดูฝุ่น {p90:+.2f}% (<= +{TOL_BURN_P90}%)  -> {'ผ่าน' if ok else 'ไม่ผ่าน'}")
    print(f"\nผลรวม: {'ตัด pm25_neighbor_avg ได้ (เทรน v3 ใหม่ 20 feature)' if ok_all else 'ตัดไม่ได้ตามเกณฑ์ (คงชุด 21)'}")


if __name__ == "__main__":
    main()

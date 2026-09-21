"""
P3b ของแผนแก้ป้ายเวลา OpenAQ (CHANGELOG 145): ตัดสินชุด feature ของโมเดล v3 ด้วย rolling-origin (expanding window) CV แบบบล็อกเวลา ที่ครอบคลุมฤดูฝุ่น (ก.พ.-เม.ย.)
เหตุผล: calibration set (ก.ย.-ม.ค.) มองไม่เห็นฤดูฝุ่น ทั้งที่เป็นช่วงที่การตัดสินใจ 2 เรื่องขึ้นอยู่กับมัน (ใช้ neighbor ชุด k=1+nb_trend ไหม / ใช้เวลาเชิงวงกลมไหม)

Fold (train = ทุกแถวถึง val_start-24h เพื่อไม่ให้ target 24 ชม.ข้างหน้าซ้อนช่วง validate; ทุกจังหวัดที่เทรนได้ 17 จังหวัด, ปฏิทินเดียวกัน):
  A  val 2025-02-01..2025-04-30  (ฤดูฝุ่น 2025)          train: 2024-01..2025-01 (มีฤดูฝุ่น 2024 ฤดูเดียว)
  B  val 2025-09-01..2026-01-31  (ปลายฝน-หนาว)            train: 2024-01..2025-08
  C  val 2026-02-01..2026-04-30  (ฤดูฝุ่น 2026)          train: 2024-01..2026-01
  D  val 2026-05-01..2026-07-01  (ต้นฝน)                  train: 2024-01..2026-04
Variant (ทุก variant ใช้แถวชุดเดียวกัน): 18 (ไม่มี neighbor/fire) | +fire | +fire+time | 21 (nb_k1+trend+fire) | 21+time
horizon [1,12,24], LGB+XGB (hyperparameter ชุดเดิม จะจูนซ้ำใน P4), วัด MAE รวม และ MAE ที่ target > p90 ของ fold นั้น

เกณฑ์ตัดสิน — "ตั้งไว้ก่อนรัน" (จะไม่ปรับหลังเห็นผล; ถ้าผลก้ำกึ่งให้รายงานตามจริง):
  ตัวเลข = ค่าเฉลี่ยข้าม 4 fold (macro) ของ MAE รวม ต่อโมเดล; "fold ฤดูฝุ่น" = A และ C
  (1) ใช้ชุด neighbor (21 เทียบ +fire): macro MAE ดีขึ้น >= 0.5% ทั้ง LGB และ XGB  และ  MAE ของ fold ฤดูฝุ่น (A,C เฉลี่ย) ไม่แย่ลงเกิน 0.5% ในโมเดลใดเลย
      และ MAE >p90 ของ fold ฤดูฝุ่นไม่แย่ลงเกิน 1% ในโมเดลใดเลย
  (2) ใช้เวลาเชิงวงกลม (X+time เทียบ X โดย X = ชุดที่ตัดสินจากข้อ 1): เงื่อนไขเดียวกัน 3 ข้อ
  ไม่ผ่านข้อใด = ไม่ใช้ feature นั้น
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
HORIZONS = [1, 12, 24]
TIME_COLS = ["hour_sin", "hour_cos", "month_sin", "month_cos", "dayofweek"]
FOLDS = [("A ฤดูฝุ่น 2025", "2025-02-01", "2025-04-30"), ("B ปลายฝน-หนาว 2025/26", "2025-09-01", "2026-01-31"),
         ("C ฤดูฝุ่น 2026", "2026-02-01", "2026-04-30"), ("D ต้นฝน 2026", "2026-05-01", "2026-07-01")]
BURNING_FOLDS = ["A ฤดูฝุ่น 2025", "C ฤดูฝุ่น 2026"]
PASS_MACRO, PASS_BURN_MAE, PASS_BURN_P90 = 0.5, 0.5, 1.0  # เปอร์เซ็นต์


def load_tm():
    spec = importlib.util.spec_from_file_location("tm", os.path.join(_ROOT, "scripts", "train_multiprovince_models.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def fit_p50(kind, X, y, tm):
    if kind == "lgb":
        import lightgbm as lgb
        return lgb.LGBMRegressor(objective="quantile", alpha=0.5, random_state=42, verbose=-1, **tm.BAGGING_PARAMS_LGB, **tm.TUNED_PARAMS_LGB).fit(X, y)
    import xgboost as xgb
    return xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, tree_method="hist", random_state=42, verbosity=0,
                            **tm.BAGGING_PARAMS_XGB, **tm.TUNED_PARAMS_XGB).fit(X, y)


def main():
    tm = load_tm()
    man = json.load(open(sorted(glob.glob(os.path.join(DATA_DIR, "rebuild_clean_manifest_*.json")))[-1], encoding="utf-8"))
    df = pd.read_csv(man["final"])
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    nb1, trend, fire = "pm25_neighbor_avg", "nb_trend_12h", "fire_hotspot_count_100km_24h"
    base = [c for c in tm.FEATURE_COLS if c not in (nb1, trend, fire)]
    variants = {"18": base, "+fire": base + [fire], "+fire+time": base + [fire] + TIME_COLS,
                "21 (nb+trend+fire)": base + [nb1, trend, fire], "21+time": base + [nb1, trend, fire] + TIME_COLS}
    cols = sorted({c for v in variants.values() for c in v})
    d = df[~df["is_zero_shot"]].dropna(subset=cols).reset_index(drop=True)
    print(f"dataset สะอาด: {man['final']}\nแถวที่ใช้ {len(d)} ({d['province_en'].nunique()} จังหวัด)")

    rows = []
    for fname, v0, v1 in FOLDS:
        vs, ve = pd.Timestamp(v0, tz="UTC"), pd.Timestamp(v1, tz="UTC") + pd.Timedelta(hours=23)
        tr_all = d[d["datetime_utc"] <= vs - pd.Timedelta(hours=24)]
        va_all = d[(d["datetime_utc"] >= vs) & (d["datetime_utc"] <= ve)]
        print(f"\n=== fold {fname}: train {len(tr_all)} แถว, validate {len(va_all)} แถว ({va_all['province_en'].nunique()} จังหวัด) ===", flush=True)
        for kind in ("lgb", "xgb"):
            for vname, feats in variants.items():
                for h in HORIZONS:
                    t = f"pm25_t+{h}"
                    tr, va = tr_all.dropna(subset=[t]), va_all.dropna(subset=[t])
                    m = fit_p50(kind, tr[feats], tr[t] - tr["pm25"], tm)
                    pred = m.predict(va[feats]) + va["pm25"].values
                    y = va[t].values
                    hi = y > np.percentile(y, 90)
                    rows.append({"fold": fname, "model": kind, "variant": vname, "horizon": h, "mae": mean_absolute_error(y, pred),
                                 "mae_gt_p90": mean_absolute_error(y[hi], pred[hi]), "n": len(va)})
                r = pd.DataFrame([x for x in rows if x["fold"] == fname and x["model"] == kind and x["variant"] == vname])
                print(f"  [{kind}/{vname}] MAE={r['mae'].mean():.3f} >p90={r['mae_gt_p90'].mean():.3f}", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(MODEL_DIR, f"validate_features_timeblocked_cv_{TIMESTAMP}.csv"), index=False)

    fm = res.groupby(["fold", "model", "variant"])[["mae", "mae_gt_p90"]].mean().reset_index()
    pd.set_option("display.width", 230)
    print("\n" + "=" * 110 + "\nMAE เฉลี่ย 3 horizon ต่อ fold (แถว = variant)\n" + "=" * 110)
    for kind in ("lgb", "xgb"):
        pt = fm[fm["model"] == kind].pivot(index="variant", columns="fold", values="mae").reindex(list(variants))
        pt["macro"] = pt.mean(axis=1)
        print(f"\n{kind.upper()} — MAE\n{pt.round(3).to_string()}")
        p9 = fm[fm["model"] == kind].pivot(index="variant", columns="fold", values="mae_gt_p90").reindex(list(variants))
        p9["macro"] = p9.mean(axis=1)
        print(f"\n{kind.upper()} — MAE ที่ target > p90 ของ fold\n{p9.round(3).to_string()}")

    def stats(kind, variant):
        s = fm[(fm["model"] == kind) & (fm["variant"] == variant)].set_index("fold")
        return s["mae"].mean(), s.loc[BURNING_FOLDS, "mae"].mean(), s.loc[BURNING_FOLDS, "mae_gt_p90"].mean()

    def compare(new, old):
        out = {}
        for kind in ("lgb", "xgb"):
            (mn, bn, pn), (mo, bo, po) = stats(kind, new), stats(kind, old)
            out[kind] = ((mn - mo) / mo * 100, (bn - bo) / bo * 100, (pn - po) / po * 100)
        ok = all(out[k][0] <= -PASS_MACRO and out[k][1] <= PASS_BURN_MAE and out[k][2] <= PASS_BURN_P90 for k in out)
        return ok, out

    print("\n" + "=" * 110 + "\nการตัดสินตามเกณฑ์ที่ตั้งไว้ก่อนรัน (delta% ติดลบ = ดีกว่า; macro / fold ฤดูฝุ่น MAE / fold ฤดูฝุ่น >p90)\n" + "=" * 110)
    ok1, o1 = compare("21 (nb+trend+fire)", "+fire")
    for k, (a, b, c) in o1.items():
        print(f"  (1) ชุด neighbor (21 vs +fire) {k.upper()}: macro {a:+.2f}%  ฤดูฝุ่น {b:+.2f}%  >p90 ฤดูฝุ่น {c:+.2f}%")
    x_name = "21 (nb+trend+fire)" if ok1 else "+fire"
    print(f"  -> ข้อ (1): {'ผ่าน ใช้ neighbor ชุด k=1+nb_trend' if ok1 else 'ไม่ผ่าน ไม่ใช้ neighbor'}  (X = {x_name})")
    ok2, o2 = compare("21+time" if ok1 else "+fire+time", x_name)
    for k, (a, b, c) in o2.items():
        print(f"  (2) เวลาเชิงวงกลม ({'21+time' if ok1 else '+fire+time'} vs {x_name}) {k.upper()}: macro {a:+.2f}%  ฤดูฝุ่น {b:+.2f}%  >p90 ฤดูฝุ่น {c:+.2f}%")
    print(f"  -> ข้อ (2): {'ผ่าน ใช้เวลาเชิงวงกลม' if ok2 else 'ไม่ผ่าน ไม่ใช้เวลาเชิงวงกลม'}")
    final = (x_name + ("+time" if ok2 and x_name == "+fire" else "")) if not (ok2 and ok1) else "21+time"
    if ok2 and not ok1:
        final = "+fire+time"
    print(f"\nชุด feature ที่เกณฑ์เลือก: {final}")


if __name__ == "__main__":
    main()

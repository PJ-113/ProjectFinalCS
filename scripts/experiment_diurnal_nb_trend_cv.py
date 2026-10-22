"""
ทดสอบว่าแยก nb_trend_12h ปัจจุบันออกเป็น 2 feature — nb_trend_12h เดิม (แนวโน้ม 12 ชม.ล่าสุด)
+ nb_trend_prev12h (แนวโน้ม 12 ชม.ก่อนหน้านั้น) — ช่วยพยากรณ์ตอนเช้า (06:00-09:00 ไทย) ได้จริงไหม
วัด MAE รวม/เฉพาะช่วงเช้า/เฉพาะฤดูฝุ่น ด้วย time-blocked CV (4 fold ครอบคลุม 2 ฤดูฝุ่น)

เกณฑ์ตัดสิน: ใช้ชุดแยก 2 feature แทนชุดเดิมได้ ถ้า MAE ช่วงเช้า (macro เฉลี่ย 4 fold) ดีขึ้น >=1%
ทั้ง LGB และ XGB และ MAE รวม/MAE ฤดูฝุ่น ไม่แย่ลงเกิน 0.5% ในโมเดลใดเลย
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
FOLDS = [("A ฤดูฝุ่น 2025", "2025-02-01", "2025-04-30"), ("B ปลายฝน-หนาว 2025/26", "2025-09-01", "2026-01-31"),
         ("C ฤดูฝุ่น 2026", "2026-02-01", "2026-04-30"), ("D ต้นฝน 2026", "2026-05-01", "2026-07-01")]
BURNING_FOLDS = ["A ฤดูฝุ่น 2025", "C ฤดูฝุ่น 2026"]
PASS_MORNING, PASS_OVERALL, PASS_BURN = 1.0, 0.5, 0.5  # เปอร์เซ็นต์ (delta% ที่ยอมรับได้)
MORNING_HOURS = {6, 7, 8, 9}  # เวลาไทยของ target ที่ถือว่าเป็น "ช่วงเช้า"


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


def add_prev12h_trend(df):
    """nb_trend_prev12h(t) = nb_trend_12h(t-12h) (แถวเดียวกัน จังหวัดเดียวกัน) ตรวจซ้ำผ่าน pivot recompute อิสระ"""
    lag = df[["province_en", "datetime_utc", "nb_trend_12h"]].copy()
    lag["datetime_utc"] = lag["datetime_utc"] + pd.Timedelta(hours=12)
    lag = lag.rename(columns={"nb_trend_12h": "nb_trend_prev12h"})
    n0 = len(df)
    out = df.merge(lag, on=["province_en", "datetime_utc"], how="left")
    assert len(out) == n0, "merge ทำให้จำนวนแถวเปลี่ยน"

    piv = df.pivot_table(index="datetime_utc", columns="province_en", values="nb_trend_12h")
    ref = piv.shift(freq=pd.Timedelta(hours=12)).stack().rename("ref").reset_index()
    chk = out.merge(ref, on=["datetime_utc", "province_en"], how="left")
    ok = np.allclose(chk["nb_trend_prev12h"].fillna(-999).values, chk["ref"].fillna(-999).values)
    assert ok, "nb_trend_prev12h ไม่ตรงกับ recompute"
    print(f"nb_trend_prev12h ตรงกับ recompute (allclose); NaN {out['nb_trend_prev12h'].isna().mean():.3%}")
    return out


def main():
    tm = load_tm()
    man = json.load(open(sorted(glob.glob(os.path.join(DATA_DIR, "rebuild_clean_manifest_*.json")))[-1], encoding="utf-8"))
    df = pd.read_csv(man["final"])
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df = add_prev12h_trend(df)

    baseline = tm.FEATURE_COLS
    assert "nb_trend_12h" in baseline
    diurnal = baseline + ["nb_trend_prev12h"]
    variants = {"baseline (21, nb_trend_12h)": baseline, "diurnal (+nb_trend_prev12h)": diurnal}
    cols = sorted({c for v in variants.values() for c in v})
    d = df[~df["is_zero_shot"]].dropna(subset=cols).reset_index(drop=True)
    print(f"dataset สะอาด: {man['final']}\nแถวที่ใช้ (มี nb_trend_prev12h ครบ) {len(d)} ({d['province_en'].nunique()} จังหวัด)")

    rows = []
    for fname, v0, v1 in FOLDS:
        vs, ve = pd.Timestamp(v0, tz="UTC"), pd.Timestamp(v1, tz="UTC") + pd.Timedelta(hours=23)
        tr_all = d[d["datetime_utc"] <= vs - pd.Timedelta(hours=24)]
        va_all = d[(d["datetime_utc"] >= vs) & (d["datetime_utc"] <= ve)]
        print(f"\n=== fold {fname}: train {len(tr_all)} แถว, validate {len(va_all)} แถว ===", flush=True)
        for kind in ("lgb", "xgb"):
            for vname, feats in variants.items():
                for h in HORIZONS:
                    t = f"pm25_t+{h}"
                    tr, va = tr_all.dropna(subset=[t]), va_all.dropna(subset=[t])
                    m = fit_p50(kind, tr[feats], tr[t] - tr["pm25"], tm)
                    pred = m.predict(va[feats]) + va["pm25"].values
                    y = va[t].values
                    hi = y > np.percentile(y, 90)
                    target_hour_th = (va["datetime_utc"] + pd.Timedelta(hours=h)).dt.tz_convert("Asia/Bangkok").dt.hour.values
                    morning = np.isin(target_hour_th, list(MORNING_HOURS))
                    row = {"fold": fname, "model": kind, "variant": vname, "horizon": h, "n": len(va),
                           "mae": mean_absolute_error(y, pred), "mae_gt_p90": mean_absolute_error(y[hi], pred[hi]),
                           "n_morning": int(morning.sum())}
                    row["mae_morning"] = mean_absolute_error(y[morning], pred[morning]) if morning.sum() >= 10 else np.nan
                    rows.append(row)
                r = pd.DataFrame([x for x in rows if x["fold"] == fname and x["model"] == kind and x["variant"] == vname])
                print(f"  [{kind}/{vname}] MAE={r['mae'].mean():.3f} >p90={r['mae_gt_p90'].mean():.3f} "
                      f"เช้า(06-09น.)={r['mae_morning'].mean():.3f} (n_เช้าเฉลี่ย={r['n_morning'].mean():.0f})", flush=True)

    res = pd.DataFrame(rows)
    out_path = os.path.join(MODEL_DIR, f"experiment_diurnal_nb_trend_cv_{TIMESTAMP}.csv")
    res.to_csv(out_path, index=False)
    print(f"\nผลดิบ: {out_path}")

    fm = res.groupby(["fold", "model", "variant"])[["mae", "mae_gt_p90", "mae_morning"]].mean().reset_index()
    pd.set_option("display.width", 230)
    print("\n" + "=" * 110 + "\nMAE เฉลี่ย 3 horizon ต่อ fold\n" + "=" * 110)
    for kind in ("lgb", "xgb"):
        for metric, title in (("mae", "MAE รวม"), ("mae_gt_p90", "MAE ที่ target > p90"), ("mae_morning", "MAE ช่วงเช้า (06-09น.)")):
            pt = fm[fm["model"] == kind].pivot(index="variant", columns="fold", values=metric).reindex(list(variants))
            pt["macro"] = pt.mean(axis=1)
            print(f"\n{kind.upper()} — {title}\n{pt.round(3).to_string()}")

    def stats(kind, variant):
        s = fm[(fm["model"] == kind) & (fm["variant"] == variant)].set_index("fold")
        return s["mae"].mean(), s.loc[BURNING_FOLDS, "mae"].mean(), s["mae_morning"].mean()

    print("\n" + "=" * 110 + "\nการตัดสินตามเกณฑ์ที่ตั้งไว้ก่อนรัน (delta% ติดลบ = ดีกว่า)\n" + "=" * 110)
    ok_all = True
    for kind in ("lgb", "xgb"):
        (mn, bn, morn_n), (mo, bo, morn_o) = stats(kind, "diurnal (+nb_trend_prev12h)"), stats(kind, "baseline (21, nb_trend_12h)")
        d_overall, d_burn, d_morning = (mn - mo) / mo * 100, (bn - bo) / bo * 100, (morn_n - morn_o) / morn_o * 100
        ok = d_morning <= -PASS_MORNING and d_overall <= PASS_OVERALL and d_burn <= PASS_BURN
        ok_all = ok_all and ok
        print(f"  {kind.upper()}: MAE ช่วงเช้า {d_morning:+.2f}% (เกณฑ์ <= -{PASS_MORNING}%)  MAE รวม {d_overall:+.2f}% (เกณฑ์ <= +{PASS_OVERALL}%)  "
              f"MAE ฤดูฝุ่น {d_burn:+.2f}% (เกณฑ์ <= +{PASS_BURN}%)  -> {'ผ่าน' if ok else 'ไม่ผ่าน'}")
    print(f"\nสรุป: {'ผ่านทั้ง 2 โมเดล -> เก็บเป็น candidate (ยังไม่ deploy)' if ok_all else 'ไม่ผ่าน -> ไม่ใช้แนวทางนี้ รายงานผลลบ'}")


if __name__ == "__main__":
    main()

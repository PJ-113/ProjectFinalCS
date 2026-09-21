"""
CHANGELOG 144 — 9 สถานีมีป้ายเวลาของ OpenAQ /measurements เลื่อน +23 ชม. (ดู check_openaq_timestamp_shift.py) ค่าที่ป้าย t คือค่าจริงของ t+23
ผลที่ต้องวัด: จังหวัด "ปกติ" ที่เพื่อนบ้านอันดับ 1 เป็นสถานีเลื่อน -> pm25_neighbor_avg / nb_trend_12h ในแถวเทรน ณ เวลา t มี "ข้อมูลอนาคต 23 ชม." รั่วเข้าไป
(ซึ่ง live ไม่มี) ทำให้ประโยชน์ของ feature จังหวัดข้างเคียงที่รายงานไว้ (CHANGELOG 94-110, 133-137) อาจถูกประเมินสูงเกินจริง โดยเฉพาะ horizon ไกล

การทดลอง (แยกผลกระทบให้สะอาด): เทรน/ทดสอบเฉพาะจังหวัดที่ป้ายเวลาของตัวเองถูกต้อง (9 จังหวัดปกติที่เทรนได้) — target/lag ของจังหวัดเหล่านี้ตรงความจริง
  A = feature เพื่อนบ้านตามที่เทรนจริง (มีรั่วสำหรับจังหวัดที่เพื่อนบ้านเป็นสถานีเลื่อน)
  B = feature เพื่อนบ้านที่แก้ป้ายเวลาแล้ว (ซีรีส์สถานีเลื่อนถูกเลื่อน +23 ชม. ให้ตรงความจริง -> ไม่มีรั่ว, สูตรเดิมทุกอย่าง)
  C = ไม่ใช้ feature เพื่อนบ้านเลย
  D = เทรนแบบ A (มีรั่ว) แต่ตอนใช้งานจริงป้อน feature แบบ B (ไม่รั่ว) = สถานการณ์ live ของโมเดลที่ deploy (เทียบ C: ถ้า D แย่กว่า C แปลว่าโมเดลที่ deploy แย่กว่าไม่มี neighbor เลย)
วัด MAE (P50, hyperparameter จูนแล้ว, 5 horizon) บน test แยก "จังหวัดที่เพื่อนบ้านเป็นสถานีเลื่อน (เสี่ยงรั่ว)" กับ "ไม่เสี่ยง"
ถ้า A ดีกว่า B มากเฉพาะกลุ่มเสี่ยงและ horizon ไกล = หลักฐานว่าส่วนหนึ่งของผลเดิมมาจากการรั่ว
"""
import importlib.util
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(_ROOT, "models")
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
SHIFTED = {"surin", "sisaket", "kalasin", "yasothon", "chaiyaphum", "amnatcharoen", "nongbualamphu", "buengkan", "mahasarakham"}
SHIFT_H = 23
HORIZONS = [1, 6, 12, 18, 24]
NB_COLS = ["pm25_neighbor_avg", "nb_trend_12h"]


def load_tm():
    spec = importlib.util.spec_from_file_location("tm", os.path.join(_ROOT, "scripts", "train_multiprovince_models.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def hav(a, b, reg):
    p1, p2 = np.radians(reg[a]["lat"]), np.radians(reg[b]["lat"])
    x = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(reg[b]["lon"] - reg[a]["lon"]) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(x))


def fit_p50(kind, X, y, tm):
    if kind == "lgb":
        import lightgbm as lgb
        return lgb.LGBMRegressor(objective="quantile", alpha=0.5, random_state=42, verbose=-1, **tm.BAGGING_PARAMS_LGB, **tm.TUNED_PARAMS_LGB).fit(X, y)
    import xgboost as xgb
    return xgb.XGBRegressor(objective="reg:quantileerror", quantile_alpha=0.5, tree_method="hist", random_state=42, verbosity=0,
                            **tm.BAGGING_PARAMS_XGB, **tm.TUNED_PARAMS_XGB).fit(X, y)


def main():
    tm = load_tm()
    df = pd.read_csv(tm.DATA_PATH)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    reg = {q["province_en"]: q for q in json.load(open(os.path.join(_ROOT, "data", "province_station_registry.json"), encoding="utf-8"))["provinces"]}
    nearest = {a: min((b for b in reg if b != a), key=lambda b: hav(a, b, reg)) for a in reg}

    # ซีรีส์ pm25 รายชั่วโมงของทุกจังหวัด (ป้ายเวลาตามที่เทรน) และฉบับแก้ป้ายเวลา (สถานีเลื่อน: ค่าที่ป้าย t เป็นของ t+23 -> ค่าจริง ณ t = ค่าที่ป้าย t-23)
    pv = df.pivot_table(index="datetime_utc", columns="province_en", values="pm25")
    pv = pv.reindex(pd.date_range(pv.index.min(), pv.index.max(), freq="h", tz="UTC"))
    pv_corr = pv.copy()
    for p in SHIFTED:
        if p in pv_corr:
            pv_corr[p] = pv[p].shift(SHIFT_H)

    d = df[(~df["is_zero_shot"]) & (~df["province_en"].isin(SHIFTED))].copy()   # 9 จังหวัดที่ป้ายเวลาของตัวเองถูกต้อง
    print("จังหวัดที่ใช้ (ป้ายเวลาถูกต้อง):", sorted(d["province_en"].unique()))
    exposed = {p for p in d["province_en"].unique() if nearest[p] in SHIFTED}
    print("เสี่ยงรั่ว (เพื่อนบ้านอันดับ 1 เป็นสถานีเลื่อน):", {p: nearest[p] for p in sorted(exposed)})
    print("ไม่เสี่ยง:", {p: nearest[p] for p in sorted(set(d['province_en'].unique()) - exposed)})

    # feature เพื่อนบ้านฉบับแก้ป้ายเวลา (สูตรเดิม: ค่าของเพื่อนบ้านอันดับ 1 ณ t และ ณ t-12)
    nb_corr = np.full(len(d), np.nan)
    tr_corr = np.full(len(d), np.nan)
    for p, g in d.groupby("province_en"):
        s = pv_corr[nearest[p]]
        now_v = s.reindex(g["datetime_utc"]).values
        ago_v = s.reindex(g["datetime_utc"] - pd.Timedelta(hours=12)).values
        pos = d.index.get_indexer(g.index)
        nb_corr[pos], tr_corr[pos] = now_v, now_v - ago_v
    d["nb_avg_corr"], d["nb_trend_corr"] = nb_corr, tr_corr
    # ตรวจ: ฉบับเดิมของจังหวัดที่ไม่เสี่ยงต้องตรงกับฉบับแก้ (เพื่อนบ้านปกติ) -> พิสูจน์ว่าการคำนวณถูก
    safe = d[~d["province_en"].isin(exposed)]
    ok = (safe["pm25_neighbor_avg"] - safe["nb_avg_corr"]).abs().le(1e-6).mean()
    print(f"ตรวจสูตร: จังหวัดไม่เสี่ยง feature เดิม == feature ที่คำนวณใหม่ {ok:.1%} ของแถว (ควรใกล้ 100%)")
    d = d.dropna(subset=["nb_avg_corr", "nb_trend_corr"]).reset_index(drop=True)

    base_wo_nb = [c for c in tm.FEATURE_COLS if c not in NB_COLS]
    variants = {"A: เดิม (มีรั่ว)": (base_wo_nb + NB_COLS, None), "B: แก้ป้ายเวลา (ไม่รั่ว)": (base_wo_nb + ["nb_avg_corr", "nb_trend_corr"], None),
                "C: ไม่มี neighbor": (base_wo_nb, None)}
    train_df, _, test_df = tm.split_train_calib_test(d)
    print(f"train={len(train_df)} test={len(test_df)}")
    rows = []
    for kind in ("lgb", "xgb"):
        for vname, (feats, _) in variants.items():
            for h in HORIZONS:
                t = f"pm25_t+{h}"
                tr, te = train_df.dropna(subset=[t]), test_df.dropna(subset=[t])
                m = fit_p50(kind, tr[feats], tr[t] - tr["pm25"], tm)
                pred = m.predict(te[feats]) + te["pm25"].values
                preds = [(vname, pred)]
                if vname.startswith("A"):
                    # D: โมเดลที่เทรนด้วย A (มีรั่ว) แต่ตอนใช้งานจริง feature เพื่อนบ้านไม่รั่ว (= ฉบับแก้ป้ายเวลา) -> สถานการณ์ live ของโมเดลที่ deploy
                    te_live = te[feats].copy()
                    te_live["pm25_neighbor_avg"], te_live["nb_trend_12h"] = te["nb_avg_corr"].values, te["nb_trend_corr"].values
                    preds.append(("D: เทรนแบบ A แต่ใช้จริงแบบ B (live)", m.predict(te_live) + te["pm25"].values))
                for vn, pr in preds:
                    for grp, mask in (("เสี่ยงรั่ว", te["province_en"].isin(exposed)), ("ไม่เสี่ยง", ~te["province_en"].isin(exposed)), ("ทั้งหมด", te["province_en"].notna())):
                        if mask.sum():
                            rows.append({"model": kind, "variant": vn, "horizon": h, "group": grp, "mae": mean_absolute_error(te.loc[mask, t], pr[mask.values]), "n": int(mask.sum())})
            print(f"  {kind} {vname} เสร็จ", flush=True)
    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(MODEL_DIR, f"analyze_timestamp_shift_leakage_{TIMESTAMP}.csv"), index=False)
    pd.set_option("display.width", 220)
    for kind in ("lgb", "xgb"):
        print("\n" + "=" * 100 + f"\n{kind.upper()} — MAE (P50) ต่อ horizon; delta% เทียบ C (ไม่มี neighbor) ; ติดลบ = ดีกว่า\n" + "=" * 100)
        for grp in ("เสี่ยงรั่ว", "ไม่เสี่ยง", "ทั้งหมด"):
            sub = res[(res["model"] == kind) & (res["group"] == grp)]
            pt = sub.pivot(index="horizon", columns="variant", values="mae")
            c = pt["C: ไม่มี neighbor"]
            out = pd.DataFrame({"C": c, "A เดิม": pt["A: เดิม (มีรั่ว)"], "B แก้แล้ว": pt["B: แก้ป้ายเวลา (ไม่รั่ว)"],
                                "D live": pt["D: เทรนแบบ A แต่ใช้จริงแบบ B (live)"]})
            out["A vs C %"] = (out["A เดิม"] - out["C"]) / out["C"] * 100
            out["B vs C %"] = (out["B แก้แล้ว"] - out["C"]) / out["C"] * 100
            out["D vs C %"] = (out["D live"] - out["C"]) / out["C"] * 100
            n = int(sub["n"].iloc[0])
            print(f"\n[{grp}] (แถว test ต่อ horizon ~{n})\n{out.round(3).to_string()}")


if __name__ == "__main__":
    main()

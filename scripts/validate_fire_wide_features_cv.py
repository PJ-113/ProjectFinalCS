"""
ทดลองข้อ 2 (CHANGELOG 153): ปรับ feature จุดความร้อนให้ละเอียดขึ้น เพื่อดูว่าลด MAE ช่วงฝุ่นวิกฤต (>p90) ได้ไหม
- ข้อมูล: FIRMS กรอบกว้าง (lon 97-108.5, lat 9.5-22 — รวมลาว/กัมพูชา/เมียนมา/เวียดนาม) จาก fetch_firms_hotspots_wide.py
- เหตุผลที่ทดสอบซ้ำ: CHANGELOG 135 เคยทดสอบ FRP/ต้นลม แล้วไม่ช่วย แต่ (ก) ใช้ข้อมูลที่ป้ายเวลาผิด (ข) เลือกบน calibration split
  ที่ไม่มีฤดูฝุ่น (ค) ไม่มีไฟข้ามพรมแดน (ข้อมูลเดิมดึงแค่กรอบอีสาน) ตอนนี้ใช้ CV แบ่งตามเวลา 4 fold ที่ครอบคลุมฤดูฝุ่นเหมือน CHANGELOG 146

Feature ตัวเลือก (เพิ่มเข้าชุด 20 feature ปัจจุบัน, ทุกตัวนับในหน้าต่าง 24 ชม. ที่ผ่านมา):
  C1 fire_250km_24h      จำนวนจุดรัศมี 250 กม. (รวมข้ามพรมแดน)
  C2 fire_upwind_250km   จำนวนจุดรัศมี 250 กม. ที่อยู่ทิศต้นลม (sector 30 องศา 12 ช่อง; ช่องของทิศลมขณะนั้น + ช่องข้างเคียง 2 ช่อง = 90 องศา)
                         ใช้ wind_dir ณ เวลา t (ตอนพยากรณ์สดมีจากข้อมูลอากาศปัจจุบัน)
  C3 fire_frp_100km_24h  ผลรวม FRP รัศมี 100 กม.
  C4 C1+C2+C3
  P  placebo             เหมือน C1 แต่ใช้ไฟจากอีก 364 วัน (คงฤดูกาลและ weekday ทำลายความสัมพันธ์รายวัน) ไว้วัดระดับ noise
                         (ใช้ t+364 วันถ้ามีข้อมูล ไม่งั้น t-364 วัน)

เกณฑ์ผ่าน — ตั้งไว้ก่อนรัน ห้ามปรับหลังเห็นผล (delta% ติดลบ = ดีกว่า baseline 20 feature; "ฤดูฝุ่น" = fold A,C):
  ต้องผ่านครบทุกข้อ ทั้ง LGB และ XGB:
  (ก) MAE ที่ target > p90 ของ fold ฤดูฝุ่น (เฉลี่ย A,C) ดีขึ้น >= 2.0%
  (ข) ดีขึ้นทั้ง fold A และ fold C แยกกัน (>p90 delta < 0 ทั้งสอง)
  (ค) macro MAE ข้าม 4 fold ไม่แย่ลงเกิน +0.2%  และ MAE รวมของ fold ฤดูฝุ่นไม่แย่ลงเกิน +0.5%
  (ง) ดีกว่า placebo อย่างน้อย 1.0 จุดเปอร์เซ็นต์ในข้อ (ก) (delta_p90 ของตัวเลือก <= delta_p90 ของ placebo - 1.0)
  ไม่ผ่านข้อใด = ไม่เปลี่ยนโมเดล ; ผ่านแล้วค่อยยืนยันบน test ครั้งเดียวก่อนตัดสินใจ retrain
  หมายเหตุ: ทดสอบ 4 ตัวเลือกพร้อมกัน เกณฑ์ (ก) 2.0% + (ง) placebo ตั้งไว้เพื่อกันผลบังเอิญ
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
WINDOW = np.timedelta64(24, "h").astype("timedelta64[ns]").astype(np.int64)
SHIFT_NS = np.timedelta64(364, "D").astype("timedelta64[ns]").astype(np.int64)
R_WIDE, R_NEAR = 250.0, 100.0
N_SECT = 12
PASS_P90, PASS_MACRO, PASS_BURN, PASS_PLACEBO_GAP = 2.0, 0.2, 0.5, 1.0


def load_cv_module():
    spec = importlib.util.spec_from_file_location("cvmod", os.path.join(_ROOT, "scripts", "validate_features_timeblocked_cv.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def haversine_bearing(lat0, lon0, lat, lon):
    r = 6371.0
    p1, p2 = np.radians(lat0), np.radians(lat)
    dphi, dl = p2 - p1, np.radians(lon - lon0)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    dist = 2 * r * np.arcsin(np.sqrt(a))
    y = np.sin(dl) * np.cos(p2)
    x = np.cos(p1) * np.sin(p2) - np.sin(p1) * np.cos(p2) * np.cos(dl)
    bearing = (np.degrees(np.arctan2(y, x)) + 360.0) % 360.0
    return dist, bearing


def to_ns(series):
    """แปลง datetime series (ความละเอียด ns/us ก็ได้) เป็น int64 นาโนวินาที UTC ให้ตรงกับ WINDOW/SHIFT_NS"""
    return series.dt.tz_convert("UTC").dt.tz_localize(None).astype("datetime64[ns]").astype("int64").to_numpy()


def window_counts(sorted_t, q_ns):
    hi = np.searchsorted(sorted_t, q_ns, side="right")
    lo = np.searchsorted(sorted_t, q_ns - WINDOW, side="right")
    return hi - lo


def build_features(df, hs):
    """คืน DataFrame index เดียวกับ df: fire_100km_24h(ตรวจ), C1, C2, C3, placebo"""
    hlat, hlon = hs["latitude"].to_numpy(), hs["longitude"].to_numpy()
    ht = to_ns(hs["acq_datetime_utc"])
    hfrp = hs["frp"].fillna(0).to_numpy(dtype=float)
    t_max = ht.max()
    out = pd.DataFrame(index=df.index, columns=["fire100_check", "fire_250km_24h", "fire_upwind_250km", "fire_frp_100km_24h",
                                                  "fire_placebo_250km"], dtype=float)
    for prov, g in df.groupby("province_en"):
        lat0, lon0 = g["lat"].iloc[0], g["lon"].iloc[0]
        dist, bearing = haversine_bearing(lat0, lon0, hlat, hlon)
        q = to_ns(g["datetime_utc"])
        wind_from = (np.degrees(np.arctan2(g["wind_dir_sin"].to_numpy(), g["wind_dir_cos"].to_numpy())) + 360.0) % 360.0

        m_w = dist <= R_WIDE
        order = np.argsort(ht[m_w])
        t_w, b_w = ht[m_w][order], bearing[m_w][order]
        d_w = dist[m_w][order]
        frp_w = hfrp[m_w][order]

        c1 = window_counts(t_w, q)
        # placebo: t+364d ถ้ามีข้อมูล ไม่งั้น t-364d
        use_fwd = (q + SHIFT_NS) <= t_max
        cp = np.where(use_fwd, window_counts(t_w, q + SHIFT_NS), window_counts(t_w, q - SHIFT_NS))

        # 100 กม.
        m_n = d_w <= R_NEAR
        t_n, frp_n = t_w[m_n], frp_w[m_n]
        c100 = window_counts(t_n, q)
        cs = np.concatenate([[0.0], np.cumsum(frp_n)])
        hi = np.searchsorted(t_n, q, side="right")
        lo = np.searchsorted(t_n, q - WINDOW, side="right")
        frp_sum = cs[hi] - cs[lo]

        # ต้นลม: 12 sector (center = k*30 องศา) นับต่อ sector แล้วเลือก sector ของทิศลม +- 1
        sect = np.round(b_w / (360.0 / N_SECT)).astype(int) % N_SECT
        per_sect = np.zeros((len(q), N_SECT))
        for s in range(N_SECT):
            per_sect[:, s] = window_counts(np.sort(t_w[sect == s]), q)
        s_w = np.round(wind_from / (360.0 / N_SECT)).astype(int) % N_SECT
        idx = np.arange(len(q))
        upwind = per_sect[idx, s_w] + per_sect[idx, (s_w - 1) % N_SECT] + per_sect[idx, (s_w + 1) % N_SECT]

        out.loc[g.index, "fire100_check"] = c100
        out.loc[g.index, "fire_250km_24h"] = c1
        out.loc[g.index, "fire_upwind_250km"] = upwind
        out.loc[g.index, "fire_frp_100km_24h"] = frp_sum
        out.loc[g.index, "fire_placebo_250km"] = cp
    return out


def main():
    cv = load_cv_module()
    tm = cv.load_tm()
    man = json.load(open(sorted(glob.glob(os.path.join(DATA_DIR, "rebuild_clean_manifest_*.json")))[-1], encoding="utf-8"))
    df = pd.read_csv(man["final"])
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    hs_path = sorted(glob.glob(os.path.join(DATA_DIR, "firms_hotspots_wide_2*.csv")))[-1]
    hs = pd.read_csv(hs_path, usecols=["latitude", "longitude", "frp", "acq_datetime_utc"])
    hs["acq_datetime_utc"] = pd.to_datetime(hs["acq_datetime_utc"], utc=True)
    print(f"dataset: {man['final']}\nhotspot (กว้าง): {hs_path}  {len(hs)} จุด", flush=True)

    feats = build_features(df, hs)
    chk = np.corrcoef(feats["fire100_check"].to_numpy(dtype=float), df["fire_hotspot_count_100km_24h"].to_numpy(dtype=float))[0, 1]
    print(f"sanity: corr(นับ 100km จากข้อมูลกว้าง, feature เดิมใน dataset) = {chk:.4f}  "
          f"mean ใหม่={feats['fire100_check'].mean():.2f} เดิม={df['fire_hotspot_count_100km_24h'].mean():.2f}", flush=True)
    assert chk > 0.9, f"sanity ไม่ผ่าน (corr={chk:.3f}) — feature ที่คำนวณใหม่ไม่ตรงกับ feature เดิม หยุดก่อนรัน CV"
    feat_path = os.path.join(DATA_DIR, f"fire_wide_candidate_features_{TIMESTAMP}.csv")
    pd.concat([df[["province_en", "datetime_utc"]], feats], axis=1).to_csv(feat_path, index=False)
    print(f"บันทึกฟีเจอร์ตัวเลือกที่: {feat_path}", flush=True)
    for c in feats.columns:
        df[c] = feats[c].astype(float)

    base = list(tm.FEATURE_COLS)
    C1, C2, C3, PL = "fire_250km_24h", "fire_upwind_250km", "fire_frp_100km_24h", "fire_placebo_250km"
    variants = {"base20": base, "+C1 250km": base + [C1], "+C2 upwind": base + [C2], "+C3 frp": base + [C3],
                "+C4 all": base + [C1, C2, C3], "placebo": base + [PL]}
    cols = sorted({c for v in variants.values() for c in v})
    d = df[~df["is_zero_shot"]].dropna(subset=cols).reset_index(drop=True)
    print(f"แถวที่ใช้ {len(d)} ({d['province_en'].nunique()} จังหวัด)", flush=True)

    rows = []
    for fname, v0, v1 in cv.FOLDS:
        vs, ve = pd.Timestamp(v0, tz="UTC"), pd.Timestamp(v1, tz="UTC") + pd.Timedelta(hours=23)
        tr_all = d[d["datetime_utc"] <= vs - pd.Timedelta(hours=24)]
        va_all = d[(d["datetime_utc"] >= vs) & (d["datetime_utc"] <= ve)]
        print(f"\n=== fold {fname}: train {len(tr_all)}, validate {len(va_all)} ===", flush=True)
        for kind in ("lgb", "xgb"):
            for vname, fs in variants.items():
                for h in HORIZONS:
                    t = f"pm25_t+{h}"
                    tr, va = tr_all.dropna(subset=[t]), va_all.dropna(subset=[t])
                    m = cv.fit_p50(kind, tr[fs], tr[t] - tr["pm25"], tm)
                    pred = m.predict(va[fs]) + va["pm25"].values
                    y = va[t].values
                    hi = y > np.percentile(y, 90)
                    rows.append({"fold": fname, "model": kind, "variant": vname, "horizon": h, "mae": mean_absolute_error(y, pred),
                                 "mae_gt_p90": mean_absolute_error(y[hi], pred[hi]), "n": len(va)})
                r = pd.DataFrame([x for x in rows if x["fold"] == fname and x["model"] == kind and x["variant"] == vname])
                print(f"  [{kind}/{vname}] MAE={r['mae'].mean():.3f} >p90={r['mae_gt_p90'].mean():.3f}", flush=True)
    res = pd.DataFrame(rows)
    res_path = os.path.join(MODEL_DIR, f"validate_fire_wide_features_cv_{TIMESTAMP}.csv")
    res.to_csv(res_path, index=False)

    fm = res.groupby(["fold", "model", "variant"])[["mae", "mae_gt_p90"]].mean().reset_index()
    fA, fC = cv.BURNING_FOLDS

    def get(kind, variant):
        s = fm[(fm["model"] == kind) & (fm["variant"] == variant)].set_index("fold")
        return s

    def pct(a, b):
        return (a - b) / b * 100

    print("\n" + "=" * 110 + "\nการตัดสินตามเกณฑ์ที่ตั้งไว้ก่อนรัน (delta% เทียบ base20; ติดลบ = ดีกว่า)\n" + "=" * 110)
    verdict = {}
    for cand in ["+C1 250km", "+C2 upwind", "+C3 frp", "+C4 all", "placebo"]:
        line = {}
        for kind in ("lgb", "xgb"):
            s0, s1 = get(kind, "base20"), get(kind, cand)
            p90_burn = pct(s1.loc[[fA, fC], "mae_gt_p90"].mean(), s0.loc[[fA, fC], "mae_gt_p90"].mean())
            p90_A = pct(s1.loc[fA, "mae_gt_p90"], s0.loc[fA, "mae_gt_p90"])
            p90_C = pct(s1.loc[fC, "mae_gt_p90"], s0.loc[fC, "mae_gt_p90"])
            macro = pct(s1["mae"].mean(), s0["mae"].mean())
            burn = pct(s1.loc[[fA, fC], "mae"].mean(), s0.loc[[fA, fC], "mae"].mean())
            line[kind] = dict(p90_burn=p90_burn, p90_A=p90_A, p90_C=p90_C, macro=macro, burn=burn)
            print(f"  {cand:11s} {kind.upper()}: >p90 ฤดูฝุ่น {p90_burn:+.2f}% (A {p90_A:+.2f}, C {p90_C:+.2f})  "
                  f"macro {macro:+.2f}%  MAE ฤดูฝุ่น {burn:+.2f}%")
        verdict[cand] = line
    print("\nสรุปตามเกณฑ์:")
    for cand in ["+C1 250km", "+C2 upwind", "+C3 frp", "+C4 all"]:
        oks = []
        for kind in ("lgb", "xgb"):
            v, p = verdict[cand][kind], verdict["placebo"][kind]
            ok = (v["p90_burn"] <= -PASS_P90 and v["p90_A"] < 0 and v["p90_C"] < 0 and v["macro"] <= PASS_MACRO
                  and v["burn"] <= PASS_BURN and v["p90_burn"] <= p["p90_burn"] - PASS_PLACEBO_GAP)
            oks.append(ok)
        print(f"  {cand}: {'ผ่าน' if all(oks) else 'ไม่ผ่าน'}  (LGB={oks[0]}, XGB={oks[1]})")
    print(f"\nผลดิบ: {res_path}")


if __name__ == "__main__":
    main()

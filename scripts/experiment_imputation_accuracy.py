"""
Track A ของการทดลอง Data Imputation (คอมเมนต์กรรมการข้อ 4) — "ค่าที่เติมแม่นแค่ไหน"

ซ่อนค่าที่ "วัดได้จริง" (จาก data/raw_pm25_isaan_*.csv ที่ไม่เคย interpolate) แล้วเติมด้วยวิธีต่างตระกูล เทียบกับค่าจริง
สองสถานการณ์:
  interior : ช่องว่างยาว L ชม. ที่รู้ค่าทั้งสองข้าง (L = 1,2,3,6,12,24,48)  — วิธี M0 LOCF, M1 linear (ที่ใช้อยู่), M2 PCHIP,
             M3i diurnal-anomaly interp, M4i neighbor-informed interp, M5i ridge, M6i KNN, M7i MICE(IterativeImputer)
  trailing : สถานีเงียบ j ชม. หลังค่าจริงล่าสุด รู้แค่ฝั่งซ้าย (j = 1,2,3,6,12,24,48) = กรณี live ที่ต้องเลือกระหว่างเติมกับ fallback
             วิธี M0 Persistence/LOCF, M3 diurnal-anomaly, M4 spatial IDW-delta, M5 ridge, M6 KNN, M7 MICE, M8 model-based
             (โมเดลหลัก v3 จากจุดสังเกตล่าสุด, j<=24), M9 ผสม (NNLS ต่อช่วง j)
กฎกันรั่ว: วิธีที่ต้องเรียนรู้ fit บน train 65% ของ "17 จังหวัดที่เทรน" เท่านั้น (จึงทดสอบ zero-shot ของ 3 จังหวัดที่เหลือด้วย)
จูนพารามิเตอร์บน calib 15% ประเมินบน test 20% ครั้งเดียว (split ตามเวลาต่อจังหวัดจาก train_multiprovince_models.split_train_calib_test)
ข้อมูลจังหวัดข้างเคียงใช้ "ค่าที่วัดได้จริง ณ เวลานั้น" ไม่ interpolate ข้ามอนาคต; ไม่มีข้างเคียงที่ใช้ได้ -> ตกไป M3 (เรียงลำดับตายตัว)
ไฟล์นี้ยังเป็นโมดูลให้ experiment_imputation_downstream.py / test_imputation_methods.py โหลดผ่าน importlib (ฟังก์ชัน load_panels,
make_feats, fit_trailing, impute_trailing ...)
รัน: python scripts/experiment_imputation_accuracy.py [--quick]   (--quick = ตัวอย่างน้อยลง สำหรับ smoke test เท่านั้น)
"""
import glob
import importlib.util
import json
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.optimize import nnls
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import BayesianRidge, Ridge
from sklearn.preprocessing import StandardScaler

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
MODEL_DIR = os.path.join(_ROOT, "models")
MODEL_TAG = "20260919_203455"  # โมเดล v3 ที่ deploy (ใช้กับ M8 และ Track B)
SEED = 42
QUICK = "--quick" in sys.argv
BURNING = {2, 3, 4}
K_LIST = [1, 2, 3, 6, 12, 24, 48]  # ความยาวช่องว่าง L (interior) และระยะเงียบ j (trailing)
N_NEIGHBORS = 5  # เพื่อนบ้านใกล้สุดที่พิจารณา (ใช้ที่ "ใช้ได้" ไม่เกิน 3 จังหวัด ตามนโยบายเดิม NEIGHBOR_AVG_K)
PROFILE_DAYS = 14
MIN_HISTORY = 24 * (PROFILE_DAYS + 3)  # ต้องมีประวัติพอสำหรับโปรไฟล์รายชั่วโมง


def load_module(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def latest_raw_path():
    files = sorted(glob.glob(os.path.join(_ROOT, "data", "raw_pm25_isaan_2*.csv")))
    if not files:
        raise SystemExit("ยังไม่มี data/raw_pm25_isaan_*.csv — รัน scripts/fetch_raw_pm25_isaan.py ก่อน")
    return files[-1]


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


# ============================================================ ข้อมูล
def load_panels(need_full_dataset=False):
    """panel รายชั่วโมง (T x P): obs = ค่าดิบที่วัดได้จริง (NaN = ไม่มี), filled = pm25 ใน dataset เทรน (เติมช่องว่าง <=48 ชม.แล้ว
    เหมือนที่โมเดล/ live เห็น), weather (interpolate แล้ว — ใช้เป็น covariate), ranking เพื่อนบ้านตามระยะทางจากทั้ง 20 จังหวัด,
    ขอบ train/calib/test ต่อจังหวัด (index ของกริดเวลา)"""
    tm = load_module("tm", "scripts/train_multiprovince_models.py")
    bm_window = (pd.Timestamp("2024-01-01 00:00:00", tz="UTC"), pd.Timestamp("2026-07-02 00:00:00", tz="UTC"))
    ds = pd.read_csv(tm.DATA_PATH)
    ds["datetime_utc"] = pd.to_datetime(ds["datetime_utc"], utc=True)
    provs = sorted(ds["province_en"].unique())
    grid = pd.date_range(bm_window[0], bm_window[1], freq="h", tz="UTC")  # รวมปลายช่วง เหมือน build_one_station
    raw = pd.read_csv(latest_raw_path())
    raw["datetime_utc"] = pd.to_datetime(raw["datetime_utc"], utc=True)
    obs = raw.pivot_table(index="datetime_utc", columns="province_en", values="pm25").reindex(index=grid, columns=provs)
    filled = ds.pivot_table(index="datetime_utc", columns="province_en", values="pm25").reindex(index=grid, columns=provs)
    wx = {c: ds.pivot_table(index="datetime_utc", columns="province_en", values=c).reindex(index=grid, columns=provs)
          .interpolate(limit_direction="both") for c in ("temp", "humidity", "wind_speed")}
    reg = {q["province_en"]: q for q in json.load(open(os.path.join(_ROOT, "data", "province_station_registry.json"),
                                                        encoding="utf-8"))["provinces"]}
    nbr_idx = np.zeros((len(provs), N_NEIGHBORS), dtype=int)
    nbr_dist = np.zeros((len(provs), N_NEIGHBORS))
    for i, p in enumerate(provs):
        others = sorted((q for q in provs if q != p),
                        key=lambda q: haversine_km(reg[p]["lat"], reg[p]["lon"], reg[q]["lat"], reg[q]["lon"]))[:N_NEIGHBORS]
        nbr_idx[i] = [provs.index(q) for q in others]
        nbr_dist[i] = [haversine_km(reg[p]["lat"], reg[p]["lon"], reg[q]["lat"], reg[q]["lon"]) for q in others]
    trains, calibs, tests = tm.split_train_calib_test(ds)
    bounds = {}
    for i, p in enumerate(provs):
        tr_end = grid.get_loc(trains[trains["province_en"] == p]["datetime_utc"].max())
        ca_end = grid.get_loc(calibs[calibs["province_en"] == p]["datetime_utc"].max())
        te_end = grid.get_loc(tests[tests["province_en"] == p]["datetime_utc"].max())
        tr_start = grid.get_loc(trains[trains["province_en"] == p]["datetime_utc"].min())
        bounds[i] = {"train": (max(tr_start, MIN_HISTORY), tr_end), "calib": (tr_end + 1, ca_end), "test": (ca_end + 1, te_end)}
    zero_shot = {p: bool(ds[ds["province_en"] == p]["is_zero_shot"].iloc[0]) for p in provs}
    P = dict(tm=tm, grid=grid, provs=provs, obs=obs.values, filled=filled.values,
             temp=wx["temp"].values, humidity=wx["humidity"].values, wind=wx["wind_speed"].values,
             nbr_idx=nbr_idx, nbr_dist=nbr_dist, bounds=bounds, zero_shot=zero_shot,
             hour=grid.hour.values, month=grid.month.values, T=len(grid))
    if need_full_dataset:
        P["ds"] = ds
    return P


# ============================================================ การสุ่มจุดทดสอบ
def _run_valid(valid, length):
    """True ที่ index t ถ้า valid[t..t+length-1] เป็น True ทั้งหมด (vectorized ด้วย cumsum)"""
    cs = np.concatenate([[0], np.cumsum(valid.astype(int))])
    n = len(valid)
    out = np.zeros(n, dtype=bool)
    m = n - length + 1
    if m > 0:
        out[:m] = (cs[length:length + m] - cs[:m]) == length
    return out


def sample_trailing(P, split, j, n_per_prov, rng, provinces=None):
    """คืน (p, t0) ที่ obs[t0] และ obs[t0+j] วัดได้จริง และ t0+j อยู่ใน split เดียวกัน"""
    rows_p, rows_t = [], []
    for p in (provinces if provinces is not None else range(len(P["provs"]))):
        lo, hi = P["bounds"][p][split]
        valid = ~np.isnan(P["obs"][:, p])
        ok = valid.copy()
        ok[:len(ok) - j] &= valid[j:]
        ok[len(ok) - j:] = False
        idx = np.where(ok)[0]
        idx = idx[(idx >= lo) & (idx + j <= hi)]
        if len(idx) == 0:
            continue
        pick = rng.choice(idx, size=min(n_per_prov, len(idx)), replace=False)
        rows_p.append(np.full(len(pick), p)); rows_t.append(pick)
    if not rows_p:
        return np.array([], dtype=int), np.array([], dtype=int)
    return np.concatenate(rows_p), np.concatenate(rows_t)


def sample_interior(P, split, L, n_per_prov, rng, provinces=None):
    """(p, t0) ที่ obs[t0], obs[t0+1..t0+L] (เฉลย), obs[t0+L+1] (สมอขวา) วัดได้จริงทั้งหมด"""
    rows_p, rows_t = [], []
    for p in (provinces if provinces is not None else range(len(P["provs"]))):
        lo, hi = P["bounds"][p][split]
        valid = ~np.isnan(P["obs"][:, p])
        ok = _run_valid(valid, L + 2)
        idx = np.where(ok)[0]
        idx = idx[(idx >= lo) & (idx + L + 1 <= hi)]
        if len(idx) == 0:
            continue
        pick = rng.choice(idx, size=min(n_per_prov, len(idx)), replace=False)
        rows_p.append(np.full(len(pick), p)); rows_t.append(pick)
    if not rows_p:
        return np.array([], dtype=int), np.array([], dtype=int)
    return np.concatenate(rows_p), np.concatenate(rows_t)


# ============================================================ feature ของแต่ละ (ตัวอย่าง, ตำแหน่ง j)
def _profile(P, p, idx_target, dmin):
    """ค่าเฉลี่ยชั่วโมงเดียวกันของวันก่อนหน้า PROFILE_DAYS วัน (วันที่ใหม่สุดคือ dmin วันก่อน idx_target) จาก obs ดิบ"""
    d = dmin[:, None] + np.arange(PROFILE_DAYS)[None, :]
    idx = idx_target[:, None] - 24 * d
    idx = np.clip(idx, 0, P["T"] - 1)
    vals = P["obs"][idx, p[:, None]]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(vals, axis=1)


def make_feats(P, p, t0, j, L=None):
    """feature ต่อแถว (p, t0, j): ตำแหน่ง tau = t0 + j. L != None = interior (มีสมอขวา t1 = t0 + L + 1)
    ทุกอย่างของสถานีตัวเองใช้เฉพาะข้อมูล <= t0 (+ สมอขวา t1 ใน interior เท่านั้น) ส่วนเพื่อนบ้านใช้ค่าที่วัดได้จริง ณ tau"""
    obs = P["obs"]
    n = len(p)
    tau = t0 + j
    F = {"p": p, "t0": t0, "j": j, "tau": tau}
    F["x0"] = obs[t0, p]
    nb = P["nbr_idx"][p]
    w = 1.0 / np.maximum(P["nbr_dist"][p], 1.0) ** 2
    nb_tau, nb_t0 = obs[tau[:, None], nb], obs[t0[:, None], nb]
    valid = ~np.isnan(nb_tau) & ~np.isnan(nb_t0)
    valid &= np.cumsum(valid, axis=1) <= 3
    ws = np.where(valid, w, 0.0)
    wsum = ws.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        F["dnb"] = np.where(wsum > 0, (ws * (np.nan_to_num(nb_tau) - np.nan_to_num(nb_t0))).sum(1) / wsum, np.nan)
        F["nb_now"] = np.where(wsum > 0, (ws * np.nan_to_num(nb_tau)).sum(1) / wsum, np.nan)
        F["nb_0"] = np.where(wsum > 0, (ws * np.nan_to_num(nb_t0)).sum(1) / wsum, np.nan)
    F["has_nb"] = wsum > 0
    dmin = np.maximum(1, np.ceil(j / 24.0).astype(int))
    F["prof_tau"] = _profile(P, p, tau, dmin)
    F["prof_0"] = _profile(P, p, t0, np.ones(n, dtype=int))
    F["hs"] = np.sin(2 * np.pi * P["hour"][tau] / 24)
    F["hc"] = np.cos(2 * np.pi * P["hour"][tau] / 24)
    F["temp"], F["hum"], F["wind"] = P["temp"][tau, p], P["humidity"][tau, p], P["wind"][tau, p]
    if L is not None:
        t1 = t0 + L + 1
        F["L"] = np.full(n, L)
        F["x1"] = obs[t1, p]
        frac = j / (L + 1.0)
        F["frac"] = frac
        F["lin"] = F["x0"] + (F["x1"] - F["x0"]) * frac
        nb_t1 = obs[t1[:, None], nb]
        v2 = ~np.isnan(nb_tau) & ~np.isnan(nb_t0) & ~np.isnan(nb_t1)
        v2 &= np.cumsum(v2, axis=1) <= 3
        ws2 = np.where(v2, w, 0.0)
        wsum2 = ws2.sum(1)
        lin_nb = np.nan_to_num(nb_t0) + (np.nan_to_num(nb_t1) - np.nan_to_num(nb_t0)) * frac[:, None]
        with np.errstate(invalid="ignore", divide="ignore"):
            F["nb_dev"] = np.where(wsum2 > 0, (ws2 * (np.nan_to_num(nb_tau) - lin_nb)).sum(1) / wsum2, np.nan)
        F["has_nb_i"] = wsum2 > 0
        F["prof_1"] = _profile(P, p, t1, np.ones(n, dtype=int))
        F["anom0"] = F["x0"] - F["prof_0"]
        F["anom1"] = F["x1"] - F["prof_1"]
    return F


# ============================================================ วิธีเติมค่าแบบไม่ต้องเรียนรู้
def clip0(v):
    return np.maximum(v, 0.0)


def m0_locf(F):
    return F["x0"].copy()


def m1_linear(F):
    return F["lin"].copy()


def m2_pchip(P, F):
    """PCHIP ผ่านจุดสังเกตจริง 3 ชม.ก่อน t0 และ 3 ชม.หลัง t1 (เฉพาะที่วัดได้) — วนต่อตัวอย่าง"""
    out = np.empty(len(F["p"]))
    obs = P["obs"]
    for i in range(len(out)):
        p, t0, j, L = F["p"][i], F["t0"][i], F["j"][i], F["L"][i]
        t1 = t0 + L + 1
        xs = [t for t in list(range(t0 - 3, t0 + 1)) + list(range(t1, t1 + 4)) if 0 <= t < len(obs) and not np.isnan(obs[t, p])]
        if len(xs) < 2:
            out[i] = F["lin"][i]
            continue
        out[i] = PchipInterpolator(xs, obs[xs, p])(t0 + j)
    return out


def m3_diurnal(F, tau_h):
    """โปรไฟล์รายชั่วโมงของวันก่อนๆ + ส่วนเบี่ยงเบน ณ t0 ที่จางลงแบบ exp(-j/tau_h) (trailing)"""
    prof_tau = np.where(np.isnan(F["prof_tau"]), F["x0"], F["prof_tau"])
    anom0 = F["x0"] - np.where(np.isnan(F["prof_0"]), F["x0"], F["prof_0"])
    return prof_tau + anom0 * np.exp(-F["j"] / tau_h)


def m3i_diurnal(F):
    """interior: โปรไฟล์ + ส่วนเบี่ยงเบนที่ interpolate เชิงเส้นระหว่างสองสมอ"""
    prof_tau = np.where(np.isnan(F["prof_tau"]), F["lin"], F["prof_tau"])
    a0 = np.where(np.isnan(F["anom0"]), 0.0, F["anom0"])
    a1 = np.where(np.isnan(F["anom1"]), 0.0, F["anom1"])
    return prof_tau + a0 + (a1 - a0) * F["frac"]


def m4_spatial(F, base=None):
    """trailing: x(t0) + IDW-delta ของเพื่อนบ้าน; ไม่มีเพื่อนบ้าน -> เรียกผู้ใช้ให้ fallback (คืน NaN)"""
    return np.where(F["has_nb"], F["x0"] + F["dnb"], np.nan)


def m4i_spatial(F):
    """interior: linear + ส่วนที่เพื่อนบ้านเบี่ยงจาก linear ของตัวเอง (neighbor-informed interpolation)"""
    return np.where(F["has_nb_i"], F["lin"] + F["nb_dev"], np.nan)


# ============================================================ วิธีที่ต้องเรียนรู้ (ridge / KNN / MICE)
def matrix_trailing(F):
    return np.column_stack([F["x0"], F["nb_now"], F["dnb"], F["nb_0"], F["j"], np.log1p(F["j"]), F["hs"], F["hc"],
                            F["temp"], F["hum"], F["wind"],
                            np.where(np.isnan(F["prof_tau"]), F["x0"], F["prof_tau"]) - F["x0"]])


def matrix_trailing_lite(F):
    """H-lite: เฉพาะ feature ที่ live มีอยู่แล้ว (ค่าล่าสุด, เพื่อนบ้าน ณ ตำแหน่งที่เติมและ ณ t0, ระยะเงียบ, ชั่วโมง UTC ของเวลาเป้าหมาย)
    ตัดสภาพอากาศรายชั่วโมงย้อนหลังและโปรไฟล์ 14 วันที่ live ไม่มี"""
    return np.column_stack([F["x0"], F["nb_now"], F["dnb"], F["nb_0"], F["j"], np.log1p(F["j"]), F["hs"], F["hc"]])


def matrix_interior(F):
    return np.column_stack([F["x0"], F["x1"], F["lin"], F["nb_dev"], F["frac"], F["L"], F["hs"], F["hc"],
                            F["temp"], F["hum"], F["wind"],
                            np.where(np.isnan(F["prof_tau"]), F["lin"], F["prof_tau"]) - F["lin"]])


def truth(P, F):
    return P["obs"][F["tau"], F["p"]]


def fit_learned(P, F_train, kind, mat_fn, base_key, alpha=10.0, knn_k=30, rng=None):
    """เรียนรู้จากแถว train: ridge ทำนาย (x(tau) - base) ; KNN/MICE ต่อคอลัมน์เป้าหมายเข้าเมทริกซ์แล้วเติมค่า"""
    ok = F_train["has_nb"] if base_key == "x0" else F_train["has_nb_i"]
    y = truth(P, F_train)
    X = mat_fn(F_train)
    ok = ok & ~np.isnan(y) & ~np.isnan(X).any(axis=1)
    X, y, base = X[ok], y[ok], F_train[base_key][ok]
    sc = StandardScaler().fit(X)
    if kind == "ridge":
        model = Ridge(alpha=alpha).fit(sc.transform(X), y - base)
        return {"kind": kind, "sc": sc, "model": model}
    n_fit = min(len(X), 15000 if kind == "knn" else 30000)
    sel = (rng or np.random.default_rng(SEED)).choice(len(X), size=n_fit, replace=False)
    Z = np.column_stack([sc.transform(X[sel]), (y[sel] - base[sel]) / (y[sel].std() + 1e-9)])
    scale = float(y[sel].std() + 1e-9)
    if kind == "knn":
        imp = KNNImputer(n_neighbors=knn_k, weights="distance").fit(Z)
    else:
        imp = IterativeImputer(estimator=BayesianRidge(), max_iter=8, tol=1e-2, random_state=SEED,
                               initial_strategy="mean").fit(Z)
    return {"kind": kind, "sc": sc, "model": imp, "scale": scale}


def apply_learned(P, fitted, F, mat_fn, base_key, chunk=20000):
    X = mat_fn(F)
    base = F[base_key]
    ok = (F["has_nb"] if base_key == "x0" else F["has_nb_i"]) & ~np.isnan(X).any(axis=1) & ~np.isnan(base)
    out = np.full(len(base), np.nan)
    if not ok.any():
        return out
    Xs = fitted["sc"].transform(X[ok])
    if fitted["kind"] == "ridge":
        out[ok] = base[ok] + fitted["model"].predict(Xs)
        return out
    pred = np.empty(len(Xs))
    for a in range(0, len(Xs), chunk):
        Z = np.column_stack([Xs[a:a + chunk], np.full(len(Xs[a:a + chunk]), np.nan)])
        pred[a:a + chunk] = fitted["model"].transform(Z)[:, -1] * fitted["scale"]
    out[ok] = base[ok] + pred
    return out


# ============================================================ M8: โมเดลหลักจากจุดสังเกตล่าสุด
def load_v3_models():
    return {t: pickle.load(open(os.path.join(MODEL_DIR, f"{t}_pm25_multiprovince_residual_bagged_quantile_{MODEL_TAG}.pkl"), "rb"))
            for t in ("lgb", "xgb")}


def m8_model_from_origin(P, ds_index, models, F, kind="lgb"):
    """พยากรณ์ x(t0+j) ด้วยโมเดลหลักที่ origin t0 (horizon j<=24) — ใช้ feature ของ dataset ณ (p, t0) ตามที่ live ทำ"""
    ds = P["ds"]
    feats = P["tm"].FEATURE_COLS
    out = np.full(len(F["p"]), np.nan)
    for j in np.unique(F["j"]):
        if j > 24:
            continue
        sel = np.where(F["j"] == j)[0]
        keys = pd.MultiIndex.from_arrays([np.array(P["provs"])[F["p"][sel]], P["grid"][F["t0"][sel]]])
        rows = ds_index.reindex(keys)
        good = ~rows["_row"].isna().values
        if not good.any():
            continue
        X = ds.iloc[rows["_row"].values[good].astype(int)][feats]
        base = ds.iloc[rows["_row"].values[good].astype(int)]["pm25"].values
        out[sel[good]] = models[kind]["models"][int(j)][0.5].predict(X) + base
    return out


def build_ds_index(P):
    ds = P["ds"]
    idx = pd.DataFrame({"_row": np.arange(len(ds))}, index=pd.MultiIndex.from_arrays([ds["province_en"].values, ds["datetime_utc"]]))
    return idx


# ============================================================ ชุดวิธี trailing ที่ fit แล้ว (ใช้ซ้ำใน downstream)
def fit_trailing(P, rng, n_train_rows=60000, tune=None, fast=False):
    """fit ridge/KNN/MICE + เลือก tau ของ M3 ฯลฯ บน train (17 จังหวัดที่เทรน) — tune = dict พารามิเตอร์จาก calib (ถ้ามี)"""
    tune = tune or {"tau_h": 12.0, "ridge_alpha": 10.0, "knn_k": 30}
    trained = [i for i, p in enumerate(P["provs"]) if not P["zero_shot"][p]]
    ps, ts, js = [], [], []
    per = max(n_train_rows // (len(trained) * len(K_LIST)), 20)
    for j in K_LIST + [4, 5, 8, 10, 16, 20, 30, 36]:
        p_, t_ = sample_trailing(P, "train", j, per, rng, trained)
        ps.append(p_); ts.append(t_); js.append(np.full(len(p_), j))
    p, t0, j = np.concatenate(ps), np.concatenate(ts), np.concatenate(js)
    F = make_feats(P, p, t0, j)
    fitted = {
        "ridge": fit_learned(P, F, "ridge", matrix_trailing, "x0", alpha=tune["ridge_alpha"]),
        "ridge_lite": fit_learned(P, F, "ridge", matrix_trailing_lite, "x0", alpha=tune.get("ridge_alpha_lite", tune["ridge_alpha"])),
    }
    if not fast:  # KNN/MICE ช้าและไม่ใช่ส่วนของ H-lite — ข้ามได้เมื่อจูน/ส่งออกเฉพาะ H-lite
        fitted["knn"] = fit_learned(P, F, "knn", matrix_trailing, "x0", knn_k=tune["knn_k"], rng=rng)
        fitted["mice"] = fit_learned(P, F, "mice", matrix_trailing, "x0", rng=rng)
    return {"fitted": fitted, "tune": tune}


def impute_trailing(P, model, F, methods=("M0", "M3", "M4", "M5", "M6", "M7", "H", "HL")):
    """คืน dict {ชื่อวิธี: ค่าที่เติมต่อแถว} วิธีที่ต้องการเพื่อนบ้านแต่ไม่มี -> ตกไปใช้ M3 (ลำดับ fallback ตายตัว) ห้ามคืน NaN
    (ยกเว้น HL = H-lite ที่ตกไปใช้ Persistence x0 แทน M3 ตรงกับสิ่งที่ live ทำ)"""
    tune = model["tune"]
    base3 = m3_diurnal(F, tune["tau_h"])
    out = {}
    if "M0" in methods:
        out["M0"] = clip0(m0_locf(F))
    if "M3" in methods:
        out["M3"] = clip0(base3)
    if "M4" in methods:
        out["M4"] = clip0(np.where(np.isnan(m4_spatial(F)), base3, m4_spatial(F)))
    for name, key in (("M5", "ridge"), ("M6", "knn"), ("M7", "mice")):
        if name in methods:
            v = apply_learned(P, model["fitted"][key], F, matrix_trailing, "x0")
            out[name] = clip0(np.where(np.isnan(v), base3, v))
    if "H" in methods:
        # วิธีหลักที่เสนอ (H = hybrid): IDW-difference เมื่อ j <= switch_j แล้วเปลี่ยนเป็น Ridge (จุดสลับเลือกจาก calib)
        m4 = np.where(np.isnan(m4_spatial(F)), base3, m4_spatial(F))
        v5 = apply_learned(P, model["fitted"]["ridge"], F, matrix_trailing, "x0")
        m5 = np.where(np.isnan(v5), base3, v5)
        out["H"] = clip0(np.where(F["j"] <= tune.get("switch_j", 3), m4, m5))
    if "HL" in methods:
        # H-lite = วิธีที่นำไปใช้ใน app.py: IDW-difference (j <= switch_j_lite) แล้ว Ridge-lite; ไม่มีเพื่อนบ้านที่ใช้ได้ -> Persistence
        m4l = np.where(np.isnan(m4_spatial(F)), F["x0"], m4_spatial(F))
        vl = apply_learned(P, model["fitted"]["ridge_lite"], F, matrix_trailing_lite, "x0")
        m5l = np.where(np.isnan(vl), F["x0"], vl)
        out["HL"] = clip0(np.where(F["j"] <= tune.get("switch_j_lite", 3), m4l, m5l))
    return out


# ============================================================ metric
def metrics(y, yhat):
    e = yhat - y
    return {"mae": float(np.mean(np.abs(e))), "rmse": float(np.sqrt(np.mean(e ** 2))), "bias": float(np.mean(e)), "n": int(len(e))}


def report_rows(scenario, method, size, y, yhat, months, p90):
    rows = []
    for grp, sel in (("all", np.ones(len(y), bool)), ("burning", np.isin(months, list(BURNING))),
                     ("non_burning", ~np.isin(months, list(BURNING))), ("gt_p90", y > p90)):
        if sel.sum() == 0:
            continue
        rows.append({"scenario": scenario, "method": method, "size": size, "group": grp, **metrics(y[sel], yhat[sel])})
    return rows


# ============================================================ main
def main():
    rng = np.random.default_rng(SEED)
    P = load_panels(need_full_dataset=True)
    provs = P["provs"]
    n_te = 40 if QUICK else 200      # ต่อจังหวัดต่อ L/j (test)
    n_ca = 15 if QUICK else 60       # calib
    print(f"panel: {P['T']} ชั่วโมง x {len(provs)} จังหวัด  QUICK={QUICK}")
    trained_idx = [i for i, p in enumerate(provs) if not P["zero_shot"][p]]
    p90 = float(np.nanpercentile(P["obs"][:, trained_idx], 90))

    # ---------- จูนบน calib (trailing): tau ของ M3, alpha ของ ridge, k ของ KNN ----------
    print("จูนพารามิเตอร์บน calib ...", flush=True)
    calib_sets = []
    for j in K_LIST:
        p_, t_ = sample_trailing(P, "calib", j, n_ca, rng)
        if len(p_):
            calib_sets.append((j, p_, t_))
    def calib_mae(fn):
        errs = []
        for j, p_, t_ in calib_sets:
            F = make_feats(P, p_, t_, np.full(len(p_), j))
            errs.append(np.mean(np.abs(fn(F) - truth(P, F))))
        return float(np.mean(errs))
    tau_best = min([3, 6, 12, 24, 48], key=lambda tau: calib_mae(lambda F: clip0(m3_diurnal(F, tau))))
    best = {"tau_h": float(tau_best), "ridge_alpha": 10.0, "knn_k": 30}
    for a in ([1.0, 10.0, 100.0] if not QUICK else [10.0]):
        cand = fit_trailing(P, np.random.default_rng(SEED), 20000 if QUICK else 60000, {**best, "ridge_alpha": a})
        m = calib_mae(lambda F: impute_trailing(P, cand, F, ("M5",))["M5"])
        if a == 1.0 or m < best.get("_m", 1e9):
            best["ridge_alpha"], best["_m"] = a, m
    best.pop("_m", None)
    print("  พารามิเตอร์ที่เลือก:", best, flush=True)
    model = fit_trailing(P, np.random.default_rng(SEED), 20000 if QUICK else 60000, best)

    # ---------- H: เลือกจุดสลับ IDW-difference -> Ridge จาก calib เท่านั้น (ไม่ดู test) ----------
    switch_mae = {}
    for sj in [0, 1, 2, 3, 6, 12, 24]:
        best["switch_j"] = sj
        errs = []
        for j, p_, t_ in calib_sets:
            F = make_feats(P, p_, t_, np.full(len(p_), j))
            errs.append(np.mean(np.abs(impute_trailing(P, model, F, ("H",))["H"] - truth(P, F))))
        switch_mae[sj] = float(np.mean(errs))
    best["switch_j"] = min(switch_mae, key=switch_mae.get)
    best["switch_mae_calib"] = switch_mae
    print("  จุดสลับ H เลือกจาก calib:", best["switch_j"], {k: round(v, 3) for k, v in switch_mae.items()}, flush=True)

    # ---------- H-lite (HL): จูน alpha ของ Ridge-lite แล้วเลือกจุดสลับ จาก calib เท่านั้น ----------
    n_fit = 20000 if QUICK else 60000
    alpha_lite_mae = {}
    for a in ([10.0] if QUICK else [1.0, 10.0, 100.0, 1000.0]):
        cand = fit_trailing(P, np.random.default_rng(SEED), n_fit, {**best, "ridge_alpha_lite": a}, fast=True)
        errs = []
        for j, p_, t_ in calib_sets:
            F = make_feats(P, p_, t_, np.full(len(p_), j))
            tune_a = {**best, "ridge_alpha_lite": a, "switch_j_lite": 0}  # switch 0 = ใช้ Ridge-lite ล้วน
            errs.append(np.mean(np.abs(impute_trailing(P, {**cand, "tune": tune_a}, F, ("HL",))["HL"] - truth(P, F))))
        alpha_lite_mae[a] = float(np.mean(errs))
    best["ridge_alpha_lite"] = min(alpha_lite_mae, key=alpha_lite_mae.get)
    model["fitted"]["ridge_lite"] = fit_trailing(P, np.random.default_rng(SEED), n_fit, best, fast=True)["fitted"]["ridge_lite"]
    switch_lite = {}
    for sj in [0, 1, 2, 3, 6, 12, 24]:
        best["switch_j_lite"] = sj
        errs = []
        for j, p_, t_ in calib_sets:
            F = make_feats(P, p_, t_, np.full(len(p_), j))
            errs.append(np.mean(np.abs(impute_trailing(P, model, F, ("HL",))["HL"] - truth(P, F))))
        switch_lite[sj] = float(np.mean(errs))
    best["switch_j_lite"] = min(switch_lite, key=switch_lite.get)
    best["ridge_alpha_lite_calib"] = alpha_lite_mae
    best["switch_mae_lite_calib"] = switch_lite
    print("  H-lite: alpha =", best["ridge_alpha_lite"], " จุดสลับ =", best["switch_j_lite"],
          {k: round(v, 3) for k, v in switch_lite.items()}, flush=True)

    # ---------- M9: น้ำหนักผสม NNLS ต่อช่วง j จาก calib ----------
    cand_names = ["M3", "M4", "M5", "M6"]
    blend = {}
    for jb in K_LIST:
        Xs, ys = [], []
        for j, p_, t_ in calib_sets:
            if j != jb:
                continue
            F = make_feats(P, p_, t_, np.full(len(p_), j))
            out = impute_trailing(P, model, F, cand_names)
            Xs.append(np.column_stack([out[c] for c in cand_names])); ys.append(truth(P, F))
        if Xs:
            w, _ = nnls(np.vstack(Xs), np.concatenate(ys))
            blend[jb] = w
    print("  น้ำหนัก M9 ต่อ j:", {k: np.round(v, 2).tolist() for k, v in blend.items()}, flush=True)
    # เก็บพารามิเตอร์ที่จูนบน calib ให้ experiment_imputation_downstream.py ใช้ต่อ (ห้ามจูนซ้ำบน test)
    tuned_path = os.path.join(MODEL_DIR, f"imputation_tuned_params_{TIMESTAMP}{'_quick' if QUICK else ''}.json")
    with open(tuned_path, "w", encoding="utf-8") as f:
        json.dump({"tune": best, "blend_candidates": cand_names, "blend": {str(k): v.tolist() for k, v in blend.items()},
                   "k_list": K_LIST, "seed": SEED}, f, ensure_ascii=False, indent=1)

    # ---------- ประเมิน trailing บน test ----------
    models_v3 = load_v3_models()
    ds_index = build_ds_index(P)
    rows = []
    prov_rows = []
    for j in K_LIST:
        p_, t_ = sample_trailing(P, "test", j, n_te, rng)
        F = make_feats(P, p_, t_, np.full(len(p_), j))
        y = truth(P, F)
        months = P["month"][F["tau"]]
        out = impute_trailing(P, model, F)
        w = blend.get(j)
        if w is not None:
            out["M9"] = clip0(np.column_stack([out[c] for c in cand_names]) @ w)
        if j <= 24:
            m8 = m8_model_from_origin(P, ds_index, models_v3, F, "lgb")
            out["M8"] = clip0(np.where(np.isnan(m8), out["M3"], m8))
        for name, v in out.items():
            rows += report_rows("trailing", name, int(j), y, v, months, p90)
            if j in (12, 24):
                for pi in np.unique(F["p"]):
                    s = F["p"] == pi
                    prov_rows.append({"scenario": "trailing", "method": name, "size": int(j), "province": provs[pi],
                                      "zero_shot": P["zero_shot"][provs[pi]], **metrics(y[s], v[s])})
        print(f"  trailing j={j:2d}: n={len(y)}  " + "  ".join(f"{k}={np.mean(np.abs(v - y)):.2f}" for k, v in out.items()), flush=True)

    # ---------- interior ----------
    trained = trained_idx
    for L in K_LIST:
        # ตัวอย่างฝึก interior (ridge/KNN/MICE) จาก train
        ps, ts, js = [], [], []
        for Lk in [L]:
            p_, t_ = sample_interior(P, "train", Lk, 60 if QUICK else 250, rng, trained)
            for jj in range(1, Lk + 1):
                ps.append(p_); ts.append(t_); js.append(np.full(len(p_), jj))
        Ftr = make_feats(P, np.concatenate(ps), np.concatenate(ts), np.concatenate(js), L)
        fit_i = {"ridge": fit_learned(P, Ftr, "ridge", matrix_interior, "lin", alpha=best["ridge_alpha"]),
                 "knn": fit_learned(P, Ftr, "knn", matrix_interior, "lin", knn_k=best["knn_k"], rng=rng),
                 "mice": fit_learned(P, Ftr, "mice", matrix_interior, "lin", rng=rng)}
        p_, t_ = sample_interior(P, "test", L, n_te if L <= 12 else max(n_te // 2, 20), rng)
        pp, tt, jj = [], [], []
        for pos in range(1, L + 1):
            pp.append(p_); tt.append(t_); jj.append(np.full(len(p_), pos))
        F = make_feats(P, np.concatenate(pp), np.concatenate(tt), np.concatenate(jj), L)
        y = truth(P, F)
        months = P["month"][F["tau"]]
        m3i = m3i_diurnal(F)
        out = {"M0": clip0(m0_locf(F)), "M1": clip0(m1_linear(F)), "M2": clip0(m2_pchip(P, F)), "M3i": clip0(m3i),
               "M4i": clip0(np.where(np.isnan(m4i_spatial(F)), F["lin"], m4i_spatial(F)))}
        for name, key in (("M5i", "ridge"), ("M6i", "knn"), ("M7i", "mice")):
            v = apply_learned(P, fit_i[key], F, matrix_interior, "lin")
            out[name] = clip0(np.where(np.isnan(v), F["lin"], v))
        for name, v in out.items():
            rows += report_rows("interior", name, int(L), y, v, months, p90)
        print(f"  interior L={L:2d}: n={len(y)}  " + "  ".join(f"{k}={np.mean(np.abs(v - y)):.2f}" for k, v in out.items()), flush=True)

    res = pd.DataFrame(rows)
    out_path = os.path.join(MODEL_DIR, f"imputation_accuracy_{TIMESTAMP}{'_quick' if QUICK else ''}.csv")
    res.to_csv(out_path, index=False)
    pd.DataFrame(prov_rows).to_csv(os.path.join(MODEL_DIR, f"imputation_accuracy_by_province_{TIMESTAMP}{'_quick' if QUICK else ''}.csv"), index=False)
    print(f"\nบันทึก {out_path}")
    for scen in ("trailing", "interior"):
        t = res[(res.scenario == scen) & (res.group == "all")].pivot(index="method", columns="size", values="mae").round(2)
        print(f"\n=== MAE ของค่าที่เติม ({scen}) — คอลัมน์ = {'j (ชม.หลังค่าจริงล่าสุด)' if scen == 'trailing' else 'L (ความยาวช่องว่าง)'} ===")
        print(t.to_string())


if __name__ == "__main__":
    main()

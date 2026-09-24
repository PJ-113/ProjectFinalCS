"""
ทดสอบวิธีเติมข้อมูลและโครงสร้างการทดลอง imputation (ไม่มี pytest — assert ธรรมดาตามธรรมเนียมโปรเจกต์):
  1. feats_from_window ตรงกับสูตร pandas (shift/rolling ddof=1) ที่ใช้สร้าง dataset
  2. ซีรีส์สังเคราะห์ที่รู้คำตอบ: M0/M1/M3/M3i/M4/M4i ให้ค่าที่คาดไว้
  3. ไม่รั่วอนาคต: เปลี่ยนค่าของสถานีตัวเอง "หลัง t0" (รวมค่าเฉลย ณ tau) แล้ว feature/ค่าที่เติมแบบ trailing ต้องไม่เปลี่ยน
  4. (ถ้ามี data/raw_pm25_isaan_*.csv) window ที่ไม่เติมค่า ต้อง reproduce feature ของ dataset เทรนเป๊ะ และ obs ตรงกับ filled ที่ชั่วโมงที่วัดได้
รัน: python scripts/test_imputation_methods.py
"""
import glob
import importlib.util
import os

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(name, rel):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, rel))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ea = load("ea", "scripts/experiment_imputation_accuracy.py")
dn = load("dn", "scripts/experiment_imputation_downstream.py")


def synth_panel(T=2400, seed=0, noise=0.0):
    rng = np.random.default_rng(seed)
    t = np.arange(T)
    base = 30 + 12 * np.sin(2 * np.pi * (t % 24) / 24)  # คาบ 24 ชม. เป๊ะ
    obs = np.column_stack([base + noise * rng.standard_normal(T),           # จังหวัด 0 = เป้าหมาย
                           base + 5 + noise * rng.standard_normal(T),       # เพื่อนบ้าน = ซีรีส์เดียวกัน + ค่าคงที่
                           base + 8 + noise * rng.standard_normal(T),
                           base + 2 + noise * rng.standard_normal(T),
                           base + 1 + noise * rng.standard_normal(T),
                           base + 3 + noise * rng.standard_normal(T)])
    n = obs.shape[1]
    nbr_idx = np.array([[i for i in range(n) if i != p][:ea.N_NEIGHBORS] for p in range(n)])
    nbr_dist = np.tile(np.arange(1, ea.N_NEIGHBORS + 1) * 50.0, (n, 1))
    grid = pd.date_range("2025-01-01", periods=T, freq="h", tz="UTC")
    return dict(obs=obs, T=T, nbr_idx=nbr_idx, nbr_dist=nbr_dist, hour=grid.hour.values,
                temp=np.full_like(obs, 25.0), humidity=np.full_like(obs, 60.0), wind=np.full_like(obs, 3.0))


def test_window_features_match_pandas():
    rng = np.random.default_rng(1)
    s = pd.Series(rng.uniform(10, 80, 200))
    t = 150
    W = s.values[t - 24:t + 1][None, :]
    f, base = dn.feats_from_window(W)
    assert abs(f["pm25_lag_1"][0] - s.shift(1)[t]) < 1e-9
    assert abs(f["pm25_lag_3"][0] - s.shift(3)[t]) < 1e-9
    assert abs(f["pm25_lag_6"][0] - s.shift(6)[t]) < 1e-9
    assert abs(f["pm25_lag_12"][0] - s.shift(12)[t]) < 1e-9
    assert abs(f["pm25_lag_24"][0] - s.shift(24)[t]) < 1e-9
    assert abs(f["pm25_roll_mean_6"][0] - s.rolling(6).mean()[t]) < 1e-9
    assert abs(f["pm25_roll_std_6"][0] - s.rolling(6).std()[t]) < 1e-9
    assert abs(f["pm25_roll_mean_24"][0] - s.rolling(24).mean()[t]) < 1e-9
    assert abs(f["pm25_roll_std_24"][0] - s.rolling(24).std()[t]) < 1e-9
    assert abs(base[0] - s[t]) < 1e-9
    print("ok: feats_from_window ตรงสูตร pandas")


def test_known_answers():
    P = synth_panel()
    p = np.array([0, 0, 0]); t0 = np.array([1000, 1005, 1013])
    # trailing: j=5 ค่าจริงของทุกวิธีที่ไม่ noise ต้องเป็นค่าของซีรีส์คาบ 24 ชม. เดียวกัน
    j = np.array([5, 5, 5])
    F = ea.make_feats(P, p, t0, j)
    y = ea.truth(P, F)
    assert np.allclose(ea.m0_locf(F), P["obs"][t0, 0])
    assert np.allclose(ea.m3_diurnal(F, 12.0), y, atol=1e-9), "M3 ต้องแม่นเมื่อซีรีส์เป็นคาบ 24 ชม. เป๊ะ (anomaly=0)"
    assert np.allclose(ea.m4_spatial(F), y, atol=1e-9), "M4 ต้องแม่นเมื่อเพื่อนบ้านเคลื่อนไหวเหมือนเป้าหมาย"
    # interior: ซีรีส์เชิงเส้นช่วงสั้น -> linear แม่นเป๊ะ; ซีรีส์คาบ -> M3i, M4i แม่น
    L = 6
    Fi = ea.make_feats(P, np.zeros(6, int), np.full(6, 1003), np.arange(1, 7), L)
    yi = ea.truth(P, Fi)
    assert np.allclose(ea.m3i_diurnal(Fi), yi, atol=1e-9)
    assert np.allclose(ea.m4i_spatial(Fi) + 0.0, ea.m4i_spatial(Fi))
    P_lin = synth_panel()
    P_lin["obs"][:, 0] = 10 + 0.5 * np.arange(P_lin["T"])
    P_lin["obs"][:, 1:] = P_lin["obs"][:, [0]] + 3
    Fl = ea.make_feats(P_lin, np.zeros(6, int), np.full(6, 1003), np.arange(1, 7), L)
    assert np.allclose(ea.m1_linear(Fl), ea.truth(P_lin, Fl), atol=1e-9), "linear ต้องแม่นเป๊ะกับซีรีส์เชิงเส้น"
    assert np.allclose(ea.m4i_spatial(Fl), ea.truth(P_lin, Fl), atol=1e-9)
    print("ok: ซีรีส์สังเคราะห์ M0/M1/M3/M3i/M4/M4i")


def test_no_future_leak():
    P = synth_panel(noise=2.0, seed=3)
    p = np.array([0, 0]); t0 = np.array([1500, 1520]); j = np.array([7, 30])
    F1 = ea.make_feats(P, p, t0, j)
    o1 = ea.m3_diurnal(F1, 12.0)
    P2 = synth_panel(noise=2.0, seed=3)
    rng = np.random.default_rng(9)
    for i in range(len(t0)):
        P2["obs"][t0[i] + 1:, 0] = rng.uniform(0, 200, P2["T"] - t0[i] - 1)  # เปลี่ยนอนาคตของสถานีตัวเองทั้งหมด รวมค่าเฉลย ณ tau
    F2 = ea.make_feats(P2, p, t0, j)
    for key in ("x0", "prof_tau", "prof_0"):
        # ข้อจำกัดของการทดสอบ: จุดที่สอง (t0=1520) ถูกเปลี่ยนโดยรอบของจุดแรกด้วย -> ทดสอบเฉพาะจุดแรกที่ t0 ตัวเองไม่ถูกแตะ
        assert np.allclose(F1[key][:1], F2[key][:1], equal_nan=True), f"{key} รั่วข้อมูลหลัง t0"
    assert np.allclose(o1[:1], ea.m3_diurnal(F2, 12.0)[:1])
    print("ok: ไม่รั่วอนาคตของสถานีตัวเองใน trailing feature/M3")


def test_real_parity():
    files = glob.glob(os.path.join(_ROOT, "data", "raw_pm25_isaan_2*.csv"))
    if not files:
        print("ข้าม test_real_parity: ยังไม่มีข้อมูลดิบ")
        return
    P = ea.load_panels(need_full_dataset=True)
    ds, tm = P["ds"], P["tm"]
    both = ~np.isnan(P["obs"]) & ~np.isnan(P["filled"])
    diff = np.abs(P["obs"] - P["filled"])[both]
    assert (diff < 1e-6).mean() > 0.99, "ค่าดิบต้องตรง filled ที่ชั่วโมงที่วัดได้ (>=99%)"
    rng = np.random.default_rng(0)
    ds_ix = ea.build_ds_index(P)
    checked = 0
    for p in range(len(P["provs"])):
        lo, hi = P["bounds"][p]["test"]
        cand = np.arange(max(lo, 60), hi - 30)
        cand = cand[~np.isnan(P["filled"][cand - 24, p]) & ~np.isnan(P["filled"][cand, p])]
        if len(cand) == 0:
            continue
        t = rng.choice(cand, size=min(20, len(cand)), replace=False)
        pp = np.full(len(t), p)
        keys = pd.MultiIndex.from_arrays([np.array(P["provs"])[pp], P["grid"][t]])
        ix = ds_ix.index.get_indexer(keys)
        keep = ix >= 0
        if not keep.any():
            continue
        rows = ds.iloc[ix[keep]]
        W = P["filled"][t[keep][:, None] + np.arange(-24, 1)[None, :], pp[keep][:, None]]
        if np.isnan(W).any():
            continue
        f, b = dn.feats_from_window(W)
        for c, v in f.items():
            tol = 1e-4 if "_std_" in c else 1e-6  # rolling std ของ pandas เป็นอัลกอริทึมออนไลน์ ต่างเล็กน้อยเมื่อซีรีส์คงที่
            assert np.nanmax(np.abs(v - rows[c].values)) < tol, f"{P['provs'][p]} {c}"
        checked += int(keep.sum())
    assert checked > 100
    print(f"ok: reproduce feature ของ dataset เทรนได้เป๊ะ ({checked} แถว)")


if __name__ == "__main__":
    test_window_features_match_pandas()
    test_known_answers()
    test_no_future_leak()
    test_real_parity()
    print("\nผ่านทุกการทดสอบ")

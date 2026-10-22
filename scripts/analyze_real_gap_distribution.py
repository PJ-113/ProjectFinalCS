"""
วิเคราะห์การกระจายตัวของช่องว่าง (gap) ในข้อมูล PM2.5 ดิบ 20 จังหวัด: ความยาวช่องว่าง
(histogram/quantile) และ exposure ของ staleness k (จำนวนชั่วโมงนับจากค่าจริงล่าสุด) แยกตาม
จังหวัด/ฤดูฝุ่น

ผลลัพธ์:
  models/gap_runs_<ts>.csv          ทุกช่องว่าง (province, เริ่ม, ความยาว ชม.)
  models/gap_distribution_<ts>.csv  สรุปการกระจายความยาวช่องว่างและ staleness exposure
"""
import glob
import os

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
BURNING = {2, 3, 4}
K_BINS = [(1, 1), (2, 3), (4, 6), (7, 12), (13, 24), (25, 48), (49, 10**9)]
L_BINS = [(1, 1), (2, 3), (4, 6), (7, 12), (13, 24), (25, 48), (49, 168), (169, 10**9)]


def latest_raw_path():
    files = sorted(f for f in glob.glob(os.path.join(_ROOT, "data", "raw_pm25_isaan_2*.csv")))
    if not files:
        raise SystemExit("ยังไม่มีไฟล์ data/raw_pm25_isaan_*.csv — รัน scripts/fetch_raw_pm25_isaan.py ก่อน")
    return files[-1]


def gap_runs_for(obs_times, grid):
    """ช่องว่าง = ชั่วโมงบนกริดที่ไม่มีค่า ระหว่างค่าจริงชุดแรกและชุดสุดท้ายของสถานี (ไม่นับก่อนเริ่ม/หลังจบข้อมูลของสถานี)"""
    present = pd.Series(grid.isin(obs_times), index=grid)
    first, last = obs_times.min(), obs_times.max()
    present = present[(present.index >= first) & (present.index <= last)]
    missing = ~present.values
    runs, i, n = [], 0, len(missing)
    while i < n:
        if missing[i]:
            j = i
            while j < n and missing[j]:
                j += 1
            runs.append((present.index[i], j - i))
            i = j
        else:
            i += 1
    return runs, present


def bin_label(v, bins):
    for lo, hi in bins:
        if lo <= v <= hi:
            return f"{lo}" if lo == hi else (f"{lo}-{hi}" if hi < 10**9 else f">={lo}")
    return "?"


def main():
    path = latest_raw_path()
    print(f"อ่าน {path}")
    raw = pd.read_csv(path)
    raw["datetime_utc"] = pd.to_datetime(raw["datetime_utc"], utc=True)

    run_rows, k_rows = [], []
    for prov, g in raw.groupby("province_en"):
        obs = pd.DatetimeIndex(g["datetime_utc"].sort_values().unique())
        grid = pd.date_range(obs.min(), obs.max(), freq="h", tz="UTC")
        runs, present = gap_runs_for(obs, grid)
        for start, length in runs:
            run_rows.append({"province_en": prov, "gap_start_utc": start, "length_h": length,
                             "start_month": start.month})
            # แต่ละชั่วโมงในช่องว่างมี staleness k = 1..length
            for k in range(1, length + 1):
                k_rows.append((prov, (start + pd.Timedelta(hours=k - 1)).month, k))
        n_hours = len(present)
        print(f"  {prov:18s} ชั่วโมงในช่วงข้อมูล {n_hours:6d}  ขาด {int((~present).sum()):5d} "
              f"({(~present).mean()*100:4.1f}%)  ช่องว่าง {len(runs):4d} ช่อง")

    runs_df = pd.DataFrame(run_rows)
    runs_path = os.path.join(_ROOT, "models", f"gap_runs_{TIMESTAMP}.csv")
    runs_df.to_csv(runs_path, index=False)

    summary = []
    # 1) การกระจายความยาวช่องว่าง (ต่อช่อง)
    runs_df["L_bin"] = runs_df["length_h"].map(lambda v: bin_label(v, L_BINS))
    for scope, sub in (("all", runs_df), ("burning_Feb-Apr", runs_df[runs_df["start_month"].isin(BURNING)]),
                       ("non_burning", runs_df[~runs_df["start_month"].isin(BURNING)])):
        for b, c in sub["L_bin"].value_counts().items():
            summary.append({"table": "gap_length_hist", "scope": scope, "bin": b, "count": int(c),
                            "share": c / len(sub)})
        q = sub["length_h"].quantile([0.5, 0.9, 0.99]).to_dict()
        summary.append({"table": "gap_length_quantile", "scope": scope, "bin": "median/p90/p99/max",
                        "count": len(sub), "share": np.nan,
                        "note": f"{q[0.5]:.0f}/{q[0.9]:.0f}/{q[0.99]:.0f}/{sub['length_h'].max()}"})

    # 2) exposure ของ staleness k (ต่อชั่วโมงที่ไม่มีค่า)
    kdf = pd.DataFrame(k_rows, columns=["province_en", "month", "k"])
    kdf["k_bin"] = kdf["k"].map(lambda v: bin_label(v, K_BINS))
    for scope, sub in (("all", kdf), ("burning_Feb-Apr", kdf[kdf["month"].isin(BURNING)]),
                       ("non_burning", kdf[~kdf["month"].isin(BURNING)])):
        for b, c in sub["k_bin"].value_counts().items():
            summary.append({"table": "staleness_exposure", "scope": scope, "bin": b, "count": int(c),
                            "share": c / len(sub)})
    # 3) สัดส่วนชั่วโมงขาดต่อจังหวัด
    for prov, g in raw.groupby("province_en"):
        obs = pd.DatetimeIndex(g["datetime_utc"].sort_values().unique())
        grid = pd.date_range(obs.min(), obs.max(), freq="h", tz="UTC")
        summary.append({"table": "missing_share_by_province", "scope": prov, "bin": "all",
                        "count": int(len(grid) - len(obs)), "share": (len(grid) - len(obs)) / len(grid)})

    out = pd.DataFrame(summary)
    out_path = os.path.join(_ROOT, "models", f"gap_distribution_{TIMESTAMP}.csv")
    out.to_csv(out_path, index=False)

    print("\n=== การกระจายความยาวช่องว่าง (ทุกจังหวัด) ===")
    print(out[(out.table == "gap_length_hist") & (out.scope == "all")].to_string(index=False))
    print(out[(out.table == "gap_length_quantile")].to_string(index=False))
    print("\n=== exposure ของ staleness k (สัดส่วนชั่วโมงที่ไม่มีค่า จัดตามอายุข้อมูลล่าสุด) ===")
    print(out[(out.table == "staleness_exposure") & (out.scope == "all")].to_string(index=False))
    print(f"\nบันทึก {runs_path}\nบันทึก {out_path}")


if __name__ == "__main__":
    main()

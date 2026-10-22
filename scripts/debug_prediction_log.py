"""
Debug prediction_log ใน predictions.db: หา error สูงสุด, เจาะลึก horizon=11,
แล้ว re-fetch ค่าจริงจาก Air4Thai ใหม่มาเทียบกับ actual_pm25 ที่ log ไว้
เพื่อแยกว่า error สูงเกิดจาก (ก) timestamp/logging bug หรือ (ข) โมเดลพยากรณ์แย่จริง
"""
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import sqlite3
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(_ROOT, r"backups\predictions.db")  # sqlite เดิม ก่อน migrate ไป Postgres
STATION_ID_AIR4THAI = "46t"
OUT_FIG = os.path.join(_ROOT, r"figures\debug_horizon11_predicted_vs_actual.png")
TARGET_HORIZON = 11
LOGGING_MISMATCH_TOL = 0.5  # ต่างกันเกินนี้ถือว่าเป็น mismatch จริง ไม่ใช่ rounding noise


def load_matched_pairs():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        """SELECT id, predicted_at_utc, target_time_utc, horizon, predicted_pm25, actual_pm25
           FROM prediction_log
           WHERE actual_pm25 IS NOT NULL""",
        conn,
    )
    conn.close()
    df["predicted_at_utc"] = pd.to_datetime(df["predicted_at_utc"])
    df["target_time_utc"] = pd.to_datetime(df["target_time_utc"])
    df["error"] = (df["predicted_pm25"] - df["actual_pm25"]).abs()
    return df


def fetch_air4thai_range(sdate, edate):
    """ดึง PM2.5 ย้อนหลังจาก Air4Thai ช่วง sdate-edate ด้วย logic เดียวกับ fetch_recent_pm25() ใน app.py"""
    url = "https://air4thai.com/forweb/getHistoryData.php"
    params = {
        "stationID": STATION_ID_AIR4THAI,
        "param": "PM25",
        "type": "hr",
        "sdate": sdate,
        "edate": edate,
        "stime": "00",
        "etime": "23",
    }
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    r = requests.get(url, params=params, headers=headers, timeout=15, verify=False)
    r.raise_for_status()
    records = r.json()["stations"][0]["data"]
    s = pd.DataFrame(records).rename(columns={"DATETIMEDATA": "datetime_utc", "PM25": "pm25"})
    s["datetime_utc"] = (
        pd.to_datetime(s["datetime_utc"]).dt.tz_localize("Asia/Bangkok").dt.tz_convert("UTC")
    )
    s = s.dropna(subset=["pm25"])
    return s.set_index("datetime_utc")["pm25"]


def refetch_actuals(target_times_utc):
    """ดึงค่าจริงใหม่จาก Air4Thai สำหรับชุด target_time ที่ให้มา (คืน dict: Timestamp -> value or NaN)"""
    times = pd.to_datetime(pd.Series(target_times_utc)).drop_duplicates()
    min_t, max_t = times.min(), times.max()
    # ครอบช่วงวันที่ +-1 วัน (ตาม local date ฝั่งไทย) กันพลาดขอบเขตวันจาก tz conversion
    sdate = (min_t.tz_convert("Asia/Bangkok") - timedelta(days=1)).strftime("%Y-%m-%d")
    edate = (max_t.tz_convert("Asia/Bangkok") + timedelta(days=1)).strftime("%Y-%m-%d")

    series = fetch_air4thai_range(sdate, edate)
    return {t: series.get(t, np.nan) for t in times}


def check_timestamp_math(df):
    """ตรวจว่า target_time_utc == predicted_at_utc + horizon ชั่วโมง จริงหรือไม่"""
    expected_target = df["predicted_at_utc"] + pd.to_timedelta(df["horizon"], unit="h")
    return (df["target_time_utc"] - expected_target).dt.total_seconds() / 3600


def print_pairs_table(d, extra_cols=None):
    cols = ["horizon", "predicted_at_utc", "target_time_utc", "predicted_pm25", "actual_pm25", "error"]
    if extra_cols:
        cols += extra_cols
    with pd.option_context("display.max_rows", None, "display.width", 220):
        print(d[cols].to_string(index=False))


def plot_horizon(h_df, refetched):
    if h_df.empty:
        print(f"\n(no data for horizon={TARGET_HORIZON}, skipping plot)")
        return

    h_df = h_df.copy()
    h_df["actual_refetched"] = h_df["target_time_utc"].map(refetched)
    h_df["time_th"] = h_df["target_time_utc"].dt.tz_convert("Asia/Bangkok")
    h_df = h_df.sort_values("time_th")

    COL_SURFACE = "#fcfcfb"
    COL_PRED = "#2a78d6"            # categorical slot 1: blue
    COL_ACTUAL_DB = "#1baf7a"       # categorical slot 2: aqua
    COL_ACTUAL_REFETCH = "#eda100"  # categorical slot 3: yellow
    COL_INK_PRIMARY = "#0b0b0b"
    COL_INK_SECONDARY = "#52514e"
    COL_MUTED = "#898781"
    COL_GRID = "#e1e0d9"
    COL_AXIS = "#c3c2b7"

    plt.rcParams.update({
        "font.family": "TH SarabunPSK",
        "font.size": 12,
        "text.color": COL_INK_PRIMARY,
        "axes.edgecolor": COL_AXIS,
        "axes.labelcolor": COL_INK_SECONDARY,
        "xtick.color": COL_MUTED,
        "ytick.color": COL_MUTED,
    })

    fig, ax = plt.subplots(figsize=(11, 6), facecolor=COL_SURFACE)
    ax.set_facecolor(COL_SURFACE)
    ax.grid(axis="y", color=COL_GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(COL_AXIS)

    x = h_df["time_th"]
    ax.plot(x, h_df["predicted_pm25"], color=COL_PRED, linewidth=2, marker="o",
            markersize=5, markeredgewidth=0, label="พยากรณ์ (predicted)", zorder=3)
    ax.plot(x, h_df["actual_pm25"], color=COL_ACTUAL_DB, linewidth=2, linestyle="--",
            marker="s", markersize=5, markeredgewidth=0, label="ค่าจริงใน DB (logged actual)", zorder=3)
    ax.plot(x, h_df["actual_refetched"], color=COL_ACTUAL_REFETCH, linewidth=2, linestyle=":",
            marker="^", markersize=5, markeredgewidth=0, label="ค่าจริง re-fetch จาก Air4Thai", zorder=2)

    ax.set_title(f"horizon = t+{TARGET_HORIZON}: predicted vs actual (DB) vs actual (re-fetched)",
                 fontsize=13.5, color=COL_INK_PRIMARY, fontweight="bold", pad=14, loc="left")
    ax.set_xlabel("target_time (เวลาไทย)")
    ax.set_ylabel("PM2.5 (µg/m³)")
    fig.autofmt_xdate(rotation=30)
    ax.legend(loc="best", frameon=False, fontsize=10)

    fig.tight_layout(rect=[0, 0.04, 1, 1])
    fig.savefig(OUT_FIG, dpi=200, facecolor=COL_SURFACE, bbox_inches="tight", pad_inches=0.25)
    print(f"\nSaved plot: {OUT_FIG}")


def main():
    df = load_matched_pairs()
    print(f"Loaded {len(df)} matched predicted/actual pairs (backfilled) from {DB_PATH}\n")

    if df.empty:
        print("No backfilled actual_pm25 pairs yet. Wait for more log data to accumulate.")
        return

    # 1) top 10 worst error overall
    top10 = df.sort_values("error", ascending=False).head(10)
    print("=" * 110)
    print("TOP 10 WORST ERRORS (all horizons)")
    print("=" * 110)
    print_pairs_table(top10)

    # 2) all horizon == TARGET_HORIZON pairs
    h_df = df[df["horizon"] == TARGET_HORIZON].sort_values("target_time_utc")
    print("\n" + "=" * 110)
    print(f"ALL PAIRS FOR horizon = {TARGET_HORIZON}  (n={len(h_df)})")
    print("=" * 110)
    if h_df.empty:
        print("(no backfilled pairs for this horizon yet)")
    else:
        print_pairs_table(h_df)

    # 3) re-fetch จาก Air4Thai ใหม่ สำหรับ top10 + horizon-11 ทั้งหมด
    recheck_df = pd.concat([top10, h_df]).drop_duplicates(subset=["id"]).copy()
    print(f"\nRe-fetching from Air4Thai for {len(recheck_df)} target_time(s) to verify...")
    refetched = refetch_actuals(recheck_df["target_time_utc"])
    recheck_df["actual_refetched"] = recheck_df["target_time_utc"].map(refetched)
    recheck_df["db_vs_refetch_diff"] = (recheck_df["actual_pm25"] - recheck_df["actual_refetched"]).abs()
    recheck_df["ts_math_ok"] = check_timestamp_math(recheck_df).abs() <= 1e-6
    recheck_df["logging_mismatch"] = recheck_df["db_vs_refetch_diff"] > LOGGING_MISMATCH_TOL

    print("\n" + "=" * 110)
    print("VERIFICATION: DB actual_pm25 vs Air4Thai re-fetched actual")
    print("=" * 110)
    print_pairs_table(
        recheck_df,
        extra_cols=["actual_refetched", "db_vs_refetch_diff", "ts_math_ok", "logging_mismatch"],
    )

    n_mismatch = int(recheck_df["logging_mismatch"].sum())
    n_ts_bad = int((~recheck_df["ts_math_ok"]).sum())
    n_nan_refetch = int(recheck_df["actual_refetched"].isna().sum())
    print(f"\n- {n_mismatch} / {len(recheck_df)} pairs: DB actual_pm25 differs from re-fetched value by more than "
          f"{LOGGING_MISMATCH_TOL} ug/m3 (signal of a logging/matching bug)")
    print(f"- {n_ts_bad} / {len(recheck_df)} pairs: target_time_utc != predicted_at_utc + horizon hours (timestamp bug)")
    if n_nan_refetch:
        print(f"- {n_nan_refetch} / {len(recheck_df)} pairs: re-fetch found no data at target_time "
              f"(beyond Air4Thai retention, or station gap at that hour)")

    # 4) plot pattern สำหรับ horizon 11 ทั้งหมด
    plot_horizon(h_df, refetched)

    return recheck_df


if __name__ == "__main__":
    main()

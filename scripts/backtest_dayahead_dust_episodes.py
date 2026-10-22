"""
Backtest พยากรณ์วันต่อวันในช่วงฝุ่นสูง: ออกพยากรณ์เวลา 00:00 น. เวลาไทยของวัน D+1 โดยใช้ข้อมูลถึง
เวลานั้น พยากรณ์ horizon 1-24 ชม. (ทั้งวัน D+1) เทียบกับค่าจริง บนชุดทดสอบเดิม (17 จังหวัดที่เทรน)
วันเป้าหมาย 1 ก.พ.-30 เม.ย. 2026

ชุดที่วัด: A ทุกวันฤดูฝุ่น, B วันที่ค่าจริงเฉลี่ยรายวัน > 37.5, C วันวิกฤตสูงมาก (>75),
วันตัวอย่าง 2026-04-02, และ 3 วันเด่นที่ค่าเฉลี่ยจริงข้าม 17 จังหวัดสูงสุด
ตัวชี้วัด: MAE รายชั่วโมง, Accuracy = 100 - MAPE, coverage ช่วง P05-P95, MAE/bias ของค่าเฉลี่ยรายวัน,
และการแจ้งเตือน "ค่าเฉลี่ยรายวัน > 37.5" (recall/precision/false alarm rate ของกฎ point และ upper)
เทียบกับ baseline persistence และ yesterday-average
"""
import importlib.util
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR, FIG_DIR = os.path.join(_ROOT, "models"), os.path.join(_ROOT, "figures")
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
THRESH = 37.5
SEVERE = 75.0
LOW_ACTUAL = 5.0
TZ = pd.Timedelta(hours=7)
DATE_FROM, DATE_TO = pd.Timestamp("2026-02-01").date(), pd.Timestamp("2026-04-30").date()
ADVISOR_DAY = pd.Timestamp("2026-04-02").date()
N_TOP_DAYS, MIN_SEP_DAYS, N_PROV_PLOT = 3, 3, 4

INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
BLUE = "#2a78d6"


def load_bt():
    spec = importlib.util.spec_from_file_location("bt", os.path.join(_ROOT, "scripts", "analyze_burning_season_backtest_clean.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def accuracy(actual, pred):
    m = actual >= LOW_ACTUAL
    if m.sum() == 0:
        return np.nan
    return 100 - (np.abs(pred[m] - actual[m]) / actual[m]).mean() * 100


def alert_stats(actual_day, score_day):
    """actual_day, score_day: ค่าเฉลี่ยรายวันจริง / ตัวที่ใช้ตัดสินแจ้งเตือน (> THRESH)"""
    a, p = actual_day > THRESH, score_day > THRESH
    tp, fn, fp, tn = int((a & p).sum()), int((a & ~p).sum()), int((~a & p).sum()), int((~a & ~p).sum())
    return {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "recall": tp / (tp + fn) if tp + fn else np.nan, "precision": tp / (tp + fp) if tp + fp else np.nan,
            "false_alarm_rate": fp / (fp + tn) if fp + tn else np.nan}


def style_axes(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    plt.rcParams["font.family"] = "Tahoma"
    bt = load_bt()
    tm = bt.load_train_module()
    df = pd.read_csv(tm.DATA_PATH)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df = df[~df["is_zero_shot"]].reset_index(drop=True)
    _, _, test_df = tm.split_train_calib_test(df)

    # baseline yesterday: ค่าเฉลี่ย pm25 ของ 24 ชม.ที่ผ่านมา ณ เวลาออกพยากรณ์ (ต้องมี >= 20 ชม.)
    hist = df[["province_en", "datetime_utc", "pm25"]].sort_values(["province_en", "datetime_utc"])
    roll = []
    for prov, g in hist.groupby("province_en"):
        s = g.set_index("datetime_utc")["pm25"].rolling("24h", min_periods=20).mean()
        roll.append(pd.DataFrame({"province_en": prov, "datetime_utc": s.index, "yday_mean": s.values}))
    roll = pd.concat(roll, ignore_index=True)

    issue = test_df[test_df["datetime_utc"].dt.hour == 17].copy()
    issue["target_date"] = (issue["datetime_utc"] + TZ).dt.date
    issue = issue[(issue["target_date"] >= DATE_FROM) & (issue["target_date"] <= DATE_TO)]
    issue = issue.merge(roll, on=["province_en", "datetime_utc"], how="left")
    print(f"เวลาออกพยากรณ์ (จังหวัด x วัน): {len(issue)}  วันเป้าหมาย {issue['target_date'].min()} .. {issue['target_date'].max()}")
    print(f"ช่วงชุดทดสอบ: {test_df['datetime_utc'].min()} .. {test_df['datetime_utc'].max()}")
    if ADVISOR_DAY not in set(issue["target_date"]):
        print(f"เตือน: ไม่มีข้อมูลวันตัวอย่าง {ADVISOR_DAY}")

    long_by_model, daily_by_model = {}, {}
    for tag in ("lgb", "xgb"):
        pkg = bt.load_model_pkg(tag)
        res = bt.predict_all_horizons(pkg, issue, pkg["feature_cols"])
        res["datetime_utc"] = pd.to_datetime(res["datetime_utc"], utc=True)
        res = res.merge(issue[["province_en", "datetime_utc", "target_date", "pm25", "yday_mean"]], on=["province_en", "datetime_utc"], how="left")
        long_by_model[tag] = res
        g = res.groupby(["province_en", "datetime_utc", "target_date"])
        d = g.agg(n_h=("horizon", "count"), pred_day=("predicted", "mean"), actual_day=("actual", "mean"),
                  hi_day=("pm25_hi", "mean"), lo_day=("pm25_lo", "mean"), pm25_now=("pm25", "first"),
                  yday_mean=("yday_mean", "first")).reset_index()
        daily_by_model[tag] = d[d["n_h"] == 24].reset_index(drop=True)
        print(f"{tag}: จังหวัด-วันที่ครบ 24 horizon = {len(daily_by_model[tag])}")

    # ---- เลือกวันเด่นตามกฎ (จากค่าจริงเท่านั้น) ----
    d0 = daily_by_model["lgb"]
    day_mean = d0.groupby("target_date").agg(mean_actual=("actual_day", "mean"), n=("province_en", "count"))
    day_mean = day_mean[day_mean["n"] >= 10].sort_values("mean_actual", ascending=False)
    chosen = []
    for dt_, row in day_mean.iterrows():
        if dt_ == ADVISOR_DAY:
            continue
        if all(abs((dt_ - c).days) >= MIN_SEP_DAYS for c in chosen):
            chosen.append(dt_)
        if len(chosen) == N_TOP_DAYS:
            break
    episodes = [ADVISOR_DAY] + chosen
    print("\nวันเด่น (เลือกจากค่าจริง):")
    for e in episodes:
        r = day_mean.loc[e] if e in day_mean.index else None
        print(f"  {e}  ค่าจริงเฉลี่ย 17 จังหวัด = {r['mean_actual']:.1f}" if r is not None else f"  {e}  (ไม่มีข้อมูลพอ)")

    # ---- ตารางสรุป ----
    sets = {"A ทุกวันฤดูฝุ่น": lambda d: d.index == d.index,
            f"B วันวิกฤต (จริง>{THRESH})": lambda d: d["actual_day"] > THRESH,
            f"C วิกฤตสูงมาก (จริง>{SEVERE:.0f})": lambda d: d["actual_day"] > SEVERE,
            f"ตัวอย่างอาจารย์ {ADVISOR_DAY}": lambda d: d["target_date"] == ADVISOR_DAY}
    rows = []
    for tag in ("lgb", "xgb"):
        d, res = daily_by_model[tag], long_by_model[tag]
        keys = d[["province_en", "datetime_utc"]]
        res_full = res.merge(keys, on=["province_en", "datetime_utc"])
        res_full = res_full.merge(d[["province_en", "datetime_utc", "actual_day"]], on=["province_en", "datetime_utc"])
        for sname, sel in sets.items():
            ds = d[sel(d)]
            if len(ds) == 0:
                continue
            rk = ds[["province_en", "datetime_utc"]]
            rs = res_full.merge(rk, on=["province_en", "datetime_utc"])
            a, p = rs["actual"].to_numpy(), rs["predicted"].to_numpy()
            covered = ((rs["actual"] >= rs["pm25_lo"]) & (rs["actual"] <= rs["pm25_hi"])).mean() * 100
            base = {"model": tag, "set": sname, "n_province_days": len(ds), "hourly_mae": np.abs(a - p).mean(),
                    "accuracy_pct": accuracy(a, p), "coverage_pct": covered,
                    "daily_mean_mae": (ds["pred_day"] - ds["actual_day"]).abs().mean(),
                    "daily_mean_bias": (ds["pred_day"] - ds["actual_day"]).mean()}
            # baseline ชั่วโมงระดับ persistence flat
            pers = rs["pm25"].to_numpy()
            base["hourly_mae_persistence"] = np.abs(a - pers).mean()
            base["accuracy_persistence"] = accuracy(a, pers)
            for rname, score in (("point", ds["pred_day"]), ("upper", ds["hi_day"])):
                st = alert_stats(ds["actual_day"], score)
                base.update({f"{rname}_{k}": v for k, v in st.items()})
            rows.append(base)
    # baseline แจ้งเตือน (ใช้ชุด lgb เป็นฐานวัน)
    base_rows = []
    d = daily_by_model["lgb"]
    for sname, sel in sets.items():
        ds = d[sel(d)]
        if len(ds) == 0:
            continue
        for bname, score in (("persistence", ds["pm25_now"]), ("yesterday", ds["yday_mean"])):
            ok = score.notna()
            st = alert_stats(ds.loc[ok, "actual_day"], score[ok])
            base_rows.append({"set": sname, "baseline": bname, "n": int(ok.sum()),
                              "daily_mean_mae": (score[ok] - ds.loc[ok, "actual_day"]).abs().mean(), **st})
    summary = pd.DataFrame(rows)
    baselines = pd.DataFrame(base_rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print("\n" + "=" * 100 + "\nสรุปโมเดล (hourly = 24 horizon x จังหวัด-วัน)\n" + "=" * 100)
    print(summary[["model", "set", "n_province_days", "hourly_mae", "hourly_mae_persistence", "accuracy_pct", "accuracy_persistence",
                   "coverage_pct", "daily_mean_mae", "daily_mean_bias"]].round(2).to_string(index=False))
    print("\nการแจ้งเตือนเกิน 37.5 (ค่าเฉลี่ยรายวัน): recall / precision / false alarm ; tp fn fp tn")
    for r in rows:
        print(f"  {r['model'].upper()} {r['set']}: point recall={r['point_recall']:.2f} prec={r['point_precision']:.2f} FAR={r['point_false_alarm_rate']:.2f} "
              f"({r['point_tp']},{r['point_fn']},{r['point_fp']},{r['point_tn']}) | upper recall={r['upper_recall']:.2f} prec={r['upper_precision']:.2f} "
              f"FAR={r['upper_false_alarm_rate']:.2f} ({r['upper_tp']},{r['upper_fn']},{r['upper_fp']},{r['upper_tn']})")
    print("\nBaseline (ค่าเฉลี่ยรายวัน):")
    print(baselines.round(2).to_string(index=False))

    # ---- กราฟรายวันเด่น (LightGBM) ----
    res = long_by_model["lgb"]
    hourly_actual = df[["province_en", "datetime_utc", "pm25"]]
    figs = []
    for ep in episodes:
        de = d0[d0["target_date"] == ep].sort_values("actual_day", ascending=False).head(N_PROV_PLOT)
        if de.empty:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(12, 7.2), sharey=False)
        fig.patch.set_facecolor(SURFACE)
        for ax, (_, r) in zip(axes.ravel(), de.iterrows()):
            style_axes(ax)
            t0 = r["datetime_utc"]
            hh = hourly_actual[(hourly_actual["province_en"] == r["province_en"]) &
                               (hourly_actual["datetime_utc"] >= t0 - pd.Timedelta(hours=24)) &
                               (hourly_actual["datetime_utc"] <= t0 + pd.Timedelta(hours=24))]
            fr = res[(res["province_en"] == r["province_en"]) & (res["datetime_utc"] == t0)].sort_values("horizon")
            tt = lambda x: (pd.to_datetime(x) + TZ).tz_localize(None) if getattr(pd.to_datetime(x), "tz", None) is None else (pd.to_datetime(x) + TZ).dt.tz_localize(None)
            ax.plot((hh["datetime_utc"] + TZ).dt.tz_localize(None), hh["pm25"], color=INK, linewidth=2, label="ค่าจริง")
            fx = t0 + pd.to_timedelta(fr["horizon"].to_numpy(), unit="h")
            fx = (fx + TZ).tz_localize(None)
            ax.fill_between(fx, fr["pm25_lo"], fr["pm25_hi"], color=BLUE, alpha=0.15, linewidth=0, label="ช่วงความเชื่อมั่น 90%")
            ax.plot(fx, fr["predicted"], color=BLUE, linewidth=2, label="พยากรณ์ (ออก 00:00 น.)")
            ax.axhline(THRESH, color=INK2, linewidth=1.2, linestyle=(0, (4, 3)))
            ax.text(ax.get_xlim()[0], THRESH, " เกณฑ์ 37.5", color=INK2, fontsize=8.5, va="bottom")
            ax.axvline((t0 + TZ).tz_localize(None), color=INK2, linewidth=1)
            ax.set_title(f"{r['province_en']}  (ค่าจริงเฉลี่ยวัน {r['actual_day']:.0f} | พยากรณ์ {r['pred_day']:.0f})", color=INK, fontsize=10.5, loc="left")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %H:%M"))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=12))
            ax.set_ylabel("PM2.5 (µg/m³)", color=INK2, fontsize=9)
        h, l = axes.ravel()[0].get_legend_handles_labels()
        fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.945), ncol=3, frameon=False, fontsize=10, labelcolor=INK)
        fig.suptitle(f"พยากรณ์วัน {ep} โดยใช้ข้อมูลถึงสิ้นวันก่อนหน้า (เส้นตั้ง = เวลาออกพยากรณ์ 00:00 น.)", color=INK, fontsize=12, y=0.985)
        fig.tight_layout(rect=(0, 0, 1, 0.90))
        path = os.path.join(FIG_DIR, f"dayahead_episode_{ep}_{TIMESTAMP}.png")
        fig.savefig(path, dpi=150, facecolor=SURFACE)
        plt.close(fig)
        figs.append(path)

    # ---- กราฟภาพรวม: พยากรณ์รายวัน vs ค่าจริงรายวัน (LightGBM) ----
    fig, ax = plt.subplots(figsize=(6.6, 6.2))
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.scatter(d0["actual_day"], d0["pred_day"], s=14, color=BLUE, alpha=0.55, linewidths=0)
    lim = max(d0["actual_day"].max(), d0["pred_day"].max()) * 1.05
    ax.plot([0, lim], [0, lim], color=INK2, linewidth=1.2)
    ax.axvline(THRESH, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.axhline(THRESH, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("ค่าจริงเฉลี่ยรายวัน (µg/m³)", color=INK2)
    ax.set_ylabel("ค่าพยากรณ์เฉลี่ยรายวัน (µg/m³)", color=INK2)
    ax.set_title(f"พยากรณ์วันถัดไป เทียบค่าจริง ฤดูฝุ่น {DATE_FROM}..{DATE_TO} (n={len(d0)} จังหวัด-วัน)", color=INK, fontsize=11, loc="left")
    ax.text(lim * 0.98, lim * 0.04, "เส้นทแยง = พยากรณ์ตรงค่าจริง", color=INK2, fontsize=9, ha="right")
    ax.text(THRESH, lim * 0.02, " เกณฑ์ 37.5", color=INK2, fontsize=8.5)
    fig.tight_layout()
    scatter_path = os.path.join(FIG_DIR, f"dayahead_scatter_{TIMESTAMP}.png")
    fig.savefig(scatter_path, dpi=150, facecolor=SURFACE)
    plt.close(fig)

    out_daily = os.path.join(MODEL_DIR, f"dayahead_backtest_daily_{TIMESTAMP}.csv")
    pd.concat([daily_by_model[t].assign(model=t) for t in ("lgb", "xgb")]).to_csv(out_daily, index=False)
    summary.to_csv(os.path.join(MODEL_DIR, f"dayahead_backtest_summary_{TIMESTAMP}.csv"), index=False)
    baselines.to_csv(os.path.join(MODEL_DIR, f"dayahead_backtest_baselines_{TIMESTAMP}.csv"), index=False)
    print("\nบันทึก:", out_daily)
    for p in figs + [scatter_path]:
        print("กราฟ:", p)


if __name__ == "__main__":
    main()

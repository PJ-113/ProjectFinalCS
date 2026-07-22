import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(_ROOT, r"models\r2_comparison_3models_by_horizon.csv")
OUT_PATH = os.path.join(_ROOT, r"figures\r2_comparison_3models_by_horizon.png")

df = pd.read_csv(CSV_PATH).sort_values("horizon")
prophet_r2 = df["R2_prophet_flat"].iloc[0]

COL_SURFACE = "#fcfcfb"
COL_LGB = "#2a78d6"      # slot 1 blue
COL_XGB = "#1baf7a"      # slot 2 aqua
COL_PROPHET = "#eda100"  # slot 3 yellow
COL_INK_PRIMARY = "#0b0b0b"
COL_INK_SECONDARY = "#52514e"
COL_MUTED = "#898781"
COL_GRID = "#e1e0d9"
COL_AXIS = "#c3c2b7"

plt.rcParams.update({
    "font.family": "TH SarabunPSK",
    "font.size": 13,
    "text.color": COL_INK_PRIMARY,
    "axes.edgecolor": COL_AXIS,
    "axes.labelcolor": COL_INK_SECONDARY,
    "xtick.color": COL_MUTED,
    "ytick.color": COL_MUTED,
})

fig, ax = plt.subplots(figsize=(9, 6), facecolor=COL_SURFACE)
ax.set_facecolor(COL_SURFACE)
ax.grid(axis="y", color=COL_GRID, linewidth=1, zorder=0)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(COL_AXIS)

ax.plot(df["horizon"], df["R2_lgb"],
        color=COL_LGB, linewidth=2, linestyle="-",
        marker="o", markersize=5, markerfacecolor=COL_LGB, markeredgewidth=0,
        label="LightGBM (multi-horizon)", zorder=3)
ax.plot(df["horizon"], df["R2_xgb"],
        color=COL_XGB, linewidth=2, linestyle="--",
        marker="s", markersize=5, markerfacecolor=COL_XGB, markeredgewidth=0,
        label="XGBoost (multi-horizon)", zorder=3)
ax.axhline(prophet_r2, color=COL_PROPHET, linewidth=2, linestyle=":", zorder=2,
           label="Prophet (nowcast เดียว ไม่ใช่ multi-horizon)")

# direct end-labels
ax.annotate(f"{df['R2_lgb'].iloc[-1]:.3f}", xy=(24, df["R2_lgb"].iloc[-1]),
            xytext=(6, 4), textcoords="offset points", va="center",
            fontsize=9.5, color=COL_INK_PRIMARY, fontweight="bold")
ax.annotate(f"{df['R2_xgb'].iloc[-1]:.3f}", xy=(24, df["R2_xgb"].iloc[-1]),
            xytext=(6, -6), textcoords="offset points", va="center",
            fontsize=9.5, color=COL_INK_PRIMARY, fontweight="bold")
ax.annotate(f"{prophet_r2:.3f}", xy=(24, prophet_r2),
            xytext=(6, 4), textcoords="offset points", va="center",
            fontsize=9.5, color=COL_INK_PRIMARY, fontweight="bold")

ax.set_title("R² by horizon: LightGBM vs XGBoost vs Prophet",
             fontsize=13.5, color=COL_INK_PRIMARY, fontweight="bold", pad=14, loc="left")
ax.set_xlabel("Forecast horizon (ชั่วโมงข้างหน้า)")
ax.set_ylabel("R² (test set)")
ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
ax.set_xlim(0.5, 27)
ax.set_ylim(0.45, 1.03)

ax.legend(loc="center left", frameon=False, fontsize=10, bbox_to_anchor=(0.0, 0.32))

fig.tight_layout(rect=[0, 0.08, 1, 1])
fig.text(
    0.02, 0.01,
    "หมายเหตุ: Prophet ทำนายค่า PM2.5 ณ เวลาปัจจุบันจาก weather เท่านั้น (ไม่มี lag feature) "
    "จึงไม่ใช่การเปรียบเทียบที่ fair 100% กับอีก 2 โมเดลซึ่งพยากรณ์ล่วงหน้าแบบ direct multi-horizon",
    fontsize=9, color=COL_INK_SECONDARY, ha="left", va="bottom",
)
fig.savefig(OUT_PATH, dpi=200, facecolor=COL_SURFACE, bbox_inches="tight", pad_inches=0.25)
print(f"Saved: {OUT_PATH}")

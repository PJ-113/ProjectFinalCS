import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(_ROOT, r"models\eval_weighted_vs_baseline_20260709_004955.csv")
OUT_PATH = os.path.join(_ROOT, r"figures\bias_comparison_baseline_vs_weighted.png")

df = pd.read_csv(CSV_PATH).sort_values("horizon")

COL_SURFACE = "#fcfcfb"
COL_BASELINE = "#2a78d6"
COL_WEIGHTED = "#1baf7a"
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

fig, ax = plt.subplots(figsize=(8, 5.5), facecolor=COL_SURFACE)
ax.set_facecolor(COL_SURFACE)
ax.grid(axis="y", color=COL_GRID, linewidth=1, zorder=0)
ax.set_axisbelow(True)
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)
for spine in ["left", "bottom"]:
    ax.spines[spine].set_color(COL_AXIS)

# reference line: bias = 0 คือไม่มี bias เลย (เป้าหมายในอุดมคติ)
ax.axhline(0, color=COL_MUTED, linewidth=1.2, linestyle=":", zorder=1)
ax.annotate("bias = 0 (ไม่มีอคติ)", xy=(13, 0), xytext=(0, 6), textcoords="offset points",
            ha="center", va="bottom", fontsize=9.5, color=COL_MUTED)

ax.plot(df["horizon"], df["bias_high_baseline"],
        color=COL_BASELINE, linewidth=2, linestyle="-",
        marker="o", markersize=5, markerfacecolor=COL_BASELINE, markeredgewidth=0,
        label="Baseline (default params)", zorder=3)
ax.plot(df["horizon"], df["bias_high_weighted"],
        color=COL_WEIGHTED, linewidth=2, linestyle="--",
        marker="s", markersize=5, markerfacecolor=COL_WEIGHTED, markeredgewidth=0,
        label="Weighted (sample-weighted)", zorder=3)

last_h = df["horizon"].iloc[-1]
ax.annotate(f"{df['bias_high_baseline'].iloc[-1]:+.2f}", xy=(last_h, df["bias_high_baseline"].iloc[-1]),
            xytext=(6, 3), textcoords="offset points", va="center",
            fontsize=9.5, color=COL_INK_PRIMARY, fontweight="bold")
ax.annotate(f"{df['bias_high_weighted'].iloc[-1]:+.2f}", xy=(last_h, df["bias_high_weighted"].iloc[-1]),
            xytext=(6, -3), textcoords="offset points", va="center",
            fontsize=9.5, color=COL_INK_PRIMARY, fontweight="bold")

ax.set_title("Bias เฉพาะช่วง PM2.5 สูง (>37.5 µg/m³): baseline vs weighted",
             fontsize=13.5, color=COL_INK_PRIMARY, fontweight="bold", pad=18, loc="left")
ax.set_xlabel("Forecast horizon (ชั่วโมงข้างหน้า)")
ax.set_ylabel("Bias เฉลี่ย (µg/m³) — ค่าบวก = โมเดลทายต่ำกว่าจริง")
ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
ax.set_xlim(0.5, 27)
ax.set_ylim(-0.3, 5.3)

ax.legend(loc="upper left", frameon=False, fontsize=10, bbox_to_anchor=(0.0, 0.97))

fig.tight_layout()
fig.savefig(OUT_PATH, dpi=200, facecolor=COL_SURFACE, bbox_inches="tight", pad_inches=0.25)
print(f"Saved: {OUT_PATH}")

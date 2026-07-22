import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(_ROOT, r"models\eval_weighted_vs_baseline_20260709_004955.csv")
OUT_PATH = os.path.join(_ROOT, r"figures\mae_comparison_baseline_vs_weighted.png")

df = pd.read_csv(CSV_PATH).sort_values("horizon")

# --- palette (validated: blue/aqua categorical pair, CVD-safe) ---
COL_SURFACE = "#fcfcfb"
COL_BASELINE = "#2a78d6"   # categorical slot 1 (blue)
COL_WEIGHTED = "#1baf7a"   # categorical slot 2 (aqua)
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

fig, axes = plt.subplots(1, 2, figsize=(12, 5), facecolor=COL_SURFACE)

panels = [
    ("MAE_all_baseline", "MAE_all_weighted", "MAE ภาพรวมทุกช่วง PM2.5", axes[0]),
    ("MAE_high_baseline", "MAE_high_weighted", "MAE เฉพาะช่วง PM2.5 สูง (>37.5 µg/m³)", axes[1]),
]

for baseline_col, weighted_col, title, ax in panels:
    ax.set_facecolor(COL_SURFACE)
    ax.grid(axis="y", color=COL_GRID, linewidth=1, zorder=0)
    ax.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    for spine in ["left", "bottom"]:
        ax.spines[spine].set_color(COL_AXIS)

    ax.plot(df["horizon"], df[baseline_col],
            color=COL_BASELINE, linewidth=2, linestyle="-",
            marker="o", markersize=5, markerfacecolor=COL_BASELINE, markeredgewidth=0,
            label="Baseline (default params)", zorder=3)
    ax.plot(df["horizon"], df[weighted_col],
            color=COL_WEIGHTED, linewidth=2, linestyle="--",
            marker="s", markersize=5, markerfacecolor=COL_WEIGHTED, markeredgewidth=0,
            label="Weighted (sample-weighted)", zorder=3)

    # direct end-of-line labels (mitigates low-contrast aqua vs surface)
    last_h = df["horizon"].iloc[-1]
    ax.annotate(f"{df[baseline_col].iloc[-1]:.2f}", xy=(last_h, df[baseline_col].iloc[-1]),
                xytext=(6, 0), textcoords="offset points", va="center",
                fontsize=9.5, color=COL_INK_PRIMARY, fontweight="bold")
    ax.annotate(f"{df[weighted_col].iloc[-1]:.2f}", xy=(last_h, df[weighted_col].iloc[-1]),
                xytext=(6, -2), textcoords="offset points", va="center",
                fontsize=9.5, color=COL_INK_PRIMARY, fontweight="bold")

    ax.set_title(title, fontsize=12.5, color=COL_INK_PRIMARY, fontweight="bold", pad=12, loc="left")
    ax.set_xlabel("Forecast horizon (ชั่วโมงข้างหน้า)")
    ax.set_ylabel("MAE (µg/m³)")
    ax.xaxis.set_major_locator(mticker.MultipleLocator(4))
    ax.set_xlim(0.5, 26.5)

axes[0].legend(loc="upper left", frameon=False, fontsize=10)

fig.suptitle("เปรียบเทียบความแม่นยำโมเดล LightGBM: Baseline vs Sample-Weighted (ทดสอบบน test set จริง)",
             fontsize=13.5, color=COL_INK_PRIMARY, fontweight="bold", y=1.02)

fig.tight_layout()
fig.savefig(OUT_PATH, dpi=200, facecolor=COL_SURFACE, bbox_inches="tight", pad_inches=0.25)
print(f"Saved: {OUT_PATH}")

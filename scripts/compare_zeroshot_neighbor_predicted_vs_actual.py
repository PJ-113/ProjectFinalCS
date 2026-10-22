"""
เทียบเส้นค่าพยากรณ์ (ไม่ใช่แค่ค่าจริง) ของจังหวัด zero-shot กับเพื่อนบ้านที่เทรนจริง ที่ horizon=24
(LightGBM P50) วัด correlation ระหว่างค่าพยากรณ์ของ zero-shot กับค่าจริงของเพื่อนบ้าน และกับค่าจริง
ของตัวเอง timestamp ของค่าพยากรณ์เลื่อนไปข้างหน้า 24 ชม. ก่อนพล็อต ให้ตรงกับเวลาเป้าหมายจริง
"""
import importlib.util
import os
import pickle

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

matplotlib.rcParams["font.family"] = "Tahoma"  # รองรับข้อความไทย

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(_ROOT, "models")
FIGURES_DIR = os.path.join(_ROOT, "figures")
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

HORIZON = 24
MODEL_TAG = "20260919_203455"  # v3 ที่ deploy จริง
WINDOW_START = pd.Timestamp("2026-01-01", tz="UTC")
WINDOW_END = pd.Timestamp("2026-03-31", tz="UTC")

PAIRS = [
    ("mahasarakham", "kalasin", "มหาสารคาม", "กาฬสินธุ์"),
    ("roiet", "kalasin", "ร้อยเอ็ด", "กาฬสินธุ์"),
    ("nakhonphanom", "sakonnakhon", "นครพนม", "สกลนคร"),
]
# คู่หลักที่วาดกราฟเต็ม; อีก 2 คู่คำนวณแค่ correlation
PRIMARY_PAIR_IDX = 0

COLOR_ZS_ACTUAL = "#2a78d6"     # น้ำเงิน — มหาสารคามค่าจริง
COLOR_ZS_PRED = "#9ab8e8"       # น้ำเงินอ่อน — มหาสารคามค่าพยากรณ์
COLOR_NB_ACTUAL = "#eb6834"     # ส้ม — กาฬสินธุ์ค่าจริง
COLOR_NB_PRED = "#f0b291"       # ส้มอ่อน — กาฬสินธุ์ค่าพยากรณ์


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, "scripts", filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def predict_series(df_province, model_h24_p50, feature_cols):
    """คืน Series ของค่าพยากรณ์ (P50) ที่ horizon=24 index ด้วยเวลาเป้าหมายจริง (datetime_utc + 24h)
    ไม่ใช่เวลาที่สร้างค่าพยากรณ์"""
    target_col = f"pm25_t+{HORIZON}"
    sub = df_province.dropna(subset=[target_col] + feature_cols).copy()
    X = sub[feature_cols]
    pred_resid = model_h24_p50.predict(X)
    pred_abs = pred_resid + sub["pm25"].values
    target_time = sub["datetime_utc"] + pd.Timedelta(hours=HORIZON)
    return pd.Series(pred_abs, index=target_time).sort_index()


def window(s):
    return s.loc[(s.index >= WINDOW_START) & (s.index <= WINDOW_END)]


def main():
    train_mod = load_module("train_multiprovince_models", "train_multiprovince_models.py")
    feature_cols = train_mod.FEATURE_COLS

    model_path = os.path.join(MODEL_DIR, f"lgb_pm25_multiprovince_residual_bagged_quantile_{MODEL_TAG}.pkl")
    print(f"โหลดโมเดล: {model_path}")
    with open(model_path, "rb") as f:
        pkg = pickle.load(f)
    model_h24_p50 = pkg["models"][HORIZON][0.5]

    print(f"โหลดข้อมูล: {train_mod.DATA_PATH}")
    df = pd.read_csv(train_mod.DATA_PATH)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])

    series_cache = {}

    def get_series(prov):
        if prov not in series_cache:
            df_prov = df[df["province_en"] == prov]
            actual = df_prov.set_index("datetime_utc")["pm25"].sort_index()
            predicted = predict_series(df_prov, model_h24_p50, feature_cols)
            series_cache[prov] = {"actual": actual, "predicted": predicted}
            print(f"{prov}: actual n={len(actual)}, predicted(t+{HORIZON}) n={len(predicted)}")
        return series_cache[prov]

    summary_rows = []
    os.makedirs(FIGURES_DIR, exist_ok=True)

    for idx, (zs_prov, nb_prov, zs_th, nb_th) in enumerate(PAIRS):
        zs = get_series(zs_prov)
        nb = get_series(nb_prov)
        zs_actual_w = window(zs["actual"])
        zs_pred_w = window(zs["predicted"])
        nb_actual_w = window(nb["actual"])
        nb_pred_w = window(nb["predicted"])

        merged = pd.concat([zs_pred_w.rename("zs_pred"), nb_actual_w.rename("nb_actual")], axis=1).dropna()
        corr_pred_vs_neighbor_actual = merged["zs_pred"].corr(merged["nb_actual"])
        merged_self = pd.concat([zs_pred_w.rename("zs_pred"), zs_actual_w.rename("zs_actual")], axis=1).dropna()
        corr_pred_vs_own_actual = merged_self["zs_pred"].corr(merged_self["zs_actual"])
        print(f"\n{zs_th} (zero-shot) <-> {nb_th} (เทรนจริง):")
        print(f"  correlation(ค่าพยากรณ์ t+24 ของ {zs_th}, ค่าจริงของ {nb_th}) = {corr_pred_vs_neighbor_actual:.3f}")
        print(f"  correlation(ค่าพยากรณ์ t+24 ของ {zs_th}, ค่าจริงของตัวเอง) = {corr_pred_vs_own_actual:.3f}")

        summary_rows.append({
            "zero_shot_province": zs_th, "neighbor_province": nb_th, "horizon": HORIZON, "model": "lightgbm",
            "corr_predicted_vs_neighbor_actual": round(float(corr_pred_vs_neighbor_actual), 3),
            "corr_predicted_vs_own_actual": round(float(corr_pred_vs_own_actual), 3),
            "n_window": len(merged),
        })

        if idx == PRIMARY_PAIR_IDX:
            fig, ax = plt.subplots(figsize=(11, 5))
            ax.plot(zs_actual_w.index, zs_actual_w.values, color=COLOR_ZS_ACTUAL, linewidth=1.5,
                    label=f"{zs_th} (Zero-shot) — ค่าจริง")
            ax.plot(zs_pred_w.index, zs_pred_w.values, color=COLOR_ZS_PRED, linewidth=1.3, linestyle="--",
                    label=f"{zs_th} (Zero-shot) — ค่าพยากรณ์ t+{HORIZON}")
            ax.plot(nb_actual_w.index, nb_actual_w.values, color=COLOR_NB_ACTUAL, linewidth=1.5,
                    label=f"{nb_th} (เทรนจริง) — ค่าจริง")
            ax.plot(nb_pred_w.index, nb_pred_w.values, color=COLOR_NB_PRED, linewidth=1.3, linestyle="--",
                    label=f"{nb_th} (เทรนจริง) — ค่าพยากรณ์ t+{HORIZON}")

            ax.set_ylabel("PM2.5 (µg/m³)")
            # ไม่ใส่ title ในรูป
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
            ax.grid(True, axis="y", linewidth=0.5, alpha=0.3)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.legend(loc="upper right", frameon=False, fontsize=9)
            fig.tight_layout()

            fig_path = os.path.join(FIGURES_DIR, f"zeroshot_predicted_vs_neighbor_actual_{zs_prov}_{TIMESTAMP}.png")
            fig.savefig(fig_path, dpi=150)
            plt.close(fig)
            print(f"  บันทึกกราฟที่: {fig_path}")

    summary = pd.DataFrame(summary_rows)
    csv_path = os.path.join(MODEL_DIR, f"zeroshot_predicted_vs_neighbor_summary_{TIMESTAMP}.csv")
    summary.to_csv(csv_path, index=False)
    print(f"\nบันทึกตารางสรุปที่: {csv_path}")
    print("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()

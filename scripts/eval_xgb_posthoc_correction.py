"""
ประเมิน post-hoc bias correction สำหรับ XGBoost: ไม่แตะโมเดล/ไม่ retrain เลย แค่ชดเชย raw prediction
บวก bias_high ต่อ horizon (จาก xgb_bias_high_by_horizon_20260712_191350.csv) ด้วย ramp เชิงเส้น
ระหว่าง (LOW, HIGH) เฉพาะตอน raw prediction เข้าใกล้โซนมลพิษสูง (แทน hard cutoff ที่มี discontinuity)

เลือกโซน ramp และ context-gate variant จาก valid_tune (15% ท้ายของ train) เท่านั้น เพื่อไม่ leak
กับ test ที่ bias_high_by_horizon คำนวณมาจาก แล้วรันยืนยันครั้งเดียวบน test ด้วย config ที่เลือกไว้
"""
import pickle

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(_ROOT, r"data\khonkaen_pm25_features_raw_20260712_191249.csv")
XGB_PATH = os.path.join(_ROOT, r"models\xgb_pm25_multistep_20260712_191350.pkl")
BIAS_CSV_PATH = os.path.join(_ROOT, r"models\xgb_bias_high_by_horizon_20260712_191350.csv")
HIGH_THRESHOLD = 37.5  # นิยาม "มลพิษสูง" ไว้ตัดสิน bias_high/MAE_high
MAE_DEGRADE_GUARD = 0.08  # MAE_all ยอมแย่ลงได้ไม่เกินเท่านี้
CONTEXT_THRESHOLD = 25.0  # pm25_lag_1 ต่ำกว่านี้ถือว่ายังไม่เข้าสู่ช่วงมลพิษสูง

ZONE_CANDIDATES = [(20.0, 30.0), (25.0, 35.0), (28.0, 37.5), (22.0, 33.0)]

with open(XGB_PATH, "rb") as f:
    xgb_pkg = pickle.load(f)
FEATURE_COLS = xgb_pkg["feature_cols"]

bias_df = pd.read_csv(BIAS_CSV_PATH)
BIAS_HIGH_BY_HORIZON = {int(r.horizon): float(r.bias_high) for r in bias_df.itertuples()}

df = pd.read_csv(DATA_PATH)
df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
split_idx = int(len(df) * 0.8)
train = df.iloc[:split_idx]
test = df.iloc[split_idx:]

tune_split_idx = int(len(train) * 0.85)
valid_tune = train.iloc[tune_split_idx:]  # 15% ท้ายของ train, ใช้เลือก config เท่านั้น


def ramp_weight(raw_pred, low, high):
    if raw_pred <= low:
        return 0.0
    if raw_pred >= high:
        return 1.0
    return (raw_pred - low) / (high - low)


def correct(raw_pred, bias, low, high, context, use_context_gate):
    w = ramp_weight(raw_pred, low, high)
    if use_context_gate and context < CONTEXT_THRESHOLD:
        w *= 0.5
    return raw_pred + w * bias


def metrics_for(y_true, y_pred):
    mask_high = y_true > HIGH_THRESHOLD
    mae_all = float(mean_absolute_error(y_true, y_pred))
    if mask_high.sum() > 0:
        mae_high = float(mean_absolute_error(y_true[mask_high], y_pred[mask_high]))
        bias_high = float((y_true[mask_high] - y_pred[mask_high]).mean())
    else:
        mae_high = float("nan")
        bias_high = float("nan")
    return mae_all, mae_high, bias_high, int(mask_high.sum())


def predict_and_context(split_df, h):
    target = f"pm25_t+{h}"
    valid_rows = split_df.dropna(subset=[target])
    model = xgb_pkg["models"][h]
    y_true = valid_rows[target].values
    y_pred_raw = model.predict(valid_rows[FEATURE_COLS])
    context = valid_rows["pm25_lag_1"].values
    return y_true, y_pred_raw, context


# ===== ขั้นที่ 1: เลือก config (zone, gate) จาก valid_tune เท่านั้น ห้ามแตะ test =====
select_rows = []
for h in range(1, 25):
    y_true_v, y_pred_v, ctx_v = predict_and_context(valid_tune, h)
    bias = BIAS_HIGH_BY_HORIZON[h]
    mae_all_base_v, mae_high_base_v, bias_high_base_v, n_high_v = metrics_for(y_true_v, y_pred_v)

    for low, high in ZONE_CANDIDATES:
        for use_gate in (False, True):
            y_pred_corr_v = np.array([
                correct(p, bias, low, high, c, use_gate) for p, c in zip(y_pred_v, ctx_v)
            ])
            mae_all_c, mae_high_c, bias_high_c, _ = metrics_for(y_true_v, y_pred_corr_v)
            select_rows.append({
                "horizon": h, "zone_low": low, "zone_high": high, "context_gate": use_gate,
                "MAE_all_baseline": mae_all_base_v, "MAE_all_corrected": mae_all_c,
                "bias_high_baseline": bias_high_base_v, "bias_high_corrected": bias_high_c,
            })

select_df = pd.DataFrame(select_rows)
summary = (
    select_df.groupby(["zone_low", "zone_high", "context_gate"])
    .apply(lambda g: pd.Series({
        "MAE_all_pct_change": (g["MAE_all_corrected"].mean() - g["MAE_all_baseline"].mean())
        / g["MAE_all_baseline"].mean(),
        "bias_high_baseline_mean_abs": g["bias_high_baseline"].abs().mean(),
        "bias_high_corrected_mean_abs": g["bias_high_corrected"].abs().mean(),
        "bias_high_pct_improvement": 1 - g["bias_high_corrected"].abs().mean() / g["bias_high_baseline"].abs().mean(),
        "n_horizons_worse_both": int((
            (g["MAE_all_corrected"] > g["MAE_all_baseline"]) &
            (g["bias_high_corrected"].abs() > g["bias_high_baseline"].abs())
        ).sum()),
    }), include_groups=False)
    .reset_index()
)
summary = summary.sort_values("bias_high_pct_improvement", ascending=False)

print("=== เลือก config จาก valid_tune (15% ท้ายของ train, คนละช่วงเวลากับ test) ===")
print(summary.to_string(index=False))

passing = summary[summary["MAE_all_pct_change"] <= MAE_DEGRADE_GUARD].copy()
print(f"\n=== ผ่านเกณฑ์ MAE_all ไม่แย่ลงเกิน {MAE_DEGRADE_GUARD:.0%} (บน valid_tune) ===")
print(passing.to_string(index=False))

timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
select_path = os.path.join(_ROOT, rf"models\eval_xgb_posthoc_correction_select_{timestamp}.csv")
summary.to_csv(select_path, index=False)
print(f"\nsaved (selection results, all config, from valid_tune): {select_path}")

if len(passing) == 0:
    print("\nไม่มี config ไหนผ่านเกณฑ์ — หยุดตรงนี้ ไม่แนะนำให้ deploy post-hoc correction")
else:
    best = passing.sort_values("bias_high_pct_improvement", ascending=False).iloc[0]
    chosen_low, chosen_high, chosen_gate = best["zone_low"], best["zone_high"], bool(best["context_gate"])
    print(f"\n=== config ที่เลือก: zone=({chosen_low},{chosen_high}) context_gate={chosen_gate} ===")

    # ===== ขั้นที่ 2: รันยืนยันครั้งเดียวบน test จริง ด้วย config ที่เลือกไว้แล้ว =====
    confirm_rows = []
    for h in range(1, 25):
        y_true_t, y_pred_t, ctx_t = predict_and_context(test, h)
        bias = BIAS_HIGH_BY_HORIZON[h]
        mae_all_base_t, mae_high_base_t, bias_high_base_t, n_high_t = metrics_for(y_true_t, y_pred_t)
        y_pred_corr_t = np.array([
            correct(p, bias, chosen_low, chosen_high, c, chosen_gate) for p, c in zip(y_pred_t, ctx_t)
        ])
        mae_all_c, mae_high_c, bias_high_c, _ = metrics_for(y_true_t, y_pred_corr_t)
        confirm_rows.append({
            "horizon": h,
            "MAE_all_baseline": mae_all_base_t, "MAE_all_corrected": mae_all_c,
            "MAE_high_baseline": mae_high_base_t, "MAE_high_corrected": mae_high_c,
            "bias_high_baseline": bias_high_base_t, "bias_high_corrected": bias_high_c,
            "n_high": n_high_t,
        })
    confirm_df = pd.DataFrame(confirm_rows)
    print("\n=== ผลยืนยันบน test จริง (หมายเหตุ: bias_high_by_horizon มาจาก test set นี้เอง ตัวเลข")
    print("bias_high_corrected ฝั่งนี้จึงดีกว่าความเป็นจริงบางส่วน — ตัวเลข generalization ที่เชื่อถือได้คือฝั่ง valid_tune ด้านบน) ===")
    print(confirm_df.to_string(index=False))

    confirm_path = os.path.join(_ROOT, rf"models\eval_xgb_posthoc_correction_confirm_{timestamp}.csv")
    confirm_df.to_csv(confirm_path, index=False)
    print(f"\nchosen_zone_low={chosen_low}")
    print(f"chosen_zone_high={chosen_high}")
    print(f"chosen_context_gate={chosen_gate}")
    print(f"saved: {confirm_path}")

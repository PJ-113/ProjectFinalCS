"""
ทดสอบความสามารถในการเตือนภัยล่วงหน้า (Proactive Alert Capability) — objective 5 ของ proposal

ใช้ผล backtest วันต่อวันที่มีอยู่แล้ว (models/dayahead_backtest_daily_20260920_234307.csv, จาก
scripts/backtest_dayahead_dust_episodes.py) ไม่ดึงข้อมูลใหม่/ไม่ retrain คำนวณ Recall/Precision/F1
เทียบ 3 วิธีเตือน (ค่าพยากรณ์ตรงๆ, persistence, ค่าเฉลี่ยวันก่อนหน้า) ที่ 2 เกณฑ์ (37.5 µg/m³ มาตรฐาน
คพ. ปัจจุบัน = หลัก, 50 µg/m³ = เกณฑ์เดียวกับ ESAN-Air เพื่อเทียบตรงๆ เท่านั้น) รูปแบบตารางอิงตามที่
ESAN-Air รายงาน (Recall/Precision/F1) แต่แยกครบ 17 จังหวัด แทนที่จะแค่ 4 จังหวัด

ตัวเลข pooled ที่ threshold=37.5, method=pred_day ต้องตรงกับที่เคยยืนยันในเอกสารอาจารย์แล้ว
(recall ~93% LightGBM เทียบ persistence ~90%) ใช้เป็น sanity check ก่อนเชื่อผลแยกจังหวัด
"""
import os

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAILY_CSV = os.path.join(_ROOT, "models", "dayahead_backtest_daily_20260920_234307.csv")
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")

THRESHOLDS = [37.5, 50.0]
METHODS = {"pred_day": "ค่าพยากรณ์", "pm25_now": "persistence", "yday_mean": "ค่าเฉลี่ยวันก่อนหน้า"}
MODELS = ["lgb", "xgb"]
RECALL_ACCEPT = 0.80  # เกณฑ์ตัดสินตั้งไว้ก่อนดูผล (อิงช่วงของ ESAN-Air 70-81%)
RECALL_FLAG = 0.70  # ต่ำกว่านี้ต้องรายงานแยกพร้อมสมมติฐาน


def compute_confusion(df, pred_col, threshold):
    """คืน dict TP/FP/FN/TN + recall/precision/f1/false_alarm_rate สำหรับ (pred_col > threshold)
    เทียบ actual_day > threshold เป็นเหตุการณ์จริง"""
    pred_pos = df[pred_col] > threshold
    actual_pos = df["actual_day"] > threshold
    tp = int((pred_pos & actual_pos).sum())
    fp = int((pred_pos & ~actual_pos).sum())
    fn = int((~pred_pos & actual_pos).sum())
    tn = int((~pred_pos & ~actual_pos).sum())
    recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    precision = tp / (tp + fp) if (tp + fp) > 0 else np.nan
    f1 = 2 * precision * recall / (precision + recall) if (precision and recall and (precision + recall) > 0) else np.nan
    false_alarm_rate = fp / (tp + fp) if (tp + fp) > 0 else np.nan
    return {
        "n_events": int(actual_pos.sum()), "n": len(df),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "recall": recall, "precision": precision, "f1": f1, "false_alarm_rate": false_alarm_rate,
    }


def main():
    df = pd.read_csv(DAILY_CSV)
    print(f"โหลด: {DAILY_CSV}  ({len(df)} แถว, {df.province_en.nunique()} จังหวัด, model={sorted(df.model.unique())})")

    # ---------- pooled: model x threshold x method ----------
    pooled_rows = []
    for model in MODELS:
        sub_m = df[df.model == model]
        for threshold in THRESHOLDS:
            for col, label in METHODS.items():
                r = compute_confusion(sub_m, col, threshold)
                pooled_rows.append({"model": model, "threshold": threshold, "method": label, **r})
            # แถวเสริม: ขอบบนช่วงความเชื่อมั่น (ไม่รวมในตารางหลัก แค่ไว้พูดเรื่อง trade-off)
            r_hi = compute_confusion(sub_m, "hi_day", threshold)
            pooled_rows.append({"model": model, "threshold": threshold, "method": "ขอบบนช่วงความเชื่อมั่น (P95)", **r_hi})
    pooled = pd.DataFrame(pooled_rows)

    pooled_path = os.path.join(_ROOT, "models", f"alert_capability_pooled_{TIMESTAMP}.csv")
    pooled.to_csv(pooled_path, index=False)

    # ---------- sanity check: pooled, threshold=37.5, method=pred_day, model=lgb ----------
    chk = pooled[(pooled.model == "lgb") & (pooled.threshold == 37.5) & (pooled.method == "ค่าพยากรณ์")].iloc[0]
    print("\n" + "=" * 78)
    print("SANITY CHECK (ต้องตรงกับเอกสารอาจารย์: recall~93%, false_alarm~7%)")
    print("=" * 78)
    print(f"LGB, threshold=37.5, ค่าพยากรณ์: recall={chk.recall*100:.1f}%  false_alarm={chk.false_alarm_rate*100:.1f}%  "
          f"n_events={chk.n_events}  tp={chk.tp} fp={chk.fp} fn={chk.fn}")
    ok = abs(chk.recall - 0.9253) < 0.01 and abs(chk.false_alarm_rate - 0.1335) < 0.02
    print("sanity check:", "PASS" if ok else "FAIL — ตรวจโค้ดก่อนเชื่อผลต่อ!")
    assert ok, "sanity check ไม่ผ่าน หยุดก่อน"

    # ---------- ตารางสรุป: เทียบ 3 วิธี ที่แต่ละเกณฑ์ ----------
    for threshold in THRESHOLDS:
        print(f"\n{'='*90}\nเทียบ 3 วิธีเตือน ที่เกณฑ์ {threshold} µg/m³ (n_events={int(pooled[(pooled.threshold==threshold)&(pooled.model=='lgb')].n_events.iloc[0])} จาก {len(df[df.model=='lgb'])} วัน)\n{'='*90}")
        t = pooled[(pooled.threshold == threshold) & (pooled.method != "ขอบบนช่วงความเชื่อมั่น (P95)")]
        disp = t.pivot_table(index="method", columns="model", values=["recall", "precision", "f1", "false_alarm_rate"])
        print((disp * 100).round(1))
        hi = pooled[(pooled.threshold == threshold) & (pooled.method == "ขอบบนช่วงความเชื่อมั่น (P95)") & (pooled.model == "lgb")].iloc[0]
        print(f"  (อ้างอิง) ขอบบนช่วงความเชื่อมั่น LGB: recall={hi.recall*100:.1f}%  false_alarm={hi.false_alarm_rate*100:.1f}%")

    # ---------- รายจังหวัด: threshold=37.5, method=pred_day ----------
    prov_rows = []
    for model in MODELS:
        sub_m = df[df.model == model]
        for prov, g in sub_m.groupby("province_en"):
            r = compute_confusion(g, "pred_day", 37.5)
            prov_rows.append({"model": model, "province_en": prov, **r})
    prov_df = pd.DataFrame(prov_rows).sort_values(["model", "recall"])
    prov_path = os.path.join(_ROOT, "models", f"alert_capability_per_province_{TIMESTAMP}.csv")
    prov_df.to_csv(prov_path, index=False)

    print(f"\n{'='*90}\nรายจังหวัด (เกณฑ์ 37.5, วิธีค่าพยากรณ์ตรงๆ) เรียงจาก recall น้อยไปมาก\n{'='*90}")
    for model in MODELS:
        print(f"\n--- {model.upper()} ---")
        d = prov_df[prov_df.model == model][["province_en", "n_events", "recall", "precision", "f1", "false_alarm_rate"]].copy()
        for c in ("recall", "precision", "f1", "false_alarm_rate"):
            d[c] = (d[c] * 100).round(1)
        print(d.to_string(index=False))

    flagged = prov_df[(prov_df.model == "lgb") & (prov_df.recall < RECALL_FLAG)]
    print(f"\nจังหวัดที่ recall < {RECALL_FLAG*100:.0f}% (LGB): {flagged.province_en.tolist() if len(flagged) else 'ไม่มี'}")
    accept = prov_df[(prov_df.model == "lgb") & (prov_df.recall >= RECALL_ACCEPT)]
    print(f"จังหวัดที่ recall >= {RECALL_ACCEPT*100:.0f}% (LGB): {len(accept)}/{prov_df.province_en.nunique()} จังหวัด")

    print(f"\nบันทึก: {pooled_path}\nบันทึก: {prov_path}")


if __name__ == "__main__":
    main()

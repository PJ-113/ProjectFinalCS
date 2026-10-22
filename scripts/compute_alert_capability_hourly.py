"""
ทดสอบความสามารถในการแจ้งเตือนแบบรายชั่วโมงภายใน 24 ชม.: เตือนถ้ามี horizon ใดก็ได้ใน 24 ชม.
ข้างหน้าที่ค่าพยากรณ์ (point forecast หรือ P95 upper bound) เกินเกณฑ์ 37.5 วัด recall/precision/
f1/false alarm rate เทียบกับ persistence
"""
import os
import pickle

import numpy as np
import pandas as pd

_ROOT = r"C:\FinalCS\ProjectFinalCS"
DATA_PATH = os.path.join(_ROOT, "data", "multiprovince_pm25_features_raw_spatial_k1_firehotspot_nbtrend_20260919_174805.csv")
MODEL_DIR = os.path.join(_ROOT, "models")
TAG = "20260919_203455"
THRESHOLD = 37.5


def split_train_calib_test(df, fracs=(0.65, 0.15, 0.20)):
    trains, calibs, tests = [], [], []
    for _, g in df.groupby("province_en", sort=False):
        g = g.sort_values("datetime_utc").reset_index(drop=True)
        n = len(g)
        n_train = int(n * fracs[0])
        n_calib = int(n * fracs[1])
        trains.append(g.iloc[:n_train])
        calibs.append(g.iloc[n_train:n_train + n_calib])
        tests.append(g.iloc[n_train + n_calib:])
    return (pd.concat(trains, ignore_index=True), pd.concat(calibs, ignore_index=True), pd.concat(tests, ignore_index=True))


def confusion_metrics(pred_alert, actual_alert):
    pred_alert = np.asarray(pred_alert, dtype=bool)
    actual_alert = np.asarray(actual_alert, dtype=bool)
    tp = int(np.sum(pred_alert & actual_alert))
    fp = int(np.sum(pred_alert & ~actual_alert))
    fn = int(np.sum(~pred_alert & actual_alert))
    tn = int(np.sum(~pred_alert & ~actual_alert))
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")
    false_alarm_rate = fp / (fp + tp) if (fp + tp) else float("nan")
    return {"n_events": int(np.sum(actual_alert)), "n": len(actual_alert), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "recall": recall, "precision": precision, "f1": f1, "false_alarm_rate": false_alarm_rate}


def main():
    df = pd.read_csv(DATA_PATH)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    df = df[~df["is_zero_shot"]].reset_index(drop=True)
    _, _, test_df = split_train_calib_test(df)

    pk = {t: pickle.load(open(os.path.join(MODEL_DIR, f"{t}_pm25_multiprovince_residual_bagged_quantile_{TAG}.pkl"), "rb"))
          for t in ("lgb", "xgb")}
    feats = pk["lgb"]["feature_cols"]
    assert pk["xgb"]["feature_cols"] == feats

    qhat = {t: pd.read_csv(os.path.join(MODEL_DIR, f"conformal_calibration_{t}_{TAG}.csv")).set_index("horizon")["q_hat"].to_dict()
            for t in ("lgb", "xgb")}

    target_cols = [f"pm25_t+{h}" for h in range(1, 25)]
    complete = test_df.dropna(subset=target_cols).reset_index(drop=True)
    print(f"test_df ทั้งหมด: {len(test_df)} แถว | มีค่าจริงครบ 24 horizon: {len(complete)} แถว")

    actual_alert = (complete[target_cols] > THRESHOLD).any(axis=1).to_numpy()
    persistence_alert = (complete["pm25"] > THRESHOLD).to_numpy()

    rows = []
    for tag in ("lgb", "xgb"):
        X = complete[feats]
        cur = complete["pm25"].to_numpy()
        point_exceed = np.zeros(len(complete), dtype=bool)
        p95_exceed = np.zeros(len(complete), dtype=bool)
        for h in range(1, 25):
            model_q50 = pk[tag]["models"][h][0.5]
            model_q95 = pk[tag]["models"][h][0.95]
            pred_q50 = model_q50.predict(X) + cur
            pred_q95 = model_q95.predict(X) + cur + qhat[tag][h]
            point_exceed |= (pred_q50 > THRESHOLD)
            p95_exceed |= (pred_q95 > THRESHOLD)

        m_point = confusion_metrics(point_exceed, actual_alert)
        m_p95 = confusion_metrics(p95_exceed, actual_alert)
        for method, m in (("point_forecast", m_point), ("p95_upper", m_p95)):
            rows.append({"model": tag, "method": method, **m})
        print(f"\n{tag.upper()} point_forecast: recall={m_point['recall']:.4f} precision={m_point['precision']:.4f} "
              f"f1={m_point['f1']:.4f} false_alarm={m_point['false_alarm_rate']:.4f}")
        print(f"{tag.upper()} p95_upper:      recall={m_p95['recall']:.4f} precision={m_p95['precision']:.4f} "
              f"f1={m_p95['f1']:.4f} false_alarm={m_p95['false_alarm_rate']:.4f}")

    m_persist = confusion_metrics(persistence_alert, actual_alert)
    rows.append({"model": "persistence", "method": "persistence", **m_persist})
    print(f"\nPERSISTENCE:        recall={m_persist['recall']:.4f} precision={m_persist['precision']:.4f} "
          f"f1={m_persist['f1']:.4f} false_alarm={m_persist['false_alarm_rate']:.4f}")

    out = pd.DataFrame(rows)
    out_path = os.path.join(MODEL_DIR, f"alert_capability_hourly_window_{TAG}.csv")
    out.to_csv(out_path, index=False)
    print(f"\nบันทึกที่: {out_path}")


if __name__ == "__main__":
    main()

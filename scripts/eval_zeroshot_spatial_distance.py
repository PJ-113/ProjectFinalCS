"""
Task C (feedback อาจารย์ ข้อ 1, ต่อยอดจาก task B): ทดสอบว่าโมเดลเรียนรู้ spatial pattern จริง
ไม่ใช่แค่จำจังหวัดที่เห็นตอนเทรน — สมมติฐาน: จังหวัด zero-shot ที่อยู่ใกล้จังหวัดที่เทรนจริง ควรมี
MAE ต่ำกว่าจังหวัดที่อยู่ไกล (ถ้าจริง = สอดคล้องกับ task B ที่พบว่า PM2.5 correlation ลดลงตาม
ระยะทาง, r=-0.644)

reuse `eval_zeroshot_multiprovince.py` ทั้งชุด (FEATURE_COLS, HELD_OUT, split_train_test,
fit_p50) ผ่าน importlib แทนที่จะเขียนซ้ำ — ต่างจากสคริปต์เดิมตรงที่: (1) evaluate แยกต่อจังหวัด
held-out แทนที่จะรวมเป็นก้อนเดียว (สคริปต์เดิมคำนวณ MAE รวม 3 จังหวัดเป็นค่าเดียวต่อ horizon)
(2) save per-horizon ผลลง CSV จริง (สคริปต์เดิมคำนวณแล้วทิ้ง) (3) เพิ่มระยะทางไปจังหวัดเทรนที่
ใกล้ที่สุดต่อจังหวัด held-out

ข้อจำกัดสำคัญที่ต้องระบุชัด: มีแค่ 3 จังหวัด held-out = 3 จุดข้อมูล ไม่พอสำหรับ statistical
significance test ใดๆ — ผลเป็นแค่เชิงคุณภาพ (illustrative) ใช้ประกอบ task B ไม่ใช่หลักฐานยืนยัน
เดี่ยวๆ
"""
import importlib.util
import json
import os

import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REGISTRY_PATH = os.path.join(_ROOT, "data", "province_station_registry.json")
MODEL_DIR = os.path.join(_ROOT, "models")
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, os.path.join(_ROOT, "scripts", filename))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def eval_per_province(model, eval_df, feature_cols, target_prefix="pm25_t+"):
    """ประเมิน MAE/R2 ต่อ (province, horizon) จากโมเดลที่เทรนแล้ว 1 ตัว (ไม่ retrain ต่อ horizon
    เพราะโมเดล zero-shot ในสคริปต์ต้นฉบับ fit แยกต่อ horizon อยู่แล้ว — ฟังก์ชันนี้รับโมเดลต่อ
    horizon มาทีละตัว)"""
    rows = []
    for province, g in eval_df.groupby("province_en"):
        target_col = target_prefix
        g_valid = g.dropna(subset=[target_col])
        if g_valid.empty:
            continue
        X = g_valid[feature_cols]
        current = g_valid["pm25"].values
        actual = g_valid[target_col].values
        pred = model.predict(X) + current
        rows.append({
            "province_en": province, "mae": mean_absolute_error(actual, pred),
            "r2": r2_score(actual, pred), "n": len(g_valid),
        })
    return rows


def main():
    zs = load_module("eval_zeroshot_multiprovince", "eval_zeroshot_multiprovince.py")
    reg_mod = load_module("province_registry", "province_registry.py")
    haversine_km = reg_mod.haversine_km

    with open(REGISTRY_PATH, encoding="utf-8") as f:
        registry = json.load(f)
    provinces_meta = {p["province_en"]: p for p in registry["provinces"]}

    print(f"โหลดข้อมูล: {zs.DATA_PATH}")
    df = pd.read_csv(zs.DATA_PATH)
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"])
    df = df[~df["is_zero_shot"]].reset_index(drop=True)

    zero_shot_pool = df[~df["province_en"].isin(zs.HELD_OUT)].reset_index(drop=True)  # 14 จังหวัด
    train_zs, _ = zs.split_train_test(zero_shot_pool)

    held_out_only = df[df["province_en"].isin(zs.HELD_OUT)].reset_index(drop=True)
    _, held_out_test = zs.split_train_test(held_out_only)
    print(f"train (14 จังหวัด zero-shot pool): {len(train_zs)} แถว")
    print(f"test (3 จังหวัด held-out: {zs.HELD_OUT}): {len(held_out_test)} แถว")

    trained_provinces = sorted(zero_shot_pool["province_en"].unique().tolist())
    dist_rows = []
    for held_out_province in zs.HELD_OUT:
        p_meta = provinces_meta[held_out_province]
        dists = {
            tp: haversine_km(p_meta["lat"], p_meta["lon"], provinces_meta[tp]["lat"], provinces_meta[tp]["lon"])
            for tp in trained_provinces
        }
        nearest_province = min(dists, key=dists.get)
        dist_rows.append({
            "held_out_province": held_out_province,
            "nearest_trained_province": nearest_province,
            "distance_to_nearest_trained_km": dists[nearest_province],
        })
    dist_df = pd.DataFrame(dist_rows)
    print("\nระยะทางจากแต่ละจังหวัด held-out ไปจังหวัดเทรนที่ใกล้ที่สุด:")
    print(dist_df.to_string(index=False))

    per_horizon_rows = []
    for kind in ("lgb", "xgb"):
        print(f"\n=== {kind.upper()} — เทรนแยกต่อ horizon บน 14 จังหวัด zero-shot pool ===")
        for h in range(1, 25):
            target_col = f"pm25_t+{h}"
            tr = train_zs.dropna(subset=[target_col])
            X_tr, y_tr_resid = tr[zs.FEATURE_COLS], tr[target_col] - tr["pm25"]
            model = zs.fit_p50(X_tr, y_tr_resid, kind)

            ev = held_out_test.dropna(subset=[target_col])
            for row in eval_per_province(model, ev, zs.FEATURE_COLS, target_col):
                row.update({"model": kind, "horizon": h})
                per_horizon_rows.append(row)
        print(f"  {kind}: ประเมินครบ 24 horizon x 3 จังหวัด held-out")

    result_df = pd.DataFrame(per_horizon_rows)
    out_path = os.path.join(MODEL_DIR, f"zeroshot_spatial_distance_per_horizon_{TIMESTAMP}.csv")
    result_df.to_csv(out_path, index=False)
    print(f"\nบันทึก per-horizon per-province ที่: {out_path}")

    summary = result_df.groupby(["model", "province_en"])["mae"].mean().reset_index()
    summary = summary.merge(dist_df, left_on="province_en", right_on="held_out_province")

    print("\n" + "=" * 90)
    print("สรุป: MAE เฉลี่ย 24h ต่อจังหวัด held-out เทียบกับระยะทางไปจังหวัดเทรนที่ใกล้ที่สุด")
    print("=" * 90)
    for kind in ("lgb", "xgb"):
        sub = summary[summary["model"] == kind].sort_values("distance_to_nearest_trained_km")
        print(f"\n{kind.upper()}:")
        print(sub[["province_en", "nearest_trained_province", "distance_to_nearest_trained_km", "mae"]]
              .to_string(index=False))
        corr = sub["distance_to_nearest_trained_km"].corr(sub["mae"])
        print(f"  correlation(distance, mae) = {corr:.3f} (n=3 จุด — เชิงคุณภาพเท่านั้น ไม่มีนัยสำคัญทางสถิติ)")
        if corr > 0:
            print("  -> ทิศทางตรงสมมติฐาน: จังหวัดใกล้แม่นกว่าจังหวัดไกล")
        else:
            print("  -> ทิศทางขัดสมมติฐาน หรือ noise จากตัวอย่างน้อยเกินไปที่จะสรุป")

    summary_path = os.path.join(MODEL_DIR, f"zeroshot_spatial_distance_summary_{TIMESTAMP}.csv")
    summary.to_csv(summary_path, index=False)
    print(f"\nบันทึกตารางสรุปที่: {summary_path}")


if __name__ == "__main__":
    main()

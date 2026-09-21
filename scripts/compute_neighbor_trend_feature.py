"""
เพิ่ม nb_trend_12h = pm25_neighbor_avg(t) - pm25_neighbor_avg(t-12h) ต่อจังหวัด (time-indexed ไม่ใช่ positional เพราะข้อมูลมี gap)
ลงใน dataset ที่มี fire_hotspot แล้ว (ผลจาก validate_feature_refinements.py + validate_neighbor_trend_variants.py:
หน้าต่าง 12h ดีที่สุดเล็กน้อย MAE -2.5%/-2.9% บน calib; หน้าต่างอื่นใกล้เคียงกัน) ใช้เฉพาะข้อมูล <= t ไม่ leak อนาคต
output: ไฟล์ใหม่ (ไม่ทับ) — DATA_PATH ของ train_multiprovince_models.py จะชี้มาที่ไฟล์นี้
ตรวจซ้ำด้วย np.allclose กับ recompute ผ่าน pivot (วิธีอิสระ)
"""
import os

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOURCE = os.path.join(_ROOT, "data", "multiprovince_pm25_features_raw_spatial_k1_with_firehotspot_20260918_155102.csv")
TIMESTAMP = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
OUT = os.path.join(_ROOT, "data", f"multiprovince_pm25_features_raw_spatial_k1_firehotspot_nbtrend_{TIMESTAMP}.csv")
HOURS = 12


def main():
    df = pd.read_csv(SOURCE)
    df["_dt"] = pd.to_datetime(df["datetime_utc"], utc=True)
    n0 = len(df)

    lag = df[["province_en", "_dt", "pm25_neighbor_avg"]].copy()
    lag["_dt"] = lag["_dt"] + pd.Timedelta(hours=HOURS)
    lag = lag.rename(columns={"pm25_neighbor_avg": "_lag"})
    out = df.merge(lag, on=["province_en", "_dt"], how="left")
    assert len(out) == n0, "merge ทำให้จำนวนแถวเปลี่ยน"
    out["nb_trend_12h"] = out["pm25_neighbor_avg"] - out["_lag"]

    # recompute อิสระผ่าน pivot (province x time)
    piv = df.pivot_table(index="_dt", columns="province_en", values="pm25_neighbor_avg")
    shifted = piv.reindex(piv.index - pd.Timedelta(hours=HOURS))
    shifted.index = piv.index
    ref = (piv - shifted).stack().rename("ref").reset_index()
    chk = out[["_dt", "province_en", "nb_trend_12h"]].merge(ref, on=["_dt", "province_en"], how="left")
    ok = np.allclose(chk["nb_trend_12h"].fillna(-999).values, chk["ref"].fillna(-999).values)
    assert ok, "nb_trend_12h ไม่ตรงกับ recompute"
    print("nb_trend_12h ตรงกับ recompute ผ่าน pivot (allclose)")

    out = out.drop(columns=["_dt", "_lag"])
    print(f"NaN nb_trend_12h: {out['nb_trend_12h'].isna().mean():.3%}  (12 ชม.แรกของแต่ละจังหวัด/ช่วง gap) "
          f"mean={out['nb_trend_12h'].mean():.3f} std={out['nb_trend_12h'].std():.3f}")
    out.to_csv(OUT, index=False)
    print(f"บันทึกที่: {OUT}  ({len(out)} แถว)")


if __name__ == "__main__":
    main()

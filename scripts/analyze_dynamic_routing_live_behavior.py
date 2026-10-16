"""
พฤติกรรมจริงของ Dynamic Routing (objective 2 ของ proposal) — สัดส่วนที่ select_model() ใน app.py
เลือก LightGBM vs XGBoost vs naive_fallback (circuit breaker) จริงในการใช้งาน (ไม่ใช่ backtest)

ดึงจาก prediction_log ตั้งแต่ deploy v3 (2026-09-19) เท่านั้น (แถวก่อนหน้าเป็นโมเดล v2/เก่ากว่าปนกัน)
"""
import os

import pandas as pd
import psycopg2

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SINCE_UTC = "2026-09-19"


def load_env():
    env = {}
    with open(os.path.join(_ROOT, ".env"), encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                env[k.strip()] = v.strip()
    return env


def main():
    env = load_env()
    conn = psycopg2.connect(host=env["POSTGRES_HOST"], port=env["POSTGRES_PORT"], dbname=env["POSTGRES_DB"],
                             user=env["POSTGRES_USER"], password=env["POSTGRES_PASSWORD"])
    cur = conn.cursor()
    cur.execute(
        "SELECT model_used, horizon, province, predicted_at_utc FROM prediction_log WHERE predicted_at_utc >= %s",
        (SINCE_UTC,),
    )
    df = pd.DataFrame(cur.fetchall(), columns=["model_used", "horizon", "province", "predicted_at_utc"])
    cur.close()
    conn.close()

    print(f"แถวทั้งหมดตั้งแต่ deploy v3 ({SINCE_UTC}): {len(df)}")
    print(f"รุ่นโมเดลที่พบ: {sorted(df.model_used.unique())}\n")

    print("=" * 70)
    print("สัดส่วนที่ระบบเลือกใช้แต่ละโมเดล (ทุก horizon รวมกัน)")
    print("=" * 70)
    counts = df.model_used.value_counts()
    pct = (counts / len(df) * 100).round(2)
    print(pd.DataFrame({"n": counts, "%": pct}))

    print("\n" + "=" * 70)
    print("สัดส่วนแยกตาม horizon bucket (เฉพาะโมเดล v3 vs naive_fallback)")
    print("=" * 70)
    df["hbucket"] = pd.cut(df.horizon, bins=[0, 1, 6, 12, 18, 24],
                            labels=["t+1", "t+2..6", "t+7..12", "t+13..18", "t+19..24"])
    tab = pd.crosstab(df.hbucket, df.model_used, normalize="index") * 100
    print(tab.round(1))

    print("\n" + "=" * 70)
    print("จำนวนจังหวัดที่มี naive_fallback อย่างน้อย 1 ครั้ง")
    print("=" * 70)
    nf = df[df.model_used == "naive_fallback"]
    print(f"{nf.province.nunique()} จังหวัด จาก {df.province.nunique()} จังหวัดทั้งหมดที่มีข้อมูล")
    if len(nf):
        print(nf.province.value_counts())


if __name__ == "__main__":
    main()

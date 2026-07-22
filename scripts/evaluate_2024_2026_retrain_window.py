"""
ประเมินว่าถ้าใช้เฉพาะช่วง 2024-01-01 ถึง 2026-07-02 (missing rate ต่ำกว่าตามที่เจอใน
scripts/check_openaq_missing_rate_5y.py) จะได้ dataset สุดท้ายหน้าตาเป็นอย่างไร:
- จำนวน row สุดท้ายหลัง impute gap สั้น (<=48h) + ตัด gap ยาว (>48h) ทิ้ง
- coverage รายเดือน โดยเฉพาะเดือน ธ.ค.-มี.ค. (ฤดูหมอกควันภาคอีสาน)
- ถ้าเดือนฤดูหมอกควันปีไหน missing สูงเกินไป เสนอทางเลือกดึงปีอื่นมาเสริมเฉพาะเดือนนั้น

ใช้ cache จาก scripts/_openaq_raw_5y_cache.csv (ไม่ต้องดึง API ซ้ำ)
"""
import pandas as pd

import os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_PATH = os.path.join(_ROOT, r"scripts\_openaq_raw_5y_cache.csv")
GAP_CUTOFF_HOURS = 48
ORIGINAL_ROW_COUNT = 37909  # จำนวนแถวเดิมทั้งหมดใน data/khonkaen_pm25_features_final.csv

raw = pd.read_csv(CACHE_PATH)
raw["datetime_utc"] = pd.to_datetime(raw["datetime_utc"])
raw = raw.sort_values("datetime_utc").drop_duplicates(subset=["datetime_utc"]).reset_index(drop=True)

WINDOW_START = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
WINDOW_END = pd.Timestamp("2026-07-02 00:00:00", tz="UTC")

window_raw = raw[(raw["datetime_utc"] >= WINDOW_START) & (raw["datetime_utc"] <= WINDOW_END)].copy()

# ===== 1) จำนวน row สุดท้ายหลัง impute gap สั้น + ตัด gap ยาว =====
full_grid = pd.date_range(WINDOW_START, WINDOW_END, freq="h", tz="UTC")
grid_df = pd.DataFrame({"datetime_utc": full_grid})
grid_df = grid_df.merge(window_raw[["datetime_utc", "pm25"]], on="datetime_utc", how="left")
grid_df["has_data"] = grid_df["pm25"].notna()

# หา gap run (ชั่วโมงที่ขาดต่อเนื่องกัน)
grid_df["missing"] = ~grid_df["has_data"]
grid_df["gap_group"] = (grid_df["missing"] != grid_df["missing"].shift()).cumsum()

gap_runs = grid_df[grid_df["missing"]].groupby("gap_group").size()
short_gap_groups = gap_runs[gap_runs <= GAP_CUTOFF_HOURS].index
long_gap_groups = gap_runs[gap_runs > GAP_CUTOFF_HOURS].index

n_total_hours = len(grid_df)
n_real = grid_df["has_data"].sum()
n_short_gap_hours = grid_df[grid_df["gap_group"].isin(short_gap_groups)].shape[0]
n_long_gap_hours = grid_df[grid_df["gap_group"].isin(long_gap_groups)].shape[0]

final_row_count = n_real + n_short_gap_hours  # real + imputed-short-gap; long-gap hours ถูกตัดทิ้ง

print("=" * 90)
print(f"STEP 1: จำนวน row สุดท้าย ({WINDOW_START.date()} -> {WINDOW_END.date()})")
print("=" * 90)
print(f"ชั่วโมงทั้งหมดในช่วงนี้ (ถ้าครบ 100%): {n_total_hours}")
print(f"ชั่วโมงที่มีข้อมูลจริง (raw)          : {n_real} ({n_real/n_total_hours*100:.1f}%)")
print(f"ชั่วโมงที่ขาด <= {GAP_CUTOFF_HOURS}h (impute ได้)   : {n_short_gap_hours}")
print(f"ชั่วโมงที่ขาด > {GAP_CUTOFF_HOURS}h (ตัดทิ้ง)        : {n_long_gap_hours}")
print(f"จำนวน gap run ที่ยาวเกิน {GAP_CUTOFF_HOURS}h: {len(long_gap_groups)} ครั้ง")
print()
print(f">>> จำนวน row สุดท้ายที่ใช้ได้ (real + imputed-short-gap): {final_row_count}")
print(f">>> เทียบกับ dataset เดิม {ORIGINAL_ROW_COUNT} แถว = {final_row_count/ORIGINAL_ROW_COUNT*100:.1f}%")

# ===== 2) coverage รายเดือน =====
print("\n" + "=" * 90)
print("STEP 2: coverage รายเดือน (% ชั่วโมงที่มีข้อมูลจริง ไม่นับ imputed)")
print("=" * 90)

grid_df["year_month"] = grid_df["datetime_utc"].dt.to_period("M")
monthly = grid_df.groupby("year_month").agg(
    expected=("has_data", "size"),
    real=("has_data", "sum"),
).reset_index()
monthly["coverage_pct"] = monthly["real"] / monthly["expected"] * 100
monthly["month_num"] = monthly["year_month"].dt.month
monthly["is_haze_season"] = monthly["month_num"].isin([12, 1, 2, 3])

for _, row in monthly.iterrows():
    tag = " <-- ฤดูหมอกควัน" if row["is_haze_season"] else ""
    print(f"{str(row['year_month']):>10}: {row['coverage_pct']:>6.1f}% ({int(row['real'])}/{int(row['expected'])}){tag}")

print("\n--- สรุปเฉพาะฤดูหมอกควัน (ธ.ค.-มี.ค.) ---")
haze = monthly[monthly["is_haze_season"]]
print(haze[["year_month", "coverage_pct"]].to_string(index=False))
print(f"\nเฉลี่ย coverage ช่วงฤดูหมอกควันทั้งหมด: {haze['coverage_pct'].mean():.1f}%")

# ===== 3) เช็คปีอื่น (2021-2023) เผื่อต้องดึงมาเสริมเฉพาะเดือนฤดูหมอกควันที่ขาด =====
print("\n" + "=" * 90)
print("STEP 3: เทียบ coverage เดือนฤดูหมอกควัน ปีอื่น (2021-2023) เผื่อต้องดึงมาเสริม")
print("=" * 90)

full_grid_all = pd.date_range(raw["datetime_utc"].min(), raw["datetime_utc"].max(), freq="h", tz="UTC")
grid_all = pd.DataFrame({"datetime_utc": full_grid_all})
grid_all = grid_all.merge(raw[["datetime_utc", "pm25"]], on="datetime_utc", how="left")
grid_all["has_data"] = grid_all["pm25"].notna()
grid_all["year_month"] = grid_all["datetime_utc"].dt.to_period("M")
grid_all["month_num"] = grid_all["datetime_utc"].dt.month

monthly_all = grid_all.groupby("year_month").agg(expected=("has_data", "size"), real=("has_data", "sum")).reset_index()
monthly_all["coverage_pct"] = monthly_all["real"] / monthly_all["expected"] * 100
monthly_all["month_num"] = monthly_all["year_month"].dt.month
monthly_all["year"] = monthly_all["year_month"].dt.year
haze_all = monthly_all[monthly_all["month_num"].isin([12, 1, 2, 3])]

print("coverage % เดือนฤดูหมอกควัน แยกตามปี (ทุกปีที่มีข้อมูล):")
pivot = haze_all.pivot_table(index="month_num", columns="year", values="coverage_pct")
print(pivot.round(1).to_string())

print("\n" + "=" * 90)
print("SUMMARY")
print("=" * 90)
low_coverage_haze = haze[haze["coverage_pct"] < 40]
if len(low_coverage_haze) > 0:
    print("เดือนฤดูหมอกควันใน 2024-2026 ที่ coverage ต่ำกว่า 40% (น่ากังวล):")
    print(low_coverage_haze[["year_month", "coverage_pct"]].to_string(index=False))
else:
    print("ไม่มีเดือนฤดูหมอกควันใน 2024-2026 ที่ coverage ต่ำกว่า 40%")

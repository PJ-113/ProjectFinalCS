# PM2.5 24-Hour Forecast & Health Advisory (20 จังหวัดภาคอีสาน)

ระบบพยากรณ์ค่าฝุ่น PM2.5 ล่วงหน้า 24 ชั่วโมง ครอบคลุม 20 จังหวัดภาคตะวันออกเฉียงเหนือ (โครงงาน
capstone) — ใช้โมเดลเดียวที่ฝึกสอนร่วมจากหลายจังหวัด (Pooled Model) หลัง **Dynamic Routing**,
อธิบายผลด้วย SHAP + ข้อความภาษาไทย (NLG), ประเมินความเสี่ยงเฉพาะบุคคล และแสดงผลผ่าน
Progressive Web App (Vue 3)

## คุณสมบัติหลัก

- **พยากรณ์ 24 ชม.** ด้วย LightGBM / XGBoost (Pooled Model เดียวฝึกสอนร่วมจาก 17 จังหวัด ใช้
  พิกัด lat/lon เป็นฟีเจอร์ จึงพยากรณ์อีก 3 จังหวัดที่ไม่มีข้อมูลฝึกสอนได้ — Zero-shot) แบบ Direct
  Multi-Horizon (24 horizon × 3 ควอนไทล์ต่อโมเดล) พร้อมช่วงพยากรณ์ P05–P95 (Conformal Prediction)
- **Dynamic Routing** — เลือกโมเดลที่ดีที่สุดต่อจังหวัดและ horizon โดยอัตโนมัติ 3 ระดับ: ความสดของ
  ข้อมูลเพื่อนบ้าน → Circuit Breaker (ข้อมูลสูญหายเกินเกณฑ์) → เทียบ Rolling R² ระหว่าง XGBoost/
  LightGBM — ตกไปใช้ Persistence (คงค่าล่าสุด) เมื่อข้อมูลไม่พร้อม
- **ไม่มี Calibration Layer** — ทั้งข้อมูลฝึกสอนและข้อมูล live ดึงจาก **OpenAQ v3** สเกลเดียวกันอยู่แล้ว
- **กลไกจัดการข้อมูลสูญหาย** — เติมค่าสถานีเงียบด้วย IDW+Ridge Regression (H-lite), ทางสำรองเมื่อ
  OpenAQ/Open-Meteo ดึงข้อมูลไม่ได้ชั่วคราว
- **Explainable Forecast** — SHAP + คำอธิบายภาษาไทยผ่าน Google Gemini (มี NLG safety check
  ตรวจสอบว่าข้อความตรงกับระดับคุณภาพอากาศจริงก่อนแสดงผล)
- **Personalized Risk Score** — ผสานค่าพยากรณ์ + กลุ่มความเสี่ยง + ความหนักกิจกรรมเป็นคะแนน 0–100
  พร้อมหาช่วงเวลาที่เหมาะกับกิจกรรมกลางแจ้งที่สุดในแต่ละวัน
- **จุดความร้อนไฟป่า** — ใช้ข้อมูลจาก NASA FIRMS เป็นฟีเจอร์เสริมการพยากรณ์
- **PWA** — ติดตั้งได้/ออฟไลน์, ระบบผู้ใช้ (login/register/guest/admin), แผนที่ภาพรวมทุกจังหวัด,
  ประวัติการแจ้งเตือนตามจังหวัดหลักที่ตั้งไว้

## Tech stack

| ส่วน | เทคโนโลยี |
|---|---|
| Backend | Flask (`app.py`), APScheduler, psycopg2 |
| ML | LightGBM, XGBoost, scikit-learn, SHAP |
| Database | PostgreSQL (`pm25_forecast`) |
| Frontend | Vue 3 + Vite + vite-plugin-pwa (`frontend/`) |
| NLG | Google Gemini API |
| แหล่งข้อมูล | OpenAQ v3 (PM2.5 — ทั้งฝึกสอนและ live), Open-Meteo (สภาพอากาศ), NASA FIRMS (จุดความร้อนไฟป่า) |

## การติดตั้ง & รัน

```bash
# 1) Python dependencies
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

# 2) ตั้งค่าความลับ — คัดลอก .env.example เป็น .env แล้วเติมค่าจริง
copy .env.example .env
# ต้องมี OPENAQ_KEY, GEMINI_KEY, SECRET_KEY, POSTGRES_* ครบ — FIRMS_KEY ไม่บังคับ (ไม่ใส่ก็รันได้
# แค่ฟีเจอร์จุดความร้อนไฟป่าจะปิดตัวเองแทน)

# 3) โมเดล — ไฟล์ .pkl ของ LightGBM/XGBoost ไม่ได้ขึ้น git (ไฟล์ใหญ่เกินลิมิตของ GitHub)
#    ดูวิธีขอไฟล์/checksum ที่หัวข้อ "ไฟล์โมเดลที่ไม่ได้อยู่ใน git" ด้านล่างก่อนรันเซิร์ฟเวอร์

# 4) Build frontend (จำเป็น — app.py เสิร์ฟจาก frontend/dist)
cd frontend && npm install && npm run build && cd ..

# 5) รันเซิร์ฟเวอร์ (พอร์ต 5000)
python app.py
```

> ระหว่างพัฒนา frontend: `cd frontend && npm run dev` (hot-reload + proxy `/api` ไป :5000)
> รันคู่กับ `python app.py`

## โครงสร้างโปรเจกต์

```
app.py                 # Flask backend (REST API + เสิร์ฟ SPA + Dynamic Routing + NLG)
requirements.txt       # Python dependencies
frontend/              # Vue 3 SPA + PWA (build ด้วย Vite)
models/                # โมเดล .pkl (LightGBM/XGBoost pooled) + ตาราง conformal calibration
data/                  # ชุดข้อมูลเทรน (.csv) + ทะเบียนจังหวัด (province_station_registry.json)
scripts/               # สคริปต์เทรน/วิเคราะห์/ทดสอบ (แต่ละไฟล์รันเดี่ยวได้ ดูสคริปต์สำคัญด้านล่าง)
figures/                # กราฟ/ภาพผลการทดลอง (ส่วนใหญ่ไม่ได้ขึ้น git — regenerate ได้จาก scripts/)
```

## สคริปต์สำคัญ (`scripts/`)

`scripts/` มีสคริปต์ one-off (ทดลอง/ตรวจสอบ) อีกจำนวนมากที่ไม่ได้ลงไว้ที่นี่ — รายการด้านล่างคือไฟล์ที่จำเป็นต่อการรัน

**Pipeline เตรียมข้อมูล + เทรนโมเดล** (รันตามลำดับถ้าต้องสร้างโมเดลใหม่ ดูหัวข้อ "ถ้าไฟล์หาย" ด้านล่าง):
- `province_registry.py` — สร้าง registry 20 จังหวัด (query OpenAQ v3 โดยตรง)
- `build_raw_dataset_multiprovince.py` → `augment_raw_dataset_with_spatial_lag_k1.py` → `compute_fire_hotspot_feature.py` → `compute_neighbor_trend_feature.py` — 4 ขั้นตอนเตรียมข้อมูล (เรียกรวมด้วย `rebuild_dataset_clean.py`)
- `verify_clean_dataset_alignment.py` — ตรวจสอบ timestamp alignment หลัง rebuild
- `annotate_registry_timestamp_offset.py` — วัด/เขียน offset เวลาของแต่ละสถานีลง registry
- `train_multiprovince_models.py` — **สคริปต์เทรนหลัก** (LightGBM+XGBoost pooled v3, source of truth ของ `FEATURE_COLS`/`DATA_PATH`/split ที่สคริปต์อื่นโหลดผ่าน `importlib`)

**ประเมินผล/ใช้งานจริงใน `app.py`**:
- `shap_analysis_clean.py` — **ถูก `app.py` import ตรง** สำหรับคำอธิบาย SHAP
- `backtest_recent_multiprovince.py`, `eval_zeroshot_multiprovince.py`, `eval_zeroshot_real_provinces_clean.py` — ประเมินผล/zero-shot
- `validate_features_timeblocked_cv.py` — วิธีมาตรฐานของโปรเจกต์สำหรับตัดสินใจเพิ่ม/ตัดฟีเจอร์
- `experiment_imputation_accuracy.py`, `export_imputation_h_lite.py` — สูตร/config การเติมค่าสถานีเงียบ (H-lite)
- `summarize_target_latency.py` — คำนวณค่าคงที่ `IMPUTATION_MIN_K` จาก latency log จริง
- `promote_admin.py` — ops utility สร้าง admin user

**เทส** (รันมือ ไม่มี pytest): ไฟล์ `test_*.py` ทั้งหมด 14 ไฟล์

## ไฟล์โมเดลที่ไม่ได้อยู่ใน git

โมเดลที่ `app.py` ใช้อยู่ (v3, `_20260919_203455`) มีขนาดเกิน 100 MB ต่อไฟล์ ซึ่งเป็นขีดจำกัดของ GitHub จึงไม่ได้ commit
ไฟล์ conformal และตารางสรุปผลของโมเดลนี้ (ขนาดเล็ก) อยู่ใน git แล้ว (`models/conformal_calibration_{lgb,xgb}_20260919_203455.csv`,
`models/multiprovince_eval_summary_20260919_203455.csv`)

### ไฟล์ที่ต้องมีจึงจะรัน `python app.py` ได้

วางไว้ในโฟลเดอร์ `models/` (ชื่อไฟล์ต้องตรงตามนี้ เพราะ `app.py` อ้างชื่อเหล่านี้)

| ไฟล์ | ขนาด (ไบต์) | SHA-256 |
|---|---|---|
| `lgb_pm25_multiprovince_residual_bagged_quantile_20260919_203455.pkl` | 167,347,473 | `6596666cd72f450613b6d2a15896700dc936c222d3e9253742e4edff897ad898` |
| `xgb_pm25_multiprovince_residual_bagged_quantile_20260919_203455.pkl` | 598,681,006 | `37ecdb91f0bb532873b4baed38e56bb3dcc96ff800225a9cfe65e16a520cb6d8` |

ตรวจว่าไฟล์ที่ดาวน์โหลด/กู้คืนมาไม่เสียหาย (ค่า SHA-256 ต้องตรงกับตาราง):

```
python -c "import hashlib,sys; h=hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest(); print(h)" models/<ชื่อไฟล์>.pkl
```

### ที่เก็บสำรอง

- GitHub Release : https://github.com/PJ-113/ProjectFinalCS/releases/tag/model-v3-20260919_203455 — อัปโหลดเมื่อ 2026-09-21 ขนาดตรงกับต้นฉบับ
  ดาวน์โหลดด้วย `gh release download model-v3-20260919_203455 --repo PJ-113/ProjectFinalCS --dir models`

### ถ้าไฟล์หาย

เทรนใหม่ได้จากสคริปต์ แต่ผลอาจไม่เหมือนเดิมทุกตัวเลข เพราะ OpenAQ อาจแก้ข้อมูลย้อนหลัง
1. `python scripts/rebuild_dataset_clean.py` (ดึงข้อมูลและสร้างชุดข้อมูลสะอาด พร้อม offset ป้ายเวลาจาก registry)
2. `python scripts/train_multiprovince_models.py` (20 feature, hyperparameter ที่จูนแล้วอยู่ในสคริปต์)
3. ปรับ path/ชื่อโมเดลใน `app.py` ให้ตรงกับไฟล์ใหม่ (`LGB_MODEL_PATH`, `XGB_MODEL_PATH`, `CONFORMAL_*`, `MODEL_LGB`, `MODEL_XGB`)

ไฟล์อื่นที่ไม่ได้อยู่ใน git และควรสำรองไว้ด้วยเพื่อความสามารถทำซ้ำ: ชุดข้อมูลเทรน
`data/multiprovince_pm25_features_raw_spatial_k1_firehotspot_nbtrend_20260919_174805.csv` และข้อมูลจุดความร้อน
`data/firms_hotspots_isaan_*.csv`

### ไฟล์เล็กที่ app.py อ่านเพิ่ม (อยู่ใน git)

`models/imputation_h_lite_20260924_204956.json` — ค่าสัมประสิทธิ์วิธีเติมข้อมูลสถานีเงียบ (H-lite) + ตาราง q(k,h) ของช่วงความเชื่อมั่น ไม่จำเป็นต่อการรัน: ถ้าไม่มี/อ่านไม่ได้ ระบบปิดฟีเจอร์เติมค่าเองแล้วทำงานแบบเดิม สร้างใหม่ด้วย `scripts/export_imputation_h_lite.py`

`models/weather_climatology_20260924_211741.json` — ตาราง climatology สภาพอากาศ (จังหวัด x เดือน x ชั่วโมง) ใช้เป็นทางสุดท้ายเมื่อดึง Open-Meteo ไม่ได้ ไม่จำเป็นต่อการรัน: ถ้าไม่มี/อ่านไม่ได้ ระบบปิดทางสุดท้ายนี้เอง (ดึงสภาพอากาศไม่ได้ก็ตอบ 500 เหมือนเดิม) สร้างใหม่ด้วย `scripts/export_weather_climatology.py` (เรียก Open-Meteo archive 20 ครั้ง)

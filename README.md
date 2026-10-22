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
#    ดูวิธีขอไฟล์/checksum ที่ docs/MODEL_FILES.md ก่อนรันเซิร์ฟเวอร์

# 4) Build frontend (จำเป็น — app.py เสิร์ฟจาก frontend/dist)
cd frontend && npm install && npm run build && cd ..

# 5) รันเซิร์ฟเวอร์ (พอร์ต 5000)
python app.py
```

> ระหว่างพัฒนา frontend: `cd frontend && npm run dev` (hot-reload + proxy `/api` ไป :5000)
> รันคู่กับ `python app.py`

เอกสาร API สำหรับเทส endpoint ทั้งหมด (ตัวอย่าง curl + response จริง) อยู่ที่
[`docs/API_TESTING.md`](docs/API_TESTING.md)

## โครงสร้างโปรเจกต์

```
app.py                 # Flask backend (REST API + เสิร์ฟ SPA + Dynamic Routing + NLG)
requirements.txt       # Python dependencies
frontend/              # Vue 3 SPA + PWA (build ด้วย Vite)
models/                # โมเดล .pkl (LightGBM/XGBoost pooled) + ตาราง conformal calibration
data/                  # ชุดข้อมูลเทรน (.csv) + ทะเบียนจังหวัด (province_station_registry.json)
scripts/               # สคริปต์เทรน/วิเคราะห์/ทดสอบ (แต่ละไฟล์รันเดี่ยวได้)
notebooks/             # notebook สำรวจ (historical)
docs/                  # ไดอะแกรม (use case, system architecture), ERD, proposal, API_TESTING.md
CHANGELOG.md           # ประวัติการเปลี่ยนแปลง (ทำไม ไม่ใช่แค่อะไร)
CLAUDE.md              # คู่มือสถาปัตยกรรม/ข้อควรระวังเชิงลึก
```

ดูไดอะแกรมได้ที่ [`docs/`](docs/): `usecase_diagram.puml`, `system_architecture.puml`/`.html`,
`database_schema.md` (ERD)

## หมายเหตุ

- โมเดลฝึกสอนและ live inference ใช้ **OpenAQ v3** แหล่งเดียวกันทั้งคู่ จึงไม่มี calibration layer
  (ประวัติ: ระบบเดิมก่อน 2026-09-13 ใช้ Air4Thai เป็น live source คนละสเกลกับข้อมูลฝึกสอน ต้องมี
  calibration — เลิกใช้ไปแล้ว รายละเอียดใน `CLAUDE.md`)
- ไฟล์ `.env` (ความลับ), โมเดล `.pkl` ขนาดใหญ่, และ `backups/` (มีข้อมูลผู้ใช้) ถูก `.gitignore` ไว้
  ไม่ขึ้น repo — ดูวิธีขอไฟล์โมเดลที่ `docs/MODEL_FILES.md`

# PM2.5 24-Hour Forecast & Health Advisory (Khon Kaen)

ระบบพยากรณ์ค่าฝุ่น PM2.5 ล่วงหน้า 24 ชั่วโมงสำหรับ จ.ขอนแก่น (โครงงาน capstone) — ใช้โมเดล
Machine Learning 3 ตัวหลัง **Dynamic Routing**, อธิบายผลด้วย SHAP + ข้อความภาษาไทย (NLG),
และแสดงผลผ่าน Progressive Web App (Vue 3)

## คุณสมบัติหลัก

- **พยากรณ์ 24 ชม.** ด้วย LightGBM / XGBoost / Prophet (direct multi-horizon 24 โมเดลต่อ 1 แบบจำลอง)
- **Dynamic Routing** — เลือกโมเดลที่ดีที่สุดต่อ horizon อัตโนมัติ (Circuit Breaker + rolling R²)
- **Calibration Layer** — แปลงค่า Air4Thai (live) ให้อยู่สเกลเดียวกับ OpenAQ (ข้อมูลเทรน)
- **Explainable Forecast** — SHAP + คำอธิบายภาษาไทยผ่าน Google Gemini (มี rule-based safety check)
- **Personalized Risk Score** — ผสานค่าพยากรณ์ + กลุ่มความเสี่ยง + ความหนักกิจกรรมเป็นคะแนน 0–100
- **PWA** — ติดตั้งได้/ออฟไลน์, ระบบผู้ใช้ (login/register/guest), ประวัติการแจ้งเตือน

## Tech stack

| ส่วน | เทคโนโลยี |
|---|---|
| Backend | Flask (`app.py`), APScheduler, psycopg2 |
| ML | LightGBM, XGBoost, Prophet, scikit-learn, SHAP |
| Database | PostgreSQL (`pm25_forecast`) |
| Frontend | Vue 3 + Vite + vite-plugin-pwa (`frontend/`) |
| NLG | Google Gemini API |
| แหล่งข้อมูล | Air4Thai (live), Open-Meteo (weather), OpenAQ (training) |

## การติดตั้ง & รัน

```bash
# 1) Python dependencies
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt

# 2) ตั้งค่าความลับ — คัดลอก .env.example เป็น .env แล้วเติมค่าจริง
copy .env.example .env

# 3) Build frontend (จำเป็น — app.py เสิร์ฟจาก frontend/dist)
cd frontend && npm install && npm run build && cd ..

# 4) รันเซิร์ฟเวอร์ (พอร์ต 5000)
python app.py
```

> ระหว่างพัฒนา frontend: `cd frontend && npm run dev` (hot-reload + proxy `/api` ไป :5000)
> รันคู่กับ `python app.py`

## โครงสร้างโปรเจกต์

```
app.py                 # Flask backend (REST API + เสิร์ฟ SPA + Dynamic Routing + NLG)
requirements.txt       # Python dependencies
frontend/              # Vue 3 SPA + PWA (build ด้วย Vite)
models/                # โมเดล .pkl + ไฟล์ bias/calibration
data/                  # ชุดข้อมูลเทรน (.csv)
scripts/               # สคริปต์เทรน/วิเคราะห์/ทดสอบ (แต่ละไฟล์รันเดี่ยวได้)
notebooks/             # notebook สำรวจ (historical)
docs/                  # ไดอะแกรม (use case, system architecture), ERD, proposal
CHANGELOG.md           # ประวัติการเปลี่ยนแปลง (ทำไม ไม่ใช่แค่อะไร)
CLAUDE.md              # คู่มือสถาปัตยกรรม/ข้อควรระวังเชิงลึก
```

ดูไดอะแกรมได้ที่ [`docs/`](docs/): `usecase_diagram.puml`, `system_architecture.puml`/`.html`,
`database_schema.md` (ERD)

## หมายเหตุ

- โมเดลเทรนบน **OpenAQ** แต่ inference ดึงจาก **Air4Thai** (สถานีเดียวกัน แต่ค่าต่างสเกล) จึงมี
  calibration layer คั่นก่อนสร้าง feature — รายละเอียดใน `CLAUDE.md`
- ไฟล์ `.env` (ความลับ) และ `backups/` (มีข้อมูลผู้ใช้) ถูก `.gitignore` ไว้ ไม่ขึ้น repo

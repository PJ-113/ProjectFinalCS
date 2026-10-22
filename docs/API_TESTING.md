# API Testing Guide

เอกสารนี้ไว้ใช้เทส endpoint ของ `app.py` ด้วย `curl` ตรงๆ (หรือ Postman/Thunder Client ก็ได้
แค่แปลง curl เป็น request ตามปกติ) ครอบคลุมทุก route จริงในโค้ด ตัวอย่าง response เป็นรูปแบบ
โดยย่อ (ตัดบางฟิลด์ที่ยาวออก) ไม่ใช่ copy ตรงจาก server

**Base URL**: `http://localhost:5000` (รัน `python app.py` ก่อน — ดู [README](../README.md))

**Session**: route ที่ต้อง login ใช้ Flask session cookie ไม่ใช่ token — เทสด้วย curl ต้องส่ง
`-c cookies.txt` ตอน login แล้ว `-b cookies.txt` ต่อ request ถัดไปเพื่อให้ cookie ติดไปด้วย

---

## 1. จังหวัดและภาพรวม (ไม่ต้อง login)

### `GET /api/provinces`

รายชื่อ 20 จังหวัดในระบบ

```bash
curl http://localhost:5000/api/provinces
```

```json
[
  { "province_th": "ขอนแก่น", "province_en": "khonkaen", "lat": 16.4419, "lon": 102.8360, "is_zero_shot": false },
  { "province_th": "มหาสารคาม", "province_en": "mahasarakham", "lat": 16.1830, "lon": 103.3000, "is_zero_shot": true }
]
```

### `GET /api/provinces/overview?horizon=`

สรุปค่าปัจจุบัน/ค่าพยากรณ์ที่ horizon ที่ขอของทุกจังหวัดพร้อมกัน (ใช้ทำแผนที่) — `horizon` เป็น
จำนวนเต็ม 0–24 (0 = ไม่ต้องมีค่าพยากรณ์ แสดงแค่ค่าปัจจุบัน), ไม่ใส่ param จะ default เป็น 0

```bash
curl "http://localhost:5000/api/provinces/overview?horizon=6"
```

```json
[
  {
    "province_th": "ขอนแก่น", "province_en": "khonkaen", "lat": 16.4419, "lon": 102.836, "is_zero_shot": false,
    "horizon": 6,
    "current_pm25": 23.7, "current_status": "ดี", "current_recorded_at_th": "2026-10-18 10:00",
    "forecast_pm25": 26.1, "forecast_status": "ดี",
    "forecast_target_time_th": "2026-10-18 16:00", "forecast_model_used": "xgboost_pooled_v3"
  }
]
```

ค่า `current_*`/`forecast_*` เป็น `null` ได้ถ้ายังไม่เคยมี scheduled job รันผ่านจังหวัดนั้นมาก่อน
(เช่น server เพิ่งสตาร์ต) — ไม่ error แต่ field จะเป็น null ทั้งก้อน

**Error**: `horizon` นอกช่วง 0–24 หรือไม่ใช่จำนวนเต็ม → `400`

```bash
curl -i "http://localhost:5000/api/provinces/overview?horizon=99"
# HTTP/1.1 400 BAD REQUEST
# {"error": "horizon ต้องเป็นจำนวนเต็ม 0-24: 99"}
```

---

## 2. พยากรณ์ (ไม่ต้อง login — risk_group จะเป็น "ทั่วไป" เสมอถ้าไม่ login)

### `GET /api/forecast?province=`

Endpoint หลัก — เรียก `get_forecast()` เต็มรูปแบบ (ดึงข้อมูล → Dynamic Routing → พยากรณ์ 24 ชม. →
บันทึกผล) จึงหนักกว่า endpoint อื่น ไม่ใส่ `province` จะ default เป็น `khonkaen`

```bash
curl "http://localhost:5000/api/forecast?province=khonkaen"
```

```json
{
  "province": { "th": "ขอนแก่น", "en": "khonkaen", "is_zero_shot": false },
  "current": { "pm25": 23.7, "datetime_th": "2026-10-18 10:00", "status": "ดี" },
  "current_weather": { "temp_c": 31.7, "humidity_pct": 54, "wind_speed_kmh": 4.6 },
  "forecast": [
    { "time": "11:00", "hour_thai": 11, "pm25": 24.1, "pm25_lo": 19.2, "pm25_hi": 29.8, "model_used": "xgboost_pooled_v3" }
  ],
  "best_time": { "time": "05:00", "pm25": 18.2, "...": "..." },
  "high_pollution_risk": false,
  "risk_horizons": [],
  "fallback_triggered_horizons": [],
  "data_quality": {
    "target_age_hours": 0.2, "target_stale": false,
    "series_age_hours": 0.0, "weather_source": "live", "weather_age_hours": 0.0,
    "neighbor_status": "ok", "neighbor_policy": "k1_exact", "neighbors_used": ["kalasin"],
    "fire_source": "live", "fire_count_24h": 0, "input_imputed": null
  },
  "daily_summary": { "date_thai": "2026-10-18", "daily_avg_pm25": 24.5, "explanation": "...", "used_fallback": false },
  "risk_group": "ทั่วไป",
  "risk_score_config": { "risk_group_offset": 0.0, "activity_offsets": { "...": "..." }, "breakpoints": [["...", "..."]] }
}
```

**Errors**:
- `province` ไม่ตรงกับ registry (เช่น `?province=bangkok`) → `400`
- สถานีไม่มีข้อมูลเลยใน 72 ชม. (OpenAQ ขัดข้อง) → `503` พร้อม `"data_unavailable": true`

```bash
curl -i "http://localhost:5000/api/forecast?province=bangkok"
# 400 {"error": "ไม่รู้จักจังหวัด: bangkok"}
```

### `GET /api/current_pm25?province=`

ค่า PM2.5 ล่าสุดอย่างเดียว ไม่พยากรณ์ — เบากว่า `/api/forecast` มาก ใช้ตอน poll บ่อยๆ

```bash
curl "http://localhost:5000/api/current_pm25?province=khonkaen"
```

```json
{ "pm25": 23.7, "datetime_th": "2026-10-18 10:00", "status": "ดี" }
```

---

## 3. เปรียบเทียบผลพยากรณ์ (ไม่ต้อง login ฝั่ง backend — ดูหมายเหตุท้ายไฟล์)

### `GET /api/comparison?horizon=&model=&province=`

ข้อมูลเทียบ predicted/actual รายจุดของ horizon เดียว — `horizon` 1–24 (ไม่ใส่ default = 1),
`model` คือ `all` หรือชื่อโมเดล (`lightgbm_pooled_v3`, `xgboost_pooled_v3`, `naive_fallback`, ฯลฯ —
ดู `MODEL_LGB`/`MODEL_XGB` ใน `app.py`), ไม่ใส่ default = `all`

```bash
curl "http://localhost:5000/api/comparison?horizon=1&model=all&province=khonkaen"
```

```json
{
  "horizon": 1, "model": "all",
  "points": [
    { "time": "10-17 09:00", "predicted": 22.1, "actual": 21.8, "model_used": "xgboost_pooled_v3", "source": "live", "pm25_lo": 17.0, "pm25_hi": 27.5, "in_interval": true }
  ],
  "summary": { "n_total": 169, "n_matched": 166, "n_backfill": 0, "n_used": 166, "basis": "live", "mae": 0.266, "bias": 0.044, "coverage_pct": 87.3, "n_coverage": 166 }
}
```

### `GET /api/comparison/overview?province=`

สรุปภาพรวมทุก horizon/โมเดลของจังหวัดเดียว

```bash
curl "http://localhost:5000/api/comparison/overview?province=khonkaen"
```

```json
{
  "n_total": 4056, "n_matched": 3880,
  "overall_mae": 1.79, "overall_bias": -0.02, "overall_coverage_pct": 90.9,
  "by_horizon": [ { "horizon": 1, "mae": 0.27, "n": 166, "coverage_pct": 87.3, "n_coverage": 166 } ],
  "by_model": [ { "model_used": "xgboost_pooled_v3", "mae": 1.69, "n": 2039, "coverage_pct": 91.9 } ],
  "best_horizon": { "horizon": 1, "mae": 0.27 }, "worst_horizon": { "horizon": 24, "mae": 3.15 }
}
```

---

## 4. บัญชีผู้ใช้ (Auth)

### `POST /api/auth/register`

```bash
curl -c cookies.txt -X POST http://localhost:5000/api/auth/register \
  -H "Content-Type: application/json" \
  --data-binary '{"email":"test@example.com","password":"TestPass1234","risk_group":"ทั่วไป"}'
```

> ⚠️ ใช้ `--data-binary` ไม่ใช่ `-d` เฉยๆ — บน Windows/git-bash บางครั้ง `-d` กับ JSON ที่มี `"` ซ้อน
> กันทำให้ quoting พังจนฝั่ง server เห็น body ว่างแล้วตอบ "รูปแบบอีเมลไม่ถูกต้อง" ทั้งที่ email ที่ส่งถูกต้อง

`risk_group` ต้องเป็นหนึ่งใน `"ทั่วไป"`, `"เด็ก-ผู้สูงอายุ"`, `"มีโรคประจำตัว"` — `password` ≥ 8 ตัวอักษร

```json
{ "id": 10, "email": "test@example.com", "risk_group": "ทั่วไป", "age": null, "created_at": "...", "role": "user", "home_province": null }
```

Login สำเร็จอัตโนมัติหลังสมัคร (session cookie ติดมาใน `cookies.txt` แล้ว)

**Errors**: อีเมลรูปแบบผิด / รหัสผ่านสั้นไป / `risk_group` ไม่ถูกต้อง / `home_province` ไม่อยู่ใน
registry → `400` · อีเมลซ้ำ → `409`

### `POST /api/auth/login`

```bash
curl -c cookies.txt -X POST http://localhost:5000/api/auth/login \
  -H "Content-Type: application/json" \
  --data-binary '{"email":"test@example.com","password":"TestPass1234"}'
```

ผิด email/password → `401`

### `GET /api/auth/me` — ต้อง login

```bash
curl -b cookies.txt http://localhost:5000/api/auth/me
```

ไม่มี session (ไม่ login หรือ cookie หมดอายุ) → `401 {"error": "กรุณาเข้าสู่ระบบ"}`

### `PUT /api/auth/profile` — ต้อง login

```bash
curl -b cookies.txt -X PUT http://localhost:5000/api/auth/profile \
  -H "Content-Type: application/json" \
  --data-binary '{"risk_group":"เด็ก-ผู้สูงอายุ","home_province":"khonkaen"}'
```

ส่งเฉพาะ field ที่อยากแก้ได้ (field ที่ไม่ส่งจะคงค่าเดิม)

### `POST /api/auth/change-password` — ต้อง login

```bash
curl -b cookies.txt -X POST http://localhost:5000/api/auth/change-password \
  -H "Content-Type: application/json" \
  --data-binary '{"current_password":"TestPass1234","new_password":"NewPass5678"}'
```

### `POST /api/auth/logout`

```bash
curl -b cookies.txt -X POST http://localhost:5000/api/auth/logout
# {"ok": true}
```

---

## 5. การแจ้งเตือน (ไม่ต้อง login — risk_group ตาม session ถ้ามี)

### `GET /api/notifications?province=&limit=`

ประวัติรอบที่เคยมีความเสี่ยง (ข้ามรอบที่ไม่เสี่ยงออกจาก response) — `limit` 1–100, default 20

```bash
curl "http://localhost:5000/api/notifications?province=khonkaen&limit=10"
```

### `GET /api/notifications/status?province=`

สถานะล่าสุดรอบเดียว (ใช้กับกระดิ่งแจ้งเตือนบน header)

```bash
curl "http://localhost:5000/api/notifications/status?province=khonkaen"
```

---

## หมายเหตุด้านสิทธิ์การเข้าถึง

`frontend/src/router/index.js` ตั้ง `meta.requiresAdmin` ไว้ที่ route `/compare` (หน้าเปรียบเทียบ)
เพื่อซ่อนลิงก์จากผู้ใช้ทั่วไปในหน้าเว็บ — **แต่ endpoint `/api/comparison` และ
`/api/comparison/overview` ฝั่ง backend ไม่มีการเช็คสิทธิ์ใดๆ เลย** (ไม่มี `@login_required` หรือเช็ค
`role` ใน `app.py`) ใครก็เรียก curl ตรงๆ ได้โดยไม่ต้อง login เป็น admin — เป็นการซ่อนที่ frontend
เท่านั้น ไม่ใช่การป้องกันจริงในระดับ API หากต้องการบังคับสิทธิ์จริงต้องเพิ่ม decorator ตรวจ `role` ที่
ฝั่ง backend ด้วย (ปัจจุบันยังไม่มี เพราะข้อมูลเปรียบเทียบไม่ใช่ข้อมูลอ่อนไหว จึงยังไม่ได้ทำเพิ่ม)

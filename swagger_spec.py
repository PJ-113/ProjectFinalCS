"""
สเปก Swagger/OpenAPI 2.0 สำหรับ flasgger — แยกไฟล์ออกจาก app.py เพื่อไม่ให้ YAML docstring
ยาวๆ ปนกับ logic ของแต่ละ route (app.py ยาวอยู่แล้ว 2500+ บรรทัด) ดู docs/API_TESTING.md
สำหรับตัวอย่าง curl/response แบบเต็ม — ไฟล์นี้คือ source เดียวกันแต่แปลงเป็นรูปแบบที่ Swagger UI
อ่านได้ ถ้าแก้ route ใน app.py (เพิ่ม/ลบ param, เปลี่ยน response shape) ต้องแก้ไฟล์นี้ให้ตรงกันด้วย
เอกสารนี้ไม่ validate กับโค้ดจริงอัตโนมัติ เป็นสเปกที่เขียนแยกต่างหาก
"""

SWAGGER_CONFIG = {
    "headers": [],
    "specs": [{"endpoint": "apispec", "route": "/apispec.json", "rule_filter": lambda rule: True, "model_filter": lambda tag: True}],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/apidocs/",
}

SWAGGER_TEMPLATE = {
    "swagger": "2.0",
    "info": {
        "title": "PM2.5 Forecast API",
        "description": (
            "API ของระบบพยากรณ์ค่า PM2.5 และให้คำแนะนำด้านสุขภาพอากาศตามกลุ่มผู้ใช้ — 20 จังหวัดภาคอีสาน "
            "(โครงงาน capstone). เอกสารนี้สร้างจากสเปกที่เขียนแยกต่างหาก (swagger_spec.py) ไม่ใช่ auto-generate "
            "จากโค้ด — ดู docs/API_TESTING.md สำหรับตัวอย่าง curl เต็มรูปแบบ.\n\n"
            "**คำเตือน**: endpoint ที่เขียนข้อมูลจริง (สมัครสมาชิก, เปลี่ยนรหัสผ่าน ฯลฯ) ถ้ากด Try it out "
            "จะเขียนลงฐานข้อมูลจริงของเซิร์ฟเวอร์ที่กำลังรันอยู่ ไม่ใช่ sandbox."
        ),
        "version": "1.0.0",
    },
    "basePath": "/",
    "tags": [
        {"name": "จังหวัด", "description": "รายชื่อจังหวัดและภาพรวมทุกจังหวัด"},
        {"name": "พยากรณ์", "description": "ค่าพยากรณ์และค่าปัจจุบัน"},
        {"name": "เปรียบเทียบ", "description": "เทียบค่าพยากรณ์กับค่าจริงย้อนหลัง (ไม่มีการเช็คสิทธิ์ฝั่ง backend — ดูหมายเหตุใน description)"},
        {"name": "บัญชีผู้ใช้", "description": "สมัคร/เข้าสู่ระบบ/จัดการโปรไฟล์ — ใช้ session cookie"},
        {"name": "แจ้งเตือน", "description": "ประวัติและสถานะการแจ้งเตือน"},
    ],
    "paths": {
        "/api/provinces": {
            "get": {
                "tags": ["จังหวัด"],
                "summary": "รายชื่อ 20 จังหวัดในระบบ",
                "responses": {
                    "200": {
                        "description": "รายชื่อจังหวัด",
                        "examples": {"application/json": [
                            {"province_th": "ขอนแก่น", "province_en": "khonkaen", "lat": 16.4419, "lon": 102.836, "is_zero_shot": False}
                        ]},
                    }
                },
            }
        },
        "/api/provinces/overview": {
            "get": {
                "tags": ["จังหวัด"],
                "summary": "สรุปค่าปัจจุบัน/ค่าพยากรณ์ของทุกจังหวัดพร้อมกัน (ใช้ทำแผนที่)",
                "parameters": [
                    {"name": "horizon", "in": "query", "type": "integer", "default": 0,
                     "description": "0-24 — 0 คือไม่เอาค่าพยากรณ์ แสดงแค่ค่าปัจจุบัน"}
                ],
                "responses": {
                    "200": {"description": "รายการสรุปรายจังหวัด"},
                    "400": {"description": "horizon นอกช่วง 0-24 หรือไม่ใช่จำนวนเต็ม"},
                },
            }
        },
        "/api/forecast": {
            "get": {
                "tags": ["พยากรณ์"],
                "summary": "พยากรณ์เต็มรูปแบบของจังหวัดเดียว (หนัก — เรียก get_forecast() ทั้งก้อน)",
                "parameters": [
                    {"name": "province", "in": "query", "type": "string", "default": "khonkaen",
                     "description": "province_en slug เช่น khonkaen, mahasarakham"}
                ],
                "responses": {
                    "200": {"description": "ผลพยากรณ์ 24 ชม. พร้อม data_quality/daily_summary/risk_score_config"},
                    "400": {"description": "province ไม่ตรงกับทะเบียนจังหวัด"},
                    "503": {"description": "ไม่มีข้อมูล PM2.5 เลยใน 72 ชม. (สถานีขัดข้อง) — data_unavailable: true"},
                },
            }
        },
        "/api/current_pm25": {
            "get": {
                "tags": ["พยากรณ์"],
                "summary": "ค่า PM2.5 ล่าสุดอย่างเดียว (เบากว่า /api/forecast มาก ไม่พยากรณ์)",
                "parameters": [
                    {"name": "province", "in": "query", "type": "string", "default": "khonkaen"}
                ],
                "responses": {
                    "200": {"description": "ค่าปัจจุบัน", "examples": {"application/json": {"pm25": 23.7, "datetime_th": "2026-10-18 10:00", "status": "ดี"}}},
                    "503": {"description": "ไม่มีข้อมูล PM2.5 ล่าสุด"},
                },
            }
        },
        "/api/comparison": {
            "get": {
                "tags": ["เปรียบเทียบ"],
                "summary": "ข้อมูลเทียบ predicted/actual รายจุดของ horizon เดียว",
                "description": "⚠️ ไม่มีการเช็คสิทธิ์ฝั่ง backend แม้หน้าเว็บจะซ่อนลิงก์นี้ไว้สำหรับ admin เท่านั้นก็ตาม",
                "parameters": [
                    {"name": "horizon", "in": "query", "type": "integer", "default": 1, "description": "1-24"},
                    {"name": "model", "in": "query", "type": "string", "default": "all",
                     "description": "all, lightgbm_pooled_v3, xgboost_pooled_v3, naive_fallback, หรือชื่อรุ่นเก่า"},
                    {"name": "province", "in": "query", "type": "string", "default": "khonkaen"},
                ],
                "responses": {
                    "200": {"description": "points รายจุด + summary (mae, bias, coverage_pct)"},
                    "400": {"description": "horizon/model/province ไม่ถูกต้อง"},
                },
            }
        },
        "/api/comparison/overview": {
            "get": {
                "tags": ["เปรียบเทียบ"],
                "summary": "สรุปภาพรวม MAE/bias/coverage ทุก horizon และทุกโมเดลของจังหวัดเดียว",
                "description": "⚠️ ไม่มีการเช็คสิทธิ์ฝั่ง backend เช่นเดียวกับ /api/comparison",
                "parameters": [
                    {"name": "province", "in": "query", "type": "string", "default": "khonkaen"}
                ],
                "responses": {"200": {"description": "by_horizon / by_model / best_horizon / worst_horizon"}},
            }
        },
        "/api/auth/register": {
            "post": {
                "tags": ["บัญชีผู้ใช้"],
                "summary": "สมัครสมาชิก (login อัตโนมัติหลังสมัครสำเร็จ)",
                "consumes": ["application/json"],
                "parameters": [{
                    "name": "body", "in": "body", "required": True,
                    "schema": {
                        "type": "object",
                        "required": ["email", "password"],
                        "properties": {
                            "email": {"type": "string"},
                            "password": {"type": "string", "description": "≥ 8 ตัวอักษร"},
                            "risk_group": {"type": "string", "enum": ["ทั่วไป", "เด็ก-ผู้สูงอายุ", "มีโรคประจำตัว"], "default": "ทั่วไป"},
                            "age": {"type": "integer"},
                            "home_province": {"type": "string"},
                        },
                    },
                }],
                "responses": {
                    "201": {"description": "สมัครสำเร็จ คืนโปรไฟล์ผู้ใช้"},
                    "400": {"description": "อีเมล/รหัสผ่าน/risk_group/home_province ไม่ถูกต้อง"},
                    "409": {"description": "อีเมลนี้ถูกใช้งานแล้ว"},
                },
            }
        },
        "/api/auth/login": {
            "post": {
                "tags": ["บัญชีผู้ใช้"],
                "summary": "เข้าสู่ระบบ",
                "consumes": ["application/json"],
                "parameters": [{
                    "name": "body", "in": "body", "required": True,
                    "schema": {"type": "object", "required": ["email", "password"],
                               "properties": {"email": {"type": "string"}, "password": {"type": "string"}}},
                }],
                "responses": {"200": {"description": "login สำเร็จ"}, "401": {"description": "อีเมลหรือรหัสผ่านไม่ถูกต้อง"}},
            }
        },
        "/api/auth/logout": {
            "post": {"tags": ["บัญชีผู้ใช้"], "summary": "ออกจากระบบ", "responses": {"200": {"description": "ok: true"}}}
        },
        "/api/auth/me": {
            "get": {
                "tags": ["บัญชีผู้ใช้"],
                "summary": "ดูโปรไฟล์ตัวเอง (ต้อง login)",
                "security": [{"session_cookie": []}],
                "responses": {"200": {"description": "โปรไฟล์ผู้ใช้"}, "401": {"description": "กรุณาเข้าสู่ระบบ"}},
            }
        },
        "/api/auth/profile": {
            "put": {
                "tags": ["บัญชีผู้ใช้"],
                "summary": "แก้โปรไฟล์ (ต้อง login) — ส่งเฉพาะ field ที่อยากแก้",
                "security": [{"session_cookie": []}],
                "consumes": ["application/json"],
                "parameters": [{
                    "name": "body", "in": "body",
                    "schema": {"type": "object", "properties": {
                        "email": {"type": "string"}, "risk_group": {"type": "string"},
                        "age": {"type": "integer"}, "home_province": {"type": "string"},
                    }},
                }],
                "responses": {"200": {"description": "โปรไฟล์ที่อัปเดตแล้ว"}, "400": {"description": "ข้อมูลไม่ถูกต้อง"}, "401": {"description": "กรุณาเข้าสู่ระบบ"}},
            }
        },
        "/api/auth/change-password": {
            "post": {
                "tags": ["บัญชีผู้ใช้"],
                "summary": "เปลี่ยนรหัสผ่าน (ต้อง login)",
                "security": [{"session_cookie": []}],
                "consumes": ["application/json"],
                "parameters": [{
                    "name": "body", "in": "body", "required": True,
                    "schema": {"type": "object", "required": ["current_password", "new_password"],
                               "properties": {"current_password": {"type": "string"}, "new_password": {"type": "string"}}},
                }],
                "responses": {"200": {"description": "ok: true"}, "400": {"description": "รหัสผ่านปัจจุบันผิด หรือรหัสใหม่สั้นไป"}},
            }
        },
        "/api/notifications": {
            "get": {
                "tags": ["แจ้งเตือน"],
                "summary": "ประวัติรอบที่เคยมีความเสี่ยง (ข้ามรอบที่ไม่เสี่ยงออก)",
                "parameters": [
                    {"name": "province", "in": "query", "type": "string", "default": "khonkaen"},
                    {"name": "limit", "in": "query", "type": "integer", "default": 20, "description": "1-100"},
                ],
                "responses": {"200": {"description": "รายการประวัติ"}},
            }
        },
        "/api/notifications/status": {
            "get": {
                "tags": ["แจ้งเตือน"],
                "summary": "สถานะแจ้งเตือนล่าสุดรอบเดียว (ใช้กับกระดิ่งบน header)",
                "parameters": [{"name": "province", "in": "query", "type": "string", "default": "khonkaen"}],
                "responses": {"200": {"description": "สถานะล่าสุด"}},
            }
        },
    },
}

import os
import pickle
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import timedelta
from functools import wraps

import numpy as np
import pandas as pd
import psycopg2
import requests
import urllib3
from apscheduler.schedulers.background import BackgroundScheduler
from flask import Flask, jsonify, redirect, request, send_from_directory, session
from flask_cors import CORS
from werkzeug.security import check_password_hash, generate_password_hash
from google import genai
from google.genai import errors as genai_errors
from google.genai import types as genai_types
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
from shap_analysis import get_shap_explanation  # noqa: E402 (ต้องเข้า sys.path ก่อน import ได้)

# Air4Thai ใช้ certificate ที่ verify ไม่ผ่านผ่าน requests (ทดสอบแล้วใน Testforcest.ipynb)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# ===== Config =====

MODEL_PATH = "models/lgb_pm25_multistep_weighted_20260712_191056.pkl"
XGB_MODEL_PATH = "models/xgb_pm25_multistep_20260712_191350.pkl"
PROPHET_MODEL_PATH = "models/prophet_pm25_20260712_191350.pkl"
BIAS_CSV_PATH = "models/eval_weighted_vs_baseline_20260712_191056.csv"
XGB_BIAS_CSV_PATH = "models/xgb_bias_high_by_horizon_20260712_191350.csv"
CALIBRATION_PATH = "models/air4thai_openaq_calibration.pkl"
HIGH_POLLUTION_THRESHOLD = 37.5  # ใช้ตรงๆ กับค่าจริง (latest_pm25) เพราะไม่มี bias จากโมเดล
STATION_ID_AIR4THAI = "46t"
STATION_LAT = 16.44538333
STATION_LON = 102.83525
# ต้องรัน `npm run build` ใน frontend/ ก่อน ถึงจะมีโฟลเดอร์นี้ (ดู CHANGELOG)
FRONTEND_DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "dist")
DEBUG = True  # ต้องอ่านจากตรงนี้ ไม่ใช่ app.debug เพราะ Flask ยังไม่ set app.debug จนกว่าจะเรียก app.run()
SIGMA_CRITICAL = 0.3  # Dynamic Routing (proposal 4.1.6): missing_ratio เกินนี้ถือว่า input เสียหายหนัก ให้ fallback ไป Prophet
EXPLORATION_RATE = 0.2  # โอกาสที่จะ "ลอง" โมเดลที่ยังไม่มีข้อมูลพอประเมิน R² แก้ปัญหา cold-start deadlock (ดู select_model)


# ===== .env / Database config =====

def _load_env():
    """อ่าน .env เป็น dict เดียว (GEMINI_KEY + POSTGRES_*) ห้าม print/log ค่าที่ได้จากตรงนี้ที่ไหนเลย"""
    env = {}
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env


_ENV = _load_env()
GEMINI_KEY = _ENV["GEMINI_KEY"]
SECRET_KEY = _ENV["SECRET_KEY"]  # เซ็น Flask session cookie (login/register) — ห้าม print/log

# PostgreSQL (ย้ายจาก sqlite แล้ว ดู CHANGELOG) — get_db_connection() เปิด connection ใหม่ทุกครั้ง
# ที่เรียก ตาม pattern เดิมของ sqlite3.connect() ต่อ call
POSTGRES_HOST = _ENV["POSTGRES_HOST"]
POSTGRES_PORT = _ENV["POSTGRES_PORT"]
POSTGRES_DB = _ENV["POSTGRES_DB"]
POSTGRES_USER = _ENV["POSTGRES_USER"]
POSTGRES_PASSWORD = _ENV["POSTGRES_PASSWORD"]


def get_db_connection():
    return psycopg2.connect(
        host=POSTGRES_HOST,
        port=POSTGRES_PORT,
        dbname=POSTGRES_DB,
        user=POSTGRES_USER,
        password=POSTGRES_PASSWORD,
    )

# ===== Flask app + model loading =====

app = Flask(__name__)
CORS(app)

app.config["SECRET_KEY"] = SECRET_KEY
app.config["SESSION_COOKIE_HTTPONLY"] = True
app.config["SESSION_COOKIE_SAMESITE"] = "Lax"
app.config["SESSION_COOKIE_SECURE"] = False  # demo รันบน HTTP local — เปลี่ยนเป็น True ถ้า deploy บน HTTPS
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=7)

with open(MODEL_PATH, "rb") as f:
    model_package = pickle.load(f)

MODELS = model_package["models"]
FEATURE_COLS = model_package["feature_cols"]

# XGBoost/Prophet โหลดครั้งเดียวตอน startup เหมือน LightGBM (Dynamic Routing)
with open(XGB_MODEL_PATH, "rb") as f:
    xgb_package = pickle.load(f)
XGB_MODELS = xgb_package["models"]

with open(PROPHET_MODEL_PATH, "rb") as f:
    prophet_package = pickle.load(f)
PROPHET_MODEL = prophet_package["model"]

# โมเดลเทรนด้วย OpenAQ แต่ live inference ดึงจาก Air4Thai (fetch_recent_pm25) — ค่าจาก 2 แหล่ง
# ต่างสเกลกันมาก (fit ด้วย scripts/fit_air4thai_openaq_calibration.py: R²=0.397, n=1395 ชม.)
# จึงต้องแปลงค่าดิบ Air4Thai ให้อยู่สเกล OpenAQ ก่อนสร้าง feature ทุกครั้ง
with open(CALIBRATION_PATH, "rb") as f:
    _calibration = pickle.load(f)
CALIBRATION_A = _calibration["a"]
CALIBRATION_B = _calibration["b"]

# bias_high ต่อ horizon ต่อโมเดล โหลดจาก CSV (ไม่ hardcode) เพื่อให้แก้ไฟล์ CSV พอเวลาเทรนใหม่
_lgb_bias_df = pd.read_csv(BIAS_CSV_PATH)
_xgb_bias_df = pd.read_csv(XGB_BIAS_CSV_PATH)
BIAS_HIGH_BY_MODEL = {
    "lightgbm": {int(r.horizon): float(r.bias_high_weighted) for r in _lgb_bias_df.itertuples()},
    # XGBoost ยังไม่ผ่าน sample-weight retrain เหมือน LightGBM — นี่คือ bias ตามธรรมชาติของโมเดล unweighted
    "xgboost": {int(r.horizon): float(r.bias_high) for r in _xgb_bias_df.itertuples()},
    # Prophet evaluate แบบ 1-step ไม่ใช่ multi-horizon เหมือนอีก 2 โมเดล เลยไม่มี bias ต่อ horizon ที่เทียบตรงๆ ได้ — ใช้ 37.5 ตรงๆ
    "prophet": {h: 0.0 for h in range(1, 25)},
}

# threshold ลดตาม bias ของแต่ละโมเดล**แยกต่อโมเดล** เพราะ Dynamic Routing เลือกโมเดลต่างกันได้ในแต่ละ horizon
EFFECTIVE_THRESHOLD_BY_HORIZON_BY_MODEL = {
    model_name: {h: HIGH_POLLUTION_THRESHOLD - bias for h, bias in bias_by_horizon.items()}
    for model_name, bias_by_horizon in BIAS_HIGH_BY_MODEL.items()
}

# กลุ่มเปราะบาง (เด็ก/ผู้สูงอายุ, มีโรคประจำตัว) ควรเริ่มระมัดระวังเร็วกว่าคนทั่วไป — ตาม
# แนวทาง WHO/US EPA ที่แนะนำให้กลุ่มเสี่ยงเริ่มระวังตัวก่อนคนทั่วไปประมาณ 1 ระดับ AQI ไม่ใช่รอถึง
# ระดับ "เริ่มมีผลกระทบต่อสุขภาพ" เหมือนคนทั่วไป threshold ปกติ (37.5) อยู่ตรงรอยต่อ "ปานกลาง"
# (25.1-37.5 ตาม PCD)/"เริ่มมีผลกระทบต่อสุขภาพ" พอดี ลด 10 หน่วย (~27.5) ทำให้กลุ่มเปราะบางเริ่ม
# เตือนตั้งแต่อยู่ในช่วง "ปานกลาง" แทนที่จะรอข้ามระดับเหมือนคนทั่วไป — เป็นตัวเลขที่อธิบายได้
# ตามหลักการทั่วไป ไม่มี dataset เฉพาะมารองรับตัวเลขเป๊ะๆ (ตามที่ตกลงกันไว้) ทั้งสองกลุ่มเปราะบาง
# ใช้ค่าเดียวกันเพราะไม่มีหลักฐานแยกความรุนแรงระหว่างสองกลุ่มนี้
RISK_GROUP_THRESHOLD_OFFSET = {"ทั่วไป": 0.0, "เด็ก-ผู้สูงอายุ": -10.0, "มีโรคประจำตัว": -10.0}

# Personalized Risk Score (proposal) — กิจกรรมกลางแจ้งที่หนักขึ้นทำให้อัตราการหายใจ (minute
# ventilation) สูงขึ้น จึงสูดฝุ่นเข้าร่างกายมากขึ้นที่ความเข้มข้นอากาศเท่าเดิม จำลองเป็นการลด
# effective PM2.5 ลงเหมือนกลไกเดียวกับ RISK_GROUP_THRESHOLD_OFFSET ด้านบน (sign convention เดียวกัน
# ยิ่งติดลบมาก = ยิ่งอ่อนไหวต่อฝุ่นมาก) ใช้เพดาน -10 เท่ากับกลุ่มเปราะบางเพื่อความสอดคล้องกัน — เป็น
# ตัวเลข heuristic สำหรับสาธิตหลักการ ไม่ใช่ dose-response ที่มาจากงานวิจัยทางการแพทย์
ACTIVITY_INTENSITY_OFFSET = {"เบา": 0.0, "ปานกลาง": -5.0, "หนัก": -10.0}

# แปลง effective PM2.5 เป็นคะแนน 0-100 ต่อเนื่อง ด้วย linear interpolation ข้าม breakpoint ที่ยึดตาม
# PM25_LEVELS (มาตรฐาน คพ. 2566 ด้านล่าง) ทุกเป๊ะ — แต่ละระดับใน PM25_LEVELS เท่ากับ 20 คะแนนพอดี
# (15/25/37.5/75 -> 20/40/60/80) ส่วน 150 เป็นเพดานที่ตั้งขึ้นเพื่อ scale คะแนนเท่านั้น ไม่ใช่ breakpoint
# ทางการ ค่าที่เกิน 150 ถูก clamp เป็น 100 คะแนน
RISK_SCORE_BREAKPOINTS = [(0, 0), (15, 20), (25, 40), (37.5, 60), (75, 80), (150, 100)]


def effective_threshold_for(model_name, horizon, risk_group):
    return EFFECTIVE_THRESHOLD_BY_HORIZON_BY_MODEL[model_name][horizon] + RISK_GROUP_THRESHOLD_OFFSET.get(risk_group, 0.0)


# แยกเป็นฟังก์ชันเล็กๆ ให้ get_forecast() (คำนวณสด) และ /api/notifications (คำนวณย้อนหลังจาก
# prediction_log) ใช้ตรรกะ "อะไรถือว่าเสี่ยง" อันเดียวกันเป๊ะ กันสองที่ตัดสินไม่ตรงกัน
def is_horizon_at_risk(pred, model_name, horizon, risk_group):
    return pred > effective_threshold_for(model_name, horizon, risk_group)


def is_current_at_risk(latest_pm25, risk_group):
    # bool(...) เพราะ latest_pm25 มักเป็น numpy.float64 (มาจาก pandas) เทียบกับ float ธรรมดาแล้วได้
    # numpy.bool_ กลับมา ไม่ใช่ bool ปกติ — ถ้าไม่แปลง ตอน get_forecast() ทำ "is_current_at_risk(...)
    # or len(risk_horizons) > 0" แล้วฝั่งซ้าย True (numpy.bool_) จะถูกคืนตรงๆ โดย or ไม่แปลงชนิดให้
    # ทำให้ jsonify() พังตอน current PM2.5 สูงกว่า effective threshold จริง (เจอบั๊กนี้ตอนรัน
    # test_risk_group_e2e.py กับข้อมูลสดที่ PM2.5 ขึ้นมาแตะ threshold ของกลุ่มเปราะบางพอดี)
    return bool(latest_pm25 > HIGH_POLLUTION_THRESHOLD + RISK_GROUP_THRESHOLD_OFFSET.get(risk_group, 0.0))


ACTIVE_HOURS_START_TH = 6
ACTIVE_HOURS_END_TH = 20


def find_best_time(forecast):
    """หาจุดที่ PM2.5 ต่ำสุดใน forecast[] โดยกรองเฉพาะ 06:00-20:00 เวลาไทยก่อน (ช่วงที่ใช้งานจริงได้
    เช่น ออกกำลังกาย/ทำกิจกรรมกลางแจ้ง) กัน suggest ช่วงดึกที่ PM2.5 ต่ำแต่ไม่มีประโยชน์ในทางปฏิบัติ —
    fallback ไป forecast ทั้งหมดเฉพาะกรณี active_hours ว่างเปล่าจริงๆ (ไม่ควรเกิดในทางปฏิบัติ เพราะ
    forecast มี 24 จุดครอบคลุมทุกชั่วโมงเสมอ แต่กันไว้เผื่อ edge case)
    """
    active_hours = [f for f in forecast if ACTIVE_HOURS_START_TH <= f["hour_thai"] <= ACTIVE_HOURS_END_TH]
    return min(active_hours or forecast, key=lambda x: x["pm25"])


def compute_risk_score(pm25, risk_group, activity):
    """Personalized Risk Score (proposal) — ผสาน pm25 (ค่าจริงหรือค่าพยากรณ์ก็ได้ เป็น pure function
    ไม่ผูกกับ live data) + risk_group + activity เป็นคะแนนเดียว 0-100 ผ่าน effective PM2.5 (ดู
    ACTIVITY_INTENSITY_OFFSET/RISK_GROUP_THRESHOLD_OFFSET ด้านบน) แล้ว interpolate เชิงเส้นข้าม
    RISK_SCORE_BREAKPOINTS คืนค่า level เป็นชื่อระดับเดียวกับ PM25_LEVELS (ผ่าน _classify_pm25_level
    ที่นิยามด้านล่าง ณ ตอนเรียกฟังก์ชันนี้จริง โมดูลโหลดครบแล้วเสมอ ไม่มีปัญหา late binding)
    """
    effective_pm25 = pm25 - RISK_GROUP_THRESHOLD_OFFSET.get(risk_group, 0.0) - ACTIVITY_INTENSITY_OFFSET.get(activity, 0.0)
    cap = RISK_SCORE_BREAKPOINTS[-1][0]
    clamped = max(0.0, min(effective_pm25, cap))
    score = RISK_SCORE_BREAKPOINTS[-1][1]
    for (x0, y0), (x1, y1) in zip(RISK_SCORE_BREAKPOINTS, RISK_SCORE_BREAKPOINTS[1:]):
        if clamped <= x1:
            score = y0 + (y1 - y0) * (clamped - x0) / (x1 - x0)
            break
    return {
        "score": round(score),
        "level": _classify_pm25_level(effective_pm25),
        "effective_pm25": round(effective_pm25, 1),
    }


# ===== Dynamic Routing (proposal 4.1.6) =====

def compute_missing_ratio(feature_row):
    """สัดส่วนค่า NaN ใน feature_row (DataFrame แถวเดียวจาก build_feature_row()) เทียบกับ FEATURE_COLS ทั้งหมด — คืนค่า 0-1"""
    n_missing = int(feature_row[FEATURE_COLS].isna().sum().sum())
    return n_missing / len(FEATURE_COLS)


def compute_rolling_r2(model_name, horizon, N=24):
    """R² แบบ rolling จาก N แถวล่าสุดของ model_name/horizon ที่มี actual_pm25_calibrated แล้ว —
    ใช้ actual_pm25_calibrated ไม่ใช่ actual_pm25 ดิบ เพราะ predicted_pm25 มาจาก input ที่ calibrate
    แล้ว ต้องเทียบสเกลเดียวกัน (ดู build_feature_row()) เป็นตัวชี้วัด performance ล่าสุดสำหรับ
    Dynamic Routing คืนค่า None ถ้าข้อมูลไม่พอ (< 5 คู่ — น้อยไปจนไม่น่าเชื่อถือ)
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """SELECT predicted_pm25, actual_pm25_calibrated
           FROM prediction_log
           WHERE model_used = %s AND horizon = %s AND actual_pm25_calibrated IS NOT NULL
           ORDER BY target_time_utc DESC
           LIMIT %s""",
        (model_name, horizon, N),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if len(rows) < 5:
        return None

    y_pred = [r[0] for r in rows]
    y_true = [r[1] for r in rows]
    return r2_score(y_true, y_pred)


def select_model(feature_row, horizon):
    """เลือกโมเดลพยากรณ์สำหรับ horizon ที่ระบุ (Dynamic Routing, proposal 4.1.6):

    1. feature_row เสียหายหนัก (missing_ratio > SIGMA_CRITICAL) → Prophet (ไม่พึ่ง lag/rolling)
    2. ไม่งั้นเทียบ rolling R² ล่าสุดของ XGBoost/LightGBM ที่ horizon นั้น — ต้องเช็ค None ทั้ง 2
       ทิศทาง ไม่ใช่แค่ r2_xgb เพราะโมเดลไหนก็ขาดข้อมูลได้ (เช็คแค่ฝั่งเดียวเคยทำให้ exploration
       branch ไม่ทำงานเมื่อ LightGBM เป็นฝั่งที่ขาดข้อมูล กลายเป็น fallback LightGBM 100% เสมอ
       ผิดเจตนา ε-greedy ของ proposal — แก้ไว้แล้ว ดู CHANGELOG (20)):
       - ทั้งคู่ไม่มีข้อมูล → สุ่ม 50/50
       - มีแค่ตัวเดียว → explore ตัวที่ขาดข้อมูลด้วยความน่าจะเป็น EXPLORATION_RATE ไม่งั้น fallback ไปตัวที่มีข้อมูลแล้ว
       - มี R² ครบทั้งคู่ → เลือกตัวที่ R² สูงกว่า
    """
    if compute_missing_ratio(feature_row) > SIGMA_CRITICAL:
        return "prophet"

    r2_xgb = compute_rolling_r2("xgboost", horizon)
    r2_lgb = compute_rolling_r2("lightgbm", horizon)

    if r2_xgb is None and r2_lgb is None:
        return random.choice(["xgboost", "lightgbm"])
    elif r2_xgb is None:
        return "xgboost" if random.random() < EXPLORATION_RATE else "lightgbm"
    elif r2_lgb is None:
        return "lightgbm" if random.random() < EXPLORATION_RATE else "xgboost"
    else:
        return "xgboost" if r2_xgb > r2_lgb else "lightgbm"


# ===== Database schema =====

def init_db():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS prediction_log (
            id SERIAL PRIMARY KEY,
            predicted_at_utc TEXT NOT NULL,
            target_time_utc TEXT NOT NULL,
            horizon INTEGER NOT NULL,
            predicted_pm25 DOUBLE PRECISION NOT NULL,
            actual_pm25 DOUBLE PRECISION,
            model_used TEXT DEFAULT 'lightgbm',
            source TEXT DEFAULT 'live',
            actual_pm25_calibrated DOUBLE PRECISION,
            UNIQUE(predicted_at_utc, horizon)
        )
    """)

    # เก็บ NLG รายวัน (สร้างแค่ครั้งเดียวต่อวันตอน 06:00 น. เวลาไทย — ดู maybe_generate_daily_nlg())
    # date_thai เป็น "YYYY-MM-DD" ตามเวลาไทยเสมอ ไม่ใช่วันที่ UTC (ดู get_thai_now())
    cur.execute("""
        CREATE TABLE IF NOT EXISTS daily_nlg_log (
            id SERIAL PRIMARY KEY,
            date_thai TEXT NOT NULL UNIQUE,
            daily_avg_pm25 DOUBLE PRECISION,
            explanation_text TEXT,
            used_fallback BOOLEAN,
            generated_at_utc TEXT
        )
    """)

    # ประวัติค่าจริงล่าสุดต่อรอบ (สำหรับ /api/notifications) — เก็บแค่ latest_pm25 เท่านั้น เพราะเป็น
    # ค่าเดียวที่ไม่มีที่เก็บอยู่แล้ว ส่วน risk_horizons ต่อ horizon คำนวณสดตอน query โดย join กับ
    # prediction_log ผ่าน recorded_at_utc = predicted_at_utc (ค่าเดียวกัน มาจาก latest_time ตัวเดียวกัน
    # ตอน get_forecast() เรียก log_predictions()/log_notification_snapshot() คู่กัน) เหตุผลที่ไม่เก็บ
    # risk_horizons ไว้ตรงๆ: risk ขึ้นกับ risk_group (ดู RISK_GROUP_THRESHOLD_OFFSET) เก็บสดจะผิดสำหรับ
    # กลุ่มอื่น เก็บ 3 ชุดจะซ้ำซ้อน คำนวณสดครั้งเดียวถูกทุกกลุ่มเสมอ แม้ threshold จะเปลี่ยนในอนาคต
    cur.execute("""
        CREATE TABLE IF NOT EXISTS notification_log (
            id SERIAL PRIMARY KEY,
            recorded_at_utc TEXT NOT NULL UNIQUE,
            latest_pm25 DOUBLE PRECISION NOT NULL
        )
    """)

    # risk_group เก็บ Thai string ตรงๆ (คีย์เดียวกับ _USER_GROUP_TONE) ผูกกับโทน NLG รายวันและ
    # alert threshold แล้ว (ดู RISK_GROUP_THRESHOLD_OFFSET, effective_threshold_for(),
    # _build_daily_nlg_prompt()) — NLG รายชั่วโมงถูกลบทิ้งแล้ว (ดู memory project_hourly_nlg_deferred)
    # เพราะซ้ำซ้อนกับ NLG รายวัน และ proposal 4.1.4 ไม่ได้ระบุว่าต้องเป็นรายชั่วโมง
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id SERIAL PRIMARY KEY,
            email TEXT NOT NULL UNIQUE,
            password_hash TEXT NOT NULL,
            risk_group TEXT NOT NULL DEFAULT 'ทั่วไป',
            age INTEGER,
            created_at TEXT NOT NULL
        )
    """)
    conn.commit()
    cur.close()
    conn.close()


init_db()


# ===== User accounts (auth) =====

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
MIN_PASSWORD_LENGTH = 8


def _public_user(row):
    """แปลง DB row -> dict โปรไฟล์สาธารณะ ไม่มี password_hash เด็ดขาด"""
    return {
        "id": row[0],
        "email": row[1],
        "risk_group": row[2],
        "age": row[3],
        "created_at": row[4],
    }


def get_user_by_email(email):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, email, risk_group, age, created_at FROM users WHERE email = %s",
        (email,),
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    return _public_user(row) if row else None


def get_user_by_email_with_hash(email):
    """เหมือน get_user_by_email() แต่ดึง password_hash มาด้วย — ใช้เฉพาะ login/change-password"""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, email, risk_group, age, created_at, password_hash FROM users WHERE email = %s",
        (email,),
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    if row is None:
        return None
    return {**_public_user(row[:5]), "password_hash": row[5]}


def get_user_by_id(user_id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, email, risk_group, age, created_at FROM users WHERE id = %s",
        (user_id,),
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    return _public_user(row) if row else None


def create_user(email, password_hash, risk_group, age):
    conn = get_db_connection()
    cur = conn.cursor()
    created_at = pd.Timestamp.now(tz="UTC").isoformat()
    cur.execute(
        """INSERT INTO users (email, password_hash, risk_group, age, created_at)
           VALUES (%s, %s, %s, %s, %s) RETURNING id""",
        (email, password_hash, risk_group, age, created_at),
    )
    user_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return user_id


def update_user_profile(user_id, email, risk_group, age):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "UPDATE users SET email = %s, risk_group = %s, age = %s WHERE id = %s",
        (email, risk_group, age, user_id),
    )
    conn.commit()
    cur.close()
    conn.close()


def update_user_password(user_id, new_password_hash):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("UPDATE users SET password_hash = %s WHERE id = %s", (new_password_hash, user_id))
    conn.commit()
    cur.close()
    conn.close()


def login_required(view_func):
    @wraps(view_func)
    def wrapper(*args, **kwargs):
        if "user_id" not in session:
            return jsonify({"error": "กรุณาเข้าสู่ระบบ"}), 401
        return view_func(*args, **kwargs)
    return wrapper


# ===== Live data fetching + calibration =====

def fetch_recent_pm25():
    today = pd.Timestamp.now()
    sdate = (today - timedelta(days=3)).strftime("%Y-%m-%d")
    edate = today.strftime("%Y-%m-%d")

    url = "https://air4thai.com/forweb/getHistoryData.php"
    params = {
        "stationID": STATION_ID_AIR4THAI,
        "param": "PM25",
        "type": "hr",
        "sdate": sdate,
        "edate": edate,
        "stime": "00",
        "etime": "23",
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }

    r = requests.get(url, params=params, headers=headers, timeout=15, verify=False)
    r.raise_for_status()
    records = r.json()["stations"][0]["data"]

    df = pd.DataFrame(records).rename(columns={"DATETIMEDATA": "datetime_utc", "PM25": "pm25"})
    df["datetime_utc"] = (
        pd.to_datetime(df["datetime_utc"])
        .dt.tz_localize("Asia/Bangkok")
        .dt.tz_convert("UTC")
    )
    df = df.dropna(subset=["pm25"]).sort_values("datetime_utc").reset_index(drop=True)
    return df


def fetch_current_weather():
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": STATION_LAT,
        "longitude": STATION_LON,
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation",
        "timezone": "UTC",
    }
    r = requests.get(url, params=params, timeout=15)
    r.raise_for_status()
    return r.json()["current"]


def calibrate_pm25(value):
    """แปลงค่าดิบจาก Air4Thai ให้อยู่สเกล OpenAQ ที่โมเดลเทรนด้วย (ดู CALIBRATION_A/B)"""
    return CALIBRATION_A * value + CALIBRATION_B


def migrate_calibrated_actuals():
    """Backfill actual_pm25_calibrated ย้อนหลังให้แถวที่มี actual_pm25 อยู่แล้วก่อนคอลัมน์นี้จะถูก
    เพิ่ม (idempotent, ไม่แตะ actual_pm25 เอง) สเกลต่างกันตาม source:
    - 'live': ค่าดิบจาก Air4Thai → calibrated = a*x+b
    - 'backfill': actual มาจาก CSV สเกล OpenAQ อยู่แล้ว → calibrated = ค่าเดิมตรงๆ ห้ามผ่าน a*x+b
      ซ้ำ (migration รอบแรกเคยพลาดตรงนี้เป็น double-transform บั๊ก — ฟังก์ชันนี้ self-heal ให้ด้วย)
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, actual_pm25 FROM prediction_log "
        "WHERE actual_pm25 IS NOT NULL AND actual_pm25_calibrated IS NULL AND source != 'backfill'"
    )
    rows = cur.fetchall()
    if rows:
        updates = [(calibrate_pm25(actual), row_id) for row_id, actual in rows]
        cur.executemany("UPDATE prediction_log SET actual_pm25_calibrated = %s WHERE id = %s", updates)
        print(f"[migration] backfilled actual_pm25_calibrated for {len(rows)} live rows (a*x+b)")

    # แถว backfill: calibrated ต้องเท่ากับ actual เป๊ะ — เงื่อนไข != ครอบคลุมทั้งแถวที่ยัง NULL
    # และแถวที่เคยถูก migration รอบแรก apply a*x+b ผิดไป (self-healing, idempotent)
    cur.execute(
        "UPDATE prediction_log SET actual_pm25_calibrated = actual_pm25 "
        "WHERE source = 'backfill' AND actual_pm25 IS NOT NULL "
        "AND (actual_pm25_calibrated IS NULL OR actual_pm25_calibrated != actual_pm25)"
    )
    n_fixed = cur.rowcount
    if n_fixed:
        print(f"[migration] set actual_pm25_calibrated = actual_pm25 for {n_fixed} backfill rows (already OpenAQ scale)")
    conn.commit()
    cur.close()
    conn.close()


migrate_calibrated_actuals()


def migrate_daily_nlg_risk_group():
    """daily_nlg_log มีอยู่แล้วตั้งแต่ Postgres migration (CHANGELOG 30) — CREATE TABLE IF NOT EXISTS
    ไม่ช่วยเพิ่ม column ให้ตารางที่มีอยู่แล้ว ต้อง ALTER จริง เปลี่ยน unique จาก date_thai อย่างเดียว
    เป็น (date_thai, risk_group) — แถวเก่าทุกแถวถือเป็นสรุปของกลุ่ม 'ทั่วไป' โดยปริยาย (เพราะตอนนั้น
    ทุกคนเห็นข้อความเดียวกัน) ปลอดภัย ไม่มี conflict เพราะ date_thai เดิม unique อยู่แล้ว idempotent"""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("ALTER TABLE daily_nlg_log ADD COLUMN IF NOT EXISTS risk_group TEXT NOT NULL DEFAULT 'ทั่วไป'")
    cur.execute("""
        SELECT conname FROM pg_constraint
        WHERE conrelid = 'daily_nlg_log'::regclass AND contype = 'u'
    """)
    existing = {r[0] for r in cur.fetchall()}
    if "daily_nlg_log_date_thai_risk_group_key" not in existing:
        for name in existing:  # ลบ unique constraint เก่า (date_thai เดี่ยว) ถ้ายังอยู่ — ชื่อมาจาก
            cur.execute(f"ALTER TABLE daily_nlg_log DROP CONSTRAINT {name}")  # pg_constraint catalog เอง ไม่ใช่ input ภายนอก
        cur.execute(
            "ALTER TABLE daily_nlg_log ADD CONSTRAINT daily_nlg_log_date_thai_risk_group_key "
            "UNIQUE (date_thai, risk_group)"
        )
    conn.commit()
    cur.close()
    conn.close()


migrate_daily_nlg_risk_group()


# ===== Feature engineering & prediction logging =====

def build_feature_row(df_recent, current_weather):
    latest_time = df_recent["datetime_utc"].max()
    pm25_series_raw = df_recent.set_index("datetime_utc")["pm25"]
    # ค่า lag/rolling ทั้งหมดต้องผ่าน calibration ก่อน เพราะโมเดลเทรนด้วยสเกลของ OpenAQ
    # แต่ fetch_recent_pm25() ดึงค่าดิบจาก Air4Thai ที่ต่างสเกลกัน (ดูคอมเมนต์ที่ CALIBRATION_PATH)
    pm25_series = pm25_series_raw.apply(calibrate_pm25)

    features = {
        "temp": current_weather["temperature_2m"],
        "humidity": current_weather["relative_humidity_2m"],
        "wind_speed": current_weather["wind_speed_10m"],
        "precipitation": current_weather["precipitation"],
        "hour_sin": np.sin(2 * np.pi * latest_time.hour / 24),
        "hour_cos": np.cos(2 * np.pi * latest_time.hour / 24),
        "month_sin": np.sin(2 * np.pi * latest_time.month / 12),
        "month_cos": np.cos(2 * np.pi * latest_time.month / 12),
        "dayofweek": latest_time.dayofweek,
        "pm25_lag_1": pm25_series.iloc[-2],
        "pm25_lag_3": pm25_series.iloc[-4],
        "pm25_lag_6": pm25_series.iloc[-7],
        "pm25_lag_12": pm25_series.iloc[-13],
        "pm25_lag_24": pm25_series.iloc[-25],
        # rolling window รวมค่าปัจจุบันด้วย (ตรงกับ pipeline ตอนเทรนที่ใช้ pandas .rolling(N) มาตรฐาน
        # ยืนยันด้วยการเทียบย้อนหลังกับ data/khonkaen_pm25_features_final.csv แล้วตรงเป๊ะ 0 mismatch)
        "pm25_roll_mean_6": pm25_series.iloc[-6:].mean(),
        "pm25_roll_std_6": pm25_series.iloc[-6:].std(),
        "pm25_roll_mean_24": pm25_series.iloc[-24:].mean(),
        "pm25_roll_std_24": pm25_series.iloc[-24:].std(),
    }

    X_now = pd.DataFrame([features])[FEATURE_COLS]
    # latest_pm25 (แสดงเป็น "current" ให้ผู้ใช้ + เทียบ HIGH_POLLUTION_THRESHOLD) เป็นค่าจริงดิบ
    # จาก Air4Thai ไม่ใช่ค่าที่ calibrate แล้ว เพราะเป็นค่าอ่านจริงของสถานี ไม่ใช่ input โมเดล
    return X_now, latest_time, pm25_series_raw.iloc[-1]


def predict_with_prophet(prophet_model, current_weather, latest_time, horizon):
    """พยากรณ์ PM2.5 ที่ t+horizon ด้วย Prophet

    ข้อจำกัด (simplification ชั่วคราว): Prophet ต้องการ weather regressor ณ เวลาเป้าหมายในอนาคต
    แต่ระบบมีแค่ current weather จาก Open-Meteo (ไม่มี hourly forecast) จึงใช้ current_weather
    ซ้ำทุก horizon แทน — ยิ่ง horizon ไกล ยิ่งคลาดเคลื่อนจากสภาพอากาศจริง ควรปรับเป็นดึง
    Open-Meteo hourly forecast จริงในอนาคต
    """
    target_time_utc = latest_time + pd.Timedelta(hours=horizon)
    future = pd.DataFrame([{
        "ds": target_time_utc.tz_localize(None),  # Prophet เทรนด้วย ds ที่เป็น naive UTC (ดู train_xgb_prophet.py)
        "temp": current_weather["temperature_2m"],
        "humidity": current_weather["relative_humidity_2m"],
        "wind_speed": current_weather["wind_speed_10m"],
        "precipitation": current_weather["precipitation"],
    }])
    forecast = prophet_model.predict(future)
    return float(forecast["yhat"].iloc[0])


def log_predictions(predicted_at, rows):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.executemany(
        """INSERT INTO prediction_log
           (predicted_at_utc, target_time_utc, horizon, predicted_pm25, model_used, source)
           VALUES (%s, %s, %s, %s, %s, 'live')
           ON CONFLICT (predicted_at_utc, horizon) DO NOTHING""",
        [(predicted_at.isoformat(), row["target_time_utc"].isoformat(), row["horizon"], row["pm25"], row["model_used"])
         for row in rows],
    )
    conn.commit()
    cur.close()
    conn.close()


def log_notification_snapshot(recorded_at_utc, latest_pm25):
    """เก็บ latest_pm25 ของรอบนี้ไว้ใน notification_log — recorded_at_utc ต้องเป็นค่าเดียวกับ
    predicted_at_utc ที่ log_predictions() ใช้ (ทั้งคู่มาจาก latest_time เดียวกัน) เพื่อให้
    /api/notifications join กับ prediction_log ได้ตรง ON CONFLICT DO NOTHING เหมือน log_predictions()
    ทำให้เรียกจาก get_forecast() ได้ทุกครั้งไม่ว่าจะมาจาก scheduler หรือ user เปิดเว็บ ไม่ต้อง
    special-case แยก เพราะ latest_time เปลี่ยนแค่ตอน Air4Thai มีข้อมูลใหม่เท่านั้น (ราวชั่วโมงละครั้ง)
    """
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO notification_log (recorded_at_utc, latest_pm25)
           VALUES (%s, %s)
           ON CONFLICT (recorded_at_utc) DO NOTHING""",
        (recorded_at_utc, latest_pm25),
    )
    conn.commit()
    cur.close()
    conn.close()


def backfill_actuals(df_recent):
    """เติมค่าจริงให้ prediction ที่ log ไว้ก่อนหน้า จากข้อมูล PM2.5 ย้อนหลังที่เพิ่งดึงมา (ไม่ยิง API เพิ่ม)
    เติมทั้ง actual_pm25 (ค่าดิบ) และ actual_pm25_calibrated (ผ่าน calibrate_pm25 แล้ว — ต้องใช้
    คอลัมน์นี้เทียบ error/R² เพราะ predicted_pm25 อยู่สเกล OpenAQ ดู compute_rolling_r2)"""
    pm25_by_time = df_recent.set_index("datetime_utc")["pm25"].to_dict()

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, target_time_utc FROM prediction_log WHERE actual_pm25 IS NULL")
    pending = cur.fetchall()

    updates = []
    for row_id, target_time_str in pending:
        target_time = pd.Timestamp(target_time_str)
        actual = pm25_by_time.get(target_time)
        if actual is not None:
            updates.append((float(actual), calibrate_pm25(float(actual)), row_id))

    if updates:
        cur.executemany(
            "UPDATE prediction_log SET actual_pm25 = %s, actual_pm25_calibrated = %s WHERE id = %s",
            updates,
        )
        conn.commit()
    cur.close()
    conn.close()


# ===== ดัชนีคุณภาพอากาศ PM2.5 มาตรฐานกรมควบคุมมลพิษ (คพ.) ประกาศ พ.ศ. 2566 =====
# (ราชกิจจานุเบกษา 3 ก.ค. 2566, มีผล 1 มิ.ย. 2566 เป็นต้นไป) 5 ระดับ ค่าเฉลี่ย 24 ชม.:
#   0.0-15.0 ดีมาก, 15.1-25.0 ดี, 25.1-37.5 ปานกลาง,
#   37.6-75.0 เริ่มมีผลกระทบต่อสุขภาพ, 75.1+ มีผลกระทบต่อสุขภาพ
# ยืนยันแล้วจาก 2 แหล่งอิสระ: ประกาศของ คพ. เอง (pcd.go.th/pcd_news/30028) และข่าว Thai PBS
# (thaipbs.or.th/news/content/327288) — cutoff 25/37.5 ตรงกับ HIGH_POLLUTION_THRESHOLD เดิมเป๊ะ
PM25_LEVELS = [
    (15, "ดีมาก", ["ดีมาก", "อากาศดีมาก"]),
    (25, "ดี", [("อากาศ", "ดี"), ("คุณภาพอากาศ", "ดี"), "ปลอดภัย"]),
    (37.5, "ปานกลาง", ["ปานกลาง"]),
    (75, "เริ่มมีผลกระทบต่อสุขภาพ", ["เริ่มมีผลกระทบต่อสุขภาพ", "เริ่มมีผลกระทบ", "ส่งผลกระทบต่อสุขภาพ"]),
    (float("inf"), "มีผลกระทบต่อสุขภาพ", ["มีผลกระทบต่อสุขภาพ", "อันตราย", "อันตรายถึงชีวิต"]),
]
# keyword แต่ละตัวเป็นได้ทั้ง str (ต้องตรงตัวอักษรติดกันเป๊ะ) หรือ tuple ของ chunk 2 ชิ้นขึ้นไป
# (ยอมให้มีคำแทรกสั้นๆ คั่นระหว่าง chunk ได้ เช่น ("อากาศ","ดี") ต้อง match ทั้ง "อากาศดี" ตรงๆ
# และ "อากาศจะอยู่ในเกณฑ์ดี" ที่มีคำแทรก — ดู _find_keyword_occurrences())

# คำเตือนทั่วไปที่ "ใช้ได้ตั้งแต่ระดับหนึ่งขึ้นไป" ไม่ผูกกับระดับเดียวเป๊ะๆ เหมือน PM25_LEVELS
# (เช่น "หลีกเลี่ยงกิจกรรมกลางแจ้ง" ใช้ได้ตั้งแต่ "ปานกลาง" ขึ้นไปทั้งหมด) เก็บเป็น (ชื่อระดับ
# floor ต่ำสุดที่ใช้คำนี้ได้, keywords) แล้วเช็คแค่ฝั่ง "รุนแรงเกินจริง" เท่านั้น (ระดับจริงต่ำกว่า
# floor = เกินจริงแน่ๆ) — ไม่เช็คฝั่ง "ประเมินต่ำไป" เพราะคำพวกนี้ไม่ยืนยันเพดานบน
_GENERIC_CAUTION_KEYWORDS = [
    ("ปานกลาง", [("หลีกเลี่ยง", "กิจกรรมกลางแจ้ง"), ("ลด", "กิจกรรมกลางแจ้ง"), "ระคายเคือง"]),
]

_NEGATION_WORDS = ["ไม่ใช่", "ไม่ค่อย", "มิได้", "ไม่"]  # ถ้าอยู่ในช่วงคำแทรก ถือว่าความหมายกลับกัน
_MAX_FLEX_GAP = 20  # ช่องว่างของ keyword ทั่วไป (เช่น _GENERIC_CAUTION_KEYWORDS) — คำแทรกควรสั้น
# PM25_LEVELS เป็น "ประธาน...คำขยาย" ข้ามประโยคได้ (เช่น Gemini เขียน "อากาศในอีก 12 ชั่วโมง
# ข้างหน้าอยู่ในเกณฑ์ดี" แทรกวลีบอกเวลายาวๆ คั่นกลาง) จึงให้ช่องว่างกว้างกว่า _MAX_FLEX_GAP ทั่วไป
_MAX_FLEX_GAP_STATUS = 50


def _classify_pm25_level(value):
    for threshold, level_name, _ in PM25_LEVELS:
        if value <= threshold:
            return level_name
    return PM25_LEVELS[-1][1]


def pm25_status(value):
    return _classify_pm25_level(value)


# ===== NLG Safety Check (proposal 6.1.4) =====
# ก่อนส่งข้อความคำแนะนำผ่าน Push Notification ต้องเทียบระดับคำแนะนำที่ได้จาก NLG กับเกณฑ์
# มาตรฐานดัชนีคุณภาพอากาศ (PM25_LEVELS ด้านบน) หากไม่สอดคล้องกัน ใช้ข้อความสำเร็จรูป (template)
# แทน — rule-based ล้วนๆ ไม่พึ่ง SHAP หรือโมเดลใดๆ เลย
_HEALTH_MESSAGE_TEMPLATES = {
    "ดีมาก": "คุณภาพอากาศอยู่ในเกณฑ์ดีมาก สามารถทำกิจกรรมกลางแจ้งได้ตามปกติ",
    "ดี": "คุณภาพอากาศอยู่ในเกณฑ์ดี สามารถทำกิจกรรมกลางแจ้งได้ตามปกติ",
    "ปานกลาง": "คุณภาพอากาศอยู่ในเกณฑ์ปานกลาง กลุ่มเสี่ยงควรระมัดระวัง",
    "เริ่มมีผลกระทบต่อสุขภาพ": "คุณภาพอากาศเริ่มมีผลกระทบต่อสุขภาพ ควรลดกิจกรรมกลางแจ้งที่ใช้แรงมาก",
    "มีผลกระทบต่อสุขภาพ": "คุณภาพอากาศมีผลกระทบต่อสุขภาพ ควรหลีกเลี่ยงกิจกรรมกลางแจ้ง",
}


def _level_index(level_name):
    return next(i for i, (_, name, _) in enumerate(PM25_LEVELS) if name == level_name)


def _find_keyword_occurrences(message_text, keyword, max_gap=_MAX_FLEX_GAP):
    """คืน list ของ (start, end) ทุกตำแหน่งที่ keyword ปรากฏในข้อความ

    keyword เป็น str -> ต้องตรงตัวอักษรติดกันเป๊ะ
    keyword เป็น tuple ของ chunk >= 2 ชิ้น -> ยอมให้มีคำแทรกสั้นๆ (ไม่เกิน max_gap ตัวอักษร) คั่น
    ระหว่าง chunk ได้ (เช่น ("อากาศ","ดี") match ทั้ง "อากาศดี" และ "อากาศจะอยู่ในเกณฑ์ดี") แต่ถ้า
    คำแทรกมีคำปฏิเสธ (เช่น "ไม่") ถือว่าไม่ match เพราะความหมายกลับกัน (เช่น "อากาศไม่ดี")
    """
    if isinstance(keyword, str):
        occurrences = []
        start = 0
        while True:
            idx = message_text.find(keyword, start)
            if idx == -1:
                break
            occurrences.append((idx, idx + len(keyword)))
            start = idx + 1
        return occurrences

    occurrences = []
    search_start = 0
    first_chunk = keyword[0]
    while True:
        first_idx = message_text.find(first_chunk, search_start)
        if first_idx == -1:
            break
        pos = first_idx + len(first_chunk)
        matched = True
        for chunk in keyword[1:]:
            window_end = pos + max_gap + len(chunk)
            next_idx = message_text.find(chunk, pos, window_end)
            if next_idx == -1:
                matched = False
                break
            gap_text = message_text[pos:next_idx]
            if any(neg in gap_text for neg in _NEGATION_WORDS):
                matched = False
                break
            pos = next_idx + len(chunk)
        if matched:
            occurrences.append((first_idx, pos))
        search_start = first_idx + 1
    return occurrences


def _mentioned_levels(message_text):
    """หา level ทั้งหมดที่ข้อความ "พูดถึง" ผ่าน keyword ของ PM25_LEVELS (ไม่รวม
    _GENERIC_CAUTION_KEYWORDS ซึ่งเช็คแยกใน validate_health_message())

    ประมวลผล literal match (ตรงตัวอักษรเป๊ะ) ก่อนเสมอ แล้วค่อย flexible match (มีคำแทรกได้)
    ทีหลัง — ไม่ใช่แค่เรียงตามความยาว span เพราะ flexible match ที่กินคำแทรกยาวๆ อาจดูยาวกว่า
    literal match ที่แม่นยำกว่าทั้งที่ควรแพ้ ภายในกลุ่มเดียวกันถึงเรียงตาม span ยาวสุดไปสั้นสุด
    แล้ว mark ช่วงที่จับไปแล้วว่า "consumed" กันไม่ให้ match ที่หลวมกว่าซ้อนจับซ้ำ — จำเป็นเพราะ
    ชื่อระดับ PM2.5 บางคู่เป็น substring ของกันเอง (เช่น "มีผลกระทบต่อสุขภาพ" อยู่ใน "เริ่มมีผล
    กระทบต่อสุขภาพ", "อากาศดี" อยู่ใน "อากาศดีมาก") ถ้าไม่กันจุดนี้ ข้อความที่ถูกอยู่แล้วจะโดน
    flag ผิดว่าขัดแย้งกับตัวเอง
    """
    all_occurrences = []
    for _, level_name, keywords in PM25_LEVELS:
        for kw in keywords:
            is_flexible = not isinstance(kw, str)
            for start, end in _find_keyword_occurrences(message_text, kw, max_gap=_MAX_FLEX_GAP_STATUS):
                all_occurrences.append((start, end, level_name, is_flexible))

    all_occurrences.sort(key=lambda o: (o[3], -(o[1] - o[0])))

    consumed = [False] * len(message_text)
    mentioned = []
    for start, end, level_name, _ in all_occurrences:
        span = range(start, end)
        if not any(consumed[i] for i in span):
            if level_name not in mentioned:
                mentioned.append(level_name)
            for i in span:
                consumed[i] = True
    return mentioned


def _mentions_generic_caution_below(message_text, actual_idx):
    """True ถ้าข้อความมีคำเตือนทั่วไป (_GENERIC_CAUTION_KEYWORDS) ที่ floor สูงกว่าระดับจริง
    (เช่น เตือน "หลีกเลี่ยงกิจกรรมกลางแจ้ง" ทั้งที่ระดับจริงคือ "ดีมาก" ซึ่งต่ำกว่า floor "ปานกลาง")
    """
    for floor_level_name, keywords in _GENERIC_CAUTION_KEYWORDS:
        floor_idx = _level_index(floor_level_name)
        if floor_idx <= actual_idx:
            continue
        for kw in keywords:
            if _find_keyword_occurrences(message_text, kw):
                return True
    return False


def _fallback_message(level):
    if level is None:
        return "ไม่สามารถประเมินคุณภาพอากาศได้ในขณะนี้ กรุณาตรวจสอบใหม่ภายหลัง"
    return _HEALTH_MESSAGE_TEMPLATES[level]


def validate_health_message(message_text, predicted_pm25, risk_group="ทั่วไป"):
    """ตรวจว่าข้อความคำแนะนำสุขภาพ (จาก NLG) สอดคล้องกับระดับ PM2.5 จริง (PM25_LEVELS) ก่อนส่ง
    แจ้งเตือน — ไม่ผ่านถ้า: มี keyword ของระดับที่รุนแรงกว่าจริง (เกินจริง), keyword ของระดับที่
    เบากว่าจริงทั้งหมด (ประเมินต่ำไป, อันตรายเพราะทำให้ชะล่าใจ), หรือไม่มี keyword ระดับไหนเลย
    (fail-safe เพราะยืนยันความสอดคล้องไม่ได้)

    risk_group != "ทั่วไป": ข้าม _mentions_generic_caution_below() เพราะคำเตือนอย่าง "หลีกเลี่ยง
    กิจกรรมกลางแจ้ง" ที่ floor ปกติต้องระดับ "ปานกลาง"+ ถึงจะไม่ถือว่าเกินจริง เป็นคำแนะนำที่
    สมเหตุสมผล/ตั้งใจให้เกิดขึ้นสำหรับกลุ่มเปราะบางแม้ระดับจริงจะต่ำกว่านั้น (สอดคล้องกับ
    RISK_GROUP_THRESHOLD_OFFSET ที่ตั้งใจให้กลุ่มนี้ระวังตัวเร็วกว่าคนทั่วไปอยู่แล้ว) — ไม่ใช่การ
    อ้างระดับคุณภาพอากาศโดยรวมผิดไปจากจริง (เช็คระดับหลักที่ mentioned_levels ด้านล่างยังคงเข้มงวด
    เท่าเดิมทุกกลุ่ม)

    คืนค่า: (is_valid, level หรือ None, message_to_send) — ถ้าไม่ผ่าน message_to_send เป็น
    template สำเร็จรูปของระดับนั้นแทน (หรือข้อความทั่วไปถ้าประเมินระดับไม่ได้เลย)
    """
    if predicted_pm25 is None or (isinstance(predicted_pm25, float) and np.isnan(predicted_pm25)):
        return False, None, _fallback_message(None)

    if not message_text or not message_text.strip():
        actual_level = _classify_pm25_level(predicted_pm25)
        return False, actual_level, _fallback_message(actual_level)

    actual_level = _classify_pm25_level(predicted_pm25)
    actual_idx = _level_index(actual_level)

    mentioned_levels = _mentioned_levels(message_text)
    if not mentioned_levels:
        return False, actual_level, _fallback_message(actual_level)

    mentioned_indices = [_level_index(name) for name in mentioned_levels]
    if any(idx > actual_idx for idx in mentioned_indices):
        return False, actual_level, _fallback_message(actual_level)  # ข้อความรุนแรงเกินจริง
    if all(idx < actual_idx for idx in mentioned_indices):
        return False, actual_level, _fallback_message(actual_level)  # ข้อความประเมินต่ำเกินไป
    if risk_group == "ทั่วไป" and _mentions_generic_caution_below(message_text, actual_idx):
        return False, actual_level, _fallback_message(actual_level)  # คำเตือนทั่วไปที่สูงกว่า floor ของระดับจริง

    return True, actual_level, message_text


# ===== NLG Generation (proposal 4.1.4 / 6.1.4) =====
# แปลงค่าพยากรณ์ + เหตุผลจาก SHAP ให้เป็นคำอธิบายภาษาไทยสั้นๆ ผ่าน Gemini แล้วตรวจสอบด้วย
# validate_health_message() ก่อนคืนค่าเสมอ — ยังไม่ integrate เข้า get_forecast() (ดู
# scripts/test_nlg_generation.py สำหรับการทดสอบเรียกฟังก์ชันนี้ตรงๆ)
_gemini_client = genai.Client(api_key=GEMINI_KEY)
# gemini-2.5-flash เลิกให้ API key ใหม่ใช้แล้ว (ยืนยันจากการเรียกจริง) เปลี่ยนมาใช้
# gemini-3.1-flash-lite แทน — GA, มี free tier (15 RPM/1000 RPD), เร็ว/ถูกสุดในกลุ่ม flash
GEMINI_MODEL = "gemini-3.1-flash-lite"
GEMINI_TIMEOUT_SECONDS = 5
GEMINI_MAX_RETRIES = 3  # 429 (quota) backoff: รอ 1s, 2s, 4s ก่อนลองใหม่
# Gemini SDK บังคับ minimum server-side deadline 10s (พารามิเตอร์ timeout ตั้งต่ำกว่านั้นไม่ได้ —
# เจอตอนทดสอบจริง) จึงบังคับ timeout ฝั่ง client เองผ่าน thread แยก: รอผลไม่เกิน
# GEMINI_TIMEOUT_SECONDS แล้ว fallback ทันทีถ้าเกิน (thread เดิมรันต่อเบื้องหลังจนจบเอง ไม่ถูก
# cancel แต่ caller ไม่ต้องรอ — กัน get_forecast() ค้าง)
_gemini_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="gemini-nlg")

# ชื่อ feature ทางเทคนิค -> คำอธิบายภาษาไทยที่คนทั่วไปเข้าใจ ใช้ตอนสร้าง prompt ส่งให้ Gemini
_FEATURE_NAME_TH = {
    "pm25_lag_1": "ค่าฝุ่นเมื่อชั่วโมงที่แล้ว",
    "pm25_lag_3": "ค่าฝุ่นเมื่อ 3 ชั่วโมงก่อน",
    "pm25_lag_6": "ค่าฝุ่นเมื่อ 6 ชั่วโมงก่อน",
    "pm25_lag_12": "ค่าฝุ่นเมื่อ 12 ชั่วโมงก่อน",
    "pm25_lag_24": "ค่าฝุ่นเมื่อวานช่วงเวลาเดียวกัน",
    "pm25_roll_mean_6": "ค่าเฉลี่ยฝุ่นในช่วง 6 ชั่วโมงล่าสุด",
    "pm25_roll_std_6": "ความผันผวนของฝุ่นในช่วง 6 ชั่วโมงล่าสุด",
    "pm25_roll_mean_24": "ค่าเฉลี่ยฝุ่นในช่วง 24 ชั่วโมงล่าสุด",
    "pm25_roll_std_24": "ความผันผวนของฝุ่นในช่วง 24 ชั่วโมงล่าสุด",
    "temp": "อุณหภูมิ",
    "humidity": "ความชื้นในอากาศ",
    "wind_speed": "ความเร็วลม",
    "precipitation": "ปริมาณฝน",
    "hour_sin": "ช่วงเวลาของวัน",
    "hour_cos": "ช่วงเวลาของวัน",
    "month_sin": "ช่วงฤดูกาล",
    "month_cos": "ช่วงฤดูกาล",
    "dayofweek": "วันในสัปดาห์",
}

# แต่ละ tone มีเงื่อนไข "สอดคล้องกับระดับจริง" กำกับไว้เสมอ กันข้อความย้อนแย้งในตัวเอง (เคยเกิดจริง:
# วันที่อากาศ "ดี" แต่ tone กลุ่มเปราะบางไปเน้นให้ Gemini เขียนแนะนำ "หลีกเลี่ยงกิจกรรมกลางแจ้ง"
# แบบไม่มีเงื่อนไข ได้ข้อความที่บอกว่าอากาศดีแต่ก็เตือนให้เลี่ยงออกไปข้างนอกในประโยคเดียวกัน อ่านแล้ว
# งงว่าตกลงจะดีหรือไม่ดี — แก้โดยสั่งชัดเจนว่าให้ระดับความเข้มของคำแนะนำแปรตามสถานการณ์จริง ไม่ใช่
# เข้มข้นคงที่ทุกระดับ)
_USER_GROUP_TONE = {
    "ทั่วไป": "ให้คำแนะนำทั่วไปที่เหมาะกับคนทั่วไป",
    "เด็ก-ผู้สูงอายุ": (
        "เน้นย้ำว่ากลุ่มเด็กและผู้สูงอายุไวต่อฝุ่นมากกว่าคนทั่วไป แต่คำแนะนำต้องสอดคล้องกับระดับ"
        "คุณภาพอากาศจริงเสมอ ห้ามย้อนแย้งกัน — ถ้าอากาศอยู่ในเกณฑ์ดีมาก/ดี ให้เขียนเชิงให้ความมั่นใจ "
        "แค่แนะนำให้สังเกตอาการตนเองเบาๆ พอ ห้ามแนะนำให้หลีกเลี่ยงหรือลดกิจกรรมกลางแจ้งโดยไม่มีเหตุผล"
        "รองรับจากระดับจริง"
    ),
    "มีโรคประจำตัว": (
        "เน้นย้ำว่าผู้มีโรคประจำตัวทางเดินหายใจไวต่อฝุ่นมากกว่าคนทั่วไป แต่คำแนะนำต้องสอดคล้องกับระดับ"
        "คุณภาพอากาศจริงเสมอ ห้ามย้อนแย้งกัน — ถ้าอากาศอยู่ในเกณฑ์ดีมาก/ดี ให้เขียนเชิงให้ความมั่นใจ "
        "แค่แนะนำให้สังเกตอาการตนเองเบาๆ พอ ห้ามแนะนำให้หลีกเลี่ยงหรือลดกิจกรรมกลางแจ้งโดยไม่มีเหตุผล"
        "รองรับจากระดับจริง"
    ),
}

# ใช้ตรวจสอบค่า risk_group ตอนสมัคร/แก้โปรไฟล์ (ดู ===== User accounts (auth) ===== ด้านบน) —
# ผูกกับโทน NLG รายวัน (_build_daily_nlg_prompt) และ alert threshold (RISK_GROUP_THRESHOLD_OFFSET)
# แล้ว — NLG ราย*ชั่วโมง* ถูกลบทิ้งแล้ว (ดู memory project_hourly_nlg_deferred) ไม่ใช่แค่ยังไม่ผูก
VALID_RISK_GROUPS = set(_USER_GROUP_TONE.keys())

_NLG_SYSTEM_PROMPT = (
    "คุณเป็นผู้ช่วยอธิบายค่าพยากรณ์คุณภาพอากาศ PM2.5 ให้ประชาชนทั่วไปฟัง ตอบเป็นภาษาไทยเท่านั้น "
    "ความยาวไม่เกิน 2-3 ประโยค กระชับ เข้าใจง่าย ห้ามใช้ศัพท์เทคนิคหรือชื่อโมเดล/ฟีเจอร์ตรงๆ "
    "(ห้ามพูดคำว่า SHAP, feature, lag, rolling, โมเดล) ให้อธิบายด้วยภาษาที่คนทั่วไปคุ้นเคยแทน"
)

def _feature_name_th(feature_name):
    return _FEATURE_NAME_TH.get(feature_name, feature_name)


def _format_shap_factor_lines(shap_pairs, limit=3):
    top_factors = (shap_pairs or [])[:limit]
    if not top_factors:
        return "- ไม่มีข้อมูลปัจจัยเพิ่มเติมสำหรับรอบนี้"
    return "\n".join(
        f"- {_feature_name_th(name)}: {'ดันค่าฝุ่นให้สูงขึ้น' if val > 0 else 'ดันค่าฝุ่นให้ลดลง'}"
        for name, val in top_factors
    )


def _build_daily_nlg_prompt(daily_avg_pm25, status, shap_pairs, risk_group="ทั่วไป"):
    factor_lines = _format_shap_factor_lines(shap_pairs)
    tone = _USER_GROUP_TONE.get(risk_group, _USER_GROUP_TONE["ทั่วไป"])

    return (
        f'ค่าเฉลี่ย PM2.5 ตลอดวันนี้ (สรุปจากค่าพยากรณ์ล่วงหน้า 24 ชั่วโมง) '
        f'อยู่ที่ {daily_avg_pm25} ไมโครกรัม/ลูกบาศก์เมตร ซึ่งจัดอยู่ในระดับคุณภาพอากาศ "{status}"\n\n'
        f"ปัจจัยที่มีอิทธิพลต่อแนวโน้มตลอดวันนี้มากที่สุด (ประเมินจากช่วงกลางวัน):\n{factor_lines}\n\n"
        f"ช่วยเขียนคำอธิบายแนวโน้มคุณภาพอากาศตลอดทั้งวันนี้สั้นๆ ให้ผู้ใช้งานทั่วไปเข้าใจง่าย "
        f"โดย{tone} "
        f'ห้ามเอ่ยชื่อระดับคุณภาพอากาศอื่นที่ไม่ใช่ "{status}" เด็ดขาด (เช่น ห้ามพูดถึงคำว่า '
        f'"ปานกลาง" หรือ "เริ่มมีผลกระทบต่อสุขภาพ" ถ้าระดับปัจจุบันไม่ใช่ระดับนั้น) '
        f"ให้บรรยายความระมัดระวังด้วยคำทั่วไปแทนการเอ่ยชื่อระดับ"
    )


def _call_gemini(prompt):
    """เรียก Gemini พร้อม timeout (client-side, ดู _gemini_executor) + retry แบบ exponential
    backoff เฉพาะ 429 (quota) คืนค่า None ถ้าไม่สำเร็จไม่ว่าด้วยสาเหตุใด เพื่อให้ผู้เรียก fallback
    ไป template message ได้ทันที
    """
    delay = 1
    for attempt in range(GEMINI_MAX_RETRIES):
        future = _gemini_executor.submit(
            _gemini_client.models.generate_content,
            model=GEMINI_MODEL,
            contents=prompt,
            config=genai_types.GenerateContentConfig(system_instruction=_NLG_SYSTEM_PROMPT),
        )
        try:
            response = future.result(timeout=GEMINI_TIMEOUT_SECONDS)
            return (response.text or "").strip()
        except FutureTimeoutError:
            print(f"[nlg] Gemini call exceeded {GEMINI_TIMEOUT_SECONDS}s timeout, falling back to template")
            return None
        except genai_errors.APIError as e:
            if e.code == 429 and attempt < GEMINI_MAX_RETRIES - 1:
                print(f"[nlg] Gemini 429 (quota), retry {attempt + 1}/{GEMINI_MAX_RETRIES} in {delay}s")
                time.sleep(delay)
                delay *= 2
                continue
            print(f"[nlg] Gemini API error (code={e.code}), falling back to template")
            return None
        except Exception as e:
            print(f"[nlg] Gemini call failed ({type(e).__name__}), falling back to template")
            return None
    return None


# ===== NLG รายวัน (สรุปทั้งวันแทนรายชั่วโมง) =====
# แทนที่จะเรียก Gemini ทุกชั่วโมง สร้างคำอธิบายแค่ 1 ครั้งต่อวันตอน 06:00 น. เวลาไทย โดยใช้
# ค่าเฉลี่ยพยากรณ์ 24 ชม. ข้างหน้าเป็นตัวแทนทั้งวัน — เก็บผลไว้ใน daily_nlg_log กัน generate ซ้ำ

def get_thai_now():
    """เวลาปัจจุบันแปลงเป็นเวลาไทย (Asia/Bangkok) — pattern .tz_convert() เดียวกับที่ใช้ทั่วไฟล์
    (เช่น latest_time_th ใน get_forecast()) แยกเป็นฟังก์ชันต่างหาก (ไม่ inline) เพื่อให้ test
    monkeypatch แทนเวลาจริงได้ — สำคัญเพราะวันที่ต้องอิงเวลาไทยหลังแปลงแล้วเสมอ (ห้ามใช้วันที่
    ฝั่ง UTC ตรงๆ เนื่องจากต่างโซนเวลากันเกือบครึ่งวัน จะข้ามวันผิดช่วง 17:00-23:59 UTC)
    """
    return pd.Timestamp.now(tz="UTC").tz_convert("Asia/Bangkok")


def get_daily_nlg_row(date_thai, risk_group="ทั่วไป"):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """SELECT date_thai, daily_avg_pm25, explanation_text, used_fallback, generated_at_utc
           FROM daily_nlg_log WHERE date_thai = %s AND risk_group = %s""",
        (date_thai, risk_group),
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    if row is None:
        return None
    return {
        "date_thai": row[0],
        "daily_avg_pm25": row[1],
        "explanation_text": row[2],
        "used_fallback": bool(row[3]),
        "generated_at_utc": row[4],
    }


def save_daily_nlg_row(date_thai, risk_group, daily_avg_pm25, explanation_text, used_fallback, generated_at_utc):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO daily_nlg_log
           (date_thai, risk_group, daily_avg_pm25, explanation_text, used_fallback, generated_at_utc)
           VALUES (%s, %s, %s, %s, %s, %s)""",
        (date_thai, risk_group, daily_avg_pm25, explanation_text, bool(used_fallback), generated_at_utc),
    )
    conn.commit()
    cur.close()
    conn.close()


def generate_daily_nlg_explanation(forecast, feature_row, risk_group="ทั่วไป"):
    """สร้างคำอธิบาย NLG แทนทั้งวัน โดยใช้ค่าเฉลี่ยของ forecast[] ทั้ง 24 จุดเป็นตัวแทน และ SHAP
    ของ horizon t+12 (จุดกึ่งกลางวัน) เป็นตัวแทนอธิบายปัจจัยหลัก — validate ผ่าน validate_health_message()
    เสมอก่อนคืนค่า ไม่ผ่านหรือเรียก Gemini ไม่สำเร็จ fallback ไป _fallback_message() แทน

    risk_group ปรับโทนคำแนะนำท้ายข้อความผ่าน _USER_GROUP_TONE

    คืนค่า (daily_avg_pm25, explanation_text, used_fallback)
    """
    daily_avg_pm25 = round(sum(f["pm25"] for f in forecast) / len(forecast), 1)
    status = pm25_status(daily_avg_pm25)

    # h=12 อยู่ที่ index 11 ใน forecast[] (h วนตั้งแต่ 1 ถึง 24 -> index 0 ถึง 23)
    model_used_h12 = forecast[11]["model_used"]
    shap_pairs = get_shap_explanation(model_used_h12, 12, feature_row)

    prompt = _build_daily_nlg_prompt(daily_avg_pm25, status, shap_pairs, risk_group)
    message_text = _call_gemini(prompt)

    used_fallback = False
    if message_text:
        is_valid, _, final_message = validate_health_message(message_text, daily_avg_pm25, risk_group)
        if not is_valid:
            print(f"[nlg-daily] safety check failed: Gemini message did not match actual level "
                  f"'{status}', using fallback template")
            used_fallback = True
    else:
        final_message = _fallback_message(status)
        used_fallback = True

    return daily_avg_pm25, final_message, used_fallback


def maybe_generate_daily_nlg(forecast, feature_row, risk_group="ทั่วไป"):
    """เรียก generate_daily_nlg_explanation() แค่ครั้งเดียวต่อวัน**ต่อ risk_group** (เวลาไทย) ตั้งแต่
    06:00 น. เป็นต้นไป (>= 6 ไม่ใช่ == 6 พอดี กัน process ที่ไม่ได้รันตรงช่วง 06:00-06:59 น. พลาดไปทั้งวัน
    — จะ generate ทันทีที่เรียก get_forecast() ครั้งแรกของวันหลัง 06:00 น. แทน) ไม่ raise exception ออกไป
    (แค่ log) กัน /api/forecast พังเพราะ NLG รายวันจุดเดียว

    ตัวไหนถูกเรียกจริงเป็น lazy ตาม risk_group ที่มีคนขอ (สูงสุด 3 ครั้ง/วัน ไม่ใช่ generate ล่วงหน้า
    ทั้ง 3 กลุ่ม) 'ทั่วไป' ยังถูก warm ทุกวันผ่าน scheduled_forecast_job() ที่ไม่ผูก user เหมือนเดิม
    """
    thai_now = get_thai_now()
    if thai_now.hour < 6:
        return

    date_thai = thai_now.strftime("%Y-%m-%d")
    if get_daily_nlg_row(date_thai, risk_group) is not None:
        return

    try:
        daily_avg_pm25, explanation_text, used_fallback = generate_daily_nlg_explanation(forecast, feature_row, risk_group)
        generated_at_utc = thai_now.tz_convert("UTC").isoformat()
        save_daily_nlg_row(date_thai, risk_group, daily_avg_pm25, explanation_text, used_fallback, generated_at_utc)
        print(f"[nlg-daily] generated for {date_thai} ({risk_group}): avg_pm25={daily_avg_pm25}, fallback={used_fallback}")
    except Exception as e:
        print(f"[nlg-daily] generation failed for {date_thai} ({risk_group}): {e}")


def get_today_daily_nlg(risk_group="ทั่วไป"):
    """ดึงข้อความ NLG รายวันของวันนี้ (เวลาไทย) ตาม risk_group — อ่านอย่างเดียว ไม่เรียก Gemini ใหม่
    (การสร้างจริงเกิดที่ maybe_generate_daily_nlg() เท่านั้น) ถ้ายังไม่มีแถวของวันนี้เลย คืน placeholder แทน error
    """
    date_thai = get_thai_now().strftime("%Y-%m-%d")
    row = get_daily_nlg_row(date_thai, risk_group)
    if row is None:
        return {
            "date_thai": date_thai,
            "daily_avg_pm25": None,
            "explanation": "กำลังเตรียมข้อมูลสำหรับวันนี้",
            "used_fallback": None,
        }
    return {
        "date_thai": row["date_thai"],
        "daily_avg_pm25": row["daily_avg_pm25"],
        "explanation": row["explanation_text"],
        "used_fallback": row["used_fallback"],
    }


MIN_LIVE_FOR_SUMMARY = 5  # เกณฑ์เดียวกับ compute_rolling_r2(): live pair น้อยกว่านี้ไม่น่าเชื่อถือพอ


def summarize_errors(all_errors, live_errors):
    """ใช้ error จาก live เท่านั้นถ้ามีพอ (>= MIN_LIVE_FOR_SUMMARY) กัน MAE/bias ดูดีเกินจริงจากแถว
    backfill (คัดจาก test set แบบไม่มี gap เลย จึง optimistic กว่า live performance จริง) — live
    ยังน้อยเกินไปให้ fallback ไปรวม backfill เหมือนเดิม (basis='mixed')

    คืนค่า (mae, bias, n_used, basis) โดย basis คือ 'live' หรือ 'mixed'
    """
    if len(live_errors) >= MIN_LIVE_FOR_SUMMARY:
        errs, basis = live_errors, "live"
    else:
        errs, basis = all_errors, "mixed"
    if not errs:
        return None, None, 0, basis
    mae = sum(abs(e) for e in errs) / len(errs)
    bias = sum(errs) / len(errs)
    return mae, bias, len(errs), basis


# ===== Core forecast pipeline =====

def get_forecast(risk_group="ทั่วไป"):
    df_recent = fetch_recent_pm25()
    current_weather = fetch_current_weather()
    X_now, latest_time, latest_pm25 = build_feature_row(df_recent, current_weather)
    latest_time_th = latest_time.tz_convert("Asia/Bangkok")

    forecast = []
    log_rows = []
    risk_horizons = []
    prophet_triggered_horizons = []
    for h in range(1, 25):
        model_name = select_model(X_now, h)

        if model_name == "xgboost":
            pred = round(float(XGB_MODELS[h].predict(X_now)[0]), 1)
        elif model_name == "prophet":
            pred = round(predict_with_prophet(PROPHET_MODEL, current_weather, latest_time, h), 1)
            prophet_triggered_horizons.append(h)
        else:  # lightgbm (default)
            pred = round(float(MODELS[h].predict(X_now)[0]), 1)

        forecast_time_utc = latest_time + pd.Timedelta(hours=h)
        forecast_time_th = forecast_time_utc.tz_convert("Asia/Bangkok")
        forecast.append({
            "time": forecast_time_th.strftime("%H:%M"),
            "hour_thai": forecast_time_th.hour,
            "pm25": pred,
            "model_used": model_name,
        })
        log_rows.append({
            "horizon": h,
            "target_time_utc": forecast_time_utc,
            "pm25": pred,
            "model_used": model_name,
        })

        # threshold แยกตามโมเดลที่ทำนายจริง (bias_high ไม่เท่ากันแต่ละโมเดล — ดู
        # EFFECTIVE_THRESHOLD_BY_HORIZON_BY_MODEL) และตาม risk_group ของผู้เรียก (ดู
        # RISK_GROUP_THRESHOLD_OFFSET) Prophet ไม่มี bias ที่เชื่อถือได้ เทียบ 37.5 (+offset) ตรงๆ
        if is_horizon_at_risk(pred, model_name, h, risk_group):
            risk_horizons.append(h)

    log_predictions(latest_time, log_rows)
    log_notification_snapshot(latest_time.isoformat(), float(latest_pm25))
    backfill_actuals(df_recent)
    maybe_generate_daily_nlg(forecast, X_now, risk_group)

    best_time = find_best_time(forecast)

    # latest_pm25 เป็นข้อมูลจริง ไม่มี bias จากโมเดล จึงเทียบกับ 37.5 (+ offset ตาม risk_group) ตรงๆ
    # (ต่างจาก risk_horizons ด้านบนที่ยังต้องหัก bias ของโมเดลด้วย)
    high_pollution_risk = is_current_at_risk(latest_pm25, risk_group) or len(risk_horizons) > 0

    return {
        "current": {
            "pm25": round(float(latest_pm25), 1),
            "datetime_th": latest_time_th.strftime("%Y-%m-%d %H:%M"),
            "status": pm25_status(latest_pm25),
        },
        # current_weather ดึงมาใช้เป็น model feature อยู่แล้ว (ดู build_feature_row()) แค่ส่งต่อให้
        # frontend ด้วย ไม่มีการยิง API เพิ่ม — ตั้งชื่อ field ใหม่ให้หน่วยชัดเจน (Open-Meteo คืนค่า
        # เป็น °C/%/km/h เป็นค่าเริ่มต้นอยู่แล้ว)
        "current_weather": {
            "temp_c": round(current_weather["temperature_2m"], 1),
            "humidity_pct": round(current_weather["relative_humidity_2m"]),
            "wind_speed_kmh": round(current_weather["wind_speed_10m"], 1),
        },
        "forecast": forecast,
        "best_time": best_time,
        "high_pollution_risk": high_pollution_risk,
        "risk_horizons": risk_horizons,
        "prophet_triggered_horizons": prophet_triggered_horizons,
        "daily_summary": get_today_daily_nlg(risk_group),
        "risk_group": risk_group,
        # activity เลือกฝั่ง frontend เท่านั้น (localStorage, ไม่ persist ที่ backend — ดู
        # compute_risk_score()) จึงส่งแค่ config ให้ frontend คำนวณคะแนนเองจาก pm25 ที่มีอยู่แล้วใน
        # response นี้ (current.pm25/forecast[].pm25) เลขคงที่ (offset/breakpoint) ยังอยู่จุดเดียวคือ
        # ที่นี่ ป้องกัน frontend/backend คำนวณไม่ตรงกัน
        "risk_score_config": {
            "risk_group_offset": RISK_GROUP_THRESHOLD_OFFSET.get(risk_group, 0.0),
            "activity_offsets": ACTIVITY_INTENSITY_OFFSET,
            "breakpoints": RISK_SCORE_BREAKPOINTS,
        },
    }


def scheduled_forecast_job():
    """เรียก get_forecast() อัตโนมัติทุกชั่วโมง ให้ prediction_log สะสมข้อมูลต่อเนื่อง แม้ไม่มีใครเปิดหน้าเว็บ"""
    try:
        get_forecast()
        print("[scheduler] auto forecast logged OK")
    except Exception as e:
        print(f"[scheduler] auto forecast failed: {e}")


# ===== Flask routes =====

@app.route("/index.html")
def index_html_redirect():
    return redirect("/")


@app.route("/compare.html")
def compare_html_redirect():
    return redirect("/compare")


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_spa(path):
    """เสิร์ฟ Vue SPA ที่ build แล้ว (frontend/dist) — path ที่ไม่ใช่ไฟล์จริงใน dist (เช่น
    client-side route /compare ของ Vue Router) fallback ไป index.html เสมอ Werkzeug จับ static
    route อย่าง /api/forecast ก่อน catch-all นี้เสมอไม่ว่าจะลงทะเบียนก่อนหลัง จึงไม่บัง API ด้านล่าง"""
    full_path = os.path.join(FRONTEND_DIST, path)
    if path and os.path.isfile(full_path):
        return send_from_directory(FRONTEND_DIST, path)
    return send_from_directory(FRONTEND_DIST, "index.html")


# ===== API routes =====

@app.route("/api/auth/register", methods=["POST"])
def api_auth_register():
    try:
        data = request.get_json(silent=True) or {}
        email = str(data.get("email", "")).strip().lower()
        password = str(data.get("password", ""))
        risk_group = data.get("risk_group") or "ทั่วไป"
        age = data.get("age")

        if not _EMAIL_RE.match(email):
            return jsonify({"error": "รูปแบบอีเมลไม่ถูกต้อง"}), 400
        if len(password) < MIN_PASSWORD_LENGTH:
            return jsonify({"error": f"รหัสผ่านต้องมีอย่างน้อย {MIN_PASSWORD_LENGTH} ตัวอักษร"}), 400
        if risk_group not in VALID_RISK_GROUPS:
            return jsonify({"error": "risk_group ไม่ถูกต้อง"}), 400
        if age is not None:
            try:
                age = int(age)
            except (TypeError, ValueError):
                return jsonify({"error": "age ต้องเป็นตัวเลข"}), 400
            if not 1 <= age <= 120:
                return jsonify({"error": "age ต้องอยู่ระหว่าง 1-120"}), 400

        if get_user_by_email(email) is not None:
            return jsonify({"error": "อีเมลนี้ถูกใช้งานแล้ว"}), 409

        user_id = create_user(email, generate_password_hash(password), risk_group, age)
        session.clear()
        session.permanent = True
        session["user_id"] = user_id
        return jsonify(get_user_by_id(user_id)), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/auth/login", methods=["POST"])
def api_auth_login():
    try:
        data = request.get_json(silent=True) or {}
        email = str(data.get("email", "")).strip().lower()
        password = str(data.get("password", ""))

        user = get_user_by_email_with_hash(email)
        if user is None or not check_password_hash(user["password_hash"], password):
            return jsonify({"error": "อีเมลหรือรหัสผ่านไม่ถูกต้อง"}), 401

        session.clear()
        session.permanent = True
        session["user_id"] = user["id"]
        return jsonify(get_user_by_id(user["id"])), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/auth/logout", methods=["POST"])
def api_auth_logout():
    session.clear()
    return jsonify({"ok": True}), 200


@app.route("/api/auth/me", methods=["GET"])
@login_required
def api_auth_me():
    user = get_user_by_id(session["user_id"])
    if user is None:  # แถวถูกลบไปแล้วแต่ session ยังค้างอยู่
        session.clear()
        return jsonify({"error": "กรุณาเข้าสู่ระบบ"}), 401
    return jsonify(user), 200


@app.route("/api/auth/profile", methods=["PUT"])
@login_required
def api_auth_update_profile():
    try:
        current = get_user_by_id(session["user_id"])
        if current is None:
            session.clear()
            return jsonify({"error": "กรุณาเข้าสู่ระบบ"}), 401

        data = request.get_json(silent=True) or {}
        email = str(data.get("email", current["email"])).strip().lower()
        risk_group = data.get("risk_group", current["risk_group"])
        age = data.get("age", current["age"])

        if not _EMAIL_RE.match(email):
            return jsonify({"error": "รูปแบบอีเมลไม่ถูกต้อง"}), 400
        if risk_group not in VALID_RISK_GROUPS:
            return jsonify({"error": "risk_group ไม่ถูกต้อง"}), 400
        if age is not None:
            try:
                age = int(age)
            except (TypeError, ValueError):
                return jsonify({"error": "age ต้องเป็นตัวเลข"}), 400
            if not 1 <= age <= 120:
                return jsonify({"error": "age ต้องอยู่ระหว่าง 1-120"}), 400

        existing = get_user_by_email(email)
        if existing is not None and existing["id"] != session["user_id"]:
            return jsonify({"error": "อีเมลนี้ถูกใช้งานแล้ว"}), 409

        update_user_profile(session["user_id"], email, risk_group, age)
        return jsonify(get_user_by_id(session["user_id"])), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/auth/change-password", methods=["POST"])
@login_required
def api_auth_change_password():
    try:
        user = get_user_by_email_with_hash(get_user_by_id(session["user_id"])["email"])
        data = request.get_json(silent=True) or {}
        current_password = str(data.get("current_password", ""))
        new_password = str(data.get("new_password", ""))

        if not check_password_hash(user["password_hash"], current_password):
            return jsonify({"error": "รหัสผ่านปัจจุบันไม่ถูกต้อง"}), 400
        if len(new_password) < MIN_PASSWORD_LENGTH:
            return jsonify({"error": f"รหัสผ่านใหม่ต้องมีอย่างน้อย {MIN_PASSWORD_LENGTH} ตัวอักษร"}), 400

        update_user_password(user["id"], generate_password_hash(new_password))
        return jsonify({"ok": True}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def resolve_risk_group_from_session():
    """ไม่บังคับ login — ผู้ใช้ที่ไม่ login (guest) ได้ risk_group = "ทั่วไป" เสมอ ใช้ร่วมกันทั้ง
    /api/forecast และ /api/notifications"""
    risk_group = "ทั่วไป"
    user_id = session.get("user_id")
    if user_id is not None:
        user = get_user_by_id(user_id)
        if user is not None:
            risk_group = user["risk_group"]
    return risk_group


@app.route("/api/forecast", methods=["GET"])
def api_forecast():
    try:
        return jsonify(get_forecast(resolve_risk_group_from_session()))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/notifications", methods=["GET"])
def api_notifications():
    """ประวัติการแจ้งเตือน (เฉพาะรอบที่ high_pollution_risk=true) — คำนวณ risk_horizons สดจาก
    prediction_log ตาม risk_group ของผู้เรียก ณ เวลา query ไม่ใช่ค่าที่ freeze ไว้ตอนบันทึก (ดูเหตุผล
    เต็มที่ comment เหนือ notification_log ใน init_db())"""
    try:
        risk_group = resolve_risk_group_from_session()
        limit = request.args.get("limit", default=20, type=int)
        limit = max(1, min(limit, 100))

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            "SELECT recorded_at_utc, latest_pm25 FROM notification_log ORDER BY recorded_at_utc DESC LIMIT %s",
            (limit,),
        )
        snapshots = cur.fetchall()

        results = []
        for recorded_at_utc, latest_pm25 in snapshots:
            cur.execute(
                "SELECT horizon, predicted_pm25, model_used FROM prediction_log WHERE predicted_at_utc = %s",
                (recorded_at_utc,),
            )
            horizon_rows = cur.fetchall()

            risk_horizons = sorted(
                h for h, pred, model_used in horizon_rows
                if is_horizon_at_risk(pred, model_used, h, risk_group)
            )
            prophet_horizons = sorted(h for h, _, model_used in horizon_rows if model_used == "prophet")

            if not (is_current_at_risk(latest_pm25, risk_group) or risk_horizons):
                continue

            recorded_at_th = pd.Timestamp(recorded_at_utc).tz_convert("Asia/Bangkok")
            results.append({
                "recorded_at_th": recorded_at_th.strftime("%Y-%m-%d %H:%M"),
                "latest_pm25": round(float(latest_pm25), 1),
                "risk_horizons": risk_horizons,
                "prophet_horizons": prophet_horizons,
            })

        cur.close()
        conn.close()
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/comparison", methods=["GET"])
def api_comparison():
    try:
        horizon = int(request.args.get("horizon", 1))
        if not 1 <= horizon <= 24:
            return jsonify({"error": "horizon ต้องอยู่ระหว่าง 1-24"}), 400

        model = request.args.get("model", "all")
        if model not in ("all", "lightgbm", "xgboost", "prophet"):
            return jsonify({"error": "model ต้องเป็น all/lightgbm/xgboost/prophet"}), 400

        conn = get_db_connection()
        cur = conn.cursor()
        # MAE/bias ต้องเทียบสเกลเดียวกับ predicted_pm25 จึงใช้ actual_pm25_calibrated — แถวที่ยังเป็น
        # NULL (ยังไม่ผ่าน migration) fallback คำนวณสดใน query: backfill สเกล OpenAQ อยู่แล้วใช้ตรงๆ,
        # live เป็นค่าดิบ Air4Thai ต้องผ่าน a*x+b
        calibrated_expr = """COALESCE(actual_pm25_calibrated,
                       CASE WHEN source = 'backfill' THEN actual_pm25
                            ELSE %s * actual_pm25 + %s END)"""
        if model == "all":
            cur.execute(
                f"""SELECT target_time_utc, predicted_pm25, actual_pm25, {calibrated_expr},
                          model_used, source
                   FROM prediction_log
                   WHERE horizon = %s
                   ORDER BY target_time_utc""",
                (CALIBRATION_A, CALIBRATION_B, horizon),
            )
        else:
            cur.execute(
                f"""SELECT target_time_utc, predicted_pm25, actual_pm25, {calibrated_expr},
                          model_used, source
                   FROM prediction_log
                   WHERE horizon = %s AND model_used = %s
                   ORDER BY target_time_utc""",
                (CALIBRATION_A, CALIBRATION_B, horizon, model),
            )
        rows = cur.fetchall()
        cur.close()
        conn.close()

        points = []
        errors = []
        live_errors = []
        for target_time_str, predicted, actual, actual_calibrated, model_used, source in rows:
            target_time_th = pd.Timestamp(target_time_str).tz_convert("Asia/Bangkok")
            points.append({
                "time": target_time_th.strftime("%m-%d %H:%M"),
                "predicted": round(predicted, 1),
                # แสดงค่า calibrated ในกราฟ เพื่อให้เส้น actual/predicted อยู่สเกลเดียวกัน
                # (ค่าดิบ Air4Thai ยังอยู่ใน actual_raw เผื่ออยากดูค่าอ่านจริงของสถานี)
                "actual": round(actual_calibrated, 1) if actual_calibrated is not None else None,
                "actual_raw": round(actual, 1) if actual is not None else None,
                "model_used": model_used,
                "source": source,
            })
            if actual_calibrated is not None:
                err = actual_calibrated - predicted
                errors.append(err)
                if source != "backfill":
                    live_errors.append(err)

        n_matched = len(errors)
        n_backfill = sum(1 for p in points if p["source"] == "backfill")

        # เลือกโมเดลเจาะจง (ไม่ใช่ "all") -> ใช้ live-only สำหรับ MAE/bias ถ้ามีพอ กันเลขดูดีเกินจริง
        # จาก backfill ที่คัดมาสะอาดเป็นพิเศษ (ดู summarize_errors) — "all" รวมหลายโมเดลอยู่แล้วไม่ apply switch นี้
        if model != "all":
            mae, bias, n_used, basis = summarize_errors(errors, live_errors)
        else:
            mae = sum(abs(e) for e in errors) / n_matched if n_matched else None
            bias = sum(errors) / n_matched if n_matched else None
            n_used, basis = n_matched, None

        return jsonify({
            "horizon": horizon,
            "model": model,
            "points": points,
            "summary": {
                "n_total": len(points),
                "n_matched": n_matched,
                "n_backfill": n_backfill,
                "n_used": n_used,
                "basis": basis,
                "mae": round(mae, 3) if mae is not None else None,
                "bias": round(bias, 3) if bias is not None else None,
            },
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/comparison/overview", methods=["GET"])
def api_comparison_overview():
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM prediction_log")
        n_total = cur.fetchone()[0]
        # ใช้ actual_pm25_calibrated (สเกลเดียวกับ predicted_pm25) — fallback logic เดียวกับ
        # /api/comparison: แถว backfill ใช้ค่าเดิม (OpenAQ scale อยู่แล้ว), แถว live ผ่าน a*x+b
        cur.execute(
            """SELECT horizon, predicted_pm25,
                      COALESCE(actual_pm25_calibrated,
                               CASE WHEN source = 'backfill' THEN actual_pm25
                                    ELSE %s * actual_pm25 + %s END),
                      model_used, source
               FROM prediction_log WHERE actual_pm25 IS NOT NULL""",
            (CALIBRATION_A, CALIBRATION_B),
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()

        by_horizon_errors = {h: {"all": [], "live": []} for h in range(1, 25)}
        by_model_errors = {}
        all_errors, live_errors = [], []
        for horizon, predicted, actual, model_used, source in rows:
            err = actual - predicted
            is_live = source != "backfill"

            all_errors.append(err)
            by_horizon_errors[horizon]["all"].append(err)
            by_model_errors.setdefault(model_used, {"all": [], "live": [], "n_backfill": 0, "n_live": 0})
            by_model_errors[model_used]["all"].append(err)

            if is_live:
                live_errors.append(err)
                by_horizon_errors[horizon]["live"].append(err)
                by_model_errors[model_used]["live"].append(err)
                by_model_errors[model_used]["n_live"] += 1
            else:
                by_model_errors[model_used]["n_backfill"] += 1

        n_matched = len(all_errors)
        overall_mae, overall_bias, overall_n_used, overall_basis = summarize_errors(all_errors, live_errors)

        by_horizon = []
        for h in range(1, 25):
            mae_h, _, n_h, basis_h = summarize_errors(by_horizon_errors[h]["all"], by_horizon_errors[h]["live"])
            by_horizon.append({
                "horizon": h,
                "mae": round(mae_h, 2) if mae_h is not None else None,
                "n": n_h,
                "basis": basis_h,
            })

        with_data = [b for b in by_horizon if b["mae"] is not None]
        best = min(with_data, key=lambda b: b["mae"]) if with_data else None
        worst = max(with_data, key=lambda b: b["mae"]) if with_data else None

        by_model = []
        for model_used, d in sorted(by_model_errors.items()):
            mae_m, bias_m, n_m, basis_m = summarize_errors(d["all"], d["live"])
            by_model.append({
                "model_used": model_used,
                "mae": round(mae_m, 2) if mae_m is not None else None,
                "bias": round(bias_m, 2) if bias_m is not None else None,
                "n": n_m,
                "basis": basis_m,
                "n_backfill": d["n_backfill"],
                "n_live": d["n_live"],
            })

        return jsonify({
            "n_total": n_total,
            "n_matched": n_matched,
            "overall_mae": round(overall_mae, 2) if overall_mae is not None else None,
            "overall_bias": round(overall_bias, 2) if overall_bias is not None else None,
            "overall_n_used": overall_n_used,
            "overall_basis": overall_basis,
            "by_horizon": by_horizon,
            "by_model": by_model,
            "best_horizon": best,
            "worst_horizon": worst,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # DEBUG=True ทำให้ Flask สร้าง reloader process แยก ถ้าไม่กันไว้ scheduler จะเริ่มซ้ำ 2 รอบ
    # (WERKZEUG_RUN_MAIN ถูกตั้งเฉพาะใน process ลูกที่รันจริง ไม่ใช่ process reloader ตัวนอก
    #  ใช้ค่าคงที่ DEBUG แทน app.debug เพราะ Flask ยังไม่ set app.debug ก่อนเรียก app.run())
    if not DEBUG or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        scheduler = BackgroundScheduler()
        scheduler.add_job(
            scheduled_forecast_job,
            "interval",
            hours=1,
            id="hourly_forecast",
            next_run_time=pd.Timestamp.now(),  # ยิงทันทีตอนสตาร์ท แล้วค่อยทุก 1 ชม.หลังจากนั้น
        )
        scheduler.start()
        print("[scheduler] started - auto forecast every 1 hour")

    app.run(port=5000, debug=DEBUG)

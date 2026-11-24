import json
import os
import pickle
import random
import re
import sys
import threading
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
from pywebpush import WebPushException, webpush
from sklearn.metrics import r2_score


sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "scripts"))
from shap_analysis_clean import get_shap_explanation

# ===== Config =====

LGB_MODEL_PATH = "models/lgb_pm25_multiprovince_residual_bagged_quantile_20260919_203455.pkl"
XGB_MODEL_PATH = "models/xgb_pm25_multiprovince_residual_bagged_quantile_20260919_203455.pkl"
CONFORMAL_LGB_PATH = "models/conformal_calibration_lgb_20260919_203455.csv"
CONFORMAL_XGB_PATH = "models/conformal_calibration_xgb_20260919_203455.csv"

MODEL_LGB = "lightgbm_pooled_v3"
MODEL_XGB = "xgboost_pooled_v3"
MODEL_NAIVE = "naive_fallback"
PROVINCE_REGISTRY_PATH = "data/province_station_registry.json"

USE_TIMESTAMP_OFFSET = True
OFFSET_FUTURE_TOLERANCE_H = 1
# OpenAQ v3 คืน period.datetimeFrom (ต้นช่วง 1 ชม.) เป็นตัว datetime_utc ที่ build_feature_row()/
# fetch_recent_pm25_openaq() ใช้ทั้งหมด (คงไว้แบบนี้ต่อไป ห้ามเปลี่ยน — ต้องสม่ำเสมอกับตอนเทรนที่ใช้
# datetimeFrom เหมือนกัน) แต่ค่าของจุดนั้นเป็นค่าเฉลี่ยเคลื่อนที่ 24 ชม.ที่จบ ณ period.datetimeTo (=
# datetimeFrom + 1h) ซึ่งเป็นแบบแผนเดียวกับที่เว็บ explore.openaq.org ใช้ติดป้ายจุด (ยืนยันแล้ว: ค่าเดียวกัน
# ของเราติดป้าย 18:00 เว็บเขาติดป้าย 19:00 เป๊ะ ต่างกันคงที่ 1 ชม.) ค่านี้ใช้ "เฉพาะตอนแปลงเป็นข้อความเวลาโชว์
# ผู้ใช้" เท่านั้น (ดู build_hourly_timeline()/get_forecast()) ห้ามใช้ปรับ latest_time ที่ป้อนเข้า feature
# engineering/DB logging/backfill มิฉะนั้นจะเกิด train/serve skew
OPENAQ_DISPLAY_LABEL_SHIFT_H = 1
HIGH_POLLUTION_THRESHOLD = 37.5  # ค่าเดียวกันทุกจังหวัด (มาตรฐาน คพ. ไม่ใช่ค่าที่ผูกกับโมเดล)
# ต้องรัน `npm run build` ใน frontend/ ก่อน ถึงจะมีโฟลเดอร์นี้ 
FRONTEND_DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "dist")
DEBUG = True  # อ่านค่าคงที่นี้เสมอ ห้ามใช้ app.debug แทน เพราะ app.debug ยังไม่ถูก set จนกว่าจะเรียก app.run()
SIGMA_CRITICAL = 0.3
EXPLORATION_RATE = 0.2
MIN_R2_PAIRS = 15

CONFORMAL_ALPHA = 0.10  


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
SECRET_KEY = _ENV["SECRET_KEY"]  
OPENAQ_KEY = _ENV["OPENAQ_KEY"]  
OPENAQ_HEADERS = {"X-API-Key": OPENAQ_KEY}

FIRMS_KEY = _ENV.get("FIRMS_KEY")

VAPID_PRIVATE_KEY_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), _ENV["VAPID_PRIVATE_KEY_PATH"])
VAPID_PUBLIC_KEY = _ENV["VAPID_PUBLIC_KEY"]
VAPID_CLAIMS_SUB = _ENV["VAPID_CLAIMS_SUB"]


# PostgreSQL (ย้ายจาก sqlite แล้ว) — get_db_connection() เปิด connection ใหม่ทุกครั้งที่เรียก
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
app.config["SESSION_COOKIE_SECURE"] = False 
app.config["PERMANENT_SESSION_LIFETIME"] = timedelta(days=7)

# province registry (20 จังหวัด: 17 เทรน + 3 zero-shot — ดู scripts/province_registry.py)
with open(PROVINCE_REGISTRY_PATH, encoding="utf-8") as f:
    _registry = json.load(f)
PROVINCE_REGISTRY = _registry["provinces"]
PROVINCES_BY_SLUG = {p["province_en"]: p for p in PROVINCE_REGISTRY}
DEFAULT_PROVINCE = "khonkaen"
assert DEFAULT_PROVINCE in PROVINCES_BY_SLUG, f"{DEFAULT_PROVINCE} ต้องอยู่ใน province registry"

# pooled model: {horizon(1..24): {quantile(0.05|0.5|0.95): fitted_model}} เดียวกันทุกจังหวัด
with open(LGB_MODEL_PATH, "rb") as f:
    lgb_package = pickle.load(f)
with open(XGB_MODEL_PATH, "rb") as f:
    xgb_package = pickle.load(f)

MODELS = lgb_package["models"]
XGB_MODELS = xgb_package["models"]
FEATURE_COLS = lgb_package["feature_cols"]
assert FEATURE_COLS == xgb_package["feature_cols"], "feature_cols ของ LightGBM กับ XGBoost ต้องตรงกันเป๊ะ"
QUANTILES = lgb_package["quantiles"]  # [0.05, 0.5, 0.95]

USES_NEIGHBOR = any(c in FEATURE_COLS for c in ("pm25_neighbor_avg", "nb_trend_12h"))
USES_FIRE = "fire_hotspot_count_100km_24h" in FEATURE_COLS

_conformal_lgb_df = pd.read_csv(CONFORMAL_LGB_PATH)
_conformal_xgb_df = pd.read_csv(CONFORMAL_XGB_PATH)
CONFORMAL_QHAT = {
    MODEL_LGB: {int(r.horizon): float(r.q_hat) for r in _conformal_lgb_df.itertuples()},
    MODEL_XGB: {int(r.horizon): float(r.q_hat) for r in _conformal_xgb_df.itertuples()},
}

# ===== เติมข้อมูลของสถานีเป้าหมายที่เงียบ (Data Imputation, H-lite) =====

USE_TARGET_IMPUTATION = True
IMPUTATION_CONFIG_PATH = "models/imputation_h_lite_20260924_204956.json"
IMPUTATION_MIN_K = 3   
IMPUTATION_MAX_K = 48  

# ===== เครื่องมือ debug ถาวร: วัดเวลาแต่ละขั้นตอนของ get_forecast() (ใช้ตอนไล่ปัญหา "โหลดนาน" ดู
# CHANGELOG 222) ปิดเป็นค่าเริ่มต้น — เปิดเฉพาะตอนต้องการดู log [timing] ไล่คอขวดใหม่ =====
DEBUG_TIMING = False

# ===== ทางสำรองเมื่อดึง OpenAQ / Open-Meteo ไม่ได้  =====

USE_INPUT_FALLBACK = True
WEATHER_TTL_S = 600                 
WEATHER_CACHE_MAX_AGE_H = 2        
WEATHER_CLIMATOLOGY_PATH = "models/weather_climatology_20260924_211741.json"
TARGET_FALLBACK_MAX_AGE_S = 6 * 3600  
SERIES_DISK_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runtime_cache")

RISK_GROUP_THRESHOLD_OFFSET = {"ทั่วไป": 0.0, "เด็ก-ผู้สูงอายุ": -10.0, "มีโรคประจำตัว": -10.0}

ACTIVITY_INTENSITY_OFFSET = {"เบา": 0.0, "ปานกลาง": -5.0, "หนัก": -10.0}

RISK_SCORE_BREAKPOINTS = [(0, 0), (15, 20), (25, 40), (37.5, 60), (75, 80), (150, 100)]


def effective_threshold_for(risk_group):
    
    return HIGH_POLLUTION_THRESHOLD + RISK_GROUP_THRESHOLD_OFFSET.get(risk_group, 0.0)



def is_horizon_at_risk(predicted_pm25, risk_group):
    return predicted_pm25 > effective_threshold_for(risk_group)


def is_current_at_risk(latest_pm25, risk_group):

    return bool(latest_pm25 > effective_threshold_for(risk_group))


# ฟีเจอร์แจ้งเตือนเพิ่มเติม (ข้อ 2 "ใกล้ถึงช่วงเสี่ยงแล้ว") — แยกจาก risk_horizons เดิมที่บอกแค่ "มีช่วงเสี่ยง
# บางจุดใน 24 ชม." กว้างเกินไป อันนี้เจาะจงว่าช่วงเสี่ยงที่ใกล้ที่สุด "ใกล้แค่ไหน"
IMMINENT_HORIZON_H = 2


ACTIVE_HOURS_START_TH = 6
ACTIVE_HOURS_END_TH = 20


def find_best_time(forecast):

    active_hours = [f for f in forecast if ACTIVE_HOURS_START_TH <= f["hour_thai"] <= ACTIVE_HOURS_END_TH]
    return min(active_hours or forecast, key=lambda x: x["pm25"])


# ช่วงเวลาของวัน ใช้ display_hour_thai (เวลาที่ "ควรโชว์ผู้ใช้" ตาม OPENAQ_DISPLAY_LABEL_SHIFT_H ไม่ใช่ hour_thai
# ดิบ) — ใช้สำหรับสรุปแนวโน้มรายวันเท่านั้น ไม่เกี่ยวกับ ACTIVE_HOURS_START_TH/END_TH ด้านบนซึ่งเป็นคนละ convention
# (ช่วงเวลาที่ "ควรออกไปทำกิจกรรม" ไม่ใช่ช่วงเวลาของวัน)
DAILY_TREND_PERIODS = (
    ("เช้า", range(6, 12)),
    ("บ่าย", range(12, 18)),
    ("เย็น-กลางคืน", list(range(18, 24)) + list(range(0, 6))),
)


def classify_daily_trend(forecast, flat_threshold_ugm3=3.0):
    """แบ่ง forecast (24 จุด) ตามช่วงเวลาของวัน เทียบค่าเฉลี่ยแต่ละช่วงกับค่าเฉลี่ยของ "ช่วงเวลาที่เหลือทั้งหมด"
    (leave-one-out ไม่ใช่ค่าเฉลี่ยทั้งวันรวมตัวเอง) หาช่วงที่เบี่ยงเบนมากที่สุด — จงใจไม่เทียบกับค่าเฉลี่ยทั้งวัน
    เพราะช่วงเย็น-กลางคืนมี 12 ชม. พอๆ กับเช้า+บ่ายรวมกัน ถ้าเทียบกับค่าเฉลี่ยทั้งวัน (ซึ่งรวมตัวมันเองด้วยน้ำหนัก
    ครึ่งหนึ่ง) ช่วง 12 ชม. นี้จะไม่มีทางชนะช่วง 6 ชม. ได้เลยไม่ว่าค่าจะต่างกันแค่ไหน (พิสูจน์ด้วยพีชคณิต: ผลรวม
    ถ่วงน้ำหนักของค่าเบี่ยงเบนทุกจุดต้องเป็น 0 เสมอ ทำให้ magnitude ของกลุ่มที่มีน้ำหนักเท่ากันสองฝั่งเท่ากันเป๊ะ)
    คืน (direction, period_label): direction คือ "flat"/"up"/"down", period_label คือชื่อช่วงเวลาที่เบี่ยงเบน
    มากที่สุด (None ถ้า flat) — ใช้ทั้งใน prompt (บอกข้อเท็จจริงให้ Gemini) และ validator (เช็คว่า Gemini ไม่พูด
    สวนทาง) ต้องเรียกด้วยค่าเดียวกันทั้งสองที่"""

    deviations = []
    for label, hours in DAILY_TREND_PERIODS:
        in_bucket = [f["pm25"] for f in forecast if f["display_hour_thai"] in hours]
        out_bucket = [f["pm25"] for f in forecast if f["display_hour_thai"] not in hours]
        if not in_bucket or not out_bucket:
            continue
        deviations.append((label, (sum(in_bucket) / len(in_bucket)) - (sum(out_bucket) / len(out_bucket))))

    if not deviations:
        return "flat", None

    label, dev = max(deviations, key=lambda d: abs(d[1]))
    if abs(dev) < flat_threshold_ugm3:
        return "flat", None
    return ("up" if dev > 0 else "down"), label


def compute_risk_score(pm25, risk_group, activity):

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


# ===== Dynamic Routing  =====

def compute_missing_ratio(feature_row):
  
    n_missing = int(feature_row[FEATURE_COLS].isna().sum().sum())
    return n_missing / len(FEATURE_COLS)


def compute_rolling_r2(model_name, horizon, province, N=24):
  
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """SELECT predicted_pm25, actual_pm25
           FROM prediction_log
           WHERE model_used = %s AND horizon = %s AND province = %s AND actual_pm25 IS NOT NULL
           ORDER BY target_time_utc DESC
           LIMIT %s""",
        (model_name, horizon, province, N),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    if len(rows) < MIN_R2_PAIRS:
        return None

    y_pred = [r[0] for r in rows]
    y_true = [r[1] for r in rows]
    return r2_score(y_true, y_pred)


ROLLING_R2_MODELS = (MODEL_XGB, MODEL_LGB)


def fetch_rolling_r2_lookup(province, N=24):
   
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """SELECT model_used, horizon, predicted_pm25, actual_pm25
           FROM (
               SELECT model_used, horizon, predicted_pm25, actual_pm25,
                      ROW_NUMBER() OVER (PARTITION BY model_used, horizon ORDER BY target_time_utc DESC) AS rn
               FROM prediction_log
               WHERE province = %s AND actual_pm25 IS NOT NULL AND model_used = ANY(%s)
           ) sub
           WHERE rn <= %s""",
        (province, list(ROLLING_R2_MODELS), N),
    )
    rows = cur.fetchall()
    cur.close()
    conn.close()

    grouped = {}
    for model_used, horizon, predicted, actual in rows:
        grouped.setdefault((model_used, horizon), []).append((predicted, actual))

    lookup = {}
    for model_used in ROLLING_R2_MODELS:
        for h in range(1, 25):
            pairs = grouped.get((model_used, h), [])
            if len(pairs) < MIN_R2_PAIRS:
                lookup[(model_used, h)] = None
            else:
                y_pred = [p[0] for p in pairs]
                y_true = [p[1] for p in pairs]
                lookup[(model_used, h)] = r2_score(y_true, y_pred)
    return lookup


def select_model(feature_row, horizon, province, r2_lookup, force_naive=False):
    
    if force_naive or compute_missing_ratio(feature_row) > SIGMA_CRITICAL:
        return MODEL_NAIVE

    r2_xgb = r2_lookup.get((MODEL_XGB, horizon))
    r2_lgb = r2_lookup.get((MODEL_LGB, horizon))

    if r2_xgb is None and r2_lgb is None:
        return random.choice([MODEL_XGB, MODEL_LGB])
    elif r2_xgb is None:
        return MODEL_XGB if random.random() < EXPLORATION_RATE else MODEL_LGB
    elif r2_lgb is None:
        return MODEL_LGB if random.random() < EXPLORATION_RATE else MODEL_XGB
    else:
        return MODEL_XGB if r2_xgb > r2_lgb else MODEL_LGB


# ===== Database schema =====
# init_db() มีแค่ schema พื้นฐาน — คอลัมน์ที่เพิ่มทีหลัง (role, home_province, province, pm25_lo/hi)
# อยู่ในฟังก์ชัน migrate_* ด้านล่าง ซึ่งรัน ALTER TABLE ... ADD COLUMN IF NOT EXISTS ทุกครั้งที่แอปเริ่ม
# ทำงาน (idempotent) จำเป็นต้องแยกแบบนี้เพราะ DB จริงที่ deploy อยู่แล้วมีข้อมูลสะสม รวม column เข้า
# CREATE TABLE อย่างเดียวจะไม่มีผลกับตารางที่มีอยู่แล้ว (CREATE TABLE IF NOT EXISTS ข้ามไปเลย)

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

    # NLG รายวัน (สร้าง 1 ครั้ง/วัน ตอน 06:00 น. เวลาไทย)
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

    # ประวัติค่าจริงล่าสุดต่อรอบ (/api/notifications) — เก็บแค่ latest_pm25, risk_horizons คำนวณสด
    # ตอน query โดย join กับ prediction_log ผ่าน recorded_at_utc=predicted_at_utc แทน (ไม่เก็บตรงๆเพราะ risk ขึ้นกับ risk_group ที่เปลี่ยนได้ เก็บสดจะผิดสำหรับกลุ่มอื่น)
    cur.execute("""
        CREATE TABLE IF NOT EXISTS notification_log (
            id SERIAL PRIMARY KEY,
            recorded_at_utc TEXT NOT NULL UNIQUE,
            latest_pm25 DOUBLE PRECISION NOT NULL
        )
    """)

    # risk_group เก็บ Thai string ตรงๆ (คีย์เดียวกับ _USER_GROUP_TONE) ผูกกับโทน NLG รายวันและ
    # alert threshold (RISK_GROUP_THRESHOLD_OFFSET, effective_threshold_for())
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

    # Web Push subscription — endpoint คือ identity ตาม Web Push spec (unique) user_id เป็น null ได้
    # (guest, สอดคล้องกับ /api/notifications ที่ไม่บังคับ login) province ถูก pin ตอน subscribe ไม่ sync
    # ตาม home_province อัตโนมัติ (guest ไม่มี home_province ให้ sync ตั้งแต่แรก) push_dedup_state เก็บ
    # ค่าล่าสุดที่ push ไปแล้วต่อเงื่อนไข (threshold/transition/imminent/trend/silence) กันแจ้งซ้ำทุกรอบ scheduler
    cur.execute("""
        CREATE TABLE IF NOT EXISTS push_subscriptions (
            id SERIAL PRIMARY KEY,
            endpoint TEXT NOT NULL UNIQUE,
            p256dh TEXT NOT NULL,
            auth TEXT NOT NULL,
            user_id INTEGER REFERENCES users(id) ON DELETE CASCADE,
            province TEXT NOT NULL,
            created_at TEXT NOT NULL,
            push_dedup_state JSONB NOT NULL DEFAULT '{}'::jsonb
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
        "role": row[5],
        "home_province": row[6],
    }


def get_user_by_email(email):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, email, risk_group, age, created_at, role, home_province FROM users WHERE email = %s",
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
        "SELECT id, email, risk_group, age, created_at, role, home_province, password_hash FROM users WHERE email = %s",
        (email,),
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    if row is None:
        return None
    return {**_public_user(row[:7]), "password_hash": row[7]}


def get_user_by_id(user_id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, email, risk_group, age, created_at, role, home_province FROM users WHERE id = %s",
        (user_id,),
    )
    row = cur.fetchone()
    cur.close()
    conn.close()
    return _public_user(row) if row else None


def create_user(email, password_hash, risk_group, age, home_province=None):
    conn = get_db_connection()
    cur = conn.cursor()
    created_at = pd.Timestamp.now(tz="UTC").isoformat()
    cur.execute(
        """INSERT INTO users (email, password_hash, risk_group, age, created_at, home_province)
           VALUES (%s, %s, %s, %s, %s, %s) RETURNING id""",
        (email, password_hash, risk_group, age, created_at, home_province),
    )
    user_id = cur.fetchone()[0]
    conn.commit()
    cur.close()
    conn.close()
    return user_id


def update_user_profile(user_id, email, risk_group, age, home_province=None):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "UPDATE users SET email = %s, risk_group = %s, age = %s, home_province = %s WHERE id = %s",
        (email, risk_group, age, home_province, user_id),
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


# ===== Live data fetching (OpenAQ + Open-Meteo) =====

class _CircuitBreaker:


    def __init__(self, name, threshold, cooldown_s):
        self.name, self.threshold, self.cooldown_s = name, threshold, cooldown_s
        self._fails, self._open_until = 0, 0.0

    def allow(self):
        return time.monotonic() >= self._open_until

    def success(self):
        self._fails = 0

    def failure(self):
        self._fails += 1
        if self._fails >= self.threshold:
            self._open_until = time.monotonic() + self.cooldown_s
            self._fails = 0
            print(f"[{self.name}] ล้มเหลวต่อเนื่อง {self.threshold} ครั้ง — พักการเรียก {self.cooldown_s}s ใช้ cache/ทางสำรองแทน")


_OPENAQ_BREAKER = _CircuitBreaker("openaq", threshold=5, cooldown_s=60)
_WEATHER_BREAKER = _CircuitBreaker("open-meteo", threshold=5, cooldown_s=60)


def _openaq_get(kind, sensor_id, params, attempts, timeout=15):
    """GET /v3/sensors/{id}/{kind} พร้อม retry แบบ backoff เฉพาะ 429/5xx/network — 4xx อื่น raise ทันที; retry หมด -> RuntimeError"""
    _t0 = time.perf_counter() if DEBUG_TIMING else None
    for attempt in range(attempts):
        try:
            r = requests.get(f"https://api.openaq.org/v3/sensors/{sensor_id}/{kind}",
                              headers=OPENAQ_HEADERS, params=params, timeout=timeout)
        except requests.exceptions.RequestException:
            r = None
        if r is not None:
            if not (r.status_code == 429 or r.status_code >= 500):
                r.raise_for_status()
                if DEBUG_TIMING and attempt > 0:
                    print(f"[timing] _openaq_get {kind} sensor {sensor_id}: สำเร็จหลัง retry {attempt} ครั้ง "
                          f"({time.perf_counter() - _t0:.2f}s)")
                return r
        if attempt < attempts - 1:
            time.sleep(2 ** attempt)
    if DEBUG_TIMING:
        print(f"[timing] _openaq_get {kind} sensor {sensor_id}: ล้มเหลวทั้งหมด {attempts} ครั้ง "
              f"({time.perf_counter() - _t0:.2f}s)")
    raise RuntimeError(f"ดึง pm25 จาก OpenAQ ({kind}) ไม่สำเร็จหลัง retry (sensor {sensor_id})")


# ===== ตรวจสอบคุณภาพข้อมูลนำเข้าแบบ real-time (กรองค่า PM2.5 ผิดปกติจากเซนเซอร์) =====
# False = เหมือนพฤติกรรมเดิมเป๊ะ (dropna เฉพาะ null เหมือนเดิม ไม่เช็คค่าสมเหตุสมผล) — ดู scripts/test_outlier_filter.py
USE_OUTLIER_FILTER = True

PM25_HARD_MIN = 0.0
PM25_HARD_MAX = 1500.0  # เพดาน sentinel (ดักค่าพัง เช่น -999/9999) สูงกว่าค่าสูงสุดจริงที่เคยเจอหลายเท่า
                         # (226.4 µg/m³ ที่ buengkan 2026-04-17 05:00 UTC — เหตุการณ์หมอกควันรุนแรงจริง ไม่ใช่ sensor
                         # glitch) ไม่ใช่เพดาน "ค่าฝุ่นสูงสุดที่น่าเชื่อ" — เหตุการณ์จริงต้องผ่านชั้นนี้ได้เสมอ
PM25_SPIKE_REVERT_THRESHOLD = 15.0  # sweep จริงกับข้อมูล 20 จังหวัด 363,306 แถว (multiprovince_pm25_features_raw_
                                     # 20260919_172201.csv): threshold 12-25 ให้ผลเหมือนกันเป๊ะ (เจอแค่ 2 จุด,
                                     # khonkaen 2024-06-14 03:00 และ 2024-08-28 05:00 รูปแบบ ~18->~55->~17 ดูเหมือน
                                     # sensor glitch ทั้งคู่) ไม่แตะเหตุการณ์หมอกควันจริงที่ buengkan เลย (226.4,
                                     # ค่อยๆ ลดลงต่อเนื่อง ไม่ revert) — เลือก 15 เพราะอยู่กลาง plateau ที่เสถียร
PM25_ANCHOR_JUMP_MULTIPLIER = 8.0
PM25_ANCHOR_JUMP_FLOOR = 150.0
_OUTLIER_NEIGHBOR_MAX_GAP_H = 2.0


def _filter_pm25_outliers(df):
    """กรองค่า PM2.5 ที่ผิดปกติจากเซนเซอร์ก่อนเข้ากระบวนการพยากรณ์ — ตั้งค่าที่ตัดสินว่าผิดปกติเป็น NaN (ผู้เรียก
    dropna ทิ้งเองต่อ ไม่ลบแถวตรงนี้โดยตรง) ออกแบบอนุรักษ์นิยม (เอนเอียงไปทาง "ปล่อยผ่านเมื่อไม่แน่ใจ") เพราะเป็น
    health-alert system — กรองเหตุการณ์ฝุ่นพุ่งสูงจริงทิ้งเสียหายกว่าปล่อยให้ sensor glitch หลุดเข้ามาบ้าง

    3 ชั้น เรียงจากเข้มงวดน้อยไปมาก:
    1) hard bound (ทุกแถว รวมแถวล่าสุด): pm25 นอกช่วง [PM25_HARD_MIN, PM25_HARD_MAX]
    2) isolated spike-revert (เฉพาะแถวกลาง ไม่ใช่แถวแรก/แถวล่าสุด, มีเพื่อนบ้านทั้งสองข้างห่าง ≤2 ชม.): ต่างจากทั้ง
       แถวก่อนหน้าและแถวถัดไปเกิน threshold แต่แถวก่อนหน้า/ถัดไปใกล้เคียงกันเอง (แยก "จุดเดียวกระโดดแล้วกลับคืน"
       ออกจากเหตุการณ์จริงที่ค่อยๆ เปลี่ยนแปลงต่อเนื่องหลายชั่วโมง ซึ่งจะไม่ "กลับคืน" ในชั่วโมงถัดไป)
    3) anchor one-sided jump (เฉพาะแถวล่าสุด เพราะไม่มีแถวถัดไปให้ยืนยันแบบข้อ 2): กระโดดสุดโต่งจากฐานปกติเท่านั้น
       (prev ต่ำกว่า floor ด้วย — ถ้า prev เองก็สูงอยู่แล้วถือเป็นเหตุการณ์ต่อเนื่อง ไม่ใช่ sensor error)

    ข้อจำกัดที่ทราบและยอมรับ: ค่าที่เพี้ยนแต่ "ดูสมเหตุสมผลในตัวเอง" (เช่น sensor ค้างค่าเดิมซ้ำ หรือคลาดเคลื่อน
    ปานกลางที่ไม่โดดจากบริบท) จะหลุดผ่านไปได้เสมอถ้าไม่มีจุดข้อมูลที่สองมายืนยัน — ข้อจำกัดโดยธรรมชาติ ไม่ใช่บั๊ก"""
    df = df.copy()
    pm25 = df["pm25"].to_numpy(dtype=float)
    n = len(df)
    flagged = (pm25 < PM25_HARD_MIN) | (pm25 > PM25_HARD_MAX)

    if n >= 3:
        gap_h = df["datetime_utc"].diff().dt.total_seconds().to_numpy() / 3600.0
        for i in range(1, n - 1):
            if flagged[i] or gap_h[i] > _OUTLIER_NEIGHBOR_MAX_GAP_H or gap_h[i + 1] > _OUTLIER_NEIGHBOR_MAX_GAP_H:
                continue
            prev_v, cur_v, next_v = pm25[i - 1], pm25[i], pm25[i + 1]
            if (abs(cur_v - prev_v) > PM25_SPIKE_REVERT_THRESHOLD and
                    abs(cur_v - next_v) > PM25_SPIKE_REVERT_THRESHOLD and
                    abs(prev_v - next_v) <= PM25_SPIKE_REVERT_THRESHOLD):
                flagged[i] = True

    if n >= 2 and not flagged[n - 1]:
        gap_last_h = (df["datetime_utc"].iloc[-1] - df["datetime_utc"].iloc[-2]).total_seconds() / 3600.0
        if gap_last_h <= _OUTLIER_NEIGHBOR_MAX_GAP_H:
            prev_v, last_v = pm25[n - 2], pm25[n - 1]
            if (prev_v < PM25_ANCHOR_JUMP_FLOOR and
                    last_v > max(PM25_ANCHOR_JUMP_FLOOR, PM25_ANCHOR_JUMP_MULTIPLIER * prev_v)):
                flagged[n - 1] = True

    n_flagged = int(flagged.sum())
    if n_flagged:
        print(f"[openaq] กรองค่า PM2.5 ผิดปกติทิ้ง {n_flagged} จุด จาก {n} แถว")
        df.loc[flagged, "pm25"] = np.nan
    return df


def fetch_recent_pm25_openaq(sensor_id, hours_back=72, offset_hours=0):

    end = pd.Timestamp.now(tz="UTC")
    offset = pd.Timedelta(hours=offset_hours)
    start = end - pd.Timedelta(hours=hours_back)
    params = {"datetime_from": (start - offset).isoformat(), "datetime_to": end.isoformat(), "limit": 1000}

    r = _openaq_get("measurements", sensor_id, params, attempts=5)

    results = r.json().get("results", [])
    df = pd.DataFrame([{"datetime_utc": rr["period"]["datetimeFrom"]["utc"], "pm25": rr["value"]} for rr in results])
    if df.empty:
        return df
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df = df.dropna(subset=["pm25"]).drop_duplicates(subset=["datetime_utc"]).sort_values("datetime_utc").reset_index(drop=True)
    if USE_OUTLIER_FILTER:
        df = _filter_pm25_outliers(df).dropna(subset=["pm25"]).reset_index(drop=True)
    if offset_hours and not df.empty:
        shifted_max = df["datetime_utc"].max() + offset
        if shifted_max > end + pd.Timedelta(hours=OFFSET_FUTURE_TOLERANCE_H):
            print(f"[openaq] sensor {sensor_id}: หลังบวก offset {offset_hours}h ข้อมูลล้ำอนาคต -> OpenAQ น่าจะแก้ป้ายเวลาแล้ว ทิ้ง offset "
                  f"(รัน scripts/annotate_registry_timestamp_offset.py ใหม่)")
        else:
            df["datetime_utc"] = df["datetime_utc"] + offset
    return df


def fetch_recent_pm25_openaq_hours(sensor_id, hours_back=72):
    
    end = pd.Timestamp.now(tz="UTC")
    params = {"datetime_from": (end - pd.Timedelta(hours=hours_back)).isoformat(), "datetime_to": end.isoformat(), "limit": 1000}
    r = _openaq_get("hours", sensor_id, params, attempts=3)
    results = r.json().get("results", [])
    df = pd.DataFrame([{"datetime_utc": rr["period"]["datetimeFrom"]["utc"], "pm25": rr["value"]} for rr in results])
    if df.empty:
        return df
    df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
    df = df.dropna(subset=["pm25"]).drop_duplicates(subset=["datetime_utc"]).sort_values("datetime_utc").reset_index(drop=True)
    if USE_OUTLIER_FILTER:
        df = _filter_pm25_outliers(df).dropna(subset=["pm25"]).reset_index(drop=True)
    return df


WEATHER_CURRENT_VARS = "temperature_2m,relative_humidity_2m,wind_speed_10m,precipitation,wind_direction_10m,surface_pressure"
WEATHER_REQUIRED_KEYS = ("temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation", "wind_direction_10m", "surface_pressure")


def _open_meteo_get(params, attempts, timeout):
    """GET Open-Meteo forecast API พร้อม retry backoff เฉพาะ 429/5xx/network (ไม่มี key ในคำขอ จึงไม่มีอะไรต้องปิดบังใน log)"""
    for attempt in range(attempts):
        try:
            r = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=timeout)
        except requests.exceptions.RequestException:
            r = None
        if r is not None:
            if not (r.status_code == 429 or r.status_code >= 500):
                r.raise_for_status()
                return r.json()
        if attempt < attempts - 1:
            time.sleep(2 ** attempt)
    raise RuntimeError("ดึงสภาพอากาศจาก Open-Meteo ไม่สำเร็จหลัง retry")


# ช่วงที่เป็นไปได้ทางฟิสิกส์ กว้างกว่าสภาพอากาศจริงของอีสานมาก (จงใจ) — เป้าหมายคือดักค่า API พัง (เช่น -999/9999)
# ไม่ใช่ตรวจจับสภาพอากาศผิดปกติทางอุตุนิยมวิทยา ส่วนหนึ่งของ USE_OUTLIER_FILTER (ดู fetch_recent_pm25_openaq ด้านบน)
WEATHER_PLAUSIBLE_RANGES = {
    "relative_humidity_2m": (0, 100),
    "wind_speed_10m": (0, 300),
    "precipitation": (0, 300),
    "surface_pressure": (850, 1050),
    "temperature_2m": (-10, 55),
}


def _check_weather(w):

    if not isinstance(w, dict) or any(w.get(k) is None for k in WEATHER_REQUIRED_KEYS):
        raise ValueError("ข้อมูลสภาพอากาศจาก Open-Meteo ไม่ครบ")
    if USE_OUTLIER_FILTER:
        for key, (lo, hi) in WEATHER_PLAUSIBLE_RANGES.items():
            v = w.get(key)
            if v is not None and not (lo <= v <= hi):
                raise ValueError(f"ค่าสภาพอากาศผิดปกติ: {key}={v} (ช่วงที่เป็นไปได้ {lo}-{hi})")


def fetch_current_weather_openmeteo(lat, lon):

    params = {"latitude": lat, "longitude": lon, "current": WEATHER_CURRENT_VARS, "timezone": "UTC"}
    return _open_meteo_get(params, attempts=2 if USE_INPUT_FALLBACK else 1, timeout=10 if USE_INPUT_FALLBACK else 15)["current"]


WEATHER_HOURLY_VARS = "temperature_2m,precipitation_probability,weather_code"


def fetch_hourly_weather_openmeteo(lat, lon, past_days=1, forecast_days=2):
    """ดึงสภาพอากาศรายชั่วโมง (อดีต past_days วัน + อนาคต forecast_days วัน) จาก Open-Meteo คนละคำขอกับ
    fetch_current_weather_openmeteo() (ซึ่งดึงแค่ค่าปัจจุบันจุดเดียวไว้เป็น feature โมเดล) — ใช้ประกอบกราฟ
    'พยากรณ์อากาศ' รายชั่วโมงฝั่ง frontend เท่านั้น ไม่เกี่ยวกับ pipeline การพยากรณ์ PM2.5"""
    params = {
        "latitude": lat, "longitude": lon, "hourly": WEATHER_HOURLY_VARS,
        "past_days": past_days, "forecast_days": forecast_days, "timezone": "UTC",
    }
    return _open_meteo_get(params, attempts=2 if USE_INPUT_FALLBACK else 1, timeout=10 if USE_INPUT_FALLBACK else 15)["hourly"]


def fetch_current_weather_batch(provinces):

    params = {"latitude": ",".join(str(p["lat"]) for p in provinces), "longitude": ",".join(str(p["lon"]) for p in provinces),
              "current": WEATHER_CURRENT_VARS, "timezone": "UTC"}
    data = _open_meteo_get(params, attempts=3, timeout=20)
    if isinstance(data, dict):
        data = [data]
    if len(data) != len(provinces):
        raise ValueError("จำนวนผลลัพธ์ batch ไม่ตรงกับจำนวนจังหวัด")
    out = {}
    for prov, item in zip(provinces, data):
        _check_weather(item["current"])
        out[prov["province_en"]] = item["current"]
    return out


def _load_weather_climatology():
    
    if not USE_INPUT_FALLBACK:
        return None
    try:
        with open(WEATHER_CLIMATOLOGY_PATH, encoding="utf-8") as f:
            cfg = json.load(f)
        if cfg["features"] != ["temp", "humidity", "wind_speed", "precipitation", "wind_dir_sin", "wind_dir_cos", "pressure"]:
            raise ValueError("ลำดับ feature ไม่ตรง")
        missing = [p["province_en"] for p in PROVINCE_REGISTRY if p["province_en"] not in cfg["table"]]
        if missing:
            raise ValueError(f"ไม่มีจังหวัด {missing}")
        return cfg
    except Exception as e:
        print(f"[weather] ปิด climatology สำรอง ({type(e).__name__}: {e})")
        return None


WEATHER_CLIM = _load_weather_climatology()
_WEATHER_CACHE = {}  # province_en -> (เวลา monotonic ที่ดึงสำเร็จ, dict current weather)
_WEATHER_LOCK = threading.Lock()

HOURLY_WEATHER_TTL_S = 30 * 60        # ข้อมูลรายชั่วโมงไม่ต้องสดเท่า current weather — กันยิง Open-Meteo ซ้ำ
                                       # ทุก request (เดิมไม่มี cache เลย เป็นจุดที่กินโควตารายวันมากสุด)
HOURLY_WEATHER_STALE_MAX_AGE_H = 24   # ดึงสดไม่ได้เลย (เช่น โควตารายวันหมด) ใช้ของเก่าได้ไม่เกินนี้ กันกราฟว่าง
_HOURLY_WEATHER_CACHE = {}  # province_en -> (เวลา monotonic ที่ดึงสำเร็จ, dict hourly จาก fetch_hourly_weather_openmeteo)


def weather_climatology(province, at=None):
   
    if WEATHER_CLIM is None:
        return None
    t = pd.Timestamp.now(tz="UTC") if at is None else pd.Timestamp(at)
    row = WEATHER_CLIM["table"][province][t.month - 1][t.hour]
    d = dict(zip(WEATHER_CLIM["features"], row))
    return {
        "temperature_2m": d["temp"], "relative_humidity_2m": d["humidity"], "wind_speed_10m": d["wind_speed"],
        "precipitation": d["precipitation"], "surface_pressure": d["pressure"],
        "wind_direction_10m": float(np.degrees(np.arctan2(d["wind_dir_sin"], d["wind_dir_cos"])) % 360),
        "_wind_dir_sin": d["wind_dir_sin"], "_wind_dir_cos": d["wind_dir_cos"],
    }


def prefetch_weather_all():

    if not USE_INPUT_FALLBACK or not _WEATHER_BREAKER.allow():
        return
    try:
        data = fetch_current_weather_batch(PROVINCE_REGISTRY)
    except Exception as e:
        _WEATHER_BREAKER.failure()
        print(f"[weather] ดึงแบบ batch ไม่สำเร็จ ({type(e).__name__}) — แต่ละจังหวัดจะลองดึงเอง")
        return
    _WEATHER_BREAKER.success()
    now = time.monotonic()
    with _WEATHER_LOCK:
        for prov, w in data.items():
            _WEATHER_CACHE[prov] = (now, w)


def get_current_weather(province_meta, at=None):

    prov = province_meta["province_en"]
    now = time.monotonic()
    with _WEATHER_LOCK:
        hit = _WEATHER_CACHE.get(prov)
    if USE_INPUT_FALLBACK and hit is not None and now - hit[0] < WEATHER_TTL_S:
        return hit[1], {"source": "live", "age_hours": round((now - hit[0]) / 3600, 2)}
    err = None
    if USE_INPUT_FALLBACK and not _WEATHER_BREAKER.allow():
        err = RuntimeError("Open-Meteo circuit open")
    else:
        try:
            w = fetch_current_weather_openmeteo(province_meta["lat"], province_meta["lon"])
            if USE_INPUT_FALLBACK:
                _check_weather(w)
                _WEATHER_BREAKER.success()
                with _WEATHER_LOCK:
                    _WEATHER_CACHE[prov] = (time.monotonic(), w)
            return w, {"source": "live", "age_hours": 0.0}
        except Exception as e:
            err = e
            if USE_INPUT_FALLBACK:
                _WEATHER_BREAKER.failure()
    if not USE_INPUT_FALLBACK:
        raise err
    age_s = time.monotonic() - hit[0] if hit is not None else None
    if hit is not None and age_s <= WEATHER_CACHE_MAX_AGE_H * 3600:
        print(f"[weather] {prov}: ดึงสด ไม่ได้ ({type(err).__name__}) — ใช้ค่าที่ cache ไว้ {age_s / 60:.0f} นาทีที่แล้ว")
        return hit[1], {"source": "stale_cache", "age_hours": round(age_s / 3600, 2)}
    clim = weather_climatology(prov, at)
    if clim is not None:
        print(f"[weather] {prov}: ดึงสด ไม่ได้ ({type(err).__name__}) และไม่มีค่า cache ที่ใหม่พอ — ใช้ climatology (ไม่บันทึก log)")
        return clim, {"source": "climatology", "age_hours": None}
    raise err


# ===== Live features ของโมเดล =====


# --- ความสดของข้อมูลเพื่อนบ้าน (ใช้เป็น feature) ---
NEIGHBOR_FRESH_HOURS = 3       # เพื่อนบ้านนับว่า "สด" ถ้า reading ล่าสุดห่างจากเวลาจังหวัดเป้าหมายไม่เกิน 3 ชม.
NEIGHBOR_AVG_K = 3             # ถ้าเพื่อนบ้านอันดับ 1 ไม่สด ให้เฉลี่ยเพื่อนบ้านที่สดที่สุด 3 จังหวัดแทน
NEIGHBOR_MAX_FETCH = 8         # ดึงข้อมูลเพื่อนบ้านสูงสุด 8 จังหวัดต่อรอบ (เรียงตามระยะทางใกล้สุดก่อน)

# --- ความสดของข้อมูลจังหวัดเป้าหมาย ---
TARGET_STALE_HOURS = 6         # reading ล่าสุดเก่ากว่า 6 ชม. = ข้อมูลเก่า (ไม่บันทึก log, ไม่ทำ NLG)
TARGET_CACHE_TTL_S = 300       # cache สั้น 5 นาที เพราะค่า "ปัจจุบัน" ต้องสดเสมอ
SERIES_CACHE_TTL_S = 1800      # cache เพื่อนบ้าน 30 นาที (พอสำหรับ job รายชั่วโมงที่วน 20 จังหวัด)
SERIES_CACHE_MAX_AGE_S = 6 * 3600  # ดึงเพื่อนบ้านไม่สำเร็จ ใช้ cache เก่าได้สูงสุด 6 ชม.

# --- จุดความร้อนไฟป่า (FIRMS) ---
FIRE_CACHE_TTL_S = 3600
FIRE_CACHE_MAX_AGE_S = 6 * 3600
FIRE_RADIUS_KM = 100            # นับจุดความร้อนในรัศมี 100 กม. จากจังหวัดเป้าหมาย
FIRE_WINDOW_HOURS = 24          # นับย้อนหลัง 24 ชม.
FIRMS_AREA = "100.5,13.5,106.5,19.0"  # bbox ภาคอีสาน + กันชนลาว/กัมพูชา
FIRMS_NRT_SOURCE = "VIIRS_NOAA20_NRT"  # แหล่งข้อมูล live 


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


# ===== Air4Thai (PCD) — ค่า PM2.5 "ตอนนี้" ที่สดกว่า OpenAQ มาก (วัดจริง ~28 นาที vs ~2.5 ชม. ที่สถานี
# ขอนแก่นเดียวกัน ช่วงเวลาเดียวกัน) ใช้แสดงผลอย่างเดียว ("ตอนนี้"/จุด "ตอนนี้" บนกราฟ) ห้ามใช้แทน latest_pm25
# ที่เป็นฐานพยากรณ์ของโมเดล เพราะ Air4Thai อาจสเกลไม่ตรงกับที่โมเดลเทรนบน OpenAQ เป๊ะ (โปรเจกต์นี้เคยมีชั้น
# calibration Air4Thai<->OpenAQ มาก่อนตอนยุค Khon Kaen เดี่ยว แล้วเลิกใช้ไปตอนย้ายมา OpenAQ ล้วนทั้ง pipeline
# — ดู "Historical note" ใน CLAUDE.md — จึงไม่เอากลับมาผสมกับฐานพยากรณ์อีก)
AIR4THAI_TTL_S = 10 * 60  # คำขอเดียวได้ทุกสถานีทั่วประเทศ ไม่ต้องรีเฟรชถี่ (ฝั่ง Air4Thai เองอัปเดตรายชั่วโมง)
_AIR4THAI_BREAKER = _CircuitBreaker("air4thai", threshold=5, cooldown_s=60)
_AIR4THAI_CACHE = None  # (เวลา monotonic ที่ดึงสำเร็จล่าสุด, {province_en: {"pm25":, "datetime_utc":, "station_id":}})
_AIR4THAI_LOCK = threading.Lock()


def fetch_air4thai_live_all():
    """ดึงสถานี Air4Thai (PCD) ทั้งหมดทั่วประเทศในคำขอเดียว (getNewAQI_JSON.php, คนละ endpoint กับ
    getHistoryData.php ที่ scripts/archive_air4thai.py ใช้) แล้วจับคู่แต่ละจังหวัดใน PROVINCE_REGISTRY กับ
    สถานีที่ใกล้ที่สุด — ตรวจแล้วทั้ง 20 จังหวัดอีสานมีสถานี Air4Thai ที่พิกัดตรงกันเป๊ะ (0.0 กม. ห่าง แปลว่า
    ทะเบียนจังหวัดนี้น่าจะสร้างจากพิกัดสถานี Air4Thai ชุดเดียวกันมาตั้งแต่ต้น) คืน dict ว่างถ้าดึงไม่สำเร็จ
    (ไม่ raise — ให้ caller fallback ไป OpenAQ ที่มีอยู่แล้วเสมอ)"""
    r = requests.get("https://air4thai.pcd.go.th/services/getNewAQI_JSON.php",
                      headers={"User-Agent": "Mozilla/5.0"}, timeout=15, verify=False)
    r.raise_for_status()
    stations = r.json()["stations"]

    out = {}
    for p in PROVINCE_REGISTRY:
        best = None
        for s in stations:
            try:
                slat, slon = float(s["lat"]), float(s["long"])
            except (TypeError, ValueError):
                continue
            d = haversine_km(p["lat"], p["lon"], slat, slon)
            if best is None or d < best[0]:
                best = (d, s)
        if best is None:
            continue
        _, s = best
        aqi = s.get("AQILast", {})
        try:
            pm25 = float(aqi.get("PM25", {}).get("value"))
        except (TypeError, ValueError):
            continue
        if pm25 < 0:  # Air4Thai ใช้ -1 แทน "ไม่มีค่า" (เซนเซอร์เสีย/ยังไม่รายงาน)
            continue
        try:
            dt_th = pd.Timestamp(f"{aqi['date']} {aqi['time']}").tz_localize("Asia/Bangkok")
        except Exception:
            continue
        out[p["province_en"]] = {"pm25": pm25, "datetime_utc": dt_th.tz_convert("UTC"), "station_id": s["stationID"]}
    return out


def get_air4thai_live(province):
    """คืน {"pm25":, "datetime_utc":} ของจังหวัดนี้จากแคช Air4Thai (รีเฟรชทุก AIR4THAI_TTL_S) หรือ None ถ้า
    ดึงไม่สำเร็จ/ยังไม่มีแคชเลย/จังหวัดนี้ไม่มีสถานีคู่ — ไม่ throw ไม่ว่ากรณีใด เพราะนี่เป็นแค่ทางเลือกเสริมของ
    การแสดงผล "ตอนนี้" ให้สดขึ้น ไม่ใช่ critical path (พลาดแล้ว caller ใช้ OpenAQ เดิมต่อได้ปกติ)"""
    global _AIR4THAI_CACHE
    now = time.monotonic()
    with _AIR4THAI_LOCK:
        cached = _AIR4THAI_CACHE
    if cached is not None and now - cached[0] < AIR4THAI_TTL_S:
        return cached[1].get(province)
    if not _AIR4THAI_BREAKER.allow():
        return cached[1].get(province) if cached else None
    try:
        data = fetch_air4thai_live_all()
    except Exception as e:
        _AIR4THAI_BREAKER.failure()
        print(f"[air4thai] ดึงไม่สำเร็จ ({type(e).__name__}) — ใช้ OpenAQ แทนสำหรับค่า \"ตอนนี้\"")
        return cached[1].get(province) if cached else None
    _AIR4THAI_BREAKER.success()
    with _AIR4THAI_LOCK:
        _AIR4THAI_CACHE = (now, data)
    return data.get(province)


# เพื่อนบ้านเรียงตามระยะทางจริง
NEIGHBOR_RANKING = {
    p["province_en"]: [
        q["province_en"] for q in sorted(
            (q for q in PROVINCE_REGISTRY if q["province_en"] != p["province_en"]),
            key=lambda q: haversine_km(p["lat"], p["lon"], q["lat"], q["lon"]),
        )
    ]
    for p in PROVINCE_REGISTRY
}

_SERIES_CACHE = {}  # province_en -> (เวลา monotonic ที่ดึง, DataFrame[datetime_utc, pm25])
_SERIES_LOCK = threading.Lock()

# เรียกข้อมูลสดหลายจุดของ get_forecast() พร้อมกันแทนทีละจุด (target/weather/fire/r2/air4thai ไม่ขึ้นกับกันเลย
# วัดแล้วว่าเวลารวมตอนสลับจังหวัดไปจังหวัดที่ cache เย็นส่วนใหญ่มาจากการต่อคิวนี้ (วัดจริงด้วย timing log ชั่วคราว
# ดู CHANGELOG 222) เลียน pattern _gemini_executor ที่มีอยู่แล้ว
_IO_EXECUTOR = ThreadPoolExecutor(max_workers=8, thread_name_prefix="forecast-io")


def _series_disk_path(province):
    return os.path.join(SERIES_DISK_CACHE_DIR, f"openaq_series_{province}.csv")


def _save_series_to_disk(province, df):

    if not USE_INPUT_FALLBACK or df is None:
        return
    try:
        os.makedirs(SERIES_DISK_CACHE_DIR, exist_ok=True)
        path = _series_disk_path(province)
        tmp = path + ".tmp"
        df.to_csv(tmp, index=False)
        os.replace(tmp, path)
    except Exception:
        pass


def _load_series_disk_cache():
    
    if not USE_INPUT_FALLBACK:
        return
    loaded = 0
    for p in PROVINCE_REGISTRY:
        path = _series_disk_path(p["province_en"])
        try:
            age_s = time.time() - os.path.getmtime(path)
            if age_s > TARGET_FALLBACK_MAX_AGE_S:
                continue
            df = pd.read_csv(path)
            df["datetime_utc"] = pd.to_datetime(df["datetime_utc"], utc=True)
            if df.empty:
                continue
            _SERIES_CACHE[p["province_en"]] = (time.monotonic() - age_s, df, "stale_cache")
            loaded += 1
        except Exception:
            continue
    if loaded:
        print(f"[openaq] โหลด cache จากไฟล์ {loaded} จังหวัด (ใช้เป็นทางสำรองเท่านั้น ไม่ถือว่าสด)")


def fetch_series_from_openaq(meta):

    sensor_id = meta["sensor_id_pm25"]
    offset = int(meta.get("openaq_timestamp_offset_hours", 0)) if USE_TIMESTAMP_OFFSET else 0
    if not USE_INPUT_FALLBACK:
        return fetch_recent_pm25_openaq(sensor_id, offset_hours=offset), "live"
    if not _OPENAQ_BREAKER.allow():
        raise RuntimeError("OpenAQ circuit open")
    try:
        df, source = fetch_recent_pm25_openaq(sensor_id, offset_hours=offset), "live"
    except Exception as primary_err:
        try:
            df, source = fetch_recent_pm25_openaq_hours(sensor_id), "hours_endpoint"
            print(f"[openaq] {meta['province_en']}: /measurements ไม่สำเร็จ ({type(primary_err).__name__}) — ใช้ /hours แทน")
        except Exception:
            _OPENAQ_BREAKER.failure()
            raise primary_err
    _OPENAQ_BREAKER.success()
    return df, source


def get_recent_series(province, strict=False, max_age_s=SERIES_CACHE_TTL_S, info=None):

    now = time.monotonic()
    with _SERIES_LOCK:
        hit = _SERIES_CACHE.get(province)
    if hit is not None and now - hit[0] < max_age_s:
        if info is not None:
            info.update({"source": hit[2] if len(hit) > 2 else "live", "age_hours": round((now - hit[0]) / 3600, 2)})
        return hit[1]
    try:
        df, source = fetch_series_from_openaq(PROVINCES_BY_SLUG[province])
    except Exception:
        now = time.monotonic()
        limit = TARGET_FALLBACK_MAX_AGE_S if (strict and USE_INPUT_FALLBACK) else (None if strict else SERIES_CACHE_MAX_AGE_S)
        if hit is not None and limit is not None and now - hit[0] < limit and (hit[1] is not None or not strict):
            if info is not None:
                info.update({"source": "stale_cache", "age_hours": round((now - hit[0]) / 3600, 2)})
            return hit[1]
        if strict:
            raise
        return None
    if df is None or df.empty:
        df = None
    with _SERIES_LOCK:
        _SERIES_CACHE[province] = (now, df, source)
    _save_series_to_disk(province, df)
    if info is not None:
        info.update({"source": source, "age_hours": 0.0})
    return df


_load_series_disk_cache()


def _filled_grid(df_recent, t):
    
    s = df_recent.set_index("datetime_utc")["pm25"]
    hist = s[(s.index <= t) & (s.index >= t - pd.Timedelta(hours=72))]
    grid = pd.date_range(t - pd.Timedelta(hours=48), t, freq="h", tz="UTC")
    return hist.reindex(grid).interpolate(method="linear", limit=48, limit_direction="both")


def _neighbor_is_fresh(df_neighbor, latest_time):
    if df_neighbor is None or df_neighbor.empty:
        return False
    n_latest = df_neighbor["datetime_utc"].max()
    window = pd.Timedelta(hours=NEIGHBOR_FRESH_HOURS)
    return (latest_time - n_latest) <= window and (n_latest - latest_time) <= window


def compute_neighbor_features(province, latest_time, fetch=None):
    
    fetch = fetch or get_recent_series
    fresh = []  # [(province, DataFrame)] เรียงตามระยะทาง
    for idx, q in enumerate(NEIGHBOR_RANKING[province][:NEIGHBOR_MAX_FETCH]):
        df_q = fetch(q)
        is_fresh = _neighbor_is_fresh(df_q, latest_time)
        if is_fresh:
            fresh.append((q, df_q))
        if idx == 0 and is_fresh:
            break
        if len(fresh) >= NEIGHBOR_AVG_K:
            break

    result = {"status": "no_fresh_neighbor", "policy": None, "neighbors_used": [],
              "pm25_neighbor_avg": np.nan, "nb_trend_12h": np.nan}
    if not fresh:
        return result

    used = fresh[:1] if NEIGHBOR_RANKING[province][0] == fresh[0][0] else fresh[:NEIGHBOR_AVG_K]
    now_vals, ago_vals = [], []
    for q, df_q in used:
        grid = _filled_grid(df_q, latest_time)
        now_vals.append(grid.get(latest_time, np.nan))
        ago_vals.append(grid.get(latest_time - pd.Timedelta(hours=12), np.nan))
    avg_now = np.nanmean(now_vals) if not np.all(np.isnan(now_vals)) else np.nan
    if np.isnan(avg_now):
        return result
    avg_ago = np.nanmean(ago_vals) if not np.all(np.isnan(ago_vals)) else np.nan
    result.update({
        "status": "ok",
        "policy": "k1_exact" if used[0][0] == NEIGHBOR_RANKING[province][0] else "avg_nearest_fresh",
        "neighbors_used": [q for q, _ in used],
        "pm25_neighbor_avg": float(avg_now),
        # ไม่มีประวัติ 12 ชม. ก่อนหน้า -> trend = 0 (E4: +3.3%) แทน NaN
        "nb_trend_12h": float(avg_now - avg_ago) if not np.isnan(avg_ago) else 0.0,
    })
    return result


IMPUTATION_FEATURES = ["x0", "nb_now", "dnb", "nb_0", "j", "log1p_j", "hour_sin", "hour_cos"]  # ลำดับตรง matrix_trailing_lite ในสคริปต์ทดลอง


def _load_imputation_config():
    if not USE_TARGET_IMPUTATION:
        return None
    try:
        with open(IMPUTATION_CONFIG_PATH, encoding="utf-8") as f:
            cfg = json.load(f)
        if cfg["features"] != IMPUTATION_FEATURES:
            raise ValueError("ลำดับ feature ไม่ตรง")
        cfg["_mean"] = np.array(cfg["scaler_mean"], dtype=float)
        cfg["_scale"] = np.array(cfg["scaler_scale"], dtype=float)
        cfg["_coef"] = np.array(cfg["coef"], dtype=float)
        if not (len(cfg["_mean"]) == len(cfg["_scale"]) == len(cfg["_coef"]) == len(IMPUTATION_FEATURES)):
            raise ValueError("ขนาดเวกเตอร์ไม่ตรง")
        cfg["_q"] = {
            MODEL_LGB: {int(k): {int(h): float(v) for h, v in hs.items()} for k, hs in cfg["q_hat_imputed"]["lgb"].items()},
            MODEL_XGB: {int(k): {int(h): float(v) for h, v in hs.items()} for k, hs in cfg["q_hat_imputed"]["xgb"].items()},
        }
        return cfg
    except Exception as e:
        print(f"[imputation] ปิดฟีเจอร์เติมค่าสถานีเป้าหมาย ({type(e).__name__}: {e}) — ใช้พฤติกรรมเดิม")
        return None


IMPUTATION_CFG = _load_imputation_config()


def imputed_q_hat(model_name, k, h):
    
    table = IMPUTATION_CFG["_q"][model_name]
    ks = sorted(table)
    k_use = next((x for x in ks if x >= k), ks[-1])
    hs = sorted(table[k_use])
    return float(np.interp(h, hs, [table[k_use][x] for x in hs]))


def impute_target_series(province, df_recent, now=None, fetch=None):
    
    cfg = IMPUTATION_CFG
    if cfg is None or df_recent is None or df_recent.empty:
        return df_recent, None
    fetch = fetch or get_recent_series
    now = now if now is not None else pd.Timestamp.now(tz="UTC")
    anchor = now.floor("h") - pd.Timedelta(hours=1)
    t0 = df_recent["datetime_utc"].max()
    k = int((anchor - t0) / pd.Timedelta(hours=1))
    if k < IMPUTATION_MIN_K or k > IMPUTATION_MAX_K:
        return df_recent, None
    x0 = float(df_recent.loc[df_recent["datetime_utc"] == t0, "pm25"].iloc[0])
    meta = PROVINCES_BY_SLUG[province]
    series, dist = {}, {}
    for q in NEIGHBOR_RANKING[province][:cfg["n_neighbors_pool"]]:
        df_q = fetch(q)
        if df_q is None or df_q.empty:
            continue
        series[q] = df_q.drop_duplicates("datetime_utc").set_index("datetime_utc")["pm25"]
        qm = PROVINCES_BY_SLUG[q]
        dist[q] = float(haversine_km(meta["lat"], meta["lon"], qm["lat"], qm["lon"]))

    rows = []
    neighbors_now = []
    for j in range(1, k + 1):
        tau = t0 + pd.Timedelta(hours=j)
        w, v_tau, v_0, used = [], [], [], []
        for q in NEIGHBOR_RANKING[province][:cfg["n_neighbors_pool"]]:
            s = series.get(q)
            if s is None:
                continue
            a, b = s.get(tau, np.nan), s.get(t0, np.nan)
            if np.isnan(a) or np.isnan(b):
                continue
            w.append(1.0 / max(dist[q], 1.0) ** 2)
            v_tau.append(float(a))
            v_0.append(float(b))
            used.append(q)
            if len(w) == cfg["max_valid_neighbors"]:
                break
        if not w:
            if j == k:
                return df_recent, None  # ไม่มีเพื่อนบ้านที่ใช้ได้ "ตอนนี้" -> ไม่เติม (กลับไปพฤติกรรมเดิม)
            rows.append((tau, max(x0, 0.0)))
            continue
        w, v_tau, v_0 = np.array(w), np.array(v_tau), np.array(v_0)
        wsum = w.sum()
        dnb = float((w * (v_tau - v_0)).sum() / wsum)
        if j <= cfg["switch_j"]:
            val = x0 + dnb
        else:
            f = np.array([x0, float((w * v_tau).sum() / wsum), dnb, float((w * v_0).sum() / wsum), j, np.log1p(j),
                          np.sin(2 * np.pi * tau.hour / 24), np.cos(2 * np.pi * tau.hour / 24)])
            val = x0 + cfg["intercept"] + float(cfg["_coef"] @ ((f - cfg["_mean"]) / cfg["_scale"]))
        rows.append((tau, max(float(val), 0.0)))
        if j == k:
            neighbors_now = used

    aug = pd.concat([df_recent, pd.DataFrame(rows, columns=["datetime_utc", "pm25"])], ignore_index=True)
    info = {"k_hours": k, "method": "H-lite", "last_observed_utc": t0, "last_observed_pm25": x0,
            "estimated_pm25": rows[-1][1], "neighbors_used": neighbors_now}
    return aug, info


_FIRE_CACHE = {"fetched": None, "df": None}
_FIRE_LOCK = threading.Lock()


def get_fire_hotspots():

    if not FIRMS_KEY:
        return None, "no_key"
    now = time.monotonic()
    with _FIRE_LOCK:
        cached = dict(_FIRE_CACHE)
    if cached["df"] is not None and now - cached["fetched"] < FIRE_CACHE_TTL_S:
        return cached["df"], "cache"
    try:
        url = f"https://firms.modaps.eosdis.nasa.gov/api/area/csv/{FIRMS_KEY}/{FIRMS_NRT_SOURCE}/{FIRMS_AREA}/2"
        r = requests.get(url, timeout=20)
        text = r.text if r.status_code == 200 else ""
        if not text or "latitude" not in text.split("\n")[0]:
            raise ValueError("FIRMS response ไม่ใช่ CSV")
        from io import StringIO
        df = pd.read_csv(StringIO(text))
        if len(df):
            df["t"] = pd.to_datetime(df["acq_date"] + " " + df["acq_time"].astype(str).str.zfill(4),
                                      format="%Y-%m-%d %H%M", utc=True)
        else:
            df["t"] = pd.Series(dtype="datetime64[ns, UTC]")
    except Exception:
        print("[firms] ดึงจุดความร้อนไม่สำเร็จ (ไม่แสดงรายละเอียดเพราะ URL มีคีย์)")
        if cached["df"] is not None and now - cached["fetched"] < FIRE_CACHE_MAX_AGE_S:
            return cached["df"], "stale_cache"
        return None, "unavailable"
    with _FIRE_LOCK:
        _FIRE_CACHE["fetched"], _FIRE_CACHE["df"] = now, df
    return df, "live"


def count_fire_hotspots(province_meta, latest_time, hotspots_df):
  
    if hotspots_df is None or hotspots_df.empty:
        return 0
    near = haversine_km(province_meta["lat"], province_meta["lon"],
                        hotspots_df["latitude"].values, hotspots_df["longitude"].values) <= FIRE_RADIUS_KM
    t = hotspots_df["t"][near]
    return int(((t > latest_time - pd.Timedelta(hours=FIRE_WINDOW_HOURS)) & (t <= latest_time)).sum())


def migrate_daily_nlg_risk_group():
   
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("ALTER TABLE daily_nlg_log ADD COLUMN IF NOT EXISTS risk_group TEXT NOT NULL DEFAULT 'ทั่วไป'")
    cur.execute("""
        SELECT conname FROM pg_constraint
        WHERE conrelid = 'daily_nlg_log'::regclass AND contype = 'u'
    """)
    existing = {r[0] for r in cur.fetchall()}
    
    if ("daily_nlg_log_date_thai_risk_group_key" not in existing
            and "daily_nlg_log_date_thai_risk_group_province_key" not in existing):
        for name in existing:  
            cur.execute(f"ALTER TABLE daily_nlg_log DROP CONSTRAINT {name}")  
        cur.execute(
            "ALTER TABLE daily_nlg_log ADD CONSTRAINT daily_nlg_log_date_thai_risk_group_key "
            "UNIQUE (date_thai, risk_group)"
        )
    conn.commit()
    cur.close()
    conn.close()


migrate_daily_nlg_risk_group()


def migrate_province_column():
    
    tables = {
        "prediction_log": (["predicted_at_utc", "horizon"], "prediction_log_predicted_at_utc_horizon_province_key"),
        "daily_nlg_log": (["date_thai", "risk_group"], "daily_nlg_log_date_thai_risk_group_province_key"),
        "notification_log": (["recorded_at_utc"], "notification_log_recorded_at_utc_province_key"),
    }
    conn = get_db_connection()
    cur = conn.cursor()
    for table, (unique_cols, new_constraint_name) in tables.items():
        cur.execute(f"ALTER TABLE {table} ADD COLUMN IF NOT EXISTS province TEXT NOT NULL DEFAULT 'khonkaen'")
        cur.execute(
            "SELECT conname FROM pg_constraint WHERE conrelid = %s::regclass AND contype = 'u'",
            (table,),
        )
        existing = {r[0] for r in cur.fetchall()}
        if new_constraint_name not in existing:
            for name in existing:  # ลบ unique constraint เก่า (ไม่รวม province) 
                cur.execute(f"ALTER TABLE {table} DROP CONSTRAINT {name}")  # pg_constraint catalog เอง ไม่ใช่ input ภายนอก
            new_cols = ", ".join(unique_cols + ["province"])
            cur.execute(f"ALTER TABLE {table} ADD CONSTRAINT {new_constraint_name} UNIQUE ({new_cols})")
    conn.commit()
    cur.close()
    conn.close()


migrate_province_column()


def migrate_notification_trend_columns():
    # เก็บผล classify_daily_trend() ของแต่ละ snapshot ไว้ด้วย (ข้อ 3 ของฟีเจอร์แจ้งเตือนเพิ่มเติม) — nullable
    # เพราะแถวเก่าก่อน migration นี้ไม่มีค่าย้อนหลังให้คำนวณใหม่ (forecast list ตอนนั้นไม่ได้ถูกเก็บไว้)
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("ALTER TABLE notification_log ADD COLUMN IF NOT EXISTS trend_direction TEXT")
    cur.execute("ALTER TABLE notification_log ADD COLUMN IF NOT EXISTS trend_period TEXT")
    conn.commit()
    cur.close()
    conn.close()


migrate_notification_trend_columns()


def migrate_prediction_interval_columns():

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("ALTER TABLE prediction_log ADD COLUMN IF NOT EXISTS pm25_lo DOUBLE PRECISION")
    cur.execute("ALTER TABLE prediction_log ADD COLUMN IF NOT EXISTS pm25_hi DOUBLE PRECISION")
    conn.commit()
    cur.close()
    conn.close()


migrate_prediction_interval_columns()


def migrate_users_role():

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS role TEXT NOT NULL DEFAULT 'user'")
    conn.commit()
    cur.close()
    conn.close()


migrate_users_role()


def migrate_users_home_province():
   
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS home_province TEXT")
    conn.commit()
    cur.close()
    conn.close()


migrate_users_home_province()


# ===== Feature engineering & prediction logging =====

def build_feature_row(df_recent, current_weather, province_meta, neighbor=None, fire_count=0):
    
    latest_time = df_recent["datetime_utc"].max()
    pm25_by_time = df_recent.set_index("datetime_utc")["pm25"]

    full_grid = pd.date_range(latest_time - pd.Timedelta(hours=48), latest_time, freq="h", tz="UTC")
    s_filled = pm25_by_time.reindex(full_grid).interpolate(method="linear", limit=48, limit_direction="both")

    def lag(hours_ago):
        return s_filled.get(latest_time - pd.Timedelta(hours=hours_ago), np.nan)

    def roll_stats(n_hours):
        window = s_filled.loc[latest_time - pd.Timedelta(hours=n_hours - 1):latest_time]
        return window.mean(), window.std()

    roll_mean_6, roll_std_6 = roll_stats(6)
    roll_mean_24, roll_std_24 = roll_stats(24)
    wind_dir_deg = current_weather["wind_direction_10m"]

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
        "pm25_lag_1": lag(1),
        "pm25_lag_3": lag(3),
        "pm25_lag_6": lag(6),
        "pm25_lag_12": lag(12),
        "pm25_lag_24": lag(24),
        "pm25_roll_mean_6": roll_mean_6,
        "pm25_roll_std_6": roll_std_6,
        "pm25_roll_mean_24": roll_mean_24,
        "pm25_roll_std_24": roll_std_24,
        # climatology (weather_climatology) ส่งค่าเฉลี่ย sin/cos มาตรงๆ (ตรงกับที่ทดลอง) — ค่าจริงจาก Open-Meteo คิดจากองศาตามเดิม
        "wind_dir_sin": current_weather.get("_wind_dir_sin", np.sin(2 * np.pi * wind_dir_deg / 360)),
        "wind_dir_cos": current_weather.get("_wind_dir_cos", np.cos(2 * np.pi * wind_dir_deg / 360)),
        "pressure": current_weather["surface_pressure"],
        "lat": province_meta["lat"],
        "lon": province_meta["lon"],
        "pm25_neighbor_avg": neighbor["pm25_neighbor_avg"] if neighbor else np.nan,
        "nb_trend_12h": neighbor["nb_trend_12h"] if neighbor else np.nan,
        "fire_hotspot_count_100km_24h": fire_count,
    }

    X_now = pd.DataFrame([features])[FEATURE_COLS]
  
    return X_now, latest_time, float(pm25_by_time.loc[latest_time])


def log_predictions(predicted_at, rows, province):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.executemany(
        """INSERT INTO prediction_log
           (predicted_at_utc, target_time_utc, horizon, predicted_pm25, model_used, source, province, pm25_lo, pm25_hi)
           VALUES (%s, %s, %s, %s, %s, 'live', %s, %s, %s)
           ON CONFLICT (predicted_at_utc, horizon, province) DO NOTHING""",
        [(predicted_at.isoformat(), row["target_time_utc"].isoformat(), row["horizon"], row["pm25"],
          row["model_used"], province, row["pm25_lo"], row["pm25_hi"]) for row in rows],
    )
    conn.commit()
    cur.close()
    conn.close()


def log_notification_snapshot(recorded_at_utc, latest_pm25, province, trend_direction=None, trend_period=None):

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO notification_log (recorded_at_utc, latest_pm25, province, trend_direction, trend_period)
           VALUES (%s, %s, %s, %s, %s)
           ON CONFLICT (recorded_at_utc, province) DO NOTHING""",
        (recorded_at_utc, latest_pm25, province, trend_direction, trend_period),
    )
    conn.commit()
    cur.close()
    conn.close()


def backfill_actuals(df_recent, province):
    
    pm25_by_time = df_recent.set_index("datetime_utc")["pm25"].to_dict()

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, target_time_utc FROM prediction_log WHERE actual_pm25 IS NULL AND province = %s", (province,))
    pending = cur.fetchall()

    updates = []
    for row_id, target_time_str in pending:
        target_time = pd.Timestamp(target_time_str)
        actual = pm25_by_time.get(target_time)
        if actual is not None:
            updates.append((float(actual), float(actual), row_id))

    if updates:
        cur.executemany(
            "UPDATE prediction_log SET actual_pm25 = %s, actual_pm25_calibrated = %s WHERE id = %s",
            updates,
        )
        conn.commit()
    cur.close()
    conn.close()


# ===== ดัชนีคุณภาพอากาศ PM2.5 มาตรฐาน คพ. ประกาศ พ.ศ. 2566 (ราชกิจจานุเบกษา 3 ก.ค. 2566) =====
PM25_LEVELS = [
    (15, "ดีมาก", ["ดีมาก", "อากาศดีมาก"]),
    (25, "ดี", [("อากาศ", "ดี"), ("คุณภาพอากาศ", "ดี"), ("อยู่ใน", "เกณฑ์ดี"), "ปลอดภัย", "อากาศโปร่ง", "น่าพอใจ"]),
    (37.5, "ปานกลาง", ["ปานกลาง"]),
    (75, "เริ่มมีผลกระทบต่อสุขภาพ", ["เริ่มมีผลกระทบต่อสุขภาพ", "เริ่มมีผลกระทบ", "ส่งผลกระทบต่อสุขภาพ"]),
    (float("inf"), "มีผลกระทบต่อสุขภาพ", ["มีผลกระทบต่อสุขภาพ", "อันตราย", "อันตรายถึงชีวิต"]),
]

_GENERIC_CAUTION_KEYWORDS = [
    # "ลด"+"กิจกรรมกลางแจ้ง" ถูกตัดออก (เคยอยู่ตรงนี้) — "ลด" เป็นคำทั่วไปเกินไป (ใช้บ่อยตอนพูดว่า "ฝุ่นลดลง"
    # ซึ่งเป็นข่าวดี) พบจริงว่าทำให้ข้อความที่ถูกต้องสมบูรณ์ถูก reject ผิดๆ ตอนทดสอบกับ Gemini จริง เพราะ "ลด" ใน
    # ประโยคนึง (เช่น "ฝุ่นลดลง") บังเอิญอยู่ในระยะ max_gap=20 ตัวอักษรจาก "กิจกรรมกลางแจ้ง" ในอีกประโยคที่ไม่
    # เกี่ยวข้องกันเลย — "หลีกเลี่ยง"/"ระคายเคือง" เป็นคำเฉพาะเจาะจงกว่ามาก ไม่มีปัญหานี้ เก็บไว้ตามเดิม
    ("ปานกลาง", [("หลีกเลี่ยง", "กิจกรรมกลางแจ้ง"), "ระคายเคือง"]),
]

_NEGATION_WORDS = ["ไม่ใช่", "ไม่ค่อย", "มิได้", "ไม่"]  
_MAX_FLEX_GAP = 20  
_MAX_FLEX_GAP_STATUS = 50  


def _classify_pm25_level(value):
    for threshold, level_name, _ in PM25_LEVELS:
        if value <= threshold:
            return level_name
    return PM25_LEVELS[-1][1]


def pm25_status(value):
    return _classify_pm25_level(value)


# ===== NLG Safety Check  =====
_HEALTH_MESSAGE_TEMPLATES = {
    "ดีมาก": "คุณภาพอากาศอยู่ในเกณฑ์ดีมาก สามารถทำกิจกรรมกลางแจ้งได้ตามปกติ",
    "ดี": "คุณภาพอากาศอยู่ในเกณฑ์ดี สามารถทำกิจกรรมกลางแจ้งได้ตามปกติ",
    "ปานกลาง": "คุณภาพอากาศอยู่ในเกณฑ์ปานกลาง กลุ่มเสี่ยงควรระมัดระวัง",
    "เริ่มมีผลกระทบต่อสุขภาพ": "คุณภาพอากาศเริ่มมีผลกระทบต่อสุขภาพ ควรลดกิจกรรมกลางแจ้งที่ใช้แรงมาก",
    "มีผลกระทบต่อสุขภาพ": "คุณภาพอากาศมีผลกระทบต่อสุขภาพ ควรหลีกเลี่ยงกิจกรรมกลางแจ้ง",
}


def _level_index(level_name):
    return next(i for i, (_, name, _) in enumerate(PM25_LEVELS) if name == level_name)


def level_transition(current_pm25, previous_pm25):
    """ข้อ 1 ของฟีเจอร์แจ้งเตือนเพิ่มเติม "ระดับเปลี่ยนสถานะ" — คืน None ถ้าอยู่ระดับเดียวกัน (หรือไม่มีค่า
    ก่อนหน้าให้เทียบ) ไม่งั้นคืน dict บอกทิศทาง/จำนวนขั้นที่เปลี่ยน ใช้ _level_index() เทียบอันดับ ไม่ใช่เทียบ
    ชื่อ string ตรงๆ เพราะต้องรู้ทิศทาง (แย่ลง/ดีขึ้น) ด้วย"""
    if previous_pm25 is None:
        return None
    from_level = pm25_status(previous_pm25)
    to_level = pm25_status(current_pm25)
    if from_level == to_level:
        return None
    from_idx, to_idx = _level_index(from_level), _level_index(to_level)
    return {
        "from_level": from_level,
        "to_level": to_level,
        "from_pm25": previous_pm25,
        "to_pm25": current_pm25,
        "direction": "worse" if to_idx > from_idx else "better",
        "levels_moved": abs(to_idx - from_idx),
    }


def _find_keyword_occurrences(message_text, keyword, max_gap=_MAX_FLEX_GAP):
    
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

    all_occurrences = []
    for _, level_name, keywords in PM25_LEVELS:
        for kw in keywords:
            is_flexible = not isinstance(kw, str)
            for start, end in _find_keyword_occurrences(message_text, kw, max_gap=_MAX_FLEX_GAP_STATUS):
                all_occurrences.append((start, end, level_name, is_flexible))

    # flexible match (เช่น ("อากาศ","ดี")) เป็นแค่ตัวสำรองไว้ใช้ตอนไม่มี keyword ตรงตัว (literal) เจอเลยในทั้ง
    # ข้อความ — ถ้ามี literal match อยู่แล้วที่ไหนก็ได้ในข้อความ (ไม่ว่าระดับไหน) ไม่ให้ flexible match ของระดับอื่น
    # มาเพิ่มเป็น mention ซ้อนอีก เพราะมักเป็นแค่คำบรรยายทั่วไปที่ไม่เกี่ยวกับการประกาศระดับ (เจอจริง: ข้อความประกาศ
    # "ดีมาก" ไปแล้วตอนต้น แต่ท้ายประโยคพูด "สภาพอากาศที่ดีนี้" เป็นคำบรรยายทั่วไป ไม่ใช่การประกาศระดับ "ดี" ซ้ำ
    # ทำให้ false-reject เพราะดูเหมือนพูดถึง 2 ระดับ)
    if any(not is_flexible for _, _, _, is_flexible in all_occurrences):
        all_occurrences = [o for o in all_occurrences if not o[3]]

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
    for floor_level_name, keywords in _GENERIC_CAUTION_KEYWORDS:
        floor_idx = _level_index(floor_level_name)
        if floor_idx <= actual_idx:
            continue
        for kw in keywords:
            if _find_keyword_occurrences(message_text, kw):
                return True
    return False


# เช็คเฉพาะตอนคำนวณแนวโน้มได้ชัดเจน ("up"/"down" จาก classify_daily_trend) — ตอน "flat" ปล่อยผ่านเสมอ เพราะคำทั่วไป
# เช่น "ลดลงเล็กน้อย" ใช้บรรยายความผันผวนเล็กๆ ได้โดยไม่ได้ขัดแย้งกับ "ทรงตัว" จริงๆ (เข้มงวดเกินจะ false-reject ข้อความ
# ที่ถูกต้องอยู่แล้ว เหมือนที่เจอใน scripts/test_nlg_safety_check.py เคส (e))
#
# ต้องเช็คแบบมีคำว่า "แนวโน้ม" อยู่ใกล้ๆ ด้วย (ไม่ใช่แค่หาคำว่า "ลดลง"/"สูงขึ้น" เดี่ยวๆ ทั้งข้อความ) — ทดสอบจริงกับ
# Gemini พบว่าส่วน (2) ของ prompt (สาเหตุจาก SHAP factor) มักมีคำพวกนี้ปนอยู่แล้วเพื่อบรรยาย "ปัจจัยย่อย" (เช่น
# "ฝุ่นจากพื้นที่ใกล้เคียงลดลง" ซึ่งเป็นสาเหตุหนึ่ง ไม่ใช่ข้อสรุปแนวโน้มรวม) ถ้าจับแบบเจอคำเดี่ยวๆ ทั้งข้อความจะ
# false-reject ข้อความที่ถูกต้องอยู่แล้วบ่อยมาก เพราะ prompt สั่งให้พูดถึงปัจจัยพวกนี้อยู่แล้ว — การกำหนดให้ต้องอยู่
# ใกล้คำว่า "แนวโน้ม" (คำที่ _trend_fact_th() ใช้เองตอนป้อนข้อเท็จจริงให้ Gemini เขียนตาม) ปลอดภัยกว่ามาก เพราะ
# Gemini มักเอาคำนี้มาใช้ซ้ำตอนพูดถึง "แนวโน้มรวม" จริงๆ ตามที่สั่งไว้ใน prompt ข้อ (1)
_TREND_CONTRADICTION_KEYWORDS = {
    "up": ["ลดลง", "ดีขึ้น"],
    "down": ["สูงขึ้น", "เพิ่มขึ้น", "แย่ลง"],
}
_TREND_PROXIMITY_WORD = "แนวโน้ม"
_TREND_PROXIMITY_GAP = 15


def _mentions_contradicting_trend(message_text, trend):
    direction, _period = trend
    for kw in _TREND_CONTRADICTION_KEYWORDS.get(direction, []):
        if (_find_keyword_occurrences(message_text, (_TREND_PROXIMITY_WORD, kw), max_gap=_TREND_PROXIMITY_GAP) or
                _find_keyword_occurrences(message_text, (kw, _TREND_PROXIMITY_WORD), max_gap=_TREND_PROXIMITY_GAP)):
            return True
    return False


def _fallback_message(level):
    if level is None:
        return "ไม่สามารถประเมินคุณภาพอากาศได้ในขณะนี้ กรุณาตรวจสอบใหม่ภายหลัง"
    return _HEALTH_MESSAGE_TEMPLATES[level]


def validate_health_message(message_text, predicted_pm25, risk_group="ทั่วไป", trend=("flat", None)):

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
    # เช็คทุก risk_group เหมือนกัน (ไม่ใช่แค่ "ทั่วไป") — คำเตือนระดับสูงกว่าจริงผิดสำหรับทุกกลุ่ม ไม่ใช่แค่กลุ่มทั่วไป
    if _mentions_generic_caution_below(message_text, actual_idx):
        return False, actual_level, _fallback_message(actual_level)  # คำเตือนทั่วไปที่สูงกว่า floor ของระดับจริง
    if _mentions_contradicting_trend(message_text, trend):
        return False, actual_level, _fallback_message(actual_level)  # พูดแนวโน้มสวนทางกับที่คำนวณได้จริง

    return True, actual_level, message_text


# ===== NLG Generation =====
# แปลงค่าพยากรณ์เป็นคำอธิบายไทยสั้นๆ ผ่าน Gemini ตรวจด้วย validate_health_message() ก่อนคืนค่าเสมอ
try:
    _gemini_client = genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None
except Exception as e:
    print(f"[nlg] Gemini client init failed ({e}); NLG จะ fallback ไป template เสมอจนกว่าจะตั้ง GEMINI_KEY")
    _gemini_client = None
GEMINI_MODEL = "gemini-3.1-flash-lite" 
GEMINI_FALLBACK_MODEL = "gemini-3.5-flash-lite"
GEMINI_TIMEOUT_SECONDS = 11
GEMINI_MAX_RETRIES = 3  

_gemini_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="gemini-nlg")


GEMINI_MIN_CALL_SPACING_S = 4.5  # >4.0s (=60/15) เผื่อ margin
_gemini_rate_lock = threading.Lock()
_gemini_last_call_started_at = [0.0]

# ชื่อ feature ทางเทคนิค  คำอธิบายภาษาไทยที่คนทั่วไปเข้าใจ ใช้ตอนสร้าง prompt ส่งให้ Gemini
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
    "wind_dir_sin": "ทิศทางลม",
    "wind_dir_cos": "ทิศทางลม",
    "pressure": "ความกดอากาศ",
    "lat": "ตำแหน่งทางภูมิศาสตร์",
    "lon": "ตำแหน่งทางภูมิศาสตร์",
    "fire_hotspot_count_100km_24h": "จุดความร้อนจากไฟป่าในพื้นที่ใกล้เคียง",
    "nb_trend_12h": "แนวโน้มฝุ่นของจังหวัดข้างเคียง",
}

# คำอธิบายเชิงบรรยาย (ซ่อนชื่อ feature ดิบ เช่น "pm25_lag_3", พูดถึงปรากฏการณ์แทน) ต่อคู่ (feature, ทิศทาง
# SHAP) ใช้กับ /api/forecast/shap ให้ผู้ใช้ทั่วไปอ่านแล้วเข้าใจทันทีแบบเดียวกับที่ daily NLG ทำกับข้อความสรุป
# รายวัน — เขียนเฉพาะคู่ที่มีเหตุผลเชิงอุตุนิยมวิทยาที่ยืนยันได้ชัดเจน (อิงทิศทางเดียวกับ
# METEOROLOGICAL_HYPOTHESES ใน scripts/shap_analysis_clean.py ซึ่งผ่านการเช็ค plausibility กับข้อมูลจริงแล้ว)
# ส่วนที่ไม่มีสมมติฐานยืนยันชัดเจน (temp, wind_dir, lat/lon, roll_std) ปล่อย fallback ไปใช้ _FEATURE_NAME_TH
# ตรงๆ แทนที่จะเดาเหตุผลที่ไม่ได้ตรวจสอบมาเขียนเป็นคำอธิบายฟันธง
_FACTOR_NARRATIVE_TH = {
    ("pm25_lag_1", "up"): "ค่าฝุ่นชั่วโมงที่แล้วยังอยู่ในระดับสูง",
    ("pm25_lag_1", "down"): "ค่าฝุ่นชั่วโมงที่แล้วเริ่มลดลง",
    ("pm25_lag_3", "up"): "ฝุ่นสะสมต่อเนื่องในช่วง 3 ชั่วโมงที่ผ่านมา",
    ("pm25_lag_3", "down"): "ฝุ่นลดลงต่อเนื่องในช่วง 3 ชั่วโมงที่ผ่านมา",
    ("pm25_lag_6", "up"): "ฝุ่นสะสมต่อเนื่องตลอดช่วง 6 ชั่วโมงที่ผ่านมา",
    ("pm25_lag_6", "down"): "ฝุ่นลดลงต่อเนื่องตลอดช่วง 6 ชั่วโมงที่ผ่านมา",
    ("pm25_lag_12", "up"): "แนวโน้มฝุ่นสูงต่อเนื่องมาตั้งแต่ 12 ชั่วโมงก่อน",
    ("pm25_lag_12", "down"): "แนวโน้มฝุ่นลดลงต่อเนื่องมาตั้งแต่ 12 ชั่วโมงก่อน",
    ("pm25_lag_24", "up"): "ฝุ่นสูงกว่าช่วงเวลาเดียวกันของเมื่อวาน",
    ("pm25_lag_24", "down"): "ฝุ่นต่ำกว่าช่วงเวลาเดียวกันของเมื่อวาน",
    ("pm25_roll_mean_6", "up"): "ค่าเฉลี่ยฝุ่นในช่วง 6 ชั่วโมงล่าสุดอยู่ในระดับสูง",
    ("pm25_roll_mean_6", "down"): "ค่าเฉลี่ยฝุ่นในช่วง 6 ชั่วโมงล่าสุดอยู่ในระดับต่ำ",
    ("pm25_roll_mean_24", "up"): "ค่าเฉลี่ยฝุ่นในช่วง 24 ชั่วโมงล่าสุดอยู่ในระดับสูง",
    ("pm25_roll_mean_24", "down"): "ค่าเฉลี่ยฝุ่นในช่วง 24 ชั่วโมงล่าสุดอยู่ในระดับต่ำ",
    ("wind_speed", "down"): "ลมพัดแรงขึ้น ช่วยระบายฝุ่น",
    ("wind_speed", "up"): "ลมอ่อน อากาศนิ่ง ฝุ่นสะสมมากขึ้น",
    ("precipitation", "down"): "มีฝนตกในพื้นที่ ช่วยชะล้างฝุ่นออกจากอากาศ",
    ("precipitation", "up"): "อากาศแห้ง ไม่มีฝนช่วยชะล้างฝุ่น",
    ("pressure", "up"): "ความกดอากาศสูง อากาศปิด กักฝุ่นไว้ใกล้พื้น",
    ("pressure", "down"): "ความกดอากาศต่ำ อากาศถ่ายเทดีขึ้น",
    ("humidity", "up"): "ความชื้นสูงและอากาศนิ่ง ฝุ่นจับตัวหนาแน่นขึ้น",
    ("humidity", "down"): "อากาศแห้งขึ้น ฝุ่นในอากาศเจือจางลง",
    ("fire_hotspot_count_100km_24h", "up"): "พบจุดความร้อนจากการเผาในพื้นที่ใกล้เคียงหนาแน่นขึ้น",
    ("fire_hotspot_count_100km_24h", "down"): "จุดความร้อนจากการเผาในพื้นที่ใกล้เคียงลดลง",
    ("nb_trend_12h", "up"): "ฝุ่นจากจังหวัดข้างเคียงมีแนวโน้มเพิ่มขึ้น กำลังเคลื่อนเข้าพื้นที่",
    ("nb_trend_12h", "down"): "ฝุ่นจากจังหวัดข้างเคียงมีแนวโน้มลดลง",
}


def _factor_narrative_th(feature_name, direction):
    return _FACTOR_NARRATIVE_TH.get((feature_name, direction), _feature_name_th(feature_name))


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

# ใช้ตรวจ risk_group ตอนสมัคร/แก้โปรไฟล์ — ผูกกับโทน NLG รายวันและ alert threshold แล้ว
VALID_RISK_GROUPS = set(_USER_GROUP_TONE.keys())

# คำแนะนำเจาะจงต่อ (risk_group, ระดับ PM2.5 จริง) ใช้เสริม _USER_GROUP_TONE ด้านบนใน prompt (ไม่ได้แทนที่ — ห้าม
# ลบ/แก้ _USER_GROUP_TONE เพราะ scripts/test_risk_group_e2e.py เช็ค string ของมันตรงๆ) ส่งเป็น "แนวทางให้เลือกใช้"
# ไม่ใช่ข้อความบังคับยกคำ — ยังต้องผ่าน _mentions_generic_caution_below เหมือนข้อความอื่น (ไม่ใช่ safety bypass)
# หมายเหตุ: ร่างโดยอิงโทนเดิม + ตัวอย่างจากเอกสาร ไม่ใช่คำแนะนำทางการแพทย์ ควรให้ผู้ใช้ทบทวนเนื้อหาก่อน ship จริง
_RISK_GROUP_ACTION_HINTS = {
    ("ทั่วไป", "ดีมาก"): "ทำกิจกรรมกลางแจ้งได้เต็มที่ตามปกติ",
    ("ทั่วไป", "ดี"): "ทำกิจกรรมกลางแจ้งได้ตามปกติ",
    ("ทั่วไป", "ปานกลาง"): "ยังทำกิจกรรมกลางแจ้งได้ตามปกติ แต่ควรสังเกตอาการตนเองหากรู้สึกระคายเคือง",
    ("ทั่วไป", "เริ่มมีผลกระทบต่อสุขภาพ"): "ควรลดระยะเวลาหรือความหนักของกิจกรรมกลางแจ้งลงจากปกติ",
    ("ทั่วไป", "มีผลกระทบต่อสุขภาพ"): "ควรเลี่ยงกิจกรรมกลางแจ้งที่ใช้แรงมาก และสวมหน้ากากป้องกันฝุ่นหากจำเป็นต้องออกนอกอาคาร",
    ("เด็ก-ผู้สูงอายุ", "ดีมาก"): "เด็กและผู้สูงอายุทำกิจกรรมกลางแจ้งได้ตามปกติ",
    ("เด็ก-ผู้สูงอายุ", "ดี"): "เด็กและผู้สูงอายุทำกิจกรรมกลางแจ้งได้ตามปกติ แต่ควรสังเกตอาการหากทำกิจกรรมต่อเนื่องเป็นเวลานาน",
    ("เด็ก-ผู้สูงอายุ", "ปานกลาง"): "เด็กและผู้สูงอายุควรลดระยะเวลาทำกิจกรรมกลางแจ้งลงบ้าง และสังเกตอาการระคายเคืองตาหรือจมูก",
    ("เด็ก-ผู้สูงอายุ", "เริ่มมีผลกระทบต่อสุขภาพ"): "เด็กและผู้สูงอายุควรเลี่ยงกิจกรรมกลางแจ้งที่ใช้แรงมาก และสวมหน้ากากป้องกันฝุ่นหากต้องออกนอกอาคาร",
    ("เด็ก-ผู้สูงอายุ", "มีผลกระทบต่อสุขภาพ"): "เด็กและผู้สูงอายุควรงดกิจกรรมกลางแจ้ง อยู่ในอาคารที่มีอากาศถ่ายเทดี และสวมหน้ากาก N95 หากจำเป็นต้องออกนอกอาคาร",
    ("มีโรคประจำตัว", "ดีมาก"): "ผู้มีโรคประจำตัวทำกิจกรรมกลางแจ้งได้ตามปกติ",
    ("มีโรคประจำตัว", "ดี"): "ผู้มีโรคประจำตัวทำกิจกรรมกลางแจ้งได้ตามปกติ แต่ควรพกยาประจำตัวติดตัวไว้เสมอ",
    ("มีโรคประจำตัว", "ปานกลาง"): "ผู้มีโรคประจำตัวทางเดินหายใจควรลดกิจกรรมกลางแจ้งที่ใช้แรงมาก และพกยาประจำตัวติดตัว",
    ("มีโรคประจำตัว", "เริ่มมีผลกระทบต่อสุขภาพ"): "ผู้มีโรคประจำตัวควรเลี่ยงกิจกรรมกลางแจ้ง พกยาประจำตัวติดตัวเสมอ และสวมหน้ากาก N95 หากต้องออกนอกอาคาร",
    ("มีโรคประจำตัว", "มีผลกระทบต่อสุขภาพ"): "ผู้มีโรคประจำตัวควรงดออกนอกอาคารโดยไม่จำเป็น พกยาประจำตัวติดตัวตลอดเวลา และสวมหน้ากาก N95 เมื่อจำเป็นต้องออกนอกอาคาร",
}


def _action_hint_th(risk_group, status):
    return _RISK_GROUP_ACTION_HINTS.get((risk_group, status), "")

_NLG_SYSTEM_PROMPT = (
    "คุณเป็นผู้ช่วยอธิบายค่าพยากรณ์คุณภาพอากาศ PM2.5 ให้ประชาชนทั่วไปฟัง ตอบเป็นภาษาไทยเท่านั้น "
    "ความยาวไม่เกิน 2-3 ประโยค กระชับ เข้าใจง่าย ห้ามใช้ศัพท์เทคนิคหรือชื่อโมเดล/ฟีเจอร์ตรงๆ "
    "(ห้ามพูดคำว่า SHAP, feature, lag, rolling, โมเดล) ให้อธิบายด้วยภาษาที่คนทั่วไปคุ้นเคยแทน "
    "ห้ามวินิจฉัยโรคหรือให้คำแนะนำเชิงคลินิกใดๆ ให้แนะนำเฉพาะการวางแผนกิจกรรมกลางแจ้งตามระดับฝุ่นเท่านั้น"
)

def _feature_name_th(feature_name):
    return _FEATURE_NAME_TH.get(feature_name, feature_name)


def _format_shap_factor_lines(shap_pairs, limit=3):
    top_factors = (shap_pairs or [])[:limit]
    if not top_factors:
        return "- ไม่มีข้อมูลปัจจัยเพิ่มเติมสำหรับรอบนี้"
    return "\n".join(
        f"- {_factor_narrative_th(name, 'up' if val > 0 else 'down')}"
        for name, val in top_factors
    )


def _trend_fact_th(trend):
    direction, period = trend
    if direction == "flat":
        return "มีแนวโน้มทรงตัวตลอด 24 ชั่วโมงข้างหน้า ไม่เปลี่ยนแปลงมาก"
    verb = "สูงขึ้น" if direction == "up" else "ลดลง"
    return f"มีแนวโน้ม{verb}ชัดเจนในช่วง{period}"


def _build_daily_nlg_prompt(daily_avg_pm25, status, shap_pairs, risk_group="ทั่วไป", trend=("flat", None)):
    factor_lines = _format_shap_factor_lines(shap_pairs)
    tone = _USER_GROUP_TONE.get(risk_group, _USER_GROUP_TONE["ทั่วไป"])
    action_hint = _action_hint_th(risk_group, status)
    trend_fact = _trend_fact_th(trend)

    return (
        f'ค่าเฉลี่ย PM2.5 ตลอดวันนี้ (สรุปจากค่าพยากรณ์ล่วงหน้า 24 ชั่วโมง) '
        f'อยู่ที่ {daily_avg_pm25} ไมโครกรัม/ลูกบาศก์เมตร ซึ่งจัดอยู่ในระดับคุณภาพอากาศ "{status}" {trend_fact}\n\n'
        f"ปัจจัยที่มีอิทธิพลต่อแนวโน้มตลอดวันนี้มากที่สุด (ประเมินจากช่วงกลางวัน):\n{factor_lines}\n\n"
        f"แนวทางคำแนะนำที่ควรอ้างอิง (เลือกใช้ตามความเหมาะสม ไม่ต้องยกคำมาทั้งหมด): {action_hint}\n\n"
        f"ช่วยเขียนคำอธิบายคุณภาพอากาศวันนี้สั้นๆ ไม่เกิน 2-3 ประโยค ให้ผู้ใช้งานทั่วไปเข้าใจง่าย "
        f"เรียงเนื้อหาตามลำดับนี้: "
        f"(1) สถานการณ์วันนี้ (ระดับคุณภาพอากาศ) และแนวโน้มตามข้อเท็จจริงที่ให้ไว้ข้างต้นเป๊ะๆ ห้ามเปลี่ยนทิศทาง"
        f"แนวโน้มเอง ห้ามพูดตัวเลขค่าเฉลี่ย {daily_avg_pm25} ซ้ำอีก (มีตัวเลขแสดงแยกไว้บนหน้าจออยู่แล้ว ไม่ต้องย้ำ) "
        f"(2) สาเหตุหลักโดยอ้างอิงจากปัจจัยที่ให้มาข้างต้นเท่านั้น ห้ามเดาสาเหตุเอง "
        f"(3) คำแนะนำโดย{tone} "
        f'ต้องใช้คำว่า "{status}" ตรงๆ ในส่วน (1) อย่างน้อยหนึ่งครั้งเสมอ (เช่น "อยู่ในระดับ{status}") '
        f'แม้จะเป็นคำยาวหรือฟังดูเป็นทางการก็ตาม ห้ามเลี่ยงไปใช้คำอื่นแทนทั้งหมด '
        f'ห้ามเอ่ยชื่อระดับคุณภาพอากาศอื่นที่ไม่ใช่ "{status}" เด็ดขาด (เช่น ห้ามพูดถึงคำว่า '
        f'"ปานกลาง" หรือ "เริ่มมีผลกระทบต่อสุขภาพ" ถ้าระดับปัจจุบันไม่ใช่ระดับนั้น) '
        f"ให้บรรยายความระมัดระวังด้วยคำทั่วไปแทนการเอ่ยชื่อระดับ"
    )


def _wait_for_gemini_rate_limit():

    with _gemini_rate_lock:
        now = time.monotonic()
        wait = GEMINI_MIN_CALL_SPACING_S - (now - _gemini_last_call_started_at[0])
        if wait > 0:
            time.sleep(wait)
        _gemini_last_call_started_at[0] = time.monotonic()


def _parse_gemini_retry_delay_s(e):

    try:
        err = e.details.get("error", e.details)
        for d in err.get("details", []):
            if str(d.get("@type", "")).endswith("RetryInfo"):
                return float(str(d["retryDelay"]).rstrip("s"))
    except Exception:
        pass
    return None


def _call_single_gemini_attempt(model, prompt):

    future = _gemini_executor.submit(
        _gemini_client.models.generate_content,
        model=model,
        contents=prompt,
        config=genai_types.GenerateContentConfig(system_instruction=_NLG_SYSTEM_PROMPT),
    )
    try:
        response = future.result(timeout=GEMINI_TIMEOUT_SECONDS)
        return (response.text or "").strip(), None
    except Exception as e:
        return None, e


def _call_gemini(prompt):

    if _gemini_client is None:
        return None

    delay = 1
    for attempt in range(GEMINI_MAX_RETRIES):
        _wait_for_gemini_rate_limit()
        text, e = _call_single_gemini_attempt(GEMINI_MODEL, prompt)
        if e is None:
            return text
        if isinstance(e, FutureTimeoutError):
            print(f"[nlg] Gemini call exceeded {GEMINI_TIMEOUT_SECONDS}s timeout")
            break
        if isinstance(e, genai_errors.APIError):
 
            if e.code in (429, 503) and attempt < GEMINI_MAX_RETRIES - 1:
          
                retry_delay = _parse_gemini_retry_delay_s(e)
                wait_s = retry_delay if retry_delay is not None else delay
                source = "retryDelay จริง" if retry_delay is not None else "backoff เดิม (parse ไม่ได้)"
                reason = "429 (quota)" if e.code == 429 else "503 (โมเดลโหลดสูงชั่วคราว)"
                print(f"[nlg] Gemini {reason}, retry {attempt + 1}/{GEMINI_MAX_RETRIES} in {wait_s}s ({source})")
                time.sleep(wait_s)
                delay *= 2
                continue
            print(f"[nlg] Gemini API error (code={e.code})")
        else:
            print(f"[nlg] Gemini call failed ({type(e).__name__})")
        break

    # โมเดลหลักล้มเหลวทุกช่องทางแล้ว (timeout หรือ retry หมด) ลองโมเดลสำรอง 1 ครั้งก่อนยอมแพ้ไป template
    print(f"[nlg] primary model ({GEMINI_MODEL}) failed, trying fallback model ({GEMINI_FALLBACK_MODEL})")
    _wait_for_gemini_rate_limit()
    text, e = _call_single_gemini_attempt(GEMINI_FALLBACK_MODEL, prompt)
    if e is None:
        print(f"[nlg] fallback model ({GEMINI_FALLBACK_MODEL}) succeeded")
        return text
    print(f"[nlg] fallback model also failed ({type(e).__name__}), falling back to template")
    return None


# ===== NLG รายวัน (สรุปทั้งวันแทนรายชั่วโมง) =====


def get_thai_now():
    """เวลาไทย (Asia/Bangkok) — แยกฟังก์ชันให้ test monkeypatch ได้ ห้ามใช้วันที่ฝั่ง UTC ตรงๆ
    (ต่างโซนเวลาเกือบครึ่งวัน จะข้ามวันผิดช่วง 17:00-23:59 UTC)"""
    return pd.Timestamp.now(tz="UTC").tz_convert("Asia/Bangkok")


def get_daily_nlg_row(date_thai, risk_group, province):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """SELECT date_thai, daily_avg_pm25, explanation_text, used_fallback, generated_at_utc
           FROM daily_nlg_log WHERE date_thai = %s AND risk_group = %s AND province = %s""",
        (date_thai, risk_group, province),
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


def save_daily_nlg_row(date_thai, risk_group, province, daily_avg_pm25, explanation_text, used_fallback, generated_at_utc):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO daily_nlg_log
           (date_thai, risk_group, province, daily_avg_pm25, explanation_text, used_fallback, generated_at_utc)
           VALUES (%s, %s, %s, %s, %s, %s, %s)""",
        (date_thai, risk_group, province, daily_avg_pm25, explanation_text, bool(used_fallback), generated_at_utc),
    )
    conn.commit()
    cur.close()
    conn.close()


REP_HORIZON_FOR_SHAP = 12


def generate_daily_nlg_explanation(forecast, feature_row, model_used, risk_group="ทั่วไป"):

    daily_avg_pm25 = round(sum(f["pm25"] for f in forecast) / len(forecast), 1)
    status = pm25_status(daily_avg_pm25)
    trend = classify_daily_trend(forecast)

    try:
        shap_pairs = get_shap_explanation(model_used, REP_HORIZON_FOR_SHAP, feature_row)
    except Exception as e:
        print(f"[nlg-daily] SHAP explanation failed ({type(e).__name__}): {e}")
        shap_pairs = None

    prompt = _build_daily_nlg_prompt(daily_avg_pm25, status, shap_pairs, risk_group, trend)
    message_text = _call_gemini(prompt)

    used_fallback = False
    if message_text:
        is_valid, _, final_message = validate_health_message(message_text, daily_avg_pm25, risk_group, trend)
        if not is_valid:
            # log ข้อความดิบที่ Gemini ตอบมาไว้ด้วย (ไม่ใช่แค่บอกว่า fail) เพื่อให้ปรับ keyword list ใน
            # PM25_LEVELS/_TREND_CONTRADICTION_KEYWORDS ได้จากของจริง แทนการเดา
            print(f"[nlg-daily] safety check failed: Gemini message did not match actual level "
                  f"'{status}' (trend={trend}), using fallback template. ข้อความที่ถูกปฏิเสธ: {message_text!r}")
            used_fallback = True
    else:
        final_message = _fallback_message(status)
        used_fallback = True

    return daily_avg_pm25, final_message, used_fallback


def maybe_generate_daily_nlg(forecast, feature_row, risk_group, province, model_used):
  
    thai_now = get_thai_now()
    if thai_now.hour < 6:
        return

    date_thai = thai_now.strftime("%Y-%m-%d")
    if get_daily_nlg_row(date_thai, risk_group, province) is not None:
        return

    try:
        daily_avg_pm25, explanation_text, used_fallback = generate_daily_nlg_explanation(
            forecast, feature_row, model_used, risk_group
        )
        generated_at_utc = thai_now.tz_convert("UTC").isoformat()
        save_daily_nlg_row(date_thai, risk_group, province, daily_avg_pm25, explanation_text, used_fallback, generated_at_utc)
        print(f"[nlg-daily] generated for {date_thai} ({risk_group}, {province}): avg_pm25={daily_avg_pm25}, fallback={used_fallback}")
    except Exception as e:
        print(f"[nlg-daily] generation failed for {date_thai} ({risk_group}, {province}): {e}")


def get_today_daily_nlg(risk_group, province):
    date_thai = get_thai_now().strftime("%Y-%m-%d")
    row = get_daily_nlg_row(date_thai, risk_group, province)
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


MIN_LIVE_FOR_SUMMARY = 5  


def summarize_errors(all_errors, live_errors):
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

class DataUnavailableError(RuntimeError):
    """OpenAQ ไม่มี reading ของจังหวัดเป้าหมายเลยใน 72 ชม.ล่าสุด (สถานีขัดข้อง) — /api/forecast ตอบ 503 แทน 500"""


def _nan_to_none(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else x


def get_hourly_weather_cached(province_meta):
    """fetch_hourly_weather_openmeteo() พร้อม cache (TTL HOURLY_WEATHER_TTL_S) — เดิมจุดนี้ไม่มี cache เลย
    ยิง Open-Meteo ใหม่ทุกครั้งที่มีคนเปิด/รีเฟรชพยากรณ์จังหวัดไหนก็ได้ เป็นจุดที่กินโควตารายวันมากที่สุด
    (เทียบกับ get_current_weather() ที่มี cache 10 นาทีอยู่แล้ว) ถ้าดึงสดไม่ได้เลย (เช่น โควตารายวันหมด)
    จะใช้ค่า cache เก่าต่อได้ไม่เกิน HOURLY_WEATHER_STALE_MAX_AGE_H ชม. ก่อนจะคืน {} (เลียน pattern
    get_current_weather() ทุกจุด) คืน dict เวลา(floor ชม.)->(temp, precip_prob, weather_code) หรือ {}"""
    prov = province_meta["province_en"]
    now = time.monotonic()
    with _WEATHER_LOCK:
        hit = _HOURLY_WEATHER_CACHE.get(prov)
    if hit is not None and now - hit[0] < HOURLY_WEATHER_TTL_S:
        return hit[1]
    try:
        hourly = fetch_hourly_weather_openmeteo(province_meta["lat"], province_meta["lon"])
        weather_times = pd.to_datetime(hourly["time"], utc=True)
        weather_by_hour = {
            t: (hourly["temperature_2m"][i], hourly["precipitation_probability"][i], hourly["weather_code"][i])
            for i, t in enumerate(weather_times)
        }
        with _WEATHER_LOCK:
            _HOURLY_WEATHER_CACHE[prov] = (time.monotonic(), weather_by_hour)
        return weather_by_hour
    except Exception as e:
        age_s = time.monotonic() - hit[0] if hit is not None else None
        if hit is not None and age_s <= HOURLY_WEATHER_STALE_MAX_AGE_H * 3600:
            print(f"[hourly_timeline] {prov}: ดึงสภาพอากาศรายชั่วโมงไม่สำเร็จ ({type(e).__name__}) — "
                  f"ใช้ของเก่าที่ cache ไว้ {age_s / 60:.0f} นาทีที่แล้ว")
            return hit[1]
        print(f"[hourly_timeline] {prov}: ดึงสภาพอากาศรายชั่วโมงไม่สำเร็จ ({type(e).__name__}) และไม่มีค่า "
              f"cache ที่ใหม่พอ — กราฟจะไม่มีอุณหภูมิ/โอกาสฝน")
        return {}


def build_hourly_timeline(weather_by_hour, latest_time, df_observed, forecast, live_now=None):
    """รวม PM2.5+สภาพอากาศ 'จริง' ของ 24 ชม.ที่ผ่านมา กับ PM2.5 พยากรณ์ 24 ชม.ข้างหน้า (จาก forecast[] ที่คำนวณ
    ไปแล้ว บวกอุณหภูมิ/โอกาสฝนรายชั่วโมงจาก Open-Meteo) เป็นเส้นเวลาเดียว 49 จุด สำหรับกราฟ 'พยากรณ์อากาศ'
    รายชั่วโมงฝั่ง frontend (เลื่อนดูอดีตได้) — ล้มเหลวได้แบบไม่กระทบ /api/forecast เดิม (คืน [] ให้ frontend
    fallback ไปโหมดไม่มีสภาพอากาศ/อดีต)

    weather_by_hour: ผลจาก get_hourly_weather_cached(province_meta) — ย้ายมาดึง "ล่วงหน้า" แบบขนานกับ
    target/weather/fire ตั้งแต่ build_live_feature_row() แทนที่จะดึงตรงนี้ (เดิมดึงตอนนี้เลย ต้องรอจน
    model ทำนายครบ 24 horizon เสร็จก่อนจะเริ่มดึงได้ ทั้งที่ไม่ได้ขึ้นกับ forecast เลย)

    live_now: ผลจาก get_air4thai_live() (หรือ None) — ถ้ามี ใช้ค่า pm25 สดของ Air4Thai ที่จุด "ตอนนี้" เท่านั้น
    ข้อมูลย้อนหลังเป็น OpenAQ ล้วน (เคยลองใช้ Air4Thai ด้วยใน CHANGELOG 188 แล้วย้อนกลับใน 189: ผู้ใช้เทียบ
    ข้อมูลจริงแล้วพบว่า OpenAQ เป็นค่าเฉลี่ยเคลื่อนที่ 24 ชม. ราบเรียบ ต่างจาก Air4Thai ที่เป็นค่าดิบผันผวนมาก
    ทำให้เส้นย้อนหลัง(ผันผวน)กับเส้นพยากรณ์ที่เทรนบน OpenAQ (ราบเรียบ)ดูไม่กลืนกัน จึงเลือกคงย้อนหลังเป็น OpenAQ
    ไว้ ยอมรับว่าจะมีช่องว่าง 1-2 ชม.ก่อนถึง "ตอนนี้" เพราะ OpenAQ ล่าช้ากว่า Air4Thai)"""

    def weather_at(ts):
        return weather_by_hour.get(ts.floor("h"), (None, None, None))

    pm25_by_hour = {ts.floor("h"): val for ts, val in zip(df_observed["datetime_utc"], df_observed["pm25"])}

    def past_pm25_at(ts):
        return _nan_to_none(round(pm25_by_hour[ts], 1)) if ts in pm25_by_hour else None

    anchor = latest_time.floor("h")
    # ป้ายเวลาที่โชว์ผู้ใช้สำหรับจุดที่มาจาก OpenAQ (อดีต/anchor) ต้องบวก OPENAQ_DISPLAY_LABEL_SHIFT_H เสมอ
    # (ดูคำอธิบายที่ค่าคงที่) — การ "ค้นหาค่า PM2.5" (pm25_by_hour) ยังต้องคีย์ด้วยเวลาดิบ (ts/anchor) เป๊ะ
    # เหมือนเดิม (ค่า PM2.5 ถูกเก็บโดย OpenAQ ภายใต้เวลาดิบนั้น) แต่ "การค้นหาค่าอากาศ" (weather_at, จาก
    # Open-Meteo ซึ่งไม่มีปัญหา period ambiguity แบบ OpenAQ) ต้องคีย์ด้วยเวลาจริง (ts+label_shift) ถึงจะตรงกับ
    # ชั่วโมงเดียวกับที่ค่า PM2.5 ของจุดนั้น "หมายถึง" จริงๆ — ไม่งั้นอากาศจะเป็นของชั่วโมงก่อนหน้า 1 ชม.เทียบ
    # กับป้าย/ค่า PM2.5 ที่โชว์คู่กัน (พบระหว่างตรวจสอบหลังแก้ป้ายเวลา PM2.5 แล้ว ดู CHANGELOG ถัดจาก 193)
    label_shift = pd.Timedelta(hours=OPENAQ_DISPLAY_LABEL_SHIFT_H)
    timeline = []
    for k in range(24, 0, -1):
        ts = anchor - pd.Timedelta(hours=k)
        display_ts = (ts + label_shift).tz_convert("Asia/Bangkok")
        temp, rain_pct, code = weather_at(ts + label_shift)
        timeline.append({
            "time": display_ts.strftime("%H:%M"),
            "hour_thai": display_ts.hour,
            "pm25": past_pm25_at(ts),
            "temp_c": _nan_to_none(temp), "rain_pct": _nan_to_none(rain_pct), "weather_code": _nan_to_none(code),
            "is_forecast": False, "is_now": False,
        })

    # ถ้า Air4Thai สดกว่า anchor ของ OpenAQ พอที่จะ "ตรงชั่วโมง" กับจุดพยากรณ์ใดจุดหนึ่งพอดี ให้ย้ายตำแหน่ง
    # "ตอนนี้" ไปที่จุดนั้นแทน (แทนที่ anchor เดิม) เพื่อให้กราฟสะท้อนเวลาจริงจริง ๆ — สำคัญ: ต้อง "ย้าย flag"
    # ไม่ใช่แค่ "เขียนทับ label" ของจุด anchor เฉย ๆ เพราะทุกจุดในไทม์ไลน์มีเวลาจริงของตัวเองอยู่แล้ว
    # (ตามปฏิทิน ไม่ใช่ตามรอบ 24 ชม.) การเขียนทับ label ของจุดเดียวเคยทำให้เลขชั่วโมงหาย/ซ้ำ (CHANGELOG 186
    # ย้อนกลับใน 187) การย้าย flag แทนทำให้ทุกจุดยังมีป้ายเวลาถูกต้องของตัวเองเสมอ ไม่ซ้ำไม่หาย
    offset_h = None
    if live_now is not None:
        real_now_hour = live_now["datetime_utc"].floor("h")
        # เทียบกับ "เวลาจริง" ของ anchor (anchor + label_shift) ไม่ใช่ anchor ดิบ — ถ้า OpenAQ ตรงเวลาจริงเป๊ะ
        # (เคสปกติที่เจอบ่อยสุด) diff_h จะได้ 0 ซึ่งหลุดช่วง 1..24 ด้านล่างพอดี ตกไปใช้ anchor เป็น "ตอนนี้" เอง
        # (branch offset_h is None) ถูกต้องตามความหมายใหม่ ไม่ต้องเพิ่ม branch พิเศษ
        diff_h = int(round((real_now_hour - (anchor + label_shift)) / pd.Timedelta(hours=1)))
        if 1 <= diff_h <= 24:
            offset_h = diff_h

    temp, rain_pct, code = weather_at(anchor + label_shift)
    anchor_pm25 = past_pm25_at(anchor)
    display_anchor = (anchor + label_shift).tz_convert("Asia/Bangkok")
    now_idx = len(timeline)
    timeline.append({
        "time": "ตอนนี้" if offset_h is None else display_anchor.strftime("%H:%M"),
        "hour_thai": display_anchor.hour,
        "pm25": (round(live_now["pm25"], 1) if live_now is not None else anchor_pm25) if offset_h is None else anchor_pm25,
        "temp_c": _nan_to_none(temp), "rain_pct": _nan_to_none(rain_pct), "weather_code": _nan_to_none(code),
        "is_forecast": False, "is_now": offset_h is None,
    })

    for h, f in enumerate(forecast, start=1):
        temp, rain_pct, code = weather_at(anchor + label_shift + pd.Timedelta(hours=h))
        timeline.append({
            "time": f["display_time"], "hour_thai": f["display_hour_thai"], "pm25": f["pm25"],
            "pm25_lo": f["pm25_lo"], "pm25_hi": f["pm25_hi"], "horizon": h,
            "temp_c": _nan_to_none(temp), "rain_pct": _nan_to_none(rain_pct), "weather_code": _nan_to_none(code),
            "is_forecast": True, "is_now": False,
        })

    if offset_h is not None:
        # ชั่วโมงที่อยู่ระหว่าง anchor (ข้อมูลจริงล่าสุดของ OpenAQ) กับ "ตอนนี้" จริง (ถ้า offset_h > 1) ยังไม่มี
        # ข้อมูลจริงให้ใช้เลย ต้องอาศัยค่าพยากรณ์ h=1..offset_h-1 ของโมเดลไปพลางๆ — ทำเครื่องหมาย is_gap_fill
        # ไว้ให้ frontend แสดงเป็นสถานะที่ 3 (ไม่ใช่ทั้งอดีตจริง ไม่ใช่ทั้งอนาคตปกติ) ไม่แตะ is_forecast เดิม
        # เผื่อมีโค้ดอื่นอ้างอิง field นี้อยู่ (CHANGELOG 190)
        for gap_i in range(now_idx + 1, now_idx + offset_h):
            timeline[gap_i]["is_gap_fill"] = True

        target = timeline[now_idx + offset_h]
        target["is_now"] = True
        target["is_forecast"] = False
        target["pm25_forecast_was"] = target["pm25"]  # ค่าที่เคยพยากรณ์ไว้ล่วงหน้าสำหรับชั่วโมงนี้ (เก็บไว้เทียบ)
        target["pm25"] = round(live_now["pm25"], 1)
        target["pm25_lo"] = None
        target["pm25_hi"] = None

    return timeline


def build_live_feature_row(province):
    """ดึงข้อมูลสดล่าสุด (PM2.5/สภาพอากาศ/เพื่อนบ้าน/จุดความร้อน) ของจังหวัดนี้ แล้วสร้าง X_now แถวเดียว —
    เป็นส่วน "เตรียม feature" ของ get_forecast() เดิม แยกออกมาเป็นฟังก์ชันของตัวเอง เพื่อให้
    /api/forecast/shap (อธิบายค่าพยากรณ์ชั่วโมงที่ผู้ใช้เลือกด้วย SHAP) เรียกซ้ำได้โดยไม่ต้อง copy โค้ด
    fetch ข้อมูลสดทั้งชุด — raise DataUnavailableError เหมือนเดิมถ้าไม่มีข้อมูล PM2.5 เลย"""
    _t = {}  # DEBUG_TIMING: ชื่อขั้นตอน -> วินาทีที่ใช้ (ดูแผนวัดเวลา)
    province_meta = PROVINCES_BY_SLUG[province]
    series_info = {}
    # target/weather/fire ไม่ขึ้นกับกันเลย (weather/fire ไม่ต้องรอผลของ target) ยิงพร้อมกันแทนทีละจุด —
    # weather ใช้ at=None (เวลา "ตอนนี้" แทน reference_time ของ target) เพราะ at มีผลแค่ชั้น fallback
    # climatology เท่านั้น (ชั้น live fetch หลักไม่ใช้เวลาเลย) ผลต่างที่ยอมรับได้แลกกับการไม่ต้องรอ target ก่อน
    _t_parallel0 = time.perf_counter()
    _target_future = _IO_EXECUTOR.submit(get_recent_series, province, strict=True, max_age_s=TARGET_CACHE_TTL_S, info=series_info)
    _weather_future = _IO_EXECUTOR.submit(get_current_weather, province_meta, at=None)
    _fire_future = _IO_EXECUTOR.submit(get_fire_hotspots) if USES_FIRE else None
    # ดึงสภาพอากาศรายชั่วโมง (ใช้แค่กราฟ "พยากรณ์อากาศ" ฝั่ง frontend) ล่วงหน้าตั้งแต่ตรงนี้เลย — ไม่ได้ขึ้นกับ
    # forecast[] ที่ยังไม่คำนวณเลย (เดิมดึงตอนท้ายสุดใน build_hourly_timeline() ทำให้ต้องรอ 24 horizon เสร็จก่อน)
    _hourly_weather_future = _IO_EXECUTOR.submit(get_hourly_weather_cached, province_meta)

    df_recent = _target_future.result()
    _t["target"] = time.perf_counter() - _t_parallel0
    if df_recent is None or df_recent.empty:
        raise DataUnavailableError(f"ไม่มีข้อมูล PM2.5 ล่าสุดของ {province} จาก OpenAQ (สถานีอาจขัดข้อง)")
    df_observed = df_recent  # ข้อมูลจริงล้วน (ก่อนเติมค่า) — ใช้กับ backfill_actuals()/อายุข้อมูลเสมอ ห้ามใช้ df_recent หลัง impute
    imputed_info = None
    _t0 = time.perf_counter()
    if IMPUTATION_CFG is not None:
        df_recent, imputed_info = impute_target_series(province, df_recent)
    _t["impute"] = time.perf_counter() - _t0
    imputed = imputed_info is not None

    # feature เพื่อนบ้าน + จุดความร้อน — เวลาอ้างอิงเดียวกับที่ใช้คิด lag; คำนวณ/ดึงเฉพาะเมื่อโมเดลที่โหลดใช้ feature นั้นจริง
    reference_time = df_recent["datetime_utc"].max()
    # สภาพอากาศพร้อมทางสำรอง : ค่าที่ดึงไว้ -> สด -> ค่าเก่า <=2 ชม. -> climatology; source != live เก็บไว้ใน data_quality
    _t0 = time.perf_counter()
    current_weather, weather_info = _weather_future.result()
    _t["weather"] = time.perf_counter() - _t0
    input_degraded = series_info.get("source") == "stale_cache" or weather_info["source"] == "climatology"
    _t0 = time.perf_counter()
    _neighbor_fetch_count = [0]
    if USES_NEIGHBOR:
        def _counting_fetch(q):
            _neighbor_fetch_count[0] += 1
            return get_recent_series(q)
        neighbor = compute_neighbor_features(province, reference_time, fetch=_counting_fetch)
    else:
        neighbor = {"status": "not_used", "policy": None, "neighbors_used": [], "pm25_neighbor_avg": np.nan, "nb_trend_12h": np.nan}
    _t["neighbor"] = time.perf_counter() - _t0
    _t["neighbor_fetch_n"] = _neighbor_fetch_count[0]
    _t0 = time.perf_counter()
    if USES_FIRE:
        hotspots_df, fire_source = _fire_future.result()
        fire_count = count_fire_hotspots(province_meta, reference_time, hotspots_df)
    else:
        fire_source, fire_count = "not_used", 0
    _t["fire"] = time.perf_counter() - _t0
    _t0 = time.perf_counter()
    hourly_weather_by_hour = _hourly_weather_future.result()
    _t["hourly_weather_prefetch"] = time.perf_counter() - _t0
    # โมเดลใช้เพื่อนบ้านแต่ไม่มีเพื่อนบ้านสดเลย -> ไม่มีค่า neighbor ให้โมเดล ต้องใช้ naive_fallback ทุก horizon (ห้ามส่ง NaN เข้าโมเดล)
    force_naive = USES_NEIGHBOR and neighbor["status"] != "ok"

    X_now, latest_time, latest_pm25 = build_feature_row(df_recent, current_weather, province_meta, neighbor, fire_count)

    target_age_hours = (pd.Timestamp.now(tz="UTC") - df_observed["datetime_utc"].max()).total_seconds() / 3600
    target_stale = bool(target_age_hours > TARGET_STALE_HOURS) and not imputed

    return {
        "province_meta": province_meta, "X_now": X_now, "latest_time": latest_time, "latest_pm25": latest_pm25,
        "df_observed": df_observed, "imputed_info": imputed_info, "imputed": imputed,
        "target_age_hours": target_age_hours, "target_stale": target_stale, "input_degraded": input_degraded,
        "force_naive": force_naive, "current_weather": current_weather, "weather_info": weather_info,
        "series_info": series_info, "fire_source": fire_source, "fire_count": fire_count, "neighbor": neighbor,
        "hourly_weather_by_hour": hourly_weather_by_hour,
        "_t": _t,
    }


def get_forecast(province=DEFAULT_PROVINCE, risk_group="ทั่วไป"):
    _t_req0 = time.perf_counter()
    # r2/air4thai ไม่ขึ้นกับ build_live_feature_row() เลย (ใช้แค่ province) ยิงพร้อมกันไปตั้งแต่ตอนนี้ ผลจะ
    # พร้อมใช้ (หรือเกือบพร้อม) ตอนที่โค้ดด้านล่างต้องใช้จริงโดยไม่ต้องรอเพิ่ม
    _r2_future = _IO_EXECUTOR.submit(fetch_rolling_r2_lookup, province)
    _air4thai_future = _IO_EXECUTOR.submit(get_air4thai_live, province)
    ctx = build_live_feature_row(province)
    province_meta = ctx["province_meta"]
    hourly_weather_by_hour = ctx["hourly_weather_by_hour"]
    X_now, latest_time, latest_pm25 = ctx["X_now"], ctx["latest_time"], ctx["latest_pm25"]
    df_observed, imputed_info, imputed = ctx["df_observed"], ctx["imputed_info"], ctx["imputed"]
    target_age_hours, target_stale, input_degraded = ctx["target_age_hours"], ctx["target_stale"], ctx["input_degraded"]
    force_naive, current_weather, weather_info = ctx["force_naive"], ctx["current_weather"], ctx["weather_info"]
    series_info, fire_source, fire_count, neighbor = ctx["series_info"], ctx["fire_source"], ctx["fire_count"], ctx["neighbor"]
    latest_time_th = latest_time.tz_convert("Asia/Bangkok")

    _t = ctx["_t"]
    _t0 = time.perf_counter()
    r2_lookup = _r2_future.result()
    _t["r2"] = time.perf_counter() - _t0

    forecast = []
    log_rows = []
    risk_horizons = []
    fallback_triggered_horizons = []
    for h in range(1, 25):
        model_name = select_model(X_now, h, province, r2_lookup, force_naive)

        if model_name == MODEL_NAIVE:
            pred = round(float(latest_pm25), 1)
            p05_adj = pred
            p95_upper_adj = pred
            fallback_triggered_horizons.append(h)
        else:
            models_by_q = MODELS[h] if model_name == MODEL_LGB else XGB_MODELS[h]
            # residual forecasting: โมเดลทาย pm25_t+h - pm25_now ต้องบวก latest_pm25 กลับทุกจุด
            pred = round(float(models_by_q[0.5].predict(X_now)[0]) + latest_pm25, 1)
            p05_pred = float(models_by_q[0.05].predict(X_now)[0]) + latest_pm25
            p95_pred = float(models_by_q[0.95].predict(X_now)[0]) + latest_pm25
            q_hat = imputed_q_hat(model_name, imputed_info["k_hours"], h) if imputed else CONFORMAL_QHAT[model_name][h]
            p05_adj = p05_pred - q_hat
            p95_upper_adj = p95_pred + q_hat

        forecast_time_utc = latest_time + pd.Timedelta(hours=h)
        forecast_time_th = forecast_time_utc.tz_convert("Asia/Bangkok")
        # display_time/display_hour_thai = เวลาที่ "ควรโชว์ผู้ใช้" ตาม convention ของ OpenAQ เอง (ดู
        # OPENAQ_DISPLAY_LABEL_SHIFT_H ด้านบน) — time/hour_thai เดิมคงไว้เป๊ะไม่แก้ค่า เพราะยังใช้คำนวณภายใน
        # (เช่น find_best_time()) ที่ต้องสม่ำเสมอกับ latest_time ดิบ
        display_time_th = forecast_time_th + pd.Timedelta(hours=OPENAQ_DISPLAY_LABEL_SHIFT_H)
        forecast.append({
            "time": forecast_time_th.strftime("%H:%M"),
            "hour_thai": forecast_time_th.hour,
            "display_time": display_time_th.strftime("%H:%M"),
            "display_hour_thai": display_time_th.hour,
            "pm25": pred,
            "pm25_lo": round(max(0.0, p05_adj), 1),
            "pm25_hi": round(max(0.0, p95_upper_adj), 1),
            "model_used": model_name,
        })
        log_rows.append({
            "horizon": h,
            "target_time_utc": forecast_time_utc,
            "pm25": pred,
            "model_used": model_name,
            "pm25_lo": round(max(0.0, p05_adj), 1),
            "pm25_hi": round(max(0.0, p95_upper_adj), 1),
        })

        if is_horizon_at_risk(pred, risk_group):
            risk_horizons.append(h)

    # เรียก classify_daily_trend() แยกต่างหากจากที่ generate_daily_nlg_explanation() เรียกอยู่แล้ว (คนละจุด คนละ
    # gating — อันนี้คำนวณทุกครั้งที่เรียก get_forecast() ไม่ใช่แค่วันละครั้งต่อ risk_group) ตั้งใจไม่รีแฟคเตอร์ให้
    # ใช้ค่าร่วมกัน เพราะเป็น pure function รับ forecast เดียวกัน ผลจึงตรงกันเสมอ ไม่เสี่ยงขัดแย้งกัน — คำนวณตรงนี้
    # (ไม่ใช่แค่ใน if loggable: ด้านล่าง) เพื่อให้ /api/forecast เองก็มีค่านี้ใช้แสดงผลสดได้ทันทีด้วย
    trend_direction, trend_period = classify_daily_trend(forecast)

    loggable = not target_stale and not imputed and not input_degraded
    if loggable:
        log_predictions(latest_time, log_rows, province)
        log_notification_snapshot(
            latest_time.isoformat(), float(latest_pm25), province, trend_direction, trend_period
        )
    backfill_actuals(df_observed, province)
    if loggable:
        rep_model_used = forecast[REP_HORIZON_FOR_SHAP - 1]["model_used"]
        maybe_generate_daily_nlg(forecast, X_now, risk_group, province, rep_model_used)

    best_time = find_best_time(forecast)

    high_pollution_risk = is_current_at_risk(latest_pm25, risk_group) or len(risk_horizons) > 0

    # ข้อ 2 "ใกล้ถึงช่วงเสี่ยงแล้ว" — forecast ถูกสร้างตามลำดับ h=1..24 ต่อเนื่อง ดัชนี h-1 จึงตรงกับ horizon h เสมอ
    min_risk_horizon = min(risk_horizons) if risk_horizons else None
    risk_imminent = min_risk_horizon is not None and min_risk_horizon <= IMMINENT_HORIZON_H
    min_risk_horizon_pm25 = forecast[min_risk_horizon - 1]["pm25"] if min_risk_horizon is not None else None
    # forecast[].display_time คำนวณไว้แล้ว (บวก OPENAQ_DISPLAY_LABEL_SHIFT_H) ใช้ตรงๆ ให้ตรงกับเวลาที่กราฟ
    # พยากรณ์หลักโชว์เป๊ะ ไม่ต้องคำนวณซ้ำ
    min_risk_horizon_display_time = forecast[min_risk_horizon - 1]["display_time"] if min_risk_horizon is not None else None

    # Air4Thai สดกว่า OpenAQ มาก (~28 นาที vs ~2.5 ชม. วัดจริง) — ใช้แสดงผล "ตอนนี้" (การ์ด PM2.5 ตอนนี้ +
    # จุด "ตอนนี้" บนกราฟพยากรณ์รายชั่วโมง) เท่านั้น latest_pm25 ด้านบน (ฐานพยากรณ์ของโมเดล, บวกเข้ากับ
    # residual ทุก horizon) ยังคงมาจาก OpenAQ เหมือนเดิมทุกจุด ไม่แตะ เพราะ Air4Thai อาจสเกลไม่ตรงกับที่โมเดล
    # เทรนมา (ดู docstring get_air4thai_live/fetch_air4thai_live_all)
    _t0 = time.perf_counter()
    live_now = _air4thai_future.result()
    _t["air4thai"] = time.perf_counter() - _t0
    current_pm25 = live_now["pm25"] if live_now is not None else float(latest_pm25)
    # Air4Thai ไม่มีปัญหา period-label ambiguity แบบ OpenAQ เลยไม่ต้อง shift — fallback ไป latest_time_th
    # (OpenAQ) เท่านั้นที่ต้องบวก OPENAQ_DISPLAY_LABEL_SHIFT_H ให้ตรง convention เดียวกับจุดอื่นบนหน้า
    current_datetime_th = (
        live_now["datetime_utc"].tz_convert("Asia/Bangkok") if live_now is not None
        else latest_time_th + pd.Timedelta(hours=OPENAQ_DISPLAY_LABEL_SHIFT_H)
    )

    _t0 = time.perf_counter()
    hourly_timeline_value = build_hourly_timeline(hourly_weather_by_hour, latest_time, df_observed, forecast, live_now=live_now)
    _t["hourly_timeline"] = time.perf_counter() - _t0

    if DEBUG_TIMING:
        _t["total"] = time.perf_counter() - _t_req0
        _parts = " ".join(f"{k}={v:.2f}s" if k != "neighbor_fetch_n" else f"neighbor_fetch_n={v}" for k, v in _t.items())
        print(f"[timing] {province}: {_parts}")

    return {
        "province": {"th": province_meta["province_th"], "en": province_meta["province_en"],
                      "is_zero_shot": province_meta["is_zero_shot"]},
        "current": {
            "pm25": round(current_pm25, 1),
            "datetime_th": current_datetime_th.strftime("%Y-%m-%d %H:%M"),
            "status": pm25_status(current_pm25),
        },
        "current_weather": {
            "temp_c": round(current_weather["temperature_2m"], 1),
            "humidity_pct": round(current_weather["relative_humidity_2m"]),
            "wind_speed_kmh": round(current_weather["wind_speed_10m"], 1),
        },
        "forecast": forecast,
        "hourly_timeline": hourly_timeline_value,
        "best_time": best_time,
        "high_pollution_risk": high_pollution_risk,
        "risk_horizons": risk_horizons,
        "risk_imminent": risk_imminent,
        "min_risk_horizon": min_risk_horizon,
        "min_risk_horizon_pm25": min_risk_horizon_pm25,
        "min_risk_horizon_display_time": min_risk_horizon_display_time,
        "trend_direction": trend_direction,
        "trend_period": trend_period,
        "fallback_triggered_horizons": fallback_triggered_horizons,
        "data_quality": {
            "target_age_hours": round(float(target_age_hours), 1),
            "observed_lag_k": int((pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(hours=1)
                                   - df_observed["datetime_utc"].max()) / pd.Timedelta(hours=1)),
            "target_stale": target_stale,
            "series_age_hours": series_info.get("age_hours", 0.0),
            "weather_source": weather_info["source"],
            "weather_age_hours": weather_info["age_hours"],
            "neighbor_status": neighbor["status"],
            "neighbor_policy": neighbor["policy"],
            "neighbors_used": neighbor["neighbors_used"],
            "fire_source": fire_source,
            "fire_count_24h": int(fire_count),
            "input_imputed": None if not imputed else {
                "k_hours": imputed_info["k_hours"],
                "method": imputed_info["method"],
                "last_observed_datetime_th": imputed_info["last_observed_utc"].tz_convert("Asia/Bangkok").strftime("%Y-%m-%d %H:%M"),
                "last_observed_pm25": round(imputed_info["last_observed_pm25"], 1),
                "estimated_pm25": round(imputed_info["estimated_pm25"], 1),
                "neighbors_used": imputed_info["neighbors_used"],
            },
        },
        "daily_summary": get_today_daily_nlg(risk_group, province),
        "risk_group": risk_group,
        "risk_score_config": {
            "risk_group_offset": RISK_GROUP_THRESHOLD_OFFSET.get(risk_group, 0.0),
            "activity_offsets": ACTIVITY_INTENSITY_OFFSET,
            "breakpoints": RISK_SCORE_BREAKPOINTS,
        },
    }


TARGET_LATENCY_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "target_latency_scheduler_log.csv")


TARGET_LATENCY_LOG_HEADER_OLD = "measured_utc,province_en,observed_lag_k,target_age_hours,imputed"  
TARGET_LATENCY_LOG_HEADER = TARGET_LATENCY_LOG_HEADER_OLD + ",series_source"
_LATENCY_LOG_LOCK = threading.Lock()  
_LATENCY_LOG_CHECKED = set()  


def _upgrade_latency_log_header(path):
    with open(path, encoding="utf-8", newline="") as f:
        lines = f.readlines()
    if not lines or lines[0].strip() != TARGET_LATENCY_LOG_HEADER_OLD:
        return
    lines[0] = TARGET_LATENCY_LOG_HEADER + "\n"
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        f.writelines(lines)
    os.replace(tmp, path)


def log_target_latency(province, data_quality):
    try:
        path = TARGET_LATENCY_LOG_PATH
        with _LATENCY_LOG_LOCK:
            new_file = not os.path.exists(path) or os.path.getsize(path) == 0
            if not new_file and path not in _LATENCY_LOG_CHECKED:
                try:
                    _upgrade_latency_log_header(path)
                except Exception:
                    pass  
            _LATENCY_LOG_CHECKED.add(path)
            with open(path, "a", encoding="utf-8", newline="") as f:
                if new_file:
                    f.write(TARGET_LATENCY_LOG_HEADER + "\n")
                f.write(f"{pd.Timestamp.now(tz='UTC').isoformat(timespec='seconds')},{province},"
                        f"{data_quality.get('observed_lag_k')},{data_quality.get('target_age_hours')},"
                        f"{int(bool(data_quality.get('input_imputed')))},{data_quality.get('series_source', 'live')}\n")
    except Exception:
        pass


def scheduled_forecast_job():
    summary = {"ok": 0, "stale": 0, "naive": 0, "imputed": 0, "input_fallback": 0, "failed": 0}
    prefetch_weather_all()
    for p in PROVINCE_REGISTRY:
        try:
            result = get_forecast(province=p["province_en"])
            dq = result["data_quality"]
            log_target_latency(p["province_en"], dq)
            try:
                dispatch_push_for_province(p["province_en"])
            except Exception as push_err:
                # push dispatch ต้องไม่ทำให้รอบ dq-summary ของจังหวัดนี้นับเป็น "failed" ไปด้วย — แยก
                # try/except ของตัวเองเพื่อกัน subscription/push ที่พังไม่ให้กระทบการ log/backfill หลัก
                print(f"[push] dispatch ล้มเหลว ({p['province_en']}): {push_err}")
            if dq.get("series_source") == "stale_cache" or dq.get("weather_source") == "climatology":
                summary["input_fallback"] += 1
                print(f"[scheduler] {p['province_en']}: ใช้ข้อมูลสำรอง (pm25={dq.get('series_source')}, weather={dq.get('weather_source')}) — ไม่บันทึก log")
            if dq["target_stale"]:
                summary["stale"] += 1
                print(f"[scheduler] {p['province_en']}: ข้อมูลเก่า {dq['target_age_hours']} ชม. — ไม่บันทึก log")
            elif dq.get("input_imputed"):
                summary["imputed"] += 1
                print(f"[scheduler] {p['province_en']}: สถานีเงียบ {dq['input_imputed']['k_hours']} ชม. — เติมค่าด้วย H-lite (ไม่บันทึก log)")
            elif dq["neighbor_status"] not in ("ok", "not_used"):
                summary["naive"] += 1
                print(f"[scheduler] {p['province_en']}: ไม่มีเพื่อนบ้านที่ข้อมูลสด — ใช้ naive_fallback ทุก horizon")
            else:
                summary["ok"] += 1
                print(f"[scheduler] auto forecast logged OK ({p['province_en']}, neighbor={dq['neighbor_policy']})")
        except Exception as e:
            summary["failed"] += 1
            print(f"[scheduler] auto forecast failed ({p['province_en']}): {e}")
        time.sleep(0.5)
    print(f"[scheduler] สรุปรอบนี้: {summary}")


# ===== Flask routes =====

@app.route("/index.html")
def index_html_redirect():
    return redirect("/")


@app.route("/compare.html")
def compare_html_redirect():
    return redirect("/compare")


@app.route("/dev-cert.pem")
def dev_cert_download():
    """ดาวน์โหลด public cert (dev_cert.pem จาก scripts/generate_dev_cert.py) เพื่อติดตั้งเป็น trusted บนมือถือ
    ตอนทดสอบ HTTPS — เสิร์ฟแค่ cert สาธารณะ (ไม่ใช่ private key) ไม่มีความลับ แต่กันไว้เฉพาะ DEBUG เพราะเป็น
    ไฟล์สำหรับ dev เท่านั้น ไม่ควรมีใน production จริง"""
    if not DEBUG:
        return jsonify({"error": "not available"}), 404
    cert_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dev_cert.pem")
    if not os.path.exists(cert_path):
        return jsonify({"error": "ยังไม่ได้สร้าง dev_cert.pem — รัน scripts/generate_dev_cert.py ก่อน"}), 404
    return send_from_directory(
        os.path.dirname(cert_path), "dev_cert.pem",
        mimetype="application/x-x509-ca-cert", as_attachment=False,
    )


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_spa(path):
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
        home_province = data.get("home_province") or None  # "" -> None (ยังไม่ตั้ง)

        if not _EMAIL_RE.match(email):
            return jsonify({"error": "รูปแบบอีเมลไม่ถูกต้อง"}), 400
        if len(password) < MIN_PASSWORD_LENGTH:
            return jsonify({"error": f"รหัสผ่านต้องมีอย่างน้อย {MIN_PASSWORD_LENGTH} ตัวอักษร"}), 400
        if risk_group not in VALID_RISK_GROUPS:
            return jsonify({"error": "risk_group ไม่ถูกต้อง"}), 400
        if home_province is not None and home_province not in PROVINCES_BY_SLUG:
            return jsonify({"error": "home_province ไม่ถูกต้อง"}), 400
        if age is not None:
            try:
                age = int(age)
            except (TypeError, ValueError):
                return jsonify({"error": "age ต้องเป็นตัวเลข"}), 400
            if not 1 <= age <= 120:
                return jsonify({"error": "age ต้องอยู่ระหว่าง 1-120"}), 400

        if get_user_by_email(email) is not None:
            return jsonify({"error": "อีเมลนี้ถูกใช้งานแล้ว"}), 409

        user_id = create_user(email, generate_password_hash(password), risk_group, age, home_province)
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
    if user is None:  
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
        home_province = data.get("home_province", current["home_province"]) or None  # "" -> None (ล้างค่า)

        if not _EMAIL_RE.match(email):
            return jsonify({"error": "รูปแบบอีเมลไม่ถูกต้อง"}), 400
        if risk_group not in VALID_RISK_GROUPS:
            return jsonify({"error": "risk_group ไม่ถูกต้อง"}), 400
        if home_province is not None and home_province not in PROVINCES_BY_SLUG:
            return jsonify({"error": "home_province ไม่ถูกต้อง"}), 400
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

        update_user_profile(session["user_id"], email, risk_group, age, home_province)
        old_home = current["home_province"]
        if old_home and home_province and home_province != old_home:
            move_push_subscriptions_with_home(session["user_id"], old_home, home_province)
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


def resolve_province_from_request():
    """province query param — default khonkaen ถ้าไม่ส่งมา แต่ raise ValueError ถ้าส่งมาไม่ตรง
    registry (route คืน 400) ไม่ silent fallback เพราะจะกลบบั๊ก typo ฝั่ง frontend"""
    province = request.args.get("province", DEFAULT_PROVINCE)
    if province not in PROVINCES_BY_SLUG:
        raise ValueError(f"ไม่รู้จักจังหวัด: {province}")
    return province


@app.route("/api/provinces", methods=["GET"])
def api_provinces():
    return jsonify([
        {"province_th": p["province_th"], "province_en": p["province_en"],
         "lat": p["lat"], "lon": p["lon"], "is_zero_shot": p["is_zero_shot"]}
        for p in PROVINCE_REGISTRY
    ])


@app.route("/api/provinces/overview", methods=["GET"])
def api_provinces_overview():
    horizon_raw = request.args.get("horizon", "0")
    try:
        horizon = int(horizon_raw)
        if not (0 <= horizon <= 24):
            raise ValueError
    except ValueError:
        return jsonify({"error": f"horizon ต้องเป็นจำนวนเต็ม 0-24: {horizon_raw}"}), 400

    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT DISTINCT ON (province) province, recorded_at_utc, latest_pm25
            FROM notification_log
            ORDER BY province, recorded_at_utc DESC
        """)
        current_by_province = {province: (recorded_at_utc, latest_pm25) for province, recorded_at_utc, latest_pm25 in cur.fetchall()}

        forecast_by_province = {}
        if horizon > 0:
            cur.execute("""
                SELECT DISTINCT ON (province) province, target_time_utc, predicted_pm25, model_used
                FROM prediction_log
                WHERE horizon = %s
                ORDER BY province, predicted_at_utc DESC
            """, (horizon,))
            forecast_by_province = {
                province: (target_time_utc, predicted_pm25, model_used)
                for province, target_time_utc, predicted_pm25, model_used in cur.fetchall()
            }

        cur.close()
        conn.close()

        results = []
        for p in PROVINCE_REGISTRY:
            row = {
                "province_th": p["province_th"], "province_en": p["province_en"],
                "lat": p["lat"], "lon": p["lon"], "is_zero_shot": p["is_zero_shot"],
                "horizon": horizon,
                "current_pm25": None, "current_status": None, "current_recorded_at_th": None,
                "forecast_pm25": None, "forecast_status": None,
                "forecast_target_time_th": None, "forecast_model_used": None,
            }
            current_snapshot = current_by_province.get(p["province_en"])
            if current_snapshot is not None:
                recorded_at_utc, latest_pm25 = current_snapshot
                row["current_pm25"] = round(float(latest_pm25), 1)
                row["current_status"] = pm25_status(latest_pm25)
                row["current_recorded_at_th"] = pd.Timestamp(recorded_at_utc).tz_convert("Asia/Bangkok").strftime("%Y-%m-%d %H:%M")

            forecast_snapshot = forecast_by_province.get(p["province_en"])
            if forecast_snapshot is not None:
                target_time_utc, predicted_pm25, model_used = forecast_snapshot
                row["forecast_pm25"] = round(float(predicted_pm25), 1)
                row["forecast_status"] = pm25_status(predicted_pm25)
                row["forecast_target_time_th"] = pd.Timestamp(target_time_utc).tz_convert("Asia/Bangkok").strftime("%Y-%m-%d %H:%M")
                row["forecast_model_used"] = model_used

            results.append(row)

        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/forecast", methods=["GET"])
def api_forecast():
    try:
        province = resolve_province_from_request()
        result = get_forecast(province, resolve_risk_group_from_session())
        log_target_latency(province, result["data_quality"])
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except DataUnavailableError as e:
        return jsonify({"error": str(e), "data_unavailable": True}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


SHAP_API_TOP_N = 5  # พอให้มีโอกาสเห็นทั้งปัจจัยดันขึ้น/ลดลงพร้อมกัน (ใช้กับการ์ดสรุปแยก 2 ฝั่งฝั่ง frontend)


@app.route("/api/forecast/shap", methods=["GET"])
def api_forecast_shap():
    """อธิบายค่าพยากรณ์ของ "ชั่วโมงเดียว" ที่ผู้ใช้เลือกบน Dashboard ด้วย SHAP (ตอบคอมเมนต์กรรมการ:
    Actual vs Predicted ต้องมีอิทธิพลจาก SHAP กำกับ ไม่ใช่แค่ค่าพยากรณ์เฉยๆ) — ตั้งใจแยกจาก /api/forecast
    เพราะ SHAP มีต้นทุนคำนวณสูง (ดู REP_HORIZON_FOR_SHAP ด้านบน: daily NLG เองก็อธิบายแค่ 1 horizon
    ตัวแทน ไม่ใช่ทั้ง 24 ชม.) คำนวณเฉพาะ horizon ที่ผู้ใช้แตะดูจริงเท่านั้น ไม่รวมอยู่ใน /api/forecast"""
    try:
        province = resolve_province_from_request()
        horizon = request.args.get("horizon", type=int)
        if horizon is None or not (1 <= horizon <= 24):
            return jsonify({"error": "horizon ต้องเป็นตัวเลข 1-24"}), 400

        ctx = build_live_feature_row(province)
        r2_lookup = fetch_rolling_r2_lookup(province)
        model_used = select_model(ctx["X_now"], horizon, province, r2_lookup, ctx["force_naive"])

        if model_used == MODEL_NAIVE:
            return jsonify({"model_used": model_used, "horizon": horizon, "factors": None})

        shap_pairs = get_shap_explanation(model_used, horizon, ctx["X_now"])
        factors = []
        for name, val in shap_pairs[:SHAP_API_TOP_N]:
            direction = "up" if val > 0 else "down"
            factors.append({
                "feature": name, "feature_th": _feature_name_th(name),
                "narrative_th": _factor_narrative_th(name, direction),
                "shap_value": round(float(val), 3), "direction": direction,
            })
        return jsonify({"model_used": model_used, "horizon": horizon, "factors": factors})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except DataUnavailableError as e:
        return jsonify({"error": str(e), "data_unavailable": True}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/current_pm25", methods=["GET"])
def api_current_pm25():
    try:
        province = resolve_province_from_request()
        # Air4Thai สดกว่า OpenAQ มาก (~28 นาที vs ~2.5 ชม. วัดจริง) ใช้ก่อนถ้ามี — ล้มเหลว/ไม่มีจังหวัดนี้
        # ค่อย fallback ไป OpenAQ เดิมด้านล่าง (พฤติกรรมเดิมเป๊ะ ไม่กระทบ error path/503 เดิม)
        live = get_air4thai_live(province)
        if live is not None:
            return jsonify({
                "pm25": round(live["pm25"], 1),
                "datetime_th": live["datetime_utc"].tz_convert("Asia/Bangkok").strftime("%Y-%m-%d %H:%M"),
                "status": pm25_status(live["pm25"]),
            })

        df = get_recent_series(province, strict=True, max_age_s=TARGET_CACHE_TTL_S)
        if df is None or df.empty:
            return jsonify({"error": f"ไม่มีข้อมูล PM2.5 ล่าสุดของ {province} จาก OpenAQ (สถานีอาจขัดข้อง)",
                             "data_unavailable": True}), 503
        latest = df.iloc[-1]
        latest_time_th = pd.Timestamp(latest["datetime_utc"]).tz_convert("Asia/Bangkok")
        return jsonify({
            "pm25": round(float(latest["pm25"]), 1),
            "datetime_th": latest_time_th.strftime("%Y-%m-%d %H:%M"),
            "status": pm25_status(latest["pm25"]),
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500



NOTIF_SCAN_WINDOW = 500

# ข้อ 1 "ระดับเปลี่ยนสถานะ" — กันไม่ให้ค่าที่แกว่งไปมาระหว่าง 2 ระดับติดกันพอดี (เช่นเด้งข้ามเพดานทุกชั่วโมง)
# ยิงแจ้งเตือนซ้ำทุกครั้ง — scheduler รันทุก 1 ชม. จึงนับ "แถว" แทน "ชั่วโมง" ได้ตรงๆ
LEVEL_TRANSITION_DEBOUNCE_ROWS = 3


def _recent_levels(snapshots, start_idx, count):
    """ระดับ PM2.5 (ชื่อสถานะ) ของ snapshots[start_idx : start_idx+count] — ใช้เช็ค debounce"""
    return {pm25_status(snapshots[j][1]) for j in range(start_idx, min(start_idx + count, len(snapshots)))}


def _level_transition_for(snapshots, idx):
    """transition ของ snapshots[idx] เทียบกับแถวก่อนหน้า (idx+1 เพราะ snapshots เรียง DESC) พร้อม debounce —
    คืน None ถ้าไม่มีแถวก่อนหน้า, ไม่มีการเปลี่ยนระดับ, หรือเปลี่ยนไปหาระดับที่เพิ่งเจอเมื่อไม่กี่แถวที่แล้ว"""
    if idx + 1 >= len(snapshots):
        return None
    transition = level_transition(snapshots[idx][1], snapshots[idx + 1][1])
    if transition is None:
        return None
    recent = _recent_levels(snapshots, idx + 1, LEVEL_TRANSITION_DEBOUNCE_ROWS)
    if transition["to_level"] in recent:
        return None
    return transition


def _station_silence_for(recorded_at_utc, now_utc=None):
    """ข้อ 4 'สถานีเงียบนานผิดปกติ' — คืน (stale_hours, data_stale) เทียบ TARGET_STALE_HOURS เดิม
    now_utc รับ override ได้ (สำหรับเทส) ปกติใช้เวลาจริง ณ ตอนเรียก"""
    if now_utc is None:
        now_utc = pd.Timestamp.now(tz="UTC")
    stale_hours = (now_utc - pd.Timestamp(recorded_at_utc)).total_seconds() / 3600
    return stale_hours, stale_hours > TARGET_STALE_HOURS


def _horizon_display_time(recorded_at_utc, horizon):
    """เวลาที่ควรโชว์ผู้ใช้สำหรับ horizon นี้ นับจาก recorded_at_utc (= latest_time ตอน log_notification_snapshot
    เขียนแถวนี้) บวก OPENAQ_DISPLAY_LABEL_SHIFT_H เหมือนที่ forecast[].display_time ใน get_forecast() ใช้ —
    ให้ตรงกับเวลาที่กราฟพยากรณ์หลักโชว์เป๊ะ ไม่ใช่คนละ convention กัน"""
    t = pd.Timestamp(recorded_at_utc) + pd.Timedelta(hours=horizon + OPENAQ_DISPLAY_LABEL_SHIFT_H)
    return t.tz_convert("Asia/Bangkok").strftime("%H:%M")


def _imminent_risk_for(horizon_rows, risk_group, recorded_at_utc):
    """ข้อ 2 'ใกล้ถึงช่วงเสี่ยงแล้ว' — คืน (risk_horizons, fallback_horizons, risk_imminent,
    min_risk_horizon, min_risk_horizon_pm25, min_risk_horizon_display_time) จาก horizon_rows ที่ query มาแล้ว
    (ไม่ query ซ้ำ)"""
    risk_horizons = sorted(h for h, pred, model_used in horizon_rows if is_horizon_at_risk(pred, risk_group))
    fallback_horizons = sorted(h for h, _, model_used in horizon_rows if model_used == MODEL_NAIVE)
    pm25_by_horizon = {h: pred for h, pred, _ in horizon_rows}
    min_risk_horizon = min(risk_horizons) if risk_horizons else None
    risk_imminent = min_risk_horizon is not None and min_risk_horizon <= IMMINENT_HORIZON_H
    min_risk_horizon_pm25 = pm25_by_horizon.get(min_risk_horizon) if min_risk_horizon is not None else None
    min_risk_horizon_display_time = (
        _horizon_display_time(recorded_at_utc, min_risk_horizon) if min_risk_horizon is not None else None
    )
    return (risk_horizons, fallback_horizons, risk_imminent, min_risk_horizon, min_risk_horizon_pm25,
            min_risk_horizon_display_time)


# ===== Web Push notifications =====
# ส่ง push ผ่าน VAPID มาตรฐาน Web Push — subscription เก็บใน push_subscriptions (endpoint คือ identity ตาม
# Web Push spec) payload ที่ส่งไปเป็น JSON {title, body, tag, url} รับที่ frontend/src/sw.js push handler

def send_push(endpoint, p256dh, auth, title, body, tag, url="/notifications"):
    """ส่ง push 1 ครั้ง คืน True ถ้าสำเร็จ — ถ้า subscription หมดอายุ (404/410) ลบแถวทิ้งแล้วคืน False เงียบๆ
    ไม่ throw ออกไปให้ caller ต้อง try/except เอง (ผู้โทรเป็น scheduler loop วนหลาย subscription ต่อรอบ)"""
    try:
        webpush(
            subscription_info={"endpoint": endpoint, "keys": {"p256dh": p256dh, "auth": auth}},
            data=json.dumps({"title": title, "body": body, "tag": tag, "url": url}),
            vapid_private_key=VAPID_PRIVATE_KEY_PATH,
            vapid_claims={"sub": VAPID_CLAIMS_SUB},
            # ttl=0 (default ของ pywebpush) ชนกับ WNS (endpoint notify.windows.com — Chrome/Edge บน
            # Windows desktop ใช้): WNS ตอบ 400 "Ttl value conflicts with X-WNS-Cache-Policy." เสมอถ้า
            # ttl=0 — endpoint อื่น (FCM/APNs) รับ ttl=0 ได้ปกติ แต่ตั้งเป็นค่าบวกไว้เลยให้ใช้ได้ทุก endpoint
            ttl=60,
        )
        return True
    except WebPushException as e:
        if e.response is not None and e.response.status_code in (404, 410):
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("DELETE FROM push_subscriptions WHERE endpoint = %s", (endpoint,))
            conn.commit()
            cur.close()
            conn.close()
        else:
            print(f"[push] ส่งไม่สำเร็จ ({endpoint[:40]}...): {e}")
        return False


_PUSH_WARNING_ADVICE_BY_RISK_GROUP = {
    "ทั่วไป": "โปรดระมัดระวังเพิ่มเติม",
    "เด็ก-ผู้สูงอายุ": "กลุ่มเด็ก/ผู้สูงอายุควรหลีกเลี่ยงกิจกรรมกลางแจ้งเป็นเวลานานและระมัดระวังเป็นพิเศษ",
    "มีโรคประจำตัว": "ผู้มีโรคประจำตัวทางเดินหายใจควรหลีกเลี่ยงกิจกรรมกลางแจ้งและเตรียมอุปกรณ์ป้องกัน",
}


def _push_payload_for(condition, risk_group, transition, min_risk_horizon, min_risk_horizon_pm25,
                       min_risk_horizon_display_time, latest_pm25, trend_period, stale_hours):
    """title/body ของ push ต่อเงื่อนไข — เลียนข้อความจาก frontend/src/notifications.js builder เป๊ะ
    (duplicate เป็น Python ด้วยมือเพราะไม่มี module ร่วมระหว่าง JS/Python ในโปรเจกต์นี้)"""
    if condition == "threshold":
        advice = _PUSH_WARNING_ADVICE_BY_RISK_GROUP.get(risk_group, _PUSH_WARNING_ADVICE_BY_RISK_GROUP["ทั่วไป"])
        return "แจ้งเตือนคุณภาพอากาศ", (
            f"ช่วงนี้ PM2.5 อยู่ในระดับสูง ค่าพยากรณ์อาจต่ำกว่าความจริงเล็กน้อยในช่วงมลพิษสูง {advice}"
        )
    if condition == "transition":
        direction_word = "แย่ลง" if transition["direction"] == "worse" else "ดีขึ้น"
        magnitude = "มาก" if transition["levels_moved"] >= 2 else "เล็กน้อย"
        return "ระดับคุณภาพอากาศเปลี่ยน", (
            f'อากาศ{direction_word}{magnitude} จาก "{transition["from_level"]}" เป็น "{transition["to_level"]}" '
            f'({transition["to_pm25"]} µg/m³)'
        )
    if condition == "imminent":
        return "ใกล้ถึงช่วงเสี่ยงแล้ว", (
            f"ฝุ่นจะเกินเกณฑ์ {min_risk_horizon_display_time} น. (อีก {min_risk_horizon} ชม.) "
            f"คาดการณ์ {min_risk_horizon_pm25} µg/m³"
        )
    if condition == "trend":
        return "แนวโน้มฝุ่นดีขึ้น", f"แนวโน้มช่วง{trend_period}ฝุ่นจะลดลง"
    if condition == "silence":
        return "ไม่มีข้อมูลสถานีใหม่", f"ไม่มีข้อมูลสถานีใหม่มา {round(stale_hours)} ชั่วโมงแล้ว"
    return "แจ้งเตือนคุณภาพอากาศ", ""


def dispatch_push_for_province(province, now_utc=None):
    """เรียกจาก scheduled_forecast_job() ทุกจังหวัดทุกชั่วโมง (ไม่ใช่จาก get_forecast()/live request — ดู
    เหตุผลในแผน) self-contained อ่าน notification_log/prediction_log ตรงๆ (query เดียวกับ
    api_notifications_status()) ไม่ต้องแก้ signature ของ get_forecast() เลย

    now_utc: รับ override ได้ (ส่งต่อให้ _station_silence_for()) สำหรับเทสเท่านั้น ปกติใช้เวลาจริง ณ ตอนเรียก

    Dedup: threshold/silence ใช้ sentinel คงที่ "active" (ไม่ใช่ค่าที่แกว่งได้ เช่น pm25/stale_hours ดิบ)
    เพราะเงื่อนไขพวกนี้ "ค้าง" ได้หลายชั่วโมงติดกัน ถ้าใช้ค่าดิบจะ push ซ้ำทุกชั่วโมงที่ค่าขยับนิดเดียว —
    transition เคลียร์ตัวเองเป็น None อัตโนมัติอยู่แล้วหลังผ่านไป 1 ชม. (ดู _level_transition_for) ส่วน
    imminent/trend ใช้ค่าจริง (horizon/period) ได้เพราะเปลี่ยนแค่ไม่กี่ครั้งต่อเหตุการณ์ ไม่ใช่ทุกชั่วโมง"""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        "SELECT id, endpoint, p256dh, auth, user_id, push_dedup_state FROM push_subscriptions WHERE province = %s",
        (province,),
    )
    subscriptions = cur.fetchall()
    if not subscriptions:
        cur.close()
        conn.close()
        return

    cur.execute(
        "SELECT recorded_at_utc, latest_pm25, trend_direction, trend_period FROM notification_log "
        "WHERE province = %s ORDER BY recorded_at_utc DESC LIMIT %s",
        (province, 1 + LEVEL_TRANSITION_DEBOUNCE_ROWS),
    )
    snapshots = cur.fetchall()
    if not snapshots:
        cur.close()
        conn.close()
        return

    recorded_at_utc, latest_pm25, trend_direction, trend_period = snapshots[0]
    transition = _level_transition_for(snapshots, 0)
    stale_hours, data_stale = _station_silence_for(recorded_at_utc, now_utc=now_utc)
    transition_value = f'{transition["from_level"]}->{transition["to_level"]}' if transition else None
    trend_value = trend_period if trend_direction == "down" else None
    silence_value = "active" if data_stale else None

    cur.execute(
        "SELECT horizon, predicted_pm25, model_used FROM prediction_log "
        "WHERE predicted_at_utc = %s AND province = %s",
        (recorded_at_utc, province),
    )
    horizon_rows = cur.fetchall()

    for sub_id, endpoint, p256dh, auth, user_id, dedup_state in subscriptions:
        risk_group = "ทั่วไป"
        if user_id is not None:
            user = get_user_by_id(user_id)
            if user is not None:
                risk_group = user["risk_group"]

        (risk_horizons, _fallback_horizons, risk_imminent, min_risk_horizon, min_risk_horizon_pm25,
         min_risk_horizon_display_time) = _imminent_risk_for(horizon_rows, risk_group, recorded_at_utc)
        current_at_risk = is_current_at_risk(latest_pm25, risk_group)
        threshold_value = "active" if (current_at_risk or bool(risk_horizons)) else None
        imminent_value = min_risk_horizon if risk_imminent else None

        events = {
            "threshold": threshold_value,
            "transition": transition_value,
            "imminent": imminent_value,
            "trend": trend_value,
            "silence": silence_value,
        }

        new_state = dict(dedup_state or {})
        changed = False
        for key, value in events.items():
            if new_state.get(key) == value:
                continue
            if value is None:
                new_state[key] = value  # เงื่อนไขหายไปแล้ว — แค่เคลียร์ state ไม่ต้องส่ง push
                changed = True
                continue
            title, body = _push_payload_for(
                key, risk_group, transition, min_risk_horizon, min_risk_horizon_pm25,
                min_risk_horizon_display_time, latest_pm25, trend_period, stale_hours,
            )
            # อัปเดต state เฉพาะตอนส่งสำเร็จเท่านั้น — ถ้าพลาด (เครือข่าย/push service ขัดข้องชั่วคราว) ปล่อย
            # ให้ลองใหม่รอบ scheduler ถัดไปแทนที่จะเงียบหายไปเฉยๆ
            if send_push(endpoint, p256dh, auth, title, body, key):
                new_state[key] = value
                changed = True

        if changed:
            cur2 = conn.cursor()
            cur2.execute(
                "UPDATE push_subscriptions SET push_dedup_state = %s::jsonb WHERE id = %s",
                (json.dumps(new_state), sub_id),
            )
            conn.commit()
            cur2.close()

    cur.close()
    conn.close()


@app.route("/api/push/vapid-public-key", methods=["GET"])
def api_push_vapid_public_key():
    return jsonify({"public_key": VAPID_PUBLIC_KEY})


@app.route("/api/push/subscribe", methods=["POST"])
def api_push_subscribe():
    try:
        data = request.get_json(silent=True) or {}
        endpoint = str(data.get("endpoint", "")).strip()
        keys = data.get("keys") or {}
        p256dh = str(keys.get("p256dh", "")).strip()
        auth = str(keys.get("auth", "")).strip()
        province = str(data.get("province", "")).strip()

        if not endpoint or not p256dh or not auth:
            return jsonify({"error": "ข้อมูล subscription ไม่ครบ"}), 400
        if province not in PROVINCES_BY_SLUG:
            return jsonify({"error": "ไม่พบจังหวัดนี้"}), 400

        user_id = session.get("user_id")  # None = guest (ไม่บังคับ login เหมือน /api/notifications)
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """INSERT INTO push_subscriptions (endpoint, p256dh, auth, user_id, province, created_at, push_dedup_state)
               VALUES (%s, %s, %s, %s, %s, %s, '{}'::jsonb)
               ON CONFLICT (endpoint) DO UPDATE SET
                 p256dh = EXCLUDED.p256dh, auth = EXCLUDED.auth, user_id = EXCLUDED.user_id,
                 province = EXCLUDED.province, push_dedup_state = '{}'::jsonb""",
            (endpoint, p256dh, auth, user_id, province, pd.Timestamp.now(tz="UTC").isoformat()),
        )
        conn.commit()
        cur.close()
        conn.close()
        return jsonify({"ok": True}), 201
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/push/status", methods=["POST"])
def api_push_status():
    """จังหวัดที่ subscription ของเครื่องนี้ล็อกไว้จริง (หน้าตั้งค่าโชว์ค่านี้ ไม่ใช่จังหวัดที่กำลังดูอยู่)
    POST ไม่ใช่ GET เพื่อไม่ให้ endpoint URL ของ push service ไปโผล่ใน query string/access log
    ไม่บังคับ login เหมือน /api/push/subscribe — endpoint เองเป็นตัวระบุเครื่อง"""
    try:
        data = request.get_json(silent=True) or {}
        endpoint = str(data.get("endpoint", "")).strip()
        if not endpoint:
            return jsonify({"error": "ไม่มี endpoint"}), 400
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT province FROM push_subscriptions WHERE endpoint = %s", (endpoint,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row is None:
            return jsonify({"subscribed": False, "province": None}), 200
        return jsonify({"subscribed": True, "province": row[0]}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def move_push_subscriptions_with_home(user_id, old_home, new_home):
    """ผู้ใช้เปลี่ยนจังหวัดหลัก -> ย้าย push ของทุกเครื่องในบัญชีนี้ไปจังหวัดใหม่ แต่เฉพาะ subscription ที่ยังล็อก
    อยู่ที่จังหวัดหลักเดิม (province = old_home) — ถ้าผู้ใช้เลือกจังหวัดแจ้งเตือนเองเป็นจังหวัดอื่นไว้ในหน้าตั้งค่า
    จะไม่ทับ; รีเซ็ต push_dedup_state เพราะเป็นจังหวัดใหม่ (เหมือนตอน subscribe ใหม่)
    ห่อ try/except: ย้าย push ไม่สำเร็จต้องไม่ทำให้การบันทึกโปรไฟล์ล้ม"""
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute(
            """UPDATE push_subscriptions SET province = %s, push_dedup_state = '{}'::jsonb
               WHERE user_id = %s AND province = %s""",
            (new_home, user_id, old_home),
        )
        moved = cur.rowcount
        conn.commit()
        cur.close()
        conn.close()
        if moved:
            print(f"[push] moved {moved} subscription(s) of user {user_id}: {old_home} -> {new_home}")
    except Exception as e:
        print(f"[push] move subscriptions failed for user {user_id}: {type(e).__name__}")


@app.route("/api/push/unsubscribe", methods=["POST"])
def api_push_unsubscribe():
    try:
        data = request.get_json(silent=True) or {}
        endpoint = str(data.get("endpoint", "")).strip()
        if endpoint:
            conn = get_db_connection()
            cur = conn.cursor()
            cur.execute("DELETE FROM push_subscriptions WHERE endpoint = %s", (endpoint,))
            conn.commit()
            cur.close()
            conn.close()
        return jsonify({"ok": True}), 200  # idempotent เสมอ แม้ไม่มีแถวตรงให้ลบ
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/push/test", methods=["POST"])
def api_push_test():
    """ทดสอบส่ง push ตรงๆ ไม่ต้องรอ scheduler รอบถัดไป — ใช้ตอน dev เท่านั้น"""
    if not DEBUG:
        return jsonify({"error": "not available"}), 404
    try:
        data = request.get_json(silent=True) or {}
        endpoint = str(data.get("endpoint", "")).strip()
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT p256dh, auth FROM push_subscriptions WHERE endpoint = %s", (endpoint,))
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row is None:
            return jsonify({"error": "ไม่พบ subscription นี้"}), 404
        p256dh, auth = row
        ok = send_push(endpoint, p256dh, auth, "ทดสอบการแจ้งเตือน",
                        "นี่คือข้อความทดสอบจากระบบพยากรณ์ PM2.5", "test")
        return jsonify({"ok": ok})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/notifications", methods=["GET"])
def api_notifications():
    try:
        province = resolve_province_from_request()
        risk_group = resolve_risk_group_from_session()
        limit = request.args.get("limit", default=20, type=int)
        limit = max(1, min(limit, 100))

        conn = get_db_connection()
        cur = conn.cursor()

        cur.execute(
            "SELECT recorded_at_utc, latest_pm25, trend_direction, trend_period FROM notification_log "
            "WHERE province = %s ORDER BY recorded_at_utc DESC LIMIT %s",
            (province, NOTIF_SCAN_WINDOW),
        )
        snapshots = cur.fetchall()
        # _level_transition_for()/_recent_levels() อ่านแค่ (recorded_at_utc, latest_pm25) สองคอลัมน์แรกของแต่ละ
        # แถว ใช้ snapshots ตัวเต็มนี้ตรงๆ ได้เลยไม่ต้อง slice

        results = []
        for i, (recorded_at_utc, latest_pm25, trend_direction, trend_period) in enumerate(snapshots):
            cur.execute(
                "SELECT horizon, predicted_pm25, model_used FROM prediction_log "
                "WHERE predicted_at_utc = %s AND province = %s",
                (recorded_at_utc, province),
            )
            horizon_rows = cur.fetchall()

            (risk_horizons, fallback_horizons, risk_imminent, min_risk_horizon, min_risk_horizon_pm25,
             min_risk_horizon_display_time) = _imminent_risk_for(horizon_rows, risk_group, recorded_at_utc)
            transition = _level_transition_for(snapshots, i)
            improving = trend_direction == "down"

            # เดิมโชว์แค่แถวที่ "เสี่ยง" เท่านั้น — ตอนนี้โชว์แถวที่มีเหตุการณ์น่าสนใจแบบใดแบบหนึ่งใน 4 แบบด้วย
            # (ระดับเปลี่ยน/เสี่ยงจริง/แนวโน้มดีขึ้น) ข้อ 4 "สถานีเงียบ" ไม่เกี่ยวกับ endpoint นี้ (ดูคอมเมนต์ที่
            # api_notifications_status())
            current_at_risk = is_current_at_risk(latest_pm25, risk_group)
            if not (current_at_risk or risk_horizons or transition or improving):
                continue

            recorded_at_th = pd.Timestamp(recorded_at_utc).tz_convert("Asia/Bangkok")
            results.append({
                "recorded_at_th": recorded_at_th.strftime("%Y-%m-%d %H:%M"),
                "latest_pm25": round(float(latest_pm25), 1),
                "current_at_risk": current_at_risk,
                "risk_horizons": risk_horizons,
                "fallback_horizons": fallback_horizons,
                "risk_imminent": risk_imminent,
                "min_risk_horizon": min_risk_horizon,
                "min_risk_horizon_pm25": min_risk_horizon_pm25,
                "min_risk_horizon_display_time": min_risk_horizon_display_time,
                "transition": transition,
                "trend_direction": trend_direction,
                "trend_period": trend_period,
            })
            if len(results) >= limit:
                break

        cur.close()
        conn.close()
        return jsonify(results)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/notifications/status", methods=["GET"])
def api_notifications_status():
    try:
        province = resolve_province_from_request()
        risk_group = resolve_risk_group_from_session()

        conn = get_db_connection()
        cur = conn.cursor()
        # LIMIT 1 ของจริง + LEVEL_TRANSITION_DEBOUNCE_ROWS แถวย้อนหลังเพิ่ม เพื่อให้ _level_transition_for()
        # เช็ค debounce กับแถวปัจจุบัน (index 0) ได้ครบเหมือนที่ api_notifications() ทำ
        cur.execute(
            "SELECT recorded_at_utc, latest_pm25, trend_direction, trend_period FROM notification_log "
            "WHERE province = %s ORDER BY recorded_at_utc DESC LIMIT %s",
            (province, 1 + LEVEL_TRANSITION_DEBOUNCE_ROWS),
        )
        snapshots = cur.fetchall()

        if not snapshots:
            cur.close()
            conn.close()
            return jsonify({
                "high_pollution_risk": False, "risk_horizons": [], "fallback_horizons": [],
                "risk_imminent": False, "min_risk_horizon": None, "min_risk_horizon_pm25": None,
                "min_risk_horizon_display_time": None,
                "transition": None, "trend_direction": None, "trend_period": None,
                "data_stale": None, "stale_hours": None,
                "current_pm25": None, "current_status": None, "recorded_at_th": None,
                "risk_group": risk_group,
            })

        recorded_at_utc, latest_pm25, trend_direction, trend_period = snapshots[0]
        cur.execute(
            "SELECT horizon, predicted_pm25, model_used FROM prediction_log "
            "WHERE predicted_at_utc = %s AND province = %s",
            (recorded_at_utc, province),
        )
        horizon_rows = cur.fetchall()
        cur.close()
        conn.close()

        (risk_horizons, fallback_horizons, risk_imminent, min_risk_horizon, min_risk_horizon_pm25,
         min_risk_horizon_display_time) = _imminent_risk_for(horizon_rows, risk_group, recorded_at_utc)
        transition = _level_transition_for(snapshots, 0)
        high_pollution_risk = is_current_at_risk(latest_pm25, risk_group) or bool(risk_horizons)
        recorded_at_th = pd.Timestamp(recorded_at_utc).tz_convert("Asia/Bangkok")

        # ข้อ 4 "สถานีเงียบนานผิดปกติ" — อนุมานจากช่องว่างของเวลา ไม่ใช่ field ใหม่ที่ต้องเขียนตอน log เพราะ
        # ช่วงที่ไม่ loggable (target_stale/imputed/fallback) ก็คือช่วงที่ไม่มีแถวใหม่เข้ามาอยู่แล้วโดยธรรมชาติ
        stale_hours, data_stale = _station_silence_for(recorded_at_utc)

        return jsonify({
            "high_pollution_risk": high_pollution_risk,
            "risk_horizons": risk_horizons,
            "fallback_horizons": fallback_horizons,
            "risk_imminent": risk_imminent,
            "min_risk_horizon": min_risk_horizon,
            "min_risk_horizon_pm25": min_risk_horizon_pm25,
            "min_risk_horizon_display_time": min_risk_horizon_display_time,
            "transition": transition,
            "trend_direction": trend_direction,
            "trend_period": trend_period,
            "data_stale": data_stale,
            "stale_hours": round(stale_hours, 1),
            "current_pm25": round(float(latest_pm25), 1),
            "current_status": pm25_status(latest_pm25),
            "recorded_at_th": recorded_at_th.strftime("%Y-%m-%d %H:%M"),
            "risk_group": risk_group,
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/comparison", methods=["GET"])
def api_comparison():
    try:
        province = resolve_province_from_request()

        horizon = int(request.args.get("horizon", 1))
        if not 1 <= horizon <= 24:
            return jsonify({"error": "horizon ต้องอยู่ระหว่าง 1-24"}), 400

        model = request.args.get("model", "all")
        # ชื่อรุ่นเก่า (lightgbm_pooled/xgboost_pooled) ยังรับอยู่ เพราะแถวเก่าใน prediction_log ใช้ชื่อเหล่านี้
        valid_models = ("all", MODEL_LGB, MODEL_XGB, "lightgbm_pooled_v2", "xgboost_pooled_v2", "lightgbm_pooled", "xgboost_pooled", MODEL_NAIVE)
        if model not in valid_models:
            return jsonify({"error": "model ต้องเป็น " + "/".join(valid_models)}), 400

        conn = get_db_connection()
        cur = conn.cursor()
        # ไม่มี calibration อีกต่อไป — actual_pm25_calibrated เท่ากับ actual_pm25 เสมอสำหรับแถวใหม่
        
        calibrated_expr = "COALESCE(actual_pm25_calibrated, actual_pm25)"
        if model == "all":
            cur.execute(
                f"""SELECT target_time_utc, predicted_pm25, actual_pm25, {calibrated_expr},
                          model_used, source, pm25_lo, pm25_hi
                   FROM prediction_log
                   WHERE horizon = %s AND province = %s
                   ORDER BY target_time_utc""",
                (horizon, province),
            )
        else:
            cur.execute(
                f"""SELECT target_time_utc, predicted_pm25, actual_pm25, {calibrated_expr},
                          model_used, source, pm25_lo, pm25_hi
                   FROM prediction_log
                   WHERE horizon = %s AND model_used = %s AND province = %s
                   ORDER BY target_time_utc""",
                (horizon, model, province),
            )
        rows = cur.fetchall()
        cur.close()
        conn.close()

        points = []
        errors = []
        live_errors = []
        in_interval_flags = []
        for target_time_str, predicted, actual, actual_calibrated, model_used, source, pm25_lo, pm25_hi in rows:
            target_time_th = pd.Timestamp(target_time_str).tz_convert("Asia/Bangkok")
            in_interval = None
            if actual_calibrated is not None and pm25_lo is not None and pm25_hi is not None and model_used != "naive_fallback":
                in_interval = pm25_lo <= actual_calibrated <= pm25_hi
                in_interval_flags.append(in_interval)
            points.append({
                "time": target_time_th.strftime("%m-%d %H:%M"),
                "predicted": round(predicted, 1),
                "actual": round(actual_calibrated, 1) if actual_calibrated is not None else None,
                "actual_raw": round(actual, 1) if actual is not None else None,
                "model_used": model_used,
                "source": source,
                "pm25_lo": round(pm25_lo, 1) if pm25_lo is not None else None,
                "pm25_hi": round(pm25_hi, 1) if pm25_hi is not None else None,
                "in_interval": in_interval,
            })
            if actual_calibrated is not None:
                err = actual_calibrated - predicted
                errors.append(err)
                if source != "backfill":
                    live_errors.append(err)

        n_matched = len(errors)
        n_backfill = sum(1 for p in points if p["source"] == "backfill")

        # เลือกโมเดลเจาะจง -> ใช้ live-only สำหรับ MAE/bias ถ้ามีพอ (ดู summarize_errors) "all" ไม่ apply
        if model != "all":
            mae, bias, n_used, basis = summarize_errors(errors, live_errors)
        else:
            mae = sum(abs(e) for e in errors) / n_matched if n_matched else None
            bias = sum(errors) / n_matched if n_matched else None
            n_used, basis = n_matched, None

        coverage_pct = round(100 * sum(in_interval_flags) / len(in_interval_flags), 1) if in_interval_flags else None

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
                "coverage_pct": coverage_pct,
                "n_coverage": len(in_interval_flags),
            },
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/comparison/overview", methods=["GET"])
def api_comparison_overview():
    try:
        province = resolve_province_from_request()

        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT COUNT(*) FROM prediction_log WHERE province = %s", (province,))
        n_total = cur.fetchone()[0]
        cur.execute(
            """SELECT horizon, predicted_pm25, COALESCE(actual_pm25_calibrated, actual_pm25),
                      model_used, source, pm25_lo, pm25_hi
               FROM prediction_log WHERE actual_pm25 IS NOT NULL AND province = %s""",
            (province,),
        )
        rows = cur.fetchall()
        cur.close()
        conn.close()

        by_horizon_errors = {h: {"all": [], "live": []} for h in range(1, 25)}
        by_horizon_coverage = {h: [] for h in range(1, 25)}
        by_model_errors = {}
        all_errors, live_errors = [], []
        overall_coverage_flags = []
        for horizon, predicted, actual, model_used, source, pm25_lo, pm25_hi in rows:
            err = actual - predicted
            is_live = source != "backfill"

            all_errors.append(err)
            by_horizon_errors[horizon]["all"].append(err)
            by_model_errors.setdefault(model_used, {"all": [], "live": [], "n_backfill": 0, "n_live": 0, "coverage": []})
            by_model_errors[model_used]["all"].append(err)

            # coverage: เหมือน /api/comparison (ดูคอมเมนต์ที่นั่น)
            if pm25_lo is not None and pm25_hi is not None and model_used != "naive_fallback":
                covered = pm25_lo <= actual <= pm25_hi
                overall_coverage_flags.append(covered)
                by_horizon_coverage[horizon].append(covered)
                by_model_errors[model_used]["coverage"].append(covered)

            if is_live:
                live_errors.append(err)
                by_horizon_errors[horizon]["live"].append(err)
                by_model_errors[model_used]["live"].append(err)
                by_model_errors[model_used]["n_live"] += 1
            else:
                by_model_errors[model_used]["n_backfill"] += 1

        n_matched = len(all_errors)
        overall_mae, overall_bias, overall_n_used, overall_basis = summarize_errors(all_errors, live_errors)
        overall_coverage_pct = (
            round(100 * sum(overall_coverage_flags) / len(overall_coverage_flags), 1)
            if overall_coverage_flags else None
        )

        by_horizon = []
        for h in range(1, 25):
            mae_h, _, n_h, basis_h = summarize_errors(by_horizon_errors[h]["all"], by_horizon_errors[h]["live"])
            cov_flags_h = by_horizon_coverage[h]
            by_horizon.append({
                "horizon": h,
                "mae": round(mae_h, 2) if mae_h is not None else None,
                "n": n_h,
                "basis": basis_h,
                "coverage_pct": round(100 * sum(cov_flags_h) / len(cov_flags_h), 1) if cov_flags_h else None,
                "n_coverage": len(cov_flags_h),
            })

        with_data = [b for b in by_horizon if b["mae"] is not None]
        best = min(with_data, key=lambda b: b["mae"]) if with_data else None
        worst = max(with_data, key=lambda b: b["mae"]) if with_data else None

        by_model = []
        for model_used, d in sorted(by_model_errors.items()):
            mae_m, bias_m, n_m, basis_m = summarize_errors(d["all"], d["live"])
            cov_flags_m = d["coverage"]
            by_model.append({
                "model_used": model_used,
                "mae": round(mae_m, 2) if mae_m is not None else None,
                "bias": round(bias_m, 2) if bias_m is not None else None,
                "n": n_m,
                "basis": basis_m,
                "n_backfill": d["n_backfill"],
                "n_live": d["n_live"],
                "coverage_pct": round(100 * sum(cov_flags_m) / len(cov_flags_m), 1) if cov_flags_m else None,
                "n_coverage": len(cov_flags_m),
            })

        return jsonify({
            "n_total": n_total,
            "n_matched": n_matched,
            "overall_mae": round(overall_mae, 2) if overall_mae is not None else None,
            "overall_bias": round(overall_bias, 2) if overall_bias is not None else None,
            "overall_n_used": overall_n_used,
            "overall_basis": overall_basis,
            "overall_coverage_pct": overall_coverage_pct,
            "overall_n_coverage": len(overall_coverage_flags),
            "by_horizon": by_horizon,
            "by_model": by_model,
            "best_horizon": best,
            "worst_horizon": worst,
        })
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":

    if not DEBUG or os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        scheduler = BackgroundScheduler()
        scheduler.add_job(
            scheduled_forecast_job,
            "interval",
            hours=1,
            id="hourly_forecast",
            next_run_time=pd.Timestamp.now(),  
        )
        scheduler.start()
        print("[scheduler] started - auto forecast every 1 hour")

    # PORT ปรับได้ผ่าน env var — ใช้ตอน debug ปัญหา cache ของ service worker บนมือถือ (origin ใหม่ = cache ใหม่
    # เสมอ โดยไม่ต้องง้อการล้าง cache ฝั่งอุปกรณ์ที่บางทีไม่เวิร์กจริงบน iOS Web Clip) ปกติไม่ตั้งค่า = 5000 เหมือนเดิม
    port = int(os.environ.get("PORT", 5000))
    # HTTPS=1 เปิด self-signed cert สำหรับทดสอบ — Notification/PushManager/serviceWorker ต้องการ "secure
    # context" เสมอ (HTTPS หรือ localhost เท่านั้น) ทดสอบผ่าน LAN IP ด้วย http:// ธรรมดาจะไม่มีทาง trigger
    # permission prompt จริงได้เลยไม่ว่าจะแก้โค้ดยังไงก็ตาม
    # ใช้ไฟล์ถาวรจาก scripts/generate_dev_cert.py ถ้ามี (ไม่ต้องกดผ่านคำเตือน cert ซ้ำทุกครั้งที่ restart
    # เหมือน adhoc ที่ Werkzeug สุ่ม cert ใหม่ทุกรอบ) ไม่มีไฟล์ค่อย fallback ไปใช้ adhoc
    _dev_cert = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dev_cert.pem")
    _dev_key = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dev_key.pem")
    if os.environ.get("HTTPS") != "1":
        ssl_context = None
    elif os.path.exists(_dev_cert) and os.path.exists(_dev_key):
        ssl_context = (_dev_cert, _dev_key)
    else:
        ssl_context = "adhoc"
    # LAN=1 เปิดให้อุปกรณ์อื่นในวงแลนเดียวกันเข้าถึงได้ (ทดสอบมือถือ) — ปกติไม่ตั้งค่า = "127.0.0.1" เหมือนเดิม
    # ก่อนเพิ่มงาน Web Push (ปลอดภัยกว่าเป็นค่าเริ่มต้น เพราะ DEBUG=True เปิด Werkzeug interactive debugger
    # อยู่ด้วย เปิดให้ทั้งวงแลนเข้าถึงได้ตลอดเวลาโดยไม่ตั้งใจจะเพิ่มพื้นที่เสี่ยงโดยไม่จำเป็น)
    host = "0.0.0.0" if os.environ.get("LAN") == "1" else "127.0.0.1"
    app.run(host=host, port=port, debug=DEBUG, ssl_context=ssl_context)

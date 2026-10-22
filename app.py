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
HIGH_POLLUTION_THRESHOLD = 37.5  # ค่าเดียวกันทุกจังหวัด (มาตรฐาน คพ. ไม่ใช่ค่าที่ผูกกับโมเดล)
# ต้องรัน `npm run build` ใน frontend/ ก่อน ถึงจะมีโฟลเดอร์นี้ 
FRONTEND_DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "dist")
DEBUG = True  # ต้องอ่านจากตรงนี้ ไม่ใช่ app.debug เพราะ Flask ยังไม่ set app.debug จนกว่าจะเรียก app.run()
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
    """threshold เดียว (37.5 + offset ตาม risk_group) ไม่ใช้ bias CSV แบบเดิม (per-province, เลิกใช้
    ตั้งแต่ย้ายมาโมเดล pooled)"""
    return HIGH_POLLUTION_THRESHOLD + RISK_GROUP_THRESHOLD_OFFSET.get(risk_group, 0.0)



def is_horizon_at_risk(predicted_pm25, risk_group):
    return predicted_pm25 > effective_threshold_for(risk_group)


def is_current_at_risk(latest_pm25, risk_group):

    return bool(latest_pm25 > effective_threshold_for(risk_group))


ACTIVE_HOURS_START_TH = 6
ACTIVE_HOURS_END_TH = 20


def find_best_time(forecast):

    active_hours = [f for f in forecast if ACTIVE_HOURS_START_TH <= f["hour_thai"] <= ACTIVE_HOURS_END_TH]
    return min(active_hours or forecast, key=lambda x: x["pm25"])


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


# ===== Dynamic Routing (proposal 4.1.6) =====

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
    # ตอน query โดย join กับ prediction_log ผ่าน recorded_at_utc=predicted_at_utc แทน (ไม่เก็บตรงๆ
    # เพราะ risk ขึ้นกับ risk_group ที่เปลี่ยนได้ เก็บสดจะผิดสำหรับกลุ่มอื่น)
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


# ===== Live data fetching (OpenAQ + Open-Meteo — ไม่มี Air4Thai/calibration อีกต่อไป) =====

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
    for attempt in range(attempts):
        try:
            r = requests.get(f"https://api.openaq.org/v3/sensors/{sensor_id}/{kind}",
                              headers=OPENAQ_HEADERS, params=params, timeout=timeout)
        except requests.exceptions.RequestException:
            r = None
        if r is not None:
            if not (r.status_code == 429 or r.status_code >= 500):
                r.raise_for_status()
                return r
        if attempt < attempts - 1:
            time.sleep(2 ** attempt)
    raise RuntimeError(f"ดึง pm25 จาก OpenAQ ({kind}) ไม่สำเร็จหลัง retry (sensor {sensor_id})")


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
    return df.dropna(subset=["pm25"]).drop_duplicates(subset=["datetime_utc"]).sort_values("datetime_utc").reset_index(drop=True)


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


def _check_weather(w):

    if not isinstance(w, dict) or any(w.get(k) is None for k in WEATHER_REQUIRED_KEYS):
        raise ValueError("ข้อมูลสภาพอากาศจาก Open-Meteo ไม่ครบ")


def fetch_current_weather_openmeteo(lat, lon):
   
    params = {"latitude": lat, "longitude": lon, "current": WEATHER_CURRENT_VARS, "timezone": "UTC"}
    return _open_meteo_get(params, attempts=2 if USE_INPUT_FALLBACK else 1, timeout=10 if USE_INPUT_FALLBACK else 15)["current"]


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


# ===== Live features ใหม่ของโมเดล : จังหวัดข้างเคียง + จุดความร้อน =====


NEIGHBOR_FRESH_HOURS = 3       # reading ล่าสุดของเพื่อนบ้านห่างจากเวลาของจังหวัดเป้าหมายไม่เกินนี้ = "สด" (เก่า 1-3 ชม. เสียหาย <0.4%)
NEIGHBOR_AVG_K = 3             # อันดับ 1 ไม่สด -> เฉลี่ยเพื่อนบ้านสดใกล้สุดสูงสุดกี่จังหวัด (MAE +2.4%/+2.5% เทียบ NaN +32%/+65%)
NEIGHBOR_MAX_FETCH = 8         # ดึงเพื่อนบ้านตามลำดับระยะทางสูงสุดกี่จังหวัดต่อรอบ 
TARGET_STALE_HOURS = 6         # จังหวัดเป้าหมายมี reading ล่าสุดเก่ากว่านี้ = ข้อมูลเก่า (ไม่ log, ไม่ทำ NLG — ดู get_forecast)
TARGET_CACHE_TTL_S = 300       # จังหวัดเป้าหมายใช้ cache สั้น (ค่า "ปัจจุบัน" ต้องสด)
SERIES_CACHE_TTL_S = 1800      # เพื่อนบ้านใช้ cache 30 นาที (scheduled job วน 20 จังหวัดต่อชั่วโมง = ดึงจังหวัดละ 1 ครั้ง)
SERIES_CACHE_MAX_AGE_S = 6 * 3600  # ดึงเพื่อนบ้านไม่สำเร็จ -> ใช้ cache เก่าได้ไม่เกินนี้ (ยังผ่านเกณฑ์ "สด" อีกที)
FIRE_CACHE_TTL_S = 3600
FIRE_CACHE_MAX_AGE_S = 6 * 3600
FIRE_RADIUS_KM = 100
FIRE_WINDOW_HOURS = 24
FIRMS_AREA = "100.5,13.5,106.5,19.0"  # bbox เดียวกับตอนเทรน (อีสาน + กันชนลาว/กัมพูชา)
FIRMS_NRT_SOURCE = "VIIRS_NOAA20_NRT"  # ตอนเทรนใช้ _SP (archive, ล่าช้า ~2.5 เดือน) live ต้องใช้ NRT — เทียบตรงไม่ได้ (CHANGELOG 140)


def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(a))


# เพื่อนบ้านเรียงตามระยะทางจริง (คิดจากทั้ง 20 จังหวัด รวม zero-shot — เหมือน add_spatial_lag_feature ตอนเทรน)
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


def _series_disk_path(province):
    return os.path.join(SERIES_DISK_CACHE_DIR, f"openaq_series_{province}.csv")


def _save_series_to_disk(province, df):
    """เก็บซีรีส์ที่ดึงสำเร็จล่าสุดลงไฟล์ (เขียน tmp แล้ว replace) ให้ทางสำรองรอดการรีสตาร์ตแอป — ล้มเหลวเงียบๆ ได้ (เป็นแค่ safety net)"""
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
    """ตอนเริ่มแอป: โหลดไฟล์ cache ที่อายุไม่เกิน TARGET_FALLBACK_MAX_AGE_S เข้า _SERIES_CACHE (เวลา monotonic ย้อนตามอายุไฟล์)"""
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
    """q_hat ของช่วงความเชื่อมั่นเมื่อ input ถูกเติมค่า (ทดสอบไว้ที่ k บางค่า: k จริงที่ไม่ตรง -> ใช้ k ถัดไปที่มากกว่า = conservative,
    h ระหว่างค่าที่ทดสอบ -> interpolate เชิงเส้น) — แทน CONFORMAL_QHAT ที่สอบเทียบโดยถือว่าค่าปัจจุบันเป็นค่าจริง"""
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
    """สร้าง feature row (superset: 23 ตัวของรุ่นเก่า + 3 ตัวใหม่ของ v2 แล้วเลือกตาม FEATURE_COLS ของโมเดลที่โหลด)
    — สูตร lag/rolling เหมือนตอนเทรนเป๊ะ แต่ยึดตาม "เวลาจริง" ไม่ใช่ตำแหน่งแถว (OpenAQ อาจมีชั่วโมงขาดหาย)
    gap <=48h interpolate ยาวกว่านั้นปล่อย NaN ให้ compute_missing_ratio() จับ — ต้องซิงค์กับ
    scripts/build_raw_dataset_multiprovince.py + compute_neighbor_trend_feature.py + compute_fire_hotspot_feature.py
    เสมอ (ดู CLAUDE.md)
    neighbor: dict จาก compute_neighbor_features() (status != "ok" -> ค่าเพื่อนบ้านเป็น NaN ผู้เรียกต้อง force_naive)
    fire_count: จำนวนจุดความร้อนจาก count_fire_hotspots()
    """
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


def log_notification_snapshot(recorded_at_utc, latest_pm25, province):

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """INSERT INTO notification_log (recorded_at_utc, latest_pm25, province)
           VALUES (%s, %s, %s)
           ON CONFLICT (recorded_at_utc, province) DO NOTHING""",
        (recorded_at_utc, latest_pm25, province),
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
    (25, "ดี", [("อากาศ", "ดี"), ("คุณภาพอากาศ", "ดี"), "ปลอดภัย"]),
    (37.5, "ปานกลาง", ["ปานกลาง"]),
    (75, "เริ่มมีผลกระทบต่อสุขภาพ", ["เริ่มมีผลกระทบต่อสุขภาพ", "เริ่มมีผลกระทบ", "ส่งผลกระทบต่อสุขภาพ"]),
    (float("inf"), "มีผลกระทบต่อสุขภาพ", ["มีผลกระทบต่อสุขภาพ", "อันตราย", "อันตรายถึงชีวิต"]),
]

_GENERIC_CAUTION_KEYWORDS = [
    ("ปานกลาง", [("หลีกเลี่ยง", "กิจกรรมกลางแจ้ง"), ("ลด", "กิจกรรมกลางแจ้ง"), "ระคายเคือง"]),
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


def _fallback_message(level):
    if level is None:
        return "ไม่สามารถประเมินคุณภาพอากาศได้ในขณะนี้ กรุณาตรวจสอบใหม่ภายหลัง"
    return _HEALTH_MESSAGE_TEMPLATES[level]


def validate_health_message(message_text, predicted_pm25, risk_group="ทั่วไป"):
 
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
GEMINI_MAX_RETRIES = 3  # 429 (quota) backoff: รอ 1s, 2s, 4s

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


    try:
        shap_pairs = get_shap_explanation(model_used, REP_HORIZON_FOR_SHAP, feature_row)
    except Exception as e:
        print(f"[nlg-daily] SHAP explanation failed ({type(e).__name__}): {e}")
        shap_pairs = None

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


def get_forecast(province=DEFAULT_PROVINCE, risk_group="ทั่วไป"):
    province_meta = PROVINCES_BY_SLUG[province]
    series_info = {}
    df_recent = get_recent_series(province, strict=True, max_age_s=TARGET_CACHE_TTL_S, info=series_info)
    if df_recent is None or df_recent.empty:
        raise DataUnavailableError(f"ไม่มีข้อมูล PM2.5 ล่าสุดของ {province} จาก OpenAQ (สถานีอาจขัดข้อง)")
    imputed_info = None
    if IMPUTATION_CFG is not None:
        df_recent, imputed_info = impute_target_series(province, df_recent)
    imputed = imputed_info is not None

    # feature เพื่อนบ้าน + จุดความร้อน — เวลาอ้างอิงเดียวกับที่ใช้คิด lag; คำนวณ/ดึงเฉพาะเมื่อโมเดลที่โหลดใช้ feature นั้นจริง
    reference_time = df_recent["datetime_utc"].max()
    # สภาพอากาศพร้อมทางสำรอง : ค่าที่ดึงไว้ -> สด -> ค่าเก่า <=2 ชม. -> climatology; source != live เก็บไว้ใน data_quality
    current_weather, weather_info = get_current_weather(province_meta, at=reference_time)
    input_degraded = series_info.get("source") == "stale_cache" or weather_info["source"] == "climatology"
    if USES_NEIGHBOR:
        neighbor = compute_neighbor_features(province, reference_time)
    else:
        neighbor = {"status": "not_used", "policy": None, "neighbors_used": [], "pm25_neighbor_avg": np.nan, "nb_trend_12h": np.nan}
    if USES_FIRE:
        hotspots_df, fire_source = get_fire_hotspots()
        fire_count = count_fire_hotspots(province_meta, reference_time, hotspots_df)
    else:
        fire_source, fire_count = "not_used", 0
    # โมเดลใช้เพื่อนบ้านแต่ไม่มีเพื่อนบ้านสดเลย -> ไม่มีค่า neighbor ให้โมเดล ต้องใช้ naive_fallback ทุก horizon (ห้ามส่ง NaN เข้าโมเดล)
    force_naive = USES_NEIGHBOR and neighbor["status"] != "ok"

    X_now, latest_time, latest_pm25 = build_feature_row(df_recent, current_weather, province_meta, neighbor, fire_count)
    latest_time_th = latest_time.tz_convert("Asia/Bangkok")

    target_age_hours = (pd.Timestamp.now(tz="UTC") - df_observed["datetime_utc"].max()).total_seconds() / 3600
    target_stale = bool(target_age_hours > TARGET_STALE_HOURS) and not imputed

    r2_lookup = fetch_rolling_r2_lookup(province)

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
        forecast.append({
            "time": forecast_time_th.strftime("%H:%M"),
            "hour_thai": forecast_time_th.hour,
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

    loggable = not target_stale and not imputed and not input_degraded
    if loggable:
        log_predictions(latest_time, log_rows, province)
        log_notification_snapshot(latest_time.isoformat(), float(latest_pm25), province)
    backfill_actuals(df_observed, province)  
    if loggable:
        rep_model_used = forecast[REP_HORIZON_FOR_SHAP - 1]["model_used"]
        maybe_generate_daily_nlg(forecast, X_now, risk_group, province, rep_model_used)

    best_time = find_best_time(forecast)

    high_pollution_risk = is_current_at_risk(latest_pm25, risk_group) or len(risk_horizons) > 0

    return {
        "province": {"th": province_meta["province_th"], "en": province_meta["province_en"],
                      "is_zero_shot": province_meta["is_zero_shot"]},
        "current": {
            "pm25": round(float(latest_pm25), 1),
            "datetime_th": latest_time_th.strftime("%Y-%m-%d %H:%M"),
            "status": pm25_status(latest_pm25),
        },
        "current_weather": {
            "temp_c": round(current_weather["temperature_2m"], 1),
            "humidity_pct": round(current_weather["relative_humidity_2m"]),
            "wind_speed_kmh": round(current_weather["wind_speed_10m"], 1),
        },
        "forecast": forecast,
        "best_time": best_time,
        "high_pollution_risk": high_pollution_risk,
        "risk_horizons": risk_horizons,
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


@app.route("/api/current_pm25", methods=["GET"])
def api_current_pm25():
    try:
        province = resolve_province_from_request()
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
            "SELECT recorded_at_utc, latest_pm25 FROM notification_log WHERE province = %s "
            "ORDER BY recorded_at_utc DESC LIMIT %s",
            (province, NOTIF_SCAN_WINDOW),
        )
        snapshots = cur.fetchall()

        results = []
        for recorded_at_utc, latest_pm25 in snapshots:
            cur.execute(
                "SELECT horizon, predicted_pm25, model_used FROM prediction_log "
                "WHERE predicted_at_utc = %s AND province = %s",
                (recorded_at_utc, province),
            )
            horizon_rows = cur.fetchall()

            risk_horizons = sorted(
                h for h, pred, model_used in horizon_rows
                if is_horizon_at_risk(pred, risk_group)
            )
            fallback_horizons = sorted(h for h, _, model_used in horizon_rows if model_used == MODEL_NAIVE)

            if not (is_current_at_risk(latest_pm25, risk_group) or risk_horizons):
                continue

            recorded_at_th = pd.Timestamp(recorded_at_utc).tz_convert("Asia/Bangkok")
            results.append({
                "recorded_at_th": recorded_at_th.strftime("%Y-%m-%d %H:%M"),
                "latest_pm25": round(float(latest_pm25), 1),
                "risk_horizons": risk_horizons,
                "fallback_horizons": fallback_horizons,
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
        cur.execute(
            "SELECT recorded_at_utc, latest_pm25 FROM notification_log WHERE province = %s "
            "ORDER BY recorded_at_utc DESC LIMIT 1",
            (province,),
        )
        snapshot = cur.fetchone()

        if snapshot is None:
            cur.close()
            conn.close()
            return jsonify({
                "high_pollution_risk": False, "risk_horizons": [], "fallback_horizons": [],
                "current_pm25": None, "current_status": None, "recorded_at_th": None,
                "risk_group": risk_group,
            })

        recorded_at_utc, latest_pm25 = snapshot
        cur.execute(
            "SELECT horizon, predicted_pm25, model_used FROM prediction_log "
            "WHERE predicted_at_utc = %s AND province = %s",
            (recorded_at_utc, province),
        )
        horizon_rows = cur.fetchall()
        cur.close()
        conn.close()

        risk_horizons = sorted(h for h, pred, model_used in horizon_rows if is_horizon_at_risk(pred, risk_group))
        fallback_horizons = sorted(h for h, _, model_used in horizon_rows if model_used == MODEL_NAIVE)
        high_pollution_risk = is_current_at_risk(latest_pm25, risk_group) or bool(risk_horizons)
        recorded_at_th = pd.Timestamp(recorded_at_utc).tz_convert("Asia/Bangkok")

        return jsonify({
            "high_pollution_risk": high_pollution_risk,
            "risk_horizons": risk_horizons,
            "fallback_horizons": fallback_horizons,
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

    app.run(port=5000, debug=DEBUG)

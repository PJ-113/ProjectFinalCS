import json
import os
import pickle
import random
import re
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

# SHAP explanation (scripts/shap_analysis.py) ปิดไว้ชั่วคราว — ยังผูกกับโครงสร้างโมเดลเก่า
# ({horizon: model} ไม่ใช่ {horizon: {quantile: model}}) ดู generate_daily_nlg_explanation()

# ===== Config =====

# pooled model: โมเดลเดียวรับใช้ 17 จังหวัดที่เทรน + 3 zero-shot (lat/lon เป็น feature — CHANGELOG 75)
# v3 = _20260919_203455 (20 feature: ประวัติ PM2.5 + อากาศ + lat/lon + nb_trend_12h + fire_hotspot_count_100km_24h, hyperparameter จูนแล้ว)
# เทรนจาก dataset ที่แก้ป้ายเวลา OpenAQ แล้ว (CHANGELOG 144-149) — ต้องใช้คู่กับ USE_TIMESTAMP_OFFSET = True เสมอ
# รุ่นก่อนหน้าทั้งหมด (v2 = _20260919_135655 เทรนจากข้อมูลป้ายเวลาผิด และ _20260913_014901) ยังอยู่ใน models/ แต่ห้ามกลับไปใช้โดยไม่ปิด offset
# (ย้อนกลับ: คืน path + MODEL_LGB/MODEL_XGB + USE_TIMESTAMP_OFFSET=False — ดู CLAUDE.md)
LGB_MODEL_PATH = "models/lgb_pm25_multiprovince_residual_bagged_quantile_20260919_203455.pkl"
XGB_MODEL_PATH = "models/xgb_pm25_multiprovince_residual_bagged_quantile_20260919_203455.pkl"
CONFORMAL_LGB_PATH = "models/conformal_calibration_lgb_20260919_203455.csv"
CONFORMAL_XGB_PATH = "models/conformal_calibration_xgb_20260919_203455.csv"
# ชื่อโมเดลที่บันทึกใน prediction_log.model_used — รุ่นใหม่ใช้ชื่อ _v2 แยกจากรุ่นเก่า
# ("lightgbm_pooled"/"xgboost_pooled" = รุ่นเก่าสุด, "*_v2" = รุ่นที่เทรนจากข้อมูลป้ายเวลาผิด ยังมีอยู่ในแถวเก่า) กัน compute_rolling_r2() ปนผลรุ่นเก่าเข้ากับรุ่นใหม่
MODEL_LGB = "lightgbm_pooled_v3"
MODEL_XGB = "xgboost_pooled_v3"
MODEL_NAIVE = "naive_fallback"
PROVINCE_REGISTRY_PATH = "data/province_station_registry.json"
# ป้ายเวลา OpenAQ /measurements ของ 9 สถานีเลื่อนไป 23 ชม. (CHANGELOG 144-145): registry เก็บ offset ต่อสถานีใน openaq_timestamp_offset_hours (วัดโดย
# scripts/annotate_registry_timestamp_offset.py) ต้องบวกเข้า datetime_utc เสมอ — เฉพาะโมเดลที่เทรนจาก dataset ที่แก้ป้ายเวลาแล้ว (v3 ขึ้นไป) เท่านั้น
# (โมเดลรุ่นก่อนเทรนจากข้อมูลป้ายเวลาผิด ใช้กับข้อมูลที่แก้แล้วไม่ได้)
USE_TIMESTAMP_OFFSET = True
OFFSET_FUTURE_TOLERANCE_H = 1  # หลังบวก offset แล้ว reading ล่าสุดล้ำอนาคตเกินนี้ = OpenAQ แก้ป้ายเวลาแล้ว offset จะแก้ซ้ำ -> ทิ้ง offset (ดู fetch_recent_pm25_openaq)
DEFAULT_PROVINCE = "khonkaen"
HIGH_POLLUTION_THRESHOLD = 37.5  # ค่าเดียวกันทุกจังหวัด (มาตรฐาน คพ. ไม่ใช่ค่าที่ผูกกับโมเดล)
# ต้องรัน `npm run build` ใน frontend/ ก่อน ถึงจะมีโฟลเดอร์นี้ (ดู CHANGELOG)
FRONTEND_DIST = os.path.join(os.path.dirname(os.path.abspath(__file__)), "frontend", "dist")
DEBUG = True  # ต้องอ่านจากตรงนี้ ไม่ใช่ app.debug เพราะ Flask ยังไม่ set app.debug จนกว่าจะเรียก app.run()
SIGMA_CRITICAL = 0.3  # Dynamic Routing (proposal 4.1.6): missing_ratio เกินนี้ถือว่า input เสียหายหนัก ให้ fallback ไปพยากรณ์แบบ naive persistence (ดู select_model)
EXPLORATION_RATE = 0.2  # โอกาสที่จะ "ลอง" โมเดลที่ยังไม่มีข้อมูลพอประเมิน R² แก้ปัญหา cold-start deadlock (ดู select_model)
CONFORMAL_ALPHA = 0.10  # เป้าหมาย coverage 90% (ดู train_multiprovince_models.py) ใช้แค่ใน docstring/comment ไม่ได้ใช้คำนวณตรงนี้


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
OPENAQ_KEY = _ENV["OPENAQ_KEY"]  # ส่งเป็น header X-API-Key (fetch_recent_pm25_openaq) — ห้าม print/log
OPENAQ_HEADERS = {"X-API-Key": OPENAQ_KEY}
# NASA FIRMS MAP_KEY (fire_hotspot_count_100km_24h) — ไม่บังคับ: ไม่มี/ใช้ไม่ได้ -> feature = 0 + flag ใน data_quality
# (แอปต้องไม่ล่มเพราะ FIRMS) ห้าม print/log คีย์นี้ และห้าม log URL/ข้อความ exception ของ requests (URL มีคีย์)
FIRMS_KEY = _ENV.get("FIRMS_KEY")

# PostgreSQL (ย้ายจาก sqlite แล้ว ดู CHANGELOG) — get_db_connection() เปิด connection ใหม่ทุกครั้งที่เรียก
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
# โมเดลใช้ feature เพื่อนบ้าน/จุดความร้อนไหม (ตาม FEATURE_COLS ของ pkl) — ถ้าไม่ใช้ get_forecast() จะไม่ดึงข้อมูลนั้นและไม่บังคับ naive_fallback
USES_NEIGHBOR = any(c in FEATURE_COLS for c in ("pm25_neighbor_avg", "nb_trend_12h"))
USES_FIRE = "fire_hotspot_count_100km_24h" in FEATURE_COLS

# conformal calibration q_hat ต่อ horizon ต่อโมเดล (train_multiprovince_models.py) — ใช้ปรับขอบ
# P05/P95 ให้ coverage เข้าใกล้เป้าหมาย 90% จริง ไม่ใช่แค่ P05/P95 ดิบจาก quantile regression
_conformal_lgb_df = pd.read_csv(CONFORMAL_LGB_PATH)
_conformal_xgb_df = pd.read_csv(CONFORMAL_XGB_PATH)
CONFORMAL_QHAT = {
    MODEL_LGB: {int(r.horizon): float(r.q_hat) for r in _conformal_lgb_df.itertuples()},
    MODEL_XGB: {int(r.horizon): float(r.q_hat) for r in _conformal_xgb_df.itertuples()},
}

# ===== เติมข้อมูลของสถานีเป้าหมายที่เงียบ (Data Imputation, H-lite) =====
# ผลทดลอง: CHANGELOG 169-171, docs/thesis_imputation_experiment.md — เมื่อ reading ล่าสุดของสถานีเก่ากว่าปกติ เดิมระบบยึดเวลา reading
# ล่าสุดเป็น "ตอนนี้" (พยากรณ์นับจากเวลาเก่า); ตอนนี้เติมค่าช่วงที่เงียบด้วยจังหวัดข้างเคียง (IDW-difference แล้ว Ridge) เพื่อพยากรณ์จากเวลาปัจจุบันจริง
# ปิดได้ทันทีที่ USE_TARGET_IMPUTATION=False (พฤติกรรมกลับเป็นเดิมทั้งหมด) ไฟล์ค่าสัมประสิทธิ์หาย/พัง -> ปิดเองพร้อม log (แอปไม่ล่ม)
USE_TARGET_IMPUTATION = True
IMPUTATION_CONFIG_PATH = "models/imputation_h_lite_20260924_204956.json"
IMPUTATION_MIN_K = 2   # k = ชั่วโมงเต็มที่ reading ล่าสุดตามหลัง "ชั่วโมงที่เพิ่งจบ" (now-1h); k<2 = ความหน่วงปกติของ OpenAQ ไม่เติม (ค่านี้ยังไม่ได้วัดจริง — scripts/measure_target_data_latency.py)
IMPUTATION_MAX_K = 48  # เกินนี้ยังไม่เคยทดสอบ -> คงพฤติกรรมเดิม (ข้อมูลเก่า + banner)

# ===== ทางสำรองเมื่อดึง OpenAQ / Open-Meteo ไม่ได้ (CHANGELOG 172) =====
# ผลวัด: scripts/experiment_weather_fallback.py (สภาพอากาศเก่า <=2 ชม. เสีย MAE <1%; ค่า climatology ยังดีกว่า Persistence ทุก horizon) และ
# /hours ของ OpenAQ ให้ค่าตรง /measurements ทุกจุดที่ซ้อนกัน (ตรวจ 19 สถานี 72 ชม. ต่างเฉลี่ย 0.0000 ไม่ต้องบวก offset) แต่ **ช้ากว่า ~1 ชม.**
# (วัดที่นาที 24 ของชั่วโมง: /measurements ถึง 13:00, /hours ถึง 12:00 ทั้ง 19 สถานี) -> ตอนใช้ทางสำรอง k เพิ่มได้ ~1 (ระวังเทียบกับ IMPUTATION_MIN_K)
# USE_INPUT_FALLBACK=False = พฤติกรรมเดิมทั้งหมด (ดึงพลาด -> raise; ไม่มี retry/breaker/cache สภาพอากาศ/cache จังหวัดเป้าหมาย/ไฟล์ cache)
USE_INPUT_FALLBACK = True
WEATHER_TTL_S = 600                 # ใช้ค่าสภาพอากาศที่ดึงไว้ซ้ำได้ 10 นาที (Open-Meteo อัปเดต current ทุก 15 นาที; ลดจำนวน request)
WEATHER_CACHE_MAX_AGE_H = 2         # ดึงไม่สำเร็จ -> ใช้ค่าเก่าได้ไม่เกินนี้ (ตามเกณฑ์ที่ตั้งก่อนดูผลใน experiment_weather_fallback.py: k=2)
WEATHER_CLIMATOLOGY_PATH = "models/weather_climatology_20260924_211741.json"
TARGET_FALLBACK_MAX_AGE_S = 6 * 3600  # OpenAQ ดึงจังหวัดเป้าหมายไม่ได้ -> ใช้ cache ได้ไม่เกินนี้ (เท่ากับที่เพื่อนบ้านใช้อยู่ ไม่ได้วัดแยก)
SERIES_DISK_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "runtime_cache")  # กัน cache หายเมื่อแอปรีสตาร์ต (gitignored)

# กลุ่มเปราะบางเริ่มระวังเร็วกว่าคนทั่วไป ~1 ระดับ AQI (ตาม WHO/US EPA) ลด threshold 10 หน่วย
# ให้เริ่มเตือนตั้งแต่ช่วง "ปานกลาง" แทนที่จะรอถึง "เริ่มมีผลกระทบต่อสุขภาพ" เหมือนคนทั่วไป —
# เป็นตัวเลข heuristic ไม่มี dataset เฉพาะรองรับเป๊ะๆ ทั้งสองกลุ่มใช้ค่าเดียวกัน (ไม่มีหลักฐานแยกความรุนแรง)
RISK_GROUP_THRESHOLD_OFFSET = {"ทั่วไป": 0.0, "เด็ก-ผู้สูงอายุ": -10.0, "มีโรคประจำตัว": -10.0}

# Personalized Risk Score: กิจกรรมหนักขึ้น = หายใจถี่ขึ้น = สูดฝุ่นมากขึ้นที่ความเข้มข้นเท่าเดิม
# จำลองเป็นการลด effective PM2.5 (sign convention เดียวกับ RISK_GROUP_THRESHOLD_OFFSET) —
# heuristic สาธิตหลักการ ไม่ใช่ dose-response จากงานวิจัยทางการแพทย์
ACTIVITY_INTENSITY_OFFSET = {"เบา": 0.0, "ปานกลาง": -5.0, "หนัก": -10.0}

# แปลง effective PM2.5 เป็นคะแนน 0-100 ด้วย linear interpolation ข้าม breakpoint ที่ยึดตาม PM25_LEVELS
# เป๊ะ (แต่ละระดับ = 20 คะแนน) 150 เป็นเพดาน scale คะแนนเท่านั้น ไม่ใช่ breakpoint ทางการ
RISK_SCORE_BREAKPOINTS = [(0, 0), (15, 20), (25, 40), (37.5, 60), (75, 80), (150, 100)]


def effective_threshold_for(risk_group):
    """threshold เดียว (37.5 + offset ตาม risk_group) ไม่ใช้ bias CSV แบบเดิม (per-province, เลิกใช้
    ตั้งแต่ย้ายมาโมเดล pooled)"""
    return HIGH_POLLUTION_THRESHOLD + RISK_GROUP_THRESHOLD_OFFSET.get(risk_group, 0.0)


# ให้ get_forecast() และ /api/notifications ใช้ตรรกะ "อะไรถือว่าเสี่ยง" เดียวกันเป๊ะ — เทียบค่าพยากรณ์
# ตรงๆ (median prediction) กับ threshold ไม่ใช่ขอบบน P95 (เคยใช้ P95 มาก่อน 2026-09-23 เพื่อชดเชย
# "โมเดลทายพลาดช่วงมลพิษสูง" แต่เป็นการเลือกแบบ heuristic ไม่เคยผ่านการทดลองเทียบวิธี พอทดสอบจริงบน
# day-ahead backtest (models/alert_capability_pooled_20260922_203655.csv, 1,372 คู่ threshold 37.5)
# พบว่า P95 ทำ false-alarm สูงถึง 36.8-37.1% (เกือบ 4 ใน 10 ครั้งที่เตือนคือเตือนผิด) ขณะที่ค่าพยากรณ์
# ตรงๆ ได้ recall 92.3-92.5% กับ false-alarm แค่ 13.4% ชนะทั้งสองมิติพร้อมกันเทียบกับ persistence/
# ค่าเฉลี่ยวันก่อนหน้า จึงเปลี่ยนมาใช้ค่าพยากรณ์ตรงๆ เป็นเกณฑ์เดียวทั้งระบบ)
def is_horizon_at_risk(predicted_pm25, risk_group):
    return predicted_pm25 > effective_threshold_for(risk_group)


def is_current_at_risk(latest_pm25, risk_group):
    # bool(...) กัน numpy.bool_ หลุดจาก "is_current_at_risk(...) or ..." ไปพัง jsonify() (latest_pm25
    # เป็น numpy.float64 จาก pandas — เจอบั๊กนี้จริงตอน PM2.5 แตะ threshold กลุ่มเปราะบางพอดี)
    return bool(latest_pm25 > effective_threshold_for(risk_group))


ACTIVE_HOURS_START_TH = 6
ACTIVE_HOURS_END_TH = 20


def find_best_time(forecast):
    """หาจุด PM2.5 ต่ำสุด กรองเฉพาะ 06:00-20:00 เวลาไทย (ช่วงใช้งานได้จริง กัน suggest ช่วงดึก)
    fallback ไปทั้งหมดถ้า active_hours ว่างเปล่า (edge case ไม่ควรเกิดจริงเพราะมี 24 จุดครบทุกชม.)"""
    active_hours = [f for f in forecast if ACTIVE_HOURS_START_TH <= f["hour_thai"] <= ACTIVE_HOURS_END_TH]
    return min(active_hours or forecast, key=lambda x: x["pm25"])


def compute_risk_score(pm25, risk_group, activity):
    """Personalized Risk Score: pm25 + risk_group + activity -> คะแนน 0-100 ผ่าน effective PM2.5
    (ดู ACTIVITY_INTENSITY_OFFSET/RISK_GROUP_THRESHOLD_OFFSET) แล้ว interpolate เชิงเส้นข้าม
    RISK_SCORE_BREAKPOINTS — pure function ไม่ผูกกับ live data"""
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


def compute_rolling_r2(model_name, horizon, province, N=24):
    """R² แบบ rolling จาก N แถวล่าสุดของ (model_name, horizon, province) — query เดี่ยวสำหรับคู่เดียว
    ใช้แค่ diagnostics/test เท่านั้น hot path จริงใช้ fetch_rolling_r2_lookup() (query เดียวทั้ง
    จังหวัดแทน 48 query — ดู CHANGELOG 86) คืน None ถ้าข้อมูลไม่พอ (< 5 คู่)"""
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

    if len(rows) < 5:
        return None

    y_pred = [r[0] for r in rows]
    y_true = [r[1] for r in rows]
    return r2_score(y_true, y_pred)


ROLLING_R2_MODELS = (MODEL_XGB, MODEL_LGB)


def fetch_rolling_r2_lookup(province, N=24):
    """R² ของทั้ง 2 โมเดลทุก horizon ด้วย query เดียว (window function หา N แถวล่าสุดต่อ (model,
    horizon)) — get_forecast() เรียกครั้งเดียวก่อน loop 24 horizon คืน dict {(model_used, horizon): r2|None}"""
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
            if len(pairs) < 5:
                lookup[(model_used, h)] = None
            else:
                y_pred = [p[0] for p in pairs]
                y_true = [p[1] for p in pairs]
                lookup[(model_used, h)] = r2_score(y_true, y_pred)
    return lookup


def select_model(feature_row, horizon, province, r2_lookup, force_naive=False):
    """Dynamic Routing (proposal 4.1.6, 2 ทางหลังตัด Prophet ออก):
    0. force_naive -> "naive_fallback" ทันที (CHANGELOG 141: ไม่มีจังหวัดข้างเคียงที่ข้อมูลสดเลย จึงไม่มีค่า
       pm25_neighbor_avg/nb_trend_12h ให้โมเดล — ต้องแยก flag นี้ออกมาต่างหาก เพราะ NaN แค่ 2 จาก 21
       feature = 0.095 < SIGMA_CRITICAL จะไม่ trigger ข้อ 1 เอง แล้วโมเดลจะได้รับ NaN ซึ่งแย่กว่า persistence
       ชัดเจน: MAE +32%/+65% ใน scripts/check_model_sensitivity_live_degradation.py)
    1. missing_ratio > SIGMA_CRITICAL -> "naive_fallback" (โมเดล pooled ไม่เคยเห็นแถวมี NaN ตอนเทรน
       เลย ป้อน NaN ตรงๆ คือ out-of-distribution ใช้ persistence แทน)
    2. ไม่งั้นเทียบ rolling R² ล่าสุดจาก r2_lookup (precompute โดย fetch_rolling_r2_lookup()) — เช็ค
       None ทั้ง 2 ทิศทางเสมอ (เช็คแค่ฝั่งเดียวเคยทำให้โมเดลนึงไม่ได้ explore เลย ผิดเจตนา ε-greedy —
       CHANGELOG 20): ไม่มีข้อมูลทั้งคู่ -> สุ่ม 50/50, มีแค่ตัวเดียว -> explore ตัวที่ขาดด้วย
       EXPLORATION_RATE ไม่งั้น fallback ไปตัวที่มีข้อมูล, มีครบ -> เลือก R² สูงกว่า
    """
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

    # NLG รายวัน (สร้าง 1 ครั้ง/วัน ตอน 06:00 น. เวลาไทย — ดู maybe_generate_daily_nlg()) date_thai
    # เป็น "YYYY-MM-DD" เวลาไทยเสมอ ไม่ใช่ UTC (ดู get_thai_now())
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
    """เปิดวงจรชั่วคราวหลังล้มเหลวต่อเนื่อง `threshold` ครั้ง: ช่วง cooldown ไม่ยิง request เลย (ไปใช้ cache/ทางสำรองทันที)
    กันงานรายชั่วโมงที่วน 20 จังหวัด x (retry+timeout ~1-2 นาที) ยาวเกิน 1 ชม. ตอน API ล่มทั้งระบบ; สำเร็จครั้งเดียวรีเซ็ตตัวนับ"""

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
    """ดึง pm25 ย้อนหลัง hours_back ชม. จาก OpenAQ v3 — สเกลเดียวกับตอนเทรน (ไม่ต้อง calibrate)
    72h พอสำหรับ lag_24/roll_24 + เผื่อ gap, retry แบบ backoff เฉพาะ 429/5xx

    offset_hours (CHANGELOG 149): ป้ายเวลาของสถานีบางแห่งเลื่อน -> ดึง label ย้อนหลังเพิ่มอีก offset แล้วบวก offset เข้า datetime_utc ให้ตรงเวลาจริง
    (ค่าที่ป้าย L คือค่าจริงของ L+offset) **ไม่ตัดปลายช่วงที่ now-offset** โดยตั้งใจ เพื่อให้จับกรณี OpenAQ แก้ป้ายเวลาแล้วได้: ถ้าหลังบวก offset
    reading ล่าสุดล้ำอนาคตเกิน OFFSET_FUTURE_TOLERANCE_H ชม. แปลว่า offset ไม่จำเป็นแล้ว (ถ้าไม่ตรวจ ข้อมูลเก่า 23 ชม.จะถูกเลื่อนให้ดูสดโดยไม่มีอะไรเตือน)
    -> ทิ้ง offset ชั่วคราวและ log เตือน (ให้รัน scripts/annotate_registry_timestamp_offset.py ใหม่)"""
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
    """ทางสำรองภายใน OpenAQ: /v3/sensors/{id}/hours (คนละ endpoint กับ /measurements, key เดียวกัน) — ตรวจแล้วว่าค่าและป้ายเวลาตรงกับ
    /measurements หลังบวก offset ทุกจุดที่ซ้อนกัน (CHANGELOG 144, 172) จึง **ไม่ต้องบวก offset** (คืน DataFrame รูปแบบเดียวกัน) ข้อแลกเปลี่ยน: reading ล่าสุด
    ช้ากว่า /measurements ~1 ชม. (วัดจริง 19/19 สถานี) จึงใช้เป็นทางสำรองเท่านั้น ไม่ใช่ทางหลัก"""
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
    """ค่าที่ได้ต้องครบ 6 ตัวแปรและไม่เป็น None (Open-Meteo คืน null ได้) — ไม่ครบ = ถือว่าดึงไม่สำเร็จ ไม่ส่งเข้าโมเดล"""
    if not isinstance(w, dict) or any(w.get(k) is None for k in WEATHER_REQUIRED_KEYS):
        raise ValueError("ข้อมูลสภาพอากาศจาก Open-Meteo ไม่ครบ")


def fetch_current_weather_openmeteo(lat, lon):
    """weather ไม่ต้อง lag/rolling เหมือน pm25 (มีแค่ค่า "ปัจจุบัน" เป็น feature) พอแค่ current= เดียว
    ผู้เรียกในแอปใช้ผ่าน get_current_weather() (cache + ทางสำรอง) ไม่เรียกตรง"""
    params = {"latitude": lat, "longitude": lon, "current": WEATHER_CURRENT_VARS, "timezone": "UTC"}
    return _open_meteo_get(params, attempts=2 if USE_INPUT_FALLBACK else 1, timeout=10 if USE_INPUT_FALLBACK else 15)["current"]


def fetch_current_weather_batch(provinces):
    """ค่าปัจจุบันของทุกจังหวัดใน request เดียว (Open-Meteo รับหลายพิกัดคั่น , คืน list ตามลำดับ) -> {province_en: current dict}
    ตรวจแล้วว่าค่าเท่ากับเรียกทีละจังหวัด (CHANGELOG 172) — ลด 20 request/รอบเหลือ 1 โอกาสโดน 429 ลดลง"""
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
    """ตาราง climatology จาก scripts/export_weather_climatology.py — ผิดพลาด -> None (ไม่มีทางสุดท้าย: ดึงสภาพอากาศไม่ได้ก็ raise เหมือนเดิม)"""
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
    """ค่าเฉลี่ย (จังหวัด, เดือน, ชั่วโมง UTC) ในรูป dict เดียวกับ Open-Meteo current — ใส่ _wind_dir_sin/_cos ไว้ให้ build_feature_row ใช้ค่าเฉลี่ยตรงๆ
    (สูตรเดียวกับการทดลอง ไม่แปลงกลับเป็นองศาแล้วคิด sin/cos ใหม่ ซึ่งจะได้เวกเตอร์หน่วยต่างจากค่าเฉลี่ย); wind_direction_10m ไว้แสดงผลเท่านั้น"""
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
    """เรียกต้นงานรายชั่วโมง: ดึงสภาพอากาศทุกจังหวัดใน request เดียวแล้วใส่ cache (ล้มเหลว -> ปล่อยให้แต่ละจังหวัดดึงเอง/ใช้ทางสำรองตามปกติ)"""
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
    """สภาพอากาศปัจจุบันของจังหวัด พร้อมทางสำรอง -> (dict, {"source": "live"|"stale_cache"|"climatology", "age_hours"})
    ลำดับ: ค่าที่ดึงไว้ไม่เกิน WEATHER_TTL_S -> ดึงสด -> (พลาด) ค่าเก่าไม่เกิน WEATHER_CACHE_MAX_AGE_H -> climatology ของเดือน/ชั่วโมงที่ `at`
    -> raise (เหมือนเดิม) ; USE_INPUT_FALLBACK=False = ดึงสดอย่างเดียวเหมือนเดิม"""
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


# ===== Live features ใหม่ของโมเดล v2: จังหวัดข้างเคียง + จุดความร้อน (CHANGELOG 141) =====
# ตรรกะทั้งหมดทดสอบแล้วใน scripts/check_live_feature_parity.py (สูตรตรง dataset เทรน 97-100%) และ
# scripts/check_model_sensitivity_live_degradation.py (เลือกนโยบายสำรองด้วยตัวเลข) — แก้ที่นี่ต้องตรวจซ้ำด้วย
# scripts/verify_app_feature_parity.py

NEIGHBOR_FRESH_HOURS = 3       # reading ล่าสุดของเพื่อนบ้านห่างจากเวลาของจังหวัดเป้าหมายไม่เกินนี้ = "สด" (เก่า 1-3 ชม. เสียหาย <0.4%)
NEIGHBOR_AVG_K = 3             # อันดับ 1 ไม่สด -> เฉลี่ยเพื่อนบ้านสดใกล้สุดสูงสุดกี่จังหวัด (MAE +2.4%/+2.5% เทียบ NaN +32%/+65%)
NEIGHBOR_MAX_FETCH = 8         # ดึงเพื่อนบ้านตามลำดับระยะทางสูงสุดกี่จังหวัดต่อรอบ (กัน request บาน)
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
            _SERIES_CACHE[p["province_en"]] = (time.monotonic() - age_s, df)
            loaded += 1
        except Exception:
            continue
    if loaded:
        print(f"[openaq] โหลด cache จากไฟล์ {loaded} จังหวัด (ใช้เป็นทางสำรองเท่านั้น ไม่ถือว่าสด)")


def fetch_series_from_openaq(meta):
    """ดึงซีรีส์ pm25 72 ชม. ของสถานี -> (DataFrame, source) โดย source = "live" (/measurements + offset) หรือ "hours_endpoint" (/hours ไม่มี offset)
    /measurements พลาด -> ลอง /hours; ทั้งคู่พลาด -> raise (ข้อผิดพลาดของ /measurements) และนับเข้า breaker; breaker เปิดอยู่ -> raise ทันทีไม่ยิง request"""
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
    """pm25 72 ชม.ล่าสุดของจังหวัด ผ่าน cache ร่วม (scheduled job ที่วน 20 จังหวัด + /api/forecast ใช้ร่วมกัน
    จำนวน request OpenAQ ต่อรอบเท่าเดิม = จังหวัดละ 1 ครั้ง) คืน DataFrame หรือ None (ไม่มีข้อมูล)
    strict=True (จังหวัดเป้าหมาย): ดึงไม่สำเร็จ -> ใช้ cache เก่าได้ไม่เกิน TARGET_FALLBACK_MAX_AGE_S (เฉพาะ USE_INPUT_FALLBACK) ไม่งั้น raise
    strict=False (เพื่อนบ้าน): ดึงไม่สำเร็จ -> ใช้ cache เก่า (<= SERIES_CACHE_MAX_AGE_S) ไม่งั้น None (ไม่ raise)
    info (dict, ไม่บังคับ): ผู้เรียกส่งมารับ {"source": "live"|"hours_endpoint"|"stale_cache", "age_hours"} — stale_cache = ใช้ข้อมูลที่ดึงไว้ก่อนหน้า"""
    now = time.monotonic()
    with _SERIES_LOCK:
        hit = _SERIES_CACHE.get(province)
    if hit is not None and now - hit[0] < max_age_s:
        if info is not None:
            info.update({"source": "live", "age_hours": round((now - hit[0]) / 3600, 2)})
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
        _SERIES_CACHE[province] = (now, df)
    _save_series_to_disk(province, df)
    if info is not None:
        info.update({"source": source, "age_hours": 0.0})
    return df


_load_series_disk_cache()


def _filled_grid(df_recent, t):
    """กริดรายชั่วโมง 48 ชม.ย้อนหลังจาก t (ใช้ข้อมูล <= t เท่านั้น) interpolate เชิงเส้น limit=48 — สูตรเดียวกับ
    build_feature_row() ของ lag/rolling และกับ scripts/check_live_feature_parity.py"""
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
    """pm25_neighbor_avg + nb_trend_12h ของจังหวัดนี้ ณ latest_time ด้วยนโยบายที่เลือกจากผลทดลอง (CHANGELOG 140):
      - เพื่อนบ้านอันดับ 1 "สด" -> ใช้ตามสูตรเทรนเป๊ะ (policy "k1_exact")
      - ไม่สด -> เฉลี่ยเพื่อนบ้านสดที่ใกล้สุดสูงสุด NEIGHBOR_AVG_K จังหวัด, trend คิดจากค่าเฉลี่ยชุดเดียวกัน
        (policy "avg_nearest_fresh")
      - ไม่มีเพื่อนบ้านสดเลย -> status "no_fresh_neighbor" (ผู้เรียกต้องบังคับ naive_fallback ห้ามส่ง NaN เข้าโมเดล)
    คืน dict: status, policy, neighbors_used, pm25_neighbor_avg, nb_trend_12h
    fetch: ฟังก์ชัน province -> DataFrame|None (default get_recent_series) ใส่ไว้ให้ unit test ฉีดข้อมูลสังเคราะห์"""
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
    """อ่านค่าสัมประสิทธิ์ Ridge-lite + จุดสลับ + ตาราง q(k,h) จาก JSON (สร้างโดย scripts/export_imputation_h_lite.py) ผิดพลาด -> None (ปิดฟีเจอร์)"""
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
    """ต่อท้าย df_recent ด้วยค่า PM2.5 ที่เติมของช่วงที่สถานีเงียบ (t0, anchor] โดย t0 = reading จริงล่าสุด, anchor = ชั่วโมงที่เพิ่งจบ
    (now-1h) วิธี H-lite (ผลทดลองเดียวกับที่ลงเล่ม): ตำแหน่ง j<=switch_j ชม.หลัง t0 = x0 + IDW ของการเปลี่ยนแปลงจังหวัดข้างเคียง,
    ไกลกว่านั้น = Ridge บน [x0, ค่าเพื่อนบ้าน ณ tau/t0, ผลต่าง, j, log1p(j), ชั่วโมง UTC] (ค่าสัมประสิทธิ์จาก JSON)
    เพื่อนบ้าน = ที่ใกล้สุด <=5 จังหวัดจากทั้ง 20 ใช้ที่ "วัดได้จริง ณ tau และ t0" ไม่เกิน 3 จังหวัด ถ่วง 1/ระยะทาง² (ไม่ interpolate)
    ตำแหน่งที่ไม่มีเพื่อนบ้านที่ใช้ได้ -> Persistence (x0); ถ้าตำแหน่งสุดท้าย (ปัจจุบัน) ไม่มีเลย -> ไม่เติม (คืนสภาพเดิม)
    คืน (df_ใหม่, info) — info=None เมื่อไม่ได้เติม (k อยู่นอก [MIN_K, MAX_K], ฟีเจอร์ปิด, ไม่มีเพื่อนบ้านตอนนี้)
    ห้ามส่ง df ที่เติมแล้วไป backfill_actuals() (จะบันทึกค่าประมาณเป็นค่าจริง)"""
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
    """จุดความร้อน FIRMS NRT 2 วันล่าสุดทั้ง bbox อีสาน (1 request/ชม. ใช้ร่วมทุกจังหวัด) คืน (DataFrame|None, source)
    source: "live" | "cache" | "stale_cache" | "unavailable" | "no_key" — ไม่ raise (แอปต้องไม่ล่มเพราะ FIRMS)
    ห้าม log ข้อความ exception ของ requests: URL มี MAP_KEY"""
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
    """จำนวนจุดใน (latest_time-24h, latest_time] รัศมี FIRE_RADIUS_KM — ตรรกะเดียวกับ compute_fire_hotspot_feature.py
    (ไม่มีข้อมูล -> 0, E4: แย่ลงเพียง +1.8%)"""
    if hotspots_df is None or hotspots_df.empty:
        return 0
    near = haversine_km(province_meta["lat"], province_meta["lon"],
                        hotspots_df["latitude"].values, hotspots_df["longitude"].values) <= FIRE_RADIUS_KM
    t = hotspots_df["t"][near]
    return int(((t > latest_time - pd.Timedelta(hours=FIRE_WINDOW_HOURS)) & (t <= latest_time)).sum())


def migrate_daily_nlg_risk_group():
    """เพิ่ม risk_group ให้ daily_nlg_log (CREATE TABLE IF NOT EXISTS ไม่เพิ่ม column ให้ตารางเดิม
    ต้อง ALTER จริง) เปลี่ยน unique เป็น (date_thai, risk_group) แถวเก่าถือเป็นกลุ่ม 'ทั่วไป' โดยปริยาย"""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("ALTER TABLE daily_nlg_log ADD COLUMN IF NOT EXISTS risk_group TEXT NOT NULL DEFAULT 'ทั่วไป'")
    cur.execute("""
        SELECT conname FROM pg_constraint
        WHERE conrelid = 'daily_nlg_log'::regclass AND contype = 'u'
    """)
    existing = {r[0] for r in cur.fetchall()}
    # เช็คทั้งชื่อของ migration นี้ และชื่อที่ migrate_province_column() เปลี่ยนไปแล้ว กัน
    # UniqueViolation ตอนรันซ้ำหลัง migrate_province_column() ทำงานไปแล้ว (เจอบั๊กนี้จริงตอนทดสอบ)
    if ("daily_nlg_log_date_thai_risk_group_key" not in existing
            and "daily_nlg_log_date_thai_risk_group_province_key" not in existing):
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


def migrate_province_column():
    """เพิ่ม province ให้ 3 ตาราง (แถวเก่าทั้งหมดเป็นขอนแก่นจริง จึง DEFAULT 'khonkaen' ไม่ใช่
    placeholder) แล้วสลับ UNIQUE constraint ให้รวม province ด้วย (pattern เดียวกับ
    migrate_daily_nlg_risk_group()) ต้องรันหลัง migrate_daily_nlg_risk_group() เสมอ"""
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
            for name in existing:  # ลบ unique constraint เก่า (ไม่รวม province) — ชื่อมาจาก
                cur.execute(f"ALTER TABLE {table} DROP CONSTRAINT {name}")  # pg_constraint catalog เอง ไม่ใช่ input ภายนอก
            new_cols = ", ".join(unique_cols + ["province"])
            cur.execute(f"ALTER TABLE {table} ADD CONSTRAINT {new_constraint_name} UNIQUE ({new_cols})")
    conn.commit()
    cur.close()
    conn.close()


migrate_province_column()


def migrate_prediction_interval_columns():
    """เพิ่ม pm25_lo/pm25_hi ให้ prediction_log — get_forecast() คำนวณอยู่แล้วแต่ไม่เคย log ลง DB
    ทำให้ /api/comparison วัด empirical coverage ไม่ได้ (metric หลักของงานวิจัยนี้) แถวเก่าก่อน
    migration นี้เป็น NULL ถาวร (ไม่มี quantile prediction ย้อนหลังให้ backfill) — /api/comparison*
    ต้องกรอง NULL ออกเอง"""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("ALTER TABLE prediction_log ADD COLUMN IF NOT EXISTS pm25_lo DOUBLE PRECISION")
    cur.execute("ALTER TABLE prediction_log ADD COLUMN IF NOT EXISTS pm25_hi DOUBLE PRECISION")
    conn.commit()
    cur.close()
    conn.close()


migrate_prediction_interval_columns()


def migrate_users_role():
    """เพิ่ม role ให้ users (แยกจาก risk_group — risk_group คือกลุ่มสุขภาพ ใช้ปรับ threshold/NLG,
    role คือสิทธิ์ใช้งานระบบ ไม่เกี่ยวกัน) ค่าเริ่มต้น 'user' ทุกแถวเก่า ไม่มีทางสมัครเป็น 'admin' เอง
    ได้ผ่าน /api/auth/register (ดู api_auth_register — ไม่รับ field role จาก request เลย) ต้องตั้งผ่าน
    scripts/promote_admin.py เท่านั้น"""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("ALTER TABLE users ADD COLUMN IF NOT EXISTS role TEXT NOT NULL DEFAULT 'user'")
    conn.commit()
    cur.close()
    conn.close()


migrate_users_role()


def migrate_users_home_province():
    """เพิ่ม home_province ให้ users (nullable, ไม่มี default — "ยังไม่ตั้ง" เป็นสถานะปกติ ต่างจาก
    risk_group ที่มี default 'ทั่วไป' เพราะจังหวัดไม่มีค่าเริ่มต้นที่สมเหตุสมผล) เก็บเป็น province_en
    slug เดียวกับที่ใช้ทั้งระบบ ใช้ให้กระดิ่งแจ้งเตือนเฝ้าจังหวัดนี้แทน provinceState.selected (dropdown)
    เมื่อตั้งไว้ — ดู resolve_home_province_status()/api_notifications_status()"""
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
    # latest_pm25: ค่าจริงล่าสุดจาก OpenAQ ตรงๆ ใช้ทั้งแสดง current, เทียบ threshold, และบวกกลับ
    # ตอน predict (residual forecasting — ดู get_forecast())
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
    """เก็บ latest_pm25 ของรอบนี้ — recorded_at_utc ต้องตรงกับ predicted_at_utc ของ log_predictions()
    (มาจาก latest_time เดียวกัน) ให้ /api/notifications join ได้ตรง ON CONFLICT DO NOTHING เหมือนกัน"""
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
    """เติมค่าจริงให้ prediction ที่ log ไว้ก่อนหน้า จาก PM2.5 ย้อนหลังที่เพิ่งดึงมา (ไม่ยิง API เพิ่ม)
    กรอง province เสมอ (กัน backfill ข้ามจังหวัดที่ target_time_utc บังเอิญตรงกัน) actual_pm25_calibrated
    เก็บค่าเดียวกับ actual_pm25 ตรงๆ (คงคอลัมน์ไว้เผื่อ query เดิมอ้างถึง)"""
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
# 5 ระดับ ค่าเฉลี่ย 24 ชม.: 0-15 ดีมาก, 15.1-25 ดี, 25.1-37.5 ปานกลาง, 37.6-75 เริ่มมีผลกระทบต่อสุขภาพ,
# 75.1+ มีผลกระทบต่อสุขภาพ (ยืนยันจาก pcd.go.th/pcd_news/30028 — cutoff 25/37.5 ตรงกับ HIGH_POLLUTION_THRESHOLD)
PM25_LEVELS = [
    (15, "ดีมาก", ["ดีมาก", "อากาศดีมาก"]),
    (25, "ดี", [("อากาศ", "ดี"), ("คุณภาพอากาศ", "ดี"), "ปลอดภัย"]),
    (37.5, "ปานกลาง", ["ปานกลาง"]),
    (75, "เริ่มมีผลกระทบต่อสุขภาพ", ["เริ่มมีผลกระทบต่อสุขภาพ", "เริ่มมีผลกระทบ", "ส่งผลกระทบต่อสุขภาพ"]),
    (float("inf"), "มีผลกระทบต่อสุขภาพ", ["มีผลกระทบต่อสุขภาพ", "อันตราย", "อันตรายถึงชีวิต"]),
]
# keyword เป็น str (ตรงตัวอักษรเป๊ะ) หรือ tuple ของ chunk (ยอมให้มีคำแทรกสั้นๆ คั่นได้ เช่น
# ("อากาศ","ดี") match ทั้ง "อากาศดี" และ "อากาศจะอยู่ในเกณฑ์ดี" — ดู _find_keyword_occurrences())

# คำเตือนทั่วไปที่ใช้ได้ตั้งแต่ระดับหนึ่งขึ้นไป (ไม่ผูกกับระดับเดียวเป๊ะแบบ PM25_LEVELS) เก็บเป็น
# (floor ต่ำสุดที่ใช้ได้, keywords) เช็คแค่ฝั่ง "รุนแรงเกินจริง" (ระดับจริงต่ำกว่า floor)
_GENERIC_CAUTION_KEYWORDS = [
    ("ปานกลาง", [("หลีกเลี่ยง", "กิจกรรมกลางแจ้ง"), ("ลด", "กิจกรรมกลางแจ้ง"), "ระคายเคือง"]),
]

_NEGATION_WORDS = ["ไม่ใช่", "ไม่ค่อย", "มิได้", "ไม่"]  # อยู่ในช่วงคำแทรก = ความหมายกลับกัน
_MAX_FLEX_GAP = 20  # ช่องว่างของ keyword ทั่วไป
_MAX_FLEX_GAP_STATUS = 50  # PM25_LEVELS ข้ามประโยคได้ (เช่น Gemini แทรกวลีบอกเวลายาวๆ) จึงให้ช่องกว้างกว่า


def _classify_pm25_level(value):
    for threshold, level_name, _ in PM25_LEVELS:
        if value <= threshold:
            return level_name
    return PM25_LEVELS[-1][1]


def pm25_status(value):
    return _classify_pm25_level(value)


# ===== NLG Safety Check (proposal 6.1.4) =====
# เทียบระดับคำแนะนำจาก NLG กับ PM25_LEVELS ก่อนส่งแจ้งเตือน ไม่สอดคล้องก็ใช้ template แทน — rule-based ล้วนๆ
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
    """คืน list ของ (start, end) ที่ keyword ปรากฏ — str ต้องตรงเป๊ะ, tuple ของ chunk ยอมให้มีคำแทรก
    สั้นๆ คั่นได้ (เช่น ("อากาศ","ดี") match "อากาศจะอยู่ในเกณฑ์ดี") แต่ไม่ match ถ้าคำแทรกมีคำปฏิเสธ
    (ความหมายกลับกัน เช่น "อากาศไม่ดี")"""
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
    """หา level ทั้งหมดที่ข้อความพูดถึงผ่าน keyword ของ PM25_LEVELS — ประมวลผล literal match ก่อน
    flexible match เสมอ (ไม่ใช่แค่เรียงตาม span ยาวสุด เพราะ flexible match ที่กินคำแทรกยาวๆ อาจดู
    ยาวกว่า literal ที่แม่นยำกว่า) แล้ว mark ช่วงที่จับแล้วว่า "consumed" กันซ้อนจับซ้ำ — จำเป็นเพราะ
    ชื่อระดับบางคู่เป็น substring กันเอง (เช่น "อากาศดี" อยู่ใน "อากาศดีมาก")"""
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
    """True ถ้าข้อความมีคำเตือนทั่วไปที่ floor สูงกว่าระดับจริง (เช่น เตือน "หลีกเลี่ยงกิจกรรมกลางแจ้ง"
    ทั้งที่ระดับจริงคือ "ดีมาก" ซึ่งต่ำกว่า floor "ปานกลาง")"""
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
    """ตรวจว่าข้อความ NLG สอดคล้องกับระดับ PM2.5 จริงก่อนส่งแจ้งเตือน — ไม่ผ่านถ้า: keyword ของ
    ระดับที่รุนแรงกว่าจริง, keyword ของระดับที่เบากว่าจริงทั้งหมด (ชะล่าใจ), หรือไม่มี keyword เลย
    (fail-safe) risk_group != "ทั่วไป" ข้าม _mentions_generic_caution_below() เพราะคำเตือนล่วงหน้า
    เป็นเจตนาสำหรับกลุ่มเปราะบาง (สอดคล้อง RISK_GROUP_THRESHOLD_OFFSET) ไม่ใช่อ้างระดับผิด

    คืนค่า (is_valid, level หรือ None, message_to_send) — ไม่ผ่านจะคืน template แทน
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
# แปลงค่าพยากรณ์เป็นคำอธิบายไทยสั้นๆ ผ่าน Gemini ตรวจด้วย validate_health_message() ก่อนคืนค่าเสมอ
try:
    _gemini_client = genai.Client(api_key=GEMINI_KEY) if GEMINI_KEY else None
except Exception as e:
    print(f"[nlg] Gemini client init failed ({e}); NLG จะ fallback ไป template เสมอจนกว่าจะตั้ง GEMINI_KEY")
    _gemini_client = None
GEMINI_MODEL = "gemini-3.1-flash-lite"  # gemini-2.5-flash เลิกให้ API key ใหม่ใช้แล้ว — GA, free tier, เร็ว/ถูกสุด
GEMINI_TIMEOUT_SECONDS = 5
GEMINI_MAX_RETRIES = 3  # 429 (quota) backoff: รอ 1s, 2s, 4s
# Gemini SDK บังคับ min server-side deadline 10s เอง จึงบังคับ timeout ฝั่ง client ผ่าน thread แยกแทน
# (thread เดิมรันต่อเบื้องหลังจนจบ ไม่ถูก cancel แต่ caller ไม่ต้องรอ กัน get_forecast() ค้าง)
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

# แต่ละ tone กำกับ "ต้องสอดคล้องกับระดับจริง" เสมอ กันข้อความย้อนแย้งในตัวเอง (เคยเกิดจริง: อากาศ
# "ดี" แต่ tone กลุ่มเปราะบางสั่งเตือนแบบไม่มีเงื่อนไข ได้ข้อความที่ขัดแย้งกันเอง)
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
    """เรียก Gemini พร้อม client-side timeout (ดู _gemini_executor) + retry backoff เฉพาะ 429
    คืน None ถ้าไม่สำเร็จไม่ว่าสาเหตุใด ให้ผู้เรียก fallback ไป template ได้ทันที"""
    if _gemini_client is None:
        return None
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
# สร้างคำอธิบายแค่ 1 ครั้ง/วันตอน 06:00 น. เวลาไทย จากค่าเฉลี่ยพยากรณ์ 24 ชม. เก็บใน daily_nlg_log กัน generate ซ้ำ

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


def generate_daily_nlg_explanation(forecast, feature_row, risk_group="ทั่วไป"):
    """สร้างคำอธิบาย NLG แทนทั้งวัน จากค่าเฉลี่ยของ forecast[] ทั้ง 24 จุด validate ผ่าน
    validate_health_message() เสมอ ไม่ผ่าน/เรียก Gemini ไม่สำเร็จ fallback ไป _fallback_message()
    risk_group ปรับโทนผ่าน _USER_GROUP_TONE คืนค่า (daily_avg_pm25, explanation_text, used_fallback)
    """
    daily_avg_pm25 = round(sum(f["pm25"] for f in forecast) / len(forecast), 1)
    status = pm25_status(daily_avg_pm25)

    # SHAP ปิดไว้ชั่วคราว (ดูคอมเมนต์ตอนบนไฟล์ที่เคย import get_shap_explanation) — ส่ง None
    # ไปตรงๆ, _format_shap_factor_lines() รองรับ None อยู่แล้ว (fallback เป็น "ไม่มีข้อมูลปัจจัยเพิ่มเติม")
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


def maybe_generate_daily_nlg(forecast, feature_row, risk_group, province):
    """เรียก generate_daily_nlg_explanation() ครั้งเดียวต่อวันต่อ (risk_group, province) ตั้งแต่
    06:00 น. เวลาไทยเป็นต้นไป (>= 6 กัน process ที่ไม่ได้รันตรงช่วง 06:00-06:59 พลาดไปทั้งวัน) ไม่
    raise exception ออกไป กัน /api/forecast พัง — เป็น lazy ตาม (risk_group, province) ที่มีคนขอ
    (กลุ่ม 'ทั่วไป' ถูก warm ทุกวันผ่าน scheduled_forecast_job())
    """
    thai_now = get_thai_now()
    if thai_now.hour < 6:
        return

    date_thai = thai_now.strftime("%Y-%m-%d")
    if get_daily_nlg_row(date_thai, risk_group, province) is not None:
        return

    try:
        daily_avg_pm25, explanation_text, used_fallback = generate_daily_nlg_explanation(forecast, feature_row, risk_group)
        generated_at_utc = thai_now.tz_convert("UTC").isoformat()
        save_daily_nlg_row(date_thai, risk_group, province, daily_avg_pm25, explanation_text, used_fallback, generated_at_utc)
        print(f"[nlg-daily] generated for {date_thai} ({risk_group}, {province}): avg_pm25={daily_avg_pm25}, fallback={used_fallback}")
    except Exception as e:
        print(f"[nlg-daily] generation failed for {date_thai} ({risk_group}, {province}): {e}")


def get_today_daily_nlg(risk_group, province):
    """อ่าน NLG รายวันของวันนี้ (ไม่เรียก Gemini ใหม่ — สร้างจริงที่ maybe_generate_daily_nlg() เท่านั้น)
    คืน placeholder แทน error ถ้ายังไม่มีแถวของวันนี้"""
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


MIN_LIVE_FOR_SUMMARY = 5  # เกณฑ์เดียวกับ compute_rolling_r2(): live pair น้อยกว่านี้ไม่น่าเชื่อถือพอ


def summarize_errors(all_errors, live_errors):
    """ใช้ error จาก live เท่านั้นถ้ามีพอ (>= MIN_LIVE_FOR_SUMMARY) กัน MAE/bias ดูดีเกินจริงจาก
    backfill (optimistic กว่า live จริง) ไม่พอให้ fallback ไปรวม backfill (basis='mixed')
    คืนค่า (mae, bias, n_used, basis)"""
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
    # เติมค่าช่วงที่สถานีเงียบ (ถ้าเข้าเงื่อนไข) — df_observed คือข้อมูลจริงล้วน ใช้กับ backfill_actuals()/อายุข้อมูลเสมอ ห้ามใช้ df ที่เติมแล้ว
    df_observed = df_recent
    imputed_info = None
    if IMPUTATION_CFG is not None:
        df_recent, imputed_info = impute_target_series(province, df_recent)
    imputed = imputed_info is not None

    # feature เพื่อนบ้าน + จุดความร้อน (CHANGELOG 141/149) — เวลาอ้างอิงเดียวกับที่ใช้คิด lag; คำนวณ/ดึงเฉพาะเมื่อโมเดลที่โหลดใช้ feature นั้นจริง
    reference_time = df_recent["datetime_utc"].max()
    # สภาพอากาศพร้อมทางสำรอง (CHANGELOG 172): ค่าที่ดึงไว้ -> สด -> ค่าเก่า <=2 ชม. -> climatology; source != live เก็บไว้ใน data_quality
    current_weather, weather_info = get_current_weather(province_meta, at=reference_time)
    # ค่าจากทางสำรองที่ยังไม่ผ่านเกณฑ์ "เทียบเท่าของจริง": ซีรีส์ pm25 จาก cache เก่า / สภาพอากาศ climatology -> ไม่ log (กัน R² ของ routing ปน)
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

    # reading ล่าสุดของจังหวัดนี้เก่ากว่า TARGET_STALE_HOURS: ยังตอบได้ (พร้อม flag ใน data_quality) แต่ไม่ log/ไม่ทำ NLG
    # — target_time ของ horizon 1-24 จะตกอยู่ในอดีต ถูก backfill ค่าจริงทันที ปน rolling R² ของ routing แบบรั่ว, และ
    # daily_nlg_log UNIQUE ต่อวัน เขียนครั้งแรกชนะ สรุปจากข้อมูลเก่าจะยึดทั้งวัน
    # อายุวัดจาก reading จริงล่าสุด (ไม่ใช่ค่าที่เติม); เมื่อเติมค่าแล้วพยากรณ์นับจากเวลาปัจจุบันจริง จึงไม่ถือเป็น stale (แต่ไม่ log — ดูด้านล่าง)
    target_age_hours = (pd.Timestamp.now(tz="UTC") - df_observed["datetime_utc"].max()).total_seconds() / 3600
    target_stale = bool(target_age_hours > TARGET_STALE_HOURS) and not imputed

    # คำนวณ rolling R² ของทั้ง 2 โมเดลทุก horizon ล่วงหน้าด้วย query เดียว (ไม่ใช่ query แยกทีละ
    # (model, horizon) ในลูปข้างล่าง 48 ครั้ง) — ดูคอมเมนต์ที่ fetch_rolling_r2_lookup()
    r2_lookup = fetch_rolling_r2_lookup(province)

    forecast = []
    log_rows = []
    risk_horizons = []
    fallback_triggered_horizons = []
    for h in range(1, 25):
        model_name = select_model(X_now, h, province, r2_lookup, force_naive)

        if model_name == MODEL_NAIVE:
            # ข้อมูลขาดหนักเกินป้อนโมเดล pooled ได้ (ไม่เคยเห็นแถวมี NaN ตอนเทรน) ใช้ persistence
            # แทน ไม่มีขอบเขต uncertainty (ใช้ pred ตรงๆ เป็นทั้ง P05/P95 — สื่อ "ความเชื่อมั่นต่ำ"
            # ผ่าน model_used แทน)
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
            # ช่วงความเชื่อมั่น ~90% (P05-P95 conformal) — clamp ขอบล่างไม่ให้ติดลบ (PM2.5 เป็นบวกเสมอ
            # แต่ residual บวกกลับอาจดันค่าต่ำกว่า 0 ได้ทางคณิตศาสตร์)
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

    # ไม่ log/ไม่ทำ NLG เมื่อ input ถูกเติมค่า (รุ่นแรก: กัน prediction_log/R² ของ routing ปนพยากรณ์จากค่าประมาณ — CHANGELOG 171)
    loggable = not target_stale and not imputed and not input_degraded
    if loggable:
        log_predictions(latest_time, log_rows, province)
        log_notification_snapshot(latest_time.isoformat(), float(latest_pm25), province)
    backfill_actuals(df_observed, province)  # เติมค่าจริงให้ prediction เก่าจากข้อมูลที่เพิ่งดึง — ต้องเป็นข้อมูลจริงล้วนเสมอ
    if loggable:
        maybe_generate_daily_nlg(forecast, X_now, risk_group, province)

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
        # current_weather ดึงมาใช้เป็น model feature อยู่แล้ว แค่ส่งต่อให้ frontend ด้วย (ไม่ยิง API เพิ่ม)
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
        # สถานะข้อมูลที่ใช้ทำพยากรณ์รอบนี้ (ไม่มี column ใน DB — บอกผ่าน response เท่านั้น)
        "data_quality": {
            "target_age_hours": round(float(target_age_hours), 1),
            # k ตามนิยามของ impute_target_series: ชั่วโมงเต็มที่ reading จริงล่าสุดตามหลัง "ชั่วโมงที่เพิ่งจบ" (now-1h) — ปกติ 0
            # (ใช้บันทึกความหน่วงปกติสะสมเพื่อตั้ง IMPUTATION_MIN_K: ดู log_target_latency / scripts/summarize_target_latency.py)
            "observed_lag_k": int((pd.Timestamp.now(tz="UTC").floor("h") - pd.Timedelta(hours=1)
                                   - df_observed["datetime_utc"].max()) / pd.Timedelta(hours=1)),
            "target_stale": target_stale,
            # แหล่งของข้อมูลนำเข้า (CHANGELOG 172): series_source = live|hours_endpoint|stale_cache, weather_source = live|stale_cache|climatology
            "series_source": series_info.get("source", "live"),
            "series_age_hours": series_info.get("age_hours", 0.0),
            "weather_source": weather_info["source"],
            "weather_age_hours": weather_info["age_hours"],
            "neighbor_status": neighbor["status"],
            "neighbor_policy": neighbor["policy"],
            "neighbors_used": neighbor["neighbors_used"],
            "fire_source": fire_source,
            "fire_count_24h": int(fire_count),
            # ไม่ null = สถานีเงียบ k ชม. ค่า current/พยากรณ์ตั้งอยู่บนค่าประมาณ (H-lite) — frontend ต้องบอกผู้ใช้
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
        # activity เลือกฝั่ง frontend เท่านั้น (localStorage) ส่งแค่ config ให้ frontend คำนวณคะแนนเอง
        # จาก pm25 ที่มีอยู่แล้ว — เลขคงที่ยังอยู่จุดเดียวที่นี่ กัน frontend/backend คำนวณไม่ตรงกัน
        "risk_score_config": {
            "risk_group_offset": RISK_GROUP_THRESHOLD_OFFSET.get(risk_group, 0.0),
            "activity_offsets": ACTIVITY_INTENSITY_OFFSET,
            "breakpoints": RISK_SCORE_BREAKPOINTS,
        },
    }


TARGET_LATENCY_LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "target_latency_scheduler_log.csv")


TARGET_LATENCY_LOG_HEADER_OLD = "measured_utc,province_en,observed_lag_k,target_age_hours,imputed"  # ก่อน CHANGELOG 173
TARGET_LATENCY_LOG_HEADER = TARGET_LATENCY_LOG_HEADER_OLD + ",series_source"
_LATENCY_LOG_LOCK = threading.Lock()  # scheduler thread + request thread เขียนไฟล์เดียวกัน (และตอนย้าย header ต้องไม่มี append แทรก)
_LATENCY_LOG_CHECKED = set()  # path ที่ตรวจ/ย้าย header แล้วในโปรเซสนี้


def _upgrade_latency_log_header(path):
    """ไฟล์ log เก่า (header 5 คอลัมน์) -> แก้บรรทัดแรกเป็น 6 คอลัมน์ในที่เดิม (เขียน tmp แล้ว replace) แถวเก่าคงเดิม ไม่มีฟิลด์ series_source
    = "legacy" (เขียนก่อนมีทางสำรอง จึงมาจาก /measurements เสมอ — summarize_target_latency.py ตีความเป็น live) header อื่นที่ไม่รู้จัก -> ไม่แตะ"""
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
    """ต่อท้ายไฟล์ CSV: เวลาที่วัด (UTC), จังหวัด, observed_lag_k, อายุข้อมูล (ชม.), เติมค่าไหม, series_source — สะสมจากงานรายชั่วโมงเพื่อดูการกระจายความหน่วงปกติ
    ของ OpenAQ ตามนาทีของชั่วโมง/จังหวัด ก่อนสรุปค่า IMPUTATION_MIN_K (scripts/summarize_target_latency.py) series_source (live|hours_endpoint|stale_cache) ให้ตัดแถวที่มาจาก
    ทางสำรองออกจากการสรุป: /hours ช้ากว่า /measurements ~1 ชม. และ stale_cache มี k โตตามอายุ cache ไม่ใช่ความหน่วงปกติ (CHANGELOG 172-173)
    ห้าม raise: ล้มเหลวเงียบๆ ได้ เพราะเป็นแค่ telemetry ไม่ใช่ส่วนของการพยากรณ์ (ไม่เขียนคีย์/URL ใดๆ)"""
    try:
        path = TARGET_LATENCY_LOG_PATH
        with _LATENCY_LOG_LOCK:
            new_file = not os.path.exists(path) or os.path.getsize(path) == 0
            if not new_file and path not in _LATENCY_LOG_CHECKED:
                try:
                    _upgrade_latency_log_header(path)
                except Exception:
                    pass  # ย้าย header ไม่ได้ก็ยังต่อท้ายแถวใหม่ได้ (สคริปต์สรุปอ่านด้วย names= จึงไม่พัง)
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
    """เรียก get_forecast() ทุกชั่วโมงต่อจังหวัด (loop ทั้ง 20 จังหวัด) ให้ prediction_log สะสม
    ข้อมูลต่อเนื่องแม้ไม่มีใครเปิดเว็บ — try/except ต่อจังหวัด กัน API hiccup จังหวัดเดียวล้ม job ทั้งหมด"""
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
    """เสิร์ฟ Vue SPA (frontend/dist) — path ที่ไม่ใช่ไฟล์จริง (เช่น client-side route /compare)
    fallback ไป index.html เสมอ Werkzeug จับ static route อย่าง /api/forecast ก่อน catch-all นี้เสมอ"""
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
    """สรุปสถานะ PM2.5 ทุกจังหวัดพร้อมกัน (หน้า /map) — อ่านจาก notification_log/prediction_log แทนการ
    เรียก get_forecast() 20 ครั้งสด (แพง: OpenAQ/Open-Meteo/FIRMS/neighbor ต่อจังหวัด) ค่าที่ได้จึงหน่วง
    สุด ~1 ชม. ตามรอบ scheduled_forecast_job() ยอมรับได้สำหรับภาพรวมระดับจังหวัด ไม่ใช่หน้าเดี่ยวที่ต้อง
    real-time เป๊ะ (หน้านั้นคือ / ซึ่งยังคง live เหมือนเดิม)

    ?horizon=0 (ค่าเริ่มต้น): คืนแค่ current_* (ค่าปัจจุบันจาก notification_log)
    ?horizon=1..24: คืน current_* คู่กับ forecast_* (ค่าพยากรณ์ ณ horizon นั้นจาก prediction_log) เพื่อให้
    popup ฝั่ง frontend โชว์ "ตอนนี้" เทียบกับ "พยากรณ์ที่เลือก" ได้โดยไม่ต้อง fetch ซ้ำ"""
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

            # ยังไม่มี snapshot เลย (DB ใหม่/scheduler ยังไม่เคยรันสำเร็จจังหวัดนี้) — ส่ง null ไม่ raise
            # ให้จังหวัดอื่นยังโชว์ได้ปกติ
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
        log_target_latency(province, result["data_quality"])  # คำขอจากผู้ใช้กระจายตามนาทีของชั่วโมง (งาน scheduler รันนาทีคงที่)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except DataUnavailableError as e:
        return jsonify({"error": str(e), "data_unavailable": True}), 503
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/notifications", methods=["GET"])
def api_notifications():
    """ประวัติการแจ้งเตือน (เฉพาะรอบที่ high_pollution_risk=true) — คำนวณ risk_horizons สดตาม
    risk_group ของผู้เรียก ณ เวลา query (ไม่ใช่ค่า freeze ไว้ตอนบันทึก) กรอง province เสมอ กัน
    cross จังหวัดที่ timestamp ตรงกันบังเอิญ ใช้ predicted_pm25 ตรงๆ เข้า is_horizon_at_risk() —
    วิธีเดียวกันเป๊ะกับ get_forecast() แล้ว (ทั้งคู่ใช้ค่าพยากรณ์ตรงๆ ไม่ใช่ P95 อีกต่อไป ตั้งแต่
    2026-09-23) ไม่ใช่การประมาณหยาบกว่าเหมือนก่อนหน้านี้
    """
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
            (province, limit),
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

        cur.close()
        conn.close()
        return jsonify(results)
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/notifications/status", methods=["GET"])
def api_notifications_status():
    """สถานะเสี่ยงปัจจุบันของจังหวัดที่ระบุ อ่านจาก notification_log/prediction_log ล่าสุดเท่านั้น
    (ไม่เรียก get_forecast()/live API — ถูกกว่ามาก) ใช้สำหรับกระดิ่งแจ้งเตือนเฝ้า "จังหวัดหลัก" ของ
    user (home_province) แยกจากจังหวัดที่เลือกดูอยู่ใน dropdown (provinceState.selected) — เดียวกับ
    pattern ที่ /api/provinces/overview ใช้สำหรับหน้าแผนที่ ข้อมูลหน่วงสุด ~1 ชม. ตามรอบ scheduler"""
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
        # (ดู backfill_actuals()) COALESCE ไว้เผื่อแถวที่ยังไม่ผ่าน backfill
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
            # ค่าจริงตกในช่วง P05-P95 ไหม — None ถ้ายังไม่มีค่าจริง/แถวเก่าไม่มี pm25_lo/pm25_hi
            # ไม่นับ naive_fallback (ช่วงกว้าง 0 เสมอ ไม่ใช่ conformal interval จริง)
            in_interval = None
            if actual_calibrated is not None and pm25_lo is not None and pm25_hi is not None and model_used != "naive_fallback":
                in_interval = pm25_lo <= actual_calibrated <= pm25_hi
                in_interval_flags.append(in_interval)
            points.append({
                "time": target_time_th.strftime("%m-%d %H:%M"),
                "predicted": round(predicted, 1),
                # actual_raw/actual เท่ากันเสมอสำหรับแถวใหม่ เก็บทั้งคู่ไว้ไม่ต้องแก้ frontend ที่อ้างถึง
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
                # % ของคู่ที่มีช่วงบันทึกไว้ (n_coverage) ที่ค่าจริงตกในช่วงจริง — เป้าหมาย ~90% (ALPHA=0.10)
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
        # ไม่มี calibration อีกต่อไป (ดูคอมเมนต์เดียวกันใน /api/comparison)
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
    # DEBUG=True ทำให้ Flask สร้าง reloader process แยก กันไว้ไม่ให้ scheduler เริ่มซ้ำ 2 รอบ
    # (WERKZEUG_RUN_MAIN ตั้งเฉพาะใน process ลูกที่รันจริง)
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

# scripts/ — ดัชนีไฟล์

มี 143 ไฟล์ `.py` ใน `scripts/` สะสมมาตลอดการพัฒนาโปรเจกต์ (ไม่มีไลบรารีร่วม แต่ละไฟล์คือ entry point เดี่ยวๆ ตาม CLAUDE.md — ลดจาก 175 ไฟล์เดิมเมื่อ 2026-10-22 เป็น 2 รอบ: (1) ลบ `shap_analysis.py`/`shap_analysis_nbtrend.py`/`backfill_xgboost.py`/`retrain_weighted.py`/`train_xgb_prophet.py` ที่ CLAUDE.md/CHANGELOG ยืนยันแล้วว่าถูกแทนที่ด้วย `_clean`/`_v2`; (2) ไล่เช็ค `MODEL_TAG`/`DATA_PATH` จริงในโค้ดของทุกกลุ่มที่ชื่อคล้ายกัน (`_clean`/`_nbtrend`/`_firehotspot`/`_k1`/`_latest`) พบว่าทุกกลุ่มเป็น**รุ่นอัปเกรดที่แทนกันได้จริง ไม่ใช่การทดลองคนละสาย** — ของเก่าชี้ไป MODEL_TAG/dataset ที่เลิกใช้แล้ว (เช่น `_20260919_135655` ที่ CLAUDE.md ระบุว่าปนเปื้อนข้อมูลเวลาเพี้ยน) ของปัจจุบันมีแต่ `_clean.py` (หรือ `_k1.py` สำหรับ `augment_raw_dataset_with_spatial_lag`) ที่ชี้โมเดล/dataset ที่ deploy จริงตัวเดียว — ลบไปอีก 27 ไฟล์ ยืนยันด้วย grep ว่าไม่มีสคริปต์ที่เหลือ import ไฟล์ที่ลบไปเลย) ไฟล์นี้เป็นดัชนีช่วยหา — **ไม่มีการย้าย/เปลี่ยนชื่อไฟล์ที่เหลือ** เพราะหลายสิบไฟล์ผูก path กันแบบ hardcode ผ่าน `importlib.util.spec_from_file_location` + `os.path.join(_ROOT, "scripts", ...)` และ `app.py:29` เองก็ `sys.path.insert` ไปที่ `scripts/` ตรงๆ แบบ flat directory — การย้ายไฟล์เข้าโฟลเดอร์ย่อยจะพังเป็นลูกโซ่โดยไม่จำเป็น

## Core / production

ไฟล์หลักที่ `app.py` import ตรง หรือเป็น canonical pipeline ปัจจุบันตาม CLAUDE.md:

- `province_registry.py` — สร้าง registry 20 จังหวัด (query OpenAQ v3 โดยตรง) — ใช้ตอนตั้งโปรเจกต์/รีเฟรช registry
- `build_raw_dataset_multiprovince.py` — pipeline เตรียมข้อมูล step 1 — ฟีเจอร์ดิบของทุกจังหวัด (มี offset fix)
- `augment_raw_dataset_with_spatial_lag_k1.py` — pipeline step 2 — เพิ่ม spatial-lag เพื่อนบ้าน k=1
- `compute_fire_hotspot_feature.py` — pipeline step 3 — fire_hotspot_count_100km_24h จาก FIRMS
- `compute_neighbor_trend_feature.py` — pipeline step 4 — nb_trend_12h
- `rebuild_dataset_clean.py` — orchestrator เรียก 4 step ข้างบนตามลำดับ ใช้ตัวนี้แทนรันทีละไฟล์
- `verify_clean_dataset_alignment.py` — ตรวจสอบ timestamp alignment ของ dataset ที่ rebuild แล้ว
- `annotate_registry_timestamp_offset.py` — วัด/เขียน openaq_timestamp_offset_hours ลง registry
- `train_multiprovince_models.py` — **สคริปต์เทรนหลัก** (LightGBM+XGBoost pooled v3) — เป็น source of truth ของ FEATURE_COLS/DATA_PATH/split ที่สคริปต์ทดลองอีก ~35 ไฟล์โหลดผ่าน importlib
- `backtest_recent_multiprovince.py` — backtest ช่วงหลังตัด training window
- `eval_zeroshot_multiprovince.py` — ประเมิน zero-shot transfer (phase 2b)
- `eval_zeroshot_real_provinces_clean.py` — ประเมิน zero-shot กับ 3 จังหวัดจริง (ใช้สำหรับเล่ม/ตอบกรรมการ)
- `validate_features_timeblocked_cv.py` — วิธีมาตรฐานของโปรเจกต์สำหรับตัดสินใจเพิ่ม/ตัดฟีเจอร์ (time-blocked CV ครอบคลุมฤดูฝุ่น)
- `shap_analysis_clean.py` — **ถูก import ตรงโดย app.py** (`from shap_analysis_clean import get_shap_explanation`) — ตัวเดียวจาก 3 ตัวแปร shap_analysis* ที่ชี้โมเดล v3 ที่ deploy จริง
- `experiment_imputation_accuracy.py` — source of truth ของ IMPUTATION_FEATURES ridge formula (H-lite)
- `export_imputation_h_lite.py` — export/ตรวจสอบ config imputation ที่ app.py ใช้จริง
- `summarize_target_latency.py` — คำนวณ IMPUTATION_MIN_K จาก latency log จริง
- `promote_admin.py` — ops utility — ทางเดียวที่สร้าง admin user ได้

## งานล่าสุดสำหรับเล่มวิทยานิพนธ์ (Progress 2 feedback)

- `train_lstm_multiprovince.py` — เทรน Hybrid LSTM+Dense (survey โมเดลเทียบ LightGBM/XGBoost ตาม comment กรรมการ 'โมเดลทำ survey ใหม่')
- `train_lstm_plain_multiprovince.py` — LSTM แบบ lag-only (ไม่มี static branch) — baseline ที่แย่กว่าโดยตั้งใจ เพื่อเทียบอย่างเป็นธรรม ไม่ปิดบังวิธี
- `compare_zeroshot_neighbor_pm25.py` — เทียบค่า PM2.5 จริงของจังหวัด zero-shot กับเพื่อนบ้านที่เทรนจริง (ตอบ comment '1 จุดต่อ 1 จังหวัด')
- `compare_zeroshot_neighbor_predicted_vs_actual.py` — ขยายผลข้างบน: เทียบค่าพยากรณ์ t+24 ของ zero-shot กับค่าจริงของเพื่อนบ้าน (หลักฐานว่าโมเดลสะท้อนพฤติกรรมจริง ไม่ใช่แค่ข้อมูลดิบสัมพันธ์กัน)

## Legacy (เก็บไว้เป็นประวัติ ไม่ใช้งานจริงแล้ว)

- `build_raw_dataset.py` — pipeline เตรียมข้อมูลยุค Khon Kaen เดี่ยว (ก่อน pooled multi-province) — เก็บไว้เป็น reference ตามธรรมเนียม ไม่ได้ใช้งานจริงแล้ว
- `archive_air4thai.py` — snapshot ข้อมูล Air4Thai ยุคที่ live inference ยังดึงจาก Air4Thai — เลิกใช้แล้ว (เปลี่ยนเป็น OpenAQ-only) เก็บไว้เป็นประวัติ

## Tests (14 ไฟล์)

รันมือ ไม่มี pytest (ดู CLAUDE.md "Commands") — ทุกไฟล์ขึ้นต้น `test_`:

- `test_auth_routes.py`
- `test_best_time_window.py`
- `test_daily_nlg.py`
- `test_dynamic_routing.py`
- `test_effective_threshold_by_model.py`
- `test_evaluate_live_v2_vs_persistence.py`
- `test_imputation_methods.py`
- `test_input_fallback.py`
- `test_live_features_v2.py`
- `test_live_imputation.py`
- `test_nlg_safety_check.py`
- `test_notification_history.py`
- `test_risk_group_e2e.py`
- `test_risk_score.py`

## Experiments / diagnostics (105 ไฟล์)

การทดลอง/ตรวจสอบแบบ one-off ตลอดการพัฒนา (ablation, validate, check, analyze ฯลฯ) — ผลลัพธ์ส่วนใหญ่ ไม่ได้ commit เข้า git (ตามธรรมเนียมเดิม ไม่ commit output ทุกไฟล์) จัดกลุ่มตาม prefix ชื่อไฟล์:

> **คำเตือน**: กลุ่มที่ลงท้าย `_clean`/`_nbtrend`/`_firehotspot`/`_k1`/`_latest`/`_v2` ส่วนใหญ่เป็น **การทดลองคนละสาย** (ทดสอบฟีเจอร์/dataset คนละชุด) **ไม่ใช่รุ่นอัปเกรดที่แทนกันตรงไปตรงมา** (ยกเว้น `shap_analysis_clean.py` ที่ CLAUDE.md ฟันธงชัดว่าเป็นตัวที่ใช้จริง ตัวอื่นในกลุ่มเดียวกันห้าม import เข้า app.py) — ก่อนจะถือว่าไฟล์ไหน "เก่ากว่าเลิกใช้ได้" ให้เช็ค `git log -- scripts/<ไฟล์>` และ CHANGELOG.md ก่อนเสมอ อย่าสมมติจากชื่อไฟล์เพียงอย่างเดียว

**`analyze_*`** (13): `analyze_alert_capability_per_province.py`, `analyze_amenity_density_vs_pm25_pattern_clean.py`, `analyze_burning_season_backtest_clean.py`, `analyze_dynamic_routing_live_behavior.py`, `analyze_elevation_vs_pm25_pattern_clean.py`, `analyze_humidity_raw_correlation_clean.py`, `analyze_live_vs_backtest_by_horizon.py`, `analyze_rain_vs_error.py`, `analyze_real_gap_distribution.py`, `analyze_shock_events.py`, `analyze_spatial_correlation_clean.py`, `analyze_timestamp_shift_leakage.py`, `analyze_zeroshot_lopo_confounders_clean.py`
**`check_*`** (12): `check_air4thai_missing_rate.py`, `check_fallback_interval_coverage.py`, `check_firms_nrt_vs_sp.py`, `check_live_data_availability.py`, `check_live_feature_parity.py`, `check_loei_station_status.py`, `check_model_sensitivity_live_degradation.py`, `check_neighbor_staleness_sensitivity_clean.py`, `check_openaq_missing_rate_5y.py`, `check_openaq_timestamp_shift.py`, `check_quantile_heads_equal_width.py`, `check_rolling_r2_stability.py`
**`validate_*`** (10): `validate_amenity_density_feature.py`, `validate_drop_neighbor_avg_cv.py`, `validate_drop_time_cyclical.py`, `validate_elevation_feature.py`, `validate_feature_refinements.py`, `validate_features_clean.py`, `validate_fire_hotspot_feature.py`, `validate_fire_wide_features_cv.py`, `validate_neighbor_trend_variants.py`, `validate_spatial_lag_feature.py`
**`eval_*`** (7): `eval_ensemble_vs_routing_clean.py`, `eval_haze_season_backtest.py`, `eval_high_pollution.py`, `eval_xgb_bias_high_by_horizon.py`, `eval_xgb_posthoc_correction.py`, `eval_zeroshot_lopo_spatial_distance.py`, `eval_zeroshot_spatial_distance.py`
**`experiment_*`** (7): `experiment_diurnal_nb_trend_cv.py`, `experiment_diurnal_nb_trend_hypothesis.py`, `experiment_imputation_downstream.py`, `experiment_imputation_interval_recalibration.py`, `experiment_weather_fallback.py`, `experiment_weather_horizon_oracle.py`, `experiment_weather_train_serve_skew.py`
**`fetch_*`** (7): `fetch_extra_weather_isaan.py`, `fetch_firms_hotspots_isaan.py`, `fetch_firms_hotspots_wide.py`, `fetch_province_amenity_density.py`, `fetch_province_building_density.py`, `fetch_province_elevation.py`, `fetch_raw_pm25_isaan.py`
**`diagnose_*`** (5): `diagnose_amenity_density.py`, `diagnose_boundary_layer_height.py`, `diagnose_building_count_anomaly.py`, `diagnose_firms_hotspot_access.py`, `diagnose_mahasarakham_sensor_quality.py`
**`recalibrate_*`** (5): `recalibrate_conformal_by_volatility_k1.py`, `recalibrate_conformal_configs_clean.py`, `recalibrate_conformal_per_province_global_severity_k1.py`, `recalibrate_conformal_per_province_severity_k1.py`, `recalibrate_conformal_per_severity.py`
**`tune_*`** (5): `tune_fire_hotspot_hyperparams.py`, `tune_model_hyperparams_pooled.py`, `tune_quantile_heads_pooled.py`, `tune_sample_weight_pooled.py`, `tune_spatial_lag_k.py`
**`compute_*`** (3): `compute_accuracy_mape_clean.py`, `compute_alert_capability_hourly.py`, `compute_dynamic_routing_table14_row.py`
**`confirm_*`** (3): `confirm_fire_upwind_on_test.py`, `confirm_improvements_on_test.py`, `confirm_v3_on_test.py`
**`plot_*`** (3): `plot_bias_comparison.py`, `plot_mae_comparison.py`, `plot_r2_3models.py`
**`train_*`** (3): `train_baseline_lgb_v2.py`, `train_multiprovince_models_all20.py`, `train_xgb_prophet_v2.py`
**`ablation_*`** (2): `ablation_feature_groups_clean.py`, `ablation_individual_features_clean.py`
**`backtest_*`** (2): `backtest_dayahead_dust_episodes.py`, `backtest_dynamic_routing_offline.py`
**`compare_*`** (2): `compare_model_vs_persistence_test.py`, `compare_raw_vs_interpolated.py`
**`evaluate_*`** (2): `evaluate_2024_2026_retrain_window.py`, `evaluate_live_v2_vs_persistence.py`
**`measure_*`** (2): `measure_gemini_latency.py`, `measure_target_data_latency.py`
**`retrain_*`** (2): `retrain_weighted_v2.py`, `retrain_xgb_weighted.py`
**`verify_*`** (2): `verify_app_feature_parity.py`, `verify_forecast_all_provinces.py`
**`backfill_*`** (1): `backfill_xgboost_v2.py`
**`debug_*`** (1): `debug_prediction_log.py`
**`dryrun_*`** (1): `dryrun_deploy_latest_model.py`
**`export_*`** (1): `export_weather_climatology.py`
**`fit_*`** (1): `fit_air4thai_openaq_calibration.py`
**`investigate_*`** (1): `investigate_time_cyclical_drop.py`
**`recompute_*`** (1): `recompute_r2_3models.py`
**`shap_*`** (1): `shap_divergence_by_horizon.py`


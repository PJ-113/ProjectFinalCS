# scripts/

สคริปต์รันเดี่ยว (`python scripts/<หมวด>/<ไฟล์>.py` จาก root ของโปรเจกต์) — ไม่มี shared library
ทุกไฟล์หา root ด้วย `_ROOT = dirname(dirname(dirname(__file__)))` และสคริปต์ที่ใช้โค้ดของอีกไฟล์จะโหลดผ่าน `importlib` ด้วย path
(เช่น `os.path.join(_ROOT, "scripts", "training", "train_models.py")`) — ย้าย/เปลี่ยนชื่อไฟล์ต้องแก้ผู้เรียกด้วย

| โฟลเดอร์ | ใช้ทำอะไร |
|---|---|
| `data/` | ดึงข้อมูลดิบ (OpenAQ, Open-Meteo, FIRMS, OSM) และสร้าง training dataset + ทะเบียนจังหวัด |
| `training/` | เทรน/จูนโมเดลที่ใช้จริง และ baseline เปรียบเทียบ (LSTM, Random Forest), export ค่าที่ `app.py` โหลด |
| `evaluation/` | วัดผลโมเดล: backtest, zero-shot, เทียบ persistence, alert capability, SHAP, ตัวเลขสำหรับเล่ม |
| `experiments/` | งานวิจัยตัดสินใจออกแบบ: เลือก feature (`feature_*`), ablation, conformal, imputation, weather fallback, ยืนยันบน test split (`confirm_*`) |
| `diagnostics/` | ตรวจข้อมูลสด/สถานี/API และ train-serve parity |
| `tools/` | เครื่องมือ dev/ops: cert มือถือ, ไอคอน badge, ตั้ง admin, จำลอง Web Push |
| `legacy/` | ระบบเก่า (ขอนแก่นสถานีเดียว, Air4Thai, Prophet, โมเดล v1/v2) — ไม่ได้ใช้แล้ว เก็บไว้เป็นประวัติ ชื่อไฟล์คงเดิม |

เทสอยู่ที่ `tests/`: `tests/unit/` = pytest ที่ CI รัน, `tests/manual/` = สคริปต์รันตรงที่แตะ DB/เครือข่ายจริง

## ชื่อเก่า → ที่อยู่ใหม่ (จัดระเบียบ 2026-11-25, CHANGELOG 278)

CHANGELOG ก่อนข้อ 278 อ้างชื่อเก่าในคอลัมน์ซ้าย

| ชื่อเดิม (`scripts/`) | ที่อยู่ใหม่ |
|---|---|
| `ablation_feature_groups_clean.py` | `scripts/experiments/ablation_feature_groups.py` |
| `ablation_individual_features_clean.py` | `scripts/experiments/ablation_individual_features.py` |
| `analyze_alert_capability_per_province.py` | `scripts/evaluation/alert_capability_per_province.py` |
| `analyze_amenity_density_vs_pm25_pattern_clean.py` | `scripts/experiments/amenity_density_vs_pm25.py` |
| `analyze_burning_season_backtest_clean.py` | `scripts/evaluation/backtest_burning_season.py` |
| `analyze_dynamic_routing_live_behavior.py` | `scripts/evaluation/dynamic_routing_live_behavior.py` |
| `analyze_elevation_vs_pm25_pattern_clean.py` | `scripts/experiments/elevation_vs_pm25.py` |
| `analyze_humidity_raw_correlation_clean.py` | `scripts/experiments/humidity_correlation.py` |
| `analyze_live_vs_backtest_by_horizon.py` | `scripts/legacy/analyze_live_vs_backtest_by_horizon.py` |
| `analyze_rain_vs_error.py` | `scripts/experiments/rain_vs_error.py` |
| `analyze_real_gap_distribution.py` | `scripts/experiments/gap_distribution.py` |
| `analyze_shock_events.py` | `scripts/legacy/analyze_shock_events.py` |
| `analyze_spatial_correlation_clean.py` | `scripts/experiments/spatial_correlation.py` |
| `analyze_timestamp_shift_leakage.py` | `scripts/experiments/timestamp_shift_leakage.py` |
| `analyze_zeroshot_lopo_confounders_clean.py` | `scripts/evaluation/zeroshot_lopo_confounders.py` |
| `annotate_registry_timestamp_offset.py` | `scripts/data/annotate_registry_timestamp_offset.py` |
| `archive_air4thai.py` | `scripts/legacy/archive_air4thai.py` |
| `augment_raw_dataset_with_spatial_lag_k1.py` | `scripts/data/add_neighbor_feature.py` |
| `backfill_xgboost_v2.py` | `scripts/legacy/backfill_xgboost_v2.py` |
| `backtest_dayahead_dust_episodes.py` | `scripts/evaluation/backtest_dayahead_dust_episodes.py` |
| `backtest_dynamic_routing_offline.py` | `scripts/evaluation/backtest_dynamic_routing.py` |
| `backtest_recent_multiprovince.py` | `scripts/evaluation/backtest_recent.py` |
| `build_raw_dataset.py` | `scripts/legacy/build_raw_dataset.py` |
| `build_raw_dataset_multiprovince.py` | `scripts/data/build_dataset.py` |
| `check_air4thai_missing_rate.py` | `scripts/legacy/check_air4thai_missing_rate.py` |
| `check_fallback_interval_coverage.py` | `scripts/evaluation/fallback_interval_coverage.py` |
| `check_firms_nrt_vs_sp.py` | `scripts/diagnostics/firms_nrt_vs_sp.py` |
| `check_live_data_availability.py` | `scripts/diagnostics/live_data_availability.py` |
| `check_live_feature_parity.py` | `scripts/diagnostics/live_feature_parity.py` |
| `check_loei_station_status.py` | `scripts/diagnostics/loei_station_status.py` |
| `check_model_sensitivity_live_degradation.py` | `scripts/evaluation/live_degradation_sensitivity.py` |
| `check_neighbor_staleness_sensitivity_clean.py` | `scripts/evaluation/neighbor_staleness_sensitivity.py` |
| `check_openaq_missing_rate_5y.py` | `scripts/legacy/check_openaq_missing_rate_5y.py` |
| `check_openaq_timestamp_shift.py` | `scripts/diagnostics/openaq_timestamp_shift.py` |
| `check_quantile_heads_equal_width.py` | `scripts/experiments/quantile_heads_equal_width.py` |
| `check_rolling_r2_stability.py` | `scripts/evaluation/rolling_r2_stability.py` |
| `compare_model_vs_persistence_test.py` | `scripts/evaluation/compare_model_vs_persistence.py` |
| `compare_raw_vs_interpolated.py` | `scripts/legacy/compare_raw_vs_interpolated.py` |
| `compare_zeroshot_neighbor_pm25.py` | `scripts/evaluation/compare_zeroshot_neighbor_pm25.py` |
| `compare_zeroshot_neighbor_predicted_vs_actual.py` | `scripts/evaluation/compare_zeroshot_neighbor_predicted.py` |
| `compute_accuracy_mape_clean.py` | `scripts/evaluation/accuracy_mape.py` |
| `compute_alert_capability_hourly.py` | `scripts/evaluation/alert_capability_hourly.py` |
| `compute_dynamic_routing_table14_row.py` | `scripts/evaluation/dynamic_routing_table14.py` |
| `compute_fire_hotspot_feature.py` | `scripts/data/add_fire_hotspot_feature.py` |
| `compute_neighbor_trend_feature.py` | `scripts/data/add_neighbor_trend_feature.py` |
| `confirm_fire_upwind_on_test.py` | `scripts/experiments/confirm_fire_upwind_on_test.py` |
| `confirm_improvements_on_test.py` | `scripts/experiments/confirm_improvements_on_test.py` |
| `confirm_v3_on_test.py` | `scripts/experiments/confirm_v3_on_test.py` |
| `debug_prediction_log.py` | `scripts/legacy/debug_prediction_log.py` |
| `diagnose_amenity_density.py` | `scripts/diagnostics/amenity_density_quality.py` |
| `diagnose_boundary_layer_height.py` | `scripts/diagnostics/boundary_layer_height_availability.py` |
| `diagnose_building_count_anomaly.py` | `scripts/diagnostics/building_count_anomaly.py` |
| `diagnose_firms_hotspot_access.py` | `scripts/diagnostics/firms_api_access.py` |
| `diagnose_mahasarakham_sensor_quality.py` | `scripts/diagnostics/mahasarakham_sensor_quality.py` |
| `dryrun_deploy_latest_model.py` | `scripts/legacy/dryrun_deploy_latest_model.py` |
| `eval_ensemble_vs_routing_clean.py` | `scripts/evaluation/ensemble_vs_routing.py` |
| `eval_haze_season_backtest.py` | `scripts/legacy/eval_haze_season_backtest.py` |
| `eval_high_pollution.py` | `scripts/legacy/eval_high_pollution.py` |
| `eval_xgb_bias_high_by_horizon.py` | `scripts/legacy/eval_xgb_bias_high_by_horizon.py` |
| `eval_xgb_posthoc_correction.py` | `scripts/legacy/eval_xgb_posthoc_correction.py` |
| `eval_zeroshot_lopo_spatial_distance.py` | `scripts/evaluation/zeroshot_lopo.py` |
| `eval_zeroshot_multiprovince.py` | `scripts/evaluation/zeroshot_transfer.py` |
| `eval_zeroshot_real_provinces_clean.py` | `scripts/evaluation/zeroshot_real_provinces.py` |
| `eval_zeroshot_spatial_distance.py` | `scripts/evaluation/zeroshot_spatial_distance.py` |
| `evaluate_2024_2026_retrain_window.py` | `scripts/experiments/retrain_window_2024_2026.py` |
| `evaluate_live_v2_vs_persistence.py` | `scripts/evaluation/live_vs_persistence.py` |
| `experiment_diurnal_nb_trend_cv.py` | `scripts/experiments/diurnal_nb_trend_cv.py` |
| `experiment_diurnal_nb_trend_hypothesis.py` | `scripts/experiments/diurnal_nb_trend_hypothesis.py` |
| `experiment_imputation_accuracy.py` | `scripts/experiments/imputation_accuracy.py` |
| `experiment_imputation_downstream.py` | `scripts/experiments/imputation_downstream.py` |
| `experiment_imputation_interval_recalibration.py` | `scripts/experiments/imputation_interval_recalibration.py` |
| `experiment_weather_fallback.py` | `scripts/experiments/weather_fallback.py` |
| `experiment_weather_horizon_oracle.py` | `scripts/experiments/weather_horizon_oracle.py` |
| `experiment_weather_train_serve_skew.py` | `scripts/experiments/weather_train_serve_skew.py` |
| `export_imputation_h_lite.py` | `scripts/training/export_imputation_h_lite.py` |
| `export_weather_climatology.py` | `scripts/data/export_weather_climatology.py` |
| `fetch_extra_weather_isaan.py` | `scripts/data/fetch_extra_weather.py` |
| `fetch_firms_hotspots_isaan.py` | `scripts/data/fetch_firms_hotspots.py` |
| `fetch_firms_hotspots_wide.py` | `scripts/data/fetch_firms_hotspots_wide.py` |
| `fetch_province_amenity_density.py` | `scripts/data/fetch_amenity_density.py` |
| `fetch_province_building_density.py` | `scripts/data/fetch_building_density.py` |
| `fetch_province_elevation.py` | `scripts/data/fetch_elevation.py` |
| `fetch_raw_pm25_isaan.py` | `scripts/data/fetch_raw_pm25.py` |
| `fit_air4thai_openaq_calibration.py` | `scripts/legacy/fit_air4thai_openaq_calibration.py` |
| `generate_dev_cert.py` | `scripts/tools/generate_dev_cert.py` |
| `generate_push_badge.py` | `scripts/tools/generate_push_badge.py` |
| `investigate_time_cyclical_drop.py` | `scripts/experiments/time_cyclical_drop.py` |
| `measure_gemini_latency.py` | `scripts/diagnostics/gemini_latency.py` |
| `measure_target_data_latency.py` | `scripts/diagnostics/target_data_latency.py` |
| `plot_bias_comparison.py` | `scripts/legacy/plot_bias_comparison.py` |
| `plot_mae_comparison.py` | `scripts/legacy/plot_mae_comparison.py` |
| `plot_r2_3models.py` | `scripts/legacy/plot_r2_3models.py` |
| `promote_admin.py` | `scripts/tools/promote_admin.py` |
| `province_registry.py` | `scripts/data/build_province_registry.py` |
| `rebuild_dataset_clean.py` | `scripts/data/rebuild_dataset_pipeline.py` |
| `recalibrate_conformal_by_volatility_k1.py` | `scripts/experiments/conformal_by_volatility.py` |
| `recalibrate_conformal_configs_clean.py` | `scripts/experiments/conformal_mondrian_configs.py` |
| `recalibrate_conformal_per_province_global_severity_k1.py` | `scripts/experiments/conformal_province_global_severity.py` |
| `recalibrate_conformal_per_province_severity_k1.py` | `scripts/experiments/conformal_province_severity.py` |
| `recalibrate_conformal_per_severity.py` | `scripts/experiments/conformal_by_severity.py` |
| `recompute_r2_3models.py` | `scripts/legacy/recompute_r2_3models.py` |
| `retrain_weighted_v2.py` | `scripts/legacy/retrain_weighted_v2.py` |
| `retrain_xgb_weighted.py` | `scripts/legacy/retrain_xgb_weighted.py` |
| `run_mobile_test.ps1` | `scripts/tools/run_mobile_test.ps1` |
| `shap_analysis_clean.py` | `scripts/evaluation/shap_analysis.py` |
| `shap_divergence_by_horizon.py` | `scripts/evaluation/shap_divergence_by_horizon.py` |
| `simulate_push_notification.py` | `scripts/tools/simulate_push_notification.py` |
| `summarize_target_latency.py` | `scripts/diagnostics/summarize_target_latency.py` |
| `test_auth_routes.py` | `tests/manual/test_auth_routes.py` |
| `test_best_time_window.py` | `tests/manual/test_best_time_window.py` |
| `test_daily_nlg.py` | `tests/manual/test_daily_nlg.py` |
| `test_daily_summary_structure.py` | `tests/manual/test_daily_summary_structure.py` |
| `test_dynamic_routing.py` | `tests/manual/test_dynamic_routing.py` |
| `test_effective_threshold_by_model.py` | `scripts/legacy/test_effective_threshold_by_model.py` |
| `test_evaluate_live_v2_vs_persistence.py` | `tests/manual/test_evaluate_live_v2_vs_persistence.py` |
| `test_imputation_methods.py` | `tests/manual/test_imputation_methods.py` |
| `test_input_fallback.py` | `tests/manual/test_input_fallback.py` |
| `test_live_features_v2.py` | `tests/manual/test_live_features_v2.py` |
| `test_live_imputation.py` | `tests/manual/test_live_imputation.py` |
| `test_nlg_safety_check.py` | `tests/manual/test_nlg_safety_check.py` |
| `test_notification_history.py` | `tests/manual/test_notification_history.py` |
| `test_outlier_filter.py` | `tests/manual/test_outlier_filter.py` |
| `test_protective_advice.py` | `tests/manual/test_protective_advice.py` |
| `test_risk_alert.py` | `tests/manual/test_risk_alert.py` |
| `test_risk_group_e2e.py` | `tests/manual/test_risk_group_e2e.py` |
| `test_risk_score.py` | `tests/manual/test_risk_score.py` |
| `test_web_push.py` | `tests/manual/test_web_push.py` |
| `train_baseline_lgb_v2.py` | `scripts/legacy/train_baseline_lgb_v2.py` |
| `train_lstm_multiprovince.py` | `scripts/training/train_lstm_hybrid.py` |
| `train_lstm_plain_multiprovince.py` | `scripts/training/train_lstm_plain.py` |
| `train_multiprovince_models.py` | `scripts/training/train_models.py` |
| `train_multiprovince_models_all20.py` | `scripts/experiments/train_models_all20.py` |
| `train_rf_multiprovince.py` | `scripts/training/train_random_forest.py` |
| `train_xgb_prophet_v2.py` | `scripts/legacy/train_xgb_prophet_v2.py` |
| `tune_fire_hotspot_hyperparams.py` | `scripts/experiments/feature_fire_hotspot_window.py` |
| `tune_model_hyperparams_pooled.py` | `scripts/training/tune_hyperparams.py` |
| `tune_quantile_heads_pooled.py` | `scripts/training/tune_quantile_heads.py` |
| `tune_sample_weight_pooled.py` | `scripts/training/tune_sample_weight.py` |
| `tune_spatial_lag_k.py` | `scripts/experiments/feature_spatial_lag_k.py` |
| `validate_amenity_density_feature.py` | `scripts/experiments/feature_amenity_density.py` |
| `validate_drop_neighbor_avg_cv.py` | `scripts/experiments/feature_drop_neighbor_avg_cv.py` |
| `validate_drop_time_cyclical.py` | `scripts/experiments/feature_drop_time_cyclical.py` |
| `validate_elevation_feature.py` | `scripts/experiments/feature_elevation.py` |
| `validate_feature_refinements.py` | `scripts/experiments/feature_candidates.py` |
| `validate_features_clean.py` | `scripts/experiments/feature_set_clean_data.py` |
| `validate_features_timeblocked_cv.py` | `scripts/experiments/feature_set_timeblocked_cv.py` |
| `validate_fire_hotspot_feature.py` | `scripts/experiments/feature_fire_hotspot.py` |
| `validate_fire_wide_features_cv.py` | `scripts/experiments/feature_fire_wide_cv.py` |
| `validate_neighbor_trend_variants.py` | `scripts/experiments/feature_neighbor_trend_variants.py` |
| `validate_spatial_lag_feature.py` | `scripts/experiments/feature_spatial_lag.py` |
| `verify_app_feature_parity.py` | `scripts/diagnostics/app_feature_parity.py` |
| `verify_clean_dataset_alignment.py` | `scripts/diagnostics/dataset_alignment.py` |
| `verify_forecast_all_provinces.py` | `scripts/diagnostics/forecast_all_provinces.py` |

`tests/*.py` (pytest เดิม) → `tests/unit/`

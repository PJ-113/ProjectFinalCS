# Changelog

## 2026-09-14 (100) — เพิ่ม humidity + แยกวิเคราะห์หน้าหนาวใน shap_analysis.py (feedback อาจารย์ข้อ 2, ส่วนสุดท้ายที่เหลือ)

**ที่มา**: ประเด็นย่อยสุดท้ายที่ยังไม่ทำจาก feedback อาจารย์ — "ความชื้น/น้ำค้างในหน้าหนาวมีผลต่อ PM2.5 ไหม" ต่อยอดจาก task E (SHAP plausibility check ที่ทำแค่ precipitation/pressure/wind_speed)

**วิธีทำ**: แก้ `scripts/shap_analysis.py` — เพิ่ม `humidity` เข้า `METEOROLOGICAL_HYPOTHESES` (ตั้งสมมติฐานล่วงหน้า: humidity สูง → ฝุ่นควรเพิ่มขึ้น จาก hygroscopic growth + temperature inversion ตอนเช้าหน้าหนาว) เพิ่ม `WINTER_STRATIFIED_FEATURES = ["humidity"]` + `WINTER_MONTHS = {11,12,1,2}` แล้วแก้ `_meteorological_plausibility_check()` ให้เช็คแยก correlation ระหว่างหน้าหนาว vs นอกหน้าหนาวสำหรับ feature ที่ระบุ (ใช้ `datetime_utc` จาก test split เดิม)

**ผลลัพธ์**: **ขัดกับสมมติฐานเดิมอย่างชัดเจนและสม่ำเสมอ** — correlation(humidity, shap) = **-0.79 ถึง -0.95** (ลบแรงมาก) ทุก horizon (t+1/12/24) ทั้ง 2 โมเดล **และไม่ต่างกันระหว่างหน้าหนาวกับนอกหน้าหนาวเลย** (เช่น LGB t+12: หน้าหนาว -0.948 vs นอกหน้าหนาว -0.955 — แทบเท่ากัน)

**การตีความที่แก้ไขจากสมมติฐานเดิม**: สมมติฐาน hygroscopic growth (ความชื้นทำให้ฝุ่นหนักขึ้น) ไม่ใช่กลไกหลักที่โมเดลเรียนรู้ — คำอธิบายที่สอดคล้องกับผลจริงมากกว่าคือ **humidity เป็นตัวแทน (proxy) ของฤดูแล้ง/ฤดูฝนในภูมิอากาศไทย**: ฤดูแล้ง (ความชื้นต่ำ) ตรงกับฤดูเผาไร่/ฝุ่นสูง (สอดคล้องกับ task A ที่พบ MAE/ฝุ่นสูงช่วง ก.พ.-เม.ย.) ส่วนฤดูฝน (ความชื้นสูง) มีฝนช่วยชะล้างฝุ่น (สอดคล้องกับ precipitation ที่ตรงสมมติฐานอยู่แล้วใน task E) — กลไกนี้แรงกว่าและครอบงำผลของ hygroscopic growth จนไม่เห็นผลต่างระหว่างฤดู **โมเดลยังคงเรียนรู้สิ่งที่สมเหตุสมผลทางภูมิอากาศ เพียงแต่ผ่านกลไกคนละแบบจากที่ตั้งสมมติฐานไว้ตอนแรก** — เป็นตัวอย่างที่ดีของการที่ SHAP ช่วยแก้ไขความเข้าใจผิดเชิงสมมติฐานได้จริง

**สรุป feedback อาจารย์ข้อ 2 (อุตุนิยมวิทยา) ครบทุกส่วนย่อยแล้ว**: precipitation/pressure/wind_speed (task E) + humidity/หน้าหนาว (entry นี้) — ทุกตัวโมเดลเรียนรู้ pattern ที่อธิบายได้ทางวิทยาศาสตร์ ไม่มีตัวไหนดูเหมือน fit มั่วๆ

**ไฟล์ที่แก้/เพิ่ม**: `scripts/shap_analysis.py`, `models/shap_values_by_horizon_20260914_215121.pkl`, `models/shap_top_features_by_horizon_20260914_215121.csv`, `CHANGELOG.md` (entry นี้)

## 2026-09-14 (99) — Validate elevation เป็น model feature: ไม่คุ้มลงทุน (สรุปประเด็นภูมิประเทศ)

**ที่มา**: ต่อจาก entry 98 (elevation ไม่สัมพันธ์กับ PM2.5 volatility) — ทดสอบคนละคำถาม: แม้ elevation จะไม่อธิบาย volatility ตรงๆ มันอาจยังช่วยโมเดลแยกแยะ "ลักษณะพื้นที่" ได้ (คล้าย lat/lon) จึง validate แบบเดียวกับ spatial-lag feature (task F) ก่อนตัดสินใจ

**วิธีทำ**: `scripts/validate_elevation_feature.py` (ใหม่) — เทียบ baseline (19 feature ปัจจุบัน) vs +elevation_m (20 feature) วิธีเดียวกับ `validate_spatial_lag_feature.py` เป๊ะ (P50-only, 5 horizon ตัวแทน, reuse `split_train_calib_test()`)

**ผลลัพธ์**: **delta เล็กน้อยมาก แทบเป็น noise** — LightGBM -0.11% (2.644→2.641), XGBoost -0.20% (2.759→2.754) เทียบกับ spatial-lag feature ที่ validate ได้ -4.1%/-4.3% (entry 94) ต่างกันลิบลับ — **ไม่คุ้มที่จะลงทุน sync feature formula 4 จุด + retrain เต็มรูปแบบ**

**สรุปประเด็นภูมิประเทศทั้งหมด (entry 98-99)**: ตรวจสอบครบทั้ง 2 มุม (ความสัมพันธ์กับ volatility และประโยชน์ต่อความแม่นยำโมเดล) แล้วไม่พบผลที่มีนัยสำคัญทั้งคู่ — สรุปตรงไปตรงมาได้ว่า **ภายในขอบเขต 20 จังหวัดอีสาน (elevation ต่างกันแค่ ~100m) ปัจจัยภูมิประเทศไม่ใช่ตัวขับสำคัญของ PM2.5 pattern หรือความแม่นยำของโมเดล** ต่างจากการเปรียบเทียบข้ามภาค (อีสาน vs เหนือ) ที่อาจารย์ยกตัวอย่างซึ่งมีความต่างของภูมิประเทศมากกว่านี้มาก แต่โปรเจกต์นี้ไม่มีข้อมูลภาคเหนือให้ทดสอบ — นี่คือคำตอบที่สมบูรณ์และซื่อตรงต่อ feedback ข้อนี้ (ไม่ใช่ทุก feature ที่มีเหตุผลทางทฤษฎีดีจะช่วยจริงเสมอ ต้อง validate ด้วยข้อมูลจริงก่อนเชื่อ)

**ไฟล์ที่เพิ่ม**: `scripts/validate_elevation_feature.py`, `models/elevation_feature_validation_20260914_214630.csv`, `CHANGELOG.md` (entry นี้)

## 2026-09-14 (98) — เพิ่ม elevation เข้า registry + วิเคราะห์ความสัมพันธ์กับ PM2.5 pattern (ภูมิประเทศ, feedback อาจารย์ข้อ 1)

**ที่มา**: ตรวจ feedback อาจารย์ละเอียดอีกรอบพบประเด็นย่อยที่ยังไม่ทำ — "พิจารณาสภาพแวดล้อมทางภูมิประเทศ เช่น ภาคอีสานที่เป็นพื้นที่สูงมักมีความรุนแรงของฝุ่นไม่ต่อเนื่อง" ขอบเขตที่ตกลงกับผู้ใช้: โฟกัสภายในอีสาน 20 จังหวัดเอง ไม่เทียบกับภาคเหนือ (ไม่มีข้อมูล)

**วิธีทำ**:
1. `scripts/fetch_province_elevation.py` (ใหม่) — ดึง elevation จาก Open-Meteo Elevation API (ฟรี ไม่ต้อง key, Copernicus DEM 2021, 90m resolution) 1 request เดียวครบ 20 จังหวัด เขียน `elevation_m` กลับเข้า `data/province_station_registry.json` in-place
2. `scripts/analyze_elevation_vs_pm25_pattern.py` (ใหม่) — ทดสอบสมมติฐานอาจารย์ตรงๆ ด้วย 2 metric: pm25_std (ความผันผวนโดยรวม) และ mean|hour-to-hour diff| (ความไม่ต่อเนื่องเชิงเวลา) เทียบ correlation กับ elevation_m ต่อจังหวัด

**ผลลัพธ์**:
- Elevation ของ 20 จังหวัดอีสานอยู่ในช่วงแคบ 123-237m (เฉลี่ย 166.3m) — เพราะเป็นที่ราบสูงโคราชทั้งหมด ไม่ใช่ภูเขาสูงแบบภาคเหนือ (เลย/หนองบัวลำภูสูงสุด, อุบลราชธานีต่ำสุด)
- **ไม่ยืนยันสมมติฐานอาจารย์** — correlation(elevation, pm25_std) = -0.116 (อ่อนมาก ไม่มีนัย), correlation(elevation, hourly_volatility) = -0.323 (**ขัดทิศทาง** — จังหวัดสูงกว่ากลับผันผวนน้อยกว่าเล็กน้อย ไม่ใช่มากกว่าตามที่คาด)
- จังหวัดผันผวนสูงสุด (นครพนม, บึงกาฬ) เป็นจังหวัดริมโขง elevation ปานกลาง-ต่ำ ไม่ใช่จังหวัดที่สูงที่สุด — บ่งชี้ว่าปัจจัยอื่น (เช่น ความใกล้แม่น้ำโขง, หมอกควันข้ามพรมแดนจากลาว) มีผลมากกว่า elevation ในช่วงแคบแบบนี้

**การตีความ**: ไม่ใช่ว่าภูมิประเทศไม่มีผลต่อ PM2.5 เลย — แต่**ในขอบเขตของอีสาน 20 จังหวัดที่ elevation ต่างกันแค่ ~100m** (เทียบกับภาคเหนือที่มีหุบเขา/ภูเขาต่างกันเป็นพันเมตร) ผลของ elevation ไม่ชัดพอจะยืนยันสมมติฐานที่อาจารย์ยกตัวอย่างมา (ซึ่งเปรียบเทียบข้ามภาค ไม่ใช่ภายในอีสาน) — เป็นคำตอบที่ถูกต้องเช่นกันตามหลักวิทยาศาสตร์ (ไม่พบความสัมพันธ์ก็เป็นผลที่ต้องรายงานตรงไปตรงมา) ต่อไปจะ validate ว่า elevation ช่วย MAE ของโมเดล predict ได้ไหม (คนละคำถามกับเรื่อง volatility)

**ไฟล์ที่แก้/เพิ่ม**: `scripts/fetch_province_elevation.py` (ใหม่), `scripts/analyze_elevation_vs_pm25_pattern.py` (ใหม่), `data/province_station_registry.json` (เพิ่ม elevation_m), `models/elevation_vs_pm25_pattern_20260914_214519.csv`, `figures/elevation_vs_pm25_pattern_20260914_214519.png`, `CHANGELOG.md` (entry นี้)

## 2026-09-14 (97) — Retrain รอบสุดท้าย: spatial-lag + ตัด time_cyclical (19 feature, ยังไม่ deploy)

**ที่มา**: ต่อจาก entry 96 (ยืนยันซ้ำแล้วว่าตัด time_cyclical ยังช่วยหลังมี spatial-lag) — retrain เต็มรูปแบบรวมทั้ง 2 การปรับปรุง (เพิ่ม `pm25_neighbor_avg`, ตัด `hour_sin/cos, month_sin/cos, dayofweek` ออก) เป็น 19 feature

**วิธีทำ**: แก้ `FEATURE_COLS` ใน `scripts/train_multiprovince_models.py` เหลือ 19 ตัว (`assert len(FEATURE_COLS) == 19`) แล้วรัน training เต็ม (residual + bagging + quantile P05/P50/P95 + conformal เหมือนเดิมทุกจุด, `DATA_PATH` เดิมจาก entry 95)

**ผลลัพธ์เทียบทั้ง 3 เวอร์ชัน**:

| เวอร์ชัน | feature | LGB MAE(P50) | LGB coverage | XGB MAE(P50) | XGB coverage |
|---|---|---|---|---|---|
| เดิม (deploy อยู่) | 23 | 2.914 | 85.8%→89.4% | 3.008 | 82.0%→88.8% |
| +spatial-lag (entry 95) | 24 | 2.797 (-4.0%) | 85.9%→89.6% | 2.897 (-3.7%) | 81.2%→88.6% |
| **+spatial-lag -time_cyclical (รอบนี้)** | 19 | **2.754 (-5.5%)** | 85.4%→88.8% | **2.858 (-5.0%)** | 81.9%→88.2% |

**ข้อสังเกตที่ต้องรายงานตรงไปตรงมา**: MAE ดีขึ้นสุดในรอบนี้ (-5.5%/-5.0% จากเดิม) แต่ **coverage หลัง conformal กลับลดลงเล็กน้อยเทียบกับเวอร์ชัน spatial-lag อย่างเดียว** (LGB 89.6%→88.8%, XGB 88.6%→88.2%) — ยังต่ำกว่าเป้าหมาย 90% ทั้งคู่เหมือนเดิม (ไม่ใช่เรื่องใหม่ ปัญหานี้มีมาตั้งแต่โมเดลแรกและเป็นสิ่งที่ entry 89 เจอว่าแย่ลงอีกช่วงฝุ่นสูง) แต่ในรอบนี้ห่างจากเป้าเพิ่มขึ้นอีกนิด — เป็น trade-off เล็กน้อยระหว่าง MAE กับ coverage ที่ควรบันทึกไว้ ไม่ใช่ประเด็นที่ทำให้เวอร์ชันนี้แย่กว่าเดิม (ยังดีกว่าเดิมทั้งคู่ในภาพรวม) แต่เป็นข้อมูลสำหรับพิจารณาต่อว่าควรทำ per-severity conformal calibration (ข้อเสนอจาก entry 89) ควบคู่ไปด้วย

**สถานะ**: โมเดลสุดท้ายบันทึกที่ `models/{lgb,xgb}_pm25_multiprovince_residual_bagged_quantile_20260914_211416.pkl` — **`app.py` ยังไม่แก้ ยังใช้โมเดลเดิมให้บริการอยู่** รอผู้ใช้ตัดสินใจเรื่อง deploy (ต้องออกแบบ live-inference fetch ข้อมูลจังหวัดเพื่อนบ้านก่อน)

**ไฟล์ที่แก้/เพิ่ม**: `scripts/train_multiprovince_models.py`, `models/{lgb,xgb}_pm25_multiprovince_residual_bagged_quantile_20260914_211416.pkl`, `models/conformal_calibration_{lgb,xgb}_20260914_211416.csv`, `models/multiprovince_eval_summary_20260914_211416.csv`, `CHANGELOG.md` (entry นี้)

## 2026-09-14 (96) — ตรวจซ้ำ finding "ตัด time_cyclical แล้ว MAE ดีขึ้น" บนโมเดลที่มี spatial-lag แล้ว

**ที่มา**: task D (entry 90) พบว่าตัด time_cyclical ออกช่วย MAE ดีขึ้น แต่ทดสอบตอนยังไม่มี `pm25_neighbor_avg` (task F, entry 94-95) — feature ใหม่อาจเปลี่ยน interaction ทำให้ finding เดิมไม่จริงอีกต่อไป ต้องตรวจซ้ำก่อนตัดสินใจ retrain รอบสุดท้าย

**วิธีทำ**: `scripts/validate_drop_time_cyclical.py` (ใหม่) — วิธีเดียวกับ D (P50-only, 5 horizon ตัวแทน) แต่รันบนชุดข้อมูล+FEATURE_COLS 24-feature ปัจจุบัน (มี `pm25_neighbor_avg` แล้ว) เทียบ baseline (24 feature) vs without_time_cyclical (19 feature)

**ผลลัพธ์**: **ยืนยันซ้ำ ทิศทางเดิม** — LightGBM -1.49% (เดิมตอน D วัดได้ -1.7%), XGBoost -0.69% (เดิม -2.5%, ขนาดเล็กลงแต่ยังเป็นบวกทั้งคู่) → ตัด time_cyclical ออกยังคงช่วยแม้หลังเพิ่ม spatial-lag แล้ว เดินหน้า retrain รอบสุดท้ายด้วย 19 feature (23 เดิม - time_cyclical 5 + spatial-lag 1)

**ไฟล์ที่เพิ่ม**: `scripts/validate_drop_time_cyclical.py`, `models/validate_drop_time_cyclical_20260914_211141.csv`, `CHANGELOG.md` (entry นี้)

## 2026-09-14 (95) — Retrain โมเดลเต็มรูปแบบพร้อม spatial-lag feature (task F, ยังไม่ deploy)

**ที่มา**: task ต่อยอดจาก entry 94 (validate scaled-down แล้วว่า `pm25_neighbor_avg` ช่วย MAE จริง) — ผู้ใช้ยืนยันให้ retrain เต็มรูปแบบ (quantile P05/P50/P95 + conformal calibration ตามสถาปัตยกรรม deploy จริง) แต่**ยังไม่ต้อง deploy** เข้า `app.py`

**วิธีทำ**:
1. แก้ `scripts/build_raw_dataset_multiprovince.py`: เพิ่ม `SPATIAL_FEATURE_COLS = ["pm25_neighbor_avg"]`, ฟังก์ชัน `haversine_km()` + `add_spatial_lag_feature()` (คำนวณหลัง concat ทุกจังหวัดแล้วเท่านั้น เพราะต้องเห็นข้อมูลข้ามจังหวัดพร้อมกัน ต่างจาก feature อื่นที่คำนวณต่อสถานีได้) — `FEATURE_COLS` เป็น 24 ตัวแล้ว (`BASE_FEATURE_COLS` 23 ตัวเดิม + spatial 1 ตัว) เพื่อให้การ rebuild เต็มจาก OpenAQ/Open-Meteo ในอนาคตมี feature นี้ในตัวโดยอัตโนมัติ
2. `scripts/augment_raw_dataset_with_spatial_lag.py` (ใหม่) — เพิ่ม `pm25_neighbor_avg` เข้าไฟล์ raw ที่มีอยู่แล้วโดยไม่ fetch ใหม่ (feature คำนวณได้จาก pm25 ที่มีอยู่แล้วล้วนๆ) reuse `add_spatial_lag_feature()` จากข้อ 1 → ได้ `data/multiprovince_pm25_features_raw_spatial_20260914_203924.csv` (363,000 แถว, ตัดทิ้ง 332 แถวที่ไม่มีเพื่อนบ้านให้ข้อมูล ณ เวลานั้น)
3. แก้ `scripts/train_multiprovince_models.py`: `FEATURE_COLS` เพิ่ม `pm25_neighbor_avg` (23→24), `DATA_PATH` ชี้ไปไฟล์ augmented ใหม่ แล้วรัน training เต็ม (residual + bagging + quantile P05/P50/P95 + conformal, เหมือนเดิมทุกจุด)

**ผลลัพธ์**: **สอดคล้องกับผล validate ก่อนหน้าเป๊ะ** (ยืนยันว่า scaled-down validation เชื่อถือได้จริง):
- LightGBM: MAE(P50) **2.797** (เดิม 2.914, -4.0%), coverage 85.9%→89.6% (ใกล้เคียงเดิม)
- XGBoost: MAE(P50) **2.897** (เดิม 3.008, -3.7%), coverage 81.2%→88.6% (ใกล้เคียงเดิม)

**สถานะ**: โมเดลใหม่บันทึกแล้วที่ `models/{lgb,xgb}_pm25_multiprovince_residual_bagged_quantile_20260914_204009.pkl` — **`app.py` ยังไม่ได้แก้ ยังชี้ไปโมเดลเดิม (`_20260913_014901`) ตามที่ผู้ใช้ระบุ** ถ้าจะ deploy จริงต้องตัดสินใจเพิ่มเติมเรื่อง live inference (`build_feature_row()` ต้องรู้ pm25 ปัจจุบันของ 3 จังหวัดเพื่อนบ้านด้วย — ยังไม่ได้ออกแบบว่าจะ fetch/cache ยังไง)

**ไฟล์ที่แก้/เพิ่ม**: `scripts/build_raw_dataset_multiprovince.py`, `scripts/augment_raw_dataset_with_spatial_lag.py` (ใหม่), `scripts/train_multiprovince_models.py`, `data/multiprovince_pm25_features_raw_spatial_20260914_203924.csv`, `models/{lgb,xgb}_pm25_multiprovince_residual_bagged_quantile_20260914_204009.pkl`, `models/conformal_calibration_{lgb,xgb}_20260914_204009.csv`, `models/multiprovince_eval_summary_20260914_204009.csv`, `CHANGELOG.md` (entry นี้)

## 2026-09-14 (94) — Validate spatial-lag feature ก่อนตัดสินใจทำ task F เต็มรูปแบบ

**ที่มา**: task B ยืนยัน spatial correlation แข็งแรง (r=-0.644) แต่ task C ไม่ยืนยันในระดับโมเดลจริง (n=3 จุดน้อยเกินไป) — แทนที่จะลงทุนแก้ feature formula ที่ซ้ำกัน 4 จุด (`app.py`, `build_raw_dataset_multiprovince.py`, `backtest_recent_multiprovince.py`, `train_multiprovince_models.py`) + retrain เต็มระบบไปเลยโดยไม่รู้ผล ทำ scaled-down validation ก่อน (เพิ่ม feature เดียว วัดผลแบบ ablation)

**วิธีทำ**: `scripts/validate_spatial_lag_feature.py` (ใหม่) — สร้าง feature `pm25_neighbor_avg` = ค่าเฉลี่ย pm25 ปัจจุบันของ 3 จังหวัดที่ใกล้ที่สุด (haversine, จากทั้ง 20 จังหวัดรวม zero-shot เพราะ production จริงมีข้อมูลนี้ให้ใช้อยู่แล้วผ่าน `scheduled_forecast_job()`) ณ เวลาเดียวกับแถวนั้น (contemporaneous ไม่ leak อนาคต) เทียบ MAE แบบ baseline (23 feature) vs with_spatial_lag (24 feature) บน train/test split เดียวกับ task A/D (P50-only, representative horizons)

**ผลลัพธ์**: **MAE ดีขึ้นชัดเจนทั้ง 2 โมเดล** — LightGBM -4.26% (2.804→2.684), XGBoost -4.10% (2.897→2.778) และผลดีขึ้นมากกว่าที่ horizon ไกล (t+24: LGB 5.16→4.89 คือ -5.2%, XGB 5.43→5.13 คือ -5.6%) — เกือบเท่ากับผลของทั้งกลุ่ม weather รวมกัน (จาก task D, +1.3-1.7%) ด้วย feature เดียว

**สรุป**: **คุ้มที่จะลงทุนทำ F เต็มรูปแบบ** — มีหลักฐานเชิงประจักษ์ชัดเจนรองรับแล้ว (ต่างจากตอนเริ่ม B/C ที่เป็นแค่สมมติฐาน) ขั้นต่อไปคือแก้ feature formula 4 จุด (เพิ่ม `pm25_neighbor_avg` ให้สอดคล้องกัน) + ออกแบบว่า `app.py`'s live inference จะ fetch ข้อมูลจังหวัดเพื่อนบ้านยังไง (ยังไม่ได้ทำ ต้องคุยกับผู้ใช้ก่อนเพราะกระทบโค้ด production จริง ไม่ใช่แค่ script วิจัย) + retrain โมเดล deploy ใหม่ทั้งชุด

**ไฟล์ที่เพิ่ม**: `scripts/validate_spatial_lag_feature.py`, `models/spatial_lag_validation_20260914_203234.csv`, `CHANGELOG.md` (entry นี้)

## 2026-09-14 (93) — Zero-shot MAE เทียบระยะทางจังหวัดเทรนที่ใกล้ที่สุด (task C, งานสุดท้ายก่อนตัดสินใจ task F)

**ที่มา**: ต่อยอดจาก task B (correlation -0.644 ยืนยันว่าจังหวัดใกล้กัน PM2.5 สัมพันธ์กันสูงกว่า) — ทดสอบว่า pattern นี้สะท้อนออกมาใน zero-shot generalization ของโมเดลจริงไหม (จังหวัดที่ไม่เคยเทรน แต่อยู่ใกล้จังหวัดที่เทรน ควรทายแม่นกว่าจังหวัดที่อยู่ไกล)

**วิธีทำ**: `scripts/eval_zeroshot_spatial_distance.py` (ใหม่) — reuse `eval_zeroshot_multiprovince.py` ทั้งชุด (FEATURE_COLS, HELD_OUT, split_train_test, fit_p50) ผ่าน importlib เทรนโมเดล P50 บน 14 จังหวัด zero-shot pool ครบ 24 horizon แล้ว evaluate แยกต่อจังหวัด held-out ทั้ง 3 (อุดรธานี/อุบลราชธานี/ชัยภูมิ — ต่างชุดกับ 3 จังหวัด zero-shot จริง) แทนที่จะรวมเป็นก้อนเดียวแบบสคริปต์ต้นฉบับ + save per-horizon per-province ผลลง CSV จริง (ของเดิมคำนวณแล้วทิ้ง) + หาระยะทางไปจังหวัดเทรนที่ใกล้ที่สุดต่อจังหวัด held-out (reuse `haversine_km()`)

**ผลลัพธ์**: **ไม่ยืนยันสมมติฐาน** — อุดรธานี (ใกล้สุด 42km) MAE=2.89, อุบลราชธานี (กลาง 58km) MAE=3.93 (สูงสุด!), ชัยภูมิ (ไกลสุด 93km) MAE=2.75 (ต่ำสุด) — correlation(distance, mae) = -0.31 (ทิศทางขัดสมมติฐานเล็กน้อย) ทั้ง LightGBM และ XGBoost ให้ผลทิศทางเดียวกัน

**การตีความ**: ตามที่ระบุ limitation ไว้ล่วงหน้าในสคริปต์แล้ว — **n=3 จุดข้อมูลน้อยเกินไปสำหรับสรุปทางสถิติ** ผลนี้น่าจะเป็น noise จากปัจจัยอื่นที่ไม่ใช่ระยะทางล้วน (เช่น คุณภาพ/ความสมบูรณ์ของข้อมูลเซนเซอร์แต่ละจังหวัด, ลักษณะภูมิประเทศเฉพาะจุด) มากกว่าที่จะหักล้าง task B — **สรุปโดยรวมของ 2 งานนี้**: หลักฐานเชิงสถิติที่แข็งแรงกว่า (task B, n=190 คู่, r=-0.644) ยืนยันว่า spatial correlation มีอยู่จริงในข้อมูลดิบ ส่วน task C (n=3, ไม่มีนัยสำคัญ) แค่ไม่สามารถยืนยัน/ปฏิเสธได้ว่าโมเดลปัจจุบันใช้ประโยชน์จาก pattern นั้นแล้วหรือยัง — **แนะนำเดินหน้า task F (spatial-lag feature) โดยอิงหลักฐานหลักจาก B** แต่ต้องระวังว่าการปรับปรุงอาจไม่ได้สะท้อนออกมาใน metric ระดับจังหวัดเดี่ยวๆ ชัดเจนแบบ 1-1 เพราะปัจจัยกวนอื่นเยอะ

**ไฟล์ที่เพิ่ม**: `scripts/eval_zeroshot_spatial_distance.py`, `models/zeroshot_spatial_distance_per_horizon_20260914_201822.csv`, `models/zeroshot_spatial_distance_summary_20260914_201822.csv`, `CHANGELOG.md` (entry นี้)

## 2026-09-14 (92) — Spatial correlation heatmap ระหว่างจังหวัด (task B ตาม feedback อาจารย์ ข้อ 1)

**ที่มา**: อาจารย์เตือนว่า lat/lon เฉยๆ ไม่พอ ต้องวิเคราะห์ spatial correlation ระหว่างจุดวัดต่างๆ ไม่ใช่มองจุดเดียว — ทำก่อนตัดสินใจว่าจะ engineer spatial feature เพิ่ม (task F) คุ้มไหม

**วิธีทำ**: `scripts/analyze_spatial_correlation.py` (ใหม่) — pivot `data/multiprovince_pm25_features_raw_20260913_012629.csv` เป็น matrix (datetime_utc × province_en) หา Pearson correlation แบบ pairwise (min_periods=100 ชม. overlap) ทั้ง 20 จังหวัด (190 คู่) คำนวณระยะทาง haversine ระหว่างทุกคู่ (reuse `haversine_km()` จาก `province_registry.py`) เทียบ correlation กับระยะทาง

**สมมติฐาน**: correlation ลดลงตามระยะทาง (Tobler's First Law of Geography)

**ผลลัพธ์**: **ยืนยันสมมติฐานชัดเจนมาก** — Pearson correlation(distance_km, pm25_correlation) = **-0.644** (ครบทั้ง 190/190 คู่จังหวัดมีข้อมูลพอ) คู่จังหวัดใกล้กันสุด (กาฬสินธุ์-มหาสารคาม 37km) correlation=0.94, คู่ไกลสุด (บึงกาฬ-นครราชสีมา 411km) correlation=0.46 — ความสัมพันธ์เชิงพื้นที่ของ PM2.5 มีอยู่จริงและแรงพอสมควร **เป็นหลักฐานสนับสนุนว่าควรทำ task F (spatial-lag feature) ต่อ** — ไม่ใช่แค่ lat/lon ดิบที่ใช้อยู่ตอนนี้

**ไฟล์ที่เพิ่ม**: `scripts/analyze_spatial_correlation.py`, `models/spatial_correlation_pairs_20260914_201542.csv`, `figures/spatial_correlation_heatmap_20260914_201542.png`, `figures/spatial_correlation_vs_distance_20260914_201542.png`, `CHANGELOG.md` (entry นี้)

## 2026-09-14 (91) — แก้ shap_analysis.py ให้ใช้กับโมเดล pooled ได้ + เช็คความสมเหตุสมผลเชิงอุตุนิยมวิทยา (task E)

**ที่มา**: อาจารย์ให้ feedback ว่าต้องเข้าใจหลักอุตุนิยมวิทยาเพื่อตรวจสอบว่าโมเดลเรียนรู้สมเหตุสมผลจริง — ทำต่อจาก task D (ดู entry 90) `scripts/shap_analysis.py` เดิมผูกกับโมเดล single-station เก่า (path เก่า + structure `models[h]`=model เดียว) ใช้กับโมเดล pooled ไม่ได้เลย (`models[h]` ของ pooled จริงคือ dict ของ 3 quantile ไม่ใช่ model เดียว — ยืนยันจาก `train_multiprovince_models.py`)

**วิธีทำ**: แก้ `scripts/shap_analysis.py` ทั้งไฟล์ (ไม่ใช่แค่สลับ path) — (1) เปลี่ยน DATA_PATH/LGB_MODEL_PATH/XGB_MODEL_PATH เป็นไฟล์ pooled ปัจจุบัน (2) แก้ indexing เป็น `models[h][0.5]` (อธิบายเฉพาะ P50) (3) เปลี่ยน MODEL_PACKAGES key เป็น `lightgbm_pooled`/`xgboost_pooled` ตรงกับชื่อจริงใน Dynamic Routing, ตัด branch "prophet" ทิ้ง (สถาปัตยกรรมเก่าที่ retired ไปแล้ว) แทนด้วย "naive_fallback" (คืน None เหมือนเดิม เพราะไม่มี quantile model ให้อธิบาย) (4) ระบุชัดเจนในทุก output ว่า SHAP อธิบาย residual (pm25_t+h - pm25_now) ไม่ใช่ค่าสัมบูรณ์ (5) reuse `split_train_calib_test()` จาก `train_multiprovince_models.py` ให้ได้ test split ชุดเดียวกับ task A/D (6) เพิ่มฟังก์ชันใหม่ `_meteorological_plausibility_check()` — ตั้งสมมติฐานทิศทางไว้ก่อน (precipitation → ฝุ่นลด/scavenging, pressure → ฝุ่นเพิ่ม/inversion trapping, wind_speed → ฝุ่นลด/dispersion) แล้วเช็คด้วย correlation ระหว่างค่า feature กับ SHAP value ของมันจริง

**ผลลัพธ์**: **ทุกสมมติฐาน (3 feature × 3 horizon ตัวแทน × 2 โมเดล = 18 จุดตรวจ) ตรงทิศทางที่คาดหมดทุกจุด**:
- `pressure`: correlation +0.23 ถึง +0.90 (แรงสุดที่ LightGBM, horizon ไกล) — ยืนยัน temperature inversion trapping
- `wind_speed`: correlation -0.74 ถึง -0.79 — ยืนยัน dispersion effect ชัดเจนมาก สม่ำเสมอทุก horizon/โมเดล
- `precipitation`: correlation -0.28 ถึง -0.51 — ยืนยัน scavenging effect (อ่อนกว่า 2 ตัวข้างต้นแต่ทิศทางถูกทุกจุด)

โมเดลเรียนรู้ความสัมพันธ์เชิงฟิสิกส์ได้สมเหตุสมผล ไม่ใช่แค่ fit ตัวเลขมั่วๆ — ตอบ feedback อาจารย์ข้อ 2 ได้ตรงจุด

**ข้อสังเกตเพิ่มเติมที่เชื่อมกับ task D**: ที่ t+24 ของ LightGBM, `month_sin`/`temp`/`humidity` ขึ้น top-5 feature สำคัญ (SHAP magnitude สูง) แต่ ablation study (entry 90) กลับพบว่าตัด `time_cyclical` ทั้งกลุ่มออกทำให้ MAE ดีขึ้น — ไม่ขัดแย้งกันเสียทีเดียว (SHAP วัด "มีอิทธิพลต่อ prediction จริง" ส่วน ablation วัด "ช่วย generalize บน test set จริงไหม" คนละมิติ) แต่ชี้ว่า `month_sin` อาจถูกใช้ไปในทาง overfit ต่อ pattern เฉพาะช่วงเทรน มากกว่าช่วยจริง — ควรตรวจเพิ่มถ้ามีเวลา (ไม่ใช่ขอบเขต task นี้)

**ไฟล์ที่แก้/เพิ่ม**: `scripts/shap_analysis.py` (แก้ทั้งไฟล์), `models/shap_values_by_horizon_20260914_201025.pkl`, `models/shap_top_features_by_horizon_20260914_201025.csv`, `CHANGELOG.md` (entry นี้)

## 2026-09-14 (90) — Ablation study ทีละ feature group (task D ตาม feedback อาจารย์ ข้อ 4)

**ที่มา**: อาจารย์ให้ feedback ว่าต้องออกแบบการทดลองแบบตั้งสมมติฐานก่อน แล้วทดสอบทีละตัวแปร ไม่ใช่โยนทุก feature เข้าโมเดลพร้อมกันโดยไม่เคยแยกวัดผล — ทำต่อจาก task A (ดู entry 89)

**วิธีทำ**: `scripts/ablation_feature_groups.py` (ใหม่) — ตั้งสมมติฐานล่วงหน้าต่อ 5 feature group (pm25_lag_rolling 9 ตัว, weather 4 ตัว, wind_pressure 3 ตัว, time_cyclical 5 ตัว, lat_lon 2 ตัว) แล้ว leave-one-group-out เทียบ baseline (feature ครบ 23 ตัว) บน train/test split เดียวกับ task A (P50-only, reuse `split_train_calib_test()`) วัดที่ horizon ตัวแทน 5 จุด (t+1/6/12/18/24) ไม่ครบ 24 horizon เพื่อความเร็ว (60 model fits รวม)

**ผลลัพธ์** (delta MAE จาก baseline, LGB baseline=2.806, XGB baseline=2.900):
- `pm25_lag_rolling`: ตัดออก MAE แย่ลง **+9.1% (LGB) / +8.0% (XGB)** — ยืนยันสมมติฐาน เป็นกลุ่มสำคัญที่สุดชัดเจน
- `weather`: +1.3% / +1.7% — ช่วยจริงแต่เล็กกว่าที่คาด (ไม่ใช่ "พอสมควร" ตามสมมติฐาน)
- `wind_pressure`: +0.5% / +0.1% — ผลเล็กมาก ตรงสมมติฐานว่าเล็กกว่า weather/lag
- `lat_lon`: +1.0% / -0.3% — ผลต่อ MAE ใน-training แทบไม่มี (ผลจริงของ lat/lon คือ zero-shot generalization ไม่ใช่ MAE ใน-training ตรงตามสมมติฐาน)
- **`time_cyclical`: -1.7% (LGB) / -2.5% (XGB) — ขัดกับสมมติฐาน!** ตัด hour/month/dayofweek ออกกลับทำให้ MAE **ดีขึ้น** ทั้ง 2 โมเดล ที่ representative horizons ที่ทดสอบ — เป็นไปได้ว่า feature กลุ่มนี้ไม่ได้ช่วยจับ pattern เพิ่มเติมนอกเหนือจากที่ `pm25_lag/rolling` (ซึ่งมี autocorrelation แฝง diurnal/seasonal pattern อยู่แล้วในตัว) ให้ไปแล้ว กลับเพิ่ม noise/overfitting เล็กน้อยแทน — ควรตรวจสอบเพิ่มเติม (เช่น ลองตัดเฉพาะบางตัวในกลุ่มนี้ ไม่ใช่ตัดทั้งกลุ่ม) ก่อนตัดสินใจเอาออกจากโมเดล deploy จริง

**ไฟล์ที่เพิ่ม**: `scripts/ablation_feature_groups.py`, `models/ablation_feature_groups_20260914_200754.csv`, `CHANGELOG.md` (entry นี้)

## 2026-09-14 (89) — Backtest ช่วงฤดูฝุ่น (ก.พ.-เม.ย.) เทียบกับช่วงปกติ (task A ตาม feedback อาจารย์)

**ที่มา**: อาจารย์ที่ปรึกษาให้ feedback ว่า evaluation ต้อง backtest แบบ time-series โดยเฉพาะช่วงฝุ่นสูง (ยกตัวอย่าง 1-2 เม.ย.) ไม่ใช่ดูแค่ MAE เฉลี่ยรวม — วางแผนงานร่วมกับผู้ใช้เป็น 6 งาน (A-F, ดู plan file) เริ่มจาก A ก่อนเพราะใช้ test split ที่มีอยู่แล้วได้เลย ไม่ต้องดึงข้อมูลใหม่

**วิธีทำ**: `scripts/analyze_burning_season_backtest.py` (ใหม่) — reuse `split_train_calib_test()` จาก `train_multiprovince_models.py` ตรงๆ ให้ได้ test split ชุดเดียวกับที่รายงาน MAE(P50)/coverage อย่างเป็นทางการใน CHANGELOG (75) (กันอคติเลือกช่วงทดสอบเอง) predict ทุก horizon บน test rows (64,686 แถว, ช่วง 2026-01-20 ถึง 2026-07-02) ใช้ q_hat ที่บันทึกไว้ใน `eval_by_horizon` ของแต่ละโมเดลอยู่แล้ว (ไม่ต้อง reconstruct calibration split ใหม่) แท็ก `is_burning_season` = เดือน ก.พ.-เม.ย. + breakdown เพิ่มตาม severity percentile ของ PM2.5 จริง

**ผลลัพธ์**:
- MAE ช่วงฤดูฝุ่นสูงกว่าช่วงปกติชัดเจน — LightGBM สูงกว่า **87.2%** (1.96→3.66), XGBoost สูงกว่า **89.2%** (2.00→3.79) ยืนยันสมมติฐาน (เหตุการณ์ฝุ่นหนักมีสัดส่วนน้อยในข้อมูลเทรน)
- ที่ horizon ไกล (t+24) ต่างกันหนักกว่าอีก — ปกติ MAE~3.4-3.5, ฤดูฝุ่น MAE~6.6-6.9 (เกือบเท่าตัว)
- **finding ที่ไม่ได้คาดไว้ในสมมติฐานเดิม**: coverage ที่ severity bin สูงสุด (>p90 ของ PM2.5 จริง) ต่ำกว่าเป้าหมาย 90% ชัดเจน — LightGBM 85.5%, XGBoost 81.9% (เทียบกับ bin อื่นที่ยังใกล้ 90%) แปลว่าช่วงความเชื่อมั่น (conformal interval) **under-cover พอดีตอนฝุ่นวิกฤตที่สุด** ซึ่งเป็นช่วงที่ผู้ใช้ต้องการความแม่นยำของ uncertainty มากที่สุด — สอดคล้อง/เสริมข้อสังเกตเดิมใน CHANGELOG (76) ที่พบว่า conformal q_hat รวมทุกจังหวัดเป็นค่าเดียวทำให้ coverage เบี่ยงต่อจังหวัด ตอนนี้พบเพิ่มว่าเบี่ยงตาม**ความรุนแรงของเหตุการณ์**ด้วย — เป็นหลักฐานสนับสนุนแนวทางปรับปรุง conformal calibration แบบ stratified (ต่อจังหวัด และ/หรือต่อช่วงความรุนแรง) ในอนาคต

**ไฟล์ที่เพิ่ม**: `scripts/analyze_burning_season_backtest.py`, `CHANGELOG.md` (entry นี้)

## 2026-09-13 (88) — แก้ dropdown เลือกจังหวัดอ่านไม่ออกตอนเปิด (popup โปร่งแสงทะลุพื้นหลังหน้าเว็บ)

**ที่มา**: ผู้ใช้ส่งสกรีนช็อตมาว่าเปิด dropdown เลือกจังหวัดแล้วอ่านตัวเลือกไม่ออก (ตัวอักษรจางมากจนแทบมองไม่เห็น ยกเว้นตัวที่เลือกอยู่) ตรวจ CSS พบว่า `.province-select` ตั้งใจให้ `background: transparent` เพื่อให้ตัว select เข้ากับทรงพิลกระจกของ header ตอนปิดอยู่ — แต่ native `<select>` popup (ตัว listbox ที่เด้งขึ้นมาตอนคลิก) ไม่รับค่า transparent นี้แบบตรงไปตรงมา กลายเป็นโชว์ gradient สีฟ้าของพื้นหลังหน้าเว็บทะลุเข้ามาแทน ทำให้ตัวอักษร option (สีเข้มจาก `--auth-text` ที่ออกแบบไว้สำหรับพื้นขาว) ไปชนกับพื้นหลังสีฟ้าแทน อ่านไม่ออก

**สิ่งที่ทำ**: เพิ่ม CSS rule `.province-select option` ให้สีพื้นหลัง+ตัวอักษรทึบตรงๆ แยกจาก `.province-select` เอง (ขาว/เข้มสลับตาม light/dark mode) — `<option>` ส่วนใหญ่ยอมรับ background/color ตรงๆ แม้ browser จะไม่ยอมให้ปรับแต่ง popup listbox ทั้งกล่องแบบเต็มที่ก็ตาม ไม่แตะ `.province-select` เอง (ทรงพิลกระจกตอนปิดยังสวยเหมือนเดิม)

**ทดสอบ**: `npm run build` ผ่าน, สกรีนช็อต Playwright คลิกเปิด dropdown จริงยืนยันตัวเลือกทุกจังหวัดอ่านออกชัดเจน (พื้นขาวทึบ ตัวอักษรเข้ม) ทั้งจังหวัดปกติและ zero-shot

**ไฟล์ที่แก้**: `frontend/src/components/AppHeader.vue`, `CHANGELOG.md` (entry นี้)

## 2026-09-13 (87) — ย่อ comment ใน app.py ให้กระชับขึ้น

**ที่มา**: ผู้ใช้บอกว่า comment ใน app.py รกไป ขอให้จัดการให้ดูสะอาดขึ้น ถามกลับว่าจะเอาแบบไหน (ย่อสั้นลง / ย้ายไปไฟล์แยก / ตัดเกือบหมด) ผู้ใช้เลือก "ย่อให้สั้นลง เก็บแค่ WHY สำคัญ" เพราะ comment ยาวๆ จำนวนมากซ้ำกับ CHANGELOG.md อยู่แล้ว (ประวัติ/เหตุผลละเอียดยังหาได้จากที่นั่น ไม่ต้องเล่าซ้ำในโค้ด)

**สิ่งที่ทำ**: ไล่ทั้งไฟล์ย่อ docstring/comment ที่ยาวเกิน 1-2 บรรทัดลงเกือบทั้งหมด (~60 จุด) ตัดส่วนที่เป็นประวัติศาสตร์/เล่าที่มาซ้ำ CHANGELOG ออก เหลือแค่ WHY ที่จำเป็นต่อการอ่านโค้ดจริง — comment ที่อธิบาย algorithm ที่ไม่ชัดเจนในตัวเอง (เช่น `_find_keyword_occurrences`/`_mentioned_levels` ของ NLG safety check, Dynamic Routing's None-handling ทั้ง 2 ทิศทาง) ยังคงรายละเอียดไว้พอสมควร ไม่ตัดทิ้งจนขาดบริบทที่จำเป็น

**ทดสอบ**: `python -c "import ast; ast.parse(...)"` ผ่าน (ไม่มี syntax error จากการแก้), `scripts/test_dynamic_routing.py` รันซ้ำหลังแก้ **4/4 PASS เหมือนเดิมทุกเคส** (ยืนยันว่าย่อ comment ไม่ได้แตะ logic เลย), รีสตาร์ท `app.py` สะอาดๆ แล้วยิง `/api/forecast`/`/api/comparison/overview` ยืนยันคืนค่าถูกต้อง (HTTP 200) ทั้งคู่

**ผล**: ไฟล์ลดจาก 1889 บรรทัด เหลือ 1720 บรรทัด (~170 บรรทัด, ตัด comment ไปมากกว่านั้นเพราะมีการเพิ่ม logic เล็กน้อยระหว่างทางด้วย)

**ไฟล์ที่แก้**: `app.py`, `CHANGELOG.md` (entry นี้)

## 2026-09-13 (86) — ลด Dynamic Routing จาก 48 DB connection/request เหลือ 1 (แก้จังหวัดโหลดช้า)

**ที่มา**: ผู้ใช้ถามว่าทำไมสลับจังหวัดโหลดข้อมูลนาน วัดเวลาจริงพบว่า `/api/forecast` ใช้เวลารวม ~6 วิ แยกได้เป็น OpenAQ fetch ~1.7 วิ, Open-Meteo ~0.8 วิ, และ **`select_model()` ที่ถูกเรียก 24 รอบ (1 ต่อ horizon) ใช้เวลารวม ~2.3 วิ** ต้นเหตุคือแต่ละรอบเรียก `compute_rolling_r2()` 2 ครั้ง (เช็ค R² ของ XGBoost/LightGBM แยกกัน) และ `get_db_connection()` เปิด connection Postgres ใหม่ทุกครั้งไม่มี pooling เลย — รวมแล้วเปิด-ปิด connection ใหม่ **48 ครั้งต่อการพยากรณ์ 1 จังหวัด 1 ครั้ง**

**สิ่งที่ทำ**:
- เพิ่ม `fetch_rolling_r2_lookup(province, N=24)` — query เดียว ใช้ window function (`ROW_NUMBER() OVER (PARTITION BY model_used, horizon ORDER BY target_time_utc DESC)`) หา N แถวล่าสุดต่อ (model, horizon) ทุกคู่พร้อมกัน แทนที่จะ query แยกทีละคู่ คำนวณ R² ในหน่วยความจำ (Python) จาก resultset เดียว คืน dict `{(model_used, horizon): r2 หรือ None}`
- `select_model(feature_row, horizon, province, r2_lookup)` — เพิ่ม param `r2_lookup` (บังคับ ไม่มี default) อ่านจาก dict ที่คำนวณไว้ล่วงหน้าแทนการเรียก `compute_rolling_r2()` เอง (ตัด DB call ออกจากฟังก์ชันนี้ทั้งหมด)
- `get_forecast()` เรียก `fetch_rolling_r2_lookup(province)` แค่ครั้งเดียวก่อนเริ่ม loop 24 horizon แล้วส่ง `r2_lookup` เดียวกันเข้า `select_model()` ทุกรอบ
- `compute_rolling_r2()` (query แยกทีละคู่) **ยังเก็บไว้** ไม่ลบทิ้ง — ยังมีประโยชน์สำหรับ diagnostics/cross-check เดี่ยวๆ แค่ไม่ได้อยู่ใน hot path ของ `select_model()` อีกต่อไป

**ทดสอบ**: อัปเดต `scripts/test_dynamic_routing.py` ให้ตรงกับสถาปัตยกรรม pooled model ปัจจุบัน (เดิม broken อยู่แล้ว อ้างอิง "prophet"/ไม่มี province param ใช้กับ app.py ตอนนี้ไม่ได้เลย) เพิ่ม 2 เคสใหม่ (cold-start ทั้งคู่ไม่มีข้อมูล, cold-start มีแค่ฝั่งเดียว — regression guard สำหรับบั๊กที่เคยแก้ใน CHANGELOG (20)) รันเป็น **baseline ก่อนแก้** (4/4 PASS ด้วย logic เดิม) แล้วรันซ้ำ **หลังแก้** (4/4 PASS เหมือนเดิมทุกเคส รวม cross-check ว่า `fetch_rolling_r2_lookup()` ให้ค่า R² ตรงกับ `compute_rolling_r2()` แบบ query เดี่ยวเป๊ะ) ยืนยันว่า refactor ไม่เปลี่ยนพฤติกรรม Dynamic Routing เลย

วัดเวลาจริงหลังแก้: `fetch_rolling_r2_lookup()` (1 query) ใช้ ~0.03 วิ, `select_model()` x24 (อ่านจาก lookup อย่างเดียว ไม่ query) ใช้ ~0.01 วิ (จากเดิม 2.3 วิ) `/api/forecast` รวมทั้ง request ลดจาก ~6 วิ เหลือ **~2-3 วิ** (ส่วนที่เหลือเป็น OpenAQ/Open-Meteo fetch ซึ่งเป็น external API ควบคุมเวลาไม่ได้)

**ไฟล์ที่แก้**: `app.py`, `scripts/test_dynamic_routing.py`, `CHANGELOG.md` (entry นี้)

## 2026-09-13 (85) — ลบข้อมูลปนเปื้อนจาก process app.py ที่ค้างซ้อนกันหลายตัว

**ที่มา**: ผู้ใช้สังเกตว่าค่าพยากรณ์ขอนแก่นคลาดเคลื่อนมาก ถามว่าเกี่ยวกับฝนไหม — ตรวจ `prediction_log` จริงพบว่า MAE ที่ดูแย่ (~5.7) มาจากข้อมูลปนเปื้อน 2 แถวที่ `model_used='xgboost'` (ชื่อรุ่นเก่า ไม่มี `_pooled`) ทายผิดมหาศาล (9.6 เทียบค่าจริง 26.2) ไม่เกี่ยวกับฝนเลย ตรวจ process เพิ่มเติมพบว่ามี **`python app.py` ค้างอยู่พร้อมกันถึง 4 โปรเซส** (เศษจากตอนเซิร์ฟเวอร์ crash/รีสตาร์ทซ้ำๆ ระหว่างดีบัก — ดู entry ก่อนหน้าเรื่อง `WinError 10038`) โปรเซสเก่าที่ค้างยังใช้โค้ด/ชื่อโมเดลก่อนเปลี่ยนมาใช้ pooled model log ข้อมูลปนเข้ากับโปรเซสใหม่ที่ถูกต้อง

**สิ่งที่ทำ**: ปิดโปรเซส `app.py` ที่ค้างอยู่ทั้ง 4 ตัว (`taskkill /T`) เหลือรันแค่ชุดเดียว (parent+reloader child ตามปกติ) ตรวจสอบด้วย SQL ว่าทุกแถวที่มี `model_used IN ('xgboost','lightgbm','prophet')` (24 แถวทั้งหมด) เป็นข้อมูลปนเปื้อนจาก `predicted_at_utc='2026-09-13T07:00:00+00:00'` ของขอนแก่นล้วนๆ (ไม่ใช่ข้อมูลประวัติศาสตร์จริง เพราะเครื่องนี้เพิ่งตั้งใหม่ ไม่มีข้อมูลเก่าจากยุค Air4Thai ตกค้างอยู่แล้ว) ลบทิ้งทั้งหมด

**ทดสอบ**: หลังลบ+เริ่ม process ใหม่สะอาดๆ `/api/comparison/overview?province=khonkaen` MAE ลดจาก 5.72 เหลือ **0.32** (n_matched เหลือแค่แถว `xgboost_pooled` ที่ถูกต้อง) ยืนยันว่า pooled model ที่ขอนแก่นแม่นยำมากอยู่แล้ว ปัญหาเป็นเรื่อง data hygiene ไม่ใช่โมเดล

**ไฟล์ที่แก้**: ไม่มีไฟล์โค้ด (แก้ที่ระดับ process/DB เท่านั้น), `CHANGELOG.md` (entry นี้)

## 2026-09-13 (84) — ปรับ design ตัวแสดงช่วงความเชื่อมั่นบนกราฟพยากรณ์อีกรอบ (เลิกใช้ dot+line)

**ที่มา**: entry 80 เคยปรับจาก tick-cap ไปเป็นเส้นบาง+จุดกลมบน-ล่าง (dot+line) ไปแล้วรอบนึง ผู้ใช้ดูของจริงบนเครื่องแล้วบอกว่ายังไม่สวย สกรีนช็อตของจริงมาดูใกล้ๆ พบสาเหตุ: เส้น+จุดลอยพ้นแท่งขึ้นไปสูงในพื้นที่ว่างด้านบน หน้าตาเหมือน "หนวดแมลง"/ลูกโป่งปักอยู่บนแท่งสี ไม่ใช่องค์ประกอบเดียวกับแท่งเลย เป็นการซ้อน mark คนละชนิด (line+dot บนแท่ง) ที่ดูแปลกแยกจากกัน

**สิ่งที่ทำ**: เปลี่ยนแนวทางใหม่ทั้งหมด — เลิกใช้ dot/line ไปเลย ใช้ **ส่วนขยายโปร่งแสงของแท่งเดียวกัน** แทน: `.bar-range-cap` เป็นแท่งเงาสีเดียวกับ `.bar` (ผ่าน `color-mix(in srgb, var(--severity) 28%, transparent)`) สูงเท่า `pm25_hi` เต็ม วางอยู่หลัง `.bar` ทึบ (z-index ต่ำกว่า) ทำให้ส่วนล่าง (0→pm25 จุดพยากรณ์) ถูกแท่งทึบบังไว้หมด เหลือให้เห็นแค่ "ส่วนเกิน" จากจุดพยากรณ์ขึ้นไปถึงขอบบน P95 จริง — อ่านเป็นแท่งเดียวกันที่ค่อยๆ จางขึ้นไป ไม่ใช่ของแปลกปลอมที่ลอยแยก ไม่แสดง `pm25_lo` แยกอีกต่อไป (อยู่ใต้แท่งทึบเสมออยู่แล้ว ไม่มีอะไรให้เห็นเพิ่มบนกราฟ ค่าเต็มยังอยู่ใน tooltip) ลบ `--range-ink`/`--range-ring` และ CSS ของ dot/line ทิ้งทั้งหมด (ไม่ใช้แล้ว) ปรับ caption ใต้หัวข้อกราฟเป็น "ส่วนจางด้านบนคือขอบบนของช่วงความเชื่อมั่น ~90%"

**ทดสอบ**: `npm run build` ผ่าน, สกรีนช็อต Playwright ทั้ง light/dark mode ยืนยันส่วนขยายเรนเดอร์เนียนต่อจากแท่งทึบจริง ไม่มี artifact ขอบ, zoom ดูใกล้ๆ ยืนยันอ่านง่ายกว่าเดิมมาก (ไม่มี element ลอยแยกจากแท่งอีกต่อไป)

**ไฟล์ที่แก้**: `frontend/src/views/ForecastView.vue`, `CHANGELOG.md` (entry นี้)

## 2026-09-13 (83) — แก้หน้าเปรียบเทียบไม่รีเฟรชข้อมูลเมื่อสลับจังหวัดจาก header

**ที่มา**: ผู้ใช้ถามว่าหน้าเปรียบเทียบควรแยกตามจังหวัดด้วยไหม — ตรวจสอบแล้วพบว่าจริงๆ `/api/comparison`/`/api/comparison/overview` กรองตาม `province` อยู่แล้ว (ผูกกับตัวเลือกจังหวัดบน header) แต่ `CompareView.vue` ขาด `watch()` บน `provinceState.selected` ต่างจาก `ForecastView.vue` ที่มีอยู่แล้ว (`watch(() => provinceState.selected, loadForecast)`) — ทำให้ถ้าเปิดหน้า compare ค้างไว้แล้วสลับจังหวัดจาก dropdown ข้อมูลจะไม่รีเฟรชตาม ต้องรีโหลดหน้าเองถึงจะเห็นจังหวัดใหม่ ยืนยันแนวทางแก้กับผู้ใช้แล้วดำเนินการ

**สิ่งที่ทำ**: `CompareView.vue` เพิ่ม `watch(() => provinceState.selected, () => { loadOverview(); loadComparison(); })` (pattern เดียวกับ ForecastView.vue เป๊ะ) รีโหลดทั้งการ์ดภาพรวมและกราฟรายละเอียดต่อ horizon เมื่อจังหวัดเปลี่ยน

**ทดสอบ**: สคริปต์ Playwright คลิกสลับจังหวัดจาก dropdown บนหน้า `/compare` ที่เปิดค้างไว้ (ไม่รีเฟรชหน้า) ดัก network request ยืนยันยิง `/api/comparison?...province=loei` และ `/api/comparison/overview?province=loei` ใหม่ถูกต้องหลังสลับ, สกรีนช็อตยืนยันการ์ดภาพรวมเปลี่ยนเป็นข้อมูลของเลยจริง (n=48, MAE ±0.10 ตามข้อมูลจริงของจังหวัดนั้น) ไม่ใช่ข้อมูลขอนแก่นค้างอยู่, `npm run build` ผ่าน

**ไฟล์ที่แก้**: `frontend/src/views/CompareView.vue`, `CHANGELOG.md` (entry นี้)

## 2026-09-13 (82) — หน้าเปรียบเทียบแสดงช่วงความเชื่อมั่น 90% + empirical coverage (ให้สอดคล้องกับ pooled model)

**ที่มา**: ผู้ใช้ขอให้ปรับหน้าเปรียบเทียบให้สอดคล้องกับโมเดล pooled ปัจจุบัน สำรวจโค้ดแล้วพบจุดที่ไม่ตรงกันจริง: `get_forecast()` คำนวณช่วงความเชื่อมั่น P05-P95 (conformal calibration, เป้าหมาย ~90% coverage ตาม ALPHA=0.10) และส่งให้ ForecastView แสดงเป็น error bar อยู่แล้ว (CHANGELOG 79) แต่ `prediction_log` ไม่เคยเก็บ `pm25_lo`/`pm25_hi` เลย มีแค่ `predicted_pm25` (จุดเดียว) — ทำให้หน้าเปรียบเทียบไม่มีทางรู้ได้เลยว่า conformal calibration ที่เป็นเทคนิคหลักของงานวิจัยนี้ ทำงานตรงเป้า ~90% coverage จริงไหมเมื่อเทียบกับค่าจริงที่วัดได้ ถือเป็นช่องว่างสำคัญที่สุดระหว่างหน้าเปรียบเทียบกับตัวโมเดลจริง

**สิ่งที่ทำ**:
- **`app.py`**: migration ใหม่ `migrate_prediction_interval_columns()` เพิ่มคอลัมน์ `pm25_lo`/`pm25_hi` (nullable) ให้ `prediction_log` — แถวเก่าก่อนหน้านี้เป็น NULL ถาวร (ไม่มีทาง backfill ย้อนหลัง เพราะไม่ได้เก็บ quantile prediction ตอนนั้นไว้) `log_predictions()`/`get_forecast()` แก้ให้ log ค่าช่วงที่คำนวณอยู่แล้วลง DB ด้วย (ก่อนหน้านี้คำนวณแล้วทิ้ง ไม่เคย persist)
- **`/api/comparison`**: คืน `pm25_lo`/`pm25_hi`/`in_interval` ต่อจุด + `coverage_pct`/`n_coverage` ใน summary (% ของคู่ที่ค่าจริงตกอยู่ในช่วง 90% จริง) กรอง `naive_fallback` ออกจาก coverage เสมอ (ช่วงกว้าง 0 เสมอ ไม่ใช่ conformal interval จริง วัด coverage กับมันไม่มีความหมาย) กรองแถวที่ไม่มี `pm25_lo`/`pm25_hi` (แถวเก่า) ออกจากการคำนวณด้วยเช่นกัน
- **`/api/comparison/overview`**: เพิ่ม `overall_coverage_pct`/`overall_n_coverage` + `coverage_pct`/`n_coverage` แยกตาม `by_horizon` และ `by_model` ด้วย pattern เดียวกัน
- **`CompareView.vue`**: วาดช่วงความเชื่อมั่นเป็น area wash สีน้ำเงิน ~12% opacity ซ้อนใต้เส้นพยากรณ์/ค่าจริง (ตาม mark spec ของ `dataviz` skill — area fill = series hue ที่ ~10% opacity ไม่ใช่บล็อกทึบ) วาดแยกเป็นช่วงต่อเนื่อง (run) เพื่อข้ามจุดที่ไม่มีค่าช่วงแทนที่จะทำให้แถบขาด เพิ่มข้อความสรุป coverage แบบสามทาง (ใกล้เป้า ~90% / ต่ำกว่าเป้า / สูงกว่าเป้า) ทั้งในการ์ดภาพรวมและรายละเอียดต่อ horizon เพิ่ม legend swatch ใหม่ และ stat tile "Coverage" ในส่วนรายละเอียดทางเทคนิค

**ทดสอบ**: unit test แบบ marker-row (predicted_at ปี 2100 ตาม pattern เดิมของโปรเจกต์ ลบทิ้งใน `finally`) ยืนยัน `log_predictions()` เขียน `pm25_lo`/`pm25_hi` ถูกต้อง และ `/api/comparison/overview` คำนวณ coverage ถูกต้อง (100% เมื่อค่าจริงอยู่ในช่วง) แทรกชุดข้อมูลจำลอง 6 จุดที่มี 1 จุดหลุดช่วงโดยตั้งใจ ยืนยัน `/api/comparison` คืน `coverage_pct=83.3%` ตรงตามที่คำนวณมือ และสกรีนช็อต Playwright ยืนยันแถบสีเรนเดอร์ถูกตำแหน่งจริง (เห็นเส้นค่าจริงหลุดออกนอกแถบตรงจุดที่ตั้งใจให้หลุด) ลบ marker rows ออกหมดหลังทดสอบ, `npm run build` ผ่าน

**ไฟล์ที่แก้**: `app.py`, `frontend/src/views/CompareView.vue`, `CHANGELOG.md` (entry นี้)

## 2026-09-13 (81) — เปิดหน้า "เปรียบเทียบ" (/compare) ในเมนูอีกครั้ง แยกจาก SHOW_EXTRA_FEATURES

**ที่มา**: ผู้ใช้ขอให้วางแผนและทำหน้าเปรียบเทียบผลพยากรณ์กับค่าจริง สำรวจโค้ดก่อนแล้วพบว่า `CompareView.vue` + `/api/comparison`, `/api/comparison/overview` มีอยู่ครบแล้ว (hero card สรุปความแม่นยำ, กราฟแท่งความคลาดเคลื่อนตาม horizon, กราฟเส้นพยากรณ์เทียบค่าจริงต่อจุด, breakdown ตามโมเดล) ใช้ชื่อโมเดล pooled + query param `province` ถูกต้องอยู่แล้ว (sync กับ entry 78) — แค่ถูกซ่อนไว้เพราะ entry ตอนตั้งเครื่องใหม่ปิด `SHOW_EXTRA_FEATURES` รวบทุกฟีเจอร์เสริม (compare, notifications, login) ไว้ด้วยกัน ทั้งที่หน้า compare ใช้แค่ Postgres (`prediction_log`) ซึ่งตั้งค่าเสร็จแล้ว ไม่ต้องพึ่ง Gemini/auth เหมือนอีกสองอย่าง จึงไม่จำเป็นต้องรอ

**สิ่งที่ทำ**: `AppHeader.vue` แยกธง `SHOW_COMPARE = true` ออกจาก `SHOW_EXTRA_FEATURES` (ที่ยังคง `false` ไว้สำหรับ notifications/login) ผูกกับ tab "เปรียบเทียบ" ในนำทางแทน ไม่ต้องแก้ `CompareView.vue`/`app.py`/router เลยเพราะของเดิมถูกต่อสายไว้ถูกต้องอยู่แล้วตั้งแต่ entry 78

**ทดสอบ**: `curl /api/comparison/overview?province=khonkaen` ยืนยันคืนข้อมูลจริงจาก `prediction_log` (n_total=48 จากที่ scheduler ยิงไปแล้ว, n_matched=0 เพราะยังไม่ถึงเวลาของ horizon ไหนเลย — ถูกต้องตามที่คาดสำหรับเครื่องที่เพิ่งรันได้ไม่นาน), `npm run build` ผ่าน, สกรีนช็อต Playwright ที่ `/compare` ยืนยัน nav tab ใหม่ขึ้นและ active ถูกต้อง, หน้าเปรียบเทียบเรนเดอร์ empty-state message ที่เป็นมิตรถูกต้อง (ไม่ error) รอข้อมูล backfill เข้ามาเรื่อยๆ ตามเวลาจริง

**ไฟล์ที่แก้**: `frontend/src/components/AppHeader.vue`, `CHANGELOG.md` (entry นี้)

## 2026-09-13 (80) — ปรับ design ตัวแสดงช่วงความเชื่อมั่นบนกราฟให้สวยขึ้น (ตาม dataviz skill)

**ที่มา**: entry 79 เพิ่ม error bar แบบ tick-cap (เส้นตรง + ขีดแนวนอนบน-ล่าง) ผู้ใช้บอกว่ายังไม่สวย โหลด `dataviz` skill มาอ้างอิง mark spec ก่อนแก้ (line 2px round cap, marker >=8px มี surface ring แยกจากพื้นหลังที่ทับซ้อนกัน)

**สิ่งที่ทำ**: `ForecastView.vue` เปลี่ยนจาก tick-cap (`.error-bar`/`.error-bar-cap`) เป็นเส้นบาง + จุดกลมเล็กบน-ล่าง (`.range-indicator`/`.range-line`/`.range-dot`) ทรงกลมเข้าชุดกับองค์ประกอบอื่นที่มีอยู่แล้วในหน้านี้ (mood-badge, model-dot ใต้กราฟ ก็เป็นวงกลมทั้งคู่) จุดแต่ละจุดมี ring สีทึบ (`--range-ring`) ล้อมรอบ (ไม่ใช้ `--card-bg` ที่โปร่งแสงแบบกระจก เพราะจะกลืนกับสีแท่งอิ่มตัวด้านล่าง อ่านไม่ออก) ให้ตัดกับสีแท่ง PM2.5 (เขียว/เหลือง/ส้ม/แดง) ได้ชัดในทุกกรณี เพิ่ม `--range-ink`/`--range-ring` เป็น CSS var แยก light/dark mode ตาม pattern เดิมของไฟล์ (เหมือน `--model-*`)

**ทดสอบ**: `npm run build` ผ่าน, สกรีนช็อต Playwright zoom เข้าไปดูใกล้ๆ ยืนยันจุด+เส้นอ่านง่าย ไม่กลืนกับสีแท่งด้านล่างเลยสักจุด (ตรวจครบ 5 สีระดับ PM2.5 ที่ปรากฏในกราฟจริง)

**ไฟล์ที่แก้**: `frontend/src/views/ForecastView.vue`, `CHANGELOG.md` (entry นี้)

## 2026-09-13 (79) — แสดงช่วงความเชื่อมั่น (P05-P95 conformal) บนกราฟพยากรณ์

**ที่มา**: หลังเปลี่ยนไปใช้ pooled model (entry 78) ผู้ใช้ถามว่าควรโชว์ช่วงความเชื่อมั่น (P90) ให้ผู้ใช้เห็นด้วยไหม เพราะ backend คำนวณ P05/P95 + conformal calibration ไว้อยู่แล้ว (ใช้แค่เช็ค threshold เตือนภัยข้างใน ไม่เคยส่งออกมาให้ frontend เลย) — เห็นด้วยว่าควรโชว์เพราะเป็นเทคนิคหลักของงานวิจัยนี้ ผู้ใช้ยืนยันให้ทำเลย

**สิ่งที่ทำ**:
- **`app.py`** `get_forecast()`: เดิมคำนวณแค่ P95 upper bound (`p95_upper_adj`) สำหรับเช็ค risk threshold เพิ่มการคำนวณ P05 lower bound (`p05_adj = p05_pred - q_hat`) คู่กัน แล้วใส่ `pm25_lo`/`pm25_hi` (clamp ไม่ให้ติดลบ เพราะ PM2.5 เป็นปริมาณจริงเชิงบวกเสมอ แต่ residual ที่บวกกลับอาจดันค่าต่ำกว่า 0 ได้ทางคณิตศาสตร์) เข้าไปในแต่ละจุดของ `forecast[]` ที่ตอบกลับ — แถว `naive_fallback` ใช้ `pred` ตรงๆ เป็นทั้ง lo/hi (ไม่มี margin สื่อว่า "ความเชื่อมั่นต่ำ" ผ่าน `model_used` แทนที่จะเป็นช่วงกว้าง)
- **`frontend/src/views/ForecastView.vue`**: เพิ่ม error bar (เส้นขีด + cap บน-ล่าง) ซ้อนบนกราฟแท่งเดิม แสดงช่วง `pm25_lo`-`pm25_hi` ต่อจุด, ปรับ `maxPm` ให้รวม `pm25_hi` ด้วย (ไม่งั้นเส้น error bar โผล่พ้นกราฟตอน horizon ไกลๆ ที่ช่วงกว้าง), เพิ่ม caption "เส้นขีดคือช่วงความเชื่อมั่น ~90%" ใต้หัวข้อกราฟ, tooltip แต่ละแท่งโชว์ช่วงด้วย

**ทดสอบ**: `curl /api/forecast` ยืนยัน `pm25_lo`/`pm25_hi` มาถูกต้อง (กว้างขึ้นตาม horizon ตามที่คาด), `npm run build` ผ่าน, สกรีนช็อต Playwright ยืนยันเส้น error bar เรนเดอร์ถูกตำแหน่งและกว้างขึ้นตาม horizon จริง

**ไฟล์ที่แก้**: `app.py`, `frontend/src/views/ForecastView.vue`, `CHANGELOG.md` (entry นี้)

## 2026-09-13 (78) — เปลี่ยน app.py + frontend ไปใช้ pooled model จริง เปิดให้บริการครบ 20 จังหวัด

**ที่มา**: เทรนโมเดล pooled ใหม่เสร็จแล้ว (CHANGELOG 75-76) ผู้ใช้ต้องการเอาไปใช้งานจริง — ยืนยันขอบเขตผ่าน AskUserQuestion ว่าต้องการ "เปิดให้บริการครบ 20 จังหวัด (มีตัวเลือกจังหวัด)" ไม่ใช่แค่สลับโมเดลขอนแก่นเฉยๆ วางแผนละเอียดผ่าน Plan mode (Explore + Plan agent สำรวจโค้ด `app.py` เต็มไฟล์ก่อน) แล้วดำเนินการตามแผน

**การตัดสินใจสำคัญ (ยืนยันกับผู้ใช้ผ่าน AskUserQuestion)**:
1. ตัด Prophet ออกทั้งหมด (รวมขอนแก่นด้วย) — เหลือแค่ pooled LightGBM/XGBoost ทุกจังหวัด เลิกใช้ Air4Thai/calibration ไปพร้อมกัน (pooled model เทรนบน OpenAQ ล้วน live inference จึงดึงจาก OpenAQ ตรงๆ ไม่ต้อง calibrate)
2. Alert threshold ใช้ขอบบน P95 (conformal-adjusted) แทน bias-hack เดิม (โมเดลใหม่ไม่มี bias-by-horizon CSV แยกต่อจังหวัด) — `HIGH_POLLUTION_THRESHOLD=37.5` ยังเป็นค่าเดียวกันทุกจังหวัด
3. `province` query param default เป็น `khonkaen` ถ้าไม่ส่งมา (backward compatible) แต่ 400 ถ้าส่งมาแล้วไม่ตรง registry
4. SHAP-based NLG factor explanation ปิดไว้ชั่วคราว (`scripts/shap_analysis.py` ยังผูกกับโครงสร้างโมเดลเก่า `{horizon: model}` ไม่ใช่ `{horizon: {quantile: model}}` ของ pooled model) — fallback ไปข้อความทั่วไปที่มีอยู่แล้ว ไม่ซ่อมให้เข้ากับโมเดลใหม่ในรอบนี้

**สิ่งที่ทำใน `app.py`**:
- Constants/loading: โหลด `data/province_station_registry.json` เป็น `PROVINCE_REGISTRY`/`PROVINCES_BY_SLUG` ตอน startup, เปลี่ยนไปโหลด pooled pkl 2 ไฟล์ (`lgb_pm25_multiprovince_residual_bagged_quantile_20260913_014901.pkl`, `xgb_..._20260913_014901.pkl`) + conformal CSV คู่ (`{horizon: q_hat}` ต่อโมเดล) ลบ `PROPHET_MODEL_PATH`/`BIAS_CSV_PATH`/`XGB_BIAS_CSV_PATH`/`CALIBRATION_PATH`/`CALIBRATION_A/B`/`STATION_ID_AIR4THAI`/`STATION_LAT`/`STATION_LON` ทิ้งทั้งหมด (ไฟล์เก่ายังอยู่ใน `models/` ตาม convention)
- `fetch_recent_pm25()` → `fetch_recent_pm25_openaq(sensor_id)`: ดึงจาก OpenAQ v3 (`/v3/sensors/{id}/measurements`) ย้อนหลัง 72 ชม. พร้อม retry/backoff (429/5xx — เจอ 500 ชั่วคราวจริงตอนเก็บข้อมูลเทรน)
- `fetch_current_weather()` → `fetch_current_weather_openmeteo(lat, lon)`: เพิ่ม `wind_direction_10m`/`surface_pressure` รับ lat/lon เป็น argument
- `build_feature_row()`: เขียนใหม่เป็น 23 feature (เพิ่ม wind_dir_sin/cos, pressure, lat, lon) **และแก้จาก positional indexing (`.iloc[-N]`) เป็น time-indexed lookup** เพราะ OpenAQ อาจมีชั่วโมงขาดหายได้ (Air4Thai เดิมมักไม่มี gap ทำให้ positional ใช้ได้ แต่ผิดหลักการ) — interpolate gap ≤48h เหมือนตอนเทรน ปล่อย NaN ถ้ายาวกว่านั้นให้ circuit breaker จับได้ถูกต้อง
- `select_model()`/`compute_rolling_r2()`/`compute_missing_ratio()`: เพิ่ม `province` param ทุกจุด, ตัด Prophet branch ใส่ `"naive_fallback"` แทน (persistence: ทายเท่า `latest_pm25` ทุก horizon เมื่อ missing_ratio สูงเกิน — ไม่ป้อน NaN เข้าโมเดลเพราะโมเดล pooled ไม่เคยเห็นแถวมี NaN ตอนเทรนเลย), เปลี่ยนชื่อ `model_used` เป็น `"lightgbm_pooled"`/`"xgboost_pooled"` (กัน `compute_rolling_r2()` ปนผลงานโมเดลเก่ากับใหม่ช่วงเปลี่ยนผ่าน)
- DB migration ใหม่ `migrate_province_column()`: เพิ่ม `province TEXT NOT NULL DEFAULT 'khonkaen'` ให้ `prediction_log`/`daily_nlg_log`/`notification_log` ทั้ง 3 ตาราง + สลับ UNIQUE constraint ให้รวม province (pattern เดียวกับ `migrate_daily_nlg_risk_group()`) — **เจอบั๊กจริงระหว่างทดสอบ**: `migrate_daily_nlg_risk_group()` เช็คแค่ชื่อ constraint ของตัวเอง พอ `migrate_province_column()` รันสำเร็จไปแล้วครั้งนึง (เปลี่ยนชื่อ constraint ไปแล้ว) รอบถัดไป `migrate_daily_nlg_risk_group()` เห็นว่าไม่มีชื่อของตัวเอง พยายามสร้าง constraint 2 คอลัมน์ซ้ำ ชนกับข้อมูลจริงที่มีหลายจังหวัดวันเดียวกันแล้ว (`UniqueViolation`) — แก้โดยให้เช็คทั้ง 2 ชื่อ constraint (ของตัวเองและของ migration ถัดไป) ก่อนตัดสินใจ
- `get_forecast(province, risk_group)`, `scheduled_forecast_job()` (loop ทุกจังหวัดใน registry, try/except ต่อจังหวัด + sleep กันยิง API รัว): thread province ผ่านทุกจุด, ตัด Prophet fields ใส่ `fallback_triggered_horizons`, echo `"province": {"th","en","is_zero_shot"}` กลับไปให้ frontend
- Routes: `/api/forecast`, `/api/comparison`, `/api/comparison/overview`, `/api/notifications` เพิ่ม `province` query param + filter SQL ทุกจุด (รวม `/api/notifications`'s join ระหว่าง `notification_log`/`prediction_log` ที่ต้องเพิ่มเงื่อนไข province เท่ากันด้วย ไม่งั้น cross จังหวัดที่ timestamp ตรงกันได้), เพิ่ม `GET /api/provinces` ใหม่คืน registry ให้ frontend
- ปิด SHAP call ใน `generate_daily_nlg_explanation()` (ส่ง `shap_pairs=None` ตรงๆ — `_format_shap_factor_lines()` รองรับ None อยู่แล้ว), ลบ `import get_shap_explanation`/`sys.path.insert` ที่ไม่ใช้แล้ว

**สิ่งที่ทำใน `frontend/`**:
- `provinces.js` (ใหม่): reactive state (`list`, `selected` — persist ผ่าน `localStorage`) + `fetchProvinces()`
- `notifications.js`: `fetchForecast()`/`fetchNotificationHistory()` แนบ `?province=`, เปลี่ยน `prophetHorizons`→`fallbackHorizons` field
- `AppHeader.vue`: เพิ่ม province `<select>` แสดงเสมอ (ไม่ผูกกับ `SHOW_EXTRA_FEATURES` ที่ซ่อน compare/notif/login อยู่แล้ว), เปลี่ยนแบรนด์ "PM2.5 Khon Kaen" → "PM2.5 ภาคอีสาน", แก้ข้อความ Prophet ในกระดิ่งแจ้งเตือน
- `ForecastView.vue`: `watch(() => provinceState.selected, loadForecast)` ให้สลับจังหวัดแล้วโหลดใหม่, ชื่อจังหวัดในหัวข้อมาจาก response จริงแทน hardcode "สถานี:ส่วนอุทกวิทยา...", แก้สี/legend กราฟ (LightGBM/XGBoost/Fallback แทน Prophet)
- `CompareView.vue`: แก้ model names + เพิ่ม province param (แม้หน้านี้ถูกซ่อนอยู่หลัง `SHOW_EXTRA_FEATURES=false` ก็แก้ให้ถูกไว้)

**ทดสอบ**:
- `python -c "import ast; ast.parse(...)"` ผ่าน, boot `app.py` สำเร็จไม่มี exception
- `curl /api/forecast` (default → ขอนแก่น) และ `?province=loei/nongkhai/nakhonphanom/roiet/mahasarakham` (รวม 3 จังหวัด zero-shot) → 200 ครบ, `province`/`model_used` (`lightgbm_pooled`/`xgboost_pooled`) ถูกต้อง
- `curl /api/forecast?province=bangkok` (จังหวัดไม่รู้จัก) → 400 ตามที่ออกแบบไว้
- `curl /api/provinces` → 20 จังหวัดครบ
- Unit test ตรง (import `app.py` ผ่าน `importlib`, มาตรฐาน `scripts/test_dynamic_routing.py`): feature row เต็ม NaN → `select_model()` คืน `"naive_fallback"` ถูกต้อง, feature row ปกติ → คืน `"lightgbm_pooled"`/`"xgboost_pooled"` ถูกต้อง — ระหว่างทดสอบเจอบั๊ก migration idempotency ข้างต้น แก้แล้วทดสอบซ้ำผ่าน
- `npm run build` ผ่าน ไม่มี error
- สกรีนช็อตจริงผ่าน Playwright: province selector แสดง "ขอนแก่น", หัวข้อ "จังหวัดขอนแก่น" ถูกต้อง, กราฟพยากรณ์แสดง LightGBM/XGBoost (ไม่มี Prophet) ตามที่ออกแบบ

**หมายเหตุ/ขอบเขตที่ยังไม่ทำ**: SHAP explanation, `/api/notifications`'s historical risk check ใช้ `predicted_pm25` ตรงๆ (ไม่มีขอบบน P95 เพราะไม่ได้ persist ค่านี้ลง DB — หน้านี้ถูกซ่อนอยู่ใน frontend ตอนนี้อยู่แล้ว ยอมรับความแม่นยำที่ลดลงได้), ฤดุฝุ่นหนารอบหน้า (พ.ย. 2569) ยังไม่ได้ backtest กับสถาปัตยกรรมนี้ (ดู CHANGELOG 76 เรื่องข้อจำกัดของการ backtest ฤดูฝนที่ทำไปก่อนหน้า)

**ไฟล์ที่แก้**: `app.py` (แก้ใหญ่ทั้งไฟล์), `frontend/src/provinces.js` (ใหม่), `frontend/src/notifications.js`, `frontend/src/components/AppHeader.vue`, `frontend/src/views/ForecastView.vue`, `frontend/src/views/CompareView.vue`, `CLAUDE.md` (อัปเดตสถาปัตยกรรมทุก section ที่เกี่ยวข้อง), `CHANGELOG.md` (entry นี้)

## 2026-09-13 (77) — ตั้งค่ารัน app.py + frontend บน PC เครื่องใหม่ (Gemini client lazy-init + ปิดฟีเจอร์นอกเหนือจากพยากรณ์ชั่วคราว)

**ที่มา**: ผู้ใช้ย้ายมาทำงานบน PC เครื่องนี้ (โน้ตบุ๊กเดิมพัง) ต้องการให้ `app.py`/`frontend/` รันได้ในเครื่องนี้ — เครื่องนี้ยังไม่เคยตั้งค่าอะไรเลย (ไม่มี PostgreSQL, ไม่มี `.env` ค่าอื่นนอกจาก `OPENAQ_KEY`, ไม่มี `frontend/node_modules`) จากนั้นผู้ใช้ขอให้ปิดฟีเจอร์อื่นที่ไม่ใช่พยากรณ์/ค่าฝุ่นปัจจุบันไว้ก่อน เพื่อลดพื้นผิวที่ต้องทดสอบตอนตั้งเครื่องใหม่

**สิ่งที่ทำ**:
1. ติดตั้ง PostgreSQL 18 ในเครื่อง (ผู้ใช้ติดตั้งเอง) สร้างฐานข้อมูล `pm25_forecast` ผ่าน `psql` (`CREATE DATABASE`) — `app.py`'s `init_db()`/`migrate_*()` สร้างตารางที่เหลือเองตอน boot ตามที่ออกแบบไว้แล้ว
2. เติม `.env` ให้ครบตามที่ `_load_env()` ต้องการ (เดิมมีแค่ `OPENAQ_KEY`): สุ่ม `SECRET_KEY` ใหม่ (48 ตัวอักษร), `POSTGRES_HOST=localhost`/`POSTGRES_PORT=5432`/`POSTGRES_DB=pm25_forecast`/`POSTGRES_USER=postgres`/`POSTGRES_PASSWORD` (ตามที่ผู้ใช้ตั้งตอนติดตั้ง Postgres), `GEMINI_KEY` เว้นว่างไว้ก่อน (ผู้ใช้ยังไม่มี key)
3. **`app.py`**: `_gemini_client = genai.Client(api_key=GEMINI_KEY)` (บรรทัด 898 เดิม) crash ทันทีตอน import ถ้า `GEMINI_KEY` ว่าง เพราะ `google-genai` เช็ค key ตอนสร้าง client เลย — แก้เป็น lazy: สร้าง client เฉพาะเมื่อมี key จริง (`None` ถ้าไม่มี/สร้างไม่สำเร็จ) และเพิ่ม guard ใน `_call_gemini()` ให้คืน `None` ทันทีถ้า client เป็น `None` (fallback ไป template message ผ่าน logic เดิมที่มีอยู่แล้วใน `generate_daily_nlg_explanation()` ไม่ต้องเพิ่ม logic ใหม่)
4. **`frontend/src/components/AppHeader.vue`**: เพิ่ม constant `SHOW_EXTRA_FEATURES = false` ปิดลิงก์ "เปรียบเทียบ", กระดิ่งแจ้งเตือน, เมนู login/register/profile ไว้ก่อน (เหลือแค่แบรนด์ + แท็บ "พยากรณ์") และข้าม `fetchCurrentUser()`/`fetchForecast()` (ที่ prewarm สำหรับฟีเจอร์ที่ซ่อนอยู่) ตอน mount
5. **`frontend/src/views/ForecastView.vue`**: เพิ่ม constant `SHOW_EXTRA_CARDS = false` ปิดการ์ดสรุปวันนี้ (NLG), ช่วงเวลาอากาศดีที่สุด, คะแนนความเสี่ยงส่วนบุคคล ไว้ก่อน เหลือแค่การ์ดค่าฝุ่นปัจจุบัน (+สภาพอากาศ) กับกราฟพยากรณ์ 24 ชม. ตามที่ผู้ใช้ขอ — ทั้งสอง flag เปลี่ยนกลับเป็น `true` ได้ทันทีเมื่อพร้อมเปิดฟีเจอร์อื่นอีกครั้ง (ไม่ได้ลบโค้ดทิ้ง)
6. `pip install -r requirements.txt` + `cd frontend && npm install && npm run build` (ยังไม่เคย install ในเครื่องนี้เลย)

**ทดสอบ**:
- `python app.py` บูตสำเร็จ ไม่มี exception, DB connect ได้, โมเดลโหลดครบ
- `curl /api/forecast` → 200, คืนค่าพยากรณ์ 24 ชม. + ค่าฝุ่นปัจจุบันจริงครบ (`daily_summary.used_fallback=true` ตามคาด เพราะไม่มี Gemini key)
- `curl /` และ `/assets/*.js|css` → 200 (frontend build เสิร์ฟผ่าน Flask ถูกต้อง)
- สกรีนช็อตจริงผ่าน Playwright ยืนยัน: หน้า dashboard เหลือแค่การ์ดค่าฝุ่นปัจจุบัน + กราฟพยากรณ์ 24 ชม. ตามที่ขอ, header เหลือแค่แท็บ "พยากรณ์" ไม่มีเปรียบเทียบ/แจ้งเตือน/login

**หมายเหตุ**: ยังไม่มี `GEMINI_KEY` จริง — ฟีเจอร์ NLG (ที่ถูกซ่อนอยู่แล้วตอนนี้) จะ fallback ไป template message เสมอจนกว่าจะตั้งค่า ไม่กระทบการทำงานส่วนอื่น

**ไฟล์ที่แก้**: `.env` (เติมค่า ไม่ commit), `app.py` (Gemini client lazy-init), `frontend/src/components/AppHeader.vue`, `frontend/src/views/ForecastView.vue`, `CHANGELOG.md` (entry นี้)

## 2026-09-13 (76) — Backtest โมเดล multiprovince กับข้อมูลจริงหลังช่วงเทรน (2 ก.ค.–13 ก.ย. 2569)

**ที่มา**: หลังเทรนโมเดล multiprovince เสร็จ (entry 75) ผู้ใช้ถามว่าควรเอาโมเดลไปทดสอบกับข้อมูลจริง/ค่าจริงไหม — แนะนำว่าควรทำ เพราะ test set เดิม (20% ท้ายของแต่ละจังหวัด) ยังอยู่ในช่วงเวลาเดียวกับที่เทรน (2024-01-01..2026-07-02) การทดสอบกับข้อมูลที่เกิด**หลัง**ช่วงเทรนเป็นการยืนยันที่แข็งแรงกว่าเพราะรั่วเข้าไปตอนเทรนไม่ได้เลย ผู้ใช้ขอบเขตแค่ backtest ก่อน (ยังไม่ต้อง live monitoring ต่อเนื่อง)

**สิ่งที่ทำ**: `scripts/backtest_recent_multiprovince.py` (ใหม่) — ดึงข้อมูลจริงจาก OpenAQ/Open-Meteo ช่วง 2026-06-27 (buffer 5 วันให้ lag_24/roll_24 มีข้อมูลย้อนหลังพอ) ถึงปัจจุบัน (13 ก.ย. 2569) ของ 20 จังหวัด คำนวณ feature ด้วยสูตรเดียวกับตอนเทรนเป๊ะ (ใช้ TRAIN_WINDOW_END=2026-07-02 ตัดเอาเฉพาะแถวที่อยู่**หลัง**ช่วงเทรนจริงมาประเมิน) โหลดโมเดล+conformal calibration ที่เทรนไว้ (`*_20260913_014901.*`) มา predict ทุก horizon แล้วเทียบกับค่าจริงที่เกิดขึ้นแล้ว (MAE, R², coverage หลัง conformal) ไม่แตะ `app.py`/DB

**ผลทดสอบ**:
- 17 จังหวัดเทรน: LightGBM MAE=1.02 R²=0.61 coverage=92.3% | XGBoost MAE=1.06 R²=0.58 coverage=93.5%
- 3 จังหวัด zero-shot: LightGBM MAE=1.21 R²=0.76 coverage=87.6% | XGBoost MAE=1.28 R²=0.74 coverage=90.5%
- **MAE ต่ำกว่า test set เดิมมาก (1.0-1.3 เทียบ 2.9) แต่ R² ตกจาก 0.92-0.93 เหลือ 0.58-0.76** — วิเคราะห์เพิ่มพบว่าไม่ใช่โมเดลมีปัญหา แต่เป็นผล**ฤดูกาล**: ช่วง backtest (ก.ค.-ก.ย.) เป็นฤดูฝนล้วน pm25 นิ่งและต่ำกว่าฤดูฝุ่นหนามาก (std pm25 ฤดูฝุ่นหนา vs ฤดูฝน: บึงกาฬ 29.9 vs 6.6 = ต่างกัน 4.5 เท่า, อุดรธานี/สกลนคร ~2.1-2.2 เท่า) — target variance ต่ำทำให้ R² ไวต่อ noise ผิดปกติ (ดูได้จาก R² ต่อ horizon ที่ไล่จาก 0.93-0.98 ที่ t+1 ลงไปติดลบที่ t+24 อย่างค่อยเป็นค่อยไป สอดคล้องกับ MAE ที่โตขึ้นราบเรียบ ไม่กระโดด) จังหวัดที่ variance ลดเยอะที่สุด (บึงกาฬ) ก็เป็นจังหวัดที่ R² แย่ที่สุด (0.16) พอดี ยืนยันสมมติฐาน
- Coverage ต่อจังหวัดเริ่มเบี่ยงจาก 90% ที่ horizon ไกล (อุดรธานี ขึ้นไปถึง ~96% ที่ t+24 = interval กว้างเกิน, บึงกาฬ ลงมา ~84% = แคบไป) เพราะ conformal calibration ปัจจุบันคำนวณ q_hat รวมทุกจังหวัดเป็นค่าเดียวต่อ horizon ไม่ได้แยกต่อจังหวัด

**ข้อสรุป/สิ่งที่ต้องติดตามต่อ**:
- ตัวชี้วัดที่มีผลใช้งานจริง (MAE ขนาด, coverage) ยังอยู่ในเกณฑ์ดี ไม่ถือว่าน่าห่วง — R² ที่ตกเป็นข้อจำกัดทางสถิติของช่วงที่ทดสอบ ไม่ใช่สัญญาณโมเดลพัง
- **ยังไม่รู้ผลในฤดุฝุ่นหนารอบหน้า** (พ.ย. 2569 เป็นต้นไป) — ต้อง backtest ซ้ำตอนนั้นถึงจะเทียบกับ R² ของ test set เดิม (0.92-0.93) ได้อย่างยุติธรรม
- แนวทางปรับปรุงที่พบจากรอบนี้: พิจารณาทำ conformal calibration **แยกต่อจังหวัด** แทนการรวมทุกจังหวัดเป็นค่าเดียว เพื่อให้ coverage แม่นต่อจังหวัดมากขึ้น (ยังไม่ได้ทำ เป็นแค่ข้อสังเกตสำหรับเฟสถัดไป)

**ไฟล์ที่แก้/เพิ่ม**: `scripts/backtest_recent_multiprovince.py` (ใหม่), `models/backtest_recent_by_horizon_20260913_135247.csv` (ผลรายละเอียดต่อ horizon ต่อจังหวัด), `CHANGELOG.md` (entry นี้)

## 2026-09-13 (75) — เทรนโมเดล PM2.5 สถาปัตยกรรมใหม่: 17 จังหวัด + 3 zero-shot (residual + bagging + quantile + conformal calibration)

**ที่มา**: ผู้ใช้ต้องการขยายระบบจากขอนแก่นจังหวัดเดียวเป็น 17 จังหวัดภาคอีสานที่เทรนจริง (เลย, หนองคาย, อุดรธานี, นครราชสีมา, อุบลราชธานี, บุรีรัมย์, สุรินทร์, ศรีสะเกษ, สกลนคร, กาฬสินธุ์, มุกดาหาร, ยโสธร, ชัยภูมิ, อำนาจเจริญ, หนองบัวลำภู, บึงกาฬ + ขอนแก่นเดิม) บวก 3 จังหวัด zero-shot ที่ให้บริการได้แต่ไม่เทรน (นครพนม, ร้อยเอ็ด, มหาสารคาม) ตามผลงานวิจัยที่ทำไว้ก่อนหน้านอก repo นี้ (โน้ตบุ๊กเดิมใช้งานไม่ได้แล้ว ย้ายมาทำต่อบน PC เครื่องนี้ ข้อมูล/โมเดลเดิมไม่มีเหลืออยู่เลย ต้องสร้าง pipeline ใหม่ทั้งหมด) สเปกที่ผู้ใช้ระบุ: 23 features (21 หลัก + lat/lon), เทคนิค residual forecasting + bagging + quantile regression (P05/P50/P95) + conformal calibration, split 65/15/20 (train/calibration/test) แบบ chronological ต่อจังหวัด — **ขอบเขตรอบนี้ตามที่ผู้ใช้ยืนยัน: เทรนโมเดลอย่างเดียว ยังไม่แตะ `app.py`/DB/Dynamic Routing/frontend**

**สิ่งที่ทำ**:
1. ติดตั้ง `requirements.txt` (ยังไม่เคยติดตั้งในเครื่องนี้เลย) + ผู้ใช้สร้าง `.env` ใหม่ใส่ `OPENAQ_KEY` เอง (สมัครที่ explore.openaq.org)
2. `scripts/province_registry.py` (ใหม่) — query OpenAQ v3 API จริงหา location/sensor pm25 ของ 20 จังหวัด (ห้ามเดา ID) เลือกด้วยเกณฑ์: provider=`Air4Thai` (เครือข่าย PCD ทางการ ยืนยันจากขอนแก่น 7747 เอง) ก่อน → coverage overlap กับช่วง 2024-01-01..2026-07-02 (query จริงจาก `datetimeFirst/Last`) → ระยะทางจากจุดค้นหา เจอปัญหาว่า 4 จังหวัด (หนองคาย/อุบลราชธานี/บุรีรัมย์/มุกดาหาร) มี pm25 sensor ซ้ำ 2 ตัวต่อสถานี — ตัวใหม่ (id 8 หลัก ขึ้นต้น `1631`) คืน `500 Internal Server Error` จาก OpenAQ เองทุกครั้ง (ยืนยันด้วย query ตรงหลายรอบ ไม่ใช่ปัญหาชั่วคราว) ส่วนตัวเก่า (id เลขน้อย) ใช้งานได้ปกติ — แก้โดยเลือก sensor id ต่ำสุดเสมอเมื่อมีมากกว่า 1 ตัว บันทึกผลลง `data/province_station_registry.json`
3. `scripts/build_raw_dataset_multiprovince.py` (ใหม่) — ต่อยอด pattern จาก `scripts/build_raw_dataset.py` เดิม (gap ≤48h interpolate / >48h mask ต่อสถานี) ขยายเป็น 20 จังหวัด: ดึง pm25 จาก OpenAQ v3 (ไม่ใช้ Air4Thai's API ตรงๆ เลย — "Air4Thai" ที่เห็นเป็นแค่ชื่อ provider ใน metadata ของ OpenAQ เอง) + weather จาก Open-Meteo archive (เพิ่ม `wind_direction_10m`/`surface_pressure` จากเดิมที่มีแค่ temp/humidity/wind_speed/precip) คำนวณ 23 features (weather 4 + cyclical เวลา 5 + pm25 lag/rolling 9 + wind_dir_sin/cos+pressure 3 + lat/lon 2) มี retry/backoff รองรับ HTTP 500 ชั่วคราวจาก OpenAQ (เจอจริงระหว่างรัน) และ try/except ต่อจังหวัดไม่ให้ 1 จังหวัดพังทั้ง pipeline
4. `scripts/train_multiprovince_models.py` (ใหม่) — เทรน LightGBM+XGBoost บน 17 จังหวัด (ไม่รวม zero-shot) แบบ residual forecasting (target = pm25_t+h − pm25 ปัจจุบัน) + bagging (per-tree subsample/colsample_bytree) + quantile regression (P05/P50/P95, 24 horizon ตรงแบบ direct multi-step) + conformal calibration (CQR มาตรฐาน, ALPHA=0.10 เป้า coverage 90%) บน split 65/15/20 ต่อจังหวัด
5. `scripts/eval_zeroshot_multiprovince.py` (ใหม่) — ทดลอง zero-shot ตามวิธีผู้ใช้ระบุ: กันจังหวัดทดสอบ 3 จังหวัด (อุดรธานี/อุบลราชธานี/ชัยภูมิ — คนละชุดกับ 3 จังหวัด zero-shot จริงที่ไม่เคยเทรนเลย) เทรน 2 แบบ (มี/ไม่มีในชุดเทรน) ประเมินบน test rows ชุดเดียวกันของ 3 จังหวัดนั้น (ยืนยันด้วย assert ว่า test rows เท่ากันจริงเพราะ split ต่อจังหวัดแบบ chronological ไม่ขึ้นกับ pool)

**ผลทดสอบ**:
- `data/multiprovince_pm25_features_raw_20260913_012629.csv`: รวม 363,332 แถว — **17 จังหวัดเทรน = 323,347 แถวพอดี ตรงกับตัวเลขที่ผู้ใช้รายงานไว้เป๊ะ**, 3 จังหวัด zero-shot = 39,985 แถว (มหาสารคาม/ร้อยเอ็ด gap เยอะกว่าปกติ ~10,400 แถว เพราะช่วง >48h ถูก mask เยอะ) self-check `pm25_lag_1` ตรงสูตรอ้างอิง (เฉพาะคู่แถวที่ติดกันจริง 1 ชม.)
- โมเดล deploy (`*_multiprovince_residual_bagged_quantile_20260913_014901.pkl`): LightGBM MAE(P50)=2.914 (ผู้ใช้รายงาน 2.912), coverage 85.8%→89.4% (รายงาน 86.7%→89.5%) | XGBoost MAE(P50)=3.008 (รายงาน 2.993), coverage 82.0%→88.8% (รายงาน 83.1%→88.9%) — **ใกล้เคียงตัวเลขต้นฉบับมาก** ยืนยันว่า pipeline ที่สร้างใหม่สอดคล้องกับงานวิจัยเดิม
- Zero-shot eval (`zeroshot_eval_20260913_021526.csv`): LightGBM in-training MAE=3.061→zero-shot MAE=3.188 (R²=0.920) | XGBoost in-training MAE=3.113→zero-shot MAE=3.273 (R²=0.915) — ทิศทาง/ขนาดผลกระทบสอดคล้องกับที่ผู้ใช้รายงาน (in-training 3.278→zero-shot 3.375, R²=0.912) แม้ตัวเลขสัมบูรณ์ต่างเล็กน้อยเพราะเป็นชุดข้อมูลที่ดึงใหม่ทั้งหมด

**หมายเหตุ/ข้อควรตรวจสอบเพิ่ม**: สถานีที่ registry เลือกให้มหาสารคาม (zero-shot) เป็น Air4Thai "Pramuan Suk Health Park" (coverage 544.9/913 วัน) ซึ่งขัดกับที่ผู้ใช้เคยระบุว่ามหาสารคามมีแต่เซนเซอร์ AirGradient ราคาถูก ไม่ใช่ reference-grade — อาจเป็นเพราะสถานีนี้เพิ่งเริ่มมีข้อมูลใน OpenAQ ทีหลัง ยังไม่ต้องแก้อะไรตอนนี้ (มหาสารคามไม่ได้อยู่ใน 17 จังหวัดเทรน ไม่กระทบผลเทรนรอบนี้) แต่ต้องตัดสินใจตอนเข้า phase serving จริง

**ไฟล์ที่แก้/เพิ่ม**: `scripts/province_registry.py`, `scripts/build_raw_dataset_multiprovince.py`, `scripts/train_multiprovince_models.py`, `scripts/eval_zeroshot_multiprovince.py` (ใหม่ทั้งหมด), `data/province_station_registry.json`, `data/multiprovince_pm25_features_raw_20260913_012629.csv`, `data/multiprovince_build_log_20260913_012629.csv`, `models/{lgb,xgb}_pm25_multiprovince_residual_bagged_quantile_20260913_014901.pkl`, `models/conformal_calibration_{lgb,xgb}_20260913_014901.csv`, `models/multiprovince_eval_summary_20260913_014901.csv`, `models/zeroshot_eval_20260913_021526.csv`, `.env` (ผู้ใช้สร้างเอง, ไม่ commit), `CHANGELOG.md` (entry นี้)

## 2026-07-22 (74) — ทำ scripts/ ให้ path-independent (รองรับการเปลี่ยนชื่อ/ย้ายโฟลเดอร์)

**ที่มา**: ผู้ใช้ถามว่าเปลี่ยนชื่อโฟลเดอร์โปรเจกต์ (`c:\Project1\Model1`) เป็นชื่ออื่นได้ไหม — ตรวจแล้วพบว่า `app.py` portable อยู่แล้ว (relative path + `os.path.dirname(os.path.abspath(__file__))`) แต่ `scripts/*.py` ~30 ไฟล์ hardcode `c:\Project1\Model1\...` เต็มๆ 79 จุด ซึ่งจะพังเวลาเปลี่ยนชื่อ ผู้ใช้เลือกให้ทำ scripts เป็น path-independent (ผ่าน AskUserQuestion)

**วิธีทำ**: เขียนตัวแปลง (scratchpad, ไม่ commit) แทนทุก literal `r"c:\Project1\Model1\SUB"` → `os.path.join(_ROOT, r"SUB")` และเพิ่ม `_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))` (+ `import os` เฉพาะไฟล์ที่ยังไม่มี) ก่อนบรรทัดแรกที่ใช้ path — `_ROOT` = โฟลเดอร์แม่ของ `scripts/` = repo root เสมอ ไม่ว่าโฟลเดอร์จะชื่ออะไร/อยู่ที่ไหน **backup `scripts/*.py` ทั้ง 37 ไฟล์ไว้ก่อน** (โปรเจกต์ไม่ใช่ git) แล้วรัน dry-run ตรวจก่อน apply

**ผล**: แก้ 36 ไฟล์ 79 จุด (อีก 1 ไฟล์ใน scripts ไม่มี hardcode path เลย)

**ทดสอบ**:
- `ast.parse` ทุกไฟล์ใน `scripts/` ผ่านหมด (ไม่มี syntax error จากการแทรก)
- `grep` ยืนยันไม่เหลือ `c:\Project1\Model1` ใน `scripts/*.py` เลย (0 จุด)
- `import shap_analysis` (ตัวที่ `app.py` import ตอน startup) สำเร็จ — `_ROOT` ชี้ repo root ถูกต้อง, `DATA_PATH`/`LGB_MODEL_PATH`/`XGB_MODEL_PATH` `exists=True` ทุกตัว, `get_shap_explanation` ยัง callable
- ตรวจ static อีกหลายไฟล์ (`eval_haze_season_backtest`, `train_xgb_prophet_v2`, `test_risk_score`) — path constant ที่เป็น input ชี้ไฟล์จริงที่มีอยู่ทุกตัว

**สิ่งที่ยังผูกกับ path เดิม (แจ้งผู้ใช้แล้ว)**: `.venv/` (activate scripts hardcode `C:\Project1\Model1\.venv`) — regenerate ใหม่จาก `requirements.txt` ได้ ไม่กระทบการรัน `python app.py` ด้วย global Python; และการเปลี่ยนชื่อโฟลเดอร์เองผู้ใช้ต้องทำใน File Explorer (ปิด IDE ก่อน)

**ไฟล์ที่แก้**: `scripts/*.py` (36 ไฟล์ — เฉพาะ path constants + เพิ่ม `_ROOT`), `CLAUDE.md` (แก้ประโยคที่ระบุว่า scripts hardcode path → ระบุว่า path-independent แล้ว + เตือนไม่ให้ reintroduce)

## 2026-07-22 (73) — จัดระเบียบไฟล์ root: แยก docs/ + notebooks/ ออกจากระดับบนสุด

**ที่มา**: ผู้ใช้บอกว่า root ของโปรเจกต์ดูรก/ไม่เป็นระเบียบ — มีไฟล์เอกสาร ไดอะแกรม proposal notebook และ PNG byproduct ปนกับโค้ดในระดับบนสุด

**ตรวจก่อนย้าย**: `grep` หา reference ของไฟล์ที่จะย้ายใน `app.py`/`scripts/*.py` ทั้งหมด — พบเฉพาะในคอมเมนต์/docstring (เช่น "proposal 4.1.6", "train_models.ipynb cell ...") ไม่มีการโหลดไฟล์จริง จึงย้ายได้โดยไม่กระทบ runtime (โปรเจกต์ไม่ใช่ git repo จึงเช็คให้แน่ใจก่อนเพราะย้อนกลับอัตโนมัติไม่ได้)

**สิ่งที่ทำ**:
- สร้าง `docs/` → ย้าย `database_schema.md`, `usecase_diagram.puml`, `system_architecture.puml`/`.html`/`.png`, `proposal.docx`, `proposal_14.pdf` เข้าไป + เปลี่ยนชื่อ PNG byproduct จาก PlantUML ให้สื่อความหมาย (`PM25_Forecast_UseCase.png`→`docs/usecase_diagram.png`, `PM25_Forecast_SystemArchitecture.png`→`docs/system_architecture_plantuml.png`)
- สร้าง `notebooks/` → ย้าย `Testforcest.ipynb`, `train_models.ipynb`
- ลบ `__pycache__/` (root + `scripts/`) — Python สร้างใหม่เองตอนรัน ไม่ใช่ไฟล์ต้นฉบับ
- **คงไว้ที่ root**: `app.py`, `.env`, `requirements.txt`, `CLAUDE.md`, `CHANGELOG.md`, `TEST_RESULTS.md` + โฟลเดอร์ทำงานเดิม (`models/ data/ scripts/ frontend/ backups/ figures/ pre_vue_frontend_backup/`) — ไม่แตะ path ที่ `app.py` hardcode (`models/*.pkl`, `frontend/dist`) เลย
- อัปเดต `CLAUDE.md`: เพิ่มหัวข้อ "Repo layout for docs & notebooks" + แก้ path notebook 2 ไฟล์ให้ชี้ `notebooks/`

**ทดสอบ**: `ast.parse(app.py)` ผ่าน (ยืนยันไม่มีอะไรพังจากการย้าย เพราะไม่ได้แตะโค้ดหรือ path ที่โหลดจริง), ตรวจ root ใหม่เหลือแต่โค้ด + ไฟล์โปรเจกต์ระดับบน + โฟลเดอร์ทำงาน

**หมายเหตุ**: markdown link ใน CHANGELOG entry เก่า (70/71/72) ที่ชี้ `system_architecture.*`/`usecase_diagram.puml` แบบ root-relative จะไม่ตรงตำแหน่งใหม่แล้ว (เป็นประวัติ ไม่แก้ย้อนหลัง) — ไฟล์จริงอยู่ใน `docs/`; artifact ที่ publish ไปแล้ว (system_architecture.html) ยัง update ผ่าน url เดิมได้ แค่ต้องชี้ path ใหม่ `docs/system_architecture.html`

**ไฟล์ที่แก้**: ย้ายไฟล์ (ไม่แก้เนื้อหา) เข้า `docs/`+`notebooks/`, ลบ `__pycache__`, `CLAUDE.md` (เพิ่ม repo layout + แก้ path notebook)

## 2026-07-22 (72) — System Architecture Diagram เวอร์ชันภาพ (SVG/HTML) แบบโปร่งเน้นไอคอน

**ที่มา**: ต่อจาก (71) — ผู้ใช้ส่งภาพตัวอย่างสไตล์ที่ต้องการ (pictorial, เน้นไอคอนอุปกรณ์/คน/คลาวด์, กล่องน้อย, ข้อความน้อย, ลิงก์มี label สั้นๆ) บอกว่าเวอร์ชัน PlantUML (71) "ดูเยอะไป" — อธิบายให้ผู้ใช้ทราบว่า PlantUML ทำสไตล์ภาพแบบนี้ได้ไม่สวยเท่า SVG/HTML แล้วถามยืนยันผ่าน AskUserQuestion → ผู้ใช้เลือก **Visual SVG/HTML**

**เพิ่ม [system_architecture.html](system_architecture.html)** (ใหม่): หน้า HTML self-contained ที่มี inline SVG เดียว (viewBox 1200×760) วาดสถาปัตยกรรมแบบ layered flow ซ้าย→ขวา — ผู้ใช้ → Vue 3 PWA → Flask (app.py: Dynamic Routing / 3 โมเดล / NLG) → PostgreSQL, + APScheduler (บน), External APIs (Air4Thai/Open-Meteo/Gemini/OpenAQ, ขวา), Offline training (ล่างซ้าย) ไอคอนวาดเป็น SVG symbol (line-style, currentColor), เส้นทึบ = runtime / เส้นประ = external+offline, label สองภาษาบนลิงก์ (HTTPS·REST / psycopg2 / PM2.5 / hourly ฯลฯ), theme-aware (light/dark ผ่าน CSS custom properties + prefers-color-scheme + data-theme), รองรับภาษาไทยด้วย system font stack (Segoe UI/Leelawadee/Noto Sans Thai), เนื้อหากระชับกว่า (71) มาก (6 กล่องหลัก แทน 20+ node)

**เพิ่ม [system_architecture.png](system_architecture.png)** (ใหม่): ภาพ render (light theme, 1240×860) สำหรับแปะในเล่มรายงานได้ทันที

**publish เป็น Artifact**: https://claude.ai/code/artifact/34eba19f-fa13-430e-9adb-3e56030a5cfe (private, ผู้ใช้เปิดดู/สลับ light-dark/ดาวน์โหลดได้)

**ทดสอบ**: render จริงด้วย headless Chrome ทั้ง 2 ธีม (dark = ค่า default จาก prefers-color-scheme, light = force override) — เปิดภาพยืนยัน: layout โปร่งตามภาพตัวอย่าง, ภาษาไทยแสดงถูกทุกจุด, ไอคอน/เส้น/ลูกศร/legend ครบ, ไม่มี element ซ้อนทับ, ทั้งสองธีมอ่านได้ชัด

**หมายเหตุ**: `system_architecture.puml` จาก (71) ยังเก็บไว้เป็นทางเลือก (source แก้ได้/เหมาะเชิงเทคนิค) — เวอร์ชันภาพนี้เป็นตัวที่แนะนำสำหรับใส่เล่ม

**ไฟล์ที่แก้**: `system_architecture.html` (ใหม่), `system_architecture.png` (ใหม่) — ไฟล์เอกสารอย่างเดียว ไม่แตะโค้ด/DB

## 2026-07-22 (71) — เพิ่ม System Architecture Diagram (PlantUML, layered) ประกอบเล่ม

**ที่มา**: ต่อเนื่องจาก (70) — ผู้ใช้ขอ System Architecture Diagram ของโปรเจกต์เพื่อประกอบเล่ม capstone (มี ERD `database_schema.md` + `usecase_diagram.puml` แล้ว แต่ยังไม่มี diagram ระดับสถาปัตยกรรมทางเทคนิค/การไหลข้อมูลระหว่างชั้น) ตกลงร่วมกับผู้ใช้ผ่าน AskUserQuestion: (1) รูปแบบ = PlantUML source, (2) ชนิด = Layered logical (แบ่งชั้น Frontend/Backend/ML/Data/External), (3) ภาษา label = สองภาษา (ไทย + อังกฤษ) — สอดคล้องกับ `usecase_diagram.puml`

**เพิ่ม [system_architecture.puml](system_architecture.puml)** (ใหม่, root เดียวกับ `database_schema.md`/`usecase_diagram.puml`): layered diagram 6 ชั้น — (1) Client/Presentation (Vue 3 SPA + PWA: Views, shared state `auth.js`/`notifications.js`/`riskScore.js`, service worker), (2) Backend/Application (Flask `app.py`: REST routes, `serve_spa`, Flask-session auth, pipeline `get_forecast`, APScheduler, daily NLG + safety check), (3) ML/Inference (Dynamic Routing/circuit breaker, LightGBM/XGBoost/Prophet, calibration layer, threshold-bias correction, SHAP, `compute_risk_score`), (4) Data/Persistence (PostgreSQL 4 ตาราง + `models/*.pkl|*.csv` + `data/*.csv`), (5) External Services (Air4Thai/Open-Meteo/Gemini/OpenAQ), (6) Offline training pipeline (`scripts/`) — เส้นทึบ = runtime flow, เส้นประ = offline/training-time; ลูกศรกำกับ protocol (HTTPS/REST/psycopg2)

**ยืนยันความถูกต้องจากโค้ดจริง** (ไม่ใช่จาก proposal): model loading (`app.py` บรรทัด 99-120), `get_db_connection`/psycopg2, scheduler (1680-1692), `serve_spa` (1287), 4 ตารางจาก `init_db` (287/304/321/333), frontend `frontend/src/views/*` + shared-state modules

**ทดสอบ**: PlantUML jar v1.2024.7 — `-checkonly -failfast2` ผ่าน (exit 0), render PNG สำเร็จ (exit 0), เปิดภาพยืนยัน: ภาษาไทยแสดงถูก, 6 ชั้นครบพร้อมสีแยกชั้น, เส้นทึบ/เส้นประแยก runtime vs offline ถูกต้อง, ลูกศรมี label protocol ครบ

**ไฟล์ที่แก้**: `system_architecture.puml` (ใหม่ — ไฟล์เอกสารอย่างเดียว ไม่แตะโค้ด/DB)

## 2026-07-22 (70) — เพิ่ม Use Case Diagram (PlantUML) ประกอบเล่ม

**ที่มา**: ผู้ใช้ขอ Use Case Diagram ของโปรเจกต์เพื่อประกอบเล่ม/รายงาน capstone — เดิมมีแค่ `database_schema.md` (ERD) ยังไม่มี diagram ระดับ behavior ที่แสดงว่า actor แต่ละประเภททำอะไรกับระบบได้บ้าง ตกลงร่วมกับผู้ใช้ผ่าน AskUserQuestion 3 จุด: (1) รูปแบบ = PlantUML source, (2) actor = เฉพาะที่เป็นคน (Guest + Registered User) ไม่วาด Scheduler/Air4Thai/Open-Meteo/Gemini เป็น actor, (3) ภาษา label = สองภาษา (ไทย + อังกฤษ)

**เพิ่ม [usecase_diagram.puml](usecase_diagram.puml)** (ใหม่, วางที่ root เดียวกับ `database_schema.md`): `@startuml … @enduml` — 2 actor (`Registered User --|> Guest` generalization), system boundary "ระบบพยากรณ์ PM2.5", use case ~18 อัน แบ่งเป็น guest-accessible / registered-only / internal (include-extend) โดย diagram สะท้อน "สิ่งที่ระบบทำได้จริงตอนนี้" ตาม CHANGELOG ล่าสุด — ทุก use case ยืนยันแล้วว่ามี route/ฟังก์ชันจริงรองรับ (`/api/forecast`, `/api/comparison*`, `/api/notifications`, `/api/auth/*`, `get_forecast`, `select_model`, `log_predictions`, `backfill_actuals`, `find_best_time`, `compute_risk_score`, `maybe_generate_daily_nlg`) + view จริงใน `frontend/src/views/` (Forecast/Compare/Login/Register/Profile/Notifications) — Web Push แบบ proactive กำกับด้วย note ชัดเจนว่า "ยังไม่ได้ทำจริง ปัจจุบัน pull-based" ไม่ปนกับของที่ทำแล้ว, บริการภายนอก/งาน scheduler รายชั่วโมงกำกับด้วย `note` ไม่ใช่ actor

**ทดสอบ**: verify ด้วย PlantUML jar v1.2024.7 (ดาวน์โหลดลง scratchpad ชั่วคราว ไม่ commit) — `-checkonly -failfast2` ผ่าน (exit 0 ไม่มี syntax error), render เป็น PNG สำเร็จ (exit 0) เปิดดูภาพจริงยืนยัน: ภาษาไทยแสดงถูกต้อง, actor 2 ตัว + generalization arrow, system boundary, use case ครบ (bilingual), เส้น `<<include>>`/`<<extend>>` และ note ทั้ง 4 อันแสดงครบถูกต้อง

**ไฟล์ที่แก้**: `usecase_diagram.puml` (ใหม่ — ไฟล์เอกสารอย่างเดียว ไม่แตะโค้ด/DB)

## 2026-07-22 (69) — ลบ hourly NLG dead code ทิ้ง + แก้บั๊ก numpy.bool_ ที่เจอระหว่างทดสอบ

**ที่มา**: สำรวจโปรเจกต์หาจุดที่ควรพัฒนาต่อ พบว่า `generate_nlg_explanation()` (NLG ราย*ชั่วโมง*) เป็น dead code ค้างมานาน เขียนสมบูรณ์แต่ไม่เคยถูกเรียกจาก `get_forecast()` เลย (กังวลเรื่อง cost Gemini สูงสุด 24 ครั้ง/รอบพยากรณ์) เปิดอ่าน `proposal.docx` เทียบ requirement จริง (หัวข้อ 4.1.4) พบว่าข้อความในเล่มไม่ได้ระบุว่าต้องเป็นรายชั่วโมง — NLG รายวันที่มีอยู่แล้ว (risk_group-aware ตั้งแต่ CHANGELOG 52) ตอบโจทย์ 4.1.4 ครบถ้วนแล้ว จึงตัดสินใจลบทิ้งแทนที่จะเสียเวลา wiring เข้าจริง

**แก้** (`app.py`):
- ลบ `_NLG_CACHE`, `_build_nlg_prompt()`, `generate_nlg_explanation()` ทั้งหมด (ตรวจ call site ทั่ว repo แล้วว่าไม่มีที่อื่นเรียกใช้ นอกจาก `scripts/test_nlg_generation.py` ที่ลบทิ้งไปด้วย) — ฟังก์ชันที่ daily NLG ยังใช้ร่วม (`get_shap_explanation`, `_format_shap_factor_lines`, `_fallback_message`, `validate_health_message`) **ไม่ได้แตะ**
- ลบ import `datetime` ที่ไม่ได้ใช้แล้ว (เหลือแค่ `timedelta` ที่ยังใช้จริง)
- แก้ comment ล้าสมัย 3 จุดที่อ้างถึง `generate_nlg_explanation` ว่า "ยัง dead code/deferred" (เหนือ `CREATE TABLE users`, เหนือ `VALID_RISK_GROUPS`, และใน docstring ของ `generate_daily_nlg_explanation()`) ให้ตรงกับความจริงว่าถูกลบทิ้งแล้ว

**บั๊กที่เจอระหว่างรันเทส (ไม่เกี่ยวกับงานหลัก แต่แก้ไปด้วยเพราะบล็อกการยืนยันผล)**: `is_current_at_risk()` เทียบ `latest_pm25` (เป็น `numpy.float64` จาก pandas) กับ threshold ตรงๆ ได้ผลลัพธ์เป็น `numpy.bool_` ไม่ใช่ `bool` ปกติ — ใน `get_forecast()` ที่เขียน `is_current_at_risk(...) or len(risk_horizons) > 0` ถ้าฝั่งซ้าย truthy (numpy.bool_) Python's `or` จะคืนค่านั้นตรงๆ โดยไม่แปลงชนิด ทำให้ `jsonify()` พัง (`Object of type bool is not JSON serializable`) — เจอบั๊กนี้เพราะรัน `test_risk_group_e2e.py` ด้วยข้อมูลสดจริงตอนที่ PM2.5 ปัจจุบัน (27.9) ขึ้นมาแตะ effective threshold ของกลุ่มเปราะบางพอดี (27.5) แต่ยังไม่ถึงของกลุ่มทั่วไป (37.5) จึงไม่เคยถูกเทสจับได้มาก่อน แก้ด้วยการห่อ `bool(...)` ให้ `is_current_at_risk()` คืนค่า native bool เสมอ

**ทดสอบ**: `python -c "import ast; ast.parse(...)"` ยืนยัน `app.py` parse สะอาด, รัน `test_daily_nlg.py`/`test_nlg_safety_check.py`/`test_risk_group_e2e.py`/`test_notification_history.py`/`test_best_time_window.py`/`test_risk_score.py`/`test_dynamic_routing.py`/`test_effective_threshold_by_model.py`/`test_auth_routes.py` ผ่านหมด (ก่อนแก้บั๊ก numpy.bool_ ตัว `test_risk_group_e2e.py` fail 4 จุดในกลุ่มเปราะบาง หลังแก้ผ่านหมด) เช็ค DB ไม่มี marker ตกค้าง

**ไฟล์ที่แก้**: `app.py`, ลบ `scripts/test_nlg_generation.py`

## 2026-07-21 (68) — แยก label ระดับความเสี่ยงส่วนบุคคลออกจากระดับ PM2.5 ดิบให้ชัดเจน

**ที่มา**: ผู้ใช้สงสัยว่าคะแนนความเสี่ยงส่วนบุคคล (CHANGELOG 62) ทำงานถูกต้องไหม เพราะ chip ระดับในการ์ดนี้ใช้ชื่อชุดเดียวกับ chip ระดับ PM2.5 ดิบในการ์ดปัจจุบันด้านบน (มาจาก `PM25_LEVELS` เดิม ใช้ `statusClass()` ร่วมกัน) ทำให้ดูเหมือนขัดแย้งกัน (เช่น อากาศจริง "ดี" แต่ความเสี่ยงส่วนบุคคล "ปานกลาง") ทั้งที่เป็นคนละแกนกัน (อากาศตอนนี้ vs ความเสี่ยงเฉพาะบุคคล+กิจกรรม) — ตรวจสอบร่วมกับผู้ใช้แล้วว่าเลขคำนวณถูกต้องตามสูตรทุกประการ ปัญหาอยู่ที่การนำเสนอ ไม่ใช่ตัวคะแนน

**แก้** (`frontend/src/views/ForecastView.vue`, frontend-only, ไม่แตะค่า `level`/logic คำนวณใดๆ): เพิ่ม label เล็กๆ "ความเสี่ยงของคุณ" ไว้เหนือ chip ระดับในการ์ด Personalized Risk Score (ไม่ยัดคำนำหน้าเข้าไปในคำเดียวกับ chip โดยตรง เพราะ chip ปัจจุบันออกแบบมาสำหรับคำสั้นๆ ยัดประโยคยาวเข้าไปเสี่ยงล้น/ตัดคำบนจอเล็ก — ใช้ wrapper `.risk-level-tag` แยก label ไว้ข้างนอก pill แทน) ทำให้แยกความหมายจากการ์ด PM2.5 ปัจจุบันชัดเจนขึ้นโดยไม่กระทบ `statusClass()`/สี/ค่า `level` ที่ใช้ในการทดสอบ

**ทดสอบ**: `npm run build` clean, Selenium ยืนยันด้วยเคสระดับยาวที่สุด ("เริ่มมีผลกระทบต่อสุขภาพ") บนจอมือถือ (430px) ไม่มีข้อความล้น/ตัดคำ และค่าคะแนน/สี/ระดับยังตรงตามสูตรเดิมทุกกิจกรรม (regression ผ่านหมด)

**ไฟล์ที่แก้**: `frontend/src/views/ForecastView.vue`

## 2026-07-21 (67) — จัดวางเลย์เอาต์การ์ดหน้า Forecast ใหม่ให้สวยงามและสมดุล

**ที่มา**: หลังแก้บั๊ก layout ใน CHANGELOG 66 (`align-items: start`) การ์ด PM2.5 หลักไม่มีช่องว่างผิดปกติแล้ว แต่ผู้ใช้ชี้ว่าโดยรวมยังจัดวางไม่สวยงาม — เพราะคอลัมน์หลัก (การ์ด PM2.5) สั้นกว่าคอลัมน์ข้าง (3 การ์ดซ้อนกัน: สรุปวันนี้/ช่วงเวลาที่ดีที่สุด/คะแนนความเสี่ยง) มาก เกิดเป็นพื้นที่ว่างเปล่าด้านล่างคอลัมน์หลักก่อนถึงกราฟ

**แก้** (`frontend/src/views/ForecastView.vue`, CSS ล้วน ไม่แตะ template/script): เปลี่ยนจากเลย์เอาต์ 2 คอลัมน์ (หลัก 1.6fr ซ้าย / รอง 1fr ขวาซ้อน 3 การ์ด) เป็นโครงสร้างแนวตั้งใหม่ 3 ชั้น:
1. การ์ด PM2.5 ปัจจุบันเต็มความกว้าง — ปรับ internal layout จากเรียงซ้อนแนวตั้ง (ค่า PM2.5 บน / สภาพอากาศล่าง คั่นด้วยเส้นแนวนอน) เป็นวางเคียงข้างกัน (ค่า PM2.5 ซ้าย / สภาพอากาศขวา คั่นด้วยเส้นแนวตั้ง) ใช้ความกว้างที่เพิ่มขึ้นให้เป็นประโยชน์แทนที่จะเหลือที่ว่าง
2. แถวการ์ดรอง 3 ใบเท่ากัน (สรุปวันนี้ / ช่วงเวลาที่ดีที่สุด / คะแนนความเสี่ยงส่วนบุคคล) เป็น grid 3 คอลัมน์เท่ากัน
3. กราฟ 24 ชม. เต็มความกว้างเหมือนเดิม

Responsive: จอเล็กกว่า 880px กลับไปเป็นเลย์เอาต์เดิม (สภาพอากาศเรียงล่างมีเส้นคั่นแนวนอน, การ์ดรองเรียงซ้อนคอลัมน์เดียว) เหมือนก่อนแก้ทุกประการ

**ทดสอบ**: `npm run build` clean, Selenium screenshot ยืนยันทั้ง 3 breakpoint (desktop 1200px, mobile 430px, dark mode) จัดวางถูกต้องสมดุล ไม่มีช่องว่างผิดปกติเหลืออยู่

**ไฟล์ที่แก้**: `frontend/src/views/ForecastView.vue`

## 2026-07-21 (66) — แก้บั๊ก layout การ์ด PM2.5 ปัจจุบันมีช่องว่างเยอะผิดปกติ

**ที่มา**: ผู้ใช้ส่ง screenshot ชี้ว่าการ์ด "PM2.5 ล่าสุดรายชั่วโมง" มีช่องว่างเปล่าๆ ด้านบนก้อนใหญ่หลังจากเพิ่มการ์ด Personalized Risk Score (CHANGELOG 62) — root cause: `.dashboard-grid` มี `align-items: stretch` บังคับให้ทั้ง 2 คอลัมน์สูงเท่ากัน เดิมคอลัมน์ข้าง (`.grid-col-side`) มีแค่ 2 การ์ด สูงพอๆ กับการ์ดหลัก แต่หลังเพิ่มการ์ดที่ 3 (risk score) คอลัมน์ข้างสูงขึ้นมาก การ์ดหลัก (`.grid-col-main`) เลยถูก stretch สูงตาม แล้ว `.current-info { flex:1; justify-content:center }` ก็ดันเนื้อหาไปกึ่งกลางของพื้นที่ที่สูงขึ้นนั้น เกิดเป็นช่องว่างก้อนใหญ่ด้านบน

**แก้** (`frontend/src/views/ForecastView.vue`): เปลี่ยน `.dashboard-grid` จาก `align-items: stretch` เป็น `align-items: start` — แต่ละคอลัมน์สูงตามเนื้อหาตัวเองอิสระจากกัน ไม่บังคับให้เท่ากันอีกต่อไป

**ทดสอบ**: `npm run build` clean, Selenium ยืนยัน `.grid-col-main` สูงตามเนื้อหาจริง (~263px ไม่ใช่ยืดตาม `.grid-col-side` ที่ ~684px) และช่องว่างเหนือค่า PM2.5 กลับมาเป็นค่าปกติ (~49px, เท่า padding การ์ดทั่วไป ไม่ใช่ช่องว่างขนาดใหญ่ผิดปกติ)

**ไฟล์ที่แก้**: `frontend/src/views/ForecastView.vue`

## 2026-07-21 (65) — Redesign หน้า /compare ให้เข้าใจง่าย (progressive disclosure)

**ที่มา**: ผู้ใช้ขอ redesign หน้า "เปรียบเทียบผลพยากรณ์กับค่าจริง" ให้ดูเข้าใจง่ายกว่าเดิม โครงสร้างเดิมมีทั้ง "ภาพรวม" และ "รายละเอียดต่อ horizon/โมเดล" อยู่คนละ card แต่พูดเรื่องเดียวกันซ้ำ 2 รอบ และมีศัพท์เทคนิค (MAE, Bias, backfill/live, ชื่อโมเดล LightGBM/XGBoost/Prophet) ปนอยู่ทั่วหน้าโดยไม่มีลำดับความสำคัญ ปรึกษาผู้ใช้ผ่าน AskUserQuestion 2 จุด: (1) ปรับโครงสร้างใหม่ทั้งหมด ไม่ใช่แค่ปรับผิว, (2) ซ่อนชื่อโมเดลไว้ในส่วนรายละเอียดขั้นสูงเท่านั้น

**แก้** (`frontend/src/views/CompareView.vue`, frontend-only เขียนใหม่เกือบทั้งไฟล์ — ไม่แตะ `/api/comparison`/`/api/comparison/overview` เลย):
- โครงสร้างใหม่แบบ progressive disclosure: (1) การ์ด **"คำตอบหลัก"** ใหม่ — ไอคอน + ข้อความคุณภาพตัวใหญ่ (แม่นยำมาก/ดี/ปานกลาง/คลาดเคลื่อนค่อนข้างมาก, ไอคอนจาก `@lucide/vue` CircleCheck/CircleAlert/CircleX — คนละชุดกับไอคอนหน้ายิ้มของ ForecastView โดยตั้งใจ กันสับสนว่าเป็นสถานะอากาศ) + ประโยคอธิบายสั้น 1 ย่อหน้า reuse `qualityForMae()`/`biasPhrase()` เดิมทั้งหมด (2) การ์ดแนวโน้มตามชั่วโมงพยากรณ์ (คง mini-bar chart เดิม เพิ่ม legend สีแบบ text ชัดๆ แทนพึ่ง hover tooltip อย่างเดียว) (3) การ์ดตัวอย่างกราฟค่าพยากรณ์เทียบค่าจริง (เหลือแค่ horizon selector, เอา model selector ออก) (4) ปุ่ม toggle **"ดูรายละเอียดทางเทคนิค"** (ใหม่ `showAdvanced` ref, ปิดเป็นค่าเริ่มต้น) ซ่อน stats-row (MAE/Bias ตัวเลขดิบ), model-breakdown chips (LightGBM/XGBoost/Prophet), และ model selector ไว้ทั้งหมด — ไม่ได้ลบ logic ใดๆ ทิ้ง แค่ย้ายตำแหน่งให้ผู้ใช้ทั่วไปไม่ต้องเห็นโดยไม่ตั้งใจ
- `drawChart()`: จุดค่าพยากรณ์เปลี่ยนจากสีตามโมเดล (`modelHex(p.model_used)`) + วงกลมกลวง/ทึบ (backfill/live) เป็นสีน้ำเงินเดียวกันหมด ไม่ encode โมเดล/แหล่งข้อมูลด้วยภาพอีกต่อไป — ข้อมูล `model_used`/`source` ยังอยู่ใน `<title>` tooltip เดิม (hover ดูได้ ไม่ได้ทิ้งข้อมูล แค่ไม่บังคับทุกคนต้องแปลรหัสสี) legend เหลือ 2 รายการ: ค่าพยากรณ์/ค่าจริง

**ทดสอบ**: `npm run build` clean, Selenium ยืนยัน (1) หน้า default ไม่มีชื่อ LightGBM/XGBoost/Prophet โผล่เลย (2) กดปุ่ม advanced แล้วเห็นชื่อโมเดล+ตัวเลขดิบครบ (3) horizon selector ยังทำงานถูกต้องเหมือนเดิม (4) dark mode: ตรวจ computed style ยืนยัน `--card-bg`/`--text` สลับเป็นค่า dark ถูกต้อง (พื้นหลังท้องฟ้าไม่เปลี่ยนตาม dark mode — เป็นพฤติกรรมเดิมของหน้านี้ตั้งแต่ก่อน redesign ไม่ใช่ regression ที่เกิดใหม่)

**ไฟล์ที่แก้**: `frontend/src/views/CompareView.vue`

## 2026-07-21 (64) — เพิ่มคำอธิบายปุ่มเลือกกิจกรรมในการ์ด Personalized Risk Score

**ที่มา**: ผู้ใช้ชี้ต่อจาก CHANGELOG 63 ว่าหัวข้อการ์ด "🎯 คะแนนความเสี่ยงส่วนบุคคล" ไม่ได้บอกอะไรเลยว่าปุ่ม เบา/ปานกลาง/หนัก ที่อยู่ข้างล่างมีไว้ทำอะไร (เลือกกิจกรรม? เลือกความรุนแรงอย่างอื่น?) — ยังเป็นช่องว่างเดิมที่ CHANGELOG 63 ยังไม่ได้แก้ (ตอนนั้นแก้แค่ตัวเลขคะแนน/ระดับ ไม่ได้แตะปุ่มเลือกกิจกรรม)

**แก้** (`frontend/src/views/ForecastView.vue`): เพิ่มบรรทัดคำอธิบายสั้นๆ "เลือกความหนักของกิจกรรมกลางแจ้งที่วางแผนจะทำ" คั่นระหว่างหัวข้อการ์ดกับปุ่มเลือกกิจกรรม 3 ปุ่ม (class `.activity-select-caption`, ใช้ `.label-muted` เดิมเป็นฐาน)

**ทดสอบ**: `npm run build` clean, Selenium ยืนยัน caption ใหม่แสดงถูกตำแหน่ง และคะแนน/ระดับ/สี/localStorage ยังทำงานถูกต้องเหมือนเดิมทุกอย่าง (ไม่มี regression)

**ไฟล์ที่แก้**: `frontend/src/views/ForecastView.vue`

## 2026-07-21 (63) — ปรับการ์ด Personalized Risk Score ให้เข้าใจง่ายขึ้น

**ที่มา**: ผู้ใช้ส่ง screenshot การ์ดคะแนนความเสี่ยง (CHANGELOG 62) ชี้ว่าตัวเลข "8" ลอยเดี่ยวๆ ข้าง chip ระดับ ไม่มี context ว่าคะแนนนี้มาจากอะไร เต็ม 100 หรือเปล่า ยิ่งสูงยิ่งดีหรือยิ่งแย่

**แก้** (`frontend/src/riskScore.js`, `frontend/src/views/ForecastView.vue`):
- เพิ่ม caption อธิบายทิศทางคะแนนไว้เหนือตัวเลข: "คะแนนความเสี่ยงสำหรับกิจกรรมนี้ (0 = เสี่ยงน้อยที่สุด, 100 = เสี่ยงมากที่สุด)"
- เพิ่ม unit "/100" ต่อท้ายตัวเลข (pattern เดียวกับ `pm-value`/`pm-unit` ของการ์ด PM2.5 ปัจจุบัน)
- เพิ่มฟังก์ชันใหม่ `riskScoreAdvice(level)` แปลระดับ (ดีมาก/ดี/ปานกลาง/เริ่มมีผลกระทบต่อสุขภาพ/มีผลกระทบต่อสุขภาพ) เป็นประโยคแนะนำสั้นๆ ว่าควรทำอย่างไรกับกิจกรรมที่เลือก (เช่น "ความเสี่ยงปานกลาง ทำได้แต่ควรสังเกตอาการตนเอง") แสดงต่อท้ายตัวเลข+chip ทันที

**ทดสอบ**: `npm run build` clean, Selenium ยืนยัน caption/unit/advice แสดงถูกต้องและเปลี่ยนตามกิจกรรมที่เลือก (คะแนน/ระดับ/สี/ข้อความยังตรงตามที่ทดสอบไว้ใน CHANGELOG 62 ทุกอย่าง ไม่มี regression)

**ไฟล์ที่แก้**: `frontend/src/riskScore.js`, `frontend/src/views/ForecastView.vue`

## 2026-07-21 (62) — Personalized Risk Score (คะแนนความเสี่ยงเฉพาะบุคคล)

**ที่มา**: เล่ม proposal ของผู้ใช้ระบุให้พัฒนาระบบให้คะแนนความเสี่ยงเฉพาะบุคคล โดยผสานผลพยากรณ์เข้ากับ 2 ปัจจัย — กลุ่มความเสี่ยงผู้ใช้ (ทั่วไป/เด็ก-ผู้สูงอายุ/มีโรคประจำตัว) และความหนักของกิจกรรมกลางแจ้งที่ต้องการทำ — เป็นคะแนนเดียว ตรวจสอบพบว่า `risk_group` ผูกอยู่แล้ว (CHANGELOG 52) แต่เป็นแค่ threshold offset ไม่ใช่คะแนน และปัจจัยความหนักกิจกรรมไม่มีอยู่ในระบบเลย (ไม่มี field, ไม่มี UI, ไม่ถูกใช้งานที่ไหน) — เข้า Plan Mode ออกแบบระบบใหม่ทั้งหมด ตัดสินใจร่วมกับผู้ใช้ผ่าน AskUserQuestion 3 จุด: (1) แบ่งกิจกรรม 3 ระดับ เบา/ปานกลาง/หนัก สมมาตรกับ risk_group, (2) คะแนนครอบคลุมทั้งค่าปัจจุบัน+แนวโน้ม 24 ชม. ข้างหน้า ไม่ใช่แค่ค่าปัจจุบัน, (3) เลือกกิจกรรมสดหน้าเว็บ+จำผ่าน localStorage ไม่ผูก backend/schema

**แก้**:
- `app.py`: เพิ่ม `ACTIVITY_INTENSITY_OFFSET = {"เบา": 0.0, "ปานกลาง": -5.0, "หนัก": -10.0}` (sign convention เดียวกับ `RISK_GROUP_THRESHOLD_OFFSET` ที่มีอยู่แล้ว — กิจกรรมหนักขึ้น = อัตราการหายใจสูงขึ้น = สูดฝุ่นมากขึ้นที่ความเข้มข้นเท่าเดิม จำลองเป็นการลด effective PM2.5) และ `RISK_SCORE_BREAKPOINTS` (6 จุด อิงตาม `PM25_LEVELS` มาตรฐาน คพ. 2566 ที่มีอยู่แล้ว ให้แต่ละระดับเท่ากับ 20 คะแนนพอดี, เพดาน interpolation ที่ 150→100)
- ฟังก์ชันใหม่ `compute_risk_score(pm25, risk_group, activity)` (pure function, วางข้าง `find_best_time`/`is_horizon_at_risk`) คำนวณ effective PM2.5 → interpolate เป็นคะแนน 0-100 เชิงเส้น → classify level ผ่าน `_classify_pm25_level()` เดิม (ไม่มี logic สีใหม่ เพราะ level ที่ได้ map ตรงกับ CSS severity token ที่ ForecastView.vue มีอยู่แล้ว)
- `get_forecast()`: เพิ่ม key `risk_score_config` ในผลลัพธ์ (ไม่ใช่คะแนนสำเร็จรูป เพราะ activity เลือกฝั่ง frontend เท่านั้น) ส่ง `risk_group_offset`/`activity_offsets`/`breakpoints` ให้ frontend คำนวณเองจาก pm25 ที่มีอยู่แล้วในผลลัพธ์เดียวกัน — ตัวเลขคงที่ (offset/breakpoint) นิยามที่ Python จุดเดียว กัน frontend/backend คำนวณไม่ตรงกัน ไม่มี schema/DB เปลี่ยนแปลง (activity ไม่ persist ที่ backend เลย)
- `frontend/src/riskScore.js` (ใหม่): reactive singleton pattern เดียวกับ `auth.js`/`notifications.js` — `riskScoreState.activity` จำค่าผ่าน `localStorage` (`pm25_activity_intensity`), `computeRiskScore()` เป็น mirror ของฟังก์ชัน backend (คำนวณจาก config ที่ backend ส่งมา ไม่ hardcode ตัวเลขซ้ำ มีแค่สูตร interpolation ที่เป็น math ทั่วไปซ้ำ ไม่เสี่ยง drift)
- `frontend/src/views/ForecastView.vue`: การ์ดใหม่ "🎯 คะแนนความเสี่ยงส่วนบุคคล" ใน `.grid-col-side` — ปุ่มเลือกกิจกรรม 3 ปุ่ม, คะแนน+ระดับ (reuse `statusClass()`/severity token เดิม), รายการ "ช่วงเวลาที่เหมาะกับกิจกรรมนี้มากที่สุด" (เหมือน `topBestTimes` แต่ sort ด้วยคะแนนแทน pm25 ดิบ personalize ตามกิจกรรมที่เลือก)

**ทดสอบ**:
- ใหม่ `scripts/test_risk_score.py` (8 tests, ไม่ต้องต่อ DB สำหรับ pure-function tests): breakpoint 6 จุดให้คะแนน 0/20/40/60/80/100 ตรงเป๊ะ (ทั่วไป+เบา), clamp ที่ 100 เมื่อ pm25 เกิน 150, monotonic ไม่ลดลง, risk_group offset ทำคะแนนสูงขึ้นสำหรับกลุ่มเปราะบาง, activity offset ทำคะแนนสูงขึ้นสำหรับกิจกรรมหนัก, offset ทั้งสองตัว stack กันถูกต้อง, level ตรงกับ `_classify_pm25_level(effective_pm25)` ไม่มี drift, `/api/forecast` ส่ง `risk_score_config` ตรงกับ risk_group ของ session จริง (login จริงผ่าน `/api/auth/register` เหมือน `test_risk_group_e2e.py`) — ผ่านหมด
- Regression: `test_risk_group_e2e.py`, `test_notification_history.py`, `test_best_time_window.py` ผ่านหมด ยืนยัน field เดิมไม่เปลี่ยน
- Selenium: mock `/api/forecast` (pm25=30 ให้ทั้ง 3 กิจกรรมข้าม breakpoint คนละช่วง) คลิกปุ่มครบ 3 ปุ่ม ยืนยันคะแนน/ระดับ/สีตรงตามคำนวณเป๊ะ (เบา=48 ปานกลาง, ปานกลาง=56 ปานกลาง, หนัก=61 เริ่มมีผลกระทบต่อสุขภาพ สีเปลี่ยนเหลือง→ส้มถูกต้อง), reload หน้าใหม่ยืนยัน `localStorage` จำค่ากิจกรรมล่าสุด ("หนัก") ถูกต้อง
- `npm run build` clean ไม่มี warning, DB ไม่มี marker ตกค้างหลังรันทุกเทส

**ไฟล์ที่แก้**: `app.py`, `frontend/src/riskScore.js` (ใหม่), `frontend/src/views/ForecastView.vue`, `scripts/test_risk_score.py` (ใหม่)

## 2026-07-20 (61) — Best Time Window Finder กรองเฉพาะ 06:00-20:00 เวลาไทย

**ที่มา**: ผู้ใช้ชี้ว่า `best_time`/`topBestTimes` เดิมหาจุด PM2.5 ต่ำสุดจากทั้ง 24 ชั่วโมง รวมช่วงดึก (เช่น ตี 3) ที่ PM2.5 มักต่ำตามธรรมชาติแต่ไม่มีประโยชน์ในทางปฏิบัติ เพราะไม่มีใครออกไปทำกิจกรรมกลางแจ้งตอนนั้น ต้องการให้กรองเฉพาะ 06:00-20:00 ก่อนหาช่วงที่ดีที่สุด

**แก้**:
- `app.py`: เพิ่ม field `hour_thai` (ชั่วโมงเวลาไทย, int) ให้แต่ละจุดใน `forecast[]` ตอน build (คำนวณจาก `forecast_time_th` ที่มีอยู่แล้ว ไม่ต้องคำนวณใหม่)
- เพิ่ม `find_best_time(forecast)` helper (วางข้าง `is_horizon_at_risk`/`is_current_at_risk` เพื่อให้ import/เทสได้ตรงๆ โดยไม่ต้องเรียก `get_forecast()` เต็มรูปแบบ) — กรอง `ACTIVE_HOURS_START_TH=6` ถึง `ACTIVE_HOURS_END_TH=20` (รวมทั้งสองขอบ) ก่อนหา min(pm25); fallback ไป forecast ทั้งหมดเฉพาะกรณี active hours ว่างเปล่าจริงๆ (ไม่ควรเกิดในทางปฏิบัติเพราะ forecast มีครบ 24 จุดเสมอ)
- `get_forecast()`: เปลี่ยนจาก `min(forecast, key=...)` ตรงๆ เป็นเรียก `find_best_time(forecast)`
- `frontend/src/views/ForecastView.vue`: `topBestTimes` (top-3 การ์ดที่แสดงบนหน้าเว็บ) กรอง `hour_thai` 6-20 ก่อน sort ด้วย logic เดียวกัน (ใช้ `hour_thai` ที่ backend ส่งมาให้แล้ว ไม่ต้อง parse string "time" เอง)

**ทดสอบ**:
- ใหม่ `scripts/test_best_time_window.py` (unit test เรียก `find_best_time()` ตรงๆ ผ่าน synthetic forecast list, ไม่ต้องต่อ DB/network): (1) pm25 ต่ำสุดจริงอยู่ตี 3 แต่ในช่วงกลางวันต่ำสุดคือ 14:00 → ต้องเลือก 14:00 ไม่ใช่ตี 3 — ผ่าน; (2) edge case ทุกจุดในช่วงกลางวันค่าสูงหมด (>50) แต่ตี 3 ต่ำกว่ามาก → ต้องยัง fallback แนะนำช่วงกลางวันที่ดีที่สุดเท่าที่มี ไม่กระโดดไปตี 3 — ผ่าน; (3) active_hours ว่างเปล่าจริง (ทุกจุดนอกช่วง) → fallback ไป forecast ทั้งหมด — ผ่าน; (4) ขอบเขต 06:00/20:00 ต้อง inclusive ทั้งสองด้าน — ผ่าน
- Selenium: mock `/api/forecast` ด้วย forecast synthetic เดียวกับเคส (1) (pm25=5 ตอนตี 3, pm25=15 ตอนเที่ยงบ่าย 14:00, ค่าอื่นในช่วงกลางวัน=30) → หน้าเว็บส่วน "ช่วงเวลาที่อากาศดีที่สุดในอีก 24 ชม." แสดงอันดับ 1 เป็น 14:00 (15 µg/m³) ถูกต้อง ไม่แสดงตี 3 เลยแม้ค่าต่ำกว่ามาก (screenshot ยืนยัน bar chart แสดงตี 3=5 อยู่จริงในกราฟ 24 ชม. แต่ไม่ถูกเลือกในการ์ด best-time)
- Regression: `scripts/test_risk_group_e2e.py` และ `scripts/test_notification_history.py` (ทั้งคู่เรียก `get_forecast()`/`/api/forecast` จริง) ผ่านหมด — ยืนยัน field อื่นใน response ไม่เปลี่ยนแปลง
- `npm run build` — clean, ไม่มี warning
- เช็ค DB หลังรันทุกเทส: ไม่มีแถว marker (`2100-`/`2199-`) ตกค้างในตารางไหนเลย

**ไฟล์ที่แก้**: `app.py`, `frontend/src/views/ForecastView.vue`, `scripts/test_best_time_window.py` (ใหม่), `CLAUDE.md` (แก้บรรทัดที่บอกว่าไม่มี `requirements.txt` — พบว่ามีอยู่แล้วจริง เป็น comment ล้าสมัยอีกจุดที่เจอระหว่างคุยกับผู้ใช้)

## 2026-07-20 (60) — ตรวจสอบทั้งโปรเจกต์ (health check) + แก้ comment ล้าสมัย 2 จุด

**ที่มา**: ผู้ใช้ขอให้ตรวจทั้งโปรเจกต์ว่ามีปัญหาตรงไหนไหม — รันทุก test script, build frontend, เช็ค DB state, หา comment/dead code ที่ล้าสมัย

**ผลตรวจสอบ**:
- Backend: รัน `test_auth_routes.py`, `test_dynamic_routing.py`, `test_effective_threshold_by_model.py`, `test_risk_group_e2e.py`, `test_notification_history.py`, `test_nlg_safety_check.py`, `test_daily_nlg.py` — **ผ่านทั้งหมด** `app.py` โหลดสำเร็จ ไม่มี syntax error (`ast.parse` ผ่าน) 15 route ลงทะเบียนครบถูกต้อง
- DB: ไม่มีแถว marker (`2100-`/`2199-`/`@test-marker.invalid`) ตกค้างในตารางไหนเลย (`prediction_log`/`daily_nlg_log`/`notification_log`/`users`) row count โดยรวมดูสมเหตุสมผล
- Frontend: `npm run build` ผ่านสะอาด ไม่มี unused-import warning ตรวจแล้วว่า `buildWarningMessage`/`WARNING_ADVICE_BY_RISK_GROUP` นิยามอยู่จุดเดียวใน `notifications.js` ไม่มี logic ซ้ำ, ไม่มี reference เหลือค้างของ `API_URL` ที่ลบไปตอน (57), ไม่มี `console.log`/`debugger` หลงเหลือ
- **พบ comment ล้าสมัย 2 จุดใน `app.py`** (บรรทัดเหนือ `CREATE TABLE users` และเหนือ `VALID_RISK_GROUPS`) ที่ยังเขียนว่า "risk_group ยังไม่ผูกกับ NLG (deferred)" ทั้งที่ผูกไปแล้วจริงตั้งแต่ (52)/(53) — **แก้แล้ว** ให้ระบุชัดว่าผูกกับ NLG รายวัน/threshold แล้ว เหลือแค่ NLG ราย*ชั่วโมง* (`generate_nlg_explanation`) ที่ยังเป็น dead code ตามเดิม (คนละเรื่องกับที่ผูกไปแล้ว)
- พบ background dev server (`python app.py`/`vite`) ที่เปิดค้างจากการทดสอบ session ก่อนหน้าหลายตัว — ปิดหมดแล้ว (มี stale entry ใน `netstat` ของ Windows ที่ยังโชว์ PID เดิมค้างอยู่แม้ process จะถูกฆ่าไปแล้วจริง ยืนยันด้วย `Get-Process`/`curl` ว่าไม่มี process หรือ server ทำงานจริง — เป็นแค่ kernel table ที่ยังไม่เคลียร์ ไม่กระทบอะไร)

**สรุป**: โปรเจกต์อยู่ในสภาพดี ไม่พบบั๊ก ไม่มี regression แก้แค่ comment ล้าสมัย 2 จุด ไม่มีการเปลี่ยน logic ใดๆ

**ไฟล์ที่แก้**: `app.py` (comment เท่านั้น)

---

## 2026-07-20 (59) — แก้พื้นหลังหน้า /notifications ให้เหมือนหน้าตระกูล auth

**ที่มา**: ผู้ใช้สังเกตว่าพื้นหลังหน้า "ประวัติการแจ้งเตือนทั้งหมด" (`/notifications`, เพิ่งสร้างใน (58)) ไม่เหมือนหน้าอื่น — ตอนสร้างครั้งแรกไม่ได้ตั้ง background ให้ `.notifications-page` เลย (โปร่งใสล้วน อาศัย browser default)

**แก้ [frontend/src/views/NotificationsView.vue](frontend/src/views/NotificationsView.vue)**: เพิ่ม `background: var(--auth-bg)` + glowing orb effect (`::before` radial-gradient) ให้ `.notifications-page` — คัดลอก pattern เดียวกับ `.auth-page` ใน `LoginView.vue`/`RegisterView.vue`/`ProfileView.vue` เป๊ะ (หน้าตระกูลเดียวกัน ใช้ `--auth-*` tokens/`.glass-card` อยู่แล้วสำหรับการ์ดด้านใน แค่ตัว page container ไม่เคยตั้ง background) เพิ่ม `position: relative; z-index: 1;` ให้ `.wrap` กันเนื้อหาไปอยู่หลัง orb

**ทดสอบ**: `npm run build` ผ่าน ตรวจ computed `background-color` ของ `.notifications-page` ตรงกับ `--auth-bg` เป๊ะทั้ง 2 โหมด (light `rgb(248,250,252)`, dark `rgb(15,23,42)`, ยืนยันผ่าน CDP `Emulation.setEmulatedMedia`) screenshot เทียบกับ `/login` โดยตรง — พื้นหลัง/glow orb เหมือนกันทั้ง 2 โหมดแล้ว

**ไฟล์ที่แก้**: `frontend/src/views/NotificationsView.vue`

---

## 2026-07-20 (58) — เพิ่มประวัติการแจ้งเตือน (notification_log + /api/notifications + หน้า /notifications)

**ที่มา**: ผู้ใช้ถามหลังได้กระดิ่ง+dropdown (57) ว่ามีประวัติย้อนหลังหรือยัง — คุยกันถึงผลกระทบ (ต้องแตะ schema/backend จริง ต่างจาก 57 ที่เป็น frontend ล้วน) แล้วเข้า plan mode ก่อนเริ่ม

**การออกแบบที่ตัดสินใจไว้ก่อนเริ่ม (อธิบายไว้ใน context ของแผน)**:
1. **ไม่เก็บ `risk_horizons` ตรงๆ** — เก็บแค่ `latest_pm25` (ค่าเดียวที่ไม่มีที่เก็บอยู่แล้ว) ส่วน risk ต่อ horizon คำนวณสดตอน query โดย join กับ `prediction_log` ที่มีข้อมูลนี้อยู่แล้ว เหตุผล: risk ขึ้นกับ `risk_group` (ดู (52)) เก็บสดแบบเดียวจะผิดสำหรับกลุ่มอื่น เก็บ 3 ชุดจะซ้ำซ้อนและ desync ได้ถ้า threshold เปลี่ยนในอนาคต
2. **ไม่ special-case "log เฉพาะจาก scheduled job"** — เรียกจาก `get_forecast()` ตรงๆ ทุกครั้งเหมือน `log_predictions()`/`backfill_actuals()` ที่มีอยู่แล้ว อาศัย `UNIQUE(recorded_at_utc)` + `ON CONFLICT DO NOTHING` dedupe เอง (เพราะ `latest_time` เปลี่ยนแค่ตอน Air4Thai มีข้อมูลใหม่ ราวชั่วโมงละครั้งเท่านั้น — pattern เดียวกับที่ `prediction_log` ใช้อยู่แล้ว)

**เพิ่ม [app.py](app.py)**:
- ตาราง `notification_log(id, recorded_at_utc TEXT UNIQUE, latest_pm25 DOUBLE PRECISION)` ใน `init_db()` — ตารางใหม่ล้วน ไม่ต้อง ALTER
- `log_notification_snapshot()` (mirror `log_predictions()`) เรียกจาก `get_forecast()` คู่กับ `log_predictions(latest_time, log_rows)`
- แยก `is_horizon_at_risk()`/`is_current_at_risk()` ออกจาก logic เดิมที่ inline อยู่ใน `get_forecast()` ให้ `get_forecast()` และ route ใหม่ใช้ตรรกะ "อะไรถือว่าเสี่ยง" ชุดเดียวกันเป๊ะ (behavior-preserving refactor — ยืนยันด้วย `test_risk_group_e2e.py` เดิมผ่านหมดไม่มี regression)
- แยก `resolve_risk_group_from_session()` ออกจาก `api_forecast()` ให้ route ใหม่ใช้ร่วมได้ (DRY)
- Route ใหม่ `GET /api/notifications?limit=` — join `notification_log` กับ `prediction_log` ต่อแถว คืนเฉพาะรอบที่ `high_pollution_risk=true` จริงตาม risk_group ของผู้เรียก (คำนวณสด ไม่ผูก login)

**ทดสอบ**: เพิ่ม `scripts/test_notification_history.py` (ใหม่) — สร้างข้อมูลสังเคราะห์ 3 รอบ (ชายขอบ/เงียบสนิท/เกิน threshold ทั่วไปด้วย) ยืนยัน: กลุ่มเปราะบางเห็นรอบชายขอบที่ "ทั่วไป" ไม่เห็น (offset -10 ทำงานถูกต้อง), รอบเงียบไม่ปรากฏกับใครเลย, รอบที่เกิน threshold ทั่วไปเห็นทุกกลุ่ม, `prophet_horizons` ถูกต้องแม้ horizon นั้นเองไม่ใช่ตัวกระตุ้น risk, `limit` param ทำงาน — **ผ่านทั้งหมด** ลบ marker rows หมดยืนยันแล้ว (`notification_log`/`prediction_log`/`users` = 0) รัน `test_risk_group_e2e.py` ซ้ำยืนยันไม่มี regression จากการ refactor

**เพิ่ม frontend**: `frontend/src/notifications.js` เพิ่ม `notificationHistory`/`fetchNotificationHistory()`/`summarizeHistoryItem()` (ดึงตอนเปิด dropdown เท่านั้น ไม่ prewarm ตอน mount เหมือน `fetchForecast()`) `AppHeader.vue` dropdown เพิ่ม section "ประวัติล่าสุด" (5 รายการ, scroll ได้) ลิงก์ "ดูทั้งหมด →" ไปหน้าใหม่ `/notifications` (`frontend/src/views/NotificationsView.vue`, route ใหม่ไม่บังคับ login เหมือน `/`/`/compare`) แสดงเต็ม 50 รายการ

**ทดสอบ frontend**: `npm run build` ผ่าน ทดสอบจริงกับ backend จริง (`curl /api/notifications` คืน `[]` ถูกต้อง เพราะวันนี้อากาศดีไม่มี alert จริงสะสม) ทดสอบ UI ด้วย Selenium mock ทั้ง `/api/forecast`/`/api/notifications` — dropdown history section render ครบ 3 รายการถูกต้อง, ลิงก์ "ดูทั้งหมด" นำไปหน้า `/notifications` จริง, หน้าเต็มแสดงครบเหมือนกัน

**สรุป**: มีประวัติการแจ้งเตือนจริงแล้ว ไม่ซ้ำซ้อนกับ `prediction_log` เดิม ถูกต้องตาม risk_group เสมอแม้ threshold จะเปลี่ยนในอนาคต

**ไฟล์ที่แก้**: `app.py`, `scripts/test_notification_history.py` (ใหม่), `frontend/src/notifications.js`, `frontend/src/components/AppHeader.vue`, `frontend/src/views/NotificationsView.vue` (ใหม่), `frontend/src/router/index.js`

---

## 2026-07-20 (57) — เพิ่มไอคอนกระดิ่งแจ้งเตือน + dropdown popup ที่ header

**ที่มา**: ผู้ใช้อยากให้มีหน้าหรือ popup แสดงการแจ้งเตือน พร้อม icon ที่ header หลังคุยกันถึงตัวเลือก (dropdown popup vs หน้าเต็ม, สถานะปัจจุบันเท่านั้น vs ประวัติย้อนหลัง) ผู้ใช้เลือกทั้งคู่ตามที่แนะนำ: **dropdown popup** + **สถานะปัจจุบันเท่านั้น** (ไม่ต้องมี backend/DB ใหม่ ใช้ข้อมูลจาก `/api/forecast` ที่มีอยู่แล้ว)

**ปัญหาที่ต้องระวังก่อนเริ่ม**: `/api/forecast` เป็น endpoint ที่หนัก (เรียก API ภายนอก + inference 3 โมเดล + บางครั้งเรียก Gemini) `AppHeader.vue` mount อยู่ทุกหน้า ส่วน `ForecastView.vue` (ที่ fetch endpoint นี้อยู่แล้ว) mount เฉพาะหน้า "/" — ถ้ากระดิ่ง fetch เองอิสระจะยิงซ้ำ 2 ครั้งตอนโหลดหน้า "/"

**เพิ่ม [frontend/src/notifications.js](frontend/src/notifications.js)** (ใหม่): shared state ตาม pattern เดียวกับ `auth.js` เป๊ะ (reactive singleton + dedup ผ่าน module-scoped promise cache — โปรเจกต์นี้ไม่ใช้ Pinia) มี `notificationState`, `fetchForecast()` (ย้าย fetch logic ออกจาก `ForecastView.vue`), และ `buildWarningMessage(riskGroup)` (ย้าย `_WARNING_ADVICE_BY_RISK_GROUP` จาก CHANGELOG 55 มาไว้ที่เดียว กันข้อความไม่ตรงกันระหว่างแบนเนอร์บนหน้ากับ dropdown)

**แก้ [frontend/src/views/ForecastView.vue](frontend/src/views/ForecastView.vue)**: `loadForecast()` เรียก `fetchForecast()` แทน `fetch(API_URL)` ตรงๆ (พฤติกรรม/ref ที่ assign เหมือนเดิมทุกอย่าง) `warningMessage` computed เรียก `buildWarningMessage()` แทน logic ในไฟล์เอง

**แก้ [frontend/src/components/AppHeader.vue](frontend/src/components/AppHeader.vue)**: เพิ่มปุ่มกระดิ่ง (`Bell`/`BellDot` จาก `@lucide/vue`) เป็น child แรกของ `.app-header-auth` มี badge จุดแดง pulse (mirror `.live-dot`/`@keyframes pulse-dot` จาก `ForecastView.vue` แค่เปลี่ยนสีเป็น `--auth-error`) เมื่อ `high_pollution_risk=true` คลิกเปิด dropdown (`.glass-card` จาก `main.css` + `--auth-*` tokens ให้เข้ากับ header ไม่ใช่ token ของ `.forecast-page`) แสดง 1 ใน 3 สถานะ: ข้อความเตือน (เหมือนแบนเนอร์บนหน้าเป๊ะ) / ข้อความ Prophet fallback / "ไม่มีการแจ้งเตือน" มีลิงก์ "ดูรายละเอียดที่หน้าพยากรณ์" ปิด dropdown เมื่อคลิกนอกกล่อง (ใช้ `document` click listener ตรงๆ — ไม่มี modal/dropdown pattern เดิมให้ reuse ในโปรเจกต์) `onMounted` เรียกทั้ง `fetchCurrentUser()` และ `fetchForecast()`

**ทดสอบ**: `npm run build` ผ่าน ทดสอบด้วย Selenium mock `window.fetch` (CDP `Page.addScriptToEvaluateOnNewDocument`) นับจำนวนครั้งที่เรียก `/api/forecast` จริง — **ยืนยัน dedup ทำงาน: โหลดหน้า "/" ยิงแค่ 1 ครั้ง ไม่ใช่ 2** (แม้ทั้ง `AppHeader` และ `ForecastView` จะ mount พร้อมกัน) ทดสอบครบ 3 สถานะ (เตือน/info/ว่างเปล่า) — เนื้อหา dropdown ตรงกับที่ออกแบบทุกกรณี, badge จุดแดงขึ้นเฉพาะตอน `high_pollution_risk=true`, ปิด dropdown เมื่อคลิกนอกกล่องได้ถูกต้อง — screenshot ยืนยันแล้วว่า UI จัดวางถูกต้อง ข้อความ dropdown ตรงกับแบนเนอร์บนหน้าเป๊ะ (มาจากฟังก์ชันเดียวกัน)

**ไฟล์ที่แก้**: `frontend/src/notifications.js` (ใหม่), `frontend/src/views/ForecastView.vue`, `frontend/src/components/AppHeader.vue`

---

## 2026-07-20 (56) — แก้ NLG รายวันของกลุ่มเปราะบางย้อนแย้งในตัวเองตอนอากาศดี

**ที่มา**: ผู้ใช้ส่งภาพข้อความสรุปวันนี้ของกลุ่ม "เด็ก-ผู้สูงอายุ" ("...กลุ่มเด็กและผู้สูงอายุควรเฝ้าระวังสุขภาพเป็นพิเศษและเลี่ยงการทำกิจกรรมกลางแจ้งนานๆ หากรู้สึกผิดปกติครับ") บอกว่า "แปลกๆ" ถามแล้วยืนยันว่าเป็น 2 ปัญหา: (1) เนื้อหาย้อนแย้งในตัวเอง — บอกว่าอากาศ "ดี" แต่ดันแนะนำให้เลี่ยงกิจกรรมกลางแจ้งในประโยคเดียวกัน (2) ภาษา/ไวยากรณ์งงๆ

**สาเหตุ**: `_USER_GROUP_TONE` เดิม (จาก (52)) สั่ง Gemini ให้ "เน้นย้ำ...ควรระมัดระวังเป็นพิเศษ" แบบไม่มีเงื่อนไขติดกับระดับ PM2.5 จริงเลย พอวันนั้นอากาศ "ดี" จริง Gemini พยายามทำตามคำสั่ง "เน้นย้ำความระมัดระวัง" อยู่ดี เลยใส่คำแนะนำ "เลี่ยงกิจกรรมกลางแจ้ง" เข้าไปทั้งที่ไม่มีเหตุผลรองรับจากระดับจริง แล้วพยายามประนีประนอมด้วยการเติม "หากรู้สึกผิดปกติ" ต่อท้าย กลายเป็นประโยคที่ทั้งย้อนแย้งและอ่านงง (safety-check ผ่านได้เพราะ (53) ผ่อนกฎ generic-caution-floor ให้กลุ่มเปราะบางไปแล้ว — กฎยังทำงานถูกต้อง แค่ prompt ต้นทางเป็นปัญหา)

**แก้ [app.py](app.py)**: เขียน `_USER_GROUP_TONE["เด็ก-ผู้สูงอายุ"]`/`["มีโรคประจำตัว"]` ใหม่ เพิ่มเงื่อนไขชัดเจน: คำแนะนำต้อง**สอดคล้องกับระดับคุณภาพอากาศจริงเสมอ ห้ามย้อนแย้งกัน** — ถ้าอากาศดีมาก/ดี ให้เขียนเชิงให้ความมั่นใจ แนะนำแค่ "สังเกตอาการตนเองเบาๆ" ห้ามแนะนำให้หลีกเลี่ยง/ลดกิจกรรมกลางแจ้งโดยไม่มีเหตุผลรองรับจากระดับจริง (ระดับสูงกว่านั้นยังคงแนะนำเข้มขึ้นตามเดิม ไม่ได้ลดความเข้มงวดของคำเตือนตอนอากาศแย่จริง)

**ทดสอบ**: เรียก Gemini จริง (ไม่ mock) ด้วยข้อมูลจริงวันนี้ (avg_pm25=20.5, "ดี") 3 รอบ × 2 กลุ่มเปราะบาง — **6/6 ไม่ย้อนแย้งแล้ว** ทุกข้อความพูดว่า "สามารถทำกิจกรรมกลางแจ้งได้ตามปกติ" สอดคล้องกับระดับ "ดี" จริง พร้อมยังคงมีคำแนะนำเฉพาะกลุ่มแบบไม่ขัดแย้ง (เช่น "คอยสังเกตอาการตนเองเบาๆ") ภาษาลื่นไหลขึ้น ไม่มีประโยคเติมแก้เก้อแบบเดิม `scripts/test_nlg_safety_check.py`, `scripts/test_daily_nlg.py`, `scripts/test_risk_group_e2e.py` ผ่านทั้งหมด ไม่มี regression regenerate แถวจริงของวันนี้ทั้ง 2 กลุ่มเปราะบางแล้ว (`เด็ก-ผู้สูงอายุ` เจอ Gemini timeout รอบแรกเลย fallback ไปหนึ่งครั้ง ลองซ้ำแล้วผ่าน)

**สรุป**: NLG รายวันของกลุ่มเปราะบางจะไม่มีประโยคย้อนแย้งกันเองอีกในวันที่อากาศดี ยังคงความเข้มงวดของคำเตือนไว้เต็มที่เมื่ออากาศแย่จริง

**ไฟล์ที่แก้**: `app.py` (`_USER_GROUP_TONE`)

---

## 2026-07-20 (55) — ปรับข้อความเตือน "PM2.5 อยู่ในระดับสูง" ให้เข้าใจง่ายขึ้น + ปรับตาม risk_group

**ที่มา**: ผู้ใช้สังเกตกล่องเตือน `⚠️ ช่วงนี้ PM2.5 อยู่ในระดับสูง...` ใน `ForecastView.vue` แล้วถามที่มา หลังอธิบายว่าเป็น static text ผูกกับ `high_pollution_risk` ผู้ใช้ถามต่อว่าควรเอาออกไหม (แนะนำว่าไม่ควรเอาออกทั้งหมดเพราะผูกกับกลไก bias-correction ที่เป็นฟีเจอร์หลัก แต่เสนอปรับข้อความให้เข้าใจง่ายขึ้น + ปรับตาม risk_group เหมือน NLG รายวัน) ผู้ใช้เลือกให้ปรับข้อความ

**แก้ [frontend/src/views/ForecastView.vue](frontend/src/views/ForecastView.vue)**:
- ข้อความเดิม "จากการทดสอบโมเดลพบว่าค่าพยากรณ์อาจต่ำกว่าความเป็นจริง...โปรดใช้ความระมัดระวังเพิ่มเติม" เปลี่ยนเป็น 2 ส่วน: ส่วนแรก (เหตุผลเรื่อง bias ของโมเดล) เหมือนกันทุกคน เขียนให้เข้าใจง่ายขึ้น ตัดคำว่า "จากการทดสอบโมเดล" ที่เป็นศัพท์เทคนิคออก + ส่วนท้าย (คำแนะนำ) เปลี่ยนตาม `risk_group` ผ่าน dict ใหม่ `_WARNING_ADVICE_BY_RISK_GROUP` (static text ฝั่ง frontend ไม่เรียก Gemini เพิ่ม ต่างจาก NLG รายวัน)
- เพิ่ม `riskGroup` ref ดึงจาก `data.risk_group` ที่ `/api/forecast` ส่งมาอยู่แล้ว (เพิ่มเข้า response ตอน CHANGELOG 52) ครอบคลุมทั้ง guest และ login ทุกกลุ่มโดยไม่ต้องเช็ค auth state แยก
- เพิ่ม computed `warningMessage` ประกอบข้อความทั้ง 2 ส่วน แทนที่ template เดิมที่ hardcode ข้อความเต็มไว้

**ทดสอบ**: `npm run build` ผ่าน ทดสอบด้วย Selenium + mock `window.fetch` (inject ผ่าน CDP `Page.addScriptToEvaluateOnNewDocument` กันโดน reset ตอน navigate) จำลอง response `high_pollution_risk=true` ทั้ง 3 risk_group — ยืนยันข้อความและ screenshot ถูกต้องครบทั้ง 3 แบบ, กล่องยังจัดวาง/ตัดบรรทัดปกติไม่ล้น

**ไฟล์ที่แก้**: `frontend/src/views/ForecastView.vue`

---

## 2026-07-20 (54) — เอาปุ่ม "ออกจากระบบ" ออกจาก header เหลือแค่ในหน้า profile

**ที่มา**: ผู้ใช้ขอให้เอาปุ่ม "ออกจากระบบ" ที่ header (`AppHeader.vue`) ออก เหลือไว้แค่ในหน้า profile (`ProfileView.vue` มีปุ่มนี้อยู่แล้วอิสระจาก header)

**แก้ [frontend/src/components/AppHeader.vue](frontend/src/components/AppHeader.vue)**: ลบปุ่ม `<button class="logout-btn" @click="handleLogout">` ออกจาก `.app-header-auth` (ตอน login แล้วเหลือแค่ลิงก์อีเมลไปหน้า profile) ลบโค้ดที่กลายเป็น dead code ตามไปด้วย: ฟังก์ชัน `handleLogout()`, import `useRouter`/`logout` ที่ใช้แค่ในฟังก์ชันนั้น, และ CSS `.logout-btn`/`.logout-btn:hover` (ไฟล์นี้เท่านั้น — สำเนาใน `ProfileView.vue` เป็น scoped style แยกกัน ไม่ถูกแตะ)

**ทดสอบ**: `npm run build` ผ่าน ไม่มี error สมัคร test user จริง (`@test-marker.invalid`) ผ่าน Selenium บน dev server ยืนยัน header หลัง login ไม่มีคำว่า "ออกจากระบบ" เลยและยังเห็นอีเมล user ปกติ, หน้า `/profile` ยังมีปุ่ม "ออกจากระบบ" ทำงานได้ตามเดิม — screenshot ยืนยันแล้ว ลบ test user ทิ้งหลังทดสอบ

**ไฟล์ที่แก้**: `frontend/src/components/AppHeader.vue`

---

## 2026-07-20 (53) — แก้ NLG ของกลุ่มเปราะบางตกไป fallback บ่อยเกินไป (safety check เข้มงวดเกินไป)

**ที่มา**: ระหว่างตรวจสอบการทำงานจริงของ (52) พบว่าข้อความ NLG รายวันของ "มีโรคประจำตัว" ตกไปใช้ fallback template บ่อยผิดปกติ (log: `safety check failed: Gemini message did not match actual level`) ทั้งที่ระดับที่ Gemini พูดถึง ("ดี") ตรงกับความจริงเป๊ะ

**สาเหตุที่แท้จริง (วินิจฉัยจากการ dump raw Gemini output + `_mentioned_levels()` ตรงๆ)**: ไม่ใช่เพราะ Gemini เอ่ยชื่อระดับผิด (`mentioned_levels` ตรงกับ `actual_level` ทุกครั้ง) แต่เพราะ `_GENERIC_CAUTION_KEYWORDS` ใน `validate_health_message()` ถือว่าวลี "หลีกเลี่ยงกิจกรรมกลางแจ้ง" ต้องมาพร้อมระดับจริง "ปานกลาง" ขึ้นไปเท่านั้น ถึงจะไม่นับว่าเกินจริง — ซึ่งสมเหตุสมผลสำหรับกลุ่ม "ทั่วไป" แต่ผิดสำหรับกลุ่มเปราะบาง เพราะโทน `_USER_GROUP_TONE["มีโรคประจำตัว"]` ที่เพิ่งผูกเข้า prompt ใน (52) ตั้งใจให้ Gemini แนะนำระมัดระวังเร็วกว่าคนทั่วไปอยู่แล้ว (คนละหลักการเดียวกับ `RISK_GROUP_THRESHOLD_OFFSET`) พอ Gemini ทำตามคำสั่งจริงๆ (พูดว่า "ควรหลีกเลี่ยงกิจกรรมกลางแจ้ง" ทั้งที่ระดับจริงคือ "ดี") กลับโดนกฎเก่าที่ไม่รู้จัก risk_group ตัดสินว่า "เกินจริง" ทั้งที่เป็นคำแนะนำที่ถูกต้องตามเจตนา

**แก้ 2 จุดใน `app.py`**:
1. `_build_nlg_prompt()`/`_build_daily_nlg_prompt()` เพิ่มบรรทัดสั่ง Gemini ห้ามเอ่ยชื่อระดับ AQI อื่นที่ไม่ใช่ระดับจริงเด็ดขาด — ช่วยลดความเสี่ยงเอ่ยชื่อระดับผิด (แม้ไม่ใช่สาเหตุหลักของเคสนี้ แต่เป็นการป้องกันเชิงรุกที่เข้ากันได้ ไม่มีผลเสีย)
2. **แก้จริง**: `validate_health_message(message_text, predicted_pm25, risk_group="ทั่วไป")` เพิ่มพารามิเตอร์ risk_group — ข้าม `_mentions_generic_caution_below()` เมื่อ `risk_group != "ทั่วไป"` เท่านั้น (เช็คระดับหลักที่เอ่ยชื่อตรงๆ ยังเข้มงวดเท่าเดิมทุกกลุ่ม กันข้อความอ้างระดับคุณภาพอากาศผิดจากจริงอยู่) อัปเดต call site ทั้ง 2 จุด (`generate_nlg_explanation()` ส่ง `user_group`, `generate_daily_nlg_explanation()` ส่ง `risk_group`) ให้ส่งค่าจริงเข้าไปแทนใช้ default เฉยๆ

**ทดสอบ**: 
- `scripts/test_nlg_safety_check.py` (เดิม) — ผ่านทั้งหมด ไม่มี regression (default `risk_group="ทั่วไป"` คงพฤติกรรมเข้มงวดเดิม 100% สำหรับ backward compat)
- `scripts/test_risk_group_e2e.py` (จาก 52) — ผ่านทั้งหมดเหมือนเดิม
- เรียก Gemini จริงซ้ำ 6 รอบให้กลุ่ม "มีโรคประจำตัว" หลังแก้ — **ผ่าน safety check 6/6** (ก่อนแก้วัดได้ล้มเหลวส่วนใหญ่ ~5/8 รอบจากการสุ่มตรวจตอน (52)) รวมถึงข้อความที่มีคำว่า "หลีกเลี่ยง...กิจกรรมกลางแจ้ง" ตรงๆ ซึ่งก่อนหน้านี้จะตกทุกครั้ง
- regenerate แถวจริงของวันนี้ (`มีโรคประจำตัว`) ใหม่ — ได้ข้อความ personalized จริงแล้ว (`used_fallback=False`) แทนที่ fallback เดิม

**สรุป**: ข้อความ NLG ของกลุ่มเปราะบางจะ personalized ได้จริงตามที่ (52) ตั้งใจไว้ แทนที่จะตกไปใช้ template ทั่วไปบ่อยๆ โดยไม่ลดความเข้มงวดของ safety check สำหรับกลุ่ม "ทั่วไป" เลย

**ไฟล์ที่แก้ในรอบนี้**: `app.py` (`_build_nlg_prompt`, `_build_daily_nlg_prompt`, `validate_health_message`, `generate_nlg_explanation`, `generate_daily_nlg_explanation`)

---

## 2026-07-20 (52) — ผูก risk_group เข้ากับ NLG รายวันและ alert threshold จริง

**ที่มา**: `risk_group` มีอยู่ในตาราง `users` ตั้งแต่ระบบ auth (CHANGELOG 40) แต่ตั้งใจปล่อยไว้ไม่ผูกกับอะไรเลย (deferred เพราะกังวลเรื่อง Gemini quota — ดู memory `project-hourly-nlg-deferred`) ผู้ใช้ตัดสินใจผูกจริงตอนนี้: (1) โทนคำอธิบาย NLG รายวันปรับตาม risk_group (2) alert threshold ลดลงสำหรับกลุ่มเปราะบาง

**ทำไมไม่ขัดกับเหตุผลเดิมที่ deferred hourly NLG**: ของเดิมกังวลเรื่องเรียก Gemini สูงสุด 24 ครั้ง/รอบพยากรณ์ (ต่อ horizon) งานนี้แตะแค่สรุป**รายวัน** ที่เดิมถูกจำกัดไว้แล้วเหลือ ~1 ครั้ง/วัน (guard ผ่าน unique `date_thai`) — ขยายเป็นต่อ `risk_group` ทำให้เพดานขยับเป็น**สูงสุด 3 ครั้ง/วัน** (ต่อกลุ่มที่มีคนขอจริงเท่านั้น ไม่ generate ล่วงหน้าทั้ง 3) ไม่ใช่ 24 — เป็น tradeoff ที่ตั้งใจเปิดเผยชัดเจน ไม่ใช่ quota regression แบบเงียบๆ

**Schema migration**: เพิ่ม `migrate_daily_nlg_risk_group()` ใน `app.py` (เรียกที่ module level ต่อจาก `migrate_calibrated_actuals()`) — `daily_nlg_log` มีอยู่แล้วตั้งแต่ Postgres migration (CHANGELOG 30) `CREATE TABLE IF NOT EXISTS` เดิมช่วยไม่ได้เพราะตารางมีอยู่แล้ว ต้อง `ALTER TABLE ... ADD COLUMN IF NOT EXISTS risk_group ... DEFAULT 'ทั่วไป'` จริง แล้วเปลี่ยน unique constraint จาก `date_thai` เดี่ยวเป็น `(date_thai, risk_group)` คู่กัน (ลบ constraint เก่า, เพิ่มใหม่, idempotent — เช็คจาก `pg_constraint` ก่อนทุกครั้ง) แถวเก่าทั้งหมดได้ `risk_group='ทั่วไป'` โดยอัตโนมัติ (ค่า DEFAULT) ซึ่งตรงกับความจริงอยู่แล้ว เพราะตอนนั้นทุกคนเห็นข้อความเดียวกัน **ทำ `pg_dump` backup ไปที่ `backups/pm25_forecast_pg_20260720_144847.dump` ก่อนรันจริงกับ DB จริง** ตาม convention มาตรฐานของโปรเจกต์สำหรับการเปลี่ยน schema ยืนยัน idempotency แล้วโดยโหลด `app.py` ซ้ำ 2 รอบติดกัน (ดูผลทดสอบด้านล่าง)

**เพิ่ม NLG per-risk-group**: `get_daily_nlg_row`/`save_daily_nlg_row`/`get_today_daily_nlg`/`maybe_generate_daily_nlg`/`generate_daily_nlg_explanation` ทุกตัวรับ `risk_group="ทั่วไป"` (default) เพิ่ม — `_build_daily_nlg_prompt()` ผูก `_USER_GROUP_TONE` เข้ากับ prompt จริง (`f"โดย{tone}"`) เหมือน pattern ของ `_build_nlg_prompt()` (hourly, dead-code path) ที่มีอยู่แล้วทุกประการ — ใช้ tone dict เดิม ไม่สร้างใหม่

**เพิ่ม per-risk-group effective threshold**: `RISK_GROUP_THRESHOLD_OFFSET = {"ทั่วไป": 0.0, "เด็ก-ผู้สูงอายุ": -10.0, "มีโรคประจำตัว": -10.0}` + `effective_threshold_for(model_name, horizon, risk_group)` ต่อยอดจาก `EFFECTIVE_THRESHOLD_BY_HORIZON_BY_MODEL` เดิม (ไม่แก้ของเดิม แค่บวก offset เพิ่ม) **เหตุผลเลือก -10**: threshold คนทั่วไป (37.5) อยู่ตรงรอยต่อ "ปานกลาง" (25.1-37.5 ตาม PCD)/"เริ่มมีผลกระทบต่อสุขภาพ" พอดี แนวทาง WHO/US EPA แนะนำให้กลุ่มเปราะบางเริ่มระวังตัวก่อนคนทั่วไปประมาณ 1 ระดับ AQI (ตั้งแต่ "ปานกลาง" ไม่ใช่รอ "เริ่มมีผลกระทบ") ลด 10 หน่วย (→ ~27.5) ทำให้เข้าเกณฑ์นี้พอดี เป็นตัวเลขกลมๆ ที่อธิบายได้ ไม่มี dataset เฉพาะรองรับ (ตามที่ตกลงกันไว้ล่วงหน้า) ทั้งสองกลุ่มเปราะบางใช้ค่าเดียวกันเพราะไม่มีหลักฐานแยกความรุนแรง

**แก้ `get_forecast()`/`api_forecast()`**: `get_forecast(risk_group="ทั่วไป")` ใช้ `effective_threshold_for()` แทน `EFFECTIVE_THRESHOLD_BY_HORIZON_BY_MODEL[model_name][h]` ตรงๆ ทั้งใน loop คำนวณ `risk_horizons` และเทียบ `latest_pm25` กับ `HIGH_POLLUTION_THRESHOLD` เพิ่ม `"risk_group"` เข้า response dict `api_forecast()` อ่าน `session.get("user_id")` (ไม่บังคับ login) ถ้ามี user จริงดึง `risk_group` จาก DB ถ้าไม่มี fallback `"ทั่วไป"` — route ยังคง public เหมือนเดิม การ predict/`select_model()`/`prediction_log` **ไม่ถูกแตะเลย** risk_group มีผลแค่ตอนเทียบ threshold กับ NLG variant ที่แนบไปเท่านั้น

**ทดสอบ**: เพิ่ม `scripts/test_risk_group_e2e.py` (ใหม่) — 4 ส่วน:
- STEP A: `effective_threshold_for()` offset ถูกต้องทุกโมเดล/horizon แบบ structural (ไม่พึ่งข้อมูลสด) — **PASS ทั้งหมด**
- STEP B: `get_forecast(risk_group)` จริง 3 กลุ่ม + guest default เทียบกันบนข้อมูลสดชุดเดียวกัน — ยืนยัน guest = ทั่วไป เป๊ะ, `risk_horizons` ของกลุ่มเปราะบางเป็น superset ของทั่วไปเสมอ (structural guarantee ไม่ขึ้นกับสภาพอากาศวันที่รัน) — วันที่รันจริงอากาศดี (avg 20.3, "ดี") เลยไม่มี horizon ไหน trigger เลยทั้ง 3 กลุ่ม (ไม่ใช่บั๊ก แค่ยังไม่ถึงเกณฑ์แม้จะลดแล้ว) — **PASS ทั้งหมด**
- STEP C: prompt ที่ capture จริง (mock `_call_gemini`) มีโทนตรงกลุ่ม ไม่ปนกลุ่มอื่น เรียก Gemini ครบ 3 ครั้ง (1/กลุ่ม) ไม่เกิน เรียกซ้ำวันเดียวกันไม่เรียกซ้ำ (cache ผ่าน `daily_nlg_log`) แถว composite key แยกกันจริงต่อกลุ่ม — **PASS ทั้งหมด**
- STEP D: ผ่าน HTTP จริง (`test_client()`) login 3 คนคนละ risk_group เรียก `/api/forecast` จริง ยืนยัน session resolve `risk_group` ถูกต้อง + guest ไม่ login ได้ `"ทั่วไป"` — **PASS ทั้งหมด**

ลบแถว marker (`daily_nlg_log` ปี 2199, `users` โดเมน `@test-marker.invalid`) ใน `finally` แล้วยืนยันว่าไม่เหลือตกค้างจริง (query แยกหลังรัน = 0 ทั้งคู่)

**ตรวจสอบเพิ่มเติม (ไม่ mock, เรียก Gemini จริง)**: เทียบข้อความ NLG รายวันจริงระหว่าง "ทั่วไป" กับ "มีโรคประจำตัว" ที่ avg_pm25 เดียวกัน (38.6, "เริ่มมีผลกระทบต่อสุขภาพ") — ฉบับ "มีโรคประจำตัว" ระบุชัดเจนว่า "ผู้ที่มีโรคประจำตัวเกี่ยวกับทางเดินหายใจควรหลีกเลี่ยงกิจกรรมกลางแจ้งและระมัดระวังสุขภาพเป็นพิเศษ" ต่างจากฉบับทั่วไปที่แนะนำทั่วไป ("หลีกเลี่ยงกิจกรรมกลางแจ้งและสวมหน้ากากอนามัย") ยืนยันว่าโทนต่างกันจริงตามที่ตั้งใจ

**สรุป**: ผูกสำเร็จทั้ง NLG และ threshold ตามที่ขอ ไม่กระทบ model inference/prediction_log/rolling-R² เลย เพดาน Gemini เพิ่มจาก 1 เป็นสูงสุด 3 ครั้ง/วัน (เปิดเผยชัดเจน ไม่ใช่ regression เงียบ)

**ไฟล์ที่แก้ในรอบนี้**: `app.py` (schema migration ใหม่, thread risk_group ผ่าน daily NLG functions ทั้งหมด, `RISK_GROUP_THRESHOLD_OFFSET`/`effective_threshold_for()` ใหม่, `get_forecast()`/`api_forecast()`), `scripts/test_risk_group_e2e.py` (ใหม่), `backups/pm25_forecast_pg_20260720_144847.dump` (ใหม่)

---

## 2026-07-20 (51) — ทดสอบ post-hoc bias correction สำหรับ XGBoost — ล้มเหลว ไม่ deploy

**ที่มา**: sample weighting ของ XGBoost ล้มเหลวไปแล้ว 2 รอบ (ดู (45)-(46)) ผู้ใช้เสนอวิธีใหม่ที่ไม่แตะโมเดล/ไม่ retrain — เขียนฟังก์ชัน `correct_xgboost_prediction(raw_pred, horizon, actual_context)` ชดเชย raw prediction บวก `bias_high` ต่อ horizon (จาก `models/xgb_bias_high_by_horizon_20260712_191350.csv` ที่คำนวณไว้แล้ว) เฉพาะตอน raw prediction เข้าใกล้ช่วงมลพิษสูง แล้วให้ทดสอบ MAE/bias ก่อน-หลัง ก่อนจะไปแก้ `get_forecast()`/`EFFECTIVE_THRESHOLD_BY_HORIZON_BY_MODEL` จริง

**เพิ่ม [scripts/eval_xgb_posthoc_correction.py](scripts/eval_xgb_posthoc_correction.py)**: ramp เชิงเส้นระหว่างโซน (LOW,HIGH) ของ raw_pred แทน hard cutoff (กันรอยต่อ/discontinuity) มี context gate เสริม (ลดการชดเชยลงครึ่งหนึ่งถ้า `pm25_lag_1` ยังต่ำ — ไม่มีหลักฐานจริงว่าเข้าสู่ช่วงมลพิษสูง) ลอง 4 โซน × มี/ไม่มี gate = 8 config

**บั๊กที่พบและแก้ระหว่างเขียนสคริปต์**: ร่างแรกทดสอบ correction บน test set เดิม — ตัวเดียวกับที่ใช้คำนวณ `bias_high_by_horizon` เอง เป็นการทดสอบวนกลับ (circular): บวกเลขที่คำนวณจากชุดข้อมูลนั้นเองกลับเข้าไปในชุดเดิม วัด bias ดีขึ้น 94-99% ซึ่งเป็นภาพลวงตาล้วนๆ ไม่มีความหมาย แก้โดยเลือก config จาก `valid_tune` (15% ท้ายของ train, คนละช่วงเวลากับ test) เท่านั้น แล้วค่อยยืนยันครั้งเดียวบน test จริง — ตาม pattern nested split เดียวกับที่ `retrain_weighted_v2.py`/`retrain_xgb_weighted.py` ใช้อยู่แล้ว

**ผลลัพธ์**: **ล้มเหลวชัดเจน ไม่มี config ไหนผ่านเกณฑ์ MAE_all ไม่แย่ลงเกิน 8% เลยแม้แต่อันเดียว** — บน `valid_tune` ทุก config ทำให้ `|bias_high|` แย่ลงจาก 0.040 เป็น ~3.0 (แย่ลง ~74 เท่า) และ MAE_all แย่ลง 138-234% พร้อมกันทุก horizon (24/24) สาเหตุ: `bias_high` ไม่ใช่คุณสมบัติคงที่ของโมเดลตลอดเวลา แต่กระจุกตัวเฉพาะฤดูฝุ่น (`valid_tune` ซึ่งส่วนใหญ่อยู่นอกฤดูฝุ่นแทบไม่มี bias เลย ขณะที่ test set ซึ่งรวม ก.พ.-มี.ค. มี bias เฉลี่ยสูงถึง 3.04 ตรงกับ backtest ฤดูฝุ่นที่เคยวัดไว้ +2.473 ดู TEST_RESULTS.md 2026-07-16) ค่าคงที่เฉลี่ยทั้งปีที่บวกกลับเข้า prediction ทุกครั้งจึงชดเชยเกินจริงอย่างรุนแรงนอกฤดูฝุ่น (ส่วนใหญ่ของปี) รายละเอียดเต็มใน [TEST_RESULTS.md](TEST_RESULTS.md)

**สรุป**: **ไม่ deploy ไม่แก้ `app.py`/`prediction_log` เลย** — ล้มเหลวชัดเจนกว่า sample weighting เสียอีก ไม่แนะนำให้ลองปรับ zone/gate เพิ่มด้วย recipe เดิม เพราะปัญหาไม่ใช่พารามิเตอร์ผิดแต่เป็นเพราะ `bias_high_by_horizon` เองไม่คงที่ตามเวลา ระบบ `EFFECTIVE_THRESHOLD_BY_HORIZON_BY_MODEL` ที่ใช้อยู่จริง (ขยับแค่เกณฑ์เปรียบเทียบ ไม่แก้ค่าที่แสดง) ปลอดภัยกว่าวิธีนี้มากเพราะไม่มีความเสี่ยง "ชดเชยผิดช่วงเวลา" แบบนี้ ยังคงใช้งานตามเดิมไม่เปลี่ยนแปลง

**ไฟล์ที่แก้ในรอบนี้**: `scripts/eval_xgb_posthoc_correction.py` (ใหม่), `models/eval_xgb_posthoc_correction_select_20260720_142320.csv` (ใหม่), `TEST_RESULTS.md`

---

## 2026-07-20 (50) — เพิ่มโลโก้ในหัวเว็บ (AppHeader)

**ที่มา**: ผู้ใช้ส่งไฟล์ `Logo.svg` (โลโก้วงกลมใบไม้/ปอดสีเขียวอมฟ้า ผสมคำว่า "PM2.5 Forecast" สร้างจาก Canva AI มี C2PA manifest ระบุ `ContainsAiGeneratedContent: Yes`) ให้ช่วยใส่ในเว็บและถามว่าควรอยู่ตรงไหน

**เหตุผลเลือกตำแหน่ง/รูปแบบ**: ไฟล์ต้นฉบับเป็น canvas 1500x1500 ที่มีขอบขาวว่างเยอะรอบโลโก้จริง (ตัวไอคอน+ข้อความ "PM2.5 Forecast" กินพื้นที่จริงแค่ราวกลางภาพ) ลองทั้ง 2 แบบก่อนตัดสินใจ: (1) crop ทั้ง lockup (ไอคอน+คำว่า "PM2.5 Forecast") กับ (2) crop เฉพาะไอคอนวงกลม — เลือกแบบ (2) เพราะข้อความ "Forecast" ในโลโก้จะซ้ำซ้อน/ขัดกับข้อความ "PM2.5 Khon Kaen" ที่มีอยู่แล้วใน `.brand` (ซึ่งระบุชื่อสถานีเฉพาะเจาะจงกว่า) — จึงใช้แค่ไอคอนวงกลมคู่กับข้อความเดิม

**Crop**: ไม่ได้พิมพ์ไฟล์ SVG ใหม่เอง (เสี่ยงทำ base64 image data ที่ฝังอยู่เสียหาย — เคยพลาดมาก่อนตอนพยายาม retype จาก paste ในแชท) ใช้วิธีคัดลอกไฟล์จริงแล้วแก้แค่ attribute `viewBox` เดียว (`"0 0 1500 1499.999933"` → `"247 505 466 470"`) ซึ่งอิงจาก `<clipPath>` ที่มีอยู่แล้วในไฟล์ต้นฉบับ (`M 257.25 515.25 ... 703.5 985.5`) เป็นตัวกำหนดกรอบไอคอน — ปลอดภัยเพราะไม่แตะ coordinate ภายในใดๆ เลย พิสูจน์ด้วย screenshot จริงก่อนใช้งานจริง

**เพิ่ม [frontend/src/assets/logo-icon.svg](frontend/src/assets/logo-icon.svg)**: ไฟล์ crop แล้ว (สำเนาของ `Logo.svg` ที่ผู้ใช้วางไว้ที่ `frontend/src/assets/Logo.svg`)

**แก้ [frontend/src/components/AppHeader.vue](frontend/src/components/AppHeader.vue)**: import `logo-icon.svg`, ใส่ `<img>` ขนาด 28x28px ไว้หน้าข้อความ `.brand` ("PM2.5 Khon Kaen") ปรับ `.brand` เป็น flex row จัดกึ่งกลางแนวตั้ง

**ทดสอบ**: render โลโก้เต็ม + เวอร์ชัน crop ผ่าน Selenium ก่อนตัดสินใจ, ตรวจ header จริงบน dev server เห็นไอคอนวงกลมคมชัดคู่กับข้อความ ไม่มีขอบขาวเหลือ

**ไฟล์ที่แก้**: `frontend/src/assets/logo-icon.svg` (ใหม่), `frontend/src/components/AppHeader.vue`

---

## 2026-07-19 (49) — ปรับดีไซน์การ์ด PM2.5 ปัจจุบัน ให้ดูเป็นมืออาชีพ/สอดคล้องกัน (CSS/layout เท่านั้น)

**ที่มา**: ผู้ใช้ชี้ปัญหา 4 ข้อในการ์ด "PM2.5 ล่าสุดรายชั่วโมง": (1) ไอคอนปนกัน — emoji หน้ายิ้ม 😊 กับอิโมจิเทอร์โมมิเตอร์/หยดน้ำ/ใบไม้ คนละสไตล์ (2) badge สีเขียวเล็กๆ ข้างขวาดูเหมือน debug element ไม่เหมือน status indicator ที่ตั้งใจออกแบบ (3) visual hierarchy ไม่ชัด ตัวเลข 18.2 แข่งกับไอคอน/badge รอบข้าง (4) spacing ระหว่าง section ไม่สม่ำเสมอ — ขอให้แก้เฉพาะ CSS/layout ไม่แตะโครงสร้างข้อมูลหรือ logic

**เพิ่ม dependency `@lucide/vue`** (ไม่ใช่ `lucide-vue-next` ที่ผู้ใช้เอ่ยถึง เพราะ deprecated แล้ว หันไปใช้ตัวที่ยังดูแลอยู่แทน — ตรวจสอบก่อนติดตั้งจริง) ใช้ 8 ไอคอน: `Laugh/Smile/Meh/Frown/Angry` (แทน emoji หน้าคน ไล่ตามระดับความรุนแรง PM2.5) และ `Thermometer/Droplet/Wind` (แทน 🌡️💧🍃)

**แก้ [frontend/src/views/ForecastView.vue](frontend/src/views/ForecastView.vue)**:
- `iconFor()` (คืน emoji string) → `moodIconFor()` (คืน component เดียวกับ Lucide) ผูกกับ `<component :is>`
- เพิ่ม `.mood-badge`: วงกลมพื้นหลังไล่สีตามระดับจริงผ่าน `color-mix(in srgb, currentColor 16%, transparent)` โดยใช้ class `statusClass()` เดิม (ดี/ปานกลาง/ฯลฯ) ร่วมกับ badge และ status chip — สีจึงเปลี่ยนตามระดับจริงไม่ใช่ค่าคงที่
- `.status-badge` → `.status-chip`: เพิ่ม padding/font-weight ให้เป็น pill ชัดเจน อ่านคำเต็ม ("ดี"/"ปานกลาง"), พื้นหลัง tint ผ่าน `--card-bg` (ปรับตามธีม ไม่ hardcode white เหมือนเดิม)
- `.pm-value`: 3rem/700 → 3.6rem/800 ให้เป็นจุดเด่นหลักของการ์ดชัดเจนกว่าไอคอน/chip รอบข้าง
- `.stat-tiles`: จาก `flex-wrap` → CSS grid 3 คอลัมน์เท่ากัน มี `border-left` คั่นระหว่างช่อง, ไอคอนแยกสีความหมาย (เทอร์โมมิเตอร์แดง/หยดน้ำฟ้า/ลมเขียว) แทนสีจาก emoji เดิม
- `.card:hover`: เพิ่ม accent-ring กลอย (`box-shadow` ring สีฟ้าอ่อน + border-color) เข้ากับ glassmorphism เดิม
- media query `max-width: 480px` ใหม่ ลดขนาดตัวเลข/badge/สถิติย่อยกันล้นบนมือถือ

**ทดสอบ**: รัน dev server จริง (`npm run dev` + `python app.py`), ถ่าย screenshot ด้วย Selenium ที่ desktop (1400px) และ mobile (380px), ยืนยัน hover-glow ด้วย `getComputedStyle(...).boxShadow` ก่อน/หลัง hover จริง (ไม่ใช่แค่ดูภาพ) — สลับไฟล์กลับไปเวอร์ชันเดิมชั่วคราวเพื่อถ่ายภาพ "ก่อนแก้" มาเทียบ แล้ว restore กลับเวอร์ชันใหม่ ตรวจ `diff` ยืนยันว่าไฟล์ตรงกับที่ตั้งใจ 100% ก่อน/หลังสลับ

**ไฟล์ที่แก้**: `frontend/src/views/ForecastView.vue`, `frontend/package.json`/`package-lock.json` (เพิ่ม `@lucide/vue`)

---

## 2026-07-19 (48) — แก้ช่องว่างในการ์ด PM2.5 ล่าสุด (จัดกึ่งกลางแทนดันลงล่าง)

**ที่มา**: ผู้ใช้ส่งภาพชี้ว่าการ์ด "PM2.5 ล่าสุดรายชั่วโมง" ยังมีช่องว่างดูแปลกอยู่ใต้ข้อความ "อัปเดตล่าสุด" — แม้ (43) จะแก้ปัญหาความสูงการ์ดไม่เท่ากันไปแล้วด้วย `align-items: stretch` + `.stat-tiles { margin-top: auto }`

**สาเหตุ**: `margin-top: auto` ใน (43) ทำงานถูกต้องตามที่ตั้งใจ (ดัน `.stat-tiles` ไปชิดขอบล่างของการ์ด) แต่เพราะคอลัมน์ขวา (การ์ดสรุปวันนี้ + การ์ดช่วงเวลาที่ดีที่สุด ซ้อนกัน 2 การ์ด) สูงกว่าเนื้อหาฝั่งซ้ายตามธรรมชาติ ผลคือช่องว่างที่เกิดจากการ stretch ไม่ได้หายไป แค่ถูกต้อนไปกองรวมเป็นก้อนเดียวอยู่ใต้ข้อความ "อัปเดตล่าสุด" แทน — ดูเป็นช่องว่างเปล่าๆ มากกว่าจะดูสมดุล

**แก้ [frontend/src/views/ForecastView.vue](frontend/src/views/ForecastView.vue)**: ห่อ `.current-row` + `.updated-at` ด้วย `<div class="current-info">` ใหม่ ตั้ง `flex: 1; display: flex; flex-direction: column; justify-content: center` ให้บล็อกนี้ขยายเต็มพื้นที่ที่เหลือแล้ว "จัดกึ่งกลาง" เนื้อหาของตัวเองในแนวตั้ง แทนที่จะปล่อยให้ช่องว่างกองอยู่ด้านล่างเนื้อหาเป็นก้อนเดียว — เอา `margin-top: auto` ออกจาก `.stat-tiles` (ไม่จำเป็นอีกต่อไปเพราะ `.current-info` ที่ยืดตัวเองจะดัน `.stat-tiles` ลงไปตามลำดับ flow ปกติอยู่แล้ว) เปลี่ยนเป็น `margin-top: 16px` ปกติ

**ทดสอบ**: รัน dev server จริง (`npm run dev` + `python app.py`) ถ่ายภาพหน้าจอการ์ดจริงด้วย Selenium ทั้งโหมดสว่าง/มืด — ยืนยันว่าช่องว่างถูกกระจายสมดุลรอบเนื้อหาแทนที่จะกองเป็นก้อนเดียวใต้ "อัปเดตล่าสุด" เหมือนก่อนแก้

**ไฟล์ที่แก้**: `frontend/src/views/ForecastView.vue`

---

## 2026-07-19 (47) — เพิ่ม requirements.txt (Python dependency manifest ที่ไม่เคยมี)

**ที่มา**: โปรเจกต์ไม่เคยมี `requirements.txt`/`pyproject.toml` เลย (ระบุไว้เป็น backlog ใน CLAUDE.md) ผู้ใช้ขอให้สร้างจาก virtual environment ที่ใช้รันโปรเจกต์จริง — ตรวจสอบแล้วพบว่าไม่มี venv เฉพาะของโปรเจกต์อยู่จริง (`python` ที่รันอยู่คือ global install Python 3.12 ที่มี package ปนจากโปรเจกต์อื่นอยู่ 178 ตัว) จึงถามผู้ใช้และเลือกแนวทาง "สร้าง venv ใหม่ + ติดตั้งเฉพาะจาก import จริงในโค้ด" แทนการ `pip freeze` จาก global env ตรงๆ

**วิธีทำ**: grep หา `import`/`from` ทั้งหมดใน `app.py` และทุกไฟล์ใน `scripts/` เพื่อยืนยัน third-party package ที่ใช้จริง (ไม่มี SQLAlchemy หรือ package อื่นนอกเหนือจากที่ระบุใน CLAUDE.md) → สร้าง `.venv` ใหม่ในโปรเจกต์ → ติดตั้ง 16 package หลักที่ใช้จริง (flask, flask-cors, apscheduler, pandas, numpy, scikit-learn, lightgbm, xgboost, prophet, requests, urllib3, matplotlib, psycopg2-binary, werkzeug, shap, google-genai) → `pip freeze` ได้ 68 บรรทัด (รวม transitive dependencies) → ทดสอบซ้ำโดยสร้าง venv ว่างอีกอันแยกต่างหาก แล้ว `pip install -r requirements.txt` + import ทุก package หลัก ผ่านหมดไม่มี error

**ผลตรวจสอบ**: ไม่พบ package ที่น่าสงสัย (local path/editable install) ในรายการ — ทุกตัวเป็น pinned version จาก PyPI ปกติ, `frontend/package.json` มีอยู่แล้วจึงไม่ต้องสร้างใหม่ฝั่ง Node

**ทดสอบ**: `pip install -r requirements.txt` บน venv ว่างสำเร็จไม่มี conflict, import `flask/flask_cors/apscheduler/pandas/numpy/sklearn/lightgbm/xgboost/prophet/requests/urllib3/matplotlib/psycopg2/werkzeug/shap/google.genai` ผ่านหมด (มี warning `plotly` ของ prophet ซึ่งเป็น optional dependency ที่โปรเจกต์นี้ไม่ได้ใช้ ไม่กระทบอะไร)

**ไฟล์ที่แก้**: `requirements.txt` (ใหม่), `.venv/` (ใหม่ — local เท่านั้น ไม่ใช่ไฟล์ที่ commit)

---

## 2026-07-17 (46) — สรุปยุติความพยายาม retrain XGBoost แบบ weighted (structural mismatch) — ไม่ deploy

**ที่มา**: ต่อจาก (45) — ผู้ใช้ตั้งสมมติฐานว่าปัญหาอาจอยู่ที่เลือก representative horizon ผิด (copy t+1/t+12/t+24 จาก LightGBM ตรงๆ แทนที่จะเลือกจุดที่มีความหมายกับ XGBoost เอง) และวางเกณฑ์ไว้ล่วงหน้าเลยว่า ถ้าลองรอบเดียวด้วย horizon ใหม่แล้วยังเจอ pattern เดิม (แย่ลงพร้อมกันหลาย horizon) ให้สรุปว่าเป็น structural mismatch จริง แล้วหยุด

**เพิ่ม [scripts/shap_divergence_by_horizon.py](scripts/shap_divergence_by_horizon.py)**: คำนวณ SHAP feature importance ทั้ง 24 horizon ของทั้ง LightGBM/XGBoost ที่ deploy จริง วัด L1 distance ของสัดส่วน mean|SHAP| ต่อ feature ระหว่างสองโมเดล — พบว่าสมมติฐาน "t+24 เน้น temp" ไม่จริง ทั้งสองโมเดลยึด `pm25_lag_1` เป็น top feature ทุก horizon เหมือนกันหมด เลือก 3 horizon ที่ต่างกันมากสุดได้ `[1, 21, 24]` (t+21 แทน t+12 เดิม)

**รัน [scripts/retrain_xgb_weighted.py](scripts/retrain_xgb_weighted.py) ซ้ำด้วย `REP_HORIZONS=[1,21,24]`**: bias_high เฉลี่ยดีขึ้นเกือบเท่าตัว (-7% เทียบ -3% รอบแรก) แต่ยังมี **7/24 horizon ที่แย่ลงพร้อมกันทั้ง MAE และ bias** (ลดจาก 9/24 รอบแรก แต่ไม่หมดไป) ที่สำคัญคือ **t+21 เอง** — horizon ที่เลือกมาเพราะ SHAP บอกว่าต่างชัดสุด และ grid search เลือก weight=10 ให้เพราะดูดีตอน validate — กลับแย่ลงทั้งคู่บน test set จริง (bias 4.898→5.373) ชัดเจนว่าเป็น overfitting กับ tuning split ไม่ใช่การแก้ที่ generalize จริง

**สรุป**: **ยุติความพยายามนี้ ไม่ deploy ทั้ง 2 รอบ** ตามเกณฑ์ที่ตกลงไว้ล่วงหน้า — sample-weighting แบบเดียวกับที่ใช้ได้ผลกับ LightGBM ไม่เข้ากับโครงสร้าง error ของ XGBoost จริงๆ ไม่ใช่แค่เลือก horizon ผิด กลับไปใช้ XGBoost แบบ unweighted เดิมที่ deploy อยู่ (`models/xgb_pm25_multistep_20260712_191350.pkl`) — **ไม่มีการแก้ `app.py` หรือ `prediction_log` เลยตลอดทั้ง (45)-(46)** ยังคง Phase A only ตามแผน รายละเอียดตัวเลขเต็มอยู่ใน [TEST_RESULTS.md](TEST_RESULTS.md)

**ไฟล์ที่แก้ในรอบนี้**: `scripts/shap_divergence_by_horizon.py` (ใหม่), `models/shap_divergence_by_horizon.csv` (ใหม่), `scripts/retrain_xgb_weighted.py` (แก้ `REP_HORIZONS`), `models/xgb_pm25_multistep_weighted_20260717_132430.pkl` (ใหม่ — ไม่ deploy), `models/eval_xgb_weighted_vs_baseline_20260717_132430.csv` (ใหม่), `TEST_RESULTS.md`

---

## 2026-07-17 (45) — Retrain XGBoost แบบ weighted (Phase A: เทรน+ประเมินเท่านั้น ยังไม่ deploy)

**ที่มา**: ผู้ใช้ตัดสินใจ retrain XGBoost ด้วย sample weighting แบบเดียวกับที่ทำกับ LightGBM ไปแล้ว (ย้อนการตัดสินใจเดิมที่เคยเลื่อนไว้ ดู memory `project-xgb-weighted-retrain-deferred`) วางแผนเป็น 2 phase: Phase A (เทรน+ประเมิน, ไม่แตะของจริง) ก่อน แล้วหยุดรีวิวผลก่อนเข้า Phase B (deploy จริง)

**เพิ่ม [scripts/retrain_xgb_weighted.py](scripts/retrain_xgb_weighted.py)**: ใช้ recipe เดียวกับ `retrain_weighted_v2.py` (ของ LightGBM) — nested 85/15 train_tune/valid_tune split หา weight ที่ 3 representative horizon (t+1/t+12/t+24) จาก `WEIGHT_GRID=[1,3,5,10]` (เลือก |bias_high| น้อยสุดโดย MAE_all ไม่แย่ลงเกิน 8%) แล้ว `np.interp` กระจาย weight ไปทั้ง 24 horizon ต่างจากสคริปต์ของ LightGBM 2 จุด: (1) feature_cols/target_cols/station_id โหลดจาก pkl ของ XGBoost เองที่ deploy อยู่ ไม่ใช่จาก pkl ของ LightGBM (2) baseline สำหรับเทียบคำนวณสดในสคริปต์เดียวกันจากโมเดล deploy จริง ไม่อ่านจาก CSV ประวัติแยก (เพราะไฟล์ผลลัพธ์ของสคริปต์นี้ตั้งใจจะมาแทน `xgb_bias_high_by_horizon_*.csv` ไปเลย)

**ผลลัพธ์**: **ไม่ชัดเจน ไม่แนะนำให้ deploy ตอนนี้** — MAE_all เฉลี่ยแย่ลงเล็กน้อย (+0.8%), bias_high เฉลี่ยดีขึ้นแค่ ~3% (ต่างจาก LightGBM ที่ recipe เดียวกันนี้ได้ผลชัดเจนกว่ามาก) และมีถึง 9 จาก 24 horizon ที่ทั้ง MAE_all และ bias_high แย่ลงพร้อมกันหลัง weight — รายละเอียดเต็มบันทึกไว้ใน [TEST_RESULTS.md](TEST_RESULTS.md) (entry 2026-07-17)

**ทดสอบ**: รันสคริปต์จริงบน test set จริง (chronological 80/20 เดียวกับตอนเทรนโมเดล deploy), ตรวจสอบว่า weight-search เคารพเกณฑ์ MAE_all +8% ที่ 3 representative horizon ทั้งหมด, เทียบ baseline vs weighted บน test set เดียวกันเป๊ะ (ไม่ปนกับ train)

**Phase B (deploy จริง — เปลี่ยน `app.py`, reset `prediction_log` เฉพาะแถว xgboost, backfill) ยังไม่เริ่ม** เพราะผลลัพธ์ Phase A ยังไม่ชัดเจนพอ กำลังรอ input จากผู้ใช้ว่าจะไปต่อ (ลอง weight grid อื่น) หรือหยุดไว้แค่นี้

**ไฟล์ที่แก้ในรอบนี้**: `scripts/retrain_xgb_weighted.py` (ใหม่), `models/xgb_pm25_multistep_weighted_20260717_131755.pkl` (ใหม่ — ยังไม่ deploy), `models/eval_xgb_weighted_vs_baseline_20260717_131755.csv` (ใหม่), `TEST_RESULTS.md`

---

## 2026-07-17 (44) — แก้เลข "1" ในวงกลมอันดับหายไปในโหมดมืด

**ที่มา**: ผู้ใช้ส่งภาพชี้ว่าตัวเลขในวงกลมอันดับ 1 (การ์ด "ช่วงเวลาที่อากาศดีที่สุด") มองไม่เห็น

**สาเหตุ**: `.rank-badge.rank-1` ใน (41) ใช้ `background: var(--text)` — ตัวแปรนี้เป็นสีตัวอักษรที่ถูกออกแบบให้ "สลับ" ตามโหมด (เข้มในโหมดสว่าง, สว่างในโหมดมืด เพื่อให้อ่านง่ายบนพื้นการ์ด) พอเอามาใช้เป็นพื้นหลังวงกลมเอง ในโหมดมืด `--text` กลายเป็นสีอ่อน (`#eceef1`) ทำให้เลข "1" สีขาว (`color:#fff`) ที่ตั้งไว้ตายตัวกลืนไปกับพื้นหลังวงกลมที่อ่อนเกือบเท่ากัน

**แก้ [frontend/src/views/ForecastView.vue](frontend/src/views/ForecastView.vue)**: เปลี่ยนพื้นหลังวงกลมอันดับ 1 จาก `var(--text)` เป็น `var(--accent)` (สีฟ้า ไม่ถูก override ในโหมดมืด — เช็คแล้วว่าคงที่ทั้ง 2 โหมด) เลขสีขาวจึงอ่านออกชัดเจนเสมอไม่ว่าโหมดไหน

**ทดสอบ**: `npm run build` ผ่าน, screenshot เทียบทั้งโหมดมืด/สว่างยืนยันเลข "1" มองเห็นชัดเจนทั้งคู่ ไม่มี console error

**ไฟล์ที่แก้ในรอบนี้**: `frontend/src/views/ForecastView.vue`

---

## 2026-07-17 (43) — การ์ด PM2.5 ล่าสุดยืดเต็มความสูงเท่าคอลัมน์ขวา

**ที่มา**: หลังแก้ (42) ผู้ใช้ชี้ว่าการ์ด "PM2.5 ล่าสุดรายชั่วโมง" ฝั่งซ้าย ยังเหลือช่องว่างว่างเปล่าด้านล่างการ์ด (สั้นกว่าคอลัมน์ขวาที่มี 2 การ์ดซ้อนกัน) อยากให้เต็มพอดี

**แก้ [frontend/src/views/ForecastView.vue](frontend/src/views/ForecastView.vue)**: `.dashboard-grid` เปลี่ยน `align-items: start` เป็น `stretch` (ให้ทั้ง 2 คอลัมน์สูงเท่ากันตามคอลัมน์ที่สูงกว่าเสมอ), เพิ่ม `.grid-col-main { display:flex; flex-direction:column; }` แล้วเปลี่ยน `.stat-tiles` จาก `margin-top: 16px` เป็น `margin-top: auto` — ดันแถวอุณหภูมิ/ความชื้น/ลมไปชิดขอบล่างของการ์ดเสมอ ช่องว่างส่วนเกินจึงไปรวมอยู่เหนือเส้นคั่นแทนที่จะค้างอยู่ใต้สุดการ์ดแบบเดิม

**ทดสอบ**: `npm run build` ผ่าน, screenshot ที่ความกว้าง 970px (จุดเดียวกับที่ผู้ใช้เจอปัญหา) ยืนยันการ์ดซ้ายยืดเต็มความสูงพอดีกับคอลัมน์ขวาแล้ว ไม่มี console error

**ไฟล์ที่แก้ในรอบนี้**: `frontend/src/views/ForecastView.vue`

---

## 2026-07-17 (42) — แก้การ์ด "ช่วงเวลาที่อากาศดีที่สุด" ตกบรรทัดจนเหลือช่องว่างแปลกๆ

**ที่มา**: ผู้ใช้ส่งภาพจอที่ความกว้างจอกลางๆ (~960px, ยังไม่ยุบเป็น 1 คอลัมน์ที่ 880px) พบว่า tile ที่ 3 ของการ์ด "ช่วงเวลาที่อากาศดีที่สุด" ตกไปขึ้นบรรทัดใหม่ (เพราะ `flex: 1 1 100px` กว้างเกินพื้นที่คอลัมน์ฝั่งขวาที่แคบลงตามสัดส่วน grid `1.6fr 1fr` จาก (41)) ทำให้การ์ดฝั่งขวาสูงกว่าการ์ด PM2.5 ปัจจุบันฝั่งซ้ายมาก เหลือช่องว่างว่างเปล่าแปลกๆ ด้านล่างการ์ดซ้าย

**แก้ [frontend/src/views/ForecastView.vue](frontend/src/views/ForecastView.vue)**: `.best-times-list` เปลี่ยนจาก `flex-wrap: wrap` เป็น `nowrap`, `.best-time-row` เปลี่ยนจาก `flex: 1 1 100px` เป็น `flex: 1 1 0; min-width: 0` (แบ่งพื้นที่เท่ากัน 3 ส่วนแล้วย่อขนาดพร้อมกันแทนการตกบรรทัด) เพิ่ม `text-overflow: ellipsis` กันตัวเลข/เวลาล้นถ้าคอลัมน์แคบมากๆ

**ทดสอบ**: `npm run build` ผ่าน, screenshot ที่ความกว้าง 970px (จุดที่เคยพัง) ยืนยันทั้ง 3 tile อยู่แถวเดียวกัน ความสูง 2 คอลัมน์ใกล้เคียงกันแล้ว ไม่มีช่องว่างแปลกๆ ไม่มี console error

**ไฟล์ที่แก้ในรอบนี้**: `frontend/src/views/ForecastView.vue`

---

## 2026-07-17 (41) — Redesign ForecastView.vue + AppHeader.vue ตาม mockup "Atmospheric Intelligence"

**ที่มา**: ผู้ใช้ส่ง mockup หน้า dashboard สไตล์ SaaS สว่าง (nav ขาว, การ์ดโค้งมน, กราฟแท่งไล่สีฟ้า, legend โมเดล) แล้วขอให้ปรับหน้า ForecastView ให้เหมือนแบบนี้ ก่อนเริ่มถามผู้ใช้ยืนยัน 3 จุดผ่าน AskUserQuestion: (1) คงภาษาไทยเดิมทั้งหมด เอาแค่สไตล์/เลย์เอาต์ ไม่แปลเป็นอังกฤษ (2) คงพื้นหลังไล่สีตามระดับฝุ่น (`.aqi-excellent/good/moderate/unhealthy/bad`) ไว้ ไม่เปลี่ยนเป็นพื้นขาวแบบ mockup (3) คงจุดสีบอกโมเดล (`.model-dot`) ใต้แท่งไว้ แม้ mockup จะไม่มี เพราะเป็นฟีเจอร์ Dynamic Routing attribution

**แก้ [frontend/src/views/ForecastView.vue](frontend/src/views/ForecastView.vue)** (ปรับสไตล์ล้วนๆ ไม่แตะ state/logic/ข้อมูลเลย):
- จัด layout ใหม่เป็น 2 คอลัมน์ (`.dashboard-grid`: การ์ด PM2.5 ปัจจุบัน+สภาพอากาศ กว้างซ้าย, การ์ดสรุปวันนี้+ช่วงเวลาดีที่สุด เรียงขวา) ยุบเหลือ 1 คอลัมน์ที่จอแคบกว่า 880px กราฟแท่งยังกว้างเต็มด้านล่างเหมือนเดิม, ขยาย `.wrap` เป็น 1180px (จาก 760px)
- `--card-bg`/`--card-border` เข้มขึ้นเกือบทึบ (0.72→0.94 alpha) + ลด blur (16px→8px) ให้การ์ดอ่านง่ายชัดเจนบนพื้นหลังไล่สีทุกระดับ (โดยเฉพาะ `.aqi-bad` ที่สีพื้นเข้มมาก) แต่ยังเห็นเงาสีจางๆ ผ่านได้ ไม่ทึบขาวสนิทแบบ mockup
- เพิ่ม class `.eyebrow` (label สั้นตัวเดียว เช่น "อุณหภูมิ") กับ `.label-muted` (label ประโยคยาว เช่น disclaimer ของค่า PM2.5) แยกกัน เพราะ label ในแอปนี้หลายจุดเป็นประโยคเตือนยาว ใส่ small-caps ทั้งประโยคจะอ่านยาก
- `.status-badge` เปลี่ยนจาก solid fill เป็น light-tint pill ด้วย `color-mix(in srgb, currentColor 16%, white)` (สีมาจากตัวแปรระดับ PM2.5 เดิมอัตโนมัติ ไม่มี hex ใหม่ซ้ำซ้อน)
- `.rank-badge` (การ์ดช่วงเวลาดีที่สุด) เปลี่ยนจาก solid สีต่างกันตามอันดับ เป็นลำดับชั้นล้วนๆ (อันดับ 1 ทึบเข้ม, 2-3 กรอบจาง) เพราะสีเดิมไม่ได้ผูกกับความรุนแรงจริง (ทั้ง 3 คือช่วง "ดี" อยู่แล้ว)
- `barColor()` เปลี่ยนจากสีทึบเป็น gradient (เข้ม→อ่อนของสีเดียวกัน) **ยังคงเปลี่ยนสีตามระดับ PM2.5 เหมือนเดิม** (ไม่ใช้สีฟ้าเดียวทุกแท่งแบบ mockup เพราะสีระดับฝุ่นเป็นข้อมูลจริงที่ต้องคงไว้) `.bar` มุมโค้งขึ้น (3px→6px), ระยะห่างแท่งกว้างขึ้นเล็กน้อย, ย้าย legend ขึ้นมาอยู่แถวเดียวกับหัวข้อกราฟ (`.chart-header`)
- แยก `<h1>` เป็น `.title-main`/`.title-sub` (ตัวอักษรเดิมทุกตัว แค่ห่อ span ให้ตัวหนา/จาง), เพิ่มจุดกระพริบ "live" เล็กๆ (ไม่มีข้อความใหม่ ไม่ประดิษฐ์คำ), ลิงก์ "ดูความแม่นยำจริง →" (ของเดิม ผูกกับ `/compare` อยู่แล้ว) ปรับสีให้เด่นขึ้น

**แก้ [frontend/src/components/AppHeader.vue](frontend/src/components/AppHeader.vue)** (ปรับสไตล์ล้วนๆ ไม่แตะ auth logic เลย):
- ชื่อแบรนด์เปลี่ยนเป็นสีฟ้า (`--auth-accent`) ตัวหนาขึ้น
- เพิ่ม class `.nav-tab` เฉพาะ 2 ลิงก์จริง (พยากรณ์/เปรียบเทียบ) + underline อัตโนมัติผ่าน Vue Router's `router-link-exact-active` (ไม่ track state เอง) ไม่เพิ่มแท็บที่ 3 เพราะไม่มีหน้าจริงรองรับ (mockup มี "Air Stations" แต่แอปนี้ไม่มี)
- "เข้าสู่ระบบ" ยังเป็น plain text เหมือนเดิม, "สมัครสมาชิก" เปลี่ยนเป็นปุ่ม pill สีฟ้าทึบ (คู่กับ Sign-In/Get-Started ของ mockup), ปุ่ม logout โค้งมนขึ้นให้เข้าชุดกัน

**ทดสอบ**: `npm run build` ผ่านไม่มี compile error, Selenium (headless Chrome, บังคับทั้ง light/dark color-scheme ผ่าน CDP) ยืนยัน: layout 2 คอลัมน์ถูกต้อง, การ์ดอ่านง่ายทั้ง light/dark mode, กราฟแท่ง gradient + จุดสีโมเดลยังชัดเจนถูกต้อง, badge อันดับและ status pill แสดงถูกต้อง, nav แสดง tab-underline ตาม route จริงและปุ่มสมัครสมาชิกเป็น pill, ไม่มี console error ใหม่, เช็ค `/compare` ยืนยันไม่ได้รับผลกระทบใดๆ (CSS token คนละชุดกับ ForecastView อยู่แล้ว)

**ไฟล์ที่แก้ในรอบนี้**: `frontend/src/views/ForecastView.vue`, `frontend/src/components/AppHeader.vue`

---

## 2026-07-16 (40) — เพิ่มระบบ login/register/guest/user profile

**ที่มา**: ผู้ใช้ขอฟีเจอร์บัญชีผู้ใช้ (login, register, guest access, user profile) ระบบเดิมไม่มีการยืนยันตัวตนเลย ทุกคนเข้าถึง `/api/forecast`/`/api/comparison*` ได้แบบ guest อยู่แล้วโดยปริยาย มีเอกสารออกแบบ schema เดิม `database_schema.md` (จาก proposal) ระบุตาราง `users` (email, password_hash, risk_group, age) ไว้แล้วแต่ไม่เคย implement จริง — risk_group (ทั่วไป/เด็ก-ผู้สูงอายุ/มีโรคประจำตัว) ตรงกับ `_USER_GROUP_TONE` ที่มีอยู่แล้วสำหรับปรับโทน NLG (แต่ไม่เคยผูกกับ user จริง)

ตัดสินใจร่วมกับผู้ใช้ก่อนเริ่ม (ผ่าน AskUserQuestion): (1) auth ใช้ Flask session cookie ไม่ใช่ JWT เพราะ SPA กับ API อยู่ origin เดียวกันเสมอทั้ง dev/prod (2) สมัครสมาชิกแบบง่าย ไม่มี email verification/forgot-password (ไม่มี SMTP infra) (3) risk_group เก็บไว้ใน profile ก่อนเฉยๆ **ยังไม่ผูกกับ NLG** เพราะกังวลเรื่อง Gemini quota เหมือนกับที่เคยตัดสินใจเลื่อน hourly NLG ไปแล้ว (ดู memory project_hourly_nlg_deferred)

**แก้ [app.py](app.py)**:
- เพิ่ม `SECRET_KEY` (สุ่มด้วย `secrets.token_hex(32)`) ลง `.env`, โหลดผ่าน `_load_env()` เดิม, ตั้ง `app.config["SECRET_KEY"]`/`SESSION_COOKIE_HTTPONLY`/`SESSION_COOKIE_SAMESITE="Lax"`/`PERMANENT_SESSION_LIFETIME=7 วัน`
- เพิ่ม import `session` (flask), `werkzeug.security.generate_password_hash/check_password_hash` (มากับ Flask อยู่แล้ว ไม่ต้องติดตั้งใหม่), `functools.wraps`, `re`
- เพิ่มตาราง `users` ใน `init_db()` (SERIAL/TEXT ตาม convention เดิมของไฟล์นี้ ไม่ใช่ UUID/native-TIMESTAMP แบบใน `database_schema.md`) — `risk_group` เก็บ Thai string ตรงๆ (คีย์เดียวกับ `_USER_GROUP_TONE`)
- section ใหม่ `===== User accounts (auth) =====`: DB helpers (`get_user_by_email`, `create_user`, `update_user_profile`, ฯลฯ — SELECT โปรไฟล์สาธารณะไม่ดึง `password_hash` เลย), `login_required` decorator
- 6 route ใหม่: `POST /api/auth/register` (auto-login), `POST /api/auth/login`, `POST /api/auth/logout`, `GET /api/auth/me`, `PUT /api/auth/profile`, `POST /api/auth/change-password` — ทุก route แยก `password_hash` ออกจาก response เสมอ, error message ตอน login ไม่บอกว่าอีเมลหรือรหัสผ่านผิดจุดไหน
- `/api/forecast`/`/api/comparison*` **ไม่แตะเลย** ยังคง unauthenticated 100% เหมือนเดิม = guest access

**เพิ่ม [frontend/src/auth.js](frontend/src/auth.js)** (ใหม่): reactive singleton (`reactive({user, status})`) แทนการติดตั้ง Pinia ใหม่ — โปรเจกต์นี้ไม่มี shared state เลยตอนนี้ (ทุก view ใช้ `ref()` แยกกัน) การเพิ่ม state เพียงก้อนเดียว (ใครล็อกอินอยู่) ไม่คุ้มที่จะเพิ่ม dependency ใหม่

**เพิ่ม [frontend/src/router/index.js](frontend/src/router/index.js)**: route ใหม่ `/login`, `/register` (`meta: guestOnly`), `/profile` (`meta: requiresAuth`) + `router.beforeEach` guard — `/` และ `/compare` ไม่แตะเลย (นี่คือ guest access ตัวจริง)

**บั๊กที่เจอระหว่างทดสอบและแก้แล้ว**: guard เดิมเช็ค `authState.status === 'idle'` ก่อนจะ `await fetchCurrentUser()` — แต่ `AppHeader.vue`'s `onMounted()` (อยู่นอก `<RouterView/>` เลยไม่ถูก router บล็อก) อาจเรียก `fetchCurrentUser()` ไปก่อนแล้วเปลี่ยน status เป็น `loading` ทำให้ guard ข้ามการ await ไปเจอ `authState.user` ที่ยังเป็น `null` แล้วรีไดเรกต์ไป `/login` ทั้งที่ล็อกอินอยู่จริง (race condition, เจอจริงตอน Selenium เทส hard-reload หน้า `/profile`) แก้โดยให้ guard เรียก `await fetchCurrentUser()` เสมอโดยไม่เช็ค status ก่อน (ฟังก์ชันมี in-flight promise cache อยู่แล้ว เรียกซ้ำราคาถูก)

**เพิ่มไฟล์ใหม่**: `frontend/src/components/AppHeader.vue` (nav bar ใหม่ทั้งหมด เพราะเดิมไม่มี shared header เลย), `frontend/src/views/{Login,Register,Profile}View.vue`, `frontend/src/assets/auth-theme.css` (token ชุดเล็กใช้เฉพาะ header + 3 view ใหม่ ไม่แตะ token เดิมของ ForecastView/CompareView), `scripts/test_auth_routes.py`

**ทดสอบ**:
- `python scripts/test_auth_routes.py` (marker email `@test-marker.invalid`, ลบตัวเองใน `finally`): 16/16 PASS (register→201, ไม่มี password_hash ใน response, ซ้ำอีเมล→409, รหัสผ่านผิด→401, login→200, GET/me ตรงกัน, PUT/profile อัปเดตจริง, เปลี่ยนรหัสผ่านผิด/ถูก→400/200, login ด้วยรหัสผ่านใหม่ยืนยันว่า hash หมุนจริง, logout แล้ว /me→401, client ใหม่ไม่มี cookie→401)
- `npm run build` ผ่าน, Selenium (headless Chrome) ครบตามแผน: guest โหลด `/` เห็น header แบบ guest ไม่มี console error, สมัครสมาชิกจริงแล้ว header เปลี่ยนเป็น logged-in, hard-reload `/profile` โหลดข้อมูลถูกต้อง (ยืนยันบั๊ก race condition ด้านบนแก้แล้วจริง), แก้ risk_group/age แล้ว reload ยืนยันค่าถูกบันทึกจริงใน Postgres, logout กลับเป็น guest, session ใหม่เข้า `/profile` ตรงๆ ถูก redirect ไป `/login`, session ใหม่เข้า `/` และ `/compare` ใช้งานได้เต็มที่โดยไม่ต้องล็อกอินและไม่มี console error
- ลบแถวทดสอบ (`@test-marker.invalid`) ออกจาก `users` หลังเทสเสร็จแล้ว

**ไฟล์ที่แก้ในรอบนี้**: `app.py`, `.env` (เพิ่ม `SECRET_KEY`), `frontend/src/auth.js` (ใหม่), `frontend/src/router/index.js`, `frontend/src/App.vue`, `frontend/src/main.js`, `frontend/src/components/AppHeader.vue` (ใหม่), `frontend/src/views/LoginView.vue`/`RegisterView.vue`/`ProfileView.vue` (ใหม่), `frontend/src/assets/auth-theme.css` (ใหม่), `scripts/test_auth_routes.py` (ใหม่)

**นอกขอบเขตตั้งใจ**: email verification/forgot-password (ไม่มี SMTP), rate-limiting การ login (ไม่มี dependency เดิมรองรับ), ผูก risk_group เข้ากับ NLG (แยกเป็นการตัดสินใจอนาคต)

---

## 2026-07-16 (39) — จัดระเบียบ comment ใน app.py (ไม่แตะ logic)

**ที่มา**: ผู้ใช้เปิด `app.py` ใน IDE แล้วขอให้จัดระเบียบ comment — ชี้ชัดว่าหมายถึง (1) เพิ่ม section header (`===== ... =====`) ให้ครบทุกส่วน ตอนนี้มีแค่บางส่วน (Dynamic Routing, PM25_LEVELS, NLG ฯลฯ) ส่วน config/DB/model loading/Flask routes ไม่มี (2) ตัด comment/docstring ที่ยาวซ้ำซ้อนให้กระชับ โดยเก็บใจความ "ทำไม" ที่สำคัญไว้ (3) ลบส่วนที่ไม่จำเป็น

**แก้ [app.py](app.py)** (เฉพาะ comment/docstring, logic โค้ดไม่เปลี่ยนเลย):
- เพิ่ม section header ให้ครบ 14 ส่วน: Config, .env/Database config, Flask app + model loading, Dynamic Routing, Database schema, Live data fetching + calibration, Feature engineering & prediction logging, PM25_LEVELS, NLG Safety Check, NLG Generation, NLG รายวัน, Core forecast pipeline, Flask routes, API routes
- แปล comment/docstring ที่เป็นภาษาอังกฤษล้วน (หลงเหลือจากตอนเขียนคนละช่วงเวลา) เป็นภาษาไทยให้สอดคล้องกับที่เหลือทั้งไฟล์: comment เรื่อง calibration ก่อน `CALIBRATION_PATH`, docstring ของ `calibrate_pm25()`, `serve_spa()`, `scheduled_forecast_job()`
- ตัด docstring ที่ยาวเวิ่นเว้อให้กระชับ โดยเก็บเหตุผลสำคัญไว้ครบ: `select_model()` (เหตุผล ε-greedy ต้องเช็ค None ทั้ง 2 ทาง อ้างอิง CHANGELOG (20)), `migrate_calibrated_actuals()`, `predict_with_prophet()`, `_find_keyword_occurrences()`/`_mentioned_levels()`/`validate_health_message()` (NLG safety check), `get_thai_now()` (เหตุผลเรื่อง timezone), `generate_nlg_explanation()`, comment เรื่องเลือกรุ่น Gemini/timeout
- ไฟล์ลดจาก 1288 → 1241 บรรทัด

**ทดสอบ**: `python -m py_compile app.py` ผ่าน ไม่มี syntax error (ตรวจแค่ syntax เพราะ import จริงจะ trigger side effect เช่น DB connection/genai client — ไม่ใช่ขอบเขตของการแก้รอบนี้ที่แตะแค่ comment)

**ไฟล์ที่แก้ในรอบนี้**: `app.py`

---

## 2026-07-16 (38) — เปลี่ยนสีจุดระบุโมเดล (LightGBM/XGBoost/Prophet) ในกราฟแท่งของ ForecastView.vue

**ที่มา**: ผู้ใช้ถามทำไมสีแท่งในกราฟไม่เหมือนกัน (ตอบว่าสีแท่งคือระดับ PM2.5 ตาม `barColor()` ไม่ใช่สุ่ม) แล้วถามต่อว่าสีน้ำเงินหมายถึงอะไร (`--excellent` = ดีมาก) จากนั้นชี้ให้เห็นว่าสีจุด `model-dot` ใต้แท่ง (บอกว่าชั่วโมงนั้นเลือกโมเดลไหนจาก Dynamic Routing) ใช้โทนสีเดิมซ้ำกับสีระดับ PM2.5 ของแท่งเกือบสนิท ทำให้แยกยาก: `--model-lightgbm` (`#2a78d6` น้ำเงิน) ชนกับ `--excellent` (`#1d68b3` น้ำเงิน), `--model-xgboost` (`#1baf7a` เขียวอมฟ้า) ใกล้เคียง `--good` (`#16a34a` เขียว), `--model-prophet` (`#eda100` เหลือง) ใกล้เคียง `--moderate`/`--unhealthy` (เหลือง/ส้ม) — ขอให้เปลี่ยนสีกลุ่มโมเดลให้ไม่ซ้ำโทนกับกราฟแท่ง

**แก้ [frontend/src/views/ForecastView.vue](frontend/src/views/ForecastView.vue)**: เปลี่ยนค่า `--model-lightgbm`/`--model-xgboost`/`--model-prophet` โดยเลือกจากชุดสี categorical มาตรฐาน (8 hue family: blue/green/magenta/yellow/aqua/orange/violet/red) เฉพาะ hue ที่ยังไม่ถูกใช้โดยสีระดับ PM2.5 (ซึ่งกินไปแล้ว 5 จาก 8: blue/green/yellow/orange/red) คือ violet และ magenta ส่วน aqua ของ XGBoost คงค่าเดิมไว้ (ยังพอแยกจาก green ได้ด้วยตา แม้ตระกูล hue จะใกล้กัน):
- LightGBM: น้ำเงิน `#2a78d6` → ม่วง `#4a3aa7` (dark mode `#9085e9`)
- XGBoost: เขียวอมฟ้า `#1baf7a` (คงเดิม, dark mode เพิ่มเป็น `#199e70`)
- Prophet: เหลือง `#eda100` → ชมพูอมม่วง `#e87ba4` (dark mode `#d55181`)

ตรวจสอบด้วยสคริปต์คำนวณระยะห่างสี (OKLab ΔE, จำลอง protanopia/deuteranopia แบบ Machado-Oliveira-Fernandes 2009) ยืนยันว่าคู่สีใหม่ทั้ง 3 อยู่ห่างจากสีระดับ PM2.5 ทั้ง 5 สี (ΔE ปกติ ≥ 16) และห่างจากกันเองพอสมควรสำหรับสายตาปกติ — **ข้อจำกัดที่ทราบ**: ในโหมดมืด คู่ aqua↔magenta (XGBoost↔Prophet) วัดได้ ΔE ต่ำมากภายใต้การจำลองสายตาบอดสีเขียว-แดง (deuteranopia) เพราะเหลือ hue ว่างจากชุด 8 สีมาตรฐานให้เลือกแค่ 3 ตัว (violet/magenta/aqua) ไม่พอจะแยกทุกคู่ให้ผ่านเกณฑ์ทั้งหมดพร้อมกัน — บรรเทาด้วย secondary encoding ที่มีอยู่แล้วในหน้านี้: legend มีชื่อโมเดลเป็นตัวอักษรกำกับทุกจุด และแต่ละจุดมี native `<title>` tooltip บอกชื่อโมเดลตอน hover (ไม่ต้องพึ่งสีอย่างเดียว)

**ทดสอบ**: `npm run build` ผ่านไม่มี compile error, screenshot (headless Selenium) ยืนยันจุด LightGBM (ม่วง) และ XGBoost (เขียวอมฟ้า) แยกจากสีแท่งเขียว/น้ำเงินได้ชัดเจนกว่าเดิมมาก ไม่มี console error

**ไฟล์ที่แก้ในรอบนี้**: `frontend/src/views/ForecastView.vue`

---

## 2026-07-16 (37) — จัดการ์ด "ช่วงเวลาที่อากาศดีที่สุด" ให้เรียงแนวนอน

**ที่มา**: ผู้ใช้ขอให้เปลี่ยน layout ของ 3 อันดับเวลาที่อากาศดีที่สุด (เพิ่มใน (34)) จากเรียงแนวตั้ง (บรรทัดต่อบรรทัด) เป็นเรียงแนวนอน

**แก้ [frontend/src/views/ForecastView.vue](frontend/src/views/ForecastView.vue)**: `.best-times-list` เปลี่ยนจาก `flex-direction: column` เป็น `flex-wrap: wrap` (แนวนอน), `.best-time-row` แต่ละอันดับเปลี่ยนจากแถวแนวนอน (badge+เวลา+ค่า เรียงข้างกัน) เป็นแท่งแนวตั้งขนาดย่อม (badge บน, เวลา, ค่า pm25 เรียงจากบนลงล่าง) วางเรียงข้างกัน 3 แท่ง — pattern เดียวกับ `.stat-tiles` ที่มีอยู่แล้วในไฟล์เดียวกัน ไม่ได้สร้าง pattern ใหม่

**ทดสอบ**: `npm run build` ผ่านไม่มี compile error, screenshot ยืนยันแสดงแนวนอนถูกต้อง ไม่มี console error

**ไฟล์ที่แก้ในรอบนี้**: `frontend/src/views/ForecastView.vue`

---

## 2026-07-16 (36) — เอาการ์ด "คำแนะนำสุขภาพ" ออกจาก ForecastView.vue

**ที่มา**: การ์ดนี้ที่เพิ่มใน (34) มีเนื้อหา checklist (เช่น "สวมหน้ากากป้องกันฝุ่น", "ปิดหน้าต่างช่วงฝุ่นสูง") ที่ผู้เขียนโค้ด (Claude) เขียนขึ้นเองจากวิจารณญาณ ไม่ได้อ้างอิงเอกสารทางการฉบับใดฉบับหนึ่งโดยตรง (แค่ให้ทิศทางสอดคล้องกับ `_HEALTH_MESSAGE_TEMPLATES` ใน `app.py` เท่านั้น) ผู้ใช้ถามที่มาแล้วขอให้เอาออก เพราะไม่ใช่ข้อมูลที่มาจากแหล่งที่เชื่อถือได้/ตรวจสอบแล้วเหมือนส่วนอื่นของหน้านี้ทั้งหมด

**แก้ [frontend/src/views/ForecastView.vue](frontend/src/views/ForecastView.vue)**: ลบฟังก์ชัน `healthTips(status)`, การ์ด template ทั้งก้อน, และ CSS ที่เกี่ยวข้อง (`.health-checklist`, `.health-item`, `.health-check`) ออกทั้งหมด ส่วนอื่นที่เพิ่มใน (34) (stat tile สภาพอากาศ, อันดับเวลาดีที่สุด 1-3) ยังคงอยู่ ไม่กระทบ

**ทดสอบ**: `npm run build` ผ่านไม่มี compile error, screenshot ยืนยันการ์ดหายไปแล้ว เลย์เอาต์ไม่เหลือช่องว่างแปลกๆ ไม่มี console error

**ไฟล์ที่แก้ในรอบนี้**: `frontend/src/views/ForecastView.vue`

---

## 2026-07-16 (35) — เปลี่ยนกราฟหลักของ ForecastView.vue กลับเป็นกราฟแท่ง (SVG line/area ดูยาก)

**ที่มา**: หลังจาก (34) เปลี่ยนกราฟพยากรณ์หลักเป็น SVG area/line chart ผู้ใช้ทดลองใช้จริงแล้วพบว่า **ดูยาก** โดยเฉพาะในวันที่ฝุ่นน้อย (~13-20 µg/m³) เพราะแกน Y ตรึงสูงสุดไว้ที่ 75 เสมอ (ตามแผนเดิม เพื่อให้เห็นเส้น threshold บนสุดตลอด) ทำให้เส้นกราฟดูแบนราบอยู่ล่างๆ มองไม่เห็นความผันผวนของค่าที่แท้จริง — ขอให้เปลี่ยนกลับเป็นกราฟแท่งแบบเดิมก่อน (กราฟแท่งสเกลตาม max ของ 24 ชั่วโมงนั้นเอง (`Math.max(...forecast, 37.5)`) จึงเห็นความต่างระหว่างชั่วโมงชัดกว่า)

**แก้ [frontend/src/views/ForecastView.vue](frontend/src/views/ForecastView.vue)**:
- ลบ SVG chart (`drawChart()`, `chartEl` ref, `THRESHOLDS` const, `modelHex()`, `nextTick` import) ทั้งหมด
- คืนกราฟแท่ง CSS เดิม (`.bars`/`.bar-col`/`.bar-value`/`.bar-track`/`.bar`/`.bar-label`/`.model-dot` + `maxPm` computed) กลับมาแทนที่
- **ส่วนอื่นที่เพิ่มใน (34) ยังคงอยู่ทั้งหมด** ไม่ได้ revert: stat tile สภาพอากาศ, อันดับเวลาดีที่สุด 1-3, การ์ดคำแนะนำสุขภาพ — กระทบแค่ส่วนกราฟจุดเดียว
- ไม่แตะ `app.py` เลยในรอบนี้ (field `current_weather` จาก (34) ยังใช้งานอยู่ตามเดิม)

**ทดสอบ** (Selenium จริง): `npm run build` ผ่านไม่มี compile error, screenshot หน้า `/` แสดงกราฟแท่งถูกต้อง เห็นความผันผวนราย ชม. ชัดเจนขึ้นกว่าเดิมมาก ไม่มี console error

**ไฟล์ที่แก้ในรอบนี้**: `frontend/src/views/ForecastView.vue` (เฉพาะส่วนกราฟ)

---

## 2026-07-16 (34) — ปรับดีไซน์ ForecastView.vue ให้มีลูกเล่นมากขึ้น (ใช้เฉพาะข้อมูลจริงที่มีอยู่)

**ที่มา**: ผู้ใช้ส่งภาพตัวอย่าง dashboard (mockup ชื่อ "AirSense AI") ขอให้ปรับหน้า `ForecastView.vue` ให้มีลูกเล่น/สวยงามขึ้น โดยระบุชัดว่าไม่ต้องเหมือนภาพตัวอย่างทั้งหมด เอาเฉพาะสิ่งที่ข้อมูลปัจจุบันรองรับ — ก่อนเริ่มถามผู้ใช้ยืนยัน 3 จุดสำคัญ: (1) กราฟทำ SVG เองไม่ใช้ library ใหม่ (2) การ์ด Model Accuracy (MAE/R² ต่อโมเดล) **ข้ามไปก่อน** (3) stat tile สภาพอากาศ **เพิ่ม** ได้ (ต้องแก้ backend เล็กน้อย)

**แก้ [app.py](app.py)**: `get_forecast()` เพิ่ม field `"current_weather"` เข้า response dict (`temp_c`/`humidity_pct`/`wind_speed_kmh` จากตัวแปร `current_weather` ที่ `fetch_current_weather()` ดึงมาใช้เป็น model feature อยู่แล้ว — ไม่มีการยิง API เพิ่ม ไม่แตะ route/schema อื่น)

**แก้ [frontend/src/views/ForecastView.vue](frontend/src/views/ForecastView.vue)**:
- เพิ่มแถว stat tile สภาพอากาศ (🌡️ อุณหภูมิ, 💧 ความชื้น, 🍃 ความเร็วลม) ในการ์ด PM2.5 ปัจจุบัน
- เปลี่ยนการ์ด "ช่วงเวลาที่อากาศดีที่สุด" จากบรรทัดเดียวเป็น**อันดับ 1-3** (`topBestTimes` คำนวณสดจาก `forecast[]` ที่มีอยู่แล้ว เรียงจาก pm25 น้อยสุด — ไม่ต้องแก้ backend เพิ่ม)
- แทนที่กราฟแท่ง CSS เดิมด้วย **SVG area/line chart** (เขียนเอง ไม่ใช้ library ใหม่) อิง pattern เดียวกับ `CompareView.vue`'s `drawChart()`: เส้นพยากรณ์ + gradient fill ใต้เส้น (เป็นภาพประกอบ ไม่ใช่ confidence band ทางสถิติ — ไม่มีข้อความไหนในหน้าเว็บเรียกมันว่า "confidence" เพราะไม่มีข้อมูลแบบนั้นจริง) + เส้นประอ้างอิงระดับ PM25_LEVELS (15/25/37.5/75, ตรงกับ `barColor()`/backend เป๊ะ) + จุดสีตามโมเดลที่ใช้จริงต่อชั่วโมง (คงการแสดง Dynamic Routing per-horizon ไว้เหมือนกราฟแท่งเดิม ผ่าน tooltip บนจุด)
- เพิ่มการ์ด "คำแนะนำสุขภาพ" — checklist 2-4 ข้อต่อระดับ PM2.5 (ฟังก์ชัน `healthTips(status)` รูปแบบเดียวกับ `statusClass`/`iconFor`/`bodyClassFor` ที่มีอยู่แล้ว เนื้อหาสอดคล้องทิศทางเดียวกับ `_HEALTH_MESSAGE_TEMPLATES` ฝั่ง backend ไม่ขัดแย้งกัน — เป็น UI copy ตายตัว ไม่ใช่ข้อมูลสร้างขึ้นเอง)
- CSS ใหม่ (`.stat-tiles`, `.best-times-list`/`.rank-badge`, `.health-checklist`) ใช้ design token (`--card-bg`/`--text`/`--muted`/`--accent` ฯลฯ) ที่มีอยู่แล้วทั้งหมด ไม่เพิ่ม token ใหม่

**สิ่งที่ตั้งใจไม่ทำตาม mockup** (ไม่มีข้อมูลจริงรองรับ จึงไม่ปั้นข้อมูลปลอมขึ้นมา): ตัวเลข confidence % ของ AI summary, AQI index แยกจาก PM2.5, การ์ด Model Accuracy (R² ต่อโมเดล — ผู้ใช้เลือกข้ามไปก่อน), sidebar หลายหน้า, notification bell, user profile widget, ตัวเลือกเปลี่ยนเมือง/สถานี (ระบบนี้มีสถานีเดียว)

**ทดสอบ** (Selenium จริงผ่าน headless Chrome + live server):
- `npm run build` ผ่านไม่มี compile error
- Backend: ยิง `/api/forecast` ยืนยัน `current_weather` ส่งค่าจริงกลับมาถูกต้องหลัง reloader restart อัตโนมัติ
- Screenshot หน้า `/` แสดงผลครบทุกส่วนใหม่ถูกต้อง (stat tiles, อันดับเวลาดีที่สุด 1-3, กราฟเส้น+พื้นที่+จุดสีโมเดล+เส้นประ threshold, checklist สุขภาพ) ไม่มี console error เลย
- ตรวจ dark mode: ยืนยันด้วย `window.matchMedia('(prefers-color-scheme: dark)').matches === true` และ computed `--card-bg` เปลี่ยนค่าถูกต้องตาม media query (พื้นหลัง "ท้องฟ้า" ยังดูคล้ายเดิมเพราะผูกกับระดับ AQI ไม่ใช่ light/dark theme — เป็น design เดิมของไฟล์ ไม่ใช่บั๊กใหม่)
- ไม่แตะ `CompareView.vue`/Postgres/route อื่นเลย ตามขอบเขตที่วางแผนไว้

**ไฟล์ที่แก้ในรอบนี้**: `app.py` (1 field เพิ่ม), `frontend/src/views/ForecastView.vue` (เขียนใหม่เกือบทั้งไฟล์)

---

## 2026-07-16 (33) — จัดระเบียบไฟล์ root: ย้าย backup .db เข้า backups/, ลบ __pycache__

**ที่มา**: ผู้ใช้ขอให้จัดระเบียบไฟล์ในโปรเจกต์ — เนื่องจากมีกฎเข้มว่าห้ามลบ/overwrite ไฟล์โมเดล/ข้อมูลเก่า และ `app.py` มี `sys.path.insert` อ้าง `scripts/` ตรงๆ (import `shap_analysis.py`) การจัดระเบียบแบบกว้าง (ย้าย `scripts/`/`models/` เป็น subfolder ตามหมวด) มีความเสี่ยงพังสูง จึงถามผู้ใช้ก่อนแล้วเลือกทำแค่ 2 จุดที่ปลอดภัย

**แก้**:
- สร้างโฟลเดอร์ `backups/` ใหม่ ย้าย `predictions.db` (ไฟล์ sqlite เดิม เลิกใช้แล้วหลัง migrate ไป Postgres ใน (30)) และ `predictions.db.bak_*` ทั้ง 4 ไฟล์เข้าไป
- พบว่ามี 4 script เก่าที่ hardcode path ไปที่ `predictions.db` ตรงๆ (`scripts/analyze_live_vs_backtest_by_horizon.py`, `scripts/backfill_xgboost.py`, `scripts/backfill_xgboost_v2.py`, `scripts/debug_prediction_log.py`) — เป็น script ยุคก่อน Postgres migration ที่ยังมีประโยชน์ถ้าอยากย้อนดูข้อมูล sqlite เดิม จึงอัปเดต `DB_PATH` ในทั้ง 4 ไฟล์ให้ชี้ไปที่ `backups/predictions.db` แทนที่จะปล่อยให้พังเงียบๆ ถ้าถูกเรียกใช้อีกในอนาคต
- ลบโฟลเดอร์ `__pycache__` ทั้ง root และ `scripts/` (build artifact ที่ Python สร้างใหม่เองได้เสมอ ไม่ใช่ output ที่ต้องเก็บ ลบได้ปลอดภัย 100% ไม่ขัดกับกฎ "ห้ามลบไฟล์โมเดล/ข้อมูล")

**ทดสอบ**: parse ไฟล์ทั้ง 4 ที่แก้ path ผ่าน `ast.parse()` ไม่มี syntax error

**ไฟล์ที่แก้ในรอบนี้**: ย้าย 5 ไฟล์เข้า `backups/` ใหม่, แก้ `DB_PATH` ใน 4 scripts, ลบ `__pycache__` 2 ที่

---

## 2026-07-16 (32) — Migrate dashboard เป็น Vue 3 SPA + PWA (manifest/service worker)

**ที่มา**: ผู้ใช้ชี้ว่า proposal ของโปรเจกต์เขียนไว้ว่าจะใช้ React หรือ Vue.js สำหรับพัฒนา Dashboard แบบ PWA — ข้อ (31) ก่อนหน้านี้ที่แนะนำให้คงเป็น vanilla JS (เพราะ PWA-readiness ไม่ขึ้นกับ framework) ใช้ไม่ได้แล้วเพราะเป็น requirement ทางวิชาการที่ต้องตรงกับ proposal ไม่ใช่แค่การตัดสินใจเชิงวิศวกรรม เลือก **Vue.js** (ไม่ใช่ React) เพราะทีมยังไม่เคยใช้ทั้งคู่มาก่อน และ syntax ของ Vue ใกล้เคียง HTML เดิมที่มีอยู่แล้วมากกว่า JSX ทำให้ port โค้ดเดิมง่ายกว่า สถาปัตยกรรมที่เลือก: **SPA เดียว + Vue Router** (ไม่ใช่ 2 mini-app แยก) และทำ PWA (manifest.json + service worker) **พร้อมกัน** ในรอบเดียวกับการ migrate ไม่แยกทำทีหลัง

**โครงสร้างใหม่**:
- `frontend/` — โปรเจกต์ Vue 3 + Vite แยกต่างหาก (`npm create vue@latest -- --router`), มี `package.json`/`node_modules` ของตัวเอง ไม่ปนกับ Python dependencies
- `frontend/src/views/ForecastView.vue` — port จาก `index.html`/`static/index.js` เดิม (จาก CHANGELOG (31)) ทุกฟังก์ชัน/logic เหมือนเดิม (statusClass, bodyClassFor, iconFor, barColor, modelColor) เปลี่ยนแค่การ render จาก manual DOM manipulation (`getElementById`, `createElement`) เป็น Vue reactive template + `v-for`
- `frontend/src/views/CompareView.vue` — port จาก `compare.html`/`static/compare.js` เดิม ส่วน SVG line chart (`drawChart`) **ยังคงเป็น imperative function เดิมทุกประการ** (สร้าง DOM ผ่าน `createElementNS` ในฟังก์ชันเดียว) เพราะแปลงเป็น Vue template reactive ล้วนๆ เสี่ยง bug โดยไม่จำเป็น แค่เรียกผ่าน template ref + `nextTick()` แทน — ส่วนที่เหลือ (overview bars, model breakdown chips) แปลงเป็น `v-for` ได้ตรงไปตรงมา
- `frontend/src/router/index.js` — `/` → ForecastView, `/compare` → CompareView (Vue Router, client-side, ไม่ reload หน้าเว็บ) **หมายเหตุ URL เปลี่ยน**: `/compare.html` เดิม → `/compare` ใหม่ (ไม่มี `.html` เพราะเป็น SPA route ไม่ใช่ไฟล์จริง)
- CSS design tokens (สี, การ์ด, skeleton ฯลฯ) จาก `static/index.css`/`static/compare.css` เดิม ย้ายเป็น **scoped style ในแต่ละ View** (ไม่ merge ระหว่าง 2 หน้า ด้วยเหตุผลเดิมจาก CHANGELOG (31): ตัวแปรชื่อคล้ายกันแต่ค่า/ความหมายต่างกันจริงระหว่าง 2 หน้า) CSS variable ที่เคยประกาศบน `:root`/`body.aqi-*` ย้ายไปประกาศบน class ของ wrapper div แต่ละหน้าแทน (`.forecast-page`, `.forecast-page.aqi-good` ฯลฯ) เพราะ Vue scoped style ทำงานกับ `:root`/`body` ไม่ได้ตรงๆ (ไม่ใช่ element ที่ Vue render)
- `vite-plugin-pwa` (`registerType: 'autoUpdate'`) generate `manifest.webmanifest` + service worker (`sw.js` + workbox runtime) อัตโนมัติตอน build — ตั้ง `navigateFallbackDenylist: [/^\/api\//]` เพื่อไม่ให้ service worker cache ผลพยากรณ์แบบ offline (ต้องเป็นข้อมูลสดเสมอ), icon PWA (`pwa-192x192.png`/`pwa-512x512.png`) generate ใหม่ด้วย Pillow (ธีมสีเดียวกับหน้าเว็บเดิม, ยังไม่มี branding จริงของทีม)

**แก้ [app.py](app.py)**:
- ลบ route `/`, `/index.html`, `/compare.html` แบบเดิม (`send_file`) เพิ่ม `FRONTEND_DIST` ชี้ไปที่ `frontend/dist` (build output จาก `npm run build`)
- เพิ่ม SPA catch-all route: `serve_spa(path)` คืนไฟล์จริงถ้ามีใน `frontend/dist` (เช่น `/assets/*.js`, `/manifest.webmanifest`, `/sw.js`) ไม่งั้น fallback ไป `index.html` เสมอ (ให้ Vue Router จัดการ client-side route เอง) — ยืนยันด้วยการทดสอบจริงว่า Werkzeug จัดลำดับ static route (เช่น `/api/forecast`) ให้ priority เหนือ catch-all แบบ `<path:path>` เสมอไม่ว่าจะประกาศก่อนหรือหลัง ไม่มีทางชน `/api/*`
- `/index.html`, `/compare.html` เดิมเปลี่ยนเป็น redirect (302) ไป `/` และ `/compare` ตามลำดับ กัน bookmark เก่าพัง

**Archive ไฟล์เดิม (ไม่ลบทิ้ง ตาม convention)**: `index.html`, `compare.html`, `static/index.css`, `static/index.js`, `static/compare.css`, `static/compare.js` ย้ายไปที่ `pre_vue_frontend_backup/` ทั้งหมด

**ทดสอบ** (Selenium จริงผ่าน headless Chrome + live server):
- `npm run build` ผ่านไม่มี compile error, ได้ `frontend/dist` ครบ (`index.html`, `assets/*.js/css`, `manifest.webmanifest`, `sw.js`, `workbox-*.js`, `pwa-*.png`)
- ทุก route ตอบ 200: `/`, `/compare`, `/manifest.webmanifest`, `/sw.js`, `/api/forecast`; `/compare.html` ตอบ 302 redirect ไป `/compare` ถูกต้อง
- Screenshot ทั้ง 2 หน้าหลังโหลดข้อมูลจริงแล้ว **เหมือนกับก่อน migrate (CHANGELOG 31) ทุกพิกเซล** ไม่มี regression ทางภาพเลย
- Browser console ไม่มี severe error เลย (แม้แต่ favicon 404 เดิมก็หายไปเพราะตอนนี้มี favicon ของ Vue app เอง)
- ทดสอบ client-side routing จริง: คลิกลิงก์ "ดูความแม่นยำจริง →" จาก `/` ไป `/compare` โดย URL เปลี่ยนแต่ไม่มี full page reload ได้ถูกต้อง, คลิก "← กลับหน้าแรก" กลับมาที่ `/` ได้ถูกต้อง
- ทดสอบ interactivity: เปลี่ยน horizon dropdown เป็น t+12 แล้ว MAE stat อัปเดตค่าใหม่ถูกต้อง (3.37)
- ทดสอบ PWA infrastructure: `<link rel="manifest">` ชี้ไปที่ `/manifest.webmanifest` ถูกต้อง, `navigator.serviceWorker.getRegistrations()` ยืนยันว่า service worker ลงทะเบียนและ `state: "activated"` สำเร็จ

**Dev workflow ใหม่ที่ต้องรู้**: แก้ frontend ตอนนี้ต้องรัน `cd frontend && npm run build` ก่อนถึงจะเห็นผลใน Flask (ต่างจากเดิมที่แก้ `.html`/`static/*.js` แล้วรีเฟรชเห็นผลทันที) หรือใช้ `npm run dev` (Vite dev server, มี proxy `/api` ไป `localhost:5000` ใน `vite.config.js`) สำหรับ dev แบบ hot-reload ระหว่างพัฒนา

**ไฟล์ที่แก้ในรอบนี้**: `app.py` (routing), ไฟล์ใหม่ทั้งหมดใน `frontend/`, ย้าย 6 ไฟล์เดิมไป `pre_vue_frontend_backup/`

---

## 2026-07-16 (31) — แยก CSS/JS ของ index.html/compare.html ออกเป็นไฟล์ static แยกต่างหาก

**ที่มา**: ผู้ใช้ขอให้จัดไฟล์ `index.html`/`compare.html` ให้เป็นระเบียบ ทั้งสองไฟล์เดิม inline ทั้ง `<style>` และ `<script>` ไว้ในไฟล์เดียวกัน (index.html ~545 บรรทัด, compare.html ~768 บรรทัด) ทำให้หาโค้ดยากและปนกันระหว่าง markup/style/behavior — ไม่ได้เปลี่ยนไปใช้ framework ใดๆ (ดูเหตุผลที่ตัดสินใจไม่ใช้ Vue.js: PWA-readiness ไม่ขึ้นกับ JS framework เลย เพิ่ม manifest.json/service worker เข้า vanilla JS เดิมได้ตรงๆ)

**แก้**:
- สร้างโฟลเดอร์ `static/` ใหม่ (Flask serve ให้อัตโนมัติที่ `/static/<file>` เพราะ `app = Flask(__name__)` ใช้ static folder ค่าเริ่มต้นอยู่แล้ว ไม่ต้องเพิ่ม route เอง)
- ย้าย `<style>` ของ `index.html` → `static/index.css`, `<script>` → `static/index.js` แบบ **lift-and-shift** (เนื้อหาเหมือนเดิมทุกตัวอักษร ไม่ได้ merge/dedupe กับของ compare.html แม้จะมีตัวแปร CSS/ฟังก์ชัน JS ที่คล้ายกันอยู่ เพราะบางค่าที่ดูซ้ำกัน เช่น `--moderate`/`--good` จริงๆ แล้วสื่อความหมายคนละอย่างระหว่าง 2 หน้า — ผสมรวมจะเสี่ยงพังโดยไม่ตั้งใจ เกินสโคปของงาน "จัดให้เป็นระเบียบ")
- ย้าย `<style>`/`<script>` ของ `compare.html` → `static/compare.css`/`static/compare.js` แบบเดียวกัน
- `index.html`/`compare.html` เหลือแค่ markup + `<link rel="stylesheet">` + `<script src="...">` (index.html เหลือ ~96 บรรทัด, compare.html เหลือ ~101 บรรทัด)

**ทดสอบ** (Selenium screenshot จริง ผ่าน headless Chrome + live server, เทียบก่อน/หลัง):
- ทั้ง 2 หน้าโหลดสำเร็จ (HTTP 200) ทั้ง `.html` และไฟล์ static ใหม่ทั้ง 4 ไฟล์
- Browser console ไม่มี error ใหม่เกิดขึ้นเลย (มีแค่ favicon 404 เดิมที่ไม่เกี่ยวข้อง)
- Screenshot ของทั้ง 2 หน้าหลังโหลดข้อมูลจริงแล้ว หน้าตา สี badge สถานะ กราฟแท่ง/เส้น legend เหมือนเดิมทุกจุด ไม่มี regression ทางภาพ

**ไฟล์ที่แก้ในรอบนี้**: `index.html`, `compare.html` (ตัดเนื้อหาลง), ไฟล์ใหม่ `static/index.css`, `static/index.js`, `static/compare.css`, `static/compare.js`

---

## 2026-07-16 (30) — ย้าย `predictions.db` จาก SQLite ไป PostgreSQL

**ที่มา**: ผู้ใช้ตัดสินใจย้ายฐานข้อมูลของระบบ demo เดิมทั้งหมดจาก SQLite ไปเป็น PostgreSQL โดยตรง (แทนที่ `app.py` เดิม ไม่ได้สร้างระบบใหม่แยกต่างหาก) เลือก PostgreSQL 18 (ที่ติดตั้งอยู่บนเครื่องอยู่แล้ว) แทน MySQL 8.0 (ที่ก็ติดตั้งอยู่เช่นกัน) เพราะ schema ที่ร่างไว้ล่วงหน้าใน `database_schema.md` (สำหรับแผน PWA/DB ที่ถูกพักไว้) ออกแบบโดยอิง Postgres เป็นหลัก

**Backup**: `predictions.db` สำรองไว้ที่ `predictions.db.bak_20260716_000516` ก่อนแตะอะไรเลย (ตาม convention เดิมของโปรเจกต์)

**Migration**: สร้าง database ใหม่ชื่อ `pm25_forecast` ใน Postgres แล้วเขียน script คัดลอกข้อมูลทั้งหมดจาก SQLite ไปยัง Postgres แบบรักษา `id` เดิมทุกแถว (`prediction_log` 1,728 แถว, `daily_nlg_log` 1 แถว) ยืนยันด้วยการเทียบ row count และ checksum (`SUM()`) ของคอลัมน์ตัวเลขสำคัญทุกตัวตรงกันเป๊ะก่อนเริ่มแก้ `app.py`

**บั๊กที่เจอระหว่าง migrate (แก้ก่อนใช้งานจริง)**: รอบแรกสร้าง schema Postgres ด้วย type `REAL` ตรงชื่อกับ SQLite แต่ **`REAL` ใน Postgres คือ 4-byte single precision** ในขณะที่ `REAL` ใน SQLite คือ 8-byte double เสมอ ทำให้ checksum ของ `predicted_pm25`/`actual_pm25_calibrated` เพี้ยนไปจากต้นฉบับ (เช่น `40650.4` → `40650.402`) — แก้โดยเปลี่ยนทุกคอลัมน์ตัวเลขที่เคยเป็น `REAL` เป็น `DOUBLE PRECISION` แทน แล้ว migrate ใหม่ ยืนยัน checksum ตรงกันในระดับ floating-point tolerance ปกติ (~1e-10)

**แก้ [app.py](app.py)**:
- ลบ `import sqlite3`, เพิ่ม `import psycopg2`
- ลบ `DB_PATH` constant, เพิ่ม `_load_env()` (ขยายจาก logic เดิมที่อ่านแค่ `GEMINI_KEY` ให้อ่านทุก key ใน `.env` เข้า dict เดียว) + `get_db_connection()` ที่เปิด psycopg2 connection ใหม่ทุกครั้งที่เรียก (ตาม pattern เดิมของ `sqlite3.connect()` ต่อ call ไม่เปลี่ยนเป็น connection pool เพื่อลด risk ตอน migrate)
- ทุกจุดที่เคยใช้ `sqlite3.connect(DB_PATH)` (`compute_rolling_r2`, `init_db`, `migrate_calibrated_actuals`, `log_predictions`, `backfill_actuals`, `get_daily_nlg_row`, `save_daily_nlg_row`, `api_comparison`, `api_comparison_overview`) เปลี่ยนเป็น `get_db_connection()` + `cursor()` ตาม psycopg2 API, placeholder `?` → `%s` ทั้งหมด
- `init_db()`: `AUTOINCREMENT` → `SERIAL`, `REAL` → `DOUBLE PRECISION`, ลบ logic `PRAGMA table_info` + `ALTER TABLE ADD COLUMN actual_pm25_calibrated` (เป็น schema-evolution เฉพาะของ SQLite ที่ไม่จำเป็นอีกต่อไปเพราะ Postgres schema ใหม่มีคอลัมน์นี้ตั้งแต่แรก)
- `log_predictions()`: `INSERT OR IGNORE` (SQLite) → `INSERT ... ON CONFLICT (predicted_at_utc, horizon) DO NOTHING` (Postgres)
- `save_daily_nlg_row()`: ส่ง `bool(used_fallback)` ตรงๆ แทน `int(used_fallback)` เพราะ Postgres มี native `BOOLEAN` type (SQLite ไม่มี เก็บเป็น 0/1)

**แก้ test scripts** (ทั้งคู่เคยใช้ pattern `app.DB_PATH = temp_sqlite_path` ซึ่งใช้ไม่ได้แล้วเพราะไม่มี `DB_PATH` อีกต่อไป):
- [scripts/test_dynamic_routing.py](scripts/test_dynamic_routing.py): เปลี่ยนจากสร้าง temp SQLite file (ที่ schema เพี้ยนจากของจริง ขาดคอลัมน์ `actual_pm25_calibrated` — บั๊กที่มีมาก่อนหน้านี้แล้ว ดู CLAUDE.md) เป็นเขียน/ลบแถวทดสอบลง `prediction_log` จริงใน Postgres โดยใช้ `predicted_at_utc` ปี 2100 เป็น marker (ไม่ชนข้อมูลจริง) ลบทิ้งเองทั้งก่อน/หลังรัน (รวมใน `finally` กันแถวค้างถ้า assertion พังกลางคัน) **การแก้นี้แก้บั๊ก schema เดิมไปในตัวด้วย** เพราะตอนนี้ insert เข้าตารางจริงที่มีคอลัมน์ครบ
- [scripts/test_daily_nlg.py](scripts/test_daily_nlg.py): เปลี่ยนจาก temp SQLite file เป็นเขียน/ลบแถวทดสอบลง `daily_nlg_log` จริง โดยใช้ `date_thai` ปี **2199** เป็น marker (เลือกปีนี้แทนปี 2100/9999 เพราะ test ต้อง mock `get_thai_now()` ให้คืนค่า `pd.Timestamp` จริง ซึ่งมีขอบเขตสูงสุดราวปี 2262 เท่านั้น)
- ลบไฟล์ temp SQLite เก่าที่ค้างอยู่ (`scripts/_test_routing_temp.db`) เพราะไม่มีโค้ดอ้างอิงแล้ว

**ทดสอบ**:
- `scripts/test_dynamic_routing.py`: ผ่านทั้ง 2 test (circuit breaker, rolling R2 ทั้ง 2 ทิศทาง) — และเป็นครั้งแรกที่ Test 2 ผ่านโดยไม่มี `no such column: actual_pm25_calibrated` error (บั๊กเดิมที่บันทึกไว้ใน CLAUDE.md ว่า schema ของ test เพี้ยนจากของจริง หายไปเองเพราะตอนนี้ insert เข้า schema จริง)
- `scripts/test_daily_nlg.py`: ผ่านครบทั้ง 9 เคส (UTC-crossing, no-duplicate, cross-day, avg calculation, hour>=6 edge cases, placeholder, real Gemini call)
- รัน `app.py` จริงผ่าน Postgres แล้วยิง `/api/forecast` — ได้ครบ 24 จุดพยากรณ์ ไม่มี error, log เข้า Postgres ถูกต้อง (1728 → 1752 แถว ตรงกับ +24 ที่คาดไว้), `daily_summary` แสดง placeholder ถูกต้อง (ยังไม่ถึง 06:00 น.)
- `/api/comparison?horizon=1&model=all` และ `/api/comparison/overview` ทำงานถูกต้อง ตัวเลข MAE/bias/basis สมเหตุสมผล (overall MAE=3.06, basis=live, n_matched=1572)

**ไฟล์ที่แก้ในรอบนี้**: `app.py` (DB layer ทั้งหมด), `scripts/test_dynamic_routing.py`, `scripts/test_daily_nlg.py`, `.env` (เพิ่ม `POSTGRES_HOST/PORT/DB/USER/PASSWORD`, ไม่ commit ค่าจริงเข้า repo), ลบ `scripts/_test_routing_temp.db`

**สิ่งที่ยังไม่เปลี่ยน**: `index.html`/`compare.html` ไม่ถูกแตะเลย เพราะ API response shape เดิมทุกประการ, ไฟล์ `predictions.db` (SQLite) เดิมยังเก็บไว้เป็น fallback ไม่ลบทิ้ง

---

## 2026-07-15 (29) — maybe_generate_daily_nlg(): trigger ตั้งแต่ 06:00 น. เป็นต้นไป (>=6) แทนชั่วโมง 6 พอดี (==6)

**ที่มา**: (27) ตั้งเงื่อนไข trigger เป็น `hour == 6` พอดี — ถ้า `app.py` ไม่ได้รันช่วง 06:00-06:59 น. พอดี (เช่น เปิดเครื่องแค่บางช่วงของวัน) จะพลาด NLG รายวันไปทั้งวัน ต้องรอวันถัดไป ผู้ใช้ขอให้เปลี่ยนเป็น trigger ได้ตั้งแต่ 06:00 น. เป็นต้นไปจนกว่าจะมีแถวของวันนั้นแล้ว

**แก้ [app.py](app.py)**: `maybe_generate_daily_nlg()` เปลี่ยนเงื่อนไขจาก `thai_now.hour != 6: return` เป็น `thai_now.hour < 6: return` — เงื่อนไข "ยังไม่มีแถวของ `date_thai` นี้" (กัน generate ซ้ำ) ไม่เปลี่ยน

**พฤติกรรมที่เปลี่ยนไปจากเดิม** (ยืนยันด้วย [scripts/test_daily_nlg.py](scripts/test_daily_nlg.py)):

| เวลาไทย (hour) | พฤติกรรมเดิม (`==6`) | พฤติกรรมใหม่ (`>=6`) |
|---|---|---|
| < 06:00 (เช่น 05:59) | ไม่ trigger | ไม่ trigger (**ไม่เปลี่ยน**) |
| 06:00-06:59 (hour=6) | trigger | trigger (**ไม่เปลี่ยน**) |
| 07:30 (hour=7), ยังไม่มีแถวของวันนั้น | ไม่ trigger (ต้องรอวันถัดไป) | **trigger ทันที** (เปลี่ยน — คือจุดที่ตั้งใจแก้) |
| 09:00 (hour=9), ยังไม่มีแถวของวันนั้น | ไม่ trigger | **trigger ทันที** (เปลี่ยน) |
| 23:00 (hour=23), ยังไม่มีแถวเลยทั้งวัน | ไม่ trigger | **trigger** (เปลี่ยน — ช้ามากแต่ดีกว่าพลาดไปทั้งวัน) |

**ทดสอบ** (`scripts/test_daily_nlg.py`, เพิ่มเป็น 9 เคส): เคสเดิม 1-4, 6, 7 ไม่เปลี่ยนพฤติกรรม ผ่านเหมือนเดิม เคส 5 (edge case) ปรับ assertion ส่วน "07:30 น. ไทย" จากคาดหวัง "ไม่ trigger" เป็น "trigger" ตามพฤติกรรมใหม่ (ส่วน "05:59 น. ไทย" ยังคาดหวัง "ไม่ trigger" เหมือนเดิม) เพิ่มเคส 8 (09:00 น. ไทย, ไม่มีแถวเลย → trigger ทันที) และเคส 9 (23:00 น. ไทย, ไม่มีแถวเลยทั้งวัน → ยัง trigger ได้) ทั้ง 9 เคสผ่านหมด — regression `test_nlg_safety_check.py` ผ่านครบเหมือนเดิม

**ไฟล์ที่แก้ในรอบนี้**: `app.py` (`maybe_generate_daily_nlg()` เงื่อนไขเดียว), `scripts/test_daily_nlg.py` (แก้เคส 5 + เพิ่มเคส 8, 9)

---

## 2026-07-15 (28) — เพิ่มการ์ด "สรุปวันนี้" แสดง NLG รายวันใน index.html (ยังไม่มีตอน (27))

**ที่มา**: (27) ทำ backend ของ NLG รายวันเสร็จแล้ว (field `daily_summary` ใน `/api/forecast`) แต่ลืมต่อฝั่งแสดงผล — ผู้ใช้เปิดหน้าเว็บแล้วไม่เห็นอะไรเลย

**แก้ [index.html](index.html)**:
- เพิ่มการ์ดใหม่ "📋 สรุปวันนี้ (ค่าเฉลี่ยพยากรณ์ 24 ชม., อัปเดตตอน 06:00 น.)" วางไว้ระหว่างการ์ด PM2.5 ปัจจุบันกับการ์ดช่วงเวลาอากาศดีที่สุด แสดงค่าเฉลี่ย (`daily_avg_pm25`) + ข้อความอธิบาย (`explanation`) จาก field `daily_summary` ที่มีอยู่แล้วใน response
- เพิ่ม CSS class ใหม่ `.daily-summary-title`/`.daily-summary-value`/`.daily-summary-unit`/`.daily-summary-text` (สไตล์เดียวกับการ์ดอื่นๆ ในหน้า ไม่ได้สร้าง pattern ใหม่)
- `renderForecast(data)`: เพิ่ม logic แสดงผล `dailyAvgPm`/`dailySummaryText` จาก `data.daily_summary` — ถ้า `daily_avg_pm25` เป็น `null` (ยังไม่ถึง 06:00 น. ของวันนี้) แสดง `-` แทน และใช้ placeholder `"กำลังเตรียมข้อมูลสำหรับวันนี้"` ถ้า `explanation` ว่าง

**ทดสอบ** (Selenium screenshot, headless Chrome จริง ผ่าน live server):
- สถานะ placeholder (ยังไม่มีแถวของวันนี้): การ์ดแสดง `- µg/m³` และ `"กำลังเตรียมข้อมูลสำหรับวันนี้"` ถูกต้อง ไม่มี JS error (นอกจาก favicon 404 เดิมที่ไม่เกี่ยวข้อง)
- สถานะมีข้อมูลจริง: insert แถวทดสอบชั่วคราวใน `daily_nlg_log` (`date_thai=2026-07-15`, avg=18.2) ผ่าน API เดิม → การ์ดแสดง `18.2 µg/m³` ถูกต้อง → **ลบแถวทดสอบออกทันทีหลัง screenshot** เพื่อไม่ให้ปนกับข้อมูลจริงที่จะถูกสร้างตอน 06:00 น. ของวันจริง

**ไฟล์ที่แก้ในรอบนี้**: `index.html` (การ์ดใหม่ + CSS + JS render logic)

---

## 2026-07-15 (27) — NLG รายวัน: เปลี่ยนจากเรียก Gemini ทุกชั่วโมงเป็นวันละ 1 ครั้งตอน 06:00 น. เวลาไทย

**ที่มา**: (24)/(25) ทำ `generate_nlg_explanation()` แบบต่อ horizon/ชั่วโมงไว้แล้ว — ผู้ใช้ขอให้เพิ่มอีกโหมดที่สรุปทั้งวันครั้งเดียว (ค่าเฉลี่ยพยากรณ์ 24 ชม.) trigger เฉพาะตอนเช้าแทนทุกชั่วโมง เพื่อลดจำนวนครั้งที่เรียก Gemini ต่อวันจาก 24 (ถ้า integrate เต็มรูปแบบ) เหลือ 1

**แก้ [app.py](app.py)**:
- `init_db()`: เพิ่มตาราง `daily_nlg_log` (`date_thai TEXT UNIQUE`, `daily_avg_pm25`, `explanation_text`, `used_fallback`, `generated_at_utc`)
- `get_thai_now()`: คืนเวลาไทยผ่าน `pd.Timestamp.now(tz="UTC").tz_convert("Asia/Bangkok")` — ใช้ pattern `.tz_convert()` เดียวกับที่มีอยู่แล้วทั่วไฟล์ (ไม่ใช่คำนวณ offset +7 เอง) เป็นฟังก์ชันแยกต่างหากเพื่อให้ test monkeypatch เวลาได้ตรงๆ
- `generate_daily_nlg_explanation(forecast, feature_row)`: ค่าเฉลี่ยจาก `forecast[]` ทั้ง 24 จุด → `pm25_status()` → SHAP จาก horizon t+12 (index 11) เป็นตัวแทนอธิบายปัจจัยหลักของวัน → prompt ใหม่ `_build_daily_nlg_prompt()` (พูดถึง "ค่าเฉลี่ยทั้งวัน"/"แนวโน้มตลอดวัน") → `_call_gemini()` → `validate_health_message()` เหมือนเดิมทุกประการ (reuse logic เดิมทั้งหมด ไม่ทำ safety check ใหม่)
- แยก `_format_shap_factor_lines()` ออกจาก `_build_nlg_prompt()` เดิม (refactor เล็กน้อย ไม่เปลี่ยนพฤติกรรม) ให้ `_build_daily_nlg_prompt()` เรียกใช้ร่วมกัน
- `maybe_generate_daily_nlg(forecast, feature_row)`: เรียกจาก `get_forecast()` ทุกครั้งที่ทำงาน (ทั้ง scheduler รายชั่วโมงและตอนมีคนเปิดเว็บ) แต่จะเรียก Gemini จริงก็ต่อเมื่อ `get_thai_now().hour == 6` **และ** ยังไม่มีแถวของ `date_thai` นั้นใน `daily_nlg_log` — ครอบ try/except กันไม่ให้ error จุดนี้ทำให้ `/api/forecast` ทั้งเส้นพัง
- `get_today_daily_nlg()`: อ่านแถวของวันนี้ (เวลาไทย) จาก DB อย่างเดียว ไม่เรียก Gemini — ถ้ายังไม่มีแถวเลย (เช่น deploy ก่อนถึง 06:00 น.) คืน placeholder `"กำลังเตรียมข้อมูลสำหรับวันนี้"` แทน error
- `get_forecast()`: เพิ่ม `maybe_generate_daily_nlg(forecast, X_now)` หลัง `backfill_actuals()`, response เพิ่ม field `daily_summary` (จาก `get_today_daily_nlg()`)

**จุดที่ต้องระวังเรื่องข้ามวัน (ตามที่ผู้ใช้เตือนไว้)**: 06:00 น. เวลาไทย = 23:00 UTC ของวันก่อนหน้า — `date_thai` ทุกจุดคำนวณจาก `get_thai_now().strftime("%Y-%m-%d")` (เวลาไทยหลังแปลงแล้วเสมอ) ไม่มีจุดไหนใช้วันที่ฝั่ง UTC ตรงๆ เลย

**ทดสอบ** ([scripts/test_daily_nlg.py](scripts/test_daily_nlg.py), 7 เคส):
1. **23:00 UTC 14 ก.ค. (=06:00 น. ไทย 15 ก.ค.)** → บันทึก `date_thai="2026-07-15"` ถูกต้อง ยืนยันไม่มีแถว `"2026-07-14"` ผิดๆ เกิดขึ้น
2. เรียกซ้ำ **23:45 UTC วันเดิม** → ไม่เรียก Gemini ซ้ำ (0 ครั้ง, ใช้ cache จาก DB)
3. ข้ามวัน **23:00 UTC 15 ก.ค. (=06:00 น. ไทย 16 ก.ค.)** → trigger ใหม่ สร้างแถว `"2026-07-16"` — เคสนี้บังเอิญ mock message ("ดี") ไม่ตรงกับระดับจริง ("ดีมาก") ทำให้เห็น safety-check ทำงานจริง (`used_fallback=True`) ยืนยันว่า integration เชื่อมกันจริง ไม่ใช่ pass ทุกครั้งแบบไม่มีความหมาย
4. `daily_avg_pm25` คำนวณตรงกับค่าเฉลี่ยจริงของ `forecast[]` (15.6 ตรงตามคาด)
5. Edge case: **22:59 UTC (=05:59 น. ไทย)** และ **00:30 UTC วันถัดไป (=07:30 น. ไทย)** — ทั้งคู่ไม่ trigger (นอกช่วง 06:00-06:59 น.)
6. `get_today_daily_nlg()` คืน placeholder ถูกต้องเมื่อยังไม่มีแถวของวันนั้นเลย
7. **เรียก Gemini จริง (ไม่ mock)** ด้วยค่าเฉลี่ยจริง 18.2 (ระดับ "ดี") — ได้ข้อความ: *"วันนี้คุณภาพอากาศโดยรวมอยู่ในเกณฑ์ดี โดยค่าฝุ่นมีแนวโน้มลดลงจากสภาพอากาศที่มีความชื้นช่วยดักจับฝุ่นไว้ แม้จะมีผลตกค้างจากค่าฝุ่นสะสมในช่วงครึ่งวันก่อนหน้าอยู่บ้าง แต่ภาพรวมตลอดทั้งวันยังถือว่าปลอดภัยต่อสุขภาพครับ"* ผ่าน safety check ตรงๆ ไม่ fallback

ทั้ง 7 เคสผ่านหมด — regression เดิม (`test_nlg_safety_check.py`) ผ่านครบเหมือนเดิม, live server (`DEBUG=True` auto-reload) ยืนยันแล้วว่า `/api/forecast` คืน field `daily_summary` ใหม่ถูกต้อง (placeholder เพราะยังไม่ถึง 06:00 น. ของวันนี้)

**ไฟล์ที่แก้ในรอบนี้**: `app.py` (`init_db()` เพิ่มตาราง, ฟังก์ชันใหม่ 6 ตัว: `get_thai_now`/`get_daily_nlg_row`/`save_daily_nlg_row`/`generate_daily_nlg_explanation`/`maybe_generate_daily_nlg`/`get_today_daily_nlg`, `_build_daily_nlg_prompt`, `_format_shap_factor_lines` refactor, `get_forecast()` เพิ่ม 2 บรรทัด), `scripts/test_daily_nlg.py` (ใหม่)

---

## 2026-07-15 (26) — threshold แจ้งเตือนมลพิษสูงแยกตามโมเดล (ไม่ calibrate จาก LightGBM แล้วใช้กับทุกโมเดล)

**ที่มา**: ระหว่างสรุปผล MAE/RMSE/R² ของ 3 โมเดลให้ผู้ใช้ พบว่า XGBoost ไม่เคยผ่าน sample-weight retrain แก้ bias ช่วงมลพิษสูงเหมือน LightGBM (คนละรอบงานกัน — LightGBM แก้ตั้งแต่ (2026-07-09) ก่อน XGBoost จะเข้าระบบ (2026-07-12 (1))) ตรวจโค้ดพบว่า `EFFECTIVE_THRESHOLD_BY_HORIZON` เดิมคำนวณจาก `bias_high_weighted` ของ LightGBM เพียงตัวเดียว แต่ถูกใช้เทียบกับค่าพยากรณ์ **ทุก horizon ไม่ว่า Dynamic Routing จะเลือกโมเดลไหน** — โค้ดเดิมมีคอมเมนต์ยอมรับ trade-off นี้ไว้ตรงๆ ("ยังไม่มี bias correction แยกสำหรับ XGBoost/Prophet") ผู้ใช้ขอให้แก้เป็นแยกตามโมเดลจริง โดยยังไม่ต้อง retrain XGBoost ก่อน

**สิ่งที่ทำ**:
1. [scripts/eval_xgb_bias_high_by_horizon.py](scripts/eval_xgb_bias_high_by_horizon.py) (ใหม่): โหลด XGBoost ที่ deploy จริง (`models/xgb_pm25_multistep_20260712_191350.pkl`) มา inference ซ้ำบน test set เดิม (ไม่ retrain) คำนวณ `bias_high`/`mae_high` ต่อ horizon เฉพาะกรณี PM2.5 จริง > 37.5 (นิยามเดียวกับที่ `retrain_weighted_v2.py` ใช้กับ LightGBM) บันทึกที่ `models/xgb_bias_high_by_horizon_20260712_191350.csv`
2. [app.py](app.py): เปลี่ยน `BIAS_HIGH_BY_HORIZON`/`EFFECTIVE_THRESHOLD_BY_HORIZON` (dict เดียว, LightGBM เท่านั้น) เป็น `BIAS_HIGH_BY_MODEL`/`EFFECTIVE_THRESHOLD_BY_HORIZON_BY_MODEL` (dict ซ้อน แยกตามชื่อโมเดล):
   - `lightgbm`: bias จาก `BIAS_CSV_PATH` เดิม (ไม่เปลี่ยนพฤติกรรม)
   - `xgboost`: bias จากไฟล์ใหม่ในข้อ 1 — เป็น bias "ตามธรรมชาติ" ของโมเดล unweighted ปัจจุบัน ไม่ใช่ bias หลัง correction เหมือน LightGBM
   - `prophet`: ไม่ correct (ใช้ 37.5 ตรงๆ ทุก horizon) เพราะ Prophet evaluate แบบ 1-step ไม่ใช่ multi-horizon จริงเหมือนอีก 2 โมเดล จึงไม่มี bias ต่อ horizon ที่เทียบกับการใช้งานจริงได้ตรงๆ (การเอา bias ของโมเดลอื่นมาใช้แทนจะผิดกว่าไม่ correct เลย)
   - `get_forecast()`: เปลี่ยนจุดเทียบ `risk_horizons` จาก `EFFECTIVE_THRESHOLD_BY_HORIZON[h]` เป็น `EFFECTIVE_THRESHOLD_BY_HORIZON_BY_MODEL[model_name][h]` (ใช้ `model_name` ที่ `select_model()` เลือกจริงในรอบนั้น)

**ทดสอบ** ([scripts/test_effective_threshold_by_model.py](scripts/test_effective_threshold_by_model.py), โหลด `app.py` จริงผ่าน `importlib`):
- มี threshold ครบ 3 โมเดล × 24 horizon
- `lightgbm` ตรงกับสูตรเดิมเป๊ะ (ไม่กระทบพฤติกรรมเดิม)
- `xgboost` คำนวณแยกจริง ไม่ใช่ copy ค่า lightgbm (t+24: lightgbm threshold=33.103, xgboost threshold=31.434 — ต่างกันเพราะ xgboost ยังไม่ผ่าน bias correction)
- `prophet` = 37.5 ทุก horizon
- จำลอง `get_forecast()` loop: ค่าพยากรณ์เดียวกันที่ t+24 ถูก flag เป็นความเสี่ยงถ้า route ไป xgboost แต่ไม่ถูก flag ถ้า route ไป lightgbm — ยืนยันว่าผลลัพธ์ขึ้นกับโมเดลที่ route ไปจริงแล้ว ไม่ใช่ threshold เดียวทั้งระบบเหมือนก่อนแก้

ทุก assertion ผ่านหมด

**หมายเหตุ**: นี่แก้แค่ threshold logic ให้ถูกต้องขึ้นตามข้อมูลที่มีอยู่ตอนนี้ — XGBoost ยังไม่ได้ bias correction แบบ sample-weight เหมือน LightGBM จริงๆ (แค่ threshold ที่ใช้เทียบตอนนี้ตรงกับ bias จริงของมันแล้ว) ถ้าจะแก้ที่ต้นตอต้อง retrain XGBoost แบบ weighted ด้วย (ยังไม่ทำในรอบนี้ตามที่ผู้ใช้ระบุ)

**ไฟล์ที่แก้ในรอบนี้**: `app.py` (`BIAS_HIGH_BY_MODEL`/`EFFECTIVE_THRESHOLD_BY_HORIZON_BY_MODEL` ใหม่แทนที่ dict เดิม, แก้จุดเทียบใน `get_forecast()`), `scripts/eval_xgb_bias_high_by_horizon.py` (ใหม่), `models/xgb_bias_high_by_horizon_20260712_191350.csv` (ใหม่), `scripts/test_effective_threshold_by_model.py` (ใหม่)

---

## 2026-07-15 (25) — validate_health_message() ทนทานต่อความหลากหลายทางภาษามากขึ้น (ยังคง rule-based)

**ที่มา**: (24) รายงานว่า 3/5 เคสทดสอบ NLG generation จริง Gemini ตอบถูกต้องตามระดับแต่ถูก safety check ปฏิเสธเพราะ literal substring matching เข้มงวดเกินไป — ขอให้แก้ให้ทนทานขึ้นโดยยังเป็น rule-based เหมือนเดิม (ไม่ใช้ AI ตัวที่สองตรวจสอบ)

**แก้ [app.py](app.py)**:
- **Flexible keyword matching**: keyword ใน `PM25_LEVELS` เป็นได้ทั้ง `str` (ตรงตัวอักษรเป๊ะ เหมือนเดิม) หรือ `tuple` ของ chunk หลายชิ้น (ยอมให้มีคำแทรกคั่นระหว่าง chunk ได้ เช่น `("อากาศ","ดี")` match ทั้ง "อากาศดี" ตรงๆ และ "อากาศจะอยู่ในเกณฑ์ดี" ที่มีคำแทรก) — ใช้ฟังก์ชันใหม่ `_find_keyword_occurrences()` แทนการเรียก `str.find()` ตรงๆ ใน `_mentioned_levels()`
  - กันคำปฏิเสธกลับความหมาย: ถ้าคำแทรกระหว่าง chunk มีคำอย่าง "ไม่"/"มิได้" ถือว่าไม่ match (เช่น "อากาศไม่ดี" ต้องไม่ match "อากาศดี")
  - ช่องว่างคำแทรกจำกัดตายตัว 2 ระดับ: `_MAX_FLEX_GAP=20` (ทั่วไป) กับ `_MAX_FLEX_GAP_STATUS=50` (เฉพาะ `PM25_LEVELS` เพราะ Gemini มักแทรกวลีบอกเวลายาวๆ เช่น "ในอีก 12 ชั่วโมงข้างหน้า" คั่นระหว่างประธานกับคำขยาย)
- **แยกคำเตือนทั่วไปออกจากคำบ่งชี้ระดับเฉพาะ**: เพิ่ม `_GENERIC_CAUTION_KEYWORDS` (ใหม่) เก็บวลีอย่าง "หลีกเลี่ยงกิจกรรมกลางแจ้ง"/"ลดกิจกรรมกลางแจ้ง"/"ระคายเคือง" ที่สมเหตุสมผลตั้งแต่ระดับ "ปานกลาง" ขึ้นไป ไม่ใช่คำเฉพาะของระดับสูงสุดอีกต่อไป (ย้ายออกจาก `PM25_LEVELS["มีผลกระทบต่อสุขภาพ"]`/`["เริ่มมีผลกระทบต่อสุขภาพ"]` เดิม) เช็คแยกผ่าน `_mentions_generic_caution_below()` — ใช้ตรวจแค่ฝั่ง "เตือนเกินจริง" (ถ้าระดับจริงต่ำกว่า floor ของคำเตือนนี้) ไม่เอาไปนับเป็นหลักฐาน "ประเมินต่ำไป" เพราะคำเหล่านี้ไม่ยืนยันเพดานบน ส่วน "อันตราย"/"อันตรายถึงชีวิต" ยังคงผูกกับระดับสูงสุดเท่านั้นเหมือนเดิม (ตามที่ควรเป็นจริง)
- เพิ่ม `"ส่งผลกระทบต่อสุขภาพ"` เป็น keyword คำพ้องของระดับ "เริ่มมีผลกระทบต่อสุขภาพ" (คำพ้องธรรมชาติที่ Gemini ใช้บ่อย)

**🔴 บั๊กที่เจอระหว่างแก้ (รอบแรก)**: `_mentioned_levels()` เดิมเรียง match ทั้งหมดตาม "ความยาวช่วงที่ match จริง" อย่างเดียว — ทำให้ flexible match ที่บังเอิญกินคำแทรกยาวๆ (เช่น `("อากาศ","ดี")` match เข้าไปในคำว่า "ดีมาก" เพราะ "ดี" เป็นตัวอักษรขึ้นต้นของ "มาก") มี span "ยาวกว่า" literal keyword "ดีมาก" ที่แม่นยำกว่า ทำให้เลือกผิดระดับ ("ดี" แทนที่จะเป็น "ดีมาก" ที่ถูกต้อง) แก้โดยให้ literal match ทุกตัวมีสิทธิ์เหนือกว่า flexible match เสมอ (ไม่ว่า span จะสั้นกว่าแค่ไหน) แล้วค่อยเรียงตามความยาว span ภายในกลุ่มเดียวกัน

**ทดสอบ** ([scripts/test_nlg_safety_check.py](scripts/test_nlg_safety_check.py)):
- Regression เดิมทั้งหมด (5 เคสหลัก + 2 substring-collision + 1 fail-safe) ยังผ่านครบ
- เพิ่ม 3 เคสเป้าหมาย (c/d/e) ใช้ข้อความจริงจาก Gemini ที่เคยตกใน (24) (pm25=20/30/50) — ผ่านครบหลังแก้
- เพิ่ม 1 เคสอีก (f) เจาะจงเช็คบั๊กรอบแรกที่เจอ (flexible match แย่ง literal match ผิดระดับ) — ผ่าน
- รวม 12 assertion ผ่านหมด

**ผล [scripts/test_nlg_generation.py](scripts/test_nlg_generation.py) เทียบก่อน/หลัง**: เรียก Gemini จริง (ไม่ mock) รันซ้ำ 4 รอบ (5 เคส/รอบ = 20 ครั้ง) — **ก่อนแก้ 2/5 เคสผ่านแบบใช้ข้อความ Gemini ตรงๆ (ที่เหลือ fallback ไป template), หลังแก้ 20/20 ผ่านแบบใช้ข้อความ Gemini ตรงๆ ทุกครั้ง ไม่มี fallback เลย**

**ไฟล์ที่แก้ในรอบนี้**: `app.py` (`PM25_LEVELS` เปลี่ยน keyword บางตัวเป็น flexible tuple, เพิ่ม `_GENERIC_CAUTION_KEYWORDS`/`_find_keyword_occurrences()`/`_mentions_generic_caution_below()`, แก้ `_mentioned_levels()`, แก้ `validate_health_message()` เพิ่มเช็ค generic caution), `scripts/test_nlg_safety_check.py` (เพิ่ม 4 เคสใหม่ c/d/e/f)

---

## 2026-07-15 (24) — NLG Generation ผ่าน Gemini API (proposal 4.1.4/6.1.4) — ยังไม่ integrate เข้า get_forecast()

**ที่มา**: ต่อจาก (22) ที่ทำ SHAP analysis + NLG Safety Check แยกกันไว้แล้ว รอบนี้ต่อ NLG Generation จริง (แปลงค่าพยากรณ์ + SHAP ให้เป็นข้อความภาษาไทยผ่าน LLM) ตามสเปกที่ผู้ใช้ระบุ

**Setup**:
- เพิ่ม `GEMINI_KEY` ใน `.env` (รูปแบบเดียวกับ `OPENAQ_KEY`), อ่านด้วย manual line-parsing แบบเดิม — ไม่ print/log ค่านี้ที่ไหนเลย
- **SDK ที่ระบุไว้ตอนแรก (`google-generativeai`) ถูก Google เลิกสนับสนุนแล้วจริง** (`FutureWarning: All support for the google.generativeai package has ended`) — เปลี่ยนไปใช้ `google-genai` (SDK ใหม่ที่ยังอัปเดตอยู่) หลังถามยืนยันกับผู้ใช้แล้ว
- **`gemini-2.5-flash` (และ `gemini-2.5-flash-lite`) เรียกไม่ได้จริงกับ API key นี้** — ได้ error 404 `"no longer available to new users"` (โมเดลตระกูล Gemini 2.x ทั้งสายถูกบล็อกสำหรับ API key ที่สร้างใหม่ แม้หน้า pricing ทางการจะยังแสดงว่ามี free tier อยู่ก็ตาม) ทดสอบเรียกจริงแล้วเปลี่ยนไปใช้ **`gemini-3.1-flash-lite`** แทน (GA ไม่ใช่ preview, ยืนยันมี free tier จริง 15 RPM/1000 RPD, เร็ว/ถูกที่สุดในกลุ่ม flash ปัจจุบัน) หลังถามยืนยันกับผู้ใช้แล้ว

**แก้ [app.py](app.py)**:
- Import `get_shap_explanation` จาก `scripts/shap_analysis.py` ผ่าน `sys.path` (ยังคงหลักการเดิม: ไม่แก้ `shap_analysis.py`, reuse ของที่มีอยู่แล้วตรงๆ)
- `generate_nlg_explanation(pm25_value, horizon, model_used, feature_row, user_group=None)`: ดึง top-3 SHAP feature (แปลชื่อเป็นภาษาไทยผ่าน `_FEATURE_NAME_TH`) + `pm25_status()` มาประกอบ prompt ส่ง Gemini, ปรับโทนท้ายประโยคตาม `user_group` (`ทั่วไป`/`เด็ก-ผู้สูงอายุ`/`มีโรคประจำตัว`), ส่งผลลัพธ์ผ่าน `validate_health_message()` เสมอก่อนคืนค่า — ไม่ผ่านหรือเรียก Gemini ไม่สำเร็จเลย คืน template message เสมอ
- **Cache**: in-memory dict คีย์ตาม (ชั่วโมง UTC ปัจจุบัน, horizon, model_used, user_group, pm25 ปัดเศษ) กันเรียก Gemini ซ้ำถ้าเรียกซ้ำในชั่วโมงเดียวกัน (cache ผลลัพธ์สุดท้ายไว้ทั้งกรณีสำเร็จและ fallback เพื่อไม่ยิงซ้ำถี่ๆ ตอน Gemini ล่มอยู่)
- **Timeout**: พบว่า SDK เอง (`http_options.timeout`) บังคับ minimum server-side deadline ที่ 10 วินาที (`"Manually set deadline 5s is too short"` — เจอตอนทดสอบจริง) เอา 5 วินาทีตามที่ต้องการผ่านพารามิเตอร์นี้ไม่ได้ จึงบังคับ timeout ฝั่ง client เองแทนผ่าน `ThreadPoolExecutor` + `future.result(timeout=5)` (thread เดิมรันต่อเบื้องหลังจนจบเอง ไม่ถูก cancel จริง แต่ caller ไม่ต้องรอเกิน 5 วิ ตรงเจตนา "ไม่ให้ hang การพยากรณ์หลัก")
- **Retry**: เฉพาะ error code 429 (quota) เท่านั้น, exponential backoff 1s/2s/4s, เหลือ error อื่นๆ fallback ทันทีไม่ retry

**ทดสอบ** ([scripts/test_nlg_generation.py](scripts/test_nlg_generation.py), เรียก Gemini จริง ไม่ mock): ครบ 5 ระดับ PM2.5 (t+1/t+12/t+24, lightgbm/xgboost/prophet) + จำลอง timeout (monkeypatch ให้ raise `TimeoutError`) → fallback ไป template ถูกต้อง ทั้งหมดผ่าน

**🟡 ข้อสังเกตสำคัญที่ต้องรายงาน (ยังไม่แก้ในรอบนี้)**: จาก 5 เคสทดสอบ มี **3/5 เคส** (pm25=20/"ดี", pm25=30/"ปานกลาง", pm25=50/"เริ่มมีผลกระทบต่อสุขภาพ") ที่ Gemini ตอบ**ถูกต้องตามระดับจริง**แต่ `validate_health_message()` ปฏิเสธเพราะ **ไม่มี keyword ใน `PM25_LEVELS` ตรงตัวอักษรเป๊ะ** (เช่น Gemini เขียน "อากาศจะอยู่ในเกณฑ์ดี" แทนที่จะเป็น "อากาศดี"/"คุณภาพอากาศดี" ตรงๆ) หรือใน 1 เคสเจอ **false positive จากคำที่ overlap ข้ามระดับ** ("หลีกเลี่ยงกิจกรรมกลางแจ้ง" เป็น keyword เฉพาะของระดับสูงสุดในโค้ด แต่เป็นวลีที่ข้อความระดับ "ปานกลาง" ก็ใช้พูดเตือนทั่วไปได้เหมือนกัน) — ระบบ fail-safe ไปใช้ template ถูกต้องเสมอ (ไม่มีข้อความผิดหลุดออกไป) แต่การตรวจสอบแบบ literal keyword matching ปัจจุบันเปราะบางเกินไปสำหรับข้อความ LLM ที่เขียนเป็นธรรมชาติ ทำให้ในทางปฏิบัติข้อความ Gemini ที่ได้เปรียบ (อธิบายเหตุผลจาก SHAP) อาจถูกแทนที่ด้วย template ทั่วไปบ่อยกว่าที่ควร — ยังไม่แก้ไข รอผู้ใช้ตัดสินใจแนวทาง (ขยาย keyword ให้ยืดหยุ่นขึ้น vs. คงไว้ตามสเปก proposal 6.3 ที่วางแผนเปลี่ยนเป็น Generator-Critic Loop ในอนาคตอยู่แล้ว)

**ไฟล์ที่แก้ในรอบนี้**: `app.py` (import `time`/`sys`/`concurrent.futures`/`google.genai`, โหลด `GEMINI_KEY`, section ใหม่ "NLG Generation"), `.env` (เพิ่ม `GEMINI_KEY` — ผู้ใช้ใส่เอง), `scripts/test_nlg_generation.py` (ใหม่), pip: ติดตั้ง `google-genai` (ถอน `google-generativeai` ที่ deprecated ออก)

---

## 2026-07-14 (23) — แก้ pm25_status()/PM25_LEVELS ให้ตรงประกาศ คพ. จริง + อัปเกรด index.html เป็น 5 ระดับ

**ที่มา**: (22) เดา breakpoint 50/90 ไว้ชั่วคราวใน `PM25_LEVELS` — ขอให้เช็คเกณฑ์จริงจากกรมควบคุมมลพิษแล้วแก้ให้ถูกต้อง

**เช็คแหล่งที่มา** (ยืนยันตรงกันจาก 2 แหล่งอิสระ): [ประกาศ คพ. พ.ศ.2566](https://www.pcd.go.th/pcd_news/30028/) (ราชกิจจานุเบกษา 3 ก.ค. 2566, มีผล 1 มิ.ย. 2566) และ [ข่าว Thai PBS](https://www.thaipbs.or.th/news/content/327288) — breakpoint จริงคือ **0-15 ดีมาก, 15.1-25 ดี, 25.1-37.5 ปานกลาง, 37.6-75 เริ่มมีผลกระทบต่อสุขภาพ, 75.1+ มีผลกระทบต่อสุขภาพ** (5 ระดับ ไม่ใช่ 3 หรือมี "อันตราย" แบบที่เดาไว้ — "มีผลกระทบต่อสุขภาพ" คือระดับสูงสุดตามประกาศจริง)

**แก้ [app.py](app.py)**:
- `PM25_LEVELS`: เปลี่ยนเป็น breakpoint จริง (15/25/37.5/75), เพิ่มระดับ "ดีมาก" ที่ขาดไป, ลบ "อันตราย" ที่เดาผิดออก
- `pm25_status()`: **อัปเกรดจาก 3 ระดับเป็น 5 ระดับทางการเดียวกับ `PM25_LEVELS`** (ใช้ `_classify_pm25_level()` ร่วมกัน ไม่ duplicate breakpoint 2 ที่ — จุดนี้ user ยืนยันให้ทำหลังถามเพราะกระทบ `index.html`)

**🔴 บั๊กที่เจอระหว่างแก้ (สำคัญ)**: ชื่อระดับ PM2.5 ที่ติดกันเป็น **substring ของกันเองจริง** — `"มีผลกระทบต่อสุขภาพ"` เป็น substring ของ `"เริ่มมีผลกระทบต่อสุขภาพ"`, `"อากาศดี"` เป็น substring ของ `"อากาศดีมาก"` — ถ้าใช้ `keyword in message_text` ตรงๆ แบบเดิม ข้อความที่ถูกต้องเป๊ะ (เช่น "เริ่มมีผลกระทบต่อสุขภาพ" ตอน pm25 อยู่ในช่วงนั้นจริง) จะโดน flag ผิดว่าขัดแย้งกับตัวเอง เพราะ match เข้าทั้ง 2 ระดับพร้อมกัน — แก้ด้วย `_mentioned_levels()` ใหม่: เรียง keyword ทุกระดับจากยาวสุดไปสั้นสุดก่อนค้นหา แล้ว mark ช่วงตัวอักษรที่จับไปแล้วว่า "consumed" กัน keyword สั้นกว่าซ้อนจับช่วงเดียวกัน (ต้องใส่ชื่อระดับเต็มของแต่ละระดับเป็น keyword ของตัวเองด้วย ไม่ใช่แค่คำย่อ ไม่งั้น sort ตามความยาวจะเลือกผิดตัว — เจอเคสนี้จริงตอน "เริ่มมีผลกระทบ" (คำย่อเดิม) สั้นกว่า "มีผลกระทบต่อสุขภาพ" ของระดับถัดไป)

**แก้ [index.html](index.html)** (จำเป็นเพราะ `statusClass()`/`bodyClassFor()`/`iconFor()` เทียบ string ตรงกับค่าที่ `pm25_status()` คืน — ถ้าไม่แก้คู่กัน ทุกระดับยกเว้น "ดี"/"ปานกลาง" จะ fall through ไปโชว์สไตล์ "แย่ที่สุด" ผิดๆ):
- เพิ่ม CSS var `--excellent` (น้ำเงิน), `--unhealthy` (ส้ม, ย้ายมาจาก `--moderate` เดิม), เปลี่ยน `--moderate` เป็นสีเหลืองจริง (เดิมเป็นสีส้มซึ่งเป็นสีของ "เริ่มมีผลกระทบต่อสุขภาพ" ตามมาตรฐานจริง ไม่ใช่ "ปานกลาง")
- เพิ่ม `body.aqi-excellent`/`body.aqi-unhealthy` (sky gradient 2 เฉดใหม่), `.status-excellent`/`.status-unhealthy` (badge)
- `statusClass()`/`bodyClassFor()`/`iconFor()`: ครบ 5 exact-match ตรงกับ `pm25_status()` ทุกคำ
- `barColor()`: เพิ่ม threshold 15/75 ให้ตรงกับ `PM25_LEVELS` (เดิมมีแค่ 25/37.5)

**ทดสอบ**:
- `pm25_status()` ทุก breakpoint (0,10,15,15.1,20,25,25.1,30,37.5,37.6,50,75,75.1,90,150) จัดระดับถูกต้องครบทุกจุด
- เพิ่ม regression test 2 เคสใน [scripts/test_nlg_safety_check.py](scripts/test_nlg_safety_check.py) เจาะจงเช็คบั๊ก substring: ข้อความ "เริ่มมีผลกระทบต่อสุขภาพ" ตรงระดับจริง (pm25=60) และ "อากาศดีมาก" ตรงระดับจริง (pm25=10) ต้องผ่าน (`is_valid=True`) — ก่อนแก้ `_mentioned_levels()` เคสแรกล้มเหลวจริง (`is_valid=False` ผิด) ยืนยันว่าเป็นบั๊กจริงที่แก้ได้ผล ทดสอบทั้งหมด (5 เคสเดิม + 2 regression + fail-safe) ผ่านครบ
- เปิดเบราว์เซอร์จริงยิง `/api/forecast` ผ่าน live server (auto-reload แล้ว): `current.status` คืน `"ดีมาก"` ถูกต้องสำหรับ pm25=13.9, หน้า index.html แสดง badge สีน้ำเงิน + ☀️ ถูกต้อง ไม่มี JS error (นอกจาก favicon 404 เดิม)

**ไฟล์ที่แก้ในรอบนี้**: `app.py` (`PM25_LEVELS`, `pm25_status()`, `_mentioned_levels()` ใหม่), `index.html` (CSS vars + body classes + JS mapping functions 3 ตัว), `scripts/test_nlg_safety_check.py` (แก้ค่าคาดหวัง Test 3 + เพิ่ม regression test 2 เคส)

---

## 2026-07-14 (22) — SHAP analysis (LightGBM+XGBoost) + NLG Safety Check — 2 งานอิสระ ยังไม่ integrate

**ที่มา**: เริ่มทำ Health Advisory System ตาม proposal 4.1.3/4.1.4/6.1.4 — แบ่งเป็น 2 ส่วนที่ไม่พึ่งกัน (ตรวจสอบ dependency ในโค้ดก่อนแล้วว่าทำคู่ขนานได้จริง) ยังไม่ integrate เข้า `get_forecast()` ตามที่ตกลงไว้ — จะทำเป็นขั้นตอนสุดท้ายแยกต่างหาก

### PART A — SHAP Analysis

[scripts/shap_analysis.py](scripts/shap_analysis.py) (ใหม่ ไม่แตะ `app.py`): `shap.TreeExplainer` บน LightGBM+XGBoost ที่ deploy จริง (ไม่ทำ Prophet เพราะไม่ใช่ tree-based และอธิบายตัวเองได้อยู่แล้ว) ที่ horizon ตัวแทน t+1/t+12/t+24 บน test set จริง (n≈3800-3833 ต่อ horizon)

**ผลลัพธ์**: `pm25_lag_1` ครองอันดับ 1 ทุก (โมเดล, horizon) ชัดเจน — แต่ overlap ของ top-5 ระหว่าง LightGBM/XGBoost **ลดลงตาม horizon** (4/5 ที่ t+1 → 3/5 ที่ t+12 → 3/5 ที่ t+24) โมเดลเห็นตรงกันมากตอน horizon สั้น แต่เริ่มให้น้ำหนัก weather feature ต่างกันตอน horizon ไกล (เช่น t+24: LightGBM ให้น้ำหนัก `month_cos`/`pm25_lag_12`สูง ขณะที่ XGBoost ให้น้ำหนัก `temp`/`pm25_lag_24` แทน) — สอดคล้องกับที่ proposal 4.1.2 พูดถึง temperature inversion/scavenging effect ที่ยิ่งสำคัญตอน horizon ไกล

**Save**: `models/shap_values_by_horizon.pkl` (`{feature_cols, shap_values: {model_name: {horizon: array}}}`), `models/shap_top_features_by_horizon.csv` (30 แถว = 2 โมเดล × 3 horizon × top-5)

**`get_shap_explanation(model_used, horizon, feature_row)`**: routing function ใหม่ — รับ `model_used` ตรงจาก `select_model()` เลือก explainer ที่ตรงโมเดลจริง (cache ต่อ (model, horizon) ไม่สร้างซ้ำ) คืน `None` ถ้า `model_used="prophet"` — ทดสอบแล้วว่า route ถูกต้องทั้ง 2 โมเดล + prophet คืน None ตามคาด ยังไม่ import ไปใช้ที่ไหน (พร้อม integrate ทีหลัง)

### PART B — NLG Safety Check

[app.py](app.py) เพิ่มฟังก์ชันใหม่ (ไม่แก้โค้ดเดิม): `validate_health_message(message_text, predicted_pm25)` — rule-based ล้วนๆ ไม่พึ่ง SHAP หรือโมเดลใดๆ

- `PM25_LEVELS`: ขยาย `pm25_status()` (3 ระดับ) เป็น 5 ระดับ (ดี/ปานกลาง/เริ่มมีผลกระทบ/มีผลกระทบ/อันตราย) — cutoff แรก 2 ตัว (25, 37.5) ตรงกับ `pm25_status()`/`HIGH_POLLUTION_THRESHOLD` เป๊ะ ตามที่สั่งไว้ว่าไม่สร้างเกณฑ์ใหม่ทับของเดิม ส่วน 50/90 เป็น cutoff เพิ่มเพื่อแยก 2 ระดับบนที่ `pm25_status()` เดิมยุบรวมเป็นระดับเดียว (ใช้ breakpoint ที่พบทั่วไปในตาราง PM2.5/AQI ไทย — ไม่ได้อ้างอิงประกาศฉบับใดฉบับหนึ่งเจาะจง)
- ตรวจ: ข้อความมี keyword ระดับ "รุนแรงกว่า" ระดับจริง → fail (เกินจริง); มี keyword เฉพาะระดับ "เบากว่า" ทั้งหมด → fail (ประเมินต่ำไป อันตรายกว่าเพราะทำให้ชะล่าใจ); ไม่มี keyword ระดับไหนเลย → fail-safe (ยืนยันความสอดคล้องไม่ได้)
- ไม่ผ่าน → คืน template message ของระดับจริง (`_fallback_message()`) แทนข้อความเดิม

**ทดสอบ** [scripts/test_nlg_safety_check.py](scripts/test_nlg_safety_check.py) (แบบเดียวกับ `test_dynamic_routing.py`, โหลด `app.py` ผ่าน `importlib`) — ครบ 5 เคสตามที่สั่ง + เคสเสริม 1 อัน ผ่านหมด:
1. ข้อความตรงตามระดับ (ปานกลาง) → `is_valid=True`, คืนข้อความเดิม
2. ข้อความเกินจริง (ปานกลางจริง แต่บอกอันตรายถึงชีวิต) → `is_valid=False`, คืน template "ปานกลาง"
3. ข้อความประเมินต่ำไป (อันตรายจริง แต่บอกอากาศดี) → `is_valid=False`, คืน template "อันตราย"
4. `predicted_pm25=NaN` → `is_valid=False`, `level=None`
5. ข้อความว่างเปล่า → `is_valid=False` (ยังประเมิน level จาก pm25 ได้ปกติ)
6. (เสริม) ข้อความไม่มี keyword ระดับไหนเลย → fail-safe

**ไฟล์ที่แก้ในรอบนี้**: `scripts/shap_analysis.py` (ใหม่), `models/shap_values_by_horizon.pkl` (ใหม่), `models/shap_top_features_by_horizon.csv` (ใหม่), `app.py` (เพิ่ม `PM25_LEVELS`/`validate_health_message()`/helper functions — ไม่แก้โค้ดเดิม), `scripts/test_nlg_safety_check.py` (ใหม่)

---

## 2026-07-14 (21) — หน้า compare.html เลิกผสม backfill กับ live ใน MAE/bias เมื่อ live มีพอแล้ว

**ที่มา**: XGBoost มี backfill 720 แถว (คัดมาจาก test set แบบไม่มี gap เลย) ผสมกับ live ที่ค่อยๆ สะสม — MAE ที่โชว์ในหน้า "ภาพรวม" จึง optimistic กว่า live performance จริง ยิ่งเจอชัดตอนนี้ที่ live สะสมถึง 170 คู่แล้วแต่ backfill 720 ยังครอบงำตัวเลขอยู่

**แก้** [app.py](app.py): เพิ่ม `summarize_errors(all_errors, live_errors)` — ใช้ live-only ถ้า live ≥ `MIN_LIVE_FOR_SUMMARY=5` (เกณฑ์เดียวกับ `compute_rolling_r2()`) ไม่งั้น fallback ไปรวม backfill เหมือนเดิม คืนค่า `(mae, bias, n_used, basis)` โดย `basis` เป็น `'live'`/`'mixed'`
- `api_comparison_overview()`: apply แยกตามแกนของแต่ละ view — `overall_*` (รวมทั้งหมด), `by_horizon[].{mae,n,basis}` (รวมข้าม horizon), `by_model[].{mae,bias,n,basis}` (รวมข้าม model) — แต่ละอันเช็ค live count ในสโคปของตัวเองแยกกัน ไม่ใช่ threshold เดียวรวมทั้งหน้า
- `api_comparison()`: ใช้ switch นี้เฉพาะตอนเลือก `model` เจาะจง (ไม่ใช่ `all`) เพราะ `model=all` รวมหลายโมเดลอยู่แล้ว เป็นคนละความหมายกับ "ผสม backfill/live" — `points` array ไม่แตะ (มี `source` ต่อจุดอยู่แล้ว โปร่งใสในตัว)

**แก้ [compare.html](compare.html)**: เพิ่ม caveat text เล็กๆ (`.basis-note`) ต่อท้ายประโยคสรุปทั้ง 2 จุด (ภาพรวม + รายละเอียดต่อ horizon) และใน model breakdown chip แสดงว่าเลขนี้อิงจาก live ล้วนหรือยังรวม backfill อยู่ พร้อม n ที่ใช้จริง

**ผลทดสอบ** (ยิงจริงผ่าน Flask test client + เปิดเบราว์เซอร์จริงเช็ค):
- `xgboost, h=1`: basis=**live**, MAE=2.358 (n=14) — ต่างจากตัวเลขผสมเดิมชัดเจน
- `xgboost, h=24`: basis=**mixed** (live แค่ 3 คู่ ยังไม่ถึงเกณฑ์) — fallback ถูกต้อง
- `model=all`: basis=None เสมอ (ไม่ apply switch) ตามดีไซน์
- overview `by_model`: **xgboost MAE เปลี่ยนจากตัวเลขผสม (~2.29) เป็น live ล้วน 3.45 (n=170)** — แสดง MAE ที่สะท้อนความจริงมากขึ้นชัดเจน, lightgbm ก็สลับเป็น live แล้วเช่นกัน (n=60)
- เบราว์เซอร์จริง: ไม่มี JS error (นอกจาก favicon 404 เดิม), ข้อความ caveat แสดงถูกต้องทั้ง 2 หน้า

**ไฟล์ที่แก้ในรอบนี้**: `app.py` (`summarize_errors()` ใหม่ + `api_comparison()` + `api_comparison_overview()`), `compare.html` (caveat text 3 จุด + `.basis-note` style)

---

## 2026-07-12 (20) — แก้บั๊ก select_model(): exploration logic เช็คแค่ r2_xgb is None ไม่ครอบคลุม r2_lgb is None

**ที่มา**: หลัง reset `predictions.db` + backfill เฉพาะ XGBoost (720 แถว, (19)) พบว่า live forecast เลือก `lightgbm` 100% ทุกครั้งไม่มีข้อยกเว้น — ตรวจโค้ด `select_model()` พบว่า exploration branch เช็คเจาะจงแค่ `r2_xgb is None` เท่านั้น ไม่ได้เช็ค `r2_lgb is None` ด้วย ทำให้เคส "XGBoost มี R² แล้ว (จาก backfill) แต่ LightGBM ยังไม่มี" (สถานการณ์จริงตอนนี้พอดี) fallback ไป LightGBM ตรงๆ ไม่มีการสุ่มสำรวจเลย — ผิดเจตนา ε-greedy ของ proposal 4.1.6 ที่ต้องการสำรวจโมเดลที่ข้อมูลไม่พอ **ไม่ว่าจะเป็นตัวไหน** ไม่ใช่แค่ XGBoost

**แก้** [app.py](app.py) `select_model()`: แยกเป็น 4 กรณีชัดเจน — ทั้งคู่ None (สุ่ม 50/50), เฉพาะ xgb None (explore xgboost ด้วย `EXPLORATION_RATE`), เฉพาะ lgb None (explore lightgbm ด้วย `EXPLORATION_RATE`), มีครบทั้งคู่ (เทียบ R² ปกติ)

**ทดสอบ** (mock `compute_rolling_r2`/`compute_missing_ratio`, รัน 100 ครั้งต่อเคส):

| เคส | คาดไว้ | ผลจริง |
|---|---|---|
| r2_xgb=None, r2_lgb=None | ~50/50 | xgboost 54%, lightgbm 46% |
| r2_xgb=None, r2_lgb=0.9 | xgb ~20%, lgb ~80% | xgboost 20%, lightgbm 80% |
| **r2_xgb=0.9, r2_lgb=None (เคสที่เคยพัง)** | xgb ~80%, lgb ~20% | xgboost 83%, **lightgbm 17%** (ก่อนแก้คือ 0%) |
| r2_xgb=0.9, r2_lgb=0.7 (มีครบ) | xgboost 100% | xgboost 100/100 |

ทุก assert ผ่าน — ยืนยันเคสที่ 3 (สถานการณ์จริงตอนนี้) ทำงานถูกต้องแล้ว lightgbm จะเริ่มถูกสำรวจ ~20% ของเวลา ทำให้สะสม R² ของตัวเองได้ในที่สุด แทนที่จะถูกบังคับใช้ 100% ตลอดไปเหมือนก่อนแก้

**หมายเหตุที่ยังไม่ได้แก้ (คนละประเด็น จาก (19))**: R² ของ XGBoost ตอนนี้ยังมาจาก backfill 720 แถว (จุดสะอาดที่คัดจาก test set) ซึ่งอาจ optimistic กว่าที่ XGBoost จะทำได้จริงตอน live — จะ self-correct เองเมื่อ live xgboost สะสมพอแทนที่ backfill ในหน้าต่าง rolling 24 คู่ล่าสุด

**ไฟล์ที่แก้ในรอบนี้**: `app.py` (`select_model()`)

---

## 2026-07-12 (19) — retrain ทั้ง 3 โมเดล (LightGBM/XGBoost/Prophet) บนข้อมูลดิบ + deploy จริง (Step 2b-6 ของแผน)

**ที่มา**: (14) หยุดแผน retrain ไว้ที่ Step 2a เพื่อไปตรวจ root cause การที่ MAE ดีขึ้นแค่เล็กน้อย พบว่าเป็นปัญหา data source mismatch (15) แก้ด้วย calibration layer (16-18) — ตอนนี้ input ตอน inference ถูก calibrate ไปสเกล OpenAQ แล้ว การ retrain โมเดลด้วย raw OpenAQ ต่อจึงสมเหตุสมผล เพราะจะได้ทั้งเทรนและ inference สอดคล้องกันเป็นครั้งแรก

**Step 2b** — [scripts/retrain_weighted_v2.py](scripts/retrain_weighted_v2.py) (สำเนา `retrain_weighted.py` ชี้ไปข้อมูลดิบ + grid search ใหม่ทั้งหมด ไม่ reuse weight เดิม): weight ที่เลือกได้ t+1=1.0, t+12=1.0, t+24=10.0 (interpolate ระหว่างกลาง) ผล **MAE_all เฉลี่ย 3.337→3.355 (แทบไม่เพิ่ม), MAE_high เฉลี่ย 4.908→4.506, bias_high เฉลี่ย 3.362→2.700** (ลดลงชัดเจน สมกับจุดประสงค์ของ sample weighting) → `models/lgb_pm25_multistep_weighted_20260712_191056.pkl`

**🔴 บั๊กที่เจอกลางทาง** — `build_raw_dataset.py` (Step 1 จาก (14)) merge คอลัมน์ `pm25` ดิบเข้า `grid` แล้วไม่เคย drop ก่อน rename `pm25_filled`→`pm25` ทำให้ CSV มี 2 คอลัมน์ชื่อ `pm25` ซ้ำกัน (อ่านกลับมาเจอ `pm25`/`pm25.1` เพราะ pandas mangle ชื่อ) — **ไม่กระทบ Step 2a/2b** (ใช้แค่ lag/rolling feature ไม่แตะ `pm25` ตรงๆ, verify ด้วย `np.allclose` เทียบทุกคอลัมน์ feature+target ระหว่างไฟล์เก่า/ใหม่ตรงกันเป๊ะ 42/42 คอลัมน์) แต่จะพัง Prophet training (Step 3) ที่ต้องใช้ `pm25` เป็น target ตรงๆ — แก้ต้นเหตุใน `build_raw_dataset.py` (`drop(columns=["pm25"])` ก่อน rename) แล้ว regenerate ไฟล์ใหม่ `data/khonkaen_pm25_features_raw_20260712_191249.csv` (แถวเท่าเดิม 19,166 แถว)

**Step 3** — [scripts/train_xgb_prophet_v2.py](scripts/train_xgb_prophet_v2.py) (ชี้ข้อมูล/โมเดลชุดใหม่ + XGBoost dropna แยกตาม horizon เหมือน 2a/2b): **XGBoost MAE เฉลี่ย 3.472** (t+1=0.594, t+24=5.977), **Prophet MAE 9.253** (t+1 เดียว ไม่ recursive ตาม simplification เดิมที่ยอมรับไว้แล้ว) → `models/xgb_pm25_multistep_20260712_191350.pkl`, `models/prophet_pm25_20260712_191350.pkl`

**Step 4** — [app.py](app.py): ชี้ `MODEL_PATH`/`XGB_MODEL_PATH`/`PROPHET_MODEL_PATH`/`BIAS_CSV_PATH` ไปไฟล์ใหม่ทั้ง 4 ตัว (ไม่แก้ logic อื่น)

**Step 5** — backup `predictions.db` → `predictions.db.bak_20260712_191432` (204,800 bytes) แล้ว `DROP TABLE prediction_log` + reinit ผ่าน `init_db()` (schema ครบรวม `actual_pm25_calibrated` จาก (17)) — [scripts/backfill_xgboost_v2.py](scripts/backfill_xgboost_v2.py) (สำเนา `backfill_xgboost.py` ชี้ข้อมูล/โมเดลใหม่ + insert `actual_pm25_calibrated=actual_pm25` แบบ identity ตาม convention จาก (18)): backfill 30 จุด × 24 horizon = **720 แถวสำเร็จ ทุก assert (leak check, feature reconstruction, target alignment) ผ่านหมด**

**Step 6 — verify**:
- `/api/forecast`: 200 OK, 24 จุด, ไม่มีค่าติดลบ/ค่าประหลาด, dynamic routing เลือก lightgbm ถูกต้องตาม logic (xgboost มี R² แล้วจาก backfill แต่ lightgbm ยังไม่มี → fallback lightgbm) — log ลง `predictions.db` สำเร็จ 24 แถว
- `/api/comparison`/`overview` ทำงานปกติ, `compute_rolling_r2("xgboost", h)` คืนค่าจริงทุก horizon ที่เช็ค (h=1→0.994, h=12→0.828, h=24→0.539), `compute_rolling_r2("lightgbm", h)` เป็น `None` ตามคาด (ยังไม่มีข้อมูล live/backfill พอ รอสะสม)
- **เทียบ backtest MAE เก่า vs ใหม่ที่ deploy จริง**: t+1 **0.445→0.547**, t+2 **0.696→0.785**, t+24 **4.491→5.748** — **สูงขึ้นทุกจุดตามที่คาดในแผน** (ตัวเลขเก่าคือค่าที่ optimistic เกินจริงจากการ interpolate ข้อมูลเทรน ตัวเลขใหม่สะท้อนความยากจริงของปัญหา) ยังไม่มี live data สะสมพอจะเทียบ live/backtest ratio ใหม่ (รอตามแผน — คาดว่าจะลดจาก 13.3x/9.1x เดิมลงมาใกล้ 1x เพราะตอนนี้ทั้งเทรนและ inference ใช้สเกล+ความผันผวนที่สอดคล้องกันแล้ว)

**ไฟล์ที่แก้ในรอบนี้**: `scripts/retrain_weighted_v2.py` (ใหม่), `scripts/build_raw_dataset.py` (แก้บั๊ก duplicate column), `data/khonkaen_pm25_features_raw_20260712_191249.csv` (ใหม่ — เวอร์ชันสะอาด), `scripts/train_xgb_prophet_v2.py` (ใหม่), `scripts/backfill_xgboost_v2.py` (ใหม่), `app.py` (4 path constants), `predictions.db` (reset), `predictions.db.bak_20260712_191432` (ใหม่ — backup ก่อน reset), `models/lgb_pm25_multistep_weighted_20260712_191056.pkl`, `models/eval_weighted_vs_baseline_20260712_191056.csv`, `models/xgb_pm25_multistep_20260712_191350.pkl`, `models/prophet_pm25_20260712_191350.pkl` (ทั้งหมดใหม่)

---

## 2026-07-12 (18) — แก้ 2 จุดสุดท้ายที่ยังเทียบ actual_pm25 ดิบกับ predicted ที่ calibrate แล้ว

**ที่มา**: (17) เหลือ 2 จุดที่ยังใช้ `actual_pm25` ดิบ: `/api/comparison*` (หน้า compare.html) และ `scripts/backfill_xgboost.py`

**จุดสำคัญที่พบระหว่างแก้ (ต่างจากที่คาดตอนสั่งงาน)**: แถว `source='backfill'` จาก `backfill_xgboost.py` มี `actual_pm25` มาจาก CSV เทรน (`khonkaen_pm25_features_final.csv`) ซึ่งเป็น**สเกล OpenAQ อยู่แล้ว** ไม่ใช่ค่าดิบ Air4Thai — การผ่าน `a*x+b` ซ้ำจะเป็น double-transform ที่ผิด ค่า calibrated ที่ถูกต้องของแถวพวกนี้คือ `actual_pm25` ตรงๆ (identity) และ migration รอบแรกใน (17) ที่ apply `a*x+b` แบบเหมารวมทั้ง 1,063 แถวจึง**ผิดสำหรับ 720 แถว backfill** — รอบนี้แก้ย้อนหลังให้ด้วย

**แก้**:
1. `/api/comparison` + `/api/comparison/overview` ([app.py](app.py)): คำนวณ MAE/bias จาก `actual_pm25_calibrated` — fallback สำหรับแถวที่ยัง NULL คำนวณสดใน SQL แบบ source-aware (`backfill` → ใช้ค่าเดิม, `live` → `a*x+b`) — จุด points ในกราฟส่ง `actual` เป็นค่า calibrated (สเกลเดียวกับเส้น predicted) และเพิ่ม field `actual_raw` เก็บค่าอ่านจริงของสถานีไว้ด้วย
2. `migrate_calibrated_actuals()` ([app.py](app.py)): แยก logic ตาม source (live → `a*x+b`, backfill → identity) + self-healing แก้แถว backfill ที่เคยถูก transform ผิดจากรอบแรกอัตโนมัติ (idempotent)
3. [scripts/backfill_xgboost.py](scripts/backfill_xgboost.py): insert `actual_pm25_calibrated = actual_pm25` (identity) พร้อมกันตั้งแต่แรก

**ผลทดสอบ**:
- แถว xgboost backfill 720/720 มี calibrated ครบ = `actual_pm25` เป๊ะ (0 แถวที่ต่าง) / แถว live calibrated ตรงสูตร `a*x+b` ทุกแถวที่สุ่มเช็ค
- overall MAE เปลี่ยนจาก 3.299 (วิธีเดิม เทียบดิบ) → **3.03** (เทียบสเกลเดียวกัน) — ลดลงตามคาด
- `compute_rolling_r2("xgboost", h)` คืนค่าปกติไม่ใช่ None: h=1 → 0.997, h=12 → 0.957, h=24 → 0.833 (สูงเพราะ dominated ด้วยแถว backfill จาก test set)
- หมายเหตุ: migration ที่แก้แล้วถูกรันไปรอบหนึ่งแล้วตอน user เปิด app หลัง (17) — รอบ verify นี้ยืนยัน idempotency (ไม่มีอะไรเปลี่ยนเพิ่ม)

**ไฟล์ที่แก้ในรอบนี้**: `app.py` (`api_comparison`, `api_comparison_overview`, `migrate_calibrated_actuals()`), `scripts/backfill_xgboost.py` (insert 2 คอลัมน์)

---

## 2026-07-12 (17) — calibrate actual_pm25 ใน backfill ด้วย เพื่อให้ compute_rolling_r2() เทียบสเกลเดียวกัน

**ที่มา**: (16) calibrate เฉพาะ input ฝั่ง `build_feature_row()` แล้ว แต่ `backfill_actuals()` ยังเติม `actual_pm25` จาก Air4Thai ดิบเหมือนเดิม ทำให้ `compute_rolling_r2()` (ใช้ใน Dynamic Routing เทียบ XGBoost vs LightGBM) และ `/api/comparison*` เทียบ `predicted_pm25` (สเกล OpenAQ) กับ `actual_pm25` (สเกล Air4Thai ดิบ) คนละสเกลกันอยู่ดี

**แก้**: เพิ่มคอลัมน์ `actual_pm25_calibrated` แยกใน `prediction_log` (ไม่เขียนทับ `actual_pm25` เดิม) — migration ใน `init_db()` เช็ค `PRAGMA table_info` ก่อน `ALTER TABLE` (idempotent) — `backfill_actuals()` เติมทั้งสองคอลัมน์พร้อมกัน (`actual_pm25` ดิบ + `actual_pm25_calibrated = calibrate_pm25(actual)`) — `compute_rolling_r2()` เปลี่ยนไปอ่าน `actual_pm25_calibrated` แทน

**เพิ่มเติมนอกคำขอเดิม**: รันฟังก์ชัน `migrate_calibrated_actuals()` ครั้งเดียวตอน startup เพื่อ backfill คอลัมน์ใหม่ให้ **แถวเก่าที่มีอยู่แล้วด้วย** (1,063 แถว ทั้งจาก live `backfill_actuals()` เดิมและจาก `scripts/backfill_xgboost.py`) ไม่ใช่แค่แถวใหม่ในอนาคต — เพราะถ้าไม่ทำ `compute_rolling_r2()` จะกลับไปเจอปัญหา cold-start (ไม่มีข้อมูล calibrated ให้ใช้เลยจนกว่าจะสะสมใหม่) ซึ่งขัดกับ Dynamic Routing ที่ต้องพึ่งข้อมูลนี้อยู่แล้ว

**ทดสอบ**: หลัง migration รันซ้ำ (idempotent, ไม่มี error ซ้ำ, ไม่มีอะไรให้ backfill เพิ่มแล้ว) query 5 แถวล่าสุดเทียบ `actual_pm25` vs `actual_pm25_calibrated` — ตรงกับสูตร `a*x+b` เป๊ะทุกแถว (`abs diff < 1e-9`)

**🟡 จุดที่ยังไม่แก้ (นอก scope รอบนี้)**: `/api/comparison` และ `/api/comparison/overview` ยังคำนวณ MAE/bias จาก `actual_pm25` ดิบ (ไม่ใช่ `actual_pm25_calibrated`) เทียบกับ `predicted_pm25` ที่ calibrate แล้ว — ยังคนละสเกลอยู่สำหรับหน้า compare.html/dashboard ถ้าต้องการให้ metric ตรงนี้ถูกต้องด้วยต้องแก้เพิ่ม; `scripts/backfill_xgboost.py` เองก็ยังเขียน `actual_pm25` ดิบตรงๆ ไม่มี calibrated ให้ (แถวใหม่จากสคริปต์นี้ในอนาคตจะต้องรอ migration รอบถัดไปหรือแก้สคริปต์ให้เขียนคอลัมน์ใหม่ด้วย)

**ไฟล์ที่แก้ในรอบนี้**: `app.py` (`init_db()` เพิ่ม migration คอลัมน์, เพิ่ม `migrate_calibrated_actuals()`, แก้ `backfill_actuals()` และ `compute_rolling_r2()`), `predictions.db` (schema + data migration, ไม่ได้ backup แยกเพราะเป็นการเพิ่มคอลัมน์ใหม่ ไม่ได้แก้/ลบข้อมูลเดิม)

---

## 2026-07-12 (16) — แก้ live/backtest mismatch ด้วย calibration layer (Air4Thai → OpenAQ scale)

**ที่มา**: (15) ยืนยันแล้วว่า OpenAQ (เทรนโมเดล) กับ Air4Thai (live inference ใน `fetch_recent_pm25()`) เป็นเซ็นเซอร์ตัวเดียวกันจริง (พิกัดตรงกัน 8 ตำแหน่งทศนิยม) แต่ค่าไม่ตรงกัน (correlation ~0.54) อย่างสม่ำเสมอตลอดเวลา ไม่ใช่ปัญหา time-shift/units — ตัดสินใจแก้ด้วย calibration layer แทนที่จะเปลี่ยน data source หรือรีเทรนด้วย OpenAQ ต่อ (เพราะรีเทรนอย่างเดียวไม่แก้ mismatch ตอน inference)

**STEP A** — fit [scripts/fit_air4thai_openaq_calibration.py](scripts/fit_air4thai_openaq_calibration.py): ดึง Air4Thai ย้อนหลัง 75 วัน (retention limit ~2 เดือน) เทียบกับ OpenAQ cache ช่วงเวลาเดียวกัน (overlap พอดี n=1,395 ชั่วโมง เม.ย.-ก.ค. 2026) fit `LinearRegression`: `openaq_equiv = a * air4thai + b`
- **a = 0.4553, b = 11.2194, R² = 0.397** (สอดคล้องกับ correlation ~0.54 ที่เจอใน (15), 0.54² ≈ 0.29-0.4)
- Save `models/air4thai_openaq_calibration.pkl` (a, b, r2, n, fitted_at, overlap range)

**STEP B** — แก้ [app.py](app.py) `build_feature_row()`: คำนวณ `pm25_lag_*`/`pm25_roll_*` ทั้งหมดจาก `pm25_series` ที่ผ่าน `calibrate_pm25(x) = a*x+b` แล้ว (ทั้ง current value และ historical series ที่ใช้เป็น lag/rolling input) — ส่วน `latest_pm25` ที่ return ไปแสดงเป็น "current" ให้ผู้ใช้ + เทียบ `HIGH_POLLUTION_THRESHOLD` ยังคงเป็นค่าดิบจาก Air4Thai เหมือนเดิม (เป็นค่าอ่านจริงของสถานี ไม่ใช่ input โมเดล)

**ทดสอบ**: เรียก `build_feature_row()` ด้วย mock data 30 ชั่วโมง เทียบ `pm25_lag_1`/`pm25_roll_mean_6` ก่อน/หลัง — ตรงกับสูตร `a*x+b` เป๊ะ (`abs diff < 1e-9`), `latest_pm25` ยืนยันว่ายังเป็นค่าดิบ ไม่ถูก calibrate

**🟡 จุดที่ยังไม่ได้แก้ (นอก scope รอบนี้)**: `backfill_actuals()` ยังเติม `actual_pm25` ในตาราง `prediction_log` จากค่าดิบ Air4Thai เหมือนเดิม ขณะที่ prediction ตอนนี้ถูก calibrate ไปใกล้สเกล OpenAQ แล้ว — แปลว่า live MAE/R² ที่คำนวณใน `/api/comparison*` (และ `compute_rolling_r2` สำหรับ Dynamic Routing) จะยังเทียบกันคนละสเกลอยู่ดี ถ้าต้องการแก้ต้อง calibrate ค่าที่ใช้ backfill ด้วย — ยังไม่ได้ทำเพราะ user ระบุ scope แค่ build_feature_row()

**ไฟล์ที่แก้ในรอบนี้**: `scripts/fit_air4thai_openaq_calibration.py` (ใหม่), `models/air4thai_openaq_calibration.pkl` (ใหม่), `app.py` (แก้ `build_feature_row()` + เพิ่ม `CALIBRATION_PATH`/`CALIBRATION_A`/`CALIBRATION_B`/`calibrate_pm25()`)

---

## 2026-07-12 (15) — ตรวจสอบ 4 จุดก่อนตัดสินใจแก้ OpenAQ vs Air4Thai mismatch

**ที่มา**: (14) เจอว่า baseline LightGBM ที่เทรนใหม่ด้วยข้อมูลดิบดีขึ้นจากของเดิมแค่เล็กน้อย (0.547 vs 0.445 ที่ t+1) ทั้งที่คาดว่าจะดีขึ้นมาก สืบไปเจอว่า OpenAQ (ข้อมูลเทรน) กับ Air4Thai (ข้อมูล live inference จริง) รายงานค่าต่างกันมากสำหรับสถานีเดียวกัน (correlation เพียง 0.54) — ก่อนตัดสินใจว่าจะแก้ยังไง ต้องเช็คว่าความต่างนี้อธิบายได้ด้วยเหตุผลง่ายๆ (เซ็นเซอร์คนละตัว/time-shift/หน่วยต่างกัน) หรือเป็นความจริงที่ต้องแก้

**เช็คที่ 1 — เซ็นเซอร์ตัวเดียวกันจริงไหม (metadata)**: เทียบพิกัด OpenAQ location 7747 กับ Air4Thai station 46t ตรงๆ — **พิกัดตรงกัน 8 ตำแหน่งทศนิยม (16.44538333, 102.83525)** ชื่อสถานีตรงกัน ยืนยันเป็นสถานีเดียวกันจริง ไม่ใช่ความสับสนเรื่อง metadata

**เช็คที่ 2 — ขยายช่วงเวลาเทียบให้ยาวกว่า 1 เดือน**: correlation คงที่ 0.53-0.63 ตลอด 3 เดือนที่เช็ค — ไม่ใช่ความผิดปกติชั่วคราวของเดือนใดเดือนหนึ่ง เป็นความต่างเชิงระบบ

**เช็คที่ 3 — หน่วย/วิธีวัดต่างกันไหม**: หน่วยตรงกัน (µg/m³) ทั้งคู่ — วิธี averaging/processing ที่แน่ชัดเช็คไม่ได้ชัดจาก metadata สาธารณะที่มี

**เช็คที่ 4 — DustBoy (3151) เป็นจุดอ้างอิงที่ 3**: ไม่ได้ทำ — ไม่มี API endpoint ที่ยืนยันได้ (ลอง cusense.cmu.ac.th, dustboy.info, cmuccdc.org ทุกตัวไม่สำเร็จ) และตัดสินใจไม่เดา URL มั่วต่อ

**เพิ่มเติม (ตามคำขอ) — air4thai.com มี endpoint ย้อนหลังไกลกว่า `getHistoryData.php` ไหม**: สำรวจผ่าน Selenium + Chrome DevTools performance log — "รายปี" เป็นแค่รายงาน PDF สรุปรายปีระดับประเทศ (ไม่ใช่ raw data ต่อสถานี, dead end) "รายชั่วโมง" มี query UI จริงแต่ widget เป็น Vue SPA (vue-multiselect + vue-datetime) ที่ automate ด้วย synthetic click ไม่สำเร็จ (station selection ไม่ persist เป็น state จริง) — จาก endpoint name `getHistoryData.php` อยู่ใต้ `/forweb/` (สร้างมาสำหรับหน้าเว็บนี้โดยเฉพาะ) และเป็น endpoint เดียวกับที่ `app.py` ใช้อยู่แล้ว เชื่อว่าหน้าเว็บใช้ endpoint เดียวกับที่ทดสอบตรงไปแล้ว (retention ~2 เดือนเท่ากัน) แต่ไม่ได้พิสูจน์ผ่าน UI จริง 100%

**สรุป**: ทั้ง 4+1 จุด ไม่มีจุดไหนอธิบายความต่างได้ด้วยเหตุผลง่ายๆ — เป็นสถานีเดียวกันจริง ต่างกันสม่ำเสมอตลอดเวลา ไม่ใช่ time-shift → **นำไปสู่การตัดสินใจแก้ด้วย calibration layer ใน (16)**

**ไฟล์ที่แก้ในรอบนี้**: ไม่มีไฟล์ใหม่ถาวร (เป็นการสำรวจ/verify เท่านั้น สคริปต์ Selenium อยู่ใน scratchpad ชั่วคราว)

---

## 2026-07-12 (14) — สร้างข้อมูลดิบ + เทรน baseline LightGBM ใหม่ (พบปัญหาที่นำไปสู่ (15))

**ที่มา**: หลังจาก (13) ยืนยันว่า retrain window 2024-01-01 ถึง 2026-07-02 มีข้อมูลพอ เริ่ม implement ตามแผนที่ approve แล้ว (`C:\Users\junta\.claude\plans\zany-sprouting-wren.md`) Step 1-2a

**Step 1** — [scripts/build_raw_dataset.py](scripts/build_raw_dataset.py): สร้าง `data/khonkaen_pm25_features_raw_20260712_142359.csv` จาก OpenAQ raw pm25 + Open-Meteo archive weather (2024-01-01 ถึง 2026-07-02) — gap ≤48h interpolate, gap >48h ตัดทิ้งทั้งแถวที่ feature/target แตะ (`touches_long_gap` rolling mask) คำนวณ feature ด้วยสูตรเดียวกับ `app.py:build_feature_row()`/`backfill_xgboost.py:build_feature_row_at()` เป๊ะ, target คำนวณแบบ timestamp-based lookup (ไม่ใช่ row-shift) ผลลัพธ์: **19,166 แถวสุดท้าย** (จาก 21,913 ชั่วโมงทั้งหมด, impute 5,491 ชั่วโมง, ตัดทิ้ง 2,627 ชั่วโมง) sanity check 0 formula mismatch, 0 stray NaN

**Step 2a** — [scripts/train_baseline_lgb_v2.py](scripts/train_baseline_lgb_v2.py): เทรน unweighted baseline LightGBM (24 horizon, `n_estimators=200, random_state=42`) บนข้อมูลใหม่ — train 15,332 แถว (2024-01-02 ถึง 2026-01-23), test 3,834 แถว (2026-01-23 ถึง 2026-07-02) ผล: **t+1 MAE_all=0.547** (เทียบ baseline เดิม 0.445), t+24 MAE_all=5.693, R²=0.713, MAE_high เฉลี่ย 4.908, bias_high เฉลี่ย +3.362

**🔴 ปัญหาที่เจอ**: คาดว่า MAE จะดีขึ้นชัดเจนเพราะใช้ข้อมูลดิบไม่ interpolate แต่กลับดีขึ้นแค่เล็กน้อย (0.547 vs 0.445) — สืบสาเหตุพบว่า OpenAQ (ข้อมูลเทรน) กับ Air4Thai (ข้อมูล live) รายงานค่าต่างกันมากสำหรับสถานีเดียวกัน (correlation 0.54) ทำให้ retrain ด้วย OpenAQ อย่างเดียวไม่ช่วยแก้ mismatch ตอน inference จริง — **หยุด implement แผนต่อ (Step 2b-6) ไปตรวจสอบ root cause นี้ก่อน ดู (15)/(16)**

**ไฟล์ที่แก้ในรอบนี้**: `scripts/build_raw_dataset.py` (ใหม่), `data/khonkaen_pm25_features_raw_20260712_142359.csv` (ใหม่), `scripts/train_baseline_lgb_v2.py` (ใหม่), `models/lgb_pm25_multistep_baseline_*.pkl` (ใหม่), `models/eval_high_pollution_by_horizon_v2.csv` (ใหม่)

---

## 2026-07-12 (13) — ประเมิน retrain window 2024-01-01 ถึง 2026-07-02 ก่อนตัดสินใจ retrain จริง

**ที่มา**: (12) แนะนำให้ใช้ช่วง 2024 เป็นต้นมาแทนทั้ง 5 ปี เพราะ missing rate ต่ำกว่า — ก่อนตัดสินใจจริงต้องรู้ว่าหลัง impute gap สั้น + ตัด gap ยาวทิ้ง จะเหลือกี่แถว และฤดูหมอกควัน (ธ.ค.-มี.ค., สำคัญที่สุดสำหรับระบบ) มีข้อมูลพอในทุกปีไหม

**สร้าง**: [scripts/evaluate_2024_2026_retrain_window.py](scripts/evaluate_2024_2026_retrain_window.py) (ใช้ cache เดิมจาก (12) ไม่ดึง API ซ้ำ) — impute gap <=48h, ตัด gap >48h ทิ้ง, คำนวณ coverage รายเดือน, เทียบฤดูหมอกควันข้ามปี 2021-2026

**ผลลัพธ์**:
- **จำนวนแถวสุดท้าย: 19,286 แถว (50.9% ของ dataset เดิม 37,909 แถว)** — ตัด gap ยาว >48h ทิ้งไป 2,627 ชั่วโมง (4 ครั้ง หลักๆ จาก ก.ค. 2025 ที่ coverage 0% ทั้งเดือน และ ส.ค. 2025 ที่ 10.1%)
- **Coverage ฤดูหมอกควัน (ธ.ค.-มี.ค.) เฉลี่ย 72.9%** ในปี 2024-2026 ต่ำสุดคือ ธ.ค. 2024 ที่ 52.6% — ไม่มีเดือนไหนต่ำกว่า 40% ที่ตั้งเกณฑ์ไว้
- ตัวเลข "2025Q4 missing 62.8%" ที่เคยเจอใน (12) เป็นการเข้าใจผิดจาก quarterly aggregation — จริงๆ ธ.ค. 2025 (เดือนฤดูหมอกควัน) coverage ปกติที่ 64.1% ตัวที่ลากลงคือ ต.ค./พ.ย. 2025 (20.4%/26.7%) ซึ่งไม่ใช่ฤดูหมอกควัน
- เทียบข้ามปี: **2021-2023 แย่กว่า 2024-2026 มากในทุกเดือนฤดูหมอกควัน** (บางเดือนแค่ 2.5-25%) จึง**ไม่แนะนำให้ดึงมาเสริม** เพราะแย่กว่าที่มีอยู่แล้ว

**คำแนะนำ**: ใช้ช่วง 2024-01-01 ถึง 2026-07-02 เป็น retrain window ได้เลย โดยไม่ต้องดึงปีอื่นมาเสริมฤดูหมอกควัน — แม้ row count จะเหลือครึ่งหนึ่งของเดิม แต่ฤดูที่สำคัญที่สุดมี coverage เพียงพอและดีที่สุดในข้อมูลทั้ง 5 ปีแล้ว

**ไฟล์ที่แก้ในรอบนี้**: `scripts/evaluate_2024_2026_retrain_window.py` (ใหม่) — ยังไม่ได้ retrain จริง เป็นการประเมิน window ก่อนตัดสินใจ

---

## 2026-07-12 (12) — เช็ค missing rate 5 ปีของ OpenAQ + เทียบกับ Air4Thai ก่อนวางแผน retrain

**ที่มา**: ก่อนวางแผน retrain ด้วยข้อมูลดิบ ต้องรู้ก่อนว่า missing rate 42.49% ที่เจอใน (11) เป็นปัญหาเฉพาะ test set หรือแย่ตลอด 5 ปี และ Air4Thai (ที่ live app ใช้จริง) มี raw data ครบกว่า OpenAQ ไหม

**เช็ค 1 — missing rate OpenAQ ตลอด 5 ปี** ([scripts/check_openaq_missing_rate_5y.py](scripts/check_openaq_missing_rate_5y.py)):
- Missing rate รวม 5 ปี (2021-04 ถึง 2026-07): **54.64%** (แย่กว่าช่วง test set เดี่ยวๆ ที่ 42.49% อีก)
- **ไม่คงที่ ผันผวนหนักตามไตรมาส**: แย่สุด 2022Q4 (98.60% missing) ดีสุด 2026Q2 (14.61% missing) — มีแนวโน้มดีขึ้นในช่วงหลังๆ (2024-2026) แต่ไม่ใช่เส้นตรง มี blip แย่ลงที่ 2025Q3-Q4 (75.5%, 62.8%) ก่อนดีขึ้นอีกใน 2026

**เช็ค 2 — เทียบ Air4Thai vs OpenAQ ช่วงเวลาเดียวกัน** ([scripts/check_air4thai_missing_rate.py](scripts/check_air4thai_missing_rate.py)):
- **Air4Thai มี hard retention limit ~2 เดือนเท่านั้น** — ขอข้อมูลย้อนหลังเกินนั้น (ทดสอบ 3 ช่วงในปี 2025) ได้ **0 แถวทุกครั้ง** ไม่ใช่แค่ข้อมูลน้อย แต่ API ไม่คืนอะไรมาเลย
- ในช่วงที่ Air4Thai มีข้อมูลจริง (2026-04-30 ถึง 2026-07-01): Air4Thai missing แค่ **0.40%** เทียบ OpenAQ ช่วงเดียวกัน **6.45%** — Air4Thai ครบกว่า ~16 เท่า

**สรุปคำแนะนำสำหรับแผน retrain**:
- Air4Thai ให้คุณภาพข้อมูลดีกว่ามาก แต่ **retention สั้นเกินไป (~2 เดือน) ใช้แทน OpenAQ เป็นแหล่งหลักไม่ได้** เพราะต้องการข้อมูลย้อนหลังอย่างน้อยเท่า test set เดิม (14 เดือน) หรือมากกว่า
- OpenAQ ยังต้องเป็นแหล่งหลักเพราะมี retention ยาวพอ แต่ควร**เลือกช่วงเวลาที่ missing rate ต่ำกว่า** (เช่น 2024 เป็นต้นมา ซึ่ง missing rate ~15-40%) แทนการรวมช่วง 2021-2023 ที่ missing สูงถึง 60-98% ซึ่งต้อง interpolate มากจนข้อมูลไม่ต่างจากที่มีปัญหาอยู่แล้ว
- อาจใช้ Air4Thai (2 เดือนล่าสุด, missing 0.4%) เป็น validation set คุณภาพสูงเสริม แทนที่จะเป็นแหล่งเทรนหลัก

**ไฟล์ที่แก้ในรอบนี้**: `scripts/check_openaq_missing_rate_5y.py` (ใหม่), `scripts/check_air4thai_missing_rate.py` (ใหม่), `scripts/_openaq_raw_5y_cache.csv` (ใหม่ — cache ข้อมูลดิบ 5 ปี ไม่ต้องดึงซ้ำ)

---

## 2026-07-12 (11) — พิสูจน์ว่าข้อมูลเทรนถูก interpolate จนลบความผันผวนจริงทิ้ง (root cause ตัวจริง)

**ที่มา**: (10) หา shock event ใน test set ได้แค่ 2 จุด ทั้งที่ live เจอบ่อย — สงสัยว่า test set นิ่งผิดปกติจริง หรือแค่ threshold ไม่เหมาะ ทดสอบสมมติฐาน "shock อธิบาย live MAE สูง" ตรงๆ กับ live data (132 shock vs 211 normal) พบว่า MAE ใกล้เคียงกันมาก (5.16 vs 4.98, ratio 1.0x) — ปัดสมมติฐาน shock-only ตกไป จึงต้องสืบต่อว่าทำไม test set ถึงนิ่งผิดปกติ

**TASK 1** — เทียบข้อมูลดิบจาก OpenAQ (sensor 1304249, location 7747 = สถานีเดียวกับที่ใช้เทรนโมเดล ยืนยันชื่อสถานีตรงกัน) กับ `data/khonkaen_pm25_features_final.csv` ช่วงเวลาเดียวกับ test set (พ.ค. 2025 – ก.ค. 2026): สร้าง [scripts/compare_raw_vs_interpolated.py](scripts/compare_raw_vs_interpolated.py)

ผล (**ยืนยันสมมติฐาน B**: interpolation ลบความผันผวนออกไปจริง):
| | RAW (OpenAQ) | CLEAN (CSV) |
|---|---|---|
| max \|change\|/ชม. | **16.1** | 6.4 |
| count >3 | 44 | 2 |
| count >10 | 4 | 0 |

**42.49% ของชั่วโมงในช่วง test set ไม่มีข้อมูลจริงจาก OpenAQ เลย** (ต้องถูก interpolate ถึงจะเติมให้ครบ) — เกือบครึ่งหนึ่งของ "ข้อมูลจริง" ที่ใช้เทรน/ประเมินโมเดลทั้งหมด เป็นค่าที่แต่งขึ้นจาก linear interpolation ไม่ใช่ค่าที่วัดจริง

**TASK 2** — เทียบ live MAE กับ backtest MAE (`test_mae_by_horizon` จาก pkl โมเดล weighted ที่ deploy จริง) แบบ horizon ต่อ horizon: สร้าง [scripts/analyze_live_vs_backtest_by_horizon.py](scripts/analyze_live_vs_backtest_by_horizon.py)

ผล: ratio สูงสุดที่ horizon สั้น แล้วลดลงเรื่อยๆ ตาม horizon —
- t+1: live 5.91 vs backtest 0.445 → **13.3x**
- t+2: live 6.36 vs backtest 0.696 → **9.1x**
- t+5: 4.1x, t+10: 2.2x, t+20: 1.0x, t+24: 1.3x

ตรงกับ Task 1 พอดี: horizon สั้นพึ่ง lag feature หนักที่สุด ซึ่งถูก smooth มากที่สุดในข้อมูลเทรน จึง "ดูแม่นเกินจริง" ตอน backtest มากที่สุดที่ horizon สั้น

**สรุป**: ควร **retrain ด้วยข้อมูลดิบจาก OpenAQ/Air4Thai โดยตรง** (ไม่ผ่าน interpolation ก่อน) — มีหลักฐานเชิงปริมาณสนับสนุนชัดเจนทั้ง 2 มุม (raw data ผันผวนใกล้เคียง live จริง + ratio live/backtest เรียงตาม horizon สอดคล้องกับกลไกที่อธิบายได้) ไม่ใช่แค่ anecdote จาก log ไม่กี่จุดอีกต่อไป

**ไฟล์ที่แก้ในรอบนี้**: `scripts/compare_raw_vs_interpolated.py` (ใหม่), `scripts/analyze_live_vs_backtest_by_horizon.py` (ใหม่) — ยังไม่ได้ retrain โมเดลจริง เป็นแค่การวิเคราะห์ยืนยัน root cause

---

## 2026-07-12 (10) — วิเคราะห์ shock event จาก historical backtest (พบว่า test set นิ่งผิดปกติ)

**ที่มา**: สงสัยว่า LightGBM (weighted, deployed) พลาดหนักตอน PM2.5 เปลี่ยนแปลงฉับพลันหรือไม่ ต้องการวิเคราะห์จาก backtest แทนที่จะรอสะสม live data

**สร้าง**: [scripts/analyze_shock_events.py](scripts/analyze_shock_events.py) — หา shock event (`|pm25-pm25_lag_1| > threshold`) ใน test set, เทียบ MAE shock vs baseline (`test_mae_by_horizon` จาก pkl), แยกทิศทางร่วง/พุ่ง, เช็ค precipitation (Scavenging Effect)

**พบว่า test set มี shock event น้อยผิดปกติ**: threshold >10 และ >7 ไม่พบเลยแม้แต่จุดเดียว, >5 พบ 1 จุด, >3 พบแค่ 2 จุด (จาก 7,582 แถว) ขณะที่ train set ช่วงเวลาเดียวกันมี 106 จุดที่ threshold เดียวกัน — เป็นสัญญาณแรกว่า test set ผิดปกติ (ภายหลังพิสูจน์ใน (11) ว่าเป็นเพราะ interpolation ไม่ใช่อากาศนิ่งจริง)

**ผลที่ได้ (มีข้อจำกัดเรื่อง n)**:
- Test set (held-out, n=2 เท่านั้น): MAE ตอน shock สูงกว่าปกติ 3.9x-11.4x (t+1..t+4)
- Train set (in-sample, n=106, ไม่ใช่ held-out): สูงกว่าปกติ 2.4x-6.1x, ทิศ "พุ่ง" (5.7x) แย่กว่า "ร่วง" (2.7x) เฉลี่ย — ตรงข้ามกับที่สังเกตจาก live log
- Precipitation: จุดร่วงฮวบมีโอกาสฝนตกสูงกว่าเฉลี่ยทั่วไป 1.5 เท่า (สนับสนุน Scavenging Effect แบบอ่อนๆ)

**ไฟล์ที่แก้ในรอบนี้**: `scripts/analyze_shock_events.py` (ใหม่)

---

## 2026-07-12 (9) — เปลี่ยน label ค่า PM2.5 ปัจจุบันใน index.html ให้ชัดว่าเป็นค่าดิบรายชั่วโมง

**ที่มา**: ผู้ใช้สังเกตว่าค่า PM2.5 ใน demo (ค่าดิบรายชั่วโมงล่าสุด) ไม่ตรงกับที่เว็บ Air4Thai แสดง ตรวจสอบแล้วพบว่า Air4Thai แสดงค่าเฉลี่ยเคลื่อนที่ 24 ชม. (ตามมาตรฐาน AQI ของกรมควบคุมมลพิษ) ไม่ใช่ค่าดิบรายชั่วโมง — ยืนยันด้วยการดึง API สถานีเดียวกัน (`46t`) มาคำนวณเฉลี่ย 24 ชม. เอง ได้ 19.4 ตรงกับตัวเลขในภาพที่ผู้ใช้แนบมาเป๊ะ ขณะที่ค่าดิบล่าสุดคือ 12.9 คนละค่ากัน (ไม่ใช่บั๊กเรื่องดึงข้อมูลผิดสถานี)

**การตัดสินใจ**: คงค่าดิบรายชั่วโมงไว้ตามเดิม (ไม่เปลี่ยนไปคำนวณค่าเฉลี่ย 24 ชม.) แต่เปลี่ยน label ให้ชัดเจนว่าไม่ใช่ AQI เฉลี่ยทางการ

**สิ่งที่ทำ** ใน [index.html](index.html): เพิ่มบรรทัด label เล็กๆ เหนือตัวเลข PM2.5 ปัจจุบัน — "PM2.5 ล่าสุดรายชั่วโมง (ไม่ใช่ค่าเฉลี่ย 24 ชม. แบบ AQI ทางการ)"

**ทดสอบแล้ว**: รัน `app.py` จริง เปิดด้วย Selenium + headless Chrome สกรีนช็อตยืนยันว่า label ใหม่ render ถูกต้อง ไม่ชนกับ layout อื่น

**ไฟล์ที่แก้ในรอบนี้**: `index.html`

---

## 2026-07-12 (8) — ปรับหน้าเว็บ index.html และ compare.html ให้สอดคล้องกับ Dynamic Routing

**ที่มา**: index.html/compare.html ยังเป็นดีไซน์เดิมตั้งแต่ก่อนมี Dynamic Routing — ไม่เคยแสดง model_used/source เลย ทั้งที่ backend เปลี่ยนไปมากตั้งแต่ (4)-(7)

**Backend** ([app.py](app.py)):
- `/api/comparison`: เพิ่ม query param `model` (all/lightgbm/xgboost/prophet) กรองผลลัพธ์, เพิ่ม `model_used`/`source` ในแต่ละ point, เพิ่ม `n_backfill` ใน summary
- `/api/comparison/overview`: เพิ่ม `by_model` — breakdown MAE/bias/n/n_backfill/n_live แยกตามโมเดล

**[index.html](index.html)**:
- Subtitle บอกว่าใช้ Dynamic Routing (LightGBM/XGBoost/Prophet) แทนที่จะบอกว่าใช้ LightGBM อย่างเดียว
- เพิ่มจุดสีใต้แต่ละแท่งกราฟบอกว่า horizon นั้นใช้โมเดลไหน (lightgbm=น้ำเงิน, xgboost=เขียวอมฟ้า, prophet=เหลือง) พร้อม legend
- เพิ่ม info banner แสดงเมื่อ `prophet_triggered_horizons` ไม่ว่าง (บอกว่า Circuit Breaker สลับไป Prophet horizon ไหนบ้าง)

**[compare.html](compare.html)**:
- เพิ่ม dropdown กรองโมเดล (ทุกโมเดล/LightGBM/XGBoost/Prophet) คู่กับตัวเลือก horizon เดิม
- เพิ่มสถิติ "เป็น backfill" ในแถว stats ของรายละเอียดแต่ละ horizon
- กราฟ: จุดค่าพยากรณ์เปลี่ยนเป็นสีตามโมเดลที่ใช้จริง, วงกลมกลวง = backfill, วงกลมทึบ = live (ค่าจริงยังคงเป็นจุดเขียวทึบเหมือนเดิม เพราะเป็น ground truth ไม่ขึ้นกับโมเดล)
- เพิ่ม model breakdown chips ในการ์ดภาพรวม แสดง MAE/bias/n/backfill-live แยกตามโมเดล

**ทดสอบแล้ว**: รัน `app.py` จริง ใช้ Selenium + headless Chrome เปิดทั้ง 2 หน้า, เช็ค console ไม่มี error (นอกจาก favicon 404 ที่ไม่เกี่ยวข้อง), สกรีนช็อตยืนยันว่า UI ใหม่ render ถูกต้องไม่มี layout ชนกัน — เห็น routing จริงทำงาน (index.html แสดงจุดเขียว/XGBoost เกือบทุก horizon หลัง backfill, compare.html แสดง LightGBM MAE ±4.97 (n=338, live ทั้งหมด) vs XGBoost MAE ±2.23 (n=720, backfill ทั้งหมด) ถูกต้องตรงกับข้อมูลจริงใน DB)

**ไฟล์ที่แก้ในรอบนี้**: `app.py` (`/api/comparison`, `/api/comparison/overview`), `index.html`, `compare.html`

---

## 2026-07-12 (7) — Backfill ข้อมูล xgboost ย้อนหลัง แก้ cold-start deadlock ให้เสร็จจริง

**ที่มา**: (6) พบว่า Dynamic Routing ทำงานถูกแล้วแต่ predictions.db ยังไม่มีแถว xgboost เลยสักแถว การรอสะสมจาก live (แม้จะมี exploration แล้ว) ใช้เวลาหลายวันโดยเฉพาะถ้าไม่ได้เปิด app.py ค้างไว้ 10+ ชม./วัน จึง backfill ย้อนหลังจาก test set แทน

**เพิ่มคอลัมน์ `source` ใน predictions.db**: สำรอง DB ก่อน (`predictions.db.bak_20260712_023451`), `ALTER TABLE ... ADD COLUMN source TEXT DEFAULT 'live'` (432 แถวเดิมทั้งหมดกลายเป็น `'live'` อัตโนมัติ), แก้ `init_db()` และ `log_predictions()` ใน [app.py](app.py) ให้ตรงกับ schema ใหม่ (ป้องกันบั๊กแบบเดียวกับ `model_used` ที่เจอใน (6) ไม่ให้เกิดซ้ำเป็นครั้งที่ 3)

**สร้าง**: [scripts/backfill_xgboost.py](scripts/backfill_xgboost.py)
- สุ่ม 30 จุดเวลาจาก test set ของ `data/khonkaen_pm25_features_final.csv` (split เดียวกับตอนเทรน XGBoost, แถวที่โมเดลไม่เคยเห็นตอนเทรน) กระจายทั่วช่วง 2025-05-24 ถึง 2026-06-23 — กรองเฉพาะจุดที่หน้าต่าง 48 ชม. รอบข้างไม่มี gap ของข้อมูล
- สร้าง feature row จากข้อมูล `datetime_utc <= backtest_time` เท่านั้น (logic เดียวกับ `build_feature_row()`) — มี `assert` 3 ชั้นกันข้อมูลอนาคตหลุดเข้ามา: (1) history slice ไม่เกิน backtest_time, (2) lag/rolling window ไม่เกิน backtest_time, (3) feature ที่ reconstruct ได้ตรงกับคอลัมน์ precompute ในไฟล์ CSV เป๊ะ (`np.allclose`)
- Cross-check เพิ่มว่าคอลัมน์ `pm25_t+{h}` ตรงกับค่า `pm25` จริงที่แถวห่างไป h ชั่วโมง (กัน target เพี้ยนจาก gap)
- Predict ด้วย XGBoost ทุก horizon แล้ว insert พร้อม `actual_pm25` จริงทันที, mark `model_used='xgboost'`, `source='backfill'`

**ผลลัพธ์**: 720 แถว (30 จุด × 24 horizon) ผ่าน assert ทั้งหมด, insert สำเร็จ (432 → 1152 แถว) ทุก horizon มีครบ 30 คู่ backfill (เกิน threshold 5 คู่ที่ `compute_rolling_r2()` ต้องการ)

**Sanity check**: `compute_rolling_r2("xgboost", h)` คืนค่าได้จริงทุก horizon แล้ว (เช่น t+1=0.998, t+24=0.833 — pattern ลดลงตาม horizon ตรงกับตอนเทรน) MAE เฉลี่ย 24 horizons จากชุด backfill นี้ = **2.230** เทียบกับ MAE backtest เต็ม (7582 แถว) ตอนเทรน = **2.728** — ใกล้เคียงกันสมเหตุสมผล (ต่างกันเพราะ sample เล็กแค่ 30 จุด/horizon)

**ไฟล์ที่แก้ในรอบนี้**: `app.py` (`init_db()`, `log_predictions()` เพิ่ม `source`), `predictions.db` (ALTER TABLE + insert 720 แถว), `predictions.db.bak_20260712_023451` (ใหม่ — backup), `scripts/backfill_xgboost.py` (ใหม่)

---

## 2026-07-12 (6) — ทดสอบระบบ Circuit Breaker + Dynamic Routing, พบและแก้บั๊ก init_db() ขาดคอลัมน์ model_used

**ที่มา**: ทดสอบระบบ Dynamic Routing ที่เพิ่ง implement เสร็จตามที่วางแผนไว้ 4 ข้อ (circuit breaker, rolling R² routing, end-to-end, เช็คข้อมูลจริงใน DB)

**บั๊กที่พบระหว่างทดสอบ**: `init_db()` ใน [app.py](app.py) ยัง `CREATE TABLE` แบบเดิมที่ไม่มีคอลัมน์ `model_used` เลย (คอลัมน์นี้ถูกเพิ่มเข้า `predictions.db` จริงด้วย `ALTER TABLE` ครั้งเดียวเมื่อก่อนหน้านี้เท่านั้น ไม่เคยแก้ต้นทางใน `init_db()`) — ถ้า `predictions.db` ถูกลบแล้วสร้างใหม่ (เคยเกิดขึ้นแล้วครั้งหนึ่งในโปรเจกต์นี้) แอปจะ crash ทันทีที่เรียก `log_predictions()` ครั้งแรกด้วย `no such column: model_used` **แก้แล้ว**: เพิ่ม `model_used TEXT DEFAULT 'lightgbm'` เข้าไปใน `CREATE TABLE` ให้ตรงกับ schema จริง

**สร้าง**: [scripts/test_dynamic_routing.py](scripts/test_dynamic_routing.py) — ใช้ temp sqlite db แยกต่างหาก (ไม่แตะ `predictions.db` จริง) สำหรับ test 1-2

**ผลทดสอบ**:
1. **Circuit Breaker** (missing_ratio > 0.3 → prophet): **PASS** — missing_ratio=0.333, `select_model()` คืน `"prophet"` ถูกต้อง
2. **Rolling R² routing ถูกทั้ง 2 ทาง**: **PASS** — กรณี xgboost ดีกว่า (R²=0.999 vs -1.000) เลือก `"xgboost"`; กรณีสลับ (R²=-1.025 vs 0.999) เลือก `"lightgbm"` ถูกต้องทั้งคู่
3. **End-to-end** (`python app.py` + `GET /api/forecast` จริง): **PASS** — response 24 จุด มี `model_used` ครบทุกจุด ไม่มี error, ไม่มีค่าติดลบ, hour-to-hour jump สูงสุด 8.7 µg/m³ (สมเหตุสมผล), เห็น routing จริงทำงาน (17 lightgbm / 7 xgboost ในคำเรียกนี้)
4. **ข้อมูลจริงใน predictions.db**: 24/24 (model, horizon) คู่ที่มี ≥5 matched pairs — **แต่ทั้งหมดเป็น lightgbm ล้วน ยังไม่มีแถว xgboost เลยแม้แต่แถวเดียว** (0/432)

**ข้อสังเกตเพิ่มเติมจากการทดสอบข้อ 3+4 ร่วมกัน**: การเรียก `/api/forecast` ในข้อ 3 จริงๆ แล้ว route ไป xgboost 7 horizon (เห็นใน response) แต่เช็คใน DB แล้วพบว่าทุก horizon นั้นถูกบันทึกเป็น `lightgbm` เพราะ scheduler เรียก `get_forecast()` ไปก่อนแล้วในชั่วโมงเดียวกัน (ชนกับ `UNIQUE(predicted_at_utc, horizon)` → `INSERT OR IGNORE` ทิ้งผลของการเรียกที่สอง) สรุปคือ **ต่อ 1 ชั่วโมงข้อมูลจริง มีโอกาส "นับ" ได้แค่ครั้งเดียว** ไม่ว่าจะเรียก `/api/forecast` กี่ครั้งก็ตาม — ไม่ใช่บั๊ก (เป็น dedup ที่ตั้งใจออกแบบไว้) แต่มีผลต่อการประมาณเวลาสะสมข้อมูล

**ไฟล์ที่แก้ในรอบนี้**: `app.py` (แก้ `init_db()`), `scripts/test_dynamic_routing.py` (ใหม่)

---

## 2026-07-12 (5) — แก้ cold-start deadlock ใน select_model() ด้วย exploration

**ที่มา**: ทดสอบ (4) แล้วพบว่าทุก horizon route ไป `"lightgbm"` เสมอ เพราะ `select_model()` เลือก xgboost ได้ก็ต่อเมื่อมี rolling R² ของ xgboost ที่ horizon นั้น (≥5 คู่ backfill แล้ว) แต่ข้อมูลนั้นจะมีได้ก็ต่อเมื่อเคย route ไป xgboost มาก่อน — chicken-and-egg ที่ทำให้ xgboost ไม่มีวันถูกเลือกเลย

**สิ่งที่ทำ** ใน [app.py](app.py):
- เพิ่ม `import random` และ config `EXPLORATION_RATE = 0.2`
- แก้ `select_model()`: เมื่อ `compute_rolling_r2("xgboost", horizon)` เป็น `None` (ข้อมูลไม่พอ) ให้สุ่มเลือก `"xgboost"` ด้วยความน่าจะเป็น `EXPLORATION_RATE` แทนที่จะ fallback ไป lightgbm เสมอ — เพื่อให้ log สะสมได้จนพอประเมิน R² ในที่สุด นอกเหนือจากนั้น (ไม่ explore) ยัง fallback lightgbm เหมือนเดิม เมื่อ xgboost มีข้อมูลครบแล้วจะกลับไปเทียบ R² จริงตามปกติ (exploration หยุดมีผลเองโดยอัตโนมัติ)

**ทดสอบแล้ว**: รัน `select_model()` 2000 ครั้งด้วย horizon ที่ยังไม่มีข้อมูล xgboost เลย ได้สัดส่วน xgboost 19.35% (เป้าหมาย 20%) ยืนยันว่า exploration ทำงานถูกต้องตามความน่าจะเป็นที่ตั้งไว้

**ไฟล์ที่แก้ในรอบนี้**: `app.py` (เพิ่ม `import random`, config `EXPLORATION_RATE`, แก้ `select_model()`)

---

## 2026-07-12 (4) — ผูก Dynamic Routing เข้ากับ get_forecast() จริง (XGBoost/Prophet ใช้งานได้แล้ว)

**ที่มา**: มีฟังก์ชัน `select_model()`/`compute_rolling_r2()`/`compute_missing_ratio()` และโมเดล XGBoost/Prophet พร้อมแล้วจาก step ก่อนหน้า ถึงเวลาผูกเข้ากับ pipeline พยากรณ์จริง

**สิ่งที่ทำ** ใน [app.py](app.py):
- โหลด `XGB_MODELS`, `PROPHET_MODEL` ตอน startup (ครั้งเดียว เหมือน LightGBM) จาก `models/xgb_pm25_multistep_20260712_015251.pkl` และ `models/prophet_pm25_20260712_015251.pkl`
- เพิ่ม `predict_with_prophet(prophet_model, current_weather, latest_time, horizon)`: สร้าง future dataframe ให้ Prophet โดยใช้ `current_weather` ซ้ำทุก horizon (ยังไม่มี weather forecast รายชั่วโมงจริง — จดไว้เป็น limitation ในโค้ด)
- Loop ใน `get_forecast()`: เรียก `select_model(X_now, h)` ก่อนทุก horizon แล้วเลือกเรียก `XGB_MODELS[h]`, `MODELS[h]` (lightgbm), หรือ `predict_with_prophet()` ตามผลลัพธ์ที่ได้ — เก็บ `model_used` ไว้ในแต่ละจุดของ `forecast[]` และใน `log_rows`
- `log_predictions()`: เพิ่มคอลัมน์ `model_used` ใน INSERT ตาม schema ที่ขยายไว้ก่อนหน้านี้
- Response `/api/forecast`: เพิ่ม field `"prophet_triggered_horizons"`

**ทดสอบแล้ว**: เรียก `get_forecast()` จริงครบ 24 horizon ผ่านปกติ, ทดสอบ `log_predictions()` เขียนคอลัมน์ `model_used` ถูกต้องทั้ง 3 ค่า (`xgboost`/`prophet`/`lightgbm`) ด้วย timestamp ปลอมแล้วลบทิ้งหลังยืนยัน

**ปัญหาที่พบระหว่างทดสอบ (ยังไม่แก้ — รอตัดสินใจ)**: รันจริงแล้วทุก horizon route ไป `"lightgbm"` หมด เพราะ `select_model()` จะเลือก xgboost ได้ก็ต่อเมื่อมี rolling R² ของ xgboost ที่ horizon นั้น (≥5 คู่ backfill แล้ว) แต่จะมีข้อมูลนั้นได้ก็ต่อเมื่อเคย route ไป xgboost มาก่อน — เป็น cold-start deadlock ในตัว logic proposal 4.1.6 เอง ไม่ใช่บั๊กจาก implementation

**หมายเหตุที่ยังไม่แก้ในรอบนี้**: `EFFECTIVE_THRESHOLD_BY_HORIZON` (ใช้เช็ค `risk_horizons`) คำนวณจาก bias ของ LightGBM เท่านั้น ถ้า horizon ไหนถูก route ไป xgboost/prophet threshold ที่ใช้เทียบจะไม่ได้ calibrate ตรงกับโมเดลนั้น

**ไฟล์ที่แก้ในรอบนี้**: `app.py` (โหลดโมเดลเพิ่ม, ฟังก์ชัน `predict_with_prophet`, แก้ loop ใน `get_forecast()`, แก้ `log_predictions()`, เพิ่ม field ใน response)

---

## 2026-07-12 (3) — เพิ่มฟังก์ชัน Dynamic Routing (compute_missing_ratio, compute_rolling_r2, select_model) ใน app.py

**ที่มา**: เตรียม logic การเลือกโมเดลตาม proposal หัวข้อ 4.1.6 — ยังไม่ผูกเข้ากับ `get_forecast()` ในรอบนี้ตามที่ตกลงไว้ (จะทำ step ถัดไป)

**สิ่งที่ทำ** ใน [app.py](app.py):
- เพิ่ม config `SIGMA_CRITICAL = 0.3`
- `compute_missing_ratio(feature_row)`: สัดส่วน NaN ใน feature_row เทียบกับ `FEATURE_COLS` ทั้งหมด (0-1)
- `compute_rolling_r2(model_name, horizon, N=24)`: query `predictions.db` เอา N แถวล่าสุดของ model_name/horizon นั้นที่มี `actual_pm25` แล้ว มาคำนวณ `r2_score`; คืน `None` ถ้าน้อยกว่า 5 คู่
- `select_model(feature_row, horizon)`: ถ้า missing_ratio > SIGMA_CRITICAL → `"prophet"`; ไม่งั้นเทียบ rolling R² ของ xgboost vs lightgbm ที่ horizon นั้น ถ้าตัวใดตัวหนึ่งเป็น `None` → fallback `"lightgbm"`, ถ้ามีครบทั้งคู่ → เลือกตัวที่ R² สูงกว่า

**ทดสอบแล้ว**: เทียบ missing_ratio หลายระดับ (0.0 / 0.278 / 0.333) ตรงตามที่คำนวณเอง, `compute_rolling_r2("xgboost", 1)` คืน `None` ถูกต้อง (ยังไม่เคย log แถวไหนเป็น xgboost เลย), `select_model` fallback เป็น lightgbm ตอน missing_ratio 0.278 (<0.3) และ xgboost data ไม่พอ, สลับไป prophet ตอน missing_ratio 0.333 (>0.3) ถูกต้องทั้งหมด

**ไฟล์ที่แก้ในรอบนี้**: `app.py` (เพิ่ม import `r2_score`, constant `SIGMA_CRITICAL`, และ 3 ฟังก์ชัน)

---

## 2026-07-12 (2) — เพิ่มคอลัมน์ model_used ใน predictions.db

**ที่มา**: เตรียมรองรับ Dynamic Routing — เดิม `prediction_log` ไม่มีคอลัมน์ระบุว่าแต่ละแถวพยากรณ์ด้วยโมเดลไหน (เพราะที่ผ่านมาใช้ weighted LightGBM ตัวเดียวเสมอ)

**สิ่งที่ทำ**:
- สำรอง `predictions.db` ไว้ที่ `predictions.db.bak_20260712_015529` ก่อนแก้ schema
- `ALTER TABLE prediction_log ADD COLUMN model_used TEXT DEFAULT 'lightgbm'`

**ยืนยันแล้ว**:
- จำนวนแถวก่อน/หลัง: 408 / 408 (ไม่มีข้อมูลหาย)
- `model_used` เป็น `'lightgbm'` ครบทุกแถวเก่า 408/408 (ไม่มี NULL หรือค่าอื่น)
- Schema ใหม่เป็น `TEXT` ไม่มี constraint จำกัดค่า พร้อมรับ `"xgboost"`/`"prophet"` ในอนาคต

**หมายเหตุ**: `log_predictions()` ใน `app.py` ยังไม่ได้ระบุ `model_used` ตอน INSERT (ยังพึ่ง column default อยู่) — ต้องแก้ตอนที่ Dynamic Routing เริ่มใช้งานจริง ไม่งั้นทุกแถวจะถูก mark เป็น lightgbm ผิดๆ

**ไฟล์ที่แก้ในรอบนี้**: `predictions.db` (ALTER TABLE), `predictions.db.bak_20260712_015529` (ใหม่ — backup)

---

## 2026-07-12 (1) — เทรน XGBoost และ Prophet เตรียมสำหรับ Circuit Breaker + Dynamic Routing

**ที่มา**: กำลังจะทำระบบ Circuit Breaker + Dynamic Routing ที่เลือกระหว่าง 3 โมเดล (XGBoost, LightGBM, Prophet) แต่มีแค่ LightGBM ที่ deploy จริง (`.pkl`) ส่วน XGBoost/Prophet เคยมีแค่ผลเปรียบเทียบใน `train_models.ipynb` ไม่เคย save เป็นไฟล์โมเดล

**เช็คก่อนเทรน**: ค้นหาทั้งโฟลเดอร์ `models/` และ notebook (`train_models.ipynb`, `Testforcest.ipynb`) — ไม่พบไฟล์ `.pkl` ของ XGBoost หรือ Prophet ที่เทรนไว้แล้วเลย จึงเทรนใหม่

**สิ่งที่ทำ**: สร้าง [scripts/train_xgb_prophet.py](scripts/train_xgb_prophet.py)
- โหลด `feature_cols`/`target_cols`/`station_id` จาก metadata ของ `models/lgb_pm25_multistep_weighted_20260709_004955.pkl` โดยตรง (ไม่ hardcode) เพื่อให้ตรงกับ LightGBM เป๊ะ
- ใช้ train/test split แบบ time-based 80/20 เดียวกับ `train_models.ipynb` (train 30327 แถว, test 7582 แถว)
- **XGBoost**: เทรนแบบ direct multi-step 24 โมเดลแยกตาม horizon, `n_estimators=200` (default, ตรงกับตอนเปรียบเทียบครั้งแรก), save เป็น dict `{horizon: model}` พร้อม metadata แบบเดียวกับ LightGBM
- **Prophet**: เทรน 1 โมเดล ด้วย `pm25` เป็น y, `temp`/`humidity`/`wind_speed`/`precipitation` เป็น extra regressors

**ผลลัพธ์ (ตรงกับผลเดิมเป๊ะ)**:
- XGBoost MAE เฉลี่ย 24 horizons: **2.728** (เดิม ~2.728) — MAE ต่อ horizon ตรงกับ `train_models.ipynb` ทุกค่า
- Prophet MAE: **7.586** (เดิม ~7.586)
- ทดสอบ predict 1 จุดจากโมเดลที่ save แล้วโหลดกลับมาใหม่ เทียบกับค่าจริง — ทำงานถูกต้องทั้งคู่

**ไฟล์โมเดลที่ได้**:
- `models/xgb_pm25_multistep_20260712_015251.pkl` (21.9 MB)
- `models/prophet_pm25_20260712_015251.pkl` (3.7 MB)

**หมายเหตุ**: โมเดลทั้ง 2 นี้ยังไม่ได้ผูกเข้ากับ `app.py`/`get_forecast()` — เป็นแค่ขั้นเตรียมโมเดลให้พร้อมสำหรับระบบ Circuit Breaker + Dynamic Routing ที่จะทำต่อ

**ไฟล์ที่แก้ในรอบนี้**: `scripts/train_xgb_prophet.py` (ใหม่), `models/xgb_pm25_multistep_20260712_015251.pkl` (ใหม่), `models/prophet_pm25_20260712_015251.pkl` (ใหม่)

---

## 2026-07-10 — แก้บั๊ก rolling window ผิดใน build_feature_row() (live/training feature mismatch)

**ที่มา**: สังเกตว่า live pooled MAE จาก `predictions.db` (4.64, n=68) สูงกว่า backtest (2.729) อย่างมีนัยสำคัญ จึงตรวจสอบว่า feature pipeline ตอน live (`build_feature_row()` ใน `app.py`) กับตอนเทรน (ไฟล์ `data/khonkaen_pm25_features_final.csv`) ตรงกันจริงหรือไม่

**บั๊กที่พบ**: `pm25_roll_mean_6`, `pm25_roll_std_6`, `pm25_roll_mean_24`, `pm25_roll_std_24` คำนวณด้วย window **ไม่รวมค่าปัจจุบัน** (`iloc[-7:-1]`, `iloc[-25:-1]`) แต่ training data ใช้ window **รวมค่าปัจจุบัน** (มาตรฐาน pandas `.rolling(N).mean()`) — ยืนยันด้วยการจำลองข้อมูลย้อนหลัง 10 จุดกระจายทั่ว dataset (2021-2026): window แบบเดิมผิด 39/40 ค่า, window ที่แก้แล้วตรง 0 mismatch (ทดสอบทั้งแบบ logic เดียวกันและเรียกฟังก์ชันจริงจาก `app.py`)

Lag features (`pm25_lag_1/3/6/12/24`) และ time features (hour_sin/cos ฯลฯ) ตรงกันอยู่แล้ว ไม่มีปัญหา — บั๊กจำกัดเฉพาะ 4 ฟีเจอร์ rolling แต่เป็นฟีเจอร์ที่มีน้ำหนักสูงในโมเดล

**สิ่งที่ทำ**:
1. แก้ `build_feature_row()` ใน [app.py](app.py): `iloc[-7:-1]` → `iloc[-6:]`, `iloc[-25:-1]` → `iloc[-24:]`
2. **ลบ `predictions.db` ทิ้งแล้วเริ่มนับใหม่** ตามที่ตัดสินใจ — ข้อมูล 240 รายการ (68 คู่เทียบได้) ที่ log ไว้ก่อนหน้านี้ทั้งหมดคำนวณด้วย feature ที่ผิด จึงไม่เก็บไว้ปนกับข้อมูลหลังแก้

**ผลกระทบที่ประเมินไว้ก่อนแก้ (ยืนยันแล้วว่าตรงตามคาด)**:
- กระทบเฉพาะ `build_feature_row()` ที่ใช้ใน `get_forecast()` (ทั้ง `/api/forecast` และ scheduler) — จุดเดียว
- **ไม่กระทบ** backtest/training pipeline ทั้งหมด (`train_models.ipynb`, `retrain_weighted.py`, `eval_high_pollution.py`, กราฟทั้ง 3 รูป, ตัวเลข MAE 2.729/bias/effective threshold ที่รายงานไปแล้ว) เพราะอ่าน feature จาก CSV ที่ถูกต้องอยู่แล้วโดยตรง ไม่เคยผ่าน `build_feature_row()`

**ทดสอบแล้ว**: 
- เทียบ `build_feature_row()` เวอร์ชันแก้แล้วกับค่าจริงใน CSV 10 จุดกระจายทั่ว dataset — 0 mismatch
- รัน `app.py` จริง เรียก `/api/forecast` ยืนยันว่าได้ค่าพยากรณ์ใหม่ (เปลี่ยนจากเดิมเพราะ input feature เปลี่ยน), `predictions.db` ถูกสร้างใหม่สะอาด (24 แถวจากการเรียกครั้งแรก ตรงตามคาด)
- เช็ค `/`, `/compare.html` ยังทำงานปกติ 200 ทั้งคู่

**สิ่งที่ต้องติดตามต่อ**: รอสะสม log ใหม่ (post-fix) ให้ได้ n มากพอ แล้วเทียบ live pooled MAE ใหม่กับ backtest อีกครั้ง ว่าช่องว่างที่เคยเห็น (4.64 vs 2.729) แคบลงจริงหรือไม่ — จะเป็นหลักฐานว่าบั๊กนี้คือสาเหตุหลักของ gap ที่เจอ

**ไฟล์ที่แก้ในรอบนี้**: `app.py` (แก้ rolling window logic), `predictions.db` (ลบแล้วสร้างใหม่)

---

## 2026-07-09 (10) — เพิ่ม scheduler เก็บ forecast อัตโนมัติทุก 1 ชม.

**เหตุผล**: ระบบ log เดิมเป็น lazy (ทำงานเฉพาะตอนมีคนเรียก `/api/forecast`) ทำให้เกิดช่องว่างข้อมูล (เช่น t+14 ที่ค้างไม่มีค่าจริงเพราะไม่มีใครเปิดเว็บพอดีเวลา) ผู้ใช้ยืนยันว่าบางครั้งจะรัน `app.py` ทิ้งไว้เฉยๆ โดยไม่เปิดหน้าเว็บดู จึงเพิ่ม background scheduler ในตัว `app.py` เอง (ไม่ต้องพึ่ง Windows Task Scheduler)

**สิ่งที่ทำ**: เพิ่ม `APScheduler` (`BackgroundScheduler`) เรียก `get_forecast()` เองทุก 1 ชั่วโมง (ยิงทันทีตอนสตาร์ทด้วย) ตราบใดที่ `app.py` ยังรันอยู่

**บั๊กที่เจอระหว่างทำ (สำคัญ ควรรู้)**:

1. **Reloader ทำให้ scheduler เริ่มซ้ำ 2 รอบ** — Flask `debug=True` สร้าง reloader process แยก (parent คอย watch ไฟล์ + child ที่รันจริง) ถ้าไม่กันไว้ scheduler จะถูกสร้างทั้ง 2 process แก้ด้วยเช็ค `WERKZEUG_RUN_MAIN` env var (ตั้งเฉพาะใน child) ก่อน start scheduler
2. **แก้ผิดจุดแรก**: เดิมเช็ค `app.debug` เพื่อสลับ logic กรณีไม่ใช้ debug mode — แต่ Flask ยังไม่ set `app.debug = True` จนกว่าจะเรียก `app.run(debug=True)` ทำให้เช็คตอนนั้นเจอ `app.debug == False` เสมอ (ค่า default) เงื่อนไข `not app.debug` เลยเป็น `True` ตลอด ทำให้ scheduler เริ่มใน**ทั้ง 2 process อยู่ดี** ทั้งที่ตั้งใจจะกันไว้ — แก้โดยใช้ตัวแปรคงที่ `DEBUG = True` ของเราเองแทน `app.debug`
3. **Print ภาษาไทยทำแอปพังตอน startup** — `print()` ข้อความไทยใน `scheduled_forecast_job()` และตอน log scheduler start ชนกับ default console encoding (cp1252) บน Windows ทำให้เกิด `UnicodeEncodeError` แล้ว **แอปทั้งตัวไม่ขึ้นเลย** (crash ตั้งแต่ module level ก่อนถึง `app.run()`) แก้โดยเปลี่ยนข้อความ print ที่ขึ้นจอ console เป็นภาษาอังกฤษล้วน (comment ในโค้ดยังเป็นไทยได้ปกติเพราะไม่ถูก print)

**วิธี debug ที่ใช้**: เนื่องจาก process ลูกที่ Werkzeug reloader spawn บน Windows ไม่ได้ inherit stdout ผ่าน redirect แบบเดียวกับ parent (เห็น log แค่บางส่วน) จึงเปลี่ยนมาเขียน debug marker ลงไฟล์ (`scheduler_debug.log`) แทน print ชั่วคราว ยืนยันได้ชัดว่า scheduler เริ่มถูก process เดียว (child, `WERKZEUG_RUN_MAIN='true'`) และ job แรกทำงานสำเร็จภายใน ~2.5 วินาที แล้วลบไฟล์ debug ทิ้งหลังยืนยันเสร็จ

**ทดสอบแล้ว**: รันจริงหลายรอบ ยืนยันว่า (1) ไม่ crash ตอน startup อีกแล้ว (2) scheduler เริ่มแค่ครั้งเดียวไม่ซ้ำ (3) job แรกยิงทันทีและเขียนลง `predictions.db` สำเร็จ (4) endpoint อื่นทั้งหมด (`/`, `/api/forecast`, `/compare.html`, `/api/comparison/overview`) ยังทำงานปกติไม่ได้รับผลกระทบ

**Dependency ใหม่**: `pip install apscheduler` (เพิ่มเข้า requirements ที่ต้องติดตั้ง)

**ไฟล์ที่แก้ในรอบนี้**: `app.py` (เพิ่ม scheduler + แก้ debug-detection bug + แก้ print encoding)

---

## 2026-07-09 (9) — แก้ high_pollution_risk ให้ใช้ per-horizon threshold แทนค่าคงที่ 37.5

**ปัญหาที่พบ** (จากการรีวิวก่อนปิดงาน): `HIGH_POLLUTION_THRESHOLD = 37.5` เดิมเทียบกับค่า predicted ตรงๆ ทุก horizon โดยไม่หัก bias ที่ยังเหลืออยู่ในโมเดล weighted (+1.03 เฉลี่ย, +2.53 ที่ t+24) ทำให้เกิดช่องโหว่: ถ้าโมเดลทาย 36.0 ที่ t+24 (ต่ำกว่า threshold, banner ไม่ขึ้น) แต่ค่าจริงอาจสูงถึง ~38.5 (36+2.53) ซึ่งเกิน 37.5 ไปแล้ว — ระบบจะไม่เตือนทั้งที่ควรเตือน

**สิ่งที่ทำ**:

1. โหลด `bias_high_weighted` ต่อ horizon จาก `models/eval_weighted_vs_baseline_20260709_004955.csv` ตอน startup (`BIAS_HIGH_BY_HORIZON` dict) — ไม่ hardcode ในโค้ด เผื่อเทรนโมเดลใหม่ทีหลังแค่ชี้ไปไฟล์ CSV ใหม่พอ
2. คำนวณ `EFFECTIVE_THRESHOLD_BY_HORIZON[h] = 37.5 - bias_high_weighted[h]` ต่อ horizon
3. แก้ loop ใน `get_forecast()` ให้เทียบแต่ละจุดพยากรณ์กับ effective threshold **ของ horizon นั้นๆ** แทนค่าคงที่ 37.5 (ค่าปัจจุบัน `latest_pm25` ยังเทียบกับ 37.5 ตรงๆ เหมือนเดิม เพราะเป็นข้อมูลจริงไม่มี bias จากโมเดล)
4. เพิ่ม field `risk_horizons` ใน response ของ `/api/forecast` — list ของ horizon ที่ trigger จริง (เช่น `[12, 24]`) เผื่ออนาคตทำ UI ระบุช่วงเวลาเสี่ยงเจาะจงได้แทนบอกแค่ true/false รวม

**Effective threshold ที่ได้จริงทุก horizon** (37.5 − bias_high_weighted):

| Horizon | Bias | Threshold || Horizon | Bias | Threshold || Horizon | Bias | Threshold |
|---|---|---||---|---|---||---|---|---|
| t+1 | 0.140 | 37.36 || t+9 | 0.659 | 36.84 || t+17 | 1.189 | 36.31 |
| t+2 | 0.194 | 37.31 || t+10 | 0.802 | 36.70 || t+18 | 1.385 | 36.12 |
| t+3 | 0.310 | 37.19 || t+11 | 0.870 | 36.63 || t+19 | 1.441 | 36.06 |
| t+4 | 0.375 | 37.13 || t+12 | 0.837 | 36.66 || t+20 | 1.579 | 35.92 |
| t+5 | 0.408 | 37.09 || t+13 | 0.996 | 36.50 || t+21 | 1.976 | 35.52 |
| t+6 | 0.569 | 36.93 || t+14 | 0.863 | 36.64 || t+22 | 1.813 | 35.69 |
| t+7 | 0.612 | 36.89 || t+15 | 0.923 | 36.58 || t+23 | 2.350 | 35.15 |
| t+8 | 0.667 | 36.83 || t+16 | 1.179 | 36.32 || t+24 | 2.528 | 34.97 |

ไล่ลดลงตามแนวโน้มที่คาด (มี wobble เล็กน้อยที่ t+9, t+12, t+14 เพราะเป็น bias ที่วัดจริงจาก test set ไม่ใช่ค่า smooth)

**ทดสอบแล้ว**: `import app` โหลด CSV และคำนวณ threshold ได้ครบ 24 ค่า, รัน `app.py` จริงเรียก `/api/forecast` ยืนยันว่า `risk_horizons` และ `high_pollution_risk` คำนวณถูกต้อง (ตอนทดสอบ pm25 จริง/พยากรณ์อยู่ที่ ~20-24 ต่ำกว่าทุก threshold จึงได้ `risk_horizons: []`), เช็ค `/api/comparison` และ `/api/comparison/overview` ยังทำงานปกติไม่ได้รับผลกระทบ

**ไฟล์ที่แก้ในรอบนี้**: `app.py` (เพิ่ม per-horizon threshold logic + field ใหม่)

---

## 2026-07-09 (8) — เพิ่มสรุปความคลาดเคลื่อนรวมทุก horizon

**เหตุผล**: ส่วนสรุปที่ทำไปรอบก่อน (2026-07-09 (7)) แสดงผลได้ทีละ 1 horizon ตามที่เลือกใน dropdown เท่านั้น ยังไม่มีภาพรวม "ทั้งระบบ" ให้ดูในแวบเดียว

**วิธีสรุป "ความคลาดเคลื่อนรวม" ที่เลือกใช้**: MAE/bias **แบบ pooled** — เอา error ของทุกคู่ (predicted, actual) จากทุก horizon มารวมกันแล้วหาค่าเฉลี่ยเดียว (ไม่ใช่เฉลี่ยของ MAE รายhorizon อีกที) เพราะสะท้อน "ประสบการณ์จริง" ของผู้ใช้ที่เจอค่าพยากรณ์คลาดเคลื่อนเท่าไรโดยเฉลี่ยไม่ว่าจะดูที่ horizon ไหน แต่เสริมด้วย breakdown รายhorizon (กราฟแท่งเล็ก + best/worst horizon) กำกับไว้ด้วย เพื่อไม่ให้ตัวเลขเดียวบดบังความจริงที่ว่า horizon ไกลคลาดเคลื่อนมากกว่าโดยธรรมชาติ

**สิ่งที่ทำ**:

1. **Backend**: เพิ่ม endpoint `GET /api/comparison/overview` ใน [app.py](app.py) — query `prediction_log` ทั้งตาราง (ไม่กรอง horizon) คืน overall MAE/bias (pooled), MAE รายhorizon ทั้ง 24 ตัว (null ถ้ายังไม่มีข้อมูล), และ horizon ที่แม่นยำที่สุด/คลาดเคลื่อนมากที่สุด
2. **Frontend**: เพิ่มการ์ด "ความคลาดเคลื่อนรวม (ทุก horizon รวมกัน)" ไว้บนสุดของ [compare.html](compare.html) (เหนือตัวเลือก horizon เดิม) ประกอบด้วย badge คุณภาพ, ประโยคสรุปรวม, หมายเหตุ best/worst horizon, และกราฟแท่งเล็ก 24 แท่งแสดง MAE แต่ละ horizon (horizon ที่ยังไม่มีข้อมูลจริงมาเทียบแสดงเป็นแท่งจางๆ แทน)

**ทดสอบแล้ว**: รัน `app.py` จริง เรียก `/api/comparison/overview` ได้ผลจริง (overall_mae=3.06, best=t+3 mae 0.9, worst=t+11 mae 7.4, horizon 13-24 เป็น null ถูกต้องเพราะยังไม่ถึงเวลา) และเช็ค `/compare.html` คืน 200 ปกติ

**ไฟล์ที่แก้ในรอบนี้**: `app.py` (เพิ่ม endpoint ใหม่), `compare.html` (เพิ่ม CSS/markup/JS การ์ดภาพรวม)

---

## 2026-07-09 (7) — เพิ่มสรุปภาษาคนอ่านง่ายในหน้าเปรียบเทียบ

**สิ่งที่ทำ**: เพิ่มส่วนสรุปแบบประโยคภาษาไทยธรรมดาใน [compare.html](compare.html) เหนือกลุ่มตัวเลข MAE/Bias/n_matched เดิม (ตัวเลขดิบยังอยู่เหมือนเดิมด้านล่าง ไม่ได้ลบ)

- **Badge คุณภาพ** (แม่นยำมาก/แม่นยำดี/แม่นยำปานกลาง/คลาดเคลื่อนค่อนข้างมาก) คำนวณจาก MAE เทียบ threshold
- **ประโยคสรุป** อ่านแล้วเข้าใจทันที เช่น "จากพยากรณ์ล่วงหน้า 1 ชั่วโมง ที่มีค่าจริงมาเทียบได้แล้ว 1 ครั้ง (จากทั้งหมด 2 ครั้งที่เคยพยากรณ์ไว้) ค่าพยากรณ์คลาดเคลื่อนจากความเป็นจริงโดยเฉลี่ย ±2.80 µg/m³ และมีแนวโน้มทายค่าสูงกว่าความเป็นจริง โดยเฉลี่ยประมาณ 2.80 µg/m³"
- **หมายเหตุกำกับ** เมื่อเลือก horizon ≥ 12 ชม. เตือนว่าความคลาดเคลื่อนที่มากขึ้นตามธรรมชาติของ horizon ไกล ไม่ได้แปลว่าโมเดลแย่ลง
- จัดการเคส **ยังไม่มีข้อมูลมาเทียบเลย** (n_matched=0) ด้วยข้อความอธิบายแทนตัวเลข 0 ที่อาจสับสน

**ทดสอบแล้ว**: รัน `app.py` จริง เรียก `/api/comparison?horizon=1` ได้ข้อมูลจริง (mae=2.8, bias=-2.8, n_matched=1/2) ตรวจสอบ logic การแปลงเป็นประโยคด้วยมือว่าตรงกับเคสนี้ถูกต้อง (bias ติดลบ = โมเดลทายสูงกว่าจริง)

**ไฟล์ที่แก้ในรอบนี้**: `compare.html` (เพิ่ม CSS + markup + JS ส่วนสรุป ไม่แตะ backend)

---

## 2026-07-09 (6) — จัดระเบียบไฟล์โปรเจกต์

**สิ่งที่ทำ**: ย้ายไฟล์ script/รูปที่กองอยู่ที่ root เข้าโฟลเดอร์ให้เป็นระเบียบ

- สร้าง `scripts/` ย้าย script วิเคราะห์/เทรนทั้งหมดเข้าไป: `eval_high_pollution.py`, `retrain_weighted.py`, `recompute_r2_3models.py`, `plot_mae_comparison.py`, `plot_bias_comparison.py`, `plot_r2_3models.py`
- สร้าง `figures/` ย้ายรูปที่ generate ไว้ทั้งหมดเข้าไป: `forecast_test.png`, `mae_comparison_baseline_vs_weighted.png`, `bias_comparison_baseline_vs_weighted.png`, `r2_comparison_3models_by_horizon.png`
- อัปเดต `OUT_PATH` ในทั้ง 3 plot script ให้ชี้ไปที่ `figures/` แทน root เพื่อให้ re-run แล้วได้ไฟล์ตำแหน่งเดิมที่ถูกต้อง
- อัปเดตลิงก์ในไฟล์นี้ (CHANGELOG.md) ให้ตรงกับตำแหน่งใหม่

**สิ่งที่ตั้งใจ "ไม่ย้าย"** (เพราะ path ข้างในอ้างอิงแบบ relative หรือเป็น entry point ที่ Flask ต้องหาเจอ):
- `app.py`, `index.html`, `compare.html`, `predictions.db`, `.env` — ต้องอยู่ root เดียวกันเพราะ `app.py` ใช้ relative path (`models/...`, `index.html`, `predictions.db`) อ้างอิงตำแหน่งปัจจุบัน
- `train_models.ipynb`, `Testforcest.ipynb` — ใช้ relative path อ้าง `data/...`, `models/...` เหมือนกัน ถ้าย้ายเข้า subfolder จะพังตอนรันซ้ำ (ต้องแก้ path ในทุกเซลล์ก่อน ซึ่งเสี่ยงมากกว่าประโยชน์ที่ได้)
- `data/`, `models/` — เป็น input/output หลักที่ทั้ง notebook และ `app.py` ผูก path ไว้อยู่แล้ว

**ทดสอบแล้ว**: รัน 3 plot scripts จากตำแหน่งใหม่ (`scripts/`) ยืนยันว่า save ไป `figures/` ถูกต้อง, `import app` ยืนยันว่า `app.py` ยังโหลดโมเดลได้ปกติ 24 ตัวหลังย้ายไฟล์ (ไม่ได้แตะของที่มันพึ่งพา)

**โครงสร้างหลังจัดระเบียบ**:
```
Model1/
├── app.py, index.html, compare.html, predictions.db, .env
├── train_models.ipynb, Testforcest.ipynb
├── CHANGELOG.md
├── data/khonkaen_pm25_features_final.csv
├── models/  (โมเดล .pkl + ผลประเมิน .csv ทั้งหมด)
├── scripts/ (script วิเคราะห์/เทรน/พล็อต แบบ standalone)
└── figures/ (รูปที่ generate ไว้ใช้ในเล่ม progress)
```

---

## 2026-07-09 (5) — เพิ่มกราฟ bias comparison + R² by horizon 3 โมเดล

**สิ่งที่ทำ**:

1. **[bias_comparison_baseline_vs_weighted.png](figures/bias_comparison_baseline_vs_weighted.png)** — เส้น bias เฉพาะช่วง PM2.5 สูง (baseline vs weighted) ตาม horizon พร้อมเส้นอ้างอิง `bias = 0` (ไม่มีอคติ) เห็นชัดว่า weighted model มี bias ต่ำกว่า baseline ตลอดทุก horizon แม้ยังไม่เป็นศูนย์สนิท
2. **[r2_comparison_3models_by_horizon.png](figures/r2_comparison_3models_by_horizon.png)** — เทียบ R² ตาม horizon ของ LightGBM/XGBoost (คำนวณใหม่จากข้อมูลจริงเพื่อความชัวร์ ไม่ใช่จำจากบทสนทนาเก่า, สคริปต์ [recompute_r2_3models.py](scripts/recompute_r2_3models.py)) กับ Prophet ซึ่งแสดงเป็นเส้นแนวนอน (ค่าเดียว 0.540) พร้อมหมายเหตุกำกับว่า Prophet เป็น nowcast ไม่ใช่ multi-horizon เทียบกันตรงๆ ไม่ได้ 100%

**ปัญหาที่เจอระหว่างทำ**: กราฟรอบแรกมี legend/title/หมายเหตุทับกันในทั้ง 2 ภาพ (Segoe UI ก็อ่านอักษรไทยไม่ได้ในกราฟก่อนหน้านี้ด้วย) แก้ด้วยการย้าย legend ไปช่องว่างที่ไม่ทับข้อมูล, ย้ายหมายเหตุ Prophet ออกไปเป็น footnote นอกกรอบกราฟ, เพิ่ม pad_inches กัน crop ชิดเกินไป

**ไฟล์ที่เพิ่มในรอบนี้**: `bias_comparison_baseline_vs_weighted.png`, `r2_comparison_3models_by_horizon.png`, `plot_bias_comparison.py`, `plot_r2_3models.py`, `recompute_r2_3models.py`, `models/r2_comparison_3models_by_horizon.csv`

---

## 2026-07-09 (4) — สร้างกราฟเปรียบเทียบ MAE (baseline vs weighted) สำหรับเล่ม progress

**สิ่งที่ทำ**: พล็อตกราฟ 2 panel เปรียบเทียบ MAE ตาม horizon ระหว่างโมเดล baseline กับ weighted (จากข้อมูลใน `models/eval_weighted_vs_baseline_20260709_004955.csv`) — panel ซ้าย = MAE ภาพรวมทุกช่วง, panel ขวา = MAE เฉพาะช่วง PM2.5 สูง (>37.5) เพื่อให้เห็น trade-off ชัดเจนในภาพเดียว

- ใช้สีจาก validated categorical palette (blue/aqua ผ่าน dataviz color validator, ΔE 73.6 — ผ่านเกณฑ์ colorblind-safe) + เส้นทึบ/ประ แยกกันเป็น secondary encoding เผื่อพิมพ์ขาวดำ
- เพิ่ม direct label ค่าตัวเลขที่ปลายเส้น (แก้ปัญหา contrast ของสี aqua บนพื้นขาวที่ไม่ผ่านเกณฑ์ 3:1)
- ใช้ฟอนต์ TH SarabunPSK (ฟอนต์ราชการไทยที่มีอยู่ในเครื่อง) แทน Segoe UI เพราะ Segoe UI ไม่มี glyph ภาษาไทย
- บันทึกที่ [mae_comparison_baseline_vs_weighted.png](figures/mae_comparison_baseline_vs_weighted.png) ความละเอียด 200 DPI พร้อมใช้ในเอกสาร
- Script พล็อตเก็บไว้ที่ [plot_mae_comparison.py](scripts/plot_mae_comparison.py) เผื่อต้องปรับ/re-generate

**ไฟล์ที่เพิ่มในรอบนี้**: `mae_comparison_baseline_vs_weighted.png` (ใหม่), `plot_mae_comparison.py` (ใหม่)

---

## 2026-07-09 (3) — เปลี่ยนธีมหน้าเว็บเป็น "ท้องฟ้าเปลี่ยนสีตามคุณภาพอากาศ"

**สิ่งที่ทำ**: ปรับดีไซน์ [index.html](index.html) และ [compare.html](compare.html) ใหม่ทั้งคู่ให้เป็นธีมเดียวกัน

- **พื้นหลังไล่สีตามสถานะ PM2.5 จริง** (เฉพาะ index.html เพราะมีค่า "ปัจจุบัน" ให้อ้างอิง): ฟ้าใส (ดี) → ส้มเหลืองหมอกบาง (ปานกลาง) → น้ำตาลเข้มหมอกหนา (เริ่มมีผลกระทบต่อสุขภาพ) เปลี่ยนอัตโนมัติทุกครั้งที่โหลดข้อมูลใหม่ผ่าน JS (`document.body.className`)
- **การ์ดแบบ glassmorphism** (โปร่งแสง + เบลอพื้นหลัง) ลอยอยู่บนพื้นหลังท้องฟ้า แยกอิสระจาก dark/light mode ของเครื่อง (ธีมสีท้องฟ้ามาจากข้อมูล, โทนการ์ดมาจาก OS preference)
- **ไอคอนสภาพอากาศ** เปลี่ยนตามสถานะ: ☀️ ดี / 🌤️ ปานกลาง / 😷 เริ่มมีผลกระทบต่อสุขภาพ
- **ก้อนหมอกลอยเบาๆ** เป็นฉากหลัง (CSS blur + animation) เพิ่มบรรยากาศ
- compare.html ใช้พื้นหลังฟ้าแบบคงที่ (ไม่ dynamic เพราะหน้านี้เทียบข้อมูลย้อนหลังหลาย horizon ไม่มี "สถานะปัจจุบัน" เดียวให้อ้างอิง) แต่ใช้ภาษาดีไซน์เดียวกัน (การ์ด, ก้อนหมอก, ฟอนต์)

**ทดสอบแล้ว**: รัน `app.py` จริง เช็คทั้ง `/`, `/compare.html`, `/api/forecast` คืน 200 ปกติ, ตรวจไฟล์ HTML ทั้งสองไม่มี tag ขาด/เกิน

**ไฟล์ที่แก้ในรอบนี้**: `index.html`, `compare.html` (ปรับ CSS/markup ทั้งคู่ ไม่แตะ backend/logic)

---

## 2026-07-09 (2) — เพิ่มระบบ log เปรียบเทียบผลพยากรณ์กับค่าจริง + หน้าเว็บดูผล

**เหตุผล**: metric ทั้งหมดที่มี (รวมถึงผล weighted model) มาจาก backtest บนไฟล์ CSV เท่านั้น ยังไม่เคยพิสูจน์กับข้อมูลสดจริง จึงต้อง log ทุกครั้งที่พยากรณ์ แล้วเทียบกับค่าจริงเมื่อถึงเวลานั้น

**สิ่งที่ทำ**:

1. **Logging + backfill** ใน [app.py](app.py)
   - เพิ่ม SQLite (`predictions.db`, ตาราง `prediction_log`) เก็บทุก prediction: `predicted_at_utc`, `target_time_utc`, `horizon`, `predicted_pm25`, `actual_pm25`
   - `log_predictions()` บันทึกทุกครั้งที่ `/api/forecast` ถูกเรียก (กันซ้ำด้วย `UNIQUE(predicted_at_utc, horizon)` — เรียกรัวๆ ในชั่วโมงเดียวกันไม่สร้างแถวซ้ำ)
   - `backfill_actuals()` เติมค่าจริงย้อนหลังให้ prediction เก่าโดยใช้ข้อมูล PM2.5 ที่ดึงมาอยู่แล้วในทุกคำขอ (ไม่ต้องยิง API เพิ่ม, ไม่ต้องมี scheduler แยก — เติมค่าแบบ lazy ทุกครั้งที่มีคนเปิดหน้าเว็บ)
   - เพิ่ม endpoint `GET /api/comparison?horizon=<1-24>` คืนจุดข้อมูล (พยากรณ์/ค่าจริง) และสรุป MAE, bias, จำนวนคู่ที่เทียบได้ เฉพาะ horizon ที่เลือก

2. **หน้าใหม่ [compare.html](compare.html)** (route `/compare.html`)
   - เลือก horizon (t+1 ถึง t+24) แล้ววาดกราฟเส้นเทียบค่าพยากรณ์ (น้ำเงิน) กับค่าจริง (เขียว) ด้วย SVG ล้วน ไม่พึ่ง library ภายนอก
   - แสดงสรุป MAE / bias / จำนวนคู่ที่เทียบได้ ด้านบนกราฟ
   - มี loading skeleton + error state + ปุ่มลองใหม่ เหมือนหน้าแรก
   - เพิ่มลิงก์ "ดูความแม่นยำจริง →" จากหน้าแรก ([index.html](index.html)) ไปหน้านี้

**ข้อจำกัดที่รู้อยู่แล้ว**: การ log/backfill ทำงานแบบ lazy (ต่อเมื่อมีคนเรียก `/api/forecast`) ไม่มี background job แยก — ถ้าไม่มีใครเปิดเว็บเลยหลายวัน ข้อมูลจะไม่ถูกบันทึก/เติมค่าจริงในช่วงนั้น และเพิ่งเริ่ม log จากวันนี้ ข้อมูลเปรียบเทียบจะค่อยๆ สะสมขึ้น (ยังไม่มี historical backfill ของ prediction เก่าก่อนหน้านี้)

**ทดสอบแล้ว**: รัน `app.py` จริง เรียก `/api/forecast` 2 ครั้งติดกันยืนยัน dedup ทำงาน (24 แถวไม่ซ้ำ ไม่ใช่ 48), เพิ่มแถวทดสอบเทียม target time ย้อนหลังแล้วเรียก `/api/forecast` อีกครั้งยืนยันว่า `backfill_actuals()` เติมค่าจริงถูกต้อง (predicted=20.0, ได้ actual=22.1 ตรงกับข้อมูลจริง), เช็ค `/compare.html` และ `/api/comparison?horizon=1` คืนค่า 200 ถูกต้อง

**ไฟล์ที่เพิ่ม/แก้ในรอบนี้**: `compare.html` (ใหม่), `app.py` (แก้ — เพิ่ม logging/backfill + 2 endpoints), `index.html` (แก้ — เพิ่มลิงก์), `predictions.db` (ไฟล์ข้อมูลใหม่ สร้างอัตโนมัติตอนรัน)

---

## 2026-07-09 — แก้ bias ประเมินค่าต่ำในช่วงมลพิษสูง + เพิ่มคำเตือนใน UI

**ปัญหาที่พบ**: จากการประเมินโมเดล LightGBM เดิมเฉพาะช่วงที่ PM2.5 จริงเกิน 37.5 µg/m³ พบว่าโมเดลทายค่าต่ำกว่าความจริงอย่างเป็นระบบ (bias เฉลี่ย +1.96 ทุก horizon, สูงสุด +4.47 ที่ t+24) ซึ่งเป็นความเสี่ยงสำหรับการใช้เตือนภัยสุขภาพ

**สิ่งที่ทำ**:

1. **Retrain ด้วย sample weighting** ([retrain_weighted.py](scripts/retrain_weighted.py))
   - เพิ่มน้ำหนักให้ training rows ที่ actual PM2.5 > 37.5 (เลือก weight ต่อ horizon ผ่าน validation split แยกจาก test set, interpolate ค่า weight 1.0–10.0 ตาม horizon)
   - เทรนโมเดลใหม่ทั้ง 24 horizons บน train set เต็ม แล้ว evaluate บน test set จริง
   - บันทึกโมเดลใหม่: `models/lgb_pm25_multistep_weighted_20260709_004955.pkl`
   - บันทึกผลเปรียบเทียบ: `models/eval_weighted_vs_baseline_20260709_004955.csv`

2. **อัปเดต [app.py](app.py)**
   - เปลี่ยน `MODEL_PATH` ให้ชี้ไปที่โมเดลใหม่ (weighted)
   - เพิ่ม field `high_pollution_risk` ใน response ของ `/api/forecast` (true ถ้าค่าปัจจุบันหรือค่าพยากรณ์จุดใดเกิน 37.5)

3. **อัปเดต [index.html](index.html)**
   - เพิ่มกล่องเตือนสีแดง (`#biasWarning`) แสดง/ซ่อนอัตโนมัติตามฟิลด์ `high_pollution_risk` เตือนว่าค่าจริงอาจสูงกว่าที่พยากรณ์ในช่วงมลพิษสูง

**ผลลัพธ์ (บน test set จริง, เทียบ baseline vs weighted)**:

| Metric | Baseline | Weighted |
|---|---|---|
| MAE ภาพรวม (เฉลี่ยทุก horizon) | 2.511 | 2.729 (+8.7%) |
| MAE เฉพาะช่วง PM2.5 > 37.5 | 4.071 | 3.709 (−8.9%) |
| Bias เฉลี่ยช่วงสูง (ทายต่ำกว่าจริง) | +1.959 | +1.028 (−47.5%) |

**Trade-off**: แม่นยำภาพรวมลดลงเล็กน้อยเพื่อแลกกับการลด bias ในช่วงที่สำคัญที่สุด (มลพิษสูง) — ยัง**ไม่ได้กำจัด bias หมด** โดยเฉพาะที่ horizon ไกลๆ (t+24 ยังมี bias +2.53) จึงเพิ่มคำเตือนใน UI เป็นการป้องกันชั้นที่สองด้วย

**ทดสอบแล้ว**: รัน `app.py` จริง เรียก `/api/forecast` จริง ยืนยันว่าโหลดโมเดลใหม่ได้ และ field `high_pollution_risk` คำนวณถูกต้อง (ตอนทดสอบค่าจริงอยู่ที่ ~16-22 จึงเป็น `false` — ยังไม่เห็น banner ทำงานจริงเพราะไม่มีข้อมูลช่วงมลพิษสูงในเวลาที่ทดสอบ)

**ไฟล์ที่เพิ่ม/แก้ในรอบนี้**: `retrain_weighted.py` (ใหม่), `eval_high_pollution.py` (ใหม่, ก่อนหน้านี้ใช้วิเคราะห์ bias), `app.py` (แก้), `index.html` (แก้)

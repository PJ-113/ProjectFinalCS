# Test Results

บันทึกผลการทดสอบ/ประเมินโมเดลที่ไม่ใช่แค่ unit test ปกติ (เช่น backtest, การประเมิน scenario เฉพาะ, การเทียบ performance ระหว่างเงื่อนไข) — เรียงจากล่าสุดไปเก่าสุด (prepend เหมือน CHANGELOG.md)

---

## 2026-07-20 — Post-hoc bias correction สำหรับ XGBoost (ทางเลือกแทน sample weighting) — **ล้มเหลวชัดเจน ไม่ deploy**

**วัตถุประสงค์**: sample weighting ของ XGBoost ล้มเหลวไปแล้ว 2 รอบ (ดูรายการด้านล่าง, สรุปว่า structural mismatch จริง) ผู้ใช้เสนอวิธีใหม่ที่ไม่แตะโมเดล/ไม่ retrain เลย — ชดเชย raw prediction ของ XGBoost บวก `bias_high` ต่อ horizon ที่คำนวณไว้แล้ว (`models/xgb_bias_high_by_horizon_20260712_191350.csv`) เฉพาะตอน raw prediction เข้าใกล้โซนมลพิษสูง (ramp เชิงเส้นแทน hard cutoff เพื่อไม่ให้เกิดรอยต่อ) โดยมี "context gate" เสริม (ใช้ `pm25_lag_1` — ค่าจริงล่าสุดที่รู้แน่นอน ณ เวลาพยากรณ์ — เป็นหลักฐานยืนยันก่อนชดเชยเต็ม)

**วิธีทดสอบ**: `scripts/eval_xgb_posthoc_correction.py` (สร้างใหม่)
- ข้อมูล/split เดียวกับที่ใช้เทรนโมเดล deploy จริง: `data/khonkaen_pm25_features_raw_20260712_191249.csv`, chronological 80/20 (train/test) แล้ว nested 85/15 ของ train (`train_tune`/`valid_tune`, ช่วง 2025-09-08 ถึง 2026-01-23)
- **จุดสำคัญที่แก้หลังพบว่าพลาดตอนร่างสคริปต์รอบแรก**: `bias_high_by_horizon` มาจาก test set (20% ท้ายสุด, 2026-01-23 ถึง 2026-07-02) เดิมอยู่แล้ว — ถ้าเอา correction ที่ได้จากตัวเลขนั้นไปทดสอบกับ test set เดิมซ้ำจะเป็นการทดสอบวนกลับ (circular): บวกเลขที่คำนวณจากชุดข้อมูลนั้นเองกลับเข้าไปในชุดเดิม แล้วบอกว่า "bias หายเกือบหมด" (รอบแรกวัดได้ bias ดีขึ้น 94-99% ซึ่งเป็นภาพลวงตาล้วนๆ) แก้โดยเลือก config (โซน ramp + context gate) จาก `valid_tune` เท่านั้น (คนละช่วงเวลากับ test, ไม่เคยถูกใช้คำนวณ `bias_high_by_horizon`) แล้วค่อยยืนยันครั้งเดียวบน test จริง
- เกณฑ์รับ (ตามที่ผู้ใช้กำหนด): MAE_all ต้องไม่แย่ลงเกิน 8% เทียบ baseline (guard เดียวกับตอนเลือก weight ของ sample weighting)

**ผลลัพธ์ (บน `valid_tune` — ตัวเลข generalization ที่เชื่อถือได้, เฉลี่ยทั้ง 24 horizon, ทุก config ที่ลอง)**:

| Metric | Baseline (raw XGBoost) | Corrected (ทุก config ที่ลอง) |
|---|---|---|
| `\|bias_high\|` เฉลี่ย | **0.040** | 2.99–3.00 |
| MAE_all เปลี่ยนแปลง | — | **แย่ลง 138% ถึง 234%** |
| Horizon ที่แย่ลงพร้อมกันทั้ง MAE และ bias | — | **24/24 ทุก config** |

ไม่มี config ไหน (โซน ramp 4 แบบ × มี/ไม่มี context gate = 8 config) ผ่านเกณฑ์ 8% เลยแม้แต่อันเดียว — อันที่แย่น้อยที่สุด (`zone=(28,37.5)`, มี context gate) ก็ยังแย่ลง 138%

**ข้อค้นพบสำคัญที่สุด**: บน `valid_tune` (ช่วง ก.ย. 2025 – ม.ค. 2026, ส่วนใหญ่นอกฤดูฝุ่น) โมเดล XGBoost ที่ deploy จริง**แทบไม่มี bias ในช่วงมลพิษสูงเลย** (`|bias_high|` เฉลี่ยแค่ 0.04, มีตัวอย่างช่วงสูงเพียงพอให้เชื่อถือได้ — 441 แถวต่อ horizon ไม่ใช่ noise จากตัวอย่างน้อย) ต่างจากบน test set (ช่วง ม.ค.–ก.ค. 2026 ซึ่งรวมฤดูฝุ่นเข้มข้น ก.พ.-มี.ค.) ที่ bias เฉลี่ยสูงถึง 3.04 — ตรงกับ backtest ฤดูฝุ่นที่เคยวัดไว้ (`bias_high` XGBoost = +2.473 เฉพาะช่วง ธ.ค.-มี.ค., ดูรายการ 2026-07-16 ด้านล่าง) **แปลว่า bias_high ไม่ใช่คุณสมบัติคงที่ของโมเดลตลอดเวลา แต่กระจุกตัวเฉพาะช่วงฤดูฝุ่น** — ค่าคงที่ (fixed additive constant) ที่คำนวณจากค่าเฉลี่ยทั้งปีของ test set แล้วเอาไปบวกกลับเข้า prediction ทุกครั้งที่เข้าใกล้โซนสูง จึงเป็นการ**ชดเชยเกินจริงอย่างรุนแรงในช่วงที่ไม่ใช่ฤดูฝุ่น** (ซึ่งเป็นส่วนใหญ่ของปี) ทำให้ MAE รวมพังทั้งระบบ นี่คือเหตุผลที่ threshold-compensation ที่ใช้อยู่จริงใน `app.py` (`EFFECTIVE_THRESHOLD_BY_HORIZON_BY_MODEL`) ปลอดภัยกว่าวิธีนี้มาก — มันแค่ขยับ**เกณฑ์เปรียบเทียบ**ตายตัวไม่ได้แก้ค่าที่แสดงจริง เลยไม่มีความเสี่ยง "ชดเชยผิดช่วงเวลา" แบบนี้

**สรุป**: **ไม่ deploy** — ล้มเหลวชัดเจนกว่า sample weighting เสียอีก (sample weighting อย่างน้อยยังดีขึ้นเล็กน้อยเฉลี่ยรวม แค่ไม่สม่ำเสมอข้าม horizon; post-hoc correction แบบนี้แย่ลงทุก horizon ทุก config อย่างมาก) ไม่แนะนำให้ลองปรับ zone/gate เพิ่มเติมด้วย recipe เดิม เพราะปัญหาไม่ใช่พารามิเตอร์ผิด แต่เป็นเพราะตัว `bias_high_by_horizon` เองไม่ใช่ค่าคงที่ตามเวลา ถ้าจะทำ post-hoc correction ให้ได้ผลจริงต้องเป็น bias ที่ **แปรตามฤดูกาล/บริบทปัจจุบัน** ไม่ใช่ค่าเฉลี่ยทั้งปีคงที่ — เป็นงานเพิ่มเติมที่ใหญ่กว่านี้มาก ไม่ได้ทำต่อในรอบนี้

**ไฟล์ที่เกี่ยวข้อง**: `scripts/eval_xgb_posthoc_correction.py` (script ใหม่, ไม่มีการแก้ `app.py`/`prediction_log` เลย), `models/eval_xgb_posthoc_correction_select_20260720_142320.csv` (ผลทุก config บน `valid_tune` — ไม่มีไฟล์ `_confirm_*` เพราะไม่มี config ไหนผ่านเกณฑ์เลย จึงไม่ถึงขั้นยืนยันบน test)

---

## 2026-07-17 — Retrain XGBoost ด้วย sample weighting (แก้ bias มลพิษสูง) — **ผลลัพธ์ไม่ชัดเจน ยังไม่แนะนำให้ deploy**

**วัตถุประสงค์**: ใช้ recipe เดียวกับที่ retrain LightGBM ไปแล้ว (nested weight grid search + interpolate ข้าม 24 horizon) กับ XGBoost ซึ่งเดิมตั้งใจปล่อยให้เป็น unweighted baseline ไว้ก่อน (ดู memory `project-xgb-weighted-retrain-deferred`) ผู้ใช้ตัดสินใจให้ retrain แล้วตอนนี้

**วิธีทดสอบ**: `scripts/retrain_xgb_weighted.py` (สร้างใหม่)
- ข้อมูล: `data/khonkaen_pm25_features_raw_20260712_191249.csv` (ไฟล์เดียวกับที่โมเดล deploy จริงใช้เทรน)
- Split: chronological 80/20 (train: 2024-01-02 → 2026-01-23 06:00, test: 2026-01-23 07:00 → 2026-07-02) เหมือนตอนเทรนโมเดล deploy จริง แล้ว nested 85/15 ของ train (train_tune/valid_tune) สำหรับหา weight เท่านั้น ไม่แตะ test set จริง
- Weight grid search ที่ t+1, t+12, t+24: `WEIGHT_GRID=[1,3,5,10]`, เกณฑ์เลือก weight ที่ |bias_high| น้อยสุดโดย MAE_all ต้องไม่แย่ลงเกิน 8% — ผลลัพธ์: **t+1 เลือก weight=1.0 (ไม่ต้อง weight เลย), t+12 เลือก weight=3.0, t+24 เลือก weight=5.0** แล้ว interpolate เป็นทุก horizon
- Baseline เปรียบเทียบคำนวณสดในสคริปต์เดียวกัน (predict โมเดล deploy จริงบน test set เดียวกันเป๊ะ) ไม่ใช้ไฟล์ประวัติแยก

**ผลลัพธ์** (เฉลี่ยทั้ง 24 horizon, บน test set จริง):

| Metric | Baseline (unweighted, deploy จริง) | Weighted (โมเดลใหม่) | เปลี่ยนแปลง |
|---|---|---|---|
| MAE เฉลี่ยทุก horizon | 3.472 | 3.499 | **แย่ลง +0.8%** |
| MAE เฉลี่ยเฉพาะช่วงสูง (>37.5) | 4.906 | 4.805 | ดีขึ้น -2.1% |
| Bias เฉลี่ยช่วงสูง (+ = ทายต่ำกว่าจริง) | +3.040 | +2.950 | ดีขึ้นเล็กน้อย -3.0% (~0.09) |

**ข้อค้นพบสำคัญที่สุด**: ต่างจาก LightGBM ที่ recipe นี้ได้ผลชัดเจน (bias_high ลดลงมากตอน retrain รอบก่อน) กับ XGBoost ผลลัพธ์ **ไม่ชัดเจนและไม่สม่ำเสมอ** — เช็คทีละ horizon พบว่ามีอย่างน้อย 9 horizon (t+2, t+4, t+6, t+7, t+10, t+11, t+14, t+16, t+17, t+21) ที่ **ทั้ง MAE_all และ bias_high แย่ลงพร้อมกัน** หลัง weight (เช่น t+16: MAE_all 4.434→4.638 และ bias_high 3.903→4.354 แย่ลงทั้งคู่, t+7: MAE_all 2.127→2.212 และ bias_high 1.379→1.400 แย่ลงทั้งคู่) แปลว่า weight schedule ที่ interpolate จากแค่ 3 จุดตัวแทน (t+1/t+12/t+24) ไม่ generalize ข้าม horizon ให้ XGBoost ได้ดีเท่า LightGBM — โครงสร้าง error ของโมเดลทั้งสองไม่เหมือนกันพอที่จะใช้ weight schedule เดียวกันได้ผลเหมือนกัน

**สรุป**: **ยังไม่แนะนำให้ deploy โมเดลนี้** — ค่าเฉลี่ยรวมดีขึ้นเล็กน้อยก็จริง แต่ปรับปรุงน้อยกว่าที่คาดมาก (bias ลดลงแค่ ~3% เทียบกับที่ LightGBM ทำได้มากกว่านั้นเยอะ) แถมมีหลาย horizon ที่แย่ลงจริงทั้งสองมิติพร้อมกัน ไม่ใช่แค่ tradeoff ระหว่าง MAE กับ bias ตามที่ตั้งใจไว้ ถ้าจะไปต่อควรลอง weight grid ที่กว้างขึ้น หรือเลือก representative horizon ใหม่ที่สะท้อนพฤติกรรม XGBoost ได้ดีกว่านี้ ก่อนตัดสินใจ deploy จริง

**ไฟล์ที่เกี่ยวข้อง**: `scripts/retrain_xgb_weighted.py` (script ใหม่), `models/xgb_pm25_multistep_weighted_20260717_131755.pkl` (โมเดลใหม่ — **ยังไม่ deploy**), `models/eval_xgb_weighted_vs_baseline_20260717_131755.csv` (ผลละเอียดต่อ horizon)

### รอบ 2 (2026-07-17 ต่อเนื่อง): ลอง representative horizon ที่เลือกจาก SHAP divergence แทน — **สรุปว่าเป็น structural mismatch จริง หยุด ไม่ deploy**

**วัตถุประสงค์**: ผู้ใช้ตั้งสมมติฐานว่า t+1/t+12/t+24 ที่ copy มาจาก LightGBM อาจไม่ใช่จุดที่มีความหมายสำหรับ XGBoost โดยเฉพาะ ขอให้เลือก representative horizon ใหม่จากจุดที่ SHAP feature importance ของสองโมเดลต่างกันชัดเจนที่สุดแทน แล้วลองรอบเดียว — ถ้ายังเจอ pattern เดิม (แย่ลงพร้อมกันทั้ง MAE และ bias หลาย horizon) ให้สรุปว่าเป็น structural mismatch จริง ไม่ใช่แค่เลือก horizon ผิด แล้วหยุด

**วิธีทดสอบ**: `scripts/shap_divergence_by_horizon.py` (สร้างใหม่) — คำนวณ mean|SHAP| ต่อ feature ต่อโมเดลต่อ horizon ทั้ง 24 horizon (ไม่ใช่แค่ 3 จุดแบบ `shap_analysis.py`), normalize เป็นสัดส่วนรวม 1 แล้ววัด L1 distance ระหว่างสัดส่วนของ LightGBM กับ XGBoost — **ผลลัพธ์ขัดกับสมมติฐานเดิม**: ทั้งสองโมเดลยึด `pm25_lag_1` เป็น top feature ที่ **ทุก horizon** เหมือนกันหมด ไม่มี horizon ไหนที่ temp หรือ feature อื่นขึ้นมาเป็นอันดับ 1 แทน ความต่างที่วัดได้ (L1 dist 0.117–0.285, ช่วงค่อนข้างแคบ) มาจาก secondary features เท่านั้น 3 horizon ที่ต่างกันมากสุด: **t+1, t+21, t+24** (t+1/t+24 ซ้ำกับที่ LightGBM ใช้อยู่แล้ว มีแค่ t+21 ที่เปลี่ยนจาก t+12 เดิม)

รัน `scripts/retrain_xgb_weighted.py` ซ้ำอีกรอบด้วย `REP_HORIZONS=[1,21,24]`

**ผลลัพธ์** (เฉลี่ยทั้ง 24 horizon):

| Metric | Baseline | รอบ 1 (t+1/12/24) | รอบ 2 (t+1/21/24) |
|---|---|---|---|
| MAE เฉลี่ยทุก horizon | 3.472 | 3.499 (+0.8%) | 3.517 (+1.3%) |
| MAE เฉลี่ยเฉพาะช่วงสูง | 4.906 | 4.805 (-2.1%) | 4.738 (-3.4%) |
| Bias เฉลี่ยช่วงสูง | +3.040 | +2.950 (-3.0%) | +2.826 (-7.0%) |
| Horizon ที่แย่ลงพร้อมกันทั้ง MAE และ bias | 9/24 | — | **7/24** (t+3,4,5,6,7,10,21) |

**ข้อค้นพบสำคัญที่สุด**: การเลือก representative horizon ใหม่ตาม SHAP divergence ทำให้ bias เฉลี่ยดีขึ้นเกือบเท่าตัว (-7% เทียบ -3%) และจำนวน horizon ที่แย่ลงพร้อมกันลดลงจาก 9 เหลือ 7 — แต่ **ยังไม่หายไป** และที่สำคัญที่สุดคือ **t+21 เอง** (horizon ที่เลือกมาโดยเฉพาะเพราะ SHAP บอกว่าต่างจาก LightGBM มากสุด และ grid search เลือก weight=10.0 ให้เพราะดูดีตอน validate บน `valid_tune`) กลับ **แย่ลงทั้ง MAE และ bias บน test set จริง** (bias 4.898→5.373) — สัญญาณของ overfitting กับ tuning split ไม่ใช่การแก้ไขที่ generalize จริง ยืนยันว่าปัญหาไม่ได้อยู่ที่การเลือกจุด representative horizon ผิด แต่เป็นเพราะ**โครงสร้าง error ของ XGBoost ไม่เข้ากับวิธี interpolate weight จาก 3 จุดตัวแทนแบบที่ใช้ได้ผลกับ LightGBM**

**สรุป**: **หยุดความพยายาม retrain XGBoost แบบ weighted ไว้ตรงนี้ ไม่ deploy ทั้งสองรอบ** ตามเกณฑ์ที่ตกลงกันไว้ล่วงหน้า — กลับไปใช้ XGBoost แบบ unweighted เดิมที่ deploy อยู่ (ระบบ `EFFECTIVE_THRESHOLD_BY_HORIZON_BY_MODEL` ที่ปรับ threshold แจ้งเตือนตาม bias ที่วัดได้จริงต่อโมเดลอยู่แล้ว ยังคงชดเชยความคลาดเคลื่อนนี้ในทิศทางที่ปลอดภัยเหมือนเดิม) ถ้าจะแก้ bias ของ XGBoost จริงจังในอนาคต ควรพิจารณาวิธีอื่นที่ไม่ใช่ sample-weighting แบบเดียวกับ LightGBM (เช่น ปรับ hyperparameter ของ XGBoost เอง, หรือ post-hoc bias correction แยกต่างหากจากตัวโมเดล)

**ไฟล์ที่เกี่ยวข้อง**: `scripts/shap_divergence_by_horizon.py` (ใหม่), `models/shap_divergence_by_horizon.csv` (ใหม่), `models/xgb_pm25_multistep_weighted_20260717_132430.pkl` (โมเดลรอบ 2 — **ไม่ deploy เช่นกัน**), `models/eval_xgb_weighted_vs_baseline_20260717_132430.csv`

---

## 2026-07-16 — Backtest โมเดลที่ deploy อยู่จริง เฉพาะช่วงฤดูฝุ่น (ธ.ค.-มี.ค.)

**วัตถุประสงค์**: ตอบคำถามว่าโมเดลปัจจุบัน (deployed) รับมือกับฤดูฝุ่นได้ดีแค่ไหน โดยไม่ต้องรอถึงฤดูฝุ่นจริง (ปัจจุบัน ก.ค. 2026) — ใช้ backtest กับข้อมูลย้อนหลังจริงแทน

**วิธีทดสอบ**: `scripts/eval_haze_season_backtest.py` (สร้างใหม่ ไม่กระทบไฟล์เดิม)
- ข้อมูล: `data/khonkaen_pm25_features_raw_20260712_191249.csv`
- Split: chronological 80/20 เดียวกับตอนเทรน (train: 2024-01-02 → 2026-01-23 07:00, test: 2026-01-23 07:00 → 2026-07-02) แล้วกรองเฉพาะแถวเดือน ธ.ค./ม.ค./ก.พ./มี.ค. ที่อยู่ใน **test split เท่านั้น** (ไม่ปนกับ train — เป็น out-of-sample จริง ไม่ใช่ data leakage)
- ช่วงที่ได้: 2026-01-23 07:00 → 2026-03-31 23:00 (1,625 แถว, ครอบคลุมมกราคม-มีนาคมของฤดูฝุ่นนี้ ขาดแค่ธันวาคมที่ตกอยู่ใน train)
- โมเดลที่ทดสอบ: LightGBM weighted (`lgb_pm25_multistep_weighted_20260712_191056.pkl`) และ XGBoost (`xgb_pm25_multistep_20260712_191350.pkl`) — ตัวที่ deploy จริงใน `app.py` ทั้งคู่
- คำนวณ MAE/bias ทั้งแบบรวมทุกช่วง และเฉพาะช่วงมลพิษสูง (actual > 37.5) แยกตาม horizon (1-24)

**ผลลัพธ์**:

| Metric | LightGBM (deployed) | XGBoost (deployed) |
|---|---|---|
| MAE เฉลี่ยทุก horizon | 3.561 | 3.765 |
| MAE เฉลี่ยเฉพาะช่วงสูง (>37.5) | 3.453 | 4.230 |
| Bias เฉลี่ยช่วงสูง (+ = ทายต่ำกว่าจริง) | +1.650 | +2.473 |

- ~43% ของแถวในช่วงฤดูฝุ่นมี actual PM2.5 > 37.5 (ยืนยันว่าเป็นช่วงมลพิษสูงจริงตามที่คาด ไม่ใช่แค่สมมติฐาน)
- MAE สูงกว่าค่าเฉลี่ย live ปัจจุบันที่ไม่แยกฤดู (±3.07 จาก `/api/comparison/overview` ตอนตรวจสอบวันเดียวกัน) เล็กน้อยตามคาด — ฤดูฝุ่นยากกว่าสภาพอากาศทั่วไปจริง แต่ไม่ได้ต่างกันมากจนน่าตกใจ
- LightGBM ทำได้ดีกว่า XGBoost ทั้ง MAE และ bias ช่วงสูง — ตรงกับที่ Dynamic Routing เลือก LightGBM บ่อยกว่าจาก live rolling R² อยู่แล้ว

**ข้อค้นพบสำคัญที่สุด**: เทียบ `bias_high_weighted` ที่ใช้ calibrate `EFFECTIVE_THRESHOLD_BY_HORIZON` จริงใน `app.py` (คำนวณจาก full test set ทุกเดือน ไม่แยกฤดู) กับ `bias_high` เฉพาะฤดูฝุ่นที่วัดได้รอบนี้ (เฉพาะ LightGBM เพราะเป็นโมเดลหลักที่ threshold ผูกไว้) พบว่า **bias เฉพาะฤดูฝุ่นน้อยกว่าที่ threshold ถูก calibrate ไว้ทุก horizon** (เฉลี่ยต่างกัน -1.05, ช่วง -0.19 ถึง -1.84 ยิ่ง horizon ไกลยิ่งต่างมาก) หมายความว่า threshold ที่ใช้จริงตอนนี้ถูกปรับให้ระมัดระวังเกินความจำเป็นสำหรับฤดูฝุ่นโดยเฉพาะ — คือ **ปลอดภัยเกินความจำเป็น ไม่ใช่ปลอดภัยน้อยเกินไป** ซึ่งเป็นทิศทางที่ถูกต้องสำหรับระบบแจ้งเตือนสุขภาพ (ดีกว่าเกินไปดีกว่าขาดไป)

**สรุป**: โมเดลรับมือฤดูฝุ่นได้ในระดับที่ยอมรับได้ ไม่มีสัญญาณว่าจะพังหรือแย่กว่าที่คาดไว้มาก และกลไก threshold correction ที่มีอยู่ยังปลอดภัยเผื่อไว้มากกว่าที่จำเป็นสำหรับสถานการณ์นี้โดยเฉพาะ

**ไฟล์ที่เกี่ยวข้อง**: `scripts/eval_haze_season_backtest.py` (script ใหม่), `models/eval_haze_season_backtest.csv` (ผลละเอียดต่อ horizon ต่อโมเดล)

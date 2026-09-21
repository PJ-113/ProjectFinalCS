# ไฟล์โมเดลที่ไม่ได้อยู่ใน git

โมเดลที่ `app.py` ใช้อยู่ (v3, `_20260919_203455`) มีขนาดเกิน 100 MB ต่อไฟล์ ซึ่งเป็นขีดจำกัดของ GitHub จึงไม่ได้ commit
ไฟล์ conformal และตารางสรุปผลของโมเดลนี้ (ขนาดเล็ก) อยู่ใน git แล้ว (`models/conformal_calibration_{lgb,xgb}_20260919_203455.csv`,
`models/multiprovince_eval_summary_20260919_203455.csv`)

## ไฟล์ที่ต้องมีจึงจะรัน `python app.py` ได้

วางไว้ในโฟลเดอร์ `models/` (ชื่อไฟล์ต้องตรงตามนี้ เพราะ `app.py` อ้างชื่อเหล่านี้)

| ไฟล์ | ขนาด (ไบต์) | SHA-256 |
|---|---|---|
| `lgb_pm25_multiprovince_residual_bagged_quantile_20260919_203455.pkl` | 167,347,473 | `6596666cd72f450613b6d2a15896700dc936c222d3e9253742e4edff897ad898` |
| `xgb_pm25_multiprovince_residual_bagged_quantile_20260919_203455.pkl` | 598,681,006 | `37ecdb91f0bb532873b4baed38e56bb3dcc96ff800225a9cfe65e16a520cb6d8` |

ตรวจว่าไฟล์ที่ดาวน์โหลด/กู้คืนมาไม่เสียหาย (ค่า SHA-256 ต้องตรงกับตาราง):

```
python -c "import hashlib,sys; h=hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest(); print(h)" models/<ชื่อไฟล์>.pkl
```

## ที่เก็บสำรอง

- GitHub Release: (ยังไม่ได้อัปโหลด — ใส่ลิงก์ที่นี่หลังอัปโหลด)
- สำรองอีกที่ (Drive/ฮาร์ดดิสก์): (ใส่ที่เก็บที่นี่)

## ถ้าไฟล์หาย

เทรนใหม่ได้จากสคริปต์ แต่ผลอาจไม่เหมือนเดิมทุกตัวเลข เพราะ OpenAQ อาจแก้ข้อมูลย้อนหลัง
1. `python scripts/rebuild_dataset_clean.py` (ดึงข้อมูลและสร้างชุดข้อมูลสะอาด พร้อม offset ป้ายเวลาจาก registry)
2. `python scripts/train_multiprovince_models.py` (20 feature, hyperparameter ที่จูนแล้วอยู่ในสคริปต์)
3. ปรับ path/ชื่อโมเดลใน `app.py` ให้ตรงกับไฟล์ใหม่ (`LGB_MODEL_PATH`, `XGB_MODEL_PATH`, `CONFORMAL_*`, `MODEL_LGB`, `MODEL_XGB`)

ไฟล์อื่นที่ไม่ได้อยู่ใน git และควรสำรองไว้ด้วยเพื่อความสามารถทำซ้ำ: ชุดข้อมูลเทรน
`data/multiprovince_pm25_features_raw_spatial_k1_firehotspot_nbtrend_20260919_174805.csv` และข้อมูลจุดความร้อน
`data/firms_hotspots_isaan_*.csv`

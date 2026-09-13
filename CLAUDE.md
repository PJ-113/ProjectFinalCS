# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Flask backend (`app.py`) that forecasts PM2.5 24 hours ahead for 20 Isaan provinces (17 trained + 3 zero-shot: นครพนม/ร้อยเอ็ด/มหาสารคาม — see CHANGELOG (75)) using a **pooled** LightGBM + XGBoost model pair (single shared model per algorithm, not per-province — `lat`/`lon` are features, which is what makes zero-shot serving possible) behind a Dynamic Routing layer, logs every prediction to a PostgreSQL DB (database `pm25_forecast`, tables `prediction_log`/`daily_nlg_log`/`notification_log` — migrated from SQLite on 2026-07-16, see CHANGELOG (30); all three gained a `province` column on 2026-09-13, see CHANGELOG (78)) for live evaluation against ground truth, and serves a Vue 3 SPA dashboard (`frontend/`, built with Vite, PWA-enabled via `vite-plugin-pwa` — migrated from static `index.html`/`compare.html` on 2026-07-16 per the proposal's stated PWA framework requirement, see CHANGELOG (32); the pre-Vue vanilla-JS pages are kept for reference in `pre_vue_frontend_backup/`). `scripts/` holds one-off data-prep, training, and diagnostic scripts — there is no shared library; each script is a self-contained entry point. Paths are computed relative to the repo root via `_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))` (made path-independent 2026-07-22, see CHANGELOG (74)) so the project folder can be renamed/moved freely — do NOT reintroduce hardcoded absolute paths like `c:\Project1\Model1\...` in new scripts; use `os.path.join(_ROOT, "data", ...)` etc. `app.py` itself is already path-independent (relative `models/...` + `os.path.dirname(os.path.abspath(__file__))` for `.env`/`frontend/dist`).

**Historical note**: before 2026-09-13 this app served only Khon Kaen, via 3 separate models (LightGBM/XGBoost/Prophet) trained on that one station, with live inference pulling from Air4Thai and a calibration layer to map onto the OpenAQ scale the models were trained on. That entire architecture (Prophet, Air4Thai fetch, calibration) was retired in CHANGELOG (78) in favor of the pooled multi-province model described above, which trains and serves on OpenAQ exclusively (no calibration needed since both sides are already on the same scale). The old single-station model files remain in `models/` per the never-delete convention but are no longer referenced by `app.py`.

This is a university CS capstone project (see the proposal's Dynamic Routing spec, referenced in code comments as "proposal 4.1.6").

## Commands

- Run the app: `python app.py` — serves on port 5000. `DEBUG = True` in `app.py` enables Flask's auto-reloader *and* gates the APScheduler hourly job via `WERKZEUG_RUN_MAIN` (to avoid starting the scheduler twice under the reloader). Editing `app.py` while the app is running triggers a reload, which restarts the scheduler and its "fire immediately on start" behavior.
- **Frontend build step (new since the Vue migration)**: `app.py`'s catch-all route serves `frontend/dist/`, which only exists after running `cd frontend && npm install && npm run build`. Editing files under `frontend/src/` has **no effect** until you rebuild — unlike the old `.html`/`static/*.js` files, which took effect on a plain browser refresh. For active frontend development, `cd frontend && npm run dev` runs the Vite dev server with hot-reload and a `/api` proxy to `localhost:5000` (configured in `frontend/vite.config.js`) — run this alongside `python app.py` rather than rebuilding on every change.
- Python dependencies are pinned in `requirements.txt` (plain `pip freeze` output, install with `pip install -r requirements.txt`). Third-party imports actually used across the codebase: `flask`, `flask-cors`, `apscheduler`, `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `xgboost`, `prophet`, `requests`, `urllib3`, `matplotlib`, `psycopg2`, plus `google-genai` for NLG generation. The frontend (`frontend/`) is a separate Node.js project with its own `package.json`/`node_modules` (Vue 3, Vue Router, Vite, `vite-plugin-pwa`) — unrelated to Python dependency management.
- Run any script directly: `python scripts/<name>.py`. Scripts print their own results; there's no CLI arg parsing convention.
- No test framework (no pytest). Tests are plain scripts with manual `assert`/`print`, e.g. `python scripts/test_dynamic_routing.py`, which loads `app.py` via `importlib.util`. Since the move to Postgres there's no `DB_PATH` to monkeypatch to a temp file anymore — tests instead write/read real rows in the real tables using an obviously-fake year as a marker (`predicted_at_utc`/`target_time_utc` prefixed `2100-` in `test_dynamic_routing.py`; `date_thai` prefixed `2199-` in `test_daily_nlg.py`, chosen because it still has to be mockable as a real `pd.Timestamp`, whose max year is ~2262) and delete those marker rows before and after the run (in a `finally` block) so a failed assertion never leaves test data behind. Follow this pattern for new tests of `app.py` DB logic — pick a marker year, never touch rows outside it.

## Critical architecture facts

### No calibration layer anymore — live inference and training are both OpenAQ

Unlike the old Khon Kaen-only architecture, live inference (`fetch_recent_pm25_openaq()`) pulls PM2.5 from **OpenAQ v3** (`/v3/sensors/{sensor_id}/measurements`, sensor ID per province from `data/province_station_registry.json`) — the exact same source and scale the pooled models were trained on (see `scripts/build_raw_dataset_multiprovince.py`). There is no Air4Thai fetch, no `calibrate_pm25()`, and no `CALIBRATION_A`/`CALIBRATION_B` anywhere in `app.py` anymore. `latest_pm25` (the "current" value shown to users and compared against `HIGH_POLLUTION_THRESHOLD`) is just the raw latest OpenAQ reading — no transform needed since it's already on the right scale.

### `prediction_log` (PostgreSQL, db `pm25_forecast`): `actual_pm25` vs `actual_pm25_calibrated`

The `prediction_log` table still has both columns (kept for schema/query compatibility with `/api/comparison*`), but since there's no calibration anymore, `actual_pm25_calibrated` is simply set equal to `actual_pm25` for every row `backfill_actuals()` writes going forward — it's a plain COALESCE-safe duplicate, not a real transform. (Historical note: this table lived in a SQLite file `predictions.db` until 2026-07-16, when it was migrated to PostgreSQL — see CHANGELOG (30); gained a `province` column on 2026-09-13, see CHANGELOG (78). Pre-2026-09-13 rows — all Khon Kaen, from the old Air4Thai-calibrated architecture — keep their real `a·x+b`-derived `actual_pm25_calibrated` values as history; only new rows get identity treatment.)

`prediction_log` also has `pm25_lo`/`pm25_hi` (nullable, added CHANGELOG (82)) — the conformal-calibrated P05/P95 bounds `get_forecast()` already computes per horizon, now persisted instead of discarded after the response is sent. Rows logged before (82) have both columns NULL forever (no way to reconstruct a historical quantile prediction that was never stored) — `/api/comparison` and `/api/comparison/overview` filter those out of `coverage_pct` (the % of matched pairs where `actual` fell inside `[pm25_lo, pm25_hi]`, target ~90% per `ALPHA=0.10`) and always exclude `naive_fallback` rows from it too (their interval is zero-width by construction, not a real uncertainty estimate — see the Dynamic Routing section below).

### Dynamic Routing (`select_model()` in `app.py`) — now 2-way, per province

Two-tier, filtered by `province` throughout: (1) if `compute_missing_ratio(feature_row) > SIGMA_CRITICAL` → `"naive_fallback"` (circuit breaker; the pooled models were trained with `dropna()` on any row with missing lag/rolling features, so they've never seen NaN input — feeding them a gappy row would be out-of-distribution, not graceful degradation. Fallback is flat persistence: forecast every horizon as `latest_pm25`). (2) Otherwise compare `compute_rolling_r2()` (last 24 matched prediction/actual pairs **for that province**, needs ≥5 to be trusted) between `"xgboost_pooled"` and `"lightgbm_pooled"`. Both `r2_xgb is None` and `r2_lgb is None` are independent cold-start cases and must each get ε-greedy exploration (`EXPLORATION_RATE`) — handling only one side silently starves the other model of live data forever; this was a real bug, fixed in CHANGELOG (20), and the province filter added in CHANGELOG (78) is there for the same reason (without it, different provinces' live performance blends into one routing decision).

**Model naming matters**: live model names are `"lightgbm_pooled"`/`"xgboost_pooled"`/`"naive_fallback"` — deliberately distinct from the old single-station names (`"lightgbm"`/`"xgboost"`/`"prophet"`, still present on pre-2026-09-13 rows) so `compute_rolling_r2()` never blends pre-cutover Khon Kaen-only performance with the new pooled model's live performance.

### Alert threshold: conformal P95 upper bound, not a bias-correction CSV

The old system compensated for weighted-model under-prediction in the high-pollution range via `EFFECTIVE_THRESHOLD_BY_HORIZON_BY_MODEL[model][h] = 37.5 - bias_high_weighted[h]`, loaded from bias-evaluation CSVs specific to the old per-station models. The pooled models don't have an equivalent CSV. Instead, `is_horizon_at_risk(p95_upper_adj, risk_group)` compares the **conformal-calibrated P95 upper bound** (`p95_pred + CONFORMAL_QHAT[model_name][h]`, computed fresh in `get_forecast()` from the quantile-regression P95 head + the conformal `q_hat` in `models/conformal_calibration_{lgb,xgb}_*.csv`) against `HIGH_POLLUTION_THRESHOLD + risk_group_offset`. This is conceptually the same goal (catch cases where a point prediction would under-state real risk) achieved through the quantile/conformal machinery the pooled models already have, rather than a separate bias-evaluation pipeline. `HIGH_POLLUTION_THRESHOLD = 37.5` itself is a single national constant (PCD standard), not per-province. `naive_fallback` rows have no quantile heads, so their `p95_upper_adj` is just the point prediction (no margin) — Never revert to comparing predictions against the raw threshold uniformly with no uncertainty margin at all — that was the original bug the bias-CSV mechanism (and now the conformal-P95 mechanism) fixes.

### Direct multi-horizon forecasting + residual targets, not recursive

LightGBM/XGBoost each train 24×3 independent models — one per `(pm25_t+{h}, quantile)` combination (h=1..24, quantile∈{0.05,0.5,0.95}) — not a single model applied recursively, and not a single point-estimate per horizon. Each model predicts a **residual** (`pm25_t+h - pm25_now`), not the absolute value — `get_forecast()` always adds the row's `latest_pm25` back after `.predict()`. See CHANGELOG (75) for the full training methodology (bagging via per-tree subsample/colsample, conformal calibration for the P05/P95 interval).

### Feature construction is duplicated across 4 places — keep them in sync

The 23-feature formulas (`temp/humidity/wind_speed/precipitation`, cyclical `hour/month` + `dayofweek`, `pm25_lag_1/3/6/12/24`, `pm25_roll_mean/std_6/24`, `wind_dir_sin/cos`, `pressure`, `lat`, `lon`) are independently implemented in: `app.py:build_feature_row()` (live inference — time-indexed lookback, not positional, since OpenAQ data can have gaps unlike the old Air4Thai feed), `scripts/build_raw_dataset_multiprovince.py` (vectorized batch feature engineering for training), `scripts/backtest_recent_multiprovince.py:build_station_features()` (post-training-window backtest), and `scripts/train_multiprovince_models.py`/`scripts/eval_zeroshot_multiprovince.py` (consume the same `FEATURE_COLS` list, don't recompute features themselves). A change to the feature definition must be mirrored across all of these. Existing scripts self-check with `np.allclose` against precomputed CSV columns — follow that pattern when adding new features.

### Gap handling in training data (`scripts/build_raw_dataset.py`)

Raw OpenAQ pm25 has real gaps. Gaps ≤48h are linearly interpolated (kept for training); gaps >48h are masked out entirely — both the gap itself and any lag/rolling/target value that "touches" a long gap (via a rolling window over an `is_long_gap` flag) are set to NaN rather than silently interpolated across. The original `data/khonkaen_pm25_features_final.csv` does *not* do this — it interpolates everything, which was found to suppress real PM2.5 volatility and made backtest performance look much better than live performance. New training uses `data/khonkaen_pm25_features_raw_*.csv` instead; per-horizon code must `dropna(subset=[target])` since different horizons have different valid-row counts near gaps.

### Train/test split convention

Always chronological (time-based) 80/20 — never shuffled, to avoid leaking future rows into training. LightGBM's sample-weight grid search (for high-pollution bias correction around `HIGH_THRESHOLD = 37.5`) uses a nested 85/15 split of the train portion (`train_tune`/`valid_tune`) so weight selection never touches the real test set; the chosen weight at 3 representative horizons (t+1, t+12, t+24) is linearly interpolated (`np.interp`) across all 24 horizons.

### Air4Thai's ~2-month retention (historical — no longer relevant to live serving)

`scripts/archive_air4thai.py` (snapshotting Air4Thai data into `data/air4thai_archive.csv` before its ~2-month retention window ages it out) was built for the old Khon Kaen-only Air4Thai live-fetch path, which no longer exists — live inference is OpenAQ-only now (see above). The script/archive file are kept for history, not actively used.

### Multi-province registry (`data/province_station_registry.json`)

Canonical list of the 20 Isaan provinces (17 trained + 3 zero-shot), built by `scripts/province_registry.py` by querying the OpenAQ v3 API directly (never hardcode location/sensor IDs — some OpenAQ sensor IDs for the same physical station return `500 Internal Server Error` while an older sensor ID at the same location works fine; the registry script already handles this by preferring the lowest sensor ID per location, see CHANGELOG (75)). `app.py` loads this once at startup into `PROVINCE_REGISTRY`/`PROVINCES_BY_SLUG`; `province_en` (e.g. `"khonkaen"`, `"nakhonphanom"`) is the slug used everywhere (API query params, `model_used`-adjacent DB rows, frontend `provinceState.selected`).

## API routes (`app.py`)

| Route | Purpose |
|---|---|
| `GET /`, `GET /<any-client-route>` | serves the Vue SPA shell (`frontend/dist/index.html`) via `serve_spa()`, or the matching static asset if `path` is a real file under `frontend/dist` (JS/CSS bundles, `manifest.webmanifest`, `sw.js`, icons). Vue Router then renders `/` (forecast) or `/compare` (comparison) client-side — there is no server-side distinction between these two anymore |
| `GET /index.html`, `GET /compare.html` | 302 redirect to `/` and `/compare` respectively, for old bookmarks (the routes themselves stopped being real pages when the Vue migration dropped the `.html` URLs) |
| `GET /api/forecast?province=` | runs `get_forecast()`: fetches latest data for that province, predicts all 24 horizons via Dynamic Routing, logs to `prediction_log` (Postgres), backfills past actuals. `province` defaults to `khonkaen` if omitted; 400 if given but not a known `province_en` slug |
| `GET /api/comparison?horizon=&model=&province=` | per-horizon prediction/actual series + MAE/bias/conformal-interval coverage, scoped to one province |
| `GET /api/comparison/overview?province=` | aggregate MAE/bias/coverage by horizon and by model, scoped to one province |
| `GET /api/provinces` | the province registry (`province_th`/`province_en`/`lat`/`lon`/`is_zero_shot`) — drives the frontend's province `<select>` |

`scheduled_forecast_job()` (hourly APScheduler job) loops `get_forecast()` over every province in `PROVINCE_REGISTRY`, not just one — with a per-province `try`/`except` so one province's OpenAQ/Open-Meteo hiccup doesn't kill the job for the other 19.

Note: `serve_spa`'s `/<path:path>` catch-all is a dynamic rule, but Werkzeug always gives routing priority to static rules (`/api/forecast`, etc.) over rules with converters regardless of registration order, so the catch-all can never shadow an API route — confirmed by testing, not just assumed.

## File conventions

- Model/data outputs are always saved under a new timestamped filename (e.g. `models/lgb_pm25_multiprovince_residual_bagged_quantile_{timestamp}.pkl`) — never overwrite or delete a previous version. `app.py`'s `LGB_MODEL_PATH` / `XGB_MODEL_PATH` / `CONFORMAL_LGB_PATH` / `CONFORMAL_XGB_PATH` / `PROVINCE_REGISTRY_PATH` constants point at whichever files are currently "deployed"; older files (including the entire old single-station `MODEL_PATH`/`PROPHET_MODEL_PATH`/`BIAS_CSV_PATH`/`CALIBRATION_PATH` lineage, retired in CHANGELOG (78)) remain as history/fallback, unreferenced.
- Every code change gets a new dated, numbered entry prepended to `CHANGELOG.md` (format: `## YYYY-MM-DD (N) — title`, with ที่มา (source/why) / what-changed / test-results / files-changed subsections, written in Thai). Check the latest entries there before assuming why something is implemented a certain way — it's the authoritative history of *why*, not just *what*.
- Before any destructive reset of `prediction_log` (e.g. dropping it after a model retrain), take a `pg_dump` snapshot first (the old convention was `predictions.db.bak_{timestamp}` copies of the SQLite file — the last one, `predictions.db.bak_20260716_000516`, is the final pre-migration snapshot and is kept, not deleted).
- `.env` holds `OPENAQ_KEY` (OpenAQ v3 API key) and `GEMINI_KEY` (Gemini API key), plus `POSTGRES_HOST`/`POSTGRES_PORT`/`POSTGRES_DB`/`POSTGRES_USER`/`POSTGRES_PASSWORD` for the `prediction_log`/`daily_nlg_log` database, all read via the same manual line-parsing helper (`_load_env()` in `app.py`, no `python-dotenv`). `OPENAQ_KEY` is sent as the `X-API-Key` header. Never print, log, or echo any of these values.

## Repo layout for docs & notebooks

Non-code artifacts are kept out of the project root (reorganized 2026-07-22, see CHANGELOG (73)):
- `docs/` — design docs, diagrams, and the proposal: `database_schema.md` (ERD), `usecase_diagram.puml`, `system_architecture.puml`/`.html`/`.png` (the `.html`/`.png` are the icon-based visual version, the `.puml` files are PlantUML sources; `*_plantuml.png`/`usecase_diagram.png` are their rendered images), `proposal.docx`, `proposal_14.pdf`.
- `notebooks/` — historical/exploratory notebooks (below).

The root now holds only code + top-level project files: `app.py`, `.env`, `requirements.txt`, `CLAUDE.md`, `CHANGELOG.md`, `TEST_RESULTS.md`, plus the working dirs (`models/ data/ scripts/ frontend/ backups/ figures/`).

## Notebooks (historical/exploratory, not actively maintained)

- `notebooks/train_models.ipynb`: the original cell-by-cell experiment where LightGBM/XGBoost/Prophet were first trained and compared — superseded by `scripts/train_*.py` for actual retraining.
- `notebooks/Testforcest.ipynb`: the original prototype of the live-inference fetch + feature-row logic that became `app.py`'s `fetch_recent_pm25()`/`build_feature_row()`.

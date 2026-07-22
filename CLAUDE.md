# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Flask backend (`app.py`) that forecasts PM2.5 24 hours ahead for Khon Kaen using three models (LightGBM, XGBoost, Prophet) behind a Dynamic Routing layer, logs every prediction to a PostgreSQL DB (database `pm25_forecast`, tables `prediction_log`/`daily_nlg_log` — migrated from SQLite on 2026-07-16, see CHANGELOG (30)) for live evaluation against ground truth, and serves a Vue 3 SPA dashboard (`frontend/`, built with Vite, PWA-enabled via `vite-plugin-pwa` — migrated from static `index.html`/`compare.html` on 2026-07-16 per the proposal's stated PWA framework requirement, see CHANGELOG (32); the pre-Vue vanilla-JS pages are kept for reference in `pre_vue_frontend_backup/`). `scripts/` holds one-off data-prep, training, and diagnostic scripts — there is no shared library; each script is a self-contained entry point. Paths are computed relative to the repo root via `_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))` (made path-independent 2026-07-22, see CHANGELOG (74)) so the project folder can be renamed/moved freely — do NOT reintroduce hardcoded absolute paths like `c:\Project1\Model1\...` in new scripts; use `os.path.join(_ROOT, "data", ...)` etc. `app.py` itself is already path-independent (relative `models/...` + `os.path.dirname(os.path.abspath(__file__))` for `.env`/`frontend/dist`).

This is a university CS capstone project (see the proposal's Dynamic Routing spec, referenced in code comments as "proposal 4.1.6").

## Commands

- Run the app: `python app.py` — serves on port 5000. `DEBUG = True` in `app.py` enables Flask's auto-reloader *and* gates the APScheduler hourly job via `WERKZEUG_RUN_MAIN` (to avoid starting the scheduler twice under the reloader). Editing `app.py` while the app is running triggers a reload, which restarts the scheduler and its "fire immediately on start" behavior.
- **Frontend build step (new since the Vue migration)**: `app.py`'s catch-all route serves `frontend/dist/`, which only exists after running `cd frontend && npm install && npm run build`. Editing files under `frontend/src/` has **no effect** until you rebuild — unlike the old `.html`/`static/*.js` files, which took effect on a plain browser refresh. For active frontend development, `cd frontend && npm run dev` runs the Vite dev server with hot-reload and a `/api` proxy to `localhost:5000` (configured in `frontend/vite.config.js`) — run this alongside `python app.py` rather than rebuilding on every change.
- Python dependencies are pinned in `requirements.txt` (plain `pip freeze` output, install with `pip install -r requirements.txt`). Third-party imports actually used across the codebase: `flask`, `flask-cors`, `apscheduler`, `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `xgboost`, `prophet`, `requests`, `urllib3`, `matplotlib`, `psycopg2`, plus `google-genai` for NLG generation. The frontend (`frontend/`) is a separate Node.js project with its own `package.json`/`node_modules` (Vue 3, Vue Router, Vite, `vite-plugin-pwa`) — unrelated to Python dependency management.
- Run any script directly: `python scripts/<name>.py`. Scripts print their own results; there's no CLI arg parsing convention.
- No test framework (no pytest). Tests are plain scripts with manual `assert`/`print`, e.g. `python scripts/test_dynamic_routing.py`, which loads `app.py` via `importlib.util`. Since the move to Postgres there's no `DB_PATH` to monkeypatch to a temp file anymore — tests instead write/read real rows in the real tables using an obviously-fake year as a marker (`predicted_at_utc`/`target_time_utc` prefixed `2100-` in `test_dynamic_routing.py`; `date_thai` prefixed `2199-` in `test_daily_nlg.py`, chosen because it still has to be mockable as a real `pd.Timestamp`, whose max year is ~2262) and delete those marker rows before and after the run (in a `finally` block) so a failed assertion never leaves test data behind. Follow this pattern for new tests of `app.py` DB logic — pick a marker year, never touch rows outside it.

## Critical architecture facts

### Data source mismatch + calibration layer (read before touching pm25 features)

Models are trained on **OpenAQ** historical data (station 7747 / sensor 1304249), but live inference (`fetch_recent_pm25()`) pulls from **Air4Thai** (station 46t). These are confirmed to be the same physical station (identical coordinates) but report substantially different PM2.5 values for the same hour (R² ≈ 0.4 between the two series). `calibrate_pm25()` in `app.py` maps raw Air4Thai readings onto the OpenAQ scale (`CALIBRATION_A`/`CALIBRATION_B`, fit by `scripts/fit_air4thai_openaq_calibration.py`, persisted in `models/air4thai_openaq_calibration.pkl`) before *any* lag/rolling feature is computed in `build_feature_row()`. The one exception: `latest_pm25` (the "current" value shown to users and compared against `HIGH_POLLUTION_THRESHOLD`) stays raw/uncalibrated, since it's a real station reading, not a model input.

### `prediction_log` (PostgreSQL, db `pm25_forecast`): `actual_pm25` vs `actual_pm25_calibrated`

The `prediction_log` table has both columns. (Historical note: this table lived in a SQLite file `predictions.db` until 2026-07-16, when it was migrated to PostgreSQL — see CHANGELOG (30). The old SQLite file is kept as a dated backup, not deleted, per the file-conventions backup rule below; app.py now connects via `get_db_connection()`/psycopg2, reading `POSTGRES_HOST/PORT/DB/USER/PASSWORD` from `.env`.) `actual_pm25` is always the raw reading as originally sourced; `actual_pm25_calibrated` is on the same scale as `predicted_pm25` and is what all evaluation code (`compute_rolling_r2()`, `/api/comparison*`) must compare against. Which transform applies depends on `source`:
- `source='live'`: actual comes from `fetch_recent_pm25()` (raw Air4Thai) → calibrated = `calibrate_pm25(actual)`.
- `source='backfill'`: actual comes from the training CSV, already on the OpenAQ scale (written by `scripts/backfill_xgboost*.py`) → calibrated = actual, unchanged. Applying the a·x+b transform here is a double-transform bug that has already happened once (see CHANGELOG (18)) — any new code touching this column needs to branch on `source` the same way `migrate_calibrated_actuals()` and the `/api/comparison*` SQL do.

### Dynamic Routing (`select_model()` in `app.py`)

Two-tier: (1) if `compute_missing_ratio(feature_row) > SIGMA_CRITICAL` → force Prophet (circuit breaker; Prophet doesn't depend on lag/rolling inputs so it tolerates missing data). (2) Otherwise compare `compute_rolling_r2()` (last 24 matched prediction/actual pairs, needs ≥5 to be trusted) between XGBoost and LightGBM. Both `r2_xgb is None` and `r2_lgb is None` are independent cold-start cases and must each get ε-greedy exploration (`EXPLORATION_RATE`) — handling only one side (e.g. only exploring when XGBoost lacks data) silently starves the other model of live data forever; this was a real bug, fixed in CHANGELOG (20).

### Alert threshold compensates for residual model bias

`HIGH_POLLUTION_THRESHOLD = 37.5` is the nominal alert cutoff, but weighted-model predictions still under-predict in the high-pollution range by a residual amount that grows with horizon. `EFFECTIVE_THRESHOLD_BY_HORIZON[h] = 37.5 - bias_high_weighted[h]` (loaded from a CSV produced during retraining) is what `risk_horizons`/`high_pollution_risk` actually compare each horizon's prediction against — not the raw 37.5. `latest_pm25` (real observed value, no model bias) is the one exception and still compares directly against 37.5. Never revert to comparing predictions against the raw threshold uniformly across horizons — this was the original bug this mechanism fixes.

### Direct multi-horizon forecasting, not recursive

LightGBM/XGBoost train 24 independent models, one per `pm25_t+{h}` target (h=1..24) — not a single model applied recursively. Prophet is a single model but is evaluated 1-step (predicting each row's own `ds` directly), not a true 24-step recursive forecast; this is an accepted simplification, not a bug, and Prophet's live prediction (`predict_with_prophet()`) reuses *current* weather for every horizon since there's no wired-in hourly weather forecast.

### Feature construction is duplicated across 3 places — keep them in sync

The lag/rolling formulas (`pm25_lag_1/3/6/12/24`, `pm25_roll_mean/std_6/24`, cyclical hour/month encoding, `dayofweek`) are independently implemented in: `app.py:build_feature_row()` (live inference), `scripts/backfill_xgboost*.py:build_feature_row_at()` (historical backtest simulation), and `scripts/build_raw_dataset.py` (vectorized batch feature engineering for training). A change to the feature definition must be mirrored in all three. Existing scripts self-check with `np.allclose` against precomputed CSV columns — follow that pattern when adding new features.

### Gap handling in training data (`scripts/build_raw_dataset.py`)

Raw OpenAQ pm25 has real gaps. Gaps ≤48h are linearly interpolated (kept for training); gaps >48h are masked out entirely — both the gap itself and any lag/rolling/target value that "touches" a long gap (via a rolling window over an `is_long_gap` flag) are set to NaN rather than silently interpolated across. The original `data/khonkaen_pm25_features_final.csv` does *not* do this — it interpolates everything, which was found to suppress real PM2.5 volatility and made backtest performance look much better than live performance. New training uses `data/khonkaen_pm25_features_raw_*.csv` instead; per-horizon code must `dropna(subset=[target])` since different horizons have different valid-row counts near gaps.

### Train/test split convention

Always chronological (time-based) 80/20 — never shuffled, to avoid leaking future rows into training. LightGBM's sample-weight grid search (for high-pollution bias correction around `HIGH_THRESHOLD = 37.5`) uses a nested 85/15 split of the train portion (`train_tune`/`valid_tune`) so weight selection never touches the real test set; the chosen weight at 3 representative horizons (t+1, t+12, t+24) is linearly interpolated (`np.interp`) across all 24 horizons.

### Air4Thai's ~2-month retention

`getHistoryData.php` (used by both `fetch_recent_pm25()` and calibration fitting) has a confirmed hard retention limit of ~2 months — requests for older dates silently return 0 rows rather than erroring. `scripts/archive_air4thai.py` exists to snapshot Air4Thai data into `data/air4thai_archive.csv` before it ages out, since this can't be recovered retroactively.

## API routes (`app.py`)

| Route | Purpose |
|---|---|
| `GET /`, `GET /<any-client-route>` | serves the Vue SPA shell (`frontend/dist/index.html`) via `serve_spa()`, or the matching static asset if `path` is a real file under `frontend/dist` (JS/CSS bundles, `manifest.webmanifest`, `sw.js`, icons). Vue Router then renders `/` (forecast) or `/compare` (comparison) client-side — there is no server-side distinction between these two anymore |
| `GET /index.html`, `GET /compare.html` | 302 redirect to `/` and `/compare` respectively, for old bookmarks (the routes themselves stopped being real pages when the Vue migration dropped the `.html` URLs) |
| `GET /api/forecast` | runs `get_forecast()`: fetches latest data, predicts all 24 horizons via Dynamic Routing, logs to `prediction_log` (Postgres), backfills past actuals |
| `GET /api/comparison?horizon=&model=` | per-horizon prediction/actual series + MAE/bias |
| `GET /api/comparison/overview` | aggregate MAE/bias by horizon and by model |

Note: `serve_spa`'s `/<path:path>` catch-all is a dynamic rule, but Werkzeug always gives routing priority to static rules (`/api/forecast`, etc.) over rules with converters regardless of registration order, so the catch-all can never shadow an API route — confirmed by testing, not just assumed.

## File conventions

- Model/data outputs are always saved under a new timestamped filename (e.g. `models/lgb_pm25_multistep_weighted_{timestamp}.pkl`) — never overwrite or delete a previous version. `app.py`'s `MODEL_PATH` / `XGB_MODEL_PATH` / `PROPHET_MODEL_PATH` / `BIAS_CSV_PATH` / `CALIBRATION_PATH` constants point at whichever files are currently "deployed"; older files remain as history/fallback.
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

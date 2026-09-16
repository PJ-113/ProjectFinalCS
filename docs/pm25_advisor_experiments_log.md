# สถานะการทดลองตอบ Feedback อาจารย์ (สรุปรวม)

**อัปเดตล่าสุด**: 2026-09-17

Capstone project: PM2.5 24h forecast for 20 Isaan provinces (Flask+Vue, GitHub `PJ-113/ProjectFinalCS`, branch `pooled-model-multiprovince`). Full write-up for the advisor is at `docs/advisor_feedback_response14_9.md` in the repo — combines the 3-section methodology answer with the feature-by-feature ablation table. Detailed history in `CHANGELOG.md` entries 89-110.

## Current best candidate model (not yet deployed to app.py)

`models/{lgb,xgb}_pm25_multiprovince_residual_bagged_quantile_20260917_001147.pkl` — 19 features, `pm25_neighbor_avg` computed with **k=1** (nearest province only, not k=3). MAE(P50) 24h avg: LGB 2.720, XGB 2.805. Coverage after conformal: LGB 88.8%, XGB 88.3% (target 90%). `app.py` still points at an older tag — deploying this is a separate decision the user hasn't made yet, deliberately kept out of scope each time (same "don't fix what's about to be replaced" principle as the urban/rural investigation).

## Confirmed effects (adopted into the model or reported as positive findings)

- **Spatial correlation is real**: PM2.5 correlation between province pairs vs distance, r=-0.644 (n=190 pairs).
- **`pm25_neighbor_avg` (spatial-lag feature)**: average PM2.5 of k nearest provinces at the same timestamp. Single most powerful feature in the model (leave-one-out: +4.20%/+3.01% MAE if removed, far above any other single feature). **k=1 beats k=3**: tuned k over {1,2,3,5,7,10,15,19} — MAE improves monotonically as k decreases, no reversal point found (contrary to the original hypothesis of an interior optimum). Confirmed under full retrain (-1.23%/-1.85% MAE vs the k=3 model). Likely cause: the 2nd/3rd-nearest provinces are much farther than the nearest one (Isaan geography is not evenly spaced), so averaging them in dilutes the signal.
- **Meteorological SHAP plausibility**: pressure (+), wind_speed (-), precipitation (-) all matched hypothesized direction, 18/18 checks. Humidity correlates strongly negative (-0.79 to -0.95) but via a dry-season/burning-season proxy mechanism, not the originally hypothesized hygroscopic-growth mechanism (no winter-specific difference found).
- **Dropped `time_cyclical`** (hour/month/dayofweek cyclical encoding): ablation showed removing it *improves* MAE (-1.7%/-2.5%), opposite of the original hypothesis — kept out of the final feature set.
- **Group-level vs individual-feature ablation multicollinearity proof**: `pm25_lag_rolling` group (9 features) costs +9.1%/+8.0% MAE if dropped together, but only +0.34%/-0.19% average if dropped one at a time — ~27x difference, empirically confirms multicollinearity among lag/rolling features.
- **Accuracy reporting**: advisor asked for "Accuracy" explicitly; used Accuracy = 100% - MAPE% (standard for regression/forecasting). Overall ~88% for both models. Note: MAPE-based accuracy can look *better* in burning season even though MAE is worse, because MAPE is relative to a higher baseline — report both metrics together, never MAPE alone when comparing regimes with different baseline magnitudes.

## Investigated, found no meaningful effect (still valid, reportable negative findings — not skipped items)

- **Elevation/terrain**: no significant effect within Isaan's narrow elevation range (123-237m); adding as a feature barely changed MAE (-0.11%/-0.20%).
- **Urban/rural density**: OSM raw building-count was a bulk-import data-quality artifact (`building=roof` monoculture) — fixed by switching to amenity/shop POI density. Correlates with mean PM2.5 (r=+0.32) but not with volatility/discontinuity. No model benefit when added as a feature (negligible, inconsistent-direction MAE change).
- **Ensemble (LGB+XGB) vs Dynamic Routing**: evaluated offline (simple average, weighted average tuned on calib split, oracle per-horizon routing as a hindsight-only ceiling). Result: ensembling **does not help** — LGB wins consistently at every horizon (never loses to XGB), so there's no complementary-strengths scenario for ensembling to exploit; both ensemble variants came out slightly worse than LGB alone. Coverage was also roughly unchanged (~88-89% either way). Conclusion: don't replace Dynamic Routing with a static ensemble based on this evidence. Caveat: this is necessarily an offline proxy — real Dynamic Routing is online/adaptive per-province with a rolling window and cold-start exploration, which can't be replayed exactly from one static train/calib/test split.

## Still open

- **Conformal coverage gap during high-pollution/burning-season periods**: LightGBM's coverage does not respond to any tested conformal recalibration config (group-conditional calibration helped XGBoost some: 81.4%→84.9%). Structural limitation, unresolved.
- Whether ensembling could fix that *specific* burning-season coverage gap is still untested — the ensemble evaluation above checked coverage split by calendar month (Feb-Apr), not by the severity-bin (actual >p90) slicing that revealed the original coverage problem in CHANGELOG entry 103. These are different slicings and the results aren't directly comparable — don't conflate them.

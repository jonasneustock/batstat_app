# Technical Notes

This document describes the core architecture, data pipeline, and operational
details for BatStat.

## Architecture

- `app/main.py`:
  - FastAPI routes for prediction, dashboards, feedback, data submissions,
    and admin review.
  - Model persistence to `models/trained_model.joblib`.
- `app/data/sources.py`:
  - Source adapters (Washington EV population, EPA fuel economy, Open EV Data datasets).
  - Approved user submissions source.
  - Dataset caching helpers.
- `app/ml/model.py`:
  - Training pipeline using scikit-learn with conformal prediction.
  - Predict interval logic.

## Data pipeline

1) Sources are loaded via `default_sources()` in `app/data/sources.py`.
2) Each source normalizes records into a common schema:
   - `brand`, `car_type` (model name), `age_years`, `km`
   - `fast_share`, `avg_soc`, `avg_temp_c`
   - `eol_km`
3) Remote datasets are cached under `data/source_cache` after first download.
4) Approved user submissions are loaded from `data/user_submissions_approved.csv`.

### User submissions

User submissions are stored separately and only used after admin approval.

Files:
- Pending: `data/user_submissions_pending.csv`
- Approved: `data/user_submissions_approved.csv`

Admin review endpoints:
- `/admin/submissions?token=change-me` (HTML)
- `/api/v1/admin/submissions?token=change-me` (JSON)
- `/api/v1/admin/submissions/{submission_id}/accept?token=change-me`

The approved dataset is converted into the common model schema using
range-degradation estimates. Usage-related features are derived via
deterministic hashing to keep training stable while avoiding random drift.

## Model training and persistence

Training is triggered in two ways:
- On-demand when the app starts and no model is present.
- Manually via `/api/v1/admin/retrain?token=change-me`.

The trained model is serialized using joblib to:
- `models/trained_model.joblib`

## Persistence

When using Docker Compose, named volumes persist:
- `/app/batstat_app/data` (datasets, logs, submissions)
- `/app/batstat_app/models` (trained model)

## Security notes

Admin authorization uses a static token (`ADMIN_TOKEN`) configured in
`app/main.py`. Replace this in production and consider using a proper auth
mechanism.

## Endpoints (summary)

- `/` predictor UI
- `/predict` prediction results (GET)
- `/dashboard` dataset summary
- `/contribute` submission form
- `/contribute/submit` store submission (GET)
- `/admin/submissions` admin review (GET)
- `/api/v1/admin/submissions` admin review (JSON)
- `/api/v1/admin/submissions/{id}/accept` approve submission
- `/api/v1/admin/retrain` trigger retraining
- `/api/v1/admin/status/{job_id}` check retrain status

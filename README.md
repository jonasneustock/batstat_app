# BatStat

BatStat is a FastAPI web app that predicts EV battery end-of-life mileage and
offers a simple UI for users to explore outcomes. It also supports user data
contributions that are reviewed by an admin before being used for training.

## Quick start (local)

1) Install dependencies:

```bash
python -m pip install -r requirements.txt
```

2) Run the app:

```bash
uvicorn batstat_app.app.main:app --reload --host 0.0.0.0 --port 8000
```

Open `http://localhost:8000`.

## Docker

Build and run with compose:

```bash
docker compose up --build
```

The compose file uses named volumes for:
- `batstat-data` -> `/app/batstat_app/data`
- `batstat-models` -> `/app/batstat_app/models`

## Data sources

The app loads data from:
- Washington State EV population dataset
- EPA fuel economy dataset (EV/PHEV filtered)
- Open EV Data dataset (CSV releases)
- Open EV Data specs dataset (Kilowatt JSON)
- Approved user submissions stored locally

Remote datasets are cached under `data/source_cache` after first download.

## User submissions and admin review

Users can submit vehicle data at `/contribute`. Submissions are stored in:
- `data/user_submissions_pending.csv` (pending review)
- `data/user_submissions_approved.csv` (accepted, used for training)

Admin review UI:
- `/admin/submissions?token=change-me`

Admin API:
- `/api/v1/admin/submissions?token=change-me`
- `/api/v1/admin/submissions/{submission_id}/accept?token=change-me`

Update the admin token in `app/main.py` by editing `ADMIN_TOKEN`.

## Retraining

Trigger retraining:

```text
POST /api/v1/admin/retrain?token=change-me
```

Check status:

```text
GET /api/v1/admin/status/{job_id}
```

## Local files

- Interaction logs: `data/interaction_logs.csv`
- Cached datasets: `data/source_cache`
- Pending/approved submissions: `data/user_submissions_pending.csv`, `data/user_submissions_approved.csv`
- Persisted model: `models/trained_model.joblib`

## Tests

Run all tests:

```bash
python -m unittest
```

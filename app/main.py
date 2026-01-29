"""
FastAPI application for the BatStat battery lifetime predictor.

This module defines the REST and HTML endpoints, orchestrates loading
datasets, trains the initial model, exposes a simple user interface for
prediction, and implements an admin API for asynchronous retraining.

Usage:
    uvicorn batstat_app.app.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from .data.sources import (
    DATA_CACHE_DIR,
    USER_SUBMISSIONS_APPROVED_PATH,
    default_sources,
    load_all_sources,
    normalize_car_type_name,
)
from .ml.model import TrainedModel, train_model

import numpy as np  # needed for random and array operations


# ---------------------------------------------------------------------------
# Application state

app = FastAPI(title="BatStat", docs_url=None, redoc_url=None)
templates = Jinja2Templates(directory="batstat_app/app/templates")

# Global model and lock for thread‑safe updates
MODEL: Optional[TrainedModel] = None
MODEL_LOCK = asyncio.Lock()
MODEL_DATA_SIGNATURE: Optional[str] = None

# Executor for background retraining tasks
executor = ThreadPoolExecutor(max_workers=1)

# Dataset cache and locks
DATA_CACHE: Optional[pd.DataFrame] = None
DATA_SIGNATURE: Optional[str] = None
DATA_LOCK = threading.Lock()

TRAINING_STATE = {"in_progress": False}

# Job status tracking
JOB_STATUS: Dict[str, Dict[str, Any]] = {}
ADMIN_TOKEN = os.getenv("BATSTAT_ADMIN_TOKEN", "change-me")

# ---------------------------------------------------------------------------
# Logging utilities

import csv
from pathlib import Path

STATIC_DIR = Path(__file__).resolve().parent / "static"
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Directory and file for logging requests and feedback.  This file will accumulate
# all prediction inputs along with predictions and any subsequent user
# feedback.  Storing logs in a simple CSV makes it easy to inspect and
# process the history of interactions.
LOG_FILE = Path(__file__).resolve().parent.parent / "data" / "interaction_logs.csv"
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "trained_model.joblib"
MODEL_META_PATH = Path(__file__).resolve().parent.parent / "models" / "trained_model_meta.json"
SUBMISSIONS_PENDING_FILE = (
    Path(__file__).resolve().parent.parent / "data" / "user_submissions_pending.csv"
)
SUBMISSIONS_APPROVED_FILE = (
    Path(__file__).resolve().parent.parent / "data" / "user_submissions_approved.csv"
)
SUBMISSION_FIELDS = [
    "submission_id",
    "submitted_at",
    "brand",
    "car_type",
    "model_year",
    "odometer_km",
    "range_original_km",
    "range_current_km",
]
APPROVED_FIELDS = SUBMISSION_FIELDS + ["approved_at"]

def log_interaction(
    brand: str,
    car_type: str,
    build_year: int,
    km_current: float,
    predicted_lower: float,
    predicted_upper: float,
    feedback: Optional[str] = None,
) -> None:
    """Append a single interaction record to the log file.

    Each entry captures the vehicle details, prediction interval and optional
    user feedback.  When feedback is None the entry corresponds to the
    initial prediction; when provided it represents a follow‑up by the user.
    """
    # Ensure the log directory exists
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    write_header = not LOG_FILE.exists()
    # Append the record as a row
    with LOG_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                "timestamp",
                "brand",
                "car_type",
                "build_year",
                "km_current",
                "predicted_lower",
                "predicted_upper",
                "feedback",
            ])
        writer.writerow([
            datetime.now().isoformat(),
            brand,
            car_type,
            build_year,
            km_current,
            predicted_lower,
            predicted_upper,
            feedback or "",
        ])


def _ensure_csv_with_header(path: Path, fieldnames: List[str]) -> None:
    """Ensure a CSV file exists with the given header row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)


def log_submission_pending(row: Dict[str, Any]) -> None:
    """Persist a pending user submission for admin review."""
    _ensure_csv_with_header(SUBMISSIONS_PENDING_FILE, SUBMISSION_FIELDS)
    with SUBMISSIONS_PENDING_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUBMISSION_FIELDS)
        writer.writerow(row)


def load_pending_submissions() -> List[Dict[str, str]]:
    """Load all pending submissions as a list of dicts."""
    if not SUBMISSIONS_PENDING_FILE.exists():
        return []
    with SUBMISSIONS_PENDING_FILE.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def approve_submission(submission_id: str) -> bool:
    """Move a pending submission into the approved dataset."""
    pending = load_pending_submissions()
    approved_row: Optional[Dict[str, str]] = None
    remaining: List[Dict[str, str]] = []
    for row in pending:
        if row.get("submission_id") == submission_id:
            approved_row = row
        else:
            remaining.append(row)

    if approved_row is None:
        return False

    _ensure_csv_with_header(SUBMISSIONS_APPROVED_FILE, APPROVED_FIELDS)
    approved_row = dict(approved_row)
    approved_row["approved_at"] = datetime.utcnow().isoformat()

    with SUBMISSIONS_APPROVED_FILE.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=APPROVED_FIELDS)
        writer.writerow(approved_row)

    with SUBMISSIONS_PENDING_FILE.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SUBMISSION_FIELDS)
        writer.writeheader()
        for row in remaining:
            writer.writerow(row)

    return True


def _validate_submission_values(
    *,
    brand: str,
    car_type: str,
    model_year: int,
    odometer_km: float,
    range_original_km: float,
    range_current_km: float,
) -> None:
    errors: List[str] = []
    current_year = datetime.now().year
    if not brand.strip():
        errors.append("brand is required")
    if not car_type.strip():
        errors.append("car name is required")
    if model_year < 1990 or model_year > current_year:
        errors.append("model_year must be between 1990 and the current year")
    if odometer_km < 0:
        errors.append("odometer_km must be non-negative")
    if range_original_km <= 0:
        errors.append("range_original_km must be positive")
    if range_current_km <= 0:
        errors.append("range_current_km must be positive")
    if range_current_km > range_original_km:
        errors.append("range_current_km must be less than or equal to range_original_km")
    if errors:
        raise HTTPException(status_code=400, detail="; ".join(errors))


def _require_admin_token(token: str) -> None:
    if token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")


def load_model_from_disk() -> Optional[TrainedModel]:
    """Load a persisted model from disk if it exists."""
    if not MODEL_PATH.exists():
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception:
        return None


def save_model_to_disk(model: TrainedModel) -> None:
    """Persist the trained model to disk for reuse across restarts."""
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)


def load_model_metadata() -> Optional[Dict[str, Any]]:
    """Load model metadata such as the data signature."""
    if not MODEL_META_PATH.exists():
        return None
    try:
        return json.loads(MODEL_META_PATH.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_model_metadata(data_signature: str) -> None:
    """Persist the data signature used to train the current model."""
    MODEL_META_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = {"data_signature": data_signature, "trained_at": datetime.utcnow().isoformat()}
    MODEL_META_PATH.write_text(json.dumps(payload), encoding="utf-8")


def _signature_paths() -> List[Path]:
    files: List[Path] = []
    if DATA_CACHE_DIR.exists():
        files.extend(sorted(DATA_CACHE_DIR.glob("*")))
    if USER_SUBMISSIONS_APPROVED_PATH.exists():
        files.append(USER_SUBMISSIONS_APPROVED_PATH)
    return files


def _compute_data_signature(df: pd.DataFrame) -> str:
    hasher = hashlib.sha256()
    hasher.update(f"rows:{len(df)}".encode("utf-8"))
    hasher.update((",".join(df.columns)).encode("utf-8"))
    for path in _signature_paths():
        try:
            stat = path.stat()
        except FileNotFoundError:
            continue
        hasher.update(f"{path}:{stat.st_mtime_ns}:{stat.st_size}".encode("utf-8"))
    return hasher.hexdigest()


def refresh_dataset_cache(force: bool = False) -> None:
    """Load and cache the training dataset."""
    global DATA_CACHE, DATA_SIGNATURE
    if not force and DATA_CACHE is not None:
        return
    df = load_all_sources(default_sources())
    signature = _compute_data_signature(df)
    with DATA_LOCK:
        DATA_CACHE = df
        DATA_SIGNATURE = signature


def get_cached_dataset() -> pd.DataFrame:
    """Return the cached dataset, loading it if needed."""
    if DATA_CACHE is None:
        refresh_dataset_cache(force=True)
    with DATA_LOCK:
        return DATA_CACHE.copy() if DATA_CACHE is not None else pd.DataFrame()


def get_dataset_snapshot() -> tuple[pd.DataFrame, Optional[str]]:
    """Return a snapshot of the cached dataset and its signature."""
    if DATA_CACHE is None:
        refresh_dataset_cache(force=True)
    with DATA_LOCK:
        df = DATA_CACHE.copy() if DATA_CACHE is not None else pd.DataFrame()
        signature = DATA_SIGNATURE
    if signature is None:
        signature = _compute_data_signature(df)
    return df, signature


def _schedule_training(reason: str, *, refresh_data: bool = False) -> Optional[str]:
    """Schedule model training in the background if not already running."""
    if TRAINING_STATE["in_progress"]:
        return None
    job_id = str(uuid.uuid4())
    TRAINING_STATE["in_progress"] = True
    JOB_STATUS[job_id] = {"status": "queued", "reason": reason}
    loop = asyncio.get_event_loop()
    loop.run_in_executor(executor, _run_training_job, job_id, refresh_data)
    return job_id


def _run_training_job(job_id: str, refresh_data: bool) -> None:
    """Train a model in a background thread and swap in on success."""
    try:
        JOB_STATUS[job_id]["status"] = "running"
        JOB_STATUS[job_id]["started_at"] = datetime.utcnow().isoformat()
        if refresh_data:
            refresh_dataset_cache(force=True)
        df, signature = get_dataset_snapshot()
        if df.empty:
            raise RuntimeError("No training data available for retraining")
        new_model = train_model(df, random_state=np.random.randint(0, 2**32 - 1))
        asyncio.run(_set_model(new_model, signature))
        JOB_STATUS[job_id]["status"] = "completed"
        JOB_STATUS[job_id]["finished_at"] = datetime.utcnow().isoformat()
    except Exception as exc:
        JOB_STATUS[job_id]["status"] = "failed"
        JOB_STATUS[job_id]["error"] = str(exc)
    finally:
        TRAINING_STATE["in_progress"] = False


async def get_model() -> TrainedModel:
    """Return the currently loaded model, loading if necessary."""
    global MODEL
    async with MODEL_LOCK:
        if MODEL is None:
            raise HTTPException(status_code=503, detail="Model is initializing")
        return MODEL


@app.on_event("startup")
async def startup_event() -> None:
    """Warm caches and schedule training if needed."""
    global MODEL, MODEL_DATA_SIGNATURE
    refresh_dataset_cache(force=True)
    MODEL = load_model_from_disk()
    metadata = load_model_metadata()
    MODEL_DATA_SIGNATURE = metadata.get("data_signature") if metadata else None
    if MODEL is None:
        _schedule_training("startup_initial", refresh_data=False)
    elif DATA_SIGNATURE is not None and MODEL_DATA_SIGNATURE != DATA_SIGNATURE:
        _schedule_training("startup_data_change", refresh_data=False)


# ---------------------------------------------------------------------------
# Templates helpers

def format_number(value: float, decimals: int = 0) -> str:
    """Format numeric values with thousand separators and fixed decimals."""
    return f"{value:,.{decimals}f}".replace(",", " ")  # replace comma with non‑breaking space

templates.env.filters["format_number"] = format_number

# Format any value: numbers are formatted with thousands separators, others returned unchanged
def format_value(value: Any) -> Any:
    if isinstance(value, (int, float)):
        return format_number(value)
    return value

templates.env.filters["format_value"] = format_value


# ---------------------------------------------------------------------------
# HTML routes

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """
    Render the main form for users to input vehicle details.
    """
    df = get_cached_dataset()
    brands = sorted(df["brand"].dropna().unique())
    car_types = sorted(df["car_type"].dropna().unique())
    brand_car_types = (
        df.dropna(subset=["brand", "car_type"])
        .groupby("brand")["car_type"]
        .apply(lambda values: sorted(set(values)))
        .to_dict()
    )
    current_year = datetime.now().year
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "brands": brands,
            "car_types": car_types,
            "brand_car_types": brand_car_types,
            "current_year": current_year,
        },
    )


@app.get("/contribute", response_class=HTMLResponse)
async def contribute(request: Request):
    """
    Render the form for users to contribute vehicle data.
    """
    current_year = datetime.now().year
    brands: List[str] = []
    car_types: List[str] = []
    try:
        df = get_cached_dataset()
        brands = sorted(df["brand"].dropna().unique())
        car_types = sorted(df["car_type"].dropna().unique())
        brand_car_types = (
            df.dropna(subset=["brand", "car_type"])
            .groupby("brand")["car_type"]
            .apply(lambda values: sorted(set(values)))
            .to_dict()
        )
    except Exception:
        brand_car_types = {}
    return templates.TemplateResponse(
        "contribute.html",
        {
            "request": request,
            "brands": brands,
            "car_types": car_types,
            "brand_car_types": brand_car_types,
            "current_year": current_year,
        },
    )


@app.get("/contribute/submit", response_class=HTMLResponse)
async def contribute_submit(request: Request):
    """
    Persist a user submission for admin review.
    """
    params = request.query_params
    try:
        brand = params["brand"]
        car_type = params["car_type"]
        model_year = int(params["model_year"])
        odometer_km = float(params["odometer_km"])
        range_original_km = float(params["range_original_km"])
        range_current_km = float(params["range_current_km"])
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Missing parameter: {exc}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid parameter: {exc}")

    _validate_submission_values(
        brand=brand,
        car_type=car_type,
        model_year=model_year,
        odometer_km=odometer_km,
        range_original_km=range_original_km,
        range_current_km=range_current_km,
    )

    normalized_car_type = normalize_car_type_name(car_type)
    submission_id = str(uuid.uuid4())
    row = {
        "submission_id": submission_id,
        "submitted_at": datetime.utcnow().isoformat(),
        "brand": brand.strip(),
        "car_type": normalized_car_type,
        "model_year": model_year,
        "odometer_km": odometer_km,
        "range_original_km": range_original_km,
        "range_current_km": range_current_km,
    }
    try:
        log_submission_pending(row)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to store submission: {exc}")

    return templates.TemplateResponse(
        "contribute_result.html",
        {
            "request": request,
            "submission_id": submission_id,
        },
    )


@app.get("/predict", response_class=HTMLResponse)
async def predict(request: Request):
    """
    Handle form submission via query parameters and display the prediction results.

    Using a GET request avoids the dependency on the optional python‑multipart
    package, which isn't available in this environment.  Parameters are
    extracted from the query string.
    """
    # Extract query parameters
    params = request.query_params
    try:
        brand = params["brand"]
        car_type = params["car_type"]
        build_year = int(params["build_year"])
        km_current = float(params["km_current"])
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=f"Missing parameter: {exc}")
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Invalid parameter: {exc}")
    model = await get_model()
    current_year = datetime.now().year
    age_years = max(0.0, current_year - build_year)
    normalized_car_type = normalize_car_type_name(car_type)
    # Create input dataframe with dummy values for uncollected features
    input_df = pd.DataFrame(
        [
            {
                "brand": brand,
                "car_type": normalized_car_type,
                "age_years": age_years,
                "km": km_current,
                # Use typical averages for features the user does not provide
                "fast_share": 0.3,
                "avg_soc": 0.5,
                "avg_temp_c": 25.0,
            }
        ]
    )
    y_pred, lower, upper = model.predict_interval(input_df)
    # Ensure monotonic interval (lower >= current km)
    lower = np.maximum(lower, km_current)
    upper = np.maximum(upper, lower + 1.0)
    result = {
        "point": float(y_pred[0]),
        "lower": float(lower[0]),
        "upper": float(upper[0]),
    }

    # --------------------------------------------------------------------
    # Compute 80% capacity life estimates
    #
    # We assume a roughly linear relationship between odometer mileage and
    # battery capacity.  Many manufacturers consider a battery to be at end
    # of life when it retains around 70% of its original capacity.  Dropping
    # to 80% capacity represents two‑thirds of the capacity loss from
    # 100% to 70% (a 20% drop versus a 30% drop).  Therefore, as a rough
    # approximation we take 80% of the predicted end‑of‑life odometer
    # reading.  This approach produces estimates that are in line with
    # empirical studies showing that most EV batteries still have more than
    # 80% capacity at around 200,000 km【459163475393697†L56-L61】.  If the
    # resulting 80% threshold is below the current odometer reading, we
    # clamp it to the current mileage to avoid negative remaining range.
    predicted_80_lower = result["lower"] * 0.8
    predicted_80_upper = result["upper"] * 0.8
    # Ensure the 80% thresholds are not less than the current odometer
    predicted_80_lower = max(predicted_80_lower, km_current)
    predicted_80_upper = max(predicted_80_upper, predicted_80_lower)

    # Remaining kilometres until 80% capacity threshold
    remaining_80_lower = predicted_80_lower - km_current
    remaining_80_upper = predicted_80_upper - km_current
    # Log the prediction interaction.  We don't capture feedback at this stage.
    try:
        log_interaction(
            brand=brand,
            car_type=normalized_car_type,
            build_year=build_year,
            km_current=km_current,
            predicted_lower=result["lower"],
            predicted_upper=result["upper"],
            feedback=None,
        )
    except Exception:
        # Logging should never prevent the response.  If writing the log fails,
        # silently ignore the error.
        pass
    # Compute expected remaining distance range
    remaining_lower = result["lower"] - km_current
    remaining_upper = result["upper"] - km_current
    return templates.TemplateResponse(
        "result.html",
        {
            "request": request,
            "brand": brand,
            "car_type": normalized_car_type,
            "build_year": build_year,
            "km_current": km_current,
            "result": result,
            "remaining_lower": remaining_lower,
            "remaining_upper": remaining_upper,
            # 80% capacity values
            "predicted_80_lower": predicted_80_lower,
            "predicted_80_upper": predicted_80_upper,
            "remaining_80_lower": remaining_80_lower,
            "remaining_80_upper": remaining_80_upper,
        },
    )


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """
    Display a simple dashboard with summary statistics of the training data.
    """
    df = get_cached_dataset()
    summary = {
        "count": len(df),
        "min_age": df["age_years"].min(),
        "max_age": df["age_years"].max(),
        "min_km": df["km"].min(),
        "max_km": df["km"].max(),
        "min_eol": df["eol_km"].min(),
        "max_eol": df["eol_km"].max(),
    }
    # Show a preview of the first 10 rows
    preview = df.head(10).to_dict(orient="records")
    return templates.TemplateResponse(
        "dashboard.html",
        {"request": request, "summary": summary, "preview": preview},
    )


# ---------------------------------------------------------------------------
# Feedback route

@app.get("/feedback", response_class=HTMLResponse)
async def feedback(
    request: Request,
    brand: str = Query(...),
    car_type: str = Query(...),
    build_year: int = Query(...),
    km_current: float = Query(...),
    predicted_lower: float = Query(...),
    predicted_upper: float = Query(...),
    fb: str = Query(..., alias="fb"),
):
    """
    Record user feedback for a previously generated prediction.

    Users arrive here via links in the result page specifying the vehicle
    parameters, the prediction interval and a feedback indicator.  Feedback
    values should be "positive" or "negative".  The interaction is logged
    and a confirmation page is shown.
    """
    # Normalize feedback value to one of the accepted labels
    feedback_label = "positive" if fb.lower().startswith("pos") else "negative"
    normalized_car_type = normalize_car_type_name(car_type)
    try:
        log_interaction(
            brand=brand,
            car_type=normalized_car_type,
            build_year=build_year,
            km_current=km_current,
            predicted_lower=predicted_lower,
            predicted_upper=predicted_upper,
            feedback=feedback_label,
        )
    except Exception:
        pass
    return templates.TemplateResponse(
        "feedback.html",
        {
            "request": request,
            "brand": brand,
            "car_type": normalized_car_type,
            "build_year": build_year,
            "km_current": km_current,
            "predicted_lower": predicted_lower,
            "predicted_upper": predicted_upper,
            "feedback": feedback_label,
        },
    )


# ---------------------------------------------------------------------------
# Admin routes and API

class RetrainResponse(BaseModel):
    job_id: str
    status: str


@app.get("/admin/submissions", response_class=HTMLResponse)
async def admin_submissions(request: Request, token: str = Query(...)):
    """
    Admin view for pending submissions and review actions.
    """
    _require_admin_token(token)
    pending = load_pending_submissions()
    return templates.TemplateResponse(
        "admin_submissions.html",
        {
            "request": request,
            "pending": pending,
            "token": token,
        },
    )


@app.get("/admin/submissions/{submission_id}/accept", response_class=HTMLResponse)
async def admin_accept_submission(request: Request, submission_id: str, token: str = Query(...)):
    """
    Approve a pending submission and move it into the training dataset.
    """
    _require_admin_token(token)
    accepted = approve_submission(submission_id)
    if not accepted:
        raise HTTPException(status_code=404, detail="Submission not found")
    _schedule_training("approved_submission", refresh_data=True)
    return templates.TemplateResponse(
        "admin_submission_result.html",
        {
            "request": request,
            "submission_id": submission_id,
            "token": token,
        },
    )


@app.get("/api/v1/admin/submissions")
async def admin_submissions_api(token: str = Query(...)):
    """
    Return pending submissions for admin review.
    """
    _require_admin_token(token)
    pending = load_pending_submissions()
    return {"count": len(pending), "submissions": pending}


@app.get("/api/v1/admin/submissions/{submission_id}/accept")
async def admin_accept_submission_api(submission_id: str, token: str = Query(...)):
    """
    Approve a pending submission via API.
    """
    _require_admin_token(token)
    accepted = approve_submission(submission_id)
    if not accepted:
        raise HTTPException(status_code=404, detail="Submission not found")
    _schedule_training("approved_submission", refresh_data=True)
    return {"submission_id": submission_id, "status": "accepted"}


async def _set_model(new_model: TrainedModel, data_signature: Optional[str]) -> None:
    """Replace the global model in a thread‑safe manner."""
    global MODEL, MODEL_DATA_SIGNATURE
    async with MODEL_LOCK:
        MODEL = new_model
        MODEL_DATA_SIGNATURE = data_signature
        try:
            save_model_to_disk(new_model)
            if data_signature is not None:
                save_model_metadata(data_signature)
        except Exception:
            pass


@app.post("/api/v1/admin/retrain", response_model=RetrainResponse)
async def retrain(token: str = Query(..., description="Admin token")):
    """
    Hidden admin endpoint that triggers asynchronous retraining of the model.

    You must supply the correct token as a query parameter.  The training
    process runs in a background thread; the endpoint returns immediately
    with a job ID which can be polled via the status endpoint.
    """
    # In a real application use secure authentication; here use simple token
    _require_admin_token(token)
    job_id = _schedule_training("admin_retrain", refresh_data=True)
    if job_id is None:
        raise HTTPException(status_code=409, detail="Training already running")
    return RetrainResponse(job_id=job_id, status=JOB_STATUS[job_id]["status"])


@app.get("/api/v1/admin/status/{job_id}")
async def retrain_status(job_id: str):
    """
    Poll the status of a retraining job.

    Returns the current status and metadata for the job.  If the job ID is
    unknown, a 404 error is returned.
    """
    if job_id not in JOB_STATUS:
        raise HTTPException(status_code=404, detail="Job ID not found")
    return JOB_STATUS[job_id]

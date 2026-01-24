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
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import BackgroundTasks, Depends, FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from .data.sources import ExampleSource, load_all_sources
from .ml.model import TrainedModel, train_model

import numpy as np  # needed for random and array operations


# ---------------------------------------------------------------------------
# Application state

app = FastAPI(title="BatStat", docs_url=None, redoc_url=None)
templates = Jinja2Templates(directory="batstat_app/app/templates")

# Global model and lock for thread‑safe updates
MODEL: Optional[TrainedModel] = None
MODEL_LOCK = asyncio.Lock()

# Executor for background retraining tasks
executor = ThreadPoolExecutor(max_workers=1)

# Job status tracking
JOB_STATUS: Dict[str, Dict[str, Any]] = {}

# ---------------------------------------------------------------------------
# Logging utilities

import csv
from pathlib import Path

# Directory and file for logging requests and feedback.  This file will accumulate
# all prediction inputs along with predictions and any subsequent user
# feedback.  Storing logs in a simple CSV makes it easy to inspect and
# process the history of interactions.
LOG_FILE = Path(__file__).resolve().parent.parent / "data" / "interaction_logs.csv"

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


async def get_model() -> TrainedModel:
    """Return the currently loaded model, loading if necessary."""
    global MODEL
    async with MODEL_LOCK:
        if MODEL is None:
            # Load dataset and train initial model
            df = load_all_sources([ExampleSource()])
            if df.empty:
                raise HTTPException(status_code=500, detail="No training data available")
            MODEL = train_model(df)
        return MODEL


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
    # Preload model and dataset to populate brand/type options
    model = await get_model()
    # Access the training data used to build the model; not stored directly
    # here, so regenerate using ExampleSource.  For a real implementation you
    # might persist unique values separately.
    df = load_all_sources([ExampleSource()])
    brands = sorted(df["brand"].unique())
    car_types = sorted(df["car_type"].unique())
    current_year = datetime.now().year
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "brands": brands,
            "car_types": car_types,
            "current_year": current_year,
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
    # Create input dataframe with dummy values for uncollected features
    input_df = pd.DataFrame(
        [
            {
                "brand": brand,
                "car_type": car_type,
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
            car_type=car_type,
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
            "car_type": car_type,
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
    df = load_all_sources([ExampleSource()])
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
    try:
        log_interaction(
            brand=brand,
            car_type=car_type,
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
            "car_type": car_type,
            "build_year": build_year,
            "km_current": km_current,
            "predicted_lower": predicted_lower,
            "predicted_upper": predicted_upper,
            "feedback": feedback_label,
        },
    )


# ---------------------------------------------------------------------------
# Admin API for retraining

class RetrainResponse(BaseModel):
    job_id: str
    status: str


def _run_retraining(job_id: str) -> None:
    """Internal function executed in thread pool to retrain the model."""
    try:
        JOB_STATUS[job_id]["status"] = "running"
        JOB_STATUS[job_id]["started_at"] = datetime.utcnow().isoformat()
        # Load fresh dataset (could be updated with new sources)
        df = load_all_sources([ExampleSource()])
        if df.empty:
            raise RuntimeError("No training data available for retraining")
        new_model = train_model(df, random_state=np.random.randint(0, 2**32 - 1))
        # Update global model atomically using a new event loop in this thread
        asyncio.run(_set_model(new_model))
        JOB_STATUS[job_id]["status"] = "completed"
        JOB_STATUS[job_id]["finished_at"] = datetime.utcnow().isoformat()
    except Exception as exc:
        JOB_STATUS[job_id]["status"] = "failed"
        JOB_STATUS[job_id]["error"] = str(exc)


async def _set_model(new_model: TrainedModel) -> None:
    """Replace the global model in a thread‑safe manner."""
    global MODEL
    async with MODEL_LOCK:
        MODEL = new_model


@app.post("/api/v1/admin/retrain", response_model=RetrainResponse)
async def retrain(token: str = Query(..., description="Admin token")):
    """
    Hidden admin endpoint that triggers asynchronous retraining of the model.

    You must supply the correct token as a query parameter.  The training
    process runs in a background thread; the endpoint returns immediately
    with a job ID which can be polled via the status endpoint.
    """
    # In a real application use secure authentication; here use simple token
    expected_token = "change-me"
    if token != expected_token:
        raise HTTPException(status_code=401, detail="Unauthorized")
    job_id = str(uuid.uuid4())
    JOB_STATUS[job_id] = {"status": "queued"}
    # Schedule background training
    loop = asyncio.get_event_loop()
    loop.run_in_executor(executor, _run_retraining, job_id)
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
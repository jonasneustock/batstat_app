"""
BatStat application package.

This package contains the FastAPI web application, data ingestion utilities,
machine learning training pipeline, and API endpoints for battery lifetime
prediction.  The top-level module `main` exposes a FastAPI instance that
serves both HTML pages and JSON APIs.

The design follows a minimal structure to remain easy to understand while
providing the requested features:

* A synthetic data source that simulates open-source EV battery statistics
  and is extensible so that real datasets can be plugged in.
* A model training routine based on scikit‑learn, with a conformal
  prediction wrapper to produce prediction intervals.
* A simple user interface implemented with Jinja2 templates for inputting
  vehicle information and displaying predictions.
* A hidden admin API to trigger asynchronous retraining of the model.
"""

from .main import app  # noqa: F401
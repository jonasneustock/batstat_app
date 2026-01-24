"""
Machine learning model training for BatStat.

This module exposes utilities to train a regression model predicting the
battery end‑of‑life odometer reading given vehicle and usage features.  It
also implements a simple conformal prediction wrapper to estimate
prediction intervals.  The goal is not to achieve state‑of‑the‑art
accuracy but to provide reasonable uncertainty bounds that can guide
decision making.

The modelling pipeline uses:

* OneHotEncoder for categorical variables (brand, car_type)
* StandardScaler for numeric variables (age_years, km, fast_share, avg_soc, avg_temp_c)
* GradientBoostingRegressor as the base estimator

Residual conformal prediction is applied on a held‑out calibration set to
derive quantiles of the prediction error.  These quantiles are used to
construct prediction intervals for new inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


@dataclass
class TrainedModel:
    """Container for the trained pipeline and conformal prediction quantiles."""

    pipeline: Pipeline
    q_low: float
    q_high: float
    feature_names: List[str]

    def predict_interval(self, X: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (point_estimate, lower_bound, upper_bound) arrays for input X."""
        y_pred = self.pipeline.predict(X)
        lower = y_pred + self.q_low
        upper = y_pred + self.q_high
        return y_pred, lower, upper


def train_model(data: pd.DataFrame, random_state: int = 42) -> TrainedModel:
    """
    Train the regression model and compute conformal prediction quantiles.

    :param data: DataFrame with training data.  Must include the 'eol_km'
                 column as target and the feature columns specified in
                 feature list below.
    :param random_state: Seed for reproducibility.
    :returns: TrainedModel instance containing the fitted pipeline and
              quantiles for residual conformal prediction.
    """
    # Define features
    categorical_features = ["brand", "car_type"]
    numeric_features = ["age_years", "km", "fast_share", "avg_soc", "avg_temp_c"]

    feature_names = categorical_features + numeric_features
    if not set(feature_names + ["eol_km"]).issubset(data.columns):
        missing = set(feature_names + ["eol_km"]) - set(data.columns)
        raise ValueError(f"Training data missing required columns: {missing}")

    X = data[feature_names]
    y = data["eol_km"]

    # Split into training and calibration sets
    X_train, X_calib, y_train, y_calib = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    # Preprocessing pipeline
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
            ("num", StandardScaler(), numeric_features),
        ]
    )

    regressor = GradientBoostingRegressor(random_state=random_state)
    model = Pipeline(steps=[("preprocessor", preprocessor), ("regressor", regressor)])

    # Fit model
    model.fit(X_train, y_train)

    # Calibration residuals
    y_calib_pred = model.predict(X_calib)
    residuals = y_calib - y_calib_pred
    # Compute quantiles for 90% prediction interval (alpha=0.1)
    alpha = 0.1
    lower_q = np.quantile(residuals, alpha / 2)
    upper_q = np.quantile(residuals, 1 - alpha / 2)

    return TrainedModel(pipeline=model, q_low=lower_q, q_high=upper_q, feature_names=feature_names)
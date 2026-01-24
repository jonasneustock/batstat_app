"""
Data source definitions for BatStat.

The application is designed to pull open‑source battery datasets.  To make
the app runnable out of the box, this module defines an ExampleSource
that generates a synthetic dataset.  Additional sources can be integrated
by subclassing `DataSource` and implementing the `load` method to return
a pandas DataFrame with the following columns:

    brand:          string, manufacturer name
    car_type:       string, vehicle type (e.g. sedan, SUV)
    age_years:      float, age of the vehicle in years
    km:             float, current odometer reading in kilometres
    fast_share:     float, proportion of charges performed at fast chargers (0–1)
    avg_soc:        float, average state of charge used (0–1)
    avg_temp_c:     float, average battery temperature during operation
    eol_km:         float, total kilometres travelled when the battery reached end‑of‑life

The modelling code will automatically convert these into features and a
target value.  All numeric columns must be finite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import pandas as pd


class DataSource:
    """Abstract base class for a data source."""

    def load(self) -> pd.DataFrame:
        """
        Return a pandas DataFrame containing all records from this source.

        Implementations should adhere to the column specification defined
        in this module.  Returning an empty DataFrame is permitted but
        discouraged, as it will result in training failures.
        """
        raise NotImplementedError


@dataclass
class ExampleSource(DataSource):
    """Synthetic data generator used for demonstration purposes."""

    n_samples: int = 500
    random_state: int = 42

    def load(self) -> pd.DataFrame:
        rng = np.random.default_rng(self.random_state)
        brands = np.array(["Tesla", "Nissan", "BMW", "VW", "Renault"])
        car_types = np.array(["Sedan", "SUV", "Hatchback", "Coupe"])
        brand_col = rng.choice(brands, size=self.n_samples)
        type_col = rng.choice(car_types, size=self.n_samples)
        age_years = rng.uniform(0, 12, size=self.n_samples)
        km = rng.uniform(0, 250_000, size=self.n_samples)
        fast_share = rng.uniform(0.0, 0.8, size=self.n_samples)
        avg_soc = rng.uniform(0.2, 0.9, size=self.n_samples)
        avg_temp_c = rng.uniform(10, 45, size=self.n_samples)

        # Simple heuristic for EOL km based on age, charging behaviour and noise
        baseline = 300_000
        eol_km = (
            baseline
            - age_years * rng.uniform(15_000, 25_000, size=self.n_samples)
            - fast_share * rng.uniform(40_000, 70_000, size=self.n_samples)
            - (avg_soc - 0.5) * rng.uniform(8_000, 12_000, size=self.n_samples)
            + rng.normal(0, 15_000, size=self.n_samples)
        )
        # Ensure EOL is at least the current km
        eol_km = np.maximum(eol_km, km + rng.uniform(10_000, 50_000, size=self.n_samples))

        return pd.DataFrame(
            {
                "brand": brand_col,
                "car_type": type_col,
                "age_years": age_years,
                "km": km,
                "fast_share": fast_share,
                "avg_soc": avg_soc,
                "avg_temp_c": avg_temp_c,
                "eol_km": eol_km,
            }
        )


def load_all_sources(sources: Sequence[DataSource]) -> pd.DataFrame:
    """Concatenate data from all provided sources."""
    frames: List[pd.DataFrame] = []
    for source in sources:
        df = source.load()
        if not isinstance(df, pd.DataFrame):
            raise TypeError(f"DataSource {source} returned non-DataFrame: {type(df)}")
        frames.append(df)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
"""
Data source definitions for BatStat.

The application is designed to pull open‑source battery datasets.  This
module includes real-world EV sources plus an ExampleSource used for
tests or local experimentation. Additional sources can be integrated by
subclassing `DataSource` and implementing the `load` method to return a
pandas DataFrame with the following columns:

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
from datetime import datetime
from pathlib import Path
from typing import List, Sequence
import hashlib
import urllib.request

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


DATA_CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / "source_cache"
USER_SUBMISSIONS_APPROVED_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "user_submissions_approved.csv"
)


def _read_csv_cached(url: str, cache_path: Path) -> pd.DataFrame:
    """
    Read a CSV from a URL with a local cache fallback.

    The CSV is cached on first successful download. If the download fails
    and a cache exists, the cached file is used instead.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if cache_path.exists():
        return pd.read_csv(cache_path)
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = response.read()
        cache_path.write_bytes(data)
        return pd.read_csv(cache_path)
    except Exception:
        if cache_path.exists():
            return pd.read_csv(cache_path)
        raise


def _hash_fraction(value: str, salt: str) -> float:
    """Deterministic 0-1 hash for stable pseudo-variation."""
    digest = hashlib.sha256(f"{salt}:{value}".encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


def _hash_series_fraction(series: pd.Series, salt: str) -> pd.Series:
    return series.fillna("").astype(str).apply(lambda v: _hash_fraction(v, salt))


def _derive_usage_features(base_id: pd.Series, is_phev: pd.Series) -> pd.DataFrame:
    """Derive proxy usage features when only partial telemetry is available."""
    frac_fast = _hash_series_fraction(base_id, "fast")
    frac_soc = _hash_series_fraction(base_id, "soc")
    frac_temp = _hash_series_fraction(base_id, "temp")

    fast_share = np.where(is_phev, 0.12, 0.25) + (frac_fast - 0.5) * 0.18
    fast_share = np.clip(fast_share, 0.02, 0.6)

    avg_soc = 0.55 + (frac_soc - 0.5) * 0.16
    avg_soc = np.clip(avg_soc, 0.25, 0.85)

    avg_temp_c = 12 + frac_temp * 18

    return pd.DataFrame(
        {
            "fast_share": fast_share.astype(float),
            "avg_soc": avg_soc.astype(float),
            "avg_temp_c": avg_temp_c.astype(float),
        }
    )


def _build_feature_frame(
    *,
    base_id: pd.Series,
    brand: pd.Series,
    car_type: pd.Series,
    age_years: pd.Series,
    range_km: pd.Series,
    is_phev: pd.Series,
) -> pd.DataFrame:
    """Create modeled features from partially observed EV datasets."""
    age_years = age_years.clip(lower=0)
    range_km = range_km.fillna(0).clip(lower=0)

    frac_usage = _hash_series_fraction(base_id, "usage")
    frac_eol = _hash_series_fraction(base_id, "eol")

    # Annual distance increases with electric range; PHEVs tend to drive fewer EV kms.
    annual_km = 12_000 + range_km * 12.0
    annual_km = annual_km * (0.85 + 0.3 * frac_usage)
    annual_km = annual_km * np.where(is_phev, 0.85, 1.0)
    km = age_years * annual_km

    usage_features = _derive_usage_features(base_id, is_phev)
    fast_share = usage_features["fast_share"]
    avg_soc = usage_features["avg_soc"]
    avg_temp_c = usage_features["avg_temp_c"]

    # End-of-life mileage estimate informed by range and EV/PHEV class.
    base_life = np.where(is_phev, 140_000, 190_000)
    eol_km = km + (base_life + range_km * 120.0) * (0.9 + 0.2 * frac_eol)
    eol_km = np.maximum(eol_km, km + 10_000)

    return pd.DataFrame(
        {
            "brand": brand.astype(str),
            "car_type": car_type.astype(str),
            "age_years": age_years.astype(float),
            "km": km.astype(float),
            "fast_share": fast_share.astype(float),
            "avg_soc": avg_soc.astype(float),
            "avg_temp_c": avg_temp_c.astype(float),
            "eol_km": eol_km.astype(float),
        }
    )


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


@dataclass
class WashingtonEVPopulationSource(DataSource):
    """Washington State EV population dataset."""

    url: str = "https://data.wa.gov/resource/f6w7-q2d2.csv"
    cache_path: Path = DATA_CACHE_DIR / "wa_ev_population.csv"
    limit: int = 5000

    def load(self) -> pd.DataFrame:
        url = f"{self.url}?$limit={self.limit}"
        df = _read_csv_cached(url, self.cache_path)
        df = df.rename(columns=str.lower)
        required_cols = ["vin_1_10", "model_year", "make", "ev_type", "electric_range"]
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"WA EV dataset missing columns: {missing}")

        current_year = datetime.now().year
        model_year = pd.to_numeric(df["model_year"], errors="coerce")
        age_years = current_year - model_year
        make = df["make"].astype(str).str.title()
        ev_type = df["ev_type"].astype(str)
        car_type = ev_type.replace(
            {
                "Battery Electric Vehicle (BEV)": "BEV",
                "Plug-in Hybrid Electric Vehicle (PHEV)": "PHEV",
            }
        )
        is_phev = car_type.str.contains("PHEV", na=False)
        electric_range = pd.to_numeric(df["electric_range"], errors="coerce")
        range_km = electric_range.fillna(0) * 1.60934

        base_id = df["vin_1_10"].fillna(df["make"].astype(str))
        feature_df = _build_feature_frame(
            base_id=base_id,
            brand=make,
            car_type=car_type,
            age_years=age_years,
            range_km=range_km,
            is_phev=is_phev,
        )
        return feature_df.dropna()


@dataclass
class FuelEconomyEVSource(DataSource):
    """EPA fuel economy dataset filtered to EVs and PHEVs."""

    url: str = "https://www.fueleconomy.gov/feg/epadata/vehicles.csv"
    cache_path: Path = DATA_CACHE_DIR / "epa_vehicles.csv"
    max_rows: int = 5000

    def load(self) -> pd.DataFrame:
        df = _read_csv_cached(self.url, self.cache_path)
        df = df.rename(columns=str.strip)

        fuel_type = df.get("fuelType1", pd.Series("", index=df.index, dtype=str)).astype(str)
        fuel_type2 = df.get("fuelType2", pd.Series("", index=df.index, dtype=str)).astype(str)
        atv_type = df.get("atvType", pd.Series("", index=df.index, dtype=str)).astype(str)
        ev_mask = (
            fuel_type.str.contains("Electricity", na=False)
            | fuel_type2.str.contains("Electricity", na=False)
            | atv_type.str.contains("EV", na=False)
        )
        df = df[ev_mask].copy()
        if df.empty:
            raise ValueError("EPA dataset filter produced no EV records.")

        if self.max_rows and len(df) > self.max_rows:
            df = df.sample(n=self.max_rows, random_state=42)

        current_year = datetime.now().year
        age_years = current_year - pd.to_numeric(df["year"], errors="coerce")
        brand = df["make"].astype(str).str.title()
        car_type = df.get("VClass", pd.Series("Unknown", index=df.index)).astype(str)
        range_miles = pd.to_numeric(df.get("range", 0), errors="coerce")
        range_alt = pd.to_numeric(df.get("rangeA", 0), errors="coerce")
        range_miles = range_miles.fillna(range_alt)
        range_km = range_miles.fillna(0) * 1.60934
        is_phev = fuel_type2.str.strip().ne("")

        base_id = df.get("id", pd.Series(index=df.index, data=np.arange(len(df)))).astype(str)
        feature_df = _build_feature_frame(
            base_id=base_id,
            brand=brand,
            car_type=car_type,
            age_years=age_years,
            range_km=range_km,
            is_phev=is_phev,
        )
        return feature_df.dropna()


@dataclass
class UserApprovedSubmissionsSource(DataSource):
    """Approved user submissions for model training."""

    path: Path = USER_SUBMISSIONS_APPROVED_PATH

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame()
        df = pd.read_csv(self.path)
        if df.empty:
            return pd.DataFrame()

        required = [
            "submission_id",
            "brand",
            "car_type",
            "model_year",
            "odometer_km",
            "range_original_km",
            "range_current_km",
        ]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Approved submissions missing columns: {missing}")

        current_year = datetime.now().year
        age_years = current_year - pd.to_numeric(df["model_year"], errors="coerce")
        odometer_km = pd.to_numeric(df["odometer_km"], errors="coerce")
        range_original = pd.to_numeric(df["range_original_km"], errors="coerce")
        range_current = pd.to_numeric(df["range_current_km"], errors="coerce")

        denom = range_original.replace(0, np.nan)
        capacity_ratio = (range_current / denom).clip(lower=0, upper=0.99)
        eol_km = 0.3 * odometer_km / (1 - capacity_ratio)
        eol_km = eol_km.replace([np.inf, -np.inf], np.nan)
        eol_km = np.maximum(eol_km, odometer_km)
        eol_km = eol_km.fillna(odometer_km)

        car_type = df["car_type"].astype(str)
        is_phev = car_type.str.contains("PHEV|Plug", case=False, na=False)
        base_id = df["submission_id"].astype(str)
        usage_features = _derive_usage_features(base_id, is_phev)

        feature_df = pd.DataFrame(
            {
                "brand": df["brand"].astype(str),
                "car_type": car_type,
                "age_years": age_years.astype(float),
                "km": odometer_km.astype(float),
                "fast_share": usage_features["fast_share"],
                "avg_soc": usage_features["avg_soc"],
                "avg_temp_c": usage_features["avg_temp_c"],
                "eol_km": eol_km.astype(float),
            }
        )
        return feature_df.dropna()


def default_sources() -> List[DataSource]:
    """Primary data sources used by the application."""
    return [
        WashingtonEVPopulationSource(),
        FuelEconomyEVSource(),
        UserApprovedSubmissionsSource(),
    ]

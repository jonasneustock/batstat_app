"""Unit tests for data sources using unittest framework."""

import io
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

import pandas as pd

from batstat_app.app.data import sources as data_sources
from batstat_app.app.data.sources import (
    DataSource,
    ExampleSource,
    FuelEconomyEVSource,
    OpenEVDataDatasetSource,
    OpenEVDataSpecsSource,
    UserApprovedSubmissionsSource,
    WashingtonEVPopulationSource,
    default_sources,
    load_all_sources,
)


class TestExampleSource(unittest.TestCase):
    """Tests for the ExampleSource class."""

    def test_load_returns_dataframe(self) -> None:
        """ExampleSource.load should return a DataFrame with expected columns and size."""
        source = ExampleSource(n_samples=100, random_state=123)
        df = source.load()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 100)
        required_cols = [
            "brand",
            "car_type",
            "age_years",
            "km",
            "fast_share",
            "avg_soc",
            "avg_temp_c",
            "eol_km",
        ]
        for col in required_cols:
            self.assertIn(col, df.columns)

    def test_load_reproducible(self) -> None:
        """ExampleSource should produce deterministic output for the same seed."""
        s1 = ExampleSource(n_samples=10, random_state=5)
        s2 = ExampleSource(n_samples=10, random_state=5)
        df1 = s1.load()
        df2 = s2.load()
        pd.testing.assert_frame_equal(df1, df2)


class TestLoadAllSources(unittest.TestCase):
    """Tests for the load_all_sources function."""

    def test_concatenates(self) -> None:
        """load_all_sources should concatenate multiple sources into one DataFrame."""
        s1 = ExampleSource(n_samples=5, random_state=0)
        s2 = ExampleSource(n_samples=3, random_state=1)
        df = load_all_sources([s1, s2])
        self.assertIsInstance(df, pd.DataFrame)
        self.assertEqual(len(df), 8)
        self.assertIn("brand", df.columns)
        self.assertEqual(df.isnull().sum().sum(), 0)

    def test_empty(self) -> None:
        """Passing an empty list should return an empty DataFrame."""
        df = load_all_sources([])
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.empty)


class TestReadCsvCached(unittest.TestCase):
    """Tests for the CSV caching helper."""

    def test_cache_hit_skips_network(self) -> None:
        """Cache hit should avoid network calls."""
        with TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "cache.csv"
            cache_path.write_text("a,b\n1,2\n", encoding="utf-8")
            with mock.patch("batstat_app.app.data.sources.urllib.request.urlopen") as urlopen:
                df = data_sources._read_csv_cached("http://example.com", cache_path)
                self.assertEqual(len(df), 1)
                urlopen.assert_not_called()

    def test_download_and_cache(self) -> None:
        """Download should populate cache when missing."""
        class MockResponse(io.BytesIO):
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                self.close()

        csv_bytes = b"a,b\n1,2\n"
        with TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "cache.csv"
            with mock.patch(
                "batstat_app.app.data.sources.urllib.request.urlopen",
                return_value=MockResponse(csv_bytes),
            ) as urlopen:
                df = data_sources._read_csv_cached("http://example.com", cache_path)
                self.assertTrue(cache_path.exists())
                self.assertEqual(len(df), 1)
                urlopen.assert_called_once()


class TestRealSources(unittest.TestCase):
    """Tests for the real-world data source adapters."""

    @mock.patch("batstat_app.app.data.sources._read_csv_cached")
    def test_washington_ev_source(self, mock_read_csv) -> None:
        """Washington EV source should normalize into model-ready features."""
        mock_read_csv.return_value = pd.DataFrame(
            {
                "vin_1_10": ["AAA123", "BBB456"],
                "model_year": [2020, 2018],
                "make": ["TESLA", "NISSAN"],
                "model": ["Model 3", "Leaf"],
                "ev_type": [
                    "Battery Electric Vehicle (BEV)",
                    "Plug-in Hybrid Electric Vehicle (PHEV)",
                ],
                "electric_range": [250, 30],
            }
        )
        source = WashingtonEVPopulationSource(limit=2)
        df = source.load()
        required_cols = [
            "brand",
            "car_type",
            "age_years",
            "km",
            "fast_share",
            "avg_soc",
            "avg_temp_c",
            "eol_km",
        ]
        for col in required_cols:
            self.assertIn(col, df.columns)
        self.assertTrue((df["eol_km"] >= df["km"]).all())
        self.assertTrue(df["car_type"].isin(["Model 3", "Leaf"]).all())

    @mock.patch("batstat_app.app.data.sources._read_csv_cached")
    def test_fuel_economy_source_filters_ev(self, mock_read_csv) -> None:
        """EPA source should filter to EV/PHEV rows only."""
        mock_read_csv.return_value = pd.DataFrame(
            {
                "fuelType1": ["Electricity", "Gasoline"],
                "fuelType2": ["", ""],
                "atvType": ["EV", ""],
                "model": ["Model S", "Focus"],
                "VClass": ["Hatchback", "Sedan"],
                "year": [2020, 2019],
                "make": ["Tesla", "Ford"],
                "range": [250, 0],
                "rangeA": [0, 0],
                "id": [1, 2],
            }
        )
        source = FuelEconomyEVSource(max_rows=0)
        df = source.load()
        self.assertEqual(len(df), 1)
        self.assertIn("brand", df.columns)


class TestDefaultSources(unittest.TestCase):
    """Tests for the default source list."""

    def test_default_sources_list(self) -> None:
        """default_sources should return instantiated adapters."""
        sources = default_sources()
        self.assertEqual(len(sources), 5)
        self.assertIsInstance(sources[0], WashingtonEVPopulationSource)
        self.assertIsInstance(sources[1], FuelEconomyEVSource)
        self.assertIsInstance(sources[2], OpenEVDataDatasetSource)
        self.assertIsInstance(sources[3], OpenEVDataSpecsSource)
        self.assertIsInstance(sources[4], UserApprovedSubmissionsSource)


class TestUserApprovedSubmissionsSource(unittest.TestCase):
    """Tests for approved user submissions source."""

    def test_load_empty_when_missing(self) -> None:
        """Missing submissions file should return empty DataFrame."""
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "missing.csv"
            source = UserApprovedSubmissionsSource(path=path)
            df = source.load()
            self.assertTrue(df.empty)

    def test_load_from_file(self) -> None:
        """Approved submissions should normalize into model-ready features."""
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "approved.csv"
            path.write_text(
                "submission_id,brand,car_type,model_year,odometer_km,range_original_km,range_current_km\n"
                "abc123,Tesla,Model 3,2020,35000,500,430\n",
                encoding="utf-8",
            )
            source = UserApprovedSubmissionsSource(path=path)
            df = source.load()
            self.assertEqual(len(df), 1)
            self.assertIn("eol_km", df.columns)
            self.assertGreaterEqual(df.loc[0, "eol_km"], df.loc[0, "km"])


class TestOpenEVDataDatasetSource(unittest.TestCase):
    """Tests for the Open EV Data dataset source."""

    @mock.patch("batstat_app.app.data.sources._read_csv_cached")
    def test_open_ev_dataset_load(self, mock_read_csv) -> None:
        mock_read_csv.return_value = pd.DataFrame(
            {
                "unique_code": ["abc"],
                "make_name": ["Tesla"],
                "model_name": ["Model 3"],
                "year": [2021],
                "trim_name": ["Long Range"],
                "variant_name": [""],
                "range_wltp_km": [560],
                "range_epa_km": [0],
                "battery_capacity_net_kwh": [75],
                "vehicle_type": ["PassengerCar"],
            }
        )
        source = OpenEVDataDatasetSource(max_rows=0)
        df = source.load()
        self.assertEqual(len(df), 1)
        self.assertIn("car_type", df.columns)
        self.assertTrue(df.loc[0, "car_type"].startswith("Model 3"))


class TestOpenEVDataSpecsSource(unittest.TestCase):
    """Tests for the Open EV Data specs JSON source."""

    @mock.patch("batstat_app.app.data.sources._read_json_cached")
    def test_open_ev_specs_load(self, mock_read_json) -> None:
        mock_read_json.return_value = {
            "data": [
                {
                    "id": "xyz",
                    "brand": "Nissan",
                    "model": "Leaf",
                    "release_year": 2020,
                    "usable_battery_size": 40.0,
                    "energy_consumption": {"average_consumption": 17.5},
                }
            ]
        }
        source = OpenEVDataSpecsSource(max_rows=0)
        df = source.load()
        self.assertEqual(len(df), 1)
        self.assertEqual(df.loc[0, "car_type"], "Leaf")


class TestLoadAllSourcesErrors(unittest.TestCase):
    """Tests for error handling in load_all_sources."""

    def test_non_dataframe_raises(self) -> None:
        """load_all_sources should reject sources returning non-DataFrame."""
        class BadSource(DataSource):
            def load(self):
                return ["not", "a", "dataframe"]

        with self.assertRaises(TypeError):
            load_all_sources([BadSource()])


if __name__ == "__main__":
    unittest.main()

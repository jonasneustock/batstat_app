"""Unit tests for data sources using unittest framework."""

import unittest
import pandas as pd

from batstat_app.app.data.sources import ExampleSource, load_all_sources


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


if __name__ == "__main__":
    unittest.main()
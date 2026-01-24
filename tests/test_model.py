"""Unit tests for model training and prediction using unittest."""

import unittest
import pandas as pd

from batstat_app.app.data.sources import ExampleSource
from batstat_app.app.ml.model import train_model, TrainedModel


class TestTrainModel(unittest.TestCase):
    """Tests for the train_model function."""

    def test_returns_trained_model(self) -> None:
        """train_model should return a TrainedModel with sensible attributes."""
        df = ExampleSource(n_samples=100).load()
        model = train_model(df, random_state=42)
        self.assertIsInstance(model, TrainedModel)
        self.assertTrue(hasattr(model, "pipeline"))
        self.assertTrue(hasattr(model, "q_low"))
        self.assertTrue(hasattr(model, "q_high"))
        self.assertLess(model.q_low, model.q_high)

    def test_requires_columns(self) -> None:
        """train_model should raise a ValueError if required columns are missing."""
        df = ExampleSource(n_samples=5).load().copy()
        df = df.drop(columns=["eol_km"])
        with self.assertRaises(ValueError):
            train_model(df)


class TestPredictInterval(unittest.TestCase):
    """Tests for the predict_interval method of TrainedModel."""

    def setUp(self) -> None:
        self.df = ExampleSource(n_samples=50).load()
        self.model = train_model(self.df, random_state=0)

    def test_predict_interval_shapes(self) -> None:
        """predict_interval should return arrays matching the input shape."""
        X_single = self.df.iloc[[0]][
            [
                "brand",
                "car_type",
                "age_years",
                "km",
                "fast_share",
                "avg_soc",
                "avg_temp_c",
            ]
        ]
        y_pred, lower, upper = self.model.predict_interval(X_single)
        self.assertEqual(y_pred.shape, (1,))
        self.assertEqual(lower.shape, (1,))
        self.assertEqual(upper.shape, (1,))
        self.assertLessEqual(lower[0], upper[0])
        # multiple rows
        X_multi = self.df.head(5)[
            [
                "brand",
                "car_type",
                "age_years",
                "km",
                "fast_share",
                "avg_soc",
                "avg_temp_c",
            ]
        ]
        yp2, l2, u2 = self.model.predict_interval(X_multi)
        self.assertEqual(yp2.shape[0], 5)
        self.assertTrue((l2 <= u2).all())

    def test_predict_interval_monotonic(self) -> None:
        """Predicted means should lie within the prediction interval bounds."""
        X = self.df[
            [
                "brand",
                "car_type",
                "age_years",
                "km",
                "fast_share",
                "avg_soc",
                "avg_temp_c",
            ]
        ]
        y_pred, lower, upper = self.model.predict_interval(X)
        # y_pred should be between lower and upper for all rows
        self.assertTrue(((y_pred >= (lower - 1e-6)) & (y_pred <= (upper + 1e-6))).all())


if __name__ == "__main__":
    unittest.main()
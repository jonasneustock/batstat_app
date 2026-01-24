"""Unit tests for utility functions using unittest."""

import unittest

from batstat_app.app.main import format_number, format_value


class TestFormatNumber(unittest.TestCase):
    """Tests for the format_number function."""

    def test_basic_formatting(self) -> None:
        """format_number should insert non-breaking spaces and respect decimals."""
        self.assertEqual(format_number(1000), "1 000")
        self.assertEqual(format_number(1234.567, decimals=2), "1 234.57")

    def test_zero_decimals(self) -> None:
        """format_number should handle zero and rounding behaviour."""
        self.assertEqual(format_number(0), "0")
        self.assertEqual(format_number(5.5), "6")  # rounds to nearest integer


class TestFormatValue(unittest.TestCase):
    """Tests for the format_value function."""

    def test_numbers(self) -> None:
        """format_value should format numeric values."""
        self.assertEqual(format_value(1000), "1 000")
        self.assertEqual(format_value(1234.56), "1 235")

    def test_non_number(self) -> None:
        """format_value should return the original value for non-numeric input."""
        self.assertEqual(format_value("abc"), "abc")
        class Dummy:
            pass
        dummy = Dummy()
        # returns the same object or equal value
        self.assertTrue(format_value(dummy) is dummy or format_value(dummy) == dummy)


if __name__ == "__main__":
    unittest.main()
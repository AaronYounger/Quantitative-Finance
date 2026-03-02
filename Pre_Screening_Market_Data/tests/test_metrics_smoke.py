from __future__ import annotations

"""Smoke tests that verify all registered metrics execute on a valid sample frame."""

import unittest

import pandas as pd

from metrics.registry import get_metrics


class MetricsSmokeTests(unittest.TestCase):
    """Basic execution checks for metric registry integrations."""

    def setUp(self):
        """Create deterministic synthetic OHLC data for metric computations."""
        index = pd.date_range("2024-01-01", periods=120, freq="D")
        base = pd.Series(range(100, 220), index=index, dtype="float64")
        self.df = pd.DataFrame(
            {
                "Open": base * 0.99,
                "High": base * 1.01,
                "Low": base * 0.98,
                "Close": base,
            },
            index=index,
        )

    def test_all_metrics_compute(self):
        """Ensure each metric returns at least one scalar output."""
        for metric in get_metrics():
            result = metric.compute(self.df.copy())
            self.assertTrue(isinstance(result.values, dict))
            self.assertGreater(len(result.values), 0)


if __name__ == "__main__":
    unittest.main()

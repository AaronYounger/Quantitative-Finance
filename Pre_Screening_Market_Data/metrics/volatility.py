from __future__ import annotations

"""Volatility metric implementation for daily and annualized estimates."""

import pandas as pd

from metrics.base import Metric, MetricResult
from models import normalize_price_frame, validate_price_columns


class VolatilityMetric(Metric):
    """Compute standard deviation-based volatility from close returns."""

    metric_id = "volatility"
    label = "Volatility"

    def compute(self, df: pd.DataFrame) -> MetricResult:
        """Return daily and annualized volatility values."""
        frame = normalize_price_frame(df)
        validate_price_columns(frame, required=("Close",))

        daily = frame["Close"].pct_change().dropna()
        if daily.empty:
            raise ValueError("Not enough price data for volatility.")

        daily_vol = daily.std()
        annual_vol = daily_vol * (252**0.5)
        return MetricResult(
            values={
                "Daily Volatility": float(daily_vol),
                "Annualized Volatility": float(annual_vol),
            }
        )

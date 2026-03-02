from __future__ import annotations

"""Drawdown metric implementation for peak-to-trough risk statistics."""

import pandas as pd

from metrics.base import Metric, MetricResult
from models import normalize_price_frame, validate_price_columns


class DrawdownMetric(Metric):
    """Compute max/current drawdown and drawdown volatility from close prices."""

    metric_id = "drawdown"
    label = "Drawdown"

    def compute(self, df: pd.DataFrame) -> MetricResult:
        """Return drawdown statistics as percentage values."""
        frame = normalize_price_frame(df)
        validate_price_columns(frame, required=("Close",))

        returns = frame["Close"].pct_change().fillna(0.0)
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max

        return MetricResult(
            values={
                "Max Drawdown": float(drawdown.min() * 100.0),
                "Current Drawdown": float(drawdown.iloc[-1] * 100.0),
                "Drawdown Volatility": float(drawdown.std() * 100.0),
            }
        )

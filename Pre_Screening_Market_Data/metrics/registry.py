from __future__ import annotations

"""Metric registry helpers used by the app to discover available metrics."""

from metrics.base import Metric
from metrics.drawdown import DrawdownMetric
from metrics.returns import ReturnsMetric
from metrics.trend import TrendMetric
from metrics.volatility import VolatilityMetric


def get_metrics() -> list[Metric]:
    """Return metric instances in the display order used by the UI."""
    return [
        ReturnsMetric(),
        TrendMetric(),
        VolatilityMetric(),
        DrawdownMetric(),
    ]


def metrics_by_id() -> dict[str, Metric]:
    """Return a lookup map from metric id to metric instance."""
    return {m.metric_id: m for m in get_metrics()}

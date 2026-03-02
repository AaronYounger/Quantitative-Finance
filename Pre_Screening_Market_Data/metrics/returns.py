from __future__ import annotations

"""Returns metric implementation and supporting chart output."""

import matplotlib.pyplot as plt
import pandas as pd

from metrics.base import Metric, MetricResult
from models import normalize_price_frame, validate_price_columns


class ReturnsMetric(Metric):
    """Compute simple return-based statistics from close prices."""

    metric_id = "returns"
    label = "Returns"

    def compute(self, df: pd.DataFrame) -> MetricResult:
        """Return total/average returns and a daily return time-series chart."""
        frame = normalize_price_frame(df)
        validate_price_columns(frame, required=("Close",))

        close = frame["Close"].dropna()
        if len(close) < 2:
            raise ValueError("Not enough price data for returns.")

        daily = close.pct_change().dropna()
        total_ret = (close.iloc[-1] / close.iloc[0] - 1.0) * 100.0

        # Keep this chart compact in Streamlit so it doesn't dominate the page.
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.plot(daily.index, daily.values)
        ax.set_title("Daily Return")
        ax.set_xlabel("Date")
        ax.set_ylabel("Return")
        fig.tight_layout()

        return MetricResult(
            values={
                "Total Return": float(total_ret),
                "Average Daily Return": float(daily.mean()),
            },
            chart=fig,
        )

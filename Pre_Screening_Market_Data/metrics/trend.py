from __future__ import annotations

"""Trend metric implementation using close price and SMA regime logic."""

import matplotlib.pyplot as plt
import pandas as pd

from metrics.base import Metric, MetricResult
from models import normalize_price_frame, validate_price_columns


class TrendMetric(Metric):
    """Classify trend regime and plot close price only."""

    metric_id = "trend"
    label = "Trend"

    def compute(self, df: pd.DataFrame) -> MetricResult:
        """Return trend regime label and a close-price chart."""
        frame = normalize_price_frame(df)
        validate_price_columns(frame, required=("Close",))

        close = frame["Close"]
        sma20 = close.rolling(window=20).mean()
        sma50 = close.rolling(window=50).mean()

        if close.empty:
            raise ValueError("Not enough price data for trend.")

        last_close = close.iloc[-1]
        last_sma20 = sma20.iloc[-1]
        last_sma50 = sma50.iloc[-1]

        if pd.isna(last_sma20) or pd.isna(last_sma50):
            regime = "Not enough data for SMA regime"
        else:
            price_above_sma20 = last_close > last_sma20
            price_above_sma50 = last_close > last_sma50
            sma20_above_sma50 = last_sma20 > last_sma50

            if price_above_sma50 and sma20_above_sma50 and price_above_sma20:
                regime = "Uptrend"
            elif (not price_above_sma50) and (not sma20_above_sma50):
                regime = "Downtrend"
            else:
                regime = "Mixed"

        fig, ax = plt.subplots(figsize=(8, 3.5))
        ax.plot(frame.index, close, label="Close")
        ax.set_title("Close Price")
        ax.set_xlabel("Date")
        ax.set_ylabel("Price")
        ax.legend()
        fig.tight_layout()

        return MetricResult(values={"Trend Regime": regime}, chart=fig)

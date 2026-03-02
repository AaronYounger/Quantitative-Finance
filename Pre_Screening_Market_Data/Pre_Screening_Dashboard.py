from __future__ import annotations

"""Backward-compatible API layer around current provider/chart/metrics modules."""

import matplotlib.pyplot as plt
import pandas as pd

from charts import CandleChart
from data_provider import PreScreenDataProvider
from metrics.drawdown import DrawdownMetric
from metrics.returns import ReturnsMetric
from metrics.trend import TrendMetric
from metrics.volatility import VolatilityMetric
from models import normalize_price_frame, validate_price_columns


class PreScreen(PreScreenDataProvider):
    """Backward-compatible alias for the previous provider class."""


class candle(CandleChart):
    """Backward-compatible alias for the legacy candle chart class."""

    def cplot(self):
        """Legacy method name that forwards to the current figure() method."""
        return self.figure()


class Returns:
    """Legacy returns helper that wraps current metric computations."""

    def __init__(self, df: pd.DataFrame):
        """Normalize and validate input frame with close prices available."""
        self.df = normalize_price_frame(df)
        validate_price_columns(self.df, required=("Close",))

    def daily_return_series(self) -> pd.Series:
        """Return daily percentage return series from close prices."""
        return self.df["Close"].pct_change().dropna()

    def daily_return(self):
        """Return a matplotlib figure for the daily return line plot."""
        daily = self.daily_return_series()
        fig, ax = plt.subplots()
        ax.plot(daily.index, daily.values)
        ax.set_xlabel("Date")
        ax.set_ylabel("Return")
        return fig

    def total_return(self) -> float:
        """Return total return percent over the full series."""
        result = ReturnsMetric().compute(self.df)
        return float(result.values["Total Return"])


class TrendStructure:
    """Legacy trend helper for SMA and trend regime access."""

    def __init__(self, df: pd.DataFrame):
        """Normalize and validate input frame with close prices available."""
        self.df = normalize_price_frame(df)
        validate_price_columns(self.df, required=("Close",))

    def sma20(self) -> pd.Series:
        """Compute and return the 20-day simple moving average."""
        self.df["SMA_20"] = self.df["Close"].rolling(window=20).mean()
        return self.df["SMA_20"]

    def sma50(self) -> pd.Series:
        """Compute and return the 50-day simple moving average."""
        self.df["SMA_50"] = self.df["Close"].rolling(window=50).mean()
        return self.df["SMA_50"]

    def trend_regime(self) -> str:
        """Return regime label from the current trend metric implementation."""
        result = TrendMetric().compute(self.df)
        return str(result.values["Trend Regime"])


class Volatility:
    """Legacy volatility helper that proxies to VolatilityMetric."""

    def __init__(self, df: pd.DataFrame):
        """Normalize and validate input frame with close prices available."""
        self.df = normalize_price_frame(df)
        validate_price_columns(self.df, required=("Close",))

    def daily_volatility(self) -> float:
        """Return daily volatility estimate."""
        result = VolatilityMetric().compute(self.df)
        return float(result.values["Daily Volatility"])

    def annual_volatility(self) -> float:
        """Return annualized volatility estimate using 252 trading days."""
        result = VolatilityMetric().compute(self.df)
        return float(result.values["Annualized Volatility"])


class Drawdown:
    """Legacy drawdown helper backed by DrawdownMetric."""

    def __init__(self, df: pd.DataFrame):
        """Normalize and validate input frame with close prices available."""
        self.df = normalize_price_frame(df)
        validate_price_columns(self.df, required=("Close",))

    def returns(self) -> pd.Series:
        """Return daily close-to-close returns with initial zero fill."""
        return self.df["Close"].pct_change().fillna(0.0)

    def cum_returns(self) -> pd.Series:
        """Return cumulative compounded return path."""
        return (1 + self.returns()).cumprod()

    def max_drawdown(self) -> float:
        """Return max drawdown as a decimal (legacy behavior)."""
        summary = DrawdownMetric().compute(self.df).values
        return float(summary["Max Drawdown"] / 100.0)

    def current_drawdown(self) -> float:
        """Return current drawdown as a decimal (legacy behavior)."""
        summary = DrawdownMetric().compute(self.df).values
        return float(summary["Current Drawdown"] / 100.0)

    def drawdown_volatility(self) -> float:
        """Return drawdown volatility as a decimal (legacy behavior)."""
        summary = DrawdownMetric().compute(self.df).values
        return float(summary["Drawdown Volatility"] / 100.0)

    def drawdown_summary(self) -> dict:
        """Return drawdown summary with percentage-valued fields."""
        summary = DrawdownMetric().compute(self.df).values
        return {
            "max_drawdown_pct": float(summary["Max Drawdown"]),
            "current_drawdown_pct": float(summary["Current Drawdown"]),
            "drawdown_volatility": float(summary["Drawdown Volatility"]),
        }

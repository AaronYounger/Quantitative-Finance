from __future__ import annotations

"""Chart helpers for market data visualizations."""

import pandas as pd
import plotly.graph_objects as go

from models import normalize_price_frame, validate_price_columns


class CandleChart:
    """Build a Plotly candlestick figure from a normalized OHLC frame."""

    def __init__(self, df: pd.DataFrame):
        """Normalize input data and validate required price columns."""
        self.df = normalize_price_frame(df)
        validate_price_columns(self.df, required=("Open", "High", "Low", "Close"))

    def figure(self) -> go.Figure:
        """Return the candlestick figure for rendering in Streamlit or notebooks."""
        idx = pd.to_datetime(self.df.index, errors="coerce")
        has_time = bool(
            (idx.hour != 0).any()
            or (idx.minute != 0).any()
            or (idx.second != 0).any()
            or (idx.microsecond != 0).any()
        )
        fmt = "%Y-%m-%d %H:%M" if has_time else "%Y-%m-%d"
        x_vals = idx.strftime(fmt)

        fig = go.Figure(
            data=[
                go.Candlestick(
                    x=x_vals,
                    open=self.df["Open"],
                    high=self.df["High"],
                    low=self.df["Low"],
                    close=self.df["Close"],
                )
            ]
        )
        # Use a category axis so non-trading dates do not render as empty gaps.
        fig.update_xaxes(type="category")
        return fig

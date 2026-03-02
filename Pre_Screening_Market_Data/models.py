from __future__ import annotations

"""Shared data-frame normalization and validation helpers for price data."""

import pandas as pd


REQUIRED_PRICE_COLUMNS = ("Open", "High", "Low", "Close")
NUMERIC_PRICE_COLUMNS = ("Open", "High", "Low", "Close", "Volume")


def normalize_price_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize yfinance-style frames to a clean, date-indexed OHLCV shape."""
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        if out.columns.nlevels >= 2:
            # yfinance sometimes returns two-level columns where level 0 is OHLCV.
            out.columns = [col[0] for col in out.columns]
        else:
            out.columns = [str(c) for c in out.columns]
    else:
        out.columns = [col[0] if isinstance(col, tuple) else col for col in out.columns]

    out.columns = [str(c).strip().title() for c in out.columns]
    out.index.name = out.index.name or "Date"

    # Drop non-date index rows (legacy cache files may include a second header row).
    out.index = pd.to_datetime(out.index, errors="coerce")
    out = out[~out.index.isna()]

    # Ensure numeric columns are numeric across all metrics.
    for col in NUMERIC_PRICE_COLUMNS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    # Remove rows that still have no usable OHLC values.
    present_ohlc = [c for c in REQUIRED_PRICE_COLUMNS if c in out.columns]
    if present_ohlc:
        out = out.dropna(subset=present_ohlc, how="all")

    # Enforce NaN-free imported price rows across core price fields.
    present_price_fields = [c for c in NUMERIC_PRICE_COLUMNS if c in out.columns]
    if present_price_fields:
        out = out.dropna(subset=present_price_fields, how="any")

    out = out.sort_index()
    return out


def validate_price_columns(df: pd.DataFrame, required: tuple[str, ...] = REQUIRED_PRICE_COLUMNS) -> None:
    """Raise a helpful error when expected columns are missing."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available columns: {list(df.columns)}")

from __future__ import annotations

"""Data access layer for S&P 500 universe and cached historical price downloads."""

import os
import time
from urllib.request import Request, urlopen

import pandas as pd
import yfinance as yf

from models import normalize_price_frame


class PreScreenDataProvider:
    """Fetch and cache universe + price data used by the Streamlit application."""

    WIKI_SP500_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    ALLOWED_PERIODS = ["6mo", "1y", "5y"]
    ALLOWED_INTERVALS = ["1d"]

    def __init__(
        self,
        universe_cache_path: str = "sp500_universe.csv",
        price_cache_dir: str = "cache_prices",
        default_period: str = "1y",
        default_interval: str = "1d",
        auto_adjust: bool = True,
    ):
        """Configure cache paths, download defaults, and in-memory state."""
        self.universe_cache_path = universe_cache_path
        self.price_cache_dir = price_cache_dir
        self.default_period = default_period
        self.default_interval = default_interval
        self.auto_adjust = auto_adjust

        self.universe: pd.DataFrame | None = None
        self.last_ticker: str | None = None
        self.last_prices: pd.DataFrame | None = None

        os.makedirs(self.price_cache_dir, exist_ok=True)

    def _write_normalized_cache(self, df: pd.DataFrame, cache_path: str) -> None:
        """Persist a normalized price frame with a Date column."""
        out = df.copy()
        out.index.name = "Date"
        out.reset_index().to_csv(cache_path, index=False)

    def get_sp500_universe(self) -> pd.DataFrame:
        """Download and normalize the S&P 500 constituents table from Wikipedia."""
        try:
            req = Request(self.WIKI_SP500_URL, headers={"User-Agent": "Mozilla/5.0"})
            html = urlopen(req, timeout=15).read()
            tables = pd.read_html(html)
        except Exception as exc:
            raise RuntimeError(f"Unable to download S&P 500 universe from Wikipedia: {exc}") from exc

        if not tables:
            raise RuntimeError("Wikipedia response did not include any tables.")

        df = tables[0].copy()
        required = {"Symbol", "Security", "GICS Sector", "GICS Sub-Industry"}
        if not required.issubset(df.columns):
            raise RuntimeError(
                "Wikipedia S&P 500 table format changed. Missing columns: "
                f"{sorted(required - set(df.columns))}"
            )

        out = df.rename(
            columns={
                "Symbol": "ticker",
                "Security": "name",
                "GICS Sector": "sector",
                "GICS Sub-Industry": "sub_industry",
            }
        )[["ticker", "name", "sector", "sub_industry"]]

        out["ticker"] = out["ticker"].astype(str).str.replace(".", "-", regex=False).str.strip().str.upper()
        out["name"] = out["name"].astype(str).str.strip()
        return out

    def load_sp500_universe(self, refresh: bool = False, max_age_days: int = 7) -> pd.DataFrame:
        """Load universe from cache when fresh, otherwise refresh from source."""
        cache_exists = os.path.exists(self.universe_cache_path)

        if cache_exists and not refresh:
            age_days = (time.time() - os.path.getmtime(self.universe_cache_path)) / 86400
            if age_days < max_age_days:
                df = pd.read_csv(self.universe_cache_path)
                df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
                df["name"] = df["name"].astype(str).str.strip()
                self.universe = df
                return df

        try:
            df = self.get_sp500_universe()
            df.to_csv(self.universe_cache_path, index=False)
            self.universe = df
            return df
        except Exception:
            if cache_exists:
                df = pd.read_csv(self.universe_cache_path)
                df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
                df["name"] = df["name"].astype(str).str.strip()
                self.universe = df
                return df
            raise

    def search_universe(self, query: str, limit: int = 10) -> pd.DataFrame:
        """Search ticker/name in the loaded universe and return a limited subset."""
        if self.universe is None:
            self.load_sp500_universe(refresh=False)

        df = self.universe
        q = (query or "").strip().lower()
        if not q:
            return df.head(limit)

        sym = df["ticker"].str.lower()
        name = df["name"].str.lower()
        mask = sym.str.contains(q, regex=False) | name.str.contains(q, regex=False)
        return df.loc[mask].head(limit)

    @staticmethod
    def format_suggestions(df: pd.DataFrame) -> list[str]:
        """Format rows as 'Company (TICKER)' strings for UI pickers."""
        return [f"{row['name']} ({row['ticker']})" for _, row in df.iterrows()]

    @staticmethod
    def parse_ticker(selection: str) -> str:
        """Extract a normalized ticker from raw text or 'Name (TICKER)'."""
        if not selection:
            return ""
        s = selection.strip()
        if "(" in s and ")" in s:
            s = s.split("(")[-1].split(")")[0].strip()
        return s.upper().replace(".", "-")

    def _price_cache_path(self, ticker: str, period: str, interval: str, auto_adjust: bool) -> str:
        """Resolve cache filename for a ticker/period/interval combination."""
        adj_tag = "adj" if auto_adjust else "raw"
        safe = f"{ticker}_{period}_{interval}_{adj_tag}".replace("/", "-")
        return os.path.join(self.price_cache_dir, f"{safe}.csv")

    def get_prices(
        self,
        ticker: str,
        period: str,
        interval: str,
        refresh: bool = False,
        auto_adjust: bool | None = None,
    ) -> pd.DataFrame:
        """Get normalized OHLCV prices from cache or yfinance and update cache."""
        t = self.parse_ticker(ticker)
        use_adjusted = self.auto_adjust if auto_adjust is None else bool(auto_adjust)

        if period not in self.ALLOWED_PERIODS:
            raise ValueError(f"Period '{period}' not allowed. Allowed: {self.ALLOWED_PERIODS}")
        if interval not in self.ALLOWED_INTERVALS:
            raise ValueError(f"Interval '{interval}' not allowed. Allowed: {self.ALLOWED_INTERVALS}")
        if not t:
            raise ValueError("Ticker is empty.")

        cache_path = self._price_cache_path(t, period, interval, use_adjusted)
        if (not refresh) and os.path.exists(cache_path):
            cached = pd.read_csv(cache_path, parse_dates=["Date"]).set_index("Date")
            cached = normalize_price_frame(cached)
            # Self-heal cache files so malformed rows do not persist across runs.
            self._write_normalized_cache(cached, cache_path)
            self.last_ticker = t
            self.last_prices = cached
            return cached

        try:
            df = yf.download(t, period=period, interval=interval, auto_adjust=use_adjusted, progress=False)
        except Exception as exc:
            raise RuntimeError(f"Unable to download prices for {t}: {exc}") from exc

        if df is None or df.empty:
            raise ValueError(f"No data returned for {t} (period={period}, interval={interval})")

        out = normalize_price_frame(df)
        self._write_normalized_cache(out, cache_path)
        self.last_ticker = t
        self.last_prices = out
        return out

    def heal_price_cache_file(self, filename: str) -> tuple[str, bool]:
        """Re-normalize one cached CSV file and report whether row count changed."""
        path = os.path.join(self.price_cache_dir, filename)
        raw = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
        normalized = normalize_price_frame(raw)
        before_rows = len(raw)
        after_rows = len(normalized)
        self._write_normalized_cache(normalized, path)
        changed = before_rows != after_rows
        return path, changed

    def heal_all_price_cache(self) -> dict:
        """Normalize all cached CSV files and return a summary report."""
        if not os.path.exists(self.price_cache_dir):
            return {"total_files": 0, "changed_files": 0, "paths": []}

        changed_paths: list[str] = []
        total = 0
        for name in sorted(os.listdir(self.price_cache_dir)):
            if not name.lower().endswith(".csv"):
                continue
            total += 1
            path, changed = self.heal_price_cache_file(name)
            if changed:
                changed_paths.append(path)

        return {"total_files": total, "changed_files": len(changed_paths), "paths": changed_paths}

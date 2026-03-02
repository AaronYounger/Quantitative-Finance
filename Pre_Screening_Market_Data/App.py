"""Streamlit entrypoint for browsing prices and computing pre-screening metrics."""

import math
import re
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from charts import CandleChart
from data_provider import PreScreenDataProvider
from metrics import get_metrics, metrics_by_id

st.set_page_config(page_title="Pre-Screen Market Data", layout="wide")

STORE_PATH = "data/store.json"


if "page" not in st.session_state:
    st.session_state.page = 1
if "df" not in st.session_state:
    st.session_state.df = None
if "ticker" not in st.session_state:
    st.session_state.ticker = ""
if "metric_choice" not in st.session_state:
    st.session_state.metric_choice = ""
if "last_matches" not in st.session_state:
    st.session_state.last_matches = pd.DataFrame(columns=["ticker", "name"])
if "loaded_period" not in st.session_state:
    st.session_state.loaded_period = ""
if "loaded_interval" not in st.session_state:
    st.session_state.loaded_interval = ""
if "loaded_company" not in st.session_state:
    st.session_state.loaded_company = ""


def _trades_total_closed(trades: dict) -> int:
    total = trades.get("total", {})
    closed = total.get("closed", 0)
    return int(closed) if isinstance(closed, (int, float)) else 0


def _drawdown_max_pct(drawdown: dict) -> float:
    max_section = drawdown.get("max", {})
    pct = max_section.get("drawdown", 0.0)
    return float(pct) if isinstance(pct, (int, float)) else 0.0


def _sharpe_ratio(result: dict) -> float:
    raw = result.get("sharpe", {}).get("sharperatio", float("nan"))
    return float(raw) if isinstance(raw, (int, float)) else float("nan")


def _trades_won(trades: dict) -> int:
    won = trades.get("won", {}).get("total", 0)
    return int(won) if isinstance(won, (int, float)) else 0


def _calc_cagr(start_value: float, end_value: float, df: pd.DataFrame) -> float:
    if start_value <= 0 or end_value <= 0 or df.empty:
        return float("nan")
    days = max(1.0, float((df.index.max() - df.index.min()).days))
    years = days / 365.25
    if years <= 0:
        return float("nan")
    return (end_value / start_value) ** (1.0 / years) - 1.0


def _equity_curve_frame(result: dict, starting_cash: float) -> pd.DataFrame:
    points = result.get("returns_series", [])
    if not isinstance(points, list) or not points:
        return pd.DataFrame(columns=["Equity"])
    frame = pd.DataFrame(points)
    frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
    frame = frame.dropna(subset=["date"]).sort_values("date")
    frame["return"] = pd.to_numeric(frame["return"], errors="coerce").fillna(0.0)
    frame["Equity"] = float(starting_cash) * (1.0 + frame["return"]).cumprod()
    return frame.set_index("date")[["Equity"]]


def _benchmark_curve_frame(benchmark_df: pd.DataFrame, starting_cash: float, target_index: pd.Index) -> pd.DataFrame:
    if benchmark_df.empty or "Close" not in benchmark_df.columns:
        return pd.DataFrame(columns=["Benchmark"])
    b = benchmark_df.copy()
    b = b.sort_index()
    b = b.reindex(target_index).ffill()
    first = float(b["Close"].iloc[0]) if len(b) else float("nan")
    if not math.isfinite(first) or first <= 0:
        return pd.DataFrame(columns=["Benchmark"])
    b["Benchmark"] = (b["Close"] / first) * float(starting_cash)
    return b[["Benchmark"]]


def _render_overview_metrics(result: dict, bt_df: pd.DataFrame, starting_cash: float):
    final_value = float(result.get("final_value", 0.0))
    cagr = _calc_cagr(float(starting_cash), final_value, bt_df)
    sharpe = _sharpe_ratio(result)
    max_dd = _drawdown_max_pct(result.get("drawdown", {}))
    trades_closed = _trades_total_closed(result.get("trades", {}))
    win_rate = (100.0 * _trades_won(result.get("trades", {})) / trades_closed) if trades_closed > 0 else 0.0

    cards = st.columns(6)
    cards[0].metric("Final Value", f"${final_value:,.2f}")
    cards[1].metric("CAGR", f"{(cagr * 100.0):.2f}%" if math.isfinite(cagr) else "n/a")
    cards[2].metric("Sharpe Ratio", f"{sharpe:.3f}" if math.isfinite(sharpe) else "n/a")
    cards[3].metric("Max DD", f"{max_dd:.2f}%")
    cards[4].metric("Trades", f"{trades_closed}")
    cards[5].metric("Win Rate", f"{win_rate:.2f}%")


def _build_equity_vs_benchmark_chart(equity: pd.DataFrame, benchmark: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    if not equity.empty:
        fig.add_trace(go.Scatter(x=equity.index, y=equity["Equity"], mode="lines", name="Strategy Equity"))
    if not benchmark.empty:
        fig.add_trace(go.Scatter(x=benchmark.index, y=benchmark["Benchmark"], mode="lines", name="S&P 500 Benchmark"))
    fig.update_layout(title="Equity Curve vs Benchmark (S&P 500)", xaxis_title="Date", yaxis_title="Value")
    return fig


def _build_price_markers_chart(bt_df: pd.DataFrame, strategy_cfg: dict, result: dict) -> go.Figure:
    from Backtesting import compute_indicators_pandas

    x_idx = pd.to_datetime(bt_df.index, errors="coerce")
    has_time = bool(
        (x_idx.hour != 0).any()
        or (x_idx.minute != 0).any()
        or (x_idx.second != 0).any()
        or (x_idx.microsecond != 0).any()
    )
    fmt = "%Y-%m-%d %H:%M" if has_time else "%Y-%m-%d"
    x_vals = x_idx.strftime(fmt)

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=x_vals,
                open=bt_df["Open"],
                high=bt_df["High"],
                low=bt_df["Low"],
                close=bt_df["Close"],
                name="Price",
            )
        ]
    )

    overlays = compute_indicators_pandas(bt_df.copy(), strategy_cfg.get("indicators", []))
    for spec in strategy_cfg.get("indicators", []):
        ind_id = spec.get("id", "")
        ind_type = str(spec.get("type", "")).upper()
        if ind_id in overlays.columns and ind_type in ("SMA", "EMA"):
            fig.add_trace(go.Scatter(x=x_vals, y=overlays[ind_id], mode="lines", name=ind_id))

    entry_points = pd.DataFrame(result.get("entry_points", []))
    if not entry_points.empty:
        entry_points["date"] = pd.to_datetime(entry_points["date"], errors="coerce")
        entry_points = entry_points.dropna(subset=["date"])
        entry_points["x_label"] = entry_points["date"].dt.strftime(fmt)
        fig.add_trace(
            go.Scatter(
                x=entry_points["x_label"],
                y=entry_points["price"],
                mode="markers",
                marker={"symbol": "triangle-up", "size": 10},
                name="Entry",
            )
        )

    exit_points = pd.DataFrame(result.get("exit_points", []))
    if not exit_points.empty:
        exit_points["date"] = pd.to_datetime(exit_points["date"], errors="coerce")
        exit_points = exit_points.dropna(subset=["date"])
        exit_points["x_label"] = exit_points["date"].dt.strftime(fmt)
        fig.add_trace(
            go.Scatter(
                x=exit_points["x_label"],
                y=exit_points["price"],
                mode="markers",
                marker={"symbol": "triangle-down", "size": 10},
                name="Exit",
            )
        )

    fig.update_layout(title="Price + Trade Markers", xaxis_title="Date", yaxis_title="Price")
    fig.update_xaxes(type="category")
    return fig


def _build_win_loss_distribution(result: dict) -> go.Figure:
    pnl = pd.to_numeric(pd.Series(result.get("closed_trade_pnls", []), dtype="float64"), errors="coerce").dropna()
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=wins, name="Wins", opacity=0.7))
    fig.add_trace(go.Histogram(x=losses, name="Losses", opacity=0.7))
    fig.update_layout(
        title="Win/Loss Distribution (PnL per Closed Trade)",
        xaxis_title="PnL",
        yaxis_title="Count",
        barmode="overlay",
    )
    return fig


def _build_rolling_sharpe_chart(equity: pd.DataFrame, window: int = 63) -> go.Figure:
    fig = go.Figure()
    if equity.empty:
        fig.update_layout(title="Rolling Sharpe Ratio")
        return fig
    rets = equity["Equity"].pct_change().dropna()
    roll_mean = rets.rolling(window).mean()
    roll_std = rets.rolling(window).std()
    roll_sharpe = (roll_mean / roll_std.replace(0.0, pd.NA)) * (252.0 ** 0.5)
    fig.add_trace(go.Scatter(x=roll_sharpe.index, y=roll_sharpe, mode="lines", name=f"Rolling Sharpe ({window})"))
    fig.update_layout(title="Rolling Sharpe Ratio", xaxis_title="Date", yaxis_title="Sharpe")
    return fig


def _default_backtest_config() -> dict:
    return {
        "pricePeriod": "",
        "priceInterval": "1d",
        "useAdjusted": True,
        "forceRefresh": False,
        "executionTiming": "signal_close_fill_next_open",
        "slippageBps": 2.0,
        "commission": {"model": "percent_notional", "value": 0.0005},
        "longOnly": True,
        "mode": "single",
        "multiMode": "batch",
    }


def _backtest_config_to_market_sim(backtest_config: dict) -> dict:
    commission_model = str(backtest_config.get("commission", {}).get("model", "percent_notional"))
    model_map = {
        "percent_notional": "percent",
        "per_share": "per_share",
        "per_trade": "per_trade",
    }
    return {
        "execution_timing": str(backtest_config.get("executionTiming", "signal_close_fill_next_open")),
        "slippage_bps": float(backtest_config.get("slippageBps", 0.0)),
        "commission_type": model_map.get(commission_model, "percent"),
        "commission_value": float(backtest_config.get("commission", {}).get("value", 0.0)),
        "long_only": bool(backtest_config.get("longOnly", True)),
    }


def _condition_to_engine_rule(cond: dict) -> dict:
    op = str(cond.get("op", ""))
    left = cond.get("left", {})
    right = cond.get("right", {})
    left_node = {"ref": left.get("ref")} if left.get("kind") == "series" else {"const": float(left.get("value", 0.0))}
    right_node = {"ref": right.get("ref")} if right.get("kind") == "series" else {"const": float(right.get("value", 0.0))}
    if op in ("crossup", "crossdown"):
        return {"op": op.upper(), "a": left_node, "b": right_node}
    op_map = {
        ">": "GT",
        "<": "LT",
        ">=": "GTE",
        "<=": "LTE",
        "==": "EQ",
        "!=": "NEQ",
    }
    return {"op": op_map.get(op, "EQ"), "left": left_node, "right": right_node}


def _rule_group_set_to_engine_rule(group_set: dict) -> dict:
    groups = group_set.get("groups", [])
    ordered_groups: list[tuple[dict, str]] = []
    for group in groups:
        cond_objs = group.get("conditions", [])
        if not cond_objs:
            continue
        current = _condition_to_engine_rule(cond_objs[0])
        for i in range(1, len(cond_objs)):
            next_cond = _condition_to_engine_rule(cond_objs[i])
            join = str(cond_objs[i].get("joinWithPrev", "AND")).upper()
            if join not in ("AND", "OR"):
                join = "AND"
            current = {"op": join, "args": [current, next_cond]}
        group_rule = current
        join_with_prev = str(group.get("joinWithPrev", "OR")).upper()
        if join_with_prev not in ("AND", "OR"):
            join_with_prev = "OR"
        ordered_groups.append((group_rule, join_with_prev))
    if not ordered_groups:
        return {"op": "AND", "args": []}
    current = ordered_groups[0][0]
    for idx in range(1, len(ordered_groups)):
        next_rule, join = ordered_groups[idx]
        current = {"op": join, "args": [current, next_rule]}
    return current


def _operand_to_text(opd: dict) -> str:
    if opd.get("kind") == "series":
        return str(opd.get("ref", ""))
    return str(opd.get("value", ""))


def _group_set_to_text(group_set: dict) -> str:
    groups = group_set.get("groups", [])
    parts: list[tuple[str, str]] = []
    for g in groups:
        join = str(g.get("joinWithPrev", "OR")).upper()
        if join not in ("AND", "OR"):
            join = "OR"
        cond_objs = g.get("conditions", [])
        if cond_objs:
            expr = f"{_operand_to_text(cond_objs[0].get('left', {}))} {cond_objs[0].get('op', '')} {_operand_to_text(cond_objs[0].get('right', {}))}"
            for i in range(1, len(cond_objs)):
                c = cond_objs[i]
                cjoin = str(c.get("joinWithPrev", "AND")).upper()
                if cjoin not in ("AND", "OR"):
                    cjoin = "AND"
                expr = f"{expr} {cjoin} {_operand_to_text(c.get('left', {}))} {c.get('op', '')} {_operand_to_text(c.get('right', {}))}"
            parts.append((f"({expr})", join))
    if not parts:
        return "(no conditions)"
    text = parts[0][0]
    for i in range(1, len(parts)):
        text = f"{text} {parts[i][1]} {parts[i][0]}"
    return text


def _max_required_bars(indicator_definitions: list[dict], strategy_definition: dict) -> int:
    vals = [int(x.get("requiredBars", 1)) for x in indicator_definitions if isinstance(x, dict)]
    stop_loss = strategy_definition.get("risk", {}).get("stopLoss", {})
    if isinstance(stop_loss, dict) and stop_loss.get("type") == "atr":
        atr_ref = str(stop_loss.get("atrIndicatorRef", ""))
        if atr_ref:
            atr_id = atr_ref.split(".", 1)[0]
            for ind in indicator_definitions:
                if str(ind.get("id", "")) == atr_id:
                    vals.append(int(ind.get("requiredBars", 1)))
    return max(vals) if vals else 1


def _validate_strategy_definition(strategy_definition: dict, backtest_config: dict, max_required_bars: int) -> list[str]:
    errs: list[str] = []
    signals = strategy_definition.get("signals", {})
    keys = ["longEntry", "exit"]
    if not bool(backtest_config.get("longOnly", True)):
        keys.append("shortEntry")
    comparison_ops = {">", "<", ">=", "<=", "==", "!="}
    for key in keys:
        group_set = signals.get(key)
        if not isinstance(group_set, dict):
            errs.append(f"Missing signal block: {key}")
            continue
        groups = group_set.get("groups", [])
        if not groups:
            errs.append(f"{key} must include at least one group.")
            continue
        for group in groups:
            for cond in group.get("conditions", []):
                left = cond.get("left", {})
                right = cond.get("right", {})
                op = str(cond.get("op", ""))
                if op in ("crossup", "crossdown"):
                    if left.get("kind") != "series" or right.get("kind") != "series":
                        errs.append("crossup/crossdown require series vs series.")
                if op in comparison_ops and left.get("kind") == "series" and right.get("kind") == "series":
                    if str(left.get("ref", "")) == str(right.get("ref", "")):
                        errs.append("Left and right series cannot be identical for comparison operators.")
    warmup = int(strategy_definition.get("runSettings", {}).get("warmupBars", 0))
    if warmup < max_required_bars:
        errs.append(f"Warmup bars must be >= required bars ({max_required_bars}).")
    return errs


def _build_engine_config(strategy_record: dict) -> dict:
    if "entry" in strategy_record and "exit" in strategy_record and "indicators" in strategy_record:
        return strategy_record
    strategy_definition = strategy_record.get("strategyDefinition", {})
    indicator_definitions = strategy_record.get("indicatorDefinitions", [])
    signals = strategy_definition.get("signals", {})
    run_settings = strategy_definition.get("runSettings", {})
    risk = strategy_definition.get("risk", {})
    return {
        "indicators": [
            {
                "id": ind.get("id"),
                "type": ind.get("type"),
                "input": ind.get("source"),
                "params": ind.get("params", {}),
                "outputs": ind.get("outputs", ["value"]),
                "min_bars_required": ind.get("requiredBars", 1),
            }
            for ind in indicator_definitions
        ],
        "entry": _rule_group_set_to_engine_rule(signals.get("longEntry", {"groups": []})),
        "exit": _rule_group_set_to_engine_rule(signals.get("exit", {"groups": []})),
        "execution": {
            "position_size_pct": float(risk.get("positionSizePct", 0.95)),
            "warmup_bars": int(run_settings.get("warmupBars", 60)),
            "max_orders": int(run_settings.get("maxTrades", 5000)),
            "edge_trigger": bool(run_settings.get("edgeTrigger", True)),
        },
    }


def run_backtests_for_tickers(
    provider: PreScreenDataProvider,
    tickers: list[str],
    config: dict,
    period: str,
    interval: str,
    refresh_prices: bool,
    use_adjusted: bool,
    market_sim: dict,
    starting_cash: float,
) -> pd.DataFrame:
    # Local import prevents app startup failure if backtrader is missing.
    from Backtesting import run_backtest_on_df, strategy_min_bars_required

    rows: list[dict] = []
    min_required_bars = strategy_min_bars_required(config)
    for t in tickers:
        try:
            df = provider.get_prices(
                ticker=t,
                period=period,
                interval=interval,
                refresh=refresh_prices,
                auto_adjust=use_adjusted,
            )
            if len(df) < min_required_bars:
                rows.append(
                    {
                        "ticker": t,
                        "totalReturnPct": float("nan"),
                        "cagrPct": float("nan"),
                        "sharpe": float("nan"),
                        "maxDrawdownPct": float("nan"),
                        "numTrades": 0,
                        "status": f"error: Not enough bars (need {min_required_bars}, got {len(df)})",
                    }
                )
                continue
            result = run_backtest_on_df(df=df, config=config, cash=starting_cash, market_sim=market_sim)
            final_value = float(result.get("final_value", 0.0))
            closed = _trades_total_closed(result.get("trades", {}))
            cagr = _calc_cagr(float(starting_cash), final_value, df)
            rows.append(
                {
                    "ticker": t,
                    "totalReturnPct": ((final_value / float(starting_cash)) - 1.0) * 100.0 if starting_cash > 0 else float("nan"),
                    "cagrPct": cagr * 100.0 if math.isfinite(cagr) else float("nan"),
                    "sharpe": _sharpe_ratio(result),
                    "maxDrawdownPct": _drawdown_max_pct(result.get("drawdown", {})),
                    "numTrades": closed,
                    "status": "ok",
                }
            )
        except Exception as exc:
            rows.append(
                {
                    "ticker": t,
                    "totalReturnPct": float("nan"),
                    "cagrPct": float("nan"),
                    "sharpe": float("nan"),
                    "maxDrawdownPct": float("nan"),
                    "numTrades": 0,
                    "status": f"error: {exc}",
                }
            )

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values(by="totalReturnPct", ascending=False, na_position="last").reset_index(drop=True)
    return out


@st.cache_data(ttl=60 * 60 * 24)
def cached_universe():
    """Cache the S&P 500 universe for 24 hours to reduce repeated downloads."""
    provider = PreScreenDataProvider()
    return provider.load_sp500_universe(refresh=False)


def header_bar(title: str):
    """Render the app header using a styled markdown container."""
    st.markdown(
        f"""
        <div style="
            padding: 16px 20px;
            border-radius: 12px;
            background: #0f172a;
            color: white;
            font-size: 28px;
            font-weight: 700;
            margin-bottom: 14px;">
            {title}
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metric_values(values: dict[str, float | str]):
    """Render metric outputs as Streamlit metric cards with basic formatting."""
    cols = st.columns(max(1, len(values)))
    for idx, (label, value) in enumerate(values.items()):
        target = cols[idx]
        if isinstance(value, float):
            if "Return" in label or "Drawdown" in label:
                target.metric(label, f"{value:.2f}%")
            elif "Volatility" in label:
                target.metric(label, f"{value:.4f}")
            else:
                target.metric(label, f"{value:.4f}")
        else:
            target.metric(label, str(value))


def _compose_rule(rules: list[dict], joiner: str) -> dict | None:
    clean = [r for r in rules if isinstance(r, dict)]
    if not clean:
        return None
    if len(clean) == 1:
        return clean[0]
    return {"op": joiner, "args": clean}


def page1():
    """Page 1: load ticker data and display candlestick/price table."""
    header_bar("Pre-Screen Market Data")

    provider = PreScreenDataProvider()
    provider.universe = cached_universe()

    query = st.text_input("Search ticker or company name", value="")
    matches = provider.search_universe(query, limit=15)

    st.caption("Search Results (from S&P 500 universe)")
    st.dataframe(matches, use_container_width=True, height=200)
    st.session_state.last_matches = matches.copy()

    options = (matches["name"] + " (" + matches["ticker"] + ")").tolist()
    company_name = ""
    if options:
        selection = st.selectbox("Select a company", options=options, index=0)
        ticker = provider.parse_ticker(selection)
        company_name = selection.rsplit("(", 1)[0].strip()
    else:
        raw = st.text_input("No match found. Enter ticker manually", value="AAPL")
        ticker = provider.parse_ticker(raw)
        company_name = ticker

    c1, c2, c3 = st.columns(3)
    with c1:
        period = st.selectbox(
            "Period",
            options=provider.ALLOWED_PERIODS,
            index=provider.ALLOWED_PERIODS.index(provider.default_period),
        )
    with c2:
        interval = st.selectbox("Interval", options=["1d"], index=0)
    with c3:
        refresh = st.checkbox("Force refresh", value=False)

    if st.button("Download prices", type="primary"):
        try:
            df = provider.get_prices(ticker=ticker, period=period, interval=interval, refresh=refresh)
            st.session_state.df = df
            st.session_state.ticker = ticker
            st.session_state.loaded_period = period
            st.session_state.loaded_interval = interval
            st.session_state.loaded_company = company_name
            st.success(f"Loaded {len(df)} rows for {ticker}")
        except Exception as exc:
            st.error(str(exc))

    if st.session_state.df is not None:
        plot_df = st.session_state.df
        actual_start = pd.to_datetime(plot_df.index.min(), errors="coerce")
        actual_end = pd.to_datetime(plot_df.index.max(), errors="coerce")
        date_span = "n/a"
        if pd.notna(actual_start) and pd.notna(actual_end):
            date_span = f"{actual_start.date().isoformat()} to {actual_end.date().isoformat()}"

        company_display = st.session_state.loaded_company.strip() if st.session_state.loaded_company else "Selected Company"
        ticker_display = st.session_state.ticker.strip() if st.session_state.ticker else ""
        title_company = f"{company_display} ({ticker_display})" if ticker_display else company_display

        st.subheader(f"Candlestick: {title_company}")
        st.caption(
            f"Coverage: {date_span} | Requested period: {st.session_state.loaded_period or 'n/a'} | Interval: {st.session_state.loaded_interval or 'n/a'}"
        )
        st.plotly_chart(CandleChart(st.session_state.df).figure(), use_container_width=True)

        st.subheader("Price Data")
        st.dataframe(st.session_state.df.tail(250).sort_index(ascending=False), use_container_width=True, height=280)

    right_col = st.columns(2)[1]
    with right_col:
        if st.button("Next ->", disabled=st.session_state.df is None):
            st.session_state.page = 2
            st.rerun()


def page2():
    """Page 2: compute and display selected metric outputs and optional charts."""
    header_bar("Explore Pre-Screening Metrics")

    if st.session_state.df is None:
        st.warning("No price data loaded. Go back and download prices first.")
        if st.button("<- Back"):
            st.session_state.page = 1
            st.rerun()
        return

    st.caption(f"Active ticker: {st.session_state.ticker}")
    metric_list = get_metrics()

    for idx, metric in enumerate(metric_list):
        if idx % 4 == 0:
            cols = st.columns(4)
        if cols[idx % 4].button(metric.label, key=f"metric_{metric.metric_id}", use_container_width=True):
            st.session_state.metric_choice = metric.metric_id

    st.divider()
    choice = st.session_state.metric_choice

    if choice:
        selected = metrics_by_id().get(choice)
        if selected is None:
            st.error(f"Unknown metric selected: {choice}")
        else:
            try:
                result = selected.compute(st.session_state.df.copy())
                render_metric_values(result.values)
                if result.chart is not None:
                    st.pyplot(result.chart)
            except Exception as exc:
                st.error(str(exc))
    else:
        st.info("Select one box above to compute and display that metric group.")

    st.divider()
    st.subheader("Save Ticker For Backtesting")
    try:
        from Backtesting import add_saved_ticker, list_saved_tickers

        c_save, c_jump = st.columns(2)
        with c_save:
            if st.button("Save active ticker"):
                add_saved_ticker(STORE_PATH, st.session_state.ticker)
                st.success(f"Saved {st.session_state.ticker} for backtesting.")
        with c_jump:
            saved_now = list_saved_tickers(STORE_PATH)
            st.caption("Saved tickers")
            st.write(", ".join(saved_now) if saved_now else "No saved tickers yet.")
    except Exception as exc:
        st.warning(f"Ticker storage unavailable: {exc}")

    st.divider()
    c_back, c_next = st.columns(2)
    with c_back:
        if st.button("<- Back to Prescreen"):
            st.session_state.page = 1
            st.rerun()
    with c_next:
        if st.button("Next -> Backtesting", disabled=st.session_state.df is None):
            st.session_state.page = 3
            st.rerun()


def page3():
    """Page 3: run strategy backtests on loaded ticker and optional pre-screen match list."""
    header_bar("Backtesting")

    if st.session_state.df is None:
        st.warning("No price data loaded. Go back and download prices first.")
        if st.button("<- Back"):
            st.session_state.page = 2
            st.rerun()
        return

    provider = PreScreenDataProvider()
    provider.universe = cached_universe()

    st.caption(f"Loaded ticker: {st.session_state.ticker}")

    c1, c2 = st.columns(2)
    with c1:
        period = st.selectbox(
            "Price period",
            options=provider.ALLOWED_PERIODS,
            index=provider.ALLOWED_PERIODS.index(provider.default_period),
            key="bt_period",
        )
    with c2:
        interval = st.selectbox("Price interval", options=provider.ALLOWED_INTERVALS, index=0, key="bt_interval")

    c3, c4, c5 = st.columns(3)
    with c3:
        starting_cash = st.number_input("Starting cash", min_value=100.0, value=10000.0, step=500.0)
    with c4:
        use_adjusted = st.toggle(
            "Use adjusted prices",
            value=True,
            help="Adjusted prices account for splits and dividends. Recommended for long-term equity backtests.",
        )
    with c5:
        refresh_prices = st.checkbox("Force refresh prices", value=False)

    c6, c7, c8 = st.columns(3)
    with c6:
        execution_timing_label = st.selectbox(
            "Execution timing",
            options=[
                "Signal on close -> Fill next open",
                "Signal on close -> Fill at close",
                "Signal on open -> Fill at open",
            ],
            index=0,
        )
    with c7:
        slippage_bps = float(
            st.number_input(
                "Slippage (bps)",
                min_value=0.0,
                value=2.0,
                step=0.5,
                help="Applied as adverse price adjustment. 1 bp = 0.01%.",
            )
        )
    with c8:
        long_only = st.toggle("Long-only", value=True, help="Disable opening short positions.")

    if execution_timing_label == "Signal on close -> Fill at close":
        st.warning("May introduce look-ahead bias unless signals use prior-bar data.")

    c9, c10 = st.columns(2)
    with c9:
        commission_label = st.selectbox(
            "Commission model",
            options=["Percent of notional", "$ per share", "$ per trade"],
            index=0,
        )
    with c10:
        if commission_label == "Percent of notional":
            commission_value = float(
                st.number_input(
                    "Commission rate",
                    min_value=0.0,
                    value=0.0005,
                    step=0.0001,
                    format="%.4f",
                    help="0.001 = 0.1%",
                )
            )
            commission_type = "percent"
        elif commission_label == "$ per share":
            commission_value = float(st.number_input("Commission rate", min_value=0.0, value=0.005, step=0.001))
            commission_type = "per_share"
        else:
            commission_value = float(st.number_input("Commission fee", min_value=0.0, value=1.0, step=0.25))
            commission_type = "per_trade"

    timing_to_value = {
        "Signal on close -> Fill next open": "signal_close_fill_next_open",
        "Signal on close -> Fill at close": "signal_close_fill_close",
        "Signal on open -> Fill at open": "signal_open_fill_open",
    }
    market_sim = {
        "execution_timing": timing_to_value[execution_timing_label],
        "slippage_bps": float(slippage_bps),
        "commission_type": commission_type,
        "commission_value": float(commission_value),
        "long_only": bool(long_only),
    }

    from Backtesting import (
        add_saved_ticker,
        get_indicator,
        indicator_outputs_for_type,
        get_strategy,
        list_indicators,
        list_saved_tickers,
        list_strategies,
        remove_indicator,
        remove_saved_ticker,
        run_backtest_on_df,
        strategy_min_bars_required,
        upsert_indicator,
        upsert_strategy,
        validate_and_enrich_indicator_spec,
    )

    st.divider()
    st.subheader("Saved Tickers")
    c_save_active, c_remove = st.columns(2)
    with c_save_active:
        if st.button("Save loaded ticker to list"):
            add_saved_ticker(STORE_PATH, st.session_state.ticker)
            st.success(f"Saved {st.session_state.ticker}.")
    with c_remove:
        current_saved = list_saved_tickers(STORE_PATH)
        drop_ticker = st.selectbox("Remove saved ticker", options=[""] + current_saved, key="drop_saved_ticker")
        if st.button("Remove selected ticker", disabled=drop_ticker == ""):
            remove_saved_ticker(STORE_PATH, drop_ticker)
            st.success(f"Removed {drop_ticker}.")

    saved_tickers = list_saved_tickers(STORE_PATH)
    all_bt_tickers = sorted(set(saved_tickers + [st.session_state.ticker]))
    st.caption("Backtest ticker universe")
    st.write(", ".join(all_bt_tickers) if all_bt_tickers else "No tickers available.")

    if "backtest_mode" not in st.session_state:
        st.session_state.backtest_mode = ""
    st.caption("Backtest mode")
    bm1, bm2 = st.columns(2)
    with bm1:
        if st.button("Single Backtest Mode", use_container_width=True, key="btn_mode_single"):
            st.session_state.backtest_mode = "single"
    with bm2:
        if st.button("Multi Backtest Mode", use_container_width=True, key="btn_mode_multi"):
            st.session_state.backtest_mode = "multi"

    backtest_mode = st.session_state.backtest_mode
    if backtest_mode == "":
        st.info("Select a backtest mode to continue.")
        st.divider()
        if st.button("<- Back to Metrics"):
            st.session_state.page = 2
            st.rerun()
        return
    multi_mode = "batch"
    universe_tickers: list[str] = []
    if backtest_mode == "multi":
        multi_mode = st.selectbox("Multi mode", options=["batch", "portfolio"], index=0, key="multi_mode")
        universe_tickers = st.multiselect(
            "Universe selector (saved tickers)",
            options=all_bt_tickers,
            default=saved_tickers,
            key="multi_universe_tickers",
        )

    commission_model_value = "percent_notional"
    if commission_label == "$ per share":
        commission_model_value = "per_share"
    elif commission_label == "$ per trade":
        commission_model_value = "per_trade"

    backtest_config = {
        "pricePeriod": period,
        "priceInterval": interval,
        "useAdjusted": bool(use_adjusted),
        "forceRefresh": bool(refresh_prices),
        "executionTiming": timing_to_value[execution_timing_label],
        "slippageBps": float(slippage_bps),
        "commission": {"model": commission_model_value, "value": float(commission_value)},
        "longOnly": bool(long_only),
        "mode": backtest_mode,
        "multiMode": multi_mode,
    }
    market_sim = _backtest_config_to_market_sim(backtest_config)

    st.divider()
    st.subheader("Single-Asset Ticker")
    single_asset_ticker = st.selectbox(
        "Ticker for indicator design and single-ticker backtest",
        options=all_bt_tickers if all_bt_tickers else [st.session_state.ticker],
        key="single_asset_ticker",
    )
    st.caption(f"Active single-asset ticker: {single_asset_ticker}")

    st.divider()
    st.subheader("Indicator Builder")
    ind_type = st.selectbox("Indicator type", options=["SMA", "EMA", "RSI", "MACD", "ATR"], key="ind_type")
    input_category = st.radio(
        "Price input category",
        options=["Basic pricing options", "Advanced pricing options"],
        horizontal=True,
        key="ind_input_category",
    )
    basic_inputs = ["close", "open", "high", "low"]
    advanced_inputs = ["hl2", "hlc3", "ohlc4", "volume"]
    input_options = basic_inputs if input_category == "Basic pricing options" else advanced_inputs
    ind_input = st.selectbox("Price input", options=input_options, index=0, key="ind_input")

    params: dict = {}
    if ind_type in ("SMA", "EMA", "RSI", "ATR"):
        params["period"] = int(st.number_input("Period", min_value=1, value=20, step=1, key="ind_period"))
    if ind_type == "MACD":
        params["fast"] = int(st.number_input("Fast", min_value=1, value=12, step=1, key="ind_macd_fast"))
        params["slow"] = int(st.number_input("Slow", min_value=1, value=26, step=1, key="ind_macd_slow"))
        params["signal"] = int(st.number_input("Signal", min_value=1, value=9, step=1, key="ind_macd_signal"))

    def _auto_indicator_id(indicator_type: str, source: str, indicator_params: dict) -> str:
        base = f"{indicator_type.lower()}_{source.lower()}"
        if indicator_type in ("SMA", "EMA", "RSI", "ATR"):
            core = f"{base}_{int(indicator_params.get('period', 0))}"
            return f"{single_asset_ticker.lower()}.{core}"
        if indicator_type == "MACD":
            fast = int(indicator_params.get("fast", 0))
            slow = int(indicator_params.get("slow", 0))
            signal = int(indicator_params.get("signal", 0))
            core = f"{base}_{fast}_{slow}_{signal}"
            return f"{single_asset_ticker.lower()}.{core}"
        return f"{single_asset_ticker.lower()}.{base}"

    auto_ind_id = _auto_indicator_id(ind_type, ind_input, params)
    st.caption(f"Auto ID: `{auto_ind_id}`")
    st.caption("Naming rules: lowercase; optional ticker prefix with one dot; use letters/numbers/underscores, and start with a letter.")

    use_auto_ind_id = st.checkbox("Use auto-generated ID", value=False, key="use_auto_ind_id")
    if use_auto_ind_id:
        ind_id = auto_ind_id
        st.text_input("Name_this_indicator", value=ind_id, disabled=True, key="ind_id_preview")
    else:
        if "ind_id_override" not in st.session_state:
            st.session_state.ind_id_override = auto_ind_id
        ind_id = st.text_input("Name_this_indicator", key="ind_id_override")

    indicator_ids = list_indicators(STORE_PATH)
    duplicate_indicator = ind_id.strip() in indicator_ids
    overwrite_indicator = False
    if duplicate_indicator:
        st.warning(f"Indicator ID '{ind_id.strip()}' already exists.")
        overwrite_indicator = st.checkbox("Allow overwrite for this existing ID", value=False, key="allow_ind_overwrite")

    if st.button("Save indicator"):
        if not ind_id.strip():
            st.error("Indicator ID is required.")
        elif re.fullmatch(r"[a-z][a-z0-9_]*(\.[a-z][a-z0-9_]*)?", ind_id.strip()) is None:
            st.error(
                "Invalid name. Use lowercase letters/numbers/underscores; optionally include one dot segment like 'cost.macd_close_12_26_9'."
            )
        elif duplicate_indicator and not overwrite_indicator:
            st.error("Duplicate indicator ID. Enable overwrite or choose a different ID.")
        else:
            try:
                enriched_spec = validate_and_enrich_indicator_spec(
                    {
                        "type": ind_type,
                        "input": ind_input,
                        "params": params,
                        "outputs": indicator_outputs_for_type(ind_type),
                    }
                )
                indicator_definition = {
                    "id": ind_id.strip(),
                    "type": ind_type,
                    "scope": "asset",
                    "source": ind_input,
                    "params": enriched_spec.get("params", {}),
                    "outputs": enriched_spec.get("outputs", ["value"]),
                    "requiredBars": int(enriched_spec.get("min_bars_required", 1)),
                    # Backward-compatible aliases used by existing engine paths.
                    "input": ind_input,
                    "min_bars_required": int(enriched_spec.get("min_bars_required", 1)),
                }
                upsert_indicator(STORE_PATH, ind_id.strip(), indicator_definition)
                st.success(
                    f"Saved indicator '{ind_id.strip()}' (required bars: {indicator_definition['requiredBars']})."
                )
            except Exception as exc:
                st.error(str(exc))

    indicator_ids = list_indicators(STORE_PATH)
    st.caption("Saved indicators")
    if indicator_ids:
        st.dataframe(pd.DataFrame({"indicator_id": indicator_ids}), use_container_width=True, height=180)
    else:
        st.write("No indicators saved yet.")

    st.subheader("Manage Saved Indicators")
    if indicator_ids:
        manage_indicator_id = st.selectbox("Saved indicator", options=indicator_ids, key="manage_indicator_id")
        if st.button("Remove Indicator", type="secondary"):
            remove_indicator(STORE_PATH, manage_indicator_id)
            st.success(f"Removed indicator '{manage_indicator_id}'.")
            st.rerun()
    else:
        st.caption("Save at least one indicator to manage it here.")

    st.divider()
    st.subheader("Strategy Builder")
    strategy_id_default = f"{single_asset_ticker.lower()}_strategy"
    strategy_id = st.text_input("Strategy ID", value=strategy_id_default)
    strategy_inds = st.multiselect("Indicators to include", options=indicator_ids, default=indicator_ids[:2])

    refs = ["close"]
    for ind_name in strategy_inds:
        spec = get_indicator(STORE_PATH, ind_name)
        if spec is None:
            continue
        outputs = spec.get("outputs")
        if not isinstance(outputs, list) or len(outputs) == 0:
            outputs = indicator_outputs_for_type(str(spec.get("type", "")))
        for output_name in outputs:
            refs.append(f"{ind_name}.{output_name}")

    indicator_definitions: list[dict] = []
    for ind_name in strategy_inds:
        spec = get_indicator(STORE_PATH, ind_name)
        if not isinstance(spec, dict):
            continue
        indicator_definitions.append(
            {
                "id": ind_name,
                "type": str(spec.get("type", "")).upper(),
                "scope": str(spec.get("scope", "asset")),
                "source": str(spec.get("source", spec.get("input", "close"))),
                "params": spec.get("params", {}),
                "outputs": spec.get("outputs", ["value"]),
                "requiredBars": int(spec.get("requiredBars", spec.get("min_bars_required", 1))),
            }
        )

    def build_rule_group_set(block_name: str, label: str) -> dict:
        st.markdown(f"**{label}**")
        group_count = int(st.number_input(f"{label} groups", min_value=1, max_value=5, value=1, key=f"{block_name}_groups"))
        groups: list[dict] = []
        for g in range(group_count):
            st.caption(f"{label} group {g + 1}")
            join_with_prev = "OR"
            if g > 0:
                join_with_prev = st.selectbox(
                    f"Group {g + 1} connector",
                    options=["AND", "OR"],
                    key=f"{block_name}_join_with_prev_{g}",
                )
            condition_count = int(
                st.number_input(
                    f"{label} group {g + 1} conditions",
                    min_value=1,
                    max_value=10,
                    value=1,
                    key=f"{block_name}_conditions_{g}",
                )
            )
            conditions: list[dict] = []
            for i in range(condition_count):
                st.caption(f"{label} condition {g + 1}.{i + 1}")
                cond_join = "AND"
                if i > 0:
                    cond_join = st.selectbox(
                        f"Condition {i + 1} connector",
                        options=["AND", "OR"],
                        key=f"{block_name}_cond_join_{g}_{i}",
                    )
                c_left, c_group, c_op, c_right = st.columns(4)
                with c_left:
                    left_ref = st.selectbox("Left series", options=refs, key=f"{block_name}_left_{g}_{i}")
                with c_group:
                    op_group = st.selectbox("Operator group", options=["Comparisons", "Events"], key=f"{block_name}_op_group_{g}_{i}")
                with c_op:
                    op = st.selectbox(
                        "Operator",
                        options=[">", "<", ">=", "<=", "==", "!="] if op_group == "Comparisons" else ["crossup", "crossdown"],
                        key=f"{block_name}_op_{g}_{i}",
                    )
                with c_right:
                    if op in ("crossup", "crossdown"):
                        right_ref = st.selectbox("Right series", options=refs, key=f"{block_name}_right_ref_{g}_{i}")
                        right_operand = {"kind": "series", "ref": right_ref}
                    else:
                        right_input = st.selectbox("Right input", options=["Series", "Constant"], key=f"{block_name}_right_input_{g}_{i}")
                        if right_input == "Constant":
                            right_val = float(st.number_input("Right value", value=0.0, key=f"{block_name}_right_const_{g}_{i}"))
                            right_operand = {"kind": "const", "value": right_val}
                        else:
                            right_ref = st.selectbox("Right series", options=refs, key=f"{block_name}_right_series_{g}_{i}")
                            right_operand = {"kind": "series", "ref": right_ref}
                conditions.append(
                    {
                        "joinWithPrev": cond_join,
                        "left": {"kind": "series", "ref": left_ref},
                        "op": op,
                        "right": right_operand,
                    }
                )
            groups.append({"joinWithPrev": join_with_prev, "conditions": conditions})
        return {"groups": groups}

    with st.expander("A) Signal Rules", expanded=True):
        long_entry_groups = build_rule_group_set("long_entry", "Long Entry")
        short_entry_groups = build_rule_group_set("short_entry", "Short Entry") if not bool(long_only) else None
        exit_groups = build_rule_group_set("exit", "Exit")
        st.caption(f"Long Entry: {_group_set_to_text(long_entry_groups)}")
        if short_entry_groups is not None:
            st.caption(f"Short Entry: {_group_set_to_text(short_entry_groups)}")
        st.caption(f"Exit: {_group_set_to_text(exit_groups)}")

    max_required_bars = _max_required_bars(indicator_definitions, {"risk": {}})
    if "run_warmup_bars" not in st.session_state:
        st.session_state.run_warmup_bars = max_required_bars
    if int(st.session_state.run_warmup_bars) < max_required_bars:
        st.session_state.run_warmup_bars = max_required_bars
        st.warning(f"warmupBars auto-bumped to required minimum: {max_required_bars}")

    with st.expander("B) Risk & Position", expanded=True):
        position_size_pct = float(st.number_input("positionSizePct", min_value=0.01, max_value=1.0, value=0.95, step=0.01))
        r1, r2 = st.columns(2)
        with r1:
            stop_loss_type = st.selectbox("stopLoss.type", options=["none", "atr", "percent"], index=0)
            stop_loss_value = float(st.number_input("stopLoss.value", min_value=0.0, value=0.0, step=0.1))
            atr_refs = [f"{d['id']}.value" for d in indicator_definitions if str(d.get("type", "")).upper() == "ATR"]
            stop_loss_atr_ref = st.selectbox("stopLoss.atrIndicatorRef", options=[""] + atr_refs) if stop_loss_type == "atr" else ""
        with r2:
            tp_type = st.selectbox("takeProfit.type", options=["none", "rr", "percent"], index=0)
            tp_value = float(st.number_input("takeProfit.value", min_value=0.0, value=0.0, step=0.1))

    with st.expander("C) Run Settings", expanded=True):
        warmup_bars = int(
            st.number_input("warmupBars", min_value=max_required_bars, value=int(st.session_state.run_warmup_bars), key="run_warmup_bars")
        )
        edge_trigger = st.toggle("edgeTrigger", value=True, help="Fire only on false->true transitions")
        max_orders = int(st.number_input("maxTrades", min_value=1, value=5000, step=50))

    long_summary = _group_set_to_text(long_entry_groups)
    exit_summary = _group_set_to_text(exit_groups)
    stop_loss_summary = stop_loss_type
    if stop_loss_type == "percent":
        stop_loss_summary = f"percent ({stop_loss_value:.2f})"
    elif stop_loss_type == "atr":
        atr_part = f", ref: {stop_loss_atr_ref}" if stop_loss_atr_ref else ""
        stop_loss_summary = f"atr ({stop_loss_value:.2f}{atr_part})"

    st.markdown("**Strategy Summary**")
    st.info(
        "\n".join(
            [
                f"Long when {long_summary}",
                f"Exit when {exit_summary}",
                f"Position size: {position_size_pct * 100.0:.0f}%",
                f"Stop loss: {stop_loss_summary}",
            ]
        )
    )

    risk_def = {
        "positionSizePct": float(position_size_pct),
        "stopLoss": {
            "type": stop_loss_type,
            "value": float(stop_loss_value),
            "atrIndicatorRef": stop_loss_atr_ref if stop_loss_type == "atr" else "",
        },
        "takeProfit": {"type": tp_type, "value": float(tp_value)},
    }
    strategy_definition = {
        "id": strategy_id.strip(),
        "signals": {
            "longEntry": long_entry_groups,
            "exit": exit_groups,
            **({"shortEntry": short_entry_groups} if short_entry_groups is not None else {}),
        },
        "risk": risk_def,
        "runSettings": {
            "warmupBars": int(warmup_bars),
            "edgeTrigger": bool(edge_trigger),
            "maxTrades": int(max_orders),
        },
    }

    max_required_bars = _max_required_bars(indicator_definitions, strategy_definition)
    validation_errors = _validate_strategy_definition(strategy_definition, backtest_config, max_required_bars)
    if not strategy_id.strip():
        validation_errors.append("Strategy ID is required.")
    if not strategy_inds:
        validation_errors.append("Select at least one indicator.")
    if stop_loss_type == "atr" and not stop_loss_atr_ref:
        validation_errors.append("ATR stop loss requires stopLoss.atrIndicatorRef.")
    if backtest_mode == "multi" and multi_mode == "batch" and len(universe_tickers) == 0:
        validation_errors.append("Multi batch mode requires at least one selected ticker.")
    if validation_errors:
        for msg in validation_errors:
            st.error(msg)

    if st.button("Save strategy", disabled=len(validation_errors) > 0):
        record = {
            "ticker": single_asset_ticker,
            "strategyDefinition": strategy_definition,
            "indicatorDefinitions": indicator_definitions,
            "backtestConfig": backtest_config,
        }
        engine_cfg = _build_engine_config(record)
        record.update(engine_cfg)
        upsert_strategy(STORE_PATH, strategy_definition["id"], record)
        st.success(f"Saved strategy '{strategy_definition['id']}'.")

    strategy_ids = list_strategies(STORE_PATH)
    chosen_strategy = st.selectbox("Strategy to run", options=strategy_ids if strategy_ids else [""], index=0)
    strategy_record = get_strategy(STORE_PATH, chosen_strategy) if chosen_strategy else None
    strategy_cfg = _build_engine_config(strategy_record) if isinstance(strategy_record, dict) else None
    if strategy_cfg is None:
        st.info("Save a strategy above to run a backtest.")
    else:
        try:
            required_bars = strategy_min_bars_required(strategy_cfg)
            st.caption(f"Required bars for selected strategy: {required_bars}")
        except Exception as exc:
            st.warning(f"Strategy indicator requirements are invalid: {exc}")

    st.divider()
    if backtest_mode == "single":
        st.subheader("Single Backtest")
        bt_ticker = st.selectbox("Single ticker selector", options=all_bt_tickers, index=0, key="single_run_ticker")
        if st.button("Run Single Backtest", disabled=strategy_cfg is None or len(validation_errors) > 0, type="primary"):
            try:
                bt_df = provider.get_prices(
                    ticker=bt_ticker,
                    period=backtest_config["pricePeriod"],
                    interval=backtest_config["priceInterval"],
                    refresh=backtest_config["forceRefresh"],
                    auto_adjust=backtest_config["useAdjusted"],
                )
                required_bars = strategy_min_bars_required(strategy_cfg) if strategy_cfg is not None else 1
                if len(bt_df) < required_bars:
                    st.error(f"Not enough data for selected indicators. Need at least {required_bars} bars, got {len(bt_df)}.")
                    return
                result = run_backtest_on_df(bt_df.copy(), strategy_cfg, cash=float(starting_cash), market_sim=market_sim)

                equity = _equity_curve_frame(result=result, starting_cash=float(starting_cash))
                show_perf = st.toggle("Show Performance Metrics", value=True, key="toggle_perf_metrics")
                show_charts = st.toggle("Show Charts", value=True, key="toggle_charts")
                show_trade_analytics = st.toggle("Show Trade Analytics", value=True, key="toggle_trade_analytics")
                show_raw = st.toggle("Show Raw Backtest Payload", value=False, key="toggle_raw_payload")

                if show_perf:
                    st.subheader("Performance Metrics")
                    _render_overview_metrics(result=result, bt_df=bt_df, starting_cash=float(starting_cash))
                if show_charts:
                    st.subheader("Charts")
                    benchmark = pd.DataFrame(columns=["Benchmark"])
                    try:
                        sp500_df = provider.get_prices(
                            ticker="^GSPC",
                            period=backtest_config["pricePeriod"],
                            interval=backtest_config["priceInterval"],
                            refresh=backtest_config["forceRefresh"],
                            auto_adjust=backtest_config["useAdjusted"],
                        )
                        benchmark = _benchmark_curve_frame(
                            benchmark_df=sp500_df,
                            starting_cash=float(starting_cash),
                            target_index=equity.index,
                        )
                    except Exception as bench_exc:
                        st.caption(f"S&P 500 benchmark unavailable for this run: {bench_exc}")
                    st.plotly_chart(_build_equity_vs_benchmark_chart(equity=equity, benchmark=benchmark), use_container_width=True)
                    st.plotly_chart(_build_price_markers_chart(bt_df=bt_df, strategy_cfg=strategy_cfg, result=result), use_container_width=True)
                if show_trade_analytics:
                    st.subheader("Trade Analytics")
                    st.plotly_chart(_build_win_loss_distribution(result=result), use_container_width=True)
                    st.plotly_chart(_build_rolling_sharpe_chart(equity=equity), use_container_width=True)
                if show_raw:
                    st.json(result)
            except Exception as exc:
                st.error(f"Backtest failed: {exc}")
    else:
        st.subheader("Multi-Asset Backtest")
        st.caption(f"Sub-mode: {multi_mode}")
        if multi_mode == "batch":
            if st.button(
                "Run Batch Backtest",
                disabled=strategy_cfg is None or len(universe_tickers) == 0 or len(validation_errors) > 0,
                type="primary",
            ):
                with st.spinner(f"Backtesting {len(universe_tickers)} ticker(s)..."):
                    table = run_backtests_for_tickers(
                        provider=provider,
                        tickers=universe_tickers,
                        config=strategy_cfg,
                        period=backtest_config["pricePeriod"],
                        interval=backtest_config["priceInterval"],
                        refresh_prices=backtest_config["forceRefresh"],
                        use_adjusted=backtest_config["useAdjusted"],
                        market_sim=market_sim,
                        starting_cash=float(starting_cash),
                    )
                st.dataframe(table, use_container_width=True, height=350)
        else:
            st.info("Portfolio mode UI is enabled; execution engine wiring is next.")

    st.divider()
    if st.button("<- Back to Metrics"):
        st.session_state.page = 2
        st.rerun()


if st.session_state.page == 1:
    page1()
elif st.session_state.page == 2:
    page2()
else:
    page3()

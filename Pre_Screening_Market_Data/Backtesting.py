from __future__ import annotations

import os
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import matplotlib.pyplot as plt
import backtrader as bt 
# ============================================================
# 1) JSON STORE (Indicators + Strategies)
# ============================================================

DEFAULT_STORE = {"version": 1, "indicators": {}, "strategies": {}, "saved_tickers": []}


class StoreError(Exception):
    pass


def load_store(path: str) -> Dict[str, Any]:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if not os.path.exists(path):
        save_store(path, DEFAULT_STORE)
        return dict(DEFAULT_STORE)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        return dict(DEFAULT_STORE)

    data.setdefault("version", 1)
    data.setdefault("indicators", {})
    data.setdefault("strategies", {})
    data.setdefault("saved_tickers", [])
    if (
        not isinstance(data["indicators"], dict)
        or not isinstance(data["strategies"], dict)
        or not isinstance(data["saved_tickers"], list)
    ):
        return dict(DEFAULT_STORE)

    return data


def save_store(path: str, store: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(store, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def upsert_indicator(path: str, ind_id: str, spec: Dict[str, Any]) -> None:
    store = load_store(path)
    store["indicators"][ind_id] = spec
    save_store(path, store)


def remove_indicator(path: str, ind_id: str) -> None:
    store = load_store(path)
    if ind_id in store["indicators"]:
        del store["indicators"][ind_id]
        save_store(path, store)


def upsert_strategy(path: str, strat_id: str, spec: Dict[str, Any]) -> None:
    store = load_store(path)
    store["strategies"][strat_id] = spec
    save_store(path, store)


def list_indicators(path: str) -> List[str]:
    store = load_store(path)
    return sorted(store["indicators"].keys())


def list_strategies(path: str) -> List[str]:
    store = load_store(path)
    return sorted(store["strategies"].keys())


def get_indicator(path: str, ind_id: str) -> Optional[Dict[str, Any]]:
    store = load_store(path)
    return store["indicators"].get(ind_id)


def get_strategy(path: str, strat_id: str) -> Optional[Dict[str, Any]]:
    store = load_store(path)
    return store["strategies"].get(strat_id)


def _normalized_ticker(ticker: str) -> str:
    return str(ticker).strip().upper()


def _clean_saved_ticker_list(values: Any) -> List[str]:
    if not isinstance(values, list):
        return []
    out: List[str] = []
    seen = set()
    for item in values:
        if not isinstance(item, str):
            continue
        t = _normalized_ticker(item)
        if t and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def list_saved_tickers(path: str) -> List[str]:
    store = load_store(path)
    return _clean_saved_ticker_list(store.get("saved_tickers", []))


def add_saved_ticker(path: str, ticker: str) -> None:
    t = _normalized_ticker(ticker)
    if not t:
        return
    store = load_store(path)
    saved = _clean_saved_ticker_list(store.get("saved_tickers", []))
    if t not in saved:
        saved.append(t)
        store["saved_tickers"] = saved
        save_store(path, store)


def remove_saved_ticker(path: str, ticker: str) -> None:
    t = _normalized_ticker(ticker)
    if not t:
        return
    store = load_store(path)
    saved = _clean_saved_ticker_list(store.get("saved_tickers", []))
    updated = [x for x in saved if x != t]
    if len(updated) != len(saved):
        store["saved_tickers"] = updated
        save_store(path, store)


# ============================================================
# 2) STRATEGY "DSL" (Using Symbols like >, <, >=, etc.)
# ============================================================

class ConfigError(Exception):
    pass


COMMISSION_PERCENT = "percent"
COMMISSION_PER_SHARE = "per_share"
COMMISSION_PER_TRADE = "per_trade"

TIMING_CLOSE_TO_NEXT_OPEN = "signal_close_fill_next_open"
TIMING_CLOSE_TO_CLOSE = "signal_close_fill_close"
TIMING_OPEN_TO_OPEN = "signal_open_fill_open"

INDICATOR_REGISTRY: Dict[str, Dict[str, Any]] = {
    "SMA": {
        "outputs": ["value"],
        "multi_output": False,
    },
    "EMA": {
        "outputs": ["value"],
        "multi_output": False,
    },
    "RSI": {
        "outputs": ["value"],
        "multi_output": False,
        "bounded": {"min": 0.0, "max": 100.0},
    },
    "MACD": {
        "outputs": ["macd", "signal", "hist"],
        "multi_output": True,
    },
    "ATR": {
        "outputs": ["value"],
        "multi_output": False,
    },
}

INDICATOR_RULES: Dict[str, Dict[str, Any]] = {
    "SMA": {
        "required_params": ("period",),
        "min_params": lambda p: p["period"] >= 1,
        "min_bars": lambda p: p["period"],
    },
    "EMA": {
        "required_params": ("period",),
        "min_params": lambda p: p["period"] >= 1,
        "min_bars": lambda p: p["period"],
    },
    "RSI": {
        "required_params": ("period",),
        "min_params": lambda p: p["period"] >= 1,
        "min_bars": lambda p: p["period"],
    },
    "ATR": {
        "required_params": ("period",),
        "min_params": lambda p: p["period"] >= 1,
        "min_bars": lambda p: p["period"],
    },
    "MACD": {
        "required_params": ("fast", "slow", "signal"),
        "min_params": lambda p: p["fast"] >= 1 and p["slow"] >= 1 and p["signal"] >= 1 and p["fast"] < p["slow"],
        "min_bars": lambda p: p["slow"] + p["signal"],
    },
}


# Value nodes:
#   {"ref": "close"} or {"ref": "sma20"} or {"const": 70}
ValueNode = Dict[str, Any]

# Rule nodes:
#   {"op": ">", "left": {...}, "right": {...}}
#   {"op": "AND", "args": [rule1, rule2]}
RuleNode = Dict[str, Any]


SYMBOL_TO_OP = {
    ">": "GT",
    "<": "LT",
    ">=": "GTE",
    "<=": "LTE",
    "==": "EQ",
    "!=": "NEQ",
    # Optional extras:
    "crossup": "CROSSUP",
    "crossdown": "CROSSDOWN",
}


def make_compare(symbol: str, left: ValueNode, right: ValueNode) -> RuleNode:
    """Builds a comparison rule from symbols like >, <, >=, etc."""
    if symbol not in SYMBOL_TO_OP:
        raise ConfigError(f"Unknown comparison symbol '{symbol}'. Allowed: {list(SYMBOL_TO_OP.keys())}")
    return {"op": SYMBOL_TO_OP[symbol], "left": left, "right": right}


def make_cross(symbol: str, a: ValueNode, b: ValueNode) -> RuleNode:
    """Build cross rules using 'crossup'/'crossdown'."""
    if symbol not in ("crossup", "crossdown"):
        raise ConfigError("Cross symbol must be 'crossup' or 'crossdown'")
    return {"op": SYMBOL_TO_OP[symbol], "a": a, "b": b}


def AND(*rules: RuleNode) -> RuleNode:
    return {"op": "AND", "args": list(rules)}


def OR(*rules: RuleNode) -> RuleNode:
    return {"op": "OR", "args": list(rules)}


def NOT(rule: RuleNode) -> RuleNode:
    return {"op": "NOT", "arg": rule}


# ============================================================
# 3) BACKTRADER ADAPTER (Indicators + Rule Evaluation)
# ============================================================

def _require(params: Dict[str, Any], key: str, typ: type):
    if key not in params:
        raise ConfigError(f"Missing required param '{key}'")
    if not isinstance(params[key], typ):
        raise ConfigError(f"Param '{key}' must be {typ.__name__}")
    return params[key]


def _get_input_line(strategy: bt.Strategy, input_name: str) -> bt.LineSeries:
    n = input_name.lower().strip()
    d = strategy.data
    if n in ("close", "c"): return d.close
    if n in ("open", "o"):  return d.open
    if n in ("high", "h"):  return d.high
    if n in ("low", "l"):   return d.low
    if n in ("volume", "v"):return d.volume
    if n == "hl2": return (d.high + d.low) / 2.0
    if n == "hlc3": return (d.high + d.low + d.close) / 3.0
    if n == "ohlc4": return (d.open + d.high + d.low + d.close) / 4.0
    raise ConfigError(f"Unknown input '{input_name}'")


def indicator_outputs_for_type(ind_type: str) -> List[str]:
    cfg = INDICATOR_REGISTRY.get(str(ind_type).upper(), {})
    outputs = cfg.get("outputs", ["value"])
    return [str(x) for x in outputs] if isinstance(outputs, list) else ["value"]


def indicator_outputs_for_spec(spec: Dict[str, Any]) -> List[str]:
    raw_outputs = spec.get("outputs")
    if isinstance(raw_outputs, list) and len(raw_outputs) > 0:
        return [str(x) for x in raw_outputs]
    return indicator_outputs_for_type(str(spec.get("type", "")))


def _require_positive_int(params: Dict[str, Any], key: str) -> int:
    if key not in params:
        raise ConfigError(f"Missing required param '{key}'")
    raw = params.get(key)
    if not isinstance(raw, (int, float)):
        raise ConfigError(f"Param '{key}' must be numeric")
    val = int(raw)
    if float(raw) != float(val):
        raise ConfigError(f"Param '{key}' must be a whole number")
    if val < 1:
        raise ConfigError(f"Param '{key}' must be >= 1")
    return val


def validate_and_enrich_indicator_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(spec, dict):
        raise ConfigError("Indicator spec must be dict")

    ind_type = str(spec.get("type", "")).upper()
    if not ind_type:
        raise ConfigError("Indicator spec missing type")
    if ind_type not in INDICATOR_RULES:
        raise ConfigError(f"Unknown indicator type '{ind_type}'")

    raw_params = spec.get("params", {})
    if not isinstance(raw_params, dict):
        raise ConfigError("Indicator params must be a dict")

    rule = INDICATOR_RULES[ind_type]
    required_params = rule.get("required_params", ())
    normalized_params: Dict[str, int] = {}
    for key in required_params:
        normalized_params[str(key)] = _require_positive_int(raw_params, str(key))

    min_params_fn = rule.get("min_params")
    if callable(min_params_fn) and not bool(min_params_fn(normalized_params)):
        if ind_type == "MACD":
            raise ConfigError("MACD requires fast < slow and all params >= 1")
        raise ConfigError(f"{ind_type} parameters are invalid")

    min_bars_fn = rule.get("min_bars")
    min_bars_required = int(min_bars_fn(normalized_params)) if callable(min_bars_fn) else 1
    if min_bars_required < 1:
        raise ConfigError(f"{ind_type} min_bars calculation is invalid")

    enriched = dict(spec)
    enriched["type"] = ind_type
    enriched["params"] = normalized_params
    enriched["outputs"] = indicator_outputs_for_spec(enriched)
    enriched["min_bars_required"] = min_bars_required
    return enriched


def strategy_min_bars_required(config: Dict[str, Any]) -> int:
    if not isinstance(config, dict):
        return 1
    ind_list = config.get("indicators", [])
    if not isinstance(ind_list, list) or len(ind_list) == 0:
        return 1
    required: List[int] = []
    for spec in ind_list:
        enriched = validate_and_enrich_indicator_spec(spec)
        required.append(int(enriched.get("min_bars_required", 1)))
    return max(required) if required else 1


def build_indicator(strategy: bt.Strategy, spec: Dict[str, Any]) -> bt.Indicator:
    ind_type = str(spec["type"]).upper()
    inp = str(spec.get("input", "close"))
    params = spec.get("params", {})
    if not isinstance(params, dict):
        raise ConfigError("Indicator params must be a dict")

    line = _get_input_line(strategy, inp)

    if ind_type == "SMA":
        period = _require(params, "period", int)
        return bt.ind.SMA(line, period=period)

    if ind_type == "EMA":
        period = _require(params, "period", int)
        return bt.ind.EMA(line, period=period)

    if ind_type == "RSI":
        period = _require(params, "period", int)
        return bt.ind.RSI(line, period=period)

    if ind_type == "MACD":
        pfast = int(params.get("fast", 12))
        pslow = int(params.get("slow", 26))
        psig  = int(params.get("signal", 9))
        return bt.ind.MACD(line, period_me1=pfast, period_me2=pslow, period_signal=psig)

    if ind_type == "ATR":
        period = _require(params, "period", int)
        return bt.ind.ATR(strategy.data, period=period)

    raise ConfigError(f"Unknown indicator type '{ind_type}'")


def _indicator_ref_value(
    strategy: "ConfigStrategyBT",
    indicator_id: str,
    output_name: Optional[str],
    idx: int,
) -> float:
    if indicator_id not in strategy._inds:
        raise ConfigError(f"Unknown ref '{indicator_id}'")
    ind = strategy._inds[indicator_id]
    spec = strategy._ind_specs.get(indicator_id, {})
    ind_type = str(spec.get("type", "")).upper()

    # Backward-compatible default output if no explicit output was requested.
    if output_name is None:
        return float(ind[idx])

    out = output_name.lower().strip()
    if out == "value":
        return float(ind[idx])

    if ind_type == "MACD":
        if out == "macd":
            return float(ind.macd[idx])
        if out == "signal":
            return float(ind.signal[idx])
        if out in ("hist", "histo", "histogram"):
            return float(ind.macd[idx] - ind.signal[idx])
        raise ConfigError(f"Unknown MACD output '{output_name}' for ref '{indicator_id}.{output_name}'")

    # For single-output indicators, accept ".value" as explicit output.
    raise ConfigError(f"Indicator '{indicator_id}' does not expose output '{output_name}'")


def _resolve_indicator_ref(strategy: "ConfigStrategyBT", ref_value: str) -> tuple[str, Optional[str]]:
    # Exact indicator id match takes priority, including ids containing dots.
    if ref_value in strategy._inds:
        return ref_value, None
    # Fallback: treat the final segment as output name (indicator_id.output_name).
    if "." in ref_value:
        ind_id, out = ref_value.rsplit(".", 1)
        if ind_id in strategy._inds:
            return ind_id, out
    raise ConfigError(f"Unknown ref '{ref_value}'")


def _val(strategy: "ConfigStrategyBT", node: ValueNode) -> float:
    if "const" in node:
        v = node["const"]
        if not isinstance(v, (int, float)):
            raise ConfigError("const must be numeric")
        return float(v)

    if "ref" in node:
        r = node["ref"]
        d = strategy.data
        if r in ("close", "open", "high", "low"):
            return float(getattr(d, r)[0])
        if not isinstance(r, str):
            raise ConfigError(f"Invalid ref '{r}'")
        ind_id, out = _resolve_indicator_ref(strategy, r)
        return _indicator_ref_value(strategy, ind_id, out, 0)

    raise ConfigError(f"Invalid value node: {node}")


def _val_at(strategy: "ConfigStrategyBT", node: ValueNode, idx: int) -> float:
    if "const" in node:
        return float(node["const"])

    if "ref" in node:
        r = node["ref"]
        d = strategy.data
        if r in ("close", "open", "high", "low"):
            return float(getattr(d, r)[idx])
        if not isinstance(r, str):
            raise ConfigError(f"Invalid ref '{r}'")
        ind_id, out = _resolve_indicator_ref(strategy, r)
        return _indicator_ref_value(strategy, ind_id, out, idx)

    raise ConfigError(f"Invalid value node: {node}")


def eval_rule_shift(strategy: "ConfigStrategyBT", rule: RuleNode, shift: int = 0) -> bool:
    """
    Evaluate a rule at a shifted bar.
    shift=0 uses current bar values, shift=1 uses the previous completed bar, etc.
    """
    idx_cur = -shift if shift > 0 else 0
    idx_prev = -(shift + 1) if shift >= 0 else -1

    def _val_idx(node: ValueNode, idx: int) -> float:
        return _val_at(strategy, node, idx)

    op = str(rule.get("op", "")).upper()
    if not op:
        raise ConfigError("Rule missing op")

    if op == "AND":
        args = rule.get("args", [])
        if not isinstance(args, list) or len(args) == 0:
            raise ConfigError("AND requires args list")
        return all(eval_rule_shift(strategy, r, shift=shift) for r in args)

    if op == "OR":
        args = rule.get("args", [])
        if not isinstance(args, list) or len(args) == 0:
            raise ConfigError("OR requires args list")
        return any(eval_rule_shift(strategy, r, shift=shift) for r in args)

    if op == "NOT":
        arg = rule.get("arg")
        if not isinstance(arg, dict):
            raise ConfigError("NOT requires arg")
        return not eval_rule_shift(strategy, arg, shift=shift)

    if op in ("GT", "GTE", "LT", "LTE", "EQ", "NEQ"):
        left = rule.get("left")
        right = rule.get("right")
        if not isinstance(left, dict) or not isinstance(right, dict):
            raise ConfigError(f"{op} requires left/right")
        a = _val_idx(left, idx_cur)
        b = _val_idx(right, idx_cur)
        if op == "GT":  return a > b
        if op == "GTE": return a >= b
        if op == "LT":  return a < b
        if op == "LTE": return a <= b
        if op == "EQ":  return a == b
        if op == "NEQ": return a != b

    if op in ("CROSSUP", "CROSSDOWN"):
        a = rule.get("a")
        b = rule.get("b")
        if not isinstance(a, dict) or not isinstance(b, dict):
            raise ConfigError(f"{op} requires a/b")
        if len(strategy.data) < (2 + shift):
            return False
        prev_a = _val_idx(a, idx_prev)
        prev_b = _val_idx(b, idx_prev)
        cur_a = _val_idx(a, idx_cur)
        cur_b = _val_idx(b, idx_cur)
        if op == "CROSSUP":
            return (prev_a <= prev_b) and (cur_a > cur_b)
        else:
            return (prev_a >= prev_b) and (cur_a < cur_b)

    raise ConfigError(f"Unknown op '{op}'")


def eval_rule(strategy: "ConfigStrategyBT", rule: RuleNode) -> bool:
    return eval_rule_shift(strategy, rule, shift=0)


@dataclass
class ExecSettings:
    position_size_pct: float = 0.95
    warmup_bars: int = 60
    max_orders: int = 5000
    edge_trigger: bool = True   # prevents order spam
    long_only: bool = True
    execution_timing: str = TIMING_CLOSE_TO_NEXT_OPEN


class MarketSimCommissionInfo(bt.CommInfoBase):
    params = (
        ("commission_type", COMMISSION_PERCENT),
        ("commission_value", 0.0),
        ("stocklike", True),
    )

    def _getcommission(self, size, price, pseudoexec):
        shares = abs(float(size))
        if shares <= 0:
            return 0.0
        fill_price = float(price)
        ctype = str(self.p.commission_type).lower()
        cval = float(self.p.commission_value)
        if cval <= 0:
            return 0.0
        if ctype == COMMISSION_PERCENT:
            return shares * fill_price * cval
        if ctype == COMMISSION_PER_SHARE:
            return shares * cval
        if ctype == COMMISSION_PER_TRADE:
            return cval
        raise ConfigError(f"Unknown commission type '{self.p.commission_type}'")


class ConfigStrategyBT(bt.Strategy):
    params = dict(config=None)

    def __init__(self):
        if self.p.config is None:
            raise ConfigError("Missing config")
        self.cfg: Dict[str, Any] = self.p.config

        self._inds: Dict[str, bt.Indicator] = {}
        self._ind_specs: Dict[str, Dict[str, Any]] = {}
        self._orders_sent = 0
        self.order = None
        self.entry_points: List[Dict[str, Any]] = []
        self.exit_points: List[Dict[str, Any]] = []
        self.closed_trade_pnls: List[float] = []

        # execution settings
        ex = self.cfg.get("execution", {})
        self.exec = ExecSettings(
            position_size_pct=float(ex.get("position_size_pct", 0.95)),
            warmup_bars=int(ex.get("warmup_bars", 60)),
            max_orders=int(ex.get("max_orders", 5000)),
            edge_trigger=bool(ex.get("edge_trigger", True)),
            long_only=bool(ex.get("long_only", True)),
            execution_timing=str(ex.get("execution_timing", TIMING_CLOSE_TO_NEXT_OPEN)),
        )
        if not (0 < self.exec.position_size_pct <= 1.0):
            raise ConfigError("position_size_pct must be in (0,1]")
        if self.exec.warmup_bars < 0:
            raise ConfigError("warmup_bars must be >= 0")
        if self.exec.execution_timing not in (
            TIMING_CLOSE_TO_NEXT_OPEN,
            TIMING_CLOSE_TO_CLOSE,
            TIMING_OPEN_TO_OPEN,
        ):
            raise ConfigError(f"Unknown execution_timing '{self.exec.execution_timing}'")

        # build indicators
        ind_list = self.cfg.get("indicators", [])
        if not isinstance(ind_list, list):
            raise ConfigError("'indicators' must be list")

        for spec in ind_list:
            ind_id = spec.get("id")
            if not isinstance(ind_id, str) or not ind_id:
                raise ConfigError("Indicator missing id")
            if ind_id in self._inds:
                raise ConfigError(f"Duplicate indicator id '{ind_id}'")
            validated_spec = validate_and_enrich_indicator_spec(spec)
            self._ind_specs[ind_id] = validated_spec
            self._inds[ind_id] = build_indicator(self, validated_spec)

        # rules
        self.entry_rule = self.cfg.get("entry")
        self.exit_rule = self.cfg.get("exit")
        if not isinstance(self.entry_rule, dict) or not isinstance(self.exit_rule, dict):
            raise ConfigError("Config must include dict rules 'entry' and 'exit'")

        # edge-trigger memory
        self._prev_entry = False
        self._prev_exit = False

    def _step(self, shift: int):
        min_bars = max(2 + shift, self.exec.warmup_bars + shift)
        if len(self.data) < min_bars:
            return
        if self.order:
            return
        if self._orders_sent >= self.exec.max_orders:
            return

        in_pos = self.position.size != 0
        in_long = self.position.size > 0
        in_short = self.position.size < 0

        entry_now = eval_rule_shift(self, self.entry_rule, shift=shift)
        exit_now = eval_rule_shift(self, self.exit_rule, shift=shift)

        if self.exec.edge_trigger:
            entry_sig = entry_now and not self._prev_entry
            exit_sig = exit_now and not self._prev_exit
        else:
            entry_sig = entry_now
            exit_sig = exit_now

        # exit priority
        if in_long and exit_sig:
            self.order = self.close()
            self._orders_sent += 1
        elif in_short and entry_sig:
            self.order = self.close()
            self._orders_sent += 1
        elif (not in_pos) and entry_sig:
            cash = self.broker.getcash()
            price_line = self.data.open if self.exec.execution_timing == TIMING_OPEN_TO_OPEN else self.data.close
            price = float(price_line[0])
            size = int((cash * self.exec.position_size_pct) / price)
            if size > 0:
                self.order = self.buy(size=size)
                self._orders_sent += 1
        elif (not in_pos) and (not self.exec.long_only) and exit_sig:
            cash = self.broker.getcash()
            price_line = self.data.open if self.exec.execution_timing == TIMING_OPEN_TO_OPEN else self.data.close
            price = float(price_line[0])
            size = int((cash * self.exec.position_size_pct) / price)
            if size > 0:
                self.order = self.sell(size=size)
                self._orders_sent += 1

        self._prev_entry = entry_now
        self._prev_exit = exit_now

    def next(self):
        if self.exec.execution_timing == TIMING_OPEN_TO_OPEN:
            return
        self._step(shift=0)

    def next_open(self):
        if self.exec.execution_timing != TIMING_OPEN_TO_OPEN:
            return
        # Evaluate rules on the previous completed bar and fill on this bar's open.
        self._step(shift=1)

    def notify_order(self, order):
        if order.status == order.Completed:
            dt = bt.num2date(order.executed.dt)
            point = {"date": dt.isoformat(), "price": float(order.executed.price)}
            if order.isbuy():
                self.entry_points.append(point)
            elif order.issell():
                self.exit_points.append(point)
        if order.status in (order.Completed, order.Canceled, order.Margin, order.Rejected):
            self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            self.closed_trade_pnls.append(float(trade.pnlcomm))


# ============================================================
# 4) BACKTEST RUNNER (Ticker loop + results)
# ============================================================

def df_to_bt_data(df: pd.DataFrame) -> bt.feeds.PandasData:
    """
    df must have: Open, High, Low, Close, Volume (Adj Close optional)
    index must be datetime
    """
    needed = {"Open", "High", "Low", "Close"}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame missing columns: {missing}")
    if "Volume" not in df.columns:
        df = df.copy()
        df["Volume"] = 0

    return bt.feeds.PandasData(dataname=df)


def run_backtest_on_df(
    df: pd.DataFrame,
    config: Dict[str, Any],
    cash: float = 10_000.0,
    commission: float = 0.0,
    market_sim: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    min_required = strategy_min_bars_required(config)
    if len(df) < min_required:
        raise ConfigError(
            f"Not enough data for selected indicators: need at least {min_required} bars, got {len(df)}."
        )

    sim = dict(market_sim or {})
    execution_timing = str(sim.get("execution_timing", TIMING_CLOSE_TO_NEXT_OPEN))
    if execution_timing not in (TIMING_CLOSE_TO_NEXT_OPEN, TIMING_CLOSE_TO_CLOSE, TIMING_OPEN_TO_OPEN):
        raise ConfigError(f"Unknown execution_timing '{execution_timing}'")

    cerebro = bt.Cerebro(cheat_on_open=(execution_timing == TIMING_OPEN_TO_OPEN))
    cerebro.broker.setcash(cash)

    if execution_timing == TIMING_CLOSE_TO_CLOSE:
        cerebro.broker.set_coc(True)
    elif execution_timing == TIMING_OPEN_TO_OPEN:
        cerebro.broker.set_coo(True)

    commission_type = str(sim.get("commission_type", COMMISSION_PERCENT))
    commission_value = float(sim.get("commission_value", commission))
    comminfo = MarketSimCommissionInfo(commission_type=commission_type, commission_value=commission_value)
    cerebro.broker.addcommissioninfo(comminfo)

    slippage_bps = float(sim.get("slippage_bps", 0.0))
    if slippage_bps < 0:
        raise ConfigError("slippage_bps must be >= 0")
    if slippage_bps > 0:
        cerebro.broker.set_slippage_perc(perc=slippage_bps / 10_000.0, slip_open=True)

    cfg = dict(config)
    ex = dict(cfg.get("execution", {}))
    ex["execution_timing"] = execution_timing
    ex["long_only"] = bool(sim.get("long_only", ex.get("long_only", True)))
    cfg["execution"] = ex

    data = df_to_bt_data(df)
    cerebro.adddata(data)
    cerebro.addstrategy(ConfigStrategyBT, config=cfg)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", timeframe=bt.TimeFrame.Days, annualize=True)
    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="dd")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name="timeret")

    results = cerebro.run()
    strat = results[0]
    timeret = strat.analyzers.timeret.get_analysis()
    returns_series = [
        {"date": pd.to_datetime(dt).isoformat(), "return": float(ret)}
        for dt, ret in sorted(timeret.items(), key=lambda x: x[0])
    ]
    out = {
        "final_value": float(cerebro.broker.getvalue()),
        "sharpe": strat.analyzers.sharpe.get_analysis(),
        "drawdown": strat.analyzers.dd.get_analysis(),
        "trades": strat.analyzers.trades.get_analysis(),
        "returns_series": returns_series,
        "closed_trade_pnls": [float(x) for x in strat.closed_trade_pnls],
        "entry_points": list(strat.entry_points),
        "exit_points": list(strat.exit_points),
        "market_sim_used": {
            "execution_timing": execution_timing,
            "commission_type": commission_type,
            "commission_value": commission_value,
            "slippage_bps": slippage_bps,
            "long_only": ex["long_only"],
        },
    }
    return out


# ============================================================
# 5) PLOTTING (Indicators + Entry/Exit markers)
# ============================================================

def compute_indicators_pandas(df: pd.DataFrame, indicator_specs: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Pandas-side indicator computation for plotting/rule previews.
    Backtrader will still compute everything for backtests.
    """
    out = df.copy()

    for spec in indicator_specs:
        ind_id = spec["id"]
        t = str(spec["type"]).upper()
        inp = str(spec.get("input", "Close"))
        params = spec.get("params", {})

        inp_lower = inp.lower().strip()
        if inp_lower == "open":
            source = out["Open"]
        elif inp_lower == "high":
            source = out["High"]
        elif inp_lower == "low":
            source = out["Low"]
        elif inp_lower == "volume":
            source = out["Volume"] if "Volume" in out.columns else pd.Series(0.0, index=out.index)
        elif inp_lower == "hl2":
            source = (out["High"] + out["Low"]) / 2.0
        elif inp_lower == "hlc3":
            source = (out["High"] + out["Low"] + out["Close"]) / 3.0
        elif inp_lower == "ohlc4":
            source = (out["Open"] + out["High"] + out["Low"] + out["Close"]) / 4.0
        else:
            source = out["Close"]

        if t == "SMA":
            p = int(params["period"])
            out[ind_id] = source.rolling(p).mean()

        elif t == "EMA":
            p = int(params["period"])
            out[ind_id] = source.ewm(span=p, adjust=False).mean()

        elif t == "RSI":
            p = int(params["period"])
            delta = source.diff()
            gain = delta.where(delta > 0, 0.0)
            loss = -delta.where(delta < 0, 0.0)
            avg_gain = gain.rolling(p).mean()
            avg_loss = loss.rolling(p).mean()
            rs = avg_gain / avg_loss.replace(0, pd.NA)
            out[ind_id] = 100 - (100 / (1 + rs))

        elif t == "MACD":
            fast = int(params.get("fast", 12))
            slow = int(params.get("slow", 26))
            signal = int(params.get("signal", 9))
            macd_line = source.ewm(span=fast, adjust=False).mean() - source.ewm(span=slow, adjust=False).mean()
            signal_line = macd_line.ewm(span=signal, adjust=False).mean()
            hist_line = macd_line - signal_line
            out[f"{ind_id}.macd"] = macd_line
            out[f"{ind_id}.signal"] = signal_line
            out[f"{ind_id}.hist"] = hist_line
            # Keep legacy fallback so old refs to raw id resolve to MACD line.
            out[ind_id] = macd_line

        else:
            # skip unsupported pandas plotting indicator types
            pass

    return out


def plot_strategy(df: pd.DataFrame, config: Dict[str, Any], title: str = "Strategy Plot") -> None:
    """
    Plots Close plus any SMA/EMA indicators it can compute via pandas,
    and marks entry/exit points by evaluating rules on the pandas-computed columns.
    MVP limitation: rule eval here supports refs to close + any computed columns.
    """
    dfp = compute_indicators_pandas(df, config["indicators"])

    # --- small pandas rule evaluator (for plotting markers only) ---
    def val_at(i: int, node: ValueNode) -> float:
        if "const" in node:
            return float(node["const"])
        if "ref" in node:
            r = node["ref"]
            if r == "close":
                return float(dfp["Close"].iloc[i])
            if r in dfp.columns:
                v = dfp[r].iloc[i]
                return float(v) if pd.notna(v) else float("nan")
        raise ConfigError(f"Plot eval: unknown ref {node}")

    def rule_at(i: int, rule: RuleNode) -> bool:
        op = str(rule["op"]).upper()

        if op == "AND":
            return all(rule_at(i, r) for r in rule["args"])
        if op == "OR":
            return any(rule_at(i, r) for r in rule["args"])
        if op == "NOT":
            return not rule_at(i, rule["arg"])

        if op in ("GT", "GTE", "LT", "LTE", "EQ", "NEQ"):
            a = val_at(i, rule["left"])
            b = val_at(i, rule["right"])
            if pd.isna(a) or pd.isna(b):
                return False
            if op == "GT": return a > b
            if op == "GTE": return a >= b
            if op == "LT": return a < b
            if op == "LTE": return a <= b
            if op == "EQ": return a == b
            if op == "NEQ": return a != b

        if op in ("CROSSUP", "CROSSDOWN"):
            if i < 1:
                return False
            a0 = val_at(i-1, rule["a"]); b0 = val_at(i-1, rule["b"])
            a1 = val_at(i, rule["a"]);   b1 = val_at(i, rule["b"])
            if any(pd.isna(x) for x in (a0, b0, a1, b1)):
                return False
            if op == "CROSSUP":
                return (a0 <= b0) and (a1 > b1)
            else:
                return (a0 >= b0) and (a1 < b1)

        raise ConfigError(f"Plot eval: unknown op {op}")

    warmup = int(config.get("execution", {}).get("warmup_bars", 60))

    entries = []
    exits = []
    in_pos = False

    for i in range(len(dfp)):
        if i < max(2, warmup):
            continue
        entry = rule_at(i, config["entry"])
        exit_ = rule_at(i, config["exit"])

        # exit priority
        if in_pos and exit_:
            exits.append(i)
            in_pos = False
        elif (not in_pos) and entry:
            entries.append(i)
            in_pos = True

    # --- plot ---
    plt.figure()
    plt.plot(dfp.index, dfp["Close"], label="Close")

    # plot any computed indicator columns (SMA/EMA/RSI may exist)
    for spec in config["indicators"]:
        ind_id = spec["id"]
        if ind_id in dfp.columns and spec["type"].upper() in ("SMA", "EMA"):
            plt.plot(dfp.index, dfp[ind_id], label=ind_id)

    # markers
    if entries:
        plt.scatter(dfp.index[entries], dfp["Close"].iloc[entries], marker="^", label="Entry")
    if exits:
        plt.scatter(dfp.index[exits], dfp["Close"].iloc[exits], marker="v", label="Exit")

    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()



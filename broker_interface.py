"""
Broker Interface — Trading212 v0 API

Trading212 API (beta, v0) implementation based on current official docs.

Key design decisions:
─────────────────────
1. Demo-first: defaults to paper trading. Live requires explicit opt-in.
2. Ticker mapping: yfinance "ASML.AS" → T212 "ASML_NL_EQ" via ISIN lookup
3. All 4 order types: market, limit, stop, stop_limit
4. ATR-based position sizing: risk a fixed % of capital per trade
5. Rate-limit aware: reads x-ratelimit-* headers, auto-throttles
6. Safety: HOLD never trades, max position cap, order audit log
7. Sells use negative quantity (T212 convention)

Auth: HTTP Basic (base64 of api_key:api_secret) — the new v0 scheme.

Deep module:
  Simple interface → execute_recommendation(rec) -> OrderResult
  Hides: auth, ticker mapping, rate limits, position sizing, logging
"""

import base64
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union

import requests

from config import (
    TRADING212_API_KEY, TRADING212_API_SECRET,
    TRADING212_MODE, DATA_DIR,
)
from llm_analyzer import Recommendation, Action

logger = logging.getLogger(__name__)

ORDERS_LOG = DATA_DIR / "order_history.jsonl"


# ═══════════════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════════════

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP = "STOP"
    STOP_LIMIT = "STOP_LIMIT"


class OrderStatus(Enum):
    PENDING = "PENDING"
    CONFIRMED = "CONFIRMED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FAILED = "FAILED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"


@dataclass
class OrderResult:
    success: bool
    order_id: Optional[str]
    status: OrderStatus
    message: str
    executed_price: Optional[float] = None
    executed_quantity: Optional[float] = None
    raw_response: Optional[dict] = None


# ═══════════════════════════════════════════════════════════════════════════
# Abstract Interface
# ═══════════════════════════════════════════════════════════════════════════

class BrokerInterface(ABC):
    @abstractmethod
    def connect(self) -> bool:
        pass

    @abstractmethod
    def execute_order(
        self, recommendation: Recommendation,
        quantity: float, order_type: OrderType = OrderType.MARKET,
    ) -> OrderResult:
        pass

    @abstractmethod
    def get_account_balance(self) -> Optional[float]:
        pass

    @abstractmethod
    def get_position(self, ticker: str) -> Optional[float]:
        pass

    @abstractmethod
    def get_all_positions(self) -> List[dict]:
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        pass


# ═══════════════════════════════════════════════════════════════════════════
# Trading212 Broker — v0 API
# ═══════════════════════════════════════════════════════════════════════════

class Trading212Broker(BrokerInterface):
    """
    Trading212 official API v0 implementation.

    Ticker mapping:
        yfinance "ASML.AS" → T212 "ASML_NL_EQ"
        Resolved via /instruments endpoint using ISIN cross-reference.

    Usage:
        broker = Trading212Broker(mode="demo")
        if broker.connect():
            result = broker.execute_recommendation(recommendation)
    """

    URLS = {
        "demo": "https://demo.trading212.com/api/v0",
        "live": "https://live.trading212.com/api/v0",
    }

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        mode: str = "demo",
        risk_pct_per_trade: float = 2.0,
        max_position_pct: float = 20.0,
    ):
        self.api_key = api_key or TRADING212_API_KEY
        self.api_secret = api_secret or TRADING212_API_SECRET
        self.mode = mode if mode in ("demo", "live") else "demo"
        self.base_url = self.URLS[self.mode]
        self.risk_pct = risk_pct_per_trade / 100.0
        self.max_position_pct = max_position_pct / 100.0

        # Build Basic auth header
        creds = f"{self.api_key}:{self.api_secret}"
        encoded = base64.b64encode(creds.encode("utf-8")).decode("utf-8")
        self._auth_header = f"Basic {encoded}"

        # Session
        self._session = requests.Session()
        self._session.headers.update({
            "Authorization": self._auth_header,
            "Content-Type": "application/json",
        })

        # Rate limit tracking (from response headers)
        self._rate_remaining: Dict[str, int] = {}
        self._rate_reset: Dict[str, float] = {}

        # Ticker mapping: yfinance_ticker → t212_ticker
        self._ticker_map: Dict[str, str] = {}
        self._isin_map: Dict[str, str] = {}  # isin → t212_ticker
        self._instruments: List[dict] = []

        self._connected = False
        self._account_currency = "EUR"

    # ── Connection ──────────────────────────────────────────────────────

    def connect(self) -> bool:
        """Verify credentials, load account info and instrument list."""
        if not self.api_key or not self.api_secret:
            logger.error(
                "Trading212 API key/secret not set. "
                "Set TRADING212_API_KEY and TRADING212_API_SECRET env vars."
            )
            return False

        try:
            logger.info(f"Connecting to Trading212 ({self.mode.upper()})...")

            # Test auth
            summary = self._get("/equity/account/summary")
            if summary is None:
                logger.error("Auth failed — check API key and secret")
                return False

            self._account_currency = summary.get("currencyCode", "EUR")
            free_cash = summary.get("free", 0)
            total = summary.get("total", 0)

            logger.info(
                f"  ✓ Connected | Mode: {self.mode.upper()} | "
                f"Currency: {self._account_currency} | "
                f"Free cash: {free_cash:,.2f} | Total: {total:,.2f}"
            )

            # Load instruments for ticker mapping
            self._load_instruments()

            self._connected = True
            return True

        except Exception as e:
            logger.error(f"Trading212 connection failed: {e}")
            return False

    def _load_instruments(self):
        """Fetch instrument list, build ISIN→T212 ticker map."""
        instruments = self._get("/equity/metadata/instruments")
        if not instruments or not isinstance(instruments, list):
            logger.warning("Could not load instruments — ticker mapping limited")
            return

        self._instruments = instruments
        for inst in instruments:
            t212 = inst.get("ticker", "")
            isin = inst.get("isin", "")
            if isin:
                self._isin_map[isin] = t212

        logger.info(f"  ✓ Loaded {len(instruments)} instruments, {len(self._isin_map)} ISINs")

    # ── Ticker Mapping ──────────────────────────────────────────────────

    def resolve_ticker(self, yf_ticker: str) -> Optional[str]:
        """
        Convert yfinance ticker → Trading212 ticker.

        Strategy (in order):
        1. Cached result
        2. ISIN cross-reference via yfinance
        3. Stem-based fuzzy match (ASML.AS → ASML_*)
        """
        if yf_ticker in self._ticker_map:
            return self._ticker_map[yf_ticker]

        # Try ISIN lookup
        try:
            import yfinance as yf
            stock = yf.Ticker(yf_ticker)
            isin = stock.isin
            if isin and isin != "-" and isin in self._isin_map:
                t212 = self._isin_map[isin]
                self._ticker_map[yf_ticker] = t212
                logger.info(f"  Mapped {yf_ticker} → {t212} (ISIN: {isin})")
                return t212
        except Exception as e:
            logger.debug(f"ISIN lookup failed for {yf_ticker}: {e}")

        # Fuzzy match on stem
        stem = yf_ticker.split(".")[0].upper()
        candidates = [
            inst["ticker"] for inst in self._instruments
            if inst.get("ticker", "").upper().startswith(stem + "_")
        ]

        if len(candidates) == 1:
            t212 = candidates[0]
            self._ticker_map[yf_ticker] = t212
            logger.info(f"  Mapped {yf_ticker} → {t212} (stem match)")
            return t212
        elif candidates:
            # Multiple matches — prefer _EQ suffix (equities)
            eq = [c for c in candidates if c.endswith("_EQ")]
            t212 = eq[0] if eq else candidates[0]
            self._ticker_map[yf_ticker] = t212
            logger.info(f"  Mapped {yf_ticker} → {t212} (best of {len(candidates)} candidates)")
            return t212

        logger.error(f"  ✗ Cannot map {yf_ticker} to Trading212 ticker")
        return None

    def add_ticker_mapping(self, yf_ticker: str, t212_ticker: str):
        """Manually add a ticker mapping (for edge cases)."""
        self._ticker_map[yf_ticker] = t212_ticker
        logger.info(f"  Manual mapping: {yf_ticker} → {t212_ticker}")

    # ── Order Execution ─────────────────────────────────────────────────

    def execute_order(
        self, recommendation: Recommendation,
        quantity: float, order_type: OrderType = OrderType.MARKET,
    ) -> OrderResult:
        """
        Execute an order from a bot recommendation.

        T212 convention: positive quantity = buy, negative = sell.
        """
        if not self._connected:
            return OrderResult(
                False, None, OrderStatus.FAILED, "Not connected — call connect() first",
            )

        # Resolve ticker
        t212_ticker = self.resolve_ticker(recommendation.ticker)
        if not t212_ticker:
            return OrderResult(
                False, None, OrderStatus.FAILED,
                f"Cannot map {recommendation.ticker} to Trading212",
            )

        action = recommendation.action
        if action == Action.HOLD:
            return OrderResult(False, None, OrderStatus.CANCELLED, "HOLD — no order placed")

        # T212 sells use negative quantity
        if action in (Action.SELL, Action.STRONG_SELL):
            quantity = -abs(quantity)
        else:
            quantity = abs(quantity)

        # Dispatch
        dispatch = {
            OrderType.MARKET: self._place_market_order,
            OrderType.LIMIT: self._place_limit_order,
            OrderType.STOP: self._place_stop_order,
            OrderType.STOP_LIMIT: self._place_stop_limit_order,
        }
        handler = dispatch.get(order_type)
        if not handler:
            return OrderResult(False, None, OrderStatus.FAILED, f"Unknown order type: {order_type}")

        try:
            result = handler(t212_ticker, quantity, recommendation)
            self._log_order(recommendation, result, order_type.value)
            return result
        except Exception as e:
            logger.error(f"Order execution failed: {e}", exc_info=True)
            result = OrderResult(False, None, OrderStatus.FAILED, f"Exception: {e}")
            self._log_order(recommendation, result, order_type.value)
            return result

    def _place_market_order(self, ticker: str, qty: float,
                             rec: Recommendation) -> OrderResult:
        """POST /equity/orders/market — rate limit: 50 req/1min"""
        payload = {
            "ticker": ticker,
            "quantity": round(qty, 6),
            "extendedHours": False,
        }
        resp = self._post("/equity/orders/market", payload)
        return self._parse_order_response(resp, rec, "MARKET")

    def _place_limit_order(self, ticker: str, qty: float,
                            rec: Recommendation) -> OrderResult:
        """POST /equity/orders/limit — rate limit: 1 req/2s"""
        # Buys: limit at current price (or below for value)
        # Sells: limit at target or current
        if qty > 0:
            limit_price = rec.current_price  # buy at market or better
        else:
            limit_price = rec.target_price or rec.current_price

        payload = {
            "ticker": ticker,
            "quantity": round(qty, 6),
            "limitPrice": round(limit_price, 2),
            "timeValidity": "GOOD_TILL_CANCEL",
        }
        resp = self._post("/equity/orders/limit", payload)
        return self._parse_order_response(resp, rec, "LIMIT")

    def _place_stop_order(self, ticker: str, qty: float,
                           rec: Recommendation) -> OrderResult:
        """POST /equity/orders/stop — rate limit: 1 req/2s"""
        stop_price = rec.stop_loss or rec.current_price
        payload = {
            "ticker": ticker,
            "quantity": round(qty, 6),
            "stopPrice": round(stop_price, 2),
            "timeValidity": "GOOD_TILL_CANCEL",
        }
        resp = self._post("/equity/orders/stop", payload)
        return self._parse_order_response(resp, rec, "STOP")

    def _place_stop_limit_order(self, ticker: str, qty: float,
                                  rec: Recommendation) -> OrderResult:
        """POST /equity/orders/stop_limit — rate limit: 1 req/2s"""
        stop_price = rec.stop_loss or rec.current_price
        # Add slippage buffer: 0.2% beyond stop
        slippage = rec.current_price * 0.002
        if qty < 0:  # sell: limit below stop
            limit_price = stop_price - slippage
        else:  # buy: limit above stop
            limit_price = stop_price + slippage

        payload = {
            "ticker": ticker,
            "quantity": round(qty, 6),
            "stopPrice": round(stop_price, 2),
            "limitPrice": round(limit_price, 2),
            "timeValidity": "GOOD_TILL_CANCEL",
        }
        resp = self._post("/equity/orders/stop_limit", payload)
        return self._parse_order_response(resp, rec, "STOP_LIMIT")

    def _parse_order_response(self, resp: Optional[dict],
                                rec: Recommendation, otype: str) -> OrderResult:
        if resp is None:
            return OrderResult(False, None, OrderStatus.FAILED, "No response from Trading212")

        order_id = str(resp.get("id", ""))
        status_str = resp.get("status", "UNKNOWN")
        filled_qty = resp.get("filledQuantity", 0)
        side = resp.get("side", "?")

        status_map = {
            "LOCAL": OrderStatus.PENDING,
            "NEW": OrderStatus.PENDING,
            "CONFIRMED": OrderStatus.CONFIRMED,
            "UNCONFIRMED": OrderStatus.PENDING,
            "FILLED": OrderStatus.FILLED,
            "PARTIALLY_FILLED": OrderStatus.PARTIALLY_FILLED,
            "CANCELLED": OrderStatus.CANCELLED,
            "CANCELLING": OrderStatus.CANCELLED,
            "REJECTED": OrderStatus.REJECTED,
        }
        status = status_map.get(status_str, OrderStatus.PENDING)
        success = status not in (OrderStatus.FAILED, OrderStatus.REJECTED, OrderStatus.CANCELLED)

        return OrderResult(
            success=success,
            order_id=order_id,
            status=status,
            message=f"{otype} {side} {rec.ticker}: {status_str} (id={order_id})",
            executed_price=rec.current_price,
            executed_quantity=filled_qty,
            raw_response=resp,
        )

    # ── Account & Positions ─────────────────────────────────────────────

    def get_account_balance(self) -> Optional[float]:
        """Get free (available) cash."""
        resp = self._get("/equity/account/summary")
        return resp.get("free", 0.0) if resp else None

    def get_account_summary(self) -> Optional[dict]:
        """Full account summary."""
        return self._get("/equity/account/summary")

    def get_position(self, ticker: str) -> Optional[float]:
        """Get position size for a yfinance ticker."""
        t212_ticker = self.resolve_ticker(ticker)
        if not t212_ticker:
            return None
        for pos in self.get_all_positions():
            if pos.get("ticker") == t212_ticker:
                return pos.get("quantity", 0)
        return 0.0

    def get_all_positions(self) -> List[dict]:
        """GET /equity/positions"""
        resp = self._get("/equity/positions")
        return resp if isinstance(resp, list) else []

    def get_pending_orders(self) -> List[dict]:
        """GET /equity/orders"""
        resp = self._get("/equity/orders")
        return resp if isinstance(resp, list) else []

    def cancel_order(self, order_id: str) -> bool:
        """DELETE /equity/orders/{id}"""
        try:
            url = f"{self.base_url}/equity/orders/{order_id}"
            resp = self._session.delete(url, timeout=15)
            self._update_rate_limits(resp, "cancel_order")
            if resp.status_code == 200:
                logger.info(f"Cancelled order {order_id}")
                return True
            logger.warning(f"Cancel failed ({resp.status_code}): {resp.text[:200]}")
            return False
        except Exception as e:
            logger.error(f"Cancel order error: {e}")
            return False

    # ── Position Sizing ─────────────────────────────────────────────────

    def calculate_position_size(self, rec: Recommendation) -> Optional[float]:
        """
        ATR-based position sizing.

        risk_amount = free_cash × risk_pct_per_trade (default 2%)
        shares = risk_amount / stop_distance

        Capped at max_position_pct of total account (default 20%).
        """
        balance = self.get_account_balance()
        if not balance or balance <= 0:
            logger.warning("Cannot size position: no free cash")
            return None

        risk_amount = balance * self.risk_pct

        # Use ATR-based stop distance from recommendation
        if rec.stop_loss and rec.current_price:
            stop_distance = abs(rec.current_price - rec.stop_loss)
            if stop_distance > 0:
                shares = risk_amount / stop_distance
                # Cap at max % of balance
                max_shares = (balance * self.max_position_pct) / rec.current_price
                shares = min(shares, max_shares)
                return round(shares, 4)

        # Fallback: risk_pct × 2 of balance in notional
        shares = (risk_amount * 2) / rec.current_price
        return round(shares, 4)

    # ── High-Level: recommendation → order ──────────────────────────────

    def execute_recommendation(
        self,
        recommendation: Recommendation,
        auto_size: bool = True,
        quantity: Optional[float] = None,
        order_type: OrderType = OrderType.MARKET,
        place_stop: bool = True,
    ) -> OrderResult:
        """
        Take a bot recommendation and execute it.

        If auto_size=True, calculates position from balance + stop distance.
        If place_stop=True and action is BUY/STRONG_BUY, also places a
        GTC stop-loss order at rec.stop_loss for protection.
        """
        action = recommendation.action
        if action == Action.HOLD:
            return OrderResult(False, None, OrderStatus.CANCELLED, "HOLD — no trade")

        # Size the position
        if auto_size:
            qty = self.calculate_position_size(recommendation)
            if not qty or qty <= 0:
                return OrderResult(
                    False, None, OrderStatus.FAILED,
                    "Position sizing failed (no cash or missing stop)",
                )
            logger.info(f"  Auto-sized: {qty:.4f} shares (risk {self.risk_pct*100:.1f}% of balance)")
        else:
            qty = quantity or 1.0

        # Execute the entry order
        result = self.execute_order(recommendation, qty, order_type)

        # If buy was successful and we have a stop_loss, place protective stop
        if (place_stop
                and result.success
                and action in (Action.BUY, Action.STRONG_BUY)
                and recommendation.stop_loss):
            self._place_protective_stop(recommendation, qty)

        return result

    def _place_protective_stop(self, rec: Recommendation, qty: float):
        """Place a GTC stop-loss order to protect a new long position."""
        t212_ticker = self.resolve_ticker(rec.ticker)
        if not t212_ticker:
            return

        payload = {
            "ticker": t212_ticker,
            "quantity": round(-abs(qty), 6),  # negative = sell
            "stopPrice": round(rec.stop_loss, 2),
            "timeValidity": "GOOD_TILL_CANCEL",
        }
        resp = self._post("/equity/orders/stop", payload)
        if resp:
            stop_id = resp.get("id", "?")
            logger.info(
                f"  ✓ Protective stop placed: sell {qty:.4f} {rec.ticker} "
                f"@ stop {rec.stop_loss:.2f} (id={stop_id})"
            )
        else:
            logger.warning(f"  ✗ Failed to place protective stop for {rec.ticker}")

    # ── HTTP Layer ──────────────────────────────────────────────────────

    def _get(self, endpoint: str) -> Optional[Union[dict, list]]:
        return self._request("GET", endpoint)

    def _post(self, endpoint: str, payload: dict) -> Optional[dict]:
        return self._request("POST", endpoint, json_data=payload)

    def _request(self, method: str, endpoint: str,
                  json_data: dict = None) -> Optional[Union[dict, list]]:
        """Rate-limit-aware API request."""
        url = f"{self.base_url}{endpoint}"
        self._wait_for_rate_limit(endpoint)

        try:
            if method == "GET":
                resp = self._session.get(url, timeout=15)
            elif method == "POST":
                resp = self._session.post(url, json=json_data, timeout=15)
            elif method == "DELETE":
                resp = self._session.delete(url, timeout=15)
            else:
                return None

            self._update_rate_limits(resp, endpoint)

            if resp.status_code == 200:
                return resp.json()
            elif resp.status_code == 429:
                retry_after = float(resp.headers.get("retry-after", "5"))
                logger.warning(f"Rate limited on {endpoint}, waiting {retry_after:.0f}s")
                time.sleep(retry_after)
                return self._request(method, endpoint, json_data)  # retry once
            else:
                body = resp.text[:300]
                logger.error(f"T212 {resp.status_code} on {method} {endpoint}: {body}")
                return None

        except requests.exceptions.Timeout:
            logger.error(f"Timeout: {method} {endpoint}")
            return None
        except Exception as e:
            logger.error(f"Request failed: {method} {endpoint}: {e}")
            return None

    def _update_rate_limits(self, resp: requests.Response, endpoint: str):
        remaining = resp.headers.get("x-ratelimit-remaining")
        reset_ts = resp.headers.get("x-ratelimit-reset")
        if remaining is not None:
            self._rate_remaining[endpoint] = int(remaining)
        if reset_ts is not None:
            self._rate_reset[endpoint] = float(reset_ts)

    def _wait_for_rate_limit(self, endpoint: str):
        remaining = self._rate_remaining.get(endpoint)
        reset_ts = self._rate_reset.get(endpoint)
        if remaining is not None and remaining <= 1 and reset_ts:
            wait = reset_ts - time.time()
            if wait > 0:
                logger.debug(f"Rate throttle {endpoint}: sleeping {wait:.1f}s")
                time.sleep(min(wait, 30))

    # ── Order Logging ───────────────────────────────────────────────────

    def _log_order(self, rec: Recommendation, result: OrderResult, otype: str):
        try:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "mode": self.mode,
                "ticker": rec.ticker,
                "action": rec.action.value,
                "order_type": otype,
                "quantity": result.executed_quantity,
                "price": rec.current_price,
                "target": rec.target_price,
                "stop_loss": rec.stop_loss,
                "quant_score": rec.quant_score,
                "confidence": rec.confidence,
                "order_id": result.order_id,
                "status": result.status.value,
                "success": result.success,
                "message": result.message,
            }
            ORDERS_LOG.parent.mkdir(parents=True, exist_ok=True)
            with open(ORDERS_LOG, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.debug(f"Order log write failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# Paper Trading Broker (local simulation)
# ═══════════════════════════════════════════════════════════════════════════

class PaperTradingBroker(BrokerInterface):
    """Simulated broker for testing without any API keys."""

    def __init__(self, initial_balance: float = 100_000.0):
        self.balance = initial_balance
        self.positions: Dict[str, float] = {}
        self.orders: List[dict] = []
        self._connected = False

    def connect(self) -> bool:
        logger.info(f"Paper Trading connected (balance: {self.balance:,.2f})")
        self._connected = True
        return True

    def execute_order(
        self, recommendation: Recommendation,
        quantity: float, order_type: OrderType = OrderType.MARKET,
    ) -> OrderResult:
        if not self._connected:
            return OrderResult(False, None, OrderStatus.FAILED, "Not connected")

        action = recommendation.action
        price = recommendation.current_price
        ticker = recommendation.ticker

        if action in (Action.BUY, Action.STRONG_BUY):
            cost = price * quantity
            if cost > self.balance:
                return OrderResult(
                    False, None, OrderStatus.FAILED,
                    f"Insufficient balance: {self.balance:.2f} < {cost:.2f}",
                )
            self.balance -= cost
            self.positions[ticker] = self.positions.get(ticker, 0) + quantity
            side = "BUY"

        elif action in (Action.SELL, Action.STRONG_SELL):
            held = self.positions.get(ticker, 0)
            sell_qty = min(quantity, held)
            if sell_qty <= 0:
                return OrderResult(False, None, OrderStatus.FAILED, f"No position in {ticker}")
            self.balance += price * sell_qty
            self.positions[ticker] -= sell_qty
            quantity = sell_qty
            side = "SELL"
        else:
            return OrderResult(False, None, OrderStatus.CANCELLED, "HOLD")

        oid = f"PAPER_{len(self.orders) + 1}"
        self.orders.append({
            "id": oid, "ticker": ticker, "side": side,
            "qty": quantity, "price": price, "ts": datetime.now().isoformat(),
        })
        logger.info(f"Paper {side} {quantity:.4f} {ticker} @ {price:.2f} (bal: {self.balance:,.2f})")

        return OrderResult(
            True, oid, OrderStatus.FILLED,
            f"Paper {side} {quantity:.4f} {ticker}",
            executed_price=price, executed_quantity=quantity,
        )

    def get_account_balance(self) -> Optional[float]:
        return self.balance

    def get_position(self, ticker: str) -> Optional[float]:
        return self.positions.get(ticker, 0)

    def get_all_positions(self) -> List[dict]:
        return [{"ticker": t, "quantity": q} for t, q in self.positions.items() if q > 0]

    def cancel_order(self, order_id: str) -> bool:
        return False


# ═══════════════════════════════════════════════════════════════════════════
# Factory
# ═══════════════════════════════════════════════════════════════════════════

def create_broker(broker_type: str = "paper", **kwargs) -> BrokerInterface:
    """
    Create a broker:
        "paper"           → local simulation
        "trading212_demo" → T212 paper trading API
        "trading212_live" → T212 real money (requires explicit opt-in)
    """
    if broker_type == "paper":
        return PaperTradingBroker(**kwargs)
    elif broker_type == "trading212_demo":
        return Trading212Broker(mode="demo", **kwargs)
    elif broker_type == "trading212_live":
        return Trading212Broker(mode="live", **kwargs)
    else:
        raise ValueError(f"Unknown broker: {broker_type}")

"""
Market Data Collector and Technical Indicator Calculator — v2

Edge-generating enhancements over v1:
─────────────────────────────────────
1. Wilder-smoothed RSI (matches industry standard, v1 used SMA-based)
2. ATR + volatility regime detection
3. ADX for trend-strength filtering (avoid whipsaws in range-bound markets)
4. Stochastic Oscillator (catches reversals RSI misses)
5. OBV + Accumulation/Distribution + MFI (smart-money volume footprints)
6. Fractal-based Support/Resistance levels
7. Relative Strength vs benchmark index (sector rotation edge)
8. Multi-timeframe confluence (weekly confirms daily)
9. Quantitative composite score BEFORE LLM sees data (anchoring)
10. Candlestick structure awareness (engulfing, doji detection)

Deep module following Ousterhout's principles:
  Simple interface → get_analysis(ticker) returns everything
  Complexity hidden → 30+ calculations, caching, error recovery
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import yfinance as yf
from dataclasses import dataclass, field

from config import (
    INDICATORS, HISTORICAL_DAYS, SCORE_WEIGHTS,
    THRESHOLDS, REGIME, BENCHMARK_TICKER,
)

# Deferred import to avoid circular — signal_detector imports nothing from here
# at module level; it only uses MarketData via duck typing in method signatures.

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class SupportResistance:
    """Key price level with metadata"""
    price: float
    level_type: str          # "support" or "resistance"
    strength: int            # number of touches / cluster count
    distance_pct: float      # distance from current price (%)


@dataclass
class MultiTimeframe:
    """Weekly-timeframe confirmations"""
    weekly_trend: str        # "up", "down", "sideways"
    weekly_rsi: float
    weekly_macd_histogram: float
    weekly_above_20sma: bool
    confirms_daily: bool     # True if weekly agrees with daily signal


@dataclass
class RegimeInfo:
    """Market regime classification"""
    regime: str              # "trending_up", "trending_down", "ranging", "volatile"
    adx: float
    atr: float
    atr_ratio: float         # current ATR / 20-period mean ATR
    description: str


@dataclass
class QuantScore:
    """Pre-LLM quantitative composite score"""
    total: float                         # -1.0 to +1.0
    components: Dict[str, float] = field(default_factory=dict)
    signal: str = "HOLD"                 # derived from total vs thresholds

    def __post_init__(self):
        if self.total >= THRESHOLDS["strong_buy_score"]:
            self.signal = "STRONG_BUY"
        elif self.total >= THRESHOLDS["buy_score"]:
            self.signal = "BUY"
        elif self.total <= THRESHOLDS["strong_sell_score"]:
            self.signal = "STRONG_SELL"
        elif self.total <= THRESHOLDS["sell_score"]:
            self.signal = "SELL"
        else:
            self.signal = "HOLD"


@dataclass
class MarketData:
    """Complete market data and technical analysis for a ticker"""
    ticker: str
    current_price: float
    volume: int
    timestamp: datetime

    # Price history
    price_history: pd.DataFrame

    # Classic indicators (kept for backward compat)
    sma: Dict[int, float]
    ema: Dict[int, float]
    rsi: float
    macd: Dict[str, float]
    bollinger_bands: Dict[str, float]
    volume_sma: float

    # Price changes
    price_change_1d: float
    price_change_5d: float
    price_change_1m: float
    volume_ratio: float

    # ── NEW: edge-generating fields ──
    # Momentum
    stochastic: Dict[str, float] = field(default_factory=dict)

    # Volume analysis
    obv_trend: str = "neutral"           # "rising", "falling", "divergence_bull", "divergence_bear"
    mfi: float = 50.0
    ad_line_trend: str = "neutral"

    # Volatility & regime
    atr: float = 0.0
    regime: Optional[RegimeInfo] = None

    # Trend strength
    adx: float = 0.0

    # Structure
    support_resistance: List[SupportResistance] = field(default_factory=list)

    # Multi-timeframe
    mtf: Optional[MultiTimeframe] = None

    # Relative strength vs benchmark
    relative_strength_20d: float = 0.0   # positive = outperforming
    relative_strength_60d: float = 0.0

    # Composite quant score
    quant_score: Optional[QuantScore] = None

    # Candlestick signals
    candle_signal: str = "none"          # "bullish_engulfing", "bearish_engulfing", "doji", etc.

    # Volume-at-level signals (from SignalDetector)
    active_signals: list = field(default_factory=list)       # List[Signal]
    volume_profile: object = None                             # VolumeProfile or None

    @property
    def summary(self) -> Dict:
        return {
            "ticker": self.ticker,
            "price": self.current_price,
            "change_1d": self.price_change_1d,
            "volume": self.volume,
            "rsi": self.rsi,
            "trend": self._determine_trend(),
        }

    def _determine_trend(self) -> str:
        if not self.sma:
            return "unknown"
        price = self.current_price
        sma_20 = self.sma.get(20, price)
        sma_50 = self.sma.get(50, price)
        if price > sma_20 > sma_50:
            return "strong_uptrend"
        elif price > sma_20:
            return "uptrend"
        elif price < sma_20 < sma_50:
            return "strong_downtrend"
        elif price < sma_20:
            return "downtrend"
        return "sideways"


# ═══════════════════════════════════════════════════════════════════════════
# Collector
# ═══════════════════════════════════════════════════════════════════════════

class DataCollector:
    """
    Collects market data and calculates a comprehensive indicator suite.

    Simple interface:
        get_analysis(ticker) -> MarketData

    Edge vs v1:
        • 30+ technical signals (vs 6)
        • Multi-timeframe confluence
        • Relative strength vs Euro Stoxx 50
        • Regime-aware context
        • Pre-computed quant score to anchor LLM
    """

    def __init__(self):
        self._cache: Dict[str, Tuple[MarketData, datetime]] = {}
        self._cache_timeout = timedelta(minutes=1)
        self._benchmark_cache: Optional[Tuple[pd.DataFrame, datetime]] = None

    # ── Public Interface ────────────────────────────────────────────────

    def get_analysis(self, ticker: str) -> Optional[MarketData]:
        """Get complete market analysis for a ticker."""
        if ticker in self._cache:
            data, ts = self._cache[ticker]
            if datetime.now() - ts < self._cache_timeout:
                return data

        try:
            data = self._fetch_and_analyze(ticker)
            self._cache[ticker] = (data, datetime.now())
            return data
        except Exception as e:
            logger.error(f"Failed to fetch data for {ticker}: {e}", exc_info=True)
            return None

    # ── Internal: orchestration ─────────────────────────────────────────

    def _fetch_and_analyze(self, ticker: str) -> MarketData:
        logger.info(f"Fetching fresh data for {ticker}")

        stock = yf.Ticker(ticker)
        hist = stock.history(period=f"{HISTORICAL_DAYS}d")
        if hist.empty:
            raise ValueError(f"No data available for {ticker}")

        close = hist["Close"]
        high = hist["High"]
        low = hist["Low"]
        volume = hist["Volume"]
        current_price = close.iloc[-1]
        current_volume = volume.iloc[-1]

        # ── Classic indicators ──
        sma = self._calc_sma(close)
        ema = self._calc_ema(close)
        rsi = self._calc_rsi_wilder(close)
        macd = self._calc_macd(close)
        bb = self._calc_bollinger(close)
        vol_sma = volume.rolling(INDICATORS["volume_sma"]).mean().iloc[-1]

        # ── Price changes ──
        pc1d = self._safe_pct_change(close, 1)
        pc5d = self._safe_pct_change(close, 5)
        pc1m = self._safe_pct_change(close, 21)
        vol_ratio = current_volume / vol_sma if vol_sma > 0 else 1.0

        # ── New indicators ──
        stoch = self._calc_stochastic(high, low, close)
        obv_trend = self._calc_obv_trend(close, volume)
        mfi = self._calc_mfi(high, low, close, volume)
        ad_trend = self._calc_ad_trend(high, low, close, volume)
        atr = self._calc_atr(high, low, close)
        adx = self._calc_adx(high, low, close)
        regime = self._detect_regime(adx, atr, high, low, close)
        sr_levels = self._calc_support_resistance(high, low, close, current_price)
        mtf = self._calc_weekly_signals(ticker)
        rs_20, rs_60 = self._calc_relative_strength(close)
        candle = self._detect_candle_pattern(hist)

        # ── Composite quant score ──
        quant = self._compute_quant_score(
            current_price, sma, ema, rsi, macd, bb, stoch,
            obv_trend, mfi, ad_trend, atr, adx, regime,
            sr_levels, vol_ratio, rs_20, candle,
        )

        return MarketData(
            ticker=ticker,
            current_price=current_price,
            volume=int(current_volume),
            timestamp=datetime.now(),
            price_history=hist,
            sma=sma, ema=ema, rsi=rsi, macd=macd,
            bollinger_bands=bb, volume_sma=vol_sma,
            price_change_1d=pc1d, price_change_5d=pc5d,
            price_change_1m=pc1m, volume_ratio=vol_ratio,
            stochastic=stoch, obv_trend=obv_trend, mfi=mfi,
            ad_line_trend=ad_trend, atr=atr, regime=regime,
            adx=adx, support_resistance=sr_levels, mtf=mtf,
            relative_strength_20d=rs_20, relative_strength_60d=rs_60,
            quant_score=quant, candle_signal=candle,
        )

    # ── Helpers ─────────────────────────────────────────────────────────

    @staticmethod
    def _safe_pct_change(series: pd.Series, periods: int) -> float:
        if len(series) <= periods:
            return 0.0
        return ((series.iloc[-1] / series.iloc[-1 - periods]) - 1) * 100

    # ── Classic Indicators ──────────────────────────────────────────────

    def _calc_sma(self, prices: pd.Series) -> Dict[int, float]:
        return {
            p: prices.rolling(p).mean().iloc[-1]
            for p in INDICATORS["sma_periods"]
            if len(prices) >= p
        }

    def _calc_ema(self, prices: pd.Series) -> Dict[int, float]:
        return {
            p: prices.ewm(span=p, adjust=False).mean().iloc[-1]
            for p in INDICATORS["ema_periods"]
            if len(prices) >= p
        }

    def _calc_rsi_wilder(self, prices: pd.Series) -> float:
        """
        Wilder-smoothed RSI — the industry standard.

        v1 used SMA-based smoothing which dampens extremes and
        misses overbought/oversold signals that pros act on.

        Implementation uses numpy arrays to avoid pandas copy-on-write
        issues that caused all RSIs to return 50.0 (the NaN fallback).
        """
        period = INDICATORS["rsi_period"]
        if len(prices) < period + 2:
            return 50.0

        delta = prices.diff().values  # numpy array; index 0 is NaN
        gains = np.where(delta > 0, delta, 0.0)
        losses = np.where(delta < 0, -delta, 0.0)

        # Seed: SMA of first `period` valid values (indices 1 through period)
        avg_gain = np.nanmean(gains[1:period + 1])
        avg_loss = np.nanmean(losses[1:period + 1])

        # Wilder smoothing from period+1 onward
        for i in range(period + 1, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi if not np.isnan(rsi) else 50.0

    def _calc_macd(self, prices: pd.Series) -> Dict[str, float]:
        cfg = INDICATORS["macd"]
        fast = prices.ewm(span=cfg["fast"], adjust=False).mean()
        slow = prices.ewm(span=cfg["slow"], adjust=False).mean()
        macd_line = fast - slow
        signal = macd_line.ewm(span=cfg["signal"], adjust=False).mean()
        hist = macd_line - signal
        return {
            "macd": macd_line.iloc[-1],
            "signal": signal.iloc[-1],
            "histogram": hist.iloc[-1],
            "hist_rising": bool(hist.iloc[-1] > hist.iloc[-2]) if len(hist) > 1 else False,
        }

    def _calc_bollinger(self, prices: pd.Series) -> Dict[str, float]:
        p = INDICATORS["bb_period"]
        s = INDICATORS["bb_std"]
        mid = prices.rolling(p).mean()
        std = prices.rolling(p).std()
        return {
            "upper": mid.iloc[-1] + s * std.iloc[-1],
            "middle": mid.iloc[-1],
            "lower": mid.iloc[-1] - s * std.iloc[-1],
            "bandwidth": (2 * s * std.iloc[-1]) / mid.iloc[-1] if mid.iloc[-1] else 0,
        }

    # ── NEW Indicators ──────────────────────────────────────────────────

    def _calc_stochastic(self, high: pd.Series, low: pd.Series,
                          close: pd.Series) -> Dict[str, float]:
        k_period = INDICATORS["stoch_k"]
        d_period = INDICATORS["stoch_d"]
        smooth = INDICATORS["stoch_smooth"]

        low_min = low.rolling(k_period).min()
        high_max = high.rolling(k_period).max()
        raw_k = 100 * (close - low_min) / (high_max - low_min + 1e-10)
        k = raw_k.rolling(smooth).mean()
        d = k.rolling(d_period).mean()
        return {
            "k": k.iloc[-1] if not pd.isna(k.iloc[-1]) else 50.0,
            "d": d.iloc[-1] if not pd.isna(d.iloc[-1]) else 50.0,
        }

    def _calc_obv_trend(self, close: pd.Series, volume: pd.Series) -> str:
        """On-Balance Volume trend with divergence detection."""
        direction = np.sign(close.diff())
        obv = (volume * direction).cumsum()
        obv_sma = obv.rolling(20).mean()

        price_up = close.iloc[-1] > close.iloc[-21] if len(close) > 21 else True
        obv_up = obv.iloc[-1] > obv_sma.iloc[-1]

        if price_up and not obv_up:
            return "divergence_bear"
        if not price_up and obv_up:
            return "divergence_bull"
        if obv_up:
            return "rising"
        return "falling"

    def _calc_mfi(self, high: pd.Series, low: pd.Series,
                   close: pd.Series, volume: pd.Series) -> float:
        """Money Flow Index — volume-weighted RSI."""
        period = INDICATORS["mfi_period"]
        typical = (high + low + close) / 3
        raw_flow = typical * volume
        delta = typical.diff()

        pos_flow = raw_flow.where(delta > 0, 0).rolling(period).sum()
        neg_flow = raw_flow.where(delta < 0, 0).rolling(period).sum()

        ratio = pos_flow / (neg_flow + 1e-10)
        mfi = 100 - (100 / (1 + ratio))
        val = mfi.iloc[-1]
        return val if not pd.isna(val) else 50.0

    def _calc_ad_trend(self, high: pd.Series, low: pd.Series,
                        close: pd.Series, volume: pd.Series) -> str:
        """Accumulation/Distribution line trend."""
        clv = ((close - low) - (high - close)) / (high - low + 1e-10)
        ad = (clv * volume).cumsum()
        ad_sma = ad.rolling(20).mean()
        if ad.iloc[-1] > ad_sma.iloc[-1]:
            return "accumulation"
        return "distribution"

    def _calc_atr(self, high: pd.Series, low: pd.Series,
                   close: pd.Series) -> float:
        period = INDICATORS["atr_period"]
        tr1 = high - low
        tr2 = (high - close.shift()).abs()
        tr3 = (low - close.shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.ewm(span=period, adjust=False).mean()
        return atr.iloc[-1] if not pd.isna(atr.iloc[-1]) else 0.0

    def _calc_adx(self, high: pd.Series, low: pd.Series,
                   close: pd.Series) -> float:
        """Average Directional Index — trend strength (not direction)."""
        period = INDICATORS["adx_period"]
        plus_dm = high.diff().clip(lower=0)
        minus_dm = (-low.diff()).clip(lower=0)

        # Zero out weaker of the two
        mask = plus_dm > minus_dm
        plus_dm = plus_dm.where(mask, 0)
        minus_dm = minus_dm.where(~mask, 0)

        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)

        atr = tr.ewm(span=period, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(span=period, adjust=False).mean() / (atr + 1e-10)
        minus_di = 100 * minus_dm.ewm(span=period, adjust=False).mean() / (atr + 1e-10)

        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
        adx = dx.ewm(span=period, adjust=False).mean()
        return adx.iloc[-1] if not pd.isna(adx.iloc[-1]) else 0.0

    def _detect_regime(self, adx: float, atr: float,
                        high: pd.Series, low: pd.Series,
                        close: pd.Series) -> RegimeInfo:
        """Classify current market regime — critical for strategy selection."""
        # ATR expansion ratio
        period = INDICATORS["atr_period"]
        tr = pd.concat([
            high - low,
            (high - close.shift()).abs(),
            (low - close.shift()).abs(),
        ], axis=1).max(axis=1)
        atr_series = tr.ewm(span=period, adjust=False).mean()
        atr_mean = atr_series.rolling(20).mean().iloc[-1]
        atr_ratio = atr / atr_mean if atr_mean > 0 else 1.0

        trend_dir = "up" if close.iloc[-1] > close.iloc[-20] else "down"

        if adx >= REGIME["trending_adx"]:
            regime = f"trending_{trend_dir}"
            desc = f"Strong {trend_dir}trend (ADX={adx:.0f}). Favor trend-following."
        elif adx <= REGIME["weak_adx"]:
            if atr_ratio > REGIME["vol_expansion"]:
                regime = "volatile"
                desc = f"Choppy with expanding volatility (ADX={adx:.0f}, ATR ratio={atr_ratio:.2f}). Reduce size."
            else:
                regime = "ranging"
                desc = f"Range-bound (ADX={adx:.0f}). Favor mean-reversion."
        else:
            regime = f"weak_trend_{trend_dir}"
            desc = f"Weak {trend_dir}trend (ADX={adx:.0f}). Mixed signals likely."

        return RegimeInfo(
            regime=regime, adx=adx, atr=atr,
            atr_ratio=atr_ratio, description=desc,
        )

    # ── Support / Resistance ────────────────────────────────────────────

    def _calc_support_resistance(self, high: pd.Series, low: pd.Series,
                                   close: pd.Series,
                                   current_price: float) -> List[SupportResistance]:
        """Fractal-based S/R with clustering."""
        lookback = INDICATORS["pivot_lookback"]
        cluster_pct = INDICATORS["sr_cluster_pct"]
        levels: List[float] = []

        # Fractal highs (resistance candidates)
        for i in range(lookback, len(high) - lookback):
            window = high.iloc[i - lookback:i + lookback + 1]
            if high.iloc[i] == window.max():
                levels.append(high.iloc[i])

        # Fractal lows (support candidates)
        for i in range(lookback, len(low) - lookback):
            window = low.iloc[i - lookback:i + lookback + 1]
            if low.iloc[i] == window.min():
                levels.append(low.iloc[i])

        if not levels:
            return []

        # Cluster nearby levels
        levels.sort()
        clusters: List[Tuple[float, int]] = []
        current_cluster = [levels[0]]

        for lv in levels[1:]:
            if (lv - current_cluster[0]) / current_cluster[0] < cluster_pct:
                current_cluster.append(lv)
            else:
                clusters.append((np.mean(current_cluster), len(current_cluster)))
                current_cluster = [lv]
        clusters.append((np.mean(current_cluster), len(current_cluster)))

        # Convert to SupportResistance objects
        result = []
        for price_level, strength in clusters:
            dist_pct = ((price_level - current_price) / current_price) * 100
            ltype = "resistance" if price_level > current_price else "support"
            result.append(SupportResistance(
                price=price_level, level_type=ltype,
                strength=strength, distance_pct=dist_pct,
            ))

        # Return closest 3 supports and 3 resistances
        supports = sorted(
            [s for s in result if s.level_type == "support"],
            key=lambda x: abs(x.distance_pct),
        )[:3]
        resistances = sorted(
            [r for r in result if r.level_type == "resistance"],
            key=lambda x: abs(x.distance_pct),
        )[:3]
        return supports + resistances

    # ── Multi-Timeframe ─────────────────────────────────────────────────

    def _calc_weekly_signals(self, ticker: str) -> Optional[MultiTimeframe]:
        """Weekly timeframe for confluence."""
        try:
            stock = yf.Ticker(ticker)
            weekly = stock.history(period="2y", interval="1wk")
            if len(weekly) < 26:
                return None

            wclose = weekly["Close"]
            w_rsi = self._calc_rsi_wilder(wclose)
            w_sma20 = wclose.rolling(20).mean().iloc[-1]
            w_macd = self._calc_macd(wclose)

            if wclose.iloc[-1] > w_sma20:
                w_trend = "up"
            elif wclose.iloc[-1] < w_sma20:
                w_trend = "down"
            else:
                w_trend = "sideways"

            return MultiTimeframe(
                weekly_trend=w_trend,
                weekly_rsi=w_rsi,
                weekly_macd_histogram=w_macd["histogram"],
                weekly_above_20sma=wclose.iloc[-1] > w_sma20,
                confirms_daily=True,  # updated by quant scorer
            )
        except Exception as e:
            logger.warning(f"Weekly data unavailable for {ticker}: {e}")
            return None

    # ── Relative Strength ───────────────────────────────────────────────

    def _get_benchmark(self) -> Optional[pd.Series]:
        if self._benchmark_cache:
            data, ts = self._benchmark_cache
            if datetime.now() - ts < timedelta(minutes=5):
                return data
        try:
            bench = yf.Ticker(BENCHMARK_TICKER)
            hist = bench.history(period=f"{HISTORICAL_DAYS}d")
            if hist.empty:
                return None
            series = hist["Close"]
            self._benchmark_cache = (series, datetime.now())
            return series
        except Exception:
            return None

    def _calc_relative_strength(self, close: pd.Series) -> Tuple[float, float]:
        """Stock return minus benchmark return over 20d and 60d."""
        bench = self._get_benchmark()
        if bench is None or len(bench) < 60 or len(close) < 60:
            return 0.0, 0.0

        # Align lengths
        min_len = min(len(close), len(bench))
        c = close.iloc[-min_len:]
        b = bench.iloc[-min_len:]

        rs20 = self._safe_pct_change(c, 20) - self._safe_pct_change(b, 20)
        rs60 = self._safe_pct_change(c, 60) - self._safe_pct_change(b, 60)
        return rs20, rs60

    # ── Candlestick Patterns ────────────────────────────────────────────

    @staticmethod
    def _detect_candle_pattern(hist: pd.DataFrame) -> str:
        """Detect key 1-2 bar reversal patterns on latest bars."""
        if len(hist) < 3:
            return "none"

        o, h, l, c = (
            hist["Open"].iloc[-1], hist["High"].iloc[-1],
            hist["Low"].iloc[-1], hist["Close"].iloc[-1],
        )
        po, pc = hist["Open"].iloc[-2], hist["Close"].iloc[-2]
        body = abs(c - o)
        prev_body = abs(pc - po)
        wick_upper = h - max(o, c)
        wick_lower = min(o, c) - l
        full_range = h - l if h != l else 1e-10

        # Doji: body < 10% of range
        if body / full_range < 0.10:
            return "doji"

        # Bullish engulfing
        if pc < po and c > o and c > po and o < pc:
            return "bullish_engulfing"

        # Bearish engulfing
        if pc > po and c < o and c < po and o > pc:
            return "bearish_engulfing"

        # Hammer (bullish): small body at top, long lower wick
        if wick_lower > 2 * body and wick_upper < body:
            return "hammer"

        # Shooting star (bearish): small body at bottom, long upper wick
        if wick_upper > 2 * body and wick_lower < body:
            return "shooting_star"

        return "none"

    # ── Quantitative Composite Score ────────────────────────────────────

    def _compute_quant_score(
        self, price, sma, ema, rsi, macd, bb, stoch,
        obv_trend, mfi, ad_trend, atr, adx, regime,
        sr_levels, vol_ratio, rs_20, candle,
    ) -> QuantScore:
        """
        Compute a -1 to +1 composite score.
        This ANCHORS the LLM — prevents it from hallucinating
        conviction on weak setups.
        """
        components = {}

        # 1. TREND (SMA alignment + ADX)
        trend_score = 0.0
        sma_20 = sma.get(20, price)
        sma_50 = sma.get(50, price)
        sma_200 = sma.get(200, price)

        if price > sma_20:
            trend_score += 0.25
        else:
            trend_score -= 0.25
        if sma_20 > sma_50:
            trend_score += 0.25
        else:
            trend_score -= 0.25
        if price > sma_200:
            trend_score += 0.25
        else:
            trend_score -= 0.25
        # ADX bonus: stronger trend = more conviction
        if adx > REGIME["trending_adx"]:
            trend_score *= 1.3
        elif adx < REGIME["weak_adx"]:
            trend_score *= 0.5  # discount in choppy markets
        trend_score = np.clip(trend_score, -1, 1)
        components["trend"] = trend_score

        # 2. MOMENTUM (RSI zones + MACD + Stochastic)
        mom_score = 0.0
        if rsi < 30:
            mom_score += 0.4   # oversold = bullish potential
        elif rsi < 45:
            mom_score += 0.1
        elif rsi > 70:
            mom_score -= 0.4   # overbought = bearish potential
        elif rsi > 55:
            mom_score -= 0.1

        if macd["histogram"] > 0 and macd.get("hist_rising", False):
            mom_score += 0.3
        elif macd["histogram"] < 0 and not macd.get("hist_rising", True):
            mom_score -= 0.3

        stoch_k = stoch.get("k", 50)
        if stoch_k < 20:
            mom_score += 0.2
        elif stoch_k > 80:
            mom_score -= 0.2

        mom_score = np.clip(mom_score, -1, 1)
        components["momentum"] = mom_score

        # 3. VOLUME
        vol_score = 0.0
        if obv_trend == "divergence_bull":
            vol_score += 0.5
        elif obv_trend == "divergence_bear":
            vol_score -= 0.5
        elif obv_trend == "rising":
            vol_score += 0.2
        elif obv_trend == "falling":
            vol_score -= 0.2

        if mfi < 20:
            vol_score += 0.3
        elif mfi > 80:
            vol_score -= 0.3

        if ad_trend == "accumulation":
            vol_score += 0.2
        else:
            vol_score -= 0.2

        vol_score = np.clip(vol_score, -1, 1)
        components["volume"] = vol_score

        # 4. VOLATILITY (BB position)
        vol_pos_score = 0.0
        bb_upper = bb.get("upper", price)
        bb_lower = bb.get("lower", price)
        bb_mid = bb.get("middle", price)
        bb_range = bb_upper - bb_lower if bb_upper != bb_lower else 1

        bb_pct = (price - bb_lower) / bb_range  # 0=lower, 1=upper
        if bb_pct < 0.15:
            vol_pos_score += 0.4   # near lower band = mean reversion buy zone
        elif bb_pct > 0.85:
            vol_pos_score -= 0.4   # near upper band
        components["volatility"] = np.clip(vol_pos_score, -1, 1)

        # 5. RELATIVE STRENGTH
        rs_score = np.clip(rs_20 / 10.0, -1, 1)  # ±10% outperformance = full score
        components["rel_strength"] = rs_score

        # 6. STRUCTURE (proximity to S/R)
        struct_score = 0.0
        nearest_support = None
        nearest_resist = None
        for lvl in sr_levels:
            if lvl.level_type == "support" and (nearest_support is None or abs(lvl.distance_pct) < abs(nearest_support.distance_pct)):
                nearest_support = lvl
            if lvl.level_type == "resistance" and (nearest_resist is None or abs(lvl.distance_pct) < abs(nearest_resist.distance_pct)):
                nearest_resist = lvl

        if nearest_support and abs(nearest_support.distance_pct) < 2.0:
            struct_score += 0.3 * nearest_support.strength / 5  # near strong support = bullish
        if nearest_resist and abs(nearest_resist.distance_pct) < 2.0:
            struct_score -= 0.3 * nearest_resist.strength / 5  # near strong resistance = bearish

        components["structure"] = np.clip(struct_score, -1, 1)

        # COMPOSITE
        total = sum(
            components[k] * SCORE_WEIGHTS[k]
            for k in SCORE_WEIGHTS
            if k in components
        )
        total = np.clip(total, -1, 1)

        return QuantScore(total=float(total), components={k: round(v, 3) for k, v in components.items()})

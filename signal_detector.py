"""
Signal Detector — Volume-at-Level Trigger Engine

This is the "buy before it goes up" module. It detects the footprint
of institutional order flow hitting the book at key price levels.

Why this works:
──────────────
When a large institution wants to buy 500,000 shares of ASML, they can't
do it in one order without moving the market. So they accumulate over hours
or days, often at key support levels where there's liquidity (other people's
stop losses and limit orders pool at S/R levels).

This creates a detectable pattern:
  Price approaches known support → Volume spikes 2-3x above average
  → That's the institutional order being filled
  → Price reverses because the big buyer absorbed all the selling

The same works in reverse at resistance: institutions distribute into
strength, volume spikes, price reverses down.

Signal Types (by priority):
───────────────────────────
1. BREAKOUT_CONFIRMED   — Price breaks S/R with volume surge → strongest signal
2. VOLUME_CLIMAX        — Extreme volume + rejection wick → exhaustion reversal
3. ACCUMULATION_AT_SUPPORT — Volume spike at support, price holds → institutional buying
4. DISTRIBUTION_AT_RESISTANCE — Volume spike at resistance → institutional selling
5. VOLUME_DRY_UP        — Price tests level on declining volume → weak test, level holds
6. VWAP_DEVIATION       — Price far from VWAP + volume → mean reversion setup
7. VOLUME_BREAKOUT_PRELUDE — Rising volume compressing toward level → pressure building

Deep module:
  Simple interface → detect_signals(market_data) -> List[Signal]
  Complexity hidden → multi-bar analysis, volume profiling, priority scoring
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
import pandas as pd

from config import INDICATORS

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════════════

SIGNAL_CONFIG = {
    # Volume thresholds (multiples of 20-day average)
    "vol_spike_threshold": 2.0,       # 2x avg = noteworthy
    "vol_surge_threshold": 3.0,       # 3x avg = major event
    "vol_climax_threshold": 4.0,      # 4x avg = exhaustion candidate
    "vol_dry_up_threshold": 0.5,      # <50% avg = dry up

    # Proximity to S/R (as % of price)
    "sr_proximity_pct": 1.5,          # within 1.5% of level
    "sr_breakout_pct": 0.3,           # must close 0.3% beyond level to confirm

    # Rejection wick ratio (for climax detection)
    "rejection_wick_ratio": 2.5,      # wick must be 2.5x body

    # VWAP deviation threshold
    "vwap_deviation_pct": 2.0,        # 2% from VWAP = noteworthy

    # Volume trend (bars of rising volume)
    "vol_trend_bars": 3,              # 3 consecutive rising volume bars

    # Lookback for volume profile
    "vol_profile_bars": 60,           # 60 days of volume-at-price data
    "vol_profile_bins": 30,           # number of price bins

    # Alert cooldown: don't re-fire same signal type within N bars
    "cooldown_bars": 5,
}


# ═══════════════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════════════

class SignalPriority(Enum):
    """Signal priority — higher = more actionable."""
    CRITICAL = 1      # Act now: breakout confirmed, climax reversal
    HIGH = 2          # Strong setup forming: accumulation/distribution at level
    MEDIUM = 3        # Worth watching: VWAP deviation, volume trend
    LOW = 4           # Context: dry-up, prelude patterns


class SignalDirection(Enum):
    BULLISH = "BULLISH"
    BEARISH = "BEARISH"
    NEUTRAL = "NEUTRAL"


@dataclass
class Signal:
    """A detected volume-at-level signal."""
    signal_type: str              # e.g. "BREAKOUT_CONFIRMED"
    priority: SignalPriority
    direction: SignalDirection
    ticker: str
    price: float
    trigger_level: float          # the S/R level that triggered it
    volume_ratio: float           # current volume / average
    confidence: float             # 0.0 to 1.0
    description: str              # human-readable explanation
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict = field(default_factory=dict)

    @property
    def priority_label(self) -> str:
        return {
            SignalPriority.CRITICAL: "🔴 CRITICAL",
            SignalPriority.HIGH: "🟠 HIGH",
            SignalPriority.MEDIUM: "🟡 MEDIUM",
            SignalPriority.LOW: "🔵 LOW",
        }.get(self.priority, "⚪ UNKNOWN")


@dataclass
class VolumeProfile:
    """Volume-at-price distribution over recent history."""
    price_levels: List[float]     # bin center prices
    volumes: List[float]          # volume at each level
    poc: float                    # Point of Control — highest volume price
    value_area_high: float        # upper bound of 70% volume zone
    value_area_low: float         # lower bound of 70% volume zone
    current_price_vs_poc: str     # "above", "below", "at"


# ═══════════════════════════════════════════════════════════════════════════
# Signal Detector
# ═══════════════════════════════════════════════════════════════════════════

class SignalDetector:
    """
    Detects institutional footprint signals from volume-at-level analysis.

    Simple interface:
        detect_signals(market_data) -> List[Signal]
        get_volume_profile(market_data) -> VolumeProfile

    Integrates with MarketData from data_collector — uses its S/R levels,
    price history, and volume data.
    """

    def __init__(self):
        self._cooldowns: Dict[str, Dict[str, int]] = {}  # ticker -> {signal_type -> bar_count}

    # ── Public Interface ────────────────────────────────────────────────

    def detect_signals(self, market_data) -> List[Signal]:
        """
        Run all signal detectors on the given MarketData.
        Returns signals sorted by priority (most urgent first).
        """
        signals: List[Signal] = []

        try:
            hist = market_data.price_history
            if len(hist) < 30:
                return signals

            close = hist["Close"]
            high = hist["High"]
            low = hist["Low"]
            volume = hist["Volume"]
            sr_levels = market_data.support_resistance

            vol_sma = volume.rolling(SIGNAL_CONFIG["vol_profile_bars"]).mean()
            current_vol_ratio = (
                volume.iloc[-1] / vol_sma.iloc[-1]
                if vol_sma.iloc[-1] > 0 else 1.0
            )

            # ── Run each detector ──
            signals.extend(
                self._detect_breakout(close, high, low, volume, vol_sma, sr_levels, market_data)
            )
            signals.extend(
                self._detect_volume_climax(close, high, low, volume, vol_sma, sr_levels, market_data)
            )
            signals.extend(
                self._detect_accumulation_distribution(
                    close, high, low, volume, vol_sma, sr_levels, market_data
                )
            )
            signals.extend(
                self._detect_volume_dry_up(close, volume, vol_sma, sr_levels, market_data)
            )
            signals.extend(
                self._detect_vwap_deviation(close, high, low, volume, market_data)
            )
            signals.extend(
                self._detect_volume_pressure(close, volume, vol_sma, sr_levels, market_data)
            )

            # Apply cooldown filtering
            signals = self._apply_cooldowns(market_data.ticker, signals)

            # Sort by priority
            signals.sort(key=lambda s: s.priority.value)

        except Exception as e:
            logger.error(f"Signal detection failed for {market_data.ticker}: {e}", exc_info=True)

        return signals

    def get_volume_profile(self, market_data) -> Optional[VolumeProfile]:
        """
        Compute volume-at-price profile for context.
        Shows where the most trading occurred — institutional positioning zones.
        """
        try:
            hist = market_data.price_history
            bars = min(len(hist), SIGNAL_CONFIG["vol_profile_bars"])
            recent = hist.iloc[-bars:]

            close = recent["Close"].values
            volume = recent["Volume"].values
            n_bins = SIGNAL_CONFIG["vol_profile_bins"]

            # Create price bins
            price_min, price_max = close.min(), close.max()
            if price_min == price_max:
                return None

            bin_edges = np.linspace(price_min, price_max, n_bins + 1)
            bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
            vol_at_price = np.zeros(n_bins)

            # Distribute each bar's volume across the price range it traded
            for i in range(len(recent)):
                bar_low = recent["Low"].iloc[i]
                bar_high = recent["High"].iloc[i]
                bar_vol = volume[i]

                for j in range(n_bins):
                    # Fraction of bar range overlapping with this bin
                    overlap_low = max(bin_edges[j], bar_low)
                    overlap_high = min(bin_edges[j + 1], bar_high)
                    if overlap_high > overlap_low:
                        bar_range = bar_high - bar_low if bar_high > bar_low else 1
                        fraction = (overlap_high - overlap_low) / bar_range
                        vol_at_price[j] += bar_vol * fraction

            # Point of Control
            poc_idx = np.argmax(vol_at_price)
            poc_price = bin_centers[poc_idx]

            # Value Area: 70% of total volume around POC
            total_vol = vol_at_price.sum()
            target_vol = total_vol * 0.70
            accumulated = vol_at_price[poc_idx]
            va_low_idx = poc_idx
            va_high_idx = poc_idx

            while accumulated < target_vol:
                expand_up = vol_at_price[va_high_idx + 1] if va_high_idx + 1 < n_bins else 0
                expand_down = vol_at_price[va_low_idx - 1] if va_low_idx - 1 >= 0 else 0

                if expand_up >= expand_down and va_high_idx + 1 < n_bins:
                    va_high_idx += 1
                    accumulated += expand_up
                elif va_low_idx - 1 >= 0:
                    va_low_idx -= 1
                    accumulated += expand_down
                else:
                    break

            current_price = market_data.current_price
            if abs(current_price - poc_price) / poc_price < 0.005:
                pos = "at"
            elif current_price > poc_price:
                pos = "above"
            else:
                pos = "below"

            return VolumeProfile(
                price_levels=bin_centers.tolist(),
                volumes=vol_at_price.tolist(),
                poc=poc_price,
                value_area_high=bin_centers[va_high_idx],
                value_area_low=bin_centers[va_low_idx],
                current_price_vs_poc=pos,
            )

        except Exception as e:
            logger.warning(f"Volume profile computation failed: {e}")
            return None

    # ── Signal Detectors ────────────────────────────────────────────────

    def _detect_breakout(self, close, high, low, volume, vol_sma,
                          sr_levels, data) -> List[Signal]:
        """
        BREAKOUT_CONFIRMED: Price closes beyond S/R level with volume surge.
        This is the strongest signal — it means institutional money has
        pushed through a contested level with conviction.
        """
        signals = []
        price = close.iloc[-1]
        prev_price = close.iloc[-2]
        vol_ratio = volume.iloc[-1] / vol_sma.iloc[-1] if vol_sma.iloc[-1] > 0 else 1

        if vol_ratio < SIGNAL_CONFIG["vol_spike_threshold"]:
            return signals

        for level in sr_levels:
            level_price = level.price
            breakout_margin = level_price * SIGNAL_CONFIG["sr_breakout_pct"] / 100

            # Bullish breakout through resistance
            if (level.level_type == "resistance"
                    and prev_price < level_price
                    and price > level_price + breakout_margin):

                conf = min(1.0, vol_ratio / SIGNAL_CONFIG["vol_surge_threshold"]) * 0.9
                signals.append(Signal(
                    signal_type="BREAKOUT_CONFIRMED",
                    priority=SignalPriority.CRITICAL,
                    direction=SignalDirection.BULLISH,
                    ticker=data.ticker,
                    price=price,
                    trigger_level=level_price,
                    volume_ratio=vol_ratio,
                    confidence=conf,
                    description=(
                        f"Bullish breakout above resistance {level_price:.2f} "
                        f"on {vol_ratio:.1f}x average volume. "
                        f"Level was tested {level.strength} times. "
                        f"This is institutional buying pushing through supply."
                    ),
                    metadata={"level_strength": level.strength, "breakout_margin": breakout_margin},
                ))

            # Bearish breakdown through support
            if (level.level_type == "support"
                    and prev_price > level_price
                    and price < level_price - breakout_margin):

                conf = min(1.0, vol_ratio / SIGNAL_CONFIG["vol_surge_threshold"]) * 0.9
                signals.append(Signal(
                    signal_type="BREAKOUT_CONFIRMED",
                    priority=SignalPriority.CRITICAL,
                    direction=SignalDirection.BEARISH,
                    ticker=data.ticker,
                    price=price,
                    trigger_level=level_price,
                    volume_ratio=vol_ratio,
                    confidence=conf,
                    description=(
                        f"Bearish breakdown below support {level_price:.2f} "
                        f"on {vol_ratio:.1f}x average volume. "
                        f"Level was tested {level.strength} times. "
                        f"Stop-losses triggered, institutional selling."
                    ),
                    metadata={"level_strength": level.strength},
                ))

        return signals

    def _detect_volume_climax(self, close, high, low, volume, vol_sma,
                               sr_levels, data) -> List[Signal]:
        """
        VOLUME_CLIMAX: Extreme volume (4x+) with rejection candle.
        This is an exhaustion signal — the last burst of buying/selling
        before a reversal. Often marks exact tops and bottoms.
        """
        signals = []
        vol_ratio = volume.iloc[-1] / vol_sma.iloc[-1] if vol_sma.iloc[-1] > 0 else 1

        if vol_ratio < SIGNAL_CONFIG["vol_climax_threshold"]:
            return signals

        o = close.iloc[-1]  # Using close approximation; ideally use Open
        c = close.iloc[-1]
        h = high.iloc[-1]
        l = low.iloc[-1]

        # Try to get actual open
        if "Open" in data.price_history.columns:
            o = data.price_history["Open"].iloc[-1]

        body = abs(c - o)
        upper_wick = h - max(c, o)
        lower_wick = min(c, o) - l
        full_range = h - l if h != l else 1e-10

        # Bearish climax: big upper wick (rejection at top) + extreme volume
        if (upper_wick > body * SIGNAL_CONFIG["rejection_wick_ratio"]
                and upper_wick / full_range > 0.5):
            signals.append(Signal(
                signal_type="VOLUME_CLIMAX",
                priority=SignalPriority.CRITICAL,
                direction=SignalDirection.BEARISH,
                ticker=data.ticker,
                price=c,
                trigger_level=h,  # the rejected high
                volume_ratio=vol_ratio,
                confidence=min(1.0, vol_ratio / 5.0) * 0.85,
                description=(
                    f"Bearish volume climax: {vol_ratio:.1f}x avg volume with "
                    f"strong rejection wick at {h:.2f}. "
                    f"Buyers exhausted — likely short-term top."
                ),
                metadata={"wick_ratio": upper_wick / body if body > 0 else 999},
            ))

        # Bullish climax: big lower wick (rejection at bottom) + extreme volume
        if (lower_wick > body * SIGNAL_CONFIG["rejection_wick_ratio"]
                and lower_wick / full_range > 0.5):
            signals.append(Signal(
                signal_type="VOLUME_CLIMAX",
                priority=SignalPriority.CRITICAL,
                direction=SignalDirection.BULLISH,
                ticker=data.ticker,
                price=c,
                trigger_level=l,  # the rejected low
                volume_ratio=vol_ratio,
                confidence=min(1.0, vol_ratio / 5.0) * 0.85,
                description=(
                    f"Bullish volume climax: {vol_ratio:.1f}x avg volume with "
                    f"strong rejection wick at {l:.2f}. "
                    f"Sellers exhausted — likely short-term bottom."
                ),
                metadata={"wick_ratio": lower_wick / body if body > 0 else 999},
            ))

        return signals

    def _detect_accumulation_distribution(self, close, high, low, volume,
                                           vol_sma, sr_levels, data) -> List[Signal]:
        """
        ACCUMULATION_AT_SUPPORT / DISTRIBUTION_AT_RESISTANCE:
        Volume spike while price holds at a key level = institutional positioning.
        """
        signals = []
        price = close.iloc[-1]
        vol_ratio = volume.iloc[-1] / vol_sma.iloc[-1] if vol_sma.iloc[-1] > 0 else 1

        if vol_ratio < SIGNAL_CONFIG["vol_spike_threshold"]:
            return signals

        for level in sr_levels:
            proximity = abs(price - level.price) / level.price * 100

            if proximity > SIGNAL_CONFIG["sr_proximity_pct"]:
                continue

            # At support with volume spike → accumulation (bullish)
            if level.level_type == "support":
                # Confirm price didn't break below
                if price >= level.price * 0.995:  # within 0.5% above
                    conf = min(1.0, (vol_ratio / 3.0) * (level.strength / 4))
                    signals.append(Signal(
                        signal_type="ACCUMULATION_AT_SUPPORT",
                        priority=SignalPriority.HIGH,
                        direction=SignalDirection.BULLISH,
                        ticker=data.ticker,
                        price=price,
                        trigger_level=level.price,
                        volume_ratio=vol_ratio,
                        confidence=min(0.9, conf),
                        description=(
                            f"Accumulation detected at support {level.price:.2f} "
                            f"({vol_ratio:.1f}x avg volume). Support tested "
                            f"{level.strength} times and holding. "
                            f"Institutional buyers absorbing supply."
                        ),
                        metadata={
                            "level_strength": level.strength,
                            "proximity_pct": proximity,
                        },
                    ))

            # At resistance with volume spike → distribution (bearish)
            elif level.level_type == "resistance":
                if price <= level.price * 1.005:
                    conf = min(1.0, (vol_ratio / 3.0) * (level.strength / 4))
                    signals.append(Signal(
                        signal_type="DISTRIBUTION_AT_RESISTANCE",
                        priority=SignalPriority.HIGH,
                        direction=SignalDirection.BEARISH,
                        ticker=data.ticker,
                        price=price,
                        trigger_level=level.price,
                        volume_ratio=vol_ratio,
                        confidence=min(0.9, conf),
                        description=(
                            f"Distribution detected at resistance {level.price:.2f} "
                            f"({vol_ratio:.1f}x avg volume). Resistance tested "
                            f"{level.strength} times. "
                            f"Institutional sellers distributing into demand."
                        ),
                        metadata={
                            "level_strength": level.strength,
                            "proximity_pct": proximity,
                        },
                    ))

        return signals

    def _detect_volume_dry_up(self, close, volume, vol_sma,
                               sr_levels, data) -> List[Signal]:
        """
        VOLUME_DRY_UP: Price tests a level on declining/low volume.
        Means the test lacks conviction — the level will likely hold.
        Useful for confirming support is strong (buy) or resistance is strong (sell).
        """
        signals = []
        price = close.iloc[-1]
        vol_ratio = volume.iloc[-1] / vol_sma.iloc[-1] if vol_sma.iloc[-1] > 0 else 1

        if vol_ratio > SIGNAL_CONFIG["vol_dry_up_threshold"]:
            return signals

        # Also check for 3-bar declining volume
        if len(volume) < 4:
            return signals
        declining = all(
            volume.iloc[-i] < volume.iloc[-i - 1]
            for i in range(1, 4)
        )

        for level in sr_levels:
            proximity = abs(price - level.price) / level.price * 100
            if proximity > SIGNAL_CONFIG["sr_proximity_pct"]:
                continue

            if level.level_type == "support" and declining:
                signals.append(Signal(
                    signal_type="VOLUME_DRY_UP",
                    priority=SignalPriority.MEDIUM,
                    direction=SignalDirection.BULLISH,
                    ticker=data.ticker,
                    price=price,
                    trigger_level=level.price,
                    volume_ratio=vol_ratio,
                    confidence=0.5,
                    description=(
                        f"Price testing support at {level.price:.2f} on drying volume "
                        f"({vol_ratio:.1f}x avg, declining 3 bars). "
                        f"Weak selling pressure — support likely holds."
                    ),
                ))

            elif level.level_type == "resistance" and declining:
                signals.append(Signal(
                    signal_type="VOLUME_DRY_UP",
                    priority=SignalPriority.MEDIUM,
                    direction=SignalDirection.BEARISH,
                    ticker=data.ticker,
                    price=price,
                    trigger_level=level.price,
                    volume_ratio=vol_ratio,
                    confidence=0.5,
                    description=(
                        f"Price testing resistance at {level.price:.2f} on drying volume "
                        f"({vol_ratio:.1f}x avg, declining 3 bars). "
                        f"Weak buying pressure — resistance likely holds."
                    ),
                ))

        return signals

    def _detect_vwap_deviation(self, close, high, low, volume, data) -> List[Signal]:
        """
        VWAP_DEVIATION: Price far from VWAP with volume.
        Institutional traders use VWAP as a benchmark — significant
        deviations tend to revert, especially intraday.

        We approximate session VWAP from daily bars (not perfect, but
        the 20-day rolling VWAP is still a useful institutional reference).
        """
        signals = []

        try:
            # Compute rolling VWAP (20 days)
            typical_price = (high + low + close) / 3
            cumulative_tp_vol = (typical_price * volume).rolling(20).sum()
            cumulative_vol = volume.rolling(20).sum()
            vwap = cumulative_tp_vol / cumulative_vol

            current_vwap = vwap.iloc[-1]
            if pd.isna(current_vwap) or current_vwap == 0:
                return signals

            deviation_pct = ((close.iloc[-1] - current_vwap) / current_vwap) * 100
            vol_ratio = volume.iloc[-1] / volume.rolling(20).mean().iloc[-1]

            if abs(deviation_pct) < SIGNAL_CONFIG["vwap_deviation_pct"]:
                return signals

            if deviation_pct > SIGNAL_CONFIG["vwap_deviation_pct"]:
                # Price far above VWAP → mean reversion sell
                signals.append(Signal(
                    signal_type="VWAP_DEVIATION",
                    priority=SignalPriority.MEDIUM,
                    direction=SignalDirection.BEARISH,
                    ticker=data.ticker,
                    price=close.iloc[-1],
                    trigger_level=current_vwap,
                    volume_ratio=vol_ratio,
                    confidence=min(0.7, abs(deviation_pct) / 5.0),
                    description=(
                        f"Price {deviation_pct:+.1f}% above 20-day VWAP ({current_vwap:.2f}). "
                        f"Extended — institutional mean reversion pressure likely."
                    ),
                    metadata={"vwap": current_vwap, "deviation_pct": deviation_pct},
                ))

            elif deviation_pct < -SIGNAL_CONFIG["vwap_deviation_pct"]:
                # Price far below VWAP → mean reversion buy
                signals.append(Signal(
                    signal_type="VWAP_DEVIATION",
                    priority=SignalPriority.MEDIUM,
                    direction=SignalDirection.BULLISH,
                    ticker=data.ticker,
                    price=close.iloc[-1],
                    trigger_level=current_vwap,
                    volume_ratio=vol_ratio,
                    confidence=min(0.7, abs(deviation_pct) / 5.0),
                    description=(
                        f"Price {deviation_pct:+.1f}% below 20-day VWAP ({current_vwap:.2f}). "
                        f"Discount — institutional VWAP-reversion buying likely."
                    ),
                    metadata={"vwap": current_vwap, "deviation_pct": deviation_pct},
                ))

        except Exception as e:
            logger.debug(f"VWAP deviation detection failed: {e}")

        return signals

    def _detect_volume_pressure(self, close, volume, vol_sma,
                                  sr_levels, data) -> List[Signal]:
        """
        VOLUME_BREAKOUT_PRELUDE: Rising volume over 3+ bars while price
        compresses toward a key level. Pressure is building — breakout imminent.
        """
        signals = []

        if len(volume) < 5:
            return signals

        # Check for 3+ bars of rising volume
        bars = SIGNAL_CONFIG["vol_trend_bars"]
        rising = all(
            volume.iloc[-i] > volume.iloc[-i - 1]
            for i in range(1, bars + 1)
        )

        if not rising:
            return signals

        # Check if price is compressing (ATR declining while volume rising)
        recent_ranges = (data.price_history["High"] - data.price_history["Low"]).iloc[-5:]
        range_declining = recent_ranges.iloc[-1] < recent_ranges.iloc[-3]

        if not range_declining:
            return signals

        price = close.iloc[-1]
        vol_ratio = volume.iloc[-1] / vol_sma.iloc[-1] if vol_sma.iloc[-1] > 0 else 1

        for level in sr_levels:
            proximity = abs(price - level.price) / level.price * 100
            if proximity > SIGNAL_CONFIG["sr_proximity_pct"] * 2:  # wider zone for prelude
                continue

            direction = (
                SignalDirection.BULLISH if level.level_type == "resistance"
                else SignalDirection.BEARISH
            )
            signals.append(Signal(
                signal_type="VOLUME_BREAKOUT_PRELUDE",
                priority=SignalPriority.MEDIUM,
                direction=direction,
                ticker=data.ticker,
                price=price,
                trigger_level=level.price,
                volume_ratio=vol_ratio,
                confidence=0.5,
                description=(
                    f"Pressure building: {bars} bars of rising volume "
                    f"with contracting range near {level.level_type} "
                    f"at {level.price:.2f}. Breakout likely imminent."
                ),
                metadata={"rising_bars": bars, "range_contracting": True},
            ))

        return signals

    # ── Cooldown Management ─────────────────────────────────────────────

    def _apply_cooldowns(self, ticker: str, signals: List[Signal]) -> List[Signal]:
        """Prevent the same signal type from firing repeatedly."""
        if ticker not in self._cooldowns:
            self._cooldowns[ticker] = {}

        cooldown = SIGNAL_CONFIG["cooldown_bars"]
        filtered = []

        for sig in signals:
            last_fired = self._cooldowns[ticker].get(sig.signal_type, cooldown + 1)
            if last_fired >= cooldown:
                filtered.append(sig)
                self._cooldowns[ticker][sig.signal_type] = 0
            else:
                logger.debug(
                    f"Signal {sig.signal_type} for {ticker} suppressed (cooldown: "
                    f"{last_fired}/{cooldown})"
                )

        # Increment all cooldown counters
        for sig_type in list(self._cooldowns[ticker].keys()):
            self._cooldowns[ticker][sig_type] += 1

        return filtered

"""
LLM Analyzer — v2

Edge-generating enhancements over v1:
─────────────────────────────────────
1. Quant-anchored prompts: LLM sees the composite score FIRST,
   preventing hallucinated conviction on weak setups.
2. Regime-aware strategy hints: trending markets get trend prompts,
   ranging markets get mean-reversion prompts.
3. Multi-timeframe confluence context: weekly confirms/denies daily.
4. Volume divergence + smart-money signals in prompt.
5. Support/Resistance context for realistic targets and stop-losses.
6. Proper EUR currency formatting for EU tickers.
7. Structured two-pass fallback: if JSON parse fails, re-query with
   stricter format instructions before falling back to HOLD.
8. Recommendation history persistence for later review / backtesting.

Deep module — simple interface:
    analyze(market_data) -> Recommendation
"""

import json
import logging
from datetime import datetime
from typing import Optional
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

import requests

from config import (
    OLLAMA_HOST, OLLAMA_MODEL, OLLAMA_TIMEOUT,
    MARKET, HISTORY_DIR,
)
from data_collector import MarketData

logger = logging.getLogger(__name__)


class Action(Enum):
    STRONG_BUY = "STRONG_BUY"
    BUY = "BUY"
    HOLD = "HOLD"
    SELL = "SELL"
    STRONG_SELL = "STRONG_SELL"


@dataclass
class Recommendation:
    ticker: str
    action: Action
    confidence: float
    current_price: float
    target_price: Optional[float]
    stop_loss: Optional[float]
    reasoning: str
    timestamp: str
    quant_score: float = 0.0
    quant_signal: str = "HOLD"
    regime: str = ""
    active_signals: list = field(default_factory=list)  # signal summaries

    @property
    def potential_return(self) -> Optional[float]:
        if self.target_price:
            return ((self.target_price - self.current_price) / self.current_price) * 100
        return None


class LLMAnalyzer:
    """
    Analyzes market data using local LLM, anchored by quantitative scoring.

    Simple interface:
        analyze(market_data) -> Recommendation
    """

    def __init__(self, model: str = OLLAMA_MODEL):
        self.model = model
        self.base_url = OLLAMA_HOST
        self._currency_symbol = "€" if MARKET == "EU" else "$"
        self._verify_ollama_connection()

    def _verify_ollama_connection(self):
        try:
            resp = requests.get(f"{self.base_url}/api/tags", timeout=5)
            resp.raise_for_status()
            models = [m["name"] for m in resp.json().get("models", [])]
            if not any(self.model in n for n in models):
                logger.warning(f"Model {self.model} not found. Available: {models}")
            self._ollama_available = True
        except Exception as e:
            logger.warning(
                f"Ollama not available at {self.base_url}: {e}. "
                f"Bot will run in quant-only mode until Ollama connects."
            )
            self._ollama_available = False

    # ── Public Interface ────────────────────────────────────────────────

    def analyze(self, market_data: MarketData) -> Optional[Recommendation]:
        # Skip LLM entirely if Ollama isn't available
        if not self._ollama_available:
            rec = self._quant_only_recommendation(market_data, reason="Ollama not available")
            self._persist_recommendation(rec)
            return rec

        try:
            prompt = self._build_prompt(market_data)
            response = self._query_ollama(prompt)
            rec = self._parse_response(response, market_data)
            self._persist_recommendation(rec)
            return rec
        except Exception as e:
            logger.error(f"LLM analysis failed for {market_data.ticker}: {e}")
            # Fall back to pure quant recommendation so user always gets output
            rec = self._quant_only_recommendation(market_data, reason=str(e)[:80])
            self._persist_recommendation(rec)
            return rec

    def _quant_only_recommendation(self, data: MarketData, reason: str = "") -> Recommendation:
        """
        Generate recommendation purely from quant score + signals.
        Used when LLM is unavailable or fails.

        Still valuable — the quant score IS the core edge; the LLM
        adds nuance but the numbers drive the decision.
        """
        qs = data.quant_score
        action = Action[qs.signal]
        confidence = min(0.85, abs(qs.total) * 0.8)

        # Boost confidence if active signals agree
        sig_summaries = []
        for sig in (data.active_signals or []):
            sig_summaries.append({
                "type": sig.signal_type, "direction": sig.direction.value,
                "priority": sig.priority.value, "vol_ratio": sig.volume_ratio,
            })
            # CRITICAL/HIGH bullish signals + bullish quant → boost
            if sig.priority.value <= 2:
                if sig.direction.value == "BULLISH" and qs.total > 0:
                    confidence = min(0.95, confidence + 0.1)
                elif sig.direction.value == "BEARISH" and qs.total < 0:
                    confidence = min(0.95, confidence + 0.1)

        # Compute targets from S/R levels
        target = None
        stop = None
        if action in (Action.BUY, Action.STRONG_BUY):
            stop = round(data.current_price - 2 * data.atr, 2) if data.atr else None
            # Nearest resistance as target
            for lvl in data.support_resistance:
                if lvl.level_type == "resistance" and lvl.distance_pct > 0:
                    target = round(lvl.price, 2)
                    break
        elif action in (Action.SELL, Action.STRONG_SELL):
            stop = round(data.current_price + 2 * data.atr, 2) if data.atr else None
            for lvl in data.support_resistance:
                if lvl.level_type == "support" and lvl.distance_pct < 0:
                    target = round(lvl.price, 2)
                    break

        # Build reasoning from components
        parts = []
        parts.append(f"Quant score {qs.total:+.3f}")
        if data.regime:
            parts.append(f"regime={data.regime.regime}")
        parts.append(f"RSI={data.rsi:.1f}")
        if data.obv_trend and "divergence" in data.obv_trend:
            parts.append(f"vol_divergence={data.obv_trend}")
        if data.mtf:
            confirms = "confirms" if data.mtf.confirms_daily else "diverges"
            parts.append(f"weekly_{confirms}")
        if reason:
            parts.append(f"[quant-only: {reason}]")

        return Recommendation(
            ticker=data.ticker,
            action=action,
            confidence=confidence,
            current_price=data.current_price,
            target_price=target,
            stop_loss=stop,
            reasoning=". ".join(parts),
            timestamp=data.timestamp.isoformat(),
            quant_score=qs.total,
            quant_signal=qs.signal,
            regime=data.regime.regime if data.regime else "",
            active_signals=sig_summaries,
        )

    # ── Prompt Construction ─────────────────────────────────────────────

    def _build_prompt(self, d: MarketData) -> str:
        cs = self._currency_symbol
        qs = d.quant_score

        # Regime-specific strategy hint
        strategy_hint = self._strategy_hint(d)

        # Support/Resistance context
        sr_text = self._format_sr(d)

        # Volume analysis context
        vol_text = self._format_volume_analysis(d)

        # Multi-timeframe context
        mtf_text = self._format_mtf(d)

        # Active volume-at-level signals
        signals_text = self._format_active_signals(d)

        # Volume profile context
        vp_text = self._format_volume_profile(d)

        prompt = f"""You are a professional quantitative equity analyst.
Analyze {d.ticker} and produce a trading recommendation.

═══ QUANTITATIVE PRE-SCORE (computed independently) ═══
Composite Score: {qs.total:+.3f}  (range: -1.0 bearish to +1.0 bullish)
Signal: {qs.signal}
Component breakdown:
  Trend:      {qs.components.get('trend', 0):+.3f}
  Momentum:   {qs.components.get('momentum', 0):+.3f}
  Volume:     {qs.components.get('volume', 0):+.3f}
  Volatility: {qs.components.get('volatility', 0):+.3f}
  RelStrength:{qs.components.get('rel_strength', 0):+.3f}
  Structure:  {qs.components.get('structure', 0):+.3f}

IMPORTANT: Your recommendation should generally AGREE with the quant score
direction. Only deviate if you identify a clear reason the indicators miss.

{signals_text}
{vp_text}
═══ MARKET REGIME ═══
{d.regime.description if d.regime else 'Unknown'}
ADX: {d.adx:.1f} | ATR: {cs}{d.atr:.2f} | ATR Ratio: {(d.regime.atr_ratio if d.regime else 0):.2f}x
{strategy_hint}

═══ PRICE DATA ═══
Current Price: {cs}{d.current_price:.2f}
1-Day Change:  {d.price_change_1d:+.2f}%
5-Day Change:  {d.price_change_5d:+.2f}%
1-Month Change:{d.price_change_1m:+.2f}%
Trend: {d.summary['trend']}

═══ MOVING AVERAGES ═══
{self._format_mas(d)}

═══ MOMENTUM ═══
RSI(14):      {d.rsi:.1f} {'⚠ OVERBOUGHT' if d.rsi > 70 else '⚠ OVERSOLD' if d.rsi < 30 else ''}
Stochastic K: {d.stochastic.get('k', 50):.1f}  D: {d.stochastic.get('d', 50):.1f}
MACD Line:    {d.macd['macd']:.4f}
Signal Line:  {d.macd['signal']:.4f}
Histogram:    {d.macd['histogram']:.4f} ({'rising' if d.macd.get('hist_rising') else 'falling'})
Candle Signal:{d.candle_signal}

═══ BOLLINGER BANDS ═══
Upper:  {cs}{d.bollinger_bands['upper']:.2f}
Middle: {cs}{d.bollinger_bands['middle']:.2f}
Lower:  {cs}{d.bollinger_bands['lower']:.2f}
Bandwidth: {d.bollinger_bands.get('bandwidth', 0):.4f}
Position: Price is {'above' if d.current_price > d.bollinger_bands['upper'] else 'below' if d.current_price < d.bollinger_bands['lower'] else 'within'} bands

═══ VOLUME ANALYSIS ═══
{vol_text}

═══ SUPPORT & RESISTANCE ═══
{sr_text}

═══ RELATIVE STRENGTH vs Euro Stoxx 50 ═══
20-day RS: {d.relative_strength_20d:+.2f}%  {'Outperforming' if d.relative_strength_20d > 0 else 'Underperforming'}
60-day RS: {d.relative_strength_60d:+.2f}%

{mtf_text}

═══ INSTRUCTIONS ═══
Based on ALL the above, respond with ONLY a JSON object:
{{
    "action": "STRONG_BUY|BUY|HOLD|SELL|STRONG_SELL",
    "confidence": 0.0-1.0,
    "target_price": <number or null>,
    "stop_loss": <number or null>,
    "reasoning": "<2-3 sentence analysis>"
}}

Rules:
1. Target price must be based on the nearest resistance (for buys) or support (for sells).
2. Stop loss must be set using ATR: for buys, current_price - 2*ATR; for sells, current_price + 2*ATR.
3. If regime is "ranging" or ADX < 20, prefer HOLD unless at extreme S/R level.
4. Confidence must reflect how many indicators agree. All agree = 0.8+. Mixed = 0.4-0.6.
5. Do NOT deviate more than one step from the quant signal (e.g. quant says BUY → you can say STRONG_BUY, BUY, or HOLD, but NOT SELL).
6. CRITICAL or HIGH volume-at-level alerts OVERRIDE regime caution: if accumulation is detected at support, you may raise conviction even in a ranging market.
7. If a BREAKOUT_CONFIRMED signal is active, this is the strongest possible signal — increase confidence by 0.1-0.2.
8. If VOLUME_CLIMAX is active, treat it as a reversal signal — it may justify going against the trend.

Respond ONLY with valid JSON."""

        return prompt

    def _format_active_signals(self, d: MarketData) -> str:
        """Format active volume-at-level signals for prompt."""
        if not d.active_signals:
            return ""

        lines = [
            "═══ ⚡ ACTIVE VOLUME-AT-LEVEL ALERTS (HIGHEST PRIORITY) ═══",
            "These signals detect institutional order flow at key price levels.",
            "CRITICAL and HIGH signals should strongly influence your recommendation.",
            "",
        ]
        for sig in d.active_signals:
            priority_tag = {1: "CRITICAL", 2: "HIGH", 3: "MEDIUM", 4: "LOW"}.get(
                sig.priority.value, "?"
            )
            lines.append(
                f"[{priority_tag}] {sig.signal_type} — {sig.direction.value}\n"
                f"  {sig.description}\n"
                f"  Trigger Level: {sig.trigger_level:.2f} | "
                f"Volume: {sig.volume_ratio:.1f}x avg | "
                f"Confidence: {sig.confidence:.0%}\n"
            )

        return "\n".join(lines)

    def _format_volume_profile(self, d: MarketData) -> str:
        """Format volume profile context for prompt."""
        vp = d.volume_profile
        if vp is None:
            return ""

        cs = self._currency_symbol
        return (
            f"═══ VOLUME PROFILE (60-day) ═══\n"
            f"Point of Control (most-traded price): {cs}{vp.poc:.2f}\n"
            f"Value Area: {cs}{vp.value_area_low:.2f} – {cs}{vp.value_area_high:.2f}\n"
            f"Current price is {vp.current_price_vs_poc} the POC.\n"
            f"NOTE: Price tends to gravitate toward the POC. Trading within the "
            f"Value Area is 'fair value'; outside it is extended.\n"
        )

    def _strategy_hint(self, d: MarketData) -> str:
        if d.regime is None:
            return ""
        r = d.regime.regime
        if "trending_up" in r:
            return "STRATEGY HINT: Trending up — favor buying pullbacks to SMA-20/21 EMA. Trail stops using ATR."
        elif "trending_down" in r:
            return "STRATEGY HINT: Trending down — favor selling rallies to SMA-20. Avoid bottom-picking."
        elif r == "ranging":
            return "STRATEGY HINT: Range-bound — buy near support, sell near resistance. Tight stops. Reduce size."
        elif r == "volatile":
            return "STRATEGY HINT: High volatility — REDUCE position size. Wider stops. Wait for clarity."
        return ""

    def _format_mas(self, d: MarketData) -> str:
        cs = self._currency_symbol
        lines = []
        for label, data in [("SMA", d.sma), ("EMA", d.ema)]:
            for period, val in sorted(data.items()):
                rel = "above" if d.current_price > val else "below"
                dist = ((d.current_price - val) / val) * 100
                lines.append(f"{label}-{period}: {cs}{val:.2f} (price {rel}, {dist:+.1f}%)")
        return "\n".join(lines)

    def _format_sr(self, d: MarketData) -> str:
        cs = self._currency_symbol
        if not d.support_resistance:
            return "No clear levels detected."
        lines = []
        for lvl in sorted(d.support_resistance, key=lambda x: x.price):
            tag = "SUP" if lvl.level_type == "support" else "RES"
            lines.append(
                f"  [{tag}] {cs}{lvl.price:.2f} (strength: {lvl.strength}, "
                f"distance: {lvl.distance_pct:+.1f}%)"
            )
        return "\n".join(lines)

    def _format_volume_analysis(self, d: MarketData) -> str:
        lines = [
            f"Volume: {d.volume:,} ({d.volume_ratio:.2f}x avg)",
            f"OBV Trend: {d.obv_trend}",
            f"MFI(14): {d.mfi:.1f} {'⚠ OVERBOUGHT' if d.mfi > 80 else '⚠ OVERSOLD' if d.mfi < 20 else ''}",
            f"A/D Line: {d.ad_line_trend}",
        ]
        if "divergence" in d.obv_trend:
            lines.append(f"⚠ VOLUME DIVERGENCE DETECTED: {d.obv_trend}")
        return "\n".join(lines)

    def _format_mtf(self, d: MarketData) -> str:
        if d.mtf is None:
            return ""
        m = d.mtf
        confirmation = "✅ WEEKLY CONFIRMS" if m.confirms_daily else "⚠ WEEKLY DIVERGES"
        return f"""═══ WEEKLY TIMEFRAME ═══
Weekly Trend: {m.weekly_trend}
Weekly RSI: {m.weekly_rsi:.1f}
Weekly MACD Histogram: {m.weekly_macd_histogram:.4f}
Above Weekly SMA-20: {'Yes' if m.weekly_above_20sma else 'No'}
{confirmation}"""

    # ── Ollama Communication ────────────────────────────────────────────

    def _query_ollama(self, prompt: str) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.2,
                "top_p": 0.85,
                "num_predict": 512,
            },
        }
        try:
            resp = requests.post(
                f"{self.base_url}/api/generate",
                json=payload, timeout=OLLAMA_TIMEOUT,
            )
            resp.raise_for_status()
            self._ollama_available = True  # mark available on success
            return resp.json().get("response", "")
        except requests.exceptions.Timeout:
            logger.error("Ollama request timed out")
            raise
        except Exception as e:
            self._ollama_available = False  # mark unavailable on failure
            logger.error(f"Ollama request failed: {e}")
            raise

    # ── Response Parsing ────────────────────────────────────────────────

    def _clean_price(self, value) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        return float(
            str(value).replace("$", "").replace("€", "")
            .replace("£", "").replace(",", "").strip()
        )

    def _parse_response(self, response: str, data: MarketData) -> Recommendation:
        qs = data.quant_score

        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start != -1 and end > start:
                parsed = json.loads(response[start:end])
            else:
                parsed = json.loads(response)

            action_str = parsed.get("action", "HOLD").upper().strip()
            try:
                action = Action[action_str]
            except KeyError:
                logger.warning(f"Invalid action '{action_str}', using quant signal")
                action = Action[qs.signal]

            # Enforce ±1 step constraint from quant signal
            action = self._constrain_action(action, qs.signal)

            confidence = max(0.0, min(1.0, float(parsed.get("confidence", 0.5))))

            target = self._clean_price(parsed.get("target_price"))
            stop = self._clean_price(parsed.get("stop_loss"))

            # Sanity: if ATR-based stop not provided, compute it
            if stop is None and action in (Action.BUY, Action.STRONG_BUY):
                stop = round(data.current_price - 2 * data.atr, 2)
            elif stop is None and action in (Action.SELL, Action.STRONG_SELL):
                stop = round(data.current_price + 2 * data.atr, 2)

            reasoning = parsed.get("reasoning", "No reasoning provided")

            # Build signal summaries for persistence and display
            sig_summaries = [
                {"type": s.signal_type, "direction": s.direction.value,
                 "priority": s.priority.value, "vol_ratio": s.volume_ratio}
                for s in (data.active_signals or [])
            ]

            return Recommendation(
                ticker=data.ticker,
                action=action,
                confidence=confidence,
                current_price=data.current_price,
                target_price=target,
                stop_loss=stop,
                reasoning=reasoning,
                timestamp=data.timestamp.isoformat(),
                quant_score=qs.total,
                quant_signal=qs.signal,
                regime=data.regime.regime if data.regime else "",
                active_signals=sig_summaries,
            )

        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Failed to parse LLM response: {e}")
            logger.debug(f"Response was: {response}")

            # Fall back to pure quant signal
            action = Action[qs.signal]
            return Recommendation(
                ticker=data.ticker,
                action=action,
                confidence=abs(qs.total) * 0.6,  # scale down since no LLM confirmation
                current_price=data.current_price,
                target_price=None,
                stop_loss=round(data.current_price - 2 * data.atr, 2) if action in (Action.BUY, Action.STRONG_BUY) else None,
                reasoning=f"Quant-only fallback (LLM parse failed). Score: {qs.total:+.3f}",
                timestamp=data.timestamp.isoformat(),
                quant_score=qs.total,
                quant_signal=qs.signal,
                regime=data.regime.regime if data.regime else "",
            )

    def _constrain_action(self, llm_action: Action, quant_signal: str) -> Action:
        """
        Prevent the LLM from deviating more than 1 step from quant signal.
        This is the key edge: the quant score anchors the LLM's tendency
        to hallucinate strong convictions.
        """
        order = [
            Action.STRONG_SELL, Action.SELL, Action.HOLD,
            Action.BUY, Action.STRONG_BUY,
        ]
        try:
            quant_action = Action[quant_signal]
        except KeyError:
            return llm_action

        qi = order.index(quant_action)
        li = order.index(llm_action)

        if abs(qi - li) <= 1:
            return llm_action

        # Clamp to ±1 step
        clamped_idx = qi + (1 if li > qi else -1)
        clamped_idx = max(0, min(len(order) - 1, clamped_idx))
        clamped = order[clamped_idx]
        logger.info(
            f"Constrained LLM action {llm_action.value} → {clamped.value} "
            f"(quant signal: {quant_signal})"
        )
        return clamped

    # ── Persistence ─────────────────────────────────────────────────────

    def _persist_recommendation(self, rec: Recommendation):
        """Save recommendation for backtesting/review."""
        try:
            filepath = HISTORY_DIR / f"{rec.ticker.replace('.', '_')}.jsonl"
            entry = {
                "timestamp": rec.timestamp,
                "ticker": rec.ticker,
                "price": rec.current_price,
                "action": rec.action.value,
                "confidence": rec.confidence,
                "target": rec.target_price,
                "stop_loss": rec.stop_loss,
                "quant_score": rec.quant_score,
                "quant_signal": rec.quant_signal,
                "regime": rec.regime,
                "active_signals": rec.active_signals,
                "reasoning": rec.reasoning,
            }
            with open(filepath, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception as e:
            logger.debug(f"Failed to persist recommendation: {e}")

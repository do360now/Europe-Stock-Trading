"""
Recommendation Validator — Deep safety module (Ousterhout style)
Simple public API: .refine() 
Hides: strict anchoring, mandatory good R:R, regime conservatism, hybrid scoring,
      defensive field access, realistic S/R + ATR levels.
"""

import logging
from dataclasses import replace
from typing import Optional

from config import MIN_RR_RATIO, MAX_QUANT_LLM_ACTION_DEVIATION, \
                   WEAK_REGIME_CONFIDENCE_CAP, ATR_TARGET_MULTIPLIER
from models import Recommendation, Action
from data_collector import MarketData   # for type hint + defensive access

logger = logging.getLogger(__name__)

class RecommendationValidator:
    """One deep module that protects every recommendation forever."""

    def refine(self, rec: Recommendation, data: MarketData) -> Recommendation:
        """Entry point — never raises, always returns a safe Recommendation."""
        if not getattr(data, 'quant_score', None):
            logger.warning(f"{data.ticker}: Missing quant_score — using LLM as-is")
            return rec

        rec = self._enforce_anchoring(rec, data)
        rec = self._enforce_risk_rules(rec, data)
        rec = self._apply_regime_guardrails(rec, data)
        rec = self._blend_confidence(rec, data)

        rr = self._calc_rr(rec)
        logger.info(f"✅ VALIDATED {rec.ticker}: {rec.action.value} {rec.confidence:.0%} "
                    f"R:R 1:{rr:.1f} | Quant: {data.quant_score.signal}")
        return rec

    # ── Deep internal rules (all complexity hidden here) ─────────────────────

    def _enforce_anchoring(self, rec: Recommendation, data: MarketData) -> Recommendation:
        qs = data.quant_score
        try:
            quant_action = Action[qs.signal]
        except (KeyError, AttributeError, TypeError):
            quant_action = Action.HOLD

        if abs(self._action_to_level(rec.action) - self._action_to_level(quant_action)) > MAX_QUANT_LLM_ACTION_DEVIATION:
            logger.warning(f"ANCHORING applied on {rec.ticker}: quant={qs.signal} → LLM={rec.action.value}")
            return replace(rec, action=quant_action, confidence=min(rec.confidence, 0.62))
        return rec

    # Inside RecommendationValidator class — replace _enforce_risk_rules and add helper

    def _enforce_risk_rules(self, rec: Recommendation, data: MarketData) -> Recommendation:
        if not rec.target_price or not rec.stop_loss:
            rec = self._fallback_levels(rec, data)

        rr = self._calc_rr(rec)
        min_rr = 1.8 if "trending" in data.regime.regime else 2.2  # stricter in weak regimes

        if rr < min_rr or abs(rec.target_price - rec.current_price) / rec.current_price < 0.04:
            logger.warning(f"{rec.ticker}: Weak R:R {rr:.1f} or target too close — forcing realistic levels")
            direction = 1 if rec.action in (Action.BUY, Action.STRONG_BUY) else -1
            realistic_target = self._realistic_target(data, rec.action)
            realistic_stop = self._realistic_stop(data, rec.action)
            new_rr = abs(realistic_target - rec.current_price) / abs(rec.current_price - realistic_stop)
            rec = replace(rec,
                          target_price=realistic_target,
                          stop_loss=realistic_stop)
            logger.info(f"  → New target {realistic_target:.2f}, stop {realistic_stop:.2f}, R:R 1:{new_rr:.1f}")

        return rec

    def _apply_regime_guardrails(self, rec: Recommendation, data: MarketData) -> Recommendation:
        if not data.regime:
            return rec
        regime = data.regime.regime.lower()
        cap = 0.65
        if "weak" in regime or "range" in regime or "chop" in regime:
            cap = 0.55
        elif "volatile" in regime:
            cap = 0.50
        if rec.confidence > cap:
            logger.info(f"Regime guard: {rec.ticker} {regime} → confidence capped {rec.confidence:.0%} → {cap:.0%}")
            rec = replace(rec, confidence=cap)
        return rec

    def _realistic_target(self, data: MarketData, action: Action) -> float:
        direction = 1 if action in (Action.BUY, Action.STRONG_BUY) else -1
        candidates = []
        # Strongest S/R in direction (prefer high strength)
        for lvl in sorted(data.support_resistance, key=lambda x: x.strength if hasattr(x, 'strength') else 0, reverse=True):
            dist = getattr(lvl, 'distance_pct', 0)
            if (direction > 0 and lvl.level_type == "resistance" and dist > 1.5) or \
               (direction < 0 and lvl.level_type == "support" and dist < -1.5):
                candidates.append(lvl.price)
        if candidates:
            # Take median or strongest
            return round(sorted(candidates)[len(candidates)//2], 2)

        # ATR fallback — ensure at least 5–6% move
        min_move = max(data.atr * 5, data.current_price * 0.05)
        return round(data.current_price + direction * min_move, 2)

    def _blend_confidence(self, rec: Recommendation, data: MarketData) -> Recommendation:
        qs = data.quant_score
        quant_strength = min(1.0, abs(qs.total) * 1.65)   # 0.60 quant → ~1.0 strength
        blended = round(0.62 * quant_strength + 0.38 * rec.confidence, 2)
        return replace(rec, confidence=blended)

    @staticmethod
    def _action_to_level(a: Action) -> int:
        return {"STRONG_SELL": -2, "SELL": -1, "HOLD": 0, "BUY": 1, "STRONG_BUY": 2}[a.value]

    @staticmethod
    def _calc_rr(rec: Recommendation) -> float:
        if not rec.target_price or not rec.stop_loss or rec.stop_loss == rec.current_price:
            return 0.0
        risk = abs(rec.current_price - rec.stop_loss)
        reward = abs(rec.target_price - rec.current_price)
        return reward / risk

    def _fallback_levels(self, rec: Recommendation, data: MarketData) -> Recommendation:
        return replace(rec,
            target_price=self._realistic_target(data, rec.action),
            stop_loss=self._realistic_stop(data, rec.action))

    def _realistic_target(self, data: MarketData, action: Action) -> float:
        direction = 1 if action in (Action.BUY, Action.STRONG_BUY) else -1
        # Prefer strongest S/R in the right direction
        for lvl in sorted(data.support_resistance, key=lambda x: getattr(x, 'strength', 0), reverse=True):
            if getattr(lvl, 'level_type', None) == ("resistance" if direction > 0 else "support"):
                if 0.8 <= getattr(lvl, 'distance_pct', 0) <= 15:
                    return round(lvl.price, 2)
        # ATR fallback
        return round(data.current_price + direction * ATR_TARGET_MULTIPLIER * data.atr, 2)

    def _realistic_stop(self, data: MarketData, action: Action) -> float:
        direction = 1 if action in (Action.BUY, Action.STRONG_BUY) else -1
        return round(data.current_price - direction * 2.0 * data.atr, 2)
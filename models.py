# models.py  ←  REPLACE ENTIRE FILE WITH THIS (deepened with computed properties)

"""Core data models — single source of truth (Ousterhout deep module)
All derived fields (potential_return, rr_ratio, etc.) are computed here so
display.py and validator never break on missing attributes.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List, Dict
from datetime import datetime

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
    target_price: Optional[float] = None
    stop_loss: Optional[float] = None
    reasoning: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    quant_score: float = 0.0
    quant_signal: str = "HOLD"
    regime: str = ""
    active_signals: List[Dict] = field(default_factory=list)

    # ── Computed properties (deep module magic — no breakage ever) ─────────────
    @property
    def potential_return(self) -> float:
        """Exactly what display.py expects. Auto-calculated, never missing."""
        if self.target_price and self.current_price:
            return round(((self.target_price - self.current_price) / self.current_price) * 100, 1)
        return 0.0

    @property
    def rr_ratio(self) -> float:
        """Bonus: R:R for future use (validator already uses it internally)."""
        if self.target_price and self.stop_loss and self.stop_loss != self.current_price:
            reward = abs(self.target_price - self.current_price)
            risk = abs(self.current_price - self.stop_loss)
            return round(reward / risk, 1)
        return 0.0
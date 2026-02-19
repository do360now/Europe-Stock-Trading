"""
Terminal Display Module — v2

Enhancements:
- Quant score visual bar alongside LLM recommendation
- Regime indicator per ticker
- Relative strength column
- Support/Resistance levels in recommendation cards
- Cleaner box drawing with proper width handling
"""

import logging
import os
from typing import Dict
from datetime import datetime

from llm_analyzer import Recommendation, Action
from data_collector import MarketData

logger = logging.getLogger(__name__)


class Display:
    def __init__(self, width: int = 130, currency: str = "EUR"):
        self.width = width
        self.currency = currency

    def clear_screen(self):
        os.system("clear" if os.name == "posix" else "cls")

    def _fc(self, amount: float) -> str:
        """Format currency."""
        symbols = {"EUR": "€", "GBP": "£"}
        sym = symbols.get(self.currency, "$")
        return f"{sym}{amount:.2f}"

    def show_header(self):
        w = self.width
        print("=" * w)
        title = "STOCK ANALYSIS BOT v2"
        subtitle = "Quant-Anchored LLM Analysis | Multi-Timeframe | Regime-Aware"
        print(f"{title:^{w}}")
        print(f"{subtitle:^{w}}")
        print("=" * w)
        print(f"  Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * w)
        print()

    def show_market_summary(self, market_data: Dict[str, MarketData]):
        if not market_data:
            return
        w = self.width
        print(f"┌{'─' * (w - 2)}┐")
        print(f"│{'MARKET SNAPSHOT':^{w - 2}}│")
        print(f"├{'─' * (w - 2)}┤")

        header = (
            f"│ {'Ticker':6s} │ {'Price':>9s} │ {'1D':>7s} │ {'5D':>7s} │ "
            f"{'1M':>7s} │ {'RSI':>5s}  │ {'Stoch':>5s} │ {'ADX':>4s} │ "
            f"{'Regime':12s} │ {'RS20':>6s} │ {'QScore':>7s} │ {'Signal':11s} │"
        )
        print(header)
        print(f"├{'─' * (w - 2)}┤")

        for ticker, d in market_data.items():
            display_ticker = ticker.split(".")[0] if "." in ticker else ticker
            rsi_ind = self._rsi_indicator(d.rsi)
            regime_short = (d.regime.regime if d.regime else "?")[:12]
            qs = d.quant_score
            qs_val = f"{qs.total:+.2f}" if qs else "  N/A"
            qs_sig = qs.signal if qs else "?"

            # Color the quant score
            if qs and qs.total > 0.3:
                qs_color = "\033[92m"
            elif qs and qs.total < -0.3:
                qs_color = "\033[91m"
            else:
                qs_color = "\033[93m"
            reset = "\033[0m"

            print(
                f"│ {display_ticker:6s} │ {self._fc(d.current_price):>9s} │ "
                f"{self._format_change(d.price_change_1d)} │ "
                f"{self._format_change(d.price_change_5d)} │ "
                f"{self._format_change(d.price_change_1m)} │ "
                f"{d.rsi:>5.1f}{rsi_ind} │ "
                f"{d.stochastic.get('k', 0):>5.1f} │ "
                f"{d.adx:>4.0f} │ "
                f"{regime_short:12s} │ "
                f"{d.relative_strength_20d:>+5.1f}% │ "
                f"{qs_color}{qs_val:>7s}{reset} │ "
                f"{qs_color}{qs_sig:11s}{reset} │"
            )

        print(f"└{'─' * (w - 2)}┘")
        print()

    def show_recommendations(self, recommendations: Dict[str, Recommendation]):
        if not recommendations:
            print("  No recommendations available yet...")
            return
        for _, rec in recommendations.items():
            self._show_single(rec)
            print()

    def show_signal_alerts(self, market_data: Dict[str, MarketData]):
        """Display active volume-at-level alerts — the institutional footprint panel."""
        all_signals = []
        for ticker, data in market_data.items():
            for sig in (data.active_signals or []):
                all_signals.append(sig)

        if not all_signals:
            return

        # Sort by priority
        all_signals.sort(key=lambda s: s.priority.value)

        w = self.width
        print(f"┌{'─' * (w - 2)}┐")
        print(f"│{'⚡ VOLUME-AT-LEVEL ALERTS ⚡':^{w - 2}}│")
        print(f"├{'─' * (w - 2)}┤")

        priority_colors = {
            1: "\033[91m",  # CRITICAL = red
            2: "\033[93m",  # HIGH = yellow
            3: "\033[94m",  # MEDIUM = blue
            4: "\033[90m",  # LOW = gray
        }
        reset = "\033[0m"

        for sig in all_signals:
            color = priority_colors.get(sig.priority.value, reset)
            display_ticker = sig.ticker.split(".")[0] if "." in sig.ticker else sig.ticker

            priority_tag = {1: "CRIT", 2: "HIGH", 3: "MED ", 4: "LOW "}.get(
                sig.priority.value, "?   "
            )
            dir_icon = "▲" if sig.direction.value == "BULLISH" else "▼" if sig.direction.value == "BEARISH" else "●"

            line = (
                f"│ {color}[{priority_tag}]{reset} "
                f"{display_ticker:6s} "
                f"{color}{dir_icon} {sig.signal_type:28s}{reset} "
                f"│ Vol: {sig.volume_ratio:>4.1f}x "
                f"│ Level: {self._fc(sig.trigger_level):>10s} "
                f"│ Conf: {sig.confidence:.0%}"
            )
            print(line)

            # Description on next line, indented
            desc = sig.description[:w - 12]
            print(f"│   └─ {desc}")

        print(f"└{'─' * (w - 2)}┘")
        print()

    def show_volume_profile_summary(self, market_data: Dict[str, MarketData]):
        """Show compact volume profile info per ticker."""
        has_profiles = any(d.volume_profile for d in market_data.values())
        if not has_profiles:
            return

        w = self.width
        print(f"┌{'─' * (w - 2)}┐")
        print(f"│{'VOLUME PROFILE (60-day)':^{w - 2}}│")
        print(f"├{'─' * (w - 2)}┤")

        header = (
            f"│ {'Ticker':6s} │ {'POC':>10s} │ "
            f"{'Value Area':^23s} │ {'Price vs POC':14s} │"
        )
        print(header)
        print(f"├{'─' * (w - 2)}┤")

        for ticker, data in market_data.items():
            vp = data.volume_profile
            if vp is None:
                continue
            display_ticker = ticker.split(".")[0] if "." in ticker else ticker
            va_str = f"{self._fc(vp.value_area_low)} – {self._fc(vp.value_area_high)}"
            print(
                f"│ {display_ticker:6s} │ {self._fc(vp.poc):>10s} │ "
                f"{va_str:^23s} │ {vp.current_price_vs_poc:14s} │"
            )

        print(f"└{'─' * (w - 2)}┘")
        print()

    def _show_single(self, rec: Recommendation):
        w = self.width
        action_colors = {
            Action.STRONG_BUY: "\033[92m",
            Action.BUY: "\033[94m",
            Action.HOLD: "\033[93m",
            Action.SELL: "\033[91m",
            Action.STRONG_SELL: "\033[95m",
        }
        reset = "\033[0m"
        color = action_colors.get(rec.action, reset)

        display_ticker = rec.ticker.split(".")[0] if "." in rec.ticker else rec.ticker

        # Header
        print(f"┌{'─' * (w - 2)}┐")
        line1 = (
            f"│ {color}{display_ticker:6s}{reset} │ "
            f"Action: {color}{rec.action.value:12s}{reset} │ "
            f"Confidence: {self._confidence_bar(rec.confidence)} {rec.confidence:.0%} │ "
            f"Quant: {rec.quant_score:+.2f} ({rec.quant_signal}) │ "
            f"Regime: {rec.regime}"
        )
        # Pad to width (ignoring ANSI codes for length calc)
        print(f"{line1}")
        print(f"├{'─' * (w - 2)}┤")

        # Price info
        price_line = f"│  Price: {self._fc(rec.current_price):>10s}"
        if rec.target_price:
            pot = rec.potential_return
            price_line += f"  │  Target: {self._fc(rec.target_price):>10s}  ({pot:+.1f}%)"
        if rec.stop_loss:
            risk = ((rec.stop_loss - rec.current_price) / rec.current_price) * 100
            price_line += f"  │  Stop: {self._fc(rec.stop_loss):>10s}  ({risk:+.1f}%)"

        # Risk/Reward ratio
        if rec.target_price and rec.stop_loss:
            reward = abs(rec.target_price - rec.current_price)
            risk_amt = abs(rec.current_price - rec.stop_loss)
            if risk_amt > 0:
                rr = reward / risk_amt
                price_line += f"  │  R:R = 1:{rr:.1f}"

        print(price_line)

        if rec.rr_ratio > 0 and rec.rr_ratio < 1.8:
            print(f"│  ⚠ LOW R:R ({rec.rr_ratio:.1f}) — risk management suboptimal")

        # Active signals for this ticker
        if rec.active_signals:
            print(f"├{'─' * (w - 2)}┤")
            sig_parts = []
            for s in rec.active_signals:
                dir_icon = "▲" if s["direction"] == "BULLISH" else "▼"
                sig_parts.append(f"{dir_icon}{s['type']}({s['vol_ratio']:.1f}x)")
            sig_line = "│  ⚡ TRIGGERS: " + "  ".join(sig_parts)
            print(sig_line)

        if abs(rec.quant_score) < 0.3 and rec.action in (Action.BUY, Action.SELL):
            print(f"│  ⚠️  QUANT/LLM DIVERGENCE — Quant says HOLD")

        # Reasoning
        print(f"├{'─' * (w - 2)}┤")
        self._print_wrapped(f"│  REASONING: {rec.reasoning}", w)
        print(f"└{'─' * (w - 2)}┘")

    def _print_wrapped(self, text: str, width: int):
        words = text.split()
        line = ""
        for word in words:
            if len(line) + len(word) + 1 > width - 3:
                print(f"{line:<{width - 1}}│")
                line = "│  " + word + " "
            else:
                line += word + " " if line else "│  " + word + " "
        if line:
            print(f"{line:<{width - 1}}│")

    def _confidence_bar(self, confidence: float) -> str:
        filled = int(confidence * 10)
        return f"[{'█' * filled}{'░' * (10 - filled)}]"

    def _format_change(self, change: float) -> str:
        if change > 0:
            return f"\033[92m+{change:>5.1f}%\033[0m"
        elif change < 0:
            return f"\033[91m{change:>6.1f}%\033[0m"
        return f"{change:>6.1f}%"

    @staticmethod
    def _rsi_indicator(rsi: float) -> str:
        if rsi > 70:
            return "🔥"
        elif rsi < 30:
            return "❄️"
        return "  "


def print_startup_banner():
    banner = """
╔═══════════════════════════════════════════════════════════════════════════════╗
║                                                                               ║
║                        🤖 STOCK ANALYSIS BOT v2 🤖                          ║
║                                                                               ║
║              Quant-Anchored · Multi-Timeframe · Regime-Aware                 ║
║                                                                               ║
║    • 30+ technical indicators (RSI, MACD, Stoch, ADX, ATR, OBV, MFI...)    ║
║    • Quantitative pre-scoring anchors LLM recommendations                    ║
║    • Weekly timeframe confluence confirmation                                ║
║    • Relative strength vs Euro Stoxx 50                                      ║
║    • Support/Resistance-based targets and ATR-based stops                    ║
║    • Regime detection: trending vs ranging vs volatile                        ║
║    • Recommendation history for backtesting                                  ║
║                                                                               ║
╚═══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

"""
Stock Analysis Bot — v2 Main Orchestrator

Enhancements:
- Collects benchmark data once per cycle for relative strength
- Tracks quant/LLM agreement rate for self-monitoring
- Updates weekly MTF confirmation flag after daily analysis
- Persists all recommendations to JSONL for future backtesting
"""

import logging
import time
import signal
import sys
from typing import Dict, Optional
from datetime import datetime

from config import (
    TICKERS, UPDATE_INTERVAL, LOG_LEVEL, LOG_FORMAT,
    LOGS_DIR, MARKET, TRADING212_API_KEY,
    AUTO_TRADE_MODE, MIN_TRADE_CONFIDENCE, TRADEABLE_ACTIONS,
    RISK_PCT_PER_TRADE, MAX_POSITION_PCT, TRADING212_MODE,
)
from data_collector import DataCollector, MarketData
from llm_analyzer import LLMAnalyzer
from models import Recommendation, Action
from display import Display, print_startup_banner
from signal_detector import SignalDetector
from broker_interface import (
    Trading212Broker, PaperTradingBroker, create_broker,
    OrderResult, OrderType,
)
import config

logging.basicConfig(
    level=LOG_LEVEL,
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOGS_DIR / f"bot_{datetime.now().strftime('%Y%m%d')}.log"),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


class StockAnalysisBot:
    """
    Main orchestrator.

    Flow per iteration:
    1. Collect market data + all indicators for each ticker
    2. Compute quant composite score (inside DataCollector)
    3. Send enriched data to LLM for recommendation
    4. LLM is constrained to ±1 step from quant signal
    5. Display dashboard
    6. Persist recommendations for backtesting
    """

    def __init__(self, tickers: list[str]):
        self.tickers = tickers
        self.running = False

        logger.info("Initializing bot modules...")
        self.data_collector = DataCollector()
        self.llm_analyzer = LLMAnalyzer()
        self.signal_detector = SignalDetector()

        # Broker: connect if API keys are present
        self.broker = None
        self.auto_trade = AUTO_TRADE_MODE  # "off", "confirm", "auto"
        self._trade_results: Dict[str, OrderResult] = {}

        if TRADING212_API_KEY and self.auto_trade != "off":
            broker_type = f"trading212_{TRADING212_MODE}"
            self.broker = create_broker(
                broker_type,
                risk_pct_per_trade=RISK_PCT_PER_TRADE,
                max_position_pct=MAX_POSITION_PCT,
            )
            if not self.broker.connect():
                logger.warning("Broker connection failed — trading disabled")
                self.broker = None
        elif self.auto_trade != "off":
            logger.info("No T212 API key — using paper trading")
            self.broker = PaperTradingBroker()
            self.broker.connect()

        currency = "EUR" if MARKET == "EU" else "USD"
        self.display = Display(currency=currency)

        self.market_data: Dict[str, MarketData] = {}
        self.recommendations: Dict[str, Recommendation] = {}

        # Self-monitoring: track quant/LLM agreement
        self._agreement_count = 0
        self._total_count = 0

        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        logger.info(f"Bot initialized for tickers: {', '.join(tickers)}")

    def _signal_handler(self, signum, frame):
        logger.info("Shutdown signal received")
        self.running = False

    def start(self):
        logger.info("Starting Stock Analysis Bot v2")
        self.running = True

        print_startup_banner()
        print(f"\n  Monitoring: {', '.join(self.tickers)}")
        print(f"  Update Interval: {UPDATE_INTERVAL}s")
        if self.broker:
            mode_label = "PAPER" if isinstance(self.broker, PaperTradingBroker) else TRADING212_MODE.upper()
            print(f"  Broker: Trading212 ({mode_label}) | Auto-trade: {self.auto_trade}")
        else:
            print(f"  Broker: None (analysis only)")
        print(f"  Press Ctrl+C to stop\n")
        time.sleep(3)

        iteration = 0
        while self.running:
            try:
                iteration += 1
                logger.info(f"=== Iteration {iteration} ===")

                self._collect_market_data()
                self._detect_signals()
                self._update_mtf_confirmation()
                self._analyze_all()
                self._execute_trades()
                self._update_display(iteration)

                if self.running:
                    logger.info(f"Sleeping {UPDATE_INTERVAL}s")
                    time.sleep(UPDATE_INTERVAL)

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(10)

        self._shutdown()

    def _collect_market_data(self):
        logger.info("Collecting market data...")
        for ticker in self.tickers:
            try:
                data = self.data_collector.get_analysis(ticker)
                if data:
                    self.market_data[ticker] = data
                    qs = data.quant_score
                    logger.info(
                        f"  {ticker}: {self.display._fc(data.current_price)} "
                        f"RSI={data.rsi:.1f} ADX={data.adx:.0f} "
                        f"QScore={qs.total:+.3f} ({qs.signal})"
                    )
                else:
                    logger.warning(f"  No data for {ticker}")
            except Exception as e:
                logger.error(f"  Failed {ticker}: {e}")

    def _detect_signals(self):
        """Run volume-at-level signal detection on all tickers."""
        logger.info("Running signal detection...")
        for ticker, data in self.market_data.items():
            try:
                signals = self.signal_detector.detect_signals(data)
                vol_profile = self.signal_detector.get_volume_profile(data)

                data.active_signals = signals
                data.volume_profile = vol_profile

                if signals:
                    for sig in signals:
                        logger.info(
                            f"  ⚡ {ticker}: {sig.priority_label} {sig.signal_type} "
                            f"({sig.direction.value}) — {sig.description[:80]}"
                        )
            except Exception as e:
                logger.error(f"  Signal detection failed for {ticker}: {e}")

    def _update_mtf_confirmation(self):
        """Check if weekly timeframe confirms daily signal direction."""
        for ticker, data in self.market_data.items():
            if data.mtf is None or data.quant_score is None:
                continue
            daily_bullish = data.quant_score.total > 0
            weekly_bullish = (
                data.mtf.weekly_trend == "up"
                and data.mtf.weekly_macd_histogram > 0
            )
            data.mtf.confirms_daily = (daily_bullish == weekly_bullish)

    def _analyze_all(self):
        logger.info("Analyzing with LLM...")

        # Periodically retry Ollama connection if it was down
        if not self.llm_analyzer._ollama_available:
            self.llm_analyzer._verify_ollama_connection()
            if self.llm_analyzer._ollama_available:
                logger.info("  ✓ Ollama reconnected — switching from quant-only to LLM mode")
            else:
                logger.info("  Ollama still unavailable — using quant-only mode")

        for ticker, data in self.market_data.items():
            try:
                rec = self.llm_analyzer.analyze(data)
                if rec:
                    self.recommendations[ticker] = rec

                    # Track agreement
                    self._total_count += 1
                    if rec.action.value == rec.quant_signal:
                        self._agreement_count += 1

                    logger.info(
                        f"  {ticker}: {rec.action.value} "
                        f"(conf={rec.confidence:.0%}, quant={rec.quant_signal})"
                    )
            except Exception as e:
                logger.error(f"  Failed to analyze {ticker}: {e}")

    def _execute_trades(self):
        """Execute trades based on recommendations (if broker enabled)."""
        if not self.broker or self.auto_trade == "off":
            return
        
        if rec.confidence < config.MIN_TRADE_CONFIDENCE or self.validator._calc_rr(rec) < config.MIN_RR_RATIO:
            logger.warning("Trade blocked by validator")
            return

        logger.info("Evaluating trades...")

        for ticker, rec in self.recommendations.items():
            # Filter: only trade actionable signals above confidence threshold
            if rec.action.value not in TRADEABLE_ACTIONS:
                continue
            if rec.confidence < MIN_TRADE_CONFIDENCE:
                logger.info(
                    f"  {ticker}: {rec.action.value} skipped — "
                    f"confidence {rec.confidence:.0%} < {MIN_TRADE_CONFIDENCE:.0%} threshold"
                )
                continue

            # Check if we already have a position (avoid doubling up)
            existing = self.broker.get_position(ticker)
            if existing and existing > 0 and rec.action in (Action.BUY, Action.STRONG_BUY):
                logger.info(f"  {ticker}: Already holding {existing:.4f} shares — skipping BUY")
                continue
            if (not existing or existing <= 0) and rec.action in (Action.SELL, Action.STRONG_SELL):
                logger.info(f"  {ticker}: No position to sell — skipping SELL")
                continue

            # Confirmation mode: ask user
            if self.auto_trade == "confirm":
                print(f"\n  ⚠ Trade signal: {rec.action.value} {ticker} "
                      f"(conf={rec.confidence:.0%}, quant={rec.quant_score:+.2f})")
                print(f"    Price: {rec.current_price:.2f} | "
                      f"Target: {rec.target_price} | Stop: {rec.stop_loss}")
                answer = input("    Execute? [y/N]: ").strip().lower()
                if answer != "y":
                    logger.info(f"  {ticker}: Trade declined by user")
                    continue

            # Execute
            try:
                if isinstance(self.broker, Trading212Broker):
                    result = self.broker.execute_recommendation(
                        rec, auto_size=True, order_type=OrderType.MARKET,
                        place_stop=True,
                    )
                else:
                    # Paper broker: simple execution
                    qty = 1.0  # placeholder for paper
                    balance = self.broker.get_account_balance()
                    if balance and rec.current_price > 0:
                        qty = min(
                            (balance * 0.02 * 2) / rec.current_price,
                            (balance * 0.20) / rec.current_price,
                        )
                    result = self.broker.execute_order(rec, qty)

                self._trade_results[ticker] = result

                if result.success:
                    logger.info(
                        f"  ✓ {ticker}: {result.message} "
                        f"(qty={result.executed_quantity})"
                    )
                else:
                    logger.warning(f"  ✗ {ticker}: {result.message}")

            except Exception as e:
                logger.error(f"  Trade execution error for {ticker}: {e}")

    def _update_display(self, iteration: int):
        try:
            self.display.clear_screen()
            self.display.show_header()
            self.display.show_market_summary(self.market_data)
            self.display.show_signal_alerts(self.market_data)
            self.display.show_volume_profile_summary(self.market_data)
            self.display.show_recommendations(self.recommendations)

            # Footer
            next_time = datetime.now().timestamp() + UPDATE_INTERVAL
            next_str = datetime.fromtimestamp(next_time).strftime("%H:%M:%S")
            agreement_pct = (
                f"{self._agreement_count / self._total_count:.0%}"
                if self._total_count > 0 else "N/A"
            )
            mode = "LLM+Quant" if self.llm_analyzer._ollama_available else "⚠ Quant-Only"

            # Broker status
            if self.broker and self.auto_trade != "off":
                balance = self.broker.get_account_balance()
                bal_str = f"€{balance:,.2f}" if balance else "?"
                broker_mode = "PAPER" if isinstance(self.broker, PaperTradingBroker) else TRADING212_MODE.upper()
                trade_str = f"Broker: {broker_mode} ({bal_str}) | Trade: {self.auto_trade}"
            else:
                trade_str = "Trading: off"

            print(f"\n{'─' * self.display.width}")
            print(
                f"  Iteration: {iteration} │ "
                f"Mode: {mode} │ "
                f"{trade_str} │ "
                f"Next: {next_str} │ "
                f"Agreement: {agreement_pct} │ "
                f"Ctrl+C to exit"
            )

            # Show recent trade results
            if self._trade_results:
                print(f"  Recent trades:")
                for ticker, result in self._trade_results.items():
                    icon = "✓" if result.success else "✗"
                    print(f"    {icon} {ticker}: {result.message}")

        except Exception as e:
            logger.error(f"Display error: {e}")

    def _shutdown(self):
        logger.info("Shutting down bot...")
        self.running = False
        print("\n\n  Bot stopped. Goodbye! 👋")
        sys.exit(0)

    def get_latest_recommendation(self, ticker: str) -> Optional[Recommendation]:
        return self.recommendations.get(ticker)

    def get_latest_market_data(self, ticker: str) -> Optional[MarketData]:
        return self.market_data.get(ticker)


def main():
    try:
        bot = StockAnalysisBot(tickers=TICKERS)
        bot.start()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

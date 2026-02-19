"""
Example: Bot with Trading212 for European Stocks

This script demonstrates how to use the bot with Trading212
for automated trading of European stocks.

IMPORTANT: 
- Start with DEMO mode to test
- Get API keys from Trading212 app: Settings > API (Beta)
- Trading212 is MUCH simpler than E*TRADE (no manual OAuth!)
"""

import logging
import time
import os
from datetime import datetime

from config import TICKERS, UPDATE_INTERVAL
from data_collector import DataCollector
from llm_analyzer import LLMAnalyzer, Action
from broker_interface import create_broker, OrderType
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trading212Bot:
    """
    Bot with Trading212 integration for European stocks.
    
    Features:
    - Automatic ticker conversion (SHELL.AS -> SHELL_NL_EQ)
    - EUR currency support
    - Position sizing based on balance
    - Risk management
    - Demo and Live modes
    """
    
    def __init__(
        self,
        tickers: list[str],
        api_key: str,
        api_secret: str,
        mode: str = "demo",
        risk_per_trade: float = 0.02,
        min_confidence: float = 0.7
    ):
        self.tickers = tickers
        self.risk_per_trade = risk_per_trade
        self.min_confidence = min_confidence
        
        # Initialize modules
        self.data_collector = DataCollector()
        self.llm_analyzer = LLMAnalyzer()
        
        # Initialize Trading212 broker
        logger.info(f"Initializing Trading212 broker ({mode} mode)...")
        self.broker = create_broker(
            "trading212",
            api_key=api_key,
            api_secret=api_secret,
            mode=mode
        )
        
        if not self.broker.connect():
            raise RuntimeError("Failed to connect to Trading212")
        
        logger.info("✓ Trading212 bot initialized and ready!")
    
    def run_once(self):
        """Run one iteration of analysis and trading"""
        balance = self.broker.get_account_balance()
        logger.info(f"Account balance: €{balance:.2f}")
        
        for ticker in self.tickers:
            try:
                # Display ticker (e.g., "SHELL.AS")
                display_name = ticker.split('.')[0]
                
                # Get market data (uses yfinance format)
                logger.info(f"Analyzing {display_name}...")
                market_data = self.data_collector.get_analysis(ticker)
                if not market_data:
                    logger.warning(f"No data for {ticker}")
                    continue
                
                # Get recommendation
                recommendation = self.llm_analyzer.analyze(market_data)
                if not recommendation:
                    logger.warning(f"No recommendation for {ticker}")
                    continue
                
                logger.info(
                    f"{display_name}: {recommendation.action.value} "
                    f"(confidence: {recommendation.confidence:.1%}) "
                    f"@ €{recommendation.current_price:.2f}"
                )
                
                # Check if we should trade
                if recommendation.confidence < self.min_confidence:
                    logger.info(f"⊘ Skipping {display_name} - confidence too low")
                    continue
                
                if recommendation.action == Action.HOLD:
                    logger.info(f"⊙ Holding {display_name}")
                    continue
                
                # Calculate position size
                quantity = self._calculate_position_size(
                    recommendation,
                    balance
                )
                
                if quantity == 0:
                    logger.info(f"⊘ Position size too small for {display_name}")
                    continue
                
                # Execute order
                logger.info(
                    f"→ Executing: {recommendation.action.value} "
                    f"{quantity} shares of {display_name}"
                )
                
                result = self.broker.execute_order(
                    recommendation,
                    quantity,
                    order_type=OrderType.MARKET
                )
                
                if result.success:
                    logger.info(f"✓ Order executed: {result.message}")
                else:
                    logger.error(f"✗ Order failed: {result.message}")
                
                # Update balance
                balance = self.broker.get_account_balance()
                
            except Exception as e:
                logger.error(f"Error processing {ticker}: {e}", exc_info=True)
    
    def _calculate_position_size(self, recommendation, balance: float) -> int:
        """Calculate position size based on risk management"""
        # Amount to risk on this trade
        risk_amount = balance * self.risk_per_trade
        
        # Calculate position size
        price = recommendation.current_price
        quantity = int(risk_amount / price)
        
        return quantity
    
    def run_continuous(self, interval: int = UPDATE_INTERVAL):
        """Run bot continuously"""
        logger.info("=" * 60)
        logger.info("Starting Trading212 Bot for European Stocks")
        logger.info("=" * 60)
        logger.info(f"Monitoring: {', '.join(self.tickers)}")
        logger.info(f"Update interval: {interval} seconds")
        logger.info("Press Ctrl+C to stop")
        logger.info("=" * 60)
        
        try:
            while True:
                logger.info(f"\n[{datetime.now().strftime('%H:%M:%S')}] Running iteration...")
                
                self.run_once()
                
                logger.info(f"Sleeping for {interval} seconds...\n")
                time.sleep(interval)
                
        except KeyboardInterrupt:
            logger.info("\nBot stopped by user")
        except Exception as e:
            logger.error(f"Fatal error: {e}", exc_info=True)


def main():
    """
    Example usage with Trading212.
    
    Setup:
    1. Get API keys from Trading212 app (Settings > API Beta)
    2. Set environment variables or pass them directly
    3. Start with DEMO mode!
    4. Test thoroughly before switching to LIVE
    """
    
    # Get API credentials from environment
    api_key = os.getenv("TRADING212_API_KEY", "")
    api_secret = os.getenv("TRADING212_API_SECRET", "")
    
    if not api_key or not api_secret:
        print("=" * 60)
        print("ERROR: Trading212 API credentials not found!")
        print("=" * 60)
        print()
        print("To get your API keys:")
        print("1. Open Trading212 app")
        print("2. Go to Settings > API (Beta)")
        print("3. Generate new API Key")
        print("4. Set environment variables:")
        print()
        print("   export TRADING212_API_KEY='your_key_here'")
        print("   export TRADING212_API_SECRET='your_secret_here'")
        print()
        print("Or edit this file and pass them directly (not recommended)")
        print("=" * 60)
        return
    
    # European stocks example
    # Format: ticker in yfinance format (will be auto-converted for Trading212)
    european_tickers = [
        "SHELL.AS",     # Shell - Amsterdam
        # Add more European stocks:
        # "AIR.PA",     # Airbus - Paris
        # "BMW.DE",     # BMW - Frankfurt  
        # "ASML.AS",    # ASML - Amsterdam
        # "SAP.DE",     # SAP - Frankfurt
    ]
    
    try:
        # DEMO MODE (safe for testing)
        bot = Trading212Bot(
            tickers=european_tickers,
            api_key=api_key,
            api_secret=api_secret,
            mode="live",  # Change to "live" when ready
            risk_per_trade=0.02,  # Risk 2% per trade
            min_confidence=0.7    # Only trade 70%+ confidence
        )
        
        # Run once
        bot.run_once()
        
        # Or run continuously
        # bot.run_continuous(interval=300)  # 5 minutes
        
    except Exception as e:
        logger.error(f"Failed to start bot: {e}", exc_info=True)
        print("\nCommon issues:")
        print("- Invalid API keys: Check Trading212 app settings")
        print("- Network error: Check your internet connection")
        print("- API not enabled: Make sure API Beta is enabled in your account")


if __name__ == "__main__":
    main()

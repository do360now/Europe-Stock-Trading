"""
Configuration for Stock Analysis Bot
Following Ousterhout's principles: centralized configuration, easy to modify.

Key edge-generating settings:
- Multi-timeframe analysis (daily + weekly confluence)
- Regime detection parameters
- Quantitative pre-scoring before LLM
- Benchmark-relative strength
"""
import os
from pathlib import Path

# ── Base directories ────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"
HISTORY_DIR = DATA_DIR / "recommendations"

for d in (DATA_DIR, LOGS_DIR, HISTORY_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ── Universe ────────────────────────────────────────────────────────────────
# EU-listed tickers (yfinance format)
TICKERS = [
    "ASML.AS", "SHELL.AS", "PHAG.AS",
    "RHM.DE", "DFNC.DE", "SEC0.DE", "NOV.DE",
]

# Benchmark for relative-strength comparison
# AEX for Amsterdam-heavy, STOXX 50 for broader EU
BENCHMARK_TICKER = "^STOXX50E"  # Euro Stoxx 50

# Market configuration
MARKET = "EU"  # "US" or "EU" — affects currency and session hours

# ── Timing ──────────────────────────────────────────────────────────────────
UPDATE_INTERVAL = 300        # 5 min between full analysis cycles
MARKET_DATA_INTERVAL = 60    # price-only refresh (future use)

# ── Ollama ──────────────────────────────────────────────────────────────────
OLLAMA_HOST = "http://localhost:11434"
OLLAMA_MODEL = "gemma3:4b"
OLLAMA_TIMEOUT = 120  # seconds

# ── Data ────────────────────────────────────────────────────────────────────
DATA_PROVIDER = "yfinance"
HISTORICAL_DAYS = 365

# Multi-timeframe: resample daily data to weekly for confluence signals
TIMEFRAMES = {
    "daily": "1d",
    "weekly": "1wk",
}

# ── Technical Indicators ────────────────────────────────────────────────────
INDICATORS = {
    # Trend
    "sma_periods": [10, 20, 50, 200],
    "ema_periods": [9, 12, 21, 26, 55],
    "macd": {"fast": 12, "slow": 26, "signal": 9},

    # Momentum
    "rsi_period": 14,
    "stoch_k": 14,
    "stoch_d": 3,
    "stoch_smooth": 3,

    # Volatility
    "bb_period": 20,
    "bb_std": 2,
    "atr_period": 14,

    # Volume
    "volume_sma": 20,
    "mfi_period": 14,

    # Trend strength
    "adx_period": 14,

    # Support / Resistance
    "pivot_lookback": 5,       # bars to left/right for fractal pivots
    "sr_cluster_pct": 0.015,   # 1.5% band to cluster nearby levels
}

# ── Quantitative Scoring ────────────────────────────────────────────────────
# A composite score is computed BEFORE the LLM sees the data.
# This anchors the LLM and prevents hallucinated conviction.
SCORE_WEIGHTS = {
    "trend":       0.25,   # SMA alignment, ADX
    "momentum":    0.20,   # RSI zone, MACD histogram direction
    "volume":      0.15,   # OBV trend, MFI, volume ratio
    "volatility":  0.10,   # BB position, ATR expansion/contraction
    "rel_strength": 0.15,  # vs benchmark
    "structure":   0.15,   # support/resistance proximity
}

# ── Thresholds ──────────────────────────────────────────────────────────────
THRESHOLDS = {
    "strong_buy_score":  0.60,
    "buy_score":         0.30,
    "hold_upper":        0.30,
    "hold_lower":       -0.30,
    "sell_score":       -0.30,
    "strong_sell_score": -0.60,
}

# Regime detection: ADX-based
REGIME = {
    "trending_adx":  25,    # ADX above this = trending market
    "weak_adx":      15,    # ADX below this = choppy / range-bound
    "vol_expansion": 1.3,   # ATR ratio vs 20-day mean ATR > this = expansion
}

# ── Display ─────────────────────────────────────────────────────────────────
DISPLAY_REFRESH_RATE = 1
TERMINAL_WIDTH = 130

# ── Broker configs ─────────────────────────────────────────────────────────
TRADING212_API_KEY = os.getenv("TRADING212_API_KEY", "")
TRADING212_API_SECRET = os.getenv("TRADING212_API_SECRET", "")
TRADING212_MODE = os.getenv("TRADING212_MODE", "demo")  # "demo" or "live"

# Auto-trading: "off" = display only, "confirm" = ask before each trade,
#               "auto" = execute automatically based on bot signals
AUTO_TRADE_MODE = os.getenv("AUTO_TRADE_MODE", "off")

# Position sizing
RISK_PCT_PER_TRADE = 2.0    # % of free cash risked per trade
MAX_POSITION_PCT = 20.0     # max % of account in one position

# Minimum confidence to auto-execute (0.0-1.0)
MIN_TRADE_CONFIDENCE = 0.60

# Only trade on BUY/STRONG_BUY/SELL/STRONG_SELL (skip HOLD obviously)
# Set to ["STRONG_BUY", "STRONG_SELL"] to only trade high-conviction signals
TRADEABLE_ACTIONS = ["STRONG_BUY", "BUY", "SELL", "STRONG_SELL"]

# ── Logging ─────────────────────────────────────────────────────────────────
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

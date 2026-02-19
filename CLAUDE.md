# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI-powered stock analysis bot that uses technical indicators and local LLM (Ollama) for intelligent trading recommendations. Primarily focused on European stocks with multi-timeframe analysis and regime-aware trading.

## Commands

```bash
# Start the main bot (runs continuous analysis loop)
python bot.py

# Run Trading212 example for European stocks
python example_trading212.py
```

## Prerequisites

- **Python 3.9+**
- **Ollama** running locally with a model installed:
  ```bash
  ollama serve
  ollama pull gemma3:4b  # or your preferred model
  ```

## Configuration

All settings are in `config.py`:
- `TICKERS` - List of stock symbols to monitor (EU format: "ASML.AS", "SHELL.AS", etc.)
- `BENCHMARK_TICKER` - Benchmark for relative-strength comparison (default: "^STOXX50E")
- `UPDATE_INTERVAL` - Seconds between analysis cycles (default: 300)
- `OLLAMA_MODEL` - LLM model name (default: "gemma3:4b")
- `OLLAMA_TIMEOUT` - LLM request timeout in seconds

Environment variables (in `.env`):
- `TRADING212_API_KEY` / `TRADING212_API_SECRET` - For Trading212 integration
- `AUTO_TRADE_MODE` - "off", "confirm", or "auto"

## Architecture

Built following John Ousterhout's "A Philosophy of Software Design" - deep modules with simple interfaces:

```
bot.py (thin orchestrator)
├── data_collector.py  (deep: get_analysis(ticker) → MarketData + 30+ indicators)
├── signal_detector.py (detects institutional footprint signals)
├── llm_analyzer.py    (deep: analyze(market_data) → Recommendation)
├── recommendation_validator.py (validates R:R, anchoring, regime guards)
├── broker_interface.py (Trading212 API + paper trading)
└── display.py         (terminal dashboard)
```

### Key Design: Quant-Anchored LLM

The bot computes a **quantitative composite score (-1 to +1)** BEFORE sending data to the LLM. This anchors the LLM and prevents hallucinated conviction. The LLM can only deviate ±1 step from the quant signal.

### Data Flow

Each iteration:
1. `data_collector.get_analysis(ticker)` → MarketData with 30+ indicators + quant score
2. `signal_detector.detect_signals()` → institutional footprint signals
3. `llm_analyzer.analyze()` → Recommendation (LLM or quant-only fallback)
4. `recommendation_validator.refine()` → validated with R:R, anchoring, regime guards
5. `broker.execute_recommendation()` → trades (if enabled)
6. `display.show()` → terminal dashboard

## Key Features

- **30+ Technical Indicators**: SMA, EMA, RSI (Wilder-smoothed), MACD, Bollinger Bands, ADX, Stochastic, ATR, OBV, MFI
- **Regime Detection**: Classifies market as trending_up, trending_down, ranging, or volatile
- **Multi-Timeframe Confluence**: Weekly confirms/denies daily signals
- **Volume-at-Level Signals**: Detects institutional order flow (BREAKOUT_CONFIRMED, VOLUME_CLIMAX, ACCUMULATION_AT_SUPPORT, etc.)
- **Relative Strength**: Stock vs Euro Stoxx 50 performance
- **Fallback Mode**: Runs in quant-only mode if Ollama unavailable
- **Risk Management**: R:R validation (MIN_RR_RATIO=1.8), regime confidence caps, ATR-based targets/stops

## Risk Rules (config.py)

- `MIN_RR_RATIO = 1.8` - Minimum reward:risk (1:1.8 conservative)
- `MAX_QUANT_LLM_ACTION_DEVIATION = 1` - LLM can only deviate ±1 step from quant signal
- `WEAK_REGIME_CONFIDENCE_CAP = 0.65` - Never exceed 65% confidence in weak/choppy regimes
- `ATR_TARGET_MULTIPLIER = 4.0` - Target = entry + 4×ATR (for 2:1+ R:R when stop=2×ATR)

## Key Components

- **models.py**: Core data models with computed properties (`potential_return`, `rr_ratio`) - prevents display breakage
- **data_collector.py**: Market data fetching, indicator calculation, quant score computation
- **signal_detector.py**: Institutional footprint detection at S/R levels
- **llm_analyzer.py**: Ollama integration, prompt engineering, response parsing
- **recommendation_validator.py**: Enforces anchoring, R:R ratios, regime conservatism
- **broker_interface.py**: Trading212 API with ATR-based position sizing
- **display.py**: Terminal dashboard with color-coded output

## Logging

Logs are written to `logs/bot_YYYYMMDD.log`. Use `tail -f logs/bot_*.log` to watch in real-time.

## Data Storage

- `data/recommendations/` - Historical recommendations in JSONL format
- `data/order_history.jsonl` - Trade execution history
- `.env` file is git-ignored (never commit API keys)

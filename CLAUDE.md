# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

AI-powered stock analysis bot that uses technical indicators and local LLM (Ollama) for intelligent trading recommendations. Primarily focused on European stocks with multi-timeframe analysis.

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

Built following John Ousterhout's "A Philosophy of Software Design" principles:

```
bot.py (thin orchestrator)
├── data_collector.py  (deep module: market data + technical indicators)
│   └── get_analysis(ticker) → complete market data with 10+ indicators
├── llm_analyzer.py    (deep module: LLM analysis)
│   └── analyze(market_data) → trading recommendation
└── display.py         (presentation layer)
```

**Deep Module Pattern**: `DataCollector` and `LLMAnalyzer` hide complex implementation behind simple single-method interfaces.

## Key Components

- **data_collector.py**: Fetches market data via yfinance, calculates technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands, ADX, Stochastic, ATR, MFI), supports multi-timeframe (daily + weekly), includes quantitative pre-scoring before LLM
- **llm_analyzer.py**: Integrates with Ollama for LLM-powered analysis, includes prompt engineering and response parsing
- **signal_detector.py**: Additional signal detection logic
- **broker_interface.py**: Trading212 integration (optional automated trading)
- **display.py**: Terminal dashboard with color-coded recommendations

## Logging

Logs are written to `logs/bot_YYYYMMDD.log`. Use `tail -f logs/bot_*.log` to watch in real-time.

## Data Storage

- `data/recommendations/` - Historical recommendations in JSONL format
- `.env` file is git-ignored (never commit API keys)

# Stock Analysis Bot 🤖

AI-powered stock analysis bot leveraging technical indicators and local LLM (Ollama) for intelligent trading recommendations.

## Architecture Overview

Built following **John Ousterhout's "A Philosophy of Software Design"** principles:

```
┌─────────────────┐
│   Main Bot      │  ← Orchestrator (thin layer)
└────────┬────────┘
         │
    ┌────┴─────────────────────────┐
    │                               │
┌───▼─────────┐            ┌───────▼────────┐
│ DataCollector│           │  LLMAnalyzer   │
│ (Deep Module)│           │ (Deep Module)  │
└───┬──────────┘           └────────┬───────┘
    │                               │
    ├─ Market data fetching         ├─ Prompt engineering
    ├─ Technical indicators         ├─ Ollama integration
    ├─ Caching                      └─ Response parsing
    └─ Simple interface: 
       get_analysis(ticker)
                    │
                    ▼
            ┌───────────────┐
            │    Display    │
            │  (Presentation)│
            └───────────────┘
```

**Deep Modules with Simple Interfaces:**
- `DataCollector.get_analysis(ticker)` → Returns complete market analysis
- `LLMAnalyzer.analyze(market_data)` → Returns trading recommendation
- All complexity hidden behind clean interfaces

## Features

✅ **Multi-Market Support**
- US stocks (E*TRADE, yfinance)
- **European stocks** (Trading212, Amsterdam, Paris, Frankfurt, etc.)
- Automatic ticker conversion

✅ **Real-time Market Analysis**
- Fetches live market data using yfinance
- Calculates 10+ technical indicators
- Caches data efficiently (1-minute TTL)

✅ **Advanced Technical Indicators**
- Simple Moving Averages (SMA): 20, 50, 200-day
- Exponential Moving Averages (EMA): 12, 26-day
- Relative Strength Index (RSI)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Volume analysis

✅ **LLM-Powered Analysis**
- Uses local Ollama for privacy and speed
- Sophisticated prompt engineering
- Confidence scoring
- Price targets and stop-loss suggestions

✅ **Professional Display**
- Clean terminal dashboard
- Color-coded recommendations
- Market snapshot view
- Auto-refreshing updates

✅ **Extensible Design**
- Easy to add new tickers
- Modular broker integration (E*TRADE ready)
- Simple to add new indicators
- Clean separation of concerns

## Prerequisites

1. **Python 3.9+**
2. **Ollama** (running locally)
   ```bash
   # Install Ollama: https://ollama.ai/
   ollama serve
   ollama pull llama3.1  # or your preferred model
   ```

## Installation

```bash
# Clone/create project directory
mkdir stock-bot && cd stock-bot

# Install dependencies
pip install -r requirements.txt
```

## Configuration

Edit `config.py` to customize:

```python
# Tickers to monitor
TICKERS = ["LITE", "AAPL", "TSLA"]  # Add more here

# Update frequency
UPDATE_INTERVAL = 300  # 5 minutes

# Ollama settings
OLLAMA_MODEL = "Qwen2.5:3b"  # or "mistral", "codellama", etc.

# Technical indicator periods
INDICATORS = {
    "sma_periods": [20, 50, 200],
    "ema_periods": [12, 26],
    "rsi_period": 14,
    # ... more
}
```

## Usage

### Basic Usage

```bash
# Start the bot
python bot.py
```

The bot will:
1. Fetch market data for all configured tickers
2. Calculate technical indicators
3. Analyze with LLM
4. Display recommendations
5. Auto-refresh every 5 minutes (configurable)

### Display Output

```
═══════════════════════════════════════════════════════════════
                     STOCK ANALYSIS BOT
                Powered by LLM + Technical Analysis
═══════════════════════════════════════════════════════════════
Last Updated: 2026-02-02 14:30:00
═══════════════════════════════════════════════════════════════

┌────────────────────────────────────────────────────────────┐
│             MARKET SNAPSHOT                                │
├────────────────────────────────────────────────────────────┤
│ Ticker │   Price   │  1D Chg  │  5D Chg  │ 1M Chg │   RSI   │
├────────────────────────────────────────────────────────────┤
│ LITE   │  $123.45  │  +2.3%  │  +5.1%  │ +12.4% │  65.2   │
└────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────┐
│ LITE   │ Action: BUY           │ Confidence: [████████░░] 80% │
├────────────────────────────────────────────────────────────┤
│ Current Price:    $123.45  │  Target: $135.00  │  +9.4%  │
│ Stop Loss:        $115.00  │  Risk:   -6.8%                │
├────────────────────────────────────────────────────────────┤
│ REASONING:                                                  │
│ Strong uptrend confirmed by price above all SMAs. RSI      │
│ shows momentum without being overbought. MACD bullish      │
│ crossover suggests continued strength. Volume supporting.  │
└────────────────────────────────────────────────────────────┘
```

### Adding More Tickers

Simply edit `config.py`:

```python
TICKERS = ["LITE", "AAPL", "MSFT", "GOOGL", "TSLA"]
```

### Changing LLM Models

```python
# In config.py
OLLAMA_MODEL = "mistral"  # or any Ollama model

# Pull the model first
ollama pull mistral
```

## Project Structure

```
stock-bot/
├── bot.py              # Main orchestrator
├── config.py           # Configuration
├── data_collector.py   # Market data & indicators (Deep Module)
├── llm_analyzer.py     # LLM analysis (Deep Module)
├── display.py          # Terminal display
├── broker_interface.py # Optional: E*TRADE integration
├── requirements.txt    # Dependencies
├── data/              # Cache directory
└── logs/              # Log files
```

## Design Principles (Ousterhout)

This codebase follows key principles from "A Philosophy of Software Design":

### 1. Deep Modules
**DataCollector** and **LLMAnalyzer** are deep modules:
- Simple interface (single method: `get_analysis()`, `analyze()`)
- Complex implementation hidden
- Easy to use, hard to misuse

### 2. Information Hiding
- Data fetching complexity hidden in DataCollector
- Prompt engineering hidden in LLMAnalyzer
- Indicator calculations encapsulated
- Cache management transparent

### 3. General-Purpose Modules
- `DataCollector` works with any ticker
- `LLMAnalyzer` works with any market data
- Easy to extend without modifying core

### 4. Obvious Code
- Clear naming conventions
- Comprehensive docstrings
- Logical organization
- Type hints throughout

## E*TRADE Integration (Optional)

For automated trading, E*TRADE API can be integrated:

```python
# In config.py
ETRADE_CONSUMER_KEY = "your_key"
ETRADE_CONSUMER_SECRET = "your_secret"
```

**Note:** E*TRADE requires manual OAuth each session. For fully automated trading, consider Trading212 or Alpaca.

See `broker_interface.py` for implementation details.

## Trading212 Integration (Recommended for European Stocks) 🇪🇺

**Trading212 is ideal for European stocks** - much simpler than E*TRADE!

### Quick Setup:
1. **Get API keys**: Trading212 App → Settings → API (Beta)
2. **Set environment**:
   ```bash
   export TRADING212_API_KEY='your_key'
   export TRADING212_API_SECRET='your_secret'
   ```
3. **Configure tickers**:
   ```python
   # config.py
   TICKERS = ["SHELL.AS", "AIR.PA", "BMW.DE"]
   MARKET = "EU"
   ```
4. **Run**: `python example_trading212.py`

### Advantages:
- ✅ Simple API key authentication (no manual OAuth!)
- ✅ Native European stock support (Amsterdam, Paris, Frankfurt, London, etc.)
- ✅ EUR, GBP, USD currencies
- ✅ Built-in demo mode
- ✅ Fully automated (unlike E*TRADE)

**See [EUROPEAN_STOCKS.md](EUROPEAN_STOCKS.md) for complete guide!**

## Extending the Bot

### Adding New Indicators

Edit `data_collector.py`:

```python
def _calculate_stochastic(self, data: pd.DataFrame) -> Dict[str, float]:
    # Your indicator logic
    return {"k": k_value, "d": d_value}
```

### Adding New Data Sources

Implement the interface in a new module:

```python
class AlpacaDataCollector(DataCollector):
    def _fetch_and_analyze(self, ticker: str) -> MarketData:
        # Use Alpaca API
        pass
```

### Custom LLM Prompts

Edit `llm_analyzer.py` → `_build_prompt()` method.

## Logging

Logs are written to:
- Console (INFO level)
- `logs/bot_YYYYMMDD.log` (all levels)

```bash
# Watch logs in real-time
tail -f logs/bot_20260202.log
```

## Troubleshooting

### "Cannot connect to Ollama"
```bash
# Start Ollama service
ollama serve

# Verify it's running
curl http://localhost:11434/api/tags
```

### "No data available for ticker"
- Check if market is open (data may be delayed)
- Verify ticker symbol is correct
- Check your internet connection

### "Analysis timeout"
- Increase `OLLAMA_TIMEOUT` in config.py
- Use a smaller/faster model
- Check Ollama service status

## Performance

- **Memory:** ~200MB typical usage
- **CPU:** Low (spikes during LLM analysis)
- **Network:** Minimal (1-2 requests per ticker per update)
- **Disk:** Logs only

## Security Notes

⚠️ **Never commit API keys to git**
- Use environment variables
- Create `.env` file (git-ignored)
- Use secrets management for production

⚠️ **This is for informational purposes only**
- Not financial advice
- Always verify recommendations
- Use paper trading first
- Understand the risks

## Roadmap

- [ ] Multi-timeframe analysis
- [ ] Backtesting framework
- [ ] Portfolio optimization
- [ ] Risk management system
- [ ] Web dashboard
- [ ] Real-time alerts (Discord/Telegram)
- [ ] Machine learning integration
- [ ] Multi-broker support

## Contributing

Contributions welcome! Focus areas:
- Additional technical indicators
- Better prompt engineering
- Performance optimizations
- New broker integrations

## License

MIT License - Use at your own risk

## Disclaimer

This software is for educational and informational purposes only. It is not financial advice. The authors are not responsible for any trading losses. Always do your own research and consult with qualified financial advisors before making investment decisions.

---

**Built with ❤️ following clean architecture principles**

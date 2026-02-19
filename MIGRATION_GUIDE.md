# Migration Guide: US → European Stocks 🇺🇸 → 🇪🇺

Quick guide to switch your bot from US stocks (E*TRADE) to European stocks (Trading212).

## What Changed?

### ✅ Still Works:
- All core analysis features
- LLM-powered recommendations  
- Technical indicators
- Display dashboard
- Paper trading

### 🆕 New Features:
- Trading212 broker support
- European exchange support
- EUR/GBP currency display
- Automatic ticker conversion
- Simpler authentication

## Quick Migration Steps

### 1. Update Config

**Before (US stocks):**
```python
# config.py
TICKERS = ["LITE"]
# No market setting
```

**After (European stocks):**
```python
# config.py
TICKERS = ["SHELL.AS"]  # Amsterdam exchange
MARKET = "EU"           # Enables EUR display
```

### 2. Get Trading212 API Keys

```bash
# In Trading212 mobile app:
Settings > API (Beta) > Generate API Key

# Set environment
export TRADING212_API_KEY='your_key'
export TRADING212_API_SECRET='your_secret'
```

### 3. Run Updated Bot

```bash
# Analysis only (no trading)
python bot.py

# With Trading212 trading
python example_trading212.py
```

## Ticker Format Examples

### US Stocks (yfinance):
```python
TICKERS = ["AAPL", "MSFT", "GOOGL", "TSLA"]
```

### European Stocks (yfinance):
```python
TICKERS = [
    "SHELL.AS",   # Amsterdam
    "AIR.PA",     # Paris
    "BMW.DE",     # Frankfurt
    "BP.L",       # London
]
```

**Bot auto-converts** to Trading212 format!

## Configuration Changes

### Currency Display

```python
# config.py
MARKET = "EU"  # Shows €31.50
MARKET = "US"  # Shows $423.42
```

### Trading212 vs E*TRADE

```python
# E*TRADE (US stocks only)
ETRADE_CONSUMER_KEY = "..."
ETRADE_CONSUMER_SECRET = "..."

# Trading212 (European + US stocks)
TRADING212_API_KEY = "..."
TRADING212_API_SECRET = "..."
TRADING212_MODE = "demo"  # or "live"
```

## Backward Compatibility

**Good news**: You can still use US stocks!

```python
# Mix US and European stocks
TICKERS = [
    # US
    "AAPL",       # Apple (US)
    "MSFT",       # Microsoft (US)
    
    # European
    "SHELL.AS",   # Shell (Netherlands)
    "SAP.DE",     # SAP (Germany)
]

# Bot will handle both!
```

**Trading**: 
- US stocks → Use E*TRADE or paper trading
- EU stocks → Use Trading212

## Common Issues

### "Ollama too slow"
✅ **Already fixed** - you're using `qwen2.5:3b` (faster model)

### "Can't find SHELL ticker"
```python
# ✅ Correct
TICKERS = ["SHELL.AS"]  # With exchange suffix

# ❌ Wrong
TICKERS = ["SHELL"]     # Missing .AS
```

### "Wrong currency displayed"
```python
# Set market in config.py
MARKET = "EU"  # For European stocks
```

## Testing Your Migration

### 1. Test Analysis (Safe)
```bash
python bot.py
```
Should show:
- ✅ SHELL ticker working
- ✅ EUR prices (€31.50)
- ✅ LLM recommendations

### 2. Test Paper Trading (Safe)
```bash
# In Python
from broker_interface import PaperTradingBroker
broker = PaperTradingBroker(initial_balance=10000)
broker.connect()
# Test trading...
```

### 3. Test Trading212 Demo (Safe)
```bash
python example_trading212.py
```
Uses virtual money!

### 4. Go Live (When Ready)
```python
# In example_trading212.py
mode="live"  # Real money!
```

## File Changes Summary

| File | Changes |
|------|---------|
| `config.py` | Added SHELL.AS, MARKET="EU", Trading212 settings |
| `broker_interface.py` | Added Trading212Broker class |
| `display.py` | Added EUR currency support |
| `bot.py` | Added market-aware currency |
| `example_trading212.py` | New! Trading212 integration |
| `EUROPEAN_STOCKS.md` | New! Complete EU guide |

## Rollback (If Needed)

Want to go back to US stocks?

```python
# config.py
TICKERS = ["LITE"]  # Back to US
# MARKET = "US"     # Optional
```

Everything still works!

## Pro Tips

1. **Start with demo**: Always test in Trading212 demo mode
2. **Check market hours**: European markets have different hours
3. **Watch currency**: EUR can fluctuate vs USD
4. **Diversify**: Mix US and EU stocks if you want
5. **Read EUROPEAN_STOCKS.md**: Complete guide with all details

## Need Help?

Check these files:
- `EUROPEAN_STOCKS.md` - Complete European guide
- `README.md` - General documentation
- `example_trading212.py` - Working example
- `logs/` - Check for errors

## What's Next?

1. ✅ Bot runs with SHELL.AS
2. ✅ EUR prices displayed correctly
3. ✅ LLM gives recommendations
4. 🔜 Get Trading212 API keys
5. 🔜 Test with demo mode
6. 🔜 Add more European stocks
7. 🔜 Go live (carefully!)

---

**You're all set for European markets! 🇪🇺🚀**

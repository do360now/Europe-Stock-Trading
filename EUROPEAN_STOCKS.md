# European Stocks Trading with Trading212 🇪🇺

Complete guide for trading European stocks using Trading212 API.

## Why Trading212 for European Stocks?

✅ **Better than E*TRADE for European markets:**
- Native support for European exchanges (Amsterdam, Paris, Frankfurt, etc.)
- Simple API authentication (no manual OAuth like E*TRADE)
- EUR, GBP, and other European currencies
- Commission-free trading
- Demo account for testing

## Supported Exchanges

Trading212 supports major European exchanges:

| Exchange | Code | Example Stocks |
|----------|------|----------------|
| **Euronext Amsterdam** | .AS | SHELL.AS, ASML.AS, ING.AS |
| **Euronext Paris** | .PA | AIR.PA, MC.PA, OR.PA |
| **Frankfurt (Xetra)** | .DE | BMW.DE, SAP.DE, SIE.DE |
| **London Stock Exchange** | .L | BP.L, HSBA.L, VOD.L |
| **Bolsa de Madrid** | .MC | TEF.MC, IBE.MC, SAN.MC |
| **Milan Stock Exchange** | .MI | UCG.MI, ISP.MI, ENI.MI |

Plus: Vienna, Lisbon, and more!

## Quick Start

### 1. Get Trading212 API Keys

```bash
# In Trading212 mobile app:
Settings > API (Beta) > Generate API Key
```

You'll get:
- **API Key** (username)
- **API Secret** (password)

### 2. Set Environment Variables

```bash
export TRADING212_API_KEY='your_key_here'
export TRADING212_API_SECRET='your_secret_here'
```

### 3. Update Config

Edit `config.py`:

```python
# Add European stocks
TICKERS = [
    "SHELL.AS",   # Shell - Amsterdam
    "AIR.PA",     # Airbus - Paris
    "BMW.DE",     # BMW - Frankfurt
]

# Set market to EU
MARKET = "EU"  # Enables EUR currency display
```

### 4. Run the Bot

```bash
# Analysis only (no trading)
python bot.py

# With Trading212 integration
python example_trading212.py
```

## Ticker Format Reference

### In Config (yfinance format):
```python
TICKERS = ["SHELL.AS", "AIR.PA", "BMW.DE"]
```

### Bot Auto-Converts For You:
```
SHELL.AS  →  SHELL_NL_EQ  (Trading212 API)
AIR.PA    →  AIR_PA_EQ    (Trading212 API)
BMW.DE    →  BMW_DE_EQ    (Trading212 API)
```

**You don't need to worry about conversion** - the bot handles it automatically!

## Popular European Stocks

### 🇳🇱 Netherlands (Amsterdam)
```python
"SHELL.AS"    # Shell - Energy
"ASML.AS"     # ASML - Semiconductors
"ING.AS"      # ING Bank
"PHIA.AS"     # Philips
"HEIA.AS"     # Heineken
```

### 🇫🇷 France (Paris)
```python
"AIR.PA"      # Airbus - Aerospace
"MC.PA"       # LVMH - Luxury
"OR.PA"       # L'Oréal - Cosmetics
"TTE.PA"      # TotalEnergies - Energy
"SAN.PA"      # Sanofi - Pharma
```

### 🇩🇪 Germany (Frankfurt)
```python
"BMW.DE"      # BMW - Automotive
"SAP.DE"      # SAP - Software
"SIE.DE"      # Siemens - Industrial
"VOW3.DE"     # Volkswagen
"DAI.DE"      # Mercedes-Benz
```

### 🇬🇧 United Kingdom (London)
```python
"BP.L"        # BP - Energy
"HSBA.L"      # HSBC - Banking
"ULVR.L"      # Unilever - Consumer
"GSK.L"       # GSK - Pharma
"RIO.L"       # Rio Tinto - Mining
```

## Trading212 Features

### Demo Mode (Safe Testing)
```python
bot = Trading212Bot(
    tickers=["SHELL.AS"],
    api_key=api_key,
    api_secret=api_secret,
    mode="demo"  # Virtual money
)
```

### Live Mode (Real Money)
```python
bot = Trading212Bot(
    tickers=["SHELL.AS"],
    api_key=api_key,
    api_secret=api_secret,
    mode="live"  # Real money - be careful!
)
```

### Order Types

**Demo Account:**
- ✅ Market Orders
- ✅ Limit Orders
- ✅ Stop Orders
- ✅ Stop-Limit Orders

**Live Account:**
- ✅ Market Orders only (currently)
- ⏳ Other order types coming soon

## Example: Multi-Exchange Portfolio

```python
# config.py
TICKERS = [
    # Netherlands
    "SHELL.AS",     # Shell
    "ASML.AS",      # ASML
    
    # France  
    "AIR.PA",       # Airbus
    "MC.PA",        # LVMH
    
    # Germany
    "BMW.DE",       # BMW
    "SAP.DE",       # SAP
    
    # UK
    "BP.L",         # BP
    "HSBA.L",       # HSBC
]

MARKET = "EU"
UPDATE_INTERVAL = 300  # 5 minutes
```

## Currency Handling

The bot automatically handles European currencies:

```python
MARKET = "EU"  # Uses EUR (€) for display
MARKET = "US"  # Uses USD ($) for display
```

Display will show:
- **EU market**: €31.50 (for SHELL.AS)
- **US market**: $423.42 (for LITE)

## Market Hours

Trading212 respects exchange trading hours:

| Exchange | Trading Hours (CET) |
|----------|---------------------|
| Amsterdam | 09:00 - 17:40 |
| Paris | 09:00 - 17:40 |
| Frankfurt | 09:00 - 17:30 |
| London | 08:00 - 16:30 |

**Note**: Bot fetches data 24/7, but orders only execute during market hours.

## Advanced: Custom Risk Management

```python
bot = Trading212Bot(
    tickers=["SHELL.AS"],
    api_key=api_key,
    api_secret=api_secret,
    mode="demo",
    risk_per_trade=0.01,    # Risk 1% per trade (conservative)
    min_confidence=0.80      # Only trade 80%+ confidence (strict)
)
```

## Troubleshooting

### "Invalid API credentials"
- Check keys in Trading212 app
- Make sure API is enabled (Beta feature)
- Verify environment variables are set

### "Ticker not found"
- Make sure ticker format is correct (e.g., "SHELL.AS" not "SHELL")
- Check if stock is available on Trading212
- Try searching in Trading212 app first

### "Order failed - market closed"
- Check exchange trading hours
- Wait for market to open
- Use demo mode for testing outside hours

### "Insufficient funds"
- Check account balance
- Reduce `risk_per_trade` setting
- Start with smaller positions

## Best Practices

1. **Start with Demo**: Always test strategies in demo mode first
2. **Diversify**: Don't put all capital in one stock
3. **Set Limits**: Use `min_confidence` to filter trades
4. **Monitor Logs**: Check `logs/` directory for issues
5. **Respect Hours**: Be aware of different exchange hours
6. **Currency Risk**: Consider EUR/USD exposure if mixing markets

## Security

⚠️ **Never commit API keys to git!**

```bash
# Use environment variables
export TRADING212_API_KEY='...'
export TRADING212_API_SECRET='...'

# Or create .env file (git-ignored)
echo "TRADING212_API_KEY=..." >> .env
echo "TRADING212_API_SECRET=..." >> .env
```

## Comparison: Trading212 vs E*TRADE

| Feature | Trading212 | E*TRADE |
|---------|-----------|---------|
| **European Stocks** | ✅ Native | ❌ Not available |
| **Authentication** | ✅ Simple (API Key) | ⚠️ Manual OAuth |
| **Automation** | ✅ Fully automated | ⚠️ Session-based |
| **Demo Mode** | ✅ Built-in | ✅ Sandbox |
| **Currencies** | EUR, GBP, USD | USD only |
| **Setup Time** | 5 minutes | 30+ minutes |
| **Market Orders** | ✅ Live & Demo | ✅ Live & Demo |
| **Limit Orders** | ✅ Demo only | ✅ Live & Demo |

**For European stocks**: Trading212 is the clear winner! 🏆

## Resources

- **Trading212 API Docs**: https://t212public-api-docs.redoc.ly/
- **Trading212 Community**: https://community.trading212.com/
- **Support**: In-app chat (very responsive)

## Next Steps

1. ✅ Get API keys from Trading212 app
2. ✅ Set environment variables
3. ✅ Update `config.py` with European tickers
4. ✅ Test with `python example_trading212.py`
5. ✅ Monitor logs and refine strategy
6. ✅ Start small, scale gradually

---

**Happy Trading! 🚀🇪🇺**

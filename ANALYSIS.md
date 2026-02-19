# Stock Analysis Bot v2 — Deep Dive Analysis

## The Core Problem with v1

v1 had a fundamental architectural flaw: **it trusted a 4B-parameter local LLM to make trading decisions with minimal data and no guardrails.** A small model like gemma3:4b will hallucinate conviction on weak setups — it'll say "STRONG_BUY with 85% confidence" on a stock that's in a choppy range with mixed signals, simply because the prompt asked for a recommendation and the model wants to be helpful.

Professional traders and institutional algos don't work this way. They use quantitative scoring to establish a baseline, then overlay discretionary judgment. v2 mirrors this approach.

---

## Architecture: Quant Score Anchors the LLM

```
Market Data → 30+ Indicators → Composite Score (-1 to +1) → LLM Prompt (score included)
                                       ↓                              ↓
                                  Signal: BUY               LLM constrained to ±1 step
                                                             (can say STRONG_BUY, BUY, or HOLD)
                                                             (CANNOT say SELL or STRONG_SELL)
```

This is the single biggest edge. The LLM adds nuance and reasoning, but cannot wildly deviate from what the numbers say.

---

## Edge #1: Wilder-Smoothed RSI (was broken in v1)

v1 used `rolling(window).mean()` for RSI — this is SMA-based smoothing. The industry standard (and what every platform from TradingView to Bloomberg uses) is **Wilder's exponential smoothing**:

```
avg_gain[i] = (avg_gain[i-1] * (period-1) + gain[i]) / period
```

Why this matters: SMA-based RSI dampens extremes. A stock that should read RSI=75 (overbought) might show RSI=62 with SMA smoothing. You'd miss the sell signal that every other trader (human and algo) is acting on.

## Edge #2: ADX + Regime Detection

v1 had no concept of market regime. It would apply the same analysis to a strongly trending stock (ADX=45) and a choppy sideways stock (ADX=12). These require opposite strategies:

| Regime | ADX | Strategy |
|--------|-----|----------|
| Trending | >25 | Follow trend, buy pullbacks, trail stops |
| Ranging | <15 | Mean reversion, buy support / sell resistance |
| Weak trend | 15-25 | Reduce conviction, tighter stops |
| Volatile + choppy | low ADX + high ATR | Reduce position size, wait for clarity |

The LLM prompt now includes regime-specific strategy hints, so even a small model knows which playbook to use.

## Edge #3: Volume Divergence (Smart Money Footprints)

v1 only looked at volume vs its average. v2 adds three volume-based indicators:

- **OBV (On-Balance Volume)**: Tracks cumulative volume flow. When price makes new highs but OBV doesn't → bearish divergence (smart money is selling into strength). This is one of the most reliable early warning signals.
- **MFI (Money Flow Index)**: Volume-weighted RSI. Extreme readings (<20 or >80) with high volume carry more weight than price-only RSI.
- **Accumulation/Distribution Line**: Separates whether closing action is accumulation (closes near highs) or distribution (closes near lows).

## Edge #4: Multi-Timeframe Confluence

Most retail traders only look at daily charts. Institutional traders confirm daily signals on the weekly timeframe. v2 fetches weekly data and checks:

- Does the weekly trend agree with the daily signal?
- Is the weekly MACD histogram confirming?
- Is price above/below the weekly 20-SMA?

When daily AND weekly agree, confidence is higher. When they diverge, the system flags it and reduces conviction.

## Edge #5: Relative Strength vs Benchmark

A stock that's up 3% this month sounds good — unless the Euro Stoxx 50 is up 8%. That stock is actually **underperforming** and is likely to continue lagging (sector rotation effect).

v2 computes 20-day and 60-day relative strength vs the benchmark. Outperformers tend to keep outperforming (momentum factor, well-documented in academic literature).

## Edge #6: Support/Resistance-Based Targets

v1 let the LLM hallucinate target prices. v2 computes fractal-based support/resistance levels and includes them in the prompt, with explicit instructions:

> "Target price must be based on the nearest resistance (for buys) or support (for sells)."
> "Stop loss must be set using ATR: current_price - 2×ATR for buys."

This produces realistic, actionable targets instead of round-number fantasies.

## Edge #7: ATR-Based Position Sizing Context

v1 had no volatility awareness. A €50 stock with ATR=€0.50 (1% daily range) and a €50 stock with ATR=€3.00 (6% daily range) would get the same treatment. v2 includes:

- ATR value in prompt
- ATR expansion ratio (current ATR vs 20-period mean ATR)
- Automatic stop-loss computation: price - 2×ATR

This is how professional risk management works.

## Edge #8: Stochastic Oscillator

RSI alone misses some reversals. The Stochastic K/D oscillator is better at catching short-term overbought/oversold conditions, especially in ranging markets. v2 uses both, and the quant scorer weighs them appropriately per regime.

## Edge #9: Candlestick Pattern Detection

Basic but effective: bullish/bearish engulfing, doji, hammer, shooting star. These are the patterns with the highest statistical significance. They're included in the prompt so the LLM can factor them into timing.

## Edge #10: Recommendation Persistence

Every recommendation is saved to a JSONL file per ticker. This enables:

- Backtesting: Did the bot's STRONG_BUY signals actually lead to gains?
- Self-improvement: Track quant/LLM agreement rate over time
- Pattern analysis: Do certain regimes produce better signals?

---

## Quantitative Scoring Breakdown

The composite score (-1 to +1) is a weighted sum:

| Component | Weight | What it measures |
|-----------|--------|------------------|
| Trend | 25% | SMA alignment (price > SMA20 > SMA50 > SMA200), ADX-scaled |
| Momentum | 20% | RSI zones + MACD histogram direction + Stochastic |
| Volume | 15% | OBV trend + divergences, MFI extremes, A/D line |
| Volatility | 10% | Bollinger Band position (near lower = bullish, near upper = bearish) |
| Relative Strength | 15% | 20-day performance vs Euro Stoxx 50 |
| Structure | 15% | Proximity to support (bullish) or resistance (bearish) |

Thresholds:
- Score > +0.60 → STRONG_BUY
- Score > +0.30 → BUY
- Score -0.30 to +0.30 → HOLD
- Score < -0.30 → SELL
- Score < -0.60 → STRONG_SELL

---

## Configuration Tuning Guide

### Most impactful parameters to tune:

1. **`SCORE_WEIGHTS`** — Increase `momentum` weight in trending markets, increase `structure` weight in ranging markets
2. **`REGIME.trending_adx`** — Lower from 25 to 20 if your stocks trend weakly; raise to 30 for highly liquid stocks
3. **`INDICATORS.pivot_lookback`** — Increase from 5 to 8 for less noisy S/R levels
4. **`OLLAMA_MODEL`** — Upgrade to a larger model (gemma3:12b or mistral) for better reasoning if your hardware allows

### What NOT to change:
- RSI period (14 is universal, changing it puts you out of sync with what everyone else sees)
- MACD parameters (12/26/9 is standard for the same reason)
- Wilder smoothing method (this IS the standard)

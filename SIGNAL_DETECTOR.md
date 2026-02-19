# Signal Detector — The Timing Edge

## The Problem It Solves

You asked the right question: "buy before it goes up, sell before it goes down."
The honest answer is you can't beat HFT on speed. But you **can** read the
footprint of institutional orders *as they execute* — which is the next best thing.

## How Institutions Actually Trade

When BlackRock wants to buy €50M of ASML, they don't place one order. They break
it into thousands of smaller orders executed over hours or days. They prefer to
buy at **key support levels** because:

1. There's **liquidity** there (stop-losses and limit orders pool at S/R levels)
2. They get a **better average price** buying where sellers are concentrated
3. Their buying **absorbs the selling pressure**, which is why the price then bounces

This creates a detectable pattern: **volume spikes at key price levels**.

## The 7 Signal Types

### CRITICAL Priority (act now)

**BREAKOUT_CONFIRMED** — Price closes beyond a S/R level on 2x+ volume.
This is the strongest signal. It means institutional money just pushed through
a contested level with enough conviction to trigger stop-losses on the other side.
Once a level breaks on volume, it tends to accelerate.

**VOLUME_CLIMAX** — Extreme volume (4x+) with a rejection candle (long wick).
This is an exhaustion signal. The last burst of frantic buying/selling before
everyone who wanted in/out has acted. Often marks exact tops and bottoms.

### HIGH Priority (strong setup)

**ACCUMULATION_AT_SUPPORT** — Volume spikes 2x+ while price holds at support.
This is the classic "smart money" footprint. Big buyer is absorbing all selling
at this level. Price is about to bounce because the supply is being removed.

**DISTRIBUTION_AT_RESISTANCE** — Volume spikes at resistance, price can't break through.
Institutions are selling into retail enthusiasm. Price is about to reverse down.

### MEDIUM Priority (developing)

**VOLUME_DRY_UP** — Price tests a level on declining/low volume (<50% avg).
The test lacks conviction. The level will hold. Useful for confirming that
support is real (buy signal) or resistance is real (sell signal).

**VWAP_DEVIATION** — Price extended >2% from 20-day VWAP.
Institutional traders use VWAP as a benchmark. Extended deviations revert.
This is a mean-reversion signal.

**VOLUME_BREAKOUT_PRELUDE** — 3+ bars of rising volume with contracting price range
near a key level. Pressure is building. Like a spring being compressed.
Breakout is imminent (direction TBD, but the move will be significant).

## Volume Profile

In addition to real-time signals, the detector builds a **Volume Profile** —
a histogram of where the most volume was traded over the past 60 days.

Key concepts:
- **POC (Point of Control)**: The price with the most volume. Price gravitates here.
- **Value Area**: The price range containing 70% of all volume. "Fair value."
- **Above Value Area**: Price is extended, institutional selling pressure expected.
- **Below Value Area**: Price is at a discount, institutional buying pressure expected.

## Integration Architecture

```
DataCollector                    SignalDetector
  │                                │
  ├─ Computes S/R levels ─────────►├─ Checks price proximity to each level
  ├─ Computes volume data ─────────►├─ Checks volume vs 20-day average
  ├─ Computes candlestick ─────────►├─ Checks for rejection wicks
  │                                │
  │                          ┌─────┤  Produces: List[Signal]
  │                          │     │  Produces: VolumeProfile
  │                          │     │
  MarketData ◄───────────────┘     │
     │  .active_signals            │
     │  .volume_profile            │
     │                             │
     ▼                             │
  LLMAnalyzer                      │
     │  Prompt includes:           │
     │  ═══ ⚡ ACTIVE ALERTS ═══   │
     │  [CRITICAL] BREAKOUT...     │
     │  ═══ VOLUME PROFILE ═══    │
     │  POC: €742.30              │
     │                             │
     │  Rules 6-8 tell LLM:       │
     │  "CRITICAL signals OVERRIDE │
     │   regime caution"           │
     ▼                             │
  Recommendation                   │
     .active_signals (persisted)   │
```

## How It Changes LLM Behavior

Three new rules were added to the LLM prompt:

> **Rule 6**: CRITICAL or HIGH volume-at-level alerts OVERRIDE regime caution.
> If accumulation is detected at support, you may raise conviction even in
> a ranging market.

> **Rule 7**: If a BREAKOUT_CONFIRMED signal is active, increase confidence
> by 0.1-0.2. This is the strongest possible signal.

> **Rule 8**: If VOLUME_CLIMAX is active, treat it as a reversal signal —
> it may justify going against the trend.

This means the LLM now has a hierarchy:
1. Quant score sets the baseline direction
2. Signals can boost confidence or override regime caution
3. The ±1 step constraint still prevents wild deviations

## Cooldown System

Signals have a 5-bar cooldown to prevent the same alert from firing every cycle.
Once a BREAKOUT_CONFIRMED fires, it won't fire again for 5 analysis cycles,
preventing alert fatigue and repeated trades on the same event.

## Files Modified

| File | Changes |
|------|---------|
| `signal_detector.py` | **NEW** — entire module |
| `config.py` | Added `HISTORY_DIR` for recommendation persistence |
| `data_collector.py` | Added `active_signals` and `volume_profile` fields to MarketData |
| `llm_analyzer.py` | Added signal formatting in prompt, rules 6-8, signal persistence |
| `display.py` | Added signal alerts panel, volume profile panel, signal indicators in recommendation cards |
| `bot.py` | Added SignalDetector integration, `_detect_signals()` step in main loop |

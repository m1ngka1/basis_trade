# Signals from equity, SSF and fair value

Implementation: [price_signals.py](price_signals.py). These are separate research
hypotheses, not proven profitable signals or additions automatically enabled in C.
Positive scores propose cash longs; negative scores propose cash shorts. Unlike
the original basis modifier, these candidates can disagree with the basis sign.

**Relationship to the original idea:** All eight ideas in the table are
standalone signal hypotheses to research **in parallel with the original basis
strategy**, not enhancements or T/C/R modifiers of that strategy. Evaluate each
independently against the original B and B+EPS benchmarks. Any later combination
is a separate experiment and requires evidence of incremental value.

Assume `FV` is the theoretical **futures price**, not a premium or an equity
valuation. With positive, synchronized prices:

```text
s = log(equity)
g = log(SSF / FV)
k = log(FV / equity)
q = log(SSF) - k
c = change in s over the current window
f = change in q over the current window
d = change in g = f - c
```

The carry adjustment is an accounting decomposition using the supplied FV model;
it is not an independently observed equity forecast. An FV revision can cause a
gap move without new trading information. Prices and FV must use consistent
dividend/corporate-action conventions; exclude/reset rolls and FV model breaks.

| # | Idea / Python output | Cash-long rule (reverse signs for shorts) | Inputs |
|---|---|---|---|
| 1 | Futures-led breakout / `futures_breakout` | Adjusted futures rise materially and outpace cash. | Synchronized equity, SSF, FV prices and lagged volatility |
| 2 | Improving discount / `improving_discount` | Same positive futures-led move while SSF remains below FV. | Same prices, including current gap sign |
| 3 | Cash-led gap opening / `cash_reversal` | Cash falls materially while adjusted futures stay approximately flat. | Same-window equity and adjusted futures changes |
| 4 | Cash catch-up / `cash_catchup` | Previous window had futures-led widening; cash now rises, futures retain gains, and a positive gap narrows but remains. | Two adjacent price windows and current gap |
| 5 | Futures rejection / `futures_rejection` | For a short: previous futures-led rise retraces substantially, while cash remains elevated and has not followed futures down. | Two adjacent price windows; configurable retracement fraction |
| 6 | Relative strength in a pullback / `pullback_strength` | After both prices rose, both pull back; adjusted futures fall less than cash. | Two adjacent equity/adjusted-futures windows |
| 7 | Broad futures movement / `maturity_agreement` | A futures-led move agrees across a configured fraction of maturities. | Equity plus synchronized SSF/FV pairs for at least two distinct expiries |
| 8 | Learned cash adjustment / `learned_cash_adjustment()` | A rolling pooled ridge regression forecasts a positive cash return from current gap and recent cash/futures moves. | Historical prices, configured execution delay/horizon; only matured labels used in training |

## Exact implementation conventions

- Ideas 1–7 return scores in [-1, 1], not return forecasts. Score strength is
  `tanh` of the relevant move divided by its own lagged volatility. Thresholds
  and lookbacks are configurable starting choices, not calibrated constants.
- Each row is one observation: EOD or an intraday bar endpoint. Windows, delay
  and horizon count rows. Run separately for local calendars, session treatment,
  and comparable markets. Intraday overnight transitions require an explicit
  upstream policy; this module does not infer session boundaries.
- Two-window ideas compare nonoverlapping windows ending at t-window and t.
  The previous setup uses its historical scale. No future confirmation is used.
- NaN means unavailable inputs/history; zero means observed but no setup.
  Missing observations inside a return window invalidate it. Maturity agreement
  requires all configured expiries observed, preventing changing coverage from
  silently becoming stronger agreement. With no extra expiries it is NaN.
- Ridge returns an expected cash **log return**, not a bounded score. Inputs are
  standardized within each training fit. Fits stop at outcomes ending t-1;
  signal at s targets close(s+delay) to close(s+delay+horizon). This is not a
  next-open target. It pools stocks within the supplied comparable universe.
- The pooled learner is deliberately small. It has no costs, factor controls,
  sector pooling or uncertainty calibration; these are evaluation/integration
  work, not assumed benefits. Count dates for minimum history, not stock rows.

## First experiments

Start with 1, 2 and 4. Compare every candidate to cash momentum/reversal alone,
the original B, and the user's B+EPS. Do not average all eight: 2 is a subset
of 1, and 4–6 are different paths following earlier moves.

Test next executable cash returns, net costs, turnover and separate long/short
results. Keep portfolio gross/risk comparable. For the learner, select its
lookbacks and ridge penalty on inner walk-forward data and report untouched
outer periods. Use date-block uncertainty for overlapping labels and cross-stock
dependence. Compare predicted performance by market and structural-discount state.

## Usage

```python
from price_signals import PriceConfig, calculate_price_signals, learned_cash_adjustment

cfg = PriceConfig()
# All inputs are identically indexed wide DataFrames, dates x stocks.
scores = calculate_price_signals(equity, ssf, fv, cfg)
# Optional: extra_expiries={"next": (next_ssf, next_fv)} adds idea 7.
forecast = learned_cash_adjustment(equity, ssf, fv, cfg)
```

Run the synthetic demo with `uv run --with numpy --with pandas price_signals.py`.
Tests: `uv run --with numpy --with pandas --with pytest pytest -q test_price_signals.py`.

Mechanism references (not validation of these strategies):
- [SSF fair value: dividends, financing and maturity](https://www.cmegroup.com/articles/faqs/faq-single-stock-futures.html)
- [Information revelation in SSFs](https://eprints.exchange.isb.edu/id/eprint/117/)

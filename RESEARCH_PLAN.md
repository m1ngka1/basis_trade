# Basis signal research plan

Saved: 2026-09-07. Status: proposed experiments, not validated improvements.

## Objective and methodology to preserve

Predict cash-equity residual returns using corresponding single-stock-futures
richness: long cash names with relatively rich futures and short names with
relatively cheap futures. The strategy trades cash; it is not a two-leg basis
convergence strategy.

- B: core cleaned, carry/maturity-adjusted basis score, bounded to [-1, 1].
- T: timing of the basis opportunity.
- C: confirmation aligned with the direction of B.
- R: historical relationship between B and subsequently realized cash returns.
- confidence = (T + C + R) / 3.
- signal = B * (1 + 0.5 * confidence).

The multiplier is bounded to [0.5, 1.5]. Confidence is a conviction score,
not a probability or statistical confidence level. Preserve this baseline;
evaluate methodology changes as explicit alternatives.

## Review findings

The core arithmetic, directional confirmation, and historical-return alignment
look consistent with the intended strategy. With horizon=5 and entry_delay=1,
a signal at close s targets the cash return from close s+1 to close s+6.
Historical outcomes must mature, and R is additionally lagged one session.

Verification performed on the original scripts:

- Running test_basis_tcr.py passed its bounds, direction, prefix-invariance,
  confirmation, missing-data, and return-alignment assertions.
- An additional full sign-reversal check passed: reversing basis, B, cash
  returns, futures returns, and directional confirmation leaves T/C/R unchanged
  and reverses the final signal.
- pytest collected no tests and exited with code 5: assertions currently run
  at module import rather than inside named test functions.
- No real-market predictive performance has been established.

Concerns to investigate:

1. R scaling may be aggressive. With default shrinkage and response_scale,
   partial correlation 0.10 gives R approximately 0.46 at 126 observations and
   0.66 at 504 observations. Overlapping holding-period labels are counted
   individually; the shrinkage count is not an effective sample size.
   One exploratory independent-noise simulation produced |R| > 0.5 for 10.6%
   of sampled stock-dates. This is a sensitivity example, not an estimated
   false-positive rate or a reproducible benchmark currently in the repository.
2. C uses same-date RMS normalization, so multiplying an entire feature by a
   positive constant leaves its score unchanged. Uniform values of 1e-9 and
   0.9 both normalize to approximately 0.462. Absolute activity strength is lost.
3. Every new qualifying shock resets event age and accumulated cash response.
   A prolonged episode can appear fresh again. Distinguish latest-shock age
   from the age and accumulated response of the broader opportunity.
4. Cash catch-up includes the shock-day return. This is consistent with response
   observed by the decision close, but daily data cannot identify the order of
   cash and futures moves within that day.
5. T components may repeat related evidence; equal weighting does not make
   them independent. Missing components also change effective within-group
   weights. Expose component availability alongside the scores.
6. Prefix invariance does not validate upstream point-in-time data: basis
   cleaning, factor residualization, corporate actions, universe membership,
   and publication timestamps need separate checks.

## First experiment: establish which groups add value

Compare the following using the same data, universe, execution assumptions,
portfolio construction, risk controls, and costs:

| Variant | Signal |
| --- | --- |
| Basis only | B |
| Timing only modifier | B * (1 + 0.5 * T / 3) |
| Confirmation only modifier | B * (1 + 0.5 * C / 3) |
| Historical response only modifier | B * (1 + 0.5 * R / 3) |
| Full model | B * (1 + 0.5 * (T + C + R) / 3) |

For this ablation, set excluded groups to zero and retain the denominator of
three, preserving each group's contribution in the full model. Any experiment
that gives a single group the full confidence budget is a separate variant.
Also evaluate leave-one-group-out variants to measure conditional contribution.

Evaluation requirements:

- Date-based walk-forward development and validation; keep all stocks from
  the same date together.
- Exclude training labels whose outcomes are not yet available at each cutoff.
- Select features, signs, scales, and thresholds using development data only.
- Measure incremental cross-sectional cash-return IC, net portfolio returns,
  turnover, exposures, and separate long/short contributions.
- Use identical portfolio normalization across variants so changes in gross
  exposure are not mistaken for better stock selection.
- Account for cash execution, financing, and borrow costs as applicable.
- Use date blocks for uncertainty assessment, respecting overlapping labels
  and cross-stock dependence. Record all attempted variants.
- Inspect results by time period and market; prefer stable contributions over
  one favorable aggregate result. Define acceptance criteria before testing.

## Prioritized enhancements

### 1. Stabilize historical response R

- Partially pool stock-level estimates toward country/sector estimates.
- Compare pooled cross-sectional basis-response estimation with the current
  stock-by-stock time-series partial correlation.
- Compare conservative response scaling and shrinkage alternatives.
- Evaluate uncertainty with date blocks rather than treating overlapping
  returns as independent observations.
- Retain the sign of historical evidence; do not rank negative relationships
  into positive support merely because peers have worse estimates.

### 2. Measure opportunity survival through execution

- Where data permit, compare executable next-open and next-close targets.
- Separate immediate cash response from remaining holding-period response.
- Study response across a small, predeclared set of holding horizons.
- Keep timestamp conventions explicit; next-open evaluation requires new
  price inputs and cannot be inferred from close-to-close returns alone.

### 3. Measure futures-information reliability

- Exclude stale/asynchronous or otherwise invalid quotes.
- Test interactions with futures/cash relative spreads, abnormal futures
  turnover, and reliable participation measures.
- Compare feature-specific normalization with current same-date RMS scaling.
- Separate directional evidence from its activity or reliability measure.

### 4. Separate persistent richness from new information

- Decompose basis into slow richness and innovation; test each against cash
  returns and evaluate their interaction.
- Track both latest-shock age and broader episode age/cumulative cash response.
- Test agreement across maturities after appropriate carry and roll cleaning.
- Examine whether persistent financing, dividend, or borrow effects explain
  basis levels without predicting subsequent cash returns.

### 5. Test long/short asymmetry

- Compare response relationships conditional on the sign of B.
- Examine borrow conditions for cheap-futures names.
- Pool estimates to avoid unstable fits on small conditional samples.
- Preserve the symmetric baseline for comparison; economic asymmetry is a
  hypothesis, not a reason to hard-code new signs.

## Repository and test work

Before market experiments, turn the scripts into the originally requested
extensible Python repository: src package, configuration-driven parameters,
README, example usage, dependency/project configuration, linting, and tests.
Keep a regression comparison against the original implementation.

Convert existing assertions into named tests and add focused coverage for:

- Full long/short sign symmetry.
- Missing-data gaps, missing groups, and component availability.
- Repeated shocks, direction changes, and shock-day catch-up conventions.
- Return alignment across multiple horizons and entry delays.
- Degenerate/constant inputs to the historical partial correlation.
- Confirmation scaling, outliers, and changing universe coverage.
- Validation of parameters, axes, and non-finite input values.

## Later modeling alternatives

Only after the fixed model and ablations establish a baseline, consider a
regularized interaction regression using B, B*T, B*C, and B*R. If direction
preservation is required, constrain the effective coefficient on B to remain
nonnegative. LightGBM should be a later comparator under the same validation
and execution assumptions.

## Context and source

Origin: the "Basis Trade Features Table" conversation and the repository review
on 2026-09-07. These are research proposals, not demonstrated sources of alpha.

Mechanism context: Shastri, Thirumalai and Zutter (2008),
[Information revelation in the futures market: Evidence from single stock futures](https://eprints.exchange.isb.edu/id/eprint/117/).
The study supports investigating futures price discovery; it does not validate
this daily signal, these parameter choices, or profitability in our universe.

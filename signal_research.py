"""APAC cash-equity / SSF research scaffold. Run: python signal_research.py

Dependencies: numpy, pandas (pytest for test_signal_research.py).
Temporary environment: uv run --with numpy --with pandas signal_research.py

HANDOFF TO THE NEXT AGENT
=========================
User evidence: C currently contains only EPS estimates/revisions and helps in
their backtest. Original T/R have not helped. Some stocks remain at a futures
discount without conveying new bearish information. No market data are here.

Do NOT enable every idea below. Preserve basis_tcr.py as the original benchmark.
This module implements transparent candidate calculations, a synthetic demo,
and a field/adapter inventory. It does not claim improved market performance.

Research priority:
  1. Keep the user's exact B+EPS implementation as an external benchmark.
  2. Test whether persistent-discount shorts explain poor outcomes.
  3. Test cash trade pressure beyond EPS AND same-window cash returns.
  4. Test company releases and overseas information, one family at a time.
  5. Only then test execution timing or learned historical reliability.

Persistent discounts are not automatically alpha, but neither are they proof
of no alpha. The old [0.5, 1.5] multiplier CANNOT veto any nonzero B. A separate
gate can. The lagged normal-basis estimate below is a statistical comparator,
NOT fair value or new information. It can mistakenly remove a real persistent
signal or adapt to a sustained new regime: evaluate it as an ablation.

T redesign: optional execution readiness from fresh, independent observations
(cash pressure, auction, quotes), validated at the actual entry horizon. Avoid
rewarding the same flow in both C and T. No automatic price-age decay here.
Original R measures historical cash response, not fair value. Neither T nor R
needs to add new information to be useful, but both must earn their complexity.
R redesign: optional out-of-sample reliability of incremental net cash payoff,
pooled by market/sector/direction where justified. Do not turn a noisy in-sample
correlation into confidence. Training is explicitly a data-dependent adapter.

INPUT CONTRACT
All daily frames: identical increasing DatetimeIndex and stock columns, one
local trading calendar per run. Use only available observations, with original
publication timestamps and historical vintages. EPS and other evidence passed
to compose_signals must ALREADY be signed bullish scores in [-1, 1]. A score
is not a probability. Missing is NaN; known no-event/no-pressure can be zero.
Never feed FY1 roll jumps, retrospective consensus, or unsigned trade volume
as earnings revisions or signed flow. Stock-level participant data must really
be stock-level; market-wide foreign flows are not individual-stock flows.

Upstream basis must be roll/corporate-action clean and carry/maturity adjusted.
Do not subtract borrow/carry twice. An optional normal_basis adapter must have
the SAME units/adjustments and be estimated before each decision. Do not rank
residuals to force a short when every residual is zero. Screens for executable
quotes, borrow availability, and contract validity precede portfolio sizing.

Sources motivating hypotheses, not evidence for this implementation:
https://www.cmegroup.com/articles/faqs/faq-single-stock-futures.html
https://arxiv.org/abs/1011.6402  (impact is not necessarily future alpha)
https://arxiv.org/abs/1512.03492 (one-tick prediction is not multi-day alpha)
https://www.nber.org/papers/w10925 (opening buyer-initiated options volume)
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2758776 (economic links)
https://www.jpx.co.jp/english/equities/listing/disclosure/xbrl/03.html
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Idea:
    family: str
    fields: str
    recipe: str
    caution: str


# All recipes are hypotheses. Positive directional evidence must mean bullish.
# Features with ambiguous signs belong in diagnostics/conditioning until tested.
# Select one representative per family before adding correlated variants.
IDEAS = {
    "eps": Idea("expectations", "analyst, fiscal period, estimate, available_at",
                "Keep current working EPS score; compare revision breadth later.",
                "Match fiscal periods; exclude mechanical FY rolls and revisions of history."),
    "guidance": Idea("fundamentals", "new guidance, old guidance, pre-release consensus, release time",
                     "Scale new guidance minus pre-release expectation; decay from publication.",
                     "EPS revisions may repeat this event; measure incremental contribution."),
    "operating_release": Idea("fundamentals", "sales/bookings/shipments/margins, seasonal history, release time",
                              "Surprise versus a lagged seasonal expectation or as-of consensus.",
                              "Use a sector-relevant KPI; calendar effects are not demand news."),
    "news": Idea("fundamentals", "original release text, entity, event type, original timestamp",
                 "Start with structured contracts, guidance, regulatory decisions; signed event surprise.",
                 "Deduplicate syndication; generic positive tone is not positive surprise."),
    "cash_pressure": Idea("cash_demand", "buyer/seller initiated notional, total turnover, quotes, timestamps",
                          "Use trade_pressure; signed imbalance with an activity gate.",
                          "Compare against cash return; classified flow is not investor identity."),
    "pressure_persistence": Idea("cash_demand", "intraday signed flow by interval",
                                 "Use intraday_summary; breadth of net buying across active intervals.",
                                 "Condition/replace pressure, rather than another full independent vote."),
    "pressure_change": Idea("cash_demand", "same intraday fields",
                            "Late continuous-session pressure minus earlier-session pressure.",
                            "Predeclare local-session windows, not many optimized clock cutoffs."),
    "participant_flow": Idea("cash_demand", "stock-level buys/sells by investor category, release time",
                             "Net buying / usual turnover; compare categories and persistence.",
                             "Foreign/institutional is not automatically informed; account for publication lag."),
    "futures_pressure": Idea("derivatives", "signed futures trades, notional, expiry, roll flags, timestamps",
                             "Signed trade imbalance across comparable expiries; test against cash flow.",
                             "Can duplicate B; rolls and hedges do not necessarily express stock views."),
    "options_flow": Idea("derivatives", "aggressor, delta, quantity, multiplier, opening flag, spread trade ID",
                         "Sum aggressor_sign * delta * quantity * multiplier; scale by absolute exposure.",
                         "Aggregate strategy legs; customer buys and dealer buys are different; raw put/call is insufficient."),
    "oi_confirmation": Idea("derivatives", "roll-adjusted OI, signed futures flow, publication time",
                            "Condition signed flow strength on position growth.",
                            "OI never supplies direction; not an independent vote beside the same flow."),
    "overseas_listing": Idea("external", "ADR/overseas prices, FX, conversion ratio, synchronized timestamps",
                             "Observe same-business move while local market closed, adjusted for FX.",
                             "Share classes may differ; avoid asynchronous spread artifacts."),
    "linked_company": Idea("external", "historical customer/supplier links, weights, overseas releases/returns",
                           "Small predefined economically linked basket's surprises available before local entry.",
                           "Customer demand and competitor market-share gains need different signs."),
    "input_cost_fx": Idea("external", "commodity/FX changes, historical revenue/cost exposures",
                          "Exposure-signed unexpected change in relevant input price or FX.",
                          "Hedges and producer/consumer roles matter; avoid a disguised country/sector bet."),
    "buyback": Idea("capital_flows", "executed shares/value, publication time, usual turnover",
                    "Disclosed actual repurchases / usual turnover; event continuation hypothesis.",
                    "Authorization is not execution; purchases before disclosure cannot be used early."),
    "insider": Idea("capital_flows", "disclosure time, purchase/sale, value, discretionary/scheduled flag",
                    "Discretionary open-market buying, scaled by normal activity.",
                    "Selling is not symmetric bearish evidence; exclude compensation transactions."),
    "passive_demand": Idea("capital_flows", "announced index changes, effective date, weights, assets, turnover",
                           "Expected net cash demand / normal turnover; explicit event horizon.",
                           "Anticipation and post-event reversal; ETF flows and index changes can overlap."),
    "credit": Idea("external", "bond/CDS spreads, issuer mapping, duration, quote timestamps",
                   "Idiosyncratic credit improvement as candidate equity confirmation.",
                   "Sparse/stale quotes; corporate actions can benefit debt and hurt equity."),
    "auction": Idea("execution", "imbalance side/quantity, paired volume, indicative price, timestamps",
                    "Persistent imbalance / historical auction volume before executable cutoff.",
                    "May only predict auction impact, then reverse; separate from continuous session."),
    "order_book": Idea("execution", "best bid/ask prices/sizes, additions/cancels/trades, timestamps",
                       "Dynamic best-quote order-flow imbalance normalized by depth.",
                       "Static queue size can cancel; one-tick prediction may have no daily value."),
    "absorption": Idea("diagnostic", "signed trades, cash midquote returns, liquidity",
                       "Tabulate flow versus price response; test conditional remaining returns.",
                       "Heavy buying with flat prices can mean passive selling, not bullish accumulation."),
    "borrow": Idea("eligibility", "loan balances, supply, fee, utilization, locates, release time",
                   "Explain structural discounts and short feasibility; separate supply and demand changes.",
                   "High fee alone is not bearish; include costs, recalls and lendable supply."),
    "term_structure": Idea("diagnostic", "carry-adjusted basis across expiries, liquidity, maturity",
                           "Compare stock-wide demand with one-contract distortions.",
                           "Multiple expiries are not independent confirmations of the same trade."),
    "attention": Idea("diagnostic", "coverage, release load, abnormal readership/volume, available_at",
                      "Condition response to signed news on attention; no direction from attention alone.",
                      "Popularity alone is not bullish; avoid extensive alternative-data collection."),
}


@dataclass(frozen=True)
class Config:
    history: int = 120
    min_history: int = 60
    basis_scale_floor: float = 0.0001  # One basis point if basis is a decimal ratio.
    chronic_fraction: float = 0.8
    dead_zone: float = 0.5  # Standardized incremental dislocation.
    full_strength: float = 2.0
    gate_mode: str = "chronic_short"  # alternatives: all, off
    enabled: tuple[str, ...] = ("eps",)
    c_strength: float = 1 / 6  # Original formula with T=R=0: B*(1+C/6).
    use_execution: bool = False
    use_reliability: bool = False

    def __post_init__(self):
        if (type(self.history) is not int or type(self.min_history) is not int
                or not 3 <= self.min_history <= self.history):
            raise ValueError("Require integer 3 <= min_history <= history")
        numbers = [self.basis_scale_floor, self.chronic_fraction, self.dead_zone,
                   self.full_strength, self.c_strength]
        if not np.isfinite(numbers).all():
            raise ValueError("Configuration must be finite")
        if (self.basis_scale_floor <= 0 or not 0 < self.chronic_fraction <= 1
                or not 0 <= self.dead_zone < self.full_strength
                or not 0 <= self.c_strength <= 1):
            raise ValueError("Invalid scales, gate thresholds or C strength")
        if self.gate_mode not in {"chronic_short", "all", "off"}:
            raise ValueError("Unknown gate mode")
        if len(set(self.enabled)) != len(self.enabled):
            raise ValueError("Duplicate feature")
        for name in self.enabled:
            if name not in IDEAS or IDEAS[name].family in {"execution", "diagnostic", "eligibility"}:
                raise ValueError(f"Not a directional C feature: {name}")


def aligned(x: pd.DataFrame, like: pd.DataFrame, name: str,
            bounds: tuple[float, float] | None = None) -> pd.DataFrame:
    if (not isinstance(x, pd.DataFrame) or not x.index.equals(like.index)
            or not x.columns.equals(like.columns)):
        raise ValueError(f"{name}: axes must match exactly")
    x = x.astype(float)
    if np.isinf(x.to_numpy()).any():
        raise ValueError(f"{name}: infinite input")
    if bounds and ((x < bounds[0]) | (x > bounds[1])).any().any():
        raise ValueError(f"{name}: expected {bounds}")
    return x


def validate_grid(x: pd.DataFrame) -> None:
    if (not isinstance(x, pd.DataFrame) or x.empty
            or not isinstance(x.index, pd.DatetimeIndex)
            or not x.index.is_unique or not x.index.is_monotonic_increasing
            or x.index.hasnans or not x.columns.is_unique):
        raise ValueError("Use a nonempty sorted unique daily grid")


def signed_score(raw: pd.DataFrame, scale: float) -> pd.DataFrame:
    """Economic zero preserved; fixed, development-only scale, no row RMS.

    Scale is in raw units and must be chosen by feature/market using development
    data. For known bounded scores (e.g. existing EPS), use them directly.
    """
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Positive finite scale required")
    if np.isinf(raw.to_numpy(dtype=float)).any():
        raise ValueError("Infinite raw score")
    return np.tanh(raw / scale)


def trade_pressure(buy: pd.DataFrame, sell: pd.DataFrame,
                   expected_notional: pd.DataFrame) -> pd.DataFrame:
    """Signed aggressive flow weighted by activity versus past comparable windows.

    expected_notional must be positive, lagged and seasonality-adjusted upstream.
    This activity gate prevents a tiny classified trade receiving full strength.
    Missing and no-classified-trade observations are NaN, not fabricated zeros.
    """
    sell = aligned(sell, buy, "sell")
    expected = aligned(expected_notional, buy, "expected_notional")
    buy = aligned(buy, buy, "buy")
    if (buy.lt(0) | sell.lt(0) | expected.le(0)).any().any():
        raise ValueError("Nonnegative trades and positive expected notional required")
    total = buy + sell
    return ((buy - sell) / total.where(total > 0)
            * (total / expected).clip(0, 1))


def intraday_summary(bars: pd.DataFrame, session_start: pd.Timestamp,
                     cutoff: pd.Timestamp) -> pd.DataFrame:
    """One local session -> stock rows; caller maps these onto daily decision grid.

    Required columns: timestamp (timezone aware), stock, buy, sell, continuous.
    buy/sell are classified aggressive notional in FIXED, NONOVERLAPPING bars.
    timestamp is bar END/availability time; never use a partly future bar.
    Explicit session boundaries avoid mixing overnight sessions or APAC clocks.
    Auctions are excluded. Missing active bars make that stock's result NaN.
    Classification coverage and expected activity require upstream feed QA.
    """
    required = {"timestamp", "stock", "buy", "sell", "continuous"}
    if not required <= set(bars):
        raise ValueError(f"Missing columns: {required - set(bars)}")
    start, end = pd.Timestamp(session_start), pd.Timestamp(cutoff)
    if start.tzinfo is None or end.tzinfo is None or start > end:
        raise ValueError("Ordered timezone-aware session boundaries required")
    x = bars.copy()
    x["timestamp"] = pd.to_datetime(x["timestamp"])
    if x["timestamp"].dt.tz is None or x["timestamp"].isna().any():
        raise ValueError("Bar timestamps must be timezone aware and nonmissing")
    if x["continuous"].isna().any() or not x["continuous"].isin([True, False]).all():
        raise ValueError("Explicit continuous-session flags required")
    x = x.loc[x.timestamp.between(start, end) & x.continuous.astype(bool)]
    if x.stock.isna().any() or x.duplicated(["timestamp", "stock"]).any():
        raise ValueError("Missing stock or duplicate bar")
    if (x[["buy", "sell"]].lt(0).any().any()
            or np.isinf(x[["buy", "sell"]].to_numpy(dtype=float)).any()):
        raise ValueError("Invalid notional")
    rows = {}
    for stock, group in x.groupby("stock", sort=False):
        total = group.buy + group.sell
        net = group.buy - group.sell
        active = total > 0
        ok = group[["buy", "sell"]].notna().all().all() and active.any()
        rows[stock] = {
            "imbalance": net.sum() / total.sum() if ok else np.nan,
            "persistence": np.sign(net[active]).mean() if ok else np.nan,
            "classified_notional": total.sum() if ok else np.nan,
            "active_bars": int(active.sum()),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def structural_state(basis: pd.DataFrame, B: pd.DataFrame, config: Config,
                     normal_basis: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Gate existing direction, never create the opposite position.

    Default: only shorts in persistently negative-basis names require incremental
    negative dislocation. Historical warmup is unknown -> NaN for shorts. Longs
    are unaffected. 'all' requires direction-aligned dislocation for both sides.
    External normal_basis may condition on maturity, borrow, funding, sector,
    and dividends; fit on past data, preserve units, and reset on contract breaks.
    """
    past = basis.shift(1)
    roll = past.rolling(config.history, min_periods=config.min_history)
    normal = roll.median() if normal_basis is None else aligned(normal_basis, basis, "normal_basis")
    scale = ((roll.quantile(.75) - roll.quantile(.25)) / 1.349).clip(
        lower=config.basis_scale_floor)
    fraction = past.lt(0).astype(float).where(past.notna()).rolling(
        config.history, min_periods=config.min_history).mean()
    excess = basis - normal
    directional = np.sign(B) * excess / scale
    novelty = ((directional - config.dead_zone)
               / (config.full_strength - config.dead_zone)).clip(0, 1)
    gate = pd.DataFrame(1.0, index=basis.index, columns=basis.columns)
    if config.gate_mode == "all":
        gate = novelty
    elif config.gate_mode == "chronic_short":
        chronic_short = B.lt(0) & fraction.ge(config.chronic_fraction)
        gate = gate.mask(chronic_short, novelty)
        gate = gate.mask(B.lt(0) & fraction.isna())
    return {"normal_basis": normal, "excess_basis": excess,
            "basis_scale": scale, "discount_fraction": fraction,
            "structural_gate": gate.where(basis.notna() & B.notna())}


def compose_signals(basis: pd.DataFrame, B: pd.DataFrame,
                    evidence: Mapping[str, pd.DataFrame], config: Config = Config(),
                    *, normal_basis: pd.DataFrame | None = None,
                    eligible: pd.DataFrame | None = None,
                    execution_gate: pd.DataFrame | None = None,
                    reliability_gate: pd.DataFrame | None = None) -> dict[str, pd.DataFrame]:
    """Return named ablations, not orders or calibrated expected returns.

    Evidence only enters if explicitly enabled. Families get equal weight;
    enabled features within each family get equal weight. Missing evidence has
    zero contribution without redistributing weights. Coverage is returned.
    This avoids changing all other feature weights when one feed goes missing.

    eligible: optional explicit 0/1 input (missing fails closed); omit only for
    research. Production adapter must check cash liquidity/short borrow/quotes.
    Optional gates are [0,1] attenuation. When enabled, missing fails closed.
    T/R are NOT added to C. Default candidate changes only structural eligibility.
    """
    validate_grid(basis)
    basis = aligned(basis, basis, "basis")
    B = aligned(B, basis, "B", (-1, 1)).where(basis.notna())
    state = structural_state(basis, B, config, normal_basis)
    zero = pd.DataFrame(0.0, index=basis.index, columns=basis.columns)
    families: dict[str, list[pd.DataFrame]] = {}
    observed = zero.copy()
    for name in config.enabled:
        feature = (aligned(evidence[name], basis, name, (-1, 1))
                   if name in evidence else zero * np.nan)
        observed += feature.notna().astype(float)
        families.setdefault(IDEAS[name].family, []).append(feature)
    directional = zero.copy()
    family_outputs = {}
    for family, features in families.items():
        score = sum((f.fillna(0) for f in features), start=zero.copy()) / len(features)
        family_outputs[f"family_{family}"] = score
        directional += score / len(families)
    C = np.sign(B) * directional
    available = B.notna()
    mask = available.copy()
    if eligible is not None:
        eligible = aligned(eligible, basis, "eligible", (0, 1))
        if not eligible.isin([0, 1]).where(eligible.notna(), True).all().all():
            raise ValueError("eligible must be 0/1/NaN")
        mask &= eligible.eq(1)
    reference = B * (1 + config.c_strength * C)
    gated = reference * state["structural_gate"]
    candidate = gated.copy()
    for enabled, value, name in [
        (config.use_execution, execution_gate, "execution_gate"),
        (config.use_reliability, reliability_gate, "reliability_gate"),
    ]:
        if enabled:
            if value is None:
                raise ValueError(f"{name} enabled but no adapter output supplied")
            candidate *= aligned(value, basis, name, (0, 1)).fillna(0)
    outputs = {
        "B": B, "C": C, "coverage": observed / max(len(config.enabled), 1),
        "basis_only": B, "basis_C": reference,
        "basis_gate": B * state["structural_gate"],
        "basis_C_gate": gated, "candidate": candidate,
        **state, **family_outputs,
    }
    # Zero means explicitly ineligible; NaN means missing B/structural history.
    for name in ["basis_only", "basis_C", "basis_gate", "basis_C_gate", "candidate"]:
        outputs[name] = outputs[name].where(mask, 0).where(available)
    return outputs


def matured_cash_labels(B: pd.DataFrame, cash_log_return: pd.DataFrame,
                        horizon: int = 5, entry_delay: int = 1) -> dict[str, pd.DataFrame]:
    """For a future R adapter: pair signals/outcomes at maturity, then lag once.

    At decision t, last endpoint is t-1; signal date is t-1-H-delay.
    Label is close(s+delay) -> close(s+delay+H). No next-open approximation.
    Missing any holding-period return invalidates the label.
    """
    if type(horizon) is not int or horizon < 1 or type(entry_delay) is not int or entry_delay < 0:
        raise ValueError("Invalid horizon/delay")
    validate_grid(B)
    cash = aligned(cash_log_return, B, "cash_log_return")
    return {"B": B.shift(horizon + entry_delay + 1),
            "cash_label": cash.rolling(horizon, min_periods=horizon).sum().shift(1)}


def fit_reliability_adapter(*args, **kwargs):
    """DATA-DEPENDENT PLACEHOLDER: deliberately does not fabricate trained R.

    Input: archived as-of features, cash prices/labels, costs, market/sector,
    actual execution delay and original signal variants. Output: aligned [0,1]
    attenuation with fitted_at < decision time, and a separate audit table.

    Fit only matured outcomes. Prefer simple pooled regularized estimates of
    incremental cash payoff versus B+EPS, with stock shrinkage and side effects.
    Calibrate gates on an inner walk-forward split; evaluate on untouched outer
    dates. Count independent date blocks, not stocks/overlapping labels as iid.
    Need cost/factor controls, sufficient both-side history, regime coverage,
    and calibrated uncertainty. A negative estimate should attenuate, not flip.
    Compare to constant gate=1. Do not impose this layer unless it adds value.
    """
    raise NotImplementedError("Requires point-in-time data and out-of-sample calibration")


EXPERIMENTS = (
    "Reproduce user's exact B+EPS, original full TCR, and B alone at matched gross/risk.",
    "Audit chronic-discount shorts: contribution, borrow/carry, maturity, sector, earnings news.",
    "Ablate structural gate off/chronic_short/all; test persistent-alpha loss and new-shock response.",
    "Add cash pressure only; compare to cash return only and pressure conditional on that return.",
    "Replace pressure with activity/persistence-conditioned pressure; avoid duplicate family votes.",
    "Add guidance/operating surprises then overseas information individually; hold EPS fixed.",
    "Check feature alone versus interaction: standalone alpha need not be forced into C.",
    "Split returns into immediate move and remainder after ACTUAL entry; test timing separately.",
    "Only after stable evidence, fit optional pooled R adapter and compare gate=1.",
    "Walk forward by date with matured training labels; block uncertainty, costs, turnover, long/short.",
    "Report coverage/NaNs/eligible count per market; zeroing bad shorts must not merely raise leverage elsewhere.",
    "Use fixed point-in-time universe/calendars; record all trials; no random stock-date splitting.",
)


def demo() -> None:
    """Small synthetic illustration, not a backtest or performance claim."""
    dates = pd.bdate_range("2025-01-01", periods=150)
    basis = pd.DataFrame({"structural_discount": -.02, "new_bad_news": -.02,
                          "rich_futures": .01}, index=dates)
    basis.loc[dates[-1], "new_bad_news"] = -.025
    B = pd.DataFrame({"structural_discount": -.8, "new_bad_news": -.8,
                      "rich_futures": .8}, index=dates)
    eps = pd.DataFrame({"structural_discount": 0., "new_bad_news": -.6,
                        "rich_futures": .6}, index=dates)
    result = compose_signals(basis, B, {"eps": eps})
    keys = ["basis_C", "structural_gate", "candidate"]
    print("SYNTHETIC ONLY: persistent discount is not enough to retain a short.")
    print(pd.DataFrame({k: result[k].iloc[-1] for k in keys}).round(3))
    print("\nOpt-in idea inventory:")
    for name, idea in IDEAS.items():
        print(f"  {name:23} [{idea.family}] {idea.recipe}")


if __name__ == "__main__":
    demo()

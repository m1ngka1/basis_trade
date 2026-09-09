"""Eight price-only cash signal hypotheses; see PRICE_SIGNAL_IDEAS.md.

No automatic combination, positions or performance claims. FV must be an
outright theoretical futures price. All frames share positive price units,
timestamps and axes. Original basis/TCR implementations remain untouched.
"""

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import pandas as pd

from signal_research import aligned, validate_grid


@dataclass(frozen=True)
class PriceConfig:
    window: int = 3
    scale_window: int = 60
    min_scale: int = 20
    scale_floor: float = .001  # Window log-return units, NOT price units.
    move_threshold: float = .5
    flat_threshold: float = .25
    retracement: float = .5
    agreement: float = .75
    horizon: int = 5
    entry_delay: int = 1
    train_window: int = 252
    min_train_dates: int = 60
    ridge: float = 1.

    def __post_init__(self):
        for name in ("window", "scale_window", "min_scale", "horizon",
                     "train_window", "min_train_dates"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.entry_delay) is not int or self.entry_delay < 0:
            raise ValueError("Invalid entry_delay")
        if not 2 <= self.min_scale <= self.scale_window:
            raise ValueError("Invalid scale history")
        if not 3 <= self.min_train_dates <= self.train_window:
            raise ValueError("Invalid training history")
        values = (self.scale_floor, self.move_threshold, self.flat_threshold,
                  self.retracement, self.agreement, self.ridge)
        if not np.isfinite(values).all():
            raise ValueError("Nonfinite configuration")
        if (self.scale_floor <= 0 or self.move_threshold <= 0
                or not 0 <= self.flat_threshold < self.move_threshold
                or not 0 < self.retracement <= 1 or not .5 < self.agreement <= 1
                or self.ridge <= 0):
            raise ValueError("Invalid thresholds or regularization")


def _components(equity, ssf, fv, cfg):
    validate_grid(equity)
    frames = [aligned(x, equity, name) for x, name in
              [(equity, "equity"), (ssf, "ssf"), (fv, "fv")]]
    if any(x.le(0).any().any() for x in frames):
        raise ValueError("Prices must be positive or NaN")
    s, future, fair = (np.log(x) for x in frames)
    # Joint validity avoids claiming a signal where any leg is unavailable.
    valid = s.notna() & future.notna() & fair.notna()
    s, future, fair = (x.where(valid) for x in (s, future, fair))
    g = future - fair
    q = future - (fair - s)

    def move(x):
        return x.diff(cfg.window).where(
            x.rolling(cfg.window + 1).count().eq(cfg.window + 1))

    c, f, d = move(s), move(q), move(g)

    def scale(x):
        return x.shift(1).rolling(cfg.scale_window, cfg.min_scale).std().clip(
            lower=cfg.scale_floor)

    return {"g": g, "c": c, "f": f, "d": d,
            "zc": c / scale(c), "zf": f / scale(f), "zd": d / scale(d), "s": s}


def _score(strength, condition, required):
    valid = required[0].notna()
    for frame in required[1:]:
        valid &= frame.notna()
    return np.tanh(strength).where(condition, 0.).where(valid)


def calculate_price_signals(
    equity: pd.DataFrame, ssf: pd.DataFrame, fv: pd.DataFrame,
    cfg: PriceConfig = PriceConfig(), *,
    extra_expiries: Mapping[str, tuple[pd.DataFrame, pd.DataFrame]] | None = None,
) -> dict[str, pd.DataFrame]:
    """Ideas 1–7, independent [-1,1] directional scores; no inferred alpha sign.

    Reverse setups are implemented symmetrically. Optional extra_expiries must
    refer to distinct, consistently matched expiries; primary is included once.
    """
    x = _components(equity, ssf, fv, cfg)
    g, c, f, d, zc, zf, zd = (x[k] for k in ("g", "c", "f", "d", "zc", "zf", "zd"))
    threshold, flat = cfg.move_threshold, cfg.flat_threshold
    side = np.sign(f)
    led = (zf.abs() > threshold) & (side * zd > threshold)
    breakout = _score(zf, led, [zf, zd])
    out = {
        "futures_breakout": breakout,
        "improving_discount": _score(zf, led & (side * g < 0), [zf, zd, g]),
        "cash_reversal": _score(-zc, (zc.abs() > threshold) & (zf.abs() <= flat), [zc, zf]),
    }

    # Previous and current windows do not overlap; prior scales are as of then.
    previous = {key: frame.shift(cfg.window) for key, frame in x.items()}
    ps = np.sign(previous["f"])
    prior_led = ((previous["zf"].abs() > threshold)
                 & (ps * previous["zd"] > threshold))
    required = [previous["zf"], previous["zd"], zf, zc, zd, g]
    catchup = (prior_led & (ps * zc > threshold) & (ps * zf >= -flat)
               & (ps * zd < -threshold) & (ps * g > 0))
    out["cash_catchup"] = _score(zc, catchup, required)

    # For a short: futures give back >= configured fraction of the prior rise,
    # cash stays above its pre-setup level and is not already falling strongly.
    rejected = (prior_led & (ps * zf < -threshold)
                & (-ps * f >= cfg.retracement * previous["f"].abs())
                & (ps * zc >= -flat) & (ps * (c + previous["c"]) > 0)
                & (ps * zd < -threshold))
    out["futures_rejection"] = _score(zf, rejected, required + [previous["c"]])

    # Long: preceding common rally, then both fall but cash falls further.
    pullback = ((previous["zf"].abs() > threshold)
                & (ps * previous["zc"] > threshold)
                & (ps * f < 0) & (ps * zc < -threshold) & (ps * zd > threshold))
    out["pullback_strength"] = _score(ps * zd.abs(), pullback,
                                      required + [previous["zc"]])

    out["maturity_agreement"] = g * np.nan
    if extra_expiries:
        legs = [breakout]
        for other_f, other_fv in extra_expiries.values():
            other = _components(equity, other_f, other_fv, cfg)
            condition = ((other["zf"].abs() > threshold)
                         & (np.sign(other["f"]) * other["zd"] > threshold))
            legs.append(_score(other["zf"], condition, [other["zf"], other["zd"]]))
        n = len(legs)
        mean = sum(legs) / n  # Missing any configured expiry -> missing output.
        positive = sum(leg.gt(0).astype(float) for leg in legs) / n
        negative = sum(leg.lt(0).astype(float) for leg in legs) / n
        out["maturity_agreement"] = mean.where(
            (positive >= cfg.agreement) | (negative >= cfg.agreement), 0).where(mean.notna())
    return out


def learned_cash_adjustment(equity: pd.DataFrame, ssf: pd.DataFrame,
                            fv: pd.DataFrame, cfg: PriceConfig = PriceConfig()) -> pd.DataFrame:
    """Idea 8: pooled rolling ridge, predicting holding-period cash log returns.

    Features: [gap, window cash return, window adjusted futures return]. No gap
    change feature because it is algebraically redundant. Standardize on each
    training set; intercept unpenalized. Each observation date has equal total
    weight even if coverage varies. At decision t only labels matured by t-1
    enter fitting. NaN until enough valid training dates; no cross-market pooling
    is performed automatically. No tuning or performance claims here.
    """
    x = _components(equity, ssf, fv, cfg)
    features = np.stack([x[k].to_numpy() for k in ("g", "c", "f")], axis=-1)
    # Target is cash-only: do not invalidate its future path for missing SSF.
    cash = np.log(aligned(equity, equity, "equity"))
    h, delay = cfg.horizon, cfg.entry_delay
    target = cash.diff().rolling(h, min_periods=h).sum().shift(-(h + delay)).to_numpy()
    result = np.full(equity.shape, np.nan)
    for t in range(len(equity)):
        stop = t - h - delay  # Exclusive: s+h+delay must be <= t-1.
        start = max(0, stop - cfg.train_window)
        if stop <= start:
            continue
        train = features[start:stop]
        y = target[start:stop]
        ok = np.isfinite(train).all(axis=2) & np.isfinite(y)
        counts = ok.sum(axis=1)
        if (counts > 0).sum() < cfg.min_train_dates:
            continue
        weights = np.broadcast_to(1 / np.maximum(counts, 1)[:, None], ok.shape)[ok]
        weights /= weights.sum()
        X, y = train[ok], y[ok]
        mu = np.average(X, axis=0, weights=weights)
        sd = np.sqrt(np.average((X - mu)**2, axis=0, weights=weights))
        sd = np.where(sd > 1e-12, sd, 1.)
        design = np.column_stack([np.ones(len(X)), (X - mu) / sd])
        penalty = np.diag([0., cfg.ridge, cfg.ridge, cfg.ridge])
        beta = np.linalg.solve(design.T @ (weights[:, None] * design) + penalty,
                               design.T @ (weights * y))
        current = features[t]
        valid = np.isfinite(current).all(axis=1)
        result[t, valid] = beta[0] + ((current[valid] - mu) / sd) @ beta[1:]
    return pd.DataFrame(result, index=equity.index, columns=equity.columns)


if __name__ == "__main__":
    dates = pd.bdate_range("2024-01-01", periods=100)
    equity = pd.DataFrame({"discount_improves": 100., "constant_discount": 100.}, index=dates)
    fv = equity * 1.001
    ssf = fv * np.exp(-.02)
    ssf.iloc[-1, 0] = fv.iloc[-1, 0] * np.exp(-.01)
    scores = calculate_price_signals(equity, ssf, fv)
    print("Synthetic illustration only; positive = cash long, not measured profit.")
    print(pd.DataFrame({name: score.iloc[-1] for name, score in scores.items()}))

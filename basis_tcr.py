"""Transparent daily confidence scores for a cash-equity basis strategy.

All inputs are aligned DataFrames: rows = trading sessions, columns = stocks.
Run separately for different trading calendars. Missing observations stay NaN.
Inputs must contain only information available at the decision timestamp.

basis: cleaned, carry/maturity-adjusted futures richness (higher = richer).
cash_ret: one-session cash residual log return, ending at the row's date.
futures_ret: optional, roll-clean futures residual log return, using the same
    timestamps and factor adjustment as cash_ret.
B: optional existing strategy score in [-1, 1]. Otherwise rank basis each date.
confirmation: optional mapping of signed features; positive MUST mean bullish.
    E.g. futures_flow (buy minus sell notional / total notional), options,
    eps_revision, institutional_flow. These are features, not raw trade feeds.
oi_growth: optional as-of growth in aggregate, roll-adjusted open interest.
    Modifies 'futures_flow' only; never determines direction on its own.

R uses rolling partial correlation, controlling for prior cash momentum.
With entry_delay=1, a signal dated s is evaluated against the H-session return
from close(s+1) to close(s+1+H). The fit is additionally lagged one session.
This is a next-close target, NOT a next-open target.

All lookbacks, signs, thresholds and shrinkage are research starting choices,
not fitted or established alpha relationships. No costs or portfolio sizing.
Dependencies: numpy, pandas.
"""
from __future__ import annotations

from collections.abc import Mapping
import numpy as np
import pandas as pd


def _squash_cs(x: pd.DataFrame) -> pd.DataFrame:
    """Same-date RMS scaling; preserves economic zero and bullish/bearish sign."""
    # Divide by row max first to avoid overflow when squaring large inputs.
    largest = x.abs().max(axis=1).replace(0.0, np.nan)
    relative = x.div(largest, axis=0)
    rms = relative.pow(2).mean(axis=1).pow(0.5)
    out = np.tanh(relative.div(2.0 * rms, axis=0))
    return out.mask(x.eq(0.0), 0.0).where(x.notna())


def _mean_available(parts: list[pd.DataFrame], like: pd.DataFrame) -> pd.DataFrame:
    """Equal weights over observed components; no observations means neutral."""
    total = pd.DataFrame(0.0, index=like.index, columns=like.columns)
    count = total.copy()
    for part in parts:
        total += part.fillna(0.0)
        count += part.notna().astype(float)
    return total.div(count.replace(0.0, np.nan)).fillna(0.0).clip(-1.0, 1.0)


def _event_history(direction, event, cash_ret):
    """Age and cash return since last widening shock in the current direction."""
    d, e, r = (v.to_numpy() for v in (direction, event, cash_ret))
    age = np.full(d.shape, np.nan)
    move = np.full(d.shape, np.nan)
    last_age = np.full(d.shape[1], np.nan)
    last_move = last_age.copy()
    previous = last_age.copy()
    for t in range(len(d)):
        active = np.isfinite(d[t]) & (d[t] != 0)
        same_episode = active & (d[t] == previous)
        last_age = np.where(same_episode, last_age + 1.0, np.nan)
        last_move = np.where(same_episode, last_move + r[t], np.nan)
        new_event = e[t] & active
        last_age = np.where(new_event, 0.0, last_age)
        last_move = np.where(new_event, r[t], last_move)
        age[t], move[t] = last_age, last_move
        previous = d[t]
    axes = dict(index=direction.index, columns=direction.columns)
    return pd.DataFrame(age, **axes), pd.DataFrame(move, **axes)


def calculate_tcr(
    basis: pd.DataFrame,
    cash_ret: pd.DataFrame,
    *,
    B: pd.DataFrame | None = None,
    futures_ret: pd.DataFrame | None = None,
    confirmation: Mapping[str, pd.DataFrame] | None = None,
    oi_growth: pd.DataFrame | None = None,
    horizon: int = 5,
    entry_delay: int = 1,
    response_window: int = 504,
    min_response: int = 126,
    shrinkage: float = 126.0,
    response_scale: float = 0.10,
    scale_window: int = 60,
    innovation_span: int = 20,
    source_window: int = 3,
    momentum_window: int = 5,
    shock_threshold: float = 1.5,
    freshness_half_life: float = 5.0,
) -> dict[str, pd.DataFrame]:
    """Return B, T, C, R, confidence, signal and component diagnostics.

    Higher T/C/R = more support for the trade indicated by sign(B).
    Missing components are ignored within a group; an entirely missing group
    is zero. Missing basis or B leaves scores and signal NaN.
    """
    if not isinstance(basis, pd.DataFrame) or basis.empty:
        raise ValueError('basis must be a nonempty DataFrame.')
    if not isinstance(basis.index, pd.DatetimeIndex):
        raise ValueError('Use a DatetimeIndex of trading sessions.')
    if (not basis.index.is_unique or not basis.columns.is_unique
            or not basis.index.is_monotonic_increasing):
        raise ValueError('Dates must be sorted; dates and stock labels must be unique.')
    integer_settings = [horizon, response_window, min_response, scale_window,
                        innovation_span, source_window, momentum_window]
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 1
           for v in integer_settings):
        raise ValueError('Lookbacks and horizon must be positive integers.')
    if (not isinstance(entry_delay, int) or isinstance(entry_delay, bool)
            or entry_delay < 0 or min_response > response_window
            or min_response < 5 or scale_window < 5 or innovation_span < 2):
        raise ValueError('Invalid entry delay, minimum history, or window settings.')
    if (not np.isfinite([shrinkage, response_scale, shock_threshold,
                         freshness_half_life]).all()
            or shrinkage < 0 or min(response_scale, shock_threshold,
                                    freshness_half_life) <= 0):
        raise ValueError('Invalid shrinkage, score scale, or event parameters.')

    def checked(x, name):
        if (not isinstance(x, pd.DataFrame) or not x.index.equals(basis.index)
                or not x.columns.equals(basis.columns)):
            raise ValueError(f'{name}: index and columns must exactly match basis.')
        out = x.astype(float)
        if np.isinf(out.to_numpy()).any():
            raise ValueError(f'{name} contains infinite values.')
        return out

    basis, cash_ret = checked(basis, 'basis'), checked(cash_ret, 'cash_ret')
    confirmation = {k: checked(v, k) for k, v in (confirmation or {}).items()}
    if futures_ret is not None:
        futures_ret = checked(futures_ret, 'futures_ret')
    if oi_growth is not None:
        oi_growth = checked(oi_growth, 'oi_growth')
    if B is None:
        # Centered percentile ranks: tied/all-equal values give neutral scores.
        ranks = basis.rank(axis=1, method='average')
        B = 2.0 * ranks.sub(0.5).div(basis.count(axis=1), axis=0) - 1.0
    else:
        B = checked(B, 'B')
        if B.abs().gt(1.0 + 1e-12).any().any():
            raise ValueError('B must lie in [-1, 1].')
        B = B.clip(-1.0, 1.0)
    B = B.where(basis.notna())
    direction = np.sign(B)
    diagnostics = {}
    scale_min = max(3, scale_window // 2)

    # T: timing. All volatility/forecast references exclude the current session.
    forecast = basis.shift(1).ewm(span=innovation_span, adjust=False,
                                 min_periods=innovation_span).mean()
    basis_sd = basis.shift(1).rolling(scale_window, scale_min).std().replace(0, np.nan)
    t_innovation = np.tanh(direction * (basis - forecast) / (2.0 * basis_sd))
    change = basis.diff()
    change_sd = change.shift(1).rolling(scale_window, scale_min).std().replace(0, np.nan)
    shock = direction * change / change_sd > shock_threshold
    age, cash_since = _event_history(direction, shock, cash_ret)
    t_fresh = 2.0 * np.exp(-np.log(2.0) * age / freshness_half_life) - 1.0
    cash_sd = cash_ret.shift(1).rolling(scale_window, scale_min).std().replace(0, np.nan)
    # Only penalize catch-up already realized. Do NOT reward a stock sell-off
    # as a fresh long opportunity merely because its futures basis widened.
    catchup = (direction * cash_since).clip(lower=0.0)
    t_catchup = -np.tanh(catchup / (2.0 * cash_sd * np.sqrt(age + 1.0)))
    timing_parts = [t_innovation, t_fresh, t_catchup]
    diagnostics.update(T_innovation=t_innovation, T_freshness=t_fresh,
                       T_catchup=t_catchup, event_age=age)
    if futures_ret is not None:
        f = futures_ret.rolling(source_window).sum()
        c = cash_ret.rolling(source_window).sum()
        futures_driven = (direction * f).clip(lower=0.0)
        cash_driven = (-direction * c).clip(lower=0.0)
        denom = f.abs() + c.abs()
        source = (futures_driven - cash_driven) / denom.replace(0, np.nan)
        source = source.mask(denom.eq(0.0), 0.0)
        delta = direction * basis.diff(source_window)
        source = source.where(delta > 0, 0.0).where(
            delta.notna() & f.notna() & c.notna())
        timing_parts.append(source)
        diagnostics['T_source'] = source
    T = _mean_available(timing_parts, basis)

    # C: bullish evidence supports longs; bearish evidence supports shorts.
    confirmation_parts = []
    for name, feature in confirmation.items():
        score = _squash_cs(feature)
        if name == 'futures_flow' and oi_growth is not None:
            # OI growth boosts/reduces magnitude but never supplies direction.
            score = score * (1.0 + 0.5 * _squash_cs(oi_growth).fillna(0.0))
        agreement = (direction * score).clip(-1.0, 1.0)
        confirmation_parts.append(agreement)
        diagnostics[f'C_{name}'] = agreement
    C = _mean_available(confirmation_parts, basis)

    # R: pair an OLD signal with a NOW-REALIZED holding-period cash return.
    # At date u: x is B[u-H-entry_delay]; y covers returns u-H+1,...,u.
    x = B.shift(horizon + entry_delay)
    y = cash_ret.rolling(horizon).sum()
    z = cash_ret.rolling(momentum_window).sum().shift(horizon + entry_delay)
    valid = x.notna() & y.notna() & z.notna()
    x, y, z = (v.where(valid) for v in (x, y, z))
    cxy = x.rolling(response_window, min_response).corr(y).clip(-1, 1)
    cxz = x.rolling(response_window, min_response).corr(z).clip(-1, 1)
    cyz = y.rolling(response_window, min_response).corr(z).clip(-1, 1)
    denom = np.sqrt((1.0 - cxz**2) * (1.0 - cyz**2))
    rho = ((cxy - cxz * cyz) / denom.where(denom > 1e-8)).clip(-1, 1)
    n = valid.astype(float).rolling(response_window, min_periods=1).sum()
    # Count-based heuristic shrinkage, NOT an effective sample size or t-stat.
    shrunk = rho * n / (n + shrinkage)
    R = np.tanh(shrunk / response_scale).shift(1).fillna(0.0)
    diagnostics.update(R_partial_corr=rho.shift(1), R_nobs=n.shift(1))

    T, C, R = (v.where(B.notna()) for v in (T, C, R))
    confidence = (T + C + R) / 3.0
    signal = B * (1.0 + 0.5 * confidence)
    return dict(B=B, T=T, C=C, R=R, confidence=confidence, signal=signal,
                **diagnostics)

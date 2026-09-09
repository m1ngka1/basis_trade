"""Path examples and past-only fitting checks, not performance validation."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from price_signals import PriceConfig, calculate_price_signals, learned_cash_adjustment


CFG = PriceConfig(window=1, scale_window=20, min_scale=5, scale_floor=.001,
                  horizon=1, entry_delay=1, train_window=40, min_train_dates=10)


def path(cash_moves=(0., 0.), future_moves=(0., 0.), initial_gap=.02):
    dates = pd.bdate_range("2024-01-01", periods=80)
    s = np.full(80, np.log(100.))
    q = s.copy() + initial_gap
    for i, step in enumerate(cash_moves, start=78):
        s[i] = s[i-1] + step
    for i, step in enumerate(future_moves, start=78):
        q[i] = q[i-1] + step
    equity = pd.DataFrame({"A": np.exp(s)}, index=dates)
    ssf = pd.DataFrame({"A": np.exp(q)}, index=dates)
    return equity, ssf, equity.copy()


@pytest.mark.parametrize("name,cash,future,gap,sign", [
    ("futures_breakout", (0., 0.), (0., .01), 0., 1),
    ("improving_discount", (0., 0.), (0., .01), -.02, 1),
    ("cash_reversal", (0., -.01), (0., 0.), 0., 1),
    ("cash_catchup", (0., .005), (.01, 0.), .02, 1),
    ("futures_rejection", (.005, 0.), (.01, -.01), .02, -1),
    ("pullback_strength", (.01, -.01), (.015, -.002), .02, 1),
])
def test_expected_setups_and_reverse_symmetry(name, cash, future, gap, sign):
    prices = path(cash, future, gap)
    scores = calculate_price_signals(*prices, CFG)
    assert sign * scores[name].iloc[-1, 0] > 0
    reverse = calculate_price_signals(*(1 / x for x in prices), CFG)
    for key in scores:
        assert_frame_equal(scores[key], -reverse[key], atol=1e-10, rtol=1e-10)


def test_constant_discount_does_not_create_short():
    result = calculate_price_signals(*path(initial_gap=-.02), CFG)
    for name, score in result.items():
        if name != "maturity_agreement":
            assert score.iloc[-1, 0] == 0


def test_cash_fall_is_not_futures_breakout():
    result = calculate_price_signals(*path((0., -.01), (0., 0.)), CFG)
    assert result["futures_breakout"].iloc[-1, 0] == 0
    assert result["cash_reversal"].iloc[-1, 0] > 0


def test_futures_fall_closure_is_not_cash_catchup():
    result = calculate_price_signals(*path((0., 0.), (.01, -.005)), CFG)
    assert result["cash_catchup"].iloc[-1, 0] == 0


def test_maturity_agreement_and_missing_leg():
    equity, ssf, fv = path((0., 0.), (0., .01))
    r = calculate_price_signals(equity, ssf, fv, CFG,
                                extra_expiries={"next": (ssf * 1.01, fv * 1.01)})
    assert r["maturity_agreement"].iloc[-1, 0] > 0
    missing = ssf.copy()
    missing.iloc[-1] = np.nan
    r = calculate_price_signals(equity, ssf, fv, CFG,
                                extra_expiries={"next": (missing, fv)})
    assert np.isnan(r["maturity_agreement"].iloc[-1, 0])
    _, opposed, opposed_fv = path((0., 0.), (0., -.01))
    r = calculate_price_signals(equity, ssf, fv, CFG,
                                extra_expiries={"next": (opposed, opposed_fv)})
    assert r["maturity_agreement"].iloc[-1, 0] == 0


def test_prefix_invariance_and_bounds():
    equity, ssf, fv = path((.005, -.003), (.01, -.004))
    full = calculate_price_signals(equity, ssf, fv, CFG)
    short = calculate_price_signals(equity.iloc[:-1], ssf.iloc[:-1], fv.iloc[:-1], CFG)
    for name in full:
        assert_frame_equal(full[name].iloc[:-1], short[name])
        values = full[name].to_numpy()
        assert np.all(np.abs(values[np.isfinite(values)]) <= 1)


def test_missing_inside_window_and_bad_prices():
    equity, ssf, fv = path()
    ssf.iloc[-2] = np.nan
    r = calculate_price_signals(equity, ssf, fv, PriceConfig(window=3, min_scale=5))
    assert all(np.isnan(score.iloc[-1, 0]) for score in r.values())
    ssf.iloc[-2] = 0
    with pytest.raises(ValueError, match="positive"):
        calculate_price_signals(equity, ssf, fv, CFG)


def test_carry_only_change_does_not_create_price_signal():
    equity, _, _ = path(initial_gap=0.)
    fv = equity.copy()
    fv.iloc[-1] *= 1.05
    scores = calculate_price_signals(equity, fv.copy(), fv, CFG)
    for name, score in scores.items():
        if name != "maturity_agreement":
            assert score.iloc[-1, 0] == 0


def test_learned_model_recovers_known_relation_and_is_past_only():
    rng = np.random.default_rng(55)
    dates = pd.bdate_range("2023-01-01", periods=220)
    gap = pd.DataFrame(rng.normal(0, .01, (220, 4)), index=dates)
    # B at s predicts cash return ending s+2: next-close entry, one-row hold.
    cash_return = .1 * gap.shift(2).fillna(0)
    equity = 100 * np.exp(cash_return.cumsum())
    fv = equity.copy()
    ssf = fv * np.exp(gap)
    cfg = PriceConfig(window=1, horizon=1, entry_delay=1, train_window=60,
                      min_train_dates=20, ridge=.001)
    full = learned_cash_adjustment(equity, ssf, fv, cfg)
    short = learned_cash_adjustment(equity.iloc[:170], ssf.iloc[:170], fv.iloc[:170], cfg)
    assert_frame_equal(full.iloc[:170], short, atol=1e-12, rtol=1e-12)
    assert np.corrcoef(full.iloc[100:].to_numpy().ravel(),
                       (.1 * gap.iloc[100:]).to_numpy().ravel())[0, 1] > .99
    assert full.iloc[:20].isna().all().all()


@pytest.mark.parametrize("horizon,delay", [(1, 0), (1, 1), (3, 2)])
def test_ridge_agrees_with_manually_selected_matured_sample(horizon, delay):
    rng = np.random.default_rng(37)
    dates = pd.bdate_range("2024-01-01", periods=100)
    s = pd.DataFrame(rng.normal(0, .01, (100, 3)).cumsum(axis=0), index=dates)
    g = pd.DataFrame(rng.normal(0, .01, (100, 3)), index=dates)
    equity, fv, ssf = np.exp(s), np.exp(s), np.exp(s+g)
    cfg = PriceConfig(window=1, horizon=horizon, entry_delay=delay,
                      train_window=30, min_train_dates=10, ridge=.7)
    actual = learned_cash_adjustment(equity, ssf, fv, cfg)
    t = 70
    signal_dates = np.arange(t-horizon-delay-30, t-horizon-delay)
    X = np.stack([g.to_numpy(), s.diff().to_numpy(), (s+g).diff().to_numpy()], axis=-1)
    train = X[signal_dates].reshape(-1, 3)
    y = np.array([(s.iloc[u+delay+horizon] - s.iloc[u+delay]).to_numpy()
                  for u in signal_dates]).ravel()
    mu, sd = train.mean(axis=0), train.std(axis=0)
    design = np.column_stack([np.ones(len(train)), (train-mu)/sd])
    beta = np.linalg.solve(design.T @ design / len(y) + np.diag([0, .7, .7, .7]),
                           design.T @ y / len(y))
    expected = beta[0] + (X[t]-mu)/sd @ beta[1:]
    np.testing.assert_allclose(actual.iloc[t], expected, atol=1e-12)


@pytest.mark.parametrize("kwargs", [
    {"window": 0}, {"min_scale": 1}, {"agreement": .5},
    {"flat_threshold": 1}, {"ridge": 0}, {"entry_delay": -1},
])
def test_bad_config(kwargs):
    with pytest.raises(ValueError):
        PriceConfig(**kwargs)

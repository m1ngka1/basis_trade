"""Synthetic invariants for the research scaffold; no market-alpha claims."""

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from signal_research import (
    Config,
    compose_signals,
    fit_reliability_adapter,
    intraday_summary,
    matured_cash_labels,
    signed_score,
    trade_pressure,
)


def fixture_frames(n=100):
    index = pd.bdate_range("2025-01-01", periods=n)
    basis = pd.DataFrame({"short": -.02, "long": .02}, index=index)
    B = pd.DataFrame({"short": -.8, "long": .8}, index=index)
    eps = pd.DataFrame({"short": -.6, "long": .6}, index=index)
    return basis, B, eps


def test_chronic_discount_veto_and_new_information():
    basis, B, eps = fixture_frames()
    result = compose_signals(basis, B, {"eps": eps})
    assert result["basis_C"].iloc[-1, 0] < 0
    assert result["candidate"].iloc[-1, 0] == 0
    assert result["candidate"].iloc[-1, 1] > 0
    basis.iloc[-1, 0] -= .005
    new = compose_signals(basis, B, {"eps": eps})
    assert new["candidate"].iloc[-1, 0] < 0


def test_warmup_unknown_and_missing_basis_not_zero():
    basis, B, eps = fixture_frames()
    basis.iloc[-1, 0] = np.nan
    r = compose_signals(basis, B, {"eps": eps})
    assert np.isnan(r["candidate"].iloc[0, 0])
    assert np.isnan(r["candidate"].iloc[-1, 0])


def test_gate_off_matches_original_eps_only_formula():
    basis, B, eps = fixture_frames()
    r = compose_signals(basis, B, {"eps": eps}, Config(gate_mode="off"))
    assert_frame_equal(r["candidate"], B * (1 + np.sign(B) * eps / 6))


def test_prefix_invariance_for_all_outputs():
    basis, B, eps = fixture_frames(180)
    rng = np.random.default_rng(2026)
    basis += rng.normal(0, .001, basis.shape)
    complete = compose_signals(basis, B, {"eps": eps})
    short = compose_signals(basis.iloc[:130], B.iloc[:130], {"eps": eps.iloc[:130]})
    for name in complete:
        assert_frame_equal(complete[name].iloc[:130], short[name])


def test_family_weights_and_missing_do_not_redistribute():
    basis, B, eps = fixture_frames()
    cfg = Config(enabled=("eps", "cash_pressure", "participant_flow"), gate_mode="off")
    # Expectations family .6; cash-demand family (.6 + missing)/2 = .3.
    r = compose_signals(basis, B, {"eps": eps, "cash_pressure": eps}, cfg)
    np.testing.assert_allclose(r["C"], .45)
    np.testing.assert_allclose(r["coverage"], 2 / 3)


def test_missing_eps_neutral_not_negative():
    basis, B, _ = fixture_frames()
    r = compose_signals(basis, B, {}, Config(gate_mode="off"))
    assert_frame_equal(r["candidate"], B)
    assert r["coverage"].eq(0).all().all()


def test_explicit_eligibility_and_optional_gates():
    basis, B, eps = fixture_frames()
    eligible = basis * 0 + 1
    eligible.iloc[-1] = [0, np.nan]
    r = compose_signals(basis, B, {"eps": eps}, eligible=eligible)
    assert r["candidate"].iloc[-1].eq(0).all()
    with pytest.raises(ValueError, match="no adapter"):
        compose_signals(basis, B, {"eps": eps}, Config(use_reliability=True))
    r = compose_signals(basis, B, {"eps": eps},
                        Config(gate_mode="off", use_reliability=True),
                        reliability_gate=basis * np.nan)
    assert r["candidate"].eq(0).all().all()


def test_gate_never_flips_original_direction_or_increases_magnitude():
    basis, B, eps = fixture_frames()
    basis.iloc[-1] = [-.025, .025]
    r = compose_signals(basis, B, {"eps": eps}, Config(gate_mode="all"))
    candidate = r["candidate"].dropna()
    assert (candidate * B.loc[candidate.index]).ge(0).all().all()
    assert candidate.abs().le(r["basis_C"].loc[candidate.index].abs()).all().all()


def test_flat_basis_does_not_generate_residual_rank_positions():
    basis, B, eps = fixture_frames()
    r = compose_signals(basis, B, {"eps": eps}, Config(gate_mode="all"))
    assert r["candidate"].iloc[-1].eq(0).all()


def test_scale_preserves_absolute_strength():
    raw = pd.DataFrame([[1e-9, .9]])
    out = signed_score(raw, .5)
    assert 0 < out.iloc[0, 0] < out.iloc[0, 1] < 1


def test_pressure_strength_and_absent_trades():
    buy = pd.DataFrame([[1., 100., 0., np.nan]])
    sell = buy * 0
    expected = pd.DataFrame([[100.] * 4])
    out = trade_pressure(buy, sell, expected)
    assert out.iloc[0, 0] == .01
    assert out.iloc[0, 1] == 1
    assert np.isnan(out.iloc[0, 2]) and np.isnan(out.iloc[0, 3])


def test_intraday_cutoff_auction_exclusion_and_persistence():
    times = pd.date_range("2025-01-02 10:00", periods=4, freq="15min", tz="Asia/Hong_Kong")
    bars = pd.DataFrame({"timestamp": times, "stock": "A", "buy": [10., 0., 10000., 1e6],
                         "sell": [0., 5., 0., 0.], "continuous": [True, True, False, True]})
    r = intraday_summary(bars, times[0], times[2])
    assert r.loc["A", "imbalance"] == 1 / 3
    assert r.loc["A", "persistence"] == 0
    assert r.loc["A", "classified_notional"] == 15
    assert_frame_equal(r, intraday_summary(bars.iloc[:3], times[0], times[2]))


def test_labels_use_only_matured_outcomes_and_respect_gaps():
    basis, B, _ = fixture_frames()
    returns = basis * 0 + .001
    h, delay, t = 5, 1, 20
    labels = matured_cash_labels(B, returns, h, delay)
    assert_frame_equal(labels["B"], B.shift(h + delay + 1))
    np.testing.assert_allclose(labels["cash_label"].iloc[t], .005)
    # R training data at t ends at t-1; modifying t and later cannot affect it.
    returns.iloc[t:] = 1e3
    again = matured_cash_labels(B, returns, h, delay)
    assert_frame_equal(again["cash_label"].iloc[:t+1], labels["cash_label"].iloc[:t+1])
    returns.iloc[t-2, 0] = np.nan
    assert np.isnan(matured_cash_labels(B, returns, h, delay)["cash_label"].iloc[t, 0])


@pytest.mark.parametrize("kwargs", [
    {"history": 2}, {"basis_scale_floor": 0}, {"dead_zone": 3},
    {"enabled": ("borrow",)}, {"enabled": ("eps", "eps")},
    {"gate_mode": "unknown"}, {"c_strength": np.nan},
])
def test_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


def test_bad_axes_and_unbounded_scores_rejected():
    basis, B, eps = fixture_frames()
    with pytest.raises(ValueError, match="axes"):
        compose_signals(basis, B.iloc[::-1], {"eps": eps})
    with pytest.raises(ValueError, match="expected"):
        compose_signals(basis, B, {"eps": eps * 10})


def test_training_placeholder_is_explicit():
    with pytest.raises(NotImplementedError):
        fit_reliability_adapter()

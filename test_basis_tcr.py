"""Synthetic behavior checks; these do not establish real-market alpha."""
import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal
from basis_tcr import calculate_tcr

rng = np.random.default_rng(17)
dates = pd.bdate_range('2020-01-01', periods=850)
names = ['A', 'B', 'C', 'D', 'E', 'F']

def frame(values):
    return pd.DataFrame(values, index=dates, columns=names)

B = frame(rng.uniform(-0.95, 0.95, (len(dates), len(names))))
basis = frame(rng.normal(0, 0.001, B.shape).cumsum(axis=0))
ret = frame(rng.normal(0, 0.01, B.shape))
fret = ret + basis.diff()
flow = frame(rng.normal(0, 1, B.shape))
news = frame(rng.normal(0, .1, B.shape))
oi = frame(rng.normal(0, .02, B.shape))

kwargs = dict(B=B, futures_ret=fret,
              confirmation={'futures_flow': flow, 'eps_revision': news},
              oi_growth=oi)
result = calculate_tcr(basis, ret, **kwargs)
for k in ['B','T','C','R']:
    assert result[k].shape == B.shape
    assert result[k].abs().max().max() <= 1 + 1e-12, k
assert np.array_equal(np.sign(result['signal']), np.sign(B))
multiplier = result['signal'] / B
assert multiplier.min().min() >= .5 - 1e-12
assert multiplier.max().max() <= 1.5 + 1e-12

# Prefix invariance: more future data must not change any past output.
cut = 650
short = calculate_tcr(
    basis.iloc[:cut], ret.iloc[:cut], B=B.iloc[:cut], futures_ret=fret.iloc[:cut],
    confirmation={k:v.iloc[:cut] for k,v in kwargs['confirmation'].items()},
    oi_growth=oi.iloc[:cut])
for key in result:
    assert_frame_equal(result[key].iloc[:cut], short[key], atol=1e-12, rtol=1e-12)

# Final/negative direction confirmation is symmetric for longs and shorts.
side = pd.DataFrame(np.tile([1.,-1.,1.,-1.,1.,-1.], (len(dates), 1)),
                    index=dates, columns=names)
aligned = calculate_tcr(basis, ret, B=side, confirmation={'futures_flow':side})
assert (aligned['C'] > 0).all().all()
opposed = calculate_tcr(basis, ret, B=side, confirmation={'futures_flow':-side})
assert (opposed['C'] < 0).all().all()
# Increasing OI cannot flip a flow signal, and grows its magnitude here.
boosted = calculate_tcr(basis, ret, B=side,
                       confirmation={'futures_flow':side}, oi_growth=side.abs())
assert (boosted['C'] > aligned['C']).all().all()
# Absent confirmations remain neutral rather than manufacturing a signal.
assert calculate_tcr(basis, ret, B=B)['C'].eq(0).all().all()

# Known next-close predictive relationship, with H=1 and entry_delay=1.
for sign in [-1,1]:
    target = sign * .02 * B.shift(2) + .0001 * ret
    scored = calculate_tcr(basis, target, B=B, horizon=1)
    assert sign * scored['R'].iloc[-1].min() > .99 if sign == 1 else (
        scored['R'].iloc[-1].max() < -.99)

# R alignment agrees with direct residualization and a hand-selected sample.
h, delay, W = 5, 1, 40
scored = calculate_tcr(basis, ret, B=B, horizon=h, entry_delay=delay,
                       response_window=W, min_response=20)
t, j = 300, 0
# R[t] excludes session t: endpoints u=t-W,...,t-1.
us = np.arange(t-W, t)
xs = B.iloc[us-h-delay, j].to_numpy()
ys = np.array([ret.iloc[u-h+1:u+1, j].sum() for u in us])
zs = np.array([ret.iloc[u-h-delay-4:u-h-delay+1,j].sum() for u in us])
design = np.column_stack([np.ones(W), zs])
xr = xs - design @ np.linalg.lstsq(design, xs, rcond=None)[0]
yr = ys - design @ np.linalg.lstsq(design, ys, rcond=None)[0]
expected = np.corrcoef(xr, yr)[0,1]
assert np.isclose(scored['R_partial_corr'].iloc[t,j], expected, atol=1e-10)

# Missing data: maintain the full session grid, do not fill observations.
bad_basis, bad_ret = basis.copy(), ret.copy()
bad_basis.iloc[100,0] = np.nan
bad_ret.iloc[200,1] = np.nan
sparse = calculate_tcr(bad_basis, bad_ret, B=B)
assert np.isnan(sparse['signal'].iloc[100,0])
assert sparse['R_partial_corr'].replace([np.inf,-np.inf],np.nan).equals(
    sparse['R_partial_corr'])
# All-zero input and all-equal cross-sectional ranks are neutral.
zero = ret * 0
z = calculate_tcr(zero, zero, confirmation={'test':zero})
for key in ['B','T','C','R','signal']:
    assert z[key].eq(0).all().all(), key

try:
    calculate_tcr(basis, ret.iloc[::-1])
except ValueError:
    pass
else:
    raise AssertionError('Misaligned input should fail.')

print('PASS: bounds, direction, prefix invariance, confirmation symmetry, OI sign,')
print('      missing-group neutrality, positive/negative R, target alignment,')
print('      missing data, neutral zeros, and input validation.')
print('Synthetic last-date scores:')
print(pd.DataFrame({k: result[k].iloc[-1] for k in ['B','T','C','R','signal']}).round(3))

import numpy as np
import pandas as pd

from climxai.fao56 import et0, extraterrestrial_radiation
from climxai.xai import CAUSAL, PROXIES, physics_fn, shapley


def test_fao56_example_18_brussels():
    # FAO-56 Example 18: Brussels, 6 July, 50 deg 48' N, 100 m. Published ET0 = 3.9 mm/day.
    v = et0(21.5, 12.3, None, 2.078, 22.07, None, 50.8, 187, elev=100, ea=1.409)
    assert abs(v - 3.9) < 0.05


def test_fao56_example_8_ra():
    # FAO-56 Example 8: 20 deg S, 3 September -> Ra = 32.2 MJ m-2 day-1.
    assert abs(extraterrestrial_radiation(-20.0, 246) - 32.2) < 0.1


def test_et0_monotone_in_radiation_and_humidity():
    base = dict(tmax=31.0, tmin=23.0, u2=1.5, ps=99.0, lat=-7.7, doy=200)
    lo = et0(rh=80, rs=15, **base)
    assert et0(rh=80, rs=20, **base) > lo
    assert et0(rh=60, rs=15, **base) > lo


def test_noncausal_features_get_zero_shapley():
    rng = np.random.default_rng(0)
    n = 60
    X = pd.DataFrame({
        "tmax": rng.uniform(28, 34, n), "tmin": rng.uniform(20, 25, n),
        "rh": rng.uniform(60, 90, n), "u2": rng.uniform(0.5, 4, n),
        "rs": rng.uniform(10, 25, n), "ps": rng.uniform(95, 100, n),
        "lat": -7.7, "doy": rng.integers(1, 366, n),
        "tmean_obs": rng.uniform(24, 29, n), "qv": rng.uniform(15, 20, n),
        "precip": rng.uniform(0, 30, n), "cloud": rng.uniform(20, 100, n),
    })
    cols = CAUSAL + PROXIES
    phi = shapley(physics_fn(cols), X[cols].iloc[:10], X[cols].iloc[10:])
    assert np.abs(phi[:, len(CAUSAL):]).max() < 1e-6
    # Efficiency: attributions sum to f(x) - E[f(background)].
    f = physics_fn(cols)
    gap = f(X[cols].iloc[:10].to_numpy()) - f(X[cols].iloc[10:].to_numpy()).mean()
    assert np.allclose(phi.sum(axis=1), gap, atol=1e-5)

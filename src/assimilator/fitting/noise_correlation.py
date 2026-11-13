"""Fit the depth correlation of the model-error noise (online_errors) from a free run.

    1  gap(t,z) = free_run - obs, one value per day
    2  a = gap minus its 30-day centred mean (slow bias out)
    3  eta = a_t - phi a_{t-1}, phi the lag-1 autocorrelation per depth
    4  per season: correlation of eta between depths

Fit on the series the run assimilates (filtered + thinned).
"""
from datetime import date

import numpy as np
import pandas as pd

from .sigma import gap_frame
from ..sigma import SEASONS, season_of

MIN_DAYS = 30   # a depth needs this many eta values in every season


def _psd(S):
    w, V = np.linalg.eigh(S)
    return (V * np.clip(w, 0.0, None)) @ V.T


def innovations(gap):
    """Daily AR(1) innovations eta of the de-biased gap; returns (eta, phi)."""
    daily = gap.groupby(gap.index.floor("D")).first().asfreq("D")
    a = daily - daily.rolling(30, center=True, min_periods=15).mean()
    phi = a.apply(lambda c: c.autocorr(1))
    return a - phi * a.shift(1), phi


def fit_noise_correlation(obs, free_run, year=None, max_depth=None):
    """{"depths", "corr": {season: C}, "sd": {season: [...]}, "phi", ...} for online_errors."""
    gap, _ = gap_frame(obs, free_run, year, max_depth)
    eta, phi = innovations(gap)
    season = np.array([season_of(m) for m in eta.index.month])
    keep = [c for c in eta.columns if all(eta.loc[season == s, c].count() >= MIN_DAYS for s in SEASONS)]
    if len(keep) < 2:
        raise ValueError(f"fewer than 2 depths with {MIN_DAYS}+ days in both seasons")
    eta, phi = eta[keep], phi[keep]

    corr, sd = {}, {}
    for s in SEASONS:
        S = _psd(eta[season == s].cov(min_periods=MIN_DAYS).to_numpy())
        d = np.sqrt(np.clip(np.diag(S), 1e-12, None))
        C = np.clip(S / np.outer(d, d), -1.0, 1.0)
        np.fill_diagonal(C, 1.0)
        corr[s], sd[s] = np.round(C, 4).tolist(), np.round(d, 4).tolist()
    return {
        "depths": [float(z) for z in keep],
        "corr": corr,
        "sd": sd,
        "phi": [round(float(p), 4) for p in phi],
        "fitted_on": date.today().isoformat(),
        "estimator": "correlation of daily AR(1) innovations of (free_run - obs) minus its 30-day "
                     "centred mean, per season",
    }

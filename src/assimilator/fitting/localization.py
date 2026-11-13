"""Fit the per-depth localization radius from a free ensemble (perturbed forcing, no analysis).

    1  anomalies: each member minus the ensemble mean, stratified season only
    2  correlation between depths, pooled over members and times
    3  radius at a depth: the smallest separation where the correlation first drops below threshold

A free ensemble is a normal run with an observation file that has no rows (read_ensemble).
"""
from datetime import date

import numpy as np

from ..sigma import season_of

THRESHOLD = 0.45   # 2/sqrt(20): below it a 20-member correlation is sampling noise
MIN_STEPS = 100    # stratified output times needed


def anomaly_correlation(times, arr):
    """Depth-by-depth correlation of the ensemble anomalies in the stratified season."""
    sel = np.array([season_of(m) == "stratified" for m in times.month])
    if sel.sum() < MIN_STEPS:
        raise ValueError(f"only {sel.sum()} stratified output times (need {MIN_STEPS}): "
                         f"does the run cover May-October?")
    a = arr[:, sel, :]
    a = (a - a.mean(axis=0, keepdims=True)).reshape(-1, a.shape[2])
    with np.errstate(invalid="ignore", divide="ignore"):
        C = np.corrcoef(a, rowvar=False)
    return np.clip(np.nan_to_num(C, nan=0.0), -1.0, 1.0)


def radii(corr, depths, threshold=THRESHOLD):
    """Per depth, the separation of the nearest depth whose correlation is below threshold
    (the whole column if none is)."""
    out = np.empty(len(depths))
    for i, z in enumerate(depths):
        sep = np.abs(depths - z)
        order = np.argsort(sep, kind="stable")[1:]
        below = order[corr[i, order] < threshold]
        out[i] = sep[below[0]] if len(below) else sep.max()
    return out


def fit_localization(times, depths, arr, threshold=THRESHOLD):
    """{"depths_m", "radius_m", ...} for localization.radius_file."""
    L = radii(anomaly_correlation(times, arr), depths, threshold)
    return {
        "depths_m": [round(float(z), 3) for z in depths],
        "radius_m": [round(float(r), 3) for r in L],
        "threshold": threshold,
        "n_members": int(arr.shape[0]),
        "fitted_on": date.today().isoformat(),
        "estimator": "first separation where the free-ensemble anomaly correlation drops below "
                     "threshold, stratified season (May-October)",
    }

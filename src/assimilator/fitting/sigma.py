"""Fit sigma_obs from the free-run-minus-observation gap (ported from notebooks/generate_sigma.py;
the numbers are unchanged).

    1  gap(t,z) = free_run(t,z) - obs(t,z)   at the observed hours, split by season
    2  per (depth, season): std about THAT season's own mean
    3  median over depths  ->  {"mixed": .., "stratified": ..}, what a run config carries

FIT ON THE SERIES THE RUN ASSIMILATES (filtered + thinned), and refit whenever that series changes:
sigma is a property of the series, not of the lake.

AN UPPER BOUND: the gap carries model error as well as observation error.

ASSUMES the "linear" observation operator: the gap is taken against the model's output at the
observed depths, which Simstrat writes by interpolating the state.
"""
from datetime import date

import numpy as np
import pandas as pd

from ..functions import SEASONS, season_of, bin_hourly

MIN_HOURS = 24   # a depth must be present this often in EVERY season


def gap_frame(obs, free_run, year=None, max_depth=None):
    """STEP 1: free_run - obs, one column per observed depth, on the hours the two share.
    Each obs depth is compared with the nearest free-run depth; obs below the deepest free-run
    depth are dropped (nothing to compare with). Returns (gap, dropped_depths)."""
    obs = bin_hourly(obs)
    ref = free_run
    if year is not None:
        obs = obs[obs["time"].dt.year == year]
        ref = ref[ref.index.year == year]
    if max_depth is not None:
        obs = obs[obs["depth"] <= max_depth]
    if obs.empty:
        raise ValueError("no observations left after the depth/year selection")

    model_depths = np.asarray(ref.columns, float)
    wide = obs.pivot_table(index="time", columns="depth", values="value")
    wide = wide.loc[wide.index.isin(ref.index)]
    if wide.empty:
        raise ValueError("the observations and the free run share no hour")

    bed = model_depths.max()
    gap, far = {}, []
    for d in wide.columns:
        if d > bed:
            far.append(d)
            continue
        gap[d] = ref.iloc[:, int(np.argmin(np.abs(model_depths - d)))].reindex(wide.index) - wide[d]
    if not gap:
        raise ValueError("no observed depth matches the free run's output depths")
    return pd.DataFrame(gap).sort_index(axis=1), far


def sigma_by_depth(gap):
    """STEP 2: per (depth, season), the std of the gap about THAT season's own mean. A depth with
    fewer than MIN_HOURS hours in either season is dropped from both."""
    season = pd.Index([season_of(t.month) for t in gap.index], name="season")
    out = {}
    for s in SEASONS:
        sub = gap[season == s]
        out[s] = sub.std(ddof=0).where(sub.count() >= MIN_HOURS).round(4)
    return pd.DataFrame(out).dropna()


def scalars(by_depth):
    """STEP 3: the median over depths, one number per season."""
    return {s: round(float(np.median(by_depth[s].to_numpy(dtype=float))), 3) for s in SEASONS}


def sigma_by_depth_annual(gap):
    """STEP 2 without the season split: per depth, the std of the gap about its own mean over the
    whole record. For a lake whose seasonal pair does not behave (aegeri)."""
    return gap.std(ddof=0).where(gap.count() >= MIN_HOURS).round(4).dropna()


def fit_sigma(obs, free_run, year=None, max_depth=None, seasonal=True):
    """sigma_obs for the assimilated series `obs` (time, depth, value) against `free_run`.
    Returns the value a run config carries plus its provenance:

        {"sigma_obs": {"mixed": .., "stratified": ..}, "fitted_on": "YYYY-MM-DD",
         "n_depths": n, "by_depth": {"<depth>": {"mixed": .., "stratified": ..}}}

    `seasonal=False` fits one annual value instead ("sigma_obs" is then a number)."""
    gap, _ = gap_frame(obs, free_run, year, max_depth)
    if not seasonal:
        by_depth = sigma_by_depth_annual(gap)
        if by_depth.empty:
            raise ValueError(f"no depth has {MIN_HOURS}+ hours")
        return {
            "sigma_obs": round(float(np.median(by_depth.to_numpy(dtype=float))), 3),
            "fitted_on": date.today().isoformat(),
            "estimator": "std(free_run - obs) about its own mean, ddof=0, per depth; median over "
                         "depths. An upper bound (includes model error).",
            "n_depths": int(len(by_depth)),
            "by_depth": {f"{float(d):g}": float(v) for d, v in by_depth.items()},
        }
    by_depth = sigma_by_depth(gap)
    if by_depth.empty:
        raise ValueError(f"no depth has {MIN_HOURS}+ hours in both seasons")
    return {
        "sigma_obs": scalars(by_depth),
        "fitted_on": date.today().isoformat(),
        "estimator": "std(free_run - obs) about each season's own mean, ddof=0, per (depth, "
                     "season); median over depths. An upper bound (includes model error).",
        "n_depths": int(len(by_depth)),
        "by_depth": {f"{float(d):g}": {s: float(row[s]) for s in SEASONS}
                     for d, row in by_depth.iterrows()},
    }

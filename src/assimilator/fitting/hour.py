"""Pick the analysis hour (ported from feat/da-combi notebooks/generate_assimilation_series.py,
stage 1; the numbers are unchanged).

WHY THE HOUR MATTERS. Assimilating at a fixed hour aliases that hour's model-obs mismatch into a
persistent bias. Split the FREE RUN's surface error per hour into the mean over days (the model's
own diurnal flaw -- the same every day, so correcting it injects bias the physics undoes by
tomorrow) and the day-to-day scatter (real state error the analysis removes for good).

THE RULE is a gate then a tiebreak: keep the hours whose bias is small relative to the information
at that hour (|det| <= BIAS_FRAC * rnd), then take the most information among them. A ratio, not
an absolute degC, so it scales across lakes.

THE CANDIDATE WINDOW USUALLY DECIDES THE ANSWER. Hours are restricted to `window` (default 00-06)
for operational reasons: the analysis lands before the day it forecasts and every lake updates at a
comparable point of the diurnal cycle. The unconstrained pick is recorded too, so the cost of the
constraint stays visible. Only hours the lake genuinely samples are candidates.

The free run must not have assimilated anything, or the hours it was nudged at look best.
"""
import numpy as np
import pandas as pd

from ..functions import bin_hourly
from ..observations import FILTER_DEFAULTS

# Surface zone: where the diurnal cycle lives and the hour choice is decided. Upper bound is the
# filter's THERMO_DEPTH_MIN: above it the diurnal cycle survives into the assimilated series.
SURFACE_LO = 0.0
SURFACE_HI = float(FILTER_DEFAULTS["THERMO_DEPTH_MIN"])
MIN_DAYS   = 30       # days an hour must be sampled on before it is a candidate
STRATIFIED = (5, 10)  # inclusive month bounds
BIAS_FRAC  = 0.2      # the gate: |det| <= BIAS_FRAC * rnd
WINDOW     = (0, 6)   # default candidate hours (inclusive, wraps: (22, 6) = 22,23,0..6)
CRITERIA   = ("balanced", "min-bias", "annual")
DETREND_DAYS = 15.0


def free_run_error(obs, free_run, detrend_days=DETREND_DAYS):
    """Hourly free-run error e = free_run - obs on the shared (hour, depth) points, drift removed,
    with hour/month/season labels. `obs` is the RAW record (binned hourly here)."""
    obs = bin_hourly(obs)
    m = free_run.loc[:, free_run.columns.isin(obs["depth"].unique())]
    mod = m.stack().rename("T_ref").reset_index()
    mod.columns = ["time", "depth", "T_ref"]

    j = mod.merge(obs, on=["time", "depth"]).dropna(subset=["value", "T_ref"])
    j["e"] = j["T_ref"] - j["value"]
    j = j.sort_values(["depth", "time"])
    win = int(detrend_days * 24)
    j["e_dt"] = j["e"] - (j.groupby("depth")["e"]
                          .transform(lambda s: s.rolling(win, center=True,
                                                         min_periods=win // 4).mean()))
    j["hour"] = j["time"].dt.hour
    j["month"] = j["time"].dt.month
    j["day"] = j["time"].dt.floor("D")
    j["strat"] = j["month"].between(*STRATIFIED)
    # diurnal anomaly: each day's own mean removed, so only hour-of-day structure survives
    j["ediur"] = j["e"] - j.groupby(["depth", "day"])["e"].transform("mean")
    return j.dropna(subset=["e_dt"])


def in_window(h, window):
    """Inclusive, and wraps: (22, 6) means 22,23,0,...,6. None = every hour."""
    if window is None:
        return True
    lo, hi = window
    return lo <= h <= hi if lo <= hi else (h >= lo or h <= hi)


def hour_table(j):
    """Per hour: the deterministic and random parts of the surface error, and the sample count."""
    g = j[(j["depth"] > SURFACE_LO) & (j["depth"] <= SURFACE_HI)]
    if g.empty:
        raise ValueError(f"no observations in the {SURFACE_LO}-{SURFACE_HI} m surface zone")
    strat = g[g["strat"]]
    det_s = strat.groupby("hour")["ediur"].mean()
    rnd_s = strat.groupby("hour")["ediur"].std()
    n_s   = strat.groupby("hour")["day"].nunique()
    # annual: sum |monthly mean|, so a bias in one month cannot cancel another's
    monthly = g.groupby(["hour", "month"])["ediur"].mean().abs()
    ann     = monthly.groupby("hour").sum()
    n_all   = g.groupby("hour")["day"].nunique()
    return det_s, rnd_s, n_s, ann, n_all


def pick(det_s, rnd_s, n_s, ann, n_all, window=WINDOW):
    """Best hour under each criterion, over the well-sampled hours inside `window`."""
    ok_s   = [h for h in det_s.index if n_s.get(h, 0) >= MIN_DAYS and in_window(h, window)]
    ok_all = [h for h in ann.index if n_all.get(h, 0) >= MIN_DAYS and in_window(h, window)]
    if not ok_s or not ok_all:
        raise ValueError(f"no hour both sampled on >= {MIN_DAYS} days and inside window {window}")
    low_bias = [h for h in ok_s if abs(det_s[h]) <= BIAS_FRAC * rnd_s[h]]
    if not low_bias:
        low_bias = [min(ok_s, key=lambda h: abs(det_s[h]))]
    out = {
        "balanced": max(low_bias, key=lambda h: rnd_s[h]),
        "min-bias": min(ok_s,     key=lambda h: abs(det_s[h])),
        "annual":   min(ok_all,   key=lambda h: ann[h]),
    }
    return out, ok_s, low_bias


def select_hour(obs, free_run, window=WINDOW, criterion="balanced", detrend_days=DETREND_DAYS):
    """The analysis hour for one lake, with the evidence behind it.

    obs       RAW record (time, depth, value)
    free_run  wide free-run frame (see assimilator.fitting)
    Returns {"hour": h, "criterion": .., "by_criterion": .., "unconstrained": .., "by_hour": ..}."""
    if criterion not in CRITERIA:
        raise ValueError(f"unknown criterion {criterion!r}; choose from {CRITERIA}")
    j = free_run_error(obs, free_run, detrend_days)
    det_s, rnd_s, n_s, ann, n_all = hour_table(j)
    best, candidates, low_bias = pick(det_s, rnd_s, n_s, ann, n_all, window)
    free, _, _ = pick(det_s, rnd_s, n_s, ann, n_all, None)
    return {
        "hour": int(best[criterion]),
        "criterion": criterion,
        "by_criterion": {k: int(v) for k, v in best.items()},
        "unconstrained": {k: int(v) for k, v in free.items()},
        "window": list(window) if window else None,
        "sampled_hours": [int(h) for h in sorted(n_all.index)],
        "candidate_hours": [int(h) for h in candidates],
        "low_bias_hours": [int(h) for h in sorted(low_bias)],
        "surface_zone_m": [SURFACE_LO, SURFACE_HI],
        "min_days": MIN_DAYS,
        "bias_frac": BIAS_FRAC,
        "stratified_months": list(STRATIFIED),
        "by_hour": {
            str(int(h)): {
                "det_stratified": round(float(det_s[h]), 4),
                "rnd_stratified": round(float(rnd_s[h]), 4),
                "annual_abs_sum": round(float(ann.get(h, np.nan)), 4),
                "n_days": int(n_s.get(h, 0)),
            } for h in sorted(det_s.index)
        },
    }

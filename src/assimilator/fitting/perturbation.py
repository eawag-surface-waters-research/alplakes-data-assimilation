"""Fit the AR(1) forcing perturbation (the math of notebooks/generate_perturbation.py; the numbers
are unchanged). Getting the data -- the reanalysis lake mean and the control forcing -- is the
host's job; everything here takes frames.

The calibration is TWO things, fitted from different data and kept strictly separate:

  AMPLITUDE    how big the perturbation is: the stationary std sigma/sqrt(1-phi^2), from the
               reanalysis residual (reanalysis lake mean - control forcing).   fit_amplitude

  PERSISTENCE  how long it stays correlated, phi = exp(-1/tau). It cannot be measured (that needs
               archived forecasts), so it is BRACKETED:
      lower    the ACF of the reanalysis residual -- a floor (grid-mismatch flicker can only
               shorten a measured decorrelation time). Comes with fit_amplitude.
      upper    the ACF of the forcing signal itself -- a ceiling (weather persists longer than
               forecast error).                                                  fit_upper
      envelope (default) per channel, the longer tau: under-spread is the known EnKF failure.

TWO TRAPS: never fit AR(1) to raw GLOB (its ACF is the diurnal cycle) -- GLOB goes through a
clearness index and U/V are deseasonalised by (month, hour); and sigma is one hourly kick, not
the spread, so amplitude is stored once as the stationary std and each bound derives its sigma.

Only `variables` is read at run time (assimilator.perturbate); the rest is provenance.
"""
from datetime import datetime, timezone

import numpy as np
import pandas as pd

PERTURB_VARS = ["U", "V", "GLOB"]   # channels perturbed downstream
BOUNDS    = ("lower", "upper", "envelope")
MAXLAG    = 96      # h - long enough to see the e-folding of every channel
DAY_WM2   = 20.0    # W/m2 - above this counts as daytime for the clearness index
CLEAR_PCT = 95      # percentile of GLOB per (month, hour) used as the clear-sky proxy

CAVEAT = ("phi is bracketed, not measured. lower is a floor (the reanalysis residual carries white "
          "spatial mismatch, which shortens any measured tau); upper is a ceiling (the station "
          "signal measures weather persistence, not error persistence); envelope takes the longer "
          "tau per channel. The honest number needs the ACF of (forecast - station obs), which "
          "requires archived forecasts.")


# ------------------------------------------------------------------------------ persistence --

def acf(x, maxlag=MAXLAG):
    """Sample autocorrelation to `maxlag`. NaNs dropped (the clearness index is night-masked)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    x = x - x.mean()
    denom = float(np.dot(x, x))
    return np.array([np.dot(x[:len(x) - k], x[k:]) / denom for k in range(maxlag + 1)])


def efold_tau(a):
    """First crossing of 1/e, linearly interpolated (more robust than the lag-1 estimate)."""
    thresh = np.exp(-1.0)
    for k in range(1, len(a)):
        if a[k] < thresh:
            drop = max(a[k - 1] - a[k], 1e-12)
            return float(k - 1 + (a[k - 1] - thresh) / drop)
    return float(len(a))       # never decorrelates within maxlag


def _deseasonalise(s, month, hour):
    """Remove the mean (month, hour) cycle -- the diurnal signal that corrupts a raw fit."""
    return s - s.groupby([month, hour]).transform("mean")


def clearness_index(glob, month, hour):
    """GLOB / clear-sky, daytime only; NaN at night. Clear-sky = the CLEAR_PCT-th percentile of
    GLOB within each (month, hour) cell."""
    clear = glob.groupby([month, hour]).transform(lambda x: np.percentile(x, CLEAR_PCT))
    day   = (glob > DAY_WM2) & (clear > DAY_WM2)
    return pd.Series(np.where(day, glob / clear.replace(0, np.nan), np.nan), index=glob.index)


def persistence_series(ctrl, source="clearness"):
    """The series each channel's tau is fitted on. 'raw' is diagnostic only -- see trap 1."""
    m, h = ctrl["time"].dt.month, ctrl["time"].dt.hour
    if source == "raw":
        return {v: ctrl[v] for v in PERTURB_VARS}
    series = {v: _deseasonalise(ctrl[v], m, h) for v in ("U", "V")}
    series["GLOB"] = (clearness_index(ctrl["GLOB"], m, h) if source == "clearness"
                      else _deseasonalise(ctrl["GLOB"], m, h))
    return series


def tau_of(phi):
    return float(-1.0 / np.log(phi))


def stationary_std(v):
    """The perturbation's actual spread: Var(x) = sigma^2/(1-phi^2). See trap 2."""
    return float(v["sigma"] / np.sqrt(1.0 - v["phi"] ** 2))


def variables_from(stat, phis):
    """(stationary std, phi) -> the (phi, sigma) pair the runtime reads."""
    return {v: {"phi":   round(float(phis[v]), 6),
                "sigma": round(float(stat[v] * np.sqrt(1.0 - phis[v] ** 2)), 6)}
            for v in PERTURB_VARS}


def fit_upper(ctrl, source="clearness"):
    """The upper persistence bound: tau from the control forcing's own ACF. `ctrl` has 'time'
    (UTC) and U, V, GLOB (e.g. a model's read_forcing over the fit window)."""
    series = persistence_series(ctrl, source)
    taus   = {v: efold_tau(acf(series[v])) for v in PERTURB_VARS}
    return {"source": f"ACF of the station signal ({source}); a ceiling",
            "fitted_on": datetime.now(timezone.utc).date().isoformat(),
            "tau_h": {v: round(taus[v], 3) for v in PERTURB_VARS},
            "phi":   {v: round(float(np.exp(-1.0 / taus[v])), 6) for v in PERTURB_VARS}}


def choose(persistence, bound):
    """Which end each channel takes. 'envelope' picks the LONGER tau per channel."""
    if bound != "envelope":
        return {v: bound for v in PERTURB_VARS}
    lo, up = persistence["lower"]["tau_h"], persistence["upper"]["tau_h"]
    return {v: ("lower" if lo[v] >= up[v] else "upper") for v in PERTURB_VARS}


def select(base, bound="envelope"):
    """Point `variables` at one of the stored bounds, in place, and return `base`. phi is copied
    from the chosen bound; sigma is derived from the single stored stationary std -- so switching
    bound is exactly reversible."""
    if bound not in BOUNDS:
        raise ValueError(f"unknown bound {bound!r}; choose from {BOUNDS}")
    p = base["persistence"]
    missing = [b for b in (("lower", "upper") if bound == "envelope" else (bound,)) if b not in p]
    if missing:
        raise ValueError(f"no {missing} bound stored — fit it first "
                         f"(fit_amplitude for lower, fit_upper for upper)")
    chosen = choose(p, bound)
    p["bound"], p["chosen"] = bound, chosen
    base["variables"] = variables_from(base["amplitude"]["stationary_std"],
                                       {v: p[chosen[v]]["phi"][v] for v in PERTURB_VARS})
    return base


def migrate(base):
    """Give a pre-bounds calibration (only `variables`, fitted on the residual = the LOWER bound)
    the amplitude/persistence shape, in place. A copy, never a refit."""
    if "persistence" in base and "amplitude" in base:
        return base
    old = base["variables"]
    base["amplitude"] = {
        "source": base.get("source", "reanalysis lake mean - control forcing"),
        "stationary_std": {v: round(stationary_std(old[v]), 6) for v in PERTURB_VARS},
    }
    base["persistence"] = {
        "bound": "lower",
        "chosen": {v: "lower" for v in PERTURB_VARS},
        "lower": {"source": "ACF of the reanalysis residual; a floor",
                  "tau_h": {v: round(tau_of(old[v]["phi"]), 3) for v in PERTURB_VARS},
                  "phi":   {v: old[v]["phi"] for v in PERTURB_VARS}},
        "_note": CAVEAT,
    }
    base.pop("source", None)
    return base


# -------------------------------------------------------------------------------- amplitude --

def fit_ar1(residuals):
    """Moment (Yule-Walker-type) AR(1) fit: lag-1 correlation and the implied innovation std."""
    r     = pd.Series(residuals).dropna().values
    phi   = float(np.corrcoef(r[:-1], r[1:])[0, 1])
    sigma = float(r.std() * np.sqrt(max(1 - phi ** 2, 0)))
    return {"phi": round(phi, 6), "sigma": round(sigma, 6)}


def fit_amplitude(residual, fit_window=None, source="reanalysis lake mean - control forcing"):
    """AMPLITUDE and the LOWER persistence bound from the residual {U, V, GLOB: hourly Series}
    (reanalysis lake mean minus control forcing over the fit window). Returns a calibration with
    `variables` provisional -- call select() (after fit_upper, for the envelope)."""
    variables = {v: fit_ar1(residual[v]) for v in PERTURB_VARS}
    out = {
        "fitted_on": datetime.now(timezone.utc).date().isoformat(),
        "n_timesteps": int(min(pd.Series(residual[v]).notna().sum() for v in PERTURB_VARS)),
        "variables": variables,
        "amplitude": {
            "source": source,
            "stationary_std": {v: round(stationary_std(variables[v]), 6) for v in PERTURB_VARS},
        },
        "persistence": {
            "bound": "lower",
            "chosen": {v: "lower" for v in PERTURB_VARS},
            "lower": {"source": "ACF of the reanalysis residual; a floor",
                      "tau_h": {v: round(tau_of(variables[v]["phi"]), 3) for v in PERTURB_VARS},
                      "phi":   {v: variables[v]["phi"] for v in PERTURB_VARS}},
            "_note": CAVEAT,
        },
    }
    if fit_window:
        out["fit_window"] = dict(fit_window)
    return out


def ordered(base):
    """Stable key order, so a re-fit diffs only where numbers changed."""
    p = base["persistence"]
    base["persistence"] = {k: p[k] for k in ("bound", "chosen", "lower", "upper", "_note") if k in p}
    head = ("lake", "reanalysis_lake", "fit_window", "n_timesteps", "fitted_on",
            "variables", "amplitude", "persistence")
    return {**{k: base[k] for k in head if k in base},
            **{k: v for k, v in base.items() if k not in head}}

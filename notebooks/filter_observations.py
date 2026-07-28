"""Adaptive low-pass filter for lake temperature observations -> temperature_filtered.csv.

WHY. At a fixed depth in the thermocline, most of the hour-to-hour signal is not heat content
changing but a sharp vertical gradient being displaced by basin-scale internal seiches. A 1D model
cannot reproduce that: Simstrat's a_seiche parameterises seiche *energy*, it does not move the
thermocline up and down at a point. Measured on upperlugano 2025 (stratified season), the sub-24 h
observation variability is the SAME SIZE as the whole innovation -- 1.29 K vs 1.27 K at 11 m, i.e.
~105% of innovation variance -- and across 16 depths the high-frequency fraction predicts the
ensemble's coverage failure at r = -0.909. Coverage reaches target (0.87) only at 40 m, the one
depth where the innovation is not dominated by it. So this variance is representativeness error,
not model error, and not something ensemble spread should ever have been asked to cover.

WHAT. A causal, depth- and time-varying trailing box filter. The window is

    W(z,t) = W_grad(z,t) + W_floor(z,t)
    W_grad = W_MAX * |dT/dz|(z,t) / G_MAX      (gradient from a causal GRAD_SMOOTH_H trailing mean)
    W_floor = clip((z - z_tc(t)) / (DEEP_REF - z_tc(t)), 0, 1) * (W_DEEP - W_MIN)

smoothing hardest where the gradient is sharp (displacement noise is largest) and in the deep water
where the true signal is nearly constant. Depths above THERMO_DEPTH_MIN are left untouched: there
the fast signal is real solar heating. Causal (trailing only, no lookahead), so it is valid for
operational use.

NOT a fix for the surface diurnal problem. The model reproduces only ~60% of the observed daily
amplitude at 0.5-3 m; that is real, representable signal the model gets wrong, i.e. model error.
Filtering it away would hide a defect rather than remove noise -- hence the surface exclusion.

PARAMETERS. Ten knobs, resolved per lake in this precedence order (see load_params):

    1. --w-max / --w-deep / ... on the command line          (single-lake use only)
    2. a "filter" block in the run config, per lake
    3. filter/<lake>.json, fitted by notebooks/calibrate_filter.py
    4. the module DEFAULTS below

Deleting filter/<lake>.json is the kill switch: that lake falls back to the defaults.

G_MAX defaults to None, meaning "derive from this lake's own record" -- the 95th percentile of its
own smoothed gradient below THERMO_DEPTH_MIN. A weakly stratified lake should not be judged against
Lugano's gradients.

THERMO_DEPTH_MIN defaults to 4 m, a FIXED value, and setting it to None asks for the
diurnal-coherence derivation instead (see derive_thermo_depth_min). 4 m is used because it is
uniformly conservative: measured across the seven configured lakes the derivation lands between
2.0 and 3.0 m, so a 4 m cut always excludes MORE than the physical test asks, and the two errors
are not symmetric --

    too deep  -> the filter declines to smooth a band it could have smoothed. Costs 2-8% of the
                 removed variance, measured. Harmless.
    too shallow -> the filter smooths away real solar heating, which is model error, not
                 representativeness error. It would hide a known model defect (the column
                 reproduces only ~60% of the observed diurnal amplitude at 0.5-3 m).

Note that this knob is not only a surface mask: depths above it are also excluded from the G_MAX
percentile and the thermocline search, so changing it rescales the windows at EVERY depth. Moving
from the derived cut to a fixed 4 m shifts G_MAX by -3% to +20% across the lakes. Since the gradient
term is W_MAX * gradient / G_MAX, a 20% shift in G_MAX is equivalent to a 20% shift in W_MAX --
refit W_MAX/W_DEEP/DEEP_REF/THERMO_GRAD_MIN after changing it, or they no longer mean what they
were fitted to mean.

The derivation is kept and is still run by notebooks/calibrate_filter.py as a diagnostic, which
records it as `derived_thermo_depth_min` and warns when it comes out DEEPER than the adopted value
-- the one case where a fixed 4 m would be unsafe, i.e. a lake whose sunlit layer reaches past 4 m.
No configured lake does that today.

The remaining eight are the constants carried over from the earlier standalone version
(alplakes-da-master src/adaptive_filter_general.py), which were tuned by eye on
upperlugano/geneva/murten. W_MAX, W_DEEP, DEEP_REF and THERMO_GRAD_MIN are the ones
notebooks/calibrate_filter.py fits against a free-run reference; W_MIN, GRAD_SMOOTH_H, DT_FILT and
MAX_NAN_FRAC are structural or QC choices and stay fixed.

Usage:
    python notebooks/filter_observations.py args/run_enkf.json --lake upperlugano
    python notebooks/filter_observations.py args/run_enkf.json --lakes all --plot
    python notebooks/filter_observations.py args/run_enkf.json --lakes geneva,murten

Then assimilate it with:  python src/assimilate.py args/run_enkf.json --lake upperlugano --filtered
"""
import os
import sys
import json
import logging
import argparse

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
from assimilator.functions import ROOT, merge_lake_args, resolve_obs_path   # noqa: E402

logger = logging.getLogger(__name__)

DEFAULTS = {
    "W_MIN":            1.0,    # h      - minimum window (i.e. no smoothing)
    "W_MAX":           24.0,    # h      - window at peak thermocline gradient
    "W_DEEP":         504.0,    # h      - depth-floor window reached at DEEP_REF (3 weeks)
    "DEEP_REF":        40.0,    # m      - depth at which the floor reaches W_DEEP
    "G_MAX":           None,    # degC/m - gradient mapped to W_MAX; None = auto (95th pct)
    "GRAD_SMOOTH_H":   72.0,    # h      - trailing window for smoothing the gradient itself
    "THERMO_DEPTH_MIN":  4.0,   # m      - above this the gradient term is zero; None = derive
    "THERMO_GRAD_MIN":  0.1,    # degC/m - min peak gradient for the lake to count as stratified
    "DT_FILT":         60,      # min    - resolution the filter runs at
    "MAX_NAN_FRAC":    0.30,    # -      - drop a depth if it is emptier than this
}

# Knobs for the automatic THERMO_DEPTH_MIN derivation. Not per-lake parameters: they define the
# criterion itself, so they are deliberately not in DEFAULTS and not fittable.
DIURNAL_FRAC_MIN   = 0.15   # below this share of HF variance explained by the mean diurnal cycle,
                            # the fast signal at that depth is no longer solar
THERMO_DEPTH_FLOOR = 2.0    # m - never cut shallower than this, whatever the data says
THERMO_DEPTH_CAP   = 15.0   # m - nor deeper: past here we would be declining to filter the
                            # thermocline, which is the whole point of the filter
STRAT_MIN_STEPS_D  = 30     # days of stratified record below which the season mask is abandoned
MIN_SAMPLES_PER_DAY = 4     # below this the record carries no sub-daily variability to remove


def raw_obs_path(cfg, root=ROOT):
    """The RAW observation series for a lake: always observations/<lake>/temperature.csv.

    Deliberately NOT resolve_obs_path(cfg). The config's "obs_file" names the series the
    ASSIMILATION reads, and for most lakes that is already thinned to one value per depth per day
    (temperature_noon.csv). This filter removes SUB-DAILY variability, so it has to see the
    sub-daily record; pointed at a noon series it would smooth across days using windows expressed
    in hours, which is meaningless.

    The pipeline order is raw -> filter -> thin, never raw -> thin -> filter. That order is also the
    better one: the causal window evaluated at noon averages the preceding W hours of real
    observations, which is exactly the quantity wanted.
    """
    return os.path.join(root, "observations", cfg["lake"], "temperature.csv")


# --------------------------------------------------------------------------------------- params

def filter_params_path(lake, root=ROOT):
    return os.path.join(root, "filter", f"{lake}.json")


def load_params(lake, cfg=None, root=ROOT):
    """Filter knobs for one lake, in the precedence order documented in the module docstring.

    Returns (params, source_description). The CLI layer is applied by main() on top of this.
    """
    cfg    = cfg or {}
    params = dict(DEFAULTS)
    srcs   = []

    path = filter_params_path(lake, root)
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            fitted = json.load(f)
        block = fitted.get("params", {})
        unknown = set(block) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"{os.path.relpath(path, root)}: unknown filter parameter(s) "
                             f"{sorted(unknown)}; expected a subset of {sorted(DEFAULTS)}")
        params.update(block)
        srcs.append(f"filter/{lake}.json (fitted {fitted.get('fitted_on', '?')})")

    block = cfg.get("filter", {})
    if block:
        unknown = set(block) - set(DEFAULTS)
        if unknown:
            raise ValueError(f'run config "filter" block: unknown parameter(s) {sorted(unknown)}; '
                             f"expected a subset of {sorted(DEFAULTS)}")
        params.update(block)
        srcs.append(f"run config \"filter\" block ({'/'.join(sorted(block))})")

    return params, "; ".join(srcs) if srcs else "built-in defaults (hand-tuned, not lake-specific)"


def describe(p):
    def fmt(k):
        v = p[k]
        return "auto" if v is None else (f"{v:g}")
    return (f"W_MAX={fmt('W_MAX')} h  W_DEEP={fmt('W_DEEP')} h  DEEP_REF={fmt('DEEP_REF')} m  "
            f"GRAD_MIN={fmt('THERMO_GRAD_MIN')} degC/m  THERMO_DEPTH_MIN={fmt('THERMO_DEPTH_MIN')} m  "
            f"G_MAX={fmt('G_MAX')}")


# ---------------------------------------------------------------------------------- the filter

def _grad_raw(piv, depths):
    """|dT/dz| by plain central difference, no depth masking. Used by the stratification test and
    by the diurnal-coherence derivation, both of which run BEFORE THERMO_DEPTH_MIN is known."""
    T = piv.values
    G = np.full_like(T, np.nan)
    for i in range(len(depths)):
        lo = i - 1 if i > 0 else None
        hi = i + 1 if i + 1 < len(depths) else None
        if lo is not None and hi is not None:
            G[:, i] = np.abs(T[:, hi] - T[:, lo]) / (depths[hi] - depths[lo])
        elif hi is not None:
            G[:, i] = np.abs(T[:, hi] - T[:, i]) / (depths[hi] - depths[i])
        elif lo is not None:
            G[:, i] = np.abs(T[:, i] - T[:, lo]) / (depths[i] - depths[lo])
    return pd.DataFrame(G, index=piv.index, columns=piv.columns)


def stratified_mask(piv, depths, p):
    """Timesteps where the profile shows a thermocline at all, by the same THERMO_GRAD_MIN the
    depth floor uses. Falls back to the whole record if the lake is barely ever stratified."""
    peak = pd.Series(np.nanmax(_grad_raw(piv, depths).values, axis=1), index=piv.index)
    mask = peak >= p["THERMO_GRAD_MIN"]
    days = mask.sum() * p["DT_FILT"] / 60 / 24
    if days < STRAT_MIN_STEPS_D:
        logger.warning(f"  only {days:.0f} d stratified (< {STRAT_MIN_STEPS_D} d) — "
                       f"using the whole record for the diurnal test")
        return pd.Series(True, index=piv.index)
    return mask


def diurnal_coherence(piv, p):
    """Per depth: the share of sub-daily variance explained by the MEAN diurnal cycle.

    HF(z,t) = T - causal 24 h trailing mean, the definition used throughout the findings. Composite
    HF by hour of day and take var(composite)/var(HF). Solar heating is phase-locked to the sun, so
    it survives the compositing; internal-seiche displacement is not phase-locked, so it averages
    away. The ratio therefore separates "fast because the sun is heating it" from "fast because a
    gradient is sliding past the sensor" -- which is exactly the distinction THERMO_DEPTH_MIN exists
    to draw, and which neither the HF magnitude nor HF/obs can make (HF/obs is near 0.9 in the deep,
    where the fast signal is neither solar nor large).

    Composited PER MONTH, then the median across months. A single composite over the whole season
    would understate the solar share: the daily amplitude varies by a factor of several between
    June and October, so averaging all days together shrinks var(composite) without shrinking
    var(HF). The median across months also stops one anomalous month deciding the cut.
    """
    n24 = max(1, int(round(24 * 60 / p["DT_FILT"])))
    hf  = piv - piv.rolling(n24, min_periods=n24 // 2).mean()
    out = {}
    for d in piv.columns:
        s = hf[d].dropna()
        shares = []
        for _, m in s.groupby([s.index.year, s.index.month]):
            if len(m) < n24 * 10 or not np.isfinite(m.var()) or m.var() <= 0:
                continue
            comp = m.groupby(m.index.hour).transform("mean")
            shares.append(float(comp.var() / m.var()))
        out[float(d)] = float(np.median(shares)) if shares else np.nan
    return pd.Series(out).sort_index()


def derive_thermo_depth_min(piv, depths, p):
    """The surface exclusion, from the observations alone. Returns (depth_m, coherence_series).

    Deliberately NOT a calibrated parameter. A correlation-based score would happily push this
    shallower, because smoothing 0.5-3 m raises corr(model, obs) -- the model reproduces only ~60%
    of the observed diurnal amplitude there, so filtering the obs hides that defect rather than
    removing noise. The cut is therefore derived from a physical test and clamped, never scored.
    """
    strat = stratified_mask(piv, depths, p)
    coh   = diurnal_coherence(piv[strat], p)

    # Compositing 24 hourly bins from a finite, autocorrelated month explains some variance by
    # chance, and how much depends on the record length -- the deep background runs ~0.03 on the
    # multi-year lakes and ~0.08 on the one-year ones. Subtract each lake's own background, read off
    # its deepest third (where nothing is solar), so one threshold means the same thing everywhere.
    deep_third = coh.index[int(len(coh) * 2 / 3):]
    floor = float(np.nanmedian(coh[deep_third])) if len(deep_third) else 0.0
    floor = 0.0 if not np.isfinite(floor) else min(floor, 0.5)
    coh   = ((coh - floor) / (1.0 - floor)).clip(lower=0.0)
    logger.info(f"  diurnal-coherence background (deepest third) = {floor:.3f}, subtracted")

    cand = [d for d in coh.index if np.isfinite(coh[d]) and coh[d] < DIURNAL_FRAC_MIN]
    z    = min(cand) if cand else float(np.max(depths))
    if not cand:
        logger.warning(f"  every depth is diurnally coherent (>= {DIURNAL_FRAC_MIN}) — "
                       f"falling back to the deepest observation")

    hi = min(THERMO_DEPTH_CAP, float(np.max(depths)))
    z_clamped = float(np.clip(z, THERMO_DEPTH_FLOOR, hi))
    if z_clamped != z:
        logger.warning(f"  derived THERMO_DEPTH_MIN {z:g} m clamped to {z_clamped:g} m "
                       f"(bounds {THERMO_DEPTH_FLOOR:g}-{hi:g} m)")
    return z_clamped, coh


def _gradient_field(piv, depths, p):
    """|dT/dz| at each (time, depth) for the WINDOW, with the surface excluded.

    The lower neighbour is searched upward past any depth shallower than THERMO_DEPTH_MIN, so the
    surface layer's temperature never contaminates a thermocline gradient.
    """
    zmin = p["THERMO_DEPTH_MIN"]
    T = piv.values
    G = np.full_like(T, np.nan)
    for i in range(len(depths)):
        i_lo = next((j for j in range(i - 1, -1, -1) if depths[j] >= zmin), None)
        i_hi = i + 1 if i + 1 < len(depths) else None
        if i_lo is not None and i_hi is not None:
            G[:, i] = np.abs(T[:, i_hi] - T[:, i_lo]) / (depths[i_hi] - depths[i_lo])
        elif i_hi is not None:
            G[:, i] = np.abs(T[:, i_hi] - T[:, i]) / (depths[i_hi] - depths[i])
        elif i_lo is not None:
            G[:, i] = np.abs(T[:, i] - T[:, i_lo]) / (depths[i] - depths[i_lo])
    g = pd.DataFrame(G, index=piv.index, columns=piv.columns)
    g[[d for d in g.columns if d < zmin]] = 0.0
    return g


def _windows(piv, depths, p):
    """Per (time, depth) smoothing window in hours."""
    grad = _gradient_field(piv, depths, p)
    gs   = max(1, int(p["GRAD_SMOOTH_H"] * 60 / p["DT_FILT"]))
    grad = grad.rolling(gs, center=False, min_periods=gs // 2).mean().ffill().bfill().fillna(0.0)

    thermo_cols = [d for d in grad.columns if d >= p["THERMO_DEPTH_MIN"]]
    if not thermo_cols:
        raise ValueError(f"no observation depth at or below THERMO_DEPTH_MIN="
                         f"{p['THERMO_DEPTH_MIN']:g} m — nothing to filter")
    g_max = p["G_MAX"] if p["G_MAX"] is not None else float(np.nanpercentile(grad[thermo_cols].values, 95))
    logger.info(f"  G_MAX = {g_max:.3f} degC/m "
                f"({'auto, 95th pct' if p['G_MAX'] is None else 'configured'})")
    if g_max <= 0:
        raise ValueError("G_MAX <= 0 — the record shows no vertical gradient at all")
    W = grad / g_max * p["W_MAX"]

    # depth floor: below the (time-varying) thermocline, ramp up to W_DEEP at DEEP_REF
    z_tc   = grad[thermo_cols].idxmax(axis=1)
    active = grad[thermo_cols].max(axis=1) >= p["THERMO_GRAD_MIN"]
    floor  = np.zeros((len(W), len(depths)))
    span   = p["W_DEEP"] - p["W_MIN"]
    for i, (z, on) in enumerate(zip(z_tc.values, active.values)):
        if on:
            floor[i] = np.clip((depths - z) / max(p["DEEP_REF"] - z, 1.0), 0.0, 1.0) * span
    logger.info(f"  stratified (thermocline detected) in {active.mean() * 100:.0f}% of timesteps")
    return W + pd.DataFrame(floor, index=W.index, columns=W.columns), g_max


def _active_nan_frac(piv):
    """Per depth, the share of missing hours WHILE THAT SENSOR WAS DEPLOYED.

    Measured between each depth's first and last valid sample, not over the whole record. Sensors
    come and go: on the multi-year lakes most depths cover only part of the span, and judging them
    against the full record rejects them for being absent rather than for being sparse. Measured
    over the whole record, geneva keeps 7 of 61 depths and murten 5 of 35 -- yet the rejected ones
    report 25-100 times a day whenever they are in the water, which is exactly the sub-daily
    sampling this filter needs. What the cut is meant to catch is a sensor that is patchy while
    deployed, and that is what this measures.
    """
    out = {}
    for d in piv.columns:
        s = piv[d]
        first, last = s.first_valid_index(), s.last_valid_index()
        out[d] = 1.0 if first is None else float(s.loc[first:last].isna().mean())
    return pd.Series(out)


def _variable_box(x, widths):
    """Causal trailing mean whose length varies per sample. Cumulative sums -> O(n)."""
    xf = np.where(np.isnan(x), 0.0, x)
    ok = (~np.isnan(x)).astype(float)
    cs = np.concatenate([[0.0], np.cumsum(xf)])
    cv = np.concatenate([[0.0], np.cumsum(ok)])
    hi = np.arange(len(x)) + 1
    lo = np.maximum(0, np.arange(len(x)) - widths + 1)
    n  = cv[hi] - cv[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, (cs[hi] - cs[lo]) / n, np.nan)


def _truth_proxy(x, dt_min):
    """The observed signal with only the fast noise taken off: a fixed 24 h CENTRED mean.

    Fixed and centred on purpose. It is the yardstick every configuration in a calibration grid is
    measured against, so it must not itself depend on the configuration, and it must carry no delay
    of its own.
    """
    n24 = max(1, int(round(24 * 60 / dt_min)))
    return pd.Series(x).rolling(n24, center=True, min_periods=1).mean().values


def _lag_error(truth, widths):
    """How much temperature the causal window back-dates away, in degC. Model-free.

    A trailing box of width W has its centre of mass half a window back, so the filtered value at
    time t is an estimate of the truth at t - W/2 while the analysis uses it as the truth at t. The
    error it hands the analysis is therefore how far the lake actually moved over that half-window:

        lag(t) = | truth(t) - truth(t - W(t)/2) |

    estimated from the means of the two quarter-windows [t-W/4, t] and [t-W/2, t-W/4] rather than
    from two individual samples. Their centres are W/4 apart, so twice their difference estimates
    the change over W/2 exactly for any linear trend -- but each mean averages W/4 points, which
    divides the noise by sqrt(W/4). Differencing two raw samples instead leaves the deep bands
    measuring their own noise floor: at 40 m the water moves less over ten days than the sensor
    scatter, and the estimate saturates.

    Read off `truth` -- NOT off the filtered series. Measuring the filtered series' own movement
    would let heavier smoothing report less lag, since a 504 h box flattens deep water until it
    barely moves; that inverts the ranking it is supposed to produce.

    Differencing the causal filter against a centred filter of the same width fails the same way
    for a different reason: two boxes overlapping by half also differ by their residual noise, and
    that term shrinks with W, dominating on noise-dominated deep water.
    """
    n = len(truth)
    q = np.maximum(1, widths // 4)
    i = np.arange(n)
    a = _variable_box(truth, q)                     # mean over [t - W/4, t]
    b = a[np.maximum(0, i - q)]                     # mean over [t - W/2, t - W/4]
    with np.errstate(invalid="ignore"):
        d = 2.0 * np.abs(a - b)
    d[i < 2 * q] = np.nan                           # no history to look back over yet
    return float(np.nanmean(d)) if np.isfinite(d).any() else np.nan


def filter_obs(obs, p=None, lag_diag=False):
    """Filter a long-form obs frame (time, depth, value, ...).

    Returns (filtered_long, report, resolved_params). `resolved_params` has G_MAX and
    THERMO_DEPTH_MIN replaced by whatever was derived from this record, so a caller can record
    exactly what ran.

    `lag_diag=True` adds a `lag_degC` column to the report: how far the causal window back-dates
    the observation, in degC and without reference to any model (see _lag_error). Off by default
    because only the calibrator wants it.
    """
    p = dict(DEFAULTS) if p is None else {**DEFAULTS, **p}
    dt = p["DT_FILT"]

    span = max((obs["time"].max() - obs["time"].min()).days, 1)
    rate = len(obs) / max(obs["depth"].nunique(), 1) / span
    if rate < MIN_SAMPLES_PER_DAY:
        raise ValueError(
            f"the input has {rate:.2f} samples per depth per day, below {MIN_SAMPLES_PER_DAY} — "
            f"this looks like a daily or profile series, which carries no sub-daily variability "
            f"for the filter to remove. Point it at the raw high-frequency record "
            f"(observations/<lake>/temperature.csv), not at an already-thinned one")
    logger.info(f"  {rate:.0f} samples per depth per day over {span} d")

    # Bin to the analysis grid the SAME WAY functions.load_obs does -- round to the nearest bin,
    # not to the one below. resample() alone labels each bin by its left edge, so a sample at 10:45
    # would land in the filter's 10:00 bin but the assimilator's 11:00 bin: identical timestamps
    # carrying contents half a bin apart, i.e. a free 30 min of lag on top of the W/2 the causal
    # window already costs. Rounding first makes the two aggregations the same operation.
    obs = obs.copy()
    obs["time"] = (obs["time"] + pd.Timedelta(minutes=dt / 2)).dt.floor(f"{dt}min")

    piv = (obs.pivot_table(index="time", columns="depth", values="value", aggfunc="mean")
              .sort_index().resample(f"{dt}min").mean().interpolate(method="time", limit=2))

    nan_frac = _active_nan_frac(piv)
    dropped  = nan_frac[nan_frac > p["MAX_NAN_FRAC"]].index.tolist()
    if dropped:
        logger.warning(f"  dropping depths >{p['MAX_NAN_FRAC']:.0%} empty while deployed: {dropped}")
    piv    = piv[nan_frac[nan_frac <= p["MAX_NAN_FRAC"]].index]
    depths = np.array(piv.columns, dtype=float)
    if len(depths) < 2:
        raise ValueError(f"only {len(depths)} usable depth(s) after QC — need >= 2 for a gradient")
    logger.info(f"  {len(piv)} timesteps x {len(depths)} depths at {dt} min")

    if p["THERMO_DEPTH_MIN"] is None:
        z, coh = derive_thermo_depth_min(piv, depths, p)
        p = {**p, "THERMO_DEPTH_MIN": z}
        shown = ", ".join(f"{d:g}m {coh[d]:.2f}" for d in coh.index if np.isfinite(coh[d]))
        logger.info(f"  diurnal-coherence share by depth: {shown}")
        logger.info(f"  THERMO_DEPTH_MIN = {z:g} m (auto: shallowest depth below "
                    f"{DIURNAL_FRAC_MIN:g} coherence)")

    W, g_max = _windows(piv, depths, p)
    p = {**p, "G_MAX": g_max}

    out, report = pd.DataFrame(index=piv.index, columns=piv.columns, dtype=float), []
    for d in depths:
        widths = np.maximum(1, np.round(W[d].values * 60 / dt).astype(int))
        out[d] = _variable_box(piv[d].values, widths)
        raw, flt = piv[d], out[d]
        removed  = (raw - flt).std()
        row = {"depth": d, "median_window_h": float(np.median(widths) * dt / 60),
               "max_window_h": float(widths.max() * dt / 60),
               "raw_std": float(raw.std()), "filtered_std": float(flt.std()),
               "removed_std": float(removed),
               "removed_frac_of_var": float(removed ** 2 / raw.var()) if raw.var() > 0 else np.nan}
        if lag_diag:
            row["lag_degC"] = _lag_error(_truth_proxy(piv[d].values, dt), widths)
        report.append(row)

    long = (out.stack().rename("value").reset_index()
               .rename(columns={"level_1": "depth"}).dropna(subset=["value"]))
    return long, pd.DataFrame(report), p


# ------------------------------------------------------------------------------------ per lake

def _plot(lake, obs, filtered, out_path, p):
    """Before/after check. Window and depths are derived from the record, so this works on any lake."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    piv_raw = obs.pivot_table(index="time", columns="depth", values="value", aggfunc="mean").sort_index()
    piv_flt = filtered.pivot_table(index="time", columns="depth", values="value", aggfunc="mean").sort_index()

    # the most stratified fortnight of the record — where the filter is doing the most work
    depths = np.array(piv_raw.columns, dtype=float)
    peak   = pd.Series(np.nanmax(_grad_raw(piv_raw, depths).values, axis=1), index=piv_raw.index)
    centre = peak.rolling("15D").mean().idxmax()
    if pd.isna(centre):
        centre = piv_raw.index[len(piv_raw) // 2]
    win = slice(centre - pd.Timedelta(days=15), centre)

    avail = [d for d in piv_flt.columns]
    show  = [avail[i] for i in np.unique(np.linspace(0, len(avail) - 1, min(7, len(avail))).astype(int))]

    fig, axes = plt.subplots(len(show), 1, figsize=(13, 2.1 * len(show)), sharex=True)
    for ax, d in zip(np.atleast_1d(axes), show):
        ax.plot(piv_raw.loc[win, d], lw=0.7, alpha=0.55, label="raw")
        ax.plot(piv_flt.loc[win, d], lw=1.4, label="filtered")
        ax.set_ylabel(f"{d:g} m")
        ax.legend(loc="upper right", fontsize=7)
    fig.suptitle(f"{lake}: adaptive obs filter ({win.start:%Y-%m-%d}..{win.stop:%Y-%m-%d})  "
                 f"THERMO_DEPTH_MIN={p['THERMO_DEPTH_MIN']:g} m  W_MAX={p['W_MAX']:g} h  "
                 f"W_DEEP={p['W_DEEP']:g} h")
    fig.tight_layout()
    path = os.path.join(os.path.dirname(out_path), "filter_check.png")
    fig.savefig(path, dpi=110)
    plt.close(fig)
    logger.info(f"  wrote {os.path.relpath(path, ROOT)}")


def _write_thinned_companion(cfg, filtered, in_path):
    """Also emit the filtered series thinned to the schedule this lake actually assimilates.

    Most lakes assimilate observations/<lake>/temperature_noon.csv -- one value per depth per day --
    while the filter must run on the full sub-hourly record. Handing such a lake the hourly filtered
    series would change its analysis cadence, which is a different experiment rather than a filtered
    version of the same one. So the filtered series is thinned back to exactly the (time, depth)
    pairs the configured file already contains: rows are selected, never modified or moved, the same
    guarantee local/thin_obs.py gives.

    No-op when the lake already assimilates the raw series (upperlugano).
    """
    target = resolve_obs_path(cfg)
    if os.path.abspath(target) == os.path.abspath(in_path):
        return None
    if not os.path.isfile(target):
        logger.warning(f"  configured obs file {os.path.relpath(target, ROOT)} does not exist — "
                       f"no thinned companion written")
        return None

    sched = pd.read_csv(target, usecols=["time", "depth"])
    sched["time"] = pd.to_datetime(sched["time"], utc=True, format="ISO8601")
    keys = sched.drop_duplicates()

    out = keys.merge(filtered, on=["time", "depth"], how="inner")[list(filtered.columns)]
    cover = len(out) / len(keys) if len(keys) else 0.0
    if cover < 0.9:
        logger.warning(f"  only {cover:.0%} of the assimilated schedule has a filtered value "
                       f"({len(out):,}/{len(keys):,}) — check that the two files share a time grid")

    stem, ext = os.path.splitext(target)
    path = f"{stem}_filtered{ext}"
    out.to_csv(path, index=False)
    logger.info(f"  thinned to the assimilated schedule ({cover:.0%} covered): "
                f"{len(out):,} rows -> {os.path.relpath(path, ROOT)}")
    return path


def filter_lake(cfg, overrides=None, in_file=None, out_file=None, plot=False):
    """Filter one lake end to end. Returns the resolved parameters actually used."""
    lake = cfg["lake"]

    p, src = load_params(lake, cfg)
    if overrides:
        p.update(overrides)
        src += f"; {'/'.join(sorted(overrides))} overridden on the command line"
    logger.info(f"{lake}: params <- {src}")
    logger.info(f"  {describe(p)}")

    in_path = in_file or raw_obs_path(cfg)
    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"{lake}: no observations at {os.path.relpath(in_path, ROOT)}")
    out_path = out_file or os.path.join(os.path.dirname(in_path), "temperature_filtered.csv")
    logger.info(f"  {os.path.relpath(in_path, ROOT)} -> {os.path.relpath(out_path, ROOT)}")

    obs = pd.read_csv(in_path)
    obs["time"] = pd.to_datetime(obs["time"], utc=True, format="ISO8601")
    logger.info(f"  read {len(obs):,} rows, {obs['depth'].nunique()} depths")

    filtered, report, resolved = filter_obs(obs, p)

    # Carry the non-value columns (latitude/longitude/weight/station) through unchanged, matched on
    # depth, so the output is drop-in for assimilator.functions.load_obs.
    extra = [c for c in obs.columns if c not in ("time", "depth", "value")]
    if extra:
        per_depth = obs.groupby("depth")[extra].first().reset_index()
        filtered  = filtered.merge(per_depth, on="depth", how="left")
        filtered  = filtered[["time", "depth"] + extra + ["value"]]

    filtered.to_csv(out_path, index=False)
    logger.info(f"  wrote {len(filtered):,} rows -> {os.path.relpath(out_path, ROOT)}")

    if not out_file:
        _write_thinned_companion(cfg, filtered, in_path)

    pd.set_option("display.width", 160)
    print(f"\n{lake} — what was removed, per depth "
          f"(removed_frac_of_var = share of raw variance filtered out):")
    print(report.to_string(index=False, float_format=lambda v: f"{v:9.3f}"))

    if plot:
        _plot(lake, obs, filtered, out_path, resolved)
    return resolved


def main():
    ap = argparse.ArgumentParser(description="Adaptive low-pass filter for temperature observations")
    ap.add_argument("arg_file")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--lake", default=None, help="Lake to filter from the config's \"lakes\" block")
    grp.add_argument("--lakes", default=None,
                     help="Comma-separated lakes to filter in sequence, or 'all' for every block in "
                          "the config's \"lakes\". Lakes with no observation file are skipped; a "
                          "genuine failure is logged and the batch continues, exiting non-zero.")
    ap.add_argument("--in-file", default=None, help="override the input obs CSV (single lake only)")
    ap.add_argument("--out-file", default=None, help="override the output (single lake only)")
    ap.add_argument("--plot", action="store_true", help="write a before/after check figure")
    # Per-lake knobs. Prefer filter/<lake>.json (notebooks/calibrate_filter.py) — these are for
    # one-off experiments and are refused in batch mode, where one global set cannot be right.
    ap.add_argument("--w-max", type=float, default=None,
                    help=f"window at peak thermocline gradient, h (default {DEFAULTS['W_MAX']:g})")
    ap.add_argument("--w-deep", type=float, default=None,
                    help=f"depth-floor window reached at DEEP_REF, h (default {DEFAULTS['W_DEEP']:g})")
    ap.add_argument("--deep-ref", type=float, default=None,
                    help=f"depth at which the floor reaches W_DEEP, m (default {DEFAULTS['DEEP_REF']:g})")
    ap.add_argument("--grad-min", type=float, default=None,
                    help=f"min peak gradient to count as stratified (default {DEFAULTS['THERMO_GRAD_MIN']:g})")
    ap.add_argument("--thermo-depth-min", type=float, default=None,
                    help="surface exclusion depth in m, overriding the diurnal-coherence derivation")
    cli = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")

    overrides = {k: v for k, v in (("W_MAX", cli.w_max), ("W_DEEP", cli.w_deep),
                                   ("DEEP_REF", cli.deep_ref), ("THERMO_GRAD_MIN", cli.grad_min),
                                   ("THERMO_DEPTH_MIN", cli.thermo_depth_min)) if v is not None}

    arg_file = cli.arg_file if os.path.isfile(cli.arg_file) else os.path.join(ROOT, cli.arg_file)
    if not os.path.isfile(arg_file):
        raise ValueError(f"Args file not found: {cli.arg_file}")
    with open(arg_file) as f:
        raw = json.load(f)

    if not cli.lakes:
        lakes = [cli.lake]
    elif cli.lakes.strip() == "all":
        if not raw.get("lakes"):
            raise ValueError("--lakes all requires a \"lakes\" block in the config")
        lakes = list(raw["lakes"])
    else:
        lakes = [s.strip() for s in cli.lakes.split(",") if s.strip()]

    batch = len(lakes) > 1
    if batch:
        if overrides:
            raise ValueError("the per-lake --w-max/--w-deep/--deep-ref/--grad-min/--thermo-depth-min "
                             "flags apply one value to every lake and are refused in batch mode; "
                             "put them in filter/<lake>.json instead")
        if cli.in_file or cli.out_file:
            raise ValueError("--in-file/--out-file name a single file and are refused in batch mode")

    failed, skipped = [], []
    for lake in lakes:
        try:
            filter_lake(merge_lake_args(raw, lake=lake), overrides=overrides,
                        in_file=cli.in_file, out_file=cli.out_file, plot=cli.plot)
        except FileNotFoundError as exc:
            # the normal case for a config listing more lakes than you have buoys for
            if not batch:
                raise
            logger.info(f"{lake}: {exc} — skipped")
            skipped.append(lake)
        except Exception:                       # noqa: BLE001 - isolate each lake in a batch
            if not batch:
                raise
            logger.exception(f"lake '{lake}' FAILED - continuing with the rest of the batch")
            failed.append(lake)

    if batch:
        done = len(lakes) - len(failed) - len(skipped)
        logger.info(f"=== batch complete: {done}/{len(lakes)} filtered"
                    + (f"; {len(skipped)} without observations: {', '.join(skipped)}" if skipped else "")
                    + (f"; FAILED: {', '.join(failed)}" if failed else "") + " ===")
        if failed:
            sys.exit(1)


if __name__ == "__main__":
    main()

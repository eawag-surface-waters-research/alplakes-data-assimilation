"""Fit the observation filter for one lake -> filter/<lake>.json.

WHY. notebooks/filter_observations.py has ten knobs. Two derive themselves from the record (G_MAX,
THERMO_DEPTH_MIN) and four are structural or QC choices (W_MIN, GRAD_SMOOTH_H, DT_FILT,
MAX_NAN_FRAC). The remaining four -- W_MAX, W_DEEP, DEEP_REF, THERMO_GRAD_MIN -- were "tuned by eye
on upperlugano/geneva/murten" and are the ones this script fits, per lake, from data.

THE REFERENCE IS THE FREE RUN, not an EnKF ensemble mean. inputs/<lake>/ref/T_out.dat (written by
local/make_ref.py) is a plain Simstrat simulation over the window: warm-started, unperturbed,
unassimilated. Three reasons it is the right reference:

  * It is the 1D column itself. The filter's premise is "a 1D column cannot produce this
    variability", so the thing to measure against is the column, not an analysis whose behaviour
    also depends on K, R, inflation and localization.
  * It never saw an observation. An EnKF mean has been pulled toward the raw series -- including
    toward the seiche wiggle it cannot represent -- which inflates corr(model, raw) and therefore
    biases the score AGAINST filtering.
  * It exists for every lake with observations, at the cost of one Simstrat run. Requiring a
    20-member DA run per lake before you could tune the filter would make per-lake calibration
    impossible in practice.

THE CRITERION, and why not RMSE. Smoothing the obs always lowers RMSE(model, obs) -- a smoother
target is an easier one -- so RMSE cannot say whether the filter removed noise or signal. What
separates them is CORRELATION: if the filter strips only variability the 1D column cannot
represent, the model tracks what remains BETTER, so corr(ref, filtered) rises above corr(ref, raw).
Once the filter starts eating real signal, corr falls again. The optimum is the peak.

    score(depth) = corr(ref, filtered) - corr(ref, raw)      [>0 means the filter helped]

Free-run drift does not corrupt this. The free run drifts (order 1 degC over a year), but drift is
a slow trend, correlation is nearly blind to it, and the score is a DIFFERENCE of two correlations
against the same trajectory, so what is left largely cancels.

THE LAG PENALTY, and why it is model-free. Delta-corr alone picks the un-retuned defaults, because
`_variable_box` is a CAUSAL TRAILING mean: a W-hour window lags the signal by ~W/2, so W_DEEP=504 h
back-dates the deep observation by ~10 days. Correlation is almost blind to a lag on a slowly
varying deep signal; the assimilation is not -- a stale obs pulls the analysis toward where the lake
WAS. That is the signature the first filtered-obs run actually saw in the deep (bias +0.101 ->
+0.139 at 30 m, RMSE +18-20%).

The original penalty compared model-minus-obs biases, which cannot survive the switch to a free
reference: it would be measuring the free run's own drift, not the filter's. But back-dating is a
property of the filter and the observations alone, so it needs no model at all. A trailing box of
width W carries its centre of mass W/2 back, so the value it hands the analysis at time t is really
an estimate of the truth at t - W/2, and the error is how far the lake moved in between:

    lag(z) = mean | truth(z,t) - truth(z, t - W(z,t)/2) |

`truth` is the observations under a fixed 24 h CENTRED mean -- fixed so that every configuration is
judged against the same yardstick, centred so the yardstick carries no delay of its own. See
filter_observations._lag_error for why the two obvious alternatives (differencing against a centred
filter of the same width; measuring the filtered series' own movement) both invert the ranking.

SELECTION. Among configurations whose lag stays under the tolerance, take the highest delta-corr.
The tolerance is a POLICY choice, not a fitted quantity: the filter may inject at most `--lag-frac`
of the assumed observation error as a systematic offset. It is stated rather than derived, and it is
the one number here that remains a judgement call.

The default is 0.3, i.e. 0.15 degC against sigma_obs = 0.5. That is high, and it is high because
this is what the filter actually costs: measured on upperlugano across the whole grid, the lag in
the thermocline runs 0.13-0.25 degC, or 25-50% of the assumed observation error, and a tolerance of
0.1 admits nothing at all. The number the original criterion never surfaced is that the trailing box
buys its variance reduction with a real and sizeable bias. 0.3 admits 41 of 108 configurations on
upperlugano and is the point at which the constraint starts discriminating rather than vetoing.

BANDS. The score is read over two depth bands, defined from the lake's own thermocline rather than
hardcoded: with z_tc the median depth of peak gradient over the stratified season,

    interior = THERMO_DEPTH_MIN .. 2*z_tc      (the zone the defaults already get right)
    deep     = below 2*z_tc                    (the zone they are known to break)

On upperlugano z_tc ~ 11 m, so this reproduces the hand-chosen bands (interior 15-21 m, deep
25-40 m) that the original standalone version used -- which is the check that the generalisation is
not doing something different on the lake it was designed for.

WHAT IS NOT FITTED. THERMO_DEPTH_MIN is derived, never scored. A delta-corr criterion would happily
push it shallower, because smoothing 0.5-3 m raises corr(model, obs) -- the model reproduces only
~60% of the observed diurnal amplitude there, so filtering the obs would hide that defect rather
than remove noise. It is derived by diurnal coherence in filter_observations.py, and this script
only records the value that came out.

Usage:
    python notebooks/calibrate_filter.py args/run_enkf.json --lake upperlugano
    python notebooks/calibrate_filter.py args/run_enkf.json --lakes all --quick
    python notebooks/calibrate_filter.py args/run_enkf.json --lake geneva --dry-run

Prereq: inputs/<lake>/ref/T_out.dat must exist (local/make_ref.py, after local/spinup.py).
"""
import os
import sys
import json
import logging
import argparse
import itertools
from datetime import datetime, timezone

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "notebooks"))

from assimilator.functions import ROOT, merge_lake_args                       # noqa: E402
from assimilator.models.simstrat import SIMSTRAT_REF_YEAR                      # noqa: E402
import filter_observations as F                                               # noqa: E402

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------ seiche period
#
# W_MAX is the window applied where the gradient is sharpest, and what it exists to remove is the
# internal seiche displacing the thermocline past the sensor. A trailing box of width W has exact
# spectral nulls at W, W/2, W/3, ... -- the whole harmonic series -- so a window set to ONE seiche
# period annihilates the fundamental and every higher mode at the least lag that can do so (W/2).
# That makes the seiche period the physically correct value, not a knob, which is why W_MAX is
# taken from here instead of being grid-searched.
#
# It is read two ways and they check each other:
#
#   OBSERVED   Welch spectrum over gap-free 30 d segments of the stratified season, all depths at
#              or below THERMO_DEPTH_MIN, stacked. A seiche shows as a local peak ABOVE the red
#              background, so the background is removed with a rolling median before the peak is
#              taken -- a global maximum just returns the longest period resolved.
#   THEORY     T = 2L / sqrt(g' h_eff), the two-layer Merian period, from the basin length in
#              lake_bbox and the July-August profile. No fitting, no observations of the seiche.
#
# Measured across the seven lakes the two agree to 11-34% where a line exists at all (maggiore 72
# vs 80 h, upperlugano 24 vs 32 h, geneva 120 vs 90 h) -- three independent basins, which is what
# makes the mechanism credible. On the small lakes NO line is detectable: theory puts them at
# 8-13 h, close to the resolution floor and buried under the diurnal band.
#
# THE STANDARD FOR THOSE LAKES IS 24 h, AND IT IS A PLACEHOLDER, NOT A MEASUREMENT. It is the value
# the filter used for every lake before this analysis, so adopting it changes nothing for them and
# keeps the change confined to the lakes where there is evidence. It is very likely too LONG --
# theory says 2-3x too long on aegeri, hallwil and murten -- so it over-smooths, which costs lag.
# Replacing it with the theoretical period is the obvious next step and is deliberately not taken
# here, because a 3x cut to the window on three lakes should be a measured decision, not a
# side effect of this one.
SEICHE_SEG_H       = 720     # h  - Welch segment (30 d): resolves 4-240 h on a common grid
SEICHE_BAND_H      = (4.0, 240.0)
SEICHE_EXCESS_MIN  = 0.5     # ln - a local peak must stand this far above the red background
SEICHE_EDGE_H      = 12.0    # h  - ignore peaks within this of a band edge: the rolling-median
                             #      background is unreliable there and reports spurious excess
SEICHE_AGREE_MAX   = 2.0     # x  - observed/theory must agree within this factor to be adopted
W_MAX_SMALL_LAKE   = 24.0    # h  - the standard above, for lakes with no usable line

LAG_FRAC_DEFAULT = 0.30    # the filter may inject at most this share of sigma_obs as a lag bias
SIGMA_OBS_FALLBACK = 0.5   # if the run config does not set one
DEEP_BAND_FACTOR = 2.0     # deep = below this multiple of the thermocline depth
DEPTH_TOL_M      = 0.75    # obs-to-model-grid snapping tolerance (the ref grid is 1 m)
MIN_PAIRS        = 100     # per depth, below which a correlation is not worth reading

GRID = {
    "W_MAX":           [12.0, 24.0, 48.0],
    "W_DEEP":          [72.0, 168.0, 336.0, 504.0],
    "DEEP_REF_FACTOR": [1.0, 1.5, 2.0],        # x deepest observation
    "THERMO_GRAD_MIN": [0.1, 0.2, 0.3],
}
GRID_QUICK = {
    "W_MAX":           [24.0, 48.0],
    "W_DEEP":          [168.0, 504.0],
    "DEEP_REF_FACTOR": [1.0, 2.0],
    "THERMO_GRAD_MIN": [0.1, 0.3],
}


# ------------------------------------------------------------------------------------- loading

def ref_path(lake, root=ROOT):
    return os.path.join(root, "inputs", lake, "ref", "T_out.dat")


def load_ref(lake, obs_depths, root=ROOT):
    """Free-run trajectory at the observation depths, hourly, long form (time, depth, value).

    T_out.dat columns are NEGATIVE depths on the model's output grid (1 m for these lakes), so each
    observation depth is snapped to its nearest column and only those columns are read -- the file
    is ~30 MB and ~290 columns wide for the deeper lakes.
    """
    path = ref_path(lake, root)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{lake}: no free-run reference at {os.path.relpath(path, root)} "
                                f"— run local/make_ref.py for this lake first")

    with open(path, encoding="utf-8") as f:
        header = [c.strip().strip('"') for c in f.readline().strip().split(",")]
    grid = np.array([float(c) for c in header[1:]])          # negative, ascending toward 0

    keep, pairs, collided = {}, [], []
    for d in obs_depths:
        j = int(np.argmin(np.abs(grid + d)))                 # grid is -depth
        off = abs(grid[j] + d)
        if off > DEPTH_TOL_M:
            logger.warning(f"  obs depth {d:g} m has no model output within {DEPTH_TOL_M:g} m "
                           f"(nearest {-grid[j]:g} m) — dropped")
            continue
        col = header[1 + j]
        # Two observation depths can snap to the same model column when the output grid is coarser
        # than the sensor spacing (upperlugano: 0.5 m and 1 m both land on the model's 1 m cell).
        # Keep whichever is closer and say so -- letting the later one silently overwrite would drop
        # an observation depth from the score with no trace.
        if col in keep:
            prev_d, prev_off = keep[col]
            loser = d if off >= prev_off else prev_d
            collided.append(f"{loser:g}->{-grid[j]:g}")
            if off >= prev_off:
                continue
        keep[col] = (d, off)
        pairs.append((d, -grid[j], off))
    if collided:
        logger.warning(f"  {len(collided)} obs depth(s) share a model output cell with a closer "
                       f"neighbour and are excluded from the score: {', '.join(collided)}")
    keep = {c: v[0] for c, v in keep.items()}
    if not keep:
        raise ValueError(f"{lake}: no observation depth matches the reference output grid")

    snapped = [f"{d:g}->{m:g}" for d, m, off in pairs if off > 1e-9]
    if snapped:
        logger.info(f"  snapped {len(snapped)} obs depth(s) to the model grid: {', '.join(snapped)}")

    df = pd.read_csv(path, usecols=[header[0]] + list(keep))
    df.columns = [c.strip().strip('"') for c in df.columns]
    ref_t = pd.Timestamp(f"{SIMSTRAT_REF_YEAR}-01-01", tz="UTC")
    time = (ref_t + pd.to_timedelta(df[header[0]], unit="D")).dt.round("1h")
    df = df.drop(columns=[header[0]])
    df.index = time
    df = df[~df.index.duplicated(keep="first")]
    df.columns = [keep[c] for c in df.columns]
    return df.sort_index()


def load_raw_obs(cfg):
    # the RAW high-frequency series, not the config's obs_file -- see filter_observations.raw_obs_path
    path = F.raw_obs_path(cfg)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{cfg['lake']}: no observations at {os.path.relpath(path, ROOT)}")
    obs = pd.read_csv(path)
    obs["time"] = pd.to_datetime(obs["time"], utc=True, format="ISO8601")
    return obs[["time", "depth", "value"]]


# -------------------------------------------------------------------------------------- bands

def hourly_grid(obs, p):
    """The exact (time x depth) grid the filter runs on: resampled to DT_FILT and QC'd the same
    way. Everything downstream -- bands, the raw series in the score -- is read off this, so the
    raw and filtered series being compared sit on the same timestamps. Joining the score against
    the original sub-hourly observations instead would find almost no matching timestamps at all.
    """
    piv = (obs.pivot_table(index="time", columns="depth", values="value", aggfunc="mean")
              .sort_index().resample(f"{p['DT_FILT']}min").mean().interpolate(method="time", limit=2))
    nan_frac = piv.isna().mean()
    piv = piv[nan_frac[nan_frac <= p["MAX_NAN_FRAC"]].index]
    if piv.shape[1] < 2:
        raise ValueError(f"only {piv.shape[1]} depth(s) survive the {p['MAX_NAN_FRAC']:.0%} "
                         f"missing-data cut — the record is too sparse to calibrate on")
    long = piv.stack().rename("value").reset_index()
    long.columns = ["time", "depth", "value"]
    return piv, np.array(piv.columns, dtype=float), long


def _welch(x, seg):
    """Mean periodogram over complete gap-free segments. None if fewer than two.

    Fixed segment length so every depth and every lake lands on the SAME frequency grid -- an
    earlier attempt used each depth's whole series, whose differing lengths silently dropped all
    but a handful of depths from the stack (geneva contributed 1 of 52)."""
    segs, i, n, w = [], 0, len(x), np.hanning(seg)
    while i + seg <= n:
        s = x[i:i + seg]
        if np.isfinite(s).all():
            segs.append(np.abs(np.fft.rfft((s - s.mean()) * w)) ** 2)
            i += seg // 2
        else:
            i += seg // 4
    return np.mean(segs, axis=0) if len(segs) >= 2 else None


def observed_seiche_period(piv, depths, p, seg=SEICHE_SEG_H):
    """The internal-seiche period read off the observations, or None if no line stands out.

    Stacks the normalised Welch spectra of every depth at or below THERMO_DEPTH_MIN over the
    stratified season, divides out the red background (rolling median in log-log), and takes the
    largest remaining excursion. Peaks within SEICHE_EDGE_H of a band edge are refused: the median
    has too few neighbours there and manufactures excess, which is what made three small lakes
    first appear to peak at exactly the 240 h band edge.
    """
    # BLANK the unstratified rows rather than dropping them: dropping would splice January onto
    # July and put a discontinuity inside a segment. Blanked, _welch simply skips those segments.
    # stratified_mask is the repo's own gradient-based definition, so this adapts per lake instead
    # of assuming a May-Oct calendar.
    strat = piv.where(F.stratified_mask(piv, depths, p), np.nan)
    cols = [z for z in strat.columns if float(z) >= p["THERMO_DEPTH_MIN"]]
    if len(strat) < 2 * seg or not cols:
        return None, 0
    freq = np.fft.rfftfreq(seg, d=p["DT_FILT"] / 60.0)
    per = np.divide(1.0, freq, out=np.full_like(freq, np.inf), where=freq > 0)
    acc, used = None, 0
    for z in cols:
        P = _welch(strat[z].to_numpy(dtype=float), seg)
        if P is None:
            continue
        used += 1
        Pn = P / max(P[1:].sum(), 1e-30)
        acc = Pn if acc is None else acc + Pn
    if acc is None:
        return None, 0
    lo, hi = SEICHE_BAND_H
    m = (per >= lo) & (per <= hi)
    logp = np.log(acc[m])
    bg = pd.Series(logp).rolling(21, center=True, min_periods=5).median().to_numpy()
    excess = logp - bg
    pk = per[m]
    edge = (pk <= lo + SEICHE_EDGE_H) | (pk >= hi - SEICHE_EDGE_H)
    excess = np.where(edge, -np.inf, excess)
    i = int(np.argmax(excess))
    if not np.isfinite(excess[i]) or excess[i] < SEICHE_EXCESS_MIN:
        return None, used
    return float(pk[i]), used


def theoretical_seiche_period(cfg, piv, depths):
    """Two-layer Merian period T = 2L/sqrt(g' h_eff), in hours. Needs no seiche observation.

    L from the configured lake_bbox diagonal, the density contrast from the July-August profile.
    A coarse estimate -- the bbox is not the seiche axis and the two-layer idealisation is crude --
    but it is INDEPENDENT of the spectrum, which is the point: it is what makes an observed line
    credible rather than merely present.
    """
    bb = cfg.get("lake_bbox")
    if not bb or len(bb) != 4:
        return None, None
    lat = 0.5 * (bb[0] + bb[2])
    L = float(np.hypot((bb[2] - bb[0]) * 111e3,
                       (bb[3] - bb[1]) * 111e3 * np.cos(np.radians(lat))))
    summer = piv[piv.index.month.isin((7, 8))].mean()
    if not np.isfinite(summer).any():
        return None, L
    warm, cold = float(np.nanmax(summer)), float(np.nanmin(summer))
    rho = lambda t: 1000.0 * (1.0 - 6.63e-6 * (t - 4.0) ** 2)   # noqa: E731
    g_prime = 9.81 * (rho(cold) - rho(warm)) / rho(cold)
    if g_prime <= 0:
        return None, L
    h1 = 10.0                                     # nominal epilimnion
    h2 = max(float(np.max(depths)) - h1, 5.0)
    h_eff = h1 * h2 / (h1 + h2)
    return float(2.0 * L / np.sqrt(g_prime * h_eff) / 3600.0), L


def resolve_w_max(cfg, piv, depths, p):
    """W_MAX for this lake, and where it came from. See the SEICHE block at the top."""
    obs, n_used = observed_seiche_period(piv, depths, p)
    th, L = theoretical_seiche_period(cfg, piv, depths)
    info = {"observed_peak_h": obs, "two_layer_theory_h": th,
            "basin_length_km": None if L is None else round(L / 1e3, 1),
            "depths_in_spectrum": n_used}
    if obs is not None and th is not None:
        ratio = max(obs / th, th / obs)
        info["obs_over_theory"] = round(obs / th, 2)
        if ratio <= SEICHE_AGREE_MAX:
            info["basis"] = "observed seiche period (corroborated by two-layer theory)"
            return float(obs), info
        info["basis"] = (f"PLACEHOLDER {W_MAX_SMALL_LAKE:g} h — an observed line exists but "
                         f"disagrees with theory by {ratio:.1f}x, so neither is trusted")
        return W_MAX_SMALL_LAKE, info
    info["basis"] = (f"PLACEHOLDER {W_MAX_SMALL_LAKE:g} h — no seiche line detectable"
                     + ("" if th is None else f"; theory says {th:.0f} h, i.e. this is likely "
                                              f"{W_MAX_SMALL_LAKE / th:.1f}x too long"))
    return W_MAX_SMALL_LAKE, info


def depth_bands(piv, depths, p):
    """(interior, deep) observation depths, from this lake's own thermocline.

    z_tc is the median depth of peak |dT/dz| over the stratified season; the deep band starts at
    DEEP_BAND_FACTOR * z_tc. Depths above THERMO_DEPTH_MIN are in neither band -- the filter does
    not touch them, so they carry no information about these parameters.
    """
    grad  = F._grad_raw(piv, depths)
    strat = F.stratified_mask(piv, depths, p)
    z_tc  = float(np.nanmedian(grad[strat].idxmax(axis=1).values)) if strat.any() else float(np.median(depths))

    zmin  = p["THERMO_DEPTH_MIN"]
    split = DEEP_BAND_FACTOR * z_tc
    inter = [float(d) for d in depths if zmin <= d <= split]
    deep  = [float(d) for d in depths if d > split]
    logger.info(f"  z_tc = {z_tc:.1f} m -> interior {zmin:g}-{split:.1f} m ({len(inter)} depths), "
                f"deep > {split:.1f} m ({len(deep)} depths)")
    if not inter or not deep:
        logger.warning("  one of the two bands is empty — the lag penalty and the interior guard "
                       "will be read from whichever band has depths")
    return inter, deep, z_tc


# -------------------------------------------------------------------------------------- score

def score(ref_long, raw_long, filtered, report):
    """Per depth: (d_corr, corr_raw, corr_filtered, lag_degC, n).

    d_corr is the whole criterion; lag comes straight from the filter's own causal-vs-centred
    diagnostic and never touches the reference, so free-run drift cannot reach it.
    """
    lag = report.set_index("depth")["lag_degC"].to_dict()

    f = filtered.rename(columns={"value": "v_f"})
    f["time"] = pd.to_datetime(f["time"], utc=True)

    j = (ref_long.merge(raw_long.rename(columns={"value": "v_r"}), on=["time", "depth"])
                 .merge(f[["time", "depth", "v_f"]], on=["time", "depth"]).dropna())

    out = {}
    for d, g in j.groupby("depth"):
        if len(g) < MIN_PAIRS:
            continue
        c_r = float(g["v_m"].corr(g["v_r"]))
        c_f = float(g["v_m"].corr(g["v_f"]))
        out[float(d)] = (c_f - c_r, c_r, c_f, float(lag.get(float(d), np.nan)), int(len(g)))
    return out


def _band(s, depths, idx):
    vals = [s[d][idx] for d in depths if d in s and np.isfinite(s[d][idx])]
    return float(np.mean(vals)) if vals else np.nan


def run_grid(lake, ref_long, obs_raw, raw_long, base, combos, inter, deep):
    rows = []
    for w_max, w_deep, deep_ref, grad_min in combos:
        p = {**base, "W_MAX": w_max, "W_DEEP": w_deep, "DEEP_REF": deep_ref,
             "THERMO_GRAD_MIN": grad_min}
        filtered, report, _ = F.filter_obs(obs_raw, p, lag_diag=True)
        s = score(ref_long, raw_long, filtered, report)
        row = {"W_MAX": w_max, "W_DEEP": w_deep, "DEEP_REF": deep_ref, "THERMO_GRAD_MIN": grad_min,
               "d_corr_all":   _band(s, list(s), 0),
               "d_corr_inter": _band(s, inter, 0),
               "d_corr_deep":  _band(s, deep, 0),
               "lag_inter":    _band(s, inter, 3),
               "lag_deep":     _band(s, deep, 3),
               "per_depth": s}
        rows.append(row)
        logger.info(f"  W_MAX={w_max:5.0f} W_DEEP={w_deep:5.0f} DEEP_REF={deep_ref:5.1f} "
                    f"GRAD_MIN={grad_min:4.2f} | dcorr all {row['d_corr_all']:+.4f} "
                    f"int {row['d_corr_inter']:+.4f} deep {row['d_corr_deep']:+.4f} | "
                    f"lag int {row['lag_inter']:.4f} deep {row['lag_deep']:.4f} degC")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------------- per lake

COLS = ["W_MAX", "W_DEEP", "DEEP_REF", "THERMO_GRAD_MIN",
        "d_corr_all", "d_corr_inter", "d_corr_deep", "lag_inter", "lag_deep"]


def calibrate_lake(cfg, quick=False, lag_frac=LAG_FRAC_DEFAULT, dry_run=False,
                   thermo_depth_min=None, root=ROOT):
    lake = cfg["lake"]
    logger.info(f"=== {lake} ===")

    obs_raw = load_raw_obs(cfg)
    logger.info(f"  {len(obs_raw):,} obs rows, {obs_raw['depth'].nunique()} depths")

    # Start from whatever the lake resolves to today, so an existing filter/<lake>.json or config
    # block sets the parameters this grid is NOT searching (W_MIN, GRAD_SMOOTH_H, DT_FILT, ...).
    base, src = F.load_params(lake, cfg, root)
    logger.info(f"  starting parameters <- {src}")

    # Derive THERMO_DEPTH_MIN once, from the observations, and hold it fixed for the whole grid.
    # It is never scored -- see the module docstring.
    #
    # ALWAYS re-derived, even when filter/<lake>.json already carries a value. That value is an
    # OUTPUT of this script, written so the filter does not have to redo the derivation on every
    # run. Treating it as an input would mean a second calibration silently reuses the first one's
    # cut, so any change upstream -- the observation record growing, the hourly binning changing --
    # would never reach it. Only an explicit --thermo-depth-min pins it.
    piv, depths, raw_long = hourly_grid(obs_raw, base)

    # The derivation runs on every calibration whether or not its answer is adopted. It is the only
    # check on the fixed 4 m default: a lake whose sunlit layer reaches PAST the adopted cut is the
    # one case where that default smooths real solar heating, and nothing else would notice.
    derived, coh = F.derive_thermo_depth_min(piv, depths, {**base, "THERMO_DEPTH_MIN": None})
    shown = ", ".join(f"{d:g}m {coh[d]:.2f}" for d in coh.index if np.isfinite(coh[d]))
    logger.info(f"  diurnal-coherence share by depth: {shown}")

    if thermo_depth_min is not None:
        adopted, src = float(thermo_depth_min), "pinned on the command line"
    elif base["THERMO_DEPTH_MIN"] is None:
        adopted, src = derived, "derived from diurnal coherence"
    else:
        adopted, src = float(base["THERMO_DEPTH_MIN"]), "configured"
    logger.info(f"  THERMO_DEPTH_MIN = {adopted:g} m ({src}); diurnal coherence would give "
                f"{derived:g} m — not fitted either way")
    if derived > adopted + 1e-9:
        logger.warning(f"  the sunlit layer reaches {derived:g} m, BELOW the adopted cut of "
                       f"{adopted:g} m — the filter will smooth real solar heating between them, "
                       f"which is model error and must not be hidden. Raise THERMO_DEPTH_MIN.")
    base = {**base, "THERMO_DEPTH_MIN": adopted}

    ref = load_ref(lake, [float(d) for d in depths], root)
    logger.info(f"  reference {os.path.relpath(ref_path(lake, root), root)}: "
                f"{len(ref):,} hours x {ref.shape[1]} depths "
                f"({ref.index.min():%Y-%m-%d}..{ref.index.max():%Y-%m-%d})")
    ref_long = ref.stack().rename("v_m").reset_index()
    ref_long.columns = ["time", "depth", "v_m"]

    overlap = raw_long.merge(ref_long, on=["time", "depth"])
    if overlap.empty:
        raise ValueError(f"{lake}: the reference and the observations do not overlap in time "
                         f"(ref {ref.index.min():%Y-%m-%d}..{ref.index.max():%Y-%m-%d}, "
                         f"obs {raw_long['time'].min():%Y-%m-%d}..{raw_long['time'].max():%Y-%m-%d})")
    logger.info(f"  {len(overlap):,} coincident (time, depth) pairs to score on")

    inter, deep, z_tc = depth_bands(piv, depths, base)

    sigma_obs = float(cfg.get("sigma_obs", SIGMA_OBS_FALLBACK))
    lag_cap   = lag_frac * sigma_obs
    logger.info(f"  lag tolerance = {lag_frac:g} x sigma_obs({sigma_obs:g}) = {lag_cap:.3f} degC")

    # W_MAX is DETERMINED, not searched: it is the seiche period the thermocline term exists to
    # null. Searching it would let the grid trade a physical timescale against a correlation score.
    w_max, seiche = resolve_w_max(cfg, piv, depths, base)
    base = {**base, "W_MAX": w_max}
    logger.info(f"  W_MAX = {w_max:g} h <- {seiche['basis']}")
    logger.info(f"    observed {seiche['observed_peak_h']}  theory {seiche['two_layer_theory_h']}  "
                f"basin {seiche['basin_length_km']} km  "
                f"({seiche['depths_in_spectrum']} depths in the spectrum)")

    grid = GRID_QUICK if quick else GRID
    deep_refs = sorted({round(fac * float(np.max(depths)), 1) for fac in grid["DEEP_REF_FACTOR"]})
    combos = list(itertools.product([w_max], grid["W_DEEP"], deep_refs, grid["THERMO_GRAD_MIN"]))

    logger.info(f"  baseline (current parameters):")
    b = run_grid(lake, ref_long, obs_raw, raw_long, base,
                 [(base["W_MAX"], base["W_DEEP"], base["DEEP_REF"], base["THERMO_GRAD_MIN"])],
                 inter, deep)
    logger.info(f"  grid: {len(combos)} configurations "
                f"(W_MAX x W_DEEP x DEEP_REF{deep_refs} x GRAD_MIN)")
    df = run_grid(lake, ref_long, obs_raw, raw_long, base, combos, inter, deep)

    print(f"\n{lake} — ranked by correlation alone (the criterion that fails):")
    print(df.sort_values("d_corr_all", ascending=False).head(5)[COLS].to_string(index=False))

    ok = df[(df["lag_deep"].fillna(0) <= lag_cap) & (df["lag_inter"].fillna(0) <= lag_cap)]
    print(f"\n{lake} — configurations within the {lag_cap:.3f} degC lag tolerance "
          f"({len(ok)}/{len(df)}):")
    if len(ok):
        print(ok.sort_values("d_corr_all", ascending=False).head(5)[COLS].to_string(index=False))
        pick, basis = ok.sort_values("d_corr_all", ascending=False).iloc[0], "max d_corr within lag tolerance"
    else:
        print("  none — every configuration exceeds it; taking the smallest lag instead")
        pick, basis = df.sort_values("lag_deep").iloc[0], "min lag (no configuration met the tolerance)"

    base_row = b.iloc[0]
    print(f"\n{lake} — per-depth detail, baseline vs adopted:")
    print(f"{'depth':>7}{'n':>8}{'corr raw':>11}{'dcorr base':>12}{'dcorr new':>11}"
          f"{'lag base':>11}{'lag new':>10}")
    for d in sorted(pick["per_depth"]):
        n = pick["per_depth"][d]
        o = base_row["per_depth"].get(d, (np.nan,) * 5)
        band = "int" if d in inter else ("deep" if d in deep else "  -")
        print(f"{d:>7g}{n[4]:>8d}{n[1]:>11.4f}{o[0]:>+12.4f}{n[0]:>+11.4f}"
              f"{o[3]:>11.4f}{n[3]:>10.4f}  {band}")

    adopted = {"W_MAX": float(pick["W_MAX"]), "W_DEEP": float(pick["W_DEEP"]),
               "DEEP_REF": float(pick["DEEP_REF"]),
               "THERMO_GRAD_MIN": float(pick["THERMO_GRAD_MIN"]),
               "THERMO_DEPTH_MIN": float(base["THERMO_DEPTH_MIN"])}
    logger.info(f"  ADOPTED {F.describe({**base, **adopted})}  <- {basis}")

    out = {
        "lake": lake,
        "fitted_on": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "fitted_by": "notebooks/calibrate_filter.py",
        "reference": os.path.relpath(ref_path(lake, root), root).replace(os.sep, "/"),
        "criterion": {
            "score": "mean over depths of corr(ref, filtered) - corr(ref, raw)",
            "constraint": f"back-dating lag <= {lag_frac:g} * sigma_obs = {lag_cap:.4f} degC "
                          f"in both bands",
            "basis": basis,
        },
        "params": adopted,
        "diagnostics": {
            "z_thermocline_m": round(z_tc, 2),
            # What diurnal coherence would have chosen. Recorded even when not adopted: if it ever
            # exceeds params.THERMO_DEPTH_MIN the fixed cut is smoothing real solar heating.
            "derived_thermo_depth_min": round(float(derived), 2),
            # W_MAX did not come from the grid; this is where it came from and how well the two
            # independent readings of the seiche period agreed.
            "seiche": seiche,
            "band_interior_m": [min(inter), max(inter)] if inter else None,
            "band_deep_m":     [min(deep), max(deep)] if deep else None,
            "n_grid": int(len(df)),
            "n_within_tolerance": int(len(ok)),
            "d_corr_all":   round(float(pick["d_corr_all"]), 5),
            "d_corr_inter": round(float(pick["d_corr_inter"]), 5),
            "d_corr_deep":  round(float(pick["d_corr_deep"]), 5),
            "lag_inter_degC": round(float(pick["lag_inter"]), 5),
            "lag_deep_degC":  round(float(pick["lag_deep"]), 5),
            # The score of the parameters this run STARTED from -- module defaults on a first run,
            # but the previous fit on any re-run, since load_params reads filter/<lake>.json. Equal
            # to d_corr_all means the fit did not move, NOT that it matched the hand-tuned defaults.
            "starting_d_corr_all": round(float(base_row["d_corr_all"]), 5),
            "starting_params": {k: base[k] for k in
                                ("W_MAX", "W_DEEP", "DEEP_REF", "THERMO_GRAD_MIN")},
        },
    }

    path = F.filter_params_path(lake, root)
    if dry_run:
        logger.info(f"  --dry-run: would write {os.path.relpath(path, root)}")
        print(json.dumps(out, indent=2))
        return out
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    logger.info(f"  wrote {os.path.relpath(path, root)}")
    return out


def main():
    ap = argparse.ArgumentParser(description="Fit the observation filter against the free run")
    ap.add_argument("arg_file", help="Path to a run config JSON (e.g. args/run_enkf.json)")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--lake", default=None, help="Lake to fit from the config's \"lakes\" block")
    grp.add_argument("--lakes", default=None,
                     help="Comma-separated lakes to fit in sequence, or 'all' for every block in "
                          "the config's \"lakes\". Lakes without observations or without a free run "
                          "are skipped; a genuine failure is logged and the batch continues.")
    ap.add_argument("--quick", action="store_true", help="16-point grid instead of 108")
    ap.add_argument("--lag-frac", type=float, default=LAG_FRAC_DEFAULT,
                    help=f"share of sigma_obs the filter may inject as a lag bias "
                         f"(default {LAG_FRAC_DEFAULT})")
    ap.add_argument("--thermo-depth-min", type=float, default=None,
                    help="pin the surface exclusion depth in m instead of deriving it from diurnal "
                         "coherence. Any value already in filter/<lake>.json is an OUTPUT of a "
                         "previous run and is always re-derived, never reused")
    ap.add_argument("--dry-run", action="store_true", help="print the json instead of writing it")
    cli = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
    logging.getLogger("filter_observations").setLevel(logging.WARNING)

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

    batch, failed, skipped = len(lakes) > 1, [], []
    for lake in lakes:
        try:
            calibrate_lake(merge_lake_args(raw, lake=lake), quick=cli.quick,
                           lag_frac=cli.lag_frac, dry_run=cli.dry_run,
                           thermo_depth_min=cli.thermo_depth_min)
        except FileNotFoundError as exc:
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
        logger.info(f"=== batch complete: {done}/{len(lakes)} calibrated"
                    + (f"; {len(skipped)} skipped: {', '.join(skipped)}" if skipped else "")
                    + (f"; FAILED: {', '.join(failed)}" if failed else "") + " ===")
        if failed:
            sys.exit(1)


if __name__ == "__main__":
    main()

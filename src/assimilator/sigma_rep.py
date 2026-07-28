"""Observation-error model: turn the observation set into a per-observation sigma.

The scalar `sigma_obs` the filters have used until now is really an estimate of REPRESENTATIVENESS
error, not instrument error -- a thermistor is good to ~0.05 degC, so a 0.5 degC observation error
is almost entirely "how far a point measurement sits from the lake-mean value a 1D column model
predicts". That quantity is not a constant. It peaks at the thermocline, where a sharp gradient is
displaced past a fixed sensor by internal seiches, and it collapses in the deep and in the mixed
season. On upperlugano at 11 m it runs 0.21 degC in winter against 1.17 degC in summer, and at 40 m
it is 0.03 degC all year -- so the single 0.5 is roughly half the truth at the thermocline and
seventeen times too large in the deep.

Sigma is therefore resolved per observation as

    sigma_i^2  =  sigma_common^2  +  sigma_rep(depth_i, month_i)^2 / N_i

  sigma_rep    the depth- and season-dependent representativeness error, FITTED from the
               observations and the lake's free run by notebooks/sigma_rep_from_obs.py ->
               observations/<lake>/sigma_rep.json. It is the part that AVERAGES DOWN across
               stations, which is why N_i (= n_stations) divides it; N_i is 1 on every
               single-station lake, which is all of them today.

  sigma_common the part no station can see in itself -- instrument error, plus the slower
               representativeness error the fitted term does not cover. A config scalar, not
               fitted. See DEFAULT_SIGMA_COMMON for why it is not optional.

Deliberately CLIMATOLOGICAL, not instantaneous: sigma_rep is a fitted function of (depth, month),
never the spread of the hour being assimilated. An instantaneous estimate would correlate with the
innovation it is weighting -- both are large when the lake is horizontally heterogeneous -- which
breaks the filter's assumption that R is independent of the observation.

A lake with no sigma_rep.json falls back to the scalar `sigma_obs`, so its behaviour is bit-for-bit
what it was before this module existed. Deleting the json is the kill switch.
"""
import os
import json
import logging

import numpy as np

from .functions import ROOT, resolve_root

logger = logging.getLogger(__name__)

# The instrument term, and the floor under the whole error model.
#
# It is NOT optional when a sigma_rep table is present, which is why the default is non-zero. The
# fitted sigma_rep measures only the SUB-DAILY band -- the variability the free run cannot reproduce
# within a day -- so representativeness error at synoptic scales (a storm tilting the lake for three
# days) is not in it, and sigma_rep is a lower bound. That matters most in the deep, where sigma_rep
# collapses to a few hundredths: with no floor, R there would fall ~250x against the scalar 0.5 it
# replaces and the filter would overtrust the observation. At 0.075 the deep observation error lands
# at sqrt(0.075^2 + 0.03^2) ~ 0.08 degC instead, a 38x reduction rather than 250x.
#
# 0.075 degC: a thermistor chain is good to ~0.05, the rest allows for calibration drift between
# services. Set "sigma_common" in the run config to override.
DEFAULT_SIGMA_COMMON = 0.075


def depth_key(d):
    """JSON object keys are strings; use one formatting everywhere so 1.0 and 1 are the same key."""
    return f"{float(d):g}"


def resolve_sigma_common(cfg, depth):
    """sigma_common for one depth. Accepts a scalar, or a depth-keyed step function

        "sigma_common": {"0": 0.2, "7": 0.1}

    read as "0.2 from 0 m down, 0.1 from 7 m down" -- the value of the deepest key at or above
    `depth`.

    WHY it may need to be depth-dependent. sigma_common carries the error a station cannot see in
    itself: its offset from the pelagic column the 1D model represents. That offset is largest at
    the surface, where a moored buoy sits in shallower, more sheltered water than mid-lake, and
    vanishes at depth where the basin is horizontally homogeneous. Measured on upperlugano against
    the Gandria CTD, the buoy runs +0.45 degC at 0.5 m and +0.28 at 1 m (21 of 23 casts) but is
    within noise of zero from 3 m down.

    Note what a depth step buys and what it does not: it absorbs a persistent bias as variance, so
    it improves calibration, not accuracy -- a warm surface analysis stays warm, the filter just
    stops being overconfident about it. A bias correction on the innovation is the accurate fix.
    """
    sc = cfg.get("sigma_common", DEFAULT_SIGMA_COMMON)
    if not isinstance(sc, dict):
        return float(sc)
    # deepest breakpoint at or above this depth; above every breakpoint -> the shallowest value
    edges = sorted((float(k), float(v)) for k, v in sc.items())
    value = edges[0][1]
    for z, v in edges:
        if float(depth) >= z:
            value = v
    return float(value)


def sigma_rep_path(cfg):
    """The fitted table for the run: 'sigma_rep_file' (repo-relative or absolute) if given, else
    observations/<lake>/sigma_rep.json -- beside the observations it was fitted from."""
    override = cfg.get("sigma_rep_file")
    if override:
        return resolve_root(override)
    return os.path.join(ROOT, "observations", cfg["lake"], "sigma_rep.json")


def load_sigma_rep(cfg):
    """The fitted sigma_rep table, or None when the lake has none.

    Absent is NOT an error (unlike perturbations/<lake>.json): a lake without a free run or without
    observations cannot have one, and it correctly falls back to the scalar sigma_obs."""
    path = sigma_rep_path(cfg)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        table = json.load(f)
    if not isinstance(table.get("sigma_rep"), dict):
        raise ValueError(f"{path}: malformed sigma_rep table — missing 'sigma_rep' object")
    # A table fitted on a different series than the one being assimilated double-counts: the
    # adaptive filter removes most of the sub-daily variability sigma_rep measures, so a raw-fitted
    # table applied to filtered observations overstates R several-fold at the thermocline.
    src = str(table.get("source", ""))
    obs_file = str(cfg.get("obs_file", ""))
    if ("_filtered" in obs_file) != ("_filtered" in src):
        logger.warning(f"[sigma] {os.path.relpath(path, ROOT)} was fitted on "
                       f"'{src.split(' (buoy)')[0]}' but this run assimilates "
                       f"'{obs_file or 'observations/<lake>/temperature.csv'}' — refit sigma_rep on "
                       f"the series being assimilated, or R will be wrong at the thermocline")
    return table


def _sigma_rep_at(table, depth, month):
    """sigma_rep for one (depth, month): the month's fitted value, else the depth's all-season
    value, else None (-> the caller falls back to the scalar sigma_obs)."""
    entry = table["sigma_rep"].get(depth_key(depth))
    if entry is None:
        return None
    by_month = entry.get("by_month") or {}
    v = by_month.get(str(int(month)))
    return float(v) if v is not None else (
        float(entry["all"]) if entry.get("all") is not None else None)


def resolve_sigma_obs(depths, n_stations, when, cfg, table):
    """Per-observation sigma for one analysis, aligned with the obs vector.

    `depths` are the obs depths being assimilated, `n_stations` how many stations backed each (1 on
    a single-station lake), `when` the analysis instant, whose month selects the season. Returns a
    plain float when no table applies, so the scalar path stays scalar and old behaviour is
    untouched."""
    sigma_obs = float(cfg["sigma_obs"])
    if table is None:
        return sigma_obs

    n = np.asarray(n_stations, dtype=float)
    n = np.where(n >= 1, n, 1.0)                       # a reading exists, so at least one station
    out = np.empty(len(depths), dtype=float)
    for i, d in enumerate(depths):
        rep = _sigma_rep_at(table, d, when.month)
        # No fitted entry for this depth -> keep the scalar. Mixing a fitted depth and a fallback
        # depth in one vector is fine and intended.
        sc = resolve_sigma_common(cfg, d)
        out[i] = sigma_obs if rep is None else float(np.sqrt(sc ** 2 + rep ** 2 / n[i]))
    return out


def sigma_obs_by_depth(obs_df, cfg, table):
    """One sigma per depth for OpenDA, whose stochObserver carries a single standardDeviation per
    depth time series (openda/config.py::_obs_formatter_rows) and so CANNOT express the month and
    N dependence the native engine applies per observation.

    The closest single number the format admits: the RMS, over the observations actually present, of
    the per-observation sigma the native engine would have used. The two engines therefore differ by
    design here -- within a season the native engine varies sigma and OpenDA cannot. That divergence
    is logged rather than hidden, and it is far smaller than leaving OpenDA on the scalar 0.5 while
    the native engine uses the table, which would make any cross-engine comparison meaningless."""
    sigma_obs = float(cfg["sigma_obs"])
    if table is None or obs_df.empty:
        return {float(d): sigma_obs for d in sorted(obs_df["depth"].unique())}

    out = {}
    for d, g in obs_df.groupby("depth"):
        months = g["time"].dt.month.to_numpy()
        n = g["n_stations"].to_numpy(dtype=float) if "n_stations" in g else np.ones(len(g))
        sigma_common = resolve_sigma_common(cfg, d)
        vals = []
        for m, ni in zip(months, np.where(n >= 1, n, 1.0)):
            rep = _sigma_rep_at(table, d, m)
            vals.append(sigma_obs if rep is None
                        else np.sqrt(sigma_common ** 2 + rep ** 2 / ni))
        out[float(d)] = float(np.sqrt(np.mean(np.square(vals))))
    return out


def log_sigma_summary(cfg, table, by_depth=None):
    """Say, once, what observation-error model this run is actually using — the counterpart to the
    'sigma_obs=' knob in the run header, which no longer tells the whole story."""
    if table is None:
        logger.info(f"[sigma] scalar sigma_obs={cfg['sigma_obs']} degC (no sigma_rep table)")
        return
    depths = sorted(table["sigma_rep"], key=float)
    sc = cfg.get("sigma_common", DEFAULT_SIGMA_COMMON)
    sc_txt = (f"{sc} degC" if not isinstance(sc, dict) else
              "depth-stepped " + ", ".join(f">={float(k):g}m: {float(v)}"
                                           for k, v in sorted(sc.items(), key=lambda kv: float(kv[0]))))
    logger.info(f"[sigma] sigma^2 = sigma_common^2 + sigma_rep(depth, month)^2 / n_stations  "
                f"(sigma_common={sc_txt}, fallback sigma_obs={cfg['sigma_obs']} degC)")
    logger.info(f"[sigma] sigma_rep fitted at {len(depths)} depth(s) "
                f"{depths[0]}-{depths[-1]} m <- {os.path.relpath(sigma_rep_path(cfg), ROOT)}"
                + (f" ({table['estimator'][:60]}...)" if table.get("estimator") else ""))
    if by_depth:
        logger.info("[sigma] effective sigma by depth (RMS over the obs): "
                    + ", ".join(f"{d:g}m={s:.3f}" for d, s in sorted(by_depth.items())))

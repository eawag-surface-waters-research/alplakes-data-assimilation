"""Observation-error model: turn the observation set into a per-observation sigma.

The scalar `sigma_obs` the filters have used until now is really an estimate of REPRESENTATIVENESS
error, not instrument error — a thermistor is good to ~0.05 degC, so a 0.5 degC obs error is almost
entirely "how far a point measurement sits from the lake-mean value a 1D column model predicts".
That quantity is not a constant. It is what the stations disagree about, and it swings by a factor
of three over the season (Lower Lugano at 1 m: 0.17 degC in January, when the lake is homothermal
and every station agrees, to 0.56 degC in April, at the onset of stratification).

So sigma is resolved per observation as

    sigma_i^2  =  sigma_common^2  +  sigma_rep(depth_i, month_i)^2 / N_i

  sigma_rep   the across-station scatter, FITTED from the observations themselves
              (notebooks/sigma_rep_from_obs.py -> observations/<lake>/sigma_rep.json). It is the
              station-specific, averageable part: the mean of N stations has variance
              sigma_rep^2 / N, which is why N_i (= n_stations) divides it.

  sigma_common the part the stations CANNOT see in each other — a bias shared by all of them
              against the pelagic column the model actually represents (every Lower Lugano probe
              sits in a bay). Invisible to this data by construction, so it is a config scalar,
              tuned until the EnKF's NIS sits near 1.

Deliberately CLIMATOLOGICAL, not instantaneous: sigma_rep is a fitted function of (depth, month),
never the spread of the hour being assimilated. A per-hour spread from N = 2..4 stations carries
~40% sampling error, it is undefined when only one station reports, and — worst — it correlates
with the innovation it is weighting (both are large when the lake is horizontally heterogeneous),
which breaks the filter's assumption that R is independent of the observation.

A lake with no sigma_rep.json (every single-station lake) falls back to the scalar `sigma_obs`, so
its behaviour is bit-for-bit what it was before this module existed.
"""
import os
import json
import logging

import numpy as np

from .functions import ROOT, resolve_root

logger = logging.getLogger(__name__)

DEFAULT_SIGMA_COMMON = 0.0     # opt-in: absent from the config = the old pure-sigma_obs behaviour


def depth_key(d):
    """JSON object keys are strings; use one formatting everywhere so 1.0 and 1 are the same key."""
    return f"{float(d):g}"


def sigma_rep_path(cfg):
    """The fitted table for the run: 'sigma_rep_file' (repo-relative or absolute) if given, else
    observations/<lake>/sigma_rep.json — beside the observations it was fitted from."""
    override = cfg.get("sigma_rep_file")
    if override:
        return resolve_root(override)
    return os.path.join(ROOT, "observations", cfg["lake"], "sigma_rep.json")


def load_sigma_rep(cfg):
    """The fitted sigma_rep table, or None when the lake has none.

    Absent is NOT an error (unlike perturbations/<lake>.json): only a multi-station lake can have
    one, and a lake without it correctly falls back to the scalar sigma_obs."""
    path = sigma_rep_path(cfg)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        table = json.load(f)
    if not isinstance(table.get("sigma_rep"), dict):
        raise ValueError(f"{path}: malformed sigma_rep table — missing 'sigma_rep' object")
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

    `depths` are the obs depths being assimilated, `n_stations` how many stations backed each
    (load_obs's n_stations column), `when` the analysis instant (its month selects the season).
    Returns a plain float when nothing depth/season-dependent applies, so the scalar path stays
    scalar and the old behaviour is untouched."""
    sigma_obs    = float(cfg["sigma_obs"])
    sigma_common = float(cfg.get("sigma_common", DEFAULT_SIGMA_COMMON))

    if table is None:
        return sigma_obs

    n = np.asarray(n_stations, dtype=float)
    n = np.where(n >= 1, n, 1.0)                       # a reading exists, so at least one station
    out = np.empty(len(depths), dtype=float)
    for i, d in enumerate(depths):
        rep = _sigma_rep_at(table, d, when.month)
        # No fitted entry for this depth -> this depth is single-station (or unseen): keep the
        # scalar. Mixing a fitted depth and a fallback depth in one vector is fine and intended.
        out[i] = sigma_obs if rep is None else np.sqrt(sigma_common ** 2 + rep ** 2 / n[i])
    return out


def sigma_obs_by_depth(obs_df, cfg, table):
    """One sigma per depth for OpenDA, whose stochObserver carries a single standardDeviation per
    depth time series (config.py::_obs_formatter_rows) and so CANNOT express the month/N dependence
    the native engines apply per observation.

    The closest single number the format admits: the RMS, over the observations actually present,
    of the per-observation sigma the native engine would have used. The two engines therefore differ
    by design here — that divergence is logged by the caller rather than hidden."""
    sigma_obs = float(cfg["sigma_obs"])
    if table is None or obs_df.empty:
        return {float(d): sigma_obs for d in sorted(obs_df["depth"].unique())}

    out = {}
    for d, g in obs_df.groupby("depth"):
        months = g["time"].dt.month.to_numpy()
        n      = g["n_stations"].to_numpy(dtype=float) if "n_stations" in g else np.ones(len(g))
        sigma_common = float(cfg.get("sigma_common", DEFAULT_SIGMA_COMMON))
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
    logger.info(f"[sigma] sigma^2 = sigma_common^2 + sigma_rep(depth, month)^2 / n_stations  "
                f"(sigma_common={cfg.get('sigma_common', DEFAULT_SIGMA_COMMON)} degC, "
                f"fallback sigma_obs={cfg['sigma_obs']} degC)")
    logger.info(f"[sigma] sigma_rep fitted at {len(depths)} depth(s) {depths} m "
                f"<- {os.path.relpath(sigma_rep_path(cfg), ROOT)}")
    if by_depth:
        logger.info("[sigma] effective sigma by depth (RMS over the obs): "
                    + ", ".join(f"{d:g}m={s:.3f}" for d, s in sorted(by_depth.items())))

"""Fit the vertical localization radius from the observations -> localization/<lake>.json.

WHY OBSERVATIONS. The radius in src/assimilator/localization.py (L0=4 m, slope=0.25) was read by
hand off one lake's innovation correlations, and nothing in the repo reproduced it. Innovations are
also awkward as a calibration source: they only exist after an EnKF run, they carry the observation
error R on the diagonal, and they say nothing about depths the buoy does not reach. The buoy series
itself needs no run at all, so every lake with observations can be fitted the same way.

METHOD. Correlate the DAY-TO-DAY CHANGE in observed temperature between depths, not temperature
itself: raw T at 5 m and 20 m both follow the seasonal cycle and correlate ~0.99 regardless of any
error structure, which would hand back a radius the size of the lake. First-differencing removes
that cycle and leaves the synoptic-scale variability the daily filter is actually correcting. For
each depth, the radius is the smallest separation at which that correlation falls below `threshold`.
Then L(z) = max(L0, slope*z) is grid-fitted to those per-depth cutoffs.

SEASONS. Each season is fitted separately and the ADOPTED radius is their envelope (componentwise
max), not a fit to the pooled year. Pooling looks natural and is wrong: summer surface dT has far
larger variance than winter, so the pooled correlation is a variance-weighted blend of two regimes
and can land below BOTH of them (4 of 7 configured lakes; hallwil pools to slope 0.24 against 0.45
stratified / 0.58 mixed). A shorter radius localizes harder, so that artifact would quietly throw
away real information; the envelope errs the safe way, toward keeping it.

VALIDATION. On upperlugano the STRATIFIED-season fit independently recovers the hand-fitted floor --
L0 = 4.0 m exactly, from data that never passed through the filter -- and its per-depth cutoffs match
the docstring table over 0.5-30 m. (The adopted envelope then takes the wider mixed-season floor,
6.5 m.) Every lake fitted so far wants a wider radius than the hardcoded max(4, 0.25z), by 1.3x at
upperlugano and 2.4x at hallwil.

WHAT THIS CANNOT DO. The buoy stops at 40 m while the column runs to ~288 m, so nothing here
constrains the radius over 86% of the state. The fit is an extrapolation below the deepest
observation, and the slope is doing all the work down there. That is tolerable only because the
intended behaviour in the deep is "taper to zero and free-run" -- any modest slope achieves it, and
the precision matters where the observations are. Do not read L(200 m) as a measured quantity.

Usage:  python notebooks/localization_from_obs.py args/run_enkf.json [--lake L | --lakes a,b,c]
                                                  [--threshold 0.45] [--check]
"""
import os
import sys
import json
import logging
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# this file lives at <repo>/notebooks/; add src/ so assimilator imports resolve
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from assimilator.functions import ROOT, merge_lake_args, resolve_obs_path
from assimilator import localization

logger = logging.getLogger(__name__)

THRESHOLD_DEFAULT = 0.45   # matches the 2/sqrt(20) sampling floor the original hand-fit used
MIN_DAYS          = 60     # below this a seasonal correlation is not worth reporting


# ---------------------------------------------------------------------------
# observations -> per-depth correlation cutoffs
# ---------------------------------------------------------------------------

def load_depth_matrix(obs_path):
    """Observation CSV -> daily (time x depth) matrix. Daily means collapse sub-daily sampling
    so every depth shares one time axis; the filter assimilates once a day anyway."""
    obs = pd.read_csv(obs_path)
    missing = {"time", "depth", "value"} - set(obs.columns)
    if missing:
        raise ValueError(f"{obs_path}: missing column(s) {sorted(missing)}")
    obs["time"] = pd.to_datetime(obs["time"], utc=True, format="mixed")
    wide = (obs.pivot_table(index="time", columns="depth", values="value")
               .sort_index().resample("D").mean())
    return wide


def correlation(wide):
    """Correlation of the day-to-day CHANGE between depths (see module docstring: raw temperature
    is dominated by the seasonal cycle and would correlate everything with everything)."""
    return wide.diff().dropna(how="all").corr()


def depth_cutoffs(corr, threshold):
    """Per depth, the smallest separation whose correlation falls below `threshold`.

    Returns a list of dicts carrying the censoring state, which the fit needs to stay honest:
      "ok"    - a genuine crossing bracketed by the depth grid
      "left"  - the very nearest neighbour is already below threshold, so the true radius is
                SHORTER than the grid can resolve; the value is an upper bound
      "right" - correlation never drops, so the true radius is LONGER than the deepest available
                separation; the value is a lower bound
    Both bounds are one-sided in the fit rather than dropped, so a coarse depth grid (upperlugano
    jumps to 5 m spacing below 21 m) biases the answer as little as possible.
    """
    depths = np.asarray(corr.columns, dtype=float)
    C      = corr.values
    out    = []
    for i, z in enumerate(depths):
        order = [j for j in np.argsort(np.abs(depths - z)) if j != i]
        sep   = None
        for rank, j in enumerate(order):
            if np.isnan(C[i, j]):
                continue
            if C[i, j] < threshold:
                sep    = abs(depths[j] - z)
                state  = "left" if rank == 0 else "ok"
                break
        else:
            sep, state = float(np.abs(depths - z).max()), "right"
        out.append({"depth": float(z), "cutoff": float(sep), "state": state})
    return out


# ---------------------------------------------------------------------------
# L(z) = max(L0, slope*z)
# ---------------------------------------------------------------------------

def fit_radius(cutoffs, L0_grid=None, slope_grid=None):
    """Grid-fit L(z) = max(L0, slope*z). Grid search rather than a gradient method because the
    hinge is non-smooth and the parameter space is two-dimensional and tiny.

    Censored points contribute one-sided loss only: a "right" point (correlation never dropped)
    penalises the fit only when it predicts a radius SHORTER than the observed lower bound, and a
    "left" point only when it predicts LONGER than the upper bound. Scoring them as equalities
    would drag the slope toward whatever the depth grid happened to resolve.
    """
    L0_grid    = np.arange(1.0, 15.01, 0.25) if L0_grid    is None else L0_grid
    slope_grid = np.arange(0.0,  0.81, 0.01) if slope_grid is None else slope_grid

    z     = np.array([c["depth"]  for c in cutoffs])
    obs_L = np.array([c["cutoff"] for c in cutoffs])
    state = np.array([c["state"]  for c in cutoffs])

    best = (np.inf, None, None)
    for L0 in L0_grid:
        for s in slope_grid:
            pred = np.maximum(L0, s * z)
            resid = pred - obs_L
            resid = np.where(state == "right", np.minimum(resid, 0.0), resid)   # only under-prediction hurts
            resid = np.where(state == "left",  np.maximum(resid, 0.0), resid)   # only over-prediction hurts
            rss = float(np.sum(resid ** 2))
            if rss < best[0]:
                best = (rss, float(L0), float(s))
    rss, L0, slope = best
    return L0, slope, float(np.sqrt(rss / max(len(z), 1)))


def _season_masks(index):
    """Season split matches perturbate._season_mask and sigma_rep.json: stratified = May-Oct."""
    strat = (index.month >= 5) & (index.month <= 10)
    return {"stratified": strat, "mixed": ~strat}


# ---------------------------------------------------------------------------
# per-lake driver
# ---------------------------------------------------------------------------

def fit_localization(cfg, threshold=THRESHOLD_DEFAULT, run_check=False, out_dir=None):
    lake     = cfg["lake"]
    obs_path = resolve_obs_path(cfg)
    if not os.path.isfile(obs_path):
        raise FileNotFoundError(f"{lake}: no observations at {os.path.relpath(obs_path, ROOT)}")

    wide = load_depth_matrix(obs_path)
    if wide.shape[1] < 3:
        raise ValueError(f"{lake}: only {wide.shape[1]} observation depth(s); need >= 3 to fit a radius")
    depths = np.asarray(wide.columns, dtype=float)
    logger.info(f"{lake}: {wide.shape[0]} days x {wide.shape[1]} depths "
                f"({depths.min():.1f}-{depths.max():.1f} m) from {os.path.relpath(obs_path, ROOT)}")

    # Fit each season SEPARATELY, then take the envelope. Fitting the pooled year instead looks
    # natural but is unsound: summer surface dT has far larger variance than winter, so the pooled
    # correlation is a variance-weighted blend of two regimes and can fall BELOW both (measured on
    # 4 of 7 configured lakes; hallwil pooled to slope 0.24 against 0.45 / 0.58 seasonal). A shorter
    # radius localizes harder, so that artifact silently discards real information.
    per_season = {}
    for name, mask in _season_masks(wide.index).items():
        sub = wide[mask]
        if len(sub) < MIN_DAYS:
            logger.warning(f"{lake}: {name} has only {len(sub)} days (< {MIN_DAYS}) — skipping")
            continue
        cuts = depth_cutoffs(correlation(sub), threshold)
        L0_s, slope_s, rms_s = fit_radius(cuts)
        per_season[name] = {"n_days": int(len(sub)), "L0": L0_s, "slope": slope_s,
                            "rms_residual_m": rms_s, "cutoffs": cuts}
        logger.info(f"  {name:<11} n={len(sub):3d}d  L0={L0_s:5.2f} m  slope={slope_s:.3f}  "
                    f"rms={rms_s:.2f} m  L({depths.max():.0f}m)={max(L0_s, slope_s*depths.max()):.1f} m")

    # Pooled fit: kept only as a diagnostic so the artifact above stays visible in the JSON.
    cuts_all                   = depth_cutoffs(correlation(wide), threshold)
    L0_pool, slope_pool, rms_pool = fit_radius(cuts_all)
    logger.info(f"  {'pooled':<11} n={len(wide):3d}d  L0={L0_pool:5.2f} m  slope={slope_pool:.3f}  "
                f"rms={rms_pool:.2f} m  (diagnostic only — not used)")

    if per_season:
        # Componentwise max IS the pointwise envelope of the seasonal curves, because
        # max_s max(L0_s, slope_s*z) == max(max_s L0_s, (max_s slope_s)*z).
        L0    = max(v["L0"]    for v in per_season.values())
        slope = max(v["slope"] for v in per_season.values())
        basis = f"envelope (max) of {'/'.join(per_season)}"
    else:
        L0, slope = L0_pool, slope_pool
        basis = "pooled (no season had enough days)"
    logger.info(f"  {'ADOPTED':<11}        L0={L0:5.2f} m  slope={slope:.3f}  "
                f"L({depths.max():.0f}m)={max(L0, slope*depths.max()):.1f} m  <- {basis}")
    if per_season and slope_pool < min(v["slope"] for v in per_season.values()) - 1e-9:
        logger.info(f"  (pooled slope {slope_pool:.2f} sits below both seasons — "
                    f"the variance-weighting artifact; envelope used instead)")

    n_censored = sum(c["state"] != "ok" for c in cuts_all)
    if n_censored:
        logger.info(f"  {n_censored}/{len(cuts_all)} depths censored by the observation grid "
                    f"(bounds only, fitted one-sided)")
    logger.info(f"  hardcoded default for comparison: L0={localization.L0_DEFAULT:.1f} m "
                f"slope={localization.SLOPE_DEFAULT:.2f}")

    # Seasonal disagreement is worth a warning: the shipped radius assumes one curve fits both.
    if len(per_season) == 2:
        s_max = max(per_season, key=lambda k: per_season[k]["slope"])
        s_min = min(per_season, key=lambda k: per_season[k]["slope"])
        if per_season[s_max]["slope"] > 2 * max(per_season[s_min]["slope"], 1e-6):
            logger.warning(f"{lake}: seasonal slopes differ by more than 2x "
                           f"({s_max} {per_season[s_max]['slope']:.2f} vs "
                           f"{s_min} {per_season[s_min]['slope']:.2f}) — a single radius may not fit both")

    out = {
        "lake": lake,
        "fitted_on": datetime.now(timezone.utc).date().isoformat(),
        "source": os.path.relpath(obs_path, ROOT).replace("\\", "/"),
        "estimator": ("smallest separation at which the correlation of DAY-TO-DAY observed "
                      f"temperature change falls below {threshold}; L(z)=max(L0, slope*z) grid-fitted "
                      "to the per-depth cutoffs, censored depths fitted one-sided. Each season is "
                      "fitted separately and the adopted L0/slope is their ENVELOPE (componentwise "
                      "max) — pooling the year first can fall below both seasons. Seasons: "
                      "mixed=Nov-Apr, stratified=May-Oct."),
        "basis": basis,
        "units": "m",
        "threshold": threshold,
        "n_days": int(wide.shape[0]),
        "obs_depth_range_m": [float(depths.min()), float(depths.max())],
        "caveat": (f"Constrained only over {depths.min():.1f}-{depths.max():.1f} m. Below the deepest "
                   "observation L(z) is extrapolation, not measurement."),
        "L0": L0,
        "slope": slope,
        "pooled_diagnostic": {"L0": L0_pool, "slope": slope_pool, "rms_residual_m": rms_pool,
                              "note": "single fit over the whole year; NOT used — see estimator"},
        "cutoffs": cuts_all,
        "by_season": {k: {kk: vv for kk, vv in v.items()} for k, v in per_season.items()},
    }

    out_dir = out_dir or os.path.join(ROOT, "localization")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{lake}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logger.info(f"{lake}: wrote {os.path.relpath(out_path, ROOT)}")

    if run_check:
        _plot(lake, wide, cuts_all, per_season, L0, slope, threshold, out_dir)
    return out


def _plot(lake, wide, cuts_all, per_season, L0, slope, threshold, out_dir):
    """QA figure: correlation heatmap + the fitted radius against the measured cutoffs."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    depths = np.asarray(wide.columns, dtype=float)
    C      = correlation(wide)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    im = ax1.imshow(C.values, vmin=-1, vmax=1, cmap="RdBu_r")
    ax1.set_xticks(range(len(depths))); ax1.set_xticklabels([f"{d:g}" for d in depths], rotation=90, fontsize=7)
    ax1.set_yticks(range(len(depths))); ax1.set_yticklabels([f"{d:g}" for d in depths], fontsize=7)
    ax1.set_title(f"{lake}: corr of day-to-day dT")
    ax1.set_xlabel("depth (m)"); ax1.set_ylabel("depth (m)")
    fig.colorbar(im, ax=ax1, shrink=0.8)

    marker = {"ok": "o", "left": "v", "right": "^"}
    for st in ("ok", "left", "right"):
        pts = [(c["depth"], c["cutoff"]) for c in cuts_all if c["state"] == st]
        if pts:
            ax2.scatter(*zip(*pts), marker=marker[st], s=45,
                        label={"ok": "measured", "left": "upper bound", "right": "lower bound"}[st])
    for name, v in per_season.items():
        ax2.plot(depths, np.maximum(v["L0"], v["slope"] * depths), "--", lw=1,
                 label=f"{name}: max({v['L0']:.1f}, {v['slope']:.2f}z)")
    ax2.plot(depths, np.maximum(L0, slope * depths), "k-", lw=2,
             label=f"adopted envelope: max({L0:.1f}, {slope:.2f}z)")
    ax2.plot(depths, np.maximum(localization.L0_DEFAULT, localization.SLOPE_DEFAULT * depths),
             ":", color="grey", lw=2,
             label=f"hardcoded: max({localization.L0_DEFAULT:.0f}, {localization.SLOPE_DEFAULT}z)")
    ax2.set_xlabel("depth (m)"); ax2.set_ylabel("localization radius L (m)")
    ax2.set_title(f"threshold {threshold}")
    ax2.legend(fontsize=7); ax2.grid(alpha=0.3)

    fig.tight_layout()
    png = os.path.join(out_dir, f"{lake}_check.png")
    fig.savefig(png, dpi=130)
    plt.close(fig)
    logger.info(f"{lake}: wrote {os.path.relpath(png, ROOT)}")


# ---------------------------------------------------------------------------
# CLI wrapper (mirrors perturbations_from_icon.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit the vertical localization radius from observations")
    parser.add_argument("arg_file", help="Path to a run config JSON (e.g. args/run_enkf.json)")
    lake_grp = parser.add_mutually_exclusive_group()
    lake_grp.add_argument("--lake", default=None, help="Lake to fit from the config's \"lakes\" block")
    lake_grp.add_argument("--lakes", default=None,
                          help="Comma-separated lakes to fit in sequence, or 'all' for every block "
                               "in the config's \"lakes\". A failure is logged and the batch "
                               "continues, exiting non-zero if any lake failed.")
    parser.add_argument("--threshold", type=float, default=THRESHOLD_DEFAULT,
                        help=f"Correlation below which depths count as decoupled (default {THRESHOLD_DEFAULT}, "
                             "the 2/sqrt(20) ensemble sampling floor)")
    parser.add_argument("--check", action="store_true", help="Also write <lake>_check.png")
    cli = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-8s | %(name)-16s | %(message)s",
                        datefmt="%H:%M:%S")

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

    batch, failed = len(lakes) > 1, []
    for lake in lakes:
        try:
            fit_localization(merge_lake_args(raw, lake=lake),
                             threshold=cli.threshold, run_check=cli.check)
        except Exception:                      # noqa: BLE001 - isolate each lake in a batch
            if not batch:
                raise
            logger.exception(f"lake '{lake}' FAILED - continuing with the rest of the batch")
            failed.append(lake)

    if batch:
        logger.info(f"=== batch complete: {len(lakes) - len(failed)}/{len(lakes)} ok"
                    + (f"; failed: {', '.join(failed)}" if failed else "") + " ===")
        if failed:
            sys.exit(1)

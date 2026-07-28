"""Fit sigma_rep(depth, season) from the sub-daily variability the model cannot reproduce.

THE OBSERVATION ERROR IS NOT THE SENSOR'S ACCURACY. A thermistor is good to a few hundredths of a
degree, yet what it reports differs from the pelagic column a 1D model carries by far more, because
a point measurement is not a column average. That difference is REPRESENTATIVENESS error, it
dominates the error budget, and it varies strongly with depth and season. The runtime combines the
two parts as

    sigma_i^2 = sigma_common^2 + sigma_rep(depth_i, month_i)^2 / N_i

This script fits sigma_rep. sigma_common is the instrument term and is a configuration constant
(see SIGMA_COMMON below).

THE ESTIMATOR. At a fixed depth in the thermocline most hour-to-hour movement is not heat content
changing but a sharp vertical gradient being displaced past the sensor by basin-scale internal
seiches. A 1D column has no horizontal dimension, so it cannot tilt, so it cannot produce that
signal at all. Compare, day by day and depth by depth, how much the observations move against how
much the free run moves; what the observations do and the model cannot is unrepresentable:

    sigma_rep(z, season)^2 = mean_days var_within_day(obs) - mean_days var_within_day(free run)

WITHIN-DAY, because it makes the estimate immune to the free run's drift. A free run wanders from
the truth over a season, which disqualifies it from most comparisons; over a single day it
contributes nothing to the variance. (The same reasoning made the observation filter's lag penalty
model-free.) The inner join puts the model on exactly the observations' timestamps, so both
variances come from the same samples and any sampling cadence handicaps them identically.

DIFFERENCE OF MEANS, not mean of differences. Sampling noise makes the model's daily variance exceed
the observations' on some days; clipping each day at zero would bias the result upward. Both
variances are averaged first and subtracted once, and only the final result is clipped.

WHY THIS AND NOT A CTD REFERENCE. The obvious alternative is to compare the buoy against a mid-lake
CTD cast. It was tried and is kept only as validation, because in practice it works on one lake:
upperlugano has 23 coincident casts across both seasons, while hallwil has 15 ending in 2023, murten
29 from 2022-24, aegeri 7 with a single summer cast, greifensee an automated profiler whose geometry
is unestablished, geneva two candidate references disagreeing by 4.6x, and maggiore none at all.
This estimator gives all seven lakes, 365 days each, in the run year, by one procedure -- and on
upperlugano, the one lake where the CTD fit is solid, the two agree to 7% (1.168 vs 1.255 degC at
the same peak depth, 11 m) from data that share nothing.

WHAT IT CONFLATES, and why the ambiguity is confined. The estimator cannot separate "the model
cannot represent this" from "the model should reproduce this and does not". That distinction is only
ambiguous where the model produces a substantial share of the observed daily movement. Measured on
upperlugano, that share is:

    0.5 m 58%   1 m 69%   3 m 54%  |  5 m 23%   7 m 10%   11 m 3%   40 m 8%

Above ~3 m the excess could be representativeness error or the model's known diurnal amplitude
deficit (it reproduces ~60% of the observed daily swing at 0.5-3 m), and this estimator would count
either as sigma_rep. Below ~5 m the model produces 3-10% of the movement, so the excess IS the
observed variability, and there the argument is structural rather than statistical: a column cannot
tilt. The ambiguous zone is the top few metres -- exactly where the observation filter already sets
THERMO_DEPTH_MIN and declines to act, for the same reason.

WHAT IT MISSES. Only the sub-daily band. Representativeness error also lives at synoptic scales --
wind-driven tilting persisting through a storm, upwelling moving different water past the sensor for
days -- and averaging variance within each day removes all of it. sigma_rep from this script is
therefore a LOWER BOUND, and underestimating the observation error makes the filter overtrust the
observation, which is the unsafe direction. It bites hardest in the deep, where sigma_rep collapses
to a few hundredths: sigma_common is what stops R collapsing with it, and it is not optional here.

Usage:
    python notebooks/sigma_rep_from_obs.py args/run_enkf.json --lakes all --smooth 0.5
    python notebooks/sigma_rep_from_obs.py args/run_enkf.json --lake upperlugano --dry-run

Fitting the FILTERED series (--obs-file .../temperature_filtered.csv) writes sigma_rep_filtered.json
alongside. That pairing matters: sigma_rep is the representativeness error OF THE ASSIMILATED
SERIES, and the adaptive filter has already removed a large part of the sub-daily variability this
estimator measures, so assimilating filtered observations against a raw-fitted table double-counts.
"""
import os
import sys
import json
import logging
import argparse
import datetime as dt

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "notebooks"))
from assimilator.functions import ROOT, merge_lake_args                          # noqa: E402

logger = logging.getLogger(__name__)

# The instrument term in sigma_i^2 = sigma_common^2 + sigma_rep^2/N_i. A thermistor chain is good to
# ~0.05 degC; 0.075 allows for calibration drift between services. It is a CONFIG value, not a
# fitted one -- set "sigma_common": 0.075 in the run config. It is recorded in every table written
# here so a reader can see which floor the fit was intended to sit on.
#
# It carries more weight than its size suggests. sigma_rep falls to a few hundredths in the deep, so
# without a floor R would drop ~250x against the scalar sigma_obs = 0.5 it replaces. With 0.075 the
# deep observation error lands at sqrt(0.075^2 + 0.03^2) ~ 0.08 degC rather than 0.03, a 38x
# reduction instead. That is the term standing in for the synoptic-band representativeness error
# this estimator cannot see.
SIGMA_COMMON = 0.075

MIN_HOURS_PER_DAY = 6     # distinct hours a day needs before its variance is used.
# 6 rather than 12 because several buoys report 3-hourly (aegeri and greifensee deliver exactly 8
# distinct hours a day, never more), and 12 rejected 100% of their days. The comparison stays fair
# at any cadence because the inner join puts the model on the SAME timestamps as the observations.
# What a coarse cadence costs is reach: 3-hourly sampling cannot see sub-3-hour variability at all,
# so for those lakes this is a lower bound even within the sub-daily band.
MIN_DAYS   = 20           # per (depth, season)
STRATIFIED = {5, 6, 7, 8, 9, 10}     # May-Oct
SEASON_MONTHS = {m: ("stratified" if m in STRATIFIED else "mixed") for m in range(1, 13)}


def season_of(month):
    return SEASON_MONTHS[int(month)]


def depth_key(d):
    """Match assimilator.sigma_rep.depth_key exactly — the runtime looks the table up by this."""
    return f"{float(d):g}"


# ----------------------------------------------------------------------------------- loading

def load_obs_hourly(lake, obs_file=None, root=ROOT):
    """The buoy on the hourly analysis grid, binned exactly as functions.load_obs does.

    Rounding to the NEAREST hour rather than flooring matters: it is what the assimilation does, so
    the daily variance measured here is the variance of what the analysis actually sees.
    """
    path = (obs_file if obs_file and os.path.isabs(obs_file)
            else os.path.join(root, obs_file) if obs_file
            else os.path.join(root, "observations", lake, "temperature.csv"))
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{lake}: no buoy series at {os.path.relpath(path, root)}")
    obs = pd.read_csv(path, usecols=["time", "depth", "value"])
    obs["time"] = pd.to_datetime(obs["time"], utc=True, format="ISO8601")
    obs["time"] = (obs["time"] + pd.Timedelta(minutes=30)).dt.floor("60min")
    return path, obs.groupby(["depth", "time"], as_index=False)["value"].mean()


def load_ref_hourly(lake, obs, root=ROOT):
    """Free-run temperatures on the buoy's depths, long form.

    Reuses notebooks/calibrate_filter.load_ref, which already snaps observation depths to the model
    output grid, warns when two of them share a cell, and reads the T_out.dat header — one loader,
    one set of conventions.
    """
    from calibrate_filter import load_ref                                        # noqa: PLC0415
    ref = load_ref(lake, sorted(obs["depth"].unique()), root)
    long = ref.stack().rename("value").reset_index()
    long.columns = ["time", "depth", "value"]
    return long


# ----------------------------------------------------------------------------------- fitting

def fit(obs, ref, min_hours=MIN_HOURS_PER_DAY):
    """sigma_rep per (depth, season) from the daily variance the free run does not account for."""
    j = obs.merge(ref, on=["time", "depth"], how="inner", suffixes=("_obs", "_mod"))
    if j.empty:
        raise ValueError("the observations and the free run share no (time, depth)")
    j["day"] = j["time"].dt.floor("1D")
    j["season"] = j["time"].dt.month.map(season_of)

    daily = (j.groupby(["depth", "season", "day"])
              .agg(v_obs=("value_obs", "var"), v_mod=("value_mod", "var"), n=("value_obs", "size"))
              .reset_index())
    daily = daily[(daily["n"] >= min_hours) & daily["v_obs"].notna() & daily["v_mod"].notna()]

    table = {}
    for depth, g in daily.groupby("depth"):
        by_season = {}
        for season, gs in g.groupby("season"):
            if len(gs) < MIN_DAYS:
                continue
            vo, vm = float(gs["v_obs"].mean()), float(gs["v_mod"].mean())
            by_season[season] = float(np.sqrt(max(0.0, vo - vm)))
        if not by_season:
            continue
        vo_all, vm_all = float(g["v_obs"].mean()), float(g["v_mod"].mean())
        pooled = float(np.sqrt(max(0.0, vo_all - vm_all)))
        table[depth_key(depth)] = {
            "all": round(pooled, 4),
            # what each side actually moved, so a reader can judge the fit rather than take it: a
            # model share near zero means the excess is structural, a share near half means it is
            # ambiguous against the model's own diurnal deficit
            "sigma_daily_obs": round(float(np.sqrt(vo_all)), 4),
            "sigma_daily_model": round(float(np.sqrt(vm_all)), 4),
            "model_share": round(float(np.sqrt(vm_all) / np.sqrt(vo_all)), 3) if vo_all > 0 else None,
            "n_days": int(len(g)),
            "by_month": {str(m): round(by_season.get(SEASON_MONTHS[m], pooled), 4)
                         for m in range(1, 13)},
            "by_season": {s: round(v, 4) for s, v in by_season.items()},
        }
    return table


def fill_missing_seasons(table):
    """Fill a season a depth could not resolve by interpolating it from the depths that could.

    Sensors on one chain are not always deployed together. Geneva has two groups: 27 depths with
    141-184 days in each season, and 17 that reported for a single 38-day winter window and never in
    summer. Left alone, those 17 resolve `mixed` only -- and then every stratified month inherits the
    depth's POOLED std, which was measured entirely on those 38 winter days. A winter number wearing
    a summer label is worse than no number, and it is invisible in the output.

    Interpolating instead is sound here because sigma_rep is a smooth function of depth -- smooth
    enough that the table is routinely depth-smoothed anyway -- and every such depth is BRACKETED by
    depths that did resolve the season (3 m sits between 2 m and 4 m). Only bracketed depths are
    filled; nothing is extrapolated past the shallowest or deepest depth that resolved the season,
    because there the profile shape is not constrained. Filled entries are marked so the table says
    which of its numbers were measured.
    """
    depths = sorted(table, key=float)
    z_all = np.array([float(d) for d in depths])
    filled = {}

    for season in ("mixed", "stratified"):
        have = [(zi, d) for zi, d in zip(z_all, depths)
                if table[d]["by_season"].get(season) is not None]
        if len(have) < 2:
            continue
        zk = np.array([h[0] for h in have])
        vk = np.array([table[h[1]]["by_season"][season] for h in have], dtype=float)
        for zi, d in zip(z_all, depths):
            if table[d]["by_season"].get(season) is not None:
                continue
            if zi < zk[0] or zi > zk[-1]:            # outside the covered range: do not invent
                continue
            table[d]["by_season"][season] = round(float(np.interp(zi, zk, vk)), 4)
            table[d].setdefault("interpolated_seasons", []).append(season)
            filled.setdefault(season, []).append(zi)

    for d in depths:
        e = table[d]
        e["by_month"] = {str(m): round(e["by_season"].get(SEASON_MONTHS[m], e["all"]), 4)
                         for m in range(1, 13)}
    return table, filled


def _smooth_depth(depths, values, blend, passes):
    """Centred neighbour smoothing over depth. Each interior value is nudged toward the line between
    its two depth neighbours; endpoints are held fixed, so the profile's top and bottom are never
    invented and a peak keeps its measured depth."""
    z = [float(d) for d in depths]
    y = [float(v) for v in values]
    if blend <= 0 or len(z) < 3:
        return y
    b = min(max(float(blend), 0.0), 1.0)
    for _ in range(max(1, passes)):
        prev = list(y)
        for i in range(1, len(z) - 1):
            f = (z[i] - z[i - 1]) / (z[i + 1] - z[i - 1])
            target = prev[i - 1] + (prev[i + 1] - prev[i - 1]) * f
            y[i] = (1.0 - b) * prev[i] + b * target
    return [max(v, 0.0) for v in y]


def smooth_table(table, blend, passes):
    """Smooth each season curve and the pooled 'all' over depth, then rebuild by_month from them.

    The daily-variance diagnostics describe the raw pairs and are deliberately left alone: they are
    the record of what was measured before smoothing.
    """
    depths = sorted(table, key=float)
    for d, v in zip(depths, _smooth_depth(depths, [table[d]["all"] for d in depths], blend, passes)):
        table[d]["all"] = round(v, 4)

    for season in ("mixed", "stratified"):
        present = [d for d in depths if table[d]["by_season"].get(season) is not None]
        if len(present) < 3:
            continue
        sm = _smooth_depth(present, [table[d]["by_season"][season] for d in present], blend, passes)
        for d, v in zip(present, sm):
            table[d]["by_season"][season] = round(v, 4)

    for d in depths:
        e = table[d]
        e["by_month"] = {str(m): round(e["by_season"].get(SEASON_MONTHS[m], e["all"]), 4)
                         for m in range(1, 13)}
    return table


# ---------------------------------------------------------------------------------- per lake

def out_path_for(obs_path, lake, root=ROOT):
    """sigma_rep.json, or sigma_rep_filtered.json when fitting a *_filtered series.

    The suffix has to travel: a filtered run must not be handed a raw-fitted table, and the runtime
    pairs them by this name.
    """
    suffix = "_filtered" if "_filtered" in os.path.basename(obs_path) else ""
    return os.path.join(root, "observations", lake, f"sigma_rep{suffix}.json")


def fit_lake(cfg, obs_file=None, smooth=0.0, passes=2, min_hours=MIN_HOURS_PER_DAY,
             dry_run=False, root=ROOT):
    lake = cfg["lake"]
    obs_path, obs = load_obs_hourly(lake, obs_file, root)
    ref = load_ref_hourly(lake, obs, root)
    logger.info(f"{lake}: buoy {os.path.relpath(obs_path, root)} ({obs['depth'].nunique()} depths) "
                f"vs the free run ({ref['time'].nunique():,} hours)")

    hours = obs.groupby([obs["time"].dt.floor("1D"), "depth"]).size()
    logger.info(f"  median {hours.median():.0f} distinct hours per (day, depth); "
                f"{(hours >= min_hours).mean():.0%} of days meet the {min_hours}-hour minimum")

    table = fit(obs, ref, min_hours=min_hours)
    if not table:
        raise ValueError(f"{lake}: no depth reached {MIN_DAYS} days in any season "
                         f"(try a lower --min-hours; this buoy may sample coarsely)")

    # Fill before smoothing, so the smoother sees a complete profile rather than smoothing around
    # holes and pulling the endpoints of each fragment.
    table, filled = fill_missing_seasons(table)
    for season, zs in filled.items():
        logger.warning(f"  {len(zs)} depth(s) had no {season} season and were interpolated from "
                       f"neighbours: {', '.join(f'{z:g}' for z in zs[:12])}"
                       + (" ..." if len(zs) > 12 else ""))

    smoothing = None
    if smooth > 0:
        table = smooth_table(table, smooth, passes)
        smoothing = {"method": "centred neighbour over depth (endpoints fixed)",
                     "blend": smooth, "passes": passes}
        logger.info(f"  depth-smoothed: blend={smooth}, passes={passes}")

    out = {
        "lake": lake,
        "fitted_on": dt.date.today().isoformat(),
        "fitted_by": "notebooks/sigma_rep_from_obs.py",
        "source": f"{os.path.relpath(obs_path, root).replace(os.sep, '/')} (buoy) vs "
                  f"inputs/{lake}/ref/T_out.dat (free run)",
        "estimator": "sqrt(mean within-day var(obs) - mean within-day var(free run)) per "
                     "(depth, season). Within-day so the free run's drift cannot enter. Lower "
                     "bound: the sub-daily band only, so synoptic-scale representativeness error "
                     "is not included and sigma_common carries it.",
        "units": "degC",
        "resolution": "seasonal",
        "sigma_common_assumed": SIGMA_COMMON,
        "min_hours_per_day": int(min_hours),
        **({"smoothing": smoothing} if smoothing else {}),
        "sigma_rep": table,
    }

    print(f"\n{lake}   depth   mixed   strat   obs_d   mod_d  share   days")
    for d in sorted(table, key=float):
        e = table[d]
        bs = e["by_season"]
        print(f"{'':7}{float(d):6g}  {bs.get('mixed', float('nan')):6.3f}  "
              f"{bs.get('stratified', float('nan')):6.3f}  {e['sigma_daily_obs']:6.3f}  "
              f"{e['sigma_daily_model']:6.3f}  {(e['model_share'] or 0):5.2f}  {e['n_days']:5d}")

    path = out_path_for(obs_path, lake, root)
    if dry_run:
        logger.info(f"  --dry-run: would write {os.path.relpath(path, root)}")
        return out
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    logger.info(f"  wrote {os.path.relpath(path, root)}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    ap.add_argument("arg_file")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--lake", default=None)
    grp.add_argument("--lakes", default=None,
                     help="comma-separated, or 'all' for every lake in the config with a free run")
    ap.add_argument("--obs-file", default=None,
                    help="buoy CSV to fit against (default observations/<lake>/temperature.csv; "
                         "pass a *_filtered.csv to fit the filtered series — single lake only)")
    ap.add_argument("--smooth", type=float, default=0.0,
                    help="depth-smooth the written table: neighbour blend in (0,1] per pass "
                         "(default 0 = off, raw fit)")
    ap.add_argument("--smooth-passes", type=int, default=2)
    ap.add_argument("--min-hours", type=int, default=MIN_HOURS_PER_DAY,
                    help=f"distinct hours a day needs before its variance is used "
                         f"(default {MIN_HOURS_PER_DAY}; 3-hourly buoys give 8)")
    ap.add_argument("--dry-run", action="store_true", help="print the table, write nothing")
    cli = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")

    arg_file = cli.arg_file if os.path.isfile(cli.arg_file) else os.path.join(ROOT, cli.arg_file)
    with open(arg_file) as f:
        raw = json.load(f)

    if not cli.lakes:
        lakes = [cli.lake]
    elif cli.lakes.strip() == "all":
        lakes = list(raw.get("lakes", {}))
    else:
        lakes = [s.strip() for s in cli.lakes.split(",") if s.strip()]

    batch = len(lakes) > 1
    if batch and cli.obs_file:
        raise ValueError("--obs-file names a single file and is refused in batch mode")

    failed, skipped = [], []
    for lake in lakes:
        try:
            fit_lake(merge_lake_args(raw, lake=lake), obs_file=cli.obs_file, smooth=cli.smooth,
                     passes=cli.smooth_passes, min_hours=cli.min_hours, dry_run=cli.dry_run)
        except FileNotFoundError as exc:
            if not batch:
                raise
            logger.info(f"{lake}: {exc} — skipped")
            skipped.append(lake)
        except Exception:                                    # noqa: BLE001 - isolate each lake
            if not batch:
                raise
            logger.exception(f"lake '{lake}' FAILED - continuing")
            failed.append(lake)

    if batch:
        logger.info(f"=== {len(lakes) - len(failed) - len(skipped)}/{len(lakes)} fitted"
                    + (f"; {len(skipped)} skipped: {', '.join(skipped)}" if skipped else "")
                    + (f"; FAILED: {', '.join(failed)}" if failed else "") + " ===")
        if failed:
            sys.exit(1)


if __name__ == "__main__":
    main()

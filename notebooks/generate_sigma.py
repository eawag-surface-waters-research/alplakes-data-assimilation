"""Fit sigma_obs from the free-run-minus-observation gap.

    1  gap(t,z) = ref(t,z) - obs(t,z)   at the observed instants, split by season
    2  per (depth, season): std about THAT season's own mean
    3  median over depths  ->  the {"mixed": .., "stratified": ..} a run config carries

Step 2 is written to observations/<lake>/sigma_obs.json as provenance; step 3 is the only part the
filters read, and it gets there by being pasted into the run config -- the script prints exactly
what to paste (season pair).

TWO SEASONS, one value would be a compromise compromise between them, wrong in both at once for most lakes.

This estimate is expected to be AN UPPER BOUND. The gap carries model error as well as observation error.

REFIT WHENEVER THE SERIES CHANGES. These are a property of the observation series they were fitted
on, not of the lake. A value carried across a change in the filtering or the thinning is wrong in a
direction nothing in the log reveals.

ASSUMES "obs_operator": "linear", changed in another branch. The gap is taken against T_out.dat, which Simstrat writes by
interpolating the state onto the z_out depths -- so the model equivalent here is the LINEAR one.
Simstrat centres its cells at x.25/x.75, so a round obs depth sits 0.25 m from the nearest cell; a
run using the default "nearest" operator therefore carries a discretisation error (gradient x
0.25 m) that this fit does not. Set "obs_operator": "linear" in the run config. 

    python notebooks/generate_sigma.py
    python notebooks/generate_sigma.py --lakes upperlugano 
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
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from assimilator.functions import (ROOT, SEASONS, load_obs, merge_lake_args,   # noqa: E402
                                   resolve_obs_path, resolve_root, season_of)
import visualize as viz                                                        # noqa: E402

logger = logging.getLogger("generate_sigma")

MIN_HOURS = 24              # a depth must be present this often in EVERY season (minimal requirement one day ...)


def load_ref(lake, ref_path):
    """The free run as a wide frame, time-indexed, one column per output depth."""
    with open(os.path.join(ROOT, "inputs", lake, "Settings.par")) as f:
        ref_year = json.load(f)["Simulation"]["Reference year"]
    ref = viz.load_traj(ref_path, pd.Timestamp(f"{ref_year}-01-01", tz="UTC"))
    if ref is None:
        raise FileNotFoundError(f"free run not found: {ref_path}")
    return ref


def gap_frame(lake, obs_csv, ref_path, year=None, max_depth=None):
    """STEP 1: ref - obs, one column per observed depth, on the hours the two share."""
    obs = load_obs(obs_csv)                       # time, depth, value; hour-floored, depth-meaned
    ref = load_ref(lake, ref_path)
    if year is not None:
        obs = obs[obs["time"].dt.year == year]
        ref = ref[ref.index.year == year]
    if max_depth is not None:
        obs = obs[obs["depth"] <= max_depth]
    if obs.empty:
        raise ValueError("no observations left after the depth/year selection")

    model_depths = -np.asarray(ref.columns, float)                 # the columns are negative
    wide = obs.pivot_table(index="time", columns="depth", values="value")
    wide = wide.loc[wide.index.isin(ref.index)]
    if wide.empty:
        raise ValueError("the observations and the free run share no hour")

    # Inside the grid, nearest-cell snapping is what the engines do (build_H's argmin), so no
    # distance tolerance here: a fixed one would be dz_max/2 in disguise and would start dropping
    # legitimate depths on a coarser grid. Only obs BELOW the deepest cell are dropped -- there
    # the snap has nothing to snap to, and the gap would be against an extrapolated depth.
    bed = model_depths.max()
    gap, far = {}, []
    for d in wide.columns:
        if d > bed:
            far.append(d)
            continue
        gap[d] = ref.iloc[:, int(np.argmin(np.abs(model_depths - d)))].reindex(wide.index) - wide[d]
    if far:
        logger.info(f"  {lake}: {len(far)} obs depth(s) below the free run's deepest cell "
                    f"({bed:g} m): {min(far):g}..{max(far):g} m - dropped")
    if not gap:
        raise ValueError("no observed depth matches the free run's output depths")
    return pd.DataFrame(gap).sort_index(axis=1)


def sigma_by_depth(gap):
    """STEP 2: per (depth, season), the std of the gap about THAT season's own mean.

    Removing each season's own mean is what makes the result bias-free in both: a depth that runs
    warm all summer and cold all winter contributes its scatter, not its offset.

    MIN_HOURS rejects a std computed from too few hours to mean anything -- with ddof=0 a single
    sample gives 0.0, which is not a NaN and would survive the dropna to drag the median down. The
    dropna is what pairs the columns: a depth missing from either season leaves both, so the two
    describe the same set of depths."""
    season = pd.Index([season_of(t.month) for t in gap.index], name="season")
    out = {}
    for s in SEASONS:
        sub = gap[season == s]
        out[s] = sub.std(ddof=0).where(sub.count() >= MIN_HOURS).round(4)
    return pd.DataFrame(out).dropna()


def scalars(by_depth):
    """STEP 3: the median over depths, one number per season -- what the run config carries.

    The median, not the mean: sigma_rep spans an order of magnitude down a column and the
    thermocline depths are the tail, so a mean would be dragged by the few worst depths."""
    return {s: round(float(np.median(by_depth[s].to_numpy(dtype=float))), 3) for s in SEASONS}


def build_json(lake, by_depth, sigma_obs, obs_rel, ref_rel):
    return {
        "lake": lake,
        "fitted_on": dt.date.today().isoformat(),
        "fitted_by": "notebooks/generate_sigma.py",
        "source": f"{obs_rel} vs {ref_rel} (free run)",
        "estimator": "std(ref - obs) about each season's own mean, ddof=0, per (depth, season); "
                     "sigma_obs is the median over depths",
        "units": "degC",
        "_upper_bound_note": "The gap carries model error as well as observation error, so these "
                             "are an UPPER bound on the observation error, not an estimate of it.",
        "assumes_obs_operator": "linear",
        "_obs_operator_note": "Fitted against T_out.dat, which Simstrat writes by interpolating "
                              "the state onto the z_out depths -- i.e. the LINEAR observation "
                              "operator. Under the default \"nearest\" a round obs depth sits "
                              "0.25 m from the cell it is compared with, and the run carries a "
                              "discretisation error (gradient x 0.25 m) this sigma does not.",
        "_paste_note": "'sigma_obs' is the value a run config carries; nothing here is read at run "
                       "time. 'by_depth' is the fit it is the median of, kept for checking.",
        "sigma_obs": sigma_obs,
        "n_depths": len(by_depth),
        "by_depth": {f"{float(d):g}": {s: float(row[s]) for s in SEASONS}
                     for d, row in by_depth.iterrows()},
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--args", default="args/run_enkf.json",
                    help="run config the observation series is read from, per lake, so the fit "
                         "lands on exactly the series a run assimilates (default %(default)s)")
    ap.add_argument("--lakes", nargs="*", default=None,
                    help="default: every lake in the config that also has inputs/<lake>/ref/T_out.dat")
    ap.add_argument("--obs-file", default=None,
                    help="series to fit on, repo-relative, overriding the config's obs_file. "
                         "'{lake}' is substituted, so one path covers every lake: "
                         "observations/{lake}/temperature_filtered_h06_1d.csv. Fit on the series a "
                         "run actually assimilates — sigma fitted on the raw record and applied to "
                         "a filtered one overstates the error where the filter removed variance")
    ap.add_argument("--year", type=int, default=None, help="restrict to one calendar year")
    ap.add_argument("--max-depth", type=float, default=None, help="ignore obs below this depth (m)")
    ap.add_argument("--dry-run", action="store_true", help="print everything, write nothing")
    cli = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    with open(resolve_root(cli.args), encoding="utf-8") as f:
        raw = json.load(f)
    lakes = cli.lakes or sorted(raw.get("lakes") or {})
    if not lakes:
        raise SystemExit(f"{cli.args} has no \"lakes\" block — pass --lakes")

    fitted = {}
    for lake in lakes:
        cfg = merge_lake_args(raw, lake=lake)
        if cli.obs_file:
            cfg["obs_file"] = cli.obs_file.format(lake=lake)
        obs_csv = resolve_obs_path(cfg)
        ref_path = os.path.join(ROOT, "inputs", lake, "ref", "T_out.dat")
        try:
            by_depth = sigma_by_depth(gap_frame(lake, obs_csv, ref_path, cli.year, cli.max_depth))
            if by_depth.empty:
                raise ValueError(f"no depth has {MIN_HOURS}+ hours in both seasons")
        except (ValueError, FileNotFoundError, KeyError) as exc:
            logger.warning(f"  {lake:<12} skipped - {exc}")
            continue

        sigma_obs = scalars(by_depth)
        rel = lambda p: os.path.relpath(p, ROOT).replace("\\", "/")     # noqa: E731
        out = build_json(lake, by_depth, sigma_obs, rel(obs_csv), rel(ref_path))
        dest = os.path.join(ROOT, "observations", lake, "sigma_obs.json")
        if not cli.dry_run:
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            with open(dest, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
                f.write("\n")
        fitted[lake] = sigma_obs
        logger.info(f"  {lake:<12} {len(by_depth):3d} depths  "
                    f"{'would write' if cli.dry_run else 'wrote'} {rel(dest)}")

    if not fitted:
        raise SystemExit("no lake produced a gap series")

    # The only step nothing else guards: these numbers reach the filters by being retyped into a
    # run config. Print them in the exact shape they are pasted in, so the copy is a copy.
    logger.info(f"\nPASTE INTO {cli.args} — \"sigma_obs\" in each lake's block under \"lakes\":")
    for lake, sigma_obs in fitted.items():
        body = ", ".join(f'"{s}": {sigma_obs[s]}' for s in SEASONS)
        logger.info(f'    {lake:<12} "sigma_obs": {{{body}}}')
    logger.info("\nA lake whose block names no \"sigma_obs\" falls back to the config's top-level "
                "\"sigma_default\".\nNote the OpenDA engine does not read a season pair yet — it "
                "renders one standardDeviation for\nthe whole run — so a lake you paste this into "
                "is EnKF-only until that lands.")


if __name__ == "__main__":
    main()

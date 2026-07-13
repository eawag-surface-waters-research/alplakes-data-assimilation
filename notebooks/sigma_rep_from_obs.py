"""Fit the representativeness-error table sigma_rep(depth, month) from a lake's own observations.

The sibling of perturbations_from_icon.py: a calibration fitted once, offline, and then read by
every run (assimilator/sigma_rep.py -> observations/<lake>/sigma_rep.json).

sigma_rep is the across-station standard deviation at a given depth — how far apart simultaneous
stations sit. For a 1D column model that scatter is not signal: the model has no state that could
explain horizontal structure, so it is exactly the error floor of the observation, and it is what
belongs in R. It is fitted PER MONTH because it is strongly seasonal: on Lower Lugano at 1 m it
runs ~0.17 degC in January, when the lake is homothermal and the stations agree, to ~0.56 degC in
April, at the onset of stratification when bays warm faster than open water.

The estimator is the MEDIAN of the hourly across-station std, not the mean: one storm, or one
sensor drifting before it is caught, should not set the observation error for a whole month.

Only depths with >= 2 simultaneous stations can be fitted; a single-station depth is emitted with
no entry at all, so the run falls back to the scalar sigma_obs for it.

    python notebooks/sigma_rep_from_obs.py args/run_enkf.json --lake lowerlugano
    python notebooks/sigma_rep_from_obs.py args/run_enkf.json --lake lowerlugano --dry-run
"""
import os
import sys
import json
import logging
import argparse
from datetime import date

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from assimilator.functions import ROOT, load_obs, resolve_obs_path, merge_lake_args  # noqa: E402
from assimilator.sigma_rep import sigma_rep_path, depth_key  # noqa: E402

logger = logging.getLogger("sigma_rep")

MIN_HOURS = 50    # per (depth, month) cell; below this the month is too thin -> use 'all' instead


def fit(cfg, dry_run=False):
    obs_csv = resolve_obs_path(cfg)
    obs = load_obs(obs_csv)          # already carries n_stations + spread, per (depth, hour)

    multi = obs[(obs["n_stations"] >= 2) & obs["spread"].notna()]
    if multi.empty:
        logger.warning(f"{cfg['lake']}: no hour anywhere has 2+ stations reporting — nothing to "
                       f"fit. This lake stays on the scalar sigma_obs.")
        return None

    logger.info(f"{cfg['lake']}: {len(multi)} multi-station readings of {len(obs)} "
                f"({100 * len(multi) / len(obs):.0f}%), "
                f"{multi['n_stations'].min()}-{multi['n_stations'].max()} stations")

    sigma_rep = {}
    for depth, g in multi.groupby("depth"):
        by_month = {}
        for month, gm in g.groupby(g["time"].dt.month):
            if len(gm) < MIN_HOURS:
                logger.info(f"    depth {depth:g} m, month {month:>2}: only {len(gm)} hours "
                            f"(< {MIN_HOURS}) — omitted, falls back to 'all'")
                continue
            by_month[str(int(month))] = round(float(gm["spread"].median()), 4)

        entry = {
            "all":      round(float(g["spread"].median()), 4),
            "rms":      round(float(np.sqrt(np.mean(g["spread"] ** 2))), 4),
            "n_hours":  int(len(g)),
            "by_month": by_month,
        }
        sigma_rep[depth_key(depth)] = entry
        months = " ".join(f"{m}:{v:.2f}" for m, v in sorted(by_month.items(), key=lambda x: int(x[0])))
        logger.info(f"  depth {depth:g} m — median {entry['all']:.3f}, RMS {entry['rms']:.3f} degC "
                    f"over {entry['n_hours']} hours")
        logger.info(f"    by month: {months}")

    table = {
        "lake":       cfg["lake"],
        "fitted_on":  date.today().isoformat(),
        "source":     os.path.relpath(obs_csv, ROOT).replace("\\", "/"),
        "estimator":  "median of the hourly across-station std, per (depth, month); "
                      "months with < %d hours omitted" % MIN_HOURS,
        "units":      "degC",
        "sigma_rep":  sigma_rep,
    }

    out = sigma_rep_path(cfg)
    if dry_run:
        logger.info(f"--dry-run: would write {os.path.relpath(out, ROOT)}")
        print(json.dumps(table, indent=2))
        return table

    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2)
    logger.info(f"wrote {os.path.relpath(out, ROOT)}")
    return table


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Fit sigma_rep(depth, month) from a lake's observations")
    ap.add_argument("arg_file", help="Run config JSON (e.g. args/run_enkf.json)")
    ap.add_argument("--lake", default=None, help="Lake from the config's \"lakes\" block")
    ap.add_argument("--dry-run", action="store_true", help="Print the table, write nothing")
    cli = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    arg_file = cli.arg_file if os.path.isfile(cli.arg_file) else os.path.join(ROOT, cli.arg_file)
    if not os.path.isfile(arg_file):
        raise SystemExit(f"args file not found: {cli.arg_file}")
    with open(arg_file) as f:
        fit(merge_lake_args(json.load(f), lake=cli.lake), dry_run=cli.dry_run)

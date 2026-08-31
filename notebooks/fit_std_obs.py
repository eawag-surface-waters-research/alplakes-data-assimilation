"""Fit sigma_rep(depth, season) from across-station scatter, for multi-station lakes.

The companion to sigma_rep_from_obs.py, which needs a free run and fits one station against it.
Here the stations measure each other and no model is involved, so nothing the model gets wrong can
leak into R.

    sigma_i^2 = sigma_common^2 + sigma_rep(depth_i, season_i)^2 / N_i

sigma_rep is the error of ONE station; the runtime divides by the N stations actually backing each
observation. sigma_common stays a config constant: a component shared by every station shifts their
mean, not their spread, so an across-station estimator cannot see it at all.

KNOWN LIMIT of the /N term. It credits the full 1/N, which assumes station errors are independent.
Where they are not -- stations sharing a shore, say -- the true reduction is smaller and R comes out
too small at high N. Separating the two needs a reference the stations cannot provide, so this is
left as a documented assumption rather than a fitted correction.

ESTIMATOR. Per (depth, hour) with at least two stations, s = across-station sample std (ddof=1).
Per (depth, season), sigma_rep = sqrt(mean(s^2)). The rms, because E[s^2] = sigma^2 at any N, while
the median of s runs low by a factor that depends on N (0.6745 at N=2, 0.888 at N=4) -- and N varies
hour to hour on a four-station lake.

SEASONAL, not monthly. Fitted monthly on 2022-2025 the Zurich months scatter 26% year to year
against 8% sampling noise, so a monthly table carries one year's weather into the next. Seasons cut
that to 12-19%. Normalising by the season's obs std was tested and does not help: obs_std barely
moves between years, so dividing by it removes no variance.
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
from assimilator.functions import ROOT, merge_lake_args, season_of, SEASONS   # noqa: E402
from assimilator.sigma_rep import DEFAULT_SIGMA_COMMON, depth_key             # noqa: E402

logger = logging.getLogger(__name__)

MIN_HOURS = 200      # hours a (depth, season) needs before its rms means anything
MIN_STATIONS = 2     # a spread needs two readings


def load_obs(lake, obs_file=None, root=ROOT):
    """QC'd observations as hourly per-station columns, one frame per depth.

    weight <= 0 is a QC reject and is dropped here exactly as the runtime's load_obs drops it -- on
    Lower Zurich, reading the June 2025 Tiefenbrunnen outage as valid triples the stratified fit.
    """
    path = (obs_file if obs_file and os.path.isabs(obs_file)
            else os.path.join(root, obs_file) if obs_file
            else os.path.join(root, "observations", lake, "temperature.csv"))
    obs = pd.read_csv(path)
    obs["time"] = pd.to_datetime(obs["time"], utc=True, format="ISO8601")
    if "station" not in obs:
        raise ValueError(f"{path} has no 'station' column -- single-station lake, "
                         "use sigma_rep_from_obs.py instead")
    obs = obs[obs.get("weight", 1.0) > 0].dropna(subset=["value"])

    frames = {}
    for depth, g in obs.groupby("depth"):
        wide = pd.DataFrame({
            st: s.set_index("time")["value"].resample("1h").mean()
            for st, s in g.groupby("station")})
        frames[float(depth)] = wide[wide.notna().sum(axis=1) >= MIN_STATIONS]
    return path, frames


def sigma_rep(s):
    """rms of the across-station std. NaN when the sample is too thin to trust."""
    return float(np.sqrt((s ** 2).mean())) if len(s) >= MIN_HOURS else np.nan


def fit_depth(wide):
    """One depth -> per-season sigma_rep, plus the per-year spread behind it."""
    s = wide.std(axis=1, ddof=1).dropna()
    n = wide.notna().sum(axis=1).reindex(s.index)
    seasons = s.index.month.map(season_of)

    entry, by_year = {}, {}
    for name in SEASONS:
        sel = s[seasons == name]
        entry[name] = sigma_rep(sel)
        per_year = {int(y): sigma_rep(sel[sel.index.year == y])
                    for y in sorted(sel.index.year.unique())}
        by_year[name] = {y: round(v, 4) for y, v in per_year.items() if not np.isnan(v)}

    # float(), not np.float64: json.dump refuses numpy scalars
    cv = {name: (round(float(100 * np.std(list(v.values())) / np.mean(list(v.values()))), 1)
                 if len(v) > 1 else None)
          for name, v in by_year.items()}
    return entry, by_year, cv, {"n_hours": int(len(s)), "mean_n_stations": round(float(n.mean()), 2)}


def fit_lake(cfg, obs_file=None, sigma_common=DEFAULT_SIGMA_COMMON, n_ref=None, dry_run=False,
             root=ROOT, out_suffix=""):
    """Fit every depth of one lake and write observations/<lake>/sigma_rep<suffix>.json.

    With `n_ref` the written values are the error of the mean of n_ref stations and the table
    declares 'n_ref', so the runtime scales UP by n_ref/N when fewer report. Without it they are
    per-station and the runtime scales down by 1/N. Same sigma either way -- only the stored
    number's meaning differs.
    """
    lake = cfg["lake"]
    obs_file = obs_file.format(lake=lake) if obs_file else obs_file
    path, frames = load_obs(lake, obs_file, root)
    logger.info(f"{lake}: {os.path.relpath(path, root)}, {len(frames)} depth(s)")

    table = {}
    for depth, wide in sorted(frames.items()):
        entry, by_year, cv, meta = fit_depth(wide)
        if all(np.isnan(v) for v in entry.values()):
            logger.info(f"  {depth:6.1f} m  skipped, under {MIN_HOURS} paired hours")
            continue
        pooled = sigma_rep(wide.std(axis=1, ddof=1).dropna())
        # Written as fitted. With n_ref the runtime reads this as the value for a FULL complement
        # and scales it UP by n_ref/N when stations are missing.
        table[depth_key(depth)] = {
            "all": round(pooled, 4),
            "by_season": {k: round(v, 4) for k, v in entry.items() if not np.isnan(v)},
            # the runtime looks up by month, so every month carries its season's value
            "by_month": {str(m): round(entry[season_of(m)], 4) for m in range(1, 13)
                         if not np.isnan(entry[season_of(m)])},
            "by_year": by_year,
            "cv_pct": cv,
            **meta,
        }
        summary = "  ".join(f"{k} {v:.3f}" for k, v in entry.items() if not np.isnan(v))
        logger.info(f"  {depth:6.1f} m  {summary}   N={meta['mean_n_stations']}  "
                    f"{meta['n_hours']} h  cv {cv}")

    if not table:
        raise ValueError(f"{lake}: nothing fitted -- no depth had {MIN_HOURS} paired hours")

    out = {
        "lake": lake,
        "fitted_on": dt.date.today().isoformat(),
        "source": os.path.relpath(path, root).replace("\\", "/"),
        "estimator": f"rms over hours of the across-station sample std (ddof=1, N>={MIN_STATIONS}), "
                     f"per (depth, season); seasons with < {MIN_HOURS} paired hours omitted",
        "units": "degC",
        "sigma_common": sigma_common,
        "_sigma_common_note": "NOT fitted, and not fittable from this estimator: an error shared by "
                              "every station shifts their mean, not their spread. It is the floor "
                              "under R, and it is added outside the N scaling because an error "
                              "every station shares does not average away.",
        "sigma_rep": table,
    }
    if n_ref:
        out["n_ref"] = n_ref
        out["_n_ref_note"] = (
            f"Values are read as the sigma for a FULL complement of n_ref={n_ref} stations, and the "
            "runtime scales them UP by n_ref/N when fewer report -- so a full hour gets the fitted "
            "value and a thinner hour gets more. Tables WITHOUT this key are per-station and are "
            "scaled DOWN by 1/N instead; that is every single-station lake, unaffected by this.")
    dest = os.path.join(root, "observations", lake, f"sigma_rep{out_suffix}.json")
    if dry_run:
        logger.info(f"  --dry-run: would write {os.path.relpath(dest, root)}")
        return out
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    logger.info(f"  wrote {os.path.relpath(dest, root)}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    ap.add_argument("arg_file")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--lake", default=None)
    grp.add_argument("--lakes", default=None, help="comma-separated, or 'all'")
    ap.add_argument("--obs-file", default=None,
                    help="observation CSV (default observations/<lake>/temperature.csv); "
                         "in batch it must contain {lake}")
    ap.add_argument("--suffix", default="",
                    help="appended to the output name, to write beside the table in use")
    ap.add_argument("--sigma-common", type=float, default=DEFAULT_SIGMA_COMMON,
                    help=f"recorded in the table as metadata (default {DEFAULT_SIGMA_COMMON}); "
                         "the runtime reads its own value from the run config")
    ap.add_argument("--n-ref", type=int, default=None,
                    help="write the error of the mean of N_REF stations instead of one station's, "
                         "and declare it so the runtime scales up by n_ref/N. Omit for the "
                         "per-station form the single-station lakes use")
    ap.add_argument("--dry-run", action="store_true", help="print the table, write nothing")
    cli = ap.parse_args()

    if cli.n_ref is not None and cli.n_ref < 1:
        raise ValueError(f"--n-ref must be >= 1, got {cli.n_ref}")

    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")
    arg_file = cli.arg_file if os.path.isfile(cli.arg_file) else os.path.join(ROOT, cli.arg_file)
    with open(arg_file) as f:
        raw = json.load(f)

    if cli.lakes and cli.lakes.strip() == "all":
        lakes = list(raw.get("lakes", {}))
    elif cli.lakes:
        lakes = [s.strip() for s in cli.lakes.split(",") if s.strip()]
    else:
        lakes = [cli.lake]

    if len(lakes) > 1 and cli.obs_file and "{lake}" not in cli.obs_file:
        raise ValueError("--obs-file names a single file: in batch mode it must contain {lake}")

    for lake in lakes:
        fit_lake(merge_lake_args(raw, lake=lake), obs_file=cli.obs_file,
                 sigma_common=cli.sigma_common, n_ref=cli.n_ref, dry_run=cli.dry_run,
                 out_suffix=cli.suffix)


if __name__ == "__main__":
    main()

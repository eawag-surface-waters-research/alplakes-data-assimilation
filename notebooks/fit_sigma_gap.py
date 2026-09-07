"""Fit the observation-error model from the free-run-minus-obs gap, and emit all three forms.

One recipe, three steps, three artefacts -- and the only difference between the run configs that
consume them is where you stop:

    1  gap(t,z) = ref(t,z) - obs(t,z)   at the ~365 assimilated instants, split by season
    2  per (depth, season): std about THAT season's own mean, ddof=0
                            -> the per-depth CSV, and observations/<lake>/sigma_rep_gap.json
    3  median over depths   -> the scalar sigma_obs a config carries

    step 3 only          -> args/experiments/simplified_final.json      (scalar sigma_obs)
    steps 1-2 (no 3)     -> args/experiments/gap_nocommon.json          (sigma_rep table)

Because both come out of one pass, the median over depths of the written table returns the written
scalar by construction. That check used to be a coincidence across two scripts; here it is an
invariant, and the run prints both so it stays visible.

WHERE THE SCALAR LIVES. In two places, deliberately: the --scalars-out JSON (every lake in one
blob, the paste source for a multi-lake config) and each table's own 'sigma_obs' key (one file per
lake, so a table can be checked against the config that quotes it without opening a second file).
Both come from the same `scalars()` call. The --annual rung writes no table, so its scalar has the
JSON as its only home.

TWO RUNGS, and a lake needs the right one. The default splits the seasons and removes each one's
own mean, so "bias-free" means free of BOTH seasonal offsets -> the {"mixed": .., "stratified": ..}
form. `--annual` keeps one block and one bias for the whole year, so the seasonal swing stays
inside the std -> a plain float. Six of the seven 2025 lakes take the seasonal rung; aegeri takes
the annual one, because season-keying it moved pooled NIS to 1.70 against 0.94. Only the seasonal
rung can emit tables -- an annual fit has no per-season column to write.

UPPER BOUND. The gap contains model error as well as observation error, so these are the top of
the bracket, not an estimate of the sensor; the across-station fit in fit_std_obs.py is the bottom
of it. Run them with sigma_rep_scale = 1.0, and let fit_scale_by_depth.py close the gap from a
run's own innovations. The deep entries collapse to 0.01-0.09 degC while the bias there does not 

REFIT WHENEVER THE SERIES CHANGES. These are a property of the thinned, filtered series they were
fitted on, not of the lake. A value carried across a change in the filter or the thinning is wrong
in a direction nothing in the log reveals.

    python notebooks/fit_sigma_gap.py --daily                  # seasonal: CSV + scalars + tables
    python notebooks/fit_sigma_gap.py --daily --annual         # annual rung, scalars only
    python notebooks/fit_sigma_gap.py --daily --no-tables      # steps 1 and 3 only
    python notebooks/fit_sigma_gap.py --daily --dry-run
"""
import os
import sys
import json
import glob
import logging
import argparse
import datetime as dt

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from assimilator.functions import ROOT, load_obs, merge_lake_args, season_of   # noqa: E402
from assimilator.sigma_rep import (DEFAULT_SIGMA_COMMON, depth_key,           # noqa: E402
                                   has_season_keyed_sigma_obs)
import visualize as viz                                                        # noqa: E402

logger = logging.getLogger("sigma_gap")

MIN_HOURS = 24                  # a depth must be present this often in EVERY block used
MAX_DEPTH_MISMATCH = 0.5        # obs further than this from any model cell is dropped
STRAT_MONTHS = range(5, 11)     # May-Oct, the window the rest of the project calls stratified
COLUMN = {"mixed": "std_mixed", "stratified": "std_strat"}
SOURCE_CMD = "notebooks/fit_sigma_gap.py --daily"


# ============================================================================ steps 1-2: fit ====

def regime_key(index):
    """Two seasons, not four: stratified May-Oct, mixed Nov-Apr."""
    return pd.Series(["strat" if t.month in STRAT_MONTHS else "mixed" for t in index], index=index)


def discover_lakes():
    """Lakes that have both a free run and a filtered hourly observation file."""
    obs_dir = os.path.join(ROOT, "observations")
    return sorted(l for l in os.listdir(obs_dir)
                  if os.path.isfile(os.path.join(obs_dir, l, "temperature_filtered.csv"))
                  and os.path.isfile(os.path.join(ROOT, "inputs", l, "ref", "T_out.dat")))


def obs_path(lake, daily):
    """The filtered hourly record, or the one-instant-per-day file actually assimilated.

    generate_assimilation_series.py writes temperature_filtered_<tag>_1d.csv, where the tag is
    h<HH> (the obs nearest that hour) or w<LO><HI> (the FIRST obs inside [LO, HI) that day, for a
    lake whose sampling phase drifts). Reading those files rather than re-deriving the rule keeps
    this fit on exactly the instants a run assimilates."""
    if not daily:
        return os.path.join(ROOT, "observations", lake, "temperature_filtered.csv")
    hits = sorted(glob.glob(os.path.join(ROOT, "observations", lake,
                                         "temperature_filtered_*_1d.csv")))
    if len(hits) != 1:
        raise ValueError(f"expected one thinned daily file, found {len(hits)} "
                         f"({', '.join(os.path.basename(h) for h in hits) or 'none'}) "
                         f"— run notebooks/generate_assimilation_series.py --lake {lake}")
    return hits[0]


def load_ref(lake, ref_path):
    with open(os.path.join(ROOT, "inputs", lake, "Settings.par")) as f:
        ref_year = json.load(f)["Simulation"]["Reference year"]
    ref = viz.load_traj(ref_path, pd.Timestamp(f"{ref_year}-01-01", tz="UTC"))
    if ref is None:
        raise FileNotFoundError(f"free run not found: {ref_path}")
    return ref


def gap_frame(lake, obs_csv, ref_path, year=None, max_depth=None):
    """STEP 1: wide frame of ref - obs, one column per observed depth, on the shared hours."""
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

    gap, far = {}, []
    for d in wide.columns:
        j = int(np.argmin(np.abs(model_depths - d)))
        if abs(model_depths[j] - d) > MAX_DEPTH_MISMATCH:          # below the model's deepest cell
            far.append(d)
            continue
        gap[d] = ref.iloc[:, j].reindex(wide.index) - wide[d]      # model - obs
    if far:
        logger.info(f"  {lake}: {len(far)} obs depths have no free-run depth within "
                    f"{MAX_DEPTH_MISMATCH:g} m ({min(far):g}..{max(far):g} m) - dropped")
    if not gap:
        raise ValueError("no observed depth matches the free run's output depths")
    return pd.DataFrame(gap).sort_index(axis=1)


def sigma_candidates(gap, annual=False):
    """STEP 2: per depth, the split of the gap into RMS, bias and bias-free std.

    Season-balanced by default: each season is weighted 1/2 whatever its day count, so a lake with
    a hole in one of them (June at greifensee) is not summarised mostly by the other, and the bias
    removed is each season's own mean. `annual=True` drops the split: one block, one bias per depth
    for the whole year, so the seasonal swing stays inside the std.

    Either way the three columns satisfy rms^2 = std^2 + bias^2 exactly, per depth."""
    reg = regime_key(gap.index)
    names = ("year",) if annual else ("mixed", "strat")
    parts = {}
    for s in names:
        sub = gap if annual else gap[reg == s]
        keep = sub.count() >= MIN_HOURS              # a depth must be present in EVERY block used
        parts[s] = {"rms": np.sqrt((sub ** 2).mean()).where(keep),
                    "std": sub.std(ddof=0).where(keep),
                    "bias": sub.mean().where(keep).abs()}
    out = pd.DataFrame({k: np.sqrt(sum(parts[s][k] ** 2 for s in names) / len(names))
                        for k in ("rms", "std", "bias")}).dropna()
    for s in names:
        out[f"rms_{s}"] = parts[s]["rms"]
        # The per-season std as well as the per-season rms: a season-keyed sigma_obs has to be
        # spelled the same way as the annual one it replaces, and that recipe removes the bias.
        out[f"std_{s}"] = parts[s]["std"]
    return out


def rows(lake, c, annual):
    """One summary row per depth, rounded to 3 decimals — the CSV's precision.

    The table and the scalars are both built from THESE rounded values rather than from `c`, which
    is what keeps this single script bit-identical to the two it replaces: the old chain wrote the
    CSV, then read it back to build the table and to take the median.
    """
    tag = "" if annual else "_bal"
    return [dict({"lake": lake, "depth": d, f"rms{tag}": round(float(c.loc[d, "rms"]), 3),
                  f"std{tag}": round(float(c.loc[d, "std"]), 3),
                  f"bias{tag}": round(float(c.loc[d, "bias"]), 3)},
                 **{k: round(float(c.loc[d, k]), 3) for k in c.columns
                    if k.startswith(("rms_", "std_"))})
            for d in c.index]


# ================================================================== step 3: collapse to scalar ==

def scalars(g, annual):
    """STEP 3: the median over depths — the number the run config carries.

    Annual gives one float; seasonal gives the {"mixed": .., "stratified": ..} pair, keyed the way
    resolve_scalar_sigma_obs reads it. The std column, not the rms: the recipe removes the bias.
    Taken over the ROUNDED per-depth values (see `rows`), which is the order the committed scalars
    were produced in — before the rounding, hallwil, murten and aegeri each move by one in the last
    digit and no longer match what the table's own median returns.
    """
    def med(col):
        return round(float(np.median(g[col].to_numpy(dtype=float))), 3)
    if annual:
        return med("std")
    return {"mixed": med("std_mixed"), "stratified": med("std_strat")}


# ===================================================================== step 2 output: the table ==

def build_table(lake, g, sigma_common, summary_rel, obs_file, scalar):
    """One lake's rows -> a sigma_rep table in the runtime's format.

    `obs_file` goes into 'source' so load_sigma_rep's filtered-vs-raw guard sees the series these
    numbers were actually fitted on. Naming only the CSV makes it warn on every filtered lake.

    `scalar` is step 3's median over depths, recorded here as 'sigma_obs'. The runtime does NOT
    read it -- a scalar config carries its own copy -- but a table that only ASSERTS the relation
    in prose cannot be checked against the config, which is how upperlugano's mixed value came to
    read 0.300 against a fitted 0.299. Written beside the table it is the median of, one file per
    lake answers both "what is the table" and "what scalar does it collapse to".
    """
    table = {}
    for depth, row in g.set_index("depth").iterrows():
        by_season = {s: round(float(row[c]), 4) for s, c in COLUMN.items()
                     if pd.notna(row.get(c))}
        if not by_season:
            continue
        table[depth_key(depth)] = {
            "all": round(float(row["std_bal"]), 4) if pd.notna(row.get("std_bal")) else None,
            "by_season": by_season,
            # the runtime looks up by month, so every month carries its season's value
            "by_month": {str(m): by_season[season_of(m)] for m in range(1, 13)
                         if season_of(m) in by_season},
        }
    if not table:
        raise ValueError(f"{lake}: no usable rows")

    return {
        "lake": lake,
        "fitted_on": dt.date.today().isoformat(),
        "fitted_by": SOURCE_CMD,
        "source": f"{obs_file} vs inputs/{lake}/ref/T_out.dat (free run), via {summary_rel}",
        "estimator": "std(ref - obs) about each season's own mean, ddof=0, per (depth, season), "
                     "at the ~365 assimilated instants. Steps 1-2 of the gap recipe, without "
                     "step 3's median over depths.",
        "units": "degC",
        "resolution": "seasonal",
        "sigma_common": sigma_common,
        "_upper_bound_note": "The gap contains model error as well as observation error, so this is "
                             "an UPPER bound. Run with sigma_rep_scale = 1.0: the multipliers "
                             "fitted for the across-station tables lift a LOWER bound.",
        # Step 3, carried here so the table and the scalar cannot drift apart unnoticed. The
        # runtime ignores this key: resolve_scalar_sigma_obs reads the RUN CONFIG's sigma_obs, and
        # a config using this table does not use a scalar at all.
        "sigma_obs": scalar,
        "_step3_note": "'sigma_obs' above is the median over depths of by_season -- step 3 of the "
                       "same fit, and the value a scalar config (simplified_final) carries for "
                       "this lake. Recorded for checking, never read at run time. The annual rung "
                       "emits no table, so an annual scalar lives only in the --scalars-out JSON.",
        "sigma_rep": table,
    }


# ============================================================== the config-vs-fit check ==========

def check_config(arg_file, fitted, annual):
    """Compare a run config's sigma_obs against the scalars just fitted. Returns an exit code.

    The one step of the chain nothing else guards: `scalars()` computes the median and a person
    retypes it into args/experiments/<name>.json. That is where upperlugano's mixed value became
    0.300 against a fitted 0.299 -- a hand-rounded 0.2995, invisible to every other check because
    the config is the only place the run reads.

    A config on the OTHER rung is reported, not failed: aegeri deliberately carries the annual
    scalar while the seasonal fit runs, and flagging that as an error would train the eye to ignore
    the output. Only a same-rung disagreement is a mismatch.
    """
    path = arg_file if os.path.isfile(arg_file) else os.path.join(ROOT, arg_file)
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    rung = "annual" if annual else "seasonal"
    logger.info(f"\ncheck: {os.path.relpath(path, ROOT)} against the {rung} fit above")

    bad = 0
    for lake in raw.get("lakes") or {}:
        have = merge_lake_args(raw, lake=lake).get("sigma_obs")
        want = fitted.get(lake)
        if want is None:
            logger.info(f"  {lake:<12} not fitted in this run - skipped")
            continue
        # has_season_keyed_sigma_obs, NOT np.ndim: np.ndim({...}) is 0, so a dict takes the scalar
        # branch and every lake reads as annual. That is the same confusion that makes enkf_update
        # raise on a season dict (porting_simplified.md 1.1) -- the predicate has to be structural.
        if has_season_keyed_sigma_obs({"sigma_obs": have}) == annual:
            logger.info(f"  {lake:<12} config is on the OTHER rung ({have}) - not compared")
            continue
        if have == want:
            logger.info(f"  {lake:<12} ok   {want}")
            continue
        bad += 1
        logger.warning(f"  {lake:<12} MISMATCH  config {have}  fitted {want}")
    logger.info(f"check: {bad} mismatch(es)"
                + ("" if bad else " - every compared lake matches its fit"))
    return 1 if bad else 0


# ==================================================================================== driver ====

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lakes", nargs="*", default=None,
                    help="Default: every lake with both a free run and filtered hourly obs")
    ap.add_argument("--year", type=int, default=None, help="Restrict to one calendar year")
    ap.add_argument("--max-depth", type=float, default=None, help="Ignore obs below this depth (m)")
    ap.add_argument("--daily", action="store_true",
                    help="Use the thinned one-instant-per-day file a run actually assimilates "
                         "instead of every hour. The rung the committed configs were fitted on")
    ap.add_argument("--annual", action="store_true",
                    help="No seasonal split - one bias per depth for the whole year. The aegeri "
                         "rung; emits no tables")
    ap.add_argument("--no-tables", action="store_true",
                    help="Skip observations/<lake>/sigma_rep_gap.json, keep the CSV and scalars")
    ap.add_argument("--suffix", default="_gap", help="appended to the table name")
    ap.add_argument("--sigma-common", type=float, default=DEFAULT_SIGMA_COMMON,
                    help=f"recorded as table metadata (default {DEFAULT_SIGMA_COMMON}); the runtime "
                         "reads its own value from the run config")
    ap.add_argument("--arg-file", default="args/experiments/simplified_final.json",
                    help="config the obs_file per lake is read from, for the table's 'source' field")
    ap.add_argument("--out", default=None, help="per-depth CSV (default under local/)")
    ap.add_argument("--scalars-out", default=None,
                    help="step 3 as JSON, ready to paste into a run config (default beside --out)")
    ap.add_argument("--dry-run", action="store_true", help="print everything, write nothing")
    ap.add_argument("--check-config", default=None, metavar="ARG_FILE",
                    help="refit, then compare each lake's sigma_obs in ARG_FILE against the fit "
                         "and exit non-zero on a mismatch. Implies --dry-run: it is a check, so it "
                         "must not leave new artefacts behind for the next one to agree with")
    cli = ap.parse_args()
    cli.dry_run = cli.dry_run or bool(cli.check_config)

    stem = (f"sigma_obs_candidates{'_annual' if cli.annual else ''}"
            f"{'_daily' if cli.daily else ''}")
    cli.out = cli.out or os.path.join(ROOT, "local", f"{stem}_summary.csv")
    cli.scalars_out = cli.scalars_out or os.path.splitext(cli.out)[0].replace(
        "_summary", "") + "_scalars.json"
    write_tables = not cli.no_tables and not cli.annual

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    lakes = cli.lakes or discover_lakes()
    logger.info(f"lakes: {', '.join(lakes)}")
    if cli.annual and not cli.no_tables:
        logger.info("--annual: no per-season column to write, so no tables (scalars only)")

    raw = None
    if write_tables:
        arg_file = (cli.arg_file if os.path.isfile(cli.arg_file)
                    else os.path.join(ROOT, cli.arg_file))
        with open(arg_file, encoding="utf-8") as f:
            raw = json.load(f)

    summary, out_scalars, fitted = [], {}, {}
    for lake in lakes:
        ref_path = os.path.join(ROOT, "inputs", lake, "ref", "T_out.dat")
        try:
            src = obs_path(lake, cli.daily)
            gap = gap_frame(lake, src, ref_path, cli.year, cli.max_depth)
            c = sigma_candidates(gap, cli.annual)
            if c.empty:
                raise ValueError("no depth has enough days"
                                 + ("" if cli.annual else " in both seasons"))
        except (ValueError, FileNotFoundError) as exc:
            logger.warning(f"  {lake}: skipped - {exc}")
            continue
        r = rows(lake, c, cli.annual)
        summary += r
        # Step 3 reads the ROUNDED rows, not `c` — see `rows`.
        g = pd.DataFrame(r)
        out_scalars[lake] = scalars(g, cli.annual)
        fitted[lake] = (src, g)
        shown = (f"{out_scalars[lake]:.3f}" if cli.annual
                 else "  ".join(f"{k} {v:.3f}" for k, v in out_scalars[lake].items()))
        logger.info(f"  {lake:<12} {os.path.basename(src):<38} "
                    f"{len(c):3d} depths  ->  {shown}")
    if not summary:
        raise SystemExit("no lake produced a gap series")

    summary_rel = os.path.relpath(cli.out, ROOT).replace("\\", "/")
    if not cli.dry_run:
        os.makedirs(os.path.dirname(cli.out), exist_ok=True)
        pd.DataFrame(summary).to_csv(cli.out, index=False)
        with open(cli.scalars_out, "w", encoding="utf-8") as f:
            json.dump(out_scalars, f, indent=2)
            f.write("\n")
    verb = "would write" if cli.dry_run else "wrote"
    logger.info(f"{verb} {summary_rel} ({len(summary)} depth rows, steps 1-2)")
    logger.info(f"{verb} {os.path.relpath(cli.scalars_out, ROOT)} (step 3)")

    if cli.check_config:
        raise SystemExit(check_config(cli.check_config, out_scalars, cli.annual))
    if not write_tables:
        return
    for lake, (src, g) in fitted.items():
        obs_file = merge_lake_args(raw, lake=lake).get(
            "obs_file", os.path.relpath(src, ROOT).replace("\\", "/"))
        out = build_table(lake, g, cli.sigma_common, summary_rel, obs_file, out_scalars[lake])
        # The invariant: the median of what the table carries IS the scalar written above. Both
        # sides are rounded to the CSV's 3 decimals first — an even depth count makes the median an
        # average of two entries, so an unrounded comparison shows a spurious 0.0005 half the time.
        med = {s: round(float(np.median([e["by_season"][s] for e in out["sigma_rep"].values()
                                         if s in e["by_season"]])), 3) for s in COLUMN}
        drift = {s: round(med[s] - out_scalars[lake][s], 4) for s in COLUMN}
        dest = os.path.join(ROOT, "observations", lake, f"sigma_rep{cli.suffix}.json")
        if not cli.dry_run:
            with open(dest, "w", encoding="utf-8") as f:
                json.dump(out, f, indent=2)
                f.write("\n")
        logger.info(f"{verb} {os.path.relpath(dest, ROOT)}  "
                    f"({len(out['sigma_rep'])} depths, "
                    + "  ".join(f"median({s}) {v:.3f}" for s, v in med.items())
                    + ("" if not any(drift.values()) else f"  DRIFT {drift}") + ")")


if __name__ == "__main__":
    main()

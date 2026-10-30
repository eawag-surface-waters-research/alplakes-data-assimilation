"""Standalone command line for the assimilator package. A host repository (e.g. operational-simstrat)
calls the same functions directly; this CLI is for running on plain folders and files.

  run         assimilate observations into a model setup folder
  summarize   rebuild a finished run's summary (.csv + skill .json) from its member outputs
  preprocess  raw sub-daily observations -> the series a run assimilates (filter, grid, thin)
  fit         offline fits: filter window, analysis hour, observation error, perturbation persistence

Examples:

    python src/assimilate.py run --sim-dir /data/upperlugano/inputs --obs obs.csv \\
        --perturbations upperlugano.json --start 2025-01-01 --end 2025-12-31 --out /data/da_run
    python src/assimilate.py run examples/enkf.json                     # same, from a config file
    python src/assimilate.py preprocess raw.csv -o obs.csv --w-seiche 27.3 --hour 6
    python src/assimilate.py fit sigma --free-run T_out.dat --obs obs.csv
"""

import os
import sys
import json
import logging
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # put src/ on the path

logger = logging.getLogger("assimilate")


def _setup_logging(log_file=None, verbose=False):
    # Per-window lines are logged with extra={"file_only": True}: always in the log file, never on
    # the console (which shows the progress bar, or just the milestones when it is off).
    console = logging.StreamHandler()
    console.addFilter(lambda r: not getattr(r, "file_only", False))
    handlers = [console]
    if log_file:
        os.makedirs(os.path.dirname(os.path.abspath(log_file)), exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s | %(levelname)-8s | %(name)-16s | %(message)s",
                        datefmt="%H:%M:%S", handlers=handlers)
    if log_file:
        logger.info(f"log file -> {log_file}")


def _read_csv_obs(path):
    import pandas as pd
    df = pd.read_csv(path)
    df["time"] = pd.to_datetime(df["time"], utc=True, format="ISO8601")
    return df


def _load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------------------- run --

RUN_FLAGS = {   # CLI flag dest -> config key
    "sim_dir": "model_inputs_path", "obs": "obs_file", "out": "ensemble_base",
    "start": "start_date", "end": "end_date", "lake": "lake", "engine": "engine",
    "algorithm": "algorithm", "n_members": "n_members", "model": "model",
    "max_workers": "max_workers", "rng_seed": "rng_seed", "openda_bin": "openda_bin",
}


def cmd_run(cli):
    from assimilator.pipeline import run
    from assimilator.config import build_config
    cfg = _load_json(cli.config) if cli.config else {}
    for dest, key in RUN_FLAGS.items():
        value = getattr(cli, dest)
        if value is not None:
            cfg[key] = value
    if cli.perturbations:
        cfg.pop("perturbations", None)
        cfg["perturbations_file"] = cli.perturbations
    if not cli.config:   # flags only: sensible defaults for a quick standalone run
        cfg.setdefault("algorithm", "EnKF")
        cfg.setdefault("n_members", 20)
        if cfg.get("model_inputs_path"):
            cfg.setdefault("lake", os.path.basename(os.path.normpath(cfg["model_inputs_path"])))
    if cli.reset:
        cfg["reset"] = True
    cfg["progress"] = (False if cli.no_progress
                       else cfg.get("progress", sys.stderr.isatty()))
    cfg = build_config(cfg)

    log_file = cli.log_file or os.path.join(cfg["ensemble_base"], "logs",
                                            f"pipeline_{datetime.now():%Y%m%d_%H%M%S}.log")
    _setup_logging(log_file, cli.verbose)
    result = run(cfg, force_copy=cli.force_copy, skip_oda=cli.skip_oda)
    logger.info(f"mean trajectory -> {result['mean_trajectory']}")
    if result["output_grid"]:
        logger.info(f"output grid incl. observation depths -> {result['output_grid']}")


def cmd_summarize(cli):
    from assimilator.summarize import summarize_from_config
    _setup_logging(verbose=cli.verbose)
    summarize_from_config(_load_json(cli.config))


# --------------------------------------------------------------------------------- preprocess --

def _floats(s):
    return [float(x) for x in s.split(",")] if s else None


def _window(s):
    if s in (None, "all", "any"):
        return None
    lo, hi = (int(x) for x in s.split("-"))
    return [lo, hi]


def cmd_preprocess(cli):
    from assimilator.observations import preprocess
    _setup_logging(verbose=cli.verbose)
    cfg = _load_json(cli.config) if cli.config else {}
    if cli.w_seiche is not None:
        cfg["filter"] = {"w_seiche_h": cli.w_seiche, "thermo_depth_min": cli.thermo_depth_min}
    if cli.no_filter:
        cfg.pop("filter", None)
    if cli.hour is not None or cli.hour_window:
        cfg["schedule"] = ({"hour_window": _window(cli.hour_window)} if cli.hour_window
                           else {"hour": cli.hour, "max_offset_h": cli.max_offset})
    if cli.depths:
        cfg["depths"] = _floats(cli.depths)
    out = preprocess(_read_csv_obs(cli.raw), cfg)
    out.to_csv(cli.output, index=False)
    logger.info(f"{len(out)} rows, {out['time'].nunique()} instants, {out['depth'].nunique()} depths "
                f"-> {cli.output}")


# ---------------------------------------------------------------------------------------- fit --

def cmd_fit(cli):
    from assimilator import fitting
    _setup_logging(verbose=cli.verbose)
    if cli.what == "filter":
        from assimilator.fitting.filter import derive_filter
        result = derive_filter(_read_csv_obs(cli.raw), cli.length_km, cli.mean_depth)
    elif cli.what == "hour":
        from assimilator.fitting.hour import select_hour
        result = select_hour(_read_csv_obs(cli.raw),
                             fitting.read_free_run(cli.free_run, cli.model),
                             window=_window(cli.window), criterion=cli.criterion)
    elif cli.what == "sigma":
        from assimilator.fitting.sigma import fit_sigma
        result = fit_sigma(_read_csv_obs(cli.obs), fitting.read_free_run(cli.free_run, cli.model),
                           year=cli.year, max_depth=cli.max_depth, seasonal=not cli.annual)
    else:   # perturbation persistence (upper bound) from the control forcing alone
        import pandas as pd
        from assimilator.models import get_model
        from assimilator.fitting import perturbation as P
        base = P.migrate(_load_json(cli.existing)) if cli.existing else None
        start = pd.Timestamp(cli.fit_start, tz="UTC")
        end   = pd.Timestamp(cli.fit_end, tz="UTC") + pd.Timedelta(hours=23, minutes=59)
        ctrl  = get_model(cli.model).read_forcing(cli.sim_dir)
        ctrl  = ctrl[(ctrl["time"] >= start) & (ctrl["time"] <= end)].reset_index(drop=True)
        upper = P.fit_upper(ctrl)
        if base is None:
            result = {"upper": upper,
                      "_note": "amplitude needs the reanalysis residual (fit_amplitude); pass "
                               "--existing <calibration.json> to combine with a stored one"}
        else:
            base["persistence"]["upper"] = upper
            result = P.ordered(P.select(base, cli.bound))
    text = json.dumps(result, indent=2)
    if cli.output:
        with open(cli.output, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        logger.info(f"wrote {cli.output}")
    else:
        print(text)


# --------------------------------------------------------------------------------------- main --

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog=__doc__.split("Examples:")[1])
    ap.add_argument("-v", "--verbose", action="store_true")
    sub = ap.add_subparsers(dest="command", required=True)

    r = sub.add_parser("run", help="assimilate observations into a model setup folder")
    r.add_argument("config", nargs="?", help="run config JSON (flags below override its keys)")
    r.add_argument("--sim-dir", help="model setup folder with a dated warm start (read only)")
    r.add_argument("--obs", help="observation CSV (time, depth, value)")
    r.add_argument("--perturbations", help="AR(1) calibration JSON ({variables: {U,V,GLOB}})")
    r.add_argument("--start", help="first instant assimilated from, ISO date")
    r.add_argument("--end", help="end of the run, ISO date")
    r.add_argument("--out", help="output folder (members, outputs, state)")
    r.add_argument("--lake", help="run label (default: the --sim-dir folder name)")
    r.add_argument("--engine", choices=["python", "openda"])
    r.add_argument("--algorithm", help="EnKF / PF (python); EnKF / DEnKF / EnSR / PF (openda)")
    r.add_argument("--n-members", type=int)
    r.add_argument("--model", help="forward model (default simstrat)")
    r.add_argument("--rng-seed", type=int)
    r.add_argument("--max-workers", type=int, help="cap on members run concurrently")
    r.add_argument("--openda-bin", help="OpenDA 3.4.0 bin/ folder (engine openda)")
    r.add_argument("--reset", action="store_true", help="start from scratch in --out")
    r.add_argument("--force-copy", action="store_true", help="re-copy the setup into the members")
    r.add_argument("--skip-oda", action="store_true", help="openda: prepare but do not run OpenDA")
    r.add_argument("--no-progress", action="store_true")
    r.add_argument("--log-file", help="default: <out>/logs/pipeline_<timestamp>.log")
    r.set_defaults(func=cmd_run)

    s = sub.add_parser("summarize", help="rebuild a run's summary from its member outputs")
    s.add_argument("config", help="the run's config JSON")
    s.set_defaults(func=cmd_summarize)

    p = sub.add_parser("preprocess", help="raw observations -> assimilated series")
    p.add_argument("raw", help="raw sub-daily CSV (time, depth, value)")
    p.add_argument("-o", "--output", required=True)
    p.add_argument("--config", help="JSON with an observation-processing block (filter/schedule/depths)")
    p.add_argument("--w-seiche", type=float, help="filter window, hours (enables the filter)")
    p.add_argument("--thermo-depth-min", type=float, default=4.0)
    p.add_argument("--no-filter", action="store_true")
    p.add_argument("--hour", type=int, help="thin to one profile per day nearest this UTC hour")
    p.add_argument("--hour-window", help="LO-HI: thin to the first profile in [LO, HI) instead")
    p.add_argument("--max-offset", type=float, default=1.5, help="hours, with --hour")
    p.add_argument("--depths", help="target depth grid, comma separated (nearest within 0.5 m)")
    p.set_defaults(func=cmd_preprocess)

    f = sub.add_parser("fit", help="offline fits")
    f.add_argument("what", choices=["filter", "hour", "sigma", "perturbation"])
    f.add_argument("--raw", help="filter/hour: raw sub-daily CSV")
    f.add_argument("--obs", help="sigma: the ASSIMILATED series CSV")
    f.add_argument("--free-run", help="hour/sigma: free-run output trajectory (e.g. T_out.dat)")
    f.add_argument("--model", default=None, help="forward model (default simstrat)")
    f.add_argument("--length-km", type=float, help="filter: basin length")
    f.add_argument("--mean-depth", type=float, help="filter: mean lake depth (m)")
    f.add_argument("--window", default="0-6", help="hour: candidate hours LO-HI, or 'all'")
    f.add_argument("--criterion", default="balanced", choices=["balanced", "min-bias", "annual"])
    f.add_argument("--year", type=int, help="sigma: restrict to one year")
    f.add_argument("--max-depth", type=float, help="sigma: ignore obs below this depth")
    f.add_argument("--annual", action="store_true", help="sigma: one annual value, no season split")
    f.add_argument("--sim-dir", help="perturbation: setup folder with the control forcing")
    f.add_argument("--fit-start", default=f"{datetime.now().year - 1}-01-01")
    f.add_argument("--fit-end", default=f"{datetime.now().year - 1}-12-31")
    f.add_argument("--existing", help="perturbation: stored calibration to add the upper bound to")
    f.add_argument("--bound", default="envelope", choices=["lower", "upper", "envelope"])
    f.add_argument("-o", "--output", help="write the JSON here (default: print)")
    f.set_defaults(func=cmd_fit)

    cli = ap.parse_args(argv)
    cli.func(cli)


if __name__ == "__main__":
    main()

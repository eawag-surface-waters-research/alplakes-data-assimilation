"""Prepare and run a model under OpenDA: the OpenDA engine's adapter and driver.

Copies the framework's own inputs into OpenDA's working-dir layout -- the model setup, every
member's perturbed forcing, the warm start -- and builds the stochObserver observation files, so
OpenDA runs on exactly the forcing and starting state the native engines use. Then renders the
OpenDA configuration (config.py), launches oda_run.sh and summarises.

    <model inputs>/*                 -> stochModel/template/   (the model's template inputs)
    <ensemble_base>/ensemble<i>/     -> forcings/              (member i's perturbed forcing)
    the warm start                   -> stochModel/template/   (its starting state + state file)
    the run's "obs_file"             -> stochObserver/T_<depth>m[_<season>]_real.csv

Everything model-specific -- names, layout, time axis, starting state, forcing, outputs, the
container -- comes from the model's OpenDA binding (openda/binding.py), so this module runs any
model that has one. OpenDA's coupling files in the template are never overwritten.

Usage:  python src/assimilator/openda/adapter.py args/run_openda.json
"""

import os
import sys
import glob
import json
import shutil
import logging
import re
import argparse
from datetime import datetime, timedelta

# this file lives at src/assimilator/openda/adapter.py
SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
ROOT    = os.path.dirname(SRC_DIR)                                                       # repo root
sys.path.insert(0, SRC_DIR)

import time
import subprocess

from assimilator.functions import (verify_args, resolve_obs_path, load_obs, to_utc, SEASONS,
                                   make_progress, log_run_header, log_run_footer,
                                   resolve_max_workers, display_path,
                                   start_persistent_container, stop_persistent_container,
                                   sigma_obs_spec, should_season_split,
                                   sigma_obs_for_series, season_of)
from assimilator.summarize import report_summary
from assimilator.models import get_model
from assimilator.config import DEFAULTS as RUN_DEFAULTS
from .config import (FILTERS, render as render_oda, series_specs, format_day,
                     is_season_keyed as is_season_keyed_series, results_filename)
from . import restart as restart_chain

logger = logging.getLogger(__name__)

REQUIRED = ["lake", "n_members", "ensemble_base"]

# Restart chain (openda/restart.py), continuing from a saved ensemble: the marker file tells the
# wrapper every member must have been restored, and the sentinel fills the template state so an
# un-restored member is caught. Every wrapper keeps its own copy of both (binding.py).
RESTART_MARKER   = "RESTART_EXPECTED"
RESTART_SENTINEL = -999.0

# Restart chain only: the binding's long read-only inputs are kept once in <openda_dir>/shared
# instead of in the template OpenDA clones into every work dir, and the template's control forcing
# is cut to the run (the wrapper replaces it for every member but work0). Work dirs are
# <openda_dir>/Results/work<k>, hence two levels up, inside the container's openda_dir mount.
SHARED_DIR = "shared"
SHARED_REL = "../../" + SHARED_DIR


def openda_binding(cfg):
    """The OpenDA binding of the run's model (openda/binding.py)."""
    model   = get_model(cfg.get("model") or RUN_DEFAULTS["model"])
    binding = model.openda_binding()
    if binding is None:
        raise ValueError(f"model '{model.name}' has no OpenDA binding: it cannot run with engine 'openda'")
    return binding


def _copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def _copy_path(src, dst):
    """Copy a file or a directory (dirs replaced wholesale)."""
    if os.path.isdir(src):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy2(src, dst)


def _unchanged(src, dst):
    return (os.path.isfile(dst) and os.path.getsize(src) == os.path.getsize(dst)
            and abs(os.path.getmtime(src) - os.path.getmtime(dst)) < 1e-6)


def _sync_shared(model_inputs, shared_dir, names):
    """Copy the binding's shared inputs `names` into <openda_dir>/shared, leaving unchanged files
    alone. Returns the number copied (0 on a day when the inputs did not change)."""
    copied = 0
    for name in names:
        src = os.path.join(model_inputs, name)
        if not os.path.exists(src):
            continue
        pairs = [(src, os.path.join(shared_dir, name))] if os.path.isfile(src) else \
                [(os.path.join(src, f), os.path.join(shared_dir, name, f))
                 for f in sorted(os.listdir(src)) if os.path.isfile(os.path.join(src, f))]
        for a, b in pairs:
            if not _unchanged(a, b):
                _copy(a, b)
                copied += 1
    return copied


def _build_observations(raw, openda_dir, model_inputs, binding):
    """Build the OpenDA stochObserver obs files from the observation CSV.

    Every observation is assimilated at its own time, binned to the hour exactly as the native
    engines do (functions.load_obs), so the two engines see the same observations at the same
    times, whatever the interval between them -- one a day, several a day, or irregular with gaps.
    Only observations in (start_date, end_date] are kept, the native window rule
    (functions.obs_window_boundaries): one at start_date was the previous run's last analysis.

    Writes stochObserver/T_{depth}m[_{season}]_real.csv, time in model days (the binding's).
    OpenDA analyses at exactly these times (analysisTimes "fromObservationTimes"). Depths are
    dropped if they have data on fewer than `obs_min_days` days or no matching model-output depth
    (binding.output_depths), and returned sorted for the config generator to wire everywhere.
    """
    stoch_dir = os.path.join(openda_dir, "stochObserver")
    obs_csv   = resolve_obs_path(raw)
    if not os.path.isfile(obs_csv):
        raise FileNotFoundError(
            f"observation source not found: {obs_csv} (set 'obs_file')")

    obs   = load_obs(obs_csv).dropna(subset=["value"])
    start = to_utc(raw["start_date"]) if raw.get("start_date") else None
    end   = to_utc(raw["end_date"])   if raw.get("end_date")   else None
    if start is not None:
        obs = obs[obs["time"] > start]
    if end is not None:
        obs = obs[obs["time"] <= end]

    min_days = raw.get("obs_min_days", 1)
    days     = obs.groupby("depth")["time"].agg(lambda t: t.dt.floor("D").nunique())
    depths   = sorted(float(d) for d, n in days.items() if n >= min_days)

    # Keep only depths the model actually outputs: an obs depth with no
    # matching model output depth can't be assimilated, since there is no model prediction to compare it against.
    model_depths = binding.output_depths(model_inputs)
    if model_depths:
        matched = [d for d in depths if any(abs(d - m) <= 1e-6 for m in model_depths)]
        dropped = [d for d in depths if d not in matched]
        if dropped:
            logger.warning(f"[adapter] dropping obs depths with no matching model output depth: "
                           f"{[f'{d:g}' for d in dropped]} m")
        depths = matched
    obs = obs[obs["depth"].isin(depths)]

    # One file per SERIES, and the series set is decided by config.series_specs -- season-split
    # when sigma is season-keyed, else one per depth. The two must agree or OpenDA opens a
    # timeSeries file that does not exist and dies with no member output.
    season_split = bool(raw.get("_season_series"))
    window = f"{raw.get('start_date') or 'start'}..{raw.get('end_date') or 'end'}"
    logger.info(f"[adapter] observations: {display_path(obs_csv)} -> "
                f"stochObserver/T_*_real.csv  ({len(depths)} depths {[f'{d:g}' for d in depths]}, "
                f"{obs['time'].nunique()} observation times in {window}, each at its own hour"
                f"{', season-split' if season_split else ''})")
    os.makedirs(stoch_dir, exist_ok=True)

    series_keys = []            # [(depth, season)] when split, else [(depth, None)] -- what the
    for depth in depths:        # config declares, taken from what is actually on disk.
        rows = obs[obs["depth"] == depth].sort_values("time")
        season = rows["time"].dt.month.map(season_of) if season_split else None
        # The seasons partition the observations, so none is dropped or written twice.
        for s in (SEASONS if season_split else [None]):
            part = rows[season == s] if season_split else rows
            if part.empty:
                continue
            suffix   = f"_{s}" if s else ""
            out_path = os.path.join(stoch_dir, f"T_{depth:g}m{suffix}_real.csv")
            with open(out_path, "w", newline="") as f:
                f.write("time,value\n")
                for t, v in zip(part["time"], part["value"]):
                    f.write(f"{format_day(binding.model_time(t))},{v:.6f}\n")
            series_keys.append((depth, s))
    logger.info(f"  wrote {len(series_keys)} series files -> {display_path(stoch_dir)}")

    # Distinct analysis times = how many forecast/analysis steps OpenDA will run. Drives the
    # progress-bar total, so the bar tracks the real step count for any cadence.
    return depths, int(obs["time"].nunique()), series_keys


def adapt(raw):
    verify_args(raw, REQUIRED)

    lake          = raw["lake"]
    n_members     = raw["n_members"]
    ensemble_base = raw["ensemble_base"]
    model_inputs  = raw["model_inputs_path"]
    openda_dir    = raw["openda_dir"]
    forcings_dir  = os.path.join(openda_dir, "forcings")
    template_dir  = os.path.join(openda_dir, "stochModel", "template")
    binding       = openda_binding(raw)

    # The OpenDA working dir is fully generated: create it and the template skeleton on demand.
    os.makedirs(template_dir, exist_ok=True)   # also creates openda_dir/stochModel

    logger.info(f"[adapter] lake={lake}  model={binding.name}  framework={display_path(ensemble_base)}  "
                f"-> openda={display_path(openda_dir)}")

    # ------------------------------------------------------------------
    # 0. The model's hand-maintained wrapper -> stochModel/
    #    (the only non-generated OpenDA pieces; everything else is rendered/synced)
    # ------------------------------------------------------------------
    for src, rel in binding.wrapper_files():
        _copy(src, os.path.join(openda_dir, "stochModel", rel))
    logger.info(f"[adapter] wrapper -> stochModel/: {[rel for _, rel in binding.wrapper_files()]}")

    # ------------------------------------------------------------------
    # 1. Model inputs -> template/ (not OpenDA's coupling files, and only what the binding counts
    #    as template input: its starting state is placed by step 3)
    # ------------------------------------------------------------------
    if not os.path.isdir(model_inputs):
        raise FileNotFoundError(
            f"model_inputs not found: {model_inputs} (provide the model setup with its dated warm start)")
    logger.info(f"[adapter] model inputs -> template/ (skipping OpenDA-specific {sorted(binding.coupling_files)}):")
    chain = raw.get("_restart_in") is not None     # a run of a restart chain (openda/restart.py)
    for name in sorted(os.listdir(model_inputs)):
        if name in binding.coupling_files or not binding.is_template_input(name):
            continue
        if chain and (name in binding.shared_inputs or name == binding.forcing_file):
            continue
        _copy_path(os.path.join(model_inputs, name),
                   os.path.join(template_dir, name))
    if chain:
        n = _sync_shared(model_inputs, os.path.join(openda_dir, SHARED_DIR), binding.shared_inputs)
        binding.point_config_at_shared(template_dir, SHARED_REL)
        kept, total = binding.cut_forcing(model_inputs, template_dir, binding.model_time(raw["start_date"]),
                                          binding.model_time(raw["end_date"]))
        logger.info(f"[adapter] run inputs: {sorted(binding.shared_inputs)} shared from {SHARED_DIR}/ "
                    f"({n} file(s) copied), {binding.forcing_file} cut to {kept}/{total} rows")

    # ------------------------------------------------------------------
    # 2. Perturbed forcings: member i's forcing -> forcings/<binding.forcing_name(i)>
    # ------------------------------------------------------------------
    logger.info(f"[adapter] forcings (0..{n_members}):")
    for i in range(n_members + 1):                       # 0 = control, 1..N = members
        src = binding.member_forcing(ensemble_base, i)
        if not os.path.isfile(src):
            raise FileNotFoundError(
                f"{src} missing — run the copy + perturbate steps first")
        _copy(src, os.path.join(forcings_dir, binding.forcing_name(i)))
    logger.info(f"  copied {n_members + 1} forcing files -> {display_path(forcings_dir)}")

    # ------------------------------------------------------------------
    # 3. The template's starting state: the warm start (cloned into each work dir), or -- when a
    #    restart chain continues from a saved ensemble (`_restart_in` = a file) -- none: the state
    #    is filled with RESTART_SENTINEL and RESTART_MARKER set, so the wrapper stops if a member
    #    was not restored. "" (the chain's cold start) and None use the warm start.
    # ------------------------------------------------------------------
    restart_in = raw.get("_restart_in")
    marker     = os.path.join(template_dir, RESTART_MARKER)
    if not restart_in and os.path.exists(marker):
        os.remove(marker)       # left over from an earlier run of the chain
    if restart_in:
        if not os.path.isfile(restart_in):
            raise FileNotFoundError(f"OpenDA restart file not found: {restart_in}")
        n = binding.fill_template_sentinel(model_inputs, template_dir, RESTART_SENTINEL)
        with open(marker, "w") as f:
            f.write(f"{restart_in}\n")
        logger.info(f"[adapter] continuing from restart {display_path(restart_in)}: warm start skipped, "
                    f"{binding.state_file} = {n} x {RESTART_SENTINEL:g}, {RESTART_MARKER} set")
    else:
        seeded = binding.seed_template(model_inputs, template_dir)
        if seeded is None:
            logger.warning(f"[adapter] no warm start in {display_path(model_inputs)} — skipping warmup sync")
        else:
            src, n = seeded
            logger.info(f"[adapter] warmup: {os.path.basename(src)} -> template, "
                        f"{binding.state_file} seeded ({n} values)")

    # ------------------------------------------------------------------
    # 4. Observations -> stochObserver/T_{depth}m_real.csv (every observation at its own hour, as
    #    the native engines). Returns the depth list for the config generator to wire everywhere.
    # ------------------------------------------------------------------
    obs_depths, n_analysis, series_keys = _build_observations(raw, openda_dir, model_inputs, binding)

    logger.info("[adapter] done.")
    return obs_depths, n_analysis, series_keys


# ---------------------------------------------------------------------------
# End-to-end OpenDA run (adapt -> render config -> launch oda_run.sh -> summarise)
# ---------------------------------------------------------------------------

# OpenDA writes one "Forecast from <t0> to <t1>" line per analysis step. We can't read these
# from the subprocess stdout pipe: oda_run.sh runs java with `> openda_logfile.txt 2>&1`, so all
# of OpenDA's output is redirected into that file, not the pipe. So we tail the logfile as it
# grows and advance one bar per "Forecast from" line — the file is the only place they appear.
#
# While tailing we also lift a handful of milestone lines into the unified pipeline log so the
# OpenDA run reads like the native one (the rest of the ~300k-line logfile stays only in
# log/openda_logfile.txt). Tiers extracted:
#   A header (once, file_only): version, algorithm className (filter), localization, instance count
#   B per-step (file_only):     the "Forecast from" line (also drives the bar)
#   C failures (WARNING, loud):  the wrapper's "Model finished (exit N)" with N != 0
#   D footer (file_only):       "Application Done"
# Deliberately NOT lifted (would flood): "Written T_*.csv" (~115k) and the per-member-per-step
# wrapper lines ([TIMING]/[STATE]/[RESTART]/container runs, ~7.7k each).
_FORECAST_MARK = b"Forecast from"
_FORECAST_DAYS = re.compile(r"\(([\d.]+)\s*-+>\s*([\d.]+)\)")   # the "(A-->B)" model day span


def _reformat_forecast(line, epoch):
    """OpenDA prints the Forecast line as '... <ts>UTC to <ts>UTC (A-->B)', but the <ts>UTC strings
    render the model day number against the WRONG epoch (the MJD epoch 1858-11-17, not the model's
    `epoch`), giving nonsensical dates. The '(A-->B)' model day numbers ARE correct (the same
    convention summarize/_score use), so convert THOSE to real calendar dates and drop the bogus
    UTC — and the per-step line then shows the same dates as the native engines.
    Falls back to the original line if the day span can't be parsed."""
    m = _FORECAST_DAYS.search(line)
    if not m:
        return line
    t0 = epoch + timedelta(days=float(m.group(1)))
    t1 = epoch + timedelta(days=float(m.group(2)))
    return (f"Forecast {t0:%Y-%m-%d %H:%M} -> {t1:%Y-%m-%d %H:%M} "
            f"(sim-day {m.group(1)}->{m.group(2)})")


def _launch_oda_with_progress(oda_exe, oda_file, openda_dir, env, total, enabled, epoch):
    """Launch oda_run.sh (non-blocking) and drive a progress bar by tailing
    openda_logfile.txt for "Forecast from" lines. `total` is the expected step count
    from the run dates; tqdm tolerates overflow if OpenDA's half-day spin-up/tail
    windows nudge the count past it. Raises CalledProcessError on a non-zero exit
    (mirroring the old subprocess.run(check=True))."""
    logfile = os.path.join(openda_dir, "openda_logfile.txt")
    if os.path.exists(logfile):
        os.remove(logfile)            # drop a stale logfile from a prior aborted run

    proc = subprocess.Popen([oda_exe, oda_file], cwd=openda_dir, env=env)
    bar  = make_progress(total, enabled, desc="OpenDA")
    pos       = 0                     # byte offset already consumed from the logfile
    instances = 0                     # count of "Creating model instance" (members initialized)
    pending   = None                  # the current step's Forecast line, awaiting its member tally
    step_ok = step_fail = 0           # model exits seen since `pending` (this step's instances)

    # The instances run AFTER their step's Forecast line and BEFORE the next one, so a step's
    # member tally is only complete when the next Forecast (or "Application Done") arrives. We hold
    # the Forecast line in `pending` and emit it annotated with instances_ok once the tally closes —
    # the OpenDA counterpart to the native per-step "n_updated" (symmetry-2). Per-step timing is
    # deliberately NOT parsed: the wrapper's "[TIMING]" lines are per-member compute time, not step
    # wall-time (members may overlap), so summing/maxing them would mislead; total time is in the
    # footer instead.
    def _flush_pending():
        nonlocal pending, step_ok, step_fail
        if pending is None:
            return
        n = step_ok + step_fail
        tally = f"instances={step_ok}" + (f"/{n} ({step_fail} failed)" if step_fail else "")
        logger.info(f"{pending}  {tally}", extra={"file_only": True})   # per-step record -> logs/
        pending, step_ok, step_fail = None, 0, 0

    try:
        while True:
            done = proc.poll() is not None
            if os.path.exists(logfile):
                with open(logfile, "rb") as fh:
                    fh.seek(pos)
                    data = fh.read()
                nl = data.rfind(b"\n")    # process only up to the last complete line so a
                if nl != -1:              # marker can't be split across two reads
                    complete = data[:nl + 1]
                    pos     += len(complete)
                    for raw_line in complete.split(b"\n"):
                        if _FORECAST_MARK in raw_line:
                            bar.update(1)
                            _flush_pending()              # close out the previous step's tally
                            pending = _reformat_forecast(raw_line.decode("utf-8", "replace").strip(), epoch)
                        elif b"Model finished (exit 0)" in raw_line:
                            step_ok += 1                  # this step's member ran OK
                        elif b"Model finished (exit" in raw_line:   # exit 0 handled above -> non-zero
                            step_fail += 1
                            # Tier C: a non-zero model exit — surface LOUDLY (console + file). A
                            # failed member otherwise hides in the 300k-line log while OpenDA still
                            # prints "Application Done" (this is how the §8 CRLF bug went unnoticed).
                            logger.warning("OpenDA member run failed: "
                                           + raw_line.decode("utf-8", "replace").strip())
                        elif b"Creating model instance" in raw_line:
                            instances += 1                # Tier A: count members initialized
                        elif b"Application initializing finished" in raw_line:
                            logger.info(f"OpenDA: initialized ({instances} model instances)",
                                        extra={"file_only": True})
                        elif (b"OpenDA version" in raw_line or b"className:" in raw_line
                              or b"Selected localization method" in raw_line):
                            logger.info("OpenDA: " + raw_line.decode("utf-8", "replace").strip(),
                                        extra={"file_only": True})   # Tier A: run provenance/config
                        elif b"Application Done" in raw_line:
                            _flush_pending()              # close out the final step
                            logger.info("OpenDA: application done", extra={"file_only": True})  # Tier D
            if done:
                break
            # Java flushes the logfile on its own cadence, so the bar advances in
            # bursts rather than perfectly smoothly — a buffering artefact, not a stall.
            time.sleep(0.5)
        _flush_pending()                  # emit the last step if "Application Done" never appeared
    finally:
        bar.close()

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, [oda_exe, oda_file])
    return proc.returncode


def openda_member_files(openda_dir, n_members, binding):
    """Members 1..N's trajectories in their OpenDA work dirs (this run only)."""
    return [binding.work_output(openda_dir, i) for i in range(1, n_members + 1)]


def openda_chain_member_files(openda_dir, n_members, model, results_dir="Results"):
    """Members 1..N's trajectories over a whole restart chain, kept as members of the OpenDA working
    dir (model.member_output_file): the work dirs hold only the latest run."""
    return [model.member_output_file(openda_dir, i, results_dir) for i in range(1, n_members + 1)]


def run_openda_chain(cfg, model, params, skip_oda=False, max_cycles=None):
    """Advance an OpenDA restart chain (openda/restart.py) to cfg["end_date"]: from the chain's
    end, or start_date for an empty chain, as one OpenDA run (catch-ups split into runs of at most
    cycle_max_days), perturbing the forcing for each. Returns the instant the chain now ends at --
    the time of the members' final state."""
    from ..perturbate import perturbator
    restart_dir = restart_chain.chain_dir(cfg, cfg["openda_dir"])
    start = restart_chain.utc(cfg["start_date"])
    end   = restart_chain.utc(cfg["end_date"])
    reached = restart_chain.chain_end(restart_dir)
    if reached is not None and reached < start:
        raise ValueError(f"[restart] the chain in {restart_dir} ends at {reached}, before start_date {start}: "
                         f"the run's warm start is newer than the chain. Empty the chain folder to start a "
                         f"new chain from the warm start.")
    cycles = restart_chain.plan_cycles(reached or start, end, restart_chain.settings(cfg)["cycle_max_days"])
    # OpenDA is never started on a window without observations (whether it accepts one is not
    # established): an empty run is merged into the next, and a gap at the end leaves the chain
    # where it is until the next observation arrives. The caller sees that in the returned instant.
    times = load_obs(resolve_obs_path(cfg))["time"]
    cycles = restart_chain.merge_empty_cycles(cycles, times)
    if max_cycles:
        cycles = cycles[:max_cycles]
    if not cycles:
        here = reached or start
        logger.info(f"[restart] nothing to do: the chain is at {here} and there is no observation "
                    f"after it up to {end}")
        return here
    for i, (s, e) in enumerate(cycles, 1):
        logger.info(f"=== OpenDA run {i}/{len(cycles)} of the restart chain: {s} -> {e} ===")
        run_cfg = {**cfg, "start_date": s.isoformat(), "end_date": e.isoformat()}
        logger.info(f"[3/5] perturbate forcing in ensemble1..{cfg['n_members']}")
        perturbator(run_cfg, model, params=params)
        run_openda(run_cfg, run_cfg, cfg["ensemble_base"], cfg["n_members"], skip_oda,
                   model_cfg=model.run_config(), model_name=model.name)
        if skip_oda:
            return s
    return cycles[-1][1]


def run_openda(cfg, ensemble_raw, ensemble_base, n_members, skip_oda=False,
               model_cfg=None, model_name=None):
    """OpenDA engine driver: sync inputs/forcings/warmup + build observations (adapt),
    render run.oda + the .gen.xml chain, launch oda_run.sh, then summarise.

    `model_cfg` is the selected model's runtime config (from main.py's -m/--model, i.e.
    models.MODELS). Its Docker image is written to template/model.json so the standalone
    OpenDA wrapper — a separate WSL subprocess that can't receive Python args — reads it
    from there instead of hardcoding the version.

    The generated working dir is named per run — openda_<model>_<lake>_<filter>, next to the
    members — so runs are self-describing and don't clash. Override with cfg["openda_dir"].

    cfg["openda_bin"] (the dir holding the OpenDA binaries, e.g. .../openda_3.4.0/bin) is used to
    build the full OpenDA environment (OPENDADIR/OPENDALIB, the bundled JRE + bin on PATH,
    LD_LIBRARY_PATH) for the oda_run.sh subprocess only, so it need not be sourced in the shell; the
    env is temporary to the run. Omit it (or set cfg["openda_native"], default linux64_gnu) to use an
    externally-sourced environment."""
    filter_type = cfg.get("algorithm", "EnKF")
    if filter_type not in FILTERS:
        raise ValueError(f"unknown algorithm '{filter_type}'; choose from {sorted(FILTERS)}")
    log_run_header(cfg)
    openda_dir  = cfg["openda_dir"]   # set by assimilator.config.build_config
    model       = get_model(model_name or cfg.get("model") or RUN_DEFAULTS["model"])
    binding     = openda_binding({"model": model.name})

    # Restart chain (opt-in, openda/restart.py): decided before adapt(), which needs to know whether
    # the template gets the warm start. Without "openda_restart" nothing here changes.
    chain = bool(cfg.get("openda_restart"))
    restart_in = restart_dir = identity = seed = None
    if chain:
        restart_dir = restart_chain.chain_dir(cfg, openda_dir)
        identity    = {"lake": ensemble_raw["lake"], "filter": filter_type, "n_members": n_members}
        restart_in  = restart_chain.select_restart(restart_dir, ensemble_raw["start_date"], identity)
        seed        = restart_chain.cycle_seed(ensemble_raw.get("rng_seed", 42), ensemble_raw["end_date"])
        logger.info(f"[restart] {ensemble_raw['start_date']} -> {ensemble_raw['end_date']}, OpenDA seed {seed}")

    # --- 4. adapter (always): sync inputs/forcings/warmup + build observations,
    #         returning the auto-detected obs depth list for the render below ----
    # Whether to split the observation series by season is decidable from the config alone, so the
    # adapter can write the files first and report back WHICH (depth, season) series it actually
    # wrote. The sigma per series is then keyed off those files, which is what makes the declared
    # series and the files on disk agree by construction -- deriving them from two passes over the
    # observations does not, since the adapter keeps only its target hour.
    logger.info(f"[4/5] adapt framework -> {display_path(openda_dir)}")
    adapt_raw = {**ensemble_raw, "openda_dir": openda_dir,
                 "_season_series": should_season_split(ensemble_raw)}
    if chain:
        adapt_raw["_restart_in"] = restart_in
    obs_depths, n_analysis, series_keys = adapt(adapt_raw)

    # Same resolution rule as the native engine (the config's sigma_obs, else sigma_default), so
    # the two engines cannot diverge on a config that names neither. A season pair becomes one
    # standardDeviation per (depth, season) -- exactly the value the native engine uses in that
    # season, rather than an average over the run, which would key sigma to the window.
    obs_std = (sigma_obs_for_series(series_keys, ensemble_raw)
               if should_season_split(ensemble_raw) else sigma_obs_spec(ensemble_raw))
    if is_season_keyed_series(obs_std):
        logger.info(f"[sigma] OpenDA series: {len(obs_std)} season-split (exact — one "
                    f"standardDeviation per (depth, season), matching the native engine)")

    # The config declares one timeSeries per entry of series_specs; the adapter has just written
    # one file per series. A disagreement is fatal INSIDE OpenDA — it opens a _real.csv that does
    # not exist and dies with no member output, tens of seconds later and with nothing in our log
    # to explain it. Cheap to assert here, where both numbers are in hand.
    n_declared = len(series_specs(obs_depths, obs_std))
    if n_declared != len(series_keys):
        raise ValueError(f"observation series mismatch: the config declares {n_declared} series "
                         f"but the adapter wrote {len(series_keys)}. OpenDA would die opening the "
                         f"missing one.")

    # Bridge the model image + the shared-container contract to the (separate-process) wrapper via a
    # generated file (single source of truth: the binding). The wrapper execs the model into ONE
    # persistent container that run_openda starts below (mounting openda_dir at mount_base); it maps
    # its work dir to <mount_base>/<relpath-from-openda_dir> and runs `binary` there, as the native
    # engine does.
    run_in     = binding.container(model_cfg)
    image, binary, mount_base = run_in["image"], run_in["binary"], run_in["mount_base"]
    container  = os.path.basename(openda_dir)
    model_json = os.path.join(openda_dir, "stochModel", "template", "model.json")
    with open(model_json, "w") as f:
        json.dump({"image": image, "container": container,
                   "mount_base": mount_base, "binary": binary,
                   "obs_operator": cfg.get("obs_operator", "linear")}, f)
    logger.info(f"      wrote {os.path.relpath(model_json, openda_dir)} "
                f"(image={image}, container={container})")

    # --- 5. render filter config + run OpenDA ------------------------------
    # Note: obs error std comes from the run config's "sigma_obs" (the same key the native
    # EnKF reads), so the two engines can't silently diverge. (config.py's internal param is still
    # named obs_std.)
    # OpenDA runs n_members + 1 instances (the main/control model + every member), so the cap is
    # n_members+1 — matching the original full-parallel maxThreads. (auto = min(cpu, n_members+1).)
    max_threads = resolve_max_workers(cfg, n_members + 1)
    # obs_std was resolved before step 4 — it decided the series set the adapter just wrote.
    oda_file = render_oda(openda_dir, binding, filter_type, n_members, obs_depths,
                          ensemble_raw["start_date"], ensemble_raw["end_date"],
                          obs_std=obs_std, max_threads=max_threads, restart=restart_in, seed=seed)
    logger.info(f"[5/5] rendered {oda_file} + chain for filter={filter_type} "
                f"(Results/work0..N, {len(obs_depths)} obs depths, maxThreads={max_threads})")
    # OpenDA launch: build the full OpenDA environment in-process from cfg["openda_bin"] (the dir
    # holding the OpenDA binaries, e.g. .../openda_3.4.0/bin) so it does NOT need sourcing in the
    # shell first. Everything is derived from that one path — OPENDADIR/OPENDALIB, the bundled JRE
    # and bin on PATH, LD_LIBRARY_PATH — mirroring the manual `export`s in the README. The env lives
    # only in this subprocess (temporary to the run; the parent process/shell is untouched). Omit
    # "openda_bin" to fall back to an externally-sourced environment.
    openda_bin = cfg.get("openda_bin")
    env = os.environ.copy()
    # The wrapper subprocesses OpenDA spawns must import the assimilator package (the model's modules).
    # They infer src/ by walking up from their own location, which breaks when openda_dir is rerouted
    # off the repo (run_root on ext4): pin PYTHONPATH to the repo src so the import resolves wherever
    # openda_dir lives. Java inherits this env and passes it to the wrapper processes.
    env["PYTHONPATH"] = os.pathsep.join([os.path.join(ROOT, "src"), env.get("PYTHONPATH", "")])
    if openda_bin:
        openda_bin  = os.path.expanduser(openda_bin)
        openda_root = os.path.dirname(openda_bin)                 # e.g. .../openda_3.4.0
        native      = cfg.get("openda_native", "linux64_gnu")
        openda_lib  = os.path.join(openda_bin, native)
        jre_bin     = os.path.join(openda_root, "jre", "bin")
        env["OPENDADIR"]       = openda_bin
        env["OPENDA_NATIVE"]   = native
        env["OPENDALIB"]       = openda_lib
        env["PATH"]            = os.pathsep.join([jre_bin, openda_bin, env.get("PATH", "")])
        env["LD_LIBRARY_PATH"] = os.pathsep.join([os.path.join(openda_lib, "lib"),
                                                  env.get("LD_LIBRARY_PATH", "")])
        oda_exe = os.path.join(openda_bin, "oda_run.sh")
    else:
        oda_exe = "oda_run.sh"

    if skip_oda:
        logger.info(f"      --skip-oda: run manually: "
                    f"cd {openda_dir} && {oda_exe} {oda_file}")
        return
    # Results/ holds both the per-member work dirs (Results/work0..N) and the PythonResultWriter
    # output; create it up front so OpenDA's result writer has somewhere to write.
    os.makedirs(os.path.join(openda_dir, "Results"), exist_ok=True)
    if chain:
        restart_chain.clear_openda_output(openda_dir, restart_dir)
    start_persistent_container(cfg, container, openda_dir, mount_base, image)
    logger.info(f"      started 1 persistent model container ({container})")
    logger.info(f"      {oda_exe} {oda_file}  (cwd={display_path(openda_dir)})")
    # Progress bar driven by tailing the logfile (see _launch_oda_with_progress). Same on/off flag
    # as the native engines (cfg["progress"], resolved in main.py). total = number of analysis
    # times from the adapter, so the bar tracks the real step count for any cadence; the
    # ±1 spin-up/boundary forecast is absorbed by tqdm's overflow tolerance (never asserted exact).
    progress = cfg.get("progress", False)
    total    = max(1, n_analysis)
    t0 = time.perf_counter()
    try:
        _launch_oda_with_progress(oda_exe, oda_file, openda_dir, env, total, progress, binding.model_epoch())
    except FileNotFoundError:
        raise RuntimeError(
            "oda_run.sh not found — set \"openda_bin\" in the arg file to the OpenDA bin dir, "
            "or source the OpenDA environment so oda_run.sh is on PATH")
    finally:
        # Remove the shared persistent model container (the wrapper exec'd into it instead of
        # docker-run-per-step). Runs on success or failure.
        stop_persistent_container(container)
        logger.info(f"      removed persistent model container ({container})")
    elapsed = time.perf_counter() - t0

    # Tidy the run dir: OpenDA writes its run log into the .oda cwd — move it into log/.
    log_src = os.path.join(openda_dir, "openda_logfile.txt")
    if os.path.isfile(log_src):
        log_dir = os.path.join(openda_dir, "log")
        os.makedirs(log_dir, exist_ok=True)
        shutil.move(log_src, os.path.join(log_dir, "openda_logfile.txt"))
        logger.info(f"      moved openda_logfile.txt -> {display_path(os.path.join(log_dir, 'openda_logfile.txt'))}")

    # Per-member work dirs live inside this run's dir at Results/work0..N.
    member_files = openda_member_files(openda_dir, n_members, binding)

    results_dir = cfg.get("results_dir", "Results")
    mean_path = model.mean_traj_path(openda_dir, filter_type)
    if chain:
        # The work dirs hold only this run: archive its result file and logs (the next run
        # overwrites them), append its member and mean trajectories to the chain's (only the new
        # rows, so the cost does not grow with the chain), and only then commit its restart.
        start_day = binding.model_time(ensemble_raw["start_date"])
        arch = restart_chain.archive_cycle(openda_dir, ensemble_raw["end_date"], [
            os.path.join(openda_dir, "Results", results_filename(filter_type)),
            os.path.join(openda_dir, "log", "openda_logfile.txt")] + [
            (os.path.join(openda_dir, "Results", f"work{k}", binding.wrapper_log), f"wrapper_work{k}.log")
            for k in range(n_members + 1)])
        chain_files = openda_chain_member_files(openda_dir, n_members, model, results_dir)
        counts = [restart_chain.append_trajectory(dst, src, start_day)
                  for src, dst in zip(member_files, chain_files)]
        run_mean = mean_path + ".run"
        model.accumulate_mean(member_files, run_mean)
        restart_chain.append_trajectory(mean_path, run_mean, start_day)
        os.remove(run_mean)
        member_files = chain_files
        restart_chain.commit_restart(openda_dir, restart_dir, ensemble_raw["end_date"], identity,
                                     extra={"start": str(ensemble_raw["start_date"]), "seed": seed,
                                            "started_from": restart_in or None})
        restart_chain.prune(restart_dir, restart_chain.settings(cfg)["keep"])
        logger.info(f"      appended this run to the chain's member trajectories and "
                    f"{os.path.basename(mean_path)} ({counts[0][1]} rows per member); outputs archived -> "
                    f"{os.path.relpath(arch, openda_dir)}")
    else:
        for i, member_file in enumerate(member_files, start=1):
            if os.path.exists(member_file):
                dst = model.member_output_file(openda_dir, i, results_dir)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(member_file, dst)
        model.accumulate_mean(member_files, mean_path)
        logger.info(f"      mirrored member trajectories -> ensemble{{1..{n_members}}}/ "
                    f"+ {os.path.basename(mean_path)}")

    obs_csv = resolve_obs_path(ensemble_raw)
    out_csv, skill = report_summary("openda", filter_type, member_files, ensemble_raw["lake"], obs_csv,
                                    openda_dir, model)
    # updates=None: OpenDA has no separate update count distinct from its analysis steps.
    log_run_footer(cfg, skill, n_analysis, None, elapsed, out_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export framework forcings + warmup into an OpenDA working dir")
    parser.add_argument("arg_file", help="Run config JSON (engine \"openda\", see docs/configuration.md)")
    cli = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-8s | %(name)-16s | %(message)s",
                        datefmt="%H:%M:%S")
    from assimilator.config import build_config
    with open(cli.arg_file) as f:
        adapt(build_config(json.load(f)))

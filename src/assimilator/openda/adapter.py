"""Export framework outputs into the standalone openda_simstrat/ layout.

Replaces openda_simstrat/generate_ensemble_forcings.py, generate_warmup_snapshot.py
and prepare_real_obs.py: instead of regenerating perturbed forcings (its own AR(1)),
a separate spin-up snapshot, and the observation CSVs by hand, this COPIES the Python
framework's already-generated inputs into OpenDA's layout and builds the stochObserver
observation files in one step.  Result: the OpenDA EnKF reference runs on byte-identical
forcings + warmup as the Python EnKF, so the cross-validation is rigorous, with no
duplicated generation.

Syncs:
  inputs/<lake>/*               (except OpenDA coupling files + Results/ + dated snapshots)
      -> openda_simstrat/stochModel/template/*                                 (Bathymetry, Grid, Settings.par,
                                                                                Absorption, Qin/Qout/Sin/Tin,
                                                                                InitialConditions, aed2.nml, AED2_*, ...)
  run/<lake>/ensemble{i}/Forcing.dat
      -> openda_simstrat/forcings/Forcing_{i}.dat                              (i = 0..N; 0 = control)
  inputs/<lake>/simulation-snapshot_<date>.dat
      -> openda_simstrat/stochModel/template/Results/simulation-snapshot.dat   (the warmup OpenDA reads)

Builds (formerly prepare_real_obs.py):
  observations/<lake>/temperature.csv  (raw 10-min profile observations; override with "obs_file")
      -> openda_simstrat/stochObserver/T_{depth}m_real.csv                     (one reading/day nearest noon UTC,
                                                                                time in fractional Simstrat days)

OpenDA-specific coupling files in the template are NEVER overwritten:
  temperature_state.txt, time_control.yaml, timeSeriesFormatter.xml.

OpenDA reads the warmup from template/Results/simulation-snapshot.dat (cloned into
each work dir; Simstrat "Continue from last snapshot" reads it).  The dated
simulation-snapshot_*.dat at the template ROOT is only an archive for diagnostics
and is left untouched.

OpenDA's XML configs / wrappers are left untouched (this only writes data files).
Prerequisite: run main.py (its copy + perturbate steps) first so the
ensemble Forcing.dat files exist.

Usage:  python src/assimilator/openda/adapter.py args/run_openda.json
"""

import os
import sys
import csv
import glob
import json
import shutil
import logging
import re
import argparse
from collections import defaultdict
from datetime import date, datetime, timezone, timedelta

# this file lives at src/assimilator/openda/adapter.py
SRC_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
ROOT    = os.path.dirname(SRC_DIR)                                                       # repo root
sys.path.insert(0, SRC_DIR)

import time
import subprocess

from assimilator.functions import (verify_args, resolve_src, resolve_root, resolve_obs_path,
                                   merge_lake_args, make_progress, log_run_header, log_run_footer,
                                   resolve_max_workers, resolve_run_root, display_path,
                                   start_persistent_container, stop_persistent_container,
                                   sigma_obs_spec, should_season_split,
                                   sigma_obs_for_series, season_of)
from assimilator.models.simstrat import read_snapshot, SIMSTRAT_REF_YEAR, accumulate_mean, mean_traj_path
from assimilator.summarize import report_summary
from .config import (FILTERS, render as render_oda, series_specs,
                     is_season_keyed as is_season_keyed_series)

logger = logging.getLogger(__name__)

REQUIRED = ["lake", "n_members", "ensemble_base"]

# Black-box coupling files OpenDA owns — never overwrite these in the template.
OPENDA_SPECIFIC = {"temperature_state.txt", "time_control.yaml", "timeSeriesFormatter.xml"}

# Observations: for a SUB-DAILY series, each day keeps the single reading nearest this UTC hour
# (the noon snapshot). A series already thinned to one reading a day is used at its own times
# instead -- see _is_thinned. Override the hour with "obs_target_hour" in the run config.
OBS_TARGET_HOUR = 12

# An observation this soon after the run start is dropped: it becomes the end of OpenDA's very
# first window, and too short a window leaves no model output rows to pair the observation
# against. Costs one analysis out of ~365 (first window); two hours clears any plausible output interval.
# Only reachable since observations are stamped at their own time -- a fixed noon never was.
FIRST_WINDOW_MIN_MINUTES = 120

# The only hand-maintained OpenDA pieces (everything else in openda_simstrat/ is
# generated/synced). Copied into the working dir at adapt time so the working dir
# stays fully reproducible from static/openda/.
STATIC_OPENDA = os.path.join(ROOT, "static", "openda")


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


def _simstrat_day_at(day_str, ref_date, hour=OBS_TARGET_HOUR):
    """Fractional Simstrat day at `hour` UTC for a YYYY-MM-DD string (noon -> day + 0.5).

    Kept EXACT, not rounded: the observation time becomes the window end Simstrat integrates to."""
    return (date.fromisoformat(day_str) - ref_date).days + hour / 24.0


def _utc_minutes_since_midnight(iso_str):
    # Drop sub-second precision before parsing. Whole seconds is all this needs.
    dt = datetime.fromisoformat(re.sub(r"\.\d+", "", iso_str))
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc)
    return dt.hour * 60 + dt.minute + dt.second / 60.0


def _is_thinned(obs_csv):
    """True when the file already carries at most one reading per (depth, day).

    That is the case where the target-hour bin has nothing to choose BETWEEN: every row is already
    the day's single observation, so it is assimilated at ITS OWN timestamp rather than matched
    against a fixed hour. This is what makes a series whose sampling hour drifts work, and a lake 
    sampling 00-06Z would match a noon it never samples. It is also what the native engine does: 
    load_obs keeps every reading at its own hour."""
    seen = set()
    with open(obs_csv, newline="") as f:
        for row in csv.DictReader(f):
            if not row.get("value"):
                continue
            key = (row["depth"], row["time"][:10])
            if key in seen:
                return False
            seen.add(key)
    return True


def _model_output_depths(inputs_dir):
    """Depths (positive metres) the model outputs, read from <inputs_dir>/z_out.dat. Reads the
    canonical source (model_inputs) rather than the template copy. Returns [] if absent (no
    depth filtering)."""
    path = os.path.join(inputs_dir, "z_out.dat")
    if not os.path.isfile(path):
        return []
    depths = []
    with open(path) as f:
        for line in f:
            try:
                depths.append(abs(float(line.strip())))
            except ValueError:
                continue  # header line ("Depths [m]")
    return depths


def _build_observations(raw, openda_dir, model_inputs):
    """Build the OpenDA stochObserver obs files from the raw profile CSV.

    Writes stochObserver/T_{depth}m[_{season}]_real.csv, time in fractional Simstrat days since
    1 Jan SIMSTRAT_REF_YEAR. WHEN an observation is taken depends on the series:

      already one reading per (depth, day)  -> that reading, at ITS OWN time (_is_thinned)
      sub-daily                             -> the mean of the centered target-hour bin, at that
                                               hour ([11:30, 12:30) by default; "obs_target_hour")

    Both mirror functions.load_obs's centered hourly bin, so OpenDA and the native engines
    assimilate identical obs -- and OpenDA analyses at exactly these times, since its
    analysisTimes is "fromObservationTimes", the same rule as the native window_mode="obs".

    Depths are auto-detected from the CSV, dropped if they have fewer than `obs_min_days` days or
    no matching model-output depth (model_inputs/z_out.dat), and returned sorted for the config
    generator to wire everywhere.
    """
    lake      = raw["lake"]
    stoch_dir = os.path.join(openda_dir, "stochObserver")
    obs_csv   = resolve_obs_path(raw)
    if not os.path.isfile(obs_csv):
        raise FileNotFoundError(
            f"observation source not found: {obs_csv} (set 'obs_file' or add observations/{lake}/temperature.csv)")

    ref_date     = date(SIMSTRAT_REF_YEAR, 1, 1)
    start = date.fromisoformat(raw["start_date"][:10]) if raw.get("start_date") else None
    end   = date.fromisoformat(raw["end_date"][:10])   if raw.get("end_date")   else None
    min_days = raw.get("obs_min_days", 1)
    # An explicit "obs_target_hour" always wins: it is the escape hatch for a series the one-per-day test reads wrongly.
    thinned        = raw.get("obs_target_hour") is None and _is_thinned(obs_csv)
    target_hour    = int(raw.get("obs_target_hour", OBS_TARGET_HOUR))
    target_minutes = target_hour * 60

    # acc[depth][day_str] = [sum, count, minutes] — `minutes` is when the observation is assimilated
    # (its own time when thinned, else the target hour). Mirrors functions.load_obs's centered
    # hourly bin. (A day with no sample in that bin emits no obs, like load_obs's empty bin.)
    acc = defaultdict(lambda: defaultdict(lambda: [0.0, 0, 0.0]))
    too_early = 0
    with open(obs_csv, newline="") as f:
        for row in csv.DictReader(f):
            if not row.get("value"):
                continue
            day_str = row["time"][:10]
            day = date.fromisoformat(day_str)
            if (start and day < start) or (end and day > end):
                continue
            minutes = _utc_minutes_since_midnight(row["time"])
            if not thinned and not (target_minutes - 30 <= minutes < target_minutes + 30):
                continue
            # The run's very first window ends at this observation; too short and OpenDA gets no
            # model rows to pair against (see FIRST_WINDOW_MIN_MINUTES).
            if start and day == start and minutes < FIRST_WINDOW_MIN_MINUTES:
                too_early += 1
                continue
            depth = float(row["depth"])
            cell = acc[depth][day_str]
            cell[0] += float(row["value"])
            cell[1] += 1
            cell[2] = minutes if thinned else target_minutes

    if too_early:
        logger.warning(f"[adapter] dropped {too_early} observation(s) on {start} taken less than "
                       f"{FIRST_WINDOW_MIN_MINUTES / 60:g} h after the run start — OpenDA's first "
                       f"window would be too short to produce model rows to pair against")

    depths = sorted(d for d in acc if len(acc[d]) >= min_days)

    # Keep only depths the model actually outputs (z_out.dat): an obs depth with no
    # matching model output depth can't be assimilated, since there is no model prediction to compare it against.
    model_depths = _model_output_depths(model_inputs)
    if model_depths:
        matched = [d for d in depths if any(abs(d - m) <= 1e-6 for m in model_depths)]
        dropped = [d for d in depths if d not in matched]
        if dropped:
            logger.warning(f"[adapter] dropping obs depths with no matching model output depth (z_out.dat): "
                           f"{[f'{d:g}' for d in dropped]} m")
        depths = matched

    # One file per SERIES, and the series set is decided by config.series_specs -- season-split
    # when sigma is season-keyed, else one per depth. The two must agree or OpenDA opens a
    # timeSeries file that does not exist and dies with no member output.
    season_split = bool(raw.get("_season_series"))
    window = f"{start or 'start'}..{end or 'end'}"
    when   = "at their own times (already thinned)" if thinned else f"hour {target_hour:02d}Z"
    logger.info(f"[adapter] observations: {os.path.relpath(obs_csv, ROOT)} -> "
                f"stochObserver/T_*_real.csv  ({len(depths)} depths {[f'{d:g}' for d in depths]}, "
                f"window {window}, {when}"
                f"{', season-split' if season_split else ''})")
    os.makedirs(stoch_dir, exist_ok=True)

    series_keys = []            # [(depth, season)] when split, else [(depth, None)] -- what the
    for depth in depths:        # config declares, taken from what is actually on disk.
        records = acc[depth]
        # {season: [day_str]} -- one bucket unless split, and then the buckets partition the days,
        # so no observation is dropped or written twice.
        buckets = defaultdict(list)
        for day_str in sorted(records):
            buckets[season_of(date.fromisoformat(day_str).month) if season_split else None
                    ].append(day_str)
        for season, days in buckets.items():
            suffix = f"_{season}" if season else ""
            out_path = os.path.join(stoch_dir, f"T_{depth:g}m{suffix}_real.csv")
            with open(out_path, "w", newline="") as f:
                f.write("time,value\n")
                for day_str in days:
                    s, c, minutes = records[day_str]
                    f.write(f"{_simstrat_day_at(day_str, ref_date, minutes / 60.0):.6f},"
                            f"{s / c:.6f}\n")
            series_keys.append((depth, season))
    logger.info(f"  wrote {len(series_keys)} series files -> {os.path.relpath(stoch_dir, ROOT)}")

    # Distinct analysis (noon-obs) times across the kept depths = how many forecast/analysis
    # steps OpenDA will run. Drives the progress-bar total so the bar tracks the real step count
    # for ANY cadence (daily, sub-daily, or sparse obs), not just calendar days.
    analysis_days = set()
    for d in depths:
        analysis_days.update(acc[d])
    return depths, len(analysis_days), series_keys


def adapt(raw):
    verify_args(raw, REQUIRED)

    lake          = raw["lake"]
    n_members     = raw["n_members"]
    ensemble_base = resolve_src(raw["ensemble_base"])
    model_inputs = raw.get("model_inputs_path")
    model_inputs = resolve_src(model_inputs) if model_inputs \
        else os.path.join(ROOT, "inputs", lake)

    openda_dir   = resolve_src(raw["openda_dir"]) if raw.get("openda_dir") \
        else os.path.join(ROOT, "run", "openda_simstrat")
    forcings_dir = os.path.join(openda_dir, "forcings")
    template_dir = os.path.join(openda_dir, "stochModel", "template")

    # openda_simstrat/ is fully generated — create the working dir + template skeleton
    # on demand (nothing is committed; the static wrapper lives in static/openda/).
    os.makedirs(template_dir, exist_ok=True)   # also creates openda_dir/stochModel

    logger.info(f"[adapter] lake={lake}  framework={display_path(ensemble_base)}  "
                f"-> openda={display_path(openda_dir)}")

    # ------------------------------------------------------------------
    # 0. Hand-maintained wrapper: static/openda/* -> working dir
    #    (the only non-generated OpenDA pieces; everything else is rendered/synced)
    # ------------------------------------------------------------------
    logger.info("[adapter] wrapper (static/openda) -> stochModel/:")
    _copy(os.path.join(STATIC_OPENDA, "simstratWrapperEnKF.xml"),
          os.path.join(openda_dir, "stochModel", "simstratWrapperEnKF.xml"))
    _copy(os.path.join(STATIC_OPENDA, "simstrat_wrapper_enkf.py"),
          os.path.join(openda_dir, "stochModel", "bin", "simstrat_wrapper_enkf.py"))

    # ------------------------------------------------------------------
    # 1. Model inputs: model_inputs/* -> template/
    #    (skip OpenDA coupling files, the heavy Results/, the visualization-only
    #    ref/, and dated snapshot archives — the warmup is placed into
    #    template/Results/ in step 3)
    # ------------------------------------------------------------------
    if not os.path.isdir(model_inputs):
        raise FileNotFoundError(
            f"model_inputs not found: {model_inputs} (provide it manually — Simstrat inputs + a dated simulation-snapshot_*.dat)")
    logger.info(f"[adapter] model inputs -> template/ (skipping OpenDA-specific {sorted(OPENDA_SPECIFIC)}):")
    for name in sorted(os.listdir(model_inputs)):
        if name in OPENDA_SPECIFIC or name in ("Results", "ref") or name.startswith("simulation-snapshot_"):
            continue
        _copy_path(os.path.join(model_inputs, name),
                   os.path.join(template_dir, name))

    # ------------------------------------------------------------------
    # 2. Perturbed forcings: ensemble{i}/Forcing.dat -> forcings/Forcing_{i}.dat
    # ------------------------------------------------------------------
    logger.info(f"[adapter] forcings (0..{n_members}):")
    for i in range(n_members + 1):                       # 0 = control, 1..N = members
        src = os.path.join(ensemble_base, f"ensemble{i}", "Forcing.dat")
        if not os.path.isfile(src):
            raise FileNotFoundError(
                f"{src} missing — run main.py (copy + perturbate steps) first")
        _copy(src, os.path.join(forcings_dir, f"Forcing_{i}.dat"))
    logger.info(f"  copied {n_members + 1} forcing files -> {os.path.relpath(forcings_dir, ROOT)}")

    # ------------------------------------------------------------------
    # 3. Warmup snapshot -> template/Results/simulation-snapshot.dat
    #    (the live name OpenDA clones into each work dir and continues from).
    #    Source: the dated warmup archive in model_inputs (stable; the live
    #    model_inputs/Results/ copy may have been overwritten by later runs).
    # ------------------------------------------------------------------
    dated = sorted(glob.glob(os.path.join(model_inputs, "simulation-snapshot_*.dat")))
    if not dated:
        logger.warning(f"[adapter] no simulation-snapshot_*.dat in "
                       f"{os.path.relpath(model_inputs, ROOT)} — skipping warmup sync")
    else:
        snap_src = dated[-1]
        target   = os.path.join(template_dir, "Results", "simulation-snapshot.dat")
        logger.info(f"[adapter] warmup: {os.path.basename(snap_src)} "
                    f"-> {os.path.relpath(target, openda_dir)} (overwrites OpenDA's current warmup)")
        _copy(snap_src, target)

        # Seed OpenDA's initial state (temperature_state.txt) from the warmup
        # snapshot's full-grid T profile — one value per cell.  Replaces the legacy
        # 7-value placeholder and is automatically the right size for this lake's grid.
        state_path = os.path.join(template_dir, "temperature_state.txt")
        T = read_snapshot(snap_src, par_path=os.path.join(template_dir, "Settings.par")).model["T"]
        with open(state_path, "w") as f:
            for t in T:
                f.write(f"{float(t):.6f}\n")
        logger.info(f"[adapter] temperature_state.txt seeded from warmup ({len(T)} cells)")

    # ------------------------------------------------------------------
    # 4. Observations: observations/<lake>/temperature.csv -> stochObserver/T_{depth}m_real.csv
    #    (noon-snapshot per depth; formerly prepare_real_obs.py).  Returns the
    #    auto-detected depth list for the config generator to wire everywhere.
    # ------------------------------------------------------------------
    obs_depths, n_analysis, series_keys = _build_observations(raw, openda_dir, model_inputs)

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
#   C failures (WARNING, loud):  "Simstrat finished (exit N)" with N != 0
#   D footer (file_only):       "Application Done"
# Deliberately NOT lifted (would flood): "Written T_*.csv" (~115k) and the per-member-per-step
# wrapper lines ([TIMING]/[STATE]/[RESTART]/"Running Simstrat via Docker", ~7.7k each).
_FORECAST_MARK = b"Forecast from"
_FORECAST_DAYS = re.compile(r"\(([\d.]+)\s*-+>\s*([\d.]+)\)")   # the "(A-->B)" Simstrat day span


def _reformat_forecast(line):
    """OpenDA prints the Forecast line as '... <ts>UTC to <ts>UTC (A-->B)', but the <ts>UTC strings
    render the Simstrat day number against the WRONG epoch (the MJD epoch 1858-11-17, not the
    Simstrat reference year), giving nonsensical ~1902 dates. The '(A-->B)' Simstrat day numbers ARE
    correct (the same convention summarize/_score use), so convert THOSE to real calendar dates and
    drop the bogus UTC — and the per-step line then shows the same dates as the native engines.
    Falls back to the original line if the day span can't be parsed."""
    m = _FORECAST_DAYS.search(line)
    if not m:
        return line
    epoch = datetime(SIMSTRAT_REF_YEAR, 1, 1)
    t0 = epoch + timedelta(days=float(m.group(1)))
    t1 = epoch + timedelta(days=float(m.group(2)))
    return (f"Forecast {t0:%Y-%m-%d %H:%M} -> {t1:%Y-%m-%d %H:%M} "
            f"(sim-day {m.group(1)}->{m.group(2)})")


def _launch_oda_with_progress(oda_exe, oda_file, openda_dir, env, total, enabled):
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
    step_ok = step_fail = 0           # Simstrat exits seen since `pending` (this step's instances)

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
                            pending = _reformat_forecast(raw_line.decode("utf-8", "replace").strip())
                        elif b"Simstrat finished (exit 0)" in raw_line:
                            step_ok += 1                  # this step's member ran OK
                        elif b"Simstrat finished (exit" in raw_line:   # exit 0 handled above -> non-zero
                            step_fail += 1
                            # Tier C: a non-zero Simstrat exit — surface LOUDLY (console + file). A
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


def run_openda(cfg, ensemble_raw, ensemble_base, n_members, skip_oda=False,
               model_cfg=None, model_name="simstrat"):
    """OpenDA engine driver: sync inputs/forcings/warmup + build observations (adapt),
    render run.oda + the .gen.xml chain, launch oda_run.sh, then summarise.

    `model_cfg` is the selected model's runtime config (from main.py's -m/--model, i.e.
    models.MODELS). Its Docker image is written to template/model.json so the standalone
    OpenDA wrapper — a separate WSL subprocess that can't receive Python args — reads it
    from there instead of hardcoding the version.

    The generated working dir is named per run — run/openda_<model>_<lake>_<filter> (e.g.
    run/openda_simstrat_upperlugano_enkf) — so runs are self-describing and don't clash.
    Override with cfg["openda_dir"].

    cfg["openda_bin"] (the dir holding the OpenDA binaries, e.g. .../openda_3.4.0/bin) is used to
    build the full OpenDA environment (OPENDADIR/OPENDALIB, the bundled JRE + bin on PATH,
    LD_LIBRARY_PATH) for the oda_run.sh subprocess only, so it need not be sourced in the shell; the
    env is temporary to the run. Omit it (or set cfg["openda_native"], default linux64_gnu) to use an
    externally-sourced environment."""
    filter_type = cfg.get("algorithm", "EnKF")
    if filter_type not in FILTERS:
        raise ValueError(f"unknown algorithm '{filter_type}'; choose from {sorted(FILTERS)}")
    log_run_header(cfg)
    default_dir = os.path.join(resolve_run_root(cfg),
                               f"openda_{model_name}_{ensemble_raw['lake']}_{filter_type.lower()}")
    openda_dir  = resolve_root(cfg.get("openda_dir") or default_dir)

    # --- 4. adapter (always): sync inputs/forcings/warmup + build observations,
    #         returning the auto-detected obs depth list for the render below ----
    # Whether to split the observation series by season is decidable from the config alone, so the
    # adapter can write the files first and report back WHICH (depth, season) series it actually
    # wrote. The sigma per series is then keyed off those files, which is what makes the declared
    # series and the files on disk agree by construction -- deriving them from two passes over the
    # observations does not, since the adapter keeps only its target hour.
    logger.info(f"[4/5] adapt framework -> {display_path(openda_dir)}")
    obs_depths, n_analysis, series_keys = adapt({
        **ensemble_raw, "openda_dir": openda_dir,
        "_season_series": should_season_split(ensemble_raw)})

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
    # generated file (single source of truth: models.py). The wrapper execs Simstrat into ONE
    # persistent container that run_openda starts below (mounting openda_dir at mount_base); it maps
    # its work dir to <mount_base>/<relpath-from-openda_dir> and runs `binary` there — bypassing the
    # image's /entrypoint.sh (hardcoded `cd /simstrat/run`), exactly as the native engine does.
    image      = f"eawag/simstrat:{(model_cfg or {}).get('simstrat_version', '3.0.4')}"
    binary     = (model_cfg or {}).get("simstrat_binary", "/simstrat/build/simstrat")
    container  = os.path.basename(openda_dir)
    mount_base = "/simstrat/run"
    model_json = os.path.join(openda_dir, "stochModel", "template", "model.json")
    with open(model_json, "w") as f:
        json.dump({"image": image, "container": container,
                   "mount_base": mount_base, "binary": binary}, f)
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
    oda_file = render_oda(openda_dir, filter_type, n_members, obs_depths,
                          ensemble_raw["start_date"], ensemble_raw["end_date"],
                          obs_std=obs_std, max_threads=max_threads)
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
    # The wrapper subprocesses OpenDA spawns must import the assimilator package (read_snapshot etc.).
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
                    f"cd {os.path.relpath(openda_dir, ROOT)} && {oda_exe} {oda_file}")
        return
    # Results/ holds both the per-member work dirs (Results/work0..N) and the PythonResultWriter
    # output; create it up front so OpenDA's result writer has somewhere to write.
    os.makedirs(os.path.join(openda_dir, "Results"), exist_ok=True)
    start_persistent_container(cfg, container, openda_dir, mount_base, image)
    logger.info(f"      started 1 persistent Simstrat container ({container})")
    logger.info(f"      {oda_exe} {oda_file}  (cwd={os.path.relpath(openda_dir, ROOT)})")
    # Progress bar driven by tailing the logfile (see _launch_oda_with_progress). Same on/off flag
    # as the native engines (cfg["progress"], resolved in main.py). total = number of analysis
    # (noon-obs) times from the adapter, so the bar tracks the real step count for any cadence; the
    # ±1 spin-up/boundary forecast is absorbed by tqdm's overflow tolerance (never asserted exact).
    progress = cfg.get("progress", False)
    total    = max(1, n_analysis)
    t0 = time.perf_counter()
    try:
        _launch_oda_with_progress(oda_exe, oda_file, openda_dir, env, total, progress)
    except FileNotFoundError:
        raise RuntimeError(
            "oda_run.sh not found — set \"openda_bin\" in the arg file to the OpenDA bin dir, "
            "or source the OpenDA environment so oda_run.sh is on PATH")
    finally:
        # Remove the shared persistent Simstrat container (the wrapper exec'd into it instead of
        # docker-run-per-step). Runs on success or failure.
        stop_persistent_container(container)
        logger.info(f"      removed persistent Simstrat container ({container})")
    elapsed = time.perf_counter() - t0

    # Tidy the run dir: OpenDA writes its run log into the .oda cwd — move it into log/.
    log_src = os.path.join(openda_dir, "openda_logfile.txt")
    if os.path.isfile(log_src):
        log_dir = os.path.join(openda_dir, "log")
        os.makedirs(log_dir, exist_ok=True)
        shutil.move(log_src, os.path.join(log_dir, "openda_logfile.txt"))
        logger.info(f"      moved openda_logfile.txt -> {os.path.relpath(os.path.join(log_dir, 'openda_logfile.txt'), ROOT)}")

    # Per-member work dirs live inside this run's dir at Results/work0..N.
    work_base = os.path.join(openda_dir, "Results")
    member_files = [os.path.join(work_base, f"work{i}", "Results", "T_out.dat")
                    for i in range(1, n_members + 1)]

    results_dir = cfg.get("results_dir", "Results")
    for i, member_file in enumerate(member_files, start=1):
        if os.path.exists(member_file):
            dst_dir = os.path.join(openda_dir, f"ensemble{i}", results_dir)
            os.makedirs(dst_dir, exist_ok=True)
            shutil.copy2(member_file, os.path.join(dst_dir, "T_out.dat"))
    mean_path = mean_traj_path(openda_dir, filter_type)
    accumulate_mean(member_files, mean_path)
    logger.info(f"      mirrored member trajectories -> ensemble{{1..{n_members}}}/{results_dir}/T_out.dat "
                f"+ {os.path.basename(mean_path)}")

    obs_csv = resolve_obs_path(ensemble_raw)
    out_csv, skill = report_summary("openda", filter_type, member_files, ensemble_raw["lake"], obs_csv, openda_dir)
    # updates=None: OpenDA has no separate update count distinct from its analysis steps.
    log_run_footer(cfg, skill, n_analysis, None, elapsed, out_csv)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export framework forcings + warmup into openda_simstrat/")
    parser.add_argument("arg_file", help="Path to JSON args file (e.g. args/run_openda.json)")
    parser.add_argument("--lake", default=None, help="Lake to adapt from the config's \"lakes\" block")
    cli = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-8s | %(name)-16s | %(message)s",
                        datefmt="%H:%M:%S")

    arg_file = cli.arg_file
    if not os.path.isfile(arg_file):
        arg_file = os.path.join(ROOT, arg_file)
    if not os.path.isfile(arg_file):
        raise ValueError(f"Args file not found: {cli.arg_file}")

    with open(arg_file) as f:
        raw_args = merge_lake_args(json.load(f), lake=cli.lake)

    adapt(raw_args)

"""All Simstrat-specific model behaviour, in one place.

Simstrat is a 1D lake model run in Docker (`eawag/simstrat:<version>`). This module
holds everything that knows about Simstrat's mechanics — Docker container lifecycle,
`Settings.par` editing, the day-since-reference-year time convention, the output
formats (`z_out.dat` / `T_out.dat`), the input layout, and the binary snapshot I/O
(`read_snapshot`/`write_snapshot`, at the bottom of this module) — and exposes it
through the `Simstrat` class.

The engines (`algorithms/enkf.py`, `algorithms/pf.py`) and the orchestrator
(`main.py`) call models only through the methods of this class — selecting one via
`get_model()` in `models/__init__.py` — so the contract is whatever those methods are:

    run_config                       runtime knobs merged into the engine's run args
    model_inputs_ready / warmup_snapshot / instances_ready / copy_model_inputs   input readiness & setup
    start_containers / stop_containers / run_window            per-window run machinery
    read_ref_date / model_output_depths / set_output_depths    state / output IO
    obs_to_sim_col / load_T / clear_member_outputs
    accumulate_mean / mean_traj_path
    read_snapshot_T / write_snapshot_T

Adding another model means writing a sibling class with the same surface and
registering it in `models/__init__.py`; the engines don't change. Once a second model
exists, extract the shared subset into an ABC — for now it's just Simstrat.

The functions below keep their original signatures (taking the pipeline's `args`/`raw`
dicts) and are bound onto `Simstrat` as static methods at the bottom.
"""

from __future__ import annotations

import os
import glob
import json
import shutil
import logging
import subprocess
import concurrent.futures
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Tuple

import numpy as np
# NOTE: pandas is imported lazily inside the functions that need it (accumulate_mean, load_T).
# The OpenDA wrapper imports read_snapshot/write_snapshot from this module 21×/step and those use
# only numpy; a top-level `import pandas` cost ~2.6s per process launch for nothing. (numpy ~0.7s.)

from ..functions import ROOT, GENERAL, resolve_src, verify_args

logger = logging.getLogger(__name__)

# Simstrat owns these (read from static/general.json): the day-since-reference-year epoch and the
# Forcing.dat header line. Model-specific, so they live here rather than in the generic functions.py.
SIMSTRAT_REF_YEAR = GENERAL["simstrat_ref_year"]
FORCING_HEADER    = GENERAL["forcing_header"]


# ---------------------------------------------------------------------------
# Input readiness / instance setup
# ---------------------------------------------------------------------------

def model_inputs_ready(model_inputs):
    """True if a dated simulation-snapshot_*.dat and Forcing.dat exist (step 1 precondition)."""
    return (bool(glob.glob(os.path.join(model_inputs, "simulation-snapshot_*.dat")))
            and os.path.isfile(os.path.join(model_inputs, "Forcing.dat")))


def warmup_snapshot(model_inputs):
    """The dated warm-start snapshot the members restart from — the latest
    simulation-snapshot_*.dat (same latest-wins pick as the per-member seeding), or None
    if absent. Logged by main.py step 1 so the run's starting state is named in the log."""
    snaps = sorted(glob.glob(os.path.join(model_inputs, "simulation-snapshot_*.dat")))
    return snaps[-1] if snaps else None


def instances_ready(ensemble_base, n_members):
    """True if every ensemble{0..N}/Settings.par exists (step 2 done)."""
    return all(os.path.isfile(os.path.join(ensemble_base, f"ensemble{i}", "Settings.par"))
               for i in range(n_members + 1))


# Heavy spin-up output each member regenerates; skip to avoid copying GBs per dir.
COPY_DEFAULT_SKIP = {"Results", "ref"}


def _copy_dir(src_dir, dst_dir, skip=None):
    """Copy files/subdirs from src_dir into dst_dir (overwriting), skipping names in
    `skip`. Does not wipe dst_dir, so unrelated outputs are preserved."""
    os.makedirs(dst_dir, exist_ok=True)
    for fname in os.listdir(src_dir):
        if skip and fname in skip:
            continue
        src = os.path.join(src_dir, fname)
        dst = os.path.join(dst_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
        elif os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)


def copy_model_inputs(raw):
    """Step 2: clone inputs/<lake>/ into ensemble0..N (0 = control,
    1..N = members whose Forcing.dat perturbate overwrites later). Skips the heavy
    Results/ dir — each member regenerates it and seeds the warmup from the dated
    simulation-snapshot_*.dat that IS copied."""
    verify_args(raw, ["lake", "n_members", "ensemble_base"])

    lake          = raw["lake"]
    n_members     = raw["n_members"]
    ensemble_base = resolve_src(raw["ensemble_base"])
    model_inputs = raw.get("model_inputs_path")
    model_inputs = resolve_src(model_inputs) if model_inputs \
        else os.path.join(ROOT, "inputs", lake)
    skip = set(raw.get("copy_skip", COPY_DEFAULT_SKIP))

    if not os.path.isdir(model_inputs):
        raise FileNotFoundError(
            f"model_inputs not found: {model_inputs} "
            f"(provide it manually — the Simstrat inputs + a dated simulation-snapshot_*.dat)")

    for i in range(n_members + 1):           # 0..N : control + members
        dst = os.path.join(ensemble_base, f"ensemble{i}")
        _copy_dir(model_inputs, dst, skip=skip)

    logger.info(f"{lake}: copied model_inputs -> ensemble0..{n_members} "
                f"under {ensemble_base} (skipped: {sorted(skip)})")


# ---------------------------------------------------------------------------
# Output formats (z_out.dat / T_out.dat)
# ---------------------------------------------------------------------------

def _read_z_out(path):
    """Positive-metre depths from a z_out.dat (skips the 'Depths [m]' header). [] if absent."""
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


def _write_z_out(path, depths):
    """Write z_out.dat: header + one negative-metre depth per line, surface (0) first down to
    the deepest. Depths given as positive metres."""
    with open(path, "w") as f:
        f.write("Depths [m]\n")
        for z in sorted(depths):          # ascending positive (0, 0.5, 1, …)
            f.write(f"{-z:.2f}\n")         # stored negative, so the file runs surface -> bed


def model_output_depths(ensemble_base):
    """Depths (positive metres) the model outputs, read from a member's z_out.dat.
    Mirrors openda/adapter._model_output_depths so the two engines filter obs against the
    same grid. Returns [] if the file is absent (then no depth filtering is applied)."""
    return _read_z_out(os.path.join(ensemble_base, "ensemble1", "z_out.dat"))


def set_output_depths(model_inputs, ensemble_base, n_members, obs_depths):
    """Overwrite z_out.dat — in model_inputs and every member (ensemble0..N) — with the superset
    of its current output depths and the observation depths, so no observation is dropped just for
    lack of a matching model-output depth (depths are matched at 0.01 m). Observations deeper than
    the existing grid's bed are left out (outside the simulated water column). No-op where
    z_out.dat is absent. Returns the number of files rewritten."""
    obs = {round(abs(d), 2) for d in obs_depths}
    targets = [model_inputs] + [os.path.join(ensemble_base, f"ensemble{i}") for i in range(n_members + 1)]
    rewritten = 0
    for d in targets:
        path = os.path.join(d, "z_out.dat")
        existing = _read_z_out(path)
        if not existing:
            continue                      # absent/empty: leave Simstrat's default output grid
        deepest = max(existing)
        union   = set(existing) | {o for o in obs if o <= deepest}
        if union != set(existing):
            _write_z_out(path, union)
            rewritten += 1
    return rewritten


def obs_to_sim_col(depth):
    """Map an observation depth ("X m below surface", positive) to the Simstrat output/state
    column coordinate (height -X; negative = below surface). The engines build H / score obs
    against these coordinates, so the convention is the model's."""
    return -depth


def clear_member_outputs(ensemble_base, member_ids, results_dir):
    """Delete accumulated *_out.dat in each member's results_dir. Simstrat APPENDS to its
    output files across the daily windows, so a fresh run must clear them once up front,
    otherwise it would append onto stale data from a previous run. Called on reset."""
    for i in member_ids:
        rdir = os.path.join(ensemble_base, f"ensemble{i}", results_dir)
        if os.path.isdir(rdir):
            for fname in os.listdir(rdir):
                if fname.endswith("_out.dat") or fname == T_OUT_OFFSET_FILE:
                    os.remove(os.path.join(rdir, fname))


def accumulate_mean(member_ids, args):
    """Write the ensemble-mean trajectory to args['mean_traj_path'] from each member's
    (full, accumulated) T_out.dat. One-shot: call once after the run. Averages across
    whatever members are present at each timestamp, so it tolerates a member missing a
    failed window."""
    import pandas as pd
    def _read(i):
        path = os.path.join(args["ensemble_base"], f"ensemble{i}", args["results_dir"], "T_out.dat")
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path)
        df.columns = [c.strip().strip('"') for c in df.columns]
        return df

    with concurrent.futures.ThreadPoolExecutor() as pool:
        frames = [f for f in pool.map(_read, member_ids) if f is not None]
    if not frames:
        return
    time_col = frames[0].columns[0]
    mean_df  = pd.concat(frames).groupby(time_col, as_index=False).mean()
    mean_df.to_csv(args["mean_traj_path"], index=False)


def mean_traj_path(ensemble_base, algorithm):
    """Path of the ensemble-mean trajectory file (Simstrat T_out naming, alongside members')."""
    return os.path.join(ensemble_base, f"T_out_{algorithm.lower()}_mean.dat")


def load_T(ensemble_dir, args):
    import pandas as pd
    path = os.path.join(ensemble_dir, args["results_dir"], "T_out.dat")
    ref  = pd.Timestamp(args["ref_date"])
    df = pd.read_csv(path)
    df.columns = [c.strip().strip('"') for c in df.columns]
    df["time"] = (ref + pd.to_timedelta(df["Datetime"], unit="D")).dt.round("1h")
    df = df.drop(columns=["Datetime"]).set_index("time")
    df.columns = df.columns.astype(float)
    return df


# Sidecar file (next to T_out.dat) recording the byte offset read so far, so a per-window read
# resumes from where the last one stopped instead of re-parsing the whole growing file. Single
# source of truth for BOTH engines: the native PF (load_T_window, below) and the OpenDA wrapper
# (static/openda/simstrat_wrapper_enkf.py imports this and read_t_out_tail).
T_OUT_OFFSET_FILE = ".t_out_read_offset"


def read_t_out_tail(filename, offset_path, window_start=None, window_end=None):
    """Read only the rows Simstrat appended to T_out.dat since the previous step (tail read).

    T_out.dat itself is never truncated — Simstrat keeps the full cumulative series; we just
    resume from the byte offset stored in `offset_path`.  Flexible by construction:
      * no rows-per-window assumption — works for any Simstrat output interval (hourly, daily, …);
      * works for any analysis-window length (we read whatever was appended);
      * self-healing — if the offset is missing or out of range (first run, instance dir reused,
        file shrank/rotated) it falls back to reading from just after the header;
      * optional [window_start, window_end] filter (fractional Simstrat days) as a safety net, so
        the result is the current window even when the offset had to fall back to a full re-read.

    Binary mode is used so the offset is an exact byte position.  Returns (times, depths, T_rows)
    for the new rows only.  Pure-Python parsing (no pandas): the OpenDA wrapper imports this into a
    fresh per-step subprocess, where a pandas import would cost ~2.6 s for nothing.
    """
    with open(filename, 'rb') as f:
        depths = [float(h) for h in f.readline().decode().strip().split(',')[1:]]
        data_start = f.tell()
        size = os.fstat(f.fileno()).st_size
        offset = data_start
        try:
            stored = int(open(offset_path).read().strip())
            if data_start <= stored <= size:
                offset = stored
        except (OSError, ValueError):
            pass
        f.seek(offset)
        body = f.read().decode()
        new_offset = f.tell()
    times, T_rows = [], []
    for line in body.splitlines():
        parts = line.strip().split(',')
        if not parts or not parts[0]:
            continue
        t = float(parts[0])
        if window_start is not None and t < window_start - 1e-9:
            continue
        if window_end is not None and t > window_end + 1e-9:
            continue
        times.append(t)
        T_rows.append([float(x) for x in parts[1:]])
    with open(offset_path, 'w') as f:
        f.write(str(new_offset))
    return times, depths, T_rows


def load_T_window(ensemble_dir, args, window_start, window_end):
    """Like load_T, but returns ONLY the current window's rows via a tail read of T_out.dat
    (resuming from the per-member offset sidecar) instead of re-parsing the whole accumulated
    file each window — O(1) per window vs load_T's O(n), which is O(n^2) over a long run. Shares
    read_t_out_tail with the OpenDA wrapper. Returns the same shape as load_T (datetime index,
    float depth columns) so rmse_in_window is unchanged."""
    import pandas as pd
    results_dir = args["results_dir"]
    path        = os.path.join(ensemble_dir, results_dir, "T_out.dat")
    offset_path = os.path.join(ensemble_dir, results_dir, T_OUT_OFFSET_FILE)
    ref_date    = args["ref_date"]
    # Window bounds -> fractional Simstrat days (inverse of load_T's day->datetime), as the
    # safety-net filter; the offset alone already isolates this window's appended rows.
    ws = (window_start - ref_date).total_seconds() / 86400.0
    we = (window_end   - ref_date).total_seconds() / 86400.0
    times, depths, rows = read_t_out_tail(path, offset_path, window_start=ws, window_end=we)
    cols = [float(d) for d in depths]
    if not times:
        return pd.DataFrame(columns=cols)
    idx = (pd.Timestamp(ref_date) + pd.to_timedelta(times, unit="D")).round("1h")
    df  = pd.DataFrame(rows, columns=cols, index=idx)
    df.index.name = "time"
    return df


# ---------------------------------------------------------------------------
# Per-window run machinery (Docker container lifecycle + Settings.par dates)
# ---------------------------------------------------------------------------

def _container_name(args):
    """ONE persistent container per run (was one per member). Named by algorithm + lake so
    concurrent runs of different lakes/algorithms don't clash."""
    return f"simstrat_{args['algorithm'].lower()}_{args['lake']}"


def start_containers(args, max_workers=None):
    """Start ONE persistent sleeping container for the whole run, mounting ensemble_base at the
    workdir base. Each member i is then run via `docker exec -w <workdir>/ensemble{i}` against the
    Simstrat binary directly (see run_one_window). Collapsed from one-container-per-member: all
    Simstrat paths are workdir-relative, so a single mount of the parent serves every member — far
    less startup/footprint. `max_workers` is unused here now (single container); per-member
    concurrency is the exec ThreadPool in run_window_parallel."""
    name  = _container_name(args)
    mount = args["ensemble_base"]
    if args.get("docker_dir") and args.get("repo_dir"):
        mount = os.path.join(args["docker_dir"], os.path.relpath(mount, args["repo_dir"]))
    mount = mount.replace("\\", "/")
    subprocess.run(f"docker rm -f {name}", shell=True, capture_output=True)   # drop a stale one
    cmd = (
        f"docker run -d --name {name} "
        f"-v {mount}:{args['simstrat_workdir']} "
        f"--entrypoint sleep "
        f"eawag/simstrat:{args['simstrat_version']} infinity"
    )
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"container start failed ({name}): {result.stderr.strip()}")
    logger.info(f"Started 1 persistent container ({name}) for {len(args['member_ids'])} members.")


def stop_containers(args):
    name = _container_name(args)
    subprocess.run(f"docker stop {name}", shell=True, capture_output=True)
    subprocess.run(f"docker rm   {name}", shell=True, capture_output=True)
    logger.info(f"Container stopped and removed ({name}).")


def run_one_window(i, window_start, window_end, args):
    ensemble_dir = os.path.join(args["ensemble_base"], f"ensemble{i}")
    results_dir  = os.path.join(ensemble_dir, args["results_dir"])
    os.makedirs(results_dir, exist_ok=True)

    # Note: per-window *_out.dat are NOT cleared here. Simstrat appends across windows, so the
    # output files accumulate into the full run trajectory by themselves. They are cleared once
    # up front on reset (clear_member_outputs); a non-reset run continues by appending.

    live_snap = os.path.join(results_dir, "simulation-snapshot.dat")
    if not os.path.exists(live_snap):
        dated = sorted(f for f in os.listdir(ensemble_dir) if f.startswith("simulation-snapshot_"))
        if dated:
            shutil.copy2(os.path.join(ensemble_dir, dated[-1]), live_snap)

    init_par(ensemble_dir, args)
    overwrite_par_dates(
        os.path.join(ensemble_dir, args["par_file"]),
        window_start, window_end, args["ref_date"],
    )

    name   = _container_name(args)
    # Exec the Simstrat binary directly in the member's subdir (the shared container mounts the
    # parent at simstrat_workdir). We bypass the image's /entrypoint.sh because it hardcodes
    # `cd /simstrat/run` (the mount root = ensemble_base), which would ignore -w; the entrypoint
    # otherwise only calls this same binary, so -w + the binary is equivalent and per-member-correct.
    member_workdir = f"{args['simstrat_workdir']}/ensemble{i}"
    cmd    = f"docker exec -w {member_workdir} {name} {args['simstrat_binary']} {args['par_file']}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning(f"[ensemble{i:02d}] FAILED  {window_start.date()}\n{result.stderr[-400:]}")
    return i, result.returncode


def run_window_parallel(window_start, window_end, args, max_workers=None):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(run_one_window, i, window_start, window_end, args): i
            for i in args["member_ids"]
        }
        failed = []
        for future in concurrent.futures.as_completed(futures):
            i, code = future.result()
            if code != 0:
                failed.append(i)
    return failed


# ---------------------------------------------------------------------------
# Time convention + Settings.par (JSON) read/write
# ---------------------------------------------------------------------------

def datetime_to_simstrat_time(dt, ref_date):
    delta = dt - ref_date
    return delta.days + delta.seconds / 86400


def read_ref_date(ensemble_base):
    par_path = os.path.join(ensemble_base, "ensemble1", "Settings.par")
    with open(par_path) as f:
        par = json.load(f)
    year = par["Simulation"]["Reference year"]
    return datetime(year, 1, 1, tzinfo=timezone.utc)


def init_par(ensemble_dir, args):
    src = os.path.join(ensemble_dir, "Settings.par")
    dst = os.path.join(ensemble_dir, args["par_file"])
    if os.path.exists(dst):
        return
    with open(src) as f:
        par = json.load(f)
    par["Output"]["Path"] = args["results_dir"]
    with open(dst, "w") as f:
        json.dump(par, f, indent=4)


def overwrite_par_dates(par_path, window_start, window_end, ref_date):
    # Run the exact [window_start, window_end] window. Consecutive windows share the boundary
    # instant: window N ends at, and writes its snapshot at, window_end; window N+1 continues
    # from that snapshot starting at the same instant. Simstrat emits its first output one
    # interval AFTER the start (not at t=start), so there is no duplicate row at the seam and
    # the stitched trajectory is continuous. (A former ±1h padding here created a 2h/window
    # simulation gap and was removed.)
    with open(par_path) as f:
        par = json.load(f)
    par["Simulation"]["Start d"] = datetime_to_simstrat_time(window_start, ref_date)
    par["Simulation"]["End d"]   = datetime_to_simstrat_time(window_end,   ref_date)
    with open(par_path, "w") as f:
        json.dump(par, f, indent=4)


# ---------------------------------------------------------------------------
# Snapshot temperature column read/write (full-grid T as the DA state)
# ---------------------------------------------------------------------------

# --- snapshot T-column round-trip: single source of truth for BOTH engines ------------------------
# The native engine (read_snapshot_T/write_snapshot_T, keyed by member_id) and the OpenDA wrapper
# (keyed by an explicit instance path) both read/inject the analysed temperature column. These two
# path-based helpers are that shared core; the member-id wrappers below just build the paths.

def read_snapshot_T_at(snap_path, par_path):
    """Read a snapshot's temperature column + grid geometry -> (T, z_volume, lake_level)."""
    snap  = read_snapshot(snap_path, par_path=par_path)
    T     = snap.model["T"]
    # VERIFIED: there is a vertical-alignment assumption. This takes the TOP len(T) cells of
    # z_volume, i.e. it assumes T is surface-aligned (z_volume[-1] == surface). If T is
    # bottom-aligned in the Fortran grid this silently maps every obs to the wrong depth.
    # Verified on inputs/upperlugano
    z_vol = snap.grid["z_volume"][-len(T):]
    return T.copy(), z_vol.copy(), float(snap.grid["lake_level"])


def write_snapshot_T_at(snap_path, par_path, T_new):
    """Inject an analysed temperature column into a snapshot (atomic via tmp + os.replace).
    Raises if T_new's length doesn't match the snapshot grid (caller guards/handles)."""
    snap = read_snapshot(snap_path, par_path=par_path)
    if len(T_new) != len(snap.model["T"]):
        raise ValueError(f"T length {len(T_new)} != snapshot grid {len(snap.model['T'])}")
    snap.model["T"][:] = T_new
    tmp = snap_path + ".tmp"
    write_snapshot(tmp, snap)
    os.replace(tmp, snap_path)


def _member_snapshot_paths(member_id, args):
    base = os.path.join(args["ensemble_base"], f"ensemble{member_id}")
    return (os.path.join(base, args["results_dir"], "simulation-snapshot.dat"),
            os.path.join(base, args["par_file"]))


def read_snapshot_T(member_id, args):
    """Read member `member_id`'s snapshot -> (T column, z_volume, lake_level)."""
    snap_path, par_path = _member_snapshot_paths(member_id, args)
    return read_snapshot_T_at(snap_path, par_path)


def write_snapshot_T(member_id, T_new, args):
    """Write the analysis temperature column back into member `member_id`'s snapshot."""
    snap_path, par_path = _member_snapshot_paths(member_id, args)
    write_snapshot_T_at(snap_path, par_path, T_new)


# ---------------------------------------------------------------------------
# The model: binds the functions above onto the Simstrat class.
# ---------------------------------------------------------------------------

class Simstrat:
    """The Simstrat 1D lake model, run in Docker (eawag/simstrat:<version>)."""

    name              = "simstrat"
    image             = "eawag/simstrat"
    version           = "3.0.4"
    binary            = "/simstrat/build/simstrat"   # the executable directly (image /entrypoint.sh
                                                     # is just `cd /simstrat/run; /simstrat/build/simstrat "$@"`;
                                                     # we call the binary so one container can serve all members via -w)
    workdir           = "/simstrat/run"              # ensemble_base mount point; members at workdir/ensemble{i}
    snapshot_filename = "simulation-snapshot.dat"

    def run_config(self):
        """Runtime defaults the engines read (merged into the run args). Single source of
        truth for the model's Docker invocation; the OpenDA adapter also writes the image
        (eawag/simstrat:<version>) into model.json for the standalone wrapper."""
        return {
            "simstrat_version": self.version,
            "simstrat_binary":  self.binary,
            "simstrat_workdir": self.workdir,
        }

    # input readiness / setup
    model_inputs_ready = staticmethod(model_inputs_ready)
    warmup_snapshot       = staticmethod(warmup_snapshot)
    instances_ready       = staticmethod(instances_ready)
    copy_model_inputs  = staticmethod(copy_model_inputs)
    # per-window run machinery
    start_containers      = staticmethod(start_containers)
    stop_containers       = staticmethod(stop_containers)
    run_window            = staticmethod(run_window_parallel)
    # state / output IO
    read_ref_date         = staticmethod(read_ref_date)
    model_output_depths   = staticmethod(model_output_depths)
    set_output_depths     = staticmethod(set_output_depths)
    obs_to_sim_col        = staticmethod(obs_to_sim_col)
    load_T                = staticmethod(load_T)
    load_T_window         = staticmethod(load_T_window)
    clear_member_outputs  = staticmethod(clear_member_outputs)
    accumulate_mean       = staticmethod(accumulate_mean)
    mean_traj_path        = staticmethod(mean_traj_path)
    read_snapshot_T       = staticmethod(read_snapshot_T)
    write_snapshot_T      = staticmethod(write_snapshot_T)


# ===========================================================================
# Binary snapshot I/O (simulation-snapshot.dat) — folded in from the former snapshot.py
#
# Read/write Simstrat snapshot files from Python. The snapshot is a Fortran sequential
# unformatted file produced by save_snapshot in src/simstrat.f90; this mirrors the exact
# write order so a read-then-write round trip is byte-identical. read_snapshot returns a
# Snapshot dataclass (sections are OrderedDicts mirroring the Fortran field names).
# ===========================================================================


# --- Fortran unformatted record I/O --------------------------------------
# Minimal drop-in for scipy.io.FortranFile, supporting exactly the calls the snapshot helpers make
# (read_ints / read_reals / write_record of a single array). It exists so snapshot I/O does NOT
# import scipy: `from scipy.io import FortranFile` costs ~2 s per interpreter, which the OpenDA
# wrapper paid on every member every step (a fresh process each time) — the bulk of the per-step
# "snapshot inject" storm. The native engine paid it once. numpy + struct only here.
#
# A Fortran sequential unformatted record is  [int32 nbytes][data][int32 nbytes]  with the marker in
# native byte order and 4-byte width (gfortran default) — matching scipy's defaults, so files stay
# byte-identical. read_* return writable copies (scipy semantics: callers mutate model['T'] in place).
class FortranFile:
    _MARK = np.dtype(np.uint32)   # record-length marker: uint32, native byte order (scipy default)

    def __init__(self, path, mode="r"):
        self._f = open(path, "rb" if mode == "r" else "wb")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self._f.close()

    def close(self):
        self._f.close()

    def _read_record(self) -> bytes:
        head = self._f.read(4)
        if len(head) < 4:
            raise EOFError("end of Fortran file")
        n = int(np.frombuffer(head, dtype=self._MARK)[0])
        data = self._f.read(n)
        tail = self._f.read(4)
        if len(data) != n or len(tail) < 4 or int(np.frombuffer(tail, dtype=self._MARK)[0]) != n:
            raise ValueError("corrupt Fortran record (length-marker mismatch)")
        return data

    def read_ints(self, dtype=np.int32) -> np.ndarray:
        return np.frombuffer(self._read_record(), dtype=dtype).copy()

    def read_reals(self, dtype=np.float64) -> np.ndarray:
        return np.frombuffer(self._read_record(), dtype=dtype).copy()

    def write_record(self, arr: np.ndarray) -> None:
        b = np.ascontiguousarray(arr).tobytes()
        marker = np.array(len(b), dtype=self._MARK).tobytes()
        self._f.write(marker)
        self._f.write(b)
        self._f.write(marker)


# --- low-level record helpers --------------------------------------------

def _read_array(f: FortranFile) -> Tuple[np.ndarray, Tuple[int, int]]:
    lb, ub = f.read_ints(np.int32)
    n = ub - lb + 1
    data = f.read_reals(np.float64)
    if data.size != n:
        raise ValueError(
            f"array length mismatch: bounds [{lb},{ub}] imply {n}, got {data.size}"
        )
    return data, (int(lb), int(ub))


def _write_array(f: FortranFile, data: np.ndarray, bounds: Tuple[int, int]) -> None:
    lb, ub = bounds
    if data.size != ub - lb + 1:
        raise ValueError("array length does not match bounds")
    f.write_record(np.array([lb, ub], dtype=np.int32))
    f.write_record(np.ascontiguousarray(data, dtype=np.float64))


def _read_matrix(f: FortranFile) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    lb1, ub1, lb2, ub2 = f.read_ints(np.int32)
    rows, cols = ub1 - lb1 + 1, ub2 - lb2 + 1
    flat = f.read_reals(np.float64)
    if flat.size != rows * cols:
        raise ValueError(
            f"matrix size mismatch: bounds imply {rows}x{cols}={rows*cols}, "
            f"got {flat.size}"
        )
    mat = flat.reshape((rows, cols), order="F")
    return mat, (int(lb1), int(ub1), int(lb2), int(ub2))


def _write_matrix(
    f: FortranFile, mat: np.ndarray, bounds: Tuple[int, int, int, int]
) -> None:
    lb1, ub1, lb2, ub2 = bounds
    rows, cols = ub1 - lb1 + 1, ub2 - lb2 + 1
    if mat.shape != (rows, cols):
        raise ValueError(
            f"matrix shape {mat.shape} does not match bounds {bounds}"
        )
    f.write_record(np.array([lb1, ub1, lb2, ub2], dtype=np.int32))
    f.write_record(np.asfortranarray(mat, dtype=np.float64).ravel(order="F"))


def _read_any_array(f: FortranFile):
    """Read a 1-D array (2-integer bounds) or 2-D matrix (4-integer bounds)."""
    bounds = f.read_ints(np.int32)
    data = f.read_reals(np.float64)
    if len(bounds) == 2:
        lb, ub = int(bounds[0]), int(bounds[1])
        n = ub - lb + 1
        if data.size != n:
            raise ValueError(
                f"array length mismatch: bounds [{lb},{ub}] imply {n}, got {data.size}"
            )
        return data, (lb, ub)
    elif len(bounds) == 4:
        lb1, ub1, lb2, ub2 = (int(b) for b in bounds)
        rows, cols = ub1 - lb1 + 1, ub2 - lb2 + 1
        if data.size != rows * cols:
            raise ValueError(
                f"matrix size mismatch: bounds imply {rows}x{cols}={rows*cols}, "
                f"got {data.size}"
            )
        return data.reshape((rows, cols), order="F"), (lb1, ub1, lb2, ub2)
    else:
        raise ValueError(f"Unexpected bounds record with {len(bounds)} integers")


def _write_any_array(f: FortranFile, data: np.ndarray, bounds) -> None:
    """Write a 1-D array or 2-D matrix depending on bounds length."""
    if len(bounds) == 2:
        _write_array(f, data.ravel(), bounds)
    else:
        _write_matrix(f, data, bounds)


def _read_int_array(f: FortranFile) -> Tuple[np.ndarray, Tuple[int, int]]:
    lb, ub = f.read_ints(np.int32)
    data = f.read_ints(np.int32)
    return data, (int(lb), int(ub))


def _write_int_array(f: FortranFile, data: np.ndarray, bounds: Tuple[int, int]) -> None:
    lb, ub = bounds
    f.write_record(np.array([lb, ub], dtype=np.int32))
    f.write_record(np.ascontiguousarray(data, dtype=np.int32))


def _read_logical_array(f: FortranFile) -> Tuple[np.ndarray, Tuple[int, int]]:
    # gfortran default logical is 4 bytes
    lb, ub = f.read_ints(np.int32)
    data = f.read_ints(np.int32)
    return data.astype(bool), (int(lb), int(ub))


def _write_logical_array(f: FortranFile, data: np.ndarray, bounds: Tuple[int, int]) -> None:
    lb, ub = bounds
    f.write_record(np.array([lb, ub], dtype=np.int32))
    f.write_record(np.asarray(data, dtype=bool).astype(np.int32))


# --- snapshot container --------------------------------------------------

@dataclass
class Snapshot:
    couple_aed2: bool = False
    inflow_mode: int = 0
    has_lateral_state: bool = False
    model: "OrderedDict[str, object]" = field(default_factory=OrderedDict)
    grid: "OrderedDict[str, object]" = field(default_factory=OrderedDict)
    absorption: "OrderedDict[str, object]" = field(default_factory=OrderedDict)
    lateral: "OrderedDict[str, object]" = field(default_factory=OrderedDict)
    logger: "OrderedDict[str, object]" = field(default_factory=OrderedDict)


# --- model state ---------------------------------------------------------

# Mirror src/strat_simdata.f90:370-433. Three groups of fields:
#   - 1D arrays written via save_array / save_array_pointer (bounds + data)
#   - scalar groups written as single mixed records
#   - special / conditional fields handled inline below.

_MODEL_ARRAYS_BEFORE_E_SEICHE = [
    "U", "V", "T", "S", "dS", "rho",
    "k", "ko", "avh", "eps", "num", "nuh",
    "P", "B", "NN", "cmue1", "cmue2", "P_Seiche",
]

_MODEL_SCALAR_GROUPS = [
    ("u10_v10_uv10_Wf", ["u10", "v10", "uv10", "Wf"]),
    ("u_taub_drag_u_taus_rain", ["u_taub", "drag", "u_taus", "rain"]),
    ("tx_ty", ["tx", "ty"]),
    ("C10", ["C10"]),
    ("SST_heat_group", ["SST", "heat", "heat_snow", "heat_ice", "heat_snowice"]),
    ("T_atm", ["T_atm"]),
]

_MODEL_SCALAR_SOLO_AFTER_LAT = [
    "snow_h", "total_ice_h", "black_ice_h", "white_ice_h",
    "snow_dens", "ice_temp", "precip",
    "ha", "hw", "hk", "hv", "rad0",
]


def _read_model_state(f: FortranFile, snap: Snapshot) -> None:
    m = snap.model

    for name in _MODEL_ARRAYS_BEFORE_E_SEICHE:
        m[name], m[f"{name}_bounds"] = _read_array(f)

    e_seiche, gamma = f.read_reals(np.float64)
    m["E_Seiche"] = float(e_seiche)
    m["gamma"] = float(gamma)

    m["absorb"], m["absorb_bounds"] = _read_array(f)
    m["absorb_vol"], m["absorb_vol_bounds"] = _read_array(f)

    for group_name, fields in _MODEL_SCALAR_GROUPS:
        vals = f.read_reals(np.float64)
        if vals.size != len(fields):
            raise ValueError(
                f"{group_name}: expected {len(fields)} reals, got {vals.size}"
            )
        for name, v in zip(fields, vals):
            m[name] = float(v)

    m["rad"], m["rad_bounds"] = _read_array(f)

    # albedo_data is a fixed-shape 9x12 matrix written without explicit bounds
    flat = f.read_reals(np.float64)
    if flat.size != 9 * 12:
        raise ValueError(f"albedo_data: expected 108 reals, got {flat.size}")
    m["albedo_data"] = flat.reshape((9, 12), order="F")

    m["albedo_water"] = float(f.read_reals(np.float64)[0])
    m["lat_number"] = int(f.read_ints(np.int32)[0])

    for name in _MODEL_SCALAR_SOLO_AFTER_LAT:
        m[name] = float(f.read_reals(np.float64)[0])

    cde, cm0 = f.read_reals(np.float64)
    m["cde"] = float(cde)
    m["cm0"] = float(cm0)
    m["fsed"] = float(f.read_reals(np.float64)[0])

    m["fgeo_add"], m["fgeo_add_bounds"] = _read_array(f)

    if snap.couple_aed2:
        m["AED2_state"], m["AED2_state_bounds"] = _read_any_array(f)
        m["AED2_diagnostic"], m["AED2_diagnostic_bounds"] = _read_any_array(f)
        m["AED2_diagnostic_sheet"], m["AED2_diagnostic_sheet_bounds"] = _read_any_array(f)

    if snap.inflow_mode > 0:
        m["Q_inp"], m["Q_inp_bounds"] = _read_matrix(f)
        m["Q_vert"], m["Q_vert_bounds"] = _read_array(f)


def _write_model_state(f: FortranFile, snap: Snapshot) -> None:
    m = snap.model

    for name in _MODEL_ARRAYS_BEFORE_E_SEICHE:
        _write_array(f, m[name], m[f"{name}_bounds"])

    f.write_record(np.array([m["E_Seiche"], m["gamma"]], dtype=np.float64))

    _write_array(f, m["absorb"], m["absorb_bounds"])
    _write_array(f, m["absorb_vol"], m["absorb_vol_bounds"])

    for _, fields in _MODEL_SCALAR_GROUPS:
        f.write_record(np.array([m[k] for k in fields], dtype=np.float64))

    _write_array(f, m["rad"], m["rad_bounds"])

    f.write_record(
        np.asfortranarray(m["albedo_data"], dtype=np.float64).ravel(order="F")
    )
    f.write_record(np.array([m["albedo_water"]], dtype=np.float64))
    f.write_record(np.array([m["lat_number"]], dtype=np.int32))

    for name in _MODEL_SCALAR_SOLO_AFTER_LAT:
        f.write_record(np.array([m[name]], dtype=np.float64))

    f.write_record(np.array([m["cde"], m["cm0"]], dtype=np.float64))
    f.write_record(np.array([m["fsed"]], dtype=np.float64))

    _write_array(f, m["fgeo_add"], m["fgeo_add_bounds"])

    if snap.couple_aed2:
        _write_any_array(f, m["AED2_state"], m["AED2_state_bounds"])
        _write_any_array(f, m["AED2_diagnostic"], m["AED2_diagnostic_bounds"])
        _write_any_array(f, m["AED2_diagnostic_sheet"], m["AED2_diagnostic_sheet_bounds"])

    if snap.inflow_mode > 0:
        _write_matrix(f, m["Q_inp"], m["Q_inp_bounds"])
        _write_array(f, m["Q_vert"], m["Q_vert_bounds"])


# --- grid ----------------------------------------------------------------

# Mirror src/strat_grid.f90:110-129
_GRID_ARRAYS_BEFORE_VOLUME = ["h", "z_face", "z_volume", "Az", "dAz", "meanint"]
_GRID_ARRAYS_AFTER_VOLUME = [
    "AreaFactor_1", "AreaFactor_2",
    "AreaFactor_k1", "AreaFactor_k2", "AreaFactor_eps",
]


def _read_grid(f: FortranFile, snap: Snapshot) -> None:
    g = snap.grid
    for name in _GRID_ARRAYS_BEFORE_VOLUME:
        g[name], g[f"{name}_bounds"] = _read_any_array(f)
    volume, h_old = f.read_reals(np.float64)
    g["volume"] = float(volume)
    g["h_old"] = float(h_old)
    for name in _GRID_ARRAYS_AFTER_VOLUME:
        g[name], g[f"{name}_bounds"] = _read_any_array(f)
    nz_grid, nz_occupied, max_input = f.read_ints(np.int32)
    g["nz_grid"] = int(nz_grid)
    g["nz_occupied"] = int(nz_occupied)
    g["max_length_input_data"] = int(max_input)
    ubnd_vol, ubnd_fce, length_vol, length_fce = f.read_ints(np.int32)
    g["ubnd_vol"] = int(ubnd_vol)
    g["ubnd_fce"] = int(ubnd_fce)
    g["length_vol"] = int(length_vol)
    g["length_fce"] = int(length_fce)
    z_zero, lake_level, lake_level_old, max_depth = f.read_reals(np.float64)
    g["z_zero"] = float(z_zero)
    g["lake_level"] = float(lake_level)
    g["lake_level_old"] = float(lake_level_old)
    g["max_depth"] = float(max_depth)


def _write_grid(f: FortranFile, snap: Snapshot) -> None:
    g = snap.grid
    for name in _GRID_ARRAYS_BEFORE_VOLUME:
        _write_any_array(f, g[name], g[f"{name}_bounds"])
    f.write_record(np.array([g["volume"], g["h_old"]], dtype=np.float64))
    for name in _GRID_ARRAYS_AFTER_VOLUME:
        _write_any_array(f, g[name], g[f"{name}_bounds"])
    f.write_record(np.array(
        [g["nz_grid"], g["nz_occupied"], g["max_length_input_data"]],
        dtype=np.int32,
    ))
    f.write_record(np.array(
        [g["ubnd_vol"], g["ubnd_fce"], g["length_vol"], g["length_fce"]],
        dtype=np.int32,
    ))
    f.write_record(np.array(
        [g["z_zero"], g["lake_level"], g["lake_level_old"], g["max_depth"]],
        dtype=np.float64,
    ))


# --- absorption ----------------------------------------------------------

# Mirror src/strat_absorption.f90:81-91
def _read_absorption(f: FortranFile, snap: Snapshot) -> None:
    a = snap.absorption
    a["number_of_lines_read"] = int(f.read_ints(np.int32)[0])
    tb_start, tb_end = f.read_reals(np.float64)
    a["tb_start"] = float(tb_start)
    a["tb_end"] = float(tb_end)
    eof, nval = f.read_ints(np.int32)
    a["eof"] = int(eof)
    a["nval"] = int(nval)
    a["z_absorb"], a["z_absorb_bounds"] = _read_array(f)
    a["absorb_start"], a["absorb_start_bounds"] = _read_array(f)
    a["absorb_end"], a["absorb_end_bounds"] = _read_array(f)


def _write_absorption(f: FortranFile, snap: Snapshot) -> None:
    a = snap.absorption
    f.write_record(np.array([a["number_of_lines_read"]], dtype=np.int32))
    f.write_record(np.array([a["tb_start"], a["tb_end"]], dtype=np.float64))
    f.write_record(np.array([a["eof"], a["nval"]], dtype=np.int32))
    _write_array(f, a["z_absorb"], a["z_absorb_bounds"])
    _write_array(f, a["absorb_start"], a["absorb_start_bounds"])
    _write_array(f, a["absorb_end"], a["absorb_end_bounds"])


# --- lateral (optional) --------------------------------------------------

# Mirror src/strat_lateral.f90:154-185
_LATERAL_INT_ARRAYS = [
    "number_of_lines_read", "eof", "nval", "nval_deep", "nval_surface", "fnum",
]
_LATERAL_LOGICAL_ARRAYS = ["has_surface_input", "has_deep_input"]
_LATERAL_REAL_ARRAYS = ["tb_start", "tb_end"]
_LATERAL_MATRICES = [
    "z_Inp", "Q_start", "Qs_start", "Q_end", "Qs_end",
    "Q_read_start", "Q_read_end",
    "Inp_read_start", "Inp_read_end",
    "Qs_read_start", "Qs_read_end",
]


def _read_lateral(f: FortranFile, snap: Snapshot) -> None:
    lat = snap.lateral
    has = f.read_ints(np.int32)
    lat["has_allocated"] = bool(has[0])
    for name in _LATERAL_INT_ARRAYS:
        lat[name], lat[f"{name}_bounds"] = _read_int_array(f)
    for name in _LATERAL_LOGICAL_ARRAYS:
        lat[name], lat[f"{name}_bounds"] = _read_logical_array(f)
    for name in _LATERAL_REAL_ARRAYS:
        lat[name], lat[f"{name}_bounds"] = _read_array(f)
    for name in _LATERAL_MATRICES:
        lat[name], lat[f"{name}_bounds"] = _read_matrix(f)


def _write_lateral(f: FortranFile, snap: Snapshot) -> None:
    lat = snap.lateral
    f.write_record(np.array(
        [1 if lat.get("has_allocated", True) else 0], dtype=np.int32,
    ))
    for name in _LATERAL_INT_ARRAYS:
        _write_int_array(f, lat[name], lat[f"{name}_bounds"])
    for name in _LATERAL_LOGICAL_ARRAYS:
        _write_logical_array(f, lat[name], lat[f"{name}_bounds"])
    for name in _LATERAL_REAL_ARRAYS:
        _write_array(f, lat[name], lat[f"{name}_bounds"])
    for name in _LATERAL_MATRICES:
        _write_matrix(f, lat[name], lat[f"{name}_bounds"])


# --- logger --------------------------------------------------------------

def _read_logger(f: FortranFile, snap: Snapshot) -> None:
    snap.logger["last_iteration_data"], snap.logger["last_iteration_data_bounds"] = (
        _read_matrix(f)
    )


def _write_logger(f: FortranFile, snap: Snapshot) -> None:
    _write_matrix(
        f,
        snap.logger["last_iteration_data"],
        snap.logger["last_iteration_data_bounds"],
    )


# --- top-level read/write ------------------------------------------------

def flags_from_par(par_path: str) -> dict:
    """Read CoupleAED2 and InflowMode from a Simstrat .par (JSON) config.

    Returns a dict with `couple_aed2`, `inflow_mode`, and `has_lateral_state`
    suitable for splatting into `read_snapshot(...)`. `has_lateral_state` is
    set to `inflow_mode > 0` since the lateral block is normally allocated
    whenever inflow is enabled; override explicitly if your run differs.
    """
    with open(par_path) as fh:
        cfg = json.load(fh)
    mc = cfg.get("ModelConfig", {})
    couple_aed2 = bool(mc.get("CoupleAED2", False))
    inflow_mode = int(mc.get("InflowMode", 0))
    return {
        "couple_aed2": couple_aed2,
        "inflow_mode": inflow_mode,
        "has_lateral_state": inflow_mode > 0,
    }


def read_snapshot(
    path: str,
    couple_aed2: bool = False,
    inflow_mode: int = 0,
    has_lateral_state: bool = False,
    par_path: Optional[str] = None,
) -> Snapshot:
    """Read a Simstrat snapshot file into a Snapshot dataclass.

    If `par_path` is given, all three flags are taken from the .par file and
    the explicit flag args are ignored. To override par-derived values, call
    `flags_from_par()`, edit the dict, and pass without `par_path`.
    """
    if par_path is not None:
        flags = flags_from_par(par_path)
        couple_aed2 = flags["couple_aed2"]
        inflow_mode = flags["inflow_mode"]
        has_lateral_state = flags["has_lateral_state"]
    snap = Snapshot(
        couple_aed2=couple_aed2,
        inflow_mode=inflow_mode,
        has_lateral_state=has_lateral_state,
    )
    with FortranFile(path, "r") as f:
        _read_model_state(f, snap)
        _read_grid(f, snap)
        _read_absorption(f, snap)
        if inflow_mode > 0 and has_lateral_state:
            _read_lateral(f, snap)
        _read_logger(f, snap)
    return snap


def write_snapshot(path: str, snap: Snapshot) -> None:
    """Write a Snapshot back to a Fortran unformatted file."""
    with FortranFile(path, "w") as f:
        _write_model_state(f, snap)
        _write_grid(f, snap)
        _write_absorption(f, snap)
        if snap.inflow_mode > 0 and snap.has_lateral_state:
            _write_lateral(f, snap)
        _write_logger(f, snap)

"""All Simstrat-specific model behaviour, in one place.

Simstrat is a 1D lake model run in Docker (`eawag/simstrat:<version>`). This module
holds everything that knows about Simstrat's mechanics — Docker container lifecycle,
`Settings.par` editing, the day-since-reference-year time convention, the output
formats (`z_out.dat` / `T_out.dat`), the input layout, and the binary snapshot I/O
(`read_snapshot`/`write_snapshot`, at the bottom of this module) — and exposes it
through the `Simstrat` class.

The engines, the forcing perturbation, the run summary and the pipeline call models only
through the `ForwardModel` interface (`models/base.py`), selecting one via `get_model()` in
`models/__init__.py`. This module is the ONLY place in the package (besides the
Simstrat-specific `openda/` engine) that knows Simstrat's files.

The functions below keep their original signatures (taking the pipeline's `args`/`raw`
dicts) and are bound onto `Simstrat` as static methods at the bottom.
"""

import os
import glob
import json
import shutil
import logging
import subprocess
import concurrent.futures
from datetime import datetime, timezone

import numpy as np
# NOTE: pandas is imported lazily inside the functions that need it (accumulate_mean, load_T).
# The OpenDA wrapper imports read_snapshot/write_snapshot from this module 21×/step and those use
# only numpy; a top-level `import pandas` cost ~2.6s per process launch for nothing. (numpy ~0.7s.)

from ...containers import start_persistent_container, stop_persistent_container
from ..base import ForwardModel
from .snapshot import read_snapshot_T_at, write_snapshot_T_at

logger = logging.getLogger(__name__)

# Simstrat owns these: the day-since-reference-year epoch of Forcing.dat / T_out.dat and the
# Forcing.dat header line.
SIMSTRAT_REF_YEAR = 1981
FORCING_HEADER    = ("Time [d]    u [m/s]    v [m/s]  Tair [°C] sol [W/m2] vap [mbar]  "
                     "cloud [-] rain [m/hr]")
# Forcing.dat columns, named by the channels the perturbation reads (U, V, GLOB) and passes through.
FORCING_COLUMNS   = ["time_days", "U", "V", "T", "GLOB", "vap", "cloud", "rain"]


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

    lake          = raw["lake"]
    n_members     = raw["n_members"]
    ensemble_base = raw["ensemble_base"]
    model_inputs  = raw["model_inputs_path"]
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
# Forcing.dat (read for the AR(1) perturbation, written per member)
# ---------------------------------------------------------------------------

def read_forcing(model_inputs, ref_year=None):
    """The control Forcing.dat as a frame: FORCING_COLUMNS, plus 'time' (UTC, rounded to the hour)
    and 'step' (int64 seconds since the Simstrat epoch -- the row's absolute key for the
    time-keyed perturbation noise and its persisted chain state).

    Forcing.dat time_days is 0-based (day 0 = ref_year Jan 1), the SAME axis as the par's "Start d"
    and T_out's "Datetime"."""
    import pandas as pd
    ref_year = SIMSTRAT_REF_YEAR if ref_year is None else ref_year
    t0  = pd.Timestamp(f"{ref_year}-01-01")
    df  = pd.read_csv(os.path.join(model_inputs, "Forcing.dat"), sep=r"\s+",
                      names=FORCING_COLUMNS, skiprows=1)
    df["time"] = (t0 + pd.to_timedelta(df["time_days"], unit="D")).dt.round("h").dt.tz_localize("UTC")
    df["step"] = np.rint(df["time_days"].values * 86400).astype(np.int64)
    return df


def write_forcing(member_dir, df):
    """Write a frame shaped like read_forcing's into <member_dir>/Forcing.dat."""
    rows = np.column_stack([
        df["time_days"].values,
        df["U"].values,
        df["V"].values,
        df["T"].values,
        df["GLOB"].values,
        df["vap"].fillna(0).values,
        df["cloud"].fillna(0).values,
        df["rain"].fillna(0).values,
    ])
    np.savetxt(os.path.join(member_dir, "Forcing.dat"),
               rows, fmt="%10.4f", header=FORCING_HEADER, comments="")


# ---------------------------------------------------------------------------
# Output formats (z_out.dat / T_out.dat)
# ---------------------------------------------------------------------------

def output_epoch():
    """Origin of T_out.dat's "Datetime" axis."""
    return datetime(SIMSTRAT_REF_YEAR, 1, 1, tzinfo=timezone.utc)


def member_output_file(ensemble_base, member_id, results_dir):
    """A member's accumulated T_out.dat."""
    return os.path.join(ensemble_base, f"ensemble{member_id}", results_dir, "T_out.dat")


def read_output(path):
    """T_out.dat -> (times in days since output_epoch, column heights, temps[time, depth])."""
    import pandas as pd
    df = pd.read_csv(path)
    times  = df.iloc[:, 0].to_numpy(dtype=float)
    depths = np.array([float(c) for c in df.columns[1:]])
    temps  = df.iloc[:, 1:].to_numpy(dtype=float)   # [time, depth]
    return times, depths, temps


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


def _extend_depths(existing, obs):
    """`existing` output depths plus the observation depths inside the grid (<= its bed)."""
    deepest = max(existing)
    return set(existing) | {o for o in obs if o <= deepest}


def set_output_depths(model_inputs, ensemble_base, n_members, obs_depths):
    """Overwrite z_out.dat in every member (ensemble0..N) with the superset of its current output
    depths and the observation depths, so no observation is dropped just for lack of a matching
    model-output depth (depths are matched at 0.01 m). Observations deeper than the existing grid's
    bed are left out (outside the simulated water column). No-op where z_out.dat is absent.

    `model_inputs` is only READ: the extended grid is also written to <ensemble_base>/z_out.dat, and
    its path returned, so a host that wants its own deterministic runs to output the observation
    depths can copy it explicitly. Returns (n_member_files_rewritten, extended_z_out_path or None)."""
    obs = {round(abs(d), 2) for d in obs_depths}
    rewritten = 0
    for i in range(n_members + 1):
        path = os.path.join(ensemble_base, f"ensemble{i}", "z_out.dat")
        existing = _read_z_out(path)
        if not existing:
            continue                      # absent/empty: leave Simstrat's default output grid
        union = _extend_depths(existing, obs)
        if union != set(existing):
            _write_z_out(path, union)
            rewritten += 1
    source = _read_z_out(os.path.join(model_inputs, "z_out.dat"))
    if not source:
        return rewritten, None
    extended = os.path.join(ensemble_base, "z_out.dat")
    os.makedirs(ensemble_base, exist_ok=True)
    _write_z_out(extended, _extend_depths(source, obs))
    return rewritten, extended


def obs_to_sim_col(depth):
    """Map an observation depth ("X m below surface", positive) to the Simstrat output/state
    column coordinate (height -X; negative = below surface). The engines build H / score obs
    against these coordinates, so the convention is the model's."""
    return -depth


def sim_col_to_depth(col):
    """Inverse of obs_to_sim_col: Simstrat height (negative below surface) -> positive depth."""
    return -col


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


def accumulate_mean(member_files, mean_path):
    """Write the ensemble-mean trajectory to `mean_path` from the members' (full,
    accumulated) T_out.dat files. One-shot: call once after the run. Averages across
    whatever members are present at each timestamp, so it tolerates a member missing a
    failed window."""
    import pandas as pd
    def _read(path):
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path)
        df.columns = [c.strip().strip('"') for c in df.columns]
        return df

    with concurrent.futures.ThreadPoolExecutor() as pool:
        frames = [f for f in pool.map(_read, member_files) if f is not None]
    if not frames:
        return
    time_col = frames[0].columns[0]
    mean_df  = pd.concat(frames).groupby(time_col, as_index=False).mean()
    mean_df.to_csv(mean_path, index=False)


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
    name = _container_name(args)
    start_persistent_container(args, name, args["ensemble_base"], args["simstrat_workdir"],
                               f"eawag/simstrat:{args['simstrat_version']}")
    logger.info(f"Started 1 persistent container ({name}) for {len(args['member_ids'])} members.")


def stop_containers(args):
    name = _container_name(args)
    stop_persistent_container(name)
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


def clear_member_state(ensemble_base, member_ids, results_dir):
    """Delete each member's live snapshot, so its next window restarts from the dated warm start."""
    for i in member_ids:
        live = os.path.join(ensemble_base, f"ensemble{i}", results_dir, "simulation-snapshot.dat")
        if os.path.exists(live):
            os.remove(live)


def copy_member_state(src_id, member_ids, args):
    """Copy member `src_id`'s live snapshot over every other member's (PF best-member resampling)."""
    def _snap(i):
        return os.path.join(args["ensemble_base"], f"ensemble{i}", args["results_dir"],
                            "simulation-snapshot.dat")
    src     = _snap(src_id)
    targets = [_snap(i) for i in member_ids if i != src_id]
    with concurrent.futures.ThreadPoolExecutor() as pool:
        pool.map(lambda dst: shutil.copy2(src, dst), targets)


def _analysis_member_paths(member_id, cfg):
    """Member analysis snapshot + par file, resolving the engine's layout: native members live at
    ensemble{i}/<results_dir>/, OpenDA members at <openda_dir>/Results/work{i}/."""
    if cfg.get("engine") == "openda":
        work = os.path.join(cfg["openda_dir"], "Results", f"work{member_id}")
        return (os.path.join(work, "Results", "simulation-snapshot.dat"),
                os.path.join(work, "Settings.par"))
    return _member_snapshot_paths(member_id, cfg)


def write_mean_state(cfg, out_path):
    """The deterministic warm start for a forecast: the ensemble-mean analysis. Average the
    temperature column over members 1..N (the assimilated members) and inject it into a copy of
    member 1's analysis snapshot, written to `out_path`."""
    cols = []
    for i in range(1, cfg["n_members"] + 1):
        snap, par = _analysis_member_paths(i, cfg)
        cols.append(read_snapshot_T_at(snap, par)[0])
    mean_T = np.mean(np.vstack(cols), axis=0)

    member_snap, member_par = _analysis_member_paths(1, cfg)
    shutil.copy(member_snap, out_path)
    write_snapshot_T_at(out_path, member_par, mean_T)


# ---------------------------------------------------------------------------
# The model: binds the functions above onto the Simstrat class.
# ---------------------------------------------------------------------------

class Simstrat(ForwardModel):
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
    model_inputs_ready    = staticmethod(model_inputs_ready)
    warmup_snapshot       = staticmethod(warmup_snapshot)
    instances_ready       = staticmethod(instances_ready)
    copy_model_inputs     = staticmethod(copy_model_inputs)
    # forcing
    read_forcing          = staticmethod(read_forcing)
    write_forcing         = staticmethod(write_forcing)
    # per-window run machinery
    start_containers      = staticmethod(start_containers)
    stop_containers       = staticmethod(stop_containers)
    run_window            = staticmethod(run_window_parallel)
    # state / output IO
    read_ref_date         = staticmethod(read_ref_date)
    model_output_depths   = staticmethod(model_output_depths)
    set_output_depths     = staticmethod(set_output_depths)
    obs_to_sim_col        = staticmethod(obs_to_sim_col)
    sim_col_to_depth      = staticmethod(sim_col_to_depth)
    load_T                = staticmethod(load_T)
    load_T_window         = staticmethod(load_T_window)
    clear_member_outputs  = staticmethod(clear_member_outputs)
    accumulate_mean       = staticmethod(accumulate_mean)
    mean_traj_path        = staticmethod(mean_traj_path)
    read_snapshot_T       = staticmethod(read_snapshot_T)
    write_snapshot_T      = staticmethod(write_snapshot_T)
    clear_member_state    = staticmethod(clear_member_state)
    copy_member_state     = staticmethod(copy_member_state)
    write_mean_state      = staticmethod(write_mean_state)
    # output tables
    output_epoch          = staticmethod(output_epoch)
    member_output_file    = staticmethod(member_output_file)
    read_output           = staticmethod(read_output)

    def openda_binding(self):
        """Simstrat as an OpenDA black box (imported here: the OpenDA wrapper imports this module
        once per member per analysis and does not need it)."""
        from .openda import SimstratOpenDA
        return SimstratOpenDA()



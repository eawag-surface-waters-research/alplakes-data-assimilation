"""The Simstrat model adapter: everything the engines need to run Simstrat members, as the
ForwardModel interface (models/base.py). Simstrat runs in Docker (eawag/simstrat:<version>).

pandas is imported inside the functions that need it: the OpenDA wrapper imports this package
once per member per analysis, and does not need it.
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

from ...containers import start_persistent_container, stop_persistent_container
from ..base import ForwardModel
from .snapshot import read_snapshot_T_at, write_snapshot_T_at

logger = logging.getLogger(__name__)

# Day 0 of Forcing.dat, T_out.dat and the par file's "Start d"/"End d".
SIMSTRAT_REF_YEAR = 1981
FORCING_HEADER    = ("Time [d]    u [m/s]    v [m/s]  Tair [°C] sol [W/m2] vap [mbar]  "
                     "cloud [-] rain [m/hr]")
FORCING_COLUMNS   = ["time_days", "U", "V", "T", "GLOB", "vap", "cloud", "rain"]
SNAPSHOT          = "simulation-snapshot.dat"
# Next to T_out.dat: how far it has been read, so each window reads only the rows it appended.
T_OUT_OFFSET_FILE = ".t_out_read_offset"
# Not copied into the members: the members write their own outputs.
COPY_DEFAULT_SKIP = {"Results", "ref"}


def model_inputs_ready(model_inputs):
    return (bool(glob.glob(os.path.join(model_inputs, "simulation-snapshot_*.dat")))
            and os.path.isfile(os.path.join(model_inputs, "Forcing.dat")))


def warmup_snapshot(model_inputs):
    """The newest dated snapshot, simulation-snapshot_<YYYYMMDD>.dat, or None."""
    snaps = sorted(glob.glob(os.path.join(model_inputs, "simulation-snapshot_*.dat")))
    return snaps[-1] if snaps else None


def instances_ready(ensemble_base, n_members):
    return all(os.path.isfile(os.path.join(ensemble_base, f"ensemble{i}", "Settings.par"))
               for i in range(n_members + 1))


def _copy_dir(src_dir, dst_dir, skip):
    """Copy src_dir's entries into dst_dir, replacing them; other files in dst_dir are kept."""
    os.makedirs(dst_dir, exist_ok=True)
    for name in os.listdir(src_dir):
        if name in skip:
            continue
        src, dst = os.path.join(src_dir, name), os.path.join(dst_dir, name)
        if os.path.isfile(src):
            shutil.copy2(src, dst)
        elif os.path.isdir(src):
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)


def copy_model_inputs(cfg):
    """Copy the model inputs into ensemble0..N (0 is the unperturbed control)."""
    model_inputs = cfg["model_inputs_path"]
    if not os.path.isdir(model_inputs):
        raise FileNotFoundError(f"model_inputs not found: {model_inputs} "
                                f"(the Simstrat inputs with a dated simulation-snapshot_*.dat)")
    skip = set(cfg.get("copy_skip", COPY_DEFAULT_SKIP))
    for i in range(cfg["n_members"] + 1):
        _copy_dir(model_inputs, os.path.join(cfg["ensemble_base"], f"ensemble{i}"), skip)
    logger.info(f"{cfg['lake']}: copied model_inputs -> ensemble0..{cfg['n_members']} "
                f"under {cfg['ensemble_base']} (skipped: {sorted(skip)})")


def read_forcing(model_inputs, ref_year=None):
    """The control Forcing.dat as FORCING_COLUMNS plus 'time' (UTC, to the hour) and 'step' (int
    seconds since day 0), the row's key for its perturbation noise."""
    import pandas as pd
    t0 = pd.Timestamp(f"{SIMSTRAT_REF_YEAR if ref_year is None else ref_year}-01-01")
    df = pd.read_csv(os.path.join(model_inputs, "Forcing.dat"), sep=r"\s+", names=FORCING_COLUMNS, skiprows=1)
    df["time"] = (t0 + pd.to_timedelta(df["time_days"], unit="D")).dt.round("h").dt.tz_localize("UTC")
    df["step"] = np.rint(df["time_days"].values * 86400).astype(np.int64)
    return df


def write_forcing(member_dir, df):
    rows = np.column_stack([df["time_days"].values, df["U"].values, df["V"].values, df["T"].values,
                            df["GLOB"].values, df["vap"].fillna(0).values, df["cloud"].fillna(0).values,
                            df["rain"].fillna(0).values])
    np.savetxt(os.path.join(member_dir, "Forcing.dat"), rows, fmt="%10.4f", header=FORCING_HEADER, comments="")


def output_epoch():
    return datetime(SIMSTRAT_REF_YEAR, 1, 1, tzinfo=timezone.utc)


def member_output_file(ensemble_base, member_id, results_dir):
    return os.path.join(ensemble_base, f"ensemble{member_id}", results_dir, "T_out.dat")


def read_output(path):
    """T_out.dat -> (days since output_epoch, column heights, temperatures[time, column])."""
    import pandas as pd
    df = pd.read_csv(path)
    return (df.iloc[:, 0].to_numpy(dtype=float), np.array([float(c) for c in df.columns[1:]]),
            df.iloc[:, 1:].to_numpy(dtype=float))


def read_z_out(path):
    """The depths (positive metres) in a z_out.dat, [] if there is none."""
    if not os.path.isfile(path):
        return []
    depths = []
    with open(path) as f:
        for line in f:
            try:
                depths.append(abs(float(line)))
            except ValueError:
                continue                      # the "Depths [m]" header
    return depths


def write_z_out(path, depths):
    with open(path, "w") as f:
        f.write("Depths [m]\n")
        for z in sorted(depths):
            f.write(f"{-z:.2f}\n")            # negative below the surface, surface first


def model_output_depths(ensemble_base):
    return read_z_out(os.path.join(ensemble_base, "ensemble1", "z_out.dat"))


def extend_depths(existing, obs):
    """The output depths plus the observation depths above the deepest of them (the bed)."""
    return set(existing) | {o for o in obs if o <= max(existing)}


def set_output_depths(model_inputs, ensemble_base, n_members, obs_depths):
    """Add the observation depths (to 0.01 m) to every member's z_out.dat, so no observation lacks
    a model output to compare with. The model inputs are only read: the extended grid is also
    written to <ensemble_base>/z_out.dat for the host to adopt. Returns (member files rewritten,
    that path or None)."""
    obs = {round(abs(d), 2) for d in obs_depths}
    rewritten = 0
    for i in range(n_members + 1):
        path = os.path.join(ensemble_base, f"ensemble{i}", "z_out.dat")
        existing = read_z_out(path)
        if existing and extend_depths(existing, obs) != set(existing):
            write_z_out(path, extend_depths(existing, obs))
            rewritten += 1
    source = read_z_out(os.path.join(model_inputs, "z_out.dat"))
    if not source:
        return rewritten, None
    extended = os.path.join(ensemble_base, "z_out.dat")
    os.makedirs(ensemble_base, exist_ok=True)
    write_z_out(extended, extend_depths(source, obs))
    return rewritten, extended


def obs_to_sim_col(depth):
    """Observation depth (positive metres) -> Simstrat's column coordinate (negative height)."""
    return -depth


def sim_col_to_depth(col):
    return -col


def clear_member_outputs(ensemble_base, member_ids, results_dir):
    """Delete the members' *_out.dat: Simstrat appends to them window after window."""
    for i in member_ids:
        rdir = os.path.join(ensemble_base, f"ensemble{i}", results_dir)
        if os.path.isdir(rdir):
            for name in os.listdir(rdir):
                if name.endswith("_out.dat") or name == T_OUT_OFFSET_FILE:
                    os.remove(os.path.join(rdir, name))


def accumulate_mean(member_files, mean_path):
    """Write the members' mean trajectory, over the members present at each time."""
    import pandas as pd

    def read(path):
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path)
        df.columns = [c.strip().strip('"') for c in df.columns]
        return df

    with concurrent.futures.ThreadPoolExecutor() as pool:
        frames = [f for f in pool.map(read, member_files) if f is not None]
    if frames:
        time_col = frames[0].columns[0]
        pd.concat(frames).groupby(time_col, as_index=False).mean().to_csv(mean_path, index=False)


def mean_traj_path(ensemble_base, algorithm):
    return os.path.join(ensemble_base, f"T_out_{algorithm.lower()}_mean.dat")


def _read_t_out_tail(path, offset_path, first_day, last_day):
    """The rows of T_out.dat appended since the last read and within [first_day, last_day]:
    (days, column heights, rows). Starts over after the header when the offset is missing or
    out of range."""
    with open(path, "rb") as f:
        columns = [float(h) for h in f.readline().decode().strip().split(",")[1:]]
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
    days, rows = [], []
    for line in body.splitlines():
        parts = line.strip().split(",")
        if not parts[0]:
            continue
        t = float(parts[0])
        if first_day - 1e-9 <= t <= last_day + 1e-9:
            days.append(t)
            rows.append([float(x) for x in parts[1:]])
    with open(offset_path, "w") as f:
        f.write(str(new_offset))
    return days, columns, rows


def load_T_window(ensemble_dir, args, window_start, window_end):
    import pandas as pd
    results = os.path.join(ensemble_dir, args["results_dir"])
    ref_date = args["ref_date"]
    days, columns, rows = _read_t_out_tail(os.path.join(results, "T_out.dat"),
                                           os.path.join(results, T_OUT_OFFSET_FILE),
                                           (window_start - ref_date).total_seconds() / 86400.0,
                                           (window_end - ref_date).total_seconds() / 86400.0)
    if not days:
        return pd.DataFrame(columns=columns)
    df = pd.DataFrame(rows, columns=columns,
                      index=(pd.Timestamp(ref_date) + pd.to_timedelta(days, unit="D")).round("1h"))
    df.index.name = "time"
    return df


def _container_name(args):
    return f"simstrat_{args['algorithm'].lower()}_{args['lake']}"


def start_containers(args, max_workers=None):
    """One container for all members (each runs in its own subdir of the mount)."""
    name = _container_name(args)
    start_persistent_container(args, name, args["ensemble_base"], args["simstrat_workdir"],
                               f"eawag/simstrat:{args['simstrat_version']}")
    logger.info(f"Started 1 persistent container ({name}) for {len(args['member_ids'])} members.")


def stop_containers(args):
    stop_persistent_container(_container_name(args))
    logger.info(f"Container stopped and removed ({_container_name(args)}).")


def run_one_window(i, window_start, window_end, args):
    ensemble_dir = os.path.join(args["ensemble_base"], f"ensemble{i}")
    results_dir  = os.path.join(ensemble_dir, args["results_dir"])
    os.makedirs(results_dir, exist_ok=True)
    live_snap = os.path.join(results_dir, SNAPSHOT)
    if not os.path.exists(live_snap):
        dated = sorted(f for f in os.listdir(ensemble_dir) if f.startswith("simulation-snapshot_"))
        if dated:
            shutil.copy2(os.path.join(ensemble_dir, dated[-1]), live_snap)
    init_par(ensemble_dir, args)
    overwrite_par_dates(os.path.join(ensemble_dir, args["par_file"]), window_start, window_end, args["ref_date"])

    # The binary itself, not the image's entrypoint, which would cd to the mount root.
    cmd = (f"docker exec -w {args['simstrat_workdir']}/ensemble{i} {_container_name(args)} "
           f"{args['simstrat_binary']} {args['par_file']}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        logger.warning(f"[ensemble{i:02d}] FAILED  {window_start.date()}\n{result.stderr[-400:]}")
    return i, result.returncode


def run_window_parallel(window_start, window_end, args, max_workers=None):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(run_one_window, i, window_start, window_end, args) for i in args["member_ids"]]
        return [i for i, code in (f.result() for f in concurrent.futures.as_completed(futures)) if code != 0]


def datetime_to_simstrat_time(dt, ref_date):
    delta = dt - ref_date
    return delta.days + delta.seconds / 86400


def read_ref_date(ensemble_base):
    with open(os.path.join(ensemble_base, "ensemble1", "Settings.par")) as f:
        year = json.load(f)["Simulation"]["Reference year"]
    return datetime(year, 1, 1, tzinfo=timezone.utc)


def init_par(ensemble_dir, args):
    """The engine's own par file (par_file), writing to its own results_dir."""
    dst = os.path.join(ensemble_dir, args["par_file"])
    if os.path.exists(dst):
        return
    with open(os.path.join(ensemble_dir, "Settings.par")) as f:
        par = json.load(f)
    par["Output"]["Path"] = args["results_dir"]
    with open(dst, "w") as f:
        json.dump(par, f, indent=4)


def overwrite_par_dates(par_path, window_start, window_end, ref_date):
    """Run exactly [window_start, window_end]. The next window starts at this one's end instant
    from its snapshot; Simstrat writes its first output one interval after the start, so the
    joined trajectory has neither a gap nor a duplicate row."""
    with open(par_path) as f:
        par = json.load(f)
    par["Simulation"]["Start d"] = datetime_to_simstrat_time(window_start, ref_date)
    par["Simulation"]["End d"]   = datetime_to_simstrat_time(window_end,   ref_date)
    with open(par_path, "w") as f:
        json.dump(par, f, indent=4)


def _member_snapshot_paths(member_id, args):
    base = os.path.join(args["ensemble_base"], f"ensemble{member_id}")
    return os.path.join(base, args["results_dir"], SNAPSHOT), os.path.join(base, args["par_file"])


def read_snapshot_T(member_id, args):
    return read_snapshot_T_at(*_member_snapshot_paths(member_id, args))


def write_snapshot_T(member_id, T_new, args):
    snap_path, par_path = _member_snapshot_paths(member_id, args)
    write_snapshot_T_at(snap_path, par_path, T_new)


def clear_member_state(ensemble_base, member_ids, results_dir):
    for i in member_ids:
        live = os.path.join(ensemble_base, f"ensemble{i}", results_dir, SNAPSHOT)
        if os.path.exists(live):
            os.remove(live)


def copy_member_state(src_id, member_ids, args):
    def snapshot(i):
        return os.path.join(args["ensemble_base"], f"ensemble{i}", args["results_dir"], SNAPSHOT)
    with concurrent.futures.ThreadPoolExecutor() as pool:
        pool.map(lambda dst: shutil.copy2(snapshot(src_id), dst),
                 [snapshot(i) for i in member_ids if i != src_id])


def _analysis_member_paths(member_id, cfg):
    """A member's final snapshot and par file: in its member dir for the native engines, in its
    work dir for OpenDA."""
    if cfg.get("engine") == "openda":
        work = os.path.join(cfg["openda_dir"], "Results", f"work{member_id}")
        return os.path.join(work, "Results", SNAPSHOT), os.path.join(work, "Settings.par")
    return _member_snapshot_paths(member_id, cfg)


def write_mean_state(cfg, out_path):
    """Member 1's snapshot with the mean temperature column of members 1..N."""
    mean_T = np.mean(np.vstack([read_snapshot_T_at(*_analysis_member_paths(i, cfg))[0]
                                for i in range(1, cfg["n_members"] + 1)]), axis=0)
    member_snap, member_par = _analysis_member_paths(1, cfg)
    shutil.copy(member_snap, out_path)
    write_snapshot_T_at(out_path, member_par, mean_T)


class Simstrat(ForwardModel):
    """The Simstrat 1D lake model."""

    name    = "simstrat"
    version = "3.0.4"
    binary  = "/simstrat/build/simstrat"
    workdir = "/simstrat/run"               # where ensemble_base is mounted in the container

    def run_config(self):
        return {"simstrat_version": self.version, "simstrat_binary": self.binary,
                "simstrat_workdir": self.workdir}

    model_inputs_ready    = staticmethod(model_inputs_ready)
    warmup_snapshot       = staticmethod(warmup_snapshot)
    instances_ready       = staticmethod(instances_ready)
    copy_model_inputs     = staticmethod(copy_model_inputs)
    read_forcing          = staticmethod(read_forcing)
    write_forcing         = staticmethod(write_forcing)
    start_containers      = staticmethod(start_containers)
    stop_containers       = staticmethod(stop_containers)
    run_window            = staticmethod(run_window_parallel)
    read_ref_date         = staticmethod(read_ref_date)
    model_output_depths   = staticmethod(model_output_depths)
    set_output_depths     = staticmethod(set_output_depths)
    obs_to_sim_col        = staticmethod(obs_to_sim_col)
    sim_col_to_depth      = staticmethod(sim_col_to_depth)
    load_T_window         = staticmethod(load_T_window)
    clear_member_outputs  = staticmethod(clear_member_outputs)
    accumulate_mean       = staticmethod(accumulate_mean)
    mean_traj_path        = staticmethod(mean_traj_path)
    read_snapshot_T       = staticmethod(read_snapshot_T)
    write_snapshot_T      = staticmethod(write_snapshot_T)
    clear_member_state    = staticmethod(clear_member_state)
    copy_member_state     = staticmethod(copy_member_state)
    write_mean_state      = staticmethod(write_mean_state)
    output_epoch          = staticmethod(output_epoch)
    member_output_file    = staticmethod(member_output_file)
    read_output           = staticmethod(read_output)

    def openda_binding(self):
        from .openda import SimstratOpenDA      # not imported by the OpenDA wrapper's processes
        return SimstratOpenDA()

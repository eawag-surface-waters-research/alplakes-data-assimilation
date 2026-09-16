"""OpenDA restart chain: each run (cycle) continues from the ensemble the previous cycle saved.

Opt-in through the run config:

    "openda_restart": {"dir": "<restart dir>"}      # dir optional, default <openda_dir>/restart

Without it nothing here runs. With it, src/assimilate.py plans the cycles itself (plan_cycles): from
the chain's last end (or start_date for an empty chain) to each next observation time, up to
end_date if set and the forcing's end. So the same command resumes the chain every time.

After a successful cycle, the one rst_*.zip OpenDA wrote is moved into the chain as
restart_<end_day>.zip (we name it, OpenDA's own time tag is not trusted). A sidecar .json records
the set-up (lake, filter, ensemble size) and the cycle's exact start/end. The next cycle picks the
file named by its start day. An empty chain cold-starts from the warmup; a missing file stops.
"""
import os
import re
import glob
import json
import math
import fnmatch
import shutil
import logging
from datetime import datetime, timezone, timedelta

import numpy as np

from assimilator.functions import RNG_STREAMS, AdaptiveSigma, resolve_scalar_sigma_obs
from assimilator.models.simstrat import SIMSTRAT_REF_YEAR
from .config import simstrat_time

logger = logging.getLogger(__name__)

PREFIX = "restart_"
OPENDA_GLOB = "rst_*.zip"        # what config._ODA_RESTART asks OpenDA to write

# Row-time tolerance when merging trajectories, in days (~86 s): above T_out.dat's print step,
# below any output interval.
TIME_TOL = 1e-3


def cycle_seed(rng_seed, end_iso):
    """OpenDA random seed for one cycle, from (rng_seed, stream "openda_obs", unix s of the cycle end).
    Without it every cycle perturbs its observations with the same draw. 31 bits: Java int."""
    dt = datetime.fromisoformat(re.sub(r"\.\d+", "", str(end_iso)))
    dt = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
    ss = np.random.SeedSequence([int(rng_seed), RNG_STREAMS["openda_obs"], int(dt.timestamp())])
    return int(ss.generate_state(1, dtype=np.uint32)[0] & 0x7FFFFFFF)


def _key(day):
    """Filename key of a Simstrat day, fixed precision."""
    return f"{float(day):.10f}"


def restart_path(restart_dir, day):
    return os.path.join(restart_dir, f"{PREFIX}{_key(day)}.zip")


def _sidecar(zip_path):
    return zip_path[:-len(".zip")] + ".json"


def in_chain(restart_dir, day):
    """True when a restart for `day` is committed (checked by name, not by float)."""
    return os.path.isfile(restart_path(restart_dir, day))


def chain_days(restart_dir):
    """Committed restart days, ascending (rounded to the filename key)."""
    days = []
    for p in glob.glob(os.path.join(restart_dir, f"{PREFIX}*.zip")):
        try:
            days.append(float(os.path.basename(p)[len(PREFIX):-len(".zip")]))
        except ValueError:
            continue
    return sorted(days)


def select_restart(restart_dir, start_day, identity):
    """Restart file this cycle starts from: a path, or "" to cold-start an empty chain.
    `identity` = {"lake", "filter", "n_members"}; a restart from another set-up is refused."""
    days = chain_days(restart_dir)
    if not days:
        logger.warning(f"[restart] chain {restart_dir} is empty — COLD START from the warmup at "
                       f"Simstrat day {start_day:g}")
        return ""
    path = restart_path(restart_dir, start_day)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"[restart] no restart for Simstrat day {_key(start_day)} in {restart_dir}; the chain holds "
            f"{_key(days[0])} .. {_key(days[-1])} ({len(days)} file(s)). Start the cycle at a committed time, "
            f"or empty the directory to begin a new chain from the warmup.")
    side = _sidecar(path)
    meta = json.load(open(side)) if os.path.isfile(side) else {}
    wrong = {k: (meta.get(k), v) for k, v in identity.items() if k in meta and meta[k] != v}
    if wrong:
        raise ValueError(f"[restart] {os.path.basename(path)} was written by a different set-up "
                         f"(restart vs this run): {wrong}")
    later = [d for d in days if d > float(_key(start_day))]
    if later:
        logger.warning(f"[restart] re-running from day {_key(start_day)}: {len(later)} later restart(s) "
                       f"(up to {_key(later[-1])}) exist and the next commit will overwrite {_key(later[0])}")
    logger.info(f"[restart] continuing from {path}")
    return path


def _openda_restarts(openda_dir, restart_dir):
    """rst_*.zip files under openda_dir, outside the chain dir."""
    restart_dir = os.path.abspath(restart_dir)
    found = []
    for d, subdirs, files in os.walk(openda_dir):
        if os.path.abspath(d).startswith(restart_dir):
            subdirs[:] = []
            continue
        found += [os.path.join(d, f) for f in files if fnmatch.fnmatch(f, OPENDA_GLOB)]
    return found


def clear_openda_output(openda_dir, restart_dir):
    """Remove rst_*.zip left by an earlier run, so the commit finds only this run's file."""
    for p in _openda_restarts(openda_dir, restart_dir):
        os.remove(p)
        logger.info(f"[restart] removed stale {os.path.relpath(p, openda_dir)}")


def commit_restart(openda_dir, restart_dir, end_day, identity, start_from, extra=None):
    """Move the one rst_*.zip this run wrote into the chain as restart_<end_day>.zip, plus its sidecar.
    Moved via a temporary name, so a chain file is complete or absent. `extra` goes into the sidecar."""
    found = _openda_restarts(openda_dir, restart_dir)
    if len(found) != 1:
        raise RuntimeError(f"[restart] expected exactly one {OPENDA_GLOB} under {openda_dir} after the "
                           f"run, found {len(found)}: {[os.path.relpath(p, openda_dir) for p in found]}")
    os.makedirs(restart_dir, exist_ok=True)
    target = restart_path(restart_dir, end_day)
    shutil.move(found[0], target + ".tmp")
    os.replace(target + ".tmp", target)
    meta = {**identity, **(extra or {}), "end_day": float(end_day), "started_from": start_from or None,
            "openda_file": os.path.basename(found[0]),
            "committed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with open(_sidecar(target) + ".tmp", "w") as f:
        json.dump(meta, f, indent=1)
    os.replace(_sidecar(target) + ".tmp", _sidecar(target))
    logger.info(f"[restart] committed {os.path.basename(found[0])} -> {target}")
    return target


def _row_day(line):
    return float(line.split(",", 1)[0])


def append_trajectory(accum_path, cycle_path, start_day):
    """Merge one cycle's T_out.dat into the chain's file; returns (rows kept, rows added).
    Keeps chain rows up to the cycle start, then adds the cycle's rows after it, so re-running a
    cycle replaces its rows instead of duplicating them. A missing cycle file changes nothing."""
    if not os.path.isfile(cycle_path):
        logger.warning(f"[restart] {cycle_path} missing — {accum_path} left unchanged")
        return 0, 0
    with open(cycle_path) as f:
        header, *rows = [ln for ln in f.read().splitlines() if ln.strip()]
    new = [r for r in rows if _row_day(r) > start_day + TIME_TOL]
    kept = []
    if os.path.isfile(accum_path):
        with open(accum_path) as f:
            old_header, *old = [ln for ln in f.read().splitlines() if ln.strip()]
        if old_header != header:
            raise ValueError(f"[restart] {cycle_path} has different output depths than {accum_path}")
        kept = [r for r in old if _row_day(r) <= start_day + TIME_TOL]
    os.makedirs(os.path.dirname(accum_path), exist_ok=True)
    with open(accum_path + ".tmp", "w") as f:
        f.write("\n".join([header] + kept + new) + "\n")
    os.replace(accum_path + ".tmp", accum_path)
    return len(kept), len(new)


def archive_cycle(openda_dir, end_day, paths):
    """Copy this cycle's outputs (overwritten by the next run) to cycles/<end_day>/.
    Each entry is a path or (path, archive name). Missing files are skipped. Returns the dir."""
    dst = os.path.join(openda_dir, "cycles", _key(end_day))
    os.makedirs(dst, exist_ok=True)
    for p in paths:
        src, name = p if isinstance(p, tuple) else (p, os.path.basename(p))
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(dst, name))
    return dst


# --- cycle planning ------------------------------------------------------------------------

def iso_utc(stamp):
    """ISO UTC instant of a timestamp (naive = UTC, sub-seconds dropped), e.g. 2025-01-01T06:00:00+00:00."""
    dt = datetime.fromisoformat(re.sub(r"\.\d+", "", str(stamp).strip()))
    dt = dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)
    return dt.isoformat()


def chain_start(restart_dir, cold_start):
    """Start instant of the next cycle: the end recorded by the last committed restart,
    or `cold_start` when the chain is empty."""
    days = chain_days(restart_dir)
    if not days:
        return iso_utc(cold_start)
    side = _sidecar(restart_path(restart_dir, days[-1]))
    end = (json.load(open(side)) if os.path.isfile(side) else {}).get("end_date")
    if not end or _key(simstrat_time(end)) != _key(days[-1]):
        raise ValueError(f"[restart] {side} has no end_date matching its restart day {_key(days[-1])}")
    return iso_utc(end)


def plan_cycles(start_iso, obs_times, last_day, min_minutes, max_cycles=None):
    """Cycles [(start, end)] from `start_iso`: one per observation time after it, up to `last_day`
    (Simstrat day). An end less than `min_minutes` after the previous one is skipped (the adapter
    would drop its observation)."""
    cycles, prev = [], start_iso
    for end in sorted({iso_utc(t) for t in obs_times}, key=simstrat_time):
        t = simstrat_time(end)
        if t <= simstrat_time(start_iso) + TIME_TOL:
            continue
        if t > last_day + TIME_TOL or (max_cycles and len(cycles) >= max_cycles):
            break
        if (t - simstrat_time(prev)) * 1440.0 < min_minutes:
            logger.info(f"[restart] skipped cycle end {end}: less than {min_minutes} min after {prev}")
            continue
        cycles.append((prev, end))
        prev = end
    return cycles


# --- adaptive observation error ---------------------------------------------------------------
# A cycle's result file holds, per analysis, the observations and each member's forecast at the
# observation points, in the order of THAT cycle's series; the archived formatter is the only
# record of that order. So the innovation and the prior spread the estimator needs are recoverable
# from the archives, and a resumed chain gets the sigma an uninterrupted one would.

FORMATTER = "timeSeriesFormatter.gen.xml"
_EPOCH = datetime(SIMSTRAT_REF_YEAR, 1, 1, tzinfo=timezone.utc)


def _day_to_dt(day):
    return _EPOCH + timedelta(days=float(day))


def parse_results(path):
    """{name: [vector per record]} from OpenDA's PythonResultWriter file."""
    out = {}
    for raw in open(path, encoding="latin1"):
        m = re.match(r"([A-Za-z_]\w*)\.append\(\[?(.*?)\]?\)\s*$", raw)
        if m:
            out.setdefault(m.group(1), []).append([float(v) for v in m.group(2).split(",") if v.strip()])
    return out


def series_depths(formatter_path):
    """Depth of each observation series, in the formatter's order (ids T_<d>m or T_<d>m_<season>)."""
    ids = re.findall(r'<timeSeries id="T_([0-9.]+)m(?:_[a-z]+)?"', open(formatter_path).read())
    return [float(d) for d in ids]


def adaptive_history(openda_dir, results_file, t_end, window_days):
    """[(when, depth, innovation, prior spread)] from the cycles archived in the window before
    `t_end`, oldest first, plus how many cycles were skipped for lack of an archived formatter."""
    rows, skipped, lo = [], 0, t_end - window_days
    for cdir in sorted(glob.glob(os.path.join(openda_dir, "cycles", "*"))):
        try:
            end_key = float(os.path.basename(cdir))
        except ValueError:
            continue
        if not (lo - 1.0 < end_key < t_end + TIME_TOL):          # coarse filter on the folder name
            continue
        res_path, fmt_path = os.path.join(cdir, results_file), os.path.join(cdir, FORMATTER)
        if not os.path.isfile(res_path):
            continue
        if not os.path.isfile(fmt_path):
            skipped += 1
            continue
        res, depths = parse_results(res_path), series_depths(fmt_path)
        n_members = len([k for k in res if re.fullmatch(r"pred_f_\d+", k)])
        for i, t in enumerate(v[0] for v in res.get("analysis_time", [])):
            if not (lo < t < t_end - TIME_TOL):
                continue
            y = [v for v in res["obs"][i]]
            P = [res[f"pred_f_{j}"][i] for j in range(n_members)]
            if len(y) != len(depths) or any(len(p) != len(depths) for p in P):
                logger.warning(f"[sigma] {cdir}: {len(y)} observations vs {len(depths)} series — skipped")
                continue
            for k, z in enumerate(depths):
                col = [p[k] for p in P]
                mean = sum(col) / len(col)
                var = sum((v - mean) ** 2 for v in col) / (len(col) - 1)
                rows.append((_day_to_dt(t), z, y[k] - mean, math.sqrt(var)))
    rows.sort(key=lambda r: r[0])
    return rows, skipped


def cycle_sigma(cfg, openda_dir, results_file, end_iso, depths):
    """{depth: sigma} for the cycle ending at `end_iso`, from the archived cycles in the window.
    Depths short of min_samples keep the run's configured sigma_obs. Stateless: the history is
    rebuilt every cycle, so a resumed chain resolves what an uninterrupted one would."""
    est = AdaptiveSigma(cfg)
    when = _day_to_dt(simstrat_time(end_iso))
    rows, skipped = adaptive_history(openda_dir, results_file, simstrat_time(end_iso),
                                     est.window.total_seconds() / 86400.0)
    for t, z, d, s in rows:
        est.record(t, [z], [d], [s])
    fallback = resolve_scalar_sigma_obs(cfg, when.month)
    sigma = est.resolve(depths, when, fallback)
    n_adapted = sum(1 for z in depths if len(est.hist[round(float(z), 6)]) >= est.min_samples)
    logger.info(f"[sigma] cycle {end_iso}: adaptive at {n_adapted}/{len(depths)} depths from "
                f"{len(rows)} archived observations; the rest use sigma_obs={fallback:g}"
                + (f" ({skipped} cycle(s) skipped: no archived {FORMATTER})" if skipped else ""))
    return {float(z): float(v) for z, v in zip(depths, sigma)}

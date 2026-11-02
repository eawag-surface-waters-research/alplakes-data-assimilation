"""OpenDA restart chain: each OpenDA run continues from the ensemble the previous run saved.

Within one OpenDA run the members keep their own states from one analysis to the next. Between
runs they do not: a new run builds every member from the one template state. An operational host
that runs every day would collapse the ensemble's spread every day. The chain closes that gap: at
the end of each run OpenDA saves all member states (a restart file), and the next run restores
them. Opt in through the run config:

    "openda_restart": {"dir": "<chain folder>",     # default <openda_dir>/restart
                       "keep": 7,                   # restart files kept (newest first)
                       "cycle_max_days": 31}        # longest single OpenDA run of a catch-up

A call runs from the chain's end (or start_date for an empty chain) to end_date as ONE OpenDA run,
every observation in between analysed at its own time inside it. A longer catch-up is split into
runs of at most cycle_max_days, so a failure loses at most one chunk. Restart files are named by
their UTC instant (restart_20261102T0000Z.zip), each with a .json alongside (set-up, window, seed).
A run is committed to the chain last, after its outputs are archived and appended: a run that fails
anywhere adds nothing and is simply run again.
"""
import os
import glob
import json
import fnmatch
import shutil
import logging
from datetime import datetime, timedelta, timezone

import numpy as np

from assimilator.functions import RNG_STREAMS

logger = logging.getLogger(__name__)

PREFIX      = "restart_"
OPENDA_GLOB = "rst_*.zip"          # what config._ODA_RESTART asks OpenDA to write
KEY_FORMAT  = "%Y%m%dT%H%MZ"

DEFAULTS = {"keep": 7, "cycle_max_days": 31}


def settings(cfg):
    """The run config's "openda_restart" block with its defaults."""
    spec = cfg.get("openda_restart") or {}
    unknown = set(spec) - set(DEFAULTS) - {"dir"}
    if unknown:
        raise ValueError(f"openda_restart: unknown key(s) {sorted(unknown)}; "
                         f"expected {sorted(set(DEFAULTS) | {'dir'})}")
    return {**DEFAULTS, **spec}


def utc(t):
    """A tz-aware UTC datetime from an ISO string or a datetime (naive = UTC); seconds dropped
    below the minute, the resolution restart files are named at."""
    if isinstance(t, str):
        t = datetime.fromisoformat(t)
    t = t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t.astimezone(timezone.utc)
    return t.replace(second=0, microsecond=0)


def chain_dir(cfg, openda_dir):
    return os.path.abspath(os.path.expanduser(settings(cfg).get("dir") or os.path.join(openda_dir, "restart")))


def restart_path(restart_dir, t):
    return os.path.join(restart_dir, f"{PREFIX}{utc(t).strftime(KEY_FORMAT)}.zip")


def _sidecar(zip_path):
    return zip_path[:-len(".zip")] + ".json"


def chain_instants(restart_dir):
    """The committed restart instants, ascending."""
    out = []
    for p in glob.glob(os.path.join(restart_dir, f"{PREFIX}*.zip")):
        try:
            out.append(datetime.strptime(os.path.basename(p)[len(PREFIX):-len(".zip")], KEY_FORMAT)
                       .replace(tzinfo=timezone.utc))
        except ValueError:
            continue
    return sorted(out)


def chain_end(restart_dir):
    """Where the chain stops (its newest restart), or None for an empty chain."""
    instants = chain_instants(restart_dir)
    return instants[-1] if instants else None


def plan_cycles(start, end, max_days):
    """[(start, end)] of the OpenDA runs that take the chain from `start` to `end`: one run, or
    runs of at most `max_days` for a longer catch-up. Every observation inside a run is analysed
    at its own time, so the split only bounds how much a failure can lose."""
    start, end = utc(start), utc(end)
    cycles, step = [], timedelta(days=float(max_days))
    while start < end:
        stop = min(start + step, end)
        cycles.append((start, stop))
        start = stop
    return cycles


def merge_empty_cycles(cycles, obs_times):
    """Merge each run whose window (start, end] holds no observation into the next, and drop
    trailing runs without one: OpenDA then always has something to analyse, and a gap at the end
    of the record leaves the chain where it is until the next observation."""
    times = sorted(utc(t) for t in obs_times)

    def has_obs(a, b):
        return any(a < t <= b for t in times)
    merged, pending = [], None
    for a, b in cycles:
        a = pending if pending is not None else a
        if has_obs(a, b):
            merged.append((a, b))
            pending = None
        else:
            pending = a
    return merged


def cycle_seed(rng_seed, end):
    """The OpenDA random seed of the run ending at `end`, from (rng_seed, "openda_obs", unix s of
    the end). Without it every run perturbs its observations with the same draws. 31 bits: Java int."""
    ss = np.random.SeedSequence([int(rng_seed), RNG_STREAMS["openda_obs"], int(utc(end).timestamp())])
    return int(ss.generate_state(1, dtype=np.uint32)[0] & 0x7FFFFFFF)


def select_restart(restart_dir, start, identity):
    """The restart file a run starting at `start` continues from: a path, or "" to cold-start an
    empty chain from the warm start. `identity` ({"lake", "filter", "n_members"}) must match the
    set-up that wrote it."""
    if chain_end(restart_dir) is None:
        logger.warning(f"[restart] chain {restart_dir} is empty: cold start from the warm start at {utc(start)}")
        return ""
    path = restart_path(restart_dir, start)
    if not os.path.isfile(path):
        instants = chain_instants(restart_dir)
        raise FileNotFoundError(
            f"[restart] no restart for {utc(start)} in {restart_dir}; the chain holds "
            f"{instants[0]} .. {instants[-1]} ({len(instants)} file(s)). Start at a committed time, "
            f"or empty the folder to begin a new chain from the warm start.")
    side = _sidecar(path)
    meta = json.load(open(side)) if os.path.isfile(side) else {}
    wrong = {k: (meta.get(k), v) for k, v in identity.items() if k in meta and meta[k] != v}
    if wrong:
        raise ValueError(f"[restart] {os.path.basename(path)} was written by a different set-up "
                         f"(restart vs this run): {wrong}")
    logger.info(f"[restart] continuing from {path}")
    return path


def _openda_restarts(openda_dir, restart_dir):
    """rst_*.zip files under openda_dir, outside the chain folder."""
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


def commit_restart(openda_dir, restart_dir, end, identity, extra=None):
    """Move the one rst_*.zip this run wrote into the chain as restart_<end>.zip, with its .json.
    Called LAST in a run: a chain file is there only for a run whose outputs are all saved. Moved
    via a temporary name, so a chain file is complete or absent."""
    found = _openda_restarts(openda_dir, restart_dir)
    if len(found) != 1:
        raise RuntimeError(f"[restart] expected exactly one {OPENDA_GLOB} under {openda_dir} after the "
                           f"run, found {len(found)}: {[os.path.relpath(p, openda_dir) for p in found]}")
    os.makedirs(restart_dir, exist_ok=True)
    target = restart_path(restart_dir, end)
    meta = {**identity, **(extra or {}), "end": utc(end).isoformat(),
            "openda_file": os.path.basename(found[0]),
            "committed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    with open(_sidecar(target) + ".tmp", "w") as f:
        json.dump(meta, f, indent=1)
    os.replace(_sidecar(target) + ".tmp", _sidecar(target))
    shutil.move(found[0], target + ".tmp")
    os.replace(target + ".tmp", target)
    logger.info(f"[restart] committed {os.path.basename(found[0])} -> {target}")
    return target


def prune(restart_dir, keep):
    """Keep the newest `keep` restart files (and their .json); a run only ever needs the newest."""
    instants = chain_instants(restart_dir)
    for t in instants[:-int(keep)] if keep and len(instants) > keep else []:
        path = restart_path(restart_dir, t)
        for p in (path, _sidecar(path)):
            if os.path.exists(p):
                os.remove(p)
        logger.info(f"[restart] removed old {os.path.basename(path)} (keeping the newest {keep})")


def _row_day(line):
    return float(line.split(",", 1)[0])


def append_trajectory(accum_path, run_path, start_day, tol=1e-3):
    """Merge one run's trajectory (a CSV with the time in model days in its first column) into the
    chain's: keep the chain's rows up to the run start, then add the run's rows after it. A re-run
    replaces its rows instead of duplicating them, and only the new rows are written, so the cost
    does not grow with the chain. Returns (rows kept, rows added); a missing run file changes nothing."""
    if not os.path.isfile(run_path):
        logger.warning(f"[restart] {run_path} missing: {accum_path} left unchanged")
        return 0, 0
    with open(run_path) as f:
        header, *rows = [ln for ln in f.read().splitlines() if ln.strip()]
    new = [r for r in rows if _row_day(r) > start_day + tol]
    kept = []
    if os.path.isfile(accum_path):
        with open(accum_path) as f:
            old_header, *old = [ln for ln in f.read().splitlines() if ln.strip()]
        if old_header != header:
            raise ValueError(f"[restart] {run_path} has different columns than {accum_path}")
        kept = [r for r in old if _row_day(r) <= start_day + tol]
    os.makedirs(os.path.dirname(accum_path), exist_ok=True)
    with open(accum_path + ".tmp", "w") as f:
        f.write("\n".join([header] + kept + new) + "\n")
    os.replace(accum_path + ".tmp", accum_path)
    return len(kept), len(new)


def archive_cycle(openda_dir, end, paths):
    """Copy this run's outputs, which the next run overwrites, to cycles/<end>/. Each entry is a
    path or (path, archive name); missing files are skipped. Returns the folder."""
    dst = os.path.join(openda_dir, "cycles", utc(end).strftime(KEY_FORMAT))
    os.makedirs(dst, exist_ok=True)
    for p in paths:
        src, name = p if isinstance(p, tuple) else (p, os.path.basename(p))
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(dst, name))
    return dst

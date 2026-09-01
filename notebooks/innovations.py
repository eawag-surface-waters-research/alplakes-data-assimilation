"""Per-observation innovations, prior spread and sigma for a finished EnKF run.

functions library

`innov_table(raw, lake)` is the interface: one row per assimilated observation with the innovation
`d`, the prior ensemble `spread`, the `sigma` the config's error model gives that (depth, instant),
and their ratio `nis`. sigma is rebuilt from the CONFIG.

    import innovations as mr
    sub, depths = mr.innov_table(raw, "aegeri")
"""
import os
import sys
import json
import glob
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from assimilator.functions import (load_obs, merge_lake_args,                # noqa: E402
                                   resolve_obs_path, resolve_src)
from assimilator.sigma_rep import load_sigma_rep, resolve_sigma_obs         # noqa: E402

_CACHE = {}


def _cached(key, build):
    if key not in _CACHE:
        _CACHE[key] = build()
    return _CACHE[key]


def rel(path):
    """Repo-relative path for logging — os.path.relpath raises across Windows mounts."""
    try:
        return os.path.relpath(path, ROOT)
    except ValueError:
        return str(path)


def ref_date_of(lake):
    with open(os.path.join(ROOT, "inputs", lake, "Settings.par")) as f:
        return datetime(json.load(f)["Simulation"]["Reference year"], 1, 1, tzinfo=timezone.utc)


def read_innov(base, cfg):
    """Innovations as (time, depth, d), taking the header only when it describes the rows.

    enkf.py now writes every obs depth on every row, blank where a sensor was absent, so the header
    IS the labels. It used to emit only the depths that reported, appended positionally under a
    header frozen from the first window — which puts every value after a dropped sensor on the
    wrong depth (on the 2025 seven-lake run: geneva 343 of 359 rows, hallwil 313 of 364). For those
    older files the labels are rebuilt from the observations the run would have selected.

    The header is trusted only when it carries the whole depth grid AND every row is that wide. An
    older file passing both never dropped a sensor, so its header is right anyway.
    """
    obs = load_obs(resolve_obs_path(cfg))
    path = os.path.join(base, "enkf_innov_by_depth.csv")
    with open(path, encoding="utf-8") as f:
        lines = f.read().splitlines()
    hdr = np.array([float(c[2:]) for c in lines[0].split(",")[1:]])
    all_z = np.sort(obs["depth"].unique())

    if (len(hdr) == len(all_z) and np.allclose(hdr, all_z)
            and all(len(ln.split(",")) - 1 == len(hdr) for ln in lines[1:])):
        long = pd.read_csv(path).melt(id_vars="date", var_name="depth", value_name="d")
        long["time"] = pd.to_datetime(long["date"], utc=True)
        long["depth"] = long["depth"].str[2:].astype(float)
        return long[["time", "depth", "d"]].dropna(subset=["d"])

    # Rebuild: the writer took every depth with an obs in (previous analysis, this one], so use
    # that window rather than the end instant alone — a lake whose sensors report on staggered
    # timestamps would otherwise resolve to the wrong set at the same count, and the width check
    # below cannot see that. NaN-valued obs are kept, because the writer kept them too.
    ends = pd.DatetimeIndex([ln.split(",")[0] for ln in lines[1:]])
    lo = (pd.to_datetime(cfg["start_date"], utc=True) if cfg.get("start_date")
          else ends[0] - pd.Timedelta(days=1))
    times, depths, vals, dropped = [], [], [], 0
    for k, line in enumerate(lines[1:]):
        parts = line.split(",")
        win = obs[(obs["time"] > lo) & (obs["time"] <= ends[k])]
        lo = ends[k]
        dd = np.sort(win["depth"].unique())
        if len(dd) != len(parts) - 1:
            dropped += 1
            continue
        times.append(np.repeat(ends[k], len(dd)))
        depths.append(dd)
        vals.append(np.array(parts[1:], dtype=float))
    if dropped:
        print(f"{'':<14} {dropped} innovation row(s) with an unresolvable depth set — skipped")
    return pd.DataFrame({"time": np.concatenate(times), "depth": np.concatenate(depths),
                         "d": np.concatenate(vals)}).dropna(subset=["d"])


def prior_spread(base, results_dir, depths, times, ref_date):
    """Across-member std at each (analysis time, obs depth) — the prior spread the filter used.

    enkf.py writes no per-depth spread, so it is rebuilt from the members: at the analysis instant
    the T_out row is still the PRIOR (the update first shows up an hour later), so the across-member
    std there is exactly it. Only the columns nearest the obs depths are parsed, which is what keeps
    20 members to about a second each.

    THIS FUNCTION IS AVOIDABLE. enkf.py already computes the vector, at algorithms/enkf.py:208 —
    `np.std(H_v @ X_f, axis=1, ddof=1)` — and then discards the depth axis with `.mean()` to store
    the scalar `spread_pre`. Passing it out through `diags` (beside `_innov_vec`, which the by-depth
    writer already uses) and appending it to an enkf_prior_by_depth.csv would replace everything
    below with a read_csv. Worth doing before the pass-2 refit; it cannot retire this path for runs
    that already finished, which is why the fallback stays.
    """
    paths = [p for p in sorted(glob.glob(os.path.join(base, "ensemble*", results_dir, "T_out.dat")))
             if os.path.basename(os.path.dirname(os.path.dirname(p))) != "ensemble0"]
    if not paths:
        raise FileNotFoundError(f"no member T_out.dat under {rel(base)}")
    hdr = pd.read_csv(paths[0], nrows=0).columns
    mz = np.array([float(c) for c in hdr[1:]])
    cols = [hdr[1 + int(np.argmin(np.abs(mz + z)))] for z in depths]   # obs +z -> column -z
    stack = []
    for p in paths:
        df = pd.read_csv(p, usecols=[hdr[0]] + list(dict.fromkeys(cols)))
        idx = (pd.Timestamp(ref_date)
               + pd.to_timedelta(df[hdr[0]].to_numpy(float), unit="D")).round("1h")
        df = df.set_index(idx)
        stack.append(df[~df.index.duplicated(keep="first")].reindex(times)[cols].to_numpy(float))
    return np.nanstd(np.stack(stack), axis=0, ddof=1)


def sigma_row(depths, when, cfg, table):
    """sigma at every obs depth for one analysis instant, always as a (len(depths),) array —
    resolve_sigma_obs returns a plain float on the scalar path."""
    sig = np.asarray(resolve_sigma_obs(list(depths), np.ones(len(depths)), when, cfg, table),
                     dtype=float)
    return np.broadcast_to(sig, (len(depths),))


def innov_table(raw, lake):
    """(per-observation frame with d / spread / sigma / nis, obs depths) for one lake.

    sigma is rebuilt from the CONFIG, not from the run — a run does not save its config, so this
    shows what it would score under today's R. Cached: the member read behind `spread` is the whole
    cost, and every caller wants the same block.
    """
    def build():
        cfg = merge_lake_args(raw, lake=lake)
        base = resolve_src(cfg["ensemble_base"])
        long = read_innov(base, cfg)
        times = pd.DatetimeIndex(sorted(long["time"].unique()))
        depths = np.sort(long["depth"].unique())

        def wide(arr, name):
            df = pd.DataFrame(arr, index=times, columns=depths)
            df.index.name, df.columns.name = "time", "depth"
            return df.stack().rename(name)

        table = load_sigma_rep(cfg)
        sp = wide(prior_spread(base, cfg.get("results_dir", "Results"), depths, times,
                               ref_date_of(lake)), "spread")
        sg = wide(np.stack([sigma_row(depths, t, cfg, table) for t in times]), "sigma")
        sub = long.set_index(["time", "depth"]).join(sp).join(sg).reset_index().dropna()
        sub["nis"] = sub["d"] ** 2 / (sub["spread"] ** 2 + sub["sigma"] ** 2)
        return sub, depths
    return _cached(("innov", lake), build)


def pooled_nis(sub):
    """<d²> / <spread² + σ²> — the variance identity that inverts to a sigma scale. NOT the mean of
    the per-observation ratios, which is a different number."""
    return float((sub["d"] ** 2).mean() / (sub["spread"] ** 2 + sub["sigma"] ** 2).mean())

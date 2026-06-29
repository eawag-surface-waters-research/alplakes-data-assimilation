"""Unified post-run summary of the assimilated temperature field + skill report.

Both engines — the native EnKF/PF (`assimilate.py`) and the OpenDA black-box
(`openda_assimilation.py`) — write per-member Simstrat output in the same
`T_out.dat` format ("Datetime" = Simstrat day, then one column per depth).

For each run this reads ensemble members 1..N (the control, member 0, is
excluded) and writes two files into the run's own folder (`out_dir`, under run/ — the
native engines use run/<lake>/, OpenDA its run/openda_<model>_<lake>_<filter>/ dir):

  <lake>_<engine>_<label>.csv   posterior ensemble mean + std per (time, depth):
                                time,depth,T_mean,T_std  (hourly, full column)

  <lake>_<engine>_<label>.json  skill/bias report, scoring the posterior mean
                                against the observations in observations/<lake>/temperature.csv
                                at the model-output depths (obs below the grid bed are
                                dropped — the same set the engines assimilate; model
                                interpolated in depth to each obs depth, matched to the
                                nearest model output time).  The
                                same reference is used for both engines so the
                                numbers are directly comparable.  bias = model - obs
                                (+ = model too warm).  Not a withheld set — both
                                engines assimilate these obs in some reduced form —
                                so it is an "analysis fit", labelled as such.
"""

import os
import sys
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # put src/ on the path
from assimilator.functions import (
    ROOT, load_obs, filter_obs_to_model_depths,
    load_json, merge_lake_args, resolve_src, resolve_root, resolve_obs_path,
)
from assimilator.models.simstrat import SIMSTRAT_REF_YEAR

logger = logging.getLogger(__name__)


def _read_t_out(path):
    df = pd.read_csv(path)
    times  = df.iloc[:, 0].to_numpy(dtype=float)
    depths = np.array([float(c) for c in df.columns[1:]])
    temps  = df.iloc[:, 1:].to_numpy(dtype=float)   # [time, depth]
    return times, depths, temps


def _load_members(member_files):
    """Stack members 1..N -> posterior mean/std per (time, depth)."""
    if not member_files:
        raise ValueError("no member T_out.dat files given")
    missing = [f for f in member_files if not os.path.isfile(f)]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} member output(s), e.g. {missing[0]}")

    times, depths, first = _read_t_out(member_files[0])
    members = [first] + [_read_t_out(f)[2] for f in member_files[1:]]
    n = min(m.shape[0] for m in members)            # share cadence; truncate defensively
    arr = np.stack([m[:n] for m in members])        # [member, time, depth]
    return times[:n], depths, arr.mean(axis=0), arr.std(axis=0, ddof=1)


def _write_csv(times, depths, mean, std, out_csv):
    T, D = mean.shape
    pd.DataFrame({
        "time":   np.repeat(times, D),
        "depth":  np.tile(depths, T),
        "T_mean": mean.ravel(),
        "T_std":  std.ravel(),
    }).to_csv(out_csv, index=False, float_format="%.4f")


def _r(x):
    return round(float(x), 4)


def _depth_interp(depths, x):
    """Linear-interp indices/weight for target model depth x on ascending `depths`."""
    j1 = int(np.clip(np.searchsorted(depths, x), 1, len(depths) - 1))
    j0 = j1 - 1
    w  = float(np.clip((x - depths[j0]) / (depths[j1] - depths[j0]), 0.0, 1.0))
    return j0, j1, w


def _agg(err, spread):
    return {
        "n":               int(len(err)),
        "bias":            _r(np.mean(err)),
        "rmse":            _r(np.sqrt(np.mean(err ** 2))),
        "mae":             _r(np.mean(np.abs(err))),
        "mean_spread":     _r(np.mean(spread)),
        "coverage_1sigma": _r(np.mean(np.abs(err) <= spread)),
        "coverage_2sigma": _r(np.mean(np.abs(err) <= 2 * spread)),
    }


def _score(times, depths, mean, std, obs_csv):
    """Score posterior mean vs the obs at model-output depths; returns (overall, by_depth) or None."""
    # Score against the SAME obs the engines assimilate (and visualize plots): the centered
    # hourly mean (load_obs), not the raw samples — so the JSON skill numbers line up with the
    # plot's pooled RMSE.
    obs = load_obs(obs_csv).dropna(subset=["value"])
    if obs.empty:
        return None

    # Same depth set the engines assimilate: drop obs with no model-output depth (e.g. deeper than
    # the grid bed). z_out.dat carries the obs-depth superset (main.py step 2b), so every in-grid
    # obs depth lands on an exact model column.
    obs = filter_obs_to_model_depths(obs, [abs(float(d)) for d in depths])
    if obs.empty:
        return None

    ref   = datetime(SIMSTRAT_REF_YEAR, 1, 1, tzinfo=timezone.utc)
    o_day = (pd.to_datetime(obs["time"], utc=True) - ref).dt.total_seconds().to_numpy() / 86400.0
    o_d   = obs["depth"].to_numpy(dtype=float)
    o_v   = obs["value"].to_numpy(dtype=float)

    # match each obs to the nearest model output time, within one output step
    dt   = float(np.median(np.diff(times))) if len(times) > 1 else 1.0
    idx  = np.clip(np.searchsorted(times, o_day), 0, len(times) - 1)
    left = np.clip(idx - 1, 0, len(times) - 1)
    ti   = np.where(np.abs(times[left] - o_day) < np.abs(times[idx] - o_day), left, idx)
    keep = np.abs(times[ti] - o_day) <= dt
    ti, o_d, o_v = ti[keep], o_d[keep], o_v[keep]
    if len(o_v) == 0:
        return None

    # interpolate model mean/std in depth to each obs depth (few unique depths)
    m_mean = np.empty(len(o_d))
    m_std  = np.empty(len(o_d))
    for d in np.unique(o_d):
        sel = o_d == d
        j0, j1, w = _depth_interp(depths, -d)        # obs depth d -> model coord -d
        m_mean[sel] = mean[ti[sel], j0] * (1 - w) + mean[ti[sel], j1] * w
        m_std[sel]  = std[ti[sel], j0] * (1 - w) + std[ti[sel], j1] * w

    err = m_mean - o_v
    overall  = _agg(err, m_std)
    by_depth = {f"{d:g}": _agg(err[o_d == d], m_std[o_d == d]) for d in np.unique(o_d)}
    return overall, by_depth


def summarize_run(final_dir, lake, engine, label, member_files, obs_csv=None):
    """Write <final_dir>/<lake>_<engine>_<label>.{csv,json} from members 1..N.

    The CSV is always written; the JSON skill report is written when `obs_csv`
    exists.  Returns (csv_path, n_members, n_timesteps, n_depths, overall_skill)
    where overall_skill is the scored dict (rmse/bias/n/...) or None."""
    os.makedirs(final_dir, exist_ok=True)
    base = f"{lake}_{engine}_{label}"
    times, depths, mean, std = _load_members(member_files)

    out_csv = os.path.join(final_dir, base + ".csv")
    _write_csv(times, depths, mean, std, out_csv)

    overall = None
    if obs_csv and os.path.isfile(obs_csv):
        scored = _score(times, depths, mean, std, obs_csv)
        if scored is not None:
            overall, by_depth = scored
            report = {
                "lake": lake, "engine": engine, "filter": label,
                "n_members": len(member_files),
                "period": {"start": _r(times[0]), "end": _r(times[-1])},
                "scored_against": "obs at model-output depths (model interpolated to obs depth, "
                                  "nearest output time); analysis fit, not withheld",
                "bias_sign": "model - obs (+ = model too warm)",
                "n_obs": overall["n"],
                "overall": overall,
                "by_depth": by_depth,
            }
            with open(os.path.join(final_dir, base + ".json"), "w") as f:
                json.dump(report, f, indent=2)

    return out_csv, len(member_files), mean.shape[0], mean.shape[1], overall


def report_summary(engine, label, member_files, lake, obs_csv, out_dir):
    """Write the posterior summary + skill report into `out_dir` (the run's own folder under run/)
    and print a one-line recap. Shared tail for both native (run_enkf/run_pf) and OpenDA
    (run_openda) runs. Returns (csv_path, overall_skill) for the caller's run footer."""
    out_csv, n_mem, T, D, overall = summarize_run(out_dir, lake, engine, label, member_files, obs_csv=obs_csv)
    logger.info(f"[summary] {n_mem} members, {T} steps x {D} depths "
                f"-> {os.path.relpath(out_dir, ROOT)}/{lake}_{engine}_{label}.csv")
    return out_csv, overall


# ---------------------------------------------------------------------------
# CLI — regenerate the summary (.csv + .json) from existing member T_out.dat,
# without rerunning the assimilation. Same path resolution as the engines.
# ---------------------------------------------------------------------------

def summarize_from_config(cfg, model_name="simstrat"):
    """Rebuild the summary for the run described by `cfg` (a flattened run config) by reading the
    member T_out.dat already on disk. Mirrors the member_files / out_dir each engine uses."""
    engine    = cfg.get("engine", "python")
    lake      = cfg["lake"]
    n_members = cfg["n_members"]
    obs_csv   = resolve_obs_path(cfg)

    if engine == "python":
        ensemble_base = resolve_src(cfg["ensemble_base"])
        results_dir   = cfg["results_dir"]
        label         = cfg["algorithm"]                       # "EnKF" / "PF"
        member_files  = [os.path.join(ensemble_base, f"ensemble{i}", results_dir, "T_out.dat")
                         for i in range(1, n_members + 1)]
        report_summary("python", label, member_files, lake, obs_csv, ensemble_base)
    elif engine == "openda":
        filter_type = cfg.get("filter", "EnKF")
        default_dir = f"run/openda_{model_name}_{lake}_{filter_type.lower()}"
        openda_dir  = resolve_root(cfg.get("openda_dir") or default_dir)
        work_base   = os.path.join(openda_dir, "Results")
        member_files = [os.path.join(work_base, f"work{i}", "Results", "T_out.dat")
                        for i in range(1, n_members + 1)]
        report_summary("openda", filter_type, member_files, lake, obs_csv, openda_dir)
    else:
        raise ValueError(f"unknown engine '{engine}'; choose 'python' or 'openda'")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Regenerate the run summary (.csv + .json) from existing member T_out.dat")
    parser.add_argument("arg_file", help="Run config JSON (e.g. args/run_enkf.json)")
    parser.add_argument("--lake", default=None, help="Lake to summarize from the config's \"lakes\" block")
    parser.add_argument("-m", "--model", default=None, help="Forward model (default: arg file's \"model\", else simstrat)")
    cli = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")

    cfg   = merge_lake_args(load_json(cli.arg_file), lake=cli.lake)
    model = cli.model or cfg.get("model") or "simstrat"
    summarize_from_config(cfg, model_name=model)

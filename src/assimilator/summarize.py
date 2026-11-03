"""A run's summary: the posterior ensemble mean and spread, and its skill against the observations.

From members 1..N (not the control), written to the run's folder:

  <lake>_<engine>_<label>.csv    time, depth, T_mean, T_std
  <lake>_<engine>_<label>.json   the mean scored against the run's observations, binned as the
                                 engines assimilate them, at the depths the model outputs (model
                                 interpolated in depth, nearest output time). bias = model - obs.
                                 These observations were assimilated: an analysis fit, not a
                                 withheld test. Both engines are scored the same way.
"""
import os
import json
import logging
import numpy as np
import pandas as pd

from .observations import load_obs, filter_obs_to_model_depths
from .report import display_path

logger = logging.getLogger(__name__)


def _load_members(member_files, model):
    """Stack members 1..N -> posterior mean/std per (time, depth)."""
    if not member_files:
        raise ValueError("no member output files given")
    missing = [f for f in member_files if not os.path.isfile(f)]
    if missing:
        raise FileNotFoundError(f"missing {len(missing)} member output(s), e.g. {missing[0]}")

    times, depths, first = model.read_output(member_files[0])
    members = [first] + [model.read_output(f)[2] for f in member_files[1:]]
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


def _score(times, depths, mean, std, obs_csv, model):
    """Score posterior mean vs the obs at model-output depths; returns (overall, by_depth) or None."""
    obs = load_obs(obs_csv).dropna(subset=["value"])
    if obs.empty:
        return None

    obs = filter_obs_to_model_depths(obs, [abs(float(d)) for d in depths])
    if obs.empty:
        return None

    ref   = model.output_epoch()
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
        j0, j1, w = _depth_interp(depths, model.obs_to_sim_col(d))   # obs depth -> model coord
        m_mean[sel] = mean[ti[sel], j0] * (1 - w) + mean[ti[sel], j1] * w
        m_std[sel]  = std[ti[sel], j0] * (1 - w) + std[ti[sel], j1] * w

    err = m_mean - o_v
    overall  = _agg(err, m_std)
    by_depth = {f"{d:g}": _agg(err[o_d == d], m_std[o_d == d]) for d in np.unique(o_d)}
    return overall, by_depth


def report_summary(engine, label, member_files, lake, obs_csv, out_dir, model):
    """Write the summary of members 1..N into `out_dir`. Returns (csv path, overall skill or None)."""
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.join(out_dir, f"{lake}_{engine}_{label}")
    times, depths, mean, std = _load_members(member_files, model)
    _write_csv(times, depths, mean, std, base + ".csv")

    overall = None
    scored  = _score(times, depths, mean, std, obs_csv, model) if obs_csv and os.path.isfile(obs_csv) else None
    if scored is not None:
        overall, by_depth = scored
        with open(base + ".json", "w") as f:
            json.dump({"lake": lake, "engine": engine, "filter": label, "n_members": len(member_files),
                       "period": {"start": _r(times[0]), "end": _r(times[-1])},
                       "scored_against": "obs at model-output depths (model interpolated to obs depth, "
                                         "nearest output time); analysis fit, not withheld",
                       "bias_sign": "model - obs (+ = model too warm)",
                       "n_obs": overall["n"], "overall": overall, "by_depth": by_depth}, f, indent=2)
    logger.info(f"[summary] {len(member_files)} members, {mean.shape[0]} steps x {mean.shape[1]} depths "
                f"-> {display_path(base + '.csv')}")
    return base + ".csv", overall


def summarize_from_config(cfg):
    """Rebuild a finished run's summary from its member outputs on disk."""
    from .config import build_config
    from .models import get_model
    cfg   = build_config(cfg)
    model = get_model(cfg["model"])
    out_dir = cfg["ensemble_base"] if cfg["engine"] == "python" else cfg["openda_dir"]
    member_files = [model.member_output_file(out_dir, i, cfg.get("results_dir", "Results"))
                    for i in range(1, cfg["n_members"] + 1)]
    return report_summary(cfg["engine"], cfg["algorithm"], member_files, cfg["lake"], cfg["obs_file"],
                          out_dir, model)

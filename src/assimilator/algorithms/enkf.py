"""The native Ensemble Kalman Filter: one forecast and analysis window per observation time."""
import os
import time
import logging
import concurrent.futures
import numpy as np
import pandas as pd

from ..functions import (load_obs, filter_obs_to_model_depths, obs_window_boundaries,
                         build_python_run_args, make_progress, log_obs_summary,
                         log_run_header, log_run_footer, time_keyed_rng,
                         resolve_sigma_obs, log_sigma_summary)
from ..obs_operator import build_H
from ..summarize import report_summary

logger = logging.getLogger(__name__)


def window_observations(obs_df, window_start, window_end, model):
    """Per depth, the observation in (window_start, window_end] nearest the window end, the
    instant of the snapshot the update goes into. Returns (values, model columns, depths), or
    Nones when the window holds none. The open start keeps a boundary observation in one window."""
    obs_win = obs_df[(obs_df["time"] > window_start) & (obs_df["time"] <= window_end)]
    if obs_win.empty:
        return None, None, None
    nearest = (obs_win.assign(_d=(obs_win["time"] - pd.Timestamp(window_end)).abs())
                      .sort_values("_d")
                      .groupby("depth", sort=False).first()
                      .sort_index())
    obs_depths = list(nearest.index)
    return nearest["value"].values, [model.obs_to_sim_col(d) for d in obs_depths], obs_depths


def enkf_update(X_f, y_obs, H, sigma_obs, inflation=1.0, rng=None):
    """Stochastic EnKF analysis of the forecast ensemble X_f (cells x members) with perturbed
    observations. Returns (X_a, diagnostics), diagnostics None when no observation is valid."""
    if rng is None:
        rng = np.random.default_rng()

    valid = ~np.isnan(y_obs)
    if not valid.any():
        return X_f.copy(), None

    y   = y_obs[valid]
    H_v = H[valid]
    r   = np.full(valid.sum(), sigma_obs) if np.ndim(sigma_obs) == 0 else np.asarray(sigma_obs)[valid]
    R   = np.diag(r ** 2)

    N     = X_f.shape[1]
    x_bar = X_f.mean(axis=1, keepdims=True)
    A     = (X_f - x_bar) * inflation
    X_inf = x_bar + A

    HA   = H_v @ A
    PHT  = A @ HA.T / (N - 1)
    HPHT = HA @ HA.T / (N - 1)
    K    = np.linalg.solve((HPHT + R).T, PHT.T).T            # PHT (HPHT + R)^-1, without inverting
    eps  = rng.multivariate_normal(np.zeros(len(y)), R, size=N).T   # not recentred
    X_a  = X_inf + K @ ((y[:, None] + eps) - H_v @ X_inf)

    # NIS combines the inflated spread with the raw-mean innovation: not a measure to tune
    # inflation with.
    d   = y - (H_v @ x_bar)[:, 0]
    NIS = float(d @ np.linalg.solve(HPHT + R, d))

    diags = {
        "n_obs":       int(valid.sum()),
        "innov_mean":  round(float(d.mean()), 6),
        "innov_std":   round(float(d.std()),  6),
        "NIS":         round(NIS, 6),
        "spread_pre":  round(float(np.std(H_v @ X_f, axis=1, ddof=1).mean()), 6),
        "spread_post": round(float(np.std(H_v @ X_a, axis=1, ddof=1).mean()), 6),
        "_innov_vec":  d.tolist(),
        "_valid_mask": valid.tolist(),
        "_K":          K,
    }
    return X_a, diags


def _append_row(path, row):
    pd.DataFrame([row]).to_csv(path, mode="a", header=not os.path.exists(path), index=False)


def _write_diagnostics(args, when, diags, y_obs, obs_depths, z_vol, lake_level):
    valid_mask = diags.pop("_valid_mask")
    innov_vec  = diags.pop("_innov_vec")
    K          = diags.pop("_K")
    stamp      = when.isoformat()
    _append_row(args["diag_path"], {"date": stamp, **diags})

    innov = np.full(len(y_obs), np.nan)
    innov[np.array(valid_mask)] = innov_vec
    _append_row(args["innov_depth_path"],
                {"date": stamp, **{f"d_{d}": round(float(v), 6) for d, v in zip(obs_depths, innov)}})

    depth  = lake_level - z_vol
    order  = np.argsort(depth)
    K_mean = np.interp(np.arange(0, int(lake_level) + 1), depth[order], K.mean(axis=1)[order])
    _append_row(args["kgain_depth_path"],
                {"date": stamp, **{f"K_{d}": round(float(v), 8) for d, v in enumerate(K_mean)}})


def _analyse(args, model, members, y_obs, sim_depths, obs_depths, when):
    """Read the members' end-of-window states, update them with the window's observations and
    write them back. Returns the number of members updated."""
    def read(i):
        try:
            return i, *model.read_snapshot_T(i, args)
        except Exception as e:
            logger.warning(f"[ensemble{i:02d}] snapshot read failed: {e}")
            return i, None, None, None

    with concurrent.futures.ThreadPoolExecutor() as pool:
        states = {r[0]: r[1:] for r in pool.map(read, members)}
    readable = [i for i in members if states[i][0] is not None]
    if len(readable) < 2:
        return 0

    X_f = np.column_stack([states[i][0] for i in readable])
    # The members share one grid and lake level; the first member's stand for all.
    z_vol, lake_level = states[readable[0]][1], states[readable[0]][2]
    H = build_H(z_vol, lake_level, sim_depths)
    # A vector, so that enkf_update never sees a season-keyed dict (np.ndim of a dict is 0).
    sigma = resolve_sigma_obs(obs_depths, when, args)
    # Keyed on the analysis instant: a run continued in slices draws as one continuous run.
    rng = time_keyed_rng(args.get("rng_seed", 42), "enkf_obs", int(when.timestamp()))
    X_a, diags = enkf_update(X_f, y_obs, H, sigma, inflation=args["inflation"], rng=rng)

    def write(col_i):
        col, i = col_i
        try:
            model.write_snapshot_T(i, X_a[:, col], args)
        except Exception as e:
            logger.warning(f"[ensemble{i:02d}] snapshot write failed: {e}")

    with concurrent.futures.ThreadPoolExecutor() as pool:
        pool.map(write, enumerate(readable))
    if diags is not None:
        _write_diagnostics(args, when, diags, y_obs, obs_depths, z_vol, lake_level)
    return len(readable)


def run_enkf_loop(args, model):
    member_ids  = args["member_ids"]
    max_workers = args.get("max_workers")

    if args.get("reset"):
        model.clear_member_state(args["ensemble_base"], member_ids, args["results_dir"])
        model.clear_member_outputs(args["ensemble_base"], member_ids, args["results_dir"])
        for p in [args["mean_traj_path"], args["diag_path"], args["innov_depth_path"], args["kgain_depth_path"]]:
            if os.path.exists(p):
                os.remove(p)
        logger.info(f"Reset: cleared {args['results_dir']}/ snapshots and trajectory files.")

    obs = load_obs(args["obs_path"])
    obs = filter_obs_to_model_depths(obs, model.model_output_depths(args["ensemble_base"]))
    log_obs_summary(obs, args["obs_path"])
    start_date, end_date = args["start_date"], args["end_date"]

    # The windows end at the observation times, plus a last one to end_date without an update:
    # they depend only on the observations, so a run split into slices steps as one run does.
    boundaries = obs_window_boundaries(obs, start_date, end_date)
    logger.info(f"EnKF: {start_date.date()} → {end_date.date()} ({len(boundaries)} windows, "
                f"{len(member_ids)} members, inflation={args['inflation']})")
    log_sigma_summary(args)

    model.start_containers(args, max_workers=max_workers)
    try:
        windows_updated = 0
        progress = args.get("progress", False)
        bar      = make_progress(max(1, len(boundaries)), progress, desc=f"EnKF {args['lake']}")
        current  = start_date
        for window_end in boundaries:
            t0 = time.perf_counter()
            failed = model.run_window(current, window_end, args, max_workers=max_workers)
            t_model = time.perf_counter() - t0

            y_obs, sim_depths, obs_depths = window_observations(obs, current, window_end, model)
            n_updated = 0
            if y_obs is not None:
                good = [i for i in member_ids if i not in failed]
                if len(good) >= 2:
                    n_updated = _analyse(args, model, good, y_obs, sim_depths, obs_depths, window_end)
                    windows_updated += n_updated > 0

            obs_str = f"n_obs={len(y_obs)}  n_updated={n_updated}" if y_obs is not None else "no obs"
            status  = f"failed={failed}" if failed else "ok"
            logger.info(f"  {current:%Y-%m-%d %H:%M} -> {window_end:%Y-%m-%d %H:%M}  {obs_str}  [{status}]  "
                        f"[model={t_model:.1f}s  total={time.perf_counter() - t0:.1f}s]",
                        extra={"file_only": True})
            if progress:
                bar.set_postfix_str(f"{window_end:%Y-%m-%d %H:%M}  {obs_str}  [{status}]")
                bar.update(1)
            current = window_end

        bar.close()
        member_files = [model.member_output_file(args["ensemble_base"], i, args["results_dir"])
                        for i in member_ids]
        model.accumulate_mean(member_files, args["mean_traj_path"])
        logger.info(f"Done. {len(boundaries)} windows run, {windows_updated} EnKF updates applied.")
        return len(boundaries), windows_updated
    finally:
        model.stop_containers(args)


def run_enkf(run_raw, ensemble_raw, ensemble_base, n_members, model):
    """Run the native EnKF and write its summary and skill report to ensemble_base."""
    args = build_python_run_args(run_raw, ensemble_raw, ensemble_base, n_members, model)
    log_run_header(ensemble_raw)
    t0 = time.perf_counter()
    windows_run, windows_updated = run_enkf_loop(args, model)

    member_files = [model.member_output_file(ensemble_base, i, args["results_dir"])
                    for i in args["member_ids"]]
    out_csv, skill = report_summary("python", "EnKF", member_files, args["lake"], args["obs_path"],
                                   ensemble_base, model)
    log_run_footer(ensemble_raw, skill, windows_run, windows_updated, time.perf_counter() - t0, out_csv)

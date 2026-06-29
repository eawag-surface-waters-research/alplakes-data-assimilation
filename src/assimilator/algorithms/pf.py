import os
import shutil
import time
import math
import concurrent.futures
import numpy as np
from datetime import timedelta

import logging

from ..functions import (load_obs, filter_obs_to_model_depths,
                         verify_args, build_python_run_args, make_progress, log_obs_summary,
                         log_run_header, log_run_footer)
from ..summarize import report_summary

logger = logging.getLogger(__name__)

REQUIRED_RUN = ["algorithm", "results_dir", "par_file"]

# ---------------------------------------------------------------------------
# The Python Particle Filter (a simple "best member, resample-to-all" scheme), run as a daily-window loop.
# ---------------------------------------------------------------------------
# Note: this is NOT a real Bayesian particle filter. There are no importance weights, no
# likelihood, and no resampling proportional to fit -- copy_best_to_all() just clones the single
# lowest-RMSE member onto every other member each update day, collapsing ensemble spread to zero
# (only the next day's per-member forcing perturbation re-diversifies). We call this
# "PF", which oversells it because it is a "best-member selection". Intentional and might be changed.

def compute_depth_weights(obs_df, model):
    depths = np.sort(obs_df["depth"].unique()).astype(float)
    n = len(depths)
    w = np.empty(n)
    if n == 1:
        w[0] = 1.0
    else:
        w[0]    = (depths[1]  - depths[0])  / 2
        w[-1]   = (depths[-1] - depths[-2]) / 2
        w[1:-1] = (depths[2:] - depths[:-2]) / 2
    return {model.obs_to_sim_col(d): float(wt) for d, wt in zip(depths, w)}


def rmse_in_window(sim_df, obs_df, window_start, window_end, depth_weights, model):
    obs_win = obs_df[(obs_df["time"] >= window_start) & (obs_df["time"] < window_end)]
    n_obs_raw = len(obs_win)
    if obs_win.empty:
        return np.nan, 0, 0

    obs_win = obs_win.copy()
    obs_win["sim_col"] = obs_win["depth"].map(model.obs_to_sim_col)

    obs_pivot    = obs_win.pivot_table(index="time", columns="sim_col", values="value", aggfunc="mean")
    common_times = sim_df.index.intersection(obs_pivot.index)
    if len(common_times) == 0:
        return np.nan, n_obs_raw, 0

    common_cols = [c for c in obs_pivot.columns if c in sim_df.columns]
    sim_vals = sim_df.loc[common_times, common_cols].values
    obs_vals = obs_pivot.loc[common_times, common_cols].values
    mask     = ~np.isnan(obs_vals)

    col_w   = np.array([depth_weights.get(c, 1.0) for c in common_cols])
    w_mat   = np.where(mask, col_w[np.newaxis, :], 0.0)
    sq_err  = np.where(mask, (sim_vals - obs_vals) ** 2, 0.0)
    total_w = w_mat.sum()
    rmse    = np.sqrt((sq_err * w_mat).sum() / total_w) if total_w > 0 else np.nan

    return rmse, n_obs_raw, int(mask.sum())


# "resample" step -- clones best member's snapshot onto all others. Intentional.
def copy_best_to_all(best_id, member_ids, args):
    src = os.path.join(args["ensemble_base"], f"ensemble{best_id}", args["results_dir"], "simulation-snapshot.dat")
    targets = [
        os.path.join(args["ensemble_base"], f"ensemble{i}", args["results_dir"], "simulation-snapshot.dat")
        for i in member_ids if i != best_id
    ]
    with concurrent.futures.ThreadPoolExecutor() as pool:
        pool.map(lambda dst: shutil.copy2(src, dst), targets)


def run_pf_daily(args, model):
    member_ids  = args["member_ids"]
    max_workers = args.get("max_workers")

    if args.get("reset"):
        for i in member_ids:
            live = os.path.join(args["ensemble_base"], f"ensemble{i}", args["results_dir"], "simulation-snapshot.dat")
            if os.path.exists(live):
                os.remove(live)
        model.clear_member_outputs(args["ensemble_base"], member_ids, args["results_dir"])
        if os.path.exists(args["mean_traj_path"]):
            os.remove(args["mean_traj_path"])
        logger.info(f"Reset: cleared {args['results_dir']}/ snapshots and trajectory files.")

    obs           = load_obs(args["obs_path"])
    obs           = filter_obs_to_model_depths(obs, model.model_output_depths(args["ensemble_base"]))
    log_obs_summary(obs, args["obs_path"])
    depth_weights = compute_depth_weights(obs, model)
    start_date    = args["start_date"]
    end_date      = args["end_date"]

    logger.info(f"Daily PF: {start_date.date()} → {end_date.date()} "
                f"({(end_date - start_date).days} days, {len(member_ids)} members)")
    logger.info(f"Depth weights: { {d: round(w, 2) for d, w in depth_weights.items()} }")

    model.start_containers(args, max_workers=max_workers)
    try:
        # Noon-anchor the daily windows so PF scores noon-to-noon, matching the EnKF
        # reference (whose default window_end lands on noon) instead of the midnight-to-
        # midnight a plain-date start_date would give. start_date stays a plain date
        # (shared with EnKF and the OpenDA renderer); the shift is applied locally here.
        noon        = start_date.replace(hour=12, minute=0, second=0, microsecond=0)
        current     = noon if noon >= start_date else noon + timedelta(days=1)
        logger.info(f"Noon-anchored windows: first window {current.isoformat()} → {(current + timedelta(days=1)).isoformat()}")
        days_run    = 0
        days_copied = 0

        # One progress bar per run instead of a log line per day (see enkf.py).
        # Disabled for server runs via args["progress"]; when off the per-day
        # logger.info lines below still fire. total tolerates clamped-window overflow.
        progress = args.get("progress", False)
        total    = max(1, math.ceil((end_date - current).total_seconds() / 86400))
        bar      = make_progress(total, progress, desc=f"PF {args['lake']}")

        while current < end_date:
            window_end = min(current + timedelta(days=1), end_date)
            t_day      = time.perf_counter()

            t0       = time.perf_counter()
            failed   = model.run_window(current, window_end, args, max_workers=max_workers)
            days_run += 1
            t_docker = time.perf_counter() - t0

            def _load_and_score(i):
                if i in failed:
                    return i, np.nan, 0, 0
                try:
                    # T_out.dat accumulates the whole run (Simstrat appends), so we tail-read only
                    # the rows appended for THIS window (resuming from the per-member offset sidecar)
                    # instead of re-parsing the ever-growing file every window — O(1) per window vs
                    # the old load_T's O(n) (O(n^2) over the run). Shares read_t_out_tail with the
                    # OpenDA wrapper. rmse_in_window scores the same window as before, so it's
                    # correctness-neutral. The full T_out.dat stays intact for accumulate_mean.
                    sim = model.load_T_window(os.path.join(args["ensemble_base"], f"ensemble{i}"),
                                              args, current, window_end)
                    rmse, n_raw, n_matched = rmse_in_window(sim, obs, current, window_end, depth_weights, model)
                    return i, rmse, n_raw, n_matched
                except Exception:
                    return i, np.nan, 0, 0

            t0 = time.perf_counter()
            with concurrent.futures.ThreadPoolExecutor() as pool:
                scores = {r[0]: r[1:] for r in pool.map(_load_and_score, member_ids)}
            t_score = time.perf_counter() - t0

            rmses     = [scores[i][0] for i in member_ids]
            n_obs_raw = max((scores[i][1] for i in member_ids), default=0)
            n_matched = max((scores[i][2] for i in member_ids), default=0)
            valid     = [(i, r) for i, r in zip(member_ids, rmses) if not np.isnan(r)]

            t_total = time.perf_counter() - t_day
            timing  = f"docker={t_docker:.1f}s  score={t_score:.1f}s  total={t_total:.1f}s"

            if valid:
                best_id   = min(valid, key=lambda x: x[1])[0]
                best_rmse = min(r for _, r in valid)
                copy_best_to_all(best_id, member_ids, args)
                days_copied += 1
                status = f"failed={failed}" if failed else "ok"
                # Per-day detail always to the log file (file_only keeps it off the console —
                # see main.py); the console shows the bar instead when progress is on.
                logger.info(f"  {current.date()}  best=ensemble{best_id:02d}  RMSE={best_rmse:.4f} °C  "
                            f"obs_raw={n_obs_raw}  matched={n_matched}  [{status}]  [{timing}]",
                            extra={"file_only": True})
                if progress:
                    bar.set_postfix_str(f"{current.date()}  best=ens{best_id:02d}  RMSE={best_rmse:.3f}  [{status}]")
            else:
                obs_win = obs[(obs["time"] >= current) & (obs["time"] < window_end)]
                status  = f"  failed={failed}" if failed else ""
                logger.info(f"  {current.date()}  no obs — snapshots unchanged  "
                            f"obs_raw={len(obs_win)}  matched={n_matched}{status}  [{timing}]",
                            extra={"file_only": True})
                if progress:
                    bar.set_postfix_str(f"{current.date()}  no obs{status}")
            if progress:
                bar.update(1)

            current = window_end

        bar.close()
        model.accumulate_mean(member_ids, args)   # one-shot: ensemble-mean trajectory from full T_out.dat
        logger.info(f"Done. {days_run} windows run, {days_copied} best-copy steps applied.")
        return days_run, days_copied

    finally:
        model.stop_containers(args)


# ---------------------------------------------------------------------------
# End-to-end PF run (validate -> build args -> daily loop -> summarise)
# ---------------------------------------------------------------------------

def run_pf(run_raw, ensemble_raw, ensemble_base, n_members, model):
    """Native PF engine driver: validate run args, build the merged args, run the
    daily PF loop, then write the posterior summary + skill report to the run folder (run/<lake>/).
    `model` is the selected forward model (see assimilator.models)."""
    verify_args(run_raw, REQUIRED_RUN)
    args = build_python_run_args(run_raw, ensemble_raw, ensemble_base, n_members, model)

    log_run_header(ensemble_raw)
    t0 = time.perf_counter()
    days_run, days_copied = run_pf_daily(args, model)

    member_files = [os.path.join(ensemble_base, f"ensemble{i}", args["results_dir"], "T_out.dat")
                    for i in args["member_ids"]]
    out_csv, skill = report_summary("python", "PF", member_files, args["lake"], args["obs_path"], ensemble_base)
    log_run_footer(ensemble_raw, skill, days_run, days_copied, time.perf_counter() - t0, out_csv)

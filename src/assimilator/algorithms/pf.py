"""The native "particle filter": best-member selection over one window per observation time.

Not a Bayesian particle filter: there are no weights or proportional resampling. After each window
the lowest-RMSE member is copied onto all others, collapsing the spread, which the next window's
forcing perturbation rebuilds. Kept as a simple, deliberate scheme.
"""
import os
import time
import logging
import concurrent.futures
import numpy as np

from ..functions import (load_obs, filter_obs_to_model_depths, obs_window_boundaries,
                         build_python_run_args, make_progress, log_obs_summary,
                         log_run_header, log_run_footer)
from ..summarize import report_summary

logger = logging.getLogger(__name__)


def compute_depth_weights(obs_df, model):
    """Each observed depth weighted by the thickness of water it represents."""
    depths = np.sort(obs_df["depth"].unique()).astype(float)
    w = np.empty(len(depths))
    if len(depths) == 1:
        w[0] = 1.0
    else:
        w[0]    = (depths[1]  - depths[0])  / 2
        w[-1]   = (depths[-1] - depths[-2]) / 2
        w[1:-1] = (depths[2:] - depths[:-2]) / 2
    return {model.obs_to_sim_col(d): float(wt) for d, wt in zip(depths, w)}


def rmse_in_window(sim_df, obs_df, window_start, window_end, depth_weights, model):
    """Depth-weighted RMSE of one member against the observations in (window_start, window_end].
    Returns (rmse, observations in the window, observations matched to an output row)."""
    obs_win = obs_df[(obs_df["time"] > window_start) & (obs_df["time"] <= window_end)]
    if obs_win.empty:
        return np.nan, 0, 0

    obs_win = obs_win.assign(sim_col=obs_win["depth"].map(model.obs_to_sim_col))
    obs_pivot    = obs_win.pivot_table(index="time", columns="sim_col", values="value", aggfunc="mean")
    common_times = sim_df.index.intersection(obs_pivot.index)
    if len(common_times) == 0:
        return np.nan, len(obs_win), 0

    common_cols = [c for c in obs_pivot.columns if c in sim_df.columns]
    sim_vals = sim_df.loc[common_times, common_cols].values
    obs_vals = obs_pivot.loc[common_times, common_cols].values
    mask     = ~np.isnan(obs_vals)
    col_w    = np.array([depth_weights.get(c, 1.0) for c in common_cols])
    w_mat    = np.where(mask, col_w[np.newaxis, :], 0.0)
    sq_err   = np.where(mask, (sim_vals - obs_vals) ** 2, 0.0)
    total_w  = w_mat.sum()
    rmse     = np.sqrt((sq_err * w_mat).sum() / total_w) if total_w > 0 else np.nan
    return rmse, len(obs_win), int(mask.sum())


def run_pf_loop(args, model):
    member_ids  = args["member_ids"]
    max_workers = args.get("max_workers")

    if args.get("reset"):
        model.clear_member_state(args["ensemble_base"], member_ids, args["results_dir"])
        model.clear_member_outputs(args["ensemble_base"], member_ids, args["results_dir"])
        if os.path.exists(args["mean_traj_path"]):
            os.remove(args["mean_traj_path"])
        logger.info(f"Reset: cleared {args['results_dir']}/ snapshots and trajectory files.")

    obs = load_obs(args["obs_path"])
    obs = filter_obs_to_model_depths(obs, model.model_output_depths(args["ensemble_base"]))
    log_obs_summary(obs, args["obs_path"])
    depth_weights = compute_depth_weights(obs, model)
    logger.info(f"Depth weights: { {d: round(w, 2) for d, w in depth_weights.items()} }")
    start_date, end_date = args["start_date"], args["end_date"]

    boundaries = obs_window_boundaries(obs, start_date, end_date)
    logger.info(f"PF: {start_date.date()} → {end_date.date()} ({len(boundaries)} windows, "
                f"{len(member_ids)} members)")

    model.start_containers(args, max_workers=max_workers)
    try:
        windows_copied = 0
        progress = args.get("progress", False)
        bar      = make_progress(max(1, len(boundaries)), progress, desc=f"PF {args['lake']}")
        current  = start_date
        for window_end in boundaries:
            t0 = time.perf_counter()
            failed = model.run_window(current, window_end, args, max_workers=max_workers)

            def score(i):
                if i in failed:
                    return i, np.nan, 0, 0
                try:
                    sim = model.load_T_window(os.path.join(args["ensemble_base"], f"ensemble{i}"),
                                              args, current, window_end)
                    return (i, *rmse_in_window(sim, obs, current, window_end, depth_weights, model))
                except Exception:
                    return i, np.nan, 0, 0

            with concurrent.futures.ThreadPoolExecutor() as pool:
                scores = {r[0]: r[1:] for r in pool.map(score, member_ids)}
            n_obs     = max(s[1] for s in scores.values())
            n_matched = max(s[2] for s in scores.values())
            valid     = [(i, s[0]) for i, s in scores.items() if not np.isnan(s[0])]
            status    = f"failed={failed}" if failed else "ok"
            window    = f"{current:%Y-%m-%d %H:%M} -> {window_end:%Y-%m-%d %H:%M}"

            if valid:
                best_id, best_rmse = min(valid, key=lambda x: x[1])
                model.copy_member_state(best_id, member_ids, args)
                windows_copied += 1
                logger.info(f"  {window}  best=ensemble{best_id:02d}  RMSE={best_rmse:.4f} °C  "
                            f"obs={n_obs}  matched={n_matched}  [{status}]  [{time.perf_counter() - t0:.1f}s]",
                            extra={"file_only": True})
                postfix = f"best=ens{best_id:02d}  RMSE={best_rmse:.3f}  [{status}]"
            else:
                logger.info(f"  {window}  no obs — snapshots unchanged  obs={n_obs}  matched={n_matched}  "
                            f"[{status}]  [{time.perf_counter() - t0:.1f}s]", extra={"file_only": True})
                postfix = f"no obs  [{status}]"
            if progress:
                bar.set_postfix_str(f"{window_end:%Y-%m-%d %H:%M}  {postfix}")
                bar.update(1)
            current = window_end

        bar.close()
        member_files = [model.member_output_file(args["ensemble_base"], i, args["results_dir"])
                        for i in member_ids]
        model.accumulate_mean(member_files, args["mean_traj_path"])
        logger.info(f"Done. {len(boundaries)} windows run, {windows_copied} best-copy steps applied.")
        return len(boundaries), windows_copied
    finally:
        model.stop_containers(args)


def run_pf(run_raw, ensemble_raw, ensemble_base, n_members, model):
    """Run the native PF and write its summary and skill report to ensemble_base."""
    args = build_python_run_args(run_raw, ensemble_raw, ensemble_base, n_members, model)
    log_run_header(ensemble_raw)
    t0 = time.perf_counter()
    windows_run, windows_copied = run_pf_loop(args, model)

    member_files = [model.member_output_file(ensemble_base, i, args["results_dir"])
                    for i in args["member_ids"]]
    out_csv, skill = report_summary("python", "PF", member_files, args["lake"], args["obs_path"],
                                   ensemble_base, model)
    log_run_footer(ensemble_raw, skill, windows_run, windows_copied, time.perf_counter() - t0, out_csv)

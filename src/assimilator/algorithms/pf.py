"""The native "particle filter": best-member selection over one window per observation time.

Not a Bayesian particle filter: there are no weights or proportional resampling. After each window
the lowest-RMSE member is copied onto all others, collapsing the spread, which the next window's
forcing perturbation rebuilds. Kept as a simple, deliberate scheme.
"""
import os
import logging
import concurrent.futures
import numpy as np

from .common import reset_members, run_observations, step_windows

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
    if args.get("reset"):
        reset_members(args, model)
    obs = run_observations(args, model)
    depth_weights = compute_depth_weights(obs, model)
    logger.info(f"Depth weights: { {d: round(w, 2) for d, w in depth_weights.items()} }")

    def update(window_start, window_end, failed):
        def score(i):
            if i in failed:
                return i, np.nan, 0, 0
            try:
                sim = model.load_T_window(os.path.join(args["ensemble_base"], f"ensemble{i}"),
                                          args, window_start, window_end)
                return (i, *rmse_in_window(sim, obs, window_start, window_end, depth_weights, model))
            except Exception:
                return i, np.nan, 0, 0

        with concurrent.futures.ThreadPoolExecutor() as pool:
            scores = {r[0]: r[1:] for r in pool.map(score, args["member_ids"])}
        counts = f"obs={max(s[1] for s in scores.values())}  matched={max(s[2] for s in scores.values())}"
        valid  = [(i, s[0]) for i, s in scores.items() if not np.isnan(s[0])]
        if not valid:
            return False, f"no obs — snapshots unchanged  {counts}"
        best_id, best_rmse = min(valid, key=lambda x: x[1])
        model.copy_member_state(best_id, args["member_ids"], args)
        return True, f"best=ensemble{best_id:02d}  RMSE={best_rmse:.4f} °C  {counts}"

    return step_windows(args, model, obs, update)

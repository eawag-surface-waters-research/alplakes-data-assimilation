import os
import time
import math
import logging
import concurrent.futures
import numpy as np
import pandas as pd
from datetime import timedelta

from ..functions import (load_obs, filter_obs_to_model_depths,
                         verify_args, build_python_run_args, make_progress, log_obs_summary,
                         log_run_header, log_run_footer)
from ..summarize import report_summary

logger = logging.getLogger(__name__)

REQUIRED_RUN  = ["algorithm", "results_dir", "par_file"]
REQUIRED_ENKF = ["inflation"]               # run config; Python-EnKF-only knob
REQUIRED_ENKF_ENSEMBLE = ["sigma_obs"]      # run config; obs error std, shared with OpenDA


# ---------------------------------------------------------------------------
# The Python Ensemble Kalman Filter, run as a daily-window loop.
# ---------------------------------------------------------------------------

# Note: this averages every depth's observations over the WHOLE daily window,
# then assimilates the daily mean into the end-of-window (instantaneous) snapshot.
# Representativeness mismatch (daily-mean obs vs instantaneous state) --> intended but suboptimal compromise.
# Comparing against the model's daily-mean trajectory instead isn't as simple of option as for PF: the
# EnKF needs the FULL grid state from the snapshot (~2x the output depths) to write the
# analysis back as the next IC, and T_out.dat only holds the output depths. Future dev:
# either output the full state from the model, or interpolate the snapshot state onto the
# output depths for the comparison (compute-efficiency tradeoff).
def window_obs_vector(obs_df, window_start, window_end, model):
    obs_win = obs_df[(obs_df["time"] >= window_start) & (obs_df["time"] < window_end)]
    if obs_win.empty:
        return None, None, None
    mean_per_depth = obs_win.groupby("depth")["value"].mean().dropna()
    if mean_per_depth.empty:
        return None, None, None
    obs_depths = list(mean_per_depth.index)
    sim_depths = [model.obs_to_sim_col(d) for d in obs_depths]
    return mean_per_depth.values, sim_depths, obs_depths


def window_obs_vector_consistent(obs_df, window_start, window_end, model):
    """Pick, per depth, the single observation nearest the window END (vs window_obs_vector's
    daily mean). The EnKF assimilates into the end-of-window snapshot, so the obs nearest that
    instant is temporally consistent with the model state. window_end must be noon to match the
    OpenDA reference, which the daily loop ensures by anchoring the windows noon-to-noon.

    Window is (window_start, window_end]: the inclusive upper bound selects the bin labeled
    exactly window_end (the centered noon bin from load_obs) — a half-open `<` would drop it and
    pick the hour before; the exclusive lower bound keeps each boundary bin owned by a single
    window, so it can't be assimilated on two consecutive days."""
    obs_win = obs_df[(obs_df["time"] > window_start) & (obs_df["time"] <= window_end)]
    if obs_win.empty:
        return None, None, None
    nearest = (obs_win.assign(_d=(obs_win["time"] - pd.Timestamp(window_end)).abs())
                      .sort_values("_d")
                      .groupby("depth", sort=False).first()
                      .sort_index())
    if nearest.empty:
        return None, None, None
    obs_depths = list(nearest.index)
    sim_depths = [model.obs_to_sim_col(d) for d in obs_depths]
    return nearest["value"].values, sim_depths, obs_depths


# inherits the surface-alignment assumption from read_snapshot_T (z_volume
# slicing). converts "X metres below the surface" into an actual height above the bottom
def build_H(z_volume, lake_level, sim_depths):
    H = np.zeros((len(sim_depths), len(z_volume)))
    for row, d in enumerate(sim_depths):
        z_target = lake_level + d
        H[row, int(np.argmin(np.abs(z_volume - z_target)))] = 1.0 # finds the model cell whose height is closest to that target
    return H

# wikipedia cross checked
def enkf_update(X_f, y_obs, H, sigma_obs, inflation=1.0, rng=None):
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
    # Numerical decision taken fully by Clude Code: Kalman gain K = PHT @ inv(HPHT+R), 
    # via solve (not inv) for stability; transposes turn the right-inverse into solve's
    # left-inverse (S symmetric, so S.T == S).
    K    = np.linalg.solve((HPHT + R).T, PHT.T).T
    # Minor note: perturbed-obs noise is not recentered
    eps  = rng.multivariate_normal(np.zeros(len(y)), R, size=N).T
    innov = (y[:, None] + eps) - H_v @ X_inf
    X_a   = X_inf + K @ innov

    d   = y - (H_v @ x_bar)[:, 0]
    # Note: S uses the INFLATED anomalies (HPHT from A), but d uses the raw mean.
    # don't use it directly to tune inflation.
    S   = HPHT + R
    NIS = float(d @ np.linalg.solve(S, d))

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


def run_enkf_daily(args, model):
    member_ids  = args["member_ids"]
    sigma_obs   = args["sigma_obs"]
    inflation   = args["inflation"]
    max_workers = args.get("max_workers")
    diag_path   = args["diag_path"]
    innov_path  = args["innov_depth_path"]
    kgain_path  = args["kgain_depth_path"]

    if args.get("reset"):
        for i in member_ids:
            live = os.path.join(args["ensemble_base"], f"ensemble{i}", args["results_dir"], "simulation-snapshot.dat")
            if os.path.exists(live):
                os.remove(live)
        model.clear_member_outputs(args["ensemble_base"], member_ids, args["results_dir"])
        for p in [args["mean_traj_path"], diag_path, innov_path, kgain_path]:
            if os.path.exists(p):
                os.remove(p)
        logger.info(f"Reset: cleared {args['results_dir']}/ snapshots and trajectory files.")

    obs        = load_obs(args["obs_path"])
    obs        = filter_obs_to_model_depths(obs, model.model_output_depths(args["ensemble_base"]))
    log_obs_summary(obs, args["obs_path"])
    start_date = args["start_date"]
    end_date   = args["end_date"]
    rng        = np.random.default_rng()

    # Observation selector per daily window (run-arg "obs_selector", default "window_end"):
    #   "window_end" -> window_obs_vector_consistent (single reading nearest window_end;
    #                   DEFAULT — instantaneous update at the snapshot instant. With a noon-
    #                   anchored start_date (e.g. ...T12:00:00) window_end == noon, matching
    #                   the OpenDA reference's instantaneous-noon assimilation.)
    #   "mean"       -> window_obs_vector            (daily mean per depth; opt-in)
    _OBS_SELECTORS = {"mean": window_obs_vector, "window_end": window_obs_vector_consistent}
    obs_selector_name = args.get("obs_selector", "window_end")
    if obs_selector_name not in _OBS_SELECTORS:
        raise ValueError(f"unknown obs_selector '{obs_selector_name}'; choose from {sorted(_OBS_SELECTORS)}")
    select_obs = _OBS_SELECTORS[obs_selector_name]

    logger.info(f"Daily EnKF: {start_date.date()} → {end_date.date()} "
                f"({(end_date - start_date).days} days, {len(member_ids)} members, "
                f"σ_obs={sigma_obs} °C, inflation={inflation}, obs_selector={obs_selector_name})")

    model.start_containers(args, max_workers=max_workers)
    try:
        # Instantaneous-noon update (default obs_selector="window_end"): anchor the windows so
        # each window_end lands on noon, matching the OpenDA reference's noon assimilation.
        # Equivalent to a <start_date>T12:00 start, but done here so the shared start_date (also
        # consumed by PF and the OpenDA config renderer) stays a plain date. The warmup snapshot
        # becomes the IC at this first noon.
        current = start_date
        if obs_selector_name == "window_end":
            noon    = start_date.replace(hour=12, minute=0, second=0, microsecond=0)
            current = noon if noon >= start_date else noon + timedelta(days=1)
            logger.info(f"Noon-anchored windows: first window {current.isoformat()} → {(current + timedelta(days=1)).isoformat()}")
        days_run     = 0
        days_updated = 0

        # One progress bar per run instead of a log line per day. Disabled for
        # server runs (resolved upstream into args["progress"]); when off, the
        # per-day logger.info below still fires. total is the window count from
        # the dates — tqdm tolerates overflow if the last (clamped) window nudges it.
        progress = args.get("progress", False)
        total    = max(1, math.ceil((end_date - current).total_seconds() / 86400))
        bar      = make_progress(total, progress, desc=f"EnKF {args['lake']}")

        while current < end_date:
            window_end = min(current + timedelta(days=1), end_date)
            t_day      = time.perf_counter()

            t0       = time.perf_counter()
            failed   = model.run_window(current, window_end, args, max_workers=max_workers)
            days_run += 1
            t_docker = time.perf_counter() - t0

            y_obs, sim_depths, obs_depths = select_obs(obs, current, window_end, model)

            t_enkf    = 0.0
            n_updated = 0
            if y_obs is not None:
                good_ids = [i for i in member_ids if i not in failed]
                if len(good_ids) >= 2:
                    t0 = time.perf_counter()

                    def _read_T(i):
                        try:
                            return i, *model.read_snapshot_T(i, args)
                        except Exception as e:
                            logger.warning(f"[ensemble{i:02d}] snapshot read failed: {e}")
                            return i, None, None, None

                    with concurrent.futures.ThreadPoolExecutor() as pool:
                        snap_data = {r[0]: r[1:] for r in pool.map(_read_T, good_ids)}

                    readable = [i for i in good_ids if snap_data[i][0] is not None]
                    if len(readable) >= 2:
                        X_f      = np.column_stack([snap_data[i][0] for i in readable])
                        z_vol    = snap_data[readable[0]][1]
                        lake_lev = snap_data[readable[0]][2]

                        # Note: Assuming lake levels of different members the same, T column too. Intentional.
                        H          = build_H(z_vol, lake_lev, sim_depths)
                        X_a, diags = enkf_update(X_f, y_obs, H, sigma_obs, inflation=inflation, rng=rng)

                        def _write_T(col_i):
                            col, i = col_i
                            try:
                                model.write_snapshot_T(i, X_a[:, col], args)
                            except Exception as e:
                                logger.warning(f"[ensemble{i:02d}] snapshot write failed: {e}")

                        with concurrent.futures.ThreadPoolExecutor() as pool:
                            pool.map(_write_T, enumerate(readable))

                        n_updated    = len(readable)
                        days_updated += 1

                        if diags is not None:
                            valid_mask = diags.pop("_valid_mask")
                            innov_vec  = diags.pop("_innov_vec")
                            K_arr      = diags.pop("_K")

                            pd.DataFrame([{"date": current.date(), **diags}]).to_csv(
                                diag_path, mode="a",
                                header=not os.path.exists(diag_path), index=False,
                            )

                            full_innov = np.full(len(y_obs), np.nan)
                            full_innov[np.array(valid_mask)] = innov_vec
                            pd.DataFrame([{
                                "date": current.date(),
                                **{f"d_{d}": round(float(v), 6) for d, v in zip(obs_depths, full_innov)}
                            }]).to_csv(innov_path, mode="a", header=not os.path.exists(innov_path), index=False)

                            depth_fs   = lake_lev - z_vol
                            K_mean     = K_arr.mean(axis=1)
                            sort_idx   = np.argsort(depth_fs)
                            std_depths = np.arange(0, int(lake_lev) + 1)
                            K_interp   = np.interp(std_depths, depth_fs[sort_idx], K_mean[sort_idx])
                            pd.DataFrame([{
                                "date": current.date(),
                                **{f"K_{d}": round(float(v), 8) for d, v in enumerate(K_interp)}
                            }]).to_csv(kgain_path, mode="a", header=not os.path.exists(kgain_path), index=False)

                    t_enkf = time.perf_counter() - t0

            t_total = time.perf_counter() - t_day
            timing  = f"docker={t_docker:.1f}s  enkf={t_enkf:.1f}s  total={t_total:.1f}s"
            obs_str = f"n_obs={len(y_obs)}  n_updated={n_updated}" if y_obs is not None else "no obs"
            status  = f"failed={failed}" if failed else "ok"
            # Per-day detail always goes to the log file (file_only keeps it off the console —
            # see main.py); the console shows the bar instead when progress is on.
            logger.info(f"  {current.date()}  {obs_str}  [{status}]  [{timing}]",
                        extra={"file_only": True})
            if progress:
                bar.set_postfix_str(f"{current.date()}  {obs_str}  [{status}]")
                bar.update(1)

            current = window_end

        bar.close()
        model.accumulate_mean(member_ids, args)   # one-shot: ensemble-mean trajectory from full T_out.dat
        logger.info(f"Done. {days_run} days run, {days_updated} EnKF updates applied.")
        return days_run, days_updated

    finally:
        model.stop_containers(args)


# ---------------------------------------------------------------------------
# End-to-end EnKF run (validate -> build args -> daily loop -> summarise)
# ---------------------------------------------------------------------------

def run_enkf(run_raw, ensemble_raw, ensemble_base, n_members, model):
    """Native EnKF engine driver: validate run args, build the merged args, run the
    daily EnKF loop, then write the posterior summary + skill report to the run folder (run/<lake>/).
    `model` is the selected forward model (see assimilator.models)."""
    verify_args(run_raw, REQUIRED_RUN + REQUIRED_ENKF)
    verify_args(ensemble_raw, REQUIRED_ENKF_ENSEMBLE)
    args = build_python_run_args(run_raw, ensemble_raw, ensemble_base, n_members, model)

    log_run_header(ensemble_raw)
    t0 = time.perf_counter()
    days_run, days_updated = run_enkf_daily(args, model)

    member_files = [os.path.join(ensemble_base, f"ensemble{i}", args["results_dir"], "T_out.dat")
                    for i in args["member_ids"]]
    out_csv, skill = report_summary("python", "EnKF", member_files, args["lake"], args["obs_path"], ensemble_base)
    log_run_footer(ensemble_raw, skill, days_run, days_updated, time.perf_counter() - t0, out_csv)

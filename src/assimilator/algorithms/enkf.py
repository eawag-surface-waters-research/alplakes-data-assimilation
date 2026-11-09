"""The native Ensemble Kalman Filter: one forecast and analysis window per observation time."""
import os
import logging
import concurrent.futures
import numpy as np
import pandas as pd

from .common import reset_members, run_observations, step_windows
from ..obs_operator import build_H
from ..rng import time_keyed_rng
from ..sigma import resolve_sigma_obs, log_sigma_summary
from ..online_errors import OnlineRun, draw_noise, stabilize
from ..localization import REACH_MIN, load_radii, taper, summarize

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


def enkf_update(X_f, y_obs, H, sigma_obs, inflation=1.0, rng=None, loc_so=None, loc_oo=None):
    """Stochastic EnKF analysis of the forecast ensemble X_f (cells x members) with perturbed
    observations. Returns (X_a, diagnostics), diagnostics None when no observation is valid.
    loc_so (cells x obs) and loc_oo (obs x obs) are the localization tapers over all observation
    rows, given together or not at all."""
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
    infl  = inflation
    if loc_so is not None:
        # Inflate only the cells an observation can reach: out of reach the gain is 0 and nothing
        # would pull the inflated spread back.
        loc_so, loc_oo = loc_so[:, valid], loc_oo[np.ix_(valid, valid)]
        infl = np.where(loc_so.max(axis=1) >= REACH_MIN, inflation, 1.0)[:, None]
    A     = (X_f - x_bar) * infl
    X_inf = x_bar + A

    HA   = H_v @ A
    PHT  = A @ HA.T / (N - 1)
    HPHT = HA @ HA.T / (N - 1)
    if loc_so is not None:
        PHT, HPHT = PHT * loc_so, HPHT * loc_oo
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


def _nis_check(online, y_obs, HX_f, sigma, when):
    """y_obs with the observations whose NIS exceeds max_nis set to NaN (once the check is on)."""
    y = np.asarray(y_obs, dtype=float).copy()
    if online.oe.checking(when):
        d_b = y - HX_f.mean(axis=1)
        nis = d_b ** 2 / (HX_f.var(axis=1, ddof=1) + sigma ** 2)
        y[nis > float(online.oe.spec["max_nis"])] = np.nan
    return y


def _online_after(args, model, online, members, X_f, X_a, y_obs, y_used, H, sigma,
                  z_vol, lake_level, obs_depths, when):
    """Learn R and q from this analysis, then add the model-error noise.
    Returns (X_a, n_rejected, n_stabilized)."""
    oe = online.oe
    y_obs = np.asarray(y_obs, dtype=float)
    HX_f = H @ X_f
    d_b = y_obs - HX_f.mean(axis=1)
    d_a = np.where(np.isnan(y_used), np.nan, y_obs - (H @ X_a).mean(axis=1))
    oe.learn(when, obs_depths, d_b, d_a, HX_f.var(axis=1, ddof=1), sigma)

    n_stab = 0
    L = oe.noise_factor(when)
    if L is not None:
        depth, taper = lake_level - z_vol, float(oe.spec["taper_m"])
        X_a = X_a + draw_noise(L, oe.grid, depth, X_a.shape[1], when, online.seed, taper)
        if oe.spec["stabilize"]:
            V = model.state_cell_volumes(members[0], args)
            V = np.ones(len(z_vol)) if V is None else V
            for m in range(X_a.shape[1]):
                X_a[:, m], c = stabilize(X_a[:, m], V, depth, oe.grid[-1] + taper)
                n_stab += c
    n_rej = int((np.isnan(y_used) & ~np.isnan(y_obs)).sum())
    return X_a, n_rej, n_stab


def _analyse(args, model, members, y_obs, sim_depths, obs_depths, when, online=None, loc=None):
    """Read the members' end-of-window states, update them with the window's observations and
    write them back. `loc` ({"radii": (depths, radii), "logged": bool}) turns localization on.
    Returns the number of members updated."""
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
    if online is None:
        # A vector, so that enkf_update never sees a season-keyed dict (np.ndim of a dict is 0).
        sigma, y_used = resolve_sigma_obs(obs_depths, when, args), y_obs
    else:
        sigma = online.oe.R(obs_depths)
        y_used = _nis_check(online, y_obs, H @ X_f, sigma, when)
    # Keyed on the analysis instant: a run continued in slices draws as one continuous run.
    rng = time_keyed_rng(args.get("rng_seed", 42), "enkf_obs", int(when.timestamp()))
    loc_so = loc_oo = None
    if loc is not None:
        # Rebuilt each analysis: the observed depths and the lake level move.
        state_depths = lake_level - np.asarray(z_vol, dtype=float)
        loc_so = taper(state_depths, obs_depths, *loc["radii"])
        loc_oo = taper(obs_depths, obs_depths, *loc["radii"])
        if not loc["logged"]:
            logger.info(summarize(state_depths, obs_depths, *loc["radii"]))
            loc["logged"] = True
    X_a, diags = enkf_update(X_f, y_used, H, sigma, inflation=args["inflation"], rng=rng,
                             loc_so=loc_so, loc_oo=loc_oo)
    if online is not None:
        X_a, n_rej, n_stab = _online_after(args, model, online, readable, X_f, X_a, y_obs, y_used,
                                           H, sigma, z_vol, lake_level, obs_depths, when)

    def write(col_i):
        col, i = col_i
        try:
            model.write_snapshot_T(i, X_a[:, col], args)
        except Exception as e:
            logger.warning(f"[ensemble{i:02d}] snapshot write failed: {e}")

    with concurrent.futures.ThreadPoolExecutor() as pool:
        pool.map(write, enumerate(readable))
    if diags is not None:
        _write_diagnostics(args, when, diags, y_used, obs_depths, z_vol, lake_level)
    if online is not None:
        online.commit(when, n_rej, n_stab)
    return len(readable)


def run_enkf_loop(args, model):
    if args.get("reset"):
        reset_members(args, model, [args["diag_path"], args["innov_depth_path"], args["kgain_depth_path"]])
    obs = run_observations(args, model)
    online = OnlineRun(args, obs["depth"].unique()) if args.get("online_errors") else None
    if online is None:
        log_sigma_summary(args)
    loc = {"radii": load_radii(args["localization"]), "logged": False} if args.get("localization") else None
    logger.info(f"EnKF inflation={args['inflation']}, localization={'on' if loc else 'off'}")

    def update(window_start, window_end, failed):
        y_obs, sim_depths, obs_depths = window_observations(obs, window_start, window_end, model)
        if y_obs is None:
            return False, "no obs"
        good = [i for i in args["member_ids"] if i not in failed]
        n_updated = _analyse(args, model, good, y_obs, sim_depths, obs_depths, window_end, online, loc) if len(good) >= 2 else 0
        return n_updated > 0, f"n_obs={len(y_obs)}  n_updated={n_updated}"

    return step_windows(args, model, obs, update)

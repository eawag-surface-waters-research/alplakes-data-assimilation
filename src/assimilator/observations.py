"""Observation preprocessing: raw sub-daily profiles -> the series a run assimilates.

    raw (time, depth, value), sub-daily
      │  1. filter_seiche   causal trailing box of one seiche period below THERMO_DEPTH_MIN
      │  2. select_depths   (optional) keep the depths nearest a target grid
      │  3. thin_daily      one profile per day, at a fixed hour (or the first in a window)
      ▼
    assimilated series = the analysis schedule (window_mode "obs": one analysis per obs time)

Rows are SELECTED or averaged, never invented, and every function is a pure DataFrame -> DataFrame
transform: no files, no config lookups. `preprocess` chains them from one config block, and with
`since` reprocesses only the tail of a growing record (with enough look-back that the result is
identical to processing the whole record -- the causal filter makes that possible).

Ported from the feat/da-combi scripts filter_observations.py and generate_assimilation_series.py
(stage 2); the numbers are unchanged. The per-lake parameters are fitted with assimilator.fitting.

Config block (all keys optional; an absent step is skipped):

    {"filter":   {"w_seiche_h": 27.3, "thermo_depth_min": 4.0},
     "schedule": {"hour": 6, "max_offset_h": 1.5}          # or {"hour_window": [0, 6]}
     "depths":   [0.5, 1, 3, ...], "depth_tolerance_m": 0.5}
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------------- filter --

FILTER_DEFAULTS = {
    "W_SEICHE":        24.0,    # h   - trailing window; placeholder for a lake not yet calibrated
    "THERMO_DEPTH_MIN": 4.0,    # m   - above this the observations are passed through untouched
    "DT_FILT":           60,    # min - resolution the filter runs at
    "MIN_SUPPORT":     0.25,    # -   - a smoothed value needs this share of its window to be real samples
    "MAX_AGE_FACTOR":  1.50,    # -   - and those samples' mean age may exceed the ideal (W-1)/2 by at
                                #       most this factor. Guards the window that is filled entirely
                                #       from BEFORE a gap and then presented as "now".
}

# Config-file names (lower case, with units) -> the filter's parameter names.
FILTER_KEYS = {
    "w_seiche_h":       "W_SEICHE",
    "thermo_depth_min": "THERMO_DEPTH_MIN",
    "dt_filt_min":      "DT_FILT",
    "min_support":      "MIN_SUPPORT",
    "max_age_factor":   "MAX_AGE_FACTOR",
}

# A criterion rather than a lake property: below this the record carries no sub-daily
# variability to remove.
MIN_SAMPLES_PER_DAY = 4


def filter_params(block):
    """FILTER_DEFAULTS overlaid with a config "filter" block (config names, see FILTER_KEYS).
    Keys not in FILTER_KEYS (e.g. provenance such as "derived_on") are ignored."""
    params = dict(FILTER_DEFAULTS)
    for key, value in (block or {}).items():
        if key in FILTER_KEYS:
            params[FILTER_KEYS[key]] = value
    return params


def _variable_box(x, widths, support=False):
    """Causal trailing mean whose length varies per sample. Cumulative sums -> O(n).
    `support=True` also returns how the mean was actually formed:
        n   how many real samples were in the window (its LENGTH is `widths`; on a gappy record the
            two are nothing like each other)
        age their mean age in steps. A full window gives (W-1)/2; when the recent half is missing
            the survivors sit further back, so the "mean at time t" is really a mean of the lake
            some time before t.
    """
    xf = np.where(np.isnan(x), 0.0, x)
    ok = (~np.isnan(x)).astype(float)
    idx = np.arange(len(x))
    cs = np.concatenate([[0.0], np.cumsum(xf)])
    cv = np.concatenate([[0.0], np.cumsum(ok)])
    hi = idx + 1
    lo = np.maximum(0, idx - widths + 1)
    n  = cv[hi] - cv[lo]
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.where(n > 0, (cs[hi] - cs[lo]) / n, np.nan)
    if not support:
        return mean
    ci = np.concatenate([[0.0], np.cumsum(ok * idx)])
    with np.errstate(invalid="ignore", divide="ignore"):
        age = np.where(n > 0, idx - (ci[hi] - ci[lo]) / np.where(n > 0, n, 1.0), np.nan)
    return mean, n, age


def filter_seiche(obs, params=None, check_rate=True):
    """Causal low-pass filter of a long-form obs frame (time [tz-aware], depth, value).

    In the thermocline most hour-to-hour variance is a sharp gradient being displaced by
    basin-scale seiches, which a 1D column cannot reproduce. Depths >= THERMO_DEPTH_MIN get a
    trailing box mean of W_SEICHE hours; shallower depths pass through. Values are emitted only at
    instants the instrument sampled (on the DT_FILT grid), and only if the window behind them held
    enough recent samples (MIN_SUPPORT, MAX_AGE_FACTOR).

    `params` uses FILTER_DEFAULTS names (see filter_params for config names). `check_rate=False`
    turns the "this looks like an already-thinned series" error into a warning (for short chunks).
    Returns (filtered_long, report)."""
    p  = dict(FILTER_DEFAULTS) if params is None else {**FILTER_DEFAULTS, **params}
    dt = p["DT_FILT"]
    if p["THERMO_DEPTH_MIN"] is None:
        raise ValueError("THERMO_DEPTH_MIN must be a number")

    span = max((obs["time"].max() - obs["time"].min()).days, 1)
    rate = len(obs) / max(obs["depth"].nunique(), 1) / span
    if rate < MIN_SAMPLES_PER_DAY:
        msg = (f"the input has {rate:.2f} samples per depth per day, below {MIN_SAMPLES_PER_DAY} — "
               f"this looks like a daily or profile series, which carries no sub-daily variability "
               f"for the filter to remove. Filter the raw high-frequency record, not a thinned one")
        if check_rate:
            raise ValueError(msg)
        logger.warning(msg)
    logger.debug(f"  {rate:.0f} samples per depth per day over {span} d")

    # Round to the NEAREST bin, as functions.load_obs does, so a sample lands in the same hour for
    # the filter and the assimilator.
    obs = obs.copy()
    obs["time"] = (obs["time"] + pd.Timedelta(minutes=dt / 2)).dt.floor(f"{dt}min")

    # ffill (max 2 steps) bridges the SAMPLING CADENCE, never a gap: a causal fill, so no
    # look-ahead. `real` records which rows were actually sampled; output is masked back to it.
    grid = (obs.pivot_table(index="time", columns="depth", values="value", aggfunc="mean")
               .sort_index().resample(f"{dt}min").mean())
    real = grid.notna()
    piv = grid.ffill(limit=2)

    depths = np.array(piv.columns, dtype=float)
    if len(depths) == 0:
        raise ValueError("no observation depths in the input")

    zmin  = p["THERMO_DEPTH_MIN"]
    n_win = max(1, int(round(p["W_SEICHE"] * 60 / dt)))
    below = [d for d in piv.columns if float(d) >= zmin]
    if not below:
        raise ValueError(f"no observation depth at or below THERMO_DEPTH_MIN={zmin:g} m — "
                         f"nothing to filter")
    logger.debug(f"  window {p['W_SEICHE']:g} h ({n_win} steps) at {len(below)} depths >= {zmin:g} m; "
                 f"{len(depths) - len(below)} shallower depth(s) passed through")

    out, report = pd.DataFrame(index=piv.index, columns=piv.columns, dtype=float), []
    for d in depths:
        w = n_win if float(d) >= zmin else 1
        widths = np.full(len(piv), w, dtype=int)
        mean, n_used, age = _variable_box(piv[d].values, widths, support=True)

        # Admission, per VALUE: enough samples in the window, and not bunched too far back.
        ideal = max((w - 1) / 2.0, 1e-9)
        thin  = n_used < p["MIN_SUPPORT"] * w
        stale = age > p["MAX_AGE_FACTOR"] * ideal
        drop  = (thin | stale) & np.isfinite(mean)
        mean  = np.where(drop, np.nan, mean)
        out[d] = np.where(real[d].to_numpy(), mean, np.nan)

        raw, flt = piv[d], out[d]
        removed  = (raw - flt).std()
        report.append({"depth": d, "window_h": float(w * dt / 60),
                       "raw_std": float(raw.std()), "filtered_std": float(flt.std()),
                       "removed_std": float(removed),
                       "removed_frac_of_var": float(removed ** 2 / raw.var()) if raw.var() > 0 else np.nan,
                       "withheld_thin": int((thin & np.isfinite(piv[d].values)).sum()),
                       "withheld_stale": int((stale & ~thin & np.isfinite(piv[d].values)).sum())})

    long = (out.stack().rename("value").reset_index()
               .rename(columns={"level_1": "depth"}).dropna(subset=["value"]))
    return long, pd.DataFrame(report)


def filter_lookback(params):
    """How much record before `since` the filter needs so its output from `since` on is identical
    to filtering the whole record: one window, the 2-step cadence bridge, and a day of margin."""
    p = {**FILTER_DEFAULTS, **(params or {})}
    return pd.Timedelta(hours=float(p["W_SEICHE"])) + pd.Timedelta(minutes=2 * p["DT_FILT"]) \
        + pd.Timedelta(days=1)


# -------------------------------------------------------------------------------- depth grid --

def select_depths(obs, depths, tol=0.5, strict=True):
    """Keep only the observation depths nearest each requested one.

    The requested depths are a TARGET GRID, not exact keys: each target takes the nearest available
    depth within `tol` metres. A target with nothing that close raises (strict) or is dropped with a
    warning. Returns the selected rows (depth values unchanged)."""
    have = np.sort(obs["depth"].unique())
    if len(have) == 0:
        return obs
    keep, missing = [], []
    for t in depths:
        i = int(np.argmin(np.abs(have - t)))
        if abs(have[i] - t) <= tol:
            keep.append(have[i])
        else:
            missing.append(t)
    if missing:
        msg = (f"no observation depth within {tol} m of {missing} "
               f"(available {have.min():g}..{have.max():g} m)")
        if strict:
            raise ValueError(msg)
        logger.warning(msg + " — those targets are skipped")
    return obs[obs["depth"].isin(sorted(set(keep)))]


# ---------------------------------------------------------------------------------- thinning --

_CADENCE_EPOCH = pd.Timestamp("1970-01-01", tz="UTC")


def thin_daily(obs, hour=None, hour_window=None, max_offset=1.5, every_days=1):
    """Keep ONE instant per retained day, all depths of it.

    hour         the instant nearest `hour`:00 UTC wins; a day whose winner is more than
                 `max_offset` hours away is skipped (no stand-in from another point of the cycle)
    hour_window  (lo, hi): the FIRST instant in [lo:00, hi:00) wins instead (for a lake whose
                 sampling phase drifts); max_offset does not apply
    every_days   keep every n-th day, counted from 1970-01-01 (so the choice does not depend on
                 where the record starts)

    One winner per day, not per depth, so a profile is never stitched from several instants.
    Rows are selected, never modified."""
    if (hour is None) == (hour_window is None):
        raise ValueError("thin_daily needs exactly one of `hour` or `hour_window`")
    if obs.empty:
        return obs.copy()
    obs = obs.copy()
    obs["_t"] = pd.to_datetime(obs["time"], utc=True, format="ISO8601")
    # Match load_obs's binning (round to the nearest hour) to decide which day/hour a row is in.
    binned = (obs["_t"] + pd.Timedelta(minutes=30)).dt.floor("1h")
    day = binned.dt.floor("D")
    cadence = ((day - _CADENCE_EPOCH).dt.days % every_days) == 0
    if hour_window is not None:
        lo, hi = hour_window
        h = binned.dt.hour
        inside = (h >= lo) & (h < hi)
        if not inside.any():
            return obs.iloc[0:0].drop(columns="_t")
        pick_t = obs.loc[obs["_t"][inside].groupby(day[inside]).idxmin(), "_t"]
        keep = obs["_t"].isin(set(pick_t)) & cadence
    else:
        off = (obs["_t"] - (day + pd.Timedelta(hours=hour))).abs()
        pick_t = obs.loc[off.groupby(day).idxmin(), "_t"]
        keep = obs["_t"].isin(set(pick_t)) \
            & (off <= pd.Timedelta(hours=max_offset)) \
            & cadence
    return obs[keep].drop(columns="_t")


# ------------------------------------------------------------------------------- the pipeline --

def lookback(cfg):
    """Record needed before `since` for preprocess(cfg, since) to match a full reprocess."""
    if cfg.get("filter"):
        return filter_lookback(filter_params(cfg["filter"]))
    return pd.Timedelta(days=1)


def preprocess(raw, cfg, since=None):
    """raw (time, depth, value) -> the assimilated series, per the config block (module docstring).

    With `since` (a tz-aware timestamp at a day boundary), only rows at or after `since` are
    returned, computed from `raw` rows at or after since - lookback(cfg): identical to the
    corresponding rows of a full reprocess, so a host can append each day's tail to a stored
    series. In that mode missing depth-grid targets warn instead of raising."""
    obs = raw[["time", "depth", "value"]].copy()
    obs["time"] = pd.to_datetime(obs["time"], utc=True, format="ISO8601")
    obs = obs.dropna(subset=["time", "depth", "value"])
    if since is not None:
        since = pd.Timestamp(since)
        since = since.tz_localize("UTC") if since.tzinfo is None else since.tz_convert("UTC")
        obs = obs[obs["time"] >= since - lookback(cfg)]
    if obs.empty:
        return obs.reset_index(drop=True)

    if cfg.get("filter"):
        obs, _ = filter_seiche(obs, filter_params(cfg["filter"]), check_rate=since is None)
    if cfg.get("depths"):
        obs = select_depths(obs, cfg["depths"], tol=cfg.get("depth_tolerance_m", 0.5),
                            strict=since is None)
    schedule = cfg.get("schedule")
    if schedule:
        window = schedule.get("hour_window")
        obs = thin_daily(obs,
                         hour=None if window else schedule["hour"],
                         hour_window=tuple(window) if window else None,
                         max_offset=schedule.get("max_offset_h", 1.5),
                         every_days=schedule.get("every_days", 1))
    if since is not None:
        t = pd.to_datetime(obs["time"], utc=True)
        if schedule:
            # the thinning assigns a row to the day of its nearest hour, so cut on the same binning
            t = (t + pd.Timedelta(minutes=30)).dt.floor("1h")
        obs = obs[t >= since]
    return (obs[["time", "depth", "value"]]
            .sort_values(["time", "depth"], kind="stable").reset_index(drop=True))

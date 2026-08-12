"""Causal low-pass filter for lake temperature observations -> temperature_filtered.csv.
In the thermocline most of the hour-to-hour signal is not heat content changing but a sharp
vertical gradient being displaced by basin-scale internal seiches, which a 1D column cannot
reproduce. That is representativeness error, not something ensemble spread should be asked to cover.

WHAT. A causal trailing box mean of one seiche period:

    W(z) = W_SEICHE   for z >= THERMO_DEPTH_MIN,   0 above

W_SEICHE is a physical timescale. notebooks/calibrate_filter.py derives it per lake from the two-layer model.

Constant in depth and time because the seiche is a BASIN mode -- one period, felt wherever there is
a gradient to displace; only its amplitude varies with depth. A constant also needs no observation
to evaluate.
Depths above THERMO_DEPTH_MIN pass through untouched: there the fast signal is solar heating, which
the model should reproduce and gets wrong by ~40%, so filtering it would hide a model defect rather
than remove noise. 4 m is a FIXED, deliberately conservative value -- a diurnal-coherence test put
the real cut at 2.0-3.0 m on all seven configured lakes, so 4 m always excludes more than the
physics asks, and the two errors are not symmetric: too deep costs a few percent of the removable
variance, too shallow hides a known model defect. 
PARAMETERS. Five knobs (see DEFAULTS): W_SEICHE and THERMO_DEPTH_MIN define the window, DT_FILT the
grid it runs on, MIN_SUPPORT and MAX_AGE_FACTOR reject values a gappy window cannot support. Only
W_SEICHE really varies per lake. Resolved in precedence order (see load_params): CLI overrides, a
"filter" block in the run config, filter/<lake>.json, then DEFAULTS. Deleting filter/<lake>.json is
the kill switch.
Usage:
    python notebooks/filter_observations.py args/run_enkf.json --lake upperlugano
    python notebooks/filter_observations.py args/run_enkf.json --lakes all

Output is the full sub-daily filtered record, one row per instant the instrument actually sampled.
Choosing which of those instants get assimilated is thinning, and belongs to local/thin_obs.py:

    python local/thin_obs.py --lake upperlugano --hour 6 \
        --in-file  observations/upperlugano/temperature_filtered.csv \
        --out-file observations/upperlugano/temperature_filtered_h06_1d.csv
"""
import os
import sys
import json
import logging
import argparse

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
from assimilator.functions import ROOT, merge_lake_args   # noqa: E402

logger = logging.getLogger(__name__)

DEFAULTS = {
    "W_SEICHE":        24.0,    # h   - trailing window placeholder for a lake not been calibrated.
    "THERMO_DEPTH_MIN": 4.0,    # m   - above this the observations are passed through untouched
    "DT_FILT":           60,    # min - resolution the filter runs at
    "MIN_SUPPORT":     0.25,    # -   - a smoothed value needs this share of its window to be real samples.
    "MAX_AGE_FACTOR":  1.50,    # -   - and those samples' mean age may exceed the ideal (W-1)/2 by at most this factor.                                #       Guards the window that is filled entirely from BEFORE a gap and then presented as "now".
}

# Constants that define a CRITERION rather than describe a lake. Deliberately not in DEFAULTS: they
# are not per-lake, not configurable and not fittable.
MIN_SAMPLES_PER_DAY = 4     # below this the record carries no sub-daily variability to remove



def raw_obs_path(cfg, root=ROOT):
    """The RAW sub-daily series, always observations/<lake>/temperature.csv.
    Deliberately not the configured obs_file: most lakes assimilate an already-thinned daily series,
    and a filter that removes sub-daily variability has to see the sub-daily record. The order is
    raw -> filter -> thin, never raw -> thin -> filter.
    """
    return os.path.join(root, "observations", cfg["lake"], "temperature.csv")


def filter_params_path(lake, root=ROOT):
    return os.path.join(root, "filter", f"{lake}.json")


def load_params(lake, cfg=None, root=ROOT):
    """(params, source) for one lake: DEFAULTS < filter/<lake>.json < config "filter" block.

    The CLI layer is applied by main() on top of this."""
    cfg    = cfg or {}
    params = dict(DEFAULTS)
    srcs   = []

    path = filter_params_path(lake, root)
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            fitted = json.load(f)
        block = fitted.get("params", {})
        unknown = set(block) - set(DEFAULTS)
        if unknown:
            raise ValueError(f"{os.path.relpath(path, root)}: unknown filter parameter(s) "
                             f"{sorted(unknown)}; expected a subset of {sorted(DEFAULTS)}")
        params.update(block)
        srcs.append(f"filter/{lake}.json (derived {fitted.get('derived_on', '?')})")

    block = cfg.get("filter", {})
    if block:
        unknown = set(block) - set(DEFAULTS)
        if unknown:
            raise ValueError(f'run config "filter" block: unknown parameter(s) {sorted(unknown)}; '
                             f"expected a subset of {sorted(DEFAULTS)}")
        params.update(block)
        srcs.append(f"run config \"filter\" block ({'/'.join(sorted(block))})")

    return params, "; ".join(srcs) if srcs else "built-in defaults (placeholder, not lake-specific)"


def describe(p):
    return (f"W_SEICHE={p['W_SEICHE']:g} h  THERMO_DEPTH_MIN={p['THERMO_DEPTH_MIN']:g} m  "
            f"DT_FILT={p['DT_FILT']:g} min")


def _variable_box(x, widths, support=False):
    """Causal trailing mean whose length varies per sample. Cumulative sums -> O(n).
    `support=True` also returns how the mean was actually formed, which is what tells a trustworthy
    value from a technically-computable one:
        n   how many real samples were in the window (its LENGTH is `widths`; on a gappy record the
            two are nothing like each other)
        age their mean age in steps. A full window gives (W-1)/2; when the recent half is missing
            the survivors sit further back, so the "mean at time t" is really a mean of the lake
            some time before t. Nothing else in the filter can see this.
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


def filter_obs(obs, p=None):
    """Filter a long-form obs frame (time, depth, value, ...).

    Returns (filtered_long, report, resolved_params)."""
    p = dict(DEFAULTS) if p is None else {**DEFAULTS, **p}
    dt = p["DT_FILT"]

    if p["THERMO_DEPTH_MIN"] is None:
        raise ValueError("THERMO_DEPTH_MIN must be a number. It is not derived at filter time — "
                         "run notebooks/calibrate_filter.py for this lake, which derives it, "
                         "checks it and records it in filter/<lake>.json")

    span = max((obs["time"].max() - obs["time"].min()).days, 1)
    rate = len(obs) / max(obs["depth"].nunique(), 1) / span
    if rate < MIN_SAMPLES_PER_DAY:
        raise ValueError(
            f"the input has {rate:.2f} samples per depth per day, below {MIN_SAMPLES_PER_DAY} — "
            f"this looks like a daily or profile series, which carries no sub-daily variability "
            f"for the filter to remove. Point it at the raw high-frequency record "
            f"(observations/<lake>/temperature.csv), not at an already-thinned one")
    logger.info(f"  {rate:.0f} samples per depth per day over {span} d")

    # Round to the NEAREST bin, as functions.load_obs does. resample() alone labels each bin by its
    # left edge, so a 10:45 sample would land in the filter's 10:00 bin but the assimilator's 11:00
    # bin -- a free 30 min of lag on top of the W/2 the causal window already costs.
    obs = obs.copy()
    obs["time"] = (obs["time"] + pd.Timedelta(minutes=dt / 2)).dt.floor(f"{dt}min")

    # ffill, NOT interpolate: interpolate(method="time") needs a sample on BOTH sides of a gap, so
    # filling t from t+1 read one to two hours into the future -- a lookahead in the production path.
    #
    # The ffill exists to bridge the SAMPLING CADENCE, not to invent observations. At DT_FILT=60 a
    # 3-hourly lake leaves two empty grid rows between samples, and without them the window would
    # hold a third of its nominal length and MIN_SUPPORT would reject on cadence rather than on
    # gaps. Being capped at 2 steps is what keeps it honest: a real outage is filled for two hours
    # and then stays NaN, so support collapses and the admission tests fire as intended.
    #
    # `real` records which rows the instrument actually sampled. The window may read the bridged
    # rows; the OUTPUT is masked back to `real`, so a filtered value is only ever emitted at an
    # instant that was observed. Without this the filter hands downstream one value per grid step
    # -- 3x more "observations" than the lake reports -- and those extra rows are assimilated,
    # suppress the sub-block scatter sigma_rep is fitted from, and inflate the observation count.
    grid = (obs.pivot_table(index="time", columns="depth", values="value", aggfunc="mean")
               .sort_index().resample(f"{dt}min").mean())
    real = grid.notna()
    piv = grid.ffill(limit=2)

    depths = np.array(piv.columns, dtype=float)
    if len(depths) == 0:
        raise ValueError("no observation depths in the input")
    logger.info(f"  {len(piv)} timesteps x {len(depths)} depths at {dt} min")

    # The window. One constant, applied at every depth at or below the surface exclusion. Depths
    # above it get width 1, i.e. the value passes through unchanged.
    zmin  = p["THERMO_DEPTH_MIN"]
    n_win = max(1, int(round(p["W_SEICHE"] * 60 / dt)))
    below = [d for d in piv.columns if float(d) >= zmin]
    if not below:
        raise ValueError(f"no observation depth at or below THERMO_DEPTH_MIN={zmin:g} m — "
                         f"nothing to filter")
    logger.info(f"  window {p['W_SEICHE']:g} h ({n_win} steps) at {len(below)} depths >= {zmin:g} m; "
                f"{len(depths) - len(below)} shallower depth(s) passed through")

    out, report = pd.DataFrame(index=piv.index, columns=piv.columns, dtype=float), []
    for d in depths:
        w = n_win if float(d) >= zmin else 1
        widths = np.full(len(piv), w, dtype=int)
        mean, n_used, age = _variable_box(piv[d].values, widths, support=True)

        # Admission, per VALUE rather than per depth. A mean is emitted only if the window behind
        # it actually held enough samples, and those samples were not bunched too far back. Both
        # rules are off at MIN_SUPPORT=0 / MAX_AGE_FACTOR=inf, which reproduces `n > 0`.
        ideal = max((w - 1) / 2.0, 1e-9)
        thin  = n_used < p["MIN_SUPPORT"] * w
        stale = age > p["MAX_AGE_FACTOR"] * ideal
        drop  = (thin | stale) & np.isfinite(mean)
        mean  = np.where(drop, np.nan, mean)
        # Emit only where the instrument sampled — the bridged rows served the window, not the output.
        out[d] = np.where(real[d].to_numpy(), mean, np.nan)

        raw, flt = piv[d], out[d]
        removed  = (raw - flt).std()
        row = {"depth": d, "window_h": float(w * dt / 60),
               "raw_std": float(raw.std()), "filtered_std": float(flt.std()),
               "removed_std": float(removed),
               "removed_frac_of_var": float(removed ** 2 / raw.var()) if raw.var() > 0 else np.nan,
               "withheld_thin": int((thin & np.isfinite(piv[d].values)).sum()),
               "withheld_stale": int((stale & ~thin & np.isfinite(piv[d].values)).sum())}
        report.append(row)

    long = (out.stack().rename("value").reset_index()
               .rename(columns={"level_1": "depth"}).dropna(subset=["value"]))
    return long, pd.DataFrame(report), p


# ------------------------------------------------------------------------------------ per lake

def filter_lake(cfg, overrides=None, in_file=None, out_file=None):
    """Filter one lake end to end. Returns the resolved parameters actually used."""
    lake = cfg["lake"]

    p, src = load_params(lake, cfg)
    if overrides:
        p.update(overrides)
        src += f"; {'/'.join(sorted(overrides))} overridden on the command line"
    logger.info(f"{lake}: params <- {src}")
    logger.info(f"  {describe(p)}")
    if not os.path.isfile(filter_params_path(lake)):
        logger.warning(f"  no filter/{lake}.json — using the placeholder "
                       f"W_SEICHE={DEFAULTS['W_SEICHE']:g} h. Run notebooks/calibrate_filter.py "
                       f"to derive this lake's own seiche period.")

    in_path = in_file or raw_obs_path(cfg)
    if not os.path.isfile(in_path):
        raise FileNotFoundError(f"{lake}: no observations at {os.path.relpath(in_path, ROOT)}")
    out_path = out_file or os.path.join(os.path.dirname(in_path), "temperature_filtered.csv")
    logger.info(f"  {os.path.relpath(in_path, ROOT)} -> {os.path.relpath(out_path, ROOT)}")

    obs = pd.read_csv(in_path)
    obs["time"] = pd.to_datetime(obs["time"], utc=True, format="ISO8601")
    logger.info(f"  read {len(obs):,} rows, {obs['depth'].nunique()} depths")

    filtered, report, resolved = filter_obs(obs, p)

    # Carry the non-value columns (latitude/longitude/weight/station) through unchanged, matched on
    # depth, so the output is drop-in for assimilator.functions.load_obs.
    extra = [c for c in obs.columns if c not in ("time", "depth", "value")]
    if extra:
        per_depth = obs.groupby("depth")[extra].first().reset_index()
        filtered  = filtered.merge(per_depth, on="depth", how="left")
        filtered  = filtered[["time", "depth"] + extra + ["value"]]

    filtered.to_csv(out_path, index=False)
    logger.info(f"  wrote {len(filtered):,} rows -> {os.path.relpath(out_path, ROOT)}")

    pd.set_option("display.width", 160)
    print(f"\n{lake} — what was removed, per depth "
          f"(removed_frac_of_var = share of raw variance filtered out):")
    print(report.to_string(index=False, float_format=lambda v: f"{v:9.3f}"))

    return resolved


def main():
    ap = argparse.ArgumentParser(description="Seiche-period low-pass filter for temperature observations")
    ap.add_argument("arg_file")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--lake", default=None, help="Lake to filter from the config's \"lakes\" block")
    grp.add_argument("--lakes", default=None,
                     help="Comma-separated lakes to filter in sequence, or 'all' for every block in "
                          "the config's \"lakes\". Lakes with no observation file are skipped; a "
                          "genuine failure is logged and the batch continues, exiting non-zero.")
    ap.add_argument("--in-file", default=None, help="override the input obs CSV (single lake only)")
    ap.add_argument("--out-file", default=None, help="override the output (single lake only)")
    # Per-lake knobs. Prefer filter/<lake>.json (notebooks/calibrate_filter.py) — these are for
    # one-off experiments and are refused in batch mode, where one global set cannot be right.
    ap.add_argument("--w-seiche", type=float, default=None,
                    help=f"trailing window in h, i.e. the fundamental seiche period "
                         f"(default {DEFAULTS['W_SEICHE']:g})")
    ap.add_argument("--thermo-depth-min", type=float, default=None,
                    help=f"surface exclusion depth in m (default {DEFAULTS['THERMO_DEPTH_MIN']:g})")
    cli = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")

    overrides = {k: v for k, v in (("W_SEICHE", cli.w_seiche),
                                   ("THERMO_DEPTH_MIN", cli.thermo_depth_min)) if v is not None}

    arg_file = cli.arg_file if os.path.isfile(cli.arg_file) else os.path.join(ROOT, cli.arg_file)
    if not os.path.isfile(arg_file):
        raise ValueError(f"Args file not found: {cli.arg_file}")
    with open(arg_file) as f:
        raw = json.load(f)

    if not cli.lakes:
        lakes = [cli.lake]
    elif cli.lakes.strip() == "all":
        if not raw.get("lakes"):
            raise ValueError("--lakes all requires a \"lakes\" block in the config")
        lakes = list(raw["lakes"])
    else:
        lakes = [s.strip() for s in cli.lakes.split(",") if s.strip()]

    batch = len(lakes) > 1
    if batch:
        if overrides:
            raise ValueError("the per-lake --w-seiche/--thermo-depth-min flags apply one value to "
                             "every lake and are refused in batch mode; put them in "
                             "filter/<lake>.json instead")
        if cli.in_file or cli.out_file:
            raise ValueError("--in-file/--out-file name a single file and are refused in batch mode")

    failed, skipped = [], []
    for lake in lakes:
        try:
            filter_lake(merge_lake_args(raw, lake=lake), overrides=overrides,
                        in_file=cli.in_file, out_file=cli.out_file)
        except FileNotFoundError as exc:
            # the normal case for a config listing more lakes than you have buoys for
            if not batch:
                raise
            logger.info(f"{lake}: {exc} — skipped")
            skipped.append(lake)
        except Exception:                       # noqa: BLE001 - isolate each lake in a batch
            if not batch:
                raise
            logger.exception(f"lake '{lake}' FAILED - continuing with the rest of the batch")
            failed.append(lake)

    if batch:
        done = len(lakes) - len(failed) - len(skipped)
        logger.info(f"=== batch complete: {done}/{len(lakes)} filtered"
                    + (f"; {len(skipped)} without observations: {', '.join(skipped)}" if skipped else "")
                    + (f"; FAILED: {', '.join(failed)}" if failed else "") + " ===")
        if failed:
            sys.exit(1)


if __name__ == "__main__":
    main()

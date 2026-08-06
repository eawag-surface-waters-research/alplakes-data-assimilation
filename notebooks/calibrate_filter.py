"""Derive the observation filter's per-lake parameters -> filter/<lake>.json.

NOTHING HERE IS FITTED. The filter applies a trailing box of one internal-seiche period, and a box
of width W nulls W, W/2, W/3, ... -- so the window that kills the fundamental mode and all its
harmonics at the least lag IS the fundamental period. That is a physical quantity read off the lake.

THE MODEL. Merian's formula for the fundamental (V1H1) mode of a two-layer basin:

    T = 2L / sqrt(g' * h_eff)     h_eff = h1*h2 / (h1 + h2)     g' = g (rho_2 - rho_1) / rho_2

    L     basin length, a published value in the lake's config ("length_km").
    h1    epilimnion = the observed thermocline depth, 6-10 m on these lakes.
    h2    hypolimnion = mean lake depth minus h1, likewise from the config ("mean_depth").
          Mean, not maximum: the seiche runs over the whole basin, most of which is shallower
          than the deepest point. See lake_geometry for why neither is computed any more.
    rho   at the mean epilimnion temperature and at the deepest observed depth (see
          monthly_two_layer for which way the latter biases the answer).

EVALUATED AT PEAK STRATIFICATION, the most consequential choice here. So the model is read off the months within PEAK_STRAT_FRAC of the annual maximum density contrast --
Jul-Aug on every configured lake, from the data, plus June on the two shallow ones. That is when the
thermocline is sharpest and seiche displacement injects the most variance into a fixed-depth sensor,
and it is the shortest period of the year, so the least lag that does the job. The cost is that a
longer autumn seiche is attenuated rather than nulled. Only W_SEICHE is derived. THERMO_DEPTH_MIN (conservative estimate
from analyses) is written out as the fixed 4 m it always is, so the
json records the complete parameter set the filter will run with.

Usage:
    python notebooks/calibrate_filter.py args/run_enkf.json --lake upperlugano
    python notebooks/calibrate_filter.py args/run_enkf.json --lakes all
    python notebooks/calibrate_filter.py args/run_enkf.json --lake geneva --dry-run

Each lake's config block must carry:  "length_km": 72.3,  "mean_depth": 152.7
"""
import os
import sys
import json
import logging
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.join(_ROOT, "notebooks"))

from assimilator.functions import ROOT, merge_lake_args                       # noqa: E402
import filter_observations as F                                               # noqa: E402

logger = logging.getLogger(__name__)

G = 9.81            # m/s2

PEAK_STRAT_FRAC = 0.8    # share of the annual maximum density contrast a month needs to count as
                         # peak stratification. 0.9 would take August alone, i.e. one month of one
                         # year on the single-season lakes.
MIN_MONTH_DAYS  = 10     # days of record a calendar month needs to be read at all

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def water_density(t):
    """Freshwater density, degC -> kg/m3: the standard quadratic about the 4 degC maximum.

    Good to ~0.1 kg/m3 over 0-25 degC, ample here -- g' is a difference of order 1 kg/m3."""
    return 1000.0 * (1.0 - 6.63e-6 * (np.asarray(t, dtype=float) - 4.0) ** 2)

def lake_geometry(cfg):
    """(basin length in m, mean depth in m) for one lake, both from its config block.
    Stated, not computed."""
    missing = [k for k in ("length_km", "mean_depth") if cfg.get(k) is None]
    if missing:
        raise ValueError(f"{cfg['lake']}: {' and '.join(missing)} missing from this lake's config "
                         f"block. Both are published values, e.g. "
                         f'"length_km": 72.3, "mean_depth": 152.7')
    return float(cfg["length_km"]) * 1e3, float(cfg["mean_depth"])


def window_cost(piv, w_h, dt_min=60):
    """Per depth, what a W-hour trailing box buys and what it costs, both in degC.
        removed  std(raw - smoothed): the high-frequency scatter taken out. What the filter is for.
        lag      how far the lake moves over W/2. A trailing box centres half a window back, so the
                 value handed to the analysis at t is really the truth at t - W/2. Measured on a
                 24 h CENTRED mean so the scatter being removed does not count itself as movement.
        says whether it is worth running...
    """
    n24  = max(1, int(round(24 * 60 / dt_min)))
    n_w  = max(1, int(round(w_h * 60 / dt_min)))
    half = max(1, int(round(w_h / 2 * 60 / dt_min)))

    smoothed = piv.rolling(n_w, min_periods=max(1, n_w // 4)).mean()      # causal, as the filter is
    removed  = (piv - smoothed).std()

    truth = piv.rolling(n24, center=True, min_periods=n24 // 2).mean()
    lag   = (truth - truth.shift(half)).abs().mean()
    return lag, removed

def hourly_grid(obs, dt_min=60):
    """The (time x depth) grid the filter runs on: same binning, same causal fill.
    Identical to filter_obs's own preparation, so the thermocline measured here is the one the
    filter sees."""
    obs = obs.copy()
    obs["time"] = (obs["time"] + pd.Timedelta(minutes=dt_min / 2)).dt.floor(f"{dt_min}min")
    piv = (obs.pivot_table(index="time", columns="depth", values="value", aggfunc="mean")
              .sort_index().resample(f"{dt_min}min").mean().ffill(limit=2))
    if piv.shape[1] < 2:
        raise ValueError(f"only {piv.shape[1]} observation depth(s) — need >= 2 to locate a "
                         f"thermocline, and so to derive the seiche period")
    return piv, np.array(piv.columns, dtype=float)


def _grad_raw(piv, depths):
    """|dT/dz| by plain central difference over the sorted depth grid."""
    T = piv.values
    G = np.full_like(T, np.nan)
    for i in range(len(depths)):
        lo = i - 1 if i > 0 else None
        hi = i + 1 if i + 1 < len(depths) else None
        if lo is not None and hi is not None:
            G[:, i] = np.abs(T[:, hi] - T[:, lo]) / (depths[hi] - depths[lo])
        elif hi is not None:
            G[:, i] = np.abs(T[:, hi] - T[:, i]) / (depths[hi] - depths[i])
        elif lo is not None:
            G[:, i] = np.abs(T[:, i] - T[:, lo]) / (depths[i] - depths[lo])
    return pd.DataFrame(G, index=piv.index, columns=piv.columns)


def monthly_two_layer(piv, depths, L, d_mean, dt_min=60):
    """The two-layer reduction of the observed column, per calendar month (years pooled).
    From each month's mean profile: h1 = median depth of peak |dT/dz|, T_epi = mean above it,
    T_hypo = the value at the DEEPEST OBSERVED DEPTH. Not an average below h1, which the
    metalimnion just under the thermocline would drag warm -- that is a transition, not a layer. 
    T_hypo is still too WARM wherever the chain stops well above the bed, so g' is under-estimated 
    and the period over-estimated, and the filter smooths a little longer than the physics asks. 
    Recorded rather than corrected: the fix needs a hypolimnion temperature the observations do not contain.
    """
    grad = _grad_raw(piv, depths)
    out  = {}
    for m in range(1, 13):
        sel = piv.index.month == m
        if sel.sum() * dt_min / 60 / 24 < MIN_MONTH_DAYS:
            continue
        z_tc = float(np.nanmedian(grad[sel].idxmax(axis=1).values))
        if not np.isfinite(z_tc):
            continue
        profile = piv[sel].mean()
        epi = profile[[d for d in profile.index if float(d) < z_tc]].dropna()
        deep = profile.dropna()
        if not len(epi) or not len(deep):
            continue
        t_epi, t_hypo = float(epi.mean()), float(deep.iloc[-1])
        d_rho = float(water_density(t_hypo) - water_density(t_epi))
        if d_rho <= 0:
            continue                    # mixed or inverse: no internal seiche this month
        try:
            period, terms = two_layer_period(L, z_tc, d_mean - z_tc, t_epi, t_hypo)
        except ValueError:
            continue
        out[m] = {"epilimnion_h1_m": z_tc, "T_epilimnion_degC": t_epi,
                  "T_hypolimnion_degC": t_hypo, "deepest_obs_m": float(deep.index[-1]),
                  "delta_rho": d_rho, "period_h": period, **terms}
    if not out:
        raise ValueError("no month of the record shows two-layer stratification — there is no "
                         "internal seiche to null, so no window can be derived")
    return out


def peak_stratification_months(monthly):
    """The months within PEAK_STRAT_FRAC of the annual maximum layer density contrast.
    Not a calendar rule: it resolves to Jul-Aug on every configured lake, and Jun-Aug on the two
    shallow ones (greifensee, murten) that stratify a month earlier -- the sort of thing a
    hardcoded summer would have got wrong.
    """
    peak = max(r["delta_rho"] for r in monthly.values())
    return [m for m in sorted(monthly) if monthly[m]["delta_rho"] >= PEAK_STRAT_FRAC * peak]


def two_layer_period(L, h1, h2, t_epi, t_hypo):
    """Fundamental internal-seiche period in hours, and the terms behind it."""
    rho1, rho2 = float(water_density(t_epi)), float(water_density(t_hypo))
    g_prime = G * (rho2 - rho1) / rho2
    if g_prime <= 0:
        raise ValueError(f"the epilimnion ({t_epi:.2f} degC) is not lighter than the hypolimnion "
                         f"({t_hypo:.2f} degC) over the stratified season — no two-layer "
                         f"stratification, so no internal seiche to null")
    if h2 <= 0:
        raise ValueError(f"the thermocline at {h1:g} m is at or below the mean depth "
                         f"({h1 + h2:g} m) — the two-layer reduction does not apply")
    h_eff = h1 * h2 / (h1 + h2)
    period_h = 2.0 * L / np.sqrt(g_prime * h_eff) / 3600.0
    return float(period_h), {"rho_epilimnion": round(rho1, 3), "rho_hypolimnion": round(rho2, 3),
                             "g_prime_m_s2": round(g_prime, 6), "h_eff_m": round(h_eff, 2)}
    

def derive_lake(cfg, dry_run=False, root=ROOT):
    """Derive one lake's W_SEICHE from the two-layer model and write filter/<lake>.json."""
    lake = cfg["lake"]
    logger.info(f"=== {lake} ===")

    path = F.raw_obs_path(cfg, root)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{lake}: no observations at {os.path.relpath(path, root)}")
    obs = pd.read_csv(path, usecols=["time", "depth", "value"])
    obs["time"] = pd.to_datetime(obs["time"], utc=True, format="ISO8601")
    logger.info(f"  {len(obs):,} obs rows, {obs['depth'].nunique()} depths")

    base, src = F.load_params(lake, cfg, root)
    logger.info(f"  starting parameters <- {src}")
    dt = base["DT_FILT"]

    piv, depths = hourly_grid(obs, dt)

    # --- the two-layer model, month by month, read at peak stratification
    L, d_mean = lake_geometry(cfg)
    monthly = monthly_two_layer(piv, depths, L, d_mean, dt)
    peak    = peak_stratification_months(monthly)

    def med(key):
        return float(np.median([monthly[m][key] for m in peak]))

    h1 = med("epilimnion_h1_m")
    h2 = d_mean - h1
    t_epi, t_hypo = med("T_epilimnion_degC"), med("T_hypolimnion_degC")
    period_h, terms = two_layer_period(L, h1, h2, t_epi, t_hypo)

    logger.info(f"  basin {L / 1e3:.1f} km long, {d_mean:.1f} m mean depth (config)")
    logger.info("  period by month: "
                + "  ".join(f"{MONTHS[m - 1]} {monthly[m]['period_h']:.0f}" for m in sorted(monthly)))
    logger.info(f"  peak stratification: {', '.join(MONTHS[m - 1] for m in peak)} "
                f"(>= {PEAK_STRAT_FRAC:g} x max density contrast)")
    logger.info(f"  h1 {h1:.1f} m ({t_epi:.2f} degC) / h2 {h2:.1f} m ({t_hypo:.2f} degC at "
                f"{med('deepest_obs_m'):g} m) -> h_eff {terms['h_eff_m']:.2f} m, "
                f"g' {terms['g_prime_m_s2']:.2e} m/s2")
    logger.info(f"  W_SEICHE = {period_h:.1f} h  (fundamental two-layer period)")

    params = {"W_SEICHE": round(period_h, 1),
              "THERMO_DEPTH_MIN": float(base["THERMO_DEPTH_MIN"])}

    # --- what the window costs. The window is pure physics and is never scored, so this is the
    #     only thing that says whether it is worth running on this lake.
    zmin = params["THERMO_DEPTH_MIN"]
    lag, removed = window_cost(piv, period_h, dt)
    keep = [d for d in lag.index if float(d) >= zmin and np.isfinite(removed.get(d, np.nan))
            and removed.get(d, 0) > 0]
    lag, removed = lag[keep], removed[keep]
    ratio = (lag / removed).replace([np.inf, -np.inf], np.nan)

    # Read the trade where the filter WORKS HARDEST -- the depth it removes most from. The ratio is
    # near-flat with depth anyway (geneva 0.61-0.83, upperlugano 0.22-0.36: it is a property of the
    # lake, not of the depth), but picking it by the ratio alone lands on some quiet deep channel
    # where both numbers are hundredths and the verdict is decided by noise.
    z_work  = float(removed.idxmax())
    r_work  = float(ratio[z_work])
    r_worst = float(np.nanmax(ratio.values))
    logger.info(f"  window cost at {z_work:g} m, where it removes most: {removed[z_work]:.2f} degC "
                f"of scatter for {lag[z_work]:.2f} of back-dating -> lag/removed {r_work:.2f} "
                f"(worst over depths {r_worst:.2f})")
    if r_work > 1.0:
        logger.warning(
            f"  a {period_h:.0f} h window injects MORE lag than the scatter it removes at "
            f"{z_work:g} m ({lag[z_work]:.2f} vs {removed[z_work]:.2f} degC). The analysis can "
            f"average out random scatter and cannot average out a systematic offset, so this is a "
            f"bad trade. Consider not filtering {lake}.")

    out = {
        "lake": lake,
        "derived_on": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "derived_by": "notebooks/calibrate_filter.py (fundamental two-layer seiche period)",
        "params": params,
        "two_layer": {
            "peak_stratification_months": [MONTHS[m - 1] for m in peak],
            "basin_length_km": round(L / 1e3, 1),
            "mean_depth_m": round(d_mean, 1),
            "epilimnion_h1_m": round(h1, 2),
            "hypolimnion_h2_m": round(h2, 2),
            "T_epilimnion_degC": round(t_epi, 3),
            "T_hypolimnion_degC": round(t_hypo, 3),
            "deepest_obs_m": med("deepest_obs_m"),
            "period_h": round(period_h, 3),
            **terms,
        },
        "diagnostics": {
            "n_depths": int(len(depths)),
            # The seasonal swing behind the peak-stratification choice: an order of magnitude on
            # every lake, so which months are read is not a detail. See the module docstring.
            "period_by_month_h": {MONTHS[m - 1]: round(monthly[m]["period_h"], 1)
                                  for m in sorted(monthly)},
            "thermocline_by_month_m": {MONTHS[m - 1]: round(monthly[m]["epilimnion_h1_m"], 1)
                                       for m in sorted(monthly)},
            # What the window costs against what it buys, at the depth where the trade is
            # worst. Above 1 the filter injects more bias than the scatter it removes.
            "cost_read_at_m": z_work,
            "removed_degC": round(float(removed[z_work]), 3),
            "lag_degC": round(float(lag[z_work]), 3),
            "lag_over_removed": round(r_work, 3),
            "lag_over_removed_worst_depth": round(r_worst, 3),
        },
    }

    dest = F.filter_params_path(lake, root)
    if dry_run:
        logger.info(f"  --dry-run: would write {os.path.relpath(dest, root)}")
        print(json.dumps(out, indent=2))
        return out
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    logger.info(f"  wrote {os.path.relpath(dest, root)}")
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Derive the filter's seiche window from the two-layer model")
    ap.add_argument("arg_file", help="Path to a run config JSON (e.g. args/run_enkf.json)")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--lake", default=None, help="Lake to derive from the config's \"lakes\" block")
    grp.add_argument("--lakes", default=None,
                     help="Comma-separated lakes to derive in sequence, or 'all' for every block in "
                          "the config's \"lakes\". Lakes without observations are skipped; a "
                          "genuine failure is logged and the batch continues.")
    ap.add_argument("--dry-run", action="store_true", help="print the json instead of writing it")
    cli = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")

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

    batch, failed, skipped, done = len(lakes) > 1, [], [], []
    for lake in lakes:
        try:
            out = derive_lake(merge_lake_args(raw, lake=lake), dry_run=cli.dry_run)
            done.append((lake, out["params"]["W_SEICHE"], out["two_layer"], out["diagnostics"]))
        except FileNotFoundError as exc:
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
        print(f"\n{'lake':<14}{'L km':>7}{'D_mean':>8}{'h1 m':>7}{'h_eff':>7}{'dT K':>7}"
              f"{'W_SEICHE h':>12}{'at m':>7}{'removed':>9}{'lag':>7}{'lag/rem':>9}   ")
        for lake, w, t, g in done:
            flag = "!" if g["lag_over_removed"] > 1.0 else ""
            print(f"{lake:<14}{t['basin_length_km']:>7.1f}{t['mean_depth_m']:>8.1f}"
                  f"{t['epilimnion_h1_m']:>7.1f}{t['h_eff_m']:>7.2f}"
                  f"{t['T_epilimnion_degC'] - t['T_hypolimnion_degC']:>7.2f}{w:>12.1f}"
                  f"{g['cost_read_at_m']:>7.0f}{g['removed_degC']:>9.2f}{g['lag_degC']:>7.2f}"
                  f"{g['lag_over_removed']:>9.2f}   {flag}")
        print("   removed/lag read at the depth the filter removes most from; ! = lag/rem > 1, "
              "i.e. more bias injected than scatter removed")
        logger.info(f"=== batch complete: {len(done)}/{len(lakes)} derived"
                    + (f"; {len(skipped)} skipped: {', '.join(skipped)}" if skipped else "")
                    + (f"; FAILED: {', '.join(failed)}" if failed else "") + " ===")
        if failed:
            sys.exit(1)

if __name__ == "__main__":
    main()

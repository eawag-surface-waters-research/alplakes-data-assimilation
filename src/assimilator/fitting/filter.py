"""Derive the seiche filter's window W_SEICHE for one lake (ported from feat/da-combi
notebooks/calibrate_filter.py; the numbers are unchanged).

THE MODEL. Merian's formula for the fundamental (V1H1) mode of a two-layer basin:

    T = 2L / sqrt(g' * h_eff)     h_eff = h1*h2 / (h1 + h2)     g' = g (rho_2 - rho_1) / rho_2

    L     basin length (a published value)
    h1    epilimnion = the observed thermocline depth
    h2    mean lake depth minus h1 (mean, not maximum: the seiche runs over the whole basin)
    rho   at the mean epilimnion temperature and at the deepest observed depth

EVALUATED AT PEAK STRATIFICATION: the months within PEAK_STRAT_FRAC of the annual maximum density
contrast. That is when the thermocline is sharpest and seiche displacement injects the most
variance into a fixed-depth sensor, and it is the shortest period of the year, so the least lag
that does the job. THERMO_DEPTH_MIN is not derived: it is a fixed, conservative 4 m.
"""
import logging
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from ..observations import FILTER_DEFAULTS

logger = logging.getLogger(__name__)

G = 9.81                 # m/s2
PEAK_STRAT_FRAC = 0.8    # share of the annual maximum density contrast a month needs to count
MIN_MONTH_DAYS  = 10     # days of record a calendar month needs to be read at all
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def water_density(t):
    """Freshwater density, degC -> kg/m3: the standard quadratic about the 4 degC maximum."""
    return 1000.0 * (1.0 - 6.63e-6 * (np.asarray(t, dtype=float) - 4.0) ** 2)


def window_cost(piv, w_h, dt_min=60):
    """Per depth, what a W-hour trailing box buys and what it costs, both in degC.
        removed  std(raw - smoothed): the high-frequency scatter taken out
        lag      how far the lake moves over W/2 (a trailing box centres half a window back),
                 measured on a 24 h CENTRED mean so the scatter does not count itself as movement
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
    """The (time x depth) grid the filter runs on: same binning, same causal fill."""
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
    Gr = np.full_like(T, np.nan)
    for i in range(len(depths)):
        lo = i - 1 if i > 0 else None
        hi = i + 1 if i + 1 < len(depths) else None
        if lo is not None and hi is not None:
            Gr[:, i] = np.abs(T[:, hi] - T[:, lo]) / (depths[hi] - depths[lo])
        elif hi is not None:
            Gr[:, i] = np.abs(T[:, hi] - T[:, i]) / (depths[hi] - depths[i])
        elif lo is not None:
            Gr[:, i] = np.abs(T[:, i] - T[:, lo]) / (depths[i] - depths[lo])
    return pd.DataFrame(Gr, index=piv.index, columns=piv.columns)


def two_layer_period(L, h1, h2, t_epi, t_hypo):
    """Fundamental internal-seiche period in hours, and the terms behind it."""
    rho1, rho2 = float(water_density(t_epi)), float(water_density(t_hypo))
    g_prime = G * (rho2 - rho1) / rho2
    if g_prime <= 0:
        raise ValueError(f"the epilimnion ({t_epi:.2f} degC) is not lighter than the hypolimnion "
                         f"({t_hypo:.2f} degC) — no two-layer stratification, no internal seiche")
    if h2 <= 0:
        raise ValueError(f"the thermocline at {h1:g} m is at or below the mean depth "
                         f"({h1 + h2:g} m) — the two-layer reduction does not apply")
    h_eff = h1 * h2 / (h1 + h2)
    period_h = 2.0 * L / np.sqrt(g_prime * h_eff) / 3600.0
    return float(period_h), {"rho_epilimnion": round(rho1, 3), "rho_hypolimnion": round(rho2, 3),
                             "g_prime_m_s2": round(g_prime, 6), "h_eff_m": round(h_eff, 2)}


def monthly_two_layer(piv, depths, L, d_mean, dt_min=60):
    """The two-layer reduction of the observed column, per calendar month (years pooled).
    h1 = median depth of peak |dT/dz|, T_epi = mean above it, T_hypo = the value at the DEEPEST
    OBSERVED DEPTH (still too warm where the chain stops well above the bed, so the period is
    slightly over-estimated -- recorded, not corrected)."""
    grad = _grad_raw(piv, depths)
    out  = {}
    for m in range(1, 13):
        sel = piv.index.month == m
        if sel.sum() * dt_min / 60 / 24 < MIN_MONTH_DAYS:
            continue
        g = grad[sel].dropna(how="all")      # an all-NaN hour has no thermocline to locate
        if g.empty:
            continue
        z_tc = float(np.nanmedian(g.idxmax(axis=1).values))
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
    """The months within PEAK_STRAT_FRAC of the annual maximum layer density contrast."""
    peak = max(r["delta_rho"] for r in monthly.values())
    return [m for m in sorted(monthly) if monthly[m]["delta_rho"] >= PEAK_STRAT_FRAC * peak]


def derive_filter(obs, length_km, mean_depth, thermo_depth_min=None, dt_min=None):
    """Derive W_SEICHE from the RAW sub-daily record `obs` (time, depth, value) and the basin
    geometry. Returns the config "filter" block plus provenance:

        {"w_seiche_h": .., "thermo_depth_min": .., "derived_on": "YYYY-MM-DD",
         "two_layer": {...}, "diagnostics": {...}}

    A lag/removed ratio above 1 at the working depth means the window injects more lag than the
    scatter it removes -- consider not filtering that lake (logged as a warning)."""
    zmin = float(FILTER_DEFAULTS["THERMO_DEPTH_MIN"] if thermo_depth_min is None else thermo_depth_min)
    dt   = FILTER_DEFAULTS["DT_FILT"] if dt_min is None else dt_min
    obs  = obs[["time", "depth", "value"]].copy()
    obs["time"] = pd.to_datetime(obs["time"], utc=True, format="ISO8601")

    piv, depths = hourly_grid(obs, dt)
    L, d_mean = float(length_km) * 1e3, float(mean_depth)
    monthly = monthly_two_layer(piv, depths, L, d_mean, dt)
    peak    = peak_stratification_months(monthly)

    def med(key):
        return float(np.median([monthly[m][key] for m in peak]))

    h1 = med("epilimnion_h1_m")
    h2 = d_mean - h1
    t_epi, t_hypo = med("T_epilimnion_degC"), med("T_hypolimnion_degC")
    period_h, terms = two_layer_period(L, h1, h2, t_epi, t_hypo)

    lag, removed = window_cost(piv, period_h, dt)
    keep = [d for d in lag.index if float(d) >= zmin and np.isfinite(removed.get(d, np.nan))
            and removed.get(d, 0) > 0]
    lag, removed = lag[keep], removed[keep]
    ratio = (lag / removed).replace([np.inf, -np.inf], np.nan)
    # Read the trade where the filter works hardest (largest removed scatter).
    z_work  = float(removed.idxmax())
    r_work  = float(ratio[z_work])
    r_worst = float(np.nanmax(ratio.values))
    logger.info(f"W_SEICHE = {period_h:.1f} h (peak stratification "
                f"{', '.join(MONTHS[m - 1] for m in peak)}); at {z_work:g} m it removes "
                f"{removed[z_work]:.2f} degC for {lag[z_work]:.2f} degC of lag ({r_work:.2f})")
    if r_work > 1.0:
        logger.warning(f"a {period_h:.0f} h window injects MORE lag than the scatter it removes at "
                       f"{z_work:g} m — consider not filtering this lake")

    return {
        "w_seiche_h": round(period_h, 1),
        "thermo_depth_min": zmin,
        "derived_on": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "derived_by": "assimilator.fitting.filter (fundamental two-layer seiche period)",
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
            "period_by_month_h": {MONTHS[m - 1]: round(monthly[m]["period_h"], 1)
                                  for m in sorted(monthly)},
            "thermocline_by_month_m": {MONTHS[m - 1]: round(monthly[m]["epilimnion_h1_m"], 1)
                                       for m in sorted(monthly)},
            "cost_read_at_m": z_work,
            "removed_degC": round(float(removed[z_work]), 3),
            "lag_degC": round(float(lag[z_work]), 3),
            "lag_over_removed": round(r_work, 3),
            "lag_over_removed_worst_depth": round(r_worst, 3),
        },
    }

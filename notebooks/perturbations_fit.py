"""Fit the AR(1) forcing perturbation -> perturbations/<lake>.json.

The calibration has two parts, fitted from different data and kept strictly separate:

  AMPLITUDE    the perturbation's stationary std, sigma/sqrt(1-phi^2). Fitted from the ICON
               residual (ICON lake mean - control Forcing.dat). Needs the ICON API / EAWAG
               VPN, so it is fitted once per lake and committed.

  PERSISTENCE  how long the perturbation stays correlated with itself, phi = exp(-1/tau).
               This CANNOT be measured: it needs the ACF of (forecast - truth) and there are
               no archived ICON forecasts. So it is BRACKETED by two proxies, each of which is
               contaminated in a known direction:

    lower  the ACF of the ICON residual. That residual is a difference between two datasets at
           the same instant, so much of it is grid resolution and interpolation mismatch, which
           flickers hour to hour like white noise. Adding white noise can only SHORTEN a
           measured decorrelation time: for x = s + n with n white and independent,
           ACF_x(k) = ACF_s(k) * var(s)/(var(s)+var(n)) scales every lag k >= 1 by the same
           constant < 1, so the 1/e crossing moves earlier. A floor.

    upper  the ACF of the station signal itself. This measures how long the WEATHER persists,
           not how long the ERROR does. A forecast already captures the slow synoptic structure
           and fails on the fast small-scale part, so the error is enriched in fast components
           relative to the full signal and therefore decorrelates sooner. A ceiling.

           Measured on upperlugano GLOB:   1.39 h  <  tau_err  <  7.17 h.

  Both arguments are physical, not proofs: the floor assumes the spatial mismatch is white, the
  ceiling assumes the forecast captures the slow part. Neither is verified here, and the honest
  number still needs the ACF of (forecast - station obs). Treat a run on either end as one side
  of a range, not as a measurement.

    envelope   per channel, whichever end has the LONGER tau. Measured, the bracket does not
               always hold: on maggiore V, geneva V and hallwil U the "lower" bound e-folds
               LATER than the upper one, which falsifies the floor argument there (that
               residual carries a slowly drifting bias, not white grid noise). Taking the
               upper end on those channels would shorten persistence — the one direction that
               costs ensemble spread, and under-spread is the classic EnKF failure. The
               envelope errs the safe way, and is the default choice for a production run.

  Bracket width varies enormously by lake and is worth reading before trusting either end. The
  plateau lakes exposed to the Bise (murten 2.0 -> 24.5 h on U, greifensee, geneva) persist for
  a day or more, so their ceiling is nearly vacuous; the lakes south of the Alps (upperlugano,
  maggiore) sit at 2-5 h and are much better constrained. The report prints it per channel.

HOW THE TWO BOUNDS ARE STORED, and why it is done this way. Both live in the JSON under
`persistence.bounds`, and the top-level `variables` the runtime reads is a VERBATIM COPY of the
selected one. Switching bound therefore does no arithmetic at all -- it is exactly reversible,
idempotent, and works offline in both directions once a lake has been fitted. `stationary_std`
is stored ONCE and both bounds derive their sigma from it, so the two ends cannot drift apart in
amplitude, and re-fitting the amplitude cannot silently change which bound you are on.

TWO TRAPS this encoding exists to avoid.

1. Never fit AR(1) to raw GLOB. Its autocorrelation is dominated by the diurnal cycle, not by
   cloud persistence: lag-1 r = 0.934 implies tau = 14.67 h while the ACF actually e-folds at
   4.28 h. The ACF is periodic, not exponential, so fitting it raw injects a ~10x too persistent
   radiation perturbation. GLOB is therefore fitted on the CLEARNESS INDEX (GLOB / clear-sky,
   daytime only) and wind on the diurnal anomaly (lake breeze removed). `--compare` prints tau
   under every source so this stays visible rather than assumed.

2. sigma is NOT the perturbation's size -- it is the size of one hourly kick. The spread is
   sigma/sqrt(1-phi^2), because a persistent process accumulates its past kicks. Raising phi at
   fixed sigma therefore also raises the amplitude (upperlugano GLOB: 52.1 -> 92.3 W/m2, +77%),
   which confounds persistence with amplitude and makes the experiment uninterpretable. So sigma
   is always re-derived to hold the stationary std at the ICON-fitted value:
   sigma = stat * sqrt(1 - phi^2). Note it FALLS, 45.53 -> 25.72, even as the perturbation grows
   more persistent: smaller kicks, but they accumulate further.

SCOPE. This fits persistence and amplitude only. It does not restructure the perturbation --
additive Cartesian wind is left as is, with its known rectification problem -- and does not
address whether the ICON residual understates true forecast-error amplitude.

Usage:
    python notebooks/perturbations_fit.py args/run_enkf.json --lakes all --upper-bound
    python notebooks/perturbations_fit.py args/run_enkf.json --lake upperlugano --lower-bound
    python notebooks/perturbations_fit.py args/run_enkf.json --lake upperlugano --compare
    python notebooks/perturbations_fit.py args/run_enkf.json --lake upperlugano --icon [--check]
"""
import os
import sys
import json
import logging
import argparse
from datetime import datetime, timedelta, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

# this file lives at <repo>/notebooks/; add src/ so assimilator imports resolve
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from assimilator.functions import (
    ROOT, API_BASE, VARIABLES, verify_args, resolve_src, merge_lake_args,
    select_lakes, run_lake_batch,
)
from assimilator.perturbate import perturbations_path
from assimilator.models.simstrat import SIMSTRAT_REF_YEAR

logger = logging.getLogger(__name__)

DEFAULT_WORKERS = 8
PERTURB_VARS    = ["U", "V", "GLOB"]   # channels perturbed downstream (residual = ICON - control)
MAXLAG          = 96      # h - long enough to see the e-folding of every channel here
DAY_WM2         = 20.0    # W/m2 - above this counts as daytime for the clearness index
CLEAR_PCT       = 95      # percentile of GLOB per (month, hour) used as the clear-sky proxy

CAVEAT = ("phi is bracketed, not measured. The ICON-residual ACF is contaminated by near-white "
          "spatial mismatch and is a floor; the station-signal ACF measures weather persistence "
          "rather than error persistence and is a ceiling. The honest number needs the ACF of "
          "(forecast - station obs), which requires archived ICON forecasts.")


# ---------------------------------------------------------------------------
# ICON acquisition: contours -> per-day download -> flatten -> lake mean
#
# geopandas / requests / tqdm are imported inside these functions on purpose: selecting or
# fitting a persistence bound needs none of them and runs with no network, and that is the
# whole operational point of the bounds encoding.
# ---------------------------------------------------------------------------

def fetch_contours(args: dict) -> dict:
    """Resolve each lake's contour polygon in memory.

    Lakes with a ``key`` are looked up in the bundled ``static/lakes.geojson``
    (``args["contours_geojson"]``); lakes with a ``contour`` are read from the given
    local file. Returns ``{lake_name: geojson_feature}``; nothing is written to disk.
    """
    lakes = args["lakes"]
    remote_lakes = {name: cfg for name, cfg in lakes.items() if "key"     in cfg}
    local_lakes  = {name: cfg for name, cfg in lakes.items() if "contour" in cfg}
    contours = {}

    if remote_lakes:
        geojson_path = args["contours_geojson"]
        logger.info(f"Reading contours from {geojson_path} ...")
        with open(geojson_path, encoding="utf-8") as f:
            geojson = json.load(f)
        key_to_feature = {feat["properties"]["key"]: feat for feat in geojson["features"]}
        for name, cfg in remote_lakes.items():
            key = cfg["key"]
            if key not in key_to_feature:
                logger.warning(f"Key '{key}' not found in {geojson_path} — skipping {name}")
                continue
            contours[name] = key_to_feature[key]
            logger.info(f"Resolved {name} ({key}) from bundled GeoJSON")

    for name, cfg in local_lakes.items():
        src = cfg["contour"]
        if not os.path.exists(src):
            logger.warning(f"[MISSING] {name}: contour file not found at {src}")
            continue
        with open(src, encoding="utf-8") as f:
            contours[name] = json.load(f)
        logger.info(f"Loaded {name} contour from {src}")

    return contours


def _download_day(date_str: str, url: str) -> tuple[str, dict]:
    import requests
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return date_str, r.json()


def retrieve(args: dict, workers: int = DEFAULT_WORKERS) -> dict:
    """Download one ICON reanalysis response per day; returns {date_str: payload}."""
    from tqdm import tqdm

    lake  = args.get("reanalysis_lake", args["lake"])
    start = args["start_date"].date()
    end   = args["end_date"].date()
    bbox  = args["lakes"][lake]["bbox"]

    lat1, lon1, lat2, lon2 = bbox
    dates = [start + timedelta(days=i) for i in range((end - start).days + 1)]

    def _make_task(current):
        date_str = current.strftime("%Y%m%d")
        url = (
            f"{API_BASE}/{date_str}/{date_str}"
            f"/{lat1}/{lon1}/{lat2}/{lon2}?"
            + "&".join(f"variables={v}" for v in VARIABLES)
        )
        return date_str, url

    raw = {}
    errors = 0
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_download_day, *_make_task(d)): d for d in dates}
        with tqdm(as_completed(futures), total=len(futures), desc=f"retrieve {lake}", unit="day") as bar:
            for future in bar:
                try:
                    date_str, payload = future.result()
                    raw[date_str] = payload
                except Exception as e:
                    errors += 1
                    logger.warning(f"retrieve error: {e}")
                bar.set_postfix(errors=errors)

    logger.info(f"{lake}: retrieve done  ({len(raw)} days in memory, {errors} errors)")
    return raw


def _json_to_df(d: dict) -> pd.DataFrame:
    times = d["time"]
    lat   = np.array(d["lat"])
    lon   = np.array(d["lng"])
    T, I, J = len(times), lat.shape[0], lat.shape[1]

    ti, ii, ji = np.meshgrid(range(T), range(I), range(J), indexing="ij")
    df = pd.DataFrame({
        "time": np.array(times)[ti.ravel()],
        "lat":  lat[ii.ravel(), ji.ravel()],
        "lon":  lon[ii.ravel(), ji.ravel()],
    })
    for var, meta in d["variables"].items():
        df[var] = np.array(meta["data"]).ravel()
    return df


def parse_json(args: dict, raw_data: dict) -> pd.DataFrame:
    from tqdm import tqdm

    lake = args.get("reanalysis_lake", args["lake"])
    if not raw_data:
        raise ValueError(f"No raw data to parse for {lake} — retrieve returned nothing")

    chunks = []
    with tqdm(sorted(raw_data), desc=f"parse  {lake}", unit="day") as bar:
        for date_str in bar:
            df = _json_to_df(raw_data[date_str])
            chunks.append(df)
            bar.set_postfix(rows=f"{len(df):,}")

    final = pd.concat(chunks, ignore_index=True)
    logger.info(f"{lake}: flat parsed  ({len(final):,} rows, in memory)")
    return final


def lake_mean(args: dict, flat_df: pd.DataFrame, contours: dict) -> pd.DataFrame:
    import geopandas as gpd

    lake    = args.get("reanalysis_lake", args["lake"])
    feature = (contours or {}).get(lake)
    if feature is None:
        raise ValueError(f"No contour for {lake}")

    gdf = gpd.GeoDataFrame.from_features([feature], crs="EPSG:4326")
    logger.info(f"{lake}: computing lake mask on unique grid points ...")
    unique_pts = flat_df[["lat", "lon"]].drop_duplicates()
    grid_gdf = gpd.GeoDataFrame(
        unique_pts,
        geometry=gpd.points_from_xy(unique_pts["lon"], unique_pts["lat"]),
        crs="EPSG:4326",
    )
    polygon   = gdf.unary_union
    lake_mask = grid_gdf[grid_gdf.within(polygon)][["lat", "lon"]]
    logger.info(f"{lake}: {len(lake_mask)} of {len(unique_pts)} grid points inside lake")

    inside       = flat_df.merge(lake_mask, on=["lat", "lon"])
    vars_present = [v for v in VARIABLES if v in flat_df.columns]
    mean = inside.groupby("time")[vars_present].mean().reset_index()
    logger.info(f"{lake}: lake_mean computed  ({len(mean):,} timesteps, in memory)")
    return mean


# ---------------------------------------------------------------------------
# control forcing (= the station series)
# ---------------------------------------------------------------------------

def read_control_forcing(model_inputs_path: str, ref_year: int, start, end) -> pd.DataFrame:
    """The control Forcing.dat over [start, end], with the (month, hour) keys the deseasonalising
    and clear-sky proxies group on."""
    t0 = pd.Timestamp(f"{ref_year}-01-01")
    df = pd.read_csv(
        os.path.join(model_inputs_path, "Forcing.dat"), sep=r"\s+",
        names=["time_days", "U", "V", "T", "GLOB", "vap", "cloud", "rain"], skiprows=1,
    )
    # 0-based time_days (day 0 = ref_year Jan 1), the same axis as the par/T_out — see perturbate().
    df["time"] = (t0 + pd.to_timedelta(df["time_days"], unit="D")).dt.round("h").dt.tz_localize("UTC")
    out = df[(df["time"] >= start) & (df["time"] <= end)].reset_index(drop=True)
    if out.empty:
        raise ValueError(f"no control forcing rows in {start.date()}..{end.date()} "
                         f"({model_inputs_path}/Forcing.dat)")
    out["month"] = out["time"].dt.month
    out["hour"]  = out["time"].dt.hour
    return out


# ---------------------------------------------------------------------------
# autocorrelation helpers
# ---------------------------------------------------------------------------

def acf(x, maxlag: int = MAXLAG) -> np.ndarray:
    """Sample autocorrelation to `maxlag`. NaNs dropped (the clearness index is night-masked)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    x = x - x.mean()
    denom = float(np.dot(x, x))
    return np.array([np.dot(x[:len(x) - k], x[k:]) / denom for k in range(maxlag + 1)])


def efold_tau(a: np.ndarray) -> float:
    """First crossing of 1/e, linearly interpolated.

    Preferred over the lag-1 estimate tau = -1/ln(r1): a real ACF is not exactly exponential
    (cloud has a heavier tail than AR(1)), and the lag-1 estimate is the one that blows up on a
    periodic signal — see trap 1. The e-folding uses the whole curve and degrades gracefully.
    """
    thresh = np.exp(-1.0)
    for k in range(1, len(a)):
        if a[k] < thresh:
            drop = max(a[k - 1] - a[k], 1e-12)
            return float(k - 1 + (a[k - 1] - thresh) / drop)
    return float(len(a))       # never decorrelates within maxlag


def _deseasonalise(s: pd.Series, month: pd.Series, hour: pd.Series) -> pd.Series:
    """Remove the mean (month, hour) cycle — the diurnal signal that corrupts a raw AR(1) fit."""
    return s - s.groupby([month, hour]).transform("mean")


def clearness_index(glob: pd.Series, month: pd.Series, hour: pd.Series) -> pd.Series:
    """GLOB / clear-sky, daytime only; NaN at night.

    Clear-sky is proxied by the CLEAR_PCT-th percentile of GLOB within each (month, hour) cell
    rather than a radiative-transfer model — crude, but it only has to remove the deterministic
    solar geometry, and any residual bias is common to all cells so it cancels in the ACF.
    """
    clear = glob.groupby([month, hour]).transform(lambda x: np.percentile(x, CLEAR_PCT))
    day   = (glob > DAY_WM2) & (clear > DAY_WM2)
    return pd.Series(np.where(day, glob / clear.replace(0, np.nan), np.nan), index=glob.index)


def persistence_series(ctrl: pd.DataFrame, source: str) -> dict:
    """The series each channel's tau is fitted on, per the chosen source."""
    m, h = ctrl["month"], ctrl["hour"]
    if source == "raw":                    # diagnostic only — see trap 1 in the module docstring
        return {v: ctrl[v] for v in PERTURB_VARS}
    series = {v: _deseasonalise(ctrl[v], m, h) for v in ("U", "V")}
    series["GLOB"] = (clearness_index(ctrl["GLOB"], m, h) if source == "clearness"
                      else _deseasonalise(ctrl["GLOB"], m, h))
    return series


def retained(tau: float, n: int = 12) -> float:
    """Fraction of an AR(1)'s sd that survives averaging over n hours — the quantity that decides
    how much of the perturbation reaches a near-surface layer integrating over that window."""
    phi = np.exp(-1.0 / tau)
    j   = np.arange(1, n)
    return float(np.sqrt((n + 2 * np.sum((n - j) * phi ** j)) / n ** 2))


# ---------------------------------------------------------------------------
# the (phi, sigma) <-> (tau, stationary std) encoding
# ---------------------------------------------------------------------------

def _tau_of(phi: float) -> float:
    return float(-1.0 / np.log(phi))


def _stationary_std(v: dict) -> float:
    """The perturbation's actual spread. sigma is only the size of one hourly kick; a persistent
    process accumulates them, so Var(x) = sigma^2/(1-phi^2). See trap 2."""
    return float(v["sigma"] / np.sqrt(1.0 - v["phi"] ** 2))


def _variables_from(stat: dict, phis: dict) -> dict:
    """(stationary std, phi) -> the (phi, sigma) pair the runtime reads. sigma is ALWAYS derived
    here and never carried over, which is what keeps amplitude out of the persistence question."""
    return {v: {"phi":   round(float(phis[v]), 6),
                "sigma": round(float(stat[v] * np.sqrt(1.0 - phis[v] ** 2)), 6)}
            for v in PERTURB_VARS}


def ensure_persistence(base: dict) -> dict:
    """Give a legacy calibration a `persistence` block, in place.

    Migration is a COPY, never a refit, and it deliberately does NOT touch the top-level
    `variables`: an ICON-only JSON's committed values become the lower bound verbatim, so
    selecting --lower-bound afterwards restores the file bit-for-bit. `variables` is rewritten
    only by an explicit bound selection.
    """
    if "persistence" in base:
        return base

    legacy = (base.get("persistence_refit") or {}).get("by_variable")
    if legacy:
        # written by the pre-bounds refit script: recover the ICON values it replaced, and take
        # the values it left in `variables` as the upper bound (which is what the file is on).
        lower_vars = {v: {"phi": legacy[v]["phi_old"], "sigma": legacy[v]["sigma_old"]}
                      for v in PERTURB_VARS}
        upper_vars = {v: dict(base["variables"][v]) for v in PERTURB_VARS}
        bound = "upper"
    else:
        lower_vars = {v: dict(base["variables"][v]) for v in PERTURB_VARS}
        upper_vars = None
        bound = "lower"

    bounds = {"lower": {"source": "icon-residual",
                        "tau_h": {v: round(_tau_of(lower_vars[v]["phi"]), 3) for v in PERTURB_VARS},
                        "variables": lower_vars}}
    if upper_vars:
        bounds["upper"] = {
            "source": (base.get("persistence_refit") or {}).get("source", "clearness"),
            "tau_h": {v: legacy[v]["tau_h_new"] for v in PERTURB_VARS},
            "variables": upper_vars,
        }
    base.pop("persistence_refit", None)
    base["persistence"] = {
        "bound": bound,
        "stationary_std": {v: round(_stationary_std(lower_vars[v]), 6) for v in PERTURB_VARS},
        "bounds": bounds,
        "caveat": CAVEAT,
    }
    return base


def fit_upper(ctrl: pd.DataFrame, stat: dict, source: str) -> dict:
    """The upper bound: tau from the station signal's own ACF, amplitude held at `stat`.

    A pure function of (stationary_std, control forcing, source) — it never reads the current
    phi/sigma, so re-running it cannot compound.
    """
    series = persistence_series(ctrl, source)
    taus   = {v: efold_tau(acf(series[v])) for v in PERTURB_VARS}
    phis   = {v: float(np.exp(-1.0 / taus[v])) for v in PERTURB_VARS}
    return {"source": source,
            "fitted_on": datetime.now(timezone.utc).date().isoformat(),
            "tau_h": {v: round(taus[v], 3) for v in PERTURB_VARS},
            "variables": _variables_from(stat, phis)}


def build_envelope(bounds: dict) -> dict:
    """The componentwise max-persistence envelope over the two bounds.

    Wanted because the bracket does not always hold: the floor argument assumes the ICON residual
    is contaminated by WHITE noise, and on a channel whose residual carries a slowly drifting bias
    instead (maggiore V, geneva V, hallwil U) the "lower" bound e-folds later than the "upper" one.
    There, taking the upper end would SHORTEN persistence — the one direction that costs ensemble
    spread, and under-spread is the classic EnKF failure. The envelope errs the safe way. Same
    reasoning as the localization fitter's seasonal envelope, and componentwise max is the
    pointwise envelope.

    Each channel is a verbatim COPY of whichever bound won, never a recomputation, so the envelope
    cannot drift from the ends it is built from.
    """
    lo, up = bounds["lower"], bounds["upper"]
    win = {v: ("lower" if lo["tau_h"][v] >= up["tau_h"][v] else "upper") for v in PERTURB_VARS}
    return {
        "source": "componentwise max(lower, upper)",
        "from": win,
        "tau_h": {v: max(lo["tau_h"][v], up["tau_h"][v]) for v in PERTURB_VARS},
        "variables": {v: dict(bounds[win[v]]["variables"][v]) for v in PERTURB_VARS},
    }


def select(base: dict, bound: str) -> dict:
    """Point `variables` at one of the stored bounds. A copy, not a computation."""
    p = base["persistence"]
    if bound not in p["bounds"]:
        raise ValueError(f"no '{bound}' bound stored — fit it first "
                         f"({'--icon' if bound == 'lower' else '--upper-bound'})")
    p["bound"] = bound
    base["variables"] = {v: dict(p["bounds"][bound]["variables"][v]) for v in PERTURB_VARS}
    return base


# ---------------------------------------------------------------------------
# amplitude: the ICON fit
# ---------------------------------------------------------------------------

# Note: Simplified moment-based (Yule–Walker-type) AR(1) estimator.
# Why it's fine here: large n --> estimators asymptotically equivalent, and the purpose is noise
# generation, not inference. The residual is noise-like rather than periodic, so the lag-1
# estimate that trap 1 warns against on raw GLOB is safe on it.
def _fit_ar1(residuals: pd.Series) -> dict:
    r     = residuals.dropna().values
    phi   = float(np.corrcoef(r[:-1], r[1:])[0, 1])
    sigma = float(r.std() * np.sqrt(max(1 - phi**2, 0)))
    return {"phi": round(phi, 6), "sigma": round(sigma, 6)}


def _fit_window(args):
    """Fixed fit window: explicit fit_start/fit_end, else the most recent full year."""
    if args.get("fit_start") and args.get("fit_end"):
        start = pd.Timestamp(args["fit_start"]); end = pd.Timestamp(args["fit_end"])
        start = start.tz_localize("UTC") if start.tz is None else start.tz_convert("UTC")
        end   = end.tz_localize("UTC")   if end.tz   is None else end.tz_convert("UTC")
    else:
        y = datetime.now(timezone.utc).year - 1
        start = pd.Timestamp(f"{y}-01-01", tz="UTC")
        end   = pd.Timestamp(f"{y}-12-31 23:59", tz="UTC")
    return start, end


def fit_amplitude(args: dict, base: dict, run_check: bool = False) -> dict:
    """Acquire ICON, fit AR(1) per variable against the control, and rebuild the amplitude and the
    lower bound from it. The selected bound is preserved: refreshing the amplitude must never
    change which end of the bracket a lake is on."""
    lake     = args["lake"]
    ref_year = args.get("ref_year", SIMSTRAT_REF_YEAR)

    start, end = _fit_window(args)
    args["start_date"] = start.to_pydatetime()   # acquisition reads these
    args["end_date"]   = end.to_pydatetime()
    logger.info(f"{lake}: fitting the ICON residual over {start.date()} .. {end.date()}")

    contours = fetch_contours(args)
    raw      = retrieve(args)
    flat_df  = parse_json(args, raw)
    mean_df  = lake_mean(args, flat_df, contours)

    icon = mean_df.copy()
    if icon["time"].dtype == object:
        icon["time"] = pd.to_datetime(icon["time"])
    if icon["time"].dt.tz is None:
        icon["time"] = icon["time"].dt.tz_localize("UTC")
    icon["T_2M"] -= 273.15

    ctrl = read_control_forcing(args["model_inputs_path"], ref_year, start, end)
    df = pd.merge(
        icon.rename(columns={"U": "U_icon", "V": "V_icon", "GLOB": "GLOB_icon"}),
        ctrl, on="time", how="inner",
    )
    if df.empty:
        raise ValueError(
            f"No overlap between ICON lake_mean and control Forcing.dat for {lake} over the fit window")
    logger.info(f"{lake}: {len(df)} overlapping timesteps for AR(1) fitting")

    # Sanity check: each control var should correlate most strongly with its ICON counterpart
    # (the diagonal must dominate) — catches a mislabeled/mismatched column.
    pairs     = [("U", "U_icon"), ("V", "V_icon"), ("T", "T_2M"), ("GLOB", "GLOB_icon")]
    icon_cols = [ic for _, ic in pairs if ic in df.columns]
    logger.info(f"{lake}: control-vs-ICON correlation (matched pair should be highest in its row):")
    for ctrl_col, match_col in pairs:
        if ctrl_col not in df.columns or match_col not in df.columns:
            continue
        corrs = {ic: float(df[ctrl_col].corr(df[ic])) for ic in icon_cols}
        best  = max(corrs, key=corrs.get)
        row   = "  ".join(f"{ic}={corrs[ic]:+.2f}" for ic in icon_cols)
        flag  = "OK" if best == match_col else f"!! strongest is {best}, expected {match_col}"
        logger.info(f"  control {ctrl_col:4s} -> [{row}]   [{flag}]")

    residual  = {"U": df["U_icon"] - df["U"], "V": df["V_icon"] - df["V"],
                 "GLOB": df["GLOB_icon"] - df["GLOB"]}
    lower_vars = {}
    for v in PERTURB_VARS:
        lower_vars[v] = _fit_ar1(residual[v])
        logger.info(f"  AR(1) {v:4s}  phi={lower_vars[v]['phi']:+.3f}  sigma={lower_vars[v]['sigma']:.4f}")

    # QA plots. Lazy import: check pulls matplotlib and is only needed with --check.
    if run_check:
        from check_perturbations import check
        check(args, flat_df=flat_df, mean_df=mean_df, contours=contours, fit_df=df, params=lower_vars)

    base = dict(base or {})
    base.update({
        "lake": lake,
        "reanalysis_lake": args.get("reanalysis_lake", lake),
        "source": "ICON KENDA-CH1 - control Forcing.dat",
        "fit_window": {"start": start.date().isoformat(), "end": end.date().isoformat()},
        "n_timesteps": int(len(df)),
        "fitted_on": datetime.now(timezone.utc).date().isoformat(),
    })
    kept = (base.get("persistence") or {}).get("bound", "lower")
    base["persistence"] = {
        "bound": kept,
        "stationary_std": {v: round(_stationary_std(lower_vars[v]), 6) for v in PERTURB_VARS},
        "bounds": {"lower": {"source": "icon-residual",
                             "fitted_on": base["fitted_on"],
                             "tau_h": {v: round(_tau_of(lower_vars[v]["phi"]), 3)
                                       for v in PERTURB_VARS},
                             "variables": lower_vars}},
        "caveat": CAVEAT,
    }
    base.pop("persistence_refit", None)
    return base


# ---------------------------------------------------------------------------
# per-lake driver
# ---------------------------------------------------------------------------

def _load(path: str):
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _report(lake: str, before: dict, base: dict) -> None:
    """Print the bracket, not just the selected end.

    `bracket` is tau_upper/tau_lower — how much room there is between the floor and the ceiling.
    It varies enormously by lake (upperlugano x5, murten x12), and a wide one means the selected
    end is barely constrained, so it belongs in front of anyone reading the fit.
    """
    p    = base["persistence"]
    b    = p["bound"]
    low  = p["bounds"]["lower"]["tau_h"]
    up   = (p["bounds"].get("upper") or {}).get("tau_h")
    pick = p["bounds"][b].get("from") or {v: b for v in PERTURB_VARS}
    logger.info(f"{lake}: bound = {b} ({p['bounds'][b].get('source', '?')})")
    logger.info(f"  {'var':<6}{'tau lower':>10}{'tau upper':>11}{'bracket':>9}{'pick':>9}"
                f"{'phi':>9}{'sigma':>10}{'stat':>9}{'12h kept':>10}")
    for v in PERTURB_VARS:
        new = base["variables"][v]
        tl, tu = low[v], (up or {}).get(v, float("nan"))
        logger.info(f"  {v:<6}{tl:>10.2f}{tu:>11.2f}{tu / tl:>8.1f}x{pick[v]:>9}"
                    f"{new['phi']:>9.3f}{new['sigma']:>10.4f}"
                    f"{p['stationary_std'][v]:>9.3f}{retained(_tau_of(new['phi'])):>10.1%}")
    if before and before != base["variables"]:
        logger.info(f"  changed from phi " + ", ".join(
            f"{v}={before[v]['phi']:.3f}->{base['variables'][v]['phi']:.3f}" for v in PERTURB_VARS))

    # An inverted bracket falsifies the ordering for that channel: the floor argument assumes the
    # ICON residual is signal + WHITE contamination, so a residual carrying slow structure (a
    # drifting bias rather than grid noise) can e-fold later than the station signal does. Where
    # that happens the two ends are not a bracket and neither should be read as a bound.
    inverted = [v for v in PERTURB_VARS if up and up[v] < low[v]]
    if inverted:
        held = " (envelope holds the longer one)" if b == "envelope" else ""
        logger.warning(f"  {lake}: bracket INVERTED for {', '.join(inverted)} — "
                       f"tau_upper < tau_lower, so the ICON residual is not white-contaminated "
                       f"there and neither end is a bound for that channel{held}")


def fit_lake(args: dict, bound: str = None, source: str = None, use_icon: bool = False,
             run_check: bool = False, dry_run: bool = False) -> dict:
    lake     = args["lake"]
    ref_year = args.get("ref_year", SIMSTRAT_REF_YEAR)
    path     = perturbations_path(args)
    base     = _load(path)
    before   = dict(base["variables"]) if base and "variables" in base else None

    if base is None and not use_icon:
        raise FileNotFoundError(
            f"{os.path.relpath(path, ROOT)} not found — the amplitude comes from the ICON "
            f"residual, so a new lake needs one --icon run (EAWAG VPN) before a bound can be set")

    if use_icon:
        base = fit_amplitude(args, base, run_check=run_check)
    else:
        base = ensure_persistence(base)

    p      = base["persistence"]
    target = bound or p["bound"]

    # (Re)fit the upper bound whenever it is wanted or already stored. It reads only
    # stationary_std and the control forcing — never the current phi/sigma — so this is
    # idempotent, and re-running it after an amplitude refit keeps the two ends consistent.
    if target in ("upper", "envelope") or "upper" in p["bounds"]:
        src  = source or p["bounds"].get("upper", {}).get("source") or "clearness"
        win  = base["fit_window"]
        start = pd.Timestamp(win["start"], tz="UTC")
        end   = pd.Timestamp(win["end"], tz="UTC") + pd.Timedelta(hours=23, minutes=59)
        ctrl  = read_control_forcing(args["model_inputs_path"], ref_year, start, end)
        logger.info(f"{lake}: {len(ctrl)} control rows over {start.date()}..{end.date()}, "
                    f"upper-bound source = '{src}'")
        p["bounds"]["upper"] = fit_upper(ctrl, p["stationary_std"], src)

    # The envelope is a function of the two ends, so it is rebuilt whenever they could have moved.
    if "upper" in p["bounds"]:
        p["bounds"]["envelope"] = build_envelope(p["bounds"])

    select(base, target)
    _report(lake, before, base)

    if dry_run:
        logger.info(f"  --dry-run: would write {os.path.relpath(path, ROOT)}")
        return base
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(base, f, indent=2)
        f.write("\n")
    logger.info(f"  wrote {os.path.relpath(path, ROOT)}")
    return base


def compare(args: dict) -> None:
    """Print tau under every persistence source, so the bracket's width and the raw-GLOB trap are
    both visible rather than assumed."""
    lake     = args["lake"]
    ref_year = args.get("ref_year", SIMSTRAT_REF_YEAR)
    base     = _load(perturbations_path(args))
    if base is None:
        raise FileNotFoundError(f"no calibration for {lake} — run --icon first")
    ensure_persistence(base)

    win   = base["fit_window"]
    start = pd.Timestamp(win["start"], tz="UTC")
    end   = pd.Timestamp(win["end"], tz="UTC") + pd.Timedelta(hours=23, minutes=59)
    ctrl  = read_control_forcing(args["model_inputs_path"], ref_year, start, end)
    lower = base["persistence"]["bounds"]["lower"]["tau_h"]
    srcs  = {s: persistence_series(ctrl, s) for s in ("raw", "anomaly", "clearness")}

    print(f"\n{lake}: e-fold tau (h) by persistence source, {start.date()}..{end.date()}")
    print(f"{'channel':<9}{'ICON resid':>12}{'raw':>10}{'anomaly':>10}{'clearness':>11}"
          f"   <- lower bound ................ upper bound ->")
    print("-" * 52)
    for v in PERTURB_VARS:
        taus = {s: efold_tau(acf(srcs[s][v])) for s in srcs}
        print(f"{v:<9}{lower[v]:>12.2f}{taus['raw']:>10.2f}{taus['anomaly']:>10.2f}"
              f"{taus['clearness']:>11.2f}")

    a_raw = acf(ctrl["GLOB"])
    print(f"\nGLOB raw: lag-1 r={a_raw[1]:.3f} -> AR(1) tau={-1/np.log(a_raw[1]):.2f} h, but the ACF "
          f"e-folds at {efold_tau(a_raw):.2f} h.\nThe gap is the diurnal cycle: do not fit raw GLOB "
          f"(trap 1).")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_args(raw: dict) -> dict:
    """Resolve the paths both entry points need. The ICON acquisition keys (bbox/contour/
    ensemble_base) are resolved separately in `icon_args`, so a bound selection never requires
    them."""
    args = dict(raw)
    verify_args(args, ["lake"])
    args.setdefault("model_inputs_path", os.path.join(ROOT, "inputs", args["lake"]))
    args.setdefault("perturbations_dir", os.path.join(ROOT, "perturbations"))
    return args


def icon_args(args: dict) -> dict:
    """Add what the ICON download needs, on top of build_args."""
    verify_args(args, ["lake", "lake_bbox", "ensemble_base"])
    lake_cfg = {"bbox": tuple(args["lake_bbox"])}
    if "lake_key"     in args:
        lake_cfg["key"]     = args["lake_key"]
    if "lake_contour" in args:
        lake_cfg["contour"] = args["lake_contour"]

    reanalysis_lake         = args.get("reanalysis_lake", args["lake"])
    args["reanalysis_lake"] = reanalysis_lake
    args["lakes"]           = {reanalysis_lake: lake_cfg}
    args["ensemble_base"]   = resolve_src(args["ensemble_base"])
    args.setdefault("contours_geojson", os.path.join(ROOT, "static", "lakes.geojson"))
    return args


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    ap.add_argument("arg_file", help="Path to a run config JSON (e.g. args/run_enkf.json)")
    grp = ap.add_mutually_exclusive_group()
    grp.add_argument("--lake", default=None)
    grp.add_argument("--lakes", default=None,
                     help="comma-separated, or 'all' for every lake in the config")
    bnd = ap.add_mutually_exclusive_group()
    bnd.add_argument("--lower-bound", dest="bound", action="store_const", const="lower",
                     help="persistence from the ICON-residual ACF (floor); a stored copy, no refit")
    bnd.add_argument("--upper-bound", dest="bound", action="store_const", const="upper",
                     help="persistence from the station-signal ACF (ceiling), amplitude held")
    bnd.add_argument("--envelope", dest="bound", action="store_const", const="envelope",
                     help="per channel, whichever of the two has the LONGER tau — errs toward "
                          "more ensemble spread where the bracket inverts")
    ap.add_argument("--source", default=None, choices=["clearness", "anomaly", "raw"],
                    help="series the upper bound is fitted on (default: keep stored, else "
                         "clearness — see trap 1)")
    ap.add_argument("--icon", action="store_true",
                    help="re-fit the amplitude and the lower bound from ICON (needs the EAWAG VPN)")
    ap.add_argument("--check", action="store_true", help="with --icon, also write the QA plots")
    ap.add_argument("--compare", action="store_true",
                    help="print tau under every source and exit, writing nothing")
    ap.add_argument("--dry-run", action="store_true", help="report the fit, write nothing")
    cli = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")

    arg_file = cli.arg_file if os.path.isfile(cli.arg_file) else os.path.join(ROOT, cli.arg_file)
    if not os.path.isfile(arg_file):
        raise ValueError(f"Args file not found: {cli.arg_file}")
    with open(arg_file) as f:
        raw = json.load(f)

    lakes = select_lakes(raw, cli.lakes, cli.lake)

    def run_one(lake):
        args = build_args(merge_lake_args(raw, lake=lake))
        if cli.compare:
            return compare(args)
        if cli.icon:
            args = icon_args(args)
        fit_lake(args, bound=cli.bound, source=cli.source, use_icon=cli.icon,
                 run_check=cli.check, dry_run=cli.dry_run)

    if run_lake_batch(lakes, run_one, what="fitted"):
        sys.exit(1)


if __name__ == "__main__":
    main()

"""Fit the AR(1) forcing perturbation -> perturbations/<lake>.json.

The calibration is TWO things, fitted from different data and kept strictly separate:

  AMPLITUDE    how big the perturbation is: the stationary std sigma/sqrt(1-phi^2). Fitted from
               the ICON residual (ICON lake mean - control Forcing.dat). Needs the ICON API /
               EAWAG VPN, so it is fitted once per lake and committed. --icon does this.

  PERSISTENCE  how long the perturbation stays correlated with itself, phi = exp(-1/tau).
               This CANNOT be measured: it needs the ACF of (forecast - truth), and there are no
               archived ICON forecasts. So it is BRACKETED by two proxies, each contaminated in a
               direction. Fitted from the control Forcing.dat alone -- no ICON, no VPN.

    lower   the ACF of the ICON residual. Much of that residual flickers hour to hour like 
            white noise, and adding white noise can only SHORTEN a measured decorrelation time. 
            FLOOR.

    upper   the ACF of the station signal itself. This measures how long the WEATHER persists,
            not how long the ERROR does: a forecast already captures the slow synoptic structure
            and fails on the fast small-scale part, so the error is enriched in fast components
            and decorrelates sooner. CEILING.

    envelope (default) per channel, whichever end has the LONGER tau. The bracket does not always
            hold -- measured. Taking the shorter tau would cost ensemble
            spread, and under-spread is the known EnKF failure, so the envelope errs safely.

  Both arguments are physical, not proofs. Treat a run on either end as one side of a range, not
  as a measurement. The honest number still needs the ACF of (forecast - station obs).

TWO TRAPS this exists to avoid.

  1. NEVER fit AR(1) to raw GLOB. Its autocorrelation is dominated by the diurnal cycle. U and V 
     are deseasonalised by (month, hour); GLOB uses the clearness index (GLOB / clear-sky proxy, daytime only).
  2. sigma is NOT the perturbation's spread. It is the size of one hourly kick; a persistent
     process accumulates them, so the spread is sigma/sqrt(1-phi^2). Amplitude is therefore
     stored ONCE as that stationary std and each bound DERIVES its sigma from it, so the two ends
     cannot drift apart in amplitude.

Only `variables` is read at run time (src/assimilator/perturbate.py); the rest of the JSON is
provenance. Switching bound rewrites `variables` from stored numbers, so it is exactly
reversible, and works offline once a lake has been fitted.

Folds the former fetch_contours / retrieve / parse_json / lake_mean / logging_utils.

Usage:
    python notebooks/generate_perturbation.py args/run_enkf.json --icon   # amplitude + lower (VPN)
    python notebooks/generate_perturbation.py args/run_enkf.json          # upper + envelope, offline
    python notebooks/generate_perturbation.py args/run_enkf.json --lower-bound
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
import geopandas as gpd
import requests
from tqdm import tqdm

# this file lives at <repo>/notebooks/; add src/ so assimilator imports resolve
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from assimilator.functions import (
    ROOT, API_BASE, VARIABLES, verify_args, resolve_src, merge_lake_args,
)
from assimilator.models.simstrat import SIMSTRAT_REF_YEAR

logger = logging.getLogger(__name__)

DEFAULT_WORKERS = 8
PERTURB_VARS    = ["U", "V", "GLOB"]   # channels perturbed downstream (residual = ICON - control)

BOUNDS    = ("lower", "upper", "envelope")
MAXLAG    = 96      # h - long enough to see the e-folding of every channel here
DAY_WM2   = 20.0    # W/m2 - above this counts as daytime for the clearness index
CLEAR_PCT = 95      # percentile of GLOB per (month, hour) used as the clear-sky proxy

CAVEAT = ("phi is bracketed, not measured. lower is a floor (the ICON residual carries white "
          "spatial mismatch, which shortens any measured tau); upper is a ceiling (the station "
          "signal measures weather persistence, not error persistence); envelope takes the longer "
          "tau per channel. The honest number needs the ACF of (forecast - station obs), which "
          "requires archived ICON forecasts.")


# ---------------------------------------------------------------------------
# ICON acquisition: contours -> per-day download -> flatten -> lake mean
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
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    return date_str, r.json()


def retrieve(args: dict, workers: int = DEFAULT_WORKERS) -> dict:
    """Download one ICON reanalysis response per day; returns {date_str: payload}."""
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
# AR(1) fit
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# PERSISTENCE — tau from an autocorrelation function. Control forcing only; no ICON, no VPN.
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
    (cloud has a heavier tail than AR(1)), and lag-1 is the estimate that blows up on a periodic
    signal -- trap 1. The e-folding uses the whole curve and degrades better."""
    thresh = np.exp(-1.0)
    for k in range(1, len(a)):
        if a[k] < thresh:
            drop = max(a[k - 1] - a[k], 1e-12)
            return float(k - 1 + (a[k - 1] - thresh) / drop)
    return float(len(a))       # never decorrelates within maxlag


def _deseasonalise(s: pd.Series, month: pd.Series, hour: pd.Series) -> pd.Series:
    """Remove the mean (month, hour) cycle -- the diurnal signal that corrupts a raw fit."""
    return s - s.groupby([month, hour]).transform("mean")


def clearness_index(glob: pd.Series, month: pd.Series, hour: pd.Series) -> pd.Series:
    """GLOB / clear-sky, daytime only; NaN at night.

    Clear-sky is proxied by the CLEAR_PCT-th percentile of GLOB within each (month, hour) cell
    (crude), but it only has to remove the deterministic solar geometry, and any residual bias 
    is common to all cells so it cancels in the ACF."""
    clear = glob.groupby([month, hour]).transform(lambda x: np.percentile(x, CLEAR_PCT))
    day   = (glob > DAY_WM2) & (clear > DAY_WM2)
    return pd.Series(np.where(day, glob / clear.replace(0, np.nan), np.nan), index=glob.index)


def persistence_series(ctrl: pd.DataFrame, source: str = "clearness") -> dict:
    """The series each channel's tau is fitted on. 'raw' is diagnostic only -- see trap 1."""
    m, h = ctrl["time"].dt.month, ctrl["time"].dt.hour
    if source == "raw":
        return {v: ctrl[v] for v in PERTURB_VARS}
    series = {v: _deseasonalise(ctrl[v], m, h) for v in ("U", "V")}
    series["GLOB"] = (clearness_index(ctrl["GLOB"], m, h) if source == "clearness"
                      else _deseasonalise(ctrl["GLOB"], m, h))
    return series

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


def fit_upper(ctrl: pd.DataFrame, source: str = "clearness") -> dict:
    """The upper bound: tau from the station signal's own ACF.
    A pure function of (control forcing, source) -- it never reads the current phi/sigma, so
    re-running it cannot compound. Amplitude is not touched: only tau is fitted here."""
    series = persistence_series(ctrl, source)
    taus   = {v: efold_tau(acf(series[v])) for v in PERTURB_VARS}
    return {"source": f"ACF of the station signal ({source}); a ceiling",
            "fitted_on": datetime.now(timezone.utc).date().isoformat(),
            "tau_h": {v: round(taus[v], 3) for v in PERTURB_VARS},
            "phi":   {v: round(float(np.exp(-1.0 / taus[v])), 6) for v in PERTURB_VARS}}


def choose(persistence: dict, bound: str) -> dict:
    """Which end each channel takes. 'envelope' picks the LONGER tau per channel -- the bracket
    does not always hold, and the shorter tau is the direction that costs ensemble spread."""
    if bound != "envelope":
        return {v: bound for v in PERTURB_VARS}
    lo, up = persistence["lower"]["tau_h"], persistence["upper"]["tau_h"]
    return {v: ("lower" if lo[v] >= up[v] else "upper") for v in PERTURB_VARS}


def select(base: dict, bound: str) -> dict:
    """Point `variables` at one of the stored bounds, in place.
    phi is COPIED from the chosen bound (never recomputed from the rounded tau_h) and sigma is
    DERIVED from the single stored stationary_std -- so switching bound is exactly reversible and
    idempotent, and the two ends can never disagree about amplitude."""
    p = base["persistence"]
    missing = [b for b in (("lower", "upper") if bound == "envelope" else (bound,)) if b not in p]
    if missing:
        raise ValueError(f"{base['lake']}: no {missing} bound stored — fit it first "
                         f"(--icon for lower, a plain run for upper)")
    chosen = choose(p, bound)
    p["bound"], p["chosen"] = bound, chosen
    base["variables"] = _variables_from(base["amplitude"]["stationary_std"],
                                        {v: p[chosen[v]]["phi"][v] for v in PERTURB_VARS})
    return base


def migrate(base: dict) -> dict:
    """Give a pre-bounds calibration the new shape, in place. A COPY, never a refit.
    The old files carry one AR(1) fitted on the ICON residual, i.e. exactly the LOWER bound, so
    its phi/sigma become that bound verbatim and the stationary std is recovered from them.
    `variables` is left alone: selecting --lower-bound afterwards restores the file bit-for-bit."""
    if "persistence" in base and "amplitude" in base:
        return base
    old = base["variables"]
    base["amplitude"] = {
        "source": base.get("source", "ICON KENDA-CH1 lake mean - control Forcing.dat"),
        "stationary_std": {v: round(_stationary_std(old[v]), 6) for v in PERTURB_VARS},
    }
    base["persistence"] = {
        "bound": "lower",
        "chosen": {v: "lower" for v in PERTURB_VARS},
        "lower": {"source": "ACF of the ICON residual; a floor",
                  "tau_h": {v: round(_tau_of(old[v]["phi"]), 3) for v in PERTURB_VARS},
                  "phi":   {v: old[v]["phi"] for v in PERTURB_VARS}},
        "_note": CAVEAT,
    }
    base.pop("source", None)
    return base

# ---------------------------------------------------------------------------
# AMPLITUDE — the ICON residual fit
# ---------------------------------------------------------------------------
# Note: Simplified moment-based (Yule–Walker-type) AR(1) estimator
# Can be improved just a first proof of concept ...
# Why it's fine here: Large n --> estimators asymptotically equivalent, purpose is noise generation
# and not inference, just numpy...
def _fit_ar1(residuals: pd.Series) -> dict:
    r     = residuals.dropna().values
    phi   = float(np.corrcoef(r[:-1], r[1:])[0, 1])
    sigma = float(r.std() * np.sqrt(max(1 - phi**2, 0)))
    return {"phi": round(phi, 6), "sigma": round(sigma, 6)}


def _read_control_forcing(model_inputs_path: str, ref_year: int, start, end) -> pd.DataFrame:
    t0  = pd.Timestamp(f"{ref_year}-01-01")
    std = pd.read_csv(
        os.path.join(model_inputs_path, "Forcing.dat"),
        sep=r"\s+",
        names=["time_days", "U", "V", "T", "GLOB", "vap", "cloud", "rain"],
        skiprows=1,
    )
    # 0-based time_days (day 0 = ref_year Jan 1), matching the par/T_out axis — see perturbate().
    # (Was `- 1`, which misaligned the ICON-vs-control residual by one day.)
    std["time"] = (t0 + pd.to_timedelta(std["time_days"], unit="D")).dt.round("h").dt.tz_localize("UTC")
    return std[(std["time"] >= start) & (std["time"] <= end)].reset_index(drop=True)


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


def generate_perturbations(args: dict, run_check: bool = False) -> dict:
    """Acquire ICON, fit AR(1) per variable vs the control, write perturbations/<lake>.json."""
    lake            = args["lake"]
    reanalysis_lake = args.get("reanalysis_lake", lake)
    ref_year        = args.get("ref_year", SIMSTRAT_REF_YEAR)
    perturb_dir     = args.get("perturbations_dir", os.path.join(ROOT, "perturbations"))

    start, end = _fit_window(args)
    args["start_date"] = start.to_pydatetime()   # acquisition reads these
    args["end_date"]   = end.to_pydatetime()
    logger.info(f"{lake}: fitting AR(1) over {start.date()} .. {end.date()}")

    # ICON acquisition -> lake-mean timeseries
    contours = fetch_contours(args)
    raw      = retrieve(args)
    flat_df  = parse_json(args, raw)
    mean_df  = lake_mean(args, flat_df, contours)

    # ICON lake mean, aligned to UTC, T in °C
    icon = mean_df.copy()
    if icon["time"].dtype == object:
        icon["time"] = pd.to_datetime(icon["time"])
    if icon["time"].dt.tz is None:
        icon["time"] = icon["time"].dt.tz_localize("UTC")
    icon["T_2M"] -= 273.15

    # residuals (ICON - control) over the fit window
    ctrl = _read_control_forcing(args["model_inputs_path"], ref_year, start, end)
    df = pd.merge(
        icon.rename(columns={"U": "U_icon", "V": "V_icon", "GLOB": "GLOB_icon"}),
        ctrl, on="time", how="inner",
    )
    if df.empty:
        raise ValueError(
            f"No overlap between ICON lake_mean and control Forcing.dat for {lake} over the fit window")
    logger.info(f"{lake}: {len(df)} overlapping timesteps for AR(1) fitting")

    # Sanity check: each control var should correlate most strongly with its ICON
    # counterpart (the diagonal must dominate) — catches a mislabeled/mismatched column.
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

    residual = {"U": df["U_icon"] - df["U"], "V": df["V_icon"] - df["V"], "GLOB": df["GLOB_icon"] - df["GLOB"]}
    variables = {}
    for v in PERTURB_VARS:
        variables[v] = _fit_ar1(residual[v])
        logger.info(f"  AR(1) {v:4s}  phi={variables[v]['phi']:+.3f}  sigma={variables[v]['sigma']:.4f}")

    # QA plots (acquisition: check.png; fit: residual ACF/distribution + preview ensemble: check_fit.png).
    # Lazy import: check pulls matplotlib and is only needed with --check.
    if run_check:
        from check_perturbations import check
        check(args, flat_df=flat_df, mean_df=mean_df, contours=contours, fit_df=df, params=variables)

    # The ICON residual gives BOTH halves at once: its stationary std is the amplitude, and its
    # ACF (via the fitted phi) is the LOWER persistence bound. They are stored apart so the
    # amplitude survives a change of bound, and vice versa.
    return {
        "lake": lake,
        "reanalysis_lake": reanalysis_lake,
        "fit_window": {"start": start.date().isoformat(), "end": end.date().isoformat()},
        "n_timesteps": int(len(df)),
        "fitted_on": datetime.now(timezone.utc).date().isoformat(),
        "variables": variables,                     # provisional: select() rewrites it
        "amplitude": {
            "source": "ICON KENDA-CH1 lake mean - control Forcing.dat",
            "stationary_std": {v: round(_stationary_std(variables[v]), 6) for v in PERTURB_VARS},
        },
        "persistence": {
            "bound": "lower",
            "chosen": {v: "lower" for v in PERTURB_VARS},
            "lower": {"source": "ACF of the ICON residual; a floor",
                      "tau_h": {v: round(_tau_of(variables[v]["phi"]), 3) for v in PERTURB_VARS},
                      "phi":   {v: variables[v]["phi"] for v in PERTURB_VARS}},
            "_note": CAVEAT,
        },
    }


# ---------------------------------------------------------------------------
# CLI wrapper
# ---------------------------------------------------------------------------

REQUIRED = ["lake", "lake_bbox", "ensemble_base"]


def build_args(raw: dict) -> dict:
    args = dict(raw)

    lake     = args["lake"]
    lake_cfg = {"bbox": tuple(args["lake_bbox"])}
    if "lake_key"     in args:
        lake_cfg["key"]     = args["lake_key"]
    if "lake_contour" in args:
        lake_cfg["contour"] = args["lake_contour"]

    reanalysis_lake         = args.get("reanalysis_lake", lake)
    args["reanalysis_lake"] = reanalysis_lake
    args["lakes"]           = {reanalysis_lake: lake_cfg}

    args.setdefault("contours_geojson", os.path.join(ROOT, "static", "lakes.geojson"))

    ensemble_base = resolve_src(args["ensemble_base"])
    args["ensemble_base"] = ensemble_base
    args.setdefault("model_inputs_path", os.path.join(ROOT, "inputs", args["lake"]))
    args.setdefault("perturbations_dir",    os.path.join(ROOT, "perturbations"))
    return args


def _ordered(base: dict) -> dict:
    """Stable key order on disk, so a re-fit diffs only where numbers changed."""
    p = base["persistence"]
    base["persistence"] = {k: p[k] for k in ("bound", "chosen", "lower", "upper", "_note")
                           if k in p}
    head = ("lake", "reanalysis_lake", "fit_window", "n_timesteps", "fitted_on",
            "variables", "amplitude", "persistence")
    return {**{k: base[k] for k in head if k in base},
            **{k: v for k, v in base.items() if k not in head}}


def _report(lake: str, base: dict) -> None:
    """Per channel: the bracket, which end was taken, and what reaches the runtime."""
    p, amp = base["persistence"], base["amplitude"]["stationary_std"]
    have_up = "upper" in p
    logger.info(f"{lake}: bound={p['bound']}")
    logger.info(f"  {'chan':<5} {'tau_lo':>7} {'tau_up':>7}  {'take':<6} {'phi':>8} {'sigma':>10} "
                f"{'stat.std':>9}")
    for v in PERTURB_VARS:
        lo = p["lower"]["tau_h"][v] if "lower" in p else float("nan")
        up = p["upper"]["tau_h"][v] if have_up else float("nan")
        logger.info(f"  {v:<5} {lo:7.2f} {up:7.2f}  {p['chosen'][v]:<6} "
                    f"{base['variables'][v]['phi']:8.4f} {base['variables'][v]['sigma']:10.4f} "
                    f"{amp[v]:9.4f}")


def fit_lake(args: dict, bound: str = "envelope", source: str = "clearness",
             use_icon: bool = False, run_check: bool = False, dry_run: bool = False) -> dict:
    """One lake: fit what is asked for, select a bound, write the JSON.

    --icon refits the AMPLITUDE (and with it the lower bound) and needs the VPN. Without it the
    committed calibration is read and only the upper bound is (re)fitted, from the control
    Forcing.dat -- offline, and safe to repeat, since fit_upper never reads the current phi."""
    lake        = args["lake"]
    perturb_dir = args.get("perturbations_dir", os.path.join(ROOT, "perturbations"))
    out_path    = os.path.join(perturb_dir, f"{lake}.json")

    if use_icon:
        base = generate_perturbations(args, run_check=run_check)
    else:
        if not os.path.isfile(out_path):
            raise FileNotFoundError(
                f"{out_path} not found — the amplitude has never been fitted for {lake}. "
                f"Run once with --icon (needs the ICON API / EAWAG VPN).")
        with open(out_path, encoding="utf-8") as f:
            base = migrate(json.load(f))

    # The upper bound needs only the control forcing over the fit window.
    if bound != "lower" or "upper" not in base["persistence"]:
        start, end = _fit_window({"fit_start": base["fit_window"]["start"],
                                  "fit_end":   base["fit_window"]["end"] + " 23:59"})
        ctrl = _read_control_forcing(args["model_inputs_path"],
                                     args.get("ref_year", SIMSTRAT_REF_YEAR), start, end)
        if ctrl.empty:
            raise ValueError(f"{lake}: control Forcing.dat has no rows over {start.date()}"
                             f"..{end.date()} — cannot fit the upper bound")
        base["persistence"]["upper"] = fit_upper(ctrl, source)

    select(base, bound)
    _report(lake, base)

    if dry_run:
        logger.info(f"{lake}: dry run — {os.path.relpath(out_path, ROOT)} not written")
        return base
    os.makedirs(perturb_dir, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(_ordered(base), f, indent=2)
        f.write("\n")
    logger.info(f"{lake}: wrote {os.path.relpath(out_path, ROOT)}")
    return base


def fit(raw_args: dict, run_check: bool = False, **kw) -> dict:
    verify_args(raw_args, REQUIRED)
    return fit_lake(build_args(raw_args), run_check=run_check, **kw)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("arg_file", help="Path to a run config JSON (e.g. args/run_enkf.json)")
    ap.add_argument("--lake", default=None, help="Lake to fit from the config's \"lakes\" block")
    ap.add_argument("--lakes", nargs="*", default=None,
                    help="fit several lakes from the config's \"lakes\" block (default: just --lake)")
    bnd = ap.add_mutually_exclusive_group()
    bnd.add_argument("--lower-bound", dest="bound", action="store_const", const="lower",
                     help="ICON-residual persistence — the floor")
    bnd.add_argument("--upper-bound", dest="bound", action="store_const", const="upper",
                     help="station-signal persistence — the ceiling")
    bnd.add_argument("--envelope", dest="bound", action="store_const", const="envelope",
                     help="per channel the longer tau (default)")
    ap.add_argument("--source", default="clearness", choices=["clearness", "anomaly", "raw"],
                    help="series the upper bound's tau is fitted on (default %(default)s; 'raw' is "
                         "diagnostic only — see trap 1)")
    ap.add_argument("--icon", action="store_true",
                    help="refit the amplitude from ICON (needs the API / EAWAG VPN)")
    ap.add_argument("--check", action="store_true", help="with --icon, also write the QA plots")
    ap.add_argument("--dry-run", action="store_true", help="report the fit, write nothing")
    cli = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-8s | %(name)-16s | %(message)s",
                        datefmt="%H:%M:%S")

    arg_file = cli.arg_file if os.path.isfile(cli.arg_file) else os.path.join(ROOT, cli.arg_file)
    if not os.path.isfile(arg_file):
        raise ValueError(f"Args file not found: {cli.arg_file}")
    with open(arg_file, encoding="utf-8") as f:
        raw = json.load(f)

    lakes = cli.lakes if cli.lakes else [cli.lake]
    for lake in lakes:
        fit(merge_lake_args(raw, lake=lake), run_check=cli.check,
            bound=cli.bound or "envelope", source=cli.source,
            use_icon=cli.icon, dry_run=cli.dry_run)


if __name__ == "__main__":
    main()

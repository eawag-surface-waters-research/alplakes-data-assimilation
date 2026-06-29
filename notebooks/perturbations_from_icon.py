"""Fit AR(1) forcing-noise statistics from ICON reanalysis -> perturbations/<lake>.json.

Downloads ICON KENDA-CH1 over a fixed window, reduces it to a lake-mean meteo series,
computes residuals against the control Forcing.dat, fits an AR(1) model (phi, sigma) per
perturbed variable (U, V, GLOB), and writes the calibration to perturbations/<lake>.json.

Heavy and rarely run (needs the ICON API / EAWAG VPN); run once per lake or to recalibrate.
The apply step (src/assimilator/perturbate.py, run by main.py) reuses the committed JSON
and needs none of these deps.

Folds the former fetch_contours / retrieve / parse_json / lake_mean / logging_utils.

Usage:  python notebooks/perturbations_from_icon.py args/run_enkf.json [--check]
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


def fit_perturbations(args: dict, run_check: bool = False) -> dict:
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

    out = {
        "lake": lake,
        "reanalysis_lake": reanalysis_lake,
        "source": "ICON KENDA-CH1 - control Forcing.dat",
        "fit_window": {"start": start.date().isoformat(), "end": end.date().isoformat()},
        "n_timesteps": int(len(df)),
        "fitted_on": datetime.now(timezone.utc).date().isoformat(),
        "variables": variables,
    }
    os.makedirs(perturb_dir, exist_ok=True)
    out_path = os.path.join(perturb_dir, f"{lake}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    logger.info(f"{lake}: wrote {os.path.relpath(out_path, ROOT)}")
    return out


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


def fit(raw_args: dict, run_check: bool = False) -> dict:
    verify_args(raw_args, REQUIRED)
    args = build_args(raw_args)
    return fit_perturbations(args, run_check=run_check)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fit AR(1) forcing-noise stats from ICON")
    parser.add_argument("arg_file", help="Path to a run config JSON (e.g. args/run_enkf.json)")
    parser.add_argument("--lake", default=None, help="Lake to fit from the config's \"lakes\" block")
    parser.add_argument("--check", action="store_true", help="Also write the QA check.png")
    cli = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-8s | %(name)-16s | %(message)s",
                        datefmt="%H:%M:%S")

    arg_file = cli.arg_file if os.path.isfile(cli.arg_file) else os.path.join(ROOT, cli.arg_file)
    if not os.path.isfile(arg_file):
        raise ValueError(f"Args file not found: {cli.arg_file}")

    with open(arg_file) as f:
        fit(merge_lake_args(json.load(f), lake=cli.lake), run_check=cli.check)

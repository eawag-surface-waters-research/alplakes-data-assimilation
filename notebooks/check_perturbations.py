"""QA plots for the AR(1) forcing perturbations.

`check.png`       — grid-point selection map + lake-mean meteo series (acquisition QA).
`check_fit.png`   — per perturbed variable: residual ACF vs fitted phi^k, residual
                    distribution vs fitted Gaussian, and a preview perturbed ensemble.

Called by notebooks/perturbations_from_icon.py when run with --check; not a standalone CLI.
"""
import os
import sys
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# this file lives at <repo>/notebooks/; add src/ so assimilator imports resolve
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from assimilator.functions import VARIABLES
from assimilator.perturbate import _simulate_ar1

logger = logging.getLogger(__name__)

_VAR_LABELS   = {"T_2M": "T_2M [K]", "U": "U [m/s]", "V": "V [m/s]", "GLOB": "GLOB [W/m²]"}
_PERTURB_VARS = ["U", "V", "GLOB"]   # vars perturbed downstream (residual = ICON - control)


def _acf(x: np.ndarray, nlags: int) -> np.ndarray:
    """Sample autocorrelation of x for lags 0..nlags."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)] - np.nanmean(x)
    n = len(x)
    var = np.dot(x, x)
    return np.array([np.dot(x[: n - k], x[k:]) / var for k in range(nlags + 1)])


def _acquisition_check(args: dict, flat_df, mean_df, contours: dict) -> str:
    """Grid-point selection map + lake-mean time series -> ensemble_base/check.png."""
    lake     = args.get("reanalysis_lake", args["lake"])
    base_dir = args["ensemble_base"]
    os.makedirs(base_dir, exist_ok=True)
    out_path = os.path.join(base_dir, "check.png")

    feature = (contours or {}).get(lake)
    if feature is None:
        raise ValueError(f"No contour for {lake}")
    df   = flat_df
    mean = mean_df.copy()
    if "time" in mean.columns and mean["time"].dtype == object:
        mean["time"] = pd.to_datetime(mean["time"])
    gdf = gpd.GeoDataFrame.from_features([feature], crs="EPSG:4326")

    unique_pts = df[["lat", "lon"]].drop_duplicates()
    grid_gdf = gpd.GeoDataFrame(
        unique_pts, geometry=gpd.points_from_xy(unique_pts["lon"], unique_pts["lat"]), crs="EPSG:4326")
    polygon = gdf.unary_union
    inside  = grid_gdf[ grid_gdf.within(polygon)]
    outside = grid_gdf[~grid_gdf.within(polygon)]

    vars_present = [v for v in VARIABLES if v in mean.columns]
    n_vars = len(vars_present)

    fig = plt.figure(figsize=(14, 3 + 2.5 * n_vars))
    gs  = gridspec.GridSpec(n_vars, 2, figure=fig, width_ratios=[1, 2], hspace=0.5, wspace=0.35)
    ax_map = fig.add_subplot(gs[:, 0])
    gdf.plot(ax=ax_map, facecolor="lightskyblue", edgecolor="steelblue", linewidth=1.2, alpha=0.4)
    ax_map.scatter(outside["lon"], outside["lat"], s=18, color="lightgrey", label=f"outside ({len(outside)})", zorder=2)
    ax_map.scatter(inside["lon"],  inside["lat"],  s=22, color="steelblue", label=f"inside  ({len(inside)})",  zorder=3)
    ax_map.set_title(f"{lake} — grid point selection", fontsize=10)
    ax_map.set_xlabel("lon"); ax_map.set_ylabel("lat"); ax_map.legend(fontsize=8)

    for row, var in enumerate(vars_present):
        ax = fig.add_subplot(gs[row, 1])
        ax.plot(mean["time"], mean[var], lw=0.8, color="steelblue")
        ax.set_ylabel(_VAR_LABELS.get(var, var), fontsize=8)
        ax.tick_params(labelsize=7)
        if row == 0:
            ax.set_title(f"{lake} — lake mean time series", fontsize=10)
        if row < n_vars - 1:
            ax.set_xticklabels([])

    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"{lake}: acquisition check -> {out_path}")
    return out_path


def _fit_check(args: dict, fit_df, params: dict) -> str:
    """Per perturbed variable: residual ACF (vs fitted phi^k), residual distribution
    (vs fitted Gaussian), and a preview perturbed ensemble over the first 200 steps.
    Writes ensemble_base/check_fit.png."""
    lake      = args.get("reanalysis_lake", args["lake"])
    base_dir  = args["ensemble_base"]
    os.makedirs(base_dir, exist_ok=True)
    out_path  = os.path.join(base_dir, "check_fit.png")

    n_prev    = int(min(200, len(fit_df)))
    n_members = int(args.get("n_members", 20))
    rng       = np.random.default_rng(args.get("rng_seed", 42))

    nv  = len(_PERTURB_VARS)
    fig = plt.figure(figsize=(15, 3.2 * nv))
    gs  = gridspec.GridSpec(nv, 3, figure=fig, hspace=0.5, wspace=0.3)

    for r, var in enumerate(_PERTURB_VARS):
        phi   = params[var]["phi"]
        sigma = params[var]["sigma"]
        resid = (fit_df[f"{var}_icon"] - fit_df[var]).dropna().values
        std_r = float(resid.std())

        # --- residual ACF vs fitted AR(1) ---
        nlags = 48
        lags  = np.arange(nlags + 1)
        ax = fig.add_subplot(gs[r, 0])
        ax.bar(lags, _acf(resid, nlags), width=0.8, color="steelblue", alpha=0.7, label="residual ACF")
        ax.plot(lags, phi ** lags, color="crimson", lw=1.5, label=f"AR(1) φ$^k$ (φ={phi:.2f})")
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_xlabel("lag [h]", fontsize=8); ax.set_ylabel("ACF", fontsize=8)
        ax.set_title(f"{var} — residual ACF", fontsize=10)
        ax.legend(fontsize=7); ax.tick_params(labelsize=7)

        # --- residual distribution vs fitted Gaussian ---
        ax = fig.add_subplot(gs[r, 1])
        ax.hist(resid, bins=60, density=True, color="steelblue", alpha=0.6)
        xs = np.linspace(resid.min(), resid.max(), 200)
        ax.plot(xs, np.exp(-xs ** 2 / (2 * std_r ** 2)) / (std_r * np.sqrt(2 * np.pi)),
                color="crimson", lw=1.5, label=f"N(0, {std_r:.2f})")
        ax.set_xlabel(f"residual {_VAR_LABELS.get(var, var)}", fontsize=8); ax.set_ylabel("density", fontsize=8)
        ax.set_title(f"{var} — residual distribution", fontsize=10)
        ax.legend(fontsize=7); ax.tick_params(labelsize=7)

        # --- preview perturbed ensemble (first n_prev steps) ---
        ax   = fig.add_subplot(gs[r, 2])
        base = fit_df[var].values[:n_prev].astype(float)
        pert = _simulate_ar1(phi, sigma, n_prev, n_members, rng)
        clip = var == "GLOB"
        if clip:
            night = base < 1.0
            pert[night] = 0.0
        ens = base[:, None] + pert
        if clip:
            ens[night, :] = 0.0
            ens = np.clip(ens, 0.0, None)
        steps = np.arange(n_prev)
        ax.plot(steps, ens, color="steelblue", lw=0.3, alpha=0.35)
        ax.plot(steps, base, color="black", lw=1.2, label="control")
        ax.set_xlabel("step [h]", fontsize=8); ax.set_ylabel(_VAR_LABELS.get(var, var), fontsize=8)
        ax.set_title(f"{var} — perturbed ensemble (first {n_prev}, {n_members} members)", fontsize=10)
        ax.legend(fontsize=7); ax.tick_params(labelsize=7)

    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"{lake}: fit check -> {out_path}")
    return out_path


def check(args: dict, flat_df=None, mean_df=None, contours=None, fit_df=None, params=None):
    """QA plots. Acquisition QA (grid mask + lake-mean series -> check.png) whenever the
    acquisition data is given; fit QA (residual ACF / distribution / preview ensemble ->
    check_fit.png) whenever fit_df + params are given (from perturbations_from_icon)."""
    paths = []
    if flat_df is not None and mean_df is not None and contours is not None:
        paths.append(_acquisition_check(args, flat_df, mean_df, contours))
    if fit_df is not None and params is not None:
        paths.append(_fit_check(args, fit_df, params))
    return paths

"""Apply AR(1) forcing perturbations from perturbations/<lake>.json.

Reads the fitted (phi, sigma) per variable, simulates fresh AR(1) noise for each
ensemble member, adds it to the control Forcing.dat, and writes the perturbed
Forcing.dat into ensemble1..N. Light, runs every pipeline pass — numpy/pandas + the
committed JSON only (no ICON). This is what main.py runs as step 3 (perturbator):

    python src/assimilator/perturbate.py args/run_enkf.json

Fitting the AR(1) stats from ICON (the heavy, once-per-lake step that produces the
JSON) lives in notebooks/perturbations_from_icon.py.
"""
import os
import sys
import json
import logging
import argparse

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # put src/ on the path
from assimilator.functions import (
    ROOT, verify_args, resolve_src, resolve_root, to_utc, merge_lake_args,
)
from assimilator.models.simstrat import SIMSTRAT_REF_YEAR, FORCING_HEADER

logger = logging.getLogger(__name__)

# variable -> (control Forcing.dat column, clip-to-zero at night?). The channels
# perturbed; T/vap/cloud/rain pass through unperturbed.
PERTURB_VARS = {"U": ("U_std", False), "V": ("V_std", False), "GLOB": ("GLOB_std", True)}


# Note: AR(1) cold-start. out[0]=0 for all members -> zero forcing spread at t=0,
# suppressed for the first ~1/(1-phi) steps (and noise[0] is generated but unused). Negligible
# in practice. Intended.
def _simulate_ar1(phi: float, sigma: float, n: int, n_members: int, rng: np.random.Generator) -> np.ndarray:
    noise = rng.standard_normal((n, n_members)) * sigma
    out   = np.zeros((n, n_members))
    for t in range(1, n):
        out[t] = phi * out[t - 1] + noise[t]
    return out


def perturbations_path(args: dict) -> str:
    """AR(1) calibration JSON for the run: the 'perturbations_file' override (path relative to the
    repo root, or absolute) if given, else perturbations/<lake>.json (under perturbations_dir)."""
    override = args.get("perturbations_file")
    if override:
        return resolve_root(override)
    perturb_dir = args.get("perturbations_dir", os.path.join(ROOT, "perturbations"))
    return os.path.join(perturb_dir, f"{args['lake']}.json")


def load_perturbations(args: dict) -> dict:
    """Load and validate the AR(1) calibration for the lake. Errors if the file is missing
    or its content is malformed (each of U/V/GLOB needs phi + sigma)."""
    json_path = perturbations_path(args)
    if not os.path.isfile(json_path):
        raise FileNotFoundError(
            f"{json_path} not found — fit it with notebooks/perturbations_from_icon.py "
            f"(needs the ICON API / EAWAG VPN).")
    with open(json_path, encoding="utf-8") as f:
        params = json.load(f)

    variables = params.get("variables")
    if not isinstance(variables, dict):
        raise ValueError(f"{json_path}: malformed calibration — missing 'variables' object")
    bad = [v for v in PERTURB_VARS
           if not (isinstance(variables.get(v), dict) and {"phi", "sigma"} <= variables[v].keys())]
    if bad:
        raise ValueError(f"{json_path}: malformed calibration — {bad} each need 'phi' and 'sigma'")
    return params


def perturbate(args: dict, params: dict = None) -> None:
    lake                 = args["lake"]
    model_inputs_path = args["model_inputs_path"]
    ensemble_base        = args["ensemble_base"]
    n_members            = args["n_members"]
    rng_seed             = args.get("rng_seed", 42)
    sigma_scale          = args.get("sigma_scale", 1.0)
    ref_year             = args.get("ref_year", SIMSTRAT_REF_YEAR)

    if params is None:
        params = load_perturbations(args)
    variables = params["variables"]

    # Control Forcing.dat as the base signal, over the assimilation window
    t0  = pd.Timestamp(f"{ref_year}-01-01")
    std = pd.read_csv(
        os.path.join(model_inputs_path, "Forcing.dat"),
        sep=r"\s+",
        names=["time_days", "U_std", "V_std", "T_std", "GLOB_std", "vap_std", "cloud_std", "rain_std"],
        skiprows=1,
    )
    # Forcing.dat time_days is 0-based (day 0 = ref_year Jan 1), the SAME axis as the par's
    # "Start d" and T_out's "Datetime" (load_T). Verified: file spans time_days 0..16435 =
    # 1981-01-01..2025-12-31. (Was off by one — a `- 1` here treated it as 1-based, shifting the
    # perturbation window +1 day and dropping the first assimilation day.)
    std["time"] = (t0 + pd.to_timedelta(std["time_days"], unit="D")).dt.round("h").dt.tz_localize("UTC")
    start = pd.Timestamp(args["start_date"]).tz_convert("UTC")
    end   = pd.Timestamp(args["end_date"]).tz_convert("UTC")
    df    = std[(std["time"] >= start) & (std["time"] <= end)].reset_index(drop=True)
    if df.empty:
        raise ValueError(f"Control Forcing.dat has no rows in [{start}, {end}] for {lake}")
    n = len(df)

    # Note: forcing perturbation is seeded (rng_seed) -> identical ensemble forcing every
    # run, while the EnKF obs-perturbation rng (enkf.py) is unseeded. Mixed reproducibility...
    # need to choose but for now not essential. Acknowledged.
    rng   = np.random.default_rng(rng_seed)
    night = df["GLOB_std"].values < 1.0   # night mask from the control's solar (no ICON at apply time)

    # Note: U and V are perturbed with independent AR(1) draws (separate phi/sigma), so
    # their cross-correlation is ignored. Intended.
    perturbed = {}
    for name, (std_col, clip_zero) in PERTURB_VARS.items():
        p    = variables[name]
        pert = _simulate_ar1(p["phi"], p["sigma"] * sigma_scale, n, n_members, rng)
        if clip_zero:
            pert[night] = 0.0
        ensemble = df[std_col].values[:, None] + pert
        if clip_zero:
            ensemble[night, :] = 0.0
            # Note: GLOB clipped at 0 only (no upper bound) -> a member's solar can be
            # perturbed above the physical clear-sky maximum. Acknowledged.
            ensemble = np.clip(ensemble, 0.0, None)
        perturbed[name] = ensemble

    spreads = {name: arr.std(axis=1).mean() for name, arr in perturbed.items()}
    logger.info(f"{lake}: mean ensemble spread — " + ", ".join(f"{k}={v:.3f}" for k, v in spreads.items()))

    # Overwrite Forcing.dat in each member (ensemble1..N); ensemble0 is the unperturbed control.
    for i in range(n_members):
        member_dir = os.path.join(ensemble_base, f"ensemble{i + 1}")
        if not os.path.isdir(member_dir):
            raise FileNotFoundError(
                f"{member_dir} missing — run the copy step (main.py) before perturbate")
        rows = np.column_stack([
            df["time_days"].values,
            perturbed["U"][:, i],
            perturbed["V"][:, i],
            df["T_std"].values,            # temperature: unperturbed
            perturbed["GLOB"][:, i],
            df["vap_std"].fillna(0).values,
            df["cloud_std"].fillna(0).values,
            df["rain_std"].fillna(0).values,
        ])
        np.savetxt(
            os.path.join(member_dir, "Forcing.dat"),
            rows, fmt="%10.4f", header=FORCING_HEADER, comments="",
        )

    logger.info(f"{lake}: perturbed Forcing.dat written to ensemble1..{n_members} -> {ensemble_base}")


# ---------------------------------------------------------------------------
# CLI wrapper — what main.py runs as step 3 (no ICON access)
# ---------------------------------------------------------------------------

REQUIRED = ["lake", "n_members", "ensemble_base", "start_date", "end_date"]


def build_args(raw: dict) -> dict:
    args = dict(raw)
    ensemble_base = resolve_src(args["ensemble_base"])
    args["ensemble_base"] = ensemble_base
    args.setdefault("model_inputs_path", os.path.join(ROOT, "inputs", args["lake"]))
    args.setdefault("perturbations_dir",    os.path.join(ROOT, "perturbations"))
    args.setdefault("rng_seed",    42)
    args.setdefault("sigma_scale", 1.0)

    args["start_date"] = to_utc(args["start_date"])
    args["end_date"]   = to_utc(args["end_date"])
    return args


def perturbator(raw_args: dict, params: dict = None) -> None:
    verify_args(raw_args, REQUIRED)
    perturbate(build_args(raw_args), params=params)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply AR(1) forcing perturbations")
    parser.add_argument("arg_file", help="Path to a run config JSON (e.g. args/run_enkf.json)")
    parser.add_argument("--lake", default=None, help="Lake to apply from the config's \"lakes\" block")
    cli = parser.parse_args()

    arg_file = cli.arg_file if os.path.isfile(cli.arg_file) else os.path.join(ROOT, cli.arg_file)
    if not os.path.isfile(arg_file):
        raise ValueError(f"Args file not found: {cli.arg_file}")

    with open(arg_file) as f:
        perturbator(merge_lake_args(json.load(f), lake=cli.lake))

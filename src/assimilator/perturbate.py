"""Apply AR(1) forcing perturbations from perturbations/<lake>.json.

Reads the fitted (phi, sigma) per variable, simulates AR(1) noise for each ensemble
member, adds it to the control Forcing.dat, and writes the perturbed Forcing.dat into
ensemble1..N. Light, runs every pipeline pass — numpy/pandas + the committed JSON only
(no ICON). This is what main.py runs as step 3 (perturbator):

    python src/assimilator/perturbate.py args/run_enkf.json

Operational reproducibility: the AR(1) noise for a given forcing row is keyed by that
row's ABSOLUTE time (functions.time_keyed_rng), not by its position in the window, and
the recursion state at the end of each window is persisted per member/variable in
<ensemble_base>/perturbation_state.json. A run continued in daily slices therefore
reproduces the exact perturbation series of one continuous run over the same period —
the next slice resumes the AR(1) chain instead of restarting it from zero. Cold start
(perturbation 0 at the first row) happens only on a fresh run dir, on "reset": true,
or when the perturbation parameters change.

Fitting the AR(1) stats from ICON (the heavy, once-per-lake step that produces the
JSON) lives in notebooks/generate_perturbation.py.
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
    ROOT, verify_args, resolve_src, resolve_root, to_utc, merge_lake_args, time_keyed_rng,
)
from assimilator.models.simstrat import SIMSTRAT_REF_YEAR, FORCING_HEADER

logger = logging.getLogger(__name__)

# variable -> (control Forcing.dat column, clip-to-zero at night?). The channels
# perturbed; T/vap/cloud/rain pass through unperturbed.
PERTURB_VARS = {"U": ("U_std", False), "V": ("V_std", False), "GLOB": ("GLOB_std", True)}

# AR(1) recursion state persisted across pipeline invocations (in ensemble_base), so an
# operational continuation resumes the exact perturbation chain of a continuous run. States are
# kept for the last STATE_KEEP window boundaries so re-running the same window (crash recovery,
# idempotent daily pipelines) still finds its resume point.
STATE_FILENAME = "perturbation_state.json"
STATE_KEEP     = 60


# QA-preview-only AR(1) (used by notebooks/check_perturbations.py for the fit plots).
# Production perturbation (perturbate below) instead keys each row's noise to its absolute time
# so operational continuations reproduce a continuous run; the process statistics are identical.
def _simulate_ar1(phi: float, sigma: float, n: int, n_members: int, rng: np.random.Generator) -> np.ndarray:
    noise = rng.standard_normal((n, n_members)) * sigma
    out   = np.zeros((n, n_members))
    for t in range(1, n):
        out[t] = phi * out[t - 1] + noise[t]
    return out


def _row_noise(rng_seed, var, step, n_members):
    """One row's AR(1) innovation for all members, keyed by the row's absolute time coordinate
    (`step` = seconds since the Simstrat epoch). Same (seed, var, step) -> same draw in every
    invocation, which is what makes the perturbation series chunk-invariant."""
    return time_keyed_rng(rng_seed, f"forcing_{var}", step).standard_normal(n_members)


# --- persisted AR(1) state --------------------------------------------------------------------

def _state_params_echo(variables, rng_seed, sigma_scale, n_members):
    """The parameter fingerprint stored with (and checked against) the persisted state: resuming
    a chain generated under different parameters would be inconsistent, so a mismatch cold-starts."""
    return {"rng_seed": rng_seed, "sigma_scale": sigma_scale, "n_members": n_members,
            "variables": {v: {"phi": variables[v]["phi"], "sigma": variables[v]["sigma"]}
                          for v in PERTURB_VARS}}


def _load_state(state_path, params_echo):
    """The persisted boundary states as {step:int -> {var -> [n_members values]}}; {} when the
    file is absent/unreadable or was written under different parameters (then we cold-start)."""
    if not os.path.isfile(state_path):
        return {}
    try:
        with open(state_path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        logger.warning(f"perturbation state unreadable ({e}) — cold-starting the AR(1) chain")
        return {}
    if data.get("params") != params_echo:
        logger.warning("perturbation state ignored (rng_seed/sigma_scale/n_members/phi/sigma "
                       "changed) — cold-starting the AR(1) chain")
        return {}
    return {int(k): v for k, v in data.get("states", {}).items()}


def _save_state(state_path, params_echo, states):
    """Persist the boundary states (pruned to the newest STATE_KEEP), atomically."""
    keep = dict(sorted(states.items())[-STATE_KEEP:])
    tmp  = state_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"params": params_echo,
                   "states": {str(k): v for k, v in keep.items()}}, f)
    os.replace(tmp, state_path)


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
            f"{json_path} not found — fit it with notebooks/generate_perturbation.py "
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
    # Absolute time coordinate per row (integer seconds since the Simstrat epoch): the key for the
    # row's noise draw and for the persisted boundary states. Derived from the file's time_days
    # text, so the same row keys identically in every invocation.
    steps_all = np.rint(std["time_days"].values * 86400).astype(np.int64)
    start = pd.Timestamp(args["start_date"]).tz_convert("UTC")
    end   = pd.Timestamp(args["end_date"]).tz_convert("UTC")
    mask  = (std["time"] >= start) & (std["time"] <= end)
    df    = std[mask].reset_index(drop=True)
    steps = steps_all[mask.values]
    if df.empty:
        raise ValueError(f"Control Forcing.dat has no rows in [{start}, {end}] for {lake}")
    n = len(df)

    # Persisted AR(1) state: resume from the newest stored boundary at or before this window's
    # first row; cold-start (perturbation 0 at the first row, like a fresh continuous run) when
    # there is none, on "reset", or when the parameters changed (_load_state).
    state_path  = os.path.join(ensemble_base, STATE_FILENAME)
    params_echo = _state_params_echo(variables, rng_seed, sigma_scale, n_members)
    if args.get("reset") and os.path.isfile(state_path):
        os.remove(state_path)
        logger.info(f"{lake}: reset — cleared {STATE_FILENAME} (AR(1) chain cold-starts)")
    states      = _load_state(state_path, params_echo)
    resume_step = max((s for s in states if s <= steps[0]), default=None)

    night = df["GLOB_std"].values < 1.0   # night mask from the control's solar (no ICON at apply time)

    # Note: U and V are perturbed with independent AR(1) draws (separate phi/sigma), so
    # their cross-correlation is ignored. Intended.
    perturbed = {}
    end_state = {}
    for name, (std_col, clip_zero) in PERTURB_VARS.items():
        p        = variables[name]
        phi, sig = p["phi"], p["sigma"] * sigma_scale
        pert     = np.empty((n, n_members))
        if resume_step is None:
            pert[0] = 0.0                              # cold start: first row unperturbed
        else:
            prev = np.asarray(states[resume_step][name], dtype=float)
            # Advance the chain over any grid rows between the stored boundary and this window's
            # first row (a continued run normally resumes exactly at the boundary, so this loop is
            # usually empty), then take the first row's value: at the shared boundary instant it IS
            # the stored state; past it, one more recursion step.
            for s_mid in steps_all[(steps_all > resume_step) & (steps_all < steps[0])]:
                prev = phi * prev + _row_noise(rng_seed, name, s_mid, n_members) * sig
            pert[0] = prev if resume_step == steps[0] \
                else phi * prev + _row_noise(rng_seed, name, steps[0], n_members) * sig
        for t in range(1, n):
            pert[t] = phi * pert[t - 1] + _row_noise(rng_seed, name, steps[t], n_members) * sig
        end_state[name] = pert[-1].tolist()            # raw chain value (pre night-clip)
        if clip_zero:
            pert[night] = 0.0
        ensemble = df[std_col].values[:, None] + pert
        if clip_zero:
            ensemble[night, :] = 0.0
            # Note: GLOB clipped at 0 only (no upper bound) -> a member's solar can be
            # perturbed above the physical clear-sky maximum. Acknowledged.
            ensemble = np.clip(ensemble, 0.0, None)
        perturbed[name] = ensemble

    states[int(steps[-1])] = end_state
    _save_state(state_path, params_echo, states)
    resume_str = "cold start" if resume_step is None else f"resumed from step {resume_step}"
    logger.info(f"{lake}: AR(1) chain {resume_str}; state saved at step {int(steps[-1])} "
                f"-> {STATE_FILENAME}")

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

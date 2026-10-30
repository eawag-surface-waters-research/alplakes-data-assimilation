"""Apply AR(1) forcing perturbations.

Reads the fitted (phi, sigma) per variable -- the run config's `perturbations` (a dict) or
`perturbations_file` (a JSON) -- simulates AR(1) noise for each ensemble member, adds it to the
control forcing, and writes the perturbed forcing into ensemble1..N. Forcing files are read and
written through the model (`read_forcing` / `write_forcing`), so nothing here knows a file format.
Light, runs every pipeline pass (step 3 of assimilator.pipeline.run).

Operational reproducibility: the AR(1) noise for a given forcing row is keyed by that
row's ABSOLUTE time (functions.time_keyed_rng), not by its position in the window, and
the recursion state at the end of each window is persisted per member/variable in
<ensemble_base>/perturbation_state.json. A run continued in daily slices therefore
reproduces the exact perturbation series of one continuous run over the same period —
the next slice resumes the AR(1) chain instead of restarting it from zero. Cold start
(perturbation 0 at the first row) happens only on a fresh run dir, on "reset": true,
or when the perturbation parameters change.

The AR(1) statistics themselves are fitted offline: assimilator.fitting.perturbation holds the
math, and the host supplies the data (reanalysis residuals, control forcing).
"""
import os
import json
import logging

import numpy as np
import pandas as pd

from .functions import verify_args, time_keyed_rng, display_path, to_utc

logger = logging.getLogger(__name__)

# variable -> clip-to-zero at night? The channels perturbed (named as the model's read_forcing
# names them); every other forcing column passes through unperturbed.
PERTURB_VARS = {"U": False, "V": False, "GLOB": True}

# AR(1) recursion state persisted across pipeline invocations (in ensemble_base), so an
# operational continuation resumes the exact perturbation chain of a continuous run. States are
# kept for the last STATE_KEEP window boundaries so re-running the same window (crash recovery,
# idempotent daily pipelines) still finds its resume point.
STATE_FILENAME = "perturbation_state.json"
STATE_KEEP     = 60


# QA-preview-only AR(1) (used by the host's perturbation-fit plots).
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
    (`step` = the model's absolute time key, see read_forcing). Same (seed, var, step) -> same draw in every
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


def validate_perturbations(params, source="perturbations"):
    """Normalise and validate an AR(1) calibration: {"variables": {U/V/GLOB: {phi, sigma}}, ...}.
    A flat {U/V/GLOB: {phi, sigma}} map is accepted and wrapped. Returns the normalised dict."""
    if not isinstance(params, dict):
        raise ValueError(f"{source}: expected an object, got {type(params).__name__}")
    if "variables" not in params and set(PERTURB_VARS) <= set(params):
        params = {"variables": {v: params[v] for v in PERTURB_VARS}}
    variables = params.get("variables")
    if not isinstance(variables, dict):
        raise ValueError(f"{source}: malformed calibration — missing 'variables' object")
    bad = [v for v in PERTURB_VARS
           if not (isinstance(variables.get(v), dict) and {"phi", "sigma"} <= variables[v].keys())]
    if bad:
        raise ValueError(f"{source}: malformed calibration — {bad} each need 'phi' and 'sigma'")
    return params


def load_perturbations(args: dict) -> dict:
    """The run's AR(1) calibration: args['perturbations'] (a dict) if given, else the JSON at
    args['perturbations_file']. Errors if neither is set, the file is missing, or the content is
    malformed (each of U/V/GLOB needs phi + sigma)."""
    if args.get("perturbations") is not None:
        return validate_perturbations(args["perturbations"], "config 'perturbations'")
    json_path = args.get("perturbations_file")
    if not json_path:
        raise ValueError("no forcing perturbation calibration: set 'perturbations' (a dict) or "
                         "'perturbations_file' (a JSON) in the run config")
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"{json_path} not found — fit the AR(1) calibration first "
                                f"(assimilator.fitting.perturbation)")
    with open(json_path, encoding="utf-8") as f:
        return validate_perturbations(json.load(f), json_path)


def perturbate(args: dict, model, params: dict = None) -> None:
    lake                 = args["lake"]
    model_inputs_path    = args["model_inputs_path"]
    ensemble_base        = args["ensemble_base"]
    n_members            = args["n_members"]
    rng_seed             = args.get("rng_seed", 42)
    sigma_scale          = args.get("sigma_scale", 1.0)

    if params is None:
        params = load_perturbations(args)
    variables = params["variables"]

    # Control forcing as the base signal, over the assimilation window. `step` is the row's
    # absolute time key (from the model): the key for the row's noise draw and for the persisted
    # boundary states, identical in every invocation.
    std       = model.read_forcing(model_inputs_path, args.get("ref_year"))
    steps_all = std["step"].to_numpy(dtype=np.int64)
    start = pd.Timestamp(args["start_date"]).tz_convert("UTC")
    end   = pd.Timestamp(args["end_date"]).tz_convert("UTC")
    mask  = (std["time"] >= start) & (std["time"] <= end)
    df    = std[mask].reset_index(drop=True)
    steps = steps_all[mask.values]
    if df.empty:
        raise ValueError(f"Control forcing has no rows in [{start}, {end}] for {lake}")
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

    night = df["GLOB"].values < 1.0   # night mask from the control's solar (no reanalysis at apply time)

    # Note: U and V are perturbed with independent AR(1) draws (separate phi/sigma), so
    # their cross-correlation is ignored. Intended.
    perturbed = {}
    end_state = {}
    for name, clip_zero in PERTURB_VARS.items():
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
        ensemble = df[name].values[:, None] + pert
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

    # Overwrite the forcing in each member (ensemble1..N); ensemble0 is the unperturbed control.
    for i in range(n_members):
        member_dir = os.path.join(ensemble_base, f"ensemble{i + 1}")
        if not os.path.isdir(member_dir):
            raise FileNotFoundError(
                f"{member_dir} missing — run the copy step before perturbate")
        member = df.copy()
        for name in PERTURB_VARS:
            member[name] = perturbed[name][:, i]
        model.write_forcing(member_dir, member)

    logger.info(f"{lake}: perturbed forcing written to ensemble1..{n_members} -> "
                f"{display_path(ensemble_base)}")


REQUIRED = ["lake", "n_members", "ensemble_base", "model_inputs_path", "start_date", "end_date"]


def perturbator(cfg: dict, model, params: dict = None) -> None:
    """Perturb the forcing of members 1..N for a run config (assimilator.config.build_config).
    start_date/end_date are ISO strings, read as UTC."""
    verify_args(cfg, REQUIRED)
    args = dict(cfg)
    args["start_date"] = to_utc(cfg["start_date"])
    args["end_date"]   = to_utc(cfg["end_date"])
    perturbate(args, model, params=params)

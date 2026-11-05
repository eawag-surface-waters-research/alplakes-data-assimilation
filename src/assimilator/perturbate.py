"""The ensemble's forcing: AR(1) noise on the control forcing's U, V and GLOB, one series per member.

The noise of each forcing row is keyed by the row's absolute time (assimilator.rng), and the AR(1)
state at each window end is saved in <ensemble_base>/perturbation_state.json, so a run continued
in slices perturbs exactly as one continuous run. The chain cold-starts (no perturbation at the
first row) on a fresh run, on "reset", or when the perturbation parameters change. The (phi,
sigma) per variable are fitted offline with assimilator.fitting.perturbation.
"""
import os
import json
import logging

import numpy as np
import pandas as pd

from .config import utc
from .report import display_path
from .rng import time_keyed_rng

logger = logging.getLogger(__name__)

# The perturbed channels (as the model's read_forcing names them) -> clipped to zero at night?
PERTURB_VARS = {"U": False, "V": False, "GLOB": True}

STATE_FILENAME = "perturbation_state.json"
# Window ends whose state is kept, so re-running a recent window still finds its resume point.
STATE_KEEP = 60


def _simulate_ar1(phi, sigma, n, n_members, rng):
    """A plain AR(1) series, for the fit tools' preview plots (the run uses time-keyed noise)."""
    noise = rng.standard_normal((n, n_members)) * sigma
    out   = np.zeros((n, n_members))
    for t in range(1, n):
        out[t] = phi * out[t - 1] + noise[t]
    return out


def _row_noise(rng_seed, var, step, n_members):
    return time_keyed_rng(rng_seed, f"forcing_{var}", step).standard_normal(n_members)


def _params_echo(variables, rng_seed, sigma_scale, n_members):
    """What the saved chain state was generated with: resuming it under other parameters would mix
    two chains, so a mismatch cold-starts."""
    return {"rng_seed": rng_seed, "sigma_scale": sigma_scale, "n_members": n_members,
            "variables": {v: {"phi": variables[v]["phi"], "sigma": variables[v]["sigma"]}
                          for v in PERTURB_VARS}}


def _load_state(state_path, params_echo):
    """{step: {variable: [one value per member]}}, empty for a cold start."""
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
    keep = dict(sorted(states.items())[-STATE_KEEP:])
    tmp  = state_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"params": params_echo, "states": {str(k): v for k, v in keep.items()}}, f)
    os.replace(tmp, state_path)


def validate_perturbations(params, source="perturbations"):
    """The calibration as {"variables": {U/V/GLOB: {phi, sigma}}, ...}; a flat {U/V/GLOB: ...}
    map is wrapped. Raises on a malformed one."""
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


def load_perturbations(cfg):
    """The run's calibration: cfg["perturbations"], else the JSON at cfg["perturbations_file"]."""
    if cfg.get("perturbations") is not None:
        return validate_perturbations(cfg["perturbations"], "config 'perturbations'")
    json_path = cfg.get("perturbations_file")
    if not json_path:
        raise ValueError("no forcing perturbation calibration: set 'perturbations' (a dict) or "
                         "'perturbations_file' (a JSON) in the run config")
    if not os.path.isfile(json_path):
        raise FileNotFoundError(f"{json_path} not found — fit the AR(1) calibration first "
                                f"(assimilator.fitting.perturbation)")
    with open(json_path, encoding="utf-8") as f:
        return validate_perturbations(json.load(f), json_path)


def perturbate(cfg, model, params=None):
    """Write perturbed forcing over (start_date, end_date] into members 1..N; member 0 keeps the
    control forcing."""
    lake, ensemble_base, n_members = cfg["lake"], cfg["ensemble_base"], cfg["n_members"]
    rng_seed    = cfg.get("rng_seed", 42)
    sigma_scale = cfg.get("sigma_scale", 1.0)
    variables   = (params or load_perturbations(cfg))["variables"]

    control   = model.read_forcing(cfg["model_inputs_path"], cfg.get("ref_year"))
    steps_all = control["step"].to_numpy(dtype=np.int64)
    start, end = pd.Timestamp(utc(cfg["start_date"])), pd.Timestamp(utc(cfg["end_date"]))
    # Also the row on each side of a boundary that falls between rows, so the model has forcing at
    # both ends.
    t  = control["time"]
    lo = t[t <= start].max() if (t <= start).any() else t.min()
    hi = t[t >= end].min() if (t >= end).any() else t.max()
    mask  = (t >= lo) & (t <= hi)
    df    = control[mask].reset_index(drop=True)
    steps = steps_all[mask.values]
    if df.empty:
        raise ValueError(f"Control forcing has no rows in [{start}, {end}] for {lake}")
    n = len(df)

    state_path  = os.path.join(ensemble_base, STATE_FILENAME)
    params_echo = _params_echo(variables, rng_seed, sigma_scale, n_members)
    if cfg.get("reset") and os.path.isfile(state_path):
        os.remove(state_path)
        logger.info(f"{lake}: reset — cleared {STATE_FILENAME} (AR(1) chain cold-starts)")
    states      = _load_state(state_path, params_echo)
    resume_step = max((s for s in states if s <= steps[0]), default=None)
    night       = df["GLOB"].values < 1.0
    # Saved at the last row at or before the end: where the next window resumes.
    before_end = (df["time"] <= end).to_numpy()
    i_end      = int(np.flatnonzero(before_end)[-1]) if before_end.any() else n - 1

    # U and V get independent chains: their cross-correlation is ignored, deliberately.
    perturbed, end_state = {}, {}
    for name, clip_zero in PERTURB_VARS.items():
        phi, sig = variables[name]["phi"], variables[name]["sigma"] * sigma_scale
        pert = np.empty((n, n_members))
        if resume_step is None:
            pert[0] = 0.0
        else:
            prev = np.asarray(states[resume_step][name], dtype=float)
            # Advance from the saved state to the first row: nothing to do when it is that row.
            for s_mid in steps_all[(steps_all > resume_step) & (steps_all < steps[0])]:
                prev = phi * prev + _row_noise(rng_seed, name, s_mid, n_members) * sig
            pert[0] = prev if resume_step == steps[0] \
                else phi * prev + _row_noise(rng_seed, name, steps[0], n_members) * sig
        for i in range(1, n):
            pert[i] = phi * pert[i - 1] + _row_noise(rng_seed, name, steps[i], n_members) * sig
        end_state[name] = pert[i_end].tolist()         # the chain itself, before the night clip
        if clip_zero:
            pert[night] = 0.0
        ensemble = df[name].values[:, None] + pert
        if clip_zero:
            # Clipped at zero only: a member's radiation can exceed clear sky. Accepted.
            ensemble[night, :] = 0.0
            ensemble = np.clip(ensemble, 0.0, None)
        perturbed[name] = ensemble

    if cfg.get("center_perturbations"):
        # Wind speed is not linear in the zero-mean U/V noise, so the members are windier than the
        # control. Rescale each row so the ensemble-mean speed (and mean GLOB) equals the control's.
        # The chain state is untouched, so continuations still reproduce.
        speed = np.hypot(perturbed["U"], perturbed["V"])
        ctrl  = np.hypot(df["U"].values, df["V"].values)
        mean  = speed.mean(axis=1)
        k     = np.where((mean > 0) & (ctrl > 0), ctrl / np.where(mean > 0, mean, 1.0), 1.0)
        perturbed["U"] = perturbed["U"] * k[:, None]
        perturbed["V"] = perturbed["V"] * k[:, None]
        gmean = perturbed["GLOB"].mean(axis=1)
        kg    = np.where((gmean > 0) & (df["GLOB"].values > 0),
                         df["GLOB"].values / np.where(gmean > 0, gmean, 1.0), 1.0)
        perturbed["GLOB"] = perturbed["GLOB"] * kg[:, None]
        logger.info(f"{lake}: center_perturbations — mean wind speed {ctrl.mean():.3f} (control) vs "
                    f"{np.hypot(perturbed['U'], perturbed['V']).mean():.3f} (ensemble, was {speed.mean():.3f})")

    states[int(steps[i_end])] = end_state
    _save_state(state_path, params_echo, states)
    logger.info(f"{lake}: AR(1) chain {'cold start' if resume_step is None else f'resumed from step {resume_step}'}; "
                f"state saved at step {int(steps[i_end])} -> {STATE_FILENAME}")
    logger.info(f"{lake}: mean ensemble spread — " +
                ", ".join(f"{k}={v.std(axis=1).mean():.3f}" for k, v in perturbed.items()))

    for i in range(1, n_members + 1):
        member_dir = os.path.join(ensemble_base, f"ensemble{i}")
        if not os.path.isdir(member_dir):
            raise FileNotFoundError(f"{member_dir} missing — copy the model inputs into the members first")
        member = df.copy()
        for name in PERTURB_VARS:
            member[name] = perturbed[name][:, i - 1]
        model.write_forcing(member_dir, member)
    logger.info(f"{lake}: perturbed forcing written to ensemble1..{n_members} -> {display_path(ensemble_base)}")

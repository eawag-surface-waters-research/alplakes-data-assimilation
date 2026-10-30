"""The run config: one flat dict describing one assimilation run.

    from assimilator.config import build_config
    cfg = build_config({"lake": "upperlugano", "algorithm": "EnKF", "n_members": 20,
                        "start_date": "2025-01-01", "end_date": "2025-12-31",
                        "model_inputs_path": "/data/upperlugano/inputs",
                        "ensemble_base": "/data/upperlugano/da_run",
                        "obs_file": "/data/upperlugano/obs.csv",
                        "perturbations": {"variables": {"U": {...}, "V": {...}, "GLOB": {...}}}})

`build_config` fills DEFAULTS, derives the engine's file names, makes every path absolute (relative
paths resolve against the CURRENT working directory -- never against this package) and validates
the result. It is idempotent, and returns a new dict. Keys starting with "_" are comments and are
dropped. docs/configuration.md is the full reference.
"""
import os
from datetime import datetime

# Optional keys and their values when absent.
DEFAULTS = {
    "engine":        "python",    # "python" (native) or "openda"
    "model":         "simstrat",  # forward model, see assimilator.models
    "inflation":     1.0,         # native EnKF anomaly inflation; 1.0 = off
    "rng_seed":      42,          # seeds forcing noise + EnKF obs perturbations (time-keyed)
    "sigma_scale":   1.0,         # multiplies every AR(1) noise sigma
    "sigma_default": 0.5,         # observation error (degC) when the run names no sigma_obs
    "window_mode":   "obs",       # native: one window per observation time ("daily" = legacy)
    "obs_operator":  "linear",    # native EnKF: "linear" depth interpolation or legacy "nearest"
    "reset":         False,       # True = start from scratch (clears state + perturbation chain)
    "progress":      False,       # progress bar
}

REQUIRED = ["lake", "algorithm", "n_members", "start_date", "end_date",
            "model_inputs_path", "ensemble_base", "obs_file"]

# Keys holding filesystem paths, made absolute.
PATH_KEYS = ["model_inputs_path", "ensemble_base", "obs_file", "perturbations_file",
             "openda_dir", "openda_bin"]

ALGORITHMS = {"python": ("EnKF", "PF"), "openda": ("EnKF", "DEnKF", "EnSR", "PF")}


def _abspath(path):
    return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))


def _iso(name, value):
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        raise ValueError(f"'{name}' must be an ISO date (YYYY-MM-DD[THH:MM]), got {value!r}")


def build_config(cfg):
    """Validated, defaulted, absolute-path copy of a run config. Raises ValueError on a problem."""
    out = {k: v for k, v in dict(cfg).items() if not str(k).startswith("_")}
    if "lakes" in out:
        raise ValueError("run configs are flat: one run per config. A \"lakes\" map is no longer "
                         "supported -- put the lake's keys at the top level")
    for k, v in DEFAULTS.items():
        out.setdefault(k, v)

    missing = [k for k in REQUIRED if out.get(k) in (None, "")]
    if missing:
        raise ValueError(f"run config is missing required key(s): {missing}")
    if out.get("perturbations") is None and not out.get("perturbations_file"):
        raise ValueError("run config needs the forcing perturbation calibration: 'perturbations' "
                         "(a dict) or 'perturbations_file' (a JSON path)")

    engine = out["engine"]
    if engine not in ALGORITHMS:
        raise ValueError(f"unknown engine '{engine}'; choose from {sorted(ALGORITHMS)}")
    if out["algorithm"] not in ALGORITHMS[engine]:
        raise ValueError(f"unknown algorithm '{out['algorithm']}' for engine '{engine}'; "
                         f"choose from {list(ALGORITHMS[engine])}")
    if not isinstance(out["n_members"], int) or out["n_members"] < 2:
        raise ValueError(f"'n_members' must be an integer >= 2, got {out['n_members']!r}")
    if _iso("start_date", out["start_date"]) >= _iso("end_date", out["end_date"]):
        raise ValueError(f"start_date {out['start_date']} is not before end_date {out['end_date']}")

    for k in PATH_KEYS:
        if out.get(k):
            out[k] = _abspath(out[k])

    if engine == "python":
        # Per-algorithm output folder and par copy inside each member, so an EnKF and a PF run can
        # share members without clobbering each other.
        out.setdefault("results_dir", f"Results_{out['algorithm']}")
        out.setdefault("par_file", f"Settings_{out['algorithm']}.par")
    else:
        out.setdefault("openda_dir", os.path.join(
            os.path.dirname(out["ensemble_base"]),
            f"openda_{out['model']}_{out['lake']}_{out['algorithm'].lower()}"))
    return out

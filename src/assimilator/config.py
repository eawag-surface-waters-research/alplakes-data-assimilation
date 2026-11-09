"""The run config: one flat dict describing one assimilation run.

    from assimilator.config import build_config
    cfg = build_config({"lake": "upperlugano", "algorithm": "EnKF", "n_members": 20,
                        "start_date": "2025-01-01", "end_date": "2025-12-31",
                        "model_inputs_path": "/data/upperlugano/inputs",
                        "ensemble_base": "/data/upperlugano/da_run",
                        "obs_file": "/data/upperlugano/obs.csv",
                        "perturbations": {"variables": {"U": {...}, "V": {...}, "GLOB": {...}}}})

build_config fills the defaults, parses the dates, makes the paths absolute (against the current
working directory, never this package) and validates. It is idempotent. Keys starting with "_" are
comments and are dropped. docs/configuration.md is the full reference.
"""
import os
import logging
from datetime import date, datetime, timezone

logger = logging.getLogger(__name__)

DEFAULTS = {
    "engine":        "python",    # "python" (native) or "openda"
    "model":         "simstrat",
    "inflation":     1.0,         # native EnKF; 1.0 = off
    "rng_seed":      42,
    "sigma_scale":   1.0,         # multiplies every forcing perturbation sigma
    "center_perturbations": False,  # rescale rows so ensemble-mean wind speed and GLOB = control
    "sigma_default": 0.5,         # observation error (degC) when the run names no sigma_obs
    "reset":         False,       # start from scratch: clears the members' state and the perturbation chain
    "progress":      False,
}

REQUIRED = ["lake", "algorithm", "n_members", "start_date", "end_date",
            "model_inputs_path", "ensemble_base", "obs_file"]

PATH_KEYS = ["model_inputs_path", "ensemble_base", "obs_file", "perturbations_file",
             "openda_dir", "openda_bin"]

ALGORITHMS = {"python": ("EnKF", "PF"), "openda": ("EnKF", "DEnKF", "EnSR", "PF")}


def utc(t):
    """An instant as a tz-aware UTC datetime: an ISO string, a date (its 00:00), a datetime or a
    pandas Timestamp. Naive values are UTC."""
    if isinstance(t, str):
        t = datetime.fromisoformat(t)
    if isinstance(t, date) and not isinstance(t, datetime):
        t = datetime(t.year, t.month, t.day)
    if hasattr(t, "to_pydatetime"):
        t = t.to_pydatetime()
    return t.replace(tzinfo=timezone.utc) if t.tzinfo is None else t.astimezone(timezone.utc)


def build_config(cfg):
    """Validated, defaulted copy of a run config, with absolute paths and UTC datetime dates.
    Raises ValueError on a problem."""
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
    for k in ("start_date", "end_date"):
        try:
            out[k] = utc(out[k])
        except (TypeError, ValueError):
            raise ValueError(f"'{k}' must be an ISO date (YYYY-MM-DD[THH:MM]), got {out[k]!r}")
    if out["start_date"] >= out["end_date"]:
        raise ValueError(f"start_date {out['start_date']} is not before end_date {out['end_date']}")

    for k in PATH_KEYS:
        if out.get(k):
            out[k] = os.path.abspath(os.path.expanduser(os.path.expandvars(out[k])))

    if out.get("online_errors") is not None:
        from .online_errors import validate_online_errors
        out["online_errors"] = validate_online_errors(out["online_errors"], engine, out["algorithm"])
    if out.get("localization") is not None:
        from .localization import validate_localization
        out["localization"] = validate_localization(out["localization"], engine, out["algorithm"])

    if engine == "python":
        # Per-algorithm outputs and par copy in each member, so an EnKF and a PF run can share members.
        out.setdefault("results_dir", f"Results_{out['algorithm']}")
        out.setdefault("par_file", f"Settings_{out['algorithm']}.par")
    else:
        out.setdefault("openda_dir", os.path.join(
            os.path.dirname(out["ensemble_base"]),
            f"openda_{out['model']}_{out['lake']}_{out['algorithm'].lower()}"))
    return out


def resolve_max_workers(cfg, n_members):
    """How many members to run at once: all of them unless the config's max_workers caps it.

    Not capped to the CPU count by default: measured on 8 cores, 21 concurrent members finished a
    window ~14% faster than 8, because the per-member snapshot I/O overlaps. Set max_workers only
    to throttle CPU or memory."""
    requested = cfg.get("max_workers")
    workers   = max(1, min(int(requested), n_members)) if requested else n_members
    logger.info(f"parallelism: {workers} concurrent (cpu={os.cpu_count() or 1}, cap={n_members}, "
                f"{f'cfg max_workers={requested}' if requested else 'default=full'})")
    return workers

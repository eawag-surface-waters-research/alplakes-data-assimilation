"""Prepare an OpenDA working dir from the framework's own inputs, so OpenDA runs on exactly the
forcing, starting state and observations the native engines use:

    <model inputs>/*                 -> stochModel/template/   (the model's template)
    <ensemble_base>/ensemble<i>/     -> forcings/              (member i's perturbed forcing)
    the warm start                   -> stochModel/template/   (starting state and state file)
    the run's obs_file               -> stochObserver/T_<depth>m[_<season>]_real.csv

Everything model-specific comes from the model's OpenDA binding (binding.py).
"""
import os
import shutil
import logging

from ..config import utc
from ..observations import load_obs
from ..report import display_path
from ..sigma import SEASONS, season_of
from .config import format_day

logger = logging.getLogger(__name__)

# A restart chain continuing from a saved ensemble: the marker tells the wrapper every member must
# have been restored, and the sentinel fills the template state so one that was not is caught.
# Every wrapper keeps its own copy of both (binding.py).
RESTART_MARKER   = "RESTART_EXPECTED"
RESTART_SENTINEL = -999.0

# Restart chain only: the binding's long read-only inputs are kept once in <openda_dir>/shared
# rather than in the template OpenDA clones into every work dir (<openda_dir>/Results/work<k>,
# hence two levels up), and the template's control forcing is cut to the run.
SHARED_DIR = "shared"
SHARED_REL = "../../" + SHARED_DIR


def openda_binding(model):
    binding = model.openda_binding()
    if binding is None:
        raise ValueError(f"model '{model.name}' has no OpenDA binding: it cannot run with engine 'openda'")
    return binding


def _copy(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)


def _copy_path(src, dst):
    if os.path.isdir(src):
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
    else:
        _copy(src, dst)


def _unchanged(src, dst):
    return (os.path.isfile(dst) and os.path.getsize(src) == os.path.getsize(dst)
            and abs(os.path.getmtime(src) - os.path.getmtime(dst)) < 1e-6)


def _sync_shared(model_inputs, shared_dir, names):
    """Copy the shared inputs `names` that changed. Returns how many files were copied."""
    copied = 0
    for name in names:
        src = os.path.join(model_inputs, name)
        if not os.path.exists(src):
            continue
        pairs = [(src, os.path.join(shared_dir, name))] if os.path.isfile(src) else \
                [(os.path.join(src, f), os.path.join(shared_dir, name, f))
                 for f in sorted(os.listdir(src)) if os.path.isfile(os.path.join(src, f))]
        for a, b in pairs:
            if not _unchanged(a, b):
                _copy(a, b)
                copied += 1
    return copied


def build_observations(cfg, openda_dir, model_inputs, binding, season_split=False):
    """Write stochObserver/T_<depth>m[_<season>]_real.csv: every observation in (start_date,
    end_date] at its own hour, binned as the native engines do, its time in model days. Depths
    with data on fewer than cfg["obs_min_days"] days, or that the model does not output, are left
    out. Returns (depths, number of observation times, [(depth, season or None)] written)."""
    stoch_dir = os.path.join(openda_dir, "stochObserver")
    if not os.path.isfile(cfg["obs_file"]):
        raise FileNotFoundError(f"observation source not found: {cfg['obs_file']} (set 'obs_file')")
    obs = load_obs(cfg["obs_file"]).dropna(subset=["value"])
    obs = obs[(obs["time"] > utc(cfg["start_date"])) & (obs["time"] <= utc(cfg["end_date"]))]

    days   = obs.groupby("depth")["time"].agg(lambda t: t.dt.floor("D").nunique())
    depths = sorted(float(d) for d, n in days.items() if n >= cfg.get("obs_min_days", 1))
    model_depths = binding.output_depths(model_inputs)
    if model_depths:
        matched = [d for d in depths if any(abs(d - m) <= 1e-6 for m in model_depths)]
        if len(matched) < len(depths):
            logger.warning(f"[adapter] dropping obs depths with no matching model output depth: "
                           f"{[f'{d:g}' for d in depths if d not in matched]} m")
        depths = matched
    obs = obs[obs["depth"].isin(depths)]
    logger.info(f"[adapter] observations: {display_path(cfg['obs_file'])} -> stochObserver/  "
                f"({len(depths)} depths {[f'{d:g}' for d in depths]}, {obs['time'].nunique()} observation "
                f"times{', season-split' if season_split else ''})")

    # One file per series as config.series_specs declares them: OpenDA dies on a declared series
    # with no file.
    os.makedirs(stoch_dir, exist_ok=True)
    series_keys = []
    for depth in depths:
        rows = obs[obs["depth"] == depth].sort_values("time")
        season = rows["time"].dt.month.map(season_of) if season_split else None
        for s in (SEASONS if season_split else [None]):
            part = rows[season == s] if season_split else rows
            if part.empty:
                continue
            with open(os.path.join(stoch_dir, f"T_{depth:g}m{f'_{s}' if s else ''}_real.csv"), "w") as f:
                f.write("time,value\n")
                for t, v in zip(part["time"], part["value"]):
                    f.write(f"{format_day(binding.model_time(t))},{v:.6f}\n")
            series_keys.append((depth, s))
    return depths, int(obs["time"].nunique()), series_keys


def adapt(cfg, binding, season_split=False, restart_in=None):
    """Prepare cfg["openda_dir"] for one OpenDA run. `restart_in` is None for a plain run, "" for a
    restart chain's cold start, or the restart file the chain continues from. Returns
    build_observations' result."""
    openda_dir    = cfg["openda_dir"]
    model_inputs  = cfg["model_inputs_path"]
    template_dir  = os.path.join(openda_dir, "stochModel", "template")
    chain         = restart_in is not None
    if not os.path.isdir(model_inputs):
        raise FileNotFoundError(f"model_inputs not found: {model_inputs}")
    os.makedirs(template_dir, exist_ok=True)

    for src, rel in binding.wrapper_files():
        dst = os.path.join(openda_dir, "stochModel", rel)
        _copy(src, dst)
        if rel.startswith("bin" + os.sep):
            # OpenDA runs the wrapper directly: it must be executable whatever its mode in the repo.
            os.chmod(dst, os.stat(dst).st_mode | 0o111)

    for name in sorted(os.listdir(model_inputs)):
        if name in binding.coupling_files or not binding.is_template_input(name):
            continue
        if chain and (name in binding.shared_inputs or name == binding.forcing_file):
            continue
        _copy_path(os.path.join(model_inputs, name), os.path.join(template_dir, name))
    if chain:
        n = _sync_shared(model_inputs, os.path.join(openda_dir, SHARED_DIR), binding.shared_inputs)
        binding.point_config_at_shared(template_dir, SHARED_REL)
        kept, total = binding.cut_forcing(model_inputs, template_dir, binding.model_time(cfg["start_date"]),
                                          binding.model_time(cfg["end_date"]))
        logger.info(f"[adapter] {n} shared input file(s) updated, {binding.forcing_file} cut to {kept}/{total} rows")

    for i in range(cfg["n_members"] + 1):
        src = binding.member_forcing(cfg["ensemble_base"], i)
        if not os.path.isfile(src):
            raise FileNotFoundError(f"{src} missing — run the copy and perturbation steps first")
        _copy(src, os.path.join(openda_dir, "forcings", binding.forcing_name(i)))

    marker = os.path.join(template_dir, RESTART_MARKER)
    if restart_in:
        if not os.path.isfile(restart_in):
            raise FileNotFoundError(f"OpenDA restart file not found: {restart_in}")
        n = binding.fill_template_sentinel(model_inputs, template_dir, RESTART_SENTINEL)
        with open(marker, "w") as f:
            f.write(f"{restart_in}\n")
        logger.info(f"[adapter] continuing from restart {display_path(restart_in)} ({n} state values)")
    else:
        if os.path.exists(marker):
            os.remove(marker)
        seeded = binding.seed_template(model_inputs, template_dir)
        if seeded is None:
            logger.warning(f"[adapter] no warm start in {display_path(model_inputs)}")
        else:
            logger.info(f"[adapter] warm start {os.path.basename(seeded[0])} ({seeded[1]} state values)")

    return build_observations(cfg, openda_dir, model_inputs, binding, season_split)

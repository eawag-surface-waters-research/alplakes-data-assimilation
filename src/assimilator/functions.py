import os
import sys
import json
import logging
from datetime import datetime, timezone

from tqdm import tqdm

# pandas is imported lazily inside load_obs (the only user here): this module is on the import path
# of the OpenDA wrapper (via models.simstrat), launched 21×/step, and a top-level `import pandas`
# cost ~2.6s per process for nothing.

logger = logging.getLogger(__name__)

# Repo paths, from this file at src/assimilator/functions.py
SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # …/src
ROOT    = os.path.dirname(SRC_DIR)                                      # …/alplakes-data-assimilation

# General config — see static/general.json. Loaded at import (the file is committed). Exposes the
# ICON acquisition settings (forcing fit); the model-specific values (Simstrat epoch, forcing-file
# header) are read from GENERAL by assimilator.models.simstrat, not surfaced here.
with open(os.path.join(ROOT, "static", "general.json"), encoding="utf-8") as _f:
    GENERAL = json.load(_f)
API_BASE  = GENERAL["icon_api_base"]
VARIABLES = GENERAL["icon_variables"]

# Model-specific behaviour (Docker run, Settings.par editing, snapshot I/O, z_out/T_out formats,
# the day-since-reference-year epoch, the forcing-file format, input layout) lives in
# assimilator.models.<model>; this module keeps only generic, model-agnostic helpers.

# ---------------------------------------------------------------------------
# Path / config helpers
# ---------------------------------------------------------------------------

def resolve_src(path):
    """Resolve a possibly-relative ensemble_base ('../run/<lake>') against src/."""
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(SRC_DIR, path))


def resolve_root(path):
    """Resolve a possibly-relative repo path ('openda_simstrat', 'args/x.json') against ROOT."""
    return path if os.path.isabs(path) else os.path.normpath(os.path.join(ROOT, path))


def resolve_run_root(cfg):
    """Base directory for all run OUTPUT — the per-lake ensemble instances (ensemble_base) and the
    OpenDA work dirs (openda_dir) both default under it. Model INPUTS stay in-repo (ROOT/inputs).

    Defaults to the in-repo ROOT/run, so a fresh checkout is self-contained — nothing to configure
    when running on a remote Linux server. Override to relocate ALL run output (e.g. point local
    tests at a fast native-ext4 path while the /mnt/c source tree stays put) via the 'run_root'
    config key or the ALPLAKES_RUN_ROOT env var (config key wins). '~' and $ENV are expanded; a
    relative value resolves against ROOT. Explicit 'ensemble_base'/'openda_dir' still override per-dir."""
    raw = cfg.get("run_root") or os.environ.get("ALPLAKES_RUN_ROOT")
    if not raw:
        return os.path.join(ROOT, "run")
    return resolve_root(os.path.expanduser(os.path.expandvars(raw)))


def display_path(path):
    """Pretty path for logs: relative to ROOT when inside the repo, else the absolute path with
    $HOME collapsed to '~'. Keeps in-repo logs terse while a rerouted run_root (e.g. on ext4) reads
    as '~/alplakes-data-assimilation_res/run/...' instead of '../../../../home/...'."""
    rel = os.path.relpath(path, ROOT)
    if not rel.startswith(".."):
        return rel
    home = os.path.expanduser("~")
    return "~" + path[len(home):] if path.startswith(home) else path


def load_json(path):
    """Load a config JSON, trying the path as given then under ROOT."""
    p = path if os.path.isfile(path) else resolve_root(path)
    if not os.path.isfile(p):
        raise FileNotFoundError(f"args file not found: {path}")
    with open(p) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Progress reporting
# ---------------------------------------------------------------------------
#   Both engines report run progress through a single tqdm bar instead of a log
#   line per assimilation step. The bar is auto-disabled for headless/server runs
#   (no TTY) so log files don't fill with redraw spam; when disabled the engines
#   fall back to their per-step logger.info lines. One resolver keeps the on/off
#   rule in one place (precedence: CLI --no-progress > config "progress" > isatty).

def resolve_progress(cfg, no_progress_flag=False):
    """Decide whether to show the progress bar. CLI --no-progress wins; else an
    explicit "progress" config key; else auto-detect (on only with a real TTY)."""
    if no_progress_flag:
        return False
    if "progress" in cfg:
        return bool(cfg["progress"])
    return sys.stderr.isatty()


def make_progress(total, enabled, desc=None):
    """Return a tqdm bar with consistent styling. When `enabled` is False the bar
    is a no-op (update/set_postfix do nothing), so callers need no branching to
    advance it — they only branch to keep the plain logger.info lines when off."""
    return tqdm(total=total, disable=not enabled, desc=desc,
                unit="day", dynamic_ncols=True, leave=True)


def resolve_max_workers(cfg, n_members):
    """How many ensemble members to run concurrently (the parallelisation knob, shared by both
    engines: the native ThreadPool and OpenDA's maxThreads). DEFAULT is full concurrency (all at
    once). 'max_workers' (config key or --max-workers) is a knob to RESTRICT, not the default.

    Capping to CPU cores looks tempting — the synchronized per-step snapshot read/write inflates
    sharply under oversubscription (a per-member "storm"). But it's measured to HURT wall-clock: an
    A/B on ext4 (21 vs 8 workers, 8 cores) cut the per-member inject 6.06s->2.30s yet made the WINDOW
    ~14% slower (21.5s -> 24.5s), because the members just serialise into waves. The storm is a
    symptom of beneficial overlap, not wasted work; full concurrency hides the latency (more so on a
    slow IO-bound bind mount). So don't auto-cap — set max_workers explicitly only to throttle
    CPU/RAM. An explicit value is clamped to [1, n_members] (n_members = the cap the caller passes:
    members for native, members+1 for OpenDA's main+members)."""
    cpu       = os.cpu_count() or 1
    requested = cfg.get("max_workers")
    workers   = max(1, min(int(requested), n_members)) if requested else n_members
    src       = f"cfg max_workers={requested}" if requested else "default=full"
    logger.info(f"parallelism: {workers} concurrent (cpu={cpu}, cap={n_members}, {src})")
    return workers


def log_obs_summary(obs, obs_path, label="Obs"):
    """Log the loaded + depth-filtered observation set (count, depths, time span, source).
    Setup-time narrative shared by the native engines, so the log shows exactly what will be
    assimilated — the counterpart to the OpenDA adapter's '[adapter] observations: ...' line."""
    if obs.empty:
        logger.warning(f"{label}: no observations after filtering <- {os.path.relpath(obs_path, ROOT)}")
        return
    depths = sorted(obs["depth"].unique())
    logger.info(f"{label}: {len(obs)} readings, {len(depths)} depths {[f'{d:g}' for d in depths]}, "
                f"{obs['time'].min().date()}..{obs['time'].max().date()} "
                f"<- {os.path.relpath(obs_path, ROOT)}")


# A run-config-echo header and a result footer, in ONE layout shared by both engines, so two runs'
# logs line up at the cross-validation junctions: A (same config?) and D (same result?). Defined
# here once so the native engines (run_enkf/run_pf) and OpenDA (run_openda) can't drift apart, and
# using a standardized key vocabulary (engine= algo=/filter= window= n_members= sigma_obs= ...).

def _run_selector(cfg):
    """The engine's algorithm/filter label for the header & footer (vocab: 'algo=' or 'filter=')."""
    engine = cfg.get("engine", "python")
    return f"algo={cfg.get('algorithm')}" if engine == "python" else f"filter={cfg.get('filter')}"


def log_run_header(cfg):
    """Echo the resolved run config as one diffable block (junction A). Reads cfg with .get();
    prints whichever knobs apply (inflation is native-EnKF-only)."""
    engine = cfg.get("engine", "python")
    logger.info(f"=== RUN | engine={engine} {_run_selector(cfg)} "
                f"model={cfg.get('model', 'simstrat')} lake={cfg.get('lake')} ===")
    logger.info(f"      window={cfg.get('start_date')}..{cfg.get('end_date')}  "
                f"n_members={cfg.get('n_members')}")
    knobs = [f"sigma_obs={cfg.get('sigma_obs')}"]
    if engine == "python" and cfg.get("inflation") is not None:
        knobs.append(f"inflation={cfg.get('inflation')}")
    knobs += [f"sigma_scale={cfg.get('sigma_scale', 1.0)}", f"rng_seed={cfg.get('rng_seed', 42)}"]
    logger.info("      " + "  ".join(knobs))
    logger.info(f"      obs={os.path.relpath(resolve_obs_path(cfg), ROOT)}")


def log_run_footer(cfg, skill, steps, updates, elapsed_s, out_path=None):
    """Echo the run result as one diffable block (junction D). `skill` is report_summary's overall
    dict (rmse/bias/n) or None; `updates` may be None where the engine has no separate count."""
    logger.info(f"=== DONE | engine={cfg.get('engine', 'python')} {_run_selector(cfg)} "
                f"lake={cfg.get('lake')} ===")
    line = f"      steps={steps}"
    if updates is not None:
        line += f"  updates={updates}"
    line += f"  elapsed={elapsed_s:.1f}s"
    logger.info(line)
    if skill:
        logger.info(f"      rmse={skill['rmse']}  bias={skill['bias']}  n_obs={skill['n']}  (analysis fit)")
    if out_path:
        logger.info(f"      output={os.path.relpath(out_path, ROOT)}")


def merge_lake_args(cfg, lake=None):
    """Flatten a run config that carries per-lake blocks under "lakes": pick the selected lake's
    block and overlay it on the top-level engine knobs. The lake is the explicit `lake` arg, else
    cfg['lake'], else — when there is exactly one block — that one. Add a lake = add a block.
    Sets 'lake' and defaults ensemble_base to ../run/<lake>. A config without "lakes" (already
    flat) is returned unchanged."""
    lakes = cfg.get("lakes")
    if not lakes:
        return cfg
    lake = lake or cfg.get("lake")
    if lake is None:
        if len(lakes) != 1:
            raise ValueError(f"no lake selected — pass --lake (choices: {sorted(lakes)})")
        lake = next(iter(lakes))
    if lake not in lakes:
        raise ValueError(f"lake '{lake}' not in this config's \"lakes\"; choices: {sorted(lakes)}")
    merged = {k: v for k, v in cfg.items() if k != "lakes"}
    merged.update(lakes[lake])
    merged["lake"] = lake
    merged.setdefault("ensemble_base", os.path.join(resolve_run_root(merged), lake))
    return merged


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------
def verify_args(args, required):
    for key in required:
        if key not in args:
            raise ValueError(f"Required argument '{key}' missing from args file.")


def verify_file(path):
    if os.path.isfile(path):
        return os.path.abspath(path)
    raise ValueError(f"File not found: {os.path.abspath(path)}")


def discover_n_members(ensemble_base):
    return len([
        d for d in os.listdir(ensemble_base)
        if d.startswith("ensemble") and d != "ensemble0"
        and os.path.isdir(os.path.join(ensemble_base, d))
    ])


# ---------------------------------------------------------------------------
# Observations (model-agnostic: read the obs CSV, map depths, filter to model grid)
# ---------------------------------------------------------------------------

# Hourly mean, centered on the label: the value at HH:00 is the mean of samples in
# [HH-30min, HH+30min), computed by flooring (time + 30min) to the hour. Centering aligns the
# obs with Simstrat's instantaneous hourly output, so the noon assimilation pairs the noon obs
# with the noon state. Labels stay on the hour, so the PF's obs<->T_out time intersection
# (rmse_in_window) is unaffected.
def load_obs(obs_path):
    import pandas as pd
    obs = pd.read_csv(obs_path, parse_dates=["time"])
    obs["time"] = pd.to_datetime(obs["time"], utc=True)
    obs["time"] = (obs["time"] + pd.Timedelta(minutes=30)).dt.floor("1h")
    obs = obs.groupby(["depth", "time"])["value"].mean().reset_index()
    return obs


def filter_obs_to_model_depths(obs_df, model_depths):
    """Drop obs whose depth has no matching model-output depth (abs diff <= 1e-6), so the
    Python engines assimilate the same depths OpenDA does. No-op if model_depths is empty."""
    if not model_depths:
        return obs_df
    obs_depths = sorted(obs_df["depth"].unique())
    matched = {d for d in obs_depths if any(abs(d - m) <= 1e-6 for m in model_depths)}
    dropped = [d for d in obs_depths if d not in matched]
    if dropped:
        logger.warning(f"[obs] dropping obs depths with no matching model output depth (z_out.dat): "
                       f"{[f'{d:g}' for d in dropped]} m")
    return obs_df[obs_df["depth"].isin(matched)]


def resolve_obs_path(cfg):
    """Observation CSV for the run: the 'obs_file' override (path relative to the repo root, or
    absolute) if given, else observations/<lake>/temperature.csv."""
    override = cfg.get("obs_file")
    return resolve_root(override) if override else os.path.join(ROOT, "observations", cfg["lake"], "temperature.csv")


def to_utc(iso_str):
    """Parse an ISO date/datetime string to a tz-aware UTC datetime (assumes naive input)."""
    return datetime.fromisoformat(iso_str).replace(tzinfo=timezone.utc)


# =============================================================================
# Python engine run-arg builder (shared by run_enkf / run_pf)
# =============================================================================

def build_python_run_args(run_raw, ensemble_raw, ensemble_base, n_members, model):
    """Merge run-specific knobs with shared ensemble facts; fill defaults (obs path,
    model runtime config), parse UTC dates, derive ref_date + the mean-trajectory path
    (from the selected `model`), and (for EnKF) the diagnostics output paths."""
    args = dict(run_raw)
    args["lake"]          = ensemble_raw["lake"]
    args["ensemble_base"] = ensemble_base
    args["n_members"]     = n_members
    args["member_ids"]    = list(range(1, n_members + 1))

    args["obs_path"] = resolve_obs_path(args)   # 'obs_file' override, else observations/<lake>/temperature.csv
    args["max_workers"] = resolve_max_workers(args, n_members)   # concurrent members (auto = min(cpu, members))
    for k, v in model.run_config().items():   # model's Docker/runtime defaults (fallback)
        args.setdefault(k, v)

    args["ref_date"] = model.read_ref_date(ensemble_base)

    args["start_date"] = to_utc(ensemble_raw["start_date"])
    args["end_date"]   = to_utc(ensemble_raw["end_date"])

    # Observation error std (sigma_obs) is shared with OpenDA (the same key in the run config);
    # inflation is Python-EnKF-only (OpenDA has no inflation param). Carried into args here so
    # the EnKF code reads it uniformly.
    if "sigma_obs" in ensemble_raw:
        args["sigma_obs"] = ensemble_raw["sigma_obs"]

    args.setdefault("mean_traj_path", model.mean_traj_path(ensemble_base, args["algorithm"]))
    if args["algorithm"] == "EnKF":
        args.setdefault("diag_path",        os.path.join(ensemble_base, "enkf_diagnostics.csv"))
        args.setdefault("innov_depth_path", os.path.join(ensemble_base, "enkf_innov_by_depth.csv"))
        args.setdefault("kgain_depth_path", os.path.join(ensemble_base, "enkf_kgain_by_depth.csv"))
    return args

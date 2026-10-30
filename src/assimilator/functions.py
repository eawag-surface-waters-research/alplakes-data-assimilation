import os
import sys
import logging
import subprocess
from datetime import datetime, timezone, date, timedelta

# pandas is imported lazily inside load_obs (the only user here): this module is on the import path
# of the OpenDA wrapper (via models.simstrat), launched 21×/step, and a top-level `import pandas`
# cost ~2.6s per process for nothing.

logger = logging.getLogger(__name__)

# This package is a library: it resolves no paths against its own checkout and reads no config
# of its own. Every path in a run config is used as given (assimilator.config makes them absolute
# against the caller's working directory), so the package behaves the same whether it is run from
# its own CLI or embedded in a host repository.

# Model-specific behaviour (Docker run, Settings.par editing, snapshot I/O, output formats, the
# time epoch, the forcing-file format, input layout) lives in assimilator.models.<model>; this
# module keeps only generic, model-agnostic helpers.

# ---------------------------------------------------------------------------
# Path / config helpers
# ---------------------------------------------------------------------------

def display_path(path):
    """Pretty path for logs: the absolute path with $HOME collapsed to '~'."""
    path = os.path.abspath(path)
    home = os.path.expanduser("~")
    return "~" + path[len(home):] if home != "/" and path.startswith(home + os.sep) else path


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


class _NullBar:
    def update(self, *a, **k): pass
    def set_postfix_str(self, *a, **k): pass
    def close(self, *a, **k): pass


def make_progress(total, enabled, desc=None):
    """Return a tqdm bar with consistent styling. When `enabled` is False the bar
    is a no-op (update/set_postfix do nothing), so callers need no branching to
    advance it — they only branch to keep the plain logger.info lines when off."""
    if not enabled:
        return _NullBar()
    from tqdm import tqdm
    return tqdm(total=total, disable=False, desc=desc,
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


def start_persistent_container(cfg, name, mount, target, image):
    if cfg.get("docker_dir") and cfg.get("repo_dir"):
        mount = os.path.join(cfg["docker_dir"], os.path.relpath(mount, cfg["repo_dir"]))
    mount = mount.replace("\\", "/")
    subprocess.run(f"docker rm -f {name}", shell=True, capture_output=True)
    cmd = (
        f"docker run -d --name {name} "
        f"-v {mount}:{target} "
        f"--entrypoint sleep "
        f"{image} infinity"
    )
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"container start failed ({name}): {result.stderr.strip()}")


def stop_persistent_container(name):
    subprocess.run(f"docker stop {name}", shell=True, capture_output=True)
    subprocess.run(f"docker rm   {name}", shell=True, capture_output=True)


def log_obs_summary(obs, obs_path, label="Obs"):
    """Log the loaded + depth-filtered observation set (count, depths, time span, source).
    Setup-time narrative shared by the native engines, so the log shows exactly what will be
    assimilated — the counterpart to the OpenDA adapter's '[adapter] observations: ...' line."""
    if obs.empty:
        logger.warning(f"{label}: no observations after filtering <- {display_path(obs_path)}")
        return
    depths = sorted(obs["depth"].unique())
    logger.info(f"{label}: {len(obs)} readings, {len(depths)} depths {[f'{d:g}' for d in depths]}, "
                f"{obs['time'].min().date()}..{obs['time'].max().date()} "
                f"<- {display_path(obs_path)}")


# A run-config-echo header and a result footer, in ONE layout shared by both engines, so two runs'
# logs line up at the cross-validation junctions: A (same config?) and D (same result?). Defined
# here once so the native engines (run_enkf/run_pf) and OpenDA (run_openda) can't drift apart, and
# using a standardized key vocabulary (engine= algo= window= n_members= sigma_obs= ...).

def _run_selector(cfg):
    """The DA-scheme label for the header & footer (both engines read cfg['algorithm'])."""
    return f"algo={cfg.get('algorithm')}"


def log_run_header(cfg):
    """Echo the resolved run config as one diffable block (junction A). Reads cfg with .get();
    prints whichever knobs apply (inflation is native-EnKF-only)."""
    engine = cfg.get("engine", "python")
    logger.info(f"=== RUN | engine={engine} {_run_selector(cfg)} "
                f"model={cfg.get('model')} lake={cfg.get('lake')} ===")
    logger.info(f"      window={cfg.get('start_date')}..{cfg.get('end_date')}  "
                f"n_members={cfg.get('n_members')}")
    knobs = [f"sigma_obs={cfg.get('sigma_obs')}"]
    if engine == "python" and cfg.get("inflation") is not None:
        knobs.append(f"inflation={cfg.get('inflation')}")
    if engine == "python":
        knobs.append(f"window_mode={cfg.get('window_mode', 'obs')}")
    knobs += [f"sigma_scale={cfg.get('sigma_scale', 1.0)}", f"rng_seed={cfg.get('rng_seed', 42)}"]
    logger.info("      " + "  ".join(knobs))
    logger.info(f"      obs={display_path(resolve_obs_path(cfg))}")


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
        logger.info(f"      output={display_path(out_path)}")


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
def bin_hourly(obs):
    """A (time, depth, value) frame -> hourly, centred: one mean value per (depth, hour)."""
    import pandas as pd
    obs = obs.copy()
    # format="ISO8601": the filtered/thinned CSVs mix whole-second and fractional-second stamps,
    # and pandas 2.x locks onto row 0's format and rejects every row that differs.
    obs["time"] = pd.to_datetime(obs["time"], utc=True, format="ISO8601")
    obs["time"] = (obs["time"] + pd.Timedelta(minutes=30)).dt.floor("1h")
    return obs.groupby(["depth", "time"])["value"].mean().reset_index()


def load_obs(obs_path):
    import pandas as pd
    return bin_hourly(pd.read_csv(obs_path))


def filter_obs_to_model_depths(obs_df, model_depths):
    """Drop obs whose depth has no matching model-output depth (abs diff <= 1e-6), so the
    Python engines assimilate the same depths OpenDA does. No-op if model_depths is empty."""
    if not model_depths:
        return obs_df
    obs_depths = sorted(obs_df["depth"].unique())
    matched = {d for d in obs_depths if any(abs(d - m) <= 1e-6 for m in model_depths)}
    dropped = [d for d in obs_depths if d not in matched]
    if dropped:
        logger.warning(f"[obs] dropping obs depths with no matching model output depth: "
                       f"{[f'{d:g}' for d in dropped]} m")
    return obs_df[obs_df["depth"].isin(matched)]


def resolve_obs_path(cfg):
    """Observation CSV for the run: the config's 'obs_file' (required; made absolute by
    assimilator.config.build_config)."""
    if not cfg.get("obs_file"):
        raise ValueError("run config has no 'obs_file' (the observation CSV to assimilate)")
    return cfg["obs_file"]


def obs_window_boundaries(obs_df, start_date, end_date):
    """Assimilation-window boundaries for obs-driven stepping (window_mode="obs"): every distinct
    observation time in (start_date, end_date], plus end_date as the final boundary so the model is
    always advanced to the end of the run window (the tail window has no obs, hence no update).

    Chunk-invariant by construction: the boundaries depend only on the observations, so an
    operational run continued in daily slices steps through the SAME windows (and assimilates at
    the same instants) as one continuous run over the whole period. This is also OpenDA's rule
    (analysisTimes type="fromObservationTimes"), so the two engines step alike."""
    times = sorted(t.to_pydatetime() for t in obs_df["time"].drop_duplicates())
    bounds = [t for t in times if start_date < t <= end_date]
    if not bounds or bounds[-1] < end_date:
        bounds.append(end_date)
    return bounds


# ---------------------------------------------------------------------------
# Time-keyed RNG streams (operational reproducibility)
# ---------------------------------------------------------------------------
#   Rule: every random draw that influences the assimilation result is keyed by
#   (rng_seed, stream id, absolute time) — never by position in the run. A draw for a given
#   instant is then identical no matter how the period is chunked, so an operational run
#   continued each day reproduces a continuous run exactly. Streams are registered here (one
#   integer per consumer) so two consumers can never collide on the same key.

RNG_STREAMS = {
    "forcing_U":    0,   # perturbate.py: AR(1) noise per forcing row (step = s since Simstrat epoch)
    "forcing_V":    1,
    "forcing_GLOB": 2,
    "enkf_obs":     3,   # enkf.py: obs-perturbation draw per analysis (step = unix s of window_end)
}


def time_keyed_rng(rng_seed, stream, step):
    """A numpy Generator for one (stream, absolute-time) key. `step` is a non-negative integer
    time coordinate; its epoch/unit may differ per stream (the stream id separates them)."""
    import numpy as np
    return np.random.default_rng([int(rng_seed), RNG_STREAMS[stream], int(step)])


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


# =============================================================================
# Observation-error model (sigma_obs)
# =============================================================================
# The scalar `sigma_obs` is an estimate of REPRESENTATIVENESS error, not instrument error. That
# quantity is not a constant: it peaks at the thermocline, where a sharp gradient is displaced past
# a fixed sensor by internal seiches, and collapses in the deep and in the mixed season.
#
# One axis for now, the season:
#
#     sigma_obs    a float, or {"mixed": .., "stratified": ..}
#
# A float is bit-for-bit what the filters did before this section existed.

# Two seasons, on the calendar. The project's stratified window is May-Oct everywhere: a
# stratification rule was tried and did not beat it, and a calendar split keeps the fit and the run
# configs keyed the same way without either needing the run's own state.
SEASONS      = ("mixed", "stratified")
STRAT_MONTHS = range(5, 11)


def season_of(month):
    return "stratified" if int(month) in STRAT_MONTHS else "mixed"


def _is_season_keyed(spec):
    """True for {"mixed": .., "stratified": ..}, so a plain scalar config is recognised unchanged."""
    return isinstance(spec, dict) and bool(spec) and set(spec) <= set(SEASONS)


def has_season_keyed_sigma_obs(cfg):
    """Whether the sigma_obs this run resolves to carries a season axis.

    Structural, NOT np.ndim: np.ndim({...}) is 0, so a dict takes the scalar branch and silently
    reads as annual -- and np.full(n, dict) then builds an object array enkf_update cannot square."""
    return _is_season_keyed(sigma_obs_spec(cfg))


# The fallback when a run names no sigma_obs of its own. A separate key rather than a default for
# "sigma_obs", so the log can say when the fallback (and not a fitted value) is in use.
SIGMA_DEFAULT_KEY = "sigma_default"
SIGMA_DEFAULT     = 0.5


def sigma_obs_spec(cfg):
    """What sigma_obs this run uses: the config's own value if it names one, else sigma_default.

    A sigma is only valid for the observation series it was fitted on (assimilator.fitting.sigma),
    so the host stores the fitted value next to the observation-processing settings it belongs to
    and passes it in the run config."""
    spec = cfg.get("sigma_obs")
    return cfg.get(SIGMA_DEFAULT_KEY, SIGMA_DEFAULT) if spec is None else spec


def resolve_scalar_sigma_obs(cfg, month=None):
    """The run's sigma_obs -- a plain float, or a season-keyed pair

        "sigma_obs": {"mixed": 0.31, "stratified": 0.55}

    The gap a scalar is fitted from is not one population: across the 2025 lakes the stratified
    season wants 1.1x to 2.6x the mixed-season value, so a single annual number is the
    count-weighted compromise between them -- wrong in both at once.

    A season-keyed value REQUIRES `month`; None raises rather than silently picking a season."""
    spec = sigma_obs_spec(cfg)
    if _is_season_keyed(spec):
        if month is None:
            raise ValueError("sigma_obs is season-keyed, so resolving it needs a month")
        season = season_of(month)
        if season not in spec:
            raise ValueError(f"sigma_obs has no {season!r} entry (has {sorted(spec)})")
        spec = spec[season]
    return float(spec)


def resolve_sigma_obs(depths, when, cfg):
    """Per-observation sigma for one analysis, aligned with the obs vector.

    One entry per depth in `depths` whether the config carries one number or two, so the filters
    have a single code path. `when` is the analysis instant, whose month selects the season.
    Uniform over depth today -- the vector is the shape a depth axis would fill in.

    A plain list, not an array: this module is on the OpenDA wrapper's import path (see the pandas
    note at the top), and the callers that need an array already import numpy."""
    return [resolve_scalar_sigma_obs(cfg, when.month)] * len(depths)


def should_season_split(cfg):
    """Whether the OpenDA adapter should write one observation series per (depth, season).

    OpenDA's stochObserver carries one standardDeviation per SERIES, and the series are ours to
    define -- so one series per (depth, season) reproduces the native engine's per-window sigma
    EXACTLY.
    Decidable from the config alone, which is what lets the adapter write the files BEFORE the
    sigma per series is known -- see sigma_obs_for_series."""
    spec = sigma_obs_spec(cfg)
    if not _is_season_keyed(spec):
        return False
    if len({float(v) for v in spec.values()}) == 1:
        logger.info("[sigma] OpenDA season split declined: sigma identical in both seasons")
        return False
    return True


def sigma_obs_for_series(series_keys, cfg):
    """{(depth, season): sigma} for the series the adapter ACTUALLY wrote.

    Keyed off the written files rather than off the observation frame, so the series the config
    declares and the files on disk cannot disagree -- a disagreement is fatal inside OpenDA, which
    opens a _real.csv that is not there and dies with no member output. Deriving both from one obs
    frame is not enough: the adapter keeps only the samples in its target hour, so a (depth,
    season) can exist in the observations and still have no file.
    A (depth, season) with no file is simply absent, so an operational window inside one season
    carries exact sigma at every depth that reported."""
    spec = sigma_obs_spec(cfg)
    return {(float(d), s): float(spec[s]) for d, s in series_keys}


def log_sigma_summary(cfg):
    """Say, once, what observation-error model this run is using -- the counterpart to the
    'sigma_obs=' knob in the run header, which no longer tells the whole story. Names the SOURCE
    too: falling back to sigma_default is silent otherwise, and it is the case worth noticing."""
    spec = sigma_obs_spec(cfg)
    src  = "config" if cfg.get("sigma_obs") is not None else f"{SIGMA_DEFAULT_KEY} fallback"
    txt = ("season-keyed " + ", ".join(f"{s}: {float(v):g}" for s, v in sorted(spec.items()))
           if _is_season_keyed(spec) else f"{float(spec):g} degC")
    logger.info(f"[sigma] sigma_obs={txt} (from {src})")

"""End-to-end data-assimilation orchestrator. One front door for both engines:

  1. require model_inputs (provided manually)  -> inputs/<lake>/
  2. copy_model_inputs -> ensemble0..N   3. perturbate -> Forcing.dat in 1..N
  4-5. run + summarize:  python -> run_enkf / run_pf   |   openda -> run_openda

Step 2 is skipped when already done (override: --force-copy); step 3 (perturbate)
requires a committed perturbations/<lake>.json and errors if it is missing.
Each run config is one JSON: engine/model selection + engine knobs + the run window
(n_members, dates, sigma_obs, ...) at top level, plus a "lakes" map of lake-identity
blocks. --lake picks one block (the only one by default) and merges it on top:

  {"engine": "python|openda", "model": "simstrat",
   "algorithm": "EnKF|PF",           # python: + par_file / results_dir / inflation
   "filter": "EnKF|DEnKF|EnSR|PF",   # openda:  + openda_bin
   "n_members": ..., "start_date": ..., "end_date": ..., "sigma_obs": ...,
   "lakes": {"<lake>": {"reanalysis_lake": ..., "lake_bbox": ..., "lake_key": ...}}}

    python src/main.py args/run_enkf.json   [--lake <name>] [-m simstrat] [--force-*]
    python src/main.py args/run_openda.json [--lake <name>] [--skip-oda]  # openda: WSL + Docker

The forward model is selected with -m/--model (default: simstrat; see models.MODELS).
"""

import os
import sys
import logging
import argparse
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # put src/ on the path

from assimilator.perturbate      import perturbator, load_perturbations
from assimilator.models          import get_model
from assimilator.functions       import ROOT, resolve_src, load_json, load_obs, resolve_obs_path, merge_lake_args, resolve_progress, display_path
from assimilator.algorithms.enkf import run_enkf
from assimilator.algorithms.pf   import run_pf
from assimilator.openda.adapter import run_openda

logger = logging.getLogger(__name__)


def run(cfg, model="simstrat", skip_oda=False, force=None):
    """Run the pipeline: require model_inputs, then copy -> perturbate -> engine run.
    Step 2 skips when already done (unless --force-copy); step 3 perturbates and
    errors if perturbations/<lake>.json is missing (fit it offline beforehand).
    `model` selects the forward model (see models.MODELS); its runtime config is
    merged into the Python engine's run args."""
    force  = force or {}
    model_obj = get_model(model)   # instance of the selected forward model (validates -m/--model)
    engine = cfg.get("engine", "python")
    if engine not in ("python", "openda"):
        raise ValueError(f"unknown engine '{engine}'; choose 'python' or 'openda'")

    ensemble_raw = cfg   # one flat config: ensemble facts + engine knobs in the same JSON

    lake          = ensemble_raw["lake"]
    n_members     = ensemble_raw["n_members"]
    ensemble_base = resolve_src(ensemble_raw["ensemble_base"])
    model_inputs = os.path.join(ROOT, "inputs", lake)

    # Pin every step to the same absolute ensemble_base (avoids cwd-dependent
    # resolution differences between the sub-scripts).
    ensemble_raw["ensemble_base"] = ensemble_base

    logger.info(f"=== DA pipeline - lake={lake}  base={display_path(ensemble_base)}"
                f"  model={model}  engine={engine} ===")

    # --- 1. model inputs (provided manually) ---------------------------
    if not model_obj.model_inputs_ready(model_inputs):
        raise FileNotFoundError(
            f"model_inputs not ready at {os.path.relpath(model_inputs, ROOT)}: "
            f"provide it manually — a dated simulation-snapshot_*.dat, Forcing.dat, "
            f"Settings.par and the remaining Simstrat inputs.")
    logger.info(f"[1/5] model inputs present -> {os.path.relpath(model_inputs, ROOT)}")
    warmup = model_obj.warmup_snapshot(model_inputs)
    if warmup:
        logger.info(f"      warm-start snapshot: {os.path.basename(warmup)}")

    # --- 2. copy into instances -------------------------------------------
    if force.get("copy") or not model_obj.instances_ready(ensemble_base, n_members):
        why = "forced" if force.get("copy") else "missing instances"
        logger.info(f"[2/5] copy model inputs -> ensemble0..{n_members} ({why})")
        model_obj.copy_model_inputs(ensemble_raw)
    else:
        logger.info(f"[2/5] ensemble0..{n_members} present - skip")

    # --- 2b. align model output depths to the observations ----------------
    #   Overwrite z_out.dat (inputs + every member) with the superset of its depths and the
    #   observation depths, so no observation is dropped just for falling between the default
    #   outputs. Runs whether or not step 2 copied, so a skipped copy still gets the update.
    obs_path = resolve_obs_path(ensemble_raw)
    if os.path.isfile(obs_path):
        obs_depths = sorted(load_obs(obs_path)["depth"].unique())
        n = model_obj.set_output_depths(model_inputs, ensemble_base, n_members, obs_depths)
        if n:
            logger.info(f"      z_out.dat <- model + obs depth superset ({len(obs_depths)} obs depths; {n} files updated)")

    # --- 3. perturbate forcings (always) ----------------------------------
    #   Source the AR(1) calibration from perturbations/<lake>.json. It must already
    #   exist (committed); fit it once with notebooks/perturbations_from_icon.py.
    logger.info(f"[3/5] perturbate Forcing.dat in ensemble1..{n_members}")
    params = load_perturbations(ensemble_raw)   # fail fast: errors if missing or malformed
    perturbator(ensemble_raw, params=params)

    # --- 4-5. engine-specific run + summary --------------------------------
    if engine == "python":
        run_raw = ensemble_raw   # same flat config; build_python_run_args reads the run-side keys
        for k, v in model_obj.run_config().items():   # model's Docker/runtime defaults (e.g. simstrat_version)
            run_raw.setdefault(k, v)
        algo    = run_raw.get("algorithm")
        if algo == "EnKF":
            run_enkf(run_raw, ensemble_raw, ensemble_base, n_members, model_obj)
        elif algo == "PF":
            run_pf(run_raw, ensemble_raw, ensemble_base, n_members, model_obj)
        else:
            raise ValueError(f"Unknown algorithm: '{algo}'. Use 'PF' or 'EnKF'.")
    else:
        run_openda(cfg, ensemble_raw, ensemble_base, n_members, skip_oda,
                   model_cfg=model_obj.run_config(), model_name=model)

    logger.info("=== pipeline complete ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="End-to-end data-assimilation pipeline (Python or OpenDA)")
    parser.add_argument("arg_file", help="Pipeline config JSON (e.g. args/run_enkf.json)")
    parser.add_argument("--skip-oda", action="store_true", help="OpenDA only: run setup + adapter, but not OpenDA")
    parser.add_argument("-m", "--model", default=None,
                        help="Forward model to run; overrides the arg file's \"model\" field "
                             "(default: simstrat; see models.MODELS)")
    parser.add_argument("--lake", default=None,
                        help="Lake to run from the config's \"lakes\" block "
                             "(default: the only block, if there is just one)")
    parser.add_argument("--obs-file", default=None,
                        help="Observation CSV, overriding the config's \"obs_file\" "
                             "(default: observations/<lake>/temperature.csv)")
    parser.add_argument("--perturbations-file", default=None,
                        help="AR(1) calibration JSON, overriding the config's \"perturbations_file\" "
                             "(default: perturbations/<lake>.json)")
    parser.add_argument("--force-copy",       action="store_true", help="Re-run step 2 even if present")
    parser.add_argument("--no-progress", action="store_true",
                        help="Disable the progress bar (auto-disabled when stderr is not a TTY). "
                             "Per-step detail still goes to the log file either way")
    parser.add_argument("--max-workers", type=int, default=None,
                        help="Cap concurrent members (both engines). Default: full concurrency (all at once)")
    parser.add_argument("--run-root", default=None,
                        help="Base dir for run OUTPUT (ensemble instances + OpenDA work dir); overrides "
                             "the config 'run_root' key and $ALPLAKES_RUN_ROOT. Default: in-repo ./run "
                             "(self-contained). '~'/$VARS expand; relative resolves against the repo root.")
    cli = parser.parse_args()

    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    log_file = os.path.join(ROOT, "logs", f"pipeline_{datetime.now():%Y%m%d_%H%M%S}.log")
    # Per-day assimilation lines are logged with extra={"file_only": True} so they always land in
    # the log file but never clutter the console — the console shows the progress bar (or, when it's
    # off, just the milestones). The filter on the console handler drops those file-only records.
    console = logging.StreamHandler()
    console.addFilter(lambda r: not getattr(r, "file_only", False))
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-8s | %(name)-16s | %(message)s",
                        datefmt="%H:%M:%S",
                        handlers=[console,
                                  logging.FileHandler(log_file, encoding="utf-8")])
    logger.info(f"log file -> {os.path.relpath(log_file, ROOT)}")

    raw = load_json(cli.arg_file)
    # CLI --run-root wins over the config 'run_root' key (and over $ALPLAKES_RUN_ROOT). Applied to the
    # raw config BEFORE merge_lake_args, since that's where ensemble_base picks up its run_root default.
    if cli.run_root:
        raw["run_root"] = cli.run_root
    cfg = merge_lake_args(raw, lake=cli.lake)   # pick the --lake block, flatten
    # CLI file overrides win over the config keys (resolved downstream against the repo root).
    if cli.obs_file:
        cfg["obs_file"] = cli.obs_file
    if cli.perturbations_file:
        cfg["perturbations_file"] = cli.perturbations_file
    # Progress bar on/off resolved once here (CLI --no-progress > config "progress" >
    # TTY auto-detect) and carried in cfg so both engines read the same flag.
    cfg["progress"] = resolve_progress(cfg, cli.no_progress)
    # Parallelism cap: CLI --max-workers overrides the config key; both engines resolve the
    # actual worker count (auto = min(cpu, members)) from cfg via functions.resolve_max_workers.
    if cli.max_workers is not None:
        cfg["max_workers"] = cli.max_workers
    # Model selection: CLI -m wins, else the arg file's "model" field, else simstrat.
    model = cli.model or cfg.get("model") or "simstrat"
    run(cfg,
        model=model,
        skip_oda=cli.skip_oda,
        force={"copy": cli.force_copy})

"""End-to-end assimilation run, for both engines:

  1. model inputs present (a runnable setup + a warm-start state)  -> cfg["model_inputs_path"]
  2. copy the inputs into members ensemble0..N (skipped once done, unless force_copy)
     2b. make every member output the observation depths
  3. perturb the forcing of members 1..N (AR(1), time-keyed, chain state persisted)
  4-5. run the engine + write the posterior summary:
       python -> run_enkf / run_pf     openda -> run_openda

The inputs folder is only read. Everything is written under cfg["ensemble_base"] (and
cfg["openda_dir"] for OpenDA).
"""
import os
import logging

import pandas as pd

from .config import build_config
from .models import get_model
from .functions import load_obs, display_path
from .perturbate import perturbator, load_perturbations

logger = logging.getLogger(__name__)


def run(cfg, force_copy=False, skip_oda=False):
    """Run one assimilation described by `cfg` (see assimilator.config). Returns a dict of what
    was produced:

        ensemble_base     the members and the run's outputs
        mean_trajectory   the ensemble-mean trajectory file
        output_grid       the model-inputs output grid extended by the observation depths, or
                          None (written under ensemble_base; copy it into your own setup if wanted)
        cfg               the resolved run config
    """
    cfg       = build_config(cfg)
    model     = get_model(cfg["model"])
    engine    = cfg["engine"]
    lake      = cfg["lake"]
    n_members = cfg["n_members"]
    ensemble_base = cfg["ensemble_base"]
    model_inputs  = cfg["model_inputs_path"]

    logger.info(f"=== DA pipeline - lake={lake}  base={display_path(ensemble_base)}"
                f"  model={cfg['model']}  engine={engine} ===")

    # --- 1. model inputs (provided by the caller) ------------------------------------------------
    if not model.model_inputs_ready(model_inputs):
        raise FileNotFoundError(
            f"model inputs not ready at {display_path(model_inputs)}: provide a runnable setup "
            f"with a dated warm-start state (see the '{cfg['model']}' model's model_inputs_ready)")
    logger.info(f"[1/5] model inputs present -> {display_path(model_inputs)}")
    warmup = model.warmup_snapshot(model_inputs)
    if warmup:
        logger.info(f"      warm-start snapshot: {os.path.basename(warmup)}")

    # --- 2. copy into members --------------------------------------------------------------------
    if force_copy or not model.instances_ready(ensemble_base, n_members):
        why = "forced" if force_copy else "missing instances"
        logger.info(f"[2/5] copy model inputs -> ensemble0..{n_members} ({why})")
        model.copy_model_inputs(cfg)
    else:
        logger.info(f"[2/5] ensemble0..{n_members} present - skip")

    # --- 2b. align member output depths to the observations --------------------------------------
    #   Runs whether or not step 2 copied, so a skipped copy still gets the update. The inputs
    #   folder is not touched; the extended grid is returned for the caller to adopt if it wants.
    output_grid = None
    if os.path.isfile(cfg["obs_file"]):
        obs_depths = sorted(load_obs(cfg["obs_file"])["depth"].unique())
        n, output_grid = model.set_output_depths(model_inputs, ensemble_base, n_members, obs_depths)
        if n:
            logger.info(f"      member output depths <- model + obs depth superset "
                        f"({len(obs_depths)} obs depths; {n} files updated)")

    # --- 3. perturb forcings (always) ------------------------------------------------------------
    logger.info(f"[3/5] perturbate forcing in ensemble1..{n_members}")
    params = load_perturbations(cfg)   # fail fast: errors if missing or malformed
    perturbator(cfg, model, params=params)

    # --- 4-5. engine run + summary -----------------------------------------------------------------
    if engine == "python":
        from .algorithms.enkf import run_enkf
        from .algorithms.pf import run_pf
        run_cfg = dict(cfg)
        for k, v in model.run_config().items():   # model's runtime defaults (e.g. simstrat_version)
            run_cfg.setdefault(k, v)
        if cfg["algorithm"] == "EnKF":
            run_enkf(run_cfg, run_cfg, ensemble_base, n_members, model)
        else:
            run_pf(run_cfg, run_cfg, ensemble_base, n_members, model)
        mean_path = model.mean_traj_path(ensemble_base, cfg["algorithm"])
    else:
        from .openda.adapter import run_openda
        run_openda(cfg, cfg, ensemble_base, n_members, skip_oda,
                   model_cfg=model.run_config(), model_name=cfg["model"])
        mean_path = model.mean_traj_path(cfg["openda_dir"], cfg["algorithm"])

    logger.info("=== pipeline complete ===")
    return {"ensemble_base": ensemble_base, "mean_trajectory": mean_path, "output_grid": output_grid, "cfg": cfg}


def analysis_mean_state(cfg, out_path):
    """Write the ensemble-mean analysis state of a finished run (members 1..N, for either engine)
    as a warm start at `out_path` -- the deterministic forecast's initial condition."""
    cfg = build_config(cfg)
    get_model(cfg["model"]).write_mean_state(cfg, out_path)
    return out_path


def merge_mean_trajectory(cfg, dest_dir):
    """Merge this run's ensemble-mean trajectory into the rolling one in `dest_dir`, replacing rows
    with the same time. Needed for OpenDA, whose working dir holds only the latest window; the
    native engines already accumulate in ensemble_base. Returns the merged file's path."""
    cfg   = build_config(cfg)
    model = get_model(cfg["model"])
    base  = cfg["openda_dir"] if cfg["engine"] == "openda" else cfg["ensemble_base"]
    src   = model.mean_traj_path(base, cfg["algorithm"])
    dst   = model.mean_traj_path(dest_dir, cfg["algorithm"])
    if not os.path.exists(src):
        raise FileNotFoundError(f"No ensemble-mean trajectory at {src}")
    if os.path.abspath(src) == os.path.abspath(dst):
        return dst
    new = pd.read_csv(src)
    new.columns = [c.strip().strip('"') for c in new.columns]
    if os.path.exists(dst):
        prev = pd.read_csv(dst)
        prev.columns = [c.strip().strip('"') for c in prev.columns]
        if list(prev.columns) == list(new.columns):
            time_col = new.columns[0]
            new = pd.concat([prev[~prev[time_col].isin(new[time_col])], new]).sort_values(time_col)
    new.to_csv(dst, index=False)
    return dst

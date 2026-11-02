"""One assimilation run, for either engine:

  1. check the model inputs: a runnable setup with a dated warm start (only ever read)
  2. copy them into the members ensemble0..N, and make every member output the observation depths
  3. perturb the forcing of members 1..N
  4. run the engine and write its summary

Everything is written under cfg["ensemble_base"], and cfg["openda_dir"] for OpenDA.
"""
import os
import logging

import pandas as pd

from .config import build_config
from .models import get_model
from .observations import load_obs
from .perturbate import perturbate, load_perturbations
from .report import display_path

logger = logging.getLogger(__name__)


def run(cfg, force_copy=False, skip_oda=False, max_cycles=None):
    """Run the assimilation `cfg` describes (see assimilator.config). Returns:

        ensemble_base     the members and the run's outputs
        mean_trajectory   the ensemble-mean trajectory file
        output_grid       the model's output grid extended by the observation depths (under
                          ensemble_base), or None; copy it into your own setup if wanted
        analysis_time     the instant the members' final state is at (tz-aware UTC)
        cfg               the resolved run config

    With engine "openda" and an "openda_restart" block the run continues an OpenDA restart chain
    from where it ends (assimilator.openda.restart), and analysis_time is where it ends afterwards.
    `max_cycles` caps the number of OpenDA runs of a long catch-up.
    """
    cfg   = build_config(cfg)
    model = get_model(cfg["model"])
    base, inputs, n_members = cfg["ensemble_base"], cfg["model_inputs_path"], cfg["n_members"]
    logger.info(f"=== DA pipeline - lake={cfg['lake']}  base={display_path(base)}"
                f"  model={cfg['model']}  engine={cfg['engine']} ===")

    if not model.model_inputs_ready(inputs):
        raise FileNotFoundError(
            f"model inputs not ready at {display_path(inputs)}: provide a runnable setup "
            f"with a dated warm-start state (see the '{cfg['model']}' model's model_inputs_ready)")
    warmup = model.warmup_snapshot(inputs)
    logger.info(f"[1/4] model inputs {display_path(inputs)}, warm start "
                f"{os.path.basename(warmup) if warmup else 'none'}")

    if force_copy or not model.instances_ready(base, n_members):
        logger.info(f"[2/4] copy model inputs -> ensemble0..{n_members}")
        model.copy_model_inputs(cfg)
    output_grid = None
    if os.path.isfile(cfg["obs_file"]):
        obs_depths = sorted(load_obs(cfg["obs_file"])["depth"].unique())
        n, output_grid = model.set_output_depths(inputs, base, n_members, obs_depths)
        if n:
            logger.info(f"[2/4] member output depths <- model + {len(obs_depths)} observation depths "
                        f"({n} files updated)")

    params = load_perturbations(cfg)
    analysis_time = cfg["end_date"]
    if cfg["engine"] == "openda" and cfg.get("openda_restart"):
        from .openda.engine import run_openda_chain
        analysis_time = run_openda_chain(cfg, model, params, skip_oda=skip_oda, max_cycles=max_cycles)
        mean_path = model.mean_traj_path(cfg["openda_dir"], cfg["algorithm"])
    else:
        logger.info(f"[3/4] perturbate forcing in ensemble1..{n_members}")
        perturbate(cfg, model, params)
        logger.info(f"[4/4] {cfg['engine']} {cfg['algorithm']}")
        if cfg["engine"] == "python":
            from .algorithms import run_native
            run_native(cfg, model)
            mean_path = model.mean_traj_path(base, cfg["algorithm"])
        else:
            from .openda.engine import run_openda
            run_openda(cfg, model, skip_oda)
            mean_path = model.mean_traj_path(cfg["openda_dir"], cfg["algorithm"])

    logger.info("=== pipeline complete ===")
    return {"ensemble_base": base, "mean_trajectory": mean_path, "output_grid": output_grid,
            "analysis_time": analysis_time, "cfg": cfg}


def analysis_mean_state(cfg, out_path):
    """Write the ensemble-mean analysis state of a finished run (members 1..N, either engine) as a
    warm start at `out_path`: the deterministic forecast's starting state."""
    cfg = build_config(cfg)
    get_model(cfg["model"]).write_mean_state(cfg, out_path)
    return out_path


def merge_mean_trajectory(cfg, dest_dir):
    """Merge this run's ensemble-mean trajectory into the one in `dest_dir`, replacing rows with
    the same time, and return its path. Needed for a plain OpenDA run, whose working dir holds only
    that run; the native engines and the restart chain already accumulate."""
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

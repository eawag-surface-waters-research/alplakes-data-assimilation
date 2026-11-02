"""The OpenDA engine: a run of the model under OpenDA, plain or as the next run of a restart chain."""
import os
import json
import time
import shutil
import logging

from ..config import resolve_max_workers, utc
from ..containers import start_persistent_container, stop_persistent_container
from ..observations import load_obs
from ..report import log_run_header, log_run_footer
from ..sigma import sigma_obs_spec, should_season_split, sigma_obs_for_series
from ..summarize import report_summary
from . import restart as restart_chain
from .adapter import adapt, openda_binding
from .config import render, series_specs, results_filename
from .launch import openda_environment, run_oda

logger = logging.getLogger(__name__)


def openda_member_files(openda_dir, n_members, binding):
    """Members 1..N's trajectories in their work dirs: this OpenDA run only."""
    return [binding.work_output(openda_dir, i) for i in range(1, n_members + 1)]


def openda_chain_member_files(openda_dir, n_members, model, results_dir="Results"):
    """Members 1..N's trajectories over every run so far, kept as members of the OpenDA dir."""
    return [model.member_output_file(openda_dir, i, results_dir) for i in range(1, n_members + 1)]


def run_openda_chain(cfg, model, params, skip_oda=False, max_cycles=None):
    """Advance the restart chain to cfg["end_date"] and return where it then ends: from the
    chain's end (start_date for an empty chain), as one OpenDA run, or runs of at most
    cycle_max_days for a catch-up, each with its own forcing perturbation."""
    from ..perturbate import perturbate
    restart_dir = restart_chain.chain_dir(cfg, cfg["openda_dir"])
    start, end  = utc(cfg["start_date"]), utc(cfg["end_date"])
    reached     = restart_chain.chain_end(restart_dir)
    if reached is not None and reached < start:
        raise ValueError(f"[restart] the chain in {restart_dir} ends at {reached}, before start_date {start}: "
                         f"the run's warm start is newer than the chain. Empty the chain folder to start a "
                         f"new chain from the warm start.")
    cycles = restart_chain.plan_cycles(reached or start, end, restart_chain.settings(cfg)["cycle_max_days"])
    # OpenDA refuses a window without observations ("No analysisTimes found in OBSERVER").
    cycles = restart_chain.merge_empty_cycles(cycles, load_obs(cfg["obs_file"])["time"])[:max_cycles or None]
    if not cycles:
        logger.info(f"[restart] nothing to do: the chain is at {reached or start}, no observation after it up to {end}")
        return reached or start
    for i, (s, e) in enumerate(cycles, 1):
        logger.info(f"=== OpenDA run {i}/{len(cycles)} of the restart chain: {s} -> {e} ===")
        run_cfg = {**cfg, "start_date": s, "end_date": e}
        perturbate(run_cfg, model, params)
        run_openda(run_cfg, model, skip_oda)
        if skip_oda:
            return s
    return cycles[-1][1]


def run_openda(cfg, model, skip_oda=False):
    """One OpenDA run over cfg's window: prepare the working dir, render the configuration, run
    OpenDA and write the summary. With "openda_restart" it is the restart chain's next run.
    `skip_oda` stops before OpenDA starts."""
    openda_dir, n_members = cfg["openda_dir"], cfg["n_members"]
    binding = openda_binding(model)
    log_run_header(cfg)

    restart_in = restart_dir = identity = seed = None
    if cfg.get("openda_restart"):
        restart_dir = restart_chain.chain_dir(cfg, openda_dir)
        identity    = {"lake": cfg["lake"], "filter": cfg["algorithm"], "n_members": n_members}
        restart_in  = restart_chain.select_restart(restart_dir, cfg["start_date"], identity)
        seed        = restart_chain.cycle_seed(cfg.get("rng_seed", 42), cfg["end_date"])

    season_split = should_season_split(cfg)
    obs_depths, n_analysis, series_keys = adapt(cfg, binding, season_split, restart_in)
    obs_std = sigma_obs_for_series(series_keys, cfg) if season_split else sigma_obs_spec(cfg)
    if len(series_specs(obs_depths, obs_std)) != len(series_keys):
        raise ValueError(f"observation series mismatch: {len(series_specs(obs_depths, obs_std))} "
                         f"declared, {len(series_keys)} written; OpenDA would die opening the missing one")

    container = os.path.basename(openda_dir)
    run_in = binding.container(model.run_config())
    with open(os.path.join(openda_dir, "stochModel", "template", "model.json"), "w") as f:
        json.dump({"image": run_in["image"], "container": container,
                   "mount_base": run_in["mount_base"], "binary": run_in["binary"]}, f)
    oda_file = render(openda_dir, binding, cfg["algorithm"], n_members, obs_depths,
                      cfg["start_date"], cfg["end_date"], obs_std=obs_std,
                      max_threads=resolve_max_workers(cfg, n_members + 1), restart=restart_in, seed=seed)
    oda_exe, env = openda_environment(cfg)
    if skip_oda:
        logger.info(f"skip_oda: run it with: cd {openda_dir} && {oda_exe} {oda_file}")
        return

    os.makedirs(os.path.join(openda_dir, "Results"), exist_ok=True)
    if restart_dir:
        restart_chain.clear_openda_output(openda_dir, restart_dir)
    start_persistent_container(cfg, container, openda_dir, run_in["mount_base"], run_in["image"])
    t0 = time.perf_counter()
    try:
        run_oda(oda_exe, oda_file, openda_dir, env, max(1, n_analysis), cfg.get("progress", False),
                binding.model_epoch())
    finally:
        stop_persistent_container(container)
    elapsed = time.perf_counter() - t0
    log_dir = os.path.join(openda_dir, "log")
    if os.path.isfile(os.path.join(openda_dir, "openda_logfile.txt")):
        os.makedirs(log_dir, exist_ok=True)
        shutil.move(os.path.join(openda_dir, "openda_logfile.txt"), os.path.join(log_dir, "openda_logfile.txt"))

    if restart_dir:
        member_files = _append_to_chain(cfg, model, binding, restart_dir, restart_in, identity, seed)
    else:
        member_files = _keep_members(cfg, model, binding)
    out_csv, skill = report_summary("openda", cfg["algorithm"], member_files, cfg["lake"], cfg["obs_file"],
                                    openda_dir, model)
    log_run_footer(cfg, skill, n_analysis, None, elapsed, out_csv)


def _keep_members(cfg, model, binding):
    """Copy the members' trajectories out of the work dirs and write the ensemble mean."""
    openda_dir, results_dir = cfg["openda_dir"], cfg.get("results_dir", "Results")
    member_files = openda_member_files(openda_dir, cfg["n_members"], binding)
    for i, src in enumerate(member_files, start=1):
        if os.path.exists(src):
            dst = model.member_output_file(openda_dir, i, results_dir)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
    model.accumulate_mean(member_files, model.mean_traj_path(openda_dir, cfg["algorithm"]))
    return member_files


def _append_to_chain(cfg, model, binding, restart_dir, restart_in, identity, seed):
    """Archive this run's outputs (the next run overwrites the work dirs), append its member and
    mean trajectories to the chain's, and only then commit its restart, so a run that fails
    anywhere adds nothing to the chain. Returns the chain's member trajectories."""
    openda_dir, n_members = cfg["openda_dir"], cfg["n_members"]
    start_day = binding.model_time(cfg["start_date"])
    work_files = openda_member_files(openda_dir, n_members, binding)
    restart_chain.archive_cycle(openda_dir, cfg["end_date"], [
        os.path.join(openda_dir, "Results", results_filename(cfg["algorithm"])),
        os.path.join(openda_dir, "log", "openda_logfile.txt")] + [
        (os.path.join(openda_dir, "Results", f"work{k}", binding.wrapper_log), f"wrapper_work{k}.log")
        for k in range(n_members + 1)])
    chain_files = openda_chain_member_files(openda_dir, n_members, model, cfg.get("results_dir", "Results"))
    for src, dst in zip(work_files, chain_files):
        restart_chain.append_trajectory(dst, src, start_day)
    mean_path = model.mean_traj_path(openda_dir, cfg["algorithm"])
    model.accumulate_mean(work_files, mean_path + ".run")
    restart_chain.append_trajectory(mean_path, mean_path + ".run", start_day)
    os.remove(mean_path + ".run")
    restart_chain.commit_restart(openda_dir, restart_dir, cfg["end_date"], identity,
                                 extra={"start": utc(cfg["start_date"]).isoformat(), "seed": seed,
                                        "started_from": restart_in or None})
    restart_chain.prune(restart_dir, restart_chain.settings(cfg)["keep"])
    return chain_files

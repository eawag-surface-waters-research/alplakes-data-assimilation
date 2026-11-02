"""What the native engines share: their run arguments and the window loop around their update."""
import os
import time
import logging

from ..config import resolve_max_workers
from ..observations import load_obs, filter_obs_to_model_depths, log_obs_summary, obs_window_boundaries
from ..report import make_progress

logger = logging.getLogger(__name__)


def build_run_args(cfg, model):
    """The run config, under the model's runtime defaults, plus what the engines derive."""
    base = cfg["ensemble_base"]
    args = {**model.run_config(), **cfg}
    args["member_ids"]  = list(range(1, cfg["n_members"] + 1))
    args["max_workers"] = resolve_max_workers(cfg, cfg["n_members"])
    args["ref_date"]    = model.read_ref_date(base)
    args.setdefault("mean_traj_path", model.mean_traj_path(base, cfg["algorithm"]))
    if cfg["algorithm"] == "EnKF":
        args.setdefault("diag_path",        os.path.join(base, "enkf_diagnostics.csv"))
        args.setdefault("innov_depth_path", os.path.join(base, "enkf_innov_by_depth.csv"))
        args.setdefault("kgain_depth_path", os.path.join(base, "enkf_kgain_by_depth.csv"))
    return args


def reset_members(args, model, extra_files=()):
    """Start from scratch: the members' state and outputs, the mean trajectory and `extra_files`."""
    model.clear_member_state(args["ensemble_base"], args["member_ids"], args["results_dir"])
    model.clear_member_outputs(args["ensemble_base"], args["member_ids"], args["results_dir"])
    for p in [args["mean_traj_path"], *extra_files]:
        if os.path.exists(p):
            os.remove(p)
    logger.info(f"Reset: cleared {args['results_dir']}/ snapshots and trajectory files.")


def run_observations(args, model):
    """The observations the run assimilates: the obs file, binned to the hour, at the depths the
    model outputs."""
    obs = load_obs(args["obs_file"])
    obs = filter_obs_to_model_depths(obs, model.model_output_depths(args["ensemble_base"]))
    log_obs_summary(obs, args["obs_file"])
    return obs


def step_windows(args, model, obs, update):
    """Advance the members window by window, one window per observation time, calling
    update(window_start, window_end, failed_members) after each. `update` returns (updated,
    detail): whether it changed the members, and a line for the log. Writes the ensemble-mean
    trajectory at the end. Returns (windows run, windows updated)."""
    boundaries = obs_window_boundaries(obs, args["start_date"], args["end_date"])
    logger.info(f"{args['algorithm']}: {args['start_date']:%Y-%m-%d} → {args['end_date']:%Y-%m-%d} "
                f"({len(boundaries)} windows, {len(args['member_ids'])} members)")
    model.start_containers(args, max_workers=args["max_workers"])
    try:
        updated  = 0
        progress = args.get("progress", False)
        bar      = make_progress(max(1, len(boundaries)), progress, desc=f"{args['algorithm']} {args['lake']}")
        current  = args["start_date"]
        for window_end in boundaries:
            t0 = time.perf_counter()
            failed = model.run_window(current, window_end, args, max_workers=args["max_workers"])
            changed, detail = update(current, window_end, failed)
            updated += changed
            status = f"failed={failed}" if failed else "ok"
            logger.info(f"  {current:%Y-%m-%d %H:%M} -> {window_end:%Y-%m-%d %H:%M}  {detail}  [{status}]  "
                        f"[{time.perf_counter() - t0:.1f}s]", extra={"file_only": True})
            if progress:
                bar.set_postfix_str(f"{window_end:%Y-%m-%d %H:%M}  {detail}  [{status}]")
                bar.update(1)
            current = window_end
        bar.close()

        member_files = [model.member_output_file(args["ensemble_base"], i, args["results_dir"])
                        for i in args["member_ids"]]
        model.accumulate_mean(member_files, args["mean_traj_path"])
        logger.info(f"Done. {len(boundaries)} windows run, {updated} updated.")
        return len(boundaries), updated
    finally:
        model.stop_containers(args)

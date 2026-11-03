"""The native engines: the EnKF (enkf.py) and best-member selection (pf.py)."""
import time

from .common import build_run_args
from .enkf import run_enkf_loop
from .pf import run_pf_loop
from ..report import log_run_header, log_run_footer
from ..summarize import report_summary

LOOPS = {"EnKF": run_enkf_loop, "PF": run_pf_loop}


def run_native(cfg, model):
    """Run the native engine cfg["algorithm"] and write its summary and skill report to
    ensemble_base."""
    args = build_run_args(cfg, model)
    log_run_header(cfg)
    t0 = time.perf_counter()
    windows, updated = LOOPS[cfg["algorithm"]](args, model)
    member_files = [model.member_output_file(cfg["ensemble_base"], i, args["results_dir"])
                    for i in args["member_ids"]]
    out_csv, skill = report_summary("python", cfg["algorithm"], member_files, cfg["lake"], cfg["obs_file"],
                                    cfg["ensemble_base"], model)
    log_run_footer(cfg, skill, windows, updated, time.perf_counter() - t0, out_csv)

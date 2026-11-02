#!/usr/bin/env python3
"""OpenDA black-box wrapper for Simstrat: advances one member over one window.

OpenDA runs it in the member's work dir once per window. It injects OpenDA's analysed state
(temperature_state.txt) into the Simstrat snapshot and the member's perturbed forcing, runs
Simstrat to the window end, then writes the predictors (the model at each observation depth at the
window end) and the new state back for OpenDA. The engine side of this contract is
assimilator/openda/binding.py.
"""
import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time

import numpy as np

# This file lives at <openda_dir>/stochModel/bin/; the assimilator package is found through
# PYTHONPATH, set by the engine.
OPENDA_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from assimilator.models.simstrat import read_snapshot_T_at, write_snapshot_T_at   # noqa: E402
from assimilator.obs_operator import build_H                                      # noqa: E402

SNAPSHOT_FILENAME = "simulation-snapshot.dat"
STATE_FILE        = "temperature_state.txt"
# Copies of openda/adapter.py's values: set when a restart chain continues from a saved ensemble.
RESTART_MARKER   = "RESTART_EXPECTED"
RESTART_SENTINEL = -999.0

logger = logging.getLogger(__name__)


def read_time_control(path):
    match = re.search(r"\btime\s*:\s*\[([^\]]+)\]", open(path).read())
    if not match:
        raise RuntimeError(f"Could not find 'time:' array in {path}")
    start, dt, end = (float(v) for v in match.group(1).split(","))
    return start, dt, end


def read_state(path):
    with open(path) as f:
        return [float(line) for line in f if line.strip()]


def write_state(path, values):
    with open(path, "w") as f:
        f.writelines(f"{v:.6f}\n" for v in values)


def write_predictor(path, day, value):
    with open(path, "w") as f:
        f.write(f"time,value\n{day:.8f},{value:.6f}\n")      # the observation files' precision


def run_simstrat(config):
    """Run Simstrat in this work dir, inside the engine's persistent container (template/model.json),
    calling the binary directly: the image's entrypoint would cd to the mount root instead."""
    with open(os.path.join(OPENDA_DIR, "stochModel", "template", "model.json")) as f:
        container = json.load(f)
    rel = os.path.relpath(os.getcwd(), OPENDA_DIR).replace("\\", "/")
    cmd = (f"docker exec -w {container['mount_base']}/{rel} {container['container']} "
           f"{container['binary']} {config}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    logger.debug("stdout: %s", result.stdout)
    if result.returncode != 0:
        logger.error("Simstrat stderr: %s", result.stderr)
        sys.exit(result.returncode)
    logger.info("Model finished (exit %d)", result.returncode)   # counted by the engine


def main():
    parser = argparse.ArgumentParser(description="OpenDA EnKF wrapper for Simstrat")
    parser.add_argument("--config", default="Settings.par")
    parser.add_argument("--log_level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(filename="simstrat_wrapper_enkf.log", filemode="a",
                        format="%(asctime)s %(levelname)s %(message)s", level=logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))
    console.setFormatter(logging.Formatter("%(levelname)-8s %(message)s"))
    logging.getLogger().addHandler(console)
    t0 = time.perf_counter()

    start_day, _, end_day = read_time_control("time_control.yaml")
    state = read_state(STATE_FILE)
    with open(args.config) as f:
        settings = json.load(f)
    snapshot = os.path.join(settings["Output"]["Path"], SNAPSHOT_FILENAME)
    has_snapshot = os.path.exists(snapshot)
    logger.info("Window %.4f -> %.4f, %d state values, snapshot %s", start_day, end_day, len(state),
                "present" if has_snapshot else "missing")

    if os.path.exists(RESTART_MARKER) and (
            not has_snapshot or any(abs(v - RESTART_SENTINEL) < 1e-6 for v in state)):
        logger.error("[RESTART] member state not restored from the OpenDA restart - stopping")
        sys.exit(3)

    settings["Simulation"].update({"Start d": start_day, "End d": end_day, "Save text restart": False,
                                   "Use text restart": False, "Continue from last snapshot": True})
    with open(args.config, "w") as f:
        json.dump(settings, f, indent=4)

    if has_snapshot:
        try:
            write_snapshot_T_at(snapshot, args.config, np.array(state, dtype=np.float64))
        except ValueError as e:
            logger.warning("State not injected: %s", e)

    member = int(os.path.basename(os.getcwd()).replace("work", ""))
    if member > 0:      # work0 is OpenDA's main model, run on the control forcing
        shutil.copy2(os.path.join(OPENDA_DIR, "forcings", f"Forcing_{member}.dat"), "Forcing.dat")

    run_simstrat(args.config)

    # The model equivalent of an observation is the end-of-window state (OpenDA analyses at the
    # window end), read from the snapshot: a window shorter than the output interval has no
    # T_out.dat row.
    T_end, z_vol, lake_level = read_snapshot_T_at(snapshot, args.config)
    with open(os.path.join(OPENDA_DIR, "stochModel", "template", "obs_depths.json")) as f:
        obs_depths = json.load(f)
    H = build_H(z_vol, lake_level, [-d for d in obs_depths])
    for d, value in zip(obs_depths, H @ np.asarray(T_end, dtype=float)):
        write_predictor(f"T_{d:g}m.csv", end_day, value)
    write_state(STATE_FILE, T_end)
    logger.info("[TIMING] work%d | day %.4f->%.4f | %.1f s", member, start_day, end_day,
                time.perf_counter() - t0)


if __name__ == "__main__":
    main()

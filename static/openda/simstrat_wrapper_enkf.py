#!/usr/bin/env python3
"""OpenDA black-box wrapper for Simstrat — EnKF variant with full-state injection.

1. Reads the OpenDA-written time window and temperature state (temperature_state.txt)
2. Injects the analysed temperature back into the Simstrat snapshot (so OpenDA's EnKF correction
    propagates into the next model run)
3. Injects the member's perturbed forcing (Forcing_N.dat)
4. Runs Simstrat via Docker
5. Writes the model temperature at each observation depth at the END of the window, from the
   end-of-window snapshot, with the native EnKF's observation operator -> T_<depth>m.csv (the
   predictors OpenDA compares against the observation at that time). Any window length works.
6. Writes back the updated full-grid temperature state to temperature_state.txt (so OpenDA can apply the next analysis correction)

Differences from simstrat_wrapper.py:
  - At the START of each call: if temperature_state.txt contains a full-grid
    T profile (N cells, not 7 IC-depth levels), it is injected directly into
    simulation-snapshot.dat via snapshot so that OpenDA's analysis
    corrections propagate into the next model run.
  - At the END of each call: the updated snapshot is read back and the full
    T profile (all N cells) is written to temperature_state.txt for the next
    OpenDA step.  No interpolation to IC_DEPTHS.

First-run behaviour: the template temperature_state.txt has 7 values.
Injection is skipped (size mismatch guard), and after the first Simstrat run
the state file is upgraded to 576 values.  From the second call onward the
full-state injection is active.
"""

from __future__ import print_function
import argparse
import json
import logging
import numpy as np
import os
import re
import shutil
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Anchor paths off this script's location, not the cwd.  OpenDA runs the wrapper
# with cwd = the instance dir (e.g. run/openda/work_enkf/workN), whose depth can
# change, but this file always lives at <openda_dir>/stochModel/bin/:
#   bin/ -> stochModel/ -> <openda_dir> (run/openda_simstrat) -> run/ -> alplakes-data-assimilation (root)
# snapshot lives in the assimilator package (alplakes-data-assimilation/src/); perturbed
# forcings in <openda_dir>/forcings/.
# ---------------------------------------------------------------------------
_BIN_DIR    = os.path.dirname(os.path.abspath(__file__))
_OPENDA_DIR = os.path.dirname(os.path.dirname(_BIN_DIR))
_ROOT_DIR   = os.path.dirname(os.path.dirname(_OPENDA_DIR))   # run/openda_simstrat -> run -> repo root
sys.path.insert(0, os.path.join(_ROOT_DIR, "src"))
from assimilator.models.simstrat import read_snapshot_T_at, write_snapshot_T_at
from assimilator.obs_operator import build_H, OBS_OPERATOR_DEFAULT

# --- Shared persistent Simstrat container (PERF) ---------------------------
# run_openda starts ONE long-lived sleeping container for the whole run (mounting openda_dir at
# mount_base) and we `docker exec` Simstrat into it per step — instead of a fresh `docker run --rm`
# every step (container creation dominated each step's wall time). The container name, mount_base
# and binary come from the generated template/model.json (single source of truth: models.py /
# run_openda); this work dir maps to <mount_base>/<relpath-from-openda_dir> inside the mount. We
# call the binary directly rather than the image /entrypoint.sh, whose hardcoded `cd /simstrat/run`
# (= the mount root) would ignore -w — same single-container model as the native engine.

# ---------------------------------------------------------------------------
# IC depth levels (kept for legacy size detection only)
# ---------------------------------------------------------------------------
IC_DEPTHS = [0.0, -10.0, -20.0, -30.0, -40.0, -50.0, -95.0]

IC_U   = 0.0
IC_V   = 0.0
IC_S   = 0.150
IC_K   = 3.0e-6
IC_EPS = 5.0e-10

SNAPSHOT_FILENAME        = 'simulation-snapshot.dat'

# Set by the adapter only when a restart chain continues from a saved ensemble (copies of
# openda/adapter.py's values): every member must then have been restored from the restart file.
RESTART_MARKER   = 'RESTART_EXPECTED'
RESTART_SENTINEL = -999.0

# ---------------------------------------------------------------------------
# Helper functions (unchanged from simstrat_wrapper.py)
# ---------------------------------------------------------------------------

def read_time_control(yaml_file):
    with open(yaml_file) as f:
        content = f.read()
    match = re.search(r'\btime\s*:\s*\[([^\]]+)\]', content)
    if not match:
        raise RuntimeError("Could not find 'time:' array in " + yaml_file)
    values = [float(v.strip()) for v in match.group(1).split(',')]
    return values[0], values[1], values[2]


def read_state_file(state_file):
    with open(state_file) as f:
        return [float(line.strip()) for line in f if line.strip()]


def write_state_file(state_file, temperatures):
    with open(state_file, 'w') as f:
        for t in temperatures:
            f.write("{:.6f}\n".format(t))


def write_timeseries_csv(filename, times, values):
    with open(filename, 'w') as f:
        f.write("time,value\n")
        for t, v in zip(times, values):
            f.write("{:.8f},{:.6f}\n".format(t, v))   # the precision of the observation files


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="OpenDA EnKF wrapper for Simstrat")
    parser.add_argument('--config', default='Settings.par')
    parser.add_argument('--log_level', default='INFO')
    args = parser.parse_args()

    logging.basicConfig(
        filename='simstrat_wrapper_enkf.log', filemode='a',
        format='%(asctime)s %(levelname)s %(message)s', level=logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))
    console.setFormatter(logging.Formatter('%(levelname)-8s %(message)s'))
    logging.getLogger().addHandler(console)
    logger = logging.getLogger(__name__)

    _t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # 1. Read time control written by OpenDA
    # ------------------------------------------------------------------
    start_day, dt_day, end_day = read_time_control('time_control.yaml')
    logger.info("Time window: %.4f -> %.4f (dt=%.4f days)", start_day, end_day, dt_day)

    # ------------------------------------------------------------------
    # 2. Read temperature state file (injection happens after step 3)
    # ------------------------------------------------------------------
    state_file = 'temperature_state.txt'
    T_state = read_state_file(state_file)
    logger.info("[STATE] Read %d values from %s", len(T_state), state_file)

    # ------------------------------------------------------------------
    # 3. Modify Settings.par: set simulation period
    # ------------------------------------------------------------------
    with open(args.config) as f:
        settings = json.load(f)

    output_dir = settings['Output']['Path']
    snapshot_path = os.path.join(output_dir, SNAPSHOT_FILENAME)

    logger.info("cwd: %s", os.getcwd())
    snap_exists = os.path.exists(snapshot_path)
    logger.info("[RESTART] snapshot: %s  exists=%s  size=%s",
                snapshot_path, snap_exists,
                os.path.getsize(snapshot_path) if snap_exists else 'N/A')

    # Continuing a restart chain: stop if this member's state was not restored, instead of
    # silently running from the sentinel-filled template.
    if os.path.exists(RESTART_MARKER) and (
            not snap_exists or any(abs(t - RESTART_SENTINEL) < 1e-6 for t in T_state)):
        logger.error("[RESTART] member state not restored from the OpenDA restart - stopping")
        sys.exit(3)

    settings['Simulation']['Start d'] = start_day
    settings['Simulation']['End d']   = end_day
    settings['Simulation']['Save text restart']           = False
    settings['Simulation']['Use text restart']            = False
    settings['Simulation']['Continue from last snapshot'] = True

    with open(args.config, 'w') as f:
        json.dump(settings, f, indent=4)

    # ------------------------------------------------------------------
    # 3.5  Inject full-grid T state into snapshot (EnKF correction path)
    # ------------------------------------------------------------------
    if snap_exists and len(T_state) > len(IC_DEPTHS):
        try:
            write_snapshot_T_at(snapshot_path, args.config, np.array(T_state, dtype=np.float64))
            logger.info("[STATE-INJECT] Injected %d-cell T into snapshot", len(T_state))
        except ValueError as e:
            logger.warning("[STATE-INJECT] %s — skipping", e)
    else:
        logger.info("[STATE-INJECT] Skipped (first run or legacy state, %d values)",
                    len(T_state))

    # ------------------------------------------------------------------
    # 4. Inject perturbed Forcing.dat for ensemble members
    # ------------------------------------------------------------------
    work_dir_abs = os.path.abspath(os.getcwd())
    instance_num = int(os.path.basename(work_dir_abs).replace("work", ""))
    if instance_num > 0:
        forcing_src  = os.path.join(_OPENDA_DIR, "forcings", f"Forcing_{instance_num}.dat")
        if os.path.exists(forcing_src):
            shutil.copy2(forcing_src, "Forcing.dat")
            logger.info("Injected forcings/Forcing_%d.dat", instance_num)
        else:
            logger.warning("Perturbed forcing not found: %s", forcing_src)

    # ------------------------------------------------------------------
    # 5. Run Simstrat
    # ------------------------------------------------------------------
    # The container/mount_base/binary/image come from the generated template/model.json (single
    # source of truth: models.py, written by run_openda), with fallbacks for standalone runs.
    _model_file = os.path.join(_OPENDA_DIR, "stochModel", "template", "model.json")
    try:
        with open(_model_file) as f:
            _model_cfg = json.load(f)
    except FileNotFoundError:
        _model_cfg = {}
    SIMSTRAT_IMAGE = _model_cfg.get("image", "eawag/simstrat:3.0.4")
    container      = _model_cfg.get("container")
    mount_base     = _model_cfg.get("mount_base", "/simstrat/run")
    binary         = _model_cfg.get("binary", "/simstrat/build/simstrat")
    work_dir = os.path.abspath(os.getcwd()).replace("\\", "/")

    if container:
        # Exec into the shared run-wide container (started by run_openda), in this work dir's path
        # inside the mount: run_openda mounts openda_dir at mount_base, so the container path is
        # mount_base + the work dir relative to openda_dir. Binary called directly (-w selects the dir).
        rel = os.path.relpath(work_dir, _OPENDA_DIR).replace("\\", "/")
        container_workdir = f"{mount_base}/{rel}"
        logger.info("Running Simstrat in container %s -w %s", container, container_workdir)
        cmd = f"docker exec -w {container_workdir} {container} {binary} {args.config}"
    else:
        # Standalone fallback (no run_openda-managed container): one-off container per call.
        logger.info("Running Simstrat via Docker image %s in %s", SIMSTRAT_IMAGE, work_dir)
        cmd = f"docker run --rm -v {work_dir}:/simstrat/run {SIMSTRAT_IMAGE} {args.config}"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout:
        logger.debug("stdout: %s", result.stdout)
    if result.returncode != 0:
        logger.error("Simstrat stderr: %s", result.stderr)
        sys.exit(result.returncode)
    logger.info("Simstrat finished (exit %d)", result.returncode)

    # ------------------------------------------------------------------
    # 6. Predictors: the model at each observation depth at the END of the window
    # ------------------------------------------------------------------
    # OpenDA analyses at observation times and each window ends at one, so the model equivalent is
    # the end-of-window state -- as in the native EnKF, with the same observation operator. Read
    # from the snapshot, not T_out.dat: a window shorter than the output interval has no T_out row,
    # but always has an end state.
    T_end, z_vol, lake_level = read_snapshot_T_at(snapshot_path, args.config)

    # Depths to extract come from the generated obs_depths.json (single source of
    # truth shared with the model config + formatters).
    with open(os.path.join(_OPENDA_DIR, "stochModel", "template", "obs_depths.json")) as f:
        obs_depths = json.load(f)
    operator = _model_cfg.get("obs_operator", OBS_OPERATOR_DEFAULT)
    H = build_H(z_vol, lake_level, [-d for d in obs_depths], operator=operator)
    for d, value in zip(obs_depths, H @ np.asarray(T_end, dtype=float)):
        write_timeseries_csv(f"T_{d:g}m.csv", [end_day], [value])
    logger.info("Wrote %d predictors at day %.8f (%s operator, end-of-window state)",
                len(obs_depths), end_day, operator)

    # ------------------------------------------------------------------
    # 7. Update temperature state: full grid from snapshot (no interpolation)
    # ------------------------------------------------------------------
    write_state_file(state_file, list(T_end))
    logger.info("temperature_state.txt updated (%d cells) from snapshot", len(T_end))

    elapsed = time.perf_counter() - _t0
    instance_label = os.path.basename(os.path.abspath(os.getcwd()))
    logger.info("[TIMING] %s | day %.4f->%.4f | %.1f s", instance_label, start_day, end_day, elapsed)
    logger.info("Simstrat EnKF wrapper completed successfully")

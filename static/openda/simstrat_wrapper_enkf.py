#!/usr/bin/env python3
"""OpenDA black-box wrapper for Simstrat — EnKF variant with full-state injection.

1. Reads the OpenDA-written time window and temperature state (temperature_state.txt)
2. Injects the analysed temperature back into the Simstrat snapshot (so OpenDA's EnKF correction
    propagates into the next model run)
3. Injects the member's perturbed forcing (Forcing_N.dat)
4. Runs Simstrat via Docker
5. Extracts temperatures at the 15 observation depths from T_out.dat → writes T_1m.csv … T_40m.csv (the predictors OpenDA reads to compare against observations)
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
from assimilator.models.simstrat import (read_snapshot_T_at, write_snapshot_T_at,
                                         read_t_out_tail, T_OUT_OFFSET_FILE)

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


# read_t_out_tail + T_OUT_OFFSET_FILE now live in assimilator.models.simstrat (imported above),
# shared with the native PF engine (load_T_window) so both engines read T_out.dat the same way.


def find_depth_col(depths, target_depth):
    return min(range(len(depths)), key=lambda i: abs(depths[i] - target_depth))


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
            f.write("{:.4f},{:.6f}\n".format(t, v))


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
    # 6. Read T_out.dat and write predictor CSV files
    # ------------------------------------------------------------------
    t_out_file = os.path.join(output_dir, 'T_out.dat')
    # Tail read: only the rows Simstrat appended for THIS window are read (resuming from the byte
    # offset stored next to T_out.dat), instead of re-parsing the whole growing file each step. The
    # full cumulative T_out.dat is left intact for the summary/plots; only this per-step read and the
    # predictor CSVs below see the current window. Window bounds passed as a safety net so a fallback
    # full re-read (fresh run / reused instance dir) still yields just the current window. See step 1
    # for start_day/end_day; the read is independent of the Simstrat output interval and window length.
    _t_read = time.perf_counter()
    offset_path = os.path.join(output_dir, T_OUT_OFFSET_FILE)
    times, depths, T_rows = read_t_out_tail(t_out_file, offset_path,
                                            window_start=start_day, window_end=end_day)
    logger.info("[TAIL-READ] T_out.dat: %d new rows (window %.4f->%.4f), %.2f s",
                len(times), start_day, end_day, time.perf_counter() - _t_read)

    # Depths to extract come from the generated obs_depths.json (single source of
    # truth shared with the model config + formatters); fall back to the legacy set.
    _depths_file = os.path.join(_OPENDA_DIR, "stochModel", "template", "obs_depths.json")
    try:
        with open(_depths_file) as f:
            obs_depths = json.load(f)
    except FileNotFoundError:
        obs_depths = [1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.0, 17.0, 19.0, 21.0, 25.0, 30.0, 35.0, 40.0]
    obs_specs = [(f"T_{d:g}m.csv", -d) for d in obs_depths]
    for csv_name, target_depth in obs_specs:
        col = find_depth_col(depths, target_depth)
        values = [row[col] for row in T_rows]
        write_timeseries_csv(csv_name, times, values)
        logger.info("Written %s (%d timesteps, depth %.1f m)", csv_name, len(times), target_depth)

    # ------------------------------------------------------------------
    # 7. Update temperature state: full grid from snapshot (no interpolation)
    # ------------------------------------------------------------------
    new_T = list(read_snapshot_T_at(snapshot_path, args.config)[0])
    write_state_file(state_file, new_T)
    logger.info("temperature_state.txt updated (%d cells) from snapshot", len(new_T))

    elapsed = time.perf_counter() - _t0
    instance_label = os.path.basename(os.path.abspath(os.getcwd()))
    logger.info("[TIMING] %s | day %.4f->%.4f | %.1f s", instance_label, start_day, end_day, elapsed)
    logger.info("Simstrat EnKF wrapper completed successfully")

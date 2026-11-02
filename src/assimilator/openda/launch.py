"""Running OpenDA: its environment, and oda_run.sh with its log tailed for progress and failures.

oda_run.sh sends all of OpenDA's output to openda_logfile.txt, so the log file is the only place to
follow a run. A few lines are lifted into the pipeline log (the run's set-up, one line per analysis
step with its member count, a failed member, the end); the rest stays in log/openda_logfile.txt.
"""
import os
import re
import time
import logging
import subprocess
from datetime import timedelta

from ..report import make_progress

logger = logging.getLogger(__name__)

SRC = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_FORECAST_MARK = b"Forecast from"
_FORECAST_DAYS = re.compile(r"\(([\d.]+)\s*-+>\s*([\d.]+)\)")   # the "(A-->B)" model-day span


def openda_environment(cfg):
    """(oda_run.sh path, environment) for cfg["openda_bin"], the bin/ folder of an OpenDA
    installation; without it, an OpenDA environment already set up in the shell is used. The
    wrapper processes OpenDA starts inherit PYTHONPATH, so they import this package wherever the
    working dir is."""
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join([SRC, env.get("PYTHONPATH", "")])
    openda_bin = cfg.get("openda_bin")
    if not openda_bin:
        return "oda_run.sh", env
    openda_bin = os.path.expanduser(openda_bin)
    native     = cfg.get("openda_native", "linux64_gnu")
    openda_lib = os.path.join(openda_bin, native)
    env.update({"OPENDADIR": openda_bin, "OPENDA_NATIVE": native, "OPENDALIB": openda_lib,
                "PATH": os.pathsep.join([os.path.join(os.path.dirname(openda_bin), "jre", "bin"),
                                         openda_bin, env.get("PATH", "")]),
                "LD_LIBRARY_PATH": os.pathsep.join([os.path.join(openda_lib, "lib"),
                                                    env.get("LD_LIBRARY_PATH", "")])})
    return os.path.join(openda_bin, "oda_run.sh"), env


def _reformat_forecast(line, epoch):
    """OpenDA's Forecast line dates the model days against the MJD epoch; restate the "(A-->B)"
    day span as calendar dates from the model's `epoch`."""
    m = _FORECAST_DAYS.search(line)
    if not m:
        return line
    t0 = epoch + timedelta(days=float(m.group(1)))
    t1 = epoch + timedelta(days=float(m.group(2)))
    return f"Forecast {t0:%Y-%m-%d %H:%M} -> {t1:%Y-%m-%d %H:%M} (sim-day {m.group(1)}->{m.group(2)})"


def run_oda(oda_exe, oda_file, openda_dir, env, total, progress, epoch):
    """Run oda_run.sh in `openda_dir`, following its log file. Raises if it exits non-zero or if
    OpenDA dies with a Java exception (oda_run.sh still exits 0 then)."""
    logfile = os.path.join(openda_dir, "openda_logfile.txt")
    if os.path.exists(logfile):
        os.remove(logfile)
    try:
        proc = subprocess.Popen([oda_exe, oda_file], cwd=openda_dir, env=env)
    except FileNotFoundError:
        raise RuntimeError("oda_run.sh not found — set \"openda_bin\" to the OpenDA bin dir, "
                           "or source the OpenDA environment so oda_run.sh is on PATH")
    bar = make_progress(total, progress, desc="OpenDA")
    pos, instances, crash = 0, 0, []
    # A step's members run after its Forecast line, so its tally is complete only at the next one.
    pending, step_ok, step_fail = None, 0, 0

    def flush():
        nonlocal pending, step_ok, step_fail
        if pending is not None:
            n = step_ok + step_fail
            tally = f"instances={step_ok}" + (f"/{n} ({step_fail} failed)" if step_fail else "")
            logger.info(f"{pending}  {tally}", extra={"file_only": True})
        pending, step_ok, step_fail = None, 0, 0

    try:
        while True:
            done = proc.poll() is not None
            if os.path.exists(logfile):
                with open(logfile, "rb") as fh:
                    fh.seek(pos)
                    data = fh.read()
                complete = data[:data.rfind(b"\n") + 1]     # never split a line across two reads
                pos += len(complete)
                for raw in complete.split(b"\n"):
                    line = raw.decode("utf-8", "replace").strip()
                    if _FORECAST_MARK in raw:
                        bar.update(1)
                        flush()
                        pending = _reformat_forecast(line, epoch)
                    elif b"Model finished (exit 0)" in raw:
                        step_ok += 1
                    elif b"Model finished (exit" in raw:
                        # OpenDA itself carries on and ends with "Application Done", so say it here.
                        step_fail += 1
                        logger.warning("OpenDA member run failed: " + line)
                    elif b"Creating model instance" in raw:
                        instances += 1
                    elif b"Application initializing finished" in raw:
                        logger.info(f"OpenDA: initialized ({instances} model instances)", extra={"file_only": True})
                    elif b"OpenDA version" in raw or b"className:" in raw or b"Selected localization method" in raw:
                        logger.info("OpenDA: " + line, extra={"file_only": True})
                    elif b"Exception in thread" in raw or (crash and b"Caused by:" in raw):
                        crash.append(line)
                    elif b"Application Done" in raw:
                        flush()
                        logger.info("OpenDA: application done", extra={"file_only": True})
            if done:
                break
            time.sleep(0.5)
        flush()
    finally:
        bar.close()

    if proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, [oda_exe, oda_file])
    if crash:
        raise RuntimeError("OpenDA failed: " + " | ".join(crash[-2:]) + " (see log/openda_logfile.txt)")

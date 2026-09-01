"""Spin up a warm-start snapshot for each lake that lacks one.

The DA pipeline (src/main.py step 1) refuses to run a lake until
inputs/<lake>/simulation-snapshot_<YYYYMMDD>.dat exists — the state Simstrat
restarts every ensemble member from. upperlugano ships with one; the datalakes
lakes do not. This runs a plain (non-ensemble) Simstrat simulation from --start
to --end so Simstrat writes that snapshot, then installs it under the dated name
main.py looks for.

Mechanism (mirrors what the pipeline does per window, but once, from cold):
  1. Write inputs/<lake>/Settings_spinup.par = Settings.par with Start d / End d
     set to the window (day-counts from the lake's own "Reference year").
  2. Delete any stale Results/simulation-snapshot.dat so Simstrat starts from
     InitialConditions.dat ("Continue from last snapshot" is true, but with no
     file present it falls back to IC and still writes a snapshot at the end).
  3. docker run eawag/simstrat:<ver> over the mounted lake dir.
  4. Copy Results/simulation-snapshot.dat -> inputs/<lake>/simulation-snapshot_<end>.dat.

    python notebooks/spinup.py                      # all 6 datalakes lakes, 2024-01-01 -> 2025-01-01
    python notebooks/spinup.py aegeri               # one lake
    python notebooks/spinup.py --start 2023-01-01   # longer spin-up
    python notebooks/spinup.py --dry-run            # print the docker commands, run nothing

NOTE: needs Docker + the eawag/simstrat image on the machine you run this on.
"""
import os
import sys
import json
import shutil
import argparse
import subprocess
import concurrent.futures
from datetime import datetime, timedelta

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from assimilator.models.simstrat import Simstrat, datetime_to_simstrat_time  # noqa: E402
from assimilator.functions import ROOT  # noqa: E402

# Lakes with model inputs but no committed snapshot (see readiness matrix).
DEFAULT_LAKES = ["aegeri", "geneva", "greifensee", "hallwil", "maggiore", "murten"]


def forcing_start(inp, par, ref):
    """The first timestamp in this lake's Forcing.dat (col 1 = days since ref), as a datetime,
    or None if unreadable. Simstrat cannot start before its forcing begins."""
    fname = par.get("Input", {}).get("Forcing", "Forcing.dat")
    try:
        with open(os.path.join(inp, fname)) as f:
            f.readline()                       # header row
            first_day = float(f.readline().split()[0])
        return ref + timedelta(days=first_day)
    except (OSError, ValueError, IndexError):
        return None


def spinup_one(lake, start, end, image, dry_run):
    """Spin up one lake. Returns (ok, log) — output is buffered, not printed, so parallel
    lakes don't interleave; the caller prints each block when the lake finishes."""
    log = []
    inp = os.path.join(ROOT, "inputs", lake)
    settings = os.path.join(inp, "Settings.par")
    if not os.path.isfile(settings):
        return False, f"{lake:>12}: no inputs/{lake}/Settings.par - skipped"

    with open(settings) as f:
        par = json.load(f)
    ref = datetime(par["Simulation"]["Reference year"], 1, 1)

    # A full 1981 spin-up is only possible where the forcing reaches that far back; clamp the
    # start to when this lake's Forcing.dat actually begins (aegeri's starts ~2012).
    start_lake = start
    fstart = forcing_start(inp, par, ref)
    if fstart is not None and fstart > start_lake:
        log.append(f"{lake:>12}: forcing starts {fstart.date()} - "
                   f"start clamped from {start.date()}")
        start_lake = fstart

    par["Simulation"]["Start d"] = datetime_to_simstrat_time(start_lake, ref)
    par["Simulation"]["End d"]   = datetime_to_simstrat_time(end, ref)
    results = par["Output"].get("Path", "Results")

    spin_par = "Settings_spinup.par"
    with open(os.path.join(inp, spin_par), "w") as f:
        json.dump(par, f, indent=4)

    # Clean cold start: wipe Results/ entirely (Simstrat appends to *_out.dat across runs and
    # would otherwise also read a stale snapshot), then recreate it empty for fresh output.
    results_dir = os.path.join(inp, results)
    if not dry_run:
        shutil.rmtree(results_dir, ignore_errors=True)
        os.makedirs(results_dir, exist_ok=True)

    mount = os.path.abspath(inp).replace("\\", "/")
    cmd = ["docker", "run", "--rm", "-v", f"{mount}:{Simstrat.workdir}", image, spin_par]
    log.append(f"{lake:>12}: {start_lake.date()} -> {end.date()}  "
               f"(Start d={par['Simulation']['Start d']:.1f}, "
               f"End d={par['Simulation']['End d']:.1f})")
    log.append(f"{'':>14}$ {' '.join(cmd)}")
    if dry_run:
        return True, "\n".join(log)

    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        log.append(f"{'':>14}FAILED (exit {res.returncode}):\n{res.stderr[-800:]}")
        return False, "\n".join(log)

    snap = os.path.join(inp, results, "simulation-snapshot.dat")
    if not os.path.isfile(snap):
        log.append(f"{'':>14}run finished but no snapshot at "
                   f"{os.path.relpath(snap, ROOT)} - check output")
        return False, "\n".join(log)
    dated = os.path.join(inp, f"simulation-snapshot_{end.strftime('%Y%m%d')}.dat")
    shutil.copy2(snap, dated)
    log.append(f"{'':>14}wrote {os.path.relpath(dated, ROOT)}")
    return True, "\n".join(log)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lakes", nargs="*", default=DEFAULT_LAKES,
                    help=f"Lakes to spin up (default: {', '.join(DEFAULT_LAKES)})")
    ap.add_argument("--start", default="1981-01-01",
                    help="Spin-up start (UTC midnight); per lake it is clamped to when that "
                         "lake's forcing begins. Default 1981-01-01 = full cold spin-up.")
    ap.add_argument("--end", default="2025-01-01",
                    help="Spin-up end = snapshot date (UTC midnight); must match the DA start_date")
    ap.add_argument("--image", default=f"{Simstrat.image}:{Simstrat.version}",
                    help="Simstrat Docker image (default from the Simstrat model class)")
    ap.add_argument("--jobs", "-j", type=int, default=4,
                    help="Lakes to spin up in parallel (each is its own container; default 4). "
                         "Every container is single-threaded Simstrat, so keep <= CPU cores.")
    ap.add_argument("--dry-run", action="store_true", help="Print docker commands, run nothing")
    args = ap.parse_args()

    start = datetime.fromisoformat(args.start)
    end   = datetime.fromisoformat(args.end)
    lakes = args.lakes or DEFAULT_LAKES
    workers = max(1, min(args.jobs, len(lakes)))

    print(f"Spin-up {start.date()} -> {end.date()}  image={args.image}  "
          f"({workers} in parallel)\n")
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(spinup_one, lk, start, end, args.image, args.dry_run): lk
                   for lk in lakes}
        for fut in concurrent.futures.as_completed(futures):
            ok, block = fut.result()
            results[futures[fut]] = ok
            print(block, flush=True)

    n_ok = sum(1 for ok in results.values() if ok)
    failed = [lk for lk in lakes if not results.get(lk)]
    print(f"\n{n_ok}/{len(lakes)} lakes "
          f"{'prepared (dry-run)' if args.dry_run else 'spun up'}"
          f"{'' if not failed else '  |  failed: ' + ', '.join(failed)}")
    sys.exit(0 if n_ok == len(lakes) else 1)


if __name__ == "__main__":
    main()

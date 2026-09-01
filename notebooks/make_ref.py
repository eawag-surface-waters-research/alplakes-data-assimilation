"""Create the free-run reference T_out.dat for each lake — the baseline the DA plots compare
against (notebooks/visualize.py load_e0 looks for inputs/<lake>/ref/T_out.dat first).

A free run is a plain Simstrat simulation over the assimilation window, warm-started from the
spin-up snapshot (simulation-snapshot_<date>.dat) but NOT perturbed and NOT assimilated. This
runs it per lake (in parallel, like spinup.py) with Output Path = ref/, so T_out.dat lands in
inputs/<lake>/ref/ exactly where load_e0 expects the reference.

    python notebooks/make_ref.py                     # 6 lakes, 2025-01-01 -> 2026-01-01
    python notebooks/make_ref.py aegeri
    python notebooks/make_ref.py --start 2025-01-01 --end 2026-01-01 -j 6
    python notebooks/make_ref.py --dry-run

Prereq: the spin-up snapshot must exist (run notebooks/spinup.py first). Needs Docker + the
eawag/simstrat image; run it in the WSL env, like spinup.py. The window should match the DA run
(run_*.json start_date .. end_date + 1, since OpenDA runs one day past end_date).
"""
import os
import sys
import glob
import json
import shutil
import argparse
import subprocess
import concurrent.futures
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from assimilator.models.simstrat import Simstrat, datetime_to_simstrat_time  # noqa: E402
from assimilator.functions import ROOT  # noqa: E402

DEFAULT_LAKES = ["aegeri", "geneva", "greifensee", "hallwil", "maggiore", "murten"]
REF_DIR = "ref"   # Simstrat Output Path -> inputs/<lake>/ref/ (where load_e0 looks first)


def make_ref_one(lake, start, end, image, dry_run):
    """Run one lake's free reference. Returns (ok, log); output buffered for clean parallel logs."""
    log = []
    inp = os.path.join(ROOT, "inputs", lake)
    settings = os.path.join(inp, "Settings.par")
    if not os.path.isfile(settings):
        return False, f"{lake:>12}: no inputs/{lake}/Settings.par - skipped"
    snaps = sorted(glob.glob(os.path.join(inp, "simulation-snapshot_*.dat")))
    if not snaps:
        return False, f"{lake:>12}: no simulation-snapshot_*.dat - run notebooks/spinup.py first"
    dated = snaps[-1]

    with open(settings) as f:
        par = json.load(f)
    ref = datetime(par["Simulation"]["Reference year"], 1, 1)
    par["Simulation"]["Start d"] = datetime_to_simstrat_time(start, ref)
    par["Simulation"]["End d"]   = datetime_to_simstrat_time(end, ref)
    # Warm start: read the snapshot (seeded below), then run forward unperturbed.
    par["Simulation"]["Continue from last snapshot"] = True
    par["Output"]["Path"] = REF_DIR

    ref_par = "Settings_ref.par"
    with open(os.path.join(inp, ref_par), "w") as f:
        json.dump(par, f, indent=4)

    ref_dir = os.path.join(inp, REF_DIR)
    if not dry_run:
        shutil.rmtree(ref_dir, ignore_errors=True)   # clean: Simstrat appends to *_out.dat
        os.makedirs(ref_dir, exist_ok=True)
        # Seed the warm-start state so the free run begins from the DA start, not InitialConditions.
        shutil.copy2(dated, os.path.join(ref_dir, "simulation-snapshot.dat"))

    mount = os.path.abspath(inp).replace("\\", "/")
    cmd = ["docker", "run", "--rm", "-v", f"{mount}:{Simstrat.workdir}", image, ref_par]
    log.append(f"{lake:>12}: {start.date()} -> {end.date()}  warm-start {os.path.basename(dated)}")
    log.append(f"{'':>14}$ {' '.join(cmd)}")
    if dry_run:
        return True, "\n".join(log)

    res = subprocess.run(cmd, capture_output=True, text=True)
    tout = os.path.join(ref_dir, "T_out.dat")
    if res.returncode != 0:
        log.append(f"{'':>14}FAILED (exit {res.returncode}):\n{res.stderr[-800:]}")
        return False, "\n".join(log)
    if not os.path.isfile(tout):
        log.append(f"{'':>14}run finished but no {os.path.relpath(tout, ROOT)} - check output")
        return False, "\n".join(log)
    log.append(f"{'':>14}wrote {os.path.relpath(tout, ROOT)}")
    return True, "\n".join(log)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("lakes", nargs="*", default=DEFAULT_LAKES,
                    help=f"Lakes (default: {', '.join(DEFAULT_LAKES)})")
    ap.add_argument("--start", default="2025-01-01", help="Free-run start (UTC midnight)")
    ap.add_argument("--end", default="2026-01-01",
                    help="Free-run end (UTC midnight); default = DA end_date + 1 day")
    ap.add_argument("--image", default=f"{Simstrat.image}:{Simstrat.version}",
                    help="Simstrat Docker image")
    ap.add_argument("--jobs", "-j", type=int, default=4,
                    help="Lakes in parallel (each its own container; keep <= CPU cores)")
    ap.add_argument("--dry-run", action="store_true", help="Print docker commands, run nothing")
    args = ap.parse_args()

    start = datetime.fromisoformat(args.start)
    end   = datetime.fromisoformat(args.end)
    lakes = args.lakes or DEFAULT_LAKES
    workers = max(1, min(args.jobs, len(lakes)))

    print(f"Free-run reference {start.date()} -> {end.date()}  image={args.image}  "
          f"({workers} in parallel)\n")
    results = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(make_ref_one, lk, start, end, args.image, args.dry_run): lk
                   for lk in lakes}
        for fut in concurrent.futures.as_completed(futures):
            ok, block = fut.result()
            results[futures[fut]] = ok
            print(block, flush=True)

    n_ok = sum(1 for ok in results.values() if ok)
    failed = [lk for lk in lakes if not results.get(lk)]
    print(f"\n{n_ok}/{len(lakes)} references "
          f"{'prepared (dry-run)' if args.dry_run else 'built'}"
          f"{'' if not failed else '  |  failed: ' + ', '.join(failed)}")
    sys.exit(0 if n_ok == len(lakes) else 1)


if __name__ == "__main__":
    main()

"""Per-depth localization radius from a free ensemble run -> localization/<lake>.json.

WHY LOCALIZE. With N=20 members the ensemble spans 19 directions, so a correlation below roughly
2/sqrt(N) = 0.45 cannot be told from sampling noise.
WHY A FREE ENSEMBLE RUN. The quantity localization has to judge is the ensemble's own forecast-error
correlation: the thing the Kalman gain multiplies. A free run -- perturbed forcing, no analysis --
shows it undistorted. 
WHAT. Ensemble anomalies (each member minus the ensemble mean, at every time) are correlated between
model depths, pooled over the stratified season. For each depth the RADIUS is the smallest separation
at which that correlation first falls below THRESHOLD. That radius is the SUPPORT of the
Gaspari-Cohn taper applied in assimilator/localization.py -- weight 1 at zero separation, 0 at the
radius -- so this table is unchanged by the move from a binary mask to the taper; only how the
number is used changed.

    no fit         no L0, no slope, no functional form imposed on the radius profile
    one knob       THRESHOLD

STRATIFIED SEASON ONLY. A mixed column is coherent top to bottom, so its correlations reach much
further and its radius is much wider. Adopting the stratified radius year-round therefore localizes
HARDER than winter physics requires -- it discards some real information in a mixed column, but it
never lets a spurious correlation through. That is the safe direction, and stratification is when
the column is decoupled and localization is actually doing work.
APPLIED TO BOTH SIDES. See assimilator/localization.py: the taper multiplies PHT and HPHT alike,
which a binary mask could not do because it is not positive semi-definite.
WHAT THIS CANNOT DO. Deep cells have no observation anywhere near them, so their radius says only
"nothing reaches here" -- which is the intended behaviour, not a measurement. Expect the deep water
to free-run.
THE FREE RUN lives at localization/free_run/<lake>. It is a scratch artefact, not an experiment:
`--generate` builds it, reads the correlations off it, and deletes it again, so nothing large is
left behind and the table is always traceable to a run made the same way. `--keep` leaves it in
place when you want to look.
Generating one runs Simstrat in Docker against the reanalysis forcing, so it needs the VPN. Without
`--generate` the script only reads what is already there and tells you this if it is not.

Usage:
    # first time, on the VPN: run the free ensemble, derive, clean up
    python notebooks/localization.py args/run_enkf.json --lake upperlugano --generate

    # afterwards, or against a run you staged yourself
    python notebooks/localization.py args/run_enkf.json --lake upperlugano --run-dir <dir>
"""
import os
import sys
import json
import shutil
import logging
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from assimilator.functions import (ROOT, merge_lake_args, build_python_run_args,  # noqa: E402
                                   STRATIFIED_MONTHS)
from assimilator.models import get_model                                         # noqa: E402
from assimilator.models.simstrat import SIMSTRAT_REF_YEAR                        # noqa: E402
from assimilator.perturbate import perturbator, load_perturbations               # noqa: E402

logger = logging.getLogger(__name__)

FREE_RUN_ROOT = os.path.join("localization", "free_run")   # <lake> beneath it
THRESHOLD_DEFAULT = 0.45     # 2/sqrt(20): below this an N=20 correlation is sampling noise
# STRATIFIED_MONTHS (May-Oct) comes from assimilator.functions — the single definition. The radius
# is read from the stratified season alone; this is the only calendar rule here.
MIN_STEPS = 100              # timesteps of stratified record below which the correlation is
                             # not worth reading


def member_files(run_dir, n_members, results_dir):
    """The per-member T_out.dat paths under a run directory.

    Member 0 is the unperturbed control and is excluded: it carries no ensemble anomaly.
    """
    files = [os.path.join(run_dir, f"ensemble{i}", results_dir, "T_out.dat")
             for i in range(1, n_members + 1)]
    missing = [f for f in files if not os.path.isfile(f)]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} of {n_members} member outputs missing under {run_dir}, e.g. "
            f"{os.path.relpath(missing[0], ROOT)}")
    return files


def load_ensemble(files):
    """Stack member T_out.dat into (times, depths, arr[member, time, depth]).

    T_out.dat is "Datetime" (Simstrat day) then one column per depth, negative downward; depths come
    back positive and ascending so every index below reads as a depth in metres.
    """
    def read(path):
        df = pd.read_csv(path)
        t = df.iloc[:, 0].to_numpy(dtype=float)
        z = np.array([abs(float(c)) for c in df.columns[1:]])
        return t, z, df.iloc[:, 1:].to_numpy(dtype=float)

    times, depths, first = read(files[0])
    members = [first] + [read(f)[2] for f in files[1:]]
    n = min(m.shape[0] for m in members)              # share cadence; truncate defensively
    arr = np.stack([m[:n] for m in members])          # [member, time, depth]

    order = np.argsort(depths)
    arr, depths = arr[:, :, order], depths[order]
    ref = pd.Timestamp(f"{SIMSTRAT_REF_YEAR}-01-01", tz="UTC")
    return ref + pd.to_timedelta(times[:n], unit="D"), depths, arr


def anomaly_correlation(arr, times):
    """Correlation of ensemble ANOMALIES between depths, over the stratified season.

    The anomaly is each member minus the ensemble mean at that time, which is exactly what the
    forecast error covariance P is built from -- so this is the correlation the Kalman gain uses,
    measured rather than assumed. Pooled over (member, time): one matrix, one radius per depth.
    """
    sel = np.isin(times.month, STRATIFIED_MONTHS)
    if sel.sum() < MIN_STEPS:
        raise ValueError(f"only {sel.sum()} stratified timesteps in this run (need {MIN_STEPS}) — "
                         f"does it cover May-Oct?")
    a = arr[:, sel, :]
    a = a - a.mean(axis=0, keepdims=True)             # anomaly about the ensemble mean
    a = a.reshape(-1, a.shape[2])                     # pool (member, time)
    logger.info(f"  {a.shape[0]:,} (member, time) anomalies over {sel.sum():,} stratified steps")

    a = a - a.mean(axis=0, keepdims=True)
    sd = a.std(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        C = (a.T @ a) / (len(a) - 1) / np.outer(sd, sd)
    return np.clip(np.nan_to_num(C, nan=0.0), -1.0, 1.0)


def radii(corr, depths, threshold):
    """Per depth, the smallest separation at which the correlation first drops below `threshold`.

    Walk outward from each depth over the sorted grid and stop at the first neighbour below the
    threshold; that separation is the radius, i.e. the taper's support. A depth whose
    correlation never drops gets the largest separation available, i.e. the whole column -- honest,
    since the grid cannot show a crossing that is not there.
    """
    out = np.empty(len(depths))
    for i, z in enumerate(depths):
        sep = np.abs(depths - z)
        order = np.argsort(sep)[1:]                   # nearest first, skipping itself
        below = [j for j in order if corr[i, j] < threshold]
        out[i] = sep[below[0]] if below else sep.max()
    return out


def generate_free_run(cfg, run_dir, model="simstrat"):
    """Stage and run a free ensemble into `run_dir`: copy inputs, perturbate, integrate once."
    """
    model_obj = get_model(model)
    args_cfg  = dict(cfg)
    args_cfg["ensemble_base"] = run_dir
    for k, v in model_obj.run_config().items():
        args_cfg.setdefault(k, v)

    n_members  = int(args_cfg["n_members"])
    logger.info(f"  staging {n_members} members -> {os.path.relpath(run_dir, ROOT)}")
    model_obj.copy_model_inputs(args_cfg)
    perturbator(args_cfg, params=load_perturbations(args_cfg))

    args  = build_python_run_args(args_cfg, args_cfg, run_dir, n_members, model_obj)
    start, end = args["start_date"], args["end_date"]
    logger.info(f"  integrating {start.date()} -> {end.date()} in ONE window (no analysis)")

    model_obj.start_containers(args, max_workers=args.get("max_workers"))
    try:
        failed = model_obj.run_window(start, end, args, max_workers=args.get("max_workers"))
    finally:
        model_obj.stop_containers(args)
    if failed:
        raise RuntimeError(f"{len(failed)} member(s) failed to run: {sorted(failed)}")
    logger.info(f"  free run complete, {n_members} members")


def derive_lake(cfg, run_dir, threshold=THRESHOLD_DEFAULT, dry_run=False, root=ROOT):
    """Derive one lake's per-depth radii from a free ensemble run and write the json."""
    lake = cfg["lake"]
    logger.info(f"=== {lake} ===")

    files = member_files(run_dir, int(cfg["n_members"]), cfg.get("results_dir", "Results_EnKF"))
    times, depths, arr = load_ensemble(files)
    logger.info(f"  {arr.shape[0]} members x {arr.shape[1]:,} times x {len(depths)} depths "
                f"({times.min():%Y-%m-%d}..{times.max():%Y-%m-%d})")

    corr = anomaly_correlation(arr, times)
    L    = radii(corr, depths, threshold)
    logger.info(f"  radius {L.min():.1f}-{L.max():.1f} m over {depths.min():.1f}-{depths.max():.1f} m "
                f"of column, at threshold {threshold:g}")

    step = max(1, len(depths) // 12)
    logger.info("  radius by depth: "
                + "  ".join(f"{depths[i]:g}m {L[i]:.0f}" for i in range(0, len(depths), step)))

    out = {
        "lake": lake,
        "derived_on": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "derived_by": "notebooks/localization.py (free-ensemble correlation, Gaspari-Cohn support)",
        "source_run": os.path.relpath(run_dir, root).replace(os.sep, "/"),
        "threshold": threshold,
        "season": "stratified only (months " + ",".join(str(m) for m in STRATIFIED_MONTHS) + ")",
        "n_members": int(arr.shape[0]),
        "depths_m": [round(float(z), 3) for z in depths],
        "radius_m": [round(float(r), 3) for r in L],
    }

    dest = os.path.join(root, "localization", f"{lake}.json")
    if dry_run:
        logger.info(f"  --dry-run: would write {os.path.relpath(dest, root)}")
        return out
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    logger.info(f"  wrote {os.path.relpath(dest, root)}")
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Per-depth localization radius from a free ensemble run")
    ap.add_argument("arg_file", help="Path to a run config JSON (e.g. args/run_enkf.json)")
    ap.add_argument("--lake", required=True, help="Lake from the config's \"lakes\" block")
    ap.add_argument("--generate", action="store_true",
                    help="run the free ensemble first (Docker + VPN), derive from it, then delete "
                         "it again. Without this the script only reads an existing free run.")
    ap.add_argument("--keep", action="store_true",
                    help="with --generate, leave the free run on disk instead of deleting it")
    ap.add_argument("--run-dir", default=None,
                    help=f"where the free run lives (default {FREE_RUN_ROOT}/<lake>)")
    ap.add_argument("--threshold", type=float, default=THRESHOLD_DEFAULT,
                    help=f"correlation below which a depth pair is treated as uncorrelated "
                         f"(default {THRESHOLD_DEFAULT})")
    ap.add_argument("--dry-run", action="store_true", help="print instead of writing the json")
    cli = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)-8s | %(message)s")

    arg_file = cli.arg_file if os.path.isfile(cli.arg_file) else os.path.join(ROOT, cli.arg_file)
    if not os.path.isfile(arg_file):
        raise ValueError(f"Args file not found: {cli.arg_file}")
    with open(arg_file) as f:
        raw = json.load(f)

    cfg = merge_lake_args(raw, lake=cli.lake)
    run_dir = (cli.run_dir if cli.run_dir else os.path.join(FREE_RUN_ROOT, cli.lake))
    run_dir = run_dir if os.path.isabs(run_dir) else os.path.join(ROOT, run_dir)

    generated = False
    if cli.generate:
        logger.info(f"=== {cli.lake}: generating a free ensemble run ===")
        generate_free_run(cfg, run_dir)
        generated = True
    elif not os.path.isdir(run_dir):
        raise SystemExit(
            f"no free ensemble run at {os.path.relpath(run_dir, ROOT)}.\n"
            f"Generate one. This runs Simstrat in Docker against the reanalysis forcing, so do it "
            f"ON THE VPN:\n"
            f"    python notebooks/localization.py {cli.arg_file} --lake {cli.lake} --generate\n"
            f"It runs {cfg.get('n_members', '?')} members over "
            f"{cfg.get('start_date', '?')}..{cfg.get('end_date', '?')} with no analysis, derives the "
            f"radii, then deletes the run again (--keep to retain it).")

    try:
        out = derive_lake(cfg, run_dir, threshold=cli.threshold, dry_run=cli.dry_run)
    finally:
        # Delete only what this invocation created, and only after the radii are safely written --
        # a free run that failed to yield a table is worth keeping to look at.
        if generated and not cli.keep and not cli.dry_run:
            shutil.rmtree(run_dir, ignore_errors=True)
            logger.info(f"  removed the free run ({os.path.relpath(run_dir, ROOT)}); "
                        f"--keep retains it")

    if cli.dry_run:
        print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

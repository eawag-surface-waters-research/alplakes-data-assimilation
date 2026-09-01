"""Produce the observation series a run assimilates: pick the analysis HOUR, then THIN to it.

With `window_mode="obs"` the observation file IS the analysis schedule -- enkf.py opens one window
per distinct observation time -- so this script fixes the run's analysis cadence and instant, and
no timing knob elsewhere overrides it.

    STAGE 1  hour   inputs/<lake>/ref/T_out.dat (free run) vs observations/<lake>/temperature.csv
                    -> local/best_hour.json
    STAGE 2  thin   observations/<lake>/temperature_filtered.csv
                    -> observations/<lake>/temperature_filtered_h<HH>_1d.csv

Stage 1 runs once per lake and its answer is frozen into the output filename; stage 2 is rerun
whenever the filter or the thinning changes. Stage 1 fires only under `--hour best` with a missing
cache, or `--refit-hours`, so an explicit `--hour 6` never reads a free run -- thinning must not
inherit make_ref.py's dependency on Docker.

WHY THE HOUR MATTERS. Assimilating at a fixed hour aliases that hour's model-obs mismatch into a
persistent bias. Split the free run's surface error per hour into the mean over days (the model's
own diurnal flaw -- the SAME every day, so correcting it injects bias the physics undoes by
tomorrow) and the day-to-day scatter (real state error the analysis removes for good). On Upper
Lugano the scatter varies ~1.5x across the day while the diurnal flaw varies ~90x: the hour barely
changes what you gain and overwhelmingly changes what you inject.

Minimising |bias| alone is not enough -- a daily cycle crosses zero twice, and the two crossings
carry very different amounts of correctable error. So the rule is a GATE THEN A TIEBREAK: keep the
hours whose bias is small relative to the information at that same hour (|det| <= BIAS_FRAC * rnd),
then among those take the most information. A ratio, not an absolute degC, so it scales across
lakes whose error magnitudes differ ~3x.

THE CANDIDATE WINDOW USUALLY DECIDES THE ANSWER. Hours are restricted to 23-06 by default for
operational reasons, not statistical ones (the analysis lands before the day it forecasts, every
lake updates at a comparable point in the diurnal cycle, and nobody is reading the product). That
window is normally a stronger constraint than the criterion: on upperlugano it yields h=6, where
the unconstrained pick is h=17. Both are recorded -- `by_criterion` and `unconstrained` -- so the
cost of constraining is visible. Pass `--window all` to let the data alone decide.

Only hours the lake genuinely samples are candidates.

THINNING SELECTS, NEVER MODIFIES. Each kept row is a real observation at a real time, and all
depths of the winning profile are kept, so a thinned file stays a subset of the raw record.

    python notebooks/generate_assimilation_series.py --lakes all
    python notebooks/generate_assimilation_series.py --lakes all --refit-hours
    python notebooks/generate_assimilation_series.py --hours-only --lakes all
    python notebooks/generate_assimilation_series.py --lake upperlugano --hour 16 --every-days 2
    python notebooks/generate_assimilation_series.py --lake greifensee --hour-window 0-6
"""
import os
import sys
import json
import argparse
from datetime import datetime, timezone

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "src"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from assimilator.functions import load_obs                                             # noqa: E402
from assimilator.models.simstrat import SIMSTRAT_REF_YEAR                              # noqa: E402
import filter_observations as F                                                        # noqa: E402

# Surface zone: where the diurnal cycle lives and where the hour choice is decided. Bounds rather
# than a depth list so any lake works (aegeri's shallowest sensor is 2 m, greifensee's 1 m). The
# upper bound is THERMO_DEPTH_MIN, taken from the filter rather than restated, because it is the
# SAME boundary: above it the filter does not act, so the diurnal cycle survives into the
# assimilated series; below it the filter's box nulls it outright and the hour barely matters.
SURFACE_LO = 0.0
SURFACE_HI = float(F.DEFAULTS["THERMO_DEPTH_MIN"])
MIN_DAYS   = 30       # days an hour must be sampled on before it is a candidate
STRATIFIED = (5, 10)  # inclusive month bounds, matching load_free_run's `strat`
BIAS_FRAC  = 0.2      # the gate: |det| <= BIAS_FRAC * rnd. 0.2 = "bias at most a fifth of the fix"
WINDOW     = (0, 6)   # default candidate hours; see the docstring — this usually decides the answer
BEST_HOUR_JSON = os.path.join(_ROOT, "local", "best_hour.json")
CRITERIA   = ("balanced", "min-bias", "annual")


# ==================================================================== STAGE 1: pick the hour ====

def load_free_run(lake, detrend_days):
    """Hourly free-run error e = T_ref - obs, drift removed, with hour/month/season labels."""
    obs = load_obs(os.path.join(_ROOT, "observations", lake, "temperature.csv"))
    obs["time"] = pd.to_datetime(obs["time"], utc=True)

    ref = pd.read_csv(os.path.join(_ROOT, "inputs", lake, "ref", "T_out.dat"))
    t0 = pd.Timestamp(f"{SIMSTRAT_REF_YEAR}-01-01", tz="UTC")
    idx = (t0 + pd.to_timedelta(ref.iloc[:, 0].to_numpy(float), unit="D")).round("1h")
    m = pd.DataFrame(ref.iloc[:, 1:].to_numpy(float), index=idx,
                     columns=-np.array([float(c) for c in ref.columns[1:]]))
    m = m.loc[:, m.columns.isin(obs["depth"].unique())]
    mod = m.stack().rename("T_ref").reset_index()
    mod.columns = ["time", "depth", "T_ref"]

    j = mod.merge(obs, on=["time", "depth"]).dropna(subset=["value", "T_ref"])
    j["e"] = j["T_ref"] - j["value"]
    j = j.sort_values(["depth", "time"])
    # drift removal: the free run wanders over a year and that is not correctable structure
    win = int(detrend_days * 24)
    j["e_dt"] = j["e"] - (j.groupby("depth")["e"]
                          .transform(lambda s: s.rolling(win, center=True,
                                                         min_periods=win // 4).mean()))
    j["hour"] = j["time"].dt.hour
    j["month"] = j["time"].dt.month
    j["day"] = j["time"].dt.floor("D")
    j["strat"] = j["month"].between(*STRATIFIED)
    # diurnal anomaly: each day's own mean removed, so only hour-of-day structure survives
    j["ediur"] = j["e"] - j.groupby(["depth", "day"])["e"].transform("mean")
    return j.dropna(subset=["e_dt"])


def in_window(h, window):
    """Inclusive, and wraps: (22, 6) means 22,23,0,...,6."""
    if window is None:
        return True
    lo, hi = window
    return lo <= h <= hi if lo <= hi else (h >= lo or h <= hi)


def parse_window(s):
    if s.strip() in ("all", "any"):
        return None
    lo, hi = (int(x) for x in s.split("-"))
    return lo, hi


def hour_table(j):
    """Per hour: the deterministic and random parts of the surface error, and the sample count.

    `ediur` already has each day's own mean removed, so only hour-of-day structure survives and a
    seasonal drift cannot leak into the comparison between hours.
    """
    g = j[(j["depth"] > SURFACE_LO) & (j["depth"] <= SURFACE_HI)]
    if g.empty:
        raise ValueError(f"no observations in the {SURFACE_LO}-{SURFACE_HI} m surface zone")
    strat = g[g["strat"]]
    det_s = strat.groupby("hour")["ediur"].mean()
    rnd_s = strat.groupby("hour")["ediur"].std()
    n_s   = strat.groupby("hour")["day"].nunique()
    # annual: sum |monthly mean| rather than |annual mean|, so a positive bias in one month cannot
    # cancel a negative one in another and flatter an hour that is wrong in both.
    monthly = g.groupby(["hour", "month"])["ediur"].mean().abs()
    ann     = monthly.groupby("hour").sum()
    n_all   = g.groupby("hour")["day"].nunique()
    return det_s, rnd_s, n_s, ann, n_all


def pick(det_s, rnd_s, n_s, ann, n_all, window=WINDOW):
    """Best hour under each criterion, over the well-sampled hours inside `window`."""
    ok_s   = [h for h in det_s.index
              if n_s.get(h, 0) >= MIN_DAYS and in_window(h, window)]
    ok_all = [h for h in ann.index
              if n_all.get(h, 0) >= MIN_DAYS and in_window(h, window)]
    if not ok_s or not ok_all:
        raise ValueError(f"no hour both sampled on >= {MIN_DAYS} days and inside window {window}")

    # The gate: hours whose injected bias is small next to the error they could fix. Falls back to
    # the single least-biased hour if none qualifies, rather than silently picking on noise.
    low_bias = [h for h in ok_s if abs(det_s[h]) <= BIAS_FRAC * rnd_s[h]]
    if not low_bias:
        low_bias = [min(ok_s, key=lambda h: abs(det_s[h]))]
    out = {
        # The tiebreak: among those, the most correctable error. A gate then a tiebreak, NOT a
        # tradeoff — the bias is a transient the model's physics undoes by the next day, while the
        # random part is state error removed for good, so the two have no exchange rate. Inside the
        # wedge every hour is acceptable and this tiebreak is close to cosmetic: rnd is a std over
        # ~184 days, so its own sampling error (~5%) exceeds the gaps it resolves. If the gate looks
        # too loose, lower BIAS_FRAC; do not turn the constraint into a cost.
        "balanced": max(low_bias, key=lambda h: rnd_s[h]),
        "min-bias": min(ok_s,     key=lambda h: abs(det_s[h])),
        "annual":   min(ok_all,   key=lambda h: ann[h]),
    }
    return out, ok_s, low_bias


def analyse(lake, detrend_days, window=WINDOW):
    j = load_free_run(lake, detrend_days)
    det_s, rnd_s, n_s, ann, n_all = hour_table(j)
    best, candidates, low_bias = pick(det_s, rnd_s, n_s, ann, n_all, window)
    # what the data alone would have chosen, so the cost of constraining the window is visible
    free, _, _ = pick(det_s, rnd_s, n_s, ann, n_all, None)
    return {
        "sampled_hours": [int(h) for h in sorted(n_all.index)],
        "candidate_hours": [int(h) for h in candidates],
        "low_bias_hours": [int(h) for h in sorted(low_bias)],
        "by_criterion": {k: int(v) for k, v in best.items()},
        "unconstrained": {k: int(v) for k, v in free.items()},
        "window": list(window) if window else None,
        "surface_zone_m": [SURFACE_LO, SURFACE_HI],
        "min_days": MIN_DAYS,
        "bias_frac": BIAS_FRAC,
        "stratified_months": list(STRATIFIED),
        "by_hour": {
            str(int(h)): {
                "det_stratified": round(float(det_s[h]), 4),
                "rnd_stratified": round(float(rnd_s[h]), 4),
                "annual_abs_sum": round(float(ann.get(h, np.nan)), 4),
                "n_days": int(n_s.get(h, 0)),
            } for h in sorted(det_s.index)
        },
    }


def fit_hours(lakes, criterion, window, detrend_days, out_path):
    """Run stage 1 over `lakes` and write the best_hour JSON."""
    results, failed = {}, []
    for lake in lakes:
        try:
            results[lake] = analyse(lake, detrend_days, window)
            results[lake]["best_hour"] = results[lake]["by_criterion"][criterion]
        except Exception as exc:                       # noqa: BLE001 - isolate each lake
            print(f"{lake}: FAILED - {exc}")
            failed.append(lake)

    win = f"{window[0]:02d}-{window[1]:02d}" if window else "all"
    print(f"\nwindow {win}   criterion {criterion}")
    print(f"{'lake':<13}{'best':>5}{'det':>9}{'rnd':>7}{'n':>5}  "
          f"{'noon det':>9}{'noon rnd':>9}   {'bal':>4}{'minb':>5}{'ann':>4}{'free':>7}  "
          f"{'low-bias hours':<26}sampled")
    for lake, r in results.items():
        h, e = r["best_hour"], r["by_hour"][str(r["best_hour"])]
        noon = r["by_hour"].get("12")
        bc = r["by_criterion"]
        noon_txt = (f"{noon['det_stratified']:>+9.3f}{noon['rnd_stratified']:>9.3f}"
                    if noon else f"{'not sampled':>18}")
        lb = ",".join(str(x) for x in r["low_bias_hours"])
        print(f"{lake:<13}{h:>5}{e['det_stratified']:>+9.3f}{e['rnd_stratified']:>7.3f}"
              f"{e['n_days']:>5}  {noon_txt}   {bc['balanced']:>4}{bc['min-bias']:>5}"
              f"{bc['annual']:>4}{r['unconstrained']['balanced']:>7}  "
              f"{lb[:25]:<26}{len(r['sampled_hours'])}/24")

    out = {"generated": datetime.now(timezone.utc).date().isoformat(),
           "criterion": criterion,
           "source": "inputs/<lake>/ref/T_out.dat (free run) vs observations/<lake>/temperature.csv",
           "caveat": "the hour controls the BIAS injected, not the information gained; candidates "
                     "are restricted to hours the lake actually samples, and to `window`",
           "lakes": results}
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
        f.write("\n")
    print(f"\nwrote {os.path.relpath(out_path, _ROOT)}"
          + (f"   ({len(failed)} failed: {', '.join(failed)})" if failed else ""))
    return out


def resolve_hour(lake, hour_arg, best_hour_path):
    """`--hour 16` -> 16; `--hour best` -> the lake's entry in the best_hour JSON."""
    if hour_arg != "best":
        return int(hour_arg), "explicit"
    with open(best_hour_path, encoding="utf-8") as f:
        bh = json.load(f)
    if lake not in bh["lakes"]:
        raise SystemExit(f"{lake} not in {os.path.basename(best_hour_path)} "
                         f"(has: {', '.join(bh['lakes'])}) — rerun with --refit-hours")
    return int(bh["lakes"][lake]["best_hour"]), f"best/{bh['criterion']}"


# ========================================================================= STAGE 2: thinning ====

def select_depths(obs, depths, tol=0.5):
    """Keep only the observation depths nearest each requested one.

    The requested depths are a TARGET GRID, not exact keys: chains report 12.000000001 m and a
    thinned grid is chosen by physics, not by what the sensor happens to be labelled. Each target
    takes the nearest available depth within `tol` metres; a target with nothing that close is an
    error, because silently dropping it changes the experiment without saying so.
    """
    have = np.sort(obs["depth"].unique())
    keep, missing = [], []
    for t in depths:
        i = int(np.argmin(np.abs(have - t)))
        if abs(have[i] - t) <= tol:
            keep.append(have[i])
        else:
            missing.append(t)
    if missing:
        raise SystemExit(f"no observation depth within {tol} m of {missing} "
                         f"(available {have.min():g}..{have.max():g} m)")
    return obs[obs["depth"].isin(sorted(set(keep)))]


def thin_one(lake, hour, every_days, in_file=None, out_file=None, max_offset=1.5,
             depths=None, hour_window=None):
    """Keep ONE instant per retained day: the observation time nearest `hour`, all depths.

    Nearest-to-target rather than "falls inside hour H", for two reasons. A 10-minute lake has six
    timestamps inside any hour, so an hour test yields six analyses a day, not one. And a drifting
    schedule -- greifensee profiles every 3 h but its minute-of-hour wanders across the whole hour
    over a year -- would miss the target hour entirely on many days and silently drop them.

    `max_offset` (hours) bounds how far the substitute may be, so a day whose only observation sits
    at a quite different point of the diurnal cycle is skipped instead of quietly standing in.

    `hour_window=(lo, hi)` replaces the nearest-hour rule with "the FIRST observation in [lo, hi)".
    For a lake whose sampling phase drifts across the night, a fixed target hour is present on only
    a fraction of days -- greifensee reports within 1.5 h of 01:00 on 110 days of 249, and the
    missing 123 were previously supplied by interpolation. Following the phase instead of fighting
    it yields 224 real days. `max_offset` does not apply: the window IS the bound.

    `depths` restricts the profile to a target grid (see select_depths). Rows are still only ever
    SELECTED, so a thinned file remains a subset of the raw record.
    """
    in_path = in_file or os.path.join(_ROOT, "observations", lake, "temperature.csv")
    tag = (f"w{hour_window[0]:02d}{hour_window[1]:02d}" if hour_window else f"h{hour:02d}")
    out_path = out_file or os.path.join(
        _ROOT, "observations", lake, f"temperature_{tag}_{every_days}d.csv")

    obs = pd.read_csv(in_path)
    if depths is not None:
        obs = select_depths(obs, depths)
    obs["_t"] = pd.to_datetime(obs["time"], utc=True, format="ISO8601")
    # Match load_obs's binning so a kept row lands in the hour it is labelled with: load_obs shifts
    # by +30 min then floors, i.e. it rounds to the nearest hour.
    binned = (obs["_t"] + pd.Timedelta(minutes=30)).dt.floor("1h")

    day = binned.dt.floor("D")
    day0 = day.min()
    cadence = ((day - day0).dt.days % every_days) == 0
    if hour_window is not None:
        lo, hi = hour_window
        h = binned.dt.hour
        inside = (h >= lo) & (h < hi)
        # earliest qualifying instant of each day, then every row sharing it
        pick_t = obs.loc[obs["_t"][inside].groupby(day[inside]).idxmin(), "_t"]
        keep = obs["_t"].isin(set(pick_t)) & cadence
        if not keep.any():
            raise SystemExit(f"no observation in [{lo:02d}:00, {hi:02d}:00) in {in_path}")
    else:
        off = (obs["_t"] - (day + pd.Timedelta(hours=hour))).abs()
        # one winning timestamp per day, then every row sharing it (all depths of that profile)
        pick_t = obs.loc[off.groupby(day).idxmin(), "_t"]
        keep = obs["_t"].isin(set(pick_t)) \
            & (off <= pd.Timedelta(hours=max_offset)) \
            & cadence
    out = obs[keep].drop(columns="_t")
    if out.empty:
        raise SystemExit(f"no rows within {max_offset} h of {hour:02d}:00 UTC in {in_path}")

    out.to_csv(out_path, index=False)
    kept_t = binned[keep]
    gaps = kept_t.drop_duplicates().sort_values().diff().dropna()
    return {
        "out": os.path.relpath(out_path, _ROOT).replace(os.sep, "/"),
        "rows_in": len(obs), "rows_out": len(out),
        "times_in": binned.nunique(), "times_out": kept_t.nunique(),
        "depths": out["depth"].nunique(),
        "span": f"{kept_t.min():%Y-%m-%d} .. {kept_t.max():%Y-%m-%d}",
        "gap_median": str(gaps.median()) if len(gaps) else "-",
        "gap_max": str(gaps.max()) if len(gaps) else "-",
    }


# ==================================================================================== driver ====

def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--lake", default=None)
    grp.add_argument("--lakes", default=None,
                     help="comma-separated, or 'all' for every lake in --arg-file")

    st2 = ap.add_argument_group("stage 2: thinning")
    st2.add_argument("--hour", default="best",
                     help="analysis hour UTC, or 'best' to use the stage-1 choice (default). An "
                          "explicit hour skips stage 1, so no free run is needed")
    st2.add_argument("--every-days", type=int, default=1, help="days between analyses")
    st2.add_argument("--max-offset", type=float, default=1.5,
                     help="hours the nearest observation may sit from the target before the day is "
                          "skipped (default 1.5 — half a 3-hourly lake's sampling interval)")
    st2.add_argument("--hour-window", default=None, metavar="LO-HI",
                     help="take the FIRST observation in [LO, HI) each day instead of the one "
                          "nearest --hour. For a lake whose sampling phase drifts, a fixed hour is "
                          "present on only a fraction of days (greifensee: 110 of 249 near 01:00, "
                          "224 with '0-6'). Single lake only; --max-offset does not apply")
    st2.add_argument("--depths", default=None,
                     help="comma-separated target depths in m, e.g. '1,2,3,4'. Each takes the "
                          "nearest available depth within 0.5 m; a target with none is an error. "
                          "Single lake only")
    st2.add_argument("--in-file", default=None, help="single lake only")
    st2.add_argument("--out-file", default=None, help="single lake only")

    st1 = ap.add_argument_group("stage 1: hour selection")
    st1.add_argument("--hours-only", action="store_true",
                     help="run stage 1 and stop, without thinning")
    st1.add_argument("--refit-hours", action="store_true",
                     help="recompute the hours even if the cache exists (needs the free runs)")
    st1.add_argument("--criterion", default="balanced", choices=CRITERIA,
                     help="which criterion sets `best_hour` (all three are always written)")
    st1.add_argument("--window", default=f"{WINDOW[0]}-{WINDOW[1]}",
                     help=f"candidate hours UTC, inclusive, wraps (default {WINDOW[0]}-{WINDOW[1]}). "
                          f"Usually a stronger constraint than the criterion — see the module "
                          f"docstring. 'all' to let the data decide")
    st1.add_argument("--detrend-days", type=float, default=15.0)
    st1.add_argument("--best-hour-json", default=BEST_HOUR_JSON)

    ap.add_argument("--arg-file", default=os.path.join(_ROOT, "args", "run_enkf.json"))
    cli = ap.parse_args()

    window = None
    if cli.hour_window:
        try:
            lo, hi = (int(s) for s in cli.hour_window.split("-"))
        except ValueError:
            raise SystemExit(f"--hour-window expects LO-HI, e.g. '0-6' (got {cli.hour_window!r})")
        if not 0 <= lo < hi <= 24:
            raise SystemExit(f"--hour-window needs 0 <= LO < HI <= 24 (got {lo}-{hi})")
        window = (lo, hi)
    depths = ([float(s) for s in cli.depths.split(",") if s.strip()] if cli.depths else None)

    if cli.lakes:
        with open(cli.arg_file) as f:
            cfg = json.load(f)
        lakes = (list(cfg["lakes"]) if cli.lakes.strip() == "all"
                 else [s.strip() for s in cli.lakes.split(",") if s.strip()])
        if cli.in_file or cli.out_file:
            raise SystemExit("--in-file/--out-file name a single file and are refused in batch mode")
        if window or depths:
            raise SystemExit("--hour-window/--depths are per-lake and are refused in batch mode")
    else:
        lakes = [cli.lake]

    # --- stage 1, only when its answer is needed and not already cached ------------------------
    needs_hours = cli.hours_only or (cli.hour == "best" and not window)
    if needs_hours and (cli.refit_hours or not os.path.isfile(cli.best_hour_json)):
        if not cli.refit_hours:
            print(f"{os.path.relpath(cli.best_hour_json, _ROOT)} not found — fitting the hours "
                  f"(needs inputs/<lake>/ref/T_out.dat)")
        fit_hours(lakes, cli.criterion, parse_window(cli.window), cli.detrend_days,
                  cli.best_hour_json)
    if cli.hours_only:
        return

    # --- stage 2 -------------------------------------------------------------------------------
    print(f"\n{'lake':<13}{'hour':>5}{'source':>14}{'rows':>18}{'times':>16}{'depths':>8}  "
          f"{'span':<26}{'gap'}")
    failed = []
    for lake in lakes:
        try:
            if window:
                hour, how = window[0], f"[{window[0]:02d},{window[1]:02d})"
            else:
                hour, how = resolve_hour(lake, cli.hour, cli.best_hour_json)
            r = thin_one(lake, hour, cli.every_days, cli.in_file, cli.out_file, cli.max_offset,
                         depths=depths, hour_window=window)
            rows  = f"{r['rows_in']:,} -> {r['rows_out']:,}"
            times = f"{r['times_in']:,} -> {r['times_out']:,}"
            print(f"{lake:<13}{hour:>5}{how:>14}{rows:>18}{times:>16}"
                  f"{r['depths']:>8}  {r['span']:<26}{r['gap_median']}")
        except SystemExit as exc:
            print(f"{lake:<13}  FAILED - {exc}")
            failed.append(lake)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()

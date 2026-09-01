"""Fit sigma_rep_scale as a function of DEPTH (and season) from a run's per-depth innovations.

Every observation is used at its own depth:

    <d^2>_z = <spread^2>_z + sigma_common(z)^2 + k(z,s)^2 * <sigma_rep(z,m)^2 / N>_z

    k(z,s) = sqrt( max(<d^2> - <spread^2> - sigma_common^2, 0) / <sigma_rep^2/N> )

Closed form, no optimiser, and the output shape is what `resolve_sigma_rep_scale` already accepts —
a depth-keyed step function per season, straight into the run config.

FOUR THINGS THIS FIT DOES NOT KNOW, each printed rather than assumed away:

    what it absorbs  everything the innovation carries and the spread does not lands in R -- model
                     bias included. `bias_share` = (mean d)^2/<d^2> per band flags it.
    the estimator    the gap sigma is std(ref - obs) against the FREE run, d is against a state
                     already pulled toward the obs. k < 1 is the estimator changing, not the error.
    the coupling     shrink R, the analysis pulls harder, the next d shrinks too. Refit against the
                     run that used the last k; stop when NIS crosses 1, not when k stops moving.
    the depth axis   one k per depth overfits on one year (3.9x greifensee 2.5 m, 5.4x hallwil 25 m).
                     --bands defaults to ONE per season; add breakpoints where sigma_rep itself steps.

    python notebooks/fit_scale_by_depth.py args/experiments/gap_nocommon.json --run ~/gap_nocommon
      python notebooks/fit_scale_by_depth.py args/experiments/gap_nocommon.json --run ~/gap_nocommon \\
          --bands 0,4 --target mean --out local/sigma/scale_by_depth_nocommon_b4.json
"""
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "src"), os.path.dirname(os.path.abspath(__file__))):
    sys.path.insert(0, _p)

import innovations as mr                                                          # noqa: E402
from assimilator.functions import (merge_lake_args, season_of, load_obs,          # noqa: E402
                                   resolve_obs_path)
from assimilator.sigma_rep import (load_sigma_rep, _sigma_rep_at, table_n_ref,    # noqa: E402
                                   resolve_sigma_common, resolve_sigma_rep_scale)

SEASONS = ("mixed", "stratified")
CHI2_1_MEDIAN = 0.4549364231195728      # median of chi-squared with 1 dof


def per_obs(raw, lake):
    """One row per assimilated observation: d, spread, sigma_common, sigma_rep^2/N, season.

    mr._CACHE is keyed on (kind, lake) with no run root, so it is cleared per call — otherwise a
    second run in the same process silently re-reads the first one's table.
    """
    mr._CACHE.clear()
    cfg = merge_lake_args(raw, lake=lake)
    table = load_sigma_rep(cfg)
    if table is None:
        raise SystemExit(f"{lake}: no sigma_rep table — there is nothing for a scale to multiply")
    sub, _depths = mr.innov_table(raw, lake)

    # N per observation. innov_table's frame is (time, depth, d, spread, sigma) — read_innov does
    # not carry n_stations — so it is joined back from the assimilated series, exactly as
    # resolve_sigma_obs gets it at runtime. Assuming 1 would be silently wrong by up to n_ref on a
    # multi-station lake, so an n_ref table with no match is an error rather than a default.
    n_ref = table_n_ref(table)
    obs = load_obs(resolve_obs_path(cfg))
    if "n_stations" in obs:
        # Depths are rounded on BOTH sides before matching: greifensee's obs file carries
        # 2.000000000000001 / 3.0000000000000018 while the innovation header writes 2 and 3, and an
        # exact float join drops 94% of that lake's rows.
        key = obs.set_index([pd.DatetimeIndex(obs["time"]),
                             obs["depth"].astype(float).round(6)])["n_stations"]
        idx = pd.MultiIndex.from_arrays([pd.DatetimeIndex(sub["time"]),
                                         sub["depth"].astype(float).round(6)])
        n = key.reindex(idx).to_numpy(dtype=float)
    else:
        n = np.full(len(sub), np.nan)
    missed = int(np.isnan(n).sum())
    if missed and n_ref is not None:
        raise SystemExit(f"{lake}: {missed:,} of {len(sub):,} innovations have no n_stations, and "
                         f"this table is in the n_ref={n_ref:g} form — the /N term cannot be "
                         f"guessed. Check that the obs file is the one the run assimilated.")
    if missed:
        print(f"{lake:<14} {missed:,} of {len(sub):,} innovations unmatched in the obs file — "
              f"N=1 assumed (single-station lake, so this only affects the join, not the term)")
    n = np.where(np.isfinite(n) & (n >= 1), n, 1.0)

    month = sub["time"].dt.month.to_numpy()
    rep = np.array([_sigma_rep_at(table, z, m)
                    for z, m in zip(sub["depth"].to_numpy(), month)], dtype=object)
    keep = np.array([r is not None for r in rep])
    if not keep.all():
        # No fitted entry -> the runtime used the scalar sigma_obs at that depth, which no scale
        # multiplies. Excluded rather than fitted, and counted so the omission is visible.
        print(f"{lake:<14} {int((~keep).sum()):,} of {len(sub):,} obs fall back to the scalar "
              f"sigma_obs (no table entry) — excluded from the fit")
    out = sub.loc[keep, ["time", "depth", "d", "spread"]].copy()
    rep = np.array([float(r) for r in rep[keep]])
    scale_n = (1.0 / n[keep]) if n_ref is None else (n_ref / n[keep])
    out["rep2_over_n"] = rep ** 2 * scale_n
    out["sigma_common"] = [resolve_sigma_common(cfg, z) for z in out["depth"]]
    out["k_used"] = [resolve_sigma_rep_scale(cfg, z, m)
                     for z, m in zip(out["depth"], out["time"].dt.month)]
    out["season"] = [season_of(m) for m in out["time"].dt.month]
    return out


def band_of(depths, edges):
    """The deepest breakpoint at or above each depth — the same rule _depth_stepped reads back."""
    edges = np.asarray(sorted(edges), dtype=float)
    return edges[np.clip(np.searchsorted(edges, np.asarray(depths, dtype=float), "right") - 1, 0,
                         None)]


def _nis(g, k):
    """Per-observation NIS the group would have had under scale `k`."""
    return g["d"] ** 2 / (g["spread"] ** 2 + g["sigma_common"] ** 2 + k ** 2 * g["rep2_over_n"])


def _pooled_nis(g, k):
    """<d^2>/<spread^2+sigma^2> under scale `k` — innovations.pooled_nis's identity, so these
    numbers sit beside the ones every other figure in the project prints, not beside a mean of
    ratios."""
    return float(np.mean(g["d"] ** 2)
                 / np.mean(g["spread"] ** 2 + g["sigma_common"] ** 2 + k ** 2 * g["rep2_over_n"]))


def _k_for_median(g, lo=0.01, hi=20.0, tol=1e-4):
    """k that puts the MEDIAN per-observation NIS on chi2(1)'s median.

    Bisected rather than solved: the median does not commute with the ratio, so unlike the mean
    target there is no closed form. NIS is monotone decreasing in k, so bisection is exact enough.
    Returns the bracket end when the target lies outside it — a group the scale cannot reach.
    """
    if np.median(_nis(g, lo)) < CHI2_1_MEDIAN:
        return lo
    if np.median(_nis(g, hi)) > CHI2_1_MEDIAN:
        return hi
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if np.median(_nis(g, mid)) > CHI2_1_MEDIAN:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def fit_group(g, target):
    """k for one (band, season) group, both targets, plus what it is standing on."""
    d2, spread2 = np.mean(g["d"] ** 2), np.mean(g["spread"] ** 2)
    sc2, rep2 = np.mean(g["sigma_common"] ** 2), np.mean(g["rep2_over_n"])
    k = {"mean": np.sqrt(max(d2 - spread2 - sc2, 0.0) / rep2),
         "median": _k_for_median(g)}
    return {"n": int(len(g)), "nis_before": _pooled_nis(g, g["k_used"]),
            "nis_after": _pooled_nis(g, k[target]),
            "k_mean": float(k["mean"]), "k_median": float(k["median"]), "k": float(k[target]),
            "bias_share": float(np.mean(g["d"]) ** 2 / d2),
            "rms_d": float(np.sqrt(d2)), "rms_spread": float(np.sqrt(spread2)),
            "rms_rep": float(np.sqrt(rep2))}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("config", nargs="?", default="args/experiments/gap_depth.json")
    ap.add_argument("--run", required=True, help="run root holding <lake>/enkf_diagnostics.csv")
    ap.add_argument("--lakes", default=None, help="comma list (default: every lake in the config)")
    ap.add_argument("--bands", default="0", help="comma list of band breakpoints in metres, read "
                                                 "as _depth_stepped reads them (default: one band)")
    ap.add_argument("--no-seasons", action="store_true", help="one k per band for the whole year")
    ap.add_argument("--target", default="mean", choices=["mean", "median"],
                    help="mean closes the variance identity; median asks the typical analysis to "
                         "be consistent and lets the tail stay a tail (default: mean)")
    ap.add_argument("--floor", type=float, default=0.3, help="lower clip on k (default 0.3)")
    ap.add_argument("--ceil", type=float, default=5.0, help="upper clip on k (default 5.0)")
    ap.add_argument("--out", default=None, help="write the fitted block as JSON here")
    cli = ap.parse_args()

    path = cli.config if os.path.isfile(cli.config) else os.path.join(_ROOT, cli.config)
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    raw["run_root"] = os.path.expanduser(cli.run)
    lakes = ([s.strip() for s in cli.lakes.split(",")] if cli.lakes
             else list(raw.get("lakes") or []))
    edges = [float(s) for s in cli.bands.split(",")]

    rows, blocks = [], {}
    for lake in lakes:
        try:
            obs = per_obs(raw, lake)
        except Exception as exc:                    # noqa: BLE001 - isolate each lake
            print(f"{lake}: skipped ({type(exc).__name__}: {exc})")
            continue
        obs["band"] = band_of(obs["depth"], edges)
        keys = ["band"] if cli.no_seasons else ["season", "band"]
        block = {}
        for key, g in obs.groupby(keys):
            season, band = (None, key) if cli.no_seasons else key
            fit = fit_group(g, cli.target)
            k = float(np.clip(fit["k"], cli.floor, cli.ceil))
            if k != fit["k"]:
                print(f"{lake:<14} {season or 'year':<10} >={band:g}m: k {fit['k']:.2f} clipped "
                      f"to {k:.2f}")
            rows.append({"lake": lake, "season": season or "year", "band_m": band, **fit,
                         "k_clipped": k})
            block.setdefault(season, {})[f"{band:g}"] = round(k, 3)
        blocks[lake] = (block[None] if cli.no_seasons
                        else {s: block[s] for s in SEASONS if s in block})

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no lake produced a fit")
    pd.set_option("display.width", 220)
    print()
    print(df.round(3).to_string(index=False))
    print("\nsigma_rep_scale blocks (paste into the run config, per lake):")
    print(json.dumps(blocks, indent=2))
    if cli.out:
        out = cli.out if os.path.isabs(cli.out) else os.path.join(_ROOT, cli.out)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump({"target": cli.target, "bands": edges, "run": raw["run_root"],
                       "sigma_rep_scale": blocks,
                       "diagnostics": df.round(4).to_dict(orient="records")}, f, indent=2)
        print(f"\nwrote {out}")
    print("\nOne pass is not a fixed point — rerun the experiment with these, then refit against "
          "THAT run and stop when NIS crosses 1.")


if __name__ == "__main__":
    main()

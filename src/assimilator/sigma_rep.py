"""Observation-error model: turn the observation set into a per-observation sigma.

  The scalar `sigma_obs` is really an estimate of REPRESENTATIVENESS error, not instrument error --
  a thermistor is good to ~0.05 degC, so a 0.5 degC observation error is almost entirely "how far a
  point measurement sits from the lake-mean value a 1D column model predicts", plus the observation
  variability the model cannot reproduce. That quantity is not a constant: it peaks at the
  thermocline, where a sharp gradient is displaced past a fixed sensor by internal seiches, and
  collapses in the deep and in the mixed season. 

  Sigma is resolved per observation as

      sigma_i^2  =  sigma_common(depth_i)^2  +  (scale * sigma_rep(depth_i, month_i))^2 * f(N_i)

  THE LADDER. Each live config stops at a different rung, and that is the only difference between
  the first two:

      simplified_final.json      no table -> resolve_scalar_sigma_obs IS the whole model, and every
                                 term below is unreached (the function returns before sigma_common).
                                 A float, or {"mixed": .., "stratified": ..} keyed off the window's
                                 month.
      
      gap_nocommon.json          the same fit with the depth axis KEPT
      
      gap_nocommon_scale2.json   fitted per depth band and season from that run

    sigma_rep    fitted, depth- and season-dependent. Two live estimators, which BRACKET the truth
                 rather than agreeing:
                   
                   fit_sigma_gap.py -> sigma_rep_gap.json    obs vs the free run; carries model
                                                             error too
                                                             
                   fit_std_obs.py   -> sigma_rep_nref.json   across-station scatter; no model can
                                                             leak in, but it sees only what stations
                                                             disagree about (so rather a LOWER bound)
                                                             
                 f(N) = 1/N for a per-station table (the value is one station's error); n_ref/N for
                 an n_ref table (the value is the full complement's, scaled up when fewer report --
                 see table_n_ref). Multi-station lakes exist: lowerlugano has 4, lowerzurich 2.
                 
    sigma_common the part no station can see in itself -- instrument error, plus slow
                 representativeness error a station-based estimator is structurally blind to. A
                 config value, not fitted; may be depth-stepped (see resolve_sigma_common). It is a
                 floor under a LOWER bound, so it belongs with the across-station tables and not with
                 the gap ones.
                 
    scale        "sigma_rep_scale", default 1.0. Scalar, depth-stepped, or a season-keyed dict of
                 either. Which DIRECTION it moves depends on which estimator wrote the table: it
                 lifts a lower bound toward the innovations, and walks an upper bound down -- 25 of
                 the 28 bands in gap_nocommon_scale2.json are below 1. See resolve_sigma_rep_scale.

  Deliberately CLIMATOLOGICAL

  A lake with no table falls back to the scalar `sigma_obs`, so its behaviour is bit-for-bit what it
  was before this module existed. Deleting the json is the kill switch.
"""
import os
import json
import logging

import numpy as np

from .functions import ROOT, resolve_root, SEASONS, season_of

logger = logging.getLogger(__name__)

# The instrument term, and the floor under the whole error model (optional).

DEFAULT_SIGMA_COMMON = 0.075


def depth_key(d):
    """JSON object keys are strings; use one formatting everywhere so 1.0 and 1 are the same key."""
    return f"{float(d):g}"


def _depth_stepped(spec, depth, default):
    """A config value that is either a scalar or a depth-keyed step function

        {"0": 0.2, "7": 0.1}

    read as "0.2 from 0 m down, 0.1 from 7 m down". 
    Shared by sigma_common and sigma_rep_scale so the two spell depth-dependence the same
    way in a run config.
    """
    if spec is None:
        return float(default)
    if not isinstance(spec, dict):
        return float(spec)
    # deepest breakpoint at or above this depth; above every breakpoint -> the shallowest value
    edges = sorted((float(k), float(v)) for k, v in spec.items())
    value = edges[0][1]
    for z, v in edges:
        if float(depth) >= z:
            value = v
    return float(value)


def resolve_sigma_common(cfg, depth):
    """sigma_common for one depth -- a scalar or a depth-keyed step function (see _depth_stepped).

    Note what a depth step buys and what it does not: it absorbs a persistent bias as variance, so
    it improves calibration, not accuracy -- a warm surface analysis stays warm, the filter just
    stops being overconfident about it. A bias correction on the innovation is the accurate fix.

    And note what it CANNOT do: sigma_common has no month argument. Fitting it to close the
    innovation budget year-round lands it near the summer requirement and over-damps winter. Where
    the shortfall is seasonal, sigma_rep_scale is the term to move -- see resolve_sigma_rep_scale.
    """
    return _depth_stepped(cfg.get("sigma_common"), depth, DEFAULT_SIGMA_COMMON)


def _is_season_keyed(spec):
    """True for {"mixed": ..., "stratified": ...}. Depth keys are numeric, so the two never collide
    and an old depth-only config is recognised unchanged."""
    return isinstance(spec, dict) and bool(spec) and set(spec) <= set(SEASONS)


def has_season_keyed_sigma_obs(cfg):
    """True when the config's scalar `sigma_obs` carries a season axis. The public form of the
    predicate, for the OpenDA adapter: a season-keyed scalar has to be season-split into series
    exactly like a table does, and there is no table to detect it by."""
    return _is_season_keyed(cfg.get("sigma_obs"))


def resolve_sigma_rep_scale(cfg, depth, month=None):
    """Multiplier on the fitted sigma_rep -- a scalar, a depth-keyed step function, or a season-keyed
      dict of either. Default 1.0, i.e. the table is used exactly as fitted.

          "sigma_rep_scale": {"mixed":      {"0": 0.818, "4": 0.818},
                              "stratified": {"0": 0.622, "4": 0.723}}
                                                                                                                                                                                                                                                                      
      A season-keyed scale REQUIRES `month`; passing None raises rather than silently picking a
      season.

      WHY a multiplier is a legitimate parameter: The fitted sigma_rep is a BOUND, not an estimate,
      and k walks it to what the innovations imply -- closing
      <d^2> = <spread^2> + sigma_common^2 + k^2 * <sigma_rep^2/N>. WHICH DIRECTION depends on which
      estimator wrote the table:

        gap tables       an UPPER bound (they carry model error too), so k < 1 walks them DOWN.
                         25 of the 28 bands in gap_nocommon_scale2.json are below 1, running 0.414
                         (hallwil mixed, deep) to 1.372 (geneva mixed, shallow). Most of them ... 
                         
        station tables   a LOWER bound (across-station scatter sees only what stations disagree
                         about), so k > 1 lifts them.

      Either way it inherits sigma_rep's depth profile AND its month axis rather than imposing a new
      shape, which is what makes one number per band defensible.

      WHAT k ABSORBS, and it is not only observation error. The identity pushes everything the
      innovation carries and the spread does not into R -- model bias included. `bias_share` in the
      fit JSON is (mean d)^2/<d^2> per band, and it is systematically worse in the SHALLOW band,
      which is deliberately unfiltered: aegeri mixed shallow is 41% bias by variance, greifensee
      stratified 32%, upperlugano stratified 25%. Those bands want a bias correction, not a scale.

      THE SCALE IS A PROPERTY OF THE OBSERVATION SERIES, NOT OF THE LAKE. The 4 m breakpoint is
      THERMO_DEPTH_MIN from notebooks/filter_observations.py: at or below it the assimilated series
      is a causal seiche box mean, above it the record passes through raw. So the filter, not
      the lake, is where the multiplier genuinely steps. Refit whenever the thinning or the filter
      changes.

      ON BAND COUNT. Two bands beat one and are not overfitting at this count. The overfitting risk
      would be one k PER DEPTH not per band.

      ONE PASS IS NOT A FIXED POINT. R and the innovations are coupled: shrink R, the analysis pulls
      harder, the next innovation shrinks again. Refit against the run that used the last k and stop
      when NIS aproaches 1.
    """
    spec = cfg.get("sigma_rep_scale")
    if _is_season_keyed(spec):
        if month is None:
            raise ValueError("sigma_rep_scale is season-keyed, so resolving it needs a month")
        season = season_of(month)
        if season not in spec:
            raise ValueError(f"sigma_rep_scale has no {season!r} entry (has {sorted(spec)})")
        spec = spec[season]
    return _depth_stepped(spec, depth, 1.0)


def resolve_scalar_sigma_obs(cfg, month=None):
    """The config's `sigma_obs` -- a plain scalar, or a season-keyed dict of scalars

        "sigma_obs": {"mixed": 0.31, "stratified": 0.55}

    spelled exactly like a season-keyed sigma_rep_scale (see resolve_sigma_rep_scale).

    This does NOT compete with the fitted table -- it is the SIMPLIFIED option one rung below it,
    and it applies exactly where the plain scalar applies today: as the whole error model on a lake
    with no sigma_rep.json, and as the per-depth fallback where a table has no fitted entry. A
    table, when present, still wins at every depth it covers.

    WHY the season axis is worth having on the scalar. The free-run gap a scalar is fitted from is
    not one population: on the 2025 seven-lake run the stratified season wants 1.1x (murten) to 2.6x
    (maggiore) the mixed-season value, so an annual scalar is the count-weighted compromise between
    them -- wrong in both at once, and most wrong on the lake with the widest split.

    A season-keyed value REQUIRES `month`; passing None raises rather than silently picking one.
    """
    spec = cfg["sigma_obs"]
    if _is_season_keyed(spec):
        if month is None:
            raise ValueError("sigma_obs is season-keyed, so resolving it needs a month")
        season = season_of(month)
        if season not in spec:
            raise ValueError(f"sigma_obs has no {season!r} entry (has {sorted(spec)})")
        spec = spec[season]
    return float(spec)


def _scalar_sigma_rms(cfg, months):
    """RMS of the config scalar over `months` -- for OpenDA, whose per-depth format cannot carry a
    month axis. A plain scalar is the RMS of identical values, i.e. itself, so the non-seasonal
    path is untouched."""
    if not _is_season_keyed(cfg["sigma_obs"]):
        return float(cfg["sigma_obs"])
    vals = [resolve_scalar_sigma_obs(cfg, m) for m in months]
    return float(np.sqrt(np.mean(np.square(vals))))


def sigma_rep_path(cfg):
    """The fitted table for the run: 'sigma_rep_file' (repo-relative or absolute) if given, else
    observations/<lake>/sigma_rep.json -- beside the observations it was fitted from."""
    override = cfg.get("sigma_rep_file")
    if override:
        return resolve_root(override)
    return os.path.join(ROOT, "observations", cfg["lake"], "sigma_rep.json")

# NOTE: needs rewriting ... 
def load_sigma_rep(cfg):
    """The fitted sigma_rep table, or None when the lake has none.

    Absent is NOT an error (unlike perturbations/<lake>.json): a lake without a free run or without
    observations cannot have one, and it correctly falls back to the scalar sigma_obs."""
    path = sigma_rep_path(cfg)
    if not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as f:
        table = json.load(f)
    if not isinstance(table.get("sigma_rep"), dict):
        raise ValueError(f"{path}: malformed sigma_rep table — missing 'sigma_rep' object")
    # A table fitted on a different series than the one being assimilated double-counts: the
    # adaptive filter removes most of the sub-daily variability sigma_rep measures, so a raw-fitted
    # table applied to filtered observations overstates R several-fold at the thermocline.
    src = str(table.get("source", ""))
    obs_file = str(cfg.get("obs_file", ""))
    # Filename markers. A marker list is fragile by nature: any future series naming has to be added here, and
    # the failure is a spurious warning rather than a wrong R, which is the safe direction.
    marks = ("_filtered", "temperature_new")
    is_filtered = lambda s: any(m in s for m in marks)          # noqa: E731
    if is_filtered(obs_file) != is_filtered(src):
        logger.warning(f"[sigma] {os.path.relpath(path, ROOT)} was fitted on "
                       f"'{src.split(' (buoy)')[0]}' but this run assimilates "
                       f"'{obs_file or 'observations/<lake>/temperature.csv'}' — refit sigma_rep on "
                       f"the series being assimilated, or R will be wrong at the thermocline")
    return table


def _entry_value(entry, month):
    """The month's fitted value for one table entry, else its all-season value, else None."""
    if entry is None:
        return None
    v = (entry.get("by_month") or {}).get(str(int(month)))
    return float(v) if v is not None else (
        float(entry["all"]) if entry.get("all") is not None else None)


def _sigma_rep_at(table, depth, month):
    """sigma_rep for one (depth, month), interpolating over depth when there is no exact entry.

    A sensor can be missing from the table without being missing from the lake: the fitter drops a
    (depth, season) with fewer than MIN_DAYS blocks, and on a long-block lake that is most of a year
    (geneva's block is 97.4 h, so 20 blocks is ~81 days -- its 2025 chain extension failed it at
    55/60/65/80/85 m). An exact-match-or-nothing lookup then hands those depths the scalar
    sigma_obs = 0.5 while their neighbours sit near 0.05, silently discarding them.

    Between two fitted depths, log-linear interpolation: sigma_rep spans an order of magnitude down
    a column and decays roughly geometrically, so interpolating the logarithm keeps a mid-point
    between 0.30 and 0.03 near 0.09 rather than 0.17.

    Only BETWEEN fitted depths. Outside their range there is no measurement to interpolate and
    extrapolation would invent one, so the caller still falls back to the scalar -- which is the
    honest answer for a sensor deeper than anything ever fitted.
    """
    reps = table["sigma_rep"]
    v = _entry_value(reps.get(depth_key(depth)), month)
    if v is not None:
        return v

    d = float(depth)
    have = sorted((float(k), val) for k, val in
                  ((k, _entry_value(e, month)) for k, e in reps.items()) if val is not None)
    below = [(z, val) for z, val in have if z < d]
    above = [(z, val) for z, val in have if z > d]
    if not below or not above:
        return None
    z0, v0 = below[-1]
    z1, v1 = above[0]
    if v0 <= 0 or v1 <= 0:
        return float(v0 + (v1 - v0) * (d - z0) / (z1 - z0))
    w = (d - z0) / (z1 - z0)
    return float(np.exp(np.log(v0) + w * (np.log(v1) - np.log(v0))))


def table_n_ref(table):
    """The station count a table's values refer to, or None when it is in the per-station form.

    Two conventions, told apart by whether the table carries 'n_ref':

      per-station (no key)  the value is ONE station's error, scaled by 1/N. Every single-station
                            lake's table is this, and that branch is untouched.
      n_ref                 the value is the error of the mean of n_ref stations -- the lake's full
                            complement -- scaled UP by n_ref/N when fewer report. Written by
                            notebooks/fit_std_obs.py for the multi-station lakes.

    The two are the same function of N apart from the constant n_ref, so this changes what the
    stored number means, not how sigma varies with N.
    """
    if table is None or table.get("n_ref") is None:
        return None
    n_ref = float(table["n_ref"])
    if n_ref < 1.0:
        raise ValueError(f"n_ref must be >= 1, got {n_ref}")
    return n_ref


def resolve_sigma_obs(depths, n_stations, when, cfg, table):
    """Per-observation sigma for one analysis, aligned with the obs vector.

    `depths` are the obs depths being assimilated, `n_stations` how many stations backed each (1 on
    a single-station lake), `when` the analysis instant, whose month selects the season -- for the
    table, and for `sigma_obs` itself when that is season-keyed. Returns a plain float when no table
    applies, so the scalar path stays scalar and old behaviour is untouched."""
    sigma_obs = resolve_scalar_sigma_obs(cfg, when.month)
    if table is None:
        return sigma_obs

    n_ref = table_n_ref(table)
    n = np.asarray(n_stations, dtype=float)
    n = np.where(n >= 1, n, 1.0)                       # a reading exists, so at least one station
    out = np.empty(len(depths), dtype=float)
    for i, d in enumerate(depths):
        rep = _sigma_rep_at(table, d, when.month)
        # No fitted entry for this depth -> keep the scalar. Mixing a fitted depth and a fallback
        # depth in one vector is fine and intended.
        sc = resolve_sigma_common(cfg, d)
        ks = resolve_sigma_rep_scale(cfg, d, when.month)
        scale = 1.0 / n[i] if n_ref is None else n_ref / n[i]
        out[i] = (sigma_obs if rep is None
                  else float(np.sqrt(sc ** 2 + (ks * rep) ** 2 * scale)))
    return out


def sigma_obs_by_depth(obs_df, cfg, table):
    """One sigma per depth for OpenDA, whose stochObserver carries a single standardDeviation per
      depth time series (openda/config.py::_obs_formatter_rows).

      THE FALLBACK, not the normal path: adapter.py tries sigma_obs_by_depth_season first and lands
      here only when that declines. The number is the RMS, over the observations actually present, of
      the per-observation sigma the native engine would have used.

      OFTEN STILL EXACT. Two of the season split's guards decline for lack of anything to split, not
      for lack of precision: an annual scalar with no table is one number, so its RMS is itself, and
      a table whose two seasons are equal collapses the same way.

      GENUINELY APPROXIMATE in two cases, and the engines then differ by design:
        n_stations > 1   the native engine divides sigma_rep by each observation's own N; one number
                         per depth cannot. Multi-station lakes only.
        monthly table    finer than seasonal, so sigma varies WITHIN a season. No table in use today
                         is -- sigma_rep_gap.json is 'resolution: seasonal'.
      Both are logged rather than hidden, and either beats leaving OpenDA on the scalar 0.5 while the
      native engine uses the table, which would make any cross-engine comparison meaningless."""
      
    if table is None or obs_df.empty:
        return {float(d): _scalar_sigma_rms(cfg, g["time"].dt.month.to_numpy())
                for d, g in obs_df.groupby("depth")}

    n_ref = table_n_ref(table)
    out = {}
    for d, g in obs_df.groupby("depth"):
        months = g["time"].dt.month.to_numpy()
        n = g["n_stations"].to_numpy(dtype=float) if "n_stations" in g else np.ones(len(g))
        sigma_common = resolve_sigma_common(cfg, d)
        vals = []
        for m, ni in zip(months, np.where(n >= 1, n, 1.0)):
            rep = _sigma_rep_at(table, d, m)
            # inside the loop: the scale may itself be season-keyed, and this RMS is taken over
            # observations spanning both seasons
            rep_scale = resolve_sigma_rep_scale(cfg, d, m)
            scale = 1.0 / ni if n_ref is None else n_ref / ni
            vals.append(resolve_scalar_sigma_obs(cfg, m) if rep is None
                        else np.sqrt(sigma_common ** 2 + (rep_scale * rep) ** 2 * scale))
        out[float(d)] = float(np.sqrt(np.mean(np.square(vals))))
    return out


def sigma_obs_by_depth_season(obs_df, cfg, table):
    """{(depth, season): sigma} when OpenDA can reproduce the native sigma exactly, else None.

    The stochObserver allows one standardDeviation per SERIES, so one series per (depth, season)
    expresses the season axis without sigma_obs_by_depth's RMS compromise. Exact only because the
    table is a step function in season (by_month is a fan-out of by_season).

    Returns None rather than approximating: a genuinely monthly table or n_stations > 1 would make
    two series a silent approximation. Guards -- one sigma per (depth, season), N == 1, and the
    seasons actually differ -- each fall back to the per-depth RMS.

    A (depth, season) with no observations is simply not emitted, so a window covering one season,
    or a depth that came online mid-year, still gets exact sigma at every depth that has data.

    A season-keyed scalar `sigma_obs` with NO table is the other case this expresses exactly, and
    the reason `table is None` is not disqualifying on its own: the loop below then takes the
    rep-is-None branch at every depth, which is the season's scalar. Without this OpenDA would fall
    back to sigma_obs_by_depth's RMS over the two seasons while the native engine used each season's
    own value -- a silent divergence between the engines on the same config.
    """
    if obs_df.empty or (table is None and not _is_season_keyed(cfg["sigma_obs"])):
        return None
    if "n_stations" in obs_df and not np.all(obs_df["n_stations"].to_numpy(dtype=float) == 1.0):
        logger.info("[sigma] OpenDA season split declined: n_stations > 1")
        return None

    out, differs = {}, False
    for d, g in obs_df.groupby("depth"):
        sc = resolve_sigma_common(cfg, d)
        per_season = {}
        for season, gs in g.groupby(g["time"].dt.month.map(season_of)):
            vals = set()
            for m in gs["time"].dt.month.unique():
                rep = None if table is None else _sigma_rep_at(table, d, m)
                # a season-keyed scale is constant within the season, so this stays exact
                ks = resolve_sigma_rep_scale(cfg, d, m)
                vals.add(resolve_scalar_sigma_obs(cfg, m) if rep is None
                         else round(float(np.sqrt(sc ** 2 + (ks * rep) ** 2)), 10))
            if len(vals) != 1:      # table is not a step function in season
                logger.info(f"[sigma] OpenDA season split declined: depth {d:g} m has {len(vals)} "
                            f"distinct sigma within the {season} season")
                return None
            per_season[season] = vals.pop()
        differs = differs or len(set(per_season.values())) > 1
        for season, sig in per_season.items():
            out[(float(d), season)] = float(sig)

    if not differs:
        logger.info("[sigma] OpenDA season split declined: sigma identical in both seasons")
        return None
    return out


def _depth_steps_txt(spec):
    """'>=0m: 1.52, >=4m: 2.61' for a depth-keyed step function."""
    return ", ".join(f">={float(k):g}m: {float(v)}"
                     for k, v in sorted(spec.items(), key=lambda kv: float(kv[0])))


def _scale_txt(spec):
    """sigma_rep_scale for the log line. Season-keyed is handled separately from depth-keyed:
    its keys are season names, so sorting them as floats raises (see resolve_sigma_rep_scale)."""
    if not isinstance(spec, dict):
        return f"{spec:g}"
    if _is_season_keyed(spec):
        return "season-keyed " + "; ".join(
            f"{s}: " + (_depth_steps_txt(v) if isinstance(v, dict) else f"{float(v):g}")
            for s, v in sorted(spec.items()))
    return "depth-stepped " + _depth_steps_txt(spec)


def log_sigma_summary(cfg, table, by_depth=None):
    """Say, once, what observation-error model this run is actually using — the counterpart to the
    'sigma_obs=' knob in the run header, which no longer tells the whole story."""
    if table is None:
        spec = cfg["sigma_obs"]
        txt = ("season-keyed " + ", ".join(f"{s}: {float(v):g}" for s, v in sorted(spec.items()))
               if _is_season_keyed(spec) else f"{spec} degC")
        logger.info(f"[sigma] scalar sigma_obs={txt} (no sigma_rep table)")
        return
    depths = sorted(table["sigma_rep"], key=float)
    sc = cfg.get("sigma_common", DEFAULT_SIGMA_COMMON)
    sc_txt = (f"{sc} degC" if not isinstance(sc, dict) else
              "depth-stepped " + ", ".join(f">={float(k):g}m: {float(v)}"
                                           for k, v in sorted(sc.items(), key=lambda kv: float(kv[0]))))
    ks_txt = _scale_txt(cfg.get("sigma_rep_scale", 1.0))
    logger.info(f"[sigma] sigma^2 = sigma_common^2 + (scale * sigma_rep(depth, month))^2 / n_stations"
                f"  (sigma_common={sc_txt}, scale={ks_txt}, "
                f"fallback sigma_obs={cfg['sigma_obs']} degC)")
    logger.info(f"[sigma] sigma_rep fitted at {len(depths)} depth(s) "
                f"{depths[0]}-{depths[-1]} m <- {os.path.relpath(sigma_rep_path(cfg), ROOT)}"
                + (f" ({table['estimator'][:60]}...)" if table.get("estimator") else ""))
    if by_depth:
        logger.info("[sigma] effective sigma by depth (RMS over the obs): "
                    + ", ".join(f"{d:g}m={s:.3f}" for d, s in sorted(by_depth.items())))

"""The observation error sigma_obs: one number, or one per season {"mixed": .., "stratified": ..}.

sigma_obs stands for representativeness error, which is far larger in the stratified season (1.1x
to 2.6x across the 2025 lakes), so one annual value is wrong in both seasons. A sigma is valid only
for the observation series it was fitted on (assimilator.fitting.sigma).
"""
import logging

logger = logging.getLogger(__name__)

# Seasons on the calendar: a stratification rule was tried and did not beat it, and a calendar
# split keys the fit and the run configs the same way without needing the run's own state.
SEASONS      = ("mixed", "stratified")
STRAT_MONTHS = range(5, 11)

# A key of its own rather than a default for sigma_obs, so the log can tell when it is used.
SIGMA_DEFAULT_KEY = "sigma_default"
SIGMA_DEFAULT     = 0.5


def season_of(month):
    return "stratified" if int(month) in STRAT_MONTHS else "mixed"


def is_season_keyed(spec):
    return isinstance(spec, dict) and bool(spec) and set(spec) <= set(SEASONS)


def sigma_obs_spec(cfg):
    """The run's sigma_obs as configured, or sigma_default when it names none."""
    spec = cfg.get("sigma_obs")
    return cfg.get(SIGMA_DEFAULT_KEY, SIGMA_DEFAULT) if spec is None else spec


def _sigma_in(cfg, month):
    spec = sigma_obs_spec(cfg)
    if is_season_keyed(spec):
        season = season_of(month)
        if season not in spec:
            raise ValueError(f"sigma_obs has no {season!r} entry (has {sorted(spec)})")
        spec = spec[season]
    return float(spec)


def resolve_sigma_obs(depths, when, cfg):
    """One sigma per observation depth for the analysis at `when` (its month picks the season)."""
    return [_sigma_in(cfg, when.month)] * len(depths)


def should_season_split(cfg):
    """Whether OpenDA gets one observation series per (depth, season). OpenDA carries one
    standardDeviation per series, so this reproduces the native engine's seasonal sigma exactly."""
    spec = sigma_obs_spec(cfg)
    if not is_season_keyed(spec):
        return False
    if len({float(v) for v in spec.values()}) == 1:
        logger.info("[sigma] OpenDA season split declined: sigma identical in both seasons")
        return False
    return True


def sigma_obs_for_series(series_keys, cfg):
    """{(depth, season): sigma} for the series files the OpenDA adapter wrote. Keyed off the files
    so the declared series and the files cannot disagree, which OpenDA does not survive."""
    spec = sigma_obs_spec(cfg)
    return {(float(d), s): float(spec[s]) for d, s in series_keys}


def log_sigma_summary(cfg):
    spec = sigma_obs_spec(cfg)
    src  = "config" if cfg.get("sigma_obs") is not None else f"{SIGMA_DEFAULT_KEY} fallback"
    txt  = ("season-keyed " + ", ".join(f"{s}: {float(v):g}" for s, v in sorted(spec.items()))
            if is_season_keyed(spec) else f"{float(spec):g} degC")
    logger.info(f"[sigma] sigma_obs={txt} (from {src})")

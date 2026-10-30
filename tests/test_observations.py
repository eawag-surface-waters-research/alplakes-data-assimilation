"""Observation preprocessing: filter, depth grid, thinning, and tail reprocessing."""
import numpy as np
import pandas as pd
import pytest

from assimilator.observations import (filter_seiche, filter_params, select_depths, thin_daily,
                                      preprocess, lookback)


def _raw(days=6, every_min=10, depths=(1.0, 5.0, 10.0), seed=0):
    rng = np.random.default_rng(seed)
    t = pd.date_range("2025-07-01", periods=int(days * 24 * 60 / every_min), freq=f"{every_min}min",
                      tz="UTC") + pd.Timedelta(minutes=3)
    rows = []
    for d in depths:
        seiche = 0.8 * np.sin(2 * np.pi * np.arange(len(t)) * every_min / 60 / 12.0) if d >= 4 else 0
        v = 20 - d * 0.5 + seiche + rng.normal(0, 0.05, len(t))
        rows.append(pd.DataFrame({"time": t, "depth": d, "value": v}))
    return pd.concat(rows, ignore_index=True)


def test_filter_passes_shallow_depths_through_and_smooths_deep_ones():
    raw = _raw()
    out, report = filter_seiche(raw, {"W_SEICHE": 12.0, "THERMO_DEPTH_MIN": 4.0})
    rep = report.set_index("depth")
    assert rep.loc[1.0, "window_h"] == 1.0
    assert rep.loc[10.0, "window_h"] == 12.0
    # the 12 h seiche is removed at depth, untouched at 1 m
    assert out[out.depth == 10.0]["value"].std() < 0.2 * raw[raw.depth == 10.0]["value"].std()
    assert rep.loc[1.0, "removed_std"] < 0.05
    # output only at sampled instants, on the hourly grid
    assert (pd.to_datetime(out["time"]).dt.minute == 0).all()


def test_filter_withholds_values_after_a_gap():
    raw = _raw(days=4)
    gap = (raw.time > "2025-07-02 06:00") & (raw.time < "2025-07-02 18:00") & (raw.depth == 10.0)
    raw = raw[~gap]
    out, report = filter_seiche(raw, {"W_SEICHE": 12.0, "THERMO_DEPTH_MIN": 4.0})
    r = report.set_index("depth").loc[10.0]
    assert r["withheld_thin"] + r["withheld_stale"] > 0
    # nothing is emitted inside the gap (output only where the instrument sampled)
    deep = pd.to_datetime(out[out.depth == 10.0]["time"])
    assert not ((deep > "2025-07-02 07:00") & (deep < "2025-07-02 18:00")).any()


def test_filter_rejects_an_already_thinned_series():
    raw = _raw()
    daily = raw[pd.to_datetime(raw.time).dt.hour == 6].groupby(["depth"]).head(1)
    with pytest.raises(ValueError, match="samples per depth per day"):
        filter_seiche(daily)


def test_thin_keeps_one_whole_profile_per_day_nearest_the_hour():
    raw = _raw(days=3, every_min=60)
    out = thin_daily(raw, hour=6)
    t = pd.to_datetime(out["time"])
    assert t.dt.floor("D").nunique() == 3
    assert out.groupby(t)["depth"].count().eq(3).all()          # all depths of the winner
    assert (t.dt.hour == 6).all()


def test_thin_drops_days_without_an_observation_near_the_hour():
    raw = _raw(days=3, every_min=60)
    t = pd.to_datetime(raw["time"])
    raw = raw[~((t.dt.day == 2) & (t.dt.hour.between(3, 9)))]   # nothing within 1.5 h on day 2
    out = thin_daily(raw, hour=6, max_offset=1.5)
    days = pd.to_datetime(out["time"]).dt.day.unique()
    assert 2 not in days and {1, 3} <= set(days)


def test_thin_hour_window_takes_the_first_instant_in_the_window():
    raw = _raw(days=2, every_min=60)
    out = thin_daily(raw, hour_window=(2, 6))
    assert (pd.to_datetime(out["time"]).dt.hour == 2).all()


def test_select_depths_is_a_target_grid():
    raw = _raw(depths=(1.0, 5.2, 10.0))
    sel = select_depths(raw, [1, 5, 10], tol=0.5)
    assert sorted(sel["depth"].unique()) == [1.0, 5.2, 10.0]     # depth values are not modified
    with pytest.raises(ValueError):
        select_depths(raw, [1, 7], tol=0.5)
    assert sorted(select_depths(raw, [1, 7], tol=0.5, strict=False)["depth"].unique()) == [1.0]


CFG = {"filter": {"w_seiche_h": 12.0, "thermo_depth_min": 4.0, "derived_on": "x"},
       "schedule": {"hour": 6, "max_offset_h": 1.5}}


def test_filter_params_reads_config_names_and_ignores_provenance():
    p = filter_params(CFG["filter"])
    assert p["W_SEICHE"] == 12.0 and p["THERMO_DEPTH_MIN"] == 4.0 and "derived_on" not in p


def test_tail_reprocessing_matches_a_full_reprocess():
    raw = _raw(days=10)
    full = preprocess(raw, CFG)
    for since in ["2025-07-04", "2025-07-08"]:
        since = pd.Timestamp(since, tz="UTC")
        tail = preprocess(raw[raw.time >= since - lookback(CFG)], CFG, since=since)
        exp = full[full["time"] >= since].reset_index(drop=True)
        assert len(tail) == len(exp) > 0
        assert (tail["time"].values == exp["time"].values).all()
        assert (tail["depth"].values == exp["depth"].values).all()
        # the box filter uses running sums, so float rounding depends on the chunk start
        np.testing.assert_allclose(tail["value"].values, exp["value"].values, rtol=0, atol=1e-9)

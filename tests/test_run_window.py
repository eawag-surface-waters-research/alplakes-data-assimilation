"""The run window and observation times are the same for both engines: observations in
(start_date, end_date], each at its own hour, and the members' final state at end_date."""
import os
from datetime import datetime, timezone

import pandas as pd

import assimilator
from assimilator.functions import load_obs, obs_window_boundaries
from assimilator.models import MODELS
from assimilator.openda.adapter import _build_observations
from assimilator.openda.config import render, simstrat_time, format_day
from toy_model import ToyModel

UTC = timezone.utc


def _write_obs(path, rows):
    pd.DataFrame(rows, columns=["time", "depth", "value"]).to_csv(path, index=False)


# Irregular observations: several on one day (01:40 bins to 02:00), a gap, one at the run's end
# instant, one after the end, and one exactly at the start (the previous run's last analysis).
IRREGULAR = [("2025-01-09 00:00", 1.0, 99.0),
             ("2025-01-10 01:00", 1.0, 5.0), ("2025-01-10 01:40", 1.0, 5.5),
             ("2025-01-10 07:15", 1.0, 6.0), ("2025-01-10 23:00", 1.0, 6.5),
             ("2025-01-14 06:00", 1.0, 7.0), ("2025-01-15 00:00", 1.0, 7.5),
             ("2025-01-16 06:00", 1.0, 99.0)]
START, END = "2025-01-09", "2025-01-15"
EXPECTED = ["2025-01-10 01:00", "2025-01-10 02:00", "2025-01-10 07:00", "2025-01-10 23:00",
            "2025-01-14 06:00", "2025-01-15 00:00"]


def test_native_window_is_start_exclusive_end_inclusive(tmp_path):
    _write_obs(tmp_path / "obs.csv", IRREGULAR)
    bounds = obs_window_boundaries(load_obs(tmp_path / "obs.csv"),
                                   datetime(2025, 1, 9, tzinfo=UTC), datetime(2025, 1, 15, tzinfo=UTC))
    assert [b.strftime("%Y-%m-%d %H:%M") for b in bounds] == EXPECTED


def test_openda_takes_every_observation_at_its_own_hour(tmp_path):
    _write_obs(tmp_path / "obs.csv", IRREGULAR)
    os.makedirs(tmp_path / "inputs")
    raw = {"lake": "x", "obs_file": str(tmp_path / "obs.csv"), "start_date": START, "end_date": END}
    depths, n_analysis, series = _build_observations(raw, str(tmp_path / "oda"), str(tmp_path / "inputs"))

    assert depths == [1.0] and series == [(1.0, None)] and n_analysis == len(EXPECTED)
    written = pd.read_csv(tmp_path / "oda" / "stochObserver" / "T_1m_real.csv", dtype={"time": str})
    assert list(written["time"]) == [format_day(simstrat_time(t)) for t in EXPECTED]
    # same hourly binning as the native engines: 01:40 is the 02:00 observation, nothing at noon
    assert list(written["value"]) == [5.0, 5.5, 6.0, 6.5, 7.0, 7.5]


def test_openda_season_split_partitions_the_observations(tmp_path):
    _write_obs(tmp_path / "obs.csv", [("2025-04-30 06:00", 1.0, 5.0), ("2025-04-30 18:00", 1.0, 5.5),
                                      ("2025-05-01 06:00", 1.0, 6.0)])
    os.makedirs(tmp_path / "inputs")
    raw = {"lake": "x", "obs_file": str(tmp_path / "obs.csv"), "start_date": "2025-04-29",
           "end_date": "2025-05-02", "_season_series": True}
    _, n_analysis, series = _build_observations(raw, str(tmp_path / "oda"), str(tmp_path / "inputs"))
    assert series == [(1.0, "mixed"), (1.0, "stratified")] and n_analysis == 3
    mixed = pd.read_csv(tmp_path / "oda" / "stochObserver" / "T_1m_mixed_real.csv")
    assert list(mixed["value"]) == [5.0, 5.5]


def test_openda_runs_to_end_date_not_the_day_after(tmp_path):
    render(str(tmp_path), "EnKF", 4, [1.0], START, END, obs_std=0.5)
    line = (tmp_path / "stochModel" / "template" / "time_control.yaml").read_text().splitlines()[-1]
    assert line.startswith(f"time: [{format_day(simstrat_time(START))}, 1.0, {format_day(simstrat_time(END))}]")
    assert simstrat_time(END) - simstrat_time(START) == 6.0


def test_simstrat_time_is_exact_for_instants():
    assert simstrat_time("1981-01-02") == 1.0
    assert simstrat_time("1981-01-02T06:00") == 1.25
    assert simstrat_time(pd.Timestamp("1981-01-02 07:00", tz="Europe/Zurich")) == 1.25


def test_run_reports_the_time_of_its_final_state(tmp_path):
    from test_engine_toy_model import _cfg
    assimilator.register_model(ToyModel)
    try:
        result = assimilator.run(dict(_cfg(tmp_path, "EnKF"), end_date="2024-06-03"))
    finally:
        MODELS.pop("toy", None)
    assert result["analysis_time"] == datetime(2024, 6, 3, tzinfo=UTC)

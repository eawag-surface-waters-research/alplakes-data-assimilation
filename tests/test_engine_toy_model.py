"""The EnKF and PF run end to end on a pure-Python model: no Docker, no Simstrat files."""
import os
import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

import assimilator
from assimilator.models import MODELS
from toy_model import ToyModel, write_inputs, OUT_DEPTHS

START = datetime(2024, 6, 1, tzinfo=timezone.utc)
PERT  = {"variables": {"U": {"phi": 0.8, "sigma": 0.5},
                       "V": {"phi": 0.8, "sigma": 0.5},
                       "GLOB": {"phi": 0.9, "sigma": 30.0}}}


@pytest.fixture(autouse=True)
def toy_registered():
    assimilator.register_model(ToyModel)
    yield
    MODELS.pop("toy", None)


def _obs(path, value=14.0):
    """Daily 06:00 profiles of a lake warmer than the toy's warm start, at three depths."""
    times = pd.date_range(START + pd.Timedelta(hours=6), periods=4, freq="1D")
    pd.DataFrame([(t, d, value) for t in times for d in (1.0, 5.0, 10.0)],
                 columns=["time", "depth", "value"]).to_csv(path, index=False)


def _cfg(tmp_path, algorithm):
    write_inputs(str(tmp_path / "inputs"), START, days=6)
    _obs(tmp_path / "obs.csv")
    return {"lake": "toylake", "model": "toy", "algorithm": algorithm, "n_members": 6,
            "start_date": "2024-06-01", "end_date": "2024-06-05", "sigma_obs": 0.2,
            "model_inputs_path": str(tmp_path / "inputs"), "ensemble_base": str(tmp_path / "run"),
            "obs_file": str(tmp_path / "obs.csv"), "perturbations": PERT}


@pytest.mark.parametrize("algorithm", ["EnKF", "PF"])
def test_engine_runs_on_toy_model(tmp_path, algorithm):
    cfg = _cfg(tmp_path, algorithm)
    before = {f: os.path.getmtime(tmp_path / "inputs" / f) for f in os.listdir(tmp_path / "inputs")}

    result = assimilator.run(cfg)

    assert os.path.isfile(result["mean_trajectory"])
    summary = tmp_path / "run" / f"toylake_python_{algorithm}.json"
    assert summary.is_file()
    assert json.loads(summary.read_text())["n_obs"] > 0
    # inputs are read-only
    assert before == {f: os.path.getmtime(tmp_path / "inputs" / f)
                      for f in os.listdir(tmp_path / "inputs")}
    # members got distinct (perturbed) forcing
    g1 = pd.read_csv(tmp_path / "run" / "ensemble1" / "forcing.csv")["GLOB"]
    g2 = pd.read_csv(tmp_path / "run" / "ensemble2" / "forcing.csv")["GLOB"]
    assert not np.allclose(g1, g2)


def test_enkf_pulls_state_toward_observations(tmp_path):
    cfg = _cfg(tmp_path, "EnKF")
    assimilator.run(cfg)
    analysed = assimilator.analysis_mean_state(cfg, str(tmp_path / "mean_state.npy"))
    T_da = np.load(analysed)

    free = _cfg(tmp_path / "free", "EnKF")
    free["sigma_obs"] = 1e6                     # observations carry no weight: a free ensemble
    assimilator.run(free)
    T_free = np.load(assimilator.analysis_mean_state(free, str(tmp_path / "free_state.npy")))

    from toy_model import Z_VOLUME, LAKE_LEVEL
    depth = LAKE_LEVEL - Z_VOLUME
    at = [int(np.argmin(np.abs(depth - d))) for d in (1.0, 5.0, 10.0)]
    assert np.abs(T_da[at] - 14.0).mean() < np.abs(T_free[at] - 14.0).mean()


def test_continuation_matches_one_continuous_run(tmp_path):
    """Two slices (reset off) == one run over the same period: time-keyed randomness."""
    one = _cfg(tmp_path / "one", "EnKF")
    assimilator.run(one)

    two = _cfg(tmp_path / "two", "EnKF")
    first = dict(two, end_date="2024-06-03")
    assimilator.run(first)
    assimilator.run(dict(two, start_date="2024-06-03"))

    a = pd.read_csv(tmp_path / "one" / "run" / "out_enkf_mean.csv")
    b = pd.read_csv(tmp_path / "two" / "run" / "out_enkf_mean.csv")
    pd.testing.assert_frame_equal(a, b)
    assert OUT_DEPTHS

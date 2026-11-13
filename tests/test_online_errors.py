"""Online R and q: the learning rules, the noise, stabilisation, config, and toy-model runs."""
import numpy as np
import pandas as pd
import pytest

import assimilator
from assimilator.config import build_config
from assimilator.online_errors import (OnlineErrors, DEFAULTS, LOG_FILENAME, draw_noise,
                                       stabilize, water_density)
from test_engine_toy_model import _cfg, toy_registered  # noqa: F401 (fixture)

DEPTHS = [1.0, 5.0, 10.0]
CORR   = {"depths": DEPTHS,
          "corr": {s: np.eye(3).tolist() for s in ("mixed", "stratified")}}


def _spec(**kw):
    return {**DEFAULTS, "warmup_days": 0, "check_after_days": 0, "correlation": CORR, **kw}


def test_consistency_rule_learns_the_true_error():
    oe = OnlineErrors(_spec(R_rule="consistency", q=False), [5.0])
    rng, s2, R_true = np.random.default_rng(0), 0.04, 0.3
    t, R = pd.Timestamp("2025-01-01"), []
    for _ in range(3000):
        d = rng.normal(0, np.sqrt(s2 + R_true ** 2))
        oe.learn(t, [5.0], [d], [np.nan], [s2], oe.R([5.0]))
        R.append(oe.R([5.0])[0])
        t += pd.Timedelta(days=1)
    assert abs(np.mean(R[1000:]) - R_true) < 0.03


def test_no_noise_without_q():
    oe = OnlineErrors(_spec(q=False), DEPTHS, corr={s: np.eye(3) for s in ("mixed", "stratified")})
    oe.q2[:] = 0.5
    assert oe.noise_factor("2025-07-01") is None


def test_noise_is_centred_and_tapered():
    L = np.diag([0.5, 0.3, 0.2])
    depth = np.linspace(0, 30, 61)
    e = draw_noise(L, np.array(DEPTHS), depth, 8, "2025-07-01T06:00", seed=1, taper_m=10.0)
    assert np.allclose(e.mean(axis=1), 0.0)
    assert np.all(e[depth >= 20.0] == 0.0)
    assert np.allclose(e[depth <= 1.0], e[depth == 1.0])          # constant above the grid


def test_stabilize_removes_inversions_and_keeps_heat():
    depth = np.arange(0.5, 20, 1.0)
    T = np.linspace(20, 6, len(depth))
    T[5], T[12] = 8.0, 15.0                                         # cold above, warm below
    V = np.random.default_rng(2).uniform(1, 3, len(depth))
    out, n = stabilize(T, V, depth, z_max=30.0)
    assert n > 0
    assert np.isclose((out * V).sum(), (T * V).sum())
    rho = water_density(out[np.argsort(depth)])
    assert np.all(np.diff(rho) >= -0.001)


def test_fit_noise_correlation_recovers_neighbour_correlation():
    from assimilator.fitting.noise_correlation import fit_noise_correlation
    from assimilator.online_errors import load_correlation
    rng = np.random.default_rng(3)
    times = pd.date_range("2025-01-01 06:00", "2025-12-31 06:00", freq="1D", tz="UTC")
    shared = rng.normal(size=len(times))
    eta = np.column_stack([shared + 0.3 * rng.normal(size=len(times)) for _ in range(2)]
                          + [rng.normal(size=len(times))])          # 1 m and 5 m move together
    free = pd.DataFrame(10.0, index=times, columns=DEPTHS)
    obs = pd.DataFrame([(t, d, 10.0 + 0.3 * e) for t, row in zip(times, eta) for d, e in zip(DEPTHS, row)],
                       columns=["time", "depth", "value"])
    out = fit_noise_correlation(obs, free)
    depths, corr = load_correlation({"correlation": out})
    for C in corr.values():
        assert C[0, 1] > 0.7 and abs(C[0, 2]) < 0.3


def _base(tmp_path):
    return {"lake": "x", "algorithm": "EnKF", "n_members": 4, "start_date": "2025-01-01",
            "end_date": "2025-01-02", "model_inputs_path": str(tmp_path), "ensemble_base": str(tmp_path),
            "obs_file": str(tmp_path / "o.csv"), "perturbations": {"variables": {}}}


@pytest.mark.parametrize("block, extra", [
    ({"R_rule": "nope", "q": False}, {}),
    ({"q": True}, {}),                                              # q needs a correlation
    ({"q": False, "lamda": 0.9}, {}),                               # unknown key
    ({"q": False}, {"engine": "openda"}),                           # native EnKF only
    ({"q": False, "lambda": 1.0}, {}),
])
def test_bad_online_errors_config_is_rejected(tmp_path, block, extra):
    with pytest.raises(ValueError):
        build_config({**_base(tmp_path), **extra, "online_errors": block})


def test_online_errors_defaults_are_filled(tmp_path):
    cfg = build_config({**_base(tmp_path), "online_errors": {"q": False}})
    assert cfg["online_errors"]["R_rule"] == "split" and cfg["online_errors"]["lambda"] == 0.95


def _online_cfg(tmp_path):
    return dict(_cfg(tmp_path, "EnKF"), online_errors=_spec(max_nis=1e9))


def test_toy_run_learns_q_and_logs(tmp_path):
    assimilator.run(_online_cfg(tmp_path))
    log = pd.read_csv(tmp_path / "run" / LOG_FILENAME)
    assert len(log) == 4
    assert (log.filter(like="q_").iloc[-1] > 0).any()               # the toy misses 14 degC: q grows


def test_online_continuation_matches_one_continuous_run(tmp_path):
    """Two slices (reset off) == one run: R, q and the noise carry over."""
    assimilator.run(_online_cfg(tmp_path / "one"))
    two = _online_cfg(tmp_path / "two")
    assimilator.run(dict(two, end_date="2024-06-03"))
    assimilator.run(dict(two, start_date="2024-06-03"))

    for f in ("out_enkf_mean.csv", LOG_FILENAME):
        pd.testing.assert_frame_equal(pd.read_csv(tmp_path / "one" / "run" / f),
                                      pd.read_csv(tmp_path / "two" / "run" / f))

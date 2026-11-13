"""Vertical localization: the taper, the localized update, config, the radius fit, toy-model runs."""
import json

import numpy as np
import pandas as pd
import pytest

import assimilator
from assimilator.algorithms.enkf import enkf_update
from assimilator.config import build_config
from assimilator.localization import gaspari_cohn, taper, load_radii
from assimilator.fitting.localization import radii, fit_localization
from test_engine_toy_model import _cfg, toy_registered  # noqa: F401 (fixture)

FLAT = (np.array([0.0]), np.array([5.0]))


def test_gaspari_cohn_shape():
    r = np.linspace(0, 3, 301)
    g = gaspari_cohn(r)
    assert g[0] == 1.0 and np.all(g[r >= 2] == 0.0)
    assert np.all(np.diff(g) <= 1e-12)                          # never increases with distance


def test_taper_is_symmetric_and_positive_semidefinite():
    z = np.arange(0.5, 40, 0.5)
    table = (np.array([0.0, 10.0, 40.0]), np.array([3.0, 6.0, 12.0]))
    rho = taper(z, z, *table)
    assert np.allclose(rho, rho.T) and np.allclose(np.diag(rho), 1.0)
    assert np.linalg.eigvalsh(rho).min() > -1e-3
    assert taper([0.5], [20.0], *table)[0, 0] == 0.0             # beyond the radius


def _ensemble(n_cells=30, n_members=20, seed=0):
    rng = np.random.default_rng(seed)
    X = 10 + rng.normal(size=(n_cells, n_members)).cumsum(axis=0) * 0.1
    H = np.zeros((2, n_cells)); H[0, 2] = H[1, 5] = 1.0
    return X, H, np.array([11.0, 9.0])


def test_taper_of_ones_changes_nothing():
    X, H, y = _ensemble()
    a, _ = enkf_update(X, y, H, 0.2, rng=np.random.default_rng(1))
    b, _ = enkf_update(X, y, H, 0.2, rng=np.random.default_rng(1),
                       loc_so=np.ones((X.shape[0], 2)), loc_oo=np.ones((2, 2)))
    assert np.array_equal(a, b)


def test_cells_out_of_reach_are_not_updated_nor_inflated():
    X, H, y = _ensemble()
    depth = np.arange(X.shape[0], dtype=float)                  # cell i at i metres
    obs_depths = [2.0, 5.0]
    X_a, _ = enkf_update(X, y, H, 0.2, inflation=1.2, rng=np.random.default_rng(1),
                         loc_so=taper(depth, obs_depths, *FLAT), loc_oo=taper(obs_depths, obs_depths, *FLAT))
    far = depth >= 10.0                                         # 5 m radius: nothing past 10 m
    assert np.array_equal(X_a[far], X[far])
    assert not np.allclose(X_a[~far], X[~far])


def test_flat_and_tabled_radii(tmp_path):
    assert np.allclose(load_radii({"radius_m": 5.0})[1], 5.0)
    p = tmp_path / "r.json"
    p.write_text(json.dumps({"depths_m": [0, 10], "radius_m": [2, 8]}))
    z, r = load_radii({"radius_file": str(p)})
    assert list(z) == [0, 10] and list(r) == [2, 8]


def _base(tmp_path):
    return {"lake": "x", "algorithm": "EnKF", "n_members": 4, "start_date": "2025-01-01",
            "end_date": "2025-01-02", "model_inputs_path": str(tmp_path), "ensemble_base": str(tmp_path),
            "obs_file": str(tmp_path / "o.csv"), "perturbations": {"variables": {}}}


@pytest.mark.parametrize("block, extra", [
    ({}, {}),                                                   # needs a radius
    ({"radius_m": 5, "radius_file": "r.json"}, {}),             # not both
    ({"radius_m": [3, 5]}, {}),                                 # a list needs depths_m
    ({"radius_m": 0}, {}),
    ({"radius": 5}, {}),                                        # unknown key
    ({"radius_m": 5}, {"algorithm": "PF"}),                     # native EnKF only
])
def test_bad_localization_config_is_rejected(tmp_path, block, extra):
    with pytest.raises(ValueError):
        build_config({**_base(tmp_path), **extra, "localization": block})


def test_fit_output_is_a_valid_inline_block(tmp_path):
    out = fit_localization(pd.date_range("2025-05-01", "2025-07-01", freq="1h"), np.array([0.0, 1.0]),
                           np.random.default_rng(0).normal(size=(5, 1465, 2)))
    cfg = build_config({**_base(tmp_path), "localization": out})
    assert cfg["localization"]["radius_m"] == out["radius_m"]


def test_radius_fit_on_a_known_correlation():
    depths = np.arange(0.0, 10.0, 1.0)
    corr = np.exp(-np.abs(depths[:, None] - depths[None, :]) / 3.0)   # 0.45 at ~2.4 m
    assert np.allclose(radii(corr, depths), 3.0)


def test_fit_localization_from_an_ensemble():
    rng = np.random.default_rng(0)
    times = pd.date_range("2025-05-01", "2025-07-01", freq="1h")
    shared = rng.normal(size=(10, len(times)))
    arr = np.stack([shared] * 2 + [rng.normal(size=(10, len(times)))], axis=2)   # 0 and 1 m move together
    out = fit_localization(times, np.array([0.0, 1.0, 5.0]), arr)
    assert out["radius_m"] == [5.0, 4.0, 4.0]


def test_toy_run_with_localization(tmp_path):
    cfg = dict(_cfg(tmp_path, "EnKF"), localization={"radius_m": 3.0})
    assimilator.run(cfg)
    assert (tmp_path / "run" / "out_enkf_mean.csv").is_file()


def test_huge_radius_matches_no_localization(tmp_path):
    assimilator.run(_cfg(tmp_path / "off", "EnKF"))
    assimilator.run(dict(_cfg(tmp_path / "on", "EnKF"), localization={"radius_m": 1e9}))
    a = pd.read_csv(tmp_path / "off" / "run" / "out_enkf_mean.csv")
    b = pd.read_csv(tmp_path / "on" / "run" / "out_enkf_mean.csv")
    assert np.allclose(a.to_numpy(), b.to_numpy(), atol=1e-6)

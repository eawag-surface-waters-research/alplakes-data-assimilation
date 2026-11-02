"""The OpenDA restart chain (assimilator.openda.restart), without OpenDA: planning, naming, the
commit/clean-up rules, appending trajectories, the rendered config, and forcing continuity."""
import os
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pytest

import assimilator
from assimilator.models import MODELS, get_model
from assimilator.openda import restart as rc
from assimilator.openda import adapter
from assimilator.openda.config import render
from assimilator.perturbate import perturbator
from toy_model import ToyModel, write_inputs

UTC = timezone.utc
ID  = {"lake": "x", "filter": "EnKF", "n_members": 4}


def t(s):
    return datetime.fromisoformat(s).replace(tzinfo=UTC)


def test_plan_is_one_run_or_chunks_of_a_catch_up():
    assert rc.plan_cycles("2026-11-01", "2026-11-02", 31) == [(t("2026-11-01"), t("2026-11-02"))]
    assert rc.plan_cycles("2026-01-01", "2026-03-15", 31) == [
        (t("2026-01-01"), t("2026-02-01")), (t("2026-02-01"), t("2026-03-04")), (t("2026-03-04"), t("2026-03-15"))]
    assert rc.plan_cycles("2026-11-02", "2026-11-02", 31) == []


def test_restart_files_are_named_by_utc_instant(tmp_path):
    assert os.path.basename(rc.restart_path(tmp_path, "2026-11-02")) == "restart_20261102T0000Z.zip"
    assert os.path.basename(rc.restart_path(tmp_path, "2026-11-02T01:00:00+01:00")) == "restart_20261102T0000Z.zip"


def _openda_wrote_restart(openda_dir, name="rst_16740.0.zip"):
    os.makedirs(os.path.join(openda_dir, "Results"), exist_ok=True)
    with open(os.path.join(openda_dir, "Results", name), "w") as f:
        f.write("ensemble")


def test_select_commit_and_prune(tmp_path):
    chain = str(tmp_path / "restart")
    assert rc.select_restart(chain, "2026-11-01", ID) == ""          # empty chain: cold start
    for day in ("2026-11-01", "2026-11-02", "2026-11-03"):
        _openda_wrote_restart(str(tmp_path))
        rc.commit_restart(str(tmp_path), chain, day, ID)
    assert rc.chain_end(chain) == t("2026-11-03")
    assert rc.select_restart(chain, "2026-11-03", ID).endswith("restart_20261103T0000Z.zip")
    with pytest.raises(FileNotFoundError):
        rc.select_restart(chain, "2026-10-15", ID)                    # not a committed instant
    with pytest.raises(ValueError):
        rc.select_restart(chain, "2026-11-03", {**ID, "n_members": 20})   # another set-up
    rc.prune(chain, 2)
    assert rc.chain_instants(chain) == [t("2026-11-02"), t("2026-11-03")]
    assert sorted(os.listdir(chain)) == ["restart_20261102T0000Z.json", "restart_20261102T0000Z.zip",
                                         "restart_20261103T0000Z.json", "restart_20261103T0000Z.zip"]


def test_commit_needs_exactly_the_one_restart_this_run_wrote(tmp_path):
    with pytest.raises(RuntimeError):
        rc.commit_restart(str(tmp_path), str(tmp_path / "restart"), "2026-11-02", ID)   # run wrote none
    assert rc.chain_end(str(tmp_path / "restart")) is None


def test_append_trajectory_replaces_a_rerun_instead_of_duplicating(tmp_path):
    chain, run = tmp_path / "chain.dat", tmp_path / "run.dat"
    run.write_text("Datetime,-1\n10.0,5\n10.5,6\n11.0,7\n")
    assert rc.append_trajectory(str(chain), str(run), start_day=10.0) == (0, 2)
    run.write_text("Datetime,-1\n11.0,7\n11.5,8\n12.0,9\n")
    rc.append_trajectory(str(chain), str(run), start_day=11.0)
    run.write_text("Datetime,-1\n11.0,7\n11.5,80\n12.0,90\n")                        # the same run again
    assert rc.append_trajectory(str(chain), str(run), start_day=11.0) == (2, 2)
    assert chain.read_text().splitlines() == ["Datetime,-1", "10.5,6", "11.0,7", "11.5,80", "12.0,90"]


def test_render_restart_chain(tmp_path):
    restart = str(tmp_path / "restart" / "restart_20261102T0000Z.zip")
    render(str(tmp_path), "EnKF", 4, [1.0], "2026-11-02", "2026-11-03", obs_std=0.5, restart=restart, seed=1234)
    oda = (tmp_path / "run.oda").read_text()
    assert "<restartInFile>restart/restart_20261102T0000Z.zip</restartInFile>" in oda
    assert 'seedValue="1234"' in oda and "<restartOutFileOnlyAtEndOfRun>true" in oda
    model = (tmp_path / "stochModel" / "simstratModel.gen.xml").read_text()
    assert "<modelStateFile>Results/simulation-snapshot.dat</modelStateFile>" in model
    assert os.path.isabs(model.split('key="instanceDir"  value="')[1].split('"')[0])

    render(str(tmp_path / "plain"), "EnKF", 4, [1.0], "2026-11-02", "2026-11-03", obs_std=0.5)
    assert "restartInFile" not in (tmp_path / "plain" / "run.oda").read_text()


def test_seed_differs_per_run_and_is_reproducible():
    assert rc.cycle_seed(42, "2026-11-02") == rc.cycle_seed(42, "2026-11-02")
    assert rc.cycle_seed(42, "2026-11-02") != rc.cycle_seed(42, "2026-11-03")
    assert 0 <= rc.cycle_seed(42, "2026-11-02") < 2 ** 31


class _Model:
    def run_config(self):
        return {}


def _chain_cfg(tmp_path, start, end, **restart):
    return {"lake": "x", "model": "simstrat", "algorithm": "EnKF", "n_members": 4,
            "start_date": start, "end_date": end, "openda_dir": str(tmp_path / "oda"),
            "ensemble_base": str(tmp_path / "run"), "openda_restart": restart}


def test_chain_runs_from_its_end_in_chunks(tmp_path, monkeypatch):
    runs = []
    monkeypatch.setattr(adapter, "run_openda", lambda run_cfg, *a, **k: runs.append(
        (run_cfg["start_date"][:10], run_cfg["end_date"][:10])))
    monkeypatch.setattr("assimilator.perturbate.perturbator", lambda *a, **k: None)

    cfg = _chain_cfg(tmp_path, "2026-01-01", "2026-03-15", cycle_max_days=31)
    assert adapter.run_openda_chain(cfg, _Model(), params={}) == t("2026-03-15")
    assert runs == [("2026-01-01", "2026-02-01"), ("2026-02-01", "2026-03-04"), ("2026-03-04", "2026-03-15")]

    # a chain that already ends after start_date continues from its own end
    chain = rc.chain_dir(cfg, cfg["openda_dir"])
    _openda_wrote_restart(cfg["openda_dir"])
    rc.commit_restart(cfg["openda_dir"], chain, "2026-03-14", ID)
    runs.clear()
    assert adapter.run_openda_chain(cfg, _Model(), params={}) == t("2026-03-15")
    assert runs == [("2026-03-14", "2026-03-15")]

    # already there: nothing to do; a chain ending before start_date is refused
    assert adapter.run_openda_chain(dict(cfg, end_date="2026-03-14"), _Model(), params={}) == t("2026-03-14")
    with pytest.raises(ValueError):
        adapter.run_openda_chain(dict(cfg, start_date="2026-03-20", end_date="2026-03-21"), _Model(), params={})


def test_forcing_is_continuous_when_a_boundary_falls_between_rows(tmp_path):
    """Two runs split at 06:30 (between hourly rows) perturb every row exactly as one run does."""
    assimilator.register_model(ToyModel)
    try:
        start = datetime(2024, 6, 1, tzinfo=UTC)
        pert = {"variables": {"U": {"phi": 0.8, "sigma": 0.5}, "V": {"phi": 0.8, "sigma": 0.5},
                              "GLOB": {"phi": 0.9, "sigma": 30.0}}}

        def members(base):
            write_inputs(str(base / "inputs"), start, days=2)
            cfg = {"lake": "x", "model": "toy", "n_members": 3, "model_inputs_path": str(base / "inputs"),
                   "ensemble_base": str(base / "run"), "perturbations": pert}
            get_model("toy").copy_model_inputs(cfg)
            return cfg

        def forcing(base):
            return pd.read_csv(base / "run" / "ensemble2" / "forcing.csv").set_index("time_days")["U"]

        one = members(tmp_path / "one")
        perturbator(dict(one, start_date="2024-06-01T00:00", end_date="2024-06-02T00:00"), get_model("toy"))
        whole = forcing(tmp_path / "one")

        two = members(tmp_path / "two")
        perturbator(dict(two, start_date="2024-06-01T00:00", end_date="2024-06-01T06:30"), get_model("toy"))
        first = forcing(tmp_path / "two")
        perturbator(dict(two, start_date="2024-06-01T06:30", end_date="2024-06-02T00:00"), get_model("toy"))
        second = forcing(tmp_path / "two")

        assert first.index.max() > second.index.min()             # both runs hold the 06:00-07:00 rows
        for part in (first, second):
            np.testing.assert_allclose(part.to_numpy(), whole.loc[part.index].to_numpy())
    finally:
        MODELS.pop("toy", None)

"""Run-config validation, perturbation input forms, and the model boundary."""
import ast
import os
import re

import pytest

from assimilator.config import build_config
from assimilator.perturbate import validate_perturbations

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src", "assimilator")

BASE = {"lake": "x", "algorithm": "EnKF", "n_members": 4, "start_date": "2025-01-01",
        "end_date": "2025-02-01", "model_inputs_path": "in", "ensemble_base": "run",
        "obs_file": "obs.csv", "perturbations_file": "p.json"}


def test_build_config_fills_defaults_and_absolute_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cfg = build_config({**BASE, "_comment": "dropped"})
    assert cfg["engine"] == "python" and cfg["model"] == "simstrat" and cfg["inflation"] == 1.0
    assert cfg["results_dir"] == "Results_EnKF" and cfg["par_file"] == "Settings_EnKF.par"
    assert cfg["ensemble_base"] == str(tmp_path / "run")
    assert "_comment" not in cfg
    assert build_config(cfg) == cfg                         # idempotent


@pytest.mark.parametrize("change, match", [
    ({"n_members": 1}, "n_members"),
    ({"algorithm": "DEnKF"}, "algorithm"),
    ({"start_date": "2025-03-01"}, "not before"),
    ({"lakes": {"x": {}}}, "flat"),
    ({"perturbations_file": None}, "perturbation"),
    ({"obs_file": None}, "missing"),
])
def test_build_config_rejects(change, match):
    with pytest.raises(ValueError, match=match):
        build_config({**BASE, **change})


def test_openda_dir_defaults_next_to_the_members():
    cfg = build_config({**BASE, "engine": "openda", "ensemble_base": "/data/run/x"})
    assert cfg["openda_dir"] == "/data/run/openda_simstrat_x_enkf"


def test_perturbations_accept_full_and_flat_forms():
    flat = {v: {"phi": 0.5, "sigma": 1.0} for v in ("U", "V", "GLOB")}
    assert validate_perturbations(flat)["variables"] == flat
    assert validate_perturbations({"variables": flat, "amplitude": {}})["variables"] == flat
    with pytest.raises(ValueError):
        validate_perturbations({"variables": {"U": {"phi": 0.5}}})


# Simstrat's files and names may appear only in its adapter and in the Simstrat-only OpenDA engine.
ALLOWED = {os.path.join("models", "simstrat.py"), "openda"}
FORBIDDEN = re.compile(r"\bsimstrat\b|\bt_out\b|forcing\.dat|settings\.par|\bz_out\b", re.I)
# The registry name of the default model is the one Simstrat word allowed elsewhere.
ALLOWED_TOKENS = {("config.py", "simstrat")}


def _code_tokens(path):
    """Identifiers, imports and non-docstring string constants of a module (comments excluded)."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    docstrings = {id(n.body[0].value) for n in ast.walk(tree)
                  if isinstance(n, (ast.Module, ast.FunctionDef, ast.ClassDef, ast.AsyncFunctionDef))
                  and n.body and isinstance(n.body[0], ast.Expr)
                  and isinstance(n.body[0].value, ast.Constant)}
    for n in ast.walk(tree):
        if isinstance(n, ast.Name):
            yield n.lineno, n.id
        elif isinstance(n, ast.Attribute):
            yield n.lineno, n.attr
        elif isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                yield n.lineno, f"{getattr(n, 'module', '') or ''}.{a.name}"
        elif isinstance(n, ast.Constant) and isinstance(n.value, str) and id(n) not in docstrings:
            yield n.lineno, n.value


def test_no_simstrat_knowledge_outside_its_adapter():
    leaks = []
    for root, _, files in os.walk(SRC):
        for f in files:
            if not f.endswith(".py"):
                continue
            rel = os.path.relpath(os.path.join(root, f), SRC)
            if rel in ALLOWED or rel.split(os.sep)[0] in ALLOWED or rel == os.path.join("models", "__init__.py"):
                continue
            for line, tok in _code_tokens(os.path.join(root, f)):
                if FORBIDDEN.search(tok) and (rel, tok) not in ALLOWED_TOKENS:
                    leaks.append(f"{rel}:{line}: {tok!r}")
    assert not leaks, "Simstrat-specific code outside models/simstrat.py:\n" + "\n".join(leaks)

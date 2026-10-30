# Data Assimilation for Lake Models

Correct lake-temperature simulations by blending in-situ measurements into a hydrodynamic model
(e.g. [Simstrat](https://github.com/Eawag-AppliedSystemAnalysis/Simstrat)). Built for the
[Alplakes](https://www.alplakes.eawag.ch) platform.

Give it a model setup folder and a CSV of measured temperatures. It runs an ensemble,
nudges it toward the observations at each step, and writes a corrected temperature profile
through time plus a skill report (RMSE / bias) scored against the data it assimilated.

It is a **library with a command line**: run it standalone on a folder, or embed it in another
repository (as [operational-simstrat](https://github.com/eawag-surface-waters-research/operational-simstrat)
does) and call `assimilator.run(cfg)`. The package reads no configuration of its own and
resolves no paths against this repository.

## What you can do

| You want | Config | Method |
|---|---|---|
| Ensemble Kalman Filter | `"algorithm": "EnKF"` | EnKF (native) |
| Particle Filter | `"algorithm": "PF"` | best-member selection (native) |
| Independent cross-check via [OpenDA](https://www.openda.org) | `"engine": "openda"` | EnKF / DEnKF / EnSR / PF |

The native engines are the in-house ones. The OpenDA engine runs the *same* ensemble and
observations through an established toolkit, so you can cross-validate results
(like-for-like is native-EnKF ↔ OpenDA-`EnKF`). OpenDA needs Linux/WSL + OpenDA 3.4.0 — set its
`bin/` path in `openda_bin`.

## Quick start

**1. Install** — Python 3 (`numpy pandas tqdm`, plus `matplotlib` for plots and `pytest` for the
tests) and [Docker](https://www.docker.com/) with the `eawag/simstrat:3.0.4` image (Simstrat runs
in Docker — no local build).

**2. Provide three inputs** for your lake (e.g. `upperlugano`):

- a **Simstrat setup folder** (`Settings.par`, `Forcing.dat`, bathymetry, grid, `z_out.dat` …)
  plus a dated warm-start snapshot `simulation-snapshot_<YYYYMMDD>.dat`. It is only read.
- an **observation CSV**, one reading per row:

  | `time` | `depth` | `value` |
  |---|---|---|
  | `2025-06-01T06:00:00+00:00` (UTC) | `0.5` (m, positive down) | `12.3` (°C) |

  Every distinct time is an analysis, so give it the series to assimilate, not the raw record —
  `python src/assimilate.py preprocess` makes one (seiche filter, one profile per day).
- the **forcing-perturbation calibration** (`{"variables": {"U": {"phi", "sigma"}, "V": .., "GLOB": ..}}`).

**3. Run**

```bash
python src/assimilate.py run --sim-dir inputs/upperlugano --obs obs.csv \
    --perturbations upperlugano.json --start 2025-01-01 --end 2025-12-31 --out run/upperlugano
# or, with everything in one config file (see examples/enkf.json):
python src/assimilate.py run examples/enkf.json
```

It copies the setup into an ensemble, perturbs the forcing, assimilates, and writes results.
Re-running re-uses finished steps.

**From Python** (e.g. a host repository with this one as a git submodule):

```python
sys.path.append("data-assimilation/src")
import assimilator
assimilator.run(cfg)    # the same keys as the JSON config
```

## Outputs

In `ensemble_base` (`--out`) for native, `openda_dir` for OpenDA:

- **`<lake>_<engine>_<label>.csv`** — posterior mean ± 1σ per time and depth.
- **`<lake>_<engine>_<label>.json`** — skill report (RMSE / bias) vs the assimilated obs.
- **`z_out.dat`** — the setup's output depths plus the observation depths (the members use it).
- **`logs/pipeline_<timestamp>.log`** — the full log (command line runs).

Plot a run with `python notebooks/visualize.py <config>.json --free-run T_out.dat --save`.

## What lives where

```
src/assimilate.py              command line: run / summarize / preprocess / fit
src/assimilator/
  pipeline.py                  run(cfg): inputs -> members -> perturb -> engine -> summary
  config.py                    the run config: DEFAULTS, build_config (validation)
  algorithms/                  native EnKF and PF (model-agnostic)
  perturbate.py                AR(1) forcing perturbation (model-agnostic)
  observations.py              observation preprocessing: seiche filter, depth grid, thinning
  fitting/                     offline fits: filter window, analysis hour, sigma_obs, AR(1)
  models/base.py               ForwardModel: the interface every model implements
  models/simstrat.py           the Simstrat adapter -- the only module that knows Simstrat's files
  openda/                      OpenDA engine (Simstrat only)
static/openda/                 OpenDA wrapper (Java-side XML + Python wrapper)
examples/enkf.json             a complete run config
tests/                         pytest suite, incl. a pure-Python toy model
```

Another forward model is one `ForwardModel` subclass registered with
`assimilator.register_model(...)`; the engines don't change (`tests/toy_model.py` is a minimal
example).

## Tuning a run

| Field | Controls |
|---|---|
| `n_members` | Ensemble size (more = better spread, slower) |
| `start_date`, `end_date` | Simulation window |
| `sigma_obs` | Observation error σ (°C), one value or `{"mixed", "stratified"}`; smaller = trust the data more |
| `inflation` | Variance inflation (native EnKF only); `1.0` = off |
| `sigma_scale` | Scales forcing-perturbation strength to widen spread (`1.0` = none) |
| `rng_seed` | Seed for reproducible runs (drives the forcing AND the EnKF obs perturbations) |
| `window_mode` | Native engines: `obs` (default — one window per observation time, like OpenDA) or `daily` (legacy) |

Every key: [docs/configuration.md](docs/configuration.md). Step by step: [docs/tutorial.md](docs/tutorial.md).

## Operational continuation

A run continued in slices (e.g. a daily pipeline extending `end_date` each day, with
`"reset": false`) produces **exactly the same outputs** as one continuous run over the same
period:

- All randomness is keyed by `(rng_seed, absolute time)` — the AR(1) forcing noise per forcing
  row and the EnKF observation perturbations per analysis instant — never by position in the
  run, so a draw for a given instant is identical however the period is chunked.
- The AR(1) forcing-perturbation chain persists its state per member/variable in
  `<ensemble_base>/perturbation_state.json` and resumes it on the next invocation.
- With `window_mode: "obs"` the assimilation instants are the observation times themselves,
  independent of how the period is split.

The chain cold-starts (first forcing row unperturbed) on a fresh run dir, on `"reset": true`,
or when the perturbation parameters (`rng_seed`, `sigma_scale`, `n_members`, phi/sigma) change.

## Tests

```bash
python -m pytest tests
```

The engine tests run on a pure-Python toy model (no Docker); a check fails if Simstrat-specific
code appears outside `models/simstrat.py` and `openda/`.

# Data Assimilation for Lake Models

Correct lake-temperature simulations by blending in-situ measurements into a hydrodynamic model
(e.g. [Simstrat](https://github.com/Eawag-AppliedSystemAnalysis/Simstrat)). Built for the
[Alplakes](https://www.alplakes.eawag.ch) platform.

Give it a lake's model setup and a CSV of measured temperatures. It runs an ensemble,
nudges it toward the observations at each step, and writes a corrected temperature profile
through time plus a skill report (RMSE / bias) scored against the data it assimilated.

## What you can do

Pick a method by choosing a run config — nothing else changes:

| You want | Run with | Method |
|---|---|---|
| Ensemble Kalman Filter | `args/run_enkf.json` | EnKF (native) |
| Particle Filter | `args/run_pf.json` | PF (native) |
| Independent cross-check via [OpenDA](https://www.openda.org) | `args/run_openda.json` | EnKF / DEnKF / EnSR / PF |

The native engines are the in-house ones. The OpenDA engine runs the *same* ensemble and
observations through an established toolkit, so you can cross-validate results
(like-for-like is native-EnKF ↔ OpenDA-`EnKF`). With OpenDA, set the `filter` field to pick
the variant. OpenDA needs WSL/Linux + OpenDA 3.4.0 — set its `bin/` path in `openda_bin`.

## Quick start

**1. Install** — Python 3 (`numpy pandas geopandas requests tqdm matplotlib`) and
[Docker](https://www.docker.com/) with the `eawag/simstrat:3.0.4` image (Simstrat runs in
Docker — no local build).

**2. Provide two inputs** for your lake (e.g. `upperlugano`):

- `inputs/<lake>/` — the Simstrat model setup (inputs) + a dated warm-start snapshot
  (`simulation-snapshot_<YYYYMMDD>.dat`, `Forcing.dat`, `Settings.par`, bathymetry, grid …).
- `observations/<lake>/temperature.csv` — one reading per row:

  | `time` | `depth` | `value` |
  |---|---|---|
  | `2025-06-01T11:55:00+00:00` (UTC) | `0.5` (m, positive down) | `12.3` (°C) |

You also need the forcing-perturbation calibration `perturbations/<lake>.json` (fit once,
offline, via `notebooks/perturbations_from_icon.py`).

**3. Run**

```bash
python src/main.py args/run_enkf.json
```

It copies the setup into an ensemble, perturbs the forcing, assimilates, and writes results.
Re-running re-uses finished steps. Add `--lake <name>` if a config defines several.

## Outputs

In the run folder (`run/<lake>/` for native, `run/openda_<model>_<lake>_<filter>/` for OpenDA):

- **`<lake>_<engine>_<label>.csv`** — posterior mean ± 1σ per time and depth.
- **`<lake>_<engine>_<label>.json`** — skill report (RMSE / bias) vs the assimilated obs.

Compare engines or plot a run with `python notebooks/visualize.py`.

## Tuning a run

Edit these top-level fields in the run config:

| Field | Controls |
|---|---|
| `n_members` | Ensemble size (more = better spread, slower) |
| `start_date`, `end_date` | Simulation window |
| `sigma_obs` | Observation error σ (°C); smaller = trust the data more |
| `inflation` | Variance inflation (native EnKF only); `1.0` = off |
| `sigma_scale` | Scales forcing-perturbation strength to widen spread (`1.0` = none) |
| `rng_seed` | Seed for reproducible runs |
| `filter` | OpenDA only: `EnKF` / `DEnKF` / `EnSR` / `PF` |

Useful CLI flags:

- `--no-progress` — disable the progress bar (auto-off when not a TTY; for server/headless runs).
  Per-step detail still goes to `logs/pipeline_<timestamp>.log` either way.
- `--max-workers N` — cap concurrent ensemble members (default: all at once).
- `--run-root <dir>` — write run output elsewhere. **On WSL, route output to native ext4** for a
  large speed-up — the `/mnt/c` mount is slow for the many small Docker IOs:
  ```bash
  python src/main.py args/run_openda.json --run-root ~/alplakes-data-assimilation_res/run
  ```
  (Equivalent: `ALPLAKES_RUN_ROOT` env or a `"run_root"` config key. Default: in-repo `run/`.)

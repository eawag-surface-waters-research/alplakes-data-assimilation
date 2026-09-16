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
offline, via `notebooks/generate_perturbation.py`).

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
| `rng_seed` | Seed for reproducible runs (drives the forcing AND the EnKF obs perturbations) |
| `window_mode` | Native engines: `obs` (default — one forecast+analysis window per observation time, like OpenDA) or `daily` (legacy fixed noon-to-noon windows) |
| `algorithm` | DA scheme — native: `EnKF` / `PF`; OpenDA: `EnKF` / `DEnKF` / `EnSR` / `PF` |

### Adaptive observation error (optional)

`sigma_obs` is a fixed number (or one per season). It can instead be estimated from the run's own
recent misses, per depth. Add:

```json
"adaptive_sigma": {"window_days": 14.0, "min_samples": 8, "sigma_min": 0.05}
```

At each analysis a depth has an innovation `d` (observation minus ensemble-mean forecast) and an
ensemble spread, and on average `d² = spread² + σ²`. So over the analyses in the last
`window_days`:

    σ(z)² = ⟨d²⟩ − ⟨spread²⟩

- A depth with fewer than `min_samples` analyses in the window keeps the configured `sigma_obs`
  for that month, so a run starts on the seasonal value and moves off it as the window fills. The
  fallback is per depth: some depths can be adaptive while others are not.
- `sigma_min` is a floor (°C): an ensemble wider than its own misses would otherwise give a
  negative estimate.
- σ is resolved before the analysis is recorded, so a σ never uses its own innovation.
- Bear in mind `⟨d²⟩` only constrains `R + HPHT`: if the ensemble is under-dispersed, the missing
  spread is charged to the observation error.

**Native engine.** Works in any run. The history (innovation and spread per depth) is written to
`enkf_adaptive_history.csv` in the run folder and replayed at start-up, so a run continued in
slices resolves the same σ as one continuous run.

**OpenDA.** Only on a restart chain (`openda_restart`), whose archived cycles are the history — a
config with `adaptive_sigma` and no chain is refused. Each cycle rebuilds σ from the cycles
archived in its window and declares one σ per depth, so the series are one per depth rather than
season-split, and the σ used is recorded in the cycle's sidecar. Cycles archived before this
feature existed carry no series formatter and cannot serve as history: start a new chain, or
expect the first `window_days` to run on the configured σ.

## Operational continuation

A run continued in slices (e.g. a daily pipeline extending `end_date` each day, with
`"reset": false`) produces **exactly the same outputs** as one continuous run over the same
period:

- All randomness is keyed by `(rng_seed, absolute time)` — the AR(1) forcing noise per forcing
  row and the EnKF observation perturbations per analysis instant — never by position in the
  run, so a draw for a given instant is identical however the period is chunked.
- The AR(1) forcing-perturbation chain persists its state per member/variable in
  `run/<lake>/perturbation_state.json` and resumes it on the next invocation (recent window
  boundaries are kept, so re-running a slice is idempotent).
- With `window_mode: "obs"` the assimilation instants are the observation times themselves,
  independent of how the period is split.

The chain cold-starts (first forcing row unperturbed) on a fresh run dir, on `"reset": true`,
or when the perturbation parameters (`rng_seed`, `sigma_scale`, `n_members`, phi/sigma) change.

### OpenDA: restart chain

OpenDA normally runs a whole window from the warmup. To keep an OpenDA run going as new
observations arrive, add to the OpenDA run config:

```json
"openda_restart": {"dir": "<chain folder>"}
```

Then run the same command whenever there is new data:

```bash
python src/assimilate.py args/run_openda.json --lake <name>
```

- It continues from where the chain stopped and runs one cycle per new observation time: each
  cycle perturbs the forcing, runs OpenDA up to that observation, and saves the ensemble.
- `start_date` is used only when the chain is empty (cold start from the warmup). `end_date` is
  optional: without it the chain runs up to the end of the forcing.
- No new observation (or no forcing for it): it logs `nothing to do` and exits.
- A failed cycle saves nothing and is retried on the next run.
- `--max-cycles N` runs at most N cycles (e.g. a long catch-up in parts).

What it writes:

| Where | What |
|---|---|
| `<chain folder>/restart_<day>.zip` + `.json` | the saved ensemble after each cycle; the json has the exact start/end and seed |
| `ensemble<i>/Results/T_out.dat` (run folder) | each member's trajectory over the whole chain |
| `cycles/<day>/` (run folder) | that cycle's OpenDA result file and logs |
| `shared/` (run folder) | the long inputs (`Qin/Tin/Sin`, `AED2_inflow`), kept once instead of per member |
| `<lake>_openda_<filter>.csv/.json` | summary and skill over the whole chain |

A cycle only reads one day of the inflow files and of the control forcing, so it does not copy the
whole record into every member: the inflow files are shared through `shared/` (refreshed when the
source changes, e.g. new data appended) and the template's `Forcing.dat` holds just the cycle's
rows. Same inputs, same results, about 2.5x faster per cycle. The perturbed member forcing is
unaffected — the perturbator still reads the complete control forcing in `inputs/<lake>/`.

The forcing perturbation continues exactly across cycles. OpenDA's own random draws get a new
seed per cycle (from `rng_seed` and the cycle time), so a chain is statistically, not bitwise,
equal to one continuous run. A chain belongs to one lake, filter and ensemble size; a
different set-up is refused. To start over, empty the chain folder.

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

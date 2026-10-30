# Run configuration

One run is described by one **flat** dict: as a JSON file for the command line, or passed directly by a host program.

```bash
python src/assimilate.py run examples/enkf.json            # from a file
python src/assimilate.py run --sim-dir inputs/upperlugano --obs obs.csv --perturbations p.json \
    --start 2025-01-01 --end 2025-12-31 --out run/upperlugano      # from flags
```

```python
import assimilator
result = assimilator.run(cfg)          # cfg: the same keys as the JSON
```

`assimilator.config.build_config` fills the defaults below, makes every path absolute, derives the engine's file names and validates the result before anything runs. **Relative paths resolve against the current working directory**, never against this repository. Keys starting with `_` are comments and are dropped. A `"lakes"` map (the former per-lake blocks) is rejected: one config is one run.

```json
{
    "lake": "upperlugano",
    "algorithm": "EnKF",
    "n_members": 20,
    "start_date": "2025-01-01",
    "end_date": "2025-12-31",
    "model_inputs_path": "inputs/upperlugano",
    "ensemble_base": "run/upperlugano",
    "obs_file": "observations/upperlugano/temperature_filtered_h06_1d.csv",
    "sigma_obs": {"mixed": 0.299, "stratified": 0.586},
    "perturbations_file": "upperlugano.json"
}
```

## Required

| Key | Description |
|---|---|
| `lake` | Run label: used in output file names and the Docker container name. |
| `algorithm` | Native: `"EnKF"` or `"PF"`. OpenDA: `"EnKF"`, `"DEnKF"`, `"EnSR"` or `"PF"`. The native `"PF"` is best-member selection, not a Bayesian particle filter ([Workblog](workblog.md) App. A.3). |
| `n_members` | Ensemble size, at least 2. The members are `ensemble1..N`. `ensemble0` is an unperturbed copy of the model inputs that the engines do not run. |
| `start_date`, `end_date` | Run window, ISO dates, read as UTC. `start_date` must be the date of the warm-start snapshot. |
| `model_inputs_path` | The model setup with its dated warm start. **Only read**: everything the run writes goes under `ensemble_base`. |
| `ensemble_base` | Output folder: the members `ensemble0..N`, the perturbation chain state, the mean trajectory, the summary and the logs. |
| `obs_file` | Observation CSV with `time`, `depth`, `value` columns: the series to assimilate (see `assimilator.observations` for how to make it). |
| `perturbations` or `perturbations_file` | The forcing-perturbation calibration, inline (`{"variables": {"U": {"phi", "sigma"}, "V": .., "GLOB": ..}}`; a flat `{"U": .., "V": .., "GLOB": ..}` also works) or as a JSON path. `perturbations` wins if both are set. |

## Engine and method

| Key | Default | Used by | Description |
|---|---|---|---|
| `engine` | `"python"` | all | `"python"` (native engine) or `"openda"`. |
| `model` | `"simstrat"` | all | Forward model, see `assimilator.models` and `register_model`. |
| `inflation` | `1.0` | native EnKF | Multiplies the ensemble anomalies before the update. `1.0` = off. |
| `window_mode` | `"obs"` | native | `"obs"`: one forecast and analysis per observation time, the same rule as OpenDA. `"daily"`: legacy fixed noon-to-noon windows. |
| `obs_selector` | `"window_end"` | native EnKF, `window_mode: "daily"` only | `"window_end"`: the reading nearest the end of the window. `"mean"`: the daily mean per depth. |
| `obs_operator` | `"linear"` | native EnKF | How an observation is mapped onto the model grid. `"linear"` interpolates between the two neighbouring cells; `"nearest"` snaps to the closest cell and exists only to reproduce older runs. |

## Run window and ensemble

| Key | Default | Description |
|---|---|---|
| `rng_seed` | `42` | Seeds the forcing noise and the EnKF observation perturbations. Draws are keyed by absolute time, so a run continued in slices reproduces a continuous run. |
| `reset` | `false` | Starts from scratch: clears the AR(1) chain state (`perturbation_state.json`) and, in the native engines, the live snapshots, member outputs, mean trajectory and diagnostics. Leave it off for an operational continuation. |

## Observation error

| Key | Default | Description |
|---|---|---|
| `sigma_obs` | `sigma_default` | Observation error in °C: a number, or `{"mixed": .., "stratified": ..}` with stratified = May to October. Fitted with `assimilator.fitting.sigma` (`python src/assimilate.py fit sigma`); it is only valid for the `obs_file` it was fitted on ([Workblog](workblog.md) §5.4 to §5.5). |
| `sigma_default` | `0.5` | Fallback when `sigma_obs` is not set. The log says when it is used. |
| `obs_target_hour` | `12` | OpenDA only. Hour (UTC) at which OpenDA picks the daily observation. When it is not set and the file already holds at most one reading per depth and day, each reading is assimilated at its own time instead. |
| `obs_min_days` | `1` | OpenDA only. Drops depths with fewer days of data than this. |

## Forcing perturbation

| Key | Default | Description |
|---|---|---|
| `sigma_scale` | `1.0` | Multiplies the AR(1) noise σ of every channel, to widen or narrow the spread. Changing it cold-starts the perturbation chain. |
| `ref_year` | the model's | Reference year of the forcing time axis (Simstrat: 1981). |

## Output and runtime

| Key | Default | Used by | Description |
|---|---|---|---|
| `results_dir` | `Results_<algorithm>` | native | Model output folder inside each member. |
| `par_file` | `Settings_<algorithm>.par` | native | Name of the per-member copy of `Settings.par` the engine runs. It is created once from `Settings.par` and not rewritten afterwards, so delete it from the members after changing `results_dir`. |
| `copy_skip` | `["Results", "ref"]` | all | Folders of the model inputs not copied into the members. |
| `max_workers` | all members at once | all | Cap on members run in parallel. |
| `progress` | `false` (CLI: on in a terminal) | all | Progress bar. |
| `docker_dir`, `repo_dir` | not set | all | When both are set, Docker mount paths under `repo_dir` are rewritten to `docker_dir`, for setups where Docker sees the files under another path. |

## OpenDA only

| Key | Default | Description |
|---|---|---|
| `openda_bin` | not set | The `bin/` folder of OpenDA 3.4.0. The engine builds the OpenDA environment from it, so nothing needs sourcing. Without it, an environment already set up in the shell is used. |
| `openda_native` | `"linux64_gnu"` | OpenDA native-library folder under `openda_bin`. |
| `openda_dir` | `openda_<model>_<lake>_<algorithm>` next to `ensemble_base` | OpenDA working folder for this run. |

## Model runtime

Filled in from the model's defaults. Override them only to run a different Simstrat build.

| Key | Default | Description |
|---|---|---|
| `simstrat_version` | `"3.0.4"` | Tag of the `eawag/simstrat` Docker image. |
| `simstrat_binary` | `/simstrat/build/simstrat` | Simstrat executable inside the container. |
| `simstrat_workdir` | `/simstrat/run` | Where `ensemble_base` is mounted inside the container. |

## Command line

`python src/assimilate.py <command> ...` — run `python src/assimilate.py <command> -h` for every option.

| Command | Does |
|---|---|
| `run [config.json] [--sim-dir --obs --perturbations --start --end --out ...]` | One run. Flags override the config file's keys; with flags only, `algorithm` defaults to `EnKF`, `n_members` to 20 and `lake` to the setup folder's name. `--reset`, `--force-copy` (re-copy the setup into the members), `--skip-oda` (OpenDA: prepare only), `--no-progress`, `--log-file`. |
| `summarize config.json` | Rebuild a finished run's summary CSV and skill JSON from its member outputs. |
| `preprocess raw.csv -o out.csv [--w-seiche H] [--hour H \| --hour-window LO-HI] [--depths ...]` | Raw sub-daily observations → the series to assimilate (`assimilator.observations`). |
| `fit filter\|hour\|sigma\|perturbation ...` | The offline fits (`assimilator.fitting`), from local files. |

## Using the package from another program

```python
import sys
sys.path.append("<path to this repository>/src")     # e.g. as a git submodule
import assimilator

result = assimilator.run(cfg)                              # same keys as above
assimilator.analysis_mean_state(cfg, "warm_start.dat")     # ensemble-mean analysis as a warm start

from assimilator.observations import preprocess            # raw -> assimilated series
from assimilator.fitting.sigma import fit_sigma             # and the other fits
```

A forward model other than Simstrat is added by subclassing `assimilator.models.base.ForwardModel` and calling `assimilator.register_model(MyModel)`; the engines do not change. `tests/toy_model.py` is a complete minimal example.

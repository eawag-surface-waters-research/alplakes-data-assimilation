# Run configuration

Every run is described by one JSON file in `args/` (`run_enkf.json`, `run_pf.json`, `run_openda.json`) and started with

```bash
python src/assimilate.py args/run_enkf.json --lake upperlugano
```

The file has two levels:

- **Top-level keys** set the engine, the method and the run window. They apply to every lake.
- **`lakes`** holds one block per lake: its observations, its observation error and its identity for the perturbation fit.

At run time the selected lake block is laid over the top level, so a key in the lake block wins over the same key at the top. The lake is chosen with `--lake`; it can be omitted only when the file has a single block. Keys starting with `_` (e.g. `_lakes_note`) are comments and are ignored.

```json
{
    "engine": "python",
    "algorithm": "EnKF",
    "results_dir": "Results_EnKF",
    "par_file": "Settings_EnKF.par",
    "inflation": 1.0,
    "n_members": 20,
    "start_date": "2025-01-01",
    "end_date": "2025-12-31",
    "lakes": {
        "upperlugano": {
            "obs_file": "observations/upperlugano/temperature_filtered_h06_1d.csv",
            "sigma_obs": {"mixed": 0.299, "stratified": 0.586},
            "reanalysis_lake": "lugano",
            "lake_bbox": [45.89, 8.85, 46.03, 9.13],
            "lake_key": "lugano"
        }
    }
}
```

Relative paths resolve against the repository root, except `ensemble_base` and `model_inputs_path`, which resolve against `src/`.

## Engine and method

| Key | Default | Used by | Description |
|---|---|---|---|
| `engine` | `"python"` | all | `"python"` (native engine) or `"openda"`. |
| `model` | `"simstrat"` | all | Forward model. The CLI flag `-m` overrides it. |
| `algorithm` | required (native), `"EnKF"` (OpenDA) | all | Native: `"EnKF"` or `"PF"`. OpenDA: `"EnKF"`, `"DEnKF"`, `"EnSR"` or `"PF"`. The native `"PF"` is best-member selection, not a Bayesian particle filter ([Workblog](workblog.md) App. A.3). |
| `inflation` | required (native EnKF) | native EnKF | Multiplies the ensemble anomalies before the update. `1.0` = off. Ignored by OpenDA. |
| `window_mode` | `"obs"` | native | `"obs"`: one forecast and analysis per observation time, the same rule as OpenDA. `"daily"`: legacy fixed noon-to-noon windows. |
| `obs_selector` | `"window_end"` | native EnKF, `window_mode: "daily"` only | `"window_end"`: the reading nearest the end of the window. `"mean"`: the daily mean per depth. |
| `obs_operator` | `"linear"` | native EnKF | How an observation is mapped onto the model grid. `"linear"` interpolates between the two neighbouring cells; `"nearest"` snaps to the closest cell and exists only to reproduce older runs. |

## Run window and ensemble

| Key | Default | Used by | Description |
|---|---|---|---|
| `n_members` | required | all | Ensemble size. The members are `ensemble1..N`. `ensemble0` is an unperturbed copy of the model inputs that the engines do not run right now for speed purposes. |
| `start_date`, `end_date` | required | all | Assimilation window, ISO dates, read as UTC. |
| `rng_seed` | `42` | all | Seeds the forcing noise and the EnKF observation perturbations. Draws are keyed by absolute time, so a run continued in slices reproduces a continuous run. |
| `reset` | off | all | Starts from scratch: clears the AR(1) perturbation state (`perturbation_state.json`) and, in the native engines, the live snapshots, member outputs, mean trajectory and diagnostics. Leave it off for an operational continuation. |

## Observations and observation error

| Key | Default | Where | Description |
|---|---|---|---|
| `obs_file` | `observations/<lake>/temperature.csv` | lake block | Observation CSV with `time`, `depth`, `value` columns. The CLI flag `--obs-file` overrides it. |
| `sigma_obs` | `sigma_default` | lake block | Observation error in °C: a number, or `{"mixed": .., "stratified": ..}` with stratified = May to October. Fitted with `notebooks/generate_sigma.py` and pasted in by hand; it is only valid for the `obs_file` it was fitted on ([Workblog](workblog.md) §5.4 to §5.5). |
| `sigma_default` | `0.5` | top level | Fallback for a lake block without `sigma_obs`. The log says when it is used. |
| `obs_target_hour` | `12` | OpenDA | Hour (UTC) at which OpenDA picks the daily observation. When it is not set and the file already holds at most one reading per depth and day, each reading is assimilated at its own time instead. |
| `obs_min_days` | `1` | OpenDA | Drops depths with fewer days of data than this. |

## Forcing perturbation

| Key | Default | Description |
|---|---|---|
| `perturbations_file` | `perturbations/<lake>.json` | AR(1) calibration for the lake, made with `notebooks/generate_perturbation.py` ([Workblog](workblog.md) §3 and App. B). The CLI flag `--perturbations-file` overrides it. |
| `perturbations_dir` | `perturbations/` | Folder searched for `<lake>.json` when `perturbations_file` is not set. |
| `sigma_scale` | `1.0` | Multiplies the AR(1) noise σ of every channel, to widen or narrow the spread. Changing it cold-starts the perturbation chain. |
| `ref_year` | from `static/general.json` | Simstrat reference year of the forcing time axis. |

## Output and runtime

| Key | Default | Used by | Description |
|---|---|---|---|
| `results_dir` | required (native), `"Results"` (OpenDA) | all | Simstrat output folder inside each member. |
| `par_file` | required | native | Name of the per-member copy of `Settings.par` the engine runs. It is created once from `Settings.par` and not rewritten afterwards, so delete it from the members after changing `results_dir`. |
| `run_root` | `<repo>/run` | all | Base folder for all run output. Also settable with `$ALPLAKES_RUN_ROOT` or `--run-root` (CLI wins). On WSL, a native ext4 path is much faster than `/mnt/c`. |
| `ensemble_base` | `<run_root>/<lake>` | all | Folder holding `ensemble0..N` for this lake. |
| `model_inputs_path` | `inputs/<lake>` | all | Simstrat setup copied into every member. |
| `copy_skip` | `["Results", "ref"]` | all | Folders of the model inputs not copied into the members. |
| `max_workers` | all members at once | all | Cap on members run in parallel. The CLI flag `--max-workers` overrides it. |
| `progress` | on when run in a terminal | all | Progress bar. The CLI flag `--no-progress` turns it off. |
| `docker_dir`, `repo_dir` | not set | all | When both are set, Docker mount paths under `repo_dir` are rewritten to `docker_dir`, for setups where Docker sees the repository under another path. |

## OpenDA only

| Key | Default | Description |
|---|---|---|
| `openda_bin` | not set | The `bin/` folder of OpenDA 3.4.0. The engine builds the OpenDA environment from it, so nothing needs sourcing. Without it, an environment already set up in the shell is used. |
| `openda_native` | `"linux64_gnu"` | OpenDA native-library folder under `openda_bin`. |
| `openda_dir` | `<run_root>/openda_<model>_<lake>_<algorithm>` | OpenDA working folder for this run. |

## Lake identity (perturbation fit)

These keys sit in the lake block. Only `notebooks/generate_perturbation.py` reads them, to cut the lake area out of the ICON reanalysis.

| Key | Default | Description |
|---|---|---|
| `lake_bbox` | required | Bounding box `[lat_min, lon_min, lat_max, lon_max]`. |
| `lake_key` | not set | Lake key in `static/lakes.geojson`, used for the lake outline. |
| `lake_contour` | not set | Path to a local GeoJSON outline, as an alternative to `lake_key`. |
| `reanalysis_lake` | the lake name | Lake name used by the reanalysis. |
| `contours_geojson` | `static/lakes.geojson` | GeoJSON searched for `lake_key`. |
| `fit_start`, `fit_end` | the previous full calendar year | Period the perturbation is fitted on. |

## Model runtime

Filled in from the model's defaults. Override them only to run a different Simstrat build.

| Key | Default | Description |
|---|---|---|
| `simstrat_version` | `"3.0.4"` | Tag of the `eawag/simstrat` Docker image. |
| `simstrat_binary` | `/simstrat/build/simstrat` | Simstrat executable inside the container. |
| `simstrat_workdir` | `/simstrat/run` | Where `ensemble_base` is mounted inside the container. |

## Command-line options

`python src/assimilate.py <config> [options]`

| Option | Effect |
|---|---|
| `--lake <name>` | Lake block to run. Required when the config has more than one. |
| `-m`, `--model <name>` | Forward model, overriding `model`. |
| `--obs-file <path>` | Overrides `obs_file`. |
| `--perturbations-file <path>` | Overrides `perturbations_file`. |
| `--max-workers <n>` | Overrides `max_workers`. |
| `--run-root <dir>` | Overrides `run_root` and `$ALPLAKES_RUN_ROOT`. |
| `--no-progress` | Turns the progress bar off. |
| `--force-copy` | Copies the model inputs into the members again, even if they exist. |
| `--skip-oda` | OpenDA only: prepares the run but does not start OpenDA. |

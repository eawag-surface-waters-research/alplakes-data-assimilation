# Tutorial

One assimilation run for one lake: what to install, which files to prepare, how to run, and what comes out. The examples use `upperlugano`; replace it with your lake.

The repository holds the code and an example configuration, but **no model setups and no observations**. Each user prepares those for their own lake. This page describes what the files must contain and how the operational ones were made; how you produce them is up to you. Operationally, [operational-simstrat](https://github.com/eawag-surface-waters-research/operational-simstrat) prepares all of them and calls the same `assimilator.run(cfg)` this page runs from the command line.

## 1. Install

- **Python** with `numpy`, `pandas` and `tqdm` (`matplotlib` for the plots, `pytest` for the tests).
- **[Docker](https://www.docker.com/)** with the `eawag/simstrat:3.0.4` image. Simstrat runs inside Docker, so nothing needs compiling.
- **For the OpenDA engine only:** Linux or WSL, and [OpenDA](https://www.openda.org) 3.4.0.

## 2. What a run needs

```
<setup folder>/                     Simstrat setup + warm-start snapshot      (read only)
<free run T_out.dat>                free run (only to fit sigma_obs and the hour)
<observations>.csv                  the observations to assimilate
<perturbations>.json                forcing-perturbation calibration
<config>.json                       one flat run configuration (optional: flags work too)
```

Where these files live is up to you; the paths go into the run configuration or on the command line. The examples below use `inputs/<lake>/`, `observations/<lake>/` and `run/<lake>/` inside the repository, which git ignores.

### 2.1 Simstrat setup: `inputs/<lake>/`

A complete, calibrated Simstrat setup that runs on its own: `Settings.par`, `Forcing.dat`, bathymetry, grid, initial conditions, absorption, inflow and outflow files, and `z_out.dat`. Setups for the Alplakes lakes can be downloaded from [alplakes.eawag.ch/downloads](https://www.alplakes.eawag.ch/downloads). On top of a standard setup:

- **`Settings.par`** must have `"Continue from last snapshot"` enabled, so each assimilation window restarts from the state the previous one left, and `"Reference year"` 1981 (`SIMSTRAT_REF_YEAR` in `models/simstrat.py`). The engines set the start and end dates themselves, in their own copies.
- **`Forcing.dat`** is the standard Simstrat file: one header line, then time (days since 1 January of the reference year), u and v wind (m/s), air temperature (°C), solar radiation (W/m²), vapour pressure (mbar), cloud cover (–) and rain (m/h). It must cover the whole run. Each member gets its own copy with u, v and solar radiation perturbed.
- **`z_out.dat`** lists the output depths. The run adds the observation depths to the members' copies, so you don't have to. **The setup folder itself is never modified**; the extended grid is written to `<ensemble_base>/z_out.dat` in case you want it for your own runs.

### 2.2 Warm-start snapshot and free run

**Warm-start snapshot**, `inputs/<lake>/simulation-snapshot_<YYYYMMDD>.dat`: the Simstrat snapshot of the model state at `start_date`. Every member starts from it. It comes from a spin-up: run Simstrat from its initial conditions up to `start_date` and keep the snapshot it writes at the end.

**The date must match `start_date` exactly, and nothing checks it.** A snapshot holds only the model state, not a time: the run takes its time from `start_date`, so a snapshot from another date is silently used as if it were the state at `start_date`. With several snapshots in the folder, the one with the latest date in its name is used, so keep only the one for the run.

**Free run** (e.g. `inputs/<lake>/ref/T_out.dat`): the same setup run over the assimilation period from the warm-start snapshot, with the control forcing, without perturbation and without assimilation. It is the reference that `sigma_obs` and the assimilation hour are fitted against (§2.3, §2.4), and the baseline the results are compared with. The run itself does not need it.

### 2.3 Observations: `observations/<lake>/<file>.csv`

A CSV in long format, one row per `(time, depth)`. Leave out a missing reading rather than writing a row for it: an empty `value` is skipped, but a literal `NaN` is read by the OpenDA engine as a number. Several readings at the same depth within the same hour are averaged.

```
time,depth,value
2025-01-01T06:00:00+00:00,0.5,8.03
2025-01-01T06:00:00+00:00,1.0,7.96
```

| Column | Meaning |
|---|---|
| `time` | ISO 8601 with time zone, UTC. |
| `depth` | Metres below the surface, positive. |
| `value` | Temperature, °C. |

Other columns are ignored. In particular, a quality flag is not applied: **remove rejected readings before the run.**

**The file is the analysis schedule.** Every distinct time in it becomes an analysis, so it must contain only the readings to assimilate. A raw record with a sample every 10 minutes would assimilate every hour. Readings are grouped into hourly bins centred on the full hour, and depths below the model's lake bed are dropped.

The operational files are made from the full sub-daily record in three steps, implemented in `assimilator.observations` and available as

```bash
python src/assimilate.py preprocess raw.csv -o observations/upperlugano/temperature_filtered_h06_1d.csv \
    --w-seiche 27.3 --hour 6                    # [--depths 0.5,1,3,...] [--hour-window 0-6]
```

Rows are only ever selected or averaged, never invented.

1. **Filter the internal-wave signal** (`fit filter --raw raw.csv --length-km L --mean-depth D` derives the window). Below 4 m, most hour-to-hour variance is the thermocline being moved up and down by basin-scale seiches, which a 1D model cannot reproduce. Each depth below 4 m is replaced by a trailing (causal) mean over one fundamental internal-seiche period of the lake, from Merian's two-layer formula `T = 2L / sqrt(g'·h_eff)` at peak stratification: 11 h on Ägeri, 27 h on Upper Lugano, 97 h on Geneva. A value is dropped when its window is less than a quarter full or holds only samples from before a gap. The top 4 m pass through untouched. Same rows in and out.
2. **Choose the assimilation hour.** Using the free run, split the surface error at each hour of the day into its mean over days (bias the model repeats every day) and its day-to-day scatter (error the analysis can correct). Among hours the lake samples on at least 30 days, keep those whose bias is at most 0.2 times their scatter, then take the one with the largest scatter. Candidates are limited to 00 to 06 UTC by default (`--window`, which may wrap, e.g. `23-6`), so the analysis lands before the day it forecasts and every lake is corrected at a comparable point of its daily cycle. `python src/assimilate.py fit hour --raw raw.csv --free-run T_out.dat` does this.
3. **Thin to one profile per day.** Each day, take the instant nearest the chosen hour and keep every depth at that instant. Drop the day if that instant is more than 1.5 h from the hour. For dense sensor chains, keep only the depths nearest a target grid (within 0.5 m): about 1 m spacing through the thermocline, opening to 5 m below it.

The reasons behind each step, and their effect on the results, are in the [Workblog](workblog.md), §4 and §5.

### 2.4 Observation error: `sigma_obs`

The observation error σ in °C, one value or one per season (`{"mixed": .., "stratified": ..}`, stratified = May to October). It stands for what the 1D model cannot represent, not for instrument error ([Workblog](workblog.md) §5.4 and §5.5). To fit it:

1. take the free run minus the observations at the assimilated times;
2. per depth and season, take the standard deviation about that season's own mean;
3. take the median over depths.

`python src/assimilate.py fit sigma --free-run T_out.dat --obs <the assimilated file>` does this and prints the pair to put into the run configuration. **σ belongs to the observation file it was fitted on:** refit it whenever the filtering, the hour or the depth grid changes. Without it the run uses `sigma_default` (0.5 °C).

### 2.5 Forcing-perturbation calibration

The AR(1) noise added to the forcing ([Workblog](workblog.md) §3 and App. B): `{"variables": {"U": {"phi", "sigma"}, "V": .., "GLOB": ..}}`, as a JSON file or inline in the configuration. The calibrations of the seven study lakes are kept by operational-simstrat (`static/assimilation/<lake>.json`, key `perturbations`); `examples/enkf.json` carries upperlugano's.

The math is in `assimilator.fitting.perturbation`. The amplitude needs the ICON reanalysis lake mean (Eawag network), so that fit is run from operational-simstrat (`src/assimilation_tools/fit_perturbation.py`). The persistence ceiling needs only the control forcing:

```bash
python src/assimilate.py fit perturbation --sim-dir inputs/upperlugano --existing upperlugano.json -o upperlugano.json
```

The run stops if no calibration is given.

### 2.6 Run configuration

One flat JSON per run; `examples/enkf.json` is a complete one:

```json
{
    "lake": "upperlugano", "algorithm": "EnKF", "n_members": 20,
    "start_date": "2025-01-01", "end_date": "2025-12-31",
    "model_inputs_path": "inputs/upperlugano",
    "ensemble_base": "run/upperlugano",
    "obs_file": "observations/upperlugano/temperature_filtered_h06_1d.csv",
    "sigma_obs": {"mixed": 0.299, "stratified": 0.586},
    "perturbations_file": "upperlugano.json"
}
```

`start_date` must be the date of the warm-start snapshot (§2.2). Relative paths resolve against the directory you run from. Every key is described in [Configuration](configuration.md).

## 3. Run

```bash
python src/assimilate.py run examples/enkf.json
# or without a config file:
python src/assimilate.py run --sim-dir inputs/upperlugano --obs obs.csv --perturbations upperlugano.json \
    --start 2025-01-01 --end 2025-12-31 --out run/upperlugano
```

The log goes through five steps:

1. check that the Simstrat setup and the snapshot are there;
2. copy the setup into the members (skipped if already done; `--force-copy` repeats it);
3. write a perturbed `Forcing.dat` into each member;
4. run the members window by window, assimilating at each observation time;
5. write the results and a skill report.

The console shows a progress bar; the full log is in `<ensemble_base>/logs/pipeline_<timestamp>.log`. `"algorithm": "PF"` (or `--algorithm PF`) runs best-member selection instead. `"engine": "openda"` runs the same experiment through the OpenDA engine; for it alone, set `openda_bin` to the `bin/` folder of your OpenDA installation.

**Continuing a run.** With `"reset": false`, running again with a later `end_date` carries on where the last run stopped. All random draws are tied to absolute time, so the result is the same as one continuous run. `"reset": true` starts from scratch.

## 4. Outputs

Native engine, in `ensemble_base`:

| File | Content |
|---|---|
| `<lake>_python_<algorithm>.csv` | Ensemble mean and standard deviation per time and depth. |
| `<lake>_python_<algorithm>.json` | RMSE and bias (model minus observation) against the assimilated observations, overall and per depth. |
| `T_out_<algorithm>_mean.dat` | Ensemble-mean trajectory in Simstrat's `T_out` format. |
| `enkf_diagnostics.csv`, `enkf_innov_by_depth.csv`, `enkf_kgain_by_depth.csv` | Native EnKF only. Innovations, spread, NIS and gain per analysis, tracked to analyse how the filter behaves; the run itself does not use them. |
| `ensemble<i>/<results_dir>/` | Each member's Simstrat output. |

OpenDA writes the same CSV and JSON, named `<lake>_openda_<algorithm>.*`, in `openda_dir` (default: `openda_simstrat_<lake>_<algorithm>/` next to `ensemble_base`).

The skill report scores the analysis against the observations it assimilated: it measures fit, not independent skill. For an honest check, compare against data the run did not see. `python notebooks/visualize.py <config>.json --free-run T_out.dat --save` plots a run and compares it with the free run.

## Support

For help with preparing the inputs or running the module, contact **<Thomas Toschini, thomas.toschini@eawag.ch>**.
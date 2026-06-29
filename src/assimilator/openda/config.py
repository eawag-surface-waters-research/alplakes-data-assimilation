"""Single source of truth for the OpenDA Kalman-filter configuration.

The per-filter OpenDA setup (EnKF / DEnKF / EnSR) used to be a 4-deep
chain of near-identical hand-maintained files per filter:

    <type>.oda -> parallel_<type>.xml -> simstratStochModel<Type>.xml
               -> simstratModel<Type>.xml -> simstratWrapperEnKF.xml

A diff showed the model/stochModel chain is identical across filters except the
instance (work) directory, and the only genuinely filter-specific things are the
algorithm class, its config root element/schema, and the result-writer filename —
all captured in the FILTERS spec below.

On top of that the whole setup was hardcoded to one lake: the assimilated
observation depths, the time window, and the ensemble size were baked into the
model config, the two timeSeriesFormatter files, the algorithm configs and the
wrapper.  This module renders all of it, each run, from three run inputs:

    obs_depths   - list of observation depths (auto-detected from the obs CSV)
    (start, end) - the run's date window
    n_members    - ensemble size

Generated files (all suffixed .gen so they read as build artifacts):
    run.oda                                       (openda root)
    parallel.gen.xml                              (openda root)
    stochModel/simstratStochModel.gen.xml
    stochModel/simstratModel.gen.xml
    stochObserver/timeSeriesFormatter.gen.xml     (observations + window + std)
    stochModel/template/timeSeriesFormatter.gen.xml   (model predictions)
    stochModel/template/time_control.yaml         (start/end window in Simstrat days)
    stochModel/template/obs_depths.json           (depth list the wrapper reads)
    algorithms/<filter>.gen.xml                    (full algorithm config from FILTERS)

The template's initial state file temperature_state.txt is seeded separately, by
the openda adapter (from the warmup snapshot's full-grid T profile).

Kept as-is: the wrappers and the Simstrat model template (stochModel/template/).
"""

import os
import json
import logging
from datetime import date, datetime

from assimilator.models.simstrat import SIMSTRAT_REF_YEAR

logger = logging.getLogger(__name__)

# Per-filter algorithm spec: OpenDA class, plus the config-file root element and schema (the
# only things that differ between filters' algorithm configs). root/schema are OpenDA's canonical
# names (verified against the OpenDA 3.4.0 examples): the enkf.xsd family (EnKF + DEnKF) uses root
# "EnkfConfig", ensr.xsd uses "EnsrConfig". OpenDA dispatches by "class" and does not enforce the
# root name, but we keep it canonical so EnKF and DEnKF stay consistent. Add an "extra" string for
# a variant body (e.g. localization: extra="\n\t<localization>Hamill</localization>\n\t<distance>10</distance>").
FILTERS = {
    "EnKF":  {"class": "org.openda.algorithms.kalmanFilter.EnKF",  "root": "EnkfConfig", "schema": "enkf.xsd"},
    "DEnKF": {"class": "org.openda.algorithms.kalmanFilter.DEnKF", "root": "EnkfConfig", "schema": "enkf.xsd"},
    "EnSR":  {"class": "org.openda.algorithms.kalmanFilter.EnSR",  "root": "EnsrConfig", "schema": "ensr.xsd"},
    # SIR particle filter (residual resampling). samplingMethod is optional + fixed in
    # particleFilter.xsd, so it's omitted; the shared body below is a valid ParticleFilterConfig.
    # Spread/diversity comes from the wrapper's per-instance Forcing_<i>.dat injection (same as the
    # Kalman filters above), so stochForcing stays false — there is no OpenDA noiseModel to drive.
    # needs_restart: the PF clones whole particles during resampling via
    # saveInternalState/restoreInternalState, which requires the model's restart files to be
    # declared (<restartInfo>) at both the model and stoch-model layer.  The Kalman filters work
    # purely through getState/axpyOnState and never checkpoint the model, so they omit it.  Crucially
    # the stoch-layer dirPrefix uses the INSTANCE_DIR/ token (see _STOCH_RESTART_INFO) so each
    # particle's modelState.zip lands in its own work dir; otherwise BBStochModelInstance roots every
    # member's saved state at the shared stoch configRootDir under one timestamp-named dir, they
    # collide on a single modelState.zip, and resampling's release crashes.
    "PF":    {"class": "org.openda.algorithms.kalmanFilter.ParticleFilter", "root": "ParticleFilterConfig", "schema": "particleFilter.xsd", "needs_restart": True},
}

DEFAULT_OBS_STD = 0.5  # per-depth observation standard deviation (°C)


def depth_label(d):
    """'1m', '0.5m', '40m' — matches the T_<label>_real.csv / T_<label>.csv naming."""
    return f"{d:g}m"


def _simstrat_day(d):
    """Integer Simstrat day (days since 1 Jan SIMSTRAT_REF_YEAR) for a date/datetime/ISO str."""
    if isinstance(d, str):
        d = datetime.fromisoformat(d[:10])
    if isinstance(d, datetime):
        d = d.date()
    return (d - date(SIMSTRAT_REF_YEAR, 1, 1)).days


def results_filename(filter_type):
    return f"{filter_type.lower()}_results.py"


# --- templates -------------------------------------------------------------

_ODA = """<?xml version="1.0" encoding="UTF-8"?>
<openDaApplication xmlns="http://www.openda.org" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xsi:schemaLocation="http://www.openda.org http://schemas.openda.org/openDaApplication.xsd">

\t<!-- GENERATED by src/assimilator/openda/config.py (filter={filter}). Do not edit by hand;
\t     change the FILTERS spec / templates in config.py instead. -->

\t<stochObserver className="org.openda.observers.TimeSeriesFormatterStochObserver">
\t\t<workingDirectory>./stochObserver</workingDirectory>
\t\t<configFile>timeSeriesFormatter.gen.xml</configFile>
\t</stochObserver>

\t<stochModelFactory className="org.openda.models.threadModel.ThreadStochModelFactory">
\t\t<workingDirectory>.</workingDirectory>
\t\t<configFile>parallel.gen.xml</configFile>
\t</stochModelFactory>

\t<algorithm className="{algo_class}">
\t\t<workingDirectory>./algorithms</workingDirectory>
\t\t<configString>{algo_config}</configString>
\t</algorithm>

\t<resultWriters>
\t\t<resultWriter className="org.openda.resultwriters.PythonResultWriter">
\t\t\t<workingDirectory>{results_dir}</workingDirectory>
\t\t\t<configFile>{results_file}</configFile>
\t\t\t<selection>
\t\t\t\t<!-- Verbose: stores x_f_central and x_a_central (full-grid profiles) -->
\t\t\t\t<resultItem outputLevel="Verbose" maxSize="10000000" />
\t\t\t</selection>
\t\t</resultWriter>
\t</resultWriters>

</openDaApplication>
"""

_PARALLEL = """<threadConfigstoch>
    <!-- GENERATED by src/assimilator/openda/config.py. maxThreads = n_members + 1 (main + ensemble). -->
    <maxThreads>{max_threads}</maxThreads>
    <stochModelFactory className="org.openda.blackbox.wrapper.BBStochModelFactory">
        <workingDirectory>./stochModel</workingDirectory>
        <configFile>simstratStochModel.gen.xml</configFile>
    </stochModelFactory>
</threadConfigstoch>
"""

_STOCHMODEL = """<?xml version="1.0" encoding="UTF-8"?>
<blackBoxStochModel xmlns="http://www.openda.org"
\txmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
\txsi:schemaLocation="http://www.openda.org http://schemas.openda.org/blackBoxStochModelConfig.xsd">

\t<!-- GENERATED by src/assimilator/openda/config.py. -->

\t<modelConfig>
\t\t<file>./simstratModel.gen.xml</file>
\t</modelConfig>

\t<vectorSpecification>
\t\t<!-- Full-grid T profile — updated by the analysis step -->
\t\t<state>
\t\t\t<vector id="temperature.state" />
\t\t</state>
\t\t<!-- Predictors: model output at observation depths, matched against stochObserver -->
\t\t<predictor>
{predictors}
\t\t</predictor>
\t</vectorSpecification>
{stoch_restart_info}
</blackBoxStochModel>
"""

# Stoch-layer restart declaration (PF only).  The model-layer restartInfo (below) checkpoints the
# Simstrat state per instance; this tells the BBStochModel wrapper where to bundle that state into
# its per-particle modelState.zip for resampling.  The INSTANCE_DIR/ token is rewritten by
# BBStochModelFactory.getInstance to the member's own getModelRunDir() (work_pf/workN), so each
# particle gets its OWN savedStochModelState_ dir.  Without it the prefix is rooted at the shared
# stoch configRootDir and every member collides on one modelState.zip (release double-deletes -> crash).
_STOCH_RESTART_INFO = """
\t<restartInfo dirPrefix="INSTANCE_DIR/savedStochModelState_" />
"""

_MODEL = """<?xml version="1.0" encoding="UTF-8"?>
<blackBoxModelConfig xmlns="http://www.openda.org"
\txmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
\txsi:schemaLocation="http://www.openda.org http://schemas.openda.org/blackBoxModelConfig.xsd">

\t<!-- GENERATED by src/assimilator/openda/config.py (filter={filter}). -->

\t<wrapperConfig>
\t\t<file>simstratWrapperEnKF.xml</file>
\t</wrapperConfig>

\t<aliasValues>
\t\t<alias key="templateDir"  value="template" />
\t\t<alias key="instanceDir"  value="{instance_dir}" />
\t\t<alias key="binDir"       value="bin" />
\t\t<alias key="configFile"   value="Settings.par" />
\t\t<alias key="stateFile"    value="temperature_state.txt" />
\t</aliasValues>

\t<!-- Time in Simstrat days (days since Jan 1 of reference year {ref_year}).
\t     MJD offset: Simstrat_day = MJD - 44605 -->
\t<timeInfoExchangeItems start="start_time" timeStep="time_step" end="end_time"/>

\t<exchangeItems>
\t\t<vector id="start_time"        ioObjectId="time_config" elementId="time_control@1" />
\t\t<vector id="time_step"         ioObjectId="time_config" elementId="time_control@2" />
\t\t<vector id="end_time"          ioObjectId="time_config" elementId="time_control@3" />
\t\t<vector id="temperature.state" ioObjectId="state_temperature" elementId="temperature_state" />
{outputs}
\t</exchangeItems>
{restart_info}
\t<doCleanUp>false</doCleanUp>

</blackBoxModelConfig>
"""

# Restart declaration (PF only — see FILTERS["PF"]["needs_restart"]).  Both files travel together
# on every save/restore:
#   - Results/simulation-snapshot.dat is Simstrat's full binary state (T, U, V, S, k, eps on all cells)
#     and the true continuity between windows (the wrapper runs "Continue from last snapshot").
#   - temperature_state.txt is OpenDA's T-only view, which the wrapper re-injects into the snapshot at
#     the start of every step.  Cloning the snapshot alone would leave a killed particle's stale
#     temperature_state.txt in place, and that re-injection would clobber the restored snapshot on the
#     next run — silently undoing the resampling.  Listing both keeps them in lockstep.
_RESTART_INFO = """
\t<restartInfo dirPrefix="./savedModelState_">
\t\t<modelStateFile>Results/simulation-snapshot.dat</modelStateFile>
\t\t<modelStateFile>temperature_state.txt</modelStateFile>
\t</restartInfo>
"""

# Observation formatter (stochObserver/): reads the real-obs CSVs over the window.
_OBS_FORMATTER = """<?xml version="1.0" encoding="UTF-8"?>
<timeSeriesFormatterConfig xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <!-- GENERATED by src/assimilator/openda/config.py. time column = Simstrat days (since 1 Jan {ref_year}).
       Range {start_day},{end_day} covers the configured simulation window. -->
  <formatter class="org.openda.exchange.timeseries.DelimitedTextTimeSeriesFormatter">
    <dateTimeSelector type="fixed" timeFormat="mjd">{start_day},{end_day}</dateTimeSelector>
    <valueSelector>1</valueSelector>
    <delimiter>,</delimiter>
    <commentMarker>#</commentMarker>
    <skipLines>1</skipLines>
  </formatter>
{rows}
</timeSeriesFormatterConfig>
"""

# Model-output formatter (template/): reads the predictor CSVs the wrapper writes.
_MODEL_FORMATTER = """<?xml version="1.0" encoding="UTF-8"?>
<timeSeriesFormatterConfig xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <!-- GENERATED by src/assimilator/openda/config.py. time column = Simstrat days (since 1 Jan {ref_year}). -->
  <formatter class="org.openda.exchange.timeseries.DelimitedTextTimeSeriesFormatter">
    <dateTimeSelector>0</dateTimeSelector>
    <valueSelector>1</valueSelector>
    <delimiter>,</delimiter>
    <commentMarker>#</commentMarker>
    <skipLines>1</skipLines>
  </formatter>
{rows}
</timeSeriesFormatterConfig>
"""


# Algorithm config (algorithms/): root element + schema vary per filter; the body
# (analysis at obs times, deterministic main + ensemble models, ensemble size) is shared.
_ALGORITHM = """<?xml version="1.0" encoding="UTF-8"?>
<{root} xmlns="http://www.openda.org"
\txmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
\txsi:schemaLocation="http://www.openda.org http://schemas.openda.org/algorithm/{schema}">

\t<!-- GENERATED by src/assimilator/openda/config.py (filter={filter}). -->

\t<analysisTimes type="fromObservationTimes" />
\t<mainModel stochParameter="false" stochForcing="false" stochInit="false" />
\t<ensembleSize>{n_members}</ensembleSize>
\t<ensembleModel stochParameter="false" stochForcing="false" stochInit="false" />{extra}
</{root}>
"""


# Model time control (template/): OpenDA reads/writes the #oda:time_control line.
_TIME_CONTROL = """%YAML 1.1
---
# GENERATED by src/assimilator/openda/config.py. Time in Simstrat days (days since 1 Jan {ref_year}).
# Format: [start_day, dt_day, end_day]; dt is a placeholder OpenDA overwrites each step.
time: [{start_day:.1f}, 1.0, {end_day:.1f}]  #oda:time_control
"""


def _predictor_lines(depths):
    return "\n".join(f'\t\t\t<vector id="T_{depth_label(d)}" />' for d in depths)


def _output_lines(depths):
    lines = []
    for d in depths:
        vid = f'"T_{depth_label(d)}"'
        lines.append(f'\t\t<vector id={vid:<10} ioObjectId="output" elementId="model.T_{depth_label(d)}" />')
    return "\n".join(lines)


def _obs_formatter_rows(depths, obs_std):
    return "\n".join(
        f'  <timeSeries id="T_{depth_label(d)}" status="use" standardDeviation="{obs_std}">'
        f'T_{depth_label(d)}_real.csv</timeSeries>'
        for d in depths
    )


def _model_formatter_rows(depths):
    return "\n".join(
        f'  <timeSeries id="model.T_{depth_label(d)}">T_{depth_label(d)}.csv</timeSeries>'
        for d in depths
    )


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def render(openda_dir, filter_type, n_members, obs_depths, start_date, end_date,
           obs_std=DEFAULT_OBS_STD, results_subdir="Results", max_threads=None):
    """Render the full lake-specific OpenDA config for `filter_type` into `openda_dir`.

    `obs_depths` is the list of assimilated depths (e.g. auto-detected from the obs
    CSV); it drives the model exchangeItems, the stochModel predictors, both
    timeSeriesFormatter files and the wrapper's depth list.  Returns the .oda
    filename (relative to openda_dir) to hand to oda_run.sh.

    `max_threads` caps OpenDA's concurrent model instances (the parallelisation knob); None =
    the default n_members+1 (main + every member run in parallel). Lower it to throttle CPU/IO
    contention or memory on a single machine.
    """
    if filter_type not in FILTERS:
        raise ValueError(f"unknown filter '{filter_type}'; choose from {sorted(FILTERS)}")
    if not obs_depths:
        raise ValueError("obs_depths is empty — no observations to assimilate")
    spec = FILTERS[filter_type]
    depths = sorted(obs_depths)

    stoch_dir    = os.path.join(openda_dir, "stochModel")
    template_dir = os.path.join(stoch_dir, "template")
    obs_cfg_dir  = os.path.join(openda_dir, "stochObserver")
    algo_dir     = os.path.join(openda_dir, "algorithms")
    for d in (stoch_dir, template_dir, obs_cfg_dir, algo_dir):
        os.makedirs(d, exist_ok=True)   # openda_simstrat/ is generated on demand

    algo_file = f"{filter_type}.gen.xml"
    # end_day = end_date + 1 (midnight after end_date) so the last day's noon obs (end_date + 0.5)
    # falls inside the window. Note: the native EnKF instead clamps its last window to end_date
    # midnight, so OpenDA may run one extra noon assimilation at the very end — a small span
    # difference to keep in mind when comparing the two engines, not a defect here.
    start_day, end_day = _simstrat_day(start_date), _simstrat_day(end_date) + 1

    oda = _ODA.format(
        filter=filter_type, algo_class=spec["class"], algo_config=algo_file,
        results_dir=results_subdir, results_file=results_filename(filter_type),
    )
    parallel   = _PARALLEL.format(max_threads=max_threads if max_threads else n_members + 1)
    stochmodel = _STOCHMODEL.format(predictors=_predictor_lines(depths),
                                    stoch_restart_info=_STOCH_RESTART_INFO if spec.get("needs_restart") else "")
    # instanceDir: relative is fine for the Kalman filters.  PF additionally uses the INSTANCE_DIR
    # restart token (see _STOCH_RESTART_INFO), and OpenDA only resolves that correctly when the
    # model run dir is ABSOLUTE — a relative getModelRunDir() gets re-rooted under the stoch
    # configRootDir, doubling the path (.../stochModel/stochModel/...).  config.py is generated in
    # the OpenDA runtime (WSL, see main.py), so abspath yields the /mnt/... form OpenDA sees.
    # Per-member work dirs live inside this run's own dir at Results/work0..N (instanceDir is the
    # prefix OpenDA appends the instance number to). The relative form is resolved from the stoch
    # configRootDir (openda_dir/stochModel), hence ../Results/work; PF needs it ABSOLUTE (restart token).
    if spec.get("needs_restart"):
        instance_dir = os.path.abspath(os.path.join(openda_dir, "Results", "work"))
    else:
        instance_dir = "../Results/work"
    model      = _MODEL.format(filter=filter_type, instance_dir=instance_dir,
                               ref_year=SIMSTRAT_REF_YEAR, outputs=_output_lines(depths),
                               restart_info=_RESTART_INFO if spec.get("needs_restart") else "")
    obs_fmt    = _OBS_FORMATTER.format(ref_year=SIMSTRAT_REF_YEAR, start_day=start_day,
                                       end_day=end_day, rows=_obs_formatter_rows(depths, obs_std))
    model_fmt  = _MODEL_FORMATTER.format(ref_year=SIMSTRAT_REF_YEAR, rows=_model_formatter_rows(depths))
    time_ctrl  = _TIME_CONTROL.format(ref_year=SIMSTRAT_REF_YEAR, start_day=start_day, end_day=end_day)
    algo_cfg   = _ALGORITHM.format(root=spec["root"], schema=spec["schema"], filter=filter_type,
                                   n_members=n_members, extra=spec.get("extra", ""))

    _write(os.path.join(openda_dir, "run.oda"), oda)
    _write(os.path.join(openda_dir, "parallel.gen.xml"), parallel)
    _write(os.path.join(stoch_dir, "simstratStochModel.gen.xml"), stochmodel)
    _write(os.path.join(stoch_dir, "simstratModel.gen.xml"), model)
    _write(os.path.join(obs_cfg_dir, "timeSeriesFormatter.gen.xml"), obs_fmt)
    _write(os.path.join(template_dir, "timeSeriesFormatter.gen.xml"), model_fmt)
    _write(os.path.join(template_dir, "time_control.yaml"), time_ctrl)
    _write(os.path.join(template_dir, "obs_depths.json"), json.dumps(depths))
    _write(os.path.join(algo_dir, algo_file), algo_cfg)
    return "run.oda"

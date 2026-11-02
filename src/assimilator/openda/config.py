"""Single source of truth for the OpenDA Kalman-filter configuration.

The per-filter OpenDA setup (EnKF / DEnKF / EnSR) used to be a 4-deep
chain of near-identical hand-maintained files per filter:

    <type>.oda -> parallel_<type>.xml -> <model>StochModel<Type>.xml
               -> <model>Model<Type>.xml -> <model wrapper>.xml

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
    stochModel/<model>StochModel.gen.xml
    stochModel/<model>Model.gen.xml
    stochObserver/timeSeriesFormatter.gen.xml     (observations + window + std)
    stochModel/template/timeSeriesFormatter.gen.xml   (model predictions)
    stochModel/template/time_control.yaml         (start/end window in model days)
    stochModel/template/obs_depths.json           (depth list the wrapper reads)
    algorithms/<filter>.gen.xml                    (full algorithm config from FILTERS)

Everything model-specific (names, the time axis, the state files) comes from the model's OpenDA
binding (openda/binding.py). The template's starting state is written separately, by the adapter
through the binding.
"""

import os
import json
import logging
from xml.sax.saxutils import escape as xml_escape

from ..sigma import SEASONS

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


def is_series_keyed(obs_std):
    """True if obs_std is {(depth, season): sigma} rather than a plain scalar."""
    return isinstance(obs_std, dict) and bool(obs_std) and all(isinstance(k, tuple) for k in obs_std)


def series_specs(depths, obs_std):
    """[(series_id, depth, sigma)] — the observation series to emit, one row each in all four
    generated files. The single place the series set is decided, so they cannot disagree.

    One series per depth, unless obs_std is season-keyed: then one per (depth, season), each with
    its own standardDeviation. That is how the season pair reaches OpenDA EXACTLY -- the
    stochObserver carries one standardDeviation per series, and the series are ours to define
    (assimilator.sigma). A (depth, season) with no observations in the window is
    not emitted, so an operational window inside one season declares only that season's series."""
    if is_series_keyed(obs_std):
        return [(f"T_{depth_label(d)}_{s}", d, obs_std[(float(d), s)])
                for d in depths for s in SEASONS if (float(d), s) in obs_std]
    return [(f"T_{depth_label(d)}", d, obs_std) for d in depths]


def format_day(day):
    """A model day as written to the OpenDA files: fixed precision (1e-8 d, about 1 ms), the same
    everywhere, so an observation at the run's end is never written past it."""
    return f"{day:.8f}"


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
{restart}
</openDaApplication>
"""

# Restart chain only (openda/restart.py): read the previous run's ensemble, save this run's at its
# end. An empty restartInFile is a cold start of the chain (OpenDA ignores it).
_ODA_RESTART = """
\t<restartInFile>{in_file}</restartInFile>
\t<restartOutFilePrefix>rst_</restartOutFilePrefix>
\t<restartOutFileExtension>.zip</restartOutFileExtension>
\t<restartOutFileOnlyAtEndOfRun>true</restartOutFileOnlyAtEndOfRun>{seed}
"""

# Restart chain only: this run's random seed, so each run draws new observation perturbations.
_ODA_SEED = """
\t<initialSeed type="specify" seedValue="{seed}" />"""

_PARALLEL = """<threadConfigstoch>
    <!-- GENERATED by src/assimilator/openda/config.py. maxThreads = n_members + 1 (main + ensemble). -->
    <maxThreads>{max_threads}</maxThreads>
    <stochModelFactory className="org.openda.blackbox.wrapper.BBStochModelFactory">
        <workingDirectory>./stochModel</workingDirectory>
        <configFile>{name}StochModel.gen.xml</configFile>
    </stochModelFactory>
</threadConfigstoch>
"""

_STOCHMODEL = """<?xml version="1.0" encoding="UTF-8"?>
<blackBoxStochModel xmlns="http://www.openda.org"
\txmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
\txsi:schemaLocation="http://www.openda.org http://schemas.openda.org/blackBoxStochModelConfig.xsd">

\t<!-- GENERATED by src/assimilator/openda/config.py. -->

\t<modelConfig>
\t\t<file>./{name}Model.gen.xml</file>
\t</modelConfig>

\t<vectorSpecification>
\t\t<!-- The state vector — updated by the analysis step -->
\t\t<state>
\t\t\t<vector id="{state_id}" />
\t\t</state>
\t\t<!-- Predictors: model output at observation depths, matched against stochObserver -->
\t\t<predictor>
{predictors}
\t\t</predictor>
\t</vectorSpecification>
{stoch_restart_info}
</blackBoxStochModel>
"""

# Stoch-layer restart declaration (PF, restart chain).  The model-layer restartInfo (below)
# checkpoints the model state per instance; this tells the BBStochModel wrapper where to bundle that state into
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
\t\t<file>{wrapper_xml}</file>
\t</wrapperConfig>

\t<aliasValues>
\t\t<alias key="templateDir"  value="template" />
\t\t<alias key="instanceDir"  value="{instance_dir}" />
\t\t<alias key="binDir"       value="bin" />
\t\t<alias key="configFile"   value="{config_file}" />
\t\t<alias key="stateFile"    value="{state_file}" />
\t</aliasValues>

\t<!-- Time in {time_axis}, handed to OpenDA as MJD numbers -->
\t<timeInfoExchangeItems start="start_time" timeStep="time_step" end="end_time"/>

\t<exchangeItems>
\t\t<vector id="start_time"        ioObjectId="time_config" elementId="time_control@1" />
\t\t<vector id="time_step"         ioObjectId="time_config" elementId="time_control@2" />
\t\t<vector id="end_time"          ioObjectId="time_config" elementId="time_control@3" />
\t\t<vector id="{state_id}" ioObjectId="{state_io_object}" elementId="{state_element}" />
{outputs}
\t</exchangeItems>
{restart_info}
\t<doCleanUp>false</doCleanUp>

</blackBoxModelConfig>
"""

# Restart declaration (the PF, see FILTERS["PF"]["needs_restart"], and every run of a restart
# chain): the files that make up a member's state, all saved and restored together (the binding's
# state_files). Restoring only some of them would let the rest overwrite the restored state.
_RESTART_INFO = """
\t<restartInfo dirPrefix="./savedModelState_">
{state_files}
\t</restartInfo>
"""

# Observation formatter (stochObserver/): reads the real-obs CSVs over the window.
_OBS_FORMATTER = """<?xml version="1.0" encoding="UTF-8"?>
<timeSeriesFormatterConfig xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance">
  <!-- GENERATED by src/assimilator/openda/config.py. time column = {time_axis}.
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
  <!-- GENERATED by src/assimilator/openda/config.py. time column = {time_axis}. -->
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

\t<analysisTimes type="fromObservationTimes" skipAtFinalTime="false" />
\t<mainModel stochParameter="false" stochForcing="false" stochInit="false" />
\t<ensembleSize>{n_members}</ensembleSize>
\t<ensembleModel stochParameter="false" stochForcing="false" stochInit="false" />{extra}
</{root}>
"""


# Model time control (template/): OpenDA reads/writes the #oda:time_control line.
_TIME_CONTROL = """%YAML 1.1
---
# GENERATED by src/assimilator/openda/config.py. Time in {time_axis}.
# Format: [start_day, dt_day, end_day]; dt is a placeholder OpenDA overwrites each step.
time: [{start_day}, 1.0, {end_day}]  #oda:time_control
"""


# All four row sets below are driven by series_specs, so the series a run declares are decided
# once. A disagreement between them is fatal inside OpenDA, with nothing in our log to explain it.

def _predictor_lines(depths, obs_std=DEFAULT_OBS_STD):
    return "\n".join(f'\t\t\t<vector id="{sid}" />' for sid, _, _ in series_specs(depths, obs_std))


def _output_lines(depths, obs_std=DEFAULT_OBS_STD):
    specs = series_specs(depths, obs_std)
    pad   = max([len(sid) + 2 for sid, _, _ in specs] + [10])
    lines = []
    for sid, _, _ in specs:
        vid = f'"{sid}"'
        lines.append(f'\t\t<vector id={vid:<{pad}} ioObjectId="output" elementId="model.{sid}" />')
    return "\n".join(lines)


def _obs_formatter_rows(depths, obs_std):
    """How the fitted observation-error model reaches OpenDA: the stochObserver carries one
    standardDeviation per SERIES, so a season-keyed obs_std gives each season its own sigma and
    reproduces the native engine exactly."""
    return "\n".join(
        f'  <timeSeries id="{sid}" status="use" standardDeviation="{sigma}">'
        f'{sid}_real.csv</timeSeries>'
        for sid, _, sigma in series_specs(depths, obs_std)
    )


def _model_formatter_rows(depths, obs_std=DEFAULT_OBS_STD):
    """Every series maps to its DEPTH's single predictor file: the observer samples model output
    only at its own observation times, so two seasons of one depth never collide and the wrapper
    keeps writing one T_<depth>.csv."""
    return "\n".join(
        f'  <timeSeries id="model.{sid}">T_{depth_label(d)}.csv</timeSeries>'
        for sid, d, _ in series_specs(depths, obs_std)
    )


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)


def render(openda_dir, binding, filter_type, n_members, obs_depths, start_date, end_date,
           obs_std=DEFAULT_OBS_STD, results_subdir="Results", max_threads=None, restart=None, seed=None):
    """Render the full lake-specific OpenDA config for `filter_type` into `openda_dir`, for the
    model `binding` (openda/binding.py) describes.

    `obs_depths` is the list of assimilated depths (e.g. auto-detected from the obs
    CSV); it drives the model exchangeItems, the stochModel predictors, both
    timeSeriesFormatter files and the wrapper's depth list.  Returns the .oda
    filename (relative to openda_dir) to hand to oda_run.sh.

    `max_threads` caps OpenDA's concurrent model instances (the parallelisation knob); None =
    the default n_members+1 (main + every member run in parallel). Lower it to throttle CPU/IO
    contention or memory on a single machine.

    `restart` makes this a run of a restart chain (openda/restart.py): None (default) = a plain run;
    "" = the chain's cold start; a path = continue from that restart file. `seed` is the run's
    random seed (restart chain only).
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
        os.makedirs(d, exist_ok=True)   # the OpenDA working dir is generated on demand

    algo_file = f"{filter_type}.gen.xml"
    # The run covers (start_date, end_date] exactly, as the native engines do: start_date and
    # end_date are instants (a plain date is its 00:00), and the members' final state is at end_date.
    start_day, end_day = format_day(binding.model_time(start_date)), format_day(binding.model_time(end_date))
    names = {"name": binding.name, "time_axis": binding.time_axis}

    chain = restart is not None
    # restartInFile is relative to openda_dir: OpenDA prepends that folder even to absolute paths.
    in_file = os.path.relpath(restart, openda_dir).replace(os.sep, "/") if restart else ""
    oda = _ODA.format(
        filter=filter_type, algo_class=spec["class"], algo_config=algo_file,
        results_dir=results_subdir, results_file=results_filename(filter_type),
        restart="" if not chain else _ODA_RESTART.format(
            in_file=xml_escape(in_file), seed="" if seed is None else _ODA_SEED.format(seed=int(seed))),
    )
    # Declare each member's state files (so OpenDA can save and restore them) for the PF, and in a
    # restart chain, where the ensemble is handed on to the next run.
    save_member_state = bool(spec.get("needs_restart")) or chain
    parallel   = _PARALLEL.format(max_threads=max_threads if max_threads else n_members + 1, **names)
    stochmodel = _STOCHMODEL.format(predictors=_predictor_lines(depths, obs_std), state_id=binding.state_id, **names,
                                    stoch_restart_info=_STOCH_RESTART_INFO if save_member_state else "")
    # instanceDir: relative is fine for the Kalman filters.  PF additionally uses the INSTANCE_DIR
    # restart token (see _STOCH_RESTART_INFO), and OpenDA only resolves that correctly when the
    # model run dir is ABSOLUTE — a relative getModelRunDir() gets re-rooted under the stoch
    # configRootDir, doubling the path (.../stochModel/stochModel/...).  config.py is generated in
    # the OpenDA runtime (WSL, see main.py), so abspath yields the /mnt/... form OpenDA sees.
    # Per-member work dirs live inside this run's own dir at Results/work0..N (instanceDir is the
    # prefix OpenDA appends the instance number to). The relative form is resolved from the stoch
    # configRootDir (openda_dir/stochModel), hence ../Results/work; PF needs it ABSOLUTE (restart token).
    if save_member_state:
        instance_dir = os.path.abspath(os.path.join(openda_dir, "Results", "work"))
    else:
        instance_dir = "../Results/work"
    restart_info = _RESTART_INFO.format(state_files="\n".join(
        f"\t\t<modelStateFile>{f}</modelStateFile>" for f in binding.state_files))
    model      = _MODEL.format(filter=filter_type, instance_dir=instance_dir, outputs=_output_lines(depths, obs_std),
                               wrapper_xml=binding.wrapper_xml, config_file=binding.config_file,
                               state_file=binding.state_file, state_id=binding.state_id,
                               state_io_object=binding.state_io_object, state_element=binding.state_element,
                               restart_info=restart_info if save_member_state else "", **names)
    obs_fmt    = _OBS_FORMATTER.format(start_day=start_day, end_day=end_day,
                                       rows=_obs_formatter_rows(depths, obs_std), **names)
    model_fmt  = _MODEL_FORMATTER.format(rows=_model_formatter_rows(depths, obs_std), **names)
    time_ctrl  = _TIME_CONTROL.format(start_day=start_day, end_day=end_day, **names)
    algo_cfg   = _ALGORITHM.format(root=spec["root"], schema=spec["schema"], filter=filter_type,
                                   n_members=n_members, extra=spec.get("extra", ""))

    _write(os.path.join(openda_dir, "run.oda"), oda)
    _write(os.path.join(openda_dir, "parallel.gen.xml"), parallel)
    _write(os.path.join(stoch_dir, f"{binding.name}StochModel.gen.xml"), stochmodel)
    _write(os.path.join(stoch_dir, f"{binding.name}Model.gen.xml"), model)
    _write(os.path.join(obs_cfg_dir, "timeSeriesFormatter.gen.xml"), obs_fmt)
    _write(os.path.join(template_dir, "timeSeriesFormatter.gen.xml"), model_fmt)
    _write(os.path.join(template_dir, "time_control.yaml"), time_ctrl)
    _write(os.path.join(template_dir, "obs_depths.json"), json.dumps(depths))
    _write(os.path.join(algo_dir, algo_file), algo_cfg)
    return "run.oda"

"""A model as an OpenDA black box: what the OpenDA engine needs to know about a model.

The engine (adapter.py, config.py, restart.py) holds no model knowledge. It runs any model whose
wrapper keeps this contract, in each OpenDA work dir:

    time_control.yaml      [start, dt, end] in model days, written by OpenDA before each window
    <state_file>           the analysed state vector, one value per line, read and written back
    obs_depths.json        (template) the observation depths predictors are needed at
    T_<depth>m.csv         the wrapper's predictors: the model at each depth at the window end
    forcings/<forcing_name(i)>   (openda dir) member i's perturbed forcing, which the wrapper injects

logs "Model finished (exit <code>)" after each model run (the engine counts them per analysis),
and, in a restart chain, stops when the template carries RESTART_MARKER but its state was not
restored. Everything else -- file names and layout, the time axis, how the template gets its
starting state, the forcing, the outputs, the container -- belongs to the binding.

One binding per model, returned by ForwardModel.openda_binding(); the Simstrat one is
models/simstrat_openda.py. Adding a model to OpenDA = a wrapper script + wrapper XML keeping the
contract above, and a binding.
"""
from abc import ABC, abstractmethod


class OpenDABinding(ABC):
    # --- names ---------------------------------------------------------------------------------
    name           = None   # prefix of the generated model configs (<name>Model.gen.xml, ...)
    config_file    = None   # the model's run configuration in a work dir (wrapper alias configFile)
    state_file     = None   # the state vector file (wrapper alias stateFile)
    state_id       = None   # the state vector's id in OpenDA
    state_io_object = None  # the wrapper XML's dataObject id for the state file
    state_element  = None   # the state vector's element id in that dataObject
    state_files    = ()     # files saved and restored with a member's state (restart chain, PF)
    wrapper_xml    = None   # the wrapper config, under stochModel/
    wrapper_log    = None   # the wrapper's log in a work dir, archived with each run of a chain
    coupling_files = ()     # template files OpenDA owns: never overwritten from the model inputs
    forcing_file   = None   # the control forcing in the model inputs
    shared_inputs  = ()     # long read-only inputs a restart chain keeps once in <openda_dir>/shared
    time_axis      = None   # how the time axis is described in the generated files

    # --- files ---------------------------------------------------------------------------------
    @abstractmethod
    def wrapper_files(self):
        """[(source path, path under stochModel/)]: the hand-maintained wrapper files."""

    @abstractmethod
    def is_template_input(self, name):
        """Whether an entry of the model inputs belongs in the template (not outputs or archives)."""

    # --- time ----------------------------------------------------------------------------------
    @abstractmethod
    def model_time(self, t):
        """The model-day time of an instant (ISO string, date, datetime or Timestamp; naive = UTC)."""

    @abstractmethod
    def model_epoch(self):
        """The naive UTC datetime of model day 0."""

    # --- the template's starting state ---------------------------------------------------------
    @abstractmethod
    def seed_template(self, model_inputs, template_dir):
        """Put the warm start into the template and write its state file from it. Returns
        (source path, number of state values), or None when the inputs hold no warm start."""

    @abstractmethod
    def fill_template_sentinel(self, model_inputs, template_dir, sentinel):
        """Restart chain: remove the template's starting state and fill its state file with
        `sentinel`, so a member that was not restored is caught. Returns the number of values."""

    # --- forcing -------------------------------------------------------------------------------
    @abstractmethod
    def member_forcing(self, ensemble_base, i):
        """Path of member i's perturbed forcing (0 = the unperturbed control)."""

    @abstractmethod
    def forcing_name(self, i):
        """File name of member i's forcing under <openda_dir>/forcings/."""

    @abstractmethod
    def cut_forcing(self, model_inputs, template_dir, start_day, end_day):
        """Restart chain: write the template's control forcing for the run only (with a row on each
        side). Returns (rows written, rows in the control)."""

    @abstractmethod
    def point_config_at_shared(self, template_dir, rel):
        """Restart chain: point the template's model configuration at the shared inputs in `rel`."""

    # --- outputs and running -------------------------------------------------------------------
    @abstractmethod
    def output_depths(self, model_inputs):
        """Depths (positive metres) the model outputs; [] if unknown (no filtering)."""

    @abstractmethod
    def work_output(self, openda_dir, i):
        """Path of member i's trajectory in its OpenDA work dir (this run only)."""

    @abstractmethod
    def container(self, model_cfg):
        """{"image", "binary", "mount_base"}: the container the wrapper runs the model in."""

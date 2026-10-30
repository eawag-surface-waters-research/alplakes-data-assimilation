"""The forward-model interface the engines run against.

The engines (`algorithms/enkf.py`, `algorithms/pf.py`), the forcing perturbation (`perturbate.py`),
the run summary (`summarize.py`) and the pipeline (`pipeline.py`) touch a model ONLY through the
methods below. Everything a model knows about its own files -- input layout, time convention,
forcing format, output format, restart/state format, how it is launched -- lives in its adapter
(e.g. `models/simstrat.py`).

Adding a model = one subclass implementing these methods, registered with
`assimilator.models.register_model`. The adapter can live in this package or in the host repo.

Conventions shared by every method:

  args            the engine's run args (the validated run config plus derived keys such as
                  ensemble_base, member_ids, results_dir, par_file, ref_date)
  ensemble_base   folder holding the members ensemble0..N (0 = unperturbed control)
  depth           observation depth, metres below the surface, positive
  model column    the coordinate the model's state/output uses for a depth (obs_to_sim_col)
  state column    the analysed variable (temperature) on the model's full grid, a 1D array
"""

from abc import ABC, abstractmethod


class ForwardModel(ABC):
    name = None

    # --- runtime -----------------------------------------------------------------------------
    @abstractmethod
    def run_config(self):
        """Runtime defaults merged into the run args (e.g. a Docker image version)."""

    # --- inputs and members --------------------------------------------------------------------
    @abstractmethod
    def model_inputs_ready(self, model_inputs):
        """True if `model_inputs` holds a runnable setup with a warm-start state."""

    @abstractmethod
    def warmup_snapshot(self, model_inputs):
        """Path of the warm-start state the members restart from, or None."""

    @abstractmethod
    def instances_ready(self, ensemble_base, n_members):
        """True if members 0..N exist."""

    @abstractmethod
    def copy_model_inputs(self, cfg):
        """Clone cfg['model_inputs_path'] into members 0..N under cfg['ensemble_base'].
        Must not modify the inputs folder."""

    @abstractmethod
    def set_output_depths(self, model_inputs, ensemble_base, n_members, obs_depths):
        """Make every member output the observation depths. Must not modify `model_inputs`.
        Returns (n_files_rewritten, path_of_extended_output_grid_or_None)."""

    @abstractmethod
    def model_output_depths(self, ensemble_base):
        """Depths (positive metres) the members output; [] if unknown."""

    # --- forcing (for the AR(1) perturbation) ---------------------------------------------------
    @abstractmethod
    def read_forcing(self, model_inputs, ref_year=None):
        """The control forcing as a DataFrame with at least: 'time' (tz-aware UTC), 'step'
        (int64, the row's absolute time key -- identical in every invocation), and the perturbed
        channels 'U', 'V', 'GLOB' (wind components, global radiation). Other columns pass through
        to write_forcing unchanged."""

    @abstractmethod
    def write_forcing(self, member_dir, df):
        """Write a (perturbed) forcing frame, as returned by read_forcing, into a member."""

    # --- running ------------------------------------------------------------------------------
    @abstractmethod
    def start_containers(self, args, max_workers=None):
        """Acquire whatever the members run in (called once per run)."""

    @abstractmethod
    def stop_containers(self, args):
        """Release what start_containers acquired (called in a finally)."""

    @abstractmethod
    def run_window(self, window_start, window_end, args, max_workers=None):
        """Advance every member in args['member_ids'] from window_start to window_end, each from
        its own current state. Returns the list of member ids that failed."""

    # --- time and output ----------------------------------------------------------------------
    @abstractmethod
    def read_ref_date(self, ensemble_base):
        """The model's time origin (tz-aware datetime) for its day-based time axis."""

    @abstractmethod
    def output_epoch(self):
        """Origin of the time axis in read_output tables (tz-aware datetime)."""

    @abstractmethod
    def obs_to_sim_col(self, depth):
        """Map an observation depth (positive metres) to the model's column coordinate."""

    @abstractmethod
    def sim_col_to_depth(self, col):
        """Inverse of obs_to_sim_col: a model column coordinate -> depth (positive metres)."""

    @abstractmethod
    def member_output_file(self, ensemble_base, member_id, results_dir):
        """Path of a member's accumulated output trajectory."""

    @abstractmethod
    def read_output(self, path):
        """A trajectory file -> (times in days since output_epoch, column coords, values[t, z])."""

    @abstractmethod
    def load_T_window(self, ensemble_dir, args, window_start, window_end):
        """The rows a member output in [window_start, window_end], as a DataFrame indexed by
        UTC time with one float column per model column coordinate."""

    @abstractmethod
    def clear_member_outputs(self, ensemble_base, member_ids, results_dir):
        """Delete accumulated outputs (fresh run)."""

    @abstractmethod
    def accumulate_mean(self, member_files, mean_path):
        """Write the ensemble-mean trajectory of `member_files` to `mean_path`."""

    @abstractmethod
    def mean_traj_path(self, ensemble_base, algorithm):
        """Where the ensemble-mean trajectory is written."""

    # --- state ----------------------------------------------------------------------------------
    @abstractmethod
    def read_snapshot_T(self, member_id, args):
        """A member's current state -> (state column, column heights, lake level)."""

    @abstractmethod
    def write_snapshot_T(self, member_id, T_new, args):
        """Write an analysed state column back into a member's state."""

    @abstractmethod
    def clear_member_state(self, ensemble_base, member_ids, results_dir):
        """Delete members' live state so the next window restarts from the warm start."""

    @abstractmethod
    def copy_member_state(self, src_id, member_ids, args):
        """Overwrite every other member's live state with member `src_id`'s (PF resampling)."""

    @abstractmethod
    def write_mean_state(self, cfg, out_path):
        """Write the ensemble-mean analysis state of members 1..N as a warm start at `out_path`
        (the deterministic forecast's initial condition)."""

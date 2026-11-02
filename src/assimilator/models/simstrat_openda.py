"""Simstrat as an OpenDA black box (see assimilator/openda/binding.py for the contract).

The wrapper is static/openda/simstrat_wrapper_enkf.py with static/openda/simstratWrapperEnKF.xml.
Each OpenDA work dir is a clone of the template: the model inputs, Settings.par, the warm start as
Results/simulation-snapshot.dat (Simstrat "Continue from last snapshot") and the full-grid
temperature column as temperature_state.txt, the state OpenDA analyses.
"""
import os
import re
import glob
import shutil
from datetime import date, datetime, timezone

from ..openda.binding import OpenDABinding
from .simstrat import SIMSTRAT_REF_YEAR, read_snapshot, _read_z_out

STATIC_OPENDA = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))), "static", "openda")

# Settings.par key -> the shared input it names (restart chain).
PAR_SHARED = {"Inflow": "Qin.dat", "Inflow temperature": "Tin.dat", "Inflow salinity": "Sin.dat"}


class SimstratOpenDA(OpenDABinding):
    name           = "simstrat"
    config_file    = "Settings.par"
    state_file     = "temperature_state.txt"
    state_id       = "temperature.state"
    state_io_object = "state_temperature"    # static/openda/simstratWrapperEnKF.xml
    state_element  = "temperature_state"
    # Both travel together on every save/restore: Results/simulation-snapshot.dat is Simstrat's full
    # binary state and the true continuity between windows; temperature_state.txt is OpenDA's
    # T-only view, re-injected into the snapshot at the start of every step.
    state_files    = ("Results/simulation-snapshot.dat", "temperature_state.txt")
    wrapper_xml    = "simstratWrapperEnKF.xml"
    wrapper_log    = "simstrat_wrapper_enkf.log"
    coupling_files = ("temperature_state.txt", "time_control.yaml", "timeSeriesFormatter.xml")
    forcing_file   = "Forcing.dat"
    # These cover the whole record (~35 MB for one lake) while a run reads a day of them.
    shared_inputs  = ("Qin.dat", "Tin.dat", "Sin.dat", "AED2_inflow")
    time_axis      = f"Simstrat days (days since 1 Jan {SIMSTRAT_REF_YEAR})"

    def wrapper_files(self):
        return [(os.path.join(STATIC_OPENDA, "simstratWrapperEnKF.xml"), "simstratWrapperEnKF.xml"),
                (os.path.join(STATIC_OPENDA, "simstrat_wrapper_enkf.py"), os.path.join("bin", "simstrat_wrapper_enkf.py"))]

    def is_template_input(self, name):
        # not the heavy Results/, the visualisation-only ref/, or archived dated snapshots: the
        # warm start is placed into template/Results/ by seed_template
        return name not in ("Results", "ref") and not name.startswith("simulation-snapshot_")

    def model_time(self, t):
        if isinstance(t, str):
            t = datetime.fromisoformat(t)
        if isinstance(t, date) and not isinstance(t, datetime):
            t = datetime(t.year, t.month, t.day)
        if t.tzinfo is None:
            t = t.replace(tzinfo=timezone.utc)
        return (t - datetime(SIMSTRAT_REF_YEAR, 1, 1, tzinfo=timezone.utc)).total_seconds() / 86400.0

    def model_epoch(self):
        return datetime(SIMSTRAT_REF_YEAR, 1, 1)

    @staticmethod
    def _warm_starts(model_inputs):
        # the dated archive is stable; the live model_inputs/Results/ copy may have been overwritten
        return sorted(glob.glob(os.path.join(model_inputs, "simulation-snapshot_*.dat")))

    def _write_state(self, template_dir, values):
        with open(os.path.join(template_dir, self.state_file), "w") as f:
            for v in values:
                f.write(f"{float(v):.6f}\n")

    def seed_template(self, model_inputs, template_dir):
        dated = self._warm_starts(model_inputs)
        if not dated:
            return None
        target = os.path.join(template_dir, "Results", "simulation-snapshot.dat")
        os.makedirs(os.path.dirname(target), exist_ok=True)
        shutil.copy2(dated[-1], target)
        # OpenDA's initial state: the warm start's full-grid temperature profile, one value per cell
        T = read_snapshot(dated[-1], par_path=os.path.join(template_dir, self.config_file)).model["T"]
        self._write_state(template_dir, T)
        return dated[-1], len(T)

    def fill_template_sentinel(self, model_inputs, template_dir, sentinel):
        dated = self._warm_starts(model_inputs)
        if not dated:
            raise FileNotFoundError(f"no simulation-snapshot_*.dat in {model_inputs} — needed for the "
                                    f"grid size of the restart template")
        live = os.path.join(template_dir, "Results", "simulation-snapshot.dat")
        if os.path.exists(live):
            os.remove(live)
        n = len(read_snapshot(dated[-1], par_path=os.path.join(template_dir, self.config_file)).model["T"])
        self._write_state(template_dir, [sentinel] * n)
        return n

    def member_forcing(self, ensemble_base, i):
        return os.path.join(ensemble_base, f"ensemble{i}", self.forcing_file)

    def forcing_name(self, i):
        return f"Forcing_{i}.dat"

    def cut_forcing(self, model_inputs, template_dir, start_day, end_day):
        header, *rows = open(os.path.join(model_inputs, self.forcing_file)).read().splitlines()

        def day(r):
            try:
                return float(r.split()[0])
            except (ValueError, IndexError):
                return None
        days = [(day(r), r) for r in rows]
        lo = max([t for t, _ in days if t is not None and t <= start_day] or [-1e9])
        hi = min([t for t, _ in days if t is not None and t >= end_day] or [1e9])
        keep = [r for t, r in days if t is not None and lo <= t <= hi]
        with open(os.path.join(template_dir, self.forcing_file), "w") as f:
            f.write("\n".join([header] + keep) + "\n")
        return len(keep), len(rows)

    def point_config_at_shared(self, template_dir, rel):
        par_path = os.path.join(template_dir, self.config_file)
        s = open(par_path).read()
        for key, name in PAR_SHARED.items():
            s = re.sub(rf'("{key}"\s*:\s*)"[^"]*"', lambda m: f'{m.group(1)}"{rel}/{name}"', s)
        s = re.sub(r'("PathAED2inflow"\s*:\s*)"[^"]*"', lambda m: f'{m.group(1)}"{rel}/AED2_inflow/"', s)
        with open(par_path, "w") as f:
            f.write(s)

    def output_depths(self, model_inputs):
        return _read_z_out(os.path.join(model_inputs, "z_out.dat"))

    def work_output(self, openda_dir, i):
        return os.path.join(openda_dir, "Results", f"work{i}", "Results", "T_out.dat")

    def container(self, model_cfg):
        model_cfg = model_cfg or {}
        return {"image": f"eawag/simstrat:{model_cfg.get('simstrat_version', '3.0.4')}",
                "binary": model_cfg.get("simstrat_binary", "/simstrat/build/simstrat"),
                "mount_base": "/simstrat/run"}

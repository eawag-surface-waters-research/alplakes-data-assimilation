"""A deliberately trivial forward model, to prove the engines need nothing Simstrat-specific.

State: one temperature column on a 20 m water column (0.5 m cells). Each hour every cell relaxes
toward an equilibrium profile set by the member's forcing (wind mixes, radiation warms the top).
Files per member: forcing.csv, state.npy (warm start), live_state.npy, <results_dir>/out.csv.
"""
import os
import json
import shutil
from datetime import datetime, timezone, timedelta

import numpy as np
import pandas as pd

from assimilator.models.base import ForwardModel

EPOCH       = datetime(2000, 1, 1, tzinfo=timezone.utc)
LAKE_LEVEL  = 20.0
Z_VOLUME    = np.arange(0.25, LAKE_LEVEL, 0.5)          # cell centres, height above the bed
OUT_DEPTHS  = [float(d) for d in range(1, 20)]          # output depths, positive metres
RELAX       = 0.05                                      # per hour


def equilibrium(glob, u):
    depth = LAKE_LEVEL - Z_VOLUME
    return 6.0 + (glob / 25.0) * np.exp(-depth / (3.0 + 0.5 * abs(u)))


def write_inputs(folder, start, days, glob=200.0, u=2.0):
    """A model-inputs folder: forcing over [start, start + days] and a warm-start state."""
    os.makedirs(folder, exist_ok=True)
    hours = np.arange(int(days * 24) + 1)
    t_days = ((start - EPOCH).total_seconds() + hours * 3600) / 86400
    phase = np.sin(2 * np.pi * (hours % 24) / 24)
    pd.DataFrame({"time_days": t_days, "U": u + 0 * hours, "V": 0.0 * hours,
                  "GLOB": np.clip(glob * phase, 0, None), "T": 12.0 + 0 * hours}
                 ).to_csv(os.path.join(folder, "forcing.csv"), index=False)
    np.save(os.path.join(folder, "state.npy"), np.full(len(Z_VOLUME), 8.0))


class ToyModel(ForwardModel):
    name = "toy"

    def run_config(self):
        return {}

    # inputs and members
    def model_inputs_ready(self, model_inputs):
        return all(os.path.isfile(os.path.join(model_inputs, f)) for f in ("state.npy", "forcing.csv"))

    def warmup_snapshot(self, model_inputs):
        return os.path.join(model_inputs, "state.npy")

    def instances_ready(self, ensemble_base, n_members):
        return all(os.path.isfile(os.path.join(ensemble_base, f"ensemble{i}", "state.npy"))
                   for i in range(n_members + 1))

    def copy_model_inputs(self, cfg):
        for i in range(cfg["n_members"] + 1):
            dst = os.path.join(cfg["ensemble_base"], f"ensemble{i}")
            os.makedirs(dst, exist_ok=True)
            for f in ("state.npy", "forcing.csv"):
                shutil.copy2(os.path.join(cfg["model_inputs_path"], f), dst)

    def set_output_depths(self, model_inputs, ensemble_base, n_members, obs_depths):
        return 0, None

    def model_output_depths(self, ensemble_base):
        return list(OUT_DEPTHS)

    # forcing
    def read_forcing(self, model_inputs, ref_year=None):
        df = pd.read_csv(os.path.join(model_inputs, "forcing.csv"))
        df["time"] = (pd.Timestamp(EPOCH) + pd.to_timedelta(df["time_days"], unit="D")).dt.round("h")
        df["step"] = np.rint(df["time_days"].values * 86400).astype(np.int64)
        return df

    def write_forcing(self, member_dir, df):
        df[["time_days", "U", "V", "GLOB", "T"]].to_csv(os.path.join(member_dir, "forcing.csv"),
                                                       index=False)

    # running
    def start_containers(self, args, max_workers=None):
        pass

    def stop_containers(self, args):
        pass

    def _state_path(self, args, i, live=True):
        base = os.path.join(args["ensemble_base"], f"ensemble{i}")
        return os.path.join(base, "live_state.npy") if live else os.path.join(base, "state.npy")

    def run_window(self, window_start, window_end, args, max_workers=None):
        for i in args["member_ids"]:
            member = os.path.join(args["ensemble_base"], f"ensemble{i}")
            live = self._state_path(args, i)
            T = np.load(live if os.path.exists(live) else self._state_path(args, i, live=False))
            f = self.read_forcing(member)
            out_dir = os.path.join(member, args["results_dir"])
            os.makedirs(out_dir, exist_ok=True)
            rows, t = [], window_start
            while t < window_end:
                t = min(t + timedelta(hours=1), window_end)
                row = f.iloc[int(np.argmin(np.abs(f["time"] - pd.Timestamp(t))))]
                T = T + RELAX * (equilibrium(row["GLOB"], row["U"]) - T)
                prof = np.interp(OUT_DEPTHS, (LAKE_LEVEL - Z_VOLUME)[::-1], T[::-1])
                rows.append([(t - EPOCH).total_seconds() / 86400, *prof[::-1]])
            np.save(live, T)
            path = self.member_output_file(args["ensemble_base"], i, args["results_dir"])
            header = not os.path.exists(path)
            cols = ["Datetime"] + [f"{-d:.2f}" for d in OUT_DEPTHS[::-1]]
            pd.DataFrame(rows, columns=cols).to_csv(path, mode="a", header=header, index=False)
        return []

    # time and output
    def read_ref_date(self, ensemble_base):
        return EPOCH

    def output_epoch(self):
        return EPOCH

    def obs_to_sim_col(self, depth):
        return -depth

    def sim_col_to_depth(self, col):
        return -col

    def member_output_file(self, ensemble_base, member_id, results_dir):
        return os.path.join(ensemble_base, f"ensemble{member_id}", results_dir, "out.csv")

    def read_output(self, path):
        df = pd.read_csv(path)
        return (df.iloc[:, 0].to_numpy(float), np.array([float(c) for c in df.columns[1:]]),
                df.iloc[:, 1:].to_numpy(float))

    def load_T_window(self, ensemble_dir, args, window_start, window_end):
        df = pd.read_csv(os.path.join(ensemble_dir, args["results_dir"], "out.csv"))
        idx = (pd.Timestamp(EPOCH) + pd.to_timedelta(df.pop("Datetime"), unit="D")).dt.round("1h")
        df.index, df.columns = idx, df.columns.astype(float)
        return df[(df.index >= window_start) & (df.index <= window_end)]

    def clear_member_outputs(self, ensemble_base, member_ids, results_dir):
        for i in member_ids:
            p = self.member_output_file(ensemble_base, i, results_dir)
            if os.path.exists(p):
                os.remove(p)

    def accumulate_mean(self, member_files, mean_path):
        frames = [pd.read_csv(f) for f in member_files if os.path.exists(f)]
        pd.concat(frames).groupby("Datetime", as_index=False).mean().to_csv(mean_path, index=False)

    def mean_traj_path(self, ensemble_base, algorithm):
        return os.path.join(ensemble_base, f"out_{algorithm.lower()}_mean.csv")

    # state
    def read_snapshot_T(self, member_id, args):
        live = self._state_path(args, member_id)
        return np.load(live), Z_VOLUME.copy(), LAKE_LEVEL

    def write_snapshot_T(self, member_id, T_new, args):
        np.save(self._state_path(args, member_id), np.asarray(T_new))

    def clear_member_state(self, ensemble_base, member_ids, results_dir):
        for i in member_ids:
            p = os.path.join(ensemble_base, f"ensemble{i}", "live_state.npy")
            if os.path.exists(p):
                os.remove(p)

    def copy_member_state(self, src_id, member_ids, args):
        for i in member_ids:
            if i != src_id:
                shutil.copy2(self._state_path(args, src_id), self._state_path(args, i))

    def write_mean_state(self, cfg, out_path):
        cols = [np.load(os.path.join(cfg["ensemble_base"], f"ensemble{i}", "live_state.npy"))
                for i in range(1, cfg["n_members"] + 1)]
        np.save(out_path, np.mean(cols, axis=0))
        with open(out_path + ".json", "w") as f:
            json.dump({"members": cfg["n_members"]}, f)

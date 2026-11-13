"""Online estimation of the observation error R and the model error q (native EnKF only).

Per observation depth, after each analysis (d_b, d_a: obs minus prior / analysis mean; s2: prior
variance; primes: previous analysis at that depth, <= PAIR_MAX_GAP_H earlier):
  q^2 <- max(0, q^2 + (1-lam)(d_b^2 - s2 - R^2))
  R^2 <- max(sensor^2, lam R^2 + (1-lam)(d_a' d_b' - d_b d_b'))   "split"
  R^2 <- max(sensor^2, lam R^2 + (1-lam)(d_b^2 - s2))             "consistency"
With q on, members get centred noise N(0, D C D) (D = q, C the seasonal depth correlation),
optionally followed by density stabilisation. State is saved per analysis in ensemble_base, so a
run continued in slices learns as one continuous run.
"""
import os
import json
import logging

import numpy as np
import pandas as pd

from .rng import time_keyed_rng
from .sigma import season_of, SEASONS

logger = logging.getLogger(__name__)

STATE_FILENAME = "online_errors_state.json"
LOG_FILENAME   = "online_errors_by_depth.csv"
STATE_KEEP     = 60       # saved analyses
PAIR_MAX_GAP_H = 36       # max gap between two analyses paired by the split rule
STAB_TOL       = 0.001    # kg/m3, smallest inversion removed

DEFAULTS = {
    "R_rule":           "split",   # "split" or "consistency"
    "q":                True,      # learn and add model error
    "lambda":           0.95,      # forgetting factor
    "sensor":           0.05,      # degC, R start and floor
    "warmup_days":      15,        # R fixed at sensor before this
    "max_nis":          25.0,      # gross-error threshold, 0 = off
    "check_after_days": 15,        # gross-error check starts after this
    "taper_m":          10.0,      # noise tapers to 0 over this below the deepest depth
    "stabilize":        True,      # remove density inversions after the noise
}
KEYS = set(DEFAULTS) | {"correlation", "correlation_file"}


def validate_online_errors(block, engine, algorithm):
    """The "online_errors" block with defaults filled; ValueError on a bad one."""
    if not isinstance(block, dict):
        raise ValueError(f"'online_errors' must be an object, got {type(block).__name__}")
    out = {k: v for k, v in block.items() if not str(k).startswith("_")}
    unknown = set(out) - KEYS
    if unknown:
        raise ValueError(f"'online_errors' has unknown key(s) {sorted(unknown)}; allowed: {sorted(KEYS)}")
    if engine != "python" or algorithm != "EnKF":
        raise ValueError("'online_errors' needs the native EnKF (engine 'python', algorithm 'EnKF')")
    for k, v in DEFAULTS.items():
        out.setdefault(k, v)
    if out["R_rule"] not in ("split", "consistency"):
        raise ValueError(f"online_errors.R_rule must be 'split' or 'consistency', got {out['R_rule']!r}")
    if not 0.0 < float(out["lambda"]) < 1.0:
        raise ValueError(f"online_errors.lambda must be in (0, 1), got {out['lambda']!r}")
    if float(out["sensor"]) <= 0:
        raise ValueError(f"online_errors.sensor must be > 0, got {out['sensor']!r}")
    if out["q"] and out.get("correlation") is None and not out.get("correlation_file"):
        raise ValueError("online_errors with q needs 'correlation' or 'correlation_file' "
                         "(`assimilate.py fit noise-correlation`)")
    if out.get("correlation_file"):
        out["correlation_file"] = os.path.abspath(os.path.expanduser(os.path.expandvars(out["correlation_file"])))
    return out


def load_correlation(spec):
    """(depths, {season: C}) from the inline correlation or the file; (None, None) if neither."""
    c = spec.get("correlation")
    if c is None and spec.get("correlation_file"):
        path = spec["correlation_file"]
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{path} not found -- fit it with `assimilate.py fit noise-correlation`")
        with open(path, encoding="utf-8") as f:
            c = json.load(f)
    if c is None:
        return None, None
    depths = np.asarray(c["depths"], dtype=float)
    corr = {}
    for s in SEASONS:
        C = np.asarray(c["corr"][s], dtype=float)
        if C.shape != (len(depths), len(depths)):
            raise ValueError(f"noise correlation '{s}' is {C.shape}, expected {len(depths)}x{len(depths)}")
        corr[s] = C
    return depths, corr


class OnlineErrors:
    """R and q on a depth grid, updated once per analysis."""

    def __init__(self, spec, grid, corr=None):
        self.spec   = spec
        self.grid   = np.asarray(grid, dtype=float)
        self.corr   = corr
        self.lam    = float(spec["lambda"])
        self.sensor = float(spec["sensor"])
        n = len(self.grid)
        self.R2   = np.full(n, self.sensor ** 2)
        self.q2   = np.zeros(n)
        self.n    = np.zeros(n, dtype=int)
        self.prev = {}        # grid index -> (time, d_b, d_a) of the last assimilated obs
        self.t0   = None      # first analysis

    def _index(self, depths):
        """Grid index per depth, -1 if off the grid."""
        out = []
        for z in np.asarray(depths, dtype=float):
            k = np.flatnonzero(np.isclose(self.grid, z))
            out.append(int(k[0]) if len(k) else -1)
        return np.array(out, dtype=int)

    def R(self, depths):
        """Observation error sd at `depths`, interpolated off the grid."""
        return np.sqrt(np.interp(np.asarray(depths, dtype=float), self.grid, self.R2))

    def q(self):
        return np.sqrt(self.q2)

    def checking(self, when):
        """True once the gross-error check is active."""
        return bool(self.spec["max_nis"]) and self.t0 is not None and \
            (pd.Timestamp(when) - self.t0) >= pd.Timedelta(days=int(self.spec["check_after_days"]))

    def learn(self, when, depths, d_b, d_a, s2, R_used):
        """Update R and q from one analysis; d_a is NaN for rejected observations."""
        when = pd.Timestamp(when)
        self.t0 = self.t0 if self.t0 is not None else when
        warm = (when - self.t0) < pd.Timedelta(days=int(self.spec["warmup_days"]))
        a = 1.0 - self.lam
        for k, db, da, s, r in zip(self._index(depths), d_b, d_a, s2, R_used):
            if k < 0 or not np.isfinite(db):
                continue
            if self.spec["q"]:
                self.q2[k] = max(0.0, self.q2[k] + a * (db ** 2 - s - r ** 2))
            if self.spec["R_rule"] == "consistency":
                if not warm:
                    self.R2[k] = max(self.sensor ** 2, self.lam * self.R2[k] + a * (db ** 2 - s))
            else:
                p = self.prev.get(k)
                if not warm and p is not None and (when - p[0]) <= pd.Timedelta(hours=PAIR_MAX_GAP_H):
                    self.R2[k] = max(self.sensor ** 2, self.lam * self.R2[k] + a * (p[2] * p[1] - db * p[1]))
                self.prev[k] = (when, float(db), float(da)) if np.isfinite(da) else None
            self.n[k] += 1
        self.prev = {k: v for k, v in self.prev.items() if v is not None}

    def noise_factor(self, when):
        """L with L L^T = D C D for the season of `when`; None if no noise."""
        if not self.spec["q"] or self.corr is None or not np.any(self.q2 > 0):
            return None
        q = self.q()
        w, V = np.linalg.eigh(np.outer(q, q) * self.corr[season_of(pd.Timestamp(when).month)])
        return V * np.sqrt(np.clip(w, 0.0, None))

    def echo(self):
        """Settings a saved state must match to be resumed."""
        return {"R_rule": self.spec["R_rule"], "q": bool(self.spec["q"]), "lambda": self.lam,
                "sensor": self.sensor, "depths": [round(float(z), 4) for z in self.grid]}

    def to_state(self):
        return {"t0": self.t0.isoformat(), "R2": self.R2.tolist(), "q2": self.q2.tolist(),
                "n": self.n.tolist(),
                "prev": {str(k): [t.isoformat(), db, da] for k, (t, db, da) in self.prev.items()}}

    def from_state(self, s):
        self.t0 = pd.Timestamp(s["t0"])
        self.R2, self.q2 = np.asarray(s["R2"], dtype=float), np.asarray(s["q2"], dtype=float)
        self.n = np.asarray(s["n"], dtype=int)
        self.prev = {int(k): (pd.Timestamp(v[0]), float(v[1]), float(v[2])) for k, v in s["prev"].items()}


def _load_states(path, echo):
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        logger.warning(f"online errors: state unreadable ({e}) -- cold start")
        return {}
    if data.get("params") != echo:
        logger.warning("online errors: settings changed -- cold start")
        return {}
    return {pd.Timestamp(k): v for k, v in data.get("states", {}).items()}


def _save_states(path, echo, states):
    keep = dict(sorted(states.items())[-STATE_KEEP:])
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"params": echo, "states": {k.isoformat(): v for k, v in keep.items()}}, f)
    os.replace(tmp, path)


class OnlineRun:
    """OnlineErrors for one run: resumes from the last state <= start_date, saves and logs."""

    def __init__(self, args, obs_depths):
        spec = args["online_errors"]
        depths, corr = load_correlation(spec)
        if depths is None:
            depths = np.unique(np.asarray(obs_depths, dtype=float))
        else:
            off = sorted(set(np.round(obs_depths, 4)) - set(np.round(depths, 4)))
            if off:
                logger.warning(f"online errors: obs depths {off} not on the noise grid (R interpolated, not learned)")
        self.oe = OnlineErrors(spec, depths, corr)
        self.state_path = os.path.join(args["ensemble_base"], STATE_FILENAME)
        self.log_path   = os.path.join(args["ensemble_base"], LOG_FILENAME)
        self.seed       = int(args.get("rng_seed", 42))
        if args.get("reset"):
            for p in (self.state_path, self.log_path):
                if os.path.exists(p):
                    os.remove(p)
        self.states = _load_states(self.state_path, self.oe.echo())
        start = pd.Timestamp(args["start_date"])
        resume = max((t for t in self.states if t <= start), default=None)
        if resume is not None:
            self.oe.from_state(self.states[resume])
        elif self.states:
            logger.warning(f"online errors: no saved state at or before {start} -- cold start")
        R = np.sqrt(self.oe.R2)
        logger.info(f"[online errors] R_rule={spec['R_rule']} q={'on' if spec['q'] else 'off'} "
                    f"on {len(self.oe.grid)} depths, "
                    f"{'resumed from ' + resume.isoformat() if resume is not None else 'cold start'} "
                    f"(R {R.min():.3f}-{R.max():.3f}, q max {self.oe.q().max():.3f} degC)")

    def commit(self, when, n_rejected=0, n_stabilized=0):
        """Save the state after the analysis at `when` and log R and q."""
        when = pd.Timestamp(when)
        self.states[when] = self.oe.to_state()
        _save_states(self.state_path, self.oe.echo(), self.states)
        row = {"date": when.isoformat(), "n_rejected": n_rejected, "n_stabilized": n_stabilized,
               **{f"R_{z:g}": round(float(v), 5) for z, v in zip(self.oe.grid, np.sqrt(self.oe.R2))},
               **{f"q_{z:g}": round(float(v), 5) for z, v in zip(self.oe.grid, self.oe.q())}}
        pd.DataFrame([row]).to_csv(self.log_path, mode="a", header=not os.path.exists(self.log_path), index=False)


def draw_noise(L, grid, state_depths, n_members, when, seed, taper_m):
    """Centred noise N(0, L L^T) on the state cells, (n_cells, n_members); constant above the
    grid, tapered to 0 over taper_m below it. Time-keyed RNG."""
    rng = time_keyed_rng(seed, "process_noise", int(pd.Timestamp(when).timestamp()))
    e = rng.standard_normal((n_members, len(grid))) @ L.T
    if n_members > 1:
        e = (e - e.mean(axis=0)) * np.sqrt(n_members / (n_members - 1))
    xp = np.append(grid, grid[-1] + taper_m)
    out = np.empty((len(state_depths), n_members))
    for m in range(n_members):
        out[:, m] = np.interp(state_depths, xp, np.append(e[m], 0.0), right=0.0)
    return out


def water_density(T):
    """Fresh-water density, kg/m3 (Chen & Millero 1977, no salinity)."""
    return (999.8395 + 6.7914e-2 * T - 9.0894e-3 * T ** 2 + 1.0171e-4 * T ** 3
            - 1.2846e-6 * T ** 4 + 1.1592e-8 * T ** 5 - 5.0125e-11 * T ** 6)


def stabilize(T, volumes, depth, z_max, tol=STAB_TOL):
    """Merge density inversions > tol above z_max into volume-weighted means (heat conserved).
    Returns (T, number of cells changed)."""
    T, volumes, depth = (np.asarray(a, dtype=float) for a in (T, volumes, depth))
    order = np.argsort(-depth)                          # bottom first
    sel = order[depth[order] <= z_max]
    blocks = []                                         # [T, V, cells]
    for i in sel:
        blocks.append([T[i], volumes[i], [i]])
        while len(blocks) > 1 and water_density(blocks[-1][0]) > water_density(blocks[-2][0]) + tol:
            t2, v2, c2 = blocks.pop()
            t1, v1, c1 = blocks.pop()
            blocks.append([(t1 * v1 + t2 * v2) / (v1 + v2), v1 + v2, c1 + c2])
    out = T.copy()
    for t, _, cells in blocks:
        out[cells] = t
    return out, int((out != T).sum())

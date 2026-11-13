"""Vertical localization of the native EnKF: a Gaspari-Cohn taper on the gain.

With 20 members a correlation below ~2/sqrt(20) = 0.45 cannot be told from sampling noise. The taper
damps the state-obs and obs-obs covariances (PHT and HPHT) by the depth separation: weight 1 at zero,
0 at a per-depth radius (`assimilate.py fit localization`). Both sides take the same taper, otherwise
the inverse still sees the damped observations; a binary mask is not positive semi-definite.
Cells beyond every radius are not updated and run free.
"""
import os
import json

import numpy as np

REACH_MIN = 0.01   # taper weight below which a cell counts as out of reach of an observation
KEYS = {"radius_file", "radius_m", "depths_m",
        "threshold", "n_members", "fitted_on", "estimator"}   # the last four: fit metadata, not used


def validate_localization(block, engine, algorithm):
    """The "localization" block, checked, with the radius file made absolute."""
    if not isinstance(block, dict):
        raise ValueError(f"'localization' must be an object, got {type(block).__name__}")
    out = {k: v for k, v in block.items() if not str(k).startswith("_")}
    unknown = set(out) - KEYS
    if unknown:
        raise ValueError(f"'localization' has unknown key(s) {sorted(unknown)}; allowed: {sorted(KEYS)}")
    if engine != "python" or algorithm != "EnKF":
        raise ValueError("'localization' needs the native EnKF (engine 'python', algorithm 'EnKF')")
    if bool(out.get("radius_file")) == ("radius_m" in out):
        raise ValueError("'localization' needs exactly one of 'radius_file' or 'radius_m'")
    if out.get("radius_file"):
        out["radius_file"] = os.path.abspath(os.path.expanduser(os.path.expandvars(out["radius_file"])))
    else:
        r = np.atleast_1d(np.asarray(out["radius_m"], dtype=float))
        if r.size > 1 and len(out.get("depths_m", [])) != r.size:
            raise ValueError("localization.radius_m as a list needs 'depths_m' of the same length")
        if np.any(r <= 0):
            raise ValueError("localization.radius_m must be > 0")
    return out


def load_radii(spec):
    """(depths, radii) from the block: the file's table, the inline table, or one flat radius."""
    if spec.get("radius_file"):
        with open(spec["radius_file"], encoding="utf-8") as f:
            tab = json.load(f)
        return np.asarray(tab["depths_m"], dtype=float), np.asarray(tab["radius_m"], dtype=float)
    r = np.atleast_1d(np.asarray(spec["radius_m"], dtype=float))
    z = np.asarray(spec["depths_m"], dtype=float) if r.size > 1 else np.array([0.0])
    return z, r


def radius_at(depths, table_depths, table_radii):
    """Radius at each depth, interpolated in the table and held flat past its ends."""
    return np.interp(np.asarray(depths, dtype=float), table_depths, table_radii)


def gaspari_cohn(r):
    """Gaspari & Cohn (1999, eq. 4.10) at r = separation / half-support: 1 at 0, 0 from 2 on."""
    r = np.abs(np.asarray(r, dtype=float))
    g = np.zeros_like(r)
    near = r <= 1
    g[near] = (((-0.25 * r[near] + 0.5) * r[near] + 0.625) * r[near] - 5 / 3) * r[near] ** 2 + 1
    far = (r > 1) & (r < 2)
    g[far] = ((((r[far] / 12 - 0.5) * r[far] + 0.625) * r[far] + 5 / 3) * r[far] - 5) * r[far] \
        + 4 - 2 / (3 * r[far])
    return g


def taper(depths_a, depths_b, table_depths, table_radii):
    """rho[i, j] in [0, 1], the weight depth b[j] has on depth a[i]. The support is the mean of the
    two radii, so rho is symmetric when a and b are the same depths."""
    za = np.asarray(depths_a, dtype=float)[:, None]
    zb = np.asarray(depths_b, dtype=float)[None, :]
    support = (radius_at(depths_a, table_depths, table_radii)[:, None]
               + radius_at(depths_b, table_depths, table_radii)[None, :]) / 2
    return gaspari_cohn(2 * np.abs(za - zb) / support)


def summarize(state_depths, obs_depths, table_depths, table_radii):
    """One log line: how much of the column the observations can reach."""
    rho = taper(state_depths, obs_depths, table_depths, table_radii)
    reached = rho.max(axis=1) >= REACH_MIN
    z = np.asarray(state_depths, dtype=float)
    L = radius_at(z, table_depths, table_radii)
    if not reached.any():
        return "localization: no state cell is within reach of an observation"
    return (f"localization: radius {L.min():.1f}-{L.max():.1f} m; {reached.sum()}/{len(z)} cells "
            f"updated, deepest {z[reached].max():.1f} m of {z.max():.1f} m")

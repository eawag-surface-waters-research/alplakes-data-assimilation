"""The observation operator, numpy only so the OpenDA wrapper (one process per member per
analysis) uses exactly the native EnKF's operator without importing the engine."""
import numpy as np


def build_H(z_volume, lake_level, sim_depths):
    """One row of weights over the state cells per observation: linear interpolation between the
    two cells bracketing the observation, the nearest cell outside the grid. `sim_depths` are
    model column coordinates (heights relative to the surface, negative below it).

    Interpolating matters because Simstrat centres its cells at x.25/x.75 on a 0.5 m grid, so a
    round observation depth falls halfway between two cells. Rows sum to 1."""
    H   = np.zeros((len(sim_depths), len(z_volume)))
    z   = np.asarray(z_volume, dtype=float)
    asc = z[0] <= z[-1]
    zs  = z if asc else z[::-1]
    for row, d in enumerate(sim_depths):
        z_target = lake_level + d
        j = int(np.searchsorted(zs, z_target))
        if j == 0 or j >= len(zs):
            H[row, int(np.argmin(np.abs(z - z_target)))] = 1.0
            continue
        span = zs[j] - zs[j - 1]
        w    = (z_target - zs[j - 1]) / span if span > 0 else 0.0
        lo, hi = (j - 1, j) if asc else (len(zs) - j, len(zs) - 1 - (j - 1))
        H[row, lo] = 1.0 - w
        H[row, hi] = w
    return H

"""The observation operator: model state -> model equivalents at the observation depths.

numpy only, so the OpenDA wrapper -- one process per member per analysis -- can use exactly the
operator the native EnKF uses without importing the rest of the engine.
"""
import numpy as np

# The observation operator. "linear" is the DEFAULT because it is the one the rest of the pipeline
# already assumes: it is what Simstrat does when it writes T_out.dat, so it is what summarize.py
# scores against and what notebooks/generate_sigma.py fitted every sigma_obs on. "nearest" is the
# historical behaviour, kept to reproduce runs made before this existed.
OBS_OPERATOR_DEFAULT = "linear"
OBS_OPERATORS = ("nearest", "linear")


# inherits the surface-alignment assumption from read_snapshot_T (z_volume
# slicing). converts "X metres below the surface" into an actual height above the bottom
def build_H(z_volume, lake_level, sim_depths, operator=OBS_OPERATOR_DEFAULT):
    """Observation operator: rows of weights over the state cells, one row per observation.

    "linear" spreads each row over the two bracketing cells; "nearest" puts a single 1.0 on the
    closest one.

    Why it matters: Simstrat centres its cells at x.25/x.75 on a 0.5 m grid, so every ROUND
    observation depth (0.5, 1, 2, 10 ...) falls exactly HALFWAY between two cells. 
    It is here to reduce the discretisation error rather than eliminating it.

    Rows sum to 1 either way, so H stays a valid linear operator and PHT/HPHT are unaffected.
    Outside the grid, "linear" falls back to the nearest cell -- there is nothing to interpolate
    between."""
    H = np.zeros((len(sim_depths), len(z_volume)))
    z   = np.asarray(z_volume, dtype=float)
    asc = z[0] <= z[-1]
    zs  = z if asc else z[::-1]
    for row, d in enumerate(sim_depths):
        z_target = lake_level + d
        if operator == "nearest":
            H[row, int(np.argmin(np.abs(z - z_target)))] = 1.0   # closest cell to that target
            continue
        j = int(np.searchsorted(zs, z_target))
        if j == 0 or j >= len(zs):                       # outside the grid: nearest, as before
            H[row, int(np.argmin(np.abs(z - z_target)))] = 1.0
            continue
        span = zs[j] - zs[j - 1]
        w    = (z_target - zs[j - 1]) / span if span > 0 else 0.0
        lo, hi = (j - 1, j) if asc else (len(zs) - j, len(zs) - 1 - (j - 1))
        H[row, lo] = 1.0 - w
        H[row, hi] = w
    return H

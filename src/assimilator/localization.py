"""Vertical localization for the 1D column EnKF.

WHY. With N=20 members the ensemble spans 19 directions, and a correlation below ~2/sqrt(N) = 0.45
cannot be distinguished from sampling noise. Measured on the upperlugano innovations (364 analyses,
`enkf_innov_by_depth.csv`), **80-82% of off-diagonal forecast-error correlations are below that
floor** in both seasons -- so most of what the Kalman gain currently uses is noise. Worse, the
observations stop at 40 m while the column runs to ~288 m over 576 cells: **86% of the state is
below the deepest obs** and is updated only through those noise correlations. This is the mechanism
behind the deep-water degradation already on record (40 m innov-RMSE 0.084 -> 0.223 when inflation
amplified the same spurious surface-to-deep covariance).

RADIUS. Fitted to the data rather than guessed: for each depth, the separation at which the measured
innovation correlation first drops below 0.45.

    depth   0.5-7 m   11-19 m   25 m   35-40 m
    strat     2-4        4-6     10      10
    mixed      4         4-6      6      10

which `L(z) = max(L0, slope*z)` with L0=4 m, slope=0.25 tracks in BOTH seasons -- so no seasonal
switch is needed. Gaspari-Cohn is scaled so it reaches exactly zero at |dz| = L(z).

CONSEQUENCE, stated plainly: with this radius nothing below ~50 m receives an update at all. That is
the intended behaviour, not a side effect -- there is no observational information down there, so the
model should be trusted rather than nudged by noise. Expect the deep water to free-run.

REFINEMENT (not implemented). Geometric depth is the wrong metric physically: two cells inside one
mixed layer are dynamically identical however far apart, while two cells straddling a sharp
thermocline are decoupled however close. The natural distance is the integrated buoyancy frequency
between them, and Simstrat already outputs NN (`Settings.par` -> Output -> Variables). That would
make the radius adapt to stratification automatically instead of via the depth proxy above.
"""
import os
import json

import numpy as np

L0_DEFAULT    = 4.0    # m - floor on the localization radius (surface/thermocline)
SLOPE_DEFAULT = 0.25   # - radius growth with depth; L(40 m) = 10 m, matching the measured cutoff


def load_params(lake, root, cfg=None):
    """Radius knobs for one lake, in precedence order:

        1. explicit "localization_L0" / "localization_slope" in the run config
        2. localization/<lake>.json, fitted from that lake's observations by
           notebooks/localization_from_obs.py
        3. the module defaults above

    Deleting the json is the kill switch: the run falls back to the defaults. The defaults were
    hand-read off upperlugano and are NOT lake-neutral -- fitted slopes across the configured lakes
    range 0.21-0.47 -- so prefer a per-lake fit wherever observations exist.

    Returns (kwargs_for_radius, source_description).
    """
    cfg  = cfg or {}
    path = os.path.join(root, "localization", f"{lake}.json")

    L0, slope, src = L0_DEFAULT, SLOPE_DEFAULT, "built-in defaults (upperlugano hand-fit)"
    if os.path.isfile(path):
        with open(path, encoding="utf-8") as f:
            fitted = json.load(f)
        L0, slope = float(fitted["L0"]), float(fitted["slope"])
        src = f"localization/{lake}.json (fitted {fitted.get('fitted_on', '?')})"

    overridden = []
    if "localization_L0" in cfg:
        L0 = float(cfg["localization_L0"]);       overridden.append("L0")
    if "localization_slope" in cfg:
        slope = float(cfg["localization_slope"]); overridden.append("slope")
    if overridden:
        src += f"; {'/'.join(overridden)} overridden by run config"

    return {"L0": L0, "slope": slope}, src


def gaspari_cohn(r):
    """Gaspari-Cohn (1999) 5th-order piecewise-rational correlation function.

    r = |dz| / c with half-width c; support is r < 2, so the function is exactly zero at |dz| = 2c.
    Positive definite, so the Schur product below preserves positive-definiteness of HPHT
    (Schur product theorem) -- localization cannot make the analysis solve ill-posed.
    """
    r   = np.abs(np.asarray(r, dtype=float))
    out = np.zeros_like(r)

    m = r <= 1
    x = r[m]
    out[m] = ((((-0.25 * x + 0.5) * x + 0.625) * x - 5.0 / 3.0) * x ** 2) + 1.0

    m = (r > 1) & (r < 2)
    x = r[m]
    out[m] = (((((x / 12.0 - 0.5) * x + 0.625) * x + 5.0 / 3.0) * x - 5.0) * x
              + 4.0 - 2.0 / (3.0 * np.maximum(x, 1e-12)))
    return np.clip(out, 0.0, 1.0)


def radius(depth_m, L0=L0_DEFAULT, slope=SLOPE_DEFAULT):
    """Localization radius (m) at a given depth — the distance where the taper reaches zero."""
    return np.maximum(L0, slope * np.asarray(depth_m, dtype=float))


def localization_matrix(state_depths, obs_depths, L0=L0_DEFAULT, slope=SLOPE_DEFAULT):
    """rho[i, j] in [0, 1] for state cell i against observation j.

    The pair's radius uses the MEAN of the two depths, which keeps rho symmetric where the two sets
    coincide (needed for the obs-obs matrix to stay positive definite).
    """
    zs = np.asarray(state_depths, dtype=float)[:, None]
    zo = np.asarray(obs_depths,   dtype=float)[None, :]
    L  = radius(0.5 * (zs + zo), L0, slope)          # symmetric in (zs, zo)
    return gaspari_cohn(np.abs(zs - zo) / (0.5 * L))  # half-width c = L/2 -> zero at |dz| = L


def build_matrices(state_depths, obs_depths, L0=L0_DEFAULT, slope=SLOPE_DEFAULT):
    """The (rho_state_obs, rho_obs_obs) pair the EnKF Schur-multiplies into PHT and HPHT.

    rho_obs_obs is built from the obs depths against THEMSELVES rather than from the model cells H
    picks out: those cells only approximate the obs depths, and that mismatch would make the matrix
    slightly asymmetric, forfeiting the positive-definiteness the Schur product theorem gives us.
    """
    return (localization_matrix(state_depths, obs_depths, L0, slope),
            localization_matrix(obs_depths,   obs_depths, L0, slope))


def summarize(state_depths, obs_depths, **kw):
    """Diagnostic line: how much of the state each analysis can actually touch."""
    rho     = localization_matrix(state_depths, obs_depths, **kw)
    reached = (rho > 0).any(axis=1)
    zs      = np.asarray(state_depths, dtype=float)
    return (f"localization: L={kw.get('L0', L0_DEFAULT):.0f}..{radius(zs.max(), **kw):.0f} m; "
            f"{reached.sum()}/{len(zs)} state cells updated "
            f"({reached.mean() * 100:.0f}%), deepest updated cell "
            f"{zs[reached].max():.1f} m of {zs.max():.1f} m; "
            f"mean nonzero weights per cell {np.mean((rho > 0).sum(axis=1)[reached]):.1f}/{len(obs_depths)}")

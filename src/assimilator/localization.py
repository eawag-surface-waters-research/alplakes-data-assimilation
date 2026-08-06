"""Vertical localization for the 1D column EnKF: a binary per-depth mask.

WHY. With N=20 members a correlation below ~2/sqrt(N) = 0.45 cannot be distinguished from sampling
noise, and most off-diagonal forecast-error correlations are below it. The observations also stop at
40 m while the column runs to ~288 m, so most of the state would otherwise be updated purely through
noise. Localization stops that.

WHAT. Each state cell carries a RADIUS: the separation at which its ensemble correlation falls below
a threshold, measured from a free ensemble run by notebooks/localization.py. A cell may be updated
by observations inside its radius and by nothing else:

    rho[i, j] = 1 if |z_state[i] - z_obs[j]| <= L(z_state[i])   else 0

Binary, not a Gaspari-Cohn taper, and the radius profile is measured per depth rather than fitted to
max(L0, slope*z). One threshold is the only knob.

PHT ONLY. A binary mask is not positive semi-definite -- a boxcar's transform changes sign -- so
Schur-multiplying it into HPHT would forfeit the guarantee that HPHT+R stays invertible. It is
therefore applied to the state-obs cross-covariance alone, which is where localization does its work:
deciding which state cells an observation may reach. HPHT is left intact.

CONSEQUENCE, stated plainly: nothing far below the deepest observation receives an update. That is
intended -- there is no observational information down there, so the model should be trusted rather
than nudged by noise. Expect the deep water to free-run.
"""
import os
import json

import numpy as np

RADIUS_DEFAULT = 5.0    # m - flat fallback when a lake has no measured table


def load_radii(lake, root):
    """(depths, radii, source) for one lake from localization/<lake>.json.

    Falls back to a flat RADIUS_DEFAULT when the file is absent, so deleting the json is the kill
    switch. The fallback is a placeholder, not a measurement -- run notebooks/localization.py.
    """
    path = os.path.join(root, "localization", f"{lake}.json")
    if not os.path.isfile(path):
        return None, None, f"flat {RADIUS_DEFAULT:g} m fallback (no localization/{lake}.json)"
    with open(path, encoding="utf-8") as f:
        tab = json.load(f)
    if "depths_m" not in tab:
        raise ValueError(
            f"localization/{lake}.json predates the per-depth radius table (it carries "
            f"{sorted(tab)[:4]}...). The fitted L0/slope form and its Gaspari-Cohn taper are gone; "
            f"re-derive with notebooks/localization.py against a free ensemble run.")
    z = np.asarray(tab["depths_m"], dtype=float)
    L = np.asarray(tab["radius_m"], dtype=float)
    return z, L, (f"localization/{lake}.json (threshold {tab.get('threshold', '?')}, "
                  f"derived {tab.get('derived_on', '?')})")


def radius_at(state_depths, table_depths=None, table_radii=None):
    """The radius in m at each state depth, interpolated from the measured table.

    Held flat beyond the table's ends rather than extrapolated: past the deepest measured depth the
    radius is not a measurement in any case, and a linear extrapolation could grow it without bound.
    """
    zs = np.asarray(state_depths, dtype=float)
    if table_depths is None:
        return np.full(zs.shape, RADIUS_DEFAULT)
    return np.interp(zs, table_depths, table_radii)


def localization_matrix(state_depths, obs_depths, table_depths=None, table_radii=None):
    """rho[i, j] in {0, 1}: may observation j update state cell i?

    The radius is the STATE cell's own -- "how far can information reach this cell" -- so the mask
    is read down the rows. It need not be symmetric, and does not have to be: this multiplies PHT
    only (see the module docstring).
    """
    zs = np.asarray(state_depths, dtype=float)[:, None]
    zo = np.asarray(obs_depths,   dtype=float)[None, :]
    L  = radius_at(state_depths, table_depths, table_radii)[:, None]
    return (np.abs(zs - zo) <= L).astype(float)


def summarize(state_depths, obs_depths, table_depths=None, table_radii=None):
    """Diagnostic line: how much of the state each analysis can actually touch."""
    rho     = localization_matrix(state_depths, obs_depths, table_depths, table_radii)
    reached = (rho > 0).any(axis=1)
    zs      = np.asarray(state_depths, dtype=float)
    L       = radius_at(zs, table_depths, table_radii)
    if not reached.any():
        return (f"localization: L={L.min():.1f}..{L.max():.1f} m; NO state cell is reachable by any "
                f"observation — every analysis would be a no-op")
    return (f"localization: L={L.min():.1f}..{L.max():.1f} m; "
            f"{reached.sum()}/{len(zs)} state cells updated ({reached.mean() * 100:.0f}%), "
            f"deepest updated cell {zs[reached].max():.1f} m of {zs.max():.1f} m; "
            f"mean observations per updated cell "
            f"{np.mean((rho > 0).sum(axis=1)[reached]):.1f}/{len(obs_depths)}")

"""Vertical localization for the 1D column EnKF: a Gaspari-Cohn taper.

WHY. With N=20 members a correlation below ~2/sqrt(N) = 0.45 is sampling noise, and observations
stop at 40 m while the column runs to ~288 m, so most of the state would be updated by noise alone.

WHAT. Each state cell carries a RADIUS measured by notebooks/localization.py. It is the taper's
SUPPORT: weight 1 at zero separation, falling smoothly to exactly 0 at the radius.

BOTH SIDES. The taper multiplies PHT and HPHT alike. Tapering only PHT leaves the inverse built for
the full observation set, so the two halves disagree about which observations exist and the dropped
ones re-enter through the dense inverse with the wrong sign (murten 2025-04-10: the 5 m cell gave
the 0.5 m observation -0.315 where the correct weight is +0.135; 46 of 51 cells gained variance).

NOT A BOXCAR. Consistency alone is not enough -- rho.P must stay a valid covariance, which needs
rho positive semi-definite (Schur product theorem). A binary mask is not: it claims A-B and B-C
correlate while A-C do not, and that shows up as a negative eigenvalue (-0.2569 on the same murten
state, vs -0.0009 for this taper). Applied to both sides a boxcar still made 4 of 51 cells worse.

CONSEQUENCE: nothing far below the deepest observation is updated. Expect the deep water to
free-run -- there is no information down there, so the model should be trusted rather than nudged.
"""
import os
import json

import numpy as np

RADIUS_DEFAULT = 5.0    # m - flat fallback when a lake has no measured table
REACH_MIN      = 0.01   # taper weight below which a cell counts as untouched by an observation


def load_radii(lake, root):
    """(depths, radii, source) from localization/<lake>.json; deleting the json is the kill switch,
    falling back to a flat RADIUS_DEFAULT placeholder."""
    path = os.path.join(root, "localization", f"{lake}.json")
    if not os.path.isfile(path):
        return None, None, f"flat {RADIUS_DEFAULT:g} m fallback (no localization/{lake}.json)"
    with open(path, encoding="utf-8") as f:
        tab = json.load(f)
    if "depths_m" not in tab:
        raise ValueError(
            f"localization/{lake}.json predates the per-depth radius table (it carries "
            f"{sorted(tab)[:4]}...). The fitted L0/slope form is gone; re-derive with "
            f"notebooks/localization.py against a free ensemble run.")
    z = np.asarray(tab["depths_m"], dtype=float)
    L = np.asarray(tab["radius_m"], dtype=float)
    return z, L, (f"localization/{lake}.json (threshold {tab.get('threshold', '?')}, "
                  f"derived {tab.get('derived_on', '?')})")


def radius_at(state_depths, table_depths=None, table_radii=None):
    """Radius in m at each depth, interpolated from the table. Held flat past the table's ends
    rather than extrapolated, which could grow it without bound."""
    zs = np.asarray(state_depths, dtype=float)
    if table_depths is None:
        return np.full(zs.shape, RADIUS_DEFAULT)
    return np.interp(zs, table_depths, table_radii)


def gaspari_cohn(r):
    """Gaspari-Cohn (1999, eq. 4.10) for r = separation / c: 1 at 0, 0 beyond 2, PSD in between --
    the property the whole scheme rests on. Callers pass r = 2 * separation / support."""
    r = np.abs(np.asarray(r, dtype=float))
    g = np.zeros_like(r)
    near = r <= 1
    g[near] = (((-0.25 * r[near] + 0.5) * r[near] + 0.625) * r[near] - 5 / 3) * r[near] ** 2 + 1
    far = (r > 1) & (r < 2)
    g[far] = ((((r[far] / 12 - 0.5) * r[far] + 0.625) * r[far] + 5 / 3) * r[far] - 5) * r[far] \
        + 4 - 2 / (3 * r[far])
    return g


def taper(depths_a, depths_b, table_depths=None, table_radii=None):
    """rho[i, j] in [0, 1]: how much may depth b[j] influence depth a[i]?

    Takes two arbitrary depth axes because it is used twice per analysis, (state, obs) for PHT and
    (obs, obs) for HPHT. The support is the MEAN of the two radii, which keeps rho symmetric; the
    binary mask's per-row radius has no meaning where neither index is a state cell.
    """
    za = np.asarray(depths_a, dtype=float)[:, None]
    zb = np.asarray(depths_b, dtype=float)[None, :]
    La = radius_at(depths_a, table_depths, table_radii)[:, None]
    Lb = radius_at(depths_b, table_depths, table_radii)[None, :]
    support = (La + Lb) / 2
    return gaspari_cohn(2 * np.abs(za - zb) / np.where(support > 0, support, np.inf))


def summarize(state_depths, obs_depths, table_depths=None, table_radii=None):
    """How much of the state an analysis can touch. Counted against REACH_MIN, not zero: a taper is
    nonzero almost everywhere inside its support, so `> 0` would report 100% for every lake."""
    rho     = taper(state_depths, obs_depths, table_depths, table_radii)
    reached = rho.max(axis=1) >= REACH_MIN
    zs      = np.asarray(state_depths, dtype=float)
    L       = radius_at(zs, table_depths, table_radii)
    if not reached.any():
        return (f"localization: L={L.min():.1f}..{L.max():.1f} m; NO state cell is reachable by "
                f"any observation — every analysis would be a no-op")
    return (f"localization: L={L.min():.1f}..{L.max():.1f} m (Gaspari-Cohn support); "
            f"{reached.sum()}/{len(zs)} state cells updated ({reached.mean() * 100:.0f}%), "
            f"deepest updated cell {zs[reached].max():.1f} m of {zs.max():.1f} m; "
            f"mean effective observations per updated cell "
            f"{rho.sum(axis=1)[reached].mean():.1f}/{len(obs_depths)}")

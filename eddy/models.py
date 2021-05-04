import numpy as np
from . import deprojection as dp
import numba
import scipy.constants as sc


au = sc.au
G = sc.G
msun = 1.988e30

# -- Functions to build projected velocity profiles. -- #

@numba.njit
def proj_vkep(rvals, tvals, zvals, dist, mstar, inc):
    """Projected Keplerian rotational velocity profile."""
    rvals = rvals * au * dist
    zvals = zvals * au * dist
    v_phi = G * mstar * msun * np.power(rvals, 2)
    v_phi = np.sqrt(v_phi * np.power(np.hypot(rvals, zvals), -3))
    return proj_vphi(v_phi, tvals, inc)


@numba.njit
def proj_vpow(rvals, tvals, zvals, dist, mstar, inc, vp_q, vp_100, vp_qtaper, vp_rtaper):
    """Projected power-law rotational velocity profile."""
    v_phi = (rvals * dist / 100.)**vp_q
    v_phi *= np.exp(-(rvals / vp_rtaper)**vp_qtaper)
    v_phi = proj_vphi(vp_100 * v_phi, tvals, inc)
    v_rad = (rvals * dist / 100.)**vr_q
    v_rad = proj_vrad(vr_100 * v_rad, tvals, inc)
    return v_phi + v_rad


@numba.njit
def proj_vphi(v_phi, tvals, inc):
    """Project the rotational velocity."""
    return v_phi * np.cos(tvals) * abs(np.sin(np.radians(inc)))


@numba.njit
def proj_vrad(v_rad, tvals, inc):
    """Project the radial velocity."""
    return v_rad * np.sin(tvals) * abs(np.sin(np.radians(inc)))

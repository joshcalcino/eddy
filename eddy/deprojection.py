import numpy as np
import numba



def get_flared_coords(x0, y0, xaxis, yaxis, inc, PA, z0,
                    r_cavity, r_taper, psi, q_taper, w_r, w_i, w_t, niter):
    """Return cyclindrical coords of surface in [arcsec, rad, arcsec]."""
    x_mid, y_mid = get_midplane_cart_coords(x0, y0, inc, PA, xaxis, yaxis)
    # print('sum(xmid)', np.sum(x_mid))
    # print('sum(ymid)', np.sum(y_mid))
    return _get_flared_coords(x_mid, y_mid, inc, z0,
                        r_cavity, r_taper, psi, q_taper, w_r, w_i, w_t, niter)


@numba.njit
def _get_flared_coords(x_mid, y_mid, inc, z0,
                    r_cavity, r_taper, psi, q_taper, w_r, w_i, w_t, niter):
    r_tmp, t_tmp = np.hypot(x_mid, y_mid), np.arctan2(y_mid, x_mid)
    # print('sum(r_tmp)', np.sum(r_tmp))
    # print('sum(t_tmp)', np.sum(t_tmp))
    for _ in range(niter):
        z_tmp = z_func(r_tmp, z0, r_cavity, r_taper, psi, q_taper) +\
         w_func(r_tmp, t_tmp, r_cavity, w_r, w_i, w_t)
        print('params', z0, r_cavity, r_taper, psi, q_taper, w_r, w_i, w_t)
        print('sum(z_func)', z_func(r_tmp, z0, r_cavity, r_taper, psi, q_taper))
        print('sum(w_func)', w_func(r_tmp, t_tmp, r_cavity, w_r, w_i, w_t))
        y_tmp = y_mid + z_tmp * np.tan(np.radians(inc))
        r_tmp = np.hypot(y_tmp, x_mid)
        t_tmp = np.arctan2(y_tmp, x_mid)
    return r_tmp, t_tmp, z_func(r_tmp, z0, r_cavity, r_taper, psi, q_taper)


@numba.njit
def z_func(r_in, z0, r_cavity, r_taper, psi, q_taper):
    r = np.maximum(r_in - r_cavity, 0.0)
    z = z0 * np.power(r, psi) * np.exp(-np.power(r / r_taper, q_taper))
    return np.maximum(z, 0.0)


@numba.njit
def w_func(r_in, t, r_cavity, w_r, w_i, w_t):
    r = np.maximum(r_in - r_cavity, 0.0)
    warp = np.radians(w_i) * np.exp(-0.5 * (r / w_r)**2)
    return r * np.tan(warp * np.sin(t - np.radians(w_t)))


@numba.njit
def rotate_coords(x, y, PA):
    """Rotate (x, y) by PA [deg]."""
    x_rot = y * np.cos(np.radians(PA)) + x * np.sin(np.radians(PA))
    y_rot = x * np.cos(np.radians(PA)) - y * np.sin(np.radians(PA))
    return x_rot, y_rot


@numba.njit
def deproject_coords(x, y, inc):
    """Deproject (x, y) by inc [deg]."""
    return x, y / np.cos(np.radians(inc))


def get_cart_sky_coords(x0, y0, xaxis, yaxis):
    """Return cartesian sky coordinates in [arcsec, arcsec]."""
    return np.meshgrid(xaxis - x0, yaxis - y0)


def get_midplane_cart_coords(x0, y0, inc, PA, xaxis, yaxis):
    """Return cartesian coordaintes of midplane in [arcsec, arcsec]."""
    x_sky, y_sky = get_cart_sky_coords(x0, y0, xaxis, yaxis)
    x_rot, y_rot = rotate_coords(x_sky, y_sky, PA)
    return deproject_coords(x_rot, y_rot, inc)


@numba.njit
def get_midplane_polar_coords(x0, y0, inc, PA, xaxis, yaxis):
    """Return the polar coordinates of midplane in [arcsec, radians]."""
    x_mid, y_mid = get_midplane_cart_coords(x0, y0, inc, PA)
    return np.hypot(y_mid, x_mid), np.arctan2(y_mid, x_mid)


@numba.njit
def get_r_t(x_disk, y_disk):
    r_disk = np.hypot(x_disk, y_disk)
    t_disk = np.arctan2(y_disk, x_disk)
    return r_disk, t_disk


def get_diskframe_coords(xaxis, yaxis, nxpix, nypix, extend=2.0, oversample=0.5):
    """Disk-frame coordinates based on the cube axes."""
    x_disk = np.linspace(extend * xaxis[0], extend * xaxis[-1],
                         int(nxpix * oversample))[::-1]
    y_disk = np.linspace(extend * yaxis[0], extend * yaxis[-1],
                         int(nypix * oversample))
    x_disk, y_disk = np.meshgrid(x_disk, y_disk)
    r_disk, t_disk = get_r_t(x_disk, y_disk)
    return x_disk, y_disk, r_disk, t_disk

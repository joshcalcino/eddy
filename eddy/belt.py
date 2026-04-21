# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


__all__ = ['belt']


class belt(object):
    """
    A class containing a belt of velocity with their associated radial positions.
    Based heavily on annulus object but vastly less developed.

    Args:
        spectra (ndarray): Array of shape ``[N, M]`` of spectra to shift and
            fit, where ``N`` is the number of spectra and ``M`` is the length
            of the velocity axis.
        rvals (ndarray): Radial values in [arcsec] of each of the spectra.
        velax (ndarray): Velocity axis in [m/s] of the spectra.
        inc (float): Inclination of the disk in [deg]. A positive inclination
            specifies a clockwise rotating disk.
        remove_empty (optional[bool]): Remove empty spectra.
        sort_spectra (optional[bool]): Sorted the spectra into increasing
            ``theta``.
    """

    def __init__(self, vvals, rvals, pvals, inc,
                 remove_empty=True, sort_spectra=True):

        # Read in the spectra and estimate the RMS.

        self.r = rvals
        self.data = vvals
        self.pvals = pvals
        self.inc = inc
        # if self.inc == 0.0:
        #     raise ValueError("Disk inclination must be non-zero.")
        self.inc_rad = np.radians(self.inc)
        self.rotation = 'clockwise' if self.inc > 0 else 'anticlockwise'

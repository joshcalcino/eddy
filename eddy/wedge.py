# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt


__all__ = ['wedge']


class wedge(object):
    """
    A class containing a wedge of spectra with their associated radial positions.
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

    def __init__(self, spectra, rvals, velax, inc, phi_min, phi_max, mask,
                 remove_empty=True, sort_spectra=True):

        # Read in the spectra and estimate the RMS.

        self.r = rvals
        self.spectra = spectra
        self.mask = mask
        self.phi_min = phi_min
        self.phi_max = phi_max
        self.inc = inc
        if self.inc == 0.0:
            raise ValueError("Disk inclination must be non-zero.")
        self.inc_rad = np.radians(self.inc)
        self.rotation = 'clockwise' if self.inc > 0 else 'anticlockwise'
        # self.rms = self._estimate_RMS()

        # Sort the spectra with increasing radius.
        #
        # if sort_spectra:
        #     idxs = np.argsort(self.r)
        #     self.spectra = self.spectra[idxs]
        #     self.r = self.r[idxs]

        # Remove empty pixels.

        # if remove_empty:
        #     idxa = np.sum(self.spectra, axis=-1) != 0.0
        #     idxb = np.std(self.spectra, axis=-1) != 0.0
        #     idxs = idxa & idxb
        #     self.theta = self.theta[idxs]
        #     self.spectra = self.spectra[idxs]
        # if self.theta.size < 1:
        #     raise ValueError("No finite spectra. Check for NaNs.")

        # Easier to use variables.

        # self.theta_deg = np.degrees(self.theta)
        self.spectra_flat = self.spectra.flatten()

        # Velocity axis.

        self.velax = velax
        self.chan = np.diff(velax)[0]
        self.velax_range = (self.velax[0] - 0.5 * self.chan,
                            self.velax[-1] + 0.5 * self.chan)
        self.velax_mask = np.array([self.velax[0], self.velax[-1]])

        # Check the shapes are compatible.

        if self.spectra.shape[0] != self.r.size:
            raise ValueError("Mismatch in number of radial values and spectra.")
        if self.spectra.shape[1] != self.velax.size:
            raise ValueError("Mismatch in the spectra and velocity axis.")



    # -- Plotting Functions -- #

    def plot_mountain(self, plot_kwargs=None, return_fig=False):
        """
        Make a "mountain" plot, showing how the spectra change over radius.
        Analogous to "river" plot. Just PV diagram really.
        """

        # Imports.

        from matplotlib.ticker import MultipleLocator
        from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

        # Deproject and grid the spectra.

        # if vrot is None:
        spectra = self.spectra
        # else:
            # spectra = self.deprojected_spectra(vrot=vrot, vrad=vrad)
        # spectra = self._grid_river(spectra, method=method)

        # Define the min and max for plotting.

        kw = {} if plot_kwargs is None else plot_kwargs
        # xlim = kw.pop('xlim', None)
        kw['vmax'] = kw.pop('vmax', np.nanmax(abs(spectra)))
        kw['vmin'] = kw.pop('vmin', 0.)
        kw['cmap'] = kw.pop('cmap', 'turbo')

        # Plot the data.

        fig, ax = plt.subplots(figsize=(6.0, 2.25), constrained_layout=True)
        ax_divider = make_axes_locatable(ax)
        im = ax.pcolormesh(self.r, self.velax,
                           spectra.T, **kw)
        # ax.set_ylim(-180, 180)
        # ax.yaxis.set_major_locator(MultipleLocator(60.0))
        ax.set_xlim(np.min(self.r), np.max(self.r))
        ax.set_ylabel('Velocity (m/s)')
        ax.set_xlabel(r'radius' + ' (arcsec)')
        ax.text(0.94, 0.96,
                        r'$\phi_{\rm min}, \phi_{\rm max}=$%.0f, %.0f'%(self.phi_min, self.phi_max),
                        color='0.9', ha='right', va='top',
                        transform=ax.transAxes)

        # Add the colorbar.

        cb_ax = ax_divider.append_axes('right', size='2%', pad='1%')
        cb = plt.colorbar(im, cax=cb_ax)
        # if residual:
        #     cb.set_label('Residual (mJy/beam)', rotation=270, labelpad=13)
        # else:
        cb.set_label('Intensity (Jy/beam)', rotation=270, labelpad=13)

        if return_fig:
            return fig


    def plot_mountain_keplerian_mask(self, mask=None, plot_kwargs=None, return_fig=False):
        """
        Like "plot_mountain()", but overlays contours from a second wedge
        instance, which is passed via the "mask" argument.
        """

        # Imports.

        from matplotlib.ticker import MultipleLocator
        from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable

        # Deproject and grid the spectra.

        # if vrot is None:
        spectra = self.spectra
        # else:
            # spectra = self.deprojected_spectra(vrot=vrot, vrad=vrad)
        # spectra = self._grid_river(spectra, method=method)

        # Define the min and max for plotting.

        kw = {} if plot_kwargs is None else plot_kwargs
        # xlim = kw.pop('xlim', None)
        kw['vmax'] = kw.pop('vmax', np.nanmax(abs(spectra)))
        kw['vmin'] = kw.pop('vmin', 0.)
        kw['cmap'] = kw.pop('cmap', 'turbo')

        # Plot the data.

        fig, ax = plt.subplots(figsize=(6.0, 2.25), constrained_layout=True)
        ax_divider = make_axes_locatable(ax)
        im = ax.pcolormesh(self.r, self.velax,
                           spectra.T, **kw)
        ax.contour(mask.r, mask.velax, mask.spectra.T, [0.5], colors='w', linewidths=0.25)
        # ax.set_ylim(-180, 180)
        # ax.yaxis.set_major_locator(MultipleLocator(60.0))
        ax.set_xlim(np.min(self.r), np.max(self.r))
        ax.set_ylabel('Velocity (m/s)')
        ax.set_xlabel(r'radius' + ' (arcsec)')
        ax.text(0.94, 0.96,
                        r'$\phi_{\rm min}, \phi_{\rm max}=$%.0f, %.0f'%(self.phi_min, self.phi_max),
                        color='0.9', ha='right', va='top',
                        transform=ax.transAxes)

        # Add the colorbar.

        cb_ax = ax_divider.append_axes('right', size='2%', pad='1%')
        cb = plt.colorbar(im, cax=cb_ax)
        # if residual:
        #     cb.set_label('Residual (mJy/beam)', rotation=270, labelpad=13)
        # else:
        cb.set_label('Intensity (Jy/beam)', rotation=270, labelpad=13)

        if return_fig:
            return fig

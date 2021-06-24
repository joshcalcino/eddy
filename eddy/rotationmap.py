# -*- coding: utf-8 -*-

import os
import time
import emcee
import pickle
import numpy as np
from astropy.io import fits
from astropy.convolution import (convolve, convolve_fft, Gaussian2DKernel, Gaussian1DKernel
)
import warnings
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.interpolate import griddata
import scipy.constants as sc
from scipy.ndimage import shift, rotate
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import corner
from matplotlib.patches import Ellipse
from matplotlib.ticker import MaxNLocator
from . import deprojection as dp
from . import models as mod
import scipy.constants as sc
from .datacube import datacube
try:
    import zeus
except ImportError:
    no_zeus = True


warnings.filterwarnings("ignore")


class rotationmap(datacube):
    """
    Read in the velocity maps and initialize the class. To make the fitting
    quicker, we can clip the cube to a smaller region, or downsample the
    image to get a quicker initial fit.

    Args:
        path (str): Relative path to the rotation map you want to fit.
        FOV (Optional[float]): If specified, clip the data down to a
            square field of view with sides of `FOV` [arcsec].
        uncertainty (Optional[str]): Relative path to the map of v0
            uncertainties. Must be a FITS file with the same shape as the
            data. If nothing is specified, will tried to find an uncerainty
            file following the ``bettermoments`` format. If this is not found,
            it will assume a 10% on all pixels.
        downsample (Optional[int]): Downsample the image by this factor for
            quicker fitting. For example, using ``downsample=4`` on an image
            which is 1000 x 1000 pixels will reduce it to 250 x 250 pixels.
            If you use ``downsample='beam'`` it will sample roughly
            spatially independent pixels using the beam major axis as the
            spacing.
    """

    priors = {}

    def __init__(self, path, FOV=None, uncertainty=None, downsample=None):
        datacube.__init__(self, path=path, FOV=FOV, fill=None)
        self._readuncertainty(uncertainty=uncertainty, FOV=self.FOV)
        if downsample is not None:
            self._downsample_cube(downsample)
        self.mask = np.isfinite(self.data)
        self.vlsr = np.nanmedian(self.data)
        self.vlsr_kms = self.vlsr / 1e3
        self._set_default_priors()

    # -- FITTING FUNCTIONS -- #

    def fit_map(self, p0, params, r_min=None, r_max=None, optimize=True,
                nwalkers=None, nburnin=300, nsteps=100, scatter=1e-3,
                plots=None, returns=None, pool=None, mcmc='emcee',
                mcmc_kwargs=None, niter=1, savefigs=False, savemodels=False):
        """
        Fit a rotation profile to the data. Note that for a disk with
        a non-zero height, the sign of the inclination dictates the direction
        of rotation: a positive inclination denotes a clockwise rotation, while
        a negative inclination denotes an anti-clockwise rotation.

        The function must be provided with a dictionary of parameters,
        ``params`` where key is the parameter and its value is either the fixed
        value this will take,

            ``params['PA'] = 45.0``

        would fix position angle to 45 degrees. Or, if an integeter, the index
        in the starting positions, ``p0``, such that,

            ``params['mstar'] = 0``

        would mean that the first value in ``p0`` is the guess for ``'mstar'``.

        For a list of the geometrical parameters, included flared emission
        surfaces or warps, see :func:`disk_coords`. In addition to these
        geometrical properties ``params`` also requires ``'mstar'``,
        ``'vlsr'``, ``'dist'``.

        To include a spatial convolution of the model rotation map you may also
        set ``params['beam']=True``, however this is only really necessary for
        low spatial resolution data, or for rotation maps made via the
        intensity-weighted average velocity.

        .. _Rosenfeld et al. (2013): https://ui.adsabs.harvard.edu/abs/2013ApJ...774...16R/abstract

        Args:
            p0 (list): List of the free parameters to fit.
            params (dictionary): Dictionary of the model parameters.
            r_min (optional[float]): Inner radius to fit in [arcsec].
            r_max (optional[float]): Outer radius to fit in [arcsec].
            optimize (optional[bool]): Use ``scipy.optimize`` to find the
                ``p0`` values which maximize the likelihood. Better results
                will likely be found. Note that for the masking the default
                ``p0`` and ``params`` values are used for the deprojection, or
                those foundvfrom the optimization. If this results in a poor
                initial mask, try with ``optimise=False`` or with a ``niter``
                value larger than 1.
            nwalkers (optional[int]): Number of walkers to use for the MCMC.
            nburnin (optional[int]): Number of steps to discard for burn-in.
            nsteps (optional[int]): Number of steps to use to sample the
                posterior distributions.
            scatter (optional[float]): Scatter used in distributing walker
                starting positions around the initial ``p0`` values.
            plots (optional[list]): List of the diagnostic plots to make. This
                can include ``'mask'``, ``'walkers'``, ``'corner'``,
                ``'bestfit'``, ``'residual'``, or ``'none'`` if no plots are to
                be plotted. By default, all are plotted.
            returns (optional[list]): List of items to return. Can contain
                ``'samples'``, ``'sampler'``, ``'percentiles'``, ``'dict'``,
                ``'model'``, ``'residuals'`` or ``'none'``. By default only
                ``'percentiles'`` are returned.
            pool (optional): An object with a `map` method.
            mcmc_kwargs (Optional[dict]): Dictionary to pass to the MCMC
                ``EnsembleSampler``.
            niter (optional[int]): Number of iterations to perform using the
                median PDF values from the previous MCMC run as starting
                positions. This is probably only useful if you have no idea
                about the starting positions for the emission surface or if you
                want to remove walkers stuck in local minima.

        Returns:
            to_return (list): Depending on the returns list provided.
                ``'samples'`` will be all the samples of the posteriors (with
                the burn-in periods removed). ``'percentiles'`` will return the
                16th, 50th and 84th percentiles of each posterior functions.
                ``'dict'`` will return a dictionary of the median parameters
                which can be directly input to other functions.
        """

        # Check the dictionary. May need some more work.

        if r_min is not None:
            if 'r_min' in params.keys():
                print("Found `r_min` in `params`. Overwriting value.")
            params['r_min'] = r_min
        if r_max is not None:
            if 'r_max' in params.keys():
                print("Found `r_max` in `params`. Overwriting value.")
            params['r_max'] = r_max

        params_tmp = self.verify_params_dictionary(params.copy())

        # Generate the mask for fitting based on the params.

        p0 = np.squeeze(p0).astype(float)
        temp = rotationmap._populate_dictionary(p0, params_tmp)
        self.ivar = self._calc_ivar(temp)

        # Check what the parameters are.

        labels = rotationmap._get_labels(params_tmp)
        labels_raw = []
        for label in labels:
            label_raw = label.replace('$', '').replace('{', '')
            label_raw = label_raw.replace(r'\rm ', '').replace('}', '')
            labels_raw += [label_raw]
        if len(labels) != len(p0):
            raise ValueError("Mismatch in labels and p0. Check for integers.")
        print("Assuming:\n\tp0 = [%s]." % (', '.join(labels_raw)))

        # Run an initial optimization using scipy.minimize. Recalculate the
        # inverse variance mask.

        if optimize:
            p0 = self._optimize_p0(p0, params_tmp)

        # Set up and run the MCMC with emcee.
        time.sleep(0.5)
        nsteps = np.atleast_1d(nsteps)
        nburnin = np.atleast_1d(nburnin)
        nwalkers = np.atleast_1d(nwalkers)

        mcmc_kwargs = {} if mcmc_kwargs is None else mcmc_kwargs
        mcmc_kwargs['scatter'], mcmc_kwargs['pool'] = scatter, pool

        for n in range(int(niter)):

            # Make the mask for fitting.

            temp = rotationmap._populate_dictionary(p0, params_tmp)
            temp = self.verify_params_dictionary(temp)
            self.ivar = self._calc_ivar(temp)

            # Run the sampler.

            sampler = self._run_mcmc(p0=p0, params=params_tmp,
                                     nwalkers=nwalkers[n % nwalkers.size],
                                     nburnin=nburnin[n % nburnin.size],
                                     nsteps=nsteps[n % nsteps.size],
                                     mcmc=mcmc, **mcmc_kwargs)

            if type(params_tmp['PA']) is int:
                sampler.chain[:, :, params_tmp['PA']] %= 360.0

            # Split off the samples.

            if mcmc == 'emcee':
                samples = sampler.chain[:, -int(nsteps[n % nsteps.size]):]
            else:
                samples = sampler.chain[-int(nsteps[n % nsteps.size]):]
            samples = samples.reshape(-1, samples.shape[-1])
            p0 = np.median(samples, axis=0)
            medians = rotationmap._populate_dictionary(p0, params.copy())
            medians = self.verify_params_dictionary(medians)

            # Get the max likelihood model
            idx = np.argmin(np.concatenate(sampler.lnprobability.T[nburnin:]))
            p0 = samples[idx]
            max_likelihood = rotationmap._populate_dictionary(p0, params)
            max_likelihood = self.verify_params_dictionary(max_likelihood)

        # Diagnostic plots.
        if savefigs is not None:
            save_name = self.name
        else:
            save_name = None

        if plots is None:
            plots = ['walkers', 'corner', 'bestfit', 'residual']
        plots = np.atleast_1d(plots)
        if 'none' in plots:
            plots = []
        if 'walkers' in plots:
            if mcmc == 'emcee':
                walkers = sampler.chain.T
            else:
                walkers = np.rollaxis(sampler.chain.copy(), 2)
            rotationmap._plot_walkers(walkers, nburnin[-1], labels)
        if 'corner' in plots:
            rotationmap._plot_corner(samples, labels, save_name=save_name)
        if 'bestfit' in plots:
            self.plot_model(samples=samples,
                            params=params,
                            mask=self.ivar,
                            save_name=save_name)
        if 'residual' in plots:
            self.plot_model_residual(samples=samples,
                                     params=params,
                                     mask=self.ivar,
                                     save_name=save_name)


        if savemodels:
            data = np.squeeze(fits.getdata(self.path))
            medians_mod = medians.copy()

            medians_mod['xaxis'] = self._read_position_axis(a=1)
            medians_mod['yaxis'] = self._read_position_axis(a=2)

            median_model = self.evaluate_model(medians_mod, coords_only=False)
            median_model_hdu = fits.PrimaryHDU(median_model)

            header = self.header
            for param in medians:
                header[param] = medians[param]

            median_model_hdu.header = header
            median_model_hdu.writeto('{}_median_fit.fits'.format(self.name), overwrite=True)

            residuals = data - median_model
            residuals_hdu = fits.PrimaryHDU(residuals)

            residuals_hdu.header = header
            residuals_hdu.writeto('{}_residuals_fit.fits'.format(self.name), overwrite=True)

            # Save the params dictionary
            pickle.dump(medians_mod, open("{}_median_fit_params.pkl".format(self.name), 'wb'))

        # Generate the output.

        to_return = []

        if returns is None:
            returns = ['samples']
        returns = np.atleast_1d(returns)

        if 'none' in returns:
            return None
        if 'samples' in returns:
            to_return += [samples]
        if 'sampler' in returns:
            to_return += [sampler]
        if 'lnprob' in returns:
            to_return += [sampler.lnprobability[nburnin:]]
        if 'percentiles' in returns:
            to_return += [np.percentile(samples, [16, 50, 84], axis=0)]

        if 'medians' in returns:
            to_return.append(medians)
        if 'maxl' in returns:
            to_return.append(max_likelihood)
        if 'dict' in returns:
            to_return += [medians]
        if 'model' in returns or 'residual' in returns:
            model = self.evaluate_models(samples, params)
            if 'model' in returns:
                to_return += [model]
            if 'residual' in returns:
                to_return += [self.data * 1e3 - model]

        return to_return if len(to_return) > 1 else to_return[0]

    def set_prior(self, param, args, type='flat'):
        """
        Set the prior for the given parameter. There are two types of priors
        currently usable, ``'flat'`` which requires ``args=[min, max]`` while
        for ``'gaussian'`` you need to specify ``args=[mu, sig]``.

        Args:
            param (str): Name of the parameter.
            args (list): Values to use depending on the type of prior.
            type (optional[str]): Type of prior to use.
        """
        type = type.lower()
        if type not in ['flat', 'gaussian']:
            raise ValueError("type must be 'flat' or 'gaussian'.")
        if type == 'flat':
            def prior(p):
                if not min(args) <= p <= max(args):
                   return -np.inf
                return np.log(1.0 / (args[1] - args[0]))
        else:
            def prior(p):
                lnp = np.exp(-0.5 * ((args[0] - p) / args[1])**2)
                return np.log(lnp / np.sqrt(2. * np.pi) / args[1])
        rotationmap.priors[param] = prior

    def disk_coords(self, x0=0.0, y0=0.0, xaxis=None, yaxis=None, inc=0.0, PA=0.0, z0=0.0, psi=0.0,
                    r_cavity=0.0, r_taper=None, q_taper=None, w_i=0.0, w_r=1.0,
                    w_t=0.0, frame='cylindrical', **_):
        r"""
        Get the disk coordinates given certain geometrical parameters and an
        emission surface. The emission surface is most simply described as a
        power law profile,

        .. math::

            z(r) = z_0 \times \left(\frac{r}{1^{\prime\prime}}\right)^{\psi}

        where ``z0`` and ``psi`` can be provided by the user. With the increase
        in spatial resolution afforded by interferometers such as ALMA there
        are a couple of modifications that can be used to provide a better
        match to the data.

        An inner cavity can be included with the ``r_cavity`` argument which
        makes the transformation:

        .. math::

            \tilde{r} = {\rm max}(0, r - r_{\rm cavity})

        Note that the inclusion of a cavity will mean that other parameters,
        such as ``z0``, would need to change as the radial axis has effectively
        been shifted.

        To account for the drop in emission surface in the outer disk where the
        gas surface density decreases there are two descriptions. The preferred
        way is to include an exponential taper to the power law profile,

        .. math::

            z_{\rm tapered}(r) = z(r) \times \exp\left( -\left[
            \frac{r}{r_{\rm taper}} \right]^{q_{\rm taper}} \right)

        where both ``r_taper`` and ``q_taper`` values must be set.

        We can also include a warp which is parameterized by,

        .. math::

            z_{\rm warp}(r,\, t) = r \times \tan \left(w_i \times \exp\left(-
            \frac{r^2}{2 w_r^2} \right) \times \sin(t - w_t)\right)

        where ``w_i`` is the inclination in [radians] describing the warp at
        the disk center. The width of the warp is given by ``w_r`` [arcsec] and
        ``w_t`` in [radians] is the angle of nodes (where the warp is zero),
        relative to the position angle of the disk, measured east of north.

        .. WARNING::

            The use of warps is largely untested. Use with caution!
            If you are using a warp, increase the number of iterations
            for the inference through ``self.disk_coords_niter`` (by default at
            5). For high inclinations, also set ``shadowed=True``.

        Args:
            x0 (optional[float]): Source right ascension offset [arcsec].
            y0 (optional[float]): Source declination offset [arcsec].
            inc (optional[float]): Source inclination [degrees].
            PA (optional[float]): Source position angle [degrees]. Measured
                between north and the red-shifted semi-major axis in an
                easterly direction.
            z0 (optional[float]): Aspect ratio at 1" for the emission surface.
                To get the far side of the disk, make this number negative.
            psi (optional[float]): Flaring angle for the emission surface.
            z1 (optional[float]): Correction term for ``z0``.
            phi (optional[float]): Flaring angle correction term for the
                emission surface.
            r_cavity (optional[float]): Outer radius of a cavity. Within this
                region the emission surface is taken to be zero.
            w_i (optional[float]): Warp inclination in [degrees] at the disk
                center.
            w_r (optional[float]): Scale radius of the warp in [arcsec].
            w_t (optional[float]): Angle of nodes of the warp in [degrees].
            frame (optional[str]): Frame of reference for the returned
                coordinates. Either ``'cartesian'`` or ``'cylindrical'``.

        Returns:
            array, array, array: Three coordinate arrays with ``(r, phi, z)``,
            in units of [arcsec], [radians], [arcsec], if
            ``frame='cylindrical'`` or ``(x, y, z)``, all in units of [arcsec]
            if ``frame='cartesian'``.
        """

        # Check the input variables.

        frame = frame.lower()
        if frame not in ['cylindrical', 'cartesian']:
            raise ValueError("frame must be 'cylindrical' or 'cartesian'.")

        # Get the x and y-axis
        if type(xaxis) == type(None):
            xaxis = self.xaxis
        if type(yaxis) == type(None):
            yaxis = self.yaxis

        # Apply the inclination concention.

        inc = inc if inc < 90.0 else inc - 180.0

        # Check that the necessary pairs are provided.

        msg = "Must specify either both or neither of `{}` and `{}`."
        if (r_taper is not None) != (q_taper is not None):
            raise ValueError(msg.format('r_taper', 'q_taper'))

        # Set the defaults.

        z0 = 0.0 if z0 is None else z0
        psi = 1.0 if psi is None else psi
        r_cavity = 0.0 if r_cavity is None else r_cavity
        r_taper = np.inf if r_taper is None else r_taper
        q_taper = 1.0 if q_taper is None else q_taper


        # Calculate the pixel values.

        if self.shadowed:
            coords = self.get_shadowed_coords(x0, y0, inc, PA, z_func, w_func)
        else:
            coords = dp.get_flared_coords(x0, y0, xaxis, yaxis, inc, PA, z0,
                                 r_cavity, r_taper, psi, q_taper, w_r, w_i, w_t, self.disk_coords_niter)
        if frame == 'cylindrical':
            return coords
        r, t, z = coords
        return r * np.cos(t), r * np.sin(t), z

    def plot_data(self, vmin=None, vmax=None, ivar=None, return_fig=False, save_name=None):
        """
        Plot the first moment map. By default will clip the velocity contours
        such that the velocities around the systemic velocity are highlighted.

        Args:
            levels (optional[list]): List of contour levels to use.
            ivar (optional[ndarray]): Inverse variances for each pixel. Will
                draw a solid contour around the regions with finite ``ivar``
                values and fill regions not considered.
            return_fig (optional[bool]): Return the figure.
            save_name (optional[string]): Name of the figure.
        Returns:
            fig (Matplotlib figure): If ``return_fig`` is ``True``. Can access
                the axes through ``fig.axes`` for additional plotting.
        """
        fig, ax = plt.subplots()

        vmin = np.nanpercentile(self.data, [2]) if vmin is None else vmin
        vmax = np.nanpercentile(self.data, [98]) if vmax is None else vmax
        vmax = max(abs(vmin - self.vlsr), abs(vmax - self.vlsr))
        im = ax.imshow(self.data / 1e3, origin='lower', extent=self.extent,
                       vmin=(self.vlsr-vmax)/1e3, vmax=(self.vlsr+vmax)/1e3,
                       cmap=rotationmap.cmap(), zorder=-9)
        cb = plt.colorbar(im, pad=0.03, extend='both', format='%.2f')
        cb.minorticks_on()
        cb.set_label(r'${\rm v_{0} \quad (km\,s^{-1})}$',
                     rotation=270, labelpad=15)
        if ivar is not None:
            ax.contour(self.xaxis, self.yaxis, ivar,
                       [0.0], colors='k')
            ax.contourf(self.xaxis, self.yaxis, ivar,
                        [-1.0, 0.0], colors='k', alpha=0.5)
        self._gentrify_plot(ax)

        if save_name is not None:
            plt.savefig('{0}_M1.png'.format(save_name), dpi=300)

        if return_fig:
            return fig

    # -- MCMC Functions -- #

    def _optimize_p0(self, theta, params, **kwargs):
        """Optimize the initial starting positions."""

        # TODO: implement bounds.
        # TODO: cycle through parameters one at a time for a better fit.

        # Negative log-likelihood function.
        def nlnL(theta):
            return -self._ln_probability(theta, params)

        method = kwargs.pop('method', 'TNC')
        options = kwargs.pop('options', {})
        options['maxiter'] = options.pop('maxiter', 10000)
        options['maxfun'] = options.pop('maxfun', 10000)
        options['ftol'] = options.pop('ftol', 1e-3)

        res = minimize(nlnL, x0=theta, method=method, options=options)
        if res.success:
            theta = res.x
            print("Optimized starting positions:")
        else:
            print("WARNING: scipy.optimize did not converge.")
            print("Starting positions:")
        print('\tp0 =', ['%.2e' % t for t in theta])
        return theta

    def _run_mcmc(self, p0, params, nwalkers, nburnin, nsteps, mcmc, **kwargs):
        """Run the MCMC sampling. Returns the sampler."""

        if mcmc == 'zeus':
            EnsembleSampler = zeus.EnsembleSampler
        else:
            EnsembleSampler = emcee.EnsembleSampler

        p0 = self._random_p0(p0, kwargs.pop('scatter', 1e-3), nwalkers)
        moves = kwargs.pop('moves', None)
        pool = kwargs.pop('pool', None)

        sampler = EnsembleSampler(nwalkers,
                                  p0.shape[1],
                                  self._ln_probability,
                                  args=[params, np.nan],
                                  moves=moves,
                                  pool=pool)

        progress = kwargs.pop('progress', True)

        sampler.run_mcmc(p0, nburnin + nsteps, progress=progress, **kwargs)

        return sampler

    @staticmethod
    def _random_p0(p0, scatter, nwalkers):
        """Get the starting positions."""
        p0 = np.squeeze(p0)
        dp0 = np.random.randn(nwalkers * len(p0)).reshape(nwalkers, len(p0))
        dp0 = np.where(p0 == 0.0, 1.0, p0)[None, :] * (1.0 + scatter * dp0)
        return np.where(p0[None, :] == 0.0, dp0 - 1.0, dp0)

    def _ln_likelihood(self, params):
        """Log-likelihood function. Simple chi-squared likelihood."""
        model = self._make_model(params)
        lnx2 = np.where(self.mask, np.power((self.data - model), 2), 0.0)
        # print('lnx2',lnx2)
        # print('sum(lnx2)', np.sum(lnx2))
        lnx2 = -0.5 * np.sum(lnx2) # * self.ivar)
        return lnx2 if np.isfinite(lnx2) else -np.inf

    def _ln_probability(self, theta, *params_in):
        """Log-probablility function."""
        model = rotationmap._populate_dictionary(theta, params_in[0])
        lnp = self._ln_prior(model)
        # print('lnp', lnp)
        if np.isfinite(lnp):
            return lnp + self._ln_likelihood(model)
        return -np.inf

    def _set_default_priors(self):
        """Set the default priors."""

        # Basic Geometry.

        self.set_prior('x0', [-0.5, 0.5], 'flat')
        self.set_prior('y0', [-0.5, 0.5], 'flat')
        self.set_prior('inc', [-90.0, 90.0], 'flat')
        self.set_prior('PA', [-360.0, 360.0], 'flat')
        self.set_prior('mstar', [0.1, 5.0], 'flat')
        self.set_prior('vlsr', [np.nanmin(self.data),
                                np.nanmax(self.data)], 'flat')

        # Emission Surface

        self.set_prior('z0', [0.0, 5.0], 'flat')
        self.set_prior('psi', [0.0, 5.0], 'flat')
        self.set_prior('r_cavity', [0.0, 1e30], 'flat')
        self.set_prior('r_taper', [0.0, 1e30], 'flat')
        self.set_prior('q_taper', [0.0, 5.0], 'flat')

        # Warp

        self.set_prior('w_i', [-90.0, 90.0], 'flat')
        self.set_prior('w_r', [0.0, self.xaxis.max()], 'flat')
        self.set_prior('w_t', [-180.0, 180.0], 'flat')

        # Velocity Profile

        self.set_prior('vp_100', [0.0, 1e4], 'flat')
        self.set_prior('vp_q', [-2.0, 0.0], 'flat')
        self.set_prior('vp_rtaper', [0.0, 1e30], 'flat')
        self.set_prior('vp_qtaper', [0.0, 5.0], 'flat')
        self.set_prior('vr_100', [-1e3, 1e3], 'flat')
        self.set_prior('vr_q', [-2.0, 2.0], 'flat')

    def _ln_prior(self, params):
        """Log-priors."""
        lnp = 0.0
        for key in params.keys():
            if key in rotationmap.priors.keys() and params[key] is not None:
                lnp += rotationmap.priors[key](params[key])
                # print('key', key, params[key])
                # print('lnp', lnp)
                if not np.isfinite(lnp):
                    return lnp
        return lnp

    def _calc_ivar(self, params):
        """Calculate the inverse variance including radius mask."""

        # Check the error array is the same shape as the data.

        try:
            assert self.error.shape == self.data.shape
        except AttributeError:
            self.error = self.error * np.ones(self.data.shape)

        # Deprojected coordinates.

        r, t = self.disk_coords(**params)[:2]
        t = abs(t) if params['abs_phi'] else t

        # Radial mask.

        mask_r = np.logical_and(r >= params['r_min'], r <= params['r_max'])
        mask_r = ~mask_r if params['exclude_r'] else mask_r

        # Azimuthal mask.

        mask_p = np.logical_and(t >= np.radians(params['phi_min']),
                                t <= np.radians(params['phi_max']))
        mask_p = ~mask_p if params['exclude_phi'] else mask_p

        # Finite value mask.

        mask_f = np.logical_and(np.isfinite(self.data), self.error > 0.0)

        # Velocity mask.

        v_min, v_max = params['v_min'], params['v_max']
        mask_v = np.logical_and(self.data >= v_min, self.data <= v_max)
        mask_v = ~mask_v if params['exclude_v'] else mask_v

        # Combine.

        mask = np.logical_and(np.logical_and(mask_v, mask_f),
                              np.logical_and(mask_r, mask_p))
        return np.where(mask, np.power(self.error, -2.0), 0.0)

    @staticmethod
    def _get_labels(params):
        """Return the labels of the parameters to fit."""
        idxs, labs = [], []
        for k in params.keys():
            if isinstance(params[k], int):
                if not isinstance(params[k], bool):
                    idxs.append(params[k])
                    try:
                        idx = k.index('_') + 1
                        label = k[:idx] + '{{' + k[idx:] + '}}'
                    except ValueError:
                        label = k
                    label = r'${{\rm {}}}$'.format(label)
                    labs.append(label)
        return np.array(labs)[np.argsort(idxs)]

    @staticmethod
    def _populate_dictionary(theta, dictionary_in):
        """Populate the dictionary of free parameters."""
        dictionary = dictionary_in.copy()
        for key in dictionary.keys():
            if isinstance(dictionary[key], int):
                if not isinstance(dictionary[key], bool):
                    dictionary[key] = theta[dictionary[key]]
        return dictionary

    def verify_params_dictionary(self, params):
        """
        Check that the minimum number of parameters are provided for the
        fitting. For non-essential parameters a default value is set.
        """

        # Basic geometrical properties.

        params['x0'] = params.pop('x0', 0.0)
        params['y0'] = params.pop('y0', 0.0)
        params['dist'] = params.pop('dist', 100.0)
        params['vlsr'] = params.pop('vlsr', self.vlsr)
        if params.get('inc', None) is None:
            raise KeyError("Must provide `'inc'`.")
        if params.get('PA', None) is None:
            raise KeyError("Must provide `'PA'`.")

        # Rotation profile.

        has_vp_100 = False if params.get('vp_100') is None else True
        has_mstar = False if params.get('mstar') is None else True
        if has_mstar and has_vp_100:
            raise KeyError("Only provide either `'mstar'` or `'vp_100'`.")
        if not has_mstar and not has_vp_100:
            raise KeyError("Must provide either `'mstar'` or `'vp_100'`.")
        if has_mstar:
            params['vfunc'] = 'proj_vkep' #mod.proj_vkep
        else:
            params['vfunc'] = 'proj_vpow' # mod.proj_vpow

        params['vp_q'] = params.pop('vp_q', -0.5)
        params['vp_rtaper'] = params.pop('vp_rtaper', 1e10)
        params['vp_qtaper'] = params.pop('vp_qtaper', 1.0)
        params['vr_100'] = params.pop('vr_100', 0.0)
        params['vr_q'] = params.pop('vr_q', 0.0)

        # Flared emission surface.

        params['z0'] = params.pop('z0', None)
        params['psi'] = params.pop('psi', None)
        params['r_taper'] = params.pop('r_taper', None)
        params['q_taper'] = params.pop('q_taper', None)
        params['r_cavity'] = params.pop('r_cavity', None)
        params['z_func'] = params.pop('z_func', None)

        # Warp parameters.

        params['w_i'] = params.pop('w_i', 0.0)
        params['w_r'] = params.pop('w_r', 0.5 * max(self.xaxis))
        params['w_t'] = params.pop('w_t', 0.0)
        params['shadowed'] = params.pop('shadowed', False)

        # Masking parameters.

        params['r_min'] = params.pop('r_min', 0.0)
        params['r_max'] = params.pop('r_max', 1e10)
        params['exclude_r'] = params.pop('exclude_r', False)
        params['phi_min'] = params.pop('phi_min', -180.0)
        params['phi_max'] = params.pop('phi_max', 180.0)
        params['exclude_phi'] = params.pop('exclude_phi', False)
        params['abs_phi'] = params.pop('abs_phi', False)
        params['v_min'] = params.pop('v_min', np.nanmin(self.data))
        params['v_max'] = params.pop('v_max', np.nanmax(self.data))
        params['exclude_v'] = params.pop('exclude_v', False)
        params['user_mask'] = params.pop('user_mask', np.ones(self.data.shape))

        if params['r_min'] >= params['r_max']:
            raise ValueError("`r_max` must be greater than `r_min`.")
        if params['phi_min'] >= params['phi_max']:
            raise ValueError("`phi_max` must be great than `phi_min`.")
        if params['v_min'] >= params['v_max']:
            raise ValueError("`v_max` must be greater than `v_min`.")

        # Beam convolution.

        params['beam'] = bool(params.pop('beam', False))

        return params

    def evaluate_model(self, params, coords_only=False):
        """
        Compute a model given a params dictionary.

        Args:
            params (dict): The parameter dictionary passed to ``fit_map``.
            coords_only (Optional[bool]): Return the deprojected coordinates
                rather than the v0 model. Default is False.

        Returns:
            model (ndarray): The sampled model, either the v0 model, or, if
                ``coords_only`` is True, the deprojected cylindrical
                coordinates, (r, t, z).
        """

        verified_params = self.verify_params_dictionary(params.copy())
        if coords_only:
            model = self.disk_coords(**verified_params)
        else:
            model = self._make_model(verified_params)

        return model

    def evaluate_models(self, samples=None, params=None, draws=0.5,
                        collapse_func=np.mean, coords_only=False):
        """
        Evaluate models based on the samples provided and the parameter
        dictionary. If ``draws`` is an integer, it represents the number of
        random draws from ``samples`` which are then averaged to return the
        model. Alternatively ``draws`` can be a float between 0 and 1,
        representing the percentile of the samples to use to evaluate the model
        at.

        Args:
            samples (ndarray): An array of samples returned from ``fit_map``.
            params (dict): The parameter dictionary passed to ``fit_map``.
            draws (Optional[int/float]): If an integer, describes the number of
                random draws averaged to form the returned model. If a float,
                represents the percentile used from the samples. Must be
                between 0 and 1 if a float.
            collapse_func (Optional[callable]): How to collapse the random
                number of samples. Must be a function which allows an ``axis``
                argument (as with most Numpy functions).
            coords_only (Optional[bool]): Return the deprojected coordinates
                rather than the v0 model. Default is False.

        Returns:
            model (ndarray): The sampled model, either the v0 model, or, if
                ``coords_only`` is True, the deprojected cylindrical
                coordinates, (r, t, z).
        """

        # Check the input.

        if params is None:
            raise ValueError("Must provide model parameters dictionary.")

        # Model is fully specified.

        if samples is None:
            if coords_only:
                return self.disk_coords(**params.copy())
            else:
                return self._make_model(params.copy())

        nparam = np.sum([isinstance(params[k], int) for k in params.keys()])
        if samples.shape[1] != nparam:
            warning = "Invalid number of free parameters in 'samples': {:d}."
            raise ValueError(warning.format(nparam))
        if not callable(collapse_func):
            raise ValueError("'collapse_func' must be callable.")
        verified_params = self.verify_params_dictionary(params.copy())

        # Avearge over a random draw of models.

        if isinstance(int(draws) if draws > 1.0 else draws, int):
            models = []
            for idx in np.random.randint(0, samples.shape[0], draws):
                tmp = self._populate_dictionary(samples[idx], verified_params)
                if coords_only:
                    models += [self.disk_coords(**tmp)]
                else:
                    models += [self._make_model(tmp)]
            return collapse_func(models, axis=0)

        # Take a percentile of the samples.

        elif isinstance(draws, float):
            tmp = np.percentile(samples, draws, axis=0)
            tmp = self._populate_dictionary(tmp, verified_params)
            if coords_only:
                return self.disk_coords(**tmp)
            else:
                return self._make_model(tmp)

        else:
            raise ValueError("'draws' must be a float or integer.")

    def save_model(self, samples=None, params=None, model=None, filename=None,
                   overwrite=True):
        """
        Save the model as a FITS file.
        """
        from astropy.io import fits
        if model is None:
            model = self.evaluate_models(samples, params)
        if self.header['naxis1'] > self.nypix:
            canvas = np.ones(self._original_shape) * np.nan
            canvas[self._ya:self._yb, self._xa:self._xb] = model
            model = canvas.copy()
        if filename is None:
            filename = self.path.replace('.fits', '_model.fits')
        fits.writeto(filename, model, self.header, overwrite=overwrite)

    def mirror_residual(self, samples, params, mirror_velocity_residual=True,
                        mirror_axis='minor', return_deprojected=False,
                        deprojected_dpix_scale=1.0):
        """
        Return the residuals after subtracting a mirror image of either the
        rotation map, as in _Huang et al. 2018, or the residuals, as in
        _Izquierdo et al. 2021.

        .. _Huang et al. 2018: https://ui.adsabs.harvard.edu/abs/2018ApJ...867....3H/abstract
        .. _Izquierdo et al. 2021: https://ui.adsabs.harvard.edu/abs/2021arXiv210409530V/abstract

        Args:
            samples (ndarray): An array of samples returned from ``fit_map``.
            params (dict): The parameter dictionary passed to ``fit_map``.
            mirror_velocity_residual (Optional[bool]): If ``True``, the
                default, mirror the velocity residuals, otherwise use the line
                of sight velocity corrected for the systemic velocity.
            mirror_axis (Optional[str]): Which axis to mirror the image along,
                either ``'minor'`` or ``'major'``.
            return_deprojected (Optional[bool]): If ``True``, return the
                deprojected image, or, if ``False``, reproject the data onto
                the sky.

        Returns:
            x, y, residual (array, array, array): The x- and y-axes of the
                residual (either the sky axes or deprojected, depending on the
                chosen arguments) and the residual.
        """

        # Calculate the image to mirror.

        if mirror_velocity_residual:
            to_mirror = self.data * 1e3 - self.evaluate_models(samples, params)
        else:
            to_mirror = self.data.copy() * 1e3
            if isinstance(type(params['vlsr']), int):
                to_mirror -= params['vlsr']
            else:
                to_mirror -= np.median(samples, axis=0)[params['vlsr']]

        # Generate the axes for the deprojected image.

        r, t, _ = self.evaluate_models(samples, params, coords_only=True)
        t += np.pi / 2.0 if mirror_axis.lower() == 'minor' else 0.0
        x = np.nanmax(np.where(np.isfinite(to_mirror), r, np.nan))
        x = np.arange(-x, x, deprojected_dpix_scale * self.dpix)
        x -= 0.5 * (x[0] + x[-1])

        xs = (r * np.cos(t)).flatten()
        ys = (r * np.sin(t)).flatten()

        # Deproject the image.

        from scipy.interpolate import griddata
        d = griddata((xs, ys), to_mirror.flatten(), (x[:, None], x[None, :]))

        # Either subtract or add the mirrored image. Only want to add when
        # mirroring the line-of-sight velocity and mirroring about the minor
        # axis.

        if not mirror_velocity_residual and mirror_axis == 'minor':
            d += d[:, ::-1]
        else:
            d -= d[:, ::-1]

        if return_deprojected:
            return x, x, d

        # Reproject the residuals onto the sky plane.

        dd = d.flatten()
        mask = np.isfinite(dd)
        xx, yy = np.meshgrid(x, x)
        xx, yy = xx.flatten(), yy.flatten()
        xx, yy, dd = xx[mask], yy[mask], dd[mask]

        from scipy.interpolate import interp2d

        f = interp2d(xx, yy, dd, kind='linear')
        f = np.squeeze([f(xd, yd) for xd, yd in zip(xs, ys)])
        f = f.reshape(to_mirror.shape)
        f = np.where(np.isfinite(to_mirror), f, np.nan)

        return self.xaxis, self.yaxis, f

    # -- Deprojection Functions -- #

    def vphi(self, params):
        """
        Return a rotation profile in [m/s]. If ``mstar`` is set, assume a
        Keplerian profile including corrections for non-zero emission heights.
        Otherwise, assume a powerlaw profile. See the documents in
        ``disk_coords`` for more details of the parameters.

        Args:
            params (dict): Dictionary of parameters describing the model.

        Returns:
            vproj (ndarray): Projected Keplerian rotation at each pixel (m/s).
        """
        rvals, tvals, zvals = self.disk_coords(**params)

        if params['vfunc'] == 'proj_vkep':
            # print(rvals, tvals, zvals, params['dist'], params['mstar'], params['inc'])
            # print(np.sum(rvals), np.sum(tvals), np.sum(zvals))
            v_phi = mod.proj_vkep(rvals, tvals, zvals, params['dist'], params['mstar'], params['inc'])
        else:
            # roj_vpow(rvals, tvals, zvals, dist, mstar, vp_q, vp_100, vp_qtaper, vp_rtaper):
            v_phi = mod.proj_vpow(rvals, tvals, zvals, params['dist'], params['mstar'], params['inc'],
                                params['vp_q'], params['vp_100'], params['vp_qtaper'], params['vp_rtaper'])
        return v_phi + params['vlsr']

    def get_shadowed_coords(self, x0, y0, inc, PA, z_func, w_func):
        """Return cyclindrical coords of surface in [arcsec, rad, arcsec]."""

        # Make the disk-frame coordinates.
        extend = self.shadowed_extend
        oversample = self.shadowed_oversample
        diskframe_coords = dp.get_diskframe_coords(self.xaxis, self.yaxis,
                    self.nxpix, self.nypix, extend=extend, oversample=oversample)

        xdisk, ydisk, rdisk, tdisk = diskframe_coords
        zdisk = z_func(rdisk) + w_func(rdisk, tdisk)

        # Incline the disk.
        inc = np.radians(inc)
        x_dep = xdisk
        y_dep = ydisk * np.cos(inc) - zdisk * np.sin(inc)

        # Remove shadowed pixels.
        if inc < 0.0:
            y_dep = np.maximum.accumulate(y_dep, axis=0)
        else:
            y_dep = np.minimum.accumulate(y_dep[::-1], axis=0)[::-1]

        # Rotate and the disk.
        x_rot, y_rot = dp.rotate_coords(x_dep, y_dep, PA)
        x_rot, y_rot = x_rot + x0, y_rot + y0

        # Grid the disk.
        disk = (x_rot.flatten(), y_rot.flatten())
        grid = (self.xaxis[None, :], self.yaxis[:, None])
        r_obs = griddata(disk, rdisk.flatten(), grid,
                         method=self.shadowed_method)
        t_obs = griddata(disk, tdisk.flatten(), grid,
                         method=self.shadowed_method)
        return r_obs, t_obs, z_func(r_obs)

    def _v0(self, params):
        """Projected velocity structre."""
        rvals, tvals, zvals = self.disk_coords(**params)
        v0 = params['vfunc'](rvals, tvals, zvals, params)
        return v0 + params['vlsr']


    # -- Functions to build projected velocity profiles. -- #

    def _make_model(self, params):
        """Build the velocity model from the dictionary of parameters."""
        v0 = self._v0(params)
        if params['beam']:
            v0 = datacube._convolve_image(v0, self._beamkernel())
        return v0


    def deproject_model_residuals(self, samples, params):
        """
        Deproject the residuals into cylindrical velocity components. Takes the
        median value of the samples to determine the model.

        Args:
            samples (ndarray): An array of samples returned from ``fit_map``.
            params (dict): The parameter dictionary passed to ``fit_map``.
        """
        median_samples = np.median(samples, axis=0)
        verified_params = self.verify_params_dictionary(params.copy())
        model = self._populate_dictionary(median_samples, verified_params)
        v_res = self.data - self._make_model(model)
        rvals, tvals, zvals = self.disk_coords(**model)
        v_p = v_res / np.cos(tvals) / np.sin(-np.radians(model['inc']))
        v_r = v_res / np.sin(tvals) / np.sin(abs(np.radians(model['inc'])))
        v_z = v_res / np.cos(abs(np.radians(model['inc'])))
        v_z = np.where(zvals >= 0.0, -v_z, v_z)
        return v_p, v_r, v_z

    # -- Functions to help determine the emission height. -- #

    def find_maxima(self, x0=0.0, y0=0.0, PA=0.0, vlsr=None, r_max=None,
                    r_min=None, smooth=False, through_center=True):
        """
        Find the node of velocity maxima along the major axis of the disk.

        Args:
            x0 (Optional[float]): Source center offset along x-axis in
                [arcsec].
            y0 (Optional[float]): Source center offset along y-axis in
                [arcsec].
            PA (Optioanl[float]): Source position angle in [deg].
            vlsr (Optional[float]): Systemic velocity in [m/s].
            r_max (Optional[float]): Maximum offset to consider in [arcsec].
            r_min (Optional[float]): Minimum offset to consider in [arcsec].
            smooth (Optional[bool/float]): Smooth the line of nodes. If
                ``True``, smoth with the beam kernel, otherwise ``smooth``
                describes the FWHM of the Gaussian convolution kernel in
                [arcsec].
            through_center (Optional[bool]): If ``True``, force the central
                pixel to go through ``(0, 0)``.

        Returns:
            array, array: Arrays of ``x_sky`` and ``y_sky`` values of the
            maxima.
        """

        # Default parameters.
        vlsr = np.nanmedian(self.data) if vlsr is None else vlsr
        r_max = 0.5 * self.xaxis.max() if r_max is None else r_max
        r_min = 0.0 if r_min is None else r_min

        # Shift and rotate the image.
        data = self._shift_center(dx=x0, dy=y0, save=False)
        data = self._rotate_image(PA=PA, data=data, save=False)

        # Find the maximum values. Apply some clipping to help.
        mask = np.maximum(0.3 * abs(self.xaxis), self.bmaj)
        mask = abs(self.yaxis)[:, None] > mask[None, :]
        resi = np.where(mask, 0.0, abs(data - vlsr))
        resi = np.take(self.yaxis, np.argmax(resi, axis=0))

        # Gentrification.
        if through_center:
            resi[abs(self.xaxis).argmin()] = 0.0
        if smooth:
            if isinstance(smooth, bool):
                kernel = np.hanning(self.bmaj / self.dpix)
                kernel /= kernel.sum()
            else:
                kernel = Gaussian1DKernel(smooth / self.fwhm / self.dpix)
            resi = np.convolve(resi, kernel, mode='same')
        mask = np.logical_and(abs(self.xaxis) <= r_max,
                              abs(self.xaxis) >= r_min)
        x, y = self.xaxis[mask], resi[mask]

        # Rotate back to sky-plane and return.
        x, y = self._rotate_coords(x, -y, PA)
        return x + x0, y + y0

    def find_minima(self, x0=0.0, y0=0.0, PA=0.0, vlsr=None, r_max=None,
                    r_min=None, smooth=False, through_center=True):
        """
        Find the line of nodes where v0 = v_LSR along the minor axis.

        Args:
            x0 (Optional[float]): Source center offset along x-axis in
                [arcsec].
            y0 (Optional[float]): Source center offset along y-axis in
                [arcsec].
            PA (Optioanl[float]): Source position angle in [deg].
            vlsr (Optional[float]): Systemic velocity in [m/s].
            r_max (Optional[float]): Maximum offset to consider in [arcsec].
            r_min (Optional[float]): Minimum offset to consider in [arcsec].
            smooth (Optional[bool/float]): Smooth the line of nodes. If
                ``True``, smoth with the beam kernel, otherwise ``smooth``
                describes the FWHM of the Gaussian convolution kernel in
                [arcsec].
            through_center (Optional[bool]): If ``True``, force the central
                pixel to go through ``(0, 0)``.

        Returns:
            array, array: Arrays of ``x_sky`` and ``y_sky`` values of the
            velocity minima.
        """

        # Default parameters.
        vlsr = np.nanmedian(self.data) if vlsr is None else vlsr
        r_max = 0.5 * self.xaxis.max() if r_max is None else r_max
        r_min = 0.0 if r_min is None else r_min

        # Shift and rotate the image.
        data = self._shift_center(dx=x0, dy=y0, save=False)
        data = self._rotate_image(PA=PA, data=data, save=False)

        # Find the maximum values. Apply some clipping to help.
        mask = np.maximum(0.3 * abs(self.yaxis), self.bmaj)
        mask = abs(self.xaxis)[None, :] > mask[:, None]
        resi = np.where(mask, 1e10, abs(data - vlsr))
        resi = np.take(-self.yaxis, np.argmin(resi, axis=1))

        # Gentrification.
        if through_center:
            resi[abs(self.yaxis).argmin()] = 0.0
        if smooth:
            if isinstance(smooth, bool):
                kernel = np.hanning(self.bmaj / self.dpix)
                kernel /= kernel.sum()
            else:
                kernel = Gaussian1DKernel(smooth / self.fwhm / self.dpix)
            resi = np.convolve(resi, kernel, mode='same')
        mask = np.logical_and(abs(self.yaxis) <= r_max,
                              abs(self.yaxis) >= r_min)
        x, y = resi[mask], self.yaxis[mask]

        # Rotate back to sky-plane and return.
        x, y = self._rotate_coords(x, -y, PA)
        return x + x0, y + y0

    def _fit_surface(self, inc, PA, x0=None, y0=None, r_min=None, r_max=None,
                     fit_z1=False, ycut=20.0, **kwargs):
        """
        Fit the emission surface with the parametric model based on the pixels
        of peak velocity.

        Args:
            inc (float): Disk inclination in [degrees].
            PA (float): Disk position angle in [degrees].
            x0 (Optional[float]): Disk center x-offset in [arcsec].
            y0 (Optional[float]): Disk center y-offset in [arcsec].
            r_min (Optional[float]): Minimum radius to fit in [arcsec].
            r_max (Optional[float]): Maximum radius to fit in [arcsec].
            fit_z1 (Optional[bool]): Include (z1, phi) in the fit.
            y_cut (Optional[float]): Only use the `ycut` fraction of the minor
                axis for fitting.
            kwargs (Optional[dict]): Additional kwargs for curve_fit.

        Returns:
            popt (list): Best-fit parameters of (z0, psi[, z1, phi]).
        """

        def z_func(x, z0, psi, z1=0.0, phi=1.0):
            return z0 * x**psi + z1 * x**phi

        # Get the coordinates to fit.
        r, z = self.get_peak_pix(PA=PA, inc=inc, x0=x0, y0=y0,
                                 frame='disk', ycut=ycut)

        # Mask the coordinates.
        r_min = r_min if r_min is not None else r[0]
        r_max = r_max if r_max is not None else r[-1]
        m1 = np.logical_and(np.isfinite(r), np.isfinite(z))
        m2 = (r >= r_min) & (r <= r_max)
        mask = m1 & m2
        r, z = r[mask], z[mask]

        # Initial guesses for the free params.
        p0 = [z[abs(r - 1.0).argmin()], 1.0]
        if fit_z1:
            p0 += [-0.05, 3.0]
        maxfev = kwargs.pop('maxfev', 100000)

        # First fit the inner half to get a better p0 value.
        mask = r <= 0.5 * r.max()
        p_in = curve_fit(z_func, r[mask], z[mask], p0=p0, maxfev=maxfev,
                         **kwargs)[0]

        # Remove the pixels which are 'negative' depending on the inclination.
        mask = z > 0.0 if np.sign(p_in[0]) else z < 0.0
        popt = curve_fit(z_func, r[mask], z[mask], p0=p_in, maxfev=maxfev,
                         **kwargs)[0]
        return popt

    # -- Axes Functions -- #

    def _downsample_cube(self, N, randomize=False):
        """Downsample the cube to make faster calculations."""
        N = int(np.ceil(self.bmaj / self.dpix)) if N == 'beam' else N
        if randomize:
            N0x, N0y = np.random.randint(0, N, 2)
        else:
            N0x, N0y = int(N / 2), int(N / 2)
        if N > 1:
            self.xaxis = self.xaxis[N0x::N]
            self.yaxis = self.yaxis[N0y::N]
            self.data = self.data[N0y::N, N0x::N]
            self.error = self.error[N0y::N, N0x::N]

    def _shift_center(self, dx=0.0, dy=0.0, data=None, save=True):
        """
        Shift the center of the image.

        Args:
            dx (optional[float]): shift along x-axis [arcsec].
            dy (optional[float]): Shift along y-axis [arcsec].
            data (optional[ndarray]): Data to shift. If nothing is provided,
                will shift the attached ``rotationmap.data``.
            save (optional[bool]): If True, overwrite ``rotationmap.data``.
        """

        data = self.data.copy() if data is None else data
        to_shift = np.where(np.isfinite(data), data, 0.0)
        data = shift(to_shift, [-dy / self.dpix, dx / self.dpix])
        if save:
            self.data = data
        return data

    def _rotate_image(self, PA, data=None, save=True):
        """
        Rotate the image anticlockwise about the center.

        Args:
            PA (float): Rotation angle in [degrees].
            data (optional[ndarray]): Data to shift. If nothing is provided,
                will shift the attached ``rotationmap.data``.
            save (optional[bool]): If True, overwrite ``rotationmap.data``.
        """

        data = self.data.copy() if data is None else data
        to_rotate = np.where(np.isfinite(data), data, 0.0)
        data = rotate(to_rotate, PA - 90.0, reshape=False)
        if save:
            self.data = data
        return data

    # -- DATA I/O -- #

    def _readuncertainty(self, uncertainty, FOV=None):
        """Reads the uncertainties."""
        if uncertainty is not None:
            self.error = datacube(uncertainty, FOV=FOV, fill=None).data.copy()
        else:
            try:
                uncertainty = '_'.join(self.path.split('_')[:-1])
                uncertainty += '_d' + self.path.split('_')[-1]
                print("Assuming uncertainties in {}.".format(uncertainty))
                self.error = datacube(uncertainty, FOV=FOV, fill=None)
                self.error = self.error.data.copy()
            except FileNotFoundError:
                print("No uncertainties found, assuming uncertainties of 10%.")
                print("Change this at any time with `rotationmap.error`.")
                self.error = 0.1 * (self.data - np.nanmedian(self.data))
        self.error = np.where(np.isnan(self.error), 0.0, abs(self.error))

    # -- PLOTTING -- #

    @staticmethod
    def colormap():

        c2 = plt.cm.Reds(np.linspace(0, 1, 32))
        c1 = plt.cm.Blues_r(np.linspace(0, 1, 32))
        c1 = np.vstack([c1, [1, 1, 1, 1]])
        colors = np.vstack((c1, c2))
        return mcolors.LinearSegmentedColormap.from_list('eddymap', colors)

    @property
    def extent(self):
        return [self.xaxis[0], self.xaxis[-1], self.yaxis[0], self.yaxis[-1]]

    def plot_model(self, samples=None, params=None, model=None, mask=None,
                   ax=None, imshow_kwargs=None, cb_label=None,
                   return_fig=False):
        """
        Plot a v0 model using the same scalings as the plot_data() function.

        Args:
            samples (Optional[ndarray]): An array of samples returned from
                ``fit_map``.
            params (Optional[dict]): The parameter dictionary passed to
                ``fit_map``.
            model (Optional[ndarry]): A model array from ``evaluate_models``.
            draws (Optional[int/float]): If an integer, describes the number of
                random draws averaged to form the returned model. If a float,
                represents the percentile used from the samples. Must be
                between 0 and 1 if a float.
            collapse_func (Optional[callable]): How to collapse the random
                number of samples. Must be a function which allows an ``axis``
                argument (as with most Numpy functions).
            mask (Optional[ndarray]): The mask used for the fitting to plot as
                a shaded region.
            ax (Optional[AxesSubplot]): Axis to plot onto.
            imshow_kwargs (Optional[dict]): Dictionary of imshow kwargs.
            cb_label (Optional[str]): Colorbar label.
            return_fig (Optional[bool]): Return the figure.

        Returns:
            fig (Matplotlib figure): If ``return_fig`` is ``True``. Can access
                the axes through ``fig.axes`` for additional plotting.
        """

        # Dummy axis for the plotting.

        if ax is None:
            fig, ax = plt.subplots()

        # Make the model and calculate the plotting limits.

        if model is None:
            model = self.evaluate_models(samples, params.copy())
        vmin, vmax = np.nanpercentile(model, [2, 98])
        vmax = max(abs(vmin - self.vlsr), abs(vmax - self.vlsr))
        vmin = self.vlsr - vmax
        vmax = self.vlsr + vmax

        # Initialize the plotting parameters.

        imshow_kwargs = {} if imshow_kwargs is None else imshow_kwargs
        imshow_kwargs['cmap'] = imshow_kwargs.pop('cmap', datacube.cmap())
        imshow_kwargs['zorder'] = imshow_kwargs.pop('zorder', -9)
        imshow_kwargs['extent'] = self.extent
        imshow_kwargs['origin'] = 'lower'
        imshow_kwargs['vmin'] = imshow_kwargs.pop('vmin', vmin)
        imshow_kwargs['vmax'] = imshow_kwargs.pop('vmax', vmax)
        im = ax.imshow(model, **imshow_kwargs)

        # Overplot the mask if necessary.

        if mask is not None:
            ax.contour(self.xaxis, self.yaxis, mask,
                       [0.0], colors='k')
            ax.contourf(self.xaxis, self.yaxis, mask,
                        [-1.0, 0.0], colors='k', alpha=0.5)

        if cb_label is None:
            cb_label = r'${\rm v_{0} \quad (m\,s^{-1})}$'
        if cb_label != '':
            cb = plt.colorbar(im, pad=0.03, format='%.2f', extend='both')
            cb.set_label(cb_label, rotation=270, labelpad=15)
            cb.minorticks_on()

        self._gentrify_plot(ax)

        if return_fig:
            return fig


    def plot_model_residual(self, samples=None, params=None, model=None,
                            mask=None, ax=None, imshow_kwargs=None,
                            return_fig=False, save_name=None):
        """
        Plot the residual from the provided model.

        Args:
            samples (ndarray): An array of samples returned from ``fit_map``.
            params (dict): The parameter dictionary passed to ``fit_map``.
            mask (Optional[ndarray]): The mask used for the fitting to plot as
                a shaded region.
            ax (Optional[AxesSubplot]): Axis to plot onto.
            imshow_kwargs (Optional[dict]): Dictionary of keyword arguments to
                pass to ``imshow``.
            return_fig (Optional[bool]): Return the figure.

        Returns:
            fig (Matplotlib figure): If ``return_fig`` is ``True``. Can access
                the axes through ``fig.axes`` for additional plotting.
        """

        # Dummy axis to overplot.

        if ax is None:
            fig, ax = plt.subplots()

        # Make the model and calculate the plotting limits.

        if model is None:
            model = self.evaluate_models(samples=samples, params=params.copy())
        vres = self.data - model
        mask = np.ones(vres.shape) if mask is None else mask
        masked_vres = np.where(mask, vres, np.nan)
        vmin, vmax = np.nanpercentile(masked_vres, [2, 98])
        vmax = max(abs(vmin), abs(vmax))

        # Plot the data.

        imshow_kwargs = {} if imshow_kwargs is None else imshow_kwargs
        imshow_kwargs['origin'] = 'lower'
        imshow_kwargs['extent'] = self.extent
        imshow_kwargs['cmap'] = 'RdBu_r'
        imshow_kwargs['vmin'] = imshow_kwargs.pop('vmin', -vmax)
        imshow_kwargs['vmax'] = imshow_kwargs.pop('vmax', vmax)
        im = ax.imshow(vres, **imshow_kwargs)

        # Overplot the mask is necessary.

        if mask is not None:
            ax.contour(self.xaxis, self.yaxis, mask,

                       [0.0], colors='k')
            ax.contourf(self.xaxis, self.yaxis, mask,
                        [-1.0, 0.0], colors='k', alpha=0.5)

        cb = plt.colorbar(im, pad=0.02, format='%d', ax=ax, extend='both')
        cb.set_label(r'${\rm  v_{0} - v_{mod} \quad (m\,s^{-1})}$',
                     rotation=270, labelpad=15)
        cb.minorticks_on()

        self._gentrify_plot(ax)

        if save_name is not None:
            plt.savefig('{0}_residuals.png'.format(save_name), dpi=300)

        if return_ax:
            return ax

        if return_fig:
            return fig

    def plot_model_surface(self, samples, params, plot_surface_kwargs=None,
                           mask_with_data=True, return_fig=True):
        """
        Overplot the emission surface onto the provided axis. Takes the median
        value of the samples as the model to plot.

        Args:
            samples (ndarray): An array of samples returned from ``fit_map``.
            params (dict): The parameter dictionary passed to ``fit_map``.
            plot_surface_kwargs (Optional[dict]): Dictionary of kwargs to pass
                to ``plot_surface``.
            mask_with_data (Optional[bool]): If ``True``, mask the surface to
                regions where the data is finite valued.
            return_fig (Optional[bool]): Return the figure.

        Returns:
            fig (Matplotlib figure): If ``return_fig`` is ``True``. Can access
                the axes through ``fig.axes`` for additional plotting.
        """

        # Dummy axis to overplot.
        if ax is None:
            ax = plt.subplots()[1]

        # Front half of the disk.
        rf, tf, zf = self.disk_coords(x0=x0, y0=y0, inc=inc, PA=PA, z0=z0,
                                      psi=psi, r_cavity=r_cavity,
                                      r_taper=r_taper, q_taper=q_taper)
        mf = np.logical_and(zf >= 0.0, np.isfinite(self.data))
        rf = np.where(mf, rf, np.nan)
        tf = np.where(mf, tf, np.nan)

        # Rear half of the disk.
        rb, tb, zb = self.disk_coords(x0=x0, y0=y0, inc=inc, PA=PA, z0=-z0,
                                      psi=psi, r_cavity=r_cavity,
                                      r_taper=r_taper, q_taper=q_taper)
        mb = np.logical_and(zb <= 0.0, np.isfinite(self.data))
        rb = np.where(mb, rb, np.nan)
        tb = np.where(mb, tb, np.nan)

        # Flat disk for masking.
        rr, _, _ = self.disk_coords(x0=x0, y0=y0, inc=inc, PA=PA)

        # Make sure the bounds are OK.
        r_min = 0.0 if r_min is None else r_min
        r_max = rr.max() if r_max is None else r_max

        # Make sure the front side hides the rear.
        mf = np.logical_and(rf >= r_min, rf <= r_max)
        mb = np.logical_and(rb >= r_min, rb <= r_max)
        tf = np.where(mf, tf, np.nan)
        rb = np.where(~mf, rb, np.nan)
        tb = np.where(np.logical_and(np.isfinite(rb), mb), tb, np.nan)

        # For some geometries we want to make sure they're not doing funky
        # things in the outer disk when psi is large.
        if check_mask:
            mm = rr <= check_mask * r_max
            rf = np.where(mm, rf, np.nan)
            rb = np.where(mm, rb, np.nan)
            tf = np.where(mm, tf, np.nan)
            tb = np.where(mm, tb, np.nan)

        # Popluate the kwargs with defaults.
        lw = kwargs.pop('lw', kwargs.pop('linewidth', 1.0))
        zo = kwargs.pop('zorder', 10000)
        c = kwargs.pop('colors', kwargs.pop('c', 'k'))

        radii = np.linspace(0, r_max, int(nrad + 1))[1:]
        theta = np.linspace(-np.pi, np.pi, int(ntheta))
        theta += np.diff(theta)[0]

        # Do the plotting.
        ax.contour(self.xaxis, self.yaxis, rf, levels=radii, colors=c,
                   linewidths=lw, linestyles='-', zorder=zo, **kwargs)
        for tt in theta:
            tf_tmp = np.where(abs(tf - tt) <= 0.5, tf, np.nan)
            ax.contour(self.xaxis, self.yaxis, tf_tmp, levels=[tt], colors=c,
                       linewidths=lw, linestyles='-', zorder=zo, **kwargs)
        ax.contour(self.xaxis, self.yaxis, rb, levels=radii, colors=c,
                   linewidths=lw, linestyles='--', zorder=zo, **kwargs)
        for tt in theta:
            tb_tmp = np.where(abs(tb - tt) <= 0.5, tb, np.nan)
            ax.contour(self.xaxis, self.yaxis, tb_tmp, levels=theta, colors=c,
                       linewidths=lw, linestyles='--', zorder=zo)

        return ax


    def plot_disk_axes(self, x0=0.0, y0=0.0, inc=0.0, PA=0.0, major=1.0,
                       ax=None, plot_kwargs=None, return_ax=True):
        """
        Plot the major and minor axes on the provided axis.

        Args:
            ax (Matplotlib axes): Axes instance to plot onto.
            x0 (Optional[float]): Relative x-location of the center [arcsec].
            y0 (Optional[float]): Relative y-location of the center [arcsec].
            inc (Optional[float]): Inclination of the disk in [degrees].
            PA (Optional[float]): Position angle of the disk in [degrees].
            major (Optional[float]): Size of the major axis line in [arcsec].
            plot_kwargs (Optional[dict]): Dictionary of parameters to pass to
                ``matplotlib.plot``.

        Returns:
            matplotlib axis: Matplotlib ax with axes drawn.
        """

        # Dummy axis to plot.

        if ax is None:
            fig, ax = plt.subplots()

        x = np.array([major, -major,
                      0.0, 0.0])
        y = np.array([0.0, 0.0,
                      major * np.cos(np.radians(inc)),
                      -major * np.cos(np.radians(inc))])
        x, y = self._rotate_coords(x, y, PA=PA)
        x, y = x + x0, y + y0
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs
        c = plot_kwargs.pop('c', plot_kwargs.pop('color', 'k'))
        ls = plot_kwargs.pop('ls', plot_kwargs.pop('linestyle', ':'))
        lw = plot_kwargs.pop('lw', plot_kwargs.pop('linewidth', 1.0))
        ax.plot(x[:2], y[:2], c=c, ls=ls, lw=lw)
        ax.plot(x[2:], y[2:], c=c, ls=ls, lw=lw)

        if return_ax:
            ax

    def plot_maxima(self, x0=0.0, y0=0.0, inc=0.0, PA=0.0, vlsr=None,
                    r_max=1.0, r_min=None, smooth=False, through_center=True,
                    plot_axes_kwargs=None, plot_kwargs=None,
                    return_fig=False):
        """
        Mark the position of the maximum velocity as a function of radius. This
        can help demonstrate if there is an appreciable bend in velocity
        contours indicative of a flared emission surface. This function pulls
        the results from fit_cube.get_peak_pix() for plotting.

        Args:
            x0 (Optional[float]): Source center x-offset in [arcsec].
            y0 (Optional[float]): Source center y-offset in [arcsec].
            inc (Optional[float]): Disk inclination in [degrees].
            PA (Optional[float]): Disk position angle in [degrees].
            vlsr (Optional[float]): Systemic velocity in [m/s].
            r_max (Optional[float]): Maximum offset to consider in [arcsec].
            r_min (Optional[float]): Minimum offset to consider in [arcsec].
                This is useful to skip large regions where beam convolution
                dominates the map.
            smooth (Optional[bool/float]): Smooth the line of nodes. If
                ``True``, smoth with the beam kernel, otherwise ``smooth``
                describes the FWHM of the Gaussian convolution kernel in
                [arcsec].
            through_center (Optional[bool]): If ``True``, force the central
                pixel to go through ``(0, 0)``.
            levels (Optional[array]): Levels to pass to ``plot_data``.
            plot_axes_kwargs (Optional[dict]): Dictionary of kwargs for the
                plot_axes function.
            plot_kwargs (Optional[dict]): Dictionary of kwargs for the
                plot of the maximum and minimum pixels.
            return_fig (Optional[bool]): If True, return the figure.

        Returns:
            (Matplotlib figure): If return_fig is ``True``. Can access the axis
                through ``fig.axes[0]``.
        """

        # Background figure.
        fig = self.plot_data(return_fig=True)
        ax = fig.axes[0]
        if plot_axes_kwargs is None:
            plot_axes_kwargs = dict()
        self.plot_disk_axes(ax=ax, x0=x0, y0=y0, inc=inc, PA=PA,
                            major=plot_axes_kwargs.pop('major', r_max),
                            **plot_axes_kwargs)

        # Get the pixels for the maximum and minimum values.
        x_maj, y_maj = self.find_maxima(x0=x0, y0=y0, PA=PA, vlsr=vlsr,
                                        r_max=r_max, r_min=r_min,
                                        smooth=smooth,
                                        through_center=through_center)
        r_max = r_max * np.cos(np.radians(inc))
        x_min, y_min = self.find_minima(x0=x0, y0=y0, PA=PA, vlsr=vlsr,
                                        r_max=r_max, r_min=r_min,
                                        smooth=smooth,
                                        through_center=through_center)

        # Plot the lines.
        plot_kwargs = {} if plot_kwargs is None else plot_kwargs
        c = plot_kwargs.pop('c', plot_kwargs.pop('color', 'k'))
        lw = plot_kwargs.pop('lw', plot_kwargs.pop('linewidth', 1.0))

        ax.plot(x_maj, y_maj, c=c, lw=lw, **plot_kwargs)
        ax.plot(x_min, y_min, c=c, lw=lw, **plot_kwargs)

        if return_fig:
            return fig

    @staticmethod
    def _plot_walkers(samples, nburnin=None, labels=None, histogram=True, save_name=None):
        """Plot the walkers to check if they are burning in."""
        if labels[0] == 'lnprob':
            samples = np.abs(samples)
        # Check the length of the label list.
        if labels is None:
            if samples.shape[0] != len(labels):
                raise ValueError("Not correct number of labels.")

        # Cycle through the plots.
        for s, sample in enumerate(samples):
            fig, ax = plt.subplots()
            for walker in sample.T:
                if labels[s] == 'lnprob':
                    ax.semilogy(walker, alpha=0.1, color='k')
                    ax.set_ylim([np.min(sample[~np.isnan(sample)])*0.9, np.max(sample[~np.isnan(sample)])*1.1])
                else:
                    ax.plot(walker, alpha=0.1, color='k')
            ax.set_xlabel('Steps')
            if labels is not None:
                ax.set_ylabel(labels[s])
            if nburnin is not None:
                ax.axvline(nburnin, ls=':', color='r')

            # Include the histogram.
            if histogram:
                fig.set_size_inches(1.37 * fig.get_figwidth(),
                                    fig.get_figheight(), forward=True)
                ax_divider = make_axes_locatable(ax)
                bins = np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 50)
                hist, _ = np.histogram(sample[nburnin:].flatten(), bins=bins,
                                       density=True)
                bins = np.average([bins[1:], bins[:-1]], axis=0)
                ax1 = ax_divider.append_axes("right", size="35%", pad="2%")
                ax1.fill_betweenx(bins, hist, np.zeros(bins.size), step='mid',
                                  color='darkgray', lw=0.0)
                if labels[s] == 'lnprob':
                    ax1.set_yscale('log')

                ax1.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1])
                ax1.set_xlim(0, ax1.get_xlim()[1])
                ax1.set_yticklabels([])
                ax1.set_xticklabels([])
                ax1.tick_params(which='both', left=0, bottom=0, top=0, right=0)
                ax1.spines['right'].set_visible(False)
                ax1.spines['bottom'].set_visible(False)
                ax1.spines['top'].set_visible(False)

            if save_name is not None:
                plt.savefig(save_name + '_{0}.png'.format(labels[s]) , dpi=300)

    @staticmethod
    def _plot_corner(samples, labels=None, quantiles=None, save_name=None):
        """Plot the corner plot to check for covariances."""

        quantiles = [0.16, 0.5, 0.84] if quantiles is None else quantiles
        corner.corner(samples, labels=labels, title_fmt='.4f', bins=30,
                      quantiles=quantiles, show_titles=True)
        if save_name is not None:
            plt.savefig(save_name + '_corner.png', dpi=300)


    def plot_beam(self, ax, dx=0.125, dy=0.125, **kwargs):
        """Plot the sythensized beam on the provided axes."""
        beam = Ellipse(ax.transLimits.inverted().transform((dx, dy)),
                       width=self.bmin, height=self.bmaj, angle=-self.bpa,
                       fill=kwargs.get('fill', False),
                       hatch=kwargs.get('hatch', '//////////'),
                       lw=kwargs.get('linewidth', kwargs.get('lw', 1)),
                       color=kwargs.get('color', kwargs.get('c', 'k')),
                       zorder=kwargs.get('zorder', 1000))
        ax.add_patch(beam)

    def _gentrify_plot(self, ax):
        """Gentrify the plot."""
        ax.set_aspect(1)
        ax.grid(ls=':', color='k', alpha=0.1, lw=0.5)
        ax.tick_params(which='both', right=True, top=True)
        ax.set_xlim(self.xaxis.max(), self.xaxis.min())
        ax.set_ylim(self.yaxis.min(), self.yaxis.max())
        ax.xaxis.set_major_locator(MaxNLocator(5, min_n_ticks=3))
        ax.yaxis.set_major_locator(MaxNLocator(5, min_n_ticks=3))
        ax.set_xlabel('Offset (arcsec)')
        ax.set_ylabel('Offset (arcsec)')
        if self.bmaj is not None:
            self.plot_beam(ax=ax)

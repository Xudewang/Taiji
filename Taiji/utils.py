from __future__ import division, print_function

import copy
import os
import sys
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np
import sep
from astropy import units as u
from astropy import wcs
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.table import Column, Table, vstack
from astropy.units import Quantity
from matplotlib import rcParams
from matplotlib.colorbar import Colorbar
from matplotlib.patches import Ellipse as mpl_ellip

from .imtools import ORG, SEG_CMAP, display_single


@contextmanager
def suppress_stdout():
    """Suppress the output.

    Based on: https://thesmithfam.org/blog/2012/10/25/temporarily-suppress-console-output-in-python/
    """
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


def func_brokenexp(r, I0, gamma, betta, alpha, Rb):
    S = (1 + np.exp(-alpha * Rb))**(1 / alpha * (1 / gamma - 1 / betta))
    return (S * I0 * np.exp(-r / gamma) *
            (1 + np.exp(alpha * (r - Rb)))**(1 / alpha *
                                             (1 / gamma - 1 / betta)))


def brokenexpfit_emcee(x,
                       y,
                       yerr,
                       I0_init,
                       I0_low,
                       I0_high,
                       gamma_init,
                       gamma_low,
                       gamma_high,
                       betta_init,
                       betta_low,
                       betta_high,
                       alpha_init,
                       alpha_low,
                       alpha_high,
                       Rb_init,
                       Rb_low,
                       Rb_high,
                       log_f_init=-2,
                       log_f_low=-10,
                       log_f_high=10,
                       nwalkers=100,
                       nsteps=1000,
                       discard=100,
                       nthreads=5,
                       plot=False,
                       verbose=False,
                       visual=False,
                       logger=None):
    """This is to perform exponential fit using EMCEE package.
    """

    #from multiprocessing import Pool
    import corner
    import emcee
    from pathos.multiprocessing import ProcessingPool as Pool

    def func_brokenexp(r, I0, gamma, betta, alpha, Rb):
        S = (1 + np.exp(-alpha * Rb))**(1 / alpha * (1 / gamma - 1 / betta))
        return (S * I0 * np.exp(-r / gamma) *
                (1 + np.exp(alpha * (r - Rb)))**(1 / alpha *
                                                 (1 / gamma - 1 / betta)))

    def log_likelihood(theta, x, y, yerr):
        I0, gamma, betta, alpha, Rb, log_f = theta
        model = func_brokenexp(x, I0, gamma, betta, alpha, Rb)
        sigma2 = yerr**2 + model**2 * np.exp(2 * log_f)
        return -0.5 * np.sum((y - model)**2 / sigma2 + np.log(sigma2))

    def log_prior(theta):
        I0, gamma, betta, alpha, Rb, log_f = theta
        if I0_low < I0 < I0_high and gamma_low < gamma < gamma_high and betta_low < betta < betta_high and alpha_low < alpha < alpha_high and Rb_low < Rb < Rb_high and log_f_low < log_f < log_f_high:
            return 0.0
        return -np.inf

    def log_probability(theta, x, y, yerr):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta, x, y, yerr)

    ndim = 6

    with Pool(nthreads) as pool:
        ini_geuess = (I0_init, gamma_init, betta_init, alpha_init, Rb_init,
                      log_f_init) + 1e-4 * np.random.randn(nwalkers, ndim)

        sampler = emcee.EnsembleSampler(nwalkers,
                                        ndim,
                                        log_probability,
                                        args=(x, y, yerr),
                                        pool=pool)
        sampler.run_mcmc(ini_geuess, nsteps, progress=True)

    fig, axes = plt.subplots(ndim, figsize=(10, 7), sharex=True)
    samples = sampler.get_chain()
    labels = ["I0", "gamma", "betta", "alpha", "Rb", "log_f"]
    for i in range(ndim):
        ax = axes[i]
        ax.plot(samples[:, :, i], "k", alpha=0.3)
        ax.set_xlim(0, len(samples))
        ax.set_ylabel(labels[i])
        ax.yaxis.set_label_coords(-0.1, 0.5)

    axes[-1].set_xlabel("step number")

    flat_samples = sampler.get_chain(discard=discard, thin=15, flat=True)

    fig = corner.corner(flat_samples, labels=labels)

    from IPython.display import Math, display
    parameter_arr = []
    for i in range(ndim):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        txt = "\mathrm{{{3}}} = {0:.3f}_{{-{1:.3f}}}^{{{2:.3f}}}"
        txt = txt.format(mcmc[1], q[0], q[1], labels[i])
        display(Math(txt))

        parameter_temp = [mcmc[1], q[0], q[1]]
        parameter_arr.append(parameter_temp)

    return parameter_arr


def running_percentile(x, y, percentile, bins = 20):
    """This is for calculating the running percentile of y as a function of x.

    Args:
        x (_type_): _description_
        y (_type_): _description_
        percentile (float): input 10 is the 10%
        bins (int, optional): _description_. Defaults to 20.

    Returns:
        _type_: _description_
    """
    from scipy.stats import binned_statistic

    # calculate the running percentiles of y as a function of x
    percentile_results = binned_statistic(
        x, y, statistic=lambda y: np.percentile(y, percentile), bins=bins)
    
    return percentile_results.statistic

def running_median_errorbar(x,y, method='percentile', bins=20):
    """This is for calculating the running median and std of y as a function of x.

    Args:
        x (_type_): _description_
        y (_type_): _description_
        bins (int, optional): _description_. Defaults to 20.

    Returns:
        _type_: _description_
    """
    from scipy.stats import binned_statistic

    # calculate the running percentiles of y as a function of x
    median_results = binned_statistic(
        x, y, statistic='median', bins=bins)
    
    if method == 'std':
        std_results = binned_statistic(
            x, y, statistic='std', bins=bins).statistic
        
        return median_results.statistic, std_results.statistic
        
    elif method == 'percentile':
        percentile_results_16 = running_percentile(x, y, 16, bins=bins)
        percentile_results_84 = running_percentile(x, y, 84, bins=bins)
        
        errorbar_low = median_results.statistic - percentile_results_16
        errorbar_high = percentile_results_84 - median_results.statistic
    
        return median_results.statistic, errorbar_low, errorbar_high
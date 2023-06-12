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
    
def match_sample(standard_sample, match_sample, bins, alpha, standard_sample_logM_index, match_sample_logM_index, seed = None):
    """This function is to match two samples, e.g. stellar mass.

    Args:
        standard_sample (numpy_array): The standard sample. This sample remains unchanged. We can get the numbers should be extracted for match sample combining the alpha value.
        match_sample (_type_): _description_
        bins (_type_): _description_
        alpha (float): The alpha value for matching two samples. The numbers from standard sample multiply alpha should be extracted from match sample.
    """
    
    import pandas as pd
    from astropy.table import Table
    from scipy.stats import binned_statistic
    
    if seed is None:
        seed = len(match_sample) - 1 
        
    np.random.seed(seed)
    
    print('The number of standard sample: ', len(standard_sample))
    print('The number of match sample: ', len(match_sample))
    ids = np.arange(len(match_sample))
    
    binned_count = binned_statistic(standard_sample[standard_sample_logM_index], standard_sample[standard_sample_logM_index], statistic='count', bins=bins)[0]
    extract_count = (binned_count * alpha).astype(int)
    
    indices = np.digitize(match_sample[match_sample_logM_index], bins)
    print(len(indices))
    
    sample = []
    for i in range(1, len(bins)):
        mask = indices == i
        sub_ids = np.random.choice(ids[mask], extract_count[i-1], replace=False)
        sub_sample = np.array(match_sample)[sub_ids]
        # rng = np.random.default_rng()
        # sub_sample = rng.choice(np.array(match_sample)[mask], extract_count[i-1], replace=False, axis=0)
        sample.append(sub_sample)
        
    catalog = np.concatenate(sample)
    
    # check if the match sample is pandas.dataframe or astropy.table.Table
    if isinstance(match_sample, pd.DataFrame):
        catalog_withname = Table(catalog, names=match_sample.columns)
        return catalog_withname
    
    elif isinstance(match_sample, Table):
        catalog_withname = Table(catalog, names=match_sample.colnames)
        return catalog_withname

    return catalog

def gaussian_tension_approxiation(chi2, dof = 20, x0 = 20):
    """This is for calculating equivalent gaussian sigma for a extremely low p-value due to large chi2 value. From Haslbauer et al. 2022.

    Args:
        chi2 (_type_): _description_
        dof (int, optional): _description_. Defaults to 20.
        x0 (int, optional): _description_. Defaults to 20.

    Returns:
        _type_: _description_
    """
    
    from scipy.optimize import fsolve
    
    right_chi2 = chi2**(dof/2-1)*np.exp(-chi2/2)/2**(dof/2)/np.math.factorial(dof/2-1)/(1/2-(dof/2-1)/chi2)
    right_chi2 = (dof/2-1)*np.log(chi2) - chi2/2 - np.log(2**(dof/2)) - np.log(np.math.factorial(dof/2-1)) - np.log(1/2-(dof/2-1)/chi2)
    print('right_chi2 = ', right_chi2)
    
    def func(x):
        return np.log(np.sqrt(2/np.pi)) + (-x**2-np.log(x)) - (right_chi2)
    
    if x0 is None:
        x0 = 20
        
    root = fsolve(func, x0)
    
    return root[0]*np.sqrt(2)

def gaussian_tension(chisqval, dof=20, x0=5):
    from scipy.integrate import quad
    from scipy.optimize import fsolve
    from scipy.stats import chi2

    p = 1 - chi2.cdf(chisqval, dof)
    print(f'The p-value for {chisqval} is: ', p)

    def f(x):
        return 1 - 1 / np.sqrt(2 * np.pi) * quad(lambda t: np.exp(-t**2 / 2),
                                                 -x, x)[0] - p

    x = fsolve(f, x0)

    #print('The tension:', x[0])

    return x[0]

def weighted_tng50(q_obs, logM_obs, q_tng50, logM_tng50, index_obs,
                   index_tng50, mass_bins, q_bins):
    from scipy.stats import binned_statistic

    # 1. divide the data into different bins
    # mass_bins = np.arange(10, 11.5001, 0.15)
    # print('mass_bins', mass_bins)
    # q_bins = np.arange(0, 1.0001, 0.05)
    # print('q_bins', q_bins)
    q_bins_center = (q_bins[1:] + q_bins[:-1]) / 2
    index_q_bins = [(q_obs > q_bins[i]) & (q_obs <= q_bins[i + 1])
                    for i in range(len(q_bins) - 1)]

    # calculate the weight_obs based the numbers in different mass bins.
    N_sim = binned_statistic(logM_tng50[index_tng50],
                             logM_tng50[index_tng50],
                             statistic='count',
                             bins=mass_bins)[0]
    N_obs = binned_statistic(logM_obs[index_obs],
                             logM_obs[index_obs],
                             statistic='count',
                             bins=mass_bins)[0]
    weight_obs = N_sim / N_obs
    print('Simulated number', np.sum(N_sim))
    print('Observed number', np.sum(N_obs))
    #print('weight_obs', weight_obs)

    # main step to derive the weight_total_i, this is the weight for each bin. That is to say, I directly multiply the weight_total_i to the histogram of the observations.
    N_total = 0
    weight_total_arr = []
    weight_max_arr = []
    sigma_obs_arr = []
    sigma_model_arr = []
    for i in range(len(q_bins) - 1):
        mass_qbin_arr = logM_obs[index_q_bins[i] & index_obs]

        bin_indices = np.digitize(mass_qbin_arr, mass_bins)

        weight_obs_bin_arr = weight_obs[bin_indices - 1]
        weight_total_bin = np.sum(weight_obs_bin_arr)
        weight_total_arr.append(weight_total_bin)

        N_total += len(mass_qbin_arr)

        if len(weight_obs_bin_arr) == 0:
            weight_max_arr.append(np.nan)
        else:
            weight_max_bin = np.max(weight_obs_bin_arr)
            weight_max_arr.append(weight_max_bin)
            
    weight_total_arr = np.array(weight_total_arr)
    sigma_model_arr = np.array(sigma_model_arr)

    # calculate the sigma_obs_i for each bin based on the poisson uncertainty. Equation (8) in their paper.
    for i in range(len(q_bins) - 1):
        if weight_max_arr[i] == 0:
            weight_max = np.nanmax(weight_max_arr)
            weight_total = np.nansum(weight_total_arr)
            sigma_obs_arr.append(weight_max / weight_total)
        else:
            weight_total = np.nansum(weight_total_arr)
            sigma_obs_bin = weight_max_arr[i] / weight_total * np.sqrt(
                weight_total_arr[i] / weight_max_arr[i] + 1)
            sigma_obs_arr.append(sigma_obs_bin)
            
    sigma_obs_arr = np.array(sigma_obs_arr)

    # calculate the sigma_model_i and unweighted observations for each bin based on the poisson uncertainty
    N_model_qbins = np.array(binned_statistic(q_tng50[index_tng50],
                               q_tng50[index_tng50],
                               statistic='count',
                               bins=q_bins)[0])
    N_obs_qbins = binned_statistic(q_obs[index_obs],
                                   q_obs[index_obs],
                                   statistic='count',
                                   bins=q_bins)[0]
    sigma_model_arr = [
        np.sqrt(N_model_qbins[i] + 1) / np.sum(N_model_qbins) for i in range(len(N_model_qbins))
    ]
    sigma_obs_arr_unweighted = [
        np.sqrt(N_obs_qbins[i] + 1) / np.sum(N_obs_qbins)
        for i in range(len(N_obs_qbins))
    ]
    
    sigma_model_arr = np.array(sigma_model_arr)
    
    # calculate the chi-square.
    chi_square_bin_arr = (weight_total_arr/np.sum(weight_total_arr)-N_model_qbins/np.sum(N_model_qbins))**2/(sigma_model_arr**2+sigma_obs_arr**2)
    chi_square = np.nansum(chi_square_bin_arr)

    # Combine all derived information into a dictory
    dict_info = {
        'q_bins_center': q_bins_center,
        'weight_total_arr': weight_total_arr,
        'weight_max_arr': weight_max_arr,
        'N_sim': N_sim,
        'N_obs': N_obs,
        'N_model_qbins': N_model_qbins,
        'N_obs_qbins': N_obs_qbins,
        'sigma_obs_arr': sigma_obs_arr,
        'sigma_model_arr': sigma_model_arr,
        'sigma_obs_arr_unweighted': sigma_obs_arr_unweighted,
        'chi-square': chi_square,
    }

    print('Ntotal in loop', N_total)
    print('total weight (should equal Nsim)', np.sum(weight_total_arr))

    return dict_info

def binedge_equalnumber(data, nbins):
    percentiles = np.linspace(0, 100, nbins)
    bins = np.percentile(data, percentiles)
    
    bins_center = (bins[:-1] +bins[1:]) / 2
    
    return bins, bins_center

def is_point_in_ellipse(center, width, height, angle, test_point):
    """
    Check if a test point is inside an ellipse.

    Parameters:
    center: The center point of the ellipse as a tuple (x, y).
    width: The width of the ellipse. Note: This is the full width, not the radius.
    height: The height of the ellipse.
    angle: The angle of rotation of the ellipse in degrees.
    test_point: The point to test as a tuple (x, y).

    Returns:
    True if the test point is inside the ellipse, False otherwise.
    """
    from matplotlib.patches import Ellipse

    # Create an ellipse object
    ellipse = Ellipse(xy=center, width=width, height=height, angle=angle)

    # Check if the test point is inside the ellipse
    return ellipse.contains_point(test_point)

from matplotlib.patches import Ellipse


def is_point_in_ellipse_ring(center, outer_width, outer_height, inner_width, inner_height, angle, test_point):
    """
    Check if a test point is inside an ellipse ring.

    Args:
        center (tuple): The center point of the ellipse as a tuple (x, y).
        outer_width (float): The full width of the outer ellipse.
        outer_height (float): The height of the outer ellipse.
        inner_width (float): The full width of the inner ellipse.
        inner_height (float): The height of the inner ellipse.
        angle (float): The angle of rotation of the ellipse in degrees.
        test_point (tuple): The point to test as a tuple (x, y).

    Returns:
        bool: True if the test point is inside the ellipse ring, False otherwise.
    """
    # Create an ellipse object and a smaller ellipse object
    ellipse = Ellipse(xy=center, width=outer_width, height=outer_height, angle=angle, facecolor='none', edgecolor='blue')
    inner_ellipse = Ellipse(xy=center, width=inner_width, height=inner_height, angle=angle, facecolor='none', edgecolor='green')

    # Check if the test point is inside the ellipse ring
    is_inside = ellipse.contains_point(test_point) and not inner_ellipse.contains_point(test_point)

    return is_inside
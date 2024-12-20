from __future__ import division, print_function

import copy
import os
import sys
from contextlib import contextmanager

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
    S = (1 + np.exp(-alpha * Rb)) ** (1 / alpha * (1 / gamma - 1 / betta))
    return (
        S
        * I0
        * np.exp(-r / gamma)
        * (1 + np.exp(alpha * (r - Rb))) ** (1 / alpha * (1 / gamma - 1 / betta))
    )


def brokenexpfit_emcee(
    x,
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
    logger=None,
):
    """This is to perform exponential fit using EMCEE package."""

    # from multiprocessing import Pool
    import corner
    import emcee
    from pathos.multiprocessing import ProcessingPool as Pool

    def func_brokenexp(r, I0, gamma, betta, alpha, Rb):
        S = (1 + np.exp(-alpha * Rb)) ** (1 / alpha * (1 / gamma - 1 / betta))
        return (
            S
            * I0
            * np.exp(-r / gamma)
            * (1 + np.exp(alpha * (r - Rb))) ** (1 / alpha * (1 / gamma - 1 / betta))
        )

    def log_likelihood(theta, x, y, yerr):
        I0, gamma, betta, alpha, Rb, log_f = theta
        model = func_brokenexp(x, I0, gamma, betta, alpha, Rb)
        sigma2 = yerr**2 + model**2 * np.exp(2 * log_f)
        return -0.5 * np.sum((y - model) ** 2 / sigma2 + np.log(sigma2))

    def log_prior(theta):
        I0, gamma, betta, alpha, Rb, log_f = theta
        if (
            I0_low < I0 < I0_high
            and gamma_low < gamma < gamma_high
            and betta_low < betta < betta_high
            and alpha_low < alpha < alpha_high
            and Rb_low < Rb < Rb_high
            and log_f_low < log_f < log_f_high
        ):
            return 0.0
        return -np.inf

    def log_probability(theta, x, y, yerr):
        lp = log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf
        return lp + log_likelihood(theta, x, y, yerr)

    ndim = 6

    with Pool(nthreads) as pool:
        ini_geuess = (
            I0_init,
            gamma_init,
            betta_init,
            alpha_init,
            Rb_init,
            log_f_init,
        ) + 1e-4 * np.random.randn(nwalkers, ndim)

        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, log_probability, args=(x, y, yerr), pool=pool
        )
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


def r_phy_to_ang(r_phy, redshift, cosmo="Planck18", phy_unit="kpc", ang_unit="arcsec"):
    """
    Convert physical radius into angular size.
    """
    # Cosmology
    if cosmo is None:
        from astropy.cosmology import FlatLambdaCDM

        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    elif cosmo == "Planck15":
        from astropy.cosmology import Planck15 as cosmo
    elif cosmo == "Planck18":
        from astropy.cosmology import Planck18 as cosmo

    # Convert the physical size into an Astropy quantity
    if not isinstance(r_phy, u.quantity.Quantity):
        r_phy = r_phy * u.Unit(phy_unit)

    return (r_phy / cosmo.kpc_proper_per_arcmin(redshift)).to(u.Unit(ang_unit))

def r_ang_to_phy(r_ang, redshift, cosmo="Planck18", phy_unit="kpc", ang_unit="arcsec"):
    """
    Convert angular size into physical radius.
    """
    if cosmo is None:
        from astropy.cosmology import FlatLambdaCDM

        cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
    elif cosmo == "Planck15":
        from astropy.cosmology import Planck15 as cosmo
    elif cosmo == "Planck18":
        from astropy.cosmology import Planck18 as cosmo

    # Convert the angular size into an Astropy quantity
    if not isinstance(r_ang, u.quantity.Quantity):
        r_ang = r_ang * u.Unit(ang_unit)

    r_phy = (r_ang * cosmo.kpc_proper_per_arcmin(redshift)).to(u.Unit(phy_unit))

    # convert to pure number
    r_phy = r_phy.value

    return r_phy
    
def running_percentile(x, y, percentile, bins=20):
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
        x, y, statistic=lambda y: np.nanpercentile(y, percentile), bins=bins
    )

    return percentile_results.statistic


def sliding_stats(
    x_values, y_values, bin_size, slide_size, percentiles, start=None, boundary=None
):
    """
    Calculate the running mean and percentiles of y as a function of x using a sliding window.

    Parameters:
        x_values (numpy.array): The x values.
        y_values (numpy.array): The y values.
        bin_size (float): The size of the sliding window.
        slide_size (float): The step size for the sliding window.
        percentiles (list): The percentiles to calculate.
        start (float, optional): The start value of x for the sliding window. Defaults to the minimum of x_values.
        boundary (float, optional): The upper boundary of x for the sliding window. Defaults to the maximum of x_values.

    Returns:
        x_bins (list): The center of each sliding window.
        y_means (list): The mean of y in each sliding window.
        y_percentiles (dict): The calculated percentiles of y in each sliding window. The keys are the string representation of the percentiles.
    """
    x_bins = []
    y_means = []
    y_percentiles = {str(p): [] for p in percentiles}

    if start is None:
        start = x_values.min()
    if boundary is None:
        boundary = x_values.max()

    end = start + bin_size

    while end <= boundary:

        mask = (x_values >= start) & (x_values < end)

        y_in_bin = y_values[mask]

        y_mean = np.nanmean(y_in_bin)
        for p in percentiles:
            y_percentile = np.nanpercentile(y_in_bin, p)
            y_percentiles[str(p)].append(y_percentile)

        x_bins.append((start + end) / 2)
        y_means.append(y_mean)

        start += slide_size
        end += slide_size

    return x_bins, y_means, y_percentiles


def running_median_errorbar(x, y, method="percentile", bins=20):
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
    median_results = binned_statistic(x, y, statistic="median", bins=bins)

    if method == "std":
        std_results = binned_statistic(x, y, statistic="std", bins=bins).statistic

        return median_results.statistic, std_results.statistic

    elif method == "percentile":
        percentile_results_16 = running_percentile(x, y, 16, bins=bins)
        percentile_results_84 = running_percentile(x, y, 84, bins=bins)

        errorbar_low = median_results.statistic - percentile_results_16
        errorbar_high = percentile_results_84 - median_results.statistic

        return median_results.statistic, errorbar_low, errorbar_high


def match_sample(
    standard_sample,
    match_sample,
    standard_sample_match_property,
    match_sample_match_property,
    bins,
    alpha,
    seed=None,
):
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

    print("The number of standard sample: ", len(standard_sample))
    print("The number of match sample: ", len(match_sample))
    ids = np.arange(len(match_sample))

    binned_count = binned_statistic(
        standard_sample_match_property,
        standard_sample_match_property,
        statistic="count",
        bins=bins,
    )[0]
    extract_count = (binned_count * alpha).astype(int)

    indices = np.digitize(match_sample_match_property, bins)
    print(len(indices))

    sample = []
    for i in range(1, len(bins)):
        mask = indices == i
        sub_ids = np.random.choice(ids[mask], extract_count[i - 1], replace=False)
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


def gaussian_tension_approxiation(chi2, dof=20, x0=20):
    """This is for calculating equivalent gaussian sigma for a extremely low p-value due to large chi2 value. From Haslbauer et al. 2022.

    Args:
        chi2 (_type_): _description_
        dof (int, optional): _description_. Defaults to 20.
        x0 (int, optional): _description_. Defaults to 20.

    Returns:
        _type_: _description_
    """

    from scipy.optimize import fsolve

    right_chi2 = (
        chi2 ** (dof / 2 - 1)
        * np.exp(-chi2 / 2)
        / 2 ** (dof / 2)
        / np.math.factorial(dof / 2 - 1)
        / (1 / 2 - (dof / 2 - 1) / chi2)
    )
    right_chi2 = (
        (dof / 2 - 1) * np.log(chi2)
        - chi2 / 2
        - np.log(2 ** (dof / 2))
        - np.log(np.math.factorial(dof / 2 - 1))
        - np.log(1 / 2 - (dof / 2 - 1) / chi2)
    )
    print("right_chi2 = ", right_chi2)

    def func(x):
        return np.log(np.sqrt(2 / np.pi)) + (-(x**2) - np.log(x)) - (right_chi2)

    if x0 is None:
        x0 = 20

    root = fsolve(func, x0)

    return root[0] * np.sqrt(2)


def gaussian_tension(chisqval, dof=19, x0=5, usetype="h22"):
    from scipy.integrate import quad
    from scipy.optimize import fsolve
    from scipy.stats import chi2, norm

    if usetype == "h22":

        p = 1 - chi2.cdf(chisqval, dof)
        print(f"The p-value for {chisqval} is: ", p)

        def f(x):
            return (
                1
                - 1 / np.sqrt(2 * np.pi) * quad(lambda t: np.exp(-(t**2) / 2), -x, x)[0]
                - p
            )

        x = fsolve(f, x0)

        # print('The tension:', x[0])

        return x[0]

    elif usetype == "mine":

        p = 1 - chi2.cdf(chisqval, dof)
        print(f"The p-value for {chisqval} is: ", p)

        x = norm.ppf(1 - p / 2)

        return x


def weighted_tng50(
    q_obs, logM_obs, q_tng50, logM_tng50, index_obs, index_tng50, mass_bins, q_bins
):
    from scipy.stats import binned_statistic

    # 1. divide the data into different bins
    # mass_bins = np.arange(10, 11.5001, 0.15)
    # print('mass_bins', mass_bins)
    # q_bins = np.arange(0, 1.0001, 0.05)
    # print('q_bins', q_bins)
    q_bins_center = (q_bins[1:] + q_bins[:-1]) / 2
    index_q_bins = [
        (q_obs > q_bins[i]) & (q_obs <= q_bins[i + 1]) for i in range(len(q_bins) - 1)
    ]

    # calculate the weight_obs based the numbers in different mass bins.
    N_sim = binned_statistic(
        logM_tng50[index_tng50],
        logM_tng50[index_tng50],
        statistic="count",
        bins=mass_bins,
    )[0]
    N_obs = binned_statistic(
        logM_obs[index_obs], logM_obs[index_obs], statistic="count", bins=mass_bins
    )[0]
    weight_obs = N_sim / N_obs
    print("Simulated number", np.sum(N_sim))
    print("Observed number", np.sum(N_obs))
    # print('weight_obs', weight_obs)

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
            sigma_obs_bin = (
                weight_max_arr[i]
                / weight_total
                * np.sqrt(weight_total_arr[i] / weight_max_arr[i] + 1)
            )
            sigma_obs_arr.append(sigma_obs_bin)

    sigma_obs_arr = np.array(sigma_obs_arr)

    # calculate the sigma_model_i and unweighted observations for each bin based on the poisson uncertainty
    N_model_qbins = np.array(
        binned_statistic(
            q_tng50[index_tng50], q_tng50[index_tng50], statistic="count", bins=q_bins
        )[0]
    )
    N_obs_qbins = binned_statistic(
        q_obs[index_obs], q_obs[index_obs], statistic="count", bins=q_bins
    )[0]
    sigma_model_arr = np.array(
        [
            np.sqrt(N_model_qbins[i] + 1) / np.sum(N_model_qbins)
            for i in range(len(N_model_qbins))
        ]
    )
    sigma_obs_arr_unweighted = np.array(
        [
            np.sqrt(N_obs_qbins[i] + 1) / np.sum(N_obs_qbins)
            for i in range(len(N_obs_qbins))
        ]
    )

    # calculate the chi-square.
    chi_square_bin_arr = (
        weight_total_arr / np.sum(weight_total_arr)
        - N_model_qbins / np.sum(N_model_qbins)
    ) ** 2 / (sigma_model_arr**2 + sigma_obs_arr**2)
    chi_square = np.nansum(chi_square_bin_arr)

    # Combine all derived information into a dictory
    dict_info = {
        "q_bins_center": q_bins_center,
        "weight_total_arr": weight_total_arr,
        "weight_max_arr": weight_max_arr,
        "N_sim": N_sim,
        "N_obs": N_obs,
        "N_model_qbins": N_model_qbins,
        "N_obs_qbins": N_obs_qbins,
        "sigma_obs_arr": sigma_obs_arr,
        "sigma_model_arr": sigma_model_arr,
        "sigma_obs_arr_unweighted": sigma_obs_arr_unweighted,
        "chi-square": chi_square,
    }

    print("Ntotal in loop", N_total)
    print("total weight (should equal Nsim)", np.sum(weight_total_arr))

    return dict_info


def binedge_equalnumber(data, nbins):
    percentiles = np.linspace(0, 100, nbins)
    bins = np.percentile(data, percentiles)

    bins_center = (bins[:-1] + bins[1:]) / 2

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


def is_point_in_ellipse_ring(
    center, outer_width, outer_height, inner_width, inner_height, angle, test_point
):
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
    ellipse = Ellipse(
        xy=center,
        width=outer_width,
        height=outer_height,
        angle=angle,
        facecolor="none",
        edgecolor="blue",
    )
    inner_ellipse = Ellipse(
        xy=center,
        width=inner_width,
        height=inner_height,
        angle=angle,
        facecolor="none",
        edgecolor="green",
    )

    # Check if the test point is inside the ellipse ring
    is_inside = np.logical_and(
        ellipse.contains_point(test_point),
        np.logical_not(inner_ellipse.contains_point(test_point)),
    )

    return is_inside


def match_sample_tolerance_mass(
    sample_tng,
    hsc_pool,
    N_control,
    mass_limit,
    name_mass_tng="Mstar",
    name_mass_hsc="logmstar_addbias",
    name_specz_hsc="spec_redshift",
    z_tng=0.05,
    with_replacement=False,
):
    """
    Generate a matched control sample for a given tng galaxy, based on the stellar mass and redshift. The algorithm is from Connor Bottrell.
    """

    sample_tng_retain = copy.deepcopy(sample_tng)
    sample_mass_tng = sample_tng[name_mass_tng].to_numpy()
    # hsc_pool: full hsc sample from which to match
    hsc_pool = copy.deepcopy(hsc_pool)

    # N: number of hsc controls you want per tng galaxy
    N_control = N_control
    # mass_limit: absolute match tolerance limit
    mass_limit = mass_limit

    # matches_all: a list to store all the matches
    matches_all = pd.DataFrame()
    n_grows_arr = []

    # Loop over each tng galaxy
    for idx in range(len(sample_mass_tng)):
        # z: tng galaxy redshift
        z = z_tng  # or 0?

        # M: tng galaxy stellar mass
        mass = sample_mass_tng[idx]

        # matches_sample: a list to store the best matches for the current tng galaxy
        matches_sample = pd.DataFrame()

        # match_pool: hsc_pool with |z_hsc - z| < 0.05
        match_pool = hsc_pool[np.abs(hsc_pool[name_specz_hsc] - z) < 0.05]

        # n_grows: growth factor for matching tolerance
        n_grows = 1

        # mass_tolerance: initial matching tolerance
        mass_tolerance = 0.05

        # Loop until the matching tolerance reaches the absolute limit
        while n_grows * mass_tolerance < mass_limit:
            # pool2: match_pool where |M_hsc - M| < n_grows * mass_tolerance
            pool2 = match_pool[
                np.abs(match_pool[name_mass_hsc] - mass) < n_grows * mass_tolerance
            ]

            # Find the N best matches to M from pool2
            if len(pool2) >= N_control:
                matches_sample = pool2.iloc[
                    np.argsort(np.abs(pool2[name_mass_hsc] - mass))[:N_control]
                ]
            else:
                n_grows += 1
                continue
            # Exit the loop if we have found N or more matches
            break

        # Remove matched galaxies from the hsc_pool
        if with_replacement is False:
            hsc_pool = hsc_pool[~hsc_pool.index.isin(matches_sample.index)]

        # Remove the tng galaxy from the sample if there are no matches
        if len(matches_sample) == 0:
            sample_tng_retain.drop(sample_tng.index[idx], inplace=True)

        # Add the matches to matches_all
        matches_all = pd.concat([matches_all, matches_sample])
        n_grows_arr.append(n_grows)

    return sample_tng_retain, matches_all, n_grows_arr


def determine_late(V_J, U_V, z=2):
    """This is to determine the star-forming galaxies based on Williams et al. 2009.

    Args:
        V_J (_type_): _description_
        U_V (_type_): _description_
        z (int, optional): _description_. Defaults to 2.

    Returns:
        _type_: _description_
    """
    slop = 0.88
    if (z >= 0) & (z <= 0.5):
        b = 0.69
    if (z > 0.5) & (z <= 1):
        b = 0.59
    if (z > 1) & (z <= 2):
        b = 0.49

    i_1 = U_V < 1.3
    i_2 = V_J > 1.6
    i_3 = U_V < slop * V_J + b
    i_12 = np.logical_or(i_1, i_2)
    index = np.logical_or(i_12, i_3)

    return index

def normalize_vector(vector):
    """
    Normalize a vector.

    Parameters:
    vector (array-like): Vector to be normalized.

    Returns:
    array-like: Normalized vector.
    """
    # Calculate the magnitude of the vector
    magnitude = np.linalg.norm(vector)
    
    # Normalize the vector
    normalized_vector = vector / magnitude
    
    return normalized_vector

def calculate_cross_product_and_unit_normal(vector_a, vector_b):
    """
    Calculate the cross product and unit normal vector of two vectors.

    Parameters:
    vector_a (array-like): First vector.
    vector_b (array-like): Second vector.

    Returns:
    tuple: Cross product and unit normal vector.
    """
    # Calculate the cross product
    cross_product = np.cross(vector_a, vector_b)

    # Calculate the unit normal vector
    unit_normal_vector = normalize_vector(cross_product)
    
    return cross_product, unit_normal_vector

def projected_vector_onplane(vector, plane_vector1, plane_vector2):
    """
    Project a vector onto a plane defined by two vectors.

    Parameters:
    vector (array-like): Vector to be projected.
    plane_vector1 (array-like): First vector defining the plane.
    plane_vector2 (array-like): Second vector defining the plane.

    Returns:
    array-like: Projected vector.
    """
    # Calculate the cross product of the two vectors defining the plane
    cross_product = np.cross(plane_vector1, plane_vector2)
    
    # Calculate the projection of the vector onto the plane, normalizing the cross product
    projection = vector - np.dot(vector, cross_product) * cross_product / np.dot(cross_product, cross_product)
    
    return projection

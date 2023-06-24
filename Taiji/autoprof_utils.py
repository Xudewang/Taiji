import copy

import numpy as np
from astropy.table import Table
from scipy.optimize import minimize
from scipy.stats import iqr, norm

from Taiji.constant import pixel_scale_HSCSSP
from Taiji.imtools import (GrowthCurve, bright_to_mag, get_Rmag, get_Rpercent,
                           remove_consecutive, symmetry_propagate_err_mu)


def Smooth_Mode(v):
    # set the starting point for the optimization at the median
    start = np.median(v)
    # set the smoothing scale equal to roughly 0.5% of the width of the data
    scale = iqr(v) / max(1.0, np.log10(len(v)))  # /10
    # Fit the peak of the smoothed histogram
    res = minimize(
        lambda x: -np.sum(np.exp(-(((v - x) / scale)**2))),
        x0=[start],
        method="Nelder-Mead",
    )
    return res.x[0]


def Background_Mode(IMG, mask=None):
    # Mask main body of image so only outer 1/5th is used
    # for background calculation.
    if mask is not None:
        mask = np.logical_not(mask)
    else:
        mask = np.ones(IMG.shape, dtype=bool)
        mask[int(IMG.shape[0] / 5.0):int(4.0 * IMG.shape[0] / 5.0),
             int(IMG.shape[1] / 5.0):int(4.0 * IMG.shape[1] / 5.0), ] = False

    values = IMG[mask].flatten()

    if len(values) < 1e5:
        values = IMG.flatten()

    values = values[np.isfinite(values)]

    # # Fit the peak of the smoothed histogram
    bkgrnd = Smooth_Mode(values)

    # Compute the 1sigma range using negative flux values, which should almost exclusively be sky noise

    noise = iqr(values[(values - bkgrnd) < 0], rng=[100 - 68.2689492137, 100])
    if not np.isfinite(noise):
        noise = iqr(values, rng=[16, 84]) / 2.0

    uncertainty = noise / np.sqrt(np.sum((values - bkgrnd) < 0))
    if not np.isfinite(uncertainty):
        uncertainty = noise / np.sqrt(len(values))

    return {
        "background": bkgrnd,
        "background noise": noise,
        "background uncertainty": uncertainty
    }


def My_Background_Mode(IMG, mask=None):
    # TODO: wrong, should modify.
    # Mask main body of image so only outer 1/5th is used
    # for background calculation.
    if mask is not None:
        IMG_copy = copy.deepcopy(IMG)
        IMG_copy[mask == 1] = np.nan
        values = IMG_copy.flatten()
    else:
        raise ValueError('The mask is empty!')

    if len(values) < 1e5:
        values = IMG.flatten()

    values = values[np.isfinite(values)]

    # # Fit the peak of the smoothed histogram
    bkgrnd = Smooth_Mode(values)

    # Compute the 1sigma range using negative flux values, which should almost exclusively be sky noise

    noise = iqr(values[(values - bkgrnd) < 0], rng=[100 - 68.2689492137, 100])
    if not np.isfinite(noise):
        noise = iqr(values, rng=[16, 84]) / 2.0

    return {"background": bkgrnd, "background noise": noise}


def get_geometry_ap(ap_prof_name, ap_aux_name, pixel_scale, zpt0):
    import os

    import pandas as pd
    from scipy.interpolate import interp1d
    
    if os.path.exists(ap_prof_name) and os.path.exists(ap_aux_name):
        with open(ap_aux_name, "r") as file_aux:
            num_pass = 0
            for line in file_aux:
                if line.startswith("checkfit"):
                    num_pass = num_pass + 1 if line.strip(
                    )[-4:] == "pass" else num_pass

                if line.startswith("checkfit FFT"):
                    check_fit_fft_coefficients = 1 if line.strip(
                    )[-4:] == "pass" else 0
                if line.startswith("checkfit Light"):
                    check_fit_light_symmetry = 1 if line.strip(
                    )[-4:] == "pass" else 0
                if line.startswith("checkfit initial"):
                    check_fit_initial_fit_compare = 1 if line.strip(
                    )[-4:] == "pass" else 0
                if line.startswith("checkfit isophote"):
                    check_fit_isophote_variability = 1 if line.strip(
                    )[-4:] == "pass" else 0

                if line.startswith("global ellipticity") and num_pass >= 0:
                    ell, err_ell, pa, err_pa, rad_pix = np.array(
                        line.split())[[2, 4, 6, 8, 11]]
                    if err_ell[-1] == ",":
                        err_ell = err_ell[:-1]
                    else:
                        print("error!")
                        err_ell = np.nan

                if line.startswith('background'):
                    noise = float(line.split()[-2])
                if line.startswith('central surface'):
                    central_mu = float(line.split()[3])

        data_ap = pd.read_csv(ap_prof_name, header=1)
        data_ap = Table.from_pandas(data_ap)
        sma_ap = data_ap['R'] / pixel_scale
        I_ap = data_ap['I']
        iso_err_ap = data_ap['I_e']
        ell_ap = data_ap['ellip']
        ell_err_ap = data_ap['ellip_e']
        pa_ap = data_ap['pa']
        pa_err_ap = data_ap['pa_e']

        I_err_ap = np.sqrt(iso_err_ap**2 + noise**2 / pixel_scale**4)
        mu_ap = bright_to_mag(intens=I_ap, texp=1, pixel_size=1, zpt0=zpt0)
        mu_err_ap = symmetry_propagate_err_mu(I_ap, I_err_ap)
        tflux = data_ap['totflux_direct']
        maxIsoFlux = np.max(tflux)

        index_above_sigma_temp = (I_ap > 3 * I_err_ap)

        index_above_sigma_fix = remove_consecutive(sma_ap,
                                                   index_above_sigma_temp)[0]

        # calculate the CoG by myself
        # tflux_ap_me, maxIsoSma_ap_me, maxIsoFlux_ap_me = GrowthCurve(
        #     sma_ap[index_above_sigma_fix], ell_ap[index_above_sigma_fix],
        #     I_ap[index_above_sigma_fix] * pixel_scale**2)

        # get the Rpercent radius and use this to calculate the average ellipticity and PA.
        r20 = get_Rpercent(sma_ap, tflux, maxIsoFlux, 0.2)
        r50 = get_Rpercent(sma_ap, tflux, maxIsoFlux, 0.5)
        r80 = get_Rpercent(sma_ap, tflux, maxIsoFlux, 0.8)
        r90 = get_Rpercent(sma_ap, tflux, maxIsoFlux, 0.9)
        r95 = get_Rpercent(sma_ap, tflux, maxIsoFlux, 0.95)

        R25 = get_Rmag(sma_ap, mu=mu_ap, mag_use=25)

        # get the surface brightness at rad_pix, need to interpolate the sma versus mu
        mu_interp = interp1d(sma_ap, mu_ap, kind='linear')
        mu_rad_pix = mu_interp(rad_pix)

        # get the ellipticity at different radius, need to interpolate the sma versus ell
        ell_interp = interp1d(sma_ap, ell_ap, kind='linear')
        pa_interp = interp1d(sma_ap, pa_ap, kind='linear')
        ell_R25 = ell_interp(R25)
        pa_R25 = pa_interp(R25)
        
        # derive the average ellipticity and PA within k1 and k2 times r90.
        k1 = 0.675
        k2 = 1.467
        innerRad = k1 * r90
        outRad = k2 * r90
        
        ell_avg = np.mean(ell_ap[np.logical_and(sma_ap<outRad, sma_ap>innerRad)])
        ell_avg_err = np.sqrt(np.sum(ell_err_ap[np.logical_and(sma_ap<outRad, sma_ap>innerRad)]**2))
        pa_avg = np.mean(pa_ap[np.logical_and(sma_ap<outRad, sma_ap>innerRad)])
        pa_avg_err = np.sqrt(np.sum(pa_err_ap[np.logical_and(sma_ap<outRad, sma_ap>innerRad)]**2))

    else:
        print('No such file!')
        return None

    # save all the information to pandas dataframe
    data_save = {
        'r20': r20,
        'r50': r50,
        'r80': r80,
        'r90': r90,
        'r95': r95,
        'R25': R25,
        'ell_R25': ell_R25,
        'pa_R25': pa_R25,
        'ell_avg': ell_avg,
        'ell_avg_err': ell_avg_err,
        'pa_avg': pa_avg,
        'pa_avg_err': pa_avg_err,
        'ell': ell,
        'err_ell': err_ell,
        'pa': pa,
        'err_pa': err_pa,
        'rad_pix': rad_pix,
        'check_fit_fft_coefficients': check_fit_fft_coefficients,
        'check_fit_light_symmetry': check_fit_light_symmetry,
        'check_fit_initial_fit_compare': check_fit_initial_fit_compare,
        'check_fit_isophote_variability': check_fit_isophote_variability,
        'number_of_pass': num_pass,
        'maxIsoFlux': maxIsoFlux,
        'central_mu': central_mu,
        'mu_rad_pix': mu_rad_pix,
    }

    df_save = pd.DataFrame([data_save], index=[0])

    return df_save

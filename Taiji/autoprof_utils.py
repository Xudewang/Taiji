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

def Isophote_Initialize(IMG, results, options):
    """Fit global elliptical isophote to a galaxy image using FFT coefficients.

    A global position angle and ellipticity are fit in a two step
    process.  First, a series of circular isophotes are geometrically
    sampled until they approach the background level of the image.  An
    FFT is taken for the flux values around each isophote and the
    phase of the second coefficient is used to determine a direction.
    The average direction for the outer isophotes is taken as the
    position angle of the galaxy.  Second, with fixed position angle
    the ellipticity is optimized to minimize the amplitude of the
    second FFT coefficient relative to the median flux in an isophote.

    To compute the error on position angle we use the standard
    deviation of the outer values from step one.  For ellipticity the
    error is computed by optimizing the ellipticity for multiple
    isophotes within 1 PSF length of each other.

    Parameters
    -----------------
    ap_fit_limit : float, default 2
      noise level out to which to extend the fit in units of pixel background noise level. Default is 2, smaller values will end fitting further out in the galaxy image.

    ap_isoinit_pa_set : float, default None
      User set initial position angle in degrees, will override the calculation.

    ap_isoinit_ellip_set : float, default None
      User set initial ellipticity (1 - b/a), will override the calculation.

    Notes
    ----------
    :References:
    - 'background'
    - 'background noise'
    - 'psf fwhm'
    - 'center'

    Returns
    -------
    IMG : ndarray
      Unaltered galaxy image

    results : dict
      .. code-block:: python

        {'init ellip': , # Ellipticity of the global fit (float)
         'init pa': ,# Position angle of the global fit (float)
         'init R': ,# Semi-major axis length of global fit (float)
         'auxfile initialize': # optional, message for aux file to record the global ellipticity and postition angle (string)

        }

    """

    ######################################################################
    # Initial attempt to find size of galaxy in image
    # based on when isophotes SB values start to get
    # close to the background noise level
    circ_ellipse_radii = [1.0]
    allphase = []
    dat = IMG - results["background"]
    mask = results["mask"] if "mask" in results else None
    if not np.any(mask):
        mask = None

    while circ_ellipse_radii[-1] < (len(IMG) / 2):
        circ_ellipse_radii.append(circ_ellipse_radii[-1] * (1 + 0.2))
        isovals = _iso_extract(
            dat,
            circ_ellipse_radii[-1],
            {"ellip": 0.0, "pa": 0.0},
            results["center"],
            more=True,
            mask=mask,
            sigmaclip=True,
            sclip_nsigma=3,
            interp_mask=True,
        )
        coefs = fft(isovals[0])
        allphase.append(coefs[2])
        # Stop when at 3 time background noise
        if (
            np.quantile(isovals[0], 0.8)
            < (
                (options["ap_fit_limit"] + 1 if "ap_fit_limit" in options else 3)
                * results["background noise"]
            )
            and len(circ_ellipse_radii) > 4
        ):
            break
    logging.info(
        "%s: init scale: %f pix" % (options["ap_name"], circ_ellipse_radii[-1])
    )
    # Find global position angle.
    phase = (-Angle_Median(np.angle(allphase[-5:])) / 2) % np.pi
    if "ap_isoinit_pa_set" in options:
        phase = PA_shift_convention(options["ap_isoinit_pa_set"] * np.pi / 180)

    # Find global ellipticity
    test_ellip = np.linspace(0.05, 0.95, 15)
    test_f2 = []
    for e in test_ellip:
        test_f2.append(
            sum(
                list(
                    _fitEllip_loss(
                        e,
                        dat,
                        circ_ellipse_radii[-2] * m,
                        phase,
                        results["center"],
                        results["background noise"],
                        mask,
                    )
                    for m in np.linspace(0.8, 1.2, 5)
                )
            )
        )
    # Find global ellipticity: second pass
    ellip = test_ellip[np.argmin(test_f2)]
    test_ellip = np.linspace(ellip - 0.05, ellip + 0.05, 15)
    test_f2 = []
    for e in test_ellip:
        test_f2.append(
            sum(
                list(
                    _fitEllip_loss(
                        e,
                        dat,
                        circ_ellipse_radii[-2] * m,
                        phase,
                        results["center"],
                        results["background noise"],
                        mask,
                    )
                    for m in np.linspace(0.8, 1.2, 5)
                )
            )
        )
    ellip = test_ellip[np.argmin(test_f2)]
    res = minimize(
        lambda e, d, r, p, c, n, msk: sum(
            list(
                _fitEllip_loss(_x_to_eps(e[0]), d, r * m, p, c, n, msk)
                for m in np.linspace(0.8, 1.2, 5)
            )
        ),
        x0=_inv_x_to_eps(ellip),
        args=(
            dat,
            circ_ellipse_radii[-2],
            phase,
            results["center"],
            results["background noise"],
            mask,
        ),
        method="Nelder-Mead",
        options={
            "initial_simplex": [
                [_inv_x_to_eps(ellip) - 1 / 15],
                [_inv_x_to_eps(ellip) + 1 / 15],
            ]
        },
    )
    if res.success:
        logging.debug(
            "%s: using optimal ellipticity %.3f over grid ellipticity %.3f"
            % (options["ap_name"], _x_to_eps(res.x[0]), ellip)
        )
        ellip = _x_to_eps(res.x[0])
    if "ap_isoinit_ellip_set" in options:
        ellip = options["ap_isoinit_ellip_set"]

    # Compute the error on the parameters
    ######################################################################
    RR = np.linspace(
        circ_ellipse_radii[-2] - results["psf fwhm"],
        circ_ellipse_radii[-2] + results["psf fwhm"],
        10,
    )
    errallphase = []
    for rr in RR:
        isovals = _iso_extract(
            dat,
            rr,
            {"ellip": 0.0, "pa": 0.0},
            results["center"],
            more=True,
            sigmaclip=True,
            sclip_nsigma=3,
            interp_mask=True,
        )
        coefs = fft(isovals[0])
        errallphase.append(coefs[2])
    sample_pas = (
        -np.angle(1j * np.array(errallphase) / np.mean(errallphase)) / 2
    ) % np.pi
    pa_err = iqr(sample_pas, rng=[16, 84]) / 2
    res_multi = map(
        lambda rrp: minimize(
            lambda e, d, r, p, c, n, m: _fitEllip_loss(
                _x_to_eps(e[0]), d, r, p, c, n, m
            ),
            x0=_inv_x_to_eps(ellip),
            args=(
                dat,
                rrp[0],
                rrp[1],
                results["center"],
                results["background noise"],
                mask,
            ),
            method="Nelder-Mead",
            options={
                "initial_simplex": [
                    [_inv_x_to_eps(ellip) - 1 / 15],
                    [_inv_x_to_eps(ellip) + 1 / 15],
                ]
            },
        ),
        zip(RR, sample_pas),
    )
    ellip_err = iqr(list(_x_to_eps(rm.x[0]) for rm in res_multi), rng=[16, 84]) / 2

    circ_ellipse_radii = np.array(circ_ellipse_radii)

    if "ap_doplot" in options and options["ap_doplot"]:
        Plot_Isophote_Init_Ellipse(
            dat, circ_ellipse_radii, ellip, phase, results, options
        )
        Plot_Isophote_Init_Optimize(
            circ_ellipse_radii,
            allphase,
            phase,
            pa_err,
            test_ellip,
            test_f2,
            ellip,
            ellip_err,
            results,
            options,
        )

    auxmessage = (
        "global ellipticity: %.3f +- %.3f, pa: %.3f +- %.3f deg, size: %f pix"
        % (
            ellip,
            ellip_err,
            PA_shift_convention(phase) * 180 / np.pi,
            pa_err * 180 / np.pi,
            circ_ellipse_radii[-2],
        )
    )
    return IMG, {
        "init ellip": ellip,
        "init ellip_err": ellip_err,
        "init pa": phase,
        "init pa_err": pa_err,
        "init R": circ_ellipse_radii[-2],
        "auxfile initialize": auxmessage,
    }

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
                if line.startswith('center'):
                    center_x = float(line.split()[2])
                    center_y = float(line.split()[5])

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
        'center_x': center_x,
        'center_y': center_y,
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

import copy
import logging

import numpy as np
from astropy.table import Table
from photutils.isophote import Ellipse as Photutils_Ellipse
from photutils.isophote import EllipseGeometry, EllipseSample, Isophote, IsophoteList
from scipy.optimize import minimize
from scipy.stats import iqr, norm

from Taiji.autoprof_SharedFunctions import (
    AddLogo,
    Fmode_fluxdens_to_fluxsum_errorprop,
    LSBImage,
    PA_shift_convention,
    SBprof_to_COG_errorprop,
    _average,
    _inv_x_to_eps,
    _inv_x_to_pa,
    _iso_between,
    _iso_extract,
    _scatter,
    _x_to_eps,
    _x_to_pa,
    autocolours,
    flux_to_mag,
    flux_to_sb,
    fluxdens_to_fluxsum_errorprop,
    mag_to_flux,
)
from Taiji.constant import pixel_scale_HSCSSP
from Taiji.imtools import (
    GrowthCurve,
    GrowthCurve_trapz,
    bright_to_mag,
    get_Rmag,
    get_Rpercent,
    remove_consecutive,
    symmetry_propagate_err_mu,
)


def Smooth_Mode(v):
    # set the starting point for the optimization at the median
    start = np.median(v)
    # set the smoothing scale equal to roughly 0.5% of the width of the data
    scale = iqr(v) / max(1.0, np.log10(len(v)))  # /10
    # Fit the peak of the smoothed histogram
    res = minimize(
        lambda x: -np.sum(np.exp(-(((v - x) / scale) ** 2))),
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
        mask[
            int(IMG.shape[0] / 5.0) : int(4.0 * IMG.shape[0] / 5.0),
            int(IMG.shape[1] / 5.0) : int(4.0 * IMG.shape[1] / 5.0),
        ] = False

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
        "background uncertainty": uncertainty,
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
        raise ValueError("The mask is empty!")

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


def get_geometry_ap(ap_prof_name, ap_aux_name, pixel_scale, zpt0, sky_value, sky_rms):
    import os

    import pandas as pd
    from scipy.interpolate import interp1d

    if os.path.exists(ap_prof_name) and os.path.exists(ap_aux_name):
        with open(ap_aux_name, "r") as file_aux:
            num_pass = 0
            for line in file_aux:
                if line.startswith("checkfit"):
                    num_pass = num_pass + 1 if line.strip()[-4:] == "pass" else num_pass

                if line.startswith("checkfit FFT"):
                    check_fit_fft_coefficients = 1 if line.strip()[-4:] == "pass" else 0
                if line.startswith("checkfit Light"):
                    check_fit_light_symmetry = 1 if line.strip()[-4:] == "pass" else 0
                if line.startswith("checkfit initial"):
                    check_fit_initial_fit_compare = (
                        1 if line.strip()[-4:] == "pass" else 0
                    )
                if line.startswith("checkfit isophote"):
                    check_fit_isophote_variability = (
                        1 if line.strip()[-4:] == "pass" else 0
                    )

                if line.startswith("global ellipticity") and num_pass >= 0:
                    ell, err_ell, pa, err_pa, rad_pix = np.array(line.split())[
                        [2, 4, 6, 8, 11]
                    ]
                    if err_ell[-1] == ",":
                        err_ell = err_ell[:-1]
                    else:
                        print("error!")
                        err_ell = np.nan

                if line.startswith("background"):
                    noise = float(line.split()[-2])
                if line.startswith("central surface"):
                    central_mu = float(line.split()[3])
                if line.startswith("center"):
                    center_x = float(line.split()[2])
                    center_y = float(line.split()[5])
                if line.startswith("option ap_set_center"):
                    center_x = np.nan
                    center_y = np.nan

        data_ap = pd.read_csv(ap_prof_name, header=1)
        data_ap = Table.from_pandas(data_ap)
        sma_ap = data_ap["R"] / pixel_scale
        I_ap = data_ap["I"]
        iso_err_ap = data_ap["I_e"]
        ell_ap = data_ap["ellip"]
        ell_err_ap = data_ap["ellip_e"]
        pa_ap = data_ap["pa"]
        pa_err_ap = data_ap["pa_e"]

        I_err_ap = np.sqrt(iso_err_ap**2 + sky_rms**2 / pixel_scale**4)
        mu_ap = bright_to_mag(intens=I_ap, texp=1, pixel_size=1, zpt0=zpt0)
        mu_err_ap = symmetry_propagate_err_mu(I_ap, I_err_ap)

        index_above_sigma_temp = I_ap > 3 * I_err_ap

        index_above_sigma_fix = remove_consecutive(sma_ap, index_above_sigma_temp)[0]

        # calculate the CoG by myself
        tflux = GrowthCurve_trapz(
            R=sma_ap[index_above_sigma_fix],
            I=ell_ap[index_above_sigma_fix],
            axisratio=I_ap[index_above_sigma_fix] * pixel_scale**2,
        )

        maxIsoFlux = np.max(tflux)

        # get the Rpercent radius and use this to calculate the average ellipticity and PA.
        r20 = get_Rpercent(sma_ap[index_above_sigma_fix], tflux, maxIsoFlux, 0.2)
        r50 = get_Rpercent(sma_ap[index_above_sigma_fix], tflux, maxIsoFlux, 0.5)
        r80 = get_Rpercent(sma_ap[index_above_sigma_fix], tflux, maxIsoFlux, 0.8)
        r90 = get_Rpercent(sma_ap[index_above_sigma_fix], tflux, maxIsoFlux, 0.9)
        r95 = get_Rpercent(sma_ap[index_above_sigma_fix], tflux, maxIsoFlux, 0.95)

        R25 = get_Rmag(sma_ap, mu=mu_ap, mag_use=25)

        # get the surface brightness at rad_pix, need to interpolate the sma versus mu
        mu_interp = interp1d(sma_ap, mu_ap, kind="linear")
        mu_rad_pix = mu_interp(rad_pix)

        # get the ellipticity at different radius, need to interpolate the sma versus ell
        ell_interp = interp1d(sma_ap, ell_ap, kind="linear")
        pa_interp = interp1d(sma_ap, pa_ap, kind="linear")
        ell_R25 = ell_interp(R25)
        pa_R25 = pa_interp(R25)

        # derive the average ellipticity and PA within k1 and k2 times r90.
        k1 = 0.675
        k2 = 1.467
        innerRad = k1 * r90
        outRad = k2 * r90

        ell_avg = np.mean(ell_ap[np.logical_and(sma_ap < outRad, sma_ap > innerRad)])
        ell_avg_err = np.sqrt(
            np.sum(ell_err_ap[np.logical_and(sma_ap < outRad, sma_ap > innerRad)] ** 2)
        )
        pa_avg = np.mean(pa_ap[np.logical_and(sma_ap < outRad, sma_ap > innerRad)])
        pa_avg_err = np.sqrt(
            np.sum(pa_err_ap[np.logical_and(sma_ap < outRad, sma_ap > innerRad)] ** 2)
        )

    else:
        print("No such file!")
        return None

    # save all the information to pandas dataframe

    data_save = {
        "center_x": center_x,
        "center_y": center_y,
        "r20": r20,
        "r50": r50,
        "r80": r80,
        "r90": r90,
        "r95": r95,
        "R25": R25,
        "ell_R25": ell_R25,
        "pa_R25": pa_R25,
        "ell_avg": ell_avg,
        "ell_avg_err": ell_avg_err,
        "pa_avg": pa_avg,
        "pa_avg_err": pa_avg_err,
        "ell": ell,
        "err_ell": err_ell,
        "pa": pa,
        "err_pa": err_pa,
        "rad_pix": rad_pix,
        "check_fit_fft_coefficients": check_fit_fft_coefficients,
        "check_fit_light_symmetry": check_fit_light_symmetry,
        "check_fit_initial_fit_compare": check_fit_initial_fit_compare,
        "check_fit_isophote_variability": check_fit_isophote_variability,
        "number_of_pass": num_pass,
        "maxIsoFlux": maxIsoFlux,
        "central_mu": central_mu,
        "mu_rad_pix": mu_rad_pix,
    }

    df_save = pd.DataFrame([data_save], index=[0])

    return df_save


def _Generate_Profile(IMG, results, Radius, parameters, options):

    # Create image array with background and mask applied
    try:
        if np.any(results["mask"]):
            mask = results["mask"]
        else:
            mask = None
    except:
        mask = None
    dat = IMG - results["background"]
    zeropoint = options["ap_zeropoint"] if "ap_zeropoint" in options else 22.5
    fluxunits = options["ap_fluxunits"] if "ap_fluxunits" in options else "mag"

    for p in range(len(parameters)):
        # Indicate no Fourier modes if supplied parameters does not include it
        if not "m" in parameters[p]:
            parameters[p]["m"] = None
        if not "C" in parameters[p]:
            parameters[p]["C"] = None
        # If no ellipticity error supplied, assume zero
        if not "ellip err" in parameters[p]:
            parameters[p]["ellip err"] = 0.0
        # If no position angle error supplied, assume zero
        if not "pa err" in parameters[p]:
            parameters[p]["pa err"] = 0.0

    sb = []
    sbE = []
    pixels = []
    maskedpixels = []
    cogdirect = []
    sbfix = []
    sbfixE = []
    measFmodes = []

    count_neg = 0
    medflux = np.inf
    end_prof = len(Radius)
    compare_interp = []
    for i in range(len(Radius)):
        if "ap_isoband_fixed" in options and options["ap_isoband_fixed"]:
            isobandwidth = (
                options["ap_isoband_width"] if "ap_isoband_width" in options else 0.5
            )
        else:
            isobandwidth = Radius[i] * (
                options["ap_isoband_width"] if "ap_isoband_width" in options else 0.025
            )
        isisophoteband = False
        if (
            medflux
            > (
                results["background noise"]
                * (options["ap_isoband_start"] if "ap_isoband_start" in options else 2)
            )
            or isobandwidth < 0.5
        ):
            isovals = _iso_extract(
                dat,
                Radius[i],
                parameters[i],
                results["center"],
                mask=mask,
                more=True,
                rad_interp=(
                    options["ap_iso_interpolate_start"]
                    if "ap_iso_interpolate_start" in options
                    else 5
                )
                * results["psf fwhm"],
                interp_method=(
                    options["ap_iso_interpolate_method"]
                    if "ap_iso_interpolate_method" in options
                    else "lanczos"
                ),
                interp_window=(
                    int(options["ap_iso_interpolate_window"])
                    if "ap_iso_interpolate_window" in options
                    else 5
                ),
                sigmaclip=options["ap_isoclip"] if "ap_isoclip" in options else False,
                sclip_iterations=(
                    options["ap_isoclip_iterations"]
                    if "ap_isoclip_iterations" in options
                    else 10
                ),
                sclip_nsigma=(
                    options["ap_isoclip_nsigma"]
                    if "ap_isoclip_nsigma" in options
                    else 5
                ),
            )
        else:
            isisophoteband = True
            isovals = _iso_between(
                dat,
                Radius[i] - isobandwidth,
                Radius[i] + isobandwidth,
                parameters[i],
                results["center"],
                mask=mask,
                more=True,
                sigmaclip=options["ap_isoclip"] if "ap_isoclip" in options else False,
                sclip_iterations=(
                    options["ap_isoclip_iterations"]
                    if "ap_isoclip_iterations" in options
                    else 10
                ),
                sclip_nsigma=(
                    options["ap_isoclip_nsigma"]
                    if "ap_isoclip_nsigma" in options
                    else 5
                ),
            )
        isotot = np.sum(
            _iso_between(dat, 0, Radius[i], parameters[i], results["center"], mask=mask)
        )
        medflux = _average(
            isovals[0],
            (
                options["ap_isoaverage_method"]
                if "ap_isoaverage_method" in options
                else "median"
            ),
        )
        scatflux = _scatter(
            isovals[0],
            (
                options["ap_isoaverage_method"]
                if "ap_isoaverage_method" in options
                else "median"
            ),
        )
        if (
            "ap_iso_measurecoefs" in options
            and not options["ap_iso_measurecoefs"] is None
        ):
            if (
                mask is None
                and (not "ap_isoclip" in options or not options["ap_isoclip"])
                and not isisophoteband
            ):
                coefs = fft(isovals[0])
            else:
                N = max(15, int(0.9 * 2 * np.pi * Radius[i]))
                theta = np.linspace(0, 2 * np.pi * (1.0 - 1.0 / N), N)
                coefs = fft(np.interp(theta, isovals[1], isovals[0], period=2 * np.pi))
            measFmodes.append(
                {
                    "a": [np.imag(coefs[0]) / len(coefs)]
                    + list(
                        np.imag(coefs[np.array(options["ap_iso_measurecoefs"])])
                        / (np.abs(coefs[0]))
                    ),
                    "b": [np.real(coefs[0]) / len(coefs)]
                    + list(
                        np.real(coefs[np.array(options["ap_iso_measurecoefs"])])
                        / (np.abs(coefs[0]))
                    ),
                }
            )

        pixels.append(len(isovals[0]))
        maskedpixels.append(isovals[2])
        if fluxunits == "intensity":
            sb.append(medflux / options["ap_pixscale"] ** 2)
            sbE.append(scatflux / np.sqrt(len(isovals[0])))
            cogdirect.append(isotot)
        else:
            sb.append(
                flux_to_sb(medflux, options["ap_pixscale"], zeropoint)
                if medflux > 0
                else 99.999
            )
            sbE.append(
                (2.5 * scatflux / (np.sqrt(len(isovals[0])) * medflux * np.log(10)))
                if medflux > 0
                else 99.999
            )
            cogdirect.append(flux_to_mag(isotot, zeropoint) if isotot > 0 else 99.999)
        if medflux <= 0:
            count_neg += 1
        if (
            "ap_truncate_evaluation" in options
            and options["ap_truncate_evaluation"]
            and count_neg >= 2
        ):
            end_prof = i + 1
            break

    # Compute Curve of Growth from SB profile
    if fluxunits == "intensity":
        cog, cogE = Fmode_fluxdens_to_fluxsum_errorprop(
            Radius[:end_prof] * options["ap_pixscale"],
            np.array(sb),
            np.array(sbE),
            parameters[:end_prof],
            N=100,
            symmetric_error=True,
        )

        if cog is None:
            cog = -99.999 * np.ones(len(Radius))
            cogE = -99.999 * np.ones(len(Radius))
        else:
            cog[np.logical_not(np.isfinite(cog))] = -99.999
            cogE[cog < 0] = -99.999
    else:
        cog, cogE = SBprof_to_COG_errorprop(
            Radius[:end_prof] * options["ap_pixscale"],
            np.array(sb),
            np.array(sbE),
            parameters[:end_prof],
            N=100,
            symmetric_error=True,
        )
        if cog is None:
            cog = 99.999 * np.ones(len(Radius))
            cogE = 99.999 * np.ones(len(Radius))
        else:
            cog[np.logical_not(np.isfinite(cog))] = 99.999
            cogE[cog > 99] = 99.999

    # For each radius evaluation, write the profile parameters
    if fluxunits == "intensity":
        params = [
            "R",
            "I",
            "I_e",
            "totflux",
            "totflux_e",
            "ellip",
            "ellip_e",
            "pa",
            "pa_e",
            "pixels",
            "maskedpixels",
            "totflux_direct",
        ]

        SBprof_units = {
            "R": "arcsec",
            "I": "flux*arcsec^-2",
            "I_e": "flux*arcsec^-2",
            "totflux": "flux",
            "totflux_e": "flux",
            "ellip": "unitless",
            "ellip_e": "unitless",
            "pa": "deg",
            "pa_e": "deg",
            "pixels": "count",
            "maskedpixels": "count",
            "totflux_direct": "flux",
        }
    else:
        params = [
            "R",
            "SB",
            "SB_e",
            "totmag",
            "totmag_e",
            "ellip",
            "ellip_e",
            "pa",
            "pa_e",
            "pixels",
            "maskedpixels",
            "totmag_direct",
        ]

        SBprof_units = {
            "R": "arcsec",
            "SB": "mag*arcsec^-2",
            "SB_e": "mag*arcsec^-2",
            "totmag": "mag",
            "totmag_e": "mag",
            "ellip": "unitless",
            "ellip_e": "unitless",
            "pa": "deg",
            "pa_e": "deg",
            "pixels": "count",
            "maskedpixels": "count",
            "totmag_direct": "mag",
        }

    SBprof_data = dict((h, None) for h in params)
    SBprof_data["R"] = list(Radius[:end_prof] * options["ap_pixscale"])
    SBprof_data["I" if fluxunits == "intensity" else "SB"] = list(sb)
    SBprof_data["I_e" if fluxunits == "intensity" else "SB_e"] = list(sbE)
    SBprof_data["totflux" if fluxunits == "intensity" else "totmag"] = list(cog)
    SBprof_data["totflux_e" if fluxunits == "intensity" else "totmag_e"] = list(cogE)
    SBprof_data["ellip"] = list(parameters[p]["ellip"] for p in range(end_prof))
    SBprof_data["ellip_e"] = list(parameters[p]["ellip err"] for p in range(end_prof))
    SBprof_data["pa"] = list(parameters[p]["pa"] * 180 / np.pi for p in range(end_prof))
    SBprof_data["pa_e"] = list(
        parameters[p]["pa err"] * 180 / np.pi for p in range(end_prof)
    )
    SBprof_data["pixels"] = list(pixels)
    SBprof_data["maskedpixels"] = list(maskedpixels)
    SBprof_data["totflux_direct" if fluxunits == "intensity" else "totmag_direct"] = (
        list(cogdirect)
    )

    if "ap_iso_measurecoefs" in options and not options["ap_iso_measurecoefs"] is None:
        whichcoefs = [0] + list(options["ap_iso_measurecoefs"])
        for i in list(range(len(whichcoefs))):
            aa, bb = "a%i" % whichcoefs[i], "b%i" % whichcoefs[i]
            params += [aa, bb]
            SBprof_units.update(
                {
                    aa: "flux" if whichcoefs[i] == 0 else "a%i/F0" % whichcoefs[i],
                    bb: "flux" if whichcoefs[i] == 0 else "b%i/F0" % whichcoefs[i],
                }
            )
            SBprof_data[aa] = list(F["a"][i] for F in measFmodes)
            SBprof_data[bb] = list(F["b"][i] for F in measFmodes)

    if any(not p["m"] is None for p in parameters):
        for m in range(len(parameters[0]["m"])):
            AA, PP = "A%i" % parameters[0]["m"][m], "Phi%i" % parameters[0]["m"][m]
            params += [AA, PP]
            SBprof_units.update({AA: "unitless", PP: "deg"})
            SBprof_data[AA] = list(p["Am"][m] for p in parameters[:end_prof])
            SBprof_data[PP] = list(p["Phim"][m] for p in parameters[:end_prof])
    if any(not p["C"] is None for p in parameters):
        params += ["C"]
        SBprof_units["C"] = "unitless"
        SBprof_data["C"] = list(p["C"] for p in parameters[:end_prof])

    return {"prof header": params, "prof units": SBprof_units, "prof data": SBprof_data}


def Isophote_Extract_Photutils(IMG, results, options):
    """Wrapper of photutils method for extracting SB profiles.

    This simply gives users access to the photutils isophote
    extraction methods. The one exception is that SB values are taken
    as the median instead of the mean, as recomended in the photutils
    documentation. See: `photutils
    <https://photutils.readthedocs.io/en/stable/isophote.html>`_ for
    more information.

    Parameters
    ----------
    ap_zeropoint : float, default 22.5
      Photometric zero point. For converting flux to mag units.

    ap_fluxunits : str, default "mag"
      units for outputted photometry. Can either be "mag" for log
      units, or "intensity" for linear units.

    ap_plot_sbprof_ylim : tuple, default None
      Tuple with axes limits for the y-axis in the SB profile
      diagnostic plot. Be careful when using intensity units
      since this will change the ideal axis limits.

    ap_plot_sbprof_xlim : tuple, default None
      Tuple with axes limits for the x-axis in the SB profile
      diagnostic plot.

    ap_plot_sbprof_set_errscale : float, default None
      Float value by which to scale errorbars on the SB profile
      this makes them more visible in cases where the statistical
      errors are very small.

    Notes
    ----------
    :References:
    - 'background'
    - 'background noise'
    - 'psf fwhm'
    - 'center'
    - 'init R' (optional)
    - 'init ellip' (optional)
    - 'init pa' (optional)
    - 'fit R' (optional)
    - 'fit ellip' (optional)
    - 'fit pa' (optional)
    - 'fit photutils isolist' (optional)

    Returns
    -------
    IMG : ndarray
      Unaltered galaxy image

    results : dict
      .. code-block:: python

        {'prof header': , # List object with strings giving the items in the header of the final SB profile (list)
         'prof units': , # dict object that links header strings to units (given as strings) for each variable (dict)
         'prof data': # dict object linking header strings to list objects containing the rows for a given variable (dict)

        }

    """

    zeropoint = options["ap_zeropoint"] if "ap_zeropoint" in options else 22.5
    fluxunits = options["ap_fluxunits"] if "ap_fluxunits" in options else "mag"

    if fluxunits == "intensity":
        params = [
            "R",
            "I",
            "I_e",
            "totflux",
            "totflux_e",
            "ellip",
            "ellip_e",
            "pa",
            "pa_e",
            "a3",
            "a3_e",
            "b3",
            "b3_e",
            "a4",
            "a4_e",
            "b4",
            "b4_e",
        ]
        SBprof_units = {
            "R": "arcsec",
            "I": "flux*arcsec^-2",
            "I_e": "flux*arcsec^-2",
            "totflux": "flux",
            "totflux_e": "flux",
            "ellip": "unitless",
            "ellip_e": "unitless",
            "pa": "deg",
            "pa_e": "deg",
            "a3": "unitless",
            "a3_e": "unitless",
            "b3": "unitless",
            "b3_e": "unitless",
            "a4": "unitless",
            "a4_e": "unitless",
            "b4": "unitless",
            "b4_e": "unitless",
        }
    else:
        params = [
            "R",
            "SB",
            "SB_e",
            "totmag",
            "totmag_e",
            "ellip",
            "ellip_e",
            "pa",
            "pa_e",
            "a3",
            "a3_e",
            "b3",
            "b3_e",
            "a4",
            "a4_e",
            "b4",
            "b4_e",
        ]
        SBprof_units = {
            "R": "arcsec",
            "SB": "mag*arcsec^-2",
            "SB_e": "mag*arcsec^-2",
            "totmag": "mag",
            "totmag_e": "mag",
            "ellip": "unitless",
            "ellip_e": "unitless",
            "pa": "deg",
            "pa_e": "deg",
            "a3": "unitless",
            "a3_e": "unitless",
            "b3": "unitless",
            "b3_e": "unitless",
            "a4": "unitless",
            "a4_e": "unitless",
            "b4": "unitless",
            "b4_e": "unitless",
        }
    SBprof_data = dict((h, []) for h in params)
    res = {}
    dat = IMG - results["background"]
    if not "fit R" in results and not "fit photutils isolist" in results:
        logging.info(
            "%s: photutils fitting and extracting image data" % options["ap_name"]
        )
        geo = EllipseGeometry(
            x0=results["center"]["x"],
            y0=results["center"]["y"],
            sma=results["init R"] / 2,
            eps=results["init ellip"],
            pa=results["init pa"],
        )
        ellipse = Photutils_Ellipse(dat, geometry=geo)

        isolist = ellipse.fit_image(fix_center=True, linear=False)
        res.update(
            {
                "fit photutils isolist": isolist,
                "auxfile fitlimit": "fit limit semi-major axis: %.2f pix"
                % isolist.sma[-1],
            }
        )
    elif not "fit photutils isolist" in results:
        logging.info("%s: photutils extracting image data" % options["ap_name"])
        list_iso = []
        for i in range(len(results["fit R"])):
            if results["fit R"][i] <= 0:
                continue
            # Container for ellipse geometry
            geo = EllipseGeometry(
                sma=results["fit R"][i],
                x0=results["center"]["x"],
                y0=results["center"]["y"],
                eps=results["fit ellip"][i],
                pa=results["fit pa"][i],
            )
            # Extract the isophote information
            ES = EllipseSample(dat, sma=results["fit R"][i], geometry=geo)
            ES.update(fixed_parameters=True)
            list_iso.append(Isophote(ES, niter=30, valid=True, stop_code=0))

        isolist = IsophoteList(list_iso)
        res.update(
            {
                "fit photutils isolist": isolist,
                "auxfile fitlimit": "fit limit semi-major axis: %.2f pix"
                % isolist.sma[-1],
            }
        )
    else:
        isolist = results["fit photutils isolist"]

    for i in range(len(isolist.sma)):
        SBprof_data["R"].append(isolist.sma[i] * options["ap_pixscale"])
        if fluxunits == "intensity":
            SBprof_data["I"].append(
                np.median(isolist.sample[i].values[2]) / options["ap_pixscale"] ** 2
            )
            SBprof_data["I_e"].append(isolist.int_err[i])
            SBprof_data["totflux"].append(isolist.tflux_e[i])
            SBprof_data["totflux_e"].append(isolist.rms[i] / np.sqrt(isolist.npix_e[i]))
        else:
            SBprof_data["SB"].append(
                flux_to_sb(
                    np.median(isolist.sample[i].values[2]),
                    options["ap_pixscale"],
                    zeropoint,
                )
            )
            SBprof_data["SB_e"].append(
                2.5 * isolist.int_err[i] / (isolist.intens[i] * np.log(10))
            )
            SBprof_data["totmag"].append(flux_to_mag(isolist.tflux_e[i], zeropoint))
            SBprof_data["totmag_e"].append(
                2.5
                * isolist.rms[i]
                / (np.sqrt(isolist.npix_e[i]) * isolist.tflux_e[i] * np.log(10))
            )
        SBprof_data["ellip"].append(isolist.eps[i])
        SBprof_data["ellip_e"].append(isolist.ellip_err[i])
        SBprof_data["pa"].append(isolist.pa[i] * 180 / np.pi)
        SBprof_data["pa_e"].append(isolist.pa_err[i] * 180 / np.pi)
        SBprof_data["a3"].append(isolist.a3[i])
        SBprof_data["a3_e"].append(isolist.a3_err[i])
        SBprof_data["b3"].append(isolist.b3[i])
        SBprof_data["b3_e"].append(isolist.b3_err[i])
        SBprof_data["a4"].append(isolist.a4[i])
        SBprof_data["a4_e"].append(isolist.a4_err[i])
        SBprof_data["b4"].append(isolist.b4[i])
        SBprof_data["b4_e"].append(isolist.b4_err[i])
        for k in SBprof_data.keys():
            if not np.isfinite(SBprof_data[k][-1]):
                SBprof_data[k][-1] = 99.999
    res.update(
        {"prof header": params, "prof units": SBprof_units, "prof data": SBprof_data}
    )

    if "ap_doplot" in options and options["ap_doplot"]:
        if fluxunits == "intensity":
            Plot_I_Profile(
                dat,
                np.array(SBprof_data["R"]),
                np.array(SBprof_data["I"]),
                np.array(SBprof_data["I_e"]),
                np.array(SBprof_data["ellip"]),
                np.array(SBprof_data["pa"]),
                results,
                options,
            )
        else:
            Plot_SB_Profile(
                dat,
                np.array(SBprof_data["R"]),
                np.array(SBprof_data["SB"]),
                np.array(SBprof_data["SB_e"]),
                np.array(SBprof_data["ellip"]),
                np.array(SBprof_data["pa"]),
                results,
                options,
            )

    return IMG, res
